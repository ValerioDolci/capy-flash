#include "models/qwen3_5/execution/ple_gather.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

// Columns per host worker before a mapped gather fans out; page faults dominate a cold table.
constexpr std::size_t kColumnsPerWorker = 64;
constexpr std::uint32_t kMaximumNgram   = 8;

float e4m3fn(std::uint8_t code) {
    const int sign     = code >> 7;
    const int exponent = (code >> 3) & 0xF;
    const int mantissa = code & 0x7;
    float magnitude    = 0.0F;
    if (exponent == 0xF && mantissa == 0x7) {
        magnitude = std::nanf("");
    } else if (exponent == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa) / 8.0F, -6);
    } else {
        magnitude = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
    }
    return sign != 0 ? -magnitude : magnitude;
}

// E2M1 (FP4) values by code: sign bit 3, exponent bits 2-1, mantissa bit 0.
constexpr std::array<float, 16> kE2m1{0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
                                      -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
constexpr std::uint32_t kNvfp4Group = 16;

float bf16_to_float(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16);
}

std::uint16_t float_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if (std::isnan(value)) { return static_cast<std::uint16_t>((bits >> 16) | 0x40U); }
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

} // namespace

PleGather::PleGather(PleTableView table, PleHashConfig hash, bool prefetch)
    : table_(std::move(table)), hash_(std::move(hash)), prefetch_(prefetch) {
    const std::uint32_t heads = hash_.heads();
    if (hash_.ngram_size < 2 || hash_.ngram_size > kMaximumNgram || hash_.heads_per_ngram == 0 ||
        hash_.multipliers.size() != hash_.ngram_size || hash_.head_moduli.size() != heads ||
        hash_.head_offsets.size() != heads || heads > 32) {
        throw std::invalid_argument("PLE gather hash configuration is inconsistent");
    }
    const bool fp8 = table_.format == PleRowFormat::Fp8RowBf16;
    if (table_.row_width == 0 || table_.rows == 0 || table_.segments.empty() ||
        table_.row_bytes < table_.row_record_bytes() ||
        (fp8 ? table_.scale_plane < (table_.rows - 1) * table_.row_bytes + table_.row_width
             : table_.row_width % kNvfp4Group != 0 || !std::isfinite(table_.global_scale))) {
        throw std::invalid_argument("PLE table description is inconsistent");
    }
    for (std::size_t i = 1; i < table_.segments.size(); ++i) {
        if (table_.segments[i].object_offset <= table_.segments[i - 1].object_offset) {
            throw std::invalid_argument("PLE table segments must ascend");
        }
    }
    for (std::uint32_t head = 0; head < heads; ++head) {
        if (hash_.head_moduli[head] == 0 ||
            hash_.head_offsets[head] + hash_.head_moduli[head] > table_.rows) {
            throw std::invalid_argument("PLE hash head exceeds the table");
        }
    }
    width_ = heads * table_.row_width;
    for (int code = 0; code < 256; ++code) {
        e4m3_[static_cast<std::size_t>(code)] = e4m3fn(static_cast<std::uint8_t>(code));
    }
    if (!fp8) {
        // One table holds every value an NVFP4 row can decode to: the global scale is fixed.
        nvfp4_values_.resize(256 * kE2m1.size());
        for (std::size_t scale = 0; scale < 256; ++scale) {
            for (std::size_t code = 0; code < kE2m1.size(); ++code) {
                nvfp4_values_[scale * kE2m1.size() + code] =
                    float_to_bf16((kE2m1[code] * e4m3_[scale]) * table_.global_scale);
            }
        }
    }
    for (const auto& segment : table_.segments) {
        if (segment.data == nullptr) {
            throw std::invalid_argument("mapped PLE residency needs mapped segments");
        }
    }
    if (fp8) { load_scales(); }
}

void PleGather::load_scales() {
    static const std::uintptr_t page = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
    constexpr std::uint64_t kPiece   = 64ULL << 20;
    const std::uint64_t bytes        = table_.rows * 2U;
    // A description whose segments stop before the plane (a hash-only table) keeps the mapping
    // path, which reports the missing bytes if a gather ever reads them.
    const PleTableSegment& last = table_.segments.back();
    if (last.object_offset + last.bytes < table_.scale_plane + bytes) { return; }
    // Read ahead one piece while copying the previous one: sequential reads, unlike the faults of
    // a random-access mapping (MADV_RANDOM), which would read the plane one page at a time.
    const auto advise = [&](std::uint64_t done) {
        if (done >= bytes) { return; }
        for_each_piece(
            table_.scale_plane + done, std::min(kPiece, bytes - done),
            [&](const PleTableSegment& segment, std::uint64_t within, std::uint64_t take) {
                const auto address = reinterpret_cast<std::uintptr_t>(segment.data + within);
                const auto begin   = address & ~(page - 1);
                (void)::madvise(reinterpret_cast<void*>(begin), address + take - begin,
                                MADV_WILLNEED);
            });
    };
    scales_.resize(bytes);
    advise(0);
    for (std::uint64_t done = 0; done < bytes; done += kPiece) {
        advise(done + kPiece);
        std::uint64_t at = done;
        for_each_piece(
            table_.scale_plane + done, std::min(kPiece, bytes - done),
            [&](const PleTableSegment& segment, std::uint64_t within, std::uint64_t take) {
                std::memcpy(scales_.data() + at, segment.data + within, take);
                at += take;
            });
    }
    // A plane that repeats one multiplier on every row keeps only that multiplier: every row reads
    // the same two bytes it would read from the plane.
    if (bytes >= 2 && std::memcmp(scales_.data(), scales_.data() + 2, bytes - 2) == 0) {
        std::vector<std::byte>(scales_.begin(), scales_.begin() + 2).swap(scales_);
        scale_stride_ = 0;
    }
}

PleGather::~PleGather() = default;

template <class Piece>
void PleGather::for_each_piece(std::uint64_t object_offset, std::uint64_t bytes,
                               Piece&& piece) const {
    auto it = std::upper_bound(table_.segments.begin(), table_.segments.end(), object_offset,
                               [](std::uint64_t offset, const PleTableSegment& segment) {
                                   return offset < segment.object_offset;
                               });
    if (it == table_.segments.begin()) { throw std::out_of_range("PLE table offset is unmapped"); }
    --it;
    std::uint64_t within = object_offset - it->object_offset;
    while (bytes != 0) {
        if (it == table_.segments.end() || within >= it->bytes) {
            throw std::out_of_range("PLE table range runs past the mapped segments");
        }
        const std::uint64_t take = std::min(bytes, it->bytes - within);
        piece(*it, within, take);
        bytes -= take;
        if (bytes != 0) {
            const auto next = std::next(it);
            // Segments cover the object contiguously; a gap would be a malformed table description.
            if (next == table_.segments.end() ||
                next->object_offset != it->object_offset + it->bytes) {
                throw std::out_of_range("PLE table range crosses a gap between segments");
            }
            it     = next;
            within = 0;
        }
    }
}

const std::byte* PleGather::locate(std::uint64_t object_offset, std::uint64_t bytes,
                                   std::byte* bounce) const {
    const std::byte* single = nullptr;
    std::uint64_t copied    = 0;
    std::uint32_t pieces    = 0;
    for_each_piece(object_offset, bytes,
                   [&](const PleTableSegment& segment, std::uint64_t within, std::uint64_t take) {
                       if (pieces++ == 0) { single = segment.data + within; }
                       if (take != bytes) {
                           std::memcpy(bounce + copied, segment.data + within, take);
                       }
                       copied += take;
                   });
    return pieces == 1 ? single : bounce;
}

void PleGather::rows(std::span<const std::int64_t> context, std::span<std::uint64_t> out) const {
    if (context.size() != hash_.ngram_size || out.size() != hash_.heads()) {
        throw std::invalid_argument("PLE row query has the wrong extent");
    }
    std::uint32_t head = 0;
    for (std::uint32_t n = 2; n <= hash_.ngram_size; ++n) {
        // Token ids and multipliers keep every product below 2^63, so the XOR is non-negative.
        std::uint64_t mixed = 0;
        for (std::uint32_t k = 0; k < n; ++k) {
            mixed ^= static_cast<std::uint64_t>(context[k]) * hash_.multipliers[k];
        }
        for (std::uint32_t j = 0; j < hash_.heads_per_ngram; ++j, ++head) {
            out[head] = mixed % hash_.head_moduli[head] + hash_.head_offsets[head];
        }
    }
}

void PleGather::column_rows(std::span<const std::int32_t> tokens, std::size_t position,
                            std::span<std::uint64_t> rows_out) const {
    const auto eos = static_cast<std::int64_t>(hash_.eos_token_id);
    std::array<std::int64_t, kMaximumNgram> context{};
    context[0]  = tokens[position];
    bool closed = false;
    for (std::uint32_t s = 1; s < hash_.ngram_size; ++s) {
        if (!closed && s <= position) {
            const std::int64_t token = tokens[position - s];
            if (token == eos) { closed = true; }
            context[s] = closed ? eos : token;
        } else {
            context[s] = eos;
        }
    }
    rows(std::span<const std::int64_t>(context.data(), hash_.ngram_size), rows_out);
}

void PleGather::decode_fp8_row(const std::byte* codes, const std::byte* scale_bytes,
                               std::uint16_t* out) const {
    std::uint16_t scale_word;
    std::memcpy(&scale_word, scale_bytes, 2);
    const float scale = bf16_to_float(scale_word);
    for (std::uint32_t i = 0; i < table_.row_width; ++i) {
        out[i] = float_to_bf16(e4m3_[static_cast<std::uint8_t>(codes[i])] * scale);
    }
}

void PleGather::decode_nvfp4_row(const std::byte* record, std::uint16_t* out) const {
    const std::byte* scales = record + table_.row_width / 2;
    for (std::uint32_t group = 0; group < table_.row_width / kNvfp4Group; ++group) {
        const std::uint16_t* decoded =
            nvfp4_values_.data() + static_cast<std::size_t>(scales[group]) * kE2m1.size();
        const std::byte* codes = record + group * (kNvfp4Group / 2);
        std::uint16_t* values  = out + group * kNvfp4Group;
        for (std::uint32_t i = 0; i < kNvfp4Group / 2; ++i) {
            const auto pair   = static_cast<std::uint8_t>(codes[i]);
            values[2 * i]     = decoded[pair & 0xFU];
            values[2 * i + 1] = decoded[pair >> 4];
        }
    }
}

// Starts asynchronous reads of every page the rows touch that is not resident, so the faults of
// the following decode overlap instead of paying one storage latency per row. Each distinct page
// is advised once (rows repeat within a chunk and share pages), runs of adjacent pages together.
void PleGather::prefetch_mapped(std::span<const std::uint64_t> rows) const {
    static const std::uintptr_t page = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
    std::vector<std::uintptr_t> pages;
    pages.reserve(rows.size() * 2);
    const auto add_range = [&](std::uint64_t object_offset, std::uint64_t bytes) {
        for_each_piece(
            object_offset, bytes,
            [&](const PleTableSegment& segment, std::uint64_t within, std::uint64_t take) {
                const auto address = reinterpret_cast<std::uintptr_t>(segment.data + within);
                for (std::uintptr_t p = address & ~(page - 1); p < address + take; p += page) {
                    pages.push_back(p);
                }
            });
    };
    for (const std::uint64_t row : rows) {
        add_range(row * table_.row_bytes, table_.row_record_bytes());
        if (table_.format == PleRowFormat::Fp8RowBf16 && scales_.empty()) {
            add_range(table_.scale_plane + row * 2U, 2);
        }
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    for (std::size_t i = 0; i < pages.size();) {
        std::size_t j = i + 1;
        while (j < pages.size() && pages[j] == pages[j - 1] + page) { ++j; }
        // Advice is a hint: a failure leaves the synchronous fault path, never a wrong value.
        (void)::madvise(reinterpret_cast<void*>(pages[i]), (j - i) * page, MADV_WILLNEED);
        i = j;
    }
}

void PleGather::prefetch(std::span<const std::int32_t> tokens, std::size_t first,
                         std::size_t count) {
    if (!prefetch_ || first > tokens.size() || count > tokens.size() - first) { return; }
    const std::uint32_t heads = hash_.heads();
    std::vector<std::uint64_t> rows(count * heads);
    for (std::size_t i = 0; i < count; ++i) {
        column_rows(tokens, first + i, std::span<std::uint64_t>(rows.data() + i * heads, heads));
    }
    prefetch_mapped(rows);
}

void PleGather::gather(std::span<const std::int32_t> tokens, std::size_t first, std::size_t count,
                       std::span<std::uint16_t> out) {
    if (first > tokens.size() || count > tokens.size() - first ||
        out.size() < static_cast<std::size_t>(width_) * count) {
        throw std::invalid_argument("PLE gather range is invalid");
    }
    const std::lock_guard gather_lock(gather_mutex_);
    const std::uint32_t heads = hash_.heads();
    row_scratch_.resize(count * heads);
    std::vector<std::uint64_t>& rows = row_scratch_;
    for (std::size_t i = 0; i < count; ++i) {
        column_rows(tokens, first + i, std::span<std::uint64_t>(rows.data() + i * heads, heads));
    }
    const std::size_t per_row = ranges_per_row();
    const auto record         = static_cast<std::uint32_t>(table_.row_record_bytes());
    if (prefetch_) { prefetch_mapped(rows); }
    const auto decode = [&](std::size_t begin, std::size_t end) {
        std::vector<std::byte> bounce(record); // a straddling row's record
        std::array<std::byte, 2> scale_bounce{};
        for (std::size_t i = begin; i < end; ++i) {
            for (std::uint32_t head = 0; head < heads; ++head) {
                const std::size_t r = i * heads + head;
                std::uint16_t* values =
                    out.data() + i * width_ + static_cast<std::size_t>(head) * table_.row_width;
                const std::byte* codes = locate(rows[r] * table_.row_bytes, record, bounce.data());
                if (per_row == 1) {
                    decode_nvfp4_row(codes, values);
                    continue;
                }
                const std::byte* scale =
                    !scales_.empty()
                        ? scales_.data() + rows[r] * scale_stride_
                        : locate(table_.scale_plane + rows[r] * 2U, 2, scale_bounce.data());
                decode_fp8_row(codes, scale, values);
            }
        }
    };
    // Mapped reads fault pages in on first touch, so large calls fan out across threads.
    const std::size_t hardware = std::max(1U, std::thread::hardware_concurrency());
    const std::size_t workers =
        std::min<std::size_t>(std::min<std::size_t>(hardware, 16), count / kColumnsPerWorker);
    if (workers <= 1) {
        decode(0, count);
        return;
    }
    std::vector<std::jthread> threads;
    threads.reserve(workers - 1);
    const std::size_t share = (count + workers - 1) / workers;
    for (std::size_t w = 1; w < workers; ++w) {
        const std::size_t begin = std::min(count, w * share);
        const std::size_t end   = std::min(count, begin + share);
        threads.emplace_back([&, begin, end] { decode(begin, end); });
    }
    decode(0, std::min(count, share));
}

} // namespace ninfer::models::qwen3_5::execution
