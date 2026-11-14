#pragma once

// Host gather of the Qwen4Exp n-gram (PLE) embedding. Ported from ninfer-ext
// src/models/qwen3_5/execution/ple_gather.{h,cpp} (Apache-2.0) with self-contained table and hash
// descriptions, so the loader can describe the table without this file depending on it.
//
// The table never resides in RAM as a whole (FP8 320,001,536 x 160 = 47.7 GiB): every call reads
// the hash rows of its columns either through a read-only file mapping (Mapped: rows come from
// the page cache; misses are first prefetched together with MADV_WILLNEED so a window pays about
// one storage latency instead of one per row) or with batched direct reads (Stream: O_DIRECT,
// nothing enters the page cache). The decoded BF16 embedding `[embed_dim, count]` is written to a
// caller buffer (pinned for one H2D upload per call).
//
// For the FP8 table, Mapped keeps the BF16 row multipliers (the scale plane, 2 bytes per row =
// 640 MB) in RAM, read once sequentially at construction (T171): otherwise every cold row costs
// two random page reads, its codes and its multiplier, and the storage's random-read rate bounds
// the first prefill chunk. NINFER_QWEN4_PLE_RESIDENT_SCALES=0 reads the multipliers through the
// mapping instead. An NVFP4 row carries its own scales (one 90-byte read per row), so nothing of
// that table is resident.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// One contiguous span of the table object. `data` addresses it in a read-only mapping (Mapped);
// `file` and `file_offset` locate it for direct reads (Stream).
struct PleTableSegment {
    std::uint64_t object_offset = 0;
    std::uint64_t bytes         = 0;
    const std::byte* data       = nullptr;
    std::filesystem::path file;
    std::uint64_t file_offset = 0;
};

enum class PleRowFormat : std::uint8_t {
    // fp8_e4m3fn_row_bf16, row_scale_v1: FP8 E4M3FN codes `[rows, row_width]` (row stride
    // `row_bytes`), then one BF16 multiplier per row at object offset `scale_plane`. A value is
    // bf16_rn(e4m3 * multiplier).
    Fp8RowBf16,
    // nvfp4_row_e4m3_g16, row_interleaved_v1: row r at object offset r * row_bytes holds
    // row_width / 2 bytes of E2M1 codes (low nibble = even element), then row_width / 16 E4M3FN
    // scales, one per 16 elements. A value is bf16_rn((e2m1 * e4m3) * global_scale) in FP32.
    Nvfp4RowE4m3G16,
};

struct PleTableView {
    std::vector<PleTableSegment> segments; // ascending object_offset, covering the object
    PleRowFormat format       = PleRowFormat::Fp8RowBf16;
    std::uint64_t rows        = 0;
    std::uint32_t row_width   = 0;    // values per row
    std::uint64_t row_bytes   = 0;    // row stride
    std::uint64_t scale_plane = 0;    // Fp8RowBf16 only
    float global_scale        = 0.0F; // Nvfp4RowE4m3G16 only

    // Bytes of row r read at object offset r * row_bytes: the codes (FP8) or the whole
    // codes-then-scales record (NVFP4).
    [[nodiscard]] std::uint64_t row_record_bytes() const noexcept {
        return format == PleRowFormat::Fp8RowBf16 ? row_width : row_width / 2 + row_width / 16;
    }
};

// N-gram hashing of one PLE layer (docs: Qwen4Exp model reference, "N-gram rows").
struct PleHashConfig {
    std::uint32_t ngram_size      = 0;
    std::uint32_t heads_per_ngram = 0;
    std::uint32_t eos_token_id    = 0;
    std::vector<std::uint64_t> multipliers;  // ngram_size multipliers
    std::vector<std::uint64_t> head_moduli;  // prime modulus per head
    std::vector<std::uint64_t> head_offsets; // table row offset per head

    [[nodiscard]] std::uint32_t heads() const noexcept {
        return (ngram_size - 1) * heads_per_ngram;
    }
};

enum class PleResidency : std::uint8_t { Mapped, Stream };

class PleRowReader;

class PleGather {
public:
    // `prefetch` (Mapped only) issues MADV_WILLNEED for every row page before decoding.
    PleGather(PleTableView table, PleHashConfig hash, PleResidency residency, bool prefetch = true);
    ~PleGather();

    PleGather(const PleGather&)            = delete;
    PleGather& operator=(const PleGather&) = delete;

    [[nodiscard]] PleResidency residency() const noexcept { return residency_; }

    // Embedding width: heads * row_width.
    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }

    // BF16 `[width, count]` embedding columns of `tokens[first, first + count)`. Tokens before
    // `first` (and EOS before the sequence start) form each n-gram context, and an EOS closes
    // every window it falls into. `out` holds `width * count` BF16 words.
    void gather(std::span<const std::int32_t> tokens, std::size_t first, std::size_t count,
                std::span<std::uint16_t> out);

    // Starts the page reads (MADV_WILLNEED) of the rows a later gather() of the same columns
    // will decode, and returns (Mapped with prefetch only; otherwise nothing). A hint: the
    // gather's values never depend on it.
    void prefetch(std::span<const std::int32_t> tokens, std::size_t first, std::size_t count);

    // Table rows of one position whose context tokens at distance 0..ngram_size-1 are `context`.
    void rows(std::span<const std::int64_t> context, std::span<std::uint64_t> out) const;

    // Table rows of column `position` of `tokens` (EOS-closed context).
    void column_rows(std::span<const std::int32_t> tokens, std::size_t position,
                     std::span<std::uint64_t> rows) const;

private:
    void decode_fp8_row(const std::byte* codes, const std::byte* scale, std::uint16_t* out) const;
    void decode_nvfp4_row(const std::byte* record, std::uint16_t* out) const;

    // Byte ranges read per row: codes and BF16 multiplier (FP8), or the one record (NVFP4).
    [[nodiscard]] std::size_t ranges_per_row() const noexcept {
        return table_.format == PleRowFormat::Fp8RowBf16 ? 2 : 1;
    }
    // Calls `piece(segment, within, bytes)` for each segment `[object_offset, +bytes)` touches, in
    // order. A row can straddle two segments: an artifact's part files end at payload boundaries
    // that need not fall on a row boundary (T155: "PLE table row crosses a mapped segment").
    template <class Piece>
    void for_each_piece(std::uint64_t object_offset, std::uint64_t bytes, Piece&& piece) const;
    // Mapped bytes `[object_offset, +bytes)`: a pointer into the mapping, or `bounce` filled from
    // the segments when the range straddles a segment boundary (`bounce` holds at least `bytes`).
    [[nodiscard]] const std::byte* locate(std::uint64_t object_offset, std::uint64_t bytes,
                                          std::byte* bounce) const;
    void prefetch_mapped(std::span<const std::uint64_t> rows) const;
    // Copies the scale plane `[scale_plane, + rows * 2)` from the mapping into `scales_`.
    void load_scales();

    PleTableView table_;
    PleHashConfig hash_;
    PleResidency residency_;
    bool prefetch_       = true;
    std::uint32_t width_ = 0;
    std::array<float, 256> e4m3_{};
    // NVFP4: BF16 value of (E4M3 scale code, E2M1 code) at [scale * 16 + code], global scale
    // applied: bf16_rn((e2m1 * e4m3) * global_scale).
    std::vector<std::uint16_t> nvfp4_values_;
    std::unique_ptr<PleRowReader> reader_;
    std::vector<std::uint64_t> row_scratch_;
    std::vector<std::byte>
        scales_; // resident scale plane (Mapped); empty: read through the mapping
    // Every row of the plane holds these two bytes (the checkpoint's single multiplier): no plane.
    std::array<std::byte, 2> uniform_scale_{};
    bool scale_is_uniform_ = false;
};

} // namespace ninfer::models::qwen3_5::execution
