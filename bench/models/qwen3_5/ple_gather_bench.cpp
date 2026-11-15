// Host latency of the Qwen4Exp n-gram (PLE) gather for decode windows: hash, row reads and
// BF16 decode of `T` columns (16 rows of 160 values each), from a table file on disk.
//
// The table is a synthetic file in the artifact layout of `--format`: fp8 (FP8 codes [rows,160],
// BF16 row scales at a 256-byte aligned plane) or nvfp4 (90-byte rows: 80 bytes of E2M1 codes,
// then 10 E4M3FN group scales; the FP4-Local global scale). Its size only sets how many distinct
// pages random rows touch, not the per-row cost. `--cache cold` drops the file from the page cache
// (POSIX_FADV_DONTNEED) before every timed call; `--cache warm` repeats a call whose rows are
// resident.
#include "models/qwen3_5/execution/ple_gather.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::models::qwen3_5::execution;

namespace {

struct Options {
    std::string file;
    double create_gib = 0.0;
    std::string mode = "mapped", cache = "cold", format = "fp8";
    std::vector<int> tokens{1, 4, 8};
    int repeat     = 30;
    double lead_us = 0.0; // prefetch() this long before the timed gather (0 = no prefetch call)
};

std::vector<int> list(const std::string& text) {
    std::vector<int> out;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(',', start);
        out.push_back(std::stoi(text.substr(start, end - start)));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::puts("usage: ninfer_ple_gather_bench --file PATH [--create-gib G] "
                      "[--format fp8|nvfp4] [--mode mapped|mapped-noprefetch] [--cache cold|warm] "
                      "[--tokens T,...] [--repeat N]");
            std::exit(0);
        }
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (arg == "--file")
            o.file = value;
        else if (arg == "--create-gib")
            o.create_gib = std::stod(value);
        else if (arg == "--format")
            o.format = value;
        else if (arg == "--mode")
            o.mode = value;
        else if (arg == "--cache")
            o.cache = value;
        else if (arg == "--tokens")
            o.tokens = list(value);
        else if (arg == "--repeat")
            o.repeat = std::stoi(value);
        else if (arg == "--lead-us")
            o.lead_us = std::stod(value);
        else
            throw std::invalid_argument("unknown option: " + arg);
    }
    if (o.file.empty()) throw std::invalid_argument("--file is required");
    if (o.format != "fp8" && o.format != "nvfp4") throw std::invalid_argument("unknown --format");
    if (o.mode != "mapped" && o.mode != "mapped-noprefetch")
        throw std::invalid_argument("unknown --mode");
    return o;
}

constexpr std::uint64_t kRowWidth    = 160;
constexpr std::uint64_t kNvfp4Record = kRowWidth / 2 + kRowWidth / 16;

// Rows of an FP8 table (codes and a 2-byte multiplier per row) or an NVFP4 one (90-byte records).
std::uint64_t rows_for(std::uint64_t bytes, bool nvfp4) {
    return (bytes - 4096) / (nvfp4 ? kNvfp4Record : kRowWidth + 2);
}

std::uint64_t scale_plane(std::uint64_t rows) { return (rows * kRowWidth + 255) / 256 * 256; }

void create(const std::string& path, double gib, bool nvfp4) {
    const std::uint64_t bytes = static_cast<std::uint64_t>(gib * (1ULL << 30));
    const std::uint64_t rows  = rows_for(bytes, nvfp4);
    // NVFP4: no plane; the masked words keep its E4M3 scale bytes finite as well.
    const std::uint64_t plane = nvfp4 ? rows * kNvfp4Record : scale_plane(rows);
    FILE* f                   = std::fopen(path.c_str(), "wb");
    if (f == nullptr) throw std::runtime_error("cannot create " + path);
    std::vector<std::uint8_t> chunk(1 << 22);
    std::uint64_t written = 0, state = 0x9E3779B97F4A7C15ULL;
    const std::uint64_t total = nvfp4 ? plane : plane + rows * 2;
    while (written < total) {
        const std::size_t n =
            static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(), total - written));
        for (std::size_t i = 0; i < n; i += 8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            std::uint64_t word = state;
            if (written + i >= plane) {
                word = 0x3C003C003C003C00ULL; // BF16 1.0 scales
            } else {
                word &= 0xF7F7F7F7F7F7F7F7ULL; // no E4M3 NaN codes (0x7F/0xFF)
            }
            std::memcpy(chunk.data() + i, &word, std::min<std::size_t>(8, n - i));
        }
        std::fwrite(chunk.data(), 1, n, f);
        written += n;
    }
    std::fclose(f);
}

PleHashConfig official_hash(std::uint64_t rows) {
    // The official hash geometry, with moduli scaled into the synthetic table.
    PleHashConfig hash;
    hash.ngram_size              = 3;
    hash.heads_per_ngram         = 8;
    hash.eos_token_id            = 248044;
    hash.multipliers             = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    const std::uint64_t per_head = rows / 16;
    std::uint64_t offset         = 0;
    for (int h = 0; h < 16; ++h) {
        hash.head_moduli.push_back(per_head - 1 - static_cast<std::uint64_t>(h) * 2);
        hash.head_offsets.push_back(offset);
        offset += per_head;
    }
    return hash;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o  = parse(argc, argv);
        const bool nvfp4 = o.format == "nvfp4";
        if (o.create_gib > 0.0 && !std::filesystem::exists(o.file)) {
            create(o.file, o.create_gib, nvfp4);
        }
        const std::uint64_t bytes = std::filesystem::file_size(o.file);
        const int fd              = ::open(o.file.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("cannot open " + o.file);
        void* map = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) throw std::runtime_error("mmap failed");
        ::madvise(map, bytes, MADV_RANDOM); // as the loader maps the table
        PleTableView table;
        // Row values do not affect the timing; the layout only needs to stay inside the file.
        table.rows      = rows_for(bytes, nvfp4);
        table.row_width = kRowWidth;
        if (nvfp4) {
            table.format       = PleRowFormat::Nvfp4RowE4m3G16;
            table.row_bytes    = kNvfp4Record;
            table.global_scale = 3.3242362e-05F;
        } else {
            table.row_bytes   = kRowWidth;
            table.scale_plane = scale_plane(table.rows);
        }
        table.segments = {{0, bytes, static_cast<const std::byte*>(map)}};
        PleGather gather(table, official_hash(table.rows), o.mode == "mapped");
        std::printf("table=%s %.2f GiB format=%s rows=%llu mode=%s cache=%s\n", o.file.c_str(),
                    bytes / double(1ULL << 30), o.format.c_str(),
                    static_cast<unsigned long long>(table.rows), o.mode.c_str(), o.cache.c_str());
        std::mt19937 rng(42);
        for (const int t : o.tokens) {
            std::vector<double> samples;
            std::vector<std::uint16_t> out(static_cast<std::size_t>(gather.width()) * t);
            std::vector<std::int32_t> tokens(static_cast<std::size_t>(t) + 2);
            for (int r = 0; r < o.repeat + 3; ++r) {
                if (o.cache == "cold" || r == 0) {
                    for (auto& token : tokens) token = static_cast<std::int32_t>(rng() % 248000);
                }
                if (o.cache == "cold") { ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); }
                if (o.lead_us > 0.0) {
                    // The engine's next-round prefetch, `lead_us` of other host work earlier.
                    const auto issued = std::chrono::steady_clock::now();
                    gather.prefetch(tokens, 2, static_cast<std::size_t>(t));
                    while (std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - issued)
                               .count() < o.lead_us) {}
                }
                const auto start = std::chrono::steady_clock::now();
                gather.gather(tokens, 2, static_cast<std::size_t>(t), out);
                const auto stop = std::chrono::steady_clock::now();
                if (r >= 3)
                    samples.push_back(
                        std::chrono::duration<double, std::micro>(stop - start).count());
            }
            std::sort(samples.begin(), samples.end());
            std::printf("gather T=%d lead=%.0f rows=%d median=%.1f min=%.1f p90=%.1f us\n", t,
                        o.lead_us, 16 * t, samples[samples.size() / 2], samples.front(),
                        samples[samples.size() * 9 / 10]);
        }
        ::munmap(map, bytes);
        ::close(fd);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ple_gather_bench: %s\n", error.what());
        return 1;
    }
}
