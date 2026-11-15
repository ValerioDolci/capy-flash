// Qwen4Exp n-gram (PLE) host gather.
//  1. Hash: the 16 table rows of every column equal the independent Python reference
//     (ninfer-ext tests/models/qwen4_exp/reference.py ngram_row_ids) at the published constants,
//     including sequence start, EOS-closed windows and repeated EOS (exact oracle).
//  2. Residency: mapped gathers with and without WILLNEED prefetch give the same BF16 embedding
//     over a table split across two volume files at unaligned offsets, so rows straddle 4 KiB
//     pages; mapped decode equals E4M3 x BF16 multiplier rounded once to BF16. A plane with one
//     multiplier on every row is kept as that multiplier (2 bytes) and passes the same checks.
//  3. NVFP4 rows (nvfp4_row_e4m3_g16, row_interleaved_v1): every value of every gathered column,
//     mapped with and without prefetch, equals bf16_rn((e2m1 * e4m3) * G) decoded
//     from the definition, on a synthetic table (all code pairs, all finite scales, a row
//     straddling two volumes) and, with NINFER_TEST_FP4_LOCAL_CHECKPOINT set to the FP4-Local
//     checkpoint, on >= 1000 real rows read from its n-gram fragments.
// Part 2 ports ninfer-ext tests/models/qwen3_5/test_ple_gather.cpp (Apache-2.0).
#include "models/qwen3_5/execution/ple_gather.h"

#include <nlohmann/json.hpp>
#include <sys/mman.h>

#include <cmath>
#include <cstdlib>
#include <set>
#include <string>
#include <cstdio>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace ninfer::models::qwen3_5::execution;

namespace {

#include "ple_hash_expected.inc"

constexpr std::uint32_t kOfficialEos = 248044;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

PleHashConfig official_hash() {
    PleHashConfig hash;
    hash.ngram_size      = 3;
    hash.heads_per_ngram = 8;
    hash.eos_token_id    = kOfficialEos;
    hash.multipliers     = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    hash.head_moduli     = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
                            20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
                            20000153, 20000159, 20000161, 20000171};
    std::uint64_t offset = 0;
    for (const auto modulus : hash.head_moduli) {
        hash.head_offsets.push_back(offset);
        offset += modulus;
    }
    return hash;
}

int hash_case() {
    // A one-row placeholder mapping: the hash never reads the table.
    static const std::byte placeholder[16]{};
    PleTableView table;
    table.rows        = 320001536ULL;
    table.row_width   = 160;
    table.row_bytes   = 160;
    table.scale_plane = table.rows * 160;
    table.segments    = {{0, sizeof(placeholder), placeholder}};
    PleGather gather(table, official_hash());
    const std::vector<std::int32_t> tokens(std::begin(kHashTokens), std::end(kHashTokens));
    int failures = 0;
    std::vector<std::uint64_t> rows(16);
    for (std::size_t p = 0; p < tokens.size(); ++p) {
        gather.column_rows(tokens, p, rows);
        for (std::size_t h = 0; h < 16; ++h) {
            if (rows[h] != kHashRows[p][h]) {
                std::cerr << "n-gram row mismatch at column " << p << " head " << h << ": "
                          << rows[h] << " != " << kHashRows[p][h] << '\n';
                ++failures;
                break;
            }
        }
    }
    return failures;
}

constexpr std::uint32_t kRows     = 5000;
constexpr std::uint32_t kRowWidth = 16;
constexpr std::uint32_t kEos      = 7;

void write_file(const std::filesystem::path& path, std::uint64_t leading, const std::byte* data,
                std::size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::vector<char> pad(leading, 'x');
    out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

struct Mapping {
    void* base        = MAP_FAILED;
    std::size_t bytes = 0;

    Mapping(const std::filesystem::path& path) {
        bytes   = std::filesystem::file_size(path);
        FILE* f = std::fopen(path.c_str(), "rb");
        base    = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fileno(f), 0);
        std::fclose(f);
        if (base == MAP_FAILED) throw std::runtime_error("mmap failed");
    }

    ~Mapping() {
        if (base != MAP_FAILED) ::munmap(base, bytes);
    }

    const std::byte* at(std::uint64_t offset) const {
        return static_cast<const std::byte*>(base) + offset;
    }
};

int residency_case(std::uint64_t split_bytes_extra = 0, bool uniform_scale = false) {
    // Two 2-gram heads and two 3-gram heads of 16 columns: a 64-wide embedding.
    PleHashConfig hash;
    hash.ngram_size      = 3;
    hash.heads_per_ngram = 2;
    hash.eos_token_id    = kEos;
    hash.multipliers     = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    hash.head_moduli     = {1201, 1213, 1217, 1223};
    std::uint64_t offset = 0;
    for (const auto modulus : hash.head_moduli) {
        hash.head_offsets.push_back(offset);
        offset += modulus;
    }

    // Object: FP8 codes [rows, 16], then BF16 row multipliers at a 256-byte aligned plane.
    const std::uint64_t scale_plane = (kRows * kRowWidth + 255) / 256 * 256;
    std::vector<std::byte> object(scale_plane + kRows * 2);
    std::mt19937 rng(1234);
    for (std::uint64_t i = 0; i < kRows * kRowWidth; ++i) {
        std::uint8_t code = static_cast<std::uint8_t>(rng());
        if ((code & 0x7F) == 0x7F) { code ^= 1; } // finite E4M3FN codes only
        object[i] = std::byte{code};
    }
    for (std::uint32_t r = 0; r < kRows; ++r) {
        const auto random = static_cast<std::uint16_t>(rng() % 0x0300);
        const std::uint16_t scale =
            static_cast<std::uint16_t>(0x3C00 + (uniform_scale ? 0x40 : random));
        std::memcpy(object.data() + scale_plane + r * 2, &scale, 2);
    }
    // split_bytes_extra != 0 puts the volume boundary inside row 1777 (as a real artifact whose
    // part files end at 4 KiB-aligned payloads does): that row's codes straddle two segments.
    const std::uint64_t split = 1777 * kRowWidth + split_bytes_extra;
    const auto dir            = std::filesystem::current_path() / "ple_gather_test_volumes";
    std::filesystem::create_directories(dir);
    write_file(dir / "a.bin", 1234, object.data(), split);
    write_file(dir / "b.bin", 777, object.data() + split, object.size() - split);
    int failures = 0;
    {
        // The mapped segments address real file mappings, as the loader's mapping does.
        const Mapping a(dir / "a.bin"), b(dir / "b.bin");
        PleTableView table;
        table.rows        = kRows;
        table.row_width   = kRowWidth;
        table.row_bytes   = kRowWidth;
        table.scale_plane = scale_plane;
        table.segments    = {
            {0, split, a.at(1234)},
            {split, object.size() - split, b.at(777)},
        };
        std::vector<std::int32_t> tokens(300);
        for (auto& token : tokens) { token = static_cast<std::int32_t>(rng() % 50000); }
        tokens[5] = tokens[140] = static_cast<std::int32_t>(kEos);
        if (split_bytes_extra != 0) {
            // Plant a 3-token context at positions 200..202 whose rows include the straddling row.
            PleGather probe(table, hash, false);
            const std::uint64_t target = split / kRowWidth;
            std::vector<std::uint64_t> r(hash.heads());
            bool found = false;
            for (std::uint32_t attempt = 0; attempt < 2000000 && !found; ++attempt) {
                const std::int64_t c[3] = {static_cast<std::int64_t>(rng() % 50000),
                                           static_cast<std::int64_t>(rng() % 50000),
                                           static_cast<std::int64_t>(rng() % 50000)};
                probe.rows(c, r);
                for (const auto row : r) {
                    if (row == target) {
                        tokens[202] = static_cast<std::int32_t>(c[0]);
                        tokens[201] = static_cast<std::int32_t>(c[1]);
                        tokens[200] = static_cast<std::int32_t>(c[2]);
                        found       = true;
                    }
                }
            }
            failures += check(found, "no context reaches the straddling row");
        }

        PleGather mapped(table, hash);
        PleGather plain(table, hash, false);
        failures +=
            check(mapped.resident_scale_bytes() == (uniform_scale ? 2 : kRows * 2),
                  "resident multipliers: the plane, or the one multiplier of a uniform plane");
        for (const auto [first, count] :
             {std::pair<std::size_t, std::size_t>{0, 1}, {6, 3}, {141, 1}, {10, 290}}) {
            std::vector<std::uint16_t> a_out(count * mapped.width()), c_out(a_out.size());
            mapped.gather(tokens, first, count, a_out);
            plain.gather(tokens, first, count, c_out);
            failures += check(a_out == c_out, "prefetched mapped rows differ from plain mapped");
            // An earlier prefetch() of the same columns (and a no-op one on plain) is a hint.
            std::vector<std::uint16_t> d_out(a_out.size());
            mapped.prefetch(tokens, first, count);
            plain.prefetch(tokens, first, count);
            mapped.gather(tokens, first, count, d_out);
            failures += check(a_out == d_out, "rows gathered after prefetch() differ");
        }
        // Mapped decode is independently the E4M3 row times its BF16 multiplier.
        std::vector<std::uint16_t> column(mapped.width());
        const std::size_t probe_column = split_bytes_extra != 0 ? 202 : 20;
        mapped.gather(tokens, probe_column, 1, column);
        std::vector<std::uint64_t> rows(hash.heads());
        const std::int64_t context[3] = {tokens[probe_column], tokens[probe_column - 1],
                                         tokens[probe_column - 2]};
        mapped.rows(context, rows);
        const auto e4m3 = [](std::uint8_t c) {
            const int e = (c >> 3) & 15, m = c & 7;
            const float v = e == 0 ? std::ldexp(m / 8.0F, -6) : std::ldexp(1.0F + m / 8.0F, e - 7);
            return (c & 0x80) ? -v : v;
        };
        for (std::uint32_t h = 0; h < hash.heads(); ++h) {
            std::uint16_t scale_bits;
            std::memcpy(&scale_bits, object.data() + scale_plane + rows[h] * 2, 2);
            float scale;
            const std::uint32_t scale_word = static_cast<std::uint32_t>(scale_bits) << 16;
            std::memcpy(&scale, &scale_word, 4);
            for (std::uint32_t i = 0; i < kRowWidth; ++i) {
                const float value =
                    e4m3(static_cast<std::uint8_t>(object[rows[h] * kRowWidth + i])) * scale;
                std::uint32_t bits;
                std::memcpy(&bits, &value, 4);
                const auto bf16 =
                    static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
                if (column[h * kRowWidth + i] != bf16) {
                    std::cerr << "n-gram row decode mismatch at head " << h << '\n';
                    ++failures;
                    break;
                }
            }
        }
    }
    std::filesystem::remove_all(dir);
    return failures;
}

// --- nvfp4_row_e4m3_g16, row_interleaved_v1 -------------------------------------------------

constexpr std::uint32_t kNvfp4Width  = 160;
constexpr std::uint32_t kNvfp4Record = 90; // 80 code bytes, then 10 E4M3FN group scales

// Oracle written from the format definition (exact FP64 decodes of the codes, one FP32 product
// chain, BF16 nearest-even chosen by comparing the two neighbouring BF16 values in FP64).
double e2m1_value(unsigned code) {
    const unsigned e = (code >> 1) & 3U, m = code & 1U;
    const double magnitude = e == 0 ? 0.5 * m : std::ldexp(1.0 + 0.5 * m, static_cast<int>(e) - 1);
    return (code & 8U) != 0 ? -magnitude : magnitude;
}

double e4m3fn_value(unsigned code) {
    const unsigned e = (code >> 3) & 15U, m = code & 7U;
    const double magnitude =
        e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, static_cast<int>(e) - 7);
    return (code & 0x80U) != 0 ? -magnitude : magnitude;
}

double bf16_value(std::uint16_t word) {
    const std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}

std::uint16_t bf16_nearest_even(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const auto toward_zero = static_cast<std::uint16_t>(bits >> 16);
    if ((bits & 0xFFFFU) == 0) { return toward_zero; }
    const auto away    = static_cast<std::uint16_t>(toward_zero + 1);
    const double below = std::fabs(static_cast<double>(value) - bf16_value(toward_zero));
    const double above = std::fabs(bf16_value(away) - static_cast<double>(value));
    if (below != above) { return below < above ? toward_zero : away; }
    return (toward_zero & 1U) != 0 ? away : toward_zero;
}

// Element i of one 90-byte row: bf16_rn((fp32(e2m1) * fp32(e4m3)) * G).
std::uint16_t nvfp4_reference(const std::byte* record, unsigned i, float global) {
    const auto pair     = static_cast<unsigned>(record[i / 2]);
    const auto code     = (i % 2 == 0) ? (pair & 15U) : (pair >> 4);
    const auto scale    = static_cast<unsigned>(record[kNvfp4Width / 2 + i / 16]);
    const auto a        = static_cast<float>(e2m1_value(code));
    const auto s        = static_cast<float>(e4m3fn_value(scale));
    const float product = a * s;
    return bf16_nearest_even(product * global);
}

PleHashConfig small_hash(std::vector<std::uint64_t> moduli) {
    PleHashConfig hash;
    hash.ngram_size      = 3;
    hash.heads_per_ngram = static_cast<std::uint32_t>(moduli.size() / 2);
    hash.eos_token_id    = kEos;
    hash.multipliers     = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    hash.head_moduli     = std::move(moduli);
    std::uint64_t offset = 0;
    for (const auto modulus : hash.head_moduli) {
        hash.head_offsets.push_back(offset);
        offset += modulus;
    }
    return hash;
}

// Every gathered value of `tokens[first, first + count)` against the oracle of its hashed row.
int check_nvfp4_columns(const PleGather& gather, const std::vector<std::int32_t>& tokens,
                        std::size_t first, std::size_t count,
                        const std::vector<std::uint16_t>& values,
                        const std::vector<std::byte>& object, float global,
                        std::set<std::uint64_t>& checked, const char* label) {
    const std::uint32_t heads = static_cast<std::uint32_t>(gather.width() / kNvfp4Width);
    std::vector<std::uint64_t> rows(heads);
    for (std::size_t c = 0; c < count; ++c) {
        gather.column_rows(tokens, first + c, rows);
        for (std::uint32_t h = 0; h < heads; ++h) {
            const std::byte* record = object.data() + rows[h] * kNvfp4Record;
            for (unsigned i = 0; i < kNvfp4Width; ++i) {
                const std::uint16_t got  = values[c * gather.width() + h * kNvfp4Width + i];
                const std::uint16_t want = nvfp4_reference(record, i, global);
                if (got != want) {
                    std::cerr << label << ": NVFP4 row " << rows[h] << " element " << i
                              << " decodes to 0x" << std::hex << got << ", oracle 0x" << want
                              << std::dec << '\n';
                    return 1;
                }
            }
            checked.insert(rows[h]);
        }
    }
    return 0;
}

// Writes `object` as two volume files split at `split` (leading bytes before each payload, so
// rows sit at unaligned file offsets) and checks mapped gathers against the oracle.
int nvfp4_volumes_case(const std::vector<std::byte>& object, std::uint64_t rows, float global,
                       const PleHashConfig& hash, std::uint64_t split,
                       std::vector<std::int32_t> tokens, std::mt19937& rng, const char* label,
                       std::size_t minimum_rows) {
    const auto dir = std::filesystem::current_path() / "ple_gather_nvfp4_volumes";
    std::filesystem::create_directories(dir);
    write_file(dir / "a.bin", 1234, object.data(), split);
    write_file(dir / "b.bin", 777, object.data() + split, object.size() - split);
    int failures = 0;
    {
        const Mapping a(dir / "a.bin"), b(dir / "b.bin");
        PleTableView table;
        table.format       = PleRowFormat::Nvfp4RowE4m3G16;
        table.rows         = rows;
        table.row_width    = kNvfp4Width;
        table.row_bytes    = kNvfp4Record;
        table.global_scale = global;
        table.segments     = {
            {0, split, a.at(1234)},
            {split, object.size() - split, b.at(777)},
        };
        failures += check(table.row_record_bytes() == kNvfp4Record, "NVFP4 record is not 90 B");
        // Plant a 3-token context at positions 200..202 whose rows include the row straddling
        // the volume boundary (when the split falls inside a row).
        if (split % kNvfp4Record != 0 && tokens.size() > 202) {
            PleGather probe(table, hash, false);
            const std::uint64_t target = split / kNvfp4Record;
            std::vector<std::uint64_t> r(hash.heads());
            bool found = false;
            for (std::uint32_t attempt = 0; attempt < 2000000 && !found; ++attempt) {
                const std::int64_t c[3] = {static_cast<std::int64_t>(rng() % 50000),
                                           static_cast<std::int64_t>(rng() % 50000),
                                           static_cast<std::int64_t>(rng() % 50000)};
                probe.rows(c, r);
                for (const auto row : r) {
                    if (row == target) {
                        tokens[202] = static_cast<std::int32_t>(c[0]);
                        tokens[201] = static_cast<std::int32_t>(c[1]);
                        tokens[200] = static_cast<std::int32_t>(c[2]);
                        found       = true;
                    }
                }
            }
            failures += check(found, "no context reaches the straddling NVFP4 row");
        }
        PleGather mapped(table, hash);
        PleGather plain(table, hash, false);
        std::set<std::uint64_t> checked;
        const std::size_t columns = tokens.size() - 2;
        for (const auto [first, count] :
             {std::pair<std::size_t, std::size_t>{0, 1}, {6, 3}, {141, 1}, {2, columns}}) {
            std::vector<std::uint16_t> out(count * mapped.width());
            mapped.prefetch(tokens, first, count);
            mapped.gather(tokens, first, count, out);
            failures += check_nvfp4_columns(mapped, tokens, first, count, out, object, global,
                                            checked, label);
            std::fill(out.begin(), out.end(), 0xFFFF);
            plain.gather(tokens, first, count, out);
            failures += check_nvfp4_columns(plain, tokens, first, count, out, object, global,
                                            checked, label);
        }
        if (split % kNvfp4Record != 0) {
            failures += check(checked.count(split / kNvfp4Record) == 1,
                              "the straddling NVFP4 row was not checked");
        }
        std::cout << label << ": " << checked.size()
                  << " distinct NVFP4 rows bit-exact (mapped, mapped without prefetch)\n";
        failures += check(checked.size() >= minimum_rows, "too few distinct NVFP4 rows checked");
    }
    std::filesystem::remove_all(dir);
    return failures;
}

int nvfp4_synthetic_case(float global) {
    constexpr std::uint64_t kTableRows = 5000;
    std::mt19937 rng(4321);
    std::vector<std::byte> object(kTableRows * kNvfp4Record);
    for (std::uint64_t r = 0; r < kTableRows; ++r) {
        std::byte* record = object.data() + r * kNvfp4Record;
        for (std::uint32_t i = 0; i < kNvfp4Width / 2; ++i) {
            record[i] = std::byte{static_cast<std::uint8_t>(rng())}; // every E2M1 pair
        }
        for (std::uint32_t g = 0; g < kNvfp4Width / 16; ++g) {
            std::uint8_t code = static_cast<std::uint8_t>(rng());
            if ((code & 0x7F) == 0x7F) { code ^= 1; } // finite E4M3FN scales only
            record[kNvfp4Width / 2 + g] = std::byte{code};
        }
    }
    std::vector<std::int32_t> tokens(1500);
    for (auto& token : tokens) { token = static_cast<std::int32_t>(rng() % 50000); }
    tokens[5] = tokens[140] = static_cast<std::int32_t>(kEos);
    // The volume boundary falls 5 bytes into row 1777.
    return nvfp4_volumes_case(object, kTableRows, global, small_hash({1201, 1213, 1217, 1223}),
                              1777 * kNvfp4Record + 5, tokens, rng, "NVFP4 synthetic", 3000);
}

// --- real rows of the FP4-Local checkpoint (NINFER_TEST_FP4_LOCAL_CHECKPOINT) -----------------

struct SafetensorsEntry {
    std::filesystem::path file;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0; // absolute file offset of the data
    std::uint64_t bytes  = 0;
};

SafetensorsEntry safetensors_entry(const std::filesystem::path& file, const std::string& name) {
    std::ifstream in(file, std::ios::binary);
    std::uint64_t header_bytes = 0;
    in.read(reinterpret_cast<char*>(&header_bytes), 8);
    std::string header(header_bytes, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_bytes));
    if (!in) { throw std::runtime_error(file.string() + ": unreadable safetensors header"); }
    const auto json  = nlohmann::json::parse(header);
    const auto& item = json.at(name);
    SafetensorsEntry out;
    out.file           = file;
    out.dtype          = item.at("dtype").get<std::string>();
    out.shape          = item.at("shape").get<std::vector<std::uint64_t>>();
    const auto offsets = item.at("data_offsets").get<std::vector<std::uint64_t>>();
    out.offset         = 8 + header_bytes + offsets.at(0);
    out.bytes          = offsets.at(1) - offsets.at(0);
    return out;
}

void read_at(const SafetensorsEntry& entry, std::uint64_t within, std::byte* out,
             std::uint64_t bytes) {
    if (within + bytes > entry.bytes) { throw std::runtime_error("read past a tensor"); }
    std::ifstream in(entry.file, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(entry.offset + within));
    in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(bytes));
    if (!in) { throw std::runtime_error(entry.file.string() + ": short tensor read"); }
}

int nvfp4_checkpoint_case() {
    const char* root = std::getenv("NINFER_TEST_FP4_LOCAL_CHECKPOINT");
    if (root == nullptr || root[0] == '\0') {
        std::cout << "note: NINFER_TEST_FP4_LOCAL_CHECKPOINT unset; real NVFP4 rows not checked\n";
        return 0;
    }
    const std::filesystem::path dir(root);
    std::ifstream index_file(dir / "model.safetensors.index.json");
    const auto index         = nlohmann::json::parse(index_file).at("weight_map");
    const std::string prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
    const auto entry         = [&](const std::string& name) {
        return safetensors_entry(dir / index.at(name).get<std::string>(), name);
    };
    const auto global_entry = entry(prefix + "weight_scale_2");
    if (global_entry.dtype != "F32" || global_entry.bytes != 4) {
        throw std::runtime_error("weight_scale_2 is not one F32 value");
    }
    float global = 0.0F;
    read_at(global_entry, 0, reinterpret_cast<std::byte*>(&global), 4);
    // Two 200-row blocks from each of three fragments (start, interior, end of the table).
    constexpr std::uint64_t kBlock = 200;
    std::vector<std::byte> object;
    for (const int shard : {0, 64, 127}) {
        const std::string name = prefix + "shard_" + std::to_string(shard) + ".";
        const auto codes       = entry(name + "weight");
        const auto scales      = entry(name + "weight_scale");
        if (codes.dtype != "U8" || codes.shape != std::vector<std::uint64_t>{codes.shape[0], 80} ||
            scales.dtype != "F8_E4M3" ||
            scales.shape != std::vector<std::uint64_t>{codes.shape[0], 10}) {
            throw std::runtime_error(name + " is not an NVFP4 row fragment [R,80] + [R,10]");
        }
        const std::uint64_t fragment_rows = codes.shape[0];
        for (const std::uint64_t begin :
             {std::uint64_t{0} + static_cast<std::uint64_t>(shard),
              fragment_rows / 2 + 7 * static_cast<std::uint64_t>(shard)}) {
            std::vector<std::byte> code_block(kBlock * 80), scale_block(kBlock * 10);
            read_at(codes, begin * 80, code_block.data(), code_block.size());
            read_at(scales, begin * 10, scale_block.data(), scale_block.size());
            for (std::uint64_t r = 0; r < kBlock; ++r) {
                object.insert(object.end(), code_block.begin() + r * 80,
                              code_block.begin() + (r + 1) * 80);
                object.insert(object.end(), scale_block.begin() + r * 10,
                              scale_block.begin() + (r + 1) * 10);
            }
        }
    }
    const std::uint64_t rows = object.size() / kNvfp4Record; // 1200
    std::mt19937 rng(97);
    std::vector<std::int32_t> tokens(3000);
    for (auto& token : tokens) { token = static_cast<std::int32_t>(rng() % 248000); }
    std::cout << "FP4-Local n-gram table global scale " << global << '\n';
    // Heads covering 1194 of the 1200 rows; the boundary falls 37 bytes into row 611.
    return nvfp4_volumes_case(object, rows, global, small_hash({293, 283, 307, 311}),
                              611 * kNvfp4Record + 37, tokens, rng, "NVFP4 FP4-Local rows", 1000);
}

} // namespace

int main() {
    try {
        int failures = hash_case();
        failures += residency_case();
        failures += residency_case(5);                    // a row straddling two volume segments
        failures += residency_case(5, true);              // one multiplier on every row
        failures += nvfp4_synthetic_case(3.3242362e-05F); // the FP4-Local table's scale
        failures += nvfp4_synthetic_case(0.7310586F);
        failures += nvfp4_checkpoint_case();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " PLE gather\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "PLE gather: " << error.what() << '\n';
        return 1;
    }
}
