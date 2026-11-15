#pragma once

#include "ninfer/types.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// Command-line forms of the Engine options shared by ninfer, ninfer-serve and ninfer-perplexity,
// and the names the logs and reports give their values.

// An unsigned decimal; empty, signed or out-of-range text is invalid.
[[nodiscard]] inline std::uint64_t parse_u64(const char* text, std::string_view label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

// `--kv-dtype bf16|int8|fp8|nvfp4|k8v4`.
[[nodiscard]] inline KvCacheStorage parse_kv_dtype(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

// The full name of a KV storage, as the request JSONL (`kv_cache`) and ninfer report it.
[[nodiscard]] inline const char* kv_cache_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-group64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

[[nodiscard]] inline const char* kv_capacity_mode_name(KvCapacityMode mode) noexcept {
    return mode == KvCapacityMode::Automatic ? "auto" : "explicit";
}

// `--vram-headroom-mib N`.
[[nodiscard]] inline std::size_t parse_vram_headroom_mib(const char* text) {
    const std::uint64_t mib = parse_u64(text, "vram-headroom-mib");
    if (mib > (std::numeric_limits<std::size_t>::max() >> 20)) {
        throw std::invalid_argument("--vram-headroom-mib is out of range");
    }
    return static_cast<std::size_t>(mib);
}

// Applies `--vram-headroom-mib`, which only `--kv-capacity auto` takes.
inline void apply_vram_headroom(KvCapacityPolicy& capacity, std::optional<std::size_t> mib) {
    if (!mib) { return; }
    if (capacity.mode != KvCapacityMode::Automatic) {
        throw std::invalid_argument("--vram-headroom-mib requires --kv-capacity auto");
    }
    capacity = KvCapacityPolicy::automatic(*mib << 20);
}

} // namespace ninfer::product
