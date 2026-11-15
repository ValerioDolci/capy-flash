#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product::media_acquire {

enum class SourceKind {
    Path,
    Url,
    Data,
    Bytes,
};

struct Source {
    SourceKind kind = SourceKind::Path;
    std::string value;
    std::string media_type;
    std::vector<std::uint8_t> bytes;
};

// Data for a data: URI, Url for an http(s):// URL, nullopt for anything else.
[[nodiscard]] inline std::optional<SourceKind> uri_source_kind(std::string_view value) {
    if (value.starts_with("data:")) { return SourceKind::Data; }
    if (value.starts_with("http://") || value.starts_with("https://")) { return SourceKind::Url; }
    return std::nullopt;
}

} // namespace ninfer::product::media_acquire
