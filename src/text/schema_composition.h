#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::text {
struct NormalizedSchema {
    nlohmann::ordered_json schema;
    std::vector<std::pair<std::string, std::string>> origins;
};

// Reduce a source validated by prepare_json_schema before vocabulary compilation.
// References remain a graph; dialect and keyword validation belong to the caller.
// `strict` closes merged objects with declared properties and tuples to what is declared
// (json_schema.cpp close_undeclared). A conjunction may expand to at most `max_parts` subschemas
// (400 constraint_limit_exceeded beyond); the work is charged to the caller's compilation
// deadline.
NormalizedSchema normalize_schema_composition(const nlohmann::ordered_json& source,
                                              bool draft7 = false, std::size_t max_parts = 4096,
                                              bool strict = false);
} // namespace ninfer::text
