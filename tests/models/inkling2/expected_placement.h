#pragma once

// The tensor-parallel placement the Inkling 2 tests expect for each logical parameter at tp 2,
// written from the parameter names alone (independent of load/sharding.cpp).

#include "artifact/slices.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::inkling2 {

struct Expectation {
    artifact::ShardAxis axis = artifact::ShardAxis::Replicated;
    int holder               = 0; // SingleDevice
    std::string kind;             // inventory class
};

inline bool ends_with(const std::string& name, const std::string& suffix) {
    return name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline Expectation expected_placement(const std::string& name, std::uint32_t experts) {
    using artifact::ShardAxis;
    // Every expert on both ranks at half the intermediate width: gate_up by rows (half the gate
    // rows and the matching up rows), down by columns.
    if (name.find("/moe/experts/") != std::string::npos) {
        (void)experts;
        return {ends_with(name, "/down") ? ShardAxis::Columns : ShardAxis::Rows, 0, "experts"};
    }
    if (name == "text/output_head") { return {ShardAxis::Rows, 0, "output head"}; }
    if (name == "text/token_embedding") { return {ShardAxis::Replicated, 0, "embedding"}; }
    if (name.find("/attention/") != std::string::npos) {
        if (ends_with(name, "/relative_profile")) {
            return {ShardAxis::Replicated, 0, "attention"};
        }
        if (ends_with(name, "/output") || ends_with(name, "_conv_static")) {
            return {ShardAxis::Columns, 0, "attention"};
        }
        return {ShardAxis::Rows, 0, "attention"};
    }
    if (name.find("/message_gate/") != std::string::npos) {
        return {ShardAxis::Replicated, 0, "message gates"};
    }
    if (ends_with(name, "/moe/router") || ends_with(name, "/moe/router_bias")) {
        return {ShardAxis::Replicated, 0, "router"};
    }
    return {ShardAxis::Replicated, 0, "norms, queries, convs"};
}

// The rank-local logical shape of a split parameter.
inline std::vector<std::uint64_t> rank_shape(std::vector<std::uint64_t> shape,
                                             artifact::ShardAxis axis) {
    if (axis == artifact::ShardAxis::Rows) { shape.front() /= 2; }
    if (axis == artifact::ShardAxis::Columns) { shape.back() /= 2; }
    return shape;
}

} // namespace ninfer::test::inkling2
