// AllowA4Decode moves only the routes of the columns after the prefill, and only at widths
// 1..kAllowA4DecodeMaxWidth. Host only: the route resolvers launch nothing.
//
// The test also writes the NVFP4 route inventory (route_inventory below) and compares it with
// tests/ops/nvfp4_routes.txt, so every route change shows as a change of that file.
//
// The prefill runs every AllowA4 projection under AllowA4, so the tables below are the routes the
// 27B target's projections took before AllowA4Decode existed, at every width: under AllowA4 every
// resolver must still give them. Under AllowA4Decode the widths 1..7 take the decode routes
// (ninfer_verify_width_columns_test checks their bits) and from 8 on the AllowA4 ones.
#include "ninfer/ops/linear.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/gdn_input_proj/tp2/nvfp4_gdn_input_shard.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace {

using namespace ninfer;
using ops::LinearPolicy;

constexpr std::int32_t kMaxTokens = 64;

int failures = 0;

template <class Route>
void expect(const std::string& what, std::int32_t tokens, Route got, Route want) {
    if (got == want) return;
    std::cerr << what << " T=" << tokens << ": route " << static_cast<int>(got) << ", expected "
              << static_cast<int>(want) << '\n';
    ++failures;
}

// linear() over [n,k]: A4 from `first_a4` under AllowA4 (the crossover before AllowA4Decode), the
// shape's A16 launch below it; `decode` names the shapes with an AllowA4Decode launch.
void linear_routes(std::int32_t n, std::int32_t k, std::int32_t first_a4, bool decode) {
    using ops::detail::Nvfp4LinearRoute;
    const std::string name = "linear [" + std::to_string(n) + "," + std::to_string(k) + "]";
    for (std::int32_t t = 1; t <= kMaxTokens; ++t) {
        const Nvfp4LinearRoute prefill =
            t >= first_a4 ? Nvfp4LinearRoute::A4 : Nvfp4LinearRoute::A16;
        for (const LinearPolicy a16 : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
            expect(name + " A16Only/AllowA8", t, ops::detail::nvfp4_linear_route(n, k, a16, t),
                   Nvfp4LinearRoute::A16);
        }
        expect(name + " AllowA4", t,
               ops::detail::nvfp4_linear_route(n, k, LinearPolicy::AllowA4, t), prefill);
        const Nvfp4LinearRoute after =
            decode && t <= ops::kAllowA4DecodeMaxWidth && prefill == Nvfp4LinearRoute::A16
                ? Nvfp4LinearRoute::A16Decode
                : prefill;
        expect(name + " AllowA4Decode", t,
               ops::detail::nvfp4_linear_route(n, k, LinearPolicy::AllowA4Decode, t), after);
    }
}

// linear_add over [5120,k]: the output family has A4 from 17, the MLP down family from 8.
void linear_add_routes(std::int32_t k, std::int32_t first_a4) {
    using ops::detail::Nvfp4LinearAddRoute;
    const std::string name = "linear_add [5120," + std::to_string(k) + "]";
    for (std::int32_t t = 1; t <= kMaxTokens; ++t) {
        const Nvfp4LinearAddRoute prefill =
            t >= first_a4 ? Nvfp4LinearAddRoute::A4 : Nvfp4LinearAddRoute::A16;
        for (const LinearPolicy a16 : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
            expect(name + " A16Only/AllowA8", t,
                   ops::detail::nvfp4_linear_add_route(5120, k, a16, t), Nvfp4LinearAddRoute::A16);
        }
        expect(name + " AllowA4", t,
               ops::detail::nvfp4_linear_add_route(5120, k, LinearPolicy::AllowA4, t), prefill);
        const Nvfp4LinearAddRoute after =
            t <= ops::kAllowA4DecodeMaxWidth && prefill == Nvfp4LinearAddRoute::A16
                ? Nvfp4LinearAddRoute::A16Decode
                : prefill;
        expect(name + " AllowA4Decode", t,
               ops::detail::nvfp4_linear_add_route(5120, k, LinearPolicy::AllowA4Decode, t), after);
    }
}

// linear_swiglu (every NVFP4 gate|up problem shares the plan): under AllowA4 the fused A16 decode
// at T=1, small-T at 2..4, fused A4 from 5 and its TMA GEMM from 256.
void linear_swiglu_routes() {
    using ops::detail::Nvfp4LinearSwiGluRoute;
    for (std::int32_t t = 1; t <= 320; ++t) {
        const Nvfp4LinearSwiGluRoute prefill = t == 1    ? Nvfp4LinearSwiGluRoute::DecodeFusedA16
                                               : t <= 4  ? Nvfp4LinearSwiGluRoute::SmallTFusedA16
                                               : t < 256 ? Nvfp4LinearSwiGluRoute::FusedA4
                                                         : Nvfp4LinearSwiGluRoute::TmaFusedA4;
        expect("linear_swiglu AllowA4", t,
               ops::detail::nvfp4_linear_swiglu_route(LinearPolicy::AllowA4, t), prefill);
        expect("linear_swiglu AllowA4Decode", t,
               ops::detail::nvfp4_linear_swiglu_route(LinearPolicy::AllowA4Decode, t),
               t <= ops::kAllowA4DecodeMaxWidth ? Nvfp4LinearSwiGluRoute::SlicedKFusedA16
                                                : prefill);
        for (const LinearPolicy a16 : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
            expect("linear_swiglu A16Only/AllowA8", t,
                   ops::detail::nvfp4_linear_swiglu_route(a16, t),
                   t == 1 ? Nvfp4LinearSwiGluRoute::DecodeFusedA16
                          : Nvfp4LinearSwiGluRoute::SmallTFusedA16);
        }
    }
}

// attn_input_proj over [14336,5120] and its shard: A4 from 4 under AllowA4.
void attn_input_routes() {
    using ops::detail::Nvfp4AttnInputRoute;
    for (std::int32_t t = 1; t <= kMaxTokens; ++t) {
        const Nvfp4AttnInputRoute prefill =
            t >= 4 ? Nvfp4AttnInputRoute::A4 : Nvfp4AttnInputRoute::A16;
        expect("attn_input AllowA4", t,
               ops::detail::nvfp4_attn_input_route(LinearPolicy::AllowA4, t), prefill);
        expect("attn_input AllowA4Decode", t,
               ops::detail::nvfp4_attn_input_route(LinearPolicy::AllowA4Decode, t),
               t <= ops::kAllowA4DecodeMaxWidth ? Nvfp4AttnInputRoute::A4 : prefill);
        for (const LinearPolicy a16 : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
            expect("attn_input A16Only/AllowA8", t, ops::detail::nvfp4_attn_input_route(a16, t),
                   Nvfp4AttnInputRoute::A16);
        }
    }
}

// The GDN q|k|v|z projection of the conv snapshot and record forms: at B=1 under AllowA4 the fused
// A16 decode at T=1, small-T at 2..3, the materialized A4 projection from 4; every B > 1
// materializes.
void gdn_conv_routes() {
    using ops::detail::Nvfp4GdnConvScheduleId;
    for (std::int32_t batch = 1; batch <= 8; ++batch) {
        for (std::int32_t t = 1; t <= 16; ++t) {
            const std::string name = "gdn conv B=" + std::to_string(batch);
            const Nvfp4GdnConvScheduleId prefill =
                batch > 1 || t >= 4 ? Nvfp4GdnConvScheduleId::Materialized
                : t == 1            ? Nvfp4GdnConvScheduleId::DecodeFusedA16
                                    : Nvfp4GdnConvScheduleId::SmallTFusedA16;
            expect(
                name + " AllowA4", t,
                ops::detail::nvfp4_gdn_conv_resolve_plan(LinearPolicy::AllowA4, t, batch).schedule,
                prefill);
            expect(name + " AllowA4 uses A4", t,
                   ops::detail::nvfp4_gdn_conv_uses_a4(LinearPolicy::AllowA4, t, batch),
                   prefill == Nvfp4GdnConvScheduleId::Materialized);
            const Nvfp4GdnConvScheduleId after =
                t <= ops::kAllowA4DecodeMaxWidth ? Nvfp4GdnConvScheduleId::Materialized : prefill;
            expect(name + " AllowA4Decode", t,
                   ops::detail::nvfp4_gdn_conv_resolve_plan(LinearPolicy::AllowA4Decode, t, batch)
                       .schedule,
                   after);
            expect(name + " AllowA4Decode uses A4", t,
                   ops::detail::nvfp4_gdn_conv_uses_a4(LinearPolicy::AllowA4Decode, t, batch),
                   after == Nvfp4GdnConvScheduleId::Materialized);
        }
    }
}

// The two-device GDN a|b gating shard: the GEMV at T=1, the fused split-10 kernel through 32, the
// two-kernel split-10 form above.
void gdn_gating_routes() {
    using ops::detail::Bf16GdnGatingShardRoute;
    for (std::int32_t t = 1; t <= kMaxTokens; ++t) {
        const Bf16GdnGatingShardRoute prefill = t == 1    ? Bf16GdnGatingShardRoute::Gemv
                                                : t <= 32 ? Bf16GdnGatingShardRoute::FusedSplit10
                                                          : Bf16GdnGatingShardRoute::Split10;
        for (const LinearPolicy policy :
             {LinearPolicy::A16Only, LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
            expect("gdn gating shard", t, ops::detail::bf16_gdn_gating_shard_route(policy, t),
                   prefill);
        }
        expect("gdn gating shard AllowA4Decode", t,
               ops::detail::bf16_gdn_gating_shard_route(LinearPolicy::AllowA4Decode, t),
               t == 1 ? Bf16GdnGatingShardRoute::FusedSplit10 : prefill);
    }
}

// The route inventory: for every registered NVFP4 problem of every op, every policy and every
// width 1..kInventoryTokens, the route taken, as maximal width intervals of one route. Launches
// have no name at run time: within one linear() problem `a16#i` and `a4#i` number them in order of
// first appearance (policies in the order of kPolicies, widths ascending), and an A4 launch adds
// the scale layout its quantizer writes. The fused ops show the route their public resolver gives,
// then (fused_a4_inventory) the GEMM their A4 launcher selects at every width, numbered the same
// way within each problem.
constexpr std::int32_t kInventoryTokens = 65536;
constexpr LinearPolicy kPolicies[]      = {LinearPolicy::A16Only, LinearPolicy::AllowA8,
                                           LinearPolicy::AllowA4, LinearPolicy::AllowA4Decode};

const char* policy_name(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return "A16Only";
    case LinearPolicy::AllowA8:
        return "AllowA8";
    case LinearPolicy::AllowA4:
        return "AllowA4";
    case LinearPolicy::AllowA4Decode:
        return "AllowA4Decode";
    }
    return "?";
}

// Writes `route(t)` for t = 1..kInventoryTokens as intervals of equal value, one line each.
void write_intervals(std::ostream& out, const std::string& prefix,
                     const std::function<std::string(std::int32_t)>& route) {
    std::int32_t first  = 1;
    std::string current = route(1);
    for (std::int32_t t = 2; t <= kInventoryTokens + 1; ++t) {
        std::string next = t <= kInventoryTokens ? route(t) : std::string{};
        if (t <= kInventoryTokens && next == current) continue;
        out << prefix << ' ' << first << ".." << (t - 1) << ' ' << current << '\n';
        first   = t;
        current = std::move(next);
    }
}

const char* scale_layout_name(ops::detail::Nvfp4ScaleLayout layout) {
    using ops::detail::Nvfp4ScaleLayout;
    switch (layout) {
    case Nvfp4ScaleLayout::RowMajor:
        return "RowMajor";
    case Nvfp4ScaleLayout::Tiled128:
        return "Tiled128";
    case Nvfp4ScaleLayout::Tiled256:
        return "Tiled256";
    }
    return "?";
}

void linear_inventory(std::ostream& out) {
    using ops::detail::Nvfp4LinearRoute;
    for (const auto* shape : ops::detail::nvfp4_linear_shapes()) {
        std::map<ops::detail::Nvfp4Launch, int> a16;
        std::map<ops::detail::Nvfp4A4Launch, int> a4;
        const std::string name =
            "linear " + std::to_string(shape->n) + "x" + std::to_string(shape->k);
        for (const LinearPolicy policy : kPolicies) {
            write_intervals(out, name + ' ' + policy_name(policy), [&](std::int32_t t) {
                const auto selection =
                    ops::detail::nvfp4_linear_selection(shape->n, shape->k, policy, t);
                if (selection.route == Nvfp4LinearRoute::A4) {
                    const int id =
                        a4.emplace(selection.a4.launch, static_cast<int>(a4.size())).first->second;
                    return "A4 a4#" + std::to_string(id) + ' ' +
                           scale_layout_name(selection.a4.scales);
                }
                const int id =
                    a16.emplace(selection.a16, static_cast<int>(a16.size())).first->second;
                return std::string(selection.route == Nvfp4LinearRoute::A16 ? "A16" : "A16Decode") +
                       " a16#" + std::to_string(id);
            });
        }
    }
}

const char* route_name(ops::detail::Nvfp4LinearAddRoute route) {
    using ops::detail::Nvfp4LinearAddRoute;
    switch (route) {
    case Nvfp4LinearAddRoute::A16:
        return "A16";
    case Nvfp4LinearAddRoute::A16Decode:
        return "A16Decode";
    case Nvfp4LinearAddRoute::A4:
        return "A4";
    }
    return "?";
}

const char* route_name(ops::detail::Nvfp4LinearSwiGluRoute route) {
    using ops::detail::Nvfp4LinearSwiGluRoute;
    switch (route) {
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        return "DecodeFusedA16";
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        return "SmallTFusedA16";
    case Nvfp4LinearSwiGluRoute::SlicedKFusedA16:
        return "SlicedKFusedA16";
    case Nvfp4LinearSwiGluRoute::FusedA4:
        return "FusedA4";
    case Nvfp4LinearSwiGluRoute::TmaFusedA4:
        return "TmaFusedA4";
    }
    return "?";
}

const char* route_name(ops::detail::Nvfp4AttnInputRoute route) {
    return route == ops::detail::Nvfp4AttnInputRoute::A4 ? "A4" : "A16";
}

const char* route_name(ops::detail::Nvfp4GdnConvScheduleId route) {
    using ops::detail::Nvfp4GdnConvScheduleId;
    switch (route) {
    case Nvfp4GdnConvScheduleId::DecodeFusedA16:
        return "DecodeFusedA16";
    case Nvfp4GdnConvScheduleId::SmallTFusedA16:
        return "SmallTFusedA16";
    case Nvfp4GdnConvScheduleId::Materialized:
        return "Materialized";
    }
    return "?";
}

const char* route_name(ops::detail::Bf16GdnGatingShardRoute route) {
    using ops::detail::Bf16GdnGatingShardRoute;
    switch (route) {
    case Bf16GdnGatingShardRoute::Gemv:
        return "Gemv";
    case Bf16GdnGatingShardRoute::FusedSplit10:
        return "FusedSplit10";
    case Bf16GdnGatingShardRoute::Split10:
        return "Split10";
    }
    return "?";
}

// A resolver that rejects the policy shows as `rejected` over every width.
template <class Resolve>
std::string resolve_or_rejected(Resolve resolve) {
    try {
        return route_name(resolve());
    } catch (const std::exception&) { return "rejected"; }
}

void fused_inventory(std::ostream& out) {
    for (const std::int32_t k : {6144, 3072, 17408, 8704}) {
        for (const LinearPolicy policy : kPolicies) {
            write_intervals(out, "linear_add 5120x" + std::to_string(k) + ' ' + policy_name(policy),
                            [&](std::int32_t t) {
                                return resolve_or_rejected([&] {
                                    return ops::detail::nvfp4_linear_add_route(5120, k, policy, t);
                                });
                            });
        }
    }
    for (const LinearPolicy policy : kPolicies) {
        write_intervals(out, std::string("linear_swiglu ") + policy_name(policy),
                        [&](std::int32_t t) {
                            return resolve_or_rejected(
                                [&] { return ops::detail::nvfp4_linear_swiglu_route(policy, t); });
                        });
    }
    for (const LinearPolicy policy : kPolicies) {
        write_intervals(out, std::string("attn_input ") + policy_name(policy), [&](std::int32_t t) {
            return resolve_or_rejected(
                [&] { return ops::detail::nvfp4_attn_input_route(policy, t); });
        });
    }
    for (std::int32_t batch = 1; batch <= 8; ++batch) {
        for (const LinearPolicy policy : kPolicies) {
            write_intervals(
                out, "gdn_conv B=" + std::to_string(batch) + ' ' + policy_name(policy),
                [&](std::int32_t t) {
                    return resolve_or_rejected([&] {
                        return ops::detail::nvfp4_gdn_conv_resolve_plan(policy, t, batch).schedule;
                    });
                });
        }
    }
    for (const LinearPolicy policy : kPolicies) {
        write_intervals(out, std::string("gdn_gating_shard ") + policy_name(policy),
                        [&](std::int32_t t) {
                            return resolve_or_rejected([&] {
                                return ops::detail::bf16_gdn_gating_shard_route(policy, t);
                            });
                        });
    }
}

// `select(t)` gives a launch and its scale layout; the launches are numbered as in
// linear_inventory.
template <class Select>
void write_a4_selection(std::ostream& out, const std::string& prefix, Select select) {
    std::map<decltype(select(1).launch), int> ids;
    write_intervals(out, prefix, [&](std::int32_t t) {
        const auto route = select(t);
        const int id     = ids.emplace(route.launch, static_cast<int>(ids.size())).first->second;
        return "a4#" + std::to_string(id) + ' ' + scale_layout_name(route.scales);
    });
}

// The fused ops' A4 launchers select their GEMM from the width alone, under every A4 policy. The
// attention shard's per-GPU TMA band (nvfp4_attn_input_shard_a4_launch, DeviceTuning) is outside
// the selection shown here.
void fused_a4_inventory(std::ostream& out) {
    write_a4_selection(out, "attn_input a4", ops::detail::nvfp4_attn_input_select_a4);
    write_a4_selection(out, "attn_input_shard a4", ops::detail::nvfp4_attn_input_shard_select_a4);
    write_a4_selection(out, "gdn_input a4", ops::detail::nvfp4_gdn_input_select_a4);
    write_a4_selection(out, "gdn_input_shard a4", ops::detail::nvfp4_gdn_input_shard_select_a4);
    for (const std::int32_t k : {6144, 3072, 17408, 8704}) {
        write_a4_selection(
            out, "linear_add 5120x" + std::to_string(k) + " a4",
            [k](std::int32_t t) { return ops::detail::nvfp4_linear_add_select_a4(5120, k, t); });
    }
    std::map<ops::detail::Nvfp4LinearSwiGluA4Launch, int> swiglu;
    write_intervals(out, "linear_swiglu a4", [&](std::int32_t t) {
        const auto launch = ops::detail::nvfp4_linear_swiglu_select_a4(t);
        const int id      = swiglu.emplace(launch, static_cast<int>(swiglu.size())).first->second;
        return "a4#" + std::to_string(id) + " RowMajor";
    });
}

std::string route_inventory() {
    std::ostringstream out;
    out << "# NVFP4 route inventory, widths 1.." << kInventoryTokens
        << " (tests/ops/test_decode_policy_routes.cpp).\n";
    linear_inventory(out);
    fused_inventory(out);
    fused_a4_inventory(out);
    return out.str();
}

// Compares the inventory with the file in the source tree; on a difference writes the inventory
// to nvfp4_routes.txt in the working directory and names the first differing line.
bool inventory_matches() {
    const std::string expected_path = NINFER_SOURCE_DIR "/tests/ops/nvfp4_routes.txt";
    std::ifstream file(expected_path);
    std::stringstream expected;
    expected << file.rdbuf();
    const std::string actual = route_inventory();
    if (file && expected.str() == actual) return true;
    std::ofstream("nvfp4_routes.txt") << actual;
    std::istringstream a(actual), e(expected.str());
    std::string actual_line, expected_line;
    int line = 0;
    while (true) {
        ++line;
        const bool more_a = static_cast<bool>(std::getline(a, actual_line));
        const bool more_e = static_cast<bool>(std::getline(e, expected_line));
        if (!more_a && !more_e) break;
        if (more_a && more_e && actual_line == expected_line) continue;
        std::cerr << "route inventory differs from " << expected_path << " at line " << line
                  << ":\n  expected: " << (more_e ? expected_line : "<end>")
                  << "\n  actual:   " << (more_a ? actual_line : "<end>") << '\n';
        break;
    }
    std::cerr << "the actual inventory is in nvfp4_routes.txt of the working directory\n";
    return false;
}

} // namespace

int main() {
    static_assert(ops::decode_policy(LinearPolicy::AllowA4) == LinearPolicy::AllowA4Decode);
    static_assert(ops::decode_policy(LinearPolicy::A16Only) == LinearPolicy::A16Only);
    static_assert(ops::decode_policy(LinearPolicy::AllowA8) == LinearPolicy::AllowA8);
    static_assert(ops::allows_a4(LinearPolicy::AllowA4Decode));

    // The attention/GDN output projection [5120,6144] and its two-device half.
    linear_routes(5120, 6144, 17, true);
    linear_routes(5120, 3072, 17, true);
    linear_add_routes(6144, 17);
    linear_add_routes(3072, 17);
    // The MLP down projection [5120,17408] and its half.
    linear_routes(5120, 17408, 8, true);
    linear_routes(5120, 8704, 8, true);
    linear_add_routes(17408, 8);
    linear_add_routes(8704, 8);
    // The attention q|k|gate|v parent [14336,5120] and its shard through linear().
    linear_routes(14336, 5120, 4, false);
    linear_routes(7168, 5120, 4, false);

    // The MLP gate|up.
    linear_swiglu_routes();
    // The attention q|k|gate|v and the GDN q|k|v|z of the snapshot and record forms.
    attn_input_routes();
    gdn_conv_routes();
    gdn_gating_routes();

    if (!inventory_matches()) ++failures;

    std::cout << (failures == 0 ? "OK" : "FAIL") << " decode policy routes\n";
    return failures == 0 ? 0 : 1;
}
