#pragma once

#include "artifact/slices.h"
#include "core/arena.h"
#include "core/weight_view.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::inkling2 {

// One bank of routed experts as ops::moe_bank_capped_swiglu reads it: every expert of a layer
// (IDs [0, E)) at one half of the expert intermediate width F. Half h holds channels
// [h F/2, (h+1) F/2):
//   gate_up [E*F, H]: per expert its F/2 gate rows of those channels, then the F/2 matching up rows
//   down    [E*H, F/2]: per expert the down columns of those channels
// in the artifact's format (NVFP4 block_scale_k16_m128x4 with one divisor per expert, MXFP8
// block_scale_k32, or BF16): the artifact banks' codes, scales and divisors, the scales moved
// word by word into this parent's layout (load/expert_banks.h).
struct ExpertBank {
    WeightParent gate_up_parent, down_parent;
    WeightView gate_up, down;
    std::int32_t first_expert = 0;
    std::int32_t experts      = 0;
};

struct ExpertBanks {
    // One device allocation per rank holds every parent of its banks.
    std::array<DeviceBuffer, artifact::kMaximumDevices> storage;
    // [device][layer]: a layer's banks in call order (Store, then Accumulate).
    std::array<std::vector<std::vector<std::unique_ptr<ExpertBank>>>, artifact::kMaximumDevices>
        layers;
};

} // namespace ninfer::models::inkling2
