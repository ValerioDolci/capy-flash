#pragma once
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"

namespace ninfer::ops::detail {
// The A4 MMA tiles shared by the NVFP4 Linear shapes and the fused NVFP4 projections: T tokens by
// R output rows per CTA over 256-column K steps. The Pipelined tiles use two stages, the Resident
// tile one stage and at least two CTAs per SM; a W16 tile splits its rows over four warps instead
// of two (16 warps per CTA).
using Nvfp4A4T32R64               = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using Nvfp4A4T32R128              = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using Nvfp4A4T64R128              = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using Nvfp4A4T64R128W16           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 4, 2, 1>;
using Nvfp4A4T128R128Pipelined    = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using Nvfp4A4T128R128PipelinedW16 = Nvfp4A4MmaSchedule<128, 128, 256, 4, 4, 2, 1>;
using Nvfp4A4T128R128Resident     = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;
} // namespace ninfer::ops::detail
