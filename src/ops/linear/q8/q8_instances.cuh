#pragma once
#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_schedule.cuh"

namespace ninfer::ops::detail {
namespace q8_instances {
using MmaR32T64  = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using MmaR32T128 = Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>;
using MmaR64T96  = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
// The wide route of the linear family is the one schedule measured to want cg on the predicated
// path; every other schedule, here and in the other Ops that instantiate this same tile, keeps the
// inherited ca.
using MmaR64T128 = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2, Q8MmaFragmentPipeline::PingPong,
                                    Cache::cg, Cache::cg, Cache::cg>;
using MmaR128T64 = Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>;
// K128 single-activation-stage schedules shared with fused consumers.
using MmaR64x16T48K128A1 = Q8A16MmaSchedule<64, 48, 128, 16, 24, 1, 2>;
using MmaR64x32T64K128A1 = Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>;

using SimtR8T4         = Q8A16SimtSchedule<8, 4, 1, 32, 2, Cache::cg, 1>;
using SimtR8T8         = Q8A16SimtSchedule<8, 8, 1, 32, 2, Cache::cg, 1>;
using SimtR4T4W2G16S2  = Q8A16SimtSchedule<4, 4, 2, 16, 2, Cache::cg, 1>;
using SlicedR16T16W8S2 = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared>;
using SlicedR16T16W4S2 = Q8A16SlicedKMmaSchedule<16, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca,
                                                 Cache::cg, Q8ActivationStage::PaddedZero>;
using SlicedR16T32W4S2 = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared>;

// Register-resident decode rows (T <= 4): K 2560 (one warp per row; 8 CTAs/SM caps registers
// at 64), K 640 (one warp per row), K 2560 split over two warps per row.
using WarpRowR4P10  = Q8A16WarpRowSchedule<4, 4, 1, 10, 8>;
using WarpRowR4P3   = Q8A16WarpRowSchedule<4, 4, 1, 3, 1>;
using WarpRowR2W2P5 = Q8A16WarpRowSchedule<2, 4, 2, 5, 1>;
} // namespace q8_instances
} // namespace ninfer::ops::detail
