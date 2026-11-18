#pragma once
#include "ops/linear/nvfp4/nvfp4_launch.h"

namespace ninfer::ops::detail {
// A registered [n,k] problem: its selections by token count. nvfp4_dispatch() asks them at every
// call, and the route inventory asks them for every width.
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch (*a16)(std::int32_t tokens);
    // Null when the shape has no A4 route (uses_a4 is then always false).
    Nvfp4A4Route (*a4)(std::int32_t tokens);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
    // The A16 selection of AllowA4Decode below the A4 crossover, where it differs from `a16`:
    // every width there gives a column the bits of width 4. Null: AllowA4Decode runs `a16` there.
    Nvfp4Launch (*a16_decode)(std::int32_t tokens) = nullptr;
};

extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
extern const Nvfp4LinearShape kNvfp4N17408K5120;
extern const Nvfp4LinearShape kNvfp4N5120K8704;
extern const Nvfp4LinearShape kNvfp4N5120K3072;
// Dense FFN of width 14336 over hidden 5120: gate|up, down and the down projection's two-device
// half.
extern const Nvfp4LinearShape kNvfp4N28672K5120;
extern const Nvfp4LinearShape kNvfp4N5120K14336;
extern const Nvfp4LinearShape kNvfp4N5120K7168;
// Two-device row halves of the GDN q|k|v|z and attention q|k|gate|v parents at hidden 5120 with 24
// query / 4 key-value heads, as plain Linear problems.
extern const Nvfp4LinearShape kNvfp4N8192K5120;
extern const Nvfp4LinearShape kNvfp4N7168K5120;
extern const Nvfp4LinearShape kNvfp4N8192K2560;
extern const Nvfp4LinearShape kNvfp4N6656K2560;
extern const Nvfp4LinearShape kNvfp4N124160K2560;
extern const Nvfp4LinearShape kNvfp4N512K2560;
extern const Nvfp4LinearShape kNvfp4N640K2560;
extern const Nvfp4LinearShape kNvfp4N2560K3072;
// One-device Qwen3.8-Flash-Next shapes of the NVFP4-dense recipe.
extern const Nvfp4LinearShape kNvfp4N13312K2560;
extern const Nvfp4LinearShape kNvfp4N16384K2560;
extern const Nvfp4LinearShape kNvfp4N1280K2560;
extern const Nvfp4LinearShape kNvfp4N248320K2560;
extern const Nvfp4LinearShape kNvfp4N2560K6144;
extern const Nvfp4LinearShape kNvfp4N2560K640;
} // namespace ninfer::ops::detail
