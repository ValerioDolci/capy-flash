// GPU conformance and speed of the q3_g128_fp16 A8 prefill routes.
//
// The A8 routes quantize each (token, 128-value group) of the activations to int8 with an FP32
// scale amax / 127 (round to nearest even, clamp to +-127), take exact int32 dot products with the
// 3-bit codes per group, and accumulate fmaf(w_scale * x_scale, float(partial), total) in group
// order. Two kernels implement it: the in-kernel quantizing tile (route 2, the one whose perplexity
// was qualified) and the tall workspace route (route 3).
//
// Pre-registered criteria:
//   (1) route 3 == route 2 bit for bit on every epilogue (Linear, residual add, SwiGLU, segments),
//       every 27B Text shape and its TP2 shard, T in {64, 65, 127, 128, 129, 200, 2048};
//   (2) Linear and residual add: route 3 == a host emulation of the A8 arithmetic bit for bit;
//       SwiGLU within 2^-8 relative + 2^-20 absolute of the emulated silu(g) * u (device expf);
//   (3) repeated launches are bitwise identical; T < 64 never takes an A8 route (== route 1).
// NINFER_Q3_BENCH=1 adds a timing table (routes 1/2/3 at T = 512, 2048, 4096; 27B and TP2 shapes).
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/q3/q3_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace detail = ninfer::ops::detail;

int failures = 0;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct DeviceBytes {
    void* p = nullptr;

    explicit DeviceBytes(std::size_t bytes) {
        cuda_ok(cudaMalloc(&p, bytes ? bytes : 16), "malloc");
    }

    ~DeviceBytes() { cudaFree(p); }

    DeviceBytes(const DeviceBytes&)            = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
};

float bf16_value(std::uint16_t bits) {
    std::uint32_t w = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}

std::uint16_t bf16_bits(float f) {
    std::uint32_t w;
    std::memcpy(&w, &f, 4);
    const std::uint32_t rounding = 0x7fffu + ((w >> 16) & 1u);
    return static_cast<std::uint16_t>((w + rounding) >> 16);
}

// Host model of one q3_g128_fp16 matrix.
struct HostQ3 {
    int n = 0, k = 0;
    std::vector<std::int8_t> codes;    // [n][k]
    std::vector<std::uint16_t> scales; // [n][k/128] binary16
    std::vector<std::uint8_t> payload; // row-split payload
    std::size_t scale_offset = 0;

    double w(int r, int c) const {
        return static_cast<double>(codes[static_cast<std::size_t>(r) * k + c]) *
               static_cast<double>(__half2float(__ushort_as_half(scales[r * (k / 128) + c / 128])));
    }
};

HostQ3 make_q3(int n, int k, std::uint32_t seed) {
    HostQ3 h;
    h.n = n;
    h.k = k;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(-4, 3);
    std::uniform_real_distribution<float> scale(0.004F, 0.04F);
    h.codes.resize(static_cast<std::size_t>(n) * k);
    for (auto& c : h.codes) c = static_cast<std::int8_t>(code(rng));
    h.scales.resize(static_cast<std::size_t>(n) * (k / 128));
    for (auto& s : h.scales) s = __half_as_ushort(__float2half(scale(rng)));
    const std::size_t code_bytes = static_cast<std::size_t>(n) * (k / 128) * 48;
    h.scale_offset               = (code_bytes + 255) / 256 * 256;
    h.payload.assign(h.scale_offset + h.scales.size() * 2, 0);
    for (std::size_t i = 0; i < h.codes.size(); ++i) {
        const unsigned u = static_cast<unsigned>(h.codes[i]) & 7u;
        for (int b = 0; b < 3; ++b) {
            const std::size_t bit = 3 * i + b;
            if ((u >> b) & 1u) h.payload[bit / 8] |= static_cast<std::uint8_t>(1u << (bit % 8));
        }
    }
    std::memcpy(h.payload.data() + h.scale_offset, h.scales.data(), h.scales.size() * 2);
    return h;
}

Weight device_weight(const HostQ3& h, const DeviceBytes& d) {
    Weight w;
    w.payload       = d.p;
    w.payload_bytes = h.payload.size();
    w.qtype         = QType::Q3_G128_FP16;
    w.layout        = QuantLayout::RowSplit;
    w.group_size    = 128;
    w.group         = 128;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = h.n;
    w.k = w.shape[1] = w.padded_shape[1] = h.k;
    w.qdata                              = d.p;
    w.scales                             = static_cast<std::uint8_t*>(d.p) + h.scale_offset;
    w.scale_dtype                        = DType::FP16;
    w.scale_ne[0]                        = h.k / 128;
    w.scale_ne[1]                        = h.n;
    w.scale_nb[0]                        = 2;
    w.scale_nb[1]                        = (h.k / 128) * 2;
    w.scale_nb[2] = w.scale_nb[3] = w.scale_nb[1] * h.n;
    return w;
}

std::vector<std::uint16_t> make_x(int k, int t, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * t);
    for (auto& v : x) v = bf16_bits(dist(rng));
    return x;
}

void report(const std::string& name, int bad, int total) {
    std::printf("%-64s %s (%d/%d)\n", name.c_str(), bad ? "FAIL" : "ok", bad, total);
    if (bad) ++failures;
}

std::size_t ws_bytes(int k, int t) {
    detail::q3_force_route_for_test(3);
    const std::size_t b = detail::q3_workspace_bytes(k, t);
    detail::q3_force_route_for_test(0);
    return b;
}

// Host emulation of the A8 arithmetic: FP32 totals [t][n].
std::vector<float> emulate_a8(const HostQ3& h, const std::vector<std::uint16_t>& x, int t) {
    const int groups = h.k / 128;
    std::vector<std::int8_t> q(static_cast<std::size_t>(t) * h.k);
    std::vector<float> sx(static_cast<std::size_t>(t) * groups);
    for (int tok = 0; tok < t; ++tok) {
        for (int g = 0; g < groups; ++g) {
            float amax = 0.0F;
            for (int i = 0; i < 128; ++i) {
                amax = std::fmax(amax, std::fabs(bf16_value(
                                           x[static_cast<std::size_t>(tok) * h.k + g * 128 + i])));
            }
            const float scale                              = amax > 0.0F ? amax / 127.0F : 0.0F;
            const float inv                                = amax > 0.0F ? 127.0F / amax : 0.0F;
            sx[static_cast<std::size_t>(tok) * groups + g] = scale;
            for (int i = 0; i < 128; ++i) {
                const float v =
                    bf16_value(x[static_cast<std::size_t>(tok) * h.k + g * 128 + i]) * inv;
                long r = std::lrint(v);
                r      = r < -127 ? -127 : (r > 127 ? 127 : r);
                q[static_cast<std::size_t>(tok) * h.k + g * 128 + i] = static_cast<std::int8_t>(r);
            }
        }
    }
    std::vector<float> out(static_cast<std::size_t>(t) * h.n);
    for (int tok = 0; tok < t; ++tok) {
        for (int r = 0; r < h.n; ++r) {
            float total = 0.0F;
            for (int g = 0; g < groups; ++g) {
                int part             = 0;
                const std::int8_t* c = h.codes.data() + static_cast<std::size_t>(r) * h.k + g * 128;
                const std::int8_t* a = q.data() + static_cast<std::size_t>(tok) * h.k + g * 128;
                for (int i = 0; i < 128; ++i) part += c[i] * a[i];
                const float sw = __half2float(__ushort_as_half(h.scales[r * groups + g]));
                total          = std::fma(sw * sx[static_cast<std::size_t>(tok) * groups + g],
                                          static_cast<float>(part), total);
            }
            out[static_cast<std::size_t>(tok) * h.n + r] = total;
        }
    }
    return out;
}

enum class Kind { Linear, Add, Swiglu, Segments };

// Runs one call on route `route` and returns every output (BF16 bits, segments concatenated).
std::vector<std::uint16_t> call(Kind kind, const HostQ3& h, const DeviceBytes& dw,
                                const DeviceBytes& dx, const std::vector<std::uint16_t>& residual,
                                int t, int route, const DeviceBytes* ws, std::size_t ws_size) {
    detail::q3_force_route_for_test(route);
    const Weight w = device_weight(h, dw);
    Tensor tx(dx.p, DType::BF16, {h.k, t});
    const detail::Q3Workspace q3ws{ws ? ws->p : nullptr, ws ? ws_size : 0};
    std::vector<std::uint16_t> out;
    if (kind == Kind::Linear || kind == Kind::Add) {
        DeviceBytes d(static_cast<std::size_t>(h.n) * t * 2);
        Tensor to(d.p, DType::BF16, {h.n, t});
        if (kind == Kind::Linear) {
            detail::q3_linear_dispatch(tx, w, to, nullptr, q3ws);
        } else {
            cuda_ok(cudaMemcpy(d.p, residual.data(), residual.size() * 2, cudaMemcpyHostToDevice),
                    "r");
            detail::q3_linear_add_dispatch(tx, w, to, nullptr, q3ws);
        }
        cuda_ok(cudaDeviceSynchronize(), "call");
        out.resize(static_cast<std::size_t>(h.n) * t);
        cuda_ok(cudaMemcpy(out.data(), d.p, out.size() * 2, cudaMemcpyDeviceToHost), "out");
    } else if (kind == Kind::Swiglu) {
        DeviceBytes d(static_cast<std::size_t>(h.n / 2) * t * 2);
        Tensor to(d.p, DType::BF16, {h.n / 2, t});
        detail::q3_linear_swiglu_dispatch(tx, w, to, nullptr, q3ws);
        cuda_ok(cudaDeviceSynchronize(), "call");
        out.resize(static_cast<std::size_t>(h.n / 2) * t);
        cuda_ok(cudaMemcpy(out.data(), d.p, out.size() * 2, cudaMemcpyDeviceToHost), "out");
    } else {
        // Three uneven segments (the attention/GDN parents split the same way).
        const int a = h.n / 4, b = h.n / 8 * 3;
        const int rows[3] = {a, b, h.n - a - b};
        DeviceBytes d0(static_cast<std::size_t>(rows[0]) * t * 2),
            d1(static_cast<std::size_t>(rows[1]) * t * 2),
            d2(static_cast<std::size_t>(rows[2]) * t * 2);
        detail::Q3RowSegments seg;
        seg.count   = 3;
        seg.data[0] = static_cast<__nv_bfloat16*>(d0.p);
        seg.data[1] = static_cast<__nv_bfloat16*>(d1.p);
        seg.data[2] = static_cast<__nv_bfloat16*>(d2.p);
        int end     = 0;
        for (int s = 0; s < 3; ++s) {
            end += rows[s];
            seg.ld[s]  = rows[s];
            seg.end[s] = end;
        }
        detail::q3_linear_segments_dispatch(tx, w, seg, nullptr, q3ws);
        cuda_ok(cudaDeviceSynchronize(), "call");
        const DeviceBytes* ds[3] = {&d0, &d1, &d2};
        for (int s = 0; s < 3; ++s) {
            std::vector<std::uint16_t> part(static_cast<std::size_t>(rows[s]) * t);
            cuda_ok(cudaMemcpy(part.data(), ds[s]->p, part.size() * 2, cudaMemcpyDeviceToHost),
                    "out");
            out.insert(out.end(), part.begin(), part.end());
        }
    }
    detail::q3_force_route_for_test(0);
    return out;
}

const char* kind_name(Kind k) {
    switch (k) {
    case Kind::Linear:
        return "linear";
    case Kind::Add:
        return "linear_add";
    case Kind::Swiglu:
        return "linear_swiglu";
    default:
        return "segments";
    }
}

void check(Kind kind, const HostQ3& h, const DeviceBytes& dw, int t, bool emulate) {
    const auto x        = make_x(h.k, t, 7000u + t);
    const auto residual = make_x(h.n, t, 8000u + t);
    DeviceBytes dx(x.size() * 2);
    cuda_ok(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    const std::size_t need = ws_bytes(h.k, t);
    DeviceBytes ws(need + 256);
    const std::string tag = std::string(kind_name(kind)) + " [" + std::to_string(h.n) + "," +
                            std::to_string(h.k) + "] T=" + std::to_string(t);
    if (t < 64) {
        const auto a16 = call(kind, h, dw, dx, residual, t, 1, nullptr, 0);
        const auto r3  = call(kind, h, dw, dx, residual, t, 3, &ws, need);
        report(tag + " T<64 stays A16", a16 == r3 ? 0 : 1, 1);
        return;
    }
    const auto r2  = call(kind, h, dw, dx, residual, t, 2, nullptr, 0);
    const auto r3  = call(kind, h, dw, dx, residual, t, 3, &ws, need);
    const auto r3b = call(kind, h, dw, dx, residual, t, 3, &ws, need);
    int diff       = 0;
    for (std::size_t i = 0; i < r2.size(); ++i) diff += r2[i] != r3[i];
    report(tag + " tall == fused", diff, static_cast<int>(r2.size()));
    report(tag + " tall determinism", r3 == r3b ? 0 : 1, 1);
    if (need == 0) report(tag + " workspace size", 1, 1);
    if (!emulate || kind == Kind::Segments) return;
    const auto e = emulate_a8(h, x, t);
    int bad      = 0;
    if (kind == Kind::Linear) {
        for (std::size_t i = 0; i < r3.size(); ++i) bad += r3[i] != bf16_bits(e[i]);
    } else if (kind == Kind::Add) {
        for (std::size_t i = 0; i < r3.size(); ++i)
            bad += r3[i] != bf16_bits(bf16_value(residual[i]) + e[i]);
    } else {
        const int half = h.n / 2;
        for (int tok = 0; tok < t; ++tok)
            for (int r = 0; r < half; ++r) {
                const double g   = e[static_cast<std::size_t>(tok) * h.n + r];
                const double u   = e[static_cast<std::size_t>(tok) * h.n + r + half];
                const double ref = g / (1.0 + std::exp(-g)) * u;
                const double got = bf16_value(r3[static_cast<std::size_t>(tok) * half + r]);
                if (!std::isfinite(got) ||
                    std::fabs(got - ref) > std::ldexp(std::fabs(ref), -8) + 1e-6)
                    ++bad;
            }
    }
    report(tag + " tall == host A8 emulation", bad, static_cast<int>(r3.size()));
}

void bench() {
    struct B {
        const char* name;
        Kind kind;
        int n, k;
    };

    const std::vector<B> shapes{
        {"gate|up 1gpu", Kind::Swiglu, 34816, 5120},
        {"down 1gpu", Kind::Add, 5120, 17408},
        {"attn qkvg 1gpu", Kind::Segments, 14336, 5120},
        {"gdn qkvz 1gpu", Kind::Segments, 16384, 5120},
        {"out 1gpu", Kind::Add, 5120, 6144},
        {"gate|up tp2", Kind::Swiglu, 17408, 5120},
        {"down tp2", Kind::Add, 5120, 8704},
        {"gdn qkvz tp2", Kind::Segments, 8192, 5120},
        {"out tp2", Kind::Add, 5120, 3072},
    };
    std::printf("\nbench (us per call; TOPS = 2*N*K*T)\n%-16s %6s %10s %10s %10s %8s %8s %8s\n",
                "shape", "T", "A16", "A8fused", "A8tall", "TOPS16", "TOPS8f", "TOPS8t");
    for (const auto& s : shapes) {
        const HostQ3 h = make_q3(s.n, s.k, 99u + s.n);
        DeviceBytes dw(h.payload.size());
        cuda_ok(cudaMemcpy(dw.p, h.payload.data(), h.payload.size(), cudaMemcpyHostToDevice), "w");
        const Weight w = device_weight(h, dw);
        for (int t : {512, 2048, 4096}) {
            const auto x = make_x(s.k, t, 5u);
            DeviceBytes dx(x.size() * 2), dout(static_cast<std::size_t>(s.n) * t * 2);
            cuda_ok(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
            cuda_ok(cudaMemset(dout.p, 0, static_cast<std::size_t>(s.n) * t * 2), "o");
            const std::size_t need = ws_bytes(s.k, t);
            DeviceBytes ws(need + 256);
            Tensor tx(dx.p, DType::BF16, {s.k, t});
            double us[3];
            for (int route = 1; route <= 3; ++route) {
                detail::q3_force_route_for_test(route);
                const detail::Q3Workspace q3ws{ws.p, need};
                auto once = [&] {
                    if (s.kind == Kind::Swiglu) {
                        Tensor to(dout.p, DType::BF16, {s.n / 2, t});
                        detail::q3_linear_swiglu_dispatch(tx, w, to, nullptr, q3ws);
                    } else if (s.kind == Kind::Add) {
                        Tensor to(dout.p, DType::BF16, {s.n, t});
                        detail::q3_linear_add_dispatch(tx, w, to, nullptr, q3ws);
                    } else {
                        Tensor to(dout.p, DType::BF16, {s.n, t});
                        detail::q3_linear_dispatch(tx, w, to, nullptr, q3ws);
                    }
                };
                once();
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                const int reps = 5;
                cudaEventRecord(e0);
                for (int i = 0; i < reps; ++i) once();
                cudaEventRecord(e1);
                cuda_ok(cudaEventSynchronize(e1), "bench");
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                us[route - 1] = 1000.0 * ms / reps;
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
            }
            detail::q3_force_route_for_test(0);
            const double ops = 2.0 * s.n * s.k * t;
            std::printf("%-16s %6d %10.1f %10.1f %10.1f %8.1f %8.1f %8.1f\n", s.name, t, us[0],
                        us[1], us[2], ops / us[0] * 1e-6, ops / us[1] * 1e-6, ops / us[2] * 1e-6);
        }
    }
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    const char* only_bench = std::getenv("NINFER_Q3_BENCH");
    if (only_bench != nullptr && only_bench[0] == '2') {
        bench();
        return 0;
    }

    struct Shape {
        Kind kind;
        int n, k;
    };

    // 27B Text sites, their TP2 shards and a small odd shape (partial row tiles).
    const std::vector<Shape> shapes{
        {Kind::Linear, 328, 2560},    {Kind::Add, 328, 2560},        {Kind::Swiglu, 328, 2560},
        {Kind::Segments, 328, 2560},  {Kind::Linear, 5120, 6144},    {Kind::Add, 5120, 6144},
        {Kind::Add, 5120, 17408},     {Kind::Add, 5120, 8704},       {Kind::Swiglu, 34816, 5120},
        {Kind::Swiglu, 17408, 5120},  {Kind::Segments, 14336, 5120}, {Kind::Segments, 16384, 5120},
        {Kind::Segments, 8192, 5120}, {Kind::Add, 5120, 3072},
    };
    // Default policy: with NINFER_Q3_A8 unset the prefill widths take A8 (workspace requested),
    // T < 64 none.
    if (std::getenv("NINFER_Q3_A8") == nullptr) {
        report("default: A8 workspace requested at T=2048",
               detail::q3_workspace_bytes(5120, 2048) > 0 ? 0 : 1, 1);
        report("default: no workspace at T=63", detail::q3_workspace_bytes(5120, 63) == 0 ? 0 : 1,
               1);
    }
    for (const auto& s : shapes) {
        const HostQ3 h = make_q3(s.n, s.k, 11u * s.n + s.k);
        DeviceBytes dw(h.payload.size());
        cuda_ok(cudaMemcpy(dw.p, h.payload.data(), h.payload.size(), cudaMemcpyHostToDevice), "w");
        const double macs = static_cast<double>(s.n) * s.k;
        for (int t : {9, 64, 65, 127, 128, 129, 200, 2048}) {
            // Host emulation where it stays cheap (<= ~3e9 MACs).
            check(s.kind, h, dw, t, macs * t <= 3e9);
        }
    }
    if (only_bench != nullptr && only_bench[0] == '1') bench();
    std::printf("%s (%d failing checks)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
