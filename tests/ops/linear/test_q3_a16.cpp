// GPU conformance of the q3_g128_fp16 A16 routes against an FP64 host oracle; exits 77 without a
// device.
//
// Routes: Linear through the public ops::linear entry (register-direct tensor cores T<=8,
// tensor-core tiles T>8), and the residual-add, SwiGLU and row-segment epilogues (the routes
// linear_add, linear_swiglu, attn_input_proj and gdn_input_proj take for a Q3 weight). Shapes =
// the Qwen3.8-27B Text sites ([5120,6144], [5120,17408], [34816,5120] SwiGLU, [14336,5120]
// Q|K|Gate|V, [16384,5120] QKV|Z) plus a small odd one.
//
// Pre-registered criterion, per output element o (FP64 reference r, L1 magnitude m = sum|w*x|):
//     |o - r| <= 2^-8 * |r| + 2^-14 * m        (Linear, segments)
//     |o - r| <= 2^-8 * |r| + 2^-14 * m + ulp_bf16(residual)   (residual add: one rounding)
//     SwiGLU: the oracle evaluates silu(g) * u in FP64 from FP64 g, u; same bound with m taken as
//             |silu'|-weighted: m = (|u| + |g|) * max(m_g, m_u) (loose; documents intent).
// BF16 output rounding alone is 2^-9 relative; 2^-14 * m absorbs FP32 accumulation and the
// cancellation of near-zero outputs. Any NaN/Inf fails. Determinism: two launches of the same call
// are bitwise identical (fixed reduction order).
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

struct Oracle {
    std::vector<double> value, magnitude; // [t][n]
};

Oracle oracle(const HostQ3& h, const std::vector<std::uint16_t>& x, int t) {
    Oracle o;
    o.value.assign(static_cast<std::size_t>(t) * h.n, 0.0);
    o.magnitude.assign(o.value.size(), 0.0);
    for (int tok = 0; tok < t; ++tok) {
        for (int r = 0; r < h.n; ++r) {
            double s = 0.0, m = 0.0;
            for (int c = 0; c < h.k; ++c) {
                const double p = h.w(r, c) * bf16_value(x[static_cast<std::size_t>(tok) * h.k + c]);
                s += p;
                m += std::fabs(p);
            }
            o.value[static_cast<std::size_t>(tok) * h.n + r]     = s;
            o.magnitude[static_cast<std::size_t>(tok) * h.n + r] = m;
        }
    }
    return o;
}

bool within(double got, double ref, double magnitude, double extra = 0.0) {
    if (!std::isfinite(got)) return false;
    return std::fabs(got - ref) <=
           std::ldexp(std::fabs(ref), -8) + std::ldexp(magnitude, -14) + extra;
}

void report(const std::string& name, int bad, int total) {
    std::printf("%-58s %s (%d/%d out of bound)\n", name.c_str(), bad ? "FAIL" : "ok", bad, total);
    if (bad) ++failures;
}

void run_linear(const HostQ3& h, const DeviceBytes& dw, int t) {
    const auto x = make_x(h.k, t, 1000u + t);
    DeviceBytes dx(x.size() * 2), dout(static_cast<std::size_t>(h.n) * t * 2);
    cuda_ok(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    Tensor tx(dx.p, DType::BF16, {h.k, t});
    Tensor tout(dout.p, DType::BF16, {h.n, t});
    const Weight w = device_weight(h, dw);
    ops::linear(tx, w, tout, nullptr);
    cuda_ok(cudaDeviceSynchronize(), "linear");
    std::vector<std::uint16_t> out(static_cast<std::size_t>(h.n) * t), again(out.size());
    cuda_ok(cudaMemcpy(out.data(), dout.p, out.size() * 2, cudaMemcpyDeviceToHost), "out");
    ops::linear(tx, w, tout, nullptr);
    cuda_ok(cudaMemcpy(again.data(), dout.p, again.size() * 2, cudaMemcpyDeviceToHost), "again");
    const Oracle o = oracle(h, x, t);
    int bad        = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (!within(bf16_value(out[i]), o.value[i], o.magnitude[i])) ++bad;
    }
    report("linear [" + std::to_string(h.n) + "," + std::to_string(h.k) +
               "] T=" + std::to_string(t),
           bad, static_cast<int>(out.size()));
    report("linear determinism T=" + std::to_string(t), out == again ? 0 : 1, 1);
}

void run_add(const HostQ3& h, const DeviceBytes& dw, int t) {
    const auto x        = make_x(h.k, t, 2000u + t);
    const auto residual = make_x(h.n, t, 3000u + t);
    DeviceBytes dx(x.size() * 2), dr(residual.size() * 2);
    cuda_ok(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    cuda_ok(cudaMemcpy(dr.p, residual.data(), residual.size() * 2, cudaMemcpyHostToDevice), "r");
    Tensor tx(dx.p, DType::BF16, {h.k, t});
    Tensor tr(dr.p, DType::BF16, {h.n, t});
    const Weight w = device_weight(h, dw);
    detail::q3_linear_add_dispatch(tx, w, tr, nullptr);
    cuda_ok(cudaDeviceSynchronize(), "add");
    std::vector<std::uint16_t> out(residual.size());
    cuda_ok(cudaMemcpy(out.data(), dr.p, out.size() * 2, cudaMemcpyDeviceToHost), "out");
    const Oracle o = oracle(h, x, t);
    int bad        = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double ref = o.value[i] + bf16_value(residual[i]);
        if (!within(bf16_value(out[i]), ref, o.magnitude[i])) ++bad;
    }
    report("linear_add [" + std::to_string(h.n) + "," + std::to_string(h.k) +
               "] T=" + std::to_string(t),
           bad, static_cast<int>(out.size()));
}

void run_swiglu(const HostQ3& h, const DeviceBytes& dw, int t) {
    const int half = h.n / 2;
    const auto x   = make_x(h.k, t, 4000u + t);
    DeviceBytes dx(x.size() * 2), dout(static_cast<std::size_t>(half) * t * 2);
    cuda_ok(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    Tensor tx(dx.p, DType::BF16, {h.k, t});
    Tensor tout(dout.p, DType::BF16, {half, t});
    const Weight w = device_weight(h, dw);
    detail::q3_linear_swiglu_dispatch(tx, w, tout, nullptr);
    cuda_ok(cudaDeviceSynchronize(), "swiglu");
    std::vector<std::uint16_t> out(static_cast<std::size_t>(half) * t);
    cuda_ok(cudaMemcpy(out.data(), dout.p, out.size() * 2, cudaMemcpyDeviceToHost), "out");
    const Oracle o = oracle(h, x, t);
    int bad        = 0;
    for (int tok = 0; tok < t; ++tok) {
        for (int r = 0; r < half; ++r) {
            const double g   = o.value[static_cast<std::size_t>(tok) * h.n + r];
            const double u   = o.value[static_cast<std::size_t>(tok) * h.n + r + half];
            const double mg  = o.magnitude[static_cast<std::size_t>(tok) * h.n + r];
            const double mu  = o.magnitude[static_cast<std::size_t>(tok) * h.n + r + half];
            const double ref = g / (1.0 + std::exp(-g)) * u;
            const double m   = (std::fabs(u) + std::fabs(g)) * std::fmax(mg, mu);
            if (!within(bf16_value(out[static_cast<std::size_t>(tok) * half + r]), ref, m)) ++bad;
        }
    }
    report("linear_swiglu [" + std::to_string(h.n) + "," + std::to_string(h.k) +
               "] T=" + std::to_string(t),
           bad, half * t);
}

void run_segments(const HostQ3& h, const DeviceBytes& dw, int t, std::vector<int> rows) {
    const auto x = make_x(h.k, t, 5000u + t);
    DeviceBytes dx(x.size() * 2);
    cuda_ok(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    std::vector<DeviceBytes*> outs;
    detail::Q3RowSegments seg;
    seg.count = static_cast<int>(rows.size());
    int end   = 0;
    for (std::size_t s = 0; s < rows.size(); ++s) {
        outs.push_back(new DeviceBytes(static_cast<std::size_t>(rows[s]) * t * 2));
        end += rows[s];
        seg.data[s] = static_cast<__nv_bfloat16*>(outs.back()->p);
        seg.ld[s]   = rows[s];
        seg.end[s]  = end;
    }
    Tensor tx(dx.p, DType::BF16, {h.k, t});
    const Weight w = device_weight(h, dw);
    detail::q3_linear_segments_dispatch(tx, w, seg, nullptr);
    cuda_ok(cudaDeviceSynchronize(), "segments");
    const Oracle o = oracle(h, x, t);
    int bad = 0, total = 0, begin = 0;
    for (std::size_t s = 0; s < rows.size(); ++s) {
        std::vector<std::uint16_t> out(static_cast<std::size_t>(rows[s]) * t);
        cuda_ok(cudaMemcpy(out.data(), outs[s]->p, out.size() * 2, cudaMemcpyDeviceToHost), "out");
        for (int tok = 0; tok < t; ++tok) {
            for (int r = 0; r < rows[s]; ++r) {
                const std::size_t i = static_cast<std::size_t>(tok) * h.n + begin + r;
                if (!within(bf16_value(out[static_cast<std::size_t>(tok) * rows[s] + r]),
                            o.value[i], o.magnitude[i])) {
                    ++bad;
                }
                ++total;
            }
        }
        begin += rows[s];
        delete outs[s];
    }
    report("segments [" + std::to_string(h.n) + "," + std::to_string(h.k) + "] x" +
               std::to_string(rows.size()) + " T=" + std::to_string(t),
           bad, total);
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    // The A16 routes, whatever NINFER_Q3_A8 says (the A8 prefill route is
    // ninfer_linear_q3_a8_test).
    detail::q3_force_route_for_test(1);
    // Token counts: every small-T width, the small-T/tile boundary, partial and multiple tiles.
    const std::vector<int> tokens{1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 63, 64, 65, 130};

    struct Shape {
        int n, k;
    };

    // The small odd shape exercises partial row tiles (N % 128 != 0).
    const std::vector<Shape> linear_shapes{{328, 2560}, {5120, 6144}, {5120, 17408}};
    for (const auto& s : linear_shapes) {
        const HostQ3 h = make_q3(s.n, s.k, 7u * s.n + s.k);
        DeviceBytes dw(h.payload.size());
        cuda_ok(cudaMemcpy(dw.p, h.payload.data(), h.payload.size(), cudaMemcpyHostToDevice), "w");
        for (int t : tokens) run_linear(h, dw, t);
        for (int t : {1, 4, 9, 64}) run_add(h, dw, t);
    }
    {
        const HostQ3 h = make_q3(34816, 5120, 34816u);
        DeviceBytes dw(h.payload.size());
        cuda_ok(cudaMemcpy(dw.p, h.payload.data(), h.payload.size(), cudaMemcpyHostToDevice), "w");
        for (int t : {1, 4, 8, 9, 64}) run_swiglu(h, dw, t);
    }
    {
        const HostQ3 h = make_q3(14336, 5120, 14336u);
        DeviceBytes dw(h.payload.size());
        cuda_ok(cudaMemcpy(dw.p, h.payload.data(), h.payload.size(), cudaMemcpyHostToDevice), "w");
        for (int t : {1, 4, 9, 64}) run_segments(h, dw, t, {6144, 1024, 6144, 1024});
    }
    {
        const HostQ3 h = make_q3(16384, 5120, 16384u);
        DeviceBytes dw(h.payload.size());
        cuda_ok(cudaMemcpy(dw.p, h.payload.data(), h.payload.size(), cudaMemcpyHostToDevice), "w");
        for (int t : {1, 4, 9, 64}) run_segments(h, dw, t, {10240, 6144});
    }
    std::printf("%s (%d failing checks)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
