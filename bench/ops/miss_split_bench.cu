// Probe: the GPU share of a layer's missing experts read over PCIe, against the CPU executor,
// alone and concurrently, on a synthetic pinned host bank in the Qwen4Exp layout (the engine's
// bank, read in place).
//
// GPU share modes (one call = n distinct experts, T columns):
//   staged  a streaming SM copy kernel moves the n experts' four planes into VRAM scratch slots,
//           then moe_slots_nvfp4 over the scratch slots (the copy reads each byte once, many loads
//           in flight)
//   dma     cudaMemcpyAsync of the four planes per expert into the scratch slots (host-issued;
//   reference only)
// Conditions: idle | cpu (the CPU executor runs continuously on other experts, 6 threads, as the
// CPU share would) | fill<G> (a DMA host->device copy stream at G GB/s, the cache fills) |
// combinations. Arms:
//   --arm gpu    per-call GPU time of each mode, n = 1..12, under the conditions
//   --arm cpu    per-call CPU executor time, n = 1..12, idle / with GPU reads running / with fills
//   --arm joint  n missing experts split g on the GPU, n - g on the CPU, both started together: CPU
//   end, GPU end
//                and their max (the layer's MoE stage), for every g, with the fill stream on
// One JSON object per line.
#include "ninfer/ops/moe_slots.h"
#include "ops/sparse_moe/host/host_moe_executor.h"

#include "core/arena.h"
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops::host;

namespace {

constexpr int H = ops::kMoeSlotsHidden, I = ops::kMoeSlotsIntermediate, K = ops::kMoeSlotsTopK,
              E                    = 512;
constexpr int kMaxN                = 16;
constexpr double kExpertBytes      = 2764800.0;
constexpr long long kPlaneBytes[4] = {ops::kMoeSlotsGateUpCodeBytes, ops::kMoeSlotsGateUpScaleBytes,
                                      ops::kMoeSlotsDownCodeBytes, ops::kMoeSlotsDownScaleBytes};

double now_us() {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct Args {
    std::string arm = "gpu";
    std::vector<int> ns{1, 2, 3, 4, 6, 8, 12};
    std::vector<std::string> modes{"staged"};
    std::vector<std::string> conds{"idle", "cpu", "fill8", "cpu+fill8"};
    int tokens = 4, reps = 60, threads = 6, stage_blocks = 64;
    double p2 = 0.47;
};

std::vector<int> ints(const std::string& t) {
    std::vector<int> v;
    std::size_t s = 0;
    while (s < t.size()) {
        auto e = t.find(',', s);
        v.push_back(std::stoi(t.substr(s, e - s)));
        if (e == std::string::npos) break;
        s = e + 1;
    }
    return v;
}

std::vector<std::string> strs(const std::string& t) {
    std::vector<std::string> v;
    std::size_t s = 0;
    while (s < t.size()) {
        auto e = t.find(',', s);
        v.push_back(t.substr(s, e - s));
        if (e == std::string::npos) break;
        s = e + 1;
    }
    return v;
}

// ------------------------------------------------------------------ pinned, mapped host bank (one
// layer)
struct Bank {
    std::byte* base = nullptr;
    HostExpertLayer layer;
    ops::MoeSlotPool pool;    // the bank itself as a slot pool (slot = expert id)
    float* d_div13 = nullptr; // device copies of the divisors
    float* d_div2  = nullptr;
};

Bank make_bank() {
    Bank b;
    const std::int64_t gu_rows = std::int64_t{E} * 2 * I, dn_rows = std::int64_t{E} * H;
    const std::size_t gu    = Nvfp4Plane::payload_bytes(gu_rows, H, 2 * E);
    const std::size_t dn    = Nvfp4Plane::payload_bytes(dn_rows, I, E);
    const std::size_t gu_al = (gu + 4095) / 4096 * 4096;
    const std::size_t bytes = gu_al + (dn + 4095) / 4096 * 4096;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&b.base), bytes,
                             cudaHostAllocMapped | cudaHostAllocPortable));
    std::vector<std::thread> th;
    const int nt = 8;
    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            std::uint64_t s         = 0x9E3779B97F4A7C15ULL * (t + 1);
            const std::size_t words = bytes / 8, lo = words * t / nt, hi = words * (t + 1) / nt;
            auto* w = reinterpret_cast<std::uint64_t*>(b.base);
            for (std::size_t i = lo; i < hi; ++i) {
                s ^= s << 13, s ^= s >> 7, s ^= s << 17;
                w[i] = s;
            }
        });
    }
    for (auto& t : th) t.join();
    b.layer  = {Nvfp4Plane::from_payload(b.base, gu_rows, H, 2 * E),
                Nvfp4Plane::from_payload(b.base + gu_al, dn_rows, I, E), E, H, I};
    auto fix = [](const Nvfp4Plane& p, std::int64_t divisors) {
        auto* sc            = const_cast<std::uint8_t*>(p.scales);
        const std::size_t n = static_cast<std::size_t>(p.rows) * p.k / 16;
        for (std::size_t i = 0; i < n; ++i)
            sc[i] = static_cast<std::uint8_t>(0x30 | (sc[i] & 0x0F));
        auto* d = const_cast<float*>(p.divisors);
        for (std::int64_t i = 0; i < divisors; ++i) d[i] = 1000.0F;
    };
    fix(b.layer.gate_up, 2 * E);
    fix(b.layer.down, E);
    b.pool.gate_up_codes  = {b.layer.gate_up.codes, kPlaneBytes[0]};
    b.pool.gate_up_scales = {b.layer.gate_up.scales, kPlaneBytes[1]};
    b.pool.down_codes     = {b.layer.down.codes, kPlaneBytes[2]};
    b.pool.down_scales    = {b.layer.down.scales, kPlaneBytes[3]};
    b.pool.slots          = E;
    CUDA_CHECK(cudaMalloc(&b.d_div13, sizeof(float) * 2 * E));
    CUDA_CHECK(cudaMalloc(&b.d_div2, sizeof(float) * E));
    CUDA_CHECK(cudaMemcpy(b.d_div13, b.layer.gate_up.divisors, sizeof(float) * 2 * E,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(b.d_div2, b.layer.down.divisors, sizeof(float) * E, cudaMemcpyHostToDevice));
    return b;
}

// ------------------------------------------------------------------ staging copy kernel
struct StageArgs {
    const uint4* src[4];
    uint4* dst[4];
    const int* picks; // [n] expert ids
    const int* count; // n
};

constexpr long long kV0 = kPlaneBytes[0] / 16, kV1 = kPlaneBytes[1] / 16, kV2 = kPlaneBytes[2] / 16,
                    kV3 = kPlaneBytes[3] / 16,
                    kVE = kV0 + kV1 + kV2 + kV3; // 172 800 uint4 per expert

__device__ __forceinline__ void locate(long long i, int& plane, long long& off) {
    if (i < kV0) {
        plane = 0;
        off   = i;
    } else if (i < kV0 + kV1) {
        plane = 1;
        off   = i - kV0;
    } else if (i < kV0 + kV1 + kV2) {
        plane = 2;
        off   = i - kV0 - kV1;
    } else {
        plane = 3;
        off   = i - kV0 - kV1 - kV2;
    }
}

__global__ void stage_kernel(StageArgs a) {
    const int n               = *a.count;
    const long long total     = static_cast<long long>(n) * kVE;
    const long long stride    = static_cast<long long>(gridDim.x) * blockDim.x;
    constexpr int U           = 4;
    const long long vplane[4] = {kV0, kV1, kV2, kV3};
    for (long long base = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         base < total; base += stride * U) {
        uint4 v[U];
        int pl[U], k[U];
        long long off[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const long long i = base + u * stride;
            k[u]              = -1;
            if (i < total) {
                k[u] = static_cast<int>(i / kVE);
                locate(i - static_cast<long long>(k[u]) * kVE, pl[u], off[u]);
                v[u] = a.src[pl[u]][static_cast<long long>(a.picks[k[u]]) * vplane[pl[u]] + off[u]];
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (k[u] >= 0)
                a.dst[pl[u]][static_cast<long long>(k[u]) * vplane[pl[u]] + off[u]] = v[u];
        }
    }
}

// ------------------------------------------------------------------ one call's data
struct Call {
    std::vector<int> experts;     // n distinct ids
    std::vector<int> ids;         // [T][K], only the n experts, -1 elsewhere
    std::vector<int> cols;        // columns per expert
    int* d_ids         = nullptr; // device copy of ids
    int* d_picks       = nullptr; // device [kMaxN]
    int* d_count       = nullptr; // device n
    int* d_stage_table = nullptr; // device [E]: expert -> scratch slot k, else -1
};

Call make_call(int n, int T, double p2, std::mt19937_64& rng, std::vector<int>& cursor_pool,
               std::size_t& cursor) {
    Call c;
    c.ids.assign(static_cast<std::size_t>(T) * K, -1);
    std::uniform_int_distribution<int> tok(0, T - 1);
    std::bernoulli_distribution two(p2);
    std::vector<int> fill(T, 0);
    for (int i = 0; i < n; ++i) {
        if (cursor >= cursor_pool.size()) {
            std::shuffle(cursor_pool.begin(), cursor_pool.end(), rng);
            cursor = 0;
        }
        const int e = cursor_pool[cursor++];
        c.experts.push_back(e);
        const int m  = (two(rng) && T > 1) ? 2 : 1;
        int placed   = 0;
        const int t0 = tok(rng);
        for (int j = 0; j < T && placed < m; ++j) {
            const int t = (t0 + j) % T;
            if (fill[t] < K) c.ids[t * K + fill[t]++] = e, ++placed;
        }
        c.cols.push_back(placed);
    }
    CUDA_CHECK(cudaMalloc(&c.d_ids, sizeof(int) * T * K));
    CUDA_CHECK(cudaMemcpy(c.d_ids, c.ids.data(), sizeof(int) * T * K, cudaMemcpyHostToDevice));
    std::vector<int> picks(kMaxN, 0);
    for (int i = 0; i < n; ++i) picks[i] = c.experts[i];
    CUDA_CHECK(cudaMalloc(&c.d_picks, sizeof(int) * kMaxN));
    CUDA_CHECK(cudaMemcpy(c.d_picks, picks.data(), sizeof(int) * kMaxN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&c.d_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(c.d_count, &n, sizeof(int), cudaMemcpyHostToDevice));
    std::vector<int> table(E, -1);
    for (int i = 0; i < n; ++i) table[c.experts[i]] = i;
    CUDA_CHECK(cudaMalloc(&c.d_stage_table, sizeof(int) * E));
    CUDA_CHECK(cudaMemcpy(c.d_stage_table, table.data(), sizeof(int) * E, cudaMemcpyHostToDevice));
    return c;
}

void free_call(Call& c) {
    for (int* p : {c.d_ids, c.d_picks, c.d_count, c.d_stage_table}) cudaFree(p);
}

// ------------------------------------------------------------------ GPU share
struct Gpu {
    Bank* bank = nullptr;
    std::byte* scratch[4]{};
    ops::MoeSlotPool scratch_pool;
    __nv_bfloat16* d_x   = nullptr;
    float* d_w           = nullptr;
    float* d_out         = nullptr;
    WorkspaceArena* work = nullptr;
    int stage_blocks     = 64;

    void init(Bank& b, int blocks) {
        bank         = &b;
        stage_blocks = blocks;
        for (int q = 0; q < 4; ++q) CUDA_CHECK(cudaMalloc(&scratch[q], kPlaneBytes[q] * kMaxN));
        scratch_pool.gate_up_codes  = {scratch[0], kPlaneBytes[0]};
        scratch_pool.gate_up_scales = {scratch[1], kPlaneBytes[1]};
        scratch_pool.down_codes     = {scratch[2], kPlaneBytes[2]};
        scratch_pool.down_scales    = {scratch[3], kPlaneBytes[3]};
        scratch_pool.slots          = kMaxN;
        std::vector<std::uint16_t> x(static_cast<std::size_t>(8) * H);
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        for (auto& v : x) {
            float f = nd(rng);
            std::uint32_t u;
            std::memcpy(&u, &f, 4);
            v = static_cast<std::uint16_t>(u >> 16);
        }
        CUDA_CHECK(cudaMalloc(&d_x, x.size() * 2));
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        std::vector<float> w(8 * K, 0.1F);
        CUDA_CHECK(cudaMalloc(&d_w, w.size() * 4));
        CUDA_CHECK(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * 8 * H));
        work = new WorkspaceArena(ops::moe_slots_nvfp4_workspace_bytes(8));
    }

    void compute(const Call& c, int T, cudaStream_t s) {
        ops::MoeSlotsLayer layer;
        layer.slot_table            = Tensor(c.d_stage_table, DType::I32, {E});
        layer.gate_up_divisors      = Tensor(bank->d_div13, DType::FP32, {2 * E});
        layer.down_divisors         = Tensor(bank->d_div2, DType::FP32, {E});
        layer.gate_up_input_divisor = 600.0F;
        layer.down_input_divisor    = 2000.0F;
        Tensor tx(d_x, DType::BF16, {H, T});
        Tensor ti(c.d_ids, DType::I32, {K, T});
        Tensor tw(d_w, DType::FP32, {K, T});
        Tensor to(d_out, DType::FP32, {H, T});
        ops::moe_slots_nvfp4(tx, ti, tw, layer, scratch_pool, ops::MoeSlotsOutput::Store, to, *work,
                             s);
    }

    // Enqueue one call's GPU share on s.
    void enqueue(const std::string& mode, const Call& c, int T, cudaStream_t s) {
        if (mode == "staged") {
            StageArgs a;
            const void* src[4] = {bank->pool.gate_up_codes.base, bank->pool.gate_up_scales.base,
                                  bank->pool.down_codes.base, bank->pool.down_scales.base};
            for (int q = 0; q < 4; ++q) {
                a.src[q] = static_cast<const uint4*>(src[q]);
                a.dst[q] = reinterpret_cast<uint4*>(scratch[q]);
            }
            a.picks = c.d_picks;
            a.count = c.d_count;
            stage_kernel<<<stage_blocks, 256, 0, s>>>(a);
            CUDA_CHECK(cudaGetLastError());
            compute(c, T, s);
        } else if (mode == "dma") {
            const std::byte* src[4] = {
                static_cast<const std::byte*>(bank->pool.gate_up_codes.base),
                static_cast<const std::byte*>(bank->pool.gate_up_scales.base),
                static_cast<const std::byte*>(bank->pool.down_codes.base),
                static_cast<const std::byte*>(bank->pool.down_scales.base)};
            for (std::size_t k = 0; k < c.experts.size(); ++k)
                for (int q = 0; q < 4; ++q)
                    CUDA_CHECK(cudaMemcpyAsync(scratch[q] + k * kPlaneBytes[q],
                                               src[q] + c.experts[k] * kPlaneBytes[q],
                                               kPlaneBytes[q], cudaMemcpyHostToDevice, s));
            compute(c, T, s);
        } else {
            std::fprintf(stderr, "bad mode %s\n", mode.c_str());
            std::exit(2);
        }
    }
};

// ------------------------------------------------------------------ background loads
struct Background {
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    std::atomic<long long> bytes{0};
    std::atomic<long long> calls{0};
    double t0 = 0;

    // DMA host->device at `gbs` (0 = full speed), chunks of one expert.
    void fill(const Bank& b, double gbs) {
        th.emplace_back([this, &b, gbs] {
            cudaStream_t s;
            CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            void* dst               = nullptr;
            const std::size_t chunk = static_cast<std::size_t>(kExpertBytes);
            CUDA_CHECK(cudaMalloc(&dst, 8 * chunk));
            std::size_t off = 0, k = 0;
            const double start = now_us();
            long long sent     = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                if (gbs > 0) {
                    const double due = start + sent / (gbs * 1e3);
                    while (now_us() < due && !stop.load(std::memory_order_relaxed))
                        std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
                if (off + chunk > static_cast<std::size_t>(E) * chunk) off = 0;
                CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst) + (k++ % 8) * chunk,
                                           b.base + off, chunk, cudaMemcpyHostToDevice, s));
                off += chunk;
                sent += chunk;
                if (k % 4 == 0) CUDA_CHECK(cudaStreamSynchronize(s));
            }
            CUDA_CHECK(cudaStreamSynchronize(s));
            bytes += sent;
            CUDA_CHECK(cudaFree(dst));
            CUDA_CHECK(cudaStreamDestroy(s));
        });
    }

    // CPU executor continuously on 4-column calls of 6 experts (the CPU share of a layer).
    void cpu(const Bank& b, int threads, int T) {
        th.emplace_back([this, &b, threads, T] {
            const auto cpus = spread_cpus(threads);
            HostCpuPool pool(cpus);
            HostMoeExecutor ex(pool, H, I, 8, K, {});
            std::mt19937_64 rng(11);
            std::vector<std::uint16_t> x(static_cast<std::size_t>(T) * H, 0x3f80);
            std::vector<float> alpha(static_cast<std::size_t>(T) * K, 0.1F),
                out(static_cast<std::size_t>(T) * H);
            std::uniform_int_distribution<int> ex_id(0, E - 1);
            while (!stop.load(std::memory_order_relaxed)) {
                std::vector<int> ids(static_cast<std::size_t>(T) * K, -1);
                for (int i = 0; i < 6; ++i) ids[(i % T) * K + i / T] = ex_id(rng);
                const HostMoeJob job{T, K, x.data(), ids.data(), alpha.data(), out.data()};
                ex.run(b.layer, job);
                ++calls;
            }
        });
    }

    // GPU share calls in a loop on their own stream.
    void gpu(Gpu& g, const std::string& mode, std::vector<Call>& calls_in, int T) {
        th.emplace_back([this, &g, mode, &calls_in, T] {
            cudaStream_t s;
            CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            std::size_t i = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                g.enqueue(mode, calls_in[i++ % calls_in.size()], T, s);
                CUDA_CHECK(cudaStreamSynchronize(s));
                ++calls;
            }
            CUDA_CHECK(cudaStreamDestroy(s));
        });
    }

    void start() {
        t0 = now_us();
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }

    double finish() {
        stop = true;
        for (auto& t : th) t.join();
        th.clear();
        return bytes / (now_us() - t0) / 1e3; // GB/s of fills
    }
};

struct Stat {
    double med = 0, mean = 0, p10 = 0, p90 = 0;
};

Stat stat(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[static_cast<std::size_t>(p * (v.size() - 1) + 0.5)]; };
    s.med = q(0.5), s.p10 = q(0.1), s.p90 = q(0.9);
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    return s;
}

double cond_fill(const std::string& c) {
    const auto p = c.find("fill");
    if (p == std::string::npos) return -1;
    const std::string v   = c.substr(p + 4);
    const auto plus       = v.find('+');
    const std::string num = v.substr(0, plus);
    return num == "full" ? 0.0 : std::stod(num);
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next           = [&]() {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value\n");
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (k == "--arm")
            a.arm = next();
        else if (k == "--n")
            a.ns = ints(next());
        else if (k == "--modes")
            a.modes = strs(next());
        else if (k == "--conds")
            a.conds = strs(next());
        else if (k == "--tokens")
            a.tokens = std::stoi(next());
        else if (k == "--reps")
            a.reps = std::stoi(next());
        else if (k == "--threads")
            a.threads = std::stoi(next());
        else if (k == "--stage-blocks")
            a.stage_blocks = std::stoi(next());
        else {
            std::fprintf(stderr, "unknown option %s\n", k.c_str());
            return 2;
        }
    }
    if (!host_experts_supported()) {
        std::fprintf(stderr, "needs AVX2 + FMA + F16C\n");
        return 77;
    }
    CUDA_CHECK(cudaSetDevice(0));
    Bank bank = make_bank();
    Gpu gpu;
    gpu.init(bank, a.stage_blocks);
    const int T = a.tokens;
    std::mt19937_64 rng(5);
    std::vector<int> order(E);
    std::iota(order.begin(), order.end(), 0);
    std::size_t cursor = E;
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::printf("{\"arm\":\"%s\",\"tokens\":%d,\"reps\":%d,\"threads\":%d,\"stage_blocks\":%d}\n",
                a.arm.c_str(), T, a.reps, a.threads, a.stage_blocks);
    std::fflush(stdout);

    if (a.arm == "gpu") {
        for (const auto& cond : a.conds) {
            for (const auto& mode : a.modes) {
                for (int n : a.ns) {
                    std::vector<Call> calls;
                    for (int r = 0; r < a.reps; ++r)
                        calls.push_back(make_call(n, T, a.p2, rng, order, cursor));
                    for (int w = 0; w < 3; ++w) gpu.enqueue(mode, calls[w], T, s);
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    Background bg;
                    const double fg = cond_fill(cond);
                    if (fg >= 0) bg.fill(bank, fg);
                    if (cond.find("cpu") != std::string::npos) bg.cpu(bank, a.threads, T);
                    bg.start();
                    std::vector<double> us;
                    for (auto& c : calls) {
                        CUDA_CHECK(cudaEventRecord(e0, s));
                        gpu.enqueue(mode, c, T, s);
                        CUDA_CHECK(cudaEventRecord(e1, s));
                        CUDA_CHECK(cudaEventSynchronize(e1));
                        float ms = 0;
                        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                        us.push_back(1e3 * ms);
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                    }
                    const long long cpu_calls = bg.calls.load();
                    const double fill_gbs     = bg.finish();
                    const Stat st             = stat(us);
                    std::printf("{\"arm\":\"gpu\",\"cond\":\"%s\",\"mode\":\"%s\",\"n\":%d,\"us_"
                                "med\":%.1f,\"us_p10\":%.1f,"
                                "\"us_p90\":%.1f,\"us_per_expert\":%.1f,\"GBps\":%.2f,\"fill_"
                                "GBps\":%.2f,\"bg_calls\":%lld}\n",
                                cond.c_str(), mode.c_str(), n, st.med, st.p10, st.p90, st.med / n,
                                n * kExpertBytes / st.med / 1e3, fill_gbs, cpu_calls);
                    std::fflush(stdout);
                    for (auto& c : calls) free_call(c);
                }
            }
        }
        return 0;
    }

    // CPU arms: this thread is pool member 0.
    const auto cpus = spread_cpus(a.threads);
    HostCpuPool pool(cpus);
    HostMoeExecutor ex(pool, H, I, 8, K, {});
    std::vector<std::uint16_t> x(static_cast<std::size_t>(T) * H);
    {
        std::mt19937 r(9);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        for (auto& v : x) {
            float f = nd(r);
            std::uint32_t u;
            std::memcpy(&u, &f, 4);
            v = static_cast<std::uint16_t>(u >> 16);
        }
    }
    std::vector<float> alpha(static_cast<std::size_t>(T) * K, 0.1F),
        out(static_cast<std::size_t>(T) * H);
    auto cpu_run = [&](const std::vector<int>& ids) {
        const HostMoeJob job{T, K, x.data(), ids.data(), alpha.data(), out.data()};
        const double t0 = now_us();
        ex.run(bank.layer, job);
        return now_us() - t0;
    };

    if (a.arm == "cpu") {
        // conds: idle | fill8 | staged | staged+fill8
        for (const auto& cond : a.conds) {
            for (int n : a.ns) {
                std::vector<Call> calls, bgcalls;
                for (int r = 0; r < a.reps; ++r)
                    calls.push_back(make_call(n, T, a.p2, rng, order, cursor));
                for (int r = 0; r < 16; ++r)
                    bgcalls.push_back(make_call(3, T, a.p2, rng, order, cursor));
                Background bg;
                const double fg = cond_fill(cond);
                if (fg >= 0) bg.fill(bank, fg);
                if (cond.rfind("staged", 0) == 0) bg.gpu(gpu, "staged", bgcalls, T);
                bg.start();
                for (int w = 0; w < 3; ++w) cpu_run(calls[w].ids);
                std::vector<double> us;
                for (auto& c : calls) {
                    us.push_back(cpu_run(c.ids));
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
                const long long gcalls = bg.calls.load();
                const double fill_gbs  = bg.finish();
                const Stat st          = stat(us);
                std::printf("{\"arm\":\"cpu\",\"cond\":\"%s\",\"n\":%d,\"us_med\":%.1f,\"us_p10\":%"
                            ".1f,\"us_p90\":%.1f,"
                            "\"us_per_expert\":%.1f,\"fill_GBps\":%.2f,\"bg_gpu_calls\":%lld}\n",
                            cond.c_str(), n, st.med, st.p10, st.p90, st.med / n, fill_gbs, gcalls);
                std::fflush(stdout);
                for (auto& c : calls) free_call(c);
                for (auto& c : bgcalls) free_call(c);
            }
        }
        return 0;
    }

    if (a.arm == "joint") {
        // n missing experts: g on the GPU (mode), n - g on the CPU, started together; fills per
        // --conds (one).
        for (const auto& cond : a.conds) {
            for (const auto& mode : a.modes) {
                for (int n : a.ns) {
                    for (int g = 0; g <= std::min(n, 8); ++g) {
                        std::vector<Call> gc;
                        std::vector<std::vector<int>> cc;
                        for (int r = 0; r < a.reps; ++r) {
                            Call all = make_call(n, T, a.p2, rng, order, cursor);
                            // first g experts -> GPU call, the rest -> CPU ids
                            std::vector<int> gpu_ids(static_cast<std::size_t>(T) * K, -1),
                                cpu_ids(static_cast<std::size_t>(T) * K, -1);
                            for (int j = 0; j < T * K; ++j) {
                                const int e = all.ids[j];
                                if (e < 0) continue;
                                const int k = static_cast<int>(
                                    std::find(all.experts.begin(), all.experts.end(), e) -
                                    all.experts.begin());
                                (k < g ? gpu_ids : cpu_ids)[j] = e;
                            }
                            free_call(all);
                            Call c;
                            c.ids = gpu_ids;
                            c.experts.assign(all.experts.begin(), all.experts.begin() + g);
                            CUDA_CHECK(cudaMalloc(&c.d_ids, sizeof(int) * T * K));
                            CUDA_CHECK(cudaMemcpy(c.d_ids, gpu_ids.data(), sizeof(int) * T * K,
                                                  cudaMemcpyHostToDevice));
                            std::vector<int> picks(kMaxN, 0);
                            for (int i = 0; i < g; ++i) picks[i] = c.experts[i];
                            CUDA_CHECK(cudaMalloc(&c.d_picks, sizeof(int) * kMaxN));
                            CUDA_CHECK(cudaMemcpy(c.d_picks, picks.data(), sizeof(int) * kMaxN,
                                                  cudaMemcpyHostToDevice));
                            CUDA_CHECK(cudaMalloc(&c.d_count, sizeof(int)));
                            CUDA_CHECK(
                                cudaMemcpy(c.d_count, &g, sizeof(int), cudaMemcpyHostToDevice));
                            std::vector<int> table(E, -1);
                            for (int i = 0; i < g; ++i) table[c.experts[i]] = i;
                            CUDA_CHECK(cudaMalloc(&c.d_stage_table, sizeof(int) * E));
                            CUDA_CHECK(cudaMemcpy(c.d_stage_table, table.data(), sizeof(int) * E,
                                                  cudaMemcpyHostToDevice));
                            gc.push_back(c);
                            cc.push_back(cpu_ids);
                        }
                        Background bg;
                        const double fg = cond_fill(cond);
                        if (fg >= 0) bg.fill(bank, fg);
                        bg.start();
                        std::vector<double> tc, tg, tm;
                        for (int r = 0; r < a.reps; ++r) {
                            const double t0 = now_us();
                            if (g > 0) {
                                CUDA_CHECK(cudaEventRecord(e0, s));
                                gpu.enqueue(mode, gc[r], T, s);
                                CUDA_CHECK(cudaEventRecord(e1, s));
                            }
                            const double launched = now_us();
                            double c_end          = launched;
                            if (g < n) c_end = launched + cpu_run(cc[r]);
                            double g_end = launched;
                            if (g > 0) {
                                CUDA_CHECK(cudaEventSynchronize(e1));
                                float ms = 0;
                                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                                g_end = t0 + 1e3 * ms +
                                        (launched - t0); // GPU start ~ when the launch returned
                            }
                            tc.push_back(c_end - t0);
                            tg.push_back(g_end - t0);
                            tm.push_back(std::max(c_end, g_end) - t0);
                            std::this_thread::sleep_for(std::chrono::microseconds(300));
                        }
                        const double fill_gbs = bg.finish();
                        const Stat sc = stat(tc), sg = stat(tg), sm = stat(tm);
                        std::printf("{\"arm\":\"joint\",\"cond\":\"%s\",\"mode\":\"%s\",\"n\":%d,"
                                    "\"g\":%d,\"cpu_us\":%.1f,"
                                    "\"gpu_us\":%.1f,\"max_us_med\":%.1f,\"max_us_mean\":%.1f,"
                                    "\"fill_GBps\":%.2f}\n",
                                    cond.c_str(), mode.c_str(), n, g, sc.med, sg.med, sm.med,
                                    sm.mean, fill_gbs);
                        std::fflush(stdout);
                        for (auto& c : gc) free_call(c);
                    }
                }
            }
        }
        return 0;
    }
    std::fprintf(stderr, "--arm gpu|cpu|joint\n");
    return 2;
}
