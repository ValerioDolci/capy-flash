// Host expert path: CPU executor alone, and a synthetic 48-layer window in a CUDA graph where the
// GPU runs stand-in kernels (dense part, resident experts) while the CPU computes the missing
// experts behind the HostMoeMailbox handshake.
//
//   --mode cpu    executor alone on a pinned pool: per-call and per-expert time, GB/s, for a grid
//                 of (experts, tokens per expert) and for the decode pattern
//   --mode graph  arms over the same routing, one CUDA graph of --layers layers each:
//                   gpu    dense + resident stand-ins only (no handshake)
//                   hs0    full graph, no host expert (handshake cost with an idle CPU)
//                   full   full graph with the decode pattern of missing experts
//                 per layer: GPU stamps (globaltimer) before publish and after accumulate, the
//                 resident kernel's end, and the mailbox trace (CPU clock); the clocks are tied by
//                 a ping-pong calibration (offset at minimum round trip, +-RTT/2).
//   --mode calib  only the clock calibration and a pure handshake graph (no stand-in work)
// --fill-gpus N runs a continuous H2D copy from the pinned bank on N GPUs during the measurement
// (the expert-cache fill traffic).
//
// Stand-ins: a kernel holding every SM for a fixed time (globaltimer spin), for the dense part and
// for the resident experts (~8.5 us + 3.37 us per expert at T=4).
#include "ops/host_moe/host_moe_mailbox.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer::ops::host;

namespace {

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,     \
                         __LINE__);                                                                \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int H = 2560, I = 640, TOPK = 10;
constexpr double kExpertBytes = 2764800.0;

struct Args {
    std::string mode = "graph";
    int threads = 6, layers = 48, tokens = 4, replays = 30, warmup = 3, fill_gpus = 0;
    double bank_gib = 3.0, resident_us = 47.0, dense_us = 150.0, p2 = 0.47;
    int umin = 3, umax = 5, calls = 400;
    std::string tag;
    // cpu mode: pool placement, engine-like conditions.
    std::string placement = "spread", fill_mode = "cont", grid_u = "1,2,4,5,8", grid_m = "1,2,3,4";
    int timeline = 0, cold_x = 0, huge = 1, decode_case = 1, blocks = 1;
    std::string variants;
    double gap_us = 0.0, fill_experts = 1.34;
};

double g_thp = -1; // THP-backed fraction of the bank

std::vector<int> int_list(const std::string& s) {
    std::vector<int> v;
    std::size_t i = 0;
    while (i < s.size()) {
        const std::size_t j = s.find(',', i);
        v.push_back(std::stoi(s.substr(i, j == std::string::npos ? std::string::npos : j - i)));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return v;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next           = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--mode")
            a.mode = next();
        else if (k == "--threads")
            a.threads = std::stoi(next());
        else if (k == "--layers")
            a.layers = std::stoi(next());
        else if (k == "--tokens")
            a.tokens = std::stoi(next());
        else if (k == "--replays")
            a.replays = std::stoi(next());
        else if (k == "--warmup")
            a.warmup = std::stoi(next());
        else if (k == "--fill-gpus")
            a.fill_gpus = std::stoi(next());
        else if (k == "--bank-gib")
            a.bank_gib = std::stod(next());
        else if (k == "--resident-us")
            a.resident_us = std::stod(next());
        else if (k == "--dense-us")
            a.dense_us = std::stod(next());
        else if (k == "--p2")
            a.p2 = std::stod(next());
        else if (k == "--experts-min")
            a.umin = std::stoi(next());
        else if (k == "--experts-max")
            a.umax = std::stoi(next());
        else if (k == "--calls")
            a.calls = std::stoi(next());
        else if (k == "--tag")
            a.tag = next();
        else if (k == "--variants")
            a.variants = next();
        else if (k == "--blocks")
            a.blocks = std::stoi(next());
        else if (k == "--placement")
            a.placement = next();
        else if (k == "--timeline")
            a.timeline = std::stoi(next());
        else if (k == "--gap-us")
            a.gap_us = std::stod(next());
        else if (k == "--cold-x")
            a.cold_x = std::stoi(next());
        else if (k == "--huge")
            a.huge = std::stoi(next());
        else if (k == "--fill-mode")
            a.fill_mode = next();
        else if (k == "--fill-experts")
            a.fill_experts = std::stod(next());
        else if (k == "--grid-u")
            a.grid_u = next();
        else if (k == "--grid-m")
            a.grid_m = next();
        else if (k == "--decode-case")
            a.decode_case = std::stoi(next());
        else {
            std::fprintf(stderr, "unknown option %s\n", k.c_str());
            std::exit(2);
        }
    }
    return a;
}

// ------------------------------------------------------------------ synthetic bank (pinned)
struct Bank {
    std::byte* base      = nullptr;
    std::size_t bytes    = 0;
    std::int32_t experts = 0;
    HostExpertLayer layer;
};

Bank make_bank(double gib, int threads, bool huge = true) {
    Bank b;
    b.experts                  = static_cast<std::int32_t>(gib * (1ULL << 30) / kExpertBytes);
    b.experts                  = std::max(b.experts / 4 * 4, 8);
    const std::int64_t gu_rows = std::int64_t{b.experts} * 2 * I,
                       dn_rows = std::int64_t{b.experts} * H;
    const std::size_t gu       = Nvfp4Plane::payload_bytes(gu_rows, H, 2 * b.experts);
    const std::size_t dn       = Nvfp4Plane::payload_bytes(dn_rows, I, b.experts);
    const std::size_t gu_al    = (gu + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    b.bytes                    = gu_al + (dn + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    void* p                    = mmap(nullptr, b.bytes + (2u << 20), PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        std::perror("mmap");
        std::exit(1);
    }
    auto aligned = (reinterpret_cast<std::uintptr_t>(p) + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    b.base       = reinterpret_cast<std::byte*>(aligned);
    madvise(b.base, b.bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    // Fill in parallel: random codes, E4M3 scales 0x30..0x3f (0.5..0.94), divisors 1000.
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t) {
        th.emplace_back([&, t] {
            std::uint64_t s         = 0x9E3779B97F4A7C15ULL * (t + 1);
            const std::size_t words = b.bytes / 8, lo = words * t / threads,
                              hi = words * (t + 1) / threads;
            auto* w              = reinterpret_cast<std::uint64_t*>(b.base);
            for (std::size_t i = lo; i < hi; ++i) {
                s ^= s << 13, s ^= s >> 7, s ^= s << 17;
                w[i] = s;
            }
        });
    }
    for (auto& t : th) t.join();
    auto fix_scales = [&](std::byte* plane, std::int64_t rows, int k, std::int64_t divisors) {
        const std::size_t codes = static_cast<std::size_t>(rows) * k / 2;
        auto* sc            = reinterpret_cast<std::uint8_t*>(plane + (codes + 255) / 256 * 256);
        const std::size_t n = static_cast<std::size_t>(rows) * k / 16;
        for (std::size_t i = 0; i < n; ++i)
            sc[i] = static_cast<std::uint8_t>(0x30 | (sc[i] & 0x0F));
        auto* d = reinterpret_cast<float*>(sc + n);
        for (std::int64_t i = 0; i < divisors; ++i) d[i] = 1000.0F;
    };
    fix_scales(b.base, gu_rows, H, 2 * b.experts);
    fix_scales(b.base + gu_al, dn_rows, I, b.experts);
    b.layer = {Nvfp4Plane::from_payload(b.base, gu_rows, H, 2 * b.experts),
               Nvfp4Plane::from_payload(b.base + gu_al, dn_rows, I, b.experts), b.experts, H, I};
    return b;
}

// Fraction of [p, p + bytes) backed by transparent huge pages (from /proc/self/smaps).
double thp_fraction(const void* p, std::size_t bytes) {
    std::FILE* f = std::fopen("/proc/self/smaps", "r");
    if (!f) return -1;
    const auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + bytes;
    char line[512];
    bool in        = false;
    double huge_kb = 0;
    while (std::fgets(line, sizeof line, f)) {
        unsigned long a = 0, b = 0;
        if (std::sscanf(line, "%lx-%lx ", &a, &b) == 2 && std::strchr(line, ' ') &&
            line[0] != ' ' && std::strncmp(line, "AnonHugePages", 13) != 0) {
            in = a < hi && b > lo;
            continue;
        }
        unsigned long kb = 0;
        if (in && std::sscanf(line, "AnonHugePages: %lu kB", &kb) == 1) huge_kb += kb;
    }
    std::fclose(f);
    return huge_kb * 1024.0 / bytes;
}

// ------------------------------------------------------------------ routing
struct Window {
    // [L][T][TOPK]
    std::vector<std::int32_t> ids, slot;
    std::vector<float> alpha;
    std::vector<int> host_experts; // [L]
    std::vector<int> host_pairs;   // [L]
};

// Per layer: U ~ U{umin..umax} missing experts, each routed by 1 token (2 with probability p2),
// the other top-k entries resident (slot >= 0). U = 0 when umax == 0.
Window make_window(const Args& a, int experts, std::mt19937_64& rng) {
    Window w;
    const int L = a.layers, T = a.tokens;
    w.ids.assign(static_cast<std::size_t>(L) * T * TOPK, 0);
    w.slot.assign(w.ids.size(), 0);
    w.alpha.assign(w.ids.size(), 0.1F);
    std::uniform_int_distribution<int> U(a.umin, a.umax), tok(0, T - 1), ex(0, experts - 1);
    std::bernoulli_distribution two(a.p2);
    for (int l = 0; l < L; ++l) {
        const int u = a.umax > 0 ? U(rng) : 0;
        std::vector<int> chosen;
        while (static_cast<int>(chosen.size()) < u + T * TOPK) {
            const int e = ex(rng);
            if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) chosen.push_back(e);
        }
        std::vector<std::vector<int>> host(T); // host experts of each token
        int pairs = 0;
        for (int i = 0; i < u; ++i) {
            const int m  = (two(rng) && T > 1) ? 2 : 1;
            const int t0 = tok(rng);
            for (int j = 0; j < m; ++j) {
                const int t = (t0 + j) % T;
                if (static_cast<int>(host[t].size()) < TOPK) host[t].push_back(chosen[i]), ++pairs;
            }
        }
        int next = u;
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < TOPK; ++s) {
                const std::size_t i = (static_cast<std::size_t>(l) * T + t) * TOPK + s;
                if (s < static_cast<int>(host[t].size())) {
                    w.ids[i] = host[t][s], w.slot[i] = -1;
                } else {
                    w.ids[i] = chosen[next++], w.slot[i] = s;
                }
            }
        }
        w.host_experts.push_back(u);
        w.host_pairs.push_back(pairs);
    }
    return w;
}

// ------------------------------------------------------------------ stats
struct Stat {
    double med = 0, p10 = 0, p90 = 0, mean = 0;
    std::size_t n = 0;
};

Stat stat(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[static_cast<std::size_t>(std::lround(p * (v.size() - 1)))]; };
    s.med = q(0.5), s.p10 = q(0.1), s.p90 = q(0.9), s.n = v.size();
    for (double x : v) s.mean += x;
    s.mean /= v.size();
    return s;
}

std::string js(const Stat& s) {
    char b[200];
    std::snprintf(b, sizeof b, "{\"med\":%.3f,\"p10\":%.3f,\"p90\":%.3f,\"mean\":%.3f,\"n\":%zu}",
                  s.med, s.p10, s.p90, s.mean, s.n);
    return b;
}

// ------------------------------------------------------------------ concurrent PCIe fill
struct Fill {
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    std::vector<double> gbps;

    void start(const Bank& bank, int gpus) {
        gbps.assign(gpus, 0.0);
        for (int g = 0; g < gpus; ++g) {
            th.emplace_back([this, &bank, g] {
                CK(cudaSetDevice(g));
                cudaStream_t s;
                CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
                const std::size_t chunk = 64u << 20;
                void* dst               = nullptr;
                CK(cudaMalloc(&dst, 4 * chunk));
                std::size_t off = static_cast<std::size_t>(g) * (bank.bytes / 2), bytes = 0;
                const auto t0 = std::chrono::steady_clock::now();
                while (!stop.load(std::memory_order_relaxed)) {
                    for (int k = 0; k < 4; ++k) {
                        if (off + chunk > bank.bytes) off = 0;
                        CK(cudaMemcpyAsync(static_cast<char*>(dst) + k * chunk, bank.base + off,
                                           chunk, cudaMemcpyHostToDevice, s));
                        off += chunk, bytes += chunk;
                    }
                    CK(cudaStreamSynchronize(s));
                }
                const double sec =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                gbps[g] = bytes / sec / 1e9;
                CK(cudaFree(dst));
                CK(cudaStreamDestroy(s));
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // reach steady state
    }

    void finish() {
        stop = true;
        for (auto& t : th) t.join();
        th.clear();
    }
};

std::string fill_json(const Fill& f) {
    std::string s = "[";
    for (std::size_t i = 0; i < f.gbps.size(); ++i) s += (i ? "," : "") + std::to_string(f.gbps[i]);
    return s + "]";
}

// ------------------------------------------------------------------ GPU kernels
__device__ __forceinline__ long long gtime() {
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Holds every SM for ns; block 0 records start/end.
__global__ void standin_kernel(long long ns, long long* stamp) {
    const long long t0 = gtime();
    for (long long t = t0; t - t0 < ns;) t = gtime();
    if (stamp && blockIdx.x == 0 && threadIdx.x == 0) stamp[0] = t0, stamp[1] = gtime();
}

__global__ void stamp_kernel(long long* stamp) { *stamp = gtime(); }

// Calibration only (never in an engine graph): ping a host word, spin until the CPU answers.
__global__ void calib_kernel(volatile unsigned* ping, volatile unsigned* pong, long long* out,
                             int n) {
    for (int i = 0; i < n; ++i) {
        const long long g0 = gtime();
        *ping              = i + 1;
        __threadfence_system();
        while (*pong != static_cast<unsigned>(i + 1)) {}
        out[2 * i] = g0, out[2 * i + 1] = gtime();
    }
}

struct Calib {
    double offset_ns   = 0; // cpu_ns = gpu_ns + offset
    double rtt_ns      = 0; // minimum round trip of the calibration
    double gpu_tick_ns = 0;
};

Calib calibrate(int n = 2000) {
    unsigned* words = nullptr;
    CK(cudaHostAlloc(&words, 256, cudaHostAllocMapped));
    std::memset(words, 0, 256);
    volatile unsigned* ping = words;
    volatile unsigned* pong = words + 16;
    long long* dout         = nullptr;
    CK(cudaMalloc(&dout, sizeof(long long) * 2 * n));
    std::vector<std::int64_t> c(n);
    std::thread cpu([&] {
        for (int i = 0; i < n; ++i) {
            while (*ping != static_cast<unsigned>(i + 1)) _mm_pause();
            c[i]  = HostMoeMailbox::now_ns();
            *pong = i + 1;
        }
    });
    calib_kernel<<<1, 1>>>(ping, pong, dout, n);
    CK(cudaDeviceSynchronize());
    cpu.join();
    std::vector<long long> g(2 * n);
    CK(cudaMemcpy(g.data(), dout, sizeof(long long) * 2 * n, cudaMemcpyDeviceToHost));
    Calib r;
    r.rtt_ns    = 1e30;
    double tick = 1e30;
    for (int i = 10; i < n; ++i) {
        const double rtt = g[2 * i + 1] - g[2 * i];
        if (rtt > 0) tick = std::min(tick, rtt);
        if (rtt < r.rtt_ns) r.rtt_ns = rtt, r.offset_ns = c[i] - 0.5 * (g[2 * i] + g[2 * i + 1]);
    }
    r.gpu_tick_ns = tick;
    CK(cudaFree(dout));
    CK(cudaFreeHost(words));
    return r;
}

// ------------------------------------------------------------------ probe mode: cost of each
// handshake piece
__global__ void write_host_kernel(uint4* dst, int vectors) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < vectors) dst[i] = make_uint4(i, i, i, i);
}

__global__ void read_host_kernel(const float* src, float* dst, int vectors) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= vectors) return;
    const float4 v                    = __ldcv(reinterpret_cast<const float4*>(src) + i);
    reinterpret_cast<float4*>(dst)[i] = v;
}

__global__ void empty_kernel() {}

// Data + doorbell from the kernel: the last block resets done and raises ready (after a system
// fence).
__global__ void write_host_doorbell_kernel(uint4* dst, int vectors, unsigned* counter,
                                           volatile unsigned* done, volatile unsigned* ready) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < vectors) dst[i] = make_uint4(i, i, i, i);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (atomicAdd(counter, 1U) == gridDim.x - 1) {
            *counter = 0;
            *done    = 0;
            __threadfence_system();
            *ready = 1;
        }
    }
}

void cu(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        std::fprintf(stderr, "driver error %d at %s\n", static_cast<int>(r), what);
        std::exit(1);
    }
}

void run_probe(const Args& a, const Bank& bank) {
    CK(cudaSetDevice(0));
    const int L = a.layers, T = a.tokens;
    // Bench-local mapped words and buffers: [L] ready/done lines, x [T*H] bf16, partial [T*H] f32.
    const std::size_t line = 64, words = 2 * line * L;
    std::byte* host = nullptr;
    CK(cudaHostAlloc(reinterpret_cast<void**>(&host), words + 4096 + 8 * H * T * 2,
                     cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(host, 0, words + 4096 + 8 * H * T * 2);
    auto ready = [&](int l) { return reinterpret_cast<volatile unsigned*>(host + line * 2 * l); };
    auto done  = [&](int l) {
        return reinterpret_cast<volatile unsigned*>(host + line * (2 * l + 1));
    };
    auto* one         = reinterpret_cast<unsigned*>(host + words); // constant 1
    *one              = 1;
    auto* xbuf        = reinterpret_cast<uint4*>(host + words + 4096);
    auto* pbuf        = reinterpret_cast<float*>(host + words + 4096 + 2 * H * T * 2);
    float* dtmp       = nullptr;
    long long* dstamp = nullptr;
    CK(cudaMalloc(&dtmp, 4 * H * T));
    CK(cudaMalloc(&dstamp, sizeof(long long) * 2 * L));
    long long* dmid = nullptr;
    CK(cudaMalloc(&dmid, sizeof(long long) * L));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const auto S = reinterpret_cast<CUstream>(s);
    auto dev     = [](const volatile void* p) {
        return reinterpret_cast<CUdeviceptr>(const_cast<void*>(p));
    };

    std::atomic<bool> stop{false};
    std::thread echo([&] { // CPU side of the ping-pong: done[l] = 1 as soon as ready[l] is seen
        const auto cpus = spread_cpus(1);
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[0], &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
        int l = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (*ready(l)) {
                *ready(l) = 0;
                std::atomic_thread_fence(std::memory_order_release);
                *done(l) = 1;
                l        = (l + 1) % L;
            } else {
                _mm_pause();
            }
        }
    });

    const char* names[] = {
        "stamps only",
        "empty kernel",
        "publish kernel (x T*H bf16 to host)",
        "writeValue x2",
        "waitValue on a word already 1",
        "combine-like kernel (T*H f32 from host)",
        "writeValue x2 + waitValue (CPU echo)",
        "full: publish + signal + echo + wait + combine",
        "stand-in kernel (resident_us)",
        "publish + signal + stand-in + wait (echo) + combine",
        "mailbox publish + stand-in + accumulate (no host expert)",
        "mailbox publish + accumulate (no host expert, no stand-in)",
        "mailbox publish + waitValue(constant 1) + accumulate (no host expert)",
        "kernel doorbell (data + flags from the kernel) + echo + wait + combine"};
    unsigned* dcounter = nullptr;
    CK(cudaMalloc(&dcounter, 4));
    CK(cudaMemset(dcounter, 0, 4));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    const int sms          = prop.multiProcessorCount;
    const long long res_ns = static_cast<long long>(a.resident_us * 1e3);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(L) * T * TOPK, -1);
    std::vector<float> alpha(ids.size(), 0.1F);
    std::int32_t* dids = nullptr;
    float* dalpha      = nullptr;
    __nv_bfloat16 *dx = nullptr, *dy = nullptr;
    CK(cudaMalloc(&dids, ids.size() * 4));
    CK(cudaMalloc(&dalpha, ids.size() * 4));
    CK(cudaMalloc(&dx, static_cast<std::size_t>(L) * T * H * 2));
    CK(cudaMalloc(&dy, static_cast<std::size_t>(L) * T * H * 2));
    CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dalpha, alpha.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dx, 0, static_cast<std::size_t>(L) * T * H * 2));
    CK(cudaMemset(dy, 0, static_cast<std::size_t>(L) * T * H * 2));
    HostMoeMailboxConfig cfg;
    cfg.device = 0, cfg.layers = L, cfg.max_tokens = kHostMaxTokens, cfg.top_k = TOPK,
    cfg.hidden = H, cfg.intermediate = I;
    const auto all = spread_cpus(a.threads + 1);
    cfg.cpus.assign(all.begin() + 1, all.end()); // the echo thread keeps all[0]
    cfg.trace_capacity = static_cast<std::int64_t>(L) * (a.warmup + a.replays);
    HostMoeMailbox mailbox(cfg,
                           std::vector<HostExpertLayer>(static_cast<std::size_t>(L), bank.layer));
    const int xv = T * H / 8, pv = T * H / 4;
    for (int v = 0; v < 14; ++v) {
        cudaGraph_t g;
        cudaGraphExec_t e;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < L; ++l) {
            stamp_kernel<<<1, 1, 0, s>>>(dstamp + 2 * l);
            if (v == 1) empty_kernel<<<1, 32, 0, s>>>();
            const std::size_t ox  = static_cast<std::size_t>(l) * T * H,
                              orr = static_cast<std::size_t>(l) * T * TOPK;
            if (v == 10 || v == 11 || v == 12)
                mailbox.publish(s, l, T, dx + ox, dids + orr, dalpha + orr, nullptr);
            if (v == 12) cu(cuStreamWaitValue32(S, dev(one), 1, CU_STREAM_WAIT_VALUE_GEQ), "flush");
            if (v == 11 || v == 12) stamp_kernel<<<1, 1, 0, s>>>(dmid + l);
            if (v == 13) {
                write_host_doorbell_kernel<<<(xv + 255) / 256, 256, 0, s>>>(
                    xbuf, xv, dcounter, const_cast<unsigned*>(done(l)),
                    const_cast<unsigned*>(ready(l)));
                stamp_kernel<<<1, 1, 0, s>>>(dmid + l);
                cu(cuStreamWaitValue32(S, dev(done(l)), 1, CU_STREAM_WAIT_VALUE_GEQ), "waitd13");
                read_host_kernel<<<(pv + 255) / 256, 256, 0, s>>>(pbuf, dtmp, pv);
            }
            if (v == 2 || v == 7 || v == 9)
                write_host_kernel<<<(xv + 255) / 256, 256, 0, s>>>(xbuf, xv);
            if (v == 3 || v == 6 || v == 7 || v == 9) {
                cu(cuStreamWriteValue32(S, dev(done(l)), 0, 0), "w0");
                cu(cuStreamWriteValue32(S, dev(ready(l)), v == 3 ? 0 : 1, 0), "w1");
            }
            if (v == 4) cu(cuStreamWaitValue32(S, dev(one), 1, CU_STREAM_WAIT_VALUE_GEQ), "wait1");
            if (v == 8 || v == 9 || v == 10) standin_kernel<<<sms, 32, 0, s>>>(res_ns, nullptr);
            if (v == 6 || v == 7 || v == 9)
                cu(cuStreamWaitValue32(S, dev(done(l)), 1, CU_STREAM_WAIT_VALUE_GEQ), "waitd");
            if (v == 5 || v == 7 || v == 9)
                read_host_kernel<<<(pv + 255) / 256, 256, 0, s>>>(pbuf, dtmp, pv);
            if (v == 10 || v == 11 || v == 12) mailbox.accumulate(s, l, T, dy + ox);
            stamp_kernel<<<1, 1, 0, s>>>(dstamp + 2 * l + 1);
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&e, g, 0));
        std::vector<double> seg, win, g2c, c2g, rtt, mid2c;
        const bool mb = v == 10 || v == 11 || v == 12;
        Calib cal;
        if (mb) {
            cal = calibrate(500);
            mailbox.clear_trace();
        }
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int r = 0; r < a.warmup + a.replays; ++r) {
            CK(cudaEventRecord(e0, s));
            CK(cudaGraphLaunch(e, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaStreamSynchronize(s));
            if (r < a.warmup) continue;
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            win.push_back(ms * 1e3 / L);
            std::vector<long long> st(2 * L);
            CK(cudaMemcpy(st.data(), dstamp, sizeof(long long) * 2 * L, cudaMemcpyDeviceToHost));
            for (int l = 0; l < L; ++l) seg.push_back((st[2 * l + 1] - st[2 * l]) / 1e3);
            std::vector<long long> mid(L);
            if (v >= 11)
                CK(cudaMemcpy(mid.data(), dmid, sizeof(long long) * L, cudaMemcpyDeviceToHost));
            if (mb) {
                const auto tr = mailbox.trace();
                for (int l = 0; l < L; ++l) {
                    const auto& rec = tr[tr.size() - L + l];
                    g2c.push_back((rec.seen_ns - (st[2 * l] + cal.offset_ns)) / 1e3);
                    c2g.push_back(((st[2 * l + 1] + cal.offset_ns) - rec.done_ns) / 1e3);
                    rtt.push_back((st[2 * l + 1] - st[2 * l] - (rec.done_ns - rec.seen_ns)) / 1e3);
                    if (v >= 11) mid2c.push_back((rec.seen_ns - (mid[l] + cal.offset_ns)) / 1e3);
                }
            }
        }
        if (mb) {
            std::printf(
                "{\"mode\":\"probe_split\",\"variant\":%d,\"calib_rtt_ns\":%.0f,\"gpu_to_cpu_us\":%"
                "s,\"cpu_to_gpu_us\":%s,\"rtt_us\":%s,\"after_doorbell_stamp_to_cpu_us\":%s}\n",
                v, cal.rtt_ns, js(stat(g2c)).c_str(), js(stat(c2g)).c_str(), js(stat(rtt)).c_str(),
                js(stat(mid2c)).c_str());
        }
        std::printf("{\"mode\":\"probe\",\"tag\":\"%s\",\"variant\":%d,\"what\":\"%s\",\"T\":%d,"
                    "\"fill_gpus\":%d,"
                    "\"segment_us\":%s,\"window_per_layer_us\":%s}\n",
                    a.tag.c_str(), v, names[v], T, a.fill_gpus, js(stat(seg)).c_str(),
                    js(stat(win)).c_str());
        std::fflush(stdout);
        CK(cudaGraphExecDestroy(e));
        CK(cudaGraphDestroy(g));
    }
    stop = true;
    echo.join();
    CK(cudaFree(dtmp));
    CK(cudaFree(dstamp));
    CK(cudaFree(dids));
    CK(cudaFree(dalpha));
    CK(cudaFree(dx));
    CK(cudaFree(dy));
    CK(cudaFree(dmid));
    CK(cudaFreeHost(host));
}

// ------------------------------------------------------------------ cpu mode
// Per-call expert-cache fill (fill_mode overlap | gap, --fill-gpus 1): a helper thread pinned to a
// core outside the pool issues expert-sized H2D copies (random bank offsets) when triggered, as
// the engine's policy thread does once per layer (measured 64.5 fills per token = 1.34 per layer).
struct CallFill {
    std::atomic<int> pending{0};
    std::atomic<bool> stop{false};
    std::atomic<long> issued{0};
    std::thread th;

    void start(const Bank& bank, int cpu) {
        th = std::thread([this, &bank, cpu] {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(pthread_self(), sizeof set, &set);
            CK(cudaSetDevice(0));
            cudaStream_t st;
            CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            const std::size_t E = static_cast<std::size_t>(kExpertBytes);
            void* dst           = nullptr;
            CK(cudaMalloc(&dst, 8 * E));
            std::mt19937_64 rng(7);
            std::uniform_int_distribution<std::size_t> off(0, bank.bytes / E - 1);
            long n = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                const int k = pending.exchange(0, std::memory_order_acq_rel);
                if (k == 0) {
                    _mm_pause();
                    continue;
                }
                for (int i = 0; i < k; ++i, ++n) {
                    CK(cudaMemcpyAsync(static_cast<char*>(dst) + (n % 8) * E,
                                       bank.base + off(rng) * E, E, cudaMemcpyHostToDevice, st));
                }
                issued.fetch_add(k, std::memory_order_relaxed);
            }
            CK(cudaStreamSynchronize(st));
            CK(cudaFree(dst));
            CK(cudaStreamDestroy(st));
        });
    }

    void finish() {
        stop = true;
        if (th.joinable()) th.join();
    }
};

std::vector<int> pool_cpus(const Args& a) {
    if (a.placement == "spread") return spread_cpus(a.threads);
    if (a.placement == "packed0") return packed_cpus(a.threads, 0);
    if (a.placement == "packed1") return packed_cpus(a.threads, 1);
    std::fprintf(stderr, "unknown placement %s\n", a.placement.c_str());
    std::exit(2);
}

void spin_us(double us) {
    const auto t0 = HostMoeMailbox::now_ns();
    while (HostMoeMailbox::now_ns() - t0 < us * 1e3) _mm_pause();
}

// One executor configuration of the cpu mode. --variants "name:key=val,key=val;name2:..." runs
// several over the same bank, interleaved in --blocks blocks per case (pool rebuilt per block, so
// only one pool spins at a time); keys: placement, threads, fill (none|overlap|gap; needs
// --fill-gpus 1). Without --variants: one from the flags.
struct Variant {
    std::string name;
    std::string placement, fill;
    int threads = 6;
};

std::vector<Variant> parse_variants(const Args& a) {
    Variant base{a.tag, a.placement, a.fill_gpus > 0 ? a.fill_mode : "none", a.threads};
    if (a.variants.empty()) return {base};
    std::vector<Variant> out;
    std::size_t i = 0;
    while (i < a.variants.size()) {
        std::size_t j = a.variants.find(';', i);
        if (j == std::string::npos) j = a.variants.size();
        const std::string spec = a.variants.substr(i, j - i);
        i                      = j + 1;
        if (spec.empty()) continue;
        Variant v           = base;
        const std::size_t c = spec.find(':');
        v.name              = spec.substr(0, c);
        std::string kv      = c == std::string::npos ? "" : spec.substr(c + 1);
        std::size_t k       = 0;
        while (k < kv.size()) {
            std::size_t e = kv.find(',', k);
            if (e == std::string::npos) e = kv.size();
            const std::string item = kv.substr(k, e - k);
            k                      = e + 1;
            const std::size_t eq   = item.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "bad variant item %s\n", item.c_str());
                std::exit(2);
            }
            const std::string key = item.substr(0, eq), val = item.substr(eq + 1);
            if (key == "placement")
                v.placement = val;
            else if (key == "threads")
                v.threads = std::stoi(val);
            else if (key == "fill")
                v.fill = val;
            else {
                std::fprintf(stderr, "bad variant key %s\n", key.c_str());
                std::exit(2);
            }
        }
        out.push_back(v);
    }
    return out;
}

struct CallStats {
    std::vector<double> us, per_expert, ph0, ph1, ph2;
    std::vector<double> t_setup, t_wake, t_b1, t_g_min, t_g_mean, t_g_max, t_b2, t_d_min, t_d_mean,
        t_d_max, t_tail;
    double experts = 0, pairs = 0;
    long fills = 0;
    int calls  = 0;
};

void run_cpu(const Args& a, const Bank& bank) {
    const std::vector<Variant> variants = parse_variants(a);
    std::mt19937_64 rng(1);
    // x: 64-byte aligned; --cold-x flushes it from the caches before every call (the engine's x is
    // written by the GPU into mapped memory just before the doorbell).
    const std::size_t x_elems = static_cast<std::size_t>(kHostMaxTokens) * H;
    auto* x                   = static_cast<std::uint16_t*>(std::aligned_alloc(64, x_elems * 2));
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (std::size_t i = 0; i < x_elems; ++i) {
        const float f = nd(rng);
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        x[i] = static_cast<std::uint16_t>(u >> 16);
    }
    std::vector<float> out(static_cast<std::size_t>(kHostMaxTokens) * H);
    std::uniform_int_distribution<int> ex_d(0, bank.experts - 1);
    CallFill fill;
    bool any_call_fill = false;
    std::vector<int> used;
    for (const Variant& v : variants) {
        any_call_fill |= v.fill == "overlap" || v.fill == "gap";
        Args va    = a;
        va.threads = v.threads, va.placement = v.placement;
        for (const int c : pool_cpus(va)) used.push_back(c);
    }
    if (a.fill_gpus > 0 && any_call_fill) {
        int helper = -1;
        for (const int c : spread_cpus(12)) {
            if (std::find(used.begin(), used.end(), c) == used.end()) helper = c;
        }
        if (helper < 0) {
            std::fprintf(stderr, "no free core for the fill helper\n");
            std::exit(2);
        }
        fill.start(bank, helper);
    }
    std::bernoulli_distribution fill_extra(a.fill_experts - std::floor(a.fill_experts));
    const int blocks = std::max(1, a.blocks), per_block = std::max(1, a.calls / blocks);
    auto bench = [&](const std::string& label, int T, auto make_ids) {
        std::vector<CallStats> stats(variants.size());
        std::vector<std::int32_t> ids(static_cast<std::size_t>(T) * TOPK);
        std::vector<float> alpha(ids.size(), 0.1F);
        for (int blk = 0; blk < blocks; ++blk) {
            for (std::size_t vi = 0; vi < variants.size(); ++vi) {
                const Variant& v = variants[vi];
                CallStats& cs    = stats[vi];
                Args va          = a;
                va.threads = v.threads, va.placement = v.placement;
                HostCpuPool pool(pool_cpus(va));
                HostMoeExecutor::Options opt;
                opt.record_members = a.timeline != 0;
                HostMoeExecutor ex(pool, H, I, kHostMaxTokens, TOPK, opt);
                const bool vfill   = a.fill_gpus > 0 && (v.fill == "overlap" || v.fill == "gap");
                const long issued0 = fill.issued.load();
                const int warm     = blk == 0 ? 20 : 5;
                for (int c = 0; c < per_block + warm; ++c) {
                    make_ids(ids);
                    if (a.cold_x) {
                        for (std::size_t i = 0; i < static_cast<std::size_t>(T) * H * 2; i += 64)
                            _mm_clflush(reinterpret_cast<char*>(x) + i);
                        _mm_sfence();
                    }
                    int nfill = 0;
                    if (vfill)
                        nfill = static_cast<int>(std::floor(a.fill_experts)) +
                                (fill_extra(rng) ? 1 : 0);
                    if (vfill && v.fill == "overlap") fill.pending.fetch_add(nfill);
                    const HostMoeJob job{T, TOPK, x, ids.data(), alpha.data(), out.data()};
                    const auto t0   = HostMoeMailbox::now_ns();
                    const auto st   = ex.run(bank.layer, job);
                    const double dt = (HostMoeMailbox::now_ns() - t0) / 1e3;
                    if (vfill && v.fill == "gap") fill.pending.fetch_add(nfill);
                    if (a.gap_us > 0) spin_us(a.gap_us);
                    if (c < warm) continue;
                    ++cs.calls;
                    cs.us.push_back(dt);
                    cs.per_expert.push_back(st.experts ? dt / st.experts : 0);
                    cs.ph0.push_back(st.phase_ns[0] / 1e3);
                    cs.ph1.push_back((st.phase_ns[1] - st.phase_ns[0]) / 1e3);
                    cs.ph2.push_back((st.phase_ns[2] - st.phase_ns[1]) / 1e3);
                    cs.experts += st.experts, cs.pairs += st.pairs;
                    if (st.members != nullptr && st.experts > 0) {
                        const int n = st.member_count;
                        double wake = 0, a1 = 0, l1 = 0, gmin = 1e18, gsum = 0, gmax = 0, e3 = 0,
                               l3 = 0, dmin = 1e18, dsum = 0, dmax = 0, e5 = 0;
                        for (int m = 0; m < n; ++m) {
                            const auto& ns = st.members[m].ns;
                            wake           = std::max(wake, double(ns[0] - st.dispatch_ns));
                            a1             = std::max(a1, double(ns[1]));
                            l1 += double(ns[2]) / n;
                            const double g = double(ns[3] - ns[2]), d = double(ns[5] - ns[4]);
                            gmin = std::min(gmin, g), gmax = std::max(gmax, g), gsum += g / n;
                            dmin = std::min(dmin, d), dmax = std::max(dmax, d), dsum += d / n;
                            e3 = std::max(e3, double(ns[3]));
                            l3 += double(ns[4]) / n;
                            e5 = std::max(e5, double(ns[5]));
                        }
                        cs.t_setup.push_back(st.dispatch_ns / 1e3);
                        cs.t_wake.push_back(wake / 1e3);
                        cs.t_b1.push_back(std::max(0.0, l1 - a1) / 1e3);
                        cs.t_g_min.push_back(gmin / 1e3), cs.t_g_mean.push_back(gsum / 1e3),
                            cs.t_g_max.push_back(gmax / 1e3);
                        cs.t_b2.push_back(std::max(0.0, l3 - e3) / 1e3);
                        cs.t_d_min.push_back(dmin / 1e3), cs.t_d_mean.push_back(dsum / 1e3),
                            cs.t_d_max.push_back(dmax / 1e3);
                        cs.t_tail.push_back((st.end_ns - e5) / 1e3);
                    }
                }
                cs.fills += fill.issued.load() - issued0;
            }
        }
        for (std::size_t vi = 0; vi < variants.size(); ++vi) {
            const Variant& v    = variants[vi];
            const CallStats& cs = stats[vi];
            const Stat s = stat(cs.us), pe = stat(cs.per_expert);
            const double mean_u = cs.experts / cs.calls;
            std::string tl;
            if (!cs.t_setup.empty()) {
                tl = ",\"timeline\":{\"setup\":" + js(stat(cs.t_setup)) +
                     ",\"wake\":" + js(stat(cs.t_wake)) + ",\"b1\":" + js(stat(cs.t_b1)) +
                     ",\"gate_min\":" + js(stat(cs.t_g_min)) +
                     ",\"gate_mean\":" + js(stat(cs.t_g_mean)) +
                     ",\"gate_max\":" + js(stat(cs.t_g_max)) + ",\"b2\":" + js(stat(cs.t_b2)) +
                     ",\"down_min\":" + js(stat(cs.t_d_min)) +
                     ",\"down_mean\":" + js(stat(cs.t_d_mean)) +
                     ",\"down_max\":" + js(stat(cs.t_d_max)) + ",\"tail\":" + js(stat(cs.t_tail)) +
                     "}";
            }
            char cfg[640];
            std::snprintf(cfg, sizeof cfg,
                          "\"variant\":\"%s\","
                          "\"placement\":\"%s\",\"gap_us\":%.0f,\"cold_x\":%d,\"huge\":%d,\"thp\":%"
                          ".3f,\"fill_mode\":\"%s\","
                          "\"fill_experts\":%.2f,\"fills\":%ld,\"blocks\":%d",
                          v.name.c_str(), v.placement.c_str(), a.gap_us, a.cold_x, a.huge, g_thp,
                          a.fill_gpus > 0 ? v.fill.c_str() : "none", a.fill_experts, cs.fills,
                          blocks);
            std::printf("{\"mode\":\"cpu\",\"tag\":\"%s\",\"case\":\"%s\",\"threads\":%d,"
                        "\"fill_gpus\":%d,%s,"
                        "\"T\":%d,\"experts_mean\":%.3f,\"tokens_per_expert\":%.3f,\"call_us\":%s,"
                        "\"us_per_expert\":%s,"
                        "\"gbps_med\":%.2f,\"prep_us\":%s,\"gate_up_us\":%s,\"down_us\":%s%s}\n",
                        a.tag.c_str(), label.c_str(), v.threads, a.fill_gpus, cfg, T, mean_u,
                        cs.pairs / cs.experts, js(s).c_str(), js(pe).c_str(),
                        mean_u * kExpertBytes / (s.med * 1e3), js(stat(cs.ph0)).c_str(),
                        js(stat(cs.ph1)).c_str(), js(stat(cs.ph2)).c_str(), tl.c_str());
        }
        std::fflush(stdout);
    };
    // Grid: U distinct experts, each routed by the same m tokens (T = max(m, --tokens)).
    for (const int m : int_list(a.grid_m)) {
        for (const int U : int_list(a.grid_u)) {
            const int T = std::max(m, a.tokens);
            bench("grid U" + std::to_string(U) + " m" + std::to_string(m), T,
                  [&](std::vector<std::int32_t>& ids) {
                      std::fill(ids.begin(), ids.end(), -1);
                      std::vector<int> chosen;
                      while (static_cast<int>(chosen.size()) < U) {
                          const int e = ex_d(rng);
                          if (std::find(chosen.begin(), chosen.end(), e) == chosen.end())
                              chosen.push_back(e);
                      }
                      for (int t = 0; t < m; ++t)
                          for (int u = 0; u < U; ++u)
                              ids[static_cast<std::size_t>(t) * TOPK + u] = chosen[u];
                  });
        }
    }
    if (a.decode_case) {
        // Decode pattern (same generator as the graph mode, one layer per call).
        Args one   = a;
        one.layers = 1;
        bench("decode U" + std::to_string(a.umin) + "-" + std::to_string(a.umax) + " p2 " +
                  std::to_string(a.p2),
              a.tokens, [&](std::vector<std::int32_t>& ids) {
                  const Window w = make_window(one, bank.experts, rng);
                  for (std::size_t i = 0; i < ids.size(); ++i)
                      ids[i] = w.slot[i] < 0 ? w.ids[i] : -1;
              });
    }
    fill.finish();
    std::free(x);
}

// ------------------------------------------------------------------ graph mode
struct Arm {
    std::string name;
    bool handshake = true;
    bool host_work = true;
};

void run_graph(const Args& a, const Bank& bank, bool calib_only) {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount;
    const int L = a.layers, T = a.tokens;
    const Calib c0 = calibrate();
    std::printf("{\"mode\":\"calib\",\"tag\":\"%s\",\"offset_ns\":%.0f,\"rtt_min_ns\":%.0f,\"gpu_"
                "tick_ns\":%.0f}\n",
                a.tag.c_str(), c0.offset_ns, c0.rtt_ns, c0.gpu_tick_ns);
    std::fflush(stdout);

    HostMoeMailboxConfig cfg;
    cfg.device = 0, cfg.layers = L, cfg.max_tokens = kHostMaxTokens, cfg.top_k = TOPK,
    cfg.hidden = H, cfg.intermediate = I;
    cfg.cpus           = spread_cpus(a.threads);
    cfg.trace_capacity = L * 4;
    HostMoeMailbox mailbox(cfg,
                           std::vector<HostExpertLayer>(static_cast<std::size_t>(L), bank.layer));

    const std::size_t nx = static_cast<std::size_t>(L) * T * H,
                      nr = static_cast<std::size_t>(L) * T * TOPK;
    __nv_bfloat16 *dx = nullptr, *dy = nullptr;
    std::int32_t *dids = nullptr, *dslot = nullptr;
    float* dalpha = nullptr;
    long long* dstamp =
        nullptr; // [L][4]: before publish, after accumulate, resident start, resident end
    CK(cudaMalloc(&dx, nx * 2));
    CK(cudaMalloc(&dy, nx * 2));
    CK(cudaMalloc(&dids, nr * 4));
    CK(cudaMalloc(&dslot, nr * 4));
    CK(cudaMalloc(&dalpha, nr * 4));
    CK(cudaMalloc(&dstamp, sizeof(long long) * 4 * L));
    CK(cudaMemset(dx, 0x3c, nx * 2)); // bf16 ~1.1
    CK(cudaMemset(dy, 0, nx * 2));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    const double dense_us = calib_only ? 0 : a.dense_us, res_us = calib_only ? 0 : a.resident_us;
    auto capture = [&](const Arm& arm, bool stamps) {
        cudaGraph_t g;
        cudaGraphExec_t e;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < L; ++l) {
            const std::size_t ox  = static_cast<std::size_t>(l) * T * H,
                              orr = static_cast<std::size_t>(l) * T * TOPK;
            if (dense_us > 0)
                standin_kernel<<<sms, 32, 0, s>>>(static_cast<long long>(dense_us * 1e3), nullptr);
            if (stamps) stamp_kernel<<<1, 1, 0, s>>>(dstamp + 4 * l);
            if (arm.handshake)
                mailbox.publish(s, l, T, dx + ox, dids + orr, dalpha + orr, dslot + orr);
            if (res_us > 0)
                standin_kernel<<<sms, 32, 0, s>>>(static_cast<long long>(res_us * 1e3),
                                                  stamps ? dstamp + 4 * l + 2 : nullptr);
            if (arm.handshake) mailbox.accumulate(s, l, T, dy + ox);
            if (stamps) stamp_kernel<<<1, 1, 0, s>>>(dstamp + 4 * l + 1);
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&e, g, 0));
        CK(cudaGraphDestroy(g));
        return e;
    };

    std::vector<Arm> arms;
    if (calib_only) {
        arms = {{"none", false, false}, {"hs0", true, false}};
    } else {
        arms = {{"gpu", false, false}, {"hs0", true, false}, {"full", true, true}};
    }
    std::mt19937_64 rng(11);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    Fill fill;
    if (a.fill_gpus > 0) fill.start(bank, a.fill_gpus);
    for (const Arm& arm : arms) {
        for (const bool stamps : {false, true}) {
            cudaGraphExec_t exec = capture(arm, stamps);
            // Fresh clock tie per arm: the globaltimer / CLOCK_MONOTONIC offset drifts ~10 us/s.
            const Calib cal = stamps ? calibrate(500) : c0;
            std::vector<double> window_us;
            // per-layer samples
            std::vector<double> cpu_us, per_expert, sig, comp, rtt, moe, res_dur;
            double sum_moe = 0, sum_res = 0, sum_cpu = 0, sum_ideal = 0;
            int cpu_critical = 0, layers_seen = 0;
            for (int r = 0; r < a.warmup + a.replays; ++r) {
                Args wa = a;
                if (!arm.host_work) wa.umax = 0, wa.umin = 0;
                const Window w = make_window(wa, bank.experts, rng);
                CK(cudaMemcpyAsync(dids, w.ids.data(), nr * 4, cudaMemcpyHostToDevice, s));
                CK(cudaMemcpyAsync(dslot, w.slot.data(), nr * 4, cudaMemcpyHostToDevice, s));
                CK(cudaMemcpyAsync(dalpha, w.alpha.data(), nr * 4, cudaMemcpyHostToDevice, s));
                CK(cudaStreamSynchronize(s));
                mailbox.clear_trace();
                CK(cudaEventRecord(e0, s));
                CK(cudaGraphLaunch(exec, s));
                CK(cudaEventRecord(e1, s));
                CK(cudaStreamSynchronize(s));
                if (mailbox.failed()) {
                    std::fprintf(stderr, "mailbox failed: %s\n", mailbox.error().c_str());
                    std::exit(1);
                }
                if (r < a.warmup) continue;
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                window_us.push_back(ms * 1e3);
                if (!stamps || !arm.handshake) continue;
                std::vector<long long> st(4 * L);
                CK(cudaMemcpy(st.data(), dstamp, sizeof(long long) * 4 * L,
                              cudaMemcpyDeviceToHost));
                const auto tr = mailbox.trace();
                if (static_cast<int>(tr.size()) != L) {
                    std::fprintf(stderr, "trace %zu != %d\n", tr.size(), L);
                    std::exit(1);
                }
                for (int l = 0; l < L; ++l) {
                    const auto& rec = tr[l];
                    const double g0 = st[4 * l] + cal.offset_ns, g1 = st[4 * l + 1] + cal.offset_ns;
                    const double rend = st[4 * l + 3] + cal.offset_ns;
                    const double rd   = res_us > 0 ? (st[4 * l + 3] - st[4 * l + 2]) / 1e3 : 0.0;
                    const double cu   = (rec.done_ns - rec.seen_ns) / 1e3;
                    const double m    = (g1 - g0) / 1e3;
                    cpu_us.push_back(cu);
                    if (rec.experts) per_expert.push_back(cu / rec.experts);
                    sig.push_back((rec.seen_ns - g0) / 1e3);
                    moe.push_back(m);
                    res_dur.push_back(rd);
                    ++layers_seen;
                    sum_moe += m, sum_res += rd, sum_cpu += cu, sum_ideal += std::max(rd, cu);
                    if (rec.done_ns > rend || res_us == 0) {
                        ++cpu_critical;
                        comp.push_back((g1 - rec.done_ns) / 1e3);
                        rtt.push_back(m - cu);
                    }
                }
            }
            CK(cudaGraphExecDestroy(exec));
            std::printf("{\"mode\":\"%s\",\"tag\":\"%s\",\"arm\":\"%s\",\"stamps\":%d,\"layers\":%"
                        "d,\"T\":%d,\"threads\":%d,"
                        "\"dense_us\":%.1f,\"resident_us\":%.1f,\"experts\":\"%d-%"
                        "d\",\"p2\":%.2f,"
                        "\"fill_gpus\":%d,\"window_us\":%s",
                        calib_only ? "calib" : "graph", a.tag.c_str(), arm.name.c_str(),
                        stamps ? 1 : 0, L, T, a.threads, dense_us, res_us,
                        arm.host_work ? a.umin : 0, arm.host_work ? a.umax : 0, a.p2, a.fill_gpus,
                        js(stat(window_us)).c_str());
            if (stamps && arm.handshake) {
                // Overlap efficiency over the window: (sum resident + sum cpu - sum moe) / sum
                // min(resident, cpu).
                double sum_min = 0;
                for (std::size_t i = 0; i < cpu_us.size(); ++i)
                    sum_min += std::min(res_dur[i], cpu_us[i]);
                const double eff = sum_min > 0 ? (sum_res + sum_cpu - sum_moe) / sum_min : 0.0;
                std::printf(",\"layer_moe_us\":%s,\"cpu_us\":%s,\"cpu_us_per_expert\":%s,\"gpu_to_"
                            "cpu_us\":%s,"
                            "\"cpu_to_gpu_us\":%s,\"sync_rtt_us\":%s,\"cpu_critical_layers\":%d,"
                            "\"layer_samples\":%d,"
                            "\"sum_moe_us\":%.1f,\"sum_ideal_us\":%.1f,\"sum_cpu_us\":%.1f,\"sum_"
                            "resident_us\":%.1f,"
                            "\"overlap_eff\":%.3f",
                            js(stat(moe)).c_str(), js(stat(cpu_us)).c_str(),
                            js(stat(per_expert)).c_str(), js(stat(sig)).c_str(),
                            js(stat(comp)).c_str(), js(stat(rtt)).c_str(), cpu_critical,
                            layers_seen, sum_moe / a.replays, sum_ideal / a.replays,
                            sum_cpu / a.replays, sum_res / a.replays, eff);
            }
            std::printf("}\n");
            std::fflush(stdout);
        }
    }
    if (a.fill_gpus > 0) {
        fill.finish();
        std::printf("{\"mode\":\"fill\",\"tag\":\"%s\",\"gbps\":%s}\n", a.tag.c_str(),
                    fill_json(fill).c_str());
    }
    const Calib c1 = calibrate();
    std::printf("{\"mode\":\"calib_end\",\"tag\":\"%s\",\"offset_ns\":%.0f,\"rtt_min_ns\":%.0f,"
                "\"drift_ns\":%.0f}\n",
                a.tag.c_str(), c1.offset_ns, c1.rtt_ns, c1.offset_ns - c0.offset_ns);
    CK(cudaFree(dx));
    CK(cudaFree(dy));
    CK(cudaFree(dids));
    CK(cudaFree(dslot));
    CK(cudaFree(dalpha));
    CK(cudaFree(dstamp));
}

} // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    if (!host_experts_supported()) {
        std::fprintf(stderr, "needs AVX2 + FMA + F16C\n");
        return 77;
    }
    std::fprintf(stderr, "host ISA: %s\n", host_isa_name(host_isa()));
    Bank bank            = make_bank(a.bank_gib, 12, a.huge != 0);
    const bool need_cuda = a.mode != "cpu" || a.fill_gpus > 0;
    if (need_cuda) {
        CK(cudaSetDevice(0));
        CK(cudaHostRegister(bank.base, bank.bytes,
                            cudaHostRegisterPortable | cudaHostRegisterMapped));
    }
    g_thp = thp_fraction(bank.base, bank.bytes);
    std::fprintf(stderr, "bank: %d experts, %.2f GiB, THP %.3f\n", bank.experts,
                 bank.bytes / double(1ULL << 30), g_thp);
    if (a.mode == "cpu") {
        Fill fill;
        if (a.fill_gpus > 0 && a.fill_mode == "cont") fill.start(bank, a.fill_gpus);
        run_cpu(a, bank);
        if (a.fill_gpus > 0 && a.fill_mode == "cont") {
            fill.finish();
            std::printf("{\"mode\":\"fill\",\"tag\":\"%s\",\"gbps\":%s}\n", a.tag.c_str(),
                        fill_json(fill).c_str());
        }
    } else if (a.mode == "probe") {
        Fill fill;
        if (a.fill_gpus > 0) fill.start(bank, a.fill_gpus);
        run_probe(a, bank);
        if (a.fill_gpus > 0) fill.finish();
    } else if (a.mode == "graph" || a.mode == "calib") {
        run_graph(a, bank, a.mode == "calib");
    } else {
        std::fprintf(stderr, "unknown mode\n");
        return 2;
    }
    if (need_cuda) CK(cudaHostUnregister(bank.base));
    return 0;
}
