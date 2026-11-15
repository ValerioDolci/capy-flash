#include "ops/sparse_moe/host/host_moe_mailbox.h"

#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda.h>
#include <immintrin.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::host {

// Where an id is resident on the device: per route (slot[j] >= 0) or per expert id
// (table[id] >= 0 for 0 <= id < experts); neither = every id goes to the host. Of the ids not
// resident, a share chosen by `split` (HostMoeSplit) is taken by the device itself (written to
// `share`) instead of the host.
struct HostMoeResidency {
    const std::int32_t* slot  = nullptr;
    const std::int32_t* table = nullptr;
    int experts               = 0;
    HostMoeSplit split{};
    HostMoeDeviceShare share{};
};

namespace {

using Residency = HostMoeResidency;

constexpr std::size_t kLine   = 64;
constexpr auto kResyncAfter   = std::chrono::milliseconds(1);
constexpr auto kDeepIdleAfter = std::chrono::seconds(1);

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

void cu_check(CUresult status, const char* what) {
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("host MoE mailbox: ") + what + ": " +
                                 (name ? name : "?"));
    }
}

CUdeviceptr device_word(const volatile std::uint32_t* p) {
    return reinterpret_cast<CUdeviceptr>(const_cast<std::uint32_t*>(p));
}

std::uint32_t load_acquire(const volatile std::uint32_t* p) {
    const std::uint32_t v = *p;
    std::atomic_thread_fence(std::memory_order_acquire);
    return v;
}

void store_release(volatile std::uint32_t* p, std::uint32_t v) {
    std::atomic_thread_fence(std::memory_order_release);
    *p = v;
}

struct SlotDevice {
    std::uint32_t* tokens;
    std::int32_t* ids;
    float* alpha;
    uint4* x;
};

constexpr int kMaxRoutes      = 128; // kHostMaxTokens x top-k <= 16
constexpr int kMaxDeviceReads = HostMoeSplit::kMaxReads;
constexpr int kSplitWords     = HostMoeRouteLog::kSplitWords;

// Optional routing log of the layer (publish_table) and the publish stamp.
struct LogDevice {
    std::uint32_t* published  = nullptr; // [0] published flag, [1] tokens logged
    std::int32_t* ids         = nullptr; // at the call's token offset
    std::uint32_t end_token   = 0;
    unsigned long long* stamp = nullptr;
    std::uint32_t* split_word = nullptr; // this publish's word of the layer's split counters
};

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Doorbell rung by the kernel itself (single-block launches): done = 0, then ready = 1, after
// every thread's writes are visible system-wide. Null words: the caller rings with stream memory
// operations after the kernel.
struct Doorbell {
    volatile std::uint32_t* done  = nullptr;
    volatile std::uint32_t* ready = nullptr;
};

// Cost (µs) of one expert with `columns` routed columns on the host / on the device.
__device__ __forceinline__ float host_cost(const HostMoeSplit& s, int columns) {
    return s.host_expert_us + s.host_column_us * static_cast<float>(columns - 1);
}

__device__ __forceinline__ float device_cost(const HostMoeSplit& s, int columns) {
    return s.device_expert_us + s.device_column_us * static_cast<float>(columns - 1);
}

// Length of the device's prefix of the n misses in split order (`columns[r]` of the r-th), see
// HostMoeSplit. Single thread.
__device__ int device_share_size(const HostMoeSplit& s, int n, const unsigned char* columns) {
    if (n <= 0) return 0;
    const int cap = min(n, min(s.max_reads, kMaxDeviceReads));
    switch (s.mode) {
    case HostMoeSplit::Mode::Cost: {
        if (n <= s.min_misses) return 0;
        float rest = 0.0F;
        for (int r = 0; r < n; ++r) rest += host_cost(s, columns[r]);
        float best = s.host_fixed_us + rest;
        float dev  = s.device_fixed_us;
        int g      = 0;
        for (int k = 1; k <= cap; ++k) {
            rest -= host_cost(s, columns[k - 1]);
            dev += device_cost(s, columns[k - 1]);
            const float host = k < n ? s.host_fixed_us + s.host_read_slowdown * rest : 0.0F;
            const float t    = fmaxf(host, dev);
            if (t < best) best = t, g = k;
        }
        return g;
    }
    default:
        return 0;
    }
}

__global__ void publish_kernel(const uint4* __restrict__ x, const std::int32_t* __restrict__ ids,
                               const float* __restrict__ alpha, Residency resident, int routes,
                               int x_vectors, std::uint32_t tokens, SlotDevice out, LogDevice log,
                               Doorbell bell) {
    // A programmatic dependent of the routing: everything below reads its outputs. No early
    // trigger: the kernels after a publish (shared expert) are ordinary launches.
    pdl::wait_for_dependencies();
    if (blockIdx.x == 0 && threadIdx.x == 0 && log.stamp != nullptr) *log.stamp = global_ns();
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < x_vectors;
         i += static_cast<int>(gridDim.x * blockDim.x)) {
        out.x[i] = x[i];
    }
    if (blockIdx.x == 0) {
        // Device share (decode calls only, routes <= kMaxRoutes): the distinct misses in split
        // order, then the prefix the device takes (HostMoeSplit).
        __shared__ std::int32_t picked[kMaxDeviceReads];
        __shared__ int picks;
        __shared__ int pick_routes;
        const auto resident_at = [&](int j, std::int32_t e) {
            if (resident.slot != nullptr) return resident.slot[j] >= 0;
            if (resident.table != nullptr)
                return e >= 0 && e < resident.experts && resident.table[e] >= 0;
            return false;
        };
        const HostMoeSplit split =
            resident.share.live != nullptr ? *resident.share.live : resident.split;
        if (split.mode != HostMoeSplit::Mode::Off) {
            __shared__ std::int32_t route_id[kMaxRoutes]; // missing id of the route, or -1
            __shared__ unsigned char first[kMaxRoutes];   // first route of its id
            __shared__ unsigned char columns[kMaxRoutes]; // routes of that id
            __shared__ unsigned char order_columns[kMaxRoutes];
            __shared__ std::int32_t order_id[kMaxRoutes]; // r-th distinct miss in split order
            __shared__ int misses;
            if (threadIdx.x == 0) misses = 0;
            for (int j = static_cast<int>(threadIdx.x); j < routes;
                 j += static_cast<int>(blockDim.x)) {
                const std::int32_t e = ids[j];
                route_id[j]          = (!resident_at(j, e) && e >= 0) ? e : -1;
            }
            __syncthreads();
            for (int j = static_cast<int>(threadIdx.x); j < routes;
                 j += static_cast<int>(blockDim.x)) {
                const std::int32_t e = route_id[j];
                int f = 0, c = 0;
                if (e >= 0) {
                    f = 1;
                    for (int i = 0; i < routes; ++i) {
                        if (route_id[i] == e) {
                            ++c;
                            if (i < j) f = 0;
                        }
                    }
                }
                first[j]   = static_cast<unsigned char>(f);
                columns[j] = static_cast<unsigned char>(c);
            }
            __syncthreads();
            for (int j = static_cast<int>(threadIdx.x); j < routes;
                 j += static_cast<int>(blockDim.x)) {
                if (first[j] == 0) continue;
                int rank = 0;
                for (int i = 0; i < routes; ++i) {
                    if (first[i] == 0 || i == j) continue;
                    rank += columns[i] > columns[j] || (columns[i] == columns[j] && i < j);
                }
                order_id[rank]      = route_id[j];
                order_columns[rank] = columns[j];
                atomicAdd(&misses, 1);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const int g = device_share_size(split, misses, order_columns);
                int r       = 0;
                for (int k = 0; k < g; ++k) picked[k] = order_id[k], r += order_columns[k];
                picks       = g;
                pick_routes = r;
            }
        } else if (threadIdx.x == 0) {
            picks       = 0;
            pick_routes = 0;
        }
        __syncthreads();
        for (int j = static_cast<int>(threadIdx.x); j < routes; j += static_cast<int>(blockDim.x)) {
            const std::int32_t e = ids[j];
            const bool missing   = !resident_at(j, e) && e >= 0;
            bool device_read     = false;
            for (int k = 0; k < picks; ++k) device_read = device_read || picked[k] == e;
            device_read = device_read && missing;
            if (out.ids != nullptr) out.ids[j] = missing && !device_read ? e : -1;
            if (out.alpha != nullptr) out.alpha[j] = alpha[j];
            if (log.ids != nullptr) log.ids[j] = e;
            if (resident.share.reads != nullptr) resident.share.reads[j] = device_read ? e : -1;
        }
        if (resident.share.stage_table != nullptr) {
            for (int e = static_cast<int>(threadIdx.x); e < resident.experts;
                 e += static_cast<int>(blockDim.x)) {
                int k = -1;
                for (int i = 0; i < picks; ++i) k = picked[i] == e ? i : k;
                resident.share.stage_table[e] = k;
            }
        }
        if (threadIdx.x == 0) {
            if (resident.share.picks != nullptr) {
                for (int k = 0; k < kMaxDeviceReads; ++k)
                    resident.share.picks[k] = k < picks ? picked[k] : -1;
                resident.share.picks[kMaxDeviceReads] = picks;
            }
            if (log.split_word != nullptr) {
                *log.split_word = static_cast<std::uint32_t>(picks) |
                                  (static_cast<std::uint32_t>(pick_routes) << 16);
            }
        }
        if (threadIdx.x == 0) {
            // Data words of the flags below: the token counts and the reset of done (the host
            // writes done only after it has seen this ready).
            if (out.tokens != nullptr) *out.tokens = tokens;
            if (log.published != nullptr) log.published[1] = log.end_token;
            if (bell.ready != nullptr) *bell.done = 0;
        }
        if (log.published != nullptr || bell.ready != nullptr) {
            // One system-wide fence of every writer (x, ids, alpha, log, the words above) before
            // the flags (uniform branch): the doorbell costs one fence, not three.
            __threadfence_system();
            __syncthreads();
            if (threadIdx.x == 0) {
                if (log.published != nullptr) log.published[0] = 1;
                if (bell.ready != nullptr) *bell.ready = 1;
            }
        }
    }
}

__global__ void stamp_kernel(unsigned long long* stamp) { *stamp = global_ns(); }

// One block: the routed ids into the log, then (after a system-wide fence of every thread's
// writes) the logged token count and the published flag, as publish_kernel does.
__global__ void __launch_bounds__(1024)
    log_routes_kernel(const std::int32_t* __restrict__ ids, int routes, LogDevice log) {
    for (int j = static_cast<int>(threadIdx.x); j < routes; j += static_cast<int>(blockDim.x)) {
        log.ids[j] = ids[j];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        log.published[1] = log.end_token;
        __threadfence_system();
        log.published[0] = 1;
    }
}

struct StagePlanes {
    const std::byte* src[4];
    std::byte* dst[4];
    long long units[4]; // copy units per expert in each plane
    int wide[4]; // 1: 16-byte units, 0: 4-byte units (a plane base or size not 16-byte aligned)
};

// Grid-stride copy of picks[0..n) experts' planes, 4 independent loads in flight per thread
// (host-mapped reads are latency bound: bytes in flight set the rate).
__global__ void stage_kernel(StagePlanes p, const std::int32_t* __restrict__ picks) {
    const int n = picks[kMaxDeviceReads];
    if (n <= 0) return;
    const long long per    = p.units[0] + p.units[1] + p.units[2] + p.units[3];
    const long long total  = static_cast<long long>(n) * per;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    constexpr int U        = 4;
    for (long long base = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         base < total; base += stride * U) {
        uint4 v[U];
        int plane[U], k[U];
        long long off[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const long long i = base + u * stride;
            k[u]              = -1;
            if (i < total) {
                k[u]        = static_cast<int>(i / per);
                long long r = i - static_cast<long long>(k[u]) * per;
                int q       = 0;
                while (r >= p.units[q]) r -= p.units[q++];
                plane[u]             = q;
                const long long unit = static_cast<long long>(picks[k[u]]) * p.units[q] + r;
                off[u]               = static_cast<long long>(k[u]) * p.units[q] + r;
                if (p.wide[q]) {
                    v[u] = reinterpret_cast<const uint4*>(p.src[q])[unit];
                } else {
                    v[u].x = reinterpret_cast<const std::uint32_t*>(p.src[q])[unit];
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (k[u] < 0) continue;
            if (p.wide[plane[u]]) {
                reinterpret_cast<uint4*>(p.dst[plane[u]])[off[u]] = v[u];
            } else {
                reinterpret_cast<std::uint32_t*>(p.dst[plane[u]])[off[u]] = v[u].x;
            }
        }
    }
}

// The partial is read with ld.global.cv: it was written by the CPU after this graph was launched.
// One read of the host word per block: every thread reading it costs one PCIe round trip per warp
// (measured ~20 us for 40 warps at T = 4).
__device__ __forceinline__ bool host_partial_present(const std::uint32_t* active) {
    __shared__ std::uint32_t present;
    if (threadIdx.x == 0) present = __ldcv(active);
    __syncthreads();
    return present != 0;
}

__global__ void accumulate_bf16_kernel(__nv_bfloat16* __restrict__ y, const float* partial,
                                       const std::uint32_t* active, int vectors,
                                       unsigned long long* stamp) {
    if (stamp != nullptr && blockIdx.x == 0 && threadIdx.x == 0) *stamp = global_ns();
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (!host_partial_present(active)) return;
    if (i >= vectors) return;
    const float4 a   = __ldcv(reinterpret_cast<const float4*>(partial) + 2 * i);
    const float4 b   = __ldcv(reinterpret_cast<const float4*>(partial) + 2 * i + 1);
    uint4 raw        = reinterpret_cast<uint4*>(y)[i];
    auto* h          = reinterpret_cast<__nv_bfloat162*>(&raw);
    const float p[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 v = __bfloat1622float2(h[k]);
        h[k]           = __floats2bfloat162_rn(v.x + p[2 * k], v.y + p[2 * k + 1]);
    }
    reinterpret_cast<uint4*>(y)[i] = raw;
}

// y += device_partial (when given), then y += the host partial (when the host had experts).
__global__ void accumulate_f32_kernel(float* __restrict__ y, const float* partial,
                                      const std::uint32_t* active, int vectors,
                                      unsigned long long* stamp,
                                      const float* __restrict__ device_partial, float* echo,
                                      float* echo_y) {
    if (stamp != nullptr && blockIdx.x == 0 && threadIdx.x == 0) *stamp = global_ns();
    const int i     = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool host = host_partial_present(active);
    if (!host && device_partial == nullptr && echo_y == nullptr) return;
    if (i >= vectors) return;
    float4 v = reinterpret_cast<float4*>(y)[i];
    if (echo_y != nullptr) reinterpret_cast<float4*>(echo_y)[i] = v; // the device's own partial
    if (!host && device_partial == nullptr) return;
    if (device_partial != nullptr) {
        const float4 d = reinterpret_cast<const float4*>(device_partial)[i];
        v.x += d.x;
        v.y += d.y;
        v.z += d.z;
        v.w += d.w;
    }
    if (host) {
        const float4 a = __ldcv(reinterpret_cast<const float4*>(partial) + i);
        if (echo != nullptr) reinterpret_cast<float4*>(echo)[i] = a; // what the device read
        v.x += a.x;
        v.y += a.y;
        v.z += a.z;
        v.w += a.w;
    }
    reinterpret_cast<float4*>(y)[i] = v;
}

} // namespace

void stage_device_share(cudaStream_t stream, const std::int32_t* picks,
                        const std::array<const std::byte*, 4>& bank,
                        const std::array<std::byte*, 4>& staging,
                        const std::array<std::int64_t, 4>& plane_bytes, int blocks) {
    StagePlanes p{};
    const auto aligned = [](const void* ptr, std::uintptr_t a) {
        return reinterpret_cast<std::uintptr_t>(ptr) % a == 0;
    };
    for (int q = 0; q < 4; ++q) {
        if (bank[q] == nullptr || staging[q] == nullptr || plane_bytes[q] <= 0 ||
            plane_bytes[q] % 4 != 0 || !aligned(bank[q], 4) || !aligned(staging[q], 4)) {
            throw std::invalid_argument(
                "stage_device_share: planes must be 4-byte aligned multiples of 4");
        }
        const bool wide =
            plane_bytes[q] % 16 == 0 && aligned(bank[q], 16) && aligned(staging[q], 16);
        p.src[q]   = bank[q];
        p.dst[q]   = staging[q];
        p.wide[q]  = wide ? 1 : 0;
        p.units[q] = plane_bytes[q] / (wide ? 16 : 4);
    }
    if (picks == nullptr || blocks < 1)
        throw std::invalid_argument("stage_device_share: picks and blocks >= 1");
    stage_kernel<<<blocks, 256, 0, stream>>>(p, picks);
    CUDA_CHECK(cudaGetLastError());
}

std::int64_t HostMoeMailbox::now_ns() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

HostMoeMailbox::HostMoeMailbox(HostMoeMailboxConfig config, std::vector<HostExpertLayer> layers)
    : config_(std::move(config)), layers_(std::move(layers)) {
    if (const char* v = std::getenv("NINFER_HOST_MOE_A4"); v != nullptr && *v != 0) {
        config_.executor.a4 = std::atoi(v) != 0; // CPU with the GPU's W4A4 activation codec
    }
    if (const char* v = std::getenv("NINFER_HOST_MOE_CPUS"); v != nullptr && *v != 0) {
        // Issue #11: "1-31" used to read as cpus 1 and -31. Ranges as taskset writes them.
        try {
            config_.cpus = parse_cpu_list(v);
        } catch (const std::invalid_argument& error) {
            throw std::invalid_argument(std::string("NINFER_HOST_MOE_CPUS: ") + error.what());
        }
        // A cpu outside the affinity used to abort in a worker thread ("cannot pin a thread").
        for (const int cpu : config_.cpus) {
            if (!process_may_run_on(cpu)) {
                throw std::invalid_argument("NINFER_HOST_MOE_CPUS: cpu " + std::to_string(cpu) +
                                            " does not exist or is outside the CPUs this process "
                                            "may run on (taskset, cgroup cpuset)");
            }
        }
    }
    const auto& c = config_;
    if (c.layers <= 0 || static_cast<std::size_t>(c.layers) != layers_.size() || c.max_tokens < 1 ||
        c.top_k < 1 || c.top_k > 16 || c.hidden <= 0 || c.hidden % 8 || c.cpus.empty()) {
        throw std::invalid_argument("host MoE mailbox: invalid configuration");
    }
    const auto T = static_cast<std::size_t>(c.max_tokens);
    const auto H = static_cast<std::size_t>(c.hidden);
    const auto R = T * static_cast<std::size_t>(c.top_k);
    // Flags: two lines per layer (GPU-written ready + tokens, CPU-written done + active). Data:
    // one routing / x / partial buffer for all layers (one layer is in flight at a time). The
    // whole mailbox spans a few pages: the GPU translates mapped host pages like any other, and a
    // per-layer data slot (124 KB each) cost ~10 us of doorbell latency per layer in TLB misses.
    ids_off_   = align_up(2 * kLine * static_cast<std::size_t>(c.layers), kLine);
    alpha_off_ = align_up(ids_off_ + R * sizeof(std::int32_t), kLine);
    x_off_     = align_up(alpha_off_ + R * sizeof(float), kLine);
    out_off_   = align_up(x_off_ + T * H * sizeof(std::uint16_t), kLine);
    // Routing log and stamps after the data, on their own pages (the host reads them after the
    // device wrote them; they never share a line with the doorbell flags).
    std::size_t end = align_up(out_off_ + T * H * sizeof(float), 4096);
    if (c.route_log_tokens < 0) throw std::invalid_argument("host MoE mailbox: invalid route log");
    log_off_    = end;
    log_stride_ = align_up(kLine + static_cast<std::size_t>(c.route_log_tokens) *
                                       static_cast<std::size_t>(c.top_k) * sizeof(std::int32_t),
                           kLine);
    if (c.route_log_tokens > 0)
        end = align_up(log_off_ + log_stride_ * static_cast<std::size_t>(c.layers), 4096);
    stamp_off_ = end;
    if (c.gpu_stamps)
        end = align_up(stamp_off_ +
                           kStampCount * sizeof(std::uint64_t) * static_cast<std::size_t>(c.layers),
                       kLine);
    clock_off_              = end;
    const std::size_t bytes = align_up(clock_off_ + kLine, 4096);

    CUDA_CHECK(cudaSetDevice(c.device));
    void* host = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    host_ = static_cast<std::byte*>(host);
    std::memset(host_, 0, bytes);
    void* dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dev, host, 0));
    device_ = static_cast<std::byte*>(dev);
    if (device_ != host_) {
        cudaFreeHost(host_);
        throw std::runtime_error("host MoE mailbox: mapped memory without unified addressing");
    }

    // Functional probe of stream memory operations on this device and driver.
    {
        cudaStream_t s = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        const SlotView v = slot(0);
        cu_check(cuStreamWriteValue32(reinterpret_cast<CUstream>(s), device_word(v.done), 7, 0),
                 "probe write");
        cu_check(cuStreamWaitValue32(reinterpret_cast<CUstream>(s), device_word(v.done), 7,
                                     CU_STREAM_WAIT_VALUE_GEQ),
                 "probe wait");
        cu_check(cuStreamWriteValue32(reinterpret_cast<CUstream>(s), device_word(v.done), 0, 0),
                 "probe reset");
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaStreamDestroy(s));
    }

    trace_.resize(static_cast<std::size_t>(c.trace_capacity));
    served_ = std::make_unique<std::atomic<std::uint32_t>[]>(static_cast<std::size_t>(c.layers));
    if (const char* path = std::getenv("NINFER_HOST_MOE_CALL_LOG"); path != nullptr && *path != 0) {
        call_log_path_ = path;
        call_log_.reserve(std::size_t{4} << 20);
        config_.executor.record_members = true;
    }
    if (const char* path = std::getenv("NINFER_HOST_MOE_DUMP"); path != nullptr && *path != 0) {
        dump_path_ = path;
        dump_max_  = 40000;
        if (const char* v = std::getenv("NINFER_HOST_MOE_DUMP_MAX"); v != nullptr && *v != 0)
            dump_max_ = std::atoll(v);
        const std::size_t echo_bytes = 2 * sizeof(float) * static_cast<std::size_t>(c.layers) *
                                       kHostMaxTokens * static_cast<std::size_t>(c.hidden);
        void* e = nullptr;
        CUDA_CHECK(cudaHostAlloc(&e, echo_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        std::memset(e, 0xFF, echo_bytes);
        echo_ = static_cast<float*>(e);
        dump_.reserve(std::size_t{256} << 20);
    }
    service_ = std::thread([this] { service(); });
    while (started_.load(std::memory_order_acquire) == 0) std::this_thread::yield();
    if (started_.load() < 0) {
        service_.join();
        cudaFreeHost(host_);
        throw std::runtime_error("host MoE mailbox: " + error());
    }
}

HostMoeMailbox::~HostMoeMailbox() {
    stop_.store(true, std::memory_order_release);
    release();
    if (service_.joinable()) service_.join();
    if (!dump_path_.empty()) {
        cudaDeviceSynchronize(); // every accumulate has written its echo
        dump_fill_echo();
        if (std::FILE* f = std::fopen(dump_path_.c_str(), "wb")) {
            std::fwrite(dump_.data(), 1, dump_.size(), f);
            std::fclose(f);
        }
        std::fprintf(stderr, "host MoE dump: %lld records, %zu bytes -> %s\n",
                     static_cast<long long>(dump_records_), dump_.size(), dump_path_.c_str());
    }
    if (echo_) cudaFreeHost(echo_);
    if (host_) cudaFreeHost(host_);
    if (!call_log_path_.empty()) {
        if (std::FILE* f = std::fopen(call_log_path_.c_str(), "a")) {
            std::fprintf(f, "layer,tokens,experts,pairs,seen_ns,done_ns,dispatch_ns,gate_end_ns,"
                            "down_end_ns,end_ns,"
                            "wake_max_ns,gate_min_ns,gate_max_ns,down_min_ns,down_max_ns\n");
            for (const HostMoeCallRecord& r : call_log_) {
                std::fprintf(
                    f, "%d,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n",
                    r.layer, r.tokens, r.experts, r.pairs, static_cast<long long>(r.seen_ns),
                    static_cast<long long>(r.done_ns), static_cast<long long>(r.dispatch_ns),
                    static_cast<long long>(r.gate_end_ns), static_cast<long long>(r.down_end_ns),
                    static_cast<long long>(r.end_ns), static_cast<long long>(r.wake_max_ns),
                    static_cast<long long>(r.gate_min_ns), static_cast<long long>(r.gate_max_ns),
                    static_cast<long long>(r.down_min_ns), static_cast<long long>(r.down_max_ns));
            }
            std::fclose(f);
        }
    }
}

HostMoeMailbox::SlotView HostMoeMailbox::slot(std::int32_t layer) const noexcept {
    std::byte* f = host_ + 2 * kLine * static_cast<std::size_t>(layer);
    SlotView v;
    v.ready  = reinterpret_cast<volatile std::uint32_t*>(f);
    v.tokens = reinterpret_cast<volatile std::uint32_t*>(f + 4);
    v.done   = reinterpret_cast<volatile std::uint32_t*>(f + kLine);
    v.active = reinterpret_cast<volatile std::uint32_t*>(f + kLine + 4);
    v.ids    = reinterpret_cast<std::int32_t*>(host_ + ids_off_);
    v.alpha  = reinterpret_cast<float*>(host_ + alpha_off_);
    v.x      = reinterpret_cast<std::uint16_t*>(host_ + x_off_);
    v.out    = reinterpret_cast<float*>(host_ + out_off_);
    return v;
}

std::string HostMoeMailbox::error() const {
    const std::lock_guard lock(error_mutex_);
    return error_;
}

void HostMoeMailbox::release() noexcept {
    for (std::int32_t l = 0; l < config_.layers; ++l) store_release(slot(l).done, kReleased);
}

std::vector<HostMoeTraceRecord> HostMoeMailbox::trace() const {
    const auto n = trace_size_.load(std::memory_order_acquire);
    return {trace_.begin(), trace_.begin() + n};
}

void HostMoeMailbox::clear_trace() noexcept { trace_size_.store(0, std::memory_order_release); }

void HostMoeMailbox::publish(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                             const __nv_bfloat16* x, const std::int32_t* ids, const float* alpha,
                             const std::int32_t* device_slot) const {
    publish_impl(stream, layer, tokens, x, ids, alpha,
                 Residency{device_slot, nullptr, 0, HostMoeSplit{}, HostMoeDeviceShare{}}, -1);
}

void HostMoeMailbox::publish_table(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                                   const __nv_bfloat16* x, const std::int32_t* ids,
                                   const float* alpha, const std::int32_t* slot_table,
                                   std::int32_t experts, std::int32_t log_offset,
                                   const HostMoeSplit& split,
                                   const HostMoeDeviceShare& share) const {
    if (slot_table == nullptr || experts <= 0) {
        throw std::invalid_argument("host MoE mailbox: publish_table needs a slot table");
    }
    if (config_.route_log_tokens > 0 &&
        (log_offset < 0 || log_offset + tokens > config_.route_log_tokens)) {
        throw std::invalid_argument("host MoE mailbox: routing log offset out of range");
    }
    const bool splitting = split.mode != HostMoeSplit::Mode::Off;
    if (splitting && (split.mode != HostMoeSplit::Mode::Cost || share.reads == nullptr ||
                      tokens * config_.top_k > kMaxRoutes || split.max_reads < 0 ||
                      split.max_reads > kMaxDeviceReads || split.min_misses < 0 ||
                      !(split.host_read_slowdown >= 1.0F))) {
        throw std::invalid_argument("host MoE mailbox: a split needs a reads output, <= 8 tokens, "
                                    "max_reads in [0,16], slowdown >= 1");
    }
    publish_impl(
        stream, layer, tokens, x, ids, alpha,
        Residency{nullptr, slot_table, experts, split, splitting ? share : HostMoeDeviceShare{}},
        config_.route_log_tokens > 0 ? log_offset : -1);
}

void HostMoeMailbox::publish_impl(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                                  const __nv_bfloat16* x, const std::int32_t* ids,
                                  const float* alpha, const HostMoeResidency& resident,
                                  std::int32_t log_offset) const {
    if (layer < 0 || layer >= config_.layers || tokens < 1 || tokens > config_.max_tokens) {
        throw std::invalid_argument("host MoE mailbox: publish out of range");
    }
    const SlotView v       = slot(layer);
    const int routes       = tokens * config_.top_k;
    const int x_vectors    = tokens * config_.hidden / 8;
    constexpr int kThreads = 256;
    // Decode-sized calls (x of <= 8 rows) in one block that rings the doorbell itself; wider calls
    // (prefill chunks) over many blocks, the doorbell then by two stream memory writes.
    const bool kernel_bell = tokens <= kHostMaxTokens;
    const int blocks       = kernel_bell ? 1 : (x_vectors + kThreads - 1) / kThreads;
    const SlotDevice out{const_cast<std::uint32_t*>(v.tokens), v.ids, v.alpha,
                         reinterpret_cast<uint4*>(v.x)};
    LogDevice log;
    if (log_offset >= 0) {
        std::byte* base = host_ + log_off_ + log_stride_ * static_cast<std::size_t>(layer);
        log.published   = reinterpret_cast<std::uint32_t*>(base);
        log.ids         = reinterpret_cast<std::int32_t*>(base + kLine) +
                  static_cast<std::size_t>(log_offset) * static_cast<std::size_t>(config_.top_k);
        log.end_token = static_cast<std::uint32_t>(log_offset + tokens);
        if (resident.split.mode != HostMoeSplit::Mode::Off) {
            log.split_word = log.published + 4 + log_offset % kSplitWords;
        }
    }
    if (config_.gpu_stamps) {
        log.stamp =
            reinterpret_cast<unsigned long long*>(const_cast<std::uint64_t*>(gpu_stamps(layer))) +
            kStampPublish;
    }
    Doorbell bell;
    if (kernel_bell) bell = Doorbell{v.done, v.ready};
    CUDA_CHECK(pdl::launch_dependent({dim3(blocks), dim3(kThreads), 0, stream}, publish_kernel,
                                     reinterpret_cast<const uint4*>(x), ids, alpha, resident,
                                     routes, x_vectors, static_cast<std::uint32_t>(tokens), out,
                                     log, bell));
    if (kernel_bell) return;
    const auto s = reinterpret_cast<CUstream>(stream);
    // done before ready: the service only writes done after it has seen this ready.
    cu_check(cuStreamWriteValue32(s, device_word(v.done), 0, CU_STREAM_WRITE_VALUE_DEFAULT),
             "reset done");
    cu_check(cuStreamWriteValue32(s, device_word(v.ready), 1, CU_STREAM_WRITE_VALUE_DEFAULT),
             "doorbell");
}

void consume_host_flag(cudaStream_t stream, std::uint32_t* word) {
    if (word == nullptr) throw std::invalid_argument("consume_host_flag: no word");
    const auto s = reinterpret_cast<CUstream>(stream);
    cu_check(cuStreamWaitValue32(s, device_word(word), 1, CU_STREAM_WAIT_VALUE_GEQ),
             "wait host flag");
    cu_check(cuStreamWriteValue32(s, device_word(word), 0, CU_STREAM_WRITE_VALUE_DEFAULT),
             "lower host flag");
}

void raise_host_flag(std::uint32_t* word) noexcept { store_release(word, 1); }

void lower_host_flag(std::uint32_t* word) noexcept { store_release(word, 0); }

void select_device_share(cudaStream_t stream, std::int32_t tokens, std::int32_t top_k,
                         const std::int32_t* ids, const std::int32_t* slot_table,
                         std::int32_t experts, const HostMoeSplit& split,
                         const HostMoeDeviceShare& share) {
    if (split.mode == HostMoeSplit::Mode::Off || share.reads == nullptr || slot_table == nullptr ||
        tokens < 1 || tokens * top_k > kMaxRoutes) {
        throw std::invalid_argument(
            "select_device_share: a split mode, reads, a table and <= 8 tokens");
    }
    // The publish kernel's block 0 without any mailbox output: no x, no host ids, no log, no bell.
    const SlotDevice out{nullptr, nullptr, nullptr, nullptr};
    publish_kernel<<<1, 256, 0, stream>>>(
        nullptr, ids, nullptr, Residency{nullptr, slot_table, experts, split, share},
        tokens * top_k, 0, static_cast<std::uint32_t>(tokens), out, LogDevice{}, Doorbell{});
    CUDA_CHECK(cudaGetLastError());
}

void HostMoeMailbox::log_routes(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                                const std::int32_t* ids, std::int32_t log_offset) const {
    if (config_.route_log_tokens <= 0 || layer < 0 || layer >= config_.layers || tokens < 1 ||
        log_offset < 0 || log_offset + tokens > config_.route_log_tokens) {
        throw std::invalid_argument("host MoE mailbox: log_routes out of range or without a log");
    }
    std::byte* base = host_ + log_off_ + log_stride_ * static_cast<std::size_t>(layer);
    LogDevice log;
    log.published = reinterpret_cast<std::uint32_t*>(base);
    log.ids       = reinterpret_cast<std::int32_t*>(base + kLine) +
              static_cast<std::size_t>(log_offset) * static_cast<std::size_t>(config_.top_k);
    log.end_token = static_cast<std::uint32_t>(log_offset + tokens);
    log_routes_kernel<<<1, 1024, 0, stream>>>(ids, tokens * config_.top_k, log);
    CUDA_CHECK(cudaGetLastError());
}

void HostMoeMailbox::accumulate(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                                __nv_bfloat16* y) const {
    if (layer < 0 || layer >= config_.layers || tokens < 1 || tokens > config_.max_tokens) {
        throw std::invalid_argument("host MoE mailbox: accumulate out of range");
    }
    const SlotView v = slot(layer);
    cu_check(cuStreamWaitValue32(reinterpret_cast<CUstream>(stream), device_word(v.done), 1,
                                 CU_STREAM_WAIT_VALUE_GEQ),
             "wait done");
    const int vectors      = tokens * config_.hidden / 8;
    constexpr int kThreads = 256;
    accumulate_bf16_kernel<<<(vectors + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        y, v.out, const_cast<const std::uint32_t*>(v.active), vectors, accumulate_stamp(layer));
    CUDA_CHECK(cudaGetLastError());
}

void HostMoeMailbox::accumulate(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                                float* y, const float* device_partial) const {
    if (layer < 0 || layer >= config_.layers || tokens < 1 || tokens > config_.max_tokens) {
        throw std::invalid_argument("host MoE mailbox: accumulate out of range");
    }
    const SlotView v = slot(layer);
    cu_check(cuStreamWaitValue32(reinterpret_cast<CUstream>(stream), device_word(v.done), 1,
                                 CU_STREAM_WAIT_VALUE_GEQ),
             "wait done");
    const int vectors      = tokens * config_.hidden / 4;
    constexpr int kThreads = 256;
    accumulate_f32_kernel<<<(vectors + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        y, v.out, const_cast<const std::uint32_t*>(v.active), vectors, accumulate_stamp(layer),
        device_partial,
        echo_ != nullptr && tokens <= kHostMaxTokens
            ? echo_ + static_cast<std::size_t>(2 * layer) * kHostMaxTokens *
                          static_cast<std::size_t>(config_.hidden)
            : nullptr,
        echo_ != nullptr && tokens <= kHostMaxTokens
            ? echo_ + static_cast<std::size_t>(2 * layer + 1) * kHostMaxTokens *
                          static_cast<std::size_t>(config_.hidden)
            : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

unsigned long long* HostMoeMailbox::accumulate_stamp(std::int32_t layer) const noexcept {
    if (!config_.gpu_stamps) return nullptr;
    return reinterpret_cast<unsigned long long*>(const_cast<std::uint64_t*>(gpu_stamps(layer))) +
           kStampAccumulate;
}

void HostMoeMailbox::stamp(cudaStream_t stream, std::int32_t layer, std::int32_t index) const {
    if (!config_.gpu_stamps) return;
    if (layer < 0 || layer >= config_.layers || index < 0 || index >= kStampCount) {
        throw std::invalid_argument("host MoE mailbox: stamp out of range");
    }
    stamp_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(const_cast<std::uint64_t*>(gpu_stamps(layer))) +
        index);
    CUDA_CHECK(cudaGetLastError());
}

HostMoeRouteLog HostMoeMailbox::route_log(std::int32_t layer) const noexcept {
    if (config_.route_log_tokens <= 0 || layer < 0 || layer >= config_.layers) return {};
    std::byte* base = host_ + log_off_ + log_stride_ * static_cast<std::size_t>(layer);
    HostMoeRouteLog log;
    log.published = reinterpret_cast<volatile std::uint32_t*>(base);
    log.tokens    = reinterpret_cast<volatile std::uint32_t*>(base) + 1;
    log.ids       = reinterpret_cast<const std::int32_t*>(base + kLine);
    log.split     = reinterpret_cast<volatile std::uint32_t*>(base) + 4;
    return log;
}

std::int64_t HostMoeMailbox::device_share_experts(std::int32_t layer) const noexcept {
    const HostMoeRouteLog log = route_log(layer);
    if (log.split == nullptr) return 0;
    std::int64_t n = 0;
    for (int i = 0; i < kSplitWords; ++i) n += log.split[i] & 0xFFFFU;
    return n;
}

std::int64_t HostMoeMailbox::device_share_routes(std::int32_t layer) const noexcept {
    const HostMoeRouteLog log = route_log(layer);
    if (log.split == nullptr) return 0;
    std::int64_t n = 0;
    for (int i = 0; i < kSplitWords; ++i) n += log.split[i] >> 16;
    return n;
}

void HostMoeMailbox::collect_cost_samples(std::int64_t max_samples) {
    const std::lock_guard lock(cost_mutex_);
    cost_samples_.clear();
    cost_samples_.reserve(static_cast<std::size_t>(std::max<std::int64_t>(max_samples, 0)));
    cost_capacity_.store(std::max<std::int64_t>(max_samples, 0), std::memory_order_relaxed);
}

std::vector<HostMoeCostSample> HostMoeMailbox::cost_samples() const {
    const std::lock_guard lock(cost_mutex_);
    return cost_samples_;
}

namespace {
// Least squares of us on (1, experts, pairs - experts) over `s` (weights 0/1 in `keep`), or on
// (1, experts) with the column term fixed at `column`. Returns {fixed, expert, column}; false when
// singular.
bool solve_costs(std::span<const HostMoeCostSample> s, const std::vector<char>& keep,
                 bool with_column, double column, std::array<double, 3>& out) {
    const int n    = with_column ? 3 : 2;
    double a[3][4] = {};
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (keep[i] == 0) continue;
        const double c    = static_cast<double>(s[i].pairs - s[i].experts);
        const double x[3] = {1.0, static_cast<double>(s[i].experts), c};
        const double y    = static_cast<double>(s[i].us) - (with_column ? 0.0 : column * c);
        for (int r = 0; r < n; ++r) {
            for (int k = 0; k < n; ++k) a[r][k] += x[r] * x[k];
            a[r][3] += x[r] * y;
        }
    }
    for (int col = 0; col < n; ++col) { // Gauss-Jordan with partial pivoting
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
        if (std::fabs(a[piv][col]) < 1e-9) return false;
        for (int k = 0; k < 4; ++k) std::swap(a[col][k], a[piv][k]);
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = a[r][col] / a[col][col];
            for (int k = col; k < 4; ++k) a[r][k] -= f * a[col][k];
        }
    }
    out = {a[0][3] / a[0][0], a[1][3] / a[1][1], with_column ? a[2][3] / a[2][2] : column};
    return true;
}
} // namespace

HostMoeCostFit fit_host_costs(std::span<const HostMoeCostSample> samples, float default_column_us) {
    HostMoeCostFit fit;
    if (samples.size() < 16) return fit;
    // The column term is identifiable only if calls vary in columns per expert.
    double c_min = 1e30, c_max = -1e30;
    for (const auto& s : samples) {
        const double c = static_cast<double>(s.pairs - s.experts) / std::max(s.experts, 1);
        c_min = std::min(c_min, c), c_max = std::max(c_max, c);
    }
    const bool with_column = c_max - c_min >= 0.25;
    std::vector<char> keep(samples.size(), 1);
    std::array<double, 3> k{};
    const auto residual = [&](const HostMoeCostSample& s) {
        return static_cast<double>(s.us) -
               (k[0] + k[1] * s.experts + k[2] * static_cast<double>(s.pairs - s.experts));
    };
    for (int pass = 0; pass < 2; ++pass) {
        if (!solve_costs(samples, keep, with_column, default_column_us, k)) return HostMoeCostFit{};
        if (pass == 1) break;
        std::vector<double> r;
        r.reserve(samples.size());
        for (const auto& s : samples) r.push_back(std::fabs(residual(s)));
        std::vector<double> sorted = r;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        const double mad = std::max(sorted[sorted.size() / 2], 0.5);
        for (std::size_t i = 0; i < samples.size(); ++i) keep[i] = r[i] <= 4.0 * mad ? 1 : 0;
    }
    double ss         = 0.0;
    std::int32_t kept = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (keep[i] == 0) continue;
        const double e = residual(samples[i]);
        ss += e * e, ++kept;
    }
    fit.samples   = kept;
    fit.fixed_us  = static_cast<float>(std::max(k[0], 0.0));
    fit.expert_us = static_cast<float>(std::max(k[1], 0.0));
    fit.column_us = static_cast<float>(std::max(k[2], 0.0));
    fit.rms_us    = static_cast<float>(std::sqrt(ss / std::max(kept, 1)));
    return fit;
}

void HostMoeMailbox::clear_route_flags() noexcept {
    if (config_.route_log_tokens <= 0) return;
    for (std::int32_t l = 0; l < config_.layers; ++l) {
        auto* w = reinterpret_cast<volatile std::uint32_t*>(
            host_ + log_off_ + log_stride_ * static_cast<std::size_t>(l));
        w[0] = 0;
        w[1] = 0;
        for (int i = 0; i < kSplitWords; ++i) w[4 + i] = 0;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

const volatile std::uint64_t* HostMoeMailbox::gpu_stamps(std::int32_t layer) const noexcept {
    if (!config_.gpu_stamps || layer < 0 || layer >= config_.layers) return nullptr;
    return reinterpret_cast<const volatile std::uint64_t*>(host_ + stamp_off_) +
           static_cast<std::size_t>(layer) * kStampCount;
}

namespace {
__global__ void clock_probe_kernel(unsigned long long* word) {
    __threadfence_system();
    *reinterpret_cast<volatile unsigned long long*>(word) = global_ns();
    __threadfence_system();
}
} // namespace

std::int64_t HostMoeMailbox::calibrate_clock(cudaStream_t stream, int rounds) const {
    auto* word        = reinterpret_cast<volatile std::uint64_t*>(host_ + clock_off_);
    std::int64_t best = std::numeric_limits<std::int64_t>::max();
    for (int r = 0; r < rounds; ++r) {
        *word = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        clock_probe_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<unsigned long long*>(const_cast<std::uint64_t*>(word)));
        CUDA_CHECK(cudaGetLastError());
        std::uint64_t g     = 0;
        const auto deadline = now_ns() + 1000000000LL;
        while ((g = *word) == 0 && now_ns() < deadline) _mm_pause();
        const std::int64_t host = now_ns();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (g != 0) best = std::min(best, host - static_cast<std::int64_t>(g));
    }
    if (best == std::numeric_limits<std::int64_t>::max()) {
        throw std::runtime_error("host MoE mailbox: clock probe never observed");
    }
    return best;
}

void HostMoeMailbox::serve(HostMoeExecutor& executor, std::int32_t layer) {
    const SlotView v = slot(layer);
    HostMoeTraceRecord r;
    r.seen_ns = now_ns();
    *v.ready  = 0;
    r.layer   = layer;
    r.tokens  = static_cast<std::int32_t>(load_acquire(v.tokens));
    // The previous dumped call's accumulate ran before this publish (stream order).
    if (!dump_path_.empty()) dump_fill_echo();
    const std::int32_t routes = r.tokens * config_.top_k;
    bool any                  = false;
    for (std::int32_t i = 0; i < routes && !any; ++i) any = v.ids[i] >= 0;
    HostMoeCallStats st;
    if (any) {
        const HostMoeJob job{r.tokens, config_.top_k, v.x, v.ids, v.alpha, v.out};
        st        = executor.run(layers_[static_cast<std::size_t>(layer)], job);
        r.experts = st.experts;
        r.pairs   = st.pairs;
        served_experts_.fetch_add(st.experts, std::memory_order_relaxed);
        served_calls_.fetch_add(1, std::memory_order_relaxed);
        if (r.tokens <= kHostMaxTokens && st.experts > 0 &&
            cost_capacity_.load(std::memory_order_relaxed) > 0) {
            const std::lock_guard lock(cost_mutex_);
            if (static_cast<std::int64_t>(cost_samples_.size()) <
                cost_capacity_.load(std::memory_order_relaxed)) {
                cost_samples_.push_back(
                    {st.experts, st.pairs, 1e-3F * static_cast<float>(st.end_ns)});
            }
        }
    }
    if (!dump_path_.empty() && r.tokens <= kHostMaxTokens && dump_records_ < dump_max_) {
        if (!any)
            std::memset(v.out, 0,
                        sizeof(float) * static_cast<std::size_t>(r.tokens) * config_.hidden);
        dump_record(layer, r.tokens, r.experts);
    }
    *v.active    = static_cast<std::uint32_t>(r.experts);
    r.done_ns    = now_ns();
    const auto n = trace_size_.load(std::memory_order_relaxed);
    if (n < static_cast<std::int64_t>(trace_.size())) {
        trace_[static_cast<std::size_t>(n)] = r;
        trace_size_.store(n + 1, std::memory_order_release);
    }
    store_release(v.done, 1);
    served_[static_cast<std::size_t>(layer)].fetch_add(1, std::memory_order_release);
    if (!call_log_path_.empty() && call_log_.size() < call_log_.capacity()) {
        HostMoeCallRecord c;
        c.layer = r.layer, c.tokens = r.tokens, c.experts = r.experts, c.pairs = r.pairs;
        c.seen_ns = r.seen_ns, c.done_ns = r.done_ns;
        c.dispatch_ns = st.dispatch_ns, c.gate_end_ns = st.phase_ns[1],
        c.down_end_ns = st.phase_ns[2];
        c.end_ns      = st.end_ns;
        if (st.members != nullptr) {
            std::int64_t gmin = std::numeric_limits<std::int64_t>::max(), dmin = gmin, gmax = 0,
                         dmax = 0, wake = 0;
            for (std::int32_t m = 0; m < st.member_count; ++m) {
                const auto& ns = st.members[m].ns;
                wake           = std::max(wake, ns[0] - st.dispatch_ns);
                gmin = std::min(gmin, ns[3] - ns[2]), gmax = std::max(gmax, ns[3] - ns[2]);
                dmin = std::min(dmin, ns[5] - ns[4]), dmax = std::max(dmax, ns[5] - ns[4]);
            }
            c.wake_max_ns = wake, c.gate_min_ns = gmin, c.gate_max_ns = gmax, c.down_min_ns = dmin,
            c.down_max_ns = dmax;
        }
        call_log_.push_back(c);
    }
}

// Record: u32 'T98R', i32 layer, tokens, top_k, hidden, experts, i64 serial, i32 ids[T*K] (host),
// f32 alpha[T*K], i32 routed[T*K] (all routed ids, from the routing log; -2 without a log),
// u16 x[T*H], f32 out[T*H] (0 without host experts), f32 echo[T*H] (host partial as the device
// read it), f32 ydev[T*H] (the device partial before the host one was added), u32 echo_valid.
void HostMoeMailbox::dump_record(std::int32_t layer, std::int32_t tokens, std::int32_t experts) {
    const SlotView v = slot(layer);
    const std::size_t R =
        static_cast<std::size_t>(tokens) * static_cast<std::size_t>(config_.top_k);
    const std::size_t X =
        static_cast<std::size_t>(tokens) * static_cast<std::size_t>(config_.hidden);
    const auto put = [&](const void* p, std::size_t n) {
        const auto* b = static_cast<const std::byte*>(p);
        dump_.insert(dump_.end(), b, b + n);
    };
    const std::uint32_t magic  = 0x52383954U;
    const std::int32_t head[5] = {layer, tokens, config_.top_k, config_.hidden, experts};
    put(&magic, 4);
    put(head, sizeof head);
    put(&dump_serial_, 8);
    put(v.ids, 4 * R);
    put(v.alpha, 4 * R);
    if (config_.route_log_tokens > 0) {
        put(route_log(layer).ids, 4 * R); // decode calls of the Text layers log at offset 0
    } else {
        const std::vector<std::int32_t> none(R, -2);
        put(none.data(), 4 * R);
    }
    put(v.x, 2 * X);
    put(v.out, 4 * X);
    dump_pending_        = static_cast<std::int64_t>(dump_.size());
    dump_pending_layer_  = layer;
    dump_pending_tokens_ = tokens;
    dump_.resize(dump_.size() + 8 * X + 4, std::byte{0});
    ++dump_records_;
    ++dump_serial_;
}

void HostMoeMailbox::dump_fill_echo() noexcept {
    if (dump_pending_ < 0) return;
    const std::size_t X =
        static_cast<std::size_t>(dump_pending_tokens_) * static_cast<std::size_t>(config_.hidden);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const std::size_t per = kHostMaxTokens * static_cast<std::size_t>(config_.hidden);
    float* e              = echo_ + static_cast<std::size_t>(2 * dump_pending_layer_) * per;
    std::memcpy(dump_.data() + dump_pending_, e, 4 * X);
    std::memcpy(dump_.data() + dump_pending_ + 4 * X, e + per, 4 * X);
    std::uint32_t valid = 0;
    std::memcpy(&valid, e, 4);
    valid = valid != 0xFFFFFFFFU ? 1U : 0U; // the device echoes the host partial only when present
    std::memcpy(dump_.data() + dump_pending_ + 8 * X, &valid, 4);
    // Poison both so a later call that the device does not echo cannot pass as matching.
    std::memset(e, 0xFF, 4 * X);
    std::memset(e + per, 0xFF, 4 * X);
    dump_pending_ = -1;
}

void HostMoeMailbox::service() {
    std::int32_t layer = 0;
    try {
        HostCpuPool pool(config_.cpus);
        HostMoeExecutor::Options executor_options = config_.executor;
        executor_options.gguf_only =
            std::all_of(layers_.begin(), layers_.end(), [](const HostExpertLayer& l) {
                return l.format == HostExpertFormat::Gguf;
            });
        HostMoeExecutor executor(pool, config_.hidden, config_.intermediate, config_.max_tokens,
                                 config_.top_k, executor_options);
        started_.store(1, std::memory_order_release);
        using clock         = std::chrono::steady_clock;
        auto last           = clock::now();
        const auto idle_for = std::chrono::duration<double, std::milli>(config_.idle_ms);
        bool sleeping       = false;
        bool deep           = false;
        std::int32_t served = -1;
        for (std::uint32_t polls = 1; !stop_.load(std::memory_order_acquire); ++polls) {
            std::int32_t found = -1;
            if (load_acquire(slot(layer).ready) != 0) {
                found = layer;
            } else if (served >= 0 && load_acquire(slot(served).ready) != 0) {
                found = served; // the next slice of the same layer
            } else if ((deep || (polls & 1023U) == 0) && clock::now() - last > kResyncAfter) {
                // Only after a quiet millisecond: every slot line read costs a cache miss on memory
                // the GPU wrote (a scan of 48 slots on every 256 polls cost ~10 us of doorbell
                // latency). A window aborted between layers leaves the expected slot idle.
                for (std::int32_t l = 0; l < config_.layers && found < 0; ++l) {
                    if (load_acquire(slot(l).ready) != 0) found = l;
                }
                if (found < 0 && !sleeping && clock::now() - last > idle_for) sleeping = true;
                if (found < 0 && sleeping && clock::now() - last > kDeepIdleAfter) deep = true;
            }
            if (found < 0) {
                if (deep) {
                    // 100 wakeups/s instead of ~10-20k: the service cpu can reach C6 and the CPU
                    // package its idle states (i9-14900KF: package 33 W -> 10 W at idle; 1
                    // and 2 ms polls still kept the cpu in C1, 5 ms gave 12 W). Costs at most ~10
                    // ms on the first layer after a quiet second; while deep, every poll rescans
                    // all slots, as a slot left idle by an aborted window would otherwise wait 1024
                    // polls.
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                } else if (sleeping) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                } else {
                    _mm_pause();
                }
                continue;
            }
            serve(executor, found);
            served   = found;
            layer    = (found + 1) % config_.layers;
            last     = clock::now();
            sleeping = false;
            deep     = false;
        }
    } catch (const std::exception& e) {
        {
            const std::lock_guard lock(error_mutex_);
            error_ = e.what();
        }
        if (started_.load() == 0) {
            started_.store(-1, std::memory_order_release);
            return;
        }
        for (std::int32_t l = 0; l < config_.layers; ++l) *slot(l).active = 0;
        failed_.store(true, std::memory_order_release);
        release();
        // Failed: keep answering doorbells (the GPU resets done before each one) so that no
        // window can block on this mailbox; every partial is skipped (active = 0).
        while (!stop_.load(std::memory_order_acquire)) {
            for (std::int32_t l = 0; l < config_.layers; ++l) {
                const SlotView v = slot(l);
                if (load_acquire(v.ready) != 0) {
                    *v.ready  = 0;
                    *v.active = 0;
                    store_release(v.done, kReleased);
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    }
}

} // namespace ninfer::ops::host
