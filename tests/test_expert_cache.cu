// GPU test of ExpertCache (core/expert_cache.h): fills from pinned fake banks, slot tables read by
// a captured CUDA graph, epoch fences, and (with --bench) the fill bandwidth per PCIe link.
//
// Every bank word is a hash of (layer, expert, plane, word), so a kernel can recompute what any
// slot must hold. Checks:
//   - every window: a captured graph maps the routed ids through the slot table and verifies the
//     bytes of every slot it reads against the hash, while fills of other slots are in flight;
//     the mapped slots equal the host's published view; nothing read is ever wrong;
//   - after the run: tables == policy state, every occupied slot holds its expert's exact bytes,
//     every never-filled slot and every gap between planes/slots still holds the guard byte, and
//     the bytes copied are exactly 2 764 800 per fill;
//   - fence: a window that reads a slot for 20 ms while the next decision evicts it reads intact
//     bytes, whereas the same copy issued without the fence is caught by the same kernel;
//   - mid-window fills (submit_fills_unfenced) after the window left the layer: intact reads,
//     the copy lands during the window, publication at the next window; the same fill submitted
//     before the window left the layer is caught by the same reader;
//   - elastic columns: resized between windows, moved experts read intact, no table names a
//     released column;
//   - lending (prefill streaming ring): fills in flight into lent slots complete before the
//     lend event, queued ones are dropped, lent layers insert nothing and are unpublished, the
//     caller's writes into lent slots are overwritten by the reclaim refill, and the final state
//     (tables == policy, exact slot bytes) holds;
//   - lending of the base slots only (keep_elastic): the lent layers' elastic residents stay
//     published and intact, prompts fill those slots only, a shrink while lent respects the lent
//     base slots, the reclaim tops up in rank order; final state holds.
// Exit 77 when no GPU is usable.

#include "core/device.h"
#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::expert_cache;

constexpr std::int32_t kPlanes = 4;
// NVFP4 Qwen4Exp expert planes: gate/up codes, gate/up scales, down codes, down scales.
constexpr std::int64_t kPlaneBytes[kPlanes] = {1638400, 204800, 819200, 102400};
constexpr std::int64_t kSlotBytes           = 2764800;
constexpr std::int64_t kBankPad             = 256;  // bank stride = plane bytes + pad
constexpr std::int64_t kPoolGap             = 4096; // guard bytes after every pool plane/slot
constexpr unsigned char kGuard              = 0xA5;

__host__ __device__ inline std::uint64_t pattern(std::uint32_t layer, std::uint32_t expert,
                                                 std::uint32_t plane, std::uint64_t word) {
    std::uint64_t z = (static_cast<std::uint64_t>(layer) << 56) ^
                      (static_cast<std::uint64_t>(expert) << 40) ^
                      (static_cast<std::uint64_t>(plane) << 36) ^ word;
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct PoolView {
    std::byte* base[kPlanes];
    std::int64_t stride[kPlanes];
    std::int64_t bytes[kPlanes];
};

__device__ inline void spin_ns(std::uint64_t ns) {
    std::uint64_t start = 0, now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); } while (now - start < ns);
}

// One block per routed assignment: waits `delay_ns`, then checks the slot named by slot_ids.
__global__ void verify_kernel(const std::int32_t* ids, const std::int32_t* slot_ids,
                              std::int32_t layer, PoolView pool, std::uint64_t delay_ns,
                              unsigned long long* counters /* [0] bad words, [1] slots read */) {
    const std::int32_t e = ids[blockIdx.x];
    const std::int32_t s = slot_ids[blockIdx.x];
    if (s < 0) { return; }
    if (delay_ns != 0 && threadIdx.x == 0) { spin_ns(delay_ns); }
    __syncthreads();
    unsigned long long bad = 0;
    for (int q = 0; q < kPlanes; ++q) {
        const auto* w = reinterpret_cast<const std::uint64_t*>(pool.base[q] + s * pool.stride[q]);
        const std::int64_t words = pool.bytes[q] / 8;
        for (std::int64_t i = threadIdx.x; i < words; i += blockDim.x) {
            bad += w[i] != pattern(static_cast<std::uint32_t>(layer), static_cast<std::uint32_t>(e),
                                   static_cast<std::uint32_t>(q), static_cast<std::uint64_t>(i));
        }
    }
    if (bad != 0) { atomicAdd(&counters[0], bad); }
    if (threadIdx.x == 0) { atomicAdd(&counters[1], 1ULL); }
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// Fake banks: layers x planes x experts, stride = plane bytes + pad. Pinned (one cudaHostAlloc),
// unless `pageable_layers` lists layers whose banks are plain malloc memory (pinned fallback:
// those fill through the cache's pinned staging ring).
struct Banks {
    std::byte* host     = nullptr;
    std::int32_t layers = 0, experts = 0;
    std::size_t bytes = 0;
    std::vector<std::byte*> layer_base; // [layers]
    std::vector<std::uint8_t> pageable; // [layers]

    Banks(std::int32_t L, std::int32_t E, const std::vector<std::int32_t>& pageable_layers = {})
        : layers(L), experts(E) {
        std::size_t layer_bytes = 0;
        for (int q = 0; q < kPlanes; ++q) {
            layer_bytes += static_cast<std::size_t>(E) * (kPlaneBytes[q] + kBankPad);
        }
        bytes = layer_bytes * static_cast<std::size_t>(L);
        pageable.assign(static_cast<std::size_t>(L), 0);
        for (const std::int32_t l : pageable_layers) {
            pageable.at(static_cast<std::size_t>(l)) = 1;
        }
        CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocPortable));
        for (std::int32_t l = 0; l < L; ++l) {
            std::byte* p = host + static_cast<std::size_t>(l) * layer_bytes;
            if (pageable[static_cast<std::size_t>(l)] != 0) {
                p = static_cast<std::byte*>(
                    std::aligned_alloc(4096, (layer_bytes + 4095) / 4096 * 4096));
                if (p == nullptr) { throw std::bad_alloc(); }
            }
            layer_base.push_back(p);
            for (int q = 0; q < kPlanes; ++q) {
                for (std::int32_t e = 0; e < E; ++e) {
                    auto* w = reinterpret_cast<std::uint64_t*>(p);
                    for (std::int64_t i = 0; i < kPlaneBytes[q] / 8; ++i) {
                        w[i] =
                            pattern(static_cast<std::uint32_t>(l), static_cast<std::uint32_t>(e),
                                    static_cast<std::uint32_t>(q), static_cast<std::uint64_t>(i));
                    }
                    std::memset(p + kPlaneBytes[q], 0x5A, kBankPad);
                    p += kPlaneBytes[q] + kBankPad;
                }
            }
        }
    }

    ~Banks() {
        for (std::int32_t l = 0; l < layers; ++l) {
            if (pageable[static_cast<std::size_t>(l)] != 0) {
                std::free(layer_base[static_cast<std::size_t>(l)]);
            }
        }
        (void)cudaFreeHost(host);
    }

    [[nodiscard]] const std::byte* base(std::int32_t l, int q) const {
        std::size_t off = 0;
        for (int qq = 0; qq < q; ++qq) {
            off += static_cast<std::size_t>(experts) * (kPlaneBytes[qq] + kBankPad);
        }
        return layer_base[static_cast<std::size_t>(l)] + off;
    }
};

ExpertLayout make_layout(const Banks& banks, bool contiguous_slots) {
    ExpertLayout layout;
    layout.planes = kPlanes;
    for (int q = 0; q < kPlanes; ++q) {
        layout.plane_bytes[q] = kPlaneBytes[q];
        layout.bank_stride[q] = kPlaneBytes[q] + kBankPad;
        // Contiguous: ninfer-ext's one-slot-of-all-planes layout; otherwise one array per plane.
        layout.pool_stride[q] =
            contiguous_slots ? kSlotBytes + kPoolGap : kPlaneBytes[q] + kPoolGap;
    }
    for (std::int32_t l = 0; l < banks.layers; ++l) {
        for (int q = 0; q < kPlanes; ++q) { layout.bank_base.push_back(banks.base(l, q)); }
    }
    return layout;
}

// Device pool + table of one GPU, guard-filled.
struct GpuPool {
    int device         = 0;
    std::int64_t slots = 0;
    bool contiguous    = false;
    std::byte* mem[kPlanes]{};
    std::int64_t mem_bytes[kPlanes]{};
    std::int32_t* table = nullptr;
    GpuBinding binding;
    PoolView view{};

    GpuPool(int dev, std::int64_t n_slots, std::int64_t table_entries, bool contig)
        : device(dev), slots(n_slots), contiguous(contig) {
        CUDA_CHECK(cudaSetDevice(dev));
        binding.device = dev;
        if (contig) {
            mem_bytes[0] = n_slots * (kSlotBytes + kPoolGap);
            CUDA_CHECK(cudaMalloc(&mem[0], static_cast<std::size_t>(mem_bytes[0])));
            CUDA_CHECK(cudaMemset(mem[0], kGuard, static_cast<std::size_t>(mem_bytes[0])));
            std::int64_t off = 0;
            for (int q = 0; q < kPlanes; ++q) {
                binding.pool_base[q] = mem[0] + off;
                view.stride[q]       = kSlotBytes + kPoolGap;
                off += kPlaneBytes[q];
            }
        } else {
            for (int q = 0; q < kPlanes; ++q) {
                mem_bytes[q] = n_slots * (kPlaneBytes[q] + kPoolGap);
                CUDA_CHECK(cudaMalloc(&mem[q], static_cast<std::size_t>(mem_bytes[q])));
                CUDA_CHECK(cudaMemset(mem[q], kGuard, static_cast<std::size_t>(mem_bytes[q])));
                binding.pool_base[q] = mem[q];
                view.stride[q]       = kPlaneBytes[q] + kPoolGap;
            }
        }
        for (int q = 0; q < kPlanes; ++q) {
            view.base[q]  = binding.pool_base[q];
            view.bytes[q] = kPlaneBytes[q];
        }
        CUDA_CHECK(cudaMalloc(&table, static_cast<std::size_t>(table_entries) * 4));
        binding.table = table;
        // The guard memsets above run on the legacy default stream, which the cache's
        // non-blocking fill and window streams do not wait for. Under GPU time-slicing with another
        // process they landed after the first fills and overwrote loaded experts: finish them here.
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~GpuPool() {
        (void)cudaSetDevice(device);
        for (auto* m : mem) {
            if (m != nullptr) { (void)cudaFree(m); }
        }
        (void)cudaFree(table);
    }
};

// Final state: tables == policy, slot bytes exact, guards intact.
void check_final_state(const ExpertCache& cache, const Banks& banks, std::vector<GpuPool*>& pools) {
    const ExpertCachePolicy& pol = cache.policy();
    const PolicyConfig& c        = pol.config();
    for (std::int32_t p = 0; p < cache.gpus(); ++p) {
        GpuPool& gp = *pools[static_cast<std::size_t>(p)];
        CUDA_CHECK(cudaSetDevice(gp.device));
        std::vector<std::int32_t> table(static_cast<std::size_t>(c.layers) * c.experts);
        CUDA_CHECK(cudaMemcpy(table.data(), gp.table, table.size() * 4, cudaMemcpyDeviceToHost));
        std::int64_t wrong = 0;
        for (std::int32_t l = 0; l < c.layers; ++l) {
            for (std::int32_t e = 0; e < c.experts; ++e) {
                const std::int32_t want = pol.partition_of(e) == p ? pol.slot_of(l, e) : -1;
                wrong += table[static_cast<std::size_t>(l * c.experts + e)] != want;
                if (pol.partition_of(e) == p) { wrong += cache.published_slot(l, e) != want; }
            }
        }
        check(wrong == 0, "GPU " + std::to_string(p) + ": table == policy after flush (" +
                              std::to_string(wrong) + " wrong)");
        std::vector<std::vector<unsigned char>> host(kPlanes);
        for (int q = 0; q < kPlanes; ++q) {
            if (gp.mem[q] == nullptr) { continue; }
            host[q].resize(static_cast<std::size_t>(gp.mem_bytes[q]));
            CUDA_CHECK(
                cudaMemcpy(host[q].data(), gp.mem[q], host[q].size(), cudaMemcpyDeviceToHost));
        }
        std::int64_t bad_slots = 0, bad_guard = 0, occupied = 0;
        const std::int32_t per_layer = pol.base_slots(p);
        const bool elastic           = c.elastic_columns > 0;
        for (std::int64_t s = 0; s < gp.slots; ++s) {
            const std::int32_t e = pol.expert_in(p, static_cast<std::int32_t>(s));
            // Base slots layer-major, then (no gap here) the elastic columns column-major.
            const std::int64_t base_end = static_cast<std::int64_t>(c.layers) * per_layer;
            const std::int32_t l =
                static_cast<std::int32_t>(s < base_end ? s / per_layer : (s - base_end) % c.layers);
            occupied += e >= 0;
            std::int64_t off = 0;
            for (int q = 0; q < kPlanes; ++q) {
                const int mq = gp.contiguous ? 0 : q;
                const std::int64_t stride =
                    gp.contiguous ? kSlotBytes + kPoolGap : kPlaneBytes[q] + kPoolGap;
                const unsigned char* d = host[mq].data() + s * stride + (gp.contiguous ? off : 0);
                if (e >= 0) {
                    bad_slots += std::memcmp(d,
                                             banks.base(l, q) + static_cast<std::int64_t>(e) *
                                                                    (kPlaneBytes[q] + kBankPad),
                                             static_cast<std::size_t>(kPlaneBytes[q])) != 0;
                } else if (!elastic) { // a released elastic slot keeps its last expert's bytes
                    bad_slots += std::any_of(d, d + kPlaneBytes[q],
                                             [](unsigned char b) { return b != kGuard; });
                }
                off += kPlaneBytes[q];
            }
            // Guard gap after the slot (contiguous) or after every plane (separate arrays).
            for (int q = 0; q < (gp.contiguous ? 1 : kPlanes); ++q) {
                const std::int64_t len = gp.contiguous ? kSlotBytes : kPlaneBytes[q];
                const unsigned char* g = host[q].data() + s * (len + kPoolGap) + len;
                bad_guard +=
                    std::any_of(g, g + kPoolGap, [](unsigned char b) { return b != kGuard; });
            }
        }
        check(bad_slots == 0, "GPU " + std::to_string(p) + ": slot bytes (" +
                                  std::to_string(bad_slots) + " bad planes, " +
                                  std::to_string(occupied) + " occupied slots)");
        check(bad_guard == 0, "GPU " + std::to_string(p) + ": guard bytes intact (" +
                                  std::to_string(bad_guard) + " bad)");
    }
    const CacheCounters& k = cache.counters();
    check(k.fill_bytes == k.fills_issued * kSlotBytes,
          "fill bytes = fills x 2 764 800 (" + std::to_string(k.fill_bytes) + ")");
}

struct ScenarioResult {
    std::int64_t windows = 0, reads = 0, bad_words = 0;
};

// Random top-k routing (geometric over a per-layer shuffled expert order) for `windows` windows.
ScenarioResult run_windows(const std::vector<int>& devices, const PolicyConfig& pc, bool contiguous,
                           const Banks& banks, std::int32_t windows, std::int32_t tokens,
                           std::int32_t topk, std::uint64_t delay_ns, std::uint32_t seed,
                           std::int64_t depth_bytes) {
    const std::int32_t P = pc.partitions;
    std::vector<std::unique_ptr<GpuPool>> owned;
    std::vector<GpuPool*> pools;
    std::vector<GpuBinding> bindings;
    for (std::int32_t p = 0; p < P; ++p) {
        owned.push_back(std::make_unique<GpuPool>(devices[static_cast<std::size_t>(p)],
                                                  ExpertCache::pool_slots(pc, p),
                                                  ExpertCache::table_entries(pc), contiguous));
        pools.push_back(owned.back().get());
        bindings.push_back(owned.back()->binding);
    }
    ExpertCache cache(pc, make_layout(banks, contiguous), bindings, depth_bytes);

    std::mt19937 rng(seed);
    std::vector<std::vector<std::int32_t>> order(static_cast<std::size_t>(pc.layers));
    for (auto& o : order) {
        o.resize(static_cast<std::size_t>(pc.experts));
        std::iota(o.begin(), o.end(), 0);
        std::shuffle(o.begin(), o.end(), rng);
    }
    // Profile prefill from the routing distribution's head.
    cache.initialize(order);

    const std::int32_t n = tokens * topk; // ids per layer per window

    struct Dev {
        cudaStream_t stream          = nullptr;
        std::int32_t* ids            = nullptr; // [layers, n]
        std::int32_t* slot_ids       = nullptr; // [layers, n]
        unsigned long long* counters = nullptr;
        std::int32_t* host_ids       = nullptr; // pinned
        std::int32_t* host_slots     = nullptr;
        cudaGraphExec_t graph        = nullptr;
    };

    std::vector<Dev> dev(static_cast<std::size_t>(P));
    const std::size_t idbytes = static_cast<std::size_t>(pc.layers) * n * 4;
    for (std::int32_t p = 0; p < P; ++p) {
        Dev& d = dev[static_cast<std::size_t>(p)];
        CUDA_CHECK(cudaSetDevice(pools[static_cast<std::size_t>(p)]->device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&d.ids, idbytes));
        CUDA_CHECK(cudaMalloc(&d.slot_ids, idbytes));
        CUDA_CHECK(cudaMalloc(&d.counters, 16));
        CUDA_CHECK(cudaMemsetAsync(d.counters, 0, 16, d.stream)); // ordered before the reader
        CUDA_CHECK(cudaHostAlloc(&d.host_ids, idbytes, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&d.host_slots, idbytes, cudaHostAllocPortable));
        // The window graph: per layer, map ids -> slots through the table, then read the slots.
        cudaGraph_t g = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
        for (std::int32_t l = 0; l < pc.layers; ++l) {
            expert_cache_map_slots(d.ids + l * n, n, l, pc.experts,
                                   pools[static_cast<std::size_t>(p)]->table, d.slot_ids + l * n,
                                   d.stream);
            verify_kernel<<<n, 256, 0, d.stream>>>(d.ids + l * n, d.slot_ids + l * n, l,
                                                   pools[static_cast<std::size_t>(p)]->view,
                                                   delay_ns, d.counters);
        }
        CUDA_CHECK(cudaStreamEndCapture(d.stream, &g));
        CUDA_CHECK(cudaGraphInstantiate(&d.graph, g, 0));
        CUDA_CHECK(cudaGraphDestroy(g));
    }
    std::vector<cudaStream_t> streams;
    for (auto& d : dev) { streams.push_back(d.stream); }

    std::geometric_distribution<int> geo(0.06);
    std::vector<std::vector<std::int32_t>> window_ids(static_cast<std::size_t>(pc.layers));
    std::vector<Routed> routed;
    std::int64_t map_wrong = 0;
    std::int64_t resizes = 0, released_named = 0;
    for (std::int32_t w = 0; w < windows; ++w) {
        for (std::int32_t l = 0; l < pc.layers; ++l) {
            auto& ids = window_ids[static_cast<std::size_t>(l)];
            ids.clear();
            for (std::int32_t t = 0; t < tokens; ++t) {
                std::vector<std::int32_t> tok;
                while (static_cast<std::int32_t>(tok.size()) < topk) {
                    const std::int32_t e =
                        order[static_cast<std::size_t>(l)]
                             [static_cast<std::size_t>(std::min(geo(rng), pc.experts - 1))];
                    if (std::find(tok.begin(), tok.end(), e) == tok.end()) { tok.push_back(e); }
                }
                ids.insert(ids.end(), tok.begin(), tok.end());
            }
        }
        cache.begin_window(streams);
        for (std::int32_t p = 0; p < P; ++p) {
            Dev& d = dev[static_cast<std::size_t>(p)];
            for (std::int32_t l = 0; l < pc.layers; ++l) {
                std::copy(window_ids[static_cast<std::size_t>(l)].begin(),
                          window_ids[static_cast<std::size_t>(l)].end(), d.host_ids + l * n);
            }
            CUDA_CHECK(cudaSetDevice(pools[static_cast<std::size_t>(p)]->device));
            CUDA_CHECK(
                cudaMemcpyAsync(d.ids, d.host_ids, idbytes, cudaMemcpyHostToDevice, d.stream));
            CUDA_CHECK(cudaGraphLaunch(d.graph, d.stream));
            CUDA_CHECK(cudaMemcpyAsync(d.host_slots, d.slot_ids, idbytes, cudaMemcpyDeviceToHost,
                                       d.stream));
        }
        cache.end_window(streams);
        // Expected device view of this window = published state, read before observe().
        std::vector<std::int32_t> expected(static_cast<std::size_t>(pc.layers) * n);
        for (std::int32_t l = 0; l < pc.layers; ++l) {
            for (std::int32_t i = 0; i < n; ++i) {
                expected[static_cast<std::size_t>(l * n + i)] = cache.published_slot(
                    l, window_ids[static_cast<std::size_t>(l)][static_cast<std::size_t>(i)]);
            }
        }
        for (std::int32_t l = 0; l < pc.layers; ++l) {
            count_routed(window_ids[static_cast<std::size_t>(l)], pc.experts, routed);
            const LayerView v = cache.observe(l, routed, tokens);
            for (const ResidentExpert& r : v.resident) {
                map_wrong += r.slot != cache.published_slot(l, r.expert);
            }
        }
        cache.submit_fills();
        for (std::int32_t p = 0; p < P; ++p) {
            Dev& d = dev[static_cast<std::size_t>(p)];
            CUDA_CHECK(cudaSetDevice(pools[static_cast<std::size_t>(p)]->device));
            CUDA_CHECK(cudaStreamSynchronize(d.stream));
            for (std::int32_t l = 0; l < pc.layers; ++l) {
                for (std::int32_t i = 0; i < n; ++i) {
                    const std::int32_t e =
                        window_ids[static_cast<std::size_t>(l)][static_cast<std::size_t>(i)];
                    const std::int32_t want =
                        (e % P == p) ? expected[static_cast<std::size_t>(l * n + i)] : -1;
                    map_wrong += d.host_slots[l * n + i] != want;
                }
            }
        }
        // Elastic columns: resize between windows (device idle); the next windows' verify
        // kernel checks every slot it reads (moved experts included); no table names a released
        // column afterwards.
        if (pc.elastic_columns > 0 && w % 6 == 5) {
            static const std::int32_t cycle[] = {1, 0, 2, 2, 0, 1, 2};
            const std::int32_t columns =
                std::min(pc.elastic_columns, cycle[static_cast<std::size_t>(w / 6) % 7]);
            cache.set_active_columns(columns, streams);
            ++resizes;
            const ExpertCachePolicy& pol = cache.policy();
            for (std::int32_t p = 0; p < P; ++p) {
                GpuPool& gp = *pools[static_cast<std::size_t>(p)];
                CUDA_CHECK(cudaSetDevice(gp.device));
                std::vector<std::int32_t> table(static_cast<std::size_t>(pc.layers) * pc.experts);
                CUDA_CHECK(
                    cudaMemcpy(table.data(), gp.table, table.size() * 4, cudaMemcpyDeviceToHost));
                const std::int64_t base_end =
                    static_cast<std::int64_t>(pc.layers) * pol.base_slots(p);
                for (const std::int32_t slot : table) {
                    if (slot < base_end) { continue; }
                    released_named += (slot - base_end) / pc.layers >= columns;
                }
            }
        }
    }
    cache.flush();
    cache.begin_window(streams); // publish the last fills
    cache.end_window(streams);
    ScenarioResult res;
    res.windows = windows;
    for (std::int32_t p = 0; p < P; ++p) {
        Dev& d = dev[static_cast<std::size_t>(p)];
        CUDA_CHECK(cudaSetDevice(pools[static_cast<std::size_t>(p)]->device));
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        unsigned long long c[2]{};
        CUDA_CHECK(cudaMemcpy(c, d.counters, 16, cudaMemcpyDeviceToHost));
        res.bad_words += static_cast<std::int64_t>(c[0]);
        res.reads += static_cast<std::int64_t>(c[1]);
    }
    check(map_wrong == 0,
          "mapped slots == published view (" + std::to_string(map_wrong) + " wrong)");
    if (pc.elastic_columns > 0) {
        check(resizes > 0 && released_named == 0, "elastic: no table names a released column (" +
                                                      std::to_string(released_named) + " after " +
                                                      std::to_string(resizes) + " resizes)");
        check(cache.counters().moved > 0 && cache.counters().shrink_evicted > 0,
              "elastic: shrinks moved and evicted experts (" +
                  std::to_string(cache.counters().moved) + " moved, " +
                  std::to_string(cache.counters().shrink_evicted) + " evicted)");
    }
    check_final_state(cache, banks, pools);
    const CacheCounters& k = cache.counters();
    check(k.inserts == k.fills_issued + k.fills_dropped, "every insertion copied once or dropped");
    std::printf("  windows %lld: policy hit %.3f device hit %.3f, inserts %lld, fills %lld, "
                "dropped %lld, published %lld, unpublished %lld, table launches %lld, slot reads "
                "%lld, bad words %lld\n",
                static_cast<long long>(k.windows), double(k.policy_hits) / double(k.accesses),
                double(k.device_hits) / double(k.accesses), static_cast<long long>(k.inserts),
                static_cast<long long>(k.fills_issued), static_cast<long long>(k.fills_dropped),
                static_cast<long long>(k.published), static_cast<long long>(k.unpublished),
                static_cast<long long>(k.table_launches), static_cast<long long>(res.reads),
                static_cast<long long>(res.bad_words));
    check(res.bad_words == 0, "no window ever read a wrong byte");
    check(res.reads > 0 && k.fills_issued > 0 && k.unpublished > 0,
          "scenario exercised reads, fills, evictions");
    const bool any_pageable =
        std::any_of(banks.pageable.begin(), banks.pageable.end(), [](auto v) { return v != 0; });
    for (std::int32_t l = 0; l < banks.layers; ++l) {
        check(cache.pageable_layer(l) == (banks.pageable[static_cast<std::size_t>(l)] != 0),
              "layer " + std::to_string(l) + " classified pinned/pageable");
    }
    check(any_pageable ? k.staged_fills > 0 && cache.staging_slots() > 0
                       : k.staged_fills == 0 && cache.staging_slots() == 0,
          "staged fills only for pageable banks (" + std::to_string(k.staged_fills) + ")");
    if (any_pageable) {
        std::printf("  staged fills %lld through %d staging slots per GPU\n",
                    static_cast<long long>(k.staged_fills), cache.staging_slots());
    }
    for (auto& d : dev) {
        (void)cudaGraphExecDestroy(d.graph);
        (void)cudaStreamDestroy(d.stream);
        (void)cudaFree(d.ids);
        (void)cudaFree(d.slot_ids);
        (void)cudaFree(d.counters);
        (void)cudaFreeHost(d.host_ids);
        (void)cudaFreeHost(d.host_slots);
    }
    return res;
}

// A window reads expert A's slot for 20 ms while the next decision evicts A for B. Through the
// cache the fill waits for the window's fence; a raw copy without it is caught by the same kernel.
void fence_scenario(int device, const Banks& banks) {
    PolicyConfig pc;
    pc.layers          = 1;
    pc.experts         = banks.experts;
    pc.partitions      = 1;
    pc.slots_per_layer = 1;
    for (const bool fenced : {true, false}) {
        GpuPool pool(device, 1, pc.experts, false);
        GpuBinding binding  = pool.binding;
        ExpertLayout layout = make_layout(banks, false);
        layout.bank_base.resize(kPlanes); // layer 0 only
        ExpertCache cache(pc, layout, std::span<const GpuBinding>(&binding, 1));
        const std::vector<std::vector<std::int32_t>> profile{{3}};
        cache.initialize(profile);
        cudaStream_t s = nullptr, raw = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking));
        std::int32_t *ids = nullptr, *slots = nullptr;
        unsigned long long* counters = nullptr;
        CUDA_CHECK(cudaMalloc(&ids, 4));
        CUDA_CHECK(cudaMalloc(&slots, 4));
        CUDA_CHECK(cudaMalloc(&counters, 16));
        // Counters and ids on the reader's stream (the legacy stream does not order them
        // against the non-blocking reader; under GPU time-slicing they landed after it).
        CUDA_CHECK(cudaMemsetAsync(counters, 0, 16, s));
        const std::int32_t a = 3;
        CUDA_CHECK(cudaMemcpyAsync(ids, &a, 4, cudaMemcpyHostToDevice, s));
        cache.begin_window(std::span<const cudaStream_t>(&s, 1));
        expert_cache_map_slots(ids, 1, 0, pc.experts, pool.table, slots, s);
        verify_kernel<<<1, 256, 0, s>>>(ids, slots, 0, pool.view, 20'000'000ULL, counters);
        cache.end_window(std::span<const cudaStream_t>(&s, 1));
        if (fenced) {
            const std::vector<Routed> b{{7, 1}};
            const LayerView v = cache.observe(0, b, 1);
            check(v.fills.size() == 1 && v.fills[0].victim == 3, "fence: B evicts A");
            cache.submit_fills();
        } else {
            // The same copy of B into A's slot, issued without waiting for the window.
            for (int q = 0; q < kPlanes; ++q) {
                CUDA_CHECK(cudaMemcpyAsync(
                    pool.binding.pool_base[q], banks.base(0, q) + 7 * (kPlaneBytes[q] + kBankPad),
                    static_cast<std::size_t>(kPlaneBytes[q]), cudaMemcpyHostToDevice, raw));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long c[2]{};
        CUDA_CHECK(cudaMemcpy(c, counters, 16, cudaMemcpyDeviceToHost));
        if (fenced) {
            check(c[1] == 1 && c[0] == 0,
                  "fence: the 20 ms reader saw A intact (bad words " + std::to_string(c[0]) + ")");
            cache.flush();
            cache.begin_window(std::span<const cudaStream_t>(&s, 1));
            cache.end_window(std::span<const cudaStream_t>(&s, 1));
            CUDA_CHECK(cudaStreamSynchronize(s));
            check(cache.published_slot(0, 7) == 0 && cache.published_slot(0, 3) == -1,
                  "fence: B published in A's slot after the fill");
        } else {
            check(c[1] == 1 && c[0] > 0, "fence control: an unfenced copy is detected (bad words " +
                                             std::to_string(c[0]) + ")");
        }
        std::printf("  fence %s: slot reads %llu, bad words %llu\n",
                    fenced ? "on " : "off (raw copy)", c[1], c[0]);
        (void)cudaFree(ids);
        (void)cudaFree(slots);
        (void)cudaFree(counters);
        (void)cudaStreamDestroy(s);
        (void)cudaStreamDestroy(raw);
    }
}

__global__ void flag_kernel(volatile std::uint32_t* flag) {
    __threadfence_system();
    *flag = 1;
}

// Mid-window fills (submit_fills_unfenced): layer 0 reads expert A's slot for 20 ms, then the
// window raises a host flag (as Qwen4OffloadExperts' next-layer routing publication) and reads
// layer 1 for 60 ms more. On the flag the host evicts A for B in layer 0 and submits the fill
// unfenced: the 20 ms reader must see A intact, the copy must complete while the window still
// runs (no wait for its fence), and B is published at the next window, whose reader sees it.
// Control (`premature`): the same unfenced fill submitted before the flag, while the 20 ms reader
// still has A's slot ahead of it, must be caught by the same reader (bad words > 0): the intact
// reads above then prove the flag ordering, not a reader that cannot see a rewrite.
void unfenced_scenario(int device, const Banks& banks, bool premature) {
    PolicyConfig pc;
    pc.layers          = 2;
    pc.experts         = banks.experts;
    pc.partitions      = 1;
    pc.slots_per_layer = 1;
    GpuPool pool(device, 2, 2 * pc.experts, false);
    GpuBinding binding  = pool.binding;
    ExpertLayout layout = make_layout(banks, false);
    layout.bank_base.resize(2 * kPlanes); // layers 0 and 1
    ExpertCache cache(pc, layout, std::span<const GpuBinding>(&binding, 1));
    const std::vector<std::vector<std::int32_t>> profile{{3}, {5}};
    cache.initialize(profile);
    cudaStream_t s = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    std::int32_t *ids = nullptr, *slots = nullptr;
    unsigned long long* counters = nullptr;
    std::uint32_t* flag          = nullptr;
    CUDA_CHECK(cudaMalloc(&ids, 8));
    CUDA_CHECK(cudaMalloc(&slots, 8));
    CUDA_CHECK(cudaMalloc(&counters, 16));
    CUDA_CHECK(cudaMemsetAsync(counters, 0, 16, s)); // on the reader's stream
    CUDA_CHECK(cudaHostAlloc(&flag, 64, cudaHostAllocMapped));
    const std::int32_t routed_ids[2] = {3, 5};
    CUDA_CHECK(cudaMemcpyAsync(ids, routed_ids, 8, cudaMemcpyHostToDevice, s));
    const auto streams = std::span<const cudaStream_t>(&s, 1);

    *reinterpret_cast<volatile std::uint32_t*>(flag) = 0;
    cache.begin_window(streams);
    expert_cache_map_slots(ids, 1, 0, pc.experts, pool.table, slots, s);
    verify_kernel<<<1, 256, 0, s>>>(ids, slots, 0, pool.view, 20'000'000ULL, counters);
    flag_kernel<<<1, 1, 0, s>>>(flag);
    expert_cache_map_slots(ids + 1, 1, 1, pc.experts, pool.table, slots + 1, s);
    verify_kernel<<<1, 256, 0, s>>>(ids + 1, slots + 1, 1, pool.view, 60'000'000ULL, counters);
    if (!premature) {
        while (*reinterpret_cast<volatile std::uint32_t*>(flag) == 0) {}
    }
    const std::vector<Routed> a{{3, 1}}, b{{7, 1}}, c{{5, 1}};
    (void)a;
    const LayerView v = cache.observe(0, b, 1);
    check(v.fills.size() == 1 && v.fills[0].victim == 3 && v.missing.size() == 1,
          "unfenced: B evicts A in layer 0, B computed elsewhere this window");
    cache.submit_fills_unfenced();
    if (premature) {
        // The copy (~0.1 ms) lands while the reader of layer 0 is still in its 20 ms delay.
        cache.flush();
        CUDA_CHECK(cudaStreamSynchronize(s));
        unsigned long long k[2]{};
        CUDA_CHECK(cudaMemcpy(k, counters, 16, cudaMemcpyDeviceToHost));
        const bool raised = *reinterpret_cast<volatile std::uint32_t*>(flag) != 0;
        check(raised && k[1] == 2 && k[0] > 0,
              "unfenced control: a fill submitted before the flag is detected (bad words " +
                  std::to_string(k[0]) + ")");
        std::printf("  unfenced control (fill before the flag): slot reads %llu, bad words %llu\n",
                    k[1], k[0]);
        cache.end_window(streams);
        CUDA_CHECK(cudaStreamSynchronize(s));
        (void)cudaFree(ids);
        (void)cudaFree(slots);
        (void)cudaFree(counters);
        (void)cudaFreeHost(flag);
        (void)cudaStreamDestroy(s);
        return;
    }
    bool landed_in_window = false;
    while (cudaStreamQuery(s) == cudaErrorNotReady) {
        cache.poll();
        if (cache.fills_pending() == 0) {
            landed_in_window = cudaStreamQuery(s) == cudaErrorNotReady;
            break;
        }
    }
    check(landed_in_window, "unfenced: the fill completed while the window was still running");
    cache.end_window(streams);
    (void)cache.observe(1, c, 1);
    cache.submit_fills();
    CUDA_CHECK(cudaStreamSynchronize(s));
    unsigned long long k[2]{};
    CUDA_CHECK(cudaMemcpy(k, counters, 16, cudaMemcpyDeviceToHost));
    check(k[1] == 2 && k[0] == 0,
          "unfenced: window 1 read A and layer 1 intact (bad words " + std::to_string(k[0]) + ")");
    // Next window: B is published in A's slot and reads intact.
    const std::int32_t next_ids[2] = {7, 5};
    CUDA_CHECK(cudaMemcpyAsync(ids, next_ids, 8, cudaMemcpyHostToDevice, s)); // reader's stream
    CUDA_CHECK(cudaMemsetAsync(counters, 0, 16, s));
    cache.begin_window(streams);
    check(cache.published_slot(0, 7) == 0 && cache.published_slot(0, 3) == -1,
          "unfenced: B published at the next window");
    expert_cache_map_slots(ids, 2, 0, pc.experts, pool.table, slots, s); // layer 0 table for both
    verify_kernel<<<1, 256, 0, s>>>(ids, slots, 0, pool.view, 0, counters);
    cache.end_window(streams);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(k, counters, 16, cudaMemcpyDeviceToHost));
    check(k[1] == 1 && k[0] == 0,
          "unfenced: window 2 read B intact (bad words " + std::to_string(k[0]) + ")");
    std::printf("  unfenced mid-window fill: landed during the window %d, window 2 B reads %llu "
                "bad %llu\n",
                landed_in_window ? 1 : 0, k[1], k[0]);
    (void)cudaFree(ids);
    (void)cudaFree(slots);
    (void)cudaFree(counters);
    (void)cudaFreeHost(flag);
    (void)cudaStreamDestroy(s);
}

// Lending: 4 layers x 64 experts, 8 slots per layer; layers 2..3 are lent while a backlog of
// fills into them is queued and in flight; the "ring user" overwrites every lent slot after the
// lend event; reclaim refills them from profiles.
__global__ void scribble_kernel(std::uint64_t* p, std::int64_t words) {
    for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += gridDim.x * blockDim.x) {
        p[i] = 0xDEADBEEFDEADBEEFULL;
    }
}

void lend_scenario(int device, const Banks& banks4) {
    PolicyConfig pc;
    pc.layers          = 4;
    pc.experts         = 64;
    pc.partitions      = 1;
    pc.slots_per_layer = 8;
    CUDA_CHECK(cudaSetDevice(device));
    GpuPool pool(device, ExpertCache::pool_slots(pc, 0), ExpertCache::table_entries(pc), false);
    std::vector<GpuBinding> bindings{pool.binding};
    std::vector<GpuPool*> pools{&pool};
    ExpertCache cache(pc, make_layout(banks4, false), bindings, 4 * kSlotBytes); // shallow queue
    std::vector<std::vector<std::int32_t>> profile(4);
    for (std::int32_t l = 0; l < 4; ++l) {
        for (std::int32_t e = 0; e < 64; ++e) {
            profile[static_cast<std::size_t>(l)].push_back((e * 7 + l) % 64);
        }
    }
    cache.initialize(profile);
    cudaStream_t s = nullptr, ring = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ring, cudaStreamNonBlocking));
    const cudaStream_t streams[1] = {s};
    // A prompt window that replaces every layer's residents: 32 fills queued, 4 in flight.
    cache.begin_window(streams);
    cache.end_window(streams);
    for (std::int32_t l = 0; l < 4; ++l) {
        std::vector<Routed> routed;
        for (std::int32_t e = 40; e < 48; ++e) { routed.push_back(Routed{(e + l) % 64, 5}); }
        std::sort(routed.begin(), routed.end(),
                  [](const Routed& a, const Routed& b) { return a.expert < b.expert; });
        (void)cache.observe_prompt(l, routed, 8);
    }
    cache.submit_fills();
    check(cache.fills_pending() > 0, "lend: a backlog is pending when the layers are lent");
    const auto events = cache.lend_layers(2, 2);
    check(events.size() == 1 && cache.lent(2) && cache.lent(3) && !cache.lent(1),
          "lend: layers 2..3 lent");
    // The ring user overwrites every lent slot after the lend event.
    CUDA_CHECK(cudaStreamWaitEvent(ring, events[0], 0));
    for (int q = 0; q < kPlanes; ++q) {
        for (std::int64_t sl = 2 * 8; sl < 4 * 8; ++sl) {
            scribble_kernel<<<64, 256, 0, ring>>>(
                reinterpret_cast<std::uint64_t*>(pool.binding.pool_base[q] +
                                                 sl * (kPlaneBytes[q] + kPoolGap)),
                kPlaneBytes[q] / 8);
        }
    }
    // A prompt window while lent: lent layers insert nothing and are unpublished.
    cache.begin_window(streams);
    check(cache.published_slot(2, 42) == -1 && cache.published_slot(3, (0 * 7 + 3) % 64) == -1,
          "lend: lent layers unpublished at the next window");
    cache.end_window(streams);
    std::size_t lent_inserts = 0, other_inserts = 0;
    for (std::int32_t l = 0; l < 4; ++l) {
        std::vector<Routed> routed;
        for (std::int32_t e = 10; e < 20; ++e) { routed.push_back(Routed{e, 3}); }
        const LayerView v = cache.observe_prompt(l, routed, 8);
        (l >= 2 ? lent_inserts : other_inserts) += v.fills.size();
        if (l >= 2) {
            check(v.resident.empty() && v.missing.size() == 10,
                  "lend: lent layer classifies all missing");
        }
    }
    check(lent_inserts == 0 && other_inserts > 0,
          "lend: lent layers insert nothing (" + std::to_string(lent_inserts) + ")");
    cache.submit_fills();
    CUDA_CHECK(cudaStreamSynchronize(ring));
    CUDA_CHECK(cudaStreamSynchronize(s));
    // Reclaim: refill the lent layers from profiles of >= 8 experts each.
    std::vector<std::vector<std::int32_t>> refill{{1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                  {60, 61, 62, 63, 0, 1, 2, 3}};
    cache.reclaim_layers(refill);
    check(!cache.lending(), "lend: reclaimed");
    cache.flush();
    cache.begin_window(streams); // publish
    cache.end_window(streams);
    CUDA_CHECK(cudaStreamSynchronize(s));
    check(cache.published_slot(2, 1) >= 0 && cache.published_slot(3, 63) >= 0 &&
              cache.published_slot(2, 9) == -1,
          "lend: reclaimed layers hold their profile's hottest experts");
    check_final_state(cache, banks4, pools);
    std::printf("  lend/reclaim: dropped fills %lld, inserts into lent layers %zu\n",
                static_cast<long long>(cache.counters().fills_dropped), lent_inserts);
    (void)cudaStreamDestroy(s);
    (void)cudaStreamDestroy(ring);
}

// Lending of the base slots only (lend_layers keep_elastic): 4 layers x 64 experts, base 8
// slots per layer + 2 elastic columns (per GPU), on 1 or 2 GPUs. Layers 2..3 are lent while a
// backlog is pending; the ring user overwrites every lent BASE slot; the lent layers' elastic
// residents stay published and intact (a graph reads them through the table), a prompt window
// inserts into the lent layers' elastic slots only, a shrink while lent keeps them out of the
// lent base slots, and the reclaim tops the layers up from a profile in rank order. Final state
// (tables == policy, exact slot bytes) holds.
void lend_keep_scenario(const std::vector<int>& devices, const Banks& banks4) {
    const auto P = static_cast<std::int32_t>(devices.size());
    PolicyConfig pc;
    pc.layers          = 4;
    pc.experts         = 64;
    pc.partitions      = P;
    pc.slots_per_layer = 8 * P;
    pc.elastic_columns = 2;
    std::vector<std::unique_ptr<GpuPool>> owned;
    std::vector<GpuPool*> pools;
    std::vector<GpuBinding> bindings;
    for (std::int32_t p = 0; p < P; ++p) {
        owned.push_back(std::make_unique<GpuPool>(devices[static_cast<std::size_t>(p)],
                                                  ExpertCache::pool_slots(pc, p),
                                                  ExpertCache::table_entries(pc), false));
        pools.push_back(owned.back().get());
        bindings.push_back(owned.back()->binding);
    }
    ExpertCache cache(pc, make_layout(banks4, false), bindings, 4 * kSlotBytes); // shallow queue
    std::vector<std::vector<std::int32_t>> profile(4);
    for (std::int32_t l = 0; l < 4; ++l) {
        for (std::int32_t e = 0; e < 64; ++e) {
            profile[static_cast<std::size_t>(l)].push_back((e * 7 + l) % 64);
        }
    }
    cache.initialize(profile);
    const ExpertCachePolicy& pol = cache.policy();
    std::vector<cudaStream_t> streams, rings;
    for (std::int32_t p = 0; p < P; ++p) {
        CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(p)]));
        cudaStream_t s = nullptr, r = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&r, cudaStreamNonBlocking));
        streams.push_back(s);
        rings.push_back(r);
    }
    const auto is_base = [&](std::int32_t p, std::int32_t slot) {
        return slot < pc.layers * pol.base_slots(p);
    };
    // A prompt window that replaces residents of every layer: a backlog is queued.
    cache.begin_window(streams);
    cache.end_window(streams);
    for (std::int32_t l = 0; l < 4; ++l) {
        std::vector<Routed> routed;
        for (std::int32_t e = 40; e < 52; ++e) { routed.push_back(Routed{(e + l) % 64, 5}); }
        std::sort(routed.begin(), routed.end(),
                  [](const Routed& a, const Routed& b) { return a.expert < b.expert; });
        (void)cache.observe_prompt(l, routed, 8);
    }
    cache.submit_fills();
    check(cache.fills_pending() > 0, "keep: a backlog is pending when the layers are lent");
    std::vector<std::pair<std::int32_t, std::int32_t>> kept; // (layer, expert) in elastic slots
    for (std::int32_t l = 2; l < 4; ++l) {
        for (std::int32_t e = 0; e < 64; ++e) {
            const std::int32_t s = pol.slot_of(l, e);
            if (s >= 0 && !is_base(pol.partition_of(e), s)) { kept.emplace_back(l, e); }
        }
    }
    const auto events = cache.lend_layers(2, 2, /*keep_elastic=*/true);
    check(cache.lent(2) && cache.lent(3) && !cache.lent(1), "keep: layers 2..3 lent");
    std::size_t kept_after = 0;
    for (const auto& [l, e] : kept) { kept_after += pol.slot_of(l, e) >= 0; }
    check(!kept.empty() && kept_after == kept.size(),
          "keep: elastic residents survive the lend (" + std::to_string(kept.size()) + ")");
    for (std::int32_t l = 2; l < 4; ++l) {
        for (std::int32_t e = 0; e < 64; ++e) {
            const std::int32_t s = pol.slot_of(l, e);
            check(s < 0 || !is_base(pol.partition_of(e), s),
                  "keep: no lent layer expert in a base slot");
        }
    }
    // The ring user overwrites every lent base slot after the lend event.
    for (std::int32_t p = 0; p < P; ++p) {
        CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(p)]));
        CUDA_CHECK(cudaStreamWaitEvent(rings[static_cast<std::size_t>(p)],
                                       events[static_cast<std::size_t>(p)], 0));
        const std::int32_t base = pol.base_slots(p);
        for (int q = 0; q < kPlanes; ++q) {
            for (std::int64_t sl = 2 * base; sl < 4 * base; ++sl) {
                scribble_kernel<<<64, 256, 0, rings[static_cast<std::size_t>(p)]>>>(
                    reinterpret_cast<std::uint64_t*>(
                        pools[static_cast<std::size_t>(p)]->binding.pool_base[q] +
                        sl * (kPlaneBytes[q] + kPoolGap)),
                    kPlaneBytes[q] / 8);
            }
        }
    }
    // A prompt window while lent: published tables of the lent layers name only elastic slots,
    // whose bytes a kernel reads intact (after the scribbles); the lent layers insert into their
    // elastic slots only.
    cache.flush();
    cache.begin_window(streams);
    std::int64_t lent_published = 0, base_published = 0;
    for (std::int32_t l = 2; l < 4; ++l) {
        for (std::int32_t e = 0; e < 64; ++e) {
            const std::int32_t s = cache.published_slot(l, e);
            lent_published += s >= 0;
            base_published += s >= 0 && is_base(pol.partition_of(e), s);
        }
    }
    check(lent_published > 0 && base_published == 0,
          "keep: lent layers publish elastic residents only (" + std::to_string(lent_published) +
              ")");
    unsigned long long bad_reads = 0, reads = 0;
    for (std::int32_t p = 0; p < P; ++p) {
        CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(p)]));
        CUDA_CHECK(cudaStreamSynchronize(rings[static_cast<std::size_t>(p)]));
        std::int32_t *ids = nullptr, *slots = nullptr;
        unsigned long long* counters = nullptr;
        CUDA_CHECK(cudaMalloc(&ids, 64 * 4));
        CUDA_CHECK(cudaMalloc(&slots, 64 * 4));
        CUDA_CHECK(cudaMalloc(&counters, 16));
        const cudaStream_t reader = streams[static_cast<std::size_t>(p)];
        CUDA_CHECK(cudaMemsetAsync(counters, 0, 16, reader)); // on the reader's stream
        std::vector<std::int32_t> all(64);
        std::iota(all.begin(), all.end(), 0);
        CUDA_CHECK(cudaMemcpyAsync(ids, all.data(), 64 * 4, cudaMemcpyHostToDevice, reader));
        CUDA_CHECK(cudaStreamSynchronize(reader)); // `all` is pageable and goes out of scope
        for (std::int32_t l = 2; l < 4; ++l) {
            expert_cache_map_slots(ids, 64, l, 64, pools[static_cast<std::size_t>(p)]->table, slots,
                                   streams[static_cast<std::size_t>(p)]);
            verify_kernel<<<64, 256, 0, streams[static_cast<std::size_t>(p)]>>>(
                ids, slots, l, pools[static_cast<std::size_t>(p)]->view, 0, counters);
        }
        CUDA_CHECK(cudaStreamSynchronize(streams[static_cast<std::size_t>(p)]));
        unsigned long long c[2]{};
        CUDA_CHECK(cudaMemcpy(c, counters, 16, cudaMemcpyDeviceToHost));
        bad_reads += c[0];
        reads += c[1];
        (void)cudaFree(ids);
        (void)cudaFree(slots);
        (void)cudaFree(counters);
    }
    check(reads > 0 && bad_reads == 0, "keep: kept elastic residents read intact while lent (" +
                                           std::to_string(reads) + " slots read)");
    cache.end_window(streams);
    std::size_t lent_inserts = 0;
    for (std::int32_t l = 0; l < 4; ++l) {
        std::vector<Routed> routed;
        for (std::int32_t e = 10; e < 20; ++e) { routed.push_back(Routed{e, 3}); }
        const LayerView v = cache.observe_prompt(l, routed, 8);
        if (l >= 2) {
            lent_inserts += v.fills.size();
            for (const Insert& i : v.fills) {
                check(!is_base(i.partition, i.slot), "keep: lent layer inserts into elastic slots");
            }
        }
    }
    check(lent_inserts > 0, "keep: lent layers insert into their elastic slots");
    cache.submit_fills();
    // Shrink to one column while lent: survivors of the released column move to the lent layers'
    // free elastic slots, never to their base slots.
    cache.set_active_columns(1, streams);
    for (std::int32_t l = 2; l < 4; ++l) {
        for (std::int32_t e = 0; e < 64; ++e) {
            const std::int32_t s = pol.slot_of(l, e);
            check(s < 0 || !is_base(pol.partition_of(e), s),
                  "keep: shrink keeps lent base slots empty");
        }
    }
    cache.set_active_columns(2, streams);
    // Reclaim: top up from profiles in rank order.
    std::vector<std::vector<std::int32_t>> refill{
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
        {60, 61, 62, 63, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    std::vector<std::int32_t> rank(static_cast<std::size_t>(4 * 64), 64);
    for (std::int32_t i = 0; i < 2; ++i) {
        for (std::size_t r = 0; r < refill[static_cast<std::size_t>(i)].size(); ++r) {
            rank[static_cast<std::size_t>((2 + i) * 64 + refill[static_cast<std::size_t>(i)][r])] =
                static_cast<std::int32_t>(r);
        }
    }
    cache.reclaim_layers(refill, rank);
    check(!cache.lending(), "keep: reclaimed");
    cache.flush();
    cache.begin_window(streams); // publish
    cache.end_window(streams);
    for (std::int32_t p = 0; p < P; ++p) {
        CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(p)]));
        CUDA_CHECK(cudaStreamSynchronize(streams[static_cast<std::size_t>(p)]));
    }
    std::int64_t full = 0;
    for (std::int32_t l = 2; l < 4; ++l) {
        for (std::int32_t p = 0; p < P; ++p) { full += pol.occupied(l, p) == pol.capacity(l, p); }
    }
    check(full == 2 * P, "keep: reclaimed layers are full again");
    check(cache.published_slot(2, 1) >= 0 && cache.published_slot(3, 60) >= 0 &&
              cache.published_slot(3, 63) >= 0,
          "keep: reclaimed layers hold their profile's hottest experts");
    check_final_state(cache, banks4, pools);
    std::printf("  keep-elastic lend/reclaim (%d GPU): kept %zu elastic residents, lent-layer "
                "inserts %zu, slots read while lent %llu\n",
                P, kept.size(), lent_inserts, reads);
    for (std::int32_t p = 0; p < P; ++p) {
        (void)cudaStreamDestroy(streams[static_cast<std::size_t>(p)]);
        (void)cudaStreamDestroy(rings[static_cast<std::size_t>(p)]);
    }
}

// The prefill ring in elastic columns: 4 layers x 64 experts, base 8 slots per layer + 4 elastic
// columns on one GPU. The cache shrinks to one column (the ring holds the other three), layers
// 1..3 lend their base slots as workspace, the user overwrites every slot of the ring's columns
// and every lent base slot, then the columns come back: top_up completes the layer that was not
// lent and the reclaim the lent ones, each from the residency saved before the shrink. Every
// layer holds that residency again (tables == policy, exact slot bytes).
void column_ring_scenario(int device, const Banks& banks4) {
    PolicyConfig pc;
    pc.layers          = 4;
    pc.experts         = 64;
    pc.partitions      = 1;
    pc.slots_per_layer = 8;
    pc.elastic_columns = 4;
    GpuPool pool(device, ExpertCache::pool_slots(pc, 0), ExpertCache::table_entries(pc), false);
    std::vector<GpuPool*> pools{&pool};
    std::vector<GpuBinding> bindings{pool.binding};
    ExpertCache cache(pc, make_layout(banks4, false), bindings, 4 * kSlotBytes);
    std::vector<std::vector<std::int32_t>> profile(4);
    for (std::int32_t l = 0; l < 4; ++l) {
        for (std::int32_t e = 0; e < 64; ++e) {
            profile[static_cast<std::size_t>(l)].push_back((e * 5 + l) % 64);
        }
    }
    cache.initialize(profile);
    const ExpertCachePolicy& pol = cache.policy();
    CUDA_CHECK(cudaSetDevice(device));
    cudaStream_t stream = nullptr, ring = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ring, cudaStreamNonBlocking));
    const std::vector<cudaStream_t> streams{stream};
    cache.begin_window(streams);
    cache.end_window(streams);
    // Most recently used first, as the backend saves it.
    std::vector<std::vector<std::int32_t>> saved(4);
    for (std::int32_t l = 0; l < 4; ++l) {
        const auto lru = pol.recency(l, 0);
        saved[static_cast<std::size_t>(l)].assign(lru.rbegin(), lru.rend());
        check(saved[static_cast<std::size_t>(l)].size() == 12, "column ring: layers start full");
    }
    cache.set_active_columns(1, streams);
    const std::int32_t base_end = pc.layers * pol.base_slots(0);
    for (std::int32_t l = 0; l < 4; ++l) {
        for (std::int32_t e = 0; e < 64; ++e) {
            check(pol.slot_of(l, e) < base_end + pc.layers,
                  "column ring: no resident in the ring's columns");
        }
    }
    const auto events = cache.lend_layers(1, 3, /*keep_elastic=*/true);
    CUDA_CHECK(cudaStreamWaitEvent(ring, events[0], 0));
    const auto scribble = [&](std::int64_t first, std::int64_t end) {
        for (int q = 0; q < kPlanes; ++q) {
            for (std::int64_t sl = first; sl < end; ++sl) {
                scribble_kernel<<<64, 256, 0, ring>>>(
                    reinterpret_cast<std::uint64_t*>(pool.binding.pool_base[q] +
                                                     sl * (kPlaneBytes[q] + kPoolGap)),
                    kPlaneBytes[q] / 8);
            }
        }
    };
    scribble(pol.base_offset(0, 1), base_end);                // the workspace
    scribble(base_end + pc.layers, base_end + 4 * pc.layers); // the ring: columns 1..3
    cache.begin_window(streams);
    cache.end_window(streams);
    CUDA_CHECK(cudaStreamSynchronize(ring));
    cache.set_active_columns(4, streams);
    cache.top_up(0, saved[0]);
    cache.top_up(2, saved[2]); // a lent layer: left to its reclaim
    cache.reclaim_layers(std::span<const std::vector<std::int32_t>>(saved.data() + 1, 3));
    check(!cache.lending(), "column ring: reclaimed");
    cache.flush();
    cache.begin_window(streams); // publish
    cache.end_window(streams);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::int64_t restored = 0;
    for (std::int32_t l = 0; l < 4; ++l) {
        check(pol.occupied(l, 0) == pol.capacity(l, 0) && pol.capacity(l, 0) == 12,
              "column ring: every layer is full again");
        for (const std::int32_t e : saved[static_cast<std::size_t>(l)]) {
            restored += cache.published_slot(l, e) >= 0;
        }
    }
    check(restored == 4 * 12, "column ring: every layer holds its saved residency (" +
                                  std::to_string(restored) + " of 48)");
    check_final_state(cache, banks4, pools);
    std::printf("  ring in elastic columns: %lld residents restored\n",
                static_cast<long long>(restored));
    (void)cudaStreamDestroy(stream);
    (void)cudaStreamDestroy(ring);
}

// --bench: fill bandwidth. Layer 0 only; each repetition routes a fresh set of experts that
// misses everywhere, so every slot of every GPU is refilled (`per_gpu` experts per GPU).
void bench(const std::vector<int>& devices, std::int32_t per_gpu, int reps) {
    const auto P               = static_cast<std::int32_t>(devices.size());
    const std::int32_t experts = 512;
    Banks banks(1, experts);
    PolicyConfig pc;
    pc.layers          = 1;
    pc.experts         = experts;
    pc.partitions      = P;
    pc.slots_per_layer = per_gpu * P;
    std::vector<std::unique_ptr<GpuPool>> owned;
    std::vector<GpuBinding> bindings;
    std::vector<GpuPool*> pools;
    for (std::int32_t p = 0; p < P; ++p) {
        owned.push_back(std::make_unique<GpuPool>(devices[static_cast<std::size_t>(p)], per_gpu,
                                                  experts, false));
        bindings.push_back(owned.back()->binding);
        pools.push_back(owned.back().get());
    }
    ExpertCache cache(pc, make_layout(banks, false), bindings);
    cache.initialize({});
    std::vector<cudaStream_t> streams;
    for (std::int32_t p = 0; p < P; ++p) {
        cudaStream_t s = nullptr;
        CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(p)]));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        streams.push_back(s);
    }
    std::vector<double> gbs, submit_us;
    const std::int32_t n = per_gpu * P;
    for (int r = 0; r < reps + 1; ++r) {
        // Alternate between two disjoint expert sets of n experts each (n <= 256).
        std::vector<Routed> routed;
        for (std::int32_t e = (r % 2) * n; e < (r % 2) * n + n; ++e) {
            routed.push_back(Routed{e, 1});
        }
        cache.begin_window(streams);
        cache.end_window(streams);
        (void)cache.observe(0, routed, 1);
        const std::int64_t bytes0 = cache.counters().fill_bytes;
        const auto t0             = std::chrono::steady_clock::now();
        cache.submit_fills();
        const auto t1 = std::chrono::steady_clock::now();
        cache.flush();
        const auto t2 = std::chrono::steady_clock::now();
        if (r == 0) { continue; } // warm-up
        const double secs = std::chrono::duration<double>(t2 - t0).count();
        gbs.push_back(double(cache.counters().fill_bytes - bytes0) / secs / 1e9);
        submit_us.push_back(std::chrono::duration<double>(t1 - t0).count() * 1e6 / n);
    }
    cache.begin_window(streams); // publish the last fills before checking the tables
    cache.end_window(streams);
    for (std::int32_t p = 0; p < P; ++p) {
        CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(p)]));
        CUDA_CHECK(cudaStreamSynchronize(streams[static_cast<std::size_t>(p)]));
    }
    std::sort(gbs.begin(), gbs.end());
    std::sort(submit_us.begin(), submit_us.end());
    std::printf("  batch fill, %d GPU(s) %s, %d experts/GPU (%.0f MB/GPU): %.2f GB/s aggregate "
                "(%.2f per link) median of %d [min %.2f max %.2f]; submit %.2f us/expert\n",
                P, P == 1 ? std::to_string(devices[0]).c_str() : "0+1", per_gpu,
                per_gpu * kSlotBytes / 1e6, gbs[gbs.size() / 2], gbs[gbs.size() / 2] / P, reps,
                gbs.front(), gbs.back(), submit_us[submit_us.size() / 2]);
    check_final_state(cache, banks, pools);

    // Baseline: the same copies as four cudaMemcpyAsync per expert on one stream per GPU.
    std::vector<double> raw_gbs, raw_us;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (std::int32_t p = 0; p < P; ++p) {
            CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(p)]));
            for (std::int32_t i = 0; i < per_gpu; ++i) {
                const std::int32_t e = i * P + p;
                for (int q = 0; q < kPlanes; ++q) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        pools[static_cast<std::size_t>(p)]->binding.pool_base[q] +
                            i * (kPlaneBytes[q] + kPoolGap),
                        banks.base(0, q) + e * (kPlaneBytes[q] + kBankPad),
                        static_cast<std::size_t>(kPlaneBytes[q]), cudaMemcpyHostToDevice,
                        streams[static_cast<std::size_t>(p)]));
                }
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        for (std::int32_t p = 0; p < P; ++p) {
            CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(p)]));
            CUDA_CHECK(cudaStreamSynchronize(streams[static_cast<std::size_t>(p)]));
        }
        const auto t2 = std::chrono::steady_clock::now();
        raw_gbs.push_back(double(n) * kSlotBytes / std::chrono::duration<double>(t2 - t0).count() /
                          1e9);
        raw_us.push_back(std::chrono::duration<double>(t1 - t0).count() * 1e6 / n);
    }
    std::sort(raw_gbs.begin(), raw_gbs.end());
    std::sort(raw_us.begin(), raw_us.end());
    std::printf("  raw cudaMemcpyAsync x4/expert, same copies: %.2f GB/s aggregate (%.2f per link) "
                "[min %.2f max %.2f]; issue %.2f us/expert\n",
                raw_gbs[raw_gbs.size() / 2], raw_gbs[raw_gbs.size() / 2] / P, raw_gbs.front(),
                raw_gbs.back(), raw_us[raw_us.size() / 2]);
    for (std::int32_t p = 0; p < P; ++p) {
        CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(p)]));
        (void)cudaStreamDestroy(streams[static_cast<std::size_t>(p)]);
    }
}

} // namespace

int main(int argc, char** argv) {
    int count             = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver ||
        (err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(err);
    const bool run_bench = argc > 1 && std::string(argv[1]) == "--bench";
    try {
        if (run_bench) {
            const std::int32_t per_gpu = argc > 2 ? std::stoi(argv[2]) : 128;
            const int reps             = argc > 3 ? std::stoi(argv[3]) : 10;
            for (int d = 0; d < std::min(count, 2); ++d) { bench({d}, per_gpu, reps); }
            if (count >= 2) { bench({0, 1}, per_gpu, reps); }
        } else {
            Banks banks(3, 64);
            std::printf("expert cache GPU test: %d device(s)\n", count);
            PolicyConfig lru;
            lru.layers                 = 3;
            lru.experts                = 64;
            lru.partitions             = 1;
            lru.slots_per_layer        = 10;
            const std::int64_t shallow = 8 * kSlotBytes;
            std::printf(" 1 GPU, LRU, one array per plane, fill depth 8 experts:\n");
            (void)run_windows({0}, lru, false, banks, 60, 4, 6, 200'000, 11, shallow);
            PolicyConfig adm     = lru;
            adm.admission        = true;
            adm.admit_threshold  = 1.5;
            adm.half_life_tokens = 16.0;
            adm.insert_cap       = 3;
            std::printf(" 1 GPU, admission + cap 3, contiguous slots:\n");
            (void)run_windows({0}, adm, true, banks, 60, 4, 6, 200'000, 12,
                              ExpertCache::kDefaultInFlightBytes);
            if (count >= 2) {
                PolicyConfig two = lru;
                two.partitions   = 2;
                std::printf(" 2 GPUs, LRU partitioned by id, one array per plane, depth 8:\n");
                (void)run_windows({0, 1}, two, false, banks, 60, 4, 6, 200'000, 13, shallow);
                two.insert_cap = 4;
                std::printf(" 2 GPUs, LRU + cap 4, contiguous slots:\n");
                (void)run_windows({0, 1}, two, true, banks, 60, 4, 6, 200'000, 14,
                                  ExpertCache::kDefaultInFlightBytes);
            } else {
                std::printf(" (one GPU: the two-GPU scenarios did not run)\n");
            }
            PolicyConfig ela    = adm;
            ela.slots_per_layer = 6;
            ela.elastic_columns = 3;
            std::printf(" 1 GPU, admission + cap 3, 6 base slots + 3 elastic columns resized every "
                        "6 windows:\n");
            (void)run_windows({0}, ela, false, banks, 90, 4, 6, 100'000, 15,
                              ExpertCache::kDefaultInFlightBytes);
            if (count >= 2) {
                PolicyConfig ela2 = ela;
                ela2.partitions   = 2;
                ela2.admission    = false;
                ela2.insert_cap   = 0;
                std::printf(" 2 GPUs, LRU, 6 base slots + 3 elastic columns per GPU, resized:\n");
                (void)run_windows({0, 1}, ela2, false, banks, 90, 4, 6, 100'000, 16,
                                  ExpertCache::kDefaultInFlightBytes);
            }
            // Pinned fallback: layer 1 pageable, a 2-slot staging ring (backpressure).
            setenv("NINFER_HOST_STAGING_MIB", "6", 1);
            {
                Banks mixed(3, 64, {1});
                std::printf(" 1 GPU, layer 1 pageable (2-slot staging ring), admission + cap 3:\n");
                (void)run_windows({0}, adm, false, mixed, 60, 4, 6, 200'000, 17,
                                  ExpertCache::kDefaultInFlightBytes);
                if (count >= 2) {
                    PolicyConfig two = lru;
                    two.partitions   = 2;
                    std::printf(" 2 GPUs, layer 1 pageable (2-slot ring per GPU), LRU:\n");
                    (void)run_windows({0, 1}, two, true, mixed, 60, 4, 6, 200'000, 18, shallow);
                }
                Banks all(3, 64, {0, 1, 2});
                std::printf(" 1 GPU, every layer pageable, 6 base + 3 elastic resized:\n");
                (void)run_windows({0}, ela, false, all, 90, 4, 6, 100'000, 19,
                                  ExpertCache::kDefaultInFlightBytes);
            }
            unsetenv("NINFER_HOST_STAGING_MIB");
            fence_scenario(0, banks);
            unfenced_scenario(0, banks, false);
            unfenced_scenario(0, banks, true);
            Banks banks4(4, 64);
            lend_scenario(0, banks4);
            lend_keep_scenario({0}, banks4);
            column_ring_scenario(0, banks4);
            if (count >= 2) { lend_keep_scenario({0, 1}, banks4); }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::printf("expert cache GPU test: all checks passed\n");
    return 0;
}
