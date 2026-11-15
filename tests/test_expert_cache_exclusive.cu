// GPU test of the exclusive tier of ExpertCache. Every expert's bytes are a hash of (layer,
// expert, word). Over random skewed routing it checks every 25 windows that each expert has intact
// bytes where the cache says: the device slot while published, else the host pool slot; and at the
// end that the pool lost no slot. The elastic columns are shrunk to none and grown back every 400
// windows: after a shrink no expert is published in a released column and every expert is intact.
// Exit 77 when no GPU is usable.

#include "core/device.h"
#include "core/expert_cache.h"
#include "core/expert_host_pool.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::expert_cache;

constexpr std::int32_t kLayers  = 3;
constexpr std::int32_t kExperts = 64;
constexpr std::int32_t kSlots   = 16; // one GPU, per layer
constexpr std::int32_t kSpare   = 4;
constexpr std::int32_t kColumns = 4; // elastic columns (inclusive slots)
constexpr std::int64_t kBytes   = 1 << 16;
constexpr std::int64_t kWords   = kBytes / 4;

int failures = 0;

std::uint32_t word(std::int32_t l, std::int32_t e, std::int64_t i) {
    std::uint32_t h = 2166136261U;
    for (const std::uint32_t v : {static_cast<std::uint32_t>(l), static_cast<std::uint32_t>(e),
                                  static_cast<std::uint32_t>(i)}) {
        h = (h ^ v) * 16777619U;
    }
    return h;
}

void write_expert(std::byte* to, std::int32_t l, std::int32_t e) {
    auto* w = reinterpret_cast<std::uint32_t*>(to);
    for (std::int64_t i = 0; i < kWords; ++i) { w[i] = word(l, e, i); }
}

bool intact(const std::byte* at, std::int32_t l, std::int32_t e) {
    const auto* w = reinterpret_cast<const std::uint32_t*>(at);
    for (std::int64_t i = 0; i < kWords; ++i) {
        if (w[i] != word(l, e, i)) { return false; }
    }
    return true;
}

void must(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    must(cudaSetDevice(0));

    PolicyConfig policy;
    policy.layers          = kLayers;
    policy.experts         = kExperts;
    policy.partitions      = 1;
    policy.slots_per_layer = kSlots;
    policy.elastic_columns = kColumns;

    // Layers 0 and 1 are exclusive; layer 2 stays inclusive, as a lent layer does.
    const std::vector<std::int32_t> host_slots = {kExperts - kSlots + kSpare,
                                                  kExperts - kSlots + kSpare, kExperts};
    ExpertHostPool pool(kLayers, kExperts, host_slots);

    void* bank_memory    = nullptr;
    const auto bank_size = static_cast<std::size_t>(kLayers) * kExperts * kBytes;
    must(cudaHostAlloc(&bank_memory, bank_size, cudaHostAllocPortable));
    auto* bank = static_cast<std::byte*>(bank_memory);
    std::vector<std::byte> tail(static_cast<std::size_t>(kLayers) * kSlots * kBytes);

    ExpertLayout layout;
    layout.planes         = 1;
    layout.plane_bytes[0] = kBytes;
    layout.bank_stride[0] = kBytes;
    layout.pool_stride[0] = kBytes;
    layout.host_pool      = &pool;
    for (std::int32_t l = 0; l < kLayers; ++l) {
        layout.bank_base.push_back(bank + static_cast<std::int64_t>(l) * kExperts * kBytes);
        const bool exclusive = l < 2;
        layout.exclusive_layer.push_back(exclusive ? 1 : 0);
        layout.tail_first.push_back(exclusive ? kExperts - kSlots : kExperts);
        layout.tail_base.push_back(tail.data() + static_cast<std::int64_t>(l) * kSlots * kBytes);
        const std::int32_t held = exclusive ? kExperts - kSlots : kExperts;
        for (std::int32_t e = 0; e < held; ++e) {
            const std::int32_t s = pool.reserve(l, e);
            write_expert(bank + (static_cast<std::int64_t>(l) * kExperts + s) * kBytes, l, e);
        }
        for (std::int32_t e = held; e < kExperts; ++e) {
            write_expert(tail.data() + (static_cast<std::int64_t>(l) * kSlots + e - held) * kBytes,
                         l, e);
        }
    }

    const auto device_slots = static_cast<std::size_t>(ExpertCache::pool_slots(policy, 0));
    void* device_pool       = nullptr;
    void* device_table      = nullptr;
    must(cudaMalloc(&device_pool, device_slots * kBytes));
    must(cudaMalloc(&device_table,
                    static_cast<std::size_t>(kLayers) * kExperts * sizeof(std::int32_t)));
    cudaStream_t stream = nullptr;
    must(cudaStreamCreate(&stream));

    GpuBinding binding;
    binding.device       = 0;
    binding.pool_base[0] = static_cast<std::byte*>(device_pool);
    binding.table        = static_cast<std::int32_t*>(device_table);
    const std::array<GpuBinding, 1> bindings{binding};
    ExpertCache cache(policy, layout, bindings, kBytes * 8);

    // Timed publication (lag -1) is refused with a host pool, by the setter and from the
    // environment.
    bool refused = false;
    try {
        cache.set_publish_lag(-1);
    } catch (const std::invalid_argument&) { refused = true; }
    const char* saved_lag = std::getenv("NINFER_QWEN4_FILL_PUBLISH_LAG");
    const std::string saved(saved_lag != nullptr ? saved_lag : "");
    setenv("NINFER_QWEN4_FILL_PUBLISH_LAG", "-1", 1);
    try {
        ExpertCache timed(policy, layout, bindings, kBytes * 8);
        refused = false;
    } catch (const std::invalid_argument& e) {
        refused = refused && std::strstr(e.what(), "NINFER_QWEN4_FILL_PUBLISH_LAG") != nullptr;
    }
    if (saved_lag != nullptr) {
        setenv("NINFER_QWEN4_FILL_PUBLISH_LAG", saved.c_str(), 1);
    } else {
        unsetenv("NINFER_QWEN4_FILL_PUBLISH_LAG");
    }
    if (!refused) {
        std::fprintf(stderr, "publish lag -1 accepted with the exclusive tier\n");
        ++failures;
    }

    std::vector<std::vector<std::int32_t>> profile(kLayers);
    for (std::int32_t l = 0; l < 2; ++l) {
        for (std::int32_t e = kExperts - kSlots; e < kExperts; ++e) { profile[l].push_back(e); }
    }
    cache.initialize(profile);
    cache.release_tails();
    tail.assign(tail.size(), std::byte{0}); // anything still read from it fails the hash

    const std::array<cudaStream_t, 1> streams{stream};
    std::vector<std::byte> copy(kBytes);
    std::mt19937 random(11);
    const auto check_all = [&](int window) {
        must(cudaStreamSynchronize(stream));
        cache.poll();
        for (std::int32_t l = 0; l < kLayers; ++l) {
            for (std::int32_t e = 0; e < kExperts; ++e) {
                const std::int32_t s = cache.published_slot(l, e);
                if (s >= 0) {
                    must(cudaMemcpy(copy.data(), static_cast<std::byte*>(device_pool) + s * kBytes,
                                    kBytes, cudaMemcpyDeviceToHost));
                    if (!intact(copy.data(), l, e)) {
                        std::fprintf(stderr, "window %d: layer %d expert %d wrong in slot %d\n",
                                     window, l, e, s);
                        ++failures;
                    }
                } else {
                    const std::byte* host = layout.bank(l, 0, e);
                    if (host == nullptr || !intact(host, l, e)) {
                        std::fprintf(stderr,
                                     "window %d: layer %d expert %d lost (host slot %d, %s)\n",
                                     window, l, e, pool.slot(l, e),
                                     host == nullptr ? "no bytes" : "bad bytes");
                        ++failures;
                    }
                }
            }
        }
    };

    constexpr int kWindows         = 1500;
    std::int64_t shrink_writebacks = 0;
    for (int window = 0; window < kWindows && failures < 8; ++window) {
        cache.begin_window(streams);
        cache.end_window(streams);
        must(cudaStreamSynchronize(stream));
        for (std::int32_t l = 0; l < kLayers; ++l) {
            // The hot set drifts, so experts keep trading places.
            std::vector<Routed> routed;
            std::vector<std::int32_t> ids;
            for (int i = 0; i < 12; ++i) {
                const auto u = static_cast<double>(random()) / std::mt19937::max();
                ids.push_back((window / 20 + static_cast<std::int32_t>(u * u * 40)) % kExperts);
            }
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            for (const std::int32_t e : ids) { routed.push_back({e, 2}); }
            (void)cache.observe(l, routed, 8);
        }
        cache.submit_fills();
        if (window % 400 == 300) {
            // Settled first, so the write-backs counted next are the shrink's own evictions.
            cache.settle(streams);
            const std::int64_t before = cache.counters().writebacks;
            cache.set_active_columns(0, streams);
            shrink_writebacks += cache.counters().writebacks - before;
            for (std::int32_t l = 0; l < kLayers; ++l) {
                for (std::int32_t e = 0; e < kExperts; ++e) {
                    if (cache.published_slot(l, e) >= kLayers * kSlots) {
                        std::fprintf(stderr, "window %d: layer %d expert %d in a released column\n",
                                     window, l, e);
                        ++failures;
                    }
                }
            }
            check_all(window);
        } else if (window % 400 == 350) {
            cache.set_active_columns(kColumns, streams);
        }
        if (window % 25 == 0) { check_all(window); }
    }
    cache.settle(streams);
    check_all(kWindows);

    const auto& c = cache.counters();
    std::printf("windows %lld, fills %lld, write-backs %lld, refound %lld, dropped %lld, shrinks: "
                "%lld evicted, %lld moved, %lld written back\n",
                static_cast<long long>(c.windows), static_cast<long long>(c.fills_issued),
                static_cast<long long>(c.writebacks), static_cast<long long>(c.refound),
                static_cast<long long>(c.fills_dropped), static_cast<long long>(c.shrink_evicted),
                static_cast<long long>(c.moved), static_cast<long long>(shrink_writebacks));
    if (c.writebacks < 100) {
        std::fprintf(stderr, "too few write-backs to mean anything\n");
        ++failures;
    }
    if (c.shrink_evicted == 0 || shrink_writebacks == 0) {
        std::fprintf(stderr, "the shrinks wrote back no exclusive expert\n");
        ++failures;
    }
    // Device copies of exclusive layers plus host slots account for every expert.
    for (std::int32_t l = 0; l < 2; ++l) {
        pool.retire(~std::uint64_t{0});
        std::int32_t with_host = 0;
        for (std::int32_t e = 0; e < kExperts; ++e) {
            with_host += pool.slot(l, e) != ExpertHostPool::kNoSlot ? 1 : 0;
        }
        if (static_cast<std::size_t>(with_host) + pool.free_slots(l) !=
            static_cast<std::size_t>(host_slots[static_cast<std::size_t>(l)])) {
            std::fprintf(stderr, "layer %d: host slots leaked\n", l);
            ++failures;
        }
    }
    must(cudaFree(device_pool));
    must(cudaFree(device_table));
    must(cudaFreeHost(bank_memory));
    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::puts("ok");
    return 0;
}
