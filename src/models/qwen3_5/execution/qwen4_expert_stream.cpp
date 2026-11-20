#include "models/qwen3_5/execution/qwen4_expert_stream.h"

#include "core/device.h"       // CUDA_CHECK
#include "core/device_scope.h" // ScopedCurrentDevice

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {

Qwen4ExpertStream::Qwen4ExpertStream(Config config) : config_(std::move(config)) {
    const Config& c = config_;
    if (c.layers <= 0 || c.experts <= 0 || c.partitions < 1 || c.tables == nullptr ||
        c.bank_base.size() != static_cast<std::size_t>(c.layers) * 4 || c.partitions < 1 ||
        c.partition < 0 || c.partition >= c.partitions ||
        (!c.streamable.empty() && c.streamable.size() != static_cast<std::size_t>(c.layers))) {
        throw std::invalid_argument("Qwen4Exp expert stream: invalid configuration");
    }
    for (std::size_t q = 0; q < 4; ++q) {
        if (config_.pool_stride[q] == 0) { config_.pool_stride[q] = c.plane_bytes[q]; }
        if (c.plane_bytes[q] <= 0 || c.pool_base[q] == nullptr ||
            c.pool_stride[q] < c.plane_bytes[q]) {
            throw std::invalid_argument("Qwen4Exp expert stream: invalid plane");
        }
        expert_bytes_ += c.plane_bytes[q];
    }
    ScopedCurrentDevice scope(c.device);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
    ready_.assign(static_cast<std::size_t>(c.layers), nullptr);
    done_.assign(static_cast<std::size_t>(c.layers), nullptr);
    mixed_.assign(static_cast<std::size_t>(c.layers), nullptr);
    exchanged_.assign(static_cast<std::size_t>(c.layers), nullptr);
    for (std::int32_t l = 0; l < c.layers; ++l) {
        CUDA_CHECK(
            cudaEventCreateWithFlags(&ready_[static_cast<std::size_t>(l)], cudaEventDisableTiming));
        CUDA_CHECK(
            cudaEventCreateWithFlags(&done_[static_cast<std::size_t>(l)], cudaEventDisableTiming));
        CUDA_CHECK(
            cudaEventCreateWithFlags(&mixed_[static_cast<std::size_t>(l)], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&exchanged_[static_cast<std::size_t>(l)],
                                            cudaEventDisableTiming));
    }
    void* host = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host,
                             static_cast<std::size_t>(c.layers) * c.experts * sizeof(std::int32_t),
                             cudaHostAllocPortable));
    host_tables_ = static_cast<std::int32_t*>(host);
    issued_.assign(static_cast<std::size_t>(c.layers), 0);
    next_.assign(static_cast<std::size_t>(c.layers), 0);
    recorded_.assign(static_cast<std::size_t>(c.layers), 0);
    selection_.assign(static_cast<std::size_t>(c.layers), nullptr);
}

Qwen4ExpertStream::~Qwen4ExpertStream() {
    if (cudaSetDevice(config_.device) != cudaSuccess) { return; }
    if (copy_ != nullptr) { (void)cudaStreamSynchronize(copy_); }
    for (cudaEvent_t e : ready_) {
        if (e != nullptr) { (void)cudaEventDestroy(e); }
    }
    for (cudaEvent_t e : done_) {
        if (e != nullptr) { (void)cudaEventDestroy(e); }
    }
    for (cudaEvent_t e : mixed_) {
        if (e != nullptr) { (void)cudaEventDestroy(e); }
    }
    for (cudaEvent_t e : exchanged_) {
        if (e != nullptr) { (void)cudaEventDestroy(e); }
    }
    if (copy_ != nullptr) { (void)cudaStreamDestroy(copy_); }
    if (host_tables_ != nullptr) { (void)cudaFreeHost(host_tables_); }
}

void Qwen4ExpertStream::begin_window(const expert_cache::ExpertCache* cache, cudaEvent_t wait_first,
                                     std::int32_t ring_first, std::int32_t ring_slots) {
    if (ring_first < 0 ||
        ring_slots < (config_.experts + config_.partitions - 1) / config_.partitions) {
        throw std::invalid_argument("Qwen4Exp expert stream: the ring needs at least one layer of "
                                    "this device's partition");
    }
    cache_      = cache;
    wait_first_ = wait_first;
    ring_first_ = ring_first;
    ring_slots_ = ring_slots;
    // The previous window's last expert kernel (stream order covers the earlier ones) still owns
    // the ring until it completes.
    if (last_recorded_ >= 0) {
        ScopedCurrentDevice scope(config_.device);
        CUDA_CHECK(cudaStreamWaitEvent(copy_, done_[static_cast<std::size_t>(last_recorded_)], 0));
        last_recorded_ = -1;
    }
    open_ = true;
    std::fill(issued_.begin(), issued_.end(), 0);
    std::fill(next_.begin(), next_.end(), 0);
    std::fill(recorded_.begin(), recorded_.end(), 0);
    std::fill(selection_.begin(), selection_.end(), nullptr);
    owner_.assign(static_cast<std::size_t>(ring_slots), -1);
    cursor_    = 0;
    waited_    = -1;
    wait_next_ = nullptr;
    stats_     = Stats{};
}

void Qwen4ExpertStream::set_selection(std::int32_t layer, const std::uint8_t* stream_expert) {
    if (layer < 0 || layer >= config_.layers || issued(layer)) {
        throw std::logic_error("Qwen4Exp expert stream: selection after the layer's copies");
    }
    selection_[static_cast<std::size_t>(layer)] = stream_expert;
}

void Qwen4ExpertStream::wait_for(std::int32_t owner) {
    if (owner <= waited_) { return; }
    if (recorded_[static_cast<std::size_t>(owner)] == 0) {
        throw std::logic_error("Qwen4Exp expert stream: ring slot still owned by a layer that has "
                               "not run (ring smaller than one layer?)");
    }
    CUDA_CHECK(cudaStreamWaitEvent(copy_, done_[static_cast<std::size_t>(owner)], 0));
    waited_ = owner;
}

void Qwen4ExpertStream::issue(std::int32_t layer, bool enqueued_owners_only) {
    if (!open_) { throw std::logic_error("Qwen4Exp expert stream: no open window"); }
    if (!streams(layer)) {
        throw std::logic_error("Qwen4Exp expert stream: layer " + std::to_string(layer) +
                               " is not streamed (pageable bank)");
    }
    const Config& c = config_;
    ScopedCurrentDevice scope(c.device);
    if (wait_first_ != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(copy_, wait_first_, 0));
        wait_first_ = nullptr;
    }
    if (wait_next_ != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(copy_, wait_next_, 0));
        wait_next_ = nullptr;
    }
    std::int32_t* table        = host_tables_ + static_cast<std::size_t>(layer) * c.experts;
    const std::uint8_t* select = selection_[static_cast<std::size_t>(layer)];
    const bool lent            = cache_ == nullptr || cache_->lent(layer);
    // Runs of consecutive experts of this partition (e, e + partitions, ...) placed in
    // consecutive ring slots: one copy per plane each (2D when the bank or the pool strides).
    std::int32_t run_expert = -1, run_slot = -1, run_length = 0;
    const auto flush = [&] {
        if (run_length == 0) { return; }
        for (std::size_t q = 0; q < 4; ++q) {
            const std::int64_t bytes  = c.plane_bytes[q];
            const std::int64_t dpitch = c.pool_stride[q];
            const std::int64_t spitch = bytes * c.partitions;
            std::byte* dst =
                c.pool_base[q] + static_cast<std::int64_t>(ring_first_ + run_slot) * dpitch;
            const std::byte* src = c.bank_base[static_cast<std::size_t>(layer) * 4 + q] +
                                   static_cast<std::int64_t>(run_expert) * bytes;
            if (dpitch == bytes && spitch == bytes) {
                CUDA_CHECK(cudaMemcpyAsync(dst, src,
                                           static_cast<std::size_t>(bytes) *
                                               static_cast<std::size_t>(run_length),
                                           cudaMemcpyHostToDevice, copy_));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    dst, static_cast<std::size_t>(dpitch), src, static_cast<std::size_t>(spitch),
                    static_cast<std::size_t>(bytes), static_cast<std::size_t>(run_length),
                    cudaMemcpyHostToDevice, copy_));
            }
        }
        stats_.streamed += run_length;
        stats_.bytes += expert_bytes_ * run_length;
        run_length = 0;
    };
    issued_[static_cast<std::size_t>(layer)] = 1;
    std::int32_t e                           = next_[static_cast<std::size_t>(layer)];
    for (; e < c.experts; ++e) {
        if (e % c.partitions != c.partition) {
            table[e] = -1; // the other GPU's expert: no copy, the run continues past it
            continue;
        }
        const std::int32_t resident = lent ? -1 : cache_->published_slot(layer, e);
        if (resident >= 0) {
            table[e] = resident;
            ++stats_.resident;
            continue;
        }
        if (select != nullptr && select[e] == 0) {
            table[e] = -1;
            ++stats_.host;
            continue;
        }
        const std::int32_t position = cursor_;
        const std::int32_t owner    = owner_[static_cast<std::size_t>(position)];
        if (enqueued_owners_only && owner >= 0 && recorded_[static_cast<std::size_t>(owner)] == 0) {
            break;
        }
        cursor_               = cursor_ + 1 == ring_slots_ ? 0 : cursor_ + 1;
        const bool contiguous = run_length > 0 && run_expert + run_length * c.partitions == e &&
                                run_slot + run_length == position;
        if (!contiguous) { flush(); }
        if (owner >= 0) { wait_for(owner); }
        owner_[static_cast<std::size_t>(position)] = layer;
        if (run_length == 0) {
            run_expert = e;
            run_slot   = position;
        }
        ++run_length;
        table[e] = ring_first_ + position;
    }
    flush();
    next_[static_cast<std::size_t>(layer)] = e;
    if (e < c.experts) { return; }
    CUDA_CHECK(cudaMemcpyAsync(config_.tables + static_cast<std::size_t>(layer) * c.experts, table,
                               sizeof(std::int32_t) * static_cast<std::size_t>(c.experts),
                               cudaMemcpyHostToDevice, copy_));
    CUDA_CHECK(cudaEventRecord(ready_[static_cast<std::size_t>(layer)], copy_));
    ++stats_.layers;
}

void Qwen4ExpertStream::acquire(std::int32_t layer, cudaStream_t compute) {
    if (layer < 0 || layer >= config_.layers) {
        throw std::out_of_range("Qwen4Exp expert stream: layer " + std::to_string(layer));
    }
    if (!complete(layer)) { issue(layer); }
    CUDA_CHECK(cudaStreamWaitEvent(compute, ready_[static_cast<std::size_t>(layer)], 0));
}

void Qwen4ExpertStream::release(std::int32_t layer, cudaStream_t compute) {
    CUDA_CHECK(cudaEventRecord(done_[static_cast<std::size_t>(layer)], compute));
    recorded_[static_cast<std::size_t>(layer)] = 1;
    last_recorded_                             = layer;
    if (defer_) { return; }
    // The next streamed layer (a pageable layer in between runs on the CPU path, no copies).
    std::int32_t next = layer + 1;
    while (next < config_.layers && !streams(next)) { ++next; }
    if (next < config_.layers && !complete(next)) { issue(next); }
}

void Qwen4ExpertStream::mixer_exchanged(std::int32_t layer, cudaStream_t compute) {
    if (!defer_ || !open_ || layer < 0 || layer + 1 >= config_.layers || issued(layer + 1) ||
        !streams(layer + 1)) {
        return;
    }
    const ScopedCurrentDevice scope(config_.device);
    CUDA_CHECK(cudaEventRecord(mixed_[static_cast<std::size_t>(layer)], compute));
    wait_next_ = mixed_[static_cast<std::size_t>(layer)];
    issue(layer + 1, /*enqueued_owners_only=*/true);
}

void Qwen4ExpertStream::exchanged(std::int32_t layer, cudaStream_t compute) {
    if (!defer_ || !open_ || layer < 0 || layer + 1 >= config_.layers || complete(layer + 1) ||
        !streams(layer + 1)) {
        return;
    }
    const ScopedCurrentDevice scope(config_.device);
    CUDA_CHECK(cudaEventRecord(exchanged_[static_cast<std::size_t>(layer)], compute));
    wait_next_ = exchanged_[static_cast<std::size_t>(layer)];
    issue(layer + 1);
}

void Qwen4ExpertStream::end_window() noexcept {
    if (copy_ != nullptr && cudaSetDevice(config_.device) == cudaSuccess) {
        (void)cudaStreamSynchronize(copy_);
    }
    cache_ = nullptr;
    open_  = false;
}

} // namespace ninfer::models::qwen3_5::execution
