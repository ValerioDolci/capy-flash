#include "models/qwen3_5/state/state_image.h"

#include "core/checked.h"
#include "core/device.h"
#include "core/device_scope.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::size_t kStateImageAlignment = 256;

std::uint32_t padded_dflash_capacity(std::uint32_t capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("StateImage DFlash capacity must be positive");
    }
    constexpr std::uint64_t alignment = 128;
    const std::uint64_t padded =
        (static_cast<std::uint64_t>(capacity) + alignment - 1U) & ~(alignment - 1U);
    if (padded > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("StateImage DFlash padded capacity exceeds int32");
    }
    return static_cast<std::uint32_t>(padded);
}

bool same_linear_spec(const LinearAttentionStatePoolSpec& left,
                      const LinearAttentionStatePoolSpec& right) noexcept {
    return left.layers == right.layers && left.conv_channels == right.conv_channels &&
           left.conv_width == right.conv_width && left.value_heads == right.value_heads &&
           left.value_head_dim == right.value_head_dim && left.key_head_dim == right.key_head_dim &&
           left.slot_count == right.slot_count && left.conv_dtype == right.conv_dtype;
}

bool same_dflash_spec(const std::optional<DFlashLocalStateSpec>& left,
                      const std::optional<DFlashLocalStateSpec>& right) noexcept {
    if (left.has_value() != right.has_value()) { return false; }
    if (!left) { return true; }
    return left->layers == right->layers && left->capacity == right->capacity &&
           left->kv_heads == right->kv_heads && left->head_dim == right->head_dim;
}

bool same_ple_spec(const std::optional<PleStateSpec>& left,
                   const std::optional<PleStateSpec>& right) noexcept {
    if (left.has_value() != right.has_value()) { return false; }
    return !left || (left->layers == right->layers && left->rows == right->rows);
}

bool same_region(const LayoutRegion& left, const LayoutRegion& right) noexcept {
    return left.offset == right.offset && left.bytes == right.bytes &&
           left.alignment == right.alignment;
}

bool same_optional_region(const std::optional<LayoutRegion>& left,
                          const std::optional<LayoutRegion>& right) noexcept {
    return left.has_value() == right.has_value() && (!left || same_region(*left, *right));
}

bool same_host_layout(const StateImageHostLayout& left,
                      const StateImageHostLayout& right) noexcept {
    return same_linear_spec(left.spec.linear, right.spec.linear) &&
           left.spec.hidden == right.spec.hidden &&
           same_dflash_spec(left.spec.dflash_local, right.spec.dflash_local) &&
           same_region(left.linear_conv, right.linear_conv) &&
           left.linear_conv_layer_bytes == right.linear_conv_layer_bytes &&
           same_region(left.linear_recurrent, right.linear_recurrent) &&
           left.linear_recurrent_layer_bytes == right.linear_recurrent_layer_bytes &&
           same_region(left.continuation_hidden, right.continuation_hidden) &&
           same_optional_region(left.dflash_local_k, right.dflash_local_k) &&
           same_optional_region(left.dflash_local_v, right.dflash_local_v) &&
           left.dflash_local_layer_bytes == right.dflash_local_layer_bytes &&
           same_ple_spec(left.spec.ple, right.spec.ple) &&
           same_optional_region(left.ple, right.ple) &&
           left.ple_layer_bytes == right.ple_layer_bytes && left.peer_offset == right.peer_offset &&
           left.peer_bytes == right.peer_bytes && left.image_bytes == right.image_bytes;
}

StateImageHostLayout plan_host_state_image(const StateImageSpec& spec) {
    if (spec.linear.layers == 0 || spec.linear.conv_channels <= 0 || spec.linear.conv_width <= 0 ||
        spec.linear.value_heads <= 0 || spec.linear.value_head_dim <= 0 ||
        spec.linear.key_head_dim <= 0 || spec.linear.slot_count <= 0 ||
        (spec.linear.conv_dtype != DType::BF16 && spec.linear.conv_dtype != DType::FP32) ||
        spec.hidden <= 0) {
        throw std::invalid_argument("StateImage host geometry is invalid");
    }
    if (spec.dflash_local && (spec.dflash_local->layers == 0 || spec.dflash_local->kv_heads <= 0 ||
                              spec.dflash_local->head_dim <= 0)) {
        throw std::invalid_argument("StateImage host DFlash geometry is invalid");
    }
    if (spec.ple && (spec.ple->layers == 0 || spec.ple->rows <= 0)) {
        throw std::invalid_argument("StateImage host PLE geometry is invalid");
    }

    LayoutBuilder builder;
    StateImageHostLayout host;
    host.spec = spec;
    const Tensor conv_slot(nullptr, spec.linear.conv_dtype,
                           {spec.linear.conv_channels, spec.linear.conv_width});
    const Tensor recurrent_slot(
        nullptr, DType::FP32,
        {spec.linear.key_head_dim, spec.linear.value_head_dim, spec.linear.value_heads});
    const Tensor hidden_slot(nullptr, DType::BF16, {spec.hidden});
    host.linear_conv_layer_bytes = conv_slot.bytes();
    host.linear_conv = builder.add(checked_mul(host.linear_conv_layer_bytes, spec.linear.layers,
                                               "StateImage host convolution bytes overflow"),
                                   kStateImageAlignment, "StateImage host convolution");
    host.linear_recurrent_layer_bytes = recurrent_slot.bytes();
    host.linear_recurrent =
        builder.add(checked_mul(host.linear_recurrent_layer_bytes, spec.linear.layers,
                                "StateImage host recurrent bytes overflow"),
                    kStateImageAlignment, "StateImage host recurrent");
    host.continuation_hidden = builder.add(hidden_slot.bytes(), kStateImageAlignment,
                                           "StateImage host continuation hidden");
    if (spec.dflash_local) {
        const Tensor local_slot(
            nullptr, DType::BF16,
            {spec.dflash_local->head_dim,
             static_cast<std::int32_t>(padded_dflash_capacity(spec.dflash_local->capacity)),
             spec.dflash_local->kv_heads});
        host.dflash_local_layer_bytes = local_slot.bytes();
        const std::size_t component_bytes =
            checked_mul(host.dflash_local_layer_bytes, spec.dflash_local->layers,
                        "StateImage host DFlash local bytes overflow");
        host.dflash_local_k =
            builder.add(component_bytes, kStateImageAlignment, "StateImage host DFlash local K");
        host.dflash_local_v =
            builder.add(component_bytes, kStateImageAlignment, "StateImage host DFlash local V");
    }
    if (spec.ple) {
        const Tensor ple_slot(nullptr, DType::BF16, {spec.ple->rows});
        host.ple_layer_bytes = ple_slot.bytes();
        host.ple             = builder.add(checked_mul(host.ple_layer_bytes, spec.ple->layers,
                                                       "StateImage host PLE bytes overflow"),
                                           kStateImageAlignment, "StateImage host PLE convolution");
    }
    host.image_bytes = builder.finish(kStateImageAlignment, "StateImage host image");
    return host;
}

std::byte* byte_offset(std::byte* base, std::size_t offset) noexcept { return base + offset; }

const std::byte* byte_offset(const std::byte* base, std::size_t offset) noexcept {
    return base + offset;
}

void validate_slot(std::int32_t slot, std::int32_t slot_count, const char* label) {
    if (slot < 0 || slot >= slot_count) { throw std::out_of_range(label); }
}

} // namespace

StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                    const StateImageSpec& spec) {
    if (spec.hidden <= 0) {
        throw std::invalid_argument("StateImage hidden width must be positive");
    }

    StateImageDeviceLayout out;
    out.linear = plan_linear_attention_state_pool(builder, spec.linear);
    out.continuation_hidden =
        builder.add_tensor(DType::BF16, {spec.hidden, spec.linear.slot_count}, kStateImageAlignment,
                           "StateImage continuation hidden");
    if (spec.dflash_local) {
        const DFlashLocalStateSpec& dflash = *spec.dflash_local;
        out.dflash_local =
            plan_cyclic_kv_cache(builder, dflash.layers, dflash.capacity, dflash.kv_heads,
                                 dflash.head_dim, spec.linear.slot_count);
    }
    if (spec.ple) {
        out.ple = builder.add_tensor(
            DType::BF16,
            {spec.ple->rows, spec.linear.slot_count, static_cast<std::int32_t>(spec.ple->layers)},
            kStateImageAlignment, "StateImage PLE convolution");
    }

    out.host = plan_host_state_image(spec);
    return out;
}

TransferWork state_image_transfer_work(const StateImageHostLayout& layout) {
    std::size_t payload = checked_add(layout.linear_conv.bytes, layout.linear_recurrent.bytes,
                                      "StateImage transfer payload overflow");
    payload             = checked_add(payload, layout.continuation_hidden.bytes,
                                      "StateImage transfer payload overflow");
    if (layout.dflash_local_k) {
        if (!layout.spec.dflash_local || !layout.dflash_local_v) {
            throw std::invalid_argument("StateImage DFlash transfer layout is incomplete");
        }
        const std::size_t component_bytes =
            checked_mul(layout.dflash_local_layer_bytes, layout.spec.dflash_local->layers,
                        "StateImage transfer payload overflow");
        payload = checked_add(
            payload, checked_mul(component_bytes, 2U, "StateImage transfer payload overflow"),
            "StateImage transfer payload overflow");
    }
    if (layout.ple) {
        payload = checked_add(payload, layout.ple->bytes, "StateImage transfer payload overflow");
    }
    const std::uint64_t operations =
        2ULL * layout.spec.linear.layers + 1ULL +
        (layout.spec.dflash_local ? 2ULL * layout.spec.dflash_local->layers : 0ULL) +
        (layout.spec.ple ? layout.spec.ple->layers : 0ULL);
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage transfer operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

TransferWork dflash_local_transfer_work(const StateImageHostLayout& layout) {
    if (!layout.spec.dflash_local || !layout.dflash_local_k || !layout.dflash_local_v) {
        throw std::invalid_argument("StateImage has no DFlash local component");
    }
    const std::size_t component_bytes =
        checked_mul(layout.dflash_local_layer_bytes, layout.spec.dflash_local->layers,
                    "StateImage DFlash transfer payload overflow");
    const std::uint64_t operations = 2ULL * layout.spec.dflash_local->layers;
    if (component_bytes > std::numeric_limits<std::uint64_t>::max() / 2U ||
        operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage DFlash transfer work exceeds its representation");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(2U * component_bytes),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

HostStatePool::HostStatePool(StateImageHostLayout layout, std::uint32_t capacity)
    : layout_(std::move(layout)), slots_(capacity), free_slots_(capacity), free_count_(capacity) {
    // A tp 2 image is rank 0's planned image followed by rank 1's (attach_mirror).
    StateImageHostLayout own = layout_;
    if (own.peer_bytes != 0) {
        own.image_bytes = own.peer_offset;
        own.peer_offset = 0;
        own.peer_bytes  = 0;
    }
    if (!same_host_layout(own, plan_host_state_image(own.spec)) ||
        (layout_.peer_bytes != 0 &&
         layout_.image_bytes != layout_.peer_offset + layout_.peer_bytes)) {
        throw std::invalid_argument("HostStatePool image layout is invalid");
    }
    const std::size_t bytes =
        checked_mul(layout_.image_bytes, capacity, "HostStatePool backing size overflow");
    if (bytes != 0) { backing_.emplace(bytes); }
    for (std::uint32_t index = 0; index < capacity; ++index) {
        free_slots_[index] = capacity - 1U - index;
    }
}

std::optional<HostStateSlotHandle> HostStatePool::allocate() noexcept {
    if (free_count_ == 0) { return std::nullopt; }
    const std::uint32_t index = free_slots_[--free_count_];
    Slot& slot                = slots_[index];
    slot.occupied             = true;
    ++occupied_;
    return HostStateSlotHandle{.index = index, .generation = slot.generation};
}

bool HostStatePool::release(HostStateSlotHandle handle) noexcept {
    if (!valid(handle)) { return false; }
    Slot& slot    = slots_[handle.index];
    slot.occupied = false;
    if (++slot.generation == 0) { ++slot.generation; }
    free_slots_[free_count_++] = handle.index;
    --occupied_;
    return true;
}

HostStateImageView HostStatePool::writable_view(HostStateSlotHandle handle) {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

HostStateImageConstView HostStatePool::view(HostStateSlotHandle handle) const {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

std::uint32_t HostStatePool::capacity() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
}

bool HostStatePool::valid(HostStateSlotHandle handle) const noexcept {
    return handle.index < slots_.size() && slots_[handle.index].occupied &&
           slots_[handle.index].generation == handle.generation;
}

std::byte* HostStatePool::slot_data(std::uint32_t index) const noexcept {
    return static_cast<std::byte*>(backing_->data()) +
           static_cast<std::size_t>(index) * layout_.image_bytes;
}

StateImageDevicePool::StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout)
    : linear_(backing, layout.linear),
      continuation_hidden_(layout.continuation_hidden.bind(backing)), host_layout_(layout.host) {
    if (continuation_hidden_.dtype != DType::BF16 || !continuation_hidden_.is_contiguous() ||
        continuation_hidden_.ne[0] != host_layout_.spec.hidden ||
        continuation_hidden_.ne[1] != linear_.slot_count()) {
        throw std::invalid_argument("StateImage continuation hidden layout is inconsistent");
    }
    if (layout.dflash_local.has_value() != host_layout_.spec.dflash_local.has_value()) {
        throw std::invalid_argument("StateImage DFlash layout is inconsistent");
    }
    StateImageSpec device_spec{.linear = layout.linear.spec, .hidden = continuation_hidden_.ne[0]};
    if (layout.ple) {
        ple_.emplace(layout.ple->bind(backing));
        if (ple_->dtype != DType::BF16 || !ple_->is_contiguous() ||
            ple_->ne[1] != linear_.slot_count()) {
            throw std::invalid_argument("StateImage PLE layout is inconsistent");
        }
        device_spec.ple =
            PleStateSpec{.layers = static_cast<std::uint32_t>(ple_->ne[2]), .rows = ple_->ne[0]};
    }
    if (layout.dflash_local) {
        device_spec.dflash_local = DFlashLocalStateSpec{
            .layers   = static_cast<std::uint32_t>(layout.dflash_local->k.size()),
            .capacity = layout.dflash_local->capacity,
            .kv_heads = layout.dflash_local->num_kv_heads,
            .head_dim = layout.dflash_local->head_dim,
        };
    }
    if (!same_host_layout(host_layout_, plan_host_state_image(device_spec))) {
        throw std::invalid_argument("StateImage host layout does not match its device components");
    }
    if (layout.dflash_local) {
        if (layout.dflash_local->lane_capacity != linear_.slot_count()) {
            throw std::invalid_argument("StateImage components do not share one slot geometry");
        }
        dflash_local_.emplace(backing, *layout.dflash_local);
    }
}

StateImageDeviceSlotView StateImageDevicePool::slot_view(std::int32_t slot) const {
    StateImageDeviceSlotView view{
        .linear              = linear_.slot_view(slot),
        .continuation_hidden = continuation_hidden_slot(slot),
    };
    if (dflash_local_) { view.dflash_local = dflash_local_->slot_view(slot); }
    return view;
}

Tensor StateImageDevicePool::continuation_hidden_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage slot is out of range");
    return continuation_hidden_.slice(1, slot, 1).view({host_layout_.spec.hidden});
}

Tensor StateImageDevicePool::ple_states(std::uint32_t layer) const {
    if (!ple_ || layer >= static_cast<std::uint32_t>(ple_->ne[2])) {
        throw std::out_of_range("StateImage PLE layer is out of range");
    }
    return ple_->slice(2, static_cast<std::int32_t>(layer), 1).view({ple_->ne[0], ple_->ne[1]});
}

CyclicKVCache* StateImageDevicePool::dflash_local() noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

const CyclicKVCache* StateImageDevicePool::dflash_local() const noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

void StateImageDevicePool::attach_mirror(StateImageDevicePool& mirror, int device,
                                         cudaStream_t stream, cudaStream_t transfer_stream,
                                         CudaCompletionEvent* transfer_order) {
    // A mirror holds DFlash local state only for the split DFlash2 drafter (a rank's KV heads),
    // and then with the origin's slots and layers.
    const bool dflash_mismatch =
        mirror.dflash_local_ &&
        (!dflash_local_ || mirror.dflash_local_->layer_count() != dflash_local_->layer_count());
    if (&mirror == this || mirror.slot_count() != slot_count() ||
        mirror.linear_.layer_count() != linear_.layer_count() ||
        mirror.host_layout_.spec.hidden != host_layout_.spec.hidden || device < 0 ||
        stream == nullptr || transfer_stream == nullptr || transfer_order == nullptr ||
        dflash_mismatch || mirror.mirror_ != nullptr || host_layout_.peer_bytes != 0) {
        throw std::invalid_argument("StateImage mirror pool does not match the origin pool");
    }
    mirror_                 = &mirror;
    mirror_device_          = device;
    mirror_stream_          = stream;
    mirror_transfer_stream_ = transfer_stream;
    mirror_transfer_order_  = transfer_order;
    // Both images are finished to kStateImageAlignment, so rank 1's keeps its own alignment.
    host_layout_.peer_offset = host_layout_.image_bytes;
    host_layout_.peer_bytes  = mirror.host_layout_.image_bytes;
    host_layout_.image_bytes = checked_add(host_layout_.peer_offset, host_layout_.peer_bytes,
                                           "StateImage paired host image overflow");
}

void StateImageDevicePool::zero_slot_local(std::int32_t slot, cudaStream_t stream) {
    linear_.zero_slot(slot, stream);
    const Tensor hidden = continuation_hidden_slot(slot);
    CUDA_CHECK(cudaMemsetAsync(hidden.data, 0, hidden.bytes(), stream));
    for (std::uint32_t layer = 0; layer < ple_layers(); ++layer) {
        const Tensor states = ple_states(layer).slice(1, slot, 1);
        CUDA_CHECK(cudaMemsetAsync(states.data, 0, states.bytes(), stream));
    }
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, slot, 1);
            const Tensor v                    = view.v.slice(3, slot, 1);
            CUDA_CHECK(cudaMemsetAsync(k.data, 0, k.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(v.data, 0, v.bytes(), stream));
        }
    }
}

void StateImageDevicePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    validate_slot(slot, slot_count(), "StateImage zero slot is out of range");
    zero_slot_local(slot, stream);
    if (mirror_ != nullptr) {
        // Every component the mirror holds: GDN state, the hidden column, the PLE convolution
        // history (Qwen4Exp at tp 2) and the split drafter's DFlash rings.
        const ScopedCurrentDevice scope(mirror_device_);
        mirror_->zero_slot_local(slot, mirror_stream_);
    }
}

void StateImageDevicePool::zero_all_local(cudaStream_t stream) {
    linear_.zero_all(stream);
    CUDA_CHECK(cudaMemsetAsync(continuation_hidden_.data, 0, continuation_hidden_.bytes(), stream));
    if (ple_) { CUDA_CHECK(cudaMemsetAsync(ple_->data, 0, ple_->bytes(), stream)); }
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), stream));
        }
    }
}

void StateImageDevicePool::zero_all(cudaStream_t stream) {
    zero_all_local(stream);
    if (mirror_ != nullptr) {
        const ScopedCurrentDevice scope(mirror_device_);
        mirror_->zero_all_local(mirror_stream_);
    }
}

void StateImageDevicePool::copy_slot_local(std::int32_t source, std::int32_t destination,
                                           cudaStream_t stream) {
    linear_.copy_slot(source, destination, stream);
    const Tensor source_hidden      = continuation_hidden_slot(source);
    const Tensor destination_hidden = continuation_hidden_slot(destination);
    CUDA_CHECK(::ninfer::memcpy_device_async(destination_hidden.data, source_hidden.data,
                                             destination_hidden.bytes(), stream));
    for (std::uint32_t layer = 0; layer < ple_layers(); ++layer) {
        const Tensor states = ple_states(layer);
        const Tensor from   = states.slice(1, source, 1);
        const Tensor to     = states.slice(1, destination, 1);
        CUDA_CHECK(::ninfer::memcpy_device_async(to.data, from.data, to.bytes(), stream));
    }
}

void StateImageDevicePool::copy_slot(std::int32_t source, std::int32_t destination,
                                     cudaStream_t stream) {
    validate_slot(source, slot_count(), "StateImage copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage copy destination is out of range");
    if (source == destination) { return; }
    copy_slot_local(source, destination, stream);
    if (dflash_local_) {
        dflash_local_->copy_slot_from(*dflash_local_, source, destination, stream);
    }
    if (mirror_ != nullptr) {
        const ScopedCurrentDevice scope(mirror_device_);
        mirror_->copy_slot_local(source, destination, mirror_stream_);
        if (mirror_->dflash_local_) {
            mirror_->dflash_local_->copy_slot_from(*mirror_->dflash_local_, source, destination,
                                                   mirror_stream_);
        }
    }
}

void StateImageDevicePool::copy_dflash_local(std::int32_t source, std::int32_t destination,
                                             cudaStream_t stream) {
    validate_slot(source, slot_count(), "StateImage DFlash copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage DFlash copy destination is out of range");
    if (!dflash_local_) { throw std::logic_error("StateImage has no DFlash local component"); }
    if (source != destination) {
        dflash_local_->copy_slot_from(*dflash_local_, source, destination, stream);
        if (mirror_ != nullptr && mirror_->dflash_local_) {
            const ScopedCurrentDevice scope(mirror_device_);
            mirror_->dflash_local_->copy_slot_from(*mirror_->dflash_local_, source, destination,
                                                   mirror_stream_);
        }
    }
}

void StateImageDevicePool::validate_host_layout(const StateImageHostLayout* layout,
                                                const std::byte* data) const {
    if (layout == nullptr || data == nullptr || !same_host_layout(*layout, host_layout_)) {
        throw std::invalid_argument("Host and device StateImage layouts do not match");
    }
}

void StateImageDevicePool::copy_to_host_local(std::int32_t source, std::byte* destination,
                                              cudaStream_t stream) const {
    for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
        const Tensor conv = linear_.conv_slot(layer, source);
        CUDA_CHECK(cudaMemcpyAsync(
            byte_offset(destination, host_layout_.linear_conv.offset +
                                         layer * host_layout_.linear_conv_layer_bytes),
            conv.data, conv.bytes(), cudaMemcpyDeviceToHost, stream));
        const Tensor recurrent = linear_.recurrent_slot(layer, source);
        CUDA_CHECK(cudaMemcpyAsync(
            byte_offset(destination, host_layout_.linear_recurrent.offset +
                                         layer * host_layout_.linear_recurrent_layer_bytes),
            recurrent.data, recurrent.bytes(), cudaMemcpyDeviceToHost, stream));
    }
    const Tensor hidden = continuation_hidden_slot(source);
    CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination, host_layout_.continuation_hidden.offset),
                               hidden.data, hidden.bytes(), cudaMemcpyDeviceToHost, stream));
    for (std::uint32_t layer = 0; layer < ple_layers(); ++layer) {
        const Tensor states = ple_states(layer).slice(1, source, 1);
        CUDA_CHECK(
            cudaMemcpyAsync(byte_offset(destination, host_layout_.ple->offset +
                                                         layer * host_layout_.ple_layer_bytes),
                            states.data, states.bytes(), cudaMemcpyDeviceToHost, stream));
    }
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, source, 1);
            const Tensor v                    = view.v.slice(3, source, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination, host_layout_.dflash_local_k->offset +
                                             layer * host_layout_.dflash_local_layer_bytes),
                k.data, k.bytes(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination, host_layout_.dflash_local_v->offset +
                                             layer * host_layout_.dflash_local_layer_bytes),
                v.data, v.bytes(), cudaMemcpyDeviceToHost, stream));
        }
    }
}

void StateImageDevicePool::copy_to_host(std::int32_t source, HostStateImageView destination,
                                        cudaStream_t stream) const {
    validate_slot(source, slot_count(), "StateImage copy-to-host source is out of range");
    validate_host_layout(destination.layout, destination.data);
    copy_to_host_local(source, destination.data, stream);
    if (mirror_ != nullptr) {
        const ScopedCurrentDevice scope(mirror_device_);
        mirror_transfer_order_->record(mirror_stream_);
        mirror_transfer_order_->wait(mirror_transfer_stream_);
        mirror_->copy_to_host_local(source, destination.data + host_layout_.peer_offset,
                                    mirror_transfer_stream_);
    }
}

void StateImageDevicePool::copy_from_host_local(const std::byte* source, std::int32_t destination,
                                                cudaStream_t stream) {
    for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
        const Tensor conv = linear_.conv_slot(layer, destination);
        CUDA_CHECK(
            cudaMemcpyAsync(conv.data,
                            byte_offset(source, host_layout_.linear_conv.offset +
                                                    layer * host_layout_.linear_conv_layer_bytes),
                            conv.bytes(), cudaMemcpyHostToDevice, stream));
        const Tensor recurrent = linear_.recurrent_slot(layer, destination);
        CUDA_CHECK(cudaMemcpyAsync(
            recurrent.data,
            byte_offset(source, host_layout_.linear_recurrent.offset +
                                    layer * host_layout_.linear_recurrent_layer_bytes),
            recurrent.bytes(), cudaMemcpyHostToDevice, stream));
    }
    const Tensor hidden = continuation_hidden_slot(destination);
    CUDA_CHECK(cudaMemcpyAsync(hidden.data,
                               byte_offset(source, host_layout_.continuation_hidden.offset),
                               hidden.bytes(), cudaMemcpyHostToDevice, stream));
    for (std::uint32_t layer = 0; layer < ple_layers(); ++layer) {
        const Tensor states = ple_states(layer).slice(1, destination, 1);
        CUDA_CHECK(cudaMemcpyAsync(
            states.data,
            byte_offset(source, host_layout_.ple->offset + layer * host_layout_.ple_layer_bytes),
            states.bytes(), cudaMemcpyHostToDevice, stream));
    }
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, destination, 1);
            const Tensor v                    = view.v.slice(3, destination, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                k.data,
                byte_offset(source, host_layout_.dflash_local_k->offset +
                                        layer * host_layout_.dflash_local_layer_bytes),
                k.bytes(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                v.data,
                byte_offset(source, host_layout_.dflash_local_v->offset +
                                        layer * host_layout_.dflash_local_layer_bytes),
                v.bytes(), cudaMemcpyHostToDevice, stream));
        }
    }
}

void StateImageDevicePool::copy_from_host(HostStateImageConstView source, std::int32_t destination,
                                          cudaStream_t stream) {
    validate_slot(destination, slot_count(),
                  "StateImage copy-from-host destination is out of range");
    validate_host_layout(source.layout, source.data);
    copy_from_host_local(source.data, destination, stream);
    if (mirror_ != nullptr) {
        const ScopedCurrentDevice scope(mirror_device_);
        mirror_transfer_order_->record(mirror_stream_);
        mirror_transfer_order_->wait(mirror_transfer_stream_);
        mirror_->copy_from_host_local(source.data + host_layout_.peer_offset, destination,
                                      mirror_transfer_stream_);
    }
}

} // namespace ninfer::models::qwen3_5
