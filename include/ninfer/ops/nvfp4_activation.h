#pragma once

// Sequence-parallel TP2 prefill (models/qwen3_5/execution/qwen4_forward/plain_sp.h): hands the
// NVFP4 A4 activation plane (codes + scales) of one column window from the rank that owns those
// tokens to the peer, which holds no BF16 input for them.
//
// Every NVFP4 A4 projection quantizes its input through one function. While a scope is open for
// an input pointer and token count, that function instead
//   Capture: quantizes into the scope's plane, in the layout the projection chose, and records the
//            layout, K and the bytes written; the caller then sends those bytes to the peer;
//   Consume: quantizes nothing and hands the projection the scope's plane, which the same
//            projection must have captured over the same token count (layout, K and sizes must
//            match; a mismatch throws instead of reading a plane in the wrong layout).
// Scopes are per host thread, nest LIFO and are keyed by the input's device pointer (unique
// across devices under UVA). used() tells whether a projection took the scope; in Consume mode an
// unused scope means a projection quantized an input that holds no data, so the caller must
// treat it as an error.

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct Nvfp4ActivationPlane {
    void* codes                = nullptr;
    std::size_t code_capacity  = 0;
    void* scales               = nullptr;
    std::size_t scale_capacity = 0;
    // Written by a capture, checked by a consume.
    int layout              = -1;
    std::int32_t k          = 0;
    std::size_t code_bytes  = 0;
    std::size_t scale_bytes = 0;
};

// Capacities that hold `tokens` columns of a K-row input in any layout.
std::size_t nvfp4_activation_code_capacity(std::int32_t tokens, std::int32_t k);
std::size_t nvfp4_activation_scale_capacity(std::int32_t tokens, std::int32_t k);

class ScopedNvfp4Activation {
public:
    enum class Mode { Capture, Consume };

    ScopedNvfp4Activation(Mode mode, const void* input, std::int32_t tokens,
                          Nvfp4ActivationPlane& plane);
    ~ScopedNvfp4Activation();
    ScopedNvfp4Activation(const ScopedNvfp4Activation&)            = delete;
    ScopedNvfp4Activation& operator=(const ScopedNvfp4Activation&) = delete;

    [[nodiscard]] bool used() const noexcept { return used_; }

    [[nodiscard]] Mode mode() const noexcept { return mode_; }

    [[nodiscard]] Nvfp4ActivationPlane& plane() const noexcept { return *plane_; }

    void mark_used() noexcept { used_ = true; }

    // The innermost open scope of this thread for `input` over `tokens` columns, or null.
    static ScopedNvfp4Activation* find(const void* input, std::int32_t tokens) noexcept;

private:
    Mode mode_;
    const void* input_;
    std::int32_t tokens_;
    Nvfp4ActivationPlane* plane_;
    bool used_                       = false;
    ScopedNvfp4Activation* previous_ = nullptr;
};

} // namespace ninfer::ops
