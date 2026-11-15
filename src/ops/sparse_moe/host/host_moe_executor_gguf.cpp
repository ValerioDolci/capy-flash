// GGUF layers of the host MoE executor (HostExpertLayer::format == Gguf): the decode-once kernels
// of host_gguf.h (gguf_kernels_: AVX-512 or AVX2, the same bits) in three phases over the pool,
// sharing run()'s CSR of host experts.
#include "ops/sparse_moe/host/host_moe_executor.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace ninfer::ops::host {
namespace {

constexpr std::int32_t kGgufRows  = 32; // rows per item: one 32-value block of h, 32 outputs
constexpr std::int32_t kGgufChunk = 8;  // rows decoded per decode_rows call (stay in L1)

std::size_t align64(std::size_t v) { return (v + 63) / 64 * 64; }

std::int64_t steady_ns_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace

void HostMoeExecutor::AlignedFree::operator()(std::uint8_t* p) const noexcept { std::free(p); }

void HostMoeExecutor::gguf_call() {
    const HostExpertLayer& layer = *layer_;
    const auto& g                = layer.gguf;
    if (!g.gate_up_rows || !g.down_rows ||
        g.gate_up_row_bytes != gguf::weight_row_bytes(g.gate_up, hidden_) ||
        g.down_row_bytes != gguf::weight_row_bytes(g.down, intermediate_) ||
        intermediate_ % kGgufRows || hidden_ % kGgufRows) {
        throw std::invalid_argument("host MoE executor: GGUF layer geometry mismatch");
    }
    const std::size_t x_bytes = std::max(gguf::packed_act_bytes(gguf::ActFormat::Q8_K, hidden_),
                                         gguf::packed_act_bytes(gguf::ActFormat::Q8_0, hidden_));
    const std::size_t h_bytes = gguf::packed_act_bytes(gguf::ActFormat::Q8_0, intermediate_);
    if (!gguf_x_) {
        const auto alloc = [](std::size_t n) {
            auto* p = static_cast<std::uint8_t*>(std::aligned_alloc(64, align64(n)));
            if (!p) throw std::bad_alloc();
            return AlignedBytes(p);
        };
        gguf_x_bytes_   = align64(x_bytes);
        gguf_h_bytes_   = align64(h_bytes);
        gguf_x_         = alloc(gguf_x_bytes_ * static_cast<std::size_t>(max_tokens_));
        gguf_h_         = alloc(gguf_h_bytes_ * static_cast<std::size_t>(max_tokens_) *
                                static_cast<std::size_t>(max_top_k_));
        std::size_t dec = 0;
        for (const auto f :
             {gguf::WeightFormat::IQ2_S, gguf::WeightFormat::IQ3_XXS, gguf::WeightFormat::IQ3_S,
              gguf::WeightFormat::IQ4_XS, gguf::WeightFormat::IQ4_NL}) {
            dec = std::max(dec, gguf::decoded_row_bytes(f, hidden_));
        }
        for (const auto f : {gguf::WeightFormat::IQ4_NL, gguf::WeightFormat::Q2_0}) {
            dec = std::max(dec, gguf::decoded_row_bytes(f, intermediate_));
        }
        gguf_members_.resize(static_cast<std::size_t>(pool_.size()));
        const auto T = static_cast<std::size_t>(max_tokens_);
        for (auto& m : gguf_members_) {
            m.dec = alloc(dec * kGgufChunk);
            m.gate.resize(kGgufRows * T);
            m.up.resize(kGgufRows * T);
            m.part.resize(kGgufChunk * T);
            m.acts.resize(T);
        }
    }
    // Exclusive tier: a missing expert must have a host copy; a read without one is garbage.
    if (g.host_slots != nullptr) {
        for (std::int32_t u = 0; u < unique_count_; ++u) {
            const std::int32_t e = unique_[static_cast<std::size_t>(u)];
            if (g.host_slots[e] < 0) {
                throw std::logic_error("host MoE executor: expert " + std::to_string(e) +
                                       " has neither a device slot nor a host slot");
            }
        }
    }
    for (auto& n : gguf_next_) n.store(0, std::memory_order_relaxed);
    pool_.run(&HostMoeExecutor::gguf_body, this);
}

void HostMoeExecutor::gguf_body(void* self, std::int32_t member) {
    static_cast<HostMoeExecutor*>(self)->gguf_member(member);
}

void HostMoeExecutor::gguf_member(std::int32_t member) {
    const HostExpertLayer& layer   = *layer_;
    const auto& g                  = layer.gguf;
    const HostMoeJob& job          = *job_;
    const gguf::HostGgufKernels& K = gguf_kernels_;
    const std::int32_t H = hidden_, I = intermediate_, T = job.tokens;
    const gguf::ActFormat xact = gguf::format_info(g.gate_up).act;
    GgufMember& me             = gguf_members_[static_cast<std::size_t>(member)];
    stamp(member, 0);
    // Tokens: x of every token with a host expert, quantized once for the call.
    const auto used = static_cast<std::int32_t>(used_tokens_.size());
    for (std::int32_t i; (i = gguf_next_[0].fetch_add(1, std::memory_order_relaxed)) < used;) {
        const std::int32_t t = used_tokens_[static_cast<std::size_t>(i)];
        K.quantize_pack_bf16_q8f32(xact, job.x + static_cast<std::size_t>(t) * H, H,
                                   gguf_x_.get() + gguf_x_bytes_ * static_cast<std::size_t>(t));
    }
    stamp(member, 1);
    pool_.barrier(1);
    stamp(member, 2);
    if (member == 0) phase_ns_[0] = steady_ns_now();
    // Gate/up: items (expert u, 32 intermediate rows s); h of each of its pairs.
    const std::int32_t U = unique_count_, slices = I / kGgufRows;
    const std::int32_t items = U * slices;
    for (std::int32_t it; (it = gguf_next_[1].fetch_add(1, std::memory_order_relaxed)) < items;) {
        const std::int32_t u = it / slices, s = it % slices;
        const std::int32_t e  = unique_[static_cast<std::size_t>(u)];
        const std::int32_t p0 = pair_begin_[static_cast<std::size_t>(u)];
        const std::int32_t n  = pair_begin_[static_cast<std::size_t>(u) + 1] - p0;
        for (std::int32_t i = 0; i < n; ++i) {
            me.acts[static_cast<std::size_t>(i)] =
                gguf_x_.get() + gguf_x_bytes_ * static_cast<std::size_t>(
                                                    pair_token_[static_cast<std::size_t>(p0 + i)]);
        }
        const std::int32_t row_e = g.host_slots != nullptr ? g.host_slots[e] : e;
        const std::uint8_t* gate = g.gate_up_rows + (static_cast<std::size_t>(row_e) * 2 * I +
                                                     static_cast<std::size_t>(s) * kGgufRows) *
                                                        g.gate_up_row_bytes;
        const std::uint8_t* up = gate + static_cast<std::size_t>(I) * g.gate_up_row_bytes;
        for (std::int32_t c = 0; c < kGgufRows / kGgufChunk; ++c) {
            const std::size_t off = static_cast<std::size_t>(c) * kGgufChunk * g.gate_up_row_bytes;
            K.decode_rows(g.gate_up, gate + off, g.gate_up_row_bytes, kGgufChunk, H, me.dec.get());
            K.gemm_decoded(g.gate_up, me.dec.get(), kGgufChunk, H, me.acts.data(), n,
                           me.gate.data() + static_cast<std::size_t>(c) * kGgufChunk * n,
                           static_cast<std::size_t>(n));
            K.decode_rows(g.gate_up, up + off, g.gate_up_row_bytes, kGgufChunk, H, me.dec.get());
            K.gemm_decoded(g.gate_up, me.dec.get(), kGgufChunk, H, me.acts.data(), n,
                           me.up.data() + static_cast<std::size_t>(c) * kGgufChunk * n,
                           static_cast<std::size_t>(n));
        }
        for (std::int32_t i = 0; i < n; ++i) {
            alignas(64) float gv[kGgufRows], uv[kGgufRows], hv[kGgufRows];
            for (std::int32_t r = 0; r < kGgufRows; ++r) {
                gv[r] = me.gate[static_cast<std::size_t>(r) * n + i];
                uv[r] = me.up[static_cast<std::size_t>(r) * n + i];
            }
            K.silu_mul(gv, uv, hv, kGgufRows);
            K.quantize_pack_q8_0_half_q8f32(
                hv, s / 2, s % 2, gguf_h_.get() + gguf_h_bytes_ * static_cast<std::size_t>(p0 + i));
        }
    }
    stamp(member, 3);
    pool_.barrier(2);
    stamp(member, 4);
    if (member == 0) phase_ns_[1] = steady_ns_now();
    // Down: items of 32 output rows; every host expert in ascending id order, then the next item.
    const std::int32_t outs = H / kGgufRows;
    for (std::int32_t it; (it = gguf_next_[2].fetch_add(1, std::memory_order_relaxed)) < outs;) {
        const std::int32_t r0 = it * kGgufRows;
        for (std::int32_t t = 0; t < T; ++t) {
            std::memset(job.out + static_cast<std::size_t>(t) * H + r0, 0,
                        sizeof(float) * kGgufRows);
        }
        for (std::int32_t u = 0; u < U; ++u) {
            const std::int32_t e  = unique_[static_cast<std::size_t>(u)];
            const std::int32_t p0 = pair_begin_[static_cast<std::size_t>(u)];
            const std::int32_t n  = pair_begin_[static_cast<std::size_t>(u) + 1] - p0;
            for (std::int32_t i = 0; i < n; ++i) {
                me.acts[static_cast<std::size_t>(i)] =
                    gguf_h_.get() + gguf_h_bytes_ * static_cast<std::size_t>(p0 + i);
            }
            const std::int32_t row_e = g.host_slots != nullptr ? g.host_slots[e] : e;
            const std::uint8_t* rows =
                g.down_rows + (static_cast<std::size_t>(row_e) * H + static_cast<std::size_t>(r0)) *
                                  g.down_row_bytes;
            for (std::int32_t c = 0; c < kGgufRows / kGgufChunk; ++c) {
                K.decode_rows(g.down,
                              rows + static_cast<std::size_t>(c) * kGgufChunk * g.down_row_bytes,
                              g.down_row_bytes, kGgufChunk, I, me.dec.get());
                K.gemm_decoded(g.down, me.dec.get(), kGgufChunk, I, me.acts.data(), n,
                               me.part.data(), static_cast<std::size_t>(n));
                for (std::int32_t r = 0; r < kGgufChunk; ++r) {
                    for (std::int32_t i = 0; i < n; ++i) {
                        const std::int32_t t = pair_token_[static_cast<std::size_t>(p0 + i)];
                        job.out[static_cast<std::size_t>(t) * H + r0 + c * kGgufChunk + r] +=
                            pair_alpha_[static_cast<std::size_t>(p0 + i)] *
                            me.part[static_cast<std::size_t>(r) * n + i];
                    }
                }
            }
        }
    }
    stamp(member, 5);
    if (member == 0) phase_ns_[2] = steady_ns_now();
}

} // namespace ninfer::ops::host
