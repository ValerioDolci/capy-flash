#pragma once

// L2 preload of the dense projection weights at decode and narrow verification widths. Those
// Linears stream their BF16 weights from DRAM close to its bandwidth, while the kernels between
// them (exchange waits, convolutions, block residuals, message gates, router, attention over the
// window) leave DRAM mostly idle. Each rank forks its prefetch lane (prefetch_lane.h) and preloads
// there (ops::l2_preload) the weights the next projection reads: after the expert bank, the next
// visit's input projections, while the MoE exchange, the residual round and the next visit's block
// residual run; after the input projections, the output projection, while the attention runs. At
// one column the leading kEarlyPreloadBytes of the next visit's input projections follow the
// output projection on the lane, which so keeps streaming through the attention, its exchange,
// the residual mix, the FFN gate and the router (the expert bank, which streams its own weights,
// starts after them). A lane joins its rank's stream
// once, at the end of the call, so no kernel waits for a preload. A preload writes nothing the
// forward reads: no result bit changes.
//
// Member definitions of Forward (forward.h), included only by forward.cpp.

#include "models/inkling2/execution/forward.h"

#include "core/device.h"
#include "models/inkling2/execution/prefetch_lane.h"
#include "ninfer/ops/l2_preload.h"
#include "ninfer/ops/peer_mailbox.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

namespace ninfer::models::inkling2::execution {

// Up to this many columns a call preloads. Wider, the projections read their weights at well
// below DRAM bandwidth (the GEMV at 8 columns is as fast from DRAM as from L2) and a preload only
// competes with the kernels beside it.
inline constexpr std::int32_t kPrefetchMaxTokens = 4;

// Blocks per preload (ops::l2_preload: each keeps 32 KiB of loads in flight), below what full DRAM
// bandwidth needs: the kernels running beside a preload wait on DRAM latency, which a saturated
// DRAM queue stretches. The input projections (15-20 MB) take the wider preload, the output
// projection (7 MB, beside the attention) the narrower one.
inline constexpr int kInputPreloadBlocks  = 16;
inline constexpr int kOutputPreloadBlocks = 8;

// At one column: the leading bytes of the next visit's input projections preloaded early, and
// that preload's blocks. Wider, the kernels beside it (and the bank after it) are longer and the
// early preload costs more than it saves.
inline constexpr std::size_t kEarlyPreloadBytes = std::size_t{6} << 20;
inline constexpr int kEarlyPreloadBlocks        = 8;

// The message gates of the later passes read 1.8 MB of cold weights each, at the head of the visit
// (q, k, v) and after the residual mix (FFN). Ahead of the next visit's projections (early
// preload at one column, input preload wider): its q, k and v gates (5.3 MB, needed first); with
// the output preload, after the output projection: this visit's FFN gate.
inline constexpr int kGatePreloadBlocks = 16;

template <class Arena>
void Forward<Arena>::prefetch(int r, std::uint32_t index, Prefetch what) {
    if constexpr (kExecute) {
        const PrefetchLane* lane = r_[at(r)].prefetch;
        if (lane == nullptr || T_ > kPrefetchMaxTokens || index >= config_.visits.size() ||
            (what == Prefetch::Early && T_ != 1)) {
            return;
        }
        const Visit& v = config_.visits[index];
        const auto& p  = layer(r, v.layer).attention;
        std::array<ops::L2PreloadRange, 3> ranges{};
        std::size_t count = 0;
        const auto add    = [&](const LinearParameters& projection) {
            const Weight& w = projection.weight;
            if (w.qtype == QType::BF16 && w.qdata != nullptr) {
                ranges[count++] = {w.qdata, std::size_t(w.n) * std::size_t(w.k) * 2};
            }
        };
        const bool global   = v.kind == LayerKind::Global;
        const auto add_gate = [&](std::array<ops::L2PreloadRange, 3>& into, std::size_t& n,
                                  const MessageGateParameters& g) {
            into[n++] = {g.down.data, g.down.bytes()};
            // The matrices the batched gates read: the BF16 copies.
            into[n++] = {g.forget_bf16.data, g.forget_bf16.bytes()};
            into[n++] = {g.gain_bf16.data, g.gain_bf16.bytes()};
        };
        bool forked     = false;
        const auto fork = [&] {
            if (!forked) {
                CUDA_CHECK(cudaEventRecord(lane->fork(), s(r)));
                CUDA_CHECK(cudaStreamWaitEvent(lane->stream(), lane->fork(), 0));
                forked         = true;
                forked_[at(r)] = true;
            }
        };
        // The q, k, v gates: at one column with the early preload (the input preload starts
        // too close to them), wider with the input preload.
        // At one column as bulk prefetch hints, which need SMs only to be issued (the early
        // preload's blocks wait for their loads while the bank fills the SMs).
        if (v.messages && what == (T_ == 1 ? Prefetch::Early : Prefetch::Inputs)) {
            const auto& gates = layer(r, v.layer).core->gates;
            std::array<ops::PeerL2Range, 9> hints{};
            for (int j = 0; j < 3; ++j) {
                std::array<ops::L2PreloadRange, 3> gate{};
                std::size_t n = 0;
                add_gate(gate, n, gates[at(j)]);
                fork();
                if (T_ == 1) {
                    for (int k = 0; k < 3; ++k) {
                        hints[at(3 * j + k)] = {gate[at(k)].data, gate[at(k)].bytes};
                    }
                    continue;
                }
                ops::l2_preload(std::span<const ops::L2PreloadRange>(gate.data(), n),
                                kGatePreloadBlocks, lane->stream());
            }
            if (T_ == 1) { ops::l2_prefetch(hints, lane->stream()); }
        }
        if (what == Prefetch::Output) {
            add(p.output);
        } else if (!global && !v.messages) {
            add(*p.all);
        } else {
            add(p.queries);
            add(global ? *p.keys : p.key);
            add(global ? *p.values : p.value);
        }
        if (what != Prefetch::Output && T_ == 1) {
            // Early: the leading kEarlyPreloadBytes; Inputs: the rest.
            std::array<ops::L2PreloadRange, 3> part{};
            std::size_t kept = 0, left = kEarlyPreloadBytes;
            for (std::size_t i = 0; i < count; ++i) {
                const auto* data       = static_cast<const std::byte*>(ranges[i].data);
                const std::size_t head = std::min(left, ranges[i].bytes);
                left -= head;
                if (what == Prefetch::Early && head > 0) {
                    part[kept++] = {data, head};
                } else if (what == Prefetch::Inputs && head < ranges[i].bytes) {
                    part[kept++] = {data + head, ranges[i].bytes - head};
                }
            }
            ranges = part;
            count  = kept;
        }
        if (count == 0) { return; }
        fork();
        const int blocks = what == Prefetch::Output  ? kOutputPreloadBlocks
                           : what == Prefetch::Early ? kEarlyPreloadBlocks
                                                     : kInputPreloadBlocks;
        ops::l2_preload(std::span<const ops::L2PreloadRange>(ranges.data(), count), blocks,
                        lane->stream());
        if (what == Prefetch::Output && v.messages) {
            std::array<ops::L2PreloadRange, 3> gate{};
            std::size_t n = 0;
            add_gate(gate, n, layer(r, v.layer).core->gates[3]);
            ops::l2_preload(std::span<const ops::L2PreloadRange>(gate.data(), n),
                            kOutputPreloadBlocks, lane->stream());
        }
        if (what == Prefetch::Output && T_ == 1) { prefetch(r, index + 1, Prefetch::Early); }
    } else {
        (void)r, (void)index, (void)what;
    }
}

template <class Arena>
void Forward<Arena>::join_prefetch() {
    each([&](int r) {
        if constexpr (kExecute) {
            if (!forked_[at(r)]) { return; }
            const PrefetchLane* lane = r_[at(r)].prefetch;
            CUDA_CHECK(cudaEventRecord(lane->join(), lane->stream()));
            CUDA_CHECK(cudaStreamWaitEvent(s(r), lane->join(), 0));
            forked_[at(r)] = false;
        } else {
            (void)r;
        }
    });
}

} // namespace ninfer::models::inkling2::execution
