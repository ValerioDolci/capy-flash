#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Sigmoid-scored MoE router with a selection-only bias: FP32 logits from a BF16 input, sigmoid
// scores, top-k on score + bias, weights = the selected scores renormalized and scaled.

// Largest expert count and top-k accepted by moe_sigmoid_route.
inline constexpr std::int32_t kMoeSigmoidRouteMaxExperts = 4096;
inline constexpr std::int32_t kMoeSigmoidRouteMaxTopK    = 64;

/**
 * Op: sigmoid router, top-k on biased scores, renormalized scaled weights
 *
 * Math / indexing, per column t (u = x[:,t]):
 *   l_e        = sum_h router[h,e] * u_h                  for e < E
 *   s_e        = sigmoid(l_e) = 1 / (1 + exp(-l_e))
 *   key_e      = s_e + bias[e]                           (the bias only orders the selection)
 *   ids[:,t]   = the k experts with the largest key_e, in decreasing key order; an exact tie of
 *                keys goes to the lower expert id
 *   weights[j,t] = s_{ids[j,t]} / (sum_{i<k} s_{ids[i,t]} + 1e-20) * route_scale
 *
 *   Columns past their lane's valid count (`valid_columns`, I32 [B] with B dividing T: column
 *   t = b*W + w, W = T/B, is valid iff w < valid_columns[b]) route nothing: ids[:,t] = -1 and
 *   weights[:,t] = 0, which moe_bank_capped_swiglu skips. Without `valid_columns` every column
 *   routes.
 *
 * Logical shapes: x BF16 [H,T]; router FP32 [H,E] (ne order: E rows of H, row e = expert e, the
 *   nn.Linear [E,H] weight as stored); bias FP32 [E]; ids I32 [k,T] (global expert ids in
 *   [0,E)); weights FP32 [k,T]; valid_columns null or I32 [B] (device).
 * Supported domain: H >= 1, T >= 1, 1 <= E <= kMoeSigmoidRouteMaxExperts,
 *   1 <= k <= min(E, kMoeSigmoidRouteMaxTopK); finite route_scale; contiguous tensors; finite
 *   inputs.
 * Numeric: FP32 dot products over the represented BF16 input and FP32 router (no rounding of the
 *   logits to BF16); FP32 sigmoid, key, sum and weights. The k ids of a column are distinct. A
 *   column's outputs depend only on that column's input, the router and the bias (not on T, on
 *   the column's position or on the device instance): two calls with the same values on two
 *   devices of the same architecture give the same bits, which is what keeps a router
 *   replicated across tensor-parallel ranks consistent. The oracle evaluates l, s and the keys
 *   in FP64; the selection may differ from the FP64 selection only between experts whose FP64
 *   keys are within the FP32 evaluation bound of each other, and exact FP32 key ties are
 *   resolved by the lower id.
 * Effects: writes ids and weights only; outputs must not overlap the inputs.
 * Workspace: none. Execution: stream ordered, no host synchronization, CUDA Graph capturable.
 */
void moe_sigmoid_route(const Tensor& x, const Tensor& router, const Tensor& bias,
                       std::int32_t top_k, float route_scale, Tensor& ids, Tensor& weights,
                       cudaStream_t stream, const Tensor* valid_columns = nullptr);

} // namespace ninfer::ops
