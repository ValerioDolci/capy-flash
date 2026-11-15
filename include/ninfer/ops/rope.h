#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>
#include <vector>

namespace ninfer::ops {

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * With a YaRN table installed on the device (rope_install_text_yarn below) for this rotary_dim and
 * theta, both Text modes take phi(i,t) = positions[axis(i),t] * inv_freq[i] (the FP32 product)
 * from the table, with the same axes, and multiply cos(phi) and sin(phi) by its attention factor.
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. q and k must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace; its only persistent state is the YaRN table.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);

/**
 * YaRN for Text RoPE, as Hugging Face rope_type "yarn": with factor > 1, computes the inverse
 * frequencies and the attention factor of rotary_dim 64 (ops/kernel/yarn.h) and puts them in the
 * constant memory of each of `devices`. From then on every Text RoPE of this rotary_dim and theta
 * takes them: rope above, rmsnorm_rope_text_shard and the QSA indexer (sparse_attention.h); such
 * a launch on a device not in `devices` throws. factor <= 1 removes the table: every route is the
 * plain RoPE again, bit for bit. One table per process, set before the launches that use it.
 */
void rope_install_text_yarn(const std::vector<int>& devices, float factor,
                            std::uint32_t original_max_position, int rotary_dim, float theta);

} // namespace ninfer::ops
