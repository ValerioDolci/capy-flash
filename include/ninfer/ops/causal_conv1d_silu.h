#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Depthwise causal width-4 convolution followed by SiLU. Let u[c,-3..-1] be the three values in
 * the input state and u[c,t]=x[c,t] for t>=0. Then
 *
 *   ideal[c,t] = SiLU(sum_{j=0..3} weight[c,j] * u[c,t-3+j]).
 *
 * `x` is contiguous BF16 [C,T], `weight` is contiguous BF16 [C,4], and a state is contiguous BF16
 * [C,3] ordered oldest to newest. The oracle evaluates `ideal` naively in FP64 from
 * the represented inputs. The BF16 output is promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Kernel
 * accumulator and staging precision are implementation choices. Input, weight, output, and state
 * storage do not overlap except for the explicitly allowed exact alias between state input and
 * state output. No caller workspace is used. T may be any positive value.
 */

// Split-output form: the rows of `ideal` are written to three destinations, partitioned in row
// order as out0, out1, out2. Every destination is contiguous BF16 with the column count of x, and
// every operand is four-byte aligned. The supported row profiles are (2048, 2048, 6144) over
// C = 10240 and its two-device half (1024, 1024, 3072) over C = 5120; any other profile is
// rejected. conv_state_in and conv_state_out may be disjoint or exactly the same storage;
// conv_state_out receives the trailing width-3 window of concat(conv_state_in,x). No destination
// may overlap another, x, weight, or either state.
void causal_conv1d_silu_split(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                              Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                              cudaStream_t stream);

} // namespace ninfer::ops
