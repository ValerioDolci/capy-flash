#pragma once
// INT8-KV decode/verify attention, warp-specialized pipeline (NINFER_ATTN_DEC_PIPE=1, opt-in).
//
// int8_kv_grouped_mma_kernel runs each 32-key tile in phases behind block barriers: the producer
// warps compute QK and the softmax while the other warps dequantize V into an FP16 staging tile,
// then every warp runs PV, and only then is the next tile loaded. Probes (5070 Ti, W=7, 64k keys,
// 116.7 us) show neither the loads (110.3 without them) nor the MMAs and the dequantization (106.2
// without QK, dequant and PV) bound it: the serial chain of each tile does, at 12 warps per SM.
//
// Here three producer warps (one per 16-row tile) score tile j+1 while twelve consumer warps run PV
// of tile j, through double-buffered probabilities handed over by named barriers (FULL: written,
// EMPTY: read); K and V each have two buffers, filled by the warps that read them. A consumer warp
// owns 16 rows x one 64-dimension group, so it needs one V scale per key: it loads raw INT8 V with
// ldmatrix.trans (two keys x two dimensions per register), converts it exactly to FP16 (1024 + u -
// 1152) and multiplies by the FP16 scale, which gives the dequantized value of the original kernel
// bit for bit (one rounding of an exact product). The B fragment of a register's even bytes serves
// the even dimensions, its odd bytes the odd ones: each n-tile covers 8 even or 8 odd dimensions of
// a 16-dimension block, and the epilogue stores them interleaved. Scores, probabilities and every
// PV product and sum are those of int8_kv_grouped_mma_kernel in the same order; the partition
// differs (one CTA per SM, half the splits), so the merged output may differ in the last bits.
#include "ops/softmax_attention/dense/causal_cache/int8/tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/pipe_helpers.cuh"

#include "ops/softmax_attention/dense/causal_cache/int8/schedule.cuh"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_softmax.cuh"

namespace ninfer::ops::detail {

namespace int8_kv_dec_pipe {
inline constexpr int kRowTiles      = 3;
inline constexpr int kProducerWarps = kRowTiles;
inline constexpr int kConsumerWarps = kRowTiles * 4;
inline constexpr int kWarps         = kProducerWarps + kConsumerWarps;
inline constexpr int kThreads       = kWarps * 32;
inline constexpr int kKeys          = 32;
inline constexpr int kBr            = kRowTiles * 16;
inline constexpr int kD             = 256;
inline constexpr int kGroups        = 4;
// Shared memory: Q codes, Q scales, 2 K and 2 V code buffers, their scales, 2 probability tiles, 2
// rescale rows.
inline constexpr int kQBytes      = kBr * kD;
inline constexpr int kQScaleBytes = kBr * kGroups * 4;
inline constexpr int kKvBytes     = kKeys * kD;
inline constexpr int kScaleBytes  = kKeys * kGroups * 2;
inline constexpr int kPBytes      = kBr * kKeys * 2;
inline constexpr int kSharedBytes =
    kQBytes + kQScaleBytes + 4 * kKvBytes + 4 * kScaleBytes + 2 * kPBytes + 2 * kBr * 4;
// Named barriers (0 is __syncthreads).
inline constexpr int kFull0     = 1;
inline constexpr int kEmpty0    = 3;
inline constexpr int kProducers = 5;
inline constexpr int kConsumers = 6;

static_assert(kD == int8_kv_pipe::kD);
// Barriers, the V tile layout and the INT8 -> FP16 conversion are shared with the ping-pong prefill
// kernel.
using int8_kv_pipe::arrive;
using int8_kv_pipe::ldmatrix_x4_trans_b16;
using int8_kv_pipe::sync;
using int8_kv_pipe::v_offset;
using int8_kv_pipe::v_pair;
} // namespace int8_kv_dec_pipe

template <class Geometry, class Schedule, bool MultiBatch, bool Masked, class CacheInput>
__launch_bounds__(int8_kv_dec_pipe::kThreads, 1) __global__ void int8_kv_grouped_pipe_mma_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos,
    typename Int8KvCacheView<CacheInput::writes_cache>::Code* cache_k_i8,
    typename Int8KvCacheView<CacheInput::writes_cache>::Code* cache_v_i8,
    typename Int8KvCacheView<CacheInput::writes_cache>::KeyScale* cache_k_scale,
    typename Int8KvCacheView<CacheInput::writes_cache>::ValueScale* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
    std::int32_t logical_capacity, CausalKvPartition partition, float scale, float* partial_acc,
    float* partial_m, float* partial_l) {
    using namespace int8_kv_dec_pipe;
    constexpr int TokenTile       = Schedule::kTokenTile;
    constexpr int RowCount        = TokenTile * Geometry::GroupSize;
    constexpr int Bc              = kKeys;
    constexpr int D               = kD;
    constexpr int DB16            = D / 2;
    constexpr int Groups          = kKVCacheInt8Groups;
    constexpr int GroupKc         = kKVCacheInt8Group / 32;
    constexpr int QKNt            = Bc / 8;
    constexpr int PVKs            = Bc / 16;
    constexpr int ProducerThreads = kProducerWarps * 32;
    constexpr int ConsumerThreads = kConsumerWarps * 32;
    constexpr int AllThreads      = kThreads;
    constexpr float Log2E         = kLog2E;
    constexpr unsigned FullMask   = 0xffffffffu;
    static_assert(Groups == kGroups && GroupKc == 2);
    static_assert((RowCount + 15) / 16 == kRowTiles, "the pipeline serves three 16-row tiles");
    static_assert(Bc * 2 == kPagedKVPageSize);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8    = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale_tmp   = reinterpret_cast<float*>(smem_raw + kQBytes);
    std::int8_t* k_buf   = reinterpret_cast<std::int8_t*>(smem_raw + kQBytes + kQScaleBytes); // [2]
    std::int8_t* v_buf   = k_buf + 2 * kKvBytes;                                              // [2]
    __half* k_scale_buf  = reinterpret_cast<__half*>(v_buf + 2 * kKvBytes);                   // [2]
    __half* v_scale_buf  = k_scale_buf + 2 * Bc * Groups;                                     // [2]
    __half* p_buf        = v_scale_buf + 2 * Bc * Groups;                                     // [2]
    float* alpha_buf     = reinterpret_cast<float*>(p_buf + 2 * kBr * Bc);                    // [2]
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);

    const int kv_head         = blockIdx.x;
    const int split           = static_cast<int>(blockIdx.z);
    const int batch           = MultiBatch ? static_cast<int>(blockIdx.y) : 0;
    const int split_count     = static_cast<int>(gridDim.z);
    const int tid             = static_cast<int>(threadIdx.x);
    const int warp            = tid >> 5;
    const int lane            = tid & 31;
    constexpr int tile_tokens = TokenTile;
    const int partial_width   = full_width;

    int valid_tokens = tile_tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch];
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, tile_tokens);
    }
    std::int64_t column_base = 0;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(256) * Geometry::QHeads * column_base;
    const int last_pos = pos[(MultiBatch ? batch * full_width : 0) + full_width - 1];
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(256) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(256) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * 256 * Geometry::QHeads * partial_width * split_count;
        partial_m +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
        partial_l +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
    }

    if (valid_tokens == 0) return;
    if (pos[0] < 0 || last_pos < 0 || last_pos >= logical_capacity) return;
    const int window = last_pos + 1;
    const auto owned = causal_split_run(partition, window, split);
    if (owned.begin >= window) {
        if (split < partition.bound(window)) {
            for (int row = tid; row < tile_tokens * Geometry::GroupSize; row += AllThreads) {
                int q_head = 0;
                int token  = 0;
                causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
                const auto index = causal_stat_index<Geometry>(q_head, token, split, partial_width);
                partial_m[index] = -CUDART_INF_F;
                partial_l[index] = 0.0f;
            }
        }
        return;
    }
    const int split_start = owned.begin;
    const int split_end   = owned.end;
    const int first_tile  = split_start;
    const int key_blocks  = div_up(split_end - first_tile, Bc);

    if constexpr (CacheInput::writes_cache) {
        // As int8_kv_grouped_mma_kernel: the split that holds the new positions encodes them first.
        float* k_h64_s = reinterpret_cast<float*>(k_buf);
        for (int pair = warp; pair < valid_tokens * Groups; pair += kWarps) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = causal_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = causal_new_index<Geometry>(kv_head, d1, token);
            float k_h64[2] = {__bfloat162float(input.k[src0]), __bfloat162float(input.k[src1])};
            hadamard_d64_fragment_inplace(k_h64, lane);
            k_h64_s[token * D + d0] = k_h64[0];
            k_h64_s[token * D + d1] = k_h64[1];
        }
        __syncthreads();
        for (int pair = warp; pair < valid_tokens * Groups; pair += kWarps) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = block_table[position >> kPagedKVPageShift];
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = causal_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = causal_new_index<Geometry>(kv_head, d1, token);
            float k_out[2];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int dh   = lane + half * 32;
                const float x0 = k_h64_s[token * D + dh];
                const float x1 = k_h64_s[token * D + kKVCacheInt8Group + dh];
                const float x2 = k_h64_s[token * D + 2 * kKVCacheInt8Group + dh];
                const float x3 = k_h64_s[token * D + 3 * kKVCacheInt8Group + dh];
                k_out[half]    = normalized_hadamard_d256_group_value_from_h64(x0, x1, x2, x3, grp);
            }
            const float kv0    = k_out[0];
            const float kv1    = k_out[1];
            const float vv0    = __bfloat162float(input.v[src0]);
            const float vv1    = __bfloat162float(input.v[src1]);
            float kamax        = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax        = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax              = warp_max(kamax, FullMask);
            vamax              = warp_max(vamax, FullMask);
            const auto k_quant = kv_cache_int8_quant_params(kamax);
            const auto v_quant = kv_cache_int8_quant_params(vamax);
            cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                page_offset)] =
                kv_cache_int8_quant_code(kv0, k_quant.inverse_scale);
            cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                page_offset)] =
                kv_cache_int8_quant_code(kv1, k_quant.inverse_scale);
            cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                page_offset)] =
                kv_cache_int8_quant_code(vv0, v_quant.inverse_scale);
            cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                page_offset)] =
                kv_cache_int8_quant_code(vv1, v_quant.inverse_scale);
            if (lane == 0) {
                const std::int64_t so = kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, grp, page_offset);
                cache_k_scale[so] = k_quant.scale;
                cache_v_scale[so] = v_quant.scale;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < kBr * D; i += AllThreads) { q_i8[i] = 0; }
    for (int i = tid; i < kBr * Groups; i += AllThreads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();
    for (int row = warp; row < RowCount; row += kWarps) {
        int q_head = 0;
        int token  = 0;
        causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = token < valid_tokens
                              ? __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)])
                              : 0.0f;
        }
        normalized_hadamard_d256_inplace(q_values, lane);
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float amax      = fmaxf(fabsf(x0), fabsf(x1));
            amax            = warp_max(amax, FullMask);
            const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_store_query_code(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_store_query_code(q_i8, row, d1, kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale_tmp[row * Groups + grp] = qs; }
        }
    }
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // One tile's codes and scales of K (producers) or V (consumers) into buffer `buf`.
    auto issue_tile = [&](int tile_k0, bool keys, int buf, int t, int threads) {
        const int page          = block_table[tile_k0 >> kPagedKVPageShift];
        const std::int8_t* src  = keys ? cache_k_i8 : cache_v_i8;
        const __half* scale_src = keys ? cache_k_scale : cache_v_scale;
        std::int8_t* codes      = (keys ? k_buf : v_buf) + buf * kKvBytes;
        __half* scales          = (keys ? k_scale_buf : v_scale_buf) + buf * Bc * Groups;
        for (int key_l = t; key_l < Bc; key_l += threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const std::int64_t off = kv_cache_int8_quant_scale_index<Geometry>(
                    page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<8>(&scales[key_l * Groups], &scale_src[off]);
            } else {
                store_vec(&scales[key_l * Groups], make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = t; chunk < Bc * (D / 16); chunk += threads) {
            const int key_l  = chunk / (D / 16);
            const int dc     = chunk - key_l * (D / 16);
            const int key    = tile_k0 + key_l;
            std::int8_t* dst = keys ? &codes[key_l * D + causal_swizzle(key_l, dc * 8) * 2]
                                    : &codes[v_offset(key_l, dc)];
            if (key >= split_start && key < split_end) {
                const std::int64_t off = kv_cache_int8_quant_code_index<Geometry>(
                    page, kv_head, dc * 16, key & kPagedKVPageMask);
                ninfer::ops::cp_async<16>(dst, &src[off]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    if (warp < kProducerWarps) {
        // ---- Producers: scores, softmax and probabilities of every tile, one tile ahead of PV.
        // ----
        const int producer_row_base = warp * 16;
        const int row0              = producer_row_base + gid;
        const int row1              = row0 + 8;
        float q_scale_r0[Groups];
        float q_scale_r1[Groups];
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            float qs0     = (lid == 0 && row0 < tile_tokens * Geometry::GroupSize)
                                ? q_scale_tmp[row0 * Groups + g]
                                : 0.0f;
            float qs1     = (lid == 0 && row1 < RowCount) ? q_scale_tmp[row1 * Groups + g] : 0.0f;
            q_scale_r0[g] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[g] = __shfl_sync(FullMask, qs1, gid * 4);
        }
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        causal_row_to_qt<Geometry>(row0, kv_head, q_head0, token0);
        causal_row_to_qt<Geometry>(row1, kv_head, q_head1, token1);
        const bool valid0 = row0 < tile_tokens * Geometry::GroupSize;
        const bool valid1 = row1 < tile_tokens * Geometry::GroupSize;
        const int qabs0   = valid0 ? pos[token0] : -1;
        const int qabs1   = valid1 ? pos[token1] : -1;
        int min_qabs      = pos[0];
        for (int t = 1; t < tile_tokens; ++t) { min_qabs = min(min_qabs, pos[t]); }

        float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
        float l0 = 0.0f, l1 = 0.0f;
        issue_tile(first_tile, true, 0, tid, ProducerThreads);
        if (key_blocks > 1) { issue_tile(first_tile + Bc, true, 1, tid, ProducerThreads); }
        for (int kb = 0; kb < key_blocks; ++kb) {
            const int k0  = first_tile + kb * Bc;
            const int buf = kb & 1;
            if (kb + 1 < key_blocks) {
                ninfer::ops::cp_wait<1>();
            } else {
                ninfer::ops::cp_wait<0>();
            }
            sync(kProducers, ProducerThreads); // K(kb) in shared memory
            const __nv_bfloat16* k_b16 =
                reinterpret_cast<const __nv_bfloat16*>(k_buf + buf * kKvBytes);
            const __half* k_scale_s = k_scale_buf + buf * Bc * Groups;
            __half* p_sw            = p_buf + buf * (kBr * Bc) + producer_row_base * Bc;
            float* alpha_s          = alpha_buf + buf * kBr;

            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = 0.0f;
                score[nt][1] = 0.0f;
                score[nt][2] = 0.0f;
                score[nt][3] = 0.0f;
            }
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = g * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(
                        af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                        smem_addr(&q_b16[(producer_row_base + a_rowoff) * DB16 +
                                         causal_swizzle(producer_row_base + a_rowoff, acol)]));
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = g * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + causal_swizzle(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    float ka       = 0.0f;
                    float kb2      = 0.0f;
                    if (gid == 0) {
                        ka  = __half2float(k_scale_s[keya * Groups + g]);
                        kb2 = __half2float(k_scale_s[keyb * Groups + g]);
                    }
                    ka  = __shfl_sync(FullMask, ka, lid);
                    kb2 = __shfl_sync(FullMask, kb2, lid);
                    score[nt][0] += q_scale_r0[g] * ka * static_cast<float>(c0);
                    score[nt][1] += q_scale_r0[g] * kb2 * static_cast<float>(c1);
                    score[nt][2] += q_scale_r1[g] * ka * static_cast<float>(c2);
                    score[nt][3] += q_scale_r1[g] * kb2 * static_cast<float>(c3);
                }
            }
            // K(kb) and its scales read by every producer: the buffer takes K(kb + 2).
            sync(kProducers, ProducerThreads);
            if (kb + 2 < key_blocks) { issue_tile(k0 + 2 * Bc, true, buf, tid, ProducerThreads); }

            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
            const bool full_tile =
                k0 >= split_start && k0 + Bc <= split_end && k0 + Bc - 1 <= min_qabs;
            if (full_tile) {
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    score[nt][0] = valid0 ? score[nt][0] * scale : -CUDART_INF_F;
                    score[nt][1] = valid0 ? score[nt][1] * scale : -CUDART_INF_F;
                    score[nt][2] = valid1 ? score[nt][2] * scale : -CUDART_INF_F;
                    score[nt][3] = valid1 ? score[nt][3] * scale : -CUDART_INF_F;
                    bm0          = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                    bm1          = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
                }
            } else {
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int key0 = k0 + nt * 8 + 2 * lid;
                    const int key1 = key0 + 1;
                    score[nt][0] =
                        (valid0 && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                            ? score[nt][0] * scale
                            : -CUDART_INF_F;
                    score[nt][1] =
                        (valid0 && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                            ? score[nt][1] * scale
                            : -CUDART_INF_F;
                    score[nt][2] =
                        (valid1 && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                            ? score[nt][2] * scale
                            : -CUDART_INF_F;
                    score[nt][3] =
                        (valid1 && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                            ? score[nt][3] * scale
                            : -CUDART_INF_F;
                    bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                    bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
                }
            }
            bm0             = warp_max<4>(bm0, FullMask);
            bm1             = warp_max<4>(bm1, FullMask);
            const float nm0 = fmaxf(m0, bm0);
            const float nm1 = fmaxf(m1, bm1);
            const float alpha0 =
                (m0 == -CUDART_INF_F) ? 0.0f : causal_exp_difference(m0, nm0, Log2E);
            const float alpha1 =
                (m1 == -CUDART_INF_F) ? 0.0f : causal_exp_difference(m1, nm1, Log2E);
            // P(kb - 2) in this buffer must be read by the consumers before it is rewritten.
            if (kb >= 2) { sync(kEmpty0 + buf, AllThreads); }
            float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                      ? causal_exp_difference(score[nt][0], nm0, Log2E)
                                      : 0.0f;
                const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                      ? causal_exp_difference(score[nt][1], nm0, Log2E)
                                      : 0.0f;
                const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                      ? causal_exp_difference(score[nt][2], nm1, Log2E)
                                      : 0.0f;
                const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                      ? causal_exp_difference(score[nt][3], nm1, Log2E)
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_sw[gid * Bc + causal_probability_swizzle<Bc>(gid, col0)] = __float2half_rn(p00);
                p_sw[gid * Bc + causal_probability_swizzle<Bc>(gid, col1)] = __float2half_rn(p01);
                p_sw[(gid + 8) * Bc + causal_probability_swizzle<Bc>(gid + 8, col0)] =
                    __float2half_rn(p10);
                p_sw[(gid + 8) * Bc + causal_probability_swizzle<Bc>(gid + 8, col1)] =
                    __float2half_rn(p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);
            l0  = l0 * alpha0 + bl0;
            l1  = l1 * alpha1 + bl1;
            m0  = nm0;
            m1  = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
            arrive(kFull0 + buf, AllThreads);
        }
        if (lid == 0) {
            if (valid0) {
                partial_m[causal_stat_index<Geometry>(q_head0, token0, split, partial_width)] = m0;
                partial_l[causal_stat_index<Geometry>(q_head0, token0, split, partial_width)] = l0;
            }
            if (valid1) {
                partial_m[causal_stat_index<Geometry>(q_head1, token1, split, partial_width)] = m1;
                partial_l[causal_stat_index<Geometry>(q_head1, token1, split, partial_width)] = l1;
            }
        }
    } else {
        // ---- Consumers: PV of every tile with V converted in registers, one tile behind. ----
        const int ctid              = tid - ProducerThreads;
        const int cwarp             = warp - kProducerWarps;
        const int row_tile          = cwarp % kRowTiles;
        const int grp               = cwarp / kRowTiles; // this warp's 64 dimensions
        const int consumer_row_base = row_tile * 16;
        // ldmatrix.trans rows: keys (lane & 7) + 8 * ((lane >> 3) & 1) of the k step, 16-byte chunk
        // 4 * grp + 2 * half + (lane >> 4) of the dimensions.
        const int v_key = (lane & 7) + (((lane >> 3) & 1) << 3);
        const int v_sub = lane >> 4;

        float acc[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
        }
        issue_tile(first_tile, false, 0, ctid, ConsumerThreads);
        if (key_blocks > 1) { issue_tile(first_tile + Bc, false, 1, ctid, ConsumerThreads); }
        for (int kb = 0; kb < key_blocks; ++kb) {
            const int buf = kb & 1;
            if (kb + 1 < key_blocks) {
                ninfer::ops::cp_wait<1>();
            } else {
                ninfer::ops::cp_wait<0>();
            }
            sync(kConsumers, ConsumerThreads); // V(kb) in shared memory
            sync(kFull0 + buf, AllThreads);    // P(kb), alpha(kb) written
            const __half* p_consumer = p_buf + buf * (kBr * Bc) + consumer_row_base * Bc;
            const float* alpha_s     = alpha_buf + buf * kBr;
            const std::int8_t* vb    = v_buf + buf * kKvBytes;
            const __half* vs         = v_scale_buf + buf * Bc * Groups;
            const float alpha0       = alpha_s[consumer_row_base + gid];
            const float alpha1       = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                acc[n][0] *= alpha0;
                acc[n][1] *= alpha0;
                acc[n][2] *= alpha1;
                acc[n][3] *= alpha1;
            }
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_consumer[a_rowoff * Bc +
                                                  causal_probability_swizzle<Bc>(a_rowoff, pcol)]));
                // Scales of this thread's B-fragment keys (2 lid, 2 lid + 1; + 8, + 9) for its
                // group.
                const int kb0 = k * 16 + 2 * lid;
                const __half2 s_lo =
                    __halves2half2(vs[kb0 * Groups + grp], vs[(kb0 + 1) * Groups + grp]);
                const __half2 s_hi =
                    __halves2half2(vs[(kb0 + 8) * Groups + grp], vs[(kb0 + 9) * Groups + grp]);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int key = k * 16 + v_key;
                    unsigned r0, r1, r2, r3;
                    ldmatrix_x4_trans_b16(
                        r0, r1, r2, r3, smem_addr(&vb[v_offset(key, 4 * grp + 2 * half + v_sub)]));
                    // r0: keys 0-7 x dims 32 half + 0-15, r1: keys 8-15 x the same, r2 / r3: dims
                    // + 16.
                    const unsigned x0 = r0 ^ 0x80808080u, x1 = r1 ^ 0x80808080u;
                    const unsigned x2 = r2 ^ 0x80808080u, x3 = r3 ^ 0x80808080u;
                    // n-tiles of this half: even / odd dimensions of block 0, then of block 1.
                    const int n0 = half * 4;
                    mma_f16(acc[n0][0], acc[n0][1], acc[n0][2], acc[n0][3], pf[0], pf[1], pf[2],
                            pf[3], v_pair(x0, 0x4240u, s_lo), v_pair(x1, 0x4240u, s_hi));
                    mma_f16(acc[n0 + 1][0], acc[n0 + 1][1], acc[n0 + 1][2], acc[n0 + 1][3], pf[0],
                            pf[1], pf[2], pf[3], v_pair(x0, 0x4341u, s_lo),
                            v_pair(x1, 0x4341u, s_hi));
                    mma_f16(acc[n0 + 2][0], acc[n0 + 2][1], acc[n0 + 2][2], acc[n0 + 2][3], pf[0],
                            pf[1], pf[2], pf[3], v_pair(x2, 0x4240u, s_lo),
                            v_pair(x3, 0x4240u, s_hi));
                    mma_f16(acc[n0 + 3][0], acc[n0 + 3][1], acc[n0 + 3][2], acc[n0 + 3][3], pf[0],
                            pf[1], pf[2], pf[3], v_pair(x2, 0x4341u, s_lo),
                            v_pair(x3, 0x4341u, s_hi));
                }
            }
            // P(kb) and alpha(kb) read: the producers may write tile kb + 2 there.
            if (kb + 2 < key_blocks) { arrive(kEmpty0 + buf, AllThreads); }
            if (kb + 2 < key_blocks) {
                sync(kConsumers, ConsumerThreads); // V(kb) read by every consumer
                issue_tile(first_tile + (kb + 2) * Bc, false, buf, ctid, ConsumerThreads);
            }
        }
        // n-tile 2 b + e (b = 16-dimension block of the group, e = 0 even / 1 odd dimensions):
        // column j is dimension 16 b + 2 j + e, so a thread's c0/c1 of the even and odd tiles are
        // dimensions 4 lid + {0, 1} and 4 lid + {2, 3} of the block.
        const int row0 = consumer_row_base + gid;
        const int row1 = row0 + 8;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int d       = grp * 64 + b * 16 + 4 * lid;
            const float* even = acc[2 * b];
            const float* odd  = acc[2 * b + 1];
            if (row0 < tile_tokens * Geometry::GroupSize) {
                int q_head = 0;
                int token  = 0;
                causal_row_to_qt<Geometry>(row0, kv_head, q_head, token);
                causal_store_partial_pair(&partial_acc[causal_partial_index<Geometry>(
                                              q_head, d, token, split, partial_width)],
                                          even[0], odd[0]);
                causal_store_partial_pair(&partial_acc[causal_partial_index<Geometry>(
                                              q_head, d + 2, token, split, partial_width)],
                                          even[1], odd[1]);
            }
            if (row1 < tile_tokens * Geometry::GroupSize) {
                int q_head = 0;
                int token  = 0;
                causal_row_to_qt<Geometry>(row1, kv_head, q_head, token);
                causal_store_partial_pair(&partial_acc[causal_partial_index<Geometry>(
                                              q_head, d, token, split, partial_width)],
                                          even[2], odd[2]);
                causal_store_partial_pair(&partial_acc[causal_partial_index<Geometry>(
                                              q_head, d + 2, token, split, partial_width)],
                                          even[3], odd[3]);
            }
        }
    }
}

} // namespace ninfer::ops::detail
