#pragma once

// ninfer::ops::detail - private launch prototypes of the DSpark drafter Ops (dspark_draft.h).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Key chunks per KV head: at most kBlockAttentionChunks, each at least kBlockAttentionMinKeys keys
// (a 32-key tile per warp), so a short context takes few blocks and a long one all of them.
inline constexpr int kBlockAttentionChunks   = 64;
inline constexpr int kBlockAttentionMinKeys  = 128;
inline constexpr int kBlockAttentionMaxBlock = 8;
inline constexpr int kRecurrentRank          = 256;
inline constexpr int kRecurrentMaxSteps      = 15;
inline constexpr int kRecurrentBlocks        = 256;

struct MixNorms {
    const void* p[8];
};

// The FP32 inverse frequencies of a 128-wide split-half RoPE (dspark_draft.h).
struct InvFrequencies {
    float v[64];
};

struct StoreLayer {
    const void* raw;
    const void* k_norm;
    void* k;
    void* v;
    int capacity;
};

struct StoreLayers {
    StoreLayer l[8];
};

struct RecurrentLaunch {
    const void *weight, *embedding, *state_in, *folded, *folded_bias, *base;
    void* state_out;
    const float* context;
    const int* anchor;
    int* drafts;
    float* choice;
    float* raw;
    float* best_value;
    int* best_id;
    unsigned* done;
    int in, step, rows, first, rank, ranks;
    float cap;
};

void feature_mix_rmsnorm_launch(const void* features, const void* logits, const MixNorms& norms,
                                int hidden, int width, std::size_t block_stride, int states,
                                int rows, float eps, const int* count, void* out, void* last,
                                cudaStream_t stream);
void consecutive_positions_launch(const int* start, const int* minus, int* out, int tokens,
                                  cudaStream_t stream);
void head_rmsnorm_rope_launch(void* x, int rows, int tokens, int query_heads, int key_heads,
                              const void* q_norm, const void* k_norm, const int* positions,
                              const InvFrequencies& inv, float eps, cudaStream_t stream);
void context_kv_store_launch(const StoreLayers& layers, int count_layers, int kv_heads, int width,
                             const int* positions, const int* count, const InvFrequencies& inv,
                             float eps, cudaStream_t stream);
void block_attention_launch(const void* qkv, int rows, int query_heads, int kv_heads, int block,
                            const void* k_cache, const void* v_cache, int capacity,
                            const int* frontier, int window, bool sliding, float scale,
                            float* partial, void* out, cudaStream_t stream);
void add_rmsnorm_launch(void* residual, const void* y, const void* weight, int hidden, int tokens,
                        float eps, void* out, cudaStream_t stream);
void recurrent_head_context_launch(const void* weight, int in, int rank, const void* bias,
                                   const void* hidden, int width, int steps, const void* context,
                                   float* out, cudaStream_t stream);
void recurrent_head_step_launch(const RecurrentLaunch& launch, cudaStream_t stream);
void recurrent_head_select_launch(const float* choice, int* drafts, int last, cudaStream_t stream);

} // namespace ninfer::ops::detail
