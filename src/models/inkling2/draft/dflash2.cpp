#include "models/inkling2/draft/dflash2.h"

#include "core/device.h"
#include "core/device_scope.h"
#include "core/weight_view.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/dflash2_draft.h"
#include "ninfer/ops/dspark_draft.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/tp2/linear.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::inkling2::draft {
namespace {

[[nodiscard]] std::size_t at(int rank) { return static_cast<std::size_t>(rank); }

constexpr std::int32_t kHeadDim    = 128;
constexpr std::int32_t kCandidates = 16;

// The workspace of one materialization of up to `max_width` columns, then of one proposal (the
// materialization's buffers are released before the block's).
[[nodiscard]] std::size_t workspace_for(const DFlash2Config& config, const DFlash2Config& rank,
                                        std::int32_t rows, std::int32_t max_width) {
    const std::size_t H = config.hidden_size, B = config.block_size, W = std::size_t(max_width);
    const std::size_t G = B - 1, layers = config.num_layers;
    const std::size_t kv = 2 * std::size_t(kHeadDim) * rank.num_key_value_heads;
    const std::size_t qkv =
        std::size_t(kHeadDim) * (rank.num_attention_heads + 2 * rank.num_key_value_heads);
    const std::size_t heads = std::size_t(kHeadDim) * rank.num_attention_heads;
    const std::size_t F2    = 2 * std::size_t(rank.intermediate_size);
    const std::size_t materialize =
        H * W * (kFeatures + 2) * 2 + layers * kv * W * 2 + 8 * (std::size_t(256) << 10);
    const std::size_t block =
        B * 2 * (6 * H + 2 * config.conv_coefficients() + qkv + heads + F2 + F2 / 2) +
        ops::block_attention_workspace_bytes(static_cast<std::int32_t>(rank.num_attention_heads),
                                             static_cast<std::int32_t>(B)) +
        std::size_t(rows) * G * 2 + std::size_t(config.selector_rank) * G * 2 +
        ops::vocab_top16_workspace_bytes(static_cast<std::int32_t>(G)) +
        ops::candidate_selector_path_workspace_capacity_bytes(1, static_cast<std::int32_t>(G), 1,
                                                              1) +
        2 * 2 * kCandidates * G * 2 * 4 + 32 * (std::size_t(256) << 10);
    return std::max(materialize, block) + (std::size_t(4) << 20);
}

} // namespace

// Rank r's device state and workspace.
struct DFlash2Drafter::Rank {
    std::vector<std::unique_ptr<DeviceBuffer>> memory;
    std::vector<ops::DraftKVCacheView> caches, image;
    Tensor pending; // BF16 [H, max_width, 4]
    execution::FeatureSink sink;
    Tensor input;           // BF16 [H, B]: anchor column, then the mask token's
    Tensor residual;        // BF16 [H, B]
    Tensor positions;       // I32 [max_width]
    Tensor block_positions; // I32 [B]
    Tensor count;           // I32 [1]: columns of the pending features to consume
    Tensor frontier;        // I32 [1]: the eager append's frontier
    Tensor pairs;           // FP32 [2, 16, B - 1, ranks]: the candidates' exchange
    Tensor ids, values;     // I32 / FP32 [16, B - 1]: the merged candidates
    Tensor proposal_q;      // FP32 [16, B - 1]: the selector's distribution (one-hot here)
    Tensor anchor, drafts;  // I32 [1], [B - 1]: the selector's (16-byte aligned) operands
    Tensor sampling;        // the selector's SamplingConfig (greedy)
    Weight embedding;
    std::unique_ptr<WorkspaceArena> work;

    Tensor buffer(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t& total) {
        std::size_t elements = 1;
        for (const auto d : shape) { elements *= static_cast<std::size_t>(d); }
        const std::size_t bytes = elements * (dtype == DType::BF16 ? 2 : 4);
        memory.push_back(std::make_unique<DeviceBuffer>(bytes));
        total += bytes;
        return Tensor(memory.back()->p, dtype, shape);
    }
};

DFlash2Drafter::DFlash2Drafter(const DFlash2Config& config, std::array<DFlash2Rank, 2> ranks,
                               int width, execution::ForwardCollectives collectives,
                               std::int32_t max_width, bool with_image,
                               std::array<Tensor, 2> features)
    : config_(config), ranks_(ranks), width_(width), tp_(collectives), max_width_(max_width) {
    if ((width != 1 && width != 2) || max_width < std::int32_t(config.block_size) ||
        config.head_dim != std::uint32_t(kHeadDim) || config.block_size != 8 ||
        config.selector_top_k != std::uint32_t(kCandidates) ||
        config.sliding_window > std::uint32_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Inkling 2 DFlash2 drafter: unsupported geometry or bounds");
    }
    const auto H      = static_cast<std::int32_t>(config.hidden_size);
    const auto B      = static_cast<std::int32_t>(config.block_size);
    const auto G      = B - 1;
    const auto window = static_cast<std::int32_t>(config.sliding_window);
    const float eps   = config.rms_norm_eps;
    for (int r = 0; r < width_; ++r) {
        const DFlash2Rank& d = ranks_[at(r)];
        if (d.config == nullptr || d.parameters == nullptr || d.target == nullptr ||
            d.parameters->layers.size() != config.num_layers) {
            throw std::invalid_argument("Inkling 2 DFlash2 drafter: incomplete rank " +
                                        std::to_string(r));
        }
        const ScopedCurrentDevice scope(d.device);
        rank_[at(r)]      = std::make_unique<Rank>();
        Rank& own         = *rank_[at(r)];
        std::size_t total = 0;
        const auto kv     = static_cast<std::int32_t>(d.config->num_key_value_heads);
        for (std::uint32_t l = 0; l < config.num_layers; ++l) {
            own.caches.push_back({own.buffer(DType::BF16, {kHeadDim, kv, window}, total),
                                  own.buffer(DType::BF16, {kHeadDim, kv, window}, total), window});
            if (with_image) {
                own.image.push_back({own.buffer(DType::BF16, {kHeadDim, kv, window}, total),
                                     own.buffer(DType::BF16, {kHeadDim, kv, window}, total),
                                     window});
            }
        }
        if (features[at(r)].data != nullptr) {
            const Tensor& shared = features[at(r)];
            if (shared.dtype != DType::BF16 || shared.ne[0] != H || shared.ne[1] != max_width ||
                shared.ne[2] != std::int32_t(kFeatures) || shared.nb[1] != std::int64_t(H) * 2) {
                throw std::invalid_argument(
                    "Inkling 2 DFlash2 drafter: shared features are not BF16 [H, max_width, 4]");
            }
            own.pending = shared;
        } else {
            own.pending = own.buffer(DType::BF16, {H, max_width, std::int32_t(kFeatures)}, total);
        }
        own.input           = own.buffer(DType::BF16, {H, B}, total);
        own.residual        = own.buffer(DType::BF16, {H, B}, total);
        own.positions       = own.buffer(DType::I32, {max_width}, total);
        own.block_positions = own.buffer(DType::I32, {B}, total);
        own.count           = own.buffer(DType::I32, {1}, total);
        own.frontier        = own.buffer(DType::I32, {1}, total);
        own.pairs           = own.buffer(DType::FP32, {2, kCandidates, G, width_}, total);
        own.ids             = own.buffer(DType::I32, {kCandidates, G}, total);
        own.values          = own.buffer(DType::FP32, {kCandidates, G}, total);
        own.proposal_q      = own.buffer(DType::FP32, {kCandidates, G}, total);
        own.anchor          = own.buffer(DType::I32, {1}, total);
        own.drafts          = own.buffer(DType::I32, {G}, total);
        own.sampling =
            own.buffer(DType::I32, {std::int32_t((sizeof(ops::SamplingConfig) + 3) / 4)}, total);
        const ops::SamplingConfig greedy{};
        CUDA_CHECK(cudaMemcpy(own.sampling.data, &greedy, sizeof(greedy), cudaMemcpyHostToDevice));
        const std::int32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(own.count.data, &zero, 4, cudaMemcpyHostToDevice));
        const Model& model = d.target->model;
        own.embedding      = native_weight(
            model.weight(model.weights().text.token_embedding, d.target->device).view);
        const auto rows  = static_cast<std::int32_t>(d.target->text.output_head.weight.n);
        workspace_bytes_ = workspace_for(config, *d.config, rows, max_width);
        own.work         = std::make_unique<WorkspaceArena>(workspace_bytes_);
        state_bytes_     = total;

        // The mask token's normed embedding fills columns 1 .. B - 1 of the block's input.
        const std::int32_t mask = static_cast<std::int32_t>(config.mask_token_id);
        Tensor id               = own.work->alloc(DType::I32, {1});
        CUDA_CHECK(cudaMemcpy(id.data, &mask, 4, cudaMemcpyHostToDevice));
        Tensor embedded = own.work->alloc(DType::BF16, {H, 1});
        ops::embedding(id, own.embedding, embedded, d.stream);
        Tensor column = own.input.slice(1, 1, 1);
        ops::rmsnorm(embedded, d.target->text.embedding_norm, eps, false, column, d.stream);
        for (std::int32_t c = 2; c < B; ++c) {
            CUDA_CHECK(::ninfer::memcpy_device_async(static_cast<std::byte*>(own.input.data) +
                                                         std::size_t(c) * H * 2,
                                                     column.data, std::size_t(H) * 2, d.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        own.work->reset();
    }
}

DFlash2Drafter::~DFlash2Drafter() = default;

std::size_t DFlash2Drafter::planned_bytes(const DFlash2Config& config, const DFlash2Config& rank,
                                          std::int32_t rows, std::int32_t max_width, bool image,
                                          bool own_features) {
    const std::size_t H = config.hidden_size, B = config.block_size, D = kHeadDim;
    const std::size_t kv = rank.num_key_value_heads, G = B - 1;
    std::size_t state =
        config.num_layers * 2 * D * kv * config.sliding_window * 2 * (image ? 2 : 1);
    state += (own_features ? H * std::size_t(max_width) * kFeatures * 2 : 0) + 2 * H * B * 2 +
             std::size_t(max_width) * 4 + B * 4 + 8 + 2 * kCandidates * G * 2 * 4 +
             3 * kCandidates * G * 4 + sizeof(ops::SamplingConfig) + 4 + 4 + G * 4;
    return state + workspace_for(config, rank, rows, max_width);
}

template <class Body>
void DFlash2Drafter::each(Body&& body) {
    const ScopedCurrentDevice restore;
    for (int r = 0; r < width_; ++r) {
        ScopedCurrentDevice::select(ranks_[at(r)].device);
        body(r);
    }
}

const execution::FeatureSink* DFlash2Drafter::sink(int rank, std::int32_t width) {
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 DFlash2 drafter: sink wider than planned");
    }
    Rank& own = *rank_[at(rank)];
    own.sink  = {config_.feature_visits, own.pending.slice(1, 0, width)};
    return &own.sink;
}

void DFlash2Drafter::materialize(const std::array<Tensor, 2>& frontier, std::int32_t width) {
    const auto H = static_cast<std::int32_t>(config_.hidden_size);
    const auto L = static_cast<std::int32_t>(config_.num_layers);
    each([&](int r) {
        const DFlash2Rank& d       = ranks_[at(r)];
        const DFlash2Parameters& p = *d.parameters;
        Rank& own                  = *rank_[at(r)];
        const cudaStream_t s       = d.stream;
        auto scope                 = own.work->scope();
        Tensor positions(own.positions.data, DType::I32, {width});
        ops::consecutive_positions(frontier[at(r)], &own.count, positions, s);
        // Column t of the projection's input: the four states of column t, in order.
        const auto F  = static_cast<std::int32_t>(kFeatures);
        Tensor packed = own.work->alloc(DType::BF16, {F * H, width});
        for (std::int32_t j = 0; j < F; ++j) {
            CUDA_CHECK(
                cudaMemcpy2DAsync(static_cast<std::byte*>(packed.data) + std::size_t(j) * H * 2,
                                  std::size_t(F) * H * 2,
                                  static_cast<const std::byte*>(own.pending.data) +
                                      std::size_t(j) * std::size_t(own.pending.nb[2]),
                                  std::size_t(H) * 2, std::size_t(H) * 2, std::size_t(width),
                                  cudaMemcpyDeviceToDevice, s));
        }
        Tensor projected = own.work->alloc(DType::BF16, {H, width});
        ops::linear(packed, p.feature_projection.weight, projected, p.feature_projection.policy,
                    *own.work, s);
        Tensor context = own.work->alloc(DType::BF16, {H, width});
        ops::rmsnorm(projected, p.context_norm, config_.rms_norm_eps, false, context, s);
        std::array<Tensor, 8> raw;
        std::array<const Tensor*, 8> raw_views{}, key_norms{};
        std::array<ops::DraftKVCacheView, 8> caches{};
        for (std::int32_t l = 0; l < L; ++l) {
            const auto& layer   = p.layers[std::size_t(l)];
            raw[std::size_t(l)] = own.work->alloc(DType::BF16, {layer.context.weight.n, width});
            ops::linear(context, layer.context.weight, raw[std::size_t(l)], layer.context.policy,
                        *own.work, s);
            raw_views[std::size_t(l)] = &raw[std::size_t(l)];
            key_norms[std::size_t(l)] = &layer.key_norm;
            caches[std::size_t(l)]    = own.caches[std::size_t(l)];
        }
        ops::context_kv_store(raw_views, key_norms, caches, L, positions, own.count,
                              config_.rope_theta, config_.rms_norm_eps, s);
    });
}

void DFlash2Drafter::append(std::int32_t first, std::int32_t width) {
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 DFlash2 drafter: append wider than planned");
    }
    std::array<Tensor, 2> frontier;
    each([&](int r) {
        Rank& own = *rank_[at(r)];
        CUDA_CHECK(cudaStreamSynchronize(ranks_[at(r)].stream));
        own.work->reset();
        const std::int32_t end = first + width;
        CUDA_CHECK(cudaMemcpy(own.frontier.data, &end, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(own.count.data, &width, 4, cudaMemcpyHostToDevice));
        frontier[at(r)] = own.frontier;
    });
    materialize(frontier, width);
    clear_pending();
}

void DFlash2Drafter::clear_pending() {
    each([&](int r) {
        CUDA_CHECK(cudaStreamSynchronize(ranks_[at(r)].stream));
        const std::int32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(rank_[at(r)]->count.data, &zero, 4, cudaMemcpyHostToDevice));
    });
}

void DFlash2Drafter::catch_up(std::int32_t frontier, std::int32_t width) {
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 DFlash2 drafter: pending width out of range");
    }
    std::array<Tensor, 2> frontiers;
    each([&](int r) {
        Rank& own = *rank_[at(r)];
        CUDA_CHECK(cudaStreamSynchronize(ranks_[at(r)].stream));
        own.work->reset();
        CUDA_CHECK(cudaMemcpy(own.frontier.data, &frontier, 4, cudaMemcpyHostToDevice));
        frontiers[at(r)] = own.frontier;
    });
    materialize(frontiers, width);
    clear_pending();
}

void DFlash2Drafter::follow(const std::array<Tensor, 2>& frontier, std::int32_t width) {
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 DFlash2 drafter: pending width out of range");
    }
    each([&](int r) { rank_[at(r)]->work->reset(); });
    materialize(frontier, width);
}

void DFlash2Drafter::save_image() { copy_image(true); }

void DFlash2Drafter::restore_image() {
    copy_image(false);
    clear_pending();
}

void DFlash2Drafter::copy_image(bool save) {
    each([&](int r) {
        Rank& own = *rank_[at(r)];
        if (own.image.empty()) {
            throw std::logic_error("Inkling 2 DFlash2 drafter has no anchor image");
        }
        const cudaStream_t s = ranks_[at(r)].stream;
        const auto copy      = [&](const Tensor& live, const Tensor& image) {
            CUDA_CHECK(::ninfer::memcpy_device_async(
                save ? image.data : live.data, save ? live.data : image.data, live.bytes(), s));
        };
        for (std::size_t l = 0; l < own.caches.size(); ++l) {
            copy(own.caches[l].k, own.image[l].k);
            copy(own.caches[l].v, own.image[l].v);
        }
        CUDA_CHECK(cudaStreamSynchronize(s));
    });
}

void DFlash2Drafter::commit(const std::array<Tensor, 2>& count) {
    each([&](int r) {
        CUDA_CHECK(::ninfer::memcpy_device_async(rank_[at(r)]->count.data, count[at(r)].data, 4,
                                                 ranks_[at(r)].stream));
    });
}

void DFlash2Drafter::project_rows(const std::array<Tensor, 2>& x,
                                  std::array<const LinearParameters*, 2> weight,
                                  std::array<Tensor, 2>& out) {
    if (width_ == 1) {
        const ScopedCurrentDevice scope(ranks_[0].device);
        ops::linear(x[0], weight[0]->weight, out[0], weight[0]->policy, *rank_[0]->work,
                    ranks_[0].stream);
        return;
    }
    std::array<Tensor, 2> staging;
    each([&](int r) {
        staging[at(r)] =
            rank_[at(r)]->work->alloc(DType::BF16, {out[at(r)].ne[0], out[at(r)].ne[1]});
    });
    ops::linear_row_parallel(x, {weight[0]->weight, weight[1]->weight}, out, staging,
                             weight[0]->policy, {rank_[0]->work.get(), rank_[1]->work.get()},
                             *tp_.execution, *tp_.events);
}

void DFlash2Drafter::propose(const std::array<Tensor, 2>& anchor,
                             const std::array<Tensor, 2>& frontier,
                             const std::array<Tensor, 2>& drafts, std::int32_t width,
                             std::int32_t proposals) {
    const auto H      = static_cast<std::int32_t>(config_.hidden_size);
    const auto B      = static_cast<std::int32_t>(config_.block_size);
    const auto C      = static_cast<std::int32_t>(config_.conv_coefficients());
    const auto window = static_cast<std::int32_t>(config_.sliding_window);
    const float eps   = config_.rms_norm_eps;
    const float scale = static_cast<float>(1.0 / std::sqrt(double(kHeadDim)));
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 DFlash2 drafter: pending width out of range");
    }
    if (proposals < 1 || proposals > B - 1) {
        throw std::invalid_argument("Inkling 2 DFlash2 drafter: proposals out of range");
    }
    each([&](int r) { rank_[at(r)]->work->reset(); });
    materialize(frontier, width);

    std::array<Tensor, 2> x, coefficients, z;
    each([&](int r) {
        const DFlash2Rank& d = ranks_[at(r)];
        Rank& own            = *rank_[at(r)];
        const cudaStream_t s = d.stream;
        ops::consecutive_positions(frontier[at(r)], nullptr, own.block_positions, s);
        Tensor embedded = own.work->alloc(DType::BF16, {H, 1});
        ops::embedding(anchor[at(r)], own.embedding, embedded, s);
        Tensor column = own.input.slice(1, 0, 1);
        ops::rmsnorm(embedded, d.target->text.embedding_norm, eps, false, column, s);
        CUDA_CHECK(
            ::ninfer::memcpy_device_async(own.residual.data, own.input.data, own.input.bytes(), s));
        x[at(r)]            = own.work->alloc(DType::BF16, {H, B});
        coefficients[at(r)] = own.work->alloc(DType::BF16, {C, B});
        z[at(r)]            = own.work->alloc(DType::BF16, {H, B});
    });
    // The input side of a sublayer's convolution: the norm, the kernel projection of the normed
    // input (kept for the output side) and the convolved input in `out`.
    const auto convolve_input = [&](int r, const Tensor& norm, const DFlash2ConvParameters& conv,
                                    Tensor& out) {
        const DFlash2Rank& d = ranks_[at(r)];
        Rank& own            = *rank_[at(r)];
        const cudaStream_t s = d.stream;
        ops::rmsnorm(own.residual, norm, eps, false, x[at(r)], s);
        ops::linear(x[at(r)], conv.kernel_projection.weight, coefficients[at(r)],
                    conv.kernel_projection.policy, *own.work, s);
        ops::dynamic_conv_input(x[at(r)], coefficients[at(r)], conv.base_kernel, out, s);
    };
    for (std::uint32_t l = 0; l < config_.num_layers; ++l) {
        std::array<Tensor, 2> o, h;
        each([&](int r) {
            const DFlash2Rank& d = ranks_[at(r)];
            Rank& own            = *rank_[at(r)];
            const auto& layer    = d.parameters->layers[l];
            const cudaStream_t s = d.stream;
            const auto nq        = static_cast<std::int32_t>(d.config->num_attention_heads);
            const auto nkv       = static_cast<std::int32_t>(d.config->num_key_value_heads);
            Tensor convolved     = own.work->alloc(DType::BF16, {H, B});
            convolve_input(r, layer.input_norm, layer.attention_conv, convolved);
            Tensor qkv = own.work->alloc(DType::BF16, {layer.attention_input.weight.n, B});
            ops::linear(convolved, layer.attention_input.weight, qkv, layer.attention_input.policy,
                        *own.work, s);
            ops::head_rmsnorm_rope(qkv, nq, nkv, layer.query_norm, layer.key_norm,
                                   own.block_positions, config_.rope_theta, eps, s);
            o[at(r)] = own.work->alloc(DType::BF16, {nq * kHeadDim, B});
            ops::sliding_block_attention(qkv, nq, nkv, own.caches[l], frontier[at(r)], window,
                                         scale, *own.work, o[at(r)], s);
        });
        project_rows(o,
                     {&ranks_[0].parameters->layers[l].output,
                      width_ == 2 ? &ranks_[1].parameters->layers[l].output : nullptr},
                     z);
        each([&](int r) {
            const DFlash2Rank& d = ranks_[at(r)];
            Rank& own            = *rank_[at(r)];
            const auto& layer    = d.parameters->layers[l];
            const cudaStream_t s = d.stream;
            ops::dynamic_conv_output_add(z[at(r)], coefficients[at(r)],
                                         layer.attention_conv.base_kernel, own.residual, s);
            Tensor convolved = own.work->alloc(DType::BF16, {H, B});
            convolve_input(r, layer.post_attention_norm, layer.mlp_conv, convolved);
            const std::int32_t F = layer.gate_up.weight.n / 2;
            Tensor gu            = own.work->alloc(DType::BF16, {2 * F, B});
            ops::linear(convolved, layer.gate_up.weight, gu, layer.gate_up.policy, *own.work, s);
            h[at(r)] = own.work->alloc(DType::BF16, {F, B});
            ops::silu_mul(gu.slice(0, 0, F), gu.slice(0, F, F), h[at(r)], s);
        });
        project_rows(h,
                     {&ranks_[0].parameters->layers[l].down,
                      width_ == 2 ? &ranks_[1].parameters->layers[l].down : nullptr},
                     z);
        each([&](int r) {
            const auto& layer = ranks_[at(r)].parameters->layers[l];
            ops::dynamic_conv_output_add(z[at(r)], coefficients[at(r)], layer.mlp_conv.base_kernel,
                                         rank_[at(r)]->residual, ranks_[at(r)].stream);
        });
    }

    // The proposals' hidden states, their candidates over every rank's vocabulary rows.
    const auto K = proposals;
    std::array<Tensor, 2> hidden, pairs;
    const auto first_rows = static_cast<std::int32_t>(ranks_[0].target->text.output_head.weight.n);
    each([&](int r) {
        const DFlash2Rank& d       = ranks_[at(r)];
        const DFlash2Parameters& p = *d.parameters;
        Rank& own                  = *rank_[at(r)];
        const cudaStream_t s       = d.stream;
        hidden[at(r)]              = own.work->alloc(DType::BF16, {H, K});
        ops::rmsnorm(own.residual.slice(1, 1, K), p.final_norm, eps, false, hidden[at(r)], s);
        const auto rows = static_cast<std::int32_t>(d.target->text.output_head.weight.n);
        Tensor logits   = own.work->alloc(DType::BF16, {rows, K});
        ops::linear(hidden[at(r)], d.target->text.output_head.weight, logits,
                    d.target->text.output_head.policy, *own.work, s);
        pairs[at(r)] = Tensor(own.pairs.data, DType::FP32, {2, kCandidates, K, width_});
        ops::vocab_top16_slots(logits, r * first_rows, r, pairs[at(r)], *own.work, s);
    });
    if (width_ == 2) {
        std::array<Tensor, 2> staging;
        each([&](int r) {
            staging[at(r)] = rank_[at(r)]->work->alloc(DType::FP32, {2, kCandidates, K, width_});
        });
        ops::allreduce_sum_f32(pairs, staging, nullptr, *tp_.execution, *tp_.events);
    }
    each([&](int r) {
        const DFlash2Rank& d       = ranks_[at(r)];
        const DFlash2Parameters& p = *d.parameters;
        Rank& own                  = *rank_[at(r)];
        const cudaStream_t s       = d.stream;
        Tensor ids(own.ids.data, DType::I32, {kCandidates, K});
        Tensor values(own.values.data, DType::FP32, {kCandidates, K});
        ops::vocab_top16_merge(pairs[at(r)], ids, values, s);
        const auto rank  = static_cast<std::int32_t>(config_.selector_rank);
        Tensor projected = own.work->alloc(DType::BF16, {rank, K});
        ops::linear(hidden[at(r)], p.hidden_projection.weight, projected,
                    p.hidden_projection.policy, *own.work, s);
        // The selector's operands at its alignment: the anchor in, the drafts out (the base
        // position, read only when sampling, is the block's first).
        CUDA_CHECK(::ninfer::memcpy_device_async(own.anchor.data, anchor[at(r)].data, 4, s));
        Tensor out_drafts(own.drafts.data, DType::I32, {K, 1});
        Tensor q(own.proposal_q.data, DType::FP32, {kCandidates, K, 1});
        ops::candidate_selector_path(ids.view({kCandidates, K, 1}),
                                     values.view({kCandidates, K, 1}), projected.view({rank, K, 1}),
                                     own.anchor, p.predecessor_codebook, p.successor_codebook,
                                     own.block_positions.slice(0, 0, 1),
                                     static_cast<const ops::SamplingConfig*>(own.sampling.data),
                                     out_drafts, q, *own.work, s);
        CUDA_CHECK(::ninfer::memcpy_device_async(drafts[at(r)].data, own.drafts.data,
                                                 std::size_t(K) * 4, s));
    });
}

} // namespace ninfer::models::inkling2::draft
