#include "models/inkling2/draft/draft.h"

#include "core/device.h"
#include "core/device_scope.h"
#include "core/weight_view.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/tp2/linear.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::draft {
namespace {

[[nodiscard]] std::size_t at(int rank) { return static_cast<std::size_t>(rank); }

constexpr std::int32_t kHeadDim = 128;

// The workspace of one materialization of up to `max_width` columns followed by one proposal.
[[nodiscard]] std::size_t workspace_for(const DraftConfig& config, const DraftConfig& rank,
                                        std::int32_t rows, std::int32_t qkv, std::int32_t F2,
                                        std::int32_t max_width) {
    const std::size_t H = config.hidden_size, B = config.block_size, W = std::size_t(max_width);
    const std::size_t layers = config.num_layers, G = B - 1;
    return H * W * (layers * 2 + 2) +                                          // mixed contexts
           layers * 4 * std::size_t(kHeadDim) * rank.num_key_value_heads * W + // context K|V
           layers * B * (std::size_t(qkv) + F2 + F2 / 2 + 6 * H) * 2 +         // layers
           ops::block_attention_workspace_bytes(static_cast<std::int32_t>(rank.num_attention_heads),
                                                static_cast<std::int32_t>(B)) +
           std::size_t(rows) * G * 2 + 3 * std::size_t(config.markov_rank) * G * 4 +
           ops::recurrent_head_step_workspace_bytes() + (std::size_t(4) << 20);
}

} // namespace

// Rank r's device state and workspace.
struct Drafter::Rank {
    std::vector<std::unique_ptr<DeviceBuffer>> memory;
    std::vector<ops::DraftKVCacheView> caches;
    Tensor pending; // BF16 [H, max_width, 4]
    execution::FeatureSink sink;
    Tensor f_rnn;           // BF16 [H]
    Tensor input;           // BF16 [H, B]: anchor column, then the mask embedding
    Tensor residual;        // BF16 [H, B]
    Tensor positions;       // I32 [max_width]
    Tensor block_positions; // I32 [B]
    Tensor count;           // I32 [1]: columns of the pending features to consume
    Tensor frontier;        // I32 [1]: the eager append's frontier
    std::array<Tensor, 2> state;
    Tensor choice; // FP32 [4]
    // The anchor image: the local layers' rings (in layer order) and f_rnn.
    std::vector<ops::DraftKVCacheView> image;
    Tensor image_f_rnn;
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

Drafter::Drafter(const DraftConfig& config, const TextConfig& text, std::array<DraftRank, 2> ranks,
                 int width, execution::ForwardCollectives collectives, std::int32_t max_width,
                 std::uint32_t max_context, bool with_image)
    : config_(config), text_(text), ranks_(ranks), width_(width), tp_(collectives),
      max_width_(max_width), max_context_(max_context) {
    if ((width != 1 && width != 2) || max_width < config.block_size || max_context == 0 ||
        config.head_dim != kHeadDim || config.markov_rank != 256 ||
        config.feature_visits.size() + 1 != kFeatures) {
        throw std::invalid_argument("Inkling 2 drafter: unsupported geometry or bounds");
    }
    const auto H = static_cast<std::int32_t>(config.hidden_size);
    const auto B = static_cast<std::int32_t>(config.block_size);
    for (int r = 0; r < width_; ++r) {
        const DraftRank& d = ranks_[at(r)];
        if (d.config == nullptr || d.parameters == nullptr || d.target == nullptr ||
            d.parameters->layers.size() != config.num_layers) {
            throw std::invalid_argument("Inkling 2 drafter: incomplete rank " + std::to_string(r));
        }
        const ScopedCurrentDevice scope(d.device);
        rank_[at(r)]      = std::make_unique<Rank>();
        Rank& own         = *rank_[at(r)];
        std::size_t total = 0;
        const auto kv     = static_cast<std::int32_t>(d.config->num_key_value_heads);
        for (std::uint32_t l = 0; l < config.num_layers; ++l) {
            const auto capacity =
                static_cast<std::int32_t>(config.local[l] ? config.sliding_window : max_context);
            ops::DraftKVCacheView cache;
            cache.k        = own.buffer(DType::BF16, {kHeadDim, kv, capacity}, total);
            cache.v        = own.buffer(DType::BF16, {kHeadDim, kv, capacity}, total);
            cache.capacity = capacity;
            own.caches.push_back(cache);
        }
        own.pending         = own.buffer(DType::BF16, {H, max_width, 4}, total);
        own.f_rnn           = own.buffer(DType::BF16, {H}, total);
        own.input           = own.buffer(DType::BF16, {H, B}, total);
        own.residual        = own.buffer(DType::BF16, {H, B}, total);
        own.positions       = own.buffer(DType::I32, {max_width}, total);
        own.block_positions = own.buffer(DType::I32, {B}, total);
        own.count           = own.buffer(DType::I32, {1}, total);
        own.frontier        = own.buffer(DType::I32, {1}, total);
        for (auto& state : own.state) {
            state = own.buffer(DType::BF16, {static_cast<std::int32_t>(config.markov_rank)}, total);
        }
        own.choice = own.buffer(DType::FP32, {4}, total);
        if (with_image) {
            for (std::uint32_t l = 0; l < config.num_layers; ++l) {
                if (!config.local[l]) { continue; }
                const auto capacity = static_cast<std::int32_t>(config.sliding_window);
                own.image.push_back({own.buffer(DType::BF16, {kHeadDim, kv, capacity}, total),
                                     own.buffer(DType::BF16, {kHeadDim, kv, capacity}, total),
                                     capacity});
            }
            own.image_f_rnn = own.buffer(DType::BF16, {H}, total);
        }
        const std::int32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(own.count.data, &zero, 4, cudaMemcpyHostToDevice));
        for (std::int32_t c = 1; c < B; ++c) {
            CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(own.input.data) + std::size_t(c) * H * 2,
                                  d.parameters->mask_embedding.data, std::size_t(H) * 2,
                                  cudaMemcpyDeviceToDevice));
        }
        const Model& model = d.target->model;
        own.embedding      = native_weight(
            model.weight(model.weights().text.token_embedding, d.target->device).view);
        const auto rows  = static_cast<std::int32_t>(d.parameters->folded_bias.numel());
        const auto qkv   = d.parameters->layers.front().attention_input.weight.n;
        const auto F2    = d.parameters->layers.front().gate_up.weight.n;
        workspace_bytes_ = workspace_for(config, *d.config, rows, qkv, F2, max_width);
        own.work         = std::make_unique<WorkspaceArena>(workspace_bytes_);
        state_bytes_     = total;
    }
}

Drafter::~Drafter() = default;

std::size_t Drafter::planned_bytes(const DraftConfig& config, const DraftConfig& rank,
                                   std::int32_t rows, std::int32_t attention_input_rows,
                                   std::int32_t max_width, std::uint32_t max_context, bool image) {
    const std::size_t H = config.hidden_size, B = config.block_size, D = kHeadDim;
    const std::size_t kv = rank.num_key_value_heads;
    std::size_t state    = 0;
    for (std::uint32_t l = 0; l < config.num_layers; ++l) {
        const std::size_t capacity = config.local[l] ? config.sliding_window : max_context;
        state += 2 * D * kv * capacity * 2 * (image && config.local[l] ? 2 : 1);
    }
    state += H * std::size_t(max_width) * 4 * 2 + H * 2 * (image ? 2 : 1) + 2 * H * B * 2 +
             std::size_t(max_width) * 4 + B * 4 + 8 + 2 * std::size_t(config.markov_rank) * 2 + 16;
    return state + workspace_for(config, rank, rows, attention_input_rows,
                                 2 * static_cast<std::int32_t>(rank.intermediate_size), max_width);
}

template <class Body>
void Drafter::each(Body&& body) {
    const ScopedCurrentDevice restore;
    for (int r = 0; r < width_; ++r) {
        ScopedCurrentDevice::select(ranks_[at(r)].device);
        body(r);
    }
}

const execution::FeatureSink* Drafter::sink(int rank, std::int32_t width) {
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 drafter: sink wider than planned");
    }
    Rank& own = *rank_[at(rank)];
    own.sink  = {config_.feature_visits, own.pending.slice(1, 0, width)};
    return &own.sink;
}

void Drafter::materialize(const std::array<Tensor, 2>& frontier, std::int32_t width) {
    const auto H = static_cast<std::int32_t>(config_.hidden_size);
    const auto L = static_cast<std::int32_t>(config_.num_layers);
    each([&](int r) {
        const DraftRank& d   = ranks_[at(r)];
        const Parameters& p  = *d.parameters;
        Rank& own            = *rank_[at(r)];
        const cudaStream_t s = d.stream;
        Tensor positions(own.positions.data, DType::I32, {width});
        ops::consecutive_positions(frontier[at(r)], &own.count, positions, s);
        const Tensor features = own.pending.slice(1, 0, width);
        Tensor mixed          = own.work->alloc(DType::BF16, {H, width, L});
        std::array<const Tensor*, 8> norms{};
        for (std::int32_t l = 0; l < L; ++l) { norms[std::size_t(l)] = &p.layers[l].context_norm; }
        norms[std::size_t(L)] = &p.markov_context_norm;
        ops::feature_mix_rmsnorm(features, p.fuse_weight, norms, L + 1, config_.rms_norm_eps,
                                 &own.count, mixed, own.f_rnn, s);
        std::array<Tensor, 8> raw;
        std::array<const Tensor*, 8> raw_views{}, key_norms{};
        std::array<ops::DraftKVCacheView, 8> caches{};
        for (std::int32_t l = 0; l < L; ++l) {
            const auto& layer   = p.layers[l];
            raw[std::size_t(l)] = own.work->alloc(DType::BF16, {layer.context.weight.n, width});
            ops::linear(mixed.slice(2, l, 1).view({H, width}), layer.context.weight,
                        raw[std::size_t(l)], layer.context.policy, *own.work, s);
            raw_views[std::size_t(l)] = &raw[std::size_t(l)];
            key_norms[std::size_t(l)] = &layer.context_key_norm;
            caches[std::size_t(l)]    = own.caches[std::size_t(l)];
        }
        ops::context_kv_store(raw_views, key_norms, caches, L, positions, own.count,
                              config_.rope_theta, config_.rms_norm_eps, s);
    });
}

void Drafter::append(std::int32_t first, std::int32_t width) {
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 drafter: append wider than planned");
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
    each([&](int r) {
        CUDA_CHECK(cudaStreamSynchronize(ranks_[at(r)].stream));
        const std::int32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(rank_[at(r)]->count.data, &zero, 4, cudaMemcpyHostToDevice));
    });
}

void Drafter::clear_pending() {
    each([&](int r) {
        CUDA_CHECK(cudaStreamSynchronize(ranks_[at(r)].stream));
        const std::int32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(rank_[at(r)]->count.data, &zero, 4, cudaMemcpyHostToDevice));
    });
}

void Drafter::catch_up(std::int32_t frontier, std::int32_t width) {
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 drafter: pending width out of range");
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

void Drafter::follow(const std::array<Tensor, 2>& frontier, std::int32_t width) {
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 drafter: pending width out of range");
    }
    each([&](int r) { rank_[at(r)]->work->reset(); });
    materialize(frontier, width);
}

void Drafter::save_image() { copy_image(true); }

void Drafter::restore_image() {
    copy_image(false);
    clear_pending();
}

void Drafter::copy_image(bool save) {
    each([&](int r) {
        Rank& own = *rank_[at(r)];
        if (own.image_f_rnn.data == nullptr) {
            throw std::logic_error("Inkling 2 drafter has no anchor image");
        }
        const cudaStream_t s = ranks_[at(r)].stream;
        const auto copy      = [&](const Tensor& live, const Tensor& image) {
            CUDA_CHECK(::ninfer::memcpy_device_async(
                save ? image.data : live.data, save ? live.data : image.data, live.bytes(), s));
        };
        std::size_t image = 0;
        for (std::uint32_t l = 0; l < config_.num_layers; ++l) {
            if (!config_.local[l]) { continue; }
            copy(own.caches[l].k, own.image[image].k);
            copy(own.caches[l].v, own.image[image].v);
            ++image;
        }
        copy(own.f_rnn, own.image_f_rnn);
        CUDA_CHECK(cudaStreamSynchronize(s));
    });
}

void Drafter::commit(const std::array<Tensor, 2>& count) {
    each([&](int r) {
        CUDA_CHECK(::ninfer::memcpy_device_async(rank_[at(r)]->count.data, count[at(r)].data, 4,
                                                 ranks_[at(r)].stream));
    });
}

void Drafter::project_rows(const std::array<Tensor, 2>& x,
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

void Drafter::propose(const std::array<Tensor, 2>& anchor, const std::array<Tensor, 2>& frontier,
                      const std::array<Tensor, 2>& drafts, std::int32_t width,
                      std::int32_t proposals) {
    const auto H    = static_cast<std::int32_t>(config_.hidden_size);
    const auto B    = static_cast<std::int32_t>(config_.block_size);
    const auto G    = B - 1;
    const float eps = config_.rms_norm_eps;
    each([&](int r) { rank_[at(r)]->work->reset(); });
    if (width <= 0 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 drafter: pending width out of range");
    }
    materialize(frontier, width);

    std::array<Tensor, 2> x, m, a, h, hidden;
    each([&](int r) {
        const DraftRank& d   = ranks_[at(r)];
        Rank& own            = *rank_[at(r)];
        const cudaStream_t s = d.stream;
        ops::consecutive_positions(frontier[at(r)], nullptr, own.block_positions, s);
        Tensor embedded = own.work->alloc(DType::BF16, {H, 1});
        ops::embedding(anchor[at(r)], own.embedding, embedded, s);
        Tensor column = own.input.slice(1, 0, 1);
        ops::rmsnorm(embedded, d.target->text.embedding_norm, eps, false, column, s);
        CUDA_CHECK(
            ::ninfer::memcpy_device_async(own.residual.data, own.input.data, own.input.bytes(), s));
        x[at(r)] = own.work->alloc(DType::BF16, {H, B});
        ops::rmsnorm(own.residual, d.parameters->layers.front().input_norm, eps, false, x[at(r)],
                     s);
        CUDA_CHECK(cudaMemsetAsync(own.state[0].data, 0, own.state[0].bytes(), s));
    });
    for (std::uint32_t l = 0; l < config_.num_layers; ++l) {
        std::array<Tensor, 2> o;
        each([&](int r) {
            const DraftRank& d   = ranks_[at(r)];
            Rank& own            = *rank_[at(r)];
            const auto& layer    = d.parameters->layers[l];
            const cudaStream_t s = d.stream;
            const auto nq        = static_cast<std::int32_t>(d.config->num_attention_heads);
            const auto nkv       = static_cast<std::int32_t>(d.config->num_key_value_heads);
            if (l > 0) {
                ops::add_rmsnorm(own.residual, m[at(r)], layer.input_norm, eps, x[at(r)], s);
            }
            Tensor qkv = own.work->alloc(DType::BF16, {layer.attention_input.weight.n, B});
            ops::linear(x[at(r)], layer.attention_input.weight, qkv, layer.attention_input.policy,
                        *own.work, s);
            ops::head_rmsnorm_rope(qkv, nq, nkv, layer.query_norm, layer.key_norm,
                                   own.block_positions, config_.rope_theta, eps, s);
            o[at(r)] = own.work->alloc(DType::BF16, {nq * kHeadDim, B});
            ops::block_attention(
                qkv, nq, nkv, own.caches[l], frontier[at(r)],
                config_.local[l] ? static_cast<std::int32_t>(config_.sliding_window) : 0,
                static_cast<float>(1.0 / std::sqrt(double(kHeadDim))), *own.work, o[at(r)], s);
            a[at(r)] = own.work->alloc(DType::BF16, {H, B});
        });
        project_rows(o,
                     {&ranks_[0].parameters->layers[l].output,
                      width_ == 2 ? &ranks_[1].parameters->layers[l].output : nullptr},
                     a);
        each([&](int r) {
            const DraftRank& d   = ranks_[at(r)];
            Rank& own            = *rank_[at(r)];
            const auto& layer    = d.parameters->layers[l];
            const cudaStream_t s = d.stream;
            ops::add_rmsnorm(own.residual, a[at(r)], layer.post_attention_norm, eps, x[at(r)], s);
            const std::int32_t F = layer.gate_up.weight.n / 2;
            Tensor gu            = own.work->alloc(DType::BF16, {2 * F, B});
            ops::linear(x[at(r)], layer.gate_up.weight, gu, layer.gate_up.policy, *own.work, s);
            h[at(r)] = own.work->alloc(DType::BF16, {F, B});
            ops::silu_mul(gu.slice(0, 0, F), gu.slice(0, F, F), h[at(r)], s);
            m[at(r)] = own.work->alloc(DType::BF16, {H, B});
        });
        project_rows(h,
                     {&ranks_[0].parameters->layers[l].down,
                      width_ == 2 ? &ranks_[1].parameters->layers[l].down : nullptr},
                     m);
    }
    std::array<ops::RecurrentHeadOperands, 2> head;
    const auto first_rows = static_cast<std::int32_t>(ranks_[0].parameters->folded_bias.numel());
    each([&](int r) {
        const DraftRank& d   = ranks_[at(r)];
        const Parameters& p  = *d.parameters;
        Rank& own            = *rank_[at(r)];
        const cudaStream_t s = d.stream;
        hidden[at(r)]        = own.work->alloc(DType::BF16, {H, B});
        ops::add_rmsnorm(own.residual, m[at(r)], p.final_norm, eps, hidden[at(r)], s);
        const Tensor proposals = hidden[at(r)].slice(1, 1, G);
        const auto rows        = static_cast<std::int32_t>(p.folded_bias.numel());
        Tensor base            = own.work->alloc(DType::BF16, {rows, G});
        ops::linear(proposals, d.target->text.output_head.weight, base,
                    d.target->text.output_head.policy, *own.work, s);
        Tensor context =
            own.work->alloc(DType::FP32, {3 * static_cast<std::int32_t>(config_.markov_rank), G});
        ops::recurrent_head_context(p.recurrence, p.recurrence_bias, proposals, own.f_rnn, context,
                                    s);
        ops::RecurrentHeadOperands& operands = head[at(r)];
        operands.weight                      = p.recurrence;
        operands.embedding                   = p.markov_token_embedding;
        operands.folded                      = p.folded_output;
        operands.folded_bias                 = p.folded_bias;
        operands.base                        = base;
        operands.context                     = context;
        operands.cap                         = text_.final_logit_softcap;
        operands.first                       = r * first_rows;
        operands.rank                        = r;
        operands.ranks                       = width_;
    });
    if (proposals < 1 || proposals > G) {
        throw std::invalid_argument("Inkling 2 drafter: proposals out of range");
    }
    for (std::int32_t step = 0; step < proposals; ++step) {
        each([&](int r) {
            Rank& own         = *rank_[at(r)];
            Tensor out_drafts = drafts[at(r)];
            ops::recurrent_head_step(head[at(r)], step, anchor[at(r)], own.state[step % 2],
                                     own.state[(step + 1) % 2], out_drafts, own.choice, *own.work,
                                     ranks_[at(r)].stream);
        });
        if (width_ == 2) {
            std::array<Tensor, 2> choice{rank_[0]->choice, rank_[1]->choice}, staging;
            each([&](int r) { staging[at(r)] = rank_[at(r)]->work->alloc(DType::FP32, {4}); });
            ops::allreduce_sum_f32(choice, staging, nullptr, *tp_.execution, *tp_.events);
        }
    }
    each([&](int r) {
        Tensor out_drafts = drafts[at(r)].slice(0, 0, proposals);
        ops::recurrent_head_select(rank_[at(r)]->choice, out_drafts, ranks_[at(r)].stream);
    });
}

} // namespace ninfer::models::inkling2::draft
