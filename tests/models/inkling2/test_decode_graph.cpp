// Inkling 2 decode from the captured CUDA Graph against the same decode run eagerly: two Program
// executions over one loaded model, one with CUDA Graphs and one without, run the same prompt and
// the same sampled decode steps. After the prompt and after every step the visit state of every
// rank and the soft-capped FP32 logits of the step must be the same bytes, and so the tokens.
// Usage: --artifact PATH [--tp 1|2] [--steps N] [--loopcd hidden|logits|adaptive [--loopcd-omega W]
// [--loopcd-pass B]] (the model loaded with that loop contrast, W as in ninfer-serve); without an
// artifact it skips (77).

#include "artifact/reader.h"
#include "core/device.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/load.h"
#include "models/inkling2/program/execution.h"
#include "models/inkling2/program/program_impl.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace ink = ninfer::models::inkling2;

// A context the warmup covers whole (prefill_chunk >= max_context), so every state byte the
// comparison reads was written by the same calls in both executions.
constexpr std::uint32_t kContext = 256;
constexpr std::uint32_t kPrompt  = 40;

// The first offset where two snapshots differ, or nothing.
std::optional<std::size_t> first_difference(const std::vector<std::uint8_t>& a,
                                            const std::vector<std::uint8_t>& b) {
    if (a.size() != b.size()) { return std::min(a.size(), b.size()); }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) { return i; }
    }
    return std::nullopt;
}

int run(const std::filesystem::path& path, int tp, std::uint32_t steps,
        const LoopContrastOptions& contrast) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < tp) {
        std::cout << "SKIP: needs " << tp << " CUDA devices\n";
        return 77;
    }
    artifact::Reader reader(path);
    auto plan = ink::plan_load(reader, models::LoadOptions{.tp = tp, .loop_contrast = contrast});
    std::vector<int> ids;
    for (int device = 0; device < tp; ++device) { ids.push_back(device); }
    ExecutionContext execution(ids);
    auto model = ink::materialize_model(std::move(plan), execution);
    const ink::execution::Parameters parameters(*model, 0);
    std::unique_ptr<const ink::execution::Parameters> peer;
    if (tp == 2) { peer = std::make_unique<const ink::execution::Parameters>(*model, 1); }

    EngineOptions options;
    options.tp                    = tp;
    options.max_context           = kContext;
    options.max_concurrency       = 1;
    options.prefill_chunk         = kContext;
    options.context_cache.enabled = false;
    options.use_cuda_graph        = true;
    const auto graph_layout = ink::detail::plan_sequence_layout(parameters, peer.get(), options);
    options.use_cuda_graph  = false;
    const auto eager_layout = ink::detail::plan_sequence_layout(parameters, peer.get(), options);
    auto graph =
        ink::detail::make_program_execution(parameters, peer.get(), graph_layout, execution);
    auto eager =
        ink::detail::make_program_execution(parameters, peer.get(), eager_layout, execution);

    // Sampled, so both paths bring the soft-capped FP32 logits to the Host.
    ResolvedSamplingParameters sampling;
    sampling.temperature = 0.8F;
    sampling.seed        = 7;
    std::vector<TokenId> prompt(kPrompt);
    for (std::uint32_t i = 0; i < kPrompt; ++i) { prompt[i] = TokenId((i * 37U + 11U) % 997U); }

    int failures     = 0;
    const auto check = [&](const char* what, std::uint32_t step) {
        const auto difference = first_difference(graph->snapshot(0), eager->snapshot(0));
        if (difference) {
            std::cerr << what << " " << step << ": graph and eager differ at byte " << *difference
                      << '\n';
            ++failures;
        }
    };
    graph->reset(0);
    eager->reset(0);
    const auto first_graph = graph->prefill(0, prompt, 0, &sampling);
    const auto first_eager = eager->prefill(0, prompt, 0, &sampling);
    check("prompt", 0);
    if (!first_graph || !first_eager || *first_graph != *first_eager) {
        std::cerr << "prompt: the first tokens differ\n";
        return 1;
    }
    std::vector<TokenId> generated{*first_graph};
    for (std::uint32_t step = 0; step < steps && kPrompt + step + 1U < kContext; ++step) {
        const std::vector<TokenId> token{generated.back()};
        const std::uint32_t position = kPrompt + step;
        const TokenId from_graph     = graph->decode(0, token, position, sampling, generated);
        const TokenId from_eager     = eager->decode(0, token, position, sampling, generated);
        check("decode step", step);
        if (from_graph != from_eager) {
            std::cerr << "decode step " << step << ": tokens " << from_graph << " (graph) and "
                      << from_eager << " (eager)\n";
            ++failures;
        }
        if (failures != 0) { break; }
        generated.push_back(from_graph);
    }
    std::printf("tp %d: %zu decode steps compared (state %zu bytes per snapshot)\n", tp,
                generated.size() - 1U, graph->snapshot(0).size());
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        int tp              = 2;
        std::uint32_t steps = 24;
        LoopContrastOptions contrast;
        std::optional<float> omega;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--artifact" && i + 1 < argc) {
                path = argv[++i];
            } else if (arg == "--tp" && i + 1 < argc) {
                tp = std::stoi(argv[++i]);
            } else if (arg == "--steps" && i + 1 < argc) {
                steps = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--loopcd" && i + 1 < argc) {
                const std::string mode(argv[++i]);
                contrast.mode     = mode == "hidden" ? LoopContrast::Hidden : LoopContrast::Logits;
                contrast.adaptive = mode == "adaptive";
            } else if (arg == "--loopcd-omega" && i + 1 < argc) {
                omega = std::stof(argv[++i]);
            } else if (arg == "--loopcd-pass" && i + 1 < argc) {
                contrast.reference_pass = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } else {
                std::cerr << "usage: --artifact PATH [--tp 1|2] [--steps N] [--loopcd MODE "
                             "[--loopcd-omega W] [--loopcd-pass B]]\n";
                return 2;
            }
        }
        if (omega) {
            contrast.strength = *omega;
        } else if (contrast.adaptive) {
            contrast.strength = kDefaultAdaptiveLoopContrastStrength;
        } else if (contrast.mode != LoopContrast::Off) {
            throw std::invalid_argument("--loopcd hidden and logits need --loopcd-omega");
        }
        if (path.empty()) {
            std::cout << "SKIP: supply --artifact (an Inkling 2 artifact)\n";
            return 77;
        }
        const int failures = run(path, tp, steps, contrast);
        if (failures == 77) { return 77; }
        std::cout << (failures ? "FAIL" : "OK") << " inkling2 decode graph against eager, tp " << tp
                  << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
