// Qwen4Exp (Qwen3.8-Flash-Next) at tensor-parallel width 2 through the public Engine, against the
// same artifact at tp 1. Needs NINFER_TEST_QWEN4_ARTIFACT (a Qwen4Exp artifact; the synthetic
// 4-layer artifact of tools/capy_flash/make_test_artifact.sh is the intended one) and two CUDA
// devices; skips (77) otherwise.
//
//   1. CausalScoring of 160 pg19 tokens: per-token log-probabilities at tp 2 within the tp 1
//      values' rounding envelope. tp 2 changes summation orders (two BF16 partial roundings of the
//      mixer output, the FP32 MoE partials summed across ranks) and, through them, the NVFP4
//      activation quantization of the W4A4 experts; on fn4 that measured mean |d| 0.011 and max
//      0.039 nats (0.0145 / 0.078 with QSA selection active), both PASS against the FP64 reference.
//      The bounds (mean 0.05, max 0.4) leave 3-5x margin
//      and still catch a lost expert partition, a wrong head shard or an unsynchronized rank, which
//      move the near-uniform fn4 log-probabilities by far more.
//   2. Greedy generation at tp 2 with CUDA Graphs (captured all-reduces through the pinned-host
//      mailbox) and eager (staged copies): identical tokens. Both transports and both execution
//      forms give identical bits, so any difference is a capture or ordering bug.
//   3. YaRN x4 (rope_yarn_factor 4, the table installed by the Engine on each rank), mapped
//      experts: tp 2 scores within the same tp 1 envelope, and different from the plain RoPE ones.
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// 160 pg19 tokens (Qwen tokenizer): crosses one 128-column
// prefill chunk.
const std::vector<ninfer::TokenId> kIds = {
    987,   17956, 1931,   539,   81249, 28442,  321,   279,   7849,   42500, 198,   30058, 5961,
    7659,  506,   1696,   1074,  2068,  83846,  9504,  4909,  318,    1919,  999,   557,   198,
    8293,  1931,  494,    5167,  76409, 1801,   2420,  539,   561,    198,   34499, 28732, 10052,
    15256, 52682, 5946,   26467, 669,   206717, 4616,  357,   177414, 80281, 15810, 13,    14200,
    262,   1897,  206717, 4616,  357,   177414, 80281, 15810, 271,    262,   7466,  198,   262,
    72614, 710,   4537,   49,    10369, 417,    5052,  50,    271,    262,   353,   43604, 6457,
    8733,  7466,  198,    262,   458,   13,     380,   13,    5574,   1736,  271,   262,   498,
    92831, 362,   60,     271,   262,   15689,  44762, 198,   262,    423,   13,    462,   13,
    423,   1966,  95630,  594,   73798, 198,    262,   13578, 52067,  36693, 5370,  1358,  262,
    14119, 11,    220,    16,    23,    24,     17,    13,    198,    262,   7466,  72614, 710,
    4537,  49,    10369,  417,   5052,  50,     13,    14200, 262,    4981,  198,   262,   423,
    1966,  95630, 13,     14200, 84424, 353,    13,    271,   16721,  20357, 13,    1358,  1,
    6614,  3667,  3667,   3667,
};

ninfer::EngineOptions base_options(const char* artifact, int tp,
                                   ninfer::RoutedExpertBackend experts) {
    ninfer::EngineOptions options;
    options.routed_experts.backend = experts;
    // Offload: a fixed small cache, so most routes of the synthetic checkpoint (near-uniform
    // routing) run on the CPU and the device/host split is exercised on both ranks.
    if (experts == ninfer::RoutedExpertBackend::Offload) {
        options.routed_experts.slots_per_layer = 24;
    }
    options.artifact_path = artifact;
    options.kv_cache      = ninfer::KvCacheStorage::BFloat16;
    options.prefill_chunk = 128;
    options.tp            = tp;
    if (tp == 2) { options.devices = {0, 1}; }
    return options;
}

std::vector<float> score(const char* artifact, int tp, const std::vector<ninfer::TokenId>& ids,
                         ninfer::RoutedExpertBackend experts, float rope_yarn_factor = 1.0F) {
    ninfer::EngineOptions options = base_options(artifact, tp, experts);
    options.purpose               = ninfer::EnginePurpose::CausalScoring;
    options.rope_yarn_factor      = rope_yarn_factor;
    options.max_context           = 256;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(256);
    ninfer::Engine engine(options);
    return engine.score_tokens(ids, 1);
}

std::vector<ninfer::TokenId> generate(const char* artifact, bool graph,
                                      const std::vector<ninfer::TokenId>& prompt,
                                      ninfer::RoutedExpertBackend experts) {
    ninfer::EngineOptions options = base_options(artifact, 2, experts);
    options.purpose               = ninfer::EnginePurpose::Generation;
    options.max_context           = 128;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.use_cuda_graph        = graph;
    options.context_cache.enabled = false;
    ninfer::Engine engine(options);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 48;
    request.execution.allow_prefix_reuse      = false;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    return engine.generate(engine.prepare_tokens(prompt, false), request).generated_token_ids;
}

// tp 2 log-probabilities within the tp 1 values' envelope (mean |d| <= 0.05, max <= 0.4).
int compare_scores(const std::string& label, const std::vector<float>& tp1,
                   const std::vector<float>& tp2, std::size_t expected) {
    if (tp1.size() != tp2.size() || tp1.size() != expected) {
        std::cerr << label << " score: " << tp1.size() << " / " << tp2.size() << " log-probs\n";
        return 1;
    }
    double mean = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < tp1.size(); ++i) {
        const double d = std::fabs(static_cast<double>(tp1[i]) - tp2[i]);
        if (!std::isfinite(d)) { worst = INFINITY; }
        mean += d;
        worst = std::max(worst, d);
    }
    mean /= static_cast<double>(tp1.size());
    std::cout << label << " score tp2 vs tp1: mean |d logprob| " << mean << ", max " << worst
              << '\n';
    if (!(mean <= 0.05) || !(worst <= 0.4)) {
        std::cerr << "FAIL: " << label
                  << " tp 2 log-probabilities outside the tp 1 envelope (mean <= 0.05, "
                     "max <= 0.4)\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::cout << "skip: NINFER_TEST_QWEN4_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two CUDA devices are required\n";
        return 77;
    }
    int failures = 0;
    try {
        const std::vector<ninfer::TokenId>& ids = kIds;
        // The offload backend's decode depends on fill timing (which experts are resident when):
        // freeze the residency after the prompt so graph and eager runs compute alike.
        // A token-changing validation knob: it needs the NINFER_QWEN4_VALIDATION opt-in.
        setenv("NINFER_QWEN4_VALIDATION", "1", 1);
        setenv("NINFER_QWEN4_EXPERT_FREEZE", "1", 1);
        for (const auto [experts, label] :
             {std::pair{ninfer::RoutedExpertBackend::Mapped, "mapped"},
              std::pair{ninfer::RoutedExpertBackend::Offload, "offload"}}) {
            const std::vector<float> tp1 = score(artifact, 1, ids, experts);
            const std::vector<float> tp2 = score(artifact, 2, ids, experts);
            failures += compare_scores(label, tp1, tp2, ids.size() - 1);
            if (experts == ninfer::RoutedExpertBackend::Mapped) {
                const std::vector<float> yarn1 = score(artifact, 1, ids, experts, 4.0F);
                const std::vector<float> yarn2 = score(artifact, 2, ids, experts, 4.0F);
                failures += compare_scores("yarn x4 mapped", yarn1, yarn2, ids.size() - 1);
                const bool changed = yarn1 != tp1 && yarn2 != tp2;
                std::cout << "yarn x4 scores differ from plain RoPE at tp 1 and tp 2: "
                          << (changed ? "yes" : "no") << '\n';
                if (!changed) {
                    std::cerr << "FAIL: rope_yarn_factor 4 left the scores unchanged\n";
                    ++failures;
                }
            }

            const std::vector<ninfer::TokenId> prompt(ids.begin(), ids.begin() + 32);
            const auto graph = generate(artifact, true, prompt, experts);
            const auto eager = generate(artifact, false, prompt, experts);
            std::cout << label << " generate tp2: " << graph.size() << " tokens, graph == eager "
                      << (graph == eager ? "yes" : "no") << '\n';
            if (graph.empty() || graph != eager) {
                std::cerr << "FAIL: " << label << " tp 2 CUDA Graph and eager decode differ\n";
                ++failures;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "OK qwen4 tp2 engine\n";
    return 0;
}
