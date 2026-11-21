// Prefix reuse of the Inkling 2 Program through the public Engine, at tp 2 on the real artifact.
//
// The Program keeps the last sequence of its one lane on the device after the request finishes,
// catalogued with two resume points: its endpoint (every executed token) and the prompt anchor
// (an image of the local rings and convolution tails taken after the prompt). The criterion is
// identity: a request that resumes must generate exactly what the same request generates from the
// root (allow_prefix_reuse = false). Ops choose their route from the block width, so the bits agree
// only when both runs split the prompt at the same boundaries: with --prefill-chunk 1 every prompt
// token runs alone in both, and a different token is a state error. With wider chunks the resumed
// run starts a block at the reused frontier and the cold one does not; equality is then reported,
// not required.
//
//   (i)   endpoint: request 2 = request 1's prompt + its generation + a tail
//   (ii)  anchor:   request 3 = request 1's prompt + a different continuation (the client's
//                   re-rendered assistant turn without the reasoning) + a tail
//   (iii) no common prefix: an unrelated request evicts the resident sequence and starts from root
//
// With --spec K the Program decodes by speculative rounds with the artifact's drafter (--spec mtp
// --draft-tokens K): the anchor image and the endpoint then carry the drafter's context too, and
// the criterion is the same (the target verifies every token).
//
// Usage: ninfer_inkling2_prefix_reuse_test --artifact X.ninfer [--tp 2] [--prefill-chunk N]
//        [--spec K].
// Without --artifact it skips (77).

#include "ninfer/engine.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;

int failures     = 0;
bool strict_bits = true;

void check(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "ok  " : "FAIL", what.c_str());
    if (!condition) { ++failures; }
}

// Same generated tokens: required at equal block boundaries, reported otherwise.
void same_tokens(const std::vector<TokenId>& a, const std::vector<TokenId>& b,
                 const std::string& what) {
    if (strict_bits) {
        check(a == b, what);
        return;
    }
    std::size_t first = 0;
    while (first < a.size() && first < b.size() && a[first] == b[first]) { ++first; }
    std::printf("info %s: %s (first difference at %zu of %zu/%zu)\n", what.c_str(),
                a == b ? "identical" : "different", first, a.size(), b.size());
}

std::vector<TokenId> chat(const Engine& engine, std::string_view text) {
    PromptInput input;
    input.messages.push_back(
        ChatMessage{.role = ChatRole::User, .parts = {MessagePart{.text = std::string(text)}}});
    const PreparedPrompt prepared = engine.prepare(std::move(input));
    const auto ids                = prepared.token_ids();
    return {ids.begin(), ids.end()};
}

std::vector<TokenId> join(std::vector<TokenId> a, const std::vector<TokenId>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

struct Turn {
    std::vector<TokenId> generated;
    std::uint32_t reused   = 0;
    PrefixReusePath path   = PrefixReusePath::Root;
    double prefill_seconds = 0.0;
};

Turn run(Engine& engine, const std::vector<TokenId>& prompt, bool allow_reuse, const char* label) {
    RequestOptions options;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.requested_output_tokens = 64;
    options.execution.allow_prefix_reuse      = allow_reuse;
    GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), std::move(options), nullptr);
    Turn out{.generated       = std::move(result.generated_token_ids),
             .reused          = result.reused_prompt_tokens,
             .path            = result.prefix_reuse_path,
             .prefill_seconds = result.timings.prefill_seconds};
    std::printf("%-12s prompt %5zu | reused %5u | prefilled %5zu | path %d | generated %zu | "
                "prefill %.3f s\n",
                label, prompt.size(), out.reused, prompt.size() - out.reused,
                static_cast<int>(out.path), out.generated.size(), out.prefill_seconds);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string artifact;
    int tp                      = 2;
    std::uint32_t prefill_chunk = 1024;
    std::uint32_t draft_tokens  = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string_view option = argv[i];
        if (option == "--artifact") {
            artifact = argv[i + 1];
        } else if (option == "--tp") {
            tp = std::atoi(argv[i + 1]);
        } else if (option == "--prefill-chunk") {
            prefill_chunk = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
        } else if (option == "--spec") {
            draft_tokens = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
        }
    }
    if (artifact.empty()) {
        std::printf("skip: no --artifact\n");
        return 77;
    }
    EngineOptions options;
    options.artifact_path   = artifact;
    options.tp              = tp;
    options.devices         = tp == 2 ? std::vector<int>{0, 1} : std::vector<int>{0};
    options.max_context     = 4096;
    options.kv_capacity     = KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency = 1;
    options.prefill_chunk   = prefill_chunk;
    if (draft_tokens > 0) {
        options.speculative.backend      = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = draft_tokens;
    }
    strict_bits = prefill_chunk == 1;
    Engine engine(options);

    const std::vector<TokenId> p1 = chat(engine, "Explain in two sentences why the sky is blue.");
    const std::vector<TokenId> tail =
        engine.tokenize_text(" Now say the same thing for a five year old.");
    const std::vector<TokenId> diverge = engine.tokenize_text("Rayleigh scattering of sunlight.");
    const std::vector<TokenId> p4 = chat(engine, "List three prime numbers larger than fifty.");

    const Turn r1 = run(engine, p1, true, "r1 root");
    check(r1.path == PrefixReusePath::Root && r1.reused == 0, "r1 starts from the root");

    // (i) endpoint
    const std::vector<TokenId> r2_prompt = join(join(p1, r1.generated), tail);
    const Turn r2                        = run(engine, r2_prompt, true, "r2 resumed");
    const auto expect2 = static_cast<std::uint32_t>(p1.size() + r1.generated.size() - 1);
    check(r2.path == PrefixReusePath::PrivateEndpoint && r2.reused == expect2,
          "r2 resumes at r1's endpoint (" + std::to_string(expect2) + " tokens)");
    const Turn r2_cold = run(engine, r2_prompt, false, "r2 cold");
    check(r2_cold.path == PrefixReusePath::Root, "r2 cold starts from the root (eviction)");
    same_tokens(r2.generated, r2_cold.generated, "(i) endpoint resume generates the cold tokens");

    // (ii) prompt anchor
    const Turn r1b = run(engine, p1, true, "r1 again");
    same_tokens(r1b.generated, r1.generated, "root generation is reproducible");
    const std::vector<TokenId> r3_prompt = join(join(p1, diverge), tail);
    const Turn r3                        = run(engine, r3_prompt, true, "r3 resumed");
    check(r3.path == PrefixReusePath::PrivateLongAnchor &&
              r3.reused == static_cast<std::uint32_t>(p1.size()),
          "r3 resumes at r1's prompt anchor (" + std::to_string(p1.size()) + " tokens)");

    // (iii) no common prefix: evicts r3's resident sequence
    const Turn r4 = run(engine, p4, true, "r4 root");
    check(r4.path == PrefixReusePath::Root && r4.reused == 0, "(iii) unrelated prompt: root");
    const Turn r4_cold = run(engine, p4, false, "r4 cold");
    same_tokens(r4.generated, r4_cold.generated,
                "(iii) root after eviction generates the cold tokens");

    const Turn r3_cold = run(engine, r3_prompt, false, "r3 cold");
    same_tokens(r3.generated, r3_cold.generated, "(ii) anchor resume generates the cold tokens");

    std::printf("%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
