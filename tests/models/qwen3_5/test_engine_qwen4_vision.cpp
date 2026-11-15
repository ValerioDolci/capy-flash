// Qwen4Exp (Qwen3.8-Flash-Next) with Vision through the public Engine, on a synthetic Qwen4Exp
// artifact that carries a Vision component (NINFER_TEST_QWEN4_VISION_ARTIFACT: 4 Text layers and
// the MTP layer of tools/capy_flash/make_test_artifact.sh with a Vision tower, converted with
// `--components text,vision,mtp`). Skips (77) without it or below two CUDA devices.
//
// Served as `--profile flash-next --vision`: tp 2, MTP (K=3), offload experts (24 slots per layer,
// residency fixed at initialization: NINFER_QWEN4_EXPERT_FREEZE=all, so placement does not depend
// on what an Engine served before; the GPU W4A4 and CPU W4A8 experts are not bit-identical). The
// image is a 512x512 gradient PPM (256 merged tokens at --max-vision-tokens 256) and the prefill
// chunk is 128 columns, so one Vision item spans three prompt chunks (the merged embeddings are
// scattered from an item offset on its later chunks) and its MTP alignment crosses visual columns.
//
//   1. text invariance: a text prompt answers with the same token ids with and without Vision
//      (the Vision Engine changes nothing for text).
//   2. exactness: an image prompt under MTP equals its no-draft control
//   (NINFER_QWEN4_MTP_NO_DRAFTS:
//      the same verify / rollback kernels with extent 0), so the three-axis RoPE positions of the
//      prompt and the rope delta of the decode are consistent between prefill, verification and
//      rollback; the encoder ran and the prompt counts the item's merged tokens.
//   3. copy parity: with the tower on device 1 (the merged embeddings copied from rank 1 to rank 0)
//      the image prompt gives the same token ids as with the tower on device 0. Both ranks run the
//      same kernels on the same bytes, so a late, missing or misplaced copy changes the answer.
//   4. the image reaches the model: a different image changes the answer.
//   5. prefix reuse: the second turn of the image conversation resumes the retained prefix (the
//      text-only suffix continues the multimodal prefix with its rope delta) and equals its
//      no-draft control.
//   6. one device (tp 1): the image prompt under MTP equals its no-draft control.
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext      = 2048;
constexpr std::uint32_t kMaxVisionTokens = 256;
constexpr int kImageEdge                 = 512;
constexpr std::uint32_t kOutput          = 24;

class ScopedEnv {
public:
    ScopedEnv(const char* name, const std::string& value) : name_(name) {
        ::setenv(name, value.c_str(), 1);
    }

    ~ScopedEnv() { ::unsetenv(name_); }

    ScopedEnv(const ScopedEnv&)            = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* name_;
};

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

ninfer::EngineOptions options(const char* artifact, int tp, bool vision,
                              std::optional<int> vision_device) {
    ninfer::EngineOptions o;
    o.artifact_path                  = artifact;
    o.purpose                        = ninfer::EnginePurpose::Generation;
    o.kv_cache                       = ninfer::KvCacheStorage::BFloat16;
    o.prefill_chunk                  = 128;
    o.max_context                    = kMaxContext;
    o.kv_capacity                    = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    o.max_concurrency                = 1;
    o.speculative.backend            = ninfer::SpeculativeBackend::Mtp;
    o.speculative.draft_tokens       = 3;
    o.routed_experts.backend         = ninfer::RoutedExpertBackend::Offload;
    o.routed_experts.slots_per_layer = 24;
    o.context_cache.enabled          = true;
    o.tp                             = tp;
    if (tp == 2) {
        o.devices                              = {0, 1};
        o.context_cache.host_state_slots       = 0; // rank 1 has no Host tier
        o.context_cache.host_kv_capacity_bytes = 0;
    }
    if (vision) {
        o.enable_vision     = true;
        o.vision_device     = vision_device;
        o.max_vision_tokens = kMaxVisionTokens;
    }
    return o;
}

// A smooth color gradient (the encoder sees structure, not one flat patch).
std::vector<std::uint8_t> gradient_ppm(int phase) {
    const std::string header =
        "P6\n" + std::to_string(kImageEdge) + ' ' + std::to_string(kImageEdge) + "\n255\n";
    std::vector<std::uint8_t> ppm(header.begin(), header.end());
    ppm.reserve(ppm.size() + static_cast<std::size_t>(kImageEdge) * kImageEdge * 3);
    for (int y = 0; y < kImageEdge; ++y) {
        for (int x = 0; x < kImageEdge; ++x) {
            ppm.push_back(static_cast<std::uint8_t>((x + phase) % 256));
            ppm.push_back(static_cast<std::uint8_t>((y * 2 + phase) % 256));
            ppm.push_back(static_cast<std::uint8_t>((x + y + 3 * phase) % 256));
        }
    }
    return ppm;
}

ninfer::MessagePart text_part(std::string text) {
    return ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}};
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage m;
    m.role = role;
    m.parts.push_back(text_part(std::move(text)));
    return m;
}

ninfer::PromptInput text_prompt() {
    ninfer::PromptInput input;
    input.messages.push_back(message(ninfer::ChatRole::User,
                                     "Describe the harbor of Genoa at dawn in three sentences."));
    input.options.enable_thinking = false;
    return input;
}

ninfer::PromptInput image_prompt(int phase) {
    ninfer::MessagePart media;
    media.kind              = ninfer::MessagePartKind::Media;
    media.media.kind        = ninfer::MediaKind::Image;
    media.media.bytes       = gradient_ppm(phase);
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = "gradient.ppm";
    ninfer::ChatMessage m;
    m.role = ninfer::ChatRole::User;
    m.parts.push_back(std::move(media));
    m.parts.push_back(text_part("What does this image show? Answer briefly."));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(m));
    input.options.enable_thinking   = false;
    input.context_cache.session_key = "qwen4-vision";
    input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    return input;
}

ninfer::RequestOptions greedy(bool reuse = false) {
    ninfer::RequestOptions r;
    r.execution.requested_output_tokens = kOutput;
    r.execution.allow_prefix_reuse      = reuse;
    r.execution.sampling.temperature    = 0.0F;
    r.stop.include_model_defaults       = false;
    return r;
}

ninfer::GenerationResult generate(ninfer::Engine& engine, ninfer::PromptInput input,
                                  bool reuse = false) {
    return engine.generate(engine.prepare(std::move(input)), greedy(reuse));
}

void print(const std::string& label, const ninfer::GenerationResult& r) {
    std::cout << "  " << label << ": " << r.generated_token_ids.size() << " tokens, vision "
              << r.timings.vision_seconds << " s, reused " << r.reused_prompt_tokens
              << ", MTP rounds " << r.speculative.rounds << " drafted "
              << r.speculative.drafted_tokens << " accepted " << r.speculative.accepted_tokens
              << '\n';
}

// The image prompt (and, with `second_turn`, its conversation's next text-only turn resumed from
// the retained prefix) on a fresh Engine.
struct ImageRun {
    ninfer::GenerationResult first;
    std::optional<ninfer::GenerationResult> second;
};

ImageRun run_image(const char* artifact, int tp, int vision_device, int phase, bool second_turn) {
    ninfer::Engine engine(options(artifact, tp, true, vision_device));
    ImageRun out;
    out.first = generate(engine, image_prompt(phase), second_turn);
    if (second_turn) {
        ninfer::PromptInput followup = image_prompt(phase);
        ninfer::ChatMessage assistant;
        assistant.role = ninfer::ChatRole::Assistant;
        assistant.parts.push_back(text_part(out.first.content));
        followup.messages.push_back(std::move(assistant));
        followup.messages.push_back(
            message(ninfer::ChatRole::User, "Now list three colors you can see in it."));
        out.second = generate(engine, std::move(followup), true);
    }
    return out;
}

void exercise(const char* artifact) {
    // 1. Text invariance.
    ninfer::GenerationResult text_plain;
    {
        ninfer::Engine plain(options(artifact, 2, false, std::nullopt));
        text_plain = generate(plain, text_prompt());
    }
    ninfer::GenerationResult text_vision;
    std::uint32_t image_tokens = 0;
    {
        ninfer::Engine engine(options(artifact, 2, true, 0));
        text_vision  = generate(engine, text_prompt());
        image_tokens = engine.count_tokens(image_prompt(0));
    }
    std::cout << "text invariance:\n";
    print("without Vision", text_plain);
    print("with Vision", text_vision);
    check(!text_plain.generated_token_ids.empty() &&
              text_plain.generated_token_ids == text_vision.generated_token_ids,
          "a text prompt answers alike with and without Vision");
    std::cout << "  image prompt: " << image_tokens << " tokens\n";
    check(image_tokens > kMaxVisionTokens && image_tokens < kMaxVisionTokens + 64,
          "the image prompt counts the item's 256 merged tokens");

    // 2. Exactness against the no-draft control, with the tower on device 0, and 5. prefix reuse.
    const ImageRun mtp = run_image(artifact, 2, 0, 0, true);
    ImageRun no_draft;
    {
        const ScopedEnv none("NINFER_QWEN4_MTP_NO_DRAFTS", "1");
        no_draft = run_image(artifact, 2, 0, 0, true);
    }
    std::cout << "image prompt, tp 2, tower on device 0:\n";
    print("MTP", mtp.first);
    print("no drafts", no_draft.first);
    print("MTP second turn", *mtp.second);
    print("no-draft second turn", *no_draft.second);
    check(mtp.first.timings.vision_seconds > 0.0, "the Vision encoder ran");
    check(mtp.first.speculative.rounds > 0, "the image answer ran MTP rounds");
    check(mtp.first.generated_token_ids.size() == kOutput, "the image answer has its budget");
    check(mtp.first.generated_token_ids == no_draft.first.generated_token_ids,
          "image prompt: MTP tokens equal the no-draft tokens");
    check(mtp.second->reused_prompt_tokens > kMaxVisionTokens &&
              mtp.second->prefix_reuse_path != ninfer::PrefixReusePath::Root,
          "the second turn resumes the image prefix");
    check(mtp.second->generated_token_ids == no_draft.second->generated_token_ids,
          "second turn: MTP tokens equal the no-draft tokens");

    // 3. Copy parity.
    const ImageRun rank1 = run_image(artifact, 2, 1, 0, false);
    print("tower on device 1", rank1.first);
    check(rank1.first.generated_token_ids == mtp.first.generated_token_ids,
          "copy parity: the tower on device 1 answers as on device 0");

    // 4. The image reaches the model.
    const ImageRun other = run_image(artifact, 2, 0, 97, false);
    print("another image", other.first);
    check(other.first.generated_token_ids != mtp.first.generated_token_ids,
          "a different image changes the answer");

    // 6. One device.
    const ImageRun tp1 = run_image(artifact, 1, 0, 0, false);
    ImageRun tp1_no_draft;
    {
        const ScopedEnv none("NINFER_QWEN4_MTP_NO_DRAFTS", "1");
        tp1_no_draft = run_image(artifact, 1, 0, 0, false);
    }
    std::cout << "image prompt, tp 1:\n";
    print("MTP", tp1.first);
    print("no drafts", tp1_no_draft.first);
    check(tp1.first.timings.vision_seconds > 0.0, "tp 1: the Vision encoder ran");
    check(tp1.first.generated_token_ids == tp1_no_draft.first.generated_token_ids,
          "tp 1 image prompt: MTP tokens equal the no-draft tokens");
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_QWEN4_VISION_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_QWEN4_VISION_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two CUDA devices are required\n";
        return 77;
    }
    try {
        const ScopedEnv validation("NINFER_QWEN4_VALIDATION", "1");
        const ScopedEnv freeze("NINFER_QWEN4_EXPERT_FREEZE", "all");
        exercise(artifact);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "OK qwen4 vision engine\n";
    return 0;
}
