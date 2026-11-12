// Two-device tensor-parallel (tp 2) Engine on NINFER_TEST_ARTIFACT, the public generation route
// ninfer-serve uses: Qwen3.8 27B does not fit one 16 GB device, so the checks are semantic rather
// than a comparison with tp 1. CUDA Graphs, the context cache and two request lanes are on, as in
// the served default.
//
//   * single: "What is 17*23?" is prefilled and decoded greedily; the answer contains "391".
//   * concurrent: a long answer keeps lane 0 decoding while a second prompt is admitted on lane 1,
//     prefilled and decoded in shared rounds. Rank 1 must prefill each lane through that lane's
//     own KV execution row, so both answers must still be right (a stale rank 1 row writes lane
//     1's half of the KV heads into lane 0's pages). Two short arithmetic prompts then run
//     together as well.
//   * prefix reuse: a second turn repeats the first turn's conversation and asks a new question;
//     it reuses the retained prefix (restored KV pages and StateImages on both ranks) and answers
//     it correctly.
//   * long prefix, shaped like ninfer-serve's Chat Completions traffic (no session key, the
//     default implicit shared-prefix marker after the last message): a ~6k-token document with a
//     question, then the same conversation with the answer and a new question. The second turn
//     must reuse at least 80 % of the first prompt, with 4 lanes and with 1 lane (32k context).
//   * reuse after a one-shot flood: with 8 extra Device StateImages, 32 one-shot requests (more
//     retained checkpoints than the pool holds) are served four at a time, then a new two-turn
//     conversation must still reuse at least 80 % of its first prompt. Without a Host tier the
//     pool would otherwise stay full of old continuations and skip the new turn-closure capture.
//   * Host tier: four sessions (A1..D1, then A2..D2) on one Device StateImage and a 512-token
//     KV, so MTP turn-closure checkpoints (StateImage, main and MTP KV, both ranks' halves) go to
//     the pinned Host records and come back for the second turns. The oracle is the same sequence
//     on an Engine without a Host tier and with Device room to keep every checkpoint: same path,
//     same reused prompt tokens, same 16 greedy output tokens per second turn.
//   * Device KV lease (8k context and KV, one lane): a request asking for 6,000 output tokens
//     holds its prompt plus a 4,096-token window, so a retained ~2.5k-token conversation stays on
//     the Device while it answers in a few tokens, and that conversation's next turn reuses it.
//     Then a 6,000-token answer outgrows its window: the lease grows, and when the pool runs out it
//     takes the retained conversation's pages instead of ending the answer early.
//
// Returns 77 without the artifact or below two CUDA devices.

#include "models/qwen3_5/execution/qwen4_runtime.h"
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext     = 8192;
constexpr std::uint32_t kLongMaxContext = 32768;

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.tp                   = 2;
    options.device               = 0;
    options.devices              = {0, 1};
    options.max_context          = kMaxContext;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(2 * kMaxContext);
    options.prefill_chunk        = 1024;
    options.max_concurrency      = 2;
    options.max_pending_requests = 4;
    // Rank 1's KV and StateImages have no Host tier; the Device StateImage and private catalog
    // floors come from the Engine's tp 2 defaults.
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    return options;
}

// The serving shape: 32k context and KV, `lanes` lanes, tp 2 context-cache defaults.
ninfer::EngineOptions long_engine_options(const char* artifact, std::uint32_t lanes) {
    ninfer::EngineOptions options = engine_options(artifact);
    options.max_context           = kLongMaxContext;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(kLongMaxContext);
    options.max_concurrency       = lanes;
    options.max_pending_requests  = 4;
    options.kv_cache              = ninfer::KvCacheStorage::Int8Group64;
    return options;
}

ninfer::PromptInput user_prompt(std::string text) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

ninfer::RequestOptions greedy(std::uint32_t tokens, bool allow_prefix_reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = allow_prefix_reuse;
    return options;
}

bool contains(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
}

int check_answer(const ninfer::GenerationResult& result, std::string_view expected,
                 const char* label) {
    if (contains(result.content, expected)) { return 0; }
    std::cerr << label << ": expected \"" << expected << "\" in the answer, got \""
              << result.content << "\" (" << result.generated_token_ids.size() << " tokens)\n";
    return 1;
}

int exercise_load(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.devices.size() != 2 || load.devices[0].device == load.devices[1].device ||
        load.devices[0].capacity_bytes == 0 || load.devices[1].capacity_bytes == 0 ||
        load.devices[0].sharded_bytes == 0 || load.devices[1].sharded_bytes == 0) {
        std::cerr << "load summary does not describe two ranks holding weight shards\n";
        return 1;
    }
    const ninfer::EngineOptions& options = engine.options();
    if (options.context_cache.host_state_slots != 0 ||
        options.context_cache.host_kv_capacity_bytes != 0 ||
        options.context_cache.device_state_slots.value_or(0) < 8 ||
        options.context_cache.max_private_continuations.value_or(0) < 8) {
        std::cerr << "tp 2 context-cache defaults were not applied\n";
        return 1;
    }
    return 0;
}

int exercise_single(ninfer::Engine& engine) {
    const ninfer::GenerationResult result = engine.generate(
        engine.prepare(user_prompt("What is 17*23? Answer with the number only.")), greedy(32));
    return check_answer(result, "391", "single request");
}

int exercise_concurrent(ninfer::Engine& engine) {
    int failures                      = 0;
    const ninfer::RuntimeStats before = engine.runtime_stats();
    // Submitted before either is waited on: the long answer holds lane 0 while the second prompt
    // is admitted on lane 1 and joins its decode rounds.
    ninfer::GenerationHandle counting = engine.submit(
        engine.prepare(user_prompt("Write the integers from 1 to 40, separated by a comma and a "
                                   "space, and nothing else.")),
        greedy(160));
    ninfer::GenerationHandle product = engine.submit(
        engine.prepare(user_prompt("What is 17*23? Answer with the number only.")), greedy(32));
    const ninfer::GenerationResult counted    = counting.wait();
    const ninfer::GenerationResult multiplied = product.wait();
    const ninfer::RuntimeStats after          = engine.runtime_stats();
    failures += check_answer(counted, "38, 39, 40", "concurrent long lane");
    failures += check_answer(multiplied, "391", "concurrent second lane");
    const std::uint64_t rounds = after.decode_rounds - before.decode_rounds;
    const std::uint64_t rows   = after.decode_row_rounds - before.decode_row_rounds;
    if (rounds == 0 || rows <= rounds) {
        std::cerr << "concurrent leg ran no two-row decode round (" << rounds << " rounds, " << rows
                  << " rows)\n";
        ++failures;
    }

    ninfer::GenerationHandle first = engine.submit(
        engine.prepare(user_prompt("What is 17*23? Answer with the number only.")), greedy(32));
    ninfer::GenerationHandle second = engine.submit(
        engine.prepare(user_prompt("What is 12*12? Answer with the number only.")), greedy(32));
    failures += check_answer(first.wait(), "391", "concurrent pair, first");
    failures += check_answer(second.wait(), "144", "concurrent pair, second");
    return failures;
}

int exercise_prefix_reuse(ninfer::Engine& engine) {
    const auto conversation = [] {
        ninfer::PromptInput input =
            user_prompt("Remember the code 4817 and the color green. For now, reply only: OK.");
        input.context_cache.session_key = "tp2-real";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return input;
    };
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation()), greedy(16, true));
    if (first.generated_token_ids.empty()) {
        std::cerr << "prefix reuse: the first turn generated nothing\n";
        return 1;
    }

    ninfer::PromptInput followup = conversation();
    ninfer::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
    followup.messages.push_back(std::move(assistant));
    ninfer::ChatMessage question;
    question.role = ninfer::ChatRole::User;
    question.parts.push_back(
        ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                            .text  = "What is 17*23? Answer with the number only.",
                            .media = {}});
    followup.messages.push_back(std::move(question));
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(std::move(followup)), greedy(32, true));
    int failures = check_answer(second, "391", "prefix reuse second turn");
    if (second.reused_prompt_tokens == 0 ||
        second.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
        std::cerr << "prefix reuse: the second turn reused no prompt tokens (path "
                  << static_cast<int>(second.prefix_reuse_path) << ")\n";
        ++failures;
    }
    return failures;
}

// OpenAI Chat Completions without prompt_cache_options: one default implicit shared-prefix
// candidate after the last content part, and no Engine structural candidates
// (serve/openai_common.cpp).
void mark_like_chat_completions(ninfer::PromptInput& input) {
    const ninfer::ChatMessage& last = input.messages.back();
    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
        .after_message_count      = static_cast<std::uint32_t>(input.messages.size()),
        .kind                     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence                 = ninfer::SharedCandidateEvidence::DefaultAutomatic,
        .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = static_cast<std::uint32_t>(last.parts.size()),
    });
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
}

std::string long_document(std::string_view city = "Verona", int lines = 240) {
    std::string text = "Read the warehouse log, then answer the question.\n\n";
    for (int line = 1; line <= lines; ++line) {
        text += "Line " + std::to_string(line) + ": the " + std::string(city) +
                " warehouse received " + std::to_string(line * 7) + " crates of apples and " +
                std::to_string(line * 3) + " crates of pears, shipped on day " +
                std::to_string(1 + line % 28) + ".\n";
    }
    return text;
}

void print_cache_stats(const ninfer::RuntimeStats& before, const ninfer::RuntimeStats& after) {
    std::cerr << "  captures completed " << after.active_captures_completed << " aborted "
              << after.active_captures_aborted << "; selections root "
              << after.root_selections - before.root_selections << " endpoint "
              << after.private_endpoint_selections - before.private_endpoint_selections
              << " turn-closure "
              << after.private_turn_closure_selections - before.private_turn_closure_selections
              << " anchor "
              << after.private_long_anchor_selections - before.private_long_anchor_selections
              << " shared "
              << after.shared_stable_prefix_selections - before.shared_stable_prefix_selections
              << "; computed prefill "
              << after.computed_prefill_tokens - before.computed_prefill_tokens
              << "; evicted private " << after.pressure_private_owners_evicted << " shared "
              << after.pressure_shared_owners_evicted << " checkpoints dropped "
              << after.pressure_checkpoints_dropped << '\n';
}

int exercise_long_prefix_reuse(ninfer::Engine& engine, const char* label,
                               std::string_view city = "Verona", int lines = 240,
                               std::uint32_t minimum_tokens = 4096) {
    const std::string document = long_document(city, lines);
    const auto first_turn      = [&] {
        ninfer::PromptInput input =
            user_prompt(document + "\nQuestion: what is 17*23? Answer with the number only.");
        mark_like_chat_completions(input);
        return input;
    };
    const ninfer::RuntimeStats start = engine.runtime_stats();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_turn()), greedy(32, true));
    int failures = check_answer(first, "391", label);

    ninfer::PromptInput followup = first_turn();
    followup.context_cache       = {};
    ninfer::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
    followup.messages.push_back(std::move(assistant));
    ninfer::ChatMessage question;
    question.role = ninfer::ChatRole::User;
    question.parts.push_back(
        ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                            .text  = "What is 12*12? Answer with the number only.",
                            .media = {}});
    followup.messages.push_back(std::move(question));
    mark_like_chat_completions(followup);
    const ninfer::RuntimeStats middle = engine.runtime_stats();
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(std::move(followup)), greedy(32, true));
    const ninfer::RuntimeStats end = engine.runtime_stats();
    failures += check_answer(second, "144", label);
    const std::uint32_t first_tokens = first.prompt.prompt_tokens;
    if (first_tokens < minimum_tokens ||
        static_cast<std::uint64_t>(second.reused_prompt_tokens) * 5U <
            static_cast<std::uint64_t>(first_tokens) * 4U) {
        std::cerr << label << ": second turn reused " << second.reused_prompt_tokens << " of "
                  << second.prompt.prompt_tokens << " prompt tokens (first turn " << first_tokens
                  << ", path " << static_cast<int>(second.prefix_reuse_path) << ")\n";
        std::cerr << "  after the first turn:";
        print_cache_stats(start, middle);
        std::cerr << "  after the second turn:";
        print_cache_stats(middle, end);
        ++failures;
    }
    return failures;
}

// More one-shot continuations than the Device checkpoint pool holds, four in flight at a time,
// then a new conversation that must still be captured and reused.
int exercise_reuse_after_one_shots(ninfer::Engine& engine, const char* label) {
    int failures = 0;
    for (int round = 0; round < 8; ++round) {
        std::vector<ninfer::GenerationHandle> handles;
        for (int lane = 0; lane < 4; ++lane) {
            const int left = 11 + round * 4 + lane;
            ninfer::PromptInput input =
                user_prompt("What is " + std::to_string(left) + "*3? Answer with the number only.");
            mark_like_chat_completions(input);
            handles.push_back(engine.submit(engine.prepare(std::move(input)), greedy(16, true)));
        }
        for (int lane = 0; lane < 4; ++lane) {
            const int left = 11 + round * 4 + lane;
            failures += check_answer(handles[static_cast<std::size_t>(lane)].wait(),
                                     std::to_string(left * 3), label);
        }
    }
    return failures + exercise_long_prefix_reuse(engine, label, "Trento", 40, 512);
}

// The configurations of the Host tier check: MTP with int8 KV (the --profile 27b shape), and
// DFlash2 with nvfp4 KV (the drafter's local rings travel in the StateImage; nvfp4 pages carry
// their scale planes).
enum class HostVariant { MtpInt8, DFlash2Nvfp4 };

ninfer::EngineOptions host_restore_options(const char* artifact, bool host_tier,
                                           HostVariant variant) {
    ninfer::EngineOptions options = engine_options(artifact);
    // One lane: the KV capacity is max_context, so the oracle gets room through a longer context.
    options.max_context   = host_tier ? 512 : 4096;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
    options.prefill_chunk = 256;
    const bool dflash2    = variant == HostVariant::DFlash2Nvfp4;
    options.speculative.backend =
        dflash2 ? ninfer::SpeculativeBackend::DFlash2 : ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = dflash2 ? 4 : 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    if (dflash2) { options.kv_cache = ninfer::KvCacheStorage::Nvfp4Group16; }
    options.max_concurrency                                 = 1;
    options.max_pending_requests                            = 1;
    options.context_cache.device_state_slots                = host_tier ? 1 : 8;
    options.context_cache.host_state_slots                  = host_tier ? 8 : 0;
    options.context_cache.host_kv_capacity_bytes            = host_tier ? (256ULL << 20) : 0;
    options.context_cache.max_private_continuations         = 8;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

struct HostRestoreRun {
    std::vector<ninfer::GenerationResult> resumed; // the sessions' second turns, in order
    ninfer::RuntimeStats stats;
};

// Four sessions (A1..D1, then A2..D2) on one Device StateImage and a 512-token KV: the first turns'
// checkpoints must leave the Device for the next session, and come back for the second turns.
HostRestoreRun run_host_restore(const char* artifact, bool host_tier, HostVariant variant) {
    ninfer::Engine engine(host_restore_options(artifact, host_tier, variant));
    const auto request = [](std::uint32_t outputs) {
        ninfer::RequestOptions options      = greedy(outputs, true);
        options.stop.include_model_defaults = false;
        return options;
    };
    const auto first_turn = [](const char* word, const char* session) {
        std::string text;
        for (int index = 0; index < 300; ++index) { text += std::string(word) + " "; }
        ninfer::PromptInput input       = user_prompt(std::move(text));
        input.context_cache.session_key = session;
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return input;
    };
    const auto second_turn = [](ninfer::PromptInput input, const ninfer::GenerationResult& first) {
        ninfer::ChatMessage assistant;
        assistant.role              = ninfer::ChatRole::Assistant;
        assistant.reasoning_content = first.reasoning;
        assistant.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
        input.messages.push_back(std::move(assistant));
        ninfer::ChatMessage followup;
        followup.role = ninfer::ChatRole::User;
        followup.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "Continue briefly.", .media = {}});
        input.messages.push_back(std::move(followup));
        return input;
    };
    const char* const words[]    = {"alpha", "beta", "gamma", "delta"};
    const char* const sessions[] = {"tp2-host-a", "tp2-host-b", "tp2-host-c", "tp2-host-d"};
    std::vector<ninfer::GenerationResult> first;
    for (std::size_t index = 0; index < 4; ++index) {
        first.push_back(
            engine.generate(engine.prepare(first_turn(words[index], sessions[index])), request(5)));
    }
    HostRestoreRun run;
    for (std::size_t index = 0; index < 4; ++index) {
        run.resumed.push_back(engine.generate(
            engine.prepare(second_turn(first_turn(words[index], sessions[index]), first[index])),
            request(16)));
    }
    run.stats = engine.runtime_stats();
    return run;
}

int exercise_host_restore(const char* artifact, HostVariant variant) {
    const HostRestoreRun host   = run_host_restore(artifact, true, variant);
    const HostRestoreRun device = run_host_restore(artifact, false, variant);
    const bool backend_kv       = variant == HostVariant::MtpInt8; // DFlash2: no backend KV
    int failures                = 0;
    const auto print            = [](const char* label, const ninfer::RuntimeStats& stats) {
        std::cerr << "  " << label << ": state d2h " << stats.state_d2h_count << " h2d "
                  << stats.state_h2d_count << " moves " << stats.state_moves << " forks "
                  << stats.state_forks << " restores " << stats.state_restores << "; main d2h "
                  << stats.main_kv_d2h_pages << " h2d " << stats.main_kv_h2d_pages
                  << "; backend d2h " << stats.backend_kv_d2h_pages << " h2d "
                  << stats.backend_kv_h2d_pages << "; degraded "
                  << stats.pressure_private_owners_degraded << " evicted "
                  << stats.pressure_private_owners_evicted << '\n';
    };
    if (host.stats.state_h2d_count == 0 || host.stats.main_kv_h2d_pages == 0 ||
        (backend_kv && host.stats.backend_kv_h2d_pages == 0)) {
        std::cerr << "Host tier: no checkpoint came back from the Host\n";
        print("Host tier", host.stats);
        ++failures;
    }
    if (device.stats.state_h2d_count != 0 || device.stats.main_kv_h2d_pages != 0) {
        std::cerr << "Host tier oracle restored from a Host tier it does not have\n";
        print("oracle", device.stats);
        ++failures;
    }
    for (std::size_t turn = 0; turn < host.resumed.size(); ++turn) {
        const ninfer::GenerationResult& got      = host.resumed[turn];
        const ninfer::GenerationResult& expected = device.resumed[turn];
        if (got.prefix_reuse_path == ninfer::PrefixReusePath::Root ||
            got.prefix_reuse_path != expected.prefix_reuse_path ||
            got.reused_prompt_tokens != expected.reused_prompt_tokens) {
            std::cerr << "Host tier: session " << turn << " resumed on path "
                      << static_cast<int>(got.prefix_reuse_path) << " reusing "
                      << got.reused_prompt_tokens << ", Device oracle path "
                      << static_cast<int>(expected.prefix_reuse_path) << " reusing "
                      << expected.reused_prompt_tokens << '\n';
            ++failures;
        }
        if (got.generated_token_ids != expected.generated_token_ids ||
            got.generated_token_ids.size() != 16) {
            std::cerr << "Host tier: session " << turn
                      << " tokens differ from the Device oracle (\"" << got.content << "\" vs \""
                      << expected.content << "\")\n";
            ++failures;
        }
    }
    if (failures == 0) { print("Host tier", host.stats); }
    return failures;
}

ninfer::EngineOptions lease_options(const char* artifact) {
    ninfer::EngineOptions options     = engine_options(artifact);
    options.max_context               = kMaxContext;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.max_concurrency           = 1;
    options.max_pending_requests      = 2;
    options.kv_cache                  = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.context_cache.device_state_slots = 8;
    return options;
}

int exercise_kv_lease(const char* artifact, std::uint32_t window = 4096, bool host = false) {
    ninfer::EngineOptions options  = lease_options(artifact);
    options.kv_lease_window_tokens = window;
    if (host) {
        options.context_cache.host_state_slots       = 8;
        options.context_cache.host_kv_capacity_bytes = 2ULL << 30;
    }
    ninfer::Engine engine(std::move(options));
    int failures               = 0;
    const std::string document = long_document("Bolzano", 100);
    const auto conversation    = [&] {
        ninfer::PromptInput input =
            user_prompt(document + "\nQuestion: what is 17*23? Answer with the number only.");
        mark_like_chat_completions(input);
        return input;
    };
    const auto followup = [&](const ninfer::GenerationResult& first) {
        ninfer::PromptInput input = conversation();
        input.context_cache       = {};
        ninfer::ChatMessage assistant;
        assistant.role = ninfer::ChatRole::Assistant;
        assistant.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
        input.messages.push_back(std::move(assistant));
        ninfer::ChatMessage question;
        question.role = ninfer::ChatRole::User;
        question.parts.push_back(
            ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                .text  = "What is 12*12? Answer with the number only.",
                                .media = {}});
        input.messages.push_back(std::move(question));
        mark_like_chat_completions(input);
        return input;
    };

    // 1. A short answer under a 6,000-token budget leaves the retained conversation alone.
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation()), greedy(32, true));
    failures += check_answer(first, "391", "lease: first turn");
    const ninfer::GenerationResult short_answer =
        engine.generate(engine.prepare(user_prompt("What is 6*7? Answer with the number only.")),
                        greedy(6000, true));
    failures += check_answer(short_answer, "42", "lease: short answer, large budget");
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(followup(first)), greedy(32, true));
    failures += check_answer(second, "144", "lease: second turn");
    if (window == 0) {
        // --kv-lease-window 0: the whole budget is reserved at admission, as before the lease,
        // so the 6,000-token request has to take the retained conversation's pages.
        if (static_cast<std::uint64_t>(second.reused_prompt_tokens) * 5U >=
            static_cast<std::uint64_t>(first.prompt.prompt_tokens) * 4U) {
            std::cerr << "lease off: the second turn still reused " << second.reused_prompt_tokens
                      << " tokens: --kv-lease-window 0 did not reserve the whole budget\n";
            ++failures;
        }
        std::cerr << "  lease off: second turn reused " << second.reused_prompt_tokens << " of "
                  << first.prompt.prompt_tokens << '\n';
        return failures;
    }
    if (static_cast<std::uint64_t>(second.reused_prompt_tokens) * 5U <
        static_cast<std::uint64_t>(first.prompt.prompt_tokens) * 4U) {
        std::cerr << "lease: the second turn reused " << second.reused_prompt_tokens << " of "
                  << first.prompt.prompt_tokens
                  << " tokens: the large budget evicted the retained conversation\n";
        ++failures;
    }

    // 2. A long answer grows its lease past the window and takes the retained pages.
    const ninfer::RuntimeStats before          = engine.runtime_stats();
    ninfer::RequestOptions long_request        = greedy(6000, true);
    long_request.stop.include_model_defaults   = false;
    const ninfer::GenerationResult long_answer = engine.generate(
        engine.prepare(user_prompt("Write the numbers from 1 to 5000, separated by spaces.")),
        long_request);
    const ninfer::RuntimeStats after = engine.runtime_stats();
    if (long_answer.generated_token_ids.size() != 6000 ||
        after.kv_lease_reclaimed_owners <= before.kv_lease_reclaimed_owners ||
        after.kv_lease_shortened_answers != before.kv_lease_shortened_answers) {
        std::cerr << "lease: the long answer produced " << long_answer.generated_token_ids.size()
                  << " of 6000 tokens; reclaimed owners "
                  << after.kv_lease_reclaimed_owners - before.kv_lease_reclaimed_owners
                  << ", shortened answers "
                  << after.kv_lease_shortened_answers - before.kv_lease_shortened_answers << '\n';
        ++failures;
    }
    std::cerr << "  lease" << (host ? " + Host tier" : "") << ": second turn reused "
              << second.reused_prompt_tokens << " of " << first.prompt.prompt_tokens
              << "; long answer " << long_answer.generated_token_ids.size()
              << " tokens, reclaimed owners "
              << after.kv_lease_reclaimed_owners - before.kv_lease_reclaimed_owners << ", degraded "
              << after.pressure_private_owners_degraded - before.pressure_private_owners_degraded
              << ", evicted "
              << after.pressure_private_owners_evicted - before.pressure_private_owners_evicted
              << '\n';
    if (host) {
        // 3. With a Host tier the long answer moved the conversation to the Host instead of
        // dropping it: its next turn resumes from there.
        ninfer::PromptInput third = followup(first);
        ninfer::ChatMessage assistant;
        assistant.role = ninfer::ChatRole::Assistant;
        assistant.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = second.content, .media = {}});
        third.messages.push_back(std::move(assistant));
        ninfer::ChatMessage question;
        question.role = ninfer::ChatRole::User;
        question.parts.push_back(
            ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                .text  = "What is 5*5? Answer with the number only.",
                                .media = {}});
        third.messages.push_back(std::move(question));
        mark_like_chat_completions(third);
        const ninfer::RuntimeStats before_third = engine.runtime_stats();
        const ninfer::GenerationResult third_turn =
            engine.generate(engine.prepare(std::move(third)), greedy(32, true));
        const ninfer::RuntimeStats after_third = engine.runtime_stats();
        failures += check_answer(third_turn, "25", "lease + Host tier: third turn");
        std::cerr << "  lease + Host tier: third turn reused " << third_turn.reused_prompt_tokens
                  << " of " << third_turn.prompt.prompt_tokens << " (main KV h2d "
                  << after_third.main_kv_h2d_pages - before_third.main_kv_h2d_pages << ")\n";
        if (static_cast<std::uint64_t>(third_turn.reused_prompt_tokens) * 5U <
                static_cast<std::uint64_t>(second.prompt.prompt_tokens) * 4U ||
            after_third.main_kv_h2d_pages == before_third.main_kv_h2d_pages) {
            std::cerr << "lease + Host tier: the conversation did not come back from the Host\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

// NINFER_TP_PREFILL_SP=1: the long prompts above run in half-chunks over the HostExchange, so the
// sequence-parallel prefill must have run them; with NINFER_TP_PREFILL_SP_NO_ROOM=1 every such call
// must have fallen back to the overlap path. Either way the answers above must be right. A pair
// with peer access has no HostExchange, so the SP prefill stays off there and this check is
// skipped.
int check_sp_prefill() {
    namespace exec = ninfer::models::qwen3_5::execution;
    if (!exec::qwen4_prefill_sp_env()) { return 0; }
    int forward = 0, reverse = 0;
    if (cudaDeviceCanAccessPeer(&forward, 0, 1) == cudaSuccess &&
        cudaDeviceCanAccessPeer(&reverse, 1, 0) == cudaSuccess && forward != 0 && reverse != 0) {
        std::cerr << "  SP prefill: the pair has peer access, so it stays off; not checked\n";
        return 0;
    }
    const auto& counts            = exec::qwen4_prefill_sp_counts();
    const std::uint64_t runs      = counts.runs.load();
    const std::uint64_t fallbacks = counts.fallbacks.load();
    std::cerr << "  SP prefill: " << runs << " calls ran it, " << fallbacks << " fell back\n";
    if (exec::qwen4_prefill_sp_no_room_env()) {
        if (runs == 0 && fallbacks > 0) { return 0; }
        std::cerr << "SP prefill: the forced no-room fallback did not take every call\n";
        return 1;
    }
    if (runs > 0 && fallbacks == 0) { return 0; }
    std::cerr << "SP prefill: the long prompts did not run the sequence-parallel schedule\n";
    return 1;
}

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: tensor parallelism needs two CUDA devices\n";
        return 77;
    }
    try {
        ninfer::Engine engine(engine_options(artifact));
        int failures = exercise_load(engine);
        failures += exercise_single(engine);
        failures += exercise_concurrent(engine);
        failures += exercise_prefix_reuse(engine);
        if (failures != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 Engine failed: " << error.what() << '\n';
        return 1;
    }
    try {
        int failures = 0;
        {
            ninfer::EngineOptions options            = long_engine_options(artifact, 4);
            options.context_cache.device_state_slots = 8;
            ninfer::Engine engine(std::move(options));
            failures += exercise_long_prefix_reuse(engine, "long prefix, 4 lanes");
            failures += exercise_reuse_after_one_shots(engine, "reuse after one-shot flood");
        }
        {
            ninfer::Engine engine(long_engine_options(artifact, 1));
            failures += exercise_long_prefix_reuse(engine, "long prefix, 1 lane");
        }
        if (failures != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 Engine failed: " << error.what() << '\n';
        return 1;
    }
    if (check_sp_prefill() != 0) { return 1; }
    {
        // Guardrail: a Host tier larger than the RAM the machine can spare is refused at startup
        // with a message that says what fits, instead of pinning it and courting an OOM kill.
        ninfer::EngineOptions options                = lease_options(artifact);
        options.context_cache.host_kv_capacity_bytes = 1ULL << 40;
        bool refused                                 = false;
        try {
            ninfer::Engine engine(std::move(options));
        } catch (const std::invalid_argument& error) {
            refused =
                std::string_view(error.what()).find("--host-kv-mib") != std::string_view::npos;
            if (!refused) { std::cerr << "Host tier RAM guard message: " << error.what() << '\n'; }
        } catch (const std::exception& error) {
            std::cerr << "Host tier RAM guard: unexpected failure: " << error.what() << '\n';
        }
        if (!refused) {
            std::cerr << "Host tier RAM guard: a 1 TiB Host KV tier was not refused\n";
            return 1;
        }
    }
    try {
        if (exercise_kv_lease(artifact) != 0) { return 1; }
        if (exercise_kv_lease(artifact, 0) != 0) { return 1; }
        if (exercise_kv_lease(artifact, 4096, true) != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 Device KV lease Engine failed: " << error.what() << '\n';
        return 1;
    }
    try {
        if (exercise_host_restore(artifact, HostVariant::MtpInt8) != 0) { return 1; }
        std::cerr << "  Host tier, DFlash2 with nvfp4 KV:\n";
        if (exercise_host_restore(artifact, HostVariant::DFlash2Nvfp4) != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 Host tier Engine failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
