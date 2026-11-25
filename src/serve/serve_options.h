#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"
#include "serve/request.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

inline constexpr std::size_t kDefaultMaxRequestBytes = 384ULL << 20;
// The template renderer converts client JSON recursively: about 12 000 levels exhaust an 8 MiB
// thread stack (measured), so the maximum keeps a margin.
inline constexpr std::uint32_t kMaximumMaxRequestJsonDepth = 5000;
inline constexpr std::size_t kDefaultResponseStoreRecords  = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes    = 256ULL << 20;
// --log-speculation-detail records the accepted draft length of at most this many decode steps
// per request (one byte each in the JSONL record) unless --log-speculation-detail-max-steps
// changes it.
inline constexpr std::uint32_t kDefaultSpeculationDetailMaxSteps = 4096;
inline constexpr std::uint32_t kMaximumSpeculationDetailMaxSteps = 1U << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::filesystem::path chat_template_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact metadata.name
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::uint32_t max_context            = 8192;
    KvCapacityPolicy kv_capacity         = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency        = 1;
    std::uint32_t max_pending_requests   = 16;
    std::uint32_t pending_timeout_ms     = kDefaultPendingTimeoutMs;
    float decode_share                   = 0.0F; // --decode-share (EngineOptions::decode_share)
    std::uint32_t prefill_chunk          = 1024;
    std::uint32_t kv_lease_window_tokens = 4096; // --kv-lease-window; 0 = reserve max_tokens
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms  = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes        = kDefaultMaxRequestBytes;
    std::uint32_t max_request_json_depth = kDefaultMaxRequestJsonDepth; // --max-request-json-depth
    std::size_t media_cache_bytes        = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes         = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    // Structured output limits (--constraint-*): input size/depth/items checked in the HTTP layer
    // and again before compilation, compilation deadline and concurrency, grammar cache, matcher
    // time budgets of a generating request.
    OutputConstraintLimits constraint_limits;
    int device = 0;
    int tp     = 1;
    std::vector<int> devices; // One id per tensor-parallel rank; {device} at tp 1.
    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    // YaRN RoPE extension (EngineOptions::rope_yarn_factor); factor 1 is plain RoPE.
    float rope_yarn_factor                   = 1.0F;
    std::uint32_t rope_original_max_position = 262144;
    SpeculativeOptions speculative;
    LoopContrastOptions loop_contrast; // --loopcd (Inkling 2)
    ContextCacheOptions context_cache;
    bool enable_vision = false;
    std::optional<int> vision_device;               // CUDA device of the Vision tower
    std::optional<std::uint32_t> max_vision_tokens; // merged-token ceiling of one media item
    bool use_cuda_graph = true;
    bool tp_mailbox     = true;
    bool embedding_host = false; // token embedding table in mapped pinned host memory
    // Page-locked limit of the host-mapped weights (--host-pinned-max-gib); 0 = none.
    std::uint64_t host_pinned_max_bytes = 0;
    // --q8-a8, --attn-pv-f16acc, --tp-ar-payload int8: prefill speed for a small change in the
    // numbers, all off (README «Optional speed switches»).
    LossyPrefillOptions lossy_prefill;
    bool allow_prefix_reuse = true;
    // --reasoning-close-retry (on by default, every model; --no-reasoning-close-retry):
    // a reply that ends with the model's terminal token inside its thinking (or right after
    // "</think>" with blank content and no tool call) is closed with "\n</think>\n\n" and continued
    // once (serve/reasoning_retry.h).
    bool reasoning_close_retry = true;
    // --reasoning-eos-guard (on by default, every model; --no-reasoning-eos-guard): the
    // first token of a reply that starts in thinking is never a terminal token; a terminal draw is
    // redrawn without the terminal ids (models/qwen3_5/execution/reasoning_eos_guard.h).
    // Token-identical when it does not fire.
    bool reasoning_eos_guard = true;
    // Private long-anchor opportunities at the message boundaries before the final message: the
    // last N of them (0 disables). They keep a checkpoint at the point where a cancelled or edited
    // turn branches from the conversation; see to_prompt_input.
    // Unset = the default (resolve_turn_anchors): 0 at tp 2, otherwise the model's registered
    // default (0 on Qwen4Exp, 1 elsewhere).
    // On Qwen3.8-Flash-Next every prefill pass reads the non-resident experts from host RAM
    // (~0.8 s fixed), and the anchor splits one pass more per agentic turn (TTFT 2.67 s instead of
    // 1.82 s); a cancelled or edited turn then re-prefills the conversation. On models whose
    // weights are all on the GPU (Qwen3.8-27B) the extra pass costs nothing measurable, so they
    // keep the anchor.
    std::optional<std::uint32_t> turn_anchors;
    // Qwen4Exp product options (no effect on other models): elastic KV <-> expert slots
    // (--elastic, --kv-grow, --kv-budget-gib), the decode miss split (--miss-split
    // auto|calibrate|off) and the prompt-chunk width from which the missing experts stream over
    // PCIe (--stream-min-tokens, RoutedExpertOptions::stream_min_tokens; 0 = never).
    ElasticKvOptions elastic_kv;
    // --kv-ram-active (EngineOptions::kv_ram_active): the generating conversation's KV in RAM.
    bool kv_ram_active = false;
    std::optional<MissSplitMode> miss_split;
    bool miss_split_calibrate = false; // --miss-split calibrate (MissSplitOptions::calibrate)
    std::optional<std::uint32_t> stream_min_tokens;
    // RoutedExpertOptions::cpu_threads and gguf_cpu_threads (Qwen4Exp offload backend: the CPU
    // threads of the missing experts, pinned one per allowed physical core, spread over the L3
    // domains). Unset = resolved at load from the machine (ops::host::auto_expert_cpu_threads).
    std::optional<std::uint32_t> expert_cpu_threads;
    // --expert-cpus (RoutedExpertOptions::cpus; empty = NINFER_HOST_MOE_CPUS or automatic),
    // --expert-exclusive-layers (RoutedExpertOptions::exclusive_layers) and --gguf-hybrid-cost
    // (RoutedExpertOptions::gguf_hybrid_cost); unset leaves the engine's environment and defaults.
    std::vector<int> expert_cpus;
    std::optional<std::uint32_t> expert_exclusive_layers;
    std::optional<std::array<double, 3>> gguf_hybrid_cost;
    // --kv-ram-region-tokens / --kv-ram-cache-blocks (EngineOptions; 0 = environment or default).
    std::uint32_t kv_ram_region_tokens = 0;
    std::uint32_t kv_ram_cache_blocks  = 0;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    std::optional<std::uint32_t> default_thinking_budget;
    int default_max_tokens = 0;     // --default-max-tokens; 0 = the context left after the prompt
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    product::LogLevel log_level = product::LogLevel::Info;

    // Test measurement in request_done (per-step accepted draft lengths, prompt n-gram overlap);
    // requires the JSONL log.
    bool log_speculation_detail                = false;
    std::uint32_t speculation_detail_max_steps = kDefaultSpeculationDetailMaxSteps;

    // --prefill-miss-dump DIR (diagnostic, empty => off): writes the new prompt and the closest
    // recent context for a prefix-reuse miss; criteria in serve/prefill_miss_dump.h.
    std::string prefill_miss_dump_dir;
    std::uint32_t prefill_miss_min_tokens = 10000;
    double prefill_miss_min_match         = 0.5;

    // Unload and load (opt-in; the Engine is torn down inside the server and built again the way
    // start-up builds it, the HTTP server keeps its port):
    //   --unload-endpoints        POST /unload and POST /load
    //   --idle-unload-seconds N   unload after N seconds without a request (0 = never)
    //   --load-min-free-mib N|auto  a load needs N MiB free on every device, or with auto the
    //                             device memory the first Engine took there; else 503
    bool unload_endpoints             = false;
    std::uint32_t idle_unload_seconds = 0;
    std::optional<std::uint64_t> load_min_free_mib; // set: explicit N
    bool load_min_free_auto = false;
    // --load-wait-seconds N (with --load-min-free-mib): a load short of device memory waits up to
    // N seconds for it before answering 503.
    std::uint32_t load_wait_seconds = 0;
    // --unload-release-host: an unload also frees the page-locked expert banks (Flash-Next) and
    // the --embedding-host table, for a machine where the other program needs that RAM; by
    // default they stay page-locked and the next load reuses them without reading them again.
    bool unload_release_host = false;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
// The flags `--profile NAME` inserts. `dflash2_drafter`: the artifact carries a DFlash2 drafter
// and the command line names no `--spec`, so a profile with a DFlash2 configuration (27b) uses it.
std::vector<std::string_view> serve_profile_flags(std::string_view name, bool dflash2_drafter);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name);
// The output limit of a request that sets none: --default-max-tokens when given, otherwise the
// Engine's max_context, which the Engine lowers to the context left after the prompt.
[[nodiscard]] int resolve_default_max_tokens(const ServeOptions& options,
                                             std::uint32_t max_context);
// --turn-anchors when given, otherwise the default: 0 at tp 2 (each anchor holds a StateImage per
// turn in the pools the turn closures need), at tp 1 the model's default
// (LoadSummary::turn_anchors: 0 for Qwen4ExpForCausalLM, whose prefill passes read the
// non-resident experts from host RAM).
[[nodiscard]] std::uint32_t resolve_turn_anchors(std::optional<std::uint32_t> configured,
                                                 std::uint32_t model_default, std::uint32_t tp);

std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
