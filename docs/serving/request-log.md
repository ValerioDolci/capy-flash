# Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Every line is one `ninfer_serve_request_log` schema-v28 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, Host State/KV capacity and occupancy, KV sizing ledger, CUDA Graph allowance, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, caller label, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, request-owned materialization cost/search diagnostics, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters; with `--log-speculation-detail` also the per-step accepted draft lengths and the output/prompt 4-gram overlap |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, decode-round batch statistics, and per-GPU clocks, power, energy, temperature and clock-event reasons |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the response starts in thinking mode.

`request.client` (in `request_start`, `request_rejected`, `request_done` and `request_error`) is the
value of the optional NInfer request header `X-Ninfer-Client`, or `null` when the request did not
carry it. It lets a measurement separate benchmark, smoke and gate traffic from ordinary clients
sharing one log; the label never affects generation. The header carries one value of 1 to 64
characters from `A-Z a-z 0-9 . _ : / @ + -`; a repeated or malformed header is rejected before
preparation with HTTP 400 `invalid_client_label` (param `X-Ninfer-Client`). The repository's own
clients send `<kind>/<tool>` labels: `gate/tp2` (`tools/tp2/gate_client.py`), `bench/agentic-ab`,
`smoke/serve-contract` and `smoke/thinking-preservation`. With `--cors` the header is listed in
`Access-Control-Allow-Headers`.

`request_done.result.model_thinking_tokens` counts the model-origin tokens accepted while thinking
was open, including the token that closes it, for every response that starts in thinking, with or
without a budget; it matches the usage `reasoning_tokens` / `thinking_tokens` except that inserted
thinking control is not model-origin and raw output does not hide it. `thinking_budget`,
`thinking_control_tokens`, and `thinking_control_applied` describe the optional cap. Before schema
24 the count was recorded only under a thinking budget and was 0 otherwise.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, and a stable text-fallback reason. Of the structured calls,
`salvaged_call_count` kept a name outside the effective tool set or had a catalog spelling renamed,
and `structural_salvage_count` had closers supplied or a stray `</parameter>` or `</tool_call>`
dropped;
`trailing_content_bytes` is the prose returned as content after the calls (the recovery rules are
under [OpenAI Chat Completions](chat-completions.md)). Fallback reasons are `none`,
`malformed_structure`, `duplicate_parameter`, `invalid_tool_name`, `filtered_tool`, and
`trailing_content`; recovered output reports `none`. A salvaged name also writes an Info line to
the operational log. These counters contain no tool arguments or generated text.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. `request_done.speculative` is a separate top-level object of the
record, next to `timings_seconds` (not inside it): it contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, `accepted_per_position`, and
`accepted_length_histogram`. A round is one verification of a nonempty draft; `fallback_steps` are
decode steps without a draft. `accepted_per_position[i]` counts the rounds that accepted at least
`i + 1` draft tokens (the cumulative form), and `accepted_length_histogram[i]` the rounds that accepted
exactly `i`, for `i = 0..draft_window`; it sums to `rounds` and is computed from
`accepted_per_position` and `rounds`, so the two never disagree. Both are empty without a
speculative backend. A round whose draft was cut short by the remaining output budget counts at
the length it actually accepted. Rates can be derived downstream from raw token counts and seconds
instead of rounded stderr strings.

`request_done.speculative.accepted_lengths` is the order behind that histogram, recorded only with
`--log-speculation-detail` (otherwise it and `accepted_lengths_truncated` are `null`). It is a
string with one character per decode step in execution order: the hexadecimal digit of the draft
tokens that round accepted (`0` to the draft window, at most `f` = 15), or `-` for a step without a
draft (a `fallback_steps` step). Decode it with
`[None if c == "-" else int(c, 16) for c in s]`. Each round commits its accepted drafts plus one
token, and the first output token comes from prefill, so the step at index `t` starts at output
token `1 + sum(L + 1 for earlier rounds) + (earlier "-" steps)`, plus any thinking-control tokens
inserted before it; this places a collapse of acceptance in the response (opening, code, after
tool output, prose). Recording stops after
`--log-speculation-detail-max-steps` steps (default 4096, one byte each in the record);
`accepted_lengths_truncated` is then `true` and the string is the first steps of the request, while
the counters above still cover all of it. Without a speculative backend the string is empty. The
`server_start` record reports the cap as `server.speculation_detail_max_steps` (`null` when the
detail is off). Recording is a request observation: it never changes generation, and with the
option off the Engine records nothing and allocates nothing.

With the same option `request_done.result.prompt_ngram_overlap` is the share of the response that
repeats its prompt, the part a copy-style draft (prompt lookup) could propose: the fraction of the
output's overlapping 4-token windows (`completion_tokens - 3` of them, a repeated window counted
once per position) whose exact token-id sequence also occurs anywhere in the prompt. The output is
every generated token, thinking and tool-call tokens included; the prompt is the model input as
executed, chat template, tool declarations and earlier turns included. It is `null` with the option
off or with fewer than four output tokens. Serve copies the prompt token ids before submission and
computes the value after generation, before `request_done`, on the request's HTTP thread, with a
hash set of the output 4-grams scanned once over the prompt.

For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
When Vision is enabled, `vision_workspace` reports the aggregate prompt and maximum-item token
bounds plus encode peak and handoff layout/usage within that same allocation; these bytes must not
be added to `workspace.capacity_bytes`. The field is `null` when Vision is disabled.

`request_done.engine_timing` separates FIFO `queue_wait_seconds`, blocking
`device_wait_exposed_seconds`, and five mutually exclusive Host-active exposure phases under
`host_exposed_seconds`: `engine_boundary`, `program_submit`, `program_post`,
`engine_commit_output`, and `engine_maintenance`. `total` is exactly their sum and excludes Device
wait. The nested `decode` object reports the request's decode-class Host exposure, Device wait, and
round count; `units` reports its prefill/control unit counts. In a compact batch every participating
request is delayed by the full round, so these values explain request latency but **must not be
summed across concurrent requests**.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. Operational stderr summaries are rounded and are not the
aggregation source. OpenAI Responses, OpenAI Chat, and Anthropic generation requests receive a
request ID when they enter synchronous preparation. Successful preparation produces
`request_start`; a preparation failure produces `request_rejected` without a matching start. Each
started generation transaction then has exactly one machine terminal: `request_done` when Engine
returns its outcome, or `request_error` when generation fails before an outcome exists. Later
response rendering, Responses storage, or terminal transport failures are operational response
events only and do not add a second JSONL terminal. Schema/model validation rejections before
preparation and token-count-only calls are not measurement requests and do not receive request IDs.

By default the server persistently reports aggregate activity every five seconds. `prefill` counts
prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode`
counts tokens finally committed by decode rounds, excluding the first token produced by prefill.
For MTP and DFlash2 this is the accepted committed output, not draft or rejected tokens.
Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
same interval. The
`running`, `prefilling`, `decode_ready`, `waiting`, `materializing`, `capture_pending`, and
`terminal_pending` fields are the Engine scheduler snapshot at the end of the interval. The JSONL
`context_cache` object reports selection, capture, transfer, COW, pressure spill, private/shared
owner degradation and eviction, checkpoint drop, pressure search, budget exhaustion, maximal fallback, and historical-fork
counters as interval deltas; `occupancy` and `last_selection` are end-of-interval gauges. Materialization predictions are
request-owned and appear only on the corresponding `request_done` event.
`pressure.searches` counts plans accepted into Program resource transactions, including a transaction that later ends in
request-local abort; committed victim counters likewise report the resulting stable cache changes.

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose admission, context-transaction, replica, and
stats-publication slow paths; these detail values are already contained in a top-level Host phase
and must not be added to `total`. Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. Pretty throughput contains nonzero token rates and counts,
the current running/prefill/decode-ready composition, nonzero waiting/materialization/terminal
states, average decode batch, and Host-active time plus its fraction of the interval. Use JSONL for
complete measurement analysis.
Intervals with context materialization or retention activity are retained even when they contain no
token execution; only fully idle intervals are omitted.

The JSONL `throughput.gpus` array has one object per device the Engine uses (`--devices`, rank 0
first), read through NVML at the end of the interval: `device`, `sm_clock_mhz`, `memory_clock_mhz`,
`power_watts` (the driver's board power draw, itself a short average), `temperature_celsius`,
`energy_joules` consumed over the interval (so `energy_joules / interval_seconds` is the interval's
average power), and `clock_event_reasons`, the NVML clocks-event (throttle) reasons active at that
moment (`gpu_idle`, `applications_clocks_setting` for application or locked clocks, `sw_power_cap`,
`hw_slowdown`, `sync_boost`, `sw_thermal_slowdown`, `hw_thermal_slowdown`,
`hw_power_brake_slowdown`, `display_clock_setting`; an undefined bit appears as its hexadecimal
value). A value NVML cannot read is `null`. The server loads `libnvidia-ml.so.1` (installed with
the NVIDIA driver) at run time and matches devices by UUID; without it, or with the JSONL log or the
statistics interval disabled, `gpus` is `null`. The reads happen on the statistics thread every
interval, written or not, never on the request path; pretty throughput does not show them. Downstream measurement should prefer the
raw counters and seconds over rounded stderr rates.
