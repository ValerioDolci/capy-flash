# HTTP serving

`build/apps/ninfer-serve` loads one v3 `.ninfer` artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine.

## Start the server

See [CUDA synchronization](cli.md#cuda-synchronization) for the shared `NINFER_CUDA_SYNC` setting.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. A shared
240,000-token Main Text KV pool serves admitted requests; either request may use the full capacity
when running alone, and two requests run concurrently when their complete reservations fit.
In capy-flash a request's Device KV reservation is a lease: its prompt plus a window of up to
4,096 output tokens (or the prefill chunk, if larger), extended at a decode-round boundary as the
answer grows, up to the request's own `max_tokens`. A client that asks for 64k output tokens and
uses 300 therefore no longer keeps 64k tokens of KV away from retained conversations. When the pool
cannot extend a lease, idle retained conversations and idle automatic shared prefixes are released,
least valuable first; only if that is not enough does the answer end early with `finish_reason`
`length`. The request log reports the leased pages (`device_*_kv_lease_pages`), the owners released
for a lease and the answers cut short. `--kv-lease-window N` sets the window (default 4096);
`--kv-lease-window 0` reserves the whole `max_tokens` at admission, as earlier releases did.

The Host tiers are pinned RAM that the kernel cannot page out. The server refuses to start when
`--host-kv-mib` plus the Host state slots would leave less than max(4 GiB, 5 % of RAM) available,
and says how many MiB fit (`NINFER_HOST_TIER_RAM_GUARD=0` skips the check).

With `C=2` and two extra Device checkpoint slots, the process owns two active StateImage guarantees
plus a global pool of two Device-resident checkpoints. Eight pinned Host State slots and 8 GiB of
pinned Host KV retain inactive continuations under Device pressure. Active request capacity is two.

Other artifacts use the same command shape with their own path. For 35B-A3B DFlash, replace the MTP
selection with `--spec dflash --draft-tokens 7 --lm-head-draft`. Qwen3.8-27B
artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`, with
`--lm-head-draft` optional. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests fail with HTTP 400 `vision_disabled`. Add
`--vision` when the server must accept image or video input. Speculative residency is likewise
frozen by `--spec mtp|dflash|dflash2` and `--draft-tokens`; omitting `--spec` loads no speculative backend.
`--lm-head-draft` additionally loads the optimized proposal head. DFlash on 35B-A3B and DFlash2 on Qwen3.8-27B can be combined
with `--vision`; each accelerates generated-text decode after multimodal prefill, while Vision encode
and prefill remain outside speculative acceleration. A later request cannot enable a capability
omitted at startup. The artifact need only contain the Text backbone and the optional components
selected for this process.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | Engine readiness |
| `GET /v1/models` | configured OpenAI model alias and effective `max_model_len` |
| `GET /v1/models/{id}` | lookup of the configured alias and effective `max_model_len` |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

`GET /health` returns HTTP 200 with `{"status":"ok"}` while the Engine can accept work. After an
Engine-wide failure it returns HTTP 503 with `{"status":"unavailable"}`. Temporary queue
saturation does not make the Engine unavailable. The endpoint remains unauthenticated.

An Engine-wide failure (an error thrown by the Engine worker, such as a timed-out tensor-parallel
exchange) is permanent for the process: `ninfer-serve` logs it, stops accepting connections within
about 250 ms and **exits with status 2** (startup failures exit with 1; a failed CUDA call aborts
the process). Run it under a supervisor that restarts on a non-zero exit — `Restart=on-failure` in
systemd, or llama-swap, which reloads the model on the next request — instead of relying on
`/health`, which supervisors usually probe only while the model loads.

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract.

All three generation SSE endpoints emit the standard `: keep-alive` comment after five seconds
without a protocol event. The comment is transport-only: SSE clients ignore it, and it does not
change generated text, event ordering, usage, stored Responses, or request logs. On Linux, accepted
connections also use TCP keepalive and a 15-second `TCP_USER_TIMEOUT`; together with the heartbeat,
a dead or unacknowledging peer is normally cancelled within about 20 seconds, including while the
request is waiting or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream NInfer
connection when the downstream client disappears.

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history, plus legacy `function` history;
- string content and ordered text/refusal parts; adjacent parts are preserved without inserted
  separators, and empty wire content remains an empty turn;
- User `image_url` parts, tool-result `image_url` parts used by compatible clients, and the User
  `video_url` extension using HTTP(S) or data URIs; image detail is omitted or `auto`;
- nonnegative `max_completion_tokens` and the legacy `max_tokens` spelling; zero performs prompt
  processing without generation;
- `temperature`, `top_p`, presence/frequency penalties, and signed integer `seed`;
- the compatible `top_k` (`0..20`) and `min_p` (`0..1`) sampler extensions;
- up to four non-empty stop strings, applied to both reasoning and answer output;
- `n:1`, text-only `modalities`, and `response_format: {"type":"text"}`;
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- llama.cpp-compatible terminal `timings`, plus opt-in `timings_per_token` and
  streaming `return_progress` observations;
- non-strict function tools with `tool_choice` `auto`, `none`, or `allowed_tools` in `auto` mode,
  parallel calls enabled, assistant tool-call history, tool-result messages, and legacy
  function-call history;
- the top-level `reasoning_effort` field;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes JSON constrained output, nonzero `logit_bias`, requested log probabilities,
audio/file input or audio output, `strict:true`, required or named tool choice,
`parallel_tool_calls:false` with enabled tools, explicit low/high image detail, web search,
moderation, low/high verbosity, stored Chat Completions, and non-empty legacy `functions`.
Each capability rejection identifies the affected field and the guarantee NInfer cannot provide.
Known constrained-decoding aliases (`grammar`, `structured_outputs`, `guided_json`, `guided_regex`,
`guided_choice`, and `guided_grammar`) receive the same explicit rejection instead of being treated
as unknown hints.

Semantically neutral fields do not make an otherwise executable request fail. All-zero
`logit_bias`, `logprobs:false`, `top_logprobs:0`, `verbosity:"medium"`, empty legacy tool controls,
text-only `audio` configuration, and `prediction` are accepted without changing Engine execution.
Metadata, user/safety identifiers, service-tier and prompt-cache hints are likewise advisory.
Unknown top-level fields are ignored.

A string `name` on a `tool` message is accepted as an ignored, output-neutral compatibility
extension for clients that mirror the function name onto tool results. It does not participate in
tool identity, prompt rendering, or output. Non-string values are malformed; non-empty names on
other message roles remain unsupported because they carry participant identity that the loaded chat
template cannot represent.

For commonly generated OpenAI-compatible payloads, `repetition_penalty` is accepted only at its
neutral value `1`, and `mm_processor_kwargs` when empty or containing only null values. String-form
image/video URLs are also accepted.

Malformed protocol values return field-specific HTTP 400 errors. Invalid media sources, bytes, or
decoded content use `invalid_media`; remote fetch and timeout failures retain their dedicated
server-error codes. Failures in the normalized prompt contract use `invalid_prompt`; typed capacity
and availability failures retain their dedicated codes. Internal invariant failures are not
relabeled as client input errors.

The request `model` must equal the public model ID: the artifact `identity.model_id` by default, or
the explicit `--model-id` override. Reasoning is returned separately as `reasoning_content`; answer
text remains in `content`.

Across Chat Completions, Responses, and Anthropic Messages, a direct top-level tool-parameter
`type`, or an `anyOf`/`oneOf` composed entirely of explicit primitive types, guides conversion of
Qwen's untyped parameter text. It does not decide whether structurally complete markup is a tool
call. String-admitting values remain strings, including the empty string. An empty block for a
declared non-string parameter is omitted. Admitted JSON values retain their JSON type;
case-insensitive boolean text is normalized to `true` or `false`. A nonempty schema mismatch remains
a structured call: valid JSON retains its represented type and other text becomes a JSON string so
the tool consumer can report the validation error and continue the agent loop. Schemas without a
supported explicit type retain untyped inference. NInfer does not apply defaults, enforce required
properties, perform recursive JSON Schema validation, or use constrained decoding.

String parameters preserve function/tool-call markers and balanced nested
`<parameter=...>...</parameter>` text as value bytes. The Qwen wire format has no delimiter escape,
so an unmatched nested parameter opener or a standalone `</parameter>` cannot be represented
unambiguously; either causes the complete tool-call region to fall back to ordinary content.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInfer does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.

`--chat-template FILE` selects a local Jinja template; by default, the server uses the template
stored in the artifact. See the [CLI guide](cli.md#text-input) for an example.

Control-token spellings quoted in message content, tool data or ordinary template kwargs are
encoded as text. Media placeholders come from the template and bind to actual image/video inputs.

`chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
special tokens cannot be overridden through kwargs.

`--default-thinking-budget N` sets a positive default thinking-token cap for requests that start
in thinking mode. Non-thinking requests receive no cap. It may coexist with `--no-thinking`
because requests can explicitly enable thinking. Anthropic
`thinking:{"type":"enabled","budget_tokens":N}` overrides this default for that request.

Add `--default-thinking-budget 512` to the startup command to cap model-origin thinking at 512
tokens for every thinking-enabled request.

At the cap boundary, Engine first honors a natural `</think>`, stop condition, cancellation, or
total output/context limit. If thinking remains open, it commits Qwen's canonical early-close
guidance and close marker to the same model sequence without sampling, streams the guidance as a
reasoning delta, and continues normal content or tool-call generation. Inserted tokens count in
completion usage and the request's `max_tokens`/`max_output_tokens` budget. If the effective output
capacity extends past the cap but cannot fit the complete tokenizer-derived control suffix plus one
post-close model token, preparation is rejected with HTTP 400 code
`thinking_budget_capacity_insufficient` rather than partially inserting control. The server does
not promise that the model will emit nonempty content or a tool call after the marker.

For Chat Completions, `reasoning_effort: "none"` requests disabled thinking. The server folds the
other standard values onto the tiers the Qwen3.8 template distinguishes before rendering: `high`,
`max`, and `adaptive` render as `xhigh`; `minimal` renders as `low`; `low`, `medium`, and `xhigh`
pass through unchanged. Any other value is rejected with HTTP 400. The same folding applies to
`reasoning.effort` (Responses) and `output_config.effort` (Messages), and it works with published
artifacts because the template they embed is left untouched.
Conflicting explicit `enable_thinking` and effort values return `conflicting_template_option`.

`preserve_thinking` controls reasoning retention according to the selected template. Request
options override server defaults set with `--no-thinking` and `--preserve-thinking`. Unspecified
thinking, effort and preservation options use the template's defaults.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage. Aggregate and streamed usage include cached prompt tokens
and reasoning-token details; choices carry `logprobs: null` when log probabilities were not
requested, and aggregate assistant messages carry `refusal: null` because refusal output is not
supported.

### llama.cpp-compatible request observations

Every successful Chat Completions response includes a top-level `timings` object. This is a
llama.cpp-compatible response extension, not an OpenAI field. In a stream it is attached to the
last JSON chunk before `[DONE]`: the empty `choices` usage chunk when
`stream_options.include_usage` is true, otherwise the finish-reason chunk.

```json
{
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 83.0,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 129,
    "predicted_ms": 1140.0,
    "predicted_per_token_ms": 8.90625,
    "predicted_per_second": 112.28070175438596
  }
}
```

`cache_n` is the exact Engine-proven reused prompt prefix and `prompt_n` is the remaining prompt
suffix, so `cache_n + prompt_n` equals `usage.prompt_tokens`. Prompt time starts when admission
commits that exact reuse choice and ends when the first output token is committed. Generation time
starts at that first token and ends at the last committed output token. Accordingly, generation
speed uses `max(predicted_n - 1, 0)` token intervals; the first token belongs to prompt latency and
is not counted again as a decode interval. Zero-token, one-token, zero-duration, and exact-cache-hit
cases report finite zero rates rather than `NaN` or infinity. Speculative requests additionally
include terminal `draft_n` and `draft_n_accepted` when draft work occurred.

Set top-level `timings_per_token: true` on a streaming request to attach the latest cumulative
timing snapshot to each visible reasoning or content chunk. This does not enable terminal timings,
which are always present. A model commit that is temporarily hidden by UTF-8, stop-string,
reasoning, or tool-call buffering still advances the cumulative token count; the next visible chunk
observes that committed frontier. The option increases response serialization and transport volume
and is off by default.

Set top-level `return_progress: true` together with `stream: true` to receive prompt-processing
chunks:

```json
{
  "prompt_progress": {
    "total": 8192,
    "cache": 4096,
    "processed": 6144,
    "time_ms": 41
  }
}
```

The initial event has `processed == cache`. Later cumulative events are published only after the
corresponding prefill unit commits, may be coalesced when the consumer is slower than prefill, and
never move backwards. The final event has `processed == total` and precedes the first output delta.
For an exact full-prefix hit, the initial event already has `cache == processed == total` and no
synthetic prompt work is reported. `time_ms` is elapsed wall time since committed admission;
clients may calculate actual suffix progress as `(processed-cache)/(total-cache)` when the
denominator is nonzero.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit Engine `--max-context`. The current Vision runtime also
has a 32,768 merged-token envelope (131,072 raw patches); the effective Vision limit is therefore
`min(--max-context, 32768)`. There is no fixed image/video item-count limit: item count is admitted
through aggregate source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including
the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## OpenAI prompt caching

Chat Completions and Responses translate OpenAI cache hints into optional shared-prefix write
candidates:

- omitted `prompt_cache_options` creates a default implicit candidate at the latest representable
  content boundary;
- `mode:"implicit"` requests the same automatic candidate explicitly;
- `mode:"explicit"` disables that implicit write for the request;
- `prompt_cache_breakpoint:{"mode":"explicit"}` on supported content creates an explicit
  candidate.

One request carries at most four distinct writes. An implicit target occupies one slot unless it
coincides with an explicit target; the remaining slots contain the latest explicit boundaries.
Earlier schema-valid historical breakpoints are accepted but are not new write candidates. Exact
reads of already-published prefixes do not require the request to repeat a marker.

These fields are optimization hints. A legal boundary that cannot be represented as an exact
rendered-token frontier is ignored without changing prompt content. `prompt_cache_key` is not an
Engine session key or prefix identity. Valid TTL/retention values are accepted, but NInfer does not
promise their wall-clock residency; physical retention follows the resource scheduler.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
supported model instances use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with `--default-max-tokens` but remains `null` in the Response object |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `client_metadata` | Codex client extension; an object or `null`, accepted as opaque tracing metadata with no generation effect |
| `reasoning.effort` | `none` requests disabled thinking; other standard effort values fold onto the template tiers as for Chat Completions (`high`, `max`, `adaptive` render as `xhigh`; `minimal` as `low`) |
| `chat_template_kwargs` | template parameters as a JSON object; standard options merge with typed fields |
| `preserve_thinking` | alias for `chat_template_kwargs.preserve_thinking`; conflicting values are rejected |
| `text.format` | omitted or `{"type":"text"}` only |
| `tools` | direct function definitions or namespace groups containing function definitions; see below |
| `tool_choice` | `auto`, `none`, or function-only `allowed_tools` with mode `auto`; a namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` is accepted only when no effective tool is callable |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInfer does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted or an empty array |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, and explicit breakpoints follow [OpenAI prompt caching](#openai-prompt-caching); `safety_identifier` and `user` are accepted as client hints |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `refusal` | assistant-message replay part; its text enters assistant history |
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with `reasoning_text` content; summary/encrypted metadata may accompany raw text but cannot replace it |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |

Contiguous assistant-owned Items form one assistant history turn in the representable order
`reasoning` -> assistant message content -> `function_call`. Multiple message Items append their
content parts, multiple calls retain declaration order, and a reasoning-only turn is retained. A
user, system, developer, or `function_call_output` Item ends the group; an order that would require
rearranging assistant content fails with `invalid_assistant_history`. Results are validated by
`call_id` and reordered to call declaration order before prompt rendering; unknown, duplicate, or
unrepresentable partial result sets fail with `invalid_tool_history`. Canonical input Items retain
client order. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs
fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

An `input_text`, `input_image`, or tool-result part may carry
`prompt_cache_breakpoint:{"mode":"explicit"}`. Write selection follows
[OpenAI prompt caching](#openai-prompt-caching); boundaries affect reuse opportunities, not prompt
identity or output semantics. String message status/phase metadata is accepted but has no Qwen
prompt representation.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning metadata without raw
reasoning text, partial tool Items, and other Item/content types are not supported. HTTP media URLs
stored in a response chain are fetched again when that chain is continued; use data URIs when the
historical media bytes must be immutable.

### Function tools

Responses function definitions may be declared directly rather than inside Chat Completions'
nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

They may also be grouped in a Responses namespace:

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```

NInfer gives each namespace/function pair a distinct internal Engine identity and restores the
separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The same
function name may therefore appear in different namespaces. Namespace members remain ordinary
client-executed functions; this does not add a remote MCP executor.

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. Only functions in the current effective tool set can become structured calls;
undeclared model output remains ordinary text. `allowed_tools` with mode `auto` filters that set
without changing declaration order, while `tool_choice:"none"` disables structured tool output even
when the history contains earlier calls.

NInfer does not execute functions or enforce JSON Schema through constrained decoding, so
`strict:true`, required or named tool choice, hosted tools, remote MCP tools, and custom free-form
tools are rejected. Deferred loading, output schemas, and caller restrictions that exclude direct
invocation are also rejected because their semantics cannot be honored.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` Item containing raw `reasoning_text` and an empty summary;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses. `completed_at` is populated only for completed Responses. A
reasoning-only incomplete result contains no invented empty assistant message.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
checkpoint-proven prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous `<tool_call>` suffix or the structured tool region is held.
Malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so compatible
checkpoint reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
checkpoint already proved compatible by the model runtime.

For Engine-local reuse, a stored root Response receives one bounded session key derived from its
response ID, and every `previous_response_id` child inherits that key. `store:false` roots remain
anonymous; a `store:false` child may read its inherited session checkpoint but does not replace the
stored chain's latest endpoint. Response-store eviction or deletion removes the HTTP object, not an
independently retained Engine checkpoint; the latter remains bounded by the Engine's own retention
and pressure policy. No session key or cache marker is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found`; stream recovery and non-empty `include` are rejected rather than ignored |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`); image URLs are redacted unless `include=message.input_image.image_url` |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` uses the same prompt path as Create and does not run generation.
It accepts `model`, `input`, `instructions`, `previous_response_id`, reasoning, function tools and
tool choice, supported text/truncation values, and the `preserve_thinking` extension. Parent lookup,
call-ID normalization, template rendering, and media expansion are therefore identical to the
corresponding Create request:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, Structured Outputs/JSON mode, non-empty `include`, background execution, compaction,
files/audio, and OpenAI-hosted/MCP/custom tools. These are compatibility boundaries, not silently
accepted placeholders.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint accepts top-level System text, ordered User/Assistant/System history, text and image
blocks, Thinking history, tool-use history, tool results, user-defined tools, aggregate responses,
and Anthropic SSE. Consecutive User or Assistant messages are joined without adding separators.
Mid-conversation System messages retain their input position. A final text-only Assistant message
is an Assistant prefill: generation continues its existing text instead of opening another turn.
Assistant prefill cannot contain media, Thinking, or tool calls and cannot start with Thinking
enabled.

Claude Code may place its attribution metadata in the first block of a top-level System array. If
that block is a text block beginning exactly with `x-anthropic-billing-header:`, NInfer consumes the
whole block before token counting, prompt preparation, and cache identity construction. The rule is
positional: a string-form System value, a later array block, or an inline System message with the
same text remains ordinary prompt content. A `cache_control` marker attached to the consumed block
is consumed with it rather than moved to adjacent content.

`max_tokens` is optional for local clients and otherwise uses `--default-max-tokens`; a positive
value is the complete output budget. `max_tokens:0` is rejected because NInfer does not expose a
completed zero-output cache-prewarm lifecycle. `temperature`, `top_p`, `top_k`, and
`stop_sequences` enter Engine execution. A matched custom stop is returned as
`stop_reason:"stop_sequence"` together with the actual `stop_sequence`; context exhaustion returns
`model_context_window_exceeded`.

Thinking supports `disabled`, `adaptive`, and `enabled`. Enabled Thinking requires
`budget_tokens >= 1024` and less than `max_tokens`, and that budget is passed to Engine. Visible
Thinking is returned with an opaque compatibility signature; SSE emits its `signature_delta`
before closing the block. Request lowering reconstructs the local prompt from the visible
`thinking` text and treats `signature` as non-semantic transport metadata, so retained history
remains usable across serve restarts.

`thinking.display:"omitted"`, which current Claude Code sends on every request, returns each
Thinking block with an empty `thinking` string and no `thinking_delta` events. Its `signature`
(`signature_delta` when streaming) is `ninfer-reasoning.v1:` followed by the Base64 reasoning text,
so the reasoning stays out of the visible transcript but comes back with the block the client
echoes. When a replayed Thinking block has empty `thinking` and such a signature, request lowering
restores the reasoning from it, so the prompt (and its prefix-cache identity) is the one
`summarized` display would give; non-empty `thinking` text takes precedence, any other signature
stays non-semantic metadata, and a malformed NInfer signature fails with
`invalid_thinking_signature`. The signature is not encrypted or authenticated: this is a local
trusted server, and the value is the same reasoning text `summarized` display shows. Count Tokens
ignores `display`. `preserve_thinking` remains a NInfer extension for closed-turn reasoning
history. `output_config.effort` folds onto the template tiers as for Chat
Completions: `high` and `max` render as `xhigh`, `low` and `medium` pass through unchanged.

User-defined, non-strict tools support `name`, `description`, object `input_schema`, and
`input_examples`. `tool_choice:auto` and `none` are executable. Forced or named choice,
`strict:true`, active single-call enforcement, tools that exclude direct model calls,
Anthropic-provided/server tools, toolsets, MCP, and containers are rejected because their required
constraint or executor is absent. Client-side tool search (Claude Code's `ToolSearch`) is
accepted: a `defer_loading:true` tool stays out of the prompt until a `tool_reference` block in
some `tool_result` content names it via `tool_name`; from then on it renders in the tool list like
any other tool, and the block itself renders nothing. A `tool_reference` whose `tool_name` is
missing, malformed, or not declared in `tools` fails with `invalid_tool_reference`; a top-level
`tool_reference` (Anthropic's server-side search) stays rejected. The tool list sits at the start
of the Qwen system prompt, so a newly loaded tool changes the prompt from there on: the next
request re-prefills the whole conversation instead of reusing the cached prefix (Anthropic expands
the reference inline, which keeps the prefix). `tool_result` preserves text/image order and marks
`is_error:true` explicitly in the model prompt. For a visible Assistant tool-use turn, the next
User turn must provide exactly one leading result for every declared ID; valid results are matched
by ID and normalized to call order. A history that begins with results remains valid as a truncated
or imported conversation.

Block-level ephemeral `cache_control` on tools and supported System/User/Assistant/tool-history
blocks creates explicit shared-prefix candidates. At most four distinct block-level breakpoints are
accepted. Request-level `cache_control` targets the last cacheable block: it merges with an explicit
breakpoint at the same target and TTL, conflicts at the same target with a different TTL, and needs
an available fifth slot when four different explicit targets already exist. TTL must be `5m` or
`1h`; it is a protocol hint, not a wall-clock residency guarantee.

NInfer maps representable boundaries to exact prompt frontiers and ignores a legal but
unrepresentable advisory boundary without changing the prompt. Reuse still requires exact rendered
identity and can read an existing owner without another `cache_control`. Aggregate usage reports
verified reused tokens in `cache_read_input_tokens` and leaves cache creation unknown. Streaming
emits `message_start` after Engine admission commits the prefix selection and before
transfer/prefill output, so its uncached/cache-read split is already exact; terminal cumulative
usage matches the aggregate response.

Documents, Search Results, Files, Structured Outputs, server-tool results, container uploads, and
other execution-dependent blocks are rejected with the missing capability identified. Metadata,
service tier, inference geography, protocol-version/beta headers, cache TTL, and unknown advisory
fields do not block an otherwise executable request. The request `model` is any non-empty local
proxy label and is echoed in the response; it does not select the resident artifact.

Every Messages response carries a `request-id` header; error bodies also carry `request_id` and use
Anthropic error categories. Local admission overload maps to HTTP 529 and queue/media timeouts to
HTTP 504. Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without generation. It shares the same prompt normalization, tools, Thinking mode, Assistant
prefill, media processing, and cache-marker interpretation as Messages; output-only sampling and
streaming fields do not affect the count:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

The table lists executable defaults. The startup example selects a long-context FP8/MTP3 profile.

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `identity.model_id` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--vram-headroom-mib N` | VRAM in MiB that `--kv-capacity auto` leaves free after sizing the KV pool; requires `auto` | `1024` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `30000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | pretty stderr verbosity | `info` |
| `--device N` | CUDA device index | `0` |
| `--tp 1\|2` | tensor-parallel width; see [Two GPUs](#two-gpus) | `1` |
| `--devices A,B` | one CUDA device per rank, rank 0 first; required with `--tp 2` | `--device` |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--log-speculation-detail` | add the per-step accepted draft lengths and the output/prompt 4-gram overlap to every `request_done` record (test measurement; requires `--request-log-jsonl`); see [Structured request log](#structured-request-log) | off |
| `--log-speculation-detail-max-steps N` | decode steps recorded per request by `--log-speculation-detail` (`1..1048576`); a longer request is marked truncated | `4096` |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4` | KV-cache storage | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash/DFlash2 `1..15` | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--default-max-tokens N` | output limit when omitted by a request | `8192` |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--vision-device N` | CUDA device that holds the Vision tower and encodes; one of `--devices` at `--tp 2`; see [Two GPUs](#two-gpus) | `--device` |
| `--max-vision-tokens N` | merged Vision tokens of one image or video item (`64..16384`); larger media are resized and the encode workspace is planned for `N` | `16384` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-tp-mailbox` | keep the captured `--tp 2` all-reduces on cross-device copies; see [Two GPUs](#two-gpus) | mailbox on without P2P |
| `--embedding-host` | keep the token embedding table in one copy in pinned host memory that every GPU reads in place over PCIe, instead of one copy per GPU: frees its VRAM (1.18 GiB per GPU for Qwen3.8-27B) for the KV pool, identical results; see [Two GPUs](#two-gpus) | off |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--device-state-slots N` | extra Device checkpoint StateImages beyond the active-lane guarantee | `max-concurrency`; `max(2 * max-concurrency, 8)` at `--tp 2` |
| `--host-state-slots N` | pinned Host StateImage capacity | `8`; at `--tp 2` `0` unless set (`--profile 27b`, `--profile flash-next`: `8`) |
| `--host-kv-mib N` | shared pinned Host Main/Backend KV byte capacity in MiB | `8192`; at `--tp 2` `0` unless set (`--profile 27b`: `8192`, `--profile flash-next`: `4096`) |
| `--max-private-continuations N` | private continuation descriptor capacity | `2 * max-concurrency`; `max(2 * max-concurrency, 8)` at `--tp 2` |
| `--max-shared-prefixes N` | Engine-wide shared stable-prefix descriptor capacity | `max(max-concurrency, 4)` |
| `--max-long-anchors-per-continuation N` | private long-anchor limit per continuation | `2` |
| `--turn-anchors N` | private long-anchor opportunities at the last N message boundaries before the final message, so a cancelled or edited turn resumes from the end of the previous turn instead of re-prefilling the conversation (`0` disables). Default by model: `0` on Qwen3.8-Flash-Next (each anchor costs one extra prefill pass per turn, ~0.8 s there), `1` on the others, where the extra pass is not measurable | `0` Flash-Next, `1` others |
| `--prefill-miss-dump DIR` | diagnostic for prefix-reuse misses: Serve keeps the token contexts of its last 16 requests (prompt plus committed output, cancelled requests included); a request that recomputes at least `--prefill-miss-min-tokens` prompt tokens and shares at least `--prefill-miss-min-match` of its prompt with the closest of them is dumped when the server recomputed that many tokens identical to it, or when that many of its tokens follow the divergence point (the client changed or dropped earlier text). Each dump is a new directory under `DIR` with `new_prompt.txt`, `old_context.txt`, both token-ID lists and `summary.json` (token counts, divergence point with the text on both sides, and the reading); the request also logs a `prefill miss` warning. At most 64 dumps per process. The files contain conversation text verbatim | off |
| `--prefill-miss-min-tokens N` | threshold of `--prefill-miss-dump` in tokens | `10000` |
| `--prefill-miss-min-match F` | minimum share of the prompt matching a recent context for `--prefill-miss-dump` (`0..1`) | `0.5` |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--cors` | permissive browser CORS headers | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override (`0..20`; zero selects the top-20 cap) | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

Context-cost coefficients resolve once at startup from generic defaults, matching compiled values,
and optional transfer or prefill entries from `--context-cost-presets FILE`. Prefill entries match
the hardware and a signature derived from the actual Text/Vision configuration, bindings and Uses.
A new representation without a matching measurement uses generic prefill coefficients. A malformed
file aborts startup; the operational context-cost record and JSONL `server_start` identify the
selected source.

Engine selects sampling defaults from the loaded architecture and the request's resolved thinking mode.
Qwen3.6-27B and Qwen3.8-27B use `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/1.5` in
non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`:
`C` slots guarantee active requests and `H` is a global checkpoint pool. Host State and Host KV are
independent startup-fixed pinned-memory capacities; Host KV is shared by Main and the selected
Backend pool and is consumed in physical page extents. `--no-prefix-reuse` selects root-only Engine
mode and cannot be combined with any of the seven explicit context-cache capacity flags, including
zero-valued flags.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

### Two GPUs

`--tp 2 --devices A,B` serves a dense artifact split across two GPUs, for a model that does not fit
one device:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 32768 --kv-capacity auto \
  --max-concurrency 2 --kv-dtype int8
```

Rank 0 runs on `A` and owns admission, the context cache bookkeeping and sampling; rank 1 holds the
other half of every attention and Gated DeltaNet head group, MLP intermediate width and output-head
vocabulary, plus its half of the KV pages and recurrent state, whose allocation and checkpoint
copies follow rank 0's. Both ranks reserve the same runtime layout, and `--kv-capacity auto` sizes
it from the rank with less free memory. Prefix reuse, concurrent requests and CUDA Graph decode
work as on one GPU.

Every layer ends in two cross-device all-reduces. Prefill and eager decode move them with
stream-ordered device-to-device copies, which the driver stages through host memory when the GPUs
have no peer access. In CUDA Graph decode, an all-reduce of one request's activation (the hidden
state across the verified columns) instead runs one kernel per GPU that exchanges the two halves
through a small pinned host mailbox, which costs far less than the staged copies' event chain;
wider multi-request payloads keep the copies. Both transports give identical results.
The mailbox is used only when the two GPUs have no peer access (the P2P line of the startup log
says which); with direct P2P the copies stay the transport. `--no-tp-mailbox` keeps every
all-reduce on the copies, for comparison. If a mailbox exchange ever waits too long for the other
GPU, the two ranks' results have diverged: the round fails with an error, and so does every later
round, so the server must be restarted.

At `--tp 2` the Host tiers hold both ranks' halves of a checkpoint: each pinned Host record keeps
rank 0's half followed by rank 1's, and each GPU copies its own half on its own transfer stream, so
the RAM cost per token is the same as on one GPU. They are off unless asked for: an omitted
`--host-state-slots` or `--host-kv-mib` becomes `0` (on Qwen3.8-Flash-Next the routed experts
already pin most of the RAM and the PLE table is read through the page cache); `--profile 27b`
sets `--host-state-slots 8 --host-kv-mib 8192`, `--profile flash-next` sets `--host-state-slots 8
--host-kv-mib 4096` (884 MiB of pinned StateImages on Flash-Next: without a Host StateImage tier, a
capture that finds the Device StateImages full drops a whole idle conversation, so an agent that
runs subagents in parallel re-prefills the smaller one from the root at almost every turn).
With them, a checkpoint that no longer fits on the GPUs moves to RAM and comes back when its
conversation resumes, as on one GPU.

With `--elastic` (Qwen3.8-Flash-Next) and a Host KV tier, the GPUs keep only the KV of the
requests being answered: once a request decodes, the KV of every idle retained conversation moves
to RAM, the remaining KV pages are packed at the start of the pool, and the VRAM they leave goes
back to routed-expert slots at once (the log line `idle KV to Host` reports each move). A
conversation that resumes is copied back to the GPUs before its prompt, taking the slots it needs
then. When the Host KV tier has no room for a conversation, idle conversations already in RAM are
dropped, least valuable first; one that still does not fit stays on the GPUs. The flash-next default
of 4 GiB holds about two idle conversations of 80k tokens (22.5 KB of KV per token over both
GPUs, Main and MTP); each GiB is pinned and comes out of the page cache the CPU experts' PLE rows
are read from (measured on the T71 agentic replay: 4 GiB cost +0.8 GB of disk reads and +0.6 s of
PLE gathers over 53 turns, against about 5 s of decode saved; 8 GiB added the same again with no
further gain). Without them, every
checkpoint lives in a Device StateImage, and the defaults raise the Device checkpoint pool
and the private catalog to `max(2 * max-concurrency, 8)`. Without a Host tier, a private checkpoint
captured during prefill (the turn-closure point a follow-up turn resumes from, a long-prompt anchor)
that finds the Device pool full first releases idle retained conversations without a live session,
least valuable and oldest first, until it fits; each release counts as a private eviction. A capture
that still does not fit is skipped, and that conversation's next turn re-prefills from an earlier
checkpoint or its whole prompt. On one GPU the behaviour is unchanged: with `--host-state-slots 0`
a capture that finds the Device pool full is skipped without releasing anything.
Each extra slot costs one StateImage per rank (about 73 MiB for Qwen3.8-27B); agentic or
multi-conversation servers should set `--device-state-slots 12` to `16` when memory allows (on two
16 GB boards at 196,608 tokens with Vision only 4 fit with the official NVFP4 artifact; see the
limitations in the README). The startup log prints one
line per rank and a `tensor parallel` capacity line with the Device checkpoint pool and the
transfer path: `p2p on` when the driver grants direct peer access, `p2p off (host-staged copies)`
otherwise (GeForce boards). With CUDA Graphs one `cuda graphs` line per rank compares the device
memory graph preparation took with the planned per-device allowance, and a warning follows if a
rank took more; startup still succeeds, the excess coming out of the memory KV sizing left free.
The tp 2 allowance is three times what two RTX 5070 Ti measured at 32K context and
`--max-concurrency 1`, at least 8 MiB: 8 MiB for ordinary decoding and for MTP, 11 MiB per graph
class for DFlash2 (55 MiB for its five classes up to 32K), each multiplied by
`--max-concurrency`.

MTP speculative decoding works at `--tp 2` with the same options as on one GPU:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 32768 --kv-capacity auto \
  --max-concurrency 2 --kv-dtype int8 \
  --spec mtp --draft-tokens 3
```

The MTP head is split across the ranks like a Text layer, verification and the recurrent-state
commit run on both, and acceptance and sampling run on rank 0; `--lm-head-draft` keeps the
optimized proposal head on rank 0 alone. Prefix reuse works as on one GPU: each rank keeps its own
copy of the hidden state a retained prefix ends with, and the MTP head resumes from it on both.

DFlash2 works at `--tp 2` with the optimized proposal head:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 32768 --kv-capacity auto \
  --max-concurrency 2 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 4 --lm-head-draft
```

The drafter and the optimized proposal head run on rank 0 alone, and rank 0 alone holds the
drafter's context and StateImage rings, so rank 0 has less free memory than rank 1 and
`--kv-capacity auto` sizes the KV pool from it. Rank 1 receives the draft tokens, verification and
the recurrent-state commit run on both ranks, and acceptance and sampling run on rank 0.
`--lm-head-draft` is required: the full output head is split by vocabulary across the ranks,
while the drafter ranks its candidates over one complete head.

`--vision` works at `--tp 2` with every supported decoding mode. The Vision tower is loaded on one
GPU only, `--vision-device` (default: rank 0's device, `A`); that GPU encodes each image or video
item once and copies the merged embeddings to the other GPU, and both ranks then prefill the prompt
with the same embeddings:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 --vision --vision-device 1 --max-vision-tokens 4096 \
  --max-context 32768 --kv-capacity auto \
  --max-concurrency 2 --kv-dtype int8 --spec mtp --draft-tokens 3
```

The copy goes through host memory without peer access, like the all-reduces, and costs about
10 KiB per Vision token. Only the tower's GPU holds the Vision weights and the encode workspace; the
other GPU holds just the handoff buffer the embeddings land in (`hidden x 2` bytes per token of the
largest item). `--kv-capacity auto` credits the other GPU with the encode workspace it does not
allocate, so pick the GPU with more free memory (the one without a display) as `--vision-device`.
The encode workspace grows with the item extent: `--max-vision-tokens` below the `16384` default
resizes larger images and videos to that many tokens and shrinks the workspace accordingly. MTP
and DFlash2 compose with Vision as on one GPU: the MTP head and its prefix-reuse bridge read the
visual columns on rank 0, where its token embedding lives, and the DFlash2 drafter reads rank 0's
target features.

`--embedding-host` moves the token embedding table out of both GPUs into one copy in mapped
pinned host memory, read in place by the gathers of both ranks (prefill, decode, MTP and DFlash2,
inside the CUDA Graphs too). The table is the same bytes, so outputs are identical; each GPU gets
its size back (1.18 GiB for Qwen3.8-27B, whose table is FP8 with a scale per row) for the KV pool,
and the server pins as much host RAM once. The rank lines of the startup log then show the table
as `host-mapped` instead of `replicated`, followed by a `host-mapped weights` line. It also works on
one GPU. A gather reads 5 KiB per token over PCIe: a few rows per decode round, 5 KiB per prompt
token in prefill.

Tensor parallelism covers ordinary decoding, `--spec mtp` and `--spec dflash2 --lm-head-draft`,
each with or without `--vision`, with `bf16`, `int8` or `nvfp4` KV. `--spec dflash`, `--spec dflash2` without
`--lm-head-draft` or with a drafter that has full-attention layers, the MoE architecture and the
`fp8` and `k8v4` KV types are rejected at startup (`nvfp4` is accepted since v0.4.2, see [Two GPUs](cli.md#two-gpus)), and so is a `--vision-device` outside
`--devices`.

Serve writes human-readable operational records to stderr using
`YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message`. Normal output covers material startup milestones,
readiness, request lifecycle, fixed-interval throughput, and shutdown; `--log-level debug` exposes
internal startup and resource-planning detail. A terminal may use one transient line during startup,
but Serve throughput is always a persistent record. Redirected stderr contains no terminal control
sequences. Pretty values use readable units and rounded rates; use the independent request JSONL for
complete fields and full precision. Operational records never contain prompts, generated text,
request bodies, credentials, or arbitrary client error messages.
If a tool marker is returned to text because its structure or tool identity cannot be represented,
Serve emits one warning with only the failure classification, never the generated markup.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Every line is one `ninfer_serve_request_log` schema-v27 JSON object. All events carry
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
clients send `<kind>/<tool>` labels: `gate/tp2` (`tools/tp2/gate_client.py`),
`bench/serve-corpus`, `bench/serve-concurrency`, `bench/serve-ttft`, `bench/agentic-ab`,
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
preserved for consumer validation, and a stable text-fallback reason. Fallback reasons are `none`,
`malformed_structure`, `duplicate_parameter`, `invalid_tool_name`, `undeclared_tool`, and
`trailing_content`. These counters contain no tool arguments or generated text.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, `accepted_per_position`, and
`accepted_length_histogram`. A round is one verification of a nonempty draft; `fallback_steps` are
decode steps without a draft. `accepted_per_position[i]` counts the rounds that accepted at least
`i + 1` draft tokens (the cumulative form), and `accepted_length_histogram[i]` the rounds that accepted
exactly `i`, for `i = 0..draft_window`; it sums to `rounds` and is computed from
`accepted_per_position` and `rounds`, so the two never disagree. Both are empty without a
speculative backend. A round whose draft was cut short by the remaining output budget counts at
the length it actually accepted. Rates can be derived downstream from raw token counts and seconds
instead of rounded stderr strings.

`speculative.accepted_lengths` is the order behind that histogram, recorded only with
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
For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected tokens.
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

## Execution behavior

The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation
requests. At each decode boundary, every decode-ready request is compacted into one batch and
processed by one model traversal and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
There is no admission ETA or unbounded overflow queue.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media requests additionally share one preparation permit, so a waiting
media request retains the same cancellation and timeout deadline. Model output is bounded by the
same finite request count and each request's effective output-token limit; output callbacks and
network serialization run outside the GPU executor and do not delay formation of the next batch.

`--max-context` is each sequence's logical ceiling. `--kv-capacity` fixes the shared Main Text KV
pool used by active requests and retained prefixes. `auto` accounts for the complete enabled runtime
and leaves 1 GiB of sizing headroom (`--vram-headroom-mib` changes that margin); omitting the option makes it follow `--max-context`. Capacity
resolves once at startup.

Admission reserves the full prompt-plus-effective-output page entitlement through request
completion. A request remains queued until a legal resource plan can satisfy that entitlement.

Each reusable checkpoint contains KV and complete continuation state. At admission, capture, and
finish boundaries, resource pressure may keep it on Device, move its StateImage and/or KV replicas
to pinned Host memory, or evict it. The planner compares incoming-request work with the later
recovery cost imposed on retained checkpoints. Active requests retain their state and completion
reservations, and placement choices preserve model semantics. The full policy and invariants are
defined in [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The pretty completion record
shows `cache N (P%, path)` using readable path labels; JSONL retains the exact
`prefix_cache_hit_tokens` and `prefix_reuse_path` fields. Machine paths are `root`,
`private_endpoint`, `private_turn_closure`, `private_response_replay`, `private_long_anchor`, and
`shared_stable_prefix`. Reuse validation covers KV, recurrent state, hidden state, selected-backend
state, and the exact prompt frontier. With stable `preserve_thinking=true`, the auxiliary checkpoint
rolls to the message frontier immediately before the current response's deterministic generation
prologue. A normalized response, compact-summary instruction, or replacement user suffix therefore
replays the small generation prologue and only the changed suffix while retaining the complete
stable conversation prefix. Stable `false` places the turn-closure checkpoint before the first
assistant opener in the open turn, so closing that turn can recompute its opener and omit its
reasoning without discarding the preceding conversation.

`preserve_thinking` selects the capture frontier for newly created checkpoints. Existing exact
checkpoints remain reusable across a mode change. If the desired boundary is behind the selected
reuse frontier and has no snapshot, the Engine keeps the valid hit and defers the new checkpoint. A
later request that diverges before every retained checkpoint starts from root. The JSONL completion
record exposes the restored checkpoint as `prefix_reuse_path`. Reasoning-effort changes participate
in rendered-token identity and exact-prefix selection.

An appended mid-conversation system message is an ordinary prompt suffix, so an unchanged prior
history remains eligible for `private_endpoint`. If the client modifies, removes, or moves a
historical system message, the token prefix genuinely differs and a miss/reset is correct.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools and does not enforce client JSON Schema through constrained
decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
