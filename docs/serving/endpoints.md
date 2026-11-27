# Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | Engine readiness |
| `GET /stats` | live rates, active and finished requests, and the last hour, for a dashboard ([below](#live-statistics)) |
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
| `POST /unload` | free the GPUs: tear the Engine down inside the server (`--unload-endpoints`) |
| `POST /load` | build the Engine again (`--unload-endpoints`) |

`GET /health` returns HTTP 200 with `{"status":"ok"}` while the Engine can accept work. After an
Engine-wide failure it returns HTTP 503 with `{"status":"unavailable"}`. Temporary queue
saturation does not make the Engine unavailable. The endpoint remains unauthenticated. With
`--unload-endpoints` or `--idle-unload-seconds` the body also carries the Engine's state, `"engine"`:
`loaded`, `unloading`, `unloaded` or `loading`; an unloaded Engine is a healthy server (HTTP 200),
and the next generation or token-count request loads it.

## Live statistics

`GET /stats` answers what the server is doing now and did over the last hour, as one JSON object, for
a dashboard or a monitor that polls it (twice a second is cheap). It requires the API key when one
is set. Reading it never waits for the Engine worker: everything comes from the counters the Engine
publishes at its step boundaries, the GPUs' NVML telemetry (read once a second when the driver's
`libnvidia-ml.so.1` is there), the host's `/proc` and `/sys` (once a second) and the request
lifecycle the server already logs.

`now` is the present:

- the Engine's state; requests `running` (`prefilling`, `decoding`), `waiting` and
  `materializing`; `unit`, the execution unit running at this instant and for how long (a prefill
  chunk can run for seconds and publishes its tokens only when it ends);
- `rates` over the last `1s`, `10s` and `60s`: prefill and decode tokens per second, decode rounds
  per second, the mean decode batch, tokens per decode step, the decode step's time and its host
  share, the prefill unit's time, `draft_acceptance` (drafts accepted over drafts verified),
  accepted drafts per round and the mean draft window, the prefix-cache hit ratio (prompt tokens
  reused over reused and computed), KV and state bytes copied to and from the GPUs per second, and
  for a model with offloaded experts the expert-slot hit ratio (all windows, and decode alone),
  experts copied into slots (count and bytes) and missing experts the CPU and the GPU computed per
  second;
- `kv`: device KV pages in use and reserved, the pages elastic KV backs now, the capacity
  (`page_tokens` per page), state slots, and the host KV tier's bytes and capacity;
- `prefix_cache`: reused and computed prompt tokens since start, how each admission reused its
  prefix (`selections`), and what memory pressure degraded, evicted or dropped;
- `speculation`: rounds, drafted, accepted, fallback steps, and `by_position`, how often each draft
  position was reached (every earlier draft accepted) and accepted;
- `experts` (`null` without offloaded experts): the slot cache since start, its base slots and
  elastic columns, and the decode windows' miss split;
- `transfers`: bytes copied since start, by kind and direction;
- `gpus`: per GPU, utilization, memory, power and its limit, temperature, fan, clocks and the active
  slowdown reasons (`gpu_idle` left out);
- `host`: how busy the CPUs are and how many of them this process keeps busy (the routed experts
  that are not in a GPU slot are computed there), the CPU package and hottest-core temperature
  (Intel `coretemp`, AMD `k10temp`), the mean CPU frequency, and memory: available, swap in use and
  the process's resident bytes;
- `requests`: requests done, failed by class and rejected by error code since the server started.

`requests` lists every request between its start and its answer, with the Engine's live progress
joined to it: `phase` (`queued`, `waiting`, `materializing`, `prefill`, `decode`, `finishing`),
prompt and cached tokens, `prefill_progress` over the tokens to compute, output tokens, time to
first token and the decode rate so far. `finished` is the last 100 requests, newest first, with the
figures of their `req#N done` log line, or the error of a failed or rejected one.

`history` is one point a second for the last hour, column-wise: `t` in Unix seconds, prefill and
decode tokens per second, decode batch, draft acceptance, decode step time in milliseconds, prefix
and expert-slot hit ratios, expert fill and KV transfer bytes per second, running and waiting
requests, device KV pages, CPU utilization, temperature and the CPUs this process keeps busy, and
per GPU its utilization and power. `next` is the sequence number of
the newest point; `GET /stats?since=N` returns only the points after `N`, so a page fetches the
hour once and then only what is new. A restarted server numbers its points from 1 again.

The Engine's figures restart with it after `POST /unload` and `POST /load`; no rate spans the two.

## Unload and load

For a machine that shares its GPUs with another program (image generation, say), the Engine can be
torn down inside the running server and built again later, the HTTP server keeping its port, API
key and queue (opt-in: `--unload-endpoints`, `--idle-unload-seconds`; [options](options.md)).

- `POST /unload` writes the retained conversations to the disk tier of the context cache
  (`--disk-cache-dir`, within `--disk-cache-flush-seconds`), then frees the Engine: every device
  allocation, its threads, and its page-locked host memory but the host-mapped weights (Flash-Next's
  expert banks, the `--embedding-host` table), which stay page-locked for the next load unless
  `--unload-release-host`. The CUDA contexts stay, a few hundred MiB per GPU. Answers
  `{"engine":"unloaded","seconds":S}`, or `{"engine":"unloaded"}` when it already was; HTTP 409
  (`engine_busy`) while a request holds the Engine, from its submission until its response ends.
- `POST /load` builds the Engine the way start-up does, with the same options, and warms it up;
  answers `{"engine":"loaded","seconds":S}`. With `--load-min-free-mib` it first checks the free
  memory of every device and answers HTTP 503 (`insufficient_device_memory`) instead of failing
  during the load, after waiting up to `--load-wait-seconds` for the memory. A load sizes the KV
  capacity (when automatic) and the expert slots from the free memory: with `auto` a load may build
  down to half the first Engine's expert slots, so the other program's leftover CUDA context does
  not block it. A load that fails anyway answers HTTP 500 (`engine_load_failed`) and leaves the
  Engine unloaded, the server serving.
- A generation or token-count request to an unloaded Engine loads it first; the load does not count
  against its `--pending-timeout-ms`. When that load is refused or fails, the request answers HTTP
  503 (`service_unavailable`).
- `--idle-unload-seconds N` unloads after N seconds without a request.

Both endpoints require the API key when one is set. A reloaded Engine continues a conversation from
the disk tier when it was written there; the in-memory context cache does not survive the unload.

An Engine-wide failure (an error thrown by the Engine worker) is permanent for the process:
`ninfer-serve` logs it, stops accepting connections within
about 250 ms and **exits with status 2** (startup failures exit with 1; a failed CUDA call aborts
the process). Run it under a supervisor that restarts on a non-zero exit — `Restart=on-failure` in
systemd, or llama-swap, which reloads the model on the next request — instead of relying on
`/health`, which supervisors usually probe only while the model loads.

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract. Every error answer
(4xx and 5xx, a wrong API key included) also writes one line to the server log with that id, the
status, the error code and, for a client error, the top-level field the server did not accept (never
the message or the request's content), so a client integration can be followed from the log.

An unexpected server failure is HTTP 500 with error type `server_error` on the OpenAI endpoints
(Chat Completions, Responses, models) and `api_error` on the Anthropic endpoints.

All three generation SSE endpoints emit the standard `: keep-alive` comment after five seconds
without a protocol event. The comment is transport-only: SSE clients ignore it, and it does not
change generated text, event ordering, usage, stored Responses, or request logs. On Linux, accepted
connections also use TCP keepalive and a 15-second `TCP_USER_TIMEOUT`; together with the heartbeat,
a dead or unacknowledging peer is normally cancelled within about 20 seconds, including while the
request is waiting or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream NInfer
connection when the downstream client disappears.
