# Endpoints

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
