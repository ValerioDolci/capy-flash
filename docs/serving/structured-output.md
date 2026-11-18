# Structured output

A constrained request limits the response content to a language: one JSON Schema, any JSON object,
a GBNF grammar, a regular expression, or a list of literal strings. Before every decode round a
per-request matcher (XGrammar, vendored in `third_party/xgrammar`) computes the tokens that keep the
content in the language; sampling, MTP verification and the bonus token draw only from them, so a
response that ends with the model's stop token is in the language. Requests without a constraint run
exactly as before.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Weather in Rome, as JSON."}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "weather", "schema": {
      "type": "object",
      "properties": {
        "condition": {"enum": ["sunny", "cloudy", "rain"]},
        "celsius": {"type": "number", "minimum": -60, "maximum": 60}
      },
      "required": ["condition", "celsius"],
      "additionalProperties": false}}}
  }'
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Is the sky blue?"}],
       "structured_outputs": {"choice": ["yes", "no"]}}'
```

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "max_tokens": 512,
       "messages": [{"role": "user", "content": "Invent a user, as JSON."}],
       "output_config": {"format": {"type": "json_schema", "schema": {"type": "object",
         "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
         "required": ["name", "age"], "additionalProperties": false}}}}'
```

## Request fields

| Endpoint | Field | Constraint | Error `param` |
|---|---|---|---|
| Chat Completions | `response_format: {"type":"json_object"}` | any JSON object | `response_format` |
| Chat Completions | `response_format: {"type":"json_schema","json_schema":{"name":N,"schema":S}}`, optional `description`, `strict` | schema `S` | `response_format.json_schema.schema` + JSON Pointer |
| Responses | `text.format: {"type":"json_object"}` | any JSON object | `text.format` |
| Responses | `text.format: {"type":"json_schema","name":N,"schema":S}`, optional `description`, `strict` | schema `S` | `text.format.schema` + JSON Pointer |
| Messages | `output_config.format: {"type":"json_schema","schema":S}` | schema `S` | `output_config.format.schema` + JSON Pointer |
| all three | `structured_outputs: {"grammar": G}` (NInfer extension) | GBNF `G`, root rule `root` | `structured_outputs.grammar` |
| all three | `structured_outputs: {"regex": R}` | regular expression `R`, whole content | `structured_outputs.regex` |
| all three | `structured_outputs: {"choice": [...]}` | exactly one of the strings | `structured_outputs.choice` |

OpenAI `name` is 1 to 64 letters, digits, `_` or `-`. `strict` may be `true`, `false` or `null`: the
schema is enforced either way. Responses returns `text.format` as received, in the Response object
and in `response.created`. The token-count endpoints accept the format fields and ignore them.

JSON Schema: drafts 2020-12 and 07, keywords `type`, `properties`, `required`,
`additionalProperties`, `items`, `prefixItems` (`additionalItems` in draft 07), `minItems`,
`maxItems`, `minLength`, `maxLength`, `pattern`, `minimum`, `maximum`, `exclusiveMinimum`,
`exclusiveMaximum`, `const`, `enum`, `anyOf`, `oneOf`, `allOf`, local `$ref` with `$defs` or
`definitions`; annotations such as `title`, `description`, `default` and `examples` are ignored.
Any other keyword (`format`, `uniqueItems`...) is a 400 `unsupported_json_schema` at its JSON
Pointer; a schema that no value satisfies is `unsatisfiable_json_schema`; a number whose JSON
spelling a double cannot keep (`0.30000000000000001`) is `unsupported_json_schema`. GBNF follows
XGrammar's EBNF without budgets, captures, lazy rules, temperature or token references. Regular
expressions have no backreferences or lookaround. A grammar, regex or schema whose language has no
first token (`[^\s\S]`, an empty character class) is refused before generation.

**Strict objects (default).** Like vLLM and SGLang (XGrammar's `strict_mode`), an object that
declares `properties` and says nothing about `additionalProperties` admits only the declared keys,
and after a tuple (`prefixItems`, or a draft-07 `items` array) without `items`/`additionalItems` no
further items follow. This differs from literal JSON Schema, where such objects and tuples are open:
the model cannot add a key the schema did not declare. Everything written explicitly keeps its
meaning: `additionalProperties: true` admits any extra key, `additionalProperties: {schema}` extra
keys with values of that schema, `items` the items it describes; an object without declared
properties (`{"type":"object"}`, with or without a description) and an array without a tuple stay
open; `allOf` closes to the union of the properties its branches declare. `json_object` is any
object. `--constraint-json-strict off` restores literal JSON Schema. Open objects with declared
properties are expensive for the matcher when the model adds keys (56-95 ms per token mask
measured), so a request with `additionalProperties` (`true` or a schema) next to declared properties
can end with `constraint_budget_exceeded`; close the object if that happens. In strict mode a tuple
(`prefixItems`) without `items` is closed, so a `minItems` larger than the tuple makes the schema
unsatisfiable (HTTP 400): add `items`, or use `--constraint-json-strict off`.

With thinking on, only the content is constrained: the reasoning stays free until the canonical
`\n</think>\n\n` close, then the content follows the constraint; the thinking budget applies as
usual. Unless the client sets `presence_penalty`, a constrained request uses `0` instead of the
model preset or `--presence-penalty`, which would push the model away from repeated JSON
punctuation. One request carries at most one constraint, and a constraint cannot be combined with
custom stop strings (`stop`, `stop_sequences`), active tools (`tool_choice:"none"` keeps declared
tools inactive) or an Anthropic Assistant prefill: these are 400 `unsupported_output_combination`
(`assistant_prefill_not_supported` for the prefill).

## Errors

| Code | Status | When |
|---|---:|---|
| `invalid_json_schema`, `unsupported_json_schema`, `unsatisfiable_json_schema` | 400 | the schema is malformed, uses an unsupported keyword, or accepts no value; `param` ends with the JSON Pointer |
| `invalid_grammar`, `invalid_regex`, `invalid_choice` | 400 | the source does not parse, or its language is empty |
| `unsupported_output_combination` | 400 | two constraints, or a constraint with stop strings or active tools |
| `constraint_limit_exceeded` | 400 | an input over a [limit](#structured-output-limits) or a compilation over its time limit; the message names the flag |
| `server_overloaded` | 429 | no compilation slot became free within `--constraint-compile-wait-ms` (the server is busy compiling other schemas; retry) |
| `constraint_dead_end` | 400 | during generation no token can continue the content |
| `constraint_budget_exceeded` | 400 | during generation the matcher exceeded its time or memory budget; only this request ends |

Errors found before generation are ordinary HTTP errors. The last two happen while generating: a
non-streamed request returns the error instead of a response; a stream that has already sent its
200 ends with the protocol's error event (Chat `data: {"error":...}`, Responses
`event: response.failed` with the code in `response.error`, Anthropic `event: error`), and the
content streamed until then is incomplete.

## Structured output limits

A schema or grammar comes from the client, so its cost does too. The server bounds it at every
step, and a request over a bound fails alone with a 400 that names the flag; other requests and the
server are unaffected:

| Option | Bounds | Default |
|---|---|---:|
| `--constraint-max-kib N` | size of a JSON Schema (serialized), a GBNF grammar or a regex, or the total bytes of a choice list, in KiB | `256` |
| `--constraint-max-depth N` | nesting of the schema document (objects and arrays), group nesting of GBNF and regex; `1..1000` | `256` |
| `--constraint-max-items N` | members of one object or array of the schema (properties, `enum` values, `anyOf` branches...), choice entries, GBNF rules | `4096` |
| `--constraint-compile-timeout-ms N` | wall time of one compilation, from the moment it holds a slot; `0` = none | `5000` |
| `--constraint-compile-concurrency N` | compilations at the same time (a cached grammar takes no slot); `0` = no limit | `2` |
| `--constraint-compile-wait-ms N` | wait for a compilation slot, first come first served; beyond it the request gets 429 `server_overloaded` (the server is busy compiling, not the schema too complex); `0` = no limit | `30000` |
| `--constraint-cache-mib N` | LRU cache of compiled grammars | `64` |
| `--constraint-round-budget-ms N` | matcher CPU time of one request in one decode round (masks of all positions, or the accept of its tokens); the wall clock is checked at 5× as a safety net; `0` = none | `20` |
| `--constraint-request-budget-ms N` | matcher CPU time of one request in total: `N` plus `--constraint-round-allowance-ms` per round; `0` = none | `1000` |
| `--constraint-round-allowance-ms N` | see above | `5` |
| `--constraint-matcher-mib N` | memory of one request's matcher history; `0` = none | `512` |
| `--constraint-json-strict on\|off` | strict JSON Schema objects and tuples (below); `off` = literal JSON Schema | `on` |

The size, depth and item limits are checked in the HTTP layer on the parsed request, before
anything recursive reads the schema, and again before compilation for Engine users. Compilation
runs in the request's HTTP thread on at most `--constraint-compile-concurrency` CPU cores, under the
timeout, which starts when the compilation holds a slot and covers every phase (`allOf`/`$ref`
normalization, schema conversion, the token-mask cache). Requests wait for a slot in arrival order,
for at most `--constraint-compile-wait-ms`: someone else's slow schemas can delay a compilation,
not make it fail as too complex; it gives its memory back to the system when it was
large or failed, and a failed compilation is not cached. A conjunction (`allOf`, `$ref`) expands to
at most `--constraint-max-items` subschemas; a reference reached again through another branch adds
nothing. During generation the matcher runs on the Engine thread before each round, so an
expensive one would slow every request in the batch: the round and request budgets stop that
request instead, and the memory bound stops an ambiguous grammar whose matcher state grows with the
square of the output. The budgets count the CPU time of the Engine thread, so a busy machine
(Flash-Next's expert threads, other processes) does not make a legitimate grammar fail.

Ordinary schemas stay far from these limits: a mask costs about 5 µs per position (1 µs to 0.7 ms
for the schemas measured), and the matcher history grows by about 1 KB per output token. What does
reach them: deeply nested or very large schemas, many optional properties (compilation grows with
their square: 500 take 1.6 s, 1000 exceed the 5 s timeout; prefer `required`), and ambiguous
grammars or patterns such as `e ::= e e | "a"`, `(a*)*`, `((a|b)*)*` or `^(.*a){20}$`, whose matcher
work grows with the output.

## Cost

On Qwen3.8-27B (tp 2, MTP 3) an ordinary schema costs about 18 µs of host work per round, 0.3 % of
the round; on Flash-Next the difference is below run-to-run noise. The first request with a new
constraint compiles it (milliseconds for typical schemas, up to seconds near the limits) and the
first constrained request of a server builds the vocabulary index once (about 0.1 s); later requests
with the same constraint use the cache. With DFlash2 a constrained request decodes without drafts
(ordinary rounds), so it is slower than an unconstrained one; the other requests of the batch keep
drafting.
