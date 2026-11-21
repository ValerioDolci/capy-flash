# Server options

The table lists executable defaults, without a `--profile` ([profiles](capy-flash.md#profiles) set many of them). The
upstream startup example selects a long-context FP8/MTP3 configuration.

| Option | Meaning | Default |
|---|---|---:|
| `--profile NAME` | insert a tested configuration (`flash-next`, `flash-next-multi`, `flash-next-1gpu`, `27b`, `27b-16gb`; expansions in [Profiles](capy-flash.md#profiles)) before the other flags, which override it; must follow the artifact path | none |
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--api-key-file FILE` | read the key from FILE (one line); without either flag `NINFER_API_KEY` is used when set | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `metadata.name` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--vram-headroom-mib N` | VRAM in MiB that `--kv-capacity auto` leaves free after sizing the KV pool; requires `auto` | `1024` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `600000` (10 minutes) |
| `--decode-share F` | while a request's prompt runs, the requests already decoding get `F` times the time of each prompt unit (at least one decode round per unit). `0` keeps one round per unit: with Flash-Next a prompt unit of 16k tokens takes several seconds, during which the others produce almost nothing; `1` gives both equal time. Measured (Flash-Next, 2 GPUs, one request decoding while a 48k prompt arrives): the decoding request gets 1.7 / 41.8 / 71.0 tokens/s during that prompt with `0` / `0.5` / `1`, and the prompt's first token comes after 10.2 / 14.7 / 19.5 s. A prompt unit is not split: with 16k-token units the decoding requests still wait about 3.3 s per unit | `0` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--q8-a8 32\|128` | opt-in, changes the numbers: Flash-Next's dense Q8 projections of a prompt chunk (64 tokens and more) run on int8 tensor cores, activations quantized per token in groups of 32 or 128 values; prefill +7-8 % on 2x RTX 5070 Ti, perplexity +0.21 % (NVFP4), +0.004 % (64 GB edition) ([README](../../README.md#optional-speed-switches-off-by-default)). `NINFER_Q8_A8` also sets it (the flag wins) | off |
| `--attn-pv-f16acc` | opt-in, changes the numbers: the INT8-KV prefill attention sums P x V of each 64-key tile in FP16 before adding it to the FP32 accumulator (the 27B on long prompts: prefill +6 % at 16k, +25 % at 128k; perplexity +1.1 % at 64k) ([README](../../README.md#optional-speed-switches-off-by-default)). `NINFER_ATTN_PV_F16ACC=1` also sets it | off |
| `--tp-ar-payload fp16\|int8` | `int8` is opt-in and changes the numbers: at tp 2 Flash-Next's prefill exchange between the GPUs (MoE and mixer output) carries int8 codes with an FP16 scale per 32 rows; prefill +7-8 %, perplexity −0.22 % (NVFP4); no effect on the 27B ([README](../../README.md#optional-speed-switches-off-by-default)). `NINFER_TP_AR_PAYLOAD=int8` also sets it | `fp16` |
| `--kv-lease-window N` | output tokens of Device KV a request reserves beyond its prompt (or the prefill chunk, if larger), extended as the answer grows; `0` reserves the whole `max_tokens` at admission | `4096` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | pretty stderr verbosity | `info` |
| `--device N` | CUDA device index | `0` |
| `--tp 1\|2` | tensor-parallel width; see [Two GPUs](#two-gpus) | `1` |
| `--devices A,B` | one CUDA device per rank, rank 0 first; required with `--tp 2` | `--device` |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--max-request-json-depth N` | nesting of objects and arrays in a request body and in tool-call `arguments` strings, checked on the bytes before parsing; deeper is a 400 `request_json_too_deep`; `1..5000` | `1000` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--constraint-max-kib N` | structured output: size of a schema, grammar, regex or choice list, in KiB ([limits](structured-output.md#structured-output-limits)) | `256` |
| `--constraint-max-depth N` | structured output: schema nesting, GBNF/regex group nesting; `1..1000` | `256` |
| `--constraint-max-items N` | structured output: members of one schema object or array, choice entries, GBNF rules | `4096` |
| `--constraint-compile-timeout-ms N` | structured output: wall time of one compilation, from the moment it holds a slot; `0` = none | `5000` |
| `--constraint-compile-wait-ms N` | structured output: wait for a compilation slot (first come, first served), then 429; `0` = no limit | `30000` |
| `--constraint-compile-concurrency N` | structured output: compilations at the same time; `0` = no limit | `2` |
| `--constraint-cache-mib N` | structured output: compiled grammar cache | `64` |
| `--constraint-round-budget-ms N` | structured output: matcher CPU time of one request per decode round; `0` = none | `20` |
| `--constraint-request-budget-ms N` | structured output: matcher CPU time of one request, plus `--constraint-round-allowance-ms` per round; `0` = none | `1000` |
| `--constraint-round-allowance-ms N` | structured output: per-round addition to the request budget | `5` |
| `--constraint-matcher-mib N` | structured output: memory of one request's matcher history; `0` = none | `512` |
| `--constraint-json-strict on\|off` | structured output: an object with declared properties and no `additionalProperties` admits only those keys, a tuple without `items` no more items (as vLLM/XGrammar); `off` = literal JSON Schema | `on` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--log-speculation-detail` | add the per-step accepted draft lengths and the output/prompt 4-gram overlap to every `request_done` record (test measurement; requires `--request-log-jsonl`); see [Structured request log](request-log.md) | off |
| `--log-speculation-detail-max-steps N` | decode steps recorded per request by `--log-speculation-detail` (`1..1048576`); a longer request is marked truncated | `4096` |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4` | KV-cache storage | `bf16` |
| `--spec mtp\|dflash2` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash2 `1..15` | unset |
| `--rope-yarn-factor F` | YaRN RoPE context extension, `[1,4]`: raises the accepted `max-context` ceiling to native × F (native stays 262,144) and rotates as Hugging Face's `rope_type: "yarn"` does (the same inverse frequencies, cos and sin multiplied by 0.1 ln F + 1). It covers every text RoPE on every GPU: attention at one and two GPUs, text after an image (MRoPE), the Flash-Next QSA indexer and the MTP layer; the Vision tower and the DFlash2 drafter keep their RoPE. `1` installs nothing and leaves greedy output byte-identical; above 1 it costs perplexity and is experimental: its quality past the native window is not measured. Opt-in, ported from @splickz's work ([#28](https://github.com/ValerioDolci/capy-flash/issues/28), [PR #29](https://github.com/ValerioDolci/capy-flash/pull/29)) | `1.0` |
| `--rope-original-max-position N` | the native window YaRN scales from (Hugging Face `original_max_position_embeddings`), used only when `--rope-yarn-factor` > 1 | `262144` |
| `--lm-head-draft` | optimized proposal head | off |
| `--default-max-tokens N` | output limit when omitted by a request | the context left after the prompt |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--no-vision` | turn Vision off, also dropping a profile's `--vision-device` and `--max-vision-tokens` (the 27B profiles turn Vision on) | Vision off unless `--vision` |
| `--vision-device N` | CUDA device that holds the Vision tower and encodes; one of `--devices` at `--tp 2`; see [Two GPUs](#two-gpus) | `--device` |
| `--max-vision-tokens N` | merged Vision tokens of one image or video item (`64..16384`); larger media are resized and the encode workspace is planned for `N` | `16384` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-tp-mailbox` | keep the captured `--tp 2` all-reduces on cross-device copies; see [Two GPUs](#two-gpus) | mailbox on without P2P |
| `--embedding-host` | keep the token embedding table in one copy in pinned host memory that every GPU reads in place over PCIe, instead of one copy per GPU: frees its VRAM (1.18 GiB per GPU for Qwen3.8-27B) for the KV pool, identical results; see [Two GPUs](#two-gpus) | off |
| `--host-pinned-max-gib G` | keep at most `G` GiB of the host-mapped weights (Flash-Next expert banks) page-locked; the banks of the least used layers beyond it stay in pageable RAM (CPU reads them in place, GPU copies go through a pinned staging ring, no prefill streaming for them); slower where pageable, and on long prompts the tokens can differ from a fully pinned run (the CPU computes those layers' experts instead of streaming them). Unset: the whole page-locked allocation is tried first and the fallback starts only if the driver refuses it (Windows WDDM / WSL2). With `--kv-ram-active` its Host KV planes count in the limit (the weights keep the rest page-locked). `NINFER_HOST_PINNED_MAX_GIB` also sets it | no limit |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--device-state-slots N` | extra Device checkpoint StateImages beyond the active-lane guarantee | `max-concurrency`; `max(2 * max-concurrency, 8)` at `--tp 2` |
| `--host-state-slots N` | pinned Host StateImage capacity | `8`; at `--tp 2` `0` unless set (`--profile 27b`, `--profile flash-next`: `8`) |
| `--host-kv-mib N` | shared pinned Host Main/Backend KV byte capacity in MiB | `8192`; at `--tp 2` `0` unless set (`--profile 27b`: `8192`, `--profile flash-next`: `4096`) |
| `--disk-cache-dir DIR` | disk tier of the context cache (with `--disk-cache-gib`): retained conversations are written to `DIR` and survive a restart; see [Disk tier](#disk-tier-of-the-context-cache) | off |
| `--disk-cache-gib N` | byte cap of the disk tier in GiB; least recently used conversations are deleted first | off (required with `--disk-cache-dir`) |
| `--disk-cache-min-free-gib N` | free space the file system keeps at least; the disk tier evicts, then skips saves | `max(10, 5 % of the file system)` |
| `--disk-cache-idle-ms N` | Engine inactivity after which retained conversations not yet on disk are written (`0`: only at shutdown) | `5000` |
| `--disk-cache-flush-seconds N` | at SIGTERM/SIGINT, the time allowed to write the conversations not yet on disk | `60` |
| `--max-private-continuations N` | private continuation descriptor capacity | `2 * max-concurrency`; `max(2 * max-concurrency, 8)` at `--tp 2` |
| `--max-shared-prefixes N` | Engine-wide shared stable-prefix descriptor capacity | `max(max-concurrency, 4)` |
| `--max-long-anchors-per-continuation N` | private long-anchor limit per continuation | `2` |
| `--turn-anchors N` | private long-anchor opportunities at the last N message boundaries before the final message, so a cancelled or edited turn resumes from the end of the previous turn instead of re-prefilling the conversation (`0` disables). Protocol cache markers win the shared marker budget: with a request already carrying four `cache_control` breakpoints the anchor is dropped rather than pushing the request over the limit. Default: `0` at tp 2 (each anchor holds one more StateImage per turn on both ranks, in the pools the turn closures need) and on Qwen3.8-Flash-Next (one extra prefill pass per turn, ~0.8 s there); `1` on the other models at tp 1 | `0` at tp 2 and on Flash-Next, `1` otherwise |
| `--no-endpoint-reuse` | a follow-up turn never resumes from the end of the previous reply (a state the decode left); it resumes from the last prefill checkpoint and prefills the reply again, so its tokens are the same whether or not the reply was still cached; see [Same tokens after an eviction or a restart](#same-tokens-after-an-eviction-or-a-restart) | off (endpoints resumed) |
| `--elastic` | Qwen3.8-Flash-Next with offloaded experts: the KV pool keeps its address space for `--kv-capacity` (every request together) but backs pages on demand, and the VRAM it does not use becomes extra expert slots, given back when the KV grows; no effect on models without offloaded experts | off (on in `--profile flash-next` and `flash-next-1gpu`) |
| `--kv-grow N` | with `--elastic`: step, in tokens, in which the KV backing starts and grows; must be positive | `8192` |
| `--kv-ram-active` | Qwen3.8-Flash-Next: the K/V of the text attention layers lives in pinned host RAM instead of VRAM (only the QSA indexer's block keys stay on the GPU, 0.75 KiB per token per GPU instead of 10-16 KiB). Works with any `--max-concurrency`: each decode step copies the blocks the indexer selected for every request into a shared VRAM block cache (only the ones not already there cross PCIe), and each prompt chunk copies one layer of the visible context at a time; results are identical to the default. The RAM is taken 8,192 tokens at a time as conversations grow and given back when they end, up to `--kv-capacity` tokens (which the page-locked limit reserves from the start); the VRAM this frees goes to routed-expert slots (with `--elastic`) or to more concurrent context. It costs a fixed VRAM region of one layer of `--max-context`, at most 262,144 tokens (longer contexts copy only the blocks each prompt chunk selects) ([sizes and rates](../maintainer/active-kv-in-ram.md#cost)) | off |
| `--kv-ram-region-tokens N` | with `--kv-ram-active`: tokens per layer of the VRAM staging region of a prompt chunk (capped at `--max-context`); a longer visible context stages the union of each chunk's selections. Same tokens; a smaller region frees VRAM on long contexts. `NINFER_KVRAM_REGION_TOKENS` sets it too (the flag wins) | `262144` |
| `--kv-ram-cache-blocks N` | with `--kv-ram-active`: 4-token blocks of the VRAM block cache per text attention layer, raised so the region also holds one layer of `--max-context`; more keeps more of the indexer's past selections on the GPU at the cost of VRAM ([cost per block](../maintainer/active-kv-in-ram.md#memory)). Same tokens. `NINFER_KVRAM_CACHE_BLOCKS` sets it too (the flag wins) | `4096` |
| `--kv-budget-gib G` | with `--elastic`: at most `G` GiB of KV per GPU (Text and MTP pools, every request together); the base expert slots are sized for it instead of `--kv-capacity`'s KV, and a request that needs more KV fails. Keeps more experts resident when contexts stay short | unset (sized for `--max-context`) |
| `--miss-split auto\|calibrate\|off` | Qwen3.8-Flash-Next: `auto` lets the GPU read part of each decode window's missing experts over PCIe in parallel with the CPU, choosing the share from fixed costs fitted on a Ryzen 9 9900X; `calibrate` does the same with costs measured on this machine: at startup it times how fast the GPU copies an expert over PCIe, then the first 4,096 decode calls keep every missing expert on the CPU while their time is fitted (fixed + per expert + per extra column, at the configured CPU threads), and the measured costs replace the built-in ones for the rest of the run. Both steps print a `miss split calibrate` line. `off` leaves them all to the CPU. Forced off with GGUF expert banks (the 64 GB edition) | `off` |
| `--stream-min-tokens N` | Qwen3.8-Flash-Next: prompt chunks of at least `N` tokens stream the non-resident experts to the GPU over PCIe, shorter chunks compute them on the CPU; `0` never streams. With GGUF expert banks it is the threshold from which the prefill reads them in place from host RAM | `256` |
| `--expert-cpu-threads N` | Qwen3.8-Flash-Next: CPU threads of the experts that miss the VRAM cache (`1..64`, at most the allowed physical cores, otherwise startup fails); see [CPU expert threads](capy-flash.md#cpu-expert-threads) | automatic, logged at startup |
| `--expert-cpus LIST` | Qwen3.8-Flash-Next: the CPUs of the missing experts as `taskset` writes them (`0-7,16`): one thread per listed CPU, pinned in that order, SMT siblings allowed; replaces the automatic list and `--expert-cpu-threads` (the startup line still prints the automatic or requested count). A CPU the process may not run on stops the start. Same results; see [CPU expert threads](capy-flash.md#cpu-expert-threads). `NINFER_HOST_MOE_CPUS` sets it too (the flag wins) | automatic |
| `--expert-exclusive-layers N` | 64 GB edition (ggml block expert banks): the first `N` layers keep the experts the GPUs hold at startup in VRAM only, without a page-locked RAM copy, for a machine with 64 GB of RAM and a desktop beside the server (`20`: 7.7 GiB less RAM on two GPUs; the tokens change and decode is about 2-3 % slower). `NINFER_QWEN4_EXCLUSIVE_LAYERS` sets it too (the flag wins) | `0` (off) |
| `--expert-half-life N` | Qwen3.8-Flash-Next: the expert cache's memory, as the half-life in routed tokens of the routing frequency that admits experts into a full layer and picks the one to evict. Shorter follows a change of topic sooner. It changes which experts the GPUs hold, so the tokens | `128` |
| `--gguf-hybrid-cost A,B,C` | ggml block expert banks on one GPU: the cost model, in microseconds, that decides how much of each prompt chunk's non-resident experts the CPU computes while the GPU copies the others (`A` per CPU expert, `B` per routed token of a CPU expert, `C` per expert copied over PCIe), measured on a Ryzen 9 9900X with DDR5 and PCIe 5.0 x8; slower RAM or a faster link want a smaller CPU share. Changes the CPU/GPU split, so the numerics. `NINFER_GGUF_HYBRID_COST` sets it too (the flag wins) | `35,3.6,100` |
| `--prefill-miss-dump DIR` | diagnostic for prefix-reuse misses: Serve keeps the token contexts of its last 16 requests (prompt plus committed output, cancelled requests included); a request that recomputes at least `--prefill-miss-min-tokens` prompt tokens and shares at least `--prefill-miss-min-match` of its prompt with the closest of them is dumped when the server recomputed that many tokens identical to it, or when that many of its tokens follow the divergence point (the client changed or dropped earlier text). Each dump is a new directory under `DIR` with `new_prompt.txt`, `old_context.txt`, both token-ID lists and `summary.json` (token counts, divergence point with the text on both sides, and the reading); the request also logs a `prefill miss` warning. At most 64 dumps per process. The files contain conversation text verbatim | off |
| `--prefill-miss-min-tokens N` | threshold of `--prefill-miss-dump` in tokens | `10000` |
| `--prefill-miss-min-match F` | minimum share of the prompt matching a recent context for `--prefill-miss-dump` (`0..1`) | `0.5` |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--reasoning-eos-guard`, `--no-reasoning-eos-guard` | the first token of a reply that starts in thinking is never a terminal token: a terminal draw is redrawn without the terminal ids; token-identical when it does not fire. A model whose stop configuration the guard cannot handle serves without it (log line `reasoning EOS guard off for this model`) | on |
| `--reasoning-close-retry`, `--no-reasoning-close-retry` | a reply that ends with a terminal token inside its thinking (or right after `</think>` with blank content and no tool call) gets `"\n</think>\n\n"` appended and is continued once; text requests with prefix reuse only. Turn both guards off to compare outputs with another engine or to measure raw model behaviour | on |
| `--chat-template FILE` | Jinja chat template used instead of the one embedded in the artifact; the capy-flash launchers pass `qwen3_8.jinja` ([Chat template](capy-flash.md#chat-template)) | artifact's template |
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

Engine selects sampling defaults from the request's resolved thinking mode, the same for every
model: `1.0/0.95/20/0/0` for temperature/top-p/top-k/min-p/presence penalty in thinking mode and
`0.7/0.80/20/0/1.5` in non-thinking mode. Frequency penalty is `0` in both presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

A `seed` (request field or `--seed`) fixes the random draws, not the text. Each draw is a function of the seed, the
token position and what the draw is for, so with `--spec off` a request repeats its tokens when its logits repeat bit
for bit. With `--spec mtp` or `--spec dflash2` at temperature above `0`, decode tokens come from rejection sampling
in the speculative verification: a draft is accepted with its probability under the target distribution, the first
rejected draft is replaced by a draw from the remaining distribution, and a round whose drafts are all accepted adds
a bonus draw. Which draw a position uses depends on the drafts and on where each round ends, so the same seed gives
different text with and without speculation, or with another `--spec` or `--draft-tokens`, and the same text again
only when drafts and logits repeat bit for bit; on Qwen3.8-Flash-Next they do not (see
[What the engine does](capy-flash.md#what-the-engine-does)). At temperature `0` the seed is not used.

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`:
`C` slots guarantee active requests and `H` is a global checkpoint pool. Host State and Host KV are
independent startup-fixed pinned-memory capacities; Host KV is shared by Main and the selected
Backend pool and is consumed in physical page extents. `--no-prefix-reuse` selects root-only Engine
mode and cannot be combined with any of the seven explicit context-cache capacity flags, including
zero-valued flags.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

## Disk tier of the context cache

`--disk-cache-dir DIR --disk-cache-gib N` (v0.7.2, off by default; idea and first implementation by
@KouShuo, issue #6) adds a third tier below VRAM and RAM. Retained conversations are written to `DIR` by a
thread of their own: after `--disk-cache-idle-ms` without any request in the server (default 5 s; the
write stops as soon as a request arrives and continues at the next idle moment, so a decode never waits
for the disk) and at a clean shutdown (SIGTERM/SIGINT, at most `--disk-cache-flush-seconds`). After a
restart, a request whose prompt begins with a saved conversation loads its KV, MTP KV and recurrent
state from disk and prefills only the rest of the prompt. The disk wins over a match in VRAM or RAM when it
holds more of the prompt by more than 1/16 of its own length (v0.7.4; before, a shorter in-memory match,
such as a system prompt another conversation left, made the long conversation prefill again).

- Storage is content addressed, as in LMCache and SGLang HiCache: the KV is cut into 256-token blocks
  named by a chained hash of the prefix tokens, each written once and shared by every later turn and by
  every conversation with the same prefix; the recurrent (GDN) state cannot be cut into blocks and is
  saved whole at one checkpoint per save (147 MiB for Qwen3.8-27B). A conversation itself is a small
  manifest (tokens, the blocks it uses, its checkpoints). A turn therefore writes only its new tokens'
  blocks and one state: a 32-turn agent session growing to 80k tokens wrote 3.2 GiB of KV plus one
  state per turn.
- One directory per server: `DIR/.lock` is held for the life of the process; a second server on the
  same directory does not start and names the process that holds it.
- The manifests hold the conversations' tokens, so the cache is private to the server's user: `DIR/m`
  and `DIR/b` are set to 0700 at start (which also covers files an older version wrote 0644) and new
  files are created 0600, whatever the umask.
- Space: at most `--disk-cache-gib` (least recently used conversations go first, with the blocks only
  they use; blocks of an interrupted save are reused by the next attempt or deleted at the next start),
  and the file system always keeps `--disk-cache-min-free-gib` free (default max(10 GiB, 5 % of the file
  system)): the cache evicts, and if that is not enough the save is skipped with a log line. The server
  warns at start when the cap is more than the space it can use.
- Files are written to a temporary, synced and renamed (a conversation only after its blocks): a killed
  server leaves no partial file (temporaries are deleted at the next start). Every 4 MiB block carries a
  checksum and every KV block its tokens; a broken file is deleted and the request prefills normally.
- Each file carries a validity key: artifact and executable (path, size, modification time), tensor
  parallel width, KV dtype, speculative mode and the KV/state layouts; files with another key are
  ignored. A new build or artifact starts with an empty disk tier.
- Text and image prompts (an image's placeholder tokens are named on disk by the
  item's content digest, grid and offset, so another picture never resumes a saved one; a checkpoint
  never splits an item); without speculation or with MTP (DFlash2 turns the disk tier
  off). Pinned memory: about 320 MiB plus one StateImage. Space, Qwen3.8-27B with INT8 KV (one GPU or
  both halves at `--tp 2`): about 36 KB per token of context plus 147 MiB per saved state.
- Measured on two RTX 5070 Ti, Qwen3.8-27B GSQ-3, `--profile 27b`, page cache dropped: time to first
  token after a restart 0.32 / 0.70 / 1.08 s at 20k / 80k / 140k tokens of context, against 3.8 / 21 / 48 s
  of prefill (the KV and state themselves load at about 6 GB/s); the reply is the same token for token as the prefill from zero and as
  the resume from VRAM. `usage.prompt_tokens_details.cached_tokens` does not count the tokens loaded
  from disk (the server log does: `[disk-kv] resumed`).
- What leaves the RAM tier under pressure is on disk only if the server was idle for
  `--disk-cache-idle-ms` after that conversation's last turn (or shut down cleanly); there is no
  synchronous spill at eviction time. After a restart a conversation resumes from its latest saved
  checkpoint: an edited last message falls back to a full prefill.

## Same tokens after an eviction or a restart

When the client sends the model's reply back unchanged (agents, tool calls), the next request normally
resumes from the session endpoint: the state and KV the decode left after the reply. Those reply
tokens were computed by the decode, a few rows per round; when the endpoint is no longer there
(evicted, a restart, `--no-prefix-reuse`) the same request prefills the reply in chunks instead,
and greedy output can differ from the turn after the reply on. `--no-endpoint-reuse` makes every
follow-up turn take the second path: it resumes from the last prefill checkpoint before the reply
(the turn closure or the response-replay point, in VRAM, RAM or on disk; endpoints are not written
to the disk tier either) and prefills the reply again. The output then depends only on the
prompt and the configuration, not on what the cache still held.

It is a trade-off, off by default. Measured on Qwen3.8-27B NVFP4 at `--tp 2` with the `qwen27b`
serving flags (MTP 3, 4 lanes), 53 agentic turns whose replies (at most 256 tokens) go back into
the history, replayed with the cache warm:

| | default | `--no-endpoint-reuse` |
|---|---|---|
| same tokens as a cold prefill of the same prompts (first 20 turns) | 11 of 20 | 20 of 20 |
| same tokens after five clean restarts with the disk tier | 24 of 53 | 53 of 53 |
| time to first token, sum over 53 requests (3 runs each) | 44.99 s | 48.94 s (+8.8 %, about 75 ms per request) |
| time to first token, median | 0.729 s | 0.802 s |
| prompt tokens prefilled | 225,705 | 235,852 (+4.5 %) |
| decode speed | 227.1 tok/s | 227.0 tok/s |

The extra time is one prefill of the previous reply per turn, so it grows with the reply length.

## Two GPUs

`--tp 2 --devices A,B` serves an artifact split across two GPUs, for a model that does not fit
one device. Qwen3.8-Flash-Next runs at `--tp 2` through `--profile flash-next` ([capy-flash
serving](capy-flash.md)); the examples below use the dense Qwen3.8-27B:

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
GPU (under WSL2 it occasionally does while serving), the two ranks' results have diverged: the
requests that were running fail with HTTP 503 (`service_unavailable`, an error event on a stream
already started), the context cache is dropped, the transport steps down once (the MTP draft
phase onto the copies, then the copies everywhere) and the CUDA Graphs are captured again (as long
as the `CUDA graphs ready` step of startup takes); queued requests are then served, with the same
tokens. The log records it as an
`[engine] warning: a tensor-parallel exchange timed out while serving` line. Only when nothing is
left to step down to, or the graphs cannot be captured again, is it an Engine-wide failure.

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
are read from (measured on the agentic replay of `bench/capy_flash`: 4 GiB cost +0.8 GB of disk reads and +0.6 s of
PLE gathers over 53 turns, against about 5 s of decode saved; 8 GiB added the same again with no
further gain). Without them, every
checkpoint lives in a Device StateImage, and the defaults raise the Device checkpoint pool
and the private catalog to `max(2 * max-concurrency, 8)`. At tp 2, a private checkpoint captured during
prefill (the turn-closure point a follow-up turn resumes from, a long-prompt anchor) goes to a Device
StateImage, or to a Host one when the server has them; when every pool is full it first releases idle
retained conversations without a live session, least valuable first (GreedyDual: rebuild tokens plus
the inflation when last used), until it fits; each release counts as a private eviction. A
conversation is released only when it is worth no more than what the capture protects plus the work
already lost to skipped captures since the last release: a short side request never evicts a long
session, while a session nobody returns to gives way after a few turns of a new one. A capture that
still does not fit is skipped, and that conversation's next turn re-prefills from an earlier
checkpoint or its whole prompt. On one GPU the behaviour is unchanged: with `--host-state-slots 0`
a capture that finds the Device pool full is skipped without releasing anything.
Each extra slot costs one StateImage per rank (about 73 MiB for Qwen3.8-27B); agentic or
multi-conversation servers should set `--device-state-slots 12` to `16` when memory allows (on two
16 GB boards at 196,608 tokens with Vision, 8 fit with the official NVFP4 artifact: 13.1 GiB peak per
board on the agentic replay plus an image request). The startup log prints one
line per rank and a `tensor parallel` capacity line with the Device checkpoint pool and the
transfer path: `p2p on` when the driver grants direct peer access, `p2p off (host-staged copies)`
otherwise (GeForce boards). When the driver advertises peer access but cannot map the peer's
memory, the server prints why and starts with `p2p off` (#41). Under VFIO passthrough this happens
with the default 256 MiB BAR1 (`cudaErrorMapBufferObjectFailed`): QEMU exposes the size the host BAR
has, so grow it on the hypervisor with the VM off (for example
`echo 14 > /sys/bus/pci/devices/<gpu>/resource1_resize`, 16 GiB, with Above 4G Decoding and Resizable
BAR enabled in the BIOS) and `p2p on` follows. With CUDA Graphs one `cuda graphs` line per rank compares the device
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
each with or without `--vision`, with `bf16`, `int8` or `nvfp4` KV. `--spec dflash2` without
`--lm-head-draft` and the `fp8` and `k8v4` KV types are rejected at startup (`nvfp4` is accepted since v0.4.2, see [Two GPUs](../cli.md#two-gpus)), and so is a `--vision-device` outside
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
