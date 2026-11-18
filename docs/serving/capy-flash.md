# capy-flash serving

capy-flash serves two models with the same `ninfer-serve` binary: **Qwen3.8-Flash-Next** (MoE; the routed
experts that do not fit in VRAM stay in host RAM and run on the CPU) and the dense **Qwen3.8-27B**, on one or
two 16 GB `sm_120` boards. The rest of this page is NInfer's general server reference; this section is what is
specific to capy-flash: launchers, profiles, RAM, chat template, CPU threads and client setup.

## Launchers and profiles

`install.sh` downloads one Hugging Face repository per `--model` (the whole repository) and prints the
launcher command. The launchers ship in the release tarball (`tools/release/` in the source tree), add the
profile, the model id and the chat template, and append any extra flag you give them, which then wins over
the profile:

| `install.sh --model` | artifact repository | launcher | profile | GPUs |
|---|---|---|---|---|
| `flash-next` (default) | `Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash` | `serve-flash-next.sh` | `flash-next` | 2 |
| `flash-next-64gb` | `Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash` | `serve-flash-next.sh` (install.sh adds `--host-kv-mib 1024 --host-state-slots 2`); `NINFER_DEVICES=0` for one GPU | `flash-next`; one GPU: `flash-next-1gpu` | 1 or 2 |
| `27b` | `Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer` | `serve-tp2.sh` | `27b` | 2 |
| `27b-16gb` | `Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash` | `serve-27b-16gb.sh` | `27b-16gb` | 1 |

RAM, disk and download size of each model: [Pick a model](../../README.md#pick-a-model).

Launcher environment (defaults in brackets): `NINFER_HOST` (`127.0.0.1`), `NINFER_PORT` (`8080`),
`NINFER_MAX_CONTEXT` (the profile's), `NINFER_MODEL_ID` (`capy-flash` for Flash-Next, `qwen3.8-27b` for the 27B),
`NINFER_DEVICES` (`0,1`; `serve-flash-next.sh` switches to `flash-next-1gpu` when it holds one id),
`NINFER_DEVICE` (`0`, `serve-27b-16gb.sh`) and `NINFER_VISION=0` (the two 27B launchers: `--no-vision`).

### Profiles

`--profile NAME` (after the artifact path) inserts the tested configuration of a model before the other flags,
so a flag given on the command line overrides the profile's value; the KV capacity follows `--max-context`
unless the profile sets it. `ninfer-serve --help` prints the same expansions.

| profile | expands to |
|---|---|
| `flash-next` | `--tp 2 --devices 0,1 --kv-dtype int8 --elastic --max-context 262144 --max-concurrency 1 --spec mtp --draft-tokens 3 --host-kv-mib 4096 --host-state-slots 8 --embedding-host` |
| `flash-next-multi` | `--tp 2 --devices 0,1 --kv-dtype int8 --elastic --max-context 262144 --kv-capacity 786432 --max-concurrency 3 --kv-ram-active --spec mtp --draft-tokens 1 --host-kv-mib 4096 --host-state-slots 8 --embedding-host` |
| `flash-next-1gpu` | `--device 0 --kv-dtype int8 --elastic --max-context 262144 --max-concurrency 1 --spec mtp --draft-tokens 3 --embedding-host --host-kv-mib 1024 --host-state-slots 8` |
| `27b` | `--tp 2 --devices 0,1 --kv-dtype int8 --max-context 196608 --device-state-slots 4 --max-concurrency 1 --spec mtp --draft-tokens 3 --vision --vision-device 0 --max-vision-tokens 4096 --host-state-slots 8 --host-kv-mib 8192 --prefill-chunk 2048` |
| `27b-16gb` | `--device 0 --kv-dtype int8 --max-context 65536 --kv-capacity auto --vram-headroom-mib 384 --device-state-slots 2 --max-concurrency 1 --spec mtp --draft-tokens 3 --lm-head-draft --embedding-host --vision --vision-device 0 --max-vision-tokens 4096 --host-state-slots 8 --host-kv-mib 4096 --prefill-chunk 2048` |

`flash-next-multi` is `flash-next` for two or three clients at once (agents, several chats): three requests run
together instead of queueing, the KV of the generating conversations lives in RAM (`--kv-ram-active`) so the VRAM
keeps its expert slots, the pool holds 786,432 tokens (three full contexts; the RAM is taken as they grow), and MTP
drafts one token, which is faster than three once several requests share a step. Measured on 2x RTX 5070 Ti with
the three agentic sessions of `bench/capy_flash/sessions.json` (53 turns, 256 tokens per turn, greedy), two or three
runs per cell, each run a fresh server:

| clients at once | `flash-next`: total / median time to first token | `flash-next-multi` |
|---|---|---|
| 3 | 176 s / 7.8 s | 160 s / 1.7 s |
| 2 | 122 s / 4.6 s | 122 s / 1.7 s |
| 1 | 163 s / 1.3 s | 172 s / 1.3 s |

With one client it costs about 5 % of total time (decode 137 instead of 158 tokens/s, MTP 1 instead of 3), so
`flash-next` stays the default. The launcher takes it as an extra flag: `serve-flash-next.sh <artifact>
--profile flash-next-multi`. The same three clients with the KV in VRAM (the profile without `--kv-ram-active`)
took 264 s.

`flash-next-1gpu` is meant for the 64 GB edition (GSQ-RCO experts) on one GPU. The 64 GB edition on two GPUs
uses `flash-next` with `--host-kv-mib 1024 --host-state-slots 2` (smaller RAM caches), which is what
`install.sh --model flash-next-64gb` prints.

### Vision

- The two 27B artifacts contain the vision tower, and `--profile 27b` / `--profile 27b-16gb` (so `serve-tp2.sh` and
  `serve-27b-16gb.sh`) turn Vision on. It costs VRAM that would otherwise go to the KV pool. To turn it off:
  `NINFER_VISION=0` with the launchers, or `--no-vision` after the profile (it also drops the profile's
  `--vision-device` and `--max-vision-tokens`).
- The Flash-Next artifacts that `install.sh` downloads are text only, and the Flash-Next profiles leave Vision off.
  Images on Flash-Next need the separate artifact `Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash` (download it by
  hand; `install.sh` never fetches it) or a conversion with `python tools/capy_flash/convert.py --model <checkpoint
  dir> --out <dir> --vision`, and `--vision` on the command line (`serve-flash-next.sh <vision artifact> --vision`).
  The tower goes on GPU 0 (`--vision-device 1` moves it, `--max-vision-tokens N` caps one image). Measured with
  `--max-vision-tokens 4096`: tower 0.45 GiB on its GPU, 4 expert slots per layer fewer (66 to 62), decode about 4 %
  slower on short prompts, prefill unchanged; the merged embeddings are within 0.09 (relative) of a BF16 PyTorch
  reference. Without `--vision` the tower is not loaded.

## RAM

- **Flash-Next experts.** The routed experts are one page-locked host allocation: 64.6 GiB for the NVFP4 artifact,
  48 GiB for the GSQ-RCO (64 GB) edition. While loading, the server checks that MemAvailable covers the page-locked
  host weights plus max(2 GiB, 5 % of MemTotal) and otherwise refuses to start with the numbers (`NINFER_RAM_GUARD=0`
  skips the check); the Host tiers below are checked after that, with the experts already pinned. With
  `--profile flash-next` on a 96 GB machine the two checks add up to about 75 GiB free at startup; `install.sh` fails
  below 75 GiB in total and warns below 75 GiB free (56 GiB for the 64 GB edition). The 27B needs no RAM beyond its
  context cache.
- **CMA memory (Ubuntu 26.04, Linux 7.0).** The kernel may hold part of the RAM as CMA (`CmaTotal` / `CmaFree` in
  `/proc/meminfo`; the kexec-handover scratch takes about 9 GiB on a 64 GiB machine). Free CMA counts in MemAvailable
  but cannot be page-locked, and pinning more than the RAM outside it stalls the load in reclaim until systemd-oomd
  kills the server (#40). Without `--host-pinned-max-gib` the server caps the page-locked weights at the available RAM
  outside CMA minus max(2 GiB, 5 % of MemTotal) and 8 GiB for its other page-locked buffers, keeps the rest of the
  expert banks in pageable RAM (slower on those layers) and prints the cap; the Host tier check also leaves CMA out.
  To page-lock everything, boot with `kho=off` (or `cma=0`) on the kernel command line. `NINFER_RAM_GUARD=0` turns
  the cap off together with the RAM check.
- **Context cache in RAM (Host KV and Host StateImages).** On one GPU (`--tp 1`) they are on by default: 8 Host
  StateImages and 8 GiB of pinned Host KV. At `--tp 2` an unset `--host-kv-mib` or `--host-state-slots` becomes 0.
  The profiles set them: `flash-next` 4096 MiB + 8 states, `flash-next-1gpu` 1024 MiB + 8, `27b` 8192 MiB + 8,
  `27b-16gb` 4096 MiB + 8. The server refuses to start when these pinned tiers would leave less than max(4 GiB, 5 %
  of RAM) available (`NINFER_HOST_TIER_RAM_GUARD=0` skips that check).
- **Page-locked limit (Windows WDDM / WSL2).** When the system grants less pinned memory than the experts need, the
  banks of the least used layers stay in pageable RAM: the CPU reads them in place and the GPUs copy them through a
  pinned staging ring (no prefill streaming for those layers). Slower where pageable, and on prompts long enough to
  stream experts the tokens can differ from a fully pinned run: the experts of those layers that are not on a GPU are
  computed by the CPU instead of being streamed (same weights, the CPU's arithmetic).
  `--host-pinned-max-gib G` (or `NINFER_HOST_PINNED_MAX_GIB`) sets the cap by hand.
  The value must be a complete, finite positive number of GiB representing at least one byte
  and less than 16 EiB. Fractional bytes are truncated; invalid values are rejected.
  Loading fails if weights that must stay pinned (such as host-mapped embeddings) exceed the
  remaining budget after active Host KV reservations; they cannot fall back to pageable RAM.

## Chat template

Launch through a launcher, or pass `--chat-template`: the launchers add
`--chat-template <release>/chat_templates/qwen3_8.jinja` automatically. That template is NInfer's Qwen3.8 template
(system and developer turns anywhere, final-assistant continuation). Qwen's upstream template, which older copies
of the published artifacts embed, rejects the `developer` role, the Responses `instructions` field and system
messages after the first turn, so Codex CLI and other Responses clients need `qwen3_8.jinja`. Its path:

- release tarball: `chat_templates/qwen3_8.jinja`;
- source tree: `tools/chat_templates/qwen3_8.jinja`;
- Docker image (`docker/Dockerfile.release`): `/opt/capy-flash/chat_templates/qwen3_8.jinja`.

`python3 tools/replace_chat_template.py ENTRY.ninfer TEMPLATE.jinja OUT_DIR` writes a copy of an artifact with
another embedded template instead (weights unchanged; the unchanged parts are hard links to the input, which is
never modified).

## CPU expert threads

Flash-Next computes the routed experts that miss the VRAM cache on the CPU. `--expert-cpu-threads N` (1..64) sets
the number of CPU threads that do it; unset, it is chosen at load from the CPUs the process may run on (its
affinity mask, so `taskset` and cgroup cpusets count; SMT siblings count once):

- GGUF/ggml block expert banks (the 64 GB GSQ-RCO edition): one thread per allowed physical core;
- NVFP4 expert banks: max(6, cores / 2, 2 per L3 domain);
- both capped at the core count and at 64;
- v0.7.3: when the allowed CPUs have no free SMT sibling (allowed logical CPUs ≤ allowed physical cores, e.g.
  `taskset -c 0-15` on a 16-core / 32-thread CPU, or SMT off) and the count would take every core, one core is
  left to the main thread, which spin-waits on the GPU by default (`NINFER_CUDA_SYNC=spin`) and is not pinned.
  Reported in issue #11: on an EPYC 7F52 with `taskset -c 0-15`, 16 threads dropped decode to 80-90 tok/s, against
  about 144 with 15.

Each thread is pinned to its own physical core, spread over the L3 domains. The startup log names the result:
`routed experts: N CPU threads for the missing experts (auto: <rule>; P physical cores, L logical CPUs allowed, D
L3 domains, M NUMA nodes)`; with v0.7.3's rule the reason says `one core left to the main thread (no free SMT
sibling)`, and with the flag it says `set by the caller (--expert-cpu-threads)`. A value above the allowed physical
cores stops the server at startup (`host MoE pool: fewer physical cores than requested threads`).
`--expert-cpus 0-7,16` (or `NINFER_HOST_MOE_CPUS=0-7,16`) replaces the list with explicit CPU numbers and ranges as `taskset` writes them, SMT siblings allowed
([diagnostics](../capy-flash/diagnostics.md)). Lower the count when the machine runs other work; in a container with a
CPU quota but no cpuset, the affinity mask shows every core, so set it by hand.

capy-flash computes every routed expert the router selects: an expert that misses the VRAM cache is computed on
the CPU, never skipped. Engines that skip low-ranked missing experts (Strata's route tail skip) change the output
slightly and decode faster on slow CPUs; see the README's [benchmarks](../../README.md#benchmarks).

## Connect a client

The server speaks OpenAI Chat Completions and Responses under `/v1` and Anthropic Messages at `/v1/messages` on
`http://HOST:PORT` (`127.0.0.1:8080` by default). With `--api-key KEY` every endpoint except `/health` requires
`Authorization: Bearer KEY` or `x-api-key: KEY`.

- **Model name.** The OpenAI endpoints accept only the server's model id: `--model-id` (`capy-flash` or
  `qwen3.8-27b` with the launchers; without it, the artifact's `metadata.name`); `GET /v1/models` returns it, and
  any other name gets HTTP 404 `model_not_found`. Anthropic Messages accepts any non-empty model name and echoes it.
- **Claude Code** (Anthropic Messages): set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`. With `--api-key`, Claude
  Code's `ANTHROPIC_API_KEY` is sent as `x-api-key` and `ANTHROPIC_AUTH_TOKEN` as a bearer token; both work. The
  model names Claude Code sends are accepted as they are. Claude Code's tool search (`defer_loading`,
  `tool_reference`) is supported ([Anthropic Messages](anthropic-messages.md)).
- **Codex CLI** (Responses): configure a custom model provider whose base URL is `http://127.0.0.1:8080/v1` and
  whose wire API is `responses`, and use the server's model id as the model (Codex's own documentation has the
  configuration keys). Serve with `qwen3_8.jinja` (the launchers do): Codex always sends `instructions`. Its
  encrypted-reasoning `include`, custom tools and the hosted `web_search` tool are handled as described in
  [OpenAI Responses Core](responses.md).
- **OpenCode and other OpenAI-compatible clients**: an OpenAI-compatible provider with base URL
  `http://127.0.0.1:8080/v1` and the server's model id.

## What the engine does

On Qwen3.8-Flash-Next:

- Tensor parallel over PCIe on two GPUs **without peer-to-peer** (pinned-host mailbox, FP32 partial sums); with P2P
  (for example two boards behind a PCIe switch) the prefill exchanges take the direct copy automatically (v0.6).
- A cache of routed experts in VRAM, partitioned by expert id between the two GPUs; missing experts are computed on
  the CPU (NVFP4 kernels in AVX-512 VNNI or AVX2, chosen at startup and identical bit for bit, with the GPU's W4A4
  activation quantization), inside the captured CUDA graph.
- MTP speculative decoding (K = 3, draft vocabulary shortlist; the draft head was re-distilled on this engine's own
  output in v0.6: +18 % decode in Italian, +4-8 % in English and code), INT8 KV with an elastic KV / expert-slot budget up to 262,144 tokens (`--elastic`: the KV pool
  keeps its address space for `--max-context` but backs pages on demand, and unused VRAM becomes extra expert slots),
  prefix cache across turns.
- KV of idle conversations in RAM (v0.6): with a Host KV tier, once a request decodes the KV of every idle retained
  conversation moves to RAM and the VRAM it frees goes back to expert slots at once ([Two GPUs](options.md#two-gpus)).
- Expert residency that follows the conversation (v0.6): a prompt that continues a cached conversation keeps the
  experts its next answer needs, a new conversation only updates their statistics. Prompt chunks of at least
  `--stream-min-tokens` tokens (default 256) stream the non-resident experts to the GPUs over PCIe; shorter ones
  compute them on the CPU. On one GPU, streamed chunks of up to 2,047 tokens leave the experts the previous chunk
  of the same prompt routed least to the CPU, in parallel with the copies (a new prompt assumes uniform routing, v0.11.2). The post-prompt slowdown of v0.5.x (about 250 ms per turn) is gone, and a cold 382-token
  prompt answers in 1.35 s (2.5-2.9 s with the earlier 512-token threshold).
- GGUF i-quant experts (v0.7): IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL and Q2_0 blocks with variable-size expert slots
  per layer, GPU kernels from ggml (via fngine / ninfer-all), CPU kernels in AVX-512 VNNI and AVX2; the 64 GB edition
  takes its experts verbatim from ISTA-DASLab's GSQ-RCO GGUF. With these banks the prefill reads the non-resident
  experts in place from host RAM (zero-copy, from `--stream-min-tokens` tokens), and `--miss-split` is forced off.
  Without `--stream-min-tokens` their threshold is 256 tokens on one GPU and 600 on two (v0.11.2: below 600 tokens
  the CPU is cheaper than copying at tp 2). On one GPU, prompt chunks of 256-2,047 tokens leave part of each
  layer's non-resident experts to the CPU, sized by a cost model so that the CPU and the copies of the others end
  together (`--gguf-hybrid-cost`, [diagnostics](../capy-flash/diagnostics.md)); wider chunks copy them all.
- `--turn-anchors` is 0 by default at tp 2 (any model) and on Flash-Next. On Flash-Next each anchor costs one more
  prefill pass per agentic turn (about 0.8 s on the test machine); at tp 2 it also holds one more StateImage per turn
  on both ranks, in the checkpoint pools the turn closures need (with it, an agentic replay of three sessions on the
  27B lost every turn closure after the first session). The cost is that a cancelled or edited turn re-reads more of
  the conversation (`--turn-anchors 1` restores the anchor). On one GPU the 27B keeps the anchor by default.
- NVIDIA's FP4-Local checkpoint runs too (recipe `qwen3_8_flash_next_fp4_local`), about 19 % slower than the default
  artifact.
- Empty-reply guards (v0.7, every model, on by default): a terminal token drawn right after `<think>` is redrawn
  (`--reasoning-eos-guard`), and a reply that ends inside its reasoning is closed and continued once
  (`--reasoning-close-retry`); `--no-reasoning-eos-guard` and `--no-reasoning-close-retry` turn them off.
- Output is not bit-reproducible run to run: which experts sit in VRAM (GPU path) or not (CPU path) depends on
  timing, and long streamed prefills differ in the last bits. For exact A/B comparisons freeze the residency
  ([diagnostics](../capy-flash/diagnostics.md)).

On Qwen3.8-27B:

- The same binary serves the dense Qwen3.8-27B entirely on the GPUs, at `--tp 2` exactly like ninfer-tp2 v0.4.9
  ([Two GPUs](options.md#two-gpus)), with MTP or DFlash2, Vision and the context cache in RAM (`--profile 27b`).
- 3-bit GSQ weights (v0.7, `q3_g128_fp16`): tensor-core kernels with int8 activations in the prefill, a chosen
  trade-off: prefill about 3.5x faster than 16-bit activations, perplexity within 0.5 % of them (`NINFER_Q3_A8=0`
  keeps the 16-bit, exact path; [diagnostics](../capy-flash/diagnostics.md)); `--profile 27b-16gb` fits one 16 GB card (int8 KV, 65,536 tokens, Vision, 13.9 GB of
  VRAM).

On every model, the context cache keeps retained conversations in VRAM, in RAM (Host tiers) and, opt-in, on disk
([Disk tier](options.md#disk-tier-of-the-context-cache)); a request's KV reservation is a lease that grows with its answer
([Start the server](start.md)).

Environment variables (production settings, A/B switches, dumps): [docs/capy-flash/diagnostics.md](../capy-flash/diagnostics.md).
