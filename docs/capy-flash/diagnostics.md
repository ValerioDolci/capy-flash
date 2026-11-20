# Environment variables

The engine (`ninfer-serve`, `ninfer`, and the library under `src/`) reads the environment variables below. Command-line
flags are documented in [cli.md](../cli.md) and [serving/options.md](../serving/options.md). The environment covers three things:
the production settings, A/B switches kept so that a change can be measured again on one binary, and validation
and diagnostic aids.

No engine variable needs to be set in production: the release launchers start `ninfer-serve` with a `--profile`
([serving/capy-flash.md](../serving/capy-flash.md#launchers-and-profiles)).

`NINFER_RAM_GUARD` stays at its default (on). Leave every other variable unset in a server.

## How to read the tables

- **Class**: *production* (set or kept on in serving), *performance* (an adopted optimization with a switch back to
  the old behaviour), *opt-in* (off by default: a faster path that changes the arithmetic), *diagnostic* (dumps,
  logs, probes, self-tests), *validation* (changes execution or tokens on purpose, for exactness checks),
  *benchmark* (kernel candidate selection).
- **Results**: *numerics*: the arithmetic changes, so logits and possibly tokens change. *Same bits*: the code
  states, or a test checks, that outputs are bit-identical. *Timing*: only scheduling or placement changes.
  *Tokens*: generated tokens are replaced on purpose.

**Timing and numerics are coupled in the expert offload backend.** Which experts are resident on the GPU depends on
copy timing. A resident expert (GPU) and a missing one (CPU) use the same W4A4 activation codec and the same BF16
roundings, but sum in a different order, so a change of timing, including the extra synchronization of a dump, can
still change tokens. Runs are not guaranteed to be bit-reproducible
from run to run (long prefills differ even with frozen residency). For exact A/B comparisons, freeze the
residency (`NINFER_QWEN4_EXPERT_FREEZE`, see below).

## 1. Production

| variable | values (default) | effect | class | results |
|---|---|---|---|---|
| `NINFER_HOST_ISA` | `auto` (default), `avx2`, `avx512` | Instruction set of the CPU expert kernels: `auto` takes AVX-512 VNNI where the CPU has it, else AVX2 + FMA + F16C (with the GGUF dots in AVX-VNNI `vpdpbusd` where the CPU has it, Alder Lake and later); `avx2` forces the AVX2 kernels on an AVX-512 CPU (to test them); a value the CPU cannot run stops the server at startup. The log line `qwen4 experts: host expert kernels …` names the choice. Both compute identical bits (`ninfer_host_nvfp4_isa_test`; the AVX-VNNI dots against the AVX2 ones in `ninfer_host_gguf_test`); only speed differs. | performance | timing |
| `NINFER_RAM_GUARD` | `0` / unset (on) | Startup check that MemAvailable ≥ the page-locked host weights (64.6 GiB for Flash-Next) + max(2 GiB, 5 % of MemTotal). Otherwise the server refuses to start and prints the numbers. Without `--host-pinned-max-gib` it also caps the page-locked weights outside free CMA memory (#40, see serving/capy-flash.md «RAM»). `0` skips the check and the cap. | production | none |
| `NINFER_HOST_TIER_RAM_GUARD` | `0` / unset (on) | Startup check that MemAvailable ≥ the page-locked RAM tiers of the context cache (`--host-state-slots`, `--host-kv-mib`) + max(4 GiB, 5 % of MemTotal); otherwise the server stops with the sizes. `0` skips it. | production | none |
| `NINFER_KVRAM_CACHE_BLOCKS` | blocks (unset = 4096) | Same as `--kv-ram-cache-blocks` (the flag wins). With `--kv-ram-active`: 4-token blocks of the VRAM block cache per text attention layer. Raised automatically so the region also holds one layer of `--max-context` (the prefill staging); a larger value keeps more of the indexer's past selections on the GPU at the cost of VRAM ([cost per block](../maintainer/active-kv-in-ram.md#memory)). | tuning | none (same tokens) |
| `NINFER_KVRAM_REGION_TOKENS` | tokens (unset = 262144) | Same as `--kv-ram-region-tokens` (the flag wins). With `--kv-ram-active`: the tokens of one layer the VRAM staging region holds for prompt chunks (capped at `--max-context`). A longer visible context stages the union of each chunk's selections, by column ranges, instead of whole pages. Lower it to save VRAM at long `--max-context` or to exercise that path |
| `NINFER_KVRAM_STATS` | `1` / unset | With `--kv-ram-active`: count the block cache's outcomes (hits, blocks shared by the columns of an MTP window, misses, uncacheable tail blocks, collisions), the staged prefill pages, and how many distinct blocks a prompt chunk's selections touch out of the visible ones, and print them summed over the layers before each prompt (`active KV stats`). Synchronizes the GPU at each prompt; diagnostics only. | diagnostics | timing |
| `NINFER_HOST_PINNED_MAX_GIB` | GiB, may be fractional (unset = no limit) | Same as `--host-pinned-max-gib` (the flag wins): at most this much of the host-mapped weights (Flash-Next expert banks) is page-locked; the banks of the least used layers beyond it stay in pageable RAM (CPU reads them in place, GPU copies go through a pinned staging ring, no prefill streaming for them). Unset: the whole page-locked allocation is tried first and the fallback starts only if the driver refuses it (Windows WDDM / WSL2). With `--kv-ram-active` its Host KV planes count in the limit. A value that is not a positive number stops the server at startup. | production | numerics on prompts that stream experts (the CPU computes the pageable layers' experts); slower where pageable |
| `NINFER_QWEN4_EXCLUSIVE_LAYERS` | layers (unset or `0` = off; any other text than a number stops the start) | Same as `--expert-exclusive-layers` (the flag wins). 64 GB edition (ggml block expert banks) on a machine with 64 GB of RAM and a desktop beside the server: in the first N expert layers the experts the GPUs hold get no page-locked host copy (an evicted one is first written back to a free host slot), which saves several GiB of RAM; the startup log prints how much. N must stay below the first layer the prefill lends, otherwise the server refuses to start and names the limit. Decode is about 2-3 % slower, and the tokens differ from a run without it (the GPUs hold other experts, the CPU computes the rest); with it on, `NINFER_QWEN4_FILL_PUBLISH_LAG` defaults to `1` and `-1` is refused at startup, so the start-up warm-up request is greedy and two starts give the same greedy text. | production | residency (tokens change) |
| `NINFER_HOST_STAGING_MIB` | MiB (128) | Size of the pinned staging ring per GPU that fills expert slots from pageable banks; only allocated when some bank is pageable or `NINFER_QWEN4_EXCLUSIVE_LAYERS` is set. At least one expert, at most the fill depth. If the allocation fails the server stops and suggests lowering `--host-pinned-max-gib` or this value. | performance | timing |
| `NINFER_CUDA_SYNC` | `spin` (default), `blocking`, `yield`, `auto` | CUDA device synchronization schedule ([cli.md](../cli.md#cuda-synchronization)). | performance | timing (host CPU use) |

`CUDA_VISIBLE_DEVICES=0,1` in the serve line is the standard CUDA variable, not an engine setting.

About A4: the source describes the two paths as matching ("CPU ≈ GPU numerics"), and the README calls them
numerically identical. Tests check the AVX-512 and AVX2 A4 code against scalar A4 code bit for bit
(`ninfer_host_nvfp4_isa_test`). No test in the tree compares CPU-A4 output with GPU output bit for bit, and summation
order differs between the two.

## 2. Validation gate

| variable | values (default) | effect | class | results |
|---|---|---|---|---|
| `NINFER_QWEN4_VALIDATION` | `1` / unset | Must be set for the token-changing knobs `NINFER_QWEN4_FORCE_TOKENS`, `NINFER_QWEN4_TFDUMP_FORCE`, `NINFER_QWEN4_EXPERT_FREEZE` and `NINFER_QWEN4_REPLAY`. Without it the Program throws and names the knob, so these cannot reach a server by accident. | validation | none by itself |

## 3. Performance switches (adopted defaults; unset = production behaviour)

Expert cache and prompt streaming:

| variable | values (default) | effect | class | results |
|---|---|---|---|---|
| `NINFER_GGUF_HYBRID_COST` | `35,3.6,100` (default) | Same as `--gguf-hybrid-cost` (the flag wins). The cost model of the hybrid prefill of GGUF (ggml block) expert banks on one GPU (prompt chunks of 256-2,047 tokens leave part of each layer's non-resident experts to the CPU executor while the GPU copies the others), in microseconds: CPU per expert, CPU per routed token of an expert, PCIe copy per expert (measured on a Ryzen 9 9900X with DDR5 and PCIe 5.0 x8; slower RAM or a faster link want a smaller CPU share). | performance | numerics (CPU/GPU split) |
| `NINFER_QWEN4_PLE_TIMING` | off (default), `1` | Prints the duration of every prefill PLE gather and prefetch to stderr (`ple_timing ...`). | diagnostics | none |

CPU expert executor (all bit-identical by construction):

| variable | values (default) | effect | class | results |
|---|---|---|---|---|
| `NINFER_HOST_MOE_CPUS` | comma list of CPU numbers and ranges (`0-7,16`); unset = automatic | Same as `--expert-cpus` (the flag wins). Replaces the automatic CPU list of the executor: one thread per listed CPU, pinned in that order (SMT siblings may be listed); the `routed experts: N CPU threads …` startup line still prints the automatic or requested count. Unset: `--expert-cpu-threads` threads, or the automatic count, each thread pinned to its own physical core among those the process may run on (`taskset`, cgroup cpuset), spread over their L3 domains. See [serving/capy-flash.md](../serving/capy-flash.md#cpu-expert-threads). | performance | timing |
| `NINFER_HOST_GGUF_BIG_GATHER` | unset (off), `1` | 64 GB edition on a hybrid Intel CPU without AVX-512 (Alder Lake / Raptor Lake: P- and E-cores, `/sys/devices/cpu_core/cpus`): CPU expert threads on P-cores decode IQ3_XXS / IQ3_S rows with `vpgatherdd` grid lookups (on a P-core 15 / 33 % faster than the scalar lookups, on an E-core ~2x slower, so E-cores keep them); prints `GGUF big-core decoder on N of M threads`. No effect on AVX-512 or on CPUs without E-cores. | performance | same bits |
| `NINFER_HOST_GGUF_ECORE_TAIL` | `0` (default, off), items | With `NINFER_HOST_GGUF_BIG_GATHER=1` on a hybrid CPU: E-core threads claim no gate/up or down item of a CPU expert call once fewer than N remain, so the end of each phase runs on P-cores (an E-core item takes ~3x a P-core one). i9-14900KF, 24 threads: 24 best of 16 / 24 / 32. | performance | same bits |
| `NINFER_QWEN4_FILL_SM` | thread blocks (unset = off) | Expert cache fills are copied by a kernel of N blocks that reads the host bank in place, instead of the copy engine; bounds the reads in flight so small PCIe reads (TP exchange, CPU partials) do not wait behind a fill. | performance | same bits |
| `NINFER_QWEN4_FILL_PUBLISH_LAG` | `-1` (default), `1`, `2`, `3` | When an expert cache copy is published. `-1`: at the first window after the copy completed, so which experts the GPU holds, and which ones the CPU computes with another rounding, follows copy speed. `1`: at the window fixed when the copy was issued (the next one for a copy issued inside a window, the one after for a copy issued between windows), which waits for it if needed, so the residency, and the greedy text, depend on the routing alone; `2`, `3` publish one or two windows later. With `1`..`3` the server's start-up warm-up request is greedy too, and the early reclaim of lent layers waits for its refill. | performance | residency (tokens may change) |
| `NINFER_TP_EXCHANGE_PREFETCH_KIB` | KiB (12288), `0` = off | `--tp 2`: prefetch window of the decode exchange buffers. In the Qwen4Exp hyper-connection block the next layer's mixer prefetch takes up to half the L2 whatever the value; there the value bounds only the router and shared-expert prefetch, and `0` switches the block's prefetch off. | performance | same bits |
| `NINFER_TP_XIN_KERNEL` | unset / `1` (on, default), `0` (off: the copy engine) | `--tp 2` on a pair without peer access: the receive leg of the prefill exchange (MoE output and mixer output projection) is read from the peer's pinned buffer by a 16-block kernel on a stream of the device's greatest priority instead of a host-to-device copy, so it does not queue behind streamed expert weights on the copy engine. Pieces that are not 16-byte aligned keep the copy; a pair with peer access keeps the copy. | performance | same bits |

### Opt-in, off by default

Faster paths that change the arithmetic (lossy, or a different summation order). Unset keeps the default path.

| variable | values (default) | effect | class | results |
|---|---|---|---|---|
| `NINFER_Q8_A8` | `32` (or `1`), `128`, unset / `0` (off) | Same as `--q8-a8` (the flag wins). Q8_G32 projections at prefill widths (64 tokens and more; the four dense shapes of one Flash-Next rank at tp 2) on int8 tensor cores: activations quantized per token and 32- or 128-value group into the optional linear workspace (A16 tiles when it has no room). It overrides the artifact's activation policy: the published Flash-Next artifacts declare these weights `A16Only`. Decode and verification calls stay below 64 columns and keep the A16 tiles. Prefill +7-8 % on 2x RTX 5070 Ti, perplexity +0.21 % (NVFP4), +0.004 % (64 GB edition). | opt-in | numerics |
| `NINFER_ATTN_PV_F16ACC` | `1` / unset (off) | Same as `--attn-pv-f16acc`. INT8-KV prefill attention (the tiled kernel, more than 256 columns): P x V of each 64-key tile accumulates in FP16 (GeForce Blackwell runs the FP32-accumulating FP16 MMA at half rate) and is added to the FP32 accumulator. Runs on the ping-pong kernel (`tiled_pp_mma.cuh`). A tile overflows once its sum of p x v exceeds 65504 (`ninfer_softmax_attention_test`: equal scores and V of about 1,080 and more). | opt-in | numerics |

## 4. Numerical paths of the TP2 exchanges

Defaults are the adopted values. `int8` is an opt-in, less precise payload; the other values select equivalent
transports.

| variable | values (default) | effect | class | results |
|---|---|---|---|---|
| `NINFER_TP_AR_PAYLOAD` | `fp16` (default), `int8` | Same as `--tp-ar-payload`. Payload of the TP2 prefill MoE exchange (column groups of 2,048 that overlap the expert GEMM; the FP16 payload carries v·2^-2, saturated to the FP16 range, rescaled exactly on receipt). Decode MoE exchanges are always FP32. `int8` (opt-in, lossy): 32-row blocks of int8 codes with binary16 scales, 2,720 instead of 5,120 bytes per 2,560-row column, also for the mixer output projection; columns whose rows are not a multiple of 32 use FP16. `int8` applies to Flash-Next only: the 27B has no MoE exchange and its mixer output projections are not Q8/MXFP8, so the value has no effect there. | opt-in | numerics |
| `NINFER_TP_HOST_EXCHANGE` | auto (default), `1` | Eager (prefill) exchanges go through pinned host buffers asynchronously. Auto: on without P2P or with `--elastic`, off with P2P (direct peer copy, faster); `1` forces it on (a board whose P2P is faulty). | production | timing; prefill exchange precision |
| `NINFER_TP_PEER_OVERLAP` | auto (default), `force` | Transport of the 27B prefill overlap's split-phase all-reduces on P2P boards: `ops::PeerLanes` pull the peer's operand device-to-device on a side stream per rank, so the pull overlaps the remaining compute (attached with P2P when no HostExchange is). `force` attaches the lanes without P2P too, driver-staged and host-ordered (a bit test of the P2P path, no overlap). | diagnostic (test) | same bits |

## 5. Diagnostics: observation (dumps, logs, probes)

None of these changes the arithmetic. They add I/O or synchronization, which can shift expert residency (see the
note above). Variables marked *eager* need `--no-cuda-graph`.

| variable | values | output |
|---|---|---|
| `NINFER_QWEN4_EXPERT_TRACE` | file | One JSON line per expert window: hits, CPU experts and time, fills, device timestamps. |
| `NINFER_QWEN4_ROUTING_DUMP` | file | The routing of every observed layer (binary). |
| `NINFER_CONSTRAINT_MASK_STATS` | `1` / unset | Structured output: every 256 host mask fills, the fills, positions per fill and microseconds per fill and per position on stderr (`[constraint] ...`). |
| `NINFER_QWEN4_ROUND_LOG` | file | One line per decode round: extent, accepted drafts, round time, host gap, misses, CPU and GPU-share experts, fill wait. |
| `NINFER_QWEN4_MTP_CALIB` | file | One line per verified MTP round: extent, accepted, and the proposals' own log-probabilities. At `--tp 2` a draft shortlist head stays on rank 0 (the split head has no log-probabilities). |
| `NINFER_QWEN4_TOP2` | file | The target's top-2 logits for every decoded or committed verification column. |
| `NINFER_QWEN4_MTP_PROBE` | file (eager; throws under graphs) | The inputs of every MTP-layer expert call. |
| `NINFER_QWEN4_TFDUMP` | directory | tfdump/1 files: logits per scored or decoded position, plus the wide residual after each layer in eager prefill. |
| `NINFER_QWEN4_TFDUMP_LAYERS` | `all` (default), `none`, `i,j,...` | Which layers' residuals to dump. |
| `NINFER_QWEN4_TFDUMP_LOGITS` | on (default), `0` | `0` dumps no logits. |
| `NINFER_QWEN4_TFDUMP_MTP` | `1` (eager) | Every prompt column also runs the MTP layer; draft logits go to `mtp_logits/`. |
| `NINFER_QWEN4_TFDUMP_STRIDE` | `1` (default), n | Logits only at positions divisible by n (KL over long windows). |
| `NINFER_QWEN4_TFDUMP_SEQUENCES` | `1` | One subdirectory per scored sequence. |
| `NINFER_QWEN4_TFDUMP_COMPACT` | `1` | Smaller tfdump records (no wide residuals beyond what the layers list asks). |
| `NINFER_QWEN4_TFDUMP_TOPK` | n (0 = all) | Keep only the top-n logits per position. |
| `NINFER_LEASE_DEBUG` | set | Log lines for the KV lease demotions run by admission pressure. |
| `NINFER_VISION_DUMP` | directory | Each encoded image/video item: `itemN.patches.bf16` (the patches the tower received, BF16 [patches, 1536]), `itemN.merged.bf16` (its merged embeddings, BF16 [merged, H]) and `itemN.json` (grid, counts). Synchronizes the encoding device per item. Compare with an independent tower reference on the same patches. |
| `NINFER_QWEN4_XDUMP` | directory | The two rank partials of each TP2 prefill exchange, before the sum. |
| `NINFER_QWEN4_XDUMP_STRIDE` | `16` (default), n | Column stride of the exchange taps. |
| `NINFER_QWEN4_REPLAY_RECORD` | file | Records `req` / `win` lines (prompt, MTP rounds) for the deterministic fast bench. |
| `NINFER_HOST_MOE_CALL_LOG` | file | Per-call, per-thread timestamps of the CPU expert executor (about 20 ns per stamp). |
| `NINFER_HOST_MOE_DUMP` | file | Every served decode call with host experts (layer, IDs, alpha, x, out) and the partial the device actually read. Written at exit. |
| `NINFER_HOST_MOE_DUMP_MAX` | `40000` (default) | Maximum number of dumped calls. |
| `NINFER_TP_RANK_PROBE` | log file | After every TP2 MTP verification, compares the two ranks' hidden state bit for bit and logs mismatches. No measurable cost. |

## 6. Validation: controlled execution

These variables change execution on purpose so that runs can be compared exactly. The ones marked *gated* require
`NINFER_QWEN4_VALIDATION=1`.

| variable | values | effect | results |
|---|---|---|---|
| `NINFER_CONSTRAINT_DEAD_END_AFTER` | N (unset = off) | Read only by the test program `ninfer_engine_constraint_real_test deadend` (the engine and `ninfer-serve` never read it): from the N-th output token on, every constrained request of the test reports a dead end (no legal token), exactly as the matcher would; the request fails with `ConstraintDeadEnd` and the other requests continue. | test only |
| `NINFER_QWEN4_EXPERT_FREEZE` | `1`, `all` (gated) | `1`: residency is decided by the prompt windows alone, each prompt window waits for its fills, and decode windows insert nothing, so an MTP run and its no-draft control run each expert on the same side. `all`: nothing is ever inserted or evicted (fast bench). | numerics (deterministic residency) |
| `NINFER_QWEN4_FREEZE_FLUSH` | any value | With `EXPERT_FREEZE=1`, the post-prompt refill lands before the first decode window. | residency |
| `NINFER_QWEN4_EXPERT_SYNC` | `1` | Every window waits for its fills, so residency depends on routing only, not on copy timing. | residency, slower |
| `NINFER_QWEN4_EXPERT_SNAPSHOT` | file | Each request starts from a saved residency block (request r uses block r mod blocks). | residency |
| `NINFER_QWEN4_EXPERT_SNAPSHOT_SAVE` | file | Saves the residency at each request's first decode window. | none |
| `NINFER_QWEN4_FORCE_TOKENS` | file (gated) | The committed token of each decode step is the next token of the matching line. Single-token commits only: use with `NINFER_QWEN4_MTP_NO_DRAFTS=1`. | tokens |
| `NINFER_QWEN4_TFDUMP_FORCE` | IDs file (gated) | Teacher-forced decode of the given sequence, prompt included. | tokens |
| `NINFER_QWEN4_REPLAY` | file (gated) | Deterministic replay of a recorded run: forced tokens and recorded MTP drafts and extents (fast bench). | tokens |
| `NINFER_QWEN4_REPLAY_MODEL_DRAFTS` | `1` | With `REPLAY`: where no window is recorded, keep the model's own drafts (forced targets with realistic rejections). | tokens |
| `NINFER_QWEN4_MTP_NO_DRAFTS` | `1` | Every MTP round verifies the anchor alone: the no-draft control of an MTP run. On a routed-expert model (Flash-Next) compare with `NINFER_QWEN4_EXPERT_FREEZE=1` and `NINFER_QWEN4_FREEZE_FLUSH` in both runs: otherwise the different windows change the residency, and with it the tokens. | same tokens (greedy; frozen residency on Flash-Next) |
| `NINFER_QWEN4_MTP_ORACLE` | file | Drafts are taken from whole reference sequences, so rounds are fully accepted. | same tokens (greedy) |
| `NINFER_QWEN4_MTP_EXTENT_CAP` | k | Verify at most k proposals per round in the K-wide window (measures the cost of a column). | same tokens (greedy) |
| `NINFER_QWEN4_HC_GENERAL` | `1` | Q8/NVFP4 hyper-connection mixers take the general (non-fused) route, to separate route from weight format. | may change rounding |
| `NINFER_QWEN4_FUSED` | `0` / unset (on) | Every Qwen4Exp block takes the generic chain (Linear, row copies, elementwise Ops) instead of the fused Ops registered for its geometry (hidden 5120: GDN input projection and gating, QSA input projection, SwiGLU). Flash-Next (hidden 2560) has none; a dense Qwen3.5 model's blocks always take their fused Ops. | numerics |

## 7. Precision defaults: speed against distance from BF16

Each default below is a chosen trade-off, not a kernel selector: it runs faster than the exact path at the price of a
small distance from BF16 activations. The alternative stays available for comparisons.

| variable | values (default) | effect | results |
|---|---|---|---|
| `NINFER_Q3_A8` | `1` / unset (A8, default), `0` (A16) | Activations of the 3-bit Qwen3.8-27B prefill projections at T ≥ 64 (T < 64 is always A16). Default A8: int8 activations in 64-column tiles, prefill about 3.5x faster than A16, perplexity within 0.5 % of A16; the route of `--profile 27b-16gb`. `0` = A16: exact BF16 activations, the closest to BF16, with the slower prefill. | `0` changes rounding |

The other precision defaults have no switch of their own:

- Qwen3.8-27B NVFP4: where the artifact declares a weight `AllowA4`, its projection runs with NVFP4 activations (A4)
  at the token widths its shape's route table selects, otherwise with BF16 activations (A16); `A16Only` weights
  always run A16. The artifact's activation policy chooses ([weight-conversion.md](../weight-conversion.md)); no
  measurement of the distance is recorded in these documents.
- Qwen3.8-Flash-Next: the routed experts run W4A4 (NVFP4 activations with the calibrated per-layer divisor) on the
  GPU and on the CPU alike (section «Timing and numerics are coupled» above); the dense Q8 projections run with BF16
  activations (A16), and `NINFER_Q8_A8` (section 3, opt-in) trades +7-8 % prefill for perplexity +0.21 % (NVFP4) /
  +0.004 % (64 GB edition).

## 8. Two-GPU transport

The defaults are right; these are for A/B runs and tests:

| variable | values | effect | class | results |
|---|---|---|---|---|
| `NINFER_TP_MAILBOX_PROBE` | `off`, `fail` | Skip the startup mailbox probe, or force the fallback to copies. | diagnostic | timing |
| `NINFER_TP_MAILBOX_FAULT` | `draft`, `any` | Simulates a mailbox hang to test the degradation path. | diagnostic (test) | timing |

For `--spec dflash2` only (the Qwen3.8-27B DFlash2 drafter; the `--profile` configurations draft with MTP):

| variable | values | effect |
|---|---|---|
| `NINFER_TP_SELECTOR` | `primary` | DFlash2 candidate selector on rank 0 instead of rank 1: rounds about 0.2 % faster, 250 MB less free memory on rank 0, the rank that bounds the KV capacity. Same tokens. |

## 9. Read by tests and the validation driver only

These are not read by the server. The list is not exhaustive; see `tests/README.md`.

| variable | used by |
|---|---|
| `NINFER_TEST_ARTIFACT`, `NINFER_TEST_QWEN4_ARTIFACT`, `NINFER_TEST_QWEN4_VISION_ARTIFACT` | real-artifact tests (skipped when unset) |
| `NINFER_TEST_FP4_LOCAL_CHECKPOINT` | the FP4-Local checkpoint directory whose real tensors the conversion and PLE gather tests compare (skipped when unset) |
| `NINFER_TEST_QWEN4_REUSE_ONLY`, `NINFER_PREFIX_REAL_SCENARIO`, `NINFER_TEST_PORT`, `NINFER_TEST_SCRATCH_DIR` | integration test selection and setup |
| `NINFER_TEST_TOKENIZER_DIR`, `NINFER_TEST_SCHEMA_REPORT`, `NINFER_TEST_TOKENS_OUT` | structured output tests: Qwen3.8 tokenizer files, JSONL of constrained results for `validate_constraint_report.py`, greedy token ids of the `equiv` phase |
| `NINFER_OP_REPORT_STATS` | op tests print statistics |
| `NINFER_DRIVER_TEMPS` | validation driver: one sampling temperature per prompt line |
| `NINFER_DRIVER_STOPS` | validation driver: stop at the model's default stop tokens |
