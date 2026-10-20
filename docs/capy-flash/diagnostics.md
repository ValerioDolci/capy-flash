# Environment variables

The engine (`ninfer-serve`, `ninfer`, and the library under `src/`) reads the environment variables below. Command-line
flags are documented in [cli.md](../cli.md) and [serving.md](../serving.md). The environment covers three things:
the production settings, A/B switches kept so that a change can be measured again on one binary, and validation
and diagnostic aids. `Tnn` tags refer to [the development log](development-log.md).

No engine variable needs to be set in production:

```
CUDA_VISIBLE_DEVICES=0,1 ninfer-serve <artifact> --tp 2 --devices 0,1 --spec mtp --draft-tokens 3 \
  --kv-dtype int8 --elastic --max-context 262144 --kv-capacity 262144 --max-concurrency 1
```

`NINFER_HOST_MOE_A4` is on by default since v0.5.0 (it was opt-in before: keep it on) and `NINFER_RAM_GUARD` stays
at its default (on). Leave every other variable unset in a server.

## How to read the tables

- **Class**: *production* (set or kept on in serving), *performance* (an adopted optimization with a switch back to
  the old behaviour), *legacy A/B* (restores an older numerical path), *diagnostic* (dumps, logs, probes, self-tests),
  *validation* (changes execution or tokens on purpose, for exactness checks), *benchmark* (kernel candidate
  selection).
- **Results**: *numerics*: the arithmetic changes, so logits and possibly tokens change. *Same bits*: the code
  states, or a test checks, that outputs are bit-identical. *Timing*: only scheduling or placement changes.
  *Tokens*: generated tokens are replaced on purpose.
- **Scope**: *capy* is a variable added for Qwen3.8-Flash-Next (the Qwen4Exp model family in the code). *ninfer* is
  inherited from ninfer or ninfer-tp2 for other models.

**Timing and numerics are coupled in the expert offload backend.** Which experts are resident on the GPU depends on
copy timing. Without A4, a resident expert (GPU, W4A4) and a missing one (CPU, W4A8) give different results, so any
change of timing, including the extra synchronization of a dump, can change tokens (T98, T128). With A4
(the default, `NINFER_HOST_MOE_A4`) both paths use the same activation codec. Runs are still not guaranteed to be bit-reproducible
from run to run (T158: long prefills differ even with frozen residency). For exact A/B comparisons, freeze the
residency (`NINFER_QWEN4_EXPERT_FREEZE`, see below).

## 1. Production

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_HOST_MOE_A4` | `0` / unset (on, since v0.5.0) | The CPU expert path uses the GPU kernel's W4A4 activation codec and its BF16 roundings of gate/up and h, instead of int8 activations (W4A8). Removes the CPU/GPU mix that changed answers with residency (T98). AVX-512 implementation at no measurable cost (T148). **On by default since v0.2.1; keep it on.** | production | numerics | capy |
| `NINFER_RAM_GUARD` | `0` / unset (on) | Startup check that MemAvailable ≥ the page-locked host weights (64.6 GiB for Flash-Next) + max(2 GiB, 5 % of MemTotal). Otherwise the server refuses to start and prints the numbers. `0` skips the check (T156). | production | none | capy |
| `NINFER_CUDA_SYNC` | `spin` (default), `blocking`, `yield`, `auto` | CUDA device synchronization schedule ([cli.md](../cli.md#cuda-synchronization)). | performance | timing (host CPU use) | ninfer |

`CUDA_VISIBLE_DEVICES=0,1` in the serve line is the standard CUDA variable, not an engine setting.

About A4: the source describes the two paths as matching ("CPU ≈ GPU numerics"), and the README calls them
numerically identical. Tests check the AVX-512 A4 code against the scalar A4 code bit for bit
(`NINFER_HOST_A4_SELFTEST`). No test in the tree compares CPU-A4 output with GPU output bit for bit, and summation
order differs between the two.

## 2. Validation gate

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_QWEN4_VALIDATION` | `1` / unset | Must be set for the token-changing knobs `NINFER_QWEN4_FORCE_TOKENS`, `NINFER_QWEN4_TFDUMP_FORCE`, `NINFER_QWEN4_EXPERT_FREEZE` and `NINFER_QWEN4_REPLAY`. Without it the Program throws and names the knob, so these cannot reach a server by accident (T37). | validation | none by itself | capy |

## 3. Performance switches (adopted defaults; unset = production behaviour)

Expert cache and prompt streaming:

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_QWEN4_FILLS_AFTER_HOST` | `1` (default), `0` | Cache fills for layer l are issued after the CPU finishes layer l, not at its doorbell, so the DMA does not take DRAM bandwidth from CPU experts (T35). | performance | timing | capy |
| `NINFER_QWEN4_REFILL_ORDER` | `1` (default), `0` | After a prompt, refill lent layers hottest-first across layers; `0` refills layer by layer (T69). | performance | timing (residency) | capy |
| `NINFER_QWEN4_EARLY_RECLAIM` | `1` (default), `2`, `0` | Reclaim lent layers before a short tail chunk: `1` overlaps the refill with the chunk, `2` makes the chunk wait for it, `0` turns it off (T69). | performance | timing (residency) | capy |
| `NINFER_QWEN4_PROMPT_KEEP` | `1` (default), `0` | Prompt windows change neither the residency nor the decayed frequencies, and lent layers refill with the experts they held when lent (T211); `0` restores the T69 rule (the prompt's hottest experts become resident and refill the lent layers). Ignored under `NINFER_QWEN4_EXPERT_FREEZE=1`, which keeps the T69 rule. | performance | timing (residency) | capy |
| `NINFER_QWEN4_PROMPT_KEEP_MIN_CACHED` | `0.5` (default) | Share of the prompt that must be restored from the context cache for the prompt to count as a continuation (keeps the residency); below it the prompt is a new conversation and updates the residency and frequencies (T211b). `0` = every prompt keeps. | performance | timing (residency) | capy |
| `NINFER_QWEN4_PROMPT_NEW_CAP` | `-1` (default), `0` = slot count, `N` | A new conversation's prompt counts are applied when its first decode window begins: `-1` to the frequencies only, otherwise also that many targets per layer and GPU made resident (T211b). | performance | timing (residency) | capy |
| `NINFER_QWEN4_LEND_KEEP_ELASTIC` | `1` (default), `0` | A prompt borrows only base slots; the elastic slots stay resident through the prompt (T69; with `NINFER_QWEN4_PROMPT_KEEP=0` the prompt fills them with its hottest experts). | performance | timing (residency) | capy |
| `NINFER_QWEN4_SCALE_WORKSPACE` | `1` (default), `0` | The prompt workspace borrows the layers for the narrowest planned width that fits the chunk; `0` always uses the widest (T69). | performance | timing (residency) | capy |
| `NINFER_QWEN4_PROMPT_FREQ_WEIGHT` | `1` (default), `0` | Prompt routing counts are weighted by recency in the eviction policy; `0` gives full weight (T69). | performance | timing (residency) | capy |
| `NINFER_QWEN4_BOOST_WINDOWS` | `24` (default), `0` = off | Number of decode windows after a prompt with a raised insert cap (T69 C6; on since T211: replay rounds -2.9 %). | performance | timing (residency) | capy |
| `NINFER_QWEN4_BOOST_CAP` | `8` | Insert cap per layer during the boost windows; `0` = no cap. | performance | timing (residency) | capy |
| `NINFER_QWEN4_TP2_HALF_RING` | on (default), `0` | At TP2 each GPU streams only its expert partition, so the prompt ring is half size; `0` keeps the 1-GPU size (T56). | performance | same bits | capy |
| `NINFER_QWEN4_STREAM_DEFER` | on (default), `0` | At TP2 the next layer's expert copies wait for this layer's MoE all-reduce on the shared link; `0` issues them at release (T56). | performance | same bits | capy |
| `NINFER_MOE_GROUPED_PIECES` | `1` / unset (off) | `1` computes the prefill gate/up per piece instead of over the whole call (T56). | performance | same bits | capy |
| `NINFER_HC_SCALAR` | `1` / unset (off) | `1` keeps the scalar hyper-connection kernels at every width instead of the vectorized ones (T56). | performance | same bits | capy |
| `NINFER_T53_PLE_PREFETCH` | on (default), `0` | Starts page reads of the next round's PLE rows right after the round's sync (T53). | performance | timing | capy |
| `NINFER_QWEN4_PLE_NEXT_PREFETCH` | on (default), `0` | Prefill: a helper thread starts the page reads of the next chunk's PLE rows while the GPUs run the current chunk (T165-T166). | performance | timing | capy |
| `NINFER_QWEN4_PLE_RESIDENT_SCALES` | on (default), `0` | The PLE table's BF16 row multipliers (640 MB) are read once at load and kept in RAM, so a cold row costs one page read instead of two; `0` reads them through the mapping (T171-T172). | performance | same bits | capy |
| `NINFER_QWEN4_PLE_TIMING` | off (default), `1` | Prints the duration of every prefill PLE gather and prefetch to stderr (`ple_timing ...`). | diagnostics | none | capy |
| `NINFER_QWEN4_ROUTER_MMA` | on (default since v0.5.3), `0` | Router logits of prompt chunks (64+ columns) on tensor cores (BF16 MMA, FP32 accumulation) instead of the SIMT kernel: 55k-token prefill 3.66k -> 3.91-4.04k tok/s (T183-T184). Same FP32 contract, another summation order: expert selections can differ near ties, so `0` changes tokens. Quality with it on (T185): GSM8K 0.9765, MMLU-Pro 0.8452 (GB300 0.9810 / 0.8452). | performance | different bits | capy |
| `NINFER_QWEN4_DRAFT_SPLIT` | on (default), `0` | At TP2 the shortlist draft head is split across the ranks; `0` keeps it on rank 0 (T43, T55). | performance | timing (drafts only) | capy |

CPU expert executor (all bit-identical by construction, T35):

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_HOST_MOE_SCHEDULE` | `static` (default), `dynamic` | Work split of a layer call across CPU threads. | performance | same bits | capy |
| `NINFER_HOST_MOE_STEAL` | `1` (default), `0` | Static schedule: a thread that finishes early takes single slices from the others. | performance | same bits | capy |
| `NINFER_HOST_MOE_MEMBER_QUANT` | `1` (default), `0` | Each thread quantizes the activations it needs, instead of a shared quantization phase plus a barrier. | performance | same bits | capy |
| `NINFER_HOST_MOE_PREFETCH` | `0` (default, off), rows | Software prefetch of weight codes N rows ahead. | performance | same bits | capy |
| `NINFER_HOST_MOE_CPUS` | comma list; default: the engine's thread count spread over both CCDs | CPU cores used by the executor. | performance | timing | capy |

## 4. Numerical paths of the TP2 exchanges (legacy A/B)

Defaults are the adopted values. The other values restore earlier, slightly less precise paths, or select
equivalent transports.

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_TP_AR` | on (default), `0` | `0` turns off the T62 prefill MoE exchange (narrow payload, column groups) and goes back to the T56 FP32 path. | legacy A/B | numerics (prefill only) | capy |
| `NINFER_TP_AR_PAYLOAD` | `fp16` (default since T127), `bf16`, `fp32` | Payload of the TP2 prefill MoE exchange. BF16 was the T62 default. Decode MoE exchanges are always FP32. | legacy A/B | numerics | capy |
| `NINFER_TP_AR_SHIFT` | `2` (0..14) | The FP16 payload carries v·2^-k, saturated to the FP16 range, and is rescaled exactly on receipt; this rules out overflow (T128b). | production default | numerics (tiny) | capy |
| `NINFER_TP_AR_GROUP` | 2,048 columns (default), `0` = one group, multiples of 2,048 | Column groups that overlap the prefill exchange with the expert GEMM (T62). | performance | same bits | capy |
| `NINFER_TP_AR_PIPE` | `1` / unset | `1` uses the pipelined exchange even with an FP32 payload and no groups. | performance | timing | capy |
| `NINFER_TP_MIX` | FP32 partials (default since T138), `bf16` | Attention/GDN output exchange: FP32 partials (FP16 transport in prefill, FP32 in decode, one BF16 rounding as at 1 GPU), or the older BF16 partials (T128c, T138). | legacy A/B | numerics | capy |
| `NINFER_TP_HOST_EXCHANGE` | auto (default), `0`, `1` | Transport of the eager (prefill) all-reduces. Auto: direct device-to-device split lanes (the pulled half-chunk overlaps the peer's GEMM) with P2P and no `--elastic`; pinned host staging (T56) without P2P or with `--elastic`. `1` forces host staging; `0` turns the split transport off entirely: one-call eager copies (the driver-staged peer pull with P2P, which blocks per call and is unstable on closely spaced calls, T65). The transports produce identical bits (same combine, event-ordered); the startup log names the choice (`eager all-reduces (prefill overlap): device-to-device | host staging`). | performance | timing (same bits) | capy (generic TP2 code) |
| `NINFER_TP_PREFILL_SLICE` | 1024 (default), `0` off | Tokens per exchange slice of the pipelined prefill (T213): when a prefill chunk holds at least two slices, each eager all-reduce of the layer is issued per slice on the pipeline's lanes and overlaps the compute of the later slices; `0` keeps the 2048-token half-chunk split. Prefill only; bit-identical (same combine). | performance | timing (same bits) | capy |
| `NINFER_QWEN4_REWRITE_SPLITS` | `1` / unset (off) | `1` splits the prefill at every rewrite frontier, not only at capture frontiers: much slower on multi-turn prompts (T66). | legacy A/B | numerics (chunking) and timing | capy |

## 5. Diagnostics: observation (dumps, logs, probes)

None of these changes the arithmetic. They add I/O or synchronization, which can shift expert residency (see the
note above). Variables marked *eager* need `--no-cuda-graph`.

| variable | values | output | scope |
|---|---|---|---|
| `NINFER_QWEN4_EXPERT_TRACE` | file | One JSON line per expert window: hits, CPU experts and time, fills, device timestamps. | capy |
| `NINFER_QWEN4_ROUTING_DUMP` | file | The routing of every observed layer (binary). | capy |
| `NINFER_QWEN4_ROUND_LOG` | file | One line per decode round: extent, accepted drafts, round time, host gap, misses, CPU and GPU-share experts, fill wait (T59). | capy |
| `NINFER_QWEN4_MTP_CALIB` | file | One line per verified MTP round: extent, accepted, and the proposals' own log-probabilities. | capy |
| `NINFER_QWEN4_TOP2` | file | The target's top-2 logits for every decoded or committed verification column. | capy |
| `NINFER_QWEN4_MTP_PROBE` | file (eager; throws under graphs) | The inputs of every MTP-layer expert call. | capy |
| `NINFER_QWEN4_TFDUMP` | directory | tfdump/1 files: logits per scored or decoded position, plus the wide residual after each layer in eager prefill. | capy |
| `NINFER_QWEN4_TFDUMP_LAYERS` | `all` (default), `none`, `i,j,...` | Which layers' residuals to dump. | capy |
| `NINFER_QWEN4_TFDUMP_LOGITS` | on (default), `0` | `0` dumps no logits. | capy |
| `NINFER_QWEN4_TFDUMP_MTP` | `1` (eager) | Every prompt column also runs the MTP layer; draft logits go to `mtp_logits/`. | capy |
| `NINFER_QWEN4_TFDUMP_STRIDE` | `1` (default), n | Logits only at positions divisible by n (KL over long windows). | capy |
| `NINFER_QWEN4_TFDUMP_SEQUENCES` | `1` | One subdirectory per scored sequence. | capy |
| `NINFER_VISION_DUMP` | directory | Each encoded image/video item: `itemN.patches.bf16` (the patches the tower received, BF16 [patches, 1536]), `itemN.merged.bf16` (its merged embeddings, BF16 [merged, H]) and `itemN.json` (grid, counts). Synchronizes the encoding device per item. Compare with an independent tower reference on the same patches. | capy |
| `NINFER_QWEN4_XDUMP` | directory | The two rank partials of each TP2 prefill exchange, before the sum (T128). | capy |
| `NINFER_QWEN4_XDUMP_STRIDE` | `16` (default), n | Column stride of the exchange taps. | capy |
| `NINFER_QWEN4_REPLAY_RECORD` | file | Records `req` / `win` lines (prompt, MTP rounds) for the deterministic fast bench (T59). | capy |
| `NINFER_HOST_MOE_CALL_LOG` | file | Per-call, per-thread timestamps of the CPU expert executor (about 20 ns per stamp). | capy |
| `NINFER_HOST_MOE_DUMP` | file | Every served decode call with host experts (layer, IDs, alpha, x, out) and the partial the device actually read. Written at exit (T98). | capy |
| `NINFER_HOST_MOE_DUMP_MAX` | `40000` (default) | Maximum number of dumped calls. | capy |
| `NINFER_TP_RANK_PROBE` | log file | After every TP2 MTP verification, compares the two ranks' hidden state bit for bit and logs mismatches (T123). No measurable cost. | capy |
| `NINFER_TP_AR_MAXABS` | `1` | Statistics (maximum magnitude) of the exchanged values (T128). | capy |
| `NINFER_HOST_A4_SELFTEST` | `1` | Once at startup: AVX-512 A4 code vs scalar A4 code, 200,000 random and edge blocks, prints the mismatch count (T148). | capy |

## 6. Validation: controlled execution

These variables change execution on purpose so that runs can be compared exactly. The ones marked *gated* require
`NINFER_QWEN4_VALIDATION=1`.

| variable | values | effect | results | scope |
|---|---|---|---|---|
| `NINFER_QWEN4_EXPERT_FREEZE` | `1`, `all` (gated) | `1`: residency is decided by the prompt windows alone, each prompt window waits for its fills, and decode windows insert nothing, so an MTP run and its no-draft control run each expert on the same side. `all`: nothing is ever inserted or evicted (fast bench, T59). | numerics (deterministic residency) | capy |
| `NINFER_QWEN4_FREEZE_FLUSH` | any value | With `EXPERT_FREEZE=1`, the post-prompt refill lands before the first decode window (T98). | residency | capy |
| `NINFER_QWEN4_EXPERT_SYNC` | `1` | Every window waits for its fills, so residency depends on routing only, not on copy timing (T59). | residency, slower | capy |
| `NINFER_QWEN4_EXPERT_SNAPSHOT` | file | Each request starts from a saved residency block (request r uses block r mod blocks) (T59). | residency | capy |
| `NINFER_QWEN4_EXPERT_SNAPSHOT_SAVE` | file | Saves the residency at each request's first decode window. | none | capy |
| `NINFER_QWEN4_FORCE_TOKENS` | file (gated) | The committed token of each decode step is the next token of the matching line. Single-token commits only: use with `NINFER_QWEN4_MTP_NO_DRAFTS=1`. | tokens | capy |
| `NINFER_QWEN4_TFDUMP_FORCE` | IDs file (gated) | Teacher-forced decode of the given sequence, prompt included. | tokens | capy |
| `NINFER_QWEN4_REPLAY` | file (gated) | Deterministic replay of a recorded run: forced tokens and recorded MTP drafts and extents (T59 fast bench). | tokens | capy |
| `NINFER_QWEN4_REPLAY_MODEL_DRAFTS` | `1` | With `REPLAY`: where no window is recorded, keep the model's own drafts (forced targets with realistic rejections, T121c). | tokens | capy |
| `NINFER_QWEN4_MTP_NO_DRAFTS` | `1` | Every MTP round verifies the anchor alone: the no-draft control of an MTP run. | same tokens (greedy) | capy |
| `NINFER_QWEN4_MTP_ORACLE` | file | Drafts are taken from whole reference sequences, so rounds are fully accepted. | same tokens (greedy) | capy |
| `NINFER_QWEN4_MTP_EXTENT_CAP` | k | Verify at most k proposals per round in the K-wide window (measures the cost of a column). | same tokens (greedy) | capy |
| `NINFER_TP_AR_DITHER` | seed | Deliberate perturbation of the FP32 combine of TP2 exchanges, the same on both ranks (T128 metric-noise study). | numerics | capy |
| `NINFER_TP_AR_DITHER_BIT` | 0..15 | Which bit the dither flips. | numerics | capy |
| `NINFER_HOST_MOE_W4A4` | `1` | The CPU emulates the GPU's W4A4 recipe exactly, with FP64 sums. Slow; reference for T98. | numerics | capy |
| `NINFER_QWEN4_HC_GENERAL` | `1` | Q8/NVFP4 hyper-connection mixers take the general (non-fused) route, to separate route from weight format (T103). | may change rounding | capy |

## 7. Benchmark kernel selectors

| variable | values (default) | effect | results | scope |
|---|---|---|---|---|
| `NINFER_T49_NVFP4` | candidate index (unset = tuned selection) | Kernel candidate for the Qwen4Exp NVFP4 dense decode shapes (T49). | may change rounding | capy |
| `NINFER_T49_Q8` | candidate index (unset = tuned selection) | Kernel candidate for the Qwen4Exp Q8 dense shapes (T49). | may change rounding | capy |
| `NINFER_Q6_DRAFT_ROUTE` | route index (unset = tuned) | GEMV route of the shortlist draft head at T = 1 (T43). | drafts only | capy |

## 8. Inherited from ninfer-tp2

TP2 transport. These apply to capy-flash as well, and the defaults are right for it:

| variable | values | effect | class | results |
|---|---|---|---|---|
| `NINFER_TP_MAILBOX_LEGACY` | `1` | Original mailbox exchange kernel instead of the pipelined one. | legacy A/B | same bits |
| `NINFER_TP_MAILBOX_SLOT` | `request` | No wide mailbox slots; batched exchanges use staged copies. | legacy A/B | timing |
| `NINFER_TP_MAILBOX_PROBE` | `off`, `fail` | Skip the startup mailbox probe, or force the fallback to copies. | diagnostic | timing |
| `NINFER_TP_MAILBOX_FAULT` | `draft`, `any` | Simulates a mailbox hang to test the degradation path. | diagnostic (test) | timing |
| `NINFER_TP_MAILBOX_DRAFT` | `copies` | Starts with the MTP draft exchanges on copies instead of the mailbox. | performance A/B | timing |

For other models only (DFlash drafters of the Qwen3.x 27B models; capy-flash drafts with MTP, so these have no effect
on it):

| variable | values | effect |
|---|---|---|
| `NINFER_TP_DRAFT_HEAD` | `primary` | Proposal head whole on rank 0 instead of split. |
| `NINFER_TP_DRAFTER` | `primary` | Whole DFlash drafter on rank 0. |
| `NINFER_TP_SELECTOR` | `primary` | DFlash2 candidate selector on rank 0. |

Exists, but has no execution route:

| variable | values | effect | scope |
|---|---|---|---|
| `NINFER_HC_SPLIT_ROUTE` | `composed` | Composed route of the stream-split hyper-connection ops inside captures. Those ops have no execution route (the Program refuses `tp_hc_split_streams`, T30), so this only matters to their tests. | capy (tests) |

## 9. Read by tests and the validation driver only

These are not read by the server. The list is not exhaustive; see `tests/README.md`.

| variable | used by |
|---|---|
| `NINFER_TEST_ARTIFACT`, `NINFER_TEST_QWEN4_ARTIFACT`, `NINFER_TEST_QWEN4_VISION_ARTIFACT` | real-artifact tests (skipped when unset) |
| `NINFER_TEST_QWEN4_REUSE_ONLY`, `NINFER_PREFIX_REAL_SCENARIO`, `NINFER_TEST_PORT`, `NINFER_TEST_SCRATCH_DIR` | integration test selection and setup |
| `NINFER_OP_REPORT_STATS` | op tests print statistics |
| `NINFER_DRIVER_TEMPS` | validation driver: one sampling temperature per prompt line (T121) |
| `NINFER_DRIVER_STOPS` | validation driver: stop at the model's default stop tokens (T121) |
