# capy-flash development log

Comments in the engine source cite numbered experiments as `Tnn` (for example "T54", "T128b"). Each tag is one
measurement or change from the development of capy-flash, made in a separate lab notebook that is not published.
This log explains every tag that appears in the source: what was measured or changed, the key numbers with their
scope, and the decision taken. Tags that never reached the source are not listed, so the numbering has gaps.

All numbers come from one machine: AMD Ryzen 9 9900X (12 cores, AVX-512 VNNI), 96 GB DDR5-5600, 2x NVIDIA RTX 5070 Ti
16 GB on PCIe 5.0 x8 + x8 without peer-to-peer access, CUDA 13.1. Unless an entry says otherwise, GPU clocks were
the "eco" profile (graphics clock locked at 2,100 MHz). "GB300" is the quality reference: NVIDIA's NVFP4 checkpoint
of Qwen3.8-Flash-Next served by vLLM on an NVIDIA GB300. "Strata" is an independent open-source (MIT) engine for the
same model, measured on the same machine as the speed reference. "v0.1", "v0.1.1" and "v0.2" below are internal
capy-flash builds (2026-10-14/15); the first public release with this work is v0.5.0. Numbers are measured unless they are labelled
*estimate* or *model* (an offline simulation calibrated on measurements).

## Terms used below

| term | meaning |
|---|---|
| Qwen4Exp | the engine's internal name for the Qwen3.8-Flash-Next architecture |
| 1 GPU / TP2 | one RTX 5070 Ti, or tensor parallelism over both (`--tp 2`) |
| plain / MTP | decode without speculation / with the model's MTP layer drafting K = 3 tokens per window |
| window | one decode round: verification of the anchor plus the draft columns |
| d0 / 16k | context depth of the decode measurement |
| miss, resident | a routed expert that is not / is in the GPU expert cache; misses are computed on the CPU |
| W4A4 / W4A8 | GPU expert kernel (4-bit weights, 4-bit activations) / CPU expert kernel (4-bit weights, int8 activations) |
| A4 | CPU expert path using the GPU's W4A4 activation codec (from T98; opt-in via `NINFER_HOST_MOE_A4=1` before v0.5.0, default since v0.5.0, commit 54256b95) |
| T27 workload | shared decode bench: one pg19 prompt and four chat prompts at depth 0 and 16k, greedy, 128 tokens, n = 5 |
| T71 replay | agentic replay: 3 synthetic sessions, 53 turns, contexts up to 81k tokens, 256 tokens generated per turn |
| NLL 4K | mean negative log-likelihood on 4,096-token pg19 windows (lower is better) |
| TF KL | teacher-forcing KL divergence against an FP64 reference implementation |
| 27B gate | the parent engine's regression gate on Qwen3.8-27B (ctest, golden outputs, perplexity) |
| artifact G | an early recipe: GDN projections NVFP4, rest of the dense part Q8 (T26c) |
| g4, g5, n4 | three fixed agentic contexts used as teacher-forcing probes |
| E-tr03, E2-tr04, E-fc02 | agentic coding tasks from a private bench, run with OpenCode |

## Entries

**T03**: multi-token CPU dot product for NVFP4 x int8 (AVX-512 VNNI). Each 64-weight group is decoded once, then
one `vpdpbusd` runs per token. Error vs scalar is at most 1.4e-7, and the result is bit-identical for every token
count. On a synthetic 2 GiB bank with 6 threads: 68.4 / 68.3 / 66.0 GB/s at T = 1 / 2 / 4 (FreeToken's per-token dot:
32.6 GB/s at T = 4). An 8-expert layer call is 9 % faster at T = 1 and 49 % faster at T = 4. Adopted as the CPU
expert kernel.

**T10**: hyper-connection (HC) and PLE ops, tested against FP64 oracles. HC (Q8, fused cooperative kernel) takes
19.7 / 23.3 / 27.4 µs at T = 1 / 4 / 8, about 1.9 ms per forward per GPU over 97 mixers. A cold PLE gather from host
takes 131-471 µs with `MADV_WILLNEED` (1.6-10.8 ms without). Adopted.

**T11**: QSA attention and GDN ops. Append, select and attention take 22 / 49 / 63 / 96 µs per layer at T = 1 and
depth 0 / 16k / 64k / 128k (GPU0, free clocks, BF16 KV). Block selection is 3.1-11.7x faster than ninfer-ext's at
16k-128k, with 0 blocks different from the oracle at 90k. GDN recurrence takes 7-20 µs per layer at T = 2-8. Adopted.

**T12**: model config, converter recipes and loader for Qwen4Exp. The ninfer-ext artifact format is our format plus
a per-matrix `divisors` field. Experts are pinned in host RAM (64.6 GiB) and the PLE table is memory-mapped. The real
inventory is 132.64 GB. The 27B gate passes. Adopted.

**T14**: expert cache manager: per-layer slot tables, experts partitioned across GPUs by ID parity, LRU with
admission and an insert cap, fill queue. It replays a recorded routing trace exactly (21/21). Fills with
`cudaMemcpyBatchAsync` reach 28.86 GB/s per link (57.08 on both) at 1.1 µs host time per expert. A fill lands one
window later, which lowers the simulated hit rate at 181 slots from 0.894 to 0.871. Adopted.

**T15**: resident-expert W4A4 kernel on cache slots (`moe_slots_nvfp4`, derived from qf_moe). At T = 4 it costs
11.5 µs + 3.14 µs per expert; 40 experts take 135.9 µs = 814 GB/s (free clocks). W4A4 vs FP32 relative error is
0.156-0.170 (limit 0.18). The checkpoint's activation input scale is one per layer (exact); weight divisors are per
expert. Adopted.

**T16**: CPU expert path. A weight-major executor runs alongside the GPU, with a handshake inside the CUDA graph
(`cuStreamWriteValue32/WaitValue32`, no GPU spin). Tests vs FP64 pass, and 245,760 GPU outputs are bit-exact.
Smoke only (other jobs running): 51.7 µs per expert at 54 GB/s, handshake 9-10.6 µs per layer. Finding: CPU W4A8 is
0.011-0.013 relative L2 from exact and GPU W4A4 is 0.13-0.17, so one expert's output differs by about 15 % between
CPU and GPU. This difference became T98. Adopted.

**T18**: misses split between CPU and GPU: the GPU copies part of the misses into 16 scratch slots over PCIe and
computes them. On 1 GPU, MTP, T27 workload, n = 3, the window drops from 58.7 to 52.6 ms with `--miss-split auto`
(+9.1 %) and to 51.5 ms with a fixed 0.3 share (+13.6 %). Adopted at 1 GPU. At TP2 it is off (T42).

**T19**: expert prefetch predictor `post_uni`: the router of layer l+1 applied to layer l's post-attention state.
On an HF BF16 reference over 3,351 tokens, top-10 overlap is 0.69 and precision on non-resident experts is 0.38
(break-even 0.24). The model estimates +11-14 % at 1 GPU and +3.7-4.3 % at TP2. Decision: build the prefetch (T25).

**T20**: milestone M1: Flash-Next runs on 1 GPU (chunked prefill, perplexity, CUDA-graph decode). TF KL on real
prompts: prefill 2.87e-2 (6.4x the BF16 floor), decode 2.18e-2 (4.9x), PASS. NLL 4K is 1.8062 vs 1.7969 for the HF
BF16 reference (+0.9 % PPL, attributed to 4-bit expert activations). The validation driver dates from here.

**T22**: fast expert backend at 1 GPU: the GPU computes resident experts while the CPU computes misses, cache fills
run asynchronously, and the cache auto-sizes to 77 slots per layer. Quality PASS (TF KL 3.5x floor, NLL 4K 1.805).
Plain decode: 53.3 ± 2.7 / 50.2 ± 1.0 t/s at d0 / 16k (n = 5, Q8 dense). Per token (18.2 ms), dense + HC + head take
10.3 ms and the MoE stage 7.9 ms (CPU 64 µs per miss, hit rate 0.81). Zero-copy GPU reads of misses were slower and
stay an option. Adopted.

**T24**: MTP at 1 GPU. With K = 3 the acceptance is 0.533 and a window yields 2.59 tokens (chat 2.99, pg19 2.16;
K = 1 gives 1.77, K = 5 gives 2.98). Verification is exact: 20/20 identical to the no-draft control. MTP experts have
no calibrated activation scale, so the engine uses 32x the widest Text-layer range (W4A4 error is flat at 0.1185 from
1x to 64x). Adopted with K = 3.

**T25**: prefetch with the T19 predictor, and cache warm start after the prompt (1 GPU, MTP, T27 workload, n = 5).
Prefetch: −2.7 %, not adopted, because PCIe copies take DRAM bandwidth from the CPU (+13 % µs per CPU expert).
Warm start: +4.1 %, adopted under the preregistered rule (weak evidence). Warm start is on by default. Prefetch is
off, and refused at TP2.

**T26b**: mixed dense recipes (Q8 plus NVFP4 per group: attention, GDN, shared expert, head). This followed T26
(all-NVFP4 dense: 1.37x faster, but NLL 4K +0.0139 > the 0.010 margin). On one 4K window no mix passed (attention +
GDN + head +0.0175), and per-group differences were below the resolution of a single window. The recipe stays in the
converter. Next step: T26c.

**T26c**: recipe "G": GDN projections NVFP4, the rest Q8. ΔNLL is +0.005 (95 % CI +0.000…+0.010) over 26 4K windows,
which passes the preregistered rule (point ≤ 0.010, upper bound ≤ 0.015). GSM8K-200: 0.990 vs GB300 0.985 (p 1.00).
Every variant with NVFP4 attention fails (attention alone +0.017). Speed vs Q8 on the T27 workload: +5 % / +3 %
(d0 / 16k). Adopted at the time. The product later moved to indexer BF16 + GDN Q8 (T103, T104).

**T27**: clean baseline vs Strata 0.1.40.1: same token IDs, 1 GPU, greedy, 128 tokens, n = 5. Ours (plain):
44.7 / 45.0 t/s (d0 / 16k). Strata IQ3_S without speculation: 64.1 / 61.9; with speculation: 79.3 / 74.8. Most of
the gap is the single pass (x1.265): bytes per expert account for x1.17 (*estimate*) and the engine for x1.08. MTP
accounts for only x1.06. This entry defines the T27 workload.

**T30**: TP2 shard rules for Qwen4Exp, with FP32 all-reduce of the MoE partials (+1.7 µs at T = 4). Parity holds on
2 GPUs. 102 exchanges per window cost about 0.54 ms. Splitting HC by streams was slower (−0.8 ms), so it is kept as
ops only, with no execution route.

**T31**: Flash-Next on 2 GPUs: TP2 prefill, perplexity and graph decode over the peer mailbox. NLL 4K is 1.8079 vs
1.8062 at 1 GPU (paired +0.0017, not significant). Both ranks' residuals are bit-identical in all 48 layers. Top-1
disagreement with HF BF16 is 5.7 % (1 GPU) vs 6.0 % (TP2), p 0.50: the cause is W4A4 experts, not TP. The ID-parity
expert partition is unbalanced (about 20 % of the round spent waiting). Adopted.

**T33**: integration of MTP and the fast backend: phase A at 1 GPU, phase B at TP2. TP2 on the T27 workload (n = 3):
91.3 / 91.2 t/s = 0.69x / 0.82x tuned 2-GPU Strata. Rank 0 waits for the CPU about 7 ms per window while rank 1 idles
at the all-reduce. Rank 0 publishes from a residency mirror of both GPUs. A summation-order bug in the CPU executor
was found and fixed (MTP vs control 20/20 afterwards). Adopted.

**T34**: streaming prefill: non-resident experts are copied over PCIe into a ring of slots borrowed from the decode
cache, and the next layer is copied while this one computes. Chunks wider than 2,047 tokens run no experts on the CPU.
1 GPU: 4k 1,774 ± 146 tok/s (0.80x Strata), 16k 3,222 ± 14 (1.21x); before: 413 / 371. NLL 4K 1.8070, PASS. A CPU
tail helps at chunk 1024 (1.7x) and hurts at 4096 (1.6x slower), so it is used for small chunks only. Adopted.

**T35**: CPU executor cost. Per miss inside the engine: 63.1 → 49.5 µs (microbenchmark 44.1 µs = 63-65 GB/s, at the
bandwidth roof). 1 GPU, T27 workload, n = 5: plain +10.2 %, MTP 54.3 → 62.7 t/s (+15.6 %). Fixes: cache-fill DMA is
now issued after the CPU finishes the layer instead of overlapping its reads (`NINFER_QWEN4_FILLS_AFTER_HOST`), and
work uses static slices with stealing and one barrier. All bit-identical. Adopted.

**T36**: adversarial read-only audit of the integration branch. No critical findings. Four major ones: mailbox
`failed()` never checked, a window not abandoned on a prefill error, missing MTP/offload tests, and dead HC-split
code. Minor ones include m9 (rank-1 PLE fold) and m10 (trace capacity). Fixed in T33 and T37.

**T37**: fixes for T36: M1-M3 and minors m1, m2, m3, m6, m7, m10, each with a test that failed before. Since T37,
validation knobs that change tokens are refused unless `NINFER_QWEN4_VALIDATION=1`. Adopted.

**T40**: not an experiment tag. The one match in the engine is the test case name "T40 prefill, dot groups"
(`tests/ops/test_host_moe.cpp`), where "T40" means a CPU expert call with T = 40 tokens (a prefill-like shape added with
T22). The lab task numbered T40 (product packaging) has no measurements.

**T41**: window breakdown (nsys) and an adaptive MTP window. A 1-GPU K = 3 window takes 29.6 ms: about 23.1 ms of
GPU work (dense 5.6, resident experts 6.1 + 1.2, draft heads 3.0), 6.95 ms waiting for misses and 1.64 ms of host
time between rounds. Strata: 27.3 ms. TP2: 23.5 ms vs Strata pipelined 20.3. The adaptive window (`--mtp-min-p 0.3`)
gave +2.8 % at 1 GPU (noise) and nothing at TP2, so it was not adopted. The per-column expert counters in the trace
come from here.

**T42**: merge of T33, T35, T18, T25, T26c, T34 and T37, plus a prompt workspace borrowed from the cache and TP2
streamed prefill per link. Decode: 1 GPU 74.7 / 67.3 t/s (d0 / 16k) = 0.78x / 0.76x Strata; TP2 118.2 / 102.1 =
0.89x / 0.92x. Prefill: 1 GPU 1,829 / 3,266 tok/s (4k / 16k); TP2 2,147 / 3,176. Miss split at TP2 (rank 1 takes the
GPU share) cost 2-5 %, so it is off at TP2 and auto at 1 GPU.

**T43**: draft head on a vocabulary shortlist. The proposal heads score 108,861 tokens (IDs 0-98,303, the 32,768 most
frequent and 33 specials, from a sidecar file); the verification head stays full, so greedy output is identical
(20/20). Draft cost per window: 5.94 → 3.58 ms (1 GPU). Decode at d0: 1 GPU 72.95 → 78.51 t/s (+7.6 %), TP2
119.3 → 123.3 (+3.4 %); at 16k +1.3 % (noise) / +2.6 %. Adopted. T55 made the sidecar the default.

**T45**: HC. Correction to T41: HC adds 1.60 ms per window at 1 GPU and 1.53 ms at TP2, not 3.0 ms. NVFP4 HC weights
were rejected (ΔNLL +0.022). PDL on the mixers was within noise (TP2 122.2 → 122.6). Only the TP2 part was kept: the
mixer's Down launch streams its weights under the exchange.

**T46**: resident-expert kernels: persistent grids, a list of residents only, double buffering, bit-identical output.
At T = 4 they are 10-13 % faster (82.9 µs for 17 experts = 82 % of the roof). End to end there was no gain
(1 GPU 72.7 → 71.6, TP2 119.5 → 119.7): each layer the GPU waits about 139 µs (TP2: 119 µs) for the CPU's share.
Kernel kept. Conclusion: the MoE stage is bound by the CPU.

**T49**: roofline audit of GPU work per window. Byte roof vs measured: 1 GPU 9.0 vs 21.5 ms, TP2 6.0 vs 16.5 ms.
Kept, each checked by bisection: a faster Q8 route (QSA projections 520 → 668 GB/s), PDL on router, publish and GDN
record, in-place GDN gating, and an MTP commit that does not wait for the fold. Result: 1-GPU MTP 72.3 → 75.6 t/s
(+4.5 %, n = 3); TP2 +1.7 % (noise). Rejected: a copy-engine host partial (−16 % plain). `NINFER_T49_*` selects
kernel candidates for benchmarks.

**T52**: MTP window efficiency. A K = 3 window costs 2.10 single steps at 1 GPU (Strata 1.86) and 1.96 at TP2 (Strata
1.51). At TP2 the whole gap is the draft chain (about 3.1 ms per window). Change: during MTP alignment the MoE routes
only each lane's accepted column, saving 0.15 ms per window (+0.4 / +0.8 %). Adopted.

**T53**: non-expert GPU work, one item at a time. At 1 GPU nothing moved beyond noise (32 ms windows, 18-19 ms of
them waiting on CPU experts). Kept: the next round's PLE rows are prefetched right after the sync (host gap −0.32 ms
at 1 GPU / −0.35 at TP2, neutral end to end; `NINFER_T53_PLE_PREFETCH=0` turns it off), and an early release of the
HC mixer (TP2 plain +1.4 %). The rest was rejected.

**T54**: fewer misses per token: admission threshold 2 with a 256-token half-life became the default (`--admit 0` =
LRU). Misses per token, 1-GPU MTP: 204.5 → 182.6. Throughput (T27 workload, d0, n = 5 paired): 1 GPU plain
59.16 → 61.56, MTP 79.22 → 82.52; TP2 plain 91.92 → 95.27, MTP 123.74 → 128.87 t/s, +3.6-4.2 % in every cell. NLL 4K
unchanged. Adopted.

**T55**: merge of T43 (shortlist sidecar used by default), T52, T46, T45-TP2, T49 and T54, with a same-session
comparison. MTP at 1 GPU: 80.1 / 68.7 t/s (d0 / 16k). MTP at TP2: 134.4 / 105.4 = 0.94x / 0.88x Strata IQ3_S, and
1.21x / 1.03x Strata running the same NVFP4 weights. TP2 prefill: 2,137 / 3,180 tok/s. MTP = no-draft 20/20; NLL 4K
1.79984 / 1.80370 (1 GPU / TP2). Adopted.

**T56**: prefill, second round. Without P2P, the TP2 MoE all-reduce queued on the link behind the next layer's
expert copies, leaving both GPUs idle 1.76 s of 4.93 s at 16k. Kept: next-layer copies issued after the all-reduce
(TP2 +25.6 % at 4k, +21.9 % at 16k); a non-blocking all-reduce through pinned host buffers (+~2 %); gate/up over the
whole call (+3.3 %, 1 GPU 16k); vectorized HC kernels (+4.3 %); half-size streaming ring at TP2 (+1.5 % at 4k).
Result: TP2 4k 2,894 tok/s (1.36x Strata), 16k 4,200 (1.30x). NLL bit-identical. Each change has a same-bits A/B
switch.

**T59**: deterministic fast bench. It replays a recorded greedy MTP run with forced drafts and acceptances, restores
the expert cache from a per-prompt snapshot and subtracts copy waits. A vs A: −0.05 % (CI ±0.12) at 1-GPU MTP and
−0.01 % (±0.18) at TP2, about 3.5 minutes per A/B cell (vs 30-45 minutes at ±2-3 % before). It does not judge
acceptance, quality, RAM contention or 16k. Adopted as a measurement tool (`NINFER_QWEN4_REPLAY*`,
`EXPERT_FREEZE=all`, snapshots, round log).

**T61**: eviction. *Model*: perfect (Belady) eviction would give +12.6 % at 1-GPU MTP and +16.7 % at TP2 MTP;
ARC/2Q/TinyLFU/LFU/LIRS give −9…+2 %. Adopted `fr` (frequency x recency, τ = 4 windows): +5.2 % at 1-GPU MTP and
+5.8 % at TP2 MTP vs the previous rule (n = 5 paired, d0, repeated prompts). A per-layer slot profile looked like
+13.4 % at TP2 MTP, but that was an effect of repeated prompts: +0.2 % (CI −9…+10) on non-repeated requests. Not
adopted, and reverted in T68.

**T62**: TP2 prefill all-reduce, overlapped in groups of 2,048 columns (bit-identical) with a BF16 payload:
3,048 / 4,603 tok/s at 4k / 16k (+5.4 / +9.3 % over T56). FP16 was rejected at the time (NLL 4K +0.0049 > 0.002).
BF16 became the default, with fallback `NINFER_TP_AR_PAYLOAD=fp32`. Superseded by FP16 with shift (T127, T128b).

**T64**: INT8 KV cache, an elastic split of VRAM between KV and expert slots, and 262k context at 1 GPU and TP2.
4K sentinel: INT8 TP2 +0.0017 (pass); 1 GPU +0.0088 (above the 0.003 threshold, not significant). Long-context NLL is
within ±0.0007 of BF16 up to 262k. Needle retrieval: 18/20, 20/20, 20/20 at 64k / 128k / 262k. TP2 MTP decode:
135.5 / 114.9 / 98.5 / 112.1 / 104.6 t/s at d0 / 16k / 64k / 128k / 262k (single runs above 64k, ±5 %). Adopted
(product flags `--kv-dtype int8 --elastic`, 262k). A TP2 prefill regression with elastic memory was fixed in T68.

**T65**: final merge (fast bench, prefill round 2, T53, `fr` eviction, BF16 prefill all-reduce). NLL 4K: 1 GPU
1.79984; TP2 1.80529 with BF16 vs 1.80370 with FP32 (+0.0016), so BF16 became the default. 27B gate PASS (ctest
177/177). The `allreduce_f32` test was fixed to compare against the exact value. Known issue: the old peer-copy path
(`NINFER_TP_HOST_EXCHANGE=0`) is unstable on closely spaced calls.

**T66**: prompt prefill in the server. The 320 tok/s on 1.1k-token prompts was not the engine. The frontend put a
rewrite frontier at every assistant turn, prefill split there, and each piece paid a full expert pass. Qwen4Exp now
splits only at capture frontiers: a 10-turn conversation drops from 3.13 to 1.37 s (one message: 1.28-1.37 s).
`NINFER_QWEN4_REWRITE_SPLITS=1` restores the old splits. Cost: a resumed prefill can differ from a from-scratch one,
as two chunk sizes do.

**T66b**: prefix reuse (context cache) with MTP at TP2, refused before. Also fixed: rank 1 never wrote its prompt
hidden state for capture checkpoints, so resumes read stale data. T71 replay: TTFT median 13.5 → 2.54 s (p90 3.85),
wall 790 → 268 s (Strata 0.1.40.4: 2.00 s / 4.61 s / 248 s). Adopted.

**T66c**: server options for Qwen4Exp: `--elastic`, `--kv-grow`, `--kv-budget-gib`, `--miss-split auto|off`. They have
no effect on other models. Adopted.

**T68**: merge of the KV work, plus `memcpy_device_async` for peer copies outside graphs. Without P2P, a UVA copy
cannot read another board's VMM memory (`invalid argument`), but `cudaMemcpyPeerAsync` can; all 43 device-to-device
copies were converted. Product config (INT8 + elastic, 262k): NLL 4K 1.81200 at 1 GPU (+0.0122 vs BF16) and 1.80552
at TP2 (+0.0002). Prefill at TP2: 935 / 2,726 / 4,308 tok/s at 1k / 4k / 16k (−15 / −11 / −7 % vs fixed BF16), which
is 1.09x / 1.28x / 1.34x Strata 0.1.40.4. 27B gate PASS. Adopted.

**T69**: decode after a prompt. The changes, each with an environment switch: refill the hottest experts first across
layers; reclaim lent layers before a short tail chunk; lend only base slots; size the workspace to the chunk; weight
prompt counts by recency. TP2 MTP, exploratory n = 3: time lost in the first 10 windows 247 → ~96 ms (2.6-7k-token
prompts); first 128 tokens 104 → 123 t/s (16k); T71 replay server-side decode 105.1 → 108.2 t/s. Adopted. A
post-prompt insert boost was built but not measured, and is off (`NINFER_QWEN4_BOOST_WINDOWS`).

**T98**: cause of the agentic "decision fork" drift: experts ran as a mix of CPU W4A8 (misses) and GPU W4A4
(residents), and residency changed during the answer. The CPU kernel is correct (0.009 vs FP64; the GPU's 0.147 is
the W4A4 error). Fork probability on probe g4: 0.06-0.35 in 6/6 dynamic-mix runs vs 0.73-0.98 in 12/12 consistent
configurations. Fix: `NINFER_HOST_MOE_A4=1`, where the CPU uses the GPU's W4A4 activation codec. It gives 0.977-0.980
(HF 0.969) and answer NLL 0.353 vs 0.382. Speed cost: −4 % (*estimate*), later removed by T148. Adopted; the default
since v0.5.0.

**T103**: BF16 for the indexer, HC and GDN (with A4). Indexer BF16 is free (TP2 decode 79.0 vs 79.0 t/s in this
setup) and moves the g4 fork from 0.438 to 0.835: adopted. HC BF16 has no fused route (67.2 t/s): rejected. Indexer
BF16 + HC BF16 + GDN Q8: −20 %. Teacher-forcing distance from HF over g4/g5/n4: GDN Q8 −14 %, indexer or HC about −2 %.
`NINFER_QWEN4_HC_GENERAL=1` is the diagnostic that forces the general HC route.

**T104**: recipe indexer BF16 + GDN Q8 + A4 (HC Q8). Agentic task E2-tr04 x4: 24 / 26 / 28 / 28 steps (median 27)
vs GB300 25.5 (p 0.29) and the previous product 35.5 (p 0.0066 over 11 runs). No self-verification spirals, 265 s
per run vs 378 (−30 %), decode 105.6 t/s. Adopted as the product artifact recipe.

**T107**: malformed tool calls. Census: about 6 in 175 runs on this engine, 1 in 112 on the parent engine (27B),
0 in 498 on vLLM GB300 and 0 in 28 on Strata, all with the same sampling (T 1.0, top-k 20, top-p 0.95). In the exact
contexts the broken token has logprob −15…−28 under teacher forcing, so it is an engine bug, at the opening of a tool
call after `</think>`. The validation driver gained top-k / top-p. Without drafts E-tr03 succeeded 4/6 vs 2/16 with
MTP (p ≈ 0.03, small n). Not root-caused; later runs showed an environment-dependent rate (see T123).

**T109**: no separate record in the lab notebook; described from the source comments. During the tool-call
investigation (T107-T110), the validation driver's prefix reuse at TP2 was changed to behave like ninfer-serve (no
host tier on rank 1), and `--repeat` now samples each repeat with its own seed (seed + run index), so repeats are
independent.

**T121**: replay of a broken agentic session: steps 1…K−1 forced to the recorded answers, then repeated free
sampling at the broken step K. The driver gained `NINFER_DRIVER_TEMPS` (one temperature per prompt line). The replay
in the driver did not reuse the cache as the server does, so it was inconclusive.

**T121c**: the same replay, keeping the model's own MTP drafts where no window is recorded
(`NINFER_QWEN4_REPLAY_MODEL_DRAFTS=1`): forced targets with realistic rejections. The lab notebook records the tool
but no outcome.

**T123**: TP2 rank-consistency probe (`NINFER_TP_RANK_PROBE=<log>`). After every TP2 MTP verification it compares the
two ranks' hidden state bit for bit. 0 divergences: in 5,000 idle rounds (no measurable cost), in 150,000 rounds of a
full GSM8K gate, and over 1,319 GSM8K requests later (T135). It is kept on during quality runs as a diagnostic. In
the following days the malformed-tool-call rate fell to 0/20 runs, and the v0.1 quality run had 0 malformed calls
in 1,124.

**T127**: the default TP2 prefill MoE exchange changed from BF16 to FP16: same bytes and speed, and 8x smaller
rounding of the partial sums. Decode MoE exchanges were already FP32. The full GSM8K gate stopped for futility at 990
questions (0.9747, b = 6, c = 0, p 0.031), the same as BF16 at that point (966/990). It also measured run-to-run noise:
7 of 990 questions changed outcome between two identical configurations. So the gate cannot resolve effects this
small. Adopted.

**T128**: map of the values exchanged at TP2. (1) The engine's bits depend on expert residency, which depends on
timing; with residency frozen, taps are bit-faithful. (2) The noise of the teacher-forcing distance metric is SD
1.41 / 0.34 / 3.10 % on g4 / g5 / n4, so earlier ±1.5-2 % deltas were not established. (3) Error added vs 1 GPU:
BF16 exchange 1.44-1.72x (MoE, GDN, QSA mixers), FP16 1.01-1.03x. Maximum measured magnitudes: mixer 6.1, MoE 13.3.
In absolute terms that is +0.17-0.23 % per block output, against 12-16 % for W4A4 per expert. Tools:
`NINFER_TP_AR_DITHER`, `NINFER_QWEN4_XDUMP`, `NINFER_TP_AR_MAXABS`.

**T128b**: the FP16 exchange carries v x 2^-2 with saturation (`NINFER_TP_AR_SHIFT`, default 2). Error is the same as
plain FP16, and overflow of MoE partials is ruled out. Adopted.

**T128c**: mixer (attention/GDN output) exchange with FP32 partials, transported as FP16 in prefill and FP32 in
decode, with one BF16 rounding as at 1 GPU. Verified bit-exact at layer 0. No prompt-speed cost (25.1-25.4 s vs
25.2-26.1 s). Opt-in at first, default since T138; `NINFER_TP_MIX=bf16` restores BF16 partials.

**T132**: crash `cudaErrorGraphExecUpdateFailure` (the 2-D memcpy node of GDN split_rows at the start of MTP decode),
in 8 of 25 runs after a scoring request. Fix: when a graph update is rejected, the graph is re-instantiated. Result:
0/8 crashes and 1 rejection recovered. Adopted. T139 removed the cause.

**T138**: FP32 mixer partials (T128c) made the default. Decode over 10 pairs: 105.8 vs 103.7 t/s (p 0.62, CI ±6 %).
Prefill of a 20k prompt: 25.3 s, unchanged. Agentic E-fc02 x4: step-count SD 7.9, vs GB300 8.1 and 13.2 with BF16
partials (p 0.42). Adopted.

**T139**: an `ops::copy_rows` kernel replaces the 2-D memcpy node in split_rows. Graph-update rejections: 0/8 (before:
6/15). Output bit-identical (8,645/8,645 tokens). Adopted.

**T148**: A4 CPU path in AVX-512, bit-identical to the scalar A4 code (self-test 200k blocks, 0 differences; probe
n4 8,645/8,645 tokens; `NINFER_HOST_A4_SELFTEST=1`). T71 replay, TP2 product config: TTFT 2.39 s, decode 103.8 t/s,
wall 270 s. Same week: scalar A4 93.6 t/s, no A4 103.2 t/s. A4 now has no measurable cost. Adopted.

**T152**: `--turn-anchors 0`: no capture anchor before the last message, so 2 prefill passes per agentic turn instead
of 3. T71 replay, TP2 product config: TTFT median 2.67 → 1.82 s (p90 3.33), decode 101.3 t/s, wall 278 → 243 s.
Strata 0.1.44.1 IQ3_S: 1.86 s / 236 s; Strata with the same NVFP4 weights (full residency): 2.42 s / 314 s.
Server-side decode (bench/capy_flash/summarize.py): ours 100.8 t/s, Strata NVFP4 78.3, Strata IQ3_S 124.1 (client-side
rates overstate Strata, which streams in bursts). Default since v0.2. Cost: a cancelled or
edited turn re-reads the whole conversation.

**T155**: crash "PLE table row crosses a mapped segment". The PLE table (FP8, 51.84 GB) crosses a part-file boundary
inside one row (row 140,944,145: 96 bytes in one file, 64 in the next), and that row is read after tokens
93405 → 65047. Fix: a row that straddles segments is reassembled per worker. The new test fails before the fix and
passes after. Shipped in v0.1.1.

**T156**: low host RAM. With 35 GiB taken by other processes, v0.1.1 raised memory pressure to 36 % while loading
(where the OOM killer acts). Startup guard: the server requires MemAvailable ≥ 64.6 GiB + max(2 GiB, 5 % of MemTotal)
and otherwise refuses in 44 ms, printing the numbers. `NINFER_RAM_GUARD=0` skips the check. Adopted in v0.2.

**T158**: unit tests fixed; ctest 186/186 (48 tests that need a real artifact are skipped). The `expert_cache`
failures under parallel GPU load were a test artifact: the test's setup memsets and counters ran on the legacy
default stream, which the cache's non-blocking streams do not wait for. The test now orders its setup on the reader
streams and runs serially. Open: long prefills are not bit-reproducible run to run even with
`NINFER_QWEN4_EXPERT_FREEZE=1`.

**T164-T166**: long prefills of new text slowed down between 16k and 64k tokens (T163: 2,791 tok/s at ~55k vs
3,435 at ~15k). Cause (nsys and NVMe counters): before every chunk the host gathers the chunk's PLE rows (16 rows of
160 B per token, scattered in a 47.7 GiB table mapped from the artifact, not resident) with both GPUs idle; rows not
in the page cache cost 0.4-6 s per 16k-token chunk. Fix: a helper thread issues MADV_WILLNEED for the next chunk's
rows right after the current chunk's gather, so the reads overlap the GPU work; the gather is unchanged (same bytes,
same values). Cold random text, same binary, prefetch off/on, 3 arms each: PLE wait per chunk after the first
0.4-5.9 s -> ~0.11 s; ~80k-token prompts 32 s -> 21.4 s median. Token identity with the expert residency frozen:
identical (2k and 16k prompts, 1,024-token chunks). The first chunk of a request is not covered; issuing its advice
from 16 threads instead of one did not help (T168: 1.06-1.28 s vs 1.24-1.28 s at ~15k tokens), so the read latency,
not the advice, bounds it.

**T170-T172**: the first PLE chunk of new text, which has no previous chunk to hide behind. A CPU-only bench of the
gather on the real table (T171) put the whole cost in the serial `MADV_WILLNEED` calls, two per row: the FP8 codes
and the BF16 row multiplier, which lives in a separate plane of the table. The multiplier plane (2 bytes per row,
640 MB) is now read once at load and kept in RAM, and the gather advises only the code pages, sorted, deduplicated
and merged into runs (`NINFER_QWEN4_PLE_RESIDENT_SCALES=0` restores the mapping). Values identical (bench hashes, PLE
test with and without). Server, cold random text: first chunk ~15k tokens 0.78-1.28 s -> 0.38-0.54 s. The reference
drive (DRAM-less NVMe) serves ~1.2-1.4 M fast random 4 KiB reads and then drops to 25k IOPS until it rests 15-45 s:
with two reads per row a cold 55k prompt (~1.76 M reads) exhausted that budget, with one it does not.

**T173-T174, T179**: server start. 26 s went to parsing the artifact directory (32 MB of JSON, 74,632 objects):
nlohmann's parser callback that rejects duplicate members made the parse ~100x slower; the check is now a separate
SAX pass (Reader 26.5 s -> 0.48 s). The 64.6 GiB of host experts were read with eight buffered streams; they are now
read as one batch with O_DIRECT from two streams, each 64 MiB piece through an aligned bounce buffer because the
objects are not 4 KiB-aligned in the files (a first version fell back to buffered reads for that reason). Weights
phase, cold cache, alternating: 37-44 s -> 20.6-23.4 s; whole start 56-141 s (v0.5.2) -> 26-29 s. The load progress
now counts the host bytes (it showed "1.2% ... ETA 1h" for a 40-second load).

**T175, T177**: where a decode round goes (short context, product configuration): ~21 ms per round, ~10 ms of pure
GPU work; 152.5 missing experts per window computed on the CPU (VRAM hit rate 0.903); GPU 0 idle 0.87 ms per round,
so host synchronizations are worth at most ~4 %.

**T182-T184**: a 58k-token prefill under nsys: GPUs busy 72 %, PCIe link 42 % (27 GB/s while busy); the router
logits (SIMT) took 1.21 s per GPU. `NINFER_QWEN4_ROUTER_MMA=1` (off by default) computes them on tensor cores for
chunks of 64+ columns (BF16 MMA, FP32 accumulation): 55k-token prefill 3.66k -> 3.91-4.04k tok/s. Same FP32 contract,
another summation order: expert selections can change near ties. T185, the quality gate with it on: GSM8K 1319
0.9765 (GB300 0.9810, 7 vs 1 discordant, McNemar p 0.070, pass), MMLU-Pro 1008 0.8452 (GB300 0.8452, 25 vs 25):
default since v0.5.3. T186: the draft shortlist gains the 8,192 most frequent Italian tokens it lacked (14.5 % of
held-out Italian tokens were outside it, 0.24 % after): Italian answers 0.37 -> 0.45 draft acceptance, ~+11 % decode.

**T187-T192** (v0.5.4, issue #3): the context cache for agents on Qwen3.8-27B at `--tp 2`. A RAM tier for both
GPUs' halves of a checkpoint (each GPU copies its half on its own transfer stream; restored turns match an all-VRAM
run token for token, MTP/int8 and DFlash2/nvfp4); `--profile 27b` turns it on with 8 GiB, Flash-Next keeps it off
(8 GiB of it costs ~8 % of prefill on the third 64k prompt in a row, by shrinking the page cache the PLE table is read
through). A request leases its prompt plus a 4,096-token window of KV and grows it at a round boundary, after
ninfer-custom 9f108db7; an answer that outgrows its window with the GPUs full moves idle cached conversations to RAM
first and is bounded only at the edge. Agentic replay (3 sessions, 11 subagents, production flags, 4 lanes): median
TTFT 2.14 -> 0.98 s, prompt tokens from cache 67 -> 78 %, recomputed prompt tokens -35 %, decode unchanged (two
concurrent decodes: 220 tok/s each with and without the RAM tier). Guardrails: `--kv-lease-window 0` restores the
full reservation; the server refuses a RAM tier that would leave less than max(4 GiB, 5 %) of RAM available.

**T211** (v0.6): the decode after a prompt and short-prompt TTFT on Flash-Next (`--tp 2`). After every agentic turn
the first decode rounds took 45 ms instead of 18 (~250-290 ms lost per turn). Not the copies (device hits = policy
hits): the residency the prompt left. A prompt window made its chunk's most routed experts resident without a cap
(a 3k-token chunk replaced ~3.2k of the ~8k slots), its decay erased the decode's frequencies (2^(-3200/256)), and
the 6 ring + up to 28 workspace layers a wide chunk lends were refilled with the prompt's hottest experts. The decode
after a prompt routes like the decode before it, not like the prompt: replaying the policy offline on the routing of
a T71 run, the prompt's targets cost 7.2k extra CPU experts per turn, any share of them hurts (a quarter: 1.8k), and
keeping the decode's residency, frequencies and lent layers costs none. Now prompt windows change neither residency
nor frequencies (`ExpertCachePolicy::count_prompt`), lent layers refill with what they held when lent (most recently
used first), and the T69 post-prompt boost (24 windows at 8 insertions per layer per GPU) is on. T71 replay, old
rule vs three runs of the new defaults: rounds 22.7 -> 17.9-18.3 ms, decode median 124 -> 157-160 tok/s (2.84 tokens
per round either way), first round after a prompt 45 -> 23-25 ms, steady rounds 18.3 -> 16.2-16.5 ms, TTFT median
1.55 -> 1.38-1.43 s; the boost alone is worth 18.4 -> 17.9 ms (two pairs) (`NINFER_QWEN4_PROMPT_KEEP=0`,
`NINFER_QWEN4_BOOST_WINDOWS=0` restore the old rules). Prompts stream their
non-resident experts from 256 tokens instead of 512 (`--stream-min-tokens`): streaming costs a flat ~1.0 s warm
(~1.35 s on a server's first request), the CPU path 0.44 s at 97 tokens and 0.6-1.2 s at 240-380 warm, 1.5 s at 158
and 1.8-2.9 s at 250-380 cold; 1k/4k prompts 1.69/1.85 -> 1.46/1.73 s cold. Tokens change, as with any residency
change (W4A4 on the GPU, W4A8 on the CPU): without freezing two runs of one binary already differ (agentic prompts
0/8 identical, short prompts 6/8); with the residency frozen the old and new rules give 8/8 identical answers.

**T211b** (v0.6): T211 per conversation. The T194 bench (280 independent short chats) lost decode with T211
(it/off 164.8 -> 156.4 tok/s vs T202): there the previous answer's residency is stale and the prompt predicts its own
decode. A request now continues a conversation when its first text chunk starts at or past half the prompt (at least
half restored from the context cache; T71 turns restore 66-99 %, a new chat 0 %): it keeps the decode's residency as
in T211. A new conversation's routing counts are applied when its first decode window begins, to the frequencies
only (`NINFER_QWEN4_PROMPT_NEW_CAP`, default -1); lent layers are restored either way. Measured on T194, same
binary: the T69 rule (copies during the prompt) decode 152.3 -> 156.0 tok/s but TTFT 0.58 -> 0.69 s, and the time
per generated token worse (7.74 -> 7.79 s per 1k tokens); the prompt's targets copied at the first decode window
cost decode instead (all 148.9, 32 per layer 151.5); frequencies only 153.9 tok/s, TTFT 0.577 s, 7.65 s per 1k
tokens. The replayed policy on the T194 routing agrees on the residency (targets 6.66 M CPU experts, frequencies
only 7.13 M, T211 7.25 M) but not on the price of the copies, which only the GPU shows.

**T212**: direct P2P prefill overlap. The half-chunk overlap split of the prefill (T56)
existed only in the host-staging transport, so boards with peer access (driver patch, issue #3) ran the
one-call driver-staged eager copies and lost the overlap: 7,987 / 7,684 / 5,410 tok/s (4k / 16k / 64k
cold prefill) vs 8,466 / 8,322 / 5,746 for host staging, 2x 5070 Ti PHB, 27B profile, uncapped clocks at
~3,180 MHz. `ops::DirectExchange` (`allreduce.cu`) carries the split on peer memory: two lanes (one per
2,048-token half-chunk), a non-blocking pull stream per rank, and begun/received events per half: each
rank publishes its half on the main stream, pulls the peer's half device-to-device, and the combine
(the same `residual_add` body) is ordered after the received halves, so the pull overlaps the peer's
remaining GEMM. Auto-select: direct lanes with P2P and no `--elastic`; host staging without P2P or with
`--elastic` (the lent workspace is VMM memory a UVA pull cannot read); `NINFER_TP_HOST_EXCHANGE=0` / `1`
keep their old meaning (no split transport / host staging forced). Bit-identity: the op suite checks all
three transports against the FP32-widened RNE-bf16 oracle (odd byte sizes, tiny lanes; a graph capture
stays staged), and a 27B tp2 serve gave byte-identical greedy outputs (reasoning plus content) on all
three. Measured (27B, live flags, cold prefill, max_tokens = 1): 11,685 / 11,114 / 6,926 tok/s = +47 / +45 /
28 % over the one-call copies and +38 / +34 / +21 % over host staging.
This is the first prefill measurement on the peer-access pair (driver 615.71.09 P2P patch, issue #3);
the raw one-way rates of the two transports are equal (P2P D2D ~ H2D + D2H ~ 28.5-28.9 GB/s), so the
gain is the overlap, not the copy.

**T213**: per-slice pipelined prefill. The T212 half-chunk overlap
splits each eager prefill all-reduce into two half-chunk lanes when a split transport
is attached (the P2P default); a chunk under 768 tokens (the overlap's halves must
each hold at least 384) and any chunk with `NINFER_TP_PREFILL_OVERLAP=0` take the
alternative path, `run_layers_tp2`, where this entry's pipeline runs.
`ops::PipelinedExchange` (`allreduce.cu`) subdivides the chunk into slices:
`NINFER_TP_PREFILL_SLICE` (tokens per slice, default 1024; `0` keeps the one-call
exchange), halved toward 128 by `prefill_slice.h` (`for_each_exchange_slice`,
`column_slice`) until a chunk holds at least four slices (short prompts and the
prompt's last chunk pipeline as well). `run_layers_tp2` opens a per-layer scope when
a chunk holds at least two slices: each eager all-reduce of the layer (the attention/GDN
output split and the FFN split) is issued per slice on the pipeline's own lane
streams, a slice's pull (`pull_peer`, device-to-device with P2P) and combine land and
its FFN starts while the later slices are still in flight, and the scope `join()`s at
the layer's end so the output is complete before the next layer reads it. Prefill
only — decode stays on the captured mailbox. Bit-identity: the per-slice combine is
the same `residual_add` body as the whole-tensor path, so the result is the same bits;
the op suite checks the sliced lanes against the FP32-widened RNE-bf16 oracle, and a
27B tp2 serve gave byte-identical greedy outputs (512 reasoning tokens, a fixed seeded
1,500-word prompt) across the one-call, per-slice, host-staging and direct P2P
configurations. Measured (27B, tp2, the
per-slice 9,392 / 9,352 / 6,614 tok/s, +12 / +16 / +13 % over the one-call copies
(`NINFER_TP_HOST_EXCHANGE=0 NINFER_TP_PREFILL_SLICE=0`, 8,381 / 8,061 / 5,876) — and
25 / 22 / 14 % behind the T212 half-chunk direct lanes (12,449 / 11,954 / 7,730), the
P2P default, on which this pipeline is inactive except on the sub-768-token tail
chunks. At a 1,024-token prefill chunk (a separate OC run) the half-chunk split narrows
to 512-token lanes and the direct lane's margin over the pipeline (four 256-token
slices per chunk) narrows to +21/+21/+2 % (10,486/10,331/6,340 vs 8,651/8,553/6,188
tok/s); at locked cores the 64k margin flips to the pipeline (4,813 vs 4,466 tok/s).
The per-slice pull is a driver-staged copy
without P2P, where the fine-grained overlap is not measured to help, so
non-peer-access boards should set `NINFER_TP_PREFILL_SLICE=0` (the default host
staging half-chunk overlap remains their prefill overlap).
