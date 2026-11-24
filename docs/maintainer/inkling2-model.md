# Inkling 2 model reference

This reference defines the mathematics of `Inkling2LoopForCausalLM` (`inkling2_loop_text`), the
Text backbone of Inkling 2 Mini (`Inkling2LoopForConditionalGeneration`), and how the family
`src/models/inkling2/` implements it: artifact, loading, two-rank tensor parallelism, per-sequence
state, the forward, the DSpark drafter and speculative decoding, the Program and the frontend. The
family shares no model code with
`src/models/qwen3_5/`; it reuses the Engine cores, the Ops and the artifact layer.

[Engine architecture](engine-architecture.md) defines ownership and execution;
[artifact container](artifact-container.md) the stored objects and bindings;
[tensor formats](tensor-formats.md) the NVFP4 representation;
[two-GPU tensor parallelism](tensor-parallel.md) the transports and two-device graphs;
[Op development](op-development.md) the Op contracts and their qualification.

The upstream authority is the Transformers implementation `models/inkling2_loop`
(`modeling_inkling2_loop.py`, with the load-time transforms of `core_model_loading.py` and
`conversion_mapping.py`).

## Scope

| Supported | Not supported |
|---|---|
| Text generation and CausalScoring through the public Engine (`ninfer-serve`, `ninfer-perplexity`) | Vision (`--vision` is refused at startup) |
| `--tp 1` and `--tp 2`; the released model's weights need two 16 GB boards, synthetic checkpoints run at either width | `--kv-dtype` other than `bf16` |
| Speculative decoding with the checkpoint's DSpark drafter (artifact component `mtp`): `--spec mtp --draft-tokens K` (off by default in the server, K = 3 in `capy-serve-inkling2-mini.sh`), on greedy and sampled rows | `--spec dflash2`, `--lm-head-draft` |
| One request at a time (`--max-concurrency 1`); further requests queue | Concurrent requests, decode batches of more than one row |
| Prefix reuse on the one lane: the finished sequence's endpoint and its prompt-end anchor, device only | Shared prefixes, several catalogued conversations, `Retain` sources, Host and disk tiers (`--disk-cache-dir` is refused) |
| Greedy decoding and host sampling (temperature, top-k up to 20, top-p, min-p, presence and frequency penalties) | Structured output and grammars |
| Tool calls (JSON), thinking budget, stop tokens and strings | `tool_choice` other than automatic (`required`, allowed names, `parallel=false`, strict constraints) |
| Decode replayed from one CUDA Graph (`--no-cuda-graph` keeps it eager) | Continuation of a final assistant message |
| `--embedding-host`: the token embedding in one mapped page-locked host copy ([Embedding in host memory](embedding-host.md)) | |

The refusals live in `parse_config` ([`config.cpp`](../../src/models/inkling2/config.cpp)),
`validate_options` ([`instance.cpp`](../../src/models/inkling2/instance.cpp)) and the frontend.
The Program accepts K from 1 to 7 (the drafter's block of 8 holds the anchor and 7 proposals);
the server's command line accepts 1 to 5 for `--spec mtp`, as for the Qwen models
([`speculative_options.h`](../../src/product/speculative_options.h)). An artifact without the
`mtp` component starts with `--spec mtp` and decodes without a drafter.

## Configuration

The Text config is the checkpoint's `text_config` verbatim plus `architectures`.
`parse_text_config` reads every member or checks it against the one supported value; an unknown
member is an error. `is_inkling2` accepts exactly the architecture/`model_type` pair above.

| Quantity | Inkling 2 Mini | Config field |
|---|---:|---|
| Hidden width H | 2304 | `hidden_size` |
| Physical layers; global layers | 24; 5, 11, 17, 23 | `num_hidden_layers`; `(i + 1) % 6 == 0` (fixed rule) |
| Q heads / KV heads / head dimension D | 24 / 8 / 128 | `num_attention_heads`, `num_key_value_heads`, `head_dim` |
| Relative state width dr; relative extent (global) | 16; 1024 | `d_rel`, `rel_extent` |
| Sliding window (local window and local relative extent) | 1024 | `sliding_window_size` |
| q/k clip, relative clip | √512 ≈ 22.63, 8.0 | `qk_l2_clip_threshold`, `rel_l2_clip_threshold` |
| Log scaling α, floor F | 0.1, 8192 | `log_scaling_alpha`, `log_scaling_n_floor` |
| Short-convolution taps K | 4 | `sconv_kernel_size` |
| Experts E / selected k / expert width F; route scale | 128 / 8 / 1152; 8.0 | `n_routed_experts`, `num_experts_per_tok`, `intermediate_size`, `route_scale` |
| Gate / up soft caps | 4.0 / 25.0 | `mlp_gate_softcap`, `mlp_up_softcap` |
| Final logit soft cap | 30.0 | `final_logit_softcapping` |
| Layers per residual block | 6 | `n_layers_per_attn_res_block` |
| Looped core, passes, message gate rank r | layers 6-17, 3, 64 | `loop_config` |
| Embedding rows V; output rows Vu | 201,024; 200,058 | `vocab_size`, `unpadded_vocab_size` |
| Position capacity | 1,048,576 | `model_max_length` |

`rms_norm_eps=1e-6`. There are no projection biases and no shared expert. The core must be whole
residual blocks after the first one. `TextConfig::visits` is the derived 48-visit schedule.

## Mathematics

Activations are columns: a [H, T] tensor holds T tokens of one sequence. `bf16(x)` is
round-to-nearest-even to BF16, `f32(x)` the exact FP32 value of a stored input, `rms0(x)` the RMS
normalization without gain, `softcap(x, c) = c * tanh(x / c)`.

### Visits and state slots

A physical layer can run more than once. One execution is a **visit**; the forward runs 48:

| Execution index | Layers | Pass | State slots | Messages |
|---|---|---|---|---|
| 0-5 | 0-5 (prelude) | 1 | 0-5 | no |
| 6-17 | 6-17 (core) | 1 | 6-17 | no |
| 18-29 | 6-17 (core) | 2 | 24-35 | yes |
| 30-41 | 6-17 (core) | 3 | 36-47 | yes |
| 42-47 | 18-23 (coda) | 1 | 18-23 | no |

A visit of a later pass uses the core layer's weights with its own state slot, so the slot index
does not follow the execution order. The global visits are those of layers 5, 11, 17 and 23: slots
5, 11, 17, 23, 29, 35, 41 and 47. The other 40 slots are local.

### Block attention residuals

A visit of a layer that is a multiple of 6 (0, 6, 12, 18) is a **block boundary**: its input `x`
joins the list `B` of completed block states, and the visit's residual restarts from its own
output. Every read of the residual stream (attention input, FFN input, head input) is a depth
attention of learned FP32 queries over `B` and the current state ([`depth_attention.h`](../../include/ninfer/ops/depth_attention.h)):

```text
S_s    = rms0(source_s)                     sources: B[0..n-1], then the current state
a[q]   = softmax_s(query_q · S_s / sqrt(H)) # FP32, eps 1e-6 in rms0
out[q] = sum_s a[q,s] * S_s                 # FP32 [H, T] per query
```

Keys and values are the same normalized states; tokens are independent. The pass structure:

```text
x = bf16(rmsnorm(embed[ids], embedding_norm));  B = []
visits 0-5                      # B = [emb] after layer 0
x_pre, B_pre = x, B             # saved at the first core visit
visits 6-17 (pass 1)            # adds x6 = x_pre and x12
for pass 2, 3:
    bank = B[len(B_pre)+1:]     # the previous pass's later blocks: [x12], x6 skipped
    prev = output of layer 17 of the previous pass
    x, B = x_pre, B_pre         # restart from the prelude
    visits of layers 6-17 with messages from (bank, prev)
visits 42-47 (coda)             # B of pass 3; layer 18 adds out17 of pass 3
head over B = [emb, x_pre, x12, out17] and out23
```

### Cross-pass messages and gates

In passes 2 and 3 each core visit computes four messages with its `[4, H]` FP32
`message_queries` (rows Q, K, V, FFN):
`m = depth_attention(bank, prev, message_queries)`. A message gate of rank r mixes an input
with its message ([`message_gate.h`](../../include/ninfer/ops/message_gate.h)):

```text
z   = [bf16(rms0(h)), bf16(rms0(m))]            # [2H]
d   = bf16(down · z)                            # down BF16 [r, 2H]
out = (1 - tanh(forget · d)) * h + tanh(gain · d) * m       # forget, gain FP32 [H, r]; out FP32
```

### One visit

```text
a      = x (layer 0) or depth_attention(B, x, attention_queries)  # [1|3, H] FP32 queries
x_q,k,v= local: bf16(rmsnorm(a, input_norm)) shared by Q, K and V
         global: three queries, bf16(rmsnorm(a_j, {query,key,value}_input_norm))
         later passes: a_j = gate_j(a_j, m_j) first (local: one a, three gated copies, one norm)
y      = attention(x_q, x_k, x_v)                                # BF16 [H, T]
y'     = short_conv_static(y, post_attention_conv)
r      = f32(x) + f32(y'),  or f32(y') at a block boundary (then B += [x])     # FP32
u      = depth_attention(B, r, ffn_queries); later passes: u = gate_ffn(u, m_ffn)
u      = bf16(rmsnorm(u, ffn_norm))
z      = MoE(u)                                                  # BF16
x'     = bf16(r + f32(short_conv_static(z, post_ffn_conv)))
```

At a block boundary the visit output does not contain its input, which lives on only in `B`.

### Attention

Per visit, from `x_q`: `q [24·128]`, `gate [24·128]`, `r [24·16]`, `sink [24]`; from `x_k`, `x_v`:
`k`, `v [8·128]`, then the visit's short convolution. Query head h reads KV head `h / 3`.
[`relative_sink_attention.h`](../../include/ninfer/ops/relative_sink_attention.h) then defines,
with `clip(x, t) = x * min(1, t / (|x| + 1e-6))` per head vector and
`tau(p) = 1 + α ln(max((p + 1) / F, 1))` at absolute position p:

```text
q' = bf16(bf16(clip(q * query_scale, 22.63)) * tau(p))       # tau only in global visits
r' = bf16(bf16(clip(r * relative_scale, 8.0)) * tau(p))
k' = bf16(clip(k, 22.63))                                    # no learned scale
v' = FP16(bf16(rms0(v)))                                     # per head, eps rms_norm_eps
bias(p, s)  = d < R ? bf16(r' · relative_profile[:, d]) : 0  # d = p - s
logit(p, s) = bf16(bf16(bf16(q' · k') / sqrt(128)) + bias(p, s))
prob        = softmax over {logits, sink} (FP32), the sink's own mass discarded, then bf16
o           = bf16(sum_s prob * v');  o = o * sigmoid(gate);  y = output · o
```

Local visits see `p - 1024 < s <= p` and have `R = 1024`; global visits see every `s <= p`,
`R = rel_extent`, and scale q' and r' by tau (τ = 1 up to position F - 1 = 8191). The sink enters
unscaled and without bias. There is no RoPE.

### Short convolutions

[`short_conv.h`](../../include/ninfer/ops/short_conv.h). Each holds the three previous inputs
of its channel.

```text
static  (local K and V, post-attention, post-FFN):
        out[t] = bf16((sum_{j=0..3} w[j] * u[t-3+j] + x[t]) / sqrt(5))      # tap 3 = current token
dynamic (global K and V):
        tap[l] = static_weight[l] + dyn[4g + l]     dyn = bf16(conv_projection · x_k or x_v)
        out[t] = bf16((sum_{l=0..3} tap[l] * u[t-l] + x[t]) / 3)            # lag 0 = current token
```

The two forms index their taps in opposite orders, as the checkpoint stores them.

### MoE

[`moe_sigmoid_route.h`](../../include/ninfer/ops/moe_sigmoid_route.h),
[`moe_bank.h`](../../include/ninfer/ops/moe_bank.h):

```text
s       = sigmoid(router · u)                  # FP32 router [E, H] over the BF16 input
ids     = top-8 of s + router_bias             # the bias only orders the selection
w       = s[ids] / (sum s[ids] + 1e-20) * 8
g, a    = bf16(W_gate[e] · u), bf16(W_up[e] · u)
h       = bf16(softcap(g, 4) * sigmoid(g) * softcap(a, 25))
o_e     = bf16(W_down[e] · h)
z       = bf16(sum_j w_j * o_{e_j})            # FP32 sum, one rounding
```

### Embedding and head

```text
x0     = bf16(rmsnorm(embed[ids], embedding_norm))
h      = bf16(rmsnorm(depth_attention(B, x, final_queries), final_norm))
logits = bf16(output_head · h)                 # Vu = 200,058 rows; argmax reads these
capped = softcap(f32(logits), 30)              # FP32: sampling and scoring read these
```

## Artifact and conversion

### Logical parameters

[`weights.h`](../../src/models/inkling2/weights.h) lists the names; they are the artifact
contract. Under `text/` (`<L>` physical layer, `<e>` expert):

| Logical parameter | Shape | Stored as |
|---|---|---|
| `token_embedding`; `embedding_norm`, `final_norm` | [V, H]; [H] | BF16 |
| `final_queries`; `layers/<L>/attention_queries` (L > 0); `layers/<L>/ffn_queries` | [1, H]; [1\|3, H]; [1, H] | FP32 |
| `output_head` (Use `text/final_hidden`) | [Vu, H] | BF16, first Vu rows of `unembed` |
| `layers/<L>/input_norm` (local), `{query,key,value}_input_norm` (global), `ffn_norm` | [H] | BF16 |
| `attention/{query,gate,relative,sink}` (Use `attention/query_input`) | [nh·D \| nh·D \| nh·dr \| nh, H] | BF16 |
| `attention/key`, `attention/value` (Uses `attention/{key,value}_input`) | [nkv·D, H] | BF16 |
| `attention/output` (Use `attention/gated_output`) | [H, nh·D] | BF16 |
| `attention/query_scale`, `relative_scale`, `relative_profile` | [nh, D], [nh, dr], [dr, window or rel_extent] | BF16 |
| `attention/{key,value}_conv` (local) | [nkv·D, K] | FP32, tap K-1 = current |
| `attention/{key,value}_conv_static`; `{key,value}_conv_projection` (global) | [K, nkv·D]; [nkv·K, H] | FP32; BF16 |
| `post_attention_conv`, `post_ffn_conv` | [H, K] | FP32 |
| `moe/router`, `moe/router_bias` | [E, H], [E] | FP32 |
| `moe/experts/<e>/gate_up` (Use `ffn_input`), `down` (Use `moe/experts/<e>/product`) | [2F, H], [H, F] | NVFP4 or MXFP8 (below) |
| `message_queries` (core) | [4, H] | FP32, rows Q, K, V, FFN |
| `message_gate/{query,key,value,ffn}/down` (Use `message_gate/<role>/input`) | [r, 2H] | BF16 |
| `message_gate/<role>/{forget,gain}` | [H, r] | FP32 |

Every Use is `A16Only`: activations are never quantized. The FP32 router, convolution and
forget/gain weights are BF16 in the checkpoint and widened exactly, because their Ops read FP32.
The converter ([`inkling2.py`](../../tools/convert/inkling2.py)) also maps the checkpoint's
conventions: `attn_res.attn_queries.N` belongs to layer N + 1, `attn_res.mlp_queries.N` to
layer N; the two cross-pass query tensors are reordered into one `[4, H]` matrix per core layer;
`controllers.attn` is the query gate.

### Expert banks

`w13` interleaves gate (even rows) and up (odd rows); the logical `gate_up` puts the F gate rows
first. Each layer's experts form **two banks per matrix**, IDs [0, E/2) and [E/2, E), each one
parent whose rows are its experts' rows in ID order.

- Layers stored NVFP4 by the checkpoint (0-19 in Inkling 2 Mini) are imported without
  requantization (`import_encoded`): E2M1 codes, E4M3 block scales of 16 input columns, and one
  weight divisor per expert `fl32(denominator / amax)`. A bank parent is `nvfp4` in
  `block_scale_k16_m128x4` with one divisor word per expert
  ([tensor formats §3.3](tensor-formats.md#33-nvfp4)).
- Layers stored MXFP8 (20-23) are copied verbatim (`import_verbatim`): E4M3FN codes and one UE8M0
  exponent per row and 32 input columns, `mxfp8_e4m3fn_g32_ue8m0` in `block_scale_k32_v1`. The bank
  Op gives the bits of a BF16 bank holding the dequantized values.

`Parameters` builds `ExpertBankParameters` from the consecutive resident experts that share one
parent and refuses formats other than these two (and BF16, which the Op also accepts).

### Stacked attention inputs

The recipe stacks the projections that read one input into one parent, so the forward issues one
Linear per input:

| Layers | Parent rows | Input |
|---|---|---|
| local | query · gate · relative · sink · key · value | `x_q` (= `x_k` = `x_v` outside the messages) |
| global | query · gate · relative · sink | `x_q` |
| global | key · key_conv_projection; value · value_conv_projection | `x_k`; `x_v` |

Each logical parameter keeps its own binding and Use. `Parameters` prepares the stacked operands
(`queries`, `all`, `keys`, `values`) with `prepare_linear_weight` over consecutive rows of one
parent and throws for an artifact whose projections are separate parents. In the message passes a
local visit reads `queries` on `x_q` and the `key` and `value` views on `x_k` and `x_v`.

### The drafter component

The checkpoint's DSpark drafter (`mtp.safetensors` and `mtp_config`) converts as the optional
component `mtp` (`Inkling2DSpark`, `inkling2_dspark`; [`draft/config.h`](../../src/models/inkling2/draft/config.h)).
`parse_draft_config` reads every member or checks it against the one supported value: the hidden
size is the Text component's and `target_layer_ids` names three visits of the schedule followed by
the head input. Every parameter keeps its BF16 words under `mtp/` (names in
[`weights.h`](../../src/models/inkling2/weights.h)): the mixing weights of the four target
states, a mask embedding, three pre-norm layers and the recurrent head. One tensor is derived: the
recurrent head's output projection is stored folded into the target's output head,
`markov/folded_output = bf16(f64(unembed[:Vu]) @ f64(hidden_out))` and its bias likewise, so the
drafter never reads `hidden_out` alone. The recipe stacks, per layer, `query|key|value`,
`context_key|context_value` and `mlp/gate|up` into one parent each (one Linear per input).

At tp 2 ([`load/sharding.cpp`](../../src/models/inkling2/load/sharding.cpp)) the drafter's
query rows split by query heads, its key, value and context key/value rows by KV heads, its
output projection by columns, `mlp/gate` and `mlp/up` by rows and `mlp/down` by columns, and
`markov/folded_{output,bias}` by vocabulary rows like `text/output_head`; norms, mixing weights,
the mask embedding, the recurrent token embedding and the recurrence are replicated.

### Regenerating and checking an artifact

`inkling2_mini` is an official recipe (`tools/convert/official_recipes.py`, which delegates to
`inkling2_recipes.py`):

```bash
python3 -m tools.convert --model /path/to/Inkling-2-Mini \
  --recipe inkling2_mini --components text,mtp --device cpu \
  --name inkling-2-mini --out models/capy-flash-inkling2-mini-nvfp4.ninfer
python3 -B tools/ref_ink/check_conversion.py --checkpoint /path/to/Inkling-2-Mini \
  --artifact models/capy-flash-inkling2-mini-nvfp4.ninfer --report check.json
```

`--components` is `text` (the default) or `text,mtp`. The converter refuses an existing output or
report path. `check_conversion.py` runs with the Transformers of the Inkling 2 implementation and
compares the artifact with the reference's own load-time transforms: MXFP8 layers must be
bit-identical, NVFP4 layers report the distance to the reference's BF16-rounded weights, dense
parameters must equal the checkpoint words; with an `mtp` component every drafter tensor must
equal its words and the folded head must equal the fold recomputed in FP64.
`tools/ref_ink/synthetic.py` writes a small synthetic checkpoint in the original format (same
topology, NVFP4 and MXFP8 experts, a window, relative extent and log-scaling floor short enough
for a 256-token sequence) that converts with the same recipe; `tools/ref_ink/synthetic_mtp.py`
adds a synthetic drafter of the real topology at its widths, for the `text,mtp` test artifact.

## Loading and two-rank tensor parallelism

`plan_load`/`materialize_model` ([`load.h`](../../src/models/inkling2/load.h)) bind the Text
backbone at `options.tp` devices. At tp 2 `shard_rule` ([`load/sharding.cpp`](../../src/models/inkling2/load/sharding.cpp))
places every logical parameter; `parent_placements` combines the parameters of one parent and
requires their ranges to partition it, so a stacked parent's rank shard is the rank's rows of each
block in order.

| Placement | Parameters |
|---|---|
| Rows by query heads (12 of 24 per rank) | `attention/{query,gate,relative,sink,query_scale,relative_scale}` |
| Rows by KV heads (4 of 8) | `attention/{key,value,key_conv,value_conv,key_conv_projection,value_conv_projection}` |
| Columns by query heads | `attention/output` |
| Columns by KV heads | `attention/{key,value}_conv_static` |
| Rows by vocabulary (100,029 per rank) | `text/output_head` |
| Rows / columns by intermediate channels | `moe/experts/*`: every expert at half its intermediate width F on each rank (gate and up rows [r·F/2, (r+1)·F/2), the matching `down` columns), cut by the model at load (`load/expert_banks.h`) |
| Replicated | embedding and norms, block-residual, message and final queries, message gates, router and bias, `relative_profile`, post-attention and post-FFN convolutions |
| HostMapped (`--embedding-host`, at tp 1 too) | `text/token_embedding`: one copy in mapped page-locked host memory that both ranks read in place |

At tp 1 every parameter is whole on device 0 and the rank holds both banks.
`shard_text_config` ([`execution/tp.cpp`](../../src/models/inkling2/execution/tp.cpp)) halves the
query and KV heads and the output rows; experts keep their global IDs and are routed on every rank.

Each rank keeps the whole residual and issues the replicated work itself: block residuals,
messages and gates, norms, router, post-attention and post-FFN convolutions. A visit exchanges
twice:

1. the attention output projection, row-parallel (`linear_row_parallel`): each rank's BF16
   partial over its heads, one BF16 all-reduce;
2. the MoE: each rank's half-width experts into an FP32 partial. Calls of up to 16 tokens
   (decode and drafter verification) sum the two partials with `allreduce_sum_f32` and round to
   BF16 once; wider calls (prefill) round each partial to BF16 and all-reduce in BF16 with FP32
   accumulation, half the bytes ([`forward/moe.h`](../../src/models/inkling2/execution/forward/moe.h)).

The head computes each rank's vocabulary half and gathers it on rank 0 (`concat_rows_rank0`,
[`tp2/concat_rows.h`](../../include/ninfer/ops/tp2/concat_rows.h)), which applies the soft cap and
chooses the token. The replicated Ops are deterministic, so both ranks hold bit-identical residual
streams, routing decisions and convolution tails.

Weight bytes per rank at tp 2, computed from the shapes of Inkling 2 Mini:

| Class | Per rank |
|---|---:|
| Experts at half width (128 experts × 20 NVFP4 layers + 128 × 4 MXFP8 layers) | 7.83 GB |
| Token embedding (replicated; host memory with `--embedding-host`) | 0.93 GB |
| Attention shards | 0.65 GB |
| Output head half | 0.46 GB |
| Message gates, router, queries, convolutions, norms | 0.12 GB |
| **Total** | **≈ 9.99 GB** |

The whole model at tp 1 is about 18.9 GB. With `--spec mtp` and an `mtp` component, the
drafter's weights, its context caches and the round's buffers take about 730 MiB more VRAM per
board at `--max-context 131072` and about 1.0 GiB at the launcher's 253,952.

## Per-sequence state

[`VisitState`](../../src/models/inkling2/state/visit_state.h) holds what a token leaves for the
tokens after it, per rank and per visit slot:

| State | Layout | Per rank at tp 2 |
|---|---|---|
| Local K/V (40 slots) | `CyclicKVCache` ring of `window` + 7 positions (padded to 1152), slot `x mod capacity` holds position x; K BF16, V FP16 | 2.25 MiB per slot, 90 MiB |
| Global K/V (8 slots) | page-major planes [D, 64, Hkv, pages] for `max_context`, K BF16, V FP16, read through `PagedKVBatchLayerView` with an identity block table of one row | 16 KiB per position |
| Convolution tails (48 slots) | four BF16 [C, 3] tails per slot: K and V over the rank's KV channels, post-attention and post-FFN over H | about 1.5 MiB |

`reset()` zeroes the tails and the batch-row index, the only state that must be zero at a new
sequence; K/V rows are written before they are read. The rings' 7 extra positions let a
speculative verification call write up to 7 positions past the frontier it commits without
overwriting a position that a later query still sees. At tp 2 the K/V and their tails are the
rank's heads; the post-attention and post-FFN tails are replicated and updated identically.

Block states, messages and residuals are **not** state: they are per token and per depth and are
recomputed inside every call. There is no rollback: re-running a position requires restoring the
state first.

## Execution

### The forward

[`execution/forward.h`](../../src/models/inkling2/execution/forward.h) declares one template,
`Forward<Arena>`, instantiated for execution (`WorkspaceArena`) and for workspace sizing
(`WorkspaceLayoutBuilder`: the same allocations in the same scopes, each Op replaced by its
declared scratch). `forward_workspace_bytes` runs the planning instance, so the planned workspace
is the executed one by construction. The members are split by responsibility in
`execution/forward/` (`core.h`, `visit.h`, `residual.h`, `attention.h`, `moe.h`, `head.h`) and
compiled once in `forward.cpp`.

One `run(ids, ForwardColumns{width, first_position, logit_columns}, ForwardLogits{logits, capped})`
processes T consecutive tokens of one sequence (a prefill block, or T = 1 for a decode step) and
advances every rank's `VisitState` by T positions. Positions are read on the device from
`ForwardRank::positions`. The head runs only for the last `logit_columns` columns: `logits` BF16
[Vu, n] and, when given, `capped` FP32 [Vu, n] on rank 0. A `ForwardTap` receives
teacher-forcing points (visit outputs and internal parts, routing ids and weights) and, at tp 2,
both ranks' copies of each replicated value.

### Routes and bits

The Ops choose private routes by the number of columns T of a call:

| Op | Narrow route | Wide route |
|---|---|---|
| BF16 Linear | T ≤ 16: GEMV | T > 16: tiled WMMA |
| `relative_sink_attention_*` (D = 128) | T ≤ 16 per row: each query's bits independent of T | longer calls: tensor-core products |
| `moe_bank_capped_swiglu` | T ≤ 47: one fixed per-lane order | T ≥ 48: tensor cores |
| `message_gate` | cluster and per-token routes | T ≥ 128: the wide route, with the same bits |

The routes below run faster than the simpler route of the same width and give its bits; only a
narrow/wide boundary of the table changes bits:

- **BF16 shapes of the family** ([`shapes/inkling2.cu`](../../src/ops/linear/bf16/shapes/inkling2.cu)):
  the tp 2 shards of the attention inputs (queries, all, key/value, keys/values) and of the
  attention output have their own kernels at both widths, with the generic route's arithmetic per
  output element (the same lane groups, FMA chain and sums in the GEMV; the same m16n8k16 products
  in increasing K in the tiled kernel), hence its bits. Other shapes, and unaligned operands, take
  the generic route.
- **Attention**: a one-token call on a ring (the local layers at decode) runs the cluster route,
  one cluster of 8 blocks per (token, KV head) exchanging chunk statistics through distributed
  shared memory; it evaluates the chunks with the split route's functions in its order, so the same
  bits. The output gate `o · sigmoid(gate)` is applied inside the Op (the bits of `sigmoid_mul`
  after it).
- **Message gates**: up to 16 columns a visit's gates and the RMS norms of their outputs run as
  one launch (`message_gates_rmsnorm`, bit-identical to `message_gate` then `rmsnorm_f32` per
  gate); from 128 columns `message_gate` takes its wide route, same bits.
- **Residual**: up to 16 columns (decode and verification calls, without a teacher-forcing tap) a
  residual branch's short convolution, the residual addition, the depth attention and the RMS
  norms that follow are one kernel (`residual_conv_mix`, `depth_attention_rmsnorm`,
  `residual_conv_round` in [`residual_mix.h`](../../include/ninfer/ops/residual_mix.h)), each
  bit-identical to the Ops it replaces called in sequence.
- **Router** (`moe_sigmoid_route`): one cluster per column computes the logits and selects the
  top-k in one pass; wide calls take a tile of columns per block that reads each router row once.
  A logit's reduction order depends only on H, so every route gives the same bits.
- **Expert bank**: wide NVFP4 products stream codes, scales and activations through a three-stage
  `cp.async` pipeline and run the wide route's MMAs in the same K order, so the wide route's bits.

Consequences:

- Two executions with the same call boundaries give the same bits.
- Prefill blocks of up to 16 columns equal token-by-token decode bit for bit
  (`ninfer_inkling2_forward_blocks_test`).
- Decode (T = 1) and a wide prefill block agree within the Ops' oracle criteria, not bit for
  bit; so do two prefills with different block boundaries. A teacher-forcing comparison must
  therefore fix the block structure it compares.

### Decode in one CUDA Graph

`ForwardExecution` ([`program/execution.cpp`](../../src/models/inkling2/program/execution.cpp))
captures the T = 1 forward, the soft cap and the argmax on rank 0 once, at the lane's first
single-token decode (at startup, where `prepare` runs one prompt block, one decode step and, with a
drafter, one speculative round before the first request), and replays it for every later step of
every request: the state, the token
and position buffers and the workspace keep their addresses for the Program's lifetime (the arena
is deterministic at fixed T), and the position is read on the device. At tp 2 the capture spans
both devices: a 16 KiB `PeerMailbox` is attached to the `PeerEvents` before the capture and a
`DecodeGraphPeerBridge` gates each launch ([tensor parallelism §5](tensor-parallel.md#5-cuda-graphs-across-two-devices)).
Prefill, scoring, decode calls of more than one token (after forced control tokens) and every call
under `--no-cuda-graph` run eagerly; without peer access their exchanges use a `HostExchange`.

### Speculative verification

[`program/speculative/round.h`](../../src/models/inkling2/program/speculative/round.h) verifies an
anchor and K ≤ 7 drafts in one CUDA Graph over both devices, captured once per K: one ingress
record per rank (anchor, frontier, extent P, drafts), `speculative_prepare_verify_inputs` at the
physical width K (consecutive positions), the forward at T = K + 1 with every column's logits,
the column argmax and `speculative_accept_greedy_drafts` over P on rank 0, the licensed count
pulled by rank 1, and one egress record (licensed tokens and count). During the call every short
convolution writes its input history into the rank's `ConvRecord`
([`state/conv_record.h`](../../src/models/inkling2/state/conv_record.h),
[`short_conv_record.h`](../../include/ninfer/ops/short_conv_record.h)) instead of advancing its
tail; one `conv_tail_fold` then writes all tails after the committed count. The local rings hold
window + 7 positions and the global K/V are rewritten from the frontier, so after a round the
state holds exactly the committed tokens, and every committed column's logits equal those of a
decode step at its position bit for bit. A tp 2 mailbox carries the round's exchanges in wide
slots of `round_exchange_bytes` (the MoE's FP32 all-reduce at T = K + 1). The forward driver's
`--mode spec` runs rounds with fake drafts; `--mode spec --drafter` with the artifact's drafter.

### The DSpark drafter

[`draft/draft.h`](../../src/models/inkling2/draft/draft.h) holds the drafter over one sequence, at
one device or split over two ranks; its Ops are in
[`dspark_draft.h`](../../include/ninfer/ops/dspark_draft.h).

- **Features.** The drafter reads four states of the target at every committed position: the
  inputs of three visits and the output head's normed input. The forward writes them through a
  `FeatureSink` ([`execution/forward.h`](../../src/models/inkling2/execution/forward.h)) into the
  drafter's pending features, computing those values directly into the destination, so a call with
  a sink gives the same bits as one without (`ninfer_inkling2_features_tp2_test`).
- **Context.** The pending features of the columns a call licensed are mixed per layer
  (`feature_mix_rmsnorm`), projected by the layer's context key and value rows, normed, rotated
  and stored in the layer's cache (`context_kv_store`); local layers keep a ring of `window`
  positions, the global layer every position up to `max_context`. Prefill blocks and eager calls
  (forced tokens) feed the context too; a round's features are consumed by the next round, or by
  `catch_up` before an eager call.
- **Proposal.** One forward of the block [anchor, mask × 7] at positions P .. P + 7 through the
  three layers (block attention over the cache and the block), then the recurrent head chained
  over the proposals, each the top-1 over the vocabulary; it stops after the K proposals the round
  verifies.
- **Tensor parallelism.** Each rank holds its heads, its MLP half and the folded head over its
  vocabulary rows; a layer exchanges twice (output and down projections, BF16 partials), each
  proposal once (the two ranks' top-1, 16 bytes). Both ranks take the same token. The drafts are
  not bit-identical to one device's; the target verifies them.
- **Prefix reuse.** The prompt anchor's image also holds the drafter's local rings and its
  recurrent state; the global layer's cache is per position, like the target's global K/V.

### Speculative decoding

With a drafter (`--spec mtp --draft-tokens K` and an artifact with `mtp`), `ForwardExecution`
owns the drafter and one `VerifyRound` of fixed K, whose graph also runs the drafter: it
materializes the context the last call licensed, writes the K proposals over the record's drafts,
and the forward fills the next pending features. `ProgramImpl::speculative_extent` decides per
decode step:

- A row with forced tokens to execute (after a thinking-budget control) takes an ordinary step,
  and so does a row whose K + 1 positions would pass `--max-context`; a row near its token budget
  verifies fewer drafts.
- A greedy row without penalties accepts on the device: the longest draft prefix equal to the
  argmax, then the correction or the bonus token.
- Any other row samples on the Host, column by column: column j takes the token the ordinary
  decode step's sampler draws from that column's capped logits at position frontier + j, with the
  emitted tokens and those taken earlier in the round as the penalties' history; the round goes on
  while that token equals the draft. The draw is a function of (seed, position) and the column's
  logits are the decode step's bit for bit, so every token is the one an ordinary decode step
  draws: with the same seed the text is the same with and without the drafter. The Host's count
  can differ from the device's greedy count; `settle` (`VerifyRound::recommit`) then folds the
  convolution tails and the drafter's context to it.

Acceptance and drafted tokens are reported in `SpeculativeStats` (the server's request log). A
round costs more than a decode step, so a request whose drafts are rarely accepted can decode
slower than without the drafter. Measured on this release with K = 3 (from the server, temperature 0): 149.0 tok/s
against 127.5 on a short prompt and 166.6 against 112.8 at 12,800 tokens of context, with 2.0 and 3.0 tokens a
round. By K, measured earlier the same day on the code before the faster Ops (short prompt, decode 106.3 tok/s):
K = 1 133.0, K = 3 148.8, K = 5 134.7, K = 7 119.0 tok/s.

A greedy row without penalties reads the argmax of the BF16 logits (the soft cap is monotone).
Any other row copies the capped FP32 logits of the last column to the host and samples: penalties,
the top-k best (k ≤ 20; 0 means 20), temperature, min-p, top-p, one draw from a counter keyed by
(seed, position). Scoring runs blocks of at most 64 columns with `capped` and a host FP32
log-softmax.

## Program and Engine integration

### The contract

[`program/program.h`](../../src/models/inkling2/program/program.h) declares the types the common
cores instantiate (`RuntimeTypes`, [`program/runtime_types.h`](../../src/models/inkling2/program/runtime_types.h));
`ProgramImpl` ([`program/program_impl.cpp`](../../src/models/inkling2/program/program_impl.cpp))
implements them over one lane and `ProgramExecution` drives the forward.

| | Methods |
|---|---|
| Implemented | `plan_request`, `causal_score`, `inspect_admission` (root, resident endpoint, prompt anchor), `seal_identity`, `begin_pressure_planning` with `root_maximal_target`/`assess`/`seal` only, `checkpoint_recovery_work` (continuation), `release_continuation`, `start_resource_transaction`, `progress_context_transaction`, `finalize_context_transaction`, `has_context_transaction`, `advance_prefill`, `decode`, `append_forced_tokens`, `commit`, `abort_pending`, `finish`, `finish_cancelled` (= `finish`), `abort`, `fail_all_cleanup`, `isolated_request_feasible`, `resource_revision`, `physical_usage`, `memory_summary`, `reset_memory_peaks`, `tp_transport`, `device_kv_available_pages`, `shared_capture_split_prefill_work` without frontiers, `RequestBasePlan::prefix_shortlist_key` |
| Fixed empty answers | `disk_resume_tokens` 0; `prove_persistent_backfill`, `device_kv_lease_settlement_tokens`, `device_kv_lease_shortfall`, `demote_retained_*`, `idle_device_kv_host_bytes`, `drop_continuation_anchor`, `shared_candidate_rebuild_work`: none; `resume_device_kv_lease` true; `retained_device_kv_pages`, `compact_device_kv`, `release_shared_prefix`, `releasable_device_kv_pages` of empty lists: empty; `sampling_bypass_hooks`, `shared_capture_matches`, `idle_device_kv_offload`, `peer_exchange_timed_out`, `disk_cache_enabled`: false; `host_kv_free_bytes` 0; the other `disk_cache_*` inactive |
| `std::logic_error` (not supported) | every other `PressurePlanningSession` method and `seal_capture`; every `CapturePressurePlanningSession` method; `inspect_capture`, `begin_capture_pressure_planning`, `skip_capture`, `reserve_active_capture(_with_pressure)`, `checkpoint_recovery_work` of a shared prefix, `idle_device_kv_pages`, `releasable_device_kv_pages` of non-empty lists, `recover_peer_transport` |

A constrained output row raises `RequestError(UnsupportedOutputCombination)` in `advance_prefill`
and `decode`. `advance_prefill` runs the prompt in chunks of `--prefill-chunk` tokens and chooses
the first token on the last chunk; `decode` runs `tokens[executed..)`, i.e. the last committed
token, or after a thinking-budget control also the three forced tokens, in one call.

### Engine hooks

`Engine::Impl` ([`engine.cpp`](../../src/runtime/engine/engine.cpp)) asks
`inkling2::is_inkling2_artifact` before `runtime::construct_model`: it opens the artifact's
`Reader` and compares the Text component's `architectures` and `model_type` exactly. A path that
does not open, or another family, returns false and continues on the Qwen path, which reports its
own errors. For an Inkling 2 artifact `EngineFamily::start`
([`engine_family.h`](../../src/models/inkling2/engine_family.h)) constructs the instance and the
common `EngineCore` or `CausalScoreCore` over `inkling2::ModelInstance`; every public method that
reaches the frontend or the core then begins with `if (impl_->inkling2)`: `prepare`,
`prepare_tokens`, `prepare_continuation`, `reasoning_guard_tokens`,
`configure_reasoning_eos_guard`, `tokenize_text`, `detokenize_text`, `score_tokens`,
`count_tokens`, `submit`, `memory_summary`, `media_cache_summary`, `runtime_stats`,
`is_available`, `reset_memory_peaks`, and `PreparedPrompt::token_ids`. Engine keeps option
resolution, the zero-output path and the handles.

### Prefix reuse

With the context cache on (the `ninfer-serve` default; `--no-prefix-reuse` turns it off) and a
reusable prompt, `plan_request` computes two 64-bit digests per frontier for
`prefix_shortlist_key`, and the lane's state stays on the device after `finish`, catalogued as one
continuation with two checkpoints:

- the **endpoint** at the executed tokens (prompt and output, without the last token, which the
  forward has not seen): resuming continues from the live state;
- the **prompt anchor** at the end of the prompt: an image of the rings and the convolution tails
  saved on every rank after the last prefill chunk. Restoring it leaves the global K/V in place,
  since positions at or after the anchor are rewritten before they are read.

`inspect_admission` accepts the resident sequence as source only after an exact token comparison
up to the frontier and a frontier shorter than the prompt; the source is consumed
(`ConsumeToActive`). A root admission is feasible only while no continuation holds the state; the
pressure session's one target evicts it, every checkpoint reported lost with a prefill from the
root as its recovery. One image costs the rings plus the tails, about 92 MiB per rank at tp 2,
independent of the conversation's length.

### Capacity

`--max-context` sizes everything at startup: the global K/V of every lane covers `max_context`
positions from admission and never grows, so the capacity curve has one point (the 64-token pages
of `max_context`): an explicit `--kv-capacity` that rounds to another page count fails at
startup. A prompt over `max_context` is refused in preparation with `ContextLengthExceeded`.
`model_max_length` does not bound `--max-context`.

## Frontend

[`frontend/`](../../src/models/inkling2/frontend/) is the library `ninfer_inkling2_frontend`
behind the Engine's frontend contract.

- **Tokenizer**: byte-level BPE from `tokenizer.json` over the o200k pre-tokenization, no
  normalizer, whole pre-tokenized words found in the vocabulary emitted directly, special tokens
  split out first. The loader requires the expected pre-tokenizer text and added-token flags. No
  BOS is added.
- **Template**: the checkpoint's `chat_template.jinja` (or `--chat-template`) rendered by the
  engine's Jinja interpreter. It has no generation prompt: the rendering ends with the last
  message's `<|end_message|>` and the model writes its own `<|message_model|>` header. The
  reasoning effort comes from the prompt options or `chat_template_kwargs`; `enable_thinking=false`
  selects effort `none`, since reasoning cannot be switched off. Developer messages render as
  system messages. Media parts are rejected.
- **Output channels** ([`output_parser.h`](../../src/models/inkling2/frontend/output_parser.h)):
  the model writes messages `<|message_model|>[name]<|content_KIND|>body<|end_message|>` and ends
  its turn with `<|content_model_end_sampling|>`. `content_thinking` is reasoning;
  `content_text`, `content_tool_error` and `content_invoke_tool_text` are content;
  `content_invoke_tool_json` is one tool call `{"name": ..., "args": {...}}`, held until its
  `<|end_message|>` and published as a call, or as content when it is not such an object.
- **Thinking**: a chat prompt starts in reasoning (`starts_in_reasoning`): the model always opens
  its turn with a thinking message, and those header tokens count toward the budget. A thinking
  budget closes the open thinking message with the three control tokens
  `<|end_message|><|message_model|><|content_text|>`; the Program executes them with the next
  decode step (`append_forced_tokens`).
- **Prompt layout** ([`prompt_layout.h`](../../src/models/inkling2/frontend/prompt_layout.h)):
  the token frontiers where a shorter conversation's prompt ends.
- **Sampling defaults** from `generation_config.json`, one preset for thinking and non-thinking;
  an absent `top_k` is 0 (off).

Deliberate differences from Hugging Face: client text (message content, reasoning, tool names and
arguments) is always encoded as text, including strings that spell a special token, which the
Hugging Face tokenizer parses as control tokens; developer messages are accepted as system
messages; `enable_thinking=false` maps to the lowest effort.

## Numerics and validation

### Rounding points

The forward follows the reference's BF16 rounding points: the state between visits is BF16, the
visit's residual `r` FP32; block residuals, messages and gates produce FP32 and the input norms
round once (`rmsnorm_f32`); every projection rounds its output to BF16; norms, clips, convolutions,
router and softmax compute in FP32 and round where the reference rounds; the MoE rounds `g`/`a`,
`h` and `o` and sums the weighted experts in FP32 with one rounding; the residual add rounds once
(`residual_add_round_bf16`). Each Op's header states its rounding points as semantic boundaries.

### Declared deviations from the reference

| Deviation | Where |
|---|---|
| NVFP4 experts at their exact stored weights; Transformers dequantizes them to BF16 on load (`bf16(e2m1 · e4m3 · amax / denominator)`). The divisor is applied in FP32 to each accumulated dot | `moe_bank.h` |
| At tp 2 each rank rounds its attention output partial to BF16 before the all-reduce | `linear_row_parallel` |
| At tp 2, in calls of more than 16 tokens (prefill), each rank rounds its MoE partial to BF16 before a BF16 all-reduce with FP32 accumulation | `forward/moe.h` |
| Attention calls of more than 16 tokens (prefill) round exp(logit − m) to FP16 as the operand of the value product and sum each run of 64 keys in FP16 before the FP32 total, without rounding prob to BF16; calls of up to 16 tokens whose query sees more than 128 keys keep exp(logit − m) unrounded | `relative_sink_attention.h` |
| V crosses FP16 in the cache, for cached and current-block rows alike | `relative_sink_attention.h` |
| The logits are rounded to BF16 by the output head before the soft cap, as the published Transformers does with a BF16 `lm_head`; the `bf16` and `native` references of `tools/ref_ink` keep `lm_head` in FP32 | `forward/head.h` |

The first one makes bit identity with Transformers impossible: the `native` reference
(`tools/ref_ink/native.py`, `--impl native`) is Transformers with the NVFP4 experts at their exact
weights and everything else unchanged, and is the reference for the released model.
`tools/ref_ink/emulate_v00.py` emulates the two-rank attention arithmetic and the FP16 V in the
reference at visit 0.

### Validating a change

1. **Ops**: each new Op has its FP64 oracle test (`tests/ops/test_<op>.cpp`) at small and real
   shapes, single-device and rank shapes ([Op development §6](op-development.md#6-qualification)).
2. **Synthetic model** (`NINFER_TEST_INKLING2_SYNTH_ARTIFACT`):
   `ninfer_inkling2_forward_tp2_test` (KL of tp 2 against tp 1 within the bounds registered in
   [`tests.cmake`](../../tests/models/inkling2/tests.cmake), every replicated value bit-identical
   on the two ranks) and `ninfer_inkling2_forward_blocks_test` (blocks of 16 against token by
   token, bit for bit). The engine's teacher-forcing dump is compared with the FP64 reference of
   the same checkpoint with `tools/ref_q4/tf_compare.py`, against the floor of the BF16 reference.
   The default synthetic checkpoint selects every expert; `synthetic.py --topk 4` exercises the
   selection.
3. **Real model, teacher forcing**: `ninfer_inkling2_tf_driver --mode score` writes a `tfdump/1`
   dump; `tools/ref_ink/ref_dump.py --impl native` writes the reference's, and the same reference
   with `--chunk 16` is its floor (the reference against itself in another valid order).
   `tools/ref_ink/split_compare.py` reports prompt and generated rows apart, per visit and on the
   logits; `route_compare.py` compares the routing.
4. **Statistical criterion on the generated rows** against `native`: (a) mean KL within twice the
   reference's floor; (b) top-1 disagreements no more frequent than twice the rate at which the
   engine disagrees with itself between two valid routes (for example blocks of 64 against
   blocks of 16); (c) no disagreement where the reference's top-1/top-2 margin exceeds the largest
   margin at which the engine changes its own top-1 between two valid routes. Prompt rows are
   judged against the per-row floor: near-tie router decisions amplified by the exact NVFP4
   weights move whole rows there.
5. **Greedy**: `ninfer_inkling2_tf_driver --mode generate` against `tools/ref_ink/greedy.py
   --impl native`; the first divergence must satisfy (c). A divergence beyond it is investigated
   by teacher forcing with the greedy's own structure (`--first-block`, `--dump-step`): if those
   logits equal the greedy's, the cause is route noise, not state.
6. **Engine**: `ninfer_inkling2_prefix_reuse_test` (endpoint and anchor resumes against the same
   requests from the root, and an eviction, on the real artifact at tp 2), the same with the
   drafter (`ninfer_inkling2_prefix_reuse_spec_test`, `--prefill-chunk 1 --spec 3`) and the
   frontend tests.
7. **Speculative decoding** (synthetic artifact with an `mtp` component from
   `tools/ref_ink/synthetic_mtp.py`): `ninfer_inkling2_speculative_round{,_tp2}_test` (fake
   drafts, K = 1, 3, 5, 7), `ninfer_inkling2_features_tp2_test` (the feature sink) and
   `ninfer_inkling2_drafter_round{,_tp2}_test` (the drafter proposing, K = 1, 3, 7): the ids of
   the greedy without drafter and every committed column's logits, bit for bit. The `dspark_draft`
   Ops have their FP64-oracle test. From the server, the text with `--spec mtp` must equal the
   text without it, at temperature 0 and with sampling at the same seed.

A change whose produced values are bit-identical, proven by exact comparison of two dumps
(`tools/ref_ink/bitcmp.py` compares dump trees byte for byte), needs only that proof.

## Where things are

| Path | Responsibility |
|---|---|
| `src/models/inkling2/config.{h,cpp}` | Text config, layer kinds, the visit schedule, `is_inkling2` |
| `weights.h` | logical parameter names and ids |
| `model.{h,cpp}`, `load.{h,cpp}`, `load/{bindings,text}.cpp` | load plan, bindings, materialization, resources |
| `load/sharding.{h,cpp}` | tp 2 placement rules and parent placements |
| `execution/parameters.{h,cpp}` | native operands per rank, stacked attention inputs, expert banks |
| `execution/tp.{h,cpp}` | `shard_text_config` |
| `execution/forward.{h,cpp}`, `execution/forward/*.h` | the forward and its workspace planning |
| `state/visit_state.{h,cpp}` | per-sequence state per rank |
| `program/program.{h,cpp}`, `program/runtime_types.h`, `program/prepared_context.h` | the Program contract types |
| `program/program_impl.{h,cpp}` | lanes, admission, transactions, prefix reuse, commits |
| `program/execution.{h,cpp}` | forward calls, decode graph, anchor image, sampling, scoring, startup layout |
| `program/speculative/round.{h,cpp}`, `state/conv_record.{h,cpp}` | the speculative verification round and its convolution record |
| `draft/config.{h,cpp}`, `draft/parameters.{h,cpp}`, `draft/draft.{h,cpp}`, `load/draft.cpp` | the DSpark drafter: config, per-rank operands, context and proposals, bindings |
| `instance.{h,cpp}`, `engine_family.{h,cpp}` | model instance, startup checks, KV capacity, Engine family |
| `frontend/` | tokenizer, chat template, prompt layout, output parser and session |
| `include/ninfer/ops/{relative_sink_attention,moe_sigmoid_route,moe_bank,short_conv,short_conv_record,depth_attention,message_gate,residual_mix,rmsnorm_f32,residual_f32,softcap,dspark_draft}.h`, `tp2/concat_rows.h` | the Op contracts this family added; implementations in `src/ops/<family>/` |
| `src/ops/linear/bf16/shapes/inkling2.cu` | the family's BF16 shard shapes (generic route's bits) |
| `tools/convert/inkling2.py`, `inkling2_recipes.py` | checkpoint adapter and recipe (`text`, optional `mtp`) |
| `tools/ref_ink/` | references, dumps, comparisons, synthetic checkpoint and drafter, conversion check |
| `tests/models/inkling2/` | config, shard map, loading, frontend, forward driver (also speculative rounds and the drafter), prefix reuse |

## Adding a model family

A family lives under `src/models/<family>/` with its own config, weights, loading, Parameters,
forward, state, Program and frontend; it instantiates the common Engine cores and adds no base
class or registry. What this family needed outside its directory:

- **Build**: `src/models/CMakeLists.txt` includes the family's source lists (loading and runtime
  sources join `ninfer_model_loading`/`ninfer_model_runtime`; the frontend is its own library);
  `src/runtime/CMakeLists.txt` includes the Program/instance/family list after defining
  `ninfer_engine`, whose common cores it instantiates.
- **Tests**: `tests/CMakeLists.txt` includes the family's `tests.cmake`; tests that need no GPU go
  into the CPU list of `tests/cmake/TestLabels.cmake`; artifact paths are cache variables and the
  tests skip (77) without them.
- **Ops**: one `include(...)` per new Op family in `src/ops/CMakeLists.txt`, its tests in
  `tests/ops/tests.cmake`, its benchmark in `bench/ops/benchmarks.cmake`.
- **Engine**: the family check in `Engine::Impl`'s constructor before `runtime::construct_model`,
  and one delegation per public method that reaches the frontend or the core.
- **Converter**: the family's adapter selected in `tools/convert/__main__.py` before the Qwen
  branches.

Two-device execution follows [two-GPU layer §6](two-gpu-layer.md#6-adding-a-new-model-to-tp2):
placement rules with a shard-map test, a config split, and a tp 2 check against tp 1.
