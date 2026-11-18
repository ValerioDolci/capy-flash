# The two-GPU layer: layout, hooks and gate

The two-GPU tensor parallel layer (`--tp 2`, [Two-GPU tensor parallelism](tensor-parallel.md)) sits on
top of the single-device engine. This guide says where that layer lives, which lines it still needs
inside the single-device files, and which gate a change must pass.

The rule the layout follows: **two-GPU code lives in its own files; the single-device files carry
only hooks** (an include, a registration line, one call).

## 1. Where the two-GPU code lives

| Path | What |
|---|---|
| `src/models/qwen3_5/execution/tp.{h,cpp}`, `execution/tp2/` | `TpExecution`, TextContext's tp 2 bindings and split MTP head (`text_tp2.inc`), the split attention/GDN/FFN/MTP forms (`*_split.{h,cpp}`), split projection helpers and workspace recipes; `split.h` includes them all |
| `src/models/qwen3_5/program/tp2/` | ProgramImpl's rank 1 runtime, mirrors, mailbox probe and step-down (`program_impl_tp2.inc`), the tp2 workspace plan and option checks (`startup_tp2.inc`), MTP bridge, split verification and split prefill head |
| `src/models/qwen3_5/load/sharding.{h,cpp}` | the placement rules (`shard_rule`) |
| `include/ninfer/ops/tp2/` | declarations of the column/row-parallel Op forms, defined after each Op's single-device form (`src/ops/wrapper/*.cpp`, `src/ops/linear/linear.cpp`) |
| `src/ops/{attn_input_proj,gdn_input_proj,linear_swiglu,context_kv_materialize}/tp2/` | shard and half problems built from the parent Ops' own launcher sources (§2.4), including the DFlash2 drafter's Q8 QKV shard and its four-KV-head context materialization |
| `src/ops/{rmsnorm_rope,softmax_attention/sliding_window}/tp2/`, the tp2 sections of `linear_topk.cpp`, `sliding_window_attention.cpp`, `sliding_window/launch.cu`, `q8_dynamic_grouped_conv_add_materialized.cu` | the split DFlash2 drafter's Op forms: the vocabulary-split top sixteen, the 16/4 head blocks, the convolution finish |
| `src/ops/linear/*/shapes/n*` (half shapes) | the half problems' registrations |
| `src/ops/common/{allreduce,peer_mailbox}.cu`, `src/ops/kernel/peer_exchange.cuh`, `src/ops/launcher/{argmax_split,concat_rows}.*`, `include/ninfer/ops/{allreduce,peer_mailbox}.h` | transport and gathers |
| `src/core/tp2/device_tuning.h` | the per-GPU table (§5) |
| `src/core/tp2/` | dual-device CUDA Graph capture (`decode_graph_peer.h`, `decode_graph_tp2.inc`), KV mirrors (`paged_kv_cache_tp2.inc`) |
| `src/runtime/engine/tp2/` | tp 2 Engine option checks |
| `tools/tp2/` | `gate.sh` (§4), `mailbox_probe.cu` |
| `tests/ops/test_*_split.cpp`, `test_allreduce.cpp`, `test_attention_headlocal.cpp`, `tests/models/qwen3_5/test_*tp2*.cpp` | two-device tests |

## 2. The four ways the two-GPU code attaches to the single-device code

### 2.1 Own translation units

Code that needs nothing private from the single-device code is an ordinary `.cpp`/`.cu` of its own
(`execution/tp2/*_split.cpp`, `allreduce.cu`, ...). Sources are registered by their own
`sources.cmake` files, each included with one line from the list they extend:
`src/models/qwen3_5/execution_sources.cmake` → `execution/tp2/sources.cmake`. The tp2 Op sources
are registered in their family's `sources.cmake`.

### 2.2 Translation-unit fragments (`tp2/*.inc`, included at the end of a single-device `.cpp`)

When two-GPU code needs a single-device file's private helpers (its anonymous namespace), it lives
in a fragment that the file includes **once, as its last line, after its closing namespace**:

```cpp
} // namespace ninfer::models::qwen3_5::execution

#include "models/qwen3_5/execution/tp2/text_tp2.inc"
```

The fragment includes what it needs, reopens the namespace and defines its functions; it sees the
including file's helpers because it is the same translation unit. Nothing is copied, so a fix to a
shared helper reaches the two-device code too, and the compiled code is the one of a single file.
Where a single-device function calls into the fragment before the fragment is seen, a one-line
forward declaration stays in the file's anonymous namespace (e.g. `project_split_output_head` in
`program/prefill.cpp`), and an inline tp2 block of a single-device function is one call
(`construct_tensor_parallel()` and `bind_tensor_parallel_execution()` in ProgramImpl's constructor,
`validate_tensor_parallel_options()` in two places).

### 2.3 Class-body fragments (`*.inc` included inside a single-device class)

`TextContext` and `ProgramImpl` need tp2 members. Their declarations live in
`execution/tp2/text_context_{public,private}.inc` and `program/tp2/program_impl_{public,private}.inc`,
included inside the class body by one line each. `program_impl_public.inc` sits right after
`workspace_plan`: rank 1's storage and the mailbox must be declared before rank 0's storage and the
graph families so they are destroyed after them. **Do not move that include.** The other fragments
can be anywhere in their section.

### 2.4 Shards compiled from the parent Ops' own sources (`src/ops/*/tp2/*_shard_*.cu`)

A shard (`attn_input_proj` [7168,5120], `gdn_input_proj` [8192,5120]) or half (`linear_swiglu`
[17408,5120]) is the parent's problem with fewer output rows and the same K, so it must take the
parent's routes, schedules and token cutoffs. The parent's own sources serve it:

- Where every launcher of the parent already reads its output rows from the weight and its
  template arguments depend only on K, the half needs nothing of its own: the parent's one plan
  serves it, and the wrapper registers the half's rows next to the parent's. NVFP4
  `linear_swiglu` works this way for [17408,5120] and for the dense FFN-14336 gate|up
  [28672,5120] and its half [14336,5120]; so do the FP8 `linear_swiglu` plan, matrix and A8
  launchers.
- Where a kernel takes the rows at compile time, the parent's launcher source is a template on
  the output (the section output type of `gdn_input_proj` and `attn_input_proj`, the geometry of
  the FP8 `linear_swiglu` decode and small-T kernels) and defines the shard's entry points next to
  the parent's. The shard's output types are declared next to the parent's
  (`*_output.cuh`), so the kernels keep their names. The parent's plan runs both problems with each
  one's launchers (a table per problem, e.g. `fp8_attn_input_shard_dispatch` in
  `fp8/fp8_attn_input_plan.cpp`; for FP8 `linear_swiglu` the wrapper passes the half's A16
  launchers, `tp2/linear_swiglu_half.h`). The Q8 DFlash2 drafter's [3072,5120] shard works the
  same way (`q8/q8_dflash2_attn_input.cu`).
- One exception to "next to the parent's": the NVFP4 a16 launchers of `gdn_input_proj` and
  `attn_input_proj` are header templates (`nvfp4_{gdn,attn}_input_a16.cuh`) and the shard's
  instance is compiled in its own unit (`tp2/nvfp4_*_shard_a16.cu`). With both instances in one
  unit the compiler generates the second one's A16 MMA kernels differently (same names and
  template arguments, other machine code); one instance per unit keeps the code of each.

What is genuinely the shard's own stays explicit: the FP8 split-K partial reservations
(`*_shard_partial_capacity_bytes`, next to the parent's in the A8 launcher sources) and the NVFP4
attention shard's TMA scale tile at T=1024 (`nvfp4_attn_input_shard_a4.cu`, via
`tp2::DeviceTuning`). A change to a parent launcher reaches the shard by construction; a change
that must not (a row count, not K) shows in the split tests (`*_split_test`).

One shard is still the parent's source compiled a second time:
`context_kv_materialize/tp2/materialize_shard_h4.cu` includes `materialize.cu` after
`materialize_h4_names.h`, which sets the source's overridable KV-head count to 4 and renames the
entry point. There the head count is a namespace-scope constant read by the kernels, not a template
argument; making it one would rename the kernels. `tools/tp2/check_shard_sources.py` (the gate's
`shards` stage) guards this form: it fails on a template, inline or kernel definition outside an
anonymous namespace in the re-included source (the parent's and the shard's bodies would share one
name and the linker would keep one) and on a renamed identifier appearing in a header the shard
unit does not include first.

`linear_add` does not use this pattern: its half problems differ from their parents in K, and
the parent selects the schedule family by comparing K with 6144/17408, so the halves need the
`output_family(K)` hooks in the parent's launchers (§3).

## 3. What still lives in single-device files

The two-GPU code that stays inside the single-device files is of these kinds; each file's part is
small unless noted.

| Kind | Where | Why it cannot move |
|---|---|---|
| Hooks: one include, one call, one registration line | `text.cpp/h`, `program_impl.cpp/h`, `startup.cpp`, `prefill.cpp`, `speculative/{mtp,target_verification}.cpp`, `decode_graph.{h,cpp}`, `paged_kv_cache.cpp`, `model_instance.cpp`, `basic_sources.cmake`, `execution_sources.cmake`, `linear/*/shapes.h` and `sources.cmake` | by design |
| Two-device branches inside single-device functions | `program/{graphs,prefill,decode,storage/context,transactions/*}.cpp`, `speculative/mtp.cpp` round body, `execution/{vision,draft,parameters}.cpp`, `startup.cpp` (per-rank layouts, the shared workspace helpers) | they read and write the function's locals at many points; `prepare_graphs()` is the largest (retry loop and mailbox step-down) |
| Shape registrations inside the Ops | `linear_add` (FP8/NVFP4 `output_family(K)`, half cases), `linear_swiglu` wrapper (half shape, routing), `causal_conv1d` (split geometries), `gdn_projected_conv.cu` (5120-channel geometry), `mtp_pack` (12/2 layout), `gdn_gating_proj` (24-head shard kernels), GDN replay and recurrence (8/24 heads), softmax attention (12/2 instances); for Qwen3.8-Flash-Next: GDN replay fold (36 layers, full and 8/24 shard), `gdn_gating` (24-head shard), the sigmoid epilogue of `rmsnorm` (`gated_rmsnorm_sigmoid`) and the public `gdn_projected_conv_{snapshot,record}` wrappers | the Op selects on the shape inside its own code |
| Engine-wide, public or product surface | `include/ninfer/types.h`, `apps/*/options.cpp`, `serve_options.cpp`, `serve/operational_log.cpp` (`log_tensor_parallel`), `kv_capacity.cpp`, `engine.cpp`, `resource_manager.h` (Host-less reclaim), artifact `binder`/`materializer`/`views` (multi-device materialization) | public API, CLI and loading paths |
| Head counts and geometries the two-device drafter instantiates | `softmax_attention/common/context_query.cuh` and `sliding_window/kernel.cuh` (query/KV head counts as template parameters, defaulting to 32/8), `context_kv_materialize/materialize.cu` (the KV-head count as one named, overridable constant), `weight_input.cpp` (the drafter's `[2048|512|512,5120]` QKV shard), `state/state_image.cpp` (the mirror replays DFlash rings when the mirror has them) | the kernels hard-coded one geometry; parameterizing keeps one source for both |
| Generic changes | `FuncAttrPerDevice` at every `cudaFuncSetAttribute` site, device SM count and balanced splits in the INT8 attention plan and kernels, A16 sliced-K with several row tiles per CTA (`nvfp4_a16_sliced_k_mma.cuh`), explicit rejection of unregistered head counts, per-step KV row publication (`publish_kv_rows`) | useful at one GPU too |

Tests: the tp2 cases of single-device test files (`test_decode_graph.cpp`, `test_kv_cache.cpp`,
`test_serve_options.cpp`, `test_cli_options.cpp`, ...) are inline.

## 4. The gate (mandatory for every tp2 change)

```bash
# once per intended behavior change: record the reference from the known-good build
GATE_PROMPTS=prompts60.jsonl tools/tp2/gate.sh --record <good-build> <ref-dir>
# every candidate, under a GPU lease on the workstation
tools/tp2/gate.sh [--attention] [--stages ...] <build> <ref-dir>
```

Stages: the unit/Op/split ctest set with the tp2 real tests (`--attention` adds the 12-minute
softmax attention suite, needed when `softmax_attention/` changes); the tp1 golden on the
synthetic model; `ninfer-perplexity --tp 2` at 65536/32768 and 4096/2048 with INT8 KV (every
printed digit must equal the reference; for a commit that changes bits on purpose,
`GATE_PPL_PAIRED=1` judges a changed table by a paired per-window test instead, PASS when not
significantly worse and within 0.5 %); the 60 greedy prompts through `ninfer-serve` with the
production flags (texts must be identical; ms/round is reported against the reference, ±1 % is
noise on the workstation, beyond that rerun alternated A/B before deciding); a short DFlash2 K=7 run.
About 13 minutes. `tools/tp2/README.md` has the paths and options.

## 5. Adding a new GPU

The per-GPU values live in one table, `kDeviceTuningRows` in
[`src/core/tp2/device_tuning.h`](../../src/core/tp2/device_tuning.h), matched by compute capability
and SM count (first match wins). Its last row matches every device and carries the values measured
on two RTX 5070 Ti. For a new board:

1. Build and run the gate on the new pair with the current table (`tools/tp2/gate.sh --record` on
   the new hardware gives its reference; the texts will differ from another board's).
2. Measure the fields that are per-GPU (the table documents each one's procedure): the CUDA Graph
   allowances (the server logs `cuda graphs | rank N: observed ... allowance ...`; take
   max(3 x observed, 8 MiB)), the NVFP4 attention shard's TMA tile bound (`bench/ops` on the
   [7168,5120] shard at T=1024/1025), and whether the attention plan should size its waves for
   another SM count than the device's.
3. Add a row above the catch-all with the new values and a `source` naming the board, clocks,
   date and commit; nothing else is edited.
4. For schedules and crossovers of the shard and half problems (inherited from their parents),
   follow the procedure of [Two-GPU tensor parallelism](tensor-parallel.md#11-known-limits-and-not-done-items)
   (Schedules and thresholds are per GPU); a retuned value becomes a table field read by the
   `src/ops/*/tp2/` shard files, never an edit of the parent launchers.
5. Run the full gate, with `--attention` if the attention value changed, and GSM8K on a change
   that is not bit-identical (perplexity at 1024-token chunks does not see decode-width routes).

The header of `device_tuning.h` lists the per-GPU choices that are still compile-time inside
the single-device Op code (shared by every device) and would have to become table fields first.

## 6. Adding a new model to tp2

Two-GPU execution exists for the dense Qwen3.5 family (Qwen3.6/3.8 artifacts). What is generic and
what a new model family implements:

| Generic (reuse as is) | Model-specific today (Qwen3.5), to implement for a new family |
|---|---|
| `ExecutionContext`, peer access, `--tp/--devices` options, symmetric KV capacity (`runtime/engine`) | placement rules per logical parameter: `shard_rule` in [`load/sharding.cpp`](../../src/models/qwen3_5/load/sharding.cpp) (Rows/Columns/Replicated/PrimaryOnly/SingleDevice) |
| sharded materialization (`artifact::Binder` shard resolver, `tensor_slice`, `materialize` on an `ExecutionContext`) | the config split: `shard_text_config` in [`execution/tp.cpp`](../../src/models/qwen3_5/execution/tp.cpp) (which heads, channels and widths halve) |
| transport: `allreduce_sum`, `PeerEvents`, `PeerMailbox`, `concat_rows`, split argmax | the split schedule: `execution/tp2/` (the `*_split` forms the Text forward and the MTP head use: which projection is column- or row-parallel, where the two all-reduces of a layer sit) |
| split Op forms (`include/ninfer/ops/tp2/`): column/row-parallel `linear`, `linear_add`, `linear_swiglu`, `attn_input_proj`, `gdn_input_proj`, `gdn_gating_proj` | the shard shapes of the new model's dimensions: registered in each Op like any problem (`linear` shapes files; the `tp2/*_shard_*` sets of the fused Ops, one more set per new shard shape) |
| dual-device CUDA Graph capture (`core/tp2`), KV and StateImage mirrors | the Program integration: rank 1's runtime and mirrors (`program/tp2/program_impl_tp2.inc`), the rounds' peer ingress, `TpExecution` |
| `tp2::DeviceTuning`, `tools/tp2/gate.sh` | per-rank persistent layouts and the tp2 workspace plan (`program/tp2/startup_tp2.inc`), the tp2 checks of the planner and the Engine |

The order that worked for Qwen3.5: shard rules and config split with a shard-map test; the split
Op forms at the new shapes with `*_split_test` against the single-device Op; a synthetic two-layer
model whose tp 2 logits match tp 1 within two BF16 ulps (`test_text_context_tp2.cpp`); the real
artifact tests; then the gate recorded for the new model (its own prompts and perplexity corpus).

A drafter on two devices (the tensor-parallel DFlash2 drafter) is the same work at a smaller scale:
its placement rules in `shard_rule` (today PrimaryOnly), its Q8 shard shapes as more
`src/ops/*/tp2/` sets compiled from the parent Q8 sources, the split drafter round in
`execution/tp2/` and `program/tp2/` with one call from the single-device DFlash2 round, and its state
in the mirrors.
