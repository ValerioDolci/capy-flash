# NInfer Paged KV Context Store

This document defines the physical storage and consumption contract of NInfer growing KV. It is
the maintainer authority for typed KV pools, logical pages, Device/Host replicas, address spaces,
reservations, block tables and GPU consumer views.

This document answers only two questions:

1. whether an already selected logical context target can be fulfilled by the current KV stores;
2. how the fulfilled KV is consumed directly by the model execution unit.

Request order and lifecycle are defined by [Engine architecture](engine-architecture.md); candidate,
retention, pressure target and resource cost are defined by
[Resource scheduling and context cache](resource-scheduling-and-context-cache.md). The KV Store does
not select a request, source or victim.

---

## 1. Physical model

Growing KV uses a set of homogeneous pools fixed at startup. Each pool:

- stores all planes that have the same frontier and lifetime;
- uses a fixed token page size, plane order and page-group count;
- owns an independent physical page-ID namespace and free capacity;
- provides one logical address space for each active sequence;
- is addressed directly by the consumer through a block table.

A request's KV is not required to be physically contiguous, nor is it permanently bound to a control
lane. All active and inactive address spaces share pool capacity; an active request obtains, through a
reservation, completion capacity that will not be used by any other request.

Paged storage covers KV that grows with the context. DFlash2 local cyclic state, Vision/query
temporary K/V and other fixed state have different lifetimes and are managed by their respective
StateImage or workspace contract.

---

## 2. Three independent granularities

The KV architecture distinguishes:

| Granularity | Meaning | Current contract |
|---|---|---|
| allocation granularity | how much token payload a pool acquires or releases at a time | `P=64` for every growing pool |
| valid-frontier granularity | up to which logical position the consumer can read | 1 token |
| reusable-state granularity | which frontier has a complete model continuation | target-defined checkpoint |

A page boundary is neither an Attention mask boundary nor a prefix hit boundary. A valid frontier can
lie at any offset inside a page.

The KV Store can represent, truncate or protect a prefix at any token frontier; this does not prove
that the model can resume from that position. A reusable frontier requires both a complete StateImage
and the target-defined backend state; the detailed rules are in
[Continuation and checkpoint](resource-scheduling-and-context-cache.md#4-continuation-and-checkpoint).

---

## 3. Typed pool set and capacity

### 3.1 Pool set

The model configuration and the selected speculative backend determine the pool set at startup:

```text
ordinary:
    Main Text

MTP:
    Main Text
    MTP

DFlash2:
    Main Text
```

Speculative backends are mutually exclusive within one Engine, so there are currently at most two
growing pools. The DFlash2 draft layers can only be local attention and need only the Main growing
pool; its draft context uses cyclic storage.

| Pool | Content | Logical frontier |
|---|---|---|
| Main Text | target full-attention K/V and its code/scale planes | target materialized KV frontier |
| MTP | MTP persistent K/V and its code/scale planes | MTP KV frontier |
| Draft Full | full-context K/V of the selected draft | draft context frontier |

Main Text and MTP use the BF16, INT8-G64, FP8-E4M3FN-row256, NVFP4-G16 or K8V4 KV profile selected by
the Engine; Draft Full uses its own BF16 profile. The physical layout under the name `BFloat16` is
BF16 K, FP16 V; the writer converts BF16 V to FP16 once. K8V4 is a closed asymmetric profile, not a
runtime bit-width combination: K is fixed to FP8-E4M3FN-row256 and V is fixed to NVFP4-G16.

`PagedKVStorageLayout` resolves the selected closed profile into the K/V data/scale plane schema; the
target planner expands that schema per layer and determines the plane ordinal. The common pool
implementation still receives only the expanded `KVPageGeometry`, plane inventory and capacity, and
does not interpret the storage mode.

### 3.2 Main capacity

Let:

- \(S\): the `max_context` of a single sequence;
- \(C\): `max_concurrency`;
- \(P=64\): the Main page size;
- \(L=\lceil S/P\rceil\): the logical page capacity of a single address space;
- \(M\): the physical page-group count of the Main pool.

The usable range is:

\[
M_{min}=\max(L,C)
\]

\[
M_{max}=C\,L
\]

\(M_{min}\) guarantees both that one request can reach \(S\) and that every active lane can obtain at
least one page. \(M_{max}\) is the physical upper bound when all active requests reach the
per-sequence ceiling at the same time.

The explicit policy of `kv_capacity` resolves to:

\[
M=\left\lceil K_{main}/P\right\rceil
\]

and requires \(K_{main}\ge S\) and \(M\in[M_{min},M_{max}]\).

### 3.3 Automatic capacity

After the weights are loaded, the automatic policy resolves \(M\) once from the currently available
device memory \(F\) and the required headroom \(R\). The model planner provides an affine
`SequenceCapacityCurve` based on the bound parameters and the selected execution domain:

\[
B(M)=B_{min}+(M-M_{min})B_{step}
\]

where \(B(M)\) is the complete runtime Device reservation corresponding to that Main capacity,
including:

- Main and selected backend typed pools;
- active/checkpoint State storage;
- block tables and fixed persistent state;
- unified workspace;
- CUDA Graph allowance.

Automatic selects:

\[
M=
\min\left(
M_{max},
M_{min}+
\left\lfloor
\frac{F-R-B_{min}}{B_{step}}
\right\rfloor
\right)
\]

and requires \(F\ge R+B_{min}\). When \(M_{min}=M_{max}\), that single point is taken directly. The
target uses the same production layout builder to generate \(B_{min}\), \(B_{step}\) and the final
layout, and verifies the capacity curve; the resolver does not replicate the model dimension formulas,
nor does it guess the capacity through allocation probing.

The Main KV capacity finally exposed is \(M\,P\) token-equivalents. Page rounding only adds physical
padding and does not enlarge the per-sequence logical ceiling \(S\).

### 3.4 Backend capacity

The logical page capacity of the selected backend is still \(L\). The physical capacity is derived
from the Main \(M\):

```text
backend off:
    Main physical pages = M

MTP with draft window K:
    Main physical pages = M
    MTP physical pages  = M + C * ceil((K - 1) / P)
```

The extra MTP pages cover only each active row's provisional lead over Main within one speculative
round, and do not enlarge the logical capacity of any address space. Draft Full has no such
provisional lead; draft configurations without a full layer do not allocate this pool.

The pools are physically separate. A free page of one pool cannot become payload of another pool. The
Program builds the complete typed capacity vector once at startup and does not grow or re-partition
pool geometry at runtime.

### 3.5 Host capacity

The Host replicas of Main and the selected backend share one startup-fixed pinned `HostKVArena`, but
each allocation carries its own typed page layout. Host capacity is charged by the actual packed
bytes and the allocator extent geometry; it does not enlarge the Device active entitlement or the
per-sequence context ceiling.

---

## 4. Page group and physical layout

### 4.1 Grouping invariant

Only planes that satisfy all of the following conditions go into the same pool:

1. they use the same logical cache ordinal;
2. they share the same committed frontier;
3. they are reserved, materialized, truncated, retained and released together;
4. they use the same page size;
5. they have no semantics of independent release or independent capacity reuse.

Therefore the K, V, code and scale of the Main layers can share one page-group ID; Main and MTP must
belong to different pools.

### 4.2 Page group

A pool-local page-group ID \(g\) selects, at the same time, the payload of that logical block in all
grouped planes:

```text
page group g
├── layer/plane 0 slice g
├── layer/plane 1 slice g
├── ...
└── layer/plane n slice g
```

Planes have Engine-lifetime-stable backing, but are not required to form one contiguous blob. A page
group is the smallest Device unit of allocation, reservation, reference and transfer.

Physical IDs can be in any order. The allocator may prefer to return contiguous IDs to improve
locality, but correctness, admission and kernel launch topology do not depend on contiguity.
Fixed-size page groups produce no variable-size external fragmentation and need no Device compaction.

### 4.3 Closed Device plane orders

All registered growing pools use \(P=64\) and select one of two closed orders.

Main Text and MTP use page-major:

\[
[X,P,H,N_{physical}]
\]

head-major page run (currently not used by any model pool):

\[
[X,P,N_{physical},H]
\]

where:

- \(X=D\) denotes a K/V or quantized code plane;
- the INT8-G64 scale plane uses \(X=D/64\);
- the FP8-E4M3FN-row256 scale plane uses \(X=1\);
- the NVFP4-G16 code plane uses packed U8 \(X=D/2\), and the scale plane uses U8 \(X=D/16\);
- \(H\) is the KV heads;
- \(N_{physical}\) is the physical page count of that pool.

For logical position \(p\):

\[
b=\lfloor p/P\rfloor,\qquad o=p\bmod P,\qquad g=block\_table[b]
\]

The page-major address is:

\[
address=base+d\,nb_0+o\,nb_1+h\,nb_2+g\,nb_3
\]

The head-major address is:

\[
address=base+d\,nb_0+o\,nb_1+g\,nb_2+h\,nb_3
\]

Code and scale planes use the same \(g\), but each uses the leading coordinate and strides of its own
Tensor. The exact persistent codec is defined by
[`kv_cache_append.h`](../../include/ninfer/ops/kv_cache_append.h), and the consumer arithmetic is
defined by [`softmax_attention.h`](../../include/ninfer/ops/softmax_attention.h); the allocator
interprets only plane bytes, order and page-group identity.

The physical payload per token/head of the D256 Main/MTP profiles is:

| profile | K code + scale | V code + scale | K+V |
|---|---:|---:|---:|
| BF16 | 512 B | 512 B | 1024 B |
| INT8-G64 | 256 B + 8 B | 256 B + 8 B | 528 B |
| FP8-E4M3FN-row256 | 256 B + 2 B | 256 B + 2 B | 516 B |
| NVFP4-G16 | 128 B + 16 B | 128 B + 16 B | 288 B |
| K8V4 | 256 B + 2 B | 128 B + 16 B | 402 B |

The code and scale planes of K/V each have their own dtype, leading extent and group size; they still
share page-group identity, frontier and lifetime. The capacity curve, Device/Host replica,
continuation transfer and memory summary are all computed from this typed plane inventory;
`2 * vector_bytes` cannot be used in place of the asymmetric byte count of K8V4.

### 4.4 Logical position domain

The block table uses the autoregressive cache ordinal:

```text
logical block b covers positions [b*P, (b+1)*P)
```

RoPE, MRoPE, Vision axes and `rope_delta` are Attention input metadata and do not change KV slot
ownership.

### 4.5 Page payload

The logical page-group payload of a pool is:

\[
PageBytes=\sum_{plane} PlaneBytesPerToken\cdot P
\]

Startup physical bytes are obtained from the full span and alignment of each plane slab. `PageBytes`
can differ between pools, but all page groups within one pool are equivalent.

`P=64` simultaneously satisfies the current 32/64-key Attention tiles, 128-token aligned prefill
chunks, limited block-table metadata and bounded tail slack. Prefix hit granularity plays no part in
page-size selection. Changing the page size, grouping or closed plane order is an architectural
change.

---

## 5. Logical page and replicas

### 5.1 Logical page identity

A Device page ID is only the physical location of the current Device replica, not prefix identity.
The Program maintains generation-checked logical pages for each pool:

```text
LogicalKVPage
├── object generation
├── content epoch
├── committed columns [0, P]
├── protected columns
├── address-space references
├── active references and writer state
├── optional Device page-group lease
├── optional Host extent membership
└── transaction pins
```

The canonical content of a logical page is identified by `content epoch + committed columns`.
Speculative or not-yet-committed bytes do not extend the committed coverage.

A replica is valid for the first \(n\) columns required by a checkpoint if and only if:

\[
replica.epoch=page.epoch
\quad\land\quad
replica.coverage\ge n
\]

### 5.2 Device replica

The Device replica uses the consumer-native plane layout of `4`. `DeviceKVPageLease` exclusively owns
one pool-local physical page group; the generation prevents stale handles after release/reuse.

For the same unit of capacity, the pool distinguishes:

```text
allocated page lease
reserved but not materialized page
globally available page
```

\[
allocated+reserved+available=capacity
\]

Materialize turns a reservation into a lease; dematerialize returns the lease to the same reservation.
Neither changes the global capacity available to other requests.

### 5.3 Host replica

The Host replica uses the logical-order packed `HostKVPageLayout`:

- it does not store Device page IDs or block-table holes;
- each page contains the full grouped plane payload of that typed pool;
- a variable-size extent pays only for the actual page count;
- Main/backend layouts can allocate extents with different strides in the same arena.

The Host arena is a bounded variable-size allocator. A plan must verify the extent geometry against
the complete release/allocation recipe; `free_bytes` is only an occupancy summary, not sufficient
proof of allocatability.

### 5.4 Replica transfer

A D2H/H2D transfer copies the complete page payload and can merge adjacent physical IDs into fewer
transfer runs. The publication order of a replacement replica is:

```text
reserve destination
  -> copy page payload
  -> verify epoch and committed coverage
  -> publish replica
  -> release source when no longer required
```

While the copy is in progress, both source and destination are pinned. Before the copy completes, the
destination does not enter the address space or execution table, and the only valid source cannot be
released first.

### 5.5 Descriptor lifetime

A logical descriptor does not constitute a third copy of the payload. It can continue to exist under
Device-only, Host-only or Both placements. Only when references, replicas and transaction pins are all
zero can the descriptor be reclaimed and its generation advanced.

### 5.6 Tensor-parallel mirror

With tp 2, rank 1 holds its own Device page pool and execution tables (planes are allocated for the
halved KV heads), but has no logical page, reservation or address-space bookkeeping; these exist only
on rank 0.

- `DeviceKVPagePool::attach_mirror(mirror, where)`: every `zero_pages`/`copy_page` of the rank 0 pool
  is replayed on the mirror on `where.stream` with the same physical page ID. The mirror must have the
  same page capacity; once attached, Host transfers are rejected, because the Host tier has no rank 1
  replica.
- `KVExecutionTablePool::attach_mirror(mirror, where)`: every row acquire/release and publication is
  replayed on the same row of the mirror. Rank 0 holds the mirror's row lease; rank 1 only reads
  through `mirror_row()` and does not `acquire` on its own; at attach time neither pool may have a
  bound row, and the mirror's lifetime must exceed that of the main pool.

Therefore one set of page/address decisions on rank 0 addresses the KV of both ranks at once, and the
block tables of the two ranks are identical entry by entry.

---

## 6. KV address space

### 6.1 Owning membership

Each continuation holds one `KVAddressSpace` in each enabled pool:

```text
KVAddressSpace
├── ordered logical-block -> LogicalKVPage membership
├── committed frontier
├── checkpoint-protected frontier
├── active/inactive state
├── active growth reservation
└── optional execution-row lease
```

The address space owns the logical order; the block table is only the execution mirror of the active
Device mapping. An inactive checkpoint does not occupy an execution row, nor is it bound to the
original control lane.

### 6.2 Three extents

Each address space distinguishes:

| Extent | Meaning |
|---|---|
| entitlement | the maximum Device page count an active request is guaranteed to obtain |
| membership | the logical pages that already belong to that address space |
| committed frontier | the canonical token prefix the consumer can read |

Membership can be larger than the committed frontier, for example when a prefill chunk or speculative
window is materialized in advance. These bytes become canonical state only after the target commits
the frontier.

Membership and frontier can differ between pools; the Program model schedule defines the semantic
relationship between them, and the KV Store does not derive it on its own.

### 6.3 Active bundle

An active sequence holds:

```text
SequenceKVBundle
├── Main Text KVAddressSpace
└── selected backend KVAddressSpace, when enabled
```

Activation must atomically acquire, for all enabled pools:

- address-space descriptors;
- missing Device replicas;
- future growth reservations;
- partial-tail COW destinations;
- execution rows.

A failure in any pool must not publish a partial Active bundle. An Active sequence cannot temporarily
disable the selected backend per request.

### 6.4 Execution rows

At startup, each pool establishes a fixed-address Device block-table matrix:

\[
block\_tables[N_{logical},C]
\]

where \(N_{logical}=L\) and \(C=max\_concurrency\). Each active address space leases one row; each
entry is an I32 pool-local physical page ID.

Activation publishes the Device page IDs of the address-space membership to the leased row in a
batch. An inactive address space has no row; the next activation of the same continuation can obtain
a different row. An execution row does not own logical pages, reservations or frontiers.

---

## 7. Lifecycle

### 7.1 Activation

When activating from a root, a private source or a shared source:

```text
create or claim destination address spaces
  -> reserve every typed pool
  -> restore required Host-only pages
  -> Move or Fork memberships
  -> create private partial tails when required
  -> lease execution rows
  -> publish complete table mappings
  -> publish Active sequence
```

The ResourceManager selects the logical target, and the Program decides Move/Fork/COW from the actual
references. Before Active publication the source stays valid, and any failure returns to a complete
inactive final state.

### 7.2 Prefill and decode

Before a GPU unit is launched, the target schedule gives the maximum logical position that each pool
may write to in this unit. The KV Store materializes all required pages from the reservation of that
active address space and publishes the table slice.

The argument of `ensure_mapped_to_tokens()` is a lower bound on the coverage this stage requires. If
the existing membership is sufficient it returns immediately, including when the membership is larger
than the required range; it fails only if the required page count exceeds the entitlement. This
operation does not advance the committed frontier, does not trim existing mappings, and does not
change the entitlement. New pages only convert the reservation of the current address space into
allocation; the range can be shortened only by an explicit truncate.

Each stage guarantees only the pools it will write to: target prefill/verify is responsible for Main
KV, and MTP draft for MTP KV. The DFlash2 draft context is written entirely into fixed cyclic state
and needs no paged KV materialization.

Only after the unit succeeds does the Program advance the corresponding committed frontiers. If some
layers have been written but the unit has not formed a valid commit, the new frontier is not visible.

Ordinary decode usually materializes a new Main page only when crossing a page boundary. While the
tail page still has room, no allocator work is needed.

### 7.3 Truncate and rollback

Each pool computes the membership to keep from its own canonical/provisional frontier:

\[
NeededPages_s=
\left\lceil
\frac{\max(CommittedFrontier_s,ProvisionalFrontier_s)}{P}
\right\rceil
\]

Trailing mappings can be removed; the last partial page is kept. A kept tail page whose coverage
already ends at the frontier is not rewritten, so it may remain a full page shared with a checkpoint;
a tail whose coverage shrinks must be a private writer page outside protected coverage. For an
active address space, the removed Device leases return to the same active reservation, not to the
global available capacity.
Stale bytes after the frontier inside a page do not enter the consumer valid domain.

On speculative termination, the recurrent state, hidden and draft context are first brought up to
date according to the final committed count; after the GPU work has completed, the committed frontier
is published, and only then are the uncommitted tail pages trimmed. The verify mappings must not be
trimmed early to satisfy a shorter coverage requirement of some later stage. Terminal settlement
finally releases the active reservation and keeps the checkpoint according to the retention decision.

### 7.4 Deactivation, retain and release

After an Active finish, the address space first leaves execution:

- release the execution row;
- remove the active writer/reference;
- release the growth entitlement that has not yet been materialized;
- depending on the terminal target, keep it as an immutable inactive address space, or release all
  memberships.

Retain can be published only when Main, backend and StateImage form the same complete continuation.
Release removes logical references one by one, and releases physical replicas only when the last
reference disappears.

### 7.5 Stable execution unit

From block-table publication until the GPU unit completes:

- the table entries of the selected rows do not change;
- page payload is not relocated or evicted;
- logical memberships and readable frontiers are frozen for that unit;
- the execution row is not reclaimed;
- the allocator and transfers do not enter the kernel.

Mapping, frontier commit, truncate and row recycling all happen at the GPU boundary.

---

## 8. Sharing, Move and COW

### 8.1 Private continuation

When a private source has no surviving immutable reference in the final post-state, it can Move:

```text
inactive address space -> active address space
```

Move copies no KV payload and does not change logical page identity. The sole writer of a private
tail can continue to append in the unprotected suffix, as long as it does not overwrite the committed
prefix of any surviving checkpoint.

### 8.2 Immutable Fork

A shared source, or a private source that still needs to be kept, uses Fork:

- the full pages before the frontier gain an immutable address-space reference;
- the destination obtains its own State writer and growth reservation;
- shared pages have no writer;
- the source stays valid until the whole activation commits/aborts.

### 8.3 Non-page-aligned frontier

Let \(P=64\) and the checkpoint frontier \(F=1000\):

```text
full immutable pages = 15
partial tail columns = 40
```

A private Move can continue in place from position 1000. An immutable Fork shares the first 15 full
pages and copies the 40 committed columns of the tail into a private destination page; subsequent
appends write only to the private tail.

The checkpoint frontier is not rounded down to 960. Page-size allocation and token-level validity
remain independent.

### 8.4 Writer invariant

The reference count expresses sharing and does not by itself determine write permission. The Program
verifies all of:

- surviving checkpoint protected coverage;
- writer cardinality;
- active references;
- content epoch;
- source/destination pins.

Any logical page has at most one writer at a time. A full page with multiple address-space references
is immutable; writing to a shared partial tail requires a COW first.

---

## 9. Speculative and non-growing KV

### 9.1 Independent pool frontiers

The MTP backend uses the backend pool of the Program KV Store and does not set up an independent
allocator.

Within one speculative unit, Main and backend:

- each materialize from their own active reservation;
- can have different mapped/provisional frontiers;
- each commit their accepted frontier;
- each trim their rejected trailing mappings.

During the MTP draft, the backend mapped extent can temporarily lead Main. The provisional lead
does not constitute committed checkpoint coverage. Rejected bytes can remain in a partial page, but
must be overwritten by a new canonical write before any later read.

### 9.2 Fixed and transient K/V

The following storage does not go into growing pools:

| Resource | Owner |
|---|---|
| DFlash2 local sliding-window K/V | fixed per-sequence StateImage |
| DFlash2 boundary-local snapshot | fixed checkpoint StateImage |
| Vision/query temporary K/V | Program workspace |

DFlash2 cyclic K/V uses its own `CyclicKVCacheLayerView` and modulo/window semantics; it holds no page
ID, block table or growing reservation. The model configuration determines its layer count, heads
and window.

---

## 10. Consumer contract

### 10.1 Single-sequence view

A single-sequence growing-cache Op uses the non-owning `PagedKVLayerView`:

```text
PagedKVLayerView
├── k_pages / v_pages
├── optional k_scale_pages / v_scale_pages
├── block_table          I32 [Nlogical]
├── head_dim
├── num_kv_heads
└── storage
```

The view contains only the plane tensors of one layer and one row of the block table. It contains no
allocator handle, request identity, reservation, ownership or frontier.

### 10.2 Batched view

A batched Op uses `PagedKVBatchLayerView`:

```text
PagedKVBatchLayerView
├── shared layer planes
├── block_tables         I32 [Nlogical, C]
├── head_dim / num_kv_heads
├── storage
└── table_rows[B]        separate Op input
```

`table_rows[b]` selects the corresponding active address-space row for compact row \(b\). Compact
batch order, Engine lane and table row can differ from one another. Per-row context length, valid
columns and positions are provided by the other typed inputs of the Op.

### 10.3 Address translation

Currently \(P=64\):

```text
logical_block = position >> 6
page_offset   = position & 63
physical_page = block_table[logical_block]
```

The element address is then computed with the closed plane order of `4.3`. A batched consumer first
selects the table row with `table_rows[b]` and then performs the same translation.

The wrapper verifies Tensor dtype, geometry, closed strides, table shape and execution envelope. The
caller guarantees that:

- the logical blocks this call may access have been materialized;
- the read domain does not exceed the frozen valid frontier of that sequence;
- all writable K/V code and scale are complete before frontier publication;
- the selected table rows are stable within the GPU unit.

### 10.4 Direct paged execution

Growing-cache Ops consume paged views directly. Page translation is computed and reused at page/tile
granularity; the inner loop does not interpret request, pool kind or allocator state. Kernel
correctness must not depend on adjacent logical pages being mapped to adjacent physical IDs.

Paging introduces no gather-to-contiguous cache and no staging copy proportional to the context
length. Route-specific tile, split, warp and shared-memory schemes can be optimized independently, as
long as they keep the same logical Attention, persistent codec and the address contract above. The
numerical and performance admission rules for Ops are in [Op development](op-development.md).

The only exception is the opt-in [Active KV in RAM](active-kv-in-ram.md) (`--kv-ram-active`): the K/V
code/scale and QSA record planes of the Main Text pool are marked `KVPlaneGeometry::host` and reside
in pinned, device-mapped Host memory; before QSA attention, staging Ops copy the rows read by this
call into a Program-owned device staging region (decode: selected blocks via a per-layer block cache;
prefill: the visible pages), and the bytes read by attention and their order are unchanged.
Zero/copy/Host exchange of Host planes is done by a stream-ordered kernel in core; the pool's write
listener invalidates the staging cache.

---

## 11. CUDA Graph and table publication

Plane bases and the block-table matrix base are stable for the Engine lifetime. What changes across
replays is:

- table content;
- `table_rows` selectors;
- positions, context lengths and valid counts;
- model state selectors.

Page IDs, request identity and physical contiguity do not enter the graph key.

Before an execution unit that needs new mappings, the Program:

```text
materialize all required pages
  -> update host-side membership
  -> publish one contiguous table slice
  -> launch/replay consumer on the ordered stream
```

Batched publication is used when several pages are added at the same boundary or when a complete
membership is reactivated. The mapping update must precede the consumer, and the same row must not be
rewritten while a replay is in flight.

---

## 12. Core invariants

1. Each Device page-group lease carries at most one logical page replica in its pool.
2. A pool combines only planes that share frontier, lifetime, page size and allocation semantics.
3. The K/V/code/scale planes of a pool use the same page-group ID for the same logical block.
4. Device occupancy is counted by allocated leases and reservations; logical aliases are not charged
   twice.
5. Logical page identity, content epoch and physical page ID are independent of one another.
6. The valid frontier is exact to the token; page boundaries do not change Attention or checkpoint
   semantics.
7. Coverage required by a published checkpoint cannot be overwritten by a writer; any logical page has
   at most one writer.
8. Shared full pages are immutable; a non-aligned writable tail first establishes a private COW page.
9. A Host/Device replacement is published only after the copy and the epoch/coverage verification
   have completed.
10. Active entitlement does not enter the global available capacity before terminal release.
11. Within one GPU execution unit, membership, block tables, replicas and read frontier are stable.
12. An inactive address space does not occupy an execution row; an execution row does not own logical
    pages.
13. Main and backend pools reserve, materialize, commit and truncate separately.
14. A growing-cache consumer accesses KV only through the paged view and block table, and does not
    obtain allocator or ownership authority.
15. Kernel correctness does not depend on physical page ID contiguity, nor does it build
    request-contiguous KV through gather (except for the opt-in Active KV in RAM staging, see 10.4).
16. Checkpoint reusability is proven by a complete target continuation; the existence of a KV page
    does not by itself constitute a hit.

---

## 13. Implementation locations

| Responsibility | Main location |
|---|---|
| Device page pools, reservations and execution tables | `src/core/paged_kv_cache.*` |
| closed K/V data/scale plane schema | `src/core/paged_kv_storage.h` |
| Host packed page layout and arena | `src/core/host_kv_arena.*` |
| logical pages, references and address spaces | `src/models/qwen3_5/program/storage/kv_store.h` |
| Host extent membership | `src/models/qwen3_5/program/storage/host_kv_store.h` |
| Program-level KV transition | `src/models/qwen3_5/program/transactions/` |
| model pool layout and capacity curve | `src/models/qwen3_5/program/planning/startup.cpp` |
| public paged consumer views | `src/core/paged_kv_cache.h` |
| growing-cache Ops | `include/ninfer/ops/`, `src/ops/` |
| stream-ordered raw copy of Host planes | `src/core/uva_copy.*` |
| QSA staging region (Active KV in RAM) | `src/models/qwen3_5/state/decoder_state.*`, `src/ops/sparse_attention/staging.cu` |

Exact model state and backend mathematics are described in
[Qwen3.5 model](qwen3_5-model.md) and [DFlash2](dflash.md); the persistent KV codec and the causal
consumer numerical contract are defined by the growing-cache Ops in the table above. The paths serve
to locate the current implementation; they do not promote the files or class names themselves to
external interfaces.
