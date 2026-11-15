# NInfer Resource Scheduling and Context Cache

This document defines the rules by which NInfer selects, verifies and commits context state under
limited Device/Host resources. It is the maintainer authority for admission materialization, prefix
reuse, cache retention and pressure planning.

The problem this document addresses is:

> Once the Scheduler has selected a waiting request, from which recoverable position should it start,
> and which inactive contexts should it retain or degrade, to reach at the lowest predicted cost a
> final state that is physically executable and does not break the completion guarantee of active
> requests?

Top-level request order and lifecycle are defined by [Engine architecture](engine-architecture.md).
The physical contract for KV pages, address spaces, replicas and consumer views is defined by
[Paged KV Context Store](paged-kv-cache.md).

---

## 1. Terminology and boundaries

This document uses the following terms:

| Term | Meaning |
|---|---|
| owner | an active sequence, private continuation or shared prefix |
| checkpoint | the immutable complete state of a recoverable frontier within an owner |
| candidate | a root or exact checkpoint source that the incoming request can adopt |
| planning target | the complete logical final state of the candidate and all affected inactive owners |
| reservation | capacity already deducted from global available capacity and held for future materialization or transaction purposes |
| resource plan | the ordered physical transition the Program seals for one target |
| placement | the Device/Host replica state of one complete StateImage or one KV logical page |

In the rest of this document, planning target is abbreviated to `target`, meaning the final state of
one resource decision.

Prefix reuse is one kind of source selection for admission. It reuses an already computed prefix with
complete identity and state coverage, reducing repeated prefill while preserving model semantics and
request order. Logits or generated tokens are not required to be identical across different prefill
chunkings or numerical paths.

The current product conditions are a single GPU, a single resident model, a fixed
`max_concurrency=1..8` and non-preemptive active requests. Two basic rules follow:

1. The Scheduler first determines the request for this attempt; the resource layer only optimizes
   that request's materialization;
2. Once a request is published as Active, its maximum legal execution range already has a complete
   resource guarantee, and inactive cache policy can no longer borrow that capacity.

---

## 2. Decision ownership

### 2.1 Scheduler

The Scheduler selects:

- the current FIFO head or a backfill request already proven safe;
- when to attempt admission;
- the execution order of prefill, control and decode.

The Scheduler does not read checkpoint placement, victim, page or allocator facts.

### 2.2 ResourceManager

ResourceManager owns the logical cache policy:

- private/shared catalog and SessionIndex;
- candidate shortlist and logical claim;
- retention class, hit history and publication order;
- target search and cost comparison across candidates;
- logical adoption of the Program's final-state results.

ResourceManager may know from the Program's summary that a checkpoint can be recovered from Device,
Host or root, but keeps no copy of physical occupancy, refcounts, free pages or Host arena geometry.

ResourceManager stores its logical protection of an active request as an owner edge, not as a catalog
capability or a mirror of reference counts. A capability generation is a transient structural snapshot
for planning/transaction; an owner edge is a logical lease that lasts until the active request's
terminal settlement. When another reader changes the Device/Host residency of the same owner it may
advance the generation, but this does not make existing owner edges stale. All active logical
reference counts are derived from these edges.

### 2.3 Program

Program is the sole physical authority and owns:

- the real objects, references and replicas of State and KV;
- Device pools, Host stores, allocators and reservations;
- continuation completeness and exact prefix identity;
- Move, Fork, COW, transfer and last-reference release;
- the complete post-state and the physical feasibility of every transition stage;
- the `ResourcePlan` bound to the current resource state.

The call relationship is:

```text
Scheduler selects one request
        │
        ▼
ResourceManager enumerates candidates and compares targets
        │
        ▼
Program projects each complete target from real physical state
        │
        ▼
Program seals and executes one ResourcePlan
        │
        ▼
ResourceManager adopts one stable ResourceResult
```

Logical policy and physical facts interact through candidate assessments, opaque target handles,
`ResourcePlan` and `ResourceResult`, not through a physical ledger shared across layers.

---

## 3. Resource model

### 3.1 Capacity axes

At startup the Program builds a fixed capacity vector:

```text
PhysicalCapacity = {
    Device StateImage slots,
    Host StateImage slots,
    Device page groups for each typed KV pool,
    Host KV arena bytes and allocator geometry,
}
```

Capacity is derived from the loaded model configuration, the actual weight bindings, the enabled
components and the startup options. The physical representation of the weights affects the execution
workspace and the available budget; the mathematical state of KV/GDN and its concrete storage and
capacity are the responsibility of the model implementation and the Program, respectively.

These axes are independent of each other. Headroom on one resource axis cannot compensate for a
shortfall on another. Nor can the total free bytes of Host KV substitute for the allocator's judgment
of whether a concrete extent geometry can be allocated.

The Engine also has fixed logical capacities:

- active lanes;
- private continuation catalog slots;
- shared prefix catalog slots;
- the per-continuation checkpoint/anchor limit.

Lanes and catalog slots do not count toward physical bytes, but a target can be published only if it
satisfies both logical and physical capacity.

### 3.2 Unique physical occupancy

For any physical resource dimension \(r\), the occupancy of state \(S\) is:

\[
Used_r(S)=
\sum_{x\in UniqueAllocations(S)} size_r(x)
+
\sum_{y\in ConcreteReservations(S)} size_r(y)
\]

`UniqueAllocations` includes:

- Device/Host StateImage replicas;
- Device KV page replicas;
- Host KV extents.

`ConcreteReservations` includes:

- future growth of active requests that has not yet been materialized;
- partial-tail COW destination;
- copy destinations and replacement capacity of resource transitions.

A physical allocation referenced by multiple checkpoints or address spaces is counted only once.
Capacity already materialized counts as an allocation; capacity not yet materialized but protected
counts as a reservation; the same unit of capacity cannot be counted in both.

Global physical occupancy and owner-exclusive resources are two different views. The former comes
directly from the unique allocations and concrete reservations of the State/KV stores; the latter only
represents the resources that can be settled when one owner is removed on its own. An allocation
referenced by both a private checkpoint and a shared prefix is counted once in global occupancy, but in
neither owner's exclusive resources. Owner-exclusive resources can be used for eviction effects,
ownership transfer and active entitlement checks, but not to decide whether a checkpoint has a
resident replica; for the latter, only the corresponding physical store can be queried.

Every stable state and every intermediate transition state must satisfy:

\[
Used_r(S)\le Capacity_r
\]

### 3.3 Complete post-state and joint reclaim

Resource reclaim is determined by the references that remain at the end. For a set of logical actions
\(V\):

\[
Reclaim_r(S,V)=
Used_r(S)-Used_r(PostState(S,V))
\]

`PostState` must apply all of the following together:

- the retain or consume of the selected source;
- all victim owner/checkpoint outcomes;
- all surviving Device/Host placements;
- active destination references and reservations;
- the release of zero-reference objects.

Therefore, in general:

\[
Reclaim(S,A\cup B)\ne Reclaim(S,A)+Reclaim(S,B)
\]

Deleting an alias may release no allocation at all; when two owners together delete the last
references, the same allocation is released only once. ResourceManager does not compute or sum
per-owner reclaim credit.

Move, Fork, partial-tail COW and source placement are also determined by the complete post-state. A
source having multiple references in the initial state does not mean it must end up as a Fork;
conversely, once the references needed to retain any checkpoint are kept, it cannot be treated as a
Move either.

### 3.4 Ordered stage peaks

A physical plan is an ordered sequence of stages between stable states:

\[
S_0\rightarrow S_1\rightarrow\cdots\rightarrow S_n
\]

It is executable if and only if:

\[
\forall r,\qquad
\max_{0\le i\le n} Used_r(S_i)\le Capacity_r
\]

The stage order must express the real dependencies:

```text
reserve destination
  -> copy
  -> verify and publish replacement
  -> release old replica
```

State restore, KV H2D/D2H, D2D fork, partial-tail COW and Host extent replacement all follow this rule.
The final net occupancy cannot hide the transient peak of copy-before-release.

### 3.5 Resource revision

Program exposes a monotonic `resource_revision` that binds read-only assessments and sealed plans to
one stable physical state. The following changes advance the revision:

- owner/reference topology changes;
- inactive Device/Host placement changes;
- global free/reserved capacity changes;
- geometry changes that alter subsequent allocator feasibility.

When an active request converts between mapped/reserved within its own reservation, advances its
committed frontier or updates state contents, global available capacity does not change, so the
revision does not advance.

The intermediate stages of an open resource transition do not publish a revision externally. Once it
reaches a stable commit/abort state, the revision advances only once if global physical facts changed.
Handle owner/generation validation independently prevents stale capabilities arising from slot reuse.

---

## 4. Continuation and checkpoint

### 4.1 Complete recovery conditions

The current Qwen3.5 model contains both pageable Full Attention KV and recurrent state that cannot be
rolled back losslessly from an arbitrary later state. A frontier can therefore be reused if and only
if the Program can prove that:

1. a complete StateImage exists for that frontier;
2. Main KV satisfies the typed coverage defined by the target;
3. the KV and fixed state of the selected backend satisfy the same continuation;
4. token, position, Vision and mode identity match the incoming prompt exactly.

When there is only a token match, KV bytes or a page match, what is missing is a complete
continuation; it is not a partial cache hit.

### 4.2 Continuation organization

A private continuation represents a conversation history that can continue to evolve:

```text
PrivateContinuation
├── exact target/token history
├── Main KV address space
├── optional backend KV address space
└── one or more immutable checkpoints
```

A shared prefix is an immutable reuse source that can be Forked by multiple private branches. Multiple
checkpoints within one continuation may reference different prefixes of the same address space, but
each checkpoint has its own complete StateImage identity.

### 4.3 Checkpoint kinds

| Kind | Semantics |
|---|---|
| `SessionEndpoint` | the latest continuable state of a completed request |
| `TurnClosure` | the stable state before the replaceable assistant suffix of the current turn |
| `ResponseReplay` | the stable state before the generation opener, from which the response can be regenerated |
| `LongAnchor` | an earlier long-context recovery point selected by retention policy |
| `SharedStablePrefix` | an immutable stable prefix used in common by multiple histories |

The frontiers of `TurnClosure` and `ResponseReplay` must lie before the assistant suffix that the next
request may replace. This way, when the chat system appends a new user message or rewrites the
generated tail, the stable history can still be reused without mistaking the mutable suffix for part
of the prefix identity.

Once a checkpoint is published:

- its identity, frontier and required coverage do not change;
- the StateImage and the protected KV prefix are no longer modified in place;
- the Device/Host placement can be changed by resource transitions.

### 4.4 Valid and Device-ready

Checkpoint \(c\) is valid if and only if:

- the StateImage has at least one complete Device or Host replica;
- every required KV logical page has at least one Device or Host replica with a consistent content
  epoch and sufficient coverage.

\[
\forall p\in RequiredPages(c),\qquad ValidDevice(p,c)\lor ValidHost(p,c)
\]

A valid checkpoint may need an H2D restore. It is Device-ready only when all execution requirements
are already on the Device and the active future reservation can be obtained.

### 4.5 Exact identity

The Program's exact verification may use:

- token IDs and token types;
- position/MRoPE axes and RoPE delta;
- Vision spans and media digest;
- template/runtime mode;
- checkpoint frontier.

Session keys, markers, hashes and the prefix index only narrow the candidate set; they do not prove a
hit.

`rewrite_execution_frontiers` is also part of the exact identity. It records the exact token frontiers
at which replay/root prefill must be split, so that the rebuild path uses the same execution
decomposition as the original generation path; it must not be ignored in order to adopt an endpoint.
When accepted output generated by NInfer itself forms a boundary that the history renderer can
reconstruct exactly, the ownership chain is fixed as:

```text
Frontend detects the model-output reconstruction boundary
  -> Engine transports relative accepted-prefix metadata
  -> Program atomically commits resident identity and shortlist digest
```

The Frontend publishes the metadata only when the canonical boundary ends exactly at a token frontier,
and makes preview rollback, stop truncation and speculative partial accept obey the same
accepted-prefix transaction. The Program converts the relative position into a resident absolute
frontier and commits it together with the token ledger, State/KV and backend state; it does not scan
the output text. ResourceManager does not infer or relax this fact. This way NInfer output that the
client has not rewritten can continue to adopt the exact endpoint; a client that reorders tool JSON,
fills in default values or rewrites the history still produces a different rendered identity, which
can only match an earlier legal checkpoint.

### 4.6 Request cache participation

Each request has one of the following semantics fixed before it enters the Engine:

- `ReadWrite`: may read exact candidates, capture checkpoints, and publish a continuation on finish;
- `Disabled`: only root materialization is allowed, and all active context resources are released at
  the end.

A `Disabled` request still gets the normal active completion guarantee, but leaves behind no
private/shared catalog entry, SessionIndex binding or Device/Host checkpoint replica. Serve warmup
uses these semantics.

---

## 5. Placement semantics of State and KV

### 5.1 StateImage

The StateImage is the complete unit of migration. With maximum active concurrency \(C\) and \(H\)
additional Device checkpoint slots:

\[
Capacity_{DeviceState}=C+H
\]

The first \(C\) units of capacity form the active guarantee; all slots are still managed by the same
Program pool and are not permanently paired with lanes. Host State is charged in complete images
against its own independent slot capacity.

State transitions have only these semantics:

| Operation | Condition and result |
|---|---|
| InPlace | the sole active writer advances on its own destination |
| Move | the private source is consumed and the original StateImage becomes the active destination |
| Fork | an immutable source, or one that must still be retained, is written into a new private destination |
| Freeze | the committed active StateImage becomes an immutable checkpoint |
| Snapshot/Restore | the complete StateImage is copied between Device and Host |

The Program chooses Move or Fork based on the complete post-state. A Fork source stays pinned until the
destination's first state-writing commit completes; that settlement belongs to the active model
transaction and cannot overlap a new global resource transition.

The active State binding also records which lifetime the read source belongs to. For Root, Move,
Restore and active-capture Fork, the initial read is provided by the current sequence; when a Fork is
created from a retained source, or from a consumed source that still has a surviving checkpoint
reference, the read is retained by the surviving owner/reference graph and the active primary binding
owns only the write destination. This borrowed read takes part in global physical occupancy and is
protected by the source pin, but it cannot be charged again through the primary binding, nor released
on abort/release on the strength of the primary binding alone; if the surviving checkpoint is itself
an optional checkpoint owned exclusively by the active lineage, the read counts toward entitlement
only through that checkpoint's ownership, and whether the source can be released is decided after
that ownership is settled. After Fork settlement both read and write switch to the destination, and
the lifetime belongs to the active sequence again.

### 5.2 Typed KV

Each continuation holds a separate typed address space for Main KV and for the selected backend KV.
Each pool's frontier, capacity and placement are independent.

Resource planning uses only the following semantics:

- required logical pages and committed coverage;
- the Device/Host replicas available for each page;
- immutable references and the unique writer;
- partial-tail COW requirement;
- the effect of restore, demote or release on capacity and machine work.

Concrete page geometry, Host packed extents, COW and block-table publication are defined by
[Paged KV Context Store](paged-kv-cache.md).

After a prefix Fork, the active address can contain both shared and private pages, in the following
order:

```text
immutable full pages | unique mutable full pages | optional unique mutable tail
```

Active capture only adds references to immutable full pages and Freezes uniquely writable full pages
in place; an unaligned tail page is kept for the checkpoint, and only one page is copied to become the
new active writer. An aligned frontier involves no KV copy; later growth materializes a new writer
page. The Program's capture assessment, reservation and commit use the same snapshot-shape analysis,
so resource credit and transfer cost are consistent with the page-ownership conditions that can
actually be committed.

State and KV can be placed independently, for example State on Device, part of Main KV on Device and
the rest of Main KV on Host, with backend KV using yet another coverage. A checkpoint continues to
exist only while every required component keeps at least one valid replica.

---

## 6. Active completion guarantee

### 6.1 Admission reservation

Before a candidate is published as Active, the Program guarantees it:

```text
required Device checkpoint prefix
+ remaining prompt growth
+ maximum effective output growth
+ target-defined provisional/speculative growth
+ State and partial-tail COW destinations
+ selected backend requirements
```

These resources are bound into the returned `SequenceHandle`. Materialization progress only converts
between allocation and reserved-but-unmapped, and never hands capacity to another request.

Active truncate or speculative rollback can remove mappings, but the corresponding capacity still
belongs to that active reservation. Only a terminal release or a resource transition that explicitly
shrinks the active entitlement can return capacity to the global pool.

The active entitlement contains only the active owner's destination, its private growth reservation
and the exclusive optional resources already attributed to that lineage. An immutable StateImage/KV
source borrowed during a Fork cannot be charged again through the primary binding: if it is retained
by another surviving owner it exists only in global physical occupancy, and if it is held exclusively
by an optional checkpoint of the active lineage it is counted only once, through that checkpoint.
Otherwise the same allocation would be charged twice and a legal Fork would be misjudged as exceeding
the active guarantee.

### 6.2 Terminal and capture

A TerminalPending request keeps holding its `SequenceHandle` and full reservation until the Program
completes either:

- **Finish**: publish one complete immutable checkpoint; or
- **Discard**: release the entire active continuation.

If Finish cannot form a legal retained final state, Discard must be used. Optional capture can change
inactive owners through the same pressure graph, but only a complete final state strictly better than
the private-only baseline is committed; otherwise shared publication is skipped, without blocking the
active request and without changing the private capture that was otherwise executable at the same
frontier.

### 6.3 Persistent backfill proof

Whether the Scheduler allows backfill is decided by
[Engine architecture](engine-architecture.md#52-admission-order). The physical proof provided by the
Program uses the same feasibility oracle:

```text
hypothetical state =
    current stable state
    - protected donor active reservations after their terminal release
    + borrower full active reservation
    + head root materialization
    with all unprotected inactive cache releasable
```

Only if this complete state is feasible does the borrower qualify for persistent-safe backfill. The
proof is bound to `resource_revision` and does not use the estimated completion time of the borrower
or the donors.

---

## 7. Materialization planning problem

### 7.1 Candidate

ResourceManager builds a bounded candidate set for the selected request:

- root;
- matching private endpoint;
- typed rewrite checkpoint;
- retained long anchor;
- shared stable prefix.

The same capability hit through different indexes is deduplicated first. The Program performs exact
verification on each shortlist entry and produces an `AdmissionCandidate`. A candidate fixes the
source, destination, prompt and request work, but does not decide Move/Fork, victims or the final
placements in advance.

Root always exists. It reuses no continuation state and represents a normal prefill starting from the
prompt.

### 7.2 Shared publication candidate

Three states of a shared prefix must not be confused:

- a **boundary** is a logical content boundary indicated by the protocol or the Engine;
- a **candidate** is an optional write opportunity that the Frontend has already resolved to an exact
  token frontier;
- an **owner** is an immutable checkpoint already published by the Program, with complete State/KV
  identity and placement.

The external protocol or the C++ `PromptInput` submits at most four explicit markers per request. The
Frontend can additionally generate at most three Engine candidates: after all tools, after the
consecutive leading System/Developer messages, and the full prompt. Identical frontiers are merged, so
each request has at most seven prepared candidates. This fixed limit is not a startup configuration.

The shared catalog is Engine-wide common capacity, not a per-lineage quota. When context cache is
enabled, the default logical capacity covers both the active-concurrency lower bound and the at most
four explicit markers of a single request, i.e.
`max(max_concurrency, kMaximumExplicitPromptCacheMarkers)`; an explicit configuration still fully
overrides the default. This lower bound lets an earlier stable layer and a later rolling marker become
owners at the same time, but whether to capture, retain or replace is still decided only by the
generic portfolio/pressure planning, with no special cases for Claude, compact or token position.

A candidate stores evidence flags. Their policy meaning is:

| Evidence | Admission capability on first appearance |
|---|---|
| `ExplicitBoundary` | may take part in pressure, but still requires a strictly positive net gain |
| `RequestedAutomatic` | may take part in pressure, but still requires a strictly positive net gain |
| `DefaultAutomatic` | may only use a spare final state that does not degrade existing owners |
| `EngineStructural` | may only use a spare final state that does not degrade existing owners |
| `EngineObserved` | can be created only after the same key has been observed by at least two independent reuse domains |

Markers, evidence and shortlist keys do not prove a hit; the Program re-verifies the complete identity
for read, dedup and publication. The order of sources for creating a candidate is: an exact shared
owner is deduplicated directly; a selected exact private reuse base gets a zero-prefill promotion
before the first model mutation; a later frontier is captured by the active prefill. A candidate
earlier than the selected reuse base and without an existing exact source is kept only as a demand
observation; state is not rolled back and prefill is not rerun.

Whether a shared candidate adds a split must be decided before prefill, while the physical state can
only be determined when the frontier is reached. The implementation is therefore split into two
steps:

1. after admission has selected the materialization source, ResourceManager chooses a candidate
   subset using the optimistic upper bound on the gain of the complete logical portfolio and the
   exact split cost returned by the Program; this step reserves no resources and does not promise
   publication;
2. once the capture offer arrives, `SharedCapturePlanner` searches complete physical targets from the
   current `resource_revision`, and may jointly demote/evict multiple legal inactive private/shared
   owners. When the shared catalog is full, the logical replacement and the other physical victims
   belong to the same target and the same transaction.

The actual planner's baseline is the private-only policy at the same frontier. Shared publication is
an optional investment: when there is no feasible target with a strictly positive gain, the private
baseline or Skip is executed.

### 7.3 Target

The planner's search nodes are complete targets, not isolated eviction actions. A target determines
all of the following at once:

```text
selected candidate
selected source retained or consumed-to-active
every eligible owner retained or evicted
every surviving checkpoint and required coverage
every surviving State/KV Device/Host placement
logical publication destination
```

Eligible victims include only inactive owners not protected by an active reference, selected-source
protection or an open transaction pin. Zero-reference objects produced by applying the target are
released automatically.

From this complete final state the Program re-derives:

- Move or Fork;
- State/KV replacement and partial-tail COW;
- last-reference releases;
- active completion reservation;
- Host allocator geometry;
- ordered stage peaks;
- immediate work and future recovery work.

A candidate's `identity target` applies only the incoming activation and the required source
disposition, leaving the checkpoint contents and placements of the other eligible inactive owners
unchanged. It is the root of that candidate's pressure degradation graph.

### 7.4 Monotonic degradation graph

Pressure targets expand from the candidate's identity target in the direction of monotonic
degradation:

- remove a redundant Device or Host replica;
- demote Device State/KV to Host-only;
- shorten checkpoint coverage that is no longer retained;
- delete a checkpoint;
- evict a complete inactive owner.

The Program generates semantic successors from the current residual deficit, checkpoint frontiers,
page/extent boundaries and shared references. The search does not enumerate permutations of physical
operations; the Program maps each target to a unique canonical stage order.

A single owner action that temporarily produces no reclaim still cannot be removed before the joint
post-state, because it may combine with another action to form a last-reference release.

---

## 8. Planning objective and algorithm

In this section, `incumbent` means the best feasible complete target found so far.

### 8.1 Separation of feasibility and value

The Program first decides whether a target is structurally legal, physically feasible, and whether it
can still be degraded further. ResourceManager compares value only among targets that are physically
feasible and whose logical publication can be adopted.

The cost model takes no part in capacity decisions. Any prediction error can only change "which
feasible plan is chosen"; it can never let an infeasible plan through.

### 8.2 Machine cost

The Program folds physical work into:

```text
Immediate =
    ordered transfer phases
  + remaining Text/Vision prefill
  + required State/KV copy work
```

A transfer batch within the same phase and the same direction uses:

\[
Transfer =
\max(batch\_ns+operations\cdot operation\_ns,\ bytes\cdot ns\_per\_byte)
\]

The prefill work of a Text suffix uses:

\[
AttentionPairs = B\,S+\frac{S(S+1)}{2}
\]

where \(B\) is the reused prefix tokens and \(S\) is the remaining suffix tokens. Vision item/patch
work uses separate components of the same startup-resolved machine model.

Transfer coefficients are selected by hardware; prefill coefficients are selected by hardware and by
the `prefill_signature` generated from the actual Text/Vision config, the binding structure and the
Use. The signature does not use checkpoint/release names and contains no weight values. The two sets
of coefficients are resolved independently; when there is no matching calibration, the corresponding
generic coefficients are used. Cost only affects the ranking among feasible plans; it is not a gate
for loading or execution.

### 8.3 Portfolio value and future loss

ResourceManager keeps the 32 most recent requests that were successfully materialized as Active. Each
demand record stores the reuse domain and the prefix keys that the request exact-matched, selected or
could create; it does not store the prompt, media content, raw session text or the API cache key. A
request with an Engine session key uses its stable digest as the domain; each request without a
session key forms its own domain.

For checkpoint \(p\) and complete portfolio state \(S\):

\[
Saving(p,S)=\max(0,\ Rebuild(p)-Recovery(p,S))
\]

`Rebuild` is the canonical prefill cost from root to that frontier. The Program enumerates all
supported restore/copy/interval-prefill recovery recipes from the actual State/KV placement; the
common runner prices them with the same immutable machine cost model of this planning problem and
takes the minimum as `Recovery`. The Program does not receive the cost model, nor does it choose a
recipe on behalf of the policy layer. Each demand takes only the largest saving among all matching
checkpoints:

\[
EmpiricalValue(S)=\sum_q\max_{p\ matches\ q}Saving(p,S)
\]

Nested prefixes such as tools, instructions and full-prompt are therefore not valued more than once
for the same future request. Adding the explicit shared credit that has not yet expired gives the
public value:

\[
PublicValue(S)=EmpiricalValue(S)+SharedCredit(S)
\]

Private retention represents one more unobserved future reuse per owner. When comparing baseline
\(S_b\) with target \(S_t\), checkpoint identity is held fixed: first the saving loss of each
checkpoint is computed, then the maximum is taken within the owner:

\[
PrivateLoss_o(S_b,S_t)=w_o\max_{p\in o}
\max(0,Saving(p,S_b)-Saving(p,S_t))
\]

The `PrivateLoss` of different owners is summed. This neither treats all nested checkpoints of the
same continuation as future requests, nor lets a still-usable final checkpoint mask the loss of an
earlier TurnClosure or long anchor. The weights are:

| Private retention | prior weight |
|---|---:|
| Disposable | 1 |
| RecentPrivate | 4 |
| LiveSession | 16 |

A shared owner has no fixed retention multiplier. `ExplicitBoundary` or `RequestedAutomatic` brings an
owner-scoped credit at publication; it is consumed at the first subsequent exact match, or expires
after 32 successful materializations. Multiple evidence flags at the same frontier do not stack credit.

For the candidate identity state \(I_c\) of a mandatory materialization and a pressure target \(T\):

\[
FutureLoss_c(T)=\max(0,PublicValue(I_c)-PublicValue(T))
+\sum_o PrivateLoss_o(I_c,T),\qquad
J(c,T)=Now(c,T)+FutureLoss_c(T)
\]

For optional shared capture, compare the private-only state \(S_b\) with the complete shared target
\(S_t\):

\[
NetGain=PublicValue(S_t)-PublicValue(S_b)
-\sum_o PrivateLoss_o(S_b,S_t)-ImmediateDelta
\]

The pre-execution candidate subset additionally subtracts the exact `SplitCost`; once the capture
offer arrives the split has already happened and is not charged again. `ImmediateDelta` includes the
transfer/copy cost of the shared target relative to the private baseline, multiplied by the count of
the current capture owner plus the other requests blocked by that global transition. Publication
happens only if `NetGain>0`; when equality, overflow or saturation cannot prove a strictly positive
gain, the baseline is kept.

A legal `ConsumedToActive` of the selected source is an ownership transfer and does not count as an
eviction. Other checkpoints deleted or degraded under pressure enter the future loss through public
value and private transition loss.

### 8.4 Ordering hints are not proofs

For targets that have not been assessed, the Program returns the physical residual, the estimated
number of remaining steps, transfer/recovery work and a stable ordinal. ResourceManager combines
these facts with the portfolio policy and the logical publication conditions into a priority. The
construction goal covers both physical capacity and logical adoptability; a zero physical deficit
cannot replace the publication slot check.

These values only decide the exploration order and, at the optional budget boundary, whether
continuing the search is worthwhile: they cannot mark a target feasible, cannot prove the incumbent
optimal, and cannot take part in physical readiness. Combined actions may cancel copies, so the
predicted cost is not a mathematical lower bound usable for pruning. Only a complete exact assessment
plus the logical-adoption check can produce an adoptable plan.

### 8.5 No-pressure path

The planner must first read all exact candidate identities. Only when the Program states explicitly
that no candidate needs pressure, nor can improve its current materialization work through pressure,
does it directly seal the best adoptable identity. As long as any candidate is still expandable, the
same bounded pressure domain is built; it cannot be skipped because its estimated cost is higher than
the incumbent.

The final shared-capture split is chosen by ResourceManager after the selected candidate has been
determined. The Program validates and consumes `FinalScheduleIntent` within the seal; the returned
`ResourcePlan` has no post-seal mutation API, and the summary seen by the Scheduler is already final.

### 8.6 Correctness incumbent

If no identity target satisfies both the physical and the logical publication conditions, the planner
assesses, outside the optional budget:

```text
root candidate
+ release all unprotected inactive cache
```

The root maximal target is constructed from the candidate-specific eligible domain; the selected
source has no victim cell. It keeps no sacrificable cache value, but guarantees that a failure of the
heuristic, the target budget or the wall budget never misjudges an otherwise runnable request as
blocked. Active requirements, source required coverage, writers and transaction pins always remain in
the Program's requirement union.

If the isolated root exceeds the static contract, the result is permanently infeasible; if the
isolated root is feasible but blocked by the active guarantee, lanes or an open transaction, the
result is temporarily blocked.

### 8.7 Bounded heuristic search

Materialization and shared capture use two typed entrypoints. The materialization incumbent is a
verified identity or root maximal; the shared capture incumbent is Skip, replaced only when exact
`NetGain>0`.

Within one planning problem:

1. ResourceManager mints candidate/owner IDs and keeps the unique mapping to catalog capabilities; the
   Program builds, for each candidate, an eligible victim domain from which the selected source is
   structurally absent.
2. The Program holds the opaque targets and the sliced construction state; the Runtime orders legal
   options by estimated complete J and deficit improvement, balancing value against forming a
   feasible plan quickly, and does not construct State/KV actions.
3. Complete candidates are assessed first; on failure, repair continues based on exact physical and
   logical feedback. Conditions that cannot be proven from total byte counts, such as Host geometry,
   need earlier exact feedback. Ordinary alternatives are retained outside the local construction.
4. The complete joint post-state settles unique objects, aliases, Move/Fork, COW, Host geometry and
   stage peaks. Only results that are physically feasible and whose logical adoption succeeds take
   part in the final J and stable tie-break comparison.
5. When any optional time, work or capacity budget is exhausted, the best verified incumbent is
   returned.

Construction, repair, ordinary expansion, the maximal target and seal share the candidate-specific
domain and the exact evaluator. Cursors, the target arena and borrowed summaries are valid only within
the same planning session/revision; stopping or discarding a batch cannot cause a real resource
mutation. After a search ends, its old state is not resumed across decode/prefill.

The Engine provides an optional time budget shared within an admission boundary; head/backfill
inspections cannot each reset it. The budget takes into account other runnable requests and the time
already spent at this boundary. The planner first searches with a short window, then decides whether
to extend the budget based on the estimated improvement still obtainable and the cost of completion;
incomplete predictions get only limited exploration opportunities, and gains already realized cannot
be used repeatedly to extend the budget. Targets, work and session storage remain independently
bounded.

The identity and root maximal correctness assessments are not cancelled when the optional budget is
exhausted. Time is checked between Program operations, and estimation error can cause a single step
to overrun; the final seal must still complete. The budget therefore affects only cache quality; it
does not change mandatory readiness, nor does it promise that the whole planning call can be
interrupted at any moment.
Shared capture keeps its own independent fixed target budget and Skip incumbent.

### 8.8 Objective function and determinism

For an exact feasible assessment:

```text
J = price(Program machine work)
  + ResourceManager portfolio loss
```

The Program only forms canonical transfer/prefill work and enumerates the supported physical recipes
for checkpoint recovery; ResourceManager provides demand, retention and shared credit; the runner
prices and compares everything uniformly with the immutable machine cost model of the same planning
problem. The cost model never enters the Program API; it can only change the exploration and
selection order of verified targets, and cannot change physical status, readiness, source protection
or target generation.

At equal `J`, the preferences are, in order:

1. less impact on checkpoints with observed hits, keeping recent hits;
2. fewer owner evictions, checkpoint deletions, copy operations and transferred bytes;
3. less remaining Text/Vision prefill;
4. more reused prompt tokens;
5. the current session binding;
6. the stable ordinal of the candidate and target.

Stop reasons only describe the actual boundary: `no_pressure`, `queue_exhausted`, `target_budget`,
`expansion_capacity`, `time_budget`, `work_budget` or `insufficient_expected_gain`.
`time_budget` also covers the case where the next indivisible operation is not expected to fit into
the remaining budget. They make no claim of global optimality over the current target graph or the
real TTFT.

### 8.9 Readiness

There are only four kinds of final result:

| Result | Meaning |
|---|---|
| `PermanentlyInfeasible` | the isolated root completion requirement exceeds the current Engine contract |
| `TemporarilyBlocked` | feasible as a single request, but blocked by the current active/lane/transaction state |
| `NeedsTransfer` | the target is feasible but needs asynchronous transfer or pressure stages |
| `Ready` | the target is already Device-ready and can be published synchronously at the current boundary |

`TemporarilyBlocked` is re-checked only after a change in lanes, FIFO protection, open transactions
or the resource revision. Ordinary decode does not re-run the planner.

---

## 9. Resource transition

### 9.1 Three objects

A final state that needs allocation, pressure or transfer goes through:

```text
ResourcePlan
  -> RunningTransaction
  -> ResourceResult
```

- `ResourcePlan`: the immutable, move-only, revision-bound transition sealed by the Program;
- `RunningTransaction`: the single global resource transition the Program is currently executing;
- `ResourceResult`: the complete stable result, Committed or Aborted.

A finish, discard or inactive release that is already covered by the active reservation and needs no
new allocation/transfer can complete as a zero-stage atomic transition, but it still follows the same
serialization, revision and complete-result rules.

### 9.2 Start

Before the first mutation, the Program re-verifies:

1. the plan's `resource_revision`;
2. source, victim and destination capability generations;
3. the logical publication capacity already reserved by ResourceManager;
4. the current allocator's capacity and geometry for each ordered stage;
5. source/victim dependencies and active protections.

A stale or no-longer-feasible plan has no physical side effects before start. After a successful
start, the source, victims and destinations are claimed/pinned, the plan is consumed linearly, and the
candidate or target can no longer be changed.

Before calling Program start, ResourceManager first creates the single logical transaction record and
marks all source/victim and publication destinations as claimed/reserved. The Program completes its
own preflight before the first physical mutation. When a synchronous start returns Aborted,
ResourceManager rolls back the complete logical claim; after a successful start, the logical record
and the Program transaction coexist until stable result adoption or Engine-wide cleanup.

### 9.3 Progress

A transaction can advance across synchronous copies or CUDA events. Each replacement follows:

```text
destination reserved
  -> copy complete
  -> epoch/coverage verified
  -> replacement published
  -> old replica released when unreferenced
```

Materialization and active capture share one monotonic pressure phase:

```text
HostReleases -> CopyPreparation -> CopiesInFlight? -> CopyPublication -> Committed
```

The phase is the sole authority for the global pressure order. Each owner work item may store
execution facts, such as whether the destination/copy has actually been committed and whether the
mutation has been published, used to form the absolute result; these facts cannot change or bypass
the global phase. Without copies, `CopyPreparation` goes directly to `CopyPublication`. Cancellation
is adopted only at a stable phase boundary; Host releases or victim evictions already published are
reported through the absolute `ResourceResult`.

At most one global resource transition exists at a time. Existing active execution can interleave
with a transfer only when it uses its own existing reservation and does not touch the transaction
mappings.

### 9.4 Commit

Commit publishes:

- the new active sequence or checkpoint;
- selected source disposition;
- committed owner/checkpoint degradation;
- stable State/KV placements;
- transaction pin/reservation cleanup;
- the necessary `resource_revision` change.

`ResourceResult` contains the absolute results ResourceManager needs to adopt the logical final state,
and no delta from which upper layers could rebuild physical occupancy. Each owner claim is bound to an
exact set of `CheckpointRef` deletions, not merely a deletion count; ResourceManager checks item by
item that the final summary equals exactly the original checkpoint set minus that set. A replacement
with the same count but different checkpoints therefore fails before any catalog mutation.
ResourceManager reserved the adoption storage before start, so adoption after commit cannot make any
new capacity decision.

### 9.5 Abort

Cancellation or execution failure before publication can result in a request-local abort, as long as
the Program can still form a stable physical state. Abort guarantees:

- the selected source is still valid;
- the target has no half-Active or half-published checkpoint;
- unused destinations, reservations and pins have been released;
- victim demotion/deletion that has already completed safely is recorded explicitly in
  `ResourceResult`.

Legal pressure degradation completed after a replacement was published does not need to be copied
back to the original placement. ResourceManager adopts the actual final state in the abort result
rather than replaying intermediate receipts.

If the Program can no longer form an explainable stable state, an Engine-wide failure follows.

---

## 10. Retention, session and publication

### 10.1 Retention policy

A private retention class affects portfolio value through the owner prior; it does not provide a fixed
eviction order. A shared owner's value comes from real demand and from explicit credit that has not
yet expired. The planner can combine:

- deleting redundant replicas;
- demoting Device replicas to Host-only;
- deleting checkpoints or suffix coverage that are no longer retained;
- evicting complete inactive owners.

If a placement change would delete the last complete State or required KV coverage of a checkpoint,
the same target must first establish a replacement, or delete that checkpoint at the same time.

Placement changes only at the resource boundaries of admission, capture, finish or explicit inactive
release. Ordinary decode runs no periodic promotion/demotion.

Without a Host StateImage tier (`host_state_slots == 0`; the default with tp 2, enabled by
`--host-state-slots`, `--host-kv-mib` or `--profile 27b` / `--profile flash-next`), private prefill
capture (TurnClosure rewrite, long anchor) plans no pressure and cannot snapshot the checkpoint to
Host. In that case a private capture facing a full Device State pool first reclaims in a fixed order:
only idle private continuations without LiveSession and without an active edge are considered, lowest
retention weight first and, within the same weight, earliest admitted first; they are released one at
a time with re-evaluation after each, until the capture becomes feasible or no candidates remain. Each
release counts as one private eviction. If it is still infeasible, or the reservation subsequently
Aborts, the capture is skipped and the released continuations are not restored. This reclaim does not
use portfolio value; it is not enabled when a Host tier exists.

### 10.2 Session ordering

SessionKey belongs to ResourceManager and only provides candidate lookup and binding. The Program does
not read SessionKey.

Each request that may update SessionIndex obtains a monotonic `publication_order`:

- a new result can replace the current binding only if its order is greater;
- an older request that finishes later becomes anonymous cache or is released according to policy;
- a matching private source can consume-to-active when legal;
- a shared source always stays immutable;
- an abort can only restore a former binding on which it still holds an exact claim.

Capability generation decides whether a handle is still valid; publication order decides which
completed result updates the session. The two have different responsibilities.

### 10.3 Logical publication capacity

ResourceManager must reserve the active/private/shared destination slot before the Program mutation.
Logical adoption after the Program commit no longer grows capacity or searches for slots.

A private Move can reuse the source descriptor; root or Fork needs a destination. Optional capture is
skipped when there is no slot; a terminal Finish without legal publication capacity uses Discard.

---

## 11. Configuration semantics and boundedness

`ContextCacheOptions` is resolved into fixed capacities when the Engine is constructed:

| Configuration axis | Architectural meaning |
|---|---|
| `device_state_slots` | Device checkpoint StateImages that can be retained beyond the active guarantee |
| `host_state_slots` | complete Host StateImages |
| `host_kv_capacity_bytes` | total capacity of the typed packed Host KV arena |
| `max_private_continuations` | private owner/catalog capacity |
| `max_shared_prefixes` | shared immutable owner/catalog capacity |
| `max_long_anchors_per_continuation` | the limit of retained long checkpoints per private history |

All stores, catalogs, Program unique-object scratch and reusable planner frontiers are built to the
resolved limits. The target arena, hash and assessment storage of each pressure planning session
acquire their capacity at a fixed limit when the session starts; target expansion must not exceed that
capacity. The capacity needed for logical claim manifests and adoption must be acquired before the
Program mutation. Each request has a fixed maximum of four external markers and three Frontend
candidates, and the demand window is fixed at 32; these are product semantics and algorithmic bounds,
not tunable capacity axes.
The configuration must satisfy:

- private continuation capacity covers at least all active requests;
- total Device State capacity is `max_concurrency + device_state_slots`;
- Host State and Host KV are charged independently;
- logical counts, address-space counts and storage sizes are representable;
- the Program can honor the minimum active capacity and the selected backend requirements.

When context cache is disabled, root-only semantics apply: no inactive continuation is read or
published and Device/Host checkpoint capacity is zero, but the State/KV reservation needed for active
execution stays unchanged.

Public defaults and startup commands are maintained by `EngineOptions`, [CLI](../cli.md) and
[Serving](../serving.md), and are not duplicated in this architecture document.

---

## 12. Core invariants

1. The Scheduler selects the request first; the resource layer cannot change request order on the
   basis of cache value.
2. ResourceManager is the sole authority for logical cache policy; the Program is the sole authority
   for physical state.
3. Physical occupancy is counted by unique allocations and concrete reservations, not counted
   repeatedly by logical references.
4. Reclaim, Move/Fork and COW must be derived from the complete joint post-state.
5. Every ordered transition stage must satisfy capacity and allocator geometry.
6. A prefix hit must have exact identity, a complete StateImage and all required typed KV coverage.
7. Published checkpoints are immutable; each active continuation has at most one mutable writer.
8. The active completion reservation cannot be borrowed or reclaimed before the terminal resource
   result is adopted.
9. At most one global resource transition exists at any moment, and a sealed plan is bound to
   `resource_revision`.
10. A stale rejection before start has no physical side effects; after start, the candidate or target
    cannot be changed.
11. Program commit/abort both return a complete stable final state; ResourceManager adoption is
    preallocated and cannot fail.
12. Root plus releasing all unprotected inactive cache is the correctness fallback outside the bounded
    search.
13. The planner cost model only ranks feasible targets and takes no part in physical correctness.
14. Session binding accepts only a greater publication order.
15. A retention or capture failure must fall back to skip/release, so that the active lane has a
    finite final state.
16. Shared publication must be strictly better than the private-only baseline; a single demand can
    contribute the maximum saving only once across nested prefixes.
17. Candidate selection is not a resource reservation; the actual capture target must re-prove the
    complete physical final state against the current revision after the frontier is reached.

---

## 13. Implementation locations

| Responsibility | Main location |
|---|---|
| logical catalog, claims, session policy | `src/runtime/engine/context_cache/resource_manager.h` |
| bounded target ledger and common search primitives | `src/runtime/engine/context_cache/resource_search.h` |
| cross-candidate bounded planner | `src/runtime/engine/context_cache/materialization_planner.h` |
| typed shared-capture bounded entrypoint | `src/runtime/engine/context_cache/shared_capture_planner.h` |
| shared/private portfolio value | `src/runtime/engine/context_cache/context_portfolio_value.h` |
| machine cost model | `src/runtime/engine/context_cache/context_cost.cpp` |
| prefill measurement signature | `src/models/qwen3_5/measurement.cpp` |
| common resource summaries | `src/runtime/contract/resources.h` |
| unique-object projection contract | `src/models/qwen3_5/program/planning/resource_projection.h` |
| target domain, projection and transaction | `src/models/qwen3_5/program/planning/`, `src/models/qwen3_5/program/transactions/` |
| State stores | `src/models/qwen3_5/program/storage/state_store.h` |
| logical KV/address spaces | `src/models/qwen3_5/program/storage/kv_store.h` |
| Host KV extents | `src/models/qwen3_5/program/storage/host_kv_store.h` |
| public capacity options | `include/ninfer/types.h` |

The paths locate the current implementation; they do not change the ownership boundaries defined in
this document.
