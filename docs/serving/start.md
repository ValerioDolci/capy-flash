# Start the server

See [CUDA synchronization](../cli.md#cuda-synchronization) for the shared `NINFER_CUDA_SYNC` setting.

Start the server with a launcher or a `--profile` ([Launchers and profiles](capy-flash.md#launchers-and-profiles)); this section
explains the options they set.

`--max-context` is each request's logical ceiling. `--kv-capacity` sizes the shared Main Text KV pool that serves
admitted requests; a request may use the full capacity when running alone, and requests run concurrently when their
complete reservations fit. A request's Device KV reservation is a lease: its prompt plus a window of up to
4,096 output tokens (or the prefill chunk, if larger), extended at a decode-round boundary as the
answer grows, up to the request's own `max_tokens`. A client that asks for 64k output tokens and
uses 300 therefore no longer keeps 64k tokens of KV away from retained conversations. When the pool
cannot extend a lease, the KV of idle retained conversations and idle automatic shared prefixes moves
to the Host tier, least valuable first, and without room there they are released; this includes the
prefix pages a conversation shares with the shared prefix it was built on, which neither holds alone.
A conversation is released only for pages the release returns: a page that a kept owner (a shared
prefix with client credit) also holds never qualifies it. Only if that is not enough does the answer
end early with `finish_reason` `length`. The request log reports the leased pages
(`device_*_kv_lease_pages`), the owners released for a lease and the answers cut short.
`--kv-lease-window N` sets the window (default 4096); `--kv-lease-window 0` reserves the whole
`max_tokens` at admission, as earlier releases did.

The Host tiers are pinned RAM that the kernel cannot page out. The server refuses to start when
`--host-kv-mib` plus the Host state slots would leave less than max(4 GiB, 5 % of RAM) available,
and says how many MiB fit (`NINFER_HOST_TIER_RAM_GUARD=0` skips the check).

Qwen3.8-27B artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`, with
`--lm-head-draft` optional. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and each image or video of a request reaches the model as a line of text saying that one
was attached and that the model cannot see it (the request's log line reads `media N omitted (no
Vision)`), so a conversation that holds one keeps working. Add `--vision` when the server must read
image or video input. Anthropic `document` blocks become the same kind of line, and `search_result`
blocks their title, source and text. Speculative residency is likewise
frozen by `--spec mtp|dflash2` and `--draft-tokens`; omitting `--spec` loads no speculative backend, and `--spec off` (after a `--profile`) removes the profile's backend, draft window and proposal head.
`--lm-head-draft` additionally loads the optimized proposal head. DFlash2 on Qwen3.8-27B can be combined
with `--vision`; it accelerates generated-text decode after multimodal prefill, while Vision encode
and prefill remain outside speculative acceleration. A later request cannot enable a capability
omitted at startup. The artifact need only contain the Text backbone and the optional components
selected for this process.
