# capy-flash documentation

Start with the [project README](../README.md): which model fits your machine, `install.sh`, the launchers and the
benchmarks.

## capy-flash guides

| Document | Purpose |
|---|---|
| [capy-flash serving](serving/capy-flash.md) | launchers, the four profiles in full, which `install.sh --model` runs which, RAM per GPU count, chat template, CPU expert threads, client setup (Claude Code, Codex CLI, OpenCode) |
| [Environment variables](capy-flash/diagnostics.md) | production settings, A/B switches, dumps and validation aids of the engine |
| [Release tarball](../tools/release/README-release.md) | contents of the prebuilt release, requirements, launcher commands |
| [ninfer-tp2 README](https://github.com/ValerioDolci/capy-flash/blob/v0.4.9/README.md) | the historical README of ninfer-tp2 v0.4.x, at its last tag (not current usage) |

## User guides

| Document | Purpose |
|---|---|
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, tool calls and every `ninfer-serve` option |
| [CLI](cli.md) | the one-request `ninfer` CLI (text, chat history, image/video input, sampling, MTP, two GPUs); Flash-Next is served with `ninfer-serve` |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Measurements

| Document | Purpose |
|---|---|
| [Speed report, v0.16.6](reports/capy-flash-v0.16.6-speed-report.pdf) (PDF) | every speed measurement of v0.16.6 on one workstation (2× RTX 5070 Ti), by engine part: bandwidth ceilings and how close each model gets, drafters at one request and under concurrency, prefill and long context, memory; [interactive guide](https://valeriodolci.github.io/capy-flash/) |

## Model artifacts

The artifacts `install.sh` and the launchers use, with what each needs: [Pick a model](../README.md#pick-a-model) and
[Model weights](../README.md#model-weights) in the project README.

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [Qwen4Exp model](maintainer/qwen4-exp-model.md) | Qwen3.8-Flash-Next mathematics (hyper-connections, QSA, PLE, MTP, the dense FFN), its artifact names and loading residency, and the one Text forward every model runs (Qwen3.8-27B included) |
| [Inkling 2 model](maintainer/inkling2-model.md) | Inkling 2 Mini mathematics (48 visits, block attention residuals, cross-pass messages, relative-sink attention, sigmoid-routed MoE), its artifact, two-rank placement, per-sequence state, forward, DSpark and DFlash2 drafters, `--spec auto` and speculative decoding, Program and frontend |
| [Tuning `--spec auto`](maintainer/inkling2-auto-tuning.md) | Inkling 2's choice between its two drafters: what adapts by itself, what to retune, `tools/inkling2_tune` (`tune.sh`, `check.sh`) |
| [DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | candidate selection, retention, materialization and Device/Host checkpoint policy |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [Active KV in RAM](maintainer/active-kv-in-ram.md) | `--kv-ram-active`: Flash-Next text K/V in pinned host memory, the QSA staging region and block cache, invalidation, cost |
| [Token embedding in host memory](maintainer/embedding-host.md) | `--embedding-host`: the HostMapped placement, its mapped pinned allocation, CUDA Graph rules, cost and reporting |
| [Two-GPU tensor parallelism](maintainer/tensor-parallel.md) | `--tp 2` weight sharding, rank roles, all-reduce transports, two-device CUDA Graphs, mirrored state, tests and limits |
| [Two-GPU layer](maintainer/two-gpu-layer.md) | where the two-GPU layer lives, the hooks it keeps in the single-device files, the mandatory gate, adding a GPU or a model |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
