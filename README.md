
<h1 align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/capy-flash-lockup-dark.svg">
  <img alt="capy-flash" src="docs/logo/capy-flash-lockup.svg" width="560">
</picture></h1>

**Qwen3.8-Flash-Next (125B MoE) and Qwen3.8-27B on one or two 16 GB RTX 50-series GPUs, built for agents.**

capy-flash is an inference server. It runs two models on consumer GPUs with 16 GB of VRAM and exposes an OpenAI- and
Anthropic-compatible HTTP API, so agent tools such as Claude Code, Codex CLI and OpenCode can use it like a hosted model.

- **Qwen3.8-Flash-Next** is a mixture-of-experts model too large for the GPUs. capy-flash keeps the experts that do
  not fit in VRAM in host RAM and computes them on the CPU; everything else runs on the GPU (split over two GPUs when
  there are two). That is why Flash-Next needs a lot of RAM and a CPU with AVX2 or AVX-512.
- **Qwen3.8-27B** is a dense model that runs entirely on the GPUs.
- For agents: a prefix cache across turns, idle conversations moved to RAM (and optionally to disk) instead of being
  recomputed, and the expert cache follows the conversation. For long contexts Flash-Next can also keep the KV cache
  of the conversations being generated in RAM (`--kv-ram-active`), with one request or several at once.

capy-flash **is ninfer-tp2, renamed at v0.5.0**: the two-GPU tensor-parallel fork of
[Neroued/ninfer](https://github.com/Neroued/ninfer) (v0.1-v0.4.9 under the old name); [NOTICE](NOTICE) lists every piece
of prior work it builds on. Status: **v0.14.1**. Built, tested and measured on one machine ([below](#benchmarks));
users report other setups. Not affiliated with Qwen or NVIDIA.

## Pick a model

**16 GB of VRAM per GPU is the minimum.** RAM matters for Flash-Next only.

| what you want to run | GPUs (RTX 50 series, sm_120) | RAM | download / free disk | `install.sh` |
|---|---|---|---|---|
| Qwen3.8-27B, 3-bit | **one** with 16 GB of VRAM (the minimum) | nothing special | 12.5 GB / 15 GB | `--model 27b-16gb` |
| Qwen3.8-27B, NVFP4 | two with 16 GB | nothing special (8 GiB of context cache) | 17.4 GB / 20 GB | `--model 27b` |
| Flash-Next, 64 GB edition | **one** or two with 16 GB | a **64 GB** machine (~56 GiB free) | 108.9 GB / 118 GB | `--model flash-next-64gb` |
| Flash-Next, NVIDIA NVFP4 (default) | two with 16 GB | a **96 GB** machine (~75 GiB free) | 126.6 GB / 135 GB | (default) |

- **One GPU, little RAM:** the 3-bit 27B. It costs quality: GSM8K on par with the NVFP4 27B, MMLU-Pro about 0.72
  against 0.78 ([benchmarks](#benchmarks)).
- **One GPU and 64 GB of RAM:** the 64 GB edition of Flash-Next (ISTA-DASLab's GSQ-RCO experts, 48 GiB in RAM instead
  of 65; quality on par with NVFP4 on GSM8K and MMLU-Pro).
- **Two GPUs:** the 64 GB edition is the fastest Flash-Next here; the NVFP4 edition uses NVIDIA's experts as published.
  For the 27B on two cards, the NVFP4 27B is the better choice than the 3-bit one.

## What you need

- Linux x86_64 (tested on Ubuntu). Users run the 27B on WSL2 and on Windows with Docker; a native Windows build exists
  as a community port ([OmarxGx/capy-flash](https://github.com/OmarxGx/capy-flash)), not in the releases yet.
- An NVIDIA driver with CUDA 13.1 support (R580 or newer). The release needs no CUDA toolkit; building does (CUDA 13.1).
- `sm_120` GPUs with 16 GB: RTX 5060 Ti 16 GB, 5070 Ti, 5080 and 5090 are reported to work; a mixed pair runs at the
  pace of the slower card. Peer-to-peer is not needed.
- **Flash-Next only:** a CPU with AVX-512 (F/BW/VL, VNNI) or AVX2 + FMA + F16C (Zen 2/3, Intel client cores), and the
  free RAM of the table: the routed experts are one page-locked host allocation (64.6 GiB for NVFP4, 48 GiB for the
  64 GB edition), and `--profile flash-next` also keeps the token embedding and up to 4 GiB of idle conversations in
  RAM. The server checks this at startup and refuses with the numbers when it does not fit (`NINFER_RAM_GUARD=0`
  starts anyway). On Windows/WSL2, which limit page-locked memory to about half the RAM, the layers that do not fit
  stay in pageable RAM (slower; on prompts long enough to stream experts the tokens can differ from a fully
  page-locked run, because the CPU computes those layers' experts). With the 64 GB edition on a 64 GB machine that also runs a desktop,
  `--expert-exclusive-layers 20` (or `NINFER_QWEN4_EXCLUSIVE_LAYERS=20`) keeps no RAM copy of the experts the GPUs already hold in the first 20 layers
  (7.7 GiB less on two GPUs; the tokens change, and decode is about 2-3 % slower).

## Quick start

Default model (Flash-Next NVFP4, two 16 GB cards); for another row of the table add its `--model`:
```bash
curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash -s -- --check   # can this machine run it?
curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash               # release + model into ~/capy-flash
~/capy-flash/current/serve-flash-next.sh ~/capy-flash/models/flash-next/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"capy-flash","messages":[{"role":"user","content":"What is 17*23?"}]}'                # from another shell
```
- The installer checks GPUs, driver, AVX2/AVX-512, RAM and disk and says what to fix, downloads the latest release and
  **one** model repository from Hugging Face, verifies both, and prints the command that serves it and a test request.
  `--dir` picks another place, `--no-model` downloads the release only, `--verify-model` also checks the model's digests.
- The first start takes a minute or two (a cold start of Flash-Next reads its experts from disk).
- **Launchers.** `serve-flash-next.sh` (Flash-Next; `NINFER_DEVICES=0` for one GPU), `serve-tp2.sh` (27B, two GPUs) and
  `serve-27b-16gb.sh` (27B 3-bit, one GPU) start `ninfer-serve` with the tested profile, the model id (`capy-flash` or
  `qwen3.8-27b`, the `model` name requests must use) and the chat template. `NINFER_PORT=9000` changes the port;
  extra `ninfer-serve` flags go after the model path and win over the profile.
- **Vision.** The two 27B artifacts include the vision tower and their launchers load it (it takes VRAM from the
  context); `NINFER_VISION=0` before the launcher, or `--no-vision` after the model path, turns it off. The Flash-Next
  downloads are text only: images on Flash-Next need the separate
  [Vision artifact](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash), which `install.sh` does
  not fetch, and `--vision`.
- **Keep it running.** `ninfer-serve` exits with status 2 when the Engine fails, so run it under a supervisor that
  restarts it: a systemd unit and a llama-swap entry are in [examples/serving](examples/serving).
- By hand: download a [release](https://github.com/ValerioDolci/capy-flash/releases) tarball
  ([contents](tools/release/README-release.md)) and `hf download Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash --local-dir <dir>`.

## Connect a client

The server listens on `http://127.0.0.1:8080` (`--host` / `--port`; with the launchers, `NINFER_HOST` / `NINFER_PORT`):
OpenAI Chat Completions and Responses under `/v1`, Anthropic Messages at `/v1/messages`. `--api-key KEY` makes it
require `Authorization: Bearer KEY` or `x-api-key: KEY`; `--api-key-file FILE` or the `NINFER_API_KEY` environment
variable set the key without putting it on the command line.

- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` (any model name is accepted on Messages).
- **Codex CLI:** a custom provider with base URL `http://127.0.0.1:8080/v1`, the `responses` wire API and the server's
  model id. It needs the chat template the launchers pass (`qwen3_8.jinja`), because Codex always sends `instructions`.
- **OpenCode and other OpenAI clients:** base URL `http://127.0.0.1:8080/v1`, model `capy-flash` (or `qwen3.8-27b`).

Details, including authentication and what each client sends: [docs/serving/capy-flash.md](docs/serving/capy-flash.md#connect-a-client).

**Structured output.** `response_format` (`json_object`, `json_schema`) on Chat Completions, `text.format` on
Responses, `output_config.format` on Messages, and the `structured_outputs` extension (`grammar`, `regex`, `choice`)
constrain the answer: the reply is valid JSON for the schema (or matches the grammar, regex or choice) whenever the
model ends it. With thinking on, only the answer after `</think>` is constrained.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "capy-flash", "messages": [{"role": "user", "content": "A cat, as JSON."}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "cat", "schema": {"type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}}}'
```

Schemas and grammars come from the client, so they are bounded: size, nesting and item limits before compilation, a
compilation timeout, and a time and memory budget while generating (`--constraint-*` flags). A request over a limit
fails alone with a 400 that names the flag; the other requests keep their speed. Fields, errors, limits and cost:
[docs/serving/structured-output.md](docs/serving/structured-output.md).

## The capy-flash app (alpha)

[`app/`](app) is a local window for the engine, **in alpha**: it can change from one release to the next and has been
used on few machines. On Linux with suitable GPUs it checks the machine as `install.sh` does, downloads and verifies
the release and a model that fits, starts and stops the engine with the release's launchers, tunes the settings that
leave the answers unchanged, and chats with it (streaming, reasoning, speed of every answer). On a Mac, or anywhere
the engine cannot run, it is a chat window for a capy-flash (or other OpenAI-compatible) server elsewhere. Python 3.9+,
standard library only.

```bash
git clone https://github.com/ValerioDolci/capy-flash && cd capy-flash
app/launch/capy-app             # opens the window
app/launch/capy-app --install   # adds an icon (macOS: ~/Applications, Linux: the applications menu)
```

It is not in the release tarball yet; network access from other machines, unloading when idle and starting at login
are not there yet. Problems: [open an issue](https://github.com/ValerioDolci/capy-flash/issues) with what the app
showed and its engine log. Everything else: [app/README.md](app/README.md).

## Run it other ways

### Build
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
ninja -C build apps/ninfer-serve
build/apps/ninfer-serve <dir>/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer --profile flash-next --model-id capy-flash \
  --chat-template tools/chat_templates/qwen3_8.jinja
```
Or take the prebuilt tarball of a [release](https://github.com/ValerioDolci/capy-flash/releases) (Ubuntu, `sm_120a`),
which has the three launchers ([tools/release](tools/release/README-release.md)).

### Profiles
`--profile NAME` applies the tested configuration of a model; flags given on the command line override it, in any
order. `ninfer-serve --help` and [docs/serving/capy-flash.md](docs/serving/capy-flash.md#profiles) list the full expansions.

| profile | for | in short |
|---|---|---|
| `flash-next` | Flash-Next on two GPUs (the launcher's default) | `--tp 2`, INT8 KV, elastic KV / expert slots, 262,144 tokens, MTP 3, 4 GiB of idle KV and 8 conversation states in RAM, embedding in RAM |
| `flash-next-multi` | Flash-Next on two GPUs for 2-3 clients at once (agents) | `flash-next` with 3 requests at a time, the KV in RAM (`--kv-ram-active`) for 786,432 tokens in all, MTP 1 |
| `flash-next-1gpu` | the 64 GB edition on one GPU (`NINFER_DEVICES=0`) | the same on one device, 1 GiB of idle KV and 8 states in RAM |
| `27b` | Qwen3.8-27B NVFP4 on two GPUs | `--tp 2`, INT8 KV, 196,608 tokens, MTP 3, Vision, 8 GiB of context cache in RAM |
| `27b-16gb` | Qwen3.8-27B 3-bit on one GPU | INT8 KV, 65,536 tokens, MTP 3, Vision, embedding in RAM (13.9 GB of VRAM) |

For the 64 GB edition on two GPUs add `--host-kv-mib 1024 --host-state-slots 2` to `--profile flash-next` (smaller RAM
caches); `install.sh --model flash-next-64gb` prints that command.

For long contexts on Flash-Next, add `--kv-ram-active` to either Flash-Next profile: the KV cache stays in pinned RAM,
taken as the conversations grow (about 2.5 GiB per GPU for 262k tokens, 4.2 GiB on one GPU), and the VRAM it frees
goes to expert slots. Same tokens; within a few per cent of the default speed
([costs](docs/maintainer/active-kv-in-ram.md#cost)). For several long conversations at once add
`--max-concurrency N --kv-capacity T` (T tokens for all of them together; the RAM follows what they use). Prompts are
prefilled one at a time: a request waits at most `--pending-timeout-ms` (default 10 minutes) for the others.

Past the native 262,144 tokens, `--rope-yarn-factor F` (1 to 4) raises the accepted `--max-context` to 262,144 × F
and applies YaRN as Hugging Face does (`rope_type: "yarn"`) to every text RoPE on every GPU, the Flash-Next QSA
indexer and MTP layer included. Off by default; experimental: it costs perplexity, and its quality past the native
window is not measured ([docs/serving/options.md](docs/serving/options.md)).

### Docker
[docker/Dockerfile.release](docker/Dockerfile.release) packs a release tarball (builds in about a minute; the root
[Dockerfile](Dockerfile) compiles from source instead). It needs the NVIDIA Container Toolkit:
```
docker build -f docker/Dockerfile.release -t capy-flash .
docker run --gpus all -p 8080:8080 -v ~/capy-flash/models:/models capy-flash \
  /models/flash-next/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer --profile flash-next --host 0.0.0.0 \
  --model-id capy-flash --chat-template /opt/capy-flash/chat_templates/qwen3_8.jinja
```
Tested with rootless Podman and the two GPUs passed through (Flash-Next loads its 64.6 GiB of page-locked experts
without extra limits).

### Model weights
Ready to serve on Hugging Face (`install.sh --model` downloads one of these; each has its `SHA256SUMS`):
- [`Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash)
  (126.6 GB, the default; NVIDIA Open Model License together with the Qwen Community License of Qwen3.8-Flash-Next).
  Or convert `nvidia/Qwen3.8-Flash-Next-NVFP4` yourself in one command:
  `python tools/capy_flash/convert.py --model <checkpoint dir> --out <dir>`. It runs `python -m tools.convert --recipe
  qwen3_8_flash_next_nvfp4_idxbf16_gdnq8 --components text,mtp` (experts NVFP4 as published, indexer BF16, GDN Q8,
  about 5 minutes on a GPU, about 127 GB) and `tools/qwen4_draft_shortlist.py`. The conversion keeps NVIDIA's MTP
  head; the published artifact has one retrained on this engine's output, which gives the same greedy output with more
  drafts accepted, so a faster decode. The previous text artifact, with NVIDIA's MTP head, stays available as revision
  `v0.5.4` (`hf download ... --revision v0.5.4`).
- [`Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash)
  (the same plus the Q8 vision tower; download by hand, serve with `--vision`; or `convert.py ... --vision`).
- [`Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash)
  (108.9 GB, the 64 GB edition: ISTA-DASLab's GSQ-RCO experts, the rest as above).
- [`Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash) (12.5 GB,
  ISTA-DASLab's 3-bit GSQ 27B plus the official MTP layer; recipe `qwen3_8_27b_gsq3`).
- [`Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer`](https://huggingface.co/Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer)
  (17.4 GB, the 27B for two GPUs; [model card](model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/README.md)).

How the engine runs these models, every server flag, the RAM tiers, the chat template and the CPU threads:
[docs/serving/capy-flash.md](docs/serving/capy-flash.md). Environment variables and diagnostics:
[docs/capy-flash/diagnostics.md](docs/capy-flash/diagnostics.md). All documentation: [docs/README.md](docs/README.md).

### Optional speed switches (off by default)

Three server options make prefill faster at the price of a small change in the numbers: two for Flash-Next,
one for the 27B on long prompts. All three are **off by default**. Pass them to the server, e.g.
`ninfer-serve ... --attn-pv-f16acc`; the environment variable in each row sets the same switch.

| Option (variable) | Applies to | What it changes | Prefill gain (2x RTX 5070 Ti, fresh prompts) | Loss, our measurements |
|---|---|---|---|---|
| `--tp-ar-payload int8` (`NINFER_TP_AR_PAYLOAD=int8`) | Flash-Next (no effect on the 27B) | the prefill exchange between the GPUs carries int8 codes with an FP16 scale per 32 rows instead of FP16 | +7-8 % | perplexity −0.22 % (NVFP4), −0.26 % (GSQ-RCO): within noise |
| `--q8-a8 128` or `32` (`NINFER_Q8_A8`) | Flash-Next | the dense Q8 projections of the prefill run with int8 activations on the int8 tensor cores | +7-8 % | perplexity +0.21 % (NVFP4), +0.004 % (GSQ-RCO): within noise |
| `--attn-pv-f16acc` (`NINFER_ATTN_PV_F16ACC=1`) | the 27B (Flash-Next unchanged) | the INT8-KV prefill attention sums P x V of each 64-key tile in FP16 before adding it to the FP32 accumulator (GeForce Blackwell runs FP16 accumulation at twice the rate) | +6 % at 16k, +18 % at 64k, +25 % at 128k | perplexity +1.1 % on the full 64k corpus (p 0.29, all of it from one English domain); GSM8K (1,319) 0.9629 vs 0.9666 (p 0.23); MMLU-Pro (504) 0.8095 = 0.8095; long-context retrieval at 64k and 126k unchanged |

The lossy switches change individual tokens, so they stay opt-in. On Flash-Next the loss is negligible in our tests;
`--attn-pv-f16acc` moves the 27B's perplexity more, but its answers and long-context retrieval held. The three
Flash-Next switches together on Flash-Next NVFP4: prefill +28-29 % over the default; GSM8K (1,319 questions) 0.9788, against 0.9811
for vLLM on a GB300 (McNemar p 0.38); MMLU-Pro (504 questions) 0.8472, against 0.8492 for the default under the same
conditions (paired, p 1.0). Details in [docs/capy-flash/diagnostics.md](docs/capy-flash/diagnostics.md).

## Benchmarks

Our machine: AMD Ryzen 9 9900X (12 cores), 96 GB DDR5-5600, 2x RTX 5070 Ti 16 GB on PCIe 5.0 x8 + x8 (no P2P),
Ubuntu, CUDA 13.1, full clocks. Decode is the server-side rate. v0.8.0 runs at the speed of v0.7.7 (the
[release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.8.0) have the comparison). The agentic
replay is [bench/capy_flash](bench/capy_flash): 3 synthetic coding-agent sessions, 53 turns, prompts growing to ~81k tokens with
~90 % prefix reuse, 256 output tokens per turn, temperature 0. LocalMaxxing is the reasoning-v1 benchmark (short
prompt, 512 tokens).

**Ours** (one GPU = one RTX 5070 Ti):

| model, edition | GPUs | version, date | measured | result |
|---|---|---|---|---|
| Flash-Next NVFP4 | 2 | v0.6.0, 2026-10-18 | agentic replay: decode median / total time / time to first token, median (p90) | 155 tok/s / 177 s / 1.48 s (2.62 s) |
| Flash-Next NVFP4 | 2 | v0.6.0, 2026-10-18 | prefill, cold, ~3.5k / ~15k / ~55k-token prompts | 2,422 / 4,615 / 4,828 tok/s |
| Flash-Next NVFP4 | 2 | v0.6.0, 2026-10-18 | LocalMaxxing reasoning-v1 | 178 tok/s |
| Flash-Next NVFP4 | 2 | v0.7.0, 2026-10-22 | agentic replay decode median | 155 tok/s (as v0.6) |
| Flash-Next 64 GB | 2 | v0.7.0, 2026-10-22 | agentic replay decode median | 182 tok/s |
| Flash-Next 64 GB | 2 | v0.7.0, 2026-10-22 | LocalMaxxing reasoning-v1, [verified run](https://www.localmaxxing.com/runs/cmvjb7d5d03o3rf01ex9g9wiw) | 234 tok/s |
| Flash-Next 64 GB | 2 | v0.7.2, 2026-10-23 | prefill 4k / 16k / 64k | 2,168 / 3,924 / 3,994 tok/s |
| Flash-Next 64 GB | 2 | v0.13.1, 2026-11-15 | prefill, cold, 30k / 200k-token source-code prompts | 4,876 / 4,359 tok/s (v0.13.0: 4,541 / 4,041) |
| Flash-Next 64 GB | 1 | v0.7.0, 2026-10-22 | LocalMaxxing reasoning-v1, [verified run](https://www.localmaxxing.com/runs/cmvjb7dm603o7rf01i54inoup) | 121 tok/s |
| Flash-Next 64 GB | 1 | v0.7.1, 2026-10-22 | agentic replay: decode median / total time / time to first token, median / prompt tokens from cache | 104 tok/s / 276 s / 1.9 s / 83 % |
| Flash-Next 64 GB | 1 | v0.7.2, 2026-10-23 | prefill 4k / 16k / 64k | 1,671 / 2,838 / 2,921 tok/s |
| Flash-Next 64 GB | 1 | v0.13.1, 2026-11-15 | prefill, cold, 30k / 200k-token source-code prompts | 3,738 / 3,505 tok/s (v0.13.0: 3,306 / 3,040) |
| Qwen3.8-27B 3-bit | 1 | v0.7.0, 2026-10-22 | decode, 8k-token real-text prompt, MTP | 139 tok/s |
| Qwen3.8-27B 3-bit | 1 | v0.7.0 | prefill 4k / 16k | 3,401 / 3,211 tok/s |
| Qwen3.8-27B 3-bit | 1 | v0.7.0, 2026-10-22 | LocalMaxxing reasoning-v1, [verified run](https://www.localmaxxing.com/runs/cmvjb7ebb03ofrf01mkfqs948) | 174 tok/s |
| Qwen3.8-27B 3-bit | 2 | v0.7.0, 2026-10-22 | decode, 8k-token real-text prompt, MTP | 230 tok/s |
| Qwen3.8-27B 3-bit | 2 | v0.7.0 | prefill 4k / 16k / 64k | 5,787 / 5,567 / 4,304 tok/s |
| Qwen3.8-27B 3-bit | 2 | v0.7.0, 2026-10-22 | LocalMaxxing reasoning-v1, [verified run](https://www.localmaxxing.com/runs/cmvjb7emi03oirf01vyxuhulr) | 288 tok/s |
| Qwen3.8-27B NVFP4 (QUASAR) | 2 | v0.6.0 | prefill 4k / 16k / 64k, `--profile 27b` | 9,343 / 9,234 / 6,287 tok/s |

More on the 27B NVFP4 (`--profile 27b`, MTP; 280 prompts in three languages, v0.6): the same tokens as v0.5.4 on 280 of
280 answers (and with DFlash2), decode +5 % (Italian 168 → 176 tok/s), prefill at 4k / 16k / 64k 5,777 / 5,585 / 4,352
→ 9,343 / 9,234 / 6,287 tok/s with MTP and 5,719 / 5,530 / 4,316 → 9,205 / 9,052 / 6,198 with DFlash2; concurrent decode (NInfer's `decode-saturation`
suite, MTP3, 8,192 tokens per request) 175 / 326 / 560 / 927 tok/s aggregate at 1 / 2 / 4 / 8 requests. On an agentic
replay of the 27B (3 interleaved sessions and 11 subagents, 4 lanes), the v0.5.4 context cache took the median time to
first token from 2.14 to 0.98 s and the prompt tokens served from cache from 67 to 78 %, decode unchanged.
Flash-Next decode stays flat with context: 168 / 165 / 171 / 165 tok/s with 0 / 14k / 61k / 118k tokens of prompt
(v0.6), and a cold 382-token prompt answers in 1.35 s. A cold start of the Flash-Next server takes 26-29 s here
(v0.5.3), most of it reading the 64.6 GiB of experts. The 64 GB edition peaks at 55.2 GiB of RSS inside a 64 GiB
memory cgroup without swap (57 GiB used with one GPU); its speeds were measured on the 96 GB machine.

**Reported by users** (not reproduced by us):

| setup | model, edition | version | measured | result | source |
|---|---|---|---|---|---|
| AMD EPYC 7F52 16 cores (Zen 2, AVX2), 2x RTX 5080 (overclocked) | Flash-Next NVFP4 | v0.7.2 built with CUDA 13.4 | agentic replay (53 turns, greedy): decode median | 141.3 tok/s with automatic threads; 144.4 with `--expert-cpu-threads 15` and `taskset -c 0-15`; release tarball (CUDA 13.1) 137.6; v0.7.1 tarball 121.8 | [issue #11](https://github.com/ValerioDolci/capy-flash/issues/11) |
| same | same | same | own measurement, not greedy | prefill at 131k 5,541 tok/s; time to first token 22.1 s at 131k, 1.15 s at 1k | [issue #11](https://github.com/ValerioDolci/capy-flash/issues/11) |
| AMD Threadripper PRO 5975WX (AVX2), 2x RTX PRO 4000 Blackwell 24 GB (145 W), Proxmox LXC | Flash-Next 64 GB (GSQ-RCO) | v0.7.1 built from source, CUDA 13.3 | decode / prefill / local aider coding benchmark | decode ~100-114 tok/s on prose, ~140-155 on reasoning; prefill at 40k ~3,300 tok/s; aider 145-146 of 148 | [issue #22](https://github.com/ValerioDolci/capy-flash/issues/22) |

**Against Strata.** [Strata](https://github.com/Niko1221/Strata) 0.1.45 on our machine, same day (2026-10-18), two GPUs,
capy-flash v0.6.0:

| | **capy-flash**, NVFP4 | Strata, same NVFP4 weights | Strata, 3-bit IQ3_S (tuned) |
|---|---:|---:|---:|
| agentic replay, 53 turns up to 81k tokens: total time | **177 s** | 268 s | 197 s |
| agentic replay: decode, server side (median) | **155 tok/s** | 98 tok/s | 151 tok/s |
| agentic replay: time to first token, median (p90) | **1.48 s (2.62)** | 2.35 s (4.17) | 1.60 s (3.69) |
| prefill, ~3.5k-token prompt | **2,422 tok/s** | 1,519 tok/s | 2,216 tok/s |
| prefill, ~15k-token prompt | **4,615 tok/s** | 3,513 tok/s | 3,811 tok/s |
| prefill, ~55k-token prompt | 4,828 tok/s | 4,849 tok/s | **5,233 tok/s** |
| LocalMaxxing reasoning-v1 (short prompt, 512 tokens) | 178 tok/s | 119 tok/s | **195 tok/s** |

Strata's IQ3_S experts are 26 % smaller than NVIDIA's NVFP4 ones (lower quality: KL 0.144 vs 0.070 for NVFP4, as
Strata reports), which is why it leads on the short benchmark. v0.6 against the build before the v0.6 work
(2026-10-17): replay total 212 → 177 s, decode 118 → 155 tok/s.

**Comparing decode with other engines.** capy-flash computes every routed expert the model selects; an expert that
misses the VRAM cache is computed on the CPU, never skipped. Strata's "route tail skip" skips low-ranked experts that
miss the cache, which changes the output slightly and saves CPU work. A decode-speed gap against Strata, especially on
a slower CPU, most likely comes from that. Measured here with the same GSQ-RCO IQ3_S experts on both (one RTX 5070 Ti,
LocalMaxxing decode, 2026-10-23): Strata 138 tok/s with the skip, 116.6 with `STRATA_ROUTE_TAIL_SKIP=0`, capy-flash
129.8. On two GPUs (capy-flash v0.7.0 against Strata 0.2.2, 2026-10-22) the agentic replay decode median was 200.9
against 171.3 tok/s; Strata finished that replay sooner overall (189 against 200 s) thanks to its faster prefill.

**Quality** against the GB300 reference (NVIDIA's NVFP4 checkpoint served by vLLM on a GB300, same prompts, paired):

| model, version | GSM8K (reference) | MMLU-Pro (reference) |
|---|---:|---:|
| Flash-Next NVFP4, v0.6.0 | 0.9750 (0.9810), 1,319 questions; 9 / 1 discordant, McNemar p 0.02 | 0.8413 (0.8452), 1,008 questions: non-inferior (difference -0.40 pt, lower bound -1.85 pt, margin -2.5 pt) |
| Flash-Next NVFP4, v0.7.0 | 0.985 (0.985), 330 questions | 0.829 (0.849), 252 questions; paired with v0.6: no difference |
| Flash-Next 64 GB, v0.7.0 | 0.982 (0.985), 330 questions | 0.837 (0.849), 252 questions |
| Qwen3.8-27B 3-bit, v0.7.0 | 0.968 (0.968), 1,319 questions | 0.69-0.71, about 0.72 with the empty-reply guards (0.78), 252 questions |

The v0.6.0 release binary was checked on the first 330 / 252 questions (GSM8K 0.979, MMLU-Pro 0.841). Output is not
bit-reproducible run to run (which experts sit in VRAM depends on timing), so two runs of the same build differ by a
few questions. The 3-bit 27B loses MMLU-Pro points mostly because more answers run out of their 8,192
tokens while reasoning; the v0.7 empty-reply guards bring back the ones that stopped inside their reasoning (9 of 9,
all correct: about +3.6 points, estimated on those questions). Agent tasks (2026-10-14 build): 5 multi-step coding
tasks of a private task bank, 4 runs each, OpenCode with the model's sampling: 20/20 solved, 42.9k generated tokens per
run (Strata 0.1.43 IQ3_S on the same tasks: 52.7k, 20/20; the GB300 reference: 26.5k), no malformed tool call in 1,124.

## Known limits
- Flash-Next's profiles serve one request at a time (`--max-concurrency 1`) and only the machine above is tested. With
  `--max-concurrency 2` to `4` (with or without `--kv-ram-active` and `--elastic`) the decode of the requests is
  batched: two requests of 32k-100k tokens decode 1.2-1.4x the tokens per second of one, four only 1.1-1.2x (four
  MTP windows exceed the 8-token fast path of the expert kernels; with `--draft-tokens 1` four requests decode
  1.3x the tokens of one request with MTP 3, at the cost of 10 % for a request alone). RAM is the limit: with Flash-Next loaded about
  10 GB of the 96 GB stay free, about 400k tokens of int8 KV in RAM at tp 2. The server accepts up to 8
  (`--max-concurrency`); the 27B NVFP4 was measured at 1 to 8.
- With several requests decoding, an answer that outgrows its KV window when the GPUs are full first moves idle cached
  conversations to RAM (or drops them without a RAM tier); only if that cannot free a page does it end early with
  `finish_reason: length` (the request log counts it). `--kv-lease-window 0` restores the full reservation of v0.5.3.
- Moving a conversation between RAM and the GPUs shares PCIe with the two GPUs' exchange: decode of other requests
  is slower while a move is in flight (no difference otherwise).
- Generated tokens per agent task are about 1.6x the GB300 reference (above): the remaining gap with the datacenter
  stack is length of reasoning, not correctness.
- Output is not bit-reproducible run to run: which experts sit in VRAM (GPU path) or not (CPU path) depends on timing,
  and long streamed prefills differ in the last bits; quality is measured with that in place.
- Decode of Flash-Next is bound by host-RAM bandwidth for the experts computed on the CPU (about 270-300 per MTP
  window), so a slower CPU or RAM decodes slower.

## Tests
`ctest --test-dir build` runs the unit tests. The Engine integration tests run when their artifacts are set:
`NINFER_TEST_QWEN4_ARTIFACT` (a small synthetic Flash-Next artifact with the real layer geometry, made by
`tools/capy_flash/make_test_artifact.sh <checkpoint dir> <out dir>`, CPU only) and `NINFER_TEST_ARTIFACT` (a
Qwen3.8-27B artifact, for the tests inherited from ninfer-tp2). `ctest -j4` is supported; tests that need a model too
large for one 16 GB board skip with the sizes. For the 27B, `tools/tp2/gate.sh` is the reference run: it sets the
artifacts, runs the integration tests (`GATE_CTEST_EXCLUDE` excludes more), then checks golden ids, perplexity and
greedy texts against a recorded reference ([tools/tp2/README.md](tools/tp2/README.md)).
The app's tests run without a GPU: `cd app && python3 -m unittest discover -s tests`.

## What's new

**v0.14.1** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.14.1)), a fix and two pull requests by @Mici81; same tokens
- Flash-Next with MTP no longer stops the server when the same prompt comes back a third time and its draft frontier
  ends on a KV page boundary (the trim of a shared full page is a no-op again, as for idle conversations).
- Flash-Next Q6 output head at tp 2 on a 32x8 tile for 2-8 columns ([PR #88](https://github.com/ValerioDolci/capy-flash/pull/88))
  and a 4-way set-associative QSA staging cache ([PR #91](https://github.com/ValerioDolci/capy-flash/pull/91)); same bits.

**v0.14.0** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.14.0)), a cleanup release; same speed, greedy text can differ from v0.13.3
- Two-GPU 27B keeps the active conversation in its context cache: coding-agent replay 303-337 s → 109 s, prefix reuse
  53-58 % → 90 %.
- Greedy text no longer depends on the cache's history; the 27B with MTP gives the same text at any `--draft-tokens`
  (1-5) and with `--spec off`.
- `--profile flash-next-multi` for two or three clients at once; `--no-endpoint-reuse`.
- Environment variables become options (`--expert-cpus`, `--expert-exclusive-layers`, `--gguf-hybrid-cost`,
  `--kv-ram-region-tokens`, `--kv-ram-cache-blocks`, `--q8-a8`, `--attn-pv-f16acc`, `--tp-ar-payload`); twelve variables
  that selected internal alternatives are gone (ignored without a warning, list in the release notes).
- The code every model runs is reorganised (Text forward, kernel route families, CPU experts in `src/ops/host_moe`);
  `ctest -L cpu` runs the tests that need no GPU.

**v0.13.3** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.13.3)), five pull requests by @Mici81, same tokens
- Two-GPU prefill of the 64 GB edition +3-5 %: each 8192-column group of a prompt chunk goes to the other GPU as soon
  as it is enqueued ([PR #80](https://github.com/ValerioDolci/capy-flash/pull/80)).
- With a model loaded and idle, the CPU expert service checks for work every 10 ms after a quiet second, so the CPU can
  rest (package 31 W to 23 W on a Ryzen 9 9900X) ([PR #84](https://github.com/ValerioDolci/capy-flash/pull/84)).
- The disk tier saves and resumes conversations with images ([PR #82](https://github.com/ValerioDolci/capy-flash/pull/82)),
  and `--reasoning-close-retry` works on prompts with images ([PR #83](https://github.com/ValerioDolci/capy-flash/pull/83)).
- Opt-in IQ3 decoder for hybrid Intel CPUs without AVX-512 ([PR #85](https://github.com/ValerioDolci/capy-flash/pull/85)).
- `--host-pinned-max-gib` (and the WSL2 fallback) can change the tokens on long prompts; the docs said otherwise ([#81](https://github.com/ValerioDolci/capy-flash/issues/81)).

**v0.13.2** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.13.2))
- Coding agents run without error answers (checked with Claude Code 2.1, OpenCode 1.19 and Codex CLI 0.169): an image
  or a PDF sent to a model without Vision reaches it as a line of text instead of a 400 that blocked the
  conversation; image `detail` `low`/`high` and Claude Code's `thinking.display: "updates"` are accepted; a request
  with no output limit gets the rest of the context (it was 8,192 tokens); a context overflow reads `prompt is too
  long: N tokens > M maximum`; every error answer gets a log line with its request id. Same tokens.

**v0.13.1** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.13.1))
- Faster prefill for the 64 GB edition, same tokens: the grouped GGUF expert kernels use all four warp schedulers and
  8192-column pieces ([PR #78](https://github.com/ValerioDolci/capy-flash/pull/78) by @Mici81), and the widest prompt
  chunk is 32768 where it fits ([PR #79](https://github.com/ValerioDolci/capy-flash/pull/79) by @Mici81). Cold prefill
  of a 200k-token prompt: +8 % on two GPUs, +15 % on one. Flash-Next NVFP4 and the 27B are unchanged.

**v0.13.0** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.13.0))
- A cleanup release: at every default and profile the same tokens at the same speed as v0.12.0, with about 42,000
  fewer lines outside the new app (superseded A/B switches, code no route reached, duplicated helpers). 44
  diagnostic environment switches whose default was already served are gone; the list is in the release notes.
- `--rope-yarn-factor` now applies YaRN to every text RoPE path on every GPU, Flash-Next's QSA indexer and MTP layer
  included, as Hugging Face does; before, only the 27B on one GPU got it.
- Opt-in `NINFER_QWEN4_EXCLUSIVE_LAYERS=N` for the 64 GB edition next to a desktop on 64 GB of RAM
  ([PR #77](https://github.com/ValerioDolci/capy-flash/pull/77) by @Mici81); `--help` grouped by topic
  ([PR #76](https://github.com/ValerioDolci/capy-flash/pull/76) by @fantasyz).
- Corrected requirements: Flash-Next NVFP4 needs ~75 GiB of free RAM, the 64 GB edition 118 GB of disk.
- Internal server errors of the OpenAI-compatible routes always carry `"type": "server_error"`.
- **The capy-flash app, in alpha** ([app/](app)): installs, starts, tunes and chats with the engine on Linux; a chat
  window for a remote server on a Mac.

Older releases: [GitHub releases](https://github.com/ValerioDolci/capy-flash/releases).

## License
Apache License 2.0 ([LICENSE](LICENSE)); attribution in [NOTICE](NOTICE). The original ninfer-tp2 README is the one of
[v0.4.9](https://github.com/ValerioDolci/capy-flash/blob/v0.4.9/README.md) (historical).
