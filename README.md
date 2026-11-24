
<h1 align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/capy-flash-lockup-dark.svg">
  <img alt="capy-flash" src="docs/logo/capy-flash-lockup.svg" width="560">
</picture></h1>

**Qwen3.8-Flash-Next (125B MoE), Qwen3.8-27B and Inkling 2 Mini on one or two 16 GB RTX 50-series GPUs, built for agents.**

capy-flash is an inference server. It runs three models on consumer GPUs with 16 GB of VRAM and exposes an OpenAI- and
Anthropic-compatible HTTP API, so agent tools such as Claude Code, Codex CLI and OpenCode can use it like a hosted model.

- **Qwen3.8-Flash-Next** is a mixture-of-experts model too large for the GPUs. capy-flash keeps the experts that do
  not fit in VRAM in host RAM and computes them on the CPU; everything else runs on the GPU (split over two GPUs when
  there are two). That is why Flash-Next needs a lot of RAM and a CPU with AVX2 or AVX-512.
- **Qwen3.8-27B** is a dense model that runs entirely on the GPUs.
- **Inkling 2 Mini** is a mixture-of-experts model whose weights fit two GPUs: it runs entirely on them, split over
  two 16 GB cards, with the checkpoint's own weights and nothing requantized, and its own drafter for speculative
  decoding (on in its launcher). Text only, one request at a time ([what it does and does not do](#inkling-2-mini)).
- For agents: a prefix cache across turns, idle conversations moved to RAM (and optionally to disk) instead of being
  recomputed, and the expert cache follows the conversation. For long contexts Flash-Next can also keep the KV cache
  of the conversations being generated in RAM (`--kv-ram-active`), with one request or several at once.

capy-flash **is ninfer-tp2, renamed at v0.5.0**: the two-GPU tensor-parallel fork of
[Neroued/ninfer](https://github.com/Neroued/ninfer) (v0.1-v0.4.9 under the old name); [NOTICE](NOTICE) lists every piece
of prior work it builds on. Status: **v0.16.4**. Built, tested and measured on one machine ([below](#benchmarks));
users report other setups. Not affiliated with Qwen, NVIDIA or Thinking Machines.

## Pick a model

**16 GB of VRAM per GPU is the minimum.** RAM matters for Flash-Next only.

| what you want to run | GPUs (RTX 50 series, sm_120) | RAM | download / free disk | `install.sh` |
|---|---|---|---|---|
| Qwen3.8-27B, 3-bit | **one** with 16 GB of VRAM (the minimum) | nothing special | 12.5 GB / 15 GB | `--model 27b-16gb` |
| Qwen3.8-27B, NVFP4 | two with 16 GB | nothing special (8 GiB of context cache) | 19.2 GB / 22 GB | `--model 27b` |
| Flash-Next, 64 GB edition | **one** or two with 16 GB | a **64 GB** machine (~56 GiB free) | 108.9 GB / 118 GB | `--model flash-next-64gb` |
| Flash-Next, NVIDIA NVFP4 (default) | two with 16 GB | a **96 GB** machine (~75 GiB free) | 126.6 GB / 135 GB | (default) |
| Inkling 2 Mini, NVFP4 + MXFP8 (the checkpoint's own) | **two** with 16 GB | nothing special | 19.7 GB / 23 GB | `--model inkling2-mini` |

- **One GPU, little RAM:** the 3-bit 27B. It costs quality: GSM8K on par with the NVFP4 27B, MMLU-Pro about 0.72
  against 0.78 ([benchmarks](#benchmarks)).
- **One GPU and 64 GB of RAM:** the 64 GB edition of Flash-Next (ISTA-DASLab's GSQ-RCO experts, 48 GiB in RAM instead
  of 65; quality on par with NVFP4 on GSM8K and MMLU-Pro).
- **Two GPUs:** the 64 GB edition is the fastest Flash-Next here; the NVFP4 edition uses NVIDIA's experts as published.
  For the 27B on two cards, the NVFP4 27B is the better choice than the 3-bit one.
- **Inkling 2 Mini** needs two 16 GB cards: its weights take about 10 GB per card. It does not run on one 16 GB
  card. One 32 GB card (RTX 5090) is expected to hold it at `--tp 1` but has **not been verified**.

## What you need

- Linux x86_64 (tested on Ubuntu). Users run the 27B on WSL2 and on Windows with Docker; a native Windows build exists
  as a community port ([OmarxGx/capy-flash](https://github.com/OmarxGx/capy-flash)), not in the releases yet. The
  Inkling 2 Mini code has not been built with MSVC or on Windows.
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
  `qwen3.8-27b`, the `model` name requests must use) and the chat template. `capy-serve-inkling2-mini.sh` (Inkling 2
  Mini, two GPUs, model id `inkling-2-mini`) passes its flags without a profile or template and turns the drafter on
  (`--spec mtp --draft-tokens 3`; `NINFER_DRAFTER=0` turns it off). `NINFER_PORT=9000` changes the port;
  extra `ninfer-serve` flags go after the model path and win over the profile.
- **Vision.** The two 27B artifacts include the vision tower and their launchers load it (it takes VRAM from the
  context); `NINFER_VISION=0` before the launcher, or `--no-vision` after the model path, turns it off. The Flash-Next
  downloads are text only: images on Flash-Next need the separate
  [Vision artifact](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash), which `install.sh` does
  not fetch, and `--vision`. Inkling 2 Mini is text only: `--vision` is refused at startup.
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
model ends it. With thinking on, only the answer after `</think>` is constrained. Not on Inkling 2 Mini, which refuses
constrained requests.

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

## Inkling 2 Mini

Inkling 2 Mini (`Inkling2LoopForConditionalGeneration`) is a 128-expert mixture-of-experts model with a looped core.
capy-flash runs its text part on two 16 GB GPUs from the original checkpoint: the NVFP4
experts of layers 0-19 and the MXFP8 experts of layers 20-23 are copied into the artifact as they are, the dense
weights keep their BF16 values, and nothing is requantized. The checkpoint's DSpark drafter is imported in BF16 the
same way. The drafter leaves the generated text unchanged.

- **What you need:** two `sm_120` GPUs with 16 GB, as for the other models; no special RAM or CPU (no expert runs
  on the CPU). Linux; the Windows and MSVC builds of this code are not verified.
- **Weights:** `install.sh --model inkling2-mini` downloads the release and the artifact
  [`Feyd89/Inkling-2-Mini-NVFP4-capy-flash`](https://huggingface.co/Feyd89/Inkling-2-Mini-NVFP4-capy-flash) (19.7 GB, with the drafter). To build
  the same file yourself, convert the checkpoint
  [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on the CPU (about half a
  minute), 20.2 GB for the checkpoint plus 19.7 GB for the artifact with the drafter; Python 3.11 with PyTorch
  and NumPy, from the repository root ([weight conversion](docs/weight-conversion.md#start-with-an-official-recipe)):
  ```bash
  hf download thinkingmachines/Inkling-2-Mini --local-dir /path/to/Inkling-2-Mini
  python3 -m tools.convert --model /path/to/Inkling-2-Mini --recipe inkling2_mini --components text,mtp \
    --device cpu --name inkling-2-mini --out models/capy-flash-inkling2-mini-nvfp4.ninfer
  ```
  `--components text,mtp` adds the drafter; without it (the default `text`, 19.0 GB) the artifact runs without one,
  also from the launcher.
- **Launch:**
  ```bash
  curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash -s -- --model inkling2-mini
  ~/capy-flash/current/capy-serve-inkling2-mini.sh ~/capy-flash/models/inkling2-mini/capy-flash-inkling2-mini-nvfp4.ninfer
  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":512}'
  ```
  The launcher runs `--tp 2 --devices 0,1 --max-context 253952 --prefill-chunk 4096 --embedding-host --spec mtp
  --draft-tokens 3` (`NINFER_MAX_CONTEXT`, `NINFER_PREFILL_CHUNK`, `NINFER_DEVICES`, `NINFER_PORT`, `NINFER_MODEL_ID`,
  `NINFER_DRAFTER` change them; extra flags after the artifact win, except that `--embedding-host` has no off switch and changes no result). The context is reserved at startup: 253,952 tokens
  with 4096-token prefill blocks and the drafter leave about 580 MiB free per 16 GB card, with the token embedding in
  pinned host memory (`--embedding-host`: one 0.9 GB copy in RAM that both cards read in place, identical results).
  Smaller prefill blocks (2048) prefill 6-13 % slower. The block size does not change results on prompts whose last block has
  at least 128 tokens, and can change the rounding of the few tokens of a shorter tail (a mean KL of 1.8e-3 measured
  on a 10-token tail, within the noise between two valid routes); it is not a reduction of precision. `NINFER_DEVICES=0` starts it on one GPU only if that GPU has 32 GB (not
  verified); with 16 GB the launcher refuses. From a build: `build/apps/ninfer-serve <artifact> --tp 2 --max-context
  253952 --prefill-chunk 4096 --embedding-host`.
- **Drafter:** the launcher turns it on (`--spec mtp --draft-tokens 3`); `ninfer-serve` (or `capy-serve`) started
  by hand leaves it off unless given those flags. To turn it off in the launcher:
  ```bash
  NINFER_DRAFTER=0 ~/capy-flash/current/capy-serve-inkling2-mini.sh ~/capy-flash/models/inkling2-mini/capy-flash-inkling2-mini-nvfp4.ninfer
  ```
  `--spec off`, or another `--draft-tokens`, after the artifact also wins over the launcher's. The drafter proposes
  tokens that the model verifies in one pass; it works with greedy and sampled requests, and the text is the same
  as without it (at temperature 0, and with sampling for the same seed). Steps that run forced tokens (thinking
  budgets), reach the end of the context or run beyond 245,760 tokens of context are ordinary steps. `--draft-tokens` accepts 1 to 5; 3 was the best
  measured. It takes about 1.0 GiB more VRAM per card at the launcher's 253,952-token context (0.7 GiB at 131,072). When few drafts are accepted it
  can decode slower than without it. An artifact without the drafter starts with `--spec mtp` and decodes without
  it. Speed with and without the drafter: [benchmarks](#benchmarks).
- **What it does:** chat over the same OpenAI and Anthropic endpoints, streaming, reasoning in its own field, tool
  calls, thinking budgets, stop strings, temperature / top-k (up to 20) / top-p / min-p and penalties, reuse of
  the previous conversation's prefix (on the GPUs only), and speculative decoding with the checkpoint's drafter.
  Reasoning cannot be switched off: `enable_thinking=false` selects the lowest effort. The chat template is the
  checkpoint's own.
- **What it does not do (yet):** images (`--vision` is refused); `--spec dflash2` and `--lm-head-draft` (refused);
  more than one request at a time (`--max-concurrency` must be 1; further requests queue); one 16 GB card; structured
  output and grammars; `tool_choice` other than automatic; the RAM and disk tiers of the context cache
  (`--disk-cache-dir` is refused) and shared prefixes between conversations; a
  `--kv-dtype` other than `bf16` (refused); continuing a final assistant message. The empty-reply EOS guard is off
  for this model (the server says so at startup).
- **Quality:** not measured with GSM8K or MMLU-Pro. The engine's output is checked against the reference
  implementation with the checkpoint's exact weights, token by token and on greedy generations
  ([validation](docs/maintainer/inkling2-model.md#numerics-and-validation)). With the drafter, the server's text was
  compared with the text without it: identical at temperature 0 and, with sampling, for the same seed. Test suite
  of this release: 273 of 273 tests pass, 28 of them on two GPUs.

How the engine implements it: [Inkling 2 model reference](docs/maintainer/inkling2-model.md).

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
| `27b` | Qwen3.8-27B NVFP4 on two GPUs | `--tp 2`, INT8 KV, 196,608 tokens, DFlash2 7 (MTP 3 with an artifact without the DFlash2 drafter), Vision, 8 GiB of context cache in RAM |
| `27b-16gb` | Qwen3.8-27B 3-bit on one GPU | INT8 KV, 65,536 tokens, MTP 3 with prompt lookup, Vision, embedding in RAM (13.9 GB of VRAM) |

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
  (19.2 GB, the 27B for two GPUs with the DFlash2 drafter, `qwen3_8_27b_quasar_nvfp4_dflash2.ninfer`; the earlier 17.4 GB file
  without it stays there and decodes with MTP; [model card](model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/README.md)).
- [`Feyd89/Inkling-2-Mini-NVFP4-capy-flash`](https://huggingface.co/Feyd89/Inkling-2-Mini-NVFP4-capy-flash)
  (19.7 GB, Inkling 2 Mini with its drafter, for two GPUs): the released weights of
  [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache-2.0)
  repackaged without requantizing; the recipe `inkling2_mini` with `--components text,mtp` builds the same file
  ([above](#inkling-2-mini)).

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
| Qwen3.8-27B NVFP4 (QUASAR) | 2 | v0.16.2, 2026-11-23 | decode, `--profile 27b` (DFlash2 7): conversation / JSON / code edit / Italian essay / quoting from 30K | 277 / 415 / 520 / 140 / 385 tok/s |
| Qwen3.8-27B NVFP4 (QUASAR) | 2 | v0.16.2, 2026-11-23 | agentic replay decode (53 turns, up to 81K tokens), `--profile 27b` | 234 tok/s (MTP 3: 205) |
| Qwen3.8-27B 3-bit | 1 | v0.16.2, 2026-11-23 | decode, `--profile 27b-16gb` (prompt lookup): conversation / JSON / code edit / Italian essay / quoting | 145 / 212 / 318 / 106 / 304 tok/s |
| Inkling 2 Mini | 2 | v0.16.3, 2026-11-24 | decode without the drafter, 68-token / 12.8k-token prompt | 204.5 / 185.9 tok/s (v0.16.0: 127.5 / 112.8) |
| Inkling 2 Mini | 2 | v0.16.3, 2026-11-24 | decode with `--spec mtp --draft-tokens 3` (the launcher's default), temperature 0: 68-token / 12.8k-token prompt / a coding task | 209.8 / 239.6 / 255.8 tok/s (v0.16.0: 149.0 / 166.6 / 193.9) |
| Inkling 2 Mini | 2 | v0.16.3, 2026-11-24 | decode with the drafter, temperature 0, reasoning capped at 32 tokens (`--default-thinking-budget 32`), after a ~4k-token prompt: writing code / an essay | 307.6 / 205.1 tok/s (v0.16.0: 230.2 / 149.2) |
| Inkling 2 Mini | 2 | v0.16.3, 2026-11-24 | decode, summarizing a 130k / 247k-token document (the drafter stops at 245,760 tokens of context) | 140.4 / 87.5 tok/s (v0.16.0: 72.6 / 53.4) |
| Inkling 2 Mini | 2 | v0.16.0, 2026-11-21 | decode with the drafter and the model's default sampling (one seed): 68-token prompt / an essay / a coding task | 146.1 / 129.6 / 173.9 tok/s (124.7 / 124.4 / 124.9 without the drafter) |
| Inkling 2 Mini | 2 | v0.16.3, 2026-11-24 | prefill 512 / 4k / 12.8k / 99k tokens | 5,020 / 8,209 / 7,804 / 6,610 tok/s (v0.16.0: 4,002 / 5,032 / 4,524 / 2,522) |
| Inkling 2 Mini | 2 | v0.16.3, 2026-11-24 | prefill 247k tokens | 5,078 tok/s (v0.16.0: 1,413) |
| Inkling 2 Mini | 2 | v0.16.0, 2026-11-21 | VRAM per GPU at 253,952 tokens of context with v0.16.0's launcher (2048-token blocks, embedding on the GPUs), without / with the drafter | 14.2 / 15.2 GiB |

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

**Inkling 2 Mini.** Every Inkling number here is measured with the checkpoint's own weights: the NVFP4 and MXFP8
experts as released and the dense weights in BF16, with no further quantization of weights, activations or KV.
Against llama.cpp on our machine, same day, two GPUs: capy-flash v0.16.0 from the server with those weights,
llama.cpp `29b52d3` with `llama-bench -ngl 99 -fa on -ts 1/3` and the requantized `IQ4_NL` GGUF from ggml-org
(measured in the morning session):

| | **capy-flash** | llama.cpp |
|---|---:|---:|
| decode, short prompt | 127.5 (149.0 with the drafter) | 44.0 |
| prefill, 512-token prompt | 4,002 | 1,268 |
| prefill, 12,800-token prompt | 4,524 | 1,033 |

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
- Inkling 2 Mini: text only, one request at a time, no structured output, two 16 GB GPUs (one 32 GB card not
  verified), quality not measured with benchmarks; the drafter (on in the launcher) can decode slower than without it when few
  of its drafts are accepted; the full list is in [its section](#inkling-2-mini).

## Tests
`ctest --test-dir build` runs the unit tests. The Engine integration tests run when their artifacts are set:
`NINFER_TEST_QWEN4_ARTIFACT` (a small synthetic Flash-Next artifact with the real layer geometry, made by
`tools/capy_flash/make_test_artifact.sh <checkpoint dir> <out dir>`, CPU only) and `NINFER_TEST_ARTIFACT` (a
Qwen3.8-27B artifact, for the tests inherited from ninfer-tp2); Inkling 2 Mini's tests take the CMake cache paths
`NINFER_TEST_INKLING2_SYNTH_ARTIFACT` (a small synthetic checkpoint from `tools/ref_ink/synthetic.py`, converted) and
`NINFER_TEST_INKLING2_ARTIFACT` ([validating a change](docs/maintainer/inkling2-model.md#validating-a-change)). `ctest -j4` is supported; tests that need a model too
large for one 16 GB board skip with the sizes. For the 27B, `tools/tp2/gate.sh` is the reference run: it sets the
artifacts, runs the integration tests (`GATE_CTEST_EXCLUDE` excludes more), then checks golden ids, perplexity and
greedy texts against a recorded reference ([tools/tp2/README.md](tools/tp2/README.md)).
The app's tests run without a GPU: `cd app && python3 -m unittest discover -s tests`.

## What's new

**v0.16.4** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.16.4)): adaptive draft windows for MTP
and DFlash2, a streaming fix for benchmark clients; the defaults decode as in v0.16.3
- The adaptive MTP window (`--adaptive-draft N`, off by default) no longer loses on agent work
  ([PR #119](https://github.com/ValerioDolci/capy-flash/pull/119) by @Mici81, with three follow-up fixes). On the 64 GB
  Flash-Next edition, 2 × RTX 5070 Ti, against MTP 3: a 53-turn agent replay −3.7 % → +0.1 %, a recorded 43-turn
  coding-agent session +0.6 %, JSON +11-19 %, a code edit returned whole +22 %, quoting +14 %.
- `--adaptive-draft N` also works with DFlash2 on the 27B ([PR #115](https://github.com/ValerioDolci/capy-flash/pull/115)
  by @Mici81): JSON +23 %, a code edit +37 %, conversation unchanged, but −2.4 % on the agent session, so `--profile 27b`
  keeps 7 fixed drafts. Fixed on the way: with `--adaptive-draft` and two or more requests in a round, presence and
  frequency penalties counted the wrong tokens.
- Streamed Chat Completions send the assistant-role chunk with the first token, as OpenAI, vLLM and llama.cpp do. Clients
  that time the first chunk (llama-benchy) measured the request's admission instead of the prompt: 479k tok/s instead of
  8.5k on a 24k-token prompt. During a long prompt the stream sends `: keep-alive` comments.
- The converter stops on Qwen3.5-family checkpoints whose `tokenizer.json` was resaved by transformers 5 with the Qwen2
  pre-tokenizer pattern, and shows the two `--resource` overrides
  ([PR #118](https://github.com/ValerioDolci/capy-flash/pull/118) by @nnraiden).
- Same machine code as v0.16.3 (no kernel source changed); without the new flags the gate's texts are identical.

**v0.16.3** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.16.3)): Inkling 2 Mini decodes up to
1.9× and prefills up to 3.6× faster than v0.16.0, with the same weights; nothing changes for the Qwen models
- Same precision as before: the checkpoint's NVFP4 and MXFP8 experts as released, dense weights in BF16, KV cache and
  activations at 16 bits. Only the engine changed (the order of some sums, and some partial sums in FP16 or BF16, checked
  against the reference with the exact weights).
- Decode without the drafter 127.5 → 204.5 tok/s on a short prompt (a decode step takes 4.87 ms: reading the weights at the
  cards' 896 GB/s is now 65 % of the step, from 40 %). With the drafter, after a 4k-token prompt: writing code 230 → 308,
  an essay 149 → 205; summarizing a 130k-token document 73 → 140 ([benchmarks](#benchmarks)).
- Prefill 5,032 → 8,209 tok/s at 4k tokens, 2,522 → 6,610 at 99k, 1,413 → 5,078 at 247k.
- How: every selected expert is split across the two GPUs by its intermediate size (both cards read half of each, none
  waits for the other); wider expert kernels at prefill (128 × 128 tiles, 4-bit weights decoded in registers); a
  FlashAttention-2-style prefill attention; decode attention in one launch that reads the paged keys once; batched
  gates and router; dense weights prefetched into L2 next to the small kernels; prefill exchanges between the cards in
  BF16 and overlapped with compute by halves, as on the 27B.
- `capy-serve-inkling2-mini.sh` uses 4096-token prefill blocks and `--embedding-host` (the token embedding in one pinned
  host copy, identical results): the same 253,952-token context, prefill 6-13 % faster than with 2048-token blocks.
- The drafter now stays on up to 245,760 tokens of context (it stopped at 98,304): +12 % at 130k and +9 % at 196k
  over decoding without it.
- Quality: against the reference with the checkpoint's exact weights, the KL of the generated tokens is 0.99×
  v0.16.2's; over a 13k-token document the prompt's per-token KL and NLL move less than two valid prefill block sizes
  already make them move. The text can differ from v0.16.2 where two tokens are nearly tied.
- Qwen3.8-Flash-Next and Qwen3.8-27B run the same machine code as v0.16.2 (kernel fingerprint: only Inkling's kernels
  changed).

**v0.16.2** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.16.2)): the 27B decodes with
DFlash2 by default, prompt lookup on the one-GPU 27B
- `--profile 27b` (`serve-tp2.sh`) decodes with DFlash2 (7 drafts) when the artifact carries the DFlash2 drafter: the new
  file `qwen3_8_27b_quasar_nvfp4_dflash2.ninfer` (19.2 GB, the one `install.sh --model 27b` downloads; same weights plus
  the drafter). On 2 × RTX 5070 Ti against v0.16.1 (MTP 3), same weights: conversation +53 %, JSON +67 %, a code edit
  returned whole +98 %, quoting from 30K tokens +54 %, a 53-turn agent replay +14 %; an Italian essay −7 %. The 17.4 GB
  file without the drafter keeps MTP.
- Prompt lookup and an adaptive MTP draft window in one draft policy, `--prompt-lookup` and `--adaptive-draft N`, off by
  default ([PR #113](https://github.com/ValerioDolci/capy-flash/pull/113) by @Mici81). `--profile 27b-16gb` turns prompt
  lookup on: code edit +59 %, quoting +60 %, JSON +10 %, conversation, prose and the agent replay unchanged (one RTX 5070 Ti).
  On Flash-Next both are slower on the agent replay (adaptive window −4 %), so its profiles keep them off.

**v0.16.1** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.16.1)), six pull requests; same
tokens except the prompts of the 64 GB Flash-Next edition
- `POST /unload` and `POST /load` free the GPUs for another program and bring the model back without restarting the
  server (`--unload-endpoints`, `--idle-unload-seconds`, `--load-min-free-mib`, all off by default;
  [PR #116](https://github.com/ValerioDolci/capy-flash/pull/116) by @Mici81, [#95](https://github.com/ValerioDolci/capy-flash/issues/95)).
  Flash-Next on 2 × RTX 5070 Ti: unload 0.7 s, load 5-6 s with the host weights kept, about 0.4 GB left per card;
  the text after a reload is the same.
- 64 GB Flash-Next edition: the grouped gate/up GEMM of the prompt accumulates in FP16: prompts +1.8 % at 54k and
  +1.6 % at 114k tokens, perplexity +0.02 % (within noise); the greedy text of this edition changes
  ([PR #100](https://github.com/ValerioDolci/capy-flash/pull/100) by @Mici81).
- The 27B's exchanges launch with v0.14.1's four-range prefetch argument again: decode −0.11 % / −0.33 % ms per round
  (NVFP4 / 3-bit), same tokens ([PR #112](https://github.com/ValerioDolci/capy-flash/pull/112) by @Mici81).
- No more "CUDA Graph update rejected" line at every Flash-Next start, same tokens
  ([PR #111](https://github.com/ValerioDolci/capy-flash/pull/111) by @Mici81); an explicit `NINFER_HOST_MOE_CPUS` list
  now decides the GGUF gate/up work items ([PR #110](https://github.com/ValerioDolci/capy-flash/pull/110) by @Mici81).
- Second step of the native Windows build ([#43](https://github.com/ValerioDolci/capy-flash/issues/43)): TMA descriptors
  as kernel parameters under MSVC; Linux builds the same machine code
  ([PR #114](https://github.com/ValerioDolci/capy-flash/pull/114) by @mirifiuto135-debug).

**v0.16.0** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.16.0)), a third model; nothing
changes for the Qwen models
- **Inkling 2 Mini** on two 16 GB GPUs: chat, streaming, tool calls and thinking budgets over the OpenAI and
  Anthropic endpoints, from the original checkpoint without requantizing: its NVFP4 and MXFP8 experts as published,
  dense weights in BF16 ([details](#inkling-2-mini)). Text only, one request at a time.
- Speculative decoding for Inkling 2 Mini with the checkpoint's DSpark drafter (`--spec mtp --draft-tokens 3`,
  on in `capy-serve-inkling2-mini.sh`, `NINFER_DRAFTER=0` turns it off; the artifact needs `--components text,mtp`): greedy and sampled requests, the same text as without
  the drafter. Speed at temperature 0: 149.0 tok/s against 127.5 on a short prompt, 166.6 against 112.8 at 12,800
  tokens of context; beyond 98,304 tokens of context the engine takes ordinary steps by itself.
- Qwen3.8-Flash-Next and Qwen3.8-27B run the same kernels as v0.15.1 (kernel fingerprint: none changed, none
  removed); same tokens as v0.15.1: the 27B gate texts 60/60 and DFlash2 10/10 identical, golden ids and
  perplexity identical, the coding-agent replay 24/24 and a wave of 8 concurrent requests identical.

**v0.15.1** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.15.1)), two pull requests; same tokens
- GGUF expert dots in AVX-VNNI on AVX2 CPUs that have it (Intel since Alder Lake), same bits: decode +0.6 to +1.0 % on an
  i9-14900KF ([PR #105](https://github.com/ValerioDolci/capy-flash/pull/105) by @Mici81). CPUs with AVX-512 keep their kernels.
- First step of the native Windows build ([#43](https://github.com/ValerioDolci/capy-flash/issues/43)): the sources compile
  with MSVC, no OS code yet, so Windows does not build on its own; Linux builds the same machine code
  ([PR #106](https://github.com/ValerioDolci/capy-flash/pull/106) by @mirifiuto135-debug).

**v0.15.0** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.15.0)), faster Flash-Next prompts and seven pull requests by @Mici81; the 27B's tokens are unchanged, Flash-Next's greedy text can differ
- Flash-Next on one GPU: 16,384-token prompt chunks with the expert streaming ring in the elastic columns the KV is not
  using (a cold 15k prompt 2,656 → 4,678 tokens/s); after long conversations fill the KV, chunks fall back to 8,192.
- Flash-Next at two GPUs: more of the prompt overlaps (PLE gather, sliced mixer exchange, equal-width chunks, earlier
  expert copies) and the QSA block selection runs on FP16 tensor cores ([PR #98](https://github.com/ValerioDolci/capy-flash/pull/98),
  [PR #102](https://github.com/ValerioDolci/capy-flash/pull/102)): +18 % at 15k, +24 % at 54k, +25 % at 114k (NVFP4).
- MTP verification at any width from 1 to 16 gives a token the same bits, so `--draft-tokens` 3, 4 and 5 give the same
  text, and wider rounds cost less. Pull requests [#89](https://github.com/ValerioDolci/capy-flash/pull/89), [#90](https://github.com/ValerioDolci/capy-flash/pull/90),
  [#92](https://github.com/ValerioDolci/capy-flash/pull/92), [#99](https://github.com/ValerioDolci/capy-flash/pull/99) and [#104](https://github.com/ValerioDolci/capy-flash/pull/104), same bits.

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
