# capy-flash — prebuilt binaries

capy-flash (formerly ninfer-tp2) serves **Qwen3.8-Flash-Next** (MoE, NVFP4 experts, CPU offload) and **Qwen3.8-27B**
on two 16 GB Blackwell GeForce boards; since v0.7 also Flash-Next on 64 GB of RAM and the 27B (3-bit) on one board;
since v0.16 also **Inkling 2 Mini** (text only) on two 16 GB boards. Source and documentation: https://github.com/ValerioDolci/capy-flash.
`COMMIT` records the exact commit.

## Contents

| Path | What |
|---|---|
| `bin/ninfer-serve` | OpenAI- and Anthropic-compatible server |
| `bin/ninfer` | command-line generation |
| `bin/ninfer-perplexity` | perplexity evaluator |
| `bin/capy-serve`, `bin/capy`, `bin/capy-perplexity` | the same three programs under their capy-flash names (scripts that also put `lib/` on the library path) |
| `lib/libcudart.so.13` | the CUDA runtime the binaries link (NVIDIA redistributable) |
| `serve-flash-next.sh` | launcher with the Qwen3.8-Flash-Next profile |
| `serve-tp2.sh` | launcher with the Qwen3.8-27B two-GPU production profile (as in ninfer-tp2) |
| `serve-27b-16gb.sh` | launcher for the 3-bit Qwen3.8-27B on one 16 GB board (`--profile 27b-16gb`) |
| `capy-serve-inkling2-mini.sh` | launcher for Inkling 2 Mini on two 16 GB boards (text only, one request at a time, 253,952-token context with 2048-token prefill blocks, its drafter on) |
| `chat_templates/` | `qwen3_8.jinja` (NInfer's Qwen3.8 chat template, which the launchers pass with `--chat-template`; it accepts the `developer` role and Responses `instructions`, needed by Codex CLI) and its `LICENSE` |
| `SHA256SUMS` | digests of every file above; check with `sha256sum -c SHA256SUMS` |

## Requirements

- Linux x86_64 with the distribution named in the tarball (the binaries link its FFmpeg and libcurl:
  `libavcodec`, `libavformat`, `libavutil`, `libswscale`, `libcurl4`). On another distribution
  build from source or use the Dockerfile.
- An NVIDIA driver that supports CUDA 13.1 (R580 or newer). No CUDA toolkit is needed.
- Two `sm_120` boards with 16 GB for `--tp 2` (one for the 3-bit 27B or the 64 GB edition of Flash-Next) (validated on 2x RTX 5070 Ti without P2P; with P2P, for example through
  a PCIe switch, the prefill exchanges use the direct copy automatically).
- **Qwen3.8-Flash-Next** also needs a CPU with AVX-512 (F/BW/VL, VNNI) or AVX2 + FMA + F16C (the expert kernels of
  both give the same results bit for bit) and about **75 GiB of free RAM**: the routed
  experts that stay on the host are one 64.6 GiB page-locked allocation, `--profile flash-next` keeps the token
  embedding and up to 4 GiB of idle conversations' KV in RAM as well, and the server refuses to start without room for
  the experts plus a reserve (`NINFER_RAM_GUARD=0` skips the check) or for the RAM caches plus a reserve
  (`NINFER_HOST_TIER_RAM_GUARD=0`); `install.sh` warns below 75 GiB free. In practice a 96 GB machine; the 64 GB edition (GSQ-RCO
  experts, 48 GiB) needs about 56 GiB and fits a 64 GB machine.

## Run

```bash
tar -xzf capy-flash-*.tar.gz && cd capy-flash-*/
sha256sum -c SHA256SUMS
./serve-flash-next.sh /path/to/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer          # Qwen3.8-Flash-Next
./serve-flash-next.sh /path/to/vision-artifact.ninfer --vision                    # Flash-Next with images (opt-in)
./serve-flash-next.sh /path/to/qwen3_8_flash_next_gsqrco_mtpB.ninfer --host-kv-mib 1024 --host-state-slots 2   # 64 GB edition
NINFER_DEVICES=0 ./serve-flash-next.sh /path/to/qwen3_8_flash_next_gsqrco_mtpB.ninfer                           # 64 GB edition, one GPU
./serve-tp2.sh /path/to/qwen3_8_27b_quasar_nvfp4.ninfer                           # Qwen3.8-27B
./serve-27b-16gb.sh /path/to/qwen3_8_27b_gsq3.ninfer                              # Qwen3.8-27B 3-bit, one GPU
./capy-serve-inkling2-mini.sh /path/to/capy-flash-inkling2-mini-nvfp4.ninfer       # Inkling 2 Mini, two GPUs
NINFER_DRAFTER=0 ./capy-serve-inkling2-mini.sh /path/to/capy-flash-inkling2-mini-nvfp4.ninfer                 # without its drafter
NINFER_PORT=8099 ./serve-flash-next.sh /path/to/model.ninfer --api-key secret      # extra flags are appended
```
The launchers add `--chat-template chat_templates/qwen3_8.jinja`, the model id and the profile; extra flags win
over the profile (the Inkling launcher passes no template or profile: its artifact carries its own template, and it
passes `--spec mtp --draft-tokens 3`: the drafter of an artifact converted with `--components text,mtp`, with the
same text as without it; an artifact without the drafter decodes without it). Requests
name the model `capy-flash` (Flash-Next), `qwen3.8-27b` (27B) or `inkling-2-mini`; `NINFER_MODEL_ID` or
`--model-id` changes it:
```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"capy-flash","messages":[{"role":"user","content":"What is 17*23?"}]}'
```

Artifacts: `install.sh --model NAME` downloads one Hugging Face repository; the models, their sizes and the RAM and
disk they need are in the main README (https://github.com/ValerioDolci/capy-flash#pick-a-model).

Vision: the two 27B artifacts contain the vision tower and `serve-tp2.sh` / `serve-27b-16gb.sh` load it by default
(it takes VRAM from the KV pool); `NINFER_VISION=0` before the launcher, or `--no-vision` after the artifact, turns it
off. Flash-Next loads no tower unless `--vision` is given with the Vision artifact.

Clients: Claude Code with `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`; Codex CLI, OpenCode and other OpenAI clients
with base URL `http://127.0.0.1:8080/v1` and the model id above (Codex: a custom provider with the `responses` wire
API). Details: `docs/serving/capy-flash.md` in the source repository.

`GET /health` answers 200 while the Engine accepts work and 503 after an Engine-wide failure; the
server then exits with status 2, so run it under a supervisor that restarts on a non-zero exit
(`Restart=on-failure`, or llama-swap, which reloads the model on the next request).
