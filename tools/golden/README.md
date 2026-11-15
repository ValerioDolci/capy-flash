# tp 1 golden: token ids on one device

The `golden` stage of the gate ([`tools/tp2/gate.sh`](../tp2/README.md)): a fixed runner generates token ids greedily
at `--tp 1`, and they must equal the ones of the recorded reference.

## What runs

- `tp1_golden.cpp` — loads an artifact on one device through the public `ninfer::Engine` API,
  prepares a deterministic byte-id prompt (`prepare_tokens`, no chat template, no prefix reuse),
  decodes it greedily (temperature 0) and prints the generated ids. Public headers only.
- `record.sh` — the three cases, chosen for their prefill regimes rather than their content:

  | case | prompt tokens | prefill chunks (1024) | extra flags |
  |---|---:|---:|---|
  | 1 short | 32 | 1 | `--max-context 4096` |
  | 2 medium | 2,048 | 2 | `--max-context 4096` |
  | 3 long | 6,144 | 6 | `--max-context 8192 --kv-dtype int8` |

  Common set: `--seed 7 --max-new 128`. `diff -r` between two recorded directories is the gate.
  `GOLDEN_FLAGS` is appended to every case (the tp 2 gate passes `--embedding-host` this way, see
  `GATE_EXTRA_FLAGS` in `tools/tp2/gate.sh`).

## The artifact

The gate runs the synthetic two-layer Qwen3.5 model of `tests/models/qwen3_5/test_text_context_tp2.cpp`
(one Gated DeltaNet block, one full-attention block, FP8 projections with BF16 norms, a byte-level
tokenizer). That executable writes it:

```bash
./build/tests/ninfer_qwen3_5_text_context_tp2_test write /tmp/synthetic   # -> /tmp/synthetic/model.ninfer
```

Scope: the Engine, scheduler, prefill/decode programs, attention, GDN, FP8 and BF16 Ops on one
device. Not covered: the NVFP4 and INT-quantised GEMMs. `record.sh` takes a real artifact as-is and
covers its GEMMs too: the 3-bit 27B (`--model 27b-16gb`, 12.5 GB) fits one 16 GB board, an NVFP4 27B
(17.4 GB and up) needs a larger one.

The runner: `ninja -C build ninfer-tp1-golden` (registered in `apps/CMakeLists.txt`).
