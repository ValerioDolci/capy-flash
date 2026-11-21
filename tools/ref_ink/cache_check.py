# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Cache path against the one-pass forward of the transformers reference, on the same tokens.

  cache_check.py --checkpoint DIR --impl fp64|fp32|bf16 --tokens IDS --chunks 64,1 [--threads N]

One forward without cache gives the logits of every position; then, per chunk size C, the sequence goes through a
DynamicCache C tokens at a time (C = 1 is token-by-token decoding) and the logits of every chunk are collected.
Reports max |d logit| and top-1 agreement against the one-pass logits. In fp64 the two must agree to rounding: a
larger gap is a semantic difference of the cache path (sliding-window cache, convolution states, relative distances,
loop-pass cache copies), not noise.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkref  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--impl", choices=["bf16", "bf16-hf", "fp32", "fp64"], default="fp64")
    p.add_argument("--tokens", required=True)
    p.add_argument("--chunks", default="64,1")
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()
    inkref.set_threads(a.threads)
    from transformers import DynamicCache

    ids = inkref.read_ids(a.tokens)
    model, _ = inkref.load_model(a.checkpoint, a.impl)
    x = torch.tensor([ids])
    mode = inkref.Promote64() if a.impl == "fp64" else None
    ok = True
    with torch.no_grad():
        if mode:
            mode.__enter__()
        ref = model(x, use_cache=False).logits[0].double()
        for c in [int(v) for v in a.chunks.split(",")]:
            t0 = time.time()
            cache = DynamicCache(config=model.config)
            parts = [model(x[:, s : s + c], cache=cache, use_cache=True).logits[0] for s in range(0, len(ids), c)]
            got = torch.cat(parts).double()
            d = (got - ref).abs()
            row = d.max(-1).values
            agree = float((got.argmax(-1) == ref.argmax(-1)).double().mean())
            lp, lq = torch.log_softmax(ref, -1), torch.log_softmax(got, -1)
            kl = (lp.exp() * (lp - lq)).sum(-1).clamp(min=0)
            worst = int(row.argmax())
            print(
                f"{a.impl} T={len(ids)} chunk {c}: max|dlogit| {float(d.max()):.3e} (pos {worst}) mean "
                f"{float(d.mean()):.3e} KL mean {float(kl.mean()):.3e} max {float(kl.max()):.3e} top1 agree {agree:.4f} logit rms {float(ref.pow(2).mean().sqrt()):.3f} "
                f"({time.time() - t0:.1f} s)",
                flush=True,
            )
            if a.impl == "fp64" and float(d.max()) > 1e-6:
                ok = False
        if mode:
            mode.__exit__(None, None, None)
    print("CACHE", "OK" if ok else "DIFFERS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
