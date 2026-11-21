# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Bitwise comparison of two dump trees (ref_dump.py outputs) or two greedy.py output directories.

  bitcmp.py DIR_A DIR_B

Every .npy file present in both is compared byte for byte; for the ones that differ: max |d| and, for logits, top-1
agreement. greedy.py JSON records are compared on prompt ids, generated ids and per-step logits/margins.
Exit 0 = everything identical.
"""

import json
import sys
from pathlib import Path

import numpy as np


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    same = diff = 0
    worst = []
    for fa in sorted(a.rglob("*.npy")):
        fb = b / fa.relative_to(a)
        if not fb.exists():
            continue
        if fa.read_bytes() == fb.read_bytes():
            same += 1
            continue
        diff += 1
        x, y = np.load(fa), np.load(fb)
        d = float(np.abs(x.astype(np.float64) - y).max()) if x.shape == y.shape else float("inf")
        extra = ""
        if fa.name == "logits.npy" and x.shape == y.shape:
            extra = f" top1 agree {float((x.argmax(-1) == y.argmax(-1)).mean()):.4f}"
        worst.append((d, str(fa.relative_to(a)) + extra))
    for fa in sorted(a.rglob("*.json")):
        fb = b / fa.relative_to(a)
        if not fb.exists() or fa.name == "meta.json":
            continue
        ra, rb = json.loads(fa.read_text()), json.loads(fb.read_text())
        keys = ("prompt_ids", "generated_ids", "steps")
        if all(ra.get(k) == rb.get(k) for k in keys):
            same += 1
        else:
            diff += 1
            ga, gb = ra.get("generated_ids", []), rb.get("generated_ids", [])
            first = next((i for i, (u, v) in enumerate(zip(ga, gb)) if u != v), None)
            worst.append((float("nan"), f"{fa.relative_to(a)} generated ids first differ at step {first}"))
    print(f"identical {same}, different {diff}")
    for d, name in sorted(worst, key=lambda t: -t[0] if t[0] == t[0] else 0)[:15]:
        print(f"  {name}: max|d| {d:.3e}")
    sys.exit(0 if diff == 0 else 1)


if __name__ == "__main__":
    main()
