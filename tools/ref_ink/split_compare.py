# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""tf_compare-style comparison of two Inkling dumps with prompt rows and generated rows apart, against a floor.

  split_compare.py REF ENG --prompt-len P [--floor REF2] [--k 2] [--tie 0.3] [--json out.json]

Rows r < P-1 predict prompt tokens ("prompt"), rows r >= P-1 predict generated tokens ("gen"); --prompt-len may be
a greedy.py record (its n_prompt). Logits: per-row KL(ref||eng) in FP64, top-1 agreement, every top-1 discordance
with the reference margin (top-1 minus top-2 logit). Hidden: per visit rel L2 = ||d||/||ref|| on each row set.
--floor REF2: the same reference variant computed in another valid order (e.g. ref_dump.py --chunk 16); the floor is
REF against REF2 on the same rows, and the report gives ENG/floor per visit and the first visit (and its worst row)
where ENG exceeds K x floor (floor of a visit = the larger of its prompt-row and generated-row values).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "ref_q4"))
import tfdump  # noqa: E402


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))


def logit_report(ref, eng, p, tie):
    (ra, A), (rb, B) = ref.get("logits"), eng.get("logits")
    rows, ia, ib = np.intersect1d(ra, rb, return_indices=True)
    out = {}
    for name, mask in (("prompt", rows < p - 1), ("gen", rows >= p - 1)):
        if not mask.any():
            continue
        kl, mism = [], []
        for r, a, b in zip(rows[mask], A[ia][mask], B[ib][mask]):
            lp, lq = logsoftmax(a), logsoftmax(b)
            kl.append(max(float(np.sum(np.exp(lp) * (lp - lq))), 0.0))
            o = np.argsort(a)[-2:][::-1]
            if int(o[0]) != int(np.argmax(b)):
                mism.append((int(r), round(float(a[o[0]] - a[o[1]]), 4)))
        kl = np.asarray(kl)
        out[name] = {
            "rows": int(mask.sum()),
            "kl_mean": float(kl.mean()),
            "kl_max": float(kl.max()),
            "kl_max_row": int(rows[mask][int(kl.argmax())]),
            "top1": 1.0 - len(mism) / int(mask.sum()),
            "mismatch": mism,
            "decisive": [m for m in mism if m[1] > tie],
        }
    return out


def visit_names(d):
    def key(n):
        m = re.match(r"hidden\.V(\d+)$", n)
        return int(m.group(1)) if m else 99

    return sorted((n for n in d.items if re.match(r"hidden\.(V\d+|embed|final)$", n)), key=key)


def hidden_report(ref, eng, p):
    out = {}
    for n in visit_names(ref):
        if not eng.get(n):
            continue
        (ra, A), (rb, B) = ref.get(n), eng.get(n)
        rows, ia, ib = np.intersect1d(ra, rb, return_indices=True)
        A, B = A[ia].astype(np.float64), B[ib].astype(np.float64)
        rec = {}
        for name, mask in (("prompt", rows < p - 1), ("gen", rows >= p - 1)):
            if mask.any():
                d = A[mask] - B[mask]
                row_rel = np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(A[mask], axis=1), 1e-30)
                rec[name] = {
                    "rel": float(np.linalg.norm(d) / max(np.linalg.norm(A[mask]), 1e-30)),
                    "worst_row": int(rows[mask][int(row_rel.argmax())]),
                    "worst_row_rel": float(row_rel.max()),
                }
        out[n.split(".", 1)[1]] = rec
    return out


def main():
    pa = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    pa.add_argument("ref")
    pa.add_argument("eng")
    pa.add_argument("--prompt-len", required=True)
    pa.add_argument("--floor")
    pa.add_argument("--k", type=float, default=2.0)
    pa.add_argument("--tie", type=float, default=0.3)
    pa.add_argument("--json")
    a = pa.parse_args()
    p = a.prompt_len
    p = json.loads(Path(p).read_text())["n_prompt"] if Path(p).exists() else int(p)
    ref, eng = tfdump.load(a.ref), tfdump.load(a.eng)
    rep = {"ref": a.ref, "eng": a.eng, "prompt_len": p, "logits": logit_report(ref, eng, p, a.tie)}
    rep["hidden"] = hidden_report(ref, eng, p)
    if a.floor:
        fl = tfdump.load(a.floor)
        rep["floor_logits"] = logit_report(ref, fl, p, a.tie)
        rep["floor_hidden"] = hidden_report(ref, fl, p)
    for name in ("prompt", "gen"):
        lg = rep["logits"].get(name)
        if not lg:
            continue
        line = (
            f"logits {name:6s} rows {lg['rows']:3d}: KL mean {lg['kl_mean']:.3e} max {lg['kl_max']:.3e} "
            f"(row {lg['kl_max_row']}) top1 {lg['top1']:.4f} mismatches (row, ref margin) {lg['mismatch'][:8]}"
        )
        if a.floor:
            f = rep["floor_logits"][name]
            line += (
                f" | floor KL mean {f['kl_mean']:.3e} max {f['kl_max']:.3e} top1 {f['top1']:.4f} {f['mismatch'][:6]}"
            )
        print(line)
    first = {}
    for name in ("prompt", "gen"):
        cells = []
        for v, rec in rep["hidden"].items():
            if name not in rec:
                continue
            r = rec[name]["rel"]
            if a.floor and rep["floor_hidden"].get(v):
                # floor of the visit: the larger of its prompt-row and generated-row floors (a few prompt rows can
                # be bit-identical between two orders, which would make a zero floor)
                f = max(x["rel"] for x in rep["floor_hidden"][v].values())
                cells.append(f"{v}:{r:.2e}/{f:.2e}")
                if name not in first and r > a.k * f:
                    first[name] = (v, r, f, rec[name]["worst_row"], rec[name]["worst_row_rel"])
            else:
                cells.append(f"{v}:{r:.2e}")
        if cells:
            print(f"hidden {name} rel" + (" (eng / floor)" if a.floor else "") + ": " + " ".join(cells))
    if a.floor:
        for name in ("prompt", "gen"):
            if name in first:
                v, r, f, row, rr = first[name]
                print(
                    f"first visit above {a.k:g} x floor on {name} rows: {v} rel {r:.3e} floor {f:.3e} worst row {row}"
                    f" ({rr:.3e})"
                )
            else:
                print(f"{name} rows: every visit within {a.k:g} x floor")
        rep["first_above_floor"] = {k: list(v) for k, v in first.items()}
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
