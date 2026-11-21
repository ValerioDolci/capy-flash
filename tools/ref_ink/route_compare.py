# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Routing comparison of two Inkling dumps (the route/ part of ref_dump.py; tf_compare.py covers logits and hidden).

  route_compare.py REF_DUMP ENG_DUMP [--json out.json]

route/V<vv>.idx.npy int32 [R, k] (ascending expert id), route/V<vv>.w.npy f32 [R, k] (same order),
route/V<vv>.logits.npy f32 [R, E] (optional), route/rows.npy int64 [R]. Rows are matched by sequence position.
Per visit: rows whose expert set differs, mean overlap |A & B| / k, max |d weight| over the rows with the same set,
max |d router logit|. Exit 0 always (a report: routing flips at near-ties are expected at BF16, see 04-riferimento.md).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load(d):
    r = Path(d) / "route"
    rows = np.load(r / "rows.npy")
    visits = sorted({m.group(1) for f in r.iterdir() if (m := re.match(r"(V\d+)\.idx\.npy$", f.name))})
    return rows, visits, r


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ref")
    p.add_argument("eng")
    p.add_argument("--json")
    a = p.parse_args()
    ra, va, da = load(a.ref)
    rb, vb, db = load(a.eng)
    rows, ia, ib = np.intersect1d(ra, rb, return_indices=True)
    report = {"rows": int(len(rows)), "visits": {}}
    for v in [v for v in va if v in set(vb)]:
        A, B = np.load(da / f"{v}.idx.npy")[ia], np.load(db / f"{v}.idx.npy")[ib]
        WA, WB = np.load(da / f"{v}.w.npy")[ia], np.load(db / f"{v}.w.npy")[ib]
        k = A.shape[1]
        same = (A == B).all(1)
        overlap = np.mean([len(set(x) & set(y)) / k for x, y in zip(A, B)])
        rec = {
            "set_differs": int((~same).sum()),
            "first_rows": [int(r) for r in rows[~same][:5]],
            "overlap": round(float(overlap), 5),
            "dw_max": float(np.abs(WA[same].astype(np.float64) - WB[same]).max()) if same.any() else None,
        }
        la, lb = da / f"{v}.logits.npy", db / f"{v}.logits.npy"
        if la.exists() and lb.exists():
            rec["dlogit_max"] = float(np.abs(np.load(la)[ia].astype(np.float64) - np.load(lb)[ib]).max())
        report["visits"][v] = rec
        dw = "n/a" if rec["dw_max"] is None else f"{rec['dw_max']:.3e}"
        dl = f" |drouter| max {rec['dlogit_max']:.3e}" if "dlogit_max" in rec else ""
        print(
            f"{v}: set differs on {rec['set_differs']:4d}/{len(rows)} rows {rec['first_rows']} "
            f"overlap {rec['overlap']:.4f} |dw| max {dw}{dl}"
        )
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
