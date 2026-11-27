"""The simulator (policy.py) against the engine's code (drafter_choice.cpp through the drafter_replay binary), step by step,
on recorded traces, with the constants and the cost table of drafter_tuning.h and no measured costs. Run after a change to
either side; tune.py's predictions hold only while they agree.

Usage: python3 parity.py --replay build/tests/ninfer_inkling2_drafter_replay --header src/.../drafter_tuning.h
       --data DIR [--conditions off,on] [--k 3]"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import walk  # noqa: E402
from tune import load  # noqa: E402


def constants(header):
    text = Path(header).read_text()
    c = {}
    for name, value in re.findall(r"inline constexpr (?:std::uint32_t|double|bool) (k\w+)\s*=\s*([^;]+);", text):
        value = value.strip()
        c[name] = {"true": 1, "false": 0}.get(value, None)
        if c[name] is None:
            c[name] = float(value) if "." in value else int(value)
    knots = [tuple(float(x) for x in row.split(",")) for row in re.findall(r"\{(\d+, [\d.]+, [\d.]+, [\d.]+)\}", text)]
    return c, knots


class TableCosts:
    """StepCosts without measurements: the table, linear between knots, constant below, extrapolated beyond."""

    def __init__(self, knots):
        self.knots = knots

    def _at(self, column, p):
        k = self.knots
        i = 0
        while i + 2 < len(k) and p >= k[i + 1][0]:
            i += 1
        t = max(0.0, (p - k[i][0]) / (k[i + 1][0] - k[i][0]))
        return k[i][column] + (k[i + 1][column] - k[i][column]) * t

    def round(self, d, p, s):
        return self._at(1 if d == "dsp" else 2, p)

    def step(self, p, s):
        return self._at(3, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--header", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--conditions", default="off,on")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    c, knots = constants(a.header)
    costs = TableCosts(knots)
    total = differ = 0
    for cond in a.conditions.split(","):
        for t in load(Path(a.data) / cond):
            steps = []
            walk(t, costs, c, a.k, steps)
            n = t.n
            trace = {
                "seq": t.seq,
                "n_prompt": n,
                "stopped": t.stopped,
                "k": a.k,
                "drafts": {d: [t.drafts[d][p][: a.k] for p in range(n, len(t.seq) - 1)] for d in ("dsp", "df2")},
            }
            with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
                json.dump(trace, f)
                f.flush()
                engine = subprocess.run([a.replay, f.name], check=True, capture_output=True, text=True).stdout.split()
            total += 1
            if engine != steps:
                differ += 1
                first = next((i for i, (x, y) in enumerate(zip(engine, steps)) if x != y), min(len(engine), len(steps)))
                print(
                    f"{cond}/{t.name}: differ at step {first}: engine {engine[first : first + 6]} simulator "
                    f"{steps[first : first + 6]}"
                )
    print(f"parity: {total - differ}/{total} traces step for step identical")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
