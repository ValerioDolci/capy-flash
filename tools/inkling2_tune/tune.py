"""Step 3 of tune.sh: the choice of --spec auto tuned on recorded traces (docs/maintainer/inkling2-auto-tuning.md).

Reads DATA/<condition>/{ids,drafts_dsp,drafts_df2} (record.py and the tf driver's --mode drafts) for each condition
(LoopCD off, on), fits the step costs of each condition, simulates the choice (policy.py) over a grid of its constants,
keeps the constants whose worst group is best (against the better fixed drafter and against DSpark), writes them with the
cost table into drafter_tuning.h (and drafter_tuning.json beside it: the grid's best rows, the traces and costs used) and
prints the expected decode rate of every group with them against the fixed drafters and no drafter.

Usage: python3 tune.py --data DIR [--conditions off,on] [--k 3] [--write .../drafter_tuning.h] [--table-condition off]"""

import argparse
import bisect
import hashlib
import itertools
import json
import multiprocessing
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import fixed, walk  # noqa: E402

# Following costs (ms per round) of the drafter that does not propose, measured on 2 x RTX 5070 Ti (they do not change the
# ranking), and the fed ordinary step's cost over a plain one.
FOLLOW = {"dsp": 0.083, "df2": 0.045}  # proposer -> the other's follow
FED = 0.03
KNOTS = (512, 4096, 8192, 16384, 32768, 131072, 196608)
# The criterion: auto at least the better fixed drafter less this share, and DSpark (ft3) less this one.
BEST_SHARE, FT3_SHARE = 0.02, 0.01

GRID = {
    "kWindow": [256, 2048],
    "kRecent": [16],
    "kLongPrompt": [2048],
    "kEnter": [0.1, 0.2],
    "kLeave": [0.0, 0.05],
    "kWeight": [0.05],
    "kEvidence": [3],
    "kMargin": [0.09, 0.15],
    "kProbe": [16],
    "kCompare": [1],
    "kHysteresis": [0.0, 0.1],
    "kPriorWins": [1],
    "kStart": [0, 4, 8],
    "kExplore": [32, 64],
    "kOtherWeight": [0.5, 0.75],
    "kSettle": [1.0, 0.3],
}


class Text:
    def __init__(self, ids, dsp, df2):
        self.name, self.group, self.sampled = ids["name"], ids["group"], ids["sampled"]
        self.n = len(ids["prompt_ids"])
        self.seq = ids["prompt_ids"] + ids["generated_ids"]
        self.stopped = ids["stopped"]
        self.step_ms = ids["step_ms"]
        self.round_ms = {"dsp": dsp["round_ms"], "df2": df2["round_ms"]}
        self.context = self.n + len(ids["generated_ids"]) / 2
        self.drafts = {}
        for name, d in (("dsp", dsp), ("df2", df2)):
            if d["tokens"] != self.seq or d["n_prompt"] != self.n:
                raise SystemExit(f"{self.name}: drafts of another text ({name})")
            self.drafts[name] = dict(zip(d["anchors"], d["drafts"]))

    def accepted(self, d, p, extent):
        drafts, a = self.drafts[d][p], 0
        while a < extent and drafts[a] == self.seq[p + 1 + a]:
            a += 1
        return a


class Line:
    """Piecewise-linear in the context through the mean of the points of each context (constant before the first,
    extrapolated after the last)."""

    def __init__(self, points):
        by = defaultdict(list)
        for x, y in points:
            by[round(x, -2) if x >= 2000 else 0].append((x, y))
        self.rows = sorted((st.mean(x for x, _ in v), st.mean(y for _, y in v)) for v in by.values())

    def __call__(self, x):
        rows, xs = self.rows, [r[0] for r in self.rows]
        if len(rows) == 1:
            return rows[0][1]
        i = bisect.bisect_left(xs, x)
        if i == 0:
            return rows[0][1]
        (x0, y0), (x1, y1) = (rows[-2], rows[-1]) if i == len(rows) else (rows[i - 1], rows[i])
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class Costs:
    """Milliseconds by context of a condition: rounds of each drafter (the driver's, greedy), plain steps (the recording
    server's, greedy and sampled); a sampled round pays 1.8 times a step's sampling overhead (measured: host sampling
    over K + 1 columns against one)."""

    def __init__(self, texts):
        self.rounds = {d: Line([(t.context, t.round_ms[d]) for t in texts]) for d in ("dsp", "df2")}
        greedy = [(t.context, t.step_ms) for t in texts if not t.sampled and t.step_ms]
        sampled = [(t.context, t.step_ms) for t in texts if t.sampled and t.step_ms]
        self.steps = Line(greedy)
        line = Line(sampled) if sampled else None
        self.sampling = st.mean(y - self.steps(x) for x, y in sampled) if line else 0.0

    def plain_step(self, p, s):
        return self.steps(p) + (self.sampling if s else 0.0)

    def step(self, p, s):
        return self.plain_step(p, s) + FED

    def single_round(self, d, p, s):
        return self.rounds[d](p) + (1.8 * self.sampling if s else 0.0)

    def round(self, d, p, s):
        return self.single_round(d, p, s) + FOLLOW[d]


def load(root):
    texts = []
    for f in sorted((root / "ids").glob("*.json")):
        texts.append(
            Text(
                json.loads(f.read_text()),
                json.loads((root / "drafts_dsp" / f.name).read_text()),
                json.loads((root / "drafts_df2" / f.name).read_text()),
            )
        )
    return texts


def key(t):
    return f"{t.group}_{t.name.rsplit('_', 1)[1]}" if t.group in ("summary", "essay") else t.group


def rates(texts, costs, fn):
    acc = defaultdict(lambda: [0.0, 0])
    for t in texts:
        g = acc[key(t)]
        g[0] += fn(t)
        g[1] += len(t.seq) - t.n
    return {k: 1e3 * v[1] / v[0] for k, v in acc.items()}


def evaluate(data, c, K):
    """Per condition and group: auto, DSpark, DFlash2, none (tok/s); the worst margins of the criterion."""
    out, worst_best, worst_ft3 = {}, 1e9, 1e9
    for cond, (texts, costs, base) in data.items():
        auto = rates(texts, costs, lambda t: walk(t, costs, c, K)[0])
        out[cond] = {g: (auto[g],) + base[g] for g in auto}
        for g in auto:
            dsp, df2, _ = base[g]
            worst_best = min(worst_best, auto[g] / max(dsp, df2) - 1.0)
            worst_ft3 = min(worst_ft3, auto[g] / dsp - 1.0)
    return out, worst_best, worst_ft3


_DATA, _K = None, 3


def _score(c):
    _, wb, wf = evaluate(_DATA, c, _K)
    return wb, wf


def header(c, costs, K):
    knots = ",\n".join(
        f"    {{{x}, {costs.round('dsp', x, False):.2f}, {costs.round('df2', x, False):.2f}, "
        f"{costs.plain_step(x, False) + FED:.2f}}}"
        for x in KNOTS
    )
    return f"""// Generated by tools/inkling2_tune: the constants of --spec auto's choice (drafter_choice.h) and the
// initial step costs (drafter_costs.h). Regenerate with the tool instead of editing; what they were
// measured on is in drafter_tuning.json beside this file (docs/maintainer/inkling2-auto-tuning.md).
#pragma once

#include <array>
#include <cstdint>

namespace ninfer::models::inkling2::detail::tuning {{

// The far-copy prior: the DFlash2 window, the generated tokens the share is taken over, the prompt
// length from which a request starts with DSpark, and the shares that move it to DSpark and back.
inline constexpr std::uint32_t kWindow     = {c["kWindow"]};
inline constexpr std::uint32_t kRecent     = {c["kRecent"]};
inline constexpr std::uint32_t kLongPrompt = {c["kLongPrompt"]};
inline constexpr double kEnter             = {c["kEnter"]};
inline constexpr double kLeave             = {c["kLeave"]};

// The measured choice: the weight of each drafter's mean of accepted drafts, the rounds before it
// counts, whether the better measured rate chooses (with this hysteresis) and whether far copies still
// force DSpark; on long prompts 2 kStart alternating rounds, then a round of the other drafter every
// kExplore rounds (0: never) weighing kOtherWeight, until the other is below the chosen by kSettle.
inline constexpr double kWeight          = {c["kWeight"]};
inline constexpr std::uint32_t kEvidence = {c["kEvidence"]};
inline constexpr bool kCompare           = {"true" if c["kCompare"] else "false"};
inline constexpr double kHysteresis      = {c["kHysteresis"]};
inline constexpr bool kPriorWins         = {"true" if c["kPriorWins"] else "false"};
inline constexpr std::uint32_t kStart    = {c["kStart"]};
inline constexpr std::uint32_t kExplore  = {c["kExplore"]};
inline constexpr double kOtherWeight     = {c["kOtherWeight"]};
inline constexpr double kSettle          = {c["kSettle"]};

// No drafter: the margin and the ordinary steps between two rounds while it holds.
inline constexpr double kMargin       = {c["kMargin"]};
inline constexpr std::uint32_t kProbe = {c["kProbe"]};

// Milliseconds of a DSpark round, a DFlash2 round (K = {K}) and a fed ordinary step at a context.
struct CostKnot {{
    std::uint32_t context;
    double round_mtp;
    double round_dflash2;
    double step;
}};

inline constexpr std::array<CostKnot, {len(KNOTS)}> kCosts{{{{
{knots},
}}}};

}} // namespace ninfer::models::inkling2::detail::tuning
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--conditions", default="off,on")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--write")
    ap.add_argument("--table-condition", default="off")
    ap.add_argument("--grid", help="JSON object overriding GRID entries")
    ap.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    a = ap.parse_args()
    grid = dict(GRID, **(json.loads(a.grid) if a.grid else {}))
    root = Path(a.data)
    data = {}
    for cond in a.conditions.split(","):
        texts = load(root / cond)
        costs = Costs(texts)
        base = {}
        dsp = rates(texts, costs, lambda t: fixed(t, costs, "dsp", a.k))
        df2 = rates(texts, costs, lambda t: fixed(t, costs, "df2", a.k))
        none = rates(texts, costs, lambda t: fixed(t, costs, None, a.k))
        for g in dsp:
            base[g] = (dsp[g], df2[g], none[g])
        data[cond] = (texts, costs, base)
    names = list(grid)
    configs = []
    for values in itertools.product(*(grid[n] for n in names)):
        c = dict(zip(names, values))
        if not c["kCompare"] and (c["kHysteresis"] or c["kPriorWins"]):
            continue
        if not c["kExplore"] and not c["kStart"] and c["kSettle"] < 1.0:
            continue
        configs.append(c)
    global _DATA, _K
    _DATA, _K = data, a.k
    with multiprocessing.get_context("fork").Pool(a.jobs) as pool:
        scored = pool.map(_score, configs, chunksize=4)
    rows = [(min(wb + BEST_SHARE, wf + FT3_SHARE), wb, wf, c) for (wb, wf), c in zip(scored, configs)]
    rows.sort(key=lambda r: -r[0])
    best = rows[0][3]
    table, wb, wf = evaluate(data, best, a.k)
    print(
        f"{len(rows)} configurations; best: worst against the better drafter {100 * wb:+.1f} %, against DSpark "
        f"{100 * wf:+.1f} % (criterion -{100 * BEST_SHARE:.0f} / -{100 * FT3_SHARE:.0f} %)"
    )
    print(json.dumps(best))
    for cond, groups in table.items():
        print(f"== LoopCD {cond}: tok/s auto | DSpark | DFlash2 | none, auto against the better drafter and DSpark")
        for g in sorted(groups):
            auto, dsp, df2, none = groups[g]
            flag = "" if auto >= max(dsp, df2) * (1 - BEST_SHARE) and auto >= dsp * (1 - FT3_SHARE) else "  <-"
            print(
                f"  {g:16s} {auto:7.1f} {dsp:7.1f} {df2:7.1f} {none:7.1f}  {100 * (auto / max(dsp, df2) - 1):+6.1f} "
                f"{100 * (auto / dsp - 1):+6.1f}{flag}"
            )
    print("next best:")
    for r in rows[1:6]:
        print(f"  {100 * r[1]:+.1f} {100 * r[2]:+.1f} {json.dumps(r[3])}")
    if a.write:
        out = Path(a.write)
        costs = data[a.table_condition][1]
        out.write_text(header(best, costs, a.k))
        digest = hashlib.sha256()
        for cond in data:
            for f in sorted((root / cond).rglob("*.json")):
                digest.update(f.read_bytes())
        provenance = root / "provenance.json"
        meta = {
            "recording": json.loads(provenance.read_text()) if provenance.exists() else None,
            "constants": best,
            "k": a.k,
            "table_condition": a.table_condition,
            "conditions": list(data),
            "traces_sha256": digest.hexdigest(),
            "texts": {c: len(d[0]) for c, d in data.items()},
            "worst_against_best": wb,
            "worst_against_dspark": wf,
            "expected": {
                c: {g: dict(zip(("auto", "dspark", "dflash2", "none"), v)) for g, v in t.items()}
                for c, t in table.items()
            },
            "grid": {n: grid[n] for n in names},
            "top": [{"worst_against_best": r[1], "worst_against_dspark": r[2], "constants": r[3]} for r in rows[:10]],
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
