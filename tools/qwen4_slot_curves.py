#!/usr/bin/env python3
"""Qwen3.8-Flash-Next per-layer slot demand curves: header generation and evaluation.

Input: tables of ninfer_qwen4_slot_curves (tests/qwen4_slot_curves.cpp), one per routing dump
(NINFER_QWEN4_ROUTING_DUMP): '# windows D', then 'slots layer missing wait_us' summed over the D
decode windows of the dump, slots per GPU and layer.

  qwen4_slot_curves.py header TABLE... \
      | clang-format --assume-filename=x.h > src/models/qwen3_5/qwen4_slot_curves.h
      The mean over the tables of each layer's modelled wait per decode window (every table
      weighs the same).
  qwen4_slot_curves.py profile SLOTS_PER_LAYER ELASTIC_COLUMNS [LAYERS]
      The base slots per layer that layer_slot_profile gives (from the header's float curves).
  qwen4_slot_curves.py eval TABLE...
      Modelled wait per decode window of the curve allocation (layer_slot_profile) against the
      usage weights (kQwen4LayerSlotWeights) water-filled to the same pool, at several pool sizes, on every table and leave-one-out (curves from the
      other tables, scored on the held-out one).

The allocation below mirrors Qwen4OffloadExperts::layer_slot_profile; the C++ test pins a few of
its values.
"""

import math
import re
import struct
import sys

EXPERTS = 512
MIN_BASE = 16  # kQwen4MinBaseSlots


def load(path):
    windows, rows = None, {}
    with open(path) as f:
        for line in f:
            if line.startswith("# windows"):
                windows = int(line.split()[2])
                continue
            s, l, m, w = line.split()
            rows.setdefault(int(l), {})[int(s)] = float(w)
    grid = sorted(rows[0])
    return {"windows": windows, "grid": grid, "wait": [[rows[l][s] / windows for s in grid] for l in sorted(rows)]}


def mean_curves(tables):
    grid = tables[0]["grid"]
    assert all(t["grid"] == grid for t in tables), "tables on different grids"
    layers = len(tables[0]["wait"])
    return grid, [[sum(t["wait"][l][k] for t in tables) / len(tables) for k in range(len(grid))] for l in range(layers)]


def value(grid, curve, slots):
    """Linear interpolation; below the grid the first segment extends, above it the curve is flat."""
    if slots >= grid[-1]:
        return curve[-1]
    if slots <= grid[0]:
        a, b = grid[0], grid[1]
        return curve[0] + (curve[1] - curve[0]) * (slots - a) / (b - a)
    for k in range(len(grid) - 1):
        if grid[k] <= slots <= grid[k + 1]:
            a, b = grid[k], grid[k + 1]
            return curve[k] + (curve[k + 1] - curve[k]) * (slots - a) / (b - a)
    raise AssertionError


def curve_profile(grid, curves, layers, spl, elastic, partitions=2):
    """Greedy: every slot to the layer whose modelled wait drops most (ties: fewer slots, then
    the lower layer), from MIN_BASE slots, within the partition's experts."""
    cap = EXPERTS // partitions - elastic
    floor = min(MIN_BASE, spl)
    base = [floor] * layers
    for _ in range(layers * spl - floor * layers):
        best, gain = -1, 0.0
        for l in range(layers):
            if base[l] + 1 > cap:
                continue
            g = value(grid, curves[l], base[l] + elastic) - value(grid, curves[l], base[l] + 1 + elastic)
            if best < 0 or g > gain or (g == gain and base[l] < base[best]):
                best, gain = l, g
        base[best] += 1
    return base


def weight_profile(weights, layers, spl, elastic, partitions=2):
    """kQwen4LayerSlotWeights water-filled to the pool (clamped at the cap, largest remainder)."""
    weights = weights[:layers]
    cap = EXPERTS // partitions - elastic
    total = layers * spl
    target, fixed, changed = [0.0] * layers, [False] * layers, True
    while changed:
        changed = False
        rem = total - sum(t for t, f in zip(target, fixed) if f)
        ws = sum(w for w, f in zip(weights, fixed) if not f)
        for l in range(layers):
            if not fixed[l]:
                target[l] = max(1.0, weights[l] * rem / ws)
                if target[l] > cap:
                    target[l], fixed[l], changed = cap, True, True
    base = [math.floor(t) for t in target]
    order = sorted(range(layers), key=lambda l: -(target[l] - math.floor(target[l])))
    i = 0
    while sum(base) < total:
        if base[order[i % layers]] < cap:
            base[order[i % layers]] += 1
        i += 1
    return base


def score(table, base, elastic):
    return sum(value(table["grid"], table["wait"][l], base[l] + elastic) for l in range(len(base)))


def source(name):
    return open(__file__.rsplit("/tools/", 1)[0] + "/src/models/qwen3_5/" + name).read()


def header_curves():
    """kQwen4SlotCurveSlots and kQwen4LayerSlotCurves as the engine reads them (float)."""
    src = source("qwen4_slot_curves.h")
    grid = [
        int(x)
        for x in re.search(r"kQwen4SlotCurveSlots = \{([^}]*)\}", src).group(1).replace("\n", " ").split(",")
        if x.strip()
    ]
    body = src.split("kQwen4LayerSlotCurves = {{", 1)[1]
    f32 = lambda v: struct.unpack("f", struct.pack("f", float(v)))[0]  # noqa: E731
    curves = [[f32(v) for v in re.findall(r"([0-9.]+)F", block)] for block in body.split("}},")]
    return grid, [c for c in curves if c]


def fitted_weights():
    src = source("qwen4_layer_profile.h")
    body = re.search(r"kQwen4LayerSlotWeights = \{([^}]*)\}", src).group(1)
    return [int(x) for x in body.replace("\n", " ").split(",") if x.strip()]


def header(paths):
    tables = [load(p) for p in paths]
    grid, curves = mean_curves(tables)
    names = ", ".join(p.rsplit("/", 1)[-1].split(".")[0] for p in paths)
    out = [
        "#pragma once",
        "",
        "// Generated by tools/qwen4_slot_curves.py from tests/qwen4_slot_curves.cpp tables; do not edit.",
        f"// Corpus: {len(paths)} routing dumps ({names}),",
        "// Flash-Next GSQ-RCO, tp 2.",
        "// Per expert layer (48 Text layers, then the MTP layer) the modelled GPU wait in us per decode",
        "// window (mean over the dumps) when the layer holds kQwen4SlotCurveSlots[k] slots on each GPU.",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace ninfer::models::qwen3_5 {",
        "",
        f"inline constexpr std::array<std::int16_t, {len(grid)}> kQwen4SlotCurveSlots = {{",
    ]
    out += wrap([str(s) for s in grid]) + ["};", ""]
    out.append(
        f"inline constexpr std::array<std::array<float, {len(grid)}>, {len(curves)}> kQwen4LayerSlotCurves = {{{{"
    )
    for l, c in enumerate(curves):
        out.append(f"    {{{{ // layer {l}" if l < len(curves) - 1 else "    {{ // MTP layer")
        out += wrap([f"{v:.1f}F" for v in c], indent="       ")
        out.append("    }},")
    out += ["}};", "", "} // namespace ninfer::models::qwen3_5", ""]
    sys.stdout.write("\n".join(out))


def wrap(items, width=99, indent="   "):
    lines, cur = [], indent
    for it in items:
        if len(cur) + len(it) + 2 > width:
            lines.append(cur.rstrip())
            cur = indent
        cur += " " + it + ","
    lines.append(cur.rstrip())
    return lines


def evaluate(paths):
    tables = {p.rsplit("/", 1)[-1].split(".")[0]: load(p) for p in paths}
    weights = fitted_weights()
    grid, curves = mean_curves(list(tables.values()))
    configs = [(66, 21), (66, 3), (81, 2), (108, 3), (150, 3)]
    print("modelled wait per decode window, curve allocation vs fitted weights (49 layers, tp 2)")
    print("pool (base+elastic)  " + "  ".join(f"{n:>7s}" for n in tables) + "     mean   leave-one-out mean")
    for spl, el in configs:
        wp = weight_profile(weights, 49, spl, el)
        cp = curve_profile(grid, curves, 49, spl, el)
        rel, loo = [], []
        for n, t in tables.items():
            rel.append(score(t, cp, el) / score(t, wp, el) - 1)
            g2, c2 = mean_curves([u for m, u in tables.items() if m != n])
            lp = curve_profile(g2, c2, 49, spl, el)
            loo.append(score(t, lp, el) / score(t, wp, el) - 1)
        print(
            f"{spl:4d} + {el:2d}            "
            + "  ".join(f"{100 * r:+6.1f}%" for r in rel)
            + f"  {100 * sum(rel) / len(rel):+6.1f}%  {100 * sum(loo) / len(loo):+6.1f}%"
            + f" (worst {100 * max(loo):+.1f}%)"
        )


def profile(args):
    spl, elastic = int(args[0]), int(args[1])
    layers = int(args[2]) if len(args) > 2 else 49
    grid, curves = header_curves()
    print(",".join(map(str, curve_profile(grid, curves[:layers], layers, spl, elastic))))


if __name__ == "__main__":
    commands = {"header": header, "eval": evaluate, "profile": profile}
    if len(sys.argv) < 3 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](sys.argv[2:])
