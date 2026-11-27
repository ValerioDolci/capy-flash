"""The quick check of --spec auto (check.sh): sends a few corpus requests to a running server and prints the decode rate
of each group from the server's request log; with --compare, judges auto against the fixed drafters.

Usage: python3 check.py run --url http://127.0.0.1:PORT/v1 --corpus corpus.jsonl --names a,b,... --log REQ.jsonl --out ARM.json
       python3 check.py compare DIR (ARM.json files: auto, mtp, dflash2 per condition) [--share 0.02]"""

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path


def group_of(item):
    g = item["group"]
    return f"{g}_{item['name'].rsplit('_', 1)[1]}" if g in ("summary", "essay") else g


def run(a):
    names = set(a.names.split(","))
    items = [json.loads(line) for line in open(a.corpus)]
    items = [i for i in items if i["name"] in names]
    for item in items:
        req = urllib.request.Request(
            a.url + "/chat/completions",
            data=json.dumps(dict(item["body"], stream=False)).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3600).read()
    done = [json.loads(line) for line in open(a.log)]
    done = [r for r in done if r.get("event") == "request_done"][-len(items) :]
    per = defaultdict(lambda: [0, 0.0])
    for item, r in zip(items, done):
        g = per[group_of(item)]
        g[0] += r["result"]["completion_tokens"]
        g[1] += r["timings_seconds"]["decode"]
    Path(a.out).write_text(json.dumps({g: v[0] / v[1] for g, v in per.items() if v[1] > 0}))


def compare(a):
    root = Path(a.dir)
    failed = False
    for cond in sorted(p.name for p in root.iterdir() if p.is_dir()):
        arms = {f.stem: json.loads(f.read_text()) for f in (root / cond).glob("*.json")}
        if not {"auto", "mtp", "dflash2"} <= set(arms):
            print(f"{cond}: missing arms ({sorted(arms)})")
            failed = True
            continue
        print(f"== LoopCD {cond}: decode tok/s auto | DSpark (mtp) | DFlash2, auto against the better")
        for g in sorted(arms["auto"]):
            auto, mtp, df2 = arms["auto"][g], arms["mtp"].get(g, 0), arms["dflash2"].get(g, 0)
            margin = auto / max(mtp, df2) - 1.0
            bad = margin < -a.share
            failed |= bad
            print(f"  {g:16s} {auto:7.1f} {mtp:7.1f} {df2:7.1f}  {100 * margin:+6.1f} %{'  FAIL' if bad else ''}")
    print(
        "FAIL" if failed else "PASS", f"(auto within {100 * a.share:.0f} % of the better fixed drafter in every group)"
    )
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for name in ("--url", "--corpus", "--names", "--log", "--out"):
        r.add_argument(name, required=True)
    c = sub.add_parser("compare")
    c.add_argument("dir")
    c.add_argument("--share", type=float, default=0.02)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
        return 0
    return compare(a)


if __name__ == "__main__":
    sys.exit(main())
