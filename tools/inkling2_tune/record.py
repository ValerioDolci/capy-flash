"""Records the texts of a tuning corpus with their exact token ids (tools/inkling2_tune, step 1 of tune.sh).

Every corpus request (one JSON line: name, group, body, an OpenAI chat body) is sent to a running ninfer-serve started
without a drafter and with --no-prefix-reuse --log-speculation-detail --prefill-miss-dump DIR
--prefill-miss-min-tokens 1 --prefill-miss-min-match 0.95, then sent again with max_tokens 1: the second request misses the first one's context,
which the server dumps (prompt and generated ids). A text does not depend on the drafter (greedy, or sampled with the
request's seed), so these are the texts of every --spec mode. Writes OUT/<name>.json: name, group, sampled, prompt_ids,
generated_ids, stopped (ended before max_tokens) and step_ms (the server's decode milliseconds per token: an ordinary
step at that context).

Usage: python3 record.py --url http://127.0.0.1:PORT/v1 --corpus corpus.jsonl --dumps DUMPDIR --out OUTDIR"""

import argparse
import json
import urllib.request
from pathlib import Path


def post(url, body):
    req = urllib.request.Request(
        url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, dumps = Path(a.out), Path(a.dumps)
    out.mkdir(parents=True, exist_ok=True)
    for line in open(a.corpus):
        item = json.loads(line)
        body = dict(item["body"], stream=False)
        r = post(a.url, body)
        before = {d.name for d in dumps.iterdir()} if dumps.exists() else set()
        post(a.url, dict(body, max_tokens=1))
        new = sorted(d for d in dumps.iterdir() if d.name not in before)
        if not new:
            raise SystemExit(f"{item['name']}: the server dumped no context (check --prefill-miss-dump)")
        ids = json.loads((new[-1] / "old_context.tokens.json").read_text())
        n, gen = r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]
        if len(ids) != n + gen:
            raise SystemExit(f"{item['name']}: dumped {len(ids)} ids, expected {n} + {gen}")
        per_second = r.get("timings", {}).get("predicted_per_second") or 0.0
        rec = {
            "name": item["name"],
            "group": item["group"],
            "sampled": float(body.get("temperature", 1.0)) > 0.0,
            "prompt_ids": ids[:n],
            "generated_ids": ids[n:],
            "stopped": gen < body["max_tokens"],
            "step_ms": 1e3 / per_second if per_second > 0 else None,
        }
        (out / f"{item['name']}.json").write_text(json.dumps(rec))
        print(f"{item['name']}: {n} + {gen} tokens, step {rec['step_ms'] or 0:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
