#!/usr/bin/env python3
"""Token-frequency ranking of an Italian corpus for the MTP draft shortlist (tools/qwen4_draft_shortlist.py --extra-ranking).
Counts the Flash-Next tokenizer's tokens over Wikipedia (it) parquet shards and Italian instruction sets (JSON
lists of {instruction,input,output} or conversations), holding out every 20th document. Writes
<out>.train.counts.i64 / <out>.heldout.counts.i64 in the format of tools/freq_corpus ([1, 248320] little-endian i64)."""

import argparse
import json
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

VOCAB = 248320
ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("inputs", nargs="+")
ap.add_argument("--max-docs", type=int, default=200000)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
train = np.zeros(VOCAB, np.int64)
held = np.zeros(VOCAB, np.int64)


def texts():
    for p in a.inputs:
        if p.endswith(".parquet"):
            for batch in pq.ParquetFile(p).iter_batches(columns=["text"], batch_size=512):
                for t in batch.column(0).to_pylist():
                    yield t
        else:
            for r in json.load(open(p)):
                if "conversations" in r:
                    yield "\n".join(m.get("value", "") for m in r["conversations"])
                else:
                    yield "\n".join(str(r.get(k, "")) for k in ("instruction", "input", "output"))


n = tokens = 0
batch = []


def flush():
    global tokens
    for i, ids in enumerate(tok(batch, add_special_tokens=False)["input_ids"]):
        dst = held if (n - len(batch) + i) % 20 == 0 else train
        np.add.at(dst, ids, 1)
        tokens += len(ids)
    batch.clear()


for t in texts():
    if not t:
        continue
    batch.append(t[:20000])
    n += 1
    if len(batch) == 256:
        flush()
    if n >= a.max_docs:
        break
if batch:
    flush()
train.tofile(a.out + ".train.counts.i64")
held.tofile(a.out + ".heldout.counts.i64")
print(
    json.dumps(
        {
            "docs": n,
            "tokens": int(tokens),
            "train_tokens": int(train.sum()),
            "heldout_tokens": int(held.sum()),
            "distinct": int((train > 0).sum()),
        }
    )
)
