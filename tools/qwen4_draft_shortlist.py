#!/usr/bin/env python3
"""Draft token shortlist for the Qwen4Exp MTP proposal head (EngineOptions draft_shortlist).

The shortlist is the union of
  * the token-ID prefix [0, --prefix): Qwen byte-level BPE assigns IDs in merge order, i.e. by
    pair frequency in the tokenizer's training corpus, so the low IDs are a frequency ranking of
    their own; the engine executes this block as a view of the output head (no copy);
  * the --count-top most frequent tokens of a measured ranking (default: the repository fixture
    tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64 = teacher-forced argmax counts of
    Qwen3.6-27B over 72 464 chat conversations, 51.1 M emitted positions, en/zh/other chat, code,
    math; same 248 320-token vocabulary as Qwen3.8-Flash-Next, vocab.json identical); the engine
    copies these rows of the head (the ones above the prefix);
  * every added (special) token of the tokenizer below the public token count (chat control);
  * optionally the --extra-top most frequent tokens of a second ranking (--extra-ranking) not
    already in the list, e.g. tools/freq_corpus/fixtures/ranking/it.train.counts.i64 for Italian:
    with the default list 14.5 % of held-out Italian tokens fell outside it, 0.24 % with 8 192 more.
Reproducible: deterministic (stable sort on counts, ties by lower ID); the output header records
the inputs and the ranking file's SHA-256. Output: one ID per line, in priority order (specials,
the prefix ascending, then the ranking's tokens by count).

  qwen4_draft_shortlist.py --prefix 98304 --count-top 32768 \
      --tokenizer <NVIDIA Qwen3.8-Flash-Next-NVFP4 checkpoint dir> --out shortlist.txt
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

VOCAB = 248320
HERE = Path(__file__).resolve().parent
DEFAULT_RANKING = HERE / "freq_corpus/fixtures/ranking/ranking.train.counts.i64"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--prefix", type=int, default=0, help="token-ID prefix [0, N)")
ap.add_argument("--count-top", type=int, default=0, help="most frequent N tokens of the ranking")
ap.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
ap.add_argument("--row", type=int, default=0, help="ranking row (0 = total; 1..5 = strata)")
ap.add_argument("--tokenizer", type=Path, required=True, help="directory with tokenizer.json")
ap.add_argument("--public", type=int, default=0, help="public token count (default: tokenizer size)")
ap.add_argument("--extra-ranking", type=Path, help="second ranking whose top tokens are appended")
ap.add_argument("--extra-top", type=int, default=0, help="new tokens taken from --extra-ranking")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

tok = json.loads((a.tokenizer / "tokenizer.json").read_text())
base = len(tok["model"]["vocab"])
added = sorted(t["id"] for t in tok.get("added_tokens", []))
public = a.public or max([base - 1] + added) + 1
specials = [i for i in added if i < public]
raw = a.ranking.read_bytes()
rows = len(raw) // (VOCAB * 8)
counts = np.frombuffer(raw, dtype="<i8").reshape(rows, VOCAB)[a.row][:public]
order = np.argsort(-counts, kind="stable")
top = [int(i) for i in order[: a.count_top]] if a.count_top else []
seen, ids = set(), []
for i in specials + list(range(min(a.prefix, public))) + top:
    if i not in seen:
        seen.add(i)
        ids.append(i)
extra_note = ""
if a.extra_ranking and a.extra_top:
    extra_raw = a.extra_ranking.read_bytes()
    extra = np.frombuffer(extra_raw, dtype="<i8").reshape(-1, VOCAB)[0][:public]
    added_extra = 0
    for i in np.argsort(-extra, kind="stable"):
        if added_extra == a.extra_top or extra[i] <= 0:
            break
        if int(i) not in seen:
            seen.add(int(i))
            ids.append(int(i))
            added_extra += 1
    extra_note = f", extra {added_extra} of {a.extra_ranking.name} (sha256 {hashlib.sha256(extra_raw).hexdigest()})"
sha = hashlib.sha256(raw).hexdigest()
with open(a.out, "w") as f:
    f.write(f"# T43 Qwen4Exp draft shortlist: {len(ids)} token IDs\n")
    f.write(
        f"# prefix {a.prefix}, count_top {a.count_top} of {a.ranking.name} row {a.row} "
        f"(sha256 {sha}){extra_note}, {len(specials)} added tokens, public {public}\n"
    )
    f.write(f"# tokenizer {a.tokenizer} (vocab {base} + {len(added)} added)\n")
    for i in ids:
        f.write(f"{i}\n")
print(
    json.dumps(
        {
            "out": str(a.out),
            "ids": len(ids),
            "prefix": a.prefix,
            "count_top": a.count_top,
            "outside_prefix": sum(1 for i in ids if i >= a.prefix),
            "specials": len(specials),
            "public": public,
            "ranking_sha256": sha,
        }
    )
)
