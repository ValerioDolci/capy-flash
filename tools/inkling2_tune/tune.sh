#!/usr/bin/env bash
# Retunes --spec auto of Inkling 2 (docs/maintainer/inkling2-auto-tuning.md) in one command, on the GPUs it is given:
#   1. records the corpus' texts (record.py) with LoopCD off and on, from a server without a drafter;
#   2. records both drafters' drafts at every position of every text and the time of their rounds (the tf driver's
#      --mode drafts, rounds of K drafts);
#   3. simulates the choice over a grid of its constants on those traces and costs, writes the best into
#      src/models/inkling2/program/drafter_tuning.h (+ drafter_tuning.json: what it was measured on) and prints the
#      expected decode rate per group against the fixed drafters (tune.py);
#   4. rebuilds the engine's choice with the new constants and checks, trace by trace, that it takes the simulator's
#      steps (parity.py with the drafter_replay binary).
# Usage (from the source tree, built): tools/inkling2_tune/tune.sh --artifact BOTH.ninfer --corpus corpus.jsonl
#   --out DIR [--build build] [--conditions off,on] [--steps record,drafts,tune,parity] [--port 18095] [--k 3]
# BOTH.ninfer holds the text and both drafters (components text,mtp,dflash2). Needs both GPUs (tp 2) for its whole run.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
build=$root/build conditions=off,on steps=record,drafts,tune,parity port=18095 k=3 artifact= corpus= out=
while [ $# -gt 0 ]; do
  case $1 in
    --artifact) artifact=$2; shift 2 ;;
    --corpus) corpus=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --build) build=$2; shift 2 ;;
    --conditions) conditions=$2; shift 2 ;;
    --steps) steps=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --k) k=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$artifact" ] && [ -n "$corpus" ] && [ -n "$out" ] || { sed -n 2,13p "$0" >&2; exit 2; }
serve=$build/apps/ninfer-serve driver=$build/tests/ninfer_inkling2_tf_driver
mkdir -p "$out"
has() { case ",$steps," in *",$1,"*) return 0 ;; esac; return 1; }
loopcd() { [ "$1" = on ] && echo "--loopcd adaptive --loopcd-omega 0.5" || true; }
t0=$(date +%s)
if has record; then
  python3 - "$out/provenance.json" "$root" "$artifact" "$corpus" "$conditions" "$k" <<'PY'
import hashlib, json, os, subprocess, sys, time
out, root, artifact, corpus, conditions, k = sys.argv[1:]
gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
json.dump({"recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": subprocess.run(
    ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "artifact": os.path.abspath(artifact), "artifact_bytes": os.path.getsize(artifact),
    "corpus": os.path.abspath(corpus), "corpus_sha256": hashlib.sha256(open(corpus, "rb").read()).hexdigest(),
    "conditions": conditions.split(","), "k": int(k), "gpus": gpus.split("\n")[:-1]}, open(out, "w"), indent=1)
PY
fi
for cond in ${conditions//,/ }; do
  dir=$out/$cond
  mkdir -p "$dir"
  if has record; then
    rm -rf "$dir/dumps" "$dir/ids" "$dir/req.jsonl"
    "$serve" "$artifact" --tp 2 --devices 0,1 $(loopcd $cond) --host 127.0.0.1 --port "$port" --max-context 253952 \
      --prefill-chunk 4096 --embedding-host --model-id inkling2-mini --no-prefix-reuse --log-speculation-detail \
      --request-log-jsonl "$dir/req.jsonl" --prefill-miss-dump "$dir/dumps" --prefill-miss-min-tokens 1 \
      --prefill-miss-min-match 0.95 > "$dir/serve.log" 2>&1 &
    pid=$!
    trap 'kill $pid 2>/dev/null || true' EXIT
    for _ in $(seq 1 300); do
      curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break
      kill -0 $pid 2>/dev/null || { tail -5 "$dir/serve.log" >&2; exit 1; }
      sleep 1
    done
    python3 "$here/record.py" --url "http://127.0.0.1:$port/v1" --corpus "$corpus" --dumps "$dir/dumps" \
      --out "$dir/ids" > "$dir/record.log"
    kill $pid; wait $pid 2>/dev/null || true
    trap - EXIT
    rm -rf "$dir/dumps"
    echo "$cond: texts recorded ($(ls "$dir/ids" | wc -l)) at $(( $(date +%s) - t0 )) s"
  fi
  if has drafts; then
    files=$(ls "$dir"/ids/*.json | paste -sd,)
    for drafter in dsp df2; do
      flag=--drafter; [ $drafter = df2 ] && flag=--dflash2
      rm -rf "$dir/drafts_$drafter"
      "$driver" --artifact "$artifact" --ids "$files" --mode drafts $flag --tp 2 --block 4096 --drafts-k "$k" \
        $(loopcd $cond) --out "$dir/drafts_$drafter" > "$dir/drafts_$drafter.log" 2>&1
      echo "$cond: $drafter drafts at $(( $(date +%s) - t0 )) s"
    done
  fi
done
header=$root/src/models/inkling2/program/drafter_tuning.h
if has tune; then
  python3 "$here/tune.py" --data "$out" --conditions "$conditions" --k "$k" --write "$header" | tee "$out/tune.txt"
fi
if has parity; then
  ninja -C "$build" ninfer_inkling2_drafter_replay > /dev/null
  python3 "$here/parity.py" --replay "$build/tests/ninfer_inkling2_drafter_replay" --header "$header" --data "$out" \
    --conditions "$conditions" --k "$k"
fi
echo "tune.sh: $(( $(date +%s) - t0 )) s"
