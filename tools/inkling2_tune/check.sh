#!/usr/bin/env bash
# The quick check of --spec auto of Inkling 2 (docs/maintainer/inkling2-auto-tuning.md): a few requests of the tuning
# corpus per group, served with --spec auto, mtp and dflash2 (K 3) on the artifact with both drafters, LoopCD off and on;
# fails when auto decodes more than 2 % below the better fixed drafter in some group. About 6 minutes on two GPUs.
# Usage (from the source tree, built): tools/inkling2_tune/check.sh --artifact BOTH.ninfer --corpus corpus.jsonl
#   --out DIR [--build build] [--conditions off,on] [--names LIST] [--port 18096]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
build=$root/build conditions=off,on port=18096 artifact= corpus= out=
names=m4,held_greedy_0,held_greedy_4,held_greedy_8,held_greedy_12,held_sampled_1,held_sampled_5,held_sampled_9
names=$names,held_sampled_13,agentic_13,long_greedy_5,long_sampled_5,summary_4096,summary_8192,essay_8192,essay_32768
while [ $# -gt 0 ]; do
  case $1 in
    --artifact) artifact=$2; shift 2 ;;
    --corpus) corpus=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --build) build=$2; shift 2 ;;
    --conditions) conditions=$2; shift 2 ;;
    --names) names=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$artifact" ] && [ -n "$corpus" ] && [ -n "$out" ] || { sed -n 2,6p "$0" >&2; exit 2; }
t0=$(date +%s)
for cond in ${conditions//,/ }; do
  mkdir -p "$out/$cond"
  for arm in auto mtp dflash2; do
    loopcd=""; [ "$cond" = on ] && loopcd="--loopcd adaptive --loopcd-omega 0.5"
    rm -f "$out/$cond/$arm.req.jsonl"
    "$build/apps/ninfer-serve" "$artifact" --tp 2 --devices 0,1 --spec $arm --draft-tokens 3 $loopcd --host 127.0.0.1 \
      --port "$port" --max-context 229376 --prefill-chunk 4096 --embedding-host --model-id inkling2-mini --no-prefix-reuse \
      --request-log-jsonl "$out/$cond/$arm.req.jsonl" > "$out/$cond/$arm.serve.log" 2>&1 &
    pid=$!
    trap 'kill $pid 2>/dev/null || true' EXIT
    for _ in $(seq 1 300); do
      curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break
      kill -0 $pid 2>/dev/null || { tail -5 "$out/$cond/$arm.serve.log" >&2; exit 1; }
      sleep 1
    done
    python3 "$here/check.py" run --url "http://127.0.0.1:$port/v1" --corpus "$corpus" --names "$names" \
      --log "$out/$cond/$arm.req.jsonl" --out "$out/$cond/$arm.json"
    kill $pid; wait $pid 2>/dev/null || true
    trap - EXIT
  done
done
rc=0
python3 "$here/check.py" compare "$out" || rc=$?
echo "check.sh: $(( $(date +%s) - t0 )) s"
exit $rc
