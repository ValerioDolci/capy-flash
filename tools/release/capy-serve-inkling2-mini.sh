#!/usr/bin/env bash
# Inkling 2 Mini launcher for the release tarball: capy-serve-inkling2-mini.sh <artifact> [server flags...]
# Two 16 GB sm_120 boards in tensor parallel (no P2P needed): text only, one request at a time (more
# requests queue), 253,952-token context. Environment overrides, as in the other launchers:
#   NINFER_HOST (127.0.0.1)  NINFER_PORT (8080)  NINFER_DEVICES (0,1)  NINFER_MAX_CONTEXT (253952)
#   NINFER_PREFILL_CHUNK (4096: prompt tokens per prefill block, a multiple of 128)
#   NINFER_MODEL_ID (inkling-2-mini: the `model` name requests must use)
#   NINFER_DRAFTER=0 turns the drafter off (no --spec mtp --draft-tokens 3)
# The context and the prefill blocks are reserved at startup: 253,952 tokens with 4096-token blocks and the
# drafter leave about 580 MiB free per 16 GB board, with the token embedding in pinned host memory
# (--embedding-host: one 0.9 GB copy in RAM that both boards read in place, identical results).
# Smaller blocks (2048) prefill 6-13 % slower and reserve less.
# One device in NINFER_DEVICES runs at tp 1, which needs a 32 GB board (the weights are 20 GB): this
# is not verified, and the launcher refuses a smaller board.
# Speculative decoding with the checkpoint's DSpark drafter is on (--spec mtp --draft-tokens 3): the
# text is the same as without it, and it takes VRAM from the context. An artifact without the mtp
# component (tools.convert --components text) starts and decodes without a drafter. `--spec off`
# or another `--draft-tokens` after the artifact wins.
# Extra flags are appended and win over these. Run it under a supervisor that restarts on a
# non-zero exit: the server exits with status 2 once the Engine fails (docs/serving/endpoints.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { echo "usage: $0 <artifact> [server flags...]" >&2; exit 2; }
artifact=$1; shift
devices=${NINFER_DEVICES:-0,1}
case $devices in
  *,*) parallel=(--tp 2 --devices "$devices") ;;
  *)
    mib=$(nvidia-smi -i "$devices" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    case $mib in ''|*[!0-9]*) mib='?' ;; esac
    if ! [ "$mib" -ge 30000 ] 2>/dev/null; then
      echo "$0: Inkling 2 Mini needs two 16 GB GPUs (NINFER_DEVICES=0,1) or one GPU with 32 GB;" \
        "device $devices: $mib MiB" >&2
      exit 2
    fi
    echo "$0: one GPU ($mib MiB): the one-board run of Inkling 2 Mini is not verified" >&2
    parallel=(--device "$devices") ;;
esac
drafter=(--spec mtp --draft-tokens 3)
[ "${NINFER_DRAFTER:-1}" = 0 ] && drafter=()
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/bin/ninfer-serve" "$artifact" "${parallel[@]}" "${drafter[@]}" \
  --host "${NINFER_HOST:-127.0.0.1}" --port "${NINFER_PORT:-8080}" \
  --max-context "${NINFER_MAX_CONTEXT:-253952}" --prefill-chunk "${NINFER_PREFILL_CHUNK:-4096}" --embedding-host \
  --model-id "${NINFER_MODEL_ID:-inkling-2-mini}" "$@"
