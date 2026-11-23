#!/usr/bin/env bash
# Qwen3.8-27B two-GPU launcher for the release tarball: serve-tp2.sh <artifact.ninfer> [ninfer-serve flags...]
# (Qwen3.8-Flash-Next: serve-flash-next.sh.) The ninfer-tp2 production profile, `ninfer-serve --profile 27b` (two 16 GB sm_120 boards, no P2P): INT8 KV, 196,608-token
# context, DFlash2 with 7 draft tokens when the artifact carries the DFlash2 drafter (MTP with 3 otherwise), Vision on device 0.
# Environment overrides:
#   NINFER_HOST (127.0.0.1)  NINFER_PORT (8080)  NINFER_DEVICES (0,1)  NINFER_MAX_CONTEXT (196608)
#   NINFER_MODEL_ID (qwen3.8-27b: the `model` name requests must use)
#   NINFER_VISION=0 turns Vision off (--no-vision; artifacts without a vision component).
# Extra flags are appended and win over the profile. Run it under a supervisor that restarts on a
# non-zero exit: ninfer-serve exits with status 2 once the Engine fails (docs/serving/endpoints.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { echo "usage: $0 <artifact.ninfer> [ninfer-serve flags...]" >&2; exit 2; }
artifact=$1; shift
context=${NINFER_MAX_CONTEXT:-196608}
vision=()
[ "${NINFER_VISION:-1}" = 0 ] && vision=(--no-vision)
# The NInfer Qwen3.8 chat template (system/developer turns anywhere, final-assistant continuation): many
# published artifacts embed Qwen's upstream template, which rejects the `developer` role and Responses `instructions`.
template=()
[ -f "$here/chat_templates/qwen3_8.jinja" ] && template=(--chat-template "$here/chat_templates/qwen3_8.jinja")
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/bin/ninfer-serve" "$artifact" --profile 27b \
  --host "${NINFER_HOST:-127.0.0.1}" --port "${NINFER_PORT:-8080}" \
  --devices "${NINFER_DEVICES:-0,1}" --max-context "$context" --model-id "${NINFER_MODEL_ID:-qwen3.8-27b}" \
  "${vision[@]}" "${template[@]}" "$@"
