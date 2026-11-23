#!/usr/bin/env bash
# Qwen3.8-27B one-GPU launcher for the release tarball: serve-27b-16gb.sh <artifact.ninfer> [ninfer-serve flags...]
# For the 3-bit GSQ artifact (Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash) on ONE 16 GB sm_120 board:
# `ninfer-serve --profile 27b-16gb` = INT8 KV, 65,536-token context, MTP with 3 draft tokens and prompt lookup, Vision,
# embedding in host RAM (13.9 GB of VRAM in use). Environment overrides:
#   NINFER_HOST (127.0.0.1)  NINFER_PORT (8080)  NINFER_DEVICE (0)  NINFER_MAX_CONTEXT (65536)
#   NINFER_MODEL_ID (qwen3.8-27b: the `model` name requests must use)
#   NINFER_VISION=0 turns Vision off (--no-vision: more room for the KV).
# Extra flags are appended and win over the profile. Run it under a supervisor that restarts on a
# non-zero exit: ninfer-serve exits with status 2 once the Engine fails (docs/serving/endpoints.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { echo "usage: $0 <artifact.ninfer> [ninfer-serve flags...]" >&2; exit 2; }
artifact=$1; shift
device=${NINFER_DEVICE:-0}
vision=(--vision-device "$device")
[ "${NINFER_VISION:-1}" = 0 ] && vision=(--no-vision)
# The NInfer Qwen3.8 chat template (system/developer turns anywhere, final-assistant continuation): many
# published artifacts embed Qwen's upstream template, which rejects the `developer` role and Responses `instructions`.
template=()
[ -f "$here/chat_templates/qwen3_8.jinja" ] && template=(--chat-template "$here/chat_templates/qwen3_8.jinja")
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/bin/ninfer-serve" "$artifact" --profile 27b-16gb \
  --host "${NINFER_HOST:-127.0.0.1}" --port "${NINFER_PORT:-8080}" \
  --device "$device" --max-context "${NINFER_MAX_CONTEXT:-65536}" --model-id "${NINFER_MODEL_ID:-qwen3.8-27b}" \
  "${vision[@]}" "${template[@]}" "$@"
