#!/bin/bash
# capy-flash: the small Qwen3.8-Flash-Next test artifact of the Engine integration tests
# (NINFER_TEST_QWEN4_ARTIFACT: ninfer_qwen4_tp2_engine_test, ninfer_qwen4_tp2_moe_abandon_test,
# ninfer_qwen4_mtp_engine_test).
#
#   tools/capy_flash/make_test_artifact.sh [--fp4-local|--vision] <NVIDIA Qwen3.8-Flash-Next checkpoint dir> <out dir>
#
# 1. gen_synthetic.py (sybrix, Apache-2.0): a random checkpoint with the REAL per-layer geometry
#    (hidden 2560, 4 hyper-connection streams, 512 NVFP4 experts top-10 + shared, GDN and QSA layers,
#    PLE at block 1, MTP head), only 4 Text layers (GDN GDN GDN QSA) and a tiny n-gram table; tokenizer
#    and config come from the real checkpoint (only those files are read). Seed 1234. ~11 GB, CPU only.
# 2. tools.convert --recipe qwen3_8_flash_next_nvfp4 --components text,mtp (CPU, under a minute).
# --fp4-local: the checkpoint is Qwen3.8-Flash-Next-FP4-Local and the formats follow it (MXFP8 dense,
#    CSF NVFP4 experts, row-scaled NVFP4 n-gram table, BF16 where it is BF16); recipe
#    qwen3_8_flash_next_fp4_local; files fn4l/ and fn4l.ninfer.
# --vision: the nvfp4 variant with a (random, real-geometry) Vision tower, components text,vision,mtp; files fn4v/ and
#    fn4v.ninfer for NINFER_TEST_QWEN4_VISION_ARTIFACT (ninfer_qwen4_vision_engine_test).
# Prints the export line for the tests. Needs Python with torch, numpy and safetensors.
set -euo pipefail
VARIANT=nvfp4; RECIPE=qwen3_8_flash_next_nvfp4; NAME=fn4
COMPONENTS=text,mtp; GEN_VISION=; EXPORT=NINFER_TEST_QWEN4_ARTIFACT
if [ "${1:-}" = --fp4-local ]; then VARIANT=fp4_local; RECIPE=qwen3_8_flash_next_fp4_local; NAME=fn4l; shift
elif [ "${1:-}" = --vision ]; then NAME=fn4v; COMPONENTS=text,vision,mtp; GEN_VISION=--vision
  EXPORT=NINFER_TEST_QWEN4_VISION_ARTIFACT; shift; fi
[ $# -eq 2 ] || { sed -n 2,18p "$0"; exit 2; }
SRC=$1; OUT=$2; PY=${PYTHON:-python3}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
[ -f "$OUT/$NAME/model.safetensors.index.json" ] ||
  "$PY" "$ROOT/tools/capy_flash/gen_synthetic.py" --source "$SRC" --out "$OUT/$NAME" --layers 4 --mtp --seed 1234 \
    --variant "$VARIANT" $GEN_VISION
(cd "$ROOT" && "$PY" -m tools.convert --model "$OUT/$NAME" --recipe "$RECIPE" --components "$COMPONENTS" \
   --device cpu --name "capy-flash-$NAME" --out "$OUT/$NAME.ninfer")
echo "export $EXPORT=$OUT/$NAME.ninfer"
