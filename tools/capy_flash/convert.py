#!/usr/bin/env python3
"""capy-flash: the v0.1 artifact in one command.

From the NVIDIA ModelOpt checkpoint (nvidia/Qwen3.8-Flash-Next-NVFP4) to the serving directory:
  1. tools.convert with the recipe qwen3_8_flash_next_nvfp4_idxbf16_gdnq8 (experts NVFP4 as published, indexer BF16,
     GDN Q8), components text + mtp  ->  <out>/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer
  2. the MTP draft shortlist beside it (tools/qwen4_draft_shortlist.py, --prefix 98304 --count-top 32768 over the
     repository ranking, plus the 8 192 most frequent Italian tokens it lacks)  ->  <out>/qwen3_8_flash_next_nvfp4_idxgdnq8.draft_shortlist.txt
The engine picks the shortlist up by itself (artifact sidecar).

  python tools/capy_flash/convert.py --model <checkpoint dir> --out <dir> [--vision] [--device cuda] [--skip-convert]

--vision adds the checkpoint's Vision tower (component vision, Q8 groupwise, about 0.45 GiB on its GPU):
the same Text and MTP objects, served with `ninfer-serve ... --profile flash-next --vision`; without --vision at
startup the tower is not loaded and nothing changes.

--skip-convert only (re)writes the shortlist next to an existing artifact. Run from the repository root.
"""

import argparse
import subprocess
import sys
from pathlib import Path

RECIPE = "qwen3_8_flash_next_nvfp4_idxbf16_gdnq8"
STEM = "qwen3_8_flash_next_nvfp4_idxgdnq8"
ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", type=Path, required=True, help="ModelOpt NVFP4 checkpoint directory (with tokenizer.json)")
ap.add_argument("--out", type=Path, required=True, help="output directory")
ap.add_argument("--device", default="cuda")
ap.add_argument("--name", default="qwen3.8-flash-next-idxgdnq8")
ap.add_argument("--vision", action="store_true", help="also convert the Vision tower (image input)")
ap.add_argument("--skip-convert", action="store_true", help="only write the draft shortlist")
a = ap.parse_args()

if not (a.model / "tokenizer.json").is_file():
    sys.exit(f"capy_flash_convert: {a.model} has no tokenizer.json (not a checkpoint directory?)")
a.out.mkdir(parents=True, exist_ok=True)
art = a.out / f"{STEM}.ninfer"


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


if not a.skip_convert:
    run(
        [
            sys.executable,
            "-m",
            "tools.convert",
            "--model",
            a.model,
            "--recipe",
            RECIPE,
            "--components",
            "text,vision,mtp" if a.vision else "text,mtp",
            "--device",
            a.device,
            "--name",
            a.name,
            "--out",
            art,
        ]
    )
elif not art.exists():
    sys.exit(f"capy_flash_convert: --skip-convert but {art} does not exist")
run(
    [
        sys.executable,
        ROOT / "tools/qwen4_draft_shortlist.py",
        "--prefix",
        "98304",
        "--count-top",
        "32768",
        "--extra-ranking",
        ROOT / "tools/freq_corpus/fixtures/ranking/it.train.counts.i64",
        "--extra-top",
        "8192",
        "--tokenizer",
        a.model,
        "--out",
        a.out / f"{STEM}.draft_shortlist.txt",
    ]
)
print(f"capy_flash_convert: ready: {art} (+ draft shortlist)")
