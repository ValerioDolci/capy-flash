# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""The engine's two-rank arithmetic emulated in the reference at visit 0 only (layer 0, visited once).

  emulate_v00.py --checkpoint DIR --tokens A.json [B.json ...] --rows SPEC --out DIR [--modes none,a,b,ab]

  a  output projection as at tp 2 (forward/attention.h, ops::linear_row_parallel, allreduce_sum): the gate is applied
     per rank as in the reference, each rank rounds its partial bf16(o_r Wo_r^T) over its 12 heads (contiguous columns
     of Wo), and the all-reduce adds the two BF16 partials in FP32 and rounds once: bf16(f32(p0) + f32(p1)).
  b  V through FP16 as the engine's KV storage: v' -> fp16 -> bf16 (FP16_RNE of the BF16 value).
Every mode writes OUT/<mode>/<stem>/, a ref_dump.py dump of the native reference (--impl native) with the given rows.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkref  # noqa: E402
import ref_dump  # noqa: E402

FLAGS = {"a": False, "b": False}


def rank_split_output(self, attn_output, hidden_states):
    gate = torch.sigmoid(self.gate_proj(hidden_states).float())
    o = (attn_output.float() * gate).to(attn_output.dtype)
    if not FLAGS["a"]:
        return self.linear(o)
    half = o.shape[-1] // 2
    w = self.linear.weight
    p0 = F.linear(o[..., :half], w[:, :half])
    p1 = F.linear(o[..., half:], w[:, half:])
    return (p0.float() + p1.float()).to(o.dtype)


def value_fp16(module, args, out):
    return out.to(torch.float16).to(out.dtype) if FLAGS["b"] else out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokens", nargs="+", required=True)
    p.add_argument("--rows", default="0-3,-28--1")
    p.add_argument("--modes", default="none,a,b,ab")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    threads = inkref.set_threads(a.threads)
    model, info = inkref.load_model(a.checkpoint, "native")
    attn = model.model.layers[0].self_attn
    attn.o_proj.forward = types.MethodType(rank_split_output, attn.o_proj)
    attn.v_proj.register_forward_hook(value_fp16)
    for mode in a.modes.split(","):
        FLAGS.update(a="a" in mode, b="b" in mode)
        for t in a.tokens:
            args = argparse.Namespace(
                impl="native", rows=a.rows, logit_rows="none", parts="all", chunk=0, checkpoint=a.checkpoint
            )
            ref_dump.dump_one(args, model, info, threads, inkref.read_ids(t), Path(a.out) / mode / Path(t).stem)


if __name__ == "__main__":
    main()
