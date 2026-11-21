# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Synthetic DSpark drafter (``mtp.safetensors`` + ``mtp_config``) for a synthetic Inkling 2 checkpoint.

  synthetic_mtp.py --checkpoint SYNTH_DIR --out DIR [--seed 4321]

DIR receives symbolic links to every file of SYNTH_DIR (synthetic.py), its config.json with an ``mtp_config`` of the
real drafter's topology at the checkpoint's widths (three layers, layers 0 and 2 local with the checkpoint's
window, block of 8, recurrent rank 256, target_layer_ids [5, 23, 41, 48], head dimension 128 and the checkpoint's
query/KV heads), and ``mtp.safetensors`` with the real file's tensor names, all BF16, drawn from the seed: norms near
one, projections scaled by 1/sqrt(fan-in), a recurrent head whose gates are neither saturated nor constant.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def tensors(text: dict, mtp: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    h, nh, nkv, d = mtp["hidden_size"], mtp["num_attention_heads"], mtp["num_key_value_heads"], mtp["head_dim"]
    f, r, vp = mtp["intermediate_size"], mtp["markov_rank"], text["vocab_size"]

    def normal(shape, std):
        return (torch.randn(shape, generator=g) * std).to(torch.bfloat16)

    def norm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    out = {
        "fuse_weight": normal((mtp["num_hidden_layers"] + 1, 4), 1.0),
        "mask_embed": normal((h,), 1.0),
        "norm.weight": norm(h),
    }
    for i in range(mtp["num_hidden_layers"]):
        p, a = f"layers.{i}.", f"layers.{i}.self_attn."
        out.update(
            {
                p + "dflare_norm.weight": norm(h),
                p + "input_layernorm.weight": norm(h),
                p + "post_attention_layernorm.weight": norm(h),
                a + "q_proj.weight": normal((nh * d, h), 1 / math.sqrt(h)),
                a + "k_proj.weight": normal((nkv * d, h), 1 / math.sqrt(h)),
                a + "v_proj.weight": normal((nkv * d, h), 1 / math.sqrt(h)),
                a + "o_proj.weight": normal((h, nh * d), 0.5 / math.sqrt(nh * d)),
                a + "q_norm.weight": norm(d),
                a + "k_norm.weight": norm(d),
                a + "k_proj_ctx.weight": normal((nkv * d, h), 1 / math.sqrt(h)),
                a + "v_proj_ctx.weight": normal((nkv * d, h), 1 / math.sqrt(h)),
                a + "k_norm_ctx.weight": norm(d),
                p + "mlp.gate_proj.weight": normal((f, h), 1 / math.sqrt(h)),
                p + "mlp.up_proj.weight": normal((f, h), 1 / math.sqrt(h)),
                p + "mlp.down_proj.weight": normal((h, f), 0.5 / math.sqrt(f)),
            }
        )
    m = "markov_head."
    out.update(
        {
            m + "dflare_norm.weight": norm(h),
            m + "markov_w1.weight": normal((vp, r), 1.0),
            m + "rnn_proj.weight": normal((3 * r, 2 * r + 2 * h), 1 / math.sqrt(2 * r + 2 * h)),
            m + "rnn_proj.bias": normal((3 * r,), 0.1),
            m + "hidden_out.weight": normal((h, r), 1 / math.sqrt(r)),
            m + "hidden_out.bias": normal((h,), 0.1),
        }
    )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="synthetic checkpoint (synthetic.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=4321)
    a = p.parse_args()
    from safetensors.torch import save_file

    src, out = Path(a.checkpoint).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    mtp = {
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": 3,
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text["num_key_value_heads"],
        "head_dim": 128,
        "intermediate_size": 4 * text["hidden_size"],
        "max_position_embeddings": text["model_max_length"],
        "torch_dtype": "bfloat16",
        "q_bias": False,
        "o_bias": False,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "local_layer_ids": [0, 2],
        "sliding_window_size": text["sliding_window_size"],
        "block_size": 8,
        "markov_rank": 256,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [5, 23, 41, 48],
    }
    for item in src.iterdir():
        target = out / item.name
        if item.name not in ("config.json", "mtp.safetensors") and not target.exists() and item.suffix != ".ninfer":
            target.symlink_to(item)
    config["mtp_config"] = mtp
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(tensors(text, mtp, a.seed), str(out / "mtp.safetensors"), metadata={"format": "pt"})
    print(f"wrote {out / 'mtp.safetensors'} and the mtp_config of {out / 'config.json'}")


if __name__ == "__main__":
    main()
