# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Synthetic Inkling 2 Mini checkpoint in the ORIGINAL checkpoint format, from a fixed seed.

  synthetic.py --source ~/models/inkling-2-mini --out DIR [--seed 1234] [--hidden 512 --heads 6 --kv-heads 2 ...]

Same topology as the real model (24 layers, globals 5/11/17/23, loop 6..17 x 3 passes = 48 visits, attention
residual blocks of 6, controllers on the loop layers, dynamic K/V convolutions on the globals, relative position
bias, attention sinks, 128-dim heads, real vocabulary 201024 padded / 200058 unpadded), smaller widths, and a
window / relative extent / log-scaling floor short enough that a 256-token sequence crosses all three.

Every parameter is drawn from the seed with non-trivial values (L2-clip scales that clip some heads, nonzero
residual and message queries, nonzero dynamic-convolution taps, router bias). The file holds the tensors under
their original names (model.llm.*), in the original dtypes: BF16 dense weights, FP32 residual queries / message
queries / router bias, and the experts QUANTIZED like the real checkpoint:
  layers 0-19  NVFP4: w13_weight/w2_weight U8 [E, rows, cols/2] (two E2M1 codes per byte, even column in the low
               nibble), _scale F8_E4M3 [E, rows, cols/16], _amax F32 [E], _outer_scale_denominator F32 [1];
               value = e2m1 * scale * (amax / denominator)
  layers 20-23 MXFP8: w13_weight/w2_weight F8_E4M3 [E, rows, cols], _scale U8 [E, rows, cols/32] (exponent,
               value = f8 * 2^(scale-127)), one scale per row and 32 columns (the real file's scales vary within
               32-row blocks although hf_quant_config.json says weight_block_shape [32, 32])
  w13 rows are gate/up interleaved (row 2i = gate i, row 2i+1 = up i); w2 = down [E, hidden, inter].
The unembedding is stored padded to vocab_size with zero rows. Tokenizer and template files are copied from
--source; config.json and hf_quant_config.json follow the real ones.

Check: the checkpoint is reloaded with Inkling2LoopForCausalLM.from_pretrained(dtype=float32) and every parameter
must equal, bit for bit, the generator's (BF16-rounded / dequantized) values; then one forward in fp32 and fp64.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkref  # noqa: E402

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
NVFP4_LAYERS_END = 20  # layers below are NVFP4, the rest MXFP8 (as in the real checkpoint)
# Conditioning: random weights make the 48-visit stack chaotic, and the BF16 reference would drift from the FP64
# oracle until uncorrelated, which leaves nothing to compare an engine against. Measured on this generator (T=256,
# BF16 vs FP64 at the last visit): the message gates are the main amplifier ((1 - forget) reaches 2 in each loop layer:
# gate_std 1.0 -> 0.2 takes rel 1.0 to 0.04); L2-clip scales above ~1.2 sharpen attention; any top-k < experts flips
# routings at near-ties (1-3 % of rows per visit at BF16 noise) and each flip spreads through attention to every later
# position. Default: every expert selected (topk = experts, routing weights still per token) and the knobs below.
# --topk 4 gives a routed model for checks of the selection itself (expect flips against BF16).
KNOBS = {"query_std": 0.5, "clip_hi": 1.2, "router_std": 3.0, "down_gain": 0.05, "gate_std": 0.2, "bias_std": 0.1}
COPY = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "tiktoken",
]


def synth_config(source, a):
    cfg = json.loads((Path(source) / "config.json").read_text())
    t = cfg["text_config"]
    t.update(
        hidden_size=a.hidden,
        num_attention_heads=a.heads,
        num_key_value_heads=a.kv_heads,
        head_dim=128,
        swa_num_attention_heads=a.heads,
        swa_num_key_value_heads=a.kv_heads,
        swa_head_dim=128,
        sliding_window_size=a.window,
        rel_extent=a.rel_extent,
        log_scaling_n_floor=a.n_floor,
        n_routed_experts=a.experts,
        num_experts_per_tok=a.topk,
        intermediate_size=a.inter,
    )
    cfg["vision_config"]["text_hidden_size"] = a.hidden
    cfg.pop("mtp_config", None)
    return cfg


class Gen:
    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(seed)

    def normal(self, shape, std=1.0):
        return torch.randn(shape, generator=self.g, dtype=torch.float64).mul_(std).float()

    def uniform(self, shape, lo, hi):
        return (torch.rand(shape, generator=self.g, dtype=torch.float64) * (hi - lo) + lo).float()

    def ints(self, shape, lo, hi):
        return torch.randint(lo, hi, shape, generator=self.g)


def dense_value(gen, name, shape):
    """The synthetic value of one non-expert parameter or buffer, by role."""
    if name.endswith(("layernorm.weight", "embed_norm.weight")) or name == "model.norm.weight":
        return 1.0 + gen.normal(shape, 0.1)
    if name.endswith("norm.scale"):  # q / relative-state L2-clip scales: some heads clip, some do not
        return gen.uniform(shape, 0.8, KNOBS["clip_hi"])
    if name.endswith("relative_position_bias.proj"):
        return gen.normal(shape, 0.5)
    if name.endswith("query"):  # attention-residual and message queries: soft mixing
        return gen.normal(shape, KNOBS["query_std"])
    if name.endswith("conv1d.weight"):
        return gen.normal(shape, 0.5)
    if name.endswith("static_weight"):
        return gen.normal(shape, 0.3)
    if name.endswith("sconv.weight_proj.weight"):
        return gen.normal(shape, 0.3 / math.sqrt(shape[-1]))
    if name.endswith("e_score_correction_bias"):
        return gen.normal(shape, KNOBS["bias_std"])
    if name.endswith(("forget_gate_proj.weight", "message_gate_proj.weight")):
        return gen.normal(shape, KNOBS["gate_std"] / math.sqrt(shape[-1]))
    if name.endswith("mlp.gate.weight"):  # router logits ~N(0, 3^2): selections with clear margins
        return gen.normal(shape, KNOBS["router_std"] / math.sqrt(shape[-1]))
    if name == "model.embed_tokens.weight":
        return gen.normal(shape, 1.0)
    if name == "lm_head.weight":
        return gen.normal(shape, 3.0 / math.sqrt(shape[-1]))
    if len(shape) == 2:  # every other linear
        return gen.normal(shape, 1.0 / math.sqrt(shape[-1]))
    raise ValueError(f"no synthetic rule for {name} {tuple(shape)}")


def nvfp4(gen, num_experts, rows, cols, std):
    codes = gen.ints((num_experts, rows, cols // 2), 0, 256).to(torch.uint8)
    scale = torch.exp2(gen.uniform((num_experts, rows, cols // 16), -1.0, 1.0)).to(torch.float8_e4m3fn)
    denominator = torch.tensor([6.0 * 448.0])
    rms = math.sqrt(sum(v * v for v in E2M1) / 16) * 1.1
    amax = (std / rms * denominator) * gen.uniform((num_experts,), 0.8, 1.2)
    return {"": codes, "_scale": scale, "_amax": amax.float(), "_outer_scale_denominator": denominator.float()}


def mxfp8(gen, num_experts, rows, cols, std):
    values = (gen.normal((num_experts, rows, cols), 2.0)).to(torch.float8_e4m3fn)
    exp = torch.round(math.log2(std / 2.0) + gen.uniform((num_experts, rows, cols // 32), -1.0, 1.0))
    scale = (exp + 127).to(torch.uint8)
    return {"": values, "_scale": scale}


def dequant(q):
    """The loader's arithmetic (Inkling2LoopDequantizeExperts), in FP32, checkpoint row order."""
    weight, scale = q[""], q["_scale"]
    num_experts, rows = weight.shape[:2]
    if "_amax" in q:
        e2m1 = torch.tensor(E2M1)
        byte = torch.arange(256)
        byte_to_values = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
        factor = q["_amax"].float() / q["_outer_scale_denominator"].float()
        values = byte_to_values[weight.long()].view(*scale.shape, 16) * scale.float()[..., None]
        values = values * factor[:, None, None, None]
    else:
        values = weight.float().view(*scale.shape, 32) * torch.exp2(scale.float() - 127)[..., None]
    return values.reshape(num_experts, rows, -1)


def deinterleave(w):
    """Checkpoint w13 (gate/up interleaved rows) -> transformers gate_up_proj (gate rows, then up rows)."""
    e, r, c = w.shape
    return w.reshape(e, r // 2, 2, c).transpose(1, 2).reshape(e, r, c).contiguous()


ATTN_RENAMES = [  # transformers name piece -> original name piece, inside layers.N.
    ("self_attn.q_proj.linear.weight", "attn.wq_du.weight"),
    ("self_attn.q_proj.norm.scale", "attn.q_l2_clip.scale"),
    ("self_attn.k_proj.linear.weight", "attn.wk_dv.weight"),
    ("self_attn.v_proj.linear.weight", "attn.wv_dv.weight"),
    ("self_attn.k_proj.sconv.conv1d.weight", "attn.k_sconv.weight"),
    ("self_attn.v_proj.sconv.conv1d.weight", "attn.v_sconv.weight"),
    ("self_attn.k_proj.sconv.static_weight", "attn.k_sconv.static_w"),
    ("self_attn.v_proj.sconv.static_weight", "attn.v_sconv.static_w"),
    ("self_attn.k_proj.sconv.weight_proj.weight", "attn.k_sconv.weight_proj"),
    ("self_attn.v_proj.sconv.weight_proj.weight", "attn.v_sconv.weight_proj"),
    ("self_attn.o_proj.linear.weight", "attn.wo_ud.weight"),
    ("self_attn.o_proj.gate_proj.weight", "attn.gate_attn.weight"),
    ("self_attn.sink_proj.weight", "attn.attn_sink_proj.weight"),
    ("self_attn.relative_position_bias.linear.weight", "attn.wr_du.weight"),
    ("self_attn.relative_position_bias.norm.scale", "attn.rel_l2_clip.scale"),
    ("self_attn.relative_position_bias.proj", "attn.rel_logits_proj.proj"),
    ("input_layernorm.weight", "attn_norm.weight"),
    ("q_input_layernorm.weight", "q_input_norm.weight"),
    ("k_input_layernorm.weight", "k_input_norm.weight"),
    ("v_input_layernorm.weight", "v_input_norm.weight"),
    ("post_attention_layernorm.weight", "mlp_norm.weight"),
    ("post_attention_convolution.conv1d.weight", "attn_sconv.weight"),
    ("post_mlp_convolution.conv1d.weight", "mlp_sconv.weight"),
    ("mlp.gate.weight", "mlp.gate.weight"),
    ("mlp.gate.e_score_correction_bias", "mlp.gate.bias"),
]
GATES = {"q": "attn", "k": "k", "v": "v", "mlp": "mlp"}
GATE_PROJ = {"down_proj": "down_weight", "forget_gate_proj": "forget_weight", "message_gate_proj": "gain_weight"}


def original_name(name):
    """transformers parameter name -> (original name, original dtype). None for the merged message queries."""
    fixed = {
        "lm_head.weight": "model.llm.unembed.weight",
        "model.embed_tokens.weight": "model.llm.embed.weight",
        "model.embed_tokens.embed_norm.weight": "model.llm.embed_norm.weight",
        "model.norm.weight": "model.llm.norm.weight",
        "model.final_input.query": "model.llm.attn_res.output_query",
    }
    if name in fixed:
        return fixed[name], torch.float32 if name.endswith("query") else torch.bfloat16
    if name == "model.message_input.query":
        return None, torch.float32
    parts = name.split(".")
    assert parts[:2] == ["model", "layers"], name
    i, rest = int(parts[2]), ".".join(parts[3:])
    if rest == "attention_input.query":
        return f"model.llm.attn_res.attn_queries.{i - 1}", torch.float32
    if rest == "mlp_input.query":
        return f"model.llm.attn_res.mlp_queries.{i}", torch.float32
    for gate, orig in GATES.items():
        prefix = f"{gate}_message_gate."
        if rest.startswith(prefix):
            proj = rest[len(prefix) :].removesuffix(".weight")
            return f"model.llm.layers.{i}.controllers.{orig}.{GATE_PROJ[proj]}", torch.bfloat16
    for hf, orig in ATTN_RENAMES:
        if rest == hf:
            dt = torch.float32 if hf.endswith("e_score_correction_bias") else torch.bfloat16
            return f"model.llm.layers.{i}.{orig}", dt
    raise KeyError(name)


def quant_config(tensors, nvfp4_names, mxfp8_names):
    f32 = sorted(k for k, v in tensors.items() if v.dtype == torch.float32 and "experts" not in k)
    bf16 = sorted(k for k, v in tensors.items() if v.dtype == torch.bfloat16 and k != "model.llm.unembed.weight")
    cast = {"required_activation_quantization": None, "weight_block_shape": [], "weight_transform": "cast"}
    return {
        "model": {
            "parameter_groups": [
                {"compute_dtype": "float32", "output_dtype": "float32", **cast, "tensors": f32},
                {"compute_dtype": "bfloat16", "output_dtype": "bfloat16", **cast, "tensors": bf16},
                {
                    "compute_dtype": "nvfp4",
                    "output_dtype": "uint8",
                    "required_activation_quantization": {
                        "amax_type": "per_token",
                        "block_size": 16,
                        "format": "nvfp4_e2m1",
                        "rounding": "nearest",
                        "scale_dtype": "float8_e4m3fn",
                        "scale_selection_rule": "fouroversix_mae",
                    },
                    "weight_block_shape": [1, 16],
                    "weight_scale_format": "amax_denominator",
                    "weight_transform": "nvfp4_nearest",
                    "tensors": sorted(nvfp4_names),
                },
                {
                    "compute_dtype": "mxfp8",
                    "output_dtype": "float8_e4m3fn",
                    "required_activation_quantization": {
                        "block_size": 32,
                        "format": "mxfp8_e4m3",
                        "rounding": "nearest",
                        "scale_dtype": "ue8m0",
                    },
                    "weight_block_shape": [32, 32],
                    "weight_transform": "mxfp8_nearest",
                    "tensors": sorted(mxfp8_names),
                },
                {
                    "compute_dtype": "bfloat16",
                    "fwd_output_dtype": "float32",
                    "output_dtype": "bfloat16",
                    **cast,
                    "tensors": ["model.llm.unembed.weight"],
                },
            ]
        },
        "quant_method": "inkling2",
    }


def generate(a):
    from safetensors.torch import save_file
    from transformers import Inkling2LoopForCausalLM, Inkling2LoopTextConfig

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(a.source)
    for f in COPY:
        if (src / f).is_dir():
            shutil.copytree(src / f, out / f, dirs_exist_ok=True)
        elif (src / f).exists():
            shutil.copy2(src / f, out / f)
    cfg = synth_config(src, a)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    config = Inkling2LoopTextConfig.from_pretrained(str(out))
    torch.manual_seed(a.seed)
    model = Inkling2LoopForCausalLM(config).float().eval()

    gen = Gen(a.seed)
    tensors, expected = {}, {}
    nvfp4_names, mxfp8_names = [], []
    named = dict(model.named_parameters())
    named.update(dict(model.named_buffers()))
    for name in sorted(named):
        p = named[name]
        if ".mlp.experts." in name:
            continue
        orig, dt = original_name(name)
        v = dense_value(gen, name, tuple(p.shape))
        v = v.to(dt).float()  # the value the checkpoint can hold
        expected[name] = v
        if orig is None:  # message queries: (loop layers, 4 [q, k, v, mlp], hidden) -> two (2 * loop layers, hidden)
            tensors["model.llm.middle_cycle_cross_pass_queries"] = v[:, [0, 3]].flatten(0, 1).contiguous()
            tensors["model.llm.middle_cycle_cross_pass_kv_queries"] = v[:, 1:3].flatten(0, 1).contiguous()
        elif name == "lm_head.weight":
            pad = torch.zeros(config.vocab_size - v.shape[0], v.shape[1])
            tensors[orig] = torch.cat([v, pad]).to(dt)
        else:
            tensors[orig] = v.to(dt).contiguous()
    h, inter, ne = config.hidden_size, config.moe_intermediate_size, config.n_routed_experts
    for i in range(config.num_hidden_layers):
        base = f"model.llm.layers.{i}.mlp.experts."
        quant = nvfp4 if i < NVFP4_LAYERS_END else mxfp8
        for w, rows, cols, hf, gain in (
            ("w13_weight", 2 * inter, h, "gate_up_proj", 1.0),
            ("w2_weight", h, inter, "down_proj", KNOBS["down_gain"]),
        ):
            q = quant(gen, ne, rows, cols, gain / math.sqrt(cols))
            for suffix, t in q.items():
                tensors[base + w + suffix] = t.contiguous()
            dense = dequant(q)
            expected[f"model.layers.{i}.mlp.experts.{hf}"] = deinterleave(dense) if w == "w13_weight" else dense
            (nvfp4_names if quant is nvfp4 else mxfp8_names).append(base + w)

    fname = "model-00001-of-00001.safetensors"
    save_file(tensors, str(out / fname), metadata={"format": "pt"})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": {k: fname for k in sorted(tensors)}}, indent=2)
    )
    (out / "hf_quant_config.json").write_text(json.dumps(quant_config(tensors, nvfp4_names, mxfp8_names), indent=1))
    return config, expected, tensors, total


def check_reload(out, expected):
    model, info = inkref.load_model(out, "fp32")
    sd = dict(model.named_parameters())
    sd.update(dict(model.named_buffers()))
    bad, worst = [], 0.0
    for name, v in expected.items():
        got = sd[name].detach().float()
        if got.shape != v.shape or not torch.equal(got, v):
            d = float((got - v).abs().max()) if got.shape == v.shape else float("inf")
            worst = max(worst, d)
            bad.append(f"{name} shape {tuple(got.shape)}/{tuple(v.shape)} max|d| {d:.3g}")
    extra = sorted(set(sd) - set(expected))
    return model, info, bad, extra


def sample_ids(tokenizer, n):
    text = (
        "La sequenza di prova del modello sintetico deve essere più lunga della finestra locale, dell'estensione "
        "relativa e della soglia della scala logaritmica. The quick brown fox jumps over the lazy dog, then "
        "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
    )
    ids = inkref.chat_ids(tokenizer, [{"role": "user", "content": text * 8}], effort=0.0)
    assert len(ids) >= n, len(ids)
    return ids[:n]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True, help="real checkpoint dir (config, tokenizer and template files)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--rel-extent", type=int, default=96)
    p.add_argument("--n-floor", type=int, default=128)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--inter", type=int, default=256)
    p.add_argument("--tokens", type=int, default=256, help="length of the test sequence written to tokens.txt")
    p.add_argument("--threads", type=int, default=8)
    for k, v in KNOBS.items():
        p.add_argument("--" + k.replace("_", "-"), type=float, default=v, help="conditioning knob")
    a = p.parse_args()
    KNOBS.update({k: getattr(a, k) for k in KNOBS})
    inkref.set_threads(a.threads)
    t0 = time.time()
    config, expected, tensors, total = generate(a)
    print(f"generated {len(tensors)} tensors, {total / 2**20:.1f} MiB, {time.time() - t0:.1f} s")
    model, info, bad, extra = check_reload(a.out, expected)
    print(
        "reload:",
        json.dumps(
            {
                k: info[k]
                for k in (
                    "load_s",
                    "expert_dtype",
                    "experts_implementation",
                    "dtypes",
                    "missing_keys",
                    "unexpected_keys_n",
                    "mismatched_keys",
                )
            }
        ),
    )
    print(f"reload check: {len(expected)} tensors compared, {len(bad)} differ, {len(extra)} not generated {extra[:5]}")
    for b in bad[:10]:
        print("  DIFF", b)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.out)
    ids = sample_ids(tok, a.tokens)
    (Path(a.out) / "tokens.txt").write_text(" ".join(map(str, ids)) + "\n")
    x = torch.tensor([ids])
    with torch.no_grad():
        t1 = time.time()
        l32 = model(x, use_cache=False).logits[0].double()
        s32 = time.time() - t1
        model.set_experts_implementation("eager")
        m64 = model.double()
        t1 = time.time()
        with inkref.Promote64():
            l64 = m64(x, use_cache=False).logits[0]
        s64 = time.time() - t1
    print(f"forward T={len(ids)}: fp32 {s32:.1f} s, fp64 {s64:.1f} s, logits dtype fp64 run {l64.dtype}")
    d = (l32 - l64).abs()
    agree = float((l32.argmax(-1) == l64.argmax(-1)).double().mean())
    print(
        f"fp32 vs fp64: max|dlogit| {float(d.max()):.3e} mean {float(d.mean()):.3e} logit rms "
        f"{float(l64.pow(2).mean().sqrt()):.3f} top1 agree {agree:.4f}"
    )
    ok = not bad and not info["missing_keys"] and not extra and torch.isfinite(l64).all()
    print("SYNTHETIC", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
