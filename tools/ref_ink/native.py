# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""The `native` variant of the reference: NVFP4 experts with their exact weights instead of BF16-rounded ones.

transformers dequantizes the NVFP4 experts on load to value = e2m1 * e4m3 * (amax / denominator) in FP32 and rounds
it to BF16. Everything else of the reference stays as it is; only the NVFP4 expert layers change:

  W  = Q * f            Q = e2m1 * e4m3 (at most 5 significant bits: exact in BF16), f = amax / denominator (FP32),
                        multiplied in FP32 exactly as the loader does before its BF16 rounding
  gu = bf16(u @ W13^T)  FP32 product of the BF16 input with the exact weights, rounded where the reference rounds
  h  = act(gu)          the module's softcapped SwiGLU (FP32 inside, BF16 out), unchanged
  o  = bf16(h @ W2^T)
  y  = bf16(sum_k fp32(o_k * w_k))   the top-k sum of grouped_mm (FP32, over the k slots of each token)

Memory: the model keeps Q in place of the BF16-rounded weights (same tensor, same dtype: no extra RAM) and one FP32
factor per expert; W of the experts hit by a call is built per call. MXFP8 layers (f8 * 2^k, exact in BF16) and the
routing are untouched.

  native.py --checkpoint DIR [--layer 0]   checks one layer: bf16(Q * f) equals the loaded parameter bit for bit,
                                           times and memory of building W for all experts
"""

from __future__ import annotations

import argparse
import time
import types
from pathlib import Path

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _lut():
    e2m1 = torch.tensor(E2M1)
    byte = torch.arange(256)
    return torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)


def nvfp4_q(codes, scale, lut=None):
    """Q = e2m1 * e4m3 in FP32 [E, rows, cols] (exact), from the packed codes and the block scales."""
    lut = _lut() if lut is None else lut
    e, r = codes.shape[:2]
    out = torch.empty(e, r, codes.shape[2] * 2)
    for s in range(0, e, 8):
        sl = slice(s, s + 8)
        v = lut[codes[sl].long()].view(*scale[sl].shape, 16) * scale[sl].float()[..., None]
        out[sl] = v.reshape(-1, r, codes.shape[2] * 2)
    return out


def deinterleave(w):
    e, r, c = w.shape
    return w.reshape(e, r // 2, 2, c).transpose(1, 2).reshape(e, r, c).contiguous()


def read_layer(f, i):
    """(Q13 [E, 2I, H] in the gate_up_proj row order, (amax13, den13), Q2 [E, H, I], (amax2, den2)) or None."""
    base = f"model.llm.layers.{i}.mlp.experts."
    if base + "w13_weight_amax" not in set(f.keys()):
        return None
    out = []
    for w in ("w13_weight", "w2_weight"):
        q = nvfp4_q(f.get_tensor(base + w), f.get_tensor(base + w + "_scale"))
        if w == "w13_weight":
            q = deinterleave(q)
        amax, den = f.get_tensor(base + w + "_amax"), f.get_tensor(base + w + "_outer_scale_denominator")
        out += [q, (amax, den)]
    return out


def loader_value(q, amax_den, dtype):
    """What transformers holds for an NVFP4 expert loaded in `dtype`: its loader casts every source tensor to the
    parameter dtype first, so amax and the denominator are rounded to BF16 before f = amax / den is formed."""
    amax, den = amax_den
    return (q * (amax.to(dtype).float() / den.to(dtype).float())[:, None, None]).to(dtype)


def native_forward(self, hidden_states, top_k_index, top_k_weights):
    self.native_calls += 1
    n, k = top_k_index.shape
    dt = hidden_states.dtype
    slots = torch.zeros(n, k, hidden_states.shape[-1], dtype=torch.float32)
    for e in torch.unique(top_k_index).tolist():
        tok, slot = torch.where(top_k_index == e)
        x = hidden_states[tok].float()
        w13 = self.gate_up_proj[e].float() * self.native_f13[e]
        gu = torch.nn.functional.linear(x, w13).to(dt)
        h = self._apply_gate(gu)
        w2 = self.down_proj[e].float() * self.native_f2[e]
        o = torch.nn.functional.linear(h.float(), w2).to(dt)
        slots[tok, slot] = o * top_k_weights[tok, slot, None]
    return slots.sum(dim=1).to(dt)


def install(model, checkpoint):
    """Replaces the NVFP4 expert layers of a loaded model with the native variant. Returns the layer indices."""
    from safetensors import safe_open

    files = sorted(Path(checkpoint).glob("model-*.safetensors"))
    done = []
    for path in files:
        with safe_open(str(path), framework="pt") as f:
            for i, layer in enumerate(model.model.layers):
                r = read_layer(f, i)
                if r is None:
                    continue
                q13, ad13, q2, ad2 = r
                ex = layer.mlp.experts
                for q, ad, param in ((q13, ad13, ex.gate_up_proj), (q2, ad2, ex.down_proj)):
                    if not torch.equal(loader_value(q, ad, param.dtype), param.data):
                        raise ValueError(f"layer {i}: decoded NVFP4 does not match the loaded expert weights")
                f13 = ad13[0].float() / ad13[1].float()  # the exact factor, FP32
                f2 = ad2[0].float() / ad2[1].float()
                with torch.no_grad():
                    ex.gate_up_proj.data.copy_(q13.to(ex.gate_up_proj.dtype))
                    ex.down_proj.data.copy_(q2.to(ex.down_proj.dtype))
                    if ex.gate_up_proj.dtype == torch.bfloat16:  # Q must survive the copy exactly
                        assert torch.equal(ex.gate_up_proj.float(), q13) and torch.equal(ex.down_proj.float(), q2)
                ex.native_f13, ex.native_f2, ex.native_calls = f13, f2, 0
                ex.forward = types.MethodType(native_forward, ex)
                done.append(i)
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    from safetensors import safe_open

    path = sorted(Path(a.checkpoint).glob("model-*.safetensors"))[0]
    t0 = time.time()
    with safe_open(str(path), framework="pt") as f:
        q13, ad13, q2, ad2 = read_layer(f, a.layer)
    f13, f2 = ad13[0].float() / ad13[1].float(), ad2[0].float() / ad2[1].float()
    amax_err = (ad13[0].bfloat16().float() - ad13[0].float()) / ad13[0].float()
    t_read = time.time() - t0
    exact = torch.equal(q13.to(torch.bfloat16).float(), q13) and torch.equal(q2.to(torch.bfloat16).float(), q2)
    w13 = q13 * f13[:, None, None]
    rel = float((w13.to(torch.bfloat16).float() - w13).norm() / w13.norm())
    q13b, q2b = q13.to(torch.bfloat16), q2.to(torch.bfloat16)
    t1 = time.time()
    for e in range(q13.shape[0]):
        _ = q13b[e].float() * f13[e]
        _ = q2b[e].float() * f2[e]
    t_build = time.time() - t1
    mib = (q13[0].numel() + q2[0].numel()) * 4 / 2**20
    print(
        f"layer {a.layer}: read+decode {t_read:.1f} s; Q exact in BF16: {exact}; rel L2 of the BF16 rounding of "
        f"W13: {rel:.3e}; f13 {float(f13.min()):.3e}..{float(f13.max()):.3e}"
    )
    hf = loader_value(q13, ad13, torch.bfloat16).float()
    print(
        f"transformers BF16 value (amax rounded to BF16 first): rel L2 {float((hf - w13).norm() / w13.norm()):.3e} "
        f"from the exact W13; per-expert scale error of bf16(amax): max {float(amax_err.abs().max()):.3e} "
        f"rms {float(amax_err.pow(2).mean().sqrt()):.3e}"
    )
    print(
        f"W in FP32 for all {q13.shape[0]} experts of the layer: {t_build:.2f} s ({mib:.1f} MiB per expert, "
        f"{mib * q13.shape[0] / 1024:.2f} GiB per layer, {mib * q13.shape[0] * 20 / 1024:.1f} GiB for the 20 layers)"
    )


if __name__ == "__main__":
    main()
