# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Shared pieces of the Inkling 2 Mini reference (tools/ref_ink): model loading in the three reference precisions,
the FP64 promotion mode, the visit table (48 visits over 24 layers) and prompt tokenization.

Precisions (--impl):
  bf16-hf  Inkling2LoopForCausalLM.from_pretrained(dtype=bfloat16): the transformers reference as published (experts
        dequantized NVFP4/MXFP8 -> BF16 on load, the modules in _keep_in_fp32_modules_strict in FP32). lm_head runs in
        BF16, so the logits are BF16 values (step 0.125 at |logit| 16-32) and exact top-1 ties are common.
  bf16  THE REFERENCE: bf16-hf with lm_head in FP32 (its BF16 weights and BF16 input widened exactly, FP32 output),
        as tools/ref_q4 does for Qwen4: same model, logits without the final BF16 rounding.
  native  bf16 with the NVFP4 experts at their exact weights (native.py): what an engine that keeps NVFP4 computes.
  native-fp32  fp32 with the native experts module: a check of native.py (must match fp32 up to summation order).
  fp32  the same with dtype=float32 (experts dequantized in FP32, every module in FP32).
  fp64  the fp32 load cast to float64, run under Promote64: every explicit float32 cast inside the modeling code
        (.float(), .to(float32), softmax(dtype=float32), zeros_like(dtype=float32)) is redirected to float64,
        so the whole forward runs in FP64. CPU only. Experts with the "eager" implementation (the per-expert loop
        of modeling_inkling2_loop.py); bf16 and fp32 keep the library default ("grouped_mm" when available).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

_LOW = (torch.float32, torch.float16, torch.bfloat16)


def _promote(x):
    if isinstance(x, torch.dtype):
        return torch.float64 if x in _LOW else x
    if isinstance(x, (list, tuple)):
        return type(x)(_promote(v) for v in x)
    if isinstance(x, dict):
        return {k: _promote(v) for k, v in x.items()}
    return x


class Promote64(TorchFunctionMode):
    """Runs every float op of the forward in float64 (see the module docstring)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in (torch.Tensor.float, torch.Tensor.half, torch.Tensor.bfloat16):
            func = torch.Tensor.double
        return func(*_promote(args), **_promote(kwargs or {}))

    def __enter__(self):
        self._default = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        return super().__enter__()

    def __exit__(self, *exc):
        torch.set_default_dtype(self._default)
        return super().__exit__(*exc)


def load_model(checkpoint, impl, device="cpu"):
    """Returns (model, info). info: load seconds, dtype of the expert weights, loading report."""
    from transformers import Inkling2LoopForCausalLM

    dtype = {
        "bf16": torch.bfloat16,
        "bf16-hf": torch.bfloat16,
        "native": torch.bfloat16,
        "fp32": torch.float32,
        "native-fp32": torch.float32,
        "fp64": torch.float32,
    }[impl]
    t0 = time.time()
    kwargs = {"dtype": dtype, "attn_implementation": "eager", "output_loading_info": True}
    if impl == "fp64":  # grouped_mm (the library default for the experts) has no float64 kernel
        kwargs["experts_implementation"] = "eager"
    if device == "auto":
        kwargs["device_map"] = "auto"
    model, report = Inkling2LoopForCausalLM.from_pretrained(str(checkpoint), **kwargs)
    if impl == "fp64":
        model = model.double()
    native_layers = []
    if impl.startswith("native"):
        import native

        native_layers = native.install(model, checkpoint)
    if impl in ("bf16", "native"):
        model.lm_head.float()
        model.lm_head.register_forward_pre_hook(lambda module, args: (args[0].float(),))
    if device not in ("cpu", "auto"):
        model = model.to(device)
    model.eval()
    report = {k: sorted(v) if isinstance(v, (set, list)) else str(v) for k, v in report.items()}
    experts = model.model.layers[0].mlp.experts.gate_up_proj
    info = {
        "load_s": round(time.time() - t0, 1),
        "impl": impl,
        "expert_dtype": str(experts.dtype),
        "lm_head_dtype": str(model.lm_head.weight.dtype),
        "native_layers": native_layers,
        "experts_implementation": model.config._experts_implementation,
        "dtypes": dtype_census(model),
        "missing_keys": report.get("missing_keys", []),
        "unexpected_keys_n": len(report.get("unexpected_keys", [])),
        "unexpected_keys_head": report.get("unexpected_keys", [])[:5],
        "mismatched_keys": report.get("mismatched_keys", []),
    }
    return model, info


def dtype_census(model):
    """Bytes of parameters and buffers per dtype (GiB)."""
    out = {}
    for t in list(model.parameters()) + list(model.buffers()):
        k = str(t.dtype).replace("torch.", "")
        out[k] = out.get(k, 0) + t.numel() * t.element_size()
    return {k: round(v / 2**30, 3) for k, v in sorted(out.items())}


def visit_table(cfg):
    """[(visit, layer, pass, kind)] in execution order: prelude, loop passes, coda. kind = local | global."""
    s, e, passes, n = cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.loop_num_passes, cfg.num_hidden_layers
    order = [(l, 0) for l in range(s)]
    order += [(l, p) for p in range(passes) for l in range(s, e)]
    order += [(l, 0) for l in range(e, n)]
    return [
        (v, l, p, "local" if cfg.layer_types[l] == "hybrid_sliding" else "global") for v, (l, p) in enumerate(order)
    ]


def set_threads(n):
    if n:
        torch.set_num_threads(n)
    return torch.get_num_threads()


def env_info():
    import platform

    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_path": os.path.dirname(transformers.__file__),
        "python": platform.python_version(),
        "threads": torch.get_num_threads(),
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def chat_ids(tokenizer, messages, effort=0.0, tools=None):
    """Prompt ids from the model's chat template, with the generation prompt and a fixed thinking effort."""
    kw = {"add_generation_prompt": True, "tokenize": True, "reasoning_effort": effort, "return_dict": False}
    if tools:
        kw["tools"] = tools
    ids = tokenizer.apply_chat_template(messages, **kw)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def read_ids(spec):
    """Token ids from a .npy file, a .json ({"ids": [...]}, a list, or a greedy.py record = prompt + generated),
    a text file of integers or an inline comma/space list."""
    p = Path(spec)
    if p.exists():
        if p.suffix == ".npy":
            import numpy as np

            return [int(i) for i in np.load(p)]
        if p.suffix == ".json":
            d = json.loads(p.read_text())
            if isinstance(d, dict) and "prompt_ids" in d:  # a greedy.py record: prompt + generated tokens
                return [int(i) for i in d["prompt_ids"] + d["generated_ids"]]
            return [int(i) for i in (d["ids"] if isinstance(d, dict) else d)]
        spec = p.read_text()
    return [int(i) for i in spec.replace(",", " ").split()]
