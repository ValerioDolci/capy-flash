# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Teacher-forcing dump of the Inkling 2 Mini reference (transformers Inkling2LoopForCausalLM), per visit.

  ref_dump.py --checkpoint DIR --impl bf16|fp32|fp64 --tokens IDS --out DUMP_DIR [--rows SPEC] [--logit-rows SPEC]
  ref_dump.py --checkpoint DIR --impl bf16 --chat prompt.json --effort 0 --out DUMP_DIR ...
  ref_dump.py --checkpoint DIR --impl bf16 --tokens greedy/it1.json greedy/en1.json --out DIR   (DIR/it1, DIR/en1)

One forward over the whole sequence, without cache (or --chunk N: through the KV cache N positions at a time, another
valid computation order and the only way past ~3k tokens on the real model), captured with forward hooks
(transformers is not modified).
The output is a tfdump/1 directory (tools/ref_q4/tfdump.py), so tools/ref_q4/tf_compare.py compares it as is:

  tokens.npy                  int64 [T]
  logits.npy, rows.npy        f32 [R, 200058] final logits (after the softcap) at the --logit-rows positions
  token_logprobs.npy          f32 [T-1] log p(tokens[t+1] | ..t), from the FP32 logits of every position
  hidden/embed.npy            [R, H] embed_tokens output (embedding + embed_norm)
  hidden/V<vv>.npy            [R, H] output of visit vv (the decoder layer's return value)
  hidden/V<vv>.<part>.npy     [R, ...] inside visit vv:
      in        [H]    decoder-layer input hidden state
      ain       [H]    attention input mixed by the attention residual (globals: [3H] = query, key, value inputs;
                       layer 0 has none)
      msg       [4H]   messages from the previous pass (query, key, value, mlp), loop passes 2 and 3 only
      attn      [H]    attention block output (after the gated output projection, before attn_sconv)
      attn_conv [H]    after the post-attention short convolution
      min       [H]    MLP input mixed by the attention residual (before the mlp message gate and mlp_norm)
      moe_in    [H]    input of the MoE block (after mlp_norm)
      moe       [H]    MoE output (before mlp_sconv)
  hidden/final.npy            [R, H] input of lm_head (final attention residual + final norm)
  every hidden file has its <name>.rows.npy (sequence positions); R = --rows positions
  route/V<vv>.idx.npy         int32 [R, k] selected experts, ascending expert id
  route/V<vv>.w.npy           f32 [R, k] routing weights (normalized, times route_scale), same order
  route/V<vv>.logits.npy      f32 [R, E] router logits (before the sigmoid)
  route/rows.npy              int64 [R]
  meta.json                   producer, impl, environment, visit table [[visit, layer, pass, kind]], shapes

Visits: 0-5 = layers 0-5; 6-17, 18-29, 30-41 = layers 6-17 in passes 0, 1, 2; 42-47 = layers 18-23.
Row specs: "all", "a-b,c,..." (inclusive, negative = from the end), "last:N", "none".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "ref_q4"))
import inkref  # noqa: E402
import tfdump  # noqa: E402

PARTS = ("in", "ain", "msg", "attn", "attn_conv", "min", "moe_in", "moe")


def parse_rows(spec, n):
    spec = spec.strip()
    if spec == "all":
        return np.arange(n)
    if spec == "none":
        return np.zeros(0, np.int64)
    if spec.startswith("last:"):
        k = int(spec[5:])
        return np.arange(max(0, n - k), n)
    rows = set()
    for part in spec.split(","):
        m = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part.strip())
        a, b = int(m.group(1)), int(m.group(2) if m.group(2) is not None else m.group(1))
        rows.update(range(a % n if a < 0 else a, (b % n if b < 0 else min(b, n - 1)) + 1))
    return np.asarray(sorted(r for r in rows if 0 <= r < n), np.int64)


class Recorder:
    """Forward hooks that file every captured tensor under the current visit."""

    def __init__(self, model, rows, parts):
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        self.parts = set(parts)
        self.visit = -1
        self.local = self.rows  # rows of the current call, relative to its first position
        self.data: dict[str, list] = {}
        self.route: dict[str, list] = {}
        self.handles = []
        m = model.model
        self._hook(m.embed_tokens, post=lambda mod, a, o: self.put("embed", o))
        self._hook(m.norm, post=lambda mod, a, o: self.put("final", o))
        for layer in m.layers:
            self.handles.append(layer.register_forward_pre_hook(self.enter, with_kwargs=True))
            self._hook(layer, post=lambda mod, a, o: self.put(self.name(), o))
            if layer.attention_input is not None:
                self._hook(layer.attention_input, post=self.ain)
            self._hook(layer.self_attn, post=lambda mod, a, o: self.put(self.name("attn"), o[0]))
            self._hook(layer.post_attention_convolution, post=lambda mod, a, o: self.put(self.name("attn_conv"), o))
            self._hook(layer.mlp_input, post=lambda mod, a, o: self.put(self.name("min"), o[0]))
            self.handles.append(layer.mlp.register_forward_pre_hook(lambda mod, a: self.put(self.name("moe_in"), a[0])))
            self._hook(layer.mlp, post=lambda mod, a, o: self.put(self.name("moe"), o))
            self._hook(layer.mlp.gate, post=self.gate)

    def _hook(self, module, post):
        self.handles.append(module.register_forward_hook(post))

    def name(self, part=None):
        return f"V{self.visit:02d}" + (f".{part}" if part else "")

    def enter(self, module, args, kwargs):
        self.visit += 1
        hidden = args[0] if args else kwargs["hidden_states"]
        self.put(self.name("in"), hidden)
        messages = args[2] if len(args) > 2 else kwargs.get("messages")
        if messages is not None:
            self.put(self.name("msg"), messages.flatten(-2))

    def ain(self, module, args, out):
        self.put(self.name("ain"), torch.cat(list(out), dim=-1))

    def call(self, lo, hi):
        """A forward over positions [lo, hi) starts: visits count from 0 again."""
        self.visit = -1
        self.local = self.rows[(self.rows >= lo) & (self.rows < hi)] - lo

    def sel(self, t):
        t = t.detach()
        if t.dim() == 3:  # [batch, seq, ...] -> [seq, ...]
            t = t[0]
        return t[self.local.to(t.device)]

    def put(self, key, t):
        part = key.split(".", 1)[1] if "." in key else None
        if part is not None and part not in self.parts:
            return
        if len(self.local) == 0:
            return
        self.data.setdefault(key, []).append(self.sel(t).float().cpu().numpy())

    def gate(self, module, args, out):
        logits, weights, idx = out  # [N, E], [N, k], [N, k] with N = batch * seq
        if len(self.local) == 0:
            return
        rows = self.local.to(idx.device)
        idx, weights, logits = idx[rows], weights[rows], logits[rows]
        idx_sorted, order = idx.sort(dim=-1)
        v = f"V{self.visit:02d}"
        self.route.setdefault(f"{v}.idx", []).append(idx_sorted.int().cpu().numpy())
        self.route.setdefault(f"{v}.w", []).append(weights.gather(-1, order).float().cpu().numpy())
        self.route.setdefault(f"{v}.logits", []).append(logits.float().cpu().numpy())

    def arrays(self, store):
        return {k: np.concatenate(v) for k, v in store.items()}

    def close(self):
        for h in self.handles:
            h.remove()


def build_inputs(a, tokenizer_dir):
    """[(name, ids)]: one per --tokens source, or the --chat prompt."""
    if a.tokens:
        return [(Path(t).stem if Path(t).exists() else "tokens", inkref.read_ids(t)) for t in a.tokens]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    d = json.loads(Path(a.chat).read_text())
    messages = d["messages"] if isinstance(d, dict) else d
    tools = d.get("tools") if isinstance(d, dict) else None
    ids = inkref.chat_ids(tok, messages, effort=a.effort, tools=tools)
    if isinstance(d, dict) and d.get("continuation"):
        ids += [int(i) for i in d["continuation"]]
    return [(Path(a.chat).stem, ids)]


def run(a):
    threads = inkref.set_threads(a.threads)
    inputs = build_inputs(a, a.tokenizer or a.checkpoint)
    model, info = inkref.load_model(a.checkpoint, a.impl, a.device)
    for name, ids in inputs:
        out = Path(a.out) / name if len(inputs) > 1 else Path(a.out)
        dump_one(a, model, info, threads, ids[: a.max_tokens] if a.max_tokens else ids, out)


def dump_one(a, model, info, threads, ids, out):
    T = len(ids)
    cfg = model.config
    rows = parse_rows(a.rows, T)
    logit_rows = parse_rows(a.logit_rows, T)
    parts = PARTS if a.parts == "all" else tuple(p for p in a.parts.split(",") if p)
    rec = Recorder(model, rows, parts)
    calls0 = sum(getattr(layer.mlp.experts, "native_calls", 0) for layer in model.model.layers)
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    mode = inkref.Promote64() if a.impl == "fp64" else None
    chunk = a.chunk or T
    keep = torch.as_tensor(logit_rows)
    kept, token_logprobs, argmax = [], [], []
    t0 = time.time()
    with torch.no_grad():
        if mode:
            mode.__enter__()
        cache = None
        if a.chunk:
            from transformers import DynamicCache

            cache = DynamicCache(config=cfg)
        for lo in range(0, T, chunk):
            hi = min(lo + chunk, T)
            rec.call(lo, hi)
            lg = model(x[:, lo:hi], cache=cache, use_cache=cache is not None).logits[0]
            assert rec.visit + 1 == len(inkref.visit_table(cfg)), (rec.visit, "visits")
            sel = keep[(keep >= lo) & (keep < hi)]
            if len(sel):
                kept.append(lg[(sel - lo).to(lg.device)].float().cpu().numpy())
            lp = torch.log_softmax(lg.double(), -1)
            nxt = torch.tensor(ids[lo + 1 : hi + 1], device=lg.device)
            token_logprobs.append(lp[: len(nxt)].gather(-1, nxt[:, None])[:, 0].float().cpu().numpy())
            argmax.append(lg.argmax(-1).cpu().numpy())
            del lg, lp
        if mode:
            mode.__exit__(None, None, None)
    fwd_s = time.time() - t0
    rec.close()
    native_calls = sum(getattr(layer.mlp.experts, "native_calls", 0) for layer in model.model.layers) - calls0
    token_logprobs = np.concatenate(token_logprobs)
    argmax = np.concatenate(argmax)

    meta = {
        "format": tfdump.FORMAT,
        "producer": "tools/ref_ink/ref_dump.py",
        "model": "Inkling2LoopForCausalLM",
        "checkpoint": str(a.checkpoint),
        "impl": a.impl,
        "device": str(dev),
        "threads": threads,
        "T": T,
        "forward_s": round(fwd_s, 2),
        "load": info,
        "env": inkref.env_info(),
        "visits": [list(v) for v in inkref.visit_table(cfg)],
        "chunk": a.chunk,
        "native_expert_calls": native_calls,
        "rows_spec": a.rows,
        "logit_rows_spec": a.logit_rows,
        "parts": list(parts),
        "argmax": [int(i) for i in argmax],
    }
    d = tfdump.Dump(ids, meta)
    d.token_logprobs = token_logprobs
    if len(logit_rows):
        d.put("logits", np.concatenate(kept), logit_rows)
    for k, v in rec.arrays(rec.data).items():
        d.put("hidden." + k, v, rows)
    d.save(out)
    if rec.route:
        (out / "route").mkdir(exist_ok=True)
        np.save(out / "route" / "rows.npy", rows)
        for k, v in rec.arrays(rec.route).items():
            np.save(out / "route" / f"{k}.npy", v)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(
        f"{a.impl} T={T} load {info['load_s']} s forward {fwd_s:.1f} s threads {threads} -> {out} "
        f"({size / 2**20:.1f} MiB, {len(rec.data)} hidden, {len(rec.route)} route arrays, native expert calls "
        f"{native_calls})"
    )
    print(
        "load info:",
        json.dumps(
            {
                k: info[k]
                for k in ("expert_dtype", "experts_implementation", "dtypes", "missing_keys", "unexpected_keys_n")
            }
        ),
    )
    nll = -float(token_logprobs.mean())
    print(f"mean NLL {nll:.4f} (ppl {np.exp(nll):.3f}); argmax tail {[int(i) for i in argmax[-8:]]}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--impl", choices=["bf16", "bf16-hf", "native", "native-fp32", "fp32", "fp64"], default="bf16")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--tokens",
        nargs="+",
        help="ids: .npy, .json ({'ids': [...]} or a greedy.py record = prompt + "
        "generated), text file of integers, or inline list; several sources -> OUT/<stem>/ each, one load",
    )
    src.add_argument("--chat", help="JSON {messages, tools?, continuation?}: chat template with generation prompt")
    p.add_argument("--effort", type=float, default=0.0, help="reasoning effort for --chat (default 0)")
    p.add_argument("--tokenizer", help="tokenizer dir for --chat (default: the checkpoint)")
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--rows", default="all", help="positions of the per-visit tensors")
    p.add_argument("--logit-rows", default="all", help="positions of the logits")
    p.add_argument(
        "--parts", default="all", help=f"per-visit parts besides the visit output: all or a subset of {','.join(PARTS)}"
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="run the sequence through the KV cache in chunks of N (0 = one "
        "forward without cache); visits and logits are collected per chunk",
    )
    p.add_argument("--device", default="cpu", help="cpu, cuda:N or auto (device_map=auto)")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
