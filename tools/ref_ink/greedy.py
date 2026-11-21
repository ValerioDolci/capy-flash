# Copyright 2026 Valerio Dolci. Apache License 2.0.
"""Greedy references of Inkling 2 Mini with transformers (model.generate, do_sample=False, KV cache).

  greedy.py --checkpoint DIR --prompts prompts.json [--names it1,en1] [--max-new-tokens 128] --out DIR
            [--impl bf16] [--threads N] [--check-tf]

prompts.json: [{"name", "messages", "tools"?, "effort"?, "max_new_tokens"?, "long_text"?, "long_chars"?}]; "long_text"
names a text file (next to prompts.json) appended to the last user message (cut at the last newline before
"long_chars" characters if given), for the prompt above 8192 tokens.
--prefill-chunk N: the prompt (but its last token) goes through the cache in chunks of N tokens before generate(); the
eager attention of a one-pass prefill holds several [heads, T, T] tensors (~8 GB each in FP32 at T = 9000).
Per prompt, DIR/<name>.json: prompt ids (chat template, add_generation_prompt, fixed thinking effort), generated ids,
decoded text (special tokens kept), per step the top-1 id, its logit and the top-1/top-2 margin (near-ties tell where
an engine may legitimately diverge), prefill and decode times, peak memory.
--check-tf re-scores prompt + generated tokens in one forward without cache and reports where its argmax differs
from the generated token (cache path against no-cache path).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkref  # noqa: E402


def rss_gib(field):
    for line in open("/proc/self/status"):
        if line.startswith(field):
            return round(int(line.split()[1]) / 2**20, 2)
    return None


class Clock:
    """LogitsProcessor that stamps every decoding step (the first stamp closes the prefill)."""

    def __init__(self):
        self.t = []

    def __call__(self, input_ids, scores):
        self.t.append(time.time())
        return scores


def prompt_ids(tok, p, base):
    messages = [dict(m) for m in p["messages"]]
    if p.get("long_text"):
        text = (base / p["long_text"]).read_text(encoding="utf-8")
        if p.get("long_chars"):
            text = text[: text.rfind("\n", 0, p["long_chars"]) + 1]
        messages[-1]["content"] += text
    return inkref.chat_ids(tok, messages, effort=p.get("effort", 0.0), tools=p.get("tools"))


def run_one(model, tok, p, base, a):
    from transformers import LogitsProcessorList

    ids = prompt_ids(tok, p, base)
    n_new = p.get("max_new_tokens", a.max_new_tokens)
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    clock = Clock()
    extra = {}
    t0 = time.time()
    with torch.no_grad():
        if a.prefill_chunk and len(ids) > a.prefill_chunk:
            # all prompt tokens but the last go through the cache in chunks; generate() runs the last one
            from transformers import DynamicCache

            cache = DynamicCache(config=model.config)
            for s in range(0, len(ids) - 1, a.prefill_chunk):
                model(x[:, s : min(s + a.prefill_chunk, len(ids) - 1)], cache=cache, use_cache=True)
            extra["cache"] = cache
        out = model.generate(
            x,
            attention_mask=torch.ones_like(x),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=n_new,
            output_logits=True,
            return_dict_in_generate=True,
            logits_processor=LogitsProcessorList([clock]),
            **extra,
        )
    t1 = time.time()
    gen = [int(i) for i in out.sequences[0, len(ids) :]]
    steps = []
    for lg in out.logits:
        v, i = lg[0].float().topk(2)
        steps.append({"top1": int(i[0]), "logit": round(float(v[0]), 4), "margin": round(float(v[0] - v[1]), 4)})
    prefill = clock.t[0] - t0 if clock.t else None
    decode = (t1 - clock.t[0]) / max(len(gen) - 1, 1) if clock.t else None
    rec = {
        "name": p["name"],
        "prompt_ids": ids,
        "generated_ids": gen,
        "text": tok.decode(gen, skip_special_tokens=False),
        "n_prompt": len(ids),
        "n_generated": len(gen),
        "stopped_on_eos": bool(
            gen
            and gen[-1]
            in (
                model.generation_config.eos_token_id
                if isinstance(model.generation_config.eos_token_id, list)
                else [model.generation_config.eos_token_id]
            )
        ),
        "max_new_tokens": n_new,
        "effort": p.get("effort", 0.0),
        "prefill_chunk": a.prefill_chunk if "cache" in extra else 0,
        "steps": steps,
        "min_margin": min((s["margin"] for s in steps), default=None),
        "prefill_s": round(prefill, 2) if prefill else None,
        "decode_s_per_token": round(decode, 3) if decode else None,
        "rss_peak_gib": rss_gib("VmHWM"),
    }
    if a.check_tf:
        seq = torch.tensor([ids + gen], device=dev)
        with torch.no_grad():
            lg = model(seq, use_cache=False).logits[0]
        am = lg[len(ids) - 1 : len(ids) + len(gen) - 1].argmax(-1).tolist()
        diff = [k for k, (g, t) in enumerate(zip(gen, am)) if g != t]
        rec["tf_check"] = {
            "mismatch_steps": diff[:20],
            "n_mismatch": len(diff),
            "first": (
                {"step": diff[0], "cached": gen[diff[0]], "nocache": am[diff[0]], "margin": steps[diff[0]]["margin"]}
                if diff
                else None
            ),
        }
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--names", default="", help="comma list of prompt names (default: all)")
    p.add_argument("--impl", choices=["bf16", "bf16-hf", "native", "fp32"], default="bf16")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--check-tf", action="store_true")
    p.add_argument(
        "--prefill-chunk",
        type=int,
        default=0,
        help="prefill prompts longer than this through the KV cache in chunks of this size (0 = one pass)",
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    threads = inkref.set_threads(a.threads)
    from transformers import AutoTokenizer

    base = Path(a.prompts).resolve().parent
    prompts = json.loads(Path(a.prompts).read_text())
    if a.names:
        want = a.names.split(",")
        prompts = [q for q in prompts if q["name"] in want]
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model, info = inkref.load_model(a.checkpoint, a.impl, a.device)
    print(
        "load:",
        json.dumps(info),
        "rss",
        rss_gib("VmRSS"),
        "anon",
        rss_gib("RssAnon"),
        "peak",
        rss_gib("VmHWM"),
        flush=True,
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for q in prompts:
        rec = run_one(model, tok, q, base, a)
        rec.update(
            {"impl": a.impl, "threads": threads, "load": info, "env": inkref.env_info(), "checkpoint": a.checkpoint}
        )
        (out / f"{q['name']}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print(
            f"{q['name']}: prompt {rec['n_prompt']} gen {rec['n_generated']} eos {rec['stopped_on_eos']} "
            f"prefill {rec['prefill_s']} s decode {rec['decode_s_per_token']} s/tok min margin {rec['min_margin']} "
            f"tf {rec.get('tf_check', {}).get('n_mismatch')}",
            flush=True,
        )
        print("  " + repr(rec["text"][:400]), flush=True)


if __name__ == "__main__":
    main()
