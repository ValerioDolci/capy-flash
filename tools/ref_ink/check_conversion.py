"""Compare an Inkling 2 artifact with the reference's own load-time transforms (CPU).

Run with the reference Python (transformers of the Inkling 2 PR) from the repository root:

    python -B tools/ref_ink/check_conversion.py --checkpoint DIR --artifact X.ninfer [--experts 0,63,64,127]

Experts: for every layer and both matrices, the sampled experts decoded from the artifact (codes,
E4M3 scales and per-expert divisor of an NVFP4 bank, E4M3FN codes and UE8M0 exponents of an MXFP8
bank — identical to the reference's BF16 dequantization —, or BF16 words) against the reference
``Inkling2LoopDequantizeExperts`` + ``Interleave(dim=1)`` output in BF16. MXFP8 layers must be
bit-identical. NVFP4 layers store ``e2m1 * e4m3 / d`` with ``d = fl32(den / amax)`` and are not
rounded to BF16; the report gives the largest relative distance to the reference's
``bf16((e2m1 * e4m3) * fl32(amax / den))``, the largest relative distance before the reference's
BF16 rounding (``(e2m1 * e4m3) * fl32(amax / den)`` in FP32), and how many BF16 words differ from
``bf16`` of the artifact value.

Dense parameters: the output head (first unpadded rows of ``unembed``), the cross-pass message
queries (``Inkling2LoopMergeMessageQueries``), the shifted attention-input queries and a sample of
projections, norms and convolutions must equal the checkpoint words.

Drafter (when the artifact has the ``mtp`` component): every tensor of ``mtp.safetensors`` must
equal its words, and the folded head must equal ``bf16(f64(unembed[:Vu]) @ f64(hidden_out))``
(weight and bias) recomputed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.artifact.codecs.direct import decode_direct  # noqa: E402
from tools.artifact.codecs.nvfp4 import unswizzle_nvfp4_scales  # noqa: E402
from tools.artifact.layouts import block_scale_geometry  # noqa: E402
from tools.artifact.reader import Artifact  # noqa: E402
from transformers.core_model_loading import (  # noqa: E402
    Inkling2LoopDequantizeExperts,
    Inkling2LoopMergeMessageQueries,
    Interleave,
)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


class Checkpoint:
    def __init__(self, directory: Path):
        index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {name: directory / file for name, file in index.items()}
        self.handles = {}

    def has(self, name):
        return name in self.files

    def _handle(self, name):
        path = self.files[name]
        if path not in self.handles:
            self.handles[path] = safe_open(str(path), framework="pt")
        return self.handles[path]

    def get(self, name, rows=None):
        handle = self._handle(name)
        if rows is None:
            return handle.get_tensor(name)
        return handle.get_slice(name)[rows]


class Engine:
    def __init__(self, path: Path):
        self.artifact = Artifact(path)
        self.bindings = self.artifact.directory.bindings
        self.cache = {}

    def part(self, name):
        binding = self.bindings[name]
        if "object" in binding:
            obj = self.artifact.object(binding["object"])
            total = 1
            for dim in obj.shape:
                total *= dim
            return obj, 0, total
        (part,) = binding["parts"]
        return self.artifact.object(part["object"]), part["range"][0], part["range"][1]

    def raw(self, obj):
        if obj.id not in self.cache:
            if len(self.cache) == 4:
                self.cache.pop(next(iter(self.cache)))
            self.cache[obj.id] = torch.frombuffer(bytearray(self.artifact.read_object(obj.id)), dtype=torch.uint8)
        return self.cache[obj.id]

    def values(self, name, shape):
        """Exact represented values of a logical parameter (FP32), and its stored format."""
        obj, begin, end = self.part(name)
        raw = self.raw(obj)
        if obj.format in ("bf16", "fp32"):
            flat = decode_direct(raw, obj.format, obj.shape).reshape(-1)[begin:end]
            return flat.float().reshape(shape), obj.format
        if obj.format == "mxfp8_e4m3fn_g32_ue8m0":
            # block_scale_k32_v1: row-major E4M3FN codes, then (256-aligned) row-major UE8M0.
            n, k = obj.shape
            first, last = begin // k, end // k
            offset = (n * k + 255) // 256 * 256
            codes = raw[: n * k].reshape(n, k)[first:last]
            exponents = raw[offset : offset + n * (k // 32)].reshape(n, k // 32)[first:last]
            scale = torch.ldexp(torch.ones(1), exponents.to(torch.int32) - 127).repeat_interleave(32, dim=1)
            return (codes.view(torch.float8_e4m3fn).float() * scale).reshape(shape), "mxfp8"
        if obj.format != "nvfp4":
            raise ValueError(f"{name}: unexpected format {obj.format}")
        n, k = obj.shape
        geometry = block_scale_geometry("nvfp4", obj.shape, obj.divisors)
        first, last = begin // k, end // k
        codes = raw[: geometry.code_plane_bytes].reshape(n, k // 2)[first:last]
        plane = raw[geometry.scale_plane_offset : geometry.scale_plane_offset + geometry.scale_plane_bytes]
        scales = unswizzle_nvfp4_scales(plane, obj.shape)[first:last]
        words = raw[geometry.weight_divisor_offset : geometry.weight_divisor_offset + 4 * obj.divisors]
        divisors = torch.frombuffer(bytearray(words.numpy().tobytes()), dtype=torch.float32)
        per = n // obj.divisors
        if first % per or last - first != per:
            raise ValueError(f"{name}: an expert must be one divisor block of its bank")
        nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(last - first, k)
        block = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        return (E2M1[nibbles.long()] * block / divisors[first // per]).reshape(shape), "nvfp4"


def reference_experts(ckpt, prefix, experts, gate_up):
    names = [
        prefix + suffix for suffix in ("_weight", "_weight_scale", "_weight_amax", "_weight_outer_scale_denominator")
    ]
    tensors = {}
    for name in names:
        if ckpt.has(name):
            if name.endswith("denominator"):
                tensors[name] = ckpt.get(name)
            else:
                tensors[name] = torch.cat([ckpt.get(name, slice(e, e + 1)) for e in experts])
    out = Inkling2LoopDequantizeExperts().convert(tensors, names, ["w"])["w"]
    if gate_up:
        out = Interleave(dim=1).convert({"w": out}, ["w"], ["w"])["w"]
    fp32 = None
    if len(tensors) == 4:
        # The reference's product before its BF16 rounding.
        codes = tensors[names[0]].long()
        values = torch.stack((E2M1[codes & 15], E2M1[codes >> 4]), dim=-1).reshape(codes.shape[0], codes.shape[1], -1)
        block = tensors[names[1]].float().repeat_interleave(16, dim=2)
        factor = tensors[names[2]].float() / tensors[names[3]].float()
        fp32 = (values * block) * factor[:, None, None]
        if gate_up:
            fp32 = Interleave(dim=1).convert({"w": fp32}, ["w"], ["w"])["w"]
    return out, fp32


def check_experts(ckpt, engine, config, sample, report):
    h, f = config["hidden_size"], config["intermediate_size"]
    worst = {"nvfp4_rel_bf16": 0.0, "nvfp4_rel_fp32": 0.0}
    for layer in range(config["num_hidden_layers"]):
        for matrix, role, shape in (("w13", "gate_up", (2 * f, h)), ("w2", "down", (h, f))):
            prefix = f"model.llm.layers.{layer}.mlp.experts.{matrix}"
            reference, fp32 = reference_experts(ckpt, prefix, sample, role == "gate_up")
            row = {"layer": layer, "matrix": role, "experts": sample}
            for index, expert in enumerate(sample):
                values, stored = engine.values(f"text/layers/{layer}/moe/experts/{expert}/{role}", shape)
                ref = reference[index]
                row["format"] = stored
                if stored in ("bf16", "mxfp8"):
                    same = torch.equal(values.to(torch.bfloat16).view(torch.int16), ref.view(torch.int16))
                    exact = torch.equal(values.to(torch.bfloat16).float(), values)
                    row["identical"] = row.get("identical", True) and same and exact
                else:
                    nonzero = ref.float() != 0
                    rel = ((values - ref.float()).abs() / ref.float().abs())[nonzero]
                    rel32 = ((values - fp32[index]).abs() / fp32[index].abs())[fp32[index] != 0]
                    differ = int((values.to(torch.bfloat16).view(torch.int16) != ref.view(torch.int16)).sum())
                    zeros = bool(torch.equal(values == 0, ref.float() == 0))
                    row["max_rel_vs_bf16"] = max(row.get("max_rel_vs_bf16", 0.0), float(rel.max()))
                    row["max_rel_vs_fp32"] = max(row.get("max_rel_vs_fp32", 0.0), float(rel32.max()))
                    row["bf16_words_differ"] = row.get("bf16_words_differ", 0) + differ
                    row["zeros_agree"] = row.get("zeros_agree", True) and zeros
                    worst["nvfp4_rel_bf16"] = max(worst["nvfp4_rel_bf16"], row["max_rel_vs_bf16"])
                    worst["nvfp4_rel_fp32"] = max(worst["nvfp4_rel_fp32"], row["max_rel_vs_fp32"])
                    # The stored divisor is fl32(den / amax).
                    amax = float(ckpt.get(prefix + "_weight_amax")[expert])
                    den = float(ckpt.get(prefix + "_weight_outer_scale_denominator")[0])
                    obj, begin, _ = engine.part(f"text/layers/{layer}/moe/experts/{expert}/{role}")
                    geometry = block_scale_geometry("nvfp4", obj.shape, obj.divisors)
                    word = engine.raw(obj)[geometry.weight_divisor_offset :].numpy().tobytes()
                    slot = begin // obj.shape[1] // (obj.shape[0] // obj.divisors)
                    stored_divisor = struct.unpack_from("<f", word, 4 * slot)[0]
                    expected = struct.unpack("<f", struct.pack("<f", den / amax))[0]
                    row["divisors_ok"] = row.get("divisors_ok", True) and stored_divisor == expected
            report["experts"].append(row)
            print(json.dumps(row), flush=True)
    report["worst"] = worst


def check_dense(ckpt, engine, config, report):
    h, unpadded = config["hidden_size"], config["unpadded_vocab_size"]
    core_start, core_end = config["loop_config"]["core_start"], config["loop_config"]["core_end"]
    checks = []

    def same(name, expected):
        values, stored = engine.values(name, tuple(expected.shape))
        ok = torch.equal(values.to(expected.dtype).view(torch.uint8), expected.contiguous().view(torch.uint8))
        ok = ok and torch.equal(values.to(expected.dtype).float(), values.float())
        checks.append({"parameter": name, "format": stored, "identical": bool(ok)})

    head = ckpt.get("model.llm.unembed.weight")
    same("text/output_head", head[:unpadded])
    merged = Inkling2LoopMergeMessageQueries().convert(
        {
            "q": ckpt.get("model.llm.middle_cycle_cross_pass_queries"),
            "kv": ckpt.get("model.llm.middle_cycle_cross_pass_kv_queries"),
        },
        ["q", "kv"],
        ["m"],
    )["m"]
    for layer in range(core_start, core_end):
        same(f"text/layers/{layer}/message_queries", merged[layer - core_start])
    for layer in range(1, config["num_hidden_layers"]):
        query = ckpt.get(f"model.llm.attn_res.attn_queries.{layer - 1}")
        same(f"text/layers/{layer}/attention_queries", query.reshape(-1, h))
    same("text/final_queries", ckpt.get("model.llm.attn_res.output_query").reshape(1, h))
    same("text/token_embedding", ckpt.get("model.llm.embed.weight"))
    for layer in (0, 5, 6, 11, 20, 23):
        p, sp = f"text/layers/{layer}/", f"model.llm.layers.{layer}."
        pairs = [
            ("attention/query", "attn.wq_du.weight"),
            ("attention/output", "attn.wo_ud.weight"),
            ("attention/relative_profile", "attn.rel_logits_proj.proj"),
            ("moe/router", "mlp.gate.weight"),
            ("moe/router_bias", "mlp.gate.bias"),
            ("ffn_norm", "mlp_norm.weight"),
        ]
        if (layer + 1) % 6 == 0:
            pairs += [("attention/key_conv_static", "attn.k_sconv.static_w"), ("key_input_norm", "k_input_norm.weight")]
        else:
            pairs += [("input_norm", "attn_norm.weight")]
        for name, source in pairs:
            same(p + name, ckpt.get(sp + source))
        conv = ckpt.get(sp + "attn_sconv.weight")
        same(p + "post_attention_conv", conv.reshape(conv.shape[0], -1))
        if (layer + 1) % 6:
            conv = ckpt.get(sp + "attn.v_sconv.weight")
            same(p + "attention/value_conv", conv.reshape(conv.shape[0], -1))
        if core_start <= layer < core_end:
            same(p + "message_gate/query/down", ckpt.get(sp + "controllers.attn.down_weight"))
            same(p + "message_gate/ffn/gain", ckpt.get(sp + "controllers.mlp.gain_weight"))
    report["dense"] = checks
    bad = [c for c in checks if not c["identical"]]
    print(f"dense: {len(checks) - len(bad)}/{len(checks)} identical", flush=True)
    for c in bad:
        print("  differs:", c, flush=True)


def check_mtp(directory, engine, config, report):
    mtp = json.loads((directory / "config.json").read_text())["mtp_config"]
    h, unpadded = config["hidden_size"], config["unpadded_vocab_size"]
    store = safe_open(str(directory / "mtp.safetensors"), framework="pt")
    checks = []

    def same(name, expected):
        values, stored = engine.values(name, tuple(expected.shape))
        ok = torch.equal(values.to(expected.dtype).view(torch.uint8), expected.contiguous().view(torch.uint8))
        ok = ok and torch.equal(values.to(expected.dtype).float(), values.float())
        differ = int((values.to(expected.dtype).view(torch.int16) != expected.view(torch.int16)).sum())
        checks.append({"parameter": name, "format": stored, "identical": bool(ok), "words_differ": differ})

    pairs = [("fuse_weight", "fuse_weight"), ("mask_embedding", "mask_embed"), ("final_norm", "norm.weight")]
    for layer in range(mtp["num_hidden_layers"]):
        p, sp = f"layers/{layer}/", f"layers.{layer}."
        pairs += [
            (p + "context_norm", sp + "dflare_norm.weight"),
            (p + "input_norm", sp + "input_layernorm.weight"),
            (p + "post_attention_norm", sp + "post_attention_layernorm.weight"),
            (p + "mlp/gate", sp + "mlp.gate_proj.weight"),
            (p + "mlp/up", sp + "mlp.up_proj.weight"),
            (p + "mlp/down", sp + "mlp.down_proj.weight"),
        ]
        for role, field in (
            ("query", "q_proj"),
            ("key", "k_proj"),
            ("value", "v_proj"),
            ("output", "o_proj"),
            ("context_key", "k_proj_ctx"),
            ("context_value", "v_proj_ctx"),
            ("query_norm", "q_norm"),
            ("key_norm", "k_norm"),
            ("context_key_norm", "k_norm_ctx"),
        ):
            pairs.append((p + "attention/" + role, sp + "self_attn." + field + ".weight"))
    pairs += [
        ("markov/context_norm", "markov_head.dflare_norm.weight"),
        ("markov/token_embedding", "markov_head.markov_w1.weight"),
        ("markov/recurrence", "markov_head.rnn_proj.weight"),
        ("markov/recurrence_bias", "markov_head.rnn_proj.bias"),
    ]
    for name, source in pairs:
        same("mtp/" + name, store.get_tensor(source))
    head = Checkpoint(directory).get("model.llm.unembed.weight")[:unpadded].double()
    weight = store.get_tensor("markov_head.hidden_out.weight").double()
    bias = store.get_tensor("markov_head.hidden_out.bias").double()
    same("mtp/markov/folded_output", (head @ weight).to(torch.bfloat16))
    same("mtp/markov/folded_bias", (head @ bias).to(torch.bfloat16))
    assert weight.shape[0] == h
    report["mtp"] = checks
    bad = [c for c in checks if not c["identical"]]
    print(f"mtp: {len(checks) - len(bad)}/{len(checks)} identical", flush=True)
    for c in bad:
        print("  differs:", c, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--experts", default=None, help="comma-separated expert IDs (default: bank edges)")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    experts = config["n_routed_experts"]
    sample = (
        [int(v) for v in args.experts.split(",")]
        if args.experts
        else sorted({0, experts // 2 - 1, experts // 2, experts - 1, 37 % experts})
    )
    ckpt, engine = Checkpoint(args.checkpoint), Engine(args.artifact)
    report = {"artifact": str(args.artifact), "checkpoint": str(args.checkpoint), "experts": []}
    torch.set_num_threads(8)
    check_dense(ckpt, engine, config, report)
    if "mtp/fuse_weight" in engine.bindings:
        check_mtp(args.checkpoint, engine, config, report)
    check_experts(ckpt, engine, config, sample, report)
    failures = [r for r in report["experts"] if not r.get("identical", True) or not r.get("divisors_ok", True)]
    failures += [r for r in report["experts"] if not r.get("zeros_agree", True)]
    failures += [c for c in report["dense"] + report.get("mtp", []) if not c["identical"]]
    report["failures"] = len(failures)
    print(json.dumps({"worst": report["worst"], "failures": len(failures)}), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
