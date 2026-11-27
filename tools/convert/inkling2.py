"""Inkling 2 Mini architecture adapter: Text config, logical parameters, expert banks, resources.

The mathematics and the checkpoint layout are described in docs/maintainer/inkling2-model.md;
the logical names below are the artifact contract read by src/models/inkling2/weights.h.

Checkpoint conventions mapped here (``model.llm.*``):

- ``attn_res.attn_queries.N`` belongs to layer N + 1 (layer 0 has none), ``attn_res.mlp_queries.N``
  to layer N, ``attn_res.output_query`` to the head; each becomes a ``[queries, H]`` FP32 matrix.
- ``middle_cycle_cross_pass_queries`` (rows 2i = Q, 2i + 1 = FFN of core layer i) and
  ``middle_cycle_cross_pass_kv_queries`` (rows 2i = K, 2i + 1 = V) become one ``[4, H]`` matrix per
  core layer in the order Q, K, V, FFN.
- ``unembed`` keeps its first ``unpadded_vocab_size`` rows; the embedding keeps every row.
- Routed experts: ``w13`` rows interleave gate (even rows) and up (odd rows); the logical
  ``gate_up`` of an expert is ``[2F, H]`` with the F gate rows first. NVFP4 experts (``amax`` and
  ``outer_scale_denominator`` present) expose their codes and E4M3 block scales unchanged with the
  weight divisor ``denominator / amax`` (one FP32 rounding); MXFP8 experts (E4M3 codes, one UE8M0
  exponent per 32 columns) expose their exact values.
- Every other tensor keeps its stored words (BF16 or FP32); convolution weights ``[C, 1, K]``
  become ``[C, K]``. The parameters the reference holds in FP32 and whose Ops read FP32 (router,
  every short-convolution weight, message gate forget/gain) are widened exactly to FP32.

The optional ``mtp`` component is the DSpark drafter of ``mtp.safetensors`` (its config is the
checkpoint's ``mtp_config``). Its tensors keep their BF16 words under ``mtp/`` names; the drafter
reads the Text component's embedding, embedding norm and output head. The recurrent head's
``hidden_out`` is stored folded into the output head rows as SGLang and vLLM fold it:
``folded_output = bf16(f64(unembed[:Vu]) @ f64(hidden_out.weight))`` ``[Vu, r]`` and
``folded_bias = bf16(f64(unembed[:Vu]) @ f64(hidden_out.bias))`` ``[Vu]``.

The optional ``dflash2`` component is a DFlash2 block drafter read from its own checkpoint (the
``dflash2`` source, e.g. ``incoai/Inkling-2-Mini-DFlash2``): five sliding-window layers with grouped
dynamic convolutions and a candidate selector over the output head's top candidates. It reads the
DSpark drafter's four target states, the Text component's embedding, embedding norm and output
head. Its tensors keep their words under ``dflash2/`` names; each layer's key and value rows also
project the drafter's context (the uses ``attention/input`` and ``dflash2/context``).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
import struct
from typing import Mapping

import torch

from tools.artifact.formats import valid_positive_fp32_word
from .model import Model, Parameter
from .resources import load_resources
from .sources.logical import EncodedRows, LogicalSource, select_rows
from .sources.modelopt import MXFP8, decode_mxfp8, decode_nvfp4
from .sources.safetensors import SafetensorsSource, tensor_source

ARCHITECTURES = ("Inkling2LoopForConditionalGeneration", "Inkling2LoopForCausalLM")
TEXT_ARCHITECTURE = "Inkling2LoopForCausalLM"
TEXT_MODEL_TYPE = "inkling2_loop_text"
MTP_ARCHITECTURE = "Inkling2DSpark"
MTP_MODEL_TYPE = "inkling2_dspark"
MTP_FILE = "mtp.safetensors"
DFLASH2_ARCHITECTURE = "DFlash2DraftModel"
DFLASH2_MODEL_TYPE = "inkling2_dflash2"
# The reference derives the layer schedule from the index alone: every sixth layer is global.
GLOBAL_LAYER_PERIOD = 6
MESSAGE_ROLES = ("query", "key", "value", "ffn")
_CONTROLLERS = ("attn", "k", "v", "mlp")  # checkpoint names of MESSAGE_ROLES


def is_inkling2(config: dict) -> bool:
    architectures = config.get("architectures")
    return isinstance(architectures, list) and bool(set(architectures) & set(ARCHITECTURES))


def text_config(source: dict) -> dict:
    """The checkpoint's text_config with the Text architecture; the engine checks every member."""
    architectures = source.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1 or not is_inkling2(source):
        raise ValueError(f"unsupported Inkling 2 architecture {architectures!r}")
    raw = source.get("text_config", source)
    if raw.get("model_type") != TEXT_MODEL_TYPE:
        raise ValueError(f"text.model_type: expected {TEXT_MODEL_TYPE!r}")
    config = {"architectures": [TEXT_ARCHITECTURE]}
    config.update({key: value for key, value in raw.items() if key != "architectures"})
    return config


def mtp_config(source: dict) -> dict:
    """The checkpoint's mtp_config with the drafter's architecture; the engine checks every member."""
    raw = source.get("mtp_config")
    if not isinstance(raw, dict):
        raise ValueError("mtp: the checkpoint has no mtp_config")
    config = {"architectures": [MTP_ARCHITECTURE], "model_type": MTP_MODEL_TYPE}
    config.update({key: value for key, value in raw.items() if key not in config})
    return config


def dflash2_config(raw: dict, g: dict) -> dict:
    """The DFlash2 drafter's config.json with the drafter's architecture; the engine checks every member."""
    if raw.get("architectures") != [DFLASH2_ARCHITECTURE]:
        raise ValueError(f"dflash2: expected architectures [{DFLASH2_ARCHITECTURE!r}]")
    for key, expected in (
        ("hidden_size", g["hidden_size"]),
        ("vocab_size", g["vocab_size"]),
        ("hidden_act", "silu"),
        ("attention_bias", False),
        ("is_causal", False),
    ):
        if raw.get(key) != expected:
            raise ValueError(f"dflash2.{key}: expected {expected!r}")
    layers = raw.get("num_hidden_layers")
    if raw.get("layer_types") != ["sliding_attention"] * layers:
        raise ValueError("dflash2: every layer must be sliding_attention")
    rope = raw.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError("dflash2: only the default RoPE is supported")
    config = {"architectures": [DFLASH2_ARCHITECTURE], "model_type": DFLASH2_MODEL_TYPE}
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_hidden_layers",
        "sliding_window",
        "rms_norm_eps",
        "vocab_size",
    ):
        config[key] = raw.get(key)
    config["rope_theta"] = rope.get("rope_theta", raw.get("rope_theta"))
    config["dflash_config"] = dict(raw.get("dflash_config") or {})
    return config


def global_layer(layer: int) -> bool:
    return (layer + 1) % GLOBAL_LAYER_PERIOD == 0


def _integer(config: dict, key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"text.{key}: expected a positive integer")
    return value


def _geometry(config: dict) -> dict:
    loop = config.get("loop_config") or {}
    g = {
        key: _integer(config, key)
        for key in (
            "hidden_size",
            "num_hidden_layers",
            "vocab_size",
            "unpadded_vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "d_rel",
            "rel_extent",
            "sliding_window_size",
            "sconv_kernel_size",
            "n_routed_experts",
            "intermediate_size",
        )
    }
    g.update({key: _integer(loop, key) for key in ("core_start", "core_end", "gate_rank")})
    if g["unpadded_vocab_size"] > g["vocab_size"] or g["core_end"] > g["num_hidden_layers"]:
        raise ValueError("text: unpadded vocabulary or looped core outside the model")
    if g["n_routed_experts"] % 2:
        raise ValueError("text: the experts are stored as two equal banks")
    return g


def _signature(store: SafetensorsSource, name: str, shape: tuple[int, ...], dtype: str) -> None:
    info = store.describe(name)
    if tuple(info.shape) != tuple(shape) or info.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}{shape}, got {info.dtype}{tuple(info.shape)}")


def _expert_rows(rows: int, interleaved: bool) -> torch.Tensor:
    """Checkpoint row of each logical row: gate rows (even) first, then up rows (odd)."""
    if not interleaved:
        return torch.arange(rows)
    half = rows // 2
    return torch.cat((torch.arange(half) * 2, torch.arange(half) * 2 + 1))


def _read_rows(store, name, expert, rows, width, file_rows):
    """Rows ``file_rows`` of expert ``expert`` of a [E, rows, width] tensor, in that order."""
    low, high = int(file_rows.min()), int(file_rows.max()) + 1
    base = expert * rows
    block = store.read_flat(name, (base + low) * width, (base + high) * width).reshape(high - low, -1)
    return block[file_rows - low]


def nvfp4_expert_source(store, prefix, expert, shape, *, interleaved):
    """One expert of a stacked ``[E, N, K/2]`` NVFP4 bank with its ``amax / denominator`` factor."""
    n, k = shape
    weight, scale = prefix + "_weight", prefix + "_weight_scale"
    amax, denominator = prefix + "_weight_amax", prefix + "_weight_outer_scale_denominator"
    order = _expert_rows(n, interleaved)

    @cache
    def checked() -> int:
        experts = store.describe(weight).shape[0]
        _signature(store, weight, (experts, n, k // 2), "U8")
        _signature(store, scale, (experts, n, k // 16), "F8_E4M3")
        _signature(store, amax, (experts,), "F32")
        _signature(store, denominator, (1,), "F32")
        return experts

    @cache
    def divisor() -> bytes:
        checked()
        value_amax = float(store.read_flat(amax, expert, expert + 1)[0])
        value_denominator = float(store.read_flat(denominator)[0])
        if not value_amax > 0 or not value_denominator > 0:
            raise ValueError(f"{prefix}[{expert}]: amax and denominator must be positive")
        word = struct.pack("<f", value_denominator / value_amax)
        if not valid_positive_fp32_word(struct.unpack("<I", word)[0]):
            raise ValueError(f"{prefix}[{expert}]: weight divisor is not a positive finite FP32")
        return word

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}[{expert}]: invalid encoded rows [{begin},{end})")
        checked()
        rows = order[begin:end]
        codes = _read_rows(store, weight, expert, n, k // 2, rows).contiguous()
        scales = _read_rows(store, scale, expert, n, k // 16, rows).view(torch.uint8).contiguous()
        if bool((scales > 0x7E).any()):
            raise ValueError(f"{scale}[{expert}]: expected nonnegative finite E4M3FN")
        return EncodedRows("nvfp4", codes, scales, divisor())

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        values = decode_nvfp4(words.codes, words.scales, words.weight_divisor)
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        tuple(shape), f"{store.path}:{prefix}[{expert}] (nvfp4 amax/denominator)", read, encoded, divisor
    )


def mxfp8_expert_source(store, prefix, expert, shape, *, interleaved):
    """One expert of a stacked ``[E, N, K]`` MXFP8 bank (E4M3FN codes, UE8M0 per 32 columns)."""
    n, k = shape
    weight, scale = prefix + "_weight", prefix + "_weight_scale"
    order = _expert_rows(n, interleaved)

    @cache
    def checked() -> None:
        experts = store.describe(weight).shape[0]
        _signature(store, weight, (experts, n, k), "F8_E4M3")
        _signature(store, scale, (experts, n, k // 32), "U8")

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}[{expert}]: invalid encoded rows [{begin},{end})")
        checked()
        rows = order[begin:end]
        codes = _read_rows(store, weight, expert, n, k, rows).view(torch.uint8).contiguous()
        exponents = _read_rows(store, scale, expert, n, k // 32, rows).contiguous()
        if bool(((codes & 0x7F) == 0x7F).any()) or bool((exponents == 0xFF).any()):
            raise ValueError(f"{prefix}[{expert}]: NaN code or non-finite UE8M0 scale")
        return EncodedRows(MXFP8, codes, exponents)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        values = decode_mxfp8(words.codes, words.scales)
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(tuple(shape), f"{store.path}:{prefix}[{expert}] (mxfp8)", read, encoded)


def folded_source(head: LogicalSource, store: SafetensorsSource, name: str, h: int, rank: int, bias: bool):
    """``bf16(f64(head rows) @ f64(hidden_out))``: the folded weight ``[rows, rank]`` or bias ``[rows]``."""
    rows = head.shape[0]

    @cache
    def projection() -> torch.Tensor:
        _signature(store, name, (h,) if bias else (h, rank), "BF16")
        return store.read_flat(name).double().reshape(h, -1)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.bfloat16)
        width = 1 if bias else rank
        first, last = begin // width, (end + width - 1) // width
        product = head.rows(first, last).double() @ projection()
        return product.to(torch.bfloat16).reshape(-1)[begin - first * width : end - first * width]

    shape = (rows,) if bias else (rows, rank)
    return LogicalSource(shape, f"{store.path}:{name} folded into {head.label}", read)


def message_queries_source(store, layer_index: int, h: int) -> LogicalSource:
    """Rows Q, K, V, FFN of core layer ``layer_index`` from the two cross-pass query tensors."""
    queries, kv = "model.llm.middle_cycle_cross_pass_queries", "model.llm.middle_cycle_cross_pass_kv_queries"

    def read(begin: int, end: int) -> torch.Tensor:
        rows = []
        for name, row in ((queries, 2 * layer_index), (kv, 2 * layer_index), (kv, 2 * layer_index + 1)):
            rows.append(store.read_flat(name, row * h, (row + 1) * h))
        rows.append(store.read_flat(queries, (2 * layer_index + 1) * h, (2 * layer_index + 2) * h))
        if any(store.describe(name).dtype != "F32" for name in (queries, kv)):
            raise ValueError("cross-pass queries must be FP32")
        return torch.cat(rows)[begin:end]

    return LogicalSource((4, h), f"{store.path}:cross-pass queries of core layer {layer_index}", read)


class _Builder:
    def __init__(self, model: Model, store: SafetensorsSource):
        self.model = model
        self.store = store

    def add(self, name, source, *, inputs=(), direct="bf16"):
        self.model.add(Parameter(name, tuple(source.shape), source, None, tuple(inputs), direct))

    def tensor(self, name, source_name, shape, *, inputs=(), source_shape=None, fp32=False):
        """A stored tensor with its words: FP32 stays FP32; ``fp32`` widens BF16 exactly to FP32."""
        dtype = self.store.describe(source_name).dtype
        if dtype not in ("BF16", "F32"):
            raise ValueError(f"{source_name}: expected a BF16 or FP32 tensor, got {dtype}")
        source = tensor_source(self.store, source_name, tuple(shape), source_shape=source_shape)
        self.add(name, source, inputs=inputs, direct="fp32" if fp32 or dtype == "F32" else "bf16")

    def queries(self, name, source_name, h, rows=1):
        """Learned FP32 queries of a block residual as a ``[rows, H]`` matrix."""
        self.tensor(name, source_name, (rows, h), source_shape=(h,) if rows == 1 else (rows, h))

    def attention(self, p, sp, g, layer):
        h, d, dr = g["hidden_size"], g["head_dim"], g["d_rel"]
        nh, nkv, taps = g["num_attention_heads"], g["num_key_value_heads"], g["sconv_kernel_size"]
        q, kv = nh * d, nkv * d
        a, ap = p + "attention/", sp + "attn."
        query_input = (p + "attention/query_input",)
        for role, field, rows in (
            ("query", "wq_du", q),
            ("gate", "gate_attn", q),
            ("relative", "wr_du", nh * dr),
            ("sink", "attn_sink_proj", nh),
        ):
            self.tensor(a + role, ap + field + ".weight", (rows, h), inputs=query_input)
        self.tensor(a + "key", ap + "wk_dv.weight", (kv, h), inputs=(p + "attention/key_input",))
        self.tensor(a + "value", ap + "wv_dv.weight", (kv, h), inputs=(p + "attention/value_input",))
        self.tensor(a + "output", ap + "wo_ud.weight", (h, q), inputs=(p + "attention/gated_output",))
        self.tensor(a + "query_scale", ap + "q_l2_clip.scale", (nh, d))
        self.tensor(a + "relative_scale", ap + "rel_l2_clip.scale", (nh, dr))
        extent = g["rel_extent"] if global_layer(layer) else g["sliding_window_size"]
        self.tensor(a + "relative_profile", ap + "rel_logits_proj.proj", (dr, extent))
        for role, field in (("key", "k_sconv"), ("value", "v_sconv")):
            if global_layer(layer):
                self.tensor(a + role + "_conv_static", ap + field + ".static_w", (taps, kv), fp32=True)
                self.tensor(
                    a + role + "_conv_projection",
                    ap + field + ".weight_proj",
                    (nkv * taps, h),
                    inputs=(p + f"attention/{role}_input",),
                )
            else:
                self.tensor(
                    a + role + "_conv", ap + field + ".weight", (kv, taps), source_shape=(kv, 1, taps), fp32=True
                )

    def experts(self, p, sp, g, layer):
        h, f, experts = g["hidden_size"], g["intermediate_size"], g["n_routed_experts"]
        names = {"gate_up": [], "down": []}
        for role, matrix, shape, interleaved in (
            ("gate_up", "w13", (2 * f, h), True),
            ("down", "w2", (h, f), False),
        ):
            prefix = sp + "mlp.experts." + matrix
            nvfp4 = self.store.has(prefix + "_weight_amax")
            for expert in range(experts):
                ep = p + f"moe/experts/{expert}/"
                if nvfp4:
                    source = nvfp4_expert_source(self.store, prefix, expert, shape, interleaved=interleaved)
                else:
                    source = mxfp8_expert_source(self.store, prefix, expert, shape, interleaved=interleaved)
                inputs = (p + "ffn_input",) if role == "gate_up" else (ep + "product",)
                self.add(ep + role, source, inputs=inputs)
                names[role].append(ep + role)
        # Two banks per matrix: expert IDs [0, E/2) and [E/2, E).
        half = experts // 2
        for role in ("gate_up", "down"):
            self.model.packing_groups.append(tuple(names[role][:half]))
            self.model.packing_groups.append(tuple(names[role][half:]))

    def core(self, p, sp, g, layer):
        h, rank = g["hidden_size"], g["gate_rank"]
        self.add(p + "message_queries", message_queries_source(self.store, layer - g["core_start"], h), direct="fp32")
        for role, controller in zip(MESSAGE_ROLES, _CONTROLLERS):
            mp, cp = p + f"message_gate/{role}/", sp + f"controllers.{controller}."
            self.tensor(mp + "down", cp + "down_weight", (rank, 2 * h), inputs=(mp + "input",))
            self.tensor(mp + "forget", cp + "forget_weight", (h, rank), fp32=True)
            self.tensor(mp + "gain", cp + "gain_weight", (h, rank), fp32=True)

    def mtp(self, config, g, head):
        """The DSpark drafter (``self.store`` is ``mtp.safetensors``); ``head`` is the output head's source."""
        h = g["hidden_size"]
        nh, nkv, d = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        f, rank, layers = config["intermediate_size"], config["markov_rank"], config["num_hidden_layers"]
        if config["hidden_size"] != h:
            raise ValueError("mtp: the drafter's hidden size must be the Text component's")
        self.tensor("mtp/fuse_weight", "fuse_weight", (layers + 1, len(config["target_layer_ids"])))
        self.tensor("mtp/mask_embedding", "mask_embed", (h,))
        self.tensor("mtp/final_norm", "norm.weight", (h,))
        for layer in range(layers):
            p, sp = f"mtp/layers/{layer}/", f"layers.{layer}."
            a, ap = p + "attention/", sp + "self_attn."
            self.tensor(p + "context_norm", sp + "dflare_norm.weight", (h,))
            self.tensor(p + "input_norm", sp + "input_layernorm.weight", (h,))
            self.tensor(p + "post_attention_norm", sp + "post_attention_layernorm.weight", (h,))
            for role, field, rows in (
                ("query", "q_proj", nh * d),
                ("key", "k_proj", nkv * d),
                ("value", "v_proj", nkv * d),
            ):
                self.tensor(a + role, ap + field + ".weight", (rows, h), inputs=(a + "input",))
            self.tensor(a + "output", ap + "o_proj.weight", (h, nh * d), inputs=(a + "output_input",))
            for role, field in (("context_key", "k_proj_ctx"), ("context_value", "v_proj_ctx")):
                self.tensor(a + role, ap + field + ".weight", (nkv * d, h), inputs=(p + "context",))
            for role, field in (("query_norm", "q_norm"), ("key_norm", "k_norm"), ("context_key_norm", "k_norm_ctx")):
                self.tensor(a + role, ap + field + ".weight", (d,))
            for role, field in (("gate", "gate_proj"), ("up", "up_proj")):
                self.tensor(p + "mlp/" + role, sp + "mlp." + field + ".weight", (f, h), inputs=(p + "mlp/input",))
            self.tensor(p + "mlp/down", sp + "mlp.down_proj.weight", (h, f), inputs=(p + "mlp/product",))
        m, mp = "mtp/markov/", "markov_head."
        self.tensor(m + "context_norm", mp + "dflare_norm.weight", (h,))
        self.tensor(m + "token_embedding", mp + "markov_w1.weight", (g["vocab_size"], rank))
        self.tensor(m + "recurrence", mp + "rnn_proj.weight", (3 * rank, 2 * rank + 2 * h))
        self.tensor(m + "recurrence_bias", mp + "rnn_proj.bias", (3 * rank,))
        self.add(m + "folded_output", folded_source(head, self.store, mp + "hidden_out.weight", h, rank, False))
        self.add(m + "folded_bias", folded_source(head, self.store, mp + "hidden_out.bias", h, rank, True))

    def dflash2(self, config, g):
        """The DFlash2 drafter (``self.store`` is the drafter's checkpoint)."""
        h = g["hidden_size"]
        nh, nkv, d = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        f, layers = config["intermediate_size"], config["num_hidden_layers"]
        extra = config["dflash_config"]
        features, rank = len(extra.get("target_layer_ids", ())), extra.get("selector_rank")
        taps, group = extra.get("conv_kernel_size"), extra.get("conv_group_size")
        if not features or not rank or not taps or not group or h % group:
            raise ValueError("dflash2: invalid dflash_config")
        coefficients = 2 * taps * (h // group)
        self.tensor("dflash2/feature_projection", "fc.weight", (h, features * h), inputs=("dflash2/target_features",))
        self.tensor("dflash2/context_norm", "hidden_norm.weight", (h,))
        self.tensor("dflash2/final_norm", "norm.weight", (h,))
        for layer in range(layers):
            p, sp = f"dflash2/layers/{layer}/", f"layers.{layer}."
            a, ap = p + "attention/", sp + "self_attn."
            self.tensor(p + "input_norm", sp + "input_layernorm.weight", (h,))
            self.tensor(p + "post_attention_norm", sp + "post_attention_layernorm.weight", (h,))
            self.tensor(a + "query", ap + "q_proj.weight", (nh * d, h), inputs=(a + "input",))
            for role, field in (("key", "k_proj"), ("value", "v_proj")):
                self.tensor(a + role, ap + field + ".weight", (nkv * d, h), inputs=(a + "input", "dflash2/context"))
            for role, field in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
                self.tensor(a + role, ap + field + ".weight", (d,))
            self.tensor(a + "output", ap + "o_proj.weight", (h, nh * d), inputs=(a + "output_input",))
            for role, field in (("gate", "gate_proj"), ("up", "up_proj")):
                self.tensor(p + "mlp/" + role, sp + "mlp." + field + ".weight", (f, h), inputs=(p + "mlp/input",))
            self.tensor(p + "mlp/down", sp + "mlp.down_proj.weight", (h, f), inputs=(p + "mlp/product",))
            for branch in ("attention_conv", "mlp_conv"):
                self.tensor(p + branch + "/base_kernel", sp + branch + ".base_kernel", (2, taps, h))
                self.tensor(
                    p + branch + "/kernel_projection",
                    sp + branch + ".kernel_projection.weight",
                    (coefficients, h),
                    inputs=(p + branch + "/input",),
                )
        s = "dflash2/candidate_selector/"
        self.tensor(
            s + "hidden_projection",
            "candidate_selector.hidden_projection.weight",
            (rank, h),
            inputs=("dflash2/final_hidden",),
        )
        for role in ("predecessor_codebook", "successor_codebook"):
            self.tensor(s + role, "candidate_selector." + role, (g["vocab_size"], rank))

    def layer(self, g, layer):
        h, taps = g["hidden_size"], g["sconv_kernel_size"]
        p, sp = f"text/layers/{layer}/", f"model.llm.layers.{layer}."
        if layer > 0:
            rows = 3 if global_layer(layer) else 1
            self.queries(p + "attention_queries", f"model.llm.attn_res.attn_queries.{layer - 1}", h, rows)
        if global_layer(layer):
            for role, field in (("query", "q"), ("key", "k"), ("value", "v")):
                self.tensor(p + role + "_input_norm", sp + field + "_input_norm.weight", (h,))
        else:
            self.tensor(p + "input_norm", sp + "attn_norm.weight", (h,))
        self.attention(p, sp, g, layer)
        self.tensor(
            p + "post_attention_conv", sp + "attn_sconv.weight", (h, taps), source_shape=(h, 1, taps), fp32=True
        )
        self.queries(p + "ffn_queries", f"model.llm.attn_res.mlp_queries.{layer}", h)
        self.tensor(p + "ffn_norm", sp + "mlp_norm.weight", (h,))
        experts = g["n_routed_experts"]
        self.tensor(p + "moe/router", sp + "mlp.gate.weight", (experts, h), fp32=True)
        self.tensor(p + "moe/router_bias", sp + "mlp.gate.bias", (experts,))
        self.experts(p, sp, g, layer)
        self.tensor(p + "post_ffn_conv", sp + "mlp_sconv.weight", (h, taps), source_shape=(h, 1, taps), fp32=True)
        if g["core_start"] <= layer < g["core_end"]:
            self.core(p, sp, g, layer)


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, SafetensorsSource] | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    companions = dict(companions or {})
    if components[:1] != ("text",) or not set(components[1:]) <= {"mtp", "dflash2"}:
        raise ValueError("Inkling 2 converts the text component and optionally mtp and dflash2")
    if ("dflash2" in components) != ("dflash2" in companions):
        raise ValueError("Inkling 2 dflash2 requires its source (--source dflash2=PATH)")
    config = text_config(base.config)
    g = _geometry(config)
    records = {"text": {"config": config}}
    if "mtp" in components:
        records["mtp"] = {"config": mtp_config(base.config), "target": "text"}
    if "dflash2" in components:
        records["dflash2"] = {"config": dflash2_config(companions["dflash2"].config, g), "target": "text"}
    refs, resources, count, special = load_resources(
        base.root, vocab_size=g["vocab_size"], overrides=resource_overrides
    )
    if count > g["unpadded_vocab_size"]:
        raise ValueError("tokenizer ids exceed the output head rows")
    records["text"]["resources"] = refs["text"]
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Builder(model, base)
    h, vocab, unpadded = g["hidden_size"], g["vocab_size"], g["unpadded_vocab_size"]
    builder.tensor("text/token_embedding", "model.llm.embed.weight", (vocab, h))
    builder.tensor("text/embedding_norm", "model.llm.embed_norm.weight", (h,))
    for layer in range(g["num_hidden_layers"]):
        builder.layer(g, layer)
    builder.queries("text/final_queries", "model.llm.attn_res.output_query", h)
    builder.tensor("text/final_norm", "model.llm.norm.weight", (h,))
    head = select_rows(tensor_source(base, "model.llm.unembed.weight", (vocab, h)), ((0, unpadded),))
    builder.add("text/output_head", head, inputs=("text/final_hidden",))
    if "mtp" in components:
        _Builder(model, SafetensorsSource(base.root / MTP_FILE)).mtp(records["mtp"]["config"], g, head)
    if "dflash2" in components:
        _Builder(model, companions["dflash2"]).dflash2(records["dflash2"]["config"], g)
    return model
