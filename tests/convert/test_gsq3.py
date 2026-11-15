"""27b-16gb: q3_g128_fp16 registry/codec and the GSQ pack-quantized integer source."""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from tools.artifact.codecs.row_split import decode_row_split_codes, dequantize_row_split, encode_row_split
from tools.artifact.formats import get_format
from tools.artifact.layouts import row_split_geometry
from tools.convert.quantization.groupwise import quantize_matrix
from tools.convert.sources.compressed_tensors import (
    PACKED_SCALE_ABS_ERROR,
    matrix_source,
    packed_scales_fp16,
    unpack_int_stream,
)
from tools.convert.sources.safetensors import SafetensorsSource


def _pack_stream(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """compressed-tensors dense pack: LSB-first stream of q + 2^(bits-1), as I32 words."""
    rows, k = codes.shape
    unsigned = (codes.to(torch.int64) + (1 << (bits - 1))).tolist()
    words = []
    for row in unsigned:
        value = 0
        for i, u in enumerate(row):
            value |= u << (bits * i)
        raw = value.to_bytes(k * bits // 8, "little")
        words.append(torch.frombuffer(bytearray(raw), dtype=torch.int32).clone())
    return torch.stack(words)


def test_q3_format_and_geometry():
    spec = get_format("q3_g128_fp16")
    assert (spec.bits, spec.group_size, spec.qmin, spec.qmax) == (3, 128, -4, 3)
    g = row_split_geometry("q3_g128_fp16", (1, 5120))
    assert (g.k_pad, g.groups_per_row, g.base_row_bytes, g.high_row_bytes, g.scale_row_bytes) == (5120, 40, 1920, 0, 80)
    g = row_split_geometry("q3_g128_fp16", (3, 129))
    assert (g.k_pad, g.groups_per_row, g.base_bytes, g.scale_offset) == (256, 2, 288, 512)


def test_q3_reference_vector_and_round_trip():
    codes = torch.zeros((1, 1, 128), dtype=torch.int8)
    codes[0, 0, :8] = torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3], dtype=torch.int8)
    payload = encode_row_split(codes, torch.ones((1, 1), dtype=torch.float16), "q3_g128_fp16", (1, 128))
    assert payload[:3] == b"\xac\x8f\x68"  # storage-layouts.md §3.3 and roofkid's Q3G128_F16S vector
    generator = torch.Generator().manual_seed(19)
    codes = torch.randint(-4, 4, (5, 3, 128), dtype=torch.int8, generator=generator)
    scales = (torch.rand((5, 3), generator=generator) * 0.05 + 0.001).to(torch.float16)
    payload = encode_row_split(codes, scales, "q3_g128_fp16", (5, 384))
    decoded_scales, decoded_codes = decode_row_split_codes(payload, "q3_g128_fp16", (5, 384))
    assert torch.equal(decoded_codes, codes) and torch.equal(decoded_scales, scales)
    values = dequantize_row_split(payload, "q3_g128_fp16", (5, 384), dtype=torch.float32)
    assert torch.equal(values, (codes.float() * scales.float().unsqueeze(-1)).reshape(5, 384))


def test_q3_grouped_absmax_uses_the_code_domain():
    weight = torch.linspace(-1.0, 1.0, 256).reshape(1, 256)
    q = quantize_matrix(weight, "q3_g128_fp16", device="cpu")
    assert int(q.codes.min()) >= -4 and int(q.codes.max()) <= 3 and q.scales.dtype == torch.float16


def test_unpack_int_stream_matches_reference_packer():
    generator = torch.Generator().manual_seed(7)
    for bits, k in ((3, 256), (4, 128)):
        codes = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (4, k), dtype=torch.int8, generator=generator)
        assert torch.equal(unpack_int_stream(_pack_stream(codes, bits), bits, k), codes)


def test_packed_scales_reject_lossy_binary16():
    ok = torch.tensor([0.01, 2.0**-20], dtype=torch.bfloat16)
    assert packed_scales_fp16(ok, "s").dtype == torch.float16
    tiny = torch.tensor([2.0**-26 * 1.5], dtype=torch.bfloat16)  # error 2^-27: accepted
    assert float((packed_scales_fp16(tiny, "s").float() - tiny.float()).abs()) <= PACKED_SCALE_ABS_ERROR
    for bad, signed in ((70000.0, True), (-0.01, False)):
        try:
            packed_scales_fp16(torch.tensor([bad], dtype=torch.bfloat16), "s", signed=signed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"multiplier {bad} (signed={signed}) must be rejected")
    negative = torch.tensor([-0.01, 0.02], dtype=torch.bfloat16)  # GSQ-3 groups use both signs
    assert torch.equal(packed_scales_fp16(negative, "s", signed=True), negative.to(torch.float16))


def test_gsq_pack_quantized_sources_are_verbatim(tmp_path):
    generator = torch.Generator().manual_seed(2026)
    q3 = torch.randint(-4, 4, (6, 256), dtype=torch.int8, generator=generator)
    s3 = (torch.rand((6, 2), generator=generator) * 0.03 + 0.001).to(torch.bfloat16)
    s3[::2, 1] = -s3[::2, 1]  # GSQ-3: about half of the multipliers are negative
    q4 = torch.randint(-8, 8, (4, 128), dtype=torch.int8, generator=generator)
    s4 = (torch.rand((4, 2), generator=generator) * 0.03 + 0.001).to(torch.bfloat16)
    save_file(
        {
            "body.weight_packed": _pack_stream(q3, 3),
            "body.weight_scale": s3,
            "body.weight_shape": torch.tensor([6, 256], dtype=torch.int64),
            "head.weight_packed": _pack_stream(q4, 4),
            "head.weight_scale": s4,
            "head.weight_shape": torch.tensor([4, 128], dtype=torch.int64),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        body = matrix_source(store, "body.weight", (6, 256))
        rows = body.read_encoded(1, 5)
        assert rows.format == "q3_g128_fp16"
        assert torch.equal(rows.codes, q3[1:5].reshape(4, 2, 128))
        assert torch.equal(rows.scales, s3[1:5].to(torch.float16))
        expected = (q3.float().reshape(6, 2, 128) * s3.float().to(torch.float16).float().unsqueeze(-1)).reshape(-1)
        assert torch.equal(body.values(), expected)
        assert torch.equal(body.values(300, 900), expected[300:900])
        head = matrix_source(store, "head.weight", (4, 128))
        rows = head.read_encoded(0, 4)
        assert rows.format == "q4_g64_fp16" and torch.equal(rows.codes, q4.reshape(4, 2, 64))
        # The embedding re-expression (official_recipes.import_q4_as_q8): two Q8 groups per Q4
        # group, same codes and multiplier -> identical represented values.
        q8_codes = rows.codes.reshape(4, 4, 32)
        q8_scales = rows.scales.repeat_interleave(2, dim=1)
        payload = encode_row_split(q8_codes, q8_scales, "q8_g32_fp16", (4, 128))
        assert torch.equal(
            dequantize_row_split(payload, "q8_g32_fp16", (4, 128), dtype=torch.float32), head.values().reshape(4, 128)
        )
