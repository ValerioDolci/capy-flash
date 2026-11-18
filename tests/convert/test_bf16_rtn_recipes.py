"""RTN-from-BF16 recipes: the measured 27B layouts re-created from a dense BF16 checkpoint alone."""

from __future__ import annotations

import pytest
import torch

from tests.convert import qwen4_exp_fixture as fixture
from tools.convert import qwen3_5, qwen4_exp
from tools.convert.methods import cast_direct, fp8_row_maxabs, grouped_absmax, nvfp4_mse
from tools.convert.official_recipes import RECIPES
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource

from .test_qwen3_5 import _checkpoint, _config

FP8 = "fp8_e4m3fn_row_bf16"
NVFP4 = ("nvfp4", nvfp4_mse)


def _choice(recipe, name):
    (selection,) = recipe.selections[name]
    return selection.format, selection.method


def _policies(recipe, name):
    return {policy for (parameter, _), policy in recipe.policies.items() if parameter == name}


def _layer_projections(model):
    return [
        name for name, parameter in model.parameters.items() if name.startswith("text/layers/") and parameter.projection
    ]


def _check_layout(model, recipe, attention, *, quantized):
    """Layer 0 of the two-layer fixtures is in the first 7/8 of the layers, layer 1 is not."""
    assert _choice(recipe, "text/token_embedding") == (FP8, fp8_row_maxabs)
    assert _choice(recipe, "text/output_head") == (FP8, fp8_row_maxabs)
    assert _policies(recipe, "text/output_head") == {"AllowA8"}
    checked = 0
    for projection in _layer_projections(model):
        if not quantized(projection):
            continue
        checked += 1
        if attention or ("/mlp/" in projection and projection.startswith("text/layers/0/")):
            assert _choice(recipe, projection) == NVFP4, projection
            assert _policies(recipe, projection) == {"A16Only"}, projection  # no calibration
        else:
            assert _choice(recipe, projection) == (FP8, fp8_row_maxabs), projection
            assert _policies(recipe, projection) == {"AllowA8"}, projection
    assert checked


@pytest.mark.parametrize(
    ("name", "attention"), [("qwen3_8_27b_bf16_nvfp4_rtn", False), ("qwen3_8_27b_bf16_nvfp4_rtn_all", True)]
)
def test_qwen3_5_dense_layouts(tmp_path, name, attention):
    base = _checkpoint(tmp_path / "text", _config(), {"unused": torch.ones(1)})
    model = qwen3_5.build_model(base, components=("text", "vision", "mtp"), companions={})
    recipe = Recipe(model)
    RECIPES[name](model, recipe, {"base": base})
    gdn_ab = ("/gdn/a_projection", "/gdn/b_projection")
    _check_layout(model, recipe, attention, quantized=lambda p: not p.endswith(gdn_ab))
    for projection in ("text/layers/0/gdn/a_projection", "text/layers/0/gdn/b_projection"):
        assert recipe.selections[projection][0].format == "bf16"
    assert _choice(recipe, "mtp/layers/0/mlp/down") == NVFP4
    assert _choice(recipe, "mtp/input_projection") == ("q8_g32_fp16", grouped_absmax)
    fixture.write_checkpoint(tmp_path / "qwen4", dense=True)
    with SafetensorsSource(tmp_path / "qwen4") as source:
        qwen4 = qwen4_exp.build_model(source, components=("text", "mtp"))
        with pytest.raises(ValueError, match="Qwen3.5 Dense"):
            RECIPES[name](qwen4, Recipe(qwen4), {"base": source})


def test_qwen4_dense_layout(tmp_path):
    name, attention = "qwen4_nvfp4_rtn_all", True
    source_dir = tmp_path / "source"
    fixture.write_checkpoint(source_dir, dense=True)
    with SafetensorsSource(source_dir) as source:
        model = qwen4_exp.build_model(source, components=("text", "mtp"))
        recipe = Recipe(model)
        RECIPES[name](model, recipe, {"base": source})
        reference = Recipe(model)
        RECIPES["qwen4_nvfp4"](model, reference, {"base": source})
    in_set = {name for name in _layer_projections(model) if _choice(reference, name)[0] == "nvfp4"}
    _check_layout(model, recipe, attention, quantized=in_set.__contains__)
    # Outside qwen4_nvfp4's set (hyper-connection mixers, indexer, PLE, GDN a|b) and in the MTP layer:
    # the same choice as qwen4_nvfp4.
    vocabulary = ("text/token_embedding", "text/output_head")
    for projection, parameter in model.parameters.items():
        if parameter.projection and projection not in in_set and projection not in vocabulary:
            assert _choice(recipe, projection) == _choice(reference, projection), projection
    assert _choice(recipe, "mtp/layers/0/mlp/down") == NVFP4
    assert _choice(recipe, "text/layers/0/attn_hc/up") == ("bf16", cast_direct)
