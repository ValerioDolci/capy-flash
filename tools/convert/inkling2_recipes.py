"""Inkling 2 Mini representation recipe (``--recipe tools/convert/inkling2_recipes.py:inkling2_mini``).

Dense parameters keep their stored BF16 or FP32 words (the builder's direct formats). Experts of
the NVFP4 layers are imported without requantization: codes, E4M3 block scales and one weight
divisor per expert (``import_encoded``; A16Only, activations are not quantized). Experts of the
MXFP8 layers keep their E4M3FN codes and UE8M0 exponents (``import_verbatim`` into
``mxfp8_e4m3fn_g32_ue8m0``, layout ``block_scale_k32_v1``). Each layer's experts form two parents per matrix,
expert IDs [0, E/2) and [E/2, E), so that a tensor-parallel rank holds one bank whole.

The attention input projections that read one input are stacked in one parent, so the forward
issues one Linear per input: a local layer's query, gate, relative, sink, key and value rows (one
input outside the cross-pass messages; with them, the query rows and the key and value rows are
read through separate views of the same parent), a global layer's query, gate, relative and sink
rows, its key and key-convolution projection rows, and its value and value-convolution projection
rows. Each projection keeps its head-split rows, so a tensor-parallel rank's shard of a parent is
its rows of every block, in this order.

The DSpark drafter (component ``mtp``) keeps its BF16 words and stacks, per layer, the query, key
and value rows (one input), the context key and value rows (one input) and the MLP gate and up
rows (one input), with the same head- or half-split per block.

The DFlash2 drafter (component ``dflash2``) stores its projections Q8 (``q8_g32_fp16``, group absmax),
as the official Qwen recipes store theirs, except the convolution kernel projections and the
selector's hidden projection (BF16). Per layer the query, key and value rows form one parent (the
key and value rows also project the context) and the MLP gate and up rows another.
"""

from __future__ import annotations

from tools.convert.methods import grouped_absmax, import_encoded
from tools.convert.official_recipes import MXFP8, Q8, import_verbatim


def inkling2_mini(model, recipe, sources):
    del sources
    for names in model.packing_groups:
        if not all("/moe/experts/" in name for name in names):
            raise ValueError("Inkling 2 groups only expert banks")
        formats = {model.parameters[name].source.read_encoded(0, 1).format for name in names}
        if formats == {"nvfp4"}:
            recipe.assign(names, format="nvfp4", method=import_encoded)
        elif formats == {MXFP8}:
            recipe.assign(names, format=MXFP8, layout="block_scale_k32_v1", method=import_verbatim)
        else:
            raise ValueError(f"{names[0]}: a bank mixes expert formats {sorted(formats)}")
        recipe.group(names)
    text = [name for name in model.parameters if name.startswith("text/")]
    for prefix in sorted({name.rsplit("/attention/", 1)[0] for name in text if "/attention/" in name}):
        a = prefix + "/attention/"
        query = [a + role for role in ("query", "gate", "relative", "sink")]
        if a + "key_conv_projection" in model.parameters:
            recipe.group(query)
            recipe.group([a + "key", a + "key_conv_projection"])
            recipe.group([a + "value", a + "value_conv_projection"])
        else:
            recipe.group([*query, a + "key", a + "value"])
    drafter = [name for name in model.parameters if name.startswith("mtp/layers/") and "/attention/" in name]
    for prefix in sorted({name.rsplit("/attention/", 1)[0] for name in drafter}):
        a = prefix + "/attention/"
        recipe.group([a + "query", a + "key", a + "value"])
        recipe.group([a + "context_key", a + "context_value"])
        recipe.group([prefix + "/mlp/gate", prefix + "/mlp/up"])
    dflash2 = [name for name in model.parameters if name.startswith("dflash2/")]
    for name in dflash2:
        if model.parameters[name].projection and not name.endswith(
            ("/kernel_projection", "/candidate_selector/hidden_projection")
        ):
            recipe.assign(name, format=Q8, method=grouped_absmax)
    for prefix in sorted({name.rsplit("/attention/", 1)[0] for name in dflash2 if "/attention/" in name}):
        recipe.group([prefix + "/attention/" + role for role in ("query", "key", "value")])
        recipe.group([prefix + "/mlp/gate", prefix + "/mlp/up"])
