# Vendored ggml-iq

capy-flash vendors fngine's standalone extract of ggml's reference code for the GSQ-RCO block
formats, unchanged except for the local adjustment noted at the end:

- source: <https://github.com/satellitedown/fngine>, commit `c20a885`, directory `third_party/ggml-iq`
- upstream: ggml (llama.cpp) commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`; what was extracted and
  how it was changed is in `SOURCE.md` (fngine's record)
- license: MIT; see `LICENSE`

The host kernels use the grids and block structs (`ggml-iq-tables.h`, `ggml-iq.h`); the
dequantizers, reference quantizers and generic dot products in `ggml-iq.c` are the oracles of
the GGUF tests (`ninfer_ggml_iq_reference` in tests/ops/tests.cmake, built without FMA contraction);
tools/convert/gguf_experts.py builds it to quantize IQ4_NL.

Local adjustment: `kvalues_iq4nl` in `ggml-iq-tables.h` is `static constexpr` when compiled as C++ (the AVX2
kernels read it in a constant expression) and `static const` when compiled as C (`ggml-iq.c`; C has no
`constexpr` before C23). Values unchanged.
