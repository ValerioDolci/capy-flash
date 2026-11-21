target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/bf16_generic.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n256_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n7168_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k3072.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/qwen4_exp.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/inkling2.cu"
)
