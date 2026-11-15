# q3_g128_fp16 routes (Qwen3.8-27B 3-bit GSQ body).
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q3_linear.cu"
)
