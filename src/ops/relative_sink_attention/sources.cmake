target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/relative_sink_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/relative_sink_attention_kernels.cu"
)
