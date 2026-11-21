target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/depth_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
)
