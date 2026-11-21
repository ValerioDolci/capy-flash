target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/residual_f32.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
)
