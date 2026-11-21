target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/moe_sigmoid_route.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
)
