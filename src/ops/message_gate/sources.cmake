target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/message_gate.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/wide.cu"
)
