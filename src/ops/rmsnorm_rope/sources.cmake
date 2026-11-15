target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/tp2/rmsnorm_rope_tp2.cu"
  "${CMAKE_CURRENT_LIST_DIR}/tp2/rmsnorm_rope_text_shard.cu"
)
