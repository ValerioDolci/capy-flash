target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/materialize.cu"
  "${CMAKE_CURRENT_LIST_DIR}/tp2/context_kv_materialize_tp2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/tp2/materialize_shard_h4.cu"
)
