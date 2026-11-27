target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/draft/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/bindings.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/draft.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/expert_banks.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/sharding.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/text.cpp"
)

target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/draft/dflash2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/draft/draft.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/draft/paired.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/draft/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/prefetch_lane.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/tp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/conv_record.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/visit_state.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/speculative/round.cpp"
)
