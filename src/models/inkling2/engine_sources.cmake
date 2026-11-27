# Inkling 2 Program, model instance and Engine family. They join ninfer_engine, where the common
# Engine cores they instantiate live; src/runtime/CMakeLists.txt includes this list after defining
# that target.
target_sources(ninfer_engine PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/program/drafter_choice.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/drafter_costs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/execution.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program_impl.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/instance.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/engine_family.cpp"
)
target_link_libraries(ninfer_engine PRIVATE ninfer_inkling2_frontend)
