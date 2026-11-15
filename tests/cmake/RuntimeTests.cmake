ninfer_add_test(ninfer_admission_policy_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_admission_policy.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_context_cost_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost.cpp"
  LIBRARIES ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_resource_manager_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_resource_manager.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_kv_capacity_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_capacity.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_sampling_defaults.cpp"
  LIBRARIES ninfer_engine ninfer_core)

# The tp 2 eager transport decision table, host only (header-only, no CUDA).
ninfer_add_test(ninfer_tp2_eager_transport_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tp2_eager_transport.cpp")

# The sliced tp 2 prefill's column-slice plan, host only (header-only, no CUDA).
ninfer_add_test(ninfer_tp2_prefill_slices_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tp2_prefill_slices.cpp")
