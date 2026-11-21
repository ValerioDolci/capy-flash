add_library(ninfer_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
ninfer_test_includes(ninfer_linear_test_support)
ninfer_op_oracle_options(ninfer_linear_test_support)
target_link_libraries(ninfer_linear_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

# 27b-16gb: q3_g128_fp16. The storage test is CPU-only; the A16 test needs a GPU (exit 77).
ninfer_add_test(ninfer_q3_storage_host_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q3_storage_host.cpp")

ninfer_add_op_test(ninfer_linear_q3_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q3_a16.cpp"
  LIBRARIES ninfer_ops)

# Q3 prefill: the A8 prefill routes (tall workspace route == in-kernel quantizing tile, bit for
# bit, and == a host emulation); NINFER_Q3_BENCH=1|2 adds (2: only) a timing table.
ninfer_add_op_test(ninfer_linear_q3_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q3_a8.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q6_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q6_head_route_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_head_route.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_mxfp8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mxfp8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_bf16_inkling2_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_inkling2.cpp"
  LIBRARIES ninfer_ops)
