# CTest labels: `cpu` marks the tests that run without a GPU (they pass with CUDA_VISIBLE_DEVICES
# empty: parsers, protocols, planners, host kernels), so a machine without CUDA devices runs
# `ctest -L cpu`; every other test is `gpu` (it skips or fails without a device). A new test that
# needs no GPU joins this list; check with
#   CUDA_VISIBLE_DEVICES= ctest --test-dir build -L cpu
set(NINFER_CPU_TESTS
  ninfer_admission_policy_test
  ninfer_anthropic_schema_test
  ninfer_artifact_host_residency_test
  ninfer_artifact_reader_test
  ninfer_artifact_slices_test
  ninfer_bench_support_test
  ninfer_byte_size_test
  ninfer_chat_templates_test
  ninfer_cli_options_test
  ninfer_constraint_limits_test
  ninfer_context_cost_measure_test
  ninfer_context_cost_test
  ninfer_decode_policy_routes_test
  ninfer_device_sync_empty_test
  ninfer_device_sync_invalid_test
  ninfer_disk_kv_cache_test
  ninfer_draft_policy_test
  ninfer_expert_cache_policy_test
  ninfer_expert_host_pool_test
  ninfer_gdn_replay_records_test
  ninfer_grammar_test
  ninfer_host_gguf_avx2_test
  ninfer_host_gguf_isa_test
  ninfer_host_gguf_test
  ninfer_host_isa_guard_test
  ninfer_host_moe_avx2_test
  ninfer_host_moe_gguf_avx2_test
  ninfer_host_moe_gguf_test
  ninfer_host_moe_test
  ninfer_host_nvfp4_isa_test
  ninfer_host_timing_test
  ninfer_http_error_handler_test
  ninfer_http_transport_test
  ninfer_inkling2_config_test
  ninfer_inkling2_shard_map_test
  ninfer_inkling2_frontend_contract_test
  ninfer_inkling2_frontend_test
  ninfer_jinja_test
  ninfer_json_schema_oracle_test
  ninfer_json_schema_test
  ninfer_kv_capacity_test
  ninfer_materialization_budget_test
  ninfer_media_decode_test
  ninfer_openai_responses_store_test
  ninfer_openai_responses_test
  ninfer_openai_schema_test
  ninfer_output_constraint_test
  ninfer_perplexity_evaluation_test
  ninfer_perplexity_options_test
  ninfer_prefill_miss_dump_test
  ninfer_pretty_logging_test
  ninfer_prompt_input_test
  ninfer_public_api_test
  ninfer_q3_storage_host_test
  ninfer_qwen3_5_disk_identity_test
  ninfer_qwen3_5_frontend_test
  ninfer_qwen3_5_graph_profiles_test
  ninfer_qwen3_5_loading_test
  ninfer_qwen3_5_ple_gather_test
  ninfer_qwen3_5_runtime_mechanisms_test
  ninfer_qwen3_5_shard_map_test
  ninfer_qwen3_5_state_image_layout_test
  ninfer_qwen4_draft_head_test
  ninfer_qwen4_gguf_pool_test
  ninfer_qwen4_shard_map_test
  ninfer_reasoning_retry_test
  ninfer_regex_choice_oracle_test
  ninfer_regex_choice_test
  ninfer_request_json_depth_test
  ninfer_request_log_test
  ninfer_resource_manager_test
  ninfer_sampling_defaults_test
  ninfer_serve_options_test
  ninfer_suffix_drafter_test
  ninfer_tensor_test
  ninfer_tool_call_parser_test
  ninfer_tp2_eager_transport_test
  ninfer_tp2_prefill_slices_test
  ninfer_turn_anchors_test
  ninfer_yarn_test
)

get_directory_property(ninfer_all_tests TESTS)
foreach(ninfer_test IN LISTS ninfer_all_tests)
  if(ninfer_test IN_LIST NINFER_CPU_TESTS)
    set_tests_properties(${ninfer_test} PROPERTIES LABELS cpu)
  else()
    set_tests_properties(${ninfer_test} PROPERTIES LABELS gpu)
  endif()
endforeach()
