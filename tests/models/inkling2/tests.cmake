# Inkling 2 Text config on the released and the synthetic checkpoint's config.json (copies in
# this directory), and the 48-visit schedule (CPU only).
ninfer_add_test(ninfer_inkling2_config_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_config.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_loading)

# Tensor-parallel placement of every logical parameter of the released and the synthetic
# geometry (CPU only).
ninfer_add_test(ninfer_inkling2_shard_map_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_shard_map.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_runtime)

# Loading at tp 1 and tp 2: every device parent byte-compared with the host-applied slice, the
# per-rank views and Parameters. The artifacts are not built here: tools/convert/inkling2.py
# writes them from the synthetic reference checkpoint (tools/ref_ink/synthetic.py) and from the
# released checkpoint. Left empty, the tests skip (77).
set(NINFER_TEST_INKLING2_SYNTH_ARTIFACT "" CACHE FILEPATH
  "Synthetic Inkling 2 artifact (tools/ref_ink/synthetic.py + tools/convert) for the loading tests")
set(NINFER_TEST_INKLING2_ARTIFACT "" CACHE FILEPATH
  "Inkling 2 Mini artifact for ninfer_inkling2_loading_tp2_real_test")
set(inkling2_synth_args)
if(NINFER_TEST_INKLING2_SYNTH_ARTIFACT)
  set(inkling2_synth_args --artifact "${NINFER_TEST_INKLING2_SYNTH_ARTIFACT}")
endif()
set(inkling2_real_args)
if(NINFER_TEST_INKLING2_ARTIFACT)
  set(inkling2_real_args --artifact "${NINFER_TEST_INKLING2_ARTIFACT}")
endif()
ninfer_add_test(ninfer_inkling2_loading_tp2_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_tp2.cpp"
  LIBRARIES ninfer_model_runtime
  ARGS ${inkling2_synth_args} --tp 2)
add_test(NAME ninfer_inkling2_loading_tp1_test
  COMMAND ninfer_inkling2_loading_tp2_test ${inkling2_synth_args} --tp 1)
add_test(NAME ninfer_inkling2_loading_tp2_real_test
  COMMAND ninfer_inkling2_loading_tp2_test ${inkling2_real_args} --tp 2)
# The same with the token embedding in one mapped page-locked host copy (--embedding-host).
add_test(NAME ninfer_inkling2_loading_embedding_host_tp2_test
  COMMAND ninfer_inkling2_loading_tp2_test ${inkling2_synth_args} --tp 2 --embedding-host)
add_test(NAME ninfer_inkling2_loading_embedding_host_tp1_test
  COMMAND ninfer_inkling2_loading_tp2_test ${inkling2_synth_args} --tp 1 --embedding-host)
set_tests_properties(
  ninfer_inkling2_loading_tp2_test
  ninfer_inkling2_loading_tp1_test
  ninfer_inkling2_loading_tp2_real_test
  ninfer_inkling2_loading_embedding_host_tp2_test
  ninfer_inkling2_loading_embedding_host_tp1_test
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK ninfer_gpu_memory)
# Inkling 2 frontend against the Hugging Face reference ids (tokenizer_fixture.json, written by
# tools/ref_ink/tokenizer_fixtures.py). Point NINFER_TEST_INKLING2_TOKENIZER_DIR at an Inkling 2 Mini
# checkpoint directory (tokenizer.json, tokenizer_config.json, generation_config.json,
# chat_template.jinja); without it the test skips.
set(NINFER_TEST_INKLING2_TOKENIZER_DIR "" CACHE PATH
  "Inkling 2 Mini checkpoint directory with tokenizer files for ninfer_inkling2_frontend_test")
ninfer_add_test(ninfer_inkling2_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_inkling2_frontend)
set_tests_properties(ninfer_inkling2_frontend_test PROPERTIES SKIP_RETURN_CODE 77)
if(NINFER_TEST_INKLING2_TOKENIZER_DIR)
  set_tests_properties(ninfer_inkling2_frontend_test PROPERTIES
    ENVIRONMENT "NINFER_TEST_INKLING2_TOKENIZER_DIR=${NINFER_TEST_INKLING2_TOKENIZER_DIR}")
endif()

# The Inkling 2 Frontend contract: preparation and prefix frontiers on the same fixture, real
# greedy turns of the model (greedy_fixture.json, tools/ref_ink/frontend_greedy_fixture.py) and
# recorded token sequences through output sessions. CancellationView comes from
# public_types.cpp (ninfer_runtime_support in the Engine), compiled here to keep the test CPU only.
ninfer_add_test(ninfer_inkling2_frontend_contract_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend_contract.cpp"
          "${PROJECT_SOURCE_DIR}/src/runtime/engine/public_types.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_inkling2_frontend)
set_tests_properties(ninfer_inkling2_frontend_contract_test PROPERTIES SKIP_RETURN_CODE 77)
if(NINFER_TEST_INKLING2_TOKENIZER_DIR)
  set_tests_properties(ninfer_inkling2_frontend_contract_test PROPERTIES
    ENVIRONMENT "NINFER_TEST_INKLING2_TOKENIZER_DIR=${NINFER_TEST_INKLING2_TOKENIZER_DIR}")
endif()

# The Text forward below the Engine (inkling2_tf_driver.cpp): teacher-forcing dumps (tfdump/1)
# and greedy generation by hand, and two registered checks on the synthetic artifact with the
# reference's ids (tokens.txt beside it): tp 2 against tp 1 within twice the reference's bf16
# floor (KL of the logits, bit-identical replicated values on both ranks), and prefill blocks of
# 16 against token-by-token decode, bit for bit. Without the artifact they skip (77).
ninfer_add_test(ninfer_inkling2_tf_driver
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/inkling2_tf_driver.cpp"
  LIBRARIES ninfer_model_runtime)
set(inkling2_forward_args)
if(NINFER_TEST_INKLING2_SYNTH_ARTIFACT)
  get_filename_component(inkling2_synth_dir "${NINFER_TEST_INKLING2_SYNTH_ARTIFACT}" DIRECTORY)
  set(inkling2_forward_args --artifact "${NINFER_TEST_INKLING2_SYNTH_ARTIFACT}"
    --ids "${inkling2_synth_dir}/tokens.txt")
endif()
set_tests_properties(ninfer_inkling2_tf_driver PROPERTIES DISABLED TRUE)
add_test(NAME ninfer_inkling2_forward_tp2_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode tp-parity --block 64
  --kl-mean 8.6e-3 --kl-max 2.2e-2)
add_test(NAME ninfer_inkling2_forward_blocks_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode blocks --tp 1 --block 16 --identical)
# Speculative verification rounds with fake drafts (--mode spec) at tp 1 and tp 2: the greedy's
# ids, the expected licensed count of every round and every committed column's logits equal to
# the decode step's, bit for bit, for K = 1, 3, 5, 7 and drafts all right, all wrong and right up
# to a random j.
add_test(NAME ninfer_inkling2_speculative_round_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode spec --tp 1 --block 16
  --max-new 64 --draft-tokens 1,3,5,7 --draft-rule right,wrong,random)
add_test(NAME ninfer_inkling2_speculative_round_tp2_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode spec --tp 2 --block 16
  --max-new 64 --draft-tokens 1,3,5,7 --draft-rule right,wrong,random)
# The drafter's feature sinks at tp 2 (--mode features): every rank's features equal its visit
# inputs and head input bit for bit, the ranks agree and the logits are unchanged.
add_test(NAME ninfer_inkling2_features_tp2_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode features --tp 2 --block 64)
# The verification round with the artifact's drafter proposing (--mode spec --drafter) at tp 1 and
# tp 2 on the synthetic artifact with its mtp component: the ids of the greedy without drafter and
# every committed column's logits, bit for bit, for K = 1, 3, 7.
add_test(NAME ninfer_inkling2_drafter_round_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode spec --tp 1 --block 16
  --max-new 64 --draft-tokens 1,3,7 --drafter)
add_test(NAME ninfer_inkling2_drafter_round_tp2_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode spec --tp 2 --block 16
  --max-new 64 --draft-tokens 1,3,7 --drafter)
# The same with the token embedding read in place from host memory (--embedding-host): prefill,
# the decode graph and the drafter's anchor gather it from the one mapped copy.
add_test(NAME ninfer_inkling2_drafter_round_embedding_host_tp2_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode spec --tp 2 --block 16
  --max-new 64 --draft-tokens 1,3,7 --drafter --embedding-host)
set_tests_properties(
  ninfer_inkling2_forward_tp2_test
  ninfer_inkling2_forward_blocks_test
  ninfer_inkling2_speculative_round_test
  ninfer_inkling2_speculative_round_tp2_test
  ninfer_inkling2_features_tp2_test
  ninfer_inkling2_drafter_round_test
  ninfer_inkling2_drafter_round_tp2_test
  ninfer_inkling2_drafter_round_embedding_host_tp2_test
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK ninfer_gpu_memory)

# Prefix reuse through the public Engine on the real artifact at tp 2 (test_prefix_reuse.cpp):
# endpoint and prompt-anchor resumes against the same requests from the root, and an eviction.
# Without the artifact it skips (77).
ninfer_add_test(ninfer_inkling2_prefix_reuse_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_reuse.cpp"
  LIBRARIES ninfer_engine
  ARGS ${inkling2_real_args} --tp 2)
set_tests_properties(ninfer_inkling2_prefix_reuse_test PROPERTIES SKIP_RETURN_CODE 77)
# The same with speculative rounds of the artifact's drafter (--spec 3), every prompt token in its
# own block so that resumed and cold runs compare bit for bit.
add_test(NAME ninfer_inkling2_prefix_reuse_spec_test
  COMMAND ninfer_inkling2_prefix_reuse_test ${inkling2_real_args} --tp 2 --prefill-chunk 1 --spec 3)
set_tests_properties(ninfer_inkling2_prefix_reuse_spec_test PROPERTIES SKIP_RETURN_CODE 77)

# The decode step from the captured CUDA Graph against the same step run eagerly, through the
# Program's execution (test_decode_graph.cpp): visit state and soft-capped logits byte for byte
# after every step, on the synthetic artifact at tp 1 and tp 2 and on the real one at tp 2.
# Without the artifacts they skip (77).
ninfer_add_test(ninfer_inkling2_decode_graph_tp2_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_decode_graph.cpp"
  LIBRARIES ninfer_engine ninfer_model_runtime
  ARGS ${inkling2_synth_args} --tp 2)
add_test(NAME ninfer_inkling2_decode_graph_tp1_test
  COMMAND ninfer_inkling2_decode_graph_tp2_test ${inkling2_synth_args} --tp 1)
add_test(NAME ninfer_inkling2_decode_graph_tp2_real_test
  COMMAND ninfer_inkling2_decode_graph_tp2_test ${inkling2_real_args} --tp 2)
set_tests_properties(
  ninfer_inkling2_decode_graph_tp2_test
  ninfer_inkling2_decode_graph_tp1_test
  ninfer_inkling2_decode_graph_tp2_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK ninfer_gpu_memory)

# The loop contrast (--loopcd) on the same checks: the drafter's verification rounds against the
# greedy without it, both reading the guided logits (Logits at tp 1, adaptive at the default
# strength and Hidden at tp 2, synthetic artifact); the decode graph against eager with the
# adaptive contrast (synthetic and real, tp 2); prefix reuse with the drafter on the real artifact,
# the reference coda's slots in the images.
add_test(NAME ninfer_inkling2_loopcd_drafter_round_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode spec --tp 1 --block 16
  --max-new 64 --draft-tokens 1,3,7 --drafter --loopcd logits --loopcd-omega 0.3)
add_test(NAME ninfer_inkling2_loopcd_drafter_round_tp2_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode spec --tp 2 --block 16
  --max-new 64 --draft-tokens 1,3,7 --drafter --loopcd adaptive)
add_test(NAME ninfer_inkling2_loopcd_hidden_drafter_round_tp2_test
  COMMAND ninfer_inkling2_tf_driver ${inkling2_forward_args} --mode spec --tp 2 --block 16
  --max-new 64 --draft-tokens 1,3,7 --drafter --loopcd hidden --loopcd-omega 0.3)
add_test(NAME ninfer_inkling2_loopcd_decode_graph_tp2_test
  COMMAND ninfer_inkling2_decode_graph_tp2_test ${inkling2_synth_args} --tp 2 --loopcd adaptive)
add_test(NAME ninfer_inkling2_loopcd_decode_graph_tp2_real_test
  COMMAND ninfer_inkling2_decode_graph_tp2_test ${inkling2_real_args} --tp 2 --loopcd adaptive)
add_test(NAME ninfer_inkling2_loopcd_prefix_reuse_spec_test
  COMMAND ninfer_inkling2_prefix_reuse_test ${inkling2_real_args} --tp 2 --prefill-chunk 1 --spec 3
  --loopcd adaptive)
set_tests_properties(
  ninfer_inkling2_loopcd_drafter_round_test
  ninfer_inkling2_loopcd_drafter_round_tp2_test
  ninfer_inkling2_loopcd_hidden_drafter_round_tp2_test
  ninfer_inkling2_loopcd_decode_graph_tp2_test
  ninfer_inkling2_loopcd_decode_graph_tp2_real_test
  ninfer_inkling2_loopcd_prefix_reuse_spec_test
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK ninfer_gpu_memory)
