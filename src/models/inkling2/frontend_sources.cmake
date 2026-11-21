# Inkling 2 frontend: tokenizer, chat template, prompt preparation and output sessions behind the
# Engine's frontend contract. A library of its own, linked by the inkling2 model instance; it uses
# CancellationView from ninfer_runtime_support, which the Engine links.
add_library(ninfer_inkling2_frontend STATIC
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tokenizer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/chat_template.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_parser.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/prompt_layout.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
)
ninfer_internal_includes(ninfer_inkling2_frontend)
# Fine Unicode categories of the o200k pre-tokenization come from utf8proc directly.
target_include_directories(ninfer_inkling2_frontend PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_inkling2_frontend PUBLIC ninfer_text ninfer_jinja ninfer::json)
