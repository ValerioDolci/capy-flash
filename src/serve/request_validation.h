#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

namespace ninfer::text {
struct ParsedJsonNumbers;
}

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

void require_object_body(const RequestJson& body);
std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;
// The `name` of an OpenAI function object.
std::string require_function_name(const RequestJson& object, const char* param);
// tool_choice allowed_tools (Chat Completions, Responses): keeps the declared tools named in
// `allowed`, in declaration order, and records the others in filtered_out_tool_names.
void keep_allowed_tools(GenerationRequest& request, const std::unordered_set<std::string>& allowed);

// Structured output (constrained decoding).

// Rejects a request whose output-format schema contains a number whose JSON spelling would not
// survive the double representation (1e400, 0.30000000000000001...). Only the schema locations
// are checked; tool schemas are not constrained.
void validate_schema_number_input(const text::ParsedJsonNumbers& parsed,
                                  const RequestLimits& limits);

// True when the parsed body carries an output-format schema, i.e. when the exact-number check
// above is worth a second parse. Unconstrained bodies are parsed once.
[[nodiscard]] bool has_output_format_schema(const RequestJson& body);

enum class JsonFormatProtocol { Chat, Responses, Anthropic };

// Chat response_format, Responses text.format, Anthropic output_config.format. "text" (OpenAI
// only) leaves the request unconstrained; json_object (OpenAI only) and json_schema set the
// request constraint with param pointing at the schema.
void parse_json_output_format(const RequestJson& format, GenerationRequest& request,
                              const std::string& param, JsonFormatProtocol protocol,
                              const RequestLimits& limits);

// The NInfer structured_outputs extension ({grammar|regex|choice}), the rejection of the other
// constrained-decoding spellings (guided_*, non-empty grammar), and the combination rules that
// must hold before the Engine sees a constrained request. Call after messages, tools, tool_choice
// and stop strings are parsed.
void parse_structured_outputs(const RequestJson& body, GenerationRequest& request,
                              const RequestLimits& limits);

// Nesting limit of client JSON: the request body, checked on its bytes before parsing, and JSON
// carried in strings (tool-call arguments) before anything parses it.
void check_request_json_nesting(std::string_view text, const std::string& param,
                                const RequestLimits& limits);

// True for the wire fields an output constraint can come from. Protocol encoders without a
// separate error param (Anthropic) use it to keep the schema location in the message.
[[nodiscard]] bool is_output_constraint_param(std::string_view param) noexcept;

} // namespace ninfer::serve
