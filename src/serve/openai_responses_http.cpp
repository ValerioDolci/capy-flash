#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/openai_responses.h"
#include "serve/request_validation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

struct PendingResponseStorage {
    std::vector<ChatTurn> input_turns;
    std::vector<Json> input_items;
    OpenAIResponseContext previous_context;
    std::string session_key;
    bool enabled = false;
};

ApiError responses_error(ApiError error) {
    if (error.param == "messages") { error.param = "input"; }
    if (error.param == "reasoning_effort") { error.param = "reasoning.effort"; }
    return error;
}

ApiError response_not_found(const std::string& id) {
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "response_id";
    error.code    = "response_not_found";
    error.message = "response '" + id + "' not found";
    return error;
}

OpenAIResponseContext terminal_context(OpenAIResponseContext previous,
                                       std::vector<ChatTurn> input_turns,
                                       std::vector<ChatTurn> output_history) {
    OpenAIResponseContext input =
        append_openai_response_context(std::move(previous), std::move(input_turns));
    return append_openai_response_context(std::move(input), std::move(output_history));
}

void commit_stored_response(OpenAIResponsesStore& store, PendingResponseStorage pending,
                            std::string id, const Json& response,
                            std::vector<ChatTurn> output_history,
                            std::optional<bool> preserve_thinking) {
    if (!pending.enabled) { return; }
    if (pending.session_key.empty()) {
        throw std::logic_error("stored Response has no Engine session key");
    }
    StoredOpenAIResponse stored;
    stored.id                = std::move(id);
    stored.session_key       = std::move(pending.session_key);
    stored.response          = response;
    stored.input_items       = std::move(pending.input_items);
    stored.context           = terminal_context(std::move(pending.previous_context),
                                                std::move(pending.input_turns), std::move(output_history));
    stored.preserve_thinking = preserve_thinking;
    store.put(std::move(stored));
}

OpenAIResponsesRuntimeValues runtime_values(const PreparedRequest& prepared,
                                            const GenerationOutcome* outcome = nullptr) {
    OpenAIResponsesRuntimeValues runtime;
    runtime.temperature = prepared.sampling.temperature;
    runtime.top_p       = prepared.sampling.top_p;
    if (outcome != nullptr) {
        runtime.cached_input_tokens = static_cast<int>(outcome->metrics.prefix_cache_hit_tokens);
    }
    return runtime;
}

[[noreturn]] void invalid_query(std::string message, std::string param,
                                std::string code = "invalid_value") {
    bad_request(std::move(message), std::move(param), std::move(code));
}

void reject_unknown_query(const httplib::Request& request,
                          const std::vector<std::string_view>& allowed) {
    for (const auto& [key, value] : request.params) {
        (void)value;
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
            invalid_query("unknown query parameter: " + key, key, "unknown_parameter");
        }
    }
}

bool query_bool(const httplib::Request& request, const char* key, bool fallback) {
    if (!request.has_param(key)) { return fallback; }
    const std::string value = request.get_param_value(key);
    if (value == "true") { return true; }
    if (value == "false") { return false; }
    invalid_query(std::string(key) + " must be 'true' or 'false'", key);
}

void validate_retrieve_query(const httplib::Request& request) {
    reject_unknown_query(request, {"include", "include_obfuscation", "starting_after", "stream"});
    if (request.has_param("include") && !request.get_param_value("include").empty()) {
        invalid_query("additional stored Response fields are not available", "include",
                      "include_not_supported");
    }
    (void)query_bool(request, "include_obfuscation", true);
    if (request.has_param("starting_after") && !request.get_param_value("starting_after").empty()) {
        invalid_query("stream event recovery is not available for stored Responses",
                      "starting_after", "stream_recovery_not_supported");
    }
    if (query_bool(request, "stream", false)) {
        invalid_query("stored Response event streams are not retained", "stream",
                      "retrieve_stream_not_supported");
    }
}

void redact_input_image_urls(Json& item) {
    if (!item.is_object()) { return; }
    const auto redact_parts = [](Json& parts) {
        if (!parts.is_array()) { return; }
        for (Json& part : parts) {
            if (part.is_object() && part.value("type", "") == "input_image") {
                part.erase("image_url");
            }
        }
    };
    if (item.contains("content")) { redact_parts(item["content"]); }
    const std::string type = item.value("type", "");
    if ((type == "function_call_output" || type == "custom_tool_call_output") &&
        item.contains("output")) {
        redact_parts(item["output"]);
    }
}

std::string path_response_id(const httplib::Request& request) {
    return request.matches.size() > 1 ? request.matches[1].str() : std::string();
}

int parse_limit(const httplib::Request& request) {
    if (!request.has_param("limit")) { return 20; }
    const std::string value = request.get_param_value("limit");
    int parsed              = 0;
    const auto result       = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 1 ||
        parsed > 100) {
        ApiError error;
        error.status  = 400;
        error.param   = "limit";
        error.code    = "invalid_pagination";
        error.message = "limit must be an integer in [1,100]";
        throw ApiException(std::move(error));
    }
    return parsed;
}

Json paginated_input_items(const httplib::Request& request, const std::vector<Json>& stored_items) {
    reject_unknown_query(request, {"after", "include", "limit", "order"});
    bool include_image_urls = false;
    if (request.has_param("include") && !request.get_param_value("include").empty()) {
        if (request.get_param_value("include") != "message.input_image.image_url") {
            invalid_query("only include=message.input_image.image_url is available", "include",
                          "include_not_supported");
        }
        include_image_urls = true;
    }
    const int limit   = parse_limit(request);
    std::string order = request.has_param("order") ? request.get_param_value("order") : "desc";
    if (order != "asc" && order != "desc") {
        ApiError error;
        error.status  = 400;
        error.param   = "order";
        error.code    = "invalid_pagination";
        error.message = "order must be 'asc' or 'desc'";
        throw ApiException(std::move(error));
    }

    std::vector<Json> ordered = stored_items;
    if (order == "desc") { std::reverse(ordered.begin(), ordered.end()); }
    std::size_t begin = 0;
    if (request.has_param("after")) {
        const std::string after = request.get_param_value("after");
        const auto found = std::find_if(ordered.begin(), ordered.end(), [&](const Json& item) {
            return item.contains("id") && item.at("id").is_string() &&
                   item.at("id").get<std::string>() == after;
        });
        if (found == ordered.end()) {
            ApiError error;
            error.status  = 400;
            error.param   = "after";
            error.code    = "invalid_pagination";
            error.message = "after does not identify an input Item in this response";
            throw ApiException(std::move(error));
        }
        begin = static_cast<std::size_t>(std::distance(ordered.begin(), found)) + 1;
    }
    const std::size_t end = std::min(ordered.size(), begin + static_cast<std::size_t>(limit));
    Json data             = Json::array();
    for (std::size_t index = begin; index < end; ++index) {
        Json item = ordered[index];
        if (!include_image_urls) { redact_input_image_urls(item); }
        data.push_back(std::move(item));
    }
    return Json{{"object", "list"},
                {"data", data},
                {"first_id", data.empty() ? Json(nullptr) : data.front().at("id")},
                {"last_id", data.empty() ? Json(nullptr) : data.back().at("id")},
                {"has_more", end < ordered.size()}};
}

class ResponsesEndpoint final : public GenerationEndpoint {
public:
    ResponsesEndpoint(OpenAIResponsesStore& store, std::string id,
                      OpenAIResponsesCreateRequest request, OpenAIResponseContext parent,
                      std::optional<std::string> session_key)
        : store_(store), id_(std::move(id)), request_(std::move(request)),
          parent_(std::move(parent)), session_key_(std::move(session_key)) {}

    ApiError api_error(const ApiError& error) const override { return responses_error(error); }

    void identify(const PreparedRequest&) override { created_ = unix_time_now(); }

    void render(const PreparedRequest& prepared, const GenerationOutcome& outcome) override {
        const OpenAIResponsesRuntimeValues runtime = runtime_values(prepared, &outcome);
        response_.emplace(make_openai_response_object(id_, created_, request_, runtime, outcome));
        keep_input();
    }

    void store(const PreparedRequest& prepared) override {
        BuiltOpenAIResponse& response = finished_ ? finished_->response : *response_;
        commit_stored_response(store_, std::move(storage_), id_, response.body,
                               std::move(response.output_history), prepared.preserve_thinking);
    }

    std::string body() override { return response_->body.dump(); }

    void open_stream(const PreparedRequest& prepared) override {
        keep_input();
        encoder_.emplace(id_, created_, std::move(request_), runtime_values(prepared));
    }

    void write_start(SseTransport& transport) override {
        render_and_write(transport, [&] { return encoder_->start(); });
    }

    void attach(StreamSink& output, SseTransport& transport) override {
        output.on_reasoning = [this, &transport](const std::string& text) {
            render_and_write(transport, [&] { return encoder_->reasoning_delta(text); });
        };
        output.on_content = [this, &transport](const std::string& text) {
            render_and_write(transport, [&] { return encoder_->content_delta(text); });
        };
    }

    std::vector<std::string> finish(const GenerationOutcome& outcome) override {
        finished_.emplace(encoder_->finish(outcome));
        return std::move(finished_->events_before_terminal);
    }

    std::vector<std::string> terminal() override {
        return {encoder_->terminal(finished_->response)};
    }

    void write_stream_error(SseTransport& transport, const ApiError& error) override {
        render_and_write(transport, [&] { return encoder_->failed(error); });
    }

private:
    // The input is stored with the response; a stream moves the request into its encoder.
    void keep_input() {
        storage_.input_turns      = std::move(request_.prompt.input_turns);
        storage_.input_items      = std::move(request_.prompt.input_items);
        storage_.previous_context = std::move(parent_);
        if (session_key_) { storage_.session_key = std::move(*session_key_); }
        storage_.enabled = request_.store;
    }

    OpenAIResponsesStore& store_;
    std::string id_;
    OpenAIResponsesCreateRequest request_;
    OpenAIResponseContext parent_;
    std::optional<std::string> session_key_;
    std::int64_t created_ = 0;
    PendingResponseStorage storage_;
    std::optional<BuiltOpenAIResponse> response_;
    std::optional<OpenAIResponsesEventStream> encoder_;
    std::optional<OpenAIResponsesStreamFinish> finished_;
};

} // namespace

void HttpServer::handle_responses(const httplib::Request& req, httplib::Response& res) {
    OpenAIResponsesCreateRequest request;
    OpenAIResponsesResolvedPrompt resolved;
    std::optional<std::string> client;
    const std::string id = new_openai_response_id();
    try {
        request = parse_openai_responses_create_request(parse_json_body(req, request_limits_),
                                                        request_limits_);
        validate_openai_model(request.prompt.model, public_model_id_);
        client   = parse_client_label(req);
        resolved = resolve_openai_responses_prompt(request.prompt, openai_responses_store_, id,
                                                   request.store);
    } catch (const ApiException& exception) {
        reply_openai_error(req, res, responses_error(exception.error()));
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "openai_responses",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()));
        write_openai_error(res, internal_error(exception.what()));
        return;
    }

    const RequestLogMetadata metadata{
        .model                             = request.prompt.model,
        .client                            = std::move(client),
        .stream                            = request.stream,
        .output_tokens_explicit            = request.requested_max_output_tokens.has_value(),
        .preserve_thinking_semantic_change = resolved.preserve_thinking_semantic_change,
    };
    auto endpoint = std::make_shared<ResponsesEndpoint>(
        openai_responses_store_, id, std::move(request), std::move(resolved.parent),
        std::move(resolved.session_key));
    serve_generation(req, res, std::move(endpoint), "openai_responses", resolved.generation,
                     metadata, {}, std::move(resolved.cache_hints));
}

void HttpServer::handle_response_input_tokens(const httplib::Request& req, httplib::Response& res) {
    try {
        OpenAIResponsesPromptRequest request = parse_openai_responses_input_tokens_request(
            parse_json_body(req, request_limits_), request_limits_);
        validate_openai_model(request.model, public_model_id_);
        OpenAIResponsesResolvedPrompt resolved =
            resolve_openai_responses_prompt(request, openai_responses_store_, std::nullopt, false);
        const int tokens = service_->count_prompt_tokens(
            resolved.generation, [&req] { return client_disconnected(req); });
        res.set_content(make_openai_response_input_tokens_body(tokens), "application/json");
    } catch (const ApiException& exception) {
        reply_openai_error(req, res, responses_error(exception.error()));
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "openai_responses_input_tokens",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()));
        write_openai_error(res, internal_error(exception.what()));
    }
}

void HttpServer::handle_response_get(const httplib::Request& req, httplib::Response& res) {
    try {
        validate_retrieve_query(req);
    } catch (const ApiException& exception) {
        reply_openai_error(req, res, exception.error());
        return;
    }
    const std::string id                                     = path_response_id(req);
    const std::shared_ptr<const StoredOpenAIResponse> stored = openai_responses_store_.get(id);
    if (!stored) {
        reply_openai_error(req, res, response_not_found(id));
        return;
    }
    res.set_content(stored->response.dump(), "application/json");
}

void HttpServer::handle_response_delete(const httplib::Request& req, httplib::Response& res) {
    const std::string id = path_response_id(req);
    if (!openai_responses_store_.erase(id)) {
        reply_openai_error(req, res, response_not_found(id));
        return;
    }
    res.set_content(Json{{"id", id}, {"object", "response.deleted"}, {"deleted", true}}.dump(),
                    "application/json");
}

void HttpServer::handle_response_input_items(const httplib::Request& req, httplib::Response& res) {
    const std::string id                                     = path_response_id(req);
    const std::shared_ptr<const StoredOpenAIResponse> stored = openai_responses_store_.get(id);
    if (!stored) {
        reply_openai_error(req, res, response_not_found(id));
        return;
    }
    try {
        res.set_content(paginated_input_items(req, stored->input_items).dump(), "application/json");
    } catch (const ApiException& exception) { reply_openai_error(req, res, exception.error()); }
}

void HttpServer::handle_response_cancel(const httplib::Request& req, httplib::Response& res) {
    const std::string id = path_response_id(req);
    if (!openai_responses_store_.get(id)) {
        reply_openai_error(req, res, response_not_found(id));
        return;
    }
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.code    = "background_not_supported";
    error.message = "only background responses can be cancelled; NInfer does not support "
                    "background execution";
    reply_openai_error(req, res, error);
}

void HttpServer::handle_response_compact(const httplib::Request& req, httplib::Response& res) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.param   = "context_management";
    error.code    = "compaction_not_supported";
    error.message = "Responses compaction is not supported";
    reply_openai_error(req, res, error);
}

} // namespace ninfer::serve
