#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

class MessagesEndpoint final : public GenerationEndpoint {
public:
    MessagesEndpoint(std::string request_id, const AnthropicMessagesRequest& request)
        : request_id_(std::move(request_id)), model_(request.model),
          hide_thinking_(request.hide_thinking) {}

    ApiError api_error(const ApiError& error) const override {
        return normalize_anthropic_error(error);
    }

    // The body reports every 5xx as api_error; the request logs keep this type.
    ApiError internal_error(const std::exception& exception) const override {
        ApiError error = serve::internal_error(exception.what());
        error.type     = "internal_error";
        return error;
    }

    void write_error(httplib::Response& response, const ApiError& error) const override {
        write_anthropic_error(response, error, request_id_);
    }

    // An Anthropic message has no cancelled stop reason.
    bool renders_cancelled() const override { return false; }

    void identify(const PreparedRequest& prepared) override {
        identity_     = make_anthropic_response_identity(request_id_, model_);
        input_tokens_ = prepared.prompt_tokens;
    }

    void render(const PreparedRequest&, const GenerationOutcome& outcome) override {
        body_ = make_anthropic_messages_response(identity_, outcome, hide_thinking_);
    }

    std::string body() override { return std::move(body_); }

    void open_stream(const PreparedRequest&) override {
        encoder_.emplace(identity_, input_tokens_, hide_thinking_);
    }

    void attach(StreamSink& output, SseTransport& transport) override {
        output.on_start = [this, &transport](const ninfer::GenerationStart& start) {
            render_and_write(transport, [&] { return encoder_->start(start); });
        };
        output.on_reasoning = [this, &transport](const std::string& text) {
            render_and_write(transport, [&] { return encoder_->reasoning_delta(text); });
        };
        output.on_content = [this, &transport](const std::string& text) {
            render_and_write(transport, [&] { return encoder_->content_delta(text); });
        };
    }

    std::vector<std::string> finish(const GenerationOutcome& outcome) override {
        return encoder_->finish(outcome);
    }

    void write_stream_error(SseTransport& transport, const ApiError& error) override {
        if (!encoder_->started()) {
            render_and_write(transport, [&] { return encoder_->start(); });
        }
        render_and_write(transport, [&] { return encoder_->error(error); });
    }

private:
    std::string request_id_;
    std::string model_;
    bool hide_thinking_;
    AnthropicResponseIdentity identity_;
    int input_tokens_ = 0;
    std::string body_;
    std::optional<AnthropicMessagesStream> encoder_;
};

} // namespace

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);
    try {
        const AnthropicCountTokensRequest request = parse_anthropic_count_tokens_request(
            parse_json_body(req, request_limits_), request_limits_);
        const int input_tokens = service_->count_prompt_tokens(
            request.generation, [&req] { return client_disconnected(req); });
        res.set_content(make_anthropic_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_count_tokens",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        write_anthropic_error(res, internal_error(exception.what()), request_id);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);

    AnthropicMessagesRequest request;
    std::optional<std::string> client;
    try {
        request = parse_anthropic_messages_request(parse_json_body(req, request_limits_),
                                                   request_limits_);
        client  = parse_client_label(req);
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_messages",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        write_anthropic_error(res, internal_error(exception.what()), request_id);
        return;
    }

    const RequestLogMetadata metadata{.model                  = request.model,
                                      .client                 = std::move(client),
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    serve_generation(req, res, std::make_shared<MessagesEndpoint>(request_id, request),
                     "anthropic_messages", request.generation, metadata);
}

} // namespace ninfer::serve
