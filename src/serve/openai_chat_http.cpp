#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

class ChatEndpoint final : public GenerationEndpoint {
public:
    explicit ChatEndpoint(const OpenAIChatRequest& request)
        : model_(request.model), include_usage_(request.include_usage),
          timings_(request.timings_per_token), progress_(request.return_progress) {}

    void identify(const PreparedRequest&) override {
        identity_ = make_openai_chat_response_identity(model_);
    }

    void render(const PreparedRequest&, const GenerationOutcome& outcome) override {
        body_ = make_chat_completion_response(identity_, outcome);
    }

    std::string body() override { return std::move(body_); }

    void open_stream(const PreparedRequest&) override {
        encoder_.emplace(identity_, include_usage_, timings_, progress_);
    }

    void write_start(SseTransport& transport) override {
        render_and_write(transport, [&] { return encoder_->start(); });
    }

    void attach(StreamSink& output, SseTransport& transport) override {
        output.on_start = [this, &transport](const ninfer::GenerationStart& start) {
            encoder_->note_start(start);
            if (progress_) {
                render_and_write(transport, [&] { return encoder_->initial_prompt_progress(); });
            }
        };
        if (progress_) {
            output.on_progress = [this, &transport](const ninfer::PromptProgress& progress) {
                render_and_write(transport, [&] { return encoder_->prompt_progress(progress); });
            };
        }
        if (timings_) {
            output.on_timing = [this](const ninfer::GenerationTimingObservation& timing) {
                encoder_->note_timing(timing);
            };
        }
        output.on_content = [this, &transport](const std::string& text) {
            render_and_write(transport, [&] { return encoder_->content_delta(text); });
        };
        output.on_reasoning = [this, &transport](const std::string& text) {
            render_and_write(transport, [&] { return encoder_->reasoning_delta(text); });
        };
    }

    std::vector<std::string> finish(const GenerationOutcome& outcome) override {
        return encoder_->finish(outcome);
    }

    void write_stream_error(SseTransport& transport, const ApiError& error) override {
        render_and_write(transport, [&] { return "data: " + make_error_body(error) + "\n\n"; });
    }

private:
    std::string model_;
    bool include_usage_;
    bool timings_;
    bool progress_;
    OpenAIChatResponseIdentity identity_;
    std::string body_;
    std::optional<OpenAIChatStream> encoder_;
};

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    std::optional<std::string> client;
    try {
        request =
            parse_chat_completion_request(parse_json_body(req, request_limits_), request_limits_);
        validate_openai_model(request.model, public_model_id_);
        client = parse_client_label(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    const RequestLogMetadata metadata{.model                  = request.model,
                                      .client                 = std::move(client),
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    const ninfer::GenerationObservationOptions observation{
        .phase_timings   = true,
        .live_timings    = request.stream && request.timings_per_token,
        .prompt_progress = request.stream && request.return_progress,
    };
    serve_generation(req, res, std::make_shared<ChatEndpoint>(request), "openai_chat_completions",
                     request.generation, metadata, observation);
}

} // namespace ninfer::serve
