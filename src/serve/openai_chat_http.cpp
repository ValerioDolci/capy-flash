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

    // The assistant-role chunk goes out with the first delta, or with the closing chunks when there
    // is none, as OpenAI, vLLM and llama.cpp stream it: a client that times the first chunk (a
    // benchmark's prompt time) then times the prompt, not the request's admission. A long prompt
    // keeps the connection through the transport's keep-alive comments. With prompt progress the
    // role chunk opens the stream, before the progress chunks.
    void write_start(SseTransport& transport) override {
        if (progress_) {
            render_and_write(transport, [&] { return encoder_->start(); });
            return;
        }
        try {
            pending_start_ = encoder_->start();
        } catch (const std::exception& exception) { throw ResponseRenderFailure(exception.what()); }
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
            write_pending_start(transport);
            render_and_write(transport, [&] { return encoder_->content_delta(text); });
        };
        output.on_reasoning = [this, &transport](const std::string& text) {
            write_pending_start(transport);
            render_and_write(transport, [&] { return encoder_->reasoning_delta(text); });
        };
    }

    std::vector<std::string> finish(const GenerationOutcome& outcome) override {
        std::vector<std::string> closing = encoder_->finish(outcome);
        if (!pending_start_.empty()) {
            closing.insert(closing.begin(), std::exchange(pending_start_, std::string{}));
        }
        return closing;
    }

    void write_stream_error(SseTransport& transport, const ApiError& error) override {
        write_pending_start(transport);
        render_and_write(transport, [&] { return "data: " + make_error_body(error) + "\n\n"; });
    }

private:
    void write_pending_start(SseTransport& transport) {
        if (pending_start_.empty()) { return; }
        transport.write(std::exchange(pending_start_, std::string{}));
    }

    std::string model_;
    bool include_usage_;
    bool timings_;
    bool progress_;
    OpenAIChatResponseIdentity identity_;
    std::string body_;
    std::optional<OpenAIChatStream> encoder_;
    std::string pending_start_; // the role chunk, until the first delta (write_start)
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
        reply_openai_error(req, res, exception.error());
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
