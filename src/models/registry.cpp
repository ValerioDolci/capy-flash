#include "models/registry.h"

#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::models {
namespace {

// The first row of each Architecture carries its canonical name (architecture_name()). Qwen4Exp
// keeps no turn anchor at tp 1: its prefill passes read the non-resident experts from host RAM.
constexpr std::array kModels{
    RegisteredModel{"Qwen3_5ForCausalLM", "qwen3_5_text", "Qwen3_5MTP", "qwen3_5_vision",
                    Architecture::Qwen3_5, 1U},
    RegisteredModel{"Qwen4ExpForCausalLM", "qwen4_exp_text", "Qwen4ExpMTP", "qwen4_exp_vision",
                    Architecture::Qwen4Exp, 0U},
};

} // namespace

const RegisteredModel& resolve_model(std::string_view architecture, std::string_view model_type) {
    for (const auto& model : kModels) {
        if (model.text == architecture && model.model_type == model_type) { return model; }
    }
    std::string supported;
    for (const auto& model : kModels) {
        supported += (supported.empty() ? "" : ", ") + std::string(model.text) + "/" +
                     std::string(model.model_type);
    }
    throw std::invalid_argument("unsupported architecture " + std::string(architecture) + "/" +
                                std::string(model_type) + " (supported: " + supported + ")");
}

std::string_view mtp_architecture(std::string_view architecture) noexcept {
    for (const auto& model : kModels) {
        if (model.text == architecture) { return model.mtp; }
    }
    return {};
}

bool registered_vision(std::string_view model_type) noexcept {
    for (const auto& model : kModels) {
        if (model.vision == model_type) { return true; }
    }
    return false;
}

std::string_view architecture_name(Architecture architecture) noexcept {
    for (const auto& model : kModels) {
        if (model.architecture == architecture) { return model.text; }
    }
    return {};
}

} // namespace ninfer::models
