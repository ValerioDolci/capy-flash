#pragma once

#include <cstdint>
#include <string_view>

namespace ninfer::models {

enum class Architecture { Qwen3_5, Qwen4Exp };

// One registered checkpoint: the names its converted configs carry and the implementation they
// select. A model with the mathematics of an existing Architecture is one more row; nothing outside
// the registry compares these names.
struct RegisteredModel {
    std::string_view text;       // Text component architecture
    std::string_view model_type; // Text component model_type
    std::string_view mtp;        // architecture of its MTP component
    std::string_view vision;     // model_type of its Vision component
    Architecture architecture;
    // Serving default of --turn-anchors at tp 1 (tp 2 serves 0 for every model).
    std::uint32_t turn_anchors;
};

[[nodiscard]] const RegisteredModel& resolve_model(std::string_view architecture,
                                                   std::string_view model_type);
// The MTP component architecture that pairs with a registered Text architecture name (empty for
// an unregistered one).
[[nodiscard]] std::string_view mtp_architecture(std::string_view architecture) noexcept;
// Whether a Vision component model_type belongs to a registered model.
[[nodiscard]] bool registered_vision(std::string_view model_type) noexcept;
// The canonical Text architecture name of an implementation.
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
