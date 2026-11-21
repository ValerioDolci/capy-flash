#include "models/inkling2/execution/tp.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::execution {

TextConfig shard_text_config(const TextConfig& config, int width) {
    if (width < 1) { throw std::invalid_argument("tensor-parallel width must be positive"); }
    const auto w   = static_cast<std::uint32_t>(width);
    const auto cut = [&](std::uint32_t value, const char* what) {
        if (value % w) {
            throw std::invalid_argument(std::string(what) + " does not divide by " +
                                        std::to_string(width));
        }
        return value / w;
    };
    TextConfig out          = config;
    out.num_attention_heads = cut(config.num_attention_heads, "query heads");
    out.num_key_value_heads = cut(config.num_key_value_heads, "KV heads");
    out.unpadded_vocab_size = cut(config.unpadded_vocab_size, "output head rows");
    (void)cut(config.num_experts, "experts");
    return out;
}

} // namespace ninfer::models::inkling2::execution
