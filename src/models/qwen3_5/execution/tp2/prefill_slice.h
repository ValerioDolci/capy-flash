// Tensor-parallel prefill slicing: the width and iteration of the exchange slices the pipelined
// prefill (text_tp2.inc, ops::PipelinedExchange) runs through, and the column window it applies
// them to. Shared by the schedule (this directory) and the startup status (program/), so the
// environment variable is read and interpreted in exactly one place.

#ifndef NINFER_MODELS_QWEN3_5_EXECUTION_TP2_PREFILL_SLICE_H
#define NINFER_MODELS_QWEN3_5_EXECUTION_TP2_PREFILL_SLICE_H

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::execution {

// The token width of one prefill exchange slice: NINFER_TP_PREFILL_SLICE, a positive token count.
// Unset or empty takes the default (1024, the width measured in the upstream prefill-pipeline
// experiments), and 0 turns the slicing off. The value is read once per process.
inline std::int32_t prefill_exchange_slice_tokens() {
    static const std::int32_t tokens = [] {
        const char* raw = std::getenv("NINFER_TP_PREFILL_SLICE");
        if (raw == nullptr || raw[0] == '\0') { return 1024; }
        char* end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
            throw std::invalid_argument("NINFER_TP_PREFILL_SLICE must be 0 (off) or a positive "
                                        "token count, not " +
                                        std::string(raw));
        }
        return static_cast<std::int32_t>(parsed);
    }();
    return tokens;
}

// The slice for a chunk of `columns` tokens: the configured width, halved (down to 128) until the
// chunk holds at least four slices, so short prompts and a prompt's last chunk pipeline as well;
// 0 when slicing is off or the chunk is shorter than two slices.
inline std::int32_t exchange_slice_for(std::int32_t columns) {
    std::int32_t slice = prefill_exchange_slice_tokens();
    if (slice == 0 || columns < 256) { return 0; }
    while (slice > 128 && slice * 4 > columns) {
        slice /= 2;
    }
    return slice;
}

// Iterates a chunk of `columns` tokens in `exchange_slice_for(columns)` windows.
template <class Body>
void for_each_exchange_slice(std::int32_t columns, Body&& body) {
    const std::int32_t slice = exchange_slice_for(columns);
    if (slice == 0) {
        throw std::invalid_argument("for_each_exchange_slice: called with slicing disabled");
    }
    for (std::int32_t first = 0; first < columns; first += slice) {
        body(first, std::min(slice, columns - first));
    }
}

// The [first, first+count) token window of a rank pair's column-oriented tensor.
inline std::array<Tensor, 2> column_slice(const std::array<Tensor, 2>& t, std::int32_t first,
                                           std::int32_t count) {
    return {t[0].slice(1, first, count), t[1].slice(1, first, count)};
}

} // namespace ninfer::models::qwen3_5::execution

#endif
