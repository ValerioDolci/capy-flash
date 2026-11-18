// The Text forward's two instantiations: execution (WorkspaceArena) and startup workspace sizing
// (WorkspaceLayoutBuilder). Its callers include only the declarations (text_forward.h), so the
// member definitions compile here once.
#include "models/qwen3_5/execution/text_forward.h"

#include "models/qwen3_5/execution/text_forward/attention.h"
#include "models/qwen3_5/execution/text_forward/core.h"
#include "models/qwen3_5/execution/text_forward/ffn.h"
#include "models/qwen3_5/execution/text_forward/gdn.h"
#include "models/qwen3_5/execution/text_forward/moe.h"
#include "models/qwen3_5/execution/text_forward/mtp.h"
#include "models/qwen3_5/execution/text_forward/plain.h"
#include "models/qwen3_5/execution/text_forward/projection.h"
#include "models/qwen3_5/execution/text_forward/qsa.h"
#include "models/qwen3_5/execution/text_forward/residual.h"

#include <atomic>

namespace ninfer::models::qwen3_5::execution {

namespace {
std::atomic<bool> g_prefill_moe_int8{false};
} // namespace

void qwen4_force_prefill_moe_int8() { g_prefill_moe_int8.store(true, std::memory_order_relaxed); }

bool qwen4_prefill_moe_int8_forced() { return g_prefill_moe_int8.load(std::memory_order_relaxed); }

template class TextForward<WorkspaceArena>;
template class TextForward<WorkspaceLayoutBuilder>;

} // namespace ninfer::models::qwen3_5::execution
