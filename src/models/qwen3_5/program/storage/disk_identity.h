#pragma once

// The token identity the disk tier names, hashes and compares (storage/disk_kv.cpp), apart from
// the Program so it can be tested without a device.

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// True when `frontier` falls strictly inside the token spans of a Vision item: a checkpoint there
// would hold half a picture, which neither the disk nor the in-memory cache can resume from.
[[nodiscard]] bool splits_vision(const std::vector<VisionItem>& items, std::size_t frontier);

// The token stream the disk tier names, hashes and compares. A Vision placeholder token says
// nothing about its picture, so every token of an item's spans becomes a negative id derived from
// the item (modality, grid, patches, content digest, timestamps) and the token's offset in it; real
// token ids are non-negative and never equal one. Text tokens are kept as they are, so a text-only
// conversation has the same identity, blocks and files as before media support. The positions
// (M-RoPE, rope_delta) follow from the tokens and the grids, which the identity binds. Returns
// false when the token types and the items disagree or an item crosses the end of `tokens`.
[[nodiscard]] bool disk_identity(std::span<const TokenId> tokens,
                                 std::span<const std::uint8_t> types,
                                 const std::vector<VisionItem>& items, std::vector<TokenId>& out);

} // namespace ninfer::models::qwen3_5::detail
