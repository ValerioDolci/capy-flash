// Token identity of the disk tier (disk_identity.h).
#include "models/qwen3_5/program/storage/disk_identity.h"

#include "core/disk_kv_cache.h"

namespace ninfer::models::qwen3_5::detail {
namespace {

constexpr std::uint64_t kSeedV = 0x7669736e6f6eU;

} // namespace

bool splits_vision(const std::vector<VisionItem>& items, std::size_t frontier) {
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return true; }
        const std::size_t begin = item.token_spans.front().begin;
        const std::size_t end   = item.token_spans.back().begin + item.token_spans.back().count;
        if (begin < frontier && frontier < end) { return true; }
    }
    return false;
}

bool disk_identity(std::span<const TokenId> tokens, std::span<const std::uint8_t> types,
                   const std::vector<VisionItem>& items, std::vector<TokenId>& out) {
    out.assign(tokens.begin(), tokens.end());
    std::vector<std::uint8_t> covered(tokens.size(), 0);
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        if (item.token_spans.front().begin >= tokens.size()) { continue; }
        Xxh64Stream base(kSeedV);
        const std::uint64_t fields[] = {static_cast<std::uint64_t>(item.modality),
                                        static_cast<std::uint64_t>(item.grid.temporal),
                                        static_cast<std::uint64_t>(item.grid.height),
                                        static_cast<std::uint64_t>(item.grid.width),
                                        item.patch_begin,
                                        item.patch_count,
                                        item.timestamps.size(),
                                        item.token_spans.size()};
        base.update(fields, sizeof(fields));
        base.update(item.content_digest.data(), item.content_digest.size());
        if (!item.timestamps.empty()) {
            base.update(item.timestamps.data(), item.timestamps.size() * sizeof(double));
        }
        const std::uint64_t item_digest = base.digest();
        std::uint64_t offset            = 0;
        for (const TokenSpan& span : item.token_spans) {
            if (span.begin > tokens.size() || span.count > tokens.size() - span.begin) {
                return false;
            }
            for (std::size_t k = 0; k < span.count; ++k, ++offset) {
                const std::size_t index = span.begin + k;
                if (covered[index] != 0 || index >= types.size() || types[index] == 0) {
                    return false;
                }
                covered[index]              = 1;
                const std::uint64_t pair[2] = {item_digest, offset};
                const std::uint64_t value   = xxh64(pair, sizeof(pair), kSeedV) & 0x3fffffffU;
                out[index]                  = -1 - static_cast<TokenId>(value);
            }
        }
    }
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint8_t type = index < types.size() ? types[index] : 0;
        if ((type != 0) != (covered[index] != 0)) { return false; }
    }
    return true;
}

} // namespace ninfer::models::qwen3_5::detail
