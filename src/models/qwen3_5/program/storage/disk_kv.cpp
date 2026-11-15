// Disk tier of the context cache: saving retained conversations to disk and
// resuming a root prefill from a saved checkpoint. Storage, index, key, LRU and writer thread live
// in core/disk_kv_cache.h; this file decides what is stored (format 2, content addressed):
//   KV block b = Text KV pages [4b, 4b+4) and MTP KV pages [4b, 4b+4) (tokens [256b, 256b+256)),
//     both ranks' halves at tp 2, Host page layout; id = two chained xxh64 over the tokens
//     [256b, 256b+257) from the previous block's id (token 256b+256 included: the MTP KV of the
//     block's last position depends on it). Written once, shared by every save with that prefix.
//   StateImage at checkpoint F (GDN state + the hidden of token F-1): id = two xxh64 of [0, F).
//   Manifest (one per conversation): tokens [0, F_max), blob references, checkpoints, and the
//     tail pages after the last whole block.
// A checkpoint at F needs Text KV [0, F), MTP KV [0, F-1) and its StateImage (context.cpp).
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"
#include "core/device_scope.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

using Clock = std::chrono::steady_clock;

// Shorter conversations are cheaper to prefill (about 1 s at tp 2) than to keep: each saved
// checkpoint costs a 147 MiB StateImage on disk.
constexpr std::uint32_t kDiskMinimumTokens = 4096;
constexpr std::size_t kDiskStagingBytes    = 64ULL << 20U;
constexpr std::size_t kDiskStagingBuffers  = 3;
constexpr std::size_t kDiskReadBytes       = 64ULL << 20U;
constexpr std::uint32_t kDiskEndpoint      = 1;
constexpr std::uint32_t kDiskRewrite       = 2;
constexpr std::uint32_t kDiskLongAnchor    = 3;
// A KV block: 4 pages of 64 tokens (LMCache's default chunk, SGLang's page granularity).
constexpr std::uint32_t kDiskBlockPages  = 4;
constexpr std::uint32_t kDiskBlockTokens = kDiskBlockPages * 64U;
// The endpoint is kept only when it is this far past the other checkpoints (a long reply);
// otherwise resuming at the rewrite checkpoint and recomputing the reply is cheap.
constexpr std::uint32_t kDiskEndpointMinimumGain = 2048;
constexpr std::uint64_t kSeedA = 0x6e6964b10cU, kSeedB = 0x9f2c55a1e37dU, kSeedS = 0x51a7e1U;

std::string layout_key(const HostKVPageLayout& layout) {
    std::string out = std::to_string(layout.page_stride) + "/" + std::to_string(layout.replicas) +
                      "/" + std::to_string(layout.replica_stride);
    for (const HostKVPlaneLayout& plane : layout.planes) {
        out += "/" + std::to_string(plane.offset) + "," + std::to_string(plane.page_payload_bytes) +
               "," + std::to_string(plane.head_payload_bytes);
    }
    return out;
}

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

constexpr double kMiB = 1024.0 * 1024.0;

struct DiskCheckpointPlan {
    std::uint32_t frontier = 0;
    std::uint32_t kind     = 0;
    StateImageHandle state;
};

std::uint32_t section_of(const DiskKVRecordHeader& header, DiskKVSectionKind kind) {
    for (std::uint32_t index = 0; index < header.sections.size(); ++index) {
        if (header.sections[index].kind == static_cast<std::uint32_t>(kind)) { return index; }
    }
    return static_cast<std::uint32_t>(header.sections.size());
}

} // namespace

void ProgramImpl::disk_synchronize() const {
    {
        const ScopedCurrentDevice scope(device.device);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (peer) {
        const ScopedCurrentDevice scope(peer->device.device);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void ProgramImpl::initialize_disk_cache() {
    if (context_cache.disk_directory.empty() || context_cache.disk_capacity_bytes == 0 ||
        causal_scoring) {
        return;
    }
    if (speculative_backend != SpeculativeBackend::None &&
        speculative_backend != SpeculativeBackend::Mtp) {
        std::fprintf(stderr, "[disk-kv] off: the disk tier supports no speculation or MTP, not "
                             "DFlash/DFlash2\n");
        return;
    }
    if ((speculative_backend == SpeculativeBackend::Mtp) != (backend_kv_pages != nullptr)) {
        std::fprintf(stderr, "[disk-kv] off: MTP KV pool missing\n");
        return;
    }
    const DeviceKVPagePool& text_pool = decoder->text_kv.page_pool();
    const HostKVPageLayout text_layout =
        plan_host_kv_page_layout(text_pool.geometry(), text_pool.host_replicas());
    std::string backend_key = "none";
    if (const qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
        backend_key = layout_key(plan_host_kv_page_layout(backend->page_pool().geometry(),
                                                          backend->page_pool().host_replicas()));
    }
    const StateImageHostLayout& state_layout = state_images->host_layout();
    std::string key =
        "format=" + std::to_string(kDiskKVFormatVersion) + ";" + context_cache.disk_identity +
        ";tp=" + std::to_string(peer ? 2 : 1) +
        ";kv=" + std::to_string(static_cast<int>(kv_storage)) +
        ";spec=" + std::to_string(static_cast<int>(speculative_backend)) +
        ";head=" + std::to_string(static_cast<int>(proposal_head)) +
        ";qwen4=" + std::to_string(qwen4_runtime() != nullptr ? 1 : 0) +
        ";page=" + std::to_string(kPagedKVPageSize) + ";text=" + layout_key(text_layout) +
        ";mtp=" + backend_key + ";state=" + std::to_string(state_layout.image_bytes) + "/" +
        std::to_string(state_layout.peer_offset) + "/" + std::to_string(state_layout.peer_bytes);

    DiskKVCache::Options options;
    options.directory       = context_cache.disk_directory;
    options.capacity_bytes  = context_cache.disk_capacity_bytes;
    options.min_free_bytes  = context_cache.disk_min_free_bytes;
    options.key             = std::move(key);
    options.staging_buffers = kDiskStagingBuffers;
    options.staging_bytes   = kDiskStagingBytes;
    options.allocate        = [](std::size_t bytes) -> void* {
        void* data = nullptr;
        return cudaHostAlloc(&data, bytes, cudaHostAllocPortable) == cudaSuccess ? data : nullptr;
    };
    options.release = [](void* data) { (void)cudaFreeHost(data); };
    disk_kv         = std::make_unique<DiskKVCache>(std::move(options));
    // read_region reads whole 4 KiB pages (O_DIRECT): at one device the StateImage is not a
    // multiple of 4096 bytes (27B 16 GB profile: 153 954 304), and an exact-size buffer made every
    // resume fail ("conversation ... unreadable (blob ...)").
    disk_state_scratch.emplace((state_layout.image_bytes + kDiskKVAlignment - 1U) /
                               kDiskKVAlignment * kDiskKVAlignment);
    disk_read_buffer.emplace(2U * kDiskReadBytes); // two read-ahead chunks
    disk_saved.assign(continuation_capacity, DiskSaved{});
    const DiskKVCacheStats stats = disk_kv->stats();
    const double gib             = kMiB * 1024.0;
    std::fprintf(
        stderr,
        "[disk-kv] %s: %llu conversation(s) of this configuration, %llu of another "
        "(ignored), %llu blob(s); %.2f of %.2f GiB used; file system %.1f GiB free, "
        "margin %.1f GiB; removed %llu broken, %llu orphan, %llu temporary file(s); KV page "
        "%zu B, block %u tokens, StateImage %zu B\n",
        context_cache.disk_directory.c_str(), static_cast<unsigned long long>(stats.records),
        static_cast<unsigned long long>(stats.foreign_records),
        static_cast<unsigned long long>(stats.blobs), static_cast<double>(stats.bytes) / gib,
        static_cast<double>(stats.capacity_bytes) / gib,
        static_cast<double>(stats.fs_free_bytes) / gib,
        static_cast<double>(stats.min_free_bytes) / gib,
        static_cast<unsigned long long>(stats.removed_broken),
        static_cast<unsigned long long>(stats.removed_orphans),
        static_cast<unsigned long long>(stats.removed_temp), text_layout.page_stride,
        kDiskBlockTokens, state_layout.image_bytes);
    if (stats.capacity_bytes + stats.min_free_bytes > stats.fs_free_bytes + stats.bytes) {
        std::fprintf(
            stderr,
            "[disk-kv] warning: --disk-cache-gib %.1f is more than the space the cache can "
            "use (%.1f GiB free + %.1f GiB already in the cache - %.1f GiB margin): the "
            "free-space margin will bound it first\n",
            static_cast<double>(stats.capacity_bytes) / gib,
            static_cast<double>(stats.fs_free_bytes) / gib, static_cast<double>(stats.bytes) / gib,
            static_cast<double>(stats.min_free_bytes) / gib);
    }
}

namespace {

// The checkpoints of a retained conversation the disk can hold, and their largest frontier.
template <class Program>
std::uint32_t disk_checkpoints(const Program& program, const SequenceState& sequence,
                               std::vector<DiskCheckpointPlan>& out) {
    out.clear();
    if (!sequence.kv || sequence.rope_delta != 0 || !sequence.prefix_identity.text_only()) {
        return 0;
    }
    const bool mtp = program.speculative_backend == SpeculativeBackend::Mtp;
    const auto add = [&](std::uint32_t frontier, std::uint32_t kind, StateImageHandle state) {
        if (frontier == 0 || frontier > sequence.text_kv_valid ||
            frontier > sequence.ledger.size()) {
            return;
        }
        if (mtp && (!sequence.kv->backend || frontier - 1U > sequence.mtp_kv_valid)) { return; }
        if (!program.state_store->valid(state) ||
            program.state_store->residency(state) == StateReplicaResidency::None) {
            return;
        }
        for (const DiskCheckpointPlan& existing : out) {
            if (existing.frontier == frontier) { return; }
        }
        out.push_back({frontier, kind, state});
    };
    if (sequence.endpoint_valid && sequence.state.read == sequence.state.write &&
        !sequence.state.fork_pending) {
        add(sequence.execution_frontier, kDiskEndpoint, sequence.state.read);
    }
    if (sequence.rewrite_checkpoint.valid && sequence.rewrite_state) {
        add(sequence.rewrite_checkpoint.frontier, kDiskRewrite, *sequence.rewrite_state);
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        add(anchor.frontier, kDiskLongAnchor, anchor.state);
    }
    std::uint32_t other = 0;
    for (const DiskCheckpointPlan& plan : out) {
        if (plan.kind != kDiskEndpoint) { other = std::max(other, plan.frontier); }
    }
    if (other != 0) {
        std::erase_if(out, [&](const DiskCheckpointPlan& plan) {
            return plan.kind == kDiskEndpoint && plan.frontier < other + kDiskEndpointMinimumGain;
        });
    }
    std::uint32_t frontier = 0;
    for (const DiskCheckpointPlan& plan : out) { frontier = std::max(frontier, plan.frontier); }
    return frontier;
}

// Content ids (see the file comment).
std::vector<DiskKVBlobId> disk_block_ids(std::span<const TokenId> tokens, std::uint32_t blocks,
                                         std::uint64_t key) {
    std::vector<DiskKVBlobId> ids(blocks);
    DiskKVBlobId previous{key ^ kSeedA, key ^ kSeedB};
    for (std::uint32_t block = 0; block < blocks; ++block) {
        const TokenId* first = tokens.data() + static_cast<std::size_t>(block) * kDiskBlockTokens;
        const std::size_t bytes = (kDiskBlockTokens + 1U) * sizeof(TokenId);
        previous =
            DiskKVBlobId{xxh64(first, bytes, previous.a), xxh64(first, bytes, previous.b ^ kSeedB)};
        ids[block] = previous;
    }
    return ids;
}

DiskKVBlobId disk_state_id(std::span<const TokenId> tokens, std::uint32_t frontier,
                           std::uint64_t key) {
    const std::size_t bytes = static_cast<std::size_t>(frontier) * sizeof(TokenId);
    return DiskKVBlobId{xxh64(tokens.data(), bytes, key ^ kSeedS),
                        xxh64(tokens.data(), bytes, (key ^ kSeedS) * 0x9E3779B97F4A7C15ULL)};
}

} // namespace

DiskCacheSave ProgramImpl::disk_cache_save_one(const std::function<bool()>& abort) {
    if (!disk_kv || pending_transaction_) { return DiskCacheSave::NothingToSave; }
    std::vector<DiskCheckpointPlan> plans;
    std::uint32_t best          = continuation_capacity;
    std::uint32_t best_frontier = 0;
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Catalogued) { continue; }
        const SequenceState& sequence = continuation_states[index];
        const std::uint32_t frontier  = disk_checkpoints(*this, sequence, plans);
        if (frontier < kDiskMinimumTokens) { continue; }
        const DiskSaved& saved = disk_saved[index];
        if (saved.generation == continuation_slots[index].generation &&
            saved.frontier == frontier &&
            saved.digest == xxh64(sequence.ledger.data(), frontier * sizeof(TokenId))) {
            continue;
        }
        if (frontier > best_frontier) {
            best          = index;
            best_frontier = frontier;
        }
    }
    if (best == continuation_capacity) { return DiskCacheSave::NothingToSave; }
    DiskCacheSave result = DiskCacheSave::Failed;
    try {
        result = disk_cache_save(best, abort);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[disk-kv] save failed: %s\n", error.what());
        // A CUDA error is sticky and fails the next execution unit anyway; anything else leaves
        // the Program untouched (the save only reads it).
        result = DiskCacheSave::Failed;
    }
    if (result != DiskCacheSave::Aborted) {
        const SequenceState& sequence = continuation_states[best];
        disk_saved[best]              = DiskSaved{
                         .generation = continuation_slots[best].generation,
                         .frontier   = best_frontier,
                         .digest     = xxh64(sequence.ledger.data(), best_frontier * sizeof(TokenId)),
        };
    }
    return result;
}

namespace {

// Fills the staging buffers of one record in submission order: bytes from host memory, KV pages
// from their current Host replica or by D2H from the Device. A buffer is synchronized with both
// devices before it is handed to the writer thread.
class DiskProducer {
public:
    DiskProducer(ProgramImpl& program, DiskKVCache::Writer& writer,
                 const std::function<bool()>& abort)
        : program_(program), writer_(writer), abort_(abort) {}

    [[nodiscard]] bool aborted() const noexcept { return aborted_; }

    void submit() {
        if (!buffer_) { return; }
        if (gpu_pending_) {
            program_.disk_synchronize();
            gpu_pending_ = false;
        }
        if (fill_ != 0) {
            writer_.submit(*buffer_, fill_);
        } else {
            writer_.give_back(*buffer_);
        }
        buffer_.reset();
        fill_ = 0;
    }

    void cancel() {
        if (buffer_) {
            if (gpu_pending_) { program_.disk_synchronize(); }
            writer_.give_back(*buffer_);
            buffer_.reset();
        }
        writer_.abort();
    }

    bool put(const std::byte* source, std::size_t length) {
        while (length != 0) {
            if (!room(1)) { return false; }
            const std::size_t take = std::min(length, buffer_->bytes - fill_);
            std::memcpy(buffer_->data + fill_, source, take);
            fill_ += take;
            source += take;
            length -= take;
        }
        return true;
    }

    bool put_pages(const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                   KVAddressSpaceHandle address, std::uint32_t first, std::uint32_t count,
                   const DeviceKVPagePool& pool, const HostKVPageLayout& layout) {
        const std::size_t stride = layout.page_stride;
        std::vector<DeviceKVPageHandle> batch;
        const auto host_bytes = [&](LogicalKVPageHandle logical) -> const std::byte* {
            if (program_.host_kv_extents == nullptr || !pages.host_replica_current(logical)) {
                return nullptr;
            }
            const HostKVPageReplica& replica = pages.host_replica(logical);
            const HostKVAllocationConstView view =
                program_.host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
            if (view.layout().page_stride != stride || view.layout().replicas != layout.replicas) {
                return nullptr;
            }
            return view.data();
        };
        std::uint32_t page      = first;
        const std::uint32_t end = first + count;
        while (page < end) {
            if (!room(stride)) { return false; }
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (const std::byte* host = host_bytes(logical); host != nullptr) {
                std::memcpy(buffer_->data + fill_, host, stride);
                fill_ += stride;
                ++page;
                continue;
            }
            batch.clear();
            const std::size_t space = (buffer_->bytes - fill_) / stride;
            while (page < end && batch.size() < space) {
                const LogicalKVPageHandle next = addresses.logical_page(address, page);
                if (host_bytes(next) != nullptr) { break; }
                if (!pages.device_resident(next)) {
                    throw std::logic_error("disk save: a KV page has no current replica");
                }
                batch.push_back(pages.physical(next));
                ++page;
            }
            pool.copy_to_host_buffer(batch, buffer_->data + fill_, layout, program_.device.stream);
            fill_ += batch.size() * stride;
            gpu_pending_ = true;
        }
        return true;
    }

private:
    bool room(std::size_t need) {
        if (buffer_ && buffer_->bytes - fill_ >= need) { return true; }
        submit();
        if (abort_ && abort_()) {
            aborted_ = true;
            return false;
        }
        buffer_ = writer_.acquire(abort_);
        if (!buffer_) {
            aborted_ = true;
            return false;
        }
        fill_ = 0;
        return true;
    }

    ProgramImpl& program_;
    DiskKVCache::Writer& writer_;
    const std::function<bool()>& abort_;
    std::optional<DiskKVBuffer> buffer_;
    std::size_t fill_ = 0;
    bool gpu_pending_ = false;
    bool aborted_     = false;
};

} // namespace

DiskCacheSave ProgramImpl::disk_cache_save(std::uint32_t slot, const std::function<bool()>& abort) {
    const auto started            = Clock::now();
    const SequenceState& sequence = continuation_states[slot];
    std::vector<DiskCheckpointPlan> plans;
    const std::uint32_t frontier = disk_checkpoints(*this, sequence, plans);
    if (frontier == 0) { return DiskCacheSave::Failed; }
    std::sort(plans.begin(), plans.end(),
              [](const DiskCheckpointPlan& a, const DiskCheckpointPlan& b) {
                  return a.frontier < b.frontier;
              });
    const bool mtp                    = speculative_backend == SpeculativeBackend::Mtp;
    const DeviceKVPagePool& text_pool = decoder->text_kv.page_pool();
    const HostKVPageLayout text_layout =
        plan_host_kv_page_layout(text_pool.geometry(), text_pool.host_replicas());
    const DeviceKVPagePool* backend_pool =
        mtp && backend_kv_cache() != nullptr ? &backend_kv_cache()->page_pool() : nullptr;
    HostKVPageLayout backend_layout;
    if (backend_pool != nullptr) {
        backend_layout =
            plan_host_kv_page_layout(backend_pool->geometry(), backend_pool->host_replicas());
    }
    const std::uint32_t text_pages    = kv_pages_for_frontier(frontier);
    const std::uint32_t backend_pages = mtp ? kv_pages_for_frontier(frontier - 1U) : 0U;
    if (text_pages > text_kv_addresses->mapped_pages(sequence.kv->text) ||
        (backend_pages != 0 &&
         (!sequence.kv->backend ||
          backend_pages > backend_kv_addresses->mapped_pages(*sequence.kv->backend)))) {
        return DiskCacheSave::Failed;
    }
    const std::uint32_t blocks      = (frontier - 1U) / kDiskBlockTokens;
    const std::uint32_t block_pages = blocks * kDiskBlockPages;
    const std::uint32_t tail_text   = text_pages - block_pages;
    const std::uint32_t tail_backend =
        backend_pages > block_pages ? backend_pages - block_pages : 0U;
    const StateImageHostLayout& state_layout = state_images->host_layout();
    const std::size_t image_bytes            = state_layout.image_bytes;
    const std::span<const TokenId> ledger(sequence.ledger.data(), frontier);

    std::vector<DiskKVBlobId> ids = disk_block_ids(ledger, blocks, disk_kv->key_hash());
    for (const DiskCheckpointPlan& plan : plans) {
        ids.push_back(disk_state_id(ledger, plan.frontier, disk_kv->key_hash()));
    }
    std::vector<bool> present;
    const std::uint64_t session = disk_kv->begin_session(ids, present);
    // One new StateImage per save at most: the latest checkpoint's. Earlier checkpoints (long
    // anchors) are kept only if their StateImage is already on disk (no write): writing
    // them all costs an agent turn several StateImages (147 MiB each) for a few KV blocks.
    for (std::size_t index = plans.size(); index-- > 1;) {
        const std::size_t earlier = index - 1;
        if (present[blocks + earlier]) { continue; }
        plans.erase(plans.begin() + static_cast<std::ptrdiff_t>(earlier));
        ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(blocks + earlier));
        present.erase(present.begin() + static_cast<std::ptrdiff_t>(blocks + earlier));
    }
    std::uint64_t new_bytes  = 0;
    std::uint32_t new_blocks = 0, new_states = 0;
    const auto fail = [&](DiskCacheSave result) {
        disk_kv->abort_session(session);
        return result;
    };
    const auto refused = [&](const char* what) {
        const DiskKVCacheStats stats = disk_kv->stats();
        std::fprintf(stderr,
                     "[disk-kv] save skipped: no room for a %s of a %u-token conversation (cache "
                     "%.2f of %.2f GiB, file system %.1f GiB free, margin %.1f GiB)\n",
                     what, frontier, static_cast<double>(stats.bytes) / (kMiB * 1024.0),
                     static_cast<double>(stats.capacity_bytes) / (kMiB * 1024.0),
                     static_cast<double>(stats.fs_free_bytes) / (kMiB * 1024.0),
                     static_cast<double>(stats.min_free_bytes) / (kMiB * 1024.0));
        return fail(DiskCacheSave::Failed);
    };

    // New KV blocks.
    for (std::uint32_t block = 0; block < blocks; ++block) {
        if (present[block]) { continue; }
        DiskKVRecordSpec spec;
        spec.kind    = DiskKVRecordKind::Blob;
        spec.blob    = ids[block];
        spec.session = session;
        spec.tokens.assign(ledger.begin() + static_cast<std::ptrdiff_t>(block) * kDiskBlockTokens,
                           ledger.begin() + static_cast<std::ptrdiff_t>(block) * kDiskBlockTokens +
                               kDiskBlockTokens + 1);
        spec.sections.push_back(
            {.kind  = static_cast<std::uint32_t>(DiskKVSectionKind::TextKV),
             .bytes = static_cast<std::uint64_t>(kDiskBlockPages) * text_layout.page_stride});
        spec.sections.push_back(
            {.kind  = static_cast<std::uint32_t>(DiskKVSectionKind::BackendKV),
             .bytes = mtp ? static_cast<std::uint64_t>(kDiskBlockPages) * backend_layout.page_stride
                          : 0U});
        spec.meta = {text_layout.page_stride, backend_layout.page_stride, block, 0, 0, 0, 0, 0};
        std::optional<DiskKVCache::Writer> writer = disk_kv->begin_record(std::move(spec));
        if (!writer) { return refused("KV block"); }
        DiskProducer producer(*this, *writer, abort);
        bool ok =
            producer.put_pages(*text_kv_addresses, *text_kv_pages, sequence.kv->text,
                               block * kDiskBlockPages, kDiskBlockPages, text_pool, text_layout);
        if (ok && mtp) {
            ok = producer.put_pages(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                                    block * kDiskBlockPages, kDiskBlockPages, *backend_pool,
                                    backend_layout);
        }
        if (!ok) {
            producer.cancel();
            return fail(producer.aborted() ? DiskCacheSave::Aborted : DiskCacheSave::Failed);
        }
        producer.submit();
        new_bytes += writer->payload_bytes();
        if (!writer->commit()) { return fail(DiskCacheSave::Failed); }
        ++new_blocks;
    }
    // New StateImages.
    for (std::size_t index = 0; index < plans.size(); ++index) {
        if (present[blocks + index]) { continue; }
        DiskKVRecordSpec spec;
        spec.kind    = DiskKVRecordKind::Blob;
        spec.blob    = ids[blocks + index];
        spec.session = session;
        spec.sections.push_back({.kind  = static_cast<std::uint32_t>(DiskKVSectionKind::StateImage),
                                 .bytes = image_bytes});
        spec.meta = {image_bytes, plans[index].frontier, 0, 0, 0, 0, 0, 0};
        std::optional<DiskKVCache::Writer> writer = disk_kv->begin_record(std::move(spec));
        if (!writer) { return refused("StateImage"); }
        DiskProducer producer(*this, *writer, abort);
        bool ok = false;
        if (const auto host = state_store->host_view(plans[index].state); host.has_value()) {
            ok = producer.put(host->data, image_bytes);
        } else {
            state_images->copy_to_host(state_store->physical_slot(plans[index].state),
                                       qwen3_5::HostStateImageView{.data = static_cast<std::byte*>(
                                                                       disk_state_scratch->data()),
                                                                   .layout = &state_layout},
                                       device.stream);
            disk_synchronize();
            ok = producer.put(static_cast<const std::byte*>(disk_state_scratch->data()),
                              image_bytes);
        }
        if (!ok) {
            producer.cancel();
            return fail(producer.aborted() ? DiskCacheSave::Aborted : DiskCacheSave::Failed);
        }
        producer.submit();
        new_bytes += writer->payload_bytes();
        if (!writer->commit()) { return fail(DiskCacheSave::Failed); }
        ++new_states;
    }
    // The manifest, renamed by the writer after its blobs.
    DiskKVRecordSpec spec;
    spec.kind    = DiskKVRecordKind::Manifest;
    spec.session = session;
    spec.tokens.assign(ledger.begin(), ledger.end());
    spec.blob_refs = ids;
    for (std::size_t index = 0; index < plans.size(); ++index) {
        spec.checkpoints.push_back({.frontier         = plans[index].frontier,
                                    .kind             = plans[index].kind,
                                    .backend_frontier = mtp ? plans[index].frontier - 1U : 0U,
                                    .state_section = static_cast<std::uint32_t>(blocks + index)});
    }
    spec.sections.push_back(
        {.kind  = static_cast<std::uint32_t>(DiskKVSectionKind::TextKV),
         .bytes = static_cast<std::uint64_t>(tail_text) * text_layout.page_stride});
    spec.sections.push_back(
        {.kind  = static_cast<std::uint32_t>(DiskKVSectionKind::BackendKV),
         .bytes = static_cast<std::uint64_t>(tail_backend) * backend_layout.page_stride});
    spec.meta                                 = {text_layout.page_stride,
                                                 backend_layout.page_stride,
                                                 blocks,
                                                 tail_text,
                                                 tail_backend,
                                                 image_bytes,
                                                 static_cast<std::uint64_t>(speculative_backend),
                                                 (static_cast<std::uint64_t>(kDiskBlockTokens) << 32U) |
                                                     static_cast<std::uint64_t>(kPagedKVPageSize)};
    std::optional<DiskKVCache::Writer> writer = disk_kv->begin_record(std::move(spec));
    if (!writer) { return refused("manifest"); }
    DiskProducer producer(*this, *writer, abort);
    bool ok = producer.put_pages(*text_kv_addresses, *text_kv_pages, sequence.kv->text, block_pages,
                                 tail_text, text_pool, text_layout);
    if (ok && tail_backend != 0) {
        ok = producer.put_pages(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                                block_pages, tail_backend, *backend_pool, backend_layout);
    }
    if (!ok) {
        producer.cancel();
        return fail(producer.aborted() ? DiskCacheSave::Aborted : DiskCacheSave::Failed);
    }
    producer.submit();
    new_bytes += writer->payload_bytes();
    if (!writer->commit()) { return fail(DiskCacheSave::Failed); }
    std::string kinds;
    for (const DiskCheckpointPlan& plan : plans) {
        kinds += (kinds.empty() ? "" : ",") + std::to_string(plan.frontier) +
                 (plan.kind == kDiskEndpoint  ? "e"
                  : plan.kind == kDiskRewrite ? "r"
                                              : "a");
    }
    std::fprintf(stderr,
                 "[disk-kv] saved a %u-token conversation (checkpoints %s): %u of %u KV block(s) "
                 "and %u of %zu StateImage(s) new, %.0f MiB to write, in %.0f ms\n",
                 frontier, kinds.c_str(), new_blocks, blocks, new_states, plans.size(),
                 static_cast<double>(new_bytes) / kMiB, milliseconds_since(started));
    return DiskCacheSave::Saved;
}

// The prompt tokens a root prefill of `prompt` would load from disk (the static checks
// of disk_cache_resume, then the index), 0 when none. Lets admission prefer the root over a shorter
// in-memory match.
std::uint32_t ProgramImpl::disk_resume_tokens(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (!disk_kv || prompt.has_media() || prompt.rope_delta != 0 || tokens <= kDiskMinimumTokens ||
        tokens > std::numeric_limits<std::uint32_t>::max() ||
        !std::all_of(prompt.token_types.begin(), prompt.token_types.end(),
                     [](std::uint8_t type) { return type == 0; })) {
        return 0;
    }
    return disk_kv->longest_indexed_frontier(
        std::span<const TokenId>(prompt.token_ids.data(), tokens),
        static_cast<std::uint32_t>(tokens) - 1U);
}

std::uint32_t ProgramImpl::disk_cache_resume(SequenceState& sequence, RequestControl& request) {
    RequestControl::Prefill& staged = *request.prefill;
    staged.disk_attempted           = true;
    if (!disk_kv || staged.reuse != ReusePath::Root || staged.cursor != 0 || staged.base != 0 ||
        staged.vision || staged.prompt.has_media() || staged.prompt.rope_delta != 0 ||
        staged.prompt_tokens <= kDiskMinimumTokens || !sequence.kv ||
        staged.prompt.token_ids.size() < staged.prompt_tokens ||
        !std::all_of(staged.prompt.token_types.begin(), staged.prompt.token_types.end(),
                     [](std::uint8_t type) { return type == 0; })) {
        return 0;
    }
    const bool mtp = speculative_backend == SpeculativeBackend::Mtp;
    if (mtp && (!staged.prepare_mtp || !sequence.kv->backend)) { return 0; }
    const auto started = Clock::now();
    const std::span<const TokenId> prompt(staged.prompt.token_ids.data(), staged.prompt_tokens);
    std::optional<DiskKVCache::Match> match = disk_kv->find(prompt, staged.prompt_tokens - 1U);
    if (!match) { return 0; }
    const double find_ms              = milliseconds_since(started);
    const DiskKVRecordHeader& header  = match->manifest.header();
    const DiskKVCheckpoint checkpoint = match->checkpoint;
    const std::uint32_t frontier      = checkpoint.frontier;
    const DeviceKVPagePool& text_pool = decoder->text_kv.page_pool();
    const HostKVPageLayout text_layout =
        plan_host_kv_page_layout(text_pool.geometry(), text_pool.host_replicas());
    const DeviceKVPagePool* backend_pool =
        mtp && backend_kv_cache() != nullptr ? &backend_kv_cache()->page_pool() : nullptr;
    HostKVPageLayout backend_layout;
    if (backend_pool != nullptr) {
        backend_layout =
            plan_host_kv_page_layout(backend_pool->geometry(), backend_pool->host_replicas());
    }
    const StateImageHostLayout& state_layout = state_images->host_layout();
    const std::uint32_t text_pages           = kv_pages_for_frontier(frontier);
    const std::uint32_t backend_pages        = mtp ? kv_pages_for_frontier(frontier - 1U) : 0U;
    const std::uint64_t blocks               = header.meta[2];
    const std::uint64_t block_pages          = blocks * kDiskBlockPages;
    const std::uint32_t tail_section = match->manifest.section_of(DiskKVSectionKind::TextKV);
    if (header.meta[0] != text_layout.page_stride || header.meta[1] != backend_layout.page_stride ||
        header.meta[5] != state_layout.image_bytes ||
        header.meta[6] != static_cast<std::uint64_t>(speculative_backend) ||
        header.meta[7] != ((static_cast<std::uint64_t>(kDiskBlockTokens) << 32U) |
                           static_cast<std::uint64_t>(kPagedKVPageSize)) ||
        text_pages > block_pages + header.meta[3] || backend_pages > block_pages + header.meta[4] ||
        blocks + header.checkpoints.size() > match->blob_refs.size() ||
        checkpoint.state_section >= match->blob_refs.size() ||
        tail_section >= header.sections.size() ||
        text_pages > text_kv_addresses->mapped_pages(sequence.kv->text) ||
        (backend_pages != 0 &&
         backend_pages > backend_kv_addresses->mapped_pages(*sequence.kv->backend))) {
        std::fprintf(stderr,
                     "[disk-kv] conversation %016llx does not match this layout; prefilling\n",
                     static_cast<unsigned long long>(header.id));
        return 0;
    }
    const std::int32_t state_slot = state_store->physical_slot(sequence.state.write);
    auto* read_base               = static_cast<std::byte*>(disk_read_buffer->data());
    std::byte* buffers[2]         = {read_base, read_base + kDiskReadBytes};
    auto* scratch                 = static_cast<std::byte*>(disk_state_scratch->data());

    // Units: groups of KV blocks the checkpoint needs (as many as fit in one read buffer, read by
    // one thread per file: the NVMe wants several reads in flight), then the manifest's tail.
    const std::uint32_t needed_pages = std::max(text_pages, backend_pages);
    const std::uint64_t needed_blocks =
        std::min<std::uint64_t>(blocks, (needed_pages + kDiskBlockPages - 1U) / kDiskBlockPages);
    const bool needs_tail = needed_pages > block_pages;
    const auto aligned    = [](std::uint64_t bytes) { return (bytes + 4095U) / 4096U * 4096U; };
    const std::uint64_t slot_bytes = aligned(aligned((kDiskBlockTokens + 1U) * sizeof(TokenId)) +
                                             aligned(kDiskBlockPages * text_layout.page_stride) +
                                             kDiskBlockPages * backend_layout.page_stride) +
                                     4096U;
    const std::uint64_t group  = std::max<std::uint64_t>(1U, kDiskReadBytes / slot_bytes);
    const std::uint64_t groups = (needed_blocks + group - 1U) / group;
    const std::uint64_t units  = groups + (needs_tail ? 1U : 0U);
    std::vector<DeviceKVPageHandle> text_destination(text_pages),
        backend_destination(backend_pages);
    for (std::uint32_t page = 0; page < text_pages; ++page) {
        text_destination[page] = text_kv_addresses->physical_page(sequence.kv->text, page);
    }
    for (std::uint32_t page = 0; page < backend_pages; ++page) {
        backend_destination[page] =
            backend_kv_addresses->physical_page(*sequence.kv->backend, page);
    }

    struct Loaded {
        std::optional<DiskKVRecordFile> blob;
        std::optional<std::uint64_t> start;
    };

    std::optional<DiskKVBlobId> broken_blob;
    const auto load = [&](std::uint64_t unit, std::byte* into) {
        std::vector<Loaded> loaded;
        if (unit < groups) {
            const std::uint64_t first = unit * group;
            const std::uint64_t count = std::min(group, needed_blocks - first);
            loaded.resize(count);
            std::vector<std::thread> readers;
            readers.reserve(count);
            for (std::uint64_t k = 0; k < count; ++k) {
                readers.emplace_back([&, k] {
                    Loaded& slot = loaded[k];
                    slot.blob    = disk_kv->open_blob(match->blob_refs[first + k]);
                    if (slot.blob) {
                        slot.start = slot.blob->read_region(
                            0, std::span<std::byte>(into + k * slot_bytes, slot_bytes));
                    }
                });
            }
            for (std::thread& reader : readers) { reader.join(); }
        } else {
            loaded.resize(1);
            loaded[0].start = match->manifest.read_region(
                tail_section, std::span<std::byte>(into, kDiskReadBytes));
        }
        return loaded;
    };
    // Copies pages [first, first + count) of a section that starts at page `section_first`.
    const auto copy_pages = [&](const std::byte* data, const DeviceKVPagePool& pool,
                                const HostKVPageLayout& layout,
                                const std::vector<DeviceKVPageHandle>& destination,
                                std::uint64_t section_first, std::uint64_t section_pages) {
        const std::uint64_t end =
            std::min<std::uint64_t>(section_first + section_pages, destination.size());
        if (end <= section_first) { return; }
        pool.copy_from_host_buffer(data, layout,
                                   std::span<const DeviceKVPageHandle>(
                                       destination.data() + section_first, end - section_first),
                                   device.stream);
    };

    bool ok = false;
    try {
        // The StateImage is read by a thread of its own while the KV units load.
        bool state_ok = false;
        std::thread state_reader([&] {
            std::optional<DiskKVRecordFile> state_blob =
                disk_kv->open_blob(match->blob_refs[checkpoint.state_section]);
            if (!state_blob) { return; }
            const std::uint32_t section = state_blob->section_of(DiskKVSectionKind::StateImage);
            state_ok                    = section + 1U == state_blob->header().sections.size() &&
                       state_blob->header().sections[section].bytes == state_layout.image_bytes &&
                       state_blob
                           ->read_region(section,
                                         std::span<std::byte>(scratch, disk_state_scratch->size()))
                           .has_value();
        });

        struct StateJoiner {
            std::thread& thread;

            ~StateJoiner() {
                if (thread.joinable()) { thread.join(); }
            }
        } state_joiner{state_reader};

        ok = true;
        // KV units, read ahead by a thread while the previous one is copied to the GPUs.
        std::vector<Loaded> current;
        if (ok && units != 0) { current = load(0, buffers[0]); }
        for (std::uint64_t unit = 0; ok && unit < units; ++unit) {
            std::vector<Loaded> next;
            std::thread loader;
            if (unit + 1 < units) {
                loader = std::thread([&, unit] { next = load(unit + 1, buffers[(unit + 1) % 2]); });
            }

            struct Joiner {
                std::thread& thread;

                ~Joiner() {
                    if (thread.joinable()) { thread.join(); }
                }
            } joiner{loader};

            const std::byte* base = buffers[unit % 2];
            if (unit < groups) {
                for (std::uint64_t k = 0; ok && k < current.size(); ++k) {
                    const std::uint64_t block      = unit * group + k;
                    const Loaded& slot             = current[k];
                    const std::byte* data          = base + k * slot_bytes;
                    const DiskKVRecordHeader* blob = slot.blob ? &slot.blob->header() : nullptr;
                    ok = blob != nullptr && slot.start && blob->sections.size() == 3 &&
                         blob->token_count == kDiskBlockTokens + 1U &&
                         blob->sections[1].bytes == kDiskBlockPages * text_layout.page_stride &&
                         blob->sections[2].bytes ==
                             (mtp ? kDiskBlockPages * backend_layout.page_stride : 0U);
                    if (ok) {
                        // The block's tokens are the prompt's where the checkpoint uses them.
                        const std::uint64_t first = block * kDiskBlockTokens;
                        const std::uint64_t count =
                            std::min<std::uint64_t>(kDiskBlockTokens + 1U, frontier - first);
                        ok = std::memcmp(data + (blob->sections[0].offset - *slot.start),
                                         prompt.data() + first, count * sizeof(TokenId)) == 0;
                    }
                    if (!ok) {
                        broken_blob = match->blob_refs[block];
                        break;
                    }
                    copy_pages(data + (blob->sections[1].offset - *slot.start), text_pool,
                               text_layout, text_destination, block * kDiskBlockPages,
                               kDiskBlockPages);
                    if (mtp) {
                        copy_pages(data + (blob->sections[2].offset - *slot.start), *backend_pool,
                                   backend_layout, backend_destination, block * kDiskBlockPages,
                                   kDiskBlockPages);
                    }
                }
            } else {
                ok = !current.empty() && current[0].start.has_value();
                if (ok) {
                    const std::uint64_t start = *current[0].start;
                    copy_pages(base + (header.sections[tail_section].offset - start), text_pool,
                               text_layout, text_destination, block_pages, header.meta[3]);
                    if (mtp && tail_section + 1U < header.sections.size()) {
                        copy_pages(base + (header.sections[tail_section + 1U].offset - start),
                                   *backend_pool, backend_layout, backend_destination, block_pages,
                                   header.meta[4]);
                    }
                }
            }
            disk_synchronize(); // the buffer is reused by the read after next
            if (loader.joinable()) { loader.join(); }
            current = std::move(next);
        }
        state_reader.join();
        if (!state_ok) {
            if (ok) { broken_blob = match->blob_refs[checkpoint.state_section]; }
            ok = false;
        }
        if (ok) {
            state_images->copy_from_host(
                qwen3_5::HostStateImageConstView{.data = scratch, .layout = &state_layout},
                state_slot, device.stream);
        }
        disk_synchronize();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[disk-kv] resume failed: %s\n", error.what());
        ok = false;
    }
    if (!ok) {
        // The prefill restarts from the root: its StateImage must be the zero state again (the
        // KV pages are rewritten by the prefill before they are read).
        state_images->zero_slot(state_slot, device.stream);
        disk_synchronize();
        disk_kv->discard(header.id, broken_blob);
        std::fprintf(stderr,
                     "[disk-kv] conversation %016llx unreadable (%s); dropped, prefilling from the "
                     "root\n",
                     static_cast<unsigned long long>(header.id),
                     broken_blob ? ("blob " + broken_blob->hex()).c_str() : "manifest");
        return 0;
    }
    sequence.text_kv_valid = frontier;
    if (mtp) { sequence.mtp_kv_valid = frontier - 1U; }
    commit_sequence_kv(sequence, frontier, mtp ? frontier - 1U : 0U);
    refresh_state_views(sequence);
    staged.cursor    = frontier;
    staged.disk_base = frontier;
    if (staged.prepare_mtp) { staged.mtp_bridge = MtpBridgeMode::BeforeSuffix; }
    // Captures before the loaded frontier have no state to capture: skipped.
    while (staged.next_capture < staged.capture_groups.size() &&
           staged.capture_groups[staged.next_capture].frontier <= frontier) {
        ++staged.next_capture;
    }
    const double bytes = static_cast<double>(state_layout.image_bytes) +
                         static_cast<double>(text_pages) * text_layout.page_stride +
                         static_cast<double>(backend_pages) * backend_layout.page_stride;
    std::fprintf(stderr,
                 "[disk-kv] resumed %u of %u prompt tokens from disk (conversation %016llx, %llu "
                 "block(s)%s, %.0f MiB) in %.0f ms (lookup %.1f ms)\n",
                 frontier, staged.prompt_tokens, static_cast<unsigned long long>(header.id),
                 static_cast<unsigned long long>(needed_blocks), needs_tail ? " + tail" : "",
                 bytes / kMiB, milliseconds_since(started), find_ms);
    return frontier;
}

std::string ProgramImpl::disk_cache_status() const {
    if (!disk_kv) { return "off"; }
    const DiskKVCacheStats stats = disk_kv->stats();
    char text[640];
    std::snprintf(
        text, sizeof(text),
        "%llu conversation(s), %llu blob(s), %.2f of %.2f GiB (file system %.1f GiB free); "
        "written %llu record(s) / %.2f GiB (%llu blob(s), %llu reused), resumed %llu, "
        "evicted %llu conversation(s) / %llu blob(s), superseded %llu, aborted %llu, "
        "refused for space %llu, failed %llu writes / %llu reads",
        static_cast<unsigned long long>(stats.records),
        static_cast<unsigned long long>(stats.blobs),
        static_cast<double>(stats.bytes) / (kMiB * 1024.0),
        static_cast<double>(stats.capacity_bytes) / (kMiB * 1024.0),
        static_cast<double>(stats.fs_free_bytes) / (kMiB * 1024.0),
        static_cast<unsigned long long>(stats.written),
        static_cast<double>(stats.written_bytes) / (kMiB * 1024.0),
        static_cast<unsigned long long>(stats.written_blobs),
        static_cast<unsigned long long>(stats.reused_blobs),
        static_cast<unsigned long long>(stats.hits), static_cast<unsigned long long>(stats.evicted),
        static_cast<unsigned long long>(stats.evicted_blobs),
        static_cast<unsigned long long>(stats.superseded),
        static_cast<unsigned long long>(stats.aborted),
        static_cast<unsigned long long>(stats.space_refusals),
        static_cast<unsigned long long>(stats.write_failures),
        static_cast<unsigned long long>(stats.read_failures));
    return text;
}

} // namespace ninfer::models::qwen3_5::detail
