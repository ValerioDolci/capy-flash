#pragma once

// Disk tier of the context cache (opt-in with --disk-cache-dir).
//
// Content-addressed storage (format 2), after LMCache / SGLang HiCache:
//   DIR/.lock            exclusive flock for the life of the server (one directory = one server)
//   DIR/m/<id>.nidkv     one manifest per saved conversation: tokens, the ids of its blobs, the
//                        checkpoints (frontier + StateImage blob), and its tail (sections)
//   DIR/b/<id128>.nkvb   blobs, written once and shared: KV blocks (a fixed number of tokens,
//                        addressed by a chained hash of the prefix tokens) and StateImages
//                        (addressed by a hash of the tokens before the checkpoint)
// Every file is a record: a header (format version, validity key, kind, sections, one xxh64 per
// 4 MiB block of every section) followed by 4 KiB aligned sections. The model layer decides what
// the sections and ids mean.
//
// Files are written by a thread of their own from a fixed pool of staging buffers, to a temporary,
// then header, fsync, rename and directory fsync: a record is complete or absent; a manifest is
// renamed only after its blobs. Blobs carry reference counts (manifests and saves in progress);
// blobs nobody references (an interrupted save) are kept for the next attempt, evicted first and
// deleted at the next start. Space is bounded by a byte cap with least-recently-used eviction of
// manifests (file mtime = last use) and by a free-space margin of the file system.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ninfer {

// XXH64 (Yann Collet's algorithm), one-shot and streaming.
[[nodiscard]] std::uint64_t xxh64(const void* data, std::size_t length, std::uint64_t seed = 0);

class Xxh64Stream {
public:
    explicit Xxh64Stream(std::uint64_t seed = 0) noexcept { reset(seed); }

    void reset(std::uint64_t seed = 0) noexcept;
    void update(const void* data, std::size_t length) noexcept;
    [[nodiscard]] std::uint64_t digest() const noexcept;

private:
    std::uint64_t v_[4]{};
    std::uint64_t seed_  = 0;
    std::uint64_t total_ = 0;
    unsigned char buffer_[32]{};
    std::size_t buffered_ = 0;
};

inline constexpr std::uint32_t kDiskKVFormatVersion = 2;
inline constexpr std::size_t kDiskKVBlockBytes      = 4ULL << 20U;
inline constexpr std::size_t kDiskKVAlignment       = 4096;
inline constexpr std::size_t kDiskKVMetaWords       = 8;

enum class DiskKVSectionKind : std::uint32_t {
    Tokens     = 1,
    StateImage = 2,
    TextKV     = 3,
    BackendKV  = 4,
    BlobRefs   = 5, // manifest: the ids of its blobs, 16 bytes each
};

enum class DiskKVRecordKind : std::uint32_t {
    Manifest = 1,
    Blob     = 2,
};

struct DiskKVBlobId {
    std::uint64_t a = 0;
    std::uint64_t b = 0;

    [[nodiscard]] friend bool operator==(DiskKVBlobId, DiskKVBlobId) noexcept = default;
    [[nodiscard]] std::string hex() const;
};

struct DiskKVBlobIdHash {
    std::size_t operator()(DiskKVBlobId id) const noexcept {
        return static_cast<std::size_t>(id.a ^ (id.b * 0x9E3779B97F4A7C15ULL));
    }
};

struct DiskKVCheckpoint {
    std::uint32_t frontier         = 0;
    std::uint32_t kind             = 0;
    std::uint32_t backend_frontier = 0;
    // Manifest: index of the checkpoint's StateImage in the manifest's blob references.
    std::uint32_t state_section = 0;
    std::uint64_t digest        = 0; // xxh64 of tokens [0, frontier) (filled by the cache)
};

struct DiskKVSectionInfo {
    std::uint32_t kind  = 0;
    std::uint64_t bytes = 0;
    // Filled from the layout: the section's file offset and its first block-hash index.
    std::uint64_t offset      = 0;
    std::uint64_t first_block = 0;
};

// What one record will hold. Section 0 is always the tokens; `sections` lists the rest in the
// order the producer submits them (a manifest's BlobRefs section is written by the cache).
struct DiskKVRecordSpec {
    DiskKVRecordKind kind = DiskKVRecordKind::Manifest;
    DiskKVBlobId blob;                // Blob: its id
    std::uint64_t session = 0;        // the save it belongs to (DiskKVCache::begin_session)
    std::vector<std::int32_t> tokens; // Manifest: the conversation; Blob: free use
    std::vector<DiskKVCheckpoint> checkpoints;
    std::vector<DiskKVBlobId> blob_refs; // Manifest: every blob it uses
    std::vector<DiskKVSectionInfo> sections;
    std::array<std::uint64_t, kDiskKVMetaWords> meta{};
};

// Parsed header of a stored record.
struct DiskKVRecordHeader {
    DiskKVRecordKind kind = DiskKVRecordKind::Manifest;
    std::uint64_t id      = 0;
    DiskKVBlobId blob;
    std::uint64_t key_hash = 0;
    std::string key;
    std::uint32_t token_count  = 0;
    std::uint64_t header_bytes = 0;
    std::uint64_t file_bytes   = 0;
    std::uint64_t created_ns   = 0;
    std::uint64_t full_digest  = 0; // xxh64 of every token
    std::array<std::uint64_t, kDiskKVMetaWords> meta{};
    std::vector<DiskKVCheckpoint> checkpoints;
    std::vector<DiskKVSectionInfo> sections; // section 0 = tokens
    std::vector<std::uint64_t> block_hashes;
};

struct DiskKVCacheStats {
    std::uint64_t records         = 0; // manifests with this key
    std::uint64_t foreign_records = 0; // manifests with another key: ignored, evictable
    std::uint64_t blobs           = 0;
    std::uint64_t orphan_blobs    = 0; // no reference (interrupted save)
    std::uint64_t bytes           = 0; // every manifest and blob
    std::uint64_t blob_bytes      = 0;
    std::uint64_t capacity_bytes  = 0;
    std::uint64_t min_free_bytes  = 0; // free-space margin of the file system
    std::uint64_t fs_free_bytes   = 0; // at the last check
    std::uint64_t removed_broken  = 0; // at start and on read
    std::uint64_t removed_temp    = 0; // temporaries of dead processes at start
    std::uint64_t removed_orphans = 0; // at start
    std::uint64_t evicted         = 0; // manifests, LRU under the cap or the margin
    std::uint64_t evicted_blobs   = 0;
    std::uint64_t superseded      = 0; // earlier saves of the same conversation
    std::uint64_t written         = 0; // records (manifests and blobs)
    std::uint64_t written_bytes   = 0;
    std::uint64_t written_blobs   = 0;
    std::uint64_t reused_blobs    = 0; // already on disk when a save asked for them
    std::uint64_t aborted         = 0;
    std::uint64_t write_failures  = 0;
    std::uint64_t space_refusals  = 0; // a record skipped for lack of cap or free space
    std::uint64_t hits            = 0;
    std::uint64_t read_failures   = 0;
};

class DiskKVCache;

// One staging buffer lent by the cache. Release it with Writer::submit or give_back.
struct DiskKVBuffer {
    std::byte* data     = nullptr;
    std::size_t bytes   = 0; // capacity
    std::uint32_t index = 0;
};

// An opened record (manifest or blob); reads verify every block they cover.
class DiskKVRecordFile {
public:
    DiskKVRecordFile() noexcept = default;
    DiskKVRecordFile(DiskKVRecordFile&& other) noexcept;
    DiskKVRecordFile& operator=(DiskKVRecordFile&& other) noexcept;
    ~DiskKVRecordFile();

    DiskKVRecordFile(const DiskKVRecordFile&)            = delete;
    DiskKVRecordFile& operator=(const DiskKVRecordFile&) = delete;

    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }

    [[nodiscard]] const DiskKVRecordHeader& header() const noexcept { return header_; }

    // Index of the first section of `kind`, or sections.size().
    [[nodiscard]] std::uint32_t section_of(DiskKVSectionKind kind) const noexcept;

    // Reads bytes [0, bytes) of `section` in chunks of at most `chunk.size()` bytes (a multiple
    // of kDiskKVBlockBytes) into `chunk`, verifying every block it covers, and calls
    // `sink(data, offset)` per chunk. False on an I/O error or a checksum mismatch.
    bool read_prefix(std::uint32_t section, std::uint64_t bytes, std::span<std::byte> chunk,
                     const std::function<void(std::span<const std::byte>, std::uint64_t)>& sink);

    // Reads sections [first_section, end) whole and contiguous (from the first one's offset)
    // into `into` (4 KiB aligned), verifying them. Returns the region's start offset in the file,
    // or nullopt (I/O error, checksum mismatch, region larger than `into`).
    std::optional<std::uint64_t> read_region(std::uint32_t first_section,
                                             std::span<std::byte> into);

private:
    friend class DiskKVCache;
    int fd_        = -1;
    int direct_fd_ = -1; // O_DIRECT descriptor of the same file, or -1
    DiskKVRecordHeader header_;
};

class DiskKVCache {
public:
    // Whether another process holds `directory`'s lock: its PID (or "unknown") if so. Lets a
    // server fail before loading its weights (DiskKVCache takes the lock itself later).
    [[nodiscard]] static std::optional<std::string>
    directory_holder(const std::filesystem::path& directory);

    struct Options {
        std::filesystem::path directory;
        std::uint64_t capacity_bytes = 0;
        // Free bytes the file system keeps at least; 0 = max(10 GiB, 5 % of the file system).
        std::uint64_t min_free_bytes = 0;
        std::string key;
        std::size_t staging_buffers = 4;
        std::size_t staging_bytes   = 64ULL << 20U; // multiple of kDiskKVBlockBytes
        // Allocation of the staging buffers (pinned memory in the engine); default aligned_alloc.
        std::function<void*(std::size_t)> allocate;
        std::function<void(void*)> release;
        // Test hooks: before each pwrite of a payload (writer thread); free bytes of the file
        // system instead of statvfs.
        std::function<void()> before_write;
        std::function<std::uint64_t()> free_bytes;
    };

    class Writer {
    public:
        Writer() noexcept = default;
        Writer(Writer&& other) noexcept;
        Writer& operator=(Writer&& other) noexcept;
        ~Writer();

        Writer(const Writer&)            = delete;
        Writer& operator=(const Writer&) = delete;

        [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

        // Waits for a free staging buffer while `abort` returns false; nullopt when it returned
        // true or the cache is stopping.
        [[nodiscard]] std::optional<DiskKVBuffer> acquire(const std::function<bool()>& abort = {});
        // Appends `bytes` of `buffer` to the record (sections are contiguous in submission
        // order, the padding between sections is written by the cache).
        void submit(DiskKVBuffer buffer, std::size_t bytes);
        // Returns an unused buffer.
        void give_back(DiskKVBuffer buffer);
        // Finishes the record on the writer thread (header, fsync, rename, index). Returns false
        // if the payload size does not match the spec (the record is then aborted).
        bool commit();
        void abort() noexcept;

        [[nodiscard]] std::uint64_t id() const noexcept { return id_; }

        [[nodiscard]] std::uint64_t payload_bytes() const noexcept { return expected_; }

    private:
        friend class DiskKVCache;
        DiskKVCache* cache_      = nullptr;
        std::uint64_t id_        = 0;
        std::uint64_t submitted_ = 0;
        std::uint64_t expected_  = 0;
    };

    // A manifest opened for a resume, with the checkpoint chosen and its blob references.
    struct Match {
        DiskKVRecordFile manifest;
        DiskKVCheckpoint checkpoint;
        std::vector<DiskKVBlobId> blob_refs;
    };

    explicit DiskKVCache(Options options);
    ~DiskKVCache();

    DiskKVCache(const DiskKVCache&)            = delete;
    DiskKVCache& operator=(const DiskKVCache&) = delete;

    [[nodiscard]] const Options& options() const noexcept { return options_; }

    [[nodiscard]] DiskKVCacheStats stats() const;

    [[nodiscard]] std::uint64_t key_hash() const noexcept { return key_hash_; }

    // Size of the file a spec makes (header + aligned sections).
    [[nodiscard]] static std::uint64_t record_file_bytes(const DiskKVRecordSpec& spec,
                                                         std::size_t key_bytes);

    // A save: pins the blobs of `ids` already on disk (present[i] = true) so that nothing evicts
    // them until the save ends; the missing ones are written with begin_record (kind Blob, same
    // session) and pinned when they land. A committed manifest of the session keeps the pins as
    // its references; abort_session releases them.
    [[nodiscard]] std::uint64_t begin_session(std::span<const DiskKVBlobId> ids,
                                              std::vector<bool>& present);
    void abort_session(std::uint64_t session) noexcept;

    // Starts a record: reserves its file size under the cap and the free-space margin (evicting
    // orphan blobs, then least-recently-used manifests), opens the temporary. nullopt when it
    // cannot fit or the file cannot be created.
    [[nodiscard]] std::optional<Writer> begin_record(DiskKVRecordSpec spec);

    // Best checkpoint for `prompt`: the longest frontier F <= max_frontier whose tokens equal
    // prompt[0, F) (digest, then exact comparison with the stored tokens). Refreshes the
    // manifest's last use. A broken manifest is deleted and skipped.
    [[nodiscard]] std::optional<Match> find(std::span<const std::int32_t> prompt,
                                            std::uint32_t max_frontier);
    // The frontier find() would try first: the longest F <= max_frontier whose digest matches
    // prompt[0, F), from the in-memory index alone (no file opened, last use untouched); 0 when
    // none. Admission compares it with the in-memory matches.
    [[nodiscard]] std::uint32_t longest_indexed_frontier(std::span<const std::int32_t> prompt,
                                                         std::uint32_t max_frontier);

    // Opens a blob for reading (header verified: kind, id, key).
    [[nodiscard]] std::optional<DiskKVRecordFile> open_blob(DiskKVBlobId id) const;
    [[nodiscard]] std::filesystem::path blob_path(DiskKVBlobId id) const;

    // A resume found a blob of `manifest` missing or broken: the blob and the manifest go.
    void discard(std::uint64_t manifest, std::optional<DiskKVBlobId> blob) noexcept;

    // Waits until the writer thread has finished every committed or aborted record, or until the
    // deadline (then the records still in flight are aborted). Returns true when drained.
    bool drain(std::chrono::steady_clock::time_point deadline);

private:
    struct Entry {
        std::uint64_t id = 0;
        std::filesystem::path path;
        std::uint64_t bytes       = 0;
        bool key_ok               = false;
        std::int64_t last_use_ns  = 0;
        std::uint32_t token_count = 0;
        std::uint64_t full_digest = 0;
        std::vector<DiskKVCheckpoint> checkpoints;
        std::vector<DiskKVBlobId> refs;
    };

    struct Blob {
        std::uint64_t bytes      = 0;
        std::uint32_t refs       = 0;
        std::int64_t last_use_ns = 0;
    };

    struct Job;

    void scan();
    void writer_loop();
    // Caller holds mutex_. Frees room for `bytes` more under the cap and the free-space margin.
    bool make_room(std::uint64_t bytes);
    void remove_manifest(std::vector<Entry>::iterator it); // caller holds mutex_
    void release_blob(DiskKVBlobId id);                    // caller holds mutex_
    [[nodiscard]] std::uint64_t free_bytes() const;
    void finish_job(Job& job);
    void abort_job(Job& job) noexcept;
    void return_buffer(std::uint32_t index) noexcept;
    [[nodiscard]] std::filesystem::path manifest_path(std::uint64_t id) const;
    [[nodiscard]] std::filesystem::path temp_path(const std::filesystem::path& final_path) const;
    static std::optional<DiskKVRecordHeader> read_header(int fd, std::uint64_t file_bytes);

    Options options_;
    std::uint64_t key_hash_ = 0;
    std::filesystem::path manifests_dir_;
    std::filesystem::path blobs_dir_;
    int lock_fd_ = -1;

    mutable std::mutex mutex_;
    std::condition_variable buffers_cv_;
    std::condition_variable queue_cv_;
    std::condition_variable idle_cv_;
    std::vector<Entry> entries_;
    std::unordered_map<DiskKVBlobId, Blob, DiskKVBlobIdHash> blobs_;
    std::unordered_map<std::uint64_t, std::vector<DiskKVBlobId>> sessions_; // pins per save
    std::uint64_t used_bytes_     = 0; // manifests + blobs + reservations
    std::uint64_t reserved_bytes_ = 0; // records in flight
    mutable DiskKVCacheStats stats_;

    std::vector<std::byte*> buffers_;
    std::vector<std::uint32_t> free_buffers_;

    struct Item {
        enum class Kind : std::uint8_t { Data, Commit, Abort, Unpin } kind = Kind::Data;
        std::uint64_t id = 0; // record id, or session id for Unpin
        DiskKVBuffer buffer;
        std::size_t bytes = 0;
    };

    std::deque<Item> queue_;
    std::vector<std::unique_ptr<Job>> jobs_;
    std::uint64_t in_flight_    = 0;
    bool stopping_              = false;
    std::uint64_t next_id_      = 0;
    std::uint64_t next_session_ = 0;
    std::thread thread_;
};

} // namespace ninfer
