// Translation-unit fragment of src/serve/operational_log.cpp, included once at its end (after
// its closing namespace): the startup lines of the two-device mode. Not a header. Our file; see
// docs/maintainer/upstream-merge.md.

namespace ninfer::serve {
namespace {

void log_tensor_parallel(spdlog::logger& logger, const GenerationService& service,
                         const ninfer::EngineOptions& engine,
                         const ninfer::ContextCacheOptions& cache,
                         const ninfer::MemorySummary& memory) {
    // The Host tiers are off unless --host-state-slots / --host-kv-mib (or --profile 27b) set them.
    const bool host =
        cache.enabled && (cache.host_state_slots != 0 || cache.host_kv_capacity_bytes != 0);
    logger.info("tensor parallel | tp 2 on devices {},{} | p2p {} | KV, StateImages and "
                "runtime reserved per rank | {} Device checkpoint StateImages | {}",
                engine.devices.at(0), engine.devices.at(1),
                service.load_summary().peer_access ? "on" : "off (host-staged copies)",
                cache.enabled ? *cache.device_state_slots : 0U,
                host ? "Host cache backed on both ranks: " +
                           std::to_string(cache.host_state_slots) + " Host StateImages, " +
                           std::to_string(cache.host_kv_capacity_bytes >> 20U) + " MiB Host KV"
                     : std::string("Host context-cache tiers off; a prefill checkpoint that finds "
                                   "the Device StateImages full first reclaims idle private "
                                   "continuations and is skipped only when none can be released "
                                   "(raise --device-state-slots for many conversations)"));
    {
        const LoadSummary& load = service.load_summary();
        if (!load.tp_transport.empty()) {
            const std::string kernel =
                (load.tp_mailbox_kernel.empty() ? std::string()
                                                : " | exchange kernel " + load.tp_mailbox_kernel) +
                (load.tp_mailbox_slot_bytes == 0
                     ? std::string()
                     : " | slot " + std::to_string(load.tp_mailbox_slot_bytes / 1024U) + " KiB");
            if (load.tp_mailbox_probe_ms > 0.0) {
                logger.info(
                    "tensor parallel | captured all-reduces: {}{} | mailbox probe {:.2f} ms",
                    load.tp_transport, kernel, load.tp_mailbox_probe_ms);
            } else {
                logger.info("tensor parallel | captured all-reduces: {}{}", load.tp_transport,
                            kernel);
            }
        }
        if (!load.tp_mailbox_fallback.empty() && load.tp_transport == "copies") {
            logger.warn("tensor parallel | pinned-host mailbox disabled: {}; decode uses the "
                        "cross-device copies (slower). This is expected under WSL2; pass "
                        "--no-tp-mailbox to skip the probe",
                        load.tp_mailbox_fallback);
        } else if (!load.tp_mailbox_fallback.empty()) {
            logger.warn("tensor parallel | pinned-host mailbox kept for the target forward "
                        "only: {}; the MTP draft phase uses the cross-device copies. This "
                        "is expected under WSL2; NINFER_TP_MAILBOX_DRAFT=copies starts "
                        "this way directly",
                        load.tp_mailbox_fallback);
        }
        if (!load.tp_proposal_head.empty()) {
            logger.info("tensor parallel | {} proposal head: {}",
                        engine.speculative.backend == ninfer::SpeculativeBackend::DFlash2
                            ? "DFlash2"
                            : "MTP",
                        load.tp_proposal_head);
        }
        if (!load.tp_drafter.empty()) {
            logger.info("tensor parallel | DFlash2 drafter: {}", load.tp_drafter);
        }
    }
    // Measured against the planned per-device allowance, to calibrate the tp 2 constants.
    // An overrun only uses device memory the KV sizing left unreserved, so it warns instead
    // of failing startup.
    if (engine.use_cuda_graph) {
        constexpr double kMiB = 1024.0 * 1024.0;
        for (std::size_t rank = 0; rank < memory.cuda_graph_observed_bytes.size(); ++rank) {
            const std::size_t observed  = memory.cuda_graph_observed_bytes[rank];
            const std::size_t allowance = memory.cuda_graph_allowance_bytes;
            logger.info("cuda graphs | rank {}: observed {:.1f} MiB, allowance {:.1f} MiB", rank,
                        static_cast<double>(observed) / kMiB,
                        static_cast<double>(allowance) / kMiB);
            if (observed > allowance) {
                logger.warn("cuda graphs | rank {}: observed {:.1f} MiB exceeds the "
                            "allowance of {:.1f} MiB; the excess comes out of the memory "
                            "left unreserved after KV sizing",
                            rank, static_cast<double>(observed) / kMiB,
                            static_cast<double>(allowance) / kMiB);
            }
        }
    }
}

} // namespace
} // namespace ninfer::serve
