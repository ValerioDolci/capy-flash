#pragma once

// The transports of the tp 2 Program's EAGER collectives (prefill), decided once at construction
// from the pair's peer access, elastic KV and two environment switches. Header-only and free of
// CUDA so that the decision table is unit-tested on the host (tests/test_tp2_eager_transport.cpp).
//
//   host_exchange  the pinned-host HostExchange carries every eager all-reduce
//                  (and its split-phase lanes): without peer access or with elastic KV (whose VMM
//                  workspace a peer pull cannot read), unless NINFER_TP_HOST_EXCHANGE=0; always
//                  with NINFER_TP_HOST_EXCHANGE=1.
//   peer_lanes     ops::PeerLanes carry the split-phase sums of the 27B prefill
//                  overlap over the direct peer copy: with peer access and no HostExchange, unless
//                  NINFER_TP_PEER_OVERLAP=0. NINFER_TP_PEER_OVERLAP=force attaches them whatever
//                  the pair (next to a HostExchange, which keeps every other eager call): a
//                  correctness test of the lane path on boards without P2P.
//   peer_lanes_direct  the lanes' copies are direct (peer access); false only under force
//                  without P2P (PeerLanes(direct = false): driver-staged, host-ordered, slow).
//
// Without peer access and without force, peer_lanes is false and host_exchange alone decides the
// eager path of boards without P2P.

#include <string_view>

namespace ninfer::models::qwen3_5::detail {

struct Tp2EagerTransport {
    bool host_exchange     = false;
    bool peer_lanes        = false;
    bool peer_lanes_direct = false;
};

[[nodiscard]] constexpr Tp2EagerTransport
tp2_eager_transport(bool peer_access, bool elastic_kv, std::string_view host_exchange_env,
                    std::string_view peer_overlap_env) noexcept {
    Tp2EagerTransport t;
    const bool host_exchange_auto = !peer_access || elastic_kv;
    t.host_exchange = host_exchange_env != "0" && (host_exchange_env == "1" || host_exchange_auto);
    const bool forced   = peer_overlap_env == "force";
    t.peer_lanes        = forced || (peer_overlap_env != "0" && peer_access && !t.host_exchange);
    t.peer_lanes_direct = t.peer_lanes && peer_access;
    return t;
}

} // namespace ninfer::models::qwen3_5::detail
