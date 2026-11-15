// The tp 2 eager transport decision (models/qwen3_5/program/tp2/eager_transport.h), host only.
// The property that matters for boards without P2P: for every input without peer access and
// without NINFER_TP_PEER_OVERLAP=force, no PeerLanes are attached and the HostExchange decision is
// the reference rule below, so their eager path is the HostExchange alone.

#include "models/qwen3_5/program/tp2/eager_transport.h"

#include <iostream>
#include <string>
#include <string_view>

namespace {

using ninfer::models::qwen3_5::detail::tp2_eager_transport;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// The HostExchange rule, written independently: without peer access or with elastic KV, unless
// NINFER_TP_HOST_EXCHANGE decides.
bool reference_host_exchange(bool peer_access, bool elastic_kv, std::string_view forced) {
    const bool host_exchange_auto = !peer_access || elastic_kv;
    return forced != "0" && (forced == "1" || host_exchange_auto);
}

} // namespace

int main() {
    int failures                          = 0;
    const std::string_view host_envs[]    = {"", "0", "1", "2", "on"};
    const std::string_view overlap_envs[] = {"", "0", "1", "auto", "force", "FORCE"};
    for (const bool peer : {false, true}) {
        for (const bool elastic : {false, true}) {
            for (const std::string_view host : host_envs) {
                for (const std::string_view overlap : overlap_envs) {
                    const auto t = tp2_eager_transport(peer, elastic, host, overlap);
                    const std::string label =
                        "peer=" + std::to_string(peer) + " elastic=" + std::to_string(elastic) +
                        " host='" + std::string(host) + "' overlap='" + std::string(overlap) + "'";
                    // 1. The HostExchange decision follows the reference rule for every input.
                    failures +=
                        check(t.host_exchange == reference_host_exchange(peer, elastic, host),
                              label + ": host exchange differs from the reference rule");
                    // 2. Without P2P and without force: no lanes (the boards without P2P keep
                    //    the HostExchange alone).
                    if (!peer && overlap != "force") {
                        failures += check(!t.peer_lanes, label + ": lanes attached without P2P");
                    }
                    // 3. Lanes are direct exactly when there is peer access.
                    failures += check(t.peer_lanes_direct == (t.peer_lanes && peer),
                                      label + ": direct flag");
                    // 4. force always attaches them; =0 never (unless force).
                    if (overlap == "force") {
                        failures += check(t.peer_lanes, label + ": force without lanes");
                    }
                    if (overlap == "0") {
                        failures += check(!t.peer_lanes, label + ": =0 with lanes");
                    }
                    // 5. Outside force, lanes never coexist with a HostExchange (the host lanes
                    //    keep serving the split-phase sums wherever there is one).
                    if (overlap != "force") {
                        failures += check(!(t.peer_lanes && t.host_exchange),
                                          label + ": lanes next to a HostExchange");
                    }
                }
            }
        }
    }
    // 6. The issue #3 case: P2P, no elastic KV, no switches -> direct lanes, no HostExchange.
    {
        const auto t = tp2_eager_transport(true, false, "", "");
        failures += check(!t.host_exchange && t.peer_lanes && t.peer_lanes_direct,
                          "P2P 27B default: direct peer lanes");
    }
    // 7. Our boards (no P2P), defaults: HostExchange, no lanes.
    {
        const auto t = tp2_eager_transport(false, false, "", "");
        failures += check(t.host_exchange && !t.peer_lanes, "no-P2P default: host exchange only");
    }
    // 8. The test mode on our boards: HostExchange kept, staged lanes next to it.
    {
        const auto t = tp2_eager_transport(false, false, "", "force");
        failures += check(t.host_exchange && t.peer_lanes && !t.peer_lanes_direct,
                          "no-P2P force: host exchange + driver-staged lanes");
    }
    // 9. Flash-Next on P2P (elastic KV): HostExchange, no lanes.
    {
        const auto t = tp2_eager_transport(true, true, "", "");
        failures += check(t.host_exchange && !t.peer_lanes, "P2P elastic: host exchange only");
    }
    static_assert(tp2_eager_transport(true, false, "", "").peer_lanes);
    static_assert(!tp2_eager_transport(false, false, "", "").peer_lanes);
    std::cout << (failures != 0 ? "FAIL" : "OK") << " tp2_eager_transport\n";
    return failures != 0 ? 1 : 0;
}
