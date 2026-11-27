// Replays --spec auto's choice (models/inkling2/program/drafter_choice.h) over a recorded trace,
// for tools/inkling2_tune/parity.py: the generated cost table without measurements, the rounds'
// accepted drafts from the trace. Reads one JSON object {seq, n_prompt, stopped, k, drafts: {dsp,
// df2}} (drafts[d][i]: drafter d's proposals at anchor n_prompt + i) and prints one token per step:
// M<a> / D<a> for a round of DSpark / DFlash2 that accepted a drafts, - for an ordinary step.
// Usage: ninfer_inkling2_drafter_replay TRACE.json

#include "models/inkling2/program/drafter_choice.h"
#include "models/inkling2/program/drafter_costs.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    using ninfer::SpeculativeBackend;
    using ninfer::TokenId;
    namespace ink = ninfer::models::inkling2::detail;
    if (argc != 2) {
        std::cerr << "usage: ninfer_inkling2_drafter_replay TRACE.json\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    const auto trace                                  = nlohmann::json::parse(in);
    const auto seq                                    = trace.at("seq").get<std::vector<TokenId>>();
    const auto n                                      = trace.at("n_prompt").get<std::uint32_t>();
    const bool stop                                   = trace.at("stopped").get<bool>();
    const auto K                                      = trace.at("k").get<std::uint32_t>();
    const std::vector<std::vector<TokenId>> drafts[2] = {
        trace.at("drafts").at("dsp").get<std::vector<std::vector<TokenId>>>(),
        trace.at("drafts").at("df2").get<std::vector<std::vector<TokenId>>>()};
    const auto N = static_cast<std::uint32_t>(seq.size());
    const ink::StepCosts table;
    ink::DrafterChoice choice;
    choice.start(n);
    std::string out;
    for (std::uint32_t p = n; p + 1 < N;) {
        const SpeculativeBackend chosen =
            choice.choose(std::span<const TokenId>(seq.data(), p + 1U), table);
        if (chosen == SpeculativeBackend::None) {
            out += "- ";
            ++p;
            continue;
        }
        const std::size_t d        = chosen == SpeculativeBackend::Mtp ? 0 : 1;
        const std::uint32_t limit  = N - p - (stop ? 1U : 2U);
        const std::uint32_t extent = std::min(K, limit);
        const auto& proposals      = drafts[d].at(p - n);
        std::uint32_t a            = 0;
        while (a < extent && proposals.at(a) == seq[p + 1 + a]) { ++a; }
        choice.observe(chosen, a, p, table);
        out += (d == 0 ? "M" : "D") + std::to_string(a) + " ";
        p += a + 1;
    }
    std::printf("%s\n", out.c_str());
    return 0;
}
