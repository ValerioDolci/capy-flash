#include "models/inkling2/program/drafter_costs.h"

#include <algorithm>
#include <cmath>

namespace ninfer::models::inkling2::detail {
namespace {

[[nodiscard]] double value(const tuning::CostKnot& knot, StepKind kind) noexcept {
    switch (kind) {
    case StepKind::RoundMtp:
        return knot.round_mtp;
    case StepKind::RoundDFlash2:
        return knot.round_dflash2;
    case StepKind::Ordinary:
        break;
    }
    return knot.step;
}

// The segment of the table around `position`: its first knot (the last segment beyond the table).
[[nodiscard]] std::size_t segment(std::uint32_t position) noexcept {
    std::size_t i = 0;
    while (i + 2 < tuning::kCosts.size() && position >= tuning::kCosts[i + 1].context) { ++i; }
    return i;
}

// Where `position` lies in segment i: 0 at its first knot, 1 at its second (clamped below the
// table, extrapolated beyond it).
[[nodiscard]] double fraction(std::size_t i, std::uint32_t position) noexcept {
    const double a = tuning::kCosts[i].context, b = tuning::kCosts[i + 1].context;
    return std::max(0.0, (double(position) - a) / (b - a));
}

[[nodiscard]] std::size_t nearest(std::uint32_t position) noexcept {
    const double at  = std::log(double(std::max<std::uint32_t>(position, 1)));
    std::size_t best = 0;
    double best_dist = 1e300;
    for (std::size_t i = 0; i < tuning::kCosts.size(); ++i) {
        const double d = std::abs(std::log(double(tuning::kCosts[i].context)) - at);
        if (d < best_dist) {
            best      = i;
            best_dist = d;
        }
    }
    return best;
}

} // namespace

double StepCosts::table(StepKind kind, std::uint32_t position) noexcept {
    const std::size_t i = segment(position);
    const double t      = fraction(i, position);
    const double a = value(tuning::kCosts[i], kind), b = value(tuning::kCosts[i + 1], kind);
    return a + (b - a) * t;
}

double StepCosts::kind_scale(std::size_t kind) const noexcept {
    if (kind_seen_[kind]) { return kind_[kind]; }
    double sum        = 0.0;
    std::size_t count = 0;
    for (std::size_t k = 0; k < kKinds; ++k) {
        if (kind_seen_[k]) {
            sum += kind_[k];
            ++count;
        }
    }
    return count > 0 ? sum / double(count) : 1.0;
}

double StepCosts::knot_scale(std::size_t kind, std::size_t knot) const noexcept {
    return knot_seen_[kind][knot] ? knot_[kind][knot] : kind_scale(kind);
}

double StepCosts::ms(StepKind kind, std::uint32_t position) const noexcept {
    const auto k        = static_cast<std::size_t>(kind);
    const std::size_t i = segment(position);
    const double t      = std::min(fraction(i, position), 1.0);
    const double scale  = knot_scale(k, i) + (knot_scale(k, i + 1) - knot_scale(k, i)) * t;
    return table(kind, position) * scale;
}

void StepCosts::observe(StepKind kind, std::uint32_t position, double ms) noexcept {
    if (!(ms > 0.0)) { return; }
    const auto k        = static_cast<std::size_t>(kind);
    const double now    = this->ms(kind, position);
    const double sample = std::clamp(ms, now / kClamp, now * kClamp) / table(kind, position);
    const std::size_t i = nearest(position);
    const double knot   = knot_scale(k, i);
    knot_[k][i]         = knot + kKnotWeight * (sample - knot);
    knot_seen_[k][i]    = true;
    const double whole  = kind_scale(k);
    kind_[k]            = whole + kKindWeight * (sample - whole);
    kind_seen_[k]       = true;
}

void StepCosts::calibrate(StepKind kind, std::uint32_t position, double ms) noexcept {
    if (!(ms > 0.0)) { return; }
    const auto k  = static_cast<std::size_t>(kind);
    kind_[k]      = ms / table(kind, position);
    kind_seen_[k] = true;
}

} // namespace ninfer::models::inkling2::detail
