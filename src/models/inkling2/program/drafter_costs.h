#pragma once

// The cost of each kind of decode step of --spec auto as the engine measures it while it runs: a
// round of either drafter and a fed ordinary step, by context. The shape over the context comes
// from the generated table (tuning::kCosts); one scale per kind and table knot follows the measured
// times (an exponential mean of measured / table, kept at the knot nearest in log context), one
// scale per kind covers the knots it has not seen yet, and a kind never measured takes the mean
// scale of the others. A different board, or faster kernels for one kind of step, shows up within
// tens of steps. Host only; the times are the Program's per-row wall times, which end after the
// execution's own synchronization.

#include "models/inkling2/program/drafter_tuning.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::inkling2::detail {

enum class StepKind : std::uint8_t { RoundMtp, RoundDFlash2, Ordinary };

class StepCosts {
public:
    // The weight of one measurement at its knot and in its kind's scale; measurements further than
    // kClamp times the current estimate count as kClamp times it.
    static constexpr double kKnotWeight = 0.05;
    static constexpr double kKindWeight = 0.02;
    static constexpr double kClamp      = 3.0;

    // Milliseconds of a step of `kind` at context `position`.
    [[nodiscard]] double ms(StepKind kind, std::uint32_t position) const noexcept;
    // A step of `kind` at `position` took `ms`.
    void observe(StepKind kind, std::uint32_t position, double ms) noexcept;
    // Before any measurement (the execution's startup timing): `kind` costs `ms` at `position`.
    void calibrate(StepKind kind, std::uint32_t position, double ms) noexcept;

private:
    static constexpr std::size_t kKnots = tuning::kCosts.size();
    static constexpr std::size_t kKinds = 3;

    [[nodiscard]] static double table(StepKind kind, std::uint32_t position) noexcept;
    [[nodiscard]] double kind_scale(std::size_t kind) const noexcept;
    [[nodiscard]] double knot_scale(std::size_t kind, std::size_t knot) const noexcept;

    std::array<std::array<double, kKnots>, kKinds> knot_{};
    std::array<std::array<bool, kKnots>, kKinds> knot_seen_{};
    std::array<double, kKinds> kind_{1.0, 1.0, 1.0};
    std::array<bool, kKinds> kind_seen_{};
};

} // namespace ninfer::models::inkling2::detail
