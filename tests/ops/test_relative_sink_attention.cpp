#include "ninfer/ops/relative_sink_attention.h"
#include "ninfer/ops/sigmoid_mul.h"

#include "core/device.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// BF16 output rounding alone contributes about 2^-9 relative RMS; the rest of the budget is FP32
// accumulation against the FP64 oracle. The contract lets a semantic BF16 rounding point whose
// exact value lies within FP32 error of a midpoint go either way, and one such logit can move a
// probability by a whole BF16 step of the logit. For every output element the oracle bounds the
// effect of its ambiguous logits (including those reached through an ambiguous prepared q/r/k
// element) and adds that allowance to its pointwise bound; the normwise bound is unchanged.
constexpr ReductionCriterion kCriterion{
    /*relative_l2*/ 4.0e-3,
    /*gross_absolute*/ 1.0e-2,
    /*gross_relative_to_max_reference*/ 5.0e-3,
};

// Relative reach of an FP32 evaluation against the exact value: a sum of depth <= 16 (per-lane
// partial sums and a warp butterfly) is within 16 * 2^-24 of the sum of absolute terms, and a few
// correctly rounded operations stay below the same bound.
constexpr double kAccumulationMargin = 0x1p-20;
constexpr double kOperationMargin    = 0x1p-20;

constexpr int kLanes       = 2; // the call uses lane / table row 1; row 0 must stay untouched
constexpr int kPage        = kPagedKVPageSize;
constexpr int kSpareFrames = 2;

double bf16_round(double x) {
    if (x == 0.0 || !std::isfinite(x)) return x;
    int exponent = 0;
    (void)std::frexp(x, &exponent);
    return std::ldexp(std::nearbyint(std::ldexp(x, 8 - exponent)), exponent - 8);
}

// Distance from |x| to the nearest BF16 rounding midpoint, and the BF16 step at x.
double midpoint_distance(double x) {
    const double a = std::abs(x);
    if (a == 0.0) return std::numeric_limits<double>::infinity();
    int exponent = 0;
    (void)std::frexp(a, &exponent);
    const double step = std::ldexp(1.0, exponent - 8);
    return std::abs(a - (std::floor(a / step) + 0.5) * step);
}

double bf16_step(double x) {
    if (x == 0.0) return 0.0;
    int exponent = 0;
    (void)std::frexp(x, &exponent);
    return std::ldexp(1.0, exponent - 8);
}

double fp16_round(double x) {
    return static_cast<double>(__half2float(__float2half_rn(static_cast<float>(x))));
}

std::uint16_t fp16_bits(double x) {
    const __half h = __float2half_rn(static_cast<float>(x));
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

double fp16_value(std::uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return static_cast<double>(__half2float(h));
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Per-head magnitudes so that some head vectors are clipped and some are not.
void scale_heads(std::vector<std::uint16_t>& bits, int width, int heads, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> factor(0.25f, 3.0f);
    std::vector<float> factors(heads);
    for (auto& f : factors) f = factor(generator);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const int head = static_cast<int>((i / width) % heads);
        bits[i]        = f32_to_bf16(bf16_to_f32(bits[i]) * factors[head]);
    }
}

void parallel_for(int count, const std::function<void(int)>& body) {
    const int workers = std::max(1, std::min<int>(count, std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    for (int w = 0; w < workers; ++w)
        threads.emplace_back([&, w] {
            for (int i = w; i < count; i += workers) body(i);
        });
    for (auto& thread : threads) thread.join();
}

enum class Mode { Windowed, Paged };

struct Config {
    std::string name;
    Mode mode;
    ops::AttentionHeadGeometry geometry;
    int relative_dim;
    int relative_extent;
    int window;   // windowed
    int capacity; // windowed
    float alpha;
    float floor;
    int max_length;
};

struct Totals {
    int cases            = 0;
    double max_abs_error = 0.0;
    double max_abs_clear = 0.0;
    double max_allowance = 0.0;
    double max_rel_l2    = 0.0;
    long flips           = 0;
    long loose_elements  = 0;
    long elements        = 0;
};

class Harness {
public:
    Harness(Config config, std::uint32_t seed, Totals& totals)
        : c_(std::move(config)), totals_(totals), D_(c_.geometry.head_dim),
          Hq_(c_.geometry.query_heads), Hkv_(c_.geometry.kv_heads), dr_(c_.relative_dim),
          R_(c_.relative_extent), seed_(seed) {
        params_.relative_dim    = dr_;
        params_.relative_extent = R_;
        params_.qk_clip         = 22.627416997969522f;
        params_.relative_clip   = 8.0f;
        params_.clip_epsilon    = 1e-6f;
        params_.value_epsilon   = 1e-6f;
        params_.scale           = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D_)));
        params_.log_scale_alpha = c_.alpha;
        params_.log_scale_floor = c_.floor;

        q_scale_    = random_bf16(static_cast<std::size_t>(D_) * Hq_, seed + 1, 0.5f, 2.0f);
        r_scale_    = random_bf16(static_cast<std::size_t>(dr_) * Hq_, seed + 2, 0.5f, 2.0f);
        rel_proj_   = random_bf16(static_cast<std::size_t>(R_) * dr_, seed + 3, -1.0f, 1.0f);
        d_q_scale_  = to_device(q_scale_);
        d_r_scale_  = to_device(r_scale_);
        d_rel_proj_ = to_device(rel_proj_);

        const std::size_t rows = static_cast<std::size_t>(c_.max_length) * Hkv_ * D_;
        K_.assign(rows, std::numeric_limits<double>::quiet_NaN());
        K_slack_.assign(rows, 0.0);
        V_.assign(rows, std::numeric_limits<double>::quiet_NaN());
        Kbits_.assign(rows, 0xffff);
        Vbits_.assign(rows, 0xffff);

        if (c_.mode == Mode::Windowed) {
            padded_      = (c_.capacity + 127) / 128 * 128;
            cache_elems_ = static_cast<std::size_t>(D_) * padded_ * Hkv_ * kLanes;
        } else {
            logical_pages_ = (c_.max_length + kPage - 1) / kPage;
            frames_        = logical_pages_ + kSpareFrames;
            cache_elems_   = static_cast<std::size_t>(D_) * kPage * Hkv_ * frames_;
            // Row 1 maps logical page b to a scattered physical frame; row 0 is never used.
            std::vector<int> order(frames_);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), std::mt19937(seed + 4));
            table_.assign(static_cast<std::size_t>(logical_pages_) * kLanes, -1);
            for (int b = 0; b < logical_pages_; ++b) table_[b + logical_pages_] = order[b];
            d_table_ = to_device(table_);
        }
        cache_k_ = std::make_unique<GuardedDeviceBuffer>(cache_elems_ * 2);
        cache_v_ = std::make_unique<GuardedDeviceBuffer>(cache_elems_ * 2);
        cache_k_->fill(0xff);
        cache_v_->fill(0xff);
    }

    const Config& config() const { return c_; }

    // Synthetic committed history: rows [first, length) prepared by the oracle prologue.
    void commit_history(int length) {
        const int first = c_.mode == Mode::Windowed ? std::max(0, length - c_.capacity) : 0;
        const int count = length - first;
        if (count <= 0) {
            length_ = length;
            return;
        }
        auto k = random_bf16(static_cast<std::size_t>(count) * Hkv_ * D_, seed_ + 11, -2.0f, 2.0f);
        auto v = random_bf16(k.size(), seed_ + 12, -2.0f, 2.0f);
        scale_heads(k, D_, Hkv_, seed_ + 13);
        for (int i = 0; i < count; ++i)
            for (int g = 0; g < Hkv_; ++g) {
                const std::size_t src = (static_cast<std::size_t>(i) * Hkv_ + g) * D_;
                prepare_kv(&k[src], &v[src], first + i, g);
            }
        std::vector<std::uint16_t> host_k(cache_elems_, 0xffff), host_v(cache_elems_, 0xffff);
        for (int x = first; x < length; ++x)
            for (int g = 0; g < Hkv_; ++g)
                for (int d = 0; d < D_; ++d) {
                    const std::size_t at = cache_index(x, g, d);
                    host_k[at]           = Kbits_[row(x, g) + d];
                    host_v[at]           = Vbits_[row(x, g) + d];
                }
        cache_k_->copy_from_host(host_k.data(), host_k.size() * 2);
        cache_v_->copy_from_host(host_v.data(), host_v.size() * 2);
        length_    = length;
        committed_ = first;
    }

    // One call of T tokens at the current length. With tokenwise, the same block is also run
    // one token per call on a copy of the cache: outputs and final caches must be identical.
    int run_block(int T, bool tokenwise) {
        const int L = length_;
        if (L + T > c_.max_length) throw std::logic_error("test sequence exceeds max_length");
        const std::uint32_t seed = seed_ + 1000U * static_cast<std::uint32_t>(++blocks_);
        auto q    = random_bf16(static_cast<std::size_t>(T) * Hq_ * D_, seed + 1, -2.0f, 2.0f);
        auto k    = random_bf16(static_cast<std::size_t>(T) * Hkv_ * D_, seed + 2, -2.0f, 2.0f);
        auto v    = random_bf16(k.size(), seed + 3, -2.0f, 2.0f);
        auto r    = random_bf16(static_cast<std::size_t>(T) * Hq_ * dr_, seed + 4, -2.0f, 2.0f);
        auto sink = random_bf16(static_cast<std::size_t>(T) * Hq_, seed + 5, -2.0f, 4.0f);
        scale_heads(q, D_, Hq_, seed + 6);
        scale_heads(k, D_, Hkv_, seed + 7);
        scale_heads(r, dr_, Hq_, seed + 8);
        std::vector<int> positions(T);
        std::iota(positions.begin(), positions.end(), L);
        const std::vector<int> rows{1};

        // Oracle: prepare the block, append it to the oracle history, then attend.
        std::vector<double> qp(q.size()), rp(r.size()), q_slack(q.size()), r_slack(r.size());
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < Hq_; ++h) {
                const std::size_t qi = (static_cast<std::size_t>(t) * Hq_ + h) * D_;
                const std::size_t ri = (static_cast<std::size_t>(t) * Hq_ + h) * dr_;
                prepare_scaled(&q[qi], &q_scale_[static_cast<std::size_t>(h) * D_], D_,
                               params_.qk_clip, L + t, &qp[qi], &q_slack[qi]);
                prepare_scaled(&r[ri], &r_scale_[static_cast<std::size_t>(h) * dr_], dr_,
                               params_.relative_clip, L + t, &rp[ri], &r_slack[ri]);
            }
            for (int g = 0; g < Hkv_; ++g) {
                const std::size_t src = (static_cast<std::size_t>(t) * Hkv_ + g) * D_;
                prepare_kv(&k[src], &v[src], L + t, g);
            }
        }
        std::vector<double> expected(q.size());
        std::vector<double> allowance(q.size(), 0.0);
        parallel_for(T * Hq_, [&](int item) {
            attend(item / Hq_, item % Hq_, L, qp, rp, q_slack, r_slack, sink, expected, allowance);
        });

        // Device call.
        auto d_q = to_device(q), d_k = to_device(k), d_v = to_device(v), d_r = to_device(r),
             d_sink = to_device(sink), d_pos = to_device(positions), d_rows = to_device(rows);
        GuardedDeviceBuffer out(q.size() * 2);
        out.fill(0xff);
        const auto capacity =
            ops::relative_sink_attention_workspace_capacity_bytes(c_.geometry, dr_, 1, T, T);
        GuardedDeviceBuffer scratch(capacity);
        scratch.fill(0xa7);

        // Copies of the cache before the call: one for the token-by-token run, one to repeat the
        // same call (equal calls, equal bits).
        const auto copy_cache = [&](std::unique_ptr<GuardedDeviceBuffer>& k_copy,
                                    std::unique_ptr<GuardedDeviceBuffer>& v_copy) {
            k_copy = std::make_unique<GuardedDeviceBuffer>(cache_elems_ * 2);
            v_copy = std::make_unique<GuardedDeviceBuffer>(cache_elems_ * 2);
            cuda_check(cudaMemcpy(k_copy->data(), cache_k_->data(), cache_elems_ * 2,
                                  cudaMemcpyDeviceToDevice),
                       "copy cache K");
            cuda_check(cudaMemcpy(v_copy->data(), cache_v_->data(), cache_elems_ * 2,
                                  cudaMemcpyDeviceToDevice),
                       "copy cache V");
            cuda_synchronize();
        };
        std::unique_ptr<GuardedDeviceBuffer> shadow_k, shadow_v, repeat_k, repeat_v;
        if (tokenwise) copy_cache(shadow_k, shadow_v);
        copy_cache(repeat_k, repeat_v);

        DeviceContext device;
        int failures = 0;
        {
            DeviceArena workspace(DeviceSpan{scratch.data(), capacity});
            call(d_q.p, d_k.p, d_v.p, d_r.p, d_sink.p, d_pos.p, d_rows.p, T, cache_k_->data(),
                 cache_v_->data(), workspace, out.data(), device.stream);
            cuda_synchronize(device.stream);
            if (workspace.used() != 0 || workspace.peak_used() != capacity) {
                std::cerr << label(L, T) << ": workspace query/scope mismatch\n";
                ++failures;
            }
        }
        // Checks one output against the oracle; returns 0 or 1. Pointwise error is taken beyond
        // each element's allowance for ambiguous logits; elements whose allowance exceeds the
        // pointwise limit are effectively checked only normwise.
        const auto check = [&](const std::string& name, const void* device_out) {
            const auto got             = from_device_bf16(device_out, q.size());
            const ReductionStats stats = compute_reduction_stats(
                got.data(), expected.data(), static_cast<std::int64_t>(got.size()));
            const double limit       = gross_error_limit(stats, kCriterion);
            double excess            = 0.0;
            double largest_allowance = 0.0;
            std::size_t worst        = 0;
            long flagged             = 0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                const double beyond = std::abs(got[i] - expected[i]) - allowance[i];
                if (beyond > excess) {
                    excess = beyond;
                    worst  = i;
                }
                flagged += allowance[i] > limit ? 1 : 0;
                largest_allowance = std::max(largest_allowance, allowance[i]);
            }
            const bool pass = stats.first_non_finite < 0 &&
                              stats.relative_l2 <= kCriterion.relative_l2 && excess <= limit;
            std::cout << name << ": rel_l2=" << stats.relative_l2
                      << " max_abs=" << stats.maximum_absolute_error
                      << " max_beyond_allowance=" << excess
                      << " max_ref=" << stats.maximum_absolute_reference
                      << " loose_elements=" << flagged << "/" << allowance.size()
                      << " max_allowance=" << largest_allowance << (pass ? "" : " FAIL") << '\n';
            if (!pass)
                std::cerr << name << ": criterion failed at " << worst << " actual=" << got[worst]
                          << " reference=" << expected[worst] << " allowance=" << allowance[worst]
                          << " non_finite=" << stats.first_non_finite << '\n';
            ++totals_.cases;
            totals_.max_abs_error = std::max(totals_.max_abs_error, stats.maximum_absolute_error);
            totals_.max_abs_clear = std::max(totals_.max_abs_clear, excess);
            totals_.max_allowance = std::max(totals_.max_allowance, largest_allowance);
            totals_.max_rel_l2    = std::max(totals_.max_rel_l2, stats.relative_l2);
            totals_.loose_elements += flagged;
            totals_.elements += static_cast<long>(allowance.size());
            return pass ? 0 : 1;
        };
        failures += check(label(L, T), out.data());
        {
            // The same call again, gated: equal to the first call's out after sigmoid_mul.
            const auto gate = random_bf16(q.size(), seed + 9, -6.0f, 6.0f);
            auto d_gate     = to_device(gate);
            cuda_synchronize(); // the pageable upload may still be in flight
            GuardedDeviceBuffer again(q.size() * 2), expected_gated(q.size() * 2);
            // On the Op's stream: a device-to-device cudaMemcpy runs on the legacy stream without
            // waiting on the host, and sigmoid_mul could read the copy before it lands.
            cuda_check(cudaMemcpyAsync(expected_gated.data(), out.data(), q.size() * 2,
                                       cudaMemcpyDeviceToDevice, device.stream),
                       "copy out");
            Tensor tg(d_gate.p, DType::BF16, {static_cast<std::int32_t>(q.size())}),
                te(expected_gated.data(), DType::BF16, {static_cast<std::int32_t>(q.size())});
            ops::sigmoid_mul(tg, te, device.stream);
            DeviceArena workspace(DeviceSpan{scratch.data(), capacity});
            call(d_q.p, d_k.p, d_v.p, d_r.p, d_sink.p, d_pos.p, d_rows.p, T, repeat_k->data(),
                 repeat_v->data(), workspace, again.data(), device.stream, d_gate.p);
            cuda_synchronize(device.stream);
            failures += verify_exact((label(L, T) + " repeated gated call out").c_str(),
                                     from_device<std::uint16_t>(again.data(), q.size()),
                                     from_device<std::uint16_t>(expected_gated.data(), q.size()));
            failures += verify_exact((label(L, T) + " repeated call cache K").c_str(),
                                     from_device<std::uint16_t>(repeat_k->data(), cache_elems_),
                                     from_device<std::uint16_t>(cache_k_->data(), cache_elems_));
            failures += verify_exact((label(L, T) + " repeated call cache V").c_str(),
                                     from_device<std::uint16_t>(repeat_v->data(), cache_elems_),
                                     from_device<std::uint16_t>(cache_v_->data(), cache_elems_));
        }
        failures += out.verify_guards(label(L, T) + " out") +
                    scratch.verify_guards(label(L, T) + " scratch");
        failures += verify_exact((label(L, T) + " q unchanged").c_str(),
                                 from_device<std::uint16_t>(d_q, q.size()), q);
        failures += verify_exact((label(L, T) + " v unchanged").c_str(),
                                 from_device<std::uint16_t>(d_v, v.size()), v);

        if (tokenwise) {
            GuardedDeviceBuffer single_out(q.size() * 2);
            single_out.fill(0xff);
            const auto single_capacity =
                ops::relative_sink_attention_workspace_capacity_bytes(c_.geometry, dr_, 1, 1, 1);
            GuardedDeviceBuffer single_scratch(single_capacity);
            for (int t = 0; t < T; ++t) {
                DeviceArena workspace(DeviceSpan{single_scratch.data(), single_capacity});
                const auto at = [](void* base, std::size_t elements, std::size_t bytes) {
                    return static_cast<void*>(static_cast<std::uint8_t*>(base) + elements * bytes);
                };
                call(at(d_q.p, static_cast<std::size_t>(t) * Hq_ * D_, 2),
                     at(d_k.p, static_cast<std::size_t>(t) * Hkv_ * D_, 2),
                     at(d_v.p, static_cast<std::size_t>(t) * Hkv_ * D_, 2),
                     at(d_r.p, static_cast<std::size_t>(t) * Hq_ * dr_, 2),
                     at(d_sink.p, static_cast<std::size_t>(t) * Hq_, 2), at(d_pos.p, t, 4),
                     d_rows.p, 1, shadow_k->data(), shadow_v->data(), workspace,
                     at(single_out.data(), static_cast<std::size_t>(t) * Hq_ * D_, 2),
                     device.stream);
            }
            cuda_synchronize(device.stream);
            // Calls of at most 16 tokens per row (and D > 128) share the decode route and its bits;
            // longer calls take the tensor-core route, equal to decode within the oracle criterion.
            if (T <= 16 || D_ > 128)
                failures += verify_exact((label(L, T) + " tokenwise out").c_str(),
                                         from_device<std::uint16_t>(single_out.data(), q.size()),
                                         from_device<std::uint16_t>(out.data(), q.size()));
            else
                failures += check(label(L, T) + " tokenwise", single_out.data());
            failures += verify_exact((label(L, T) + " tokenwise cache K").c_str(),
                                     from_device<std::uint16_t>(shadow_k->data(), cache_elems_),
                                     from_device<std::uint16_t>(cache_k_->data(), cache_elems_));
            failures += verify_exact((label(L, T) + " tokenwise cache V").c_str(),
                                     from_device<std::uint16_t>(shadow_v->data(), cache_elems_),
                                     from_device<std::uint16_t>(cache_v_->data(), cache_elems_));
            std::cout << label(L, T) << ": tokenwise "
                      << (T <= 16 || D_ > 128 ? "identical" : "within criterion") << "="
                      << (failures == 0) << '\n';
        }
        length_ = L + T;
        return failures;
    }

    // Final state: every retained row matches the oracle within one BF16 step, every slot or
    // frame that no retained position owns is untouched.
    int verify_cache() {
        const auto k = from_device<std::uint16_t>(cache_k_->data(), cache_elems_);
        const auto v = from_device<std::uint16_t>(cache_v_->data(), cache_elems_);
        std::vector<char> owned(cache_elems_, 0);
        const int first = c_.mode == Mode::Windowed ? std::max(0, length_ - c_.capacity) : 0;
        int bad         = 0;
        long flip       = 0;
        for (int x = first; x < length_; ++x)
            for (int g = 0; g < Hkv_; ++g)
                for (int d = 0; d < D_; ++d) {
                    const std::size_t at = cache_index(x, g, d);
                    owned[at]            = 1;
                    const double ek = K_[row(x, g) + d], ev = V_[row(x, g) + d];
                    const double gk = bf16_to_f32(k[at]), gv = fp16_value(v[at]);
                    if (k[at] != Kbits_[row(x, g) + d] || v[at] != Vbits_[row(x, g) + d]) ++flip;
                    if (!(std::abs(gk - ek) <= std::abs(ek) * 0x1p-7) ||
                        !(std::abs(gv - ev) <= std::abs(ev) * 0x1p-7 + 0x1p-24))
                        ++bad;
                }
        for (std::size_t i = 0; i < cache_elems_; ++i)
            if (!owned[i] && (k[i] != 0xffff || v[i] != 0xffff)) ++bad;
        std::cout << c_.name << ": cache rows checked, differing bits=" << flip
                  << " out-of-tolerance=" << bad << '\n';
        totals_.flips += flip;
        int failures = bad != 0 ? 1 : 0;
        failures += cache_k_->verify_guards(c_.name + " cache K") +
                    cache_v_->verify_guards(c_.name + " cache V");
        return failures;
    }

private:
    std::size_t row(int position, int g) const {
        return (static_cast<std::size_t>(position) * Hkv_ + g) * D_;
    }

    std::size_t cache_index(int position, int g, int d) const {
        if (c_.mode == Mode::Windowed) {
            const int slot = position % c_.capacity;
            return static_cast<std::size_t>(d) +
                   static_cast<std::size_t>(D_) *
                       (slot + static_cast<std::size_t>(padded_) *
                                   (g + static_cast<std::size_t>(Hkv_) * 1));
        }
        const int frame = table_[position / kPage + logical_pages_];
        return static_cast<std::size_t>(d) +
               static_cast<std::size_t>(D_) *
                   (position % kPage +
                    static_cast<std::size_t>(kPage) * (g + static_cast<std::size_t>(Hkv_) * frame));
    }

    double tau(int position) const {
        return 1.0 + static_cast<double>(c_.alpha) *
                         std::log(std::max((position + 1.0) / static_cast<double>(c_.floor), 1.0));
    }

    // A prepared element whose rounding is ambiguous carries one BF16 step of slack.
    void prepare_scaled(const std::uint16_t* x, const std::uint16_t* scale, int n, double threshold,
                        int position, double* out, double* slack) const {
        double squares = 0.0;
        std::vector<double> scaled(n);
        for (int i = 0; i < n; ++i) {
            scaled[i] = static_cast<double>(bf16_to_f32(x[i])) * bf16_to_f32(scale[i]);
            squares += scaled[i] * scaled[i];
        }
        const double factor = std::min(
            1.0, threshold / (std::sqrt(squares) + static_cast<double>(params_.clip_epsilon)));
        const double t = tau(position);
        for (int i = 0; i < n; ++i) {
            const double clipped = scaled[i] * factor;
            const double first   = bf16_round(clipped);
            out[i]               = bf16_round(first * t);
            const bool unsure =
                midpoint_distance(clipped) <= kOperationMargin * std::abs(clipped) ||
                (t != 1.0 &&
                 midpoint_distance(first * t) <= kOperationMargin * std::abs(first * t));
            slack[i] = unsure ? 2.0 * bf16_step(out[i]) : 0.0;
        }
    }

    void prepare_kv(const std::uint16_t* k, const std::uint16_t* v, int position, int g) {
        double k_squares = 0.0, v_squares = 0.0;
        for (int d = 0; d < D_; ++d) {
            const double kd = bf16_to_f32(k[d]), vd = bf16_to_f32(v[d]);
            k_squares += kd * kd;
            v_squares += vd * vd;
        }
        const double factor =
            std::min(1.0, static_cast<double>(params_.qk_clip) /
                              (std::sqrt(k_squares) + static_cast<double>(params_.clip_epsilon)));
        const double inverse =
            1.0 / std::sqrt(v_squares / D_ + static_cast<double>(params_.value_epsilon));
        for (int d = 0; d < D_; ++d) {
            const double clipped = bf16_to_f32(k[d]) * factor;
            const double kd      = bf16_round(clipped);
            K_slack_[row(position, g) + d] =
                midpoint_distance(clipped) <= kOperationMargin * std::abs(clipped) ? bf16_step(kd)
                                                                                   : 0.0;
            const double vd              = fp16_round(bf16_round(bf16_to_f32(v[d]) * inverse));
            K_[row(position, g) + d]     = kd;
            V_[row(position, g) + d]     = vd;
            Kbits_[row(position, g) + d] = f32_to_bf16(static_cast<float>(kd));
            Vbits_[row(position, g) + d] = fp16_bits(vd);
        }
    }

    // Also writes each output element's allowance: the first-order bound on how far it moves
    // when every logit that lies within FP32 (or upstream-slack) reach of a BF16 midpoint rounds
    // the other way, by one step at that and every later rounding point.
    void attend(int t, int h, int L, const std::vector<double>& qp, const std::vector<double>& rp,
                const std::vector<double>& q_slack, const std::vector<double>& r_slack,
                const std::vector<std::uint16_t>& sink, std::vector<double>& out,
                std::vector<double>& allowance) const {
        const int p      = L + t;
        const int g      = h / (Hq_ / Hkv_);
        const int lo     = c_.mode == Mode::Windowed ? std::max(0, p - c_.window + 1) : 0;
        const double* q  = &qp[(static_cast<std::size_t>(t) * Hq_ + h) * D_];
        const double* r  = &rp[(static_cast<std::size_t>(t) * Hq_ + h) * dr_];
        const double* qs = &q_slack[(static_cast<std::size_t>(t) * Hq_ + h) * D_];
        const double* rs = &r_slack[(static_cast<std::size_t>(t) * Hq_ + h) * dr_];
        std::vector<double> shift(p - lo + 1, 0.0);
        const auto near = [](double x, double margin) { return midpoint_distance(x) <= margin; };
        std::vector<double> logits(p - lo + 1);
        double m                = bf16_to_f32(sink[static_cast<std::size_t>(t) * Hq_ + h]);
        const double sink_logit = m;
        for (int s = lo; s <= p; ++s) {
            double dot = 0.0, magnitude = 0.0, reach = 0.0;
            for (int d = 0; d < D_; ++d) {
                const double kd = K_[row(s, g) + d];
                dot += q[d] * kd;
                magnitude += std::abs(q[d] * kd);
                reach += std::abs(q[d]) * K_slack_[row(s, g) + d] + std::abs(kd) * qs[d];
            }
            const double scaled = bf16_round(dot) * static_cast<double>(params_.scale);
            bool unsure         = near(dot, kAccumulationMargin * magnitude + reach) ||
                          near(scaled, kOperationMargin * std::abs(scaled));
            double score       = bf16_round(scaled);
            const int distance = p - s;
            if (distance < R_) {
                double bias = 0.0, bias_magnitude = 0.0, bias_reach = 0.0;
                for (int j = 0; j < dr_; ++j) {
                    const double pj =
                        bf16_to_f32(rel_proj_[distance + static_cast<std::size_t>(R_) * j]);
                    bias += r[j] * pj;
                    bias_magnitude += std::abs(r[j] * pj);
                    bias_reach += rs[j] * std::abs(pj);
                }
                const double rounded_bias = bf16_round(bias);
                const double sum          = score + rounded_bias;
                unsure = unsure || near(bias, kAccumulationMargin * bias_magnitude + bias_reach) ||
                         near(sum, kOperationMargin * std::abs(sum));
                if (unsure)
                    shift[s - lo] = bf16_step(score) + bf16_step(rounded_bias) + bf16_step(sum);
                score = bf16_round(sum);
            } else if (unsure) {
                shift[s - lo] = bf16_step(score);
            }
            logits[s - lo] = score;
            m              = std::max(m, score);
        }
        double z = std::exp(sink_logit - m);
        for (double x : logits) z += std::exp(x - m);
        const std::size_t base = (static_cast<std::size_t>(t) * Hq_ + h) * D_;
        double* o              = &out[base];
        std::fill(o, o + D_, 0.0);
        for (int s = lo; s <= p; ++s) {
            const double probability = bf16_round(std::exp(logits[s - lo] - m) / z);
            for (int d = 0; d < D_; ++d) o[d] += probability * V_[row(s, g) + d];
        }
        // Moving logit s by delta scales its weight by e^delta, so o moves by exactly
        // p_s (e^delta - 1) / (1 + p_s (e^delta - 1)) * (v_s - o).
        double* a = &allowance[base];
        std::fill(a, a + D_, 0.0);
        for (int s = lo; s <= p; ++s) {
            if (shift[s - lo] == 0.0) continue;
            const double ps = std::exp(logits[s - lo] - m) / z;
            double factor   = 0.0;
            for (const double delta : {shift[s - lo], -shift[s - lo]}) {
                const double grow = std::expm1(delta);
                factor            = std::max(factor, std::abs(ps * grow / (1.0 + ps * grow)));
            }
            for (int d = 0; d < D_; ++d) a[d] += factor * std::abs(V_[row(s, g) + d] - o[d]);
        }
    }

    void call(void* q, void* k, void* v, void* r, void* sink, void* positions, void* rows, int T,
              void* cache_k, void* cache_v, WorkspaceArena& workspace, void* out,
              cudaStream_t stream, void* gate = nullptr) {
        Tensor tq(q, DType::BF16, {D_, Hq_, T, 1}), tk(k, DType::BF16, {D_, Hkv_, T, 1}),
            tv(v, DType::BF16, {D_, Hkv_, T, 1}), tr(r, DType::BF16, {dr_, Hq_, T, 1}),
            ts(sink, DType::BF16, {Hq_, T, 1}), tp(positions, DType::I32, {T, 1}),
            trows(rows, DType::I32, {1}), to(out, DType::BF16, {D_, Hq_, T, 1}),
            tg(gate, DType::BF16, {D_, Hq_, T, 1});
        const Tensor* pg = gate != nullptr ? &tg : nullptr;
        Tensor qs(d_q_scale_.p, DType::BF16, {D_, Hq_}), rs(d_r_scale_.p, DType::BF16, {dr_, Hq_}),
            pr(d_rel_proj_.p, DType::BF16, {R_, dr_});
        if (c_.mode == Mode::Windowed) {
            const CyclicKVCacheLayerView view{
                .k               = Tensor(cache_k, DType::BF16, {D_, padded_, Hkv_, kLanes}),
                .v               = Tensor(cache_v, DType::FP16, {D_, padded_, Hkv_, kLanes}),
                .capacity        = static_cast<std::uint32_t>(c_.capacity),
                .padded_capacity = static_cast<std::uint32_t>(padded_),
                .num_kv_heads    = Hkv_,
                .head_dim        = D_,
                .lane_capacity   = kLanes};
            ops::relative_sink_attention_windowed(tq, tk, tv, tr, ts, tp, trows, qs, rs, pr,
                                                  c_.geometry, params_, c_.window, view, pg,
                                                  workspace, to, stream);
        } else {
            PagedKVBatchLayerView view;
            view.k_pages      = Tensor(cache_k, DType::BF16, {D_, kPage, Hkv_, frames_});
            view.v_pages      = Tensor(cache_v, DType::FP16, {D_, kPage, Hkv_, frames_});
            view.block_tables = Tensor(d_table_.p, DType::I32, {logical_pages_, kLanes});
            view.head_dim     = D_;
            view.num_kv_heads = Hkv_;
            view.storage      = KvCacheStorage::BFloat16;
            ops::relative_sink_attention_paged(tq, tk, tv, tr, ts, tp, trows, qs, rs, pr,
                                               c_.geometry, params_, view, pg, workspace, to,
                                               stream);
        }
        cuda_check_last_launch("relative_sink_attention");
    }

    std::string label(int L, int T) const {
        return c_.name + " L=" + std::to_string(L) + " T=" + std::to_string(T);
    }

    Config c_;
    Totals& totals_;
    int D_, Hq_, Hkv_, dr_, R_;
    std::uint32_t seed_;
    ops::RelativeSinkAttentionParams params_;
    std::vector<std::uint16_t> q_scale_, r_scale_, rel_proj_;
    DeviceBuffer d_q_scale_, d_r_scale_, d_rel_proj_, d_table_;
    std::vector<double> K_, V_, K_slack_;
    std::vector<std::uint16_t> Kbits_, Vbits_;
    std::vector<int> table_;
    int padded_ = 0, logical_pages_ = 0, frames_ = 0;
    std::size_t cache_elems_ = 0;
    std::unique_ptr<GuardedDeviceBuffer> cache_k_, cache_v_;
    int length_ = 0, committed_ = 0, blocks_ = 0;
};

int run_sequence(Harness& harness, int history, const std::vector<int>& blocks,
                 const std::vector<bool>& tokenwise) {
    harness.commit_history(history);
    int failures = 0;
    for (std::size_t i = 0; i < blocks.size(); ++i)
        failures += harness.run_block(blocks[i], i < tokenwise.size() && tokenwise[i]);
    return failures + harness.verify_cache();
}

int qualify(Totals& totals) {
    using G   = ops::AttentionHeadGeometry;
    int fails = 0;
    // Small shapes: the synthetic checkpoint geometry (6/2 heads, window 64, extent 96, floor 128)
    // and an edge ring of capacity window-1 with an extent shorter than the window.
    const std::vector<int> small{1, 4, 16, 17, 37, 64, 1, 1, 100, 3, 70};
    {
        Harness h(
            {"small-windowed", Mode::Windowed, G{128, 6, 2}, 16, 96, 64, 64, 0.0f, 128.0f, 400},
            101, totals);
        fails +=
            run_sequence(h, 0, small, {true, true, true, true, true, true, true, true, true, true});
    }
    {
        Harness h(
            {"small-windowed-edge", Mode::Windowed, G{64, 6, 2}, 16, 40, 64, 63, 0.0f, 128.0f, 400},
            202, totals);
        fails += run_sequence(h, 0, small, {false, false, false, true, true, false, false, true});
    }
    {
        Harness h({"small-paged", Mode::Paged, G{128, 6, 2}, 16, 96, 0, 0, 0.1f, 128.0f, 400}, 303,
                  totals);
        fails +=
            run_sequence(h, 0, small, {true, true, true, true, true, true, true, true, true, true});
    }
    // Real geometry: one device (24/8) and one tensor-parallel rank (12/4), D 128, dr 16.
    for (const G geometry : {G{128, 24, 8}, G{128, 12, 4}}) {
        const bool full = geometry.query_heads == 24;
        const auto heads =
            std::to_string(geometry.query_heads) + "/" + std::to_string(geometry.kv_heads);
        {
            // A wrapped ring before the call, then blocks that wrap it again.
            Harness h({"windowed-" + heads + "-wrapped", Mode::Windowed, geometry, 16, 1024, 1024,
                       1024, 0.0f, 8192.0f, 3100},
                      404 + geometry.query_heads, totals);
            fails += full ? run_sequence(h, 1500, {1, 4, 16, 512, 1024}, {true, true, true, true})
                          : run_sequence(h, 1500, {16, 512}, {true});
        }
        if (full) {
            // Prefill from empty with one block equal to the window, then past it.
            Harness h({"windowed-" + heads + "-prefill", Mode::Windowed, geometry, 16, 1024, 1024,
                       1024, 0.0f, 8192.0f, 1600},
                      505, totals);
            fails += run_sequence(h, 0, {1024, 512}, {});
        }
        {
            // Positions across the log-scale floor 8192, keys far beyond the bias extent.
            Harness h({"paged-" + heads + "-long", Mode::Paged, geometry, 16, 1024, 0, 0, 0.1f,
                       8192.0f, 8800},
                      606 + geometry.query_heads, totals);
            fails += full ? run_sequence(h, 8180, {1, 4, 16, 512}, {true, true, true})
                          : run_sequence(h, 8180, {16}, {true});
        }
        if (full) {
            // Prefill from empty, then a decode whose first key sits exactly at the extent.
            Harness h({"paged-" + heads + "-prefill", Mode::Paged, geometry, 16, 1024, 0, 0, 0.1f,
                       8192.0f, 1100},
                      707, totals);
            fails += run_sequence(h, 0, {1024, 1, 4}, {false, true, true});
        }
    }
    // The model's ring (window + 7 slots): one-token calls take the cluster route, longer blocks
    // the split route (up to 16 tokens) with the same bits.
    {
        Harness h({"windowed-12/4-ring1031", Mode::Windowed, G{128, 12, 4}, 16, 1024, 1024, 1031,
                   0.0f, 8192.0f, 2700},
                  808, totals);
        fails += run_sequence(h, 1500, {1, 2, 4, 8, 16, 7}, {true, true, true, true, true, true});
    }
    return fails;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    try {
        Totals totals;
        const int failures = qualify(totals);
        std::cout << "cases=" << totals.cases << " max_abs_error=" << totals.max_abs_error
                  << " max_beyond_allowance=" << totals.max_abs_clear
                  << " max_allowance=" << totals.max_allowance
                  << " max_rel_l2=" << totals.max_rel_l2
                  << " loose_elements=" << totals.loose_elements << "/" << totals.elements
                  << " cache_bit_differences=" << totals.flips << '\n';
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " relative_sink_attention\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "relative_sink_attention: " << error.what() << '\n';
        return 1;
    }
}
