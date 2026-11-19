#pragma once

// Qwen4Exp deferred decode rows: a persistent helper thread that runs one job per round (the
// round's n-gram gather into the pinned staging, then the raise of the PLE layers' words): after
// the launch, or right after the previous round synchronized for the rows gathered early
// (Qwen4PleRuntime::start_early). The execution thread meanwhile observes the round's routing for
// the expert cache policy (ForwardRuntime::decode_ple_staged, Qwen4PleRuntime::stage_round).

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <thread>
#include <utility>

namespace ninfer::models::qwen3_5 {

class Qwen4PleStager {
public:
    explicit Qwen4PleStager(std::function<void()> job)
        : job_(std::move(job)), thread_([this] { loop(); }) {}

    ~Qwen4PleStager() {
        state_.store(kStop, std::memory_order_release);
        state_.notify_all();
        thread_.join();
    }

    Qwen4PleStager(const Qwen4PleStager&)            = delete;
    Qwen4PleStager& operator=(const Qwen4PleStager&) = delete;

    // Runs the job once on the helper thread. The previous run must have been finished.
    void start() {
        state_.store(kRequested, std::memory_order_release);
        state_.notify_one();
    }

    // Waits for the run started last; rethrows its exception.
    void finish() {
        std::uint32_t s = state_.load(std::memory_order_acquire);
        while (s == kRequested) {
            state_.wait(s, std::memory_order_acquire);
            s = state_.load(std::memory_order_acquire);
        }
        state_.store(kIdle, std::memory_order_relaxed);
        if (error_) { std::rethrow_exception(std::exchange(error_, nullptr)); }
    }

private:
    static constexpr std::uint32_t kIdle = 0, kRequested = 1, kDone = 2, kStop = 3;

    void loop() {
        for (;;) {
            std::uint32_t s = state_.load(std::memory_order_acquire);
            while (s != kRequested && s != kStop) {
                state_.wait(s, std::memory_order_acquire);
                s = state_.load(std::memory_order_acquire);
            }
            if (s == kStop) { return; }
            try {
                job_();
            } catch (...) { error_ = std::current_exception(); }
            // A concurrent stop wins: the destructor joins right after.
            std::uint32_t expected = kRequested;
            state_.compare_exchange_strong(expected, kDone, std::memory_order_acq_rel);
            state_.notify_all();
        }
    }

    std::function<void()> job_;
    std::exception_ptr error_;
    std::atomic<std::uint32_t> state_{kIdle};
    std::thread thread_; // last: started once the members above exist
};

} // namespace ninfer::models::qwen3_5
