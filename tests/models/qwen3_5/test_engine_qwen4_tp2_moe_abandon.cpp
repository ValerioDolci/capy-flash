// Qwen4Exp (Qwen3.8-Flash-Next) at tensor-parallel width 2: a prompt chunk that fails inside its
// MoE exchange leaves the pinned-host exchange free for the next prompt. Needs
// NINFER_TEST_QWEN4_ARTIFACT (the synthetic artifact of tools/capy_flash/make_test_artifact.sh)
// and two CUDA devices; skips (77) otherwise.
//
// At tp 2 a prompt chunk's MoE layer sums the ranks' FP32 partials through the HostExchange by
// column groups (TextForward::moe_combine): begin_sum_f32, each rank's columns, finish_sum_f32. A
// failure between begin and finish must abandon the open call (OpenExchangeSum); otherwise every
// later prefill throws "begin_sum_f32 while a call is open". The failure is injected with link-time
// wrappers: ops::moe_accumulate_shared, which the shared expert's owner calls inside that window,
// throws once when armed and only while a call is open; the wrappers of HostExchange::
// begin_sum_f32, finish_sum_f32 and abandon_sum_f32 track the open call. The wrappers affect only
// this test executable; the production path has no fault hooks.
//
// One CausalScoring Engine (mapped experts; NINFER_TP_HOST_EXCHANGE=1, so the HostExchange carries
// the eager sums on a pair with peer access too) scores 160 tokens (two prompt chunks) twice: the
// two scores must be identical, the determinism the last check relies on. Armed, the third call
// must fail with the injected error, thrown while a call was open, and that call must have been
// abandoned. A fourth call must then reproduce the first scores bit for bit.
#include "ninfer/engine.h"
#include "ninfer/ops/allreduce.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::atomic<bool> g_armed{false};
std::atomic<bool> g_open{false};
std::atomic<int> g_injected{0};
std::atomic<int> g_abandoned_open{0};

constexpr const char* kInjected = "injected: MoE shared-expert accumulation failed (test)";

} // namespace

extern "C" {
// bool ninfer::ops::HostExchange::begin_sum_f32(const std::array<Tensor, 2>&,
//     const std::array<Tensor, 2>&, const std::array<Tensor, 2>*, Payload, const ExecutionContext&)
bool __real__ZN6ninfer3ops12HostExchange13begin_sum_f32ERKSt5arrayINS_6TensorELm2EES6_PS5_NS1_7PayloadERKNS_16ExecutionContextE(
    ninfer::ops::HostExchange* self, const std::array<ninfer::Tensor, 2>& partial,
    const std::array<ninfer::Tensor, 2>& staging, const std::array<ninfer::Tensor, 2>* bf16_out,
    ninfer::ops::HostExchange::Payload payload, const ninfer::ExecutionContext& ec);
void __real__ZN6ninfer3ops12HostExchange14finish_sum_f32Ev(ninfer::ops::HostExchange* self);
void __real__ZN6ninfer3ops12HostExchange15abandon_sum_f32Ev(ninfer::ops::HostExchange* self);
void __real__ZN6ninfer3ops21moe_accumulate_sharedERKNS_6TensorES3_RS1_P11CUstream_st(
    const ninfer::Tensor& shared_gate, const ninfer::Tensor& shared, ninfer::Tensor& partial,
    cudaStream_t stream);

bool __wrap__ZN6ninfer3ops12HostExchange13begin_sum_f32ERKSt5arrayINS_6TensorELm2EES6_PS5_NS1_7PayloadERKNS_16ExecutionContextE(
    ninfer::ops::HostExchange* self, const std::array<ninfer::Tensor, 2>& partial,
    const std::array<ninfer::Tensor, 2>& staging, const std::array<ninfer::Tensor, 2>* bf16_out,
    ninfer::ops::HostExchange::Payload payload, const ninfer::ExecutionContext& ec) {
    const bool opened =
        __real__ZN6ninfer3ops12HostExchange13begin_sum_f32ERKSt5arrayINS_6TensorELm2EES6_PS5_NS1_7PayloadERKNS_16ExecutionContextE(
            self, partial, staging, bf16_out, payload, ec);
    if (opened) { g_open.store(true); }
    return opened;
}

void __wrap__ZN6ninfer3ops12HostExchange14finish_sum_f32Ev(ninfer::ops::HostExchange* self) {
    __real__ZN6ninfer3ops12HostExchange14finish_sum_f32Ev(self);
    g_open.store(false);
}

// noexcept in the class: the wrapper must not throw either.
void __wrap__ZN6ninfer3ops12HostExchange15abandon_sum_f32Ev(ninfer::ops::HostExchange* self) {
    if (g_open.exchange(false)) { g_abandoned_open.fetch_add(1); }
    __real__ZN6ninfer3ops12HostExchange15abandon_sum_f32Ev(self);
}

void __wrap__ZN6ninfer3ops21moe_accumulate_sharedERKNS_6TensorES3_RS1_P11CUstream_st(
    const ninfer::Tensor& shared_gate, const ninfer::Tensor& shared, ninfer::Tensor& partial,
    cudaStream_t stream) {
    if (g_open.load() && g_armed.exchange(false)) {
        g_injected.fetch_add(1);
        throw std::runtime_error(kInjected);
    }
    __real__ZN6ninfer3ops21moe_accumulate_sharedERKNS_6TensorES3_RS1_P11CUstream_st(
        shared_gate, shared, partial, stream);
}
}

namespace {

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// 160 tokens: one full 128-column prompt chunk and a 32-column one.
std::vector<ninfer::TokenId> prompt() {
    std::vector<ninfer::TokenId> ids(160);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<ninfer::TokenId>(1000 + (i * 37) % 4000);
    }
    return ids;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_QWEN4_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: two CUDA devices are required\n";
        return 77;
    }
    setenv("NINFER_TP_HOST_EXCHANGE", "1", 1);
    int failures = 0;
    try {
        ninfer::EngineOptions options;
        options.artifact_path          = artifact;
        options.purpose                = ninfer::EnginePurpose::CausalScoring;
        options.routed_experts.backend = ninfer::RoutedExpertBackend::Mapped;
        options.kv_cache               = ninfer::KvCacheStorage::BFloat16;
        options.prefill_chunk          = 128;
        options.max_context            = 256;
        options.kv_capacity            = ninfer::KvCapacityPolicy::explicit_capacity(256);
        options.tp                     = 2;
        options.devices                = {0, 1};
        ninfer::Engine engine(options);
        const std::vector<ninfer::TokenId> ids = prompt();

        const std::vector<float> first  = engine.score_tokens(ids, 1);
        const std::vector<float> second = engine.score_tokens(ids, 1);
        failures += check(!first.empty() && first == second,
                          "two scorings of one prompt differ: the comparison after the failure "
                          "cannot be exact");

        g_armed.store(true);
        std::string error;
        try {
            (void)engine.score_tokens(ids, 1);
        } catch (const std::exception& e) { error = e.what(); }
        g_armed.store(false);
        std::cout << "injected MoE failure: " << (error.empty() ? "none" : error) << '\n';
        failures += check(g_injected.load() == 1,
                          "the fault was not injected inside an open MoE exchange (is the "
                          "HostExchange attached?)");
        failures += check(error.find(kInjected) != std::string::npos,
                          "the scoring did not fail with the injected error");
        failures += check(g_abandoned_open.load() == 1,
                          "the failed chunk did not abandon its open MoE exchange");
        failures += check(!g_open.load(), "a HostExchange call is still open after the failure");

        std::vector<float> after;
        try {
            after = engine.score_tokens(ids, 1);
        } catch (const std::exception& e) {
            std::cerr << "scoring after the failure: " << e.what() << '\n';
        }
        failures += check(after == first,
                          "the scoring after a failed MoE exchange does not reproduce the scores "
                          "before it");
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "OK qwen4 tp2 MoE exchange abandon\n";
    return 0;
}
