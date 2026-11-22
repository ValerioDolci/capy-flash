// Real artifact on two devices: --unload-endpoints tears the Engine down inside the running server
// and builds it again (docs/serving/endpoints.md, "Unload and load").
//   1. /health reports the Engine's state; an unloaded Engine is a healthy server, and the engine
//      watch does not stop the server for it.
//   2. Three unload/load cycles: every unload returns the device memory this process holds to the
//      level of the first unload, and greedy output after every load equals the output before the
//      first unload.
//   3. A request to an unloaded Engine loads it first and answers with the same text.
//   4. /unload answers 409 while a request runs; /load and /unload answer "already" when there is
//      nothing to do.
//   5. --idle-unload-seconds unloads after the given time without requests.
#include "product/logging/logging.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <cuda_runtime.h>
#include <httplib.h>
#include <nvml.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// Device memory this process holds on each device (NVML). cudaMemGetInfo is device-wide, and a
// desktop on device 0 moves it by hundreds of MiB.
std::vector<unsigned long long> own_memory() {
    std::vector<unsigned long long> out;
    for (int device = 0; device < 2; ++device) {
        char bus[32] = {};
        nvmlDevice_t handle{};
        if (cudaDeviceGetPCIBusId(bus, sizeof(bus), device) != cudaSuccess ||
            nvmlDeviceGetHandleByPciBusId_v2(bus, &handle) != NVML_SUCCESS) {
            throw std::runtime_error("no NVML handle for a CUDA device");
        }
        std::vector<nvmlProcessInfo_t> processes(64);
        unsigned int count = static_cast<unsigned int>(processes.size());
        if (nvmlDeviceGetComputeRunningProcesses(handle, &count, processes.data()) !=
            NVML_SUCCESS) {
            throw std::runtime_error("nvmlDeviceGetComputeRunningProcesses failed");
        }
        unsigned long long used = 0;
        for (unsigned int k = 0; k < count; ++k) {
            if (processes[k].pid == static_cast<unsigned int>(::getpid())) {
                used = processes[k].usedGpuMemory;
            }
        }
        out.push_back(used);
    }
    return out;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: tensor parallelism needs two CUDA devices\n";
        return 77;
    }

    if (nvmlInit_v2() != NVML_SUCCESS) {
        std::cout << "skip: NVML is not available\n";
        return 77;
    }

    std::string port = "18091";
    if (const char* env = std::getenv("NINFER_TEST_PORT"); env != nullptr && *env != '\0') {
        port = env;
    }
    constexpr int kIdleSeconds    = 6;
    std::vector<std::string> args = {"ninfer-serve",
                                     artifact,
                                     "--host",
                                     "127.0.0.1",
                                     "--port",
                                     port,
                                     "--tp",
                                     "2",
                                     "--devices",
                                     "0,1",
                                     "--max-context",
                                     "2048",
                                     "--kv-capacity",
                                     "2048",
                                     "--max-concurrency",
                                     "1",
                                     "--log-stats-interval-ms",
                                     "1000",
                                     "--greedy",
                                     "--unload-endpoints",
                                     "--idle-unload-seconds",
                                     std::to_string(kIdleSeconds)};
    std::vector<char*> argv;
    for (std::string& arg : args) { argv.push_back(arg.data()); }
    ninfer::serve::ServeOptions options =
        ninfer::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve-test",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});

    int failures = 0;
    try {
        ninfer::serve::HttpServer server(options, logging.logger());
        if (!server.bind()) {
            std::cerr << "FAIL: could not bind 127.0.0.1:" << options.port << '\n';
            return 1;
        }
        ninfer::serve::GenerationService service(options);
        service.warmup();
        server.attach(service);
        std::thread listener([&] { (void)server.listen(); });

        httplib::Client client("127.0.0.1", options.port);
        client.set_connection_timeout(5, 0);
        client.set_read_timeout(600, 0);
        const auto health = [&]() -> Json {
            auto res = client.Get("/health");
            return res && res->status == 200 ? Json::parse(res->body) : Json{};
        };
        const auto post = [&](const char* path) -> std::pair<int, Json> {
            auto res = client.Post(path, "", "application/json");
            if (!res) { return {0, Json{}}; }
            return {res->status, res->body.empty() ? Json{} : Json::parse(res->body)};
        };
        // httplib clients are not thread-safe: the request that holds the Engine during the 409
        // check runs on its own client.
        httplib::Client long_client("127.0.0.1", options.port);
        long_client.set_read_timeout(600, 0);
        const auto chat_on = [&](httplib::Client& on, const std::string& prompt,
                                 int max_tokens) -> std::string {
            const Json body = {{"model", server.public_model_id()},
                               {"messages", Json::array({{{"role", "user"}, {"content", prompt}}})},
                               {"max_tokens", max_tokens}};
            auto res        = on.Post("/v1/chat/completions", body.dump(), "application/json");
            if (!res || res->status != 200) {
                return "<status " + std::to_string(res ? res->status : 0) + ">";
            }
            // The whole message: a thinking reply of 48 tokens may have no content yet.
            return Json::parse(res->body)["choices"][0]["message"].dump();
        };
        const auto chat = [&](const std::string& prompt, int max_tokens) {
            return chat_on(client, prompt, max_tokens);
        };
        const std::string prompt = "Name three primary colors and one fact about each.";

        for (int i = 0; i < 100 && health().value("status", "") != "ok"; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        failures += check(health().value("engine", "") == "loaded", "/health: not loaded at start");
        const std::string reference = chat(prompt, 48);
        failures += check(!reference.empty() && reference[0] != '<', "reference request failed");

        std::vector<unsigned long long> unloaded_used;
        for (int cycle = 0; cycle < 3; ++cycle) {
            const auto [unload_status, unloaded] = post("/unload");
            failures += check(unload_status == 200 && unloaded.value("engine", "") == "unloaded",
                              "/unload did not unload");
            const Json state = health();
            failures +=
                check(state.value("status", "") == "ok" && state.value("engine", "") == "unloaded",
                      "/health: an unloaded Engine is not a healthy, unloaded server");
            const std::vector<unsigned long long> used = own_memory();
            if (unloaded_used.empty()) { unloaded_used = used; }
            for (std::size_t d = 0; d < used.size(); ++d) {
                std::cout << "cycle " << cycle << " device " << d << ": this process holds "
                          << (used[d] >> 20) << " MiB, after the first unload "
                          << (unloaded_used[d] >> 20) << " MiB\n";
                failures += check(used[d] == unloaded_used[d],
                                  "device memory after an unload differs from the first");
            }
            failures += check(post("/unload").first == 200, "a second /unload was not a no-op");
            // The engine watch polls every 250 ms: it must not take "unloaded" for a failure.
            std::this_thread::sleep_for(std::chrono::milliseconds(600));
            failures +=
                check(!server.engine_failed(), "the engine watch stopped an unloaded server");
            if (cycle == 1) {
                // A request loads the Engine first.
                failures += check(chat(prompt, 48) == reference,
                                  "greedy text after a load on request differs");
            } else {
                const auto [load_status, loaded] = post("/load");
                failures += check(load_status == 200 && loaded.value("engine", "") == "loaded",
                                  "/load did not load");
                failures += check(post("/load").first == 200, "a second /load was not a no-op");
                failures += check(chat(prompt, 48) == reference, "greedy text after /load differs");
            }
        }

        // 409 while a request holds the Engine.
        std::atomic<bool> long_done{false};
        std::thread long_request([&] {
            (void)chat_on(long_client, "Write a long story about a lighthouse keeper.", 400);
            long_done.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        const int busy = post("/unload").first;
        long_request.join();
        failures += check(busy == 409 || long_done.load(), "/unload during a request was not 409");
        failures += check(health().value("engine", "") == "loaded",
                          "the Engine is not loaded after the refused unload");

        // Idle unload.
        std::this_thread::sleep_for(std::chrono::seconds(kIdleSeconds + 3));
        failures += check(health().value("engine", "") == "unloaded",
                          "--idle-unload-seconds did not unload");
        failures +=
            check(chat(prompt, 48) == reference, "greedy text after an idle unload differs");

        server.stop();
        listener.join();
        failures += check(!server.engine_failed(), "engine_failed() set");
        (void)nvmlShutdown();
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: exception: " << exception.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " serve unload/load\n";
    return failures == 0 ? 0 : 1;
}
