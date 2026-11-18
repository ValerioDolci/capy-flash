// Microbenchmark: routed experts on the CPU, GGUF blocks (host_gguf.h, fngine port) vs NVFP4
// (host_nvfp4.h / HostMoeExecutor), on real banks extracted by tools/convert/gguf_experts.py.
//
//   bench_host_gguf BANKS_DIR [--layers 0,1,17,47] [--threads 6,12] [--seconds S]
//                   [--isa avx512,avx2] [--tokens 1,2,4,8] [--call-tokens 1,4] [--us 1,4,10,20]
//                   [--nvfp4 0|1]
//
// --isa runs the GGUF kernels of each listed ISA (host_gguf_kernels) at every point, interleaved
// in that order (default: host_gguf_isa() only); the NVFP4 reference uses host_isa().
//
// BANKS_DIR holds layer_LL.{gate_up,down}.bin + manifest.json (gguf_experts.py extract) and,
// optionally, layer_00.nvfp4.{gate_up,down}.bin (the artifact's NVFP4 bank objects of layer 0).
//
// 1. kernel, one thread: whole experts (gate|up 1280 rows, SwiGLU, h quantized, down 2560 rows)
//    in random expert order over the 512 of a layer (each expert read once per pass: DRAM, not
//    cache), T = 1, 2, 4, 8 tokens; us per expert and GB/s of weights. Also the decode alone
//    (GGUF blocks expanded to the kernel layout, no dot products).
// 2. layer call, N pinned threads (HostCpuPool, spread over the CCDs as the engine does): U
//    distinct missing experts per call, the decode pattern (~1.5 tokens per expert), T = 1 and 4;
//    NVFP4 through the engine's HostMoeExecutor, GGUF through an executor of the same two-phase
//    Static shape written here (gate/up in 32-row slices, barrier, down in 32-row output slices).
#include "ops/host_moe/host_gguf.h"
#include "ops/host_moe/host_moe_executor.h"
#include "ops/host_moe/host_nvfp4.h"

#include "nlohmann/json.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace g = ninfer::ops::host::gguf;
namespace h = ninfer::ops::host;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int H = 2560, I = 640, E = 512, kMaxT = 8;

// GGUF kernels of the point being measured (--isa).
const g::HostGgufKernels* gk = nullptr;

double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

struct Mapped {
    std::uint8_t* p = nullptr;
    std::size_t n   = 0;
    Mapped()        = default;

    explicit Mapped(const std::string& path) {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error("cannot read " + path);
        n                     = static_cast<std::size_t>(in.tellg());
        const std::size_t cap = (n + (2u << 20) - 1) / (2u << 20) * (2u << 20);
        p                     = static_cast<std::uint8_t*>(std::aligned_alloc(2u << 20, cap));
        madvise(p, cap, MADV_HUGEPAGE);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n));
        mlock(p, cap); // resident like the engine's pinned bank (best effort)
    }

    ~Mapped() { std::free(p); }

    Mapped(Mapped&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }

    Mapped& operator=(Mapped&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
};

struct AlignedBuf {
    std::uint8_t* p = nullptr;

    explicit AlignedBuf(std::size_t n = 64) {
        p = static_cast<std::uint8_t*>(std::aligned_alloc(64, (n + 63) / 64 * 64));
        std::memset(p, 0, (n + 63) / 64 * 64);
    }

    ~AlignedBuf() { std::free(p); }

    AlignedBuf(const AlignedBuf&)            = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
};

g::WeightFormat format_of(const std::string& n) {
    if (n == "gguf_iq2_s") return g::WeightFormat::IQ2_S;
    if (n == "gguf_iq3_xxs") return g::WeightFormat::IQ3_XXS;
    if (n == "gguf_iq3_s") return g::WeightFormat::IQ3_S;
    if (n == "gguf_iq4_xs") return g::WeightFormat::IQ4_XS;
    if (n == "gguf_iq4_nl") return g::WeightFormat::IQ4_NL;
    if (n == "gguf_q2_0") return g::WeightFormat::Q2_0;
    throw std::runtime_error("format " + n);
}

struct GgufLayer {
    int layer = 0;
    g::WeightFormat gu{}, dn{};
    Mapped gu_bank, dn_bank;
    std::size_t gu_rb = 0, dn_rb = 0;

    std::size_t expert_bytes() const { return 2 * I * gu_rb + H * dn_rb; }

    std::string name() const {
        return "L" + std::to_string(layer) + " " + g::format_name(gu) + "+" + g::format_name(dn);
    }
};

std::uint16_t bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

std::vector<std::uint16_t> make_x(int T, std::mt19937_64& rng) {
    std::normal_distribution<float> nd(0.F, 1.F);
    std::vector<std::uint16_t> x(std::size_t(T) * H);
    for (auto& v : x) v = bf16(nd(rng));
    return x;
}

// ---------------------------------------------------------------- one expert, one thread
struct GgufScratch {
    AlignedBuf dec, xa, ha;
    std::vector<float> gate, up, hf, y;

    GgufScratch()
        : dec(8 * g::decoded_row_bytes(g::WeightFormat::IQ4_XS, H) + 8 * 4096),
          xa(kMaxT * g::packed_act_bytes(g::ActFormat::Q8_K, H)),
          ha(kMaxT * g::packed_act_bytes(g::ActFormat::Q8_0, H)), gate(I * kMaxT), up(I * kMaxT),
          hf(I), y(std::size_t(H) * kMaxT) {}
};

// gate|up then down of expert e for T tokens whose activations are packed in s.xa (x quantized
// once per token for the call, as the executor does). Returns a checksum.
float gguf_expert(const GgufLayer& L, int e, int T, GgufScratch& s, bool decode_only = false) {
    const std::size_t pbx = g::packed_act_bytes(g::format_info(L.gu).act, H);
    const std::size_t pbh = g::packed_act_bytes(g::ActFormat::Q8_0, I);
    const void* xa[kMaxT];
    const void* ha[kMaxT];
    for (int t = 0; t < T; ++t) {
        xa[t] = s.xa.p + pbx * t;
        ha[t] = s.ha.p + pbh * t;
    }
    const std::uint8_t* gu = L.gu_bank.p + std::size_t(e) * 2 * I * L.gu_rb;
    const std::size_t drb  = g::decoded_row_bytes(L.gu, H);
    for (int r = 0; r < 2 * I; r += 8) {
        gk->decode_rows(L.gu, gu + std::size_t(r) * L.gu_rb, L.gu_rb, 8, H, s.dec.p);
        if (!decode_only) {
            float* o = (r < I ? s.gate.data() + r * kMaxT : s.up.data() + (r - I) * kMaxT);
            gk->gemm_decoded(L.gu, s.dec.p, 8, H, xa, T, o, kMaxT);
        }
    }
    (void)drb;
    if (!decode_only) {
        for (int t = 0; t < T; ++t) {
            std::vector<float> gc(I), uc(I);
            for (int i = 0; i < I; ++i) {
                gc[i] = s.gate[i * kMaxT + t];
                uc[i] = s.up[i * kMaxT + t];
            }
            gk->silu_mul(gc.data(), uc.data(), s.hf.data(), I);
            gk->quantize_pack_f32(g::ActFormat::Q8_0, s.hf.data(), I, s.ha.p + pbh * t);
        }
    }
    const std::uint8_t* dn = L.dn_bank.p + std::size_t(e) * H * L.dn_rb;
    for (int r = 0; r < H; r += 8) {
        gk->decode_rows(L.dn, dn + std::size_t(r) * L.dn_rb, L.dn_rb, 8, I, s.dec.p);
        if (!decode_only)
            gk->gemm_decoded(L.dn, s.dec.p, 8, I, ha, T, s.y.data() + r * kMaxT, kMaxT);
    }
    return s.y[0] + s.gate[0];
}

struct Nvfp4Layer {
    Mapped gu_bank, dn_bank;
    h::HostExpertLayer layer;
    static constexpr std::size_t kExpertBytes = 2764800;
};

struct Nvfp4Scratch {
    h::W4A8ActivationBuffer x{H, kMaxT}, hb{I, kMaxT};
    std::vector<float> gate = std::vector<float>(I * kMaxT), up = std::vector<float>(I * kMaxT),
                       y = std::vector<float>(std::size_t(H) * kMaxT), hf = std::vector<float>(I);
};

float nvfp4_expert(const Nvfp4Layer& L, int e, int T, Nvfp4Scratch& s) {
    h::W4A8Activation xa[kMaxT], ha[kMaxT];
    for (int t = 0; t < T; ++t) {
        xa[t] = s.x.at(t);
        ha[t] = s.hb.at(t);
    }
    h::rows_dot(L.layer.gate_up, std::int64_t(e) * 2 * I, I, xa, T, s.gate.data(), I);
    h::rows_dot(L.layer.gate_up, std::int64_t(e) * 2 * I + I, I, xa, T, s.up.data(), I);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < I; ++i) {
            const float gv = s.gate[t * I + i];
            s.hf[i]        = gv / (1.F + std::exp(-gv)) * s.up[t * I + i];
        }
        h::quantize_blocks_a4(s.hf.data(), 0, I / 16, 100.F, ha[t]);
    }
    h::rows_dot(L.layer.down, std::int64_t(e) * H, H, ha, T, s.y.data(), H);
    return s.y[0] + s.gate[0];
}

// ---------------------------------------------------------------- GGUF two-phase layer executor
struct CallPlan {
    int T = 1;
    std::vector<int> experts;            // U distinct ids
    std::vector<std::vector<int>> toks;  // per expert, its tokens
    std::vector<std::vector<float>> alp; // per expert, its weights
};

CallPlan make_plan(int U, int T, std::mt19937_64& rng) {
    CallPlan p;
    p.T = T;
    std::vector<int> ids(E);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    for (int u = 0; u < U; ++u) {
        p.experts.push_back(ids[u]);
        std::vector<int> t{u % T};
        if (T > 1 && u % 2 == 0) t.push_back((u + 1) % T); // ~1.5 tokens per expert
        std::sort(t.begin(), t.end());
        p.alp.push_back(std::vector<float>(t.size(), 0.1F));
        p.toks.push_back(std::move(t));
    }
    return p;
}

class GgufExecutor {
public:
    GgufExecutor(h::HostCpuPool& pool) : pool_(pool), members_(pool.size()) {
        const std::size_t pbx = g::packed_act_bytes(g::ActFormat::Q8_K, H);
        x_                    = std::make_unique<AlignedBuf>(pbx * kMaxT);
        h_ = std::make_unique<AlignedBuf>(g::packed_act_bytes(g::ActFormat::Q8_0, I) * kMaxT * 64);
        for (auto& m : members_) m.dec = std::make_unique<AlignedBuf>(8 * 4096);
        y_.assign(std::size_t(kMaxT) * H, 0.F);
    }

    void run(const GgufLayer& L, const CallPlan& plan, const std::vector<std::uint16_t>& x) {
        L_    = &L;
        plan_ = &plan;
        x_in_ = &x;
        // Every token's x once (Q8_K for IQ gate/up formats), as the executor's token phase.
        const std::size_t pbx = g::packed_act_bytes(g::format_info(L.gu).act, H);
        for (int t = 0; t < plan.T; ++t)
            gk->quantize_pack_bf16(g::format_info(L.gu).act, x.data() + std::size_t(t) * H, H,
                                   x_->p + pbx * t);
        std::fill(y_.begin(), y_.begin() + std::size_t(plan.T) * H, 0.F);
        pool_.run(&GgufExecutor::body, this);
    }

    float checksum() const { return y_[0]; }

private:
    struct alignas(64) Member {
        std::unique_ptr<AlignedBuf> dec;
        float g[8 * kMaxT], u[8 * kMaxT], p[8 * kMaxT];
        float gate[32 * kMaxT], up[32 * kMaxT];
    };

    static void body(void* self, std::int32_t m) { static_cast<GgufExecutor*>(self)->member(m); }

    void member(int m) {
        const GgufLayer& L = *L_;
        const CallPlan& P  = *plan_;
        const int n = pool_.size(), U = static_cast<int>(P.experts.size());
        Member& me            = members_[static_cast<std::size_t>(m)];
        const std::size_t pbx = g::packed_act_bytes(g::format_info(L.gu).act, H);
        const std::size_t pbh = g::packed_act_bytes(g::ActFormat::Q8_0, I);
        // Phase 1: items (expert, 32 intermediate rows).
        const int items1 = U * (I / 32);
        for (int it = items1 * m / n; it < items1 * (m + 1) / n; ++it) {
            const int u = it / (I / 32), s = it % (I / 32);
            const int e      = P.experts[u];
            const auto& toks = P.toks[u];
            const int T      = static_cast<int>(toks.size());
            const void* xa[kMaxT];
            for (int i = 0; i < T; ++i) xa[i] = x_->p + pbx * toks[i];
            const std::uint8_t* base = L.gu_bank.p + std::size_t(e) * 2 * I * L.gu_rb;
            for (int c = 0; c < 4; ++c) {
                const int r = s * 32 + c * 8;
                gk->decode_rows(L.gu, base + std::size_t(r) * L.gu_rb, L.gu_rb, 8, H, me.dec->p);
                gk->gemm_decoded(L.gu, me.dec->p, 8, H, xa, T, me.gate + c * 8 * kMaxT, kMaxT);
                gk->decode_rows(L.gu, base + std::size_t(I + r) * L.gu_rb, L.gu_rb, 8, H,
                                me.dec->p);
                gk->gemm_decoded(L.gu, me.dec->p, 8, H, xa, T, me.up + c * 8 * kMaxT, kMaxT);
            }
            for (int i = 0; i < T; ++i) {
                float gc[32], uc[32], hv[32];
                for (int j = 0; j < 32; ++j) {
                    gc[j] = me.gate[j * kMaxT + i];
                    uc[j] = me.up[j * kMaxT + i];
                }
                gk->silu_mul(gc, uc, hv, 32);
                gk->quantize_pack_q8_0_half(hv, s / 2, s % 2, h_->p + pbh * (u * kMaxT + i));
            }
        }
        pool_.barrier(1);
        // Phase 2: items = 32 output rows, every expert in order.
        const int items2 = H / 32;
        for (int it = items2 * m / n; it < items2 * (m + 1) / n; ++it) {
            for (int u = 0; u < U; ++u) {
                const int e      = P.experts[u];
                const auto& toks = P.toks[u];
                const int T      = static_cast<int>(toks.size());
                const void* ha[kMaxT];
                for (int i = 0; i < T; ++i) ha[i] = h_->p + pbh * (u * kMaxT + i);
                const std::uint8_t* base = L.dn_bank.p + std::size_t(e) * H * L.dn_rb;
                for (int c = 0; c < 4; ++c) {
                    const int r = it * 32 + c * 8;
                    gk->decode_rows(L.dn, base + std::size_t(r) * L.dn_rb, L.dn_rb, 8, I,
                                    me.dec->p);
                    gk->gemm_decoded(L.dn, me.dec->p, 8, I, ha, T, me.p, kMaxT);
                    for (int rr = 0; rr < 8; ++rr)
                        for (int i = 0; i < T; ++i)
                            y_[std::size_t(toks[i]) * H + r + rr] +=
                                P.alp[u][i] * me.p[rr * kMaxT + i];
                }
            }
        }
    }

    h::HostCpuPool& pool_;
    std::vector<Member> members_;
    std::unique_ptr<AlignedBuf> x_, h_;
    std::vector<float> y_;
    const GgufLayer* L_                     = nullptr;
    const CallPlan* plan_                   = nullptr;
    const std::vector<std::uint16_t>* x_in_ = nullptr;
};

std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    std::string v(s);
    std::size_t a = 0;
    while (a < v.size()) {
        const std::size_t b = v.find(',', a);
        out.push_back(std::stoi(v.substr(a, b == std::string::npos ? std::string::npos : b - a)));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s BANKS_DIR [--layers L,..] [--threads N,..] [--seconds S] [--isa "
                     "avx512,avx2]\n"
                     "       [--tokens T,..] [--call-tokens T,..] [--us U,..] [--nvfp4 0|1]\n",
                     argv[0]);
        return 2;
    }
    if (!h::host_experts_supported()) {
        std::printf("SKIP: CPU runs neither ISA of the host expert kernels\n");
        return 77;
    }
    const std::string dir = argv[1];
    std::vector<int> layers{0, 1, 17, 47}, threads{6, 12}, tokens{1, 2, 4, 8}, call_tokens{1, 4},
        ulist{1, 4, 10, 20};
    std::vector<const g::HostGgufKernels*> isas{&g::host_gguf_kernels(g::host_gguf_isa())};
    double seconds = 1.5;
    bool use_nvfp4 = true;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--layers"))
            layers = parse_list(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--threads"))
            threads = parse_list(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--seconds"))
            seconds = std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--tokens"))
            tokens = parse_list(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--call-tokens"))
            call_tokens = parse_list(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--us"))
            ulist = parse_list(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--nvfp4"))
            use_nvfp4 = std::atoi(argv[i + 1]) != 0;
        else if (!std::strcmp(argv[i], "--isa")) {
            isas.clear();
            std::string v(argv[i + 1]);
            for (std::size_t a = 0; a <= v.size();) {
                const std::size_t b = std::min(v.find(',', a), v.size());
                const std::string n = v.substr(a, b - a);
                if (n == "avx2")
                    isas.push_back(&g::host_gguf_kernels(h::HostIsa::Avx2));
                else if (n == "avx512")
                    isas.push_back(&g::host_gguf_kernels(h::HostIsa::Avx512Vnni));
                else
                    throw std::runtime_error("--isa " + n);
                a = b + 1;
            }
        }
    }
    gk = isas.front();
    std::printf("GGUF ISAs:");
    for (const auto* k : isas) std::printf(" %s", h::host_isa_name(k->isa));
    std::printf("; NVFP4 ISA: %s\n", h::host_isa_name(h::host_isa()));
    std::ifstream in(dir + "/manifest.json");
    const auto manifest = nlohmann::json::parse(in);
    std::vector<std::unique_ptr<GgufLayer>> gl;
    for (int l : layers) {
        const auto& e = manifest.at("layers").at(std::to_string(l));
        auto L        = std::make_unique<GgufLayer>();
        L->layer      = l;
        L->gu         = format_of(e.at("gate_up").at("format"));
        L->dn         = format_of(e.at("down").at("format"));
        L->gu_rb      = e.at("gate_up").at("row_bytes");
        L->dn_rb      = e.at("down").at("row_bytes");
        L->gu_bank    = Mapped(dir + "/" + e.at("gate_up").at("file").get<std::string>());
        L->dn_bank    = Mapped(dir + "/" + e.at("down").at("file").get<std::string>());
        gl.push_back(std::move(L));
    }
    std::unique_ptr<Nvfp4Layer> nv;
    if (use_nvfp4 && std::ifstream(dir + "/layer_00.nvfp4.gate_up.bin")) {
        nv          = std::make_unique<Nvfp4Layer>();
        nv->gu_bank = Mapped(dir + "/layer_00.nvfp4.gate_up.bin");
        nv->dn_bank = Mapped(dir + "/layer_00.nvfp4.down.bin");
        nv->layer.gate_up =
            h::Nvfp4Plane::from_payload(nv->gu_bank.p, std::int64_t(E) * 2 * I, H, 2 * E);
        nv->layer.down    = h::Nvfp4Plane::from_payload(nv->dn_bank.p, std::int64_t(E) * H, I, E);
        nv->layer.experts = E;
        nv->layer.hidden  = H;
        nv->layer.intermediate          = I;
        nv->layer.gate_up_input_divisor = 100.F;
        nv->layer.down_input_divisor    = 100.F;
    }
    std::mt19937_64 rng(42);
    const auto x8 = make_x(kMaxT, rng);
    std::printf("== 1. kernel, one thread (random experts of the layer, DRAM), %.1f s per point\n",
                seconds);
    std::printf("%-22s %-6s %4s %10s %8s %12s\n", "layer", "isa", "T", "us/expert", "GB/s",
                "decode-only");
    {
        const auto c1 = h::spread_cpus(1);
        h::HostCpuPool one(c1);
        GgufScratch s;
        Nvfp4Scratch ns;
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (const auto& L : gl) {
            for (int T : tokens)
                for (const auto* isa : isas) {
                    gk                    = isa;
                    const std::size_t pbx = g::packed_act_bytes(g::format_info(L->gu).act, H);
                    for (int t = 0; t < T; ++t)
                        gk->quantize_pack_bf16(g::format_info(L->gu).act,
                                               x8.data() + std::size_t(t) * H, H, s.xa.p + pbx * t);
                    double dec_us = 0.0;
                    if (T == 1) {
                        int n           = 0;
                        const double t0 = now_s();
                        while (now_s() - t0 < seconds / 2)
                            gguf_expert(*L, order[n++ % E], T, s, true);
                        dec_us = (now_s() - t0) / n * 1e6;
                    }
                    int n           = 0;
                    float sink      = 0;
                    const double t0 = now_s();
                    while (now_s() - t0 < seconds) sink += gguf_expert(*L, order[n++ % E], T, s);
                    const double us = (now_s() - t0) / n * 1e6;
                    std::printf("%-22s %-6s %4d %10.1f %8.2f %12s%s\n", L->name().c_str(),
                                h::host_isa_name(gk->isa), T, us, L->expert_bytes() / us * 1e-3,
                                T == 1 ? std::to_string(dec_us).substr(0, 6).c_str() : "",
                                sink == 12345.F ? "!" : "");
                }
        }
        if (nv) {
            for (int T : tokens) {
                for (int t = 0; t < T; ++t)
                    h::quantize_bf16_a4(x8.data() + std::size_t(t) * H, H, 100.F, ns.x.at(t));
                int n           = 0;
                float sink      = 0;
                const double t0 = now_s();
                while (now_s() - t0 < seconds) sink += nvfp4_expert(*nv, order[n++ % E], T, ns);
                const double us = (now_s() - t0) / n * 1e6;
                std::printf("%-22s %-6s %4d %10.1f %8.2f %12s%s\n", "L0 NVFP4 (W4A4 host)",
                            h::host_isa_name(h::host_isa()), T, us,
                            Nvfp4Layer::kExpertBytes / us * 1e-3, "", sink == 12345.F ? "!" : "");
            }
        }
    }
    std::printf("\n== 2. layer call, pinned pool (spread over CCDs), U missing experts, ~1.5 "
                "tokens/expert\n");
    std::printf("%-22s %-6s %3s %3s %3s %10s %10s %8s\n", "layer", "isa", "thr", "T", "U",
                "us/call", "us/expert", "GB/s");
    for (int nthr : threads) {
        const auto cpus = h::spread_cpus(nthr);
        h::HostCpuPool pool(cpus);
        GgufExecutor gx(pool);
        std::unique_ptr<h::HostMoeExecutor> nx;
        if (nv)
            nx = std::make_unique<h::HostMoeExecutor>(pool, H, I, kMaxT, 10,
                                                      h::HostMoeExecutor::Options{});
        for (int T : call_tokens) {
            for (int U : ulist) {
                if (U < T) continue;
                for (const auto& L : gl)
                    for (const auto* isa : isas) {
                        gk              = isa;
                        int n           = 0;
                        const double t0 = now_s();
                        double busy     = 0;
                        while (now_s() - t0 < seconds) {
                            const CallPlan plan = make_plan(U, T, rng);
                            const double a      = now_s();
                            gx.run(*L, plan, x8);
                            busy += now_s() - a;
                            ++n;
                        }
                        const double us = busy / n * 1e6;
                        std::printf("%-22s %-6s %3d %3d %3d %10.1f %10.2f %8.2f\n",
                                    L->name().c_str(), h::host_isa_name(gk->isa), nthr, T, U, us,
                                    us / U, U * L->expert_bytes() / us * 1e-3);
                    }
                if (nx) {
                    std::vector<std::int32_t> ids(std::size_t(kMaxT) * 10, -1);
                    std::vector<float> alpha(std::size_t(kMaxT) * 10, 0.1F),
                        out(std::size_t(kMaxT) * H);
                    int n           = 0;
                    double busy     = 0;
                    const double t0 = now_s();
                    while (now_s() - t0 < seconds) {
                        const CallPlan plan = make_plan(U, T, rng);
                        std::fill(ids.begin(), ids.end(), -1);
                        std::vector<int> fill(T, 0);
                        for (int u = 0; u < U; ++u)
                            for (int t : plan.toks[u])
                                if (fill[t] < 10)
                                    ids[std::size_t(t) * 10 + fill[t]++] = plan.experts[u];
                        h::HostMoeJob job{T, 10, x8.data(), ids.data(), alpha.data(), out.data()};
                        const double a = now_s();
                        nx->run(nv->layer, job);
                        busy += now_s() - a;
                        ++n;
                    }
                    const double us = busy / n * 1e6;
                    std::printf("%-22s %-6s %3d %3d %3d %10.1f %10.2f %8.2f\n", "L0 NVFP4 executor",
                                h::host_isa_name(h::host_isa()), nthr, T, U, us, us / U,
                                U * Nvfp4Layer::kExpertBytes / us * 1e-3);
                }
            }
        }
    }
    return 0;
}
