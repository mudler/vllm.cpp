// VT CPU PER-ELEMENT DTYPE DISPATCH SURVEY PROBE — row VT-CPU-ELEM-SURVEY,
// .agents/specs/vt-cpu-elem-survey.md.
//
// WHAT QUESTION THIS EXISTS TO ANSWER. `VT-CPU-ELEM-DISPATCH` (#2376) hoisted
// the per-element dtype dispatch out of TWO of the 64 CPU kernels that call
// `vt::cpu::LoadF32`/`StoreF32`, and measured 8.75x-11.16x for the pair. Its
// `## Owed` names the remaining 62 and states the sizing method verbatim:
//
//     for each candidate, `perf record -e cpu-clock` over a probe that runs
//     THAT op alone at a shape a shipped model actually uses, and read
//     `LoadF32`'s self percentage. Above ~30% the hoist is worth a row; below
//     it the kernel is bound by something else and a hoist buys the fraction
//     the profile names.
//
// This binary is "a probe that runs THAT op alone". One `--op <name>` runs one
// kernel in a loop with no other vt work in the process, so `perf report`'s
// whole-process self percentages ARE the kernel's, and no call-graph
// attribution is needed to read them.
//
// THE SHAPES ARE GROUNDED, not synthetic. Every entry names the model and the
// config value its extents come from in the `where` field, printed by `--list`
// and by every run. A synthetic shape ranks kernels wrongly, because the whole
// question is whether the operand stream is cache-resident (dispatch-bound) or
// DRAM-bound (dispatch hidden), and that is a property of the extents.
//
// RUN RECIPE — DIRECT COMPILE, the same one `ltx2_connector_gemm_probe.cpp`
// records and for the same reason: a full CMake build writes 9.4 GiB and this
// devbox runs three other compiling agents. The CMake target below exists so
// this file cannot rot behind a `vt::` signature change; the recorded runs come
// from a direct `g++ -std=c++20 -O2 -ffp-contract=off -g
// -fno-omit-frame-pointer -pthread -Iinclude -Isrc -isystem third_party` over
// this file plus the `vt` TU set. `-ffp-contract=off` is NOT optional
// (CMakeLists.txt:55).
//
// CI COMPILES IT AND RUNS NOTHING, like its three siblings in this directory.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using Clock = std::chrono::steady_clock;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Deterministic filler (the shape `ltx2_connector_gemm_probe.cpp` uses): a
// kernel's cost does not depend on its values, and a deterministic stream is
// what lets one binary's output be compared byte-for-byte against another's.
struct Lcg {
  uint64_t s;
  float Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint32_t top = static_cast<uint32_t>(s >> 40);
    return static_cast<float>(static_cast<int32_t>(top) - (1 << 23)) /
           static_cast<float>(1 << 23);
  }
};

std::vector<float> F32(size_t n, uint64_t seed) {
  std::vector<float> v(n);
  Lcg g{seed ? seed : 1};
  for (float& x : v) x = g.Next();
  return v;
}
// Small positive values, for the operands a kernel exponentiates or inverts.
std::vector<float> F32Pos(size_t n, uint64_t seed) {
  std::vector<float> v = F32(n, seed);
  for (float& x : v) x = 0.25f + 0.5f * std::fabs(x);
  return v;
}
std::vector<uint16_t> Bf16(size_t n, uint64_t seed) {
  std::vector<float> f = F32(n, seed);
  std::vector<uint16_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = vt::F32ToBF16(f[i]);
  return v;
}
std::vector<int32_t> Iota32(size_t n, int32_t start = 0, int32_t step = 1) {
  std::vector<int32_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = start + static_cast<int32_t>(i) * step;
  return v;
}

Tensor T(std::vector<float>& v, std::initializer_list<int64_t> shape) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), shape);
}
Tensor TB(std::vector<uint16_t>& v, std::initializer_list<int64_t> shape) {
  return Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), shape);
}
Tensor TI(std::vector<int32_t>& v, std::initializer_list<int64_t> shape) {
  return Tensor::Contiguous(v.data(), DType::kI32, Cpu(), shape);
}

struct Case {
  const char* name;      // the enclosing kernel in src/vt/cpu/cpu_ops.cpp
  const char* where;     // the model and config value the shape comes from
  const char* shape;     // the extents, printed so a reader can check them
  std::function<void(Queue&, int)> run;  // run `reps` iterations of the op
};

std::vector<Case>& Registry() {
  static std::vector<Case> r;
  return r;
}
void Add(const char* name, const char* where, const char* shape,
         std::function<void(Queue&, int)> run) {
  Registry().push_back(Case{name, where, shape, std::move(run)});
}

// ── THE SHAPES, AND WHERE EACH ONE COMES FROM ───────────────────────────────
//
// Qwen3.6-27B (`unsloth/Qwen3.6-27B-NVFP4`, dense) and Qwen3.6-35B-A3B
// (`nvidia/Qwen3.6-35B-A3B-NVFP4`, MoE) are this tree's two GATE models. Their
// config.json values are recorded in `.agents/specs/qwen36-forward-notes.md`
// §1, read off the real checkpoints on dgx; the GDN derived dims are
// `.agents/specs/gdn-semantics.md` §1. LTX-2.5's connector geometry is read out
// of the DiT checkpoint's `__metadata__` and recorded in
// `tools/bench/ltx2_connector_gemm_probe.cpp`.
//
// The two token counts are the two regimes the repo itself benchmarks:
// PREFILL = 1024 tokens (the `input-1024` leg quoted throughout the parity
// ledger, and `kLtx2GemmaTokenizerMaxLength` for the connector), DECODE = 16
// rows (the `c16` concurrency rung the 27B/35B A/B legs use).
namespace shp {
constexpr int64_t kPrefill = 1024;  // input-1024, the ledger's prefill leg
constexpr int64_t kDecode = 16;     // c16, the ledger's decode concurrency rung

// Qwen3.6-27B dense text tower.
constexpr int64_t k27Hidden = 5120;
constexpr int64_t k27Inter = 17408;
constexpr int64_t k27Heads = 24;
constexpr int64_t k27KvHeads = 4;
constexpr int64_t k27HeadDim = 256;
constexpr int64_t k27RotaryDim = 64;  // partial_rotary_factor 0.25
constexpr int64_t k27Vocab = 248320;
// Qwen3.6-27B GDN: Hk 16, Hv 48, Dk 128, Dv 128, K 4.
constexpr int64_t kGdnHk = 16;
constexpr int64_t kGdnHv = 48;
constexpr int64_t kGdnDk = 128;
constexpr int64_t kGdnDv = 128;
constexpr int64_t kGdnK = 4;
constexpr int64_t kGdnKeyDim = kGdnHk * kGdnDk;     // 2048
constexpr int64_t kGdnValueDim = kGdnHv * kGdnDv;   // 6144
constexpr int64_t kGdnConvDim = 2 * kGdnKeyDim + kGdnValueDim;  // 10240

// Qwen3.6-35B-A3B MoE text tower.
constexpr int64_t k35Hidden = 2048;
constexpr int64_t k35Experts = 256;
constexpr int64_t k35TopK = 8;
constexpr int64_t k35MoeInter = 512;

// LTX-2.5 connector (video stream): 32 heads x 128 = 4096, 1024 rows.
constexpr int64_t kLtxHeads = 32;
constexpr int64_t kLtxHeadDim = 128;
constexpr int64_t kLtxDim = kLtxHeads * kLtxHeadDim;  // 4096
constexpr int64_t kLtxRows = 1024;
}  // namespace shp

// ── The cases ───────────────────────────────────────────────────────────────
//
// One entry per enclosing function in `src/vt/cpu/cpu_ops.cpp` that calls
// `LoadF32`/`StoreF32`. Each reaches its kernel through the PRODUCTION `vt::`
// entry point, never by constructing the kernel: a probe that called the
// kernel directly would measure a function, not the path a model takes.
void RegisterCases() {
  using namespace shp;

  // --- Norms -----------------------------------------------------------------
  Add("RmsNormKernel", "Qwen3.6-27B input_layernorm, prefill",
      "x[1024,5120] f32, w[5120]", [](Queue& q, int reps) {
        auto x = F32(kPrefill * k27Hidden, 1), w = F32(k27Hidden, 2);
        std::vector<float> o(kPrefill * k27Hidden);
        Tensor tx = T(x, {kPrefill, k27Hidden}), tw = T(w, {k27Hidden});
        Tensor to = T(o, {kPrefill, k27Hidden});
        for (int i = 0; i < reps; ++i) vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{});
      });

  Add("RmsNormGroupKernel", "Qwen4-exp grouped RMSNorm, prefill",
      "x[1024,5120] f32, group=128", [](Queue& q, int reps) {
        auto x = F32(kPrefill * k27Hidden, 3), w = F32(k27Hidden, 4);
        std::vector<float> o(kPrefill * k27Hidden);
        Tensor tx = T(x, {kPrefill, k27Hidden}), tw = T(w, {k27Hidden});
        Tensor to = T(o, {kPrefill, k27Hidden});
        vt::RmsNormGroupArgs a;
        a.group_size = 128;
        for (int i = 0; i < reps; ++i) vt::RmsNormGroup(q, to, tx, tw, a);
      });

  Add("RmsNormGatedKernel", "Qwen3.6-27B GDN output gated norm, decode",
      "x[16,48,128] f32", [](Queue& q, int reps) {
        auto x = F32(kDecode * kGdnValueDim, 5), g = F32(kDecode * kGdnValueDim, 6),
             w = F32(kGdnDv, 7);
        std::vector<float> o(kDecode * kGdnValueDim);
        Tensor tx = T(x, {kDecode, kGdnHv, kGdnDv}), tg = T(g, {kDecode, kGdnHv, kGdnDv});
        Tensor tw = T(w, {kGdnDv}), to = T(o, {kDecode, kGdnHv, kGdnDv});
        for (int i = 0; i < reps; ++i)
          vt::RmsNormGated(q, to, tx, tg, tw, vt::RmsNormGatedArgs{});
      });

  Add("RmsNormGatedGroupKernel", "Mamba2 Mixer2RMSNormGated, prefill",
      "x[1024,4096] f32, n_groups=8", [](Queue& q, int reps) {
        auto x = F32(kPrefill * kLtxDim, 8), g = F32(kPrefill * kLtxDim, 9),
             w = F32(kLtxDim, 10);
        std::vector<float> o(kPrefill * kLtxDim);
        Tensor tx = T(x, {kPrefill, kLtxDim}), tg = T(g, {kPrefill, kLtxDim});
        Tensor tw = T(w, {kLtxDim}), to = T(o, {kPrefill, kLtxDim});
        vt::RmsNormGatedGroupArgs a;
        a.n_groups = 8;
        for (int i = 0; i < reps; ++i) vt::RmsNormGatedGroup(q, to, tx, tg, &tw, a);
      });

  Add("L2NormKernel", "Qwen3.6-27B GDN q/k l2norm, prefill",
      "x[1024,16,128] f32", [](Queue& q, int reps) {
        auto x = F32(kPrefill * kGdnKeyDim, 11);
        std::vector<float> o(kPrefill * kGdnKeyDim);
        Tensor tx = T(x, {kPrefill, kGdnHk, kGdnDk}), to = T(o, {kPrefill, kGdnHk, kGdnDk});
        for (int i = 0; i < reps; ++i) vt::L2Norm(q, to, tx, vt::L2NormArgs{});
      });

  // --- Elementwise activations ----------------------------------------------
  Add("SiluAndMulKernel", "Qwen3.6-27B dense MLP act, prefill",
      "x[1024,34816] f32 -> out[1024,17408]", [](Queue& q, int reps) {
        auto x = F32(kPrefill * 2 * k27Inter, 12);
        std::vector<float> o(kPrefill * k27Inter);
        Tensor tx = T(x, {kPrefill, 2 * k27Inter}), to = T(o, {kPrefill, k27Inter});
        for (int i = 0; i < reps; ++i) vt::SiluAndMul(q, to, tx);
      });

  Add("GeluAndMulKernel", "Gemma GeGLU MLP act, prefill",
      "x[1024,34816] f32 -> out[1024,17408]", [](Queue& q, int reps) {
        auto x = F32(kPrefill * 2 * k27Inter, 13);
        std::vector<float> o(kPrefill * k27Inter);
        Tensor tx = T(x, {kPrefill, 2 * k27Inter}), to = T(o, {kPrefill, k27Inter});
        for (int i = 0; i < reps; ++i) vt::GeluAndMul(q, to, tx);
      });

  Add("MoeSiluMulKernel", "Qwen3.6-35B expert act, prefill x top_k",
      "gate/up[8192,512] f32", [](Queue& q, int reps) {
        const int64_t rows = kPrefill * k35TopK;
        auto g = F32(rows * k35MoeInter, 14), u = F32(rows * k35MoeInter, 15);
        std::vector<float> o(rows * k35MoeInter);
        Tensor tg = T(g, {rows, k35MoeInter}), tu = T(u, {rows, k35MoeInter});
        Tensor to = T(o, {rows, k35MoeInter});
        for (int i = 0; i < reps; ++i) vt::MoeSiluMul(q, to, tg, tu);
      });

  Add("MoeRelu2Kernel", "NemotronH expert relu^2, prefill x top_k",
      "x[8192,512] f32", [](Queue& q, int reps) {
        const int64_t rows = kPrefill * k35TopK;
        auto x = F32(rows * k35MoeInter, 16);
        std::vector<float> o(rows * k35MoeInter);
        Tensor tx = T(x, {rows, k35MoeInter}), to = T(o, {rows, k35MoeInter});
        for (int i = 0; i < reps; ++i) vt::MoeRelu2(q, to, tx);
      });

  Add("MulScalarKernel", "Gemma embedding normalizer, prefill",
      "x[1024,5120] f32", [](Queue& q, int reps) {
        auto x = F32(kPrefill * k27Hidden, 17);
        std::vector<float> o(kPrefill * k27Hidden);
        Tensor tx = T(x, {kPrefill, k27Hidden}), to = T(o, {kPrefill, k27Hidden});
        for (int i = 0; i < reps; ++i) vt::MulScalar(q, to, tx, 71.5541752799933f);
      });

  Add("SoftCapKernel", "Gemma-2 final logit soft-cap, decode",
      "x[16,248320] f32", [](Queue& q, int reps) {
        auto x = F32(kDecode * k27Vocab, 18);
        std::vector<float> o(kDecode * k27Vocab);
        Tensor tx = T(x, {kDecode, k27Vocab}), to = T(o, {kDecode, k27Vocab});
        for (int i = 0; i < reps; ++i) vt::SoftCap(q, to, tx, 30.0);
      });

  Add("CastBf16Kernel", "f32 -> bf16 activation cast, prefill",
      "x[1024,5120] f32 -> bf16", [](Queue& q, int reps) {
        auto x = F32(kPrefill * k27Hidden, 19);
        std::vector<uint16_t> o(kPrefill * k27Hidden);
        Tensor tx = T(x, {kPrefill, k27Hidden}), to = TB(o, {kPrefill, k27Hidden});
        for (int i = 0; i < reps; ++i) vt::CastBf16(q, to, tx);
      });

  Add("CastF32Kernel", "bf16 -> f32 GEMM-result cast, prefill",
      "x[1024,5120] bf16 -> f32", [](Queue& q, int reps) {
        auto x = Bf16(kPrefill * k27Hidden, 20);
        std::vector<float> o(kPrefill * k27Hidden);
        Tensor tx = TB(x, {kPrefill, k27Hidden}), to = T(o, {kPrefill, k27Hidden});
        for (int i = 0; i < reps; ++i) vt::CastF32(q, to, tx);
      });

  Add("CastF16Kernel", "EXL3 activation narrowing cast, prefill",
      "x[1024,5120] f32 -> f16", [](Queue& q, int reps) {
        auto x = F32(kPrefill * k27Hidden, 21);
        std::vector<uint16_t> o(kPrefill * k27Hidden);
        Tensor tx = T(x, {kPrefill, k27Hidden});
        Tensor to = Tensor::Contiguous(o.data(), DType::kF16, Cpu(), {kPrefill, k27Hidden});
        for (int i = 0; i < reps; ++i) vt::CastF16(q, to, tx);
      });

  Add("MulColVecF32Kernel", "merged fp8 projection per-shard dequant, prefill",
      "x[1024,10240] f32, col[10240]", [](Queue& q, int reps) {
        auto x = F32(kPrefill * kGdnConvDim, 22), c = F32(kGdnConvDim, 23);
        Tensor tx = T(x, {kPrefill, kGdnConvDim}), tc = T(c, {kGdnConvDim});
        for (int i = 0; i < reps; ++i) vt::MulColVecF32(q, tx, tc);
      });

  Add("EmbeddingKernel", "Qwen3.6-27B embed_tokens gather, prefill",
      "table[248320,5120] bf16, ids[1024]", [](Queue& q, int reps) {
        auto tbl = Bf16(static_cast<size_t>(k27Vocab) * k27Hidden, 24);
        auto ids = Iota32(kPrefill, 7, 211);
        std::vector<float> o(kPrefill * k27Hidden);
        Tensor tt = TB(tbl, {k27Vocab, k27Hidden}), ti = TI(ids, {kPrefill});
        Tensor to = T(o, {kPrefill, k27Hidden});
        for (int i = 0; i < reps; ++i) vt::Embedding(q, to, tt, ti);
      });

  // --- Splits and gates ------------------------------------------------------
  Add("QkvSplitKernel", "Qwen3.6-27B merged QKV split, prefill",
      "qkv[1024,8192] f32", [](Queue& q, int reps) {
        const int64_t qd = k27Heads * k27HeadDim, kd = k27KvHeads * k27HeadDim;
        auto x = F32(kPrefill * (qd + 2 * kd), 25);
        std::vector<float> oq(kPrefill * qd), ok(kPrefill * kd), ov(kPrefill * kd);
        Tensor tx = T(x, {kPrefill, qd + 2 * kd});
        Tensor tq = T(oq, {kPrefill, qd}), tk = T(ok, {kPrefill, kd}),
               tv = T(ov, {kPrefill, kd});
        for (int i = 0; i < reps; ++i) vt::QkvSplit(q, tq, tk, tv, tx);
      });

  Add("GdnConvSplitKernel", "Qwen3.6-27B GDN mixed-qkv split, prefill",
      "conv[1024,10240] f32", [](Queue& q, int reps) {
        auto x = F32(kPrefill * kGdnConvDim, 26);
        std::vector<float> oq(kPrefill * kGdnKeyDim), ok(kPrefill * kGdnKeyDim),
            ov(kPrefill * kGdnValueDim);
        Tensor tx = T(x, {kPrefill, kGdnConvDim});
        Tensor tq = T(oq, {kPrefill, kGdnKeyDim}), tk = T(ok, {kPrefill, kGdnKeyDim}),
               tv = T(ov, {kPrefill, kGdnValueDim});
        for (int i = 0; i < reps; ++i) vt::GdnConvSplit(q, tq, tk, tv, tx);
      });

  Add("AttnGateSplitKernel", "Qwen3.6-27B attn_output_gate q|gate split, prefill",
      "qgate[1024,12288] f32", [](Queue& q, int reps) {
        const int64_t qd = k27Heads * k27HeadDim;
        auto x = F32(kPrefill * 2 * qd, 27);
        std::vector<float> oq(kPrefill * qd), og(kPrefill * qd);
        Tensor tx = T(x, {kPrefill, 2 * qd});
        Tensor tq = T(oq, {kPrefill, k27Heads, k27HeadDim}),
               tg = T(og, {kPrefill, k27Heads, k27HeadDim});
        for (int i = 0; i < reps; ++i) vt::AttnGateSplit(q, tq, tg, tx);
      });

  Add("SigmoidGateBf16Kernel", "Qwen3.6-27B attention output gate, prefill",
      "attn/gate[1024,6144] f32 -> bf16", [](Queue& q, int reps) {
        const int64_t qd = k27Heads * k27HeadDim;
        auto a = F32(kPrefill * qd, 28), g = F32(kPrefill * qd, 29);
        std::vector<uint16_t> o(kPrefill * qd);
        Tensor ta = T(a, {kPrefill, qd}), tg = T(g, {kPrefill, qd});
        Tensor to = TB(o, {kPrefill, qd});
        for (int i = 0; i < reps; ++i) vt::SigmoidGateBf16(q, to, ta, tg);
      });

  Add("SharedExpertGateKernel", "Qwen3.6-35B shared-expert sigmoid gate, prefill",
      "sd[1024,2048] f32, gl[1024]", [](Queue& q, int reps) {
        auto sd = F32(kPrefill * k35Hidden, 30), gl = F32(kPrefill, 31);
        std::vector<uint16_t> o(kPrefill * k35Hidden);
        Tensor ts = T(sd, {kPrefill, k35Hidden}), tg = T(gl, {kPrefill});
        Tensor to = TB(o, {kPrefill, k35Hidden});
        for (int i = 0; i < reps; ++i) vt::SharedExpertGate(q, to, ts, tg);
      });

  Add("GdnGBetaKernel", "Qwen3.6-27B GDN decay/gate derivation, prefill",
      "araw/braw[1024,48] f32", [](Queue& q, int reps) {
        auto a = F32(kPrefill * kGdnHv, 32), b = F32(kPrefill * kGdnHv, 33);
        auto al = F32Pos(kGdnHv, 34), db = F32(kGdnHv, 35);
        std::vector<float> og(kPrefill * kGdnHv), ob(kPrefill * kGdnHv);
        Tensor ta = T(a, {kPrefill, kGdnHv}), tb = T(b, {kPrefill, kGdnHv});
        Tensor tal = T(al, {kGdnHv}), tdb = T(db, {kGdnHv});
        Tensor tg = T(og, {kPrefill, kGdnHv}), tbe = T(ob, {kPrefill, kGdnHv});
        for (int i = 0; i < reps; ++i) vt::GdnGBeta(q, tg, tbe, ta, tb, tal, tdb);
      });

  Add("GdnPostConvKernel", "Qwen3.6-27B fused GDN post-conv prep, prefill",
      "conv[1024,10240] f32", [](Queue& q, int reps) {
        auto c = F32(kPrefill * kGdnConvDim, 36), a = F32(kPrefill * kGdnHv, 37),
             b = F32(kPrefill * kGdnHv, 38), al = F32Pos(kGdnHv, 39), db = F32(kGdnHv, 40);
        std::vector<float> oq(kPrefill * kGdnKeyDim), ok(kPrefill * kGdnKeyDim),
            ov(kPrefill * kGdnValueDim), og(kPrefill * kGdnHv), ob(kPrefill * kGdnHv);
        Tensor tc = T(c, {kPrefill, kGdnConvDim}), ta = T(a, {kPrefill, kGdnHv}),
               tb = T(b, {kPrefill, kGdnHv}), tal = T(al, {kGdnHv}), tdb = T(db, {kGdnHv});
        Tensor tq = T(oq, {kPrefill, kGdnHk, kGdnDk}), tk = T(ok, {kPrefill, kGdnHk, kGdnDk}),
               tv = T(ov, {kPrefill, kGdnHv, kGdnDv}), tg = T(og, {kPrefill, kGdnHv}),
               tbe = T(ob, {kPrefill, kGdnHv});
        for (int i = 0; i < reps; ++i)
          vt::GdnPostConv(q, tq, tk, tv, tg, tbe, tc, ta, tb, tal, tdb, vt::L2NormArgs{});
      });

  // --- RoPE and the fused attention preamble ---------------------------------
  Add("RopeCosSinCacheKernel", "Qwen3.6-27B rotary cache build, prefill",
      "cos_sin[1024,64] f32", [](Queue& q, int reps) {
        auto pos = Iota32(kPrefill);
        std::vector<float> cs(kPrefill * k27RotaryDim);
        Tensor tp = TI(pos, {kPrefill}), tc = T(cs, {kPrefill, k27RotaryDim});
        vt::RopeArgs a;
        a.base = 1e7f;
        a.rotary_dim = static_cast<int>(k27RotaryDim);
        for (int i = 0; i < reps; ++i) vt::RopeCosSinCache(q, tc, tp, a);
      });

  Add("RopeRotateHead", "Qwen3.6-27B RopeNeox q/k in place, prefill",
      "q[1024,24,256] k[1024,4,256] f32, rot=64", [](Queue& q, int reps) {
        auto qs = F32(kPrefill * k27Heads * k27HeadDim, 41),
             ks = F32(kPrefill * k27KvHeads * k27HeadDim, 42);
        auto pos = Iota32(kPrefill);
        Tensor tq = T(qs, {kPrefill, k27Heads, k27HeadDim}),
               tk = T(ks, {kPrefill, k27KvHeads, k27HeadDim}), tp = TI(pos, {kPrefill});
        vt::RopeArgs a;
        a.base = 1e7f;
        a.rotary_dim = static_cast<int>(k27RotaryDim);
        for (int i = 0; i < reps; ++i) vt::RopeNeox(q, tq, tk, tp, a);
      });

  Add("RopeFromCacheKernel", "Qwen3.6-27B RopeFromCache q/k, prefill",
      "q[1024,24,256] k[1024,4,256] f32, cache[1024,64]", [](Queue& q, int reps) {
        auto qs = F32(kPrefill * k27Heads * k27HeadDim, 43),
             ks = F32(kPrefill * k27KvHeads * k27HeadDim, 44), cs = F32(kPrefill * k27RotaryDim, 45);
        auto pos = Iota32(kPrefill);
        Tensor tq = T(qs, {kPrefill, k27Heads, k27HeadDim}),
               tk = T(ks, {kPrefill, k27KvHeads, k27HeadDim}), tp = TI(pos, {kPrefill}),
               tc = T(cs, {kPrefill, k27RotaryDim});
        vt::RopeArgs a;
        a.base = 1e7f;
        a.rotary_dim = static_cast<int>(k27RotaryDim);
        for (int i = 0; i < reps; ++i) vt::RopeFromCache(q, tq, &tk, tp, tc, a);
      });

  Add("AttnQkNormRopeGateKernel", "Qwen3.6-27B fused attention preamble, prefill",
      "qgate[1024,12288] kf[1024,1024] f32", [](Queue& q, int reps) {
        const int64_t qw = k27Heads * k27HeadDim, kw = k27KvHeads * k27HeadDim;
        auto qg = F32(kPrefill * 2 * qw, 46), kf = F32(kPrefill * kw, 47),
             qn = F32(k27HeadDim, 48), kn = F32(k27HeadDim, 49), cs = F32(kPrefill * k27RotaryDim, 50);
        std::vector<float> oq(kPrefill * qw), ok(kPrefill * kw), og(kPrefill * qw);
        Tensor tqg = T(qg, {kPrefill, 2 * qw}), tkf = T(kf, {kPrefill, kw});
        Tensor tqn = T(qn, {k27HeadDim}), tkn = T(kn, {k27HeadDim}),
               tcs = T(cs, {kPrefill, k27RotaryDim});
        Tensor toq = T(oq, {kPrefill, k27Heads, k27HeadDim}),
               tok = T(ok, {kPrefill, k27KvHeads, k27HeadDim}),
               tog = T(og, {kPrefill, k27Heads, k27HeadDim});
        vt::RopeArgs ra;
        ra.base = 1e7f;
        ra.rotary_dim = static_cast<int>(k27RotaryDim);
        for (int i = 0; i < reps; ++i)
          vt::AttnQkNormRopeGate(q, toq, tok, tog, tqg, tkf, tqn, tkn, tcs,
                                 vt::RmsNormArgs{1e-6f, true}, ra);
      });

  Add("FusedNormRopeKernel", "DeepSeek-V3 MLA fused kv_a norm+rope, prefill",
      "x[1024,576] f32, latent 512 | pe 64", [](Queue& q, int reps) {
        constexpr int64_t kLora = 512, kPe = 64;
        auto x = F32(kPrefill * (kLora + kPe), 51), w = F32(kLora, 52),
             cs = F32(kPrefill * kPe, 53);
        auto pos = Iota32(kPrefill);
        std::vector<float> ol(kPrefill * kLora), op(kPrefill * kPe);
        Tensor tx = T(x, {kPrefill, kLora + kPe}), tw = T(w, {kLora}),
               tp = TI(pos, {kPrefill}), tcs = T(cs, {kPrefill, kPe});
        Tensor tl = T(ol, {kPrefill, kLora}), tpe = T(op, {kPrefill, kPe});
        vt::RopeArgs ra;
        ra.base = 1e4f;
        ra.rotary_dim = static_cast<int>(kPe);
        for (int i = 0; i < reps; ++i)
          vt::FusedNormRope(q, tl, tpe, tx, tw, tp, tcs, vt::RmsNormArgs{}, ra);
      });

  Add("ConcatMlaNopeRopeKernel", "DeepSeek-V3 MLA nope|rope head concat, prefill",
      "nope[1024,128,128] rope[1024,1,64] f32", [](Queue& q, int reps) {
        constexpr int64_t kH = 128, kDn = 128, kDr = 64;
        auto n = F32(kPrefill * kH * kDn, 54), r = F32(kPrefill * kDr, 55);
        std::vector<float> o(kPrefill * kH * (kDn + kDr));
        Tensor tn = T(n, {kPrefill, kH, kDn}), tr = T(r, {kPrefill, 1, kDr});
        Tensor to = T(o, {kPrefill, kH, kDn + kDr});
        for (int i = 0; i < reps; ++i) vt::ConcatMlaNopeRope(q, to, tn, tr);
      });

  // --- Convolutions ----------------------------------------------------------
  Add("CausalConv1dFwdKernel", "Qwen3.6-27B GDN prefill conv, 1024 tokens",
      "x[1024,10240] f32, w[10240,4]", [](Queue& q, int reps) {
        auto x = F32(kGdnConvDim * kPrefill, 56), w = F32(kGdnConvDim * kGdnK, 57),
             b = F32(kGdnConvDim, 58);
        std::vector<float> st(kGdnConvDim * (kGdnK - 1), 0.0f), o(kGdnConvDim * kPrefill);
        std::vector<int32_t> qsl = {0, static_cast<int32_t>(kPrefill)}, has = {0};
        Tensor tx = T(x, {kPrefill, kGdnConvDim}), tw = T(w, {kGdnConvDim, kGdnK}),
               tb = T(b, {kGdnConvDim});
        Tensor ts = T(st, {1, kGdnConvDim, kGdnK - 1}), to = T(o, {kPrefill, kGdnConvDim});
        Tensor tq = TI(qsl, {2}), th = TI(has, {1});
        for (int i = 0; i < reps; ++i)
          vt::CausalConv1dFwd(q, to, tx, tw, &tb, ts, tq, th, vt::CausalConv1dArgs{});
      });

  Add("CausalConv1dUpdateKernel", "Qwen3.6-27B GDN decode conv step, c16",
      "x[16,10240] f32, state[16,10240,3]", [](Queue& q, int reps) {
        auto x = F32(kDecode * kGdnConvDim, 59), w = F32(kGdnConvDim * kGdnK, 60),
             b = F32(kGdnConvDim, 61);
        std::vector<float> st(kDecode * kGdnConvDim * (kGdnK - 1), 0.1f),
            o(kDecode * kGdnConvDim);
        Tensor tx = T(x, {kDecode, kGdnConvDim}), tw = T(w, {kGdnConvDim, kGdnK}),
               tb = T(b, {kGdnConvDim});
        Tensor ts = T(st, {kDecode, kGdnConvDim, kGdnK - 1}), to = T(o, {kDecode, kGdnConvDim});
        for (int i = 0; i < reps; ++i)
          vt::CausalConv1dUpdate(q, to, tx, tw, &tb, ts, vt::CausalConv1dArgs{});
      });

  // --- Mamba2 ---------------------------------------------------------------
  Add("Mamba2StateUpdateKernel", "NemotronH Mamba2 decode step, c16",
      "state[16,128,64,128] f32", [](Queue& q, int reps) {
        constexpr int64_t kH = 128, kP = 64, kN = 128, kG = 8;
        auto st = F32(kDecode * kH * kP * kN, 62), x = F32(kDecode * kH * kP, 63),
             dt = F32Pos(kDecode * kH, 64), A = F32Pos(kH, 65), B = F32(kDecode * kG * kN, 66),
             C = F32(kDecode * kG * kN, 67), D = F32(kH, 68), z = F32(kDecode * kH * kP, 69),
             dtb = F32(kH, 70);
        std::vector<float> o(kDecode * kH * kP);
        for (float& a : A) a = -a;
        Tensor tst = T(st, {kDecode, kH, kP, kN}), tx = T(x, {kDecode, kH, kP}),
               tdt = T(dt, {kDecode, kH}), tA = T(A, {kH}), tB = T(B, {kDecode, kG, kN}),
               tC = T(C, {kDecode, kG, kN}), tD = T(D, {kH}), tz = T(z, {kDecode, kH, kP}),
               tdtb = T(dtb, {kH}), to = T(o, {kDecode, kH, kP});
        vt::Mamba2Args a;
        a.dt_softplus = true;
        for (int i = 0; i < reps; ++i)
          vt::Mamba2StateUpdate(q, to, tst, tx, tdt, tA, tB, tC, &tD, &tz, &tdtb, nullptr, a);
      });

  // --- GDN recurrences -------------------------------------------------------
  Add("GdnDecodeKernel", "Qwen3.6-27B GDN decode recurrence, c16",
      "state[16,48,128,128] f32", [](Queue& q, int reps) {
        auto qi = F32(kDecode * kGdnHk * kGdnDk, 71), k = F32(kDecode * kGdnHk * kGdnDk, 72),
             v = F32(kDecode * kGdnHv * kGdnDv, 73), g = F32(kDecode * kGdnHv, 74),
             be = F32Pos(kDecode * kGdnHv, 75), st = F32(kDecode * kGdnHv * kGdnDv * kGdnDk, 76);
        std::vector<float> o(kDecode * kGdnHv * kGdnDv);
        Tensor tq = T(qi, {kDecode, kGdnHk, kGdnDk}), tk = T(k, {kDecode, kGdnHk, kGdnDk}),
               tv = T(v, {kDecode, kGdnHv, kGdnDv}), tg = T(g, {kDecode, kGdnHv}),
               tb = T(be, {kDecode, kGdnHv}),
               ts = T(st, {kDecode, kGdnHv, kGdnDv, kGdnDk}),
               to = T(o, {kDecode, kGdnHv, kGdnDv});
        vt::GdnArgs a;
        a.scale = 1.0f / std::sqrt(static_cast<float>(kGdnDk));
        for (int i = 0; i < reps; ++i) vt::GdnDecode(q, to, tq, tk, tv, tg, tb, ts, a);
      });

  Add("KdaHeadTokenStep", "Kimi KDA per-channel decay recurrence, c16 prefill",
      "T=64, state[1,48,128,128] f32", [](Queue& q, int reps) {
        constexpr int64_t kT = 64;
        auto qi = F32(kT * kGdnHk * kGdnDk, 77), k = F32(kT * kGdnHk * kGdnDk, 78),
             v = F32(kT * kGdnHv * kGdnDv, 79), g = F32(kT * kGdnHv * kGdnDk, 80),
             be = F32Pos(kT * kGdnHv, 81), st = F32(kGdnHv * kGdnDv * kGdnDk, 82);
        std::vector<float> o(kT * kGdnHv * kGdnDv);
        std::vector<int32_t> qsl = {0, static_cast<int32_t>(kT)};
        Tensor tq = T(qi, {kT, kGdnHk, kGdnDk}), tk = T(k, {kT, kGdnHk, kGdnDk}),
               tv = T(v, {kT, kGdnHv, kGdnDv}), tg = T(g, {kT, kGdnHv, kGdnDk}),
               tb = T(be, {kT, kGdnHv}), ts = T(st, {1, kGdnHv, kGdnDv, kGdnDk}),
               to = T(o, {kT, kGdnHv, kGdnDv}), tql = TI(qsl, {2});
        vt::GdnArgs a;
        a.scale = 1.0f / std::sqrt(static_cast<float>(kGdnDk));
        a.query_start_loc_host = qsl.data();
        for (int i = 0; i < reps; ++i)
          vt::KdaGatedDeltaRule(q, to, tq, tk, tv, tg, tb, ts, tql, a);
      });

  Add("GdnStateGatherKernel", "Qwen3.6-27B GDN state gather, c16",
      "cache[64,48,128,128] f32 -> working[16,...]", [](Queue& q, int reps) {
        const int64_t slots = 64, per = kGdnHv * kGdnDv * kGdnDk;
        auto cache = F32(slots * per, 83);
        std::vector<float> w(kDecode * per);
        auto idx = Iota32(kDecode, 0, 3);
        Tensor tc = T(cache, {slots, kGdnHv, kGdnDv, kGdnDk}),
               tw = T(w, {kDecode, kGdnHv, kGdnDv, kGdnDk}), ti = TI(idx, {kDecode});
        for (int i = 0; i < reps; ++i) vt::GdnStateGather(q, tw, tc, ti);
      });

  Add("GdnStateScatterKernel", "Qwen3.6-27B GDN state scatter, c16",
      "working[16,...] -> cache[64,48,128,128] f32", [](Queue& q, int reps) {
        const int64_t slots = 64, per = kGdnHv * kGdnDv * kGdnDk;
        std::vector<float> cache(slots * per, 0.0f);
        auto w = F32(kDecode * per, 84);
        auto idx = Iota32(kDecode, 0, 3);
        Tensor tc = T(cache, {slots, kGdnHv, kGdnDv, kGdnDk}),
               tw = T(w, {kDecode, kGdnHv, kGdnDv, kGdnDk}), ti = TI(idx, {kDecode});
        for (int i = 0; i < reps; ++i) vt::GdnStateScatter(q, tc, tw, ti);
      });

  // --- MoE glue --------------------------------------------------------------
  Add("MoeRouterTopKKernel", "Qwen3.6-35B router softmax top-8 of 256, prefill",
      "logits[1024,256] f32", [](Queue& q, int reps) {
        auto lg = F32(kPrefill * k35Experts, 85);
        std::vector<float> w(kPrefill * k35TopK);
        std::vector<int32_t> ix(kPrefill * k35TopK);
        Tensor tl = T(lg, {kPrefill, k35Experts}), tw = T(w, {kPrefill, k35TopK}),
               ti = TI(ix, {kPrefill, k35TopK});
        vt::MoeRouterTopKArgs a;
        a.top_k = static_cast<int>(k35TopK);
        for (int i = 0; i < reps; ++i) vt::MoeRouterTopK(q, tw, ti, tl, a);
      });

  Add("MoeRouterGroupedTopKKernel", "DeepSeek grouped sigmoid router, prefill",
      "logits[1024,256] f32, groups 8, topk_group 4", [](Queue& q, int reps) {
        auto lg = F32(kPrefill * k35Experts, 86), bias = F32(k35Experts, 87);
        std::vector<float> w(kPrefill * k35TopK);
        std::vector<int32_t> ix(kPrefill * k35TopK);
        Tensor tl = T(lg, {kPrefill, k35Experts}), tw = T(w, {kPrefill, k35TopK}),
               ti = TI(ix, {kPrefill, k35TopK}), tb = T(bias, {k35Experts});
        vt::MoeRouterTopKArgs a;
        a.top_k = static_cast<int>(k35TopK);
        a.scoring_func = vt::MoeScoringFunc::kSigmoid;
        a.num_expert_group = 8;
        a.topk_group = 4;
        for (int i = 0; i < reps; ++i) vt::MoeRouterTopK(q, tw, ti, tl, a, &tb);
      });

  Add("MoeCombineKernel", "Qwen3.6-35B weighted expert combine, prefill",
      "expert_out[1024,8,2048] f32", [](Queue& q, int reps) {
        auto eo = F32(kPrefill * k35TopK * k35Hidden, 88), w = F32(kPrefill * k35TopK, 89);
        std::vector<float> o(kPrefill * k35Hidden);
        Tensor te = T(eo, {kPrefill, k35TopK, k35Hidden}), tw = T(w, {kPrefill, k35TopK}),
               to = T(o, {kPrefill, k35Hidden});
        for (int i = 0; i < reps; ++i) vt::MoeCombine(q, to, te, tw);
      });

  // --- GEMM -----------------------------------------------------------------
  Add("MatmulOneChunk", "LTX-2.5 connector attn projection, 1024 rows",
      "a[1024,4096] x b[4096,4096]^T f32", [](Queue& q, int reps) {
        auto a = F32(kLtxRows * kLtxDim, 90), b = F32(kLtxDim * kLtxDim, 91);
        std::vector<float> o(kLtxRows * kLtxDim);
        Tensor ta = T(a, {kLtxRows, kLtxDim}), tb = T(b, {kLtxDim, kLtxDim}),
               to = T(o, {kLtxRows, kLtxDim});
        for (int i = 0; i < reps; ++i) vt::MatmulBT(q, to, ta, tb);
      });

  Add("BatchedMatmulKernel", "LTX-2.5 per-head batched GEMM, 32 heads",
      "a[32,1024,128] x b[32,128,128] f32", [](Queue& q, int reps) {
        auto a = F32(kLtxHeads * kLtxRows * kLtxHeadDim, 92),
             b = F32(kLtxHeads * kLtxHeadDim * kLtxHeadDim, 93);
        std::vector<float> o(kLtxHeads * kLtxRows * kLtxHeadDim);
        Tensor ta = T(a, {kLtxHeads, kLtxRows, kLtxHeadDim}),
               tb = T(b, {kLtxHeads, kLtxHeadDim, kLtxHeadDim}),
               to = T(o, {kLtxHeads, kLtxRows, kLtxHeadDim});
        for (int i = 0; i < reps; ++i) vt::BatchedMatmul(q, to, ta, tb);
      });

  // --- Attention (the two kernels VT-CPU-ELEM-DISPATCH already hoisted; they
  // are here as the CALIBRATION point, not as candidates). -------------------
  Add("AttentionKernel", "LTX-2.5 DiT self-attention, video stream",
      "q/k/v[1024,32,128] f32, causal", [](Queue& q, int reps) {
        auto a = F32(kLtxRows * kLtxHeads * kLtxHeadDim, 94),
             k = F32(kLtxRows * kLtxHeads * kLtxHeadDim, 95),
             v = F32(kLtxRows * kLtxHeads * kLtxHeadDim, 96);
        std::vector<float> o(kLtxRows * kLtxHeads * kLtxHeadDim);
        Tensor tq = T(a, {kLtxRows, kLtxHeads, kLtxHeadDim}),
               tk = T(k, {kLtxRows, kLtxHeads, kLtxHeadDim}),
               tv = T(v, {kLtxRows, kLtxHeads, kLtxHeadDim}),
               to = T(o, {kLtxRows, kLtxHeads, kLtxHeadDim});
        vt::AttentionArgs a2;
        a2.scale = 1.0f / std::sqrt(static_cast<float>(kLtxHeadDim));
        for (int i = 0; i < reps; ++i) vt::Attention(q, to, tq, tk, tv, a2);
      });

  Add("AttentionCrossKernel", "LTX-2.5 connector cross-attention, video stream",
      "q/k/v[1024,32,128] f32", [](Queue& q, int reps) {
        auto a = F32(kLtxRows * kLtxHeads * kLtxHeadDim, 97),
             k = F32(kLtxRows * kLtxHeads * kLtxHeadDim, 98),
             v = F32(kLtxRows * kLtxHeads * kLtxHeadDim, 99);
        std::vector<float> o(kLtxRows * kLtxHeads * kLtxHeadDim);
        Tensor tq = T(a, {kLtxRows, kLtxHeads, kLtxHeadDim}),
               tk = T(k, {kLtxRows, kLtxHeads, kLtxHeadDim}),
               tv = T(v, {kLtxRows, kLtxHeads, kLtxHeadDim}),
               to = T(o, {kLtxRows, kLtxHeads, kLtxHeadDim});
        vt::AttentionCrossArgs a2;
        a2.scale = 1.0f / std::sqrt(static_cast<float>(kLtxHeadDim));
        for (int i = 0; i < reps; ++i)
          vt::AttentionCross(q, to, tq, tk, tv, nullptr, a2);
      });

  // --- FP8 / FP4 quantization -----------------------------------------------
  Add("QuantFp8StaticKernel", "Qwen3.6-27B static per-tensor fp8 activation quant, prefill",
      "x[1024,5120] f32 -> e4m3", [](Queue& q, int reps) {
        auto x = F32(kPrefill * k27Hidden, 100);
        std::vector<int8_t> o(kPrefill * k27Hidden);
        Tensor tx = T(x, {kPrefill, k27Hidden});
        Tensor to = Tensor::Contiguous(o.data(), DType::kI8, Cpu(), {kPrefill, k27Hidden});
        for (int i = 0; i < reps; ++i) vt::QuantFp8Static(q, to, tx, 0.5f);
      });

  Add("QuantFp8GroupKernel", "DeepSeek block-fp8 per-128-group activation quant, prefill",
      "x[1024,5120] f32, group 128", [](Queue& q, int reps) {
        auto x = F32(kPrefill * k27Hidden, 101);
        std::vector<int8_t> o(kPrefill * k27Hidden);
        std::vector<float> sc(kPrefill * (k27Hidden / 128));
        Tensor tx = T(x, {kPrefill, k27Hidden});
        Tensor to = Tensor::Contiguous(o.data(), DType::kI8, Cpu(), {kPrefill, k27Hidden});
        Tensor ts = T(sc, {kPrefill, k27Hidden / 128});
        for (int i = 0; i < reps; ++i) vt::QuantFp8Group(q, to, ts, tx, 128);
      });

  Add("RmsNormQuantFp8Kernel", "Qwen3.6-27B fused RMSNorm -> fp8, prefill",
      "x[1024,5120] f32 -> e4m3 + bf16", [](Queue& q, int reps) {
        auto x = F32(kPrefill * k27Hidden, 102), w = F32(k27Hidden, 103);
        std::vector<int8_t> o(kPrefill * k27Hidden);
        std::vector<uint16_t> ob(kPrefill * k27Hidden);
        Tensor tx = T(x, {kPrefill, k27Hidden}), tw = T(w, {k27Hidden});
        Tensor to = Tensor::Contiguous(o.data(), DType::kI8, Cpu(), {kPrefill, k27Hidden});
        Tensor tob = TB(ob, {kPrefill, k27Hidden});
        for (int i = 0; i < reps; ++i)
          vt::RmsNormQuantFp8(q, to, &tob, tx, tw, vt::RmsNormArgs{}, nullptr, 0.5f);
      });

  Add("RmsNormGatedQuantFp8Kernel", "Qwen3.6-27B GDN gated norm -> fp8, decode",
      "x/gate[16,48,128] f32 -> e4m3", [](Queue& q, int reps) {
        auto x = F32(kDecode * kGdnValueDim, 104), g = F32(kDecode * kGdnValueDim, 105),
             w = F32(kGdnDv, 106);
        std::vector<int8_t> o(kDecode * kGdnValueDim);
        Tensor tx = T(x, {kDecode, kGdnHv, kGdnDv}), tg = T(g, {kDecode, kGdnHv, kGdnDv}),
               tw = T(w, {kGdnDv});
        Tensor to = Tensor::Contiguous(o.data(), DType::kI8, Cpu(),
                                       {kDecode, kGdnHv, kGdnDv});
        for (int i = 0; i < reps; ++i)
          vt::RmsNormGatedQuantFp8(q, to, tx, tg, tw, vt::RmsNormGatedArgs{}, 0.5f);
      });

  // --- DFlash draft model (hidden 5120, 32 q-heads / 8 kv-heads / head_dim 128,
  // include/vllm/model_executor/models/qwen3_dflash.h:14; block = 1 + 15
  // speculative tokens, .agents/specs/dflash-spec-decode.md:70). ------------
  Add("DFlashBlockAttentionKernel", "Qwen3-DFlash in-block attention, 16 reqs x 16-token block",
      "q/k/v[256,32,128] f32", [](Queue& q, int reps) {
        constexpr int64_t kHq = 32, kD = 128, kBlk = 16;
        const int64_t nq = kDecode * kBlk;
        auto a = F32(nq * kHq * kD, 108), k = F32(nq * kHq * kD, 109),
             v = F32(nq * kHq * kD, 110);
        std::vector<float> o(nq * kHq * kD);
        std::vector<int32_t> cu(kDecode + 1);
        for (size_t i = 0; i < cu.size(); ++i) cu[i] = static_cast<int32_t>(i) * kBlk;
        Tensor tq = T(a, {nq, kHq, kD}), tk = T(k, {nq, kHq, kD}), tv = T(v, {nq, kHq, kD});
        Tensor to = T(o, {nq, kHq, kD});
        vt::DFlashBlockAttentionArgs ar;
        ar.scale = 1.0f / std::sqrt(static_cast<float>(kD));
        ar.cu_seqlens = cu.data();
        ar.num_reqs = static_cast<int>(kDecode);
        for (int i = 0; i < reps; ++i) vt::DFlashBlockAttention(q, to, tq, tk, tv, ar);
      });

  Add("DFlashGroupedConvKernel", "Qwen3-DFlash grouped conv prepare, 16 reqs x 16-token block",
      "x[256,5120] f32, taps 4, groups 40", [](Queue& q, int reps) {
        constexpr int64_t kBlk = 16, kTaps = 4, kGroup = 128;
        const int64_t nq = kDecode * kBlk, groups = k27Hidden / kGroup;
        auto x = F32(nq * k27Hidden, 111),
             c = F32(nq * 2 * kTaps * groups, 112), b = F32(2 * kTaps * k27Hidden, 113);
        std::vector<float> o(nq * k27Hidden);
        Tensor tx = T(x, {nq, k27Hidden}), tc = T(c, {nq, 2, kTaps, groups}),
               tb = T(b, {2, kTaps, k27Hidden}), to = T(o, {nq, k27Hidden});
        vt::DFlashGroupedConvArgs ar;
        ar.block_size = kBlk;
        ar.taps = kTaps;
        ar.num_groups = groups;
        ar.group_size = kGroup;
        ar.side = 0;
        for (int i = 0; i < reps; ++i) vt::DFlashGroupedConv(q, to, tx, tc, tb, ar);
      });

  Add("CausalConv1dSpecUpdateKernel", "Qwen3.6-27B GDN speculative conv step, c16 x 4",
      "x[64,10240] f32", [](Queue& q, int reps) {
        constexpr int64_t kSpec = 4;
        const int64_t nt = kDecode * kSpec;
        auto x = F32(nt * kGdnConvDim, 114), w = F32(kGdnConvDim * kGdnK, 115),
             b = F32(kGdnConvDim, 116);
        std::vector<float> st(kDecode * kGdnConvDim * (kGdnK - 1 + kSpec), 0.1f),
            o(nt * kGdnConvDim);
        auto idx = Iota32(kDecode);
        std::vector<int32_t> acc(kDecode, 1), cu(kDecode + 1);
        for (size_t i = 0; i < cu.size(); ++i) cu[i] = static_cast<int32_t>(i) * kSpec;
        Tensor tx = T(x, {nt, kGdnConvDim}), tw = T(w, {kGdnConvDim, kGdnK}),
               tb = T(b, {kGdnConvDim});
        Tensor ts = T(st, {kDecode, kGdnConvDim, kGdnK - 1 + kSpec}),
               to = T(o, {nt, kGdnConvDim});
        Tensor ti = TI(idx, {kDecode}), ta = TI(acc, {kDecode}), tc = TI(cu, {kDecode + 1});
        for (int i = 0; i < reps; ++i)
          vt::CausalConv1dSpecUpdate(q, to, tx, tw, &tb, ts, ti, ta, tc,
                                     vt::CausalConv1dArgs{});
      });

  Add("ScaledFp4QuantKernel", "Qwen3.6-27B NVFP4 activation quant, prefill",
      "x[1024,5120] f32 -> fp4 + e4m3 scales", [](Queue& q, int reps) {
        auto x = F32(kPrefill * k27Hidden, 107);
        std::vector<int8_t> o(kPrefill * k27Hidden / 2), sc(kPrefill * k27Hidden / 16);
        Tensor tx = T(x, {kPrefill, k27Hidden});
        Tensor to = Tensor::Contiguous(o.data(), DType::kI8, Cpu(), {kPrefill, k27Hidden / 2});
        Tensor ts = Tensor::Contiguous(sc.data(), DType::kI8, Cpu(),
                                       {kPrefill, k27Hidden / 16});
        for (int i = 0; i < reps; ++i) vt::ScaledFp4Quant(q, to, ts, tx, 2.0f);
      });
}

}  // namespace

int main(int argc, char** argv) {
  RegisterCases();
  std::string op;
  int reps = 1;
  bool list = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--list") list = true;
    else if (a == "--op" && i + 1 < argc) op = argv[++i];
    else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
  }
  if (list || op.empty()) {
    for (const Case& c : Registry())
      std::printf("%-34s %-52s %s\n", c.name, c.shape, c.where);
    std::printf("# %zu cases\n", Registry().size());
    return list ? 0 : 2;
  }
  for (const Case& c : Registry()) {
    if (op != c.name) continue;
    // One untimed warm-up so first-touch page faults land outside the sample.
    Queue q{Cpu(), nullptr};
    c.run(q, 1);
    const Clock::time_point t0 = Clock::now();
    c.run(q, reps);
    const double s = Seconds(t0, Clock::now());
    std::printf("op=%s reps=%d seconds=%.4f per_rep_ms=%.4f shape=%s from=%s\n",
                c.name, reps, s, 1000.0 * s / reps, c.shape, c.where);
    return 0;
  }
  std::fprintf(stderr, "unknown op: %s\n", op.c_str());
  return 2;
}
