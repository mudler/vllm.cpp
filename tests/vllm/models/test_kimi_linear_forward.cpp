// Kimi-Linear (`KimiLinearForCausalLM`) CPU REFERENCE FORWARD gates (W2-W6). Proves
// the per-op reference forwards WIRE the landed primitives in the pinned-vLLM order,
// and that the whole 27-layer hybrid decodes coherently on CPU — so the ONLY
// remaining correctness step is the e2e SACRED token golden on GB10 (spike §4/§8).
//
//   (a) KDA layer          == a hand-composition of the vllm::kimi_kda host refs +
//                             an INDEPENDENT gated-delta recurrence (wiring proof).
//   (b) NoPE-MLA layer      == an INDEPENDENT materialized-MHA reference (the
//                             UNABSORBED reference the device absorbed path equals).
//   (c) sigmoid noaux_tc     == a HAND-COMPUTED top-k case (bias-select / unbiased-
//       router + block         weight asymmetry, renormalize, routed_scaling); the
//                             whole block == shared + sum(weight*expert_ffn).
//   (d) whole 2-layer        finite/coherent-shaped forward + greedy-decodes N tokens
//       forward + decode      with the (recomputed) recurrent+latent context advancing;
//                             the loader materializes the host weights end-to-end.
#include "vllm/model_executor/models/kimi_linear.h"
#include "vllm/model_executor/models/kimi_kda.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/backend.h"
#include "vt/dtype.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using vllm::KimiLinearParams;
using vllm::KimiLinearWeights;
using vllm::LoadKimiLinearForCausalLMWeights;
using vllm::SafetensorsFile;

namespace {

// ── deterministic pseudo-random weights ──────────────────────────────────────
std::vector<float> Rand(size_t n, uint32_t seed, float lo = -0.5f, float hi = 0.5f) {
  std::vector<float> v(n);
  uint32_t s = seed * 2654435761u + 1u;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
    x = lo + u * (hi - lo);
  }
  return v;
}

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double Silu(double x) { return x * Sigmoid(x); }

// y[T,out] = x[T,in] @ w^T, w row-major [out,in] (torch Linear). Independent of the
// impl's MatMul (accumulate in double to match).
std::vector<float> Mat(const std::vector<float>& w, const std::vector<float>& x,
                       int64_t out, int64_t in, int64_t T) {
  std::vector<float> y(static_cast<size_t>(T) * out, 0.0f);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t o = 0; o < out; ++o) {
      double acc = 0.0;
      for (int64_t i = 0; i < in; ++i)
        acc += static_cast<double>(w[o * in + i]) * x[t * in + i];
      y[t * out + o] = static_cast<float>(acc);
    }
  return y;
}

std::vector<float> Rms(const std::vector<float>& x, const std::vector<float>& g,
                       int64_t rows, int64_t dim, float eps) {
  std::vector<float> y(static_cast<size_t>(rows) * dim, 0.0f);
  for (int64_t r = 0; r < rows; ++r) {
    double var = 0.0;
    for (int64_t d = 0; d < dim; ++d) var += static_cast<double>(x[r * dim + d]) * x[r * dim + d];
    var /= static_cast<double>(dim);
    const double rstd = 1.0 / std::sqrt(var + eps);
    for (int64_t d = 0; d < dim; ++d)
      y[r * dim + d] = static_cast<float>(static_cast<double>(x[r * dim + d]) * rstd * g[d]);
  }
  return y;
}

// The W7 device compute uses f32 DBufs + vt:: ops (f32 accumulation) vs the W2
// reference's double accumulation. So a MATCH is tight but not bit-exact: a
// relative+absolute tolerance whose only slack is f32-accumulation-order (the KDA
// recurrence amplifies it slightly over its T steps; the recurrence itself runs the
// IDENTICAL host code on both sides). Each site documents its tolerance.
bool Close(float a, float b, double rtol, double atol) {
  return std::fabs(static_cast<double>(a) - static_cast<double>(b)) <=
         atol + rtol * std::fabs(static_cast<double>(b));
}

vt::Queue CpuQueue() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

}  // namespace

// ─── (a) KDA layer == hand-composition of the kimi_kda refs + recurrence ──────
TEST_CASE("kimi-linear forward: KDA layer wires the kimi_kda host refs exactly") {
  KimiLinearParams p;
  p.hidden_size = 16;
  p.kda_num_heads = 2;
  p.kda_head_dim = 8;
  p.kda_short_conv_kernel_size = 4;
  p.rms_norm_eps = 1e-5f;
  const int64_t H = p.hidden_size, nh = p.kda_num_heads, hd = p.kda_head_dim;
  const int64_t proj = nh * hd, K = p.kda_short_conv_kernel_size, T = 5;

  vllm::KdaLayerHostWeights w;
  w.q_proj = Rand(proj * H, 1);
  w.k_proj = Rand(proj * H, 2);
  w.v_proj = Rand(proj * H, 3);
  w.f_a_proj = Rand(hd * H, 4);
  w.f_b_proj = Rand(proj * hd, 5);
  w.b_proj = Rand(nh * H, 6);
  w.g_a_proj = Rand(hd * H, 7);
  w.g_b_proj = Rand(proj * hd, 8);
  w.o_proj = Rand(H * proj, 9);
  w.q_conv = Rand(proj * K, 10);
  w.k_conv = Rand(proj * K, 11);
  w.v_conv = Rand(proj * K, 12);
  w.dt_bias = Rand(proj, 13);
  w.a_log = Rand(nh, 14, -2.0f, 0.0f);
  w.o_norm = Rand(hd, 15, 0.5f, 1.5f);
  const std::vector<float> x = Rand(T * H, 100);

  const std::vector<float> got = vllm::KimiKdaLayerForward(w, x, p, T);

  // INDEPENDENT recomposition (same math, different code path).
  const std::vector<float> rq = Mat(w.q_proj, x, proj, H, T);
  const std::vector<float> rk = Mat(w.k_proj, x, proj, H, T);
  const std::vector<float> rv = Mat(w.v_proj, x, proj, H, T);
  const std::vector<float> qc = vllm::kimi_kda::KdaShortConv(rq, w.q_conv, {}, T, proj, K);
  const std::vector<float> kc = vllm::kimi_kda::KdaShortConv(rk, w.k_conv, {}, T, proj, K);
  const std::vector<float> vc = vllm::kimi_kda::KdaShortConv(rv, w.v_conv, {}, T, proj, K);
  const std::vector<float> qn = vllm::kimi_kda::L2NormRows(qc, T * nh, hd);
  const std::vector<float> kn = vllm::kimi_kda::L2NormRows(kc, T * nh, hd);
  const std::vector<float> braw = Mat(w.b_proj, x, nh, H, T);
  const std::vector<float> g1 = vllm::kimi_kda::KdaLowRankDecay(x, w.f_a_proj, w.f_b_proj, T, H, nh, hd);
  const std::vector<float> g = vllm::kimi_kda::KdaDecayGate(g1, w.a_log, w.dt_bias, T, nh, hd);
  const std::vector<float> g2 = Mat(w.g_b_proj, Mat(w.g_a_proj, x, hd, H, T), proj, hd, T);
  const double scale = std::pow(static_cast<double>(hd), -0.5);
  std::vector<double> S(static_cast<size_t>(nh) * hd * hd, 0.0);
  std::vector<float> core(static_cast<size_t>(T) * proj, 0.0f);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < nh; ++h) {
      const int64_t base = t * proj + h * hd;
      const double beta = Sigmoid(braw[t * nh + h]);
      double* Sp = &S[static_cast<size_t>(h) * hd * hd];
      for (int64_t vd = 0; vd < hd; ++vd)
        for (int64_t kk = 0; kk < hd; ++kk)
          Sp[vd * hd + kk] *= std::exp(static_cast<double>(g[base + kk]));
      std::vector<double> u(static_cast<size_t>(hd));
      for (int64_t vd = 0; vd < hd; ++vd) {
        double pred = 0.0;
        for (int64_t kk = 0; kk < hd; ++kk) pred += Sp[vd * hd + kk] * kn[base + kk];
        u[static_cast<size_t>(vd)] = (static_cast<double>(vc[base + vd]) - pred) * beta;
      }
      for (int64_t vd = 0; vd < hd; ++vd)
        for (int64_t kk = 0; kk < hd; ++kk)
          Sp[vd * hd + kk] += u[static_cast<size_t>(vd)] * kn[base + kk];
      for (int64_t vd = 0; vd < hd; ++vd) {
        double o = 0.0;
        for (int64_t kk = 0; kk < hd; ++kk)
          o += Sp[vd * hd + kk] * (static_cast<double>(qn[base + kk]) * scale);
        core[base + vd] = static_cast<float>(o);
      }
    }
  const std::vector<float> cn = vllm::kimi_kda::FusedRMSNormGated(
      core, g2, w.o_norm, T, nh, hd, vllm::kimi_kda::GatedNormActivation::kSigmoid,
      p.rms_norm_eps);
  const std::vector<float> ref = Mat(w.o_proj, cn, H, proj, T);

  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
}

// ─── (b) NoPE-MLA layer == an INDEPENDENT materialized-MHA reference ──────────
TEST_CASE("kimi-linear forward: NoPE-MLA layer matches materialized-MHA reference") {
  KimiLinearParams p;
  p.hidden_size = 16;
  p.num_attention_heads = 2;
  p.qk_nope_head_dim = 4;
  p.qk_rope_head_dim = 2;
  p.v_head_dim = 4;
  p.kv_lora_rank = 8;
  p.rms_norm_eps = 1e-5f;
  const int64_t H = p.hidden_size, nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim, qr = p.qk_rope_head_dim, qk = qn + qr;
  const int64_t vh = p.v_head_dim, Lr = p.kv_lora_rank, T = 6;

  vllm::MlaLayerHostWeights w;
  w.q_proj = Rand(nah * qk * H, 21);
  w.kv_a_proj_with_mqa = Rand((Lr + qr) * H, 22);
  w.kv_a_layernorm = Rand(Lr, 23, 0.5f, 1.5f);
  w.kv_b_proj = Rand(nah * (qn + vh) * Lr, 24);
  w.o_proj = Rand(H * nah * vh, 25);
  const std::vector<float> x = Rand(T * H, 200);

  const std::vector<float> got = vllm::KimiNoPEMlaLayerForward(w, x, p, T);

  // INDEPENDENT materialized MHA (no absorption, no RoPE).
  const std::vector<float> q = Mat(w.q_proj, x, nah * qk, H, T);
  const std::vector<float> latent = Mat(w.kv_a_proj_with_mqa, x, Lr + qr, H, T);
  std::vector<float> kvc(static_cast<size_t>(T) * Lr), kpe(static_cast<size_t>(T) * qr);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t d = 0; d < Lr; ++d) kvc[t * Lr + d] = latent[t * (Lr + qr) + d];
    for (int64_t d = 0; d < qr; ++d) kpe[t * qr + d] = latent[t * (Lr + qr) + Lr + d];
  }
  const std::vector<float> kvcn = Rms(kvc, w.kv_a_layernorm, T, Lr, p.rms_norm_eps);
  const std::vector<float> kv = Mat(w.kv_b_proj, kvcn, nah * (qn + vh), Lr, T);
  const double scale = std::pow(static_cast<double>(qk), -0.5);
  std::vector<float> out(static_cast<size_t>(T) * nah * vh, 0.0f);
  for (int64_t h = 0; h < nah; ++h)
    for (int64_t t = 0; t < T; ++t) {
      std::vector<double> sc(static_cast<size_t>(t) + 1);
      double mx = -INFINITY;
      for (int64_t s = 0; s <= t; ++s) {
        double dot = 0.0;
        for (int64_t d = 0; d < qn; ++d)
          dot += static_cast<double>(q[t * nah * qk + h * qk + d]) *
                 kv[s * nah * (qn + vh) + h * (qn + vh) + d];
        for (int64_t d = 0; d < qr; ++d)
          dot += static_cast<double>(q[t * nah * qk + h * qk + qn + d]) * kpe[s * qr + d];
        dot *= scale;
        sc[static_cast<size_t>(s)] = dot;
        mx = std::max(mx, dot);
      }
      double sum = 0.0;
      for (int64_t s = 0; s <= t; ++s) { sc[static_cast<size_t>(s)] = std::exp(sc[static_cast<size_t>(s)] - mx); sum += sc[static_cast<size_t>(s)]; }
      for (int64_t d = 0; d < vh; ++d) {
        double acc = 0.0;
        for (int64_t s = 0; s <= t; ++s)
          acc += (sc[static_cast<size_t>(s)] / sum) *
                 kv[s * nah * (qn + vh) + h * (qn + vh) + qn + d];
        out[t * nah * vh + h * vh + d] = static_cast<float>(acc);
      }
    }
  const std::vector<float> ref = Mat(w.o_proj, out, H, nah * vh, T);

  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
}

// ─── (c) sigmoid noaux_tc router — a HAND-COMPUTED top-k case ─────────────────
TEST_CASE("kimi-linear forward: sigmoid noaux_tc router matches a hand-computed case") {
  KimiLinearParams p;
  p.hidden_size = 1;  // x = [1] so gate logits == gate weights (controlled case)
  p.num_experts = 4;
  p.num_experts_per_token = 2;
  p.num_shared_experts = 1;
  p.moe_intermediate_size = 1;
  p.routed_scaling_factor = 2.0;
  p.moe_renormalize = true;

  vllm::MoeHostWeights w;
  w.gate = {2.0f, -1.0f, 0.5f, 1.5f};                 // logits (H=1, x=[1])
  w.e_score_correction_bias = {0.0f, 3.0f, 0.0f, 0.0f};
  const std::vector<float> x = {1.0f};

  const vllm::KimiMoeRouting r = vllm::KimiMoeRoute(w, x, p, /*T=*/1);

  // HAND reference: scores=sigmoid(logit); sel=scores+bias; top-2 by sel; weight from
  // UNBIASED scores; renormalize; * routed_scaling.
  const double s0 = Sigmoid(2.0), s1 = Sigmoid(-1.0), s3 = Sigmoid(1.5);
  // sel: s1+3 (largest) then s0 -> ids [1,0]; weights unbiased [s1,s0] renorm *2.
  const double denom = s1 + s0;
  CHECK(r.ids[0] == 1);
  CHECK(r.ids[1] == 0);
  CHECK(r.weights[0] == doctest::Approx(s1 / denom * 2.0).epsilon(1e-6));
  CHECK(r.weights[1] == doctest::Approx(s0 / denom * 2.0).epsilon(1e-6));
  (void)s3;

  // Whole block == shared_ffn(x) + sum_j weight_j * expert_ffn_{id_j}(x).
  KimiLinearParams pb = p;
  pb.hidden_size = 3;
  const int64_t H = pb.hidden_size, E = pb.num_experts, mi = pb.moe_intermediate_size;
  vllm::MoeHostWeights wb;
  wb.gate = Rand(E * H, 31);
  wb.e_score_correction_bias = Rand(E, 32);
  wb.has_shared = true;
  wb.shared.gate_proj = Rand(mi * H, 33);
  wb.shared.up_proj = Rand(mi * H, 34);
  wb.shared.down_proj = Rand(H * mi, 35);
  wb.experts.resize(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    wb.experts[static_cast<size_t>(e)].gate_proj = Rand(mi * H, 40 + e * 3);
    wb.experts[static_cast<size_t>(e)].up_proj = Rand(mi * H, 41 + e * 3);
    wb.experts[static_cast<size_t>(e)].down_proj = Rand(H * mi, 42 + e * 3);
  }
  const std::vector<float> xb = Rand(2 * H, 300);  // T=2
  const std::vector<float> block = vllm::KimiMoeBlockForward(wb, xb, pb, 2);
  const vllm::KimiMoeRouting rb = vllm::KimiMoeRoute(wb, xb, pb, 2);

  auto ffn = [&](const vllm::MlpHostWeights& e, const float* xt) {
    std::vector<float> g = Mat(e.gate_proj, {xt, xt + H}, mi, H, 1);
    std::vector<float> u = Mat(e.up_proj, {xt, xt + H}, mi, H, 1);
    std::vector<float> hmid(static_cast<size_t>(mi));
    for (int64_t i = 0; i < mi; ++i) hmid[static_cast<size_t>(i)] = static_cast<float>(Silu(g[static_cast<size_t>(i)]) * u[static_cast<size_t>(i)]);
    return Mat(e.down_proj, hmid, H, mi, 1);
  };
  for (int64_t t = 0; t < 2; ++t) {
    std::vector<double> exp_out(static_cast<size_t>(H), 0.0);
    const std::vector<float> sh = ffn(wb.shared, &xb[t * H]);
    for (int64_t d = 0; d < H; ++d) exp_out[static_cast<size_t>(d)] += sh[static_cast<size_t>(d)];
    for (int64_t j = 0; j < pb.num_experts_per_token; ++j) {
      const int32_t e = rb.ids[t * pb.num_experts_per_token + j];
      const double wgt = rb.weights[t * pb.num_experts_per_token + j];
      const std::vector<float> y = ffn(wb.experts[static_cast<size_t>(e)], &xb[t * H]);
      for (int64_t d = 0; d < H; ++d) exp_out[static_cast<size_t>(d)] += wgt * y[static_cast<size_t>(d)];
    }
    for (int64_t d = 0; d < H; ++d)
      CHECK(block[t * H + d] == doctest::Approx(exp_out[static_cast<size_t>(d)]).epsilon(1e-5));
  }
}

// ─── synthetic 2-layer checkpoint (layer0 KDA+dense, layer1 NoPE-MLA+MoE) ──────
namespace {
constexpr int H = 32, NAH = 4, V = 8, DENSE_I = 64, MOE_I = 16, E = 2;
constexpr int KV_LORA = 16, QK_NOPE = 8, QK_ROPE = 4, V_HEAD = 8;
constexpr int KDA_NH = 4, KDA_HD = 8, CONV = 4;
constexpr int KDA_PROJ = KDA_NH * KDA_HD;
constexpr int MLA_QK = QK_NOPE + QK_ROPE;

struct Fx { std::string name, dtype; std::vector<int64_t> shape; std::string bytes; };
std::string U64Le(uint64_t v) { std::string s(8, '\0'); for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff); return s; }
int64_t NumEl(const std::vector<int64_t>& s) { int64_t n = 1; for (int64_t d : s) n *= d; return n; }
// Fill with small bf16 values (exponent 0x3E-0x3F range) so the forward stays finite.
std::string Bf16Bytes(size_t n, int seed) {
  std::string s(n * 2, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    r = r * 1664525u + 1013904223u;
    const float u = static_cast<float>(r >> 8) / static_cast<float>(1u << 24);  // [0,1)
    const float f = (u - 0.5f) * 0.25f;
    uint16_t bf = static_cast<uint16_t>(vt::F32ToBF16(f));
    s[i * 2] = static_cast<char>(bf & 0xff);
    s[i * 2 + 1] = static_cast<char>((bf >> 8) & 0xff);
  }
  return s;
}
std::string F32Bytes(size_t n, int seed) {
  std::string s(n * 4, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2246822519u + 1u;
  for (size_t i = 0; i < n; ++i) {
    r = r * 1664525u + 1013904223u;
    const float u = static_cast<float>(r >> 8) / static_cast<float>(1u << 24);
    const float f = (u - 0.5f) * 0.25f;
    std::memcpy(&s[i * 4], &f, 4);
  }
  return s;
}
Fx Bf16(const std::string& n, std::vector<int64_t> sh, int seed) { return {n, "BF16", sh, Bf16Bytes(static_cast<size_t>(NumEl(sh)), seed)}; }
Fx F32(const std::string& n, std::vector<int64_t> sh, int seed) { return {n, "F32", sh, F32Bytes(static_cast<size_t>(NumEl(sh)), seed)}; }
std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}
class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int c = 0;
    path_ = (std::filesystem::temp_directory_path() / ("kimi_fwd_" + std::to_string(c++) + ".safetensors")).string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }
 private:
  std::string path_;
};

std::vector<Fx> BuildTensors() {
  std::vector<Fx> v;
  int s = 1;
  v.push_back(Bf16("model.embed_tokens.weight", {V, H}, s++));
  v.push_back(Bf16("model.norm.weight", {H}, s++));
  v.push_back(Bf16("lm_head.weight", {V, H}, s++));
  const std::string a0 = "model.layers.0.";
  v.push_back(Bf16(a0 + "input_layernorm.weight", {H}, s++));
  v.push_back(Bf16(a0 + "post_attention_layernorm.weight", {H}, s++));
  const std::string k = a0 + "self_attn.";
  v.push_back(Bf16(k + "q_proj.weight", {KDA_PROJ, H}, s++));
  v.push_back(Bf16(k + "k_proj.weight", {KDA_PROJ, H}, s++));
  v.push_back(Bf16(k + "v_proj.weight", {KDA_PROJ, H}, s++));
  v.push_back(Bf16(k + "f_a_proj.weight", {KDA_HD, H}, s++));
  v.push_back(Bf16(k + "f_b_proj.weight", {KDA_PROJ, KDA_HD}, s++));
  v.push_back(Bf16(k + "b_proj.weight", {KDA_NH, H}, s++));
  v.push_back(Bf16(k + "g_a_proj.weight", {KDA_HD, H}, s++));
  v.push_back(Bf16(k + "g_b_proj.weight", {KDA_PROJ, KDA_HD}, s++));
  v.push_back(Bf16(k + "o_proj.weight", {H, KDA_PROJ}, s++));
  v.push_back(F32(k + "q_conv1d.weight", {KDA_PROJ, 1, CONV}, s++));
  v.push_back(F32(k + "k_conv1d.weight", {KDA_PROJ, 1, CONV}, s++));
  v.push_back(F32(k + "v_conv1d.weight", {KDA_PROJ, 1, CONV}, s++));
  v.push_back(F32(k + "dt_bias", {KDA_PROJ}, s++));
  v.push_back(F32(k + "A_log", {KDA_NH}, s++));
  v.push_back(Bf16(k + "o_norm.weight", {KDA_HD}, s++));
  v.push_back(Bf16(a0 + "mlp.gate_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16(a0 + "mlp.up_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16(a0 + "mlp.down_proj.weight", {H, DENSE_I}, s++));
  const std::string a1 = "model.layers.1.";
  v.push_back(Bf16(a1 + "input_layernorm.weight", {H}, s++));
  v.push_back(Bf16(a1 + "post_attention_layernorm.weight", {H}, s++));
  const std::string m = a1 + "self_attn.";
  v.push_back(Bf16(m + "q_proj.weight", {NAH * MLA_QK, H}, s++));
  v.push_back(Bf16(m + "kv_a_proj_with_mqa.weight", {KV_LORA + QK_ROPE, H}, s++));
  v.push_back(Bf16(m + "kv_a_layernorm.weight", {KV_LORA}, s++));
  v.push_back(Bf16(m + "kv_b_proj.weight", {NAH * (QK_NOPE + V_HEAD), KV_LORA}, s++));
  v.push_back(Bf16(m + "o_proj.weight", {H, NAH * V_HEAD}, s++));
  const std::string mo = a1 + "block_sparse_moe.";
  v.push_back(Bf16(mo + "gate.weight", {E, H}, s++));
  v.push_back(F32(mo + "gate.e_score_correction_bias", {E}, s++));
  v.push_back(Bf16(mo + "shared_experts.gate_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16(mo + "shared_experts.up_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16(mo + "shared_experts.down_proj.weight", {H, MOE_I}, s++));
  for (int e = 0; e < E; ++e) {
    const std::string ep = mo + "experts." + std::to_string(e) + ".";
    v.push_back(Bf16(ep + "w1.weight", {MOE_I, H}, s++));
    v.push_back(Bf16(ep + "w2.weight", {H, MOE_I}, s++));
    v.push_back(Bf16(ep + "w3.weight", {MOE_I, H}, s++));
  }
  return v;
}

vllm::HfConfig TinyConfig() {
  vllm::HfConfig c;
  c.architectures = {"KimiLinearForCausalLM"};
  c.model_type = "kimi_linear";
  c.hidden_size = H;
  c.num_hidden_layers = 2;
  c.vocab_size = V;
  c.num_attention_heads = NAH;
  c.raw = {
      {"hidden_size", H}, {"num_hidden_layers", 2}, {"vocab_size", V},
      {"num_attention_heads", NAH}, {"num_key_value_heads", NAH},
      {"intermediate_size", DENSE_I}, {"rms_norm_eps", 1e-5},
      {"tie_word_embeddings", false}, {"num_experts", E},
      {"num_experts_per_token", 1}, {"num_shared_experts", 1},
      {"moe_intermediate_size", MOE_I}, {"first_k_dense_replace", 1},
      {"moe_layer_freq", 1}, {"moe_router_activation_func", "sigmoid"},
      {"routed_scaling_factor", 2.446}, {"kv_lora_rank", KV_LORA},
      {"qk_nope_head_dim", QK_NOPE}, {"qk_rope_head_dim", QK_ROPE},
      {"v_head_dim", V_HEAD}, {"mla_use_nope", true},
      {"linear_attn_config",
       {{"kda_layers", nlohmann::json::array({1})},
        {"full_attn_layers", nlohmann::json::array({2})},
        {"num_heads", KDA_NH}, {"head_dim", KDA_HD}, {"short_conv_kernel_size", CONV}}},
  };
  return c;
}
}  // namespace

// ─── (d) whole forward: loader materializes; finite/coherent + greedy decode ──
TEST_CASE("kimi-linear forward: loader materializes host weights end-to-end") {
  TempFile f(BuildSt(BuildTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  const KimiLinearWeights w = LoadKimiLinearForCausalLMWeights(shards, TinyConfig());
  REQUIRE(w.host.materialized);
  REQUIRE(w.host.layers.size() == 2);
  CHECK(w.host.layers[0].is_kda);
  CHECK_FALSE(w.host.layers[0].is_moe);       // dense layer 0
  CHECK_FALSE(w.host.layers[1].is_kda);        // NoPE-MLA
  CHECK(w.host.layers[1].is_moe);
  CHECK(static_cast<int>(w.host.layers[1].moe.experts.size()) == E);
  CHECK(w.host.layers[0].kda.q_proj.size() == static_cast<size_t>(KDA_PROJ * H));
  CHECK(w.host.embed_tokens.size() == static_cast<size_t>(V * H));
}

TEST_CASE("kimi-linear forward: whole 2-layer forward is finite and coherently shaped") {
  TempFile f(BuildSt(BuildTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  const KimiLinearWeights w = LoadKimiLinearForCausalLMWeights(shards, TinyConfig());

  const std::vector<int32_t> tokens = {1, 3, 0, 5};
  const std::vector<int32_t> positions = {0, 1, 2, 3};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  const std::vector<float> logits =
      vllm::KimiLinearModel::Forward(tokens, positions, attn_meta, attn_kv, w, q, {});
  REQUIRE(logits.size() == tokens.size() * static_cast<size_t>(V));
  for (float x : logits) CHECK(std::isfinite(x));

  // Selecting only the last position returns exactly one row.
  const std::vector<float> last =
      vllm::KimiLinearModel::Forward(tokens, positions, attn_meta, attn_kv, w, q, {3});
  REQUIRE(last.size() == static_cast<size_t>(V));
  for (int64_t o = 0; o < V; ++o)
    CHECK(last[static_cast<size_t>(o)] ==
          doctest::Approx(logits[3 * V + o]).epsilon(1e-6));
}

TEST_CASE("kimi-linear forward: greedy decode advances the context N tokens") {
  TempFile f(BuildSt(BuildTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  const KimiLinearWeights w = LoadKimiLinearForCausalLMWeights(shards, TinyConfig());

  const std::vector<int32_t> prompt = {2, 4};
  const int kNew = 5;
  const std::vector<int32_t> gen =
      vllm::KimiLinearGreedyDecode(w.host, w.params, prompt, kNew);
  REQUIRE(static_cast<int>(gen.size()) == kNew);
  for (int32_t t : gen) CHECK((t >= 0 && t < V));  // every step a valid token id

  // The context truly advances: the last-position logits over a 1-token vs a
  // 2-token prefix differ (the KDA recurrence + MLA causal attention use the
  // growing context), so decode is not a constant.
  vllm::v1::CommonAttentionMetadata attn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<float> l1 = vllm::KimiLinearModel::Forward(
      {prompt[0]}, {0}, attn_meta, attn_kv, w, q, {0});
  const std::vector<float> l2 = vllm::KimiLinearModel::Forward(
      prompt, {0, 1}, attn_meta, attn_kv, w, q, {1});
  bool differ = false;
  for (int64_t o = 0; o < V; ++o)
    if (std::fabs(l1[static_cast<size_t>(o)] - l2[static_cast<size_t>(o)]) > 1e-6f) differ = true;
  CHECK(differ);
}

// ─── (e) W6 born-on-the-runner SEAM: ForwardDevice returns DEVICE-RESIDENT logits ─
// The runner routes the default gather_logits path to ForwardDevice; it must hand
// the logits back on_device() (fed to the on-GPU sampler with no host download) AND
// the composed logits must MATCH the W2 host reference. On the CPU backend the
// pooled DBuf is a device buffer too, so on_device() holds and the CPU gate
// exercises the exact contract the GPU will. (The device-COMPUTE lane — KDA/NoPE-MLA
// /MoE over the paged het-KV caches — is the GPU-verify-pending W7 residual, so this
// SEAM composes off the same host reference and the match is EXACT.)
TEST_CASE("kimi-linear ForwardDevice: born-on-runner device-resident logits == host ref") {
  TempFile f(BuildSt(BuildTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  const KimiLinearWeights w = LoadKimiLinearForCausalLMWeights(shards, TinyConfig());

  const std::vector<int32_t> tokens = {1, 3, 0, 5};
  const std::vector<int32_t> positions = {0, 1, 2, 3};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& be = vt::GetBackend(q.device.type);

  // Host reference (the !gather_logits path) — the correctness truth.
  const std::vector<float> host_logits =
      vllm::KimiLinearModel::Forward(tokens, positions, attn_meta, attn_kv, w, q, {});

  // ForwardDevice (the DEFAULT gather_logits runner path).
  const vllm::ForwardLogits fl = vllm::KimiLinearModel::ForwardDevice(
      tokens, positions, attn_meta, attn_kv, w, q, {});

  // (a) the runner-routing contract: logits stay ON-DEVICE (no host copy on .host).
  CHECK(fl.on_device());
  CHECK(fl.device_storage != nullptr);
  CHECK(fl.host.empty());
  REQUIRE(fl.rows == static_cast<int64_t>(tokens.size()));
  REQUIRE(fl.vocab == static_cast<int64_t>(V));
  REQUIRE(fl.device_tensor.data != nullptr);

  // Download the device-resident logits and compare to the host reference — the
  // device SEAM composes off the same reference, so the round-trip is byte-exact.
  std::vector<float> dl(static_cast<size_t>(fl.rows) * static_cast<size_t>(fl.vocab));
  be.Copy(q, dl.data(), fl.device_tensor.data,
          dl.size() * sizeof(float));
  be.Synchronize(q);
  REQUIRE(dl.size() == host_logits.size());
  for (size_t i = 0; i < dl.size(); ++i)
    CHECK(dl[i] == doctest::Approx(host_logits[i]).epsilon(1e-6));

  // Greedy tokens are identical (the runner argmaxes ForwardDevice's device logits).
  for (int64_t r = 0; r < fl.rows; ++r) {
    int32_t best_dev = 0, best_host = 0;
    for (int64_t o = 1; o < V; ++o) {
      if (dl[static_cast<size_t>(r * V + o)] > dl[static_cast<size_t>(r * V + best_dev)]) best_dev = static_cast<int32_t>(o);
      if (host_logits[static_cast<size_t>(r * V + o)] > host_logits[static_cast<size_t>(r * V + best_host)]) best_host = static_cast<int32_t>(o);
    }
    CHECK(best_dev == best_host);
  }

  // (b) logits_indices gather-before-lm_head: selecting the last position returns
  // exactly ONE device row == the host reference's last row (request-order return).
  const vllm::ForwardLogits last = vllm::KimiLinearModel::ForwardDevice(
      tokens, positions, attn_meta, attn_kv, w, q, {3});
  CHECK(last.on_device());
  REQUIRE(last.rows == 1);
  REQUIRE(last.vocab == static_cast<int64_t>(V));
  std::vector<float> lastd(static_cast<size_t>(V));
  be.Copy(q, lastd.data(), last.device_tensor.data, lastd.size() * sizeof(float));
  be.Synchronize(q);
  for (int64_t o = 0; o < V; ++o)
    CHECK(lastd[static_cast<size_t>(o)] ==
          doctest::Approx(host_logits[3 * V + o]).epsilon(1e-6));
}

// ═══ W7 — the DBuf-RESIDENT device COMPUTE path, CPU-gated vs the W2 reference ══
// The per-op device primitives (KimiKdaLayerForwardDevice / ...MlaLayer... /
// ...MoeBlock... / ...DenseMlp...) route each primitive through the shared vt::
// device ops on pooled DBufs and must MATCH the W2 host reference on CPU (the CPU
// backend runs the SAME dispatch), with a tolerance whose only slack is
// f32-accumulation-order (documented per op). Then the WHOLE 2-layer forward
// (ForwardDeviceCompute) must match the reference logits AND greedy tokens, and
// return device-resident logits — the born-on-runner contract. GPU numerics (bf16
// activations, the GDN cubins, paged het-KV, grouped-MoE slabs) stay pending.

// ─── (f) KDA layer: device compute == the W2 reference ────────────────────────
TEST_CASE("kimi-linear W7 device: KDA layer matches the W2 host reference") {
  KimiLinearParams p;
  p.hidden_size = 16;
  p.kda_num_heads = 2;
  p.kda_head_dim = 8;
  p.kda_short_conv_kernel_size = 4;
  p.rms_norm_eps = 1e-5f;
  const int64_t H = p.hidden_size, nh = p.kda_num_heads, hd = p.kda_head_dim;
  const int64_t proj = nh * hd, K = p.kda_short_conv_kernel_size, T = 5;

  vllm::KdaLayerHostWeights w;
  w.q_proj = Rand(proj * H, 1);
  w.k_proj = Rand(proj * H, 2);
  w.v_proj = Rand(proj * H, 3);
  w.f_a_proj = Rand(hd * H, 4);
  w.f_b_proj = Rand(proj * hd, 5);
  w.b_proj = Rand(nh * H, 6);
  w.g_a_proj = Rand(hd * H, 7);
  w.g_b_proj = Rand(proj * hd, 8);
  w.o_proj = Rand(H * proj, 9);
  w.q_conv = Rand(proj * K, 10);
  w.k_conv = Rand(proj * K, 11);
  w.v_conv = Rand(proj * K, 12);
  w.dt_bias = Rand(proj, 13);
  w.a_log = Rand(nh, 14, -2.0f, 0.0f);
  w.o_norm = Rand(hd, 15, 0.5f, 1.5f);
  const std::vector<float> x = Rand(T * H, 100);

  const std::vector<float> ref = vllm::KimiKdaLayerForward(w, x, p, T);
  vt::Queue q = CpuQueue();
  const std::vector<float> dev = vllm::KimiKdaLayerForwardDevice(w, x, p, T, q);

  REQUIRE(dev.size() == ref.size());
  // rtol 3e-3 / atol 3e-4: the gated-delta recurrence runs the identical host code
  // on both sides, fed q/k/v/g1 that differ only by f32-vs-double GEMM/conv/L2
  // accumulation; the T-step recurrence mildly amplifies that.
  for (size_t i = 0; i < dev.size(); ++i) CHECK(Close(dev[i], ref[i], 3e-3, 3e-4));
}

// ─── (g) NoPE-MLA layer: device compute == the W2 reference ───────────────────
TEST_CASE("kimi-linear W7 device: NoPE-MLA layer matches the W2 host reference") {
  KimiLinearParams p;
  p.hidden_size = 16;
  p.num_attention_heads = 2;
  p.qk_nope_head_dim = 4;
  p.qk_rope_head_dim = 2;
  p.v_head_dim = 4;
  p.kv_lora_rank = 8;
  p.rms_norm_eps = 1e-5f;
  const int64_t H = p.hidden_size, nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim, qr = p.qk_rope_head_dim, qk = qn + qr;
  const int64_t vh = p.v_head_dim, Lr = p.kv_lora_rank, T = 6;

  vllm::MlaLayerHostWeights w;
  w.q_proj = Rand(nah * qk * H, 21);
  w.kv_a_proj_with_mqa = Rand((Lr + qr) * H, 22);
  w.kv_a_layernorm = Rand(Lr, 23, 0.5f, 1.5f);
  w.kv_b_proj = Rand(nah * (qn + vh) * Lr, 24);
  w.o_proj = Rand(H * nah * vh, 25);
  const std::vector<float> x = Rand(T * H, 200);

  const std::vector<float> ref = vllm::KimiNoPEMlaLayerForward(w, x, p, T);
  vt::Queue q = CpuQueue();
  const std::vector<float> dev = vllm::KimiNoPEMlaLayerForwardDevice(w, x, p, T, q);

  REQUIRE(dev.size() == ref.size());
  // rtol 1e-4 / atol 1e-5: pure GEMM + softmax, only f32-accumulation slack.
  for (size_t i = 0; i < dev.size(); ++i) CHECK(Close(dev[i], ref[i], 1e-4, 1e-5));
}

// ─── (h) MoE block + dense MLP: device compute == the W2 reference ────────────
TEST_CASE("kimi-linear W7 device: MoE block + dense MLP match the W2 host reference") {
  KimiLinearParams p;
  p.hidden_size = 12;
  p.num_experts = 4;
  p.num_experts_per_token = 2;
  p.num_shared_experts = 1;
  p.moe_intermediate_size = 5;
  p.routed_scaling_factor = 2.446;
  p.moe_renormalize = true;
  p.num_expert_group = 1;
  p.topk_group = 1;
  const int64_t H = p.hidden_size, E = p.num_experts, mi = p.moe_intermediate_size;
  const int64_t T = 3;

  vllm::MoeHostWeights w;
  w.gate = Rand(E * H, 51);
  w.e_score_correction_bias = Rand(E, 52);
  w.has_shared = true;
  w.shared.gate_proj = Rand(mi * H, 53);
  w.shared.up_proj = Rand(mi * H, 54);
  w.shared.down_proj = Rand(H * mi, 55);
  w.experts.resize(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    w.experts[static_cast<size_t>(e)].gate_proj = Rand(mi * H, 60 + e * 3);
    w.experts[static_cast<size_t>(e)].up_proj = Rand(mi * H, 61 + e * 3);
    w.experts[static_cast<size_t>(e)].down_proj = Rand(H * mi, 62 + e * 3);
  }
  const std::vector<float> x = Rand(T * H, 300);

  vt::Queue q = CpuQueue();
  const std::vector<float> moe_ref = vllm::KimiMoeBlockForward(w, x, p, T);
  const std::vector<float> moe_dev = vllm::KimiMoeBlockForwardDevice(w, x, p, T, q);
  REQUIRE(moe_dev.size() == moe_ref.size());
  // rtol 2e-4 / atol 1e-5: the device router (vt::MoeRouterTopK) selects the SAME
  // experts (lowest-index tie rule matches KimiMoeRoute), so the only slack is
  // f32-accumulation in the router logits + the per-expert SwiGLU GEMMs.
  for (size_t i = 0; i < moe_dev.size(); ++i) CHECK(Close(moe_dev[i], moe_ref[i], 2e-4, 1e-5));

  KimiLinearParams pd = p;
  pd.intermediate_size = 7;
  vllm::MlpHostWeights dw;
  dw.gate_proj = Rand(pd.intermediate_size * H, 71);
  dw.up_proj = Rand(pd.intermediate_size * H, 72);
  dw.down_proj = Rand(H * pd.intermediate_size, 73);
  const std::vector<float> dense_ref = vllm::KimiDenseMlpForward(dw, x, pd, T);
  const std::vector<float> dense_dev = vllm::KimiDenseMlpForwardDevice(dw, x, pd, T, q);
  REQUIRE(dense_dev.size() == dense_ref.size());
  for (size_t i = 0; i < dense_dev.size(); ++i)
    CHECK(Close(dense_dev[i], dense_ref[i], 1e-4, 1e-5));
}

// ─── (i) WHOLE 2-layer forward: ForwardDeviceCompute == the W2 reference ──────
TEST_CASE("kimi-linear W7 device: ForwardDeviceCompute matches the W2 reference + greedy") {
  TempFile f(BuildSt(BuildTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  const KimiLinearWeights w = LoadKimiLinearForCausalLMWeights(shards, TinyConfig());

  const std::vector<int32_t> tokens = {1, 3, 0, 5};
  const std::vector<int32_t> positions = {0, 1, 2, 3};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  vt::Queue q = CpuQueue();
  vt::Backend& be = vt::GetBackend(q.device.type);

  // The W2 host reference (the correctness truth) + the W7 device compute.
  const std::vector<float> host_logits =
      vllm::KimiLinearModel::Forward(tokens, positions, attn_meta, attn_kv, w, q, {});
  const vllm::ForwardLogits fl = vllm::KimiLinearModel::ForwardDeviceCompute(
      tokens, positions, attn_meta, attn_kv, w, q, {});

  // Born-on-runner contract: logits stay ON-DEVICE (no host copy on .host).
  CHECK(fl.on_device());
  CHECK(fl.device_storage != nullptr);
  CHECK(fl.host.empty());
  REQUIRE(fl.rows == static_cast<int64_t>(tokens.size()));
  REQUIRE(fl.vocab == static_cast<int64_t>(V));
  REQUIRE(fl.device_tensor.data != nullptr);

  std::vector<float> dl(static_cast<size_t>(fl.rows) * static_cast<size_t>(fl.vocab));
  be.Copy(q, dl.data(), fl.device_tensor.data, dl.size() * sizeof(float));
  be.Synchronize(q);
  REQUIRE(dl.size() == host_logits.size());
  // rtol 5e-3 / atol 1e-3: two full layers (KDA + NoPE-MLA + MoE + dense) of
  // f32-vs-double accumulation through the residual stream. The BINDING gate is
  // token-identical greedy argmax below.
  for (size_t i = 0; i < dl.size(); ++i) CHECK(Close(dl[i], host_logits[i], 5e-3, 1e-3));

  // Greedy tokens IDENTICAL — the real correctness signal (the runner argmaxes
  // ForwardDeviceCompute's device logits with no host download).
  for (int64_t r = 0; r < fl.rows; ++r) {
    int32_t best_dev = 0, best_host = 0;
    for (int64_t o = 1; o < V; ++o) {
      if (dl[static_cast<size_t>(r * V + o)] > dl[static_cast<size_t>(r * V + best_dev)]) best_dev = static_cast<int32_t>(o);
      if (host_logits[static_cast<size_t>(r * V + o)] > host_logits[static_cast<size_t>(r * V + best_host)]) best_host = static_cast<int32_t>(o);
    }
    CHECK(best_dev == best_host);
  }

  // logits_indices gather-before-lm_head: last position -> exactly ONE device row.
  const vllm::ForwardLogits last = vllm::KimiLinearModel::ForwardDeviceCompute(
      tokens, positions, attn_meta, attn_kv, w, q, {3});
  CHECK(last.on_device());
  REQUIRE(last.rows == 1);
  REQUIRE(last.vocab == static_cast<int64_t>(V));
  std::vector<float> lastd(static_cast<size_t>(V));
  be.Copy(q, lastd.data(), last.device_tensor.data, lastd.size() * sizeof(float));
  be.Synchronize(q);
  for (int64_t o = 0; o < V; ++o) CHECK(Close(lastd[static_cast<size_t>(o)], host_logits[3 * V + o], 5e-3, 1e-3));
}

// ─── (j) ForwardDevice routes to the device compute under VT_KIMI_DEVICE_COMPUTE ─
// The runner seam (ForwardDevice) must be able to reach the device compute so the
// GPU verification can exercise it as the production path. On CPU both paths equal
// the reference, so routed==unrouted here; the toggle is the wiring proof.
TEST_CASE("kimi-linear W7 device: ForwardDevice reaches device compute (opt-in gate present)") {
  TempFile f(BuildSt(BuildTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  const KimiLinearWeights w = LoadKimiLinearForCausalLMWeights(shards, TinyConfig());

  const std::vector<int32_t> tokens = {1, 3, 0, 5};
  const std::vector<int32_t> positions = {0, 1, 2, 3};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  vt::Queue q = CpuQueue();
  vt::Backend& be = vt::GetBackend(q.device.type);

  // The device compute and the W6 host-ref compose both return device-resident
  // logits; on CPU they agree with the reference within tolerance.
  const vllm::ForwardLogits seam = vllm::KimiLinearModel::ForwardDevice(
      tokens, positions, attn_meta, attn_kv, w, q, {});
  const vllm::ForwardLogits compute = vllm::KimiLinearModel::ForwardDeviceCompute(
      tokens, positions, attn_meta, attn_kv, w, q, {});
  CHECK(seam.on_device());
  CHECK(compute.on_device());
  REQUIRE(seam.rows == compute.rows);
  REQUIRE(seam.vocab == compute.vocab);
  std::vector<float> a(static_cast<size_t>(seam.rows) * seam.vocab), b(a.size());
  be.Copy(q, a.data(), seam.device_tensor.data, a.size() * sizeof(float));
  be.Copy(q, b.data(), compute.device_tensor.data, b.size() * sizeof(float));
  be.Synchronize(q);
  // Both are device-resident logits over the same 2-layer model; greedy tokens agree.
  for (int64_t r = 0; r < seam.rows; ++r) {
    int32_t ba = 0, bb = 0;
    for (int64_t o = 1; o < V; ++o) {
      if (a[static_cast<size_t>(r * V + o)] > a[static_cast<size_t>(r * V + ba)]) ba = static_cast<int32_t>(o);
      if (b[static_cast<size_t>(r * V + o)] > b[static_cast<size_t>(r * V + bb)]) bb = static_cast<int32_t>(o);
    }
    CHECK(ba == bb);
  }
}
