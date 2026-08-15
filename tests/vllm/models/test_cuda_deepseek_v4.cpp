// DeepSeek-V4-Flash W7-device — per-kernel CUDA gate + the ForwardDevice
// composition gate. Each of the four NEW V4 op families' CUDA kernels
// (src/vt/cuda/cuda_deepseek_v4.cu, dispatched through the vt OpProvider seam) is
// run at a SMALL synthetic shape and compared against its LANDED portable HOST
// reference (the oracle): BIT-EXACT where the math is integer/exact (top-k
// selection, hash route ids), NMSE / near-tie for fp reductions (Sinkhorn,
// softmax pool, sqrtsoftplus — device expf/sqrtf differ from host by ULPs). The
// device cases SKIP on a CPU-only build (no CUDA backend registered).
//
// HONEST SCOPE (mirrors W3-W7): this is the runtime evidence the kernels match
// the references at small shape on a real GPU — NOT a real-checkpoint token gate
// (the fixed-config 167B does not fit ONE GB10; that is the W8 residual). See
// .agents/specs/deepseek-v4-flash.md §W7.
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_compressor.h"
#include "vllm/model_executor/models/deepseek_v4_device.h"
#include "vllm/model_executor/models/deepseek_v4_dsa.h"
#include "vllm/model_executor/models/deepseek_v4_mhc.h"
#include "vllm/model_executor/models/deepseek_v4_moe.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"  // vt::F32ToBF16 / vt::BF16ToF32 (router_gate bf16 weights)

namespace dv4 = vllm::deepseek_v4;
using vllm::DeepseekV4HostWeights;
using vllm::DeepseekV4LayerHostWeights;
using vllm::DeepseekV4Params;

namespace {

bool HasCuda() {
  try {
    vt::GetBackend(vt::DeviceType::kCUDA);
    return vllm::deepseek_v4::V4DeviceKernelsAvailable();
  } catch (const std::runtime_error&) {
    return false;
  }
}

struct QueueGuard {
  vt::Backend& b;
  vt::Queue q;
  explicit QueueGuard(vt::Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

struct Rng {
  uint32_t s = 0x9E3779B9u;
  float next(float lo, float hi) {
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s >> 8) / 16777216.0f;
    return lo + u * (hi - lo);
  }
};
std::vector<float> Rand(Rng& r, int64_t n, float lo = -1.0f, float hi = 1.0f) {
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& e : v) e = r.next(lo, hi);
  return v;
}

// Relative L2 over two equal-length buffers.
double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(a[i]) * a[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}
float MaxAbs(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

constexpr double kTol = 1e-4;  // near-tie for fp reductions (device vs host transcendentals)

}  // namespace

// ===========================================================================
// (1) MHC family
// ===========================================================================
TEST_CASE("W7-device MHC Sinkhorn: CUDA vs host reference (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t hc = 4, iters = 5;
  const float eps = 1e-6f;
  Rng r;
  const auto logits = Rand(r, hc * hc, -2.0f, 2.0f);
  const auto ref = dv4::MhcSinkhorn(logits, hc, iters, eps);
  const auto got = dv4::MhcDevice()->sinkhorn(g.q, logits, hc, iters, eps);
  REQUIRE(got.size() == ref.size());
  CHECK(RelL2(got, ref) < kTol);
  // Doubly-stochastic property (col sums == 1 at eps~0) holds on the device too.
  for (int64_t k = 0; k < hc; ++k) {
    float c = 0.0f;
    for (int64_t j = 0; j < hc; ++j) c += got[j * hc + k];
    CHECK(c == doctest::Approx(1.0f).epsilon(1e-3));
  }
}

TEST_CASE("W7-device MhcPre: CUDA vs host reference (all four outputs near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t hc = 4, hidden = 8, iters = 5;
  const int64_t hc3 = (2 + hc) * hc, hcH = hc * hidden;
  Rng r;
  const auto residual = Rand(r, hc * hidden, -1.0f, 1.0f);
  const auto fn = Rand(r, hc3 * hcH, -0.3f, 0.3f);
  const auto scale = Rand(r, 3, -0.5f, 0.5f);
  const auto base = Rand(r, hc3, -0.3f, 0.3f);
  const auto nw = Rand(r, hidden, 0.9f, 1.1f);
  const float eps = 1e-6f;
  const auto ref = dv4::MhcPre(residual, fn, scale, base, hc, hidden, eps, eps, eps, 2.0f,
                               iters, nw, eps);
  const auto got = dv4::MhcDevice()->pre(g.q, residual, fn, scale, base, hc, hidden, eps, eps,
                                         eps, 2.0f, iters, nw, eps);
  CHECK(RelL2(got.pre_mix, ref.pre_mix) < kTol);
  CHECK(RelL2(got.post_mix, ref.post_mix) < kTol);
  CHECK(RelL2(got.comb_mix, ref.comb_mix) < kTol);
  CHECK(RelL2(got.layer_input, ref.layer_input) < kTol);
}

TEST_CASE("W7-device MhcPost + HcHeadCollapse: CUDA vs host reference (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t hc = 4, hidden = 8;
  Rng r;
  const auto x = Rand(r, hidden), residual = Rand(r, hc * hidden);
  const auto post_mix = Rand(r, hc, 0.0f, 2.0f), comb = Rand(r, hc * hc, 0.0f, 1.0f);
  const auto post_ref = dv4::MhcPost(x, residual, post_mix, comb, hc, hidden);
  const auto post_got = dv4::MhcDevice()->post(g.q, x, residual, post_mix, comb, hc, hidden);
  CHECK(RelL2(post_got, post_ref) < kTol);

  const auto xh = Rand(r, hc * hidden), fn = Rand(r, hc * hc * hidden, -0.3f, 0.3f);
  const auto base = Rand(r, hc, -0.3f, 0.3f);
  const auto head_ref = dv4::HcHeadCollapse(xh, fn, 0.5f, base, hc, hidden, 1e-6f, 1e-6f);
  const auto head_got = dv4::MhcDevice()->head(g.q, xh, fn, 0.5f, base, hc, hidden, 1e-6f, 1e-6f);
  CHECK(RelL2(head_got, head_ref) < kTol);
}

// ===========================================================================
// (2) DSA family
// ===========================================================================
TEST_CASE("W7-device DSA weight-fold + MQA logits: CUDA vs host (near-tie, -inf exact)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t T = 3, inh = 2, ihd = 4, nk = 3;
  Rng r;
  const auto wp = Rand(r, T * inh, -1.0f, 1.0f);
  const auto fold_ref = dv4::DsaIndexerWeightFold(wp, T, inh, ihd);
  const auto fold_got = dv4::DsaDevice()->weight_fold(g.q, wp, T, inh, ihd);
  CHECK(RelL2(fold_got, fold_ref) < kTol);

  const auto q = Rand(r, T * inh * ihd), k = Rand(r, nk * ihd);
  std::vector<int64_t> ws(T), we(T);
  for (int64_t t = 0; t < T; ++t) { ws[t] = 0; we[t] = t + 1; }  // causal
  const auto lref = dv4::DsaIndexerLogits(q, k, fold_ref, ws, we, T, nk, inh, ihd);
  const auto lgot = dv4::DsaDevice()->logits(g.q, q, k, fold_ref, ws, we, T, nk, inh, ihd);
  REQUIRE(lgot.size() == lref.size());
  for (size_t i = 0; i < lref.size(); ++i) {
    if (std::isinf(lref[i])) {
      CHECK(std::isinf(lgot[i]));  // out-of-window keys are -inf on BOTH
    } else {
      CHECK(lgot[i] == doctest::Approx(lref[i]).epsilon(1e-4));
    }
  }
}

TEST_CASE("W7-device DSA top-k select: CUDA matches host ids BIT-EXACT") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t T = 4, nk = 5, topk = 3;
  Rng r;
  const auto logits = Rand(r, T * nk, -3.0f, 3.0f);  // distinct random -> no ties
  std::vector<int64_t> ws(T), we(T);
  for (int64_t t = 0; t < T; ++t) { ws[t] = 0; we[t] = nk; }  // full window, n>topk
  const auto ref = dv4::DsaTopkSelect(logits, ws, we, T, nk, topk);
  const auto got = dv4::DsaDevice()->topk(g.q, logits, ws, we, T, nk, topk);
  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == ref[i]);  // selection is EXACT

  // Short-context (n<=topk) -> all candidates ascending, -1 pad; also exact.
  std::vector<int64_t> ws2(T), we2(T);
  for (int64_t t = 0; t < T; ++t) { ws2[t] = 0; we2[t] = t + 1; }  // t+1 <= topk for t<3
  const auto sref = dv4::DsaTopkSelect(logits, ws2, we2, T, nk, topk);
  const auto sgot = dv4::DsaDevice()->topk(g.q, logits, ws2, we2, T, nk, topk);
  for (size_t i = 0; i < sref.size(); ++i) CHECK(sgot[i] == sref[i]);
}

// #505 regression: the previous kernel sized `bool chosen[512]` and
// `int64_t picked[64]` by literal, so it could not represent the REAL
// `index_topk` — 512 on V4-Flash, 1024 on V4-Pro — and overflowed the thread
// stack on any candidate window wider than `topk`. Every pre-existing device
// case ran at topk=3/nk=5, which is why the bound was invisible. These are the
// real widths, with n > topk so the full-selection branch (not the
// short-context all-select) is the one under test.
TEST_CASE("W7-device DSA top-k select: REAL index_topk widths match host BIT-EXACT (#505)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;

  // (topk, nk) pairs: just past the old `picked[64]`, then the two shipped
  // index_topk values, each with a window wider than the old `chosen[512]`.
  const int64_t cases[][2] = {{65, 80}, {512, 600}, {1024, 1200}};
  for (const auto& c : cases) {
    const int64_t topk = c[0], nk = c[1], T = 3;
    const auto logits = Rand(r, T * nk, -3.0f, 3.0f);
    std::vector<int64_t> ws(T), we(T);
    for (int64_t t = 0; t < T; ++t) { ws[t] = 0; we[t] = nk; }  // n = nk > topk
    const auto ref = dv4::DsaTopkSelect(logits, ws, we, T, nk, topk);
    const auto got = dv4::DsaDevice()->topk(g.q, logits, ws, we, T, nk, topk);
    REQUIRE(got.size() == ref.size());
    int64_t mismatches = 0;
    for (size_t i = 0; i < ref.size(); ++i)
      if (got[i] != ref[i]) ++mismatches;
    CHECK(mismatches == 0);
    // The selection must be a full row of real keys — no -1 padding leaks in
    // when n > topk, and every emitted key is inside the window.
    for (int64_t t = 0; t < T; ++t)
      for (int64_t j = 0; j < topk; ++j) {
        const int64_t s = got[static_cast<size_t>(t * topk + j)];
        CHECK(s >= 0);
        CHECK(s < nk);
        if (j > 0) CHECK(s > got[static_cast<size_t>(t * topk + j - 1)]);  // ascending
      }
  }
}

// Ties are the case the total order (logit desc, index asc) exists to pin: with
// a coarsely quantized logit field many candidates share a value, so a kernel
// that resolved ties differently from the host reference would diverge here
// while passing on distinct random logits.
TEST_CASE("W7-device DSA top-k select: TIE-HEAVY rows match host BIT-EXACT (#505)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t topk = 128, nk = 300, T = 3;
  auto logits = Rand(r, T * nk, -2.0f, 2.0f);
  for (float& v : logits) v = static_cast<float>(static_cast<int>(v * 2.0f)) * 0.5f;
  std::vector<int64_t> ws(T), we(T);
  for (int64_t t = 0; t < T; ++t) { ws[t] = 0; we[t] = nk; }
  const auto ref = dv4::DsaTopkSelect(logits, ws, we, T, nk, topk);
  const auto got = dv4::DsaDevice()->topk(g.q, logits, ws, we, T, nk, topk);
  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == ref[i]);
}

// A non-zero window start is what the real indexer passes once a prefix has been
// evicted; the old emit path indexed its mask by `s - s0` and its picks by
// absolute `s`, so the two bounds interacted with s0 differently.
TEST_CASE("W7-device DSA top-k select: OFFSET window at real width matches host (#505)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t topk = 512, nk = 900, T = 2;
  const auto logits = Rand(r, T * nk, -3.0f, 3.0f);
  std::vector<int64_t> ws(T), we(T);
  for (int64_t t = 0; t < T; ++t) { ws[t] = 137; we[t] = nk; }  // n = 763 > topk
  const auto ref = dv4::DsaTopkSelect(logits, ws, we, T, nk, topk);
  const auto got = dv4::DsaDevice()->topk(g.q, logits, ws, we, T, nk, topk);
  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == ref[i]);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < topk; ++j)
      CHECK(got[static_cast<size_t>(t * topk + j)] >= 137);
}

// #552 finding 2: the kernel clamps its window with `s0 = ws[t] > 0 ? ws[t] : 0`
// and `s1 = we[t] < nk ? we[t] : nk` (cuda_deepseek_v4.cu:628-629), but no case
// passed a window that NEEDS clamping. Mutating either clamp away left all four
// #505 cases green while an off-device fuzz caught both immediately; on device
// they become out-of-bounds `logits` reads. These rows drive both clamps at a
// real width, against the same host reference.
TEST_CASE("W7-device DSA top-k select: OUT-OF-RANGE windows are clamped like host (#552)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t topk = 512, nk = 700, T = 4;
  const auto logits = Rand(r, T * nk, -3.0f, 3.0f);

  // Row 0 under-runs (ws < 0), row 1 over-runs (we > nk), row 2 does both, row 3
  // is in range as the control. Every row still has n > topk after clamping.
  std::vector<int64_t> ws{-9, 0, -4, 100};
  std::vector<int64_t> we{nk, nk + 37, nk + 12, nk};
  const auto ref = dv4::DsaTopkSelect(logits, ws, we, T, nk, topk);
  const auto got = dv4::DsaDevice()->topk(g.q, logits, ws, we, T, nk, topk);
  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == ref[i]);

  // No emitted key may escape the clamped window, which is what an unclamped
  // read would produce.
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < topk; ++j) {
      const int64_t s = got[static_cast<size_t>(t * topk + j)];
      CHECK(s >= 0);
      CHECK(s < nk);
    }
}

// #552 finding 3: `DsaTopkSelect` asserts `topk > 0` (deepseek_v4_dsa.cpp:76)
// while the device launcher used to return an empty vector instead. The two arms
// must refuse the same inputs, not just agree on the accepted ones.
TEST_CASE("W7-device DSA top-k select: non-positive topk REFUSED on both arms (#552)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t nk = 8, T = 2;
  const auto logits = Rand(r, T * nk, -1.0f, 1.0f);
  std::vector<int64_t> ws(T, 0), we(T, nk);
  for (const int64_t bad : {static_cast<int64_t>(0), static_cast<int64_t>(-1)}) {
    CHECK_THROWS(dv4::DsaTopkSelect(logits, ws, we, T, nk, bad));
    CHECK_THROWS(dv4::DsaDevice()->topk(g.q, logits, ws, we, T, nk, bad));
  }
}

TEST_CASE("W7-device attention-sink softmax + grouped output-LoRA: CUDA vs host (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const auto scores = Rand(r, 5, -2.0f, 2.0f);
  const auto sref = dv4::SoftmaxWithSink(scores, 0.5f);
  const auto sgot = dv4::DsaDevice()->softmax_sink(g.q, scores, 0.5f);
  CHECK(RelL2(sgot, sref) < kTol);
  // The sink removes probability mass: Σ prob < 1 on BOTH (property, not just tie).
  float sum = 0.0f;
  for (float p : sgot) sum += p;
  CHECK(sum < 1.0f);

  const int64_t T = 2, nh = 2, hd = 6, ng = 2, olr = 4, H = 8;
  const int64_t ipg = nh * hd / ng;
  const auto o = Rand(r, T * nh * hd), wa = Rand(r, ng * olr * ipg, -0.3f, 0.3f);
  const auto wb = Rand(r, H * ng * olr, -0.3f, 0.3f);
  const auto oref = dv4::GroupedOutputLora(o, wa, wb, T, nh, hd, ng, olr, H);
  const auto ogot = dv4::DsaDevice()->grouped_olora(g.q, o, wa, wb, T, nh, hd, ng, olr, H);
  CHECK(RelL2(ogot, oref) < kTol);
}

// ===========================================================================
// (3) Compressor family
// ===========================================================================
TEST_CASE("W7-device compressor save-APE + pool-norm: CUDA vs host (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t T = 3, width = 6, cr = 2;
  const auto score = Rand(r, T * width), ape = Rand(r, cr * width, -0.5f, 0.5f);
  std::vector<int64_t> pos = {0, 1, 2};
  const auto aref = dv4::CompressorSaveScoreApe(score, ape, pos, T, width, cr);
  const auto agot = dv4::CompressorDevice()->save_score_ape(g.q, score, ape, pos, T, width, cr);
  CHECK(RelL2(agot, aref) < kTol);

  const int64_t window = 2, hd = 6;
  const auto kv = Rand(r, window * hd), sc = Rand(r, window * hd);
  const auto rms = Rand(r, hd, 0.9f, 1.1f);
  for (auto valid : std::vector<std::vector<uint8_t>>{{1, 1}, {0, 1}}) {
    const auto pref = dv4::CompressorPoolNorm(kv, sc, valid, rms, 1e-6f, window, hd);
    const auto pgot = dv4::CompressorDevice()->pool_norm(g.q, kv, sc, valid, rms, 1e-6f, window, hd);
    CHECK(RelL2(pgot, pref) < kTol);
  }
}

TEST_CASE("W7-device fp8_ds_mla KV encode+decode round-trip: CUDA vs host (fp8 granularity)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const dv4::Fp8DsMlaLayout L = dv4::MakeFp8DsMlaLayout(/*nope=*/8, /*rope=*/4, /*qblk=*/4);
  const auto head = Rand(r, L.nope_head_dim + L.rope_head_dim, -2.0f, 2.0f);

  const auto htok = dv4::Fp8DsMlaEncodeToken(head, L);
  const auto dtok = dv4::CompressorDevice()->encode(g.q, head, L);
  // Decode BOTH tokens on host + device; the reconstructed latent agrees within
  // fp8 e4m3 granularity (3 mantissa bits -> ~0.05 relative, the W4 host bound).
  const auto hdec_h = dv4::Fp8DsMlaDecodeToken(htok, L);           // host token, host decode
  const auto ddec_d = dv4::CompressorDevice()->decode(g.q, dtok, L);  // device token, device decode
  CHECK(MaxAbs(ddec_d, hdec_h) < 0.05f * 2.0f);
  // Device decode of the HOST token vs host decode: e4m3 bit layout is standard,
  // so this is near-exact (isolates the decode path from the encode rounding).
  const auto hdec_on_dev = dv4::CompressorDevice()->decode(g.q, htok, L);
  CHECK(RelL2(hdec_on_dev, hdec_h) < 1e-3);
  // The RoPE tail is bf16-verbatim on both -> exact bit pattern round-trip.
  for (int64_t j = 0; j < L.rope_head_dim; ++j)
    CHECK(dtok.rope_bf16[j] == htok.rope_bf16[j]);
}

// ===========================================================================
// (4) MoE family
// ===========================================================================
TEST_CASE("W7-device sqrtsoftplus + clamped SwiGLU: CUDA vs host (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const auto x = Rand(r, 16, -4.0f, 4.0f);
  std::vector<float> spref(x.size());
  for (size_t i = 0; i < x.size(); ++i) spref[i] = dv4::SqrtSoftplus(x[i]);
  const auto spgot = dv4::MoeDevice()->sqrtsoftplus(g.q, x);
  CHECK(RelL2(spgot, spref) < kTol);

  // Clamped SwiGLU with values that exercise BOTH the gate max-clamp and the
  // up two-sided clamp (asymmetry load-bearing).
  const int64_t d = 6;
  std::vector<float> gate_up = {-5.0f, 3.0f, 12.0f, 0.5f, -1.0f, 8.0f,   // gate
                                -20.0f, 2.0f, 15.0f, -0.5f, 1.0f, -11.0f};  // up
  const auto cref = dv4::ClampedSwiGLU(gate_up, d, 10.0f, 1.0f, 0.0f);
  const auto cgot = dv4::MoeDevice()->clamped_swiglu(g.q, gate_up, d, 10.0f, 1.0f, 0.0f);
  CHECK(RelL2(cgot, cref) < kTol);
}

TEST_CASE("W7-device sqrtsoftplus/hash router: CUDA ids BIT-EXACT, weights near-tie") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t T = 4, E = 8, topk = 3, vocab = 12;
  const auto gating = Rand(r, T * E, -3.0f, 3.0f);
  const auto bias = Rand(r, E, -0.3f, 0.3f);

  // (a) learned top-k with the noaux_tc bias (selection biased, weights unbiased).
  {
    const auto ref = dv4::SqrtSoftplusRouteTopk(gating, T, E, topk, bias, true, 1.5f, {}, {}, vocab);
    const auto got = dv4::MoeDevice()->route(g.q, gating, T, E, topk, bias, true, 1.5f, {}, {}, vocab);
    REQUIRE(got.topk_ids.size() == ref.topk_ids.size());
    for (size_t i = 0; i < ref.topk_ids.size(); ++i) CHECK(got.topk_ids[i] == ref.topk_ids[i]);
    CHECK(RelL2(got.topk_weights, ref.topk_weights) < kTol);
  }
  // (b) hash route: tid2eid lookup bypasses top-k; weights from UNBIASED scores.
  {
    std::vector<int64_t> in_tokens = {3, 7, 1, 9};
    std::vector<int32_t> tid2eid(static_cast<size_t>(vocab * topk));
    for (int64_t tok = 0; tok < vocab; ++tok)
      for (int64_t j = 0; j < topk; ++j)
        tid2eid[static_cast<size_t>(tok * topk + j)] = static_cast<int32_t>((tok * 5 + j) % E);
    const auto ref = dv4::SqrtSoftplusRouteTopk(gating, T, E, topk, {}, true, 1.5f, in_tokens, tid2eid, vocab);
    const auto got = dv4::MoeDevice()->route(g.q, gating, T, E, topk, {}, true, 1.5f, in_tokens, tid2eid, vocab);
    for (size_t i = 0; i < ref.topk_ids.size(); ++i) CHECK(got.topk_ids[i] == ref.topk_ids[i]);
    CHECK(RelL2(got.topk_weights, ref.topk_weights) < kTol);
  }
}

// ds4-gap Lever 3 / Brick 10 — the warp-parallel router top-k (RouteWarpKernel,
// default-ON) must reproduce the single-thread RouteKernel BYTE-IDENTICALLY: same
// expert ids (argmax under a strict total order, tie-break = lower expert index) AND
// bit-exact weights (unbiased gather + renorm run the same float ops in j-order). The
// A/B flips VT_V4_ROUTE_WARP_TOPK in-process. RED-first: any divergence in selection
// order, tie-break, or weight arithmetic breaks the memcmp. Covers the DS4-realistic
// E=256/topk=6 (full 8-per-lane path) + the small structural config + a non-32-multiple
// E (invalid-lane guard).
TEST_CASE("Lever 3 warp-topk router == single-thread RouteKernel BYTE-IDENTICAL (A/B)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  struct Cfg { int64_t T, E, topk, vocab; };
  const Cfg cfgs[] = {{5, 256, 6, 100}, {4, 8, 3, 12}, {3, 33, 5, 20}};
  auto bytes_equal = [](const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
  };
  for (const Cfg& c : cfgs) {
    CAPTURE(c.E);
    CAPTURE(c.topk);
    const auto gating = Rand(r, c.T * c.E, -3.0f, 3.0f);
    const auto bias = Rand(r, c.E, -0.3f, 0.3f);
    // (a) learned biased top-k (selection biased, weights unbiased), renorm on.
    {
      setenv("VT_V4_ROUTE_WARP_TOPK", "0", 1);
      const auto st = dv4::MoeDevice()->route(g.q, gating, c.T, c.E, c.topk, bias, true, 1.5f, {},
                                              {}, c.vocab);
      setenv("VT_V4_ROUTE_WARP_TOPK", "1", 1);
      const auto wp = dv4::MoeDevice()->route(g.q, gating, c.T, c.E, c.topk, bias, true, 1.5f, {},
                                              {}, c.vocab);
      REQUIRE(wp.topk_ids.size() == st.topk_ids.size());
      for (size_t i = 0; i < st.topk_ids.size(); ++i) CHECK(wp.topk_ids[i] == st.topk_ids[i]);
      CHECK(bytes_equal(wp.topk_weights, st.topk_weights));  // BIT-EXACT, not near-tie
    }
    // (b) hash route (top-k bypass; weights gathered from UNBIASED scores).
    {
      std::vector<int64_t> in_tokens(static_cast<size_t>(c.T));
      for (int64_t t = 0; t < c.T; ++t) in_tokens[static_cast<size_t>(t)] = t * 7 + 3;
      std::vector<int32_t> tid2eid(static_cast<size_t>(c.vocab * c.topk));
      for (int64_t tok = 0; tok < c.vocab; ++tok)
        for (int64_t j = 0; j < c.topk; ++j)
          tid2eid[static_cast<size_t>(tok * c.topk + j)] =
              static_cast<int32_t>((tok * 5 + j) % c.E);
      setenv("VT_V4_ROUTE_WARP_TOPK", "0", 1);
      const auto st = dv4::MoeDevice()->route(g.q, gating, c.T, c.E, c.topk, {}, true, 1.5f,
                                              in_tokens, tid2eid, c.vocab);
      setenv("VT_V4_ROUTE_WARP_TOPK", "1", 1);
      const auto wp = dv4::MoeDevice()->route(g.q, gating, c.T, c.E, c.topk, {}, true, 1.5f,
                                              in_tokens, tid2eid, c.vocab);
      for (size_t i = 0; i < st.topk_ids.size(); ++i) CHECK(wp.topk_ids[i] == st.topk_ids[i]);
      CHECK(bytes_equal(wp.topk_weights, st.topk_weights));
    }
  }
  unsetenv("VT_V4_ROUTE_WARP_TOPK");
}

// ===========================================================================
// ForwardDevice composition gate: device forward == host forward (near-tie), at
// the tiny structural config (the four families all routed through CUDA).
// ===========================================================================
namespace {
DeepseekV4Params TinyParams() {
  DeepseekV4Params p;
  p.hidden_size = 8;
  p.num_hidden_layers = 4;
  p.vocab_size = 12;
  p.num_attention_heads = 2;
  p.num_key_value_heads = 1;
  p.rms_norm_eps = 1e-6f;
  p.max_position_embeddings = 4096;
  p.head_dim = 6;
  p.qk_rope_head_dim = 2;
  p.q_lora_rank = 4;
  p.o_lora_rank = 4;
  p.o_groups = 2;
  p.sliding_window = 128;
  p.rope_theta = 10000.0;
  p.compress_rope_theta = 160000.0;
  p.n_routed_experts = 4;
  p.num_experts_per_tok = 2;
  p.moe_intermediate_size = 6;
  p.n_shared_experts = 1;
  p.norm_topk_prob = true;
  p.routed_scaling_factor = 1.5;
  p.swiglu_limit = 10.0;
  p.scoring_func = "sqrtsoftplus";
  p.num_hash_layers = 2;
  p.expert_dtype = "fp4";
  p.hc_mult = 4;
  p.hc_sinkhorn_iters = 5;
  p.hc_eps = 1e-6;
  p.index_head_dim = 4;
  p.index_n_heads = 2;
  p.index_topk = 3;
  p.compress_ratios = {0, 4, 2, 4};
  return p;
}

DeepseekV4HostWeights TinyWeights(const DeepseekV4Params& p) {
  Rng rng;
  const int64_t H = p.hidden_size, V = p.vocab_size, hc = p.hc_mult;
  const int64_t nh = p.num_attention_heads, hd = p.head_dim, qlr = p.q_lora_rank;
  const int64_t og = p.o_groups, olr = p.o_lora_rank;
  const int64_t in_per_group = nh * hd / og;
  const int64_t ne = p.n_routed_experts, topk = p.num_experts_per_tok, mi = p.moe_intermediate_size;
  const int64_t inh = p.index_n_heads, ihd = p.index_head_dim;
  const int64_t hc3 = (2 + hc) * hc, hcH = hc * H;
  auto rnd = [&](int64_t n, float sc) { return Rand(rng, n, -sc, sc); };
  auto normw = [&](int64_t n) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& e : v) e = 1.0f + rng.next(-0.1f, 0.1f);
    return v;
  };
  DeepseekV4HostWeights hw;
  hw.embed = rnd(V * H, 0.8f);
  hw.lm_head = rnd(V * H, 0.5f);
  hw.final_norm_weight = normw(H);
  hw.hc_head_fn = rnd(hc * hcH, 0.2f);
  hw.hc_head_base = rnd(hc, 0.2f);
  hw.hc_head_scale = 0.5f;
  hw.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    DeepseekV4LayerHostWeights& L = hw.layers[static_cast<size_t>(l)];
    L.attn_norm_weight = normw(H);
    L.ffn_norm_weight = normw(H);
    L.hc_attn_fn = rnd(hc3 * hcH, 0.2f);
    L.hc_attn_base = rnd(hc3, 0.2f);
    L.hc_attn_scale = rnd(3, 0.5f);
    L.hc_ffn_fn = rnd(hc3 * hcH, 0.2f);
    L.hc_ffn_base = rnd(hc3, 0.2f);
    L.hc_ffn_scale = rnd(3, 0.5f);
    L.wq_a = rnd(qlr * H, 0.3f);
    L.q_norm_weight = normw(qlr);
    L.wq_b = rnd((nh * hd) * qlr, 0.3f);
    L.wkv = rnd(hd * H, 0.3f);
    L.kv_norm_weight = normw(hd);
    L.attn_sink = {0.7f, -0.4f};
    L.wo_a = rnd(og * olr * in_per_group, 0.3f);
    L.wo_b = rnd(H * (og * olr), 0.3f);
    if (p.has_indexer(l)) {
      L.idx_wq = rnd((inh * ihd) * H, 0.3f);
      L.idx_wk = rnd(ihd * H, 0.3f);
      L.idx_wproj = rnd(inh * H, 0.3f);
    }
    if (p.has_compressor(l)) {
      const int64_t cr = p.compress_ratio(l);
      L.comp_wgate = rnd(hd * H, 0.3f);
      L.comp_ape = rnd(cr * hd, 0.2f);
      L.comp_norm_weight = normw(hd);
    }
    L.gate_weight = rnd(ne * H, 0.4f);
    if (p.is_hash_layer(l)) {
      L.tid2eid.assign(static_cast<size_t>(V * topk), 0);
      for (int64_t tok = 0; tok < V; ++tok)
        for (int64_t j = 0; j < topk; ++j)
          L.tid2eid[static_cast<size_t>(tok * topk + j)] =
              static_cast<int32_t>((tok * 7 + j * 3 + 1) % ne);
    } else {
      L.gate_bias = rnd(ne, 0.3f);
    }
    L.shared_w1 = rnd(mi * H, 0.3f);
    L.shared_w3 = rnd(mi * H, 0.3f);
    L.shared_w2 = rnd(H * mi, 0.3f);
    L.exp_w1 = rnd(ne * mi * H, 0.3f);
    L.exp_w3 = rnd(ne * mi * H, 0.3f);
    L.exp_w2 = rnd(ne * H * mi, 0.3f);
  }
  return hw;
}
const std::vector<int32_t> kTokens = {3, 7, 1, 9, 4};
const std::vector<int32_t> kPositions = {0, 1, 2, 3, 4};
}  // namespace

TEST_CASE("W7-device ForwardDevice ASSEMBLES: device forward == host forward (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const DeepseekV4Params p = TinyParams();
  vllm::DeepseekV4Weights w;
  w.params = p;
  w.host = TinyWeights(p);
  w.has_host_weights = true;

  const std::vector<float> host =
      vllm::DeepseekV4ForwardHost(w.host, p, kTokens, kPositions, {}, vllm::V4Miswire::kNone, nullptr);
  vllm::v1::CommonAttentionMetadata attn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  const vllm::ForwardLogits dev =
      vllm::DeepseekV4Model::ForwardDevice(kTokens, kPositions, attn_meta, attn_kv, w, g.q, {});

  // ForwardDevice now returns DEVICE-RESIDENT logits on a CUDA queue (on_device(),
  // no full-[rows,vocab] D2H) — the shared-framework contract (qwen3_moe
  // WrapDeviceLogits). Materialize to host for the near-tie compare exactly as the
  // runner's VT_GPU_SAMPLE=0 A/B path does (backend Copy); ForwardComposeImpl has
  // already drained the lm_head GEMM, so the device buffer is complete.
  REQUIRE(dev.on_device());
  CHECK(dev.rows == static_cast<int64_t>(kTokens.size()));
  CHECK(dev.vocab == p.vocab_size);
  std::vector<float> devh(static_cast<size_t>(dev.rows) *
                          static_cast<size_t>(dev.vocab));
  gpu.Copy(g.q, devh.data(), dev.device_tensor.data, devh.size() * sizeof(float));
  gpu.Synchronize(g.q);
  REQUIRE(devh.size() == host.size());
  for (float v : devh) CHECK(std::isfinite(v));
  // The whole 4-family device composition matches the host oracle within the
  // accumulated near-tie envelope (device expf/sqrtf/rsqrt vs host, over 4 layers).
  CHECK(RelL2(devh, host) < 2e-3);
}

// Brick A: the device MLA decode attention kernel == the host SoftmaxWithSink
// reference (per-head sink softmax over the shared cached latent). Near-tie
// (device expf vs host std::exp is the only divergence — accumulation order is
// preserved). RED-first: dropping the sink (no_sink) changes the output.
TEST_CASE("DeepseekV4 device MLA decode attention == host SoftmaxWithSink (Brick A)") {
  if (!HasCuda()) return;  // DGX-only
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t nh = 4, hd = 64, kv_base = 5, T = 1;  // decode: 1 query, 6 keys
  const int64_t n = kv_base + T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::vector<float> q = Rand(r, T * nh * hd);
  const std::vector<float> kv = Rand(r, n * hd);
  const std::vector<float> sink = Rand(r, nh, 0.5f, 2.0f);  // positive → load-bearing

  auto host_ref = [&](bool no_sink) {
    std::vector<float> o(static_cast<size_t>(T) * nh * hd, 0.0f);
    for (int64_t h = 0; h < nh; ++h) {
      std::vector<float> sc(static_cast<size_t>(n));
      for (int64_t s = 0; s < n; ++s) {
        float acc = 0.0f;
        for (int64_t d = 0; d < hd; ++d) acc += q[h * hd + d] * kv[s * hd + d];
        sc[static_cast<size_t>(s)] = acc * scale;
      }
      const float sk = no_sink ? -std::numeric_limits<float>::infinity() : sink[h];
      const std::vector<float> probs = vllm::deepseek_v4::SoftmaxWithSink(sc, sk);
      for (int64_t s = 0; s < n; ++s)
        for (int64_t d = 0; d < hd; ++d)
          o[h * hd + d] += probs[static_cast<size_t>(s)] * kv[s * hd + d];
    }
    return o;
  };

  // device (unified-memory pointers; GB10 coherent access)
  std::vector<float> o(static_cast<size_t>(T) * nh * hd, 0.0f);
  vllm::deepseek_v4::DsaDevice()->decode_attn(g.q, o.data(), q.data(), kv.data(), sink.data(),
                                              nh, hd, kv_base, T, scale, /*no_sink=*/false);
  gpu.Synchronize(g.q);
  const std::vector<float> ref = host_ref(false);
  for (float v : o) CHECK(std::isfinite(v));
  CHECK(RelL2(o, ref) < 1e-5);  // near-tie (expf vs std::exp)

  // RED-first: no_sink must CHANGE the output (the sink is load-bearing).
  std::vector<float> o_ns(static_cast<size_t>(T) * nh * hd, 0.0f);
  vllm::deepseek_v4::DsaDevice()->decode_attn(g.q, o_ns.data(), q.data(), kv.data(), sink.data(),
                                              nh, hd, kv_base, T, scale, /*no_sink=*/true);
  gpu.Synchronize(g.q);
  CHECK(RelL2(o_ns, o) > 1e-4);
  CHECK(RelL2(o_ns, host_ref(true)) < 1e-5);  // and it matches the no-sink host ref
}

// Brick B: the IN-PLACE device clamped-SwiGLU (unified memory, no Upload/Download)
// == the host ClampedSwiGLU. Same ClampedSwiGLUKernel as the #183 clamped_swiglu.
// CHARACTERIZED NEAR-TIE (not bit-identical): the device SiLU sigmoid uses `expf`
// while the host uses `std::exp`, so results agree to ~last-ULP (RelL2 < 1e-5),
// NOT bitwise — stated, not hidden. RED-first: changing the clamp limit changes out.
TEST_CASE("DeepseekV4 device clamped-SwiGLU in place == host (Brick B)") {
  if (!HasCuda()) return;  // DGX-only
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t d = 256;
  const float limit = 7.0f, alpha = 1.0f, beta = 0.0f;
  const std::vector<float> gate_up = Rand(r, 2 * d, -10.0f, 10.0f);  // spans the clamp

  std::vector<float> o(static_cast<size_t>(d), 0.0f);
  vllm::deepseek_v4::MoeDevice()->clamped_swiglu_ip(g.q, o.data(), gate_up.data(), d, limit,
                                                    alpha, beta);
  gpu.Synchronize(g.q);
  const std::vector<float> ref = vllm::deepseek_v4::ClampedSwiGLU(gate_up, d, limit, alpha, beta);
  REQUIRE(o.size() == ref.size());
  for (float v : o) CHECK(std::isfinite(v));
  CHECK(RelL2(o, ref) < 1e-5);  // near-tie (device expf vs host std::exp in the SiLU)

  // RED-first: a different clamp limit must change the output.
  std::vector<float> o2(static_cast<size_t>(d), 0.0f);
  vllm::deepseek_v4::MoeDevice()->clamped_swiglu_ip(g.q, o2.data(), gate_up.data(), d, 1.0f,
                                                    alpha, beta);
  gpu.Synchronize(g.q);
  CHECK(RelL2(o2, o) > 1e-4);
}

// Brick B: the IN-PLACE MHC glue (post/head/pre) + router == the ROUND-TRIP #183
// launchers. Same kernels, so the results are BIT-IDENTICAL (the round-trip variant
// is already gated vs the host reference near-tie). RED-first: a perturbed input
// changes the in-place output.
TEST_CASE("DeepseekV4 device MHC + router in place == round-trip (Brick B)") {
  if (!HasCuda()) return;  // DGX-only
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t hc = 4, hidden = 8, iters = 5;
  const int64_t hc3 = (2 + hc) * hc;
  const float eps = 1e-6f;

  // MHC-post
  {
    const auto x = Rand(r, hidden), residual = Rand(r, hc * hidden);
    const auto post_mix = Rand(r, hc, 0.0f, 2.0f), comb = Rand(r, hc * hc, 0.0f, 1.0f);
    const auto rt = dv4::MhcDevice()->post(g.q, x, residual, post_mix, comb, hc, hidden);
    std::vector<float> ip(static_cast<size_t>(hc * hidden), 0.0f);
    dv4::MhcDevice()->post_ip(g.q, ip.data(), x.data(), residual.data(), post_mix.data(),
                             comb.data(), hc, hidden);
    gpu.Synchronize(g.q);
    REQUIRE(ip.size() == rt.size());
    for (size_t i = 0; i < ip.size(); ++i) CHECK(ip[i] == rt[i]);  // same kernel → bit-identical
  }
  // hc_head
  {
    const auto xh = Rand(r, hc * hidden), fn = Rand(r, hc * hc * hidden, -0.3f, 0.3f);
    const auto base = Rand(r, hc, -0.3f, 0.3f);
    const auto rt = dv4::MhcDevice()->head(g.q, xh, fn, 0.5f, base, hc, hidden, eps, eps);
    std::vector<float> ip(static_cast<size_t>(hidden), 0.0f);
    dv4::MhcDevice()->head_ip(g.q, ip.data(), xh.data(), fn.data(), 0.5f, base.data(), hc, hidden,
                             eps, eps);
    gpu.Synchronize(g.q);
    for (size_t i = 0; i < ip.size(); ++i) CHECK(ip[i] == rt[i]);
    // RED-first: a different scale changes the head output.
    std::vector<float> ip2(static_cast<size_t>(hidden), 0.0f);
    dv4::MhcDevice()->head_ip(g.q, ip2.data(), xh.data(), fn.data(), 1.7f, base.data(), hc, hidden,
                             eps, eps);
    gpu.Synchronize(g.q);
    CHECK(RelL2(ip2, ip) > 1e-4);
  }
  // MHC-pre (all four outputs)
  {
    const int64_t hcH = hc * hidden;
    const auto residual = Rand(r, hc * hidden, -1.0f, 1.0f);
    const auto fn = Rand(r, hc3 * hcH, -0.3f, 0.3f);
    const auto scale = Rand(r, 3, -0.5f, 0.5f);
    const auto base = Rand(r, hc3, -0.3f, 0.3f);
    const auto nw = Rand(r, hidden, 0.9f, 1.1f);
    const auto rt = dv4::MhcDevice()->pre(g.q, residual, fn, scale, base, hc, hidden, eps, eps,
                                          eps, 2.0f, iters, nw, eps);
    dv4::MhcPreResult ip;
    ip.pre_mix.resize(static_cast<size_t>(hc));
    ip.post_mix.resize(static_cast<size_t>(hc));
    ip.comb_mix.resize(static_cast<size_t>(hc * hc));
    ip.layer_input.resize(static_cast<size_t>(hidden));
    std::vector<float> mix(static_cast<size_t>(hc3 + 1));  // + folded-sqrsum slot (Lever 2)
    dv4::MhcDevice()->pre_ip(g.q, ip.pre_mix.data(), ip.post_mix.data(), ip.comb_mix.data(),
                            ip.layer_input.data(), mix.data(), residual.data(), fn.data(),
                            scale.data(), base.data(), hc, hidden, eps, eps, eps, 2.0f, iters,
                            nw.data(), true, eps);
    gpu.Synchronize(g.q);
    // pre_ip DEFAULTS to the ds4-fold FLOAT path (VT_V4_MHC_FUSED): the mix dots + the
    // sqrsum/norm reductions run in float (GB10 throttles FP64) vs the round-trip's
    // single-thread accumulation → CHARACTERIZED NEAR-TIE (float reduction reorder);
    // Sinkhorn + gates stay in host order. (VT_V4_MHC_FUSED=0 restores the double path,
    // itself a near-tie; both hold well within the 1e-3 tolerance at this tiny shape.)
    for (float v : ip.layer_input) CHECK(std::isfinite(v));
    CHECK(RelL2(ip.layer_input, rt.layer_input) < 1e-3);
    CHECK(RelL2(ip.comb_mix, rt.comb_mix) < 1e-3);
    // RED-first: a different residual changes the parallel output.
    dv4::MhcPreResult ip2;
    ip2.pre_mix.resize(static_cast<size_t>(hc));
    ip2.post_mix.resize(static_cast<size_t>(hc));
    ip2.comb_mix.resize(static_cast<size_t>(hc * hc));
    ip2.layer_input.resize(static_cast<size_t>(hidden));
    auto res2 = residual;
    res2[0] += 5.0f;
    dv4::MhcDevice()->pre_ip(g.q, ip2.pre_mix.data(), ip2.post_mix.data(), ip2.comb_mix.data(),
                            ip2.layer_input.data(), mix.data(), res2.data(), fn.data(),
                            scale.data(), base.data(), hc, hidden, eps, eps, eps, 2.0f, iters,
                            nw.data(), true, eps);
    gpu.Synchronize(g.q);
    CHECK(RelL2(ip2.layer_input, ip.layer_input) > 1e-4);
  }
  // router (non-hash, biased top-k)
  {
    const int64_t T = 2, E = 8, topk = 3;
    const auto gating = Rand(r, T * E, -2.0f, 2.0f);
    const auto bias = Rand(r, E, -0.5f, 0.5f);
    const auto rt = dv4::MoeDevice()->route(g.q, gating, T, E, topk, bias, true, 1.5f, {}, {}, 0);
    dv4::MoeRouteResult ip;
    ip.topk_ids.assign(static_cast<size_t>(T * topk), 0);
    ip.topk_weights.assign(static_cast<size_t>(T * topk), 0.0f);
    dv4::MoeDevice()->route_ip(g.q, ip.topk_ids.data(), ip.topk_weights.data(), gating.data(), T,
                              E, topk, bias.data(), true, nullptr, false, nullptr, 0, true, 1.5f);
    gpu.Synchronize(g.q);
    for (size_t i = 0; i < ip.topk_ids.size(); ++i) CHECK(ip.topk_ids[i] == rt.topk_ids[i]);
    for (size_t i = 0; i < ip.topk_weights.size(); ++i) CHECK(ip.topk_weights[i] == rt.topk_weights[i]);
  }
}

// Brick C: the folded-in device glue (RMSNorm / RoPE / MoE combine) == host refs.
// All three are near-ties: RMSNorm (block-reduction reorder), RoPE (device cos/sin
// vs libm), combine (device FMA contraction vs host separate mul+add). RED-first each.
TEST_CASE("DeepseekV4 device RMSNorm / RoPE / MoE combine == host (Brick C)") {
  if (!HasCuda()) return;  // DGX-only
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;

  // device RMSNorm (weighted) == host double-sequential ref (near-tie).
  {
    const int64_t n = 300;
    const float eps = 1e-6f;
    const auto x = Rand(r, n, -2.0f, 2.0f);
    const auto w = Rand(r, n, 0.5f, 1.5f);
    std::vector<float> out(static_cast<size_t>(n), 0.0f);
    dv4::DsaDevice()->rms_norm(g.q, out.data(), x.data(), w.data(), n, eps, /*has_w=*/true);
    gpu.Synchronize(g.q);
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float rr = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(n)) + eps);
    std::vector<float> ref(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) ref[i] = x[i] * rr * w[i];
    for (float v : out) CHECK(std::isfinite(v));
    CHECK(RelL2(out, ref) < 1e-5);
    // RED-first: has_w=false (no weight) differs from the weighted result.
    std::vector<float> out2(static_cast<size_t>(n), 0.0f);
    dv4::DsaDevice()->rms_norm(g.q, out2.data(), x.data(), w.data(), n, eps, /*has_w=*/false);
    gpu.Synchronize(g.q);
    CHECK(RelL2(out2, out) > 1e-4);
  }

  // device RoPE (YaRN) == host RopeInplaceLayer ref (near-tie, cos/sin lib).
  {
    const int64_t rows = 5, rd = 64, stride = 96, off = 0;
    const double base = 160000.0, fscale = 1.0 / 16.0, ext = 1.0;
    const int64_t n_ctx = 4096;
    const double bfast = 32.0, bslow = 1.0;
    std::vector<int> pos(static_cast<size_t>(rows));
    for (int64_t i = 0; i < rows; ++i) pos[static_cast<size_t>(i)] = static_cast<int>(3 + i);
    auto v = Rand(r, rows * stride, -1.0f, 1.0f);
    auto vd = v;  // device copy
    // host ref matching RopeKernel exactly (sequential recurrence).
    const double kPi = 3.14159265358979323846;
    auto corr = [&](double beta) {
      return static_cast<double>(rd) * std::log(static_cast<double>(n_ctx) / (beta * 2.0 * kPi)) /
             (2.0 * std::log(base));
    };
    const double clo = std::max(0.0, std::floor(corr(bfast)));
    const double chi = std::min(static_cast<double>(rd - 1), std::ceil(corr(bslow)));
    const double tscale = std::pow(base, -2.0 / static_cast<double>(rd));
    for (int64_t row = 0; row < rows; ++row) {
      float* vv = v.data() + row * stride + off;
      double te = static_cast<double>(pos[static_cast<size_t>(row)]);
      for (int64_t i = 0; i < rd; i += 2) {
        const double ti = fscale * te;
        double th = ti;
        const double y = (static_cast<double>(i / 2) - clo) / std::max(0.001, chi - clo);
        const double ramp = (1.0 - std::min(1.0, std::max(0.0, y))) * ext;
        th = ti * (1.0 - ramp) + te * ramp;
        const float c = static_cast<float>(std::cos(th)), s = static_cast<float>(std::sin(th));
        const float x0 = vv[i], x1 = vv[i + 1];
        vv[i] = x0 * c - x1 * s;
        vv[i + 1] = x0 * s + x1 * c;
        te *= tscale;
      }
    }
    dv4::DsaDevice()->rope(g.q, vd.data(), rows, stride, off, rd, pos.data(), base, fscale, ext,
                           n_ctx, bfast, bslow, /*inverse=*/false);
    gpu.Synchronize(g.q);
    CHECK(RelL2(vd, v) < 1e-5);
    // RED-first: inverse RoPE differs from forward.
    auto vi = Rand(r, rows * stride, -1.0f, 1.0f);
    auto vif = vi;
    dv4::DsaDevice()->rope(g.q, vi.data(), rows, stride, off, rd, pos.data(), base, fscale, ext,
                           n_ctx, bfast, bslow, /*inverse=*/false);
    dv4::DsaDevice()->rope(g.q, vif.data(), rows, stride, off, rd, pos.data(), base, fscale, ext,
                           n_ctx, bfast, bslow, /*inverse=*/true);
    gpu.Synchronize(g.q);
    CHECK(RelL2(vif, vi) > 1e-4);
  }

  // device MoE combine == host (near-tie: device FMA contraction vs host mul+add).
  {
    const int64_t A = 7, H = 64;
    const auto eo = Rand(r, A * H, -2.0f, 2.0f);
    const auto wts = Rand(r, A, -1.0f, 1.0f);
    std::vector<float> out(static_cast<size_t>(H), 0.0f);
    dv4::MoeDevice()->moe_combine(g.q, out.data(), eo.data(), wts.data(), A, H);
    gpu.Synchronize(g.q);
    std::vector<float> ref(static_cast<size_t>(H), 0.0f);
    for (int64_t h = 0; h < H; ++h) {
      float acc = 0.0f;
      for (int64_t a = 0; a < A; ++a) acc += wts[a] * eo[a * H + h];
      ref[h] = acc;
    }
    for (float v : out) CHECK(std::isfinite(v));
    // CHARACTERIZED NEAR-TIE (not bit-identical): the device compiles the combine's
    // `weights[a]*eo + acc` to a fused multiply-add (FMA contraction) while the host
    // does a separate multiply+add → ~last-ULP difference. Stated, not hidden.
    CHECK(RelL2(out, ref) < 1e-5);
    // RED-first: negated weights change the output.
    auto wn = wts;
    for (auto& x : wn) x = -x;
    std::vector<float> out2(static_cast<size_t>(H), 0.0f);
    dv4::MoeDevice()->moe_combine(g.q, out2.data(), eo.data(), wn.data(), A, H);
    gpu.Synchronize(g.q);
    CHECK(RelL2(out2, out) > 1e-4);
  }
}

// Brick C part 2 — the BATCHED per-head q-RMS: rms_norm_rows over `rows` [n]
// segments in ONE launch == rows independent rms_norm calls (per-row identical) ==
// host double-sequential ref (the same characterized near-tie). RED-first each.
TEST_CASE("DeepseekV4 device rms_norm_rows (batched per-head q-RMS) == host (Brick C part 2)") {
  if (!HasCuda()) return;  // DGX-only
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t rows = 64, n = 512;  // the real MLA per-head q-RMS (nh=64, hd=512)
  const float eps = 1e-6f;
  const auto x = Rand(r, rows * n, -2.0f, 2.0f);
  std::vector<float> out(static_cast<size_t>(rows * n), 0.0f);
  dv4::DsaDevice()->rms_norm_rows(g.q, out.data(), x.data(), nullptr, rows, n, eps, /*has_w=*/false);
  gpu.Synchronize(g.q);
  // (a) == host double-sequential per-row RMSNorm (unweighted).
  std::vector<float> ref(static_cast<size_t>(rows * n));
  for (int64_t row = 0; row < rows; ++row) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[row * n + i]) * x[row * n + i];
    const float rr = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(n)) + eps);
    for (int64_t i = 0; i < n; ++i) ref[row * n + i] = x[row * n + i] * rr;
  }
  for (float v : out) CHECK(std::isfinite(v));
  CHECK(RelL2(out, ref) < 1e-5);
  // (b) == the per-row single-launch rms_norm (subsumes it exactly).
  std::vector<float> perrow(static_cast<size_t>(rows * n), 0.0f);
  for (int64_t row = 0; row < rows; ++row)
    dv4::DsaDevice()->rms_norm(g.q, perrow.data() + row * n, x.data() + row * n, nullptr, n, eps,
                               /*has_w=*/false);
  gpu.Synchronize(g.q);
  CHECK(RelL2(out, perrow) < 1e-6);
  // RED-first: a shared weight (has_w=true) changes the result.
  const auto w = Rand(r, n, 0.5f, 1.5f);
  std::vector<float> outw(static_cast<size_t>(rows * n), 0.0f);
  dv4::DsaDevice()->rms_norm_rows(g.q, outw.data(), x.data(), w.data(), rows, n, eps, /*has_w=*/true);
  gpu.Synchronize(g.q);
  CHECK(RelL2(outw, out) > 1e-4);
}

// Brick 7 — the FUSED per-row RMSNorm+RoPE (norm_rope_rows) vs the split {rms_norm_rows ;
// rope} launch pair it replaces. norm_rope_rows DEFAULTS to the ds4-fold FLOAT kernel
// (VT_V4_ROPE_FLOAT): the RMS reduction + the RoPE pow/cos/sin run in FLOAT (GB10 throttles
// FP64) and the theta uses ds4's direct powf(base,-i/r) per pair vs the split path's DOUBLE
// block-reduce + double `theta_extrap *= theta_scale` left-fold → CHARACTERIZED NEAR-TIE
// (float reduction reorder + float transcendentals; RelL2 < 1e-3 at this tiny shape).
// (VT_V4_ROPE_FLOAT=0 restores the double NormRopeRowsKernel, itself BYTE-IDENTICAL to the
// split path.) All three resident-decode modes: q (has_w=false, do_norm, fwd) / kv (has_w=
// true, do_norm, fwd) / o (do_norm=false, inverse). RED-first each.
TEST_CASE("DeepseekV4 device norm_rope_rows == split {rms_norm_rows;rope} NEAR-TIE (Brick 7)") {
  if (!HasCuda()) return;  // DGX-only
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t rows = 64, n = 512, rope = 64, off = n - rope;  // real MLA geometry
  const float eps = 1e-6f;
  const double base = 10000.0, fscale = 0.8, ext = 1.0, bfast = 32.0, bslow = 1.0;
  const int64_t n_ctx = 4096;
  std::vector<int> pos(static_cast<size_t>(rows));
  for (int64_t i = 0; i < rows; ++i) pos[static_cast<size_t>(i)] = static_cast<int>(37 + i);
  auto* dsa = dv4::DsaDevice();

  auto split = [&](const std::vector<float>& x, const std::vector<float>* w, bool do_norm,
                   bool inverse) {
    std::vector<float> out(x.size());
    if (do_norm) {
      dsa->rms_norm_rows(g.q, out.data(), x.data(), w ? w->data() : nullptr, rows, n, eps,
                         w != nullptr);
    } else {
      out = x;  // inverse-only site (o): no norm, rope in place
    }
    dsa->rope(g.q, out.data(), rows, n, off, rope, pos.data(), base, fscale, ext, n_ctx, bfast,
              bslow, inverse);
    gpu.Synchronize(g.q);
    return out;
  };
  auto fused = [&](const std::vector<float>& x, const std::vector<float>* w, bool do_norm,
                   bool inverse) {
    // do_norm=false (inverse-o) ropes the tail IN PLACE (reads out), matching the real
    // forward's norm_rope_rows(o, o, ...); do_norm reads a separate `in` (kv: kraw→slot).
    std::vector<float> out = do_norm ? std::vector<float>(x.size(), 0.0f) : x;
    const float* in = do_norm ? x.data() : out.data();
    dsa->norm_rope_rows(g.q, out.data(), in, w ? w->data() : nullptr, rows, n, off, rope,
                        pos.data(), base, fscale, ext, n_ctx, bfast, bslow, inverse, w != nullptr,
                        do_norm, eps);
    gpu.Synchronize(g.q);
    return out;
  };

  const auto x = Rand(r, rows * n, -2.0f, 2.0f);
  const auto w = Rand(r, n, 0.5f, 1.5f);
  // (q) has_w=false, do_norm, forward — BYTE-identical.
  const auto q_split = split(x, nullptr, /*do_norm=*/true, /*inverse=*/false);
  const auto q_fused = fused(x, nullptr, /*do_norm=*/true, /*inverse=*/false);
  for (float v : q_fused) CHECK(std::isfinite(v));
  CHECK(RelL2(q_fused, q_split) < 1e-3);
  // (kv) has_w=true, do_norm, forward — float near-tie.
  const auto kv_split = split(x, &w, /*do_norm=*/true, /*inverse=*/false);
  const auto kv_fused = fused(x, &w, /*do_norm=*/true, /*inverse=*/false);
  CHECK(RelL2(kv_fused, kv_split) < 1e-3);
  // (o) do_norm=false, inverse — float near-tie (rope tail only, no norm).
  const auto o_split = split(x, nullptr, /*do_norm=*/false, /*inverse=*/true);
  const auto o_fused = fused(x, nullptr, /*do_norm=*/false, /*inverse=*/true);
  CHECK(RelL2(o_fused, o_split) < 1e-3);
  // RED-first: fwd vs inverse differ; and the weighted (kv) path differs from unweighted.
  CHECK(RelL2(q_fused, o_fused) > 1e-4);
  CHECK(RelL2(kv_fused, q_fused) > 1e-4);
}

// Brick D step 1 — the DEVICE router gate (gating[e] = Σ_h x[h]·bf16→f32(W[e*H+h]))
// == the host reference (sequential f32 dot, exact `bits<<16` bf16 upcast). A near-tie
// vs the host CPU MatmulBT (FMA-contraction), so the TRUE bar is real-model token-
// identical routing; the unit case pins the numeric match + RED-first weight change.
TEST_CASE("DeepseekV4 device router_gate (f32-act x bf16-weight) == host (Brick D)") {
  if (!HasCuda()) return;  // DGX-only
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t ne = 40, H = 128;  // small stand-in for [256,4096]
  const auto x = Rand(r, H, -1.5f, 1.5f);
  // bf16 weights (the loader's `ffn.gate.weight` dtype): round f32->bf16, store bits.
  std::vector<uint16_t> wbf16(static_cast<size_t>(ne * H));
  std::vector<float> wval(static_cast<size_t>(ne * H));
  {
    const auto wf = Rand(r, ne * H, -1.0f, 1.0f);
    for (int64_t i = 0; i < ne * H; ++i) {
      wbf16[static_cast<size_t>(i)] = vt::F32ToBF16(wf[static_cast<size_t>(i)]);
      wval[static_cast<size_t>(i)] = vt::BF16ToF32(wbf16[static_cast<size_t>(i)]);  // exact decoded
    }
  }
  std::vector<float> gating(static_cast<size_t>(ne), 0.0f);
  dv4::MoeDevice()->router_gate(g.q, gating.data(), x.data(), wbf16.data(), ne, H);
  gpu.Synchronize(g.q);
  std::vector<float> ref(static_cast<size_t>(ne));
  for (int64_t e = 0; e < ne; ++e) {
    float acc = 0.0f;
    for (int64_t h = 0; h < H; ++h) acc += x[h] * wval[e * H + h];
    ref[static_cast<size_t>(e)] = acc;
  }
  for (float v : gating) CHECK(std::isfinite(v));
  CHECK(RelL2(gating, ref) < 1e-5);  // near-tie (device FMA vs host mul+add) — see comment
  // RED-first: perturbing one expert's weights changes that gating logit.
  auto wb2 = wbf16;
  for (int64_t h = 0; h < H; ++h) wb2[static_cast<size_t>(3 * H + h)] = vt::F32ToBF16(0.7f);
  std::vector<float> g2(static_cast<size_t>(ne), 0.0f);
  dv4::MoeDevice()->router_gate(g.q, g2.data(), x.data(), wb2.data(), ne, H);
  gpu.Synchronize(g.q);
  CHECK(RelL2(g2, gating) > 1e-4);
}

// Brick D step 2 — the GRAPH decode attention decode_attn_g (attends cache[0..len)
// + the current key deck_new, len from a DEVICE buffer) == the eager decode_attn over
// the FULL cache (with deck appended at row len). Same key set + order ⇒ BIT-IDENTICAL
// (this is what lets the captured graph replay a growing context). RED-first: a wrong
// length changes the result.
TEST_CASE("DeepseekV4 device decode_attn_g == eager decode_attn (Brick D graph)") {
  if (!HasCuda()) return;  // DGX-only
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t nh = 8, hd = 32, len = 11;  // len prior keys + 1 current
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const auto q = Rand(r, nh * hd, -1.0f, 1.0f);
  const auto sink = Rand(r, nh, -0.5f, 0.5f);
  const auto prior = Rand(r, len * hd, -1.0f, 1.0f);  // cache[0..len)
  const auto deck = Rand(r, hd, -1.0f, 1.0f);         // the current token's key
  // eager reference: full cache = [prior; deck], kv_base=len, T=1.
  std::vector<float> full(static_cast<size_t>((len + 1) * hd));
  std::copy(prior.begin(), prior.end(), full.begin());
  std::copy(deck.begin(), deck.end(), full.begin() + len * hd);
  std::vector<float> ref(static_cast<size_t>(nh * hd), 0.0f);
  dv4::DsaDevice()->decode_attn(g.q, ref.data(), q.data(), full.data(), sink.data(), nh, hd,
                                /*kv_base=*/len, /*T=*/1, scale, /*no_sink=*/false);
  gpu.Synchronize(g.q);
  // graph variant: cache=prior (len), deck_new=deck, len from a device buffer.
  std::vector<int> len_dev(1, static_cast<int>(len));
  std::vector<float> got(static_cast<size_t>(nh * hd), 0.0f);
  dv4::DsaDevice()->decode_attn_g(g.q, got.data(), q.data(), prior.data(), deck.data(), sink.data(),
                                  nh, hd, len_dev.data(), /*max_cap=*/64, scale, /*no_sink=*/false);
  gpu.Synchronize(g.q);
  for (int64_t i = 0; i < nh * hd; ++i) CHECK(got[static_cast<size_t>(i)] == ref[static_cast<size_t>(i)]);
  // RED-first: a wrong length (drop the current key) changes the output.
  std::vector<int> len_bad(1, static_cast<int>(len) - 3);
  std::vector<float> got2(static_cast<size_t>(nh * hd), 0.0f);
  dv4::DsaDevice()->decode_attn_g(g.q, got2.data(), q.data(), prior.data(), deck.data(), sink.data(),
                                  nh, hd, len_bad.data(), /*max_cap=*/64, scale, /*no_sink=*/false);
  gpu.Synchronize(g.q);
  CHECK(RelL2(got2, got) > 1e-4);
}
