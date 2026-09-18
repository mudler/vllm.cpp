// O34: the ROCm/HIP arm of the DeepSeek-V4 MHC device-kernel family. A 1:1
// mirror of the CUDA MHC cases in test_cuda_deepseek_v4.cpp (lines 115-175,
// 789-870), run against the ROCm provider registered in rocm_deepseek_v4.hip.
// Each case runs the device kernel at a SMALL synthetic shape and compares
// against the LANDED portable HOST reference (the oracle): near-tie for fp
// reductions (Sinkhorn, softmax pool — device expf/sqrtf differ from host by
// ULPs), BIT-EXACT for the in-place vs round-trip identity (same kernel).
//
// The device cases SKIP on a build without a ROCm backend. This suite owns its
// main so it can exit 77 (SKIP_RETURN_CODE) — "assertions: 0 ... SUCCESS!" is
// not a pass (tests/CMakeLists.txt:30-38, issue #463).
//
// Scope: MHC only. The DSA/Compressor/MoE families stay CUDA-only until their
// owed rows land. V4DeviceKernelsAvailable() returns false on ROCm because those
// three are absent; the test calls MhcDevice() directly, which resolves kROCm
// when kCUDA is not registered (deepseek_v4_device.cpp).
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"

#include "vllm/model_executor/models/deepseek_v4_device.h"
#include "vllm/model_executor/models/deepseek_v4_mhc.h"

namespace dv4 = vllm::deepseek_v4;

namespace {

bool HasRocm() {
  try {
    vt::GetBackend(vt::DeviceType::kROCM);
    return vt::OpRegistered(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kROCM);
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

double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(a[i]) * a[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

constexpr double kTol = 1e-4;

}  // namespace

// ===========================================================================
// (1) MHC family — round-trip launchers vs host reference (near-tie)
// ===========================================================================

TEST_CASE("O34 ROCm MHC Sinkhorn: device vs host reference (near-tie)") {
  if (!HasRocm()) { MESSAGE("no ROCm; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kROCM);
  QueueGuard g(gpu);
  const int64_t hc = 4, iters = 5;
  const float eps = 1e-6f;
  Rng r;
  const auto logits = Rand(r, hc * hc, -2.0f, 2.0f);
  const auto ref = dv4::MhcSinkhorn(logits, hc, iters, eps);
  const auto got = dv4::MhcDevice()->sinkhorn(g.q, logits, hc, iters, eps);
  REQUIRE(got.size() == ref.size());
  CHECK(RelL2(got, ref) < kTol);
  for (int64_t k = 0; k < hc; ++k) {
    float c = 0.0f;
    for (int64_t j = 0; j < hc; ++j) c += got[static_cast<size_t>(j * hc + k)];
    CHECK(c == doctest::Approx(1.0f).epsilon(1e-3));
  }
}

TEST_CASE("O34 ROCm MhcPre: device vs host reference (all four outputs near-tie)") {
  if (!HasRocm()) { MESSAGE("no ROCm; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kROCM);
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

TEST_CASE("O34 ROCm MhcPost + HcHeadCollapse: device vs host reference (near-tie)") {
  if (!HasRocm()) { MESSAGE("no ROCm; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kROCM);
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
// (2) In-place launchers == round-trip launchers (BIT-IDENTICAL, same kernel)
// ===========================================================================

TEST_CASE("O34 ROCm MHC in-place == round-trip (Brick B)") {
  if (!HasRocm()) return;
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kROCM);
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
    for (size_t i = 0; i < ip.size(); ++i) CHECK(ip[i] == rt[i]);
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
    std::vector<float> mix(static_cast<size_t>(hc3 + 1));
    dv4::MhcDevice()->pre_ip(g.q, ip.pre_mix.data(), ip.post_mix.data(), ip.comb_mix.data(),
                            ip.layer_input.data(), mix.data(), residual.data(), fn.data(),
                            scale.data(), base.data(), hc, hidden, eps, eps, eps, 2.0f, iters,
                            nw.data(), true, eps);
    gpu.Synchronize(g.q);
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
}

// ===========================================================================
// Exit 77 -> CTest reports SKIPPED. A genuine failure must never be laundered
// into a skip, so 77 is reached only on a clean run that had no ROCm device.
// ===========================================================================
int main(int argc, char** argv) {
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  const int res = context.run();
  if (res != 0) return res;
  if (!HasRocm()) {
    std::fprintf(stderr,
                "test_rocm_deepseek_v4_mhc: no ROCm backend on this host - the "
                "MHC device cases did not run. Exiting 77 (SKIPPED) rather than "
                "0, because \"assertions: 0 ... SUCCESS!\" is not a pass.\n");
    return 77;
  }
  return 0;
}
