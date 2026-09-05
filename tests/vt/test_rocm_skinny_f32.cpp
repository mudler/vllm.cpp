// vllm.cpp original (vt runtime); no upstream mirror.
//
// GFX1100-TG200 lever B2 focused gate: the f32-OUTPUT decode-skinny arm
// (VT_SKINNY_BF16=1) for bf16-in/f32-out MatmulBT at M<=4. The engine
// population that motivates it is the Qwen3.5 GDN BA pair
// (ProjectGdnBA, qwen3_5.cpp:3663-3664): N=32, K=2560, m=1, which today
// falls through every decode-skinny gate in MatmulBTKernelRocm (all require
// a bf16 output) onto hipblasGemmEx -> rocBLAS's large-M Tensile tile
// MT128x32x16 (~73.7us to stream a 164 KiB weight; evidence file section
// 15.1).
//
// Numerics contract: the arm is NOT bit-exact vs the default route by
// construction (different reduction order), so unlike test_rocm_quant_dot
// this gate asserts the sibling 1e-6 NMSE band vs the CPU oracle on BOTH
// arms, a tight ON-vs-OFF agreement band, and ROUTING witnesses through
// host-side dispatch counters (outputs cannot witness routing here because
// both arms are numerically correct).
//
// RED-first contract: before the seam exists this file fails to LINK
// (SkinnyF32RouteCountsForTesting undefined) and the routing cases fail
// behaviorally once counters exist but the env arm does not engage.
//
// Skips cleanly when the build has HIP but the box has no AMD GPU.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_runtime.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace vt::rocm {
// Host-side routing witness (the test_rocm_quant_dot.cpp F1/F2 convention):
// process-global counters bumped on exactly the branch taken per
// bf16-in/f32-out MatmulBT dispatch. Both arms are numerically correct, so
// no output comparison can witness routing -- these integers can.
struct SkinnyF32RouteCounts {
  long long blas;    // fell through to hipblasGemmEx (default route)
  long long skinny;  // took the VT_SKINNY_BF16 wvSplitK-class arm
};
SkinnyF32RouteCounts SkinnyF32RouteCountsForTesting();
void SkinnyF32ResetRouteCountsForTesting();
}  // namespace vt::rocm

namespace {

Device GpuDev() { return Device{DeviceType::kROCM, 0}; }

// test_rocm_quant_dot.cpp:79 — the band the sibling gates hold their arms to.
constexpr double kMaxNmseVsCpu = 1e-6;

double Nmse(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den > 0 ? num / den : num;
}

Tensor DevTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = p;
  t.dtype = dt;
  t.device = GpuDev();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct EnvGuard {
  explicit EnvGuard(bool on) { ::setenv("VT_SKINNY_BF16", on ? "1" : "0", 1); }
  ~EnvGuard() { ::unsetenv("VT_SKINNY_BF16"); }
};

std::vector<uint16_t> RandomBf16(size_t n, uint32_t seed) {
  std::vector<uint16_t> v(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    // Small-magnitude values: keeps both arms' f32 accumulation well-
    // conditioned so the NMSE bands measure reduction order, not conditioning.
    const float f = (static_cast<float>(s >> 8) / 8388608.0f - 1.0f) * 0.125f;
    v[i] = vt::F32ToBF16(f);
  }
  return v;
}

std::vector<float> CpuOracleBt(const std::vector<uint16_t>& a_bf16,
                               const std::vector<uint16_t>& b_bf16, int64_t m,
                               int64_t n, int64_t k) {
  std::vector<float> out(static_cast<size_t>(m * n), 0.0f);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t l = 0; l < k; ++l)
        acc += static_cast<double>(
                   vt::BF16ToF32(a_bf16[static_cast<size_t>(i * k + l)])) *
               static_cast<double>(
                   vt::BF16ToF32(b_bf16[static_cast<size_t>(j * k + l)]));
      out[static_cast<size_t>(i * n + j)] = static_cast<float>(acc);
    }
  return out;
}

// One shape, both arms: returns per-arm outputs and asserts the shared
// contract (oracle band on both, ON-vs-OFF agreement). Route deltas are
// returned so callers can assert routing too.
struct ArmRun {
  std::vector<std::vector<float>> out;
  long long blas_delta;
  long long skinny_delta;
};

ArmRun RunBothArms(Backend& gpu, Queue gq, const std::vector<uint16_t>& a_bf16,
                   const std::vector<uint16_t>& b_bf16, int64_t m, int64_t n,
                   int64_t k) {
  ArmRun run;
  void* d_a = gpu.Alloc(a_bf16.size() * 2);
  void* d_b = gpu.Alloc(b_bf16.size() * 2);
  gpu.Copy(gq, d_a, a_bf16.data(), a_bf16.size() * 2);
  gpu.Copy(gq, d_b, b_bf16.data(), b_bf16.size() * 2);
  run.out.resize(2);
  vt::rocm::SkinnyF32ResetRouteCountsForTesting();
  const auto before = vt::rocm::SkinnyF32RouteCountsForTesting();
  for (int arm = 0; arm < 2; ++arm) {
    void* d_o = gpu.Alloc(4 * static_cast<size_t>(m * n));
    {
      EnvGuard guard(arm == 1);
      Tensor at = DevTensor(d_a, DType::kBF16, {m, k});
      Tensor bt = DevTensor(d_b, DType::kBF16, {n, k});
      Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
      vt::MatmulBT(gq, ot, at, bt);
      run.out[arm].resize(static_cast<size_t>(m * n), 0.0f);
      gpu.Copy(gq, run.out[arm].data(), d_o, run.out[arm].size() * 4);
      gpu.Synchronize(gq);
    }
    gpu.Free(d_o);
  }
  const auto after = vt::rocm::SkinnyF32RouteCountsForTesting();
  run.blas_delta = after.blas - before.blas;
  run.skinny_delta = after.skinny - before.skinny;
  gpu.Free(d_a);
  gpu.Free(d_b);
  return run;
}

}  // namespace

TEST_CASE("ROCm f32-out decode-skinny arm (VT_SKINNY_BF16=1): NMSE vs CPU oracle and routing witnesses") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm f32-out skinny gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();

  struct ShapeCase {
    int64_t m, n, k;
    bool arm_serves;  // expected VT_SKINNY_BF16=1 routing decision
    const char* name;
  };
  // Engine-realistic anchor first: EXACTLY the ProjectGdnBA decode shape
  // (evidence 15.1 rows 3+4). Then gate-boundary edges: even-N variants the
  // kernel serves, and the exclusions (odd N, K%8!=0, m beyond the skinny
  // range) that must stay on the default BLAS route under ON.
  const std::vector<ShapeCase> shapes = {
      {1, 32, 2560, true, "gdn-ba-engine-shape"},
      {1, 64, 4096, true, "even-n-larger-k"},
      {1, 10, 512, true, "minimal-even-n"},
      {4, 32, 2560, true, "m-at-upper-edge"},
      {1, 33, 2560, false, "odd-n-stays-blas"},
      {2, 33, 2560, false, "odd-n-and-m2-stays-blas"},
      {1, 32, 12, false, "k-not-multiple-of-8"},
      {5, 32, 2560, false, "m-past-skinny-range"},
      {1, 8, 2560, false, "n-at-feature-floor"},
  };
  for (const ShapeCase& sc : shapes) {
    CAPTURE(sc.name);
    CAPTURE(sc.m);
    CAPTURE(sc.n);
    CAPTURE(sc.k);
    const std::vector<uint16_t> a = RandomBf16(
        static_cast<size_t>(sc.m * sc.k), 0x5EEDu + static_cast<uint32_t>(sc.n));
    const std::vector<uint16_t> b = RandomBf16(
        static_cast<size_t>(sc.n * sc.k), 0xA11CEu + static_cast<uint32_t>(sc.k));

    const std::vector<float> ref =
        CpuOracleBt(a, b, sc.m, sc.n, sc.k);

    const ArmRun run = RunBothArms(gpu, gq, a, b, sc.m, sc.n, sc.k);

    // Routing witness over the TWO dispatches (OFF then ON). The counters
    // only track the bf16-in/f32-out population with M in [1,4]; inside it,
    // OFF always routes to BLAS and ON's branch is decided by the shape gate
    // alone; outside it (e.g. m=5) neither dispatch is counted.
    const bool in_pop = sc.m <= 4;
    CHECK(run.blas_delta == (in_pop ? 1 : 0) + (in_pop && !sc.arm_serves ? 1 : 0));
    CHECK(run.skinny_delta == (sc.arm_serves ? 1 : 0));

    for (int arm = 0; arm < 2; ++arm) {
      CAPTURE(arm);
      const double nmse = Nmse(run.out[static_cast<size_t>(arm)], ref);
      CAPTURE(nmse);
      CHECK(nmse <= kMaxNmseVsCpu);
    }
    // Cross-arm agreement (same accumulator precision, different tree).
    const double nmse_cross = Nmse(run.out[1], run.out[0]);
    CAPTURE(nmse_cross);
    CHECK(nmse_cross <= kMaxNmseVsCpu);
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("ROCm f32-out skinny routing witness: TRUE-unset behaves like OFF (default-OFF inertness)") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm f32-out skinny gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  // EnvGuard(false) writes "0" -- it can NEVER witness a true unset. The
  // first window below therefore constructs NO guard at all (the F-1 repair;
  // test_rocm_quant_dot.cpp F1 convention): run_window only touches the
  // environment for the explicit windows, so the true-unset dispatch sees
  // getenv()==NULL and the engine default must route to BLAS exactly as an
  // explicit "0" does.
  const std::vector<uint16_t> a = RandomBf16(2560, 0x5EEDu);
  const std::vector<uint16_t> b = RandomBf16(32 * 2560, 0xA11CEu);
  void* d_a = gpu.Alloc(a.size() * 2);
  void* d_b = gpu.Alloc(b.size() * 2);
  gpu.Copy(gq, d_a, a.data(), a.size() * 2);
  gpu.Copy(gq, d_b, b.data(), b.size() * 2);

  enum class WindowEnv { kTrueUnset, kExplicitOff, kExplicitOn };
  const auto run_window = [&](WindowEnv env) {
    void* d_o = gpu.Alloc(4 * 32);
    std::optional<EnvGuard> guard;
    if (env != WindowEnv::kTrueUnset) {
      guard.emplace(env == WindowEnv::kExplicitOn);
    }
    Tensor at = DevTensor(d_a, DType::kBF16, {1, 2560});
    Tensor bt = DevTensor(d_b, DType::kBF16, {32, 2560});
    Tensor ot = DevTensor(d_o, DType::kF32, {1, 32});
    vt::MatmulBT(gq, ot, at, bt);
    gpu.Synchronize(gq);
    gpu.Free(d_o);
  };

  vt::rocm::SkinnyF32ResetRouteCountsForTesting();
  {
    ::unsetenv("VT_SKINNY_BF16");  // true-unset window
    run_window(WindowEnv::kTrueUnset);
  }
  const auto unset_counts = vt::rocm::SkinnyF32RouteCountsForTesting();

  vt::rocm::SkinnyF32ResetRouteCountsForTesting();
  {
    EnvGuard guard(false);  // explicit "0"
    run_window(WindowEnv::kExplicitOff);
  }
  const auto off_counts = vt::rocm::SkinnyF32RouteCountsForTesting();

  vt::rocm::SkinnyF32ResetRouteCountsForTesting();
  {
    EnvGuard guard(true);  // "1"
    run_window(WindowEnv::kExplicitOn);
  }
  const auto on_counts = vt::rocm::SkinnyF32RouteCountsForTesting();

  CHECK(unset_counts.blas == 1);
  CHECK(unset_counts.skinny == 0);
  CHECK(off_counts.blas == 1);
  CHECK(off_counts.skinny == 0);
  CHECK(on_counts.blas == 0);
  CHECK(on_counts.skinny == 1);
  gpu.Free(d_a);
  gpu.Free(d_b);
  gpu.DestroyQueue(gq);
}

// Conditional regression test for the skinny-GEMM tail-store OOB
// (rocm_skinny_gemm.hip: wvSplitKSml store loop).  The kernel iterates the M
// dimension in kYtile-sized steps; when M is not a multiple of kYtile the last
// wave's y=1 store writes past the valid M range.  With the production
// kYtile=2 the caller guard (N > 8 && N % 2 == 0) and the M-in-[1,4] gate
// keep the OOB writes inside the allocated buffer (the buffer is M*N ≥ 1*10
// elements, so C[1] is still in-bounds), so the bug is invisible on the
// production bf16-output path.  It becomes visible with a guard band and is
// guarded against by the `if (m + y < M)` store check.
//
// This test is CONDITIONAL: it runs only under VT_SKINNY_BF16=1 (the
// experimental f32-output arm) and never fires on the production bf16-output
// path, per the #2894 reviewer's split demand.
TEST_CASE("ROCm f32-out skinny tail-store OOB regression (VT_SKINNY_BF16=1): guard band untouched for odd M") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm skinny OOB regression skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();

  // M=1 is not a multiple of kYtile=2: the kernel's y=1 store would write
  // C[1] past the valid M=0 row.  N=32 (>8, even) and K=256 (%8==0) satisfy
  // the dispatch guard so the skinny path is taken under VT_SKINNY_BF16=1.
  struct OobCase { int64_t m, n, k; const char* name; };
  const OobCase cases[] = {
      {1, 32, 256, "m=1 odd vs kYtile=2"},
      {3, 32, 256, "m=3 odd vs kYtile=2"},
  };
  for (const OobCase& oc : cases) {
    CAPTURE(oc.name);
    CAPTURE(oc.m);
    CAPTURE(oc.n);
    CAPTURE(oc.k);
    const std::vector<uint16_t> a =
        RandomBf16(static_cast<size_t>(oc.m * oc.k), 0xBEEFu);
    const std::vector<uint16_t> b =
        RandomBf16(static_cast<size_t>(oc.n * oc.k), 0xF00Du);

    void* d_a = gpu.Alloc(a.size() * 2);
    void* d_b = gpu.Alloc(b.size() * 2);
    gpu.Copy(gq, d_a, a.data(), a.size() * 2);
    gpu.Copy(gq, d_b, b.data(), b.size() * 2);

    const size_t out_elems = static_cast<size_t>(oc.m * oc.n);
    const size_t guard_elems = 64;
    // f32 output: 4 bytes per element.
    void* d_o = gpu.Alloc(4 * (out_elems + guard_elems));

    // Fill the entire allocation (output + guard) with a sentinel pattern.
    std::vector<uint32_t> fill(out_elems + guard_elems, 0xDEADBEEFu);
    gpu.Copy(gq, d_o, fill.data(), fill.size() * 4);
    gpu.Synchronize(gq);

    {
      EnvGuard guard(true);  // VT_SKINNY_BF16=1 — experimental arm only
      Tensor at = DevTensor(d_a, DType::kBF16, {oc.m, oc.k});
      Tensor bt = DevTensor(d_b, DType::kBF16, {oc.n, oc.k});
      Tensor ot = DevTensor(d_o, DType::kF32, {oc.m, oc.n});
      vt::MatmulBT(gq, ot, at, bt);
      gpu.Synchronize(gq);
    }

    std::vector<uint32_t> got(out_elems + guard_elems);
    gpu.Copy(gq, got.data(), d_o, got.size() * 4);
    gpu.Synchronize(gq);

    // The guard band beyond the output must be untouched.
    for (size_t i = out_elems; i < out_elems + guard_elems; ++i)
      CHECK(got[i] == 0xDEADBEEFu);

    // The output itself must match the CPU oracle (the guard fix does not
    // change correct outputs, only suppresses the OOB store).
    const std::vector<float> ref = CpuOracleBt(a, b, oc.m, oc.n, oc.k);
    std::vector<float> out_f(out_elems);
    for (size_t i = 0; i < out_elems; ++i) {
      float f;
      std::memcpy(&f, &got[i], 4);
      out_f[i] = f;
    }
    const double nmse = Nmse(out_f, ref);
    CAPTURE(nmse);
    CHECK(nmse <= kMaxNmseVsCpu);

    gpu.Free(d_a);
    gpu.Free(d_b);
    gpu.Free(d_o);
  }
  gpu.DestroyQueue(gq);
}
