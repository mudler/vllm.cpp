// The EXL3 m<=8 GEMV arm and its selection envelope — MODEL-DSV4-EXL3 W2c.
//
// PORTED FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//   exllamav3_ext/quant/exl3_gemv.cu:29-42    the two env knobs
//   exllamav3_ext/quant/exl3_gemv.cu:46-72    exl3_gemv_cfg, the shape envelope
//   exllamav3_ext/quant/exl3_gemv.cu:108-114  the hard eligibility checks
//   exllamav3_ext/quant/exl3_gemv_kernel.cuh:31  EXL3_GEMV_MAX_M
//
// WHAT IS GATED HERE, AND WHAT IS NOT.
//
// The ENVELOPE is pure integer arithmetic over (cc, m, k, n, K, cb, mode,
// narrow_coresident) and is gated on any machine. That matters more here than it
// did for the GEMM shape table, because the sentence this row inherited about
// this envelope — "Ada/Blackwell are memory-bound here and keep the regular
// kernel" — describes a guard that is COMMENTED OUT at `exl3_gemv.cu:53`. A
// quoted comment is not a gate; these cases are.
//
// The KERNEL is not gated here on a machine with no GPU, and its bound is not
// tier 3. It accumulates in fp16 and folds to f32 only every FOLD iterations
// (`exl3_gemv_kernel.cuh:37-52,317-330`), which is a different NUMERIC arm from
// the f32-accumulating regular kernel. `.agents/specs/model-dsv4-exl3.md`
// `## W2cd design` W2c-3 states its own bound, tier 3c, and the device case
// below is where it is measured. That case SKIPS loudly and still asserts.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rng;
using exl3_test::UlpF16;

bool HasCudaExl3() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kCUDA);
    return vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA);
  } catch (const std::runtime_error&) {
    return false;
  }
}

// The two shapes this checkpoint's TP1-coalesced experts have (spec
// `## The format`): w1/w3 are k=4096 n=2048, w2 is k=2048 n=4096.
constexpr int kW13K = 4096, kW13N = 2048;
constexpr int kW2K = 2048, kW2N = 4096;

}  // namespace

// ─── W2c-1: the compute-capability guard is DISABLED upstream ────────────────

TEST_CASE("exl3 gemv: no compute-capability test is live in the envelope") {
  // `exl3_gemv.cu:53` is `//if (cc != CC_AMPERE) return -1;`. If that line were
  // live, every non-Ampere bucket would return -1 for every shape. It is not, so
  // w2's shape is eligible on EVERY bucket that reaches the shape tests, and the
  // buckets differ from each other only where upstream's LIVE branches say they
  // do (`:65`, the Ada K==3 row).
  const int kMode = 1;  // the heuristic, upstream's default
  for (vt::Exl3Cc cc : {vt::Exl3Cc::kOld, vt::Exl3Cc::kAmpere, vt::Exl3Cc::kAda,
                        vt::Exl3Cc::kHopper, vt::Exl3Cc::kBlackwell}) {
    // w2, K = 3, cb = 1: `:67` `size_k <= 2048 && size_n <= 8192` fires for
    // every bucket, so config 0 (narrow), with NO occupancy input.
    CHECK(vt::Exl3GemvSelectConfig(cc, 1, kW2K, kW2N, 3, 1, kMode,
                                   /*narrow_coresident=*/0) == 0);
  }
}

// ─── W2c-2: what the envelope resolves to for THIS checkpoint ────────────────

TEST_CASE("exl3 gemv: w2 is eligible on GB10 and w1/w3 rest on an occupancy query") {
  const vt::Exl3Cc bw = vt::Exl3Cc::kBlackwell;
  const int kMode = 1;

  // w2 (k=2048, n=4096): `:67` fires. Independent of occupancy, so BOTH ends of
  // the occupancy range agree.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW2K, kW2N, 3, 1, kMode, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW2K, kW2N, 3, 1, kMode, 1 << 20) == 0);

  // w1/w3 (k=4096, n=2048): `:67` does NOT fire, so `:68` `K == 3` returns -1
  // UNLESS `:66` fires first, which needs `2048 / 32 = 64 <= narrow_coresident`.
  // The threshold is EXACTLY 64 and both sides of it are pinned, because the
  // whole point of the entry is that the verdict is a device query this row has
  // not made.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW13K, kW13N, 3, 1, kMode, 63) == -1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW13K, kW13N, 3, 1, kMode, 64) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW13K, kW13N, 3, 1, kMode, 65) == 0);
}

TEST_CASE("exl3 gemv: the other branches of the envelope are upstream's too") {
  const int kMode = 1;
  // `:64` K == 2 splits on n at 8192, on EVERY bucket.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8192, 2, 1, kMode, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8320, 2, 1, kMode, 0) == 1);
  // `:65` K == 3 on ADA takes the same split; on Blackwell it does not (that is
  // the one branch where the buckets genuinely differ).
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 4096, 8192, 3, 1, kMode, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 4096, 8320, 3, 1, kMode, 0) == 1);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8320, 3, 1, kMode, 0) == -1);
  // `:69` K == 4, big n, small k -> the wide config.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8192, 4, 1, kMode, 0) == 1);
  // `:70` is Ampere-only even though nothing else is.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAmpere, 1, 5120, 10240, 4, 1, kMode, 0) == 1);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 5120, 10240, 4, 1, kMode, 0) == -1);
  // `:48` mode 0 turns the whole path off; `:54` mode 2 takes it wherever the
  // hard constraints allow; `:55`/`:56` force one config.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW2K, kW2N, 3, 1, 0, 0) == -1);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW13K, kW13N, 3, 1, 2, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW13K, kW13N, 3, 1, 3, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW13K, kW13N, 3, 1, 4, 0) == 1);
}

TEST_CASE("exl3 gemv: the hard constraints refuse before any heuristic runs") {
  // `exl3_gemv.cu:110-114`, in upstream's own order.
  CHECK(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 1, /*has_su_sv=*/true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 1, /*has_su_sv=*/false));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 1, 1, true));   // K < 2
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 5, 1, true));   // K > 4
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 0, true));   // K != 4 && cb == 0
  CHECK(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 4, 0, true));         // K == 4 admits cb 0
  CHECK(vt::Exl3GemvHardEligible(vt::kExl3GemvMaxM, kW2K, kW2N, 3, 1, true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(vt::kExl3GemvMaxM + 1, kW2K, kW2N, 3, 1, true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, 2048 + 16, kW2N, 3, 1, true));  // size_k % 128
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, 4096 + 16, 3, 1, true));  // size_n % 128
  // The constant itself, so a change to it is a red rather than a silent
  // widening (`exl3_gemv_kernel.cuh:31`).
  CHECK(vt::kExl3GemvMaxM == 8);
  // The envelope also enforces the hard bound on m, independently, because
  // upstream repeats it inside `exl3_gemv_cfg` (`:51`).
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, vt::kExl3GemvMaxM + 1, kW2K, kW2N, 3, 1,
                                 1, 0) == -1);
}

TEST_CASE("exl3 gemv: the env knobs parse exactly as upstream's do") {
  // `exl3_gemv.cu:29-34`: unset is 1, everything else is atoi.
  CHECK(vt::Exl3GemvParseMode(nullptr) == 1);
  CHECK(vt::Exl3GemvParseMode("0") == 0);
  CHECK(vt::Exl3GemvParseMode("1") == 1);
  CHECK(vt::Exl3GemvParseMode("2") == 2);
  CHECK(vt::Exl3GemvParseMode("4") == 4);
  // `exl3_gemv.cu:37-42`: unset is -1.
  CHECK(vt::Exl3GemvParseSmemMode(nullptr) == -1);
  CHECK(vt::Exl3GemvParseSmemMode("0") == 0);
  CHECK(vt::Exl3GemvParseSmemMode("1") == 1);
}

// ─── W2c-3: the device arm, and the bound that is NOT tier 3 ─────────────────

TEST_CASE("exl3 device: the GEMV arm meets tier 3c on a shape it is eligible for") {
  if (!HasCudaExl3()) {
    MESSAGE(
        "SKIPPED, no CUDA device: MODEL-DSV4-EXL3 W2c's tier-3c bound is PENDING, and so is "
        "`narrow_coresident`, the occupancy query that alone decides whether the w1/w3 shape "
        "is GEMV-eligible at all. dgx.casa is flapping. Reproduce with: "
        "rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemv -V");
    // A skip that asserts NOTHING reports `assertions: 0`, which reads as a pass.
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA));
    return;
  }
  // FORCED, through `Exl3GemmArgs::force_gemv`, which mirrors upstream's own
  // direct entry point (`exl3_gemv.cu:171-241`, "errors if the call is not
  // hard-eligible"). Forcing is what makes this a gate rather than a coin flip
  // on a heuristic: without it a device whose occupancy declines the shape would
  // measure the REGULAR kernel and report tier 3c green.
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue dq = cb.CreateQueue();
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  const int64_t m = 1, k = kW2K, n = kW2N;
  Exl3Fixture f = MakeFixture(k, n, 3, 0x5EEDu);
  std::vector<uint16_t> a(static_cast<size_t>(m * k));
  Rng rng;
  for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

  // The reference is the CPU arm, which `test_exl3_gemm` already gates against
  // the f64 chain at tier 3. Comparing against it rather than re-deriving f64
  // here keeps ONE reference for both device arms.
  std::vector<uint16_t> ref(static_cast<size_t>(m * n), 0), got(static_cast<size_t>(m * n), 0);
  std::vector<uint16_t> a_had_h(static_cast<size_t>(m * k), 0);
  vt::Exl3GemmArgs ha;
  ha.bits = 3;
  ha.codebook = 1;
  {
    vt::Tensor ta = vt::Tensor::Contiguous(a.data(), vt::DType::kF16, hq.device, {m, k});
    vt::Tensor tah = vt::Tensor::Contiguous(a_had_h.data(), vt::DType::kF16, hq.device, {m, k});
    vt::Tensor tc = vt::Tensor::Contiguous(ref.data(), vt::DType::kF16, hq.device, {m, n});
    vt::Tensor tb = vt::Tensor::Contiguous(f.trellis.data(), vt::DType::kI8, hq.device,
                                           {k / 16, n / 16, 32 * 3});
    vt::Tensor tsuh = vt::Tensor::Contiguous(f.suh.data(), vt::DType::kF16, hq.device, {k});
    vt::Tensor tsvh = vt::Tensor::Contiguous(f.svh.data(), vt::DType::kF16, hq.device, {n});
    vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, ha);
  }

  void* d_a = cb.Alloc(a.size() * 2);
  void* d_ah = cb.Alloc(a.size() * 2);
  void* d_c = cb.Alloc(got.size() * 2);
  void* d_b = cb.Alloc(f.trellis.size() * 2);
  void* d_suh = cb.Alloc(f.suh.size() * 2);
  void* d_svh = cb.Alloc(f.svh.size() * 2);
  cb.Copy(dq, d_a, a.data(), a.size() * 2);
  cb.Copy(dq, d_b, f.trellis.data(), f.trellis.size() * 2);
  cb.Copy(dq, d_suh, f.suh.data(), f.suh.size() * 2);
  cb.Copy(dq, d_svh, f.svh.data(), f.svh.size() * 2);
  vt::Tensor da = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor dah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor dc = vt::Tensor::Contiguous(d_c, vt::DType::kF16, dq.device, {m, n});
  vt::Tensor db =
      vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device, {k / 16, n / 16, 32 * 3});
  vt::Tensor dsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
  vt::Tensor dsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
  vt::Exl3GemmArgs da_args = ha;
  da_args.force_gemv = 1;  // upstream's `force`: bypasses the heuristic, not the hard checks
  vt::Exl3Gemm(dq, dc, da, db, dsuh, dsvh, dah, da_args);
  cb.Synchronize(dq);
  cb.Copy(dq, got.data(), d_c, got.size() * 2);
  cb.Synchronize(dq);

  double sq = 0.0, rq = 0.0, worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double r = vt::F16ToF32(ref[i]);
    const double g = vt::F16ToF32(got[i]);
    sq += (g - r) * (g - r);
    rq += r * r;
    worst = std::max(worst, std::fabs(g - r));
  }
  const double rms_ref = std::sqrt(rq / static_cast<double>(got.size()));
  const double rel = std::sqrt(sq / static_cast<double>(got.size())) / rms_ref;
  MESSAGE("tier 3c: relative RMS ", rel, ", worst elementwise ", worst);
  // `## W2cd design` W2c-3. NOT tier 3's 1.0e-3: this arm accumulates in fp16.
  CHECK(rel <= 6.0e-3);
  CHECK(worst <= 64.0 * UlpF16(rms_ref));

  cb.Free(d_a);
  cb.Free(d_ah);
  cb.Free(d_c);
  cb.Free(d_b);
  cb.Free(d_suh);
  cb.Free(d_svh);
  cb.DestroyQueue(dq);
}
