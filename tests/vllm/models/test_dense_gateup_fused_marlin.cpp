// PERF-27B-DENSE-MARLIN-GATEUP (issue #365, spec
// .agents/specs/perf-27b-dense-marlin-gateup.md) — the DECISIONS that route the
// 27B's dense W4A16 MLP through the fused Marlin gate_up pair.
//
// WHAT THIS FILE CAN AND CANNOT PIN, stated plainly. The fused gate_up GEMM is
// `vt::MoeGroupedGemmNvfp4Marlin` / `vt::MarlinDenseGemm` over an N-concatenated
// [2N,K] NVFP4 operand: CUDA-only, and only in a build that has the vendored
// Marlin NVFP4 kernel (VT_MARLIN_NVFP4). Its ARITHMETIC therefore cannot execute
// here, and this file does not pretend to pin numbers. What it pins is every
// decision that is made BEFORE the kernel and that decides whether the kernel is
// legal to use at all:
//
//   * the shape/scale PRECONDITION that makes N-concatenating gate and up into
//     one operand legal — asserted term by term, including the `scale2` equality
//     that the merged single global scale depends on (spec §4/§6: the fusion is
//     unreachable, not silently relaxable, if the two shards disagree);
//   * the row's toggle and its DEFAULT, which is now ON — spec §3.3 made the
//     flip conditional on a same-binary A/B, and that A/B measured a win at
//     both c1 and c8 with complete separation and identical tokens;
//   * that a device/build with no Marlin NVFP4 op registered NEVER selects the
//     fused path, so every non-CUDA backend keeps the split pair byte-for-byte;
//   * that the fused pair resident is the EXISTING one, keyed on the gate
//     weight's own `resident_marlin_pair` slot — a dense gate weight keys it
//     exactly like a shared-expert gate weight, so there is no second cache.
//
// The numeric bar (fused == split within the split-K reduce's grouping, and
// token-exactness against the pinned oracle) belongs to the GPU box. Note that
// tests/vllm/model_executor/layers/test_linear_method.cpp:202 already records,
// from a measured run, that the fused and split Marlin paths are NOT
// bit-identical — Marlin's fp32 split-K reduce groups the K-slices differently
// for a [2N,K] operand than for two [N,K] operands — so "byte-identical to the
// split path" is the wrong bar and this file does not assert it.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include <type_traits>

#include "vllm/model_executor/models/dense_nvfp4_gemm.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vt/device.h"

namespace {

using vllm::DenseMlpWeights;
using vllm::MoeBlockWeights;
using vllm::Nvfp4Weight;
using vt::DType;

namespace dn = vllm::dense_nvfp4;

// A non-empty NVFP4 W4A16 weight (alpha == 0 => IsTrueW4A4() false), shaped like
// one half of a dense MLP gate_up: [N=intermediate, K=hidden]. The bytes are
// never read here — only the shape/scale fields the precondition inspects are.
Nvfp4Weight MakeW4A16(int64_t N, int64_t K, float scale2) {
  Nvfp4Weight w;
  w.n = N;
  w.k = K;
  w.scale2 = scale2;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K / 2;
  w.packed.bytes.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 2), 0);
  w.scale.dtype = DType::kI8;
  w.scale.rank = 2;
  w.scale.shape[0] = N;
  w.scale.shape[1] = K / 16;
  w.scale.bytes.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 16), 0);
  return w;
}

// The MXFP4 analog: compressed-tensors `mxfp4-pack-quantized` — group 32, E8M0
// block scales [N, K/32], NO per-tensor global (scale2 stays 0). This is what a
// dense MLP would carry if `LoadDenseMlp` ever grew the MXFP4 spelling the
// classic-dense loader already has (qwen3_weights.cpp:122, LoadCtMxfp4W4A16).
Nvfp4Weight MakeMxfp4(int64_t N, int64_t K) {
  Nvfp4Weight w;
  w.n = N;
  w.k = K;
  w.is_mxfp4 = true;
  w.group_size = 32;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K / 2;
  w.packed.bytes.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 2), 0);
  w.scale.dtype = DType::kI8;
  w.scale.rank = 2;
  w.scale.shape[0] = N;
  w.scale.shape[1] = K / 32;
  w.scale.bytes.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 32), 0);
  return w;
}

// Read a lever DIRECTLY from the process environment, deliberately NOT through
// the header's own parser: these cases exist to catch a reader that was
// hardwired, misspelled, or dropped, and re-using the thing under test to
// compute the expectation would make every such mutation self-consistent.
bool LeverOn(const char* name) {
  const char* e = std::getenv(name);
  return !(e != nullptr && e[0] == '0');
}

// Qwen3.6-27B dense MLP: hidden 5120, intermediate 25600.
constexpr int64_t kN = 25600;
constexpr int64_t kK = 5120;

}  // namespace

TEST_CASE("dense gate_up fusion: the pair precondition is asserted, not assumed") {
  const Nvfp4Weight gate = MakeW4A16(kN, kK, 0.5F);

  SUBCASE("a matching W4A16 pair is fusable") {
    const Nvfp4Weight up = MakeW4A16(kN, kK, 0.5F);
    CHECK(dn::GateUpPairFusableShape(gate, up));
  }

  SUBCASE("UNEQUAL scale2 is NOT fusable") {
    // The load-bearing term. The merged resident emits ONE global scale over
    // both shards (vLLM's merged parameter has exactly one weight_global_scale);
    // two shards with different scale2 cannot share it, so relaxing this would
    // change numerics rather than fuse them. Spec §6 stop condition.
    const Nvfp4Weight up = MakeW4A16(kN, kK, 0.25F);
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, up));
  }

  SUBCASE("a scale2 that differs only in the last bit is NOT fusable") {
    const Nvfp4Weight up = MakeW4A16(kN, kK, 0.5F + 6e-8F);
    REQUIRE(up.scale2 != gate.scale2);
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, up));
  }

  SUBCASE("an empty half is NOT fusable") {
    const Nvfp4Weight empty;
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, empty));
    CHECK_FALSE(dn::GateUpPairFusableShape(empty, gate));
    CHECK_FALSE(dn::GateUpPairFusableShape(empty, empty));
  }

  SUBCASE("a true-W4A4 half is NOT fusable (that is the CUTLASS merged path)") {
    Nvfp4Weight w4a4 = MakeW4A16(kN, kK, 0.5F);
    w4a4.alpha = 0.125F;  // IsTrueW4A4()
    REQUIRE(w4a4.IsTrueW4A4());
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, w4a4));
    CHECK_FALSE(dn::GateUpPairFusableShape(w4a4, gate));
  }

  SUBCASE("mismatched N or K is NOT fusable") {
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, MakeW4A16(kN / 2, kK, 0.5F)));
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, MakeW4A16(kN, kK / 2, 0.5F)));
  }

  SUBCASE("a mismatched block-scale FORMAT is NOT fusable") {
    // Row-stacking the two shards' scales is only meaningful when both are read
    // with the same group size and the same scale encoding.
    Nvfp4Weight mx = MakeW4A16(kN, kK, 0.5F);
    mx.is_mxfp4 = true;
    mx.group_size = 32;
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, mx));
    Nvfp4Weight group_only = MakeW4A16(kN, kK, 0.5F);
    group_only.group_size = 32;
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, group_only));
  }

  SUBCASE("a half flagged MXFP4 at a MATCHING group size is still NOT fusable") {
    // The subcase above moves `is_mxfp4` AND `group_size` together, so the
    // group_size term alone accounted for both CHECKs and `gw.is_mxfp4 ==
    // uw.is_mxfp4` could be deleted with everything still green (fresh-review
    // mutation M5c). Hold the group fixed and move ONLY the encoding flag: the
    // two shards then disagree about whether a scale byte is fp8-e4m3 with a
    // per-tensor global or a bare E8M0 exponent, which row-stacking cannot
    // reconcile at any group size.
    Nvfp4Weight encoding_only = MakeW4A16(kN, kK, 0.5F);
    encoding_only.is_mxfp4 = true;  // group_size deliberately left at 16
    REQUIRE(encoding_only.group_size == gate.group_size);
    REQUIRE(encoding_only.scale2 == gate.scale2);
    REQUIRE(encoding_only.n == gate.n);
    REQUIRE(encoding_only.k == gate.k);
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, encoding_only));
    CHECK_FALSE(dn::GateUpPairFusableShape(encoding_only, gate));
  }
}

TEST_CASE("dense gate_up fusion: VT_DENSE_MARLIN_GATEUP defaults ON, opt out with =0") {
  // Spec §3.3 asked for the default to move on a measured same-binary A/B, and
  // it did: interleaved 4-rep A/B on nvidia/Qwen3.6-27B-NVFP4@0893e160 (GB10),
  // toggle the only variable, gave fused +2.12% at c1 (12.0823 vs 11.8313) and
  // +1.70% at c8 (83.6186 vs 82.2217) with COMPLETE SEPARATION at both — every
  // fused rep beat every split rep — and a byte-identical 64-token greedy
  // continuation on both arms. So the lever now ships ON, per the standing
  // project rule that a parity enabler's default flips before a binding grid.
  //
  // UNSET must be ON, which is what a leftover opt-IN
  // `e && e[0]=='1' && e[1]=='\0'` spelling would break; only an explicit
  // leading '0' opts back out to the split pair.
  CHECK(dn::DenseMarlinGateUpEnabledFor(nullptr));
  CHECK(dn::DenseMarlinGateUpEnabledFor(""));
  CHECK(dn::DenseMarlinGateUpEnabledFor("1"));
  CHECK(dn::DenseMarlinGateUpEnabledFor("true"));
  CHECK(dn::DenseMarlinGateUpEnabledFor("10"));
  CHECK_FALSE(dn::DenseMarlinGateUpEnabledFor("0"));

  // The opt-out convention is the one the NEAREST parity levers use — the
  // sibling fused-gate_up toggles in the same header, VT_MARLIN_DENSE_PAIR
  // (MarlinDensePairEnabled) and VT_MOE_FUSED_W13 (FusedGateUpEnabled), both
  // `!(e != nullptr && e[0] == '0')`. That inspects the FIRST character only,
  // so a leading '0' opts out whatever follows it. Pinned so the choice is a
  // decision on record rather than an accident of spelling.
  CHECK_FALSE(dn::DenseMarlinGateUpEnabledFor("00"));
  CHECK_FALSE(dn::DenseMarlinGateUpEnabledFor("0x"));

  // The cached reader is that pure parser applied to the process environment —
  // it must not carry a second, differently-spelled default.
  CHECK(dn::DenseMarlinGateUpEnabled() ==
        dn::DenseMarlinGateUpEnabledFor(std::getenv("VT_DENSE_MARLIN_GATEUP")));
}

TEST_CASE("dense gate_up fusion: EVERY lever term in the composed guard is load-bearing") {
  // Fresh-review findings F1/F2. The composed guard ANDs three cached env
  // levers, the shape/format terms, and an op-registry probe. On a host build
  // the probe is hard-false, so the whole predicate answered false whatever the
  // levers said and deleting `DenseMarlinGateUpEnabled()`, `MarlinW4A16Enabled()`
  // or `FusedGateUpEnabled()` — or hardwiring / misspelling the cached reader —
  // all stayed green. The consequence is not cosmetic: VT_DENSE_MARLIN_GATEUP=0
  // silently becomes a no-op on a DEFAULT-ON lever, and the next same-binary A/B
  // compares fused against fused and reports a confident zero, which is exactly
  // the failure mode the row's own Outcome warns about.
  //
  // Two things make each term testable. `DenseMlpGateUpFusedMarlinEligibleWhen`
  // INJECTS the device answer, so the lever terms are reachable on CPU; and the
  // cached readers can only ever observe ONE value per process, so this case is
  // registered in ctest FOUR times (tests/CMakeLists.txt) — once with the
  // environment untouched and once with each lever forced to 0.
  const Nvfp4Weight gate = MakeW4A16(kN, kK, 0.5F);
  const Nvfp4Weight up = MakeW4A16(kN, kK, 0.5F);
  REQUIRE(dn::GateUpPairFusableShape(gate, up));

  const bool gateup_lever = LeverOn("VT_DENSE_MARLIN_GATEUP");
  const bool marlin_lever = LeverOn("VT_NVFP4_MARLIN");
  const bool fused_lever = LeverOn("VT_MOE_FUSED_W13");
  CAPTURE(gateup_lever);
  CAPTURE(marlin_lever);
  CAPTURE(fused_lever);

  // With the Marlin NVFP4 op available and a perfectly-shaped pair, the answer
  // is EXACTLY the conjunction of the three levers as the ENVIRONMENT spells
  // them. Any dropped term reads ON where the environment says OFF.
  CHECK(dn::DenseMlpGateUpFusedMarlinEligibleWhen(gate, up, true) ==
        (gateup_lever && marlin_lever && fused_lever));

  // ...and the op term is not decorative either: no realized kernel, no fusion,
  // whatever the levers say.
  CHECK_FALSE(dn::DenseMlpGateUpFusedMarlinEligibleWhen(gate, up, false));

  // The device overload adds nothing beyond the registry probe, so on a host
  // device it must agree with the injected-false answer.
  CHECK(dn::DenseMlpGateUpFusedMarlinEligible(gate, up, vt::DeviceType::kCPU) ==
        dn::DenseMlpGateUpFusedMarlinEligibleWhen(
            gate, up,
            vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin,
                             vt::DeviceType::kCPU)));

  // Each cached reader must be the pure parser applied to ITS OWN env name.
  CHECK(dn::DenseMarlinGateUpEnabled() == gateup_lever);
  CHECK(dn::MarlinW4A16Enabled() == marlin_lever);
  CHECK(dn::FusedGateUpEnabled() == fused_lever);
}

TEST_CASE("dense gate_up fusion: an MXFP4 pair is REFUSED by the DENSE guard") {
  // Fresh-review finding F3, latent. The shared shape term admits an MXFP4 pair
  // — correctly, because THIS header's own fused entry point (GateUpFusedMarlinD)
  // handles E8M0/group-32: it permutes the merged scales with
  // MarlinProcessExpertScalesMxfp4, skips the combined factor and the global, and
  // passes group_size/mxfp4 through to the GEMM args.
  //
  // The DENSE MLP call site does not reach that one. It reaches qwen3_5.cpp's
  // private NVFP4-ONLY copy (DenseGateUpFusedMarlinD:2683 ->
  // SharedGateUpFusedMarlinD:2606 -> BuildMarlinDensePairResident:2549), which
  // sizes the merged scale buffer at K/16 rows (:2556,:2558), runs the NVFP4
  // combined-scale-factor + global-scale processing, and hardcodes
  // `group_size = 16` / `mxfp4 = false` (:2632-2633). Handing it group-32 E8M0
  // scales is verbatim the defect this project already recorded RED-first for the
  // OTHER implementation at
  // tests/vllm/model_executor/layers/test_linear_method.cpp:185-201 — group-32
  // E8M0 misread as group-16 fp8-e4m3, "GROSSLY wrong".
  //
  // No dense loader sets `is_mxfp4` today (LoadNvfp4AnyNaming,
  // qwen3_5_dense_weights.cpp:359-396, only ever produces NVFP4), so this is one
  // loader line away rather than live. The guard refuses it so that line cannot
  // silently light up a mis-scaled kernel; the split pair, which routes MXFP4
  // correctly through MatmulMxfp4W4A16D, stays the answer.
  const Nvfp4Weight mx_gate = MakeMxfp4(kN, kK);
  const Nvfp4Weight mx_up = MakeMxfp4(kN, kK);

  // It passes the SHARED shape term (both halves agree on format and group),
  // which is what the header's own MXFP4-capable `GateUpFusedEligible` composes
  // — that predicate itself is CUDA-only (#ifdef VT_MARLIN_NVFP4) and so cannot
  // be named from a host build, but its shape half is exactly this.
  REQUIRE(dn::GateUpPairFusableShape(mx_gate, mx_up));

  // The DENSE guard refuses it anyway, even with the kernel available and every
  // lever ON. Deleting the `!is_mxfp4` terms turns this CHECK red.
  CHECK_FALSE(dn::DenseMlpGateUpFusedMarlinEligibleWhen(mx_gate, mx_up, true));
  CHECK_FALSE(dn::DenseMlpGateUpFusedMarlinEligible(mx_gate, mx_up,
                                                    vt::DeviceType::kCUDA));

  // A single MXFP4 half is refused too — that pair also fails the shape term, so
  // this pins the ORDER-independence of the refusal, not a second reason.
  const Nvfp4Weight nv = MakeW4A16(kN, kK, 0.5F);
  CHECK_FALSE(dn::DenseMlpGateUpFusedMarlinEligibleWhen(mx_gate, nv, true));
  CHECK_FALSE(dn::DenseMlpGateUpFusedMarlinEligibleWhen(nv, mx_up, true));
}

TEST_CASE("dense gate_up fusion: a backend with no Marlin NVFP4 op never selects it") {
  // The composed guard has no build-time gate of its own: `vt::OpRegistered` is
  // what makes a CPU/Vulkan/Metal device — and a build without VT_MARLIN_NVFP4 —
  // answer false. A perfectly-shaped pair must still be REFUSED here, because
  // there is no fused kernel to substitute; the split pair is unchanged.
  const Nvfp4Weight gate = MakeW4A16(kN, kK, 0.5F);
  const Nvfp4Weight up = MakeW4A16(kN, kK, 0.5F);
  REQUIRE(dn::GateUpPairFusableShape(gate, up));
  REQUIRE_FALSE(vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin,
                                 vt::DeviceType::kCPU));
  CHECK_FALSE(
      dn::DenseMlpGateUpFusedMarlinEligible(gate, up, vt::DeviceType::kCPU));
}

TEST_CASE(
    "dense gate_up fusion: a dense gate weight carries the SAME per-weight pair"
    " slot the shared expert keys on (the reuse itself is GPU-only)") {
  // RENAMED after the fresh review (finding F5). The previous name claimed this
  // case proved the dense path "reuses the EXISTING pair resident, not a new
  // cache", and it did not: it named no resident function at all, so it would
  // have passed byte-for-byte against a brand-new second cache or a global
  // address-keyed map — precisely what the title excluded.
  //
  // WHAT CANNOT BE CHECKED HERE, stated rather than implied. The reuse property
  // lives in `MarlinDensePairResidentFor` /`BuildMarlinDensePairResident` /
  // `SharedGateUpFusedMarlinD`, which are TU-private to
  // src/vllm/model_executor/models/qwen3_5.cpp AND compiled only with the
  // vendored Marlin NVFP4 kernel. There is no host-reachable declaration to call
  // and no CUDA in this build, so "the fused dense call lands in the shared
  // expert's existing resident" is a GPU-box observation (the row's decode-window
  // trace, which counted 129.000 Marlin calls/step with no extra repack).
  //
  // WHAT IS CHECKED HERE is the structural precondition that made reuse possible
  // without touching the cache: issue #237 moved residency OFF a static
  // address-keyed map and ONTO a `ResidentSlot` member of the weight, and
  // `MarlinDensePairResidentFor` keys on the GATE weight's `resident_marlin_pair`
  // slot. A dense MLP's `gate_proj_fp4` is the same `Nvfp4Weight` type as the
  // shared expert's `shared_gate_proj_fp4`, hence carries that same slot, hence
  // needs no second cache and no extension of the existing one.
  static_assert(std::is_same_v<decltype(DenseMlpWeights::gate_proj_fp4),
                               decltype(MoeBlockWeights::shared_gate_proj_fp4)>,
                "the dense gate weight must be the SAME type the fused pair "
                "resident is keyed on, or it cannot share that resident");
  static_assert(std::is_same_v<decltype(Nvfp4Weight::resident_marlin_pair),
                               vllm::ResidentSlot>,
                "the pair resident must live ON the weight (issue #237)");

  DenseMlpWeights w;
  CHECK(w.gate_proj_fp4.resident_marlin_pair.state == nullptr);
  CHECK(w.up_proj_fp4.resident_marlin_pair.state == nullptr);
  // The pair slot is distinct from the single-projection slot the SPLIT path
  // uses, so selecting one layout never marks the other ready.
  CHECK(w.gate_proj_fp4.resident_marlin.state == nullptr);

  w.gate_proj_fp4.resident_marlin_pair.state = std::make_shared<int>(1);
  CHECK(w.gate_proj_fp4.resident_marlin.state == nullptr);
  CHECK(w.up_proj_fp4.resident_marlin_pair.state == nullptr);

  // Per-INSTANCE, which is the half of #237 that a host build can still see: a
  // second dense MLP's slots are its own, so two layers (or two engines) never
  // inherit each other's device pointers the way the address-keyed map let them.
  DenseMlpWeights other;
  CHECK(other.gate_proj_fp4.resident_marlin_pair.state == nullptr);
  MoeBlockWeights moe;
  CHECK(moe.shared_gate_proj_fp4.resident_marlin_pair.state == nullptr);
}
