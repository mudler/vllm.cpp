// QUANT-CUDA-GATES W8 — the DEVICE-RESIDENT keep-quant MoE arm, gated against
// the reference arm it replaces at PER-PAIR BIT IDENTITY.
//
// Issue `.agents/issues/QUANT-CUDA-GATES/ISSUE-LOCAL-01M2AAACGC0SXYXFN2JQ3GFEAZ.md`,
// spec `.agents/specs/qwen4-exp-flash-next.md` §"W8 scope: a device-resident arm
// for the k-quant MoE block".
//
// ─── WHY A TOKEN GATE CANNOT SEE THIS, AND WHAT THIS SUITE DOES INSTEAD ──────
//
// The defect this row closes produced NO wrong tokens. `MoeBlock`'s reference
// arm computes the right answer; it just computes it by leaving the device four
// times per layer — copy the hidden out and synchronize, download the router
// top-k, `Download` an f32 result after each of three grouped GEMMs, run the
// SwiGLU in a host loop, and upload the routed output again. Every token gate
// in this tree passed through that for the whole life of the bug, which is
// exactly why it survived. So the assertion here is not about VALUES.
//
// It is about ARM SELECTION, and it has two halves that must BOTH hold:
//
//   1. `vllm::MoeKqDeviceCalls()` ADVANCES with `VT_MOE_KQ_FAST` unset or 1 and
//      does NOT advance with `VT_MOE_KQ_FAST=0`. This is the half a value
//      comparison cannot make: the two arms agree on every byte, so without a
//      selection probe a suite that ran the reference arm twice would be green.
//   2. The two arms' block outputs are BYTE-IDENTICAL — `std::memcmp`, not a
//      tolerance. A tolerance here would hide a routing or ordering defect,
//      which is the class of bug this seam has already produced (#2249 item 4,
//      the rank-3 tower that "matched" by shape). Bit identity is REACHABLE
//      because both arms reach the same `kMatmulBTQuant` core with the same
//      `eids` slice.
//
//   3. The ONE step that is not bit-identical by construction — `expf` against
//      `std::exp` inside the SwiGLU — is gated AT THE OP, over 2^22 pairs, and
//      NOT through the block. Half 2 has almost no power over that class: a
//      1-f32-ULP move in `silu(g)` changes `bf16(silu(g)*u)` in 1.44e-5 of
//      N(0,3) samples, so this fixture's 96 SwiGLU elements would miss a kernel
//      that disagreed on every input with probability ~0.9986. The sweep cases
//      at the foot of this file carry that measurement and their own bound.
//
// Deleting the arm-selection `if` in `MoeBlock` reddens half 1 and leaves half 2
// green. That is the mutation this file exists to survive, and it is what says
// the suite measures a capability rather than a class.
//
// THE `MoeSelFpCalls() == 0` TERM OF THE PREDICATE is gated by a SECOND ctest
// registration of this same binary with `VT_MOE_SEL_FP` armed
// (`test_qwen35_moe_kq_device_sel_fp`), under which every arm-selection
// expectation below flips to "the arm must not run". It cannot be a case: the
// tap count is cached in a process-static on first read, so a case that armed
// it would either come first and disarm the rest of the suite or come later and
// be ignored.
//
// ─── WHAT THIS SUITE RUNS THROUGH, AND WHAT IT DOES NOT REACH ────────────────
//
// It enters through `vllm::RunMoeBlock`, the production entry point that
// `qwen3_moe.cpp` and `qwen4_exp_moe.cpp` call — not through `MoeBlockKqDevice`,
// which has internal linkage and which nothing here can name. Deleting the call
// site therefore takes the capability away from this suite too.
//
// IT RUNS ON EVERY BACKEND THE ARM IS REGISTERED ON, not only CUDA — and that
// is a REQUIREMENT rather than a courtesy, because the arm asks the op table
// rather than naming a device, so it is DEFAULT-ON wherever the three ops are
// registered. CPU, CUDA and ROCm each get a case here; a box that lacks one
// prints a LOUD skip for it and still gates the rest. Every op the arm calls
// has a CPU provider and the CPU grouped kernel implements the same broadcast
// (`cpu_quant_gemm.cpp:244`), so the CPU case gates arm selection on a box with
// no GPU at all, while the device cases gate the residency that is the point of
// the row.
//
// TENSTORRENT REGISTERS ALL THREE OPS TOO (`tenstorrent_ops.cpp:8298, 8307,
// 8309`) and is NOT gated here: this suite has no Tenstorrent case and the fleet
// has no device to run one on. That is tracked by
// `.agents/issues/QUANT-CUDA-GATES/ISSUE-LOCAL-01M2D7X94RTPJ453QGHYKQZPY8.md`.
// The bf16 precedent arm shares the predicate SHAPE but not that exposure —
// Tenstorrent registers no `kMoeGroupedGemmBf16` (`grep kMoeGroupedGemmBf16
// src/vt/tenstorrent/` is empty). That `grep` is the whole of the argument.
// `test_rocm_moe_bf16.cpp` is NOT a second leg of it, and an earlier draft of
// this comment wrongly said it was: that suite never names `MoeBlock` or
// `RunMoeBlock`, so it gates no bf16 arm predicate — its `kMoeGroupedGemmBf16`
// lines (`:400-401`) are op-table registration checks — and they sit inside a
// `doctest::skip(FixtureAbsent())` case (`:303-304`) that is inert without
// `VT_ROCM_MOE_FIXTURE` and `VT_ROCM_MOE_ORACLE`.
//
// THE FIXTURE IS Q8_0 AND THAT IS A LIMIT, stated rather than implied. The
// towers are hand-written Q8_0 blocks (block scale `2^-4`, exact in f16, and the
// code itself as `qs`) so the encoding is a fact and not another arm's output.
// Q8_0 reaches `MatmulQ8_0GroupedCuda`, one of the three device lanes of
// `MatmulBTQuantGrouped`; the 32-block lane (IQ4_NL / Q5_0) and the 256-block
// Q8_K lane (the Q*_K family) are the other two. The arm is dtype-blind — it
// passes the tower straight to the same op the reference arm passes it to — so
// what those lanes need proven is the op's own contract and not this arm's. No
// k-quant tower is executed here, and nothing below should be read as if one
// were.
//
// T == 1 IS THE ARM'S SCOPE, NOT THE FIXTURE'S CONVENIENCE. The arm feeds the
// grouped GEMM the single hidden row through the broadcast the kernel already
// implements (`Pa == 1 && P > 1`), which is what removes the host gather.
// Prefill has no row-map argument on this op and keeps the reference loop; the
// `T > 1` case below asserts that refusal so the scope is gated rather than
// documented.

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_moe_block.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm {
// The W8 arm-selection probe. Defined beside `RunMoeBlock` in qwen3_5.cpp (the
// arm itself has internal linkage), and declared here rather than in the seam
// header because this row's authority does not extend to that header.
int64_t MoeKqDeviceCalls();
}  // namespace vllm

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using vllm::MoeBlockWeights;
using vllm::OwnedTensor;

// E = 6 with top_k = 3 is the smallest pair that carries a NON-TRIVIAL expert
// permutation: more than one expert per token and at least one expert never
// selected, so a wrong expert index cannot return the right answer.
constexpr int64_t kH = 64;  // whole q8_0 blocks: gate/up reduce over K = H
constexpr int64_t kE = 6;
constexpr int64_t kTopK = 3;
constexpr int64_t kI = 32;  // whole q8_0 blocks: down reduces over K = I

constexpr float kWU = 0.0625F;      // 2^-4, exact in f16 and in the q8_0 scale
constexpr float kAU = 0.00390625F;  // 2^-8
constexpr int kWMax = 8;
constexpr int kAMax = 127;  // exactly the q8_0 activation range

struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  int Next(int lo, int hi) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint64_t r = (s >> 33) % static_cast<uint64_t>(hi - lo + 1);
    return lo + static_cast<int>(r);
  }
};

std::vector<int> Codes(size_t n, uint64_t seed, int lim) {
  Lcg g(seed);
  std::vector<int> c(n);
  for (size_t i = 0; i < n; ++i) c[i] = g.Next(-lim, lim);
  return c;
}

bool HasBackend(DeviceType t) {
  try {
    vt::GetBackend(t);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

// LOUD, because a silent skip on a CPU box is how a device arm goes un-gated for
// a release.
bool SkipNoBackend(DeviceType t, const char* what) {
  if (HasBackend(t)) return false;
  std::printf("[SKIP] no %s backend: %s NOT exercised\n", vt::DeviceTypeName(t), what);
  return true;
}
bool SkipNoCuda(const char* what) { return SkipNoBackend(DeviceType::kCUDA, what); }
bool SkipNoRocm(const char* what) { return SkipNoBackend(DeviceType::kROCM, what); }

// The VT_MOE_SEL_FP tap, read the same way `MoeSelFpCalls()` reads it: a
// positive integer arms it. The arm REFUSES to run while it is armed, and this
// is how the suite knows which expectation to hold.
//
// `MoeSelFpCalls()` caches its read in a process-static, so the tap cannot be
// armed from inside a case that runs after any other case has already asked.
// That is why this is read from the PROCESS environment and why the tap arm is
// a second ctest registration of this same binary rather than a fifth case.
bool SelFpArmed() {
  const char* e = std::getenv("VT_MOE_SEL_FP");
  return e != nullptr && e[0] != '\0' && std::atoll(e) > 0;
}

Device Dev(DeviceType t) { return Device{t, 0}; }

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

OwnedTensor Make(DType dt, const std::vector<int64_t>& shape, size_t bytes, bool nk) {
  OwnedTensor o;
  o.dtype = dt;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) o.shape[i] = shape[i];
  o.bytes.resize(bytes);
  return o;
}

OwnedTensor Bf16Tensor(const std::vector<int>& codes, const std::vector<int64_t>& shape) {
  OwnedTensor o = Make(DType::kBF16, shape, codes.size() * sizeof(uint16_t), /*nk=*/false);
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (size_t i = 0; i < codes.size(); ++i)
    p[i] = vt::F32ToBF16(static_cast<float>(codes[i]) * kWU);
  return o;
}

// One q8_0 tower: block scale d = 2^-4 (exact in f16) and qs = the code itself,
// so `d * qs[i]` reproduces `code * kWU` bit for bit. Written by hand rather
// than quantized, so the encoding is an input to the suite and not another arm's
// output.
OwnedTensor Q8_0Tensor(const std::vector<int>& codes, const std::vector<int64_t>& shape,
                       int64_t rows, int64_t k) {
  REQUIRE(k % 32 == 0);
  REQUIRE(static_cast<int64_t>(codes.size()) == rows * k);
  const int64_t nblk = k / 32;
  const size_t row_bytes = static_cast<size_t>(nblk) * 34;
  OwnedTensor o = Make(DType::kQ8_0, shape, static_cast<size_t>(rows) * row_bytes, /*nk=*/false);
  uint8_t* p = o.bytes.data();
  const uint16_t d = vt::F32ToF16(kWU);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < nblk; ++b) {
      uint8_t* blk = p + static_cast<size_t>(r) * row_bytes + static_cast<size_t>(b) * 34;
      std::memcpy(blk, &d, sizeof(uint16_t));
      for (int64_t j = 0; j < 32; ++j) {
        blk[2 + j] = static_cast<uint8_t>(
            static_cast<int8_t>(codes[static_cast<size_t>(r * k + b * 32 + j)]));
      }
    }
  }
  return o;
}

vllm::HfConfig Config() {
  vllm::HfConfig c;
  c.hidden_size = kH;
  c.num_experts = kE;
  c.num_experts_per_tok = kTopK;
  c.moe_intermediate_size = kI;
  // 0 skips the shared expert on BOTH arms (it is the same `SharedExpert` call
  // on each), so what this suite compares is the ROUTED path the arm changes.
  c.shared_expert_intermediate_size = 0;
  return c;
}

// A keep-quant weight set in exactly the shape the GGUF loader produces: a bf16
// [H,E] router (`nk == false`, the `LoadExpertsT` orientation) and rank-2
// stacked `_kq` towers. Each fixture is built fresh per device, because
// `ResidentWeight` caches a device staging INSIDE the OwnedTensor.
MoeBlockWeights BuildFixture() {
  MoeBlockWeights w;
  w.router_gate = Bf16Tensor(Codes(static_cast<size_t>(kH * kE), 1656, kWMax), {kH, kE});
  w.expert_gate_kq = Q8_0Tensor(Codes(static_cast<size_t>(kE * kI * kH), 13, kWMax),
                                {kE * kI, kH}, kE * kI, kH);
  w.expert_up_kq = Q8_0Tensor(Codes(static_cast<size_t>(kE * kI * kH), 14, kWMax),
                              {kE * kI, kH}, kE * kI, kH);
  w.expert_down_kq = Q8_0Tensor(Codes(static_cast<size_t>(kE * kH * kI), 15, kWMax),
                                {kE * kH, kI}, kE * kH, kI);
  return w;
}

// Activation codes with |m| = 127 forced at element 0 of every row, so the q8_0
// activation quantizer picks d = kAU exactly and recovers every code.
std::vector<uint16_t> Hidden(int64_t T) {
  std::vector<int> c = Codes(static_cast<size_t>(T * kH), 19, kAMax);
  for (int64_t t = 0; t < T; ++t)
    c[static_cast<size_t>(t * kH)] = (t % 2 == 0) ? kAMax : -kAMax;
  std::vector<uint16_t> x(c.size());
  for (size_t i = 0; i < c.size(); ++i)
    x[i] = vt::F32ToBF16(static_cast<float>(c[i]) * kAU);
  return x;
}

// Run one MoE block through the PRODUCTION entry point and return the raw bf16
// bytes of the [T,H] output.
std::vector<uint16_t> RunBlock(Backend& b, Queue& q, DeviceType dt,
                               const MoeBlockWeights& w, const vllm::HfConfig& cfg,
                               const std::vector<uint16_t>& x, int64_t T) {
  const size_t bytes = x.size() * sizeof(uint16_t);
  void* p = b.Alloc(bytes);
  b.Copy(q, p, x.data(), bytes);
  Tensor dh = MakeT(p, DType::kBF16, Dev(dt), {T, kH});
  std::vector<uint16_t> out(static_cast<size_t>(T) * kH);
  try {
    vllm::MoeBlockOutput r = vllm::RunMoeBlock(q, w, cfg, dh, T);
    b.Copy(q, out.data(), r.tensor.data, out.size() * sizeof(uint16_t));
    b.Synchronize(q);
  } catch (...) {
    b.Free(p);
    throw;
  }
  b.Free(p);
  return out;
}

void SetArm(const char* value) {
  if (value == nullptr)
    ::unsetenv("VT_MOE_KQ_FAST");
  else
    ::setenv("VT_MOE_KQ_FAST", value, 1);
}

// THE GATE BODY, run once per backend the arm is registered on. `what` is a
// `std::string` and NOT a `const char*`: doctest stringifies a bare `char*`
// through its BOOL overload, so every `MESSAGE`/`CHECK_MESSAGE` naming the
// backend would print `1` and say nothing at all. Measured on the first run of
// this suite, and already recorded in `test_qwen4_exp_moe.cpp:RouteName`.
void GateArmOn(DeviceType dt, const std::string& what) {
  Backend& b = vt::GetBackend(dt);
  QueueGuard qg(b);
  MoeBlockWeights w = BuildFixture();
  const vllm::HfConfig cfg = Config();
  const std::vector<uint16_t> x = Hidden(/*T=*/1);
  const bool tap = SelFpArmed();
  MESSAGE("backend under test: " << what
                                 << (tap ? std::string(" [VT_MOE_SEL_FP ARMED]") : std::string()));

  // WITH THE TAP ARMED THE ARM MUST NEVER BE SELECTED, and that expectation is
  // the whole of this file's answer to `MoeSelFpCalls() == 0`. The term is
  // load-bearing: `MoeSelFp` is called only inside the reference path, so an
  // operator who arms `VT_MOE_SEL_FP` and gets the fast arm gets a SILENTLY
  // EMPTY tap rather than a diagnostic. Deleting the term leaves every
  // unarmed run green, so the arming is registered as its own ctest entry
  // (`test_qwen35_moe_kq_device_sel_fp`) and every expectation below flips.
  const int64_t expect = tap ? 0 : 1;

  // Arm A: the reference loop, forced. This is the correctness oracle, and the
  // counter must NOT move.
  SetArm("0");
  const int64_t before_ref = vllm::MoeKqDeviceCalls();
  const std::vector<uint16_t> ref = RunBlock(b, qg.q, dt, w, cfg, x, 1);
  CHECK_MESSAGE(vllm::MoeKqDeviceCalls() == before_ref,
                "VT_MOE_KQ_FAST=0 must take the reference loop, but the device arm ran "
                    << (vllm::MoeKqDeviceCalls() - before_ref) << " time(s) on " << what);

  // Arm B: the device-resident arm, forced. The counter MUST move by exactly
  // one — one call to `RunMoeBlock` is one MoE block — UNLESS the tap is armed,
  // in which case it must not move at all.
  SetArm("1");
  const int64_t before_fast = vllm::MoeKqDeviceCalls();
  const std::vector<uint16_t> fast = RunBlock(b, qg.q, dt, w, cfg, x, 1);
  CHECK_MESSAGE(vllm::MoeKqDeviceCalls() == before_fast + expect,
                "VT_MOE_KQ_FAST=1 must take the new arm exactly "
                    << expect << " time(s) on " << what
                    << (tap ? std::string(" because VT_MOE_SEL_FP is armed") : std::string())
                    << ", but the counter moved by "
                    << (vllm::MoeKqDeviceCalls() - before_fast));

  // And the DEFAULT — unset — is the new arm, because this ships ON.
  SetArm(nullptr);
  const int64_t before_dflt = vllm::MoeKqDeviceCalls();
  const std::vector<uint16_t> dflt = RunBlock(b, qg.q, dt, w, cfg, x, 1);
  CHECK_MESSAGE(vllm::MoeKqDeviceCalls() == before_dflt + expect,
                "VT_MOE_KQ_FAST unset must default to the new arm on "
                    << what
                    << (tap ? std::string(" EXCEPT while VT_MOE_SEL_FP is armed") : std::string()));

  // PER-PAIR BIT IDENTITY. memcmp, not a tolerance.
  REQUIRE(ref.size() == fast.size());
  const bool same = std::memcmp(ref.data(), fast.data(), ref.size() * sizeof(uint16_t)) == 0;
  if (!same) {
    size_t worst = 0;
    int64_t n_diff = 0;
    for (size_t i = 0; i < ref.size(); ++i)
      if (ref[i] != fast[i]) {
        if (n_diff == 0) worst = i;
        ++n_diff;
      }
    MESSAGE("first differing element " << worst << " ref=" << ref[worst]
                                       << " new=" << fast[worst] << ", differing elements "
                                       << n_diff << " of " << ref.size());
  }
  CHECK_MESSAGE(same, "the new arm must be BYTE-IDENTICAL to the reference arm on " << what);
  CHECK(std::memcmp(dflt.data(), fast.data(), dflt.size() * sizeof(uint16_t)) == 0);

  // The output is not trivially zero, so byte equality is a statement about real
  // work. A zeroed block would make every comparison above vacuous.
  bool nonzero = false;
  for (uint16_t v : ref) nonzero = nonzero || (v != 0);
  CHECK_MESSAGE(nonzero, "the reference block produced an all-zero output on " << what);
}

// The arm's T == 1 scope, gated rather than documented.
void GatePrefillRefusal(DeviceType dt, const std::string& what) {
  Backend& b = vt::GetBackend(dt);
  QueueGuard qg(b);
  MoeBlockWeights w = BuildFixture();
  const vllm::HfConfig cfg = Config();
  SetArm("1");
  const std::vector<uint16_t> x = Hidden(/*T=*/4);
  const int64_t before = vllm::MoeKqDeviceCalls();
  const std::vector<uint16_t> out = RunBlock(b, qg.q, dt, w, cfg, x, 4);
  CHECK_MESSAGE(vllm::MoeKqDeviceCalls() == before,
                "T>1 must take the reference loop on " << what << ", but the new arm ran");
  CHECK(out.size() == static_cast<size_t>(4) * kH);
  SetArm(nullptr);
}

// ─── THE ONE STEP THAT IS NOT BIT-IDENTICAL BY CONSTRUCTION ──────────────────
//
// Every other step of the arm reaches the SAME op on the SAME bytes as the
// reference step it replaces. The SwiGLU does not: the reference computes
// `Silu()` with the host `std::exp` and the arm computes it with the device
// `vt::MoeSiluMul`, whose CUDA and ROCm kernels use `expf`. The two are not
// required to agree to the last f32 ULP.
//
// THE BLOCK COMPARISON CANNOT MEASURE THAT, AND THIS IS WHY. A 1-f32-ULP
// perturbation of `silu(g)` changes `bf16(silu(g)*u)` in 288 of 20,000,000
// samples of N(0,3) — 1.44e-5 — measured directly rather than reasoned about.
// The block fixture carries `P * kI = 3 * 32 = 96` SwiGLU elements, so it would
// pass with probability ~0.9986 even against a kernel that disagreed on EVERY
// input, and that is before the down GEMM's q8_0 activation quantizer swallows
// most of what does survive. A 96-element fixture is not a measurement of this
// class; it is a coin that almost never lands.
//
// THE PRODUCTION CONVERSION POINTS THE OTHER WAY, AND IT BELONGS BESIDE THE ONE
// ABOVE. Qwen3.8-Flash-Next runs a MoE block on each of 48 layers, routed at
// top-10 into experts of `moe_intermediate_size = 640`, so ONE decode token puts
// `48 * 10 * 640 = 307,200` elements through `vt::MoeSiluMul`. At the measured
// device rate of 2.146e-06 that is 0.66 perturbed elements PER TOKEN, each up to
// one bf16 ULP — roughly half of all decode tokens carry at least one. The
// deviation is routine in production even though the fixture almost never sees
// it, which is the whole reason this sweep exists.
//
// THE DEVICE `expf` IS THE ORACLE'S OWN SPELLING, so the deviation is measured
// against OUR host reference rather than against correctness. vLLM's
// `silu_kernel` is `return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));`
// (`csrc/libtorch_stable/activation_kernels.cu:158`, anchored in this tree at
// `.agents/specs/vt-act-round-polarity.md:90`) — the same device `expf` that
// `MoeSiluMulK` calls (`src/vt/rocm/rocm_moe_router.hip:24`, quoting that
// upstream line at `:44-45`). The reference arm's host `std::exp` is the local
// artifact here, so this arm moves TOWARD the oracle at this step.
//
// So the step is gated DIRECTLY, at the op, over a sweep large enough to have
// power, against the same host formula the reference arm runs. The CPU case
// must be EXACTLY equal (`cpu_ops.cpp:733` computes `g/(1.0f+std::exp(-g))`
// and `RoundThrough` is a no-op on an f32 gate, so it is the host `Silu` and
// disagreement there would be a defect, not a deviation). The device cases
// report the OBSERVED disagreement count and hold it to one bf16 ULP, which is
// the bar this row can honestly state.
constexpr int64_t kSweepN = 1 << 22;  // 4,194,304 pairs

// THE CEILING ON THE DISAGREEMENT *RATE*, WHICH IS A SECOND ASSERTION AND NOT A
// RESTATEMENT OF THE ULP BOUND. `worst_ulp <= 1` bounds the MAGNITUDE of a
// disagreement and says nothing about how many elements disagree. The two are
// independent, and both halves of that independence have been demonstrated on
// this sweep:
//
//   * MEASURED: with `cuda_moe.cu`'s `MoeSiluMulKernel` multiplied by
//     `1.001953125f` (1 + 2^-9, half a bf16 ULP) the CUDA sweep reports
//     `disagreements=1510125 (3.600e-01) worst=1 bf16 ulp` on `thor:gpu0`, so
//     `worst_ulp <= 1` PASSES while 36.0% of the op's outputs have moved;
//   * the scoped re-review that found this gap also reports a kernel
//     disagreeing on 27.75% by up to 16 ULP that left the BLOCK byte comparison
//     green, so only a sweep assertion convicts that class at all.
//
// The record's guarantee is "bit-identical except at the SwiGLU, bounded at one
// bf16 ULP". Without a rate ceiling the gate holds only the second half of that
// sentence, which is how a rewritten SwiGLU could land unnoticed.
//
// HEADROOM, CHOSEN DELIBERATELY. The measured device rate is 9 of 4,194,304 on
// `thor:gpu0` (2.146e-06, sm_110 / CUDA 13.0.88). The ceiling is 1/1024 of the
// sweep — 4,096 elements, 9.77e-04 — which is ~455x the measurement and ~284x
// BELOW the smaller of the two disagreement rates above.
// The headroom is deliberately large in the permissive direction because this
// sweep runs on THREE backends and only CUDA has been measured: ROCm's `expf`
// and a future CUDA libdevice are entitled to round the last bit differently on
// more inputs than nine. It is not unbounded, because a `silu` that disagrees
// with `x / (1 + exp(-x))` on more than one element in a thousand is not the
// same formula rounded differently — it is a different function, and this suite
// should say so. If a real backend ever lands between 9.77e-04 and 27.75%, the
// answer is to MEASURE it and move this constant with the measurement recorded
// beside it, never to widen it to make a red run green.
constexpr int64_t kSweepMaxDiff = kSweepN / 1024;  // 4,096 elements = 9.77e-04

float HostSilu(float x) { return x / (1.0F + std::exp(-x)); }

// bf16 under the sign-magnitude TOTAL ORDER, as a signed key, so that a
// difference can be reported in ULP rather than only as "unequal".
int32_t Bf16Key(uint16_t v) {
  const int32_t mag = static_cast<int32_t>(v & 0x7FFFU);
  return (v & 0x8000U) != 0 ? -mag : mag;
}

// The sweep inputs. Two bands, and both are deliberate: the LINEAR band is the
// range the arm's own grouped-GEMM partials live in, and the LOG band spans
// 2^-14 to 2^6 so the measurement is a statement about the kernel rather than
// about one fixture's scale. Deterministic — the same `Lcg` the fixture uses,
// so the run is reproducible on every box.
void SweepInputs(std::vector<float>* g, std::vector<float>* u) {
  g->resize(static_cast<size_t>(kSweepN));
  u->resize(static_cast<size_t>(kSweepN));
  Lcg r(20260913);
  for (int64_t i = 0; i < kSweepN; ++i) {
    const float a = static_cast<float>(r.Next(0, 1 << 20)) / static_cast<float>(1 << 20);
    const float bq = static_cast<float>(r.Next(0, 1 << 20)) / static_cast<float>(1 << 20);
    const float sg = r.Next(0, 1) == 0 ? -1.0F : 1.0F;
    const float su = r.Next(0, 1) == 0 ? -1.0F : 1.0F;
    if ((i & 1) == 0) {  // linear band, +/- 20
      (*g)[static_cast<size_t>(i)] = (a * 2.0F - 1.0F) * 20.0F;
      (*u)[static_cast<size_t>(i)] = (bq * 2.0F - 1.0F) * 20.0F;
    } else {  // log band, 2^-14 .. 2^6
      (*g)[static_cast<size_t>(i)] = sg * std::exp2(-14.0F + 20.0F * a);
      (*u)[static_cast<size_t>(i)] = su * std::exp2(-14.0F + 20.0F * bq);
    }
  }
}

// `vt::MoeSiluMul` on `dt` against the reference arm's own host arithmetic, in
// the arm's exact dtype slots: f32 gate, f32 up, bf16 out.
void GateSwiGluIdentity(DeviceType dt, const std::string& what, bool require_exact) {
  Backend& b = vt::GetBackend(dt);
  QueueGuard qg(b);
  std::vector<float> hg, hu;
  SweepInputs(&hg, &hu);
  std::vector<uint16_t> want(static_cast<size_t>(kSweepN));
  for (int64_t i = 0; i < kSweepN; ++i) {
    const size_t k = static_cast<size_t>(i);
    want[k] = vt::F32ToBF16(HostSilu(hg[k]) * hu[k]);
  }

  const size_t fbytes = static_cast<size_t>(kSweepN) * sizeof(float);
  const size_t obytes = static_cast<size_t>(kSweepN) * sizeof(uint16_t);
  void* pg = b.Alloc(fbytes);
  void* pu = b.Alloc(fbytes);
  void* po = b.Alloc(obytes);
  std::vector<uint16_t> got(static_cast<size_t>(kSweepN));
  try {
    b.Copy(qg.q, pg, hg.data(), fbytes);
    b.Copy(qg.q, pu, hu.data(), fbytes);
    Tensor tg = MakeT(pg, DType::kF32, Dev(dt), {kSweepN});
    Tensor tu = MakeT(pu, DType::kF32, Dev(dt), {kSweepN});
    Tensor to = MakeT(po, DType::kBF16, Dev(dt), {kSweepN});
    vt::MoeSiluMul(qg.q, to, tg, tu);
    b.Copy(qg.q, got.data(), po, obytes);
    b.Synchronize(qg.q);
  } catch (...) {
    b.Free(pg);
    b.Free(pu);
    b.Free(po);
    throw;
  }
  b.Free(pg);
  b.Free(pu);
  b.Free(po);

  int64_t ndiff = 0, worst_ulp = 0;
  size_t first = 0;
  for (int64_t i = 0; i < kSweepN; ++i) {
    const size_t k = static_cast<size_t>(i);
    if (got[k] == want[k]) continue;
    const int64_t ulp = std::llabs(static_cast<int64_t>(Bf16Key(got[k])) -
                                   static_cast<int64_t>(Bf16Key(want[k])));
    if (ndiff == 0) first = k;
    ++ndiff;
    if (ulp > worst_ulp) worst_ulp = ulp;
  }
  // PRINTED ON THE GREEN RUN TOO. The count is the evidence this row owes; a
  // number that only appears on a failure is not a measurement.
  std::printf("[swiglu] %s: n=%lld disagreements=%lld (%.3e) worst=%lld bf16 ulp\n",
              what.c_str(), static_cast<long long>(kSweepN), static_cast<long long>(ndiff),
              static_cast<double>(ndiff) / static_cast<double>(kSweepN),
              static_cast<long long>(worst_ulp));
  if (ndiff > 0)
    MESSAGE("first differing element " << first << " g=" << hg[first] << " u=" << hu[first]
                                       << " want=" << want[first] << " got=" << got[first]);
  if (require_exact) {
    CHECK_MESSAGE(ndiff == 0,
                  "the host SwiGLU and " << what
                                         << " must be EXACTLY equal (same std::exp formula), but "
                                         << ndiff << " of " << kSweepN << " elements differ");
  } else {
    CHECK_MESSAGE(worst_ulp <= 1, "the " << what << " SwiGLU must stay within ONE bf16 ULP of the "
                                                    "reference arm's host formula, but one element "
                                                    "is off by "
                                         << worst_ulp << " ULP");
    // THE RATE, WHICH THE ULP BOUND ABOVE CANNOT SEE. See `kSweepMaxDiff`.
    CHECK_MESSAGE(ndiff <= kSweepMaxDiff,
                  "the " << what << " SwiGLU must disagree with the reference arm's host formula on "
                                    "a VANISHING fraction of inputs, but "
                         << ndiff << " of " << kSweepN << " elements differ ("
                         << static_cast<double>(ndiff) / static_cast<double>(kSweepN)
                         << "), above the ceiling of " << kSweepMaxDiff
                         << "; a last-bit rounding difference is rare, and a rate this high is a "
                            "different formula rather than a different rounding");
  }
  // The sweep must not be vacuous: a kernel that wrote zeros would agree with
  // nothing, but a HOST side that produced zeros would make the comparison
  // meaningless in the other direction.
  int64_t nonzero = 0;
  for (int64_t i = 0; i < kSweepN; ++i) nonzero += want[static_cast<size_t>(i)] != 0 ? 1 : 0;
  CHECK(nonzero > kSweepN / 2);
}

}  // namespace

// ─── THE GATE ────────────────────────────────────────────────────────────────
TEST_CASE("Qwen3.5 MoE keep-quant arm: SELECTED and byte-identical (CPU)") {
  GateArmOn(DeviceType::kCPU, "CPU");
}

TEST_CASE("Qwen3.5 MoE keep-quant arm: SELECTED and byte-identical (CUDA)") {
  if (SkipNoCuda("the W8 device-resident keep-quant MoE arm")) return;
  GateArmOn(DeviceType::kCUDA, "CUDA");
}

// ─── THE SCOPE, GATED RATHER THAN DOCUMENTED ─────────────────────────────────
TEST_CASE("Qwen3.5 MoE keep-quant arm: prefill (T>1) keeps the reference loop (CPU)") {
  GatePrefillRefusal(DeviceType::kCPU, "CPU");
}

TEST_CASE("Qwen3.5 MoE keep-quant arm: prefill (T>1) keeps the reference loop (CUDA)") {
  if (SkipNoCuda("the W8 arm's T==1 scope")) return;
  GatePrefillRefusal(DeviceType::kCUDA, "CUDA");
}

// ─── ROCm, BECAUSE THE PREDICATE ASKS THE OP TABLE AND ROCm ANSWERS YES ──────
// `kMatmulBTQuantGrouped`, `kMoeSiluMul` and `kCastBf16` are all registered on
// ROCm (`rocm_ops.hip:229, 290, 317`), so the arm is DEFAULT-ON there, and
// `gfx1151` is a live gated target. `MoeSiluMulK` (`rocm_moe_router.hip:48`)
// uses the device `expf`, which is F1's class again — so ROCm gets both the arm
// gate and the SwiGLU sweep, not one of them.
TEST_CASE("Qwen3.5 MoE keep-quant arm: SELECTED and byte-identical (ROCm)") {
  if (SkipNoRocm("the W8 device-resident keep-quant MoE arm")) return;
  GateArmOn(DeviceType::kROCM, "ROCm");
}

TEST_CASE("Qwen3.5 MoE keep-quant arm: prefill (T>1) keeps the reference loop (ROCm)") {
  if (SkipNoRocm("the W8 arm's T==1 scope")) return;
  GatePrefillRefusal(DeviceType::kROCM, "ROCm");
}

// ─── THE SwiGLU STEP, MEASURED AT THE OP RATHER THAN THROUGH THE BLOCK ───────
TEST_CASE("Qwen3.5 MoE keep-quant arm: SwiGLU is the host formula exactly (CPU)") {
  GateSwiGluIdentity(DeviceType::kCPU, "CPU", /*require_exact=*/true);
}

TEST_CASE("Qwen3.5 MoE keep-quant arm: SwiGLU expf deviation, bounded (CUDA)") {
  if (SkipNoCuda("the W8 arm's SwiGLU expf-vs-std::exp bound")) return;
  GateSwiGluIdentity(DeviceType::kCUDA, "CUDA", /*require_exact=*/false);
}

TEST_CASE("Qwen3.5 MoE keep-quant arm: SwiGLU expf deviation, bounded (ROCm)") {
  if (SkipNoRocm("the W8 arm's SwiGLU expf-vs-std::exp bound")) return;
  GateSwiGluIdentity(DeviceType::kROCM, "ROCm", /*require_exact=*/false);
}
