// #2623: `EnsureGemma4Fp8NativeOnDevice` must REFUSE the device upload on a
// device where the ops the FP8-NATIVE arm calls do not exist, instead of
// admitting and then throwing out of the middle of a decode step.
//
// THE TWIN, AND WHY IT IS A SEPARATE BINARY. `test_gemma4_moe_device_arm_guard`
// covers the BF16 device-expert arm. This file covers its FP8-native sibling,
// and the ONLY difference between the two runs is `VT_GEMMA4_FP8_NATIVE`: `0`
// there, `1` here. It cannot be a second TEST_CASE in that file, because
// `RunGemma4Moe` freezes that knob in a function-local `static const`
// (`gemma4_moe.cpp:1071-1077`) on the first call in the PROCESS. Two cases
// disagreeing about it in one executable would silently both measure whichever
// ran first.
//
// AND THIS IS THE DEFAULT ARM. `VT_GEMMA4_FP8_NATIVE` unset returns `true`, so a
// stock build takes the path this file guards and NOT the one the BF16 file
// guards. Setting it to "1" here reproduces the default rather than selecting an
// exotic mode; it is written explicitly so the precondition is visible in the log
// instead of being read out of a default.
//
// WHY THREE PREDICATES AND NOT THE SIBLING'S ONE. The BF16 guard keys on
// `vt::HasMatmulBTAlphaBeta`, the single op its promise commits the caller to.
// The native twin commits the caller to THREE: a `true` return routes into
// `ExpertGeGLUFp8TopKFusedGelu` (`gemma4_moe.cpp:134`), which calls
// `vt::MatmulBTFp8Channel` (`:165`, `:176`), and into `ExpertGeGLUFp8Native`
// (`:96`), which calls `vt::DequantFp8ChannelBf16` (`:118`, `:120`) and
// `vt::MatmulBTAlphaBeta` (`:129`). Each has exactly one arm in the tree, ROCm's,
// and each refuses on every other device (`fused_ops.cpp:177`, `:194`, `:152`).
// Reusing `HasMatmulBTAlphaBeta` alone would be a guard naming the wrong arm --
// `.agents/specs/expert-streaming.md`'s `## Owed` refuses that shape by name, and
// that is why this file asserts all three predicates rather than one.
//
// WHAT MAKES THIS A REACHABILITY GATE AND NOT A UNIT TEST. The guard is entered
// through `vllm::RunGemma4Moe` -- the Gemma-4 MoE layer entry point that
// `src/vllm/model_executor/models/gemma4.cpp:659` calls -- with a real
// `Gemma4MoeLayerWeights`, the real router, the real device-expert LRU and the
// real fallback. Nothing here constructs the LRU, the `Dev`, or any `vt::` op by
// hand. Delete the guard from `EnsureGemma4Fp8NativeOnDevice` and this file goes
// red, because the layer then throws
// `vt::MatmulBTFp8Channel: ROCm-only in this build` out of the decode step: the
// call site at `gemma4_moe.cpp:1461` takes the `true`, collects the four device
// pointers, and hands them to `ExpertGeGLUFp8TopKFusedGelu` after
// `vt::ExpertGeGLUFp8TopKM1` declines on a non-ROCm queue.
//
// WHY THE BACKEND IS DECORATED. The hazard is latent on every machine that can
// run this suite, for one reason: the LRU's `MakeRoom` refuses unless
// `vt::Backend::DeviceMemoryInfo` answers, and the CPU backend does not. It is
// NOT latent in production any more, and that is the change this file exists for
// -- `CudaBackend::DeviceMemoryInfo` now answers (`src/vt/cuda/cuda_backend.cu:93`,
// added by PERF-QWEN35-STAGE-WEIGHTS), so on a stock CUDA build every admission
// term passes by default and the upload reaches the throw. This test puts a CPU
// box in exactly that state: the registered CPU backend, wrapped, with
// `DeviceMemoryInfo` answering and nothing else altered. Every op still runs its
// real CPU kernel; none of the three FP8 ops has a CPU arm. That combination is
// the one the guard exists for, and it is the only combination under which
// deleting the guard is observable here.
//
// The decoration is a real `vt::Backend` registered through the public
// `vt::RegisterBackend`, restored in a destructor, and the test asserts BOTH
// directions of the probe (real backend: false; wrapped: true) before it
// concludes anything -- an instrument that silently failed to arm would otherwise
// report the guard as proven when the run never reached it.
//
// `vt::RegisterBackend` documents that all registration completes before
// `main()`, and this call is after it. What that contract buys is the absence of
// synchronization on the read path: the registry is a plain pointer store and
// lock-free reads thereafter. This binary is single-threaded across the swap --
// one doctest case, no queue with a worker, no other case touching the CPU
// backend -- so there is no concurrent reader to race, and the pointer is put
// back before the case returns. It is the narrowest way to reach the state the
// guard exists for; the alternative is a ROCm device, which no host running this
// suite has.
#include "vllm/model_executor/models/gemma4_moe.h"

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/fused_ops.h"
#include "vt/tensor.h"

namespace {

using vt::DType;

// Every knob `RunGemma4Moe` reads lands in a function-local `static const`, so
// the FIRST call in the process freezes it. Set them during static
// initialization rather than inside a TEST_CASE, so no ordering between cases
// can decide which arm the suite measured.
struct EnvOnce {
  EnvOnce() {
    // THE ARM UNDER TEST, and the DEFAULT one: unset already returns true
    // (`gemma4_moe.cpp:1071-1077`). Written explicitly so the precondition is in
    // the log rather than in a default. Its BF16 sibling, which sets this to
    // "0", is `test_gemma4_moe_device_arm_guard`.
    ::setenv("VT_GEMMA4_FP8_NATIVE", "1", 1);
    // A definite budget: the default is 2048 MiB, but reading it from a default
    // makes the precondition invisible in the log.
    ::setenv("VT_GEMMA4_EXPERT_VRAM_MB", "256", 1);
    // Everything else must stay off — each of these reroutes the expert loop.
    ::unsetenv("VT_GEMMA4_HOST_AXPY");
    ::unsetenv("VT_GEMMA4_BATCH_EXPERTS");
    ::unsetenv("VT_GEMMA4_CUSTOM_EXPERT");
    ::unsetenv("VT_GEMMA4_EXPERT_EVICT");
    ::unsetenv("VT_GEMMA4_PREFILL_BATCH_MOE");
    ::unsetenv("VT_GEMMA4_PROFILE");
  }
};
const EnvOnce g_env_once;

// The registered CPU backend with ONE behaviour changed: the optional device
// memory probe answers. Everything else forwards, so the allocations, copies and
// kernels under test are the production CPU ones.
class ProbingBackend final : public vt::Backend {
 public:
  explicit ProbingBackend(vt::Backend& inner) : inner_(inner) {}

  void* Alloc(size_t bytes) override { return inner_.Alloc(bytes); }
  void Free(void* p) override { inner_.Free(p); }
  void Memset(vt::Queue& q, void* p, int value, size_t bytes) override {
    inner_.Memset(q, p, value, bytes);
  }
  void Copy(vt::Queue& q, void* dst, const void* src, size_t bytes) override {
    inner_.Copy(q, dst, src, bytes);
  }
  vt::Queue CreateQueue() override { return inner_.CreateQueue(); }
  void DestroyQueue(vt::Queue& q) override { inner_.DestroyQueue(q); }
  void Synchronize(vt::Queue& q) override { inner_.Synchronize(q); }
  void FlushPending() override { inner_.FlushPending(); }
  bool UnifiedMemory() const override { return inner_.UnifiedMemory(); }
  bool DeviceMemoryIsHostAddressable() const override {
    return inner_.DeviceMemoryIsHostAddressable();
  }
  bool SupportsAuxStream() const override { return inner_.SupportsAuxStream(); }
  bool SupportsAsyncSampledTokenReadback() const override {
    return inner_.SupportsAsyncSampledTokenReadback();
  }
  bool SupportsCompressedConvState() const override {
    return inner_.SupportsCompressedConvState();
  }
  bool SupportsCompressedGdnState() const override {
    return inner_.SupportsCompressedGdnState();
  }
  bool SupportsGraphCapture() const override { return inner_.SupportsGraphCapture(); }
  int DeviceCapabilityMajor() const override { return inner_.DeviceCapabilityMajor(); }
  int DeviceCapabilityMinor() const override { return inner_.DeviceCapabilityMinor(); }

  // The one override. Numbers chosen well above `MakeRoom`'s 1.5 GiB headroom so
  // the admission decision turns on the guard and not on arithmetic.
  bool DeviceMemoryInfo(size_t* free_bytes, size_t* total_bytes) const override {
    if (free_bytes != nullptr) *free_bytes = 8ull << 30;
    if (total_bytes != nullptr) *total_bytes = 16ull << 30;
    return true;
  }

 private:
  vt::Backend& inner_;
};

struct BackendSwap {
  vt::Backend* prev;
  explicit BackendSwap(vt::Backend& replacement)
      : prev(&vt::GetBackend(vt::DeviceType::kCPU)) {
    vt::RegisterBackend(vt::DeviceType::kCPU, &replacement);
  }
  ~BackendSwap() { vt::RegisterBackend(vt::DeviceType::kCPU, prev); }
  BackendSwap(const BackendSwap&) = delete;
  BackendSwap& operator=(const BackendSwap&) = delete;
};

constexpr int64_t kE = 4;   // experts
constexpr int64_t kI = 16;  // expert intermediate
constexpr int64_t kH = 32;  // hidden
constexpr int kTopK = 2;

// Finite, well-formed F8_E4M3 codes. Random bytes would put the e4m3 NaN
// encodings (0x7F/0xFF) into the weights, which propagate through the GEMM and
// make every later comparison vacuous.
constexpr uint8_t kF8Codes[] = {0x38, 0x3C, 0x30, 0xB8, 0x34, 0x40, 0xB0, 0xBC};

uint8_t F8Code(uint64_t i) { return kF8Codes[i % (sizeof(kF8Codes) / sizeof(kF8Codes[0]))]; }

vllm::OwnedTensor MakeF8(int64_t n_rows, int64_t n_cols, uint64_t seed) {
  vllm::OwnedTensor t;
  t.dtype = DType::kI8;  // F8_E4M3 carried as I8, as the loader does
  t.rank = 2;
  t.shape[0] = n_rows;
  t.shape[1] = n_cols;
  std::vector<uint8_t> b(static_cast<size_t>(n_rows * n_cols));
  for (size_t i = 0; i < b.size(); ++i) b[i] = F8Code(seed + i);
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

vllm::OwnedTensor MakeBf16(const std::vector<int64_t>& shape, float base, float step) {
  vllm::OwnedTensor t;
  t.dtype = DType::kBF16;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= t.shape[i];
  }
  std::vector<uint8_t> b(static_cast<size_t>(n) * 2);
  auto* p = reinterpret_cast<uint16_t*>(b.data());
  for (int64_t i = 0; i < n; ++i)
    p[i] = vt::F32ToBF16(base + step * static_cast<float>(i % 7));
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

vllm::Gemma4MoeLayerWeights MakeMoeWeights() {
  vllm::Gemma4MoeLayerWeights moe;
  moe.enabled = true;
  moe.top_k = kTopK;
  moe.moe_intermediate = kI;
  // `nk = true` [E, H]: the orientation RunGemma4Moe's VT_CHECK requires.
  moe.router_proj = MakeBf16({kE, kH}, -0.25f, 0.1f);
  moe.router_proj.nk = true;

  auto& ex = moe.experts;
  ex.is_fp8 = true;
  ex.num_experts = kE;
  ex.intermediate = kI;
  ex.hidden = kH;
  ex.fp8.resize(static_cast<size_t>(kE));
  for (int64_t e = 0; e < kE; ++e) {
    auto& fex = ex.fp8[static_cast<size_t>(e)];
    const uint64_t s = static_cast<uint64_t>(e) * 131u;
    fex.gate_w = MakeF8(kI, kH, s + 1);
    fex.up_w = MakeF8(kI, kH, s + 2);
    fex.down_w = MakeF8(kH, kI, s + 3);
    fex.gate_s = MakeBf16({kI}, 0.05f, 0.01f);
    fex.up_s = MakeBf16({kI}, 0.04f, 0.01f);
    fex.down_s = MakeBf16({kH}, 0.03f, 0.01f);
  }
  return moe;
}

// One decode token through the real layer entry point. Returns the [1, H] bf16
// result copied out immediately: for T == 1 the layer hands back a view of a
// thread_local buffer that the NEXT call overwrites.
std::vector<uint16_t> RunOneToken(vt::Queue& q, const vllm::Gemma4MoeLayerWeights& moe,
                                  const std::vector<uint16_t>& x) {
  vt::Tensor xin = vt::Tensor::Contiguous(const_cast<uint16_t*>(x.data()), DType::kBF16,
                                          q.device, {1, kH});
  vllm::Gemma4MoeScratch out =
      vllm::RunGemma4Moe(q, moe, /*router_in=*/xin, /*expert_in=*/xin, /*T=*/1, kH,
                         /*rms_eps=*/1e-6f);
  std::vector<uint16_t> got(static_cast<size_t>(kH));
  std::memcpy(got.data(), out.tensor.data, got.size() * sizeof(uint16_t));
  return got;
}

bool AnyNonZero(const std::vector<uint16_t>& v) {
  for (const uint16_t h : v)
    if (h != 0) return true;
  return false;
}

}  // namespace

TEST_CASE("gemma4 moe: the FP8-native expert upload refuses where its three ops have no arm") {
  vt::Queue q;
  q.device.type = vt::DeviceType::kCPU;
  q.device.index = 0;

  // Precondition 1 — ALL THREE arms the FP8-native promise commits to are absent
  // for this queue. True on every build, HIP included: the only arms are kROCM's.
  // Asserting one of the three would let the other two be woken without this
  // file noticing, which is the drift the per-op predicates exist to stop.
  REQUIRE_FALSE(vt::HasMatmulBTFp8Channel(q));
  REQUIRE_FALSE(vt::HasDequantFp8ChannelBf16(q));
  REQUIRE_FALSE(vt::HasMatmulBTAlphaBeta(q));

  // Precondition 2 — the STOCK backend does not answer the memory probe, which
  // is why the hazard is latent and why the decoration below is needed at all.
  {
    size_t free_b = 1, total_b = 1;
    REQUIRE_FALSE(vt::GetBackend(vt::DeviceType::kCPU).DeviceMemoryInfo(&free_b, &total_b));
  }

  const vllm::Gemma4MoeLayerWeights moe = MakeMoeWeights();
  std::vector<uint16_t> x(static_cast<size_t>(kH));
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = vt::F32ToBF16(0.1f + 0.05f * static_cast<float>(i % 5));

  // Baseline: the layer on the STOCK backend, where `MakeRoom` refuses because
  // the memory probe cannot answer, so the expert loop already takes the host
  // fallback. This is the value a correct guard must reproduce exactly.
  const std::vector<uint16_t> want = RunOneToken(q, moe, x);
  REQUIRE(want.size() == static_cast<size_t>(kH));
  // A layer that produced all zeros would make the equality below vacuous.
  REQUIRE(AnyNonZero(want));

  // Now put this CPU box in the state a stock CUDA build is ALREADY in: the
  // memory probe answers (`cuda_backend.cu:93`), so the LRU's `MakeRoom` admits,
  // and only the guard stands between the upload and three ops that have no arm
  // for this device.
  std::vector<uint16_t> got;
  {
    vt::Backend& real = vt::GetBackend(vt::DeviceType::kCPU);
    ProbingBackend probing(real);
    const BackendSwap swap(probing);

    // Precondition 3 — the instrument is ARMED. Without this, a decoration that
    // failed to take would leave the run on the same refuse-on-unknown path as
    // the baseline and report the guard proven by a run that never reached it.
    size_t free_b = 0, total_b = 0;
    REQUIRE(vt::GetBackend(vt::DeviceType::kCPU).DeviceMemoryInfo(&free_b, &total_b));
    REQUIRE(free_b > (2ull << 30));

    // THE ASSERTION. With the guard: `EnsureGemma4Fp8NativeOnDevice` returns
    // false before its upload, the FP8-native branch at `gemma4_moe.cpp:1461`
    // sets `ok = false` and falls through, and the layer answers from the host
    // fallback. Without it: that branch collects the four device pointers,
    // `vt::ExpertGeGLUFp8TopKM1` declines on a non-ROCm queue, and
    // `ExpertGeGLUFp8TopKFusedGelu` reaches `vt::MatmulBTFp8Channel`, which
    // throws `ROCm-only in this build` out of the decode step.
    REQUIRE_NOTHROW(got = RunOneToken(q, moe, x));
  }

  // Degraded, but CORRECT: refusing the device arm must not change the answer.
  REQUIRE(got.size() == want.size());
  int same = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] == want[i]) ++same;
  }
  CHECK(same == static_cast<int>(want.size()));

  // The refusal must not have leaked device pointers into the expert records.
  // These are the FOUR slots `EnsureGemma4Fp8NativeOnDevice` assigns, and it
  // assigns them only on the `true` return, so a refusal leaves all four null.
  // Checking the BF16 pair here instead would pass even if the native upload had
  // run to completion, which is the arm this file is about.
  int null_dev = 0;
  for (const auto& fex : moe.experts.fp8) {
    if (fex.dev_fp8_gu == nullptr && fex.dev_fp8_dn == nullptr &&
        fex.dev_s_gu == nullptr && fex.dev_s_dn == nullptr)
      ++null_dev;
  }
  CHECK(null_dev == static_cast<int>(kE));

  // The backend registry is back to what it was.
  size_t free_b = 1, total_b = 1;
  CHECK_FALSE(vt::GetBackend(vt::DeviceType::kCPU).DeviceMemoryInfo(&free_b, &total_b));
}
