// #1205 / #1126 step 1: `EnsureGemma4Fp8ExpertOnDevice` must REFUSE the device
// upload on a device whose down-projection GEMM does not exist, instead of
// admitting and then throwing out of the middle of a decode step.
//
// WHAT MAKES THIS A REACHABILITY GATE AND NOT A UNIT TEST. The guard is entered
// through `vllm::RunGemma4Moe` — the Gemma-4 MoE layer entry point that
// `src/vllm/model_executor/models/gemma4.cpp:634` calls — with a real
// `Gemma4MoeLayerWeights`, the real router,
// the real device-expert LRU and the real fallback. Nothing here constructs the
// LRU, the `Dev`, or `vt::MatmulBTAlphaBeta` by hand. Delete the `if
// (!vt::HasMatmulBTAlphaBeta(d.q)) return false;` line from
// `EnsureGemma4Fp8ExpertOnDevice` and this file goes red, because the layer then
// throws `vt::MatmulBTAlphaBeta: no implementation for device 'cpu'` out of
// `ExpertGeGLUTopKFusedGelu`.
//
// WHY THE BACKEND IS DECORATED. The hazard is latent on every machine that can
// run this suite, and for exactly one reason: the LRU's `MakeRoom` refuses
// unless `vt::Backend::DeviceMemoryInfo` answers, and only ROCm overrides it
// (`src/vt/rocm/rocm_backend.hip`). So on a stock CPU or CUDA build the upload
// is refused by the memory probe and the guard never binds — an
// unmoved-mutation trap, not a proof. #1126 step 1 IS the change that makes
// `DeviceMemoryInfo` answer on a device with no `MatmulBTAlphaBeta` arm, so this
// test puts the tree in exactly that state: the registered CPU backend, wrapped,
// with `DeviceMemoryInfo` answering and nothing else altered. Every op still
// runs its real CPU kernel; `vt::MatmulBTAlphaBeta` still has no CPU arm. That
// combination is the one the guard exists for, and it is the only combination
// under which deleting the guard is observable.
//
// The decoration is a real `vt::Backend` registered through the public
// `vt::RegisterBackend`, restored in a destructor, and the test asserts BOTH
// directions of the probe (real backend: false; wrapped: true) before it
// concludes anything — an instrument that silently failed to arm would otherwise
// report the guard as proven when the run never reached it.
//
// `vt::RegisterBackend` documents that all registration completes before
// `main()`, and this call is after it. What that contract buys is the absence of
// synchronization on the read path: the registry is a plain pointer store and
// lock-free reads thereafter. This binary is single-threaded across the swap —
// one doctest case, no queue with a worker, no other case touching the CPU
// backend — so there is no concurrent reader to race, and the pointer is put
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
    // The BF16 device-expert LRU arm is the one #1205 traces and the one this
    // guard covers. Its FP8-native twin (`EnsureGemma4Fp8NativeOnDevice`, the
    // default) carries the same latent shape through `vt::DequantFp8ChannelBf16`
    // and is NOT guarded here — filed as its own issue and recorded under
    // `## Owed` in .agents/specs/expert-streaming.md.
    ::setenv("VT_GEMMA4_FP8_NATIVE", "0", 1);
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

TEST_CASE("gemma4 moe: the device-expert upload refuses where MatmulBTAlphaBeta has no arm") {
  vt::Queue q;
  q.device.type = vt::DeviceType::kCPU;
  q.device.index = 0;

  // Precondition 1 — the arm really is absent for this queue. True on every
  // build, HIP included: the only arm is kROCM's.
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

  // Baseline: the layer as it runs today, host-fallback expert path.
  const std::vector<uint16_t> want = RunOneToken(q, moe, x);
  REQUIRE(want.size() == static_cast<size_t>(kH));
  // A layer that produced all zeros would make the equality below vacuous.
  REQUIRE(AnyNonZero(want));

  // Now put the tree in the post-#1126 state: the memory probe answers, so the
  // LRU's `MakeRoom` will admit, and only the guard stands between the upload
  // and a `vt::MatmulBTAlphaBeta` that has no arm for this device.
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

    // THE ASSERTION. With the guard: the upload is refused, the expert loop
    // takes `EnsureGemma4Fp8ExpertCached` + `ExpertGeGLUHost`, and the layer
    // answers. Without it: `ExpertGeGLUTopKFusedGelu` reaches
    // `vt::MatmulBTAlphaBeta` and this throws out of the decode step.
    REQUIRE_NOTHROW(got = RunOneToken(q, moe, x));
  }

  // Degraded, but CORRECT: refusing the device arm must not change the answer.
  REQUIRE(got.size() == want.size());
  int same = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] == want[i]) ++same;
  }
  CHECK(same == static_cast<int>(want.size()));

  // The refusal must not have leaked device pointers into the expert records:
  // a `true` return is what assigns them, so they stay null.
  int null_dev = 0;
  for (const auto& fex : moe.experts.fp8) {
    if (fex.dev_gu == nullptr && fex.dev_dn == nullptr) ++null_dev;
  }
  CHECK(null_dev == static_cast<int>(kE));

  // The backend registry is back to what it was.
  size_t free_b = 1, total_b = 1;
  CHECK_FALSE(vt::GetBackend(vt::DeviceType::kCPU).DeviceMemoryInfo(&free_b, &total_b));
}
