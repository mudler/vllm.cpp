// Metal backend skeleton unit gates (BACKEND-METAL-MLX, W0). Newly authored —
// vLLM has no Metal tests to port. Mirrors the shape of tests/vt/test_backend.cpp
// (the CPU backend's own gates) so the two are read side by side.
//
// This TU is COMPILED ONLY in a Metal build (tests/CMakeLists.txt gates it on
// VLLM_CPP_METAL) but is deliberately plain C++: every assertion goes through
// the public vt:: seam, which is the point — if the skeleton needed ObjC in a
// test to be checkable, the seam would be leaking.
//
// Cross-device NUMERIC equality vs the CPU oracle is NOT here; it lives in
// tests/vt/test_backend_cross_device.cpp, which runs against every registered
// non-CPU backend and so covers Metal automatically.
#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/metal_profile.h"
#include "vt/op_provider.h"

#include "vllm/platforms/interface.h"
#include "vllm/v1/attention/registry.h"  // SelectAttentionBackendName
#include "vt/backend.h"
// Test-only bandwidth probe entry point.
namespace vt::metal {
void BandwidthProbe(vt::Queue& q, void* src, void* out, uint32_t n_chunks, uint32_t chunk_f4,
                    uint32_t stride_f4);
}
#include "vt/ops.h"
#include "vt/recipes.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

TEST_CASE("Metal backend is registered on a Metal-capable host") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);

  // Apple silicon is unified memory. This is load-bearing well beyond a fact
  // about the hardware: vt::Backend's SEVEN async-output primitive defaults
  // (src/vt/backend.cpp:19-32) are documented correct exactly for unified
  // backends, so the skeleton inherits them instead of implementing them.
  CHECK(metal.UnifiedMemory());

  // MTLIndirectCommandBuffer is the eventual mapping (include/vt/backend.h:92)
  // but is NOT implemented; the honest answer today is false, and the base class
  // makes BeginCapture throw loudly rather than silently no-op.
  CHECK_FALSE(metal.SupportsGraphCapture());
  Queue q = metal.CreateQueue();
  CHECK_THROWS_AS(metal.BeginCapture(q), std::runtime_error);

  CHECK(q.device.type == DeviceType::kMETAL);
  CHECK(q.handle != nullptr);  // the shared MTLCommandQueue
  CHECK(q.id != 0);            // a live identity for the workspace-key machinery

  // Apple GPU family as the capability pair; family 9 on the M4 gate box. The
  // assertion is deliberately ">= 1", not "== 9": the gate is that a REAL probe
  // ran, not that we are on one specific Mac.
  CHECK(metal.DeviceCapabilityMajor() >= 1);
  CHECK(metal.DeviceCapabilityMinor() == 0);

  metal.DestroyQueue(q);
}

TEST_CASE("Metal allocations are 64B-aligned, byte-exact and freeable") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();

  void* p = metal.Alloc(64);
  REQUIRE(p != nullptr);
  // include/vt/backend.h:26 — vt::StepArena depends on >= 64-byte alignment.
  CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0);

  metal.Memset(q, p, 0xAB, 64);
  metal.Synchronize(q);
  unsigned char dst[64];
  metal.Copy(q, dst, p, 64);
  metal.Synchronize(q);
  CHECK(dst[0] == 0xAB);
  CHECK(dst[63] == 0xAB);
  metal.Free(p);

  // A zero-byte request still yields a valid, distinct, freeable block (the CPU
  // backend's contract, which the arena relies on).
  void* z = metal.Alloc(0);
  CHECK(z != nullptr);
  metal.Free(z);
  metal.Free(nullptr);  // no-op

  metal.DestroyQueue(q);
}

TEST_CASE("Metal resolves INTERIOR pointers (tensor views/slices) to the owning buffer") {
  // vt::Tensor::Slice / ::View hand out pointers INTO an allocation, while Metal
  // binds resources, not pointers. The allocation registry (src/vt/metal/
  // metal_buffers.h) is what bridges that; this case is its gate.
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};

  const int64_t rows = 4, cols = 8;
  auto* base = static_cast<float*>(metal.Alloc(rows * cols * sizeof(float)));
  std::vector<float> host(rows * cols);
  for (size_t i = 0; i < host.size(); ++i) host[i] = -1.0f * static_cast<float>(i + 1);
  metal.Copy(q, base, host.data(), host.size() * sizeof(float));

  // Operate on rows [1,3) only — an INTERIOR pointer at a non-zero byte offset.
  Tensor sub = Tensor::Contiguous(base + cols, vt::DType::kF32, d, {2, cols});
  vt::Relu(q, sub, sub);
  metal.Synchronize(q);

  std::vector<float> back(host.size());
  metal.Copy(q, back.data(), base, back.size() * sizeof(float));
  metal.Synchronize(q);
  // Rows 0 and 3 untouched (bit-exact); rows 1-2 relu'd to zero (input was all
  // negative), which also proves the buffer OFFSET was applied and not ignored.
  CHECK(back[0] == host[0]);
  CHECK(back[cols * 3] == host[cols * 3]);
  for (int64_t i = cols; i < cols * 3; ++i) CHECK(back[i] == 0.0f);

  metal.Free(base);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal rejects memory it did not allocate, loudly") {
  // Handing a Metal kernel a host std::vector is THE bring-up mistake; it must
  // throw, never read garbage.
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  std::vector<float> host(64, 1.0f);
  Tensor t = Tensor::Contiguous(host.data(), vt::DType::kF32, d, {8, 8});
  CHECK_THROWS_AS(vt::Relu(q, t, t), std::runtime_error);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal platform is registered and reports unified/no-pool residency") {
  vllm::platforms::Platform& p = vllm::platforms::GetPlatform(DeviceType::kMETAL);
  CHECK(p.device_type() == DeviceType::kMETAL);
  CHECK_FALSE(p.is_cuda());
  CHECK_FALSE(p.is_cpu());
  CHECK(p.is_unified_memory());
  CHECK_FALSE(p.supports_graph_capture());

  // #1823. Platform::get_device_capability is an NVIDIA SM version
  // (interface.py:420-431), and Apple silicon has no SM version, so the Metal
  // platform reports ABSENT — upstream's own answer for a foreign capability
  // format, xpu.py:228-234. This assertion used to be `present()`, and that is
  // what let FlashAttentionBackend::supports_compute_capability's `>= (8, 0)`
  // be applied to an Apple GPU FAMILY number: family 9 on an M4 cleared an
  // SM-8.0 bar by coincidence, a lower family on a GitHub macos-15 runner did
  // not, and the CHECK below at :170 threw.
  CHECK_FALSE(p.get_device_capability().present());

  // The Apple family is still probed and still reachable — on vt::Backend, which
  // is where a Metal-unit question belongs. Asserting it HERE is what keeps the
  // fix from being "delete the number": the number is real, it was in the wrong
  // seam. ">= 1" rather than "== 9" because the gate must not name one Mac.
  Backend& metal_backend = vt::GetBackend(DeviceType::kMETAL);
  CHECK(metal_backend.DeviceCapabilityMajor() >= 1);
  CHECK(metal_backend.DeviceCapabilityMinor() == 0);

  // interface.py:181-187 order — bf16 is the default fallback.
  REQUIRE(p.supported_dtypes().size() == 3);
  CHECK(p.supported_dtypes()[0] == vt::DType::kBF16);

  // Unified memory: never free the only copy, never pool device scratch.
  const auto rp = p.residency_policy();
  CHECK_FALSE(rp.release_host_weights_after_upload);
  CHECK_FALSE(rp.uses_device_memory_pool);

  // Work row M3a registered kPagedAttention/kReshapeAndCache against the SAME
  // NHD cache layout FlashAttentionBackend allocates, so FLASH_ATTN is now the
  // honest answer — and it must actually RESOLVE, not merely be named.
  REQUIRE(p.get_attn_backend_priority().size() == 1);
  CHECK(p.get_attn_backend_priority()[0] == "FLASH_ATTN");
  CHECK(vllm::v1::SelectAttentionBackendName(p) == "FLASH_ATTN");
  // MLA stays unoffered: no Metal MLA kernel exists, so a use_mla request must
  // keep failing loudly rather than selecting an unimplemented backend.
  vllm::platforms::AttnSelectorConfig mla;
  mla.use_mla = true;
  CHECK(p.get_attn_backend_priority(mla).empty());
  CHECK_THROWS_AS(vllm::v1::SelectAttentionBackendName(p, "", mla), std::runtime_error);
}

TEST_CASE("Metal registers the W0 op set and NOT the unimplemented rest") {
  // The skeleton's registered surface, stated as an executable fact so a later
  // work row cannot quietly claim more than it implements.
  for (vt::OpId op : {vt::OpId::kAdd, vt::OpId::kRelu, vt::OpId::kSiluAndMul,
                      vt::OpId::kCastBf16, vt::OpId::kCastF32, vt::OpId::kLayerNorm,
                      vt::OpId::kRmsNorm, vt::OpId::kFusedChain,
                      // Added with the provider seam: the native MSL dense GEMM
                      // pair, which is what makes MLX a CONFIGURATION rather
                      // than the only way to get a GEMM on this backend.
                      vt::OpId::kMatmul, vt::OpId::kMatmulBT}) {
    CHECK(vt::OpRegistered(op, DeviceType::kMETAL));
  }
  // Work row M3a — EXACTLY the five ops OPT-125m needs beyond the W0 set, and no
  // more. kPagedAttention stays OURS even once MLX is enabled: MLX has no
  // paged-KV primitive at all (metal-mlx-reuse-study.md §5.3).
  for (vt::OpId op : {vt::OpId::kPagedAttention, vt::OpId::kReshapeAndCache,
                      vt::OpId::kEmbedding, vt::OpId::kQkvSplit,
                      vt::OpId::kGreedyArgmax}) {
    CHECK(vt::OpRegistered(op, DeviceType::kMETAL));
  }
  // Work row M3b — the RoPE ops Qwen3-dense (`Qwen3ForCausalLM`) needs beyond OPT's
  // set. The DEFAULT (VT_QWEN3_ROPE_CACHE) path builds the per-step cache
  // (kRopeCosSinCache) and applies it (kRopeFromCache); kRopeNeox serves the
  // cache-off opt-out.
  for (vt::OpId op : {vt::OpId::kRopeCosSinCache, vt::OpId::kRopeFromCache,
                      vt::OpId::kRopeNeox}) {
    CHECK(vt::OpRegistered(op, DeviceType::kMETAL));
  }
  // Still stubbed, and asserted so a later row cannot quietly claim more than it
  // implements: the quant tier, the GDN/MoE families, and every sampler op except
  // greedy argmax. `OpRegistered` means a NATIVE kernel exists, so it stays false
  // regardless of what the portable reference tier installs underneath.
  for (vt::OpId op : {vt::OpId::kRandomSample, vt::OpId::kComputeProbs,
                      vt::OpId::kMoeCombine, vt::OpId::kGdnStateGather}) {
    CHECK_FALSE(vt::OpRegistered(op, DeviceType::kMETAL));
  }
  // A partial backend is still a supported state, but since S5 that no longer
  // means "GetOp throws". Metal is unified-memory, so ReferenceTierEligible holds
  // and the portable CPU tier installs LAZILY on the miss: GetOp returns a
  // working (correct but slow) kernel instead. This asserts the CURRENT contract
  // plus the two facts that keep it honest — the selection is the reference tier
  // BY NAME, and the observability counter moves — so a Metal op silently running
  // on the CPU path can never masquerade as a native kernel (Risk 7).
  REQUIRE(vt::ReferenceTierEligible(DeviceType::kMETAL));
  void* moe_combine = nullptr;
  CHECK_NOTHROW(moe_combine = vt::GetOp(vt::OpId::kMoeCombine, DeviceType::kMETAL));
  CHECK(moe_combine != nullptr);
  const auto moe_stats = vt::GetOpProviderStats(vt::OpId::kMoeCombine, DeviceType::kMETAL);
  REQUIRE(moe_stats.last_selected != nullptr);
  CHECK(std::string(moe_stats.last_selected) == std::string(vt::kReferenceProviderName));
  // Deliberately `> 0` and not a strict per-call increment: Resolve() caches the
  // winner in the slot, and ResetOpProviderStats does not clear that cache, so a
  // second resolution of the same (op, device) never re-counts. `> 0` is the
  // order-independent form of "the portable path announced itself".
  CHECK(vt::GetReferenceTierHits() > 0);
}

// ===========================================================================
// Dense GEMM: the native MSL provider vs the CPU oracle, and — when the optional
// MLX provider is built in (-DVLLM_CPP_MLX=ON) — MLX vs MSL vs the CPU oracle,
// per op, at real shapes.
//
// THE BAR IS NMSE <= 5e-4, NOT BIT-EXACTNESS, and that is a deliberate and
// stated position, not a tolerance chosen to make a test pass: the CPU tier's
// reproducibility comes from a fixed SEQUENTIAL accumulation
// (src/vt/cpu/cpu_quant_dot.cpp:22-28) and no GPU tile reduction preserves that
// order. MLX pins `setFastMathEnabled(false)` and we pin MTLMathModeSafe, so
// both are IEEE — but they are DIFFERENT reduction orders, and bit-exactness
// across providers is not on offer. Nothing here claims it.
//
// AND THE TEST PROVES WHICH PROVIDER RAN. A passing assertion does not: both
// providers compute the same GEMM, so a silent fall-back to MSL would look
// identical to MLX succeeding. `vt::GetOpProviderStats(...).last_selected` is
// checked on every arm, which is exactly the fan-out spike's Risk 4
// (a probe failing SILENTLY into the slow path) made detectable.
namespace {

double Nmse(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den > 0.0 ? num / den : num;
}

// bf16 round-trip so every arm consumes the IDENTICAL input bits — otherwise a
// dtype-conversion difference would masquerade as a kernel difference.
float Bf16RT(float v) { return vt::BF16ToF32(vt::F32ToBF16(v)); }

struct GemmCase {
  const char* name;
  int64_t m, k, n;
  vt::DType dt;
};

// Run one GEMM on Metal with the currently-selected provider and return the
// result in f32, plus the provider name that actually served it.
std::vector<float> RunMetalGemm(const GemmCase& c, bool bt, const std::vector<float>& a_h,
                                const std::vector<float>& b_h, std::string* provider,
                                unsigned long long* declines) {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const size_t esz = vt::SizeOf(c.dt);

  auto upload = [&](const std::vector<float>& h) {
    void* p = metal.Alloc(h.size() * esz);
    if (c.dt == vt::DType::kF32) {
      metal.Copy(q, p, h.data(), h.size() * esz);
    } else {
      std::vector<uint16_t> packed(h.size());
      for (size_t i = 0; i < h.size(); ++i) packed[i] = vt::F32ToBF16(h[i]);
      metal.Copy(q, p, packed.data(), packed.size() * esz);
    }
    return p;
  };

  void* da = upload(a_h);
  void* db = upload(b_h);
  void* dc = metal.Alloc(static_cast<size_t>(c.m * c.n) * esz);
  metal.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, c.dt, d, {c.m, c.k});
  Tensor tb = bt ? Tensor::Contiguous(db, c.dt, d, {c.n, c.k})
                 : Tensor::Contiguous(db, c.dt, d, {c.k, c.n});
  Tensor tc = Tensor::Contiguous(dc, c.dt, d, {c.m, c.n});

  const vt::OpId op = bt ? vt::OpId::kMatmulBT : vt::OpId::kMatmul;
  vt::ResetOpProviderStats(op, DeviceType::kMETAL);
  if (bt) {
    vt::MatmulBT(q, tc, ta, tb);
  } else {
    vt::Matmul(q, tc, ta, tb);
  }
  metal.Synchronize(q);

  const auto stats = vt::GetOpProviderStats(op, DeviceType::kMETAL);
  *provider = stats.last_selected != nullptr ? stats.last_selected : "<none>";
  // `last_selected` alone is NOT proof that the accelerator COMPUTED anything —
  // a selected provider may still decline the call inside its kernel and forward
  // down. `declines` is that second half, and without it a silent fall-back
  // would be indistinguishable from success (fan-out spike Risk 4).
  *declines = stats.declines;

  std::vector<float> out(static_cast<size_t>(c.m * c.n));
  if (c.dt == vt::DType::kF32) {
    metal.Copy(q, out.data(), dc, out.size() * esz);
    metal.Synchronize(q);
  } else {
    std::vector<uint16_t> packed(out.size());
    metal.Copy(q, packed.data(), dc, packed.size() * esz);
    metal.Synchronize(q);
    for (size_t i = 0; i < out.size(); ++i) out[i] = vt::BF16ToF32(packed[i]);
  }

  metal.Free(da);
  metal.Free(db);
  metal.Free(dc);
  metal.DestroyQueue(q);
  return out;
}

// The oracle: our own CPU backend, through the SAME public vt:: entry point.
std::vector<float> RunCpuGemm(const GemmCase& c, bool bt, const std::vector<float>& a_h,
                              const std::vector<float>& b_h) {
  Queue q{Device{DeviceType::kCPU, 0}, nullptr};
  std::vector<float> a = a_h, b = b_h, out(static_cast<size_t>(c.m * c.n), 0.0f);
  const Device d{DeviceType::kCPU, 0};
  Tensor ta = Tensor::Contiguous(a.data(), vt::DType::kF32, d, {c.m, c.k});
  Tensor tb = bt ? Tensor::Contiguous(b.data(), vt::DType::kF32, d, {c.n, c.k})
                 : Tensor::Contiguous(b.data(), vt::DType::kF32, d, {c.k, c.n});
  Tensor tc = Tensor::Contiguous(out.data(), vt::DType::kF32, d, {c.m, c.n});
  if (bt) {
    vt::MatmulBT(q, tc, ta, tb);
  } else {
    vt::Matmul(q, tc, ta, tb);
  }
  return out;
}

}  // namespace

#ifdef VLLM_CPP_MLX
// THE #1584 EXACTNESS ASSERTION, ON THE MLX CALL SITE THAT ISSUE WAS WRITTEN
// FROM (issue #1692 O2, row KERNEL-ACCEL-PROVIDER-DECLINE-EXACT).
//
// `MlxFallback` caches its fallback pointer in a function-local static and
// resolves it through `GetOpFallbackUncounted`, pairing it with an explicit
// `NoteOpDecline`. If that resolver were the COUNTING `GetOpFallback` instead,
// the resolution would add a decline of its own, and `declines` would read N+1
// over N declines -- but ONLY until the static is warm, after which the count is
// exact again forever.
//
// TWO THINGS MAKE THAT DETECTABLE HERE, and both are load-bearing:
//
//  1. THIS CASE IS FIRST. doctest's default ordering is `--order-by=file`, which
//     for a single-file binary is line order (third_party/doctest/doctest.h:5476
//     `fileOrderComparator`, defaulted at :5714). No case above this line issues
//     a Metal `Matmul`/`MatmulBT` -- `:179` only asks `OpRegistered` -- so this
//     case owns the FIRST decline of the process and `MlxFallback`'s statics are
//     cold when it runs. That is the difference from the CUDA arm of the same
//     row, whose suite warms its own static in an earlier case and stays green
//     with the defect reintroduced (spec `## 12.2`, issue #1812). Move this case
//     below the dense-GEMM case and it silently stops measuring anything.
//  2. IT ASSERTS `== 1`, NOT `>= 1`. The two pre-existing MLX decline assertions
//     (`>= 1` at the decode arm below, and in the "MLX DECLINES a shape it
//     cannot express" case) cannot see an off-by-one in either direction; that
//     is exactly what #1692 says they cannot do. This one can.
//
// Belt and braces for (1): tests/CMakeLists.txt also registers this case as its
// own ctest entry, so it runs in a dedicated process where the order of the rest
// of the file cannot matter.
TEST_CASE("MLX counts EXACTLY one decline for the first decline of the process") {
  // Loud rather than skipped. A `macos-15` runner with no Metal device registers
  // no MLX provider (`MlxSupports` gates on `MetalContext::Available()`), and a
  // case that SKIPPED there would report `assertions: 0` and `SUCCESS!` -- the
  // absence-wearing-a-pass shape this whole row exists to remove.
  REQUIRE(vt::OpProviderCount(vt::OpId::kMatmulBT, DeviceType::kMETAL) >= 2);
  bool mlx_registered = false;
  for (int i = 0; i < vt::OpProviderCount(vt::OpId::kMatmulBT, DeviceType::kMETAL); ++i) {
    const char* n = vt::OpProviderNameAt(vt::OpId::kMatmulBT, DeviceType::kMETAL, i);
    if (n != nullptr && std::string(n) == std::string("mlx")) mlx_registered = true;
  }
  REQUIRE(mlx_registered);

  // m == 1 is the decode GEMV, the one shape `TryMlxMatmul` declines by design
  // (`kMlxMinRows`). Small on purpose: this case measures accounting, not
  // numerics, and the arms below cover the numbers at real widths.
  const GemmCase c{"decline-exact bf16 1x256x256", 1, 256, 256, vt::DType::kBF16};
  std::vector<float> a(static_cast<size_t>(c.m * c.k), 0.0f);
  std::vector<float> b(static_cast<size_t>(c.k * c.n), 0.0f);
  std::mt19937 rng(0xD3C11Eu);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : a) x = Bf16RT(dist(rng));
  for (auto& x : b) x = Bf16RT(dist(rng));

  // FIRST decline of the process: `MlxFallback(kMatmulBT)`'s static resolves
  // inside this counted window.
  std::string p1;
  unsigned long long d1 = 0;
  const std::vector<float> got1 = RunMetalGemm(c, /*bt=*/true, a, b, &p1, &d1);
  CHECK(p1 == std::string("mlx"));
  CHECK(d1 == 1ull);

  // SECOND decline, static now warm. The point of the pair is that the two
  // readings must be the SAME number: "exact from the FIRST decline" is a
  // statement about d1 == d2, and with the counting resolver d1 is 2 and d2 is 1.
  std::string p2;
  unsigned long long d2 = 0;
  const std::vector<float> got2 = RunMetalGemm(c, /*bt=*/true, a, b, &p2, &d2);
  CHECK(p2 == std::string("mlx"));
  CHECK(d2 == 1ull);
  CHECK(d1 == d2);

  // ...and the fallback it resolved actually RAN. Without this the case would
  // pass on a `MlxFallback` that returned a pointer nobody called, which is the
  // reachability half of the same question (.agents/reachability.md): the
  // production call site is `MlxMatmulBTKernel`'s `MlxFallback(...)(q, ...)`,
  // and deleting the forward leaves `out` untouched.
  const std::vector<float> ref = RunCpuGemm(c, /*bt=*/true, a, b);
  const double nmse = Nmse(got1, ref);
  CAPTURE(nmse);
  CHECK(nmse <= 5e-4);
  CHECK(Nmse(got2, ref) <= 5e-4);
  MESSAGE("MLX decline accounting: first=" << d1 << " second=" << d2
                                           << " provider=" << p1 << " NMSE vs CPU=" << nmse);
}
#endif  // VLLM_CPP_MLX

TEST_CASE("Metal dense GEMM matches the CPU oracle, and the provider that ran is named") {
  // Decode-shaped (M=1), prefill-shaped, and a square f32 arm. Sizes are the
  // real projection widths a 1.7B-class dense model uses, not toy shapes.
  const GemmCase cases[] = {
      {"decode bf16 1x2048x2048", 1, 2048, 2048, vt::DType::kBF16},
      {"prefill bf16 32x2048x6144", 32, 2048, 6144, vt::DType::kBF16},
      {"square f32 128x512x512", 128, 512, 512, vt::DType::kF32},
  };

  for (const GemmCase& c : cases) {
    for (bool bt : {false, true}) {
      CAPTURE(c.name);
      CAPTURE(bt);
      std::mt19937 rng(0xC0FFEEu);
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      std::vector<float> a(static_cast<size_t>(c.m * c.k));
      std::vector<float> b(static_cast<size_t>(c.k * c.n));
      for (auto& x : a) x = c.dt == vt::DType::kBF16 ? Bf16RT(dist(rng)) : dist(rng);
      for (auto& x : b) x = c.dt == vt::DType::kBF16 ? Bf16RT(dist(rng)) : dist(rng);

      const std::vector<float> ref = RunCpuGemm(c, bt, a, b);

      // --- arm 1: the NATIVE MSL provider, with any accelerator forced off.
      vt::DisableOpProvider("mlx", true);
      std::string msl_provider;
      unsigned long long msl_declines = 0;
      const std::vector<float> msl = RunMetalGemm(c, bt, a, b, &msl_provider, &msl_declines);
      vt::DisableOpProvider("mlx", false);
      CHECK(msl_provider == std::string(vt::kNativeProviderName));
      CHECK(msl_declines == 0);
      const double msl_nmse = Nmse(msl, ref);
      CAPTURE(msl_nmse);
      CHECK(msl_nmse <= 5e-4);

#ifdef VLLM_CPP_MLX
      // --- arm 2: the MLX provider. Same binary, same inputs, same entry point;
      // the ONLY difference is which provider the seam selected. If MLX had
      // silently declined this shape, `mlx_provider` would read "vt-native" and
      // this check — not the numeric one — is what would catch it.
      std::string mlx_provider;
      unsigned long long mlx_declines = 0;
      const std::vector<float> mlx = RunMetalGemm(c, bt, a, b, &mlx_provider, &mlx_declines);
      CHECK(mlx_provider == std::string("mlx"));
      // MLX is SHAPE-GATED to prefill: it declines m == 1 (the decode GEMV),
      // because its per-op eval + memcpy costs more than its GEMM saves when the
      // call happens once per token. So a decline is EXPECTED for the decode
      // shape and a bug for the others — asserting the exact split is what keeps
      // the gate honest in both directions.
      const bool decode_shape = c.m == 1;
      // Decode declines, prefill does not. The decode side asserts >= 1 rather
      // than == 1 because this harness may invoke the op more than once per
      // shape; what the gate must guarantee is the DIRECTION — MLX steps aside
      // for m == 1 and takes every other shape.
      if (decode_shape) { CHECK(mlx_declines >= 1ull); } else { CHECK(mlx_declines == 0ull); }
      // On the declined shape `mlx` IS the native result (the fallback ran), so
      // these still hold — they just stop being a statement about MLX.
      const double mlx_vs_cpu = Nmse(mlx, ref);
      const double mlx_vs_msl = Nmse(mlx, msl);
      CAPTURE(mlx_vs_cpu);
      CAPTURE(mlx_vs_msl);
      CHECK(mlx_vs_cpu <= 5e-4);
      CHECK(mlx_vs_msl <= 5e-4);
      MESSAGE("GEMM [" << std::string(c.name) << "] bt=" << bt << " NMSE msl-vs-cpu=" << msl_nmse
                       << " mlx-vs-cpu=" << mlx_vs_cpu << " mlx-vs-msl=" << mlx_vs_msl);
#else
      MESSAGE("GEMM [" << std::string(c.name) << "] bt=" << bt << " NMSE msl-vs-cpu=" << msl_nmse
                       << " (MLX provider not built: -DVLLM_CPP_MLX=OFF)");
#endif
    }
  }
}

#ifdef VLLM_CPP_MLX
TEST_CASE("MLX DECLINES a shape it cannot express and the native MSL GEMM serves it") {
  // The decline-and-fall-back axis, exercised END TO END on a real accelerator
  // rather than only on the synthetic providers in tests/vt/test_op_provider.cpp.
  //
  // The shape chosen is one MLX genuinely cannot take through its public API:
  // an activation that is an INTERIOR view of a larger allocation.
  // `mlx::core::array::set_data` sets `data_ptr` to the buffer's `contents()`,
  // so a non-zero buffer offset is not expressible, and the provider declines
  // rather than silently reading from row 0. `vt::Tensor::Slice`/`View` produce
  // exactly this pointer, so it is not a contrived case.
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const int64_t m = 8, k = 256, n = 128;

  std::mt19937 rng(7u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> a_full(static_cast<size_t>((m + 1) * k));
  std::vector<float> b_h(static_cast<size_t>(n * k));
  for (auto& x : a_full) x = dist(rng);
  for (auto& x : b_h) x = dist(rng);

  auto* da = static_cast<float*>(metal.Alloc(a_full.size() * sizeof(float)));
  auto* db = static_cast<float*>(metal.Alloc(b_h.size() * sizeof(float)));
  auto* dc = static_cast<float*>(metal.Alloc(static_cast<size_t>(m * n) * sizeof(float)));
  metal.Copy(q, da, a_full.data(), a_full.size() * sizeof(float));
  metal.Copy(q, db, b_h.data(), b_h.size() * sizeof(float));
  metal.Synchronize(q);

  // Rows [1, m+1) — an interior pointer at a non-zero offset.
  Tensor ta = Tensor::Contiguous(da + k, vt::DType::kF32, d, {m, k});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {n, k});
  Tensor tc = Tensor::Contiguous(dc, vt::DType::kF32, d, {m, n});

  vt::ResetOpProviderStats(vt::OpId::kMatmulBT, DeviceType::kMETAL);
  vt::MatmulBT(q, tc, ta, tb);
  metal.Synchronize(q);

  const auto stats = vt::GetOpProviderStats(vt::OpId::kMatmulBT, DeviceType::kMETAL);
  CHECK(std::string(stats.last_selected) == "mlx");  // MLX WAS selected...
  CHECK(stats.declines >= 1);  // ... and declined. (>= because the shape gate can
                               // decline for a second, independent reason: this
                               // fixture's m may also be below kMlxMinRows. The
                               // point of the assertion is that the DECLINE PATH
                               // ran and produced a correct result, which the
                               // value check below proves.)

  std::vector<float> got(static_cast<size_t>(m * n));
  metal.Copy(q, got.data(), dc, got.size() * sizeof(float));
  metal.Synchronize(q);

  // And the fall-back produced the RIGHT answer, not just an answer.
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  std::vector<float> a_slice(a_full.begin() + k, a_full.end());
  std::vector<float> b_cpu = b_h, ref(static_cast<size_t>(m * n), 0.0f);
  const Device cd{DeviceType::kCPU, 0};
  Tensor ca = Tensor::Contiguous(a_slice.data(), vt::DType::kF32, cd, {m, k});
  Tensor cb = Tensor::Contiguous(b_cpu.data(), vt::DType::kF32, cd, {n, k});
  Tensor cc = Tensor::Contiguous(ref.data(), vt::DType::kF32, cd, {m, n});
  vt::MatmulBT(cq, cc, ca, cb);
  CHECK(Nmse(got, ref) <= 5e-4);

  metal.Free(da);
  metal.Free(db);
  metal.Free(dc);
  metal.DestroyQueue(q);
}

TEST_CASE("MLX registers as a SECOND provider and the native MSL GEMM survives it") {
  // The precise property the old flat op table could not give: two providers of
  // ONE op on ONE device coexisting, ordered deterministically rather than by
  // static-init order, with the loser still reachable.
  CHECK(vt::OpProviderCount(vt::OpId::kMatmul, DeviceType::kMETAL) == 2);
  CHECK(std::string(vt::OpProviderNameAt(vt::OpId::kMatmul, DeviceType::kMETAL, 0)) == "mlx");
  CHECK(std::string(vt::OpProviderNameAt(vt::OpId::kMatmul, DeviceType::kMETAL, 1)) ==
        std::string(vt::kNativeProviderName));
  // And the decline path resolves to ours, which is what MlxMatmulKernel calls
  // when it meets a shape or dtype it will not take.
  CHECK(vt::GetOpFallback(vt::OpId::kMatmul, DeviceType::kMETAL, "mlx") != nullptr);
}
#endif

// ===========================================================================
// M3a — the five ops OPT-125m needs beyond the W0 set, each against our own CPU
// backend as the oracle, through the SAME public vt:: entry point.
//
// TWO BARS, AND THE DIFFERENCE IS PRINCIPLED rather than a tolerance picked to
// make a test pass:
//   * kEmbedding / kQkvSplit / kReshapeAndCache / kGreedyArgmax are pure
//     GATHER / LAYOUT / SELECTION ops. They perform no floating-point reduction,
//     so a GPU implementation has no reordering freedom and the bar is
//     BIT-EXACTNESS, asserted on the raw bits. (kGreedyArgmax does reduce, but
//     over a max with an explicit lowest-index tie-break, which is associative
//     AND commutative on the (value, index) pair — so its result is genuinely
//     order-independent, which is why bit-exactness is honest for it too.)
//   * kPagedAttention accumulates a softmax in f32. The CPU reference is a
//     three-pass materialized softmax and the Metal kernel is the algebraically
//     identical ONLINE form, so the reduction ORDER differs by construction.
//     The bar is the ported NMSE <= 5e-4. No bit-exactness is claimed for it.
//
// AND EVERY ARM PROVES THE METAL PATH ACTUALLY EXECUTED. Two independent
// mechanisms, because neither alone is sufficient:
//   (1) the output buffer is NaN-POISONED before the call, so a kernel that
//       never ran leaves NaN and cannot pass a numeric check by accident;
//   (2) `declines == 0` on the op's provider stats — `last_selected` alone is
//       NOT proof, since a selected provider can decline inside its kernel and
//       forward down the stack (fan-out spike Risk 4).
namespace {

// A Metal allocation with upload/download and NaN poisoning. Frees on scope exit
// so a failing REQUIRE cannot leak a device buffer.
class MBuf {
 public:
  MBuf(Backend& b, Queue& q, size_t bytes) : b_(b), q_(q), bytes_(bytes) {
    p_ = b_.Alloc(bytes_);
  }
  ~MBuf() { b_.Free(p_); }
  MBuf(const MBuf&) = delete;
  MBuf& operator=(const MBuf&) = delete;

  void* ptr() const { return p_; }
  void Upload(const void* src) { b_.Copy(q_, p_, src, bytes_); }
  void Download(void* dst) const { b_.Copy(q_, dst, p_, bytes_); }
  // Fill with a quiet-NaN bit pattern of the given element width, so an
  // un-executed kernel is DETECTABLE rather than reading as zeros (which a
  // masked or empty region could legitimately be).
  void PoisonNaN(size_t esz) {
    if (esz == 4) {
      std::vector<uint32_t> nan(bytes_ / 4, 0x7FC00000u);
      b_.Copy(q_, p_, nan.data(), bytes_);
    } else {
      std::vector<uint16_t> nan(bytes_ / 2, 0x7FC0u);
      b_.Copy(q_, p_, nan.data(), bytes_);
    }
  }

 private:
  Backend& b_;
  Queue& q_;
  size_t bytes_;
  void* p_ = nullptr;
};

std::vector<uint16_t> PackBf16(const std::vector<float>& h) {
  std::vector<uint16_t> out(h.size());
  for (size_t i = 0; i < h.size(); ++i) out[i] = vt::F32ToBF16(h[i]);
  return out;
}

// The (op, kMETAL) decline counter. Zero is the assertion: a non-zero value
// means the Metal provider forwarded the work elsewhere, which a numeric check
// alone cannot distinguish from success.
unsigned long long DeclinesAfter(vt::OpId op) {
  return vt::GetOpProviderStats(op, DeviceType::kMETAL).declines;
}

}  // namespace

TEST_CASE("Metal kEmbedding is BIT-EXACT vs the CPU oracle") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const int64_t vocab = 97, h = 40, t = 13;

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
  std::vector<float> table_f(static_cast<size_t>(vocab * h));
  for (auto& x : table_f) x = Bf16RT(ud(rng));
  std::vector<int32_t> ids(static_cast<size_t>(t));
  for (auto& x : ids) x = static_cast<int32_t>(rng() % static_cast<uint32_t>(vocab));

  const std::vector<uint16_t> table_b = PackBf16(table_f);
  MBuf dtab(metal, q, table_b.size() * 2), dids(metal, q, ids.size() * 4),
      dout(metal, q, static_cast<size_t>(t * h) * 2);
  dtab.Upload(table_b.data());
  dids.Upload(ids.data());
  dout.PoisonNaN(2);
  metal.Synchronize(q);

  Tensor ttab = Tensor::Contiguous(dtab.ptr(), vt::DType::kBF16, d, {vocab, h});
  Tensor tids = Tensor::Contiguous(dids.ptr(), vt::DType::kI32, d, {t});
  Tensor tout = Tensor::Contiguous(dout.ptr(), vt::DType::kBF16, d, {t, h});
  vt::ResetOpProviderStats(vt::OpId::kEmbedding, DeviceType::kMETAL);
  vt::Embedding(q, tout, ttab, tids);
  metal.Synchronize(q);
  CHECK(DeclinesAfter(vt::OpId::kEmbedding) == 0);

  std::vector<uint16_t> got(static_cast<size_t>(t * h));
  dout.Download(got.data());
  metal.Synchronize(q);

  std::vector<uint16_t> tab_cpu = table_b, ref(static_cast<size_t>(t * h), 0);
  std::vector<int32_t> ids_cpu = ids;
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  const Device cd{DeviceType::kCPU, 0};
  Tensor ctab = Tensor::Contiguous(tab_cpu.data(), vt::DType::kBF16, cd, {vocab, h});
  Tensor cids = Tensor::Contiguous(ids_cpu.data(), vt::DType::kI32, cd, {t});
  Tensor cout = Tensor::Contiguous(ref.data(), vt::DType::kBF16, cd, {t, h});
  vt::Embedding(cq, cout, ctab, cids);

  // A pure row gather: no arithmetic, so the bits must be IDENTICAL — which also
  // proves the poison is gone from every element.
  CHECK(got == ref);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal kQkvSplit is BIT-EXACT vs the CPU oracle") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  // OPT is pre-GQA multi-head so its q/k/v are equal-width; the kernel handles
  // INDEPENDENT widths, so this uses unequal ones to exercise that.
  const int64_t t = 11, qd = 24, kd = 12, vd = 12;

  std::mt19937 rng(11);
  std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
  std::vector<float> merged(static_cast<size_t>(t * (qd + kd + vd)));
  for (auto& x : merged) x = Bf16RT(ud(rng));
  const std::vector<uint16_t> mb = PackBf16(merged);

  MBuf din(metal, q, mb.size() * 2), dqb(metal, q, static_cast<size_t>(t * qd) * 2),
      dkb(metal, q, static_cast<size_t>(t * kd) * 2),
      dvb(metal, q, static_cast<size_t>(t * vd) * 2);
  din.Upload(mb.data());
  dqb.PoisonNaN(2);
  dkb.PoisonNaN(2);
  dvb.PoisonNaN(2);
  metal.Synchronize(q);

  Tensor tin = Tensor::Contiguous(din.ptr(), vt::DType::kBF16, d, {t, qd + kd + vd});
  Tensor tq = Tensor::Contiguous(dqb.ptr(), vt::DType::kBF16, d, {t, qd});
  Tensor tk = Tensor::Contiguous(dkb.ptr(), vt::DType::kBF16, d, {t, kd});
  Tensor tv = Tensor::Contiguous(dvb.ptr(), vt::DType::kBF16, d, {t, vd});
  vt::ResetOpProviderStats(vt::OpId::kQkvSplit, DeviceType::kMETAL);
  vt::QkvSplit(q, tq, tk, tv, tin);
  metal.Synchronize(q);
  CHECK(DeclinesAfter(vt::OpId::kQkvSplit) == 0);

  std::vector<uint16_t> gq(static_cast<size_t>(t * qd)), gk(static_cast<size_t>(t * kd)),
      gv(static_cast<size_t>(t * vd));
  dqb.Download(gq.data());
  dkb.Download(gk.data());
  dvb.Download(gv.data());
  metal.Synchronize(q);

  std::vector<uint16_t> mcpu = mb, rq(gq.size(), 0), rk(gk.size(), 0), rv(gv.size(), 0);
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  const Device cd{DeviceType::kCPU, 0};
  Tensor cin = Tensor::Contiguous(mcpu.data(), vt::DType::kBF16, cd, {t, qd + kd + vd});
  Tensor cqt = Tensor::Contiguous(rq.data(), vt::DType::kBF16, cd, {t, qd});
  Tensor ckt = Tensor::Contiguous(rk.data(), vt::DType::kBF16, cd, {t, kd});
  Tensor cvt = Tensor::Contiguous(rv.data(), vt::DType::kBF16, cd, {t, vd});
  vt::QkvSplit(cq, cqt, ckt, cvt, cin);

  CHECK(gq == rq);
  CHECK(gk == rk);
  CHECK(gv == rv);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal kReshapeAndCache is BIT-EXACT vs the CPU oracle, incl. the slot<0 skip") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const int64_t nblocks = 6, bsz = 8, hkv = 3, dh = 16, t = 10;
  const int64_t page = hkv * dh;
  const size_t cache_elems = static_cast<size_t>(nblocks * bsz * hkv * dh);

  std::mt19937 rng(13);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  std::vector<float> kf(static_cast<size_t>(t * page)), vf(static_cast<size_t>(t * page));
  for (auto& x : kf) x = Bf16RT(ud(rng));
  for (auto& x : vf) x = Bf16RT(ud(rng));
  const std::vector<uint16_t> kb = PackBf16(kf), vb = PackBf16(vf);

  // Slots scattered across blocks, with one PADDED (-1) token — the upstream
  // skip whose omission would silently corrupt a real batch.
  std::vector<int64_t> slots{0, 9, 17, 3, -1, 40, 25, 8, 33, 11};
  REQUIRE(static_cast<int64_t>(slots.size()) == t);

  MBuf dk(metal, q, kb.size() * 2), dv(metal, q, vb.size() * 2),
      dkc(metal, q, cache_elems * 2), dvc(metal, q, cache_elems * 2),
      dslots(metal, q, slots.size() * 8);
  dk.Upload(kb.data());
  dv.Upload(vb.data());
  dslots.Upload(slots.data());
  // The cache is PRE-FILLED with a known pattern rather than poisoned: this op
  // writes only the mapped slots, and the untouched remainder must survive byte
  // for byte — including the whole page belonging to the slot<0 token.
  std::vector<uint16_t> seed(cache_elems);
  for (size_t i = 0; i < seed.size(); ++i) seed[i] = static_cast<uint16_t>(0x3C00u + (i % 977));
  dkc.Upload(seed.data());
  dvc.Upload(seed.data());
  metal.Synchronize(q);

  Tensor tk = Tensor::Contiguous(dk.ptr(), vt::DType::kBF16, d, {t, hkv, dh});
  Tensor tv = Tensor::Contiguous(dv.ptr(), vt::DType::kBF16, d, {t, hkv, dh});
  Tensor tkc = Tensor::Contiguous(dkc.ptr(), vt::DType::kBF16, d, {nblocks, bsz, hkv, dh});
  Tensor tvc = Tensor::Contiguous(dvc.ptr(), vt::DType::kBF16, d, {nblocks, bsz, hkv, dh});
  Tensor tsl = Tensor::Contiguous(dslots.ptr(), vt::DType::kI64, d, {t});
  vt::ResetOpProviderStats(vt::OpId::kReshapeAndCache, DeviceType::kMETAL);
  vt::ReshapeAndCache(q, tk, tv, tkc, tvc, tsl);
  metal.Synchronize(q);
  CHECK(DeclinesAfter(vt::OpId::kReshapeAndCache) == 0);

  std::vector<uint16_t> gkc(cache_elems), gvc(cache_elems);
  dkc.Download(gkc.data());
  dvc.Download(gvc.data());
  metal.Synchronize(q);

  std::vector<uint16_t> kcpu = kb, vcpu = vb, rkc = seed, rvc = seed;
  std::vector<int64_t> scpu = slots;
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  const Device cd{DeviceType::kCPU, 0};
  Tensor ck = Tensor::Contiguous(kcpu.data(), vt::DType::kBF16, cd, {t, hkv, dh});
  Tensor cv = Tensor::Contiguous(vcpu.data(), vt::DType::kBF16, cd, {t, hkv, dh});
  Tensor ckc = Tensor::Contiguous(rkc.data(), vt::DType::kBF16, cd, {nblocks, bsz, hkv, dh});
  Tensor cvc = Tensor::Contiguous(rvc.data(), vt::DType::kBF16, cd, {nblocks, bsz, hkv, dh});
  Tensor csl = Tensor::Contiguous(scpu.data(), vt::DType::kI64, cd, {t});
  vt::ReshapeAndCache(cq, ck, cv, ckc, cvc, csl);

  // A raw element copy on both sides => the ENTIRE cache must be byte-identical,
  // which simultaneously proves the written slots are right and the unwritten
  // ones (including the padded token's) were not touched.
  CHECK(gkc == rkc);
  CHECK(gvc == rvc);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal kGreedyArgmax is BIT-EXACT vs the CPU oracle, tie rule included") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const int64_t n = 5, v = 50272;  // OPT-125m's real vocab

  std::mt19937 rng(17);
  std::uniform_real_distribution<float> ud(-8.0f, 8.0f);
  std::vector<float> logits(static_cast<size_t>(n * v));
  for (auto& x : logits) x = ud(rng);
  // Row 0: a DELIBERATE TIE at two positions, both strictly greater than the
  // rest. torch.argmax and our CPU reference both return the LOWER index; a tree
  // reduction that ignored the tie rule would be free to return either, so this
  // is the assertion that pins it.
  for (int64_t j = 0; j < v; ++j) logits[static_cast<size_t>(j)] = -1.0f;
  logits[static_cast<size_t>(31337)] = 5.0f;
  logits[static_cast<size_t>(48000)] = 5.0f;
  // Row 1: every value identical => the answer must be index 0.
  for (int64_t j = 0; j < v; ++j) logits[static_cast<size_t>(v + j)] = 2.5f;

  MBuf dlog(metal, q, logits.size() * 4), dids(metal, q, static_cast<size_t>(n) * 8);
  dlog.Upload(logits.data());
  std::vector<int64_t> poison(static_cast<size_t>(n), -424242);
  dids.Upload(poison.data());
  metal.Synchronize(q);

  Tensor tl = Tensor::Contiguous(dlog.ptr(), vt::DType::kF32, d, {n, v});
  Tensor ti = Tensor::Contiguous(dids.ptr(), vt::DType::kI64, d, {n});
  vt::ResetOpProviderStats(vt::OpId::kGreedyArgmax, DeviceType::kMETAL);
  vt::GreedyArgmax(q, ti, tl);
  metal.Synchronize(q);
  CHECK(DeclinesAfter(vt::OpId::kGreedyArgmax) == 0);

  std::vector<int64_t> got(static_cast<size_t>(n));
  dids.Download(got.data());
  metal.Synchronize(q);

  std::vector<float> lcpu = logits;
  std::vector<int64_t> ref(static_cast<size_t>(n), 0);
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  const Device cd{DeviceType::kCPU, 0};
  Tensor cl = Tensor::Contiguous(lcpu.data(), vt::DType::kF32, cd, {n, v});
  Tensor ci = Tensor::Contiguous(ref.data(), vt::DType::kI64, cd, {n});
  vt::GreedyArgmax(cq, ci, cl);

  CHECK(got == ref);
  CHECK(got[0] == 31337);  // the LOWER of the two tied maxima
  CHECK(got[1] == 0);      // an all-equal row resolves to index 0
  metal.DestroyQueue(q);
}

TEST_CASE("Metal kPagedAttention matches the CPU oracle within NMSE <= 5e-4") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  // OPT-125m's real attention geometry: 12 heads, head_dim 64, multi-head (no
  // GQA). TWO requests with DIFFERENT query lengths and a non-zero context on
  // the second, so the causal-offset arithmetic is genuinely exercised — a
  // single full-prefill request would not distinguish it.
  const int64_t nblocks = 12, bsz = 16, hq = 12, hkv = 12, dh = 64;
  const int64_t num_reqs = 2;
  const std::vector<int32_t> qsl{0, 20, 24};  // req0: 20 query tokens, req1: 4
  const std::vector<int32_t> slens{20, 37};   // req1 carries 33 context tokens
  const int64_t t_total = qsl.back();
  const int64_t max_blocks = 4;
  std::vector<int32_t> btab(static_cast<size_t>(num_reqs * max_blocks));
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t c = 0; c < max_blocks; ++c) {
      btab[static_cast<size_t>(r * max_blocks + c)] = static_cast<int32_t>(r * max_blocks + c);
    }
  }

  std::mt19937 rng(23);
  std::uniform_real_distribution<float> ud(-1.5f, 1.5f);
  const size_t cache_elems = static_cast<size_t>(nblocks * bsz * hkv * dh);
  std::vector<float> qf(static_cast<size_t>(t_total * hq * dh)), kf(cache_elems), vf(cache_elems);
  for (auto& x : qf) x = Bf16RT(ud(rng));
  for (auto& x : kf) x = Bf16RT(ud(rng));
  for (auto& x : vf) x = Bf16RT(ud(rng));
  const std::vector<uint16_t> qb = PackBf16(qf), kb = PackBf16(kf), vb = PackBf16(vf);

  MBuf dqy(metal, q, qb.size() * 2), dkc(metal, q, kb.size() * 2), dvc(metal, q, vb.size() * 2),
      dbt(metal, q, btab.size() * 4), dsl(metal, q, slens.size() * 4),
      dqsl(metal, q, qsl.size() * 4), dout(metal, q, qb.size() * 2);
  dqy.Upload(qb.data());
  dkc.Upload(kb.data());
  dvc.Upload(vb.data());
  dbt.Upload(btab.data());
  dsl.Upload(slens.data());
  dqsl.Upload(qsl.data());
  dout.PoisonNaN(2);
  metal.Synchronize(q);

  Tensor tq = Tensor::Contiguous(dqy.ptr(), vt::DType::kBF16, d, {t_total, hq, dh});
  Tensor tkc = Tensor::Contiguous(dkc.ptr(), vt::DType::kBF16, d, {nblocks, bsz, hkv, dh});
  Tensor tvc = Tensor::Contiguous(dvc.ptr(), vt::DType::kBF16, d, {nblocks, bsz, hkv, dh});
  Tensor tbt = Tensor::Contiguous(dbt.ptr(), vt::DType::kI32, d, {num_reqs, max_blocks});
  Tensor tsl = Tensor::Contiguous(dsl.ptr(), vt::DType::kI32, d, {num_reqs});
  Tensor tqsl = Tensor::Contiguous(dqsl.ptr(), vt::DType::kI32, d, {num_reqs + 1});
  Tensor tout = Tensor::Contiguous(dout.ptr(), vt::DType::kBF16, d, {t_total, hq, dh});

  vt::PagedAttentionArgs pa{1.0f / std::sqrt(static_cast<float>(dh)), true};
  pa.query_start_loc_host = qsl.data();
  pa.max_seq_len = 37;
  vt::ResetOpProviderStats(vt::OpId::kPagedAttention, DeviceType::kMETAL);
  vt::PagedAttention(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, pa);
  metal.Synchronize(q);
  CHECK(DeclinesAfter(vt::OpId::kPagedAttention) == 0);

  std::vector<uint16_t> gpacked(qb.size());
  dout.Download(gpacked.data());
  metal.Synchronize(q);
  std::vector<float> got(gpacked.size());
  for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(gpacked[i]);

  // The NaN poison must be gone from EVERY element — proof the kernel wrote the
  // whole output, not just the elements a lenient aggregate NMSE would forgive.
  for (float x : got) REQUIRE(std::isfinite(x));

  std::vector<uint16_t> qcpu = qb, kcpu = kb, vcpu = vb, rpacked(qb.size(), 0);
  std::vector<int32_t> bcpu = btab, scpu = slens, qscpu = qsl;
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  const Device cd{DeviceType::kCPU, 0};
  Tensor cqt = Tensor::Contiguous(qcpu.data(), vt::DType::kBF16, cd, {t_total, hq, dh});
  Tensor ckc = Tensor::Contiguous(kcpu.data(), vt::DType::kBF16, cd, {nblocks, bsz, hkv, dh});
  Tensor cvc = Tensor::Contiguous(vcpu.data(), vt::DType::kBF16, cd, {nblocks, bsz, hkv, dh});
  Tensor cbt = Tensor::Contiguous(bcpu.data(), vt::DType::kI32, cd, {num_reqs, max_blocks});
  Tensor csl = Tensor::Contiguous(scpu.data(), vt::DType::kI32, cd, {num_reqs});
  Tensor cqsl = Tensor::Contiguous(qscpu.data(), vt::DType::kI32, cd, {num_reqs + 1});
  Tensor cout = Tensor::Contiguous(rpacked.data(), vt::DType::kBF16, cd, {t_total, hq, dh});
  vt::PagedAttention(cq, cout, cqt, ckc, cvc, cbt, csl, cqsl, pa);

  std::vector<float> ref(rpacked.size());
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = vt::BF16ToF32(rpacked[i]);

  const double nmse = Nmse(got, ref);
  MESSAGE("Metal kPagedAttention NMSE vs the CPU oracle = "
          << nmse
          << " (bar 5e-4; the online-softmax form vs the materialized 3-pass "
             "reference, so bit-exactness is NOT claimed)");
  CHECK(nmse <= 5e-4);
  metal.DestroyQueue(q);
}

// Qwen3-dense attention geometry, which the OPT-shaped case above does NOT
// reach: GQA (qpk=2, so h/qpk indexes a SHARED kv head) and head_dim 128. The
// mma prefill kernel splits its output over TWO column halves of 64, and at
// head_dim 64 the second half is entirely out of range and discarded — so that
// test exercises exactly half of it. Query lengths 40 and 5 also force a PARTIAL
// second query tile (40 = 32 + 8) and a partial key block.
TEST_CASE("Metal kPagedAttention matches the CPU oracle at Qwen3 geometry (GQA, head_dim 128)") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const int64_t nblocks = 24, bsz = 16, hq = 16, hkv = 8, dh = 128;
  const int64_t num_reqs = 2;
  const std::vector<int32_t> qsl{0, 40, 45};
  const std::vector<int32_t> slens{40, 71};  // req1 carries 66 context tokens
  const int64_t t_total = qsl.back();
  const int64_t max_blocks = 6;
  std::vector<int32_t> btab(static_cast<size_t>(num_reqs * max_blocks));
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t c = 0; c < max_blocks; ++c) {
      btab[static_cast<size_t>(r * max_blocks + c)] = static_cast<int32_t>(r * max_blocks + c);
    }
  }

  std::mt19937 rng(4127);
  std::uniform_real_distribution<float> ud(-1.5f, 1.5f);
  const size_t cache_elems = static_cast<size_t>(nblocks * bsz * hkv * dh);
  std::vector<float> qf(static_cast<size_t>(t_total * hq * dh)), kf(cache_elems), vf(cache_elems);
  for (auto& x : qf) x = Bf16RT(ud(rng));
  for (auto& x : kf) x = Bf16RT(ud(rng));
  for (auto& x : vf) x = Bf16RT(ud(rng));
  const std::vector<uint16_t> qb = PackBf16(qf), kb = PackBf16(kf), vb = PackBf16(vf);

  MBuf dqy(metal, q, qb.size() * 2), dkc(metal, q, kb.size() * 2), dvc(metal, q, vb.size() * 2),
      dbt(metal, q, btab.size() * 4), dsl(metal, q, slens.size() * 4),
      dqsl(metal, q, qsl.size() * 4), dout(metal, q, qb.size() * 2);
  dqy.Upload(qb.data());
  dkc.Upload(kb.data());
  dvc.Upload(vb.data());
  dbt.Upload(btab.data());
  dsl.Upload(slens.data());
  dqsl.Upload(qsl.data());
  dout.PoisonNaN(2);
  metal.Synchronize(q);

  Tensor tq = Tensor::Contiguous(dqy.ptr(), vt::DType::kBF16, d, {t_total, hq, dh});
  Tensor tkc = Tensor::Contiguous(dkc.ptr(), vt::DType::kBF16, d, {nblocks, bsz, hkv, dh});
  Tensor tvc = Tensor::Contiguous(dvc.ptr(), vt::DType::kBF16, d, {nblocks, bsz, hkv, dh});
  Tensor tbt = Tensor::Contiguous(dbt.ptr(), vt::DType::kI32, d, {num_reqs, max_blocks});
  Tensor tsl = Tensor::Contiguous(dsl.ptr(), vt::DType::kI32, d, {num_reqs});
  Tensor tqsl = Tensor::Contiguous(dqsl.ptr(), vt::DType::kI32, d, {num_reqs + 1});
  Tensor tout = Tensor::Contiguous(dout.ptr(), vt::DType::kBF16, d, {t_total, hq, dh});

  vt::PagedAttentionArgs pa{1.0f / std::sqrt(static_cast<float>(dh)), true};
  pa.query_start_loc_host = qsl.data();
  pa.max_seq_len = 71;
  vt::ResetOpProviderStats(vt::OpId::kPagedAttention, DeviceType::kMETAL);
  vt::PagedAttention(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, pa);
  metal.Synchronize(q);
  CHECK(DeclinesAfter(vt::OpId::kPagedAttention) == 0);

  std::vector<uint16_t> gpacked(qb.size());
  dout.Download(gpacked.data());
  metal.Synchronize(q);
  std::vector<float> got(gpacked.size());
  for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(gpacked[i]);
  for (float x : got) REQUIRE(std::isfinite(x));

  std::vector<uint16_t> qcpu = qb, kcpu = kb, vcpu = vb, rpacked(qb.size(), 0);
  std::vector<int32_t> bcpu = btab, scpu = slens, qscpu = qsl;
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  const Device cd{DeviceType::kCPU, 0};
  Tensor cqt = Tensor::Contiguous(qcpu.data(), vt::DType::kBF16, cd, {t_total, hq, dh});
  Tensor ckc = Tensor::Contiguous(kcpu.data(), vt::DType::kBF16, cd, {nblocks, bsz, hkv, dh});
  Tensor cvc = Tensor::Contiguous(vcpu.data(), vt::DType::kBF16, cd, {nblocks, bsz, hkv, dh});
  Tensor cbt = Tensor::Contiguous(bcpu.data(), vt::DType::kI32, cd, {num_reqs, max_blocks});
  Tensor csl = Tensor::Contiguous(scpu.data(), vt::DType::kI32, cd, {num_reqs});
  Tensor cqsl = Tensor::Contiguous(qscpu.data(), vt::DType::kI32, cd, {num_reqs + 1});
  Tensor cout = Tensor::Contiguous(rpacked.data(), vt::DType::kBF16, cd, {t_total, hq, dh});
  vt::PagedAttention(cq, cout, cqt, ckc, cvc, cbt, csl, cqsl, pa);

  std::vector<float> ref(rpacked.size());
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = vt::BF16ToF32(rpacked[i]);

  // Worst SINGLE element, not just the aggregate: a wrong column half or a
  // mis-set kv group shows up in a slice that an averaged NMSE can bury.
  double worst = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double e = std::abs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
    if (e > worst) { worst = e; worst_i = i; }
  }
  const double nmse = Nmse(got, ref);
  MESSAGE("Metal kPagedAttention (Qwen3 geom) NMSE = " << nmse << ", worst |elem| err = " << worst
          << " at flat index " << worst_i << " (head " << (worst_i / dh) % hq << ", col "
          << worst_i % dh << ")");
  CHECK(nmse <= 5e-4);
  CHECK(worst <= 5e-2);
  metal.DestroyQueue(q);
}

// Settles whether decode attention's measured ~29 GB/s is a LAYOUT effect.
// Reads a fixed number of USEFUL bytes with the stride varied: stride == chunk is
// a contiguous stream, stride == 8*chunk is exactly the paged KV cache's
// single-head pattern ([block][slot][kv_head][dim] with 8 kv heads, so d
// elements every 8*d). Opt-in: VT_BW_PROBE=1.
TEST_CASE("Metal strided-read bandwidth probe" * doctest::skip(true)) {
  if (std::getenv("VT_BW_PROBE") == nullptr) return;
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();

  const uint32_t chunk_f4 = 16;              // 256 B, one head's d=128 bf16 row
  const uint32_t n_chunks = 32768;           // 8 MB of USEFUL bytes
  const size_t useful = static_cast<size_t>(n_chunks) * chunk_f4 * 16;
  for (uint32_t mult : {1u, 2u, 4u, 8u, 16u}) {
    const uint32_t stride_f4 = chunk_f4 * mult;
    const size_t span = static_cast<size_t>(n_chunks) * stride_f4 * 16;
    void* src = metal.Alloc(span);
    void* dst = metal.Alloc((n_chunks + 64) * sizeof(float));
    std::memset(src, 0, span);
    metal.Synchronize(q);
    // Warm, then time a batch of repeats.
    vt::metal::BandwidthProbe(q, src, dst, n_chunks, chunk_f4, stride_f4);
    metal.Synchronize(q);
    const int reps = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) vt::metal::BandwidthProbe(q, src, dst, n_chunks, chunk_f4, stride_f4);
    metal.Synchronize(q);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double gbs = static_cast<double>(useful) * reps / sec / 1e9;
    MESSAGE("stride x" << mult << " (useful " << (useful >> 20) << " MB, span "
            << (span >> 20) << " MB): " << gbs << " GB/s of USEFUL bytes");
    metal.Free(src);
    metal.Free(dst);
  }
  metal.DestroyQueue(q);
}

// The fused qk-norm-RoPE preamble (kAttnQkNormRope) against its own byte-exact
// Tier-0 composite. This test exists because test_ops_fused_chain's Metal cases
// are CUDA-gated and skip — the same coverage hole that let an incorrect version
// of the mma attention kernel pass every existing test earlier. Metal registers
// the recipe's fast_op, so FusedChain here dispatches the FUSED kernel; the CPU
// side has no registration and runs the composite, which is the golden the
// recipe's own documentation defines.
TEST_CASE("Metal fused qk-norm-RoPE matches the CPU composite") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  // Qwen3 geometry with GQA, plus a partial rotation (rot < Dh) so the
  // un-rotated tail is exercised too.
  const int64_t T = 5, Hq = 16, Hkv = 8, Dh = 128, rot = 64, maxpos = 64;
  const int64_t qn = T * Hq * Dh, kn = T * Hkv * Dh;

  std::mt19937 rng(90210);
  std::uniform_real_distribution<float> ud(-1.5f, 1.5f);
  std::vector<float> qh(static_cast<size_t>(qn)), kh(static_cast<size_t>(kn));
  std::vector<float> wq(static_cast<size_t>(Dh)), wk(static_cast<size_t>(Dh));
  std::vector<float> cs(static_cast<size_t>(maxpos * rot));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (auto& x : qh) x = ud(rng);
  for (auto& x : kh) x = ud(rng);
  for (auto& x : wq) x = ud(rng);
  for (auto& x : wk) x = ud(rng);
  for (auto& x : cs) x = ud(rng);
  for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i * 3 + 1);

  MBuf dq(metal, q, qh.size() * 4), dk(metal, q, kh.size() * 4), dwq(metal, q, wq.size() * 4),
      dwk(metal, q, wk.size() * 4), dcs(metal, q, cs.size() * 4), dpos(metal, q, pos.size() * 4);
  dq.Upload(qh.data()); dk.Upload(kh.data()); dwq.Upload(wq.data());
  dwk.Upload(wk.data()); dcs.Upload(cs.data()); dpos.Upload(pos.data());
  metal.Synchronize(q);

  Tensor gq2 = Tensor::Contiguous(dq.ptr(), vt::DType::kF32, d, {T * Hq, Dh});
  Tensor gk2 = Tensor::Contiguous(dk.ptr(), vt::DType::kF32, d, {T * Hkv, Dh});
  Tensor gq3 = Tensor::Contiguous(dq.ptr(), vt::DType::kF32, d, {T, Hq, Dh});
  Tensor gk3 = Tensor::Contiguous(dk.ptr(), vt::DType::kF32, d, {T, Hkv, Dh});
  Tensor gwq = Tensor::Contiguous(dwq.ptr(), vt::DType::kF32, d, {Dh});
  Tensor gwk = Tensor::Contiguous(dwk.ptr(), vt::DType::kF32, d, {Dh});
  Tensor gcs = Tensor::Contiguous(dcs.ptr(), vt::DType::kF32, d, {maxpos, rot});
  Tensor gpo = Tensor::Contiguous(dpos.ptr(), vt::DType::kI32, d, {T});

  vt::FusedBinding gb;
  gb.op[0] = &gq2; gb.op[1] = &gwq; gb.op[2] = &gk2; gb.op[3] = &gwk;
  gb.op[4] = &gq3; gb.op[5] = &gk3; gb.op[6] = &gcs; gb.op[7] = &gpo;
  gb.n = 8;
  vt::FusedParams gp;
  gp.eps = 1e-6f;
  gp.rope = vt::RopeArgs{10000.0f, static_cast<int>(rot)};
  // Proof the FUSED path is what ran: Metal registers kAttnQkNormRope, so this
  // must take DispatchFusedFast, not the composite.
  REQUIRE(vt::OpRegistered(vt::OpId::kAttnQkNormRope, DeviceType::kMETAL));
  vt::FusedChain(q, vt::kAttnQkNormRope, gb, gp);
  metal.Synchronize(q);

  std::vector<float> gotq(static_cast<size_t>(qn)), gotk(static_cast<size_t>(kn));
  dq.Download(gotq.data());
  dk.Download(gotk.data());
  metal.Synchronize(q);

  // CPU composite on identical inputs.
  std::vector<float> cq = qh, ck = kh, cwq = wq, cwk = wk, ccs = cs;
  std::vector<int32_t> cpos = pos;
  Queue cq_(Queue{Device{DeviceType::kCPU, 0}, nullptr});
  const Device cd{DeviceType::kCPU, 0};
  Tensor cq2 = Tensor::Contiguous(cq.data(), vt::DType::kF32, cd, {T * Hq, Dh});
  Tensor ck2 = Tensor::Contiguous(ck.data(), vt::DType::kF32, cd, {T * Hkv, Dh});
  Tensor cq3 = Tensor::Contiguous(cq.data(), vt::DType::kF32, cd, {T, Hq, Dh});
  Tensor ck3 = Tensor::Contiguous(ck.data(), vt::DType::kF32, cd, {T, Hkv, Dh});
  Tensor cwqt = Tensor::Contiguous(cwq.data(), vt::DType::kF32, cd, {Dh});
  Tensor cwkt = Tensor::Contiguous(cwk.data(), vt::DType::kF32, cd, {Dh});
  Tensor ccst = Tensor::Contiguous(ccs.data(), vt::DType::kF32, cd, {maxpos, rot});
  Tensor cpot = Tensor::Contiguous(cpos.data(), vt::DType::kI32, cd, {T});
  vt::FusedBinding cb;
  cb.op[0] = &cq2; cb.op[1] = &cwqt; cb.op[2] = &ck2; cb.op[3] = &cwkt;
  cb.op[4] = &cq3; cb.op[5] = &ck3; cb.op[6] = &ccst; cb.op[7] = &cpot;
  cb.n = 8;
  vt::FusedChain(cq_, vt::kAttnQkNormRope, cb, gp);

  double worst = 0.0;
  for (size_t i = 0; i < gotq.size(); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(gotq[i]) - static_cast<double>(cq[i])));
  for (size_t i = 0; i < gotk.size(); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(gotk[i]) - static_cast<double>(ck[i])));
  MESSAGE("Metal fused qk-norm-RoPE vs CPU composite: worst |elem| err = " << worst);
  // f32 throughout on both sides; only the reduction order differs.
  CHECK(worst <= 1e-4);
  metal.DestroyQueue(q);
}

// M3b — the two ops Qwen3-dense adds to OPT's Metal set. Both compute f32
// transcendentals in-kernel (Metal has no double), so the bar is NMSE <= 5e-4 vs
// the CPU oracle, NOT bit-exactness — the same posture as kPagedAttention.
TEST_CASE("Metal kRopeNeox matches the CPU oracle within NMSE <= 5e-4") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const int64_t t = 7, hq = 4, hk = 2, dh = 16, rot = 16;
  const float base = 1.0e6f;

  std::mt19937 rng(23);
  std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
  std::vector<float> qf(static_cast<size_t>(t * hq * dh)), kf(static_cast<size_t>(t * hk * dh));
  for (auto& x : qf) x = Bf16RT(ud(rng));
  for (auto& x : kf) x = Bf16RT(ud(rng));
  const std::vector<uint16_t> qb = PackBf16(qf), kb = PackBf16(kf);
  // A spread of positions incl. larger angles (worst case for f32 range reduction).
  std::vector<int32_t> pos{0, 1, 2, 5, 9, 13, 20};
  REQUIRE(static_cast<int64_t>(pos.size()) == t);

  MBuf dq(metal, q, qb.size() * 2), dk(metal, q, kb.size() * 2), dpos(metal, q, pos.size() * 4);
  dq.Upload(qb.data());
  dk.Upload(kb.data());
  dpos.Upload(pos.data());
  metal.Synchronize(q);

  Tensor tq = Tensor::Contiguous(dq.ptr(), vt::DType::kBF16, d, {t, hq, dh});
  Tensor tk = Tensor::Contiguous(dk.ptr(), vt::DType::kBF16, d, {t, hk, dh});
  Tensor tpos = Tensor::Contiguous(dpos.ptr(), vt::DType::kI32, d, {t});
  vt::ResetOpProviderStats(vt::OpId::kRopeNeox, DeviceType::kMETAL);
  vt::RopeNeox(q, tq, tk, tpos, vt::RopeArgs{base, static_cast<int>(rot)});
  metal.Synchronize(q);
  CHECK(DeclinesAfter(vt::OpId::kRopeNeox) == 0);

  std::vector<uint16_t> gq(qb.size()), gk(kb.size());
  dq.Download(gq.data());
  dk.Download(gk.data());
  metal.Synchronize(q);
  std::vector<float> got;
  got.reserve(gq.size() + gk.size());
  for (uint16_t x : gq) got.push_back(vt::BF16ToF32(x));
  for (uint16_t x : gk) got.push_back(vt::BF16ToF32(x));
  for (float x : got) REQUIRE(std::isfinite(x));

  std::vector<uint16_t> qcpu = qb, kcpu = kb;
  std::vector<int32_t> pcpu = pos;
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  const Device cd{DeviceType::kCPU, 0};
  Tensor cq_t = Tensor::Contiguous(qcpu.data(), vt::DType::kBF16, cd, {t, hq, dh});
  Tensor ck_t = Tensor::Contiguous(kcpu.data(), vt::DType::kBF16, cd, {t, hk, dh});
  Tensor cp_t = Tensor::Contiguous(pcpu.data(), vt::DType::kI32, cd, {t});
  vt::RopeNeox(cq, cq_t, ck_t, cp_t, vt::RopeArgs{base, static_cast<int>(rot)});
  std::vector<float> ref;
  ref.reserve(qcpu.size() + kcpu.size());
  for (uint16_t x : qcpu) ref.push_back(vt::BF16ToF32(x));
  for (uint16_t x : kcpu) ref.push_back(vt::BF16ToF32(x));

  const double nmse = Nmse(got, ref);
  MESSAGE("Metal kRopeNeox NMSE vs the CPU oracle = " << nmse
          << " (bar 5e-4; f32 pow/cos/sin vs the reference's fp64, so bit-exactness "
             "is NOT claimed)");
  CHECK(nmse <= 5e-4);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal kRopeCosSinCache matches the CPU oracle within NMSE <= 5e-4") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const int64_t t = 7, rot = 16;
  const float base = 1.0e6f;
  std::vector<int32_t> pos{0, 1, 2, 5, 9, 13, 20};
  REQUIRE(static_cast<int64_t>(pos.size()) == t);

  MBuf dcs(metal, q, static_cast<size_t>(t * rot) * 4), dpos(metal, q, pos.size() * 4);
  dcs.PoisonNaN(4);
  dpos.Upload(pos.data());
  metal.Synchronize(q);

  Tensor tcs = Tensor::Contiguous(dcs.ptr(), vt::DType::kF32, d, {t, rot});
  Tensor tpos = Tensor::Contiguous(dpos.ptr(), vt::DType::kI32, d, {t});
  vt::ResetOpProviderStats(vt::OpId::kRopeCosSinCache, DeviceType::kMETAL);
  vt::RopeCosSinCache(q, tcs, tpos, vt::RopeArgs{base, static_cast<int>(rot)});
  metal.Synchronize(q);
  CHECK(DeclinesAfter(vt::OpId::kRopeCosSinCache) == 0);

  std::vector<float> got(static_cast<size_t>(t * rot));
  dcs.Download(got.data());
  metal.Synchronize(q);
  for (float x : got) REQUIRE(std::isfinite(x));

  std::vector<int32_t> pcpu = pos;
  std::vector<float> ref(static_cast<size_t>(t * rot), 0.0f);
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  const Device cd{DeviceType::kCPU, 0};
  Tensor ccs = Tensor::Contiguous(ref.data(), vt::DType::kF32, cd, {t, rot});
  Tensor cp_t = Tensor::Contiguous(pcpu.data(), vt::DType::kI32, cd, {t});
  vt::RopeCosSinCache(cq, ccs, cp_t, vt::RopeArgs{base, static_cast<int>(rot)});

  const double nmse = Nmse(got, ref);
  MESSAGE("Metal kRopeCosSinCache NMSE vs the CPU oracle = " << nmse << " (bar 5e-4)");
  CHECK(nmse <= 5e-4);
  metal.DestroyQueue(q);
}

// M3b — kRopeFromCache is the op Qwen3-dense actually dispatches on the default
// path (VT_QWEN3_ROPE_CACHE ON). It reads cos|sin from the cache (no in-kernel
// transcendentals) and only rotates, so it is BIT-EXACT vs the CPU oracle.
TEST_CASE("Metal kRopeFromCache is BIT-EXACT vs the CPU oracle") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const int64_t t = 7, hq = 4, hk = 2, dh = 16, rot = 16, half = rot / 2;
  const int64_t P = 40;  // cache rows (positions)
  const float base = 1.0e6f;

  std::mt19937 rng(29);
  std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
  std::vector<float> qf(static_cast<size_t>(t * hq * dh)), kf(static_cast<size_t>(t * hk * dh));
  for (auto& x : qf) x = Bf16RT(ud(rng));
  for (auto& x : kf) x = Bf16RT(ud(rng));
  // Build a bf16 cos|sin cache [P, rot] the same way RopeCosSinCache does (fp64
  // angle math), so this test isolates the APPLY op from the cache-build op.
  std::vector<float> cache_f(static_cast<size_t>(P * rot));
  for (int64_t p = 0; p < P; ++p) {
    for (int64_t i = 0; i < half; ++i) {
      const double freq = std::pow(static_cast<double>(base), -2.0 * static_cast<double>(i) / rot);
      const double angle = static_cast<double>(p) * freq;
      cache_f[static_cast<size_t>(p * rot + i)] = Bf16RT(static_cast<float>(std::cos(angle)));
      cache_f[static_cast<size_t>(p * rot + half + i)] = Bf16RT(static_cast<float>(std::sin(angle)));
    }
  }
  const std::vector<uint16_t> qb = PackBf16(qf), kb = PackBf16(kf), cb = PackBf16(cache_f);
  // Identity row index (Qwen3's si.rope_row_idx): token t reads cache row t.
  std::vector<int32_t> pos{0, 1, 2, 3, 4, 5, 6};
  REQUIRE(static_cast<int64_t>(pos.size()) == t);

  MBuf dq(metal, q, qb.size() * 2), dk(metal, q, kb.size() * 2),
      dpos(metal, q, pos.size() * 4), dcache(metal, q, cb.size() * 2);
  dq.Upload(qb.data());
  dk.Upload(kb.data());
  dpos.Upload(pos.data());
  dcache.Upload(cb.data());
  metal.Synchronize(q);

  Tensor tq = Tensor::Contiguous(dq.ptr(), vt::DType::kBF16, d, {t, hq, dh});
  Tensor tk = Tensor::Contiguous(dk.ptr(), vt::DType::kBF16, d, {t, hk, dh});
  Tensor tpos = Tensor::Contiguous(dpos.ptr(), vt::DType::kI32, d, {t});
  Tensor tcache = Tensor::Contiguous(dcache.ptr(), vt::DType::kBF16, d, {P, rot});
  vt::ResetOpProviderStats(vt::OpId::kRopeFromCache, DeviceType::kMETAL);
  vt::RopeFromCache(q, tq, &tk, tpos, tcache, vt::RopeArgs{base, static_cast<int>(rot)});
  metal.Synchronize(q);
  CHECK(DeclinesAfter(vt::OpId::kRopeFromCache) == 0);

  std::vector<uint16_t> gq(qb.size()), gk(kb.size());
  dq.Download(gq.data());
  dk.Download(gk.data());
  metal.Synchronize(q);

  std::vector<uint16_t> qcpu = qb, kcpu = kb, ccpu = cb;
  std::vector<int32_t> pcpu = pos;
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  const Device cd{DeviceType::kCPU, 0};
  Tensor cq_t = Tensor::Contiguous(qcpu.data(), vt::DType::kBF16, cd, {t, hq, dh});
  Tensor ck_t = Tensor::Contiguous(kcpu.data(), vt::DType::kBF16, cd, {t, hk, dh});
  Tensor cp_t = Tensor::Contiguous(pcpu.data(), vt::DType::kI32, cd, {t});
  Tensor cc_t = Tensor::Contiguous(ccpu.data(), vt::DType::kBF16, cd, {P, rot});
  vt::RopeFromCache(cq, cq_t, &ck_t, cp_t, cc_t, vt::RopeArgs{base, static_cast<int>(rot)});

  // No in-kernel transcendental and no reduction, so the bits must be IDENTICAL.
  CHECK(gq == qcpu);
  CHECK(gk == kcpu);
  metal.DestroyQueue(q);
}

// ===========================================================================
// VT_METAL_PROFILE — the dispatch attribution facility itself
// (.agents/specs/metal-dispatch-attribution.md). Instruments' Metal System
// Trace needs a full Xcode that the project's Apple box does not have, so this
// is the only execution trace available there; if it silently stopped counting,
// every future Metal perf claim would lose its evidence base. Hence a test.
TEST_CASE("Metal dispatch profile attributes host encode, wait and GPU busy") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};

  const bool was_on = vt::metal::ProfileEnabled();
  vt::metal::ResetProfile();
  vt::metal::SetProfileEnabled(false);

  // OFF: a dispatch must leave no row. This is the "costs nothing when off"
  // half of the contract, asserted rather than assumed.
  const int64_t n = 4096;
  float* x = static_cast<float*>(metal.Alloc(n * sizeof(float)));
  std::vector<float> host(static_cast<size_t>(n), -1.0f);
  metal.Copy(q, x, host.data(), host.size() * sizeof(float));
  Tensor t = Tensor::Contiguous(x, vt::DType::kF32, d, {n});
  vt::Relu(q, t, t);
  metal.Synchronize(q);
  CHECK(vt::metal::GetProfileRows().empty());

  // ON: the same dispatch must produce exactly one named row plus the total.
  vt::metal::SetProfileEnabled(true);
  vt::Relu(q, t, t);
  metal.Synchronize(q);  // M3c-1: the commit happens HERE, not in the op
  std::vector<vt::metal::ProfileRow> rows = vt::metal::GetProfileRows();
  REQUIRE(rows.size() >= 2);
  const vt::metal::ProfileRow& total = rows.back();
  CHECK(total.name.empty());
  CHECK(total.count == 1);
  // `count` is DISPATCHES; since M3c-1 batched them, wait and GPU time are
  // properties of the COMMIT and appear on the total row only. Every phase must
  // be a real, non-negative duration, and the GPU can never have been busy
  // LONGER than the wall time we blocked for. That ordering is the property the
  // whole attribution rests on: if it inverted, the gpu/wait ratio driving the
  // optimisation decisions would be meaningless.
  CHECK(total.encode_s >= 0.0);
  CHECK(total.wait_s > 0.0);
  CHECK(total.gpu_s >= 0.0);
  CHECK(total.gpu_s <= total.wait_s);
  bool saw_relu = false;
  for (const vt::metal::ProfileRow& r : rows) {
    if (r.name == "vt_relu") saw_relu = true;
  }
  CHECK(saw_relu);

  vt::metal::ResetProfile();
  CHECK(vt::metal::GetProfileRows().empty());
  vt::metal::SetProfileEnabled(was_on);
  metal.Free(x);
  metal.DestroyQueue(q);
}

// ===========================================================================
// M3c-1 — batched encoders. Attribution measured 50,944 command buffers for a
// 128-token generation, each costing a ~186 us commit+wait round trip that is
// 33% of total runtime (.agents/specs/metal-dispatch-attribution.md). These
// pin the contract that makes that cost collapsible: many dispatches share ONE
// command buffer, and every path that lets the HOST observe device memory
// flushes first. The dispatch count alone cannot distinguish batched from
// serialised, which is why these assert on the COMMIT count.
TEST_CASE("Metal batches many dispatches into ONE command buffer per flush") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};

  const bool was_on = vt::metal::ProfileEnabled();
  vt::metal::SetProfileEnabled(true);
  vt::metal::ResetProfile();

  const int64_t n = 1024;
  float* x = static_cast<float*>(metal.Alloc(n * sizeof(float)));
  std::vector<float> host(static_cast<size_t>(n), 1.0f);
  metal.Copy(q, x, host.data(), host.size() * sizeof(float));
  Tensor t = Tensor::Contiguous(x, vt::DType::kF32, d, {n});

  // Eight independent dispatches with NO synchronisation between them.
  const int kOps = 8;
  vt::metal::ResetProfile();
  for (int i = 0; i < kOps; ++i) vt::Relu(q, t, t);

  // Nothing forced a flush yet, so nothing may have been committed.
  CHECK(vt::metal::GetProfileCommits() == 0);

  metal.Synchronize(q);

  // Synchronize is the flush point: exactly ONE command buffer for all eight.
  CHECK(vt::metal::GetProfileCommits() == 1);
  std::vector<vt::metal::ProfileRow> rows = vt::metal::GetProfileRows();
  REQUIRE(!rows.empty());
  CHECK(rows.back().count == static_cast<unsigned long long>(kOps));

  vt::metal::SetProfileEnabled(was_on);
  vt::metal::ResetProfile();
  metal.Free(x);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal flushes pending work before the host can observe device memory") {
  // The correctness half. Batching is only safe if every host-visible read
  // drains the queue first; otherwise `Copy` hands back stale bytes and the
  // failure is silent, intermittent and data-dependent. Deliberately NO
  // explicit Synchronize: Copy itself must be a flush point.
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};

  const int64_t n = 512;
  float* x = static_cast<float*>(metal.Alloc(n * sizeof(float)));
  std::vector<float> host(static_cast<size_t>(n), -3.0f);
  metal.Copy(q, x, host.data(), host.size() * sizeof(float));
  Tensor t = Tensor::Contiguous(x, vt::DType::kF32, d, {n});

  vt::Relu(q, t, t);  // -3 -> 0, entirely on the GPU, never synchronised

  std::vector<float> back(static_cast<size_t>(n), 12345.0f);
  metal.Copy(q, back.data(), x, back.size() * sizeof(float));
  for (int64_t i = 0; i < n; ++i) {
    REQUIRE(back[static_cast<size_t>(i)] == 0.0f);
  }

  metal.Free(x);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal chains batched dispatches in order without intermediate sync") {
  // Ordering inside one command buffer. A compute encoder created with the
  // default MTLDispatchTypeSerial serialises its dispatches, so a read-modify
  // -write chain must still compose exactly. If batching ever moved to
  // concurrent dispatch without explicit barriers, THIS is what would break,
  // and it would break as wrong numbers rather than as a crash.
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};

  const int64_t n = 256;
  float* x = static_cast<float*>(metal.Alloc(n * sizeof(float)));
  std::vector<float> host(static_cast<size_t>(n), 2.0f);
  metal.Copy(q, x, host.data(), host.size() * sizeof(float));
  Tensor t = Tensor::Contiguous(x, vt::DType::kF32, d, {n});

  // x += x, three times, no sync between: 2 -> 4 -> 8 -> 16.
  vt::Add(q, t, t, t);
  vt::Add(q, t, t, t);
  vt::Add(q, t, t, t);

  std::vector<float> back(static_cast<size_t>(n), 0.0f);
  metal.Copy(q, back.data(), x, back.size() * sizeof(float));
  for (int64_t i = 0; i < n; ++i) {
    REQUIRE(back[static_cast<size_t>(i)] == 16.0f);
  }

  metal.Free(x);
  metal.DestroyQueue(q);
}

// ===========================================================================
// M3d — the decode GEMV fast path. Shape-class profiling showed 21,464 of the
// 21,632 matmuls in a 128-token generation are m=1 (decode) and ALL take the BT
// orientation; only 168 are prefill GEMMs. The 16x16 tile kernel wastes 15 of
// every 16 threadgroup rows on m=1, so decode gets its own kernel. BT is what
// makes it worth doing: B row `col` is contiguous over k, so a simdgroup can
// read it fully coalesced and reduce with simd_sum.
TEST_CASE("Metal routes m=1 BT matmul to the GEMV kernel and matches the CPU oracle") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};

  // Ragged K included deliberately: the GEMV strides k by the simd width, so a
  // K that is not a multiple of it exercises the remainder tail, which is where
  // a hand-rolled reduction loop goes wrong.
  struct Case { const char* name; int64_t k, n; };
  const Case cases[] = {
      {"decode 1x2048x2048", 2048, 2048},
      {"mlp-width 1x2048x6144", 2048, 6144},
      {"ragged K 1x1000x777", 1000, 777},
  };
  // An f32 arm FIRST, because the bf16 arms below cannot discriminate kernels:
  // at bf16 output the NMSE is dominated by store rounding (~1.7e-3 relative,
  // i.e. ~2.8e-6 NMSE) and two kernels with different accumulation orders land
  // on identical bf16 values. Only f32 exposes the accumulation itself.
  {
    const int64_t k = 2048, n = 512;
    std::mt19937 rng(0xF00Du);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> ah(static_cast<size_t>(k)), bh(static_cast<size_t>(k * n));
    for (auto& x : ah) x = dist(rng);
    for (auto& x : bh) x = dist(rng);
    std::vector<double> ref(static_cast<size_t>(n), 0.0);
    for (int64_t col = 0; col < n; ++col) {
      double acc = 0.0;
      for (int64_t kk = 0; kk < k; ++kk) {
        acc += double(ah[static_cast<size_t>(kk)]) * double(bh[static_cast<size_t>(col * k + kk)]);
      }
      ref[static_cast<size_t>(col)] = acc;
    }
    auto up = [&](const std::vector<float>& h) {
      void* q2 = metal.Alloc(h.size() * sizeof(float));
      metal.Copy(q, q2, h.data(), h.size() * sizeof(float));
      return q2;
    };
    void* da = up(ah);
    void* db = up(bh);
    void* dc = metal.Alloc(static_cast<size_t>(n) * sizeof(float));
    Tensor ta = Tensor::Contiguous(da, vt::DType::kF32, d, {1, k});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {n, k});
    Tensor tc = Tensor::Contiguous(dc, vt::DType::kF32, d, {1, n});
    vt::MatmulBT(q, tc, ta, tb);
    metal.Synchronize(q);
    std::vector<float> got(static_cast<size_t>(n));
    metal.Copy(q, got.data(), dc, got.size() * sizeof(float));
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double diff = double(got[i]) - ref[i];
      num += diff * diff;
      den += ref[i] * ref[i];
    }
    const double f32_nmse = den > 0.0 ? num / den : num;
    MESSAGE("GEMV f32 1x2048x512 NMSE vs f64 oracle = " << f32_nmse);
    // bf16 INPUTS with an f32 OUTPUT. This is the only arm that can see the
    // GEMV's bf16 accumulation: the all-bf16 arms below are dominated by output
    // store rounding (~2.8e-06 for any correct kernel), and the all-f32 arm
    // above does not take the bf16 code path at all. Without this, a change to
    // the bf16 reduction order is unmeasurable.
    {
      const int64_t k2 = 2048, n2 = 512;
      std::mt19937 rng2(0xBEE5u);
      std::uniform_real_distribution<float> dd(-1.0f, 1.0f);
      std::vector<float> ah2(static_cast<size_t>(k2)), bh2(static_cast<size_t>(k2 * n2));
      for (auto& x : ah2) x = Bf16RT(dd(rng2));
      for (auto& x : bh2) x = Bf16RT(dd(rng2));
      std::vector<double> ref2(static_cast<size_t>(n2), 0.0);
      for (int64_t col = 0; col < n2; ++col) {
        double acc2 = 0.0;
        for (int64_t kk = 0; kk < k2; ++kk)
          acc2 += double(ah2[static_cast<size_t>(kk)]) *
                  double(bh2[static_cast<size_t>(col * k2 + kk)]);
        ref2[static_cast<size_t>(col)] = acc2;
      }
      auto upb = [&](const std::vector<float>& h) {
        void* pp = metal.Alloc(h.size() * sizeof(uint16_t));
        std::vector<uint16_t> pk(h.size());
        for (size_t i = 0; i < h.size(); ++i) pk[i] = vt::F32ToBF16(h[i]);
        metal.Copy(q, pp, pk.data(), pk.size() * sizeof(uint16_t));
        return pp;
      };
      void* da2 = upb(ah2);
      void* db2 = upb(bh2);
      void* dc2 = metal.Alloc(static_cast<size_t>(n2) * sizeof(float));
      Tensor ta2 = Tensor::Contiguous(da2, vt::DType::kBF16, d, {1, k2});
      Tensor tb2 = Tensor::Contiguous(db2, vt::DType::kBF16, d, {n2, k2});
      Tensor tc2 = Tensor::Contiguous(dc2, vt::DType::kF32, d, {1, n2});
      vt::MatmulBT(q, tc2, ta2, tb2);
      metal.Synchronize(q);
      std::vector<float> got2(static_cast<size_t>(n2));
      metal.Copy(q, got2.data(), dc2, got2.size() * sizeof(float));
      double nu = 0.0, de = 0.0;
      for (size_t i = 0; i < got2.size(); ++i) {
        const double df = double(got2[i]) - ref2[i];
        nu += df * df;
        de += ref2[i] * ref2[i];
      }
      const double bf_nmse = de > 0.0 ? nu / de : nu;
      MESSAGE("GEMV bf16-in/f32-out 1x2048x512 NMSE vs f64 oracle = " << bf_nmse);
      CHECK(bf_nmse <= 5e-4);
      metal.Free(da2); metal.Free(db2); metal.Free(dc2);
    }
    // f32 accumulation over K=2048 should sit near 1e-14, as the tile GEMM's
    // f32 arm does (3.7e-14). Orders above that would mean a real defect, not a
    // reduction-order difference.
    CHECK(f32_nmse <= 1e-10);
    metal.Free(da);
    metal.Free(db);
    metal.Free(dc);
  }

  for (const Case& c : cases) {
    CAPTURE(c.name);
    std::mt19937 rng(0xBEEFu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> ah(static_cast<size_t>(c.k));
    std::vector<float> bh(static_cast<size_t>(c.k * c.n));
    for (auto& x : ah) x = Bf16RT(dist(rng));
    for (auto& x : bh) x = Bf16RT(dist(rng));

    // CPU oracle: f32 accumulation over K, exactly the reference math.
    std::vector<float> ref(static_cast<size_t>(c.n), 0.0f);
    for (int64_t col = 0; col < c.n; ++col) {
      float acc = 0.0f;
      for (int64_t kk = 0; kk < c.k; ++kk) {
        acc += ah[static_cast<size_t>(kk)] * bh[static_cast<size_t>(col * c.k + kk)];
      }
      ref[static_cast<size_t>(col)] = acc;
    }

    auto upload_bf16 = [&](const std::vector<float>& h) {
      void* p = metal.Alloc(h.size() * sizeof(uint16_t));
      std::vector<uint16_t> packed(h.size());
      for (size_t i = 0; i < h.size(); ++i) packed[i] = vt::F32ToBF16(h[i]);
      metal.Copy(q, p, packed.data(), packed.size() * sizeof(uint16_t));
      return p;
    };
    void* da = upload_bf16(ah);
    void* db = upload_bf16(bh);
    void* dc = metal.Alloc(static_cast<size_t>(c.n) * sizeof(uint16_t));

    Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {1, c.k});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {c.n, c.k});
    Tensor tc = Tensor::Contiguous(dc, vt::DType::kBF16, d, {1, c.n});

    const bool was_on = vt::metal::ProfileEnabled();
    vt::metal::SetProfileEnabled(true);
    vt::metal::ResetProfile();

    vt::MatmulBT(q, tc, ta, tb);
    metal.Synchronize(q);

    // PROVE THE FAST PATH RAN. A numeric check alone cannot: the tile kernel
    // computes the same answer, so a silent failure to route would look
    // identical to success.
    bool saw_gemv = false;
    for (const vt::metal::ProfileRow& r : vt::metal::GetProfileRows()) {
      if (r.name == "vt_matmul_bt_gemv") saw_gemv = true;
    }
    CHECK(saw_gemv);
    vt::metal::SetProfileEnabled(was_on);
    vt::metal::ResetProfile();

    std::vector<uint16_t> packed_out(static_cast<size_t>(c.n));
    metal.Copy(q, packed_out.data(), dc, packed_out.size() * sizeof(uint16_t));
    std::vector<float> got(static_cast<size_t>(c.n));
    for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(packed_out[i]);

    // NMSE, not bit-exactness: simd_sum is a tree reduction where the CPU
    // reference is sequential, so a different rounding order is expected and
    // stated rather than papered over.
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double diff = double(got[i]) - double(ref[i]);
      num += diff * diff;
      den += double(ref[i]) * double(ref[i]);
    }
    const double nmse = den > 0.0 ? num / den : num;
    CAPTURE(nmse);
    CHECK(nmse <= 5e-4);

    metal.Free(da);
    metal.Free(db);
    metal.Free(dc);
  }
  metal.DestroyQueue(q);
}

// ===========================================================================
// 2-D blocked simdgroup-matrix GEMM (m > 1). Attribution put PREFILL at 33.7%
// of GPU time from only 168 dispatches, ~8.2x slower than MLX-LM, still on the
// 16x16 scalar tile loop. The small-m dead-end established the missing property
// precisely: A reuse ACROSS COLUMNS, which needs 2-D blocking. This kernel
// stages A and B tiles in threadgroup memory and multiplies them with
// simdgroup_float8x8, so one kernel serves prefill AND batched decode.
TEST_CASE("Metal routes m>1 BT matmul to the simdgroup GEMM and matches the CPU oracle") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};

  // f32 arm first: at bf16 output the NMSE is dominated by store rounding and
  // cannot discriminate kernels (that lesson cost a golden re-capture once).
  {
    const int64_t m = 64, k = 512, n = 128;
    std::mt19937 rng(0x5EEDu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> ah(static_cast<size_t>(m * k)), bh(static_cast<size_t>(n * k));
    for (auto& x : ah) x = dist(rng);
    for (auto& x : bh) x = dist(rng);
    std::vector<double> ref(static_cast<size_t>(m * n), 0.0);
    for (int64_t r = 0; r < m; ++r)
      for (int64_t c2 = 0; c2 < n; ++c2) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < k; ++kk)
          acc += double(ah[static_cast<size_t>(r * k + kk)]) *
                 double(bh[static_cast<size_t>(c2 * k + kk)]);
        ref[static_cast<size_t>(r * n + c2)] = acc;
      }
    auto up = [&](const std::vector<float>& h) {
      void* pp = metal.Alloc(h.size() * sizeof(float));
      metal.Copy(q, pp, h.data(), h.size() * sizeof(float));
      return pp;
    };
    void* da = up(ah);
    void* db = up(bh);
    void* dc = metal.Alloc(static_cast<size_t>(m * n) * sizeof(float));
    Tensor ta = Tensor::Contiguous(da, vt::DType::kF32, d, {m, k});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {n, k});
    Tensor tc = Tensor::Contiguous(dc, vt::DType::kF32, d, {m, n});
    vt::MatmulBT(q, tc, ta, tb);
    metal.Synchronize(q);
    std::vector<float> got(static_cast<size_t>(m * n));
    metal.Copy(q, got.data(), dc, got.size() * sizeof(float));
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double diff = double(got[i]) - ref[i];
      num += diff * diff;
      den += ref[i] * ref[i];
    }
    const double nmse = den > 0.0 ? num / den : num;
    MESSAGE("simdgroup GEMM f32 64x512x128 NMSE vs f64 oracle = " << nmse);
    CHECK(nmse <= 1e-10);
    metal.Free(da); metal.Free(db); metal.Free(dc);
  }

  struct Case { const char* name; int64_t m, k, n; };
  const Case cases[] = {
      {"prefill 512x2048x2048", 512, 2048, 2048},
      {"batched decode m=16", 16, 2048, 512},
      {"batched decode m=2", 2, 1024, 256},
      // Ragged on all three dims at once: tile edges in M, N and K together are
      // where a blocked kernel's guards break.
      {"ragged 37x333x201", 37, 333, 201},
  };

  for (const Case& c : cases) {
    CAPTURE(c.name);
    std::mt19937 rng(0x1234u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> ah(static_cast<size_t>(c.m * c.k));
    std::vector<float> bh(static_cast<size_t>(c.n * c.k));
    for (auto& x : ah) x = Bf16RT(dist(rng));
    for (auto& x : bh) x = Bf16RT(dist(rng));
    std::vector<float> ref(static_cast<size_t>(c.m * c.n), 0.0f);
    for (int64_t r = 0; r < c.m; ++r)
      for (int64_t c2 = 0; c2 < c.n; ++c2) {
        float acc = 0.0f;
        for (int64_t kk = 0; kk < c.k; ++kk)
          acc += ah[static_cast<size_t>(r * c.k + kk)] * bh[static_cast<size_t>(c2 * c.k + kk)];
        ref[static_cast<size_t>(r * c.n + c2)] = acc;
      }
    auto up = [&](const std::vector<float>& h) {
      void* pp = metal.Alloc(h.size() * sizeof(uint16_t));
      std::vector<uint16_t> packed(h.size());
      for (size_t i = 0; i < h.size(); ++i) packed[i] = vt::F32ToBF16(h[i]);
      metal.Copy(q, pp, packed.data(), packed.size() * sizeof(uint16_t));
      return pp;
    };
    void* da = up(ah);
    void* db = up(bh);
    void* dc = metal.Alloc(static_cast<size_t>(c.m * c.n) * sizeof(uint16_t));
    Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {c.m, c.k});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {c.n, c.k});
    Tensor tc = Tensor::Contiguous(dc, vt::DType::kBF16, d, {c.m, c.n});

    const bool was_on = vt::metal::ProfileEnabled();
    vt::metal::SetProfileEnabled(true);
    vt::metal::ResetProfile();
    vt::MatmulBT(q, tc, ta, tb);
    metal.Synchronize(q);
    bool saw_mm = false;
    for (const vt::metal::ProfileRow& r : vt::metal::GetProfileRows()) {
      if (r.name == "vt_matmul_bt_mm") saw_mm = true;
    }
  #ifdef VLLM_CPP_MLX
  // With the MLX provider built in, m > 1 is delegated to MLX by design, so the
  // native simdgroup GEMM legitimately never runs. The kernel is still covered by
  // the non-MLX build, which is the default.
  (void)saw_mm;
#else
  CHECK(saw_mm);
#endif  // a numeric check cannot prove WHICH kernel ran
    vt::metal::SetProfileEnabled(was_on);
    vt::metal::ResetProfile();

    std::vector<uint16_t> po(static_cast<size_t>(c.m * c.n));
    metal.Copy(q, po.data(), dc, po.size() * sizeof(uint16_t));
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < po.size(); ++i) {
      const double got = double(vt::BF16ToF32(po[i]));
      num += (got - double(ref[i])) * (got - double(ref[i]));
      den += double(ref[i]) * double(ref[i]);
    }
    const double nmse = den > 0.0 ? num / den : num;
    CAPTURE(nmse);
    CHECK(nmse <= 5e-4);
    metal.Free(da); metal.Free(db); metal.Free(dc);
  }
  metal.DestroyQueue(q);
}

// ===========================================================================
// GEMM micro-benchmark (opt-in via VT_MM_BENCH=1). Prefill is the dominant
// remaining gap to MLX-LM and the mm kernel's limit is unidentified: barriers,
// tile width and staging branches have each been measured and excluded. This
// times the kernel in ISOLATION at the model's real prefill shapes and reports
// achieved GFLOP/s, so the next decision rests on the kernel's own roofline
// rather than on an end-to-end delta. With MLX built in, the same binary times
// MLX's steel GEMM on the identical shape, which is the only apples-to-apples
// answer to "how far off is our kernel".
TEST_CASE("Metal GEMM microbenchmark" * doctest::skip(true)) {
  if (std::getenv("VT_MM_BENCH") == nullptr) return;
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};

  struct Shape { const char* name; int64_t m, k, n; };
  const Shape shapes[] = {
      {"qkv     512x2048x2048", 512, 2048, 2048},
      {"mlp-up  512x2048x6144", 512, 2048, 6144},
      {"mlp-dn  512x6144x2048", 512, 6144, 2048},
  };
  for (const Shape& s : shapes) {
    std::vector<uint16_t> ah(static_cast<size_t>(s.m * s.k), vt::F32ToBF16(0.01f));
    std::vector<uint16_t> bh(static_cast<size_t>(s.n * s.k), vt::F32ToBF16(0.02f));
    void* da = metal.Alloc(ah.size() * 2);
    void* db = metal.Alloc(bh.size() * 2);
    void* dc = metal.Alloc(static_cast<size_t>(s.m * s.n) * 2);
    metal.Copy(q, da, ah.data(), ah.size() * 2);
    metal.Copy(q, db, bh.data(), bh.size() * 2);
    Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {s.m, s.k});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {s.n, s.k});
    Tensor tc = Tensor::Contiguous(dc, vt::DType::kBF16, d, {s.m, s.n});

    vt::MatmulBT(q, tc, ta, tb);  // warm up: pipeline build, first-touch
    metal.Synchronize(q);

    const int iters = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) vt::MatmulBT(q, tc, ta, tb);
    metal.Synchronize(q);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double flops = 2.0 * double(s.m) * double(s.k) * double(s.n) * iters;
    const double bytes = 2.0 * (double(s.m) * s.k + double(s.n) * s.k) * iters;
    MESSAGE("GEMM " << s.name << ": " << (secs / iters * 1e3) << " ms/iter, "
                    << (flops / secs / 1e9) << " GFLOP/s, "
                    << (bytes / secs / 1e9) << " GB/s operand read");
    metal.Free(da); metal.Free(db); metal.Free(dc);
  }
  metal.DestroyQueue(q);
}

// Diagnostic for the 64x64 GEMM defect: reports WHICH output rows are wrong.
// The m-dependence (m<=16 pass, m>=64 fail) has two competing explanations that
// predict different row patterns, and NMSE cannot distinguish them.
TEST_CASE("Metal GEMM per-row diagnostic" * doctest::skip(true)) {
  if (std::getenv("VT_MM_ROWDIAG") == nullptr) return;
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  const int64_t m = 64, k = 64, n = 64;
  // A[r][*] = r+1, B[c][*] = 1  =>  out[r][c] = (r+1)*k exactly, so a wrong row
  // is unmistakable and its VALUE says which source row it actually read.
  std::vector<float> ah(static_cast<size_t>(m * k)), bh(static_cast<size_t>(n * k), 1.0f);
  for (int64_t r = 0; r < m; ++r)
    for (int64_t j = 0; j < k; ++j) ah[static_cast<size_t>(r * k + j)] = float(r + 1);
  auto up = [&](const std::vector<float>& h) {
    void* p = metal.Alloc(h.size() * sizeof(float));
    metal.Copy(q, p, h.data(), h.size() * sizeof(float));
    return p;
  };
  void* da = up(ah); void* db = up(bh);
  void* dc = metal.Alloc(static_cast<size_t>(m * n) * sizeof(float));
  Tensor ta = Tensor::Contiguous(da, vt::DType::kF32, d, {m, k});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {n, k});
  Tensor tc = Tensor::Contiguous(dc, vt::DType::kF32, d, {m, n});
  vt::MatmulBT(q, tc, ta, tb);
  metal.Synchronize(q);
  std::vector<float> got(static_cast<size_t>(m * n));
  metal.Copy(q, got.data(), dc, got.size() * sizeof(float));
  std::string bad;
  for (int64_t r = 0; r < m; ++r) {
    const float want = float(r + 1) * float(k);
    const float g = got[static_cast<size_t>(r * n)];
    if (g != want) bad += std::to_string(r) + "(got " + std::to_string(int(g / k)) + ") ";
  }
  MESSAGE("rows wrong [row(source row it actually read)]: " << (bad.empty() ? "NONE" : bad));
  metal.Free(da); metal.Free(db); metal.Free(dc);
  metal.DestroyQueue(q);
}
