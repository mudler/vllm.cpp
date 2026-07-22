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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/op_provider.h"

#include "vllm/platforms/interface.h"
#include "vllm/v1/attention/registry.h"  // SelectAttentionBackendName
#include "vt/backend.h"
#include "vt/ops.h"

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

  CHECK(p.get_device_capability().present());
  CHECK(p.get_device_capability().major >= 1);

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
  // Still stubbed, and asserted so a later row cannot quietly claim more than it
  // implements: the quant tier, the GDN/MoE families, RoPE (which is what
  // Qwen3-dense needs on top of this set — work row M3b), and every sampler op
  // except greedy argmax. A partial backend is a supported state: vt::GetOp
  // throws on lookup.
  for (vt::OpId op : {vt::OpId::kRopeFromCache, vt::OpId::kRopeCosSinCache,
                      vt::OpId::kRandomSample, vt::OpId::kComputeProbs,
                      vt::OpId::kMoeCombine}) {
    CHECK_FALSE(vt::OpRegistered(op, DeviceType::kMETAL));
  }
  CHECK_THROWS_AS(vt::GetOp(vt::OpId::kRopeFromCache, DeviceType::kMETAL),
                  std::runtime_error);
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
      // MLX was selected AND did not decline: the delegated GEMM really ran.
      CHECK(mlx_declines == 0);
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
  CHECK(stats.declines == 1);                        // ... and declined exactly once.

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
