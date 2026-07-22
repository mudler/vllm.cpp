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

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/op_provider.h"

#include "vllm/platforms/interface.h"
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

  // No Metal attention kernel exists yet, so the priority list is EMPTY by
  // design (see src/vllm/platforms/metal.cpp) — selection must fail loudly
  // rather than name a backend whose kernels are absent.
  CHECK(p.get_attn_backend_priority().empty());
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
  // Still stubbed — attention, KV cache, quant, sampling. A partial backend is a
  // supported state (vt::GetOp throws on lookup). kPagedAttention stays OURS
  // even once MLX is enabled: MLX has no paged-KV primitive at all.
  for (vt::OpId op : {vt::OpId::kPagedAttention, vt::OpId::kReshapeAndCache,
                      vt::OpId::kEmbedding, vt::OpId::kGreedyArgmax}) {
    CHECK_FALSE(vt::OpRegistered(op, DeviceType::kMETAL));
  }
  CHECK_THROWS_AS(vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kMETAL),
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
