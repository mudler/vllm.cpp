// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Validates the lifted vLLM cutlass W8A8 fp8 path — vt::QuantFp8Static (static
// per-tensor activation quant) + vt::MatmulFp8Cutlass (per-tensor fp8 GEMM,
// alpha = input_scale·weight_scale) — against a host W8A8 reference (dequant both
// operands, matmul) within fp8/bf16 tolerance. CUDA-only; SKIPs cleanly with no
// GPU (the fp8 cutlass op is sm120a-only, no CPU kernel).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"  // F32ToF8E4M3
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // F8E4M3ToF32
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

float Bf16ToF32(uint16_t h) { return vt::BF16ToF32(h); }

void CheckClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0, first_bad = 0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol = atol + rtol * std::fabs(want[i]);
    const float diff = std::fabs(got[i] - want[i]);
    if (diff > max_abs) max_abs = diff;
    if (!(diff <= tol)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(want[first_bad]);
    CAPTURE(max_abs);
  }
  CHECK(bad == 0);
}

// Host W8A8 reference for one (M,N,K): static-quant the f32 activation to fp8,
// then out[m,n] = alpha * Σ_k f8val(a[m,k]) * f8val(b[n,k]) — the exact math the
// GPU (QuantFp8Static + MatmulFp8Cutlass) computes, only the K-reduction order
// differs (matmul tolerance).
void RunCase(int M, int N, int K, uint32_t seed, bool cublaslt = false) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ux(-2.0f, 2.0f);
  std::uniform_int_distribution<int> ub(0, 255);

  const float input_scale = 0.035f;   // ~ amax_act / 448 for the [-2,2] range
  const float weight_scale = 0.017f;  // arbitrary positive per-tensor weight scale
  const float alpha = input_scale * weight_scale;

  std::vector<float> x(static_cast<size_t>(M) * K);
  for (auto& v : x) v = ux(rng);

  // Random non-NaN fp8-e4m3fn weight bytes [N,K].
  std::vector<uint8_t> b_fp8(static_cast<size_t>(N) * K);
  for (auto& b : b_fp8) {
    int byte = ub(rng);
    if ((byte & 0x7F) == 0x7F) byte &= ~0x7;  // avoid NaN encodings (0x7F/0xFF)
    b = static_cast<uint8_t>(byte);
  }

  // Host static activation quant (bit-identical to the GPU quant kernel).
  std::vector<uint8_t> a_fp8(static_cast<size_t>(M) * K);
  for (size_t i = 0; i < x.size(); ++i)
    a_fp8[i] = vllm::F32ToF8E4M3(x[i] / input_scale);

  // Host reference dot products.
  std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += vllm::F8E4M3ToF32(a_fp8[static_cast<size_t>(m) * K + k]) *
               vllm::F8E4M3ToF32(b_fp8[static_cast<size_t>(n) * K + k]);
      ref[static_cast<size_t>(m) * N + n] = alpha * acc;
    }

  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);

  // GPU path A: quant on device (from f32 x) then the cutlass GEMM.
  DeviceTensor dx(b, g.q, DType::kF32, {M, K}, x.data());
  DeviceTensor da(b, g.q, DType::kI8, {M, K});
  vt::QuantFp8Static(g.q, da.tensor(), dx.tensor(), input_scale);
  // Verify the device quant matches the host quant bit-for-bit.
  std::vector<uint8_t> a_gpu(a_fp8.size());
  da.Download(g.q, a_gpu.data());
  CHECK(a_gpu == a_fp8);

  DeviceTensor dw(b, g.q, DType::kI8, {N, K}, b_fp8.data());
  DeviceTensor dout(b, g.q, DType::kBF16, {M, N});
  if (cublaslt)
    vt::MatmulFp8CublasLt(g.q, dout.tensor(), da.tensor(), dw.tensor(), alpha);
  else
    vt::MatmulFp8Cutlass(g.q, dout.tensor(), da.tensor(), dw.tensor(), alpha);
  std::vector<uint16_t> out_h(static_cast<size_t>(M) * N);
  dout.Download(g.q, out_h.data());
  std::vector<float> got(out_h.size());
  for (size_t i = 0; i < out_h.size(); ++i) got[i] = Bf16ToF32(out_h[i]);

  // bf16 output + f32 tensor-core reduction (different K order): fp8/bf16 tol.
  CheckClose(got, ref, /*atol=*/2e-2f, /*rtol=*/3e-2f);
}

}  // namespace

TEST_CASE("fp8 cutlass W8A8 GEMM matches host W8A8 reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA device; skipping fp8 cutlass W8A8 test");
    return;
  }
  // Cover both M-dispatch configs (M<=256 pingpong 64-tile; M>256 default
  // 128-tile) and the small-M decode shapes (M=1/8) + a prefill shape.
  RunCase(1, 256, 128, 1);
  RunCase(8, 512, 256, 2);
  RunCase(16, 128, 512, 3);
  RunCase(64, 256, 256, 4);
  RunCase(200, 320, 128, 5);
  RunCase(512, 256, 256, 6);
  RunCase(1024, 128, 256, 7);
}

TEST_CASE("fp8 cutlass W8A8 GEMM f32-output path matches bf16") {
  if (!HasCuda()) {
    MESSAGE("no CUDA device; skipping fp8 cutlass f32-out test");
    return;
  }
  // f32 out is the bf16 epilogue value cast up; same reference within tol.
  const int M = 8, N = 256, K = 256;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> ux(-2.0f, 2.0f);
  std::uniform_int_distribution<int> ub(0, 255);
  const float input_scale = 0.03f, weight_scale = 0.02f, alpha = input_scale * weight_scale;
  std::vector<float> x(static_cast<size_t>(M) * K);
  for (auto& v : x) v = ux(rng);
  std::vector<uint8_t> b_fp8(static_cast<size_t>(N) * K);
  for (auto& bb : b_fp8) {
    int byte = ub(rng);
    if ((byte & 0x7F) == 0x7F) byte &= ~0x7;
    bb = static_cast<uint8_t>(byte);
  }
  std::vector<uint8_t> a_fp8(x.size());
  for (size_t i = 0; i < x.size(); ++i) a_fp8[i] = vllm::F32ToF8E4M3(x[i] / input_scale);
  std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += vllm::F8E4M3ToF32(a_fp8[static_cast<size_t>(m) * K + k]) *
               vllm::F8E4M3ToF32(b_fp8[static_cast<size_t>(n) * K + k]);
      ref[static_cast<size_t>(m) * N + n] = alpha * acc;
    }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  DeviceTensor da(b, g.q, DType::kI8, {M, K}, a_fp8.data());
  DeviceTensor dw(b, g.q, DType::kI8, {N, K}, b_fp8.data());
  DeviceTensor dout(b, g.q, DType::kF32, {M, N});
  vt::MatmulFp8Cutlass(g.q, dout.tensor(), da.tensor(), dw.tensor(), alpha);
  std::vector<float> got(static_cast<size_t>(M) * N);
  dout.Download(g.q, got.data());
  CheckClose(got, ref, 2e-2f, 3e-2f);
}

TEST_CASE("fp8 cuBLASLt W8A8 GEMM matches host W8A8 reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA device; skipping fp8 cuBLASLt W8A8 test");
    return;
  }
  // Same shapes as the cutlass case (small-M decode + prefill), routed through
  // vt::MatmulFp8CublasLt (the cuBLASLt e4m3 TN path). Validates the col-major
  // TN layout derivation against the host W8A8 reference. Shapes with no fp8
  // cublasLt heuristic transparently fall back to the cutlass GEMM (still fp8).
  RunCase(1, 256, 128, 1, /*cublaslt=*/true);
  RunCase(8, 512, 256, 2, /*cublaslt=*/true);
  RunCase(16, 128, 512, 3, /*cublaslt=*/true);
  RunCase(64, 256, 256, 4, /*cublaslt=*/true);
  RunCase(200, 320, 128, 5, /*cublaslt=*/true);
  RunCase(512, 256, 256, 6, /*cublaslt=*/true);
  RunCase(1024, 128, 256, 7, /*cublaslt=*/true);
}

// The fused fp8 RMSNorm->static-quant op (vt::RmsNormQuantFp8) must be BYTE-for-BYTE
// identical to the split path it replaces: vt::RmsNorm(bf16 out, residual) then
// vt::QuantFp8Static of that bf16. Covers both residual dtypes (f32 full-precision /
// bf16 vLLM model dtype) and the optional bf16 side-output (GDN emits it for
// in_proj_a/b; full-attn does not). This is the op-level token-exactness gate.
TEST_CASE("rmsnorm_quant_fp8 fused == RmsNorm(bf16)+QuantFp8Static (bit-for-bit)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA device; skipping rmsnorm_quant_fp8 fusion test");
    return;
  }
  using vt::RmsNormArgs;
  const int64_t T = 7, H = 128;  // H multiple of 16
  const size_t N = static_cast<size_t>(T) * H;
  const float eps = 1e-6f, input_scale = 0.03f;

  std::mt19937 rng(9);
  std::uniform_real_distribution<float> ux(-3.0f, 3.0f);
  std::vector<float> x(N), w(static_cast<size_t>(H)), res(N);
  for (auto& v : x) v = ux(rng);
  for (auto& v : w) v = ux(rng) * 0.1f;
  for (auto& v : res) v = ux(rng);

  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  DeviceTensor dx(b, g.q, DType::kF32, {T, H}, x.data());
  DeviceTensor dw(b, g.q, DType::kF32, {H}, w.data());

  auto run = [&](bool bf16_residual, bool want_bf16_out) {
    const DType rdt = bf16_residual ? DType::kBF16 : DType::kF32;
    // Materialize the initial residual bytes so both paths start identical.
    std::vector<uint16_t> res_bf16(N);
    for (size_t i = 0; i < N; ++i) res_bf16[i] = vt::F32ToBF16(res[i]);
    const void* res_init = bf16_residual ? static_cast<const void*>(res_bf16.data())
                                         : static_cast<const void*>(res.data());

    // Split path.
    DeviceTensor dresA(b, g.q, rdt, {T, H}, res_init);
    DeviceTensor dbf16A(b, g.q, DType::kBF16, {T, H});
    vt::RmsNorm(g.q, dbf16A.tensor(), dx.tensor(), dw.tensor(), RmsNormArgs{eps, true},
                &dresA.tensor());
    DeviceTensor dfp8A(b, g.q, DType::kI8, {T, H});
    vt::QuantFp8Static(g.q, dfp8A.tensor(), dbf16A.tensor(), input_scale);
    std::vector<uint8_t> fp8_split(N);
    dfp8A.Download(g.q, fp8_split.data());
    std::vector<uint16_t> bf16_split(N);
    dbf16A.Download(g.q, bf16_split.data());

    // Fused path (separate residual copy).
    DeviceTensor dresB(b, g.q, rdt, {T, H}, res_init);
    DeviceTensor dfp8B(b, g.q, DType::kI8, {T, H});
    DeviceTensor dbf16B(b, g.q, DType::kBF16, {T, H});
    vt::RmsNormQuantFp8(g.q, dfp8B.tensor(), want_bf16_out ? &dbf16B.tensor() : nullptr,
                        dx.tensor(), dw.tensor(), RmsNormArgs{eps, true}, &dresB.tensor(),
                        input_scale);
    std::vector<uint8_t> fp8_fused(N);
    dfp8B.Download(g.q, fp8_fused.data());

    CHECK(fp8_fused == fp8_split);  // fp8 activation byte-identical
    if (want_bf16_out) {
      std::vector<uint16_t> bf16_fused(N);
      dbf16B.Download(g.q, bf16_fused.data());
      CHECK(bf16_fused == bf16_split);  // bf16 side-output byte-identical
    }
    // Residual stream updated identically (compare raw bytes for both dtypes).
    if (bf16_residual) {
      std::vector<uint16_t> ra(N), rb(N);
      dresA.Download(g.q, ra.data());
      dresB.Download(g.q, rb.data());
      CHECK(ra == rb);
    } else {
      std::vector<float> ra(N), rb(N);
      dresA.Download(g.q, ra.data());
      dresB.Download(g.q, rb.data());
      CHECK(ra == rb);
    }
  };

  run(/*bf16_residual=*/false, /*want_bf16_out=*/true);
  run(/*bf16_residual=*/false, /*want_bf16_out=*/false);
  run(/*bf16_residual=*/true, /*want_bf16_out=*/true);
  run(/*bf16_residual=*/true, /*want_bf16_out=*/false);
}

TEST_CASE("fp8 cuBLASLt W8A8 GEMM f32-output path matches reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA device; skipping fp8 cuBLASLt f32-out test");
    return;
  }
  // cuBLASLt writes f32 out directly (no bf16 staging); same reference within tol.
  const int M = 8, N = 256, K = 256;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> ux(-2.0f, 2.0f);
  std::uniform_int_distribution<int> ub(0, 255);
  const float input_scale = 0.03f, weight_scale = 0.02f, alpha = input_scale * weight_scale;
  std::vector<float> x(static_cast<size_t>(M) * K);
  for (auto& v : x) v = ux(rng);
  std::vector<uint8_t> b_fp8(static_cast<size_t>(N) * K);
  for (auto& bb : b_fp8) {
    int byte = ub(rng);
    if ((byte & 0x7F) == 0x7F) byte &= ~0x7;
    bb = static_cast<uint8_t>(byte);
  }
  std::vector<uint8_t> a_fp8(x.size());
  for (size_t i = 0; i < x.size(); ++i) a_fp8[i] = vllm::F32ToF8E4M3(x[i] / input_scale);
  std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += vllm::F8E4M3ToF32(a_fp8[static_cast<size_t>(m) * K + k]) *
               vllm::F8E4M3ToF32(b_fp8[static_cast<size_t>(n) * K + k]);
      ref[static_cast<size_t>(m) * N + n] = alpha * acc;
    }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  DeviceTensor da(b, g.q, DType::kI8, {M, K}, a_fp8.data());
  DeviceTensor dw(b, g.q, DType::kI8, {N, K}, b_fp8.data());
  DeviceTensor dout(b, g.q, DType::kF32, {M, N});
  vt::MatmulFp8CublasLt(g.q, dout.tensor(), da.tensor(), dw.tensor(), alpha);
  std::vector<float> got(static_cast<size_t>(M) * N);
  dout.Download(g.q, got.data());
  CheckClose(got, ref, 2e-2f, 3e-2f);
}

// Byte-exactness proof for the per-device fp8 cuBLASLt PLAN CACHE
// (src/vt/cuda/fp8_plan_cache.h): the cached {desc, layouts, algo} GEMM must be
// BYTE-for-BYTE identical to the freshly-built-plan GEMM. The cache is DEFAULT
// OFF (opt-in VT_FP8_PLAN_CACHE=1). Under the cache-ON ctest arm
// (`test_ops_fp8_cutlass_plan_cache_on`, VT_FP8_PLAN_CACHE=1) the FIRST
// MatmulFp8CublasLt call on a shape builds the plan fresh (empty cache -> full
// descriptor/layout creation + cublasLtMatmulAlgoGetHeuristic, exactly the
// per-call path) and USES it; every later call on that shape is a cache HIT
// reusing the same desc/algo — so a first-call-vs-later-call byte compare in one
// process is literally fresh-plan-output == cached-plan-output. Under the DEFAULT
// (this plain test, cache OFF) every call rebuilds the plan fresh, proving the
// shipped production path is byte-stable across repeats. Both hold because
// cuBLASLt algo selection is process-deterministic per shape (the algo-latching
// record), and a shape with no fp8 heuristic falls back to the deterministic
// cutlass GEMM. Rank/dtype coverage: bf16 and f32 output, small-M decode +
// prefill shapes.
namespace {
void ByteExactReuse(int M, int N, int K, uint32_t seed, DType out_dtype) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> ub(0, 255);
  const float alpha = 0.035f * 0.017f;

  auto rand_fp8 = [&](size_t n) {
    std::vector<uint8_t> v(n);
    for (auto& x : v) {
      int byte = ub(rng);
      if ((byte & 0x7F) == 0x7F) byte &= ~0x7;  // avoid NaN encodings (0x7F/0xFF)
      x = static_cast<uint8_t>(byte);
    }
    return v;
  };
  const std::vector<uint8_t> a_fp8 = rand_fp8(static_cast<size_t>(M) * K);
  const std::vector<uint8_t> b_fp8 = rand_fp8(static_cast<size_t>(N) * K);

  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  DeviceTensor da(b, g.q, DType::kI8, {M, K}, a_fp8.data());
  DeviceTensor dw(b, g.q, DType::kI8, {N, K}, b_fp8.data());

  const size_t out_bytes = static_cast<size_t>(M) * N * vt::SizeOf(out_dtype);
  auto run_once = [&]() {
    DeviceTensor dout(b, g.q, out_dtype, {M, N});
    vt::MatmulFp8CublasLt(g.q, dout.tensor(), da.tensor(), dw.tensor(), alpha);
    std::vector<uint8_t> raw(out_bytes);
    dout.Download(g.q, raw.data());
    return raw;
  };

  const std::vector<uint8_t> fresh = run_once();   // cache empty -> fresh-built plan
  const std::vector<uint8_t> cached1 = run_once();  // cache hit -> reused plan
  const std::vector<uint8_t> cached2 = run_once();  // cache hit again
  CHECK(cached1 == fresh);   // cached-plan GEMM == fresh-plan GEMM, byte-for-byte
  CHECK(cached2 == fresh);   // stable across repeated hits
}
}  // namespace

TEST_CASE("fp8 cuBLASLt cached-plan GEMM is BYTE-identical to the fresh-plan GEMM") {
  if (!HasCuda()) {
    MESSAGE("no CUDA device; skipping fp8 cuBLASLt plan-cache byte-exact test");
    return;
  }
  // Small-M decode (c1-c4) + prefill shapes, both output dtypes. Shapes without
  // an fp8 heuristic fall back to cutlass — still byte-identical across reuse.
  ByteExactReuse(1, 256, 128, 11, DType::kBF16);
  ByteExactReuse(4, 512, 256, 12, DType::kBF16);
  ByteExactReuse(8, 6144, 2048, 13, DType::kBF16);   // 35B-family fp8 projection shape
  ByteExactReuse(64, 256, 256, 14, DType::kF32);
  ByteExactReuse(1024, 512, 256, 15, DType::kBF16);  // prefill
  ByteExactReuse(8, 256, 256, 16, DType::kF32);      // f32-out path
}
