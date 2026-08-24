// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Validates the lifted vLLM cutlass W8A8 fp8 path — vt::QuantFp8Static (static
// per-tensor activation quant) + vt::MatmulFp8Cutlass (per-tensor fp8 GEMM,
// alpha = input_scale·weight_scale) — against a host W8A8 reference (dequant both
// operands, matmul) within fp8/bf16 tolerance. CUDA-only; SKIPs cleanly with no
// GPU (the fp8 cutlass op is sm120a-only, no CPU kernel).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string_view>
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
// ON since #1843 (VT_FP8_PLAN_CACHE=0 is the rollback / A/B arm). Under the
// DEFAULT (this plain test and the named `test_ops_fp8_cutlass_plan_cache_default_on`
// ctest arm, no env) the FIRST MatmulFp8CublasLt call on a shape builds the plan
// fresh (empty cache -> full descriptor/layout creation +
// cublasLtMatmulAlgoGetHeuristic, exactly the per-call path) and USES it; every
// later call on that shape is a cache HIT reusing the same desc/algo — so a
// first-call-vs-later-call byte compare in one process is literally
// fresh-plan-output == cached-plan-output. Under the rollback arm
// (`test_ops_fp8_cutlass_plan_cache_rollback_off`, VT_FP8_PLAN_CACHE=0) every
// call rebuilds the plan fresh, proving the escape hatch stays byte-stable
// across repeats. Both hold because
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

// PERF-FP8-ALPHA-FOLD (.agents/specs/perf-fp8-alpha-fold.md, #402 §3 "Lever B").
// vt::MatmulFp8CublasLtAlphaVec applies a per-output-COLUMN alpha. When
// VT_FP8_ALPHA_VEC_EPILOGUE=1 and the selected algo advertises
// CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO it does so INSIDE the
// cuBLASLt epilogue; otherwise it runs the shipped two-launch form. Both must
// produce BYTE-identical output, because both are the same single IEEE f32
// multiply on the same f32 accumulator (`x1.0 -> store -> load -> xalpha` vs
// `xalpha`), at the same scale type and the same store dtype.
//
// The claim is SHAPE-CONDITIONAL, exactly as #213's merge equivalence is
// (.agents/specs/perf-27b-gdn-fp8-merged-qkvz.md:106-136): the pointer mode is
// part of the descriptor the heuristic reads, so it can select a different
// split-K, and f32 addition is not associative. At the real gate shapes cuBLASLt
// needs no split-K to fill the device (measured splitK=1 at M=1, K=5120,
// n=16384), which is the standing precondition of this case. If a future driver
// splits K here the case goes RED, and that is the correct signal: the two arms
// would no longer be one arithmetic. Read the selected algos with
// VT_GEMM_ALGO_LOG=1 (tags `TN-fp8` vs `TN-fp8-alphavec`).
namespace {
void AlphaVecMatchesTwoLaunch(int M, int Nqkv, int Nz, int K, uint32_t seed, float alpha_qkv,
                              float alpha_z) {
  const int N = Nqkv + Nz;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> ub(0, 255);
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
  // The model's resident vector: each shard's folded alpha per output column.
  std::vector<float> alpha_host(static_cast<size_t>(N));
  std::fill(alpha_host.begin(), alpha_host.begin() + Nqkv, alpha_qkv);
  std::fill(alpha_host.begin() + Nqkv, alpha_host.end(), alpha_z);

  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  DeviceTensor da(b, g.q, DType::kI8, {M, K}, a_fp8.data());
  DeviceTensor dw(b, g.q, DType::kI8, {N, K}, b_fp8.data());
  DeviceTensor dalpha(b, g.q, DType::kF32, {N}, alpha_host.data());

  const size_t out_bytes = static_cast<size_t>(M) * N * vt::SizeOf(DType::kF32);
  // Reference: the form this seam shipped with — GEMM at alpha=1, then the
  // column vector as a separate full-tensor pass.
  std::vector<uint8_t> two_launch(out_bytes);
  {
    DeviceTensor dout(b, g.q, DType::kF32, {M, N});
    vt::MatmulFp8CublasLt(g.q, dout.tensor(), da.tensor(), dw.tensor(), 1.0f);
    vt::MulColVecF32(g.q, dout.tensor(), dalpha.tensor());
    dout.Download(g.q, two_launch.data());
  }
  std::vector<uint8_t> alpha_vec(out_bytes);
  {
    DeviceTensor dout(b, g.q, DType::kF32, {M, N});
    vt::MatmulFp8CublasLtAlphaVec(g.q, dout.tensor(), da.tensor(), dw.tensor(), dalpha.tensor());
    dout.Download(g.q, alpha_vec.data());
  }
  CHECK(alpha_vec == two_launch);  // BYTE-identical, not merely close
}

// PERF-FP8-ALPHA-FOLD / #417 — the SAME arm-equivalence claim at a BF16 D.
//
// Narrowing this GEMM's D to bf16 is the only remaining lever on the per-column
// alpha pass (it is bandwidth-saturated, so halving its width halves its cost),
// and bf16 is what vLLM's ModelOptFp8LinearMethod emits here. But at bf16 the
// epilogue arm and the fallback arm are NOT the same arithmetic: the fallback
// rounds twice (store bf16, multiply, store bf16) and the epilogue rounds once
// (multiply, store bf16). So the op REFUSES the epilogue for a bf16 D and always
// takes the two-launch arm, keeping VT_FP8_ALPHA_VEC_EPILOGUE a pure performance
// switch that can never move a token at any dtype.
//
// This case pins that refusal, and it is written so it is BINDING with the toggle
// ON: the reference is built from the explicit two-launch calls, so if the bf16
// gate in MatmulFp8CublasLtAlphaVecKernelCuda were removed and a driver DID serve
// the epilogue, the double rounding would show up here as a byte difference. With
// the toggle OFF it degenerates to comparing identical code, which is why the
// ctest arm test_ops_fp8_alpha_vec_epilogue_on is the one that binds.
//
// WHAT THIS CASE DOES AND DOES NOT COVER — do not cite it as a token gate for
// VT_GDN_FP8_IN_BF16. Its FIRST assertion pins the epilogue REFUSAL only: both of
// those arms apply alpha after the GEMM, so both are `round -> scale`. Its SECOND
// assertion does compare the two orders the bf16-D lever actually changes —
// `round-then-scale` (bf16 D) against `scale-then-round` (f32 D) — but it bounds
// them at the arithmetic level, one ulp per word, on random fp8 operands. That is
// a bound on the KERNEL, not evidence that the model's tokens survive it: the
// SACRED engine gates on the fp8 tower decide the lever, and no committed gate
// loads that checkpoint yet (#466). See .agents/specs/perf-fp8-alpha-fold.md
// §Outcome, "What the green suite does and does not say".
void AlphaVecBf16TakesTwoLaunch(int M, int Nqkv, int Nz, int K, uint32_t seed,
                                float alpha_qkv, float alpha_z) {
  const int N = Nqkv + Nz;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> ub(0, 255);
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
  std::vector<float> alpha_host(static_cast<size_t>(N));
  std::fill(alpha_host.begin(), alpha_host.begin() + Nqkv, alpha_qkv);
  std::fill(alpha_host.begin() + Nqkv, alpha_host.end(), alpha_z);

  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  DeviceTensor da(b, g.q, DType::kI8, {M, K}, a_fp8.data());
  DeviceTensor dw(b, g.q, DType::kI8, {N, K}, b_fp8.data());
  DeviceTensor dalpha(b, g.q, DType::kF32, {N}, alpha_host.data());

  const size_t out_bytes = static_cast<size_t>(M) * N * vt::SizeOf(DType::kBF16);
  std::vector<uint8_t> two_launch(out_bytes);
  {
    DeviceTensor dout(b, g.q, DType::kBF16, {M, N});
    vt::MatmulFp8CublasLt(g.q, dout.tensor(), da.tensor(), dw.tensor(), 1.0f);
    vt::MulColVecF32(g.q, dout.tensor(), dalpha.tensor());
    dout.Download(g.q, two_launch.data());
  }
  std::vector<uint8_t> alpha_vec(out_bytes);
  {
    DeviceTensor dout(b, g.q, DType::kBF16, {M, N});
    vt::MatmulFp8CublasLtAlphaVec(g.q, dout.tensor(), da.tensor(), dw.tensor(), dalpha.tensor());
    dout.Download(g.q, alpha_vec.data());
  }
  CHECK(alpha_vec == two_launch);  // BYTE-identical: the bf16 D must not reach the epilogue

  // The bf16 result must also be the bf16 rounding of the f32 result to within
  // ONE ulp, element for element -- i.e. narrowing D changed the store width and
  // the point at which the value is rounded, and NOTHING else. This is what makes
  // the lever a width change rather than a different computation, and it is what
  // catches a bf16 GEMM that accumulated in bf16 or applied alpha at bf16: both
  // move words far beyond the one-ulp double-rounding envelope. See the bound
  // below for the measured distribution that fixes the number at one.
  std::vector<uint8_t> ref_f32(static_cast<size_t>(M) * N * vt::SizeOf(DType::kF32));
  {
    DeviceTensor dout(b, g.q, DType::kF32, {M, N});
    vt::MatmulFp8CublasLt(g.q, dout.tensor(), da.tensor(), dw.tensor(), 1.0f);
    vt::MulColVecF32(g.q, dout.tensor(), dalpha.tensor());
    dout.Download(g.q, ref_f32.data());
  }
  const auto* f32v = reinterpret_cast<const float*>(ref_f32.data());
  const auto* bf16v = reinterpret_cast<const uint16_t*>(two_launch.data());
  // Map a bf16 bit pattern to a monotonic integer: consecutive representable
  // bf16 values are exactly 1 apart and the map is continuous through zero
  // (+0 and -0 both key to 0), so |key(a) - key(b)| IS the ulp distance.
  auto ulp_key = [](uint16_t b) -> int32_t {
    return (b & 0x8000u) != 0 ? -static_cast<int32_t>(b & 0x7FFFu) : static_cast<int32_t>(b);
  };
  const size_t words = static_cast<size_t>(M) * N;
  int64_t max_ulp = 0;
  size_t worst = 0;
  size_t hist[5] = {0, 0, 0, 0, 0};  // words at 0, 1, 2, 3, >=4 ulp
  for (size_t i = 0; i < words; ++i) {
    const int64_t d = std::abs(static_cast<int64_t>(ulp_key(bf16v[i])) -
                               static_cast<int64_t>(ulp_key(vt::F32ToBF16(f32v[i]))));
    if (d > max_ulp) {
      max_ulp = d;
      worst = i;
    }
    ++hist[static_cast<size_t>(std::min<int64_t>(d, 4))];
  }
  // A power-of-two alpha makes the multiply an exact exponent shift, so rounding
  // COMMUTES with it and the two arms must agree BIT for bit — no double rounding
  // is possible. That turns the same comparison into a control that isolates the
  // mechanism: at a pow2 alpha every remaining difference would have to come from
  // the accumulators themselves disagreeing (a different reduction order between
  // the f32-D and bf16-D plans, which are separate cuBLASLt plans because
  // `out_type` is part of `Fp8PlanKey`). The spec found the same escape hatch from
  // the other side: the rejected z-slice fallback is exact iff `qkv.alpha` is a
  // power of two.
  auto is_pow2 = [](float a) {
    int e = 0;
    return a > 0.0f && std::frexp(a, &e) == 0.5f;
  };
  const int64_t bound = (is_pow2(alpha_qkv) && is_pow2(alpha_z)) ? 0 : 1;
  // Printed on every run, pass or fail: the DISTRIBUTION is the evidence, and a
  // bound whose measured distribution is not in the log is a bound nobody can
  // re-derive. MESSAGE is is_warn, so it adds no assertion to the count.
  MESSAGE("bf16-vs-f32 ulp histogram [0/1/2/3/>=4] at M=" << M << " N=" << N << " alpha="
          << alpha_qkv << "/" << alpha_z << " (bound " << bound << "): " << hist[0]
          << "/" << hist[1] << "/" << hist[2] << "/" << hist[3] << "/" << hist[4]
          << "  max=" << max_ulp << " ulp at i=" << worst << " (f32=" << f32v[worst]
          << ", bf16=" << vt::BF16ToF32(bf16v[worst])
          << ", rnd(f32)=" << vt::BF16ToF32(vt::F32ToBF16(f32v[worst])) << ")");
  // A MAGNITUDE bound, not a COUNT bound.
  //
  // What this replaces, and why the old form was wrong. It read
  // `mismatches * 100 < M * N` — "fewer than 1% of words differ at all" — and it
  // had never been executed: the .cu half of this row was written on a host with
  // no nvcc and no GPU. The first CUDA run measured 26.0/26.1/26.8/26.1% of words
  // differing at the four shapes, so it was RED on arrival. It was ALSO too weak:
  // a count bound cannot tell a 1-ulp disagreement in a quarter of the words
  // (benign) from a 10-ulp disagreement in half a percent of them (a broken
  // reduction), and only the second is a defect.
  //
  // A ~26% 1-ulp population is what this comparison MUST produce, by construction.
  // The bf16 arm computes `rnd_bf16(rnd_bf16(acc) * alpha)`; the f32 arm computes
  // `rnd_bf16(fl32(acc * alpha))`. The bf16 arm's first rounding perturbs the
  // product by at most half a bf16 ulp (relative 2^-9), so its second rounding can
  // land on an ADJACENT bf16 and never further. That is double rounding, it is the
  // arithmetic this lever deliberately introduces (`round-then-scale` instead of
  // `scale-then-round`), and the spec's rejected z-slice fallback measured the same
  // signature — 25-36% of words off by 1 ulp over 2e6 accumulators, 6/6 alpha pairs
  // (.agents/specs/perf-fp8-alpha-fold.md, "The z-slice fallback is REJECTED").
  //
  // MEASURED, which is what fixes the bound at one (GB10 sm_121a, CUDA 13.0.88,
  // Release, CUTLASS+FA2+Triton, 2026-08-12, #501). At the four production shapes
  // the histogram is 0-and-1-ulp ONLY: 4253/16384, 12814/49152, 1645/6144 and
  // 546721/2097152 words at exactly one ulp — the same counts the count bound was
  // rejecting — and ZERO words at two or beyond, 2.17M words in total. At the
  // pow2-alpha controls it is 16384/16384 and 2097152/2097152 at zero ulp.
  //
  // What the bound still catches, which is the whole point: anything that moves a
  // word by TWO ulp or more is outside the double-rounding envelope, so it is a
  // different computation — a bf16 accumulation, an alpha applied at bf16 width, a
  // split-K reduction the f32 plan did not take, or an epilogue that slipped past
  // the bf16 refusal. Those are the defects this case exists to find, and the old
  // count bound would have passed every one of them that touched under 1% of words.
  // The pow2-alpha arms hold the strictly tighter `bound == 0`, where double
  // rounding is impossible and ANY difference is one of those defects.
  CHECK(max_ulp <= bound);
}
}  // namespace

TEST_CASE("fp8 cuBLASLt per-column alpha is BYTE-identical to the two-launch form") {
  if (!HasCuda()) {
    MESSAGE("no CUDA device; skipping fp8 cuBLASLt vector-alpha byte-exact test");
    return;
  }
  // Says which arm actually ran, so a green default run is never mistaken for
  // the binding proof: with the toggle OFF both sides execute the SAME code and
  // the case only proves the fallback is unchanged. The binding comparison is
  // the ctest arm `test_ops_fp8_alpha_vec_epilogue_on`.
  const char* on = std::getenv("VT_FP8_ALPHA_VEC_EPILOGUE");
  if (on != nullptr && std::string_view(on) == "1")
    MESSAGE("VT_FP8_ALPHA_VEC_EPILOGUE=1: epilogue arm vs two-launch — the binding comparison");
  else
    MESSAGE("VT_FP8_ALPHA_VEC_EPILOGUE unset: fallback arm only (both sides identical code)");

  // The real 27B GDN in_proj_qkvz dims (hidden 5120; conv_dim 10240 + value_dim
  // 6144) and the 35B's (hidden 2048), at the prefill and decode token counts.
  AlphaVecMatchesTwoLaunch(1, 10240, 6144, 5120, 21, 0.035f, 0.017f);   // 27B, decode M=1
  AlphaVecMatchesTwoLaunch(3, 10240, 6144, 5120, 22, 0.035f, 0.017f);   // 27B, M=3
  AlphaVecMatchesTwoLaunch(1, 4096, 2048, 2048, 23, 0.041f, 0.0092f);   // 35B family, M=1
  AlphaVecMatchesTwoLaunch(3, 4096, 2048, 2048, 24, 0.041f, 0.0092f);   // 35B family, M=3
  AlphaVecMatchesTwoLaunch(128, 10240, 6144, 5120, 25, 0.035f, 0.017f); // prefill tile
  // Equal shard alphas: the vector degenerates to a constant and must still match.
  AlphaVecMatchesTwoLaunch(1, 10240, 6144, 5120, 26, 0.028f, 0.028f);
}

TEST_CASE("fp8 cuBLASLt per-column alpha at a BF16 D stays on the two-launch arm") {
  if (!HasCuda()) {
    MESSAGE("no CUDA device; skipping fp8 cuBLASLt bf16-D vector-alpha test");
    return;
  }
  const char* on = std::getenv("VT_FP8_ALPHA_VEC_EPILOGUE");
  if (on != nullptr && std::string_view(on) == "1")
    MESSAGE("VT_FP8_ALPHA_VEC_EPILOGUE=1: the bf16 D must STILL take the fallback");
  else
    MESSAGE("VT_FP8_ALPHA_VEC_EPILOGUE unset: fallback arm on both sides");

  // The same real gate shapes as the f32 case above: 27B GDN in_proj_qkvz
  // (hidden 5120; conv_dim 10240 + value_dim 6144) and the 35B's (hidden 2048),
  // at decode and prefill token counts. The probe recorded cuBLASLt serving a
  // bf16 D at every one of these shapes and every M (spec §Outcome), so a refusal
  // to build a bf16 plan here would be a REGRESSION, not an absent capability.
  AlphaVecBf16TakesTwoLaunch(1, 10240, 6144, 5120, 31, 0.035f, 0.017f);   // 27B, decode M=1
  AlphaVecBf16TakesTwoLaunch(3, 10240, 6144, 5120, 32, 0.035f, 0.017f);   // 27B, M=3
  AlphaVecBf16TakesTwoLaunch(1, 4096, 2048, 2048, 33, 0.041f, 0.0092f);   // 35B family, M=1
  AlphaVecBf16TakesTwoLaunch(128, 10240, 6144, 5120, 34, 0.035f, 0.017f); // prefill tile
  // CONTROLS. Both shard alphas are powers of two, so the alpha multiply is an
  // exact exponent shift and rounding commutes with it: these arms demand
  // BIT-equality (`bound == 0` inside the helper), at both the decode and the
  // prefill geometry. They are what separates the two candidate mechanisms — a
  // ~26% 1-ulp population that survives here would NOT be double rounding, it
  // would be the two plans' accumulators disagreeing, which is a real defect.
  AlphaVecBf16TakesTwoLaunch(1, 10240, 6144, 5120, 35, 1.0f, 0.25f);      // 27B, decode M=1
  AlphaVecBf16TakesTwoLaunch(128, 10240, 6144, 5120, 36, 0.5f, 0.25f);    // prefill tile
}

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
