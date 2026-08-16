// CUDA keep-quant GEMM gate (QUANT-GGUF-CIQ-GEMM-CUDA). The kCUDA provider for
// `OpId::kMatmulBTQuant` (src/vt/cuda/cuda_quant_dot.cu) is measured against the
// LANDED CPU keep-quant reference (src/vt/cpu/cpu_quant_gemm.cpp — the oracle,
// itself gated by test_ops_quant_dot.cpp) and against an INDEPENDENT f64
// dequantize-then-dot, on the DeepSeek-V4 encodings IQ2_XXS / IQ3_XXS / Q2_K
// plus the four other Q8_K-family k-quants the CUDA kernel also serves.
//
// The GATE (per the campaign): the CUDA kernel's Q8_K activation quant and the
// whole INTEGER dot are bit-identical to the CPU reference by construction, so
// CUDA-vs-CPU is asserted at a TIGHT NMSE (1e-6, f32 out) — only the per-super-
// block float scale sum is reassociated (warp reduction vs the CPU sequential
// add). CUDA-vs-f64-dequant is asserted at the SAME band test_ops_quant_dot uses
// (5e-4, test-backend-ops.cpp:4277). A wrong codebook index / scale unpack /
// sign would blow both bands (RED-first): the tables live device-side, so a
// transcription slip in cuda_quant_iq_tables.cuh surfaces here as an IQ2/IQ3
// divergence, not silently.
//
// Skips cleanly (returns) when no CUDA backend is present so the CPU CI leg is
// green; it only asserts on a real GB10/CUDA device.
#include <doctest/doctest.h>

#ifdef VLLM_CPP_CUDA
#include <cuda_runtime.h>  // cudaStream_t for the Brick 12 pair/group-diag externs
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/quant.h"      // vt::cpu::BlockToFloat
#include "vt/tensor.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// Brick 12 (ds4-gap launch consolidation): the paired + block-diagonal Q8_0 decode GEMV
// kernels under test (external linkage from cuda_quant_dot.cu). The A/B gates below prove
// they are BYTE-IDENTICAL to the launches they consolidate. These two call the .cu
// symbols DIRECTLY (not through the OpProvider seam), so both the <cuda_runtime.h>
// declaration and the gates themselves exist only in a CUDA build — a CPU-only or
// sanitizer build has neither the toolkit header nor the symbols to link against.
#ifdef VLLM_CPP_CUDA
namespace vt::cuda {
void MatmulQ8_0PairCuda(vt::Tensor& out0, vt::Tensor& out1, const vt::Tensor& a,
                        const vt::Tensor& b0, const vt::Tensor& b1, cudaStream_t s);
void MatmulQ8_0GroupDiagCuda(vt::Tensor& out, const vt::Tensor& a, const vt::Tensor& b,
                             int64_t ng, cudaStream_t s);
}  // namespace vt::cuda
#endif  // VLLM_CPP_CUDA

namespace {

// test-backend-ops.cpp:4277 — the MUL_MAT NMSE band the CPU keep-quant is gated
// on; the CUDA kernel must meet the same against the f64 dequant reference.
constexpr double kMaxNmseErr = 5e-4;
// CUDA-vs-CPU-oracle: integer core identical, so only float reassociation drifts.
constexpr double kMaxNmseVsCpu = 1e-6;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

struct WeightCase {
  DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;
  int dmin_off;
  const char* name;
  // Ceiling for the f64-DEQUANT case only (0 means use kMaxNmseErr). That
  // comparison holds a Q8_K-quantized activation against an f32 reference, so
  // it measures ACTIVATION error, whose reach depends on the weight
  // distribution. It does NOT bound the kernel. The CUDA-vs-CPU bound below is
  // what gates this port, it is shared by every case, and it is not relaxed.
  double nmse_ref_max = 0.0;
};

// Same table as test_ops_quant_dot.cpp: the Q8_K-family encodings the CUDA kernel
// serves natively PLUS Q8_0 (its own 32-block Q8_0-activation path — the DeepSeek-V4
// MLA/o-LoRA/shared-expert/lm_head weights) (offsets from ggml-common.h, restated
// independently).
const WeightCase kCases[] = {
    {DType::kIQ2_XXS, 256, 66, 0, -1, "iq2_xxs"},   // DeepSeek-V4 gate/up
    {DType::kIQ3_XXS, 256, 98, 0, -1, "iq3_xxs"},   // DeepSeek-V4 down
    {DType::kIQ2_S, 256, 82, 0, -1, "iq2_s"},       // DeepSeek-V4 UD-IQ2_M gate/up
    // The two sub-2-bit codebooks the Qwen3.8-2.4T checkpoints store their
    // routed experts in (96.92 % of each model). Without a CUDA arm these fall
    // to the CPU path and still emit CORRECT tokens, just at CPU speed, which
    // no token gate can see -- so device parity is gated HERE.
    // Measured, not guessed. Both are ternary lanes times a per-32 scale, so a
    // super-block spans a wider dynamic range than a 4-6 bit codebook while
    // Q8_K gives the activation ONE scale per 256 elements. iq1_s lands at
    // 5.2399e-4 on the thinnest shape (m=4, n=1 is FOUR dot products), against
    // a 5e-4 default -- the CPU test measures 5.2398e-4 at the SAME shape, and
    // that near-identity is what shows this is the activation term and not the
    // device kernel. 2e-3 is ~4x the residual and an order of magnitude below
    // the weakest decode defect (2.7e-2, measured by mutation on the CPU arm).
    {DType::kIQ1_S, 256, 50, 0, -1, "iq1_s", 2e-3},       // Qwen3.8 UD-IQ1_S experts
    {DType::kIQ1_XXXS, 256, 38, 0, -1, "iq1_xxxs", 2e-3},  // Qwen3.8 UD-Q1_0 experts
    {DType::kQ2_K, 256, 84, 80, 82, "q2_K"},        // DeepSeek-V4 UD-Q2_K_XL
    {DType::kQ3_K, 256, 110, 108, -1, "q3_K"},
    {DType::kQ4_K, 256, 144, 0, 2, "q4_K"},
    {DType::kQ5_K, 256, 176, 0, 2, "q5_K"},
    {DType::kQ6_K, 256, 210, 208, -1, "q6_K"},
    {DType::kQ8_0, 32, 34, 0, -1, "q8_0"},          // DeepSeek-V4 AProj/SExp/Out (Q8_0-act path)
    // NOTE (box-deferred): MXFP4 (39, UD-IQ2_M ffn_down) is NOT here — it dots a
    // 32-elem Q8_0 activation and has no native CUDA GEMM yet (CPU-fallback). The
    // CPU dot IS gated in tests/vt/test_ops_quant_dot.cpp.
};

void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
}

std::vector<uint8_t> RandomBlocks(const WeightCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    if (c.d_off >= 0) put_f16(c.d_off, 0.0125F * jitter);
    if (c.dmin_off >= 0) put_f16(c.dmin_off, 0.0075F * jitter);
    // IQ1_S and IQ1_XXXS carry their per-32 sub-block scale INSIDE the weight
    // (qh bits 12-14, and the sc nibble's bits 0-2). Uniformly random bits
    // spread neighbouring 32-groups over a 15x scale range, which no encoder
    // emits, and Q8_K gives the ACTIVATION one scale per 256 elements -- so the
    // f64-dequant bound below would measure that synthetic dynamic range rather
    // than the kernel. The scale still VARIES per sub-block, so a kernel that
    // dropped or hoisted it is still caught; the delta sign bit stays random.
    // Mirrors the identical narrowing in tests/vt/test_ops_quant_dot.cpp.
    if (c.dtype == DType::kIQ1_S) {
      for (int ib = 0; ib < 8; ++ib) {
        uint16_t qh = 0;
        std::memcpy(&qh, blk + 34 + 2 * ib, sizeof(qh));
        const uint16_t ls = static_cast<uint16_t>(2 + ((i + ib) % 3));
        qh = static_cast<uint16_t>((qh & 0x8FFFU) | (ls << 12));
        std::memcpy(blk + 34 + 2 * ib, &qh, sizeof(qh));
      }
    }
    if (c.dtype == DType::kIQ1_XXXS) {
      for (int ib = 0; ib < 8; ++ib) {
        uint8_t& byte = blk[34 + ib / 2];
        const int shift = 4 * (ib & 1);
        const uint8_t ls = static_cast<uint8_t>(2 + ((i + ib) % 3));
        const uint8_t keep_sign = static_cast<uint8_t>((byte >> shift) & 0x8);
        byte = static_cast<uint8_t>((byte & ~(0xFU << shift)) |
                                    ((keep_sign | ls) << shift));
      }
    }
  }
  return bytes;
}

Tensor DevTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = p;
  t.dtype = dt;
  t.device = Gpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

}  // namespace

TEST_CASE("CUDA keep-quant GEMM == CPU reference and f64 dequant (Q8_K family + Q8_0)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  for (const WeightCase& c : kCases) {
    // Decode (M=1) through prefill (M=512), plus an odd N that catches a warp/
    // chunking assumption; K = 8 super-blocks (model-ish).
    const int64_t k = 8 * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{4}, int64_t{32}, int64_t{512}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(std::string(c.name));
        CAPTURE(m);
        CAPTURE(k);
        CAPTURE(n);

        std::vector<uint8_t> wq =
            RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());

        // --- CPU oracle (the landed keep-quant kernel over host tensors) ------
        std::vector<float> cpu_out(static_cast<size_t>(m * n), 0.0F);
        {
          Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {m, k});
          Tensor bt =
              Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
          bt.dtype = c.dtype;
          Tensor ot =
              Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {m, n});
          vt::MatmulBTQuant(cq, ot, at, bt);
        }

        // --- CUDA path (device tensors, unified pool) -------------------------
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        void* d_w = gpu.Alloc(wq.size());
        void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        Tensor bt = DevTensor(d_w, c.dtype, {n, k});
        Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
        vt::MatmulBTQuant(gq, ot, at, bt);
        std::vector<float> cuda_out(static_cast<size_t>(m * n), 0.0F);
        gpu.Copy(gq, cuda_out.data(), d_o, cuda_out.size() * sizeof(float));
        gpu.Synchronize(gq);
        gpu.Free(d_a);
        gpu.Free(d_w);
        gpu.Free(d_o);

        // --- f64 independent reference: decode weight + f32 activation dot -----
        std::vector<float> w(static_cast<size_t>(n * k));
        vt::cpu::BlockToFloat(c.dtype)(wq.data(), w.data(), n * k);

        double num_ref = 0, den_ref = 0, num_cpu = 0, den_cpu = 0;
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t jj = 0; jj < n; ++jj) {
            double ref = 0;
            for (int64_t p = 0; p < k; ++p)
              ref += static_cast<double>(a[static_cast<size_t>(i * k + p)]) *
                     static_cast<double>(w[static_cast<size_t>(jj * k + p)]);
            const double got =
                cuda_out[static_cast<size_t>(i * n + jj)];
            const double cpu = cpu_out[static_cast<size_t>(i * n + jj)];
            num_ref += (got - ref) * (got - ref);
            den_ref += ref * ref;
            num_cpu += (got - cpu) * (got - cpu);
            den_cpu += cpu * cpu;
            REQUIRE(std::isfinite(got));
          }
        }
        const double nmse_ref = den_ref > 0 ? num_ref / den_ref : num_ref;
        const double nmse_cpu = den_cpu > 0 ? num_cpu / den_cpu : num_cpu;
        CAPTURE(nmse_ref);
        CAPTURE(nmse_cpu);
        const double ref_ceiling =
            c.nmse_ref_max > 0 ? c.nmse_ref_max : kMaxNmseErr;
        CAPTURE(ref_ceiling);
        CHECK(nmse_ref <= ref_ceiling);   // quantization error vs f64 dequant
        CHECK(nmse_cpu <= kMaxNmseVsCpu);  // matches the CPU oracle (int core exact)
      }
    }
  }
  gpu.DestroyQueue(gq);
}

// Brick 4 (last-mile): the CUDA coalesced-Q8_0 layout (RepackQ8_0Cuda + the
// q8_0_aligned kernel) must be BIT-IDENTICAL to the plain in-place Q8_0 path — it
// is a byte permutation of the SAME int8 + f16 scale values feeding the SAME dp4a
// dot. RED-first: a wrong deinterleave offset (qs/scale section) or a mis-aligned
// int4 load would diverge the two outputs.
TEST_CASE("Brick 4: CUDA Q8_0 aligned (coalesced) layout == plain Q8_0 (bit-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0-aligned gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const int64_t k = 8 * c.block_elems;  // 8 blocks
  for (int64_t m : {int64_t{1}, int64_t{4}}) {
    for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
      CAPTURE(m);
      CAPTURE(n);
      std::vector<uint8_t> wq = RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
      std::vector<float> a(static_cast<size_t>(m * k));
      GenerateData(1.0F, a.size(), a.data());

      void* d_a = gpu.Alloc(a.size() * sizeof(float));
      gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
      Tensor at = DevTensor(d_a, DType::kF32, {m, k});

      // plain in-place Q8_0
      void* d_w = gpu.Alloc(wq.size());
      void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
      gpu.Copy(gq, d_w, wq.data(), wq.size());
      Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});
      Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
      vt::MatmulBTQuant(gq, ot, at, bt);
      std::vector<float> plain(static_cast<size_t>(m * n), 0.0F);
      gpu.Copy(gq, plain.data(), d_o, plain.size() * sizeof(float));
      gpu.Synchronize(gq);

      // aligned (coalesced) layout: repack the SAME bytes + flag the tensor
      std::vector<uint8_t> wq_al = wq;
      vt::cpu::RepackQ8_0Cuda(wq_al.data(), n, k);
      void* d_wa = gpu.Alloc(wq_al.size());
      void* d_oa = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
      gpu.Copy(gq, d_wa, wq_al.data(), wq_al.size());
      Tensor bta = DevTensor(d_wa, DType::kQ8_0, {n, k});
      bta.q8_0_aligned = true;
      Tensor ota = DevTensor(d_oa, DType::kF32, {m, n});
      vt::MatmulBTQuant(gq, ota, at, bta);
      std::vector<float> aligned(static_cast<size_t>(m * n), 0.0F);
      gpu.Copy(gq, aligned.data(), d_oa, aligned.size() * sizeof(float));
      gpu.Synchronize(gq);

      for (size_t i = 0; i < plain.size(); ++i) {
        REQUIRE(std::isfinite(aligned[i]));
        CHECK(aligned[i] == plain[i]);  // BIT-IDENTICAL
      }
      gpu.Free(d_a);
      gpu.Free(d_w);
      gpu.Free(d_o);
      gpu.Free(d_wa);
      gpu.Free(d_oa);
    }
  }
  gpu.DestroyQueue(gq);
}

// Lever 1 (ds4-gap): the ds4-preq activation-quant grid (QuantizeQ8_0PreqKernel,
// one warp per 32-block, VT_V4_Q8_PREQ_QUANT default-ON) must be BIT-IDENTICAL to
// the legacy one-thread-per-block QuantizeQ8_0Kernel (=0). Both compute the same
// amax (MAX-reduction is associative + exact), same d = amax/127, same roundf ->
// byte-identical Q8_0 scratch feeding the same integer dot. RED-first: a wrong
// warp-reduce or lane->element mapping would diverge the two outputs.
TEST_CASE("Lever 1: CUDA Q8_0 preq-quant grid == legacy quant grid (bit-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 preq-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const int64_t k = 8 * c.block_elems;  // 8 blocks
  for (int64_t m : {int64_t{1}, int64_t{4}}) {
    for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
      CAPTURE(m);
      CAPTURE(n);
      std::vector<uint8_t> wq = RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
      std::vector<float> a(static_cast<size_t>(m * k));
      GenerateData(1.0F, a.size(), a.data());
      void* d_a = gpu.Alloc(a.size() * sizeof(float));
      gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
      Tensor at = DevTensor(d_a, DType::kF32, {m, k});
      void* d_w = gpu.Alloc(wq.size());
      gpu.Copy(gq, d_w, wq.data(), wq.size());
      Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});

      auto run = [&](const char* flag) {
        setenv("VT_V4_Q8_PREQ_QUANT", flag, 1);
        void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
        Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
        vt::MatmulBTQuant(gq, ot, at, bt);
        std::vector<float> out(static_cast<size_t>(m * n), 0.0F);
        gpu.Copy(gq, out.data(), d_o, out.size() * sizeof(float));
        gpu.Synchronize(gq);
        gpu.Free(d_o);
        return out;
      };
      std::vector<float> legacy = run("0");  // one-thread-per-block
      std::vector<float> preq = run("1");    // ds4-preq warp-per-block
      unsetenv("VT_V4_Q8_PREQ_QUANT");
      for (size_t i = 0; i < legacy.size(); ++i) {
        REQUIRE(std::isfinite(preq[i]));
        CHECK(preq[i] == legacy[i]);  // BIT-IDENTICAL
      }
      gpu.Free(d_a);
      gpu.Free(d_w);
    }
  }
  gpu.DestroyQueue(gq);
}

// Lever 2 / Brick 11 (ds4-gap): the sub-warp Q8_0 GEMV (QuantDotGemmQ8_0SubwarpKernel,
// VT_V4_Q8_SUBWARP=1) splits the 32-lane warp into LANES-wide subgroups (nb≤16→8,
// nb≤48→16, else 32), one output per subgroup. The integer __dp4a accumulation is
// order-independent → `sumi` is bit-exact for ANY LANES; only the final float scale-sum
// re-associates across fewer lanes → a characterized NEAR-TIE (NMSE≤5e-4). The big-K
// (nb=224 → LANES=32) case must be BYTE-IDENTICAL to the plain kernel (same lane→block
// map + same 32-wide reduce) — that byte-identity is the RED-first anchor: a wrong
// SubwarpMask/reduction/lane-map would diverge LANES=32 from plain and fail hard.
TEST_CASE("Lever 2: CUDA Q8_0 sub-warp GEMV == plain (bit-exact big-K, NMSE≤5e-4 short-K)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 sub-warp gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const double kMaxNmse = 5e-4;  // ratified DeepSeek-V4 near-tie band
  // nb = 16 (LANES=8, short-K MLA/kv), 48 (LANES=16, q_b/LoRA), 224 (LANES=32, big-K).
  for (int64_t nb : {int64_t{16}, int64_t{48}, int64_t{224}}) {
    const int64_t k = nb * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{3}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(nb);
        CAPTURE(m);
        CAPTURE(n);
        std::vector<uint8_t> wq = RandomBlocks(c, n * nb, 0x1EAF2U);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        void* d_w = gpu.Alloc(wq.size());
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});

        auto run = [&](const char* flag) {
          setenv("VT_V4_Q8_SUBWARP", flag, 1);
          void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
          Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
          vt::MatmulBTQuant(gq, ot, at, bt);
          std::vector<float> out(static_cast<size_t>(m * n), 0.0F);
          gpu.Copy(gq, out.data(), d_o, out.size() * sizeof(float));
          gpu.Synchronize(gq);
          gpu.Free(d_o);
          return out;
        };
        std::vector<float> plain = run("0");    // full 32-lane warp per output
        std::vector<float> subw = run("1");     // sub-warp tiling
        unsetenv("VT_V4_Q8_SUBWARP");
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < plain.size(); ++i) {
          REQUIRE(std::isfinite(subw[i]));
          const double d = static_cast<double>(subw[i]) - plain[i];
          num += d * d;
          den += static_cast<double>(plain[i]) * plain[i];
          if (nb == 224) CHECK(subw[i] == plain[i]);  // LANES=32 byte-identical (RED-first)
        }
        const double nmse = den > 0 ? num / den : num;
        CAPTURE(nmse);
        CHECK(nmse <= kMaxNmse);  // near-tie (short-K re-associates the float scale-sum)
        gpu.Free(d_a);
        gpu.Free(d_w);
      }
    }
  }
  gpu.DestroyQueue(gq);
}

// Brick 13 (ds4-gap ILP lever): the N-output-rows-per-warp Q8_0 GEMV
// (QuantDotGemmQ8_0MultiRowKernel, VT_V4_Q8_ILP=2|4) computes NROWS consecutive output
// columns per warp, sharing the activation load and issuing NROWS independent weight-load
// streams. UNLIKE the sub-warp lever (Brick 11), each output row still uses the FULL 32-lane
// warp reduce + the identical __dp4a order + the identical f16-scale fold → the result is
// BYTE-IDENTICAL to the plain one-row-per-warp kernel (not merely a near-tie). That
// byte-identity is the RED-first anchor: a wrong j0/tail/grid map or a re-associated reduce
// would diverge from plain and fail hard. Covers short-K (nb=16), mid-K (nb=48), big-K
// (nb=224); n both divisible and NOT divisible by NROWS (tail-row path); m=1 (decode) and m=3.
TEST_CASE("Brick 13: CUDA Q8_0 ILP multi-row GEMV == plain (byte-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 ILP gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  for (int64_t nb : {int64_t{16}, int64_t{48}, int64_t{224}}) {
    const int64_t k = nb * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{3}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}, int64_t{17}}) {  // 7,17 exercise tail
        CAPTURE(nb);
        CAPTURE(m);
        CAPTURE(n);
        std::vector<uint8_t> wq = RandomBlocks(c, n * nb, 0x13B1CU);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        void* d_w = gpu.Alloc(wq.size());
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});

        auto run = [&](const char* flag) {
          setenv("VT_V4_Q8_ILP", flag, 1);
          void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
          Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
          vt::MatmulBTQuant(gq, ot, at, bt);
          std::vector<float> out(static_cast<size_t>(m * n), 0.0F);
          gpu.Copy(gq, out.data(), d_o, out.size() * sizeof(float));
          gpu.Synchronize(gq);
          gpu.Free(d_o);
          return out;
        };
        std::vector<float> plain = run("1");   // one row per warp (baseline)
        std::vector<float> ilp2 = run("2");     // 2 rows per warp
        std::vector<float> ilp4 = run("4");     // 4 rows per warp
        unsetenv("VT_V4_Q8_ILP");
        for (size_t i = 0; i < plain.size(); ++i) {
          REQUIRE(std::isfinite(ilp2[i]));
          REQUIRE(std::isfinite(ilp4[i]));
          CHECK(ilp2[i] == plain[i]);  // byte-identical (RED-first)
          CHECK(ilp4[i] == plain[i]);
        }
        gpu.Free(d_a);
        gpu.Free(d_w);
      }
    }
  }
  gpu.DestroyQueue(gq);
}

// Brick 14 (ds4 raw-mechanism lever): the INTRA-ROW multi-block register-PREFETCH Q8_0 GEMV
// (QuantDotGemmQ8_0PrefetchKernel, VT_V4_Q8_PREFETCH=2|4) hoists PF super-block loads into
// registers before the dependent __dp4a chains, keeping the warp→output map (one row per warp)
// UNCHANGED — unlike the Brick-13 multi-ROW ILP. Each lane still visits its blocks in the SAME
// ascending order, runs the SAME 8×__dp4a per block, the SAME 32-wide warp reduce, and the SAME
// f16-scale fold → the result is BYTE-IDENTICAL to the plain one-row-per-warp kernel. That
// byte-identity is the RED-first anchor: a wrong group/tail stride or a re-associated accumulate
// would diverge from plain and fail hard. nb chosen to exercise groups that are NOT a multiple of
// 32*PF (tail-block path): nb=16 (< one PF=2 group), nb=48, nb=224; n both 1 and tail-y.
TEST_CASE("Brick 14: CUDA Q8_0 register-prefetch GEMV == plain (byte-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 prefetch gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  for (int64_t nb : {int64_t{16}, int64_t{48}, int64_t{224}}) {
    const int64_t k = nb * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{3}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}, int64_t{17}}) {
        CAPTURE(nb);
        CAPTURE(m);
        CAPTURE(n);
        std::vector<uint8_t> wq = RandomBlocks(c, n * nb, 0x14B14U);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        void* d_w = gpu.Alloc(wq.size());
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});

        auto run = [&](const char* flag) {
          setenv("VT_V4_Q8_PREFETCH", flag, 1);
          void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
          Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
          vt::MatmulBTQuant(gq, ot, at, bt);
          std::vector<float> out(static_cast<size_t>(m * n), 0.0F);
          gpu.Copy(gq, out.data(), d_o, out.size() * sizeof(float));
          gpu.Synchronize(gq);
          gpu.Free(d_o);
          return out;
        };
        std::vector<float> plain = run("1");   // one row per warp, no prefetch (baseline)
        std::vector<float> pf2 = run("2");      // prefetch depth 2
        std::vector<float> pf4 = run("4");      // prefetch depth 4
        unsetenv("VT_V4_Q8_PREFETCH");
        for (size_t i = 0; i < plain.size(); ++i) {
          REQUIRE(std::isfinite(pf2[i]));
          REQUIRE(std::isfinite(pf4[i]));
          CHECK(pf2[i] == plain[i]);  // byte-identical (RED-first)
          CHECK(pf4[i] == plain[i]);
        }
        gpu.Free(d_a);
        gpu.Free(d_w);
      }
    }
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("CUDA keep-quant GEMM registers the native kCUDA provider") {
  // The registration is what flips the GGUF loader's keep-quant default ON on a
  // CUDA device (GgufQuantComputeAvailable -> OpRegistered(kMatmulBTQuant,kCUDA))
  // so DeepSeek-V4's experts dispatch to the GPU. Present only in a CUDA build.
  if (!HasCuda()) return;
  CHECK(vt::OpRegistered(vt::OpId::kMatmulBTQuant, DeviceType::kCUDA));
}

#ifdef VLLM_CPP_CUDA
// Brick 12 (ds4-gap "launch consolidation") A/B gate — PAIRED Q8_0 decode GEMV. The
// paired kernel (one launch, two weights, one shared activation) must be BYTE-IDENTICAL
// to running the plain Q8_0 GEMV twice (via vt::MatmulBTQuant). RED-first: a wrong
// activation-load factoring, a swapped weight row-stride, or a reduction re-order would
// diverge the two outputs (this is the wq_a+wkv / shared-gate+up decode fusion).
TEST_CASE("Brick 12: CUDA Q8_0 PAIR == two separate matmuls (bit-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 pair gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  cudaStream_t s = static_cast<cudaStream_t>(gq.handle);
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const int64_t k = 8 * c.block_elems;  // 8 blocks (256), a model-ish decode K
  const int64_t nbk = k / c.block_elems;
  // (n0,n1): unequal out-dims (the real q_a=qlr vs kv_a=hd; gate=up=mi), and 1-row edge.
  for (auto nn : {std::pair<int64_t, int64_t>{16, 16}, {16, 7}, {7, 16}, {1, 24}, {24, 1}}) {
    const int64_t n0 = nn.first, n1 = nn.second;
    CAPTURE(n0);
    CAPTURE(n1);
    std::vector<uint8_t> w0 = RandomBlocks(c, n0 * nbk, 0xA11CEU);
    std::vector<uint8_t> w1 = RandomBlocks(c, n1 * nbk, 0xB0BAU);
    std::vector<float> a(static_cast<size_t>(k));
    GenerateData(1.0F, a.size(), a.data());

    void* d_a = gpu.Alloc(a.size() * sizeof(float));
    void* d_w0 = gpu.Alloc(w0.size());
    void* d_w1 = gpu.Alloc(w1.size());
    void* d_o0 = gpu.Alloc(static_cast<size_t>(n0) * sizeof(float));
    void* d_o1 = gpu.Alloc(static_cast<size_t>(n1) * sizeof(float));
    void* d_r0 = gpu.Alloc(static_cast<size_t>(n0) * sizeof(float));
    void* d_r1 = gpu.Alloc(static_cast<size_t>(n1) * sizeof(float));
    gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
    gpu.Copy(gq, d_w0, w0.data(), w0.size());
    gpu.Copy(gq, d_w1, w1.data(), w1.size());

    Tensor at = DevTensor(d_a, DType::kF32, {1, k});
    Tensor w0t = DevTensor(d_w0, DType::kQ8_0, {n0, k});
    Tensor w1t = DevTensor(d_w1, DType::kQ8_0, {n1, k});
    Tensor o0 = DevTensor(d_o0, DType::kF32, {1, n0});
    Tensor o1 = DevTensor(d_o1, DType::kF32, {1, n1});
    Tensor r0 = DevTensor(d_r0, DType::kF32, {1, n0});
    Tensor r1 = DevTensor(d_r1, DType::kF32, {1, n1});

    vt::cuda::MatmulQ8_0PairCuda(o0, o1, at, w0t, w1t, s);  // ONE paired launch
    vt::MatmulBTQuant(gq, r0, at, w0t);                     // reference: two separate launches
    vt::MatmulBTQuant(gq, r1, at, w1t);

    std::vector<float> got0(static_cast<size_t>(n0)), got1(static_cast<size_t>(n1));
    std::vector<float> ref0(static_cast<size_t>(n0)), ref1(static_cast<size_t>(n1));
    gpu.Copy(gq, got0.data(), d_o0, got0.size() * sizeof(float));
    gpu.Copy(gq, got1.data(), d_o1, got1.size() * sizeof(float));
    gpu.Copy(gq, ref0.data(), d_r0, ref0.size() * sizeof(float));
    gpu.Copy(gq, ref1.data(), d_r1, ref1.size() * sizeof(float));
    gpu.Synchronize(gq);
    gpu.Free(d_a); gpu.Free(d_w0); gpu.Free(d_w1);
    gpu.Free(d_o0); gpu.Free(d_o1); gpu.Free(d_r0); gpu.Free(d_r1);

    CHECK(std::memcmp(got0.data(), ref0.data(), got0.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(got1.data(), ref1.data(), got1.size() * sizeof(float)) == 0);
  }
  gpu.DestroyQueue(gq);
}

// Brick 12 (ds4-gap "row-split consolidation") A/B gate — BLOCK-DIAGONAL grouped Q8_0
// output-LoRA GEMV. The single consolidated launch (all ng groups) must be
// BYTE-IDENTICAL to the ng separate per-group slice GEMVs (each over a disjoint ipg-wide
// activation slice + its olr weight rows). RED-first: a wrong group activation offset or
// a mis-computed rows-per-group would diverge. This is the resident wo_a fusion (the 344
// launches/step, 53% of the 646).
TEST_CASE("Brick 12: CUDA Q8_0 block-diagonal o-LoRA == per-group loop (bit-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 group-diagonal gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  cudaStream_t s = static_cast<cudaStream_t>(gq.handle);
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const size_t row_bytes = static_cast<size_t>(1) * c.block_bytes;  // per 32-block
  // (ng, olr, ipg): ipg must be a multiple of 32; models use ng=8.
  struct Shape { int64_t ng, olr, ipg; };
  for (Shape sh : {Shape{4, 8, 64}, Shape{8, 16, 96}, Shape{2, 7, 32}}) {
    const int64_t ng = sh.ng, olr = sh.olr, ipg = sh.ipg;
    const int64_t nb_g = ipg / c.block_elems;
    const int64_t total_rows = ng * olr;
    CAPTURE(ng);
    CAPTURE(olr);
    CAPTURE(ipg);
    std::vector<uint8_t> w = RandomBlocks(c, total_rows * nb_g, 0xC0FFEEU);
    std::vector<float> o(static_cast<size_t>(ng * ipg));
    GenerateData(0.5F, o.size(), o.data());

    void* d_o = gpu.Alloc(o.size() * sizeof(float));
    void* d_w = gpu.Alloc(w.size());
    void* d_z = gpu.Alloc(static_cast<size_t>(total_rows) * sizeof(float));   // consolidated
    void* d_zr = gpu.Alloc(static_cast<size_t>(total_rows) * sizeof(float));  // per-group ref
    gpu.Copy(gq, d_o, o.data(), o.size() * sizeof(float));
    gpu.Copy(gq, d_w, w.data(), w.size());

    Tensor ot = DevTensor(d_o, DType::kF32, {1, ng * ipg});
    Tensor wt = DevTensor(d_w, DType::kQ8_0, {total_rows, ipg});
    Tensor zt = DevTensor(d_z, DType::kF32, {1, total_rows});
    vt::cuda::MatmulQ8_0GroupDiagCuda(zt, ot, wt, ng, s);  // ONE block-diagonal launch

    // reference: ng separate slice GEMVs (mirrors GemmRowSliceInto in the model forward)
    for (int64_t gp = 0; gp < ng; ++gp) {
      Tensor a_slice =
          DevTensor(static_cast<float*>(d_o) + gp * ipg, DType::kF32, {1, ipg});
      Tensor w_slice = DevTensor(static_cast<uint8_t*>(d_w) +
                                     static_cast<size_t>(gp * olr) * nb_g * row_bytes,
                                 DType::kQ8_0, {olr, ipg});
      Tensor z_slice =
          DevTensor(static_cast<float*>(d_zr) + gp * olr, DType::kF32, {1, olr});
      vt::MatmulBTQuant(gq, z_slice, a_slice, w_slice);
    }

    std::vector<float> got(static_cast<size_t>(total_rows)), ref(static_cast<size_t>(total_rows));
    gpu.Copy(gq, got.data(), d_z, got.size() * sizeof(float));
    gpu.Copy(gq, ref.data(), d_zr, ref.size() * sizeof(float));
    gpu.Synchronize(gq);
    gpu.Free(d_o); gpu.Free(d_w); gpu.Free(d_z); gpu.Free(d_zr);

    CHECK(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0);
  }
  gpu.DestroyQueue(gq);
}
#endif  // VLLM_CPP_CUDA
