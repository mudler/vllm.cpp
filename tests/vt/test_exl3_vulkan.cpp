// The Vulkan arm of the two ops an EXL3 checkpoint ran on the CPU reference tier
// — row BACKEND-VULKAN-EXL3, issue #2530, spec
// .agents/specs/backend-vulkan-exl3.md.
//
// WHAT THIS SUITE GATES, AND WHY THE BOUND IS ZERO.
//
// `src/vt/vulkan/shaders/vt_exl3_had.comp` and `vt_exl3_gemm.comp` are
// transcribed from the PORTABLE CPU reference (`cpu_exl3_kernels.cpp`,
// `cpu_exl3_dequant.cpp`) rather than ported from `cuda_exl3.cu`, whose 90 KiB
// shared-memory budget already exceeds Vulkan's 16 KiB GUARANTEE before one
// reaches `mma.sync`, `ldmatrix`, `cp.async` or a grid-wide barrier Vulkan has
// at no version. Because they are a transcription of that reference and use no
// cooperative matrices and no split-K reduction, all three steps run the same
// IEEE f32 operations on the same values in the same ORDER — so the gate here is
// BYTE equality with the CPU arm, and not the 1.0e-3 RMS bound
// `test_exl3_gemm`'s CUDA cases carry. A tolerance here would be slack this arm
// has not earned and does not need.
//
// fp16 IS A SOFTWARE CODEC ON BOTH SIDES, which is why a byte claim survives the
// absence of any fp16 arithmetic extension in these shaders. `vt_common.glsl`
// requires only 16-bit STORAGE, and its `vt_f16_to_f32` / `vt_f32_to_f16` are
// integer transcriptions of `src/vt/dtype.cpp` — the same functions the host
// `RoundHalf` calls. The device therefore rounds where the host rounds rather
// than where hardware fp16 would.
//
// The comparison is INTEGER: the fp16 (or f32) output words are compared as bit
// patterns. That is what makes it a decode gate as well as a GEMM gate. A single
// mis-decoded codeword changes an f32 accumulation and the stored bits with it,
// and the codebook and bits arguments are proven to REACH the device decode by
// the two discrimination cases below, which require the arms to DISAGREE with
// each other on one trellis.
//
// Every device case SKIPS when no Vulkan backend is registered, says so, and
// STILL ASSERTS the precondition it skipped on. `assertions: 0` printed under
// `SUCCESS!` is a skip wearing a pass, and this family has already paid for that
// once.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rng;

bool HasVulkanBackend() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kVULKAN);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

bool HasVulkanCast() {
  return HasVulkanBackend() && vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kVULKAN);
}

bool HasVulkanGemm() {
  return HasVulkanBackend() && vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kVULKAN);
}

vt::Queue CpuQueue() { return vt::GetBackend(vt::DeviceType::kCPU).CreateQueue(); }

// A `const char*` handed to doctest's MESSAGE stringifies as a BOOL and prints
// "1", which turns an honest skip notice into noise. std::string prints.
const std::string kNoDevice =
    "SKIPPED, no Vulkan device: BACKEND-VULKAN-EXL3 parity is PENDING. This "
    "suite needs NO GPU and NO lease — llvmpipe (mesa-vulkan-drivers, "
    "/usr/share/vulkan/icd.d/lvp_icd.json) is a conformant Vulkan 1.1 device and "
    "is what the gate ran on. Reproduce with: cmake -S . -B build -G Ninja "
    "-DVLLM_CPP_VULKAN=ON && ctest --test-dir build -R '^test_exl3_vulkan$' "
    "--output-on-failure";

// One EXL3 linear evaluated on the CPU arm, which is this suite's oracle AND
// this row's donor. `c_f32` selects upstream's `c_fp32` arm.
//
// THE ORACLE IS NOT SELF-GATED, which is the answer to "a byte gate against your
// own donor proves nothing". `tests/vt/test_exl3_gemm.cpp` holds this same host
// arm against an INDEPENDENT f64 reference built from definitions rather than
// from the implementation — `exl3_fixture.h` `SylvesterH` (popcount parity) and
// `HadRefBlock` (the docstring at hadamard.cu:83-86). This file gates the
// TRANSCRIPTION; that file gates the thing transcribed.
std::vector<uint8_t> CpuGemm(const Exl3Fixture& f, const std::vector<uint16_t>& a_h, int64_t m,
                             int codebook, bool c_f32) {
  vt::Queue hq = CpuQueue();
  const int64_t k = f.k, n = f.n;
  std::vector<uint16_t> a_had(a_h.size(), 0);
  std::vector<uint16_t> a_copy = a_h;
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = codebook;
  std::vector<uint8_t> out(static_cast<size_t>(m * n) * (c_f32 ? 4 : 2), 0);
  vt::Tensor ta = vt::Tensor::Contiguous(a_copy.data(), vt::DType::kF16, hq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(a_had.data(), vt::DType::kF16, hq.device, {m, k});
  vt::Tensor tc = vt::Tensor::Contiguous(
      out.data(), c_f32 ? vt::DType::kF32 : vt::DType::kF16, hq.device, {m, n});
  vt::Tensor tb =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()), vt::DType::kI8, hq.device,
                             {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()), vt::DType::kF16, hq.device, {k});
  vt::Tensor tsvh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()), vt::DType::kF16, hq.device, {n});
  vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, args);
  return out;
}

// The same linear on the Vulkan device. `alias_a_had` runs the input transform
// IN PLACE over `A`, which the op's contract permits and which no CPU call here
// can exercise, because in-place is a device-buffer property.
std::vector<uint8_t> VulkanGemm(const Exl3Fixture& f, const std::vector<uint16_t>& a_h, int64_t m,
                                int codebook, bool c_f32, bool alias_a_had) {
  vt::Backend& vb = vt::GetBackend(vt::DeviceType::kVULKAN);
  vt::Queue dq = vb.CreateQueue();
  const int64_t k = f.k, n = f.n;
  const size_t ab = a_h.size() * sizeof(uint16_t);
  const size_t bb = f.trellis.size() * sizeof(uint16_t);
  const size_t cbytes = static_cast<size_t>(m * n) * (c_f32 ? 4u : 2u);
  void* d_a = vb.Alloc(ab);
  void* d_ah = alias_a_had ? d_a : vb.Alloc(ab);
  void* d_b = vb.Alloc(bb);
  void* d_suh = vb.Alloc(f.suh.size() * sizeof(uint16_t));
  void* d_svh = vb.Alloc(f.svh.size() * sizeof(uint16_t));
  void* d_c = vb.Alloc(cbytes);
  vb.Copy(dq, d_a, a_h.data(), ab);
  vb.Copy(dq, d_b, f.trellis.data(), bb);
  vb.Copy(dq, d_suh, f.suh.data(), f.suh.size() * sizeof(uint16_t));
  vb.Copy(dq, d_svh, f.svh.data(), f.svh.size() * sizeof(uint16_t));
  // POISON. Not zero: a zeroed buffer is what a dead dispatch leaves behind and
  // is also a plausible GEMM answer, so zero cannot distinguish the two.
  vb.Memset(dq, d_c, 0xA5, cbytes);

  vt::Tensor ta = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tb =
      vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device, {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
  vt::Tensor tc =
      vt::Tensor::Contiguous(d_c, c_f32 ? vt::DType::kF32 : vt::DType::kF16, dq.device, {m, n});
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = codebook;
  vt::Exl3Gemm(dq, tc, ta, tb, tsuh, tsvh, tah, args);
  vb.Synchronize(dq);
  std::vector<uint8_t> out(cbytes, 0);
  vb.Copy(dq, out.data(), d_c, cbytes);
  vb.Synchronize(dq);
  vb.Free(d_a);
  if (!alias_a_had) vb.Free(d_ah);
  vb.Free(d_b);
  vb.Free(d_suh);
  vb.Free(d_svh);
  vb.Free(d_c);
  vb.DestroyQueue(dq);
  return out;
}

int FirstDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  if (a.size() != b.size()) return 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return static_cast<int>(i);
  return -1;
}

}  // namespace

// ─── V1: kCastF16 ────────────────────────────────────────────────────────────

TEST_CASE("exl3 vulkan: CastF16 is registered on Vulkan, like its two siblings") {
  // The op-table half of #2530's second reference-tier hit. `kCastBf16` and
  // `kCastF32` were registered on this backend from the W0 skeleton; `kCastF16`
  // was not, while all three are ONE kernel differing by a specialization
  // constant. This case is what reds if the registrar entry is deleted while the
  // kernel still compiles.
  if (!HasVulkanBackend()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kVULKAN));
    return;
  }
  CHECK(vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kVULKAN));
  CHECK(vt::OpRegistered(vt::OpId::kCastBf16, vt::DeviceType::kVULKAN));
  CHECK(vt::OpRegistered(vt::OpId::kCastF32, vt::DeviceType::kVULKAN));
}

TEST_CASE("exl3 vulkan: CastF16 is BYTE-identical to the CPU arm, from f32 and from bf16") {
  if (!HasVulkanCast()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kVULKAN));
    return;
  }
  vt::Backend& vb = vt::GetBackend(vt::DeviceType::kVULKAN);
  vt::Queue dq = vb.CreateQueue();
  vt::Queue hq = CpuQueue();
  const int64_t rows = 5, cols = 71;  // deliberately not a multiple of the block
  const int64_t nelem = rows * cols;

  Rng rng;
  rng.s = 0x9E3779B9u;
  std::vector<float> src_f32(static_cast<size_t>(nelem));
  for (auto& v : src_f32) v = rng.next(6.0f);
  // Values a plain conversion agrees on are not the interesting ones. These are:
  // an exact fp16 tie of each parity (rounds to EVEN), a value below the
  // subnormal floor, a subnormal, an overflow, and a signed zero. They are what
  // separates vt_common.glsl's transcribed codec from the driver's own
  // packHalf2x16, whose subnormal handling and rounding are NOT contractually
  // vt::F32ToF16's — which is exactly why that codec was transcribed.
  src_f32[0] = 1.0f + 1.0f / 2048.0f;
  src_f32[1] = 3.0f + 3.0f / 2048.0f;
  src_f32[2] = 5.0e-8f;
  src_f32[3] = 1.0e-6f;
  src_f32[4] = 1.0e6f;
  src_f32[5] = -0.0f;

  std::vector<uint16_t> src_bf16(static_cast<size_t>(nelem));
  for (int64_t i = 0; i < nelem; ++i) src_bf16[i] = vt::F32ToBF16(src_f32[i]);

  for (int arm = 0; arm < 2; ++arm) {
    const bool from_f32 = arm == 0;
    CAPTURE(from_f32);
    const size_t src_bytes =
        static_cast<size_t>(nelem) * (from_f32 ? sizeof(float) : sizeof(uint16_t));
    const void* src = from_f32 ? static_cast<const void*>(src_f32.data())
                               : static_cast<const void*>(src_bf16.data());
    const vt::DType sdt = from_f32 ? vt::DType::kF32 : vt::DType::kBF16;

    std::vector<uint16_t> host_out(static_cast<size_t>(nelem), 0);
    vt::Tensor hi = vt::Tensor::Contiguous(const_cast<void*>(src), sdt, hq.device, {rows, cols});
    vt::Tensor ho =
        vt::Tensor::Contiguous(host_out.data(), vt::DType::kF16, hq.device, {rows, cols});
    vt::CastF16(hq, ho, hi);

    void* d_in = vb.Alloc(src_bytes);
    void* d_out = vb.Alloc(static_cast<size_t>(nelem) * sizeof(uint16_t));
    vb.Copy(dq, d_in, src, src_bytes);
    std::memset(d_out, 0xFF, static_cast<size_t>(nelem) * sizeof(uint16_t));  // poison
    vt::Tensor di = vt::Tensor::Contiguous(d_in, sdt, dq.device, {rows, cols});
    vt::Tensor dof = vt::Tensor::Contiguous(d_out, vt::DType::kF16, dq.device, {rows, cols});
    vt::CastF16(dq, dof, di);
    vb.Synchronize(dq);
    std::vector<uint16_t> dev_out(static_cast<size_t>(nelem), 0);
    vb.Copy(dq, dev_out.data(), d_out, dev_out.size() * sizeof(uint16_t));
    vb.Synchronize(dq);
    vb.Free(d_in);
    vb.Free(d_out);

    int mismatches = 0;
    for (size_t i = 0; i < host_out.size(); ++i)
      if (host_out[i] != dev_out[i]) ++mismatches;
    CHECK(mismatches == 0);
    // Not vacuous: the cast must have produced something other than the poison
    // the device buffer was filled with.
    bool any_nonpoison = false;
    for (uint16_t v : dev_out)
      if (v != 0xFFFF) any_nonpoison = true;
    CHECK(any_nonpoison);
  }
  vb.DestroyQueue(dq);
}

// ─── V2: kExl3Gemm ───────────────────────────────────────────────────────────

TEST_CASE("exl3 vulkan: exl3_gemm is registered on Vulkan") {
  if (!HasVulkanBackend()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kVULKAN));
    return;
  }
  CHECK(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kVULKAN));
  // NOT registered, and deliberately: the shader exists as steps 1 and 3 of the
  // GEMM, but no dense forward path calls the op, and a registration nothing
  // reaches is what .agents/reachability.md exists to prevent. Asserted so the
  // absence is a DECISION in the gate rather than an oversight nobody notices.
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3HadR128, vt::DeviceType::kVULKAN));
}

TEST_CASE("exl3 vulkan: exl3_gemm is BYTE-identical to the CPU arm on every codebook and width") {
  if (!HasVulkanGemm()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kVULKAN));
    return;
  }
  struct Arm {
    int bits;
    int codebook;
    int64_t m;
    bool c_f32;
    const char* what;
  };
  // Every (bits, codebook) pair the tree instantiates, plus the row counts that
  // select each grid arm: m = 1 is the decode shape a served token actually has,
  // m = 3 is one 8-row workgroup block, m = 20 spans three of them and leaves the
  // last one PARTIALLY live, which is what exercises the `row < m` guard.
  const Arm arms[] = {
      {3, 0, 1, false, "a stock exl3 body at batch 1, the shape a decode step has"},
      {3, 0, 20, false, "the same, over three row-blocks of the grid, the last partial"},
      {6, 0, 3, false, "a stock exl3 lm_head"},
      {3, 1, 3, false, "the SparkInfer DeepSeek-V4 marker"},
      {3, 1, 3, true, "the same, into upstream's c_fp32 arm"},
      {4, 2, 3, false, "the Qwen3.8-27B mul1 GDN tower and attention, 270 of its 409"},
      {5, 2, 3, false, "its 5-bit tensor, and all 36 of the draft"},
      {6, 2, 3, false, "its mul1 lm_head"},
  };
  for (const Arm& arm : arms) {
    CAPTURE(arm.bits);
    CAPTURE(arm.codebook);
    CAPTURE(arm.m);
    CAPTURE(arm.c_f32);
    const int64_t k = 256, n = 128;
    const Exl3Fixture f = MakeFixture(k, n, arm.bits, 0x1D0C0DEu + arm.bits * 7 + arm.codebook);
    Rng rng;
    rng.s = 0xC0FFEEu + arm.bits;
    std::vector<uint16_t> a_h(static_cast<size_t>(arm.m * k));
    for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

    const std::vector<uint8_t> host = CpuGemm(f, a_h, arm.m, arm.codebook, arm.c_f32);
    const std::vector<uint8_t> dev = VulkanGemm(f, a_h, arm.m, arm.codebook, arm.c_f32, false);
    const int diff = FirstDiff(host, dev);
    // std::string, not the bare `const char*`: doctest stringifies a char* as a
    // BOOL and would print "1" where the arm's description belongs.
    MESSAGE("bits ", arm.bits, " cb ", arm.codebook, " m ", arm.m, " (", std::string(arm.what),
            "): first differing byte = ", diff);
    // -1 means "no differing byte". INTEGER equality over the stored bits, not a
    // tolerance: see the head of this file for why this arm can claim it.
    CHECK(diff == -1);
    // Not vacuous: an all-zero output would compare equal and prove nothing, and
    // neither would one the kernel never wrote.
    bool nonzero = false;
    for (uint8_t b : host) nonzero = nonzero || b != 0;
    REQUIRE(nonzero);
    bool any_nonpoison = false;
    for (uint8_t b : dev) any_nonpoison = any_nonpoison || b != 0xA5;
    REQUIRE(any_nonpoison);
  }
}

TEST_CASE("exl3 vulkan: the codebook argument REACHES the device decode") {
  // The byte gate above is also the decode gate — a single mis-decoded codeword
  // moves an f32 accumulation and the stored bits with it. That argument holds
  // only if the shader actually READS `args.codebook`; a kernel that hardcoded
  // one codebook would pass every row above whose codebook happened to be that
  // one. So: one trellis, three codebooks, and the three results must DIFFER
  // from each other on the device exactly as they differ on the host.
  if (!HasVulkanGemm()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kVULKAN));
    return;
  }
  const int64_t m = 2, k = 256, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x51ED2A17u);
  Rng rng;
  rng.s = 0x2545F491u;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

  std::vector<std::vector<uint8_t>> dev;
  for (int cb = 0; cb < 3; ++cb) {
    const std::vector<uint8_t> d = VulkanGemm(f, a_h, m, cb, false, false);
    CHECK(FirstDiff(CpuGemm(f, a_h, m, cb, false), d) == -1);
    dev.push_back(d);
  }
  CHECK(FirstDiff(dev[0], dev[1]) != -1);
  CHECK(FirstDiff(dev[0], dev[2]) != -1);
  CHECK(FirstDiff(dev[1], dev[2]) != -1);
}

TEST_CASE("exl3 vulkan: the bits argument REACHES the device decode") {
  // The same argument as the codebook case, on the other value that selects
  // device behaviour. `bits` sets the trellis stride AND the tail-biting window,
  // so a shader that ignored it would read the wrong 16 bits for every weight
  // past the first — and would still produce a plausible-looking answer.
  if (!HasVulkanGemm()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kVULKAN));
    return;
  }
  const int64_t m = 2, k = 256, n = 128;
  Rng rng;
  rng.s = 0x27D4EB2Fu;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

  std::vector<std::vector<uint8_t>> dev;
  for (int bits = 3; bits <= 5; ++bits) {
    // ONE seed across the widths, so the trellis BYTES that exist at every width
    // are the same bytes and the difference below is the reading of them.
    const Exl3Fixture f = MakeFixture(k, n, bits, 0x51ED2A17u);
    const std::vector<uint8_t> d = VulkanGemm(f, a_h, m, 0, false, false);
    CHECK(FirstDiff(CpuGemm(f, a_h, m, 0, false), d) == -1);
    dev.push_back(d);
  }
  CHECK(FirstDiff(dev[0], dev[1]) != -1);
  CHECK(FirstDiff(dev[1], dev[2]) != -1);
}

TEST_CASE("exl3 vulkan: the input transform runs IN PLACE when a_had aliases A") {
  // `Exl3Gemm`'s contract permits `a_had` to alias `a`, and upstream says the
  // same of its own kernel ("Works inplace if y == x", hadamard.cu:86). No CPU
  // call can exercise it meaningfully; on a device it is a claim about BARRIERS,
  // because every lane must finish LOADING its 128-block before any lane stores.
  if (!HasVulkanGemm()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kVULKAN));
    return;
  }
  const int64_t m = 3, k = 256, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x2C1B3C6Du);
  Rng rng;
  rng.s = 0x85EBCA6Bu;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));
  const std::vector<uint8_t> host = CpuGemm(f, a_h, m, 1, false);
  const std::vector<uint8_t> aliased = VulkanGemm(f, a_h, m, 1, false, /*alias_a_had=*/true);
  CHECK(FirstDiff(host, aliased) == -1);
}

TEST_CASE("exl3 vulkan: a full Exl3Gemm on a Vulkan queue takes ZERO reference-tier hits") {
  // THE REACHABILITY CLAIM, and the number #2530 slice 1 measured at TWO. It is
  // what moves when the two `RegisterOp` lines are deleted, which no byte
  // comparison can detect: the reference tier would serve the call and produce
  // the RIGHT ANSWER, slowly, and every case above would still pass.
  //
  // The counter is process-wide and counts DISTINCT (op, device) resolutions, so
  // it is read as a DELTA across this one call rather than as an absolute.
  if (!HasVulkanGemm()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kVULKAN));
    return;
  }
  const int64_t m = 1, k = 256, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x6C078965u);
  Rng rng;
  rng.s = 0x3C6EF35Fu;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

  const unsigned long long before = vt::GetReferenceTierHits();
  const std::vector<uint8_t> dev = VulkanGemm(f, a_h, m, 0, false, false);
  const unsigned long long after = vt::GetReferenceTierHits();
  MESSAGE("reference-tier hits across one Vulkan Exl3Gemm: before=", before, " after=", after);
  CHECK(after == before);
  // Not vacuous: the call has to have actually produced the right answer, or
  // "zero fallbacks" would be a statement about a call that did nothing.
  CHECK(FirstDiff(CpuGemm(f, a_h, m, 0, false), dev) == -1);
}
