// The CUDA DEQUANTIZING GATHER (KGATHER): `vt::Embedding` over a
// block-quantized table on a device.
//
// WHY THIS GATE EXISTS. `EmbeddingKernelCuda` asserted `f32/bf16` until this
// change, so `DeviceQuantGatherSupported` admitted `kCPU` alone and a kept
// gather table expanded at load on every device. For the one published
// Qwen3.8-Flash-Next artifact that fits this fleet that expansion is 26.822 GiB
// of IQ4_NL becoming 95.368 GiB of bf16 against ~119.6 GiB of usable memory, so
// the refusal was not a missing optimisation: it was the reason the model has no
// GPU arm at all.
//
// WHAT THE ORACLE IS. The CPU arm. It is itself gated bit-exactly against the
// pinned llama.cpp decoding REAL bytes of the shipped tensor
// (tests/vt/test_ops_embedding_quant.cpp), so "CUDA == CPU" chains to upstream
// rather than terminating in self-consistency. The comparison is BIT-EXACT and
// not a tolerance: both sides evaluate the same scalar expressions in the same
// order, the CPU side is compiled `-ffp-contract=off` and the device decoders
// use `__fmul_rn`/`__fsub_rn` wherever a multiply meets a subtract, so any
// difference at all is a defect rather than a rounding budget. A tolerance here
// would hide exactly the failure this row spent a day on: a decode that
// produces plausible numbers and no crash.
//
// WHAT THE TABLE BYTES ARE. Uniformly random, redrawn per block until the CPU
// decode of that block is finite. That is deliberate: it needs NO knowledge of
// where each encoding keeps its scale, so the generator cannot share a layout
// mistake with the decoder under test, and random codebook indices exercise the
// device grids far more widely than a hand-built block would. Every index the
// decoders form is masked into its grid's range, so arbitrary bytes are always
// a legal block.
// This suite OWNS its main so it can exit 77. doctest returns 0 after a case
// returns early, printing "assertions: 0 | 0 passed | 0 failed" and
// "Status: SUCCESS!", which in a log or a `&&` chain is indistinguishable from a
// gate that ran on a GPU and matched an oracle (issue #463; tests/CMakeLists.txt
// documents 77 as this tree's answer). Every value case below returns early
// without a CUDA device, so on a CPU lane this printed a green that described
// nothing -- the exact trap this row named on the thor run, one level up. The
// static `vllm_test_main` is still linked; a translation unit that defines main
// simply keeps the archive member from being extracted.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"  // vt::OpRegistered — the seam the policy queries
#include "vt/ops.h"
#include "vt/quant.h"  // vt::cpu::BlockToFloat
#include "vt/tensor.h"

#ifdef VLLM_CPP_CUDA
#include "vt/cuda/cuda_embedding_quant.h"  // EmbeddingQuantSupported
#endif

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
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
    stride *= t.shape[i];
  }
  return t;
}

// Every block encoding the loader can hand a gather, i.e. every dtype
// `vt::cpu::BlockToFloat` answers for. The list is restated here rather than
// derived so that a dtype which quietly LOSES its decoder shows up as a missing
// case instead of a silently shorter loop.
const DType kGatherDTypes[] = {
    DType::kQ4_0,   DType::kQ5_0,     DType::kQ8_0,   DType::kQ2_K,
    DType::kQ3_K,   DType::kQ4_K,     DType::kQ5_K,   DType::kQ6_K,
    DType::kQ8_K,   DType::kIQ2_XXS,  DType::kIQ3_XXS, DType::kIQ2_S,
    DType::kIQ2_XS, DType::kIQ1_S,    DType::kIQ1_XXXS, DType::kIQ4_NL,
    DType::kIQ4_XS, DType::kMXFP4,
};

// Random block bytes whose CPU decode is finite. A block is redrawn (not
// patched) when it decodes to a NaN or an infinity, which happens when random
// bits land on an f16 exponent of 0x1F; patching would need to know where that
// f16 lives, and this generator deliberately knows nothing about layout.
std::vector<uint8_t> RandomFiniteBlocks(DType dt, int64_t rows, int64_t k,
                                        uint32_t seed) {
  const int64_t be = vt::BlockElems(dt);
  const int64_t bb = vt::BlockBytes(dt);
  const int64_t nb_row = k / be;
  std::vector<uint8_t> bytes(static_cast<size_t>(rows * nb_row * bb));
  std::mt19937 rng(seed);
  std::vector<float> probe(static_cast<size_t>(be));
  const vt::cpu::ToFloatFn to_float = vt::cpu::BlockToFloat(dt);
  REQUIRE(to_float != nullptr);
  for (int64_t b = 0; b < rows * nb_row; ++b) {
    uint8_t* blk = bytes.data() + static_cast<size_t>(b * bb);
    for (int attempt = 0;; ++attempt) {
      for (int64_t i = 0; i < bb; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
      to_float(blk, probe.data(), be);
      bool finite = true;
      for (float v : probe) finite = finite && std::isfinite(v);
      if (finite) break;
      // 32/1024 of f16 bit patterns are inf/NaN, so this converges in a handful
      // of draws; a bound turns a hang into a named failure if that ever stops
      // being true for a new encoding.
      REQUIRE(attempt < 200);
    }
  }
  return bytes;
}

// The gather the CUDA kernel must reproduce: `vt::Embedding` on the CPU, over
// the SAME table bytes and the SAME ids.
std::vector<float> CpuGather(DType dt, const std::vector<uint8_t>& blocks,
                             int64_t rows, int64_t k,
                             const std::vector<int32_t>& ids) {
  Tensor table = Tensor::Contiguous(const_cast<uint8_t*>(blocks.data()), dt, Cpu(),
                                    {rows, k});
  std::vector<int32_t> id_buf = ids;
  Tensor tids = Tensor::Contiguous(id_buf.data(), DType::kI32, Cpu(),
                                   {static_cast<int64_t>(ids.size())});
  std::vector<float> out(ids.size() * static_cast<size_t>(k), 0.0F);
  Tensor tout = Tensor::Contiguous(out.data(), DType::kF32, Cpu(),
                                   {static_cast<int64_t>(ids.size()), k});
  Queue q{Cpu(), nullptr};
  vt::Embedding(q, tout, table, tids);
  return out;
}

uint32_t Bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

}  // namespace

// The capability equality the GGUF residency policy rests on. `RouteGgufTensor`
// asks `KeepQuantGatherDType` (a CPU row decoder exists) and then
// `DeviceQuantGatherSupported` (a DEVICE predicate with no dtype). That pair is
// only sound while the two lists are the SAME list; if CUDA ever decodes fewer
// encodings than the CPU, the policy admits a table the device cannot read and
// the load succeeds into a forward-time throw with the whole model resident --
// the exact failure `DeviceKeepQuantSupported` already carries a per-device set
// to avoid.
// THE TRIPWIRE, and it runs on EVERY build.
//
// `DeviceQuantGatherSupported` is dtype-BLIND: it answers for the device and not
// for the encoding, while `KeepQuantGatherDType` admits anything with a CPU
// `BlockToFloat`. That is sound only while the two sets are EQUAL. A 19th block
// dtype that gained a CPU decoder and no CUDA codec would pass the loader gate,
// be kept quantized, and then throw at the FIRST FORWARD with the whole model
// resident -- the #523 failure the sibling `DeviceKeepQuantSupported` avoids by
// carrying a per-device dtype set.
//
// The equality itself can only be asserted where the CUDA list is linked, so the
// case below is `#ifdef`-guarded. This one is not: it pins the CPU side to a
// NAMED set, so adding a 19th decoder reds on a plain CPU build -- the lane
// everyone runs -- and the failure message says what else has to move. Without
// it the whole invariant rested on a CUDA lane, which is how a dtype-blind gate
// goes wrong quietly.
TEST_CASE("the CPU row-decoder set is PINNED, because the device gate is dtype-blind") {
  const DType kDecodable[] = {
      DType::kQ4_0,   DType::kQ5_0,     DType::kQ8_0,     DType::kQ2_K,
      DType::kQ3_K,   DType::kQ4_K,     DType::kQ5_K,     DType::kQ6_K,
      DType::kQ8_K,   DType::kIQ2_XXS,  DType::kIQ3_XXS,  DType::kIQ2_S,
      DType::kIQ2_XS, DType::kIQ1_S,    DType::kIQ1_XXXS, DType::kIQ4_NL,
      DType::kIQ4_XS, DType::kMXFP4,
  };
  int decodable = 0;
  for (int i = 0; i <= static_cast<int>(DType::kIQ4_XS); ++i) {
    const DType dt = static_cast<DType>(i);
    if (vt::cpu::BlockToFloat(dt) != nullptr) ++decodable;
  }
  // If this fires you added (or removed) a CPU row decoder. A NEW one also owes
  // a codec in src/vt/cuda/cuda_quant_dequant.cuh and a row in
  // VT_DQ_GATHER_TYPES, or the CUDA gather silently stops covering what the
  // loader admits -- see issue #2394 and the equality case below.
  CHECK(decodable == static_cast<int>(std::size(kDecodable)));
  for (const DType dt : kDecodable) {
    CAPTURE(std::string(vt::Name(dt)));
    CHECK(vt::cpu::BlockToFloat(dt) != nullptr);
  }
}

TEST_CASE("CUDA gather decodes exactly the encodings the CPU decoder does") {
#ifndef VLLM_CPP_CUDA
  MESSAGE("CPU-only build: the device decoder list is not linked; skipped");
#else
  // Not gated on a device being PRESENT: this is a property of the build, and
  // asking it on a CUDA build with no card is still a real answer.
  // `DType` has no count sentinel; kIQ4_XS is its last enumerator. A dtype
  // appended after it is NOT covered here, which is why the sweep is written
  // against a named endpoint rather than a literal.
  for (int i = 0; i <= static_cast<int>(DType::kIQ4_XS); ++i) {
    const DType dt = static_cast<DType>(i);
    CAPTURE(std::string(vt::Name(dt)));
    CHECK(vt::cuda::EmbeddingQuantSupported(dt) ==
          (vt::cpu::BlockToFloat(dt) != nullptr));
  }
#endif
}

// The SEAM is reached, not merely present. `vt::Embedding` routes a block table
// to `OpId::kEmbeddingQuant` (vt/ops.cpp) and the GGUF residency policy asks
// `OpRegistered(kEmbeddingQuant, dev)` rather than naming a device, so if the
// CUDA registrar did not link, the whole chain would silently answer "this
// device cannot gather quantized" and every table would go back to expanding to
// bf16 — with no throw, no log, and every value gate in this file still green
// because it would simply skip. That is the failure this case exists to catch.
TEST_CASE("the CUDA block gather is REGISTERED, so the residency policy can find it") {
#ifndef VLLM_CPP_CUDA
  MESSAGE("CPU-only build: the CUDA registrar is not linked; skipped");
#else
  // The CPU arm, which the residency policy compares against. A build property,
  // not a device property: it holds with no card present, which is what makes
  // this a check on the registrar rather than on the host.
  CHECK(vt::OpRegistered(vt::OpId::kEmbeddingQuant, DeviceType::kCPU));

  // PRODUCTION registers the CUDA block gather (cuda_ops.cu). This is the
  // residency flip itself: `DeviceQuantGatherSupported` is
  // `OpRegistered(kEmbeddingQuant, dev)`, so this one registration is what lets
  // a GGUF gather table stay block-resident on the card. It landed only after
  // the value cases below ran green on a real GPU — thor:gpu0, sm_110, 231
  // assertions — because a registration over unexecuted decoders would route
  // every CUDA GGUF model's gather table into them silently.
  //
  // No test-scope registration exists any more: a test that registered the op
  // itself would be measuring its own setup rather than the product.
  CHECK(vt::OpRegistered(vt::OpId::kEmbeddingQuant, DeviceType::kCUDA));

  // The four with no decoder must NOT be registered in any build: registering
  // the id without writing the decoder converts a clean load-time refusal into a
  // forward-time throw with the whole model resident.
  for (DeviceType d : {DeviceType::kMETAL, DeviceType::kVULKAN, DeviceType::kROCM,
                       DeviceType::kTENSTORRENT}) {
    CAPTURE(vt::DeviceTypeName(d));
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kEmbeddingQuant, d));
  }
#endif
}

TEST_CASE("CUDA gather == CPU gather, bit-exactly, on every block encoding") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA gather gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();

  uint32_t seed = 1234;
  for (const DType dt : kGatherDTypes) {
    CAPTURE(std::string(vt::Name(dt)));
    const int64_t be = vt::BlockElems(dt);
    // Two row widths per encoding: ONE block (the shipped n-gram table's row is
    // 160 = five IQ4_NL blocks, so a single-block row is the narrow end) and
    // five, which is that table's own shape for the 32-element encodings and a
    // 1280-wide row for the 256-element ones. A kernel that mis-strides by
    // ELEMENTS instead of RowSizeBytes passes neither.
    for (const int64_t nb_row : {int64_t{1}, int64_t{5}}) {
      const int64_t k = nb_row * be;
      const int64_t rows = 17;  // not a multiple of any launch tile
      CAPTURE(k);
      const std::vector<uint8_t> blocks = RandomFiniteBlocks(dt, rows, k, seed++);

      // Ids repeat, run backwards, and touch both ends, so a kernel that walked
      // the table in order or ignored `id` cannot pass.
      const std::vector<int32_t> ids = {16, 0, 7, 7, 3, 16, 1, 0, 15, 2, 9};
      const std::vector<float> cpu = CpuGather(dt, blocks, rows, k, ids);

      void* d_tab = gpu.Alloc(blocks.size());
      void* d_ids = gpu.Alloc(ids.size() * sizeof(int32_t));
      void* d_out = gpu.Alloc(ids.size() * static_cast<size_t>(k) * sizeof(float));
      gpu.Copy(gq, d_tab, blocks.data(), blocks.size());
      gpu.Copy(gq, d_ids, ids.data(), ids.size() * sizeof(int32_t));
      Tensor table = DevTensor(d_tab, dt, {rows, k});
      Tensor tids =
          DevTensor(d_ids, DType::kI32, {static_cast<int64_t>(ids.size())});
      Tensor tout =
          DevTensor(d_out, DType::kF32, {static_cast<int64_t>(ids.size()), k});
      vt::Embedding(gq, tout, table, tids);
      std::vector<float> cuda(cpu.size(), 0.0F);
      gpu.Copy(gq, cuda.data(), d_out, cuda.size() * sizeof(float));
      gpu.Synchronize(gq);
      gpu.Free(d_tab);
      gpu.Free(d_ids);
      gpu.Free(d_out);

      // Report the WORST difference and where, then assert exactness. A bare
      // per-element CHECK on 11 x 1280 elements would bury the signal in
      // 14,080 lines when it fires.
      double max_abs = 0.0;
      size_t worst = 0;
      size_t mismatches = 0;
      for (size_t i = 0; i < cpu.size(); ++i) {
        if (Bits(cpu[i]) != Bits(cuda[i])) ++mismatches;
        const double d = std::fabs(static_cast<double>(cuda[i]) -
                                   static_cast<double>(cpu[i]));
        if (d > max_abs) {
          max_abs = d;
          worst = i;
        }
      }
      CAPTURE(max_abs);
      CAPTURE(mismatches);
      CAPTURE(worst);
      CAPTURE(cpu.empty() ? 0.0F : cpu[worst]);
      CAPTURE(cuda.empty() ? 0.0F : cuda[worst]);
      CHECK(mismatches == 0);
    }
  }
}

// The bf16 out arm. It is not a second copy of the case above: an f32-exact
// kernel can still be wrong here, because bf16 rounding happens on the DEVICE
// and `DF32ToBF16` is a hand port of `vt::F32ToBF16`. This is also the dtype the
// model path actually uses, and a token gate cannot see a store that rounds
// differently -- it moves the same bytes and usually decides the same token.
TEST_CASE("CUDA gather == CPU gather in bf16 out, bit-exactly") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA gather bf16 gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();

  uint32_t seed = 99001;
  for (const DType dt : kGatherDTypes) {
    CAPTURE(std::string(vt::Name(dt)));
    const int64_t be = vt::BlockElems(dt);
    const int64_t k = 3 * be;
    const int64_t rows = 9;
    const std::vector<uint8_t> blocks = RandomFiniteBlocks(dt, rows, k, seed++);
    const std::vector<int32_t> ids = {8, 0, 5, 5, 1, 8, 3};

    std::vector<uint16_t> cpu(ids.size() * static_cast<size_t>(k), 0);
    {
      Tensor table = Tensor::Contiguous(const_cast<uint8_t*>(blocks.data()), dt,
                                        Cpu(), {rows, k});
      std::vector<int32_t> id_buf = ids;
      Tensor tids = Tensor::Contiguous(id_buf.data(), DType::kI32, Cpu(),
                                       {static_cast<int64_t>(ids.size())});
      Tensor tout = Tensor::Contiguous(cpu.data(), DType::kBF16, Cpu(),
                                       {static_cast<int64_t>(ids.size()), k});
      Queue cq{Cpu(), nullptr};
      vt::Embedding(cq, tout, table, tids);
    }

    void* d_tab = gpu.Alloc(blocks.size());
    void* d_ids = gpu.Alloc(ids.size() * sizeof(int32_t));
    void* d_out = gpu.Alloc(cpu.size() * sizeof(uint16_t));
    gpu.Copy(gq, d_tab, blocks.data(), blocks.size());
    gpu.Copy(gq, d_ids, ids.data(), ids.size() * sizeof(int32_t));
    Tensor table = DevTensor(d_tab, dt, {rows, k});
    Tensor tids = DevTensor(d_ids, DType::kI32, {static_cast<int64_t>(ids.size())});
    Tensor tout =
        DevTensor(d_out, DType::kBF16, {static_cast<int64_t>(ids.size()), k});
    vt::Embedding(gq, tout, table, tids);
    std::vector<uint16_t> cuda(cpu.size(), 0);
    gpu.Copy(gq, cuda.data(), d_out, cuda.size() * sizeof(uint16_t));
    gpu.Synchronize(gq);
    gpu.Free(d_tab);
    gpu.Free(d_ids);
    gpu.Free(d_out);

    size_t mismatches = 0;
    for (size_t i = 0; i < cpu.size(); ++i)
      if (cpu[i] != cuda[i]) ++mismatches;
    CAPTURE(mismatches);
    CHECK(mismatches == 0);
  }
}

// i64 ids. The engine hands i64 on some paths and i32 on others, and the two
// take different template instantiations, so an i32-only gate would leave half
// the dispatch unmeasured.
TEST_CASE("CUDA gather takes i64 ids and agrees with the i32 arm") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA gather i64 gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();

  const DType dt = DType::kIQ4_NL;  // the shipped n-gram table's encoding
  const int64_t be = vt::BlockElems(dt);
  const int64_t k = 5 * be;  // 160: the shipped row width exactly
  const int64_t rows = 13;
  const std::vector<uint8_t> blocks = RandomFiniteBlocks(dt, rows, k, 4242);
  const std::vector<int32_t> ids32 = {12, 0, 6, 6, 2, 12, 1};
  const std::vector<int64_t> ids64(ids32.begin(), ids32.end());
  const std::vector<float> cpu = CpuGather(dt, blocks, rows, k, ids32);

  void* d_tab = gpu.Alloc(blocks.size());
  void* d_ids = gpu.Alloc(ids64.size() * sizeof(int64_t));
  void* d_out = gpu.Alloc(cpu.size() * sizeof(float));
  gpu.Copy(gq, d_tab, blocks.data(), blocks.size());
  gpu.Copy(gq, d_ids, ids64.data(), ids64.size() * sizeof(int64_t));
  Tensor table = DevTensor(d_tab, dt, {rows, k});
  Tensor tids = DevTensor(d_ids, DType::kI64, {static_cast<int64_t>(ids64.size())});
  Tensor tout =
      DevTensor(d_out, DType::kF32, {static_cast<int64_t>(ids64.size()), k});
  vt::Embedding(gq, tout, table, tids);
  std::vector<float> cuda(cpu.size(), 0.0F);
  gpu.Copy(gq, cuda.data(), d_out, cuda.size() * sizeof(float));
  gpu.Synchronize(gq);
  gpu.Free(d_tab);
  gpu.Free(d_ids);
  gpu.Free(d_out);

  size_t mismatches = 0;
  for (size_t i = 0; i < cpu.size(); ++i)
    if (Bits(cpu[i]) != Bits(cuda[i])) ++mismatches;
  CAPTURE(mismatches);
  CHECK(mismatches == 0);
}

// An out-of-range id. The float arm clamps in-kernel and reports through a
// deferred device flag; the block arm writes the SAME latch, so the report has
// to survive the new path. Without this the block kernel could clamp silently
// and every id gate would still be green.
TEST_CASE("CUDA gather reports an out-of-range id on a block table") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA gather id-range gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();

  const DType dt = DType::kQ4_K;
  const int64_t be = vt::BlockElems(dt);
  const int64_t k = be;
  const int64_t rows = 4;
  const std::vector<uint8_t> blocks = RandomFiniteBlocks(dt, rows, k, 777);
  const std::vector<int32_t> ids = {0, 99};  // 99 >= rows

  void* d_tab = gpu.Alloc(blocks.size());
  void* d_ids = gpu.Alloc(ids.size() * sizeof(int32_t));
  void* d_out = gpu.Alloc(ids.size() * static_cast<size_t>(k) * sizeof(float));
  gpu.Copy(gq, d_tab, blocks.data(), blocks.size());
  gpu.Copy(gq, d_ids, ids.data(), ids.size() * sizeof(int32_t));
  Tensor table = DevTensor(d_tab, dt, {rows, k});
  Tensor tids = DevTensor(d_ids, DType::kI32, {static_cast<int64_t>(ids.size())});
  Tensor tout =
      DevTensor(d_out, DType::kF32, {static_cast<int64_t>(ids.size()), k});

  // The report is DEFERRED by design (cuda_ops.cu, "Out-of-range reporting
  // WITHOUT a per-call barrier"): the offending call itself does not throw, a
  // later one does. Drive the ring until it surfaces, and bound the loop so a
  // report that never arrives is a failure rather than a hang.
  std::string msg;
  for (int i = 0; i < 32 && msg.empty(); ++i) {
    try {
      vt::Embedding(gq, tout, table, tids);
      gpu.Synchronize(gq);
    } catch (const std::exception& e) {
      msg = e.what();
    }
  }
  gpu.Free(d_tab);
  gpu.Free(d_ids);
  gpu.Free(d_out);
  CAPTURE(msg);
  CHECK_FALSE(msg.empty());
  CHECK(msg.find("99") != std::string::npos);
}

// Exit 77 -> CTest reports SKIPPED. Returns the real rc first: a FAILURE must
// never be laundered into a skip, so 77 is reached only on a clean run that had
// no device to run on.
int main(int argc, char** argv) {
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  const int rc = context.run();
  if (context.shouldExit() || rc != 0) return rc;
  if (!HasCuda()) {
    std::fprintf(stderr,
                 "test_cuda_embedding_quant: no CUDA backend on this host — the "
                 "value cases did not run. Exiting 77 (SKIPPED) rather than 0, "
                 "because \"assertions: 0 ... SUCCESS!\" is not a pass.\n");
    return 77;
  }
  return 0;
}
