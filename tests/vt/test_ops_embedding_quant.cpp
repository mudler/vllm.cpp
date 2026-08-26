// The DEQUANTIZING GATHER: `vt::Embedding` over a block-quantized table.
//
// Why this op exists at all. A gather table is not a GEMM weight, so it was
// deliberately excluded from keep-quant residency and expanded to bf16 at load
// (`gguf_keep_quant.cpp`, `KeepQuantKDim` returning -1 for `kEmbeddingTable`).
// That is affordable for a vocab matrix and fatal for an n-gram table:
// `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` ships
// `per_layer_token_embd.weight` as IQ4_NL `[160, 320001536]`, 51.2 G parameters
// — 28.8 GB of blocks, 102.4 GB expanded to bf16 on a box with ~119.6 GiB.
//
// llama.cpp does this without difficulty and this op is a port of how:
// `ggml_compute_forward_get_rows_q` (ggml/src/ggml-cpu/ops.cpp:4850
// @ b10451) dequantizes ONE ROW per gathered id through the type's `to_float`,
// never the table. The per-token cost is `nc` elements of decode; the resident
// table, not the decode, is the expense.
//
// WHAT GATES IT HERE, and why it is not a self-consistency check. The expected
// values are the PINNED ORACLE's own `dequantize_row_iq4_nl` output over REAL
// bytes of that shipped tensor (tests/vt/iq4nl_q5_0_golden_vectors.h), compared
// BIT-EXACTLY. Comparing the gather against `vt::cpu::BlockToFloat` would prove
// only that the gather calls the decoder it calls.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

#include "iq4nl_q5_0_golden_vectors.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

float BitsToF32(uint32_t bits) {
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint32_t F32ToBits(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}

// Gather `ids` out of a block table and compare BIT-EXACTLY against the golden
// f32 the oracle produced for the whole table.
void CheckGatherAgainstOracle(DType dt, const uint8_t* blocks, int64_t rows,
                              int64_t k, const uint32_t* golden,
                              const std::vector<int32_t>& ids) {
  Tensor table = Tensor::Contiguous(const_cast<uint8_t*>(blocks), dt, Cpu(),
                                    {rows, k});
  std::vector<int32_t> id_buf = ids;
  Tensor tids = Tensor::Contiguous(id_buf.data(), DType::kI32, Cpu(),
                                   {static_cast<int64_t>(ids.size())});
  std::vector<float> out(ids.size() * static_cast<size_t>(k), 0.0F);
  Tensor tout = Tensor::Contiguous(out.data(), DType::kF32, Cpu(),
                                   {static_cast<int64_t>(ids.size()), k});
  Queue q{Cpu(), nullptr};
  vt::Embedding(q, tout, table, tids);

  for (size_t t = 0; t < ids.size(); ++t) {
    for (int64_t j = 0; j < k; ++j) {
      CAPTURE(t);
      CAPTURE(j);
      CHECK(F32ToBits(out[t * static_cast<size_t>(k) + static_cast<size_t>(j)]) ==
            golden[static_cast<size_t>(ids[t]) * static_cast<size_t>(k) +
                   static_cast<size_t>(j)]);
    }
  }
}

}  // namespace

TEST_CASE("Embedding gathers IQ4_NL rows bit-exactly against the pinned oracle") {
  // 180 real bytes of `per_layer_token_embd.weight` == 10 IQ4_NL blocks == TWO
  // whole 160-element rows of the shipped n-gram table. The ids repeat and run
  // backwards so a kernel that ignored `id` and walked the table in order, or
  // that mis-strided by ELEMENTS instead of `RowSizeBytes`, cannot pass.
  CheckGatherAgainstOracle(DType::kIQ4_NL, vllm_test::kIq4nlGoldenBlocks,
                           /*rows=*/2, /*k=*/160, vllm_test::kIq4nlGoldenBits,
                           {1, 0, 1, 1, 0});
}

TEST_CASE("Embedding gathers Q5_0 rows bit-exactly against the pinned oracle") {
  // 4 Q5_0 blocks == 88 bytes == two 64-element rows.
  CheckGatherAgainstOracle(DType::kQ5_0, vllm_test::kQ50GoldenBlocks,
                           /*rows=*/2, /*k=*/64, vllm_test::kQ50GoldenBits,
                           {0, 1, 0});
}

TEST_CASE("Embedding gathers a block table into bf16, rounding ONCE") {
  // The residency this op unlocks is not only cheaper, it is SHORTER: the
  // expand-bf16 path rounds at load and the gather then widens that bf16 back
  // to f32, so a value crosses bf16 twice. Gathering from blocks rounds exactly
  // once, at the store. The gate is that the bf16 output equals the ORACLE f32
  // rounded once — not that it equals the old two-step value.
  const int64_t k = 160;
  Tensor table = Tensor::Contiguous(
      const_cast<uint8_t*>(vllm_test::kIq4nlGoldenBlocks), DType::kIQ4_NL,
      Cpu(), {2, k});
  std::vector<int32_t> ids = {1, 0};
  Tensor tids = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {2});
  std::vector<uint16_t> out(2 * static_cast<size_t>(k), 0);
  Tensor tout =
      Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {2, k});
  Queue q{Cpu(), nullptr};
  vt::Embedding(q, tout, table, tids);

  for (size_t t = 0; t < ids.size(); ++t) {
    for (int64_t j = 0; j < k; ++j) {
      CAPTURE(t);
      CAPTURE(j);
      const float want = BitsToF32(
          vllm_test::kIq4nlGoldenBits[static_cast<size_t>(ids[t]) *
                                          static_cast<size_t>(k) +
                                      static_cast<size_t>(j)]);
      CHECK(out[t * static_cast<size_t>(k) + static_cast<size_t>(j)] ==
            vt::F32ToBF16(want));
    }
  }
}

TEST_CASE("an f32-output gather from BLOCKS is NOT the bf16 round-trip") {
  // The residency this row flips is memory-only ONLY while every GGUF-path
  // gather writes bf16, and that is a property of today's call sites, not of
  // the op (#1989 review F3). On an f32 OUTPUT the two residencies genuinely
  // disagree: keeping the blocks decodes straight to f32, while expanding at
  // load rounds to bf16 first and the gather then merely widens that. This
  // case makes the disagreement executable, so a future f32-output gather
  // cannot inherit "tokens do not move" from a claim that was never about it.
  //
  // Both sides are measured against the SAME pinned oracle, so the case also
  // says WHICH one is right: the block gather is bit-exact, and the round-trip
  // is the lossy one.
  const int64_t k = 160;
  Tensor table = Tensor::Contiguous(
      const_cast<uint8_t*>(vllm_test::kIq4nlGoldenBlocks), DType::kIQ4_NL,
      Cpu(), {2, k});
  std::vector<int32_t> ids = {0, 1};
  Tensor tids = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {2});
  std::vector<float> out(2 * static_cast<size_t>(k), 0.0F);
  Tensor tout = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {2, k});
  Queue q{Cpu(), nullptr};
  vt::Embedding(q, tout, table, tids);

  int differs = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const uint32_t oracle = vllm_test::kIq4nlGoldenBits[i];
    // The kept-blocks answer IS the oracle, bit for bit.
    CHECK(F32ToBits(out[i]) == oracle);
    // The expand-bf16 answer is that value rounded through bf16 and widened —
    // what an f32-output gather would have read before this residency existed.
    const float round_trip = vt::BF16ToF32(vt::F32ToBF16(BitsToF32(oracle)));
    if (F32ToBits(round_trip) != oracle) ++differs;
  }
  // Not "may differ": on this real tensor it DOES, for most of the 320 values.
  CHECK(differs > 0);
  CAPTURE(differs);
  CHECK(differs > static_cast<int>(out.size()) / 2);
}

TEST_CASE("Embedding REFUSES a block table whose row is not whole blocks") {
  // `ggml_row_size`'s precondition. A ragged K has no row stride at all, so the
  // gather must refuse rather than mis-stride every row after the first. 160 is
  // 5 whole IQ4_NL blocks; 100 is not a whole number of 32-element blocks.
  //
  // The assertion is on the MESSAGE, and that is the whole point of this case
  // (#1989 review F7). A bare CHECK_THROWS passes on ANY throw, and there is a
  // second refusal one level down — `vt::RowSizeBytes` (src/vt/dtype.cpp:187)
  // rejects the same ragged K from inside the kernel. Neutering the op's own
  // precondition therefore left a bare CHECK_THROWS green: the case named a
  // guard it did not measure. Naming the text discriminates the two.
  Tensor bad = Tensor::Contiguous(
      const_cast<uint8_t*>(vllm_test::kIq4nlGoldenBlocks), DType::kIQ4_NL,
      Cpu(), {2, 100});
  std::vector<int32_t> ids = {0};
  Tensor tids = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {1});
  std::vector<float> out(100, 0.0F);
  Tensor tout = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 100});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_WITH_AS(
      vt::Embedding(q, tout, bad, tids),
      doctest::Contains("embedding: block table K must be a whole number of blocks"),
      std::runtime_error);
}

TEST_CASE("Embedding REFUSES an out-of-range id BEFORE decoding a row") {
  // This case used to be called "REFUSES a block table with no decoder" and it
  // never measured that (#1989 review F7): Q8_K HAS a `BlockToFloat`, and so
  // does every other `vt::DType` block encoding (all 16 of `kBlockDTypes` are
  // listed in `BlockToFloat`), so the kernel's `to_float != nullptr` guard is
  // unreachable BY CONSTRUCTION today and no test can execute it. Saying that
  // here is more useful than a case that claims to pin it.
  //
  // What IS on the block arm and IS reachable is the per-id bounds check, which
  // runs before the decode so a bad id cannot read past the table. Q8_K is a
  // fine table dtype for that: 2 rows of 256 elements, 292 B per block.
  std::vector<uint8_t> blocks(2 * 292, 0);
  Tensor table = Tensor::Contiguous(blocks.data(), DType::kQ8_K, Cpu(), {2, 256});
  std::vector<int32_t> ids = {5};  // out of range for a 2-row table
  Tensor tids = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {1});
  std::vector<float> out(256, 0.0F);
  Tensor tout = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 256});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_WITH_AS(vt::Embedding(q, tout, table, tids),
                       doctest::Contains("embedding: id out of range"),
                       std::runtime_error);
  // An IN-RANGE id on the same table decodes without complaint, so the refusal
  // above is about the id and not about the dtype.
  ids[0] = 1;
  CHECK_NOTHROW(vt::Embedding(q, tout, table, tids));
}
