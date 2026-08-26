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

TEST_CASE("Embedding REFUSES a block table whose row is not whole blocks") {
  // `ggml_row_size`'s precondition. A ragged K has no row stride at all, so the
  // gather must refuse rather than mis-stride every row after the first. 160 is
  // 5 whole IQ4_NL blocks; 100 is not a whole number of 32-element blocks.
  Tensor bad = Tensor::Contiguous(
      const_cast<uint8_t*>(vllm_test::kIq4nlGoldenBlocks), DType::kIQ4_NL,
      Cpu(), {2, 100});
  std::vector<int32_t> ids = {0};
  Tensor tids = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {1});
  std::vector<float> out(100, 0.0F);
  Tensor tout = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 100});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS(vt::Embedding(q, tout, bad, tids));
}

TEST_CASE("Embedding REFUSES a block table with no decoder") {
  // Q8_K is the K-quants' ACTIVATION encoding: it is a block dtype with a
  // `to_float`, but it never appears as a file weight. The gather's admission
  // rule is the decoder's existence, so this documents which side of the line a
  // block dtype without a gather decode would fall on. Bounds are still checked
  // BEFORE any decode, which is what this case actually pins.
  std::vector<uint8_t> blocks(2 * 292, 0);
  Tensor table = Tensor::Contiguous(blocks.data(), DType::kQ8_K, Cpu(), {2, 256});
  std::vector<int32_t> ids = {5};  // out of range for a 2-row table
  Tensor tids = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {1});
  std::vector<float> out(256, 0.0F);
  Tensor tout = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 256});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS(vt::Embedding(q, tout, table, tids));
}
