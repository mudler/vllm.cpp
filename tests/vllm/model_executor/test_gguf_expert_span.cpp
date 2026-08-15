// ENG-EXPERT-STREAM W2 (#912) — the per-expert byte span in a stacked GGUF
// expert tensor.
//
// The failure mode under test does NOT fail loudly. A wrong offset reads
// successfully, the GEMM runs, and the model emits fluent text computed from
// another expert's weights. No exception, no gate, no divergence a token test
// would see. So every case here is about refusing rather than about computing.
//
// Layout contract, stated on the role itself at `gguf_keep_quant.cpp:27`:
// a stacked expert tensor is [E, out, in] and "each expert slice is whole rows
// of the same K". The same arithmetic runs in `OwnGgufQuantBlocks`
// (`qwen3_5_gguf_weights.cpp:57`) for the load path; this unit exists so the
// streaming path cannot drift from it.
#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_expert_span.h"

using vllm::GgufExpertLayoutOf;
using vllm::GgufExpertSpanOf;
using vllm::GgufTensorInfo;

namespace {

// ggml type 8 is Q8_0: 32 elements per block in 34 bytes. Chosen because its
// block size makes the row arithmetic checkable by hand.
constexpr uint32_t kQ8_0 = 8;
constexpr int64_t kBlockElems = 32;
constexpr int64_t kBlockBytes = 34;

// A synthetic stacked tensor. `data` is never dereferenced by the span math,
// only offset, so a fixed non-null base is enough and keeps the test free of
// a real file.
GgufTensorInfo Stacked(int64_t experts, int64_t rows_per_expert, int64_t k,
                       uint32_t type = kQ8_0) {
  GgufTensorInfo t;
  t.name = "blk.0.ffn_gate_exps.weight";
  t.shape = {experts, rows_per_expert, k};
  t.ggml_type = type;
  t.data = reinterpret_cast<const uint8_t*>(0x1000);
  const int64_t row_bytes = (k / kBlockElems) * kBlockBytes;
  t.nbytes = static_cast<size_t>(experts * rows_per_expert * row_bytes);
  return t;
}

}  // namespace

TEST_CASE("layout derives rows, row bytes and expert bytes from the shape") {
  const GgufTensorInfo t = Stacked(/*experts=*/256, /*rows=*/512, /*k=*/2048);
  const auto L = GgufExpertLayoutOf(t, 256);
  CHECK(L.num_experts == 256);
  CHECK(L.rows_per_expert == 512);
  CHECK(L.k == 2048);
  // 2048 / 32 = 64 blocks, 64 * 34 = 2176 bytes per row.
  CHECK(L.row_bytes == 2176);
  CHECK(L.expert_bytes == 512u * 2176u);
  CHECK(L.valid());
}

TEST_CASE("a flattened [E*out, K] tensor gives the same layout as [E, out, K]") {
  // Both spellings reach the loader depending on how the file was written, and
  // a slicer that accepted only one would silently mis-slice the other.
  GgufTensorInfo flat = Stacked(4, 8, 64);
  flat.shape = {4 * 8, 64};
  const auto a = GgufExpertLayoutOf(flat, 4);
  const auto b = GgufExpertLayoutOf(Stacked(4, 8, 64), 4);
  CHECK(a.rows_per_expert == b.rows_per_expert);
  CHECK(a.row_bytes == b.row_bytes);
  CHECK(a.expert_bytes == b.expert_bytes);
}

TEST_CASE("spans are contiguous, equal, and tile the tensor exactly") {
  // If they did not tile exactly, some expert would read part of its neighbour
  // and nothing would report it.
  const GgufTensorInfo t = Stacked(/*experts=*/8, /*rows=*/4, /*k=*/64);
  const auto L = GgufExpertLayoutOf(t, 8);
  size_t covered = 0;
  const uint8_t* prev_end = t.data;
  for (int64_t e = 0; e < 8; ++e) {
    const auto s = GgufExpertSpanOf(t, L, e);
    REQUIRE(s.valid());
    CHECK_MESSAGE(s.data == prev_end, "expert ", e, " must start where the last ended");
    CHECK(s.bytes == L.expert_bytes);
    prev_end = s.data + s.bytes;
    covered += s.bytes;
  }
  CHECK(covered == t.nbytes);
  CHECK(prev_end == t.data + t.nbytes);
}

TEST_CASE("expert e starts at exactly e * expert_bytes") {
  const GgufTensorInfo t = Stacked(16, 2, 32);
  const auto L = GgufExpertLayoutOf(t, 16);
  for (int64_t e : {int64_t{0}, int64_t{1}, int64_t{7}, int64_t{15}}) {
    const auto s = GgufExpertSpanOf(t, L, e);
    CHECK(s.data == t.data + static_cast<size_t>(e) * L.expert_bytes);
  }
}

TEST_CASE("an out-of-range expert throws instead of reading past the tensor") {
  const GgufTensorInfo t = Stacked(4, 2, 32);
  const auto L = GgufExpertLayoutOf(t, 4);
  CHECK_THROWS_AS(GgufExpertSpanOf(t, L, 4), std::out_of_range);
  CHECK_THROWS_AS(GgufExpertSpanOf(t, L, -1), std::out_of_range);
  CHECK_THROWS_AS(GgufExpertSpanOf(t, L, 1 << 20), std::out_of_range);
  // The last valid index still works, so the bound is not off by one.
  CHECK(GgufExpertSpanOf(t, L, 3).valid());
}

TEST_CASE("a layout that would overrun the tensor is refused") {
  // A truncated or mis-declared tensor: the geometry says more bytes than the
  // mapping holds. Reading anyway would return whatever is mapped next.
  GgufTensorInfo t = Stacked(4, 2, 32);
  t.nbytes -= 1;  // one byte short of what the shape implies
  const auto L = GgufExpertLayoutOf(t, 4);
  CHECK_THROWS_AS(GgufExpertSpanOf(t, L, 3), std::invalid_argument);
  // The earlier experts are still wholly inside the mapping and remain usable.
  CHECK(GgufExpertSpanOf(t, L, 0).valid());
}

TEST_CASE("a row count that does not divide by the expert count is refused") {
  // Every offset after the first would be wrong by a growing amount, so this
  // must fail at layout time rather than produce plausible spans.
  GgufTensorInfo t = Stacked(4, 2, 32);
  t.shape = {9, 32};  // 9 rows, 4 experts
  t.nbytes = 9 * (32 / kBlockElems) * kBlockBytes;
  CHECK_THROWS_AS(GgufExpertLayoutOf(t, 4), std::invalid_argument);
}

TEST_CASE("a leading dim that disagrees with num_experts is refused") {
  // The case that isolates THIS check from the divisibility one. [8, 4, 64] has
  // 32 rows, which divides evenly by 4, so the divisibility check is happy and
  // only the leading dim reveals the disagreement.
  //
  // Without this check the layout would silently claim 8 rows per expert where
  // the tensor holds 4, and expert 0 would read its own weights concatenated
  // with expert 1's. The first version of this case used num_experts=7, where
  // 32 % 7 != 0 masked the defect behind the divisibility check.
  const GgufTensorInfo t = Stacked(8, 4, 64);
  CHECK_THROWS_AS(GgufExpertLayoutOf(t, 4), std::invalid_argument);
  CHECK_THROWS_AS(GgufExpertLayoutOf(t, 2), std::invalid_argument);
  // A non-divisible count is refused too, by the other check.
  CHECK_THROWS_AS(GgufExpertLayoutOf(t, 7), std::invalid_argument);
  CHECK_NOTHROW(GgufExpertLayoutOf(t, 8));
}

TEST_CASE("a rank that is not a stacked expert weight is refused") {
  GgufTensorInfo t = Stacked(4, 2, 32);
  t.shape = {32};
  CHECK_THROWS_AS(GgufExpertLayoutOf(t, 4), std::invalid_argument);
  t.shape = {2, 2, 2, 2};
  CHECK_THROWS_AS(GgufExpertLayoutOf(t, 4), std::invalid_argument);
  CHECK_THROWS_AS(GgufExpertLayoutOf(t, 0), std::invalid_argument);
}

TEST_CASE("a K that is not a whole number of blocks is refused") {
  // A partial block means a row boundary falls inside a block, so the slice
  // would be a repack rather than an offset. RowSizeBytes enforces it.
  GgufTensorInfo t = Stacked(2, 2, 64);
  t.shape = {2, 2, 33};  // 33 is not a multiple of the 32-element block
  CHECK_THROWS_AS(GgufExpertLayoutOf(t, 2), std::invalid_argument);
}

TEST_CASE("the Qwen3.8 geometry produces the byte budget the plan assumes") {
  // 512 experts, and the per-expert size is what decides how many fit in a
  // cache budget. This case pins the arithmetic the streaming plan quotes.
  const GgufTensorInfo t = Stacked(/*experts=*/512, /*rows=*/2048, /*k=*/8192);
  const auto L = GgufExpertLayoutOf(t, 512);
  CHECK(L.row_bytes == (8192 / kBlockElems) * kBlockBytes);
  CHECK(L.expert_bytes == 2048u * L.row_bytes);
  // Every expert is reachable and the last one ends exactly at the tensor end.
  const auto last = GgufExpertSpanOf(t, L, 511);
  CHECK(last.data + last.bytes == t.data + t.nbytes);
}
