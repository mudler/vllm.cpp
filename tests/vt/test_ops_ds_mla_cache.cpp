// vt::ConcatAndCacheDsMla / vt::DequantAndGatherDsMla unit tests —
// `KV-DSV4-MULTICACHE` W8 slices 2 and 3 ([#2455]).
//
// Kernels under port, at the parity pin `5559679229` in
// `vllm/models/deepseek_v4/common/ops/cache_utils.py`:
//   `quantize_and_insert_k_kernel`      `:36-159`  (host wrapper `:162-227`)
//   `_dequantize_and_gather_k_kernel`   `:228-341` (host wrapper `:339-389`)
//
// GOLDEN STRATEGY, and why it is not a tolerance. This is INTEGER PACKING: every
// byte the store writes is either an e4m3 code, a UE8M0 exponent byte, half a
// bf16 word, or a byte it must not touch at all. A relative-error gate cannot
// see a page that is 3.5x too wide, a scale region written at the wrong offset,
// or a padding byte left with the previous tenant's data. So the block is filled
// with POISON (0xA5) and every assertion is `==` on bytes.
//
// The oracle is the W8 slice-1 host packer (`Fp8DsMlaEncodeToken` /
// `Fp8DsMlaDecodeToken`, `deepseek_v4_compressor.h`), which is itself the ported
// reference and is gated against hand-derived cases in
// `tests/vllm/models/test_deepseek_v4_compressor.cpp`. Comparing the op against
// it proves what these two slices actually add: the SLOT arithmetic, the block
// stride, the region split across a paged multi-block cache, and the refusals.
//
// ALL HOST-SIDE AND UNCONDITIONAL. There is no `HasCuda()` early return, so a
// no-GPU run reports a real assertion count for every claim here rather than a
// skip wearing a pass.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_compressor.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm::deepseek_v4::Fp8DsMlaDecodeToken;
using vllm::deepseek_v4::Fp8DsMlaEncodeToken;
using vllm::deepseek_v4::Fp8DsMlaLayout;
using vllm::deepseek_v4::Fp8DsMlaPageLayout;
using vllm::deepseek_v4::Fp8DsMlaToken;
using vllm::deepseek_v4::MakeFp8DsMlaLayout;
using vllm::deepseek_v4::MakeFp8DsMlaPageLayout;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

constexpr uint8_t kPoison = 0xA5;
// DeepSeek-V4-Flash C4A: `cache_config.block_size` 256 at `compress_ratio` 4,
// so `storage_block_size()` is 64 (`kv_cache_interface.py:393-395`), which is
// also the 64 upstream's own comments derive at `sparse_swa.py:76-83`.
constexpr int64_t kStorageBlock = 64;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = Cpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

Fp8DsMlaLayout TokenLayout() {
  return MakeFp8DsMlaLayout(vt::kFp8DsMlaNopeDim, vt::kFp8DsMlaRopeDim,
                            vt::kFp8DsMlaQuantBlock);
}

// A latent whose seven NoPE quant blocks have DELIBERATELY different magnitudes,
// so the seven UE8M0 exponent bytes differ. Without that, a scale region written
// at the wrong offset would compare equal to itself and the byte gate would pass
// on a broken layout.
std::vector<float> MakeLatent(int64_t seed, const Fp8DsMlaLayout& L) {
  const int64_t D = L.nope_head_dim + L.rope_head_dim;
  std::vector<float> head(static_cast<size_t>(D));
  for (int64_t b = 0; b < L.n_nope_blocks; ++b) {
    // 2^b spreads the per-block absmax over seven distinct powers of two.
    const float mag = static_cast<float>(1 << b) * (0.5f + 0.125f * static_cast<float>(seed));
    for (int64_t j = 0; j < L.quant_block; ++j) {
      const int64_t d = b * L.quant_block + j;
      const float u = static_cast<float>((d * 37 + seed * 11) % 251) / 251.0f - 0.5f;
      head[static_cast<size_t>(d)] = mag * u;
    }
  }
  for (int64_t j = 0; j < L.rope_head_dim; ++j) {
    const float u = static_cast<float>((j * 53 + seed * 7) % 199) / 199.0f - 0.5f;
    head[static_cast<size_t>(L.nope_head_dim + j)] = 4.0f * u;
  }
  return head;
}

// A paged byte cache with a POISONED GUARD BLOCK IN FRONT of block 0. The guard
// is not decoration: with C++ truncating division a `slot` of exactly
// `-block_size` maps to (block -1, pos 0), a position the packer's own negative
// skip does NOT reject, so the only thing standing between that slot and a write
// one whole block below the page is the kernel's `slot < 0` guard. Without a
// guard region that write is silent memory corruption instead of a red test.
struct GuardedPage {
  std::vector<uint8_t> buf;
  Fp8DsMlaPageLayout page;
  int64_t num_blocks = 0;
  int64_t block_stride = 0;

  uint8_t* Base() { return buf.data() + block_stride; }
  const uint8_t* Base() const { return buf.data() + block_stride; }
  const uint8_t* Block(int64_t b) const { return Base() + b * block_stride; }
  Tensor View() {
    return Contig(Base(), DType::kI8, {num_blocks, block_stride});
  }
};

GuardedPage MakePage(int64_t num_blocks, int64_t block_size = kStorageBlock) {
  GuardedPage p;
  p.page = MakeFp8DsMlaPageLayout(TokenLayout(), block_size);
  p.num_blocks = num_blocks;
  p.block_stride = p.page.padded_block_bytes;
  p.buf.assign(static_cast<size_t>((num_blocks + 1) * p.block_stride), kPoison);
  return p;
}

// Every byte of the guard block is still poison, i.e. nothing wrote below the page.
void RequireGuardIntact(const GuardedPage& p) {
  for (int64_t i = 0; i < p.block_stride; ++i)
    REQUIRE(static_cast<int>(p.buf[static_cast<size_t>(i)]) == static_cast<int>(kPoison));
}

Tensor MakeK(std::vector<float>* storage, const std::vector<std::vector<float>>& rows) {
  storage->clear();
  for (const auto& r : rows) storage->insert(storage->end(), r.begin(), r.end());
  return Contig(storage->data(), DType::kF32,
                {static_cast<int64_t>(rows.size()), vt::kFp8DsMlaInputDim});
}

}  // namespace

TEST_CASE("ds_mla cache: the op's geometry constants ARE the packer's layout") {
  // Two descriptions of 448/64/64 exist — `vt::kFp8DsMla*` (which the op wrapper
  // validates against) and `MakeFp8DsMlaLayout` (which the kernels pack with) —
  // and nothing but this case stops them drifting. Upstream writes both from the
  // same literals at `cache_utils.py:180-190`.
  CHECK(vt::kFp8DsMlaNopeDim == 448);
  CHECK(vt::kFp8DsMlaRopeDim == 64);
  CHECK(vt::kFp8DsMlaScaleDim == 8);
  CHECK(vt::kFp8DsMlaQuantBlock == 64);
  CHECK(vt::kFp8DsMlaInputDim == 512);
  CHECK(vt::kFp8DsMlaTokenDataSize == 576);
  CHECK(vt::kFp8DsMlaTokenBytes == 584);

  const Fp8DsMlaLayout L = TokenLayout();
  CHECK(L.nope_head_dim == vt::kFp8DsMlaNopeDim);
  CHECK(L.rope_head_dim == vt::kFp8DsMlaRopeDim);
  CHECK(L.quant_block == vt::kFp8DsMlaQuantBlock);
  CHECK(L.token_stride_bytes == vt::kFp8DsMlaTokenDataSize);
  CHECK(L.scale_dim == vt::kFp8DsMlaScaleDim);
  CHECK(L.n_nope_blocks == 7);

  const Fp8DsMlaPageLayout P = MakeFp8DsMlaPageLayout(L, kStorageBlock);
  CHECK(P.real_block_bytes == kStorageBlock * vt::kFp8DsMlaTokenBytes);  // 37376
  CHECK(P.padded_block_bytes == 37440);
  CHECK(P.scale_region_offset == kStorageBlock * vt::kFp8DsMlaTokenDataSize);
}

TEST_CASE("ds_mla store: bytes land in both regions and NOTHING else moves") {
  const Fp8DsMlaLayout L = TokenLayout();
  GuardedPage p = MakePage(/*num_blocks=*/3);
  Tensor cache = p.View();

  // Slots spanning three blocks and both ends of a block, so a wrong per-token
  // stride, a wrong block stride and an off-by-one in `pos_in_block` are each in
  // range. The trailing -1 is upstream's padded token (`cache_utils.py:77-78`).
  const std::vector<int64_t> slots = {64 * 1 + 0,  64 * 1 + 1, 64 * 1 + 17,
                                      64 * 1 + 63, 64 * 2 + 5, -1};
  std::vector<std::vector<float>> rows;
  std::vector<Fp8DsMlaToken> expect;
  for (size_t i = 0; i < slots.size(); ++i) {
    rows.push_back(MakeLatent(static_cast<int64_t>(i) + 1, L));
    expect.push_back(Fp8DsMlaEncodeToken(rows.back(), L));
  }
  std::vector<float> kbuf;
  Tensor k = MakeK(&kbuf, rows);
  Tensor slot_t = Contig(const_cast<int64_t*>(slots.data()), DType::kI64,
                         {static_cast<int64_t>(slots.size())});

  Queue q = Q();
  vt::ConcatAndCacheDsMla(q, k, cache, slot_t, kStorageBlock);

  // FIRST, because every REQUIRE below aborts the case: nothing wrote below the
  // page, and nothing wrote into any block's alignment padding. The padding
  // assertion is the one a 3.5x f32 overrun trips — 2048 bytes per token against
  // the 584 the spec declares.
  RequireGuardIntact(p);
  for (int64_t b = 0; b < p.num_blocks; ++b)
    for (int64_t off = p.page.real_block_bytes; off < p.block_stride; ++off)
      REQUIRE(static_cast<int>(p.Block(b)[off]) == static_cast<int>(kPoison));

  // Block 0 was never addressed at all.
  for (int64_t off = 0; off < p.block_stride; ++off)
    REQUIRE(static_cast<int>(p.Block(0)[off]) == static_cast<int>(kPoison));

  for (size_t i = 0; i < slots.size(); ++i) {
    if (slots[i] < 0) continue;
    CAPTURE(i);
    const int64_t block = slots[i] / kStorageBlock;
    const int64_t pos = slots[i] % kStorageBlock;
    const uint8_t* blk = p.Block(block);
    const Fp8DsMlaToken& t = expect[i];
    // Data region: `pos * 576`, 448 fp8 bytes then 64 bf16 WORDS
    // (`cache_utils.py:92`, `:100-102`, `:155-159`). Storing the rope half as f32
    // would write 256 bytes here instead of 128 and fail both halves.
    const uint8_t* data = blk + pos * vt::kFp8DsMlaTokenDataSize;
    CHECK(std::memcmp(data, t.nope_fp8.data(), 448) == 0);
    CHECK(std::memcmp(data + 448, t.rope_bf16.data(), 64 * sizeof(uint16_t)) == 0);
    // Scale region: AFTER all token data (`:96-98`), 7 real bytes...
    const uint8_t* sc = blk + p.page.scale_region_offset + pos * vt::kFp8DsMlaScaleDim;
    CHECK(std::memcmp(sc, t.scale_ue8m0.data(), 7) == 0);
    // ...and the 8th byte EXPLICITLY zeroed (`:148-149`), not left poisoned.
    CHECK(static_cast<int>(sc[7]) == 0);
  }

  // The scale bytes must DISCRIMINATE. If all seven exponents were equal, the
  // memcmp above would pass with the scales interleaved per token instead of
  // gathered into the region, which is the layout this whole op exists to get
  // right.
  {
    std::vector<uint8_t> distinct(expect[0].scale_ue8m0);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    CHECK(distinct.size() >= 4u);
  }

  // Every row NOT addressed stays poisoned in BOTH regions. This is what an
  // off-by-one in `pos_in_block`, a 584-byte per-token stride and an interleaved
  // scale layout each disturb, and it is why the slots above are sparse.
  for (int64_t b = 0; b < p.num_blocks; ++b) {
    for (int64_t pos = 0; pos < kStorageBlock; ++pos) {
      const int64_t slot = b * kStorageBlock + pos;
      if (std::find(slots.begin(), slots.end(), slot) != slots.end()) continue;
      const uint8_t* blk = p.Block(b);
      for (int64_t j = 0; j < vt::kFp8DsMlaTokenDataSize; ++j)
        REQUIRE(static_cast<int>(blk[pos * vt::kFp8DsMlaTokenDataSize + j]) ==
                static_cast<int>(kPoison));
      for (int64_t j = 0; j < vt::kFp8DsMlaScaleDim; ++j)
        REQUIRE(static_cast<int>(blk[p.page.scale_region_offset +
                                     pos * vt::kFp8DsMlaScaleDim + j]) ==
                static_cast<int>(kPoison));
    }
  }
}

TEST_CASE("ds_mla store: a negative slot writes NOTHING, ANYWHERE") {
  // `if slot_idx == -1: return` (`cache_utils.py:77-78`). -64 is the load-bearing
  // case and not padding of the test: it is the exact negative multiple of the
  // block size, which truncating division maps to (block -1, pos 0) — a position
  // the packer's own negative-position skip accepts. The kernel's `slot < 0`
  // guard is the only thing that stops it, and the guard block is where that
  // write would land.
  const Fp8DsMlaLayout L = TokenLayout();
  GuardedPage p = MakePage(/*num_blocks=*/2);
  Tensor cache = p.View();

  const std::vector<int64_t> slots = {-1, -64, -128, -7};
  std::vector<std::vector<float>> rows;
  for (size_t i = 0; i < slots.size(); ++i)
    rows.push_back(MakeLatent(static_cast<int64_t>(i) + 5, L));
  std::vector<float> kbuf;
  Tensor k = MakeK(&kbuf, rows);
  Tensor slot_t = Contig(const_cast<int64_t*>(slots.data()), DType::kI64,
                         {static_cast<int64_t>(slots.size())});

  Queue q = Q();
  vt::ConcatAndCacheDsMla(q, k, cache, slot_t, kStorageBlock);

  for (size_t i = 0; i < p.buf.size(); ++i)
    REQUIRE(static_cast<int>(p.buf[i]) == static_cast<int>(kPoison));
}

TEST_CASE("ds_mla store: the token count comes from slot_mapping, not from k") {
  // Upstream `:186-188`: "slot_mapping.shape[0] can be less than k.shape[0] due
  // to padding. Always use slot_mapping.shape[0] as the token count." A kernel
  // that looped over `k` would write the trailing DP-padding rows into whatever
  // slots followed.
  const Fp8DsMlaLayout L = TokenLayout();
  GuardedPage p = MakePage(/*num_blocks=*/1);
  Tensor cache = p.View();

  std::vector<std::vector<float>> rows;
  for (int i = 0; i < 4; ++i) rows.push_back(MakeLatent(i + 3, L));
  std::vector<float> kbuf;
  Tensor k = MakeK(&kbuf, rows);  // four rows...
  const std::vector<int64_t> slots = {0, 1};  // ...two slots.
  Tensor slot_t = Contig(const_cast<int64_t*>(slots.data()), DType::kI64, {2});

  Queue q = Q();
  vt::ConcatAndCacheDsMla(q, k, cache, slot_t, kStorageBlock);

  RequireGuardIntact(p);
  for (int64_t pos = 2; pos < kStorageBlock; ++pos)
    for (int64_t j = 0; j < vt::kFp8DsMlaTokenDataSize; ++j)
      REQUIRE(static_cast<int>(p.Block(0)[pos * vt::kFp8DsMlaTokenDataSize + j]) ==
              static_cast<int>(kPoison));
}

TEST_CASE("ds_mla gather: store then read round-trips BIT-EXACTLY") {
  const Fp8DsMlaLayout L = TokenLayout();
  // Two requests whose sequences cross a block boundary, on a NON-identity block
  // table: request 0 uses physical blocks {3, 1} and request 1 {0, 2}. An
  // identity table would pass with the block-table lookup dropped entirely.
  GuardedPage p = MakePage(/*num_blocks=*/4);
  Tensor cache = p.View();

  const int64_t seq = 70;  // > 64, so each request spans two blocks
  const std::vector<std::vector<int32_t>> table = {{3, 1}, {0, 2}};
  const int64_t num_reqs = 2;

  std::vector<std::vector<float>> heads;   // [req][pos] latent
  std::vector<std::vector<float>> rows;    // flattened k rows, in store order
  std::vector<int64_t> slots;
  for (int64_t b = 0; b < num_reqs; ++b) {
    for (int64_t pos = 0; pos < seq; ++pos) {
      std::vector<float> h = MakeLatent(b * 131 + pos + 1, L);
      heads.push_back(h);
      rows.push_back(h);
      const int64_t phys = table[static_cast<size_t>(b)][static_cast<size_t>(pos / kStorageBlock)];
      slots.push_back(phys * kStorageBlock + pos % kStorageBlock);
    }
  }
  std::vector<float> kbuf;
  Tensor k = MakeK(&kbuf, rows);
  Tensor slot_t = Contig(slots.data(), DType::kI64, {static_cast<int64_t>(slots.size())});

  Queue q = Q();
  vt::ConcatAndCacheDsMla(q, k, cache, slot_t, kStorageBlock);
  RequireGuardIntact(p);

  std::vector<int32_t> flat_table;
  for (const auto& r : table) flat_table.insert(flat_table.end(), r.begin(), r.end());
  Tensor bt = Contig(flat_table.data(), DType::kI32, {num_reqs, 2});
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(seq), static_cast<int32_t>(seq)};
  Tensor sl = Contig(seq_lens.data(), DType::kI32, {num_reqs});

  const int64_t max_tokens = seq + 8;

  SUBCASE("gather_lens == nullptr gathers the WHOLE sequence") {
    // Upstream's `gather_lens_ptr is None` arm (`cache_utils.py:257-262`).
    std::vector<float> out(static_cast<size_t>(num_reqs * max_tokens * 512), -1.0f);
    Tensor o = Contig(out.data(), DType::kF32, {num_reqs, max_tokens, 512});
    vt::DequantAndGatherDsMlaArgs args;
    args.block_size = kStorageBlock;
    args.offset = 0;
    vt::DequantAndGatherDsMla(q, o, cache, sl, nullptr, bt, args);

    for (int64_t b = 0; b < num_reqs; ++b) {
      for (int64_t pos = 0; pos < seq; ++pos) {
        CAPTURE(b);
        CAPTURE(pos);
        // EQUAL, not close. Both sides are an e4m3 table lookup times an integer
        // exp2, so a tolerance here would accept a scale region read one byte off
        // that happened to land on a neighbouring exponent.
        const std::vector<float> want = Fp8DsMlaDecodeToken(
            Fp8DsMlaEncodeToken(heads[static_cast<size_t>(b * seq + pos)], L), L);
        const float* got = out.data() + (b * max_tokens + pos) * 512;
        for (int64_t d = 0; d < 512; ++d) REQUIRE(got[d] == want[static_cast<size_t>(d)]);
      }
    }
    // Columns past the gathered length are untouched.
    for (int64_t b = 0; b < num_reqs; ++b)
      for (int64_t pos = seq; pos < max_tokens; ++pos)
        REQUIRE(out[static_cast<size_t>((b * max_tokens + pos) * 512)] == -1.0f);
  }

  SUBCASE("gather_lens + offset gather the TAIL into a shifted column") {
    // `start_pos = seq_len - gather_len` (`:264`) and the output column
    // `offset + i` (`:295-296`). A kernel that ignored `start_pos` would gather
    // the sequence HEAD and still fill every column it was asked to.
    const int64_t glen = 5;
    const int64_t col_off = 3;
    std::vector<int32_t> gl = {static_cast<int32_t>(glen), static_cast<int32_t>(glen)};
    Tensor glt = Contig(gl.data(), DType::kI32, {num_reqs});
    std::vector<float> out(static_cast<size_t>(num_reqs * max_tokens * 512), -1.0f);
    Tensor o = Contig(out.data(), DType::kF32, {num_reqs, max_tokens, 512});
    vt::DequantAndGatherDsMlaArgs args;
    args.block_size = kStorageBlock;
    args.offset = col_off;
    vt::DequantAndGatherDsMla(q, o, cache, sl, &glt, bt, args);

    for (int64_t b = 0; b < num_reqs; ++b) {
      for (int64_t i = 0; i < glen; ++i) {
        const int64_t pos = seq - glen + i;
        CAPTURE(b);
        CAPTURE(pos);
        const std::vector<float> want = Fp8DsMlaDecodeToken(
            Fp8DsMlaEncodeToken(heads[static_cast<size_t>(b * seq + pos)], L), L);
        const float* got = out.data() + (b * max_tokens + col_off + i) * 512;
        for (int64_t d = 0; d < 512; ++d) REQUIRE(got[d] == want[static_cast<size_t>(d)]);
      }
      // The columns BEFORE `offset` are untouched: `offset` is a write position,
      // not a source position.
      for (int64_t c = 0; c < col_off; ++c)
        REQUIRE(out[static_cast<size_t>((b * max_tokens + c) * 512)] == -1.0f);
    }
  }

  SUBCASE("a bf16 out is upstream's own store, bit for bit") {
    // Upstream's `out` is bf16 (`:328`, `:337`). The f32 arm above is the
    // annotated widening — the same value BEFORE this rounding.
    std::vector<uint16_t> out(static_cast<size_t>(num_reqs * max_tokens * 512), 0);
    Tensor o = Contig(out.data(), DType::kBF16, {num_reqs, max_tokens, 512});
    vt::DequantAndGatherDsMlaArgs args;
    args.block_size = kStorageBlock;
    vt::DequantAndGatherDsMla(q, o, cache, sl, nullptr, bt, args);
    for (int64_t b = 0; b < num_reqs; ++b) {
      for (int64_t pos = 0; pos < seq; pos += 17) {
        const std::vector<float> want = Fp8DsMlaDecodeToken(
            Fp8DsMlaEncodeToken(heads[static_cast<size_t>(b * seq + pos)], L), L);
        const uint16_t* got = out.data() + (b * max_tokens + pos) * 512;
        for (int64_t d = 0; d < 512; ++d)
          REQUIRE(got[d] == vt::F32ToBF16(want[static_cast<size_t>(d)]));
      }
    }
  }
}

TEST_CASE("ds_mla gather: the 8th scale byte is written and NEVER read") {
  // The asymmetry is upstream's: the store writes `scale_dim == 8`
  // (`cache_utils.py:148-149`), the gather consumes `n_quant_blocks == 7`
  // (`:385`). Corrupting the pad byte after the store must change NOTHING the
  // read produces. A read that consumed 8 blocks would fail here, and would also
  // be reading past the seven exponents the encoder actually produced.
  const Fp8DsMlaLayout L = TokenLayout();
  GuardedPage p = MakePage(/*num_blocks=*/1);
  Tensor cache = p.View();

  std::vector<std::vector<float>> rows;
  std::vector<int64_t> slots;
  for (int64_t pos = 0; pos < 4; ++pos) {
    rows.push_back(MakeLatent(pos + 21, L));
    slots.push_back(pos);
  }
  std::vector<float> kbuf;
  Tensor k = MakeK(&kbuf, rows);
  Tensor slot_t = Contig(slots.data(), DType::kI64, {4});
  Queue q = Q();
  vt::ConcatAndCacheDsMla(q, k, cache, slot_t, kStorageBlock);

  std::vector<int32_t> bt_v = {0};
  Tensor bt = Contig(bt_v.data(), DType::kI32, {1, 1});
  std::vector<int32_t> sl_v = {4};
  Tensor sl = Contig(sl_v.data(), DType::kI32, {1});
  vt::DequantAndGatherDsMlaArgs args;
  args.block_size = kStorageBlock;

  std::vector<float> before(static_cast<size_t>(4 * 512), 0.0f);
  Tensor ob = Contig(before.data(), DType::kF32, {1, 4, 512});
  vt::DequantAndGatherDsMla(q, ob, cache, sl, nullptr, bt, args);

  // Scribble on every token's pad byte only.
  for (int64_t pos = 0; pos < 4; ++pos) {
    uint8_t* sc = const_cast<uint8_t*>(p.Block(0)) + p.page.scale_region_offset +
                  pos * vt::kFp8DsMlaScaleDim;
    REQUIRE(static_cast<int>(sc[7]) == 0);
    sc[7] = 0xFF;
  }

  std::vector<float> after(static_cast<size_t>(4 * 512), 0.0f);
  Tensor oa = Contig(after.data(), DType::kF32, {1, 4, 512});
  vt::DequantAndGatherDsMla(q, oa, cache, sl, nullptr, bt, args);
  for (size_t i = 0; i < before.size(); ++i) REQUIRE(before[i] == after[i]);
}

TEST_CASE("ds_mla: the byte page is REFUSED anything it cannot be") {
  const Fp8DsMlaLayout L = TokenLayout();
  GuardedPage p = MakePage(/*num_blocks=*/2);
  Queue q = Q();
  std::vector<std::vector<float>> rows = {MakeLatent(1, L)};
  std::vector<float> kbuf;
  Tensor k = MakeK(&kbuf, rows);
  std::vector<int64_t> slots = {0};
  Tensor slot_t = Contig(slots.data(), DType::kI64, {1});

  {  // The contract this op exists for: a FLOAT cache is the 3.5x overrun.
    Tensor bad = p.View();
    bad.dtype = DType::kF32;
    CHECK_THROWS_AS(vt::ConcatAndCacheDsMla(q, k, bad, slot_t, kStorageBlock),
                    std::runtime_error);
  }
  {  // A rank-3 page cannot describe a region-split block.
    Tensor bad = Contig(p.Base(), DType::kI8, {2, kStorageBlock, 584});
    CHECK_THROWS_AS(vt::ConcatAndCacheDsMla(q, k, bad, slot_t, kStorageBlock),
                    std::runtime_error);
  }
  {  // The latent is 512 wide (`cache_utils.py:167-169`).
    std::vector<float> narrow(576, 0.0f);
    Tensor bad = Contig(narrow.data(), DType::kF32, {1, 576});
    Tensor cache = p.View();
    CHECK_THROWS_AS(vt::ConcatAndCacheDsMla(q, bad, cache, slot_t, kStorageBlock),
                    std::runtime_error);
  }
  {  // A page row too short for block_size * 584 would put the scale region past
     // its end.
    Tensor cache = p.View();
    CHECK_THROWS_AS(vt::ConcatAndCacheDsMla(q, k, cache, slot_t, kStorageBlock + 1),
                    std::runtime_error);
  }
  {  // vt::ConcatAndCacheMla keeps its rank-3, float-only contract: it must still
     // refuse this page, because that refusal is what stops an f32 write landing
     // in a byte page while slice 4 is unwritten.
    Tensor cache = p.View();
    Tensor kvc = Contig(kbuf.data(), DType::kF32, {1, 448});
    Tensor kpe = Contig(kbuf.data(), DType::kF32, {1, 64});
    CHECK_THROWS_AS(vt::ConcatAndCacheMla(q, kvc, kpe, cache, slot_t), std::runtime_error);
  }
  {  // The gather's out must be the 512-wide latent scratch.
    std::vector<float> out(static_cast<size_t>(4 * 576), 0.0f);
    Tensor bad = Contig(out.data(), DType::kF32, {1, 4, 576});
    Tensor cache = p.View();
    std::vector<int32_t> bt_v = {0};
    Tensor bt = Contig(bt_v.data(), DType::kI32, {1, 1});
    std::vector<int32_t> sl_v = {4};
    Tensor sl = Contig(sl_v.data(), DType::kI32, {1});
    vt::DequantAndGatherDsMlaArgs args;
    args.block_size = kStorageBlock;
    CHECK_THROWS_AS(vt::DequantAndGatherDsMla(q, bad, cache, sl, nullptr, bt, args),
                    std::runtime_error);
  }
  {  // A sequence longer than its block table is a caller bug, not a wrap.
    std::vector<float> out(static_cast<size_t>(200 * 512), 0.0f);
    Tensor o = Contig(out.data(), DType::kF32, {1, 200, 512});
    Tensor cache = p.View();
    std::vector<int32_t> bt_v = {0};
    Tensor bt = Contig(bt_v.data(), DType::kI32, {1, 1});
    std::vector<int32_t> sl_v = {130};
    Tensor sl = Contig(sl_v.data(), DType::kI32, {1});
    vt::DequantAndGatherDsMlaArgs args;
    args.block_size = kStorageBlock;
    CHECK_THROWS_AS(vt::DequantAndGatherDsMla(q, o, cache, sl, nullptr, bt, args),
                    std::runtime_error);
  }
}
