// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Paged attention op unit tests. Semantics ported from
// vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl.forward @ e24d1b24
// (causal GQA softmax over the paged K/V; scale = self.scale). The cache is the
// NHD FlashAttention layout — the two dim-1 unbind slices of get_kv_cache_shape's
// (num_blocks, 2, block_size, num_kv_heads, head_size).
//
// Golden strategy (M1.6 Task-3 review): COMPOSE the reference math, do NOT dump
// backend cache bytes.
//   1. ANCHOR (strongest): on a single contiguous sequence, paged_attention MUST
//      agree with M0.9's dense vt::Attention. The K/V are written into a paged
//      cache via vt::ReshapeAndCache, so the whole pipeline is exercised. No new
//      oracle needed.
//   2. Batched varlen (2 reqs: prefill len 4 + decode at seq_len 6) validated
//      against a composed per-token causal GQA softmax reference (host math).
//   3. GQA head mapping, causal masking, block-spanning (seq > block_size).
//   4. CUDA-vs-CPU parity (guarded by HasCuda; dgx-pending on CPU-only boxes).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::AttentionArgs;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::PagedAttentionArgs;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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

// Rank-4 strided view into a single (num_blocks, 2, block_size, H, D) allocation
// at element offset off_elems — the unbind(1) slice K/V really are (block stride
// = 2*bs*H*D, NOT bs*H*D). Mirrors the reshape_and_cache strided-slice tests.
template <typename T>
Tensor StridedCache(T* data, DType dt, Device dev, int64_t off_elems,
                    const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
  Tensor t;
  t.data = data + off_elems;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride[static_cast<size_t>(i)];
  }
  return t;
}

Tensor F32(std::vector<float>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kF32, Cpu(), shape);
}
Tensor I32(std::vector<int32_t>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kI32, Cpu(), shape);
}

std::vector<float> RandF32(size_t n, uint32_t seed) {
  // Deterministic LCG in [-2,2); matches the other vt op tests.
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

// Host composed reference: per-token causal GQA softmax over K/V gathered from a
// contiguous NHD cache (block b, offset o at flat index ((b*bs+o)*H+g)*D+e). This
// is the M0.9-style reference, independent of the op's stride arithmetic.
std::vector<float> ComposedPagedRef(const std::vector<float>& q, const std::vector<float>& kc,
                                    const std::vector<float>& vc,
                                    const std::vector<int32_t>& block_table, int64_t max_blocks,
                                    const std::vector<int32_t>& seq_lens,
                                    const std::vector<int32_t>& qsl, int64_t hq, int64_t hk,
                                    int64_t d, int64_t block_size, float scale, bool causal) {
  const int64_t num_reqs = static_cast<int64_t>(seq_lens.size());
  const int64_t num_tokens = qsl.back();
  const int64_t qpk = hq / hk;
  std::vector<float> out(static_cast<size_t>(num_tokens * hq * d), 0.0f);
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q0 = qsl[static_cast<size_t>(r)], q1 = qsl[static_cast<size_t>(r + 1)];
    const int64_t qlen = q1 - q0;
    const int64_t seqlen = seq_lens[static_cast<size_t>(r)];
    const int64_t context = seqlen - qlen;
    for (int64_t local = 0; local < qlen; ++local) {
      const int64_t t = q0 + local;
      const int64_t p = context + local;
      const int64_t jmax = causal ? p : seqlen - 1;
      for (int64_t h = 0; h < hq; ++h) {
        const int64_t g = h / qpk;
        const int64_t qoff = (t * hq + h) * d;
        std::vector<float> sc(static_cast<size_t>(jmax + 1));
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j <= jmax; ++j) {
          const int64_t blk = block_table[static_cast<size_t>(r * max_blocks + j / block_size)];
          const int64_t off = j % block_size;
          const int64_t kbase = ((blk * block_size + off) * hk + g) * d;
          float dot = 0.0f;
          for (int64_t e = 0; e < d; ++e)
            dot += q[static_cast<size_t>(qoff + e)] * kc[static_cast<size_t>(kbase + e)];
          dot *= scale;
          sc[static_cast<size_t>(j)] = dot;
          if (dot > m) m = dot;
        }
        float denom = 0.0f;
        for (int64_t j = 0; j <= jmax; ++j) {
          const float e = std::exp(sc[static_cast<size_t>(j)] - m);
          sc[static_cast<size_t>(j)] = e;
          denom += e;
        }
        const float inv = 1.0f / denom;
        for (int64_t e = 0; e < d; ++e) {
          float a = 0.0f;
          for (int64_t j = 0; j <= jmax; ++j) {
            const int64_t blk = block_table[static_cast<size_t>(r * max_blocks + j / block_size)];
            const int64_t off = j % block_size;
            const int64_t vbase = ((blk * block_size + off) * hk + g) * d;
            a += sc[static_cast<size_t>(j)] * inv * vc[static_cast<size_t>(vbase + e)];
          }
          out[static_cast<size_t>(qoff + e)] = a;
        }
      }
    }
  }
  return out;
}

}  // namespace

// ===========================================================================
// ANCHOR: single contiguous sequence — paged_attention == dense vt::Attention.
// Build one sequence, write its K/V into a paged cache via ReshapeAndCache, run
// both ops, assert equal. The strongest correctness gate (no new oracle).
// ===========================================================================
TEST_CASE("paged_attention anchor: single sequence agrees with dense vt::Attention") {
  const int64_t T = 7, Hq = 4, Hk = 2, D = 8, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  auto q = RandF32(static_cast<size_t>(T * Hq * D), 11);
  auto k = RandF32(static_cast<size_t>(T * Hk * D), 22);
  auto v = RandF32(static_cast<size_t>(T * Hk * D), 33);
  Queue qq = Q();

  // Dense reference (M0.9): contiguous single-sequence causal GQA attention.
  std::vector<float> dense(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tq = F32(q, {T, Hq, D}), tk = F32(k, {T, Hk, D}), tv = F32(v, {T, Hk, D});
  Tensor td = F32(dense, {T, Hq, D});
  vt::Attention(qq, td, tq, tk, tv, AttentionArgs{scale, /*causal=*/true});

  // Paged path: allocate a cache big enough for T tokens, write K/V at slots
  // 0..T-1 via ReshapeAndCache, then run PagedAttention over the one request.
  const int64_t num_blocks = (T + block_size - 1) / block_size + 1;  // slack block
  const int64_t page = Hk * D;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  std::vector<int64_t> slots(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) slots[static_cast<size_t>(i)] = i;  // contiguous, block 0..
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {T});
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  const int64_t max_blocks = num_blocks;
  std::vector<int32_t> block_table(static_cast<size_t>(max_blocks));
  for (int64_t b = 0; b < max_blocks; ++b) block_table[static_cast<size_t>(b)] = b;  // identity
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(T)};
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};

  std::vector<float> paged(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tp = F32(paged, {T, Hq, D});
  Tensor tbt = I32(block_table, {1, max_blocks});
  Tensor tsl = I32(seq_lens, {1});
  Tensor tqsl = I32(qsl, {2});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  for (size_t i = 0; i < dense.size(); ++i)
    CHECK(paged[i] == doctest::Approx(dense[i]).epsilon(1e-5));
}

// ===========================================================================
// ANCHOR on the REAL strided K/V unbind slices: same agreement must hold when
// k_cache/v_cache are the two dim-1 slices of one (nb,2,bs,H,D) allocation
// (block stride 2*bs*H*D). This is the layout the runner actually hands us.
// ===========================================================================
TEST_CASE("paged_attention anchor on strided unbind-slice cache == dense") {
  const int64_t T = 5, Hq = 2, Hk = 1, D = 4, block_size = 2;
  const float scale = 0.35f;
  auto q = RandF32(static_cast<size_t>(T * Hq * D), 101);
  auto k = RandF32(static_cast<size_t>(T * Hk * D), 202);
  auto v = RandF32(static_cast<size_t>(T * Hk * D), 303);
  Queue qq = Q();

  std::vector<float> dense(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tq = F32(q, {T, Hq, D}), tk = F32(k, {T, Hk, D}), tv = F32(v, {T, Hk, D});
  Tensor td = F32(dense, {T, Hq, D});
  vt::Attention(qq, td, tq, tk, tv, AttentionArgs{scale, true});

  const int64_t nb = (T + block_size - 1) / block_size + 1;
  const int64_t page = Hk * D;
  const int64_t within_block = block_size * page;
  const int64_t blk_stride = 2 * within_block;
  std::vector<float> buf(static_cast<size_t>(nb * 2 * block_size * page), 0.0f);
  const std::vector<int64_t> cshape = {nb, block_size, Hk, D};
  const std::vector<int64_t> cstride = {blk_stride, page, D, 1};
  Tensor tkc = StridedCache(buf.data(), DType::kF32, Cpu(), 0, cshape, cstride);
  Tensor tvc = StridedCache(buf.data(), DType::kF32, Cpu(), within_block, cshape, cstride);
  std::vector<int64_t> slots(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) slots[static_cast<size_t>(i)] = i;
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {T});
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  std::vector<int32_t> block_table(static_cast<size_t>(nb));
  for (int64_t b = 0; b < nb; ++b) block_table[static_cast<size_t>(b)] = b;
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(T)};
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};
  std::vector<float> paged(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tp = F32(paged, {T, Hq, D});
  Tensor tbt = I32(block_table, {1, nb});
  Tensor tsl = I32(seq_lens, {1});
  Tensor tqsl = I32(qsl, {2});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  for (size_t i = 0; i < dense.size(); ++i)
    CHECK(paged[i] == doctest::Approx(dense[i]).epsilon(1e-5));
}

// ===========================================================================
// Batched varlen (M1.5 oracle shape): 2 reqs — req0 prefill len 4 (seq_len 4),
// req1 decode at seq_len 6 (query_len 1, context 5). query_start_loc=[0,4,5],
// seq_lens=[4,6]. Non-identity block tables + block-spanning (req1 uses 2 blocks
// with block_size 4). Validated against the composed host reference.
// ===========================================================================
TEST_CASE("paged_attention batched varlen (prefill + decode) vs composed reference") {
  const int64_t Hq = 4, Hk = 2, D = 8, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const int64_t num_tokens = 5;  // 4 (req0) + 1 (req1)
  std::vector<int32_t> qsl = {0, 4, 5};
  std::vector<int32_t> seq_lens = {4, 6};
  Queue qq = Q();

  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 44);

  // A paged cache with enough blocks; give the two requests DISJOINT,
  // non-identity blocks. req0 (seq 4) → 1 block; req1 (seq 6) → 2 blocks.
  const int64_t num_blocks = 6, page = Hk * D, max_blocks = 2;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  // block_table[req, blk_idx]: req0 uses block 3; req1 uses blocks 1 and 4.
  std::vector<int32_t> block_table = {3, 0,  /* req0: only block 3 needed */
                                      1, 4}; /* req1: pos 0..3 → blk1, pos 4..5 → blk4 */

  // Fill the cache directly (host) at the exact (block,offset,head) the tables
  // address, so the composed reference reads the same values the op will.
  auto set_cache = [&](std::vector<float>& cache, int64_t blk, int64_t off, uint32_t seed) {
    auto vals = RandF32(static_cast<size_t>(page), seed);
    const int64_t base = ((blk * block_size + off) * Hk) * D;
    for (int64_t i = 0; i < page; ++i) cache[static_cast<size_t>(base + i)] = vals[static_cast<size_t>(i)];
  };
  // req0 positions 0..3 → block 3, offsets 0..3.
  for (int64_t j = 0; j < 4; ++j) {
    set_cache(kc, 3, j, 500u + static_cast<uint32_t>(j));
    set_cache(vc, 3, j, 600u + static_cast<uint32_t>(j));
  }
  // req1 positions 0..3 → block 1 offsets 0..3; positions 4..5 → block 4 offsets 0..1.
  for (int64_t j = 0; j < 4; ++j) {
    set_cache(kc, 1, j, 700u + static_cast<uint32_t>(j));
    set_cache(vc, 1, j, 800u + static_cast<uint32_t>(j));
  }
  for (int64_t j = 0; j < 2; ++j) {
    set_cache(kc, 4, j, 900u + static_cast<uint32_t>(j));
    set_cache(vc, 4, j, 950u + static_cast<uint32_t>(j));
  }

  std::vector<float> ref = ComposedPagedRef(q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq,
                                            Hk, D, block_size, scale, /*causal=*/true);

  Tensor tq = F32(q, {num_tokens, Hq, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tbt = I32(block_table, {2, max_blocks});
  Tensor tsl = I32(seq_lens, {2});
  Tensor tqsl = I32(qsl, {3});
  std::vector<float> got(static_cast<size_t>(num_tokens * Hq * D), 0.0f);
  Tensor tp = F32(got, {num_tokens, Hq, D});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  for (size_t i = 0; i < ref.size(); ++i)
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
}

// ===========================================================================
// Causal masking: a decode query (context 2, seq_len 3) must ignore nothing up
// to its own position, but a prefill query at position 0 must see only key 0
// even when a huge future V sits at position 1.
// ===========================================================================
TEST_CASE("paged_attention causal mask hides future keys from an early query") {
  const int64_t Hq = 1, Hk = 1, D = 2, block_size = 4;
  const int64_t T = 2;  // one request, prefill of 2 tokens
  const float scale = 1.0f;
  std::vector<float> q = {5, 0, 0, 5};  // token0 q=[5,0], token1 q=[0,5]
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {1, 1, 1000, 1000};  // huge future v at pos 1
  Queue qq = Q();

  const int64_t num_blocks = 1, page = Hk * D;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  Tensor tk = F32(k, {T, Hk, D}), tv = F32(v, {T, Hk, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  std::vector<int64_t> slots = {0, 1};
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {2});
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  std::vector<int32_t> block_table = {0};
  std::vector<int32_t> seq_lens = {2};
  std::vector<int32_t> qsl = {0, 2};
  std::vector<float> out(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tq = F32(q, {T, Hq, D}), tp = F32(out, {T, Hq, D});
  Tensor tbt = I32(block_table, {1, 1}), tsl = I32(seq_lens, {1}), tqsl = I32(qsl, {2});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  CHECK(out[0] == doctest::Approx(1.0f));  // token 0 sees only key 0
  CHECK(out[1] == doctest::Approx(1.0f));
}

// ===========================================================================
// GQA head mapping: q-head h reads kv-head h/(Hq/Hk). Single key so softmax=1 →
// each q-head's output is exactly its mapped kv-head's V.
// ===========================================================================
TEST_CASE("paged_attention GQA maps q-head h to kv-head h/(Hq/Hk)") {
  const int64_t Hq = 4, Hk = 2, D = 1, block_size = 2;
  const float scale = 1.0f;
  std::vector<float> q = {1, 2, 3, 4};  // [T=1, Hq=4, D=1]; values irrelevant (1 key)
  std::vector<float> k = {1, 1};        // [T=1, Hk=2, D=1]
  std::vector<float> v = {7, 9};        // kv0 → 7, kv1 → 9
  Queue qq = Q();

  const int64_t num_blocks = 1, page = Hk * D;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  Tensor tk = F32(k, {1, Hk, D}), tv = F32(v, {1, Hk, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  std::vector<int64_t> slots = {0};
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {1});
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  std::vector<int32_t> block_table = {0};
  std::vector<int32_t> seq_lens = {1};
  std::vector<int32_t> qsl = {0, 1};
  std::vector<float> out(static_cast<size_t>(Hq * D), 0.0f);
  Tensor tq = F32(q, {1, Hq, D}), tp = F32(out, {1, Hq, D});
  Tensor tbt = I32(block_table, {1, 1}), tsl = I32(seq_lens, {1}), tqsl = I32(qsl, {2});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  CHECK(out[0] == doctest::Approx(7.0f));  // head 0 → kv 0
  CHECK(out[1] == doctest::Approx(7.0f));  // head 1 → kv 0
  CHECK(out[2] == doctest::Approx(9.0f));  // head 2 → kv 1
  CHECK(out[3] == doctest::Approx(9.0f));  // head 3 → kv 1
}

// ===========================================================================
// Block-spanning: a single decode query at seq_len 5 with block_size 2 reads
// keys across 3 blocks (positions 0-1 blk A, 2-3 blk B, 4 blk C) via the block
// table. Validated against the composed reference. Confirms j/block_size and
// j%block_size drive the block hop correctly.
// ===========================================================================
TEST_CASE("paged_attention block-spanning decode reads across multiple blocks") {
  const int64_t Hq = 2, Hk = 1, D = 4, block_size = 2;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const int64_t seqlen = 5;
  auto q = RandF32(static_cast<size_t>(Hq * D), 55);  // 1 decode token

  const int64_t num_blocks = 4, page = Hk * D, max_blocks = 3;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  // Non-identity, non-contiguous blocks: pos 0-1 → blk 2, pos 2-3 → blk 0, pos 4 → blk 3.
  std::vector<int32_t> block_table = {2, 0, 3};
  auto set_cache = [&](std::vector<float>& cache, int64_t blk, int64_t off, uint32_t seed) {
    auto vals = RandF32(static_cast<size_t>(page), seed);
    const int64_t base = ((blk * block_size + off) * Hk) * D;
    for (int64_t i = 0; i < page; ++i) cache[static_cast<size_t>(base + i)] = vals[static_cast<size_t>(i)];
  };
  for (int64_t j = 0; j < seqlen; ++j) {
    const int64_t blk = block_table[static_cast<size_t>(j / block_size)];
    const int64_t off = j % block_size;
    set_cache(kc, blk, off, 1000u + static_cast<uint32_t>(j));
    set_cache(vc, blk, off, 2000u + static_cast<uint32_t>(j));
  }

  std::vector<int32_t> seq_lens = {static_cast<int32_t>(seqlen)};
  std::vector<int32_t> qsl = {0, 1};  // one decode token
  std::vector<float> ref = ComposedPagedRef(q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq,
                                            Hk, D, block_size, scale, true);

  Queue qq = Q();
  Tensor tq = F32(q, {1, Hq, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tbt = I32(block_table, {1, max_blocks});
  Tensor tsl = I32(seq_lens, {1});
  Tensor tqsl = I32(qsl, {2});
  std::vector<float> got(static_cast<size_t>(Hq * D), 0.0f);
  Tensor tp = F32(got, {1, Hq, D});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
}

TEST_CASE("paged_attention validates shapes/args") {
  const int64_t block_size = 2, Hk = 1, D = 2;
  std::vector<float> kc(static_cast<size_t>(1 * block_size * Hk * D), 0.0f);
  std::vector<float> q(static_cast<size_t>(1 * 2 * D), 0.0f);
  std::vector<float> out(static_cast<size_t>(1 * 2 * D), 0.0f);
  std::vector<int32_t> bt = {0}, sl = {1}, qsl = {0, 1};
  Queue qq = Q();
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {1, block_size, Hk, D});
  Tensor tvc = Contig(kc.data(), DType::kF32, Cpu(), {1, block_size, Hk, D});
  Tensor tbt = I32(bt, {1, 1}), tsl = I32(sl, {1}), tqsl = I32(qsl, {2});

  // Hq not a multiple of Hk (3 q-heads, 1 kv-head is fine; use Hk=2 mismatch).
  {
    std::vector<float> kc2(static_cast<size_t>(1 * block_size * 2 * D), 0.0f);
    Tensor tkc2 = Contig(kc2.data(), DType::kF32, Cpu(), {1, block_size, 2, D});
    Tensor tvc2 = Contig(kc2.data(), DType::kF32, Cpu(), {1, block_size, 2, D});
    std::vector<float> q3(static_cast<size_t>(3 * D), 0.0f), o3(static_cast<size_t>(3 * D), 0.0f);
    Tensor tq3 = F32(q3, {1, 3, D}), to3 = F32(o3, {1, 3, D});
    CHECK_THROWS_AS(
        vt::PagedAttention(qq, to3, tq3, tkc2, tvc2, tbt, tsl, tqsl, PagedAttentionArgs{1.0f, true}),
        std::runtime_error);
  }
  // scale must be > 0.
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    CHECK_THROWS_AS(
        vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{0.0f, true}),
        std::runtime_error);
  }
  // query_start_loc wrong length (must be num_reqs+1).
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    std::vector<int32_t> bad = {0};
    Tensor tqsl_bad = I32(bad, {1});
    CHECK_THROWS_AS(vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl_bad,
                                       PagedAttentionArgs{1.0f, true}),
                    std::runtime_error);
  }
  // seq_lens must be i32.
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    std::vector<int64_t> sl64 = {1};
    Tensor tsl_bad = Contig(sl64.data(), DType::kI64, Cpu(), {1});
    CHECK_THROWS_AS(
        vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl_bad, tqsl, PagedAttentionArgs{1.0f, true}),
        std::runtime_error);
  }
}

// ===========================================================================
// CUDA parity: the CUDA paged kernel must match the CPU reference on random
// inputs (batched varlen + block-spanning). Guarded by HasCuda; dgx-pending on
// CPU-only boxes.
// ===========================================================================
namespace {

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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
    t_ = MakeT(p_, dt, Gpu(), shape);
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

}  // namespace

TEST_CASE("paged_attention CUDA matches CPU (batched varlen, GQA, block-spanning)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping paged_attention CUDA parity (dgx-pending)");
    return;
  }
  const int64_t Hq = 16, Hk = 2, D = 32, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  // 3 reqs: prefill len 6 (seq 6), decode seq 9, prefill len 2 (seq 11 chunk).
  std::vector<int32_t> qsl = {0, 6, 7, 9};
  std::vector<int32_t> seq_lens = {6, 9, 11};
  const int64_t num_tokens = 9;
  const int64_t num_reqs = 3;
  const int64_t num_blocks = 16, page = Hk * D, max_blocks = 3;
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 4242);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 71);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 91);
  // block tables (num_reqs x max_blocks), disjoint-ish blocks.
  std::vector<int32_t> block_table = {5, 0, 0,  /* req0 seq6 → 2 blocks: 5,0 */
                                      2, 7, 0,  /* req1 seq9 → 3 blocks: 2,7,0 */
                                      9, 3, 6}; /* req2 seq11 → 3 blocks: 9,3,6 */

  std::vector<float> ref = ComposedPagedRef(q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq,
                                            Hk, D, block_size, scale, true);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {num_tokens, Hq, D}, q.data());
  DeviceTensor dkc(gpu, g.q, DType::kF32, {num_blocks, block_size, Hk, D}, kc.data());
  DeviceTensor dvc(gpu, g.q, DType::kF32, {num_blocks, block_size, Hk, D}, vc.data());
  DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks}, block_table.data());
  DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
  DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {num_tokens, Hq, D});
  vt::PagedAttention(g.q, dout.tensor(), dq.tensor(), dkc.tensor(), dvc.tensor(), dbt.tensor(),
                     dsl.tensor(), dqsl.tensor(), PagedAttentionArgs{scale, true});
  std::vector<float> got(static_cast<size_t>(num_tokens * Hq * D), 0.0f);
  dout.Download(g.q, got.data());

  for (size_t i = 0; i < ref.size(); ++i)
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-4));
}
