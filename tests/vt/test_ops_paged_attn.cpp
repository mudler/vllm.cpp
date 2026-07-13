// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Paged attention op unit tests. Semantics ported from
// vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl.forward @ e24d1b24
// and tests/v1/attention/test_attention_backends.py:745-867 (causal decoder and
// symmetric encoder sliding-window masks over paged K/V; scale = self.scale).
// Kernel vectors additionally carry tests/kernels/attention/test_flash_attn.py:
// 95-217 (paged varlen, GQA and window boundaries).
// The cache is the
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

#ifdef VLLM_CPP_FLASH_ATTN
namespace vt::cuda::testing {
int Fa2DecodeNumSplitsForTesting(int batch_nheads_mblocks, int num_sms,
                                 int num_n_blocks, int max_splits);
void ResetFa2DecodeDebugCounters();
void DisableFa2DecodeDebugCounters();
uint64_t Fa2DecodeLaunchesForTesting();
uint64_t Fa2DecodeSplitLaunchesForTesting();
uint64_t Fa2DecodeNoSplitLaunchesForTesting();
uint64_t Fa2DecodeScratchAllocationsForTesting();
uint64_t Fa2DecodeScratchReusesForTesting();
size_t Fa2DecodeScratchShapeCountForTesting(int device, void* stream);
}  // namespace vt::cuda::testing
#endif

using vt::AttentionArgs;
using vt::AttentionWindow;
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

// Host composed reference: per-token causal/non-causal GQA softmax over K/V gathered from a
// contiguous NHD cache (block b, offset o at flat index ((b*bs+o)*H+g)*D+e). This
// is the M0.9-style reference, independent of the op's stride arithmetic.
std::vector<float> ComposedPagedRef(const std::vector<float>& q, const std::vector<float>& kc,
                                    const std::vector<float>& vc,
                                    const std::vector<int32_t>& block_table, int64_t max_blocks,
                                    const std::vector<int32_t>& seq_lens,
                                    const std::vector<int32_t>& qsl, int64_t hq, int64_t hk,
                                    int64_t d, int64_t block_size, float scale, bool causal,
                                    std::optional<AttentionWindow> window = std::nullopt) {
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
      const int64_t jmin =
          window.has_value() ? std::max<int64_t>(0, p - window->left) : 0;
      int64_t jmax = causal ? p : seqlen - 1;
      if (window.has_value()) jmax = std::min(jmax, p + window->right);
      jmax = std::min(jmax, seqlen - 1);
      for (int64_t h = 0; h < hq; ++h) {
        const int64_t g = h / qpk;
        const int64_t qoff = (t * hq + h) * d;
        std::vector<float> sc(static_cast<size_t>(jmax - jmin + 1));
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t j = jmin; j <= jmax; ++j) {
          const int64_t blk = block_table[static_cast<size_t>(r * max_blocks + j / block_size)];
          const int64_t off = j % block_size;
          const int64_t kbase = ((blk * block_size + off) * hk + g) * d;
          float dot = 0.0f;
          for (int64_t e = 0; e < d; ++e)
            dot += q[static_cast<size_t>(qoff + e)] * kc[static_cast<size_t>(kbase + e)];
          dot *= scale;
          sc[static_cast<size_t>(j - jmin)] = dot;
          if (dot > m) m = dot;
        }
        float denom = 0.0f;
        for (int64_t j = jmin; j <= jmax; ++j) {
          const float e = std::exp(sc[static_cast<size_t>(j - jmin)] - m);
          sc[static_cast<size_t>(j - jmin)] = e;
          denom += e;
        }
        const float inv = 1.0f / denom;
        for (int64_t e = 0; e < d; ++e) {
          float a = 0.0f;
          for (int64_t j = jmin; j <= jmax; ++j) {
            const int64_t blk = block_table[static_cast<size_t>(r * max_blocks + j / block_size)];
            const int64_t off = j % block_size;
            const int64_t vbase = ((blk * block_size + off) * hk + g) * d;
            a += sc[static_cast<size_t>(j - jmin)] * inv * vc[static_cast<size_t>(vbase + e)];
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

// Port of tests/v1/attention/test_attention_backends.py::
// test_sliding_window_backend_correctness. One ragged batch combines prefill,
// decode and context-bearing chunked prefill; W straddles page boundaries.
TEST_CASE("paged_attention sliding-window decoder matches bottom-right causal mask") {
  const int64_t Hq = 4, Hk = 2, D = 8, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  std::vector<int32_t> qsl = {0, 5, 6, 9};
  std::vector<int32_t> seq_lens = {5, 9, 11};
  const int64_t num_tokens = qsl.back();
  const int64_t num_blocks = 12, max_blocks = 3;
  std::vector<int32_t> block_table = {
      2, 3, 0,   // five-token prefill
      4, 7, 8,   // decode at absolute position 8
      1, 9, 10,  // three-token chunk at positions 8..10
  };
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 3101);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), 3102);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), 3103);

  Queue qq = Q();
  Tensor tq = F32(q, {num_tokens, Hq, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(),
                      {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(),
                      {num_blocks, block_size, Hk, D});
  Tensor tbt = I32(block_table, {3, max_blocks});
  Tensor tsl = I32(seq_lens, {3});
  Tensor tqsl = I32(qsl, {4});

  for (const int32_t width : {1, 3, 4, 5}) {
    CAPTURE(width);
    const AttentionWindow window{width - 1, 0};
    const std::vector<float> ref = ComposedPagedRef(
        q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
        block_size, scale, /*causal=*/true, window);
    std::vector<float> got(ref.size(), 0.0f);
    Tensor tout = F32(got, {num_tokens, Hq, D});
    PagedAttentionArgs args{scale, true};
    args.window_size = window;
    vt::PagedAttention(qq, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);
    for (size_t i = 0; i < ref.size(); ++i) {
      CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
    }
  }
}

// Port of test_sliding_window_encoder_backend_correctness. Non-causal encoder
// windows are symmetric, and the absolute query positions are bottom-right
// aligned when Q is shorter than K (context = seqlen-query_len).
TEST_CASE("paged_attention sliding-window encoder is symmetric and bottom-right aligned") {
  const int64_t Hq = 2, Hk = 1, D = 4, block_size = 4;
  const int64_t query_len = 5, seqlen = 7;
  const int64_t num_blocks = 4, max_blocks = 2;
  const float scale = 0.5f;
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(query_len)};
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(seqlen)};
  std::vector<int32_t> block_table = {3, 1};
  auto q = RandF32(static_cast<size_t>(query_len * Hq * D), 3201);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), 3202);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), 3203);
  const AttentionWindow window{/*left=*/1, /*right=*/1};  // W=2
  const std::vector<float> ref = ComposedPagedRef(
      q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
      block_size, scale, /*causal=*/false, window);

  Queue qq = Q();
  Tensor tq = F32(q, {query_len, Hq, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(),
                      {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(),
                      {num_blocks, block_size, Hk, D});
  Tensor tbt = I32(block_table, {1, max_blocks});
  Tensor tsl = I32(seq_lens, {1});
  Tensor tqsl = I32(qsl, {2});
  std::vector<float> got(ref.size(), 0.0f);
  Tensor tout = F32(got, {query_len, Hq, D});
  PagedAttentionArgs args{scale, false};
  args.window_size = window;
  vt::PagedAttention(qq, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
  }

  // The first query is at absolute p=2 and may see keys 1..3 only. Poison key
  // 0 and keys 4..6; its output must stay bit-identical while later rows may use
  // some of those keys. This directly proves both local bounds are enforced.
  std::vector<float> poisoned_v = vc;
  for (const int64_t key : {0, 4, 5, 6}) {
    const int64_t blk = block_table[static_cast<size_t>(key / block_size)];
    const int64_t off = key % block_size;
    const int64_t base = (blk * block_size + off) * Hk * D;
    for (int64_t e = 0; e < Hk * D; ++e) {
      poisoned_v[static_cast<size_t>(base + e)] = 100000.0f + static_cast<float>(e);
    }
  }
  Tensor poisoned_tvc = Contig(poisoned_v.data(), DType::kF32, Cpu(),
                               {num_blocks, block_size, Hk, D});
  std::vector<float> poisoned_out(got.size(), 0.0f);
  Tensor poisoned_tout = F32(poisoned_out, {query_len, Hq, D});
  vt::PagedAttention(qq, poisoned_tout, tq, tkc, poisoned_tvc, tbt, tsl,
                     tqsl, args);
  for (int64_t i = 0; i < Hq * D; ++i) {
    CHECK(poisoned_out[static_cast<size_t>(i)] == got[static_cast<size_t>(i)]);
  }
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
  // A present window is an actual local window, so both inclusive distances
  // must be non-negative (full attention is represented by nullopt).
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    PagedAttentionArgs bad{1.0f, true};
    bad.window_size = AttentionWindow{-1, 0};
    CHECK_THROWS_AS(
        vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, bad),
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

void CheckCudaWindowCase(const std::vector<int32_t>& qsl,
                         const std::vector<int32_t>& seq_lens, bool causal,
                         AttentionWindow window, uint32_t seed) {
  const int64_t Hq = 4, Hk = 2, D = 32, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const int64_t num_reqs = static_cast<int64_t>(seq_lens.size());
  const int64_t num_tokens = qsl.back();
  const int32_t max_seq = *std::max_element(seq_lens.begin(), seq_lens.end());
  const int64_t max_blocks = (max_seq + block_size - 1) / block_size;
  const int64_t num_blocks = num_reqs * max_blocks;
  std::vector<int32_t> block_table(static_cast<size_t>(num_blocks));
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t b = 0; b < max_blocks; ++b) {
      block_table[static_cast<size_t>(r * max_blocks + b)] =
          static_cast<int32_t>(r * max_blocks + b);
    }
  }
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), seed);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), seed + 1);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), seed + 2);
  const std::vector<float> ref = ComposedPagedRef(
      q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
      block_size, scale, causal, window);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {num_tokens, Hq, D}, q.data());
  DeviceTensor dkc(gpu, g.q, DType::kF32,
                   {num_blocks, block_size, Hk, D}, kc.data());
  DeviceTensor dvc(gpu, g.q, DType::kF32,
                   {num_blocks, block_size, Hk, D}, vc.data());
  DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks},
                   block_table.data());
  DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
  DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {num_tokens, Hq, D});
  PagedAttentionArgs args{scale, causal};
  args.window_size = window;
  vt::PagedAttention(g.q, dout.tensor(), dq.tensor(), dkc.tensor(),
                     dvc.tensor(), dbt.tensor(), dsl.tensor(), dqsl.tensor(),
                     args);
  std::vector<float> got(ref.size(), 0.0f);
  dout.Download(g.q, got.data());
  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-4));
  }
}

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

TEST_CASE("paged_attention CUDA sliding-window masks match upstream decoder and encoder vectors") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping sliding-window CUDA parity (dgx-pending)");
    return;
  }

  // Mixed prefill/decode/chunked-prefill takes the portable tiled prefill path.
  for (const int32_t width : {1, 3, 4, 5}) {
    CAPTURE(width);
    CheckCudaWindowCase({0, 5, 6, 9}, {5, 9, 11}, /*causal=*/true,
                        AttentionWindow{width - 1, 0},
                        5000u + static_cast<uint32_t>(width));
  }
  CheckCudaWindowCase({0, 129}, {129}, /*causal=*/true,
                      AttentionWindow{4095, 0}, 5050);
  // All query lengths are one, so this exercises the graph-safe decode path.
  CheckCudaWindowCase({0, 1, 2}, {9, 11}, /*causal=*/true,
                      AttentionWindow{4, 0}, 5100);
  // Q<K bottom-right alignment with the symmetric encoder window.
  CheckCudaWindowCase({0, 5}, {7}, /*causal=*/false,
                      AttentionWindow{2, 2}, 5200);
}

// ===========================================================================
// bf16 TENSOR-CORE (WMMA) prefill parity. The WMMA flash path fires only for a
// bf16 KV cache (the deployment path — vLLM's bf16 flash_attn store); it casts
// the f32 query to bf16 and runs QKᵀ / P·V on the tensor cores with f32 online
// softmax. Validate at the gate config (head_dim 256, GQA 16q/2kv, multi-tile
// prefill) against the f32 composed reference computed on the SAME bf16-rounded
// K/V — so the residual is query-bf16 + bf16-matmul rounding only (bf16 tol).
// Guarded by HasCuda; dgx-pending on CPU-only boxes.
// ===========================================================================
namespace {
// f32 -> bf16 round-to-nearest-even (host; the test .cpp is not nvcc-compiled).
inline uint16_t F32ToBf16Bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t rounding = 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>((x + rounding) >> 16);
}
inline float Bf16BitsToF32(uint16_t b) {
  uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}
}  // namespace

TEST_CASE("paged_attention CUDA WMMA (bf16 cache) matches f32 ref at head_dim 256") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping paged_attention WMMA bf16 parity (dgx-pending)");
    return;
  }
  const int64_t Hq = 16, Hk = 2, D = 256, block_size = 16;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  std::vector<int32_t> qsl = {0, 100, 101, 104};
  std::vector<int32_t> seq_lens = {100, 133, 140};
  const int64_t num_tokens = 104;
  const int64_t num_reqs = 3;
  const int64_t num_blocks = 64, page = Hk * D, max_blocks = 9;
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 2024);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 137);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 179);
  std::vector<int32_t> block_table = {5,  0,  11, 3,  0,  0, 0, 0, 0,
                                      2,  17, 9,  20, 1,  8, 0, 0, 0,
                                      30, 4,  22, 15, 6, 19, 7, 12, 0};

  // Round K/V to bf16 (bit-identical to what the kernel reads), and build the
  // f32 reference on those same values so only the compute rounding remains.
  std::vector<uint16_t> kc_b(kc.size()), vc_b(vc.size());
  std::vector<float> kc_r(kc.size()), vc_r(vc.size());
  for (size_t i = 0; i < kc.size(); ++i) {
    kc_b[i] = F32ToBf16Bits(kc[i]);
    kc_r[i] = Bf16BitsToF32(kc_b[i]);
    vc_b[i] = F32ToBf16Bits(vc[i]);
    vc_r[i] = Bf16BitsToF32(vc_b[i]);
  }
  std::vector<float> ref = ComposedPagedRef(q, kc_r, vc_r, block_table, max_blocks, seq_lens, qsl,
                                            Hq, Hk, D, block_size, scale, true);

  // Build the ENGINE cache layout: a single (num_blocks, 2, block_size, Hk, D)
  // bf16 allocation; K = unbind(1)[0], V = unbind(1)[1] (block stride 2*bs*Hk*D,
  // V at element offset bs*Hk*D). This is the M1.6 layout trap — the strided
  // slice the runner really feeds PagedAttention (KvSlice, qwen3_5.cpp:778).
  const int64_t within = block_size * Hk * D;
  std::vector<uint16_t> combined(static_cast<size_t>(num_blocks * 2 * within), 0);
  for (int64_t b = 0; b < num_blocks; ++b)
    for (int64_t e = 0; e < within; ++e) {
      combined[static_cast<size_t>((b * 2 + 0) * within + e)] = kc_b[static_cast<size_t>(b * within + e)];
      combined[static_cast<size_t>((b * 2 + 1) * within + e)] = vc_b[static_cast<size_t>(b * within + e)];
    }

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {num_tokens, Hq, D}, q.data());
  // Allocate as rank-1 (kMaxRank=4; the true (nb,2,bs,Hk,D) is rank 5) then build
  // strided rank-4 K/V views into it.
  DeviceTensor dcache(gpu, g.q, DType::kBF16, {num_blocks * 2 * within}, combined.data());
  auto SliceView = [&](int which) {
    Tensor t = dcache.tensor();
    t.data = static_cast<char*>(t.data) +
             static_cast<size_t>(which) * static_cast<size_t>(within) * vt::SizeOf(DType::kBF16);
    t.rank = 4;
    t.shape[0] = num_blocks;
    t.shape[1] = block_size;
    t.shape[2] = Hk;
    t.shape[3] = D;
    t.stride[0] = 2 * within;
    t.stride[1] = Hk * D;
    t.stride[2] = D;
    t.stride[3] = 1;
    return t;
  };
  Tensor kview = SliceView(0);
  Tensor vview = SliceView(1);
  DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks}, block_table.data());
  DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
  DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {num_tokens, Hq, D});
  vt::PagedAttention(g.q, dout.tensor(), dq.tensor(), kview, vview, dbt.tensor(),
                     dsl.tensor(), dqsl.tensor(), PagedAttentionArgs{scale, true});
  std::vector<float> got(static_cast<size_t>(num_tokens * Hq * D), 0.0f);
  dout.Download(g.q, got.data());

  double max_abs = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    max_abs = std::max(max_abs, std::abs(static_cast<double>(got[i]) - ref[i]));
  MESSAGE("WMMA bf16 prefill max_abs_err vs f32 ref = " << max_abs);
  CHECK(max_abs < 5e-2);

  // The same optimized WMMA ladder must honor the local lower bound; FA-2 is
  // ineligible here because query/out are f32.
  const AttentionWindow local_window{31, 0};
  const std::vector<float> local_ref = ComposedPagedRef(
      q, kc_r, vc_r, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
      block_size, scale, /*causal=*/true, local_window);
  DeviceTensor local_out(gpu, g.q, DType::kF32, {num_tokens, Hq, D});
  PagedAttentionArgs local_args{scale, true};
  local_args.window_size = local_window;
  vt::PagedAttention(g.q, local_out.tensor(), dq.tensor(), kview, vview,
                     dbt.tensor(), dsl.tensor(), dqsl.tensor(), local_args);
  std::vector<float> local_got(local_ref.size(), 0.0f);
  local_out.Download(g.q, local_got.data());
  double local_max_abs = 0.0;
  for (size_t i = 0; i < local_ref.size(); ++i) {
    local_max_abs = std::max(
        local_max_abs,
        std::abs(static_cast<double>(local_got[i]) - local_ref[i]));
  }
  MESSAGE("WMMA sliding-window max_abs_err vs f32 ref = " << local_max_abs);
  CHECK(local_max_abs < 5e-2);
}

// ===========================================================================
// CUDA parity at the GATE-MODEL config: head_dim 256, GQA 16q/2kv. This is the
// M2.4 flash prefill path's real shape — head_dim 256 means the warp-per-row
// kernel distributes 8 elements per lane (epl=8), and prefill lengths that span
// several BN key-tiles exercise the cross-tile online-softmax rescale. The
// smaller D=32 case above only covers epl=1, so this pins the production path.
// Guarded by HasCuda; dgx-pending on CPU-only boxes.
// ===========================================================================
TEST_CASE("paged_attention CUDA matches CPU at head_dim 256 (gate config, epl=8)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping paged_attention D=256 parity (dgx-pending)");
    return;
  }
  const int64_t Hq = 16, Hk = 2, D = 256, block_size = 16;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  // 3 reqs: a long prefill (seq 100, spans multiple BN=32 key-tiles), a decode
  // at seq 133, and a short prefill chunk (len 3, seq 140).
  std::vector<int32_t> qsl = {0, 100, 101, 104};
  std::vector<int32_t> seq_lens = {100, 133, 140};
  const int64_t num_tokens = 104;
  const int64_t num_reqs = 3;
  const int64_t num_blocks = 64, page = Hk * D, max_blocks = 9;
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 2024);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 137);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 179);
  // block tables (num_reqs x max_blocks); non-identity, disjoint blocks.
  std::vector<int32_t> block_table = {5,  0,  11, 3,  0,  0, 0, 0, 0,   // req0 seq100 → 7 blocks
                                      2,  17, 9,  20, 1,  8, 0, 0, 0,   // req1 seq133 → 9 blocks
                                      30, 4,  22, 15, 6, 19, 7, 12, 0};  // req2 seq140 → 9 blocks

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

// ===========================================================================
// Vendored FlashAttention-2 prefill (VLLM_CPP_FLASH_ATTN build + VT_FA2_PREFILL
// runtime): the exact flash_fwd_splitkv kernel vLLM runs on GB10, engaged for
// the natively-bf16 combo (bf16 query + bf16 KV + bf16 out, head_dim 256,
// prefill). Validated at the gate config against the f32 composed reference on
// the SAME bf16-rounded q/K/V (residual = FA-2's bf16 tensor-core matmuls +
// f32-softmax rounding + the bf16 output round, same tolerance class as the
// WMMA bf16 test). Also pins the sync-free host-metadata path: passing
// query_start_loc_host + a max_seq_len UPPER BOUND must be bit-identical to the
// fallback (D2H+sync) run — the host values size grids only, never geometry.
// ===========================================================================
#ifdef VLLM_CPP_FLASH_ATTN
namespace {
// Set/restore an env var for the duration of a scope (exception-safe).
struct EnvGuard {
  const char* name;
  std::optional<std::string> previous;
  explicit EnvGuard(const char* n, const char* value) : name(n) {
    if (const char* old = std::getenv(name); old != nullptr) previous = old;
    setenv(name, value, 1);
  }
  ~EnvGuard() {
    if (previous.has_value()) {
      setenv(name, previous->c_str(), 1);
    } else {
      unsetenv(name);
    }
  }
  EnvGuard(const EnvGuard&) = delete;
  EnvGuard& operator=(const EnvGuard&) = delete;
};
}  // namespace

TEST_CASE("paged_attention CUDA FA-2 prefill (bf16 q/kv/out) matches f32 ref at head_dim 256") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 prefill parity (dgx-pending)");
    return;
  }
  const int64_t Hq = 16, Hk = 2, D = 256, block_size = 16;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  // Ragged 3-req batch at the gate config: a long prefill, a decode row, and a
  // short chunked-prefill tail (context 137) — the mixed batch LaunchPaged
  // treats as one prefill call.
  std::vector<int32_t> qsl = {0, 100, 101, 104};
  std::vector<int32_t> seq_lens = {100, 133, 140};
  const int64_t num_tokens = 104;
  const int64_t num_reqs = 3;
  const int64_t num_blocks = 64, page = Hk * D, max_blocks = 9;
  auto qf = RandF32(static_cast<size_t>(num_tokens * Hq * D), 2024);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 137);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 179);
  std::vector<int32_t> block_table = {5,  0,  11, 3,  0,  0, 0, 0, 0,
                                      2,  17, 9,  20, 1,  8, 0, 0, 0,
                                      30, 4,  22, 15, 6, 19, 7, 12, 0};

  // Round q AND K/V to bf16 (bit-identical to what FA-2 reads) and build the
  // f32 reference on those same values, so only FA-2's compute rounding + the
  // bf16 output round remain.
  std::vector<uint16_t> q_b(qf.size()), kc_b(kc.size()), vc_b(vc.size());
  std::vector<float> q_r(qf.size()), kc_r(kc.size()), vc_r(vc.size());
  for (size_t i = 0; i < qf.size(); ++i) {
    q_b[i] = F32ToBf16Bits(qf[i]);
    q_r[i] = Bf16BitsToF32(q_b[i]);
  }
  for (size_t i = 0; i < kc.size(); ++i) {
    kc_b[i] = F32ToBf16Bits(kc[i]);
    kc_r[i] = Bf16BitsToF32(kc_b[i]);
    vc_b[i] = F32ToBf16Bits(vc[i]);
    vc_r[i] = Bf16BitsToF32(vc_b[i]);
  }
  std::vector<float> ref = ComposedPagedRef(q_r, kc_r, vc_r, block_table, max_blocks, seq_lens,
                                            qsl, Hq, Hk, D, block_size, scale, true);

  // ENGINE cache layout: one (num_blocks, 2, block_size, Hk, D) bf16 allocation,
  // K/V = the dim-1 unbind slices (block stride 2*bs*Hk*D) — what the runner
  // really feeds PagedAttention (KvSlice).
  const int64_t within = block_size * Hk * D;
  std::vector<uint16_t> combined(static_cast<size_t>(num_blocks * 2 * within), 0);
  for (int64_t b = 0; b < num_blocks; ++b)
    for (int64_t e = 0; e < within; ++e) {
      combined[static_cast<size_t>((b * 2 + 0) * within + e)] =
          kc_b[static_cast<size_t>(b * within + e)];
      combined[static_cast<size_t>((b * 2 + 1) * within + e)] =
          vc_b[static_cast<size_t>(b * within + e)];
    }

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kBF16, {num_tokens, Hq, D}, q_b.data());
  DeviceTensor dcache(gpu, g.q, DType::kBF16, {num_blocks * 2 * within}, combined.data());
  auto SliceView = [&](int which) {
    Tensor t = dcache.tensor();
    t.data = static_cast<char*>(t.data) +
             static_cast<size_t>(which) * static_cast<size_t>(within) * vt::SizeOf(DType::kBF16);
    t.rank = 4;
    t.shape[0] = num_blocks;
    t.shape[1] = block_size;
    t.shape[2] = Hk;
    t.shape[3] = D;
    t.stride[0] = 2 * within;
    t.stride[1] = Hk * D;
    t.stride[2] = D;
    t.stride[3] = 1;
    return t;
  };
  Tensor kview = SliceView(0);
  Tensor vview = SliceView(1);
  DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks}, block_table.data());
  DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
  DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());

  EnvGuard fa2_on("VT_FA2_PREFILL", "1");

  // Run 1: no host metadata -> the launcher's D2H+sync fallback sizes the grid.
  DeviceTensor dout1(gpu, g.q, DType::kBF16, {num_tokens, Hq, D});
  vt::PagedAttention(g.q, dout1.tensor(), dq.tensor(), kview, vview, dbt.tensor(),
                     dsl.tensor(), dqsl.tensor(), PagedAttentionArgs{scale, true});
  std::vector<uint16_t> got1(static_cast<size_t>(num_tokens * Hq * D), 0);
  dout1.Download(g.q, got1.data());

  // Run 2: sync-free host metadata, with max_seq_len an UPPER BOUND (160 > 140)
  // — must be bit-identical to run 1 (host values size grids, not geometry).
  PagedAttentionArgs host_args{scale, true};
  host_args.query_start_loc_host = qsl.data();
  host_args.max_seq_len = 160;
  DeviceTensor dout2(gpu, g.q, DType::kBF16, {num_tokens, Hq, D});
  vt::PagedAttention(g.q, dout2.tensor(), dq.tensor(), kview, vview, dbt.tensor(),
                     dsl.tensor(), dqsl.tensor(), host_args);
  std::vector<uint16_t> got2(static_cast<size_t>(num_tokens * Hq * D), 0);
  dout2.Download(g.q, got2.data());

  double max_abs = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs,
                       std::abs(static_cast<double>(Bf16BitsToF32(got1[i])) - ref[i]));
  }
  MESSAGE("FA-2 bf16 prefill max_abs_err vs f32 ref = " << max_abs);
  CHECK(max_abs < 5e-2);

  size_t mism = 0;
  for (size_t i = 0; i < got1.size(); ++i)
    if (got1[i] != got2[i]) ++mism;
  MESSAGE("FA-2 host-metadata vs fallback mismatches = " << mism);
  CHECK(mism == 0);

  // Finite windows must dispatch FA-2's LOCAL specialization. In particular,
  // the decoder keeps causal semantics via right=0 while is_causal is normalized
  // false for the local template; leaving the causal template selected would
  // silently ignore the lower bound in the pinned dependency.
  auto check_local = [&](bool causal, AttentionWindow window,
                         const char* label) {
    const std::vector<float> local_ref = ComposedPagedRef(
        q_r, kc_r, vc_r, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
        block_size, scale, causal, window);
    PagedAttentionArgs local_args{scale, causal};
    local_args.window_size = window;
    DeviceTensor local_out(gpu, g.q, DType::kBF16, {num_tokens, Hq, D});
    vt::PagedAttention(g.q, local_out.tensor(), dq.tensor(), kview, vview,
                       dbt.tensor(), dsl.tensor(), dqsl.tensor(), local_args);
    std::vector<uint16_t> local_got(local_ref.size(), 0);
    local_out.Download(g.q, local_got.data());
    double local_max_abs = 0.0;
    for (size_t i = 0; i < local_ref.size(); ++i) {
      local_max_abs = std::max(
          local_max_abs,
          std::abs(static_cast<double>(Bf16BitsToF32(local_got[i])) -
                   local_ref[i]));
    }
    INFO(label);
    CHECK(local_max_abs < 5e-2);
  };
  check_local(/*causal=*/true, AttentionWindow{31, 0},
              "FA-2 causal decoder local window");
  check_local(/*causal=*/false, AttentionWindow{31, 31},
              "FA-2 symmetric encoder local window");
}

namespace {

// Host/device fixture for the pure-decode vectors ported from
// tests/kernels/attention/test_flash_attn.py::test_varlen_with_paged_kv
// (vLLM 702f481, lines 95-217). Every request has qlen=1. Block-table rows use
// shared, permuted physical blocks so long 2k-token vectors stay compact while
// retaining nontrivial paged addressing and head-specific data.
struct Fa2DecodeCase {
  int64_t hq;
  int64_t hk;
  int64_t d = 256;
  int64_t block_size = 16;
  int64_t batch;
  int64_t max_blocks;
  int64_t num_blocks;
  float scale = std::pow(256.0F, -0.5F);
  std::vector<int32_t> seq_lens;
  std::vector<int32_t> qsl;
  std::vector<int32_t> block_table;
  std::vector<uint16_t> query_bf16;
  std::vector<uint16_t> combined_cache;
  std::vector<float> query_rounded;
  std::vector<float> key_rounded;
  std::vector<float> value_rounded;

  Fa2DecodeCase(int64_t query_heads, int64_t kv_heads,
                std::vector<int32_t> lengths, uint32_t seed,
                int64_t capacity_blocks = 0)
      : hq(query_heads),
        hk(kv_heads),
        batch(static_cast<int64_t>(lengths.size())),
        seq_lens(std::move(lengths)) {
    const int32_t max_seq = *std::max_element(seq_lens.begin(), seq_lens.end());
    max_blocks = std::max<int64_t>(capacity_blocks,
                                   (max_seq + block_size - 1) / block_size);
    num_blocks = max_blocks + 17;
    qsl.resize(static_cast<size_t>(batch + 1));
    for (int64_t i = 0; i <= batch; ++i)
      qsl[static_cast<size_t>(i)] = static_cast<int32_t>(i);

    block_table.resize(static_cast<size_t>(batch * max_blocks));
    for (int64_t r = 0; r < batch; ++r) {
      for (int64_t b = 0; b < max_blocks; ++b) {
        block_table[static_cast<size_t>(r * max_blocks + b)] =
            static_cast<int32_t>((r * 13 + b * 7) % num_blocks);
      }
    }

    const auto qf = RandF32(static_cast<size_t>(batch * hq * d), seed);
    const auto kf = RandF32(
        static_cast<size_t>(num_blocks * block_size * hk * d), seed + 1);
    const auto vf = RandF32(kf.size(), seed + 2);
    query_bf16.resize(qf.size());
    query_rounded.resize(qf.size());
    std::vector<uint16_t> key_bf16(kf.size()), value_bf16(vf.size());
    key_rounded.resize(kf.size());
    value_rounded.resize(vf.size());
    for (size_t i = 0; i < qf.size(); ++i) {
      query_bf16[i] = F32ToBf16Bits(qf[i]);
      query_rounded[i] = Bf16BitsToF32(query_bf16[i]);
    }
    for (size_t i = 0; i < kf.size(); ++i) {
      key_bf16[i] = F32ToBf16Bits(kf[i]);
      value_bf16[i] = F32ToBf16Bits(vf[i]);
      key_rounded[i] = Bf16BitsToF32(key_bf16[i]);
      value_rounded[i] = Bf16BitsToF32(value_bf16[i]);
    }

    const int64_t within = block_size * hk * d;
    combined_cache.assign(static_cast<size_t>(num_blocks * 2 * within), 0);
    for (int64_t block = 0; block < num_blocks; ++block) {
      for (int64_t e = 0; e < within; ++e) {
        const size_t source = static_cast<size_t>(block * within + e);
        combined_cache[static_cast<size_t>((block * 2) * within + e)] =
            key_bf16[source];
        combined_cache[static_cast<size_t>((block * 2 + 1) * within + e)] =
            value_bf16[source];
      }
    }
  }

  Tensor CacheView(DeviceTensor& cache, int which) const {
    const int64_t within = block_size * hk * d;
    Tensor view = cache.tensor();
    view.data = static_cast<char*>(view.data) +
                static_cast<size_t>(which * within) * vt::SizeOf(DType::kBF16);
    view.rank = 4;
    view.shape[0] = num_blocks;
    view.shape[1] = block_size;
    view.shape[2] = hk;
    view.shape[3] = d;
    view.stride[0] = 2 * within;
    view.stride[1] = hk * d;
    view.stride[2] = d;
    view.stride[3] = 1;
    return view;
  }

  std::vector<float> Reference(
      const std::vector<int32_t>& lengths,
      std::optional<AttentionWindow> window = std::nullopt) const {
    return ComposedPagedRef(query_rounded, key_rounded, value_rounded,
                            block_table, max_blocks, lengths, qsl, hq, hk, d,
                            block_size, scale, /*causal=*/true, window);
  }
};

void CheckBf16AgainstReference(const std::vector<uint16_t>& got,
                               const std::vector<float>& reference,
                               const char* label) {
  REQUIRE(got.size() == reference.size());
  size_t violations = 0;
  double max_abs = 0.0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const double actual = Bf16BitsToF32(got[i]);
    const double error = std::abs(actual - static_cast<double>(reference[i]));
    max_abs = std::max(max_abs, error);
    if (error > 1.5e-2 + 1.0e-2 * std::abs(static_cast<double>(reference[i]))) {
      ++violations;
    }
  }
  INFO(label);
  INFO(max_abs);
  CHECK(violations == 0);
}

struct Fa2DecodeRunStats {
  uint64_t launches = 0;
  uint64_t split_launches = 0;
  uint64_t no_split_launches = 0;
};

Fa2DecodeRunStats RunFa2DecodeCase(Fa2DecodeCase& c, const char* toggle,
                                   bool expect_fa2,
                                   std::optional<AttentionWindow> window = std::nullopt) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor query(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d},
                     c.query_bf16.data());
  DeviceTensor cache(gpu, guard.q, DType::kBF16,
                     {static_cast<int64_t>(c.combined_cache.size())},
                     c.combined_cache.data());
  Tensor key = c.CacheView(cache, 0);
  Tensor value = c.CacheView(cache, 1);
  DeviceTensor block_table(gpu, guard.q, DType::kI32,
                           {c.batch, c.max_blocks}, c.block_table.data());
  DeviceTensor seq_lens(gpu, guard.q, DType::kI32, {c.batch},
                        c.seq_lens.data());
  DeviceTensor qsl(gpu, guard.q, DType::kI32, {c.batch + 1}, c.qsl.data());
  DeviceTensor out(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d});

  EnvGuard decode_toggle("VT_FA2_DECODE", toggle);
  vt::cuda::testing::ResetFa2DecodeDebugCounters();
  PagedAttentionArgs args{c.scale, true};
  args.window_size = window;
  args.query_start_loc_host = c.qsl.data();
  args.max_seq_len = static_cast<int>(c.max_blocks * c.block_size);
  vt::PagedAttention(guard.q, out.tensor(), query.tensor(), key, value,
                     block_table.tensor(), seq_lens.tensor(), qsl.tensor(), args);
  std::vector<uint16_t> got(c.query_bf16.size(), 0);
  out.Download(guard.q, got.data());

  const Fa2DecodeRunStats stats{
      vt::cuda::testing::Fa2DecodeLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeSplitLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeNoSplitLaunchesForTesting()};
  vt::cuda::testing::DisableFa2DecodeDebugCounters();
  CheckBf16AgainstReference(got, c.Reference(c.seq_lens, window),
                            expect_fa2 ? "FA2 decode" : "paged fallback");
  CHECK(stats.launches == (expect_fa2 ? 1U : 0U));
  CHECK(stats.split_launches + stats.no_split_launches == stats.launches);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(
            guard.q.device.index, guard.q.handle) == (expect_fa2 ? 1U : 0U));
  return stats;
}

}  // namespace

TEST_CASE("paged_attention CUDA FA-2 split heuristic mirrors upstream") {
  using vt::cuda::testing::Fa2DecodeNumSplitsForTesting;
  CHECK(Fa2DecodeNumSplitsForTesting(32, 40, 18, 128) == 1);
  CHECK(Fa2DecodeNumSplitsForTesting(4, 40, 18, 128) == 9);
  CHECK(Fa2DecodeNumSplitsForTesting(8, 40, 32, 128) == 5);
  CHECK(Fa2DecodeNumSplitsForTesting(4, 40, 18, 2) == 2);
  CHECK(Fa2DecodeNumSplitsForTesting(1, 40, 1, 128) == 1);
}

TEST_CASE("paged_attention CUDA pure-decode upstream paged vector retains fallback correctness") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping upstream FA-2 pure-decode vector (dgx-pending)");
    return;
  }
  // Exact upstream sequence-length vector. Hq/Hkv=8/2 is outside W3-G's first
  // ratio-6 slice, so it proves honest fallback rather than broadened support.
  Fa2DecodeCase c(/*Hq=*/8, /*Hkv=*/2, {523, 37, 2011}, 6120);
  RunFa2DecodeCase(c, "1", /*expect_fa2=*/false);
}

TEST_CASE("paged_attention CUDA FA-2 ratio-6 pure decode matches composed reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 ratio-6 decode parity (dgx-pending)");
    return;
  }
  for (const int batch : {1, 2, 4, 8, 16}) {
    CAPTURE(batch);
    std::vector<int32_t> lengths(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) lengths[static_cast<size_t>(i)] = 1024 + i * 7;
    Fa2DecodeCase c(/*Hq=*/24, /*Hkv=*/4, std::move(lengths),
                    6200U + static_cast<uint32_t>(batch));
    RunFa2DecodeCase(c, "1", /*expect_fa2=*/true);
  }
}

TEST_CASE("paged_attention CUDA FA-2 decode toggle and invalid eligibility use fallback") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 decode fallback vectors (dgx-pending)");
    return;
  }
  Fa2DecodeCase ratio6(/*Hq=*/24, /*Hkv=*/4, {1024, 1057}, 6300);
  RunFa2DecodeCase(ratio6, "0", /*expect_fa2=*/false);
  RunFa2DecodeCase(ratio6, "1", /*expect_fa2=*/false,
                   AttentionWindow{127, 0});

  // The Qwen3.6-35B ratio-8 topology is deliberately inert in this slice.
  Fa2DecodeCase ratio8(/*Hq=*/16, /*Hkv=*/2, {1024, 1057}, 6310);
  RunFa2DecodeCase(ratio8, "1", /*expect_fa2=*/false);
}

TEST_CASE("paged_attention CUDA FA-2 decode scratch is capture-stable across replay") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 capture/replay vector (dgx-pending)");
    return;
  }
  Fa2DecodeCase c(/*Hq=*/24, /*Hkv=*/4, {257, 389}, 6400,
                  /*capacity_blocks=*/72);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor query(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d},
                     c.query_bf16.data());
  DeviceTensor cache(gpu, guard.q, DType::kBF16,
                     {static_cast<int64_t>(c.combined_cache.size())},
                     c.combined_cache.data());
  Tensor key = c.CacheView(cache, 0);
  Tensor value = c.CacheView(cache, 1);
  DeviceTensor block_table(gpu, guard.q, DType::kI32,
                           {c.batch, c.max_blocks}, c.block_table.data());
  DeviceTensor seq_lens(gpu, guard.q, DType::kI32, {c.batch},
                        c.seq_lens.data());
  DeviceTensor qsl(gpu, guard.q, DType::kI32, {c.batch + 1}, c.qsl.data());
  DeviceTensor out(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d});
  PagedAttentionArgs args{c.scale, true};
  args.query_start_loc_host = c.qsl.data();
  args.max_seq_len = static_cast<int>(c.max_blocks * c.block_size);

  EnvGuard decode_on("VT_FA2_DECODE", "1");
  vt::cuda::testing::ResetFa2DecodeDebugCounters();

  // Cold eager: materialize the one capture-stable shape entry.
  vt::PagedAttention(guard.q, out.tensor(), query.tensor(), key, value,
                     block_table.tensor(), seq_lens.tensor(), qsl.tensor(), args);
  std::vector<uint16_t> warm(c.query_bf16.size(), 0);
  out.Download(guard.q, warm.data());
  CheckBf16AgainstReference(warm, c.Reference(c.seq_lens), "FA2 cold eager");
  CHECK(vt::cuda::testing::Fa2DecodeScratchAllocationsForTesting() == 1);
  CHECK(vt::cuda::testing::Fa2DecodeScratchReusesForTesting() == 0);

  // Capture must be a pure pool hit: no pointer changes or allocator calls.
  gpu.BeginCapture(guard.q);
  vt::PagedAttention(guard.q, out.tensor(), query.tensor(), key, value,
                     block_table.tensor(), seq_lens.tensor(), qsl.tensor(), args);
  void* graph = gpu.EndCaptureGraph(guard.q);
  CHECK(vt::cuda::testing::Fa2DecodeScratchAllocationsForTesting() == 1);
  CHECK(vt::cuda::testing::Fa2DecodeScratchReusesForTesting() == 1);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(
            guard.q.device.index, guard.q.handle) == 1);

  // Grow actual sequence lengths inside the fixed block-table capacity. The
  // captured geometry and scratch addresses stay fixed; seqused_k changes.
  const std::vector<int32_t> grown_lens = {511, 887};
  gpu.Copy(guard.q, seq_lens.tensor().data, grown_lens.data(),
           grown_lens.size() * sizeof(int32_t));
  gpu.ReplayGraph(guard.q, graph);
  std::vector<uint16_t> replay1(c.query_bf16.size(), 0);
  out.Download(guard.q, replay1.data());
  CheckBf16AgainstReference(replay1, c.Reference(grown_lens), "FA2 replay 1");
  gpu.ReplayGraph(guard.q, graph);
  std::vector<uint16_t> replay2(c.query_bf16.size(), 0);
  out.Download(guard.q, replay2.data());
  CHECK(replay1 == replay2);
  gpu.DestroyGraph(graph);
  vt::cuda::testing::DisableFa2DecodeDebugCounters();

  // A new block-table column capacity gets a second stable entry; the first is
  // retained rather than grown/freed because an independently owned graph may
  // still reference it in production.
  Fa2DecodeCase wider(/*Hq=*/24, /*Hkv=*/4, {511, 887}, 6410,
                      /*capacity_blocks=*/80);
  DeviceTensor query2(gpu, guard.q, DType::kBF16,
                      {wider.batch, wider.hq, wider.d},
                      wider.query_bf16.data());
  DeviceTensor cache2(gpu, guard.q, DType::kBF16,
                      {static_cast<int64_t>(wider.combined_cache.size())},
                      wider.combined_cache.data());
  Tensor key2 = wider.CacheView(cache2, 0);
  Tensor value2 = wider.CacheView(cache2, 1);
  DeviceTensor block_table2(gpu, guard.q, DType::kI32,
                            {wider.batch, wider.max_blocks},
                            wider.block_table.data());
  DeviceTensor seq_lens2(gpu, guard.q, DType::kI32, {wider.batch},
                         wider.seq_lens.data());
  DeviceTensor qsl2(gpu, guard.q, DType::kI32, {wider.batch + 1},
                    wider.qsl.data());
  DeviceTensor out2(gpu, guard.q, DType::kBF16,
                    {wider.batch, wider.hq, wider.d});
  PagedAttentionArgs args2{wider.scale, true};
  args2.query_start_loc_host = wider.qsl.data();
  args2.max_seq_len = static_cast<int>(wider.max_blocks * wider.block_size);
  vt::PagedAttention(guard.q, out2.tensor(), query2.tensor(), key2, value2,
                     block_table2.tensor(), seq_lens2.tensor(), qsl2.tensor(),
                     args2);
  gpu.Synchronize(guard.q);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(
            guard.q.device.index, guard.q.handle) == 2);
}

TEST_CASE("paged_attention CUDA FA-2 decode scratch is queue-owned and released") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 queue lifecycle vector (dgx-pending)");
    return;
  }
  Fa2DecodeCase c(/*Hq=*/24, /*Hkv=*/4, {1024}, 6500);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard first(gpu);
  QueueGuard second(gpu);
  const auto launch = [&](Queue& queue) {
    DeviceTensor query(gpu, queue, DType::kBF16, {c.batch, c.hq, c.d},
                       c.query_bf16.data());
    DeviceTensor cache(gpu, queue, DType::kBF16,
                       {static_cast<int64_t>(c.combined_cache.size())},
                       c.combined_cache.data());
    Tensor key = c.CacheView(cache, 0);
    Tensor value = c.CacheView(cache, 1);
    DeviceTensor block_table(gpu, queue, DType::kI32,
                             {c.batch, c.max_blocks}, c.block_table.data());
    DeviceTensor seq_lens(gpu, queue, DType::kI32, {c.batch},
                          c.seq_lens.data());
    DeviceTensor qsl(gpu, queue, DType::kI32, {c.batch + 1}, c.qsl.data());
    DeviceTensor out(gpu, queue, DType::kBF16, {c.batch, c.hq, c.d});
    PagedAttentionArgs args{c.scale, true};
    args.query_start_loc_host = c.qsl.data();
    args.max_seq_len = static_cast<int>(c.max_blocks * c.block_size);
    vt::PagedAttention(queue, out.tensor(), query.tensor(), key, value,
                       block_table.tensor(), seq_lens.tensor(), qsl.tensor(),
                       args);
    gpu.Synchronize(queue);
  };

  EnvGuard decode_on("VT_FA2_DECODE", "1");
  launch(first.q);
  launch(second.q);
  void* first_handle = first.q.handle;
  void* second_handle = second.q.handle;
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, first_handle) == 1);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, second_handle) == 1);

  gpu.DestroyQueue(first.q);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, first_handle) == 0);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, second_handle) == 1);
  gpu.DestroyQueue(second.q);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, second_handle) == 0);
}
#else
TEST_CASE("paged_attention CUDA FA-2 prefill (bf16 q/kv/out) matches f32 ref at head_dim 256") {
  MESSAGE("built without VLLM_CPP_FLASH_ATTN; FA-2 prefill parity skipped");
}
#endif  // VLLM_CPP_FLASH_ATTN
