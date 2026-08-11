// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// PERF-CPU-ATTN-DTYPE bit-identity gate (spec
// .agents/specs/cpu-decode-barrier-and-attn-dispatch.md §3, issue #391).
//
// The CPU paged-attention kernel used to branch on the operand dtype ONCE PER
// ELEMENT inside the K/V reduction (`LoadF32(tensor, offset)`, the switch at
// the old cpu_paged_attn.cpp:29). That dispatch is now hoisted out of the loop
// the same way the elementwise GEMM hoisted its own (cpu_ops.cpp:98-107):
// the query row is widened once by `WidenRowToF32` and the cache dtype selects
// a typed element accessor once per call.
//
// A dispatch hoist changes WHEN the branch is taken, never WHAT is computed, so
// it must be BIT-IDENTICAL — not close, identical. This file pins that:
// `PagedAttentionPerElementRef` below is the pre-change kernel copied verbatim
// (same per-element `LoadF32`/`StoreF32`/`LoadK`/`LoadV`, same loop nest, same
// accumulation order), and every case compares the two outputs with `memcmp`
// over the raw storage bytes across the FULL dtype cross-product the switch
// handled: query f32/f16/bf16 x cache f32/f16/bf16/fp8 x out f32/bf16, with
// GQA, sliding windows, softcap, block-spanning sequences and varlen batches.
//
// Mutating either hoist (widen with the wrong dtype, bind the wrong typed
// accessor, drop the bf16 rounding in the row store) makes memcmp fail here.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/ops.h"

using vt::AttentionWindow;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Fp8KVCacheDataType;
using vt::PagedAttentionArgs;
using vt::Queue;
using vt::Tensor;

namespace {

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

// ---------------------------------------------------------------------------
// The PRE-CHANGE kernel, copied verbatim from cpu_paged_attn.cpp @ a26555ed.
// Deliberately not refactored: it is the reference the hoisted kernel must
// reproduce byte-for-byte, so it keeps the per-element switch it had.
// ---------------------------------------------------------------------------
float RefLoadF32(const Tensor& t, int64_t elem_offset) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[elem_offset];
    case DType::kF16: return vt::F16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    case DType::kBF16: return vt::BF16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    default: FAIL("paged_attention ref LoadF32: unsupported dtype"); return 0.0f;
  }
}

void RefStoreF32(const Tensor& t, int64_t elem_offset, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[elem_offset] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[elem_offset] = vt::F32ToBF16(v); break;
    default: FAIL("paged_attention ref StoreF32: unsupported dtype");
  }
}

void PagedAttentionPerElementRef(Tensor& out, const Tensor& query, const Tensor& k_cache,
                                 const Tensor& v_cache, const Tensor& block_table,
                                 const Tensor& seq_lens, const Tensor& query_start_loc,
                                 const PagedAttentionArgs& args) {
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t qpk = hq / num_kv_heads;
  const float scale = args.scale;
  const float softcap = args.logits_soft_cap;
  const int64_t window_left = args.window_size.has_value() ? args.window_size->left : -1;
  const int64_t window_right = args.window_size.has_value() ? args.window_size->right : -1;

  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  const int64_t kc_blk = k_cache.stride[0], kc_pg = k_cache.stride[1], kc_hd = k_cache.stride[2];
  const int64_t vc_blk = v_cache.stride[0], vc_pg = v_cache.stride[1], vc_hd = v_cache.stride[2];

  const bool kv_fp8 = args.kv_cache_dtype != Fp8KVCacheDataType::kAuto;
  const uint8_t* k_fp8 = kv_fp8 ? k_cache.Ptr<uint8_t>() : nullptr;
  const uint8_t* v_fp8 = kv_fp8 ? v_cache.Ptr<uint8_t>() : nullptr;
  const float k_scale = args.k_scale;
  const float v_scale = args.v_scale;
  auto LoadK = [&](int64_t off) -> float {
    return kv_fp8 ? vt::LoadKvFp8E4M3(k_fp8[off], k_scale) : RefLoadF32(k_cache, off);
  };
  auto LoadV = [&](int64_t off) -> float {
    return kv_fp8 ? vt::LoadKvFp8E4M3(v_fp8[off], v_scale) : RefLoadF32(v_cache, off);
  };

  std::vector<int32_t> tok_pos(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_slen(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_req(static_cast<size_t>(total_q));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q0 = qsl[r], q1 = qsl[r + 1];
    const int64_t query_len = q1 - q0;
    if (query_len <= 0) continue;
    const int64_t seqlen = slens[r];
    const int64_t context = seqlen - query_len;
    for (int64_t local = 0; local < query_len; ++local) {
      tok_pos[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(context + local);
      tok_slen[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(seqlen);
      tok_req[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(r);
    }
  }

  std::vector<float> probs;
  std::vector<float> acc(static_cast<size_t>(d));
  for (int64_t t = 0; t < total_q; ++t) {
    const int64_t r = tok_req[static_cast<size_t>(t)];
    const int64_t p = tok_pos[static_cast<size_t>(t)];
    const int64_t seqlen = tok_slen[static_cast<size_t>(t)];
    const int64_t jmin = window_left >= 0 ? std::max<int64_t>(0, p - window_left) : 0;
    int64_t jmax = args.causal ? p : seqlen - 1;
    if (window_right >= 0) jmax = std::min(jmax, p + window_right);
    jmax = std::min(jmax, seqlen - 1);
    if (jmax < jmin) continue;
    probs.assign(static_cast<size_t>(jmax - jmin + 1), 0.0f);
    for (int64_t h = 0; h < hq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (t * hq + h) * d;
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = jmin; j <= jmax; ++j) {
        const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
        const int64_t off = j % block_size;
        const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
        float dot = 0.0f;
        for (int64_t e = 0; e < d; ++e) dot += RefLoadF32(query, qoff + e) * LoadK(kbase + e);
        dot *= scale;
        if (softcap > 0.0f) dot = softcap * std::tanh(dot / softcap);
        probs[static_cast<size_t>(j - jmin)] = dot;
        if (dot > m) m = dot;
      }
      float denom = 0.0f;
      for (int64_t j = jmin; j <= jmax; ++j) {
        const float e = std::exp(probs[static_cast<size_t>(j - jmin)] - m);
        probs[static_cast<size_t>(j - jmin)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
      for (int64_t j = jmin; j <= jmax; ++j) {
        const float pw = probs[static_cast<size_t>(j - jmin)] * inv;
        const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
        const int64_t off = j % block_size;
        const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
        for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += pw * LoadV(vbase + e);
      }
      for (int64_t e = 0; e < d; ++e) RefStoreF32(out, qoff + e, acc[static_cast<size_t>(e)]);
    }
  }
}

// ---------------------------------------------------------------------------
// Deterministic operand generation. One f32 source per tensor, encoded into
// whichever storage dtype the case asks for, so the two kernels see the SAME
// bytes and any difference is the dispatch and nothing else.
// ---------------------------------------------------------------------------
uint32_t NextRand(uint32_t& s) {  // xorshift32; no <random> implementation drift
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

float RandUnit(uint32_t& s) {
  return static_cast<float>(NextRand(s) % 2048u) / 1024.0f - 1.0f;  // [-1, 1)
}

// Storage bytes for `src` at `dt`. f32/f16/bf16 round-trip the value; kI8 is
// the fp8 KV page encoding (StoreKvFp8E4M3 with the matching scale).
std::vector<uint8_t> Encode(const std::vector<float>& src, DType dt, float fp8_scale) {
  std::vector<uint8_t> bytes;
  switch (dt) {
    case DType::kF32:
      bytes.resize(src.size() * 4);
      std::memcpy(bytes.data(), src.data(), bytes.size());
      break;
    case DType::kF16: {
      bytes.resize(src.size() * 2);
      auto* p = reinterpret_cast<uint16_t*>(bytes.data());
      for (size_t i = 0; i < src.size(); ++i) p[i] = vt::F32ToF16(src[i]);
      break;
    }
    case DType::kBF16: {
      bytes.resize(src.size() * 2);
      auto* p = reinterpret_cast<uint16_t*>(bytes.data());
      for (size_t i = 0; i < src.size(); ++i) p[i] = vt::F32ToBF16(src[i]);
      break;
    }
    case DType::kI8: {
      bytes.resize(src.size());
      for (size_t i = 0; i < src.size(); ++i) bytes[i] = vt::StoreKvFp8E4M3(src[i], fp8_scale);
      break;
    }
    default: FAIL("test Encode: unsupported dtype");
  }
  return bytes;
}

size_t ElemBytes(DType dt) { return dt == DType::kI8 ? 1u : (dt == DType::kF32 ? 4u : 2u); }

// One shape/masking configuration, run through both kernels at one dtype triple.
struct Case {
  std::string name;
  int64_t num_reqs = 1;
  std::vector<int32_t> qsl;       // [num_reqs + 1]
  std::vector<int32_t> seq_lens;  // [num_reqs]
  int64_t hq = 4;
  int64_t hkv = 2;
  int64_t d = 8;
  int64_t block_size = 4;
  int64_t max_blocks = 4;
  PagedAttentionArgs args{};
};

// Runs `c` at (query dtype, cache dtype, out dtype) and returns true when the
// hoisted kernel's OUTPUT BYTES equal the per-element reference's, byte for
// byte. Any mismatch is reported with its element index.
void RunBitIdentical(const Case& c, DType q_dt, DType kv_dt, DType out_dt, uint32_t seed) {
  const bool fp8 = kv_dt == DType::kI8;
  const int64_t total_q = c.qsl.back();
  const int64_t num_blocks = c.num_reqs * c.max_blocks;
  const size_t q_elems = static_cast<size_t>(total_q * c.hq * c.d);
  const size_t cache_elems = static_cast<size_t>(num_blocks * c.block_size * c.hkv * c.d);

  uint32_t s = seed;
  std::vector<float> qf(q_elems), kf(cache_elems), vf(cache_elems);
  for (auto& x : qf) x = RandUnit(s);
  for (auto& x : kf) x = RandUnit(s);
  for (auto& x : vf) x = RandUnit(s);

  PagedAttentionArgs args = c.args;
  if (fp8) {
    args.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
    args.k_scale = 0.75f;
    args.v_scale = 1.25f;
  }

  std::vector<uint8_t> qb = Encode(qf, q_dt, 1.0f);
  std::vector<uint8_t> kb = Encode(kf, kv_dt, args.k_scale);
  std::vector<uint8_t> vb = Encode(vf, kv_dt, args.v_scale);

  std::vector<int32_t> btab(static_cast<size_t>(c.num_reqs * c.max_blocks));
  for (size_t i = 0; i < btab.size(); ++i) btab[i] = static_cast<int32_t>(i);
  std::vector<int32_t> qsl = c.qsl, slens = c.seq_lens;

  const std::vector<int64_t> q_shape = {total_q, c.hq, c.d};
  const std::vector<int64_t> c_shape = {num_blocks, c.block_size, c.hkv, c.d};
  Tensor tq = Contig(qb.data(), q_dt, q_shape);
  Tensor tk = Contig(kb.data(), kv_dt, c_shape);
  Tensor tv = Contig(vb.data(), kv_dt, c_shape);
  Tensor tbt = Contig(btab.data(), DType::kI32, {c.num_reqs, c.max_blocks});
  Tensor tsl = Contig(slens.data(), DType::kI32, {c.num_reqs});
  Tensor tqsl = Contig(qsl.data(), DType::kI32, {c.num_reqs + 1});

  // 0xA5 fill so an output element the kernel never writes still differs from
  // a zero-initialised buffer and cannot pass by accident.
  const size_t out_bytes = q_elems * ElemBytes(out_dt);
  std::vector<uint8_t> got(out_bytes, 0xA5), want(out_bytes, 0xA5);
  Tensor tgot = Contig(got.data(), out_dt, q_shape);
  Tensor twant = Contig(want.data(), out_dt, q_shape);

  Queue qq = Q();
  vt::PagedAttention(qq, tgot, tq, tk, tv, tbt, tsl, tqsl, args);
  PagedAttentionPerElementRef(twant, tq, tk, tv, tbt, tsl, tqsl, args);

  const bool identical = std::memcmp(got.data(), want.data(), out_bytes) == 0;
  CHECK_MESSAGE(identical, c.name);
  if (!identical) {
    for (size_t i = 0; i < out_bytes; ++i) {
      if (got[i] != want[i]) {
        MESSAGE("first differing storage byte at index " << i << ": got " << int(got[i])
                                                         << " want " << int(want[i]));
        break;
      }
    }
  }
}

std::vector<Case> Cases() {
  std::vector<Case> cs;

  Case decode;  // batch-1 decode: one query token over a 7-long context
  decode.name = "decode c1";
  decode.qsl = {0, 1};
  decode.seq_lens = {7};
  decode.args = PagedAttentionArgs{0.35f, true};
  cs.push_back(decode);

  Case prefill;  // block-spanning prefill: 12 tokens, block_size 4 => 3 blocks
  prefill.name = "prefill 12 tokens spanning blocks";
  prefill.qsl = {0, 12};
  prefill.seq_lens = {12};
  prefill.max_blocks = 3;
  prefill.args = PagedAttentionArgs{0.35f, true};
  cs.push_back(prefill);

  Case gqa;  // hq/hkv = 6 so the q-head -> kv-head map is exercised
  gqa.name = "GQA 6:1 prefill";
  gqa.qsl = {0, 5};
  gqa.seq_lens = {5};
  gqa.hq = 6;
  gqa.hkv = 1;
  gqa.d = 16;
  gqa.args = PagedAttentionArgs{0.25f, true};
  cs.push_back(gqa);

  Case varlen;  // two requests: a 4-token prefill and a decode at seq_len 6
  varlen.name = "varlen batch (prefill + decode)";
  varlen.num_reqs = 2;
  varlen.qsl = {0, 4, 5};
  varlen.seq_lens = {4, 6};
  varlen.args = PagedAttentionArgs{0.35f, true};
  cs.push_back(varlen);

  Case window;  // sliding window clips both sides of the causal band
  window.name = "sliding window left=2 right=0";
  window.qsl = {0, 9};
  window.seq_lens = {9};
  window.max_blocks = 3;
  window.args = PagedAttentionArgs{0.35f, true};
  window.args.window_size = AttentionWindow{2, 0};
  cs.push_back(window);

  Case noncausal;  // encoder-style full attention
  noncausal.name = "non-causal (encoder)";
  noncausal.qsl = {0, 6};
  noncausal.seq_lens = {6};
  noncausal.max_blocks = 2;
  noncausal.args = PagedAttentionArgs{0.35f, false};
  cs.push_back(noncausal);

  Case softcap;  // gemma2/3/4 logit soft-cap arm
  softcap.name = "logits_soft_cap 50";
  softcap.qsl = {0, 6};
  softcap.seq_lens = {6};
  softcap.max_blocks = 2;
  softcap.args = PagedAttentionArgs{0.35f, true};
  softcap.args.logits_soft_cap = 50.0f;
  cs.push_back(softcap);

  return cs;
}

const char* DTypeName(DType dt) {
  switch (dt) {
    case DType::kF32: return "f32";
    case DType::kF16: return "f16";
    case DType::kBF16: return "bf16";
    case DType::kI8: return "fp8-e4m3";
    default: return "?";
  }
}

}  // namespace

// Every dtype the hoisted dispatch can resolve to, on every masking shape. The
// bar is memcmp == 0, not a tolerance: a dispatch hoist that changes a single
// bit has changed something other than the dispatch.
TEST_CASE("CPU paged attention: the hoisted dtype dispatch is BIT-IDENTICAL") {
  const DType q_dtypes[] = {DType::kF32, DType::kF16, DType::kBF16};
  const DType kv_dtypes[] = {DType::kF32, DType::kF16, DType::kBF16, DType::kI8};
  const DType out_dtypes[] = {DType::kF32, DType::kBF16};

  uint32_t seed = 0x1234567u;
  for (const Case& base : Cases()) {
    for (DType q_dt : q_dtypes) {
      for (DType kv_dt : kv_dtypes) {
        for (DType out_dt : out_dtypes) {
          Case c = base;
          c.name = base.name + " [q=" + DTypeName(q_dt) + " kv=" + DTypeName(kv_dt) +
                   " out=" + DTypeName(out_dt) + "]";
          seed = seed * 1664525u + 1013904223u;
          RunBitIdentical(c, q_dt, kv_dt, out_dt, seed | 1u);
        }
      }
    }
  }
}

// The hoist widens the query row ONCE per token and reuses it across q-heads.
// A head-offset slip inside that buffer is invisible when every q-head holds
// the same values, so this case gives each q-head a distinct magnitude and
// each query token a distinct one too.
TEST_CASE("CPU paged attention: the widened query row keeps per-head offsets") {
  const int64_t T = 3, HQ = 4, HKV = 2, D = 8, BS = 4, NB = 2;
  std::vector<float> qf(static_cast<size_t>(T * HQ * D));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < HQ; ++h)
      for (int64_t e = 0; e < D; ++e)
        qf[static_cast<size_t>((t * HQ + h) * D + e)] =
            static_cast<float>(t + 1) * 8.0f + static_cast<float>(h + 1) +
            static_cast<float>(e) * 0.125f;

  uint32_t s = 0xBEEF01u;
  const size_t cache_elems = static_cast<size_t>(NB * BS * HKV * D);
  std::vector<float> kf(cache_elems), vf(cache_elems);
  for (auto& x : kf) x = RandUnit(s);
  for (auto& x : vf) x = RandUnit(s);

  for (DType q_dt : {DType::kF32, DType::kF16, DType::kBF16}) {
    std::vector<uint8_t> qb = Encode(qf, q_dt, 1.0f);
    std::vector<uint8_t> kb = Encode(kf, DType::kBF16, 1.0f);
    std::vector<uint8_t> vb = Encode(vf, DType::kBF16, 1.0f);
    std::vector<int32_t> btab = {0, 1};
    std::vector<int32_t> slens = {T};
    std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};

    Tensor tq = Contig(qb.data(), q_dt, {T, HQ, D});
    Tensor tk = Contig(kb.data(), DType::kBF16, {NB, BS, HKV, D});
    Tensor tv = Contig(vb.data(), DType::kBF16, {NB, BS, HKV, D});
    Tensor tbt = Contig(btab.data(), DType::kI32, {1, 2});
    Tensor tsl = Contig(slens.data(), DType::kI32, {1});
    Tensor tqsl = Contig(qsl.data(), DType::kI32, {2});

    std::vector<float> got(static_cast<size_t>(T * HQ * D), -1.0f);
    std::vector<float> want(static_cast<size_t>(T * HQ * D), -1.0f);
    Tensor tgot = Contig(got.data(), DType::kF32, {T, HQ, D});
    Tensor twant = Contig(want.data(), DType::kF32, {T, HQ, D});

    Queue qq = Q();
    PagedAttentionArgs args{0.35f, true};
    vt::PagedAttention(qq, tgot, tq, tk, tv, tbt, tsl, tqsl, args);
    PagedAttentionPerElementRef(twant, tq, tk, tv, tbt, tsl, tqsl, args);

    const std::string label = std::string("per-head query offsets, q=") + DTypeName(q_dt);
    CHECK_MESSAGE(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0, label);
  }
}

// The kernel resolves ONE KvKind for both caches. That is only sound because
// `vt::PagedAttention` refuses a mismatched pair up front, so this pins the
// invariant the single dispatch rests on: if this check is ever relaxed, the
// kernel needs two dispatches and this test says so.
TEST_CASE("CPU paged attention: a mismatched K/V cache dtype is refused, not dispatched") {
  const int64_t T = 4, HQ = 2, HKV = 2, D = 8, BS = 4, NB = 1;
  uint32_t s = 0xC0FFEEu;
  const size_t cache_elems = static_cast<size_t>(NB * BS * HKV * D);
  std::vector<float> qf(static_cast<size_t>(T * HQ * D)), kf(cache_elems), vf(cache_elems);
  for (auto& x : qf) x = RandUnit(s);
  for (auto& x : kf) x = RandUnit(s);
  for (auto& x : vf) x = RandUnit(s);

  std::vector<uint8_t> qb = Encode(qf, DType::kBF16, 1.0f);
  std::vector<uint8_t> kb = Encode(kf, DType::kF32, 1.0f);
  std::vector<uint8_t> vb = Encode(vf, DType::kBF16, 1.0f);
  std::vector<int32_t> btab = {0};
  std::vector<int32_t> slens = {static_cast<int32_t>(T)};
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};

  Tensor tq = Contig(qb.data(), DType::kBF16, {T, HQ, D});
  Tensor tk = Contig(kb.data(), DType::kF32, {NB, BS, HKV, D});
  Tensor tv = Contig(vb.data(), DType::kBF16, {NB, BS, HKV, D});
  Tensor tbt = Contig(btab.data(), DType::kI32, {1, 1});
  Tensor tsl = Contig(slens.data(), DType::kI32, {1});
  Tensor tqsl = Contig(qsl.data(), DType::kI32, {2});

  std::vector<float> got(static_cast<size_t>(T * HQ * D), -1.0f);
  Tensor tgot = Contig(got.data(), DType::kF32, {T, HQ, D});

  Queue qq = Q();
  PagedAttentionArgs args{0.35f, true};
  CHECK_THROWS_WITH_AS(vt::PagedAttention(qq, tgot, tq, tk, tv, tbt, tsl, tqsl, args),
                       doctest::Contains("k_cache/v_cache must share one float dtype"),
                       std::runtime_error);
}
