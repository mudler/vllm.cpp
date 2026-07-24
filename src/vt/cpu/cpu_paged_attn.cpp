// Ported from: vllm/v1/attention/backends/flash_attn.py @ e24d1b24
//   (FlashAttentionImpl.forward SEMANTICS: causal/non-causal GQA softmax over
//    the paged K/V, optional bottom-right-aligned window_size, softmax_scale =
//    self.scale, cu_seqlens_q = query_start_loc, seqused_k = seq_lens,
//    block_table = block_table_tensor). The cache READ is the NHD
//    layout FlashAttentionBackend::get_kv_cache_shape allocates
//    (num_blocks, 2, block_size, num_kv_heads, head_size), indexed by TENSOR
//    STRIDES — NOT cpu_attn.py's HND arithmetic (see the M1.6 Task-3 layout trap).
//
// Correctness-grade CPU reference: a clear per-(request, token, q-head) loop with
// a two-pass max-subtracted f32 softmax — algebraically identical to the dense
// M0.9 AttentionKernel, so on a single contiguous sequence the two agree. The
// current step's K/V are assumed already written into the cache
// (vt::ReshapeAndCache), so the read is entirely from the paged blocks.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cpu_threadpool.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

// f32 load shared with cpu_ops (redeclared locally to keep this TU standalone).
float LoadF32(const Tensor& t, int64_t elem_offset) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[elem_offset];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    default: VT_CHECK(false, "paged_attention LoadF32: unsupported dtype"); return 0.0f;
  }
}

void StoreF32(const Tensor& t, int64_t elem_offset, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[elem_offset] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[elem_offset] = F32ToBF16(v); break;
    default: VT_CHECK(false, "paged_attention StoreF32: unsupported dtype");
  }
}

// Paged causal GQA attention. For request r, query token at global index t and
// local index `local`, the absolute position is p = context + local where
// context = seq_lens[r] - query_len_r. Keys 0..p (causal) are intersected with
// [p-left,p+right] for local attention, then gathered from the paged cache:
// position j lives in block_table[r, j/block_size], offset j%block_size, at
// [block, offset, kv_head, :] — read via cache strides.
void PagedAttentionKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table,
                          const Tensor& seq_lens, const Tensor& query_start_loc,
                          const PagedAttentionArgs& args) {
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t qpk = hq / num_kv_heads;  // q-heads per kv-head (GQA ratio)
  const float scale = args.scale;
  // Attention logit soft-cap (vLLM Attention(logits_soft_cap=...), gemma2.py:202):
  // score' = cap * tanh(score / cap). 0.0 (default) leaves the plain scaled dot.
  const float softcap = args.logits_soft_cap;
  const int64_t window_left =
      args.window_size.has_value() ? args.window_size->left : -1;
  const int64_t window_right =
      args.window_size.has_value() ? args.window_size->right : -1;

  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  // Cache strides (unbind-slice aware): block / page(offset) / head / elem.
  const int64_t kc_blk = k_cache.stride[0], kc_pg = k_cache.stride[1], kc_hd = k_cache.stride[2];
  const int64_t vc_blk = v_cache.stride[0], vc_pg = v_cache.stride[1], vc_hd = v_cache.stride[2];

  // Flatten the (request, local-token) nest into the global query-token index so
  // the work is one embarrassingly-parallel axis: at c1 prefill num_reqs==1, so
  // the request loop alone is serial (kPagedAttention profiled at 10% of prefill,
  // single-threaded). Precompute each token's absolute position p, its request's
  // seqlen, and its request row (for the block table) once on the caller — cheap
  // O(total_q) — then chunk the token rows across the pool. Each output row
  // out[(t*hq+h)*d] is produced by exactly one thread with the same per-element
  // math and j-reduction order as the serial code: bit-identical by construction.
  std::vector<int32_t> tok_pos(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_slen(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_req(static_cast<size_t>(total_q));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q0 = qsl[r], q1 = qsl[r + 1];
    const int64_t query_len = q1 - q0;
    if (query_len <= 0) continue;
    const int64_t seqlen = slens[r];
    const int64_t context = seqlen - query_len;  // past positions before this chunk
    for (int64_t local = 0; local < query_len; ++local) {
      tok_pos[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(context + local);
      tok_slen[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(seqlen);
      tok_req[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(r);
    }
  }

  ParallelForRows(CurrentThreadpool(), total_q, [&](int64_t t0, int64_t t1) {
    std::vector<float> probs;
    std::vector<float> acc(static_cast<size_t>(d));
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t r = tok_req[static_cast<size_t>(t)];
      const int64_t p = tok_pos[static_cast<size_t>(t)];  // absolute position
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
        // Pass 1: scores + running max.
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t j = jmin; j <= jmax; ++j) {
          const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
          const int64_t off = j % block_size;
          const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
          float dot = 0.0f;
          for (int64_t e = 0; e < d; ++e)
            dot += LoadF32(query, qoff + e) * LoadF32(k_cache, kbase + e);
          dot *= scale;
          if (softcap > 0.0f) dot = softcap * std::tanh(dot / softcap);
          probs[static_cast<size_t>(j - jmin)] = dot;
          if (dot > m) m = dot;
        }
        // Pass 2: exp + denominator.
        float denom = 0.0f;
        for (int64_t j = jmin; j <= jmax; ++j) {
          const float e = std::exp(probs[static_cast<size_t>(j - jmin)] - m);
          probs[static_cast<size_t>(j - jmin)] = e;
          denom += e;
        }
        const float inv = 1.0f / denom;  // every valid decoder/encoder window has >= 1 key
        // Pass 3: weighted sum of V (f32 accumulation), stored at out's dtype.
        for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
        for (int64_t j = jmin; j <= jmax; ++j) {
          const float pw = probs[static_cast<size_t>(j - jmin)] * inv;
          const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
          const int64_t off = j % block_size;
          const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
          for (int64_t e = 0; e < d; ++e)
            acc[static_cast<size_t>(e)] += pw * LoadF32(v_cache, vbase + e);
        }
        for (int64_t e = 0; e < d; ++e) StoreF32(out, qoff + e, acc[static_cast<size_t>(e)]);
      }
    }
  });
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kPagedAttention, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
