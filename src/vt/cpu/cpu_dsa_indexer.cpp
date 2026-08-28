// The DSA "Lightning Indexer" selection pair (CPU) — dots3-note W4b-3c, #699.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ bc2d63e650) ─────────
//   OURS                  <-  UPSTREAM
//   DsaIndexerLogits      <-  vllm/v1/attention/ops/triton_fp8_mqa_logits.py
//                             :120-156 `_fp8_mqa_logits_kernel`'s inner tile —
//                             `tl.dot(q_block, kv_block, input_precision="ieee")`
//                             (:125,:146), `* kv_scales` (:127,:148), the ReLU
//                             `tl.maximum(scores, 0.0)` (:129,:150),
//                             `* w_block` (:130,:151), `tl.sum(_, axis=0)`
//                             (:132,:153) and the masked store whose `in_window`
//                             predicate leaves everything else at the row's
//                             pre-filled -inf (:155-156).
//   the WEIGHT FOLD       <-  vllm/model_executor/models/deepseek_v2.py:840
//                             `weights = weights * q_scale * self.softmax_scale
//                              * self.n_head_scale`, and the fused kernel's
//                             identical `weight * q_scale * softmax_scale *
//                             head_scale`
//                             (vllm/model_executor/layers/sparse_attn_indexer.py
//                              :203-207). `softmax_scale = head_dim**-0.5`
//                             (deepseek_v2.py:709), `n_head_scale =
//                             n_head**-0.5` (:742).
//   DsaTopkSelect         <-  sparse_attn_indexer.py:509 `ops.top_k_per_row_prefill`
//                             plus the short-context all-select.
//
// ─── RELATION TO THE HOST REFERENCE ─────────────────────────────────────────
// `vllm::deepseek_v4::DsaIndexerLogits` / `DsaTopkSelect`
// (include/vllm/model_executor/models/deepseek_v4_dsa.h) are W3's
// `std::vector<float>` transcription of the SAME upstream lines. They stay, and
// they are this op's ORACLE in tests/vt/test_ops_dsa_indexer.cpp. Nothing about
// the maths is re-decided here; what these ops add is a `vt::Tensor` surface a
// device path can reach, and a CUDA sibling.
//
// ─── WHAT IS NOT HERE, deliberately ─────────────────────────────────────────
// `k_norm` and the indexer's LEADING-slice rope are NOT part of this op family.
// Upstream's `k = self.k_norm(k)` is `LayerNorm(head_dim, eps=1e-6)`
// (deepseek_v2.py:708) — mean-subtracting, weight AND bias — which is
// `vt::LayerNorm`; and `q_pe, q_nope = torch.split(q, [rope_dim, head_dim -
// rope_dim])` followed by `rotary_emb(positions, q_pe, k_pe)`
// (deepseek_v2.py:804-817) is `vt::RopeFromCache` with `rotary_dim < head_dim`
// over a strided leading-slice view. Both already exist and both are already
// gated; writing a second copy of either inside this file would be the parallel
// path AGENTS.md forbids.
//
// ─── PRECISION ──────────────────────────────────────────────────────────────
// The dot, the ReLU, the fold and the head sum all accumulate in f32 whatever
// the operand dtype, which is what `input_precision="ieee"` on a bf16 `tl.dot`
// gives, and the logits are STORED f32 as upstream stores them. A key whose
// every head dots negative therefore scores EXACTLY 0.0 in both arms, which is
// why exact ties are ordinary here rather than exotic and why DsaTopkSelect's
// tie rule is pinned rather than left to a sort's whim.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

float LoadFloat(const void* base, DType dt, int64_t i) {
  switch (dt) {
    case DType::kF32: return static_cast<const float*>(base)[i];
    case DType::kBF16: return BF16ToF32(static_cast<const uint16_t*>(base)[i]);
    case DType::kF16: return F16ToF32(static_cast<const uint16_t*>(base)[i]);
    default: VT_CHECK(false, "cpu dsa_indexer: unsupported float dtype"); return 0.0f;
  }
}

void DsaIndexerLogitsKernel(Queue&, Tensor& logits, const Tensor& q, const Tensor& k,
                            const Tensor& weights, const Tensor& win_start,
                            const Tensor& win_end, const DsaIndexerLogitsArgs& args) {
  const int64_t T = q.shape[0], H = q.shape[1], D = q.shape[2];
  const int64_t S = k.shape[0];
  const int32_t* ws = win_start.Ptr<int32_t>();
  const int32_t* we = win_end.Ptr<int32_t>();
  float* out = logits.Ptr<float>();
  const float* qs = args.q_scale != nullptr ? args.q_scale->Ptr<float>() : nullptr;
  const int64_t qs_s0 = qs != nullptr ? args.q_scale->stride[0] : 0;
  // The two GLOBAL scalars of the fold. They multiply every logit of every row
  // by one positive constant, so they cannot move an argmax — see
  // DsaIndexerLogitsArgs on why their mutations read green definitionally.
  const float gfold = args.softmax_scale * args.n_head_scale;
  const float kNegInf = -std::numeric_limits<float>::infinity();

  std::vector<float> fold(static_cast<size_t>(H));
  for (int64_t t = 0; t < T; ++t) {
    // `tl.store(logits_ptrs, scores, mask=in_window)` (:156) writes only the
    // in-window columns; everything else keeps the row's pre-filled -inf, so a
    // plain top-k downstream needs no second mask.
    float* row = out + t * logits.stride[0];
    for (int64_t s = 0; s < S; ++s) row[s] = kNegInf;

    for (int64_t h = 0; h < H; ++h) {
      const float w = LoadFloat(weights.data, weights.dtype, t * weights.stride[0] + h);
      // `weights * q_scale * softmax_scale * n_head_scale` (deepseek_v2.py:840).
      // A null q_scale is upstream's unquantized arm, where it is exactly 1.
      const float per_head = qs != nullptr ? qs[t * qs_s0 + h] : 1.0f;
      fold[static_cast<size_t>(h)] = w * per_head * gfold;
    }

    const int64_t s0 = std::max<int64_t>(0, ws[t]);
    const int64_t s1 = std::min<int64_t>(S, we[t]);
    for (int64_t s = s0; s < s1; ++s) {
      float acc = 0.0f;
      for (int64_t h = 0; h < H; ++h) {
        const int64_t q_off = t * q.stride[0] + h * q.stride[1];
        const int64_t k_off = s * k.stride[0];
        float dot = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
          dot += LoadFloat(q.data, q.dtype, q_off + d) * LoadFloat(k.data, k.dtype, k_off + d);
        }
        // `tl.maximum(scores, 0.0)` (:129/:150). The ReLU is the load-bearing
        // nuance: without it a strongly ANTI-correlated head would pull the
        // logit down instead of contributing nothing.
        acc += fold[static_cast<size_t>(h)] * (dot > 0.0f ? dot : 0.0f);
      }
      row[s] = acc;
    }
  }
}

void DsaTopkSelectKernel(Queue&, Tensor& indices, Tensor& counts, const Tensor& logits,
                         const Tensor& win_start, const Tensor& win_end) {
  const int64_t T = logits.shape[0], S = logits.shape[1];
  const int64_t topk = indices.shape[1];
  const int32_t* ws = win_start.Ptr<int32_t>();
  const int32_t* we = win_end.Ptr<int32_t>();
  const float* lg = logits.Ptr<float>();
  int32_t* idx = indices.Ptr<int32_t>();
  int32_t* cnt = counts.Ptr<int32_t>();

  std::vector<int64_t> cand;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t s0 = std::max<int64_t>(0, ws[t]);
    const int64_t s1 = std::min<int64_t>(S, we[t]);
    const int64_t n = std::max<int64_t>(0, s1 - s0);
    int32_t* dst = idx + t * indices.stride[0];
    // `-1` is the "no token" sentinel the topk buffer is pre-filled with
    // (sparse_attn_indexer.py:431-432; `:426-430` is the comment above it).
    for (int64_t i = 0; i < topk; ++i) dst[i] = -1;

    if (n <= topk) {
      // SHORT CONTEXT: every candidate is selected, in ascending key order.
      // This is the branch that makes a sparse layer's answer IDENTICAL to
      // dense attention while `context + query <= index_topk`, which is exactly
      // the regime upstream keeps its dense-MHA prefill for
      // (`use_dense_mha = prefill_max_seq_len <= self.topk_tokens`,
      //  sparse_mla_attention.py:296-299).
      int64_t w = 0;
      for (int64_t s = s0; s < s1; ++s) dst[w++] = static_cast<int32_t>(s);
      cnt[t] = static_cast<int32_t>(n);
      continue;
    }

    // FULL TOP-K. Ties break toward the SMALLER key index, and the emission is
    // ASCENDING — both are load-bearing rather than cosmetic. The tie rule is
    // needed because the ReLU makes an exact 0.0 an ordinary logit value; the
    // ascending emission is what makes a FULL selection reproduce the dense
    // answer bit for bit in vt::MlaDecodeAttention, whose online softmax
    // reduces in visit order.
    const float* row = lg + t * logits.stride[0];
    cand.resize(static_cast<size_t>(n));
    std::iota(cand.begin(), cand.end(), s0);
    std::stable_sort(cand.begin(), cand.end(), [row](int64_t a, int64_t b) {
      const float la = row[a];
      const float lb = row[b];
      if (la != lb) return la > lb;
      return a < b;
    });
    cand.resize(static_cast<size_t>(topk));
    std::sort(cand.begin(), cand.end());
    for (int64_t i = 0; i < topk; ++i) dst[i] = static_cast<int32_t>(cand[static_cast<size_t>(i)]);
    cnt[t] = static_cast<int32_t>(topk);
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kDsaIndexerLogits, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<DsaIndexerLogitsFn>(&DsaIndexerLogitsKernel)));
    RegisterOp(OpId::kDsaTopkSelect, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<DsaTopkSelectFn>(&DsaTopkSelectKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
