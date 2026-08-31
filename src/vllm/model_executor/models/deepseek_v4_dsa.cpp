// DeepSeek-V4-Flash W3 primitives — host reference implementations.
// See deepseek_v4_dsa.h for the full port map (file:line on both sides).
#include "vllm/model_executor/models/deepseek_v4_dsa.h"

// For `HostBf16ToF32`, the inlined carried-tower widening `Wf` uses below.
// No cycle: `deepseek_v4_dsa.h` includes only <cstdint> and <vector>, and
// `deepseek_v4.h` does not include it back.
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_compressor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm::deepseek_v4 {

std::vector<float> DsaIndexerWeightFold(const std::vector<float>& weights_proj,
                                        int64_t num_tokens, int64_t index_n_heads,
                                        int64_t index_head_dim) {
  VT_CHECK(index_n_heads > 0 && index_head_dim > 0, "bad indexer dims");
  VT_CHECK(static_cast<int64_t>(weights_proj.size()) == num_tokens * index_n_heads,
           "weights_proj size mismatch");
  // softmax_scale = index_head_dim**-0.5 (attention.py:735);
  // head_scale    = index_n_heads**-0.5 (attention.py:843).
  const float softmax_scale =
      1.0f / std::sqrt(static_cast<float>(index_head_dim));
  const float head_scale = 1.0f / std::sqrt(static_cast<float>(index_n_heads));
  const float fold = softmax_scale * head_scale;
  std::vector<float> out(weights_proj.size());
  for (size_t i = 0; i < weights_proj.size(); ++i) out[i] = weights_proj[i] * fold;
  return out;
}

std::vector<float> DsaIndexerLogits(const std::vector<float>& q,
                                    const std::vector<float>& k,
                                    const std::vector<float>& folded_weights,
                                    const std::vector<int64_t>& win_start,
                                    const std::vector<int64_t>& win_end,
                                    int64_t num_tokens, int64_t num_keys,
                                    int64_t index_n_heads, int64_t index_head_dim) {
  const int64_t H = index_n_heads, D = index_head_dim;
  VT_CHECK(static_cast<int64_t>(q.size()) == num_tokens * H * D, "q size mismatch");
  VT_CHECK(static_cast<int64_t>(k.size()) == num_keys * D, "k size mismatch");
  VT_CHECK(static_cast<int64_t>(folded_weights.size()) == num_tokens * H,
           "folded_weights size mismatch");
  VT_CHECK(static_cast<int64_t>(win_start.size()) == num_tokens &&
               static_cast<int64_t>(win_end.size()) == num_tokens,
           "window arrays size mismatch");

  const float kNegInf = -std::numeric_limits<float>::infinity();
  std::vector<float> logits(static_cast<size_t>(num_tokens) * num_keys, kNegInf);

  for (int64_t t = 0; t < num_tokens; ++t) {
    const int64_t s0 = std::max<int64_t>(0, win_start[t]);
    const int64_t s1 = std::min<int64_t>(num_keys, win_end[t]);
    for (int64_t s = s0; s < s1; ++s) {
      float acc = 0.0f;
      for (int64_t h = 0; h < H; ++h) {
        // dot(q[t,h,:], k[s,:])
        float dot = 0.0f;
        const float* qp = &q[((t * H) + h) * D];
        const float* kp = &k[s * D];
        for (int64_t d = 0; d < D; ++d) dot += qp[d] * kp[d];
        // kv_scale == 1 in the fp32 reference; ReLU is load-bearing
        // (triton_fp8_mqa_logits.py:129).
        const float relu = dot > 0.0f ? dot : 0.0f;
        acc += folded_weights[t * H + h] * relu;
      }
      logits[t * num_keys + s] = acc;
    }
  }
  return logits;
}

std::vector<int64_t> DsaTopkSelect(const std::vector<float>& logits,
                                   const std::vector<int64_t>& win_start,
                                   const std::vector<int64_t>& win_end,
                                   int64_t num_tokens, int64_t num_keys,
                                   int64_t topk) {
  VT_CHECK(topk > 0, "topk must be positive");
  VT_CHECK(static_cast<int64_t>(logits.size()) == num_tokens * num_keys,
           "logits size mismatch");
  std::vector<int64_t> out(static_cast<size_t>(num_tokens) * topk, -1);

  for (int64_t t = 0; t < num_tokens; ++t) {
    const int64_t s0 = std::max<int64_t>(0, win_start[t]);
    const int64_t s1 = std::min<int64_t>(num_keys, win_end[t]);
    const int64_t n = std::max<int64_t>(0, s1 - s0);
    int64_t* dst = &out[t * topk];

    if (n <= topk) {
      // Short-context: EVERY candidate selected, ascending key order
      // (attention.py:70-86 _fill_short_context_topk_indices / :813-831).
      int64_t w = 0;
      for (int64_t s = s0; s < s1; ++s) dst[w++] = s;
      continue;  // remaining slots stay -1
    }

    // Full top-k: pick the `topk` keys with the largest logits; ties resolved
    // toward the SMALLER key index (stable). Sort a candidate index list by
    // (logit desc, index asc), take the first `topk`, then emit them in
    // ASCENDING key order (top_k_per_row writes indices, order-agnostic for the
    // downstream gather; ascending keeps the reference deterministic).
    std::vector<int64_t> cand(static_cast<size_t>(n));
    std::iota(cand.begin(), cand.end(), s0);
    std::stable_sort(cand.begin(), cand.end(), [&](int64_t a, int64_t b) {
      const float la = logits[t * num_keys + a];
      const float lb = logits[t * num_keys + b];
      if (la != lb) return la > lb;  // larger logit first
      return a < b;                  // tie -> smaller index
    });
    cand.resize(static_cast<size_t>(topk));
    std::sort(cand.begin(), cand.end());  // ascending key order
    for (int64_t i = 0; i < topk; ++i) dst[i] = cand[static_cast<size_t>(i)];
  }
  return out;
}

std::vector<float> SoftmaxWithSink(const std::vector<float>& scores, float sink) {
  const int64_t n = static_cast<int64_t>(scores.size());
  std::vector<float> prob(static_cast<size_t>(n), 0.0f);
  if (n == 0) return prob;

  float m = sink;
  for (float s : scores) m = std::max(m, s);
  // A fully -inf row (no keys finite AND sink -inf) would give 0/0; guard it.
  if (m == -std::numeric_limits<float>::infinity()) return prob;

  float denom = std::exp(sink - m);  // sink contributes to the denominator only
  for (int64_t j = 0; j < n; ++j) {
    const float e = std::exp(scores[static_cast<size_t>(j)] - m);
    prob[static_cast<size_t>(j)] = e;
    denom += e;
  }
  for (int64_t j = 0; j < n; ++j) prob[static_cast<size_t>(j)] /= denom;
  return prob;
}

// `Wf` is the ONE place the weight dtype is widened. On `float` it is the identity,
// so the f32 arm compiles to exactly what it compiled to before W1d; on `uint16_t`
// it is `HostBf16ToF32`, the INLINE header helper (`deepseek_v4.h`) that performs
// `vt::BF16ToF32`'s bit operation without the out-of-line call this build cannot
// inline away -- see W1d-5, and the exhaustive 65536-pattern agreement case.
inline float Wf(float w) { return w; }
inline float Wf(uint16_t w) { return HostBf16ToF32(w); }

// One body, two weight dtypes (W1d, #2186). `Wf` widens a bf16 bit pattern the
// same way `vt::BF16ToF32` does and is the identity on f32, so the f32 arm is
// BYTE-FOR-BYTE the pre-W1d function and the bf16 arm differs only in the weight
// elements themselves. Accumulators and reduction order are shared.
template <typename W>
std::vector<float> GroupedOutputLora(const std::vector<float>& o,
                                     const std::vector<W>& wo_a,
                                     const std::vector<W>& wo_b,
                                     int64_t num_tokens, int64_t n_heads,
                                     int64_t head_dim, int64_t n_groups,
                                     int64_t o_lora_rank, int64_t hidden_size) {
  VT_CHECK(n_groups > 0 && n_heads % n_groups == 0,
           "n_heads must be divisible by n_groups");
  const int64_t in_per_group = n_heads * head_dim / n_groups;  // heads_per_group*head_dim
  const int64_t z_dim = n_groups * o_lora_rank;
  VT_CHECK(static_cast<int64_t>(o.size()) == num_tokens * n_heads * head_dim,
           "o size mismatch");
  VT_CHECK(static_cast<int64_t>(wo_a.size()) == n_groups * o_lora_rank * in_per_group,
           "wo_a size mismatch");
  VT_CHECK(static_cast<int64_t>(wo_b.size()) == hidden_size * z_dim,
           "wo_b size mismatch");

  std::vector<float> out(static_cast<size_t>(num_tokens) * hidden_size, 0.0f);
  std::vector<float> z(static_cast<size_t>(z_dim));

  for (int64_t t = 0; t < num_tokens; ++t) {
    // o for token t is [n_heads, head_dim] contiguous == [n_groups, in_per_group]
    // contiguous (heads_per_group heads packed per group).
    const float* o_t = &o[t * n_heads * head_dim];
    // z[g, d] = sum_r wo_a[g, d, r] * o_group[g, r]  (per-group einsum "bhr,hdr->bhd")
    for (int64_t g = 0; g < n_groups; ++g) {
      const float* o_g = o_t + g * in_per_group;
      const W* wa_g = &wo_a[g * o_lora_rank * in_per_group];
      float* z_g = &z[g * o_lora_rank];
      for (int64_t d = 0; d < o_lora_rank; ++d) {
        float acc = 0.0f;
        const W* wa_gd = wa_g + d * in_per_group;
        for (int64_t r = 0; r < in_per_group; ++r) acc += Wf(wa_gd[r]) * o_g[r];
        z_g[d] = acc;
      }
    }
    // out[t, :] = wo_b @ z    ( [hidden_size x z_dim] @ [z_dim] )
    float* out_t = &out[t * hidden_size];
    for (int64_t h = 0; h < hidden_size; ++h) {
      float acc = 0.0f;
      const W* wb_h = &wo_b[h * z_dim];
      for (int64_t c = 0; c < z_dim; ++c) acc += Wf(wb_h[c]) * z[static_cast<size_t>(c)];
      out_t[h] = acc;
    }
  }
  return out;
}

// Both arms are instantiated HERE rather than left implicit, so a caller that
// needs a third weight dtype fails to link instead of silently instantiating a
// body whose numerics nobody reviewed.
template std::vector<float> GroupedOutputLora<float>(const std::vector<float>&,
                                                     const std::vector<float>&,
                                                     const std::vector<float>&, int64_t,
                                                     int64_t, int64_t, int64_t, int64_t,
                                                     int64_t);
template std::vector<float> GroupedOutputLora<uint16_t>(const std::vector<float>&,
                                                        const std::vector<uint16_t>&,
                                                        const std::vector<uint16_t>&,
                                                        int64_t, int64_t, int64_t, int64_t,
                                                        int64_t, int64_t);


// KV-DSV4-MULTICACHE W5 (#2323). See the header for the batch<-T mapping.
std::vector<float> PagedCausalMlaAttention(vt::Queue& queue, const std::vector<float>& q,
                                           vt::Tensor& kv_cache, int64_t num_blocks,
                                           int64_t block_size, int64_t num_tokens,
                                           int64_t num_heads, int64_t head_dim,
                                           int64_t kv_base, const std::vector<float>& sink,
                                           float scale, bool no_sink,
                                           int64_t sliding_window,
                                           std::vector<float>* out_lse) {
  VT_CHECK(num_tokens > 0 && num_heads > 0 && head_dim > 0,
           "PagedCausalMlaAttention: num_tokens/num_heads/head_dim must be > 0");
  VT_CHECK(static_cast<int64_t>(q.size()) == num_tokens * num_heads * head_dim,
           "PagedCausalMlaAttention: q must be [T, num_heads, head_dim]");
  VT_CHECK(static_cast<int64_t>(sink.size()) == num_heads,
           "PagedCausalMlaAttention: sink must be [num_heads]");

  // `seq_lens[t] = kv_base + t + 1` IS the causal mask: query t's global position
  // is `kv_base + t` and it may see `[0, kv_base + t]`, which is that many keys.
  std::vector<int32_t> seq_lens(static_cast<size_t>(num_tokens));
  for (int64_t t = 0; t < num_tokens; ++t)
    seq_lens[static_cast<size_t>(t)] = static_cast<int32_t>(kv_base + t + 1);

  // Every query row reads the SAME pages, so the table is one row repeated. The
  // op indexes it per batch row, so this cannot be a single shared row.
  std::vector<int32_t> block_table(static_cast<size_t>(num_tokens * num_blocks));
  for (int64_t b = 0; b < num_tokens; ++b)
    for (int64_t i = 0; i < num_blocks; ++i)
      block_table[static_cast<size_t>(b * num_blocks + i)] = static_cast<int32_t>(i);

  // The `kNoAttnSink` miswire feeds -inf, which adds nothing to the denominator
  // and is therefore EXACTLY "no sink" rather than an approximation of it.
  std::vector<float> sink_v = sink;
  if (no_sink)
    for (float& sv : sink_v) sv = -std::numeric_limits<float>::infinity();

  std::vector<float> out(static_cast<size_t>(num_tokens) * num_heads * head_dim, 0.0f);
  const vt::Device dev = queue.device;
  vt::Tensor t_out =
      vt::Tensor::Contiguous(out.data(), vt::DType::kF32, dev, {num_tokens, num_heads, head_dim});
  vt::Tensor t_q = vt::Tensor::Contiguous(const_cast<float*>(q.data()), vt::DType::kF32, dev,
                                          {num_tokens, num_heads, head_dim});
  vt::Tensor t_bt = vt::Tensor::Contiguous(block_table.data(), vt::DType::kI32, dev,
                                           {num_tokens, num_blocks});
  vt::Tensor t_sl =
      vt::Tensor::Contiguous(seq_lens.data(), vt::DType::kI32, dev, {num_tokens});
  vt::Tensor t_sink =
      vt::Tensor::Contiguous(sink_v.data(), vt::DType::kF32, dev, {num_heads});
  (void)block_size;

  vt::MlaDecodeAttentionArgs args;
  args.scale = scale;
  args.attn_sink = &t_sink;
  // The op's convention is `left == sliding_window - 1` (an INCLUSIVE distance),
  // and `right == 0` because a decode query is the last position of its own
  // sequence. Absent (0) the kernel keeps its full-prefix loop byte-identically.
  if (sliding_window > 0)
    args.window_size =
        vt::AttentionWindow{static_cast<int32_t>(sliding_window - 1), 0};
  vt::Tensor t_lse;
  if (out_lse != nullptr) {
    out_lse->assign(static_cast<size_t>(num_tokens) * static_cast<size_t>(num_heads), 0.0f);
    t_lse = vt::Tensor::Contiguous(out_lse->data(), vt::DType::kF32, dev,
                                   {num_tokens, num_heads});
  }
  vt::MlaDecodeAttention(queue, t_out, out_lse != nullptr ? &t_lse : nullptr, t_q, kv_cache,
                         t_bt, t_sl, args);
  return out;
}


// MODEL-DSV4-DSA-COMPOSE W1 (#2286). See the header for why no sink appears here.
std::vector<float> MergeWindowAndCompressed(vt::Queue& queue,
                                            const std::vector<float>& window_out,
                                            const std::vector<float>& window_lse,
                                            const std::vector<float>& q,
                                            const std::vector<float>& comp_rows,
                                            int64_t n_rows, int64_t num_tokens,
                                            int64_t num_heads, int64_t head_dim,
                                            float scale) {
  VT_CHECK(n_rows > 0, "MergeWindowAndCompressed: no compressed rows");
  VT_CHECK(static_cast<int64_t>(comp_rows.size()) == n_rows * head_dim,
           "MergeWindowAndCompressed: comp_rows must be [n_rows, head_dim]");
  const int64_t TH = num_tokens * num_heads;
  VT_CHECK(static_cast<int64_t>(window_out.size()) == TH * head_dim,
           "MergeWindowAndCompressed: window_out must be [T, num_heads, head_dim]");
  VT_CHECK(static_cast<int64_t>(window_lse.size()) == TH,
           "MergeWindowAndCompressed: window_lse must be [T * num_heads]");

  // The compressed rows are their own paged cache: one block wide enough to hold
  // them all, every row visible to every query.
  const int64_t block_size = 16;
  const int64_t num_blocks = (n_rows + block_size - 1) / block_size;
  std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * head_dim), 0.0f);
  std::copy(comp_rows.begin(), comp_rows.end(), cache.begin());

  const vt::Device dev = queue.device;
  std::vector<int32_t> bt(static_cast<size_t>(num_tokens * num_blocks));
  for (int64_t b = 0; b < num_tokens; ++b)
    for (int64_t i = 0; i < num_blocks; ++i)
      bt[static_cast<size_t>(b * num_blocks + i)] = static_cast<int32_t>(i);
  // EVERY query sees EVERY compressed row: a closed window is history, so there
  // is no causal bound to apply among them.
  std::vector<int32_t> sl(static_cast<size_t>(num_tokens), static_cast<int32_t>(n_rows));
  std::vector<float> comp_out(static_cast<size_t>(TH) * head_dim, 0.0f);
  std::vector<float> comp_lse(static_cast<size_t>(TH), 0.0f);

  vt::Tensor t_c = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32, dev,
                                          {num_blocks, block_size, head_dim});
  vt::Tensor t_q = vt::Tensor::Contiguous(const_cast<float*>(q.data()), vt::DType::kF32, dev,
                                          {num_tokens, num_heads, head_dim});
  vt::Tensor t_bt = vt::Tensor::Contiguous(bt.data(), vt::DType::kI32, dev,
                                           {num_tokens, num_blocks});
  vt::Tensor t_sl = vt::Tensor::Contiguous(sl.data(), vt::DType::kI32, dev, {num_tokens});
  vt::Tensor t_co = vt::Tensor::Contiguous(comp_out.data(), vt::DType::kF32, dev,
                                           {num_tokens, num_heads, head_dim});
  vt::Tensor t_cl = vt::Tensor::Contiguous(comp_lse.data(), vt::DType::kF32, dev,
                                           {num_tokens, num_heads});
  vt::MlaDecodeAttentionArgs args;
  args.scale = scale;
  // NO `attn_sink` — see the header. The window pass owns it.
  vt::MlaDecodeAttention(queue, t_co, &t_cl, t_q, t_c, t_bt, t_sl, args);

  std::vector<float> merged(static_cast<size_t>(TH) * head_dim, 0.0f);
  vt::Tensor t_out = vt::Tensor::Contiguous(merged.data(), vt::DType::kF32, dev,
                                            {num_tokens, num_heads, head_dim});
  vt::Tensor t_wo = vt::Tensor::Contiguous(const_cast<float*>(window_out.data()),
                                           vt::DType::kF32, dev,
                                           {num_tokens, num_heads, head_dim});
  // `MergeAttnStates` wants the LSEs as `[H, num_tokens]`; both buffers hold
  // `T * H` contiguous f32, so this is a reshape rather than a transpose only
  // because a decode step carries one token per row -- asserted, not assumed.
  VT_CHECK(num_tokens == 1 || num_heads == 1,
           "MergeWindowAndCompressed: the LSE layouts coincide only when T or H is "
           "1; a general step needs a transpose here");
  vt::Tensor t_wl = vt::Tensor::Contiguous(const_cast<float*>(window_lse.data()),
                                           vt::DType::kF32, dev, {num_heads, num_tokens});
  vt::Tensor t_cl2 = vt::Tensor::Contiguous(comp_lse.data(), vt::DType::kF32, dev,
                                            {num_heads, num_tokens});
  vt::MergeAttnStates(queue, t_out, nullptr, t_wo, t_wl, t_co, t_cl2, -1);
  return merged;
}

std::vector<float> CompressorLayerStep(
    vt::Queue& queue, const std::vector<float>& x, const std::vector<float>& kv,
    const std::vector<float>& q, const std::vector<float>& comp_wgate,
    const std::vector<float>& comp_ape, const std::vector<float>& comp_norm_weight,
    const std::vector<float>& attn_sink, vt::Tensor& window_cache,
    int64_t num_blocks, int64_t block_size, std::vector<float>* state_kv,
    std::vector<float>* state_score, std::vector<float>* comp_rows,
    const std::vector<int64_t>& positions, int64_t kv_base, int64_t num_tokens,
    int64_t num_heads, int64_t hidden, int64_t head_dim, int64_t compress_ratio,
    int64_t sliding_window, float eps, float scale) {
  // W3 (#2286): both shapes now. `coff = 1 + (compress_ratio == 4)`
  // (`compressor.py:247-248`); at 2 the projections are doubled and a window
  // position's ROLE picks which half it reads.
  VT_CHECK(compress_ratio == 4 || compress_ratio == 128,
           "deepseek-v4 compressor step: upstream emits compress_ratio 4 or 128 "
           "only (sparse_swa.py:44-55)");
  // Read from the TENSOR, not derived from the ratio: the synthetic suites carry a
  // collapsed `coff == 1` shape on `cr == 4` layers, which upstream cannot emit
  // but this tree has always accepted.
  const int64_t coff =
      (hidden > 0 && head_dim > 0 &&
       static_cast<int64_t>(comp_wgate.size()) == 2 * head_dim * hidden)
          ? 2
          : 1;
  VT_CHECK(state_kv != nullptr && state_score != nullptr && comp_rows != nullptr,
           "deepseek-v4 compressor step: the state is CARRIED, not owned here");
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * hidden,
           "deepseek-v4 compressor step: x must be [num_tokens, hidden]");
  VT_CHECK(static_cast<int64_t>(kv.size()) == num_tokens * coff * head_dim,
           "deepseek-v4 compressor step: kv must be [num_tokens, coff*head_dim]");
  VT_CHECK(static_cast<int64_t>(comp_wgate.size()) == coff * head_dim * hidden,
           "deepseek-v4 compressor step: comp_wgate is [coff*head_dim, hidden]");
  VT_CHECK(static_cast<int64_t>(positions.size()) == num_tokens,
           "deepseek-v4 compressor step: one position per token");

  // THE PREFIX-CACHE GUARD. The compressor pools tokens it has SEEN, so its state
  // must have seen exactly the `kv_base` tokens this step resumes after. A
  // prefix-cache hit skips recomputing cached tokens, and those tokens are the
  // ones whose rows this state would have accumulated -- so after a hit it holds
  // FEWER rows than the position implies, and the layer would attend a compressed
  // history with holes in it.
  //
  // That failure is invisible: the output stays finite and plausible, and it only
  // appears on cache hits. Refusing is never wrong here, only limiting, so the
  // mismatch is refused by name rather than resolved by a policy this row does not
  // own (`## Owed`).
  const int64_t seen = static_cast<int64_t>(state_kv->size()) / head_dim;
  VT_CHECK(seen == kv_base,
           "deepseek-v4 compressor: the carried state has seen " +
               std::to_string(seen) + " tokens but this step resumes at kv_base " +
               std::to_string(kv_base) +
               ". A prefix-cache hit skipped tokens this state needed, and the "
               "compressed history would have holes in it. Refusing "
               "(MODEL-DSV4-DSA-COMPOSE, #2286)");

  // 1. The pool score this layer selects with.
  const int64_t cw = coff * head_dim;
  std::vector<float> score(static_cast<size_t>(num_tokens) * cw, 0.0f);
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t d = 0; d < cw; ++d) {
      double acc = 0.0;
      const float* w = &comp_wgate[static_cast<size_t>(d * hidden)];
      const float* xt = &x[static_cast<size_t>(t * hidden)];
      for (int64_t h = 0; h < hidden; ++h) acc += static_cast<double>(w[h]) * xt[h];
      score[static_cast<size_t>(t * cw + d)] = static_cast<float>(acc);
    }
  }

  // 2-3. Drive the state machine and keep whatever closed this step.
  const std::vector<float> emitted =
      CompressorStepCycle(state_kv, state_score, kv, score, comp_ape, positions,
                          comp_norm_weight, eps, compress_ratio, head_dim, coff);
  comp_rows->insert(comp_rows->end(), emitted.begin(), emitted.end());

  // 4. The window pass carries the sink and keeps its LSE.
  std::vector<float> win_lse;
  const std::vector<float> win_out = PagedCausalMlaAttention(
      queue, q, window_cache, num_blocks, block_size, num_tokens, num_heads, head_dim,
      kv_base, attn_sink, scale, /*no_sink=*/false, sliding_window, &win_lse);

  // 5. Nothing closed yet => the window IS the answer. Merging an empty second
  //    contributor would divide by an empty denominator rather than be a no-op.
  const int64_t n_rows = static_cast<int64_t>(comp_rows->size()) / head_dim;
  if (n_rows == 0) return win_out;

  return MergeWindowAndCompressed(queue, win_out, win_lse, q, *comp_rows, n_rows,
                                  num_tokens, num_heads, head_dim, scale);
}

}  // namespace vllm::deepseek_v4
