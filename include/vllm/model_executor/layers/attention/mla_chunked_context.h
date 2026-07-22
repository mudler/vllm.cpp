// MLA chunked-context prefill — the workspace-bounded loop over previously
// cached context (MLA campaign W5).
//
// Ported from vllm/model_executor/layers/attention/mla_attention.py @ pin
// e24d1b24, with `file:line` on both sides:
//
//   OURS                              <-  UPSTREAM
//   DetermineChunkedPrefillWorkspaceSize
//                                     <-  :1422-1451
//                                         `MLACommonMetadataBuilder.
//                                          determine_chunked_prefill_workspace_size`
//   MlaChunkedContextMetadata         <-  :1249-1260
//                                         `MLACommonPrefillMetadata.
//                                          ChunkedContextMetadata`
//   BuildMlaChunkedContext            <-  :1667-1745 + :1837-1849 (the
//                                         non-DCP branch of
//                                         `MLACommonMetadataBuilder.build`)
//   ComputeMlaPrefillContext          <-  :2094-2199 `_compute_prefill_context`
//   ForwardMlaPrefillMha              <-  :2344-2425 `forward_mha`
//
// WHY CHUNKING EXISTS AT ALL — the arithmetic upstream spells out inline at
// :1435-1441: at 64k context tokens the LATENT workspace is 2*576*65536 = 144 MB,
// but the UP-PROJECTED context would be 2*(192*128)*65536 = 3 GB. The
// compute-friendly (materialized-MHA) prefill form cannot hold that, so the
// context is processed in workspace-bounded chunks whose partial attention
// results are merged by their log-sum-exps. A port that skips this OOMs on long
// contexts rather than merely running slow.
//
// THE PART MOST LIKELY TO HIDE BUGS is the chunk BOUNDARY, and there are three
// distinct hazards, all handled here and all covered by the gate:
//   1. `max_context_chunk` is rounded DOWN to a multiple of the page size
//      (:1687-1690) because `gather_and_maybe_dequant_cache` indexes
//      `(seq_starts[b] + within_chunk) / block_size` into the block table; an
//      unaligned start silently gathers the wrong pages.
//   2. A request whose context is SHORTER than the chunk grid has ZERO tokens in
//      the later chunks (`chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)`,
//      :1699). Its attention result for that chunk is an empty softmax whose LSE
//      is -inf, and merging two -inf partials is 0/0 — which is exactly the case
//      merge_attn_states.cu:100-106 guards, and why vt::MergeAttnStates ports
//      that guard verbatim.
//   3. The merge is a PING-PONG over two buffers (:2196-2197
//      `output, merge_output = merge_output, output`), so chunk k's result must
//      never be read from the buffer chunk k+1 is writing.
#ifndef VLLM_MODEL_EXECUTOR_LAYERS_ATTENTION_MLA_CHUNKED_CONTEXT_H_
#define VLLM_MODEL_EXECUTOR_LAYERS_ATTENTION_MLA_CHUNKED_CONTEXT_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include "vt/ops.h"

namespace vllm::mla {

// `determine_chunked_prefill_workspace_size` (mla_attention.py:1422-1451),
// verbatim including the two clamps and their reasons:
//   min(max(8 * max_model_len, 4 * max_num_seqs * block_size), 64 * 1024)
//   then max(..., max_num_seqs * block_size)   # >= 1 page per request
inline int64_t DetermineChunkedPrefillWorkspaceSize(int64_t max_model_len,
                                                    int64_t max_num_seqs, int64_t block_size) {
  int64_t size = std::min<int64_t>(
      std::max<int64_t>(8 * max_model_len, 4 * max_num_seqs * block_size), 64 * 1024);
  return std::max<int64_t>(size, max_num_seqs * block_size);
}

// The host side of `MLACommonPrefillMetadata.ChunkedContextMetadata`
// (mla_attention.py:1249-1260). Row-major `[num_chunks, ...]` exactly as
// upstream builds the tensors.
struct MlaChunkedContextMetadata {
  int32_t num_chunks = 0;
  int32_t num_prefills = 0;
  int32_t max_context_chunk = 0;
  // [num_chunks, num_prefills] — the chunk's start offset within each request's
  // CONTEXT (upstream `chunk_starts` / `ChunkedContextMetadata.starts`).
  std::vector<int32_t> starts;
  // [num_chunks, num_prefills] — tokens contributed by each request to a chunk.
  std::vector<int32_t> chunk_seq_lens;
  // [num_chunks, num_prefills + 1] — the per-chunk varlen cu_seqlens_k.
  std::vector<int32_t> cu_seq_lens;
  // [num_chunks] — sum / max over the chunk's per-request lengths.
  std::vector<int32_t> seq_tot;
  std::vector<int32_t> max_seq_lens;
  std::vector<int32_t> chunk_total_token;
  // [num_chunks, max_token_num_over_chunk] — token index -> request index.
  std::vector<int32_t> token_to_seq;
  int32_t max_token_num_over_chunk = 0;
  // `prefill_tokens_with_context` (:1806-1810): the number of leading prefill
  // QUERY tokens that belong to requests WITH context. Requests are ordered so
  // that with-context ones come first, so every token at index >= this value
  // takes the suffix (new-tokens) result verbatim in the final merge.
  int32_t prefill_tokens_with_context = 0;
};

// `MLACommonMetadataBuilder.build`, the `max_context_len_cpu > 0` branch
// (mla_attention.py:1667-1745) plus the non-DCP metadata assembly (:1837-1849).
//
//   context_lens              per prefill request: seq_len - query_len (:1663)
//   prefill_query_start_loc   [num_prefills+1] cumulative QUERY offsets, used
//                             ONLY for `prefill_tokens_with_context` (:1806-1810)
//   workspace_size            `chunked_prefill_workspace_size`
//   page_size                 the KV block size
//
// Returns `num_chunks == 0` when no request has context — upstream's
// `chunked_context_metadata is None` case, in which `forward_mha` skips the
// whole loop and the new-tokens result IS the answer (:2394, :2421-2425).
inline MlaChunkedContextMetadata BuildMlaChunkedContext(
    const std::vector<int32_t>& context_lens,
    const std::vector<int32_t>& prefill_query_start_loc, int64_t workspace_size,
    int64_t page_size) {
  MlaChunkedContextMetadata m;
  const int32_t num_prefills = static_cast<int32_t>(context_lens.size());
  m.num_prefills = num_prefills;
  if (num_prefills == 0) return m;

  int32_t max_context_len = 0;
  int32_t num_prefills_with_context = 0;
  for (int32_t c : context_lens) {
    if (c < 0) throw std::runtime_error("BuildMlaChunkedContext: negative context length");
    max_context_len = std::max(max_context_len, c);
    if (c > 0) ++num_prefills_with_context;
  }
  if (max_context_len == 0) return m;  // `if max_context_len_cpu > 0:` (:1666)

  // ":1683-1690" — an EQUAL share of the workspace per with-context prefill,
  // then rounded DOWN to a page multiple because the gather kernel cannot handle
  // a `context_chunk_start` that is not page-aligned.
  int64_t max_context_chunk = workspace_size / num_prefills_with_context;
  max_context_chunk = (max_context_chunk / page_size) * page_size;
  if (max_context_chunk <= 0) {
    throw std::runtime_error(
        "BuildMlaChunkedContext: chunked-prefill workspace too small for one page per "
        "with-context request (upstream `assert max_context_chunk > 0`, "
        "mla_attention.py:1692)");
  }
  const int32_t num_chunks =
      static_cast<int32_t>((max_context_len + max_context_chunk - 1) / max_context_chunk);
  m.num_chunks = num_chunks;
  m.max_context_chunk = static_cast<int32_t>(max_context_chunk);

  m.starts.assign(static_cast<size_t>(num_chunks) * num_prefills, 0);
  m.chunk_seq_lens.assign(static_cast<size_t>(num_chunks) * num_prefills, 0);
  m.cu_seq_lens.assign(static_cast<size_t>(num_chunks) * (num_prefills + 1), 0);
  m.seq_tot.assign(num_chunks, 0);
  m.max_seq_lens.assign(num_chunks, 0);
  m.chunk_total_token.assign(num_chunks, 0);

  // ":1694-1712" — chunk_starts = arange(num_chunks) * max_context_chunk,
  // chunk_ends = min(context_lens, chunk_starts + max_context_chunk),
  // chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0),
  // cu_seq_lens = cumsum(chunk_seq_lens, dim=1) with a leading 0.
  for (int32_t i = 0; i < num_chunks; ++i) {
    const int64_t start = static_cast<int64_t>(i) * max_context_chunk;
    int32_t running = 0;
    for (int32_t b = 0; b < num_prefills; ++b) {
      const int64_t end = std::min<int64_t>(context_lens[b], start + max_context_chunk);
      const int32_t len = static_cast<int32_t>(std::max<int64_t>(0, end - start));
      m.starts[static_cast<size_t>(i) * num_prefills + b] = static_cast<int32_t>(start);
      m.chunk_seq_lens[static_cast<size_t>(i) * num_prefills + b] = len;
      m.cu_seq_lens[static_cast<size_t>(i) * (num_prefills + 1) + b] = running;
      running += len;
      m.max_seq_lens[i] = std::max(m.max_seq_lens[i], len);
    }
    m.cu_seq_lens[static_cast<size_t>(i) * (num_prefills + 1) + num_prefills] = running;
    m.seq_tot[i] = running;
    m.chunk_total_token[i] = running;  // == cu_seq_lens[:, -1] (:1715)
    m.max_token_num_over_chunk = std::max(m.max_token_num_over_chunk, running);
  }

  // ":1717-1730" — token_to_seq = repeat_interleave(arange(num_prefills),
  // chunk_seq_lens[i]), zero-padded to max_token_num_over_chunk.
  m.token_to_seq.assign(
      static_cast<size_t>(num_chunks) * std::max<int32_t>(m.max_token_num_over_chunk, 1), 0);
  const int32_t row = std::max<int32_t>(m.max_token_num_over_chunk, 1);
  for (int32_t i = 0; i < num_chunks; ++i) {
    int32_t t = 0;
    for (int32_t b = 0; b < num_prefills; ++b) {
      const int32_t len = m.chunk_seq_lens[static_cast<size_t>(i) * num_prefills + b];
      for (int32_t j = 0; j < len; ++j) m.token_to_seq[static_cast<size_t>(i) * row + t++] = b;
    }
  }

  // ":1806-1810" — the query-token count belonging to with-context requests.
  if (num_prefills_with_context > 0) {
    if (static_cast<size_t>(num_prefills_with_context) >= prefill_query_start_loc.size()) {
      throw std::runtime_error(
          "BuildMlaChunkedContext: prefill_query_start_loc must be [num_prefills+1]");
    }
    m.prefill_tokens_with_context = prefill_query_start_loc[num_prefills_with_context];
  }

  // ":1852-1855" — a chunk can never exceed the workspace.
  for (int32_t i = 0; i < num_chunks; ++i) {
    if (m.max_seq_lens[i] > workspace_size) {
      throw std::runtime_error(
          "BuildMlaChunkedContext: chunk max_seq_len exceeds the workspace (upstream "
          "assert, mla_attention.py:1852-1855)");
    }
  }
  return m;
}

// One chunk's DEVICE-resident metadata. Upstream uploads all chunks once in
// `build()` (:1837-1849) and indexes `[chunk_idx]`; we take them pre-uploaded so
// the driver performs no host<->device traffic on the hot path.
struct MlaChunkDeviceMetadata {
  vt::Tensor cu_seq_lens;   // [num_prefills+1] i32
  vt::Tensor token_to_seq;  // [>= total_tokens] i32
  vt::Tensor starts;        // [num_prefills] i32
  int32_t total_tokens = 0;
  int32_t max_seq_len = 0;
};

// The materialized per-head K/V for one chunk. Upstream builds these with
// `kv_b_proj` + `_concat_k_nope_k_pe` (:2160-2170); the actual projection is the
// MODEL's business (W6), so the driver takes it as a callback and stays a pure
// attention-loop port.
struct MlaContextChunkKv {
  vt::Tensor k;  // [toks, num_heads, qk_head_dim]
  vt::Tensor v;  // [toks, num_heads, v_head_dim]
};

// (queue, workspace rows [toks, kv_lora_rank + qk_rope_head_dim], toks) -> K/V.
// Mirrors :2141-2170: the caller slices `workspace[:toks]`, splits the latent
// from the rope part, runs `kv_b_proj`, splits into k_nope/v and concatenates
// k_nope with the (broadcast) rope part.
using MlaUpProjectFn =
    std::function<MlaContextChunkKv(vt::Queue&, const vt::Tensor&, int64_t)>;

// Caller-owned scratch. Upstream gets these from the torch caching allocator
// (`torch.empty_like`, :2186-2187); we make them explicit so nothing allocates
// inside the loop.
struct MlaPrefillContextBuffers {
  vt::Tensor workspace;     // [>= max chunk_total_token, kv_lora_rank + rope]
  vt::Tensor chunk_output;  // [total_q, num_heads, v_head_dim]
  vt::Tensor chunk_lse;     // [num_heads, total_q] f32
  vt::Tensor accum_output;  // [total_q, num_heads, v_head_dim]
  vt::Tensor accum_lse;     // [num_heads, total_q] f32
  vt::Tensor merge_output;  // [total_q, num_heads, v_head_dim]
  vt::Tensor merge_lse;     // [num_heads, total_q] f32
};

// `_compute_prefill_context` (mla_attention.py:2094-2199).
//
// For each chunk i: gather the chunk's context rows out of the paged MLA cache
// into the workspace (:2119-2129), up-project them to per-head K/V (:2141-2170),
// run a NON-CAUSAL varlen attention of the prefill queries against them
// (:2172-2179 -> `run_prefill_context_chunk`, `causal=False` "Context is
// unmasked"), and LSE-merge into the running result (:2181-2197).
//
// On return `*out_output` / `*out_lse` alias whichever of the caller's buffers
// holds the final result — upstream returns the same way, having ping-ponged.
inline void ComputeMlaPrefillContext(vt::Queue& q, const vt::Tensor& query,
                                     const vt::Tensor& kv_cache, const vt::Tensor& block_table,
                                     const vt::Tensor& cu_seqlens_q,
                                     const std::vector<MlaChunkDeviceMetadata>& chunks,
                                     const MlaUpProjectFn& up_project, float scale,
                                     int32_t max_query_len, MlaPrefillContextBuffers& bufs,
                                     vt::Tensor* out_output, vt::Tensor* out_lse) {
  if (chunks.empty()) throw std::runtime_error("ComputeMlaPrefillContext: no chunks");
  vt::Tensor* output = &bufs.accum_output;
  vt::Tensor* output_lse = &bufs.accum_lse;
  vt::Tensor* merge_output = &bufs.merge_output;
  vt::Tensor* merge_lse = &bufs.merge_lse;
  bool have_output = false;

  for (size_t i = 0; i < chunks.size(); ++i) {
    const MlaChunkDeviceMetadata& c = chunks[i];
    const int64_t toks = c.total_tokens;

    // ":2119-2129" — gather this chunk's context out of the paged cache.
    vt::Tensor ws = bufs.workspace;
    vt::GatherMlaCache(q, ws, kv_cache, block_table, c.cu_seq_lens, c.token_to_seq, &c.starts,
                       toks);

    // ":2141-2170" — workspace[:toks] -> per-head K/V.
    vt::Tensor ws_rows = bufs.workspace;
    ws_rows.shape[0] = toks;
    const MlaContextChunkKv kv = up_project(q, ws_rows, toks);

    // ":2172-2179" — NON-causal attention against the gathered context.
    vt::MlaPrefillAttentionArgs args;
    args.scale = scale;
    args.causal = false;
    args.max_seqlen_q = max_query_len;
    args.max_seqlen_k = c.max_seq_len;
    vt::Tensor* dst = have_output ? &bufs.chunk_output : output;
    vt::Tensor* dst_lse = have_output ? &bufs.chunk_lse : output_lse;
    vt::MlaPrefillAttention(q, *dst, dst_lse, query, kv.k, kv.v, cu_seqlens_q, c.cu_seq_lens,
                            args);

    if (!have_output) {
      // ":2181-2183" — the first chunk IS the running result.
      have_output = true;
      continue;
    }
    // ":2185-2197" — merge, then PING-PONG the buffers. `prefill_tokens_with_context`
    // is NOT passed here: every chunk covers the same query rows, so all tokens
    // participate (upstream passes it only at the final :2413-2420 merge).
    vt::MergeAttnStates(q, *merge_output, merge_lse, *output, *output_lse, bufs.chunk_output,
                        bufs.chunk_lse, /*prefill_tokens_with_context=*/-1);
    std::swap(output, merge_output);
    std::swap(output_lse, merge_lse);
  }

  *out_output = *output;
  *out_lse = *output_lse;
}

// `forward_mha` (mla_attention.py:2344-2425), the non-DCP, non-fp8 path — the
// only one reachable on GB10 (`dcp_world_size == 1`; fp8 prefill needs
// device-capability family 100, :1382-1385).
//
//   query/key/value  the NEW tokens' own q and the K/V already materialized from
//                    THIS step's latent (:2371-2375)
//   chunks           empty => no request has context, so the causal new-tokens
//                    result IS the answer (:2421-2425)
//
// The final merge is PREFIX = context, SUFFIX = new tokens (:2413-2420) — the
// order matters, and it carries `prefill_tokens_with_context` so query rows
// belonging to context-free requests take the suffix verbatim.
inline void ForwardMlaPrefillMha(vt::Queue& q, vt::Tensor& output, const vt::Tensor& query,
                                 const vt::Tensor& key, const vt::Tensor& value,
                                 const vt::Tensor& kv_cache, const vt::Tensor& block_table,
                                 const vt::Tensor& cu_seqlens_q,
                                 const std::vector<MlaChunkDeviceMetadata>& chunks,
                                 const MlaUpProjectFn& up_project, float scale,
                                 int32_t max_query_len, int32_t prefill_tokens_with_context,
                                 MlaPrefillContextBuffers& bufs, vt::Tensor& suffix_output,
                                 vt::Tensor& suffix_lse) {
  const bool has_context = !chunks.empty();

  // ":2381-2392" — the causal pass over the new tokens. `return_softmax_lse` is
  // True exactly when there is context to merge with (:2385).
  vt::MlaPrefillAttentionArgs args;
  args.scale = scale;
  args.causal = true;
  args.max_seqlen_q = max_query_len;
  args.max_seqlen_k = max_query_len;
  vt::Tensor& new_out = has_context ? suffix_output : output;
  vt::MlaPrefillAttention(q, new_out, has_context ? &suffix_lse : nullptr, query, key, value,
                          cu_seqlens_q, cu_seqlens_q, args);
  if (!has_context) return;

  vt::Tensor context_output{};
  vt::Tensor context_lse{};
  ComputeMlaPrefillContext(q, query, kv_cache, block_table, cu_seqlens_q, chunks, up_project,
                           scale, max_query_len, bufs, &context_output, &context_lse);

  // ":2413-2420" — prefix = context, suffix = new tokens.
  vt::MergeAttnStates(q, output, nullptr, context_output, context_lse, suffix_output,
                      suffix_lse, prefill_tokens_with_context);
}

}  // namespace vllm::mla

#endif  // VLLM_MODEL_EXECUTOR_LAYERS_ATTENTION_MLA_CHUNKED_CONTEXT_H_
