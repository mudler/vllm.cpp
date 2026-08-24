// Host-side lane classification for vt::PagedAttention — the dispatch-selection
// seam a CPU test can enter (SPEC-DFLASH2 W10, #1857).
//
// WHY A HEADER. The CUDA dispatch (`src/vt/cuda/cuda_paged_attn.cu::LaunchPaged`)
// splits every batch into a PREFILL class (flash/WMMA ladder) and a DECODE class
// (split-KV / block kernels) with host-known shape arithmetic. That predicate
// was previously inline in the .cu, where no CPU-only build can even compile it,
// let alone test it. The spec-as-decode change moves the class split here so a
// test without a GPU pins the routing, and the .cu consumes the same bytes.
//
// WHAT IT MIRRORS. Upstream backends with a dedicated decode kernel keep a
// uniform-qlen speculative verify on that kernel: `supports_spec_as_decode`
// raises the decode-reorder threshold (vllm/v1/attention/backend.py:718-736 and
// backends/flashinfer.py:852-860 @ b389ac2946). The THRESHOLD policy and the
// classification live on the vllm side (`SpecAsDecodeQueryLen`,
// include/vllm/v1/attention/backend.h) because they read the speculative
// config; what lives here is only what the kernel dispatch itself decides:
// the shape-consistency guard and the prefill/decode class split.

#ifndef VLLM_CPP_INCLUDE_VT_PAGED_ATTN_ROUTE_H_
#define VLLM_CPP_INCLUDE_VT_PAGED_ATTN_ROUTE_H_

#include <cstdint>

namespace vt {

// The shape-consistency guard for a CLASSIFIED uniform speculative batch:
// `uniform_spec_query_len` (PagedAttentionArgs) is trusted for uniformity —
// the classifier verified it against per-request query lengths the kernel
// cannot see — but the guard still requires the batch to be exactly the shape
// the classification describes, so a stale field over a rewritten batch (the
// padded pure-decode rebuilds set num_tokens == num_reqs, and S == q*S only at
// q == 1) can never route. q <= 1 is refused: a pure decode routes decode by
// shape already, and 0 is the "not classified" value.
inline bool PagedAttnUniformSpecShape(int64_t num_tokens, int64_t num_reqs,
                                      int64_t uniform_spec_query_len) {
  return uniform_spec_query_len > 1 && num_reqs > 0 &&
         num_tokens == uniform_spec_query_len * num_reqs;
}

// The PREFILL/DECODE class split LaunchPaged applies before its per-lane gates.
// `spec_as_decode` is the full admission of the classified batch onto a decode
// lane that can serve it (shape guard AND the lane's own dtype/head/ratio/env
// gates, composed in the .cu); when false this is the shipped predicate
// verbatim — every request at query length 1 ⟺ num_tokens == num_reqs — so
// every unclassified or inadmissible batch routes byte-identically to before.
inline bool PagedAttnIsPrefill(int64_t num_tokens, int64_t num_reqs,
                               bool spec_as_decode) {
  if (spec_as_decode) return false;
  return num_tokens > num_reqs;
}

}  // namespace vt

#endif  // VLLM_CPP_INCLUDE_VT_PAGED_ATTN_ROUTE_H_
