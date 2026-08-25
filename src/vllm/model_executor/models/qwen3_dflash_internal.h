// Host-side routing seam for the DFlash draft block's ATTENTION
// (SPEC-DFLASH2 W11, #1890). Not installed; tests include it by path, the same
// way `qwen3_5_internal.h` is reached.
//
// WHY A HEADER. `ForwardPagedBody` (qwen3_dflash.cpp) decides, per layer,
// whether the draft's (1+k) query block attends through the bespoke
// `vt::DFlashPagedBlockAttention` op or through the SHARED paged-attention seam
// (`vt::ReshapeAndCache` + `vt::PagedAttention`). That decision is pure shape,
// dtype and capacity arithmetic; leaving it inline in the forward makes it
// reachable only by running a whole draft step. Extracted here it is a function
// a CPU test can call directly, and the forward consumes the same bytes — the
// extraction W10 made for the vt lane split (`include/vt/paged_attn_route.h`)
// and #1865's repair made for the model-side dtype selection.
//
// WHAT IT DOES NOT DECIDE. It does not read the FA-2 lane's own conjuncts (head
// dim, GQA ratio, compiled-in FA2, `VT_FA2_SPEC_DECODE`). Routing onto the
// paged seam is CORRECT on every backend and bit-identical on CPU; WHICH kernel
// then serves the paged read is the backend dispatch's business, exactly as
// `PagedAttentionArgs::uniform_spec_query_len` is "a ROUTING HINT, not a
// semantic change" (`include/vt/ops.h`). Deciding the kernel here would put a
// CUDA-only predicate on a path CPU also takes.

#ifndef VLLM_CPP_SRC_VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_INTERNAL_H_
#define VLLM_CPP_SRC_VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_INTERNAL_H_

#include <cstdint>

#include "vt/ops.h"

namespace vllm {
namespace detail {

// Which lane the draft block's attention takes.
enum class DflashBlockAttnRoute {
  // `vt::DFlashPagedBlockAttention` — the shipped D12/D13/D14 op, which reads
  // the block's own K/V out of contiguous per-layer tensors that are in no
  // paged cache. This is the rollback arm and the bit-identity reference.
  kBlockKernel = 0,
  // `vt::ReshapeAndCache` into the store's pages, then `vt::PagedAttention`
  // over `[0, ctx_len + tq)`. The block K/V become resident, which is the ONE
  // property that kept the draft off every split-KV lane (#1890).
  kPagedSeam = 1,
};

// Everything the decision reads, gathered on the caller so the predicate itself
// touches no tensor. All lengths are in ROWS unless named otherwise.
struct DflashBlockAttnEligibility {
  int64_t num_reqs = 0;                 // query blocks in this call
  int64_t tq = 0;                       // (1+k) query rows of the single block
  int64_t ctx_len = 0;                  // C, the store's committed context length
  int64_t max_pages = 0;                // block_table columns == pool pages
  int64_t block_size = 0;               // rows per page
  int64_t head_dim = 0;
  int64_t hq = 0;                       // query heads
  int64_t hkv = 0;                      // kv heads
  int64_t block_table_col_stride = 0;   // block_table.stride[1]
  bool bf16_query = false;
  bool bf16_pool = false;
  bool bf16_out = false;
  bool enabled = false;                 // VT_FA2_DFLASH_BLOCK (see qwen3_dflash.cpp)
};

// The admission. Every conjunct is a property `vt::ReshapeAndCache` or
// `vt::PagedAttention` would otherwise refuse by name at run time, plus the
// capacity term that is this route's own:
//
//   ctx_len + tq <= max_pages * block_size
//
// which is TWO obligations in one comparison. The speculative K/V write lands
// at slots [ctx_len, ctx_len+tq) and must fit the pool; and the extended
// `seq_lens` value the attention reads is ctx_len+tq, so the block table must
// carry enough columns to address its last position — the exact bound
// `vt::PagedAttention` checks per request (#1394) and the exact bound
// `ScatterProjectedContextRows` already applies to the accepted append.
//
// A single request only. The multi-request draft path
// (`ForwardWithCtxKVDev`) materializes `[context; block]` and owns no paged
// store, so it is out of this route's reach and out of W11's scope.
inline DflashBlockAttnRoute ClassifyDflashBlockAttn(
    const DflashBlockAttnEligibility& e) {
  // W11, THE RED COMMIT: nothing classifies onto the paged seam yet, which is
  // the pre-W11 behaviour exactly. The admission body below is unreachable and
  // lands live in the next commit; the tests that require it are RED here, on
  // purpose, and the commit message records what they print.
  (void)e;
  return DflashBlockAttnRoute::kBlockKernel;
  if (!e.enabled) return DflashBlockAttnRoute::kBlockKernel;
  if (e.num_reqs != 1) return DflashBlockAttnRoute::kBlockKernel;
  if (e.tq <= 1) return DflashBlockAttnRoute::kBlockKernel;
  if (e.block_size <= 0 || e.max_pages <= 0) return DflashBlockAttnRoute::kBlockKernel;
  if (e.head_dim <= 0 || e.hq <= 0 || e.hkv <= 0) return DflashBlockAttnRoute::kBlockKernel;
  if (e.hq % e.hkv != 0) return DflashBlockAttnRoute::kBlockKernel;
  if (e.block_table_col_stride != 1) return DflashBlockAttnRoute::kBlockKernel;
  if (!e.bf16_query || !e.bf16_pool || !e.bf16_out) return DflashBlockAttnRoute::kBlockKernel;
  if (e.ctx_len < 0) return DflashBlockAttnRoute::kBlockKernel;
  if (e.ctx_len + e.tq > e.max_pages * e.block_size) return DflashBlockAttnRoute::kBlockKernel;
  return DflashBlockAttnRoute::kPagedSeam;
}

// THE MASK, translated once. A DFlash layer states its attention as
// (`causal`, `sliding_window`) — `Qwen3DFlashLayerAttnMode`, resolved from the
// checkpoint by `qwen3_dflash_weights.cpp`. `PagedAttentionArgs` states the
// same thing as (`causal`, optional `window_size`), and the two agree exactly
// because both use FlashAttention's BOTTOM-RIGHT alignment: for query row
// `local` of a block of `tq` rows against `ctx_len + tq` keys, the absolute
// position is `p = (ctx_len + tq) - tq + local = ctx_len + local`, which is
// `ii_comb` in `DFlashPagedBlockAttentionKernel` verbatim.
//
// This lives beside the classifier, and not inline in the forward, so the
// equivalence gate exercises the PRODUCTION translation rather than a copy of
// it: `test_qwen3_dflash_block_route` feeds this function the layer mode and
// compares the resulting paged read against `vt::DFlashPagedBlockAttention` over
// the same store, element for element. A transcription in the test would gate
// nothing.
//
// `causal == false` carries NO window, because the block kernel ignores
// `sliding_window` when it is non-causal (its lower bound is guarded on
// `causal && window > 0`). A non-causal DFlash layer is bidirectional over the
// whole combined sequence.
struct DflashBlockPagedMask {
  bool causal = false;
  bool has_window = false;
  int32_t window_left = 0;
  int32_t window_right = 0;
};

inline DflashBlockPagedMask DflashBlockPagedMaskOf(bool causal, int64_t sliding_window) {
  DflashBlockPagedMask m;
  m.causal = causal;
  if (causal && sliding_window > 0) {
    m.has_window = true;
    m.window_left = static_cast<int32_t>(sliding_window - 1);
    m.window_right = 0;
  }
  return m;
}

// THE ROUTED ATTENTION ITSELF — the write and the read, as ONE call.
//
// WHY THEY ARE ONE FUNCTION AND NOT TWO LINES IN THE FORWARD. The two ops are a
// single obligation: the paged read addresses K and V exclusively through the
// block table, so without the write immediately before it the draft attends
// over whatever the pool's unwritten slots happen to hold. The W11 mutation
// pass measured what that costs when the pair is spelled inline: deleting the
// `vt::ReshapeAndCache` call left BOTH suites green, because the production
// call site is reachable only through a whole draft step and the CPU fixture's
// drafted tokens did not move. Binding the pair here makes the equivalence
// gate — which compares this function against `vt::DFlashPagedBlockAttention`
// byte for byte — the thing that catches it.
//
//   out          [tq, hq, d]  bf16, the block-query outputs
//   query        [tq, hq, d]  bf16
//   block_k/v    [tq, hkv, d] bf16, the block's OWN K/V (not yet resident)
//   pool_k/v     [pages, block_size, hkv, d] bf16, WRITTEN at the slot map
//   block_table  [1, pages] i32
//   seq_ext      [1] i32 = ctx_len + tq   (the EXTENDED context bound)
//   cu           [2] i32 = {0, tq}
//   slot_map     [tq] i64 = [ctx_len, ctx_len + tq)
//
// `uniform_spec_query_len` is the routing hint W10 (#1857) added: a uniform
// (1+k) block over one request IS a classified uniform-qlen batch, and it is
// what admits the read onto the FA-2 split-KV DECODE lane instead of the
// num_splits=1 prefill ladder. It changes no semantics — a backend that ignores
// it is still correct.
inline void DflashBlockPagedAttention(vt::Queue& q, vt::Tensor& out, const vt::Tensor& query,
                                      const vt::Tensor& block_k, const vt::Tensor& block_v,
                                      vt::Tensor& pool_k, vt::Tensor& pool_v,
                                      const vt::Tensor& block_table, const vt::Tensor& seq_ext,
                                      const vt::Tensor& cu, const vt::Tensor& slot_map,
                                      float scale, bool causal, int64_t sliding_window,
                                      int64_t ctx_len) {
  // THE EXTENDED BOUND IS THE LOAD-BEARING ARGUMENT, AND IT IS CHECKABLE.
  // `seq_ext` must read `ctx_len + tq`, not the store's own `seq_lens`: with the
  // store's value the read stops at the last COMMITTED context row and every
  // block row this call just made resident is silently dropped, so the draft
  // attends to the context alone. That is a wrong answer, and the W11 mutation
  // pass measured that this repository's DFlash2 runner fixture cannot see it —
  // its drafted blocks are a CONSTANT (`19 19 19` at every step), so a token
  // comparison through the production runner is a tautology against any
  // numerics change (owed, #1894).
  //
  // On a host-addressable device the value is right there, so read it and
  // refuse by name. The CPU production runner then catches the mis-wiring at
  // the first draft layer instead of drafting quietly from stale pages.
  if (seq_ext.device.type == vt::DeviceType::kCPU && seq_ext.data != nullptr) {
    VT_CHECK(seq_ext.Ptr<int32_t>()[0] ==
                 static_cast<int32_t>(ctx_len + query.shape[0]),
             "dflash block paged attention: seq_ext must be the EXTENDED context "
             "bound (ctx_len + block rows); the store's own seq_lens would drop "
             "every block row from the read (SPEC-DFLASH2 W11, #1890)");
  }
  vt::ReshapeAndCache(q, block_k, block_v, pool_k, pool_v, slot_map);
  vt::PagedAttentionArgs pa;
  pa.scale = scale;
  const DflashBlockPagedMask mask = DflashBlockPagedMaskOf(causal, sliding_window);
  pa.causal = mask.causal;
  if (mask.has_window)
    pa.window_size = vt::AttentionWindow{mask.window_left, mask.window_right};
  pa.uniform_spec_query_len = static_cast<int32_t>(query.shape[0]);
  vt::PagedAttention(q, out, query, pool_k, pool_v, block_table, seq_ext, cu, pa);
}

// ROUTE VISIBILITY, CPU-gateable. Which lane a draft-block attention took is
// invisible from outside the forward for the same reason `spec_as_decode_steps`
// exists one wave over: a token gate cannot see a kernel choice, and W10 spent
// a whole profiled campaign with a lane dark because nothing counted it
// (#1865). These are process counters, one increment per attention CALL (per
// layer, per draft forward), so a production-runner test can assert
// `forwards x attention layers` on one lane and zero on the other.
//
// NOT THREAD-SAFE BY DESIGN, mirroring `vllm::v1::GraphDispatchStats`: these
// are single-process diagnostics read by a gate that drives one runner on one
// thread, and an atomic would suggest a cross-thread guarantee nothing here
// provides.
struct DflashBlockRouteStats {
  int64_t paged_seam_calls = 0;
  int64_t block_kernel_calls = 0;
};
DflashBlockRouteStats GetDflashBlockRouteStats();
void ResetDflashBlockRouteStats();
void NoteDflashBlockRoute(DflashBlockAttnRoute route);

}  // namespace detail
}  // namespace vllm

#endif  // VLLM_CPP_SRC_VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_INTERNAL_H_
