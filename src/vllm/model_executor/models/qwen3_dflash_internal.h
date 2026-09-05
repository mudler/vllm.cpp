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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"  // #2274 opt-in device readback
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
  // THE THIRD ROUTE (#2089). `ForwardWithCtxKVDev` materializes one combined
  // `[context ; block]` K/V per layer and calls `vt::DFlashBlockAttention` over
  // it. This is what PRODUCTION runs at every `P > 1`, i.e. at every serving
  // concurrency above one, and until W12 NOTHING counted it: both increments
  // sat inside the `P == 1` branch, so a route gate read zero on both lanes
  // while a third lane ran. `ClassifyDflashBlockAttn` never returns this value
  // — the route is chosen by the caller's `P`, one level up — so it is noted at
  // the call site by `NoteDflashCombinedAttn` and not by the classifier.
  kMaterializedCombined = 2,
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
// `causal == false` CARRIES A SYMMETRIC WINDOW, and until #2784 it carried
// none. Upstream's `_maybe_symmetrize_window`
// (vllm/v1/attention/backends/flash_attn.py:319-330 @ the parity pin
// 5559679229) makes a causal `(w, 0)` window `(w, w)` when attention is
// non-causal, "so bidirectional queries attend in both directions", and
// `:665-696` puts that value in the metadata every DFlash group reads. This
// function used to drop the window entirely on that arm, matching a `causal &&`
// guard the eleven DFlash kernels also carried — and the published DFlash2
// drafter declares five `sliding_attention` layers, `sliding_window: 2048` and
// `is_causal: false`, so on the checkpoint this engine ships EVERY draft layer
// ran unwindowed. Below 2048 tokens of context that is inert; above it #2784
// measured acceptance falling from 0.77 at 2,307 prompt tokens to 0.06 at
// 8,159. Both backends already implement the right-hand bound
// (`src/vt/cpu/cpu_paged_attn.cpp:223-225`,
// `src/vt/cuda/cuda_paged_attn.cu:220-224`), so only this translation moved.
struct DflashBlockPagedMask {
  bool causal = false;
  bool has_window = false;
  int32_t window_left = 0;
  int32_t window_right = 0;
};

inline DflashBlockPagedMask DflashBlockPagedMaskOf(bool causal, int64_t sliding_window) {
  DflashBlockPagedMask m;
  m.causal = causal;
  if (sliding_window > 0) {
    m.has_window = true;
    m.window_left = static_cast<int32_t>(sliding_window - 1);
    // (w-1, 0) causal, (w-1, w-1) symmetric — upstream's two forms, and the
    // two `vt::AttentionWindow` names in include/vt/ops.h.
    m.window_right = causal ? 0 : static_cast<int32_t>(sliding_window - 1);
  }
  return m;
}

// THE TWO HOST-DERIVED INPUTS, FROM ONE CONTEXT LENGTH.
//
// WHY THIS FUNCTION EXISTS. The paged seam needs two values the bespoke op did
// not: the slots the speculative K/V write lands on, and the extended bound the
// read stops at. Both are a pure function of `ctx_len` and `tq`, and BOTH were
// spelled out by hand at each of the two production sites — the eager path and
// the CUDA-graph refresh. The W11 fresh review measured what that costs: adding
// one page to the slot arithmetic at both sites (`ctx_len + i` -> `ctx_len + i
// + 16`) left `test_qwen3_dflash_block_route` 28/28 and
// `test_dflash2_runner_reach` 162/162 GREEN, while every draft query attended
// to the context plus stale pool rows and never saw its own block K/V. A wrong
// answer, on every backend, invisible to both gates.
//
// It was invisible because the slot map was a PARAMETER of the gated function:
// `DflashBlockPagedAttention` received it, so the equivalence gate built its
// own correct one and compared that. Binding the write and the read into one
// call closed the CALL-DELETED variant of the defect (M5) and not the
// ARGUMENTS-WRONG variant. Deriving both values HERE closes the second one, and
// it needs no backend read:
//
//   - a defect in the ARITHMETIC now reds the byte-for-byte equivalence gate,
//     because `CheckRouteEquivalence` builds arm B's tensors from this function
//     while arm A (`vt::DFlashPagedBlockAttention`) derives nothing from it.
//     This is host arithmetic with no device term, so the CPU gate covers CUDA
//     exactly;
//   - a defect in the ARGUMENT is refused by name inside
//     `DflashBlockPagedAttention`, which re-derives the canonical value from
//     the `ctx_len` it reads off the STORE and compares. That check carries no
//     `kCPU` guard — it compares two host values and never dereferences a
//     device pointer — so on every step that ENTERS the function it holds on
//     CUDA exactly as it holds on CPU.
//
// AND THOSE ARE NOT ALL THE STEPS. The check is per CALL, not per draft STEP,
// and the W11 fresh re-review measured the difference. The graph path still
// refreshes the device buffers OUTSIDE the captured region, because a replay
// does not call this function at all: on `st.g_state == 2` the driver replays
// and returns (`qwen3_dflash.cpp:1636-1660`), so `ForwardPagedBody` — and this
// function with it — is entered on the EAGER lane and on the ONE
// warm-then-capture step per request, and on NO replay step. What the replay
// steps read is refreshed by a SECOND production site
// (`qwen3_dflash.cpp:1627-1631`) that on those steps has nothing downstream to
// check it, so a refresh that is stale or absent there is a wrong answer this
// guard cannot see. Nor can any CPU gate: the capture-capable CPU backend's
// `ReplayGraph` is a log push that executes nothing
// (`tests/vllm/models/decode_graph_seam_harness.h:117`), so the owed proof is a
// device-side one. Filed as
// [#1902](https://github.com/mudler/vllm.cpp/issues/1902) and listed under
// `## Owed` in `.agents/specs/dflash2-draft-block-fa2.md`.
struct DflashBlockPagedInputs {
  std::vector<int64_t> slots;  // [ctx_len, ctx_len + tq)
  int32_t seq_ext = 0;         // ctx_len + tq
};

inline DflashBlockPagedInputs DflashBlockPagedInputsOf(int64_t ctx_len, int64_t tq) {
  DflashBlockPagedInputs in;
  in.slots.resize(static_cast<size_t>(tq));
  for (int64_t i = 0; i < tq; ++i) in.slots[static_cast<size_t>(i)] = ctx_len + i;
  in.seq_ext = static_cast<int32_t>(ctx_len + tq);
  return in;
}

// The HOST attention metadata a draft block must hand `vt::PagedAttention`, so
// the CUDA prefill launchers size their query-tile grid from host values
// instead of taking the fallback. BOTH fields are load-bearing and `vt/ops.h`
// states the cost of each default in the same words: `query_start_loc_host`
// nullptr "=> the launcher falls back to the D2H+sync" (`:1546`) and
// `max_seq_len` 0 "=> that launcher falls back to the D2H+sync" (`:1555`).
//
// THAT FALLBACK IS `cudaStreamSynchronize` (`cuda_paged_attn.cu:2272`), and it
// is ILLEGAL INSIDE A CUDA GRAPH CAPTURE -- which is where this call runs, on
// the one lane this tree captures (`P == 1`, `qwen3_dflash.cpp:1716`). Omitting
// them is not a compile error and is merely slow wherever nothing is capturing,
// so a non-captured run cannot see it; under capture the engine dies with
// "operation not permitted when stream is capturing" (#2252).
//
// One request, so the query_start_loc is exactly [0, tq); `tq` is `1 + k`, a
// constant for the life of the graph.
//
// `max_seq_len` MUST BE THE POOL CAPACITY, NOT `ctx_len + tq`. This call is
// captured into a CUDA graph ONCE and replayed on every later draft step, and
// `MakeDeviceKVStore` states the invariant that makes that legal: the
// persistent buffers never move, so "a captured graph reads the growing context
// purely through the in-place `seq_lens` value". A HOST value derived from the
// current `ctx_len` is baked into the graph at capture and is then STALE on
// every replay, because the context has grown -- which is an illegal memory
// access inside `cudaGraphLaunch`, not a wrong number.
//
// MEASURED, 2026-08-29, three arms on one boot at the smallest workload
// (`max_num_seqs=1`, c=1, 64 tokens): with `ctx_len + tq` the engine aborted
// 134 at `cudaMemcpyAsync`, and under `CUDA_LAUNCH_BLOCKING=1` at
// `cudaGraphLaunch`, naming the replay; with `VT_DFLASH_PAGED=0`, which
// bypasses this route entirely, the same binary exited 0.
//
// The capacity is replay-stable and is a valid bound: the store refuses a
// request whose `ctx_len + append + (1+k)` would exceed it
// (`runner.cpp`, the ctx-capacity fallback), so the read never addresses past
// it. An upper bound is explicitly safe here because it only sizes grids and
// rounded dims, while per-request geometry stays on the DEVICE values
// (`ops.h:1551-1553`).
struct DflashBlockPagedHostMeta {
  std::array<int32_t, 2> qsl{};  // [0, tq) for the block's single request
  int32_t max_seq_len = 0;       // ctx_len + tq
};

inline DflashBlockPagedHostMeta DflashBlockPagedHostMetaOf(int64_t pool_capacity, int64_t tq) {
  DflashBlockPagedHostMeta m;
  m.qsl = {0, static_cast<int32_t>(tq)};
  m.max_seq_len = static_cast<int32_t>(pool_capacity);
  return m;
}

// The complete argument set for a draft block's paged attention, as a PURE
// function of the block's shape -- so the wiring is gateable on the CPU. A test
// that only checked `DflashBlockPagedHostMetaOf` would pass while production
// forgot to USE it, which is exactly how #2252 shipped.
//
// `host_meta` must outlive the returned args: `query_start_loc_host` points
// into it.
inline vt::PagedAttentionArgs DflashBlockPagedArgsOf(float scale, bool causal,
                                                     int64_t sliding_window, int64_t tq,
                                                     const DflashBlockPagedHostMeta& host_meta) {
  vt::PagedAttentionArgs pa;
  pa.scale = scale;
  const DflashBlockPagedMask mask = DflashBlockPagedMaskOf(causal, sliding_window);
  pa.causal = mask.causal;
  if (mask.has_window)
    pa.window_size = vt::AttentionWindow{mask.window_left, mask.window_right};
  pa.uniform_spec_query_len = static_cast<int32_t>(tq);
  pa.query_start_loc_host = host_meta.qsl.data();
  pa.max_seq_len = host_meta.max_seq_len;
  return pa;
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
//   host_inputs  the HOST values `slot_map` and `seq_ext` were filled from
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
                                      const DflashBlockPagedInputs& host_inputs, float scale,
                                      bool causal, int64_t sliding_window, int64_t ctx_len) {
  // BOTH DERIVED INPUTS ARE CHECKED AGAINST THE STORE'S OWN `ctx_len`, ON EVERY
  // BACKEND. `ctx_len` arrives from `store.num_ctx` — this function's one
  // trustworthy term — so re-deriving the canonical pair here and comparing is
  // what makes a wrong ARGUMENT a refusal rather than a wrong answer. The
  // comparison is between two HOST values, so it carries no `kCPU` guard and no
  // device read: on CUDA the mis-wiring is refused at the first draft layer,
  // exactly as it is on CPU. The W11 fresh review's mutation — the production
  // slot arithmetic moved one page — is what this exists for.
  const DflashBlockPagedInputs canon =
      DflashBlockPagedInputsOf(ctx_len, query.shape[0]);
  VT_CHECK(host_inputs.seq_ext == canon.seq_ext && host_inputs.slots == canon.slots,
           "dflash block paged attention: the slot map and the extended bound must be "
           "derived from the STORE's context length (slots [ctx_len, ctx_len+tq), "
           "seq_ext ctx_len+tq); anything else makes the draft attend over stale pool "
           "rows and never see its own block K/V (SPEC-DFLASH2 W11, #1890)");

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
    VT_CHECK(seq_ext.Ptr<int32_t>()[0] == canon.seq_ext,
             "dflash block paged attention: seq_ext must be the EXTENDED context "
             "bound (ctx_len + block rows); the store's own seq_lens would drop "
             "every block row from the read (SPEC-DFLASH2 W11, #1890)");
  }
  // And the same reading for the slot map, which closes the third variant: the
  // host values were right and the UPLOAD did not land on the tensor this call
  // reads (a stale graph buffer, a copy that went elsewhere). Only the two ends
  // are read, because the range is contiguous by construction and a per-row
  // scan would buy nothing this pair does not.
  if (slot_map.device.type == vt::DeviceType::kCPU && slot_map.data != nullptr &&
      query.shape[0] > 0) {
    const int64_t* s = slot_map.Ptr<int64_t>();
    VT_CHECK(s[0] == canon.slots.front() &&
                 s[query.shape[0] - 1] == canon.slots.back(),
             "dflash block paged attention: the slot map must be the contiguous "
             "range [ctx_len, ctx_len + block rows) the read's extended bound "
             "addresses (SPEC-DFLASH2 W11, #1890)");
  }
  // #2274 — OPT-IN DEVICE READBACK. The two checks above that read `seq_ext` and
  // `slot_map` are `kCPU`-guarded because they dereference the tensors, so on
  // CUDA the one variant their own comment names is unchecked: "the host values
  // were right and the UPLOAD did not land on the tensor this call reads (a
  // stale graph buffer, a copy that went elsewhere)". Every HOST-side bound this
  // function derives has been verified correct on the failing configuration
  // (`ddd527f3f`, detectors silent), so a host/device divergence is what is
  // left.
  //
  // OFF by default: the read is a `Download`, which synchronizes, and this call
  // sits on the no-sync path the whole route exists to keep. `VT_DFLASH_BOUNDS_DEVICE=1`
  // turns it on for a diagnostic run.
  static const bool bounds_device = [] {
    const char* e = std::getenv("VT_DFLASH_BOUNDS_DEVICE");
    return e != nullptr && e[0] == '1';
  }();
  if (bounds_device && seq_ext.device.type != vt::DeviceType::kCPU &&
      seq_ext.data != nullptr) {
    vt::Backend& bb = vt::GetBackend(seq_ext.device.type);
    int32_t dev_seq = -1;
    bb.Copy(q, &dev_seq, seq_ext.data, sizeof(int32_t));
    bb.Synchronize(q);
    VT_CHECK(dev_seq == canon.seq_ext,
             "dflash block paged attention: the DEVICE seq_ext does not match the "
             "host value this call derived — the upload did not land, or the "
             "buffer is stale; the paged read would use that length "
             "(SPEC-DFLASH2, #2274)");
    if (query.shape[0] > 0 && slot_map.data != nullptr) {
      const int64_t tqn = query.shape[0];
      std::vector<int64_t> dev_slots(static_cast<size_t>(tqn), -1);
      bb.Copy(q, dev_slots.data(), slot_map.data,
              static_cast<size_t>(tqn) * sizeof(int64_t));
      bb.Synchronize(q);
      VT_CHECK(dev_slots.front() == canon.slots.front() &&
                   dev_slots.back() == canon.slots.back(),
               "dflash block paged attention: the DEVICE slot map does not match "
               "the host range [ctx_len, ctx_len+tq); ReshapeAndCache would write "
               "where this call did not intend (SPEC-DFLASH2, #2274)");
    }
  }

  // #2274 — THE BOUNDS THIS CALL WRITES AND READS, CHECKED ON EVERY BACKEND.
  //
  // The two checks above are guarded on `kCPU` because they dereference device
  // tensors. These are pure HOST arithmetic over shapes, so they run on CUDA
  // too — which is where the fault is. `ReshapeAndCache` below WRITES `tq` rows
  // at `slots`, and the attention then READS `[0, seq_ext)` through
  // `block_table`; if either passes the pool, that is an illegal access inside a
  // kernel, reported later and elsewhere (a `cudaMemcpyAsync` or a `cudaFree`)
  // with nothing naming this call. #2274 is exactly that shape: an out-of-bounds
  // access on the second request of a process, surfacing far from its cause.
  //
  // This is a DETECTOR, not a repair. If it fires, the caller's accounting is
  // wrong and the message says which term; if it never fires, this class is
  // excluded and the search moves on. Cost is a few host integer comparisons per
  // draft layer.
  const int64_t pool_pages = pool_k.shape[0];
  const int64_t page_rows = pool_k.shape[1];
  const int64_t pool_capacity = pool_pages * page_rows;
  VT_CHECK(canon.seq_ext <= pool_capacity,
           "dflash block paged attention: the extended read bound passes the pool "
           "(seq_ext > pages*page_rows); the attention would read unmapped pages "
           "(SPEC-DFLASH2, #2274)");
  VT_CHECK(host_inputs.slots.empty() ||
               host_inputs.slots.back() < pool_capacity,
           "dflash block paged attention: the last write slot passes the pool "
           "(slots.back() >= pages*page_rows); ReshapeAndCache would write past "
           "the K/V pages (SPEC-DFLASH2, #2274)");
  // The block table must ADDRESS every page the extended bound reaches. It is
  // `[1, max_pages]`, so its own width is the reachable page count -- a table
  // shorter than `ceil(seq_ext / page_rows)` sends the kernel through an
  // uninitialised entry, which is an arbitrary page index rather than a refusal.
  VT_CHECK(page_rows > 0 && block_table.rank == 2 && block_table.shape[1] > 0,
           "dflash block paged attention: the block table must be [1, max_pages] "
           "over a positive page size (SPEC-DFLASH2, #2274)");
  VT_CHECK(block_table.shape[1] * page_rows >= canon.seq_ext,
           "dflash block paged attention: the block table cannot address the "
           "extended bound (max_pages*page_rows < seq_ext), so the read walks off "
           "the end of the table (SPEC-DFLASH2, #2274)");
  vt::ReshapeAndCache(q, block_k, block_v, pool_k, pool_v, slot_map);
  // #2252: `host_meta` outlives the call below, which is all it must do -- the
  // launcher reads the qsl on the host to size its grid before it launches.
  // The POOL's capacity (pages x page rows), not this step's context length --
  // see the note on `DflashBlockPagedHostMetaOf`: a per-step value baked into a
  // replayed graph reads out of bounds.
  const DflashBlockPagedHostMeta host_meta =
      DflashBlockPagedHostMetaOf(pool_capacity, query.shape[0]);
  const vt::PagedAttentionArgs pa =
      DflashBlockPagedArgsOf(scale, causal, sliding_window, query.shape[0], host_meta);
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
  // #2089: the P>1 production lane, which had no counter at all.
  int64_t materialized_combined_calls = 0;
  // SPEC-DFLASH2 W12 D1 (#2087): the LAUNCH SHAPE of the last combined
  // attention, read off the tensors that were actually passed. A count alone
  // cannot see D1 — the lane is taken either way and the tokens are identical —
  // so the gateable property is that the query spans the (1+k) BLOCK rows and
  // not the `C + Tq` combined rows. Reverting D1 puts `Ncomb` in the first
  // field and the assertion goes red on the shape, which is the only thing a
  // CPU gate can see about a cost this large.
  int64_t last_combined_query_rows = 0;
  int64_t last_combined_key_rows = 0;
};
DflashBlockRouteStats GetDflashBlockRouteStats();
// SPEC-DFLASH2 W13, closing #2112: the `[dflash-route]` line, without a trailing
// newline. The second of the two counter families #2112 names as unreadable from
// a server; the runner emits it beside `[graph-dispatch]` on the same cadence.
std::string FormatDflashBlockRouteStats(const DflashBlockRouteStats& s);
void ResetDflashBlockRouteStats();
void NoteDflashBlockRoute(DflashBlockAttnRoute route);
void NoteDflashCombinedAttn(int64_t query_rows, int64_t key_rows);

}  // namespace detail
}  // namespace vllm

#endif  // VLLM_CPP_SRC_VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_INTERNAL_H_
