// SPEC-DFLASH2 W11 (#1890) — the DFlash draft block's attention on the shared
// paged seam, and the equivalence that makes the route safe to default ON.
//
// WHAT #1890 MEASURED. The draft block's attention runs
// `DFlashPagedBlockAttentionWarpKernel` at 449.7 us/call x 5.1 calls/step
// against SGLang's 14.3 us/call for the same work, while the TARGET VERIFY —
// which W10 (#1857) moved onto the FA-2 split-KV lane — runs at 17.1 us/call
// and is marginally ours. Both engines issue 21.3 attention calls/step. One of
// our two lanes is 31x slower per call.
//
// WHY IT WAS NEVER ON THAT LANE, which is the finding this file gates. It is
// not the page size, the block table, the dtype or the head dim. It is KV
// RESIDENCY: `vt::DFlashPagedBlockAttention` reads the block's own (1+k) K/V
// out of contiguous per-layer tensors that are in NO paged cache, and every
// split-KV launcher addresses K and V exclusively through a block table. Handing
// one the store's pools would drop every block row from the attention — a wrong
// answer, not a slow one. So the route is: write the block K/V into the store's
// own pages (`vt::ReshapeAndCache`) and then read the whole thing as ONE paged
// attention, which is what upstream does (`append_paged_kv_cache` then
// `BatchPrefillWithPagedKVCache`, one kernel for both lanes).
//
// THE TWO THINGS THIS FILE ASSERTS.
//
//   1. THE ADMISSION. `ClassifyDflashBlockAttn` is the whole decision, and each
//      of its conjuncts is falsified in turn. The capacity term is the one that
//      is this route's own — the speculative write must fit the pool AND the
//      block table must reach the extended `seq_lens` the attention reads.
//   2. THE EQUIVALENCE, BYTE-FOR-BYTE. For every mask
//      `ResolveQwen3DFlashAttnModes` can produce — the four combinations of
//      `causal` with a window that is present or absent, INCLUDING the
//      non-causal-with-a-window pair the published DFlash2 checkpoint and this
//      repository's runner fixture actually resolve — and for GQA and a
//      multi-page context, the `ReshapeAndCache` + `vt::PagedAttention` pair
//      must produce the IDENTICAL BYTES `vt::DFlashPagedBlockAttention`
//      produces. Not "within a tolerance": the two CPU kernels are the same
//      three-pass online softmax in the same j-ascending order over the same
//      bf16 bits, so anything but equality is a defect. A tolerance here would
//      be a gate that cannot see a reassociation.
//
// The production-path half of the gate lives in
// `tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp`, which drives the
// real runner and reads the route counters.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash_internal.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm::detail::ClassifyDflashBlockAttn;
using vllm::detail::DflashBlockAttnEligibility;
using vllm::detail::DflashBlockAttnRoute;
using vt::Device;
using vt::DeviceType;
using vt::DType;
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

// Deterministic LCG in [-2,2), the same generator the sibling op suites use.
std::vector<uint16_t> RandBf16(size_t n, uint32_t seed) {
  std::vector<uint16_t> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    const float f = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
    x = vt::F32ToBF16(f);
  }
  return v;
}

// The eligibility of the published DFlash2 draft shape: one request, a (1+k)=9
// query block, head_dim 128, Hq 32 / Hkv 8, page 16, 256 pages, bf16
// throughout, switch on. Every case below starts here and falsifies ONE field,
// so a conjunct that stops mattering shows up as a case that goes green.
DflashBlockAttnEligibility Admissible() {
  DflashBlockAttnEligibility e;
  e.num_reqs = 1;
  e.tq = 9;
  e.ctx_len = 640;
  e.max_pages = 256;
  e.block_size = 16;
  e.head_dim = 128;
  e.hq = 32;
  e.hkv = 8;
  e.block_table_col_stride = 1;
  e.bf16_query = true;
  e.bf16_pool = true;
  e.bf16_out = true;
  e.enabled = true;
  return e;
}

// One paged store plus one (1+k) query block, run through BOTH routes, with the
// outputs compared byte-for-byte. `causal`/`window` are the draft layer's own
// `attn_mode` as `Qwen3DFlashLayerAttnMode` states it, and arm B hands them to
// the SAME function `ForwardPagedBody` calls, so the mask translation is gated
// here rather than transcribed.
void CheckRouteEquivalence(int64_t ctx_len, int64_t tq, int64_t hq, int64_t hkv, int64_t d,
                           int64_t block_size, bool causal, int64_t window,
                           uint32_t seed) {
  Queue q = Q();
  const int64_t max_pages = (ctx_len + tq + block_size - 1) / block_size + 2;
  const int64_t pool_elems = max_pages * block_size * hkv * d;

  std::vector<uint16_t> query = RandBf16(static_cast<size_t>(tq * hq * d), seed);
  std::vector<uint16_t> bk = RandBf16(static_cast<size_t>(tq * hkv * d), seed + 11);
  std::vector<uint16_t> bv = RandBf16(static_cast<size_t>(tq * hkv * d), seed + 23);
  // The pool's context rows carry data; the rows the speculative write will
  // land on are POISONED with a value the reference arm must never read, so a
  // route that quietly attends over its own scratch is caught rather than
  // averaged away.
  std::vector<uint16_t> pool_k_a = RandBf16(static_cast<size_t>(pool_elems), seed + 31);
  std::vector<uint16_t> pool_v_a = RandBf16(static_cast<size_t>(pool_elems), seed + 41);
  for (int64_t slot = ctx_len; slot < max_pages * block_size; ++slot) {
    for (int64_t e = 0; e < hkv * d; ++e) {
      pool_k_a[static_cast<size_t>(slot * hkv * d + e)] = vt::F32ToBF16(7.0f);
      pool_v_a[static_cast<size_t>(slot * hkv * d + e)] = vt::F32ToBF16(-7.0f);
    }
  }
  std::vector<uint16_t> pool_k_b = pool_k_a;
  std::vector<uint16_t> pool_v_b = pool_v_a;

  std::vector<int32_t> btab(static_cast<size_t>(max_pages));
  for (int64_t p = 0; p < max_pages; ++p) btab[static_cast<size_t>(p)] = static_cast<int32_t>(p);
  std::vector<int32_t> slen{static_cast<int32_t>(ctx_len)};
  std::vector<int32_t> cu{0, static_cast<int32_t>(tq)};
  // Arm B's TWO derived inputs come from the production derivation, not from a
  // transcription of it. That is what makes the arithmetic gateable: arm A
  // (`vt::DFlashPagedBlockAttention`) reads the block K/V from its own tensors
  // and derives nothing from this function, so moving the slots — the W11 fresh
  // review's mutation, one page up — makes arm B attend over poisoned pool rows
  // and the byte comparison below reds. Before this, the test built its own
  // correct slot map and the production one was ungated on every backend.
  const vllm::detail::DflashBlockPagedInputs paged_in =
      vllm::detail::DflashBlockPagedInputsOf(ctx_len, tq);
  std::vector<int32_t> slen_ext{paged_in.seq_ext};
  std::vector<int64_t> slots = paged_in.slots;

  std::vector<uint16_t> out_a(static_cast<size_t>(tq * hq * d), 0);
  std::vector<uint16_t> out_b(static_cast<size_t>(tq * hq * d), 0);

  const Tensor t_query = Contig(query.data(), DType::kBF16, {tq, hq, d});
  const Tensor t_bk = Contig(bk.data(), DType::kBF16, {tq, hkv, d});
  const Tensor t_bv = Contig(bv.data(), DType::kBF16, {tq, hkv, d});
  const Tensor t_btab = Contig(btab.data(), DType::kI32, {1, max_pages});
  const Tensor t_slen = Contig(slen.data(), DType::kI32, {1});
  const Tensor t_slen_ext = Contig(slen_ext.data(), DType::kI32, {1});
  const Tensor t_cu = Contig(cu.data(), DType::kI32, {2});
  const Tensor t_slots = Contig(slots.data(), DType::kI64, {tq});
  const std::vector<int64_t> pool_shape{max_pages, block_size, hkv, d};

  // Arm A — the shipped bespoke op. Block K/V come from their own tensors; the
  // pool is read only over [0, ctx_len).
  {
    Tensor out = Contig(out_a.data(), DType::kBF16, {tq, hq, d});
    Tensor pk = Contig(pool_k_a.data(), DType::kBF16, pool_shape);
    Tensor pv = Contig(pool_v_a.data(), DType::kBF16, pool_shape);
    vt::DFlashPagedBlockAttentionArgs pa;
    pa.scale = 1.0f / std::sqrt(static_cast<float>(d));
    pa.causal = causal;
    pa.sliding_window = window;
    pa.num_reqs = 1;
    pa.block_size = block_size;
    vt::DFlashPagedBlockAttention(q, out, t_query, t_bk, t_bv, pk, pv, t_cu, t_slen, t_btab,
                                  pa);
  }

  // Arm B — the shared seam, entered through the PRODUCTION function
  // `ForwardPagedBody` calls, not a transcription of it. The block K/V become
  // resident at [ctx_len, ctx_len+tq) and the whole combined sequence is one
  // paged read.
  {
    Tensor out = Contig(out_b.data(), DType::kBF16, {tq, hq, d});
    Tensor pk = Contig(pool_k_b.data(), DType::kBF16, pool_shape);
    Tensor pv = Contig(pool_v_b.data(), DType::kBF16, pool_shape);
    vllm::detail::DflashBlockPagedAttention(q, out, t_query, t_bk, t_bv, pk, pv, t_btab,
                                            t_slen_ext, t_cu, t_slots, paged_in,
                                            1.0f / std::sqrt(static_cast<float>(d)), causal,
                                            window, ctx_len);
  }

  // The claim is BYTE equality, not closeness. Report the first differing
  // element so a reassociation is diagnosable rather than merely red.
  size_t first_diff = out_a.size();
  for (size_t i = 0; i < out_a.size(); ++i) {
    if (out_a[i] != out_b[i]) {
      first_diff = i;
      break;
    }
  }
  INFO("ctx=", ctx_len, " tq=", tq, " hq=", hq, " hkv=", hkv, " d=", d, " causal=", causal,
       " window=", window, " first differing element index: ", first_diff, " of ",
       out_a.size());
  CHECK(first_diff == out_a.size());
  CHECK(std::memcmp(out_a.data(), out_b.data(), out_a.size() * sizeof(uint16_t)) == 0);
}

}  // namespace

TEST_CASE("dflash block route: the published draft shape is ADMITTED") {
  CHECK(ClassifyDflashBlockAttn(Admissible()) == DflashBlockAttnRoute::kPagedSeam);
}

TEST_CASE("dflash block route: the kill switch refuses") {
  DflashBlockAttnEligibility e = Admissible();
  e.enabled = false;
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
}

TEST_CASE("dflash block route: only a SINGLE request routes") {
  DflashBlockAttnEligibility e = Admissible();
  e.num_reqs = 2;
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
  e.num_reqs = 0;
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
}

TEST_CASE("dflash block route: a degenerate query block refuses") {
  DflashBlockAttnEligibility e = Admissible();
  e.tq = 1;
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
  e.tq = 0;
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
}

// THE CAPACITY TERM, AT ITS EDGE, IN BOTH DIRECTIONS. This is the conjunct that
// is this route's own: the speculative write lands at [ctx_len, ctx_len+tq) and
// the attention reads a `seq_lens` of ctx_len+tq, so the last position must be
// addressable by the block table. Exactly-full ADMITS and one-over REFUSES; a
// threshold moved either way fails one of these two.
TEST_CASE("dflash block route: capacity is exact at both edges") {
  DflashBlockAttnEligibility e = Admissible();
  const int64_t cap = e.max_pages * e.block_size;
  e.tq = 8;
  e.ctx_len = cap - 8;  // the write exactly fills the pool
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kPagedSeam);
  e.ctx_len = cap - 7;  // one row over
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
  e.ctx_len = cap;  // no room at all
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
}

TEST_CASE("dflash block route: every dtype conjunct is load-bearing") {
  for (int which = 0; which < 3; ++which) {
    DflashBlockAttnEligibility e = Admissible();
    if (which == 0) e.bf16_query = false;
    if (which == 1) e.bf16_pool = false;
    if (which == 2) e.bf16_out = false;
    INFO("falsified dtype conjunct #", which);
    CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
  }
}

TEST_CASE("dflash block route: a non-GQA-divisible or strided-table shape refuses") {
  DflashBlockAttnEligibility e = Admissible();
  e.hq = 30;  // 30 % 8 != 0
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
  e = Admissible();
  e.block_table_col_stride = 2;  // the launchers assume a unit column stride
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
  e = Admissible();
  e.block_size = 0;
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
  e = Admissible();
  e.head_dim = 0;
  CHECK(ClassifyDflashBlockAttn(e) == DflashBlockAttnRoute::kBlockKernel);
}

// ---------------------------------------------------------------------------
// The equivalence. One case per (causal, window-present) pair the DFlash layer
// resolver can produce, plus the shape variations.
// ---------------------------------------------------------------------------

TEST_CASE("dflash block route: FULL (non-causal) attention is byte-identical") {
  // Multi-page context, GQA 4:1, and a context length that is NOT page-aligned
  // so the speculative write straddles a page boundary.
  CheckRouteEquivalence(/*ctx_len=*/37, /*tq=*/9, /*hq=*/8, /*hkv=*/2, /*d=*/16,
                        /*block_size=*/16, /*causal=*/false, /*window=*/0, /*seed=*/1234);
}

TEST_CASE("dflash block route: causal-SWA is byte-identical, window BINDING") {
  // window 5 over a 37+9 combined sequence: the left bound actually clips, so
  // this case fails if the window is dropped or off by one.
  CheckRouteEquivalence(/*ctx_len=*/37, /*tq=*/9, /*hq=*/8, /*hkv=*/2, /*d=*/16,
                        /*block_size=*/16, /*causal=*/true, /*window=*/5, /*seed=*/4321);
}

// THE MASK THE PRODUCTION FIXTURE ACTUALLY RESOLVES, and the one the first
// battery omitted. `z-lab/Qwen3.8-27B-DFlash2` — and the DFlash2 runner fixture
// that mirrors it — declares every layer `sliding_attention` AND a top-level
// `is_causal: false`, so `ResolveQwen3DFlashAttnModes` yields (causal=false,
// sliding_window>0): a NON-CAUSAL layer that still carries a window value.
// Nothing gated that pair: deleting the `causal &&` guard in
// `DflashBlockPagedMaskOf` left both suites green, because no case combined a
// false `causal` with a non-zero window.
//
// WHAT THIS CASE MEASURES. It measures that the two routes agree BYTE FOR BYTE
// on the pair the production resolver yields, whatever the shared mask
// semantics are — it pins the two arms to EACH OTHER, not to a rule.
//
// #2784 CHANGED THAT RULE, and this case is the reason the change is safe. Both
// arms used to drop the window when `!causal`: the kernels guarded their lower
// bound on `causal && window > 0` and `DflashBlockPagedMaskOf` emitted no
// `window_size`. Against the pin that reads false —
// `vllm/model_executor/models/qwen3_dflash.py:84-146` resolves the window and
// the causal flag as two INDEPENDENT answers, `:221-234` passes
// `per_layer_sliding_window` irrespective of `causal`, and
// `vllm/v1/attention/backends/flash_attn.py:319-330`
// (`_maybe_symmetrize_window`) turns a causal `(w, 0)` into a SYMMETRIC
// `(w, w)` on the non-causal arm rather than into nothing. Both now do that,
// and because this case compares them element for element it is what catches
// one arm moving without the other. The CORRECTNESS of the new bound is gated
// separately, as a property rather than a transcription:
// `tests/vt/test_ops_dflash_paged_block_attn.cpp` asserts that context outside
// the symmetric window cannot reach the output, and
// `tests/vt/test_dflash_attn_mask.cpp` sweeps the bound itself. #1900's
// `## Owed` entry in `.agents/specs/dflash2-draft-block-fa2.md` is discharged
// by `.agents/specs/dflash2-noncausal-swa-window.md`.
//
// The window is 5 over a 37+9 combined sequence, so it BINDS if it is applied —
// a case with a window wider than the sequence would go green either way and
// gate nothing.
TEST_CASE("dflash block route: NON-CAUSAL carrying a window is byte-identical") {
  CheckRouteEquivalence(/*ctx_len=*/37, /*tq=*/9, /*hq=*/8, /*hkv=*/2, /*d=*/16,
                        /*block_size=*/16, /*causal=*/false, /*window=*/5, /*seed=*/2468);
}

TEST_CASE("dflash block route: plain causal (no window) is byte-identical") {
  CheckRouteEquivalence(/*ctx_len=*/37, /*tq=*/9, /*hq=*/8, /*hkv=*/2, /*d=*/16,
                        /*block_size=*/16, /*causal=*/true, /*window=*/0, /*seed=*/777);
}

TEST_CASE("dflash block route: MHA, page-aligned context, single page") {
  CheckRouteEquivalence(/*ctx_len=*/16, /*tq=*/4, /*hq=*/4, /*hkv=*/4, /*d=*/8,
                        /*block_size=*/16, /*causal=*/false, /*window=*/0, /*seed=*/99);
}

// An EMPTY context is the first draft step of every request, and it is the case
// where the two arms have the least in common: arm A reads no page at all while
// arm B reads only the rows it just wrote.
TEST_CASE("dflash block route: an EMPTY context is byte-identical") {
  CheckRouteEquivalence(/*ctx_len=*/0, /*tq=*/4, /*hq=*/8, /*hkv=*/2, /*d=*/16,
                        /*block_size=*/16, /*causal=*/false, /*window=*/0, /*seed=*/5150);
  CheckRouteEquivalence(/*ctx_len=*/0, /*tq=*/4, /*hq=*/8, /*hkv=*/2, /*d=*/16,
                        /*block_size=*/16, /*causal=*/true, /*window=*/2, /*seed=*/5151);
}

// ---------------------------------------------------------------------------
// #2252 — the HOST metadata that keeps this call off a synchronizing path.
//
// WHAT BROKE. `DflashBlockPagedAttention` built its `vt::PagedAttentionArgs`
// without `query_start_loc_host` or `max_seq_len`. Both are OPTIONAL by type,
// so omitting them compiles and is merely SLOW wherever nothing is capturing —
// and `vt/ops.h` spells out the cost of each default in the same words:
// nullptr "=> the launcher falls back to the D2H+sync" (`:1546`), 0 "=> that
// launcher falls back to the D2H+sync" (`:1555`). That fallback ends in
// `cudaStreamSynchronize` (`cuda_paged_attn.cu:2272`), which is ILLEGAL inside
// a CUDA graph capture. The draft block is the one lane this tree captures
// (`P == 1`, `qwen3_dflash.cpp:1716`), so the engine died with "operation not
// permitted when stream is capturing" and the committed speed gate could not
// produce a number at all.
//
// WHY THESE CASES AND NOT A NUMERICS ONE. The equivalence suite above cannot
// see this: on the CPU backend both fields are ignored, so every byte-identical
// case stayed green through the whole defect. The gate has to be on the ARGS.
// Asserting `DflashBlockPagedHostMetaOf` alone would not do it either — that
// would pass while production forgot to USE it, which is precisely how this
// shipped — so the assertion is on `DflashBlockPagedArgsOf`, the pure builder
// the production call now routes through.
TEST_CASE("dflash block paged args: the HOST metadata is populated (#2252)") {
  const int64_t pool_capacity = 26208;  // pages x page rows, the store's bound
  const int64_t tq = 9;
  const vllm::detail::DflashBlockPagedHostMeta hm =
      vllm::detail::DflashBlockPagedHostMetaOf(pool_capacity, tq);

  // One request, so the query_start_loc is exactly [0, tq).
  CHECK(hm.qsl[0] == 0);
  CHECK(hm.qsl[1] == static_cast<int32_t>(tq));
  // The POOL bound, which every replay of the captured graph still satisfies.
  CHECK(hm.max_seq_len == static_cast<int32_t>(pool_capacity));

  const vt::PagedAttentionArgs pa = vllm::detail::DflashBlockPagedArgsOf(
      /*scale=*/0.125F, /*causal=*/true, /*sliding_window=*/0, tq, hm);

  // The two fields whose ABSENCE selects `cudaStreamSynchronize`.
  REQUIRE(pa.query_start_loc_host != nullptr);
  CHECK(pa.query_start_loc_host[0] == 0);
  CHECK(pa.query_start_loc_host[1] == static_cast<int32_t>(tq));
  CHECK(pa.max_seq_len == static_cast<int32_t>(pool_capacity));
  // It must point INTO the caller-owned meta, not at a temporary.
  CHECK(pa.query_start_loc_host == hm.qsl.data());
}

// THE PROPERTY, not the value. This call is captured into a CUDA graph once and
// replayed as the context GROWS, so every host value baked into those args must
// be the same for every replay. The first #2252 fix used `ctx_len + tq` here,
// which is exact at capture and stale on every replay after it: measured as an
// illegal memory access inside `cudaGraphLaunch` (exit 134), against a clean
// exit 0 from the same binary with `VT_DFLASH_PAGED=0`.
//
// A value test alone would not have caught that -- `ctx_len + tq` passes any
// single-point assertion. Only INVARIANCE across context lengths does.
TEST_CASE("dflash block paged args: host metadata is REPLAY-STABLE (#2252)") {
  const int64_t pool_capacity = 4096;
  const int64_t tq = 8;

  const vllm::detail::DflashBlockPagedHostMeta hm =
      vllm::detail::DflashBlockPagedHostMetaOf(pool_capacity, tq);

  // The bound must cover the LARGEST sequence any replay can present. A graph
  // captured while the context is short is replayed until the store is full, so
  // the worst case is the store's own capacity.
  const int32_t worst_case_replay_seq = static_cast<int32_t>(pool_capacity);
  CHECK(hm.max_seq_len >= worst_case_replay_seq);

  // And the defect, stated executably rather than in prose: the bound the first
  // fix shipped is derived from the context AT CAPTURE, and for any capture that
  // happens before the store fills, that bound is smaller than a later replay's
  // sequence. This is the comparison the GPU reported as an illegal access.
  for (const int64_t capture_ctx : {int64_t{0}, int64_t{16}, int64_t{1200}}) {
    const int32_t stale_bound = static_cast<int32_t>(capture_ctx + tq);
    CHECK(stale_bound < worst_case_replay_seq);   // every capture-time bound is short
    CHECK(hm.max_seq_len >= worst_case_replay_seq);  // the shipped one never is
  }
}

// The builder took over the mask and scale wiring, so those must still arrive
// intact — a refactor that fixed the sync and silently dropped the window would
// be a wrong ANSWER, which is worse than the slow path it replaced.
TEST_CASE("dflash block paged args: mask, scale and uniform qlen still flow (#2252)") {
  const vllm::detail::DflashBlockPagedHostMeta hm =
      vllm::detail::DflashBlockPagedHostMetaOf(/*pool_capacity=*/4096, /*tq=*/4);

  const vt::PagedAttentionArgs full = vllm::detail::DflashBlockPagedArgsOf(
      /*scale=*/0.5F, /*causal=*/false, /*sliding_window=*/0, /*tq=*/4, hm);
  CHECK(full.scale == doctest::Approx(0.5F));
  CHECK(full.causal == false);
  CHECK(full.uniform_spec_query_len == 4);
  CHECK_FALSE(full.window_size.has_value());

  // causal + a window is the combination the file's own note calls out as the
  // one no earlier case covered.
  const vt::PagedAttentionArgs swa = vllm::detail::DflashBlockPagedArgsOf(
      /*scale=*/0.25F, /*causal=*/true, /*sliding_window=*/2, /*tq=*/4, hm);
  const vllm::detail::DflashBlockPagedMask expect =
      vllm::detail::DflashBlockPagedMaskOf(/*causal=*/true, /*sliding_window=*/2);
  CHECK(swa.causal == expect.causal);
  REQUIRE(swa.window_size.has_value() == expect.has_window);
  if (expect.has_window) {
    CHECK(swa.window_size->left == expect.window_left);
    CHECK(swa.window_size->right == expect.window_right);
  }
  // THE PAIR THE PUBLISHED CHECKPOINT RESOLVES (#2784). Five
  // `sliding_attention` layers, `sliding_window: 2048`, `is_causal: false`, so
  // `ResolveQwen3DFlashAttnModes` yields (causal=false, window>0) on every
  // layer of every draft step. Upstream's `_maybe_symmetrize_window` makes that
  // window SYMMETRIC; this arm used to emit no window at all, and the draft
  // then attended over the whole context. Spelled as LITERALS rather than
  // through `DflashBlockPagedMaskOf`, because reading the expectation out of
  // the function under test is a tautology.
  const vt::PagedAttentionArgs nc_swa = vllm::detail::DflashBlockPagedArgsOf(
      /*scale=*/0.25F, /*causal=*/false, /*sliding_window=*/2048, /*tq=*/4, hm);
  CHECK(nc_swa.causal == false);
  REQUIRE(nc_swa.window_size.has_value());
  CHECK(nc_swa.window_size->left == 2047);
  CHECK(nc_swa.window_size->right == 2047);
  // The causal arm keeps the one-sided form, so the fix cannot have widened it.
  CHECK(swa.window_size->left == 1);
  CHECK(swa.window_size->right == 0);

  // And the host metadata is not disturbed by the mask arm.
  REQUIRE(swa.query_start_loc_host != nullptr);
  CHECK(swa.max_seq_len == 4096);
}

// ---------------------------------------------------------------------------
// #2274 — the bounds this call writes and reads, refused rather than executed.
//
// WHY THESE ARE NOT THE EXISTING CHECKS. This function already validates the
// slot map and the extended bound, but BOTH are guarded on
// `device.type == kCPU` because they dereference the tensors. On CUDA they do
// not run — and CUDA is where #2274's fault is: an out-of-bounds access on the
// second request of a process, surfaced later and elsewhere as a
// `cudaMemcpyAsync` or `cudaFree` failure with nothing naming this call. The
// checks added here are pure HOST arithmetic over SHAPES, so they run on every
// backend, including the one that faults.
//
// The mutation is the pool: shrink it below what the call addresses and the
// refusal must fire by name. Without the check the same construction reaches
// `ReshapeAndCache`, which writes `tq` rows at `slots` and, on a device, past
// the allocation.
namespace {
// A pool deliberately too small for the (ctx_len + tq) this call addresses.
void ExpectPoolRefusal(int64_t ctx_len, int64_t tq, int64_t pages, int64_t block_size) {
  const int64_t hkv = 2, hq = 4, d = 8;
  vt::Queue q = Q();
  const vllm::detail::DflashBlockPagedInputs paged_in =
      vllm::detail::DflashBlockPagedInputsOf(ctx_len, tq);
  std::vector<uint16_t> query(static_cast<size_t>(tq * hq * d), 0);
  std::vector<uint16_t> bk(static_cast<size_t>(tq * hkv * d), 0);
  std::vector<uint16_t> bv(static_cast<size_t>(tq * hkv * d), 0);
  std::vector<uint16_t> outv(static_cast<size_t>(tq * hq * d), 0);
  std::vector<uint16_t> pool(static_cast<size_t>(pages * block_size * hkv * d), 0);
  std::vector<int32_t> btab(static_cast<size_t>(pages));
  for (int64_t i = 0; i < pages; ++i) btab[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> slen_ext{paged_in.seq_ext};
  std::vector<int32_t> cu{0, static_cast<int32_t>(tq)};
  std::vector<int64_t> slots = paged_in.slots;

  const std::vector<int64_t> pool_shape{pages, block_size, hkv, d};
  Tensor out = Contig(outv.data(), DType::kBF16, {tq, hq, d});
  Tensor pk = Contig(pool.data(), DType::kBF16, pool_shape);
  Tensor pv = Contig(pool.data(), DType::kBF16, pool_shape);
  const Tensor t_query = Contig(query.data(), DType::kBF16, {tq, hq, d});
  const Tensor t_bk = Contig(bk.data(), DType::kBF16, {tq, hkv, d});
  const Tensor t_bv = Contig(bv.data(), DType::kBF16, {tq, hkv, d});
  const Tensor t_btab = Contig(btab.data(), DType::kI32, {1, pages});
  const Tensor t_slen_ext = Contig(slen_ext.data(), DType::kI32, {1});
  const Tensor t_cu = Contig(cu.data(), DType::kI32, {2});
  const Tensor t_slots = Contig(slots.data(), DType::kI64, {tq});

  CHECK_THROWS(vllm::detail::DflashBlockPagedAttention(
      q, out, t_query, t_bk, t_bv, pk, pv, t_btab, t_slen_ext, t_cu, t_slots, paged_in,
      1.0f / std::sqrt(static_cast<float>(d)), /*causal=*/true, /*sliding_window=*/0,
      ctx_len));
}
}  // namespace

TEST_CASE("dflash block paged: a pool too small for the extended bound REFUSES (#2274)") {
  // ctx 60 + 8 block rows = 68 addressed, against a pool holding 4*16 = 64.
  ExpectPoolRefusal(/*ctx_len=*/60, /*tq=*/8, /*pages=*/4, /*block_size=*/16);
}

TEST_CASE("dflash block paged: a write slot past the pool REFUSES (#2274)") {
  // The last slot is ctx_len + tq - 1 = 79, one page beyond a 5*16 = 80 pool's
  // last valid row only when the read bound also passes; this case is sized so
  // the SLOT is the term that fails first.
  ExpectPoolRefusal(/*ctx_len=*/76, /*tq=*/8, /*pages=*/5, /*block_size=*/16);
}

// The control: a pool that DOES fit must not refuse, or the checks above would
// be refusing correct configurations and every equivalence case would red.
TEST_CASE("dflash block paged: an adequate pool is NOT refused (#2274)") {
  const int64_t ctx_len = 60, tq = 8, pages = 16, block_size = 16;
  const int64_t hkv = 2, hq = 4, d = 8;
  vt::Queue q = Q();
  const vllm::detail::DflashBlockPagedInputs paged_in =
      vllm::detail::DflashBlockPagedInputsOf(ctx_len, tq);
  std::vector<uint16_t> query(static_cast<size_t>(tq * hq * d), 0), bk(static_cast<size_t>(tq * hkv * d), 0),
      bv(static_cast<size_t>(tq * hkv * d), 0), outv(static_cast<size_t>(tq * hq * d), 0),
      pool(static_cast<size_t>(pages * block_size * hkv * d), 0);
  std::vector<int32_t> btab(static_cast<size_t>(pages));
  for (int64_t i = 0; i < pages; ++i) btab[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> slen_ext{paged_in.seq_ext}, cu{0, static_cast<int32_t>(tq)};
  std::vector<int64_t> slots = paged_in.slots;
  const std::vector<int64_t> pool_shape{pages, block_size, hkv, d};
  Tensor out = Contig(outv.data(), DType::kBF16, {tq, hq, d});
  Tensor pk = Contig(pool.data(), DType::kBF16, pool_shape);
  Tensor pv = Contig(pool.data(), DType::kBF16, pool_shape);
  CHECK_NOTHROW(vllm::detail::DflashBlockPagedAttention(
      q, out, Contig(query.data(), DType::kBF16, {tq, hq, d}),
      Contig(bk.data(), DType::kBF16, {tq, hkv, d}),
      Contig(bv.data(), DType::kBF16, {tq, hkv, d}), pk, pv,
      Contig(btab.data(), DType::kI32, {1, pages}),
      Contig(slen_ext.data(), DType::kI32, {1}), Contig(cu.data(), DType::kI32, {2}),
      Contig(slots.data(), DType::kI64, {tq}), paged_in,
      1.0f / std::sqrt(static_cast<float>(d)), /*causal=*/true, /*sliding_window=*/0, ctx_len));
}
