// SPEC-DFLASH2 W10 repair (#1865) — the model-side FA-2 dtype/lane class.
//
// WHY THIS FILE EXISTS. #1865's nsys table showed the q=9 DFlash2 verify still
// running `PagedFlashKernel` with `VT_FA2_SPEC_DECODE` default ON: the runner
// classified every verify, the classification reached `PagedAttentionArgs`
// (probed at `vt::PagedAttention` under the production CPU fixture), and the
// CUDA admission then died on its bf16-query conjuncts — because the
// model-side dtype selection had NO spec-as-decode arm. The verify's bf16-ness
// rode the PREFILL lever (`Fa2PrefillOn()`; `T > num_reqs`), while the CUDA
// admission reads the SPEC lane's own switches (`Fa2SpecDecodeEnabled()` and
// the decode-arm toggles). Two sides consulting different switches is a lane
// that dies silently the moment they disagree, and no CPU test could see it:
// `fa2_platform` is false on every CPU path, so the selection was untestable
// until `ClassifyDenseFa2` (qwen3_5_internal.h) extracted it — the same move
// W10 made for the vt-side split (include/vt/paged_attn_route.h).
//
// The 27B production values below are the profiled deployment's: Hq=24,
// Hkv=4 (ratio 6), head_dim 256, bf16 KV, block 16, causal, fused preamble
// with the cos|sin cache, num_speculative_tokens=8 => verify q=9 over 1
// request.
#include <doctest/doctest.h>

#include "vllm/model_executor/models/qwen3_5_internal.h"

namespace {

using vllm::ClassifyDenseFa2;
using vllm::DenseFa2Class;
using vllm::DenseFa2Eligibility;

// The 27B verify step at q=9, every lane toggle at its default (ON), on an
// FA2-capable platform. Cases perturb from here.
DenseFa2Eligibility Spec27B() {
  DenseFa2Eligibility e;
  e.num_q_heads = 24;
  e.num_kv_heads = 4;
  e.head_dim = 256;
  e.num_tokens = 9;
  e.num_reqs = 1;
  e.uniform_spec_query_len = 9;
  e.causal = true;
  e.kv_cache_bf16 = true;
  e.kv_block_multiple_16 = true;
  e.preamble_with_cos_sin = true;
  e.fa2_platform = true;
  e.prefill_on = true;
  e.decode_r4_on = true;
  e.decode_r6_on = true;
  e.decode_r8_on = true;
  e.spec_decode_on = true;
  return e;
}

}  // namespace

TEST_CASE("fa2 class: the classified 27B verify is SPEC-VERIFY, not prefill") {
  // On the healthy default build the verify must be the spec-as-decode class:
  // `PagedAttnIsPrefill` forces a classified admitted batch OFF the prefill
  // class, so a model side that presents it as kPrefill is mirroring a route
  // the dispatch does not take.
  const DenseFa2Eligibility e = Spec27B();
  CHECK(ClassifyDenseFa2(e) == DenseFa2Class::kSpecVerify);
}

TEST_CASE("fa2 class: the verify's bf16 selection survives VT_FA2_PREFILL=0") {
  // THE #1865 DEAD LINK. The prefill lever is the PREFILL lane's rollback; the
  // spec lane reads its own switch. With prefill off and spec on, the verify
  // must still select bf16 through the spec arm — pre-repair it fell to kNone
  // (f32 query) and the CUDA admission could never fire, which is exactly the
  // f32 `PagedFlashKernel` the nsys table showed.
  DenseFa2Eligibility e = Spec27B();
  e.prefill_on = false;
  CHECK(ClassifyDenseFa2(e) == DenseFa2Class::kSpecVerify);
}

TEST_CASE("fa2 class: VT_FA2_SPEC_DECODE=0 restores the prefill class") {
  // The same-binary A/B arm W10 documented: spec off routes the verify exactly
  // as before W10 — the bf16 PREFILL presentation (the vt dispatch then serves
  // it on the FA-2 prefill ladder).
  DenseFa2Eligibility e = Spec27B();
  e.spec_decode_on = false;
  CHECK(ClassifyDenseFa2(e) == DenseFa2Class::kPrefill);
  // And with the prefill lever ALSO off, nothing selects bf16.
  e.prefill_on = false;
  CHECK(ClassifyDenseFa2(e) == DenseFa2Class::kNone);
}

TEST_CASE("fa2 class: the spec arm is scoped by the decode-arm topology toggles") {
  // The spec lane rides the decode kernel, so its per-ratio rollback is the
  // decode arm's toggle (mirror of `fa2_decode_r6` in cuda_paged_attn.cu) —
  // never the prefill lever.
  DenseFa2Eligibility e = Spec27B();
  e.prefill_on = false;
  e.decode_r6_on = false;  // the 27B ratio-6 arm's own rollback
  CHECK(ClassifyDenseFa2(e) == DenseFa2Class::kNone);
  // A foreign topology never spec-classifies (Hq/Hkv = 32/2 is no d256 arm).
  DenseFa2Eligibility f = Spec27B();
  f.prefill_on = false;
  f.num_q_heads = 32;
  f.num_kv_heads = 2;
  CHECK(ClassifyDenseFa2(f) == DenseFa2Class::kNone);
}

TEST_CASE("fa2 class: an unclassified or inconsistent shape never spec-routes") {
  // uniform_spec_query_len == 0: not a verify; T > num_reqs keeps the prefill
  // class (the pre-W10 presentation, byte-identical).
  DenseFa2Eligibility e = Spec27B();
  e.uniform_spec_query_len = 0;
  CHECK(ClassifyDenseFa2(e) == DenseFa2Class::kPrefill);
  // A stale field over a rewritten pure-decode batch (the BuildPaddedDecode
  // belt): S == q*S only at q == 1, so the shape guard refuses and the batch
  // takes the DECODE class it always took.
  DenseFa2Eligibility s = Spec27B();
  s.num_tokens = 1;
  s.num_reqs = 1;
  s.uniform_spec_query_len = 9;
  CHECK(ClassifyDenseFa2(s) == DenseFa2Class::kDecode);
}

TEST_CASE("fa2 class: the pre-repair arms are byte-identical semantics") {
  // kPrefill and kDecode must be exactly the predicate FullAttnBlockPaged
  // carried inline before the extraction, so the refactor cannot move a
  // token anywhere.
  DenseFa2Eligibility e = Spec27B();
  // Pure decode: every request at query length 1.
  e.num_tokens = 1;
  e.num_reqs = 1;
  e.uniform_spec_query_len = 0;
  CHECK(ClassifyDenseFa2(e) == DenseFa2Class::kDecode);
  // Non-causal pure decode declines the decode arm; T == num_reqs also fails
  // the prefill arm's T > num_reqs, so the class is kNone (f32 fallback).
  e.causal = false;
  CHECK(ClassifyDenseFa2(e) == DenseFa2Class::kNone);
  // Prefill: T > num_reqs, no classification.
  DenseFa2Eligibility p = Spec27B();
  p.num_tokens = 82;
  p.num_reqs = 1;
  p.uniform_spec_query_len = 0;
  CHECK(ClassifyDenseFa2(p) == DenseFa2Class::kPrefill);
  // The shared base conjuncts kill every lane: no platform SASS, no bf16 KV,
  // no fused preamble, wrong head_dim.
  for (auto strike : {0, 1, 2, 3}) {
    DenseFa2Eligibility b = Spec27B();
    if (strike == 0) b.fa2_platform = false;
    if (strike == 1) b.kv_cache_bf16 = false;
    if (strike == 2) b.preamble_with_cos_sin = false;
    if (strike == 3) b.head_dim = 128;
    CHECK(ClassifyDenseFa2(b) == DenseFa2Class::kNone);
  }
}
