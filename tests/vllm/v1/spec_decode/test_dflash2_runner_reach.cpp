// SPEC-DFLASH2 W3 (#1314) — the PRODUCTION reachability gate for the DFlash2
// candidate selector, and the discharge of spec `## Owed` O5 and O7.
//
// WHAT O5 AND O7 SAID, AND WHY THIS FILE EXISTS. W1 and W2 both landed with
// their production call site UNGATED and mutation-proven so: deleting
// `RefuseDflash2CandidateSelector` from `GPUModelRunner::propose_drafts_block`,
// or `draft->weights.conv_block_size = draft->k + 1;` from `LoadDflashDraft`,
// left every suite in this repository green. The stated reason was that a gate
// on the runner would need an on-disk target plus an on-disk draft driven
// through the loader, plus a populated per-request device KV store, and that
// this was W4's harness.
//
// It was not. What was missing was a way to hand a DFlash draft to a LoadedEngine
// built from IN-MEMORY weights -- the exact seam `mtp_weights` already had, and
// which its own header comment justifies in the same words: "a synthetic spec
// engine could only ever run with a NULL drafter, which is exactly the state a
// depth gate must not mistake for working speculation". W3 adds that overload,
// and everything downstream is the production path unchanged: ResolveSpecConfig,
// the aux-multi-tap refusal, `set_dflash_draft`, `propose_drafts` ->
// `propose_drafts_dflash` -> `propose_drafts_block`.
//
// WHAT THIS GATE ASSERTS AS OF W4, and why generation SUCCEEDING is not the
// assertion. Through W3 a DFlash2 draft could not generate at all: the path walk
// was missing and the engine refused by name, so the refusal's own text -- which
// carried the selector's counts -- was the whole gate. W4 lands the walk, so the
// engine generates, and "it generated" would be satisfied by a runner that
// dropped the DFlash2 arm entirely and drafted with the DFlash1 per-slot argmax.
// That is the exact silent-wrong this row exists to remove: the verify is
// lossless, the emitted tokens stay the target's, and only acceptance falls.
//
// So this file asserts the two things an argmax fallback cannot produce:
//
//   1. THE DRAFTS EXIST AND COME OUT OF THE PROPOSE, read off the production
//      `VT_SPEC_TRACE` line at real fd 2 rather than a test-only sink.
//   2. THE DRAFTS MOVE WITH THE SELECTOR'S VALUES. Two engines that differ ONLY
//      in `output_multiplier` and `final_logit_softcapping` -- D9's Muse Glimmer
//      scalars -- must draft DIFFERENT tokens. Those two scalars touch nothing
//      but the candidate VALUES the selector's unary term reads, so the block
//      forward, the convolution and the draft logits are byte-identical between
//      the two runs. The DFlash1 argmax reads the block LOGITS and would answer
//      identically for both; only a draft produced by walking the selector's
//      lattice can differ. The difference is MEASURED by this case, not asserted
//      about the fixture.
//
// And the third leg is structural rather than observational: both propose paths
// enter the DFlash1 argmax on EMPTINESS and call
// `RefuseDflash1ArgmaxOnDflash2Block` first, so deleting the walk's call site --
// the mutation `.agents/reachability.md` requires -- throws by name instead of
// quietly drafting worse tokens.
//
// The target is the same tiny synthetic Qwen3.5 DENSE model
// tests/vllm/v1/spec_decode/test_mtp_depth.cpp builds, because
// `supports_aux_multi_tap()` is true only for the Qwen3.5/3.6 dense and MoE
// forwards and the loader refuses any other target by name.
#include <doctest/doctest.h>

#include <cstdlib>
#include <string>

#include "dflash2_runner_fixture.h"

// W10 (#1857): the spec-as-decode classification counter (src-tree header, the
// same seam test_mtp_depth reads for the W6 uniform-decode counters).
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"
// W10 repair (#1865): the classified-arrival counter at the shared
// PagedAttention dispatch wrapper.
#include "vt/ops.h"
// W11 (#1890): the draft-block attention route counters, same src-tree seam.
#include "vllm/model_executor/models/qwen3_dflash_internal.h"

namespace {
// `VT_SPEC_TRACE` is latched by a function-local `static` on the FIRST propose in
// the process, so setting it inside a test case would be a race with whichever
// case ran first. This runs before main.
const bool kSpecTraceEnabled = [] {
  ::setenv("VT_SPEC_TRACE", "1", 1);
  return true;
}();

}  // namespace

TEST_CASE("dflash2 runner: a DFlash2 draft DRAFTS through the PATH WALK") {
  std::string threw;
  const std::vector<std::string> blocks = RunAndCollectDrafts(false, &threw);
  // W3 refused here by name. W4 must not.
  INFO("threw: ", threw);
  CHECK(threw.empty());
  // The propose ran, and it produced a whole block per step. `kSpecTokens` is 3,
  // so each traced block carries three ids.
  REQUIRE_FALSE(blocks.empty());
  for (const std::string& b : blocks) {
    INFO("block: [", b, "]");
    int ids = 0;
    for (size_t i = 0; i < b.size(); ++i)
      if (b[i] == ' ') ++ids;
    CHECK(ids == kSpecTokens);
  }
  // And every drafted id is a real token of this vocabulary rather than a
  // sentinel or an uninitialized slot -- the walk gathers `candidates.ids`, and
  // a walk that emitted the SLOT INDEX instead would still print three numbers.
  // kSelTopK is 3 and the vocabulary is 24, so "every id < top_k" would be the
  // signature of that mistake; this asserts at least one id is not.
  bool any_beyond_top_k = false;
  for (const std::string& b : blocks) {
    std::istringstream is(b);
    int id = 0;
    while (is >> id) {
      CHECK(id >= 0);
      CHECK(id < kVocab);
      if (id >= kSelTopK) any_beyond_top_k = true;
    }
  }
  CHECK(any_beyond_top_k);
}

TEST_CASE("dflash2 runner (W9): level 1 does NOT emit the device split") {
  // THE LEVEL BOUNDARY, PINNED FROM BELOW (#1851 F2). The level-2 binary pins
  // that `VT_SPEC_TRACE=2` emits `[spec-phase-dev]`; nothing pinned that
  // level 1 does NOT — mutating the runner's `propose_trace_level >= 2` to
  // `>= 1` left every suite green, so "the syncs run ONLY at level >= 2" (the
  // W9 claim that keeps every level-1 recipe overlap-preserving) was asserted,
  // not gated. This binary latches level 1 pre-main, so it is the one process
  // that can observe the boundary from this side: the level-1 line must be
  // PRESENT on the same real-fd-2 capture (proving the capture and the trace
  // both worked — absence alone would also be a dead instrument) and the
  // device split must be ABSENT.
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  std::string threw;
  const std::string captured = CaptureStderr([&] {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir), MakeDflash2Draft(target, false));
    threw = GenerateAndCatch(eng, "hello");
  });
  INFO("stderr: ", captured);
  CHECK(threw.empty());
  CHECK(captured.find("[spec-phase] ") != std::string::npos);
  CHECK(captured.find("[spec-propose]") != std::string::npos);
  CHECK(captured.find("[spec-phase-dev]") == std::string::npos);
}

TEST_CASE("dflash2 runner: the STARTUP notice names what runs, not what is owed") {
  // `## Risks/decisions` D10 pays for the moved refusal with a startup notice,
  // and every wave that moves the boundary has to move the notice with it. W3's
  // fresh review found the OTHER copy of this text still naming the wave that had
  // just shipped; the obligation is that the notice is TRUE at its own commit,
  // and this is what holds it.
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  std::string captured;
  {
    CerrCapture cap;
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir), MakeDflash2Draft(target, false));
    captured = cap.str();
  }
  INFO("notice: ", captured);
  CHECK(captured.find("DFlash2DraftModel") != std::string::npos);
  CHECK(captured.find("PATH WALK are all implemented") != std::string::npos);
  CHECK(captured.find("this draft DRAFTS") != std::string::npos);
  // What is still owed, named -- the GGUF drafter's bf16 residency and the
  // absent throughput number -- so the notice is not a claim that the row is
  // finished. W5 (#1314) LANDED the GGUF arm, so the notice must no longer name
  // it as owed; a text that kept saying so would tell a user running that arm
  // that it is refused, which is the staleness this case exists to catch.
  CHECK(captured.find("wave W5") == std::string::npos);
  CHECK(captured.find("GGUF drafter arm is refused") == std::string::npos);
  CHECK(captured.find("from safetensors and from GGUF alike") != std::string::npos);
  CHECK(captured.find("DEQUANTIZED") != std::string::npos);
  CHECK(captured.find("no throughput number") != std::string::npos);
  CHECK(captured.find("#1314") != std::string::npos);
  // And that the port is BEYOND-PIN, which is the one thing a user of a DFlash2
  // checkpoint cannot discover from the checkpoint.
  CHECK(captured.find("52816") != std::string::npos);
  // THE UPSTREAM STATE, PINNED AS A WORD AND NOT ONLY AS A NUMBER. vllm#52816
  // MERGED on 2026-08-21 at 05:27:22Z, and this notice went on telling every
  // user who loads a DFlash2 draft that it "is OPEN upstream" -- through W6 and
  // through W6's first repair wave, which corrected five statements in the spec
  // and never reached the two live surfaces. The line above matches only
  // "52816", so it could not see the difference. These four can: the merged
  // wording and the merged head must be PRESENT, and both spellings of the open
  // claim must be ABSENT. When #1561 reconciles the port onto the merged head,
  // the notice and this case move together.
  CHECK(captured.find("MERGED upstream") != std::string::npos);
  CHECK(captured.find("3406ec1dae9916f920b90f0dbf90dcf54923d042") !=
        std::string::npos);
  CHECK(captured.find("OPEN upstream") == std::string::npos);
  CHECK(captured.find("is OPEN") == std::string::npos);
}

TEST_CASE("dflash2 runner: D9's SCALARS move the drafted tokens, in production") {
  // THE REACHABILITY ASSERTION, and the reason it is this one. Both runs share
  // the same draft weights, the same target, the same prompt and the same block
  // forward; the ONLY difference is `output_multiplier` and
  // `final_logit_softcapping`, which `Qwen3DFlash2Model::ComputeCandidates`
  // applies to the candidate VALUES the selector's unary term reads. Nothing
  // else in the engine sees them.
  //
  // So the DFlash1 per-slot argmax -- which reads the block LOGITS -- answers
  // IDENTICALLY for both. A difference here can only come from a draft that was
  // produced by walking the selector's lattice, on the production path, at the
  // runner's own call site. That is what no exit status and no "it generated"
  // could show.
  std::string threw_default, threw_muse;
  const std::vector<std::string> plain = RunAndCollectDrafts(false, &threw_default);
  const std::vector<std::string> muse = RunAndCollectDrafts(true, &threw_muse);
  INFO("default threw: ", threw_default);
  INFO("muse threw: ", threw_muse);
  CHECK(threw_default.empty());
  CHECK(threw_muse.empty());
  REQUIRE_FALSE(plain.empty());
  REQUIRE(plain.size() == muse.size());
  int differing = 0;
  for (size_t i = 0; i < plain.size(); ++i) {
    INFO("step ", i, " default [", plain[i], "] muse [", muse[i], "]");
    if (plain[i] != muse[i]) ++differing;
  }
  // MEASURED, not assumed, and the MARGIN is written down: on 2026-08-20 this
  // fixture drafts 8 blocks and 2 of them differ between the two scalar arms.
  // A block only moves when the scalars flip the argmax at one of the `k` slots
  // the walk actually visits, so 2-of-8 is the shape of the synthetic ramp
  // rather than a weakness in the wiring -- the model suite measures the same
  // comparison at 5 of 6 blocks over a wider sweep of inputs
  // (tests/vllm/models/test_qwen3_dflash2_draft.cpp). The count is printed so a
  // fixture change that made the two arms agree is visible as a number rather
  // than as a silent pass.
  INFO("blocks: ", plain.size(), " differing: ", differing);
  CHECK(differing > 0);
}

// SPEC-DFLASH2 W8 (#1837, #1838): the two propose lanes agree at the ENGINE.
// Since W8 a DFlash2 propose takes the single-request PAGED lane by default and
// `VT_DFLASH_PAGED=0` selects the materialized fallback — before W8 both
// spellings took the fallback, so this case became meaningful with the wave.
// The drafted blocks must agree BIT-FOR-BIT across the lanes on the production
// path end to end (aux pre-phase, block forward, device selector, walk): the
// wave's whole claim is that residency moved and no float did.
TEST_CASE("dflash2 runner (W8): the paged lane and the materialized lane draft identically") {
  std::string threw_paged, threw_mat;
  const std::vector<std::string> paged = RunAndCollectDrafts(false, &threw_paged);
  setenv("VT_DFLASH_PAGED", "0", 1);
  const std::vector<std::string> mat = RunAndCollectDrafts(false, &threw_mat);
  unsetenv("VT_DFLASH_PAGED");
  INFO("paged threw: ", threw_paged);
  INFO("materialized threw: ", threw_mat);
  CHECK(threw_paged.empty());
  CHECK(threw_mat.empty());
  REQUIRE_FALSE(paged.empty());
  REQUIRE(paged.size() == mat.size());
  for (size_t i = 0; i < paged.size(); ++i) {
    INFO("step ", i, " paged [", paged[i], "] materialized [", mat[i], "]");
    CHECK(paged[i] == mat[i]);
  }
}

// SPEC-DFLASH2 W10 (#1857): the uniform verify is CLASSIFIED spec-as-decode.
//
// The runner already names every step's verified uniform query length
// (`GraphEligibleQueryLen`, W6); W10 classifies that length onto the DECODE
// attention class through the mirrored reorder threshold
// (`SpecAsDecodeQueryLen`: k=3 with parallel drafting => threshold 1+2*3=7, and
// this fixture's verify width 1+3=4 sits inside it) and counts the decision
// (`GraphDispatchStats::spec_as_decode_steps`), because the attention lane a
// step took is invisible to a token gate — the #1020 lesson, applied to #1857.
//
// EVERY uniform q>1 step this engine produces must classify: the runner's
// verified width is bounded by 1+k, which sits strictly inside the 1+2k
// threshold, so the two counters must be EQUAL. RED before the wiring: the
// verify steps run, `uniform_spec_steps` moves, and `spec_as_decode_steps`
// stays 0 because nothing classifies. The reachability mutation deletes the
// runner's classification call site and must re-red exactly this case.
TEST_CASE("dflash2 runner (W10): every uniform verify classifies spec-as-decode") {
  vllm::v1::ResetGraphDispatchStats();
  vt::ResetPagedAttnSpecClassifiedCount();
  vllm::detail::ResetDflashBlockRouteStats();
  std::string threw;
  const std::vector<std::string> blocks = RunAndCollectDrafts(false, &threw);
  INFO("threw: ", threw);
  CHECK(threw.empty());
  // Drafts were proposed, so verify steps followed them.
  REQUIRE_FALSE(blocks.empty());
  const vllm::v1::GraphDispatchStats st = vllm::v1::GetGraphDispatchStats();
  INFO("uniform_spec_steps: ", st.uniform_spec_steps,
       " spec_as_decode_steps: ", st.spec_as_decode_steps);
  REQUIRE(st.uniform_spec_steps > 0);
  CHECK(st.spec_as_decode_steps == st.uniform_spec_steps);
  // W10 repair (#1865): the classification must ARRIVE at the attention
  // dispatch, not only be counted at the runner. The W10 fresh review declared
  // this seam dead on a CPU box — deleting the model's
  // `pa_args.uniform_spec_query_len = meta.uniform_spec_query_len` threading
  // redded nothing, because its only consumer was the HasCuda-gated CUDA lane.
  // `vt::PagedAttention` (the ONE wrapper every backend's dispatch sits
  // behind) now counts shape-consistent classified arrivals: this fixture's
  // target has exactly ONE full-attention layer, so the production engine owes
  // one arrival per classified verify step. Deleting the threading — the
  // review's exact mutation — makes this 0 and reds here.
  //
  // SINCE W11 (#1890) THE DRAFT ARRIVES HERE TOO, and the identity has to say
  // so rather than be widened into a bound. The draft block's attention now
  // reaches `vt::PagedAttention` with its own uniform (1+k) classification, so
  // this counter carries BOTH lanes: one arrival per classified verify step
  // plus one per routed draft-block attention call. Splitting the total across
  // the two named counters keeps the W10-repair guarantee exactly as strong —
  // deleting the target-side threading still makes the verify term 0 — while
  // stating the new lane's contribution instead of absorbing it. Measured on
  // this fixture: 23 = 7 verify steps + 16 draft calls (8 forwards x 2 draft
  // layers).
  const uint64_t arrivals = vt::PagedAttnSpecClassifiedCount();
  const vllm::detail::DflashBlockRouteStats route =
      vllm::detail::GetDflashBlockRouteStats();
  INFO("classified attention arrivals: ", arrivals, " = verify ",
       st.spec_as_decode_steps, " + draft ", route.paged_seam_calls);
  CHECK(arrivals == static_cast<uint64_t>(st.spec_as_decode_steps) +
                        static_cast<uint64_t>(route.paged_seam_calls));
  // And the verify term is not zero, so the sum above cannot be satisfied by
  // the draft lane alone.
  CHECK(arrivals > static_cast<uint64_t>(route.paged_seam_calls));
}

// SPEC-DFLASH2 W11 (#1890): the draft block's attention takes the SHARED PAGED
// SEAM, through the production runner, and the two routes draft identically.
//
// WHAT THIS IS FOR. #1890 measured the draft block's attention at 449.7 us/call
// against SGLang's 14.3 for the same work, while W10's target verify sits at
// 17.1. The reason it never reached a split-KV lane is KV RESIDENCY: the
// bespoke op reads the block's own (1+k) K/V out of tensors that are in no
// paged cache, and every such launcher addresses K/V through a block table.
// W11 writes those rows into the store's own pages and reads the whole thing as
// ONE paged attention, which is the presentation upstream uses for the same
// work (`append_paged_kv_cache` then a paged attention).
//
// Which lane a step took is invisible to a token gate — the #1020 lesson W10
// paid for again in #1865, where a whole profiled campaign ran with the FA-2
// arm dark and every counter green. So the decision moves a number, and this
// case reads it through the real runner: every draft-block attention call must
// be on the paged seam and none on the bespoke op. RED before the routing: the
// forward issues `vt::DFlashPagedBlockAttention` for every call, so
// `paged_seam_calls` is 0 and `block_kernel_calls` carries the total.
//
// The reachability mutation for W11 deletes the routing call site in
// `ForwardPagedBody` and must re-red exactly this case.
TEST_CASE("dflash2 runner (W11): the draft block attention takes the PAGED SEAM") {
  vllm::detail::ResetDflashBlockRouteStats();
  std::string threw;
  const std::vector<std::string> blocks = RunAndCollectDrafts(false, &threw);
  INFO("threw: ", threw);
  CHECK(threw.empty());
  REQUIRE_FALSE(blocks.empty());
  const vllm::detail::DflashBlockRouteStats st = vllm::detail::GetDflashBlockRouteStats();
  INFO("paged_seam_calls: ", st.paged_seam_calls,
       " block_kernel_calls: ", st.block_kernel_calls);
  CHECK(st.paged_seam_calls > 0);
  CHECK(st.block_kernel_calls == 0);
  // One call per DRAFT LAYER per forward, so the total is a multiple of the
  // draft's layer count. A route that fired on some layers and not others would
  // still move the counter; this says it fired on all of them.
  CHECK(st.paged_seam_calls % kDraftLayers == 0);
}

// The SAME-BINARY A/B, and an HONEST statement of what it proves.
//
// WHAT IT PROVES: the kill switch reaches the production forward and moves the
// lane, both lanes complete without throwing, and neither perturbs the engine's
// output. The counter assertions are the discriminating half — the W11 mutation
// pass reds them by deleting the routing call site.
//
// WHAT IT DOES NOT PROVE, MEASURED RATHER THAN ASSUMED: it is NOT a numerics
// gate. This fixture's drafted blocks are a CONSTANT — `19 19 19` at every one
// of its eight steps — so comparing them across two arms cannot see an
// attention change at all. Mutating the mask polarity, or neutralising the
// paged K/V write, leaves this case green while the byte-for-byte op
// equivalence in `test_qwen3_dflash_block_route` reds on every mask case. The
// degeneracy is a property of the fixture rather than of W11, and the same
// limitation applies to the W8 lane-comparison case above; it is filed as
// [#1894](https://github.com/mudler/vllm.cpp/issues/1894) and owed by this row.
// The wave's numerics claim rests on the op-level gate, not on this case.
TEST_CASE("dflash2 runner (W11): VT_FA2_DFLASH_BLOCK=0 reaches the forward and is inert") {
  vllm::detail::ResetDflashBlockRouteStats();
  std::string threw_on;
  const std::vector<std::string> on = RunAndCollectDrafts(false, &threw_on);
  const vllm::detail::DflashBlockRouteStats st_on =
      vllm::detail::GetDflashBlockRouteStats();

  vllm::detail::ResetDflashBlockRouteStats();
  ::setenv("VT_FA2_DFLASH_BLOCK", "0", 1);
  std::string threw_off;
  const std::vector<std::string> off = RunAndCollectDrafts(false, &threw_off);
  ::unsetenv("VT_FA2_DFLASH_BLOCK");
  const vllm::detail::DflashBlockRouteStats st_off =
      vllm::detail::GetDflashBlockRouteStats();

  INFO("on threw: ", threw_on, " off threw: ", threw_off);
  CHECK(threw_on.empty());
  CHECK(threw_off.empty());
  // The switch really moved the lane — otherwise this would compare a run
  // against itself, which is the tautology shape a same-binary A/B must not be.
  INFO("on: seam ", st_on.paged_seam_calls, " block ", st_on.block_kernel_calls,
       " | off: seam ", st_off.paged_seam_calls, " block ", st_off.block_kernel_calls);
  REQUIRE(st_on.paged_seam_calls > 0);
  REQUIRE(st_on.block_kernel_calls == 0);
  REQUIRE(st_off.block_kernel_calls > 0);
  REQUIRE(st_off.paged_seam_calls == 0);
  CHECK(st_on.paged_seam_calls == st_off.block_kernel_calls);

  REQUIRE_FALSE(on.empty());
  REQUIRE(on.size() == off.size());
  for (size_t i = 0; i < on.size(); ++i) {
    INFO("step ", i, " route-on [", on[i], "] route-off [", off[i], "]");
    CHECK(on[i] == off[i]);
  }
}

// ─── SPEC-DFLASH2 W12 (#2087, #2089) — the BATCHED propose, in the runner ────
//
// Every case above drives ONE request, and that is the reason this cost survived
// three profiled waves. At P == 1 the draft takes the paged graph fast path and
// both route counters report it truthfully. At every P > 1 the draft falls to
// `ForwardWithCtxKVDev`, which before W12 was counted by NOTHING: a route gate
// read zero on both lanes and could not tell "this lane did not run" from
// "nothing counts this lane" (#2089).
//
// So this case drives TWO concurrent requests through the production engine —
// `add_request` twice, then `step()` to completion, which is how the e2e tests
// drive concurrency and how `propose_drafts_block` gets P > 1 — and asserts the
// two things that lane owes:
//
//   1. it is COUNTED (#2089);
//   2. the attention it issues spans the (1+k) BLOCK rows, not the `C + Tq`
//      combined rows (#2087 D1). The tokens are identical either way, because
//      the rows D1 stops computing were discarded, so the launch SHAPE is the
//      only thing a gate without a GPU can see about a ~150x-per-row cost.
//
// The reachability mutation: restore the pre-D1 call in `ForwardWithCtxKVDev`
// and `last_combined_query_rows` becomes the combined length, which reds the
// last check here while every token assertion in this binary stays green.
TEST_CASE("dflash2 runner (W12): TWO concurrent requests draft over Tq rows, not C+Tq") {
  vllm::detail::ResetDflashBlockRouteStats();
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  std::string threw;
  int steps = 0;
  {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(), DflashSpecParams(dir),
                     MakeDflash2Draft(target, false));
    try {
      eng.engine().add_request("a", "hello", Greedy(8));
      eng.engine().add_request("b", "hello", Greedy(8));
      while (eng.engine().has_unfinished_requests() && steps < 200) {
        (void)eng.engine().step();
        ++steps;
      }
    } catch (const std::exception& e) {
      threw = e.what();
    }
  }
  INFO("threw: ", threw, " steps: ", steps);
  CHECK(threw.empty());
  REQUIRE(steps > 0);

  const vllm::detail::DflashBlockRouteStats st = vllm::detail::GetDflashBlockRouteStats();
  INFO("combined=", st.materialized_combined_calls, " seam=", st.paged_seam_calls,
       " block=", st.block_kernel_calls, " qrows=", st.last_combined_query_rows,
       " krows=", st.last_combined_key_rows);
  // #2089: the P>1 lane moves a number now. One call per draft layer per forward.
  REQUIRE(st.materialized_combined_calls > 0);
  CHECK(st.materialized_combined_calls % kDraftLayers == 0);
  // D1: the query is the batch's (1+k) block rows. Two proposing rows at
  // kSpecTokens drafts each is 2 * (1 + kSpecTokens); a batch that ended with
  // one proposing row is 1 * (1 + kSpecTokens). Both are far below the key
  // count, which carries the whole batch's context, and THAT is the assertion:
  // the query must not span the keys.
  REQUIRE(st.last_combined_key_rows > 0);
  CHECK(st.last_combined_query_rows <= 2 * (1 + kSpecTokens));
  CHECK(st.last_combined_query_rows % (1 + kSpecTokens) == 0);
  // Non-vacuous: the context really was longer than the block, so the pre-D1
  // shape would have been a strictly larger query.
  CHECK(st.last_combined_key_rows > st.last_combined_query_rows);
}

// SPEC-DFLASH2 W13 (#2117), closing [#2112](https://github.com/mudler/vllm.cpp/issues/2112):
// THE COUNTERS ARE READABLE FROM A RUNNING ENGINE.
//
// WHY THIS CASE IS NOT A DUPLICATE OF THE W10 AND W12 CASES ABOVE. Those read
// `GetGraphDispatchStats()` and `GetDflashBlockRouteStats()` directly, in
// process, which is exactly the state #2112 names as the defect: every caller of
// both accessors is a test, so the numbers exist and no server can see them.
// #2112's own words are that a counter whose only reader is a test measures a
// class rather than a capability. This case reads the numbers the way a server
// operator has to read them -- off REAL fd 2, out of the production step loop --
// and it is red until something prints them.
//
// AND IT IS WHAT MAKES #2117 DECIDABLE. That issue's `## What would settle it`
// asks for `GraphDispatchStats` at c=4 and c=8 and marks the request BLOCKED on
// #2112. The `ragged_*` split in the line is the discriminator: #2117 predicts
// admission raggedness at 3% to 7% and separately warns that a share far above
// 10% means something other than admission, and a flat `ragged` count is
// consistent with both.
//
// The identity against the accessor is the load-bearing half. A line that
// printed plausible-looking constants would satisfy a `find()`; only the
// equality can say the line reports THIS run.
namespace {

// The last occurrence of `marker` in `captured`, to end of line. Empty when the
// marker never appears -- the caller REQUIREs presence separately, so an empty
// return can never be mistaken for a line whose fields are all zero.
std::string LastLineWith(const std::string& captured, const std::string& marker) {
  const size_t at = captured.rfind(marker);
  if (at == std::string::npos) return std::string();
  const size_t end = captured.find('\n', at);
  return captured.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

// `key` is matched WITH its `=`, so `ragged=` cannot match inside `ragged_mixed=`.
// Returns -1 when the key is absent, which no counter can legitimately be.
long long FieldFrom(const std::string& line, const std::string& key) {
  const size_t at = line.find(key);
  if (at == std::string::npos) return -1;
  return std::strtoll(line.c_str() + at + key.size(), nullptr, 10);
}

}  // namespace

TEST_CASE("dflash2 runner (W13): VT_GRAPH_STATS prints both counter families from the step loop") {
  vllm::v1::ResetGraphDispatchStats();
  vllm::detail::ResetDflashBlockRouteStats();
  setenv("VT_GRAPH_STATS", "1", 1);
  std::string threw, captured;
  // "hello world" is TWO tokens in this fixture's vocabulary and "hello" is one.
  // The difference is the whole reason for the argument: a one-token prefill is
  // uniform at query length 1 and never reaches the ragged arm, so the
  // classifier this case exists to prove reached would have nothing to classify.
  const std::vector<std::string> blocks =
      RunAndCollectDrafts(false, &threw, &captured, "hello world");
  unsetenv("VT_GRAPH_STATS");
  INFO("threw: ", threw);
  CHECK(threw.empty());
  // Steps ran at all, so there was something to report. Without this the two
  // REQUIREs below could go green on an engine that never decoded.
  REQUIRE_FALSE(blocks.empty());

  INFO("captured: ", captured);
  REQUIRE(captured.find("[graph-dispatch]") != std::string::npos);
  REQUIRE(captured.find("[dflash-route]") != std::string::npos);

  // The line describes THIS run: every field it names equals what the in-process
  // accessor holds after the run. At period 1 the last line is emitted on the
  // last step, so the two are equal and not merely ordered.
  const vllm::v1::GraphDispatchStats st = vllm::v1::GetGraphDispatchStats();
  const std::string line = LastLineWith(captured, "[graph-dispatch]");
  INFO("line: ", line);
  CHECK(FieldFrom(line, "steps=") == st.uniform_steps + st.ragged_steps);
  CHECK(FieldFrom(line, "uniform=") == st.uniform_steps);
  CHECK(FieldFrom(line, "uniform_spec=") == st.uniform_spec_steps);
  CHECK(FieldFrom(line, "ragged=") == st.ragged_steps);
  CHECK(FieldFrom(line, "ragged_mixed=") == st.ragged_mixed_steps);
  CHECK(FieldFrom(line, "ragged_prefill=") == st.ragged_prefill_only_steps);
  CHECK(FieldFrom(line, "ragged_spec=") == st.ragged_spec_only_steps);
  CHECK(FieldFrom(line, "spec_as_decode=") == st.spec_as_decode_steps);
  CHECK(FieldFrom(line, "qlen_cap_declines=") == st.qlen_cap_declines);
  // THE PARTITION, on production numbers rather than on hand-built ones. The
  // unit case pins the arithmetic; this pins that the runner feeds it once per
  // ragged step and not twice or never.
  CHECK(st.ragged_mixed_steps + st.ragged_prefill_only_steps +
            st.ragged_spec_only_steps ==
        st.ragged_steps);
  // NON-VACUITY for `ClassifyStepRows` itself, which is the one part of W13 a
  // field-equality check cannot reach: every equality above still holds if the
  // runner passed a default-constructed shape on every step, because then all
  // four numbers agree at `ragged_spec == ragged`. This engine's PREFILL step
  // carries rows wider than `drafts + 1` and no verify row, so a classifier that
  // ran with the step's real query lengths must have put it in the prefill-only
  // bucket. Deleting the `ClassifyStepRows` call at the runner reds exactly
  // this line.
  CHECK(st.ragged_prefill_only_steps > 0);

  // The DFlash lane's own family, which is the second half of what #2112 owes.
  // This fixture drives a DFlash2 draft, so the block lane RAN and a zero here
  // would mean the line is reporting a different process's counters.
  const vllm::detail::DflashBlockRouteStats rt = vllm::detail::GetDflashBlockRouteStats();
  const std::string rline = LastLineWith(captured, "[dflash-route]");
  INFO("rline: ", rline);
  // BOUNDED, not equal, and the asymmetry with the line above is a real property
  // of where each family is written. `[graph-dispatch]` is emitted from the same
  // step path that increments it, so the last line and the accessor agree
  // exactly. The route counters are incremented by the DRAFT phase, which runs
  // AFTER the step that printed the last line, so the accessor has advanced past
  // it by the calls of the final propose. Asserting equality there would pin the
  // draft phase's call count into a readout gate, which is a different row's
  // business; what this gate owes is that the numbers are this run's.
  CHECK(FieldFrom(rline, "paged_seam=") > 0);
  CHECK(FieldFrom(rline, "paged_seam=") <= rt.paged_seam_calls);
  CHECK(FieldFrom(rline, "block_kernel=") <= rt.block_kernel_calls);
  CHECK(FieldFrom(rline, "combined=") <= rt.materialized_combined_calls);
  CHECK(rt.paged_seam_calls + rt.block_kernel_calls + rt.materialized_combined_calls > 0);
}
