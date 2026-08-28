// SPEC-DFLASH2 ([#2042](https://github.com/mudler/vllm.cpp/issues/2042)) —
// prefix caching and the DFlash2 draft context, gated through the PRODUCTION
// async front.
//
// WHAT #2042 MEASURED. On `3d895a202`, `sm_121a`, dgx:gpu0 under an `rc` lease,
// a DFlash2 k=8 server serving 1024 in / 512 out answered 8/8 at concurrency 1
// with `--no-enable-prefix-caching`. The same binary with
// `--enable-prefix-caching --scheduling-policy lpm` threw
//
//     vt: propose_drafts_block: context position discontinuity (accumulation
//         out of sync with the target's committed positions)
//
// from inside the EngineCore step on the FIRST request that took a cache hit,
// and every later request came back `[request submitted to a stopped AsyncLLM]`.
// The rung read `ok=0 failed=8`. So, exactly as in #1919, the defect is not that
// one request failed: it is that the ENGINE died, and every case here therefore
// asserts a request issued AFTER the one that trips the condition.
//
// WHY THE INVARIANT IS THE DETECTOR AND NOT THE DEFECT. With prefix caching on,
// the scheduler admits a request with `num_computed_tokens` already equal to the
// cached prefix length, and the worker turns that straight into absolute
// positions (`prepare_inputs.cpp`, `positions[t] = num_computed_tokens_cpu[r] +
// query_pos[t]`). The TARGET is served from cache and never runs those tokens,
// so no aux hidden state exists for them, so nothing is ever projected into the
// draft's private context store: it holds ZERO rows while the target has
// committed N. Letting the propose run would speculate off an empty context
// while claiming an N-token one — well-formed drafts, lossless verify, only
// ACCEPTANCE falls, which is the invisible-defect class this row exists to
// remove.
//
// Upstream never reaches that state because it keeps no private store. Its
// DFlash draft writes the context K/V into the engine's own paged KV cache
// (`vllm/model_executor/models/qwen3_dflash.py:601-619` at pin `5559679229`) on
// a slot mapping built from the TARGET's block table
// (`vllm/v1/spec_decode/dflash.py:145-153`), so a prefix hit hands it the draft
// context for free. Moving our store there is owed under #1919. Until then the
// mirrored answer for a request the speculator cannot serve is an EMPTY draft
// and the target running alone (`vllm/v1/spec_decode/ngram_proposer.py:150-159`,
// `vllm/v1/spec_decode/suffix_decoding.py:55-62`), which is #1919's fallback
// reached from a second place.
#include <doctest/doctest.h>

#include "dflash2_runner_fixture.h"

#include "vllm/v1/engine/async_llm.h"

namespace {

// `VT_SPEC_TRACE` is latched by a function-local `static` on the FIRST propose
// in the process, so it has to be set before any case runs. Level 1 prints the
// `first=[...]` payload `DraftedBlocks` counts, and that count is how these
// cases MEASURE whether a request speculated rather than asserting it.
const bool kSpecTraceEnabled = [] {
  ::setenv("VT_SPEC_TRACE", "1", 1);
  return true;
}();

// A prefix-cache hit needs at least one FULL KV block, and `EngineParams`'
// default block size is 32 tokens. A 64-token prompt therefore caches one whole
// block on the first request and hits exactly 32 tokens on the second (the
// matcher stops one token short of the prompt, then floors to the block), which
// is the `first_pos > 0` the fix classifies. A prompt at the shared
// `kMaxModelLen = 32` cannot produce a hit at all and would gate nothing.
constexpr int kModelLen = 256;
constexpr int kPromptTokens = 64;
constexpr int kOutTokens = 4;

// The per-request drain budget, generous on purpose: a slow CPU runner must not
// turn a pass into a timeout.
constexpr int kDrainBudgetSeconds = 60;

// One token id of the tiny BPE fixture. The prompts go in PRE-TOKENIZED so the
// context length under test is an exact number.
constexpr int32_t kTok = 13;

// One request's outcome, with the three states kept apart. `AsyncLLM::generate`
// collapses two of them: on the pre-fix tree the in-flight request's collector
// NEVER receives a terminal output, because the EngineCore that would have
// produced it is dead, so a blocking driver HANGS instead of reporting. A test
// that hangs has measured nothing.
struct RequestOutcome {
  bool finished = false;
  bool timed_out = false;
  std::string threw;
  std::vector<int32_t> tokens;
};

std::string Describe(const RequestOutcome& o) {
  if (!o.threw.empty()) return "threw: " + o.threw;
  if (o.timed_out)
    return "NO TERMINAL OUTPUT within " + std::to_string(kDrainBudgetSeconds) +
           "s (the engine died mid-step and this request's collector was never "
           "completed)";
  if (o.finished) return "finished with " + std::to_string(o.tokens.size()) + " tokens";
  return "neither finished, timed out, nor threw";
}

RequestOutcome DrainRequest(vllm::v1::AsyncLLM& async, vllm::v1::AsyncRequest& req,
                            const std::string& req_id) {
  RequestOutcome o;
  try {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(kDrainBudgetSeconds);
    for (;;) {
      std::optional<vllm::RequestOutput> out =
          async.get_output_for(req, std::chrono::milliseconds(200));
      if (out.has_value()) {
        if (out->finished) {
          o.finished = true;
          if (!out->outputs.empty()) o.tokens = out->outputs[0].token_ids;
          return o;
        }
        continue;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        o.timed_out = true;
        try {
          async.abort(req_id);
        } catch (const std::exception&) {
          // A dead engine can refuse the abort too; the timeout is the finding.
        }
        return o;
      }
    }
  } catch (const std::exception& e) {
    o.threw = e.what();
  }
  return o;
}

RequestOutcome DrainWithDeadline(vllm::v1::AsyncLLM& async, std::vector<int32_t> prompt,
                                 int max_tokens, const std::string& req_id) {
  try {
    vllm::v1::AsyncRequest req =
        async.add_request(req_id, std::move(prompt), Greedy(max_tokens));
    return DrainRequest(async, req, req_id);
  } catch (const std::exception& e) {
    RequestOutcome o;
    o.threw = e.what();
    return o;
  }
}

struct TwoRequestRun {
  RequestOutcome first;
  RequestOutcome second;
  std::vector<std::string> blocks;
  std::string stderr_text;
};

// ONE engine, the SAME prompt twice. The second request is the one under test:
// its whole prompt is in the cache when prefix caching is on, so it is admitted
// mid-sequence and its draft context starts empty.
TwoRequestRun RunSamePromptTwice(bool enable_prefix_caching) {
  TwoRequestRun r;
  const HfConfig target = MakeDenseConfig(kModelLen);
  const ScratchDraftDir dir;
  const std::vector<int32_t> prompt(kPromptTokens, kTok);
  r.stderr_text = CaptureStderr([&] {
    EngineParams params = DflashSpecParams(dir, kModelLen, /*max_num_seqs=*/1,
                                           /*max_num_batched_tokens=*/8192,
                                           /*num_blocks=*/512);
    params.enable_prefix_caching = enable_prefix_caching;
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(), params,
                     MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false));
    REQUIRE(eng.prefix_caching_enabled() == enable_prefix_caching);
    r.first = DrainWithDeadline(eng.async_engine(), prompt, kOutTokens, "req-first");
    r.second = DrainWithDeadline(eng.async_engine(), prompt, kOutTokens, "req-second");
  });
  r.blocks = DraftedBlocks(r.stderr_text);
  return r;
}


// TWO requests in flight at once on a two-row batch, the first finishing FIRST.
// `InputBatch::condense` then closes the hole row 0 leaves by moving the request
// at row 1 down into it, and nothing else is admitted that step to fill it. This
// is the batch move the four DFlash arrays do not follow, and it needs no prefix
// caching at all: it is a second, independent way for a request's draft context
// to go missing under it.
TwoRequestRun RunTwoConcurrent(bool enable_prefix_caching) {
  TwoRequestRun r;
  const HfConfig target = MakeDenseConfig(kModelLen);
  const ScratchDraftDir dir;
  const std::vector<int32_t> prompt(kPromptTokens, kTok);
  r.stderr_text = CaptureStderr([&] {
    EngineParams params = DflashSpecParams(dir, kModelLen, /*max_num_seqs=*/2,
                                           /*max_num_batched_tokens=*/8192,
                                           /*num_blocks=*/512);
    params.enable_prefix_caching = enable_prefix_caching;
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(), params,
                     MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false));
    vllm::v1::AsyncLLM& async = eng.async_engine();
    // Both in flight before either drains, so they share the batch. The SHORT
    // one is admitted first and so takes row 0; it finishes first, and the long
    // one is then moved down into its row.
    vllm::v1::AsyncRequest a =
        async.add_request("req-short", prompt, Greedy(/*max_tokens=*/1));
    vllm::v1::AsyncRequest b =
        async.add_request("req-long", prompt, Greedy(/*max_tokens=*/24));
    r.first = DrainRequest(async, a, "req-short");
    r.second = DrainRequest(async, b, "req-long");
  });
  r.blocks = DraftedBlocks(r.stderr_text);
  return r;
}

}  // namespace

// ─── G1: #2042's repro — the engine survives a prefix-cache hit ─────────────
TEST_CASE("dflash2 prefix cache (#2042): a cache hit does NOT kill the engine") {
  const TwoRequestRun on = RunSamePromptTwice(/*enable_prefix_caching=*/true);

  // THE ASSERTION. Pre-fix the second request never completes at all: the
  // position invariant takes EngineCore down mid-step and this request's
  // collector waits for a terminal frame nothing will ever write.
  INFO("second request: ", Describe(on.second));
  CHECK(on.second.finished);
  CHECK(on.second.threw.empty());
  CHECK_FALSE(on.second.timed_out);
  CHECK_FALSE(on.second.tokens.empty());

  INFO("first request: ", Describe(on.first));
  CHECK(on.first.finished);
  CHECK_FALSE(on.first.tokens.empty());

  // And nothing anywhere in the run is still refusing by that name.
  CHECK(on.stderr_text.find("context position discontinuity") == std::string::npos);
}

// ─── G2: the tokens do not move ─────────────────────────────────────────────
TEST_CASE("dflash2 prefix cache (#2042): dropping the draft changes drafts, not output") {
  const TwoRequestRun on = RunSamePromptTwice(/*enable_prefix_caching=*/true);
  const TwoRequestRun off = RunSamePromptTwice(/*enable_prefix_caching=*/false);

  REQUIRE(on.first.finished);
  REQUIRE(on.second.finished);
  REQUIRE(off.first.finished);
  REQUIRE(off.second.finished);

  // The verify is lossless, so a request that stops speculating changes what was
  // DRAFTED and not what was EMITTED. This is the half a liveness check cannot
  // see: a fallback that also moved the output would be a correctness bug
  // wearing a throughput cost.
  CHECK(on.first.tokens == off.first.tokens);
  CHECK(on.second.tokens == off.second.tokens);

  // And prefix caching's own contract: the same prompt under greedy sampling
  // emits the same tokens whether its prefix was computed or reused.
  CHECK(on.first.tokens == on.second.tokens);
}

// ─── G3: the fallback is SCOPED to the request that hit the cache ───────────
TEST_CASE("dflash2 prefix cache (#2042): the cache MISS still speculates") {
  const TwoRequestRun on = RunSamePromptTwice(/*enable_prefix_caching=*/true);
  const TwoRequestRun off = RunSamePromptTwice(/*enable_prefix_caching=*/false);

  // MEASURED, not asserted about. A fix that simply stopped speculating whenever
  // prefix caching is on would pass G1 and G2 and be invisible to a token gate,
  // because the verify is lossless and only acceptance falls. The prefix-caching
  // arm must propose STRICTLY FEWER blocks than the control and still propose
  // some: the first request is a cache MISS and speculates the whole way.
  INFO("blocks on: ", on.blocks.size(), " blocks off: ", off.blocks.size());
  CHECK_FALSE(off.blocks.empty());
  CHECK_FALSE(on.blocks.empty());
  CHECK(on.blocks.size() < off.blocks.size());

  // And the control speculates for both requests, so nothing about the fixture
  // itself is suppressing the second one.
  CHECK(off.stderr_text.find("was admitted with") == std::string::npos);
}

// ─── G4: the notice names the request, prefix caching, and fires ONCE ───────
TEST_CASE("dflash2 prefix cache (#2042): the dropped speculation is announced") {
  const TwoRequestRun on = RunSamePromptTwice(/*enable_prefix_caching=*/true);

  INFO("stderr: ", on.stderr_text);
  const size_t at = on.stderr_text.find("was admitted with");
  REQUIRE(at != std::string::npos);
  // Once. A per-step message would drown a decode loop.
  CHECK(on.stderr_text.find("was admitted with", at + 1) == std::string::npos);

  // It names the request that hit the cache, and not the one that missed.
  const size_t prev_nl = on.stderr_text.rfind('\n', at);
  const size_t line_start = (prev_nl == std::string::npos) ? 0 : prev_nl + 1;
  const size_t line_end = on.stderr_text.find('\n', at);
  const std::string line = on.stderr_text.substr(
      line_start, (line_end == std::string::npos) ? std::string::npos
                                                  : line_end - line_start);
  CHECK(line.find("req-second") != std::string::npos);
  CHECK(line.find("req-first") == std::string::npos);

  // It names the cause and the consequence, so an operator reading a server log
  // can tell this apart from #1919's capacity fallback.
  CHECK(line.find("prefix-cache") != std::string::npos);
  CHECK(line.find("WITHOUT speculation") != std::string::npos);
  CHECK(line.find("#2042") != std::string::npos);

  // It is NOT the capacity fallback: nothing here outgrew anything.
  CHECK(on.stderr_text.find("has outgrown the draft speculative context") ==
        std::string::npos);
}

// ─── G5: the CONTROL — a moved request is not a cache hit ──────────────────
TEST_CASE("dflash2 prefix cache (#2042): a request the batch MOVED is not classified as a cache hit") {
  const TwoRequestRun r = RunTwoConcurrent(/*enable_prefix_caching=*/false);

  // THIS IS THE ANTI-MASKING GATE, and it is the reason this fix could not have
  // been written before #2008/#2010.
  //
  // Prefix caching is OFF here, so every request starts at position 0 and the
  // ONLY way a request can meet the propose with context missing under it is
  // `InputBatch::condense` sliding it into a finished neighbour's row. Under the
  // old row-indexed state that presented to `propose_drafts_block` exactly as a
  // prefix-cache hit does — no context, target already past 0 — so the fallback
  // above would have swallowed #2008's crash and turned it into an acceptance
  // loss nothing can see. #2010 keys the context by REQUEST ID, which makes the
  // moved request keep its entry, so the classification is reached only by a
  // genuinely first-seen request.
  //
  // The engine surviving this is #2010's guarantee, gated by
  // `test_dflash2_concurrency`. What is gated HERE is the part that belongs to
  // #2042: that no fallback of ours is announced for it.
  INFO("short request: ", Describe(r.first));
  CHECK(r.first.finished);
  INFO("long request: ", Describe(r.second));
  CHECK(r.second.finished);
  CHECK(r.second.threw.empty());
  CHECK_FALSE(r.second.timed_out);
  CHECK_FALSE(r.second.tokens.empty());
  CHECK(r.stderr_text.find("context position discontinuity") == std::string::npos);

  // THE ASSERTION THIS CASE EXISTS FOR: the moved request KEPT its context
  // rather than being quietly dropped to the target. Deleting the freshness gate
  // from the classification above turns this one line red and nothing else,
  // which is what makes the masking executable rather than argued.
  CHECK(r.stderr_text.find("was admitted with") == std::string::npos);
  CHECK(r.stderr_text.find("has outgrown the draft speculative context") ==
        std::string::npos);
  CHECK_FALSE(r.blocks.empty());
}
