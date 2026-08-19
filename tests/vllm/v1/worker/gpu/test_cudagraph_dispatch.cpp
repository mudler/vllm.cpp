// The graph-dispatch predicate ported from vLLM (SPEC-DSPARK W8, issue #442).
//
// These cases pin the ONE divergence that the DSpark parity measurement blamed:
// upstream's captured decode shape is `1 + num_speculative_tokens`, so the T=1+k
// speculative VERIFY is graph-capturable by construction, while our
// `pure_decode == (num_actual_tokens == num_reqs)` gate only ever admits
// query_len == 1 and sends the verify down the eager path every step.
//
// Upstream anchors @ 555967922: cudagraph_utils.py:95-105 (get_uniform_token_count),
// cudagraph_dispatcher.py:37 (uniform_decode_query_len), :143-146 (FULL branch).
#include <doctest/doctest.h>

#include <cstdint>

#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"  // internal header, from src/

using namespace vllm::v1;

TEST_CASE("the captured decode length IS the speculative verify shape") {
  // cudagraph_dispatcher.py:37. This is the whole finding: with k speculative
  // tokens the graph is built for 1+k, not for 1.
  CHECK(UniformDecodeQueryLen(0) == 1);   // speculation off == today's pure decode
  CHECK(UniformDecodeQueryLen(8) == 9);   // 35B lane, k=8
  CHECK(UniformDecodeQueryLen(15) == 16); // 27B lane, k=15
}

TEST_CASE("a batch is uniform when every request shares a query_len") {
  // 2 requests x 9 tokens: the 35B verify batch. Our pure_decode predicate
  // (num_tokens == num_reqs) REJECTS this, which is the bug being ported out.
  CHECK(UniformTokenCount(/*num_reqs=*/2, /*num_tokens=*/18, /*max_query_len=*/9) == 9);
  CHECK(UniformTokenCount(1, 9, 9) == 9);
  CHECK(UniformTokenCount(4, 4, 1) == 1);  // ordinary decode stays uniform

  // Ragged batches are not uniform.
  CHECK_FALSE(UniformTokenCount(2, 5, 3).has_value());   // lens 3 + 2
  CHECK_FALSE(UniformTokenCount(3, 10, 5).has_value());  // 10 != 5*3

  CHECK(UniformTokenCount(2, 6, 3) == 3);  // 2 x 3 IS uniform

  // The SECOND clause of upstream's test is load-bearing, not redundant: 3
  // requests totalling 8 tokens with max_query_len 2 passes the division check
  // (8 / 3 == 2 == max_query_len) yet is ragged (2+2+4 say), and only
  // `num_tokens == max_query_len * num_reqs` (8 != 6) rejects it. Dropping that
  // clause would dispatch a ragged batch into a uniform graph.
  CHECK_FALSE(UniformTokenCount(3, 8, 2).has_value());

  CHECK_FALSE(UniformTokenCount(0, 4, 1).has_value());   // degenerate inputs
  CHECK_FALSE(UniformTokenCount(2, 0, 1).has_value());
  CHECK_FALSE(UniformTokenCount(2, 4, 0).has_value());
}

TEST_CASE("the speculative VERIFY batch is a uniform decode batch, and ours is not") {
  // THE REGRESSION THIS FILE EXISTS FOR. 1 request, k=8 -> 9 tokens, query_len 9.
  CHECK(IsUniformDecodeBatch(/*num_reqs=*/1, /*num_tokens=*/9, /*max_query_len=*/9,
                             /*num_spec=*/8));
  CHECK(IsUniformDecodeBatch(2, 18, 9, 8));    // 2 requests, same shape
  CHECK(IsUniformDecodeBatch(1, 16, 16, 15));  // 27B lane, k=15

  // Today's predicate would only admit these:
  CHECK(IsUniformDecodeBatch(4, 4, 1, /*num_spec=*/0));

  // A verify batch under a DIFFERENT k than configured is not the captured
  // shape, so it must not be dispatched to that graph.
  CHECK_FALSE(IsUniformDecodeBatch(1, 9, 9, /*num_spec=*/15));
  CHECK_FALSE(IsUniformDecodeBatch(1, 16, 16, /*num_spec=*/8));

  // A pure-decode batch while speculating is NOT the captured 1+k shape either
  // (upstream keys the graph on the uniform length, so this falls elsewhere).
  CHECK_FALSE(IsUniformDecodeBatch(4, 4, 1, /*num_spec=*/8));

  // Mixed prefill+decode is never uniform.
  CHECK_FALSE(IsUniformDecodeBatch(2, 40, 39, 8));
}

TEST_CASE("request count for a padded capture, and the shapes upstream refuses") {
  // cudagraph_dispatcher.py:144-145.
  CHECK(UniformDecodeNumReqs(/*padded=*/18, /*num_spec=*/8, /*max_num_seqs=*/8) == 2);
  CHECK(UniformDecodeNumReqs(9, 8, 8) == 1);
  CHECK(UniformDecodeNumReqs(64, 0, 8) == 8);  // clamped to max_num_seqs

  // Upstream asserts padded % query_len == 0; we return nullopt rather than
  // capture a shape the padding never produces.
  CHECK_FALSE(UniformDecodeNumReqs(10, 8, 8).has_value());
  CHECK_FALSE(UniformDecodeNumReqs(0, 8, 8).has_value());
  CHECK_FALSE(UniformDecodeNumReqs(9, 8, 0).has_value());
}

// ───────────────────────────────────────────────────────────────────────────
// ENG-CUDAGRAPH-BREAK W6 (#1374): the predicate reads the step's ACTUAL uniform
// query length. [#1020](https://github.com/mudler/vllm.cpp/issues/1020).
//
// These are the arithmetic cases. They say what the function computes and never
// that a step reaches it; the reachability gate is
// `tests/vllm/v1/spec_decode/test_mtp_depth.cpp` ("W6: a CLAMPED spec verify is
// graph-eligible at its actual depth"), which drives a real engine.
TEST_CASE("W6: a CLAMPED verify keeps its actual query length") {
  // THE #1020 CASE. k=3, so the configured shape is 4. A step every request
  // entered with TWO drafts is uniform at 3, which `IsUniformDecodeBatch`
  // rejects against the configured width and this function reports.
  CHECK_FALSE(IsUniformDecodeBatch(/*num_reqs=*/2, /*num_tokens=*/6,
                                   /*max_query_len=*/3, /*num_spec=*/3));
  CHECK(ActualUniformDecodeQueryLen(2, 6, 3, 3) == 3);

  // Full depth is unchanged: both agree, which is what makes the widening
  // additive rather than a replacement.
  CHECK(IsUniformDecodeBatch(2, 8, 4, 3));
  CHECK(ActualUniformDecodeQueryLen(2, 8, 4, 3) == 4);

  // Every clamped depth between the two, and the degenerate one at the bottom:
  // a step with no drafts left is an ordinary decode and reports 1.
  CHECK(ActualUniformDecodeQueryLen(4, 8, 2, 3) == 2);
  CHECK(ActualUniformDecodeQueryLen(4, 4, 1, 3) == 1);

  // THE UPPER BOUND IS KEPT, and it is not the same test as the equality it
  // replaces. A batch uniform ABOVE `1 + k` is not a verify step -- it is a
  // prefill or a chunked batch wearing a uniform shape -- and no decode driver
  // in this tree captures one. Mirrors vLLM, whose FULL branch dispatches only
  // at the configured decode length (cudagraph_dispatcher.py:143).
  CHECK_FALSE(ActualUniformDecodeQueryLen(1, 9, 9, /*num_spec=*/3).has_value());
  CHECK_FALSE(ActualUniformDecodeQueryLen(1, 2, 2, /*num_spec=*/0).has_value());

  // Speculation OFF collapses it to exactly today's pure-decode shape, which is
  // what makes every non-speculative engine byte-identical across this change.
  CHECK(ActualUniformDecodeQueryLen(4, 4, 1, /*num_spec=*/0) == 1);

  // A ragged batch has no uniform length at any configured width, so no graph
  // can serve it and the function says so rather than picking one.
  CHECK_FALSE(ActualUniformDecodeQueryLen(2, 5, 3, 3).has_value());
  CHECK_FALSE(ActualUniformDecodeQueryLen(3, 8, 2, 3).has_value());
  CHECK_FALSE(ActualUniformDecodeQueryLen(0, 4, 1, 3).has_value());
}

TEST_CASE("W6: the dispatch counters separate the clamped population") {
  // #1020 is titled on the word SILENTLY, so the counters are part of the fix
  // rather than decoration. This case pins which bucket each step lands in.
  ResetGraphDispatchStats();
  const int64_t configured = UniformDecodeQueryLen(3);  // == 4
  CHECK(configured == 4);

  NoteGraphDispatch(/*query_len=*/1, configured);  // ordinary decode
  NoteGraphDispatch(4, configured);                // full-depth verify
  NoteGraphDispatch(3, configured);                // CLAMPED verify, the #1020 one
  NoteGraphDispatch(2, configured);                // clamped harder
  NoteGraphDispatch(0, configured);                // prefill / mixed / ragged

  const GraphDispatchStats s = GetGraphDispatchStats();
  CHECK(s.uniform_steps == 4);
  CHECK(s.uniform_spec_steps == 3);
  CHECK(s.clamped_spec_steps == 2);
  CHECK(s.ragged_steps == 1);
  // The buckets SUM to the steps recorded, or one of them is double-counting.
  CHECK(s.uniform_steps + s.ragged_steps == 5);
  // And the spec buckets nest, which is what says `clamped` is a SUBSET rather
  // than a parallel count.
  CHECK(s.clamped_spec_steps <= s.uniform_spec_steps);
  CHECK(s.uniform_spec_steps <= s.uniform_steps);

  ResetGraphDispatchStats();
  const GraphDispatchStats z = GetGraphDispatchStats();
  CHECK(z.uniform_steps == 0);
  CHECK(z.uniform_spec_steps == 0);
  CHECK(z.clamped_spec_steps == 0);
  CHECK(z.ragged_steps == 0);
  CHECK(z.capture_shapes == 0);
  CHECK(z.qlen_cap_declines == 0);
}

// THE VERIFY CONJUNCT, gated on the function because no engine in this tree can
// reach it. Both models that read `uniform_query_len` are GDN hybrids, so their
// prefill steps carry `gdn_meta.num_prefill_tokens > 0` and the runner's FIRST
// conjunct refuses them before this one is consulted. Measured, not assumed:
// deleting the per-request test left `test_mtp_depth` GREEN at 104/104. It is
// defence in depth for the next model that reads the field, and the spec's
// `## Owed` says so -- so it is gated HERE, where a mutation can move it.
TEST_CASE("W6: a batch uniform by arithmetic is not automatically a verify") {
  const std::vector<int32_t> kNoDrafts;

  // THE PREFILL. One request, three tokens, k=3. Uniform at 3 and inside the
  // bound, so the shape half admits it; no request is verifying, so the step
  // half does not.
  CHECK(ActualUniformDecodeQueryLen(1, 3, 3, /*num_spec=*/3) == 3);
  CHECK_FALSE(
      GraphEligibleQueryLen(1, 3, 3, /*num_spec=*/3, kNoDrafts).has_value());

  // A REAL VERIFY at the same shape: one request carrying two drafts.
  CHECK(GraphEligibleQueryLen(1, 3, 3, /*num_spec=*/3, {2}) == 3);
  // ... and at full depth.
  CHECK(GraphEligibleQueryLen(2, 8, 4, /*num_spec=*/3, {3, 3}) == 4);

  // A MIXED step whose arithmetic happens to be uniform: both requests carry
  // three tokens, but only one of them is verifying at two drafts. Admitting it
  // would capture a shape half the batch does not have.
  CHECK_FALSE(GraphEligibleQueryLen(2, 6, 3, /*num_spec=*/3, {2, 0}).has_value());
  CHECK_FALSE(GraphEligibleQueryLen(2, 6, 3, /*num_spec=*/3, {1, 2}).has_value());
  // A count array that does not cover the batch says nothing about it.
  CHECK_FALSE(GraphEligibleQueryLen(2, 6, 3, /*num_spec=*/3, {2}).has_value());

  // QUERY LENGTH 1 NEVER CONSULTS THE DRAFT COUNTS, because that arm is
  // `pure_decode` and every driver already serves it. A step with no recorded
  // drafts still reports 1.
  CHECK(GraphEligibleQueryLen(4, 4, 1, /*num_spec=*/3, kNoDrafts) == 1);
  CHECK(GraphEligibleQueryLen(4, 4, 1, /*num_spec=*/0, kNoDrafts) == 1);

  // And the refusals the shape half already owns survive composition.
  CHECK_FALSE(GraphEligibleQueryLen(2, 5, 3, 3, {2, 2}).has_value());  // ragged
  CHECK_FALSE(GraphEligibleQueryLen(1, 9, 9, 3, {8}).has_value());     // over 1+k
}
