// CUDA-graph batch dispatch: which batches are graph-capturable, and under what
// shape key (SPEC-DSPARK W8, issue #442).
//
// Ported from vLLM @ 555967922:
//   v1/worker/gpu/cudagraph_utils.py:95-105  get_uniform_token_count
//   v1/cudagraph_dispatcher.py:37            uniform_decode_query_len
//   v1/cudagraph_dispatcher.py:143-146       the FULL-graph branch
//
// WHY THIS EXISTS. We mirrored vLLM's MODEL but not its graph DISPATCHER, and
// that single divergence is the measured DSpark parity gap. Upstream's "uniform"
// test is that every request in the batch shares a query_len -- NOT that the
// query_len is 1 -- and its captured decode length is DEFINED as the speculative
// verify shape (`1 + num_speculative_tokens`). So vLLM graphs the T=1+k verify by
// construction. Our gate is `pure_decode == (num_actual_tokens == num_reqs)`
// (qwen3_5_dense.cpp:159, qwen3_5_moe.cpp:128), which only ever matches
// query_len == 1, so our verify runs EAGER every step.
//
// Measured cost of that divergence (.agents/specs/dspark-spec-decode.md §6l/§6m):
// the 35B lane sits at 0.870x of the pinned graphed oracle where acceptance is
// high and the verify therefore runs every step, and 0.981x where acceptance is
// low. Our verify is ~32.0 ms of a 36.6 ms step; the oracle's ENTIRE step is
// ~31.8 ms.
//
// THIS HEADER WAS INERT WHEN IT LANDED AND IS NOT ANY MORE. #442 wired
// `IsUniformDecodeBatch` into the two Qwen3.5 registrations, and
// ENG-CUDAGRAPH-BREAK W6 (#1374) gives the header its RUNNER caller: the
// eligibility predicate moves out of the two model files that each re-derived
// it and into `GPUModelRunner::execute_model`, which ships the step's ACTUAL
// uniform query length on `ModelForwardInput::uniform_query_len`.
//
// The warning the first slice wrote for itself still stands and is why the key
// changed in the same commit: "flipping a predicate ALONE would be a bug: it
// would send a spec batch through a graph captured for query_len == 1". Both
// Qwen3.5 drivers key their slot ring on `(S, q, spec)` as of W6, so a widened
// predicate cannot reach a graph captured for a different shape.
#ifndef VLLM_V1_WORKER_GPU_CUDAGRAPH_DISPATCH_H_
#define VLLM_V1_WORKER_GPU_CUDAGRAPH_DISPATCH_H_

#include <cstdint>
#include <optional>
#include <vector>

namespace vllm {
namespace v1 {

// The query length a captured "decode" graph is built for.
//
// Upstream: `self.uniform_decode_query_len = 1 + num_speculative_tokens`
// (cudagraph_dispatcher.py:37). With speculation OFF this is 1, which is exactly
// today's pure-decode shape, so adopting it is a no-op for the non-spec path --
// that is what makes the eventual wiring safe to land incrementally.
inline int64_t UniformDecodeQueryLen(int64_t num_speculative_tokens) {
  return 1 + (num_speculative_tokens > 0 ? num_speculative_tokens : 0);
}

// The uniform token count of a batch, or nullopt when the batch is not uniform.
//
// Upstream get_uniform_token_count (cudagraph_utils.py:95-105): a batch is
// uniform iff every request carries the same query_len, tested as
// `max_query_len == num_tokens // num_reqs && num_tokens == max_query_len * num_reqs`.
// The second clause is not redundant -- integer division alone accepts a ragged
// batch whose total happens to floor to max_query_len (e.g. 2 reqs, lens 3 and 2,
// max 3, 5 // 2 == 2 != 3 rejects; but lens 4 and 2 with max 3 would floor to 3
// and pass without it).
inline std::optional<int64_t> UniformTokenCount(int64_t num_reqs, int64_t num_tokens,
                                                int64_t max_query_len) {
  if (num_reqs <= 0 || num_tokens <= 0 || max_query_len <= 0) return std::nullopt;
  if (max_query_len == num_tokens / num_reqs && num_tokens == max_query_len * num_reqs) {
    return max_query_len;
  }
  return std::nullopt;
}

// Whether this batch is a uniform DECODE batch for the configured speculation
// width, i.e. the shape a FULL graph is captured for.
//
// Upstream takes the FULL-graph branch when the batch is uniform and the mode
// offers FULL (cudagraph_dispatcher.py:143). `num_speculative_tokens == 0` makes
// this exactly today's pure-decode predicate; k > 0 additionally admits the
// T = num_reqs * (1 + k) speculative VERIFY, which is the batch we currently run
// eager and upstream captures.
inline bool IsUniformDecodeBatch(int64_t num_reqs, int64_t num_tokens,
                                 int64_t max_query_len,
                                 int64_t num_speculative_tokens) {
  const std::optional<int64_t> uniform =
      UniformTokenCount(num_reqs, num_tokens, max_query_len);
  return uniform.has_value() &&
         *uniform == UniformDecodeQueryLen(num_speculative_tokens);
}

// The request count a captured graph of `num_tokens_padded` tokens serves.
//
// Upstream: `num_reqs = min(num_tokens_padded // uniform_decode_query_len,
// max_num_seqs)` with `assert num_tokens_padded % uniform_decode_query_len == 0`
// (cudagraph_dispatcher.py:144-145). Returns nullopt where upstream would trip
// that assert, so a caller cannot silently capture a shape the padding never
// produces.
inline std::optional<int64_t> UniformDecodeNumReqs(int64_t num_tokens_padded,
                                                   int64_t num_speculative_tokens,
                                                   int64_t max_num_seqs) {
  const int64_t q = UniformDecodeQueryLen(num_speculative_tokens);
  if (num_tokens_padded <= 0 || max_num_seqs <= 0) return std::nullopt;
  if (num_tokens_padded % q != 0) return std::nullopt;
  const int64_t reqs = num_tokens_padded / q;
  return reqs < max_num_seqs ? reqs : max_num_seqs;
}

// The step's ACTUAL uniform query length, or nullopt when no captured decode
// graph can serve the step. ENG-CUDAGRAPH-BREAK W6 (#1374), closing
// [#1020](https://github.com/mudler/vllm.cpp/issues/1020).
//
// WHAT CHANGED AND WHY. `IsUniformDecodeBatch` above compares the batch's
// uniform length against `1 + num_speculative_tokens`, the CONFIGURED width,
// which is a constant for the engine's lifetime. The scheduler clamps a
// request's drafts to what the step's token budget fits
// (`v1/core/sched/scheduler.cpp:616-622`), so at k > 1 a clamped step can hand
// every request the same SHORTER prefix -- two drafts at k=3, uniform at query
// length 3 against a predicate expecting 4. That batch is exactly the shape a
// graph can serve and it got none, with no log and no counter. This function
// reads the length the step actually has instead.
//
// The upper bound is kept, because it is not the same test. A batch uniform at
// a length ABOVE `1 + k` is not a verify step at all -- it is a prefill or a
// chunked batch wearing a uniform shape -- and no decode driver in this tree
// captures one. Mirrors vLLM, which dispatches the FULL branch only for a batch
// whose uniform length is the configured decode length
// (`cudagraph_dispatcher.py:143`); the widening below is strictly inside that
// bound, never past it.
inline std::optional<int64_t> ActualUniformDecodeQueryLen(
    int64_t num_reqs, int64_t num_tokens, int64_t max_query_len,
    int64_t num_speculative_tokens) {
  const std::optional<int64_t> uniform =
      UniformTokenCount(num_reqs, num_tokens, max_query_len);
  if (!uniform.has_value()) return std::nullopt;
  if (*uniform > UniformDecodeQueryLen(num_speculative_tokens)) return std::nullopt;
  return uniform;
}

// THE STEP'S GRAPH-ELIGIBLE QUERY LENGTH: the shape test above, plus the
// conjunct that keeps a batch which is uniform by ARITHMETIC but is not a
// speculative VERIFY out of a decode capture. This is the function the runner
// calls; `ActualUniformDecodeQueryLen` is its shape half.
//
// WHY THE SHAPE IS NOT ENOUGH, MEASURED RATHER THAN REASONED. A single request
// prefilling three tokens is uniform at query length 3 by every arithmetic test
// vLLM applies, and at k >= 2 it sits inside the `1 + k` bound. Admitting it
// hands a prefill to a driver whose `BuildPaddedDecode` rewrites the metadata as
// S single-token requests, which is not what a multi-token-per-request batch is.
// On this tree's CPU spec fixture the bare shape test reported 27 "uniform spec"
// steps out of 28 on a 29-token run before this conjunct existed.
//
// So a length above 1 is admitted only when EVERY request in the step is
// verifying at exactly `q - 1` drafts, read off the scheduler's own per-request
// draft counts (`ExecuteModelStep::num_draft_tokens_per_req`). That is strictly
// narrower than the shape test and never wider, which is the polarity
// `## Work breakdown` W6 requires: a predicate must not admit a step no driver
// can serve.
//
// `drafts_per_req` is EMPTY on every step the scheduler recorded no drafts for,
// and then nothing above 1 is admitted at all.
inline std::optional<int64_t> GraphEligibleQueryLen(
    int64_t num_reqs, int64_t num_tokens, int64_t max_query_len,
    int64_t num_speculative_tokens,
    const std::vector<int32_t>& drafts_per_req) {
  const std::optional<int64_t> q = ActualUniformDecodeQueryLen(
      num_reqs, num_tokens, max_query_len, num_speculative_tokens);
  if (!q.has_value() || *q == 1) return q;
  if (static_cast<int64_t>(drafts_per_req.size()) != num_reqs) {
    return std::nullopt;
  }
  for (const int32_t drafts : drafts_per_req) {
    if (static_cast<int64_t>(drafts) + 1 != *q) return std::nullopt;
  }
  return q;
}

// Dispatch observability. The sibling of `vt::GraphBreakStats`, and it exists
// for the same reason W3 learned the hard way: the predicate is invisible from
// outside the runner, a token gate cannot see which arm a step took, and
// [#1020](https://github.com/mudler/vllm.cpp/issues/1020) is titled on the word
// SILENTLY. A step that falls out of the graph now moves a number.
//
// Counted in `GPUModelRunner::execute_model`, once per step, on the shared path
// every registered model reaches -- not per model, because a per-model counter
// would report the models that happen to have a driver rather than the steps
// the predicate refused.
struct GraphDispatchStats {
  // The predicate found a uniform decode shape and named its query length. This
  // is the step population any decode driver may admit; whether a given model
  // has a driver for it is the model's own question.
  int64_t uniform_steps = 0;
  // Uniform at a query length ABOVE 1 -- the population #1020 is about. Zero on
  // every non-speculative engine, so a moved number is proof the widened arm
  // was reached rather than an inference from the total.
  int64_t uniform_spec_steps = 0;
  // Uniform at a length STRICTLY BETWEEN 1 and the configured `1 + k`. This is
  // the [#1020](https://github.com/mudler/vllm.cpp/issues/1020) population
  // exactly: a verify step the scheduler clamped to a shorter draft prefix,
  // still perfectly uniform, and refused by the old predicate for no reason
  // other than that it compared against a constant. It is a SEPARATE counter
  // from `uniform_spec_steps` because the total moves on an unclamped engine
  // too, so only this one can witness the widening.
  int64_t clamped_spec_steps = 0;
  // No uniform query length exists for this step: prefill, mixed, or ragged.
  // No decode graph in this tree can serve one, which is the coverage the
  // PIECEWISE arm would have to reach and does not (spec `## Owed`).
  int64_t ragged_steps = 0;
  // DRIVER-side, and the two halves are deliberately in one struct: a widened
  // predicate that admits a step the driver then declines has changed nothing,
  // and reading the two numbers from two places is how that goes unnoticed.
  //
  // Distinct `(S, q, spec)` shapes a decode driver has opened a ring for. This
  // is the number the W6 widening moves: before it, a speculative engine could
  // reach exactly one query length, so one shape per padded size.
  int64_t capture_shapes = 0;
  // Steps refused because capturing them would pass the bound on distinct
  // speculative query lengths (`VT_SPEC_GRAPH_MAX_QLENS`). Such a step runs
  // eager, which is what it did before W6 -- the difference is that it is now
  // counted rather than silent.
  int64_t qlen_cap_declines = 0;
};
GraphDispatchStats GetGraphDispatchStats();
void ResetGraphDispatchStats();
// Records one step's dispatch decision. `query_len` is the value
// `ActualUniformDecodeQueryLen` returned, or 0 for "no uniform length";
// `configured_query_len` is `UniformDecodeQueryLen(num_speculative_tokens)`,
// which is what separates a clamped verify from a full-depth one.
void NoteGraphDispatch(int64_t query_len, int64_t configured_query_len);
// Records one newly opened decode-graph capture shape.
void NoteDecodeGraphShape();
// Records one step refused by the distinct-query-length bound.
void NoteDecodeGraphQueryLenDecline();

}  // namespace v1
}  // namespace vllm

#endif  // VLLM_V1_WORKER_GPU_CUDAGRAPH_DISPATCH_H_
