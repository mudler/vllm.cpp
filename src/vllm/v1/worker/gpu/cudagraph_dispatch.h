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
#include <string>
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

// SPEC-DFLASH2 W13 (#2117): THE SHAPE OF A STEP, reduced to the two bits that
// say why it was ragged.
//
// WHY TWO BITS AND NOT A COUNT. `GraphDispatchStats::ragged_steps` says a step
// found no captured shape and stops there, and #2117 asks a question that count
// cannot answer. That issue predicts admission raggedness -- a scheduler step
// that admitted a prefill beside the decode rows -- at 3% to 7% of steps at the
// #1574 shape, and in the same breath warns that "a share far above 10% means
// something other than admission is making steps ragged, and #1943 is the first
// suspect". One number is consistent with both readings, so it discriminates
// neither, and the whole point of reading it is to choose between them.
struct RaggedStepShape {
  bool has_prefill_row = false;
  bool has_decode_row = false;
};

// The per-request classification, and it is the SAME test `GraphEligibleQueryLen`
// applies above: request `i` is a DECODE or VERIFY row exactly when its query
// length is `drafts_i + 1`, and a PREFILL row otherwise. Using one test for both
// is deliberate -- a classifier that disagreed with the eligibility predicate
// would attribute a step to a cause the predicate did not refuse it for.
//
// `drafts_per_req` is EMPTY on every non-speculative engine and on every step
// the scheduler recorded no drafts for, and then `drafts_i` is 0 and a
// single-token row is a decode row, which is that engine's correct answer.
//
// Reads `query_start_loc` in place and allocates nothing: this runs on the
// per-step path, and `num_reqs` is bounded by `max_num_seqs`.
inline RaggedStepShape ClassifyStepRows(const std::vector<int32_t>& query_start_loc,
                                        int64_t num_reqs,
                                        const std::vector<int32_t>& drafts_per_req) {
  RaggedStepShape shape;
  if (num_reqs <= 0 ||
      static_cast<int64_t>(query_start_loc.size()) < num_reqs + 1) {
    return shape;
  }
  const bool have_drafts =
      static_cast<int64_t>(drafts_per_req.size()) == num_reqs;
  for (int64_t i = 0; i < num_reqs; ++i) {
    const size_t at = static_cast<size_t>(i);
    const int64_t qlen = query_start_loc[at + 1] - query_start_loc[at];
    const int64_t drafts = have_drafts ? drafts_per_req[at] : 0;
    if (qlen == drafts + 1) {
      shape.has_decode_row = true;
    } else {
      shape.has_prefill_row = true;
    }
  }
  return shape;
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
  // SPEC-DFLASH2 W13 (#2117): WHY, split three ways by `RaggedStepShape` above.
  // The three PARTITION `ragged_steps` -- every ragged step increments exactly
  // one of them in the same call -- which is why they are set from the same
  // `NoteGraphDispatch` argument rather than from a second entry point that
  // could be forgotten at one call site and silently break the sum.
  //
  // `ragged_mixed_steps` is #2117 mechanism 1's population EXACTLY: the step
  // carried a prefill row and a decode row together, so the decode rows lost
  // both the graph and the decode attention lane to an admission this step did
  // not have to make.
  int64_t ragged_mixed_steps = 0;
  // Only prefill rows. Ordinary prefill; nothing was degraded, and counting it
  // in with the mixed steps is how a 3% mechanism reads as a 30% one.
  int64_t ragged_prefill_only_steps = 0;
  // No prefill row and still not uniform: the verify widths differ ACROSS
  // requests. This is #1943's shape, and it is the reading #2117 names as the
  // alternative to admission.
  int64_t ragged_spec_only_steps = 0;
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
  // SPEC-DFLASH2 W10 (#1857): steps whose uniform speculative verify length was
  // classified onto the DECODE attention class by the mirrored reorder
  // threshold (`SpecAsDecodeQueryLen`, backend.h). The classification is
  // invisible from outside the runner for exactly the reason `uniform_steps`
  // exists -- a token gate cannot see which attention lane a step took -- so
  // the decision moves a number a CPU test can read. Zero on every non-spec
  // engine and on every engine whose steps never produce a uniform verify.
  int64_t spec_as_decode_steps = 0;
};
// Steps the dispatcher classified. `uniform_steps` and `ragged_steps` partition
// it, so this is the denominator every share in the readout is taken over.
inline int64_t GraphDispatchTotalSteps(const GraphDispatchStats& s) {
  return s.uniform_steps + s.ragged_steps;
}

// THE READOUT'S DUE-CHECK (#2112). `period <= 0` is OFF, which is the default and
// is byte-identical to the tree before this wave. Pure, so a gate pins the
// policy without driving a runner.
inline bool GraphStatsDumpDue(int64_t total_steps, int64_t period) {
  return period > 0 && total_steps > 0 && total_steps % period == 0;
}

GraphDispatchStats GetGraphDispatchStats();
void ResetGraphDispatchStats();
// Records one step's dispatch decision. `query_len` is the value
// `ActualUniformDecodeQueryLen` returned, or 0 for "no uniform length";
// `configured_query_len` is `UniformDecodeQueryLen(num_speculative_tokens)`,
// which is what separates a clamped verify from a full-depth one.
// `ragged_shape` is consulted ONLY when `query_len <= 0`, and it is a required
// argument rather than a defaulted one so that a new call site cannot add a
// ragged step to the total without saying what kind it was.
void NoteGraphDispatch(int64_t query_len, int64_t configured_query_len,
                       const RaggedStepShape& ragged_shape);

// The readout period from `VT_GRAPH_STATS`, 0 when unset, empty, non-numeric or
// not positive. Read FRESH on each call rather than latched into a static: the
// tree's own convention for a host-path environment read that a test must be
// able to flip inside one process (`Fa2PrefillEnabled`, cuda_paged_attn.cu,
// which reads once per full-attention layer per step and says so). A latched
// value would also make this readout depend on which test case ran first, which
// is the "state was not the one you believed" shape.
int64_t GraphStatsDumpPeriod();

// The `[graph-dispatch]` line, WITHOUT a trailing newline. Pure, so the field
// set is gateable on a CPU box with no runner: #2112's whole complaint is that
// the counters were only ever read by a test, and a formatter nobody can pin
// would be the same defect one level in.
std::string FormatGraphDispatchStats(const GraphDispatchStats& s);
// Records one newly opened decode-graph capture shape.
void NoteDecodeGraphShape();
// Records one step classified spec-as-decode (query_len > 1 admitted onto the
// decode attention class by the mirrored reorder threshold). W10 (#1857).
void NoteSpecAsDecode();
// Records one step refused by the distinct-query-length bound.
void NoteDecodeGraphQueryLenDecline();

}  // namespace v1
}  // namespace vllm

#endif  // VLLM_V1_WORKER_GPU_CUDAGRAPH_DISPATCH_H_
