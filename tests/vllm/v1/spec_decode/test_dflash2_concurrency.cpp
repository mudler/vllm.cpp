// SPEC-DFLASH2 ([#2008](https://github.com/mudler/vllm.cpp/issues/2008)) — the
// DFlash2 draft context must belong to the REQUEST, not to the batch ROW.
//
// WHAT #2008 MEASURED. A DFlash2 K=8 server on an idle leased GB10 answers
// concurrency 1 at 24.70 out tok/s (8/8 ok) and DIES at concurrency 2, one
// request in:
//
//     vt: propose_drafts_block: context position discontinuity (accumulation
//         out of sync with the target's committed positions)
//
// after which every later request on that server returns
// `[request submitted to a stopped AsyncLLM]`. The operator then ran the
// isolation: with `--speculative-config` OMITTED and everything else identical,
// the same two concurrent requests both complete. So batching, scheduling, the
// paged KV cache, the block tables and the sampler all serve two sequences
// correctly, and the defect is confined to the draft's per-request context
// accumulation.
//
// WHY IT DESYNCHRONISES. `GPUModelRunner` holds the draft context in four
// arrays indexed by BATCH ROW — `dflash_kv_store_`, `dflash_ctx_len_`,
// `dflash_ctx_reqid_` and `dflash_ctx_disabled_`. A row index is not stable for
// a request's lifetime here: `InputBatch::condense` slides a live request down
// into the hole a finished neighbour left (input_batch.cpp:611-760, the row move
// at :686-706), and `InputBatch::swap_states` exchanges two live rows
// (input_batch.cpp:762-847). Both permute every per-slot array they know about
// — and they know nothing about the runner's four, which live in
// `GPUModelRunner`, not in `InputBatch`.
//
// So when the first of two concurrent requests finishes, condense moves the
// SURVIVOR from row 1 to row 0, and the survivor now meets row 0's bookkeeping:
// `dflash_ctx_reqid_[0]` still names the departed request, the reuse test at
// runner.cpp:2895-2906 reads a changed occupant, resets the store and sets
// `dflash_ctx_len_[0] = 0` — and the survivor is mid-decode at absolute
// position L > 0. The invariant at runner.cpp:2939-2945 then refuses, which is
// exactly what it is for. The header comment two lines above the arrays already
// names the shape of this bug for a DIFFERENT array
// (input_batch.h:240-252): "Upstream needs no equivalent because it never
// condenses: states.py:132 returns a finished request's slot to a free list and
// the slot index is stable for the request's lifetime." The DFlash2 arrays never
// got that treatment.
//
// UPSTREAM. vLLM's DFlash draft keeps no runner-side per-row context length at
// all: its context K/V lives in the engine's own paged KV cache
// (`vllm/model_executor/models/qwen3_dflash.py`), addressed through the block
// table, whose rows condense DOES move (`block_table.move_row`,
// input_batch.cpp:706). The context is per-REQUEST upstream. Keying our four
// arrays by request id is that same semantics expressed in our structure.
//
// WHAT THIS FILE GATES, AND WHY "IT DID NOT THROW" IS NOT ENOUGH. There is a
// second way to make the throw stop: mark the moved row disabled and let the
// request run on the target alone, which is the #1919 fallback and which
// `continue`s BEFORE the invariant. That produces IDENTICAL TOKENS — the verify
// is lossless, so a request that stops speculating emits exactly what it emitted
// before, only slower. A token gate cannot see it. So the cases below assert
// that the survivor KEEPS PROPOSING across the move, and at every step it is
// alive for — read off the production `VT_SPEC_TRACE` line, which prints the
// proposing-row count and prints `NO proposing rows this step` when that count
// is zero (runner.cpp:3259-3262). That is the leg which refuses the fallback
// repair, and it is MEASURED to refuse it rather than argued to: built on the
// pre-change code, that repair leaves case 1 green and turns this count into 9.
//
// The block-comparison leg a reader would expect next is deliberately absent,
// and the tail of the second case records why it would gate nothing here.
#include <doctest/doctest.h>

#include "dflash2_runner_fixture.h"

#include "vllm/v1/engine/llm_engine.h"

namespace {

// Latched by a function-local static on the FIRST propose in the process, so it
// has to be set before any case runs. Level 1 is what prints `[spec-propose]`.
const bool kSpecTraceEnabled = [] {
  ::setenv("VT_SPEC_TRACE", "1", 1);
  return true;
}();

// One parsed `[spec-propose]` line.
struct ProposeLine {
  int rows = 0;              // P, the number of rows that proposed this step
  std::string first;         // the first proposing row's drafted ids
  bool none = false;         // the `NO proposing rows this step` variant
};

std::vector<ProposeLine> ParseProposeLines(const std::string& captured) {
  std::vector<ProposeLine> out;
  size_t at = 0;
  const std::string tag = "[spec-propose] ";
  while ((at = captured.find(tag, at)) != std::string::npos) {
    const size_t body = at + tag.size();
    const size_t eol = captured.find('\n', body);
    const std::string line =
        captured.substr(body, (eol == std::string::npos) ? std::string::npos : eol - body);
    ProposeLine p;
    if (line.rfind("NO proposing rows", 0) == 0) {
      p.none = true;
    } else {
      const size_t r = line.find("rows=");
      if (r != std::string::npos) p.rows = std::atoi(line.c_str() + r + 5);
      const size_t f = line.find("first=[");
      if (f != std::string::npos) {
        const size_t open = f + 7;
        const size_t close = line.find(']', open);
        if (close != std::string::npos) p.first = line.substr(open, close - open);
      }
    }
    out.push_back(p);
    at = (eol == std::string::npos) ? captured.size() : eol;
  }
  return out;
}

// The survivor: a two-token prompt and enough output that it is still decoding
// long after its neighbour has left the batch.
constexpr int kKeeperTokens = 10;
// The neighbour: added FIRST, so it takes row 0 and the survivor takes row 1,
// and finishes FIRST, so condense slides the survivor down into row 0. With
// k = kSpecTokens = 3 a step can commit up to 4 tokens, so 1 is one step.
constexpr int kLeaverTokens = 1;

struct RunResult {
  std::string threw;
  std::string keeper_text;
  std::vector<ProposeLine> lines;
};

// Drive the SYNCHRONOUS production front (LLMEngine::add_request + step), so the
// step boundaries the trace reports are the ones the scheduler took and nothing
// depends on a drain deadline.
RunResult Run(bool with_neighbour) {
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  RunResult r;
  const std::string captured = CaptureStderr([&] {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir, /*max_model_len=*/0, /*max_num_seqs=*/4),
                     MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false));
    vllm::v1::LLMEngine& e = eng.engine();
    try {
      if (with_neighbour) e.add_request("leaver", "hello", Greedy(kLeaverTokens));
      e.add_request("keeper", "hello world", Greedy(kKeeperTokens));
      while (e.has_unfinished_requests()) {
        for (const vllm::RequestOutput& o : e.step()) {
          if (o.request_id == "keeper" && o.finished && !o.outputs.empty())
            r.keeper_text = o.outputs[0].text;
        }
      }
    } catch (const std::exception& ex) {
      r.threw = ex.what();
    }
  });
  r.lines = ParseProposeLines(captured);
  return r;
}

}  // namespace

TEST_CASE("dflash2 #2008: a second request must not desynchronise the draft context") {
  const RunResult conc = Run(/*with_neighbour=*/true);
  INFO("threw: ", conc.threw);
  CHECK(conc.threw.empty());
}

TEST_CASE("dflash2 #2008: the survivor keeps its context across the row move") {
  const RunResult solo = Run(/*with_neighbour=*/false);
  const RunResult conc = Run(/*with_neighbour=*/true);
  INFO("solo threw: ", solo.threw);
  INFO("conc threw: ", conc.threw);
  REQUIRE(solo.threw.empty());
  REQUIRE(conc.threw.empty());

  // The run genuinely reached concurrency: some step had TWO proposing rows.
  // Without this the case would pass on a build that served the two requests
  // one after the other and never exercised the move at all.
  int max_rows = 0;
  for (const ProposeLine& p : conc.lines) max_rows = std::max(max_rows, p.rows);
  INFO("max proposing rows observed: ", max_rows);
  CHECK(max_rows == 2);

  // The survivor never stops proposing. The #1919 fallback (mark the moved row
  // disabled, run on the target alone) makes the throw go away and emits the
  // SAME TOKENS, and this is the leg that separates it from a real fix: a step
  // in which the only live request proposes nothing prints the `NO proposing
  // rows` line.
  int none_lines = 0;
  for (const ProposeLine& p : conc.lines) none_lines += p.none ? 1 : 0;
  CHECK(none_lines == 0);

  // And it proposes at EVERY step it is alive for, which is a count rather than
  // an absence. The concurrent run's steps are the solo run's plus the one it
  // shared with the neighbour, so the survivor's solo-row steps are one fewer.
  // A repair that let the survivor skip a single propose — the shape a
  // "re-seat it next step" patch takes — moves this number without moving a
  // token.
  std::vector<std::string> conc_tail;
  for (const ProposeLine& p : conc.lines)
    if (!p.none && p.rows == 1) conc_tail.push_back(p.first);
  std::vector<std::string> solo_all;
  for (const ProposeLine& p : solo.lines)
    if (!p.none) solo_all.push_back(p.first);
  REQUIRE_FALSE(conc_tail.empty());
  INFO("conc solo-row proposes: ", conc_tail.size(), "  solo proposes: ", solo_all.size());
  CHECK(conc_tail.size() + 1 == solo_all.size());

  // Token-exactness across the move. NECESSARY AND NOT SUFFICIENT, and this
  // comment says which because the first version of this file got it wrong.
  //
  // WHAT IS **NOT** GATED HERE, MEASURED RATHER THAN ASSUMED. The obvious
  // stronger leg is to compare the drafted BLOCKS either side of the move
  // against the solo run's, on the argument that a repair which resets the
  // context instead of moving it drafts from an empty context and is caught
  // while every token stays unchanged. That leg was written, and it is a
  // TAUTOLOGY on this fixture: with the production invariant deleted and the
  // context reset on every row move, this draft still emits `12 12 12` at every
  // single step of both runs. The synthetic draft's block is CONSTANT in the
  // context — its weights are seeded noise over a 24-token vocabulary, and the
  // selector walk collapses to one id — so a block comparison here asserts a
  // constant against itself. (`test_dflash2_runner_reach`'s value-sensitivity
  // case is not affected: it moves the drafts by changing the selector's
  // WEIGHTS, not the context.)
  //
  // What pins the survivor's context to the survivor is therefore the two
  // PRODUCTION invariants this change deliberately leaves alone, standing
  // together with the two legs above. No throw (leg 1) means that for every
  // proposing row `positions[rows[0]] == ctx_len` AND
  // `ctx_len == DeviceKVNumCtx(store)`; still proposing (leg 3) means the row
  // reached those invariants rather than skipping them down the disabled path.
  // Together they say the survivor proposed with a context length equal to its
  // OWN committed position, held by a store containing exactly that many rows.
  CHECK(conc.keeper_text == solo.keeper_text);
  CHECK_FALSE(conc.keeper_text.empty());
}
