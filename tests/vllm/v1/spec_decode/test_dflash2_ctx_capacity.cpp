// SPEC-DFLASH2 ([#1919](https://github.com/mudler/vllm.cpp/issues/1919)) — the
// draft context store's capacity, gated through the PRODUCTION async front.
//
// WHAT #1919 MEASURED, AND WHY THE SECOND REQUEST IS THE ASSERTION. On `main` @
// `9aea9efec`, a DFlash2 K=8 server at `--max-model-len 12288` answered a
// 10-prompt short suite 10/10, then took one ~8K-token prompt and threw
//
//     vt: AppendContextKVDevice: paged store capacity exceeded
//         (raise kDflashMaxCtxSlots)
//
// from inside the EngineCore step. After that EVERY request on that server, of
// any size, came back `[request submitted to a stopped AsyncLLM]`, while the
// HTTP process stayed alive and answering `/health`. So the defect is not that
// the oversized request failed. It is that the ENGINE died, and a polite error
// on the first request would leave the whole outage in place. Every case here
// therefore issues a SECOND, ordinary request afterwards and asserts that one.
//
// The store's capacity was `kDflashMaxCtxSlots = 4096`, a compile-time constant
// with no relation to the `max_model_len` the engine advertises and admits. That
// is why these cases drive a `max_model_len` ABOVE 4096: a store that is sized
// from `max_model_len` can only be SHOWN to be sized from it by a value that
// differs from the constant it replaced. A fixture at the shared
// `kMaxModelLen = 32` cannot distinguish the two.
//
// The mirrored behaviour is the pinned oracle's, at `5559679229`. Upstream's
// DFlash draft keeps no private store — its context K/V is written into the
// engine's own paged KV cache (`vllm/model_executor/models/qwen3_dflash.py`
// :604-620) whose block tables are `cdiv(max_model_len, block_size)`
// (`vllm/v1/worker/gpu/model_runner.py:426,444`) — and where a speculator cannot
// serve a request it emits an EMPTY draft and lets the target run alone
// (`vllm/v1/spec_decode/ngram_proposer.py:156-159`,
// `vllm/v1/spec_decode/suffix_decoding.py:59-62`). It never raises.
#include <doctest/doctest.h>

#include "dflash2_runner_fixture.h"

#include "vllm/v1/engine/async_llm.h"

namespace {

// `VT_SPEC_TRACE` is latched by a function-local `static` on the FIRST propose
// in the process, so it has to be set before any case runs. Level 1 is what
// prints the `first=[...]` payload `DraftedBlocks` counts; that count is how the
// capped case MEASURES the fallback engaging rather than asserting it.
const bool kSpecTraceEnabled = [] {
  ::setenv("VT_SPEC_TRACE", "1", 1);
  return true;
}();

// Above 4096 on both counts, which is the whole point: `kDflashMaxCtxSlots` was
// 4096, so a context that stops short of it cannot tell a store sized from
// `max_model_len` apart from the constant. 4300 prompt tokens crosses it inside
// the third chunked-prefill chunk (the dense per-step budget is 2048).
constexpr int kLongModelLen = 6144;
constexpr int kLongPromptTokens = 4300;

// The per-request drain budget. Generous on purpose: the 4300-token prefill
// takes single-digit seconds on a CPU box, and a slow runner must not turn a
// pass into a timeout.
constexpr int kDrainBudgetSeconds = 60;

// One token id of the tiny BPE fixture ("hello"). The prompts here go in
// PRE-TOKENIZED, through `AsyncLLM::generate(std::vector<int32_t>, ...)`, so the
// context length under test is an exact number rather than whatever the
// tokenizer happened to make of a long string.
constexpr int32_t kTok = 13;

// Set an environment variable for one scope and restore the previous state
// exactly, including "it was not set".
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const std::string& value) : name_(name) {
    const char* prev = ::getenv(name);
    had_ = prev != nullptr;
    if (had_) prev_ = prev;
    ::setenv(name, value.c_str(), 1);
  }
  ~ScopedEnv() {
    if (had_)
      ::setenv(name_, prev_.c_str(), 1);
    else
      ::unsetenv(name_);
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  const char* name_;
  bool had_ = false;
  std::string prev_;
};

// One request's outcome, with the THREE outcomes kept apart. `AsyncLLM::generate`
// collapses two of them: it blocks until a terminal output arrives, and on the
// pre-fix tree the in-flight request's collector NEVER receives one, because the
// EngineCore that would have produced it is dead. A blocking driver therefore
// HANGS instead of reporting, and a test that hangs has measured nothing — the
// instrument failed to answer, which is not the same as the code passing. So the
// drain below carries a deadline and records `timed_out` as its own state,
// distinct from both `finished` and `threw`.
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

RequestOutcome DrainWithDeadline(vllm::v1::AsyncLLM& async, std::vector<int32_t> prompt,
                                 int max_tokens, const std::string& req_id) {
  RequestOutcome o;
  try {
    vllm::v1::AsyncRequest req =
        async.add_request(req_id, std::move(prompt), Greedy(max_tokens));
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

// What one engine run produced: each request's outcome, and how many blocks the
// production propose trace reported.
struct TwoRequestRun {
  RequestOutcome first;
  RequestOutcome second;
  std::vector<std::string> blocks;
  std::string stderr_text;
};

// Build ONE engine, send `first`, then send `second` on the SAME engine. The
// second request is what #1919 is about: on the pre-fix tree it comes back
// `request submitted to a stopped AsyncLLM`.
TwoRequestRun RunTwoRequests(int max_model_len, const std::vector<int32_t>& first,
                             int first_max_tokens, const std::vector<int32_t>& second,
                             int second_max_tokens) {
  TwoRequestRun r;
  const HfConfig target = MakeDenseConfig(max_model_len);
  const ScratchDraftDir dir;
  r.stderr_text = CaptureStderr([&] {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir, max_model_len, /*max_num_seqs=*/1,
                                      /*max_num_batched_tokens=*/8192,
                                      /*num_blocks=*/512),
                     MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false));
    r.first = DrainWithDeadline(eng.async_engine(), first, first_max_tokens, "req-first");
    r.second =
        DrainWithDeadline(eng.async_engine(), second, second_max_tokens, "req-second");
  });
  r.blocks = DraftedBlocks(r.stderr_text);
  return r;
}

}  // namespace

// ─── G1: #1919's own repro, through the production async front ──────────────
TEST_CASE("dflash2 ctx capacity (#1919): a >4K-token prompt does NOT kill the engine") {
  const std::vector<int32_t> long_prompt(kLongPromptTokens, kTok);
  const std::vector<int32_t> short_prompt = {kTok};
  const TwoRequestRun r = RunTwoRequests(kLongModelLen, long_prompt, /*first_max_tokens=*/2,
                                         short_prompt, /*second_max_tokens=*/4);

  // THE ASSERTION. Pre-fix this reads
  // "request submitted to a stopped AsyncLLM" — the engine is gone and every
  // later request of any size fails with it.
  INFO("second request: ", Describe(r.second));
  CHECK(r.second.finished);
  CHECK(r.second.threw.empty());
  CHECK_FALSE(r.second.timed_out);
  CHECK_FALSE(r.second.tokens.empty());

  // And the oversized request is served rather than refused, because the target
  // can serve it: 4300 tokens is inside the advertised 6144. Pre-fix it never
  // completes at all — `AppendContextKVDevice: paged store capacity exceeded`
  // takes the EngineCore down mid-step and this request's collector is left
  // waiting for a terminal frame nothing will ever write.
  INFO("first request: ", Describe(r.first));
  CHECK(r.first.finished);
  CHECK(r.first.threw.empty());
  CHECK_FALSE(r.first.timed_out);
  CHECK_FALSE(r.first.tokens.empty());

  // Nothing in the run may still be talking about recompiling a constant.
  CHECK(r.stderr_text.find("kDflashMaxCtxSlots") == std::string::npos);

  // AND IT SPECULATED THE WHOLE WAY. Asserting only that both requests finished
  // is a gate the SIZING can walk through: the mutation pass measured it. With
  // the store pinned back to 4096 slots, or with the resolver's answer unwired
  // from the store, a 4300-token prompt still completes — the fallback catches
  // it and the request decodes on the target alone. That is the right OUTCOME
  // for a request that genuinely does not fit, and exactly the wrong one here,
  // where `max_model_len` is 6144 and the whole context is supposed to be
  // speculated. It is also invisible from the outside, because the verify is
  // lossless and only acceptance falls — this row's standing defect class. So
  // the fallback must NOT have fired, and blocks must have been drafted.
  INFO("blocks drafted: ", r.blocks.size());
  CHECK(r.stderr_text.find("has outgrown the draft speculative context") ==
        std::string::npos);
  CHECK_FALSE(r.blocks.empty());
}

// ─── G4: the startup announcement ───────────────────────────────────────────
TEST_CASE("dflash2 ctx capacity (#1919): the effective speculative context is announced") {
  const std::vector<int32_t> p = {kTok};
  const TwoRequestRun r =
      RunTwoRequests(kLongModelLen, p, /*first_max_tokens=*/2, p, /*second_max_tokens=*/2);
  INFO("stderr: ", r.stderr_text);
  // Named, once, with the resolved limit in it. `kLongModelLen + (k + 1)` is the
  // want; the resolver rounds up to the page, so the line is checked for the
  // marker and for a limit that is at least the advertised context.
  const size_t at = r.stderr_text.find("speculative context");
  REQUIRE(at != std::string::npos);
  CHECK(r.stderr_text.find("speculative context", at + 1) == std::string::npos);
  CHECK(r.stderr_text.find(std::to_string(kLongModelLen)) != std::string::npos);

  // AND IT STATES THE AGGREGATE. One store is built per batch row, so the
  // device holds `bytes_per_request * max_num_seqs` and `gpu_memory_utilization`
  // accounts none of it. A line that gives only the per-request cost leaves the
  // reader to know that multiplication exists; a concurrency ladder that does
  // not know the term is there measures it as noise.
  CHECK(r.stderr_text.find("max_num_seqs") != std::string::npos);
  CHECK(r.stderr_text.find("gpu_memory_utilization") != std::string::npos);
}

// ─── G2: the fallback, and the server surviving it ──────────────────────────
TEST_CASE("dflash2 ctx capacity (#1919): a request that outgrows the store falls back") {
  // Two arms of the SAME workload on the shared 32-token fixture, differing only
  // in the store's cap. `VT_DFLASH_CTX_MAX_TOKENS=16` is one page, and the (1+k)
  // block needs 4 of it, so the context runs out at 12 committed rows — reached
  // partway through a 16-token generation.
  const std::vector<int32_t> p = {kTok};
  const TwoRequestRun uncapped =
      RunTwoRequests(kMaxModelLen, p, /*first_max_tokens=*/16, p, /*second_max_tokens=*/4);
  TwoRequestRun capped;
  {
    const ScopedEnv cap("VT_DFLASH_CTX_MAX_TOKENS", "16");
    capped =
        RunTwoRequests(kMaxModelLen, p, /*first_max_tokens=*/16, p, /*second_max_tokens=*/4);
  }

  // Neither request errors, on either arm. The fallback is not a refusal.
  INFO("capped first: ", Describe(capped.first));
  CHECK(capped.first.finished);
  INFO("capped second: ", Describe(capped.second));
  CHECK(capped.second.finished);
  REQUIRE(uncapped.first.finished);
  REQUIRE(uncapped.second.finished);

  // THE FALLBACK IS MEASURED, not asserted about: the capped arm proposes
  // STRICTLY FEWER blocks, because the request stops speculating once its
  // context outgrows the store. A run that kept speculating would tie.
  INFO("uncapped blocks: ", uncapped.blocks.size(), " capped blocks: ", capped.blocks.size());
  CHECK(capped.blocks.size() < uncapped.blocks.size());
  CHECK_FALSE(uncapped.blocks.empty());

  // And it says so, once, naming the request.
  CHECK(capped.stderr_text.find("req-first") != std::string::npos);

  // THE TOKENS DO NOT MOVE. The verify is lossless, so dropping speculation for
  // a request changes what was DRAFTED and not what was emitted. This is the
  // half a block count cannot see: a fallback that also changed the output would
  // be a correctness bug wearing a throughput cost.
  CHECK(capped.first.tokens == uncapped.first.tokens);
  CHECK(capped.second.tokens == uncapped.second.tokens);
}

// ─── G3: the resolver, unit ─────────────────────────────────────────────────
TEST_CASE("dflash2 ctx capacity (#1919): the store's capacity derives from max_model_len") {
  const HfConfig target = MakeDenseConfig();
  const HfConfig draft = MakeDraftConfig(target, /*muse_glimmer_scalars=*/false);
  const int64_t tq = kSpecTokens + 1;
  const int64_t nreq = 1;

  // Uncapped: the capacity covers the advertised context plus the (1+k) query
  // block, which is upstream's own `min(max_seq_len + num_query_per_req,
  // max_model_len)` bound (dflash/speculator.py:331-333) read from the other
  // side.
  const auto small = vllm::Qwen3DFlashModel::ResolveCtxStoreSizing(draft, 4096, tq, nreq);
  CHECK(small.slots >= 4096 + tq);
  CHECK_FALSE(small.capped);
  CHECK(small.slots % small.page_size == 0);
  CHECK(small.bytes_per_slot > 0);
  CHECK(small.bytes_per_request == small.slots * small.bytes_per_slot);

  // A bigger advertised context buys a bigger store. Pin this against the
  // constant it replaced: 4096 was the WHOLE capacity, so a resolver that still
  // returned it would answer the same for both of these.
  const auto big = vllm::Qwen3DFlashModel::ResolveCtxStoreSizing(draft, 262144, tq, nreq);
  CHECK(big.slots > small.slots);

  // Capped: the override replaces the byte budget, `capped` says so truthfully,
  // and `want_slots` still records what was asked for.
  {
    const ScopedEnv cap("VT_DFLASH_CTX_MAX_TOKENS", "16");
    const auto c = vllm::Qwen3DFlashModel::ResolveCtxStoreSizing(draft, 4096, tq, nreq);
    CHECK(c.slots == 16);
    CHECK(c.capped);
    CHECK(c.want_slots >= 4096 + tq);
  }

  // Never below one page, however small the override: a store that cannot hold
  // one block is not a smaller store, it is a broken one.
  {
    const ScopedEnv cap("VT_DFLASH_CTX_MAX_TOKENS", "1");
    const auto c = vllm::Qwen3DFlashModel::ResolveCtxStoreSizing(draft, 4096, tq, nreq);
    CHECK(c.slots == c.page_size);
    // And the REPORTED budget is the one that was applied, not the one that was
    // asked for. Computed before the floor, this read zero bytes for a store
    // that holds a page — a struct field that lies to whoever prints it.
    CHECK(c.budget_slots == c.page_size);
    CHECK(c.budget_bytes == c.page_size * c.bytes_per_slot * c.max_num_reqs);
  }
}

// ─── G5: the budget is the AGGREGATE, because the residency is ──────────────
TEST_CASE("dflash2 ctx capacity (#1919): the byte budget bounds the DEVICE, not one row") {
  // The runner builds ONE store per BATCH ROW, so peak device residency is
  // `bytes_per_request * max_num_reqs` and `gpu_memory_utilization` accounts
  // none of it. A budget applied PER REQUEST bounds a number nothing pays: at
  // 256 MiB per request it was an 8 GiB peak at the `--max-num-seqs 32`
  // docs/USAGE.md itself shows. These cases hold the aggregate, so a resolver
  // that goes back to dividing nothing by the concurrency reds.
  const HfConfig target = MakeDenseConfig();
  const HfConfig draft = MakeDraftConfig(target, /*muse_glimmer_scalars=*/false);
  const int64_t tq = kSpecTokens + 1;

  // A geometry big enough that the budget BITES, so the cap is the thing under
  // test rather than `want_slots`. `max_model_len` is far above what any
  // aggregate budget can buy at these concurrencies.
  const int64_t huge_len = 1LL << 28;

  const auto one = vllm::Qwen3DFlashModel::ResolveCtxStoreSizing(draft, huge_len, tq, 1);
  const auto many = vllm::Qwen3DFlashModel::ResolveCtxStoreSizing(draft, huge_len, tq, 32);
  REQUIRE(one.capped);
  REQUIRE(many.capped);

  // The reported concurrency is the one that was asked for, and `bytes_total`
  // is the product a reader would otherwise have to know to compute.
  CHECK(one.max_num_reqs == 1);
  CHECK(many.max_num_reqs == 32);
  CHECK(one.bytes_total == one.bytes_per_request * one.max_num_reqs);
  CHECK(many.bytes_total == many.bytes_per_request * many.max_num_reqs);

  // THE AGGREGATE IS WHAT IS BOUNDED. Raising the concurrency shrinks the
  // per-request store instead of multiplying the device bill: a PER-REQUEST
  // budget answers `many.slots == one.slots` and `many.bytes_total ==
  // 32 * one.bytes_total`, which is the shape this case exists to red.
  CHECK(many.slots < one.slots);
  CHECK(many.bytes_total <= one.bytes_total);

  // And the bound is the 8 GiB TOTAL, held to within one page at each
  // concurrency — the budget's own arithmetic, recomputed here from the
  // struct's reported `bytes_per_slot` rather than read back from it.
  const int64_t kTotalBudget = 8LL * 1024 * 1024 * 1024;
  // The REPORTED budget is the aggregate too, because that is what the startup
  // line prints in the capped branch. Held against the field, not only against
  // the arithmetic beside it: a resolver that caps by the aggregate and reports
  // a per-request number passes every bound below while the line it feeds says
  // 256 MiB for an 8 GiB budget.
  CHECK(one.budget_bytes == one.budget_slots * one.bytes_per_slot * one.max_num_reqs);
  CHECK(many.budget_bytes == many.budget_slots * many.bytes_per_slot * many.max_num_reqs);
  CHECK(one.budget_slots * one.bytes_per_slot * one.max_num_reqs <= kTotalBudget);
  CHECK((one.budget_slots + one.page_size) * one.bytes_per_slot * one.max_num_reqs >
        kTotalBudget);
  CHECK(many.budget_slots * many.bytes_per_slot * many.max_num_reqs <= kTotalBudget);
  CHECK((many.budget_slots + many.page_size) * many.bytes_per_slot * many.max_num_reqs >
        kTotalBudget);

  // The default is deliberately behaviour-preserving at the `--max-num-seqs 32`
  // docs/USAGE.md shows: the per-request share there is the 256 MiB the
  // per-request budget used to hand out unconditionally. This is the sentence
  // the spec makes, executable.
  CHECK(many.budget_slots * many.bytes_per_slot <= 256LL * 1024 * 1024);
  CHECK((many.budget_slots + many.page_size) * many.bytes_per_slot > 256LL * 1024 * 1024);

  // A nonsense concurrency floors at one rather than dividing the whole
  // aggregate into a single request, which would be the per-request budget
  // under another name.
  const auto zero = vllm::Qwen3DFlashModel::ResolveCtxStoreSizing(draft, huge_len, tq, 0);
  CHECK(zero.max_num_reqs == 1);
  CHECK(zero.slots == one.slots);
}

// ─── The short-prompt baseline, unchanged ───────────────────────────────────
TEST_CASE("dflash2 ctx capacity (#1919): the short-prompt propose is untouched") {
  // #1919's server had just answered 10/10 short prompts before the long one
  // arrived, so "short prompts still draft" is the control this whole wave must
  // not disturb.
  std::string threw;
  const std::vector<std::string> blocks = RunAndCollectDrafts(false, &threw);
  INFO("threw: ", threw);
  CHECK(threw.empty());
  CHECK_FALSE(blocks.empty());
}
