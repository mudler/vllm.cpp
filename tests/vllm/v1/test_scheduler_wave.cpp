// Wave-boundary scheduler-composition parity test (ITL tail-stall discriminator,
// row SERVE-GATE-ONLINE, spec .agents/specs/tail-stall-analysis-2026-07-16.md).
//
// This is the C++ half of the CPU-only Phase-1 discriminator for the two failing
// ITL tail axes (c8 p99_itl 0.5599, c32 p90_itl 0.7925). The oracle half drives
// the REAL vllm.v1.core.sched Scheduler (sync, depth-1) and AsyncScheduler
// (depth-2) through the SAME scripted wave-boundary sequence
// (tools/bench/scheduler_wave_diff.py, golden
// tests/fixtures/scheduler_wave/wave_script_oracle.json) and dumps the per-step
// composition; this file drives OUR Scheduler / AsyncScheduler through the
// byte-identical script and asserts the same composition, at the binding shape
// (max_num_seqs=32, max_num_batched_tokens=2048, chunked prefill on, no prefix
// reuse). The oracle-derived constants below match that golden exactly.
//
// The scenario from the analysis: N running requests mid-decode, 2 finish the
// SAME step (max_tokens=3), 2 fresh 1024-token prefills already waiting (injected
// at step 4). Recorded outcome (oracle, both pins e24d1b24 / 702f481):
//   c8  step4: total=2048, prefill=2042 [1024,1018] chunked, 6 decodes  (~860 ms)
//   c32 step4: total=2048, prefill=2018 [1024, 994] chunked, 30 decodes (~860 ms)
// and the sync arm equals the async arm EXACTLY at that stall step.
//
// VERDICT this test pins: the AsyncScheduler placeholder accounting + the depth-2
// driver do NOT change the per-step composition — both pack the 2048-token budget
// (1024 + chunk) into a single step at the wave boundary. There is therefore NO
// scheduler-policy divergence between our sync binding binary and vLLM's async
// binding arm (H-B decode-first budgeting and H-C partial-prefill are both dead;
// H-A admission-order is dead at 89b329e). The stall-MAGNITUDE gap (vLLM ~500 ms
// vs ours ~860 ms for this identical composition) is an async depth-2 GPU/pipeline
// OVERLAP effect (W3), not a scheduler change — see the spec verdict.
#include <doctest/doctest.h>

#include <algorithm>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/async_scheduler.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::v1::AsyncScheduler;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::Request;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

constexpr int kBlockSize = 16;
constexpr int kBudget = 2048;
constexpr int kMaxNumSeqs = 32;
constexpr int kFreshPromptLen = 1024;  // one prompt fills half the 2048 budget
constexpr int kFillPromptLen = 4;      // short prompt for the running set

void EnsureNoneHash() {
  static bool done = false;
  if (!done) {
    init_none_hash(sha256_cbor);
    done = true;
  }
}

template <typename SchedT>
std::unique_ptr<SchedT> MakeScheduler() {
  SchedulerConfig cfg;
  cfg.max_num_seqs = kMaxNumSeqs;
  cfg.max_num_batched_tokens = kBudget;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;  // > 1024+128; inert vs the oracle's 2048 here
  cfg.watermark = 0.0;
  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = 200000;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(kBlockSize, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<SchedT>(cfg, kv_cfg, kBlockSize, /*enable_caching=*/true);
}

// One recorded step's composition (mirrors scheduler_wave_diff.py's record).
struct StepRec {
  int step = 0;
  int total = 0;
  int prefill_tokens = 0;
  int decode_tokens = 0;
  int n_prefill = 0;
  int n_decode = 0;
  std::vector<int> prefill_counts;  // sorted descending
  bool chunked = false;
  int running = 0;
};

int uid_counter = 0;

std::unique_ptr<Request> MakeRequest(const std::string& id, int prompt_len,
                                     int max_tokens) {
  EnsureNoneHash();
  auto block_hasher = get_request_block_hasher(kBlockSize, sha256_cbor);
  SamplingParams params;
  params.max_tokens = max_tokens;
  const int32_t fill = 1 + (uid_counter++ % 40000);  // distinct prompts
  std::vector<int32_t> prompt(static_cast<size_t>(prompt_len), fill);
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   block_hasher);
}

// Record the composition of a SchedulerOutput given the pre-schedule snapshot of
// each request's (num_computed_tokens, num_prompt_tokens).
StepRec Record(Scheduler& s, const SchedulerOutput& so, int step,
               const std::map<std::string, int>& computed_before,
               const std::map<std::string, int>& prompt_len) {
  StepRec rec;
  rec.step = step;
  rec.total = so.total_num_scheduled_tokens;
  for (const auto& [rid, n] : so.num_scheduled_tokens) {
    int cb = 0;
    auto cit = computed_before.find(rid);
    if (cit != computed_before.end()) cb = cit->second;
    int pl = 0;
    auto pit = prompt_len.find(rid);
    if (pit != prompt_len.end()) pl = pit->second;
    if (cb < pl) {
      rec.prefill_tokens += n;
      rec.prefill_counts.push_back(n);
      if (cb + n < pl) rec.chunked = true;
      rec.n_prefill += 1;
    } else {
      rec.decode_tokens += n;
      rec.n_decode += 1;
    }
  }
  std::sort(rec.prefill_counts.begin(), rec.prefill_counts.end(),
            std::greater<int>());
  rec.running = static_cast<int>(s.running.size());
  return rec;
}

// Build a runner output: one canned token for every scheduled request that is
// past its prefill chunk this step (is_prefill_chunk == false), else empty (a
// request still in chunked prefill samples nothing) — exactly how the real
// runner and the oracle harness feed update_from_output.
ModelRunnerOutput MakeRunnerOutput(Scheduler& s, const SchedulerOutput& so,
                                   const std::vector<std::string>& sampling) {
  ModelRunnerOutput mro;
  int idx = 0;
  for (const auto& [rid, n] : so.num_scheduled_tokens) {
    (void)n;
    mro.req_ids.push_back(rid);
    mro.req_id_to_index[rid] = idx++;
    const bool samples =
        std::find(sampling.begin(), sampling.end(), rid) != sampling.end() &&
        s.requests.count(rid) > 0;
    mro.sampled_token_ids.push_back(samples ? std::vector<int32_t>{7}
                                            : std::vector<int32_t>{});
  }
  return mro;
}

std::pair<SchedulerOutput, std::vector<std::string>> ScheduleAndSnapshot(
    Scheduler& s, int step, std::vector<StepRec>& recs) {
  std::map<std::string, int> computed_before;
  std::map<std::string, int> prompt_len;
  for (const auto& [rid, req] : s.requests) {
    computed_before[rid] = req->num_computed_tokens;
    prompt_len[rid] = req->num_prompt_tokens;
  }
  SchedulerOutput so = s.schedule();
  std::vector<std::string> sampling;
  for (const auto& [rid, n] : so.num_scheduled_tokens) {
    (void)n;
    if (!s.requests.at(rid)->is_prefill_chunk) sampling.push_back(rid);
  }
  recs.push_back(Record(s, so, step, computed_before, prompt_len));
  return {std::move(so), std::move(sampling)};
}

// Drive the scripted wave-boundary sequence. N fillers (short prompt); ids f0/f1
// are the finishers (max_tokens=3), the rest are stayers (max_tokens=60). Two
// fresh kFreshPromptLen prefills W0/W1 are added right before the schedule() that
// becomes step `inject_at`. depth==1 is the sync driver; depth==2 is the async
// depth-2 batch-queue driver (two SchedulerOutputs in flight).
std::vector<StepRec> RunScript(Scheduler& s, int N, int inject_at, int depth,
                               int steps) {
  for (int i = 0; i < N; ++i) {
    const int max_tokens = (i < 2) ? 3 : 60;
    s.add_request(MakeRequest("f" + std::to_string(i), kFillPromptLen, max_tokens));
  }
  std::vector<StepRec> recs;
  int step = 0;
  bool injected = false;
  auto maybe_inject = [&]() {
    if (!injected && step >= inject_at) {
      s.add_request(MakeRequest("W0", kFreshPromptLen, 200));
      s.add_request(MakeRequest("W1", kFreshPromptLen, 200));
      injected = true;
    }
  };

  if (depth == 1) {
    for (int k = 0; k < steps; ++k) {
      maybe_inject();
      auto [so, sampling] = ScheduleAndSnapshot(s, step, recs);
      ++step;
      if (so.num_scheduled_tokens.empty()) break;
      ModelRunnerOutput mro = MakeRunnerOutput(s, so, sampling);
      s.update_from_output(so, mro);
    }
  } else {
    std::deque<std::pair<SchedulerOutput, std::vector<std::string>>> q;
    for (int d = 0; d < depth; ++d) {
      maybe_inject();
      auto snap = ScheduleAndSnapshot(s, step, recs);
      ++step;
      q.push_back(std::move(snap));
    }
    for (int k = 0; k < steps; ++k) {
      if (q.empty()) break;
      auto [so, sampling] = std::move(q.front());
      q.pop_front();
      ModelRunnerOutput mro = MakeRunnerOutput(s, so, sampling);
      s.update_from_output(so, mro);
      maybe_inject();
      auto snap = ScheduleAndSnapshot(s, step, recs);
      ++step;
      if (!snap.first.num_scheduled_tokens.empty()) q.push_back(std::move(snap));
    }
  }
  return recs;
}

const StepRec& AtStep(const std::vector<StepRec>& recs, int step) {
  for (const auto& r : recs) {
    if (r.step == step) return r;
  }
  static const StepRec kEmpty;
  return kEmpty;
}

}  // namespace

// ---------------------------------------------------------------------------
// c8 wave boundary: 8 running, 2 finish, 2 fresh 1024-prefills waiting. The
// admission step (4) packs the budget to 2048 (1024 + 1018 chunk) with 6 running
// decodes — the ~860 ms two-prefill stall. Sync (Scheduler, depth-1) and async
// (AsyncScheduler, depth-2) are IDENTICAL, matching the vLLM oracle.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler wave: c8 admission step packs 2048 (sync == async oracle)") {
  auto sync = MakeScheduler<Scheduler>();
  auto async = MakeScheduler<AsyncScheduler>();
  auto rs = RunScript(*sync, /*N=*/8, /*inject_at=*/4, /*depth=*/1, /*steps=*/14);
  auto ra = RunScript(*async, /*N=*/8, /*inject_at=*/4, /*depth=*/2, /*steps=*/14);

  const StepRec& s4 = AtStep(rs, 4);
  const StepRec& a4 = AtStep(ra, 4);

  // Oracle composition at the wave-boundary stall step (both arms).
  for (const StepRec* r : {&s4, &a4}) {
    CHECK(r->total == 2048);
    CHECK(r->prefill_tokens == 2042);
    CHECK(r->decode_tokens == 6);
    CHECK(r->n_prefill == 2);
    REQUIRE(r->prefill_counts.size() == 2);
    CHECK(r->prefill_counts[0] == 1024);
    CHECK(r->prefill_counts[1] == 1018);
    CHECK(r->chunked == true);
  }
  // The core parity claim: sync and async pack the budget identically.
  CHECK(s4.total == a4.total);
  CHECK(s4.prefill_tokens == a4.prefill_tokens);
  CHECK(s4.prefill_counts == a4.prefill_counts);
  CHECK(s4.decode_tokens == a4.decode_tokens);
}

// ---------------------------------------------------------------------------
// c32 wave boundary: 32 running (max_num_seqs full), 2 finish, 2 fresh
// 1024-prefills waiting. The admission step packs 2048 (1024 + 994 chunk) with 30
// running decodes. Sync and async IDENTICAL.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler wave: c32 admission step packs 2048 (sync == async oracle)") {
  auto sync = MakeScheduler<Scheduler>();
  auto async = MakeScheduler<AsyncScheduler>();
  auto rs = RunScript(*sync, /*N=*/32, /*inject_at=*/4, /*depth=*/1, /*steps=*/14);
  auto ra = RunScript(*async, /*N=*/32, /*inject_at=*/4, /*depth=*/2, /*steps=*/14);

  const StepRec& s4 = AtStep(rs, 4);
  const StepRec& a4 = AtStep(ra, 4);

  for (const StepRec* r : {&s4, &a4}) {
    CHECK(r->total == 2048);
    CHECK(r->prefill_tokens == 2018);
    CHECK(r->decode_tokens == 30);
    CHECK(r->n_prefill == 2);
    REQUIRE(r->prefill_counts.size() == 2);
    CHECK(r->prefill_counts[0] == 1024);
    CHECK(r->prefill_counts[1] == 994);
    CHECK(r->chunked == true);
  }
  CHECK(s4.total == a4.total);
  CHECK(s4.prefill_tokens == a4.prefill_tokens);
  CHECK(s4.prefill_counts == a4.prefill_counts);
  CHECK(s4.decode_tokens == a4.decode_tokens);
}

// ---------------------------------------------------------------------------
// The knife-edge the tails ride on: a fresh 1024-prompt admitted with ANY running
// decode forces the co-scheduled second prefill to chunk so the step fills to the
// full 2048 budget (the expensive stall), instead of one 1024-token step. This is
// identical on both sides — confirming the budget-fill (not a single-prefill cap)
// is what both schedulers do. (H-C: max_num_partial_prefills=1 /
// long_prefill_token_threshold=0 are inert both sides.)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler wave: second co-scheduled prefill chunks to fill 2048") {
  auto sync = MakeScheduler<Scheduler>();
  auto rs = RunScript(*sync, /*N=*/8, /*inject_at=*/4, /*depth=*/1, /*steps=*/14);
  const StepRec& s4 = AtStep(rs, 4);
  // Exactly one full 1024 prefill + one chunked remainder that fills the budget.
  CHECK(s4.prefill_counts.size() == 2);
  CHECK(s4.prefill_counts[0] == 1024);
  CHECK(s4.prefill_counts[1] < 1024);          // the second is chunked
  CHECK(s4.total == kBudget);                  // budget fully packed (~860 ms)
}
