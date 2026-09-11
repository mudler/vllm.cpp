// SPEC-DFLASH2 A2-4 (#3004) — the propose reads this step's committed ids where
// they live.
//
// WHAT IS UNDER TEST. `sample_tokens_async`'s decode arm has three write-back
// branches and two of them leave the step's committed ids unreadable on the
// host: the device-mirror branch keeps `input_batch_.last_sampled_tokens` stale
// ON PURPOSE (reading the ids back is the cost it exists to remove), and the UMA
// branch writes that array from a `LaunchScatterLastSampled` queued on the main
// queue with nothing waiting it. #2920 could not transplant the propose tail
// past that, so it set a `committed_ids_on_host` flag and refused by name. On
// GB10's integrated default the UMA branch runs, so that refusal is what blocks
// A2-5 from flipping the veto.
//
// A2-4 removes the question rather than answering it. `DownloadCommittedIds`
// brings the ids to the host on every branch, and `ApplyCommittedIdsForPropose`
// writes the two pieces of host state every propose arm reads from that one
// array. Both are exercised here.
//
// THE RED THIS FILE WAS WRITTEN AGAINST. #2920's inline tail restored the token
// row FROM `input_batch_.last_sampled_tokens`, which is precisely the array the
// mirror branch leaves stale — so the restore was correct only on the branch
// that did not need it. `PropagatesCommittedIdsNotTheStaleAnchor` below drives
// that: seed `last_sampled_tokens` with the previous step's ids, hand this step's
// DIFFERENT committed ids, and require both arrays to carry this step's. With
// the old rule the token row carries the previous step's id and the anchor never
// moves at all. That defect costs ACCEPTANCE and emits identical text — the
// class of defect no token gate in this tree can see (#1366) — which is why it
// gets an assertion of its own rather than an end-to-end run.
//
// THE TWO `DownloadCommittedIds` CASES HAVE A WEAKER RED and this file says so
// rather than letting the run's summary imply otherwise. There was no
// pre-change version of that function to transcribe, so its red is the compile
// failure of a symbol that did not exist; in the run that reds the six cases
// above, these two pass. What they gate afterwards is the sizing, the staging
// retain rule and the ids that come back.
//
// WHY EVERY CASE THAT MATTERS HAS `num_reqs > 1` AND A MIXED DISCARD MASK. A
// per-request rule feeding a per-STEP route is the shape that survived 27
// mutations in `test_combine_row_predicate.cpp` (#2710) because every case used
// `num_reqs == 1`: at one request a per-request and a per-step reading agree on
// every input. The mixed step — some rows committing, some discarded, a
// different `num_tokens_no_spec` per row — is what separates them.
//
// WHAT THIS FILE DOES NOT CLAIM. Nothing about overlap. On the CPU backend
// `Copy` is a memcpy, `CreateQueue` returns the same null handle as the main
// queue, and every event call is a no-op, so `DownloadCommittedIds` here gates
// its SIZING, its staging retain rule and the ids it returns — never that the
// copy ran off the main stream. That is G3/G4 at A2-5 and it needs a GPU.
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "vllm/v1/worker/gpu/prepare_inputs.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::v1::ApplyCommittedIdsForPropose;
using vllm::v1::DownloadCommittedIds;
using vllm::v1::PinnedGrowStaging;

namespace {

constexpr int kMaxModelLen = 8;

// A [num_slots, kMaxModelLen] row-major token buffer seeded so that a write into
// the wrong row or the wrong column is unmistakable.
std::vector<int32_t> TokenRows(int num_slots) {
  std::vector<int32_t> t(static_cast<size_t>(num_slots) * kMaxModelLen);
  for (int r = 0; r < num_slots; ++r) {
    for (int c = 0; c < kMaxModelLen; ++c) {
      t[static_cast<size_t>(r) * kMaxModelLen + c] = 1000 * r + c;
    }
  }
  return t;
}

int32_t At(const std::vector<int32_t>& t, int row, int col) {
  return t[static_cast<size_t>(row) * kMaxModelLen + static_cast<size_t>(col)];
}

}  // namespace

// ─── ApplyCommittedIdsForPropose ────────────────────────────────────────────

// THE A2-4 RULE, on a mixed step. Rows 0 and 2 committed a token this step; row
// 1 is still prefilling and its write-back was skipped. Every row carries a
// DIFFERENT `num_tokens_no_spec`, so a per-step reading of the write column is
// wrong for at least one of them.
TEST_CASE("ApplyCommittedIdsForPropose writes the committed rows only") {
  const std::vector<int64_t> committed = {77, 88, 99};
  const std::vector<uint8_t> discard = {0, 1, 0};
  const std::vector<int32_t> num_tokens_no_spec = {3, 5, 1};
  std::vector<int32_t> last_sampled = {11, 22, 33};
  std::vector<int32_t> tokens = TokenRows(3);
  const std::vector<int32_t> before = tokens;

  ApplyCommittedIdsForPropose(committed.data(), 3, discard, num_tokens_no_spec,
                              kMaxModelLen, &last_sampled, &tokens);

  // The anchor `propose_drafts_block` reads.
  CHECK(last_sampled[0] == 77);
  CHECK(last_sampled[2] == 99);
  // The column `propose_drafts_ngram` matches over: num_tokens_no_spec - 1, per
  // row, so 2 for row 0 and 0 for row 2.
  CHECK(At(tokens, 0, 2) == 77);
  CHECK(At(tokens, 2, 0) == 99);

  // The discarded row is untouched in BOTH arrays. It produced no output token,
  // its counter was not advanced, and writing its column would overwrite the
  // newest token it actually committed.
  CHECK(last_sampled[1] == 22);
  for (int c = 0; c < kMaxModelLen; ++c) {
    CHECK(At(tokens, 1, c) == At(before, 1, c));
  }

  // Nothing outside the two written columns moved.
  for (int r : {0, 2}) {
    const int written = num_tokens_no_spec[static_cast<size_t>(r)] - 1;
    for (int c = 0; c < kMaxModelLen; ++c) {
      if (c == written) continue;
      CHECK(At(tokens, r, c) == At(before, r, c));
    }
  }
}

// THE DEFECT #2920's INLINE TAIL CARRIES, stated as an assertion. That tail
// restored the token row from `last_sampled_tokens`, which on the mirror branch
// still holds the PREVIOUS step's ids and which it never updated. Here the
// seeded anchor and this step's committed ids disagree on every row, so a rule
// that reads the anchor writes the wrong token and moves no anchor.
TEST_CASE("ApplyCommittedIdsForPropose propagates the committed ids, not the stale anchor") {
  const std::vector<int64_t> committed = {500, 501};
  const std::vector<uint8_t> discard = {0, 0};
  const std::vector<int32_t> num_tokens_no_spec = {4, 2};
  // What the mirror branch leaves behind: the PREVIOUS step's ids.
  std::vector<int32_t> last_sampled = {400, 401};
  std::vector<int32_t> tokens = TokenRows(2);

  ApplyCommittedIdsForPropose(committed.data(), 2, discard, num_tokens_no_spec,
                              kMaxModelLen, &last_sampled, &tokens);

  CHECK(last_sampled[0] == 500);
  CHECK(last_sampled[1] == 501);
  CHECK(At(tokens, 0, 3) == 500);
  CHECK(At(tokens, 1, 1) == 501);
  // And specifically NOT the stale anchor, which is what the old rule wrote.
  CHECK(At(tokens, 0, 3) != 400);
  CHECK(At(tokens, 1, 1) != 401);
}

// A `discard` mask shorter than the step's request count. The runner builds it
// per step and every other loop over it in `runner.cpp` treats a row past its
// end as committing; so does this.
TEST_CASE("ApplyCommittedIdsForPropose treats a row past the mask as committing") {
  const std::vector<int64_t> committed = {7, 8, 9};
  const std::vector<uint8_t> discard = {1};  // only row 0 is described
  const std::vector<int32_t> num_tokens_no_spec = {2, 2, 3};
  std::vector<int32_t> last_sampled = {0, 0, 0};
  std::vector<int32_t> tokens = TokenRows(3);

  ApplyCommittedIdsForPropose(committed.data(), 3, discard, num_tokens_no_spec,
                              kMaxModelLen, &last_sampled, &tokens);

  CHECK(last_sampled[0] == 0);  // discarded
  CHECK(last_sampled[1] == 8);
  CHECK(last_sampled[2] == 9);
  CHECK(At(tokens, 1, 1) == 8);
  CHECK(At(tokens, 2, 2) == 9);
}

// THE TWO SILENT SKIPS #2920 LANDED, now refusals. A fail-open `continue` and a
// fail-closed refusal are indistinguishable until something mutates them, which
// is the finding three A2-3 reviews each made independently.
TEST_CASE("ApplyCommittedIdsForPropose refuses a committed row with no committed token") {
  const std::vector<int64_t> committed = {1, 2};
  const std::vector<uint8_t> discard = {0, 0};
  // Row 1 committed a token this step (not discarded) but its counter says it
  // holds none: the write-back and the counter disagree.
  const std::vector<int32_t> num_tokens_no_spec = {2, 0};
  std::vector<int32_t> last_sampled = {0, 0};
  std::vector<int32_t> tokens = TokenRows(2);

  CHECK_THROWS(ApplyCommittedIdsForPropose(committed.data(), 2, discard,
                                           num_tokens_no_spec, kMaxModelLen,
                                           &last_sampled, &tokens));
}

TEST_CASE("ApplyCommittedIdsForPropose refuses a write column outside the row") {
  const std::vector<int64_t> committed = {1, 2};
  const std::vector<uint8_t> discard = {0, 0};
  // kMaxModelLen is 8, so column 8 is one past the end of row 1.
  const std::vector<int32_t> num_tokens_no_spec = {2, kMaxModelLen + 1};
  std::vector<int32_t> last_sampled = {0, 0};
  std::vector<int32_t> tokens = TokenRows(2);

  CHECK_THROWS(ApplyCommittedIdsForPropose(committed.data(), 2, discard,
                                           num_tokens_no_spec, kMaxModelLen,
                                           &last_sampled, &tokens));
}

TEST_CASE("ApplyCommittedIdsForPropose refuses ids the write-back never materialized") {
  const std::vector<uint8_t> discard = {0};
  const std::vector<int32_t> num_tokens_no_spec = {1};
  std::vector<int32_t> last_sampled = {0};
  std::vector<int32_t> tokens = TokenRows(1);

  CHECK_THROWS(ApplyCommittedIdsForPropose(nullptr, 1, discard,
                                           num_tokens_no_spec, kMaxModelLen,
                                           &last_sampled, &tokens));
}

// A zero-request step writes nothing and refuses nothing: `num_reqs == 0` is the
// flush step, and it is reached with a null id pointer.
TEST_CASE("ApplyCommittedIdsForPropose is a no-op at zero requests") {
  const std::vector<uint8_t> discard;
  const std::vector<int32_t> num_tokens_no_spec;
  std::vector<int32_t> last_sampled;
  std::vector<int32_t> tokens;
  ApplyCommittedIdsForPropose(nullptr, 0, discard, num_tokens_no_spec,
                              kMaxModelLen, &last_sampled, &tokens);
  CHECK(tokens.empty());
}

// ─── DownloadCommittedIds ───────────────────────────────────────────────────

// The ids that come back are the ids the sampler wrote, and the block they come
// back in is page-locked staging this object owns. On this backend the copy is a
// memcpy and the events are no-ops; the SIZING (two int32 elements per int64 id)
// and the returned values are what this gates.
TEST_CASE("DownloadCommittedIds returns the sampler's ids") {
  vt::Device dev{vt::DeviceType::kCPU, 0};
  vt::Backend& backend = vt::GetBackend(dev.type);
  vt::Queue main_q = vt::CreateQueue(dev);
  vt::Queue copy_q = vt::CreateQueue(dev);
  vt::Event fork = backend.CreateEvent();
  vt::Event ready = backend.CreateEvent();
  PinnedGrowStaging staging;

  const std::vector<int64_t> sampled = {5, 4, 3, 2};
  void* dev_ids = backend.Alloc(sampled.size() * sizeof(int64_t));
  backend.Copy(main_q, dev_ids, sampled.data(),
               sampled.size() * sizeof(int64_t));

  const int64_t* host = DownloadCommittedIds(backend, main_q, copy_q, fork, ready,
                                             dev_ids, 4, staging);
  REQUIRE(host != nullptr);
  for (int i = 0; i < 4; ++i) CHECK(host[i] == sampled[static_cast<size_t>(i)]);
  // Two int32 elements per id, so the block is sized 2 * num_reqs.
  CHECK(staging.elems() == 8);
  CHECK(staging.live_blocks() == 1);

  // A LARGER step grows the block and RETAINS the one it replaces, which is what
  // makes a copy still in flight into the old block safe.
  const std::vector<int64_t> bigger = {9, 8, 7, 6, 5, 4};
  void* dev_ids2 = backend.Alloc(bigger.size() * sizeof(int64_t));
  backend.Copy(main_q, dev_ids2, bigger.data(),
               bigger.size() * sizeof(int64_t));
  const int64_t* host2 = DownloadCommittedIds(backend, main_q, copy_q, fork, ready,
                                              dev_ids2, 6, staging);
  REQUIRE(host2 != nullptr);
  for (int i = 0; i < 6; ++i) CHECK(host2[i] == bigger[static_cast<size_t>(i)]);
  CHECK(staging.live_blocks() == 2);

  backend.Free(dev_ids);
  backend.Free(dev_ids2);
  backend.DestroyEvent(fork);
  backend.DestroyEvent(ready);
}

TEST_CASE("DownloadCommittedIds refuses a step that sampled into no buffer") {
  vt::Device dev{vt::DeviceType::kCPU, 0};
  vt::Backend& backend = vt::GetBackend(dev.type);
  vt::Queue main_q = vt::CreateQueue(dev);
  vt::Queue copy_q = vt::CreateQueue(dev);
  vt::Event fork = backend.CreateEvent();
  vt::Event ready = backend.CreateEvent();
  PinnedGrowStaging staging;

  CHECK_THROWS(DownloadCommittedIds(backend, main_q, copy_q, fork, ready,
                                    nullptr, 2, staging));
  // A zero-request step is the flush step: no buffer, no copy, no refusal.
  CHECK(DownloadCommittedIds(backend, main_q, copy_q, fork, ready, nullptr, 0,
                             staging) == nullptr);

  backend.DestroyEvent(fork);
  backend.DestroyEvent(ready);
}
