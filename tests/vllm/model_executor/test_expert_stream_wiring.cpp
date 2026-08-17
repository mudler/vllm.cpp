// ENG-EXPERT-STREAM (#912) F3: does DECODE actually reach the streamed-expert
// lane, and does the lane stay alive for the whole run?
//
// WHY THIS FILE EXISTS. Every other test of this row constructs the cache, the
// store and the streamer by hand and drives them directly. All of them passed
// while the production wiring was broken in two separate ways, because none of
// them ran a forward. An independent review measured exactly that: replacing the
// production call site with `nullptr`, and forcing streaming unconditionally ON,
// BOTH left the full gate green, and `Qwen35ExpertStream`, `KqExpertSlice` and
// `VT_MOE_EXPERT_STREAM` appeared nowhere under `tests/` at all. A unit test that
// builds the type by hand proves the class works; it never proves anything
// reaches it. See `.agents/reachability.md`.
//
// So this test enters through a PRODUCTION entry point — `Qwen3_5Model::Forward`,
// the paged forward the runner calls — over a synthetic MoE whose routed experts
// are keep-quant stacked towers, which is the shape the streaming seam serves.
// It then asks the lane what happened.
//
// THE TWO NUMBERS THAT MATTER, and why they are the ones asserted:
//
//   fills > 0   decode reached the streamer at all. Zero means the slice seam
//               is no longer wired, which is the mutation that used to pass.
//   steps == N  the step boundary ran once per forward. Zero means nothing
//               calls it, which is the defect that voided this row's published
//               decode number: `Acquire` protects every entry it serves and only
//               `EndStep` clears that protection, so a cache with no step clock
//               refuses every slice after it first fills, silently falls back to
//               the mapping, and never once exercises its own eviction policy.
//   exhausted == 0
//               nothing was refused. This is `steps == 0` seen from the other
//               side, and it is the number a benchmark can read.
//
// THIS BINARY IS DEDICATED TO STREAMING-ON. `VT_MOE_EXPERT_STREAM` is read once
// into a function-local static on first use, so a single process cannot run both
// arms by changing the environment. The comparison against the unstreamed arm is
// therefore made through the cache-exhaustion fallback, which is a REAL
// production state (a budget below one step's working set reaches it) and takes
// the identical `KqResidentSlice` path an OFF build takes.
#include <stdlib.h>

#include <doctest/doctest.h>

#if !defined(_WIN32)
#include <unistd.h>  // ::fileno, for the pread case below
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "support/expert_stream_model.h"
#include "support/test_env.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"

using expert_stream_test::CachePool;
using expert_stream_test::MakeConfig;
using expert_stream_test::MakeWeights;
using expert_stream_test::PrefillAttnMeta;
using expert_stream_test::PrefillGdnMeta;
using expert_stream_test::Q;
using vllm::HfConfig;
using vllm::Qwen3_5MoeWeights;
using vllm::Qwen3_5Model;

namespace {

// Turn the lane on BEFORE anything can read the environment. The read happens in
// a function-local static on the first slice, so setting it inside a test body
// would work today and break the moment a case ordering changed.
//
// Through `vllm_test::SetEnv` and not `::setenv`. `setenv(3)` is POSIX, MSVC's
// CRT has only `_putenv_s`, and this file is compiled on Windows too: the target
// is added unconditionally and `scripts/build-windows-release.ps1` configures
// `VLLM_CPP_BUILD_TESTS=ON`. The shim in `support/test_env.h` is the one place
// that branch lives (#603). CI cannot currently see the difference, because the
// Windows lanes fail earlier in the product library on #1068 and never reach a
// test translation unit at all.
struct EnableExpertStreaming {
  EnableExpertStreaming() {
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM", "1");
    // Comfortably more than one step's working set: 4 experts x 3 towers x 4
    // layers = 48 distinct slices per forward. A budget BELOW that would make
    // `exhausted` nonzero for an honest reason and mask the defect under test.
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_SLOTS", "64");
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_SLOT_BYTES", "8192");
    // Quiet under ctest, but only when the operator has not asked otherwise --
    // seeing the line is the only way to SEE the statistics this row added, and
    // a gate that suppresses its own evidence is a smaller version of the defect
    // it was written for. `vllm_test::SetEnv` has no overwrite=0 form (it is a
    // two-argument shim on purpose), so the condition is stated here.
    if (std::getenv("VT_MOE_EXPERT_STREAM_STATS_EVERY") == nullptr)
      vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_STATS_EVERY", "0");
    // The grouped keep-quant path stages the whole tower and cannot stream; the
    // production code already disables it when streaming is requested. Being
    // explicit here keeps the test honest about which path it is measuring.
    vllm_test::SetEnv("VT_QWEN35_GROUPED_MOE", "0");
  }
};
const EnableExpertStreaming kEnableExpertStreaming;


// One full paged forward through the PRODUCTION entry point, over a fresh cache
// so every call is independent.
std::vector<float> OneForward(const HfConfig& c, const Qwen3_5MoeWeights& w,
                              const std::vector<int32_t>& ids) {
  vt::Queue q = Q();
  const int64_t T = static_cast<int64_t>(ids.size());
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/8);
  const std::vector<int32_t> blocks = {0};
  return Qwen3_5Model::Forward(ids, pos, PrefillAttnMeta(T, blocks, 8, 0),
                               PrefillGdnMeta(T, 0), pool.attn_kv, pool.gdn_state,
                               w, c, q, {});
}

}  // namespace

TEST_CASE("decode REACHES the expert streamer, and the step clock advances") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};

  // Nothing has run: the lane must not have built a store just by existing.
  {
    const vllm::detail::ExpertStreamStats s0 = vllm::detail::ExpertStreamSnapshot();
    CHECK_FALSE(s0.active);
    CHECK(s0.steps == 0);
  }

  const int kSteps = 3;
  for (int i = 0; i < kSteps; ++i) {
    const std::vector<float> logits = OneForward(c, w, ids);
    REQUIRE(logits.size() ==
            static_cast<size_t>(ids.size()) * static_cast<size_t>(c.vocab_size));
    for (float v : logits) REQUIRE(std::isfinite(v));
  }

  const vllm::detail::ExpertStreamStats s = vllm::detail::ExpertStreamSnapshot();

  // REACHABILITY of the slice seam. Zero fills means decode no longer enters
  // KqExpertSlice's streaming branch at all: the mutation that replaced the
  // production call site with `nullptr` used to leave the whole gate green.
  CHECK(s.active);
  CHECK(s.fills > 0);
  CHECK(s.bytes_filled > 0);
  CHECK((s.hits + s.misses) > 0);

  // REACHABILITY of the STEP BOUNDARY, which is the finding that voided this
  // row's published decode number. One step per forward, no more and no fewer:
  // once per layer would give 12 here and once per expert far more, and both are
  // wrong for the cache's protection semantics.
  CHECK(s.steps == kSteps);

  // The consequence of the two above, and the number a benchmark can read. A
  // cache whose step never ends protects every entry forever, refuses every
  // slice once full, and silently serves the mapping instead.
  CHECK(s.exhausted == 0);

#if defined(__unix__)
  // F5: the MADV_WILLNEED hint is ACCEPTED on EVERY fill, not merely issued.
  //
  // madvise(2) returns EINVAL on an address that is not page-aligned, and GGUF
  // tensor data is aligned to `general.alignment`, default 32
  // (gguf_reader.cpp:401), so a slice address is essentially never a page
  // boundary. The call was made on the raw address with its return value
  // discarded, which is a hint that never fired and never said so. This counts
  // only the calls the kernel took.
  //
  // WHY `== fills` AND NOT `> 0` (#1091 finding 2). `> 0` over 48 calls is
  // satisfied whenever heap layout happens to page-align a single slice, and
  // measured here it is: reinjecting the pre-fix unaligned address exits 0 in
  // 40 of 40 runs against `> 0`, so the assertion the fix shipped with cannot
  // fail for the defect it names. The equality can: 0 != 48.
  //
  // AND `fills` IS THE RIGHT DENOMINATOR, not a literal 48. madvise runs on the
  // mapping-copy arm only (the pread arm needs no readahead hint) and only when
  // the key is NOT already resident, which is exactly the condition under which
  // `EnsureSpan` goes on to fill. The two counters therefore move together for
  // as long as nothing is refused, and `exhausted == 0` above is that premise
  // asserted. Every weight in this test owns its bytes (`mmap_fd == -1`), so
  // every fill here is a span fill.
  //
  // NO SPEEDUP IS ASSERTED, here or anywhere. This says the call is well formed.
  //
  // TWO RESIDUALS THE EQUALITY RESTS ON, stated rather than left to be
  // rediscovered.
  //
  // (1) `Slice` rounds the advised range's END UP to a page, which for a
  // heap-backed tower goes past the allocation. madvise(2) returns ENOMEM if
  // any page in the range is unmapped, so this holds because the allocator's
  // arena page is mapped, not because the arithmetic guarantees it. Production
  // towers are file mappings many pages larger than a slice and do not have the
  // question. If this ever fails with `advised` short by a small count, that is
  // the first thing to check, not the fill path.
  //
  // (2) The counters are CUMULATIVE over the process, so this equality is a
  // statement about everything that ran BEFORE it — and the pread case at the
  // end of this file fills without advising, which would break it. The order
  // holds under doctest's default file order, and it is not left implicit: the
  // `CHECK_FALSE(s0.active)` at the top of this case fails loudly if anything
  // ran first, so a reordering shows up as that assertion rather than as a
  // confusing `advised != fills` here.
  CHECK(s.advised == s.fills);
  CHECK(s.advised > 0);
#endif
}

TEST_CASE("a streamed slice and the tower view produce IDENTICAL logits") {
  // "Streaming is byte-identical" in both directions, inside one process.
  //
  // The unstreamed arm is reached through the cache-exhaustion fallback rather
  // than through the environment, because `VT_MOE_EXPERT_STREAM` is read once
  // into a function-local static and a single process cannot see it change.
  // That fallback is not a test fiction: it is the branch a real budget below
  // one step's working set takes, and it runs the same KqResidentSlice the
  // streaming-OFF build runs.
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const std::vector<int32_t> ids = {7, 1, 22, 4};

  vllm::detail::ExpertStreamSetForceFallback(false);
  const std::vector<float> streamed = OneForward(c, w, ids);
  const vllm::detail::ExpertStreamStats on = vllm::detail::ExpertStreamSnapshot();
  REQUIRE(on.active);
  const int64_t fills_after_streamed = on.fills;
  REQUIRE(fills_after_streamed > 0);  // the streamed arm really streamed

  vllm::detail::ExpertStreamSetForceFallback(true);
  const std::vector<float> tower = OneForward(c, w, ids);
  const vllm::detail::ExpertStreamStats off = vllm::detail::ExpertStreamSnapshot();
  vllm::detail::ExpertStreamSetForceFallback(false);

  // The unstreamed arm really did NOT stream: no new bytes moved, and every
  // slice it asked for was refused into the fallback.
  CHECK(off.fills == fills_after_streamed);
  CHECK(off.forced > on.forced);

  // AND IT WAS NOT COUNTED AS A BUDGET REFUSAL (#1091 finding 6). `exhausted` is
  // the operator-facing number, documented as "the budget is smaller than one
  // step's working set". The forced-fallback switch has no production caller, so
  // every increment it contributed to `exhausted` was a gate telling an operator
  // that a knob they never turned is too small. The budget here is 64 slots
  // against 48 slices and nothing was ever genuinely refused, so this stays 0.
  CHECK(off.exhausted == 0);

  // BIT-EXACT, not close. The slot holds a byte copy of the same tower bytes, so
  // the two arms feed the kernel identical inputs; anything but equality means
  // the copy is wrong, which a tolerance-based check would hide.
  REQUIRE(streamed.size() == tower.size());
  size_t differing = 0;
  for (size_t i = 0; i < streamed.size(); ++i)
    if (!(streamed[i] == tower[i])) ++differing;
  CHECK(differing == 0);
}

TEST_CASE("a SECOND model does not inherit the first model's slots") {
  // Found by the reachability case above, and not by anything before it.
  //
  // The slot store is a process-lifetime singleton, so it outlives any one
  // model. Its tower identity used to be the weight buffer's ADDRESS, whose
  // comment claimed it was "stable for the model's life because the tower is a
  // borrowed view into the mapping". The premise is true and the conclusion does
  // not follow: the CACHE is not scoped to one model's life. Free a model, load
  // another, and the allocator hands the new towers addresses the old ones held,
  // so the new model's expert resolves to an entry filled from a different
  // checkpoint. It comes back as a HIT, and a hit moves no bytes by contract, so
  // there is nothing at all to observe downstream -- just wrong weights.
  //
  // Measured, on exactly this shape: 24 towers occupied 21 distinct addresses,
  // and 20 of 222 slices returned another tower's bytes.
  const HfConfig c = MakeConfig();

  // A first model, run and then DESTROYED, so its buffers go back to the
  // allocator while the cache keeps its entries.
  {
    const Qwen3_5MoeWeights a = MakeWeights(c, /*base_seed=*/0);
    (void)OneForward(c, a, {5, 9, 2, 31});
  }

  // A second model with DIFFERENT weights. Identical weights would make an
  // address collision invisible, which is precisely why this needs its own seed.
  const Qwen3_5MoeWeights b = MakeWeights(c, /*base_seed=*/777000);
  const std::vector<int32_t> ids = {5, 9, 2, 31};

  const std::vector<float> streamed = OneForward(c, b, ids);

  // The ground truth for THIS model, taken through the fallback, which reads the
  // tower directly and so cannot be poisoned by a stale entry.
  vllm::detail::ExpertStreamSetForceFallback(true);
  const std::vector<float> truth = OneForward(c, b, ids);
  vllm::detail::ExpertStreamSetForceFallback(false);

  REQUIRE(streamed.size() == truth.size());
  size_t differing = 0;
  for (size_t i = 0; i < streamed.size(); ++i)
    if (!(streamed[i] == truth[i])) ++differing;
  CHECK(differing == 0);
}

#if !defined(_WIN32)
TEST_CASE("a FILE-backed tower is served by pread, at file_offset + slice offset") {
  // #1091 finding 4: `ExpertStreamer::EnsureFile` is the arm every REAL GGUF
  // checkpoint takes, because a borrowed mmap tower carries a descriptor, and
  // no test reached it. The other cases in this file build owned host vectors,
  // so `w.mmap_fd` is -1 and every one of them exercises `EnsureSpan` instead.
  // The spec's `## Owed` framed this as unmeasured on the model, which it also
  // is; it was additionally UNREACHED, and that part needs neither the box nor
  // the 370 GiB checkpoint.
  //
  // WHAT IS ACTUALLY UNVERIFIED HERE is the address arithmetic. `Slice` preads
  // at `file_offset + offset`: the tensor's own position in the shard plus the
  // routed expert's row offset within the tower. Get either term wrong and the
  // read still succeeds, still returns exactly the bytes asked for, and the GEMM
  // multiplies a different expert -- the "wrong shard at a plausible offset"
  // shape F4 named. So the tower is written at a deliberately AWKWARD file
  // offset: nonzero, not page-aligned and not a multiple of the 34-byte Q8_0
  // block, which makes dropping the term or rounding it detectable.
  const HfConfig c = MakeConfig();
  Qwen3_5MoeWeights w = MakeWeights(c, /*base_seed=*/31000);

  std::FILE* f = std::tmpfile();
  REQUIRE(f != nullptr);
  const int fd = ::fileno(f);
  REQUIRE(fd >= 0);

  // 4109 = 4096 + 13: past a page, not on a page, not on a Q8_0 block.
  size_t at = 4109;
  const std::vector<uint8_t> pad(at, 0xA5);
  REQUIRE(std::fwrite(pad.data(), 1, pad.size(), f) == pad.size());

  // Keep-alives for the borrowed views. A borrow with no owner is exactly the
  // dangling view OwnedBytes exists to make unrepresentable.
  std::vector<std::shared_ptr<std::vector<uint8_t>>> holds;

  auto to_file_backed = [&](vllm::OwnedTensor& t) {
    auto hold = std::make_shared<std::vector<uint8_t>>(t.bytes.begin(),
                                                       t.bytes.end());
    REQUIRE(std::fwrite(hold->data(), 1, hold->size(), f) == hold->size());
    t.bytes = vllm::OwnedBytes::Borrow(hold->data(), hold->size(), hold);
    t.mmap_fd = fd;
    t.mmap_file_offset = at;
    at += hold->size();
    holds.push_back(std::move(hold));
  };
  for (auto& layer : w.layers) {
    to_file_backed(layer.moe.expert_gate_kq);
    to_file_backed(layer.moe.expert_up_kq);
    to_file_backed(layer.moe.expert_down_kq);
  }
  REQUIRE(std::fflush(f) == 0);  // the pread must see the bytes, not the buffer

  const std::vector<int32_t> ids = {13, 6, 28, 2};

  const vllm::detail::ExpertStreamStats before =
      vllm::detail::ExpertStreamSnapshot();
  const std::vector<float> streamed = OneForward(c, w, ids);
  const vllm::detail::ExpertStreamStats after =
      vllm::detail::ExpertStreamSnapshot();

  // THE ARM IS PROVEN, not assumed. Slices really were filled, and NOT ONE of
  // them was advised: `Slice` issues MADV_WILLNEED only on the mapping-copy arm,
  // because a pread needs no readahead hint. Equal `advised` across a forward
  // that filled slots is therefore the signature of the pread path, and it is
  // the one number that separates it from EnsureSpan.
  CHECK(after.fills > before.fills);
  CHECK(after.advised == before.advised);

  // The ground truth for the same weights, read straight from the borrowed host
  // bytes through the fallback. Those bytes and the file's are the same bytes by
  // construction, so any difference is the pread landing somewhere else.
  vllm::detail::ExpertStreamSetForceFallback(true);
  const std::vector<float> truth = OneForward(c, w, ids);
  vllm::detail::ExpertStreamSetForceFallback(false);

  REQUIRE(streamed.size() == truth.size());
  size_t differing = 0;
  for (size_t i = 0; i < streamed.size(); ++i)
    if (!(streamed[i] == truth[i])) ++differing;
  CHECK(differing == 0);

  std::fclose(f);
}
#endif  // !_WIN32
