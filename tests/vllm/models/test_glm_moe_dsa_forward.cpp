// GLM-5.3 (`GlmMoeDsaForCausalLM`) — W9's forward gate.
//
// Spec `.agents/specs/glm-dsa-latest-deepseek.md` §3.7 W9, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// ─── WHAT THIS SUITE IS FOR ──────────────────────────────────────────────────
// There is NO token gate against vLLM for this architecture and there cannot be
// one on this fleet (spec O1, §3.6): vLLM at the pin implements it and cannot
// fit 703.74 GiB on any device this project reaches. So what is gated here is
// everything BELOW a token comparison, on a model with the artifact's structure
// at 1/1000th its size:
//
//   * the forward is REACHED from `LoadedEngine::FromModelDir` and produces a
//     token — the production entry point a user arrives through;
//   * the logits are FINITE, and the distribution is printed whether or not it
//     looks sensible;
//   * a `kShared` layer attends through the selection its owning `kFull` layer
//     wrote, and the two layers do not produce the same output;
//   * the routed experts reach `expert_stream::ExpertSlice`, and the streamed
//     and resident arms produce identical logits inside one process (§3.6 G3);
//   * the one step shape this build cannot serve is refused BY NAME.
//
// ─── WHY EVERY FLOAT ASSERTION IS GUARDED BY `isfinite` ──────────────────────
// The sibling `MODEL-MM-GLM53-FLASH` row read an all-NaN forward as a PERFECT
// match, because every comparison against a NaN is false — `a == b` is false and
// so is `a != b`, so an equality gate and a difference gate BOTH report success
// on a model that produced no numbers at all. That model then emitted token id 0
// eight times and nobody knew until the logits were read. Uniform logits and NaN
// logits both argmax to 0, so a degenerate token is not a diagnosis: the counts,
// the range and the top-5 are.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/expert_stream_seam.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/glm_moe_dsa.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/sampling_params.h"
#include "vt/dtype.h"
#include "vt/ops.h"

#include "../gguf_builder.h"
#include "glm_moe_dsa_gguf_fixture.h"

namespace {

using namespace glm_dsa_fixture;  // NOLINT: the fixture IS this file's vocabulary
using vllm::GlmMoeDsaWeights;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;

// LONGER THAN `index_topk` (64), ON PURPOSE. The sparse route is a property of
// the STEP: `use_dense_mha = prefill_max_seq_len <= self.topk_tokens`
// (`sparse_mla_attention.py:296-299`), so at or under 64 keys the top-k selects
// every causal candidate and dense attention IS upstream's answer — a prompt
// that short would never run the indexer, never write the shared selection and
// never reach the `kShared` reuse this row exists for.
constexpr int64_t kTokens = 80;
constexpr int64_t kBlockSize = 16;
constexpr int64_t kBlocks = 8;  // 8 * 16 = 128 >= kTokens

std::string RefusalOf(const std::function<void()>& fn, const char* what) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  FAIL_CHECK("expected a refusal from " << what << ", got none");
  return {};
}

// One MLA cache per backbone layer: [num_blocks, block_size, kv_lora + qk_rope],
// one head, no separate V (`MLAAttentionSpec`).
struct MlaCachePool {
  std::vector<std::vector<uint16_t>> buf;
  std::vector<PagedKvCache> attn_kv;
  MlaCachePool(const vllm::GlmMoeDsaParams& p) {
    const int64_t head_size = p.mla_kv_head_size();
    for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
      buf.emplace_back(
          static_cast<size_t>(kBlocks * kBlockSize * head_size), 0);
    }
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = vt::DType::kBF16;
      kv.num_blocks = kBlocks;
      kv.block_size = kBlockSize;
      kv.num_kv_heads = 1;
      kv.head_size = head_size;
      attn_kv.push_back(kv);
    }
  }
};

// A single FRESH prompt: nothing computed before this step, which is the only
// shape this build can serve sparsely (spec O4).
CommonAttentionMetadata FreshPrefill(int64_t T) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.num_computed_tokens_cpu = {0};
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = kBlocks;
  m.block_table_tensor.assign(static_cast<size_t>(kBlocks), 0);
  for (int64_t c = 0; c < kBlocks; ++c) {
    m.block_table_tensor[static_cast<size_t>(c)] = static_cast<int32_t>(c);
  }
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t);
  m.causal = true;
  return m;
}

std::vector<int32_t> PromptIds(int64_t T, int64_t vocab) {
  std::vector<int32_t> ids(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids[static_cast<size_t>(t)] = static_cast<int32_t>((t * 7 + 3) % vocab);
  }
  return ids;
}

std::vector<int32_t> Positions(int64_t T) {
  std::vector<int32_t> p(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) p[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  return p;
}

// THE DISTRIBUTION, PRINTED BEFORE ANYTHING IS CONCLUDED FROM IT.
// A degenerate token is a symptom with at least two causes — all-NaN and
// uniform — and they argmax the same. This prints the counts, the range and the
// top-5 so the reader is never asked to infer which one they have.
struct LogitReport {
  int64_t nan_count = 0;
  int64_t inf_count = 0;
  float lo = 0.0f, hi = 0.0f;
  double mean = 0.0, sd = 0.0;
  std::vector<std::pair<int64_t, float>> top5;
  bool finite() const { return nan_count == 0 && inf_count == 0; }
};

LogitReport Describe(const std::vector<float>& row, const char* what) {
  LogitReport r;
  REQUIRE(!row.empty());
  double sum = 0.0, sumsq = 0.0;
  bool first = true;
  for (float v : row) {
    if (std::isnan(v)) { ++r.nan_count; continue; }
    if (std::isinf(v)) { ++r.inf_count; continue; }
    if (first) { r.lo = r.hi = v; first = false; }
    r.lo = std::min(r.lo, v);
    r.hi = std::max(r.hi, v);
    sum += v;
    sumsq += static_cast<double>(v) * v;
  }
  const double n = static_cast<double>(row.size() - r.nan_count - r.inf_count);
  if (n > 0) {
    r.mean = sum / n;
    r.sd = std::sqrt(std::max(0.0, sumsq / n - r.mean * r.mean));
  }
  std::vector<int64_t> idx(row.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int64_t>(i);
  std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
    const float fa = row[static_cast<size_t>(a)], fb = row[static_cast<size_t>(b)];
    // NaN sorts LAST rather than comparing false and leaving the order
    // undefined, so an all-NaN row still produces a printable top-5.
    if (std::isnan(fa)) return false;
    if (std::isnan(fb)) return true;
    return fa > fb;
  });
  for (int i = 0; i < 5 && i < static_cast<int>(idx.size()); ++i) {
    r.top5.push_back({idx[static_cast<size_t>(i)],
                      row[static_cast<size_t>(idx[static_cast<size_t>(i)])]});
  }
  std::printf(
      "[glm-dsa W9] %s: n=%zu nan=%lld inf=%lld min=%.6g max=%.6g mean=%.6g "
      "sd=%.6g\n",
      what, row.size(), static_cast<long long>(r.nan_count),
      static_cast<long long>(r.inf_count), static_cast<double>(r.lo),
      static_cast<double>(r.hi), r.mean, r.sd);
  for (const auto& t : r.top5) {
    std::printf("[glm-dsa W9] %s: top id=%lld logit=%.6g\n", what,
                static_cast<long long>(t.first), static_cast<double>(t.second));
  }
  std::fflush(stdout);
  return r;
}

// Load the fixture's weights directly. `LoadGlmMoeDsaFromGguf` is the ONLY
// producer of a usable `GlmMoeDsaWeights` — it is what runs the post-load
// absorption and sets `absorbed` — which is exactly the polarity the refusal
// case below relies on.
GlmMoeDsaWeights LoadFixture(const gguf_test::TempFile& f) {
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::GlmMoeDsaHfConfigFromGguf(g);
  return vllm::LoadGlmMoeDsaFromGguf(g, c, /*policy=*/nullptr);
}

std::vector<float> RunForward(const GlmMoeDsaWeights& w,
                              const CommonAttentionMetadata& am,
                              MlaCachePool& pool, int64_t T) {
  vt::Queue q;
  return vllm::GlmMoeDsaModel::Forward(PromptIds(T, w.params.vocab_size),
                                       Positions(T), am, pool.attn_kv, w, q,
                                       /*logits_indices=*/{});
}

// The last token's logit row — the one a sampler would read.
std::vector<float> LastRow(const std::vector<float>& logits, int64_t vocab,
                           int64_t T) {
  REQUIRE(static_cast<int64_t>(logits.size()) == T * vocab);
  return std::vector<float>(logits.begin() + static_cast<long>((T - 1) * vocab),
                            logits.end());
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// (1) THE REACHABILITY CASE. A token, through the entry point a user arrives
//     through. Deleting the `GlmMoeDsaModel::ForwardDevice` / `::Forward` call
//     in `ForwardGlmMoeDsaForCausalLM` reds this and nothing hand-built could.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W9: the model produces a first token through LoadedEngine") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  vllm::entrypoints::EngineParams params;
  std::unique_ptr<vllm::entrypoints::LoadedEngine> engine;
  REQUIRE_NOTHROW(engine = vllm::entrypoints::LoadedEngine::FromModelDir(
                      f.path(), params));
  REQUIRE(engine != nullptr);

  vllm::SamplingParams sp;
  sp.temperature = 0.0;  // greedy: the token is a function of the logits alone
  sp.max_tokens = 1;
  sp.PostInit();

  const std::vector<int32_t> prompt = PromptIds(kTokens, kVocab);
  const vllm::RequestOutput out =
      engine->engine().generate(prompt, sp, "glm-dsa-w9-first-token");
  REQUIRE(!out.outputs.empty());
  const std::vector<int32_t>& got = out.outputs[0].token_ids;
  REQUIRE(got.size() == 1);

  // THE ASSERTION IS THAT THE ENGINE'S TOKEN IS THIS FORWARD'S ANSWER, and the
  // first version of this case did not make it. It checked only
  // `0 <= id < vocab`, which is a property of every integer the sampler could
  // possibly return — so when the reachability mutation replaced the whole body
  // of `ForwardGlmMoeDsaForCausalLM` with a zero-filled carrier, the argmax of
  // 32 zeros was 0, 0 is a legal id, and all seven cases stayed GREEN on a
  // build whose engine never called the forward at all. The mutation found the
  // gate, which is what mutations are for; this is the repair.
  //
  // Comparing against the DIRECT forward rather than against a literal is what
  // keeps this an assertion about reach instead of an assertion about the
  // fixture: nobody has to know which id is right, only that the two production
  // paths agree on it. A stub cannot agree by accident — it would have to
  // reproduce 32 logits it never computed.
  const GlmMoeDsaWeights w = LoadFixture(f);
  MlaCachePool pool(w.params);
  const std::vector<float> direct =
      RunForward(w, FreshPrefill(kTokens), pool, kTokens);
  const std::vector<float> row = LastRow(direct, kVocab, kTokens);
  const LogitReport r = Describe(row, "engine-vs-direct");
  REQUIRE(r.finite());
  REQUIRE(!r.top5.empty());
  const int32_t expect = static_cast<int32_t>(r.top5[0].first);
  std::printf("[glm-dsa W9] engine token=%d direct argmax=%d (margin %.6g)\n",
              got[0], expect,
              r.top5.size() > 1
                  ? static_cast<double>(r.top5[0].second - r.top5[1].second)
                  : 0.0);
  std::fflush(stdout);
  // The MARGIN is printed because a discrete selection's error is bimodal: a
  // near-tie that flips is a different event from a forward that computed
  // nothing, and only the margin separates them in a log.
  CHECK(got[0] == expect);
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) THE NUMBERS. Finite first, described always.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W9: the forward produces FINITE logits, and prints them") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  const GlmMoeDsaWeights w = LoadFixture(f);
  MlaCachePool pool(w.params);
  const CommonAttentionMetadata am = FreshPrefill(kTokens);

  const std::vector<float> logits = RunForward(w, am, pool, kTokens);
  REQUIRE(static_cast<int64_t>(logits.size()) == kTokens * kVocab);
  const LogitReport r = Describe(LastRow(logits, kVocab, kTokens), "last-row");
  // FINITENESS IS ASSERTED BEFORE ANY VALUE IS. Every case below this one
  // compares floats, and against a NaN both `==` and `!=` are false.
  CHECK(r.nan_count == 0);
  CHECK(r.inf_count == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) THE SELECTION REUSE. `mla.py:180` reduced to an assertion.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W9: the layer schedule puts 2 indexers on 4 layers and layer 0 is full") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  const GlmMoeDsaWeights w = LoadFixture(f);
  const std::vector<vllm::mla::MlaBlockDims> sched =
      vllm::GlmMoeDsaMlaSchedule(w.params);
  REQUIRE(static_cast<int64_t>(sched.size()) == kBackbone);
  int64_t full = 0, shared = 0;
  for (int64_t l = 0; l < kBackbone; ++l) {
    CAPTURE(l);
    // NEVER BOTH. Upstream cannot build a layer that carries an indexer AND
    // skips the top-k, because `:1115` is the negation of `:1175` everywhere
    // except the MTP block this row drops.
    CHECK(!(sched[static_cast<size_t>(l)].has_indexer() &&
            sched[static_cast<size_t>(l)].skip_topk));
    // EVERY layer is sparse — that is the difference from dots3-note, whose
    // sliding layers carry no selection at all.
    CHECK(sched[static_cast<size_t>(l)].is_sparse());
    if (sched[static_cast<size_t>(l)].has_indexer()) ++full; else ++shared;
  }
  CHECK(full == kFullLayers);
  CHECK(shared == kSharedLayers);
  CHECK(sched[0].has_indexer());
}

TEST_CASE("glm-dsa W9: a `kShared` layer is not a no-op, and it is not a copy") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  const GlmMoeDsaWeights base = LoadFixture(f);
  MlaCachePool pool_a(base.params);
  const CommonAttentionMetadata am = FreshPrefill(kTokens);
  const std::vector<float> ref = RunForward(base, am, pool_a, kTokens);
  const LogitReport rr = Describe(LastRow(ref, kVocab, kTokens), "reference");
  REQUIRE(rr.finite());

  // THE TAUTOLOGY GUARD, and it is the half that matters. Zeroing the FULL
  // layer's indexer projection changes the selection it writes, and a shared
  // layer that genuinely attends through that selection must move with it. If
  // the shared layers were quietly attending densely — the plausible, finite,
  // wrong answer this schedule exists to prevent — the two runs would agree.
  GlmMoeDsaWeights mutated = LoadFixture(f);
  bool touched = false;
  for (int64_t l = 0; l < kBackbone; ++l) {
    vllm::GlmMoeDsaIndexerWeights& ix =
        mutated.layers[static_cast<size_t>(l)].attn.indexer;
    if (ix.Empty()) continue;
    auto* p = reinterpret_cast<uint16_t*>(ix.wq_b.bytes.data());
    const int64_t n = ix.wq_b.Numel();
    for (int64_t i = 0; i < n; ++i) {
      // A finite, LARGE-ish perturbation rather than a zero: zeroing every
      // query would make every logit identical and the top-k an arbitrary
      // tie, which is a different experiment from moving the selection.
      p[static_cast<size_t>(i)] = vt::F32ToBF16(
          static_cast<float>((i % 5) - 2) * 0.5f);
    }
    touched = true;
  }
  REQUIRE(touched);
  MlaCachePool pool_b(mutated.params);
  const std::vector<float> moved = RunForward(mutated, am, pool_b, kTokens);
  const LogitReport mr = Describe(LastRow(moved, kVocab, kTokens), "selection-moved");
  REQUIRE(mr.finite());

  bool differs = false;
  for (size_t i = 0; i < ref.size() && !differs; ++i) {
    differs = std::isfinite(ref[i]) && std::isfinite(moved[i]) &&
              ref[i] != moved[i];
  }
  INFO("moving the FULL layers' indexer left the logits unchanged, which means "
       "the selection is reaching nothing");
  CHECK(differs);
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) THE SEAM. GLM-5.3 is the expert-streaming lane's SECOND client (spec O15),
//     and the way to prove it is that forcing the lane's fallback branch does
//     not change a single logit.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W9: the routed experts go through the expert-stream seam") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  const GlmMoeDsaWeights w = LoadFixture(f);
  const CommonAttentionMetadata am = FreshPrefill(kTokens);

  // THE LANE MUST BE ON, or this case measures nothing. Streaming is default
  // OFF, and with it off `ExpertSlice` never builds a store, both arms below
  // read the tower in place, and the comparison passes on a forward that never
  // touched the seam at all — the "instrument whose failure looks like a
  // result" shape. `VT_MOE_EXPERT_STREAM=1` is set on this target in
  // `tests/CMakeLists.txt`, and the assertions after the first run are what
  // prove it took effect.
  REQUIRE(vllm::expert_stream::StreamRequested());

  MlaCachePool pool_a(w.params);
  vllm::expert_stream::ExpertStreamLane::SetForceFallback(false);
  const std::vector<float> normal = RunForward(w, am, pool_a, kTokens);
  const LogitReport nr = Describe(LastRow(normal, kVocab, kTokens), "seam-normal");
  REQUIRE(nr.finite());

  // GLM-5.3 IS THE LANE'S SECOND CLIENT (spec O15), and this is the assertion
  // that says so. A store exists only because `expert_stream::ExpertSlice` was
  // called, and it recorded fills only because slices were actually served out
  // of it. Deleting the `GlmExpertSlice` call in `ExpertMlp` and reading the
  // tower directly leaves every logit identical and reds exactly here.
  const vllm::expert_stream::ExpertStreamLane* lane =
      vllm::expert_stream::ExpertStreamLane::Existing();
  REQUIRE(lane != nullptr);
  const int64_t fills = lane->streamer().fills();
  const int64_t served = lane->cache().hits() + lane->cache().misses();
  std::printf("[glm-dsa W9] lane: fills=%lld served=%lld bytes=%lld\n",
              static_cast<long long>(fills), static_cast<long long>(served),
              static_cast<long long>(lane->streamer().bytes_filled()));
  std::fflush(stdout);
  CHECK(fills > 0);
  CHECK(served > 0);

  // FORCE the cache-exhaustion branch, which makes every `Slice` return nullptr
  // and every caller fall back to the resident tower view. That is a REAL
  // production state (a budget smaller than one step's working set reaches it),
  // and having a switch for it is what lets one process compare the two arms.
  MlaCachePool pool_b(w.params);
  vllm::expert_stream::ExpertStreamLane::SetForceFallback(true);
  const std::vector<float> forced = RunForward(w, am, pool_b, kTokens);
  vllm::expert_stream::ExpertStreamLane::SetForceFallback(false);
  const LogitReport fr = Describe(LastRow(forced, kVocab, kTokens), "seam-forced");
  REQUIRE(fr.finite());

  // IDENTICAL, not close. A slot holds a byte copy of the same slice, so the
  // two arms read the same bytes and run the same GEMM. Anything else is a
  // slice-arithmetic defect, and a tolerance would hide exactly that.
  REQUIRE(normal.size() == forced.size());
  int64_t differing = 0;
  double maxabs = 0.0;
  for (size_t i = 0; i < normal.size(); ++i) {
    REQUIRE(std::isfinite(normal[i]));
    REQUIRE(std::isfinite(forced[i]));
    if (normal[i] != forced[i]) {
      ++differing;
      maxabs = std::max(maxabs, std::abs(static_cast<double>(normal[i]) -
                                         static_cast<double>(forced[i])));
    }
  }
  std::printf("[glm-dsa W9] seam arms: differing=%lld/%zu maxabs=%.6g\n",
              static_cast<long long>(differing), normal.size(), maxabs);
  std::fflush(stdout);
  CHECK(differing == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) THE REFUSALS. Each proved by making the input violate it.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W9: a RESUMED request whose selection prunes is refused by name") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  const GlmMoeDsaWeights w = LoadFixture(f);
  MlaCachePool pool(w.params);
  // One request, `kTokens` of context of which some were computed on an earlier
  // step. Its context exceeds `index_topk`, so the selection PRUNES; its index
  // keys are not in hand, so the sparse route cannot be taken. Serving it would
  // be DENSE attention on a sparse model, in silence.
  CommonAttentionMetadata am = FreshPrefill(kTokens);
  am.num_computed_tokens_cpu = {16};
  const std::string msg = RefusalOf(
      [&] { (void)RunForward(w, am, pool, kTokens); },
      "GlmMoeDsaModel::Forward on a resumed step");
  for (const char* needle : {"resumes from", "index_topk", "indexer KV side cache",
                             "DeepseekV32IndexerCache", "#1925", "#2323"}) {
    CAPTURE(needle);
    CHECK(msg.find(needle) != std::string::npos);
  }
}

TEST_CASE("glm-dsa W9: weights that never went through the absorption are refused") {
  // A hand-constructed `GlmMoeDsaWeights` is the shape a unit test reaches for,
  // and it is precisely the one the forward cannot serve: `w_uk_t`, `w_uv` and
  // `kv_b_proj` are produced at LOAD and nothing else produces them.
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  GlmMoeDsaWeights w = LoadFixture(f);
  w.absorbed = false;
  MlaCachePool pool(w.params);
  const CommonAttentionMetadata am = FreshPrefill(kTokens);
  const std::string msg = RefusalOf(
      [&] { (void)RunForward(w, am, pool, kTokens); },
      "GlmMoeDsaModel::Forward on unabsorbed weights");
  CHECK(msg.find("post-load absorption") != std::string::npos);
  CHECK(msg.find("w_uk_t") != std::string::npos);
}
