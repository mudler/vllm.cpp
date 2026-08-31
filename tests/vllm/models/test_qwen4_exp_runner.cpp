// Qwen4-Exp W5L — `GPUModelRunner` and `LoadedEngine`, driven end to end.
//
// Row MODEL-MM-QWEN4-EXP, issue #2031, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHAT THIS FILE ADDS THAT `test_qwen4_exp_layer_loop.cpp` CANNOT ─────────
//
// W5k runs a prefill and a decode over persistent caches and samples a token on
// each, and it says outright what it did not prove: "`GPUModelRunner` was not
// driven end to end — the two steps are assembled the way the runner assembles
// one, not BY it." Every `ModelForwardInput` in that file is built by hand, so
// it gates the FORWARD and cannot see:
//
//   * whether `initialize_kv_cache` allocates the three groups this model
//     publishes, or how many buffers each contributes;
//   * whether the by-name index the runner BUILDS carries the five names the
//     hook RESOLVES, under the group ids and payload kinds it expects;
//   * whether `gather_group_block_tables` runs for all three groups, or whether
//     group 2 gets group 0's rows;
//   * whether the recurrent slot the PLE block writes is the slot the runner
//     assigned to this request;
//   * whether `ModelFactory::consumes_multi_kv` lets the topology past
//     `ModelRegistry::Forward` on a step the ENGINE produced.
//
// Nothing in this file constructs a `ModelForwardInput`, a `CommonAttentionMetadata`
// or a `GDNAttentionMetadata`. The runner builds all three from a `SchedulerOutput`,
// which is what the engine hands it.
//
// ─── WHAT IT CAN AND CANNOT GATE ────────────────────────────────────────────
//
// It CANNOT gate cache CONTENT. W5j measured that on this fixture: across two
// prompts 0 of 128 indexer words and 0 of 192 paged K/V words moved while the
// logits moved 31.84, because the layer-3 activations sit near 2^18 where one
// bf16 ULP is about 1024 and the store saturates. A rescaled fixture is owed and
// nothing here asserts a cache value.
//
// What it gates instead is the same observable W5k found and could only read out
// of a hand-built buffer: the PLE layer's n-gram history is int64 TOKEN IDS, it
// cannot saturate, and upstream rolls it every step. Here it is read out of the
// buffer the RUNNER allocated, at the slot the RUNNER assigned, after a step the
// ENGINE scheduled — and the decode-step read's writer is a prior
// `execute_model` call.
#include <doctest/doctest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

#include "support/qwen4_exp_gguf_fixture.h"

namespace {

using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vllm::v1::CachedRequestData;
using vllm::v1::GPUModelRunner;
using vllm::v1::KVCacheConfig;
using vllm::v1::NewRequestData;
using vllm::v1::SchedulerOutput;

vt::Queue CpuQ() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// THE PAGE SIZE IS NOT FREE. `CpuAttentionBackend::get_supported_kernel_block_sizes()`
// returns {16} and `AttentionBackend::supports_block_size` accepts a multiple of
// a declared size, so 16 is the smallest the runner will build an attention
// backend for — measured: at 4 the constructor throws "No valid attention
// backend for device type 0 from {CPU_ATTN: [block_size not supported], ...}".
// It is also a multiple of this fixture's `indexer_compress_ratio` (4), which
// `RunQwen4ExpQsaBlockPaged` requires of the K/V page.
constexpr int kBlockSize = 16;
constexpr int kNumBlocks = 32;
constexpr int kMaxModelLen = 32;

// The fixture's schedule: decoder layers 0, 1, 2 are linear_attention and layer
// 3 is qwen_sparse_attention, and the PLE block sits on layer 1 — which is
// RECURRENT RANK 1, the index the runner's `gdn_state()` is keyed by. The two
// coincide here and the case says which one it means.
constexpr size_t kPleRecurrentRank = 1;
constexpr size_t kRecurrentLayers = 3;
constexpr int64_t kCtx = qwen4_exp_fixture::kNgramSize - 1;  // 2

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// THE BLOCK IDS ARE DELIBERATELY DISJOINT PER GROUP. `NewRequestData::block_ids`
// is parallel to the published group order, so giving group 0 and group 2
// different physical blocks is what makes "group 2 was gathered from group 0's
// rows" a FAILING assertion rather than an invisible one. Group 1 is the
// recurrent group and carries one state slot.
NewRequestData MakeReq(const std::string& id,
                       const std::vector<int32_t>& prompt) {
  NewRequestData nr;
  nr.req_id = id;
  nr.prompt_token_ids = prompt;
  nr.sampling_params = Greedy(8);
  nr.block_ids = {std::vector<int>{1, 2}, std::vector<int>{0},
                  std::vector<int>{3, 4}};
  nr.num_computed_tokens = 0;
  nr.prefill_token_ids = prompt;
  return nr;
}

SchedulerOutput PrefillStep(std::vector<NewRequestData> reqs) {
  SchedulerOutput so;
  so.scheduled_cached_reqs = CachedRequestData::make_empty();
  int total = 0;
  for (const NewRequestData& r : reqs) {
    REQUIRE(r.prefill_token_ids.has_value());
    const int n = static_cast<int>(r.prefill_token_ids->size());
    so.num_scheduled_tokens[r.req_id] = n;
    total += n;
  }
  so.scheduled_new_reqs = std::move(reqs);
  so.total_num_scheduled_tokens = total;
  return so;
}

SchedulerOutput DecodeStep(const std::string& id, int computed, int produced) {
  SchedulerOutput so;
  CachedRequestData cached;
  cached.req_ids = {id};
  cached.num_computed_tokens = {computed};
  cached.num_output_tokens = {produced};
  cached.new_block_ids.emplace_back(std::nullopt);
  so.scheduled_cached_reqs = std::move(cached);
  so.num_scheduled_tokens[id] = 1;
  so.total_num_scheduled_tokens = 1;
  return so;
}

// The PLE layer's n-gram history, read out of the RUNNER's own recurrent state
// at the slot the runner assigned. It is `states[3]` of the published order
// [gdn_conv, temporal, ple_conv, ngram] and it is i64 by construction
// (`MakeQwen4ExpKVCache`), which the read ASSERTS rather than assumes: a float
// state read through an int64_t pointer would produce plausible garbage.
std::vector<int64_t> HistoryOf(const GPUModelRunner& runner, size_t rank,
                               int64_t slot) {
  REQUIRE(runner.gdn_state().size() > rank);
  const std::vector<vt::Tensor>& states = runner.gdn_state()[rank].states;
  REQUIRE(states.size() == 4);
  const vt::Tensor& h = states[3];
  REQUIRE(h.dtype == vt::DType::kI64);
  REQUIRE(h.rank == 2);
  REQUIRE(h.shape[1] == kCtx);
  REQUIRE(slot < h.shape[0]);
  const auto* base = static_cast<const int64_t*>(h.data) +
                     static_cast<size_t>(slot) * static_cast<size_t>(kCtx);
  return std::vector<int64_t>(base, base + kCtx);
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. THE RUNNER ALLOCATES, GATHERS AND PUBLISHES WHAT THIS MODEL DECLARED.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE(
    "qwen4_exp runner: initialize_kv_cache allocates the three published groups "
    "and names every cache the forward resolves") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)
  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const vllm::ModelRegistration& reg = model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(config, kBlockSize, kNumBlocks);
  REQUIRE(kv.kv_cache_groups.size() == 3);

  GPUModelRunner runner(config, *model, kv, CpuQ(), /*max_num_reqs=*/2,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  // TWO paged buffers for ONE qwen_sparse_attention layer, because this model
  // publishes TWO attention groups over the same layer — group 0's K/V and
  // group 2's indexer side cache. A runner that allocated per LAYER rather than
  // per (group x layer) would report one, and the hook's by-name resolution
  // would then answer group 0's pages for the indexer.
  CHECK(runner.attn_kv().size() == 2);
  REQUIRE(runner.gdn_state().size() == kRecurrentLayers);
  for (size_t i = 0; i < kRecurrentLayers; ++i) {
    INFO("recurrent rank ", i);
    // FOUR states: the GDN conv and temporal pair plus the PLE conv ring and the
    // n-gram history. `RunQwen4ExpPleBlock` ROUTES on `states.size() >= 4`, so a
    // three-state allocation is the case that would run the PLE layer on a
    // zeroed per-call scratch.
    CHECK(runner.gdn_state()[i].states.size() == 4);
  }

  // THE BY-NAME INDEX, tuple by tuple. These are the exact strings
  // `MakeQwen4ExpKVCache` publishes and `ForwardQwen4ExpForConditionalGeneration`
  // resolves; the two derive them from one file-local builder, and this case is
  // the third party that pins the RESULT.
  const vllm::MultiKvCacheIndex& mk = runner.multi_kv_index();
  REQUIRE(mk.layer_names != nullptr);
  REQUIRE(mk.group_ids != nullptr);
  REQUIRE(mk.payload_kinds != nullptr);
  REQUIRE(mk.payload_slots != nullptr);
  REQUIRE(mk.layer_names->size() == 5);
  struct Pub {
    const char* name;
    int32_t group;
    vllm::KvCachePayload kind;
    int32_t slot;
  };
  const Pub want[5] = {
      {"model.layers.3.self_attn.attn", 0, vllm::KvCachePayload::kPaged, 0},
      {"model.layers.0.linear_attn", 1, vllm::KvCachePayload::kRecurrent, 0},
      {"model.layers.1.linear_attn", 1, vllm::KvCachePayload::kRecurrent, 1},
      {"model.layers.2.linear_attn", 1, vllm::KvCachePayload::kRecurrent, 2},
      {"model.layers.3.self_attn.indexer.k_cache", 2,
       vllm::KvCachePayload::kPaged, 1},
  };
  for (size_t i = 0; i < 5; ++i) {
    INFO("published entry ", i);
    CHECK((*mk.layer_names)[i] == want[i].name);
    CHECK((*mk.group_ids)[i] == want[i].group);
    CHECK((*mk.payload_kinds)[i] == static_cast<uint8_t>(want[i].kind));
    CHECK((*mk.payload_slots)[i] == want[i].slot);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. A PREFILL AND A DECODE, DRIVEN BY THE RUNNER.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE(
    "qwen4_exp runner: execute_model prefills then decodes over the caches it "
    "allocated") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)
  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  KVCacheConfig kv =
      model->registration().factory->make_kv_cache(config, kBlockSize, kNumBlocks);
  GPUModelRunner runner(config, *model, kv, CpuQ(), /*max_num_reqs=*/2,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  // The prompt carries an EOS in the INTERIOR — a segment boundary in the hashed
  // n-gram construction — and its last two ids are DISTINCT, which is what makes
  // the roll below observable at all. Every id is in range for the 16-entry
  // vocabulary this fixture declares.
  const std::vector<int32_t> prompt{5, 9, 13, static_cast<int32_t>(kEosTokenId),
                                    7, 2};
  for (int32_t v : prompt) REQUIRE(v < static_cast<int32_t>(kVocab));
  REQUIRE(prompt[prompt.size() - 2] != prompt.back());

  CHECK_FALSE(runner.execute_model(PrefillStep({MakeReq("r0", prompt)})).has_value());
  vllm::v1::ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  REQUIRE(m1.sampled_token_ids.size() == 1);
  REQUIRE(m1.sampled_token_ids[0].size() == 1);
  const int32_t tok1 = m1.sampled_token_ids[0][0];
  CHECK(tok1 >= 0);
  CHECK(tok1 < static_cast<int32_t>(kVocab));
  MESSAGE("runner PREFILL (T=" << prompt.size() << ") sampled " << tok1);

  // ─── THE GATHER RAN FOR ALL THREE GROUPS, AND GROUP 2 IS NOT GROUP 0 ───────
  //
  // A STRUCTURAL ROW-SET assertion and not a value one, which is the shape W5i
  // measured this fixture needs: 9 of 23 rows in the wrong physical page left a
  // paged-vs-contiguous diff at 0 of 1472 words, because the store and the read
  // share one translation. The rows are what a wrong translation changes.
  const vllm::MultiKvCacheIndex& mk = runner.multi_kv_index();
  REQUIRE(mk.group_block_tables != nullptr);
  REQUIRE(mk.group_block_table_cols != nullptr);
  REQUIRE(mk.group_block_tables->size() == 3);
  const int cols = (*mk.group_block_table_cols)[0];
  REQUIRE(cols >= 2);
  const std::vector<int32_t>& t0 = (*mk.group_block_tables)[0];
  const std::vector<int32_t>& t2 = (*mk.group_block_tables)[2];
  REQUIRE(static_cast<int>(t0.size()) >= 2);
  REQUIRE(static_cast<int>(t2.size()) >= 2);
  // The request's own block_ids, per group, in the order it declared them.
  CHECK(t0[0] == 1);
  CHECK(t0[1] == 2);
  CHECK(t2[0] == 3);
  CHECK(t2[1] == 4);
  // The two are DIFFERENT tables. A gather that answered group 0 for every group
  // — the defect the by-name channel exists to prevent — makes this equal.
  CHECK(t0 != t2);
  // And neither is the identity, so a hook that ignored the table entirely and
  // indexed pages by logical position would still read the wrong page.
  CHECK_FALSE(t0[0] == 0);
  CHECK_FALSE(t2[0] == 0);
  // Group 1 is the RECURRENT group and carries a state slot, not a page.
  REQUIRE_FALSE((*mk.group_block_tables)[1].empty());
  const int64_t slot = (*mk.group_block_tables)[1][0];

  // ─── THE PREFILL LEFT THE PROMPT'S LAST TWO IDS IN THE ENGINE'S OWN STATE ──
  //
  // Compared to the INPUT IDS, not to a value read back before the run, so a
  // cache nothing wrote fails. Upstream keeps `ngram_size - 1` raw ids and this
  // prompt's are [7, 2] (measured on transformers 5.16.0; see the W5k record).
  const std::vector<int64_t> h1 = HistoryOf(runner, kPleRecurrentRank, slot);
  for (int64_t i = 0; i < kCtx; ++i) {
    INFO("history slot ", i);
    CHECK(h1[static_cast<size_t>(i)] ==
          static_cast<int64_t>(prompt[prompt.size() - static_cast<size_t>(kCtx - i)]));
  }

  // NOTHING WROTE THE OTHER LINEAR LAYERS' PLE SLOTS. The recurrent group is
  // uniform — all three linear layers get the same state set — so a hook that
  // indexed the PLE cache by DECODER layer index rather than by rank among the
  // linear layers would write rank 3's, or rank 2's.
  for (size_t r = 0; r < kRecurrentLayers; ++r) {
    if (r == kPleRecurrentRank) continue;
    INFO("recurrent rank ", r, " must own no PLE history");
    CHECK(HistoryOf(runner, r, slot) ==
          std::vector<int64_t>(static_cast<size_t>(kCtx), 0));
  }

  // ─── THE DECODE. The runner builds this step from a CachedRequestData. ─────
  CHECK_FALSE(runner.execute_model(DecodeStep("r0", static_cast<int>(prompt.size()),
                                              /*produced=*/1))
                  .has_value());
  vllm::v1::ModelRunnerOutput m2 = runner.sample_tokens(std::nullopt);
  REQUIRE(m2.sampled_token_ids.size() == 1);
  REQUIRE(m2.sampled_token_ids[0].size() == 1);
  const int32_t tok2 = m2.sampled_token_ids[0][0];
  CHECK(tok2 >= 0);
  CHECK(tok2 < static_cast<int32_t>(kVocab));
  MESSAGE("runner DECODE (past_len=" << prompt.size() << ") sampled " << tok2);

  // THE FIFO ROLLED, and its WRITER IS A PRIOR `execute_model` CALL. The last
  // slot holds the token step 1 sampled; every earlier slot holds what the slot
  // to its right held before this step.
  const std::vector<int64_t> h2 = HistoryOf(runner, kPleRecurrentRank, slot);
  CHECK(h2[static_cast<size_t>(kCtx - 1)] == static_cast<int64_t>(tok1));
  for (int64_t i = 0; i + 1 < kCtx; ++i) {
    INFO("rolled slot ", i);
    CHECK(h2[static_cast<size_t>(i)] == h1[static_cast<size_t>(i + 1)]);
  }
  // And it is NOT what a per-call scratch would leave: that arm re-seeds the
  // history with `eos_token_id` on every step, which is a fluent wrong answer
  // with no error anywhere.
  CHECK(h2 != std::vector<int64_t>{static_cast<int64_t>(kEosTokenId),
                                   static_cast<int64_t>(tok1)});
  // The state genuinely MOVED between the two steps. Without this the two
  // assertions above pass on a prompt whose last two ids happen to equal
  // [prompt.back(), tok1], which is why the prompt asserts they are distinct.
  CHECK(h1 != h2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. A BATCHED STEP IS REFUSED, BY THIS HOOK, ON ITS OWN PREDICATE.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("qwen4_exp runner: a two-request step is refused by name") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)
  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  KVCacheConfig kv =
      model->registration().factory->make_kv_cache(config, kBlockSize, kNumBlocks);
  GPUModelRunner runner(config, *model, kv, CpuQ(), /*max_num_reqs=*/2,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  NewRequestData a = MakeReq("r0", {5, 9, 13});
  NewRequestData b = MakeReq("r1", {2, 7, 11});
  b.block_ids = {std::vector<int>{5, 6}, std::vector<int>{1},
                 std::vector<int>{7, 8}};
  std::string msg = "<the batched step RETURNED; this refusal is gone>";
  try {
    (void)runner.execute_model(PrefillStep({a, b}));
  } catch (const std::exception& e) {
    msg = e.what();
  }
  INFO("the batched-step refusal said: ", msg);
  // TWO-SIDED. The identifying bytes must be present AND a different refusal's
  // must be absent, because appending to a message leaves the old one a
  // substring of the new.
  CHECK(msg.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
  CHECK(msg.find("serves ONE sequence per call") != std::string::npos);
  CHECK(msg.find("the step carries 2") != std::string::npos);
  // NOT the engine's multi-cache guard, and NOT the PLE layout refusal: the step
  // got past both and stopped on the batch.
  CHECK(msg.find("does not consume a cache set keyed by layer name") ==
        std::string::npos);
  CHECK(msg.find("qwen4_exp ple layout") == std::string::npos);
  // NOT the continuing-step refusal either: this is a prefill.
  CHECK(msg.find("continues a sequence at past_len") == std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. THE ENGINE CLAMP, AND WHY A REFUSAL ALONE WAS NOT ENOUGH.
// ═══════════════════════════════════════════════════════════════════════════
//
// The refusal in case 3 is thrown from inside the EngineCore busy loop, and that
// loop treats a throw as FATAL rather than as one failed request. MEASURED on
// `examples/server` at `--max-num-seqs 4`: three overlapping `/v1/completions`
// calls each returned a 500 carrying this hook's own message, and the engine
// never served again. The DEFAULT `max_num_seqs` is 128, so that was the
// out-of-the-box behaviour of a server pointed at this architecture.
//
// So the model declares the limit and `LoadedEngine::ResolveMaxNumSeqs` clamps.
// The resolver is static and pure, which is what lets the polarity be gated
// without a disk load — both directions, because a clamp that fires for every
// model is not a clamp.

TEST_CASE("qwen4_exp: the engine clamps concurrency to one for this forward") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)
  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  // THE MODEL DECLARES IT. Without this the clamp below is a policy nothing
  // turns on, which is the "capability with no arm a test could drive" shape.
  CHECK(model->registration().factory->serves_one_sequence_per_step);

  KVCacheConfig kv =
      model->registration().factory->make_kv_cache(config, kBlockSize, kNumBlocks);
  EngineParams params;
  params.max_num_seqs = 8;
  params.block_size = kBlockSize;
  params.num_blocks = kNumBlocks;
  params.max_model_len = kMaxModelLen;

  // The SAME params and the SAME pool, differing only in the declaration. A
  // one-sided check cannot tell a clamp from a constant.
  CHECK(LoadedEngine::ResolveMaxNumSeqs(params, kv, /*serves_one=*/true) == 1);
  CHECK(LoadedEngine::ResolveMaxNumSeqs(params, kv, /*serves_one=*/false) > 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. A REQUEST GENERATES THROUGH THE STACK `examples/server` BUILDS.
// ═══════════════════════════════════════════════════════════════════════════
//
// `LoadedEngine::FromModelDir` is the function `vllm_server_main` calls, and it
// accepts a `.gguf` path. This case points it at the same artifact the manual
// server run used — the fixture with its `tokenizer.ggml.*` kvs — and generates.
// It is the reachability proof for BOTH the clamp (the engine resolves it here,
// not in a test's own call) and the whole three-group forward.
TEST_CASE("qwen4_exp: LoadedEngine generates over the GGUF the server loads") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)
  FixtureOpts o;
  o.with_tokenizer = true;
  const gguf_test::TempFile f(BuildFixture(o));

  EngineParams params;
  params.block_size = kBlockSize;
  params.num_blocks = kNumBlocks;
  params.max_model_len = kMaxModelLen;
  // ASKED FOR EIGHT. What the engine resolves is checked below, so this number
  // is the thing the clamp has to override rather than a value that agrees with
  // it by accident.
  params.max_num_seqs = 8;

  std::unique_ptr<LoadedEngine> eng;
  REQUIRE_NOTHROW(eng = LoadedEngine::FromModelDir(f.path(), params));
  REQUIRE(eng != nullptr);
  // THE CLAMP IS REACHED FROM THE PRODUCTION CONSTRUCTOR.
  CHECK(eng->max_num_seqs() == 1);

  // Every character of the prompt is one token of this fixture's 16-entry
  // byte-level vocabulary ('a'..'p'), so the prompt length in tokens is known.
  constexpr int kNew = 5;
  const vllm::RequestOutput out =
      eng->engine().generate("abcdef", Greedy(kNew), "req");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  CHECK(static_cast<int>(out.outputs[0].token_ids.size()) == kNew);
  for (int32_t t : out.outputs[0].token_ids) {
    CHECK(t >= 0);
    CHECK(t < static_cast<int32_t>(kVocab));
  }
  CHECK(out.prompt_token_ids.size() == 6);
  MESSAGE("engine generated " << out.outputs[0].token_ids.size()
                              << " tokens, text \"" << out.outputs[0].text
                              << "\"");
}
