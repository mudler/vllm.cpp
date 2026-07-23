// Ported from: tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
//               (the scheduler-participation cases) @ e24d1b24, PLUS the
//               end-to-end RESTART-HIT demonstration the task requires: a
//               repeated/restarted-prefix workload where a disk-tier HIT shortcuts
//               prefill through the REAL scheduler, with the hit counted and the
//               promoted bytes proven byte-identical to a cold store.
//
// KV-OFFLOAD W4, the connector/scheduler HALF. Two kinds of gate:
//   1. Plumbing (mock connector): the scheduler layers external matched tokens
//      onto get_computed_blocks and shortcuts prefill; the nullopt THIRD state
//      defers a request and re-asks next step; the hooks fire.
//   2. End-to-end (real OffloadingConnector + TieringOffloadingManager): a
//      restarted-process prefix HIT loads from disk and shortcuts prefill;
//      records the hit rate and prefill-tokens-saved, and proves byte identity.
#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/kv_offload/cache_identity.h"
#include "vllm/v1/kv_offload/cpu_manager.h"
#include "vllm/v1/kv_offload/fs_tier.h"
#include "vllm/v1/kv_offload/kv_connector.h"
#include "vllm/v1/kv_offload/tiering_manager.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using namespace vllm::v1;              // NOLINT(build/namespaces)
using namespace vllm::v1::kv_offload;  // NOLINT(build/namespaces)
using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vt::DType;

namespace {

constexpr int kBlockSize = 16;
constexpr int64_t kPage = 512;

std::unique_ptr<Scheduler> CreateScheduler(int num_blocks = 10000) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(kBlockSize, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<Scheduler>(cfg, kv_cfg, kBlockSize,
                                     /*enable_caching=*/true);
}

std::unique_ptr<Request> MakeRequest(const std::string& id, int num_tokens,
                                     int seed) {
  static bool inited = false;
  if (!inited) {
    init_none_hash(sha256_cbor);
    inited = true;
  }
  auto hasher = get_request_block_hasher(kBlockSize, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  // Distinct token stream per seed so prompts do not collide unless intended.
  std::vector<int32_t> prompt(num_tokens);
  for (int i = 0; i < num_tokens; ++i) prompt[i] = seed * 100000 + i;
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   hasher);
}

CacheIdentity BaseIdentity() {
  CacheIdentity id;
  id.model_name = "Qwen/Qwen3-4B";
  id.model_type = "qwen3";
  id.architectures = {"Qwen3ForCausalLM"};
  id.hf_config_digest = "deadbeefcafe";
  id.weight_quantization = "none";
  id.checkpoint_fingerprint = "sha256:aaaa";
  id.num_hidden_layers = 36;
  id.num_kv_heads = 8;
  id.head_size = 128;
  id.head_size_v = 128;
  id.sliding_window = -1;
  id.rope_config = "theta=1000000,dim=128,scaling=none";
  id.max_position_embeddings = 40960;
  id.kv_cache_spec_kind = "full_attention";
  id.page_size_bytes = kPage;
  id.block_size = kBlockSize;
  id.hash_block_size = kBlockSize;
  id.kv_dtype = "bf16";
  id.none_hash_hex = "00";
  return id;
}

std::vector<uint8_t> Payload(const OffloadKey& key) {
  std::vector<uint8_t> p(kPage);
  for (int64_t i = 0; i < kPage; ++i) {
    p[static_cast<size_t>(i)] =
        static_cast<uint8_t>(static_cast<unsigned char>(key[0]) +
                             static_cast<unsigned char>(key[1]) * 3 + i * 7);
  }
  return p;
}

class TempDir {
 public:
  explicit TempDir(const std::string& tag) {
    static int c = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("vllmcpp_kvconn_" + tag + "_" + std::to_string(::getpid()) + "_" +
             std::to_string(c++));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  std::string str() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

// A programmable connector for the scheduler-plumbing gate: it returns a queued
// sequence of MatchResults and records every hook call.
class MockConnector : public KVConnectorScheduler {
 public:
  std::vector<MatchResult> answers;  // consumed front-to-back
  size_t answer_idx = 0;
  int alloc_calls = 0;
  int last_alloc_external = -1;
  int build_meta_calls = 0;
  int on_schedule_end_calls = 0;
  int finished_calls = 0;

  MatchResult get_num_new_matched_tokens(const Request&, int) override {
    if (answer_idx < answers.size()) return answers[answer_idx++];
    return MatchResult{0, false};
  }
  void update_state_after_alloc(const Request&,
                                const std::vector<std::vector<int>>&,
                                int num_external_tokens) override {
    alloc_calls += 1;
    last_alloc_external = num_external_tokens;
  }
  std::vector<ConnectorLoadJob> build_connector_meta() override {
    build_meta_calls += 1;
    return {};
  }
  void request_finished(const Request&) override { finished_calls += 1; }
  void on_schedule_end() override { on_schedule_end_calls += 1; }
};

}  // namespace

// ---------------------------------------------------------------------------
// A null connector is ZERO behaviour change (offloading is opt-in, inert off).
// ---------------------------------------------------------------------------
TEST_CASE("Connector: a null connector leaves scheduling unchanged") {
  auto sched = CreateScheduler();
  sched->add_request(MakeRequest("a", 48, /*seed=*/1));
  auto out = sched->schedule();
  REQUIRE(out.num_scheduled_tokens.count("a") == 1);
  CHECK(out.num_scheduled_tokens.at("a") == 48);  // full prompt, no shortcut
}

// ---------------------------------------------------------------------------
// An external match shortcuts prefill; the alloc hook receives the ext count.
// ---------------------------------------------------------------------------
TEST_CASE("Connector: an external match shortcuts prefill by exactly ext") {
  auto sched = CreateScheduler();
  MockConnector conn;
  conn.answers = {MatchResult{16, false}};  // 1 block of external KV
  sched->set_kv_connector(&conn);
  sched->add_request(MakeRequest("a", 48, /*seed=*/2));

  auto out = sched->schedule();
  REQUIRE(out.num_scheduled_tokens.count("a") == 1);
  CHECK(out.num_scheduled_tokens.at("a") == 48 - 16);  // prefill shortened
  CHECK(conn.alloc_calls == 1);
  CHECK(conn.last_alloc_external == 16);
  CHECK(conn.build_meta_calls == 1);        // meta built (state reset)
  CHECK(conn.on_schedule_end_calls == 1);   // promotions flushed
}

// ---------------------------------------------------------------------------
// The nullopt THIRD state defers the request and re-asks next step.
// ---------------------------------------------------------------------------
TEST_CASE("Connector: the nullopt third state defers, then schedules next step") {
  auto sched = CreateScheduler();
  MockConnector conn;
  // Step 1: "not ready, re-ask". Step 2: 16 external tokens.
  conn.answers = {MatchResult{std::nullopt, false}, MatchResult{16, false}};
  sched->set_kv_connector(&conn);
  sched->add_request(MakeRequest("a", 48, /*seed=*/3));

  auto out1 = sched->schedule();
  // Deferred: nothing scheduled, request still waiting (re-queued to the front).
  CHECK(out1.num_scheduled_tokens.count("a") == 0);
  CHECK(sched->waiting->size() == 1);
  CHECK(conn.alloc_calls == 0);

  auto out2 = sched->schedule();
  REQUIRE(out2.num_scheduled_tokens.count("a") == 1);
  CHECK(out2.num_scheduled_tokens.at("a") == 48 - 16);
  CHECK(conn.alloc_calls == 1);
}

// ---------------------------------------------------------------------------
// END-TO-END: a restarted-process prefix HIT loads from disk and shortcuts
// prefill through the real scheduler + OffloadingConnector + tiering manager.
// Records the hit rate and prefill-tokens-saved; proves byte identity.
// ---------------------------------------------------------------------------
TEST_CASE("Connector e2e: a restarted-prefix disk HIT shortcuts prefill") {
  TempDir dir("restart");
  const CacheIdentity id = BaseIdentity();

  // Build the request first so its block_hashes are fixed; a 48-token prompt is
  // 3 full 16-token blocks.
  const int kPromptTokens = 48;
  const int kPromptBlocks = kPromptTokens / kBlockSize;  // 3
  auto req_probe = MakeRequest("warm", kPromptTokens, /*seed=*/7);
  REQUIRE(static_cast<int>(req_probe->block_hashes.size()) == kPromptBlocks);

  // "Process 1": populate the DISK tier with this prefix's blocks (the worker's
  // GPU->CPU->disk store path), then evict the CPU tier so the blocks live ONLY
  // on disk — a genuine restart.
  auto view = std::make_unique<HeapPrimaryByteView>(/*num_blocks=*/8, kPage);
  auto primary = std::make_unique<CPUOffloadingManager>(8, "lru");
  FileSystemTierOptions opts;
  opts.root_dir = dir.str();
  opts.identity = id;
  auto secondary = std::make_unique<FileSystemTier>(opts);
  auto mgr = std::make_unique<TieringOffloadingManager>(
      std::move(primary), std::move(secondary), *view);

  const ReqContext pctx{"populate"};
  mgr->on_new_request(pctx);
  std::vector<std::vector<uint8_t>> stored(kPromptBlocks);
  for (int j = 0; j < kPromptBlocks; ++j) {
    const OffloadKey key = make_offload_key(req_probe->block_hashes[j], 0);
    stored[j] = Payload(key);
    auto out = mgr->prepare_store({key}, pctx);
    REQUIRE(out.has_value());
    auto* spec = dynamic_cast<CPULoadStoreSpec*>(out->store_spec.get());
    REQUIRE(spec != nullptr);
    std::memcpy(view->slot(spec->block_ids[0]), stored[j].data(), kPage);
    mgr->complete_store({key}, pctx, true);
    mgr->drain_jobs();
  }
  mgr->reset_cache();  // CPU cleared; disk SURVIVES (restart)
  // Scribble the byte arena so any promotion must read from disk.
  for (int64_t b = 0; b < 8; ++b) std::memset(view->slot(b), 0xEE, kPage);

  // ---- Baseline: offloading OFF. Fresh scheduler, same prompt. ----
  int baseline_prefill = 0;
  {
    auto sched0 = CreateScheduler();
    sched0->add_request(MakeRequest("warm", kPromptTokens, /*seed=*/7));
    auto out = sched0->schedule();
    baseline_prefill = out.num_scheduled_tokens.at("warm");
  }
  CHECK(baseline_prefill == kPromptTokens);  // full prefill, nothing cached

  // ---- Treatment: offloading ON via the real connector. ----
  OffloadingConnectorConfig ccfg;
  ccfg.group_idx = 0;
  ccfg.offload_block_tokens = kBlockSize;
  OffloadingConnector conn(*mgr, ccfg);

  auto sched = CreateScheduler();
  sched->set_kv_connector(&conn);
  auto warm = MakeRequest("warm", kPromptTokens, /*seed=*/7);
  conn.on_new_request(*warm);
  sched->add_request(std::move(warm));

  // Step 1: every prefix block is on disk -> a promotion is initiated for each
  // and the request is DEFERRED (the nullopt third state). Nothing scheduled.
  auto out1 = sched->schedule();
  CHECK(out1.num_scheduled_tokens.count("warm") == 0);

  // Step 2: the promotions completed at step 1's on_schedule_end; the connector
  // now reports the external hit and prefill is shortcut.
  auto out2 = sched->schedule();
  REQUIRE(out2.num_scheduled_tokens.count("warm") == 1);
  const int treated_prefill = out2.num_scheduled_tokens.at("warm");

  // The connector recomputes at least one token (num_tokens - 1 rule), so it
  // matches the first 2 of 3 blocks: 32 external tokens, 16 recomputed.
  const int expected_hit_blocks = (kPromptTokens - 1) / kBlockSize;  // 2
  const int prefill_saved = baseline_prefill - treated_prefill;
  CHECK(prefill_saved == expected_hit_blocks * kBlockSize);  // 32 tokens saved
  CHECK(treated_prefill == kPromptTokens - prefill_saved);   // 16 recomputed

  // Hit demonstration numbers (recorded in the report / BENCHMARKS).
  const int lookups = kPromptBlocks;                 // 3 prefix blocks probed
  const int hits = expected_hit_blocks;              // 2 promoted from disk
  MESSAGE("offload hit rate = " << hits << "/" << lookups << " blocks; "
          << "prefill tokens saved = " << prefill_saved << "/"
          << kPromptTokens);
  CHECK(mgr->get_stats().promotions == hits);
  CHECK(mgr->get_stats().refusals == 0);

  // BYTE IDENTITY: the promoted CPU bytes equal the cold-stored bytes, so the
  // continuation is token-identical to a cold run. (A wrong promoted block is a
  // silent wrong-output bug — this is the guard.)
  const ReqContext wctx{"warm"};
  for (int j = 0; j < hits; ++j) {
    const OffloadKey key = make_offload_key(req_probe->block_hashes[j], 0);
    REQUIRE(mgr->lookup(key, wctx) == LookupResult::kHit);
    auto spec = mgr->prepare_load({key}, wctx);
    auto* cpu = dynamic_cast<CPULoadStoreSpec*>(spec.get());
    REQUIRE(cpu != nullptr);
    std::vector<uint8_t> got(kPage);
    std::memcpy(got.data(), view->slot(cpu->block_ids[0]), kPage);
    CHECK(got == stored[j]);  // byte-identical across the restart
    mgr->complete_load({key}, wctx);
  }
}
