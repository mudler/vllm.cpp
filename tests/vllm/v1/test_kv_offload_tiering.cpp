// Ported from: tests/v1/kv_offload/tiering/test_tiering_offloading.py @ e24d1b24
// (promotion / cascade / reset ordering) PLUS the identity-refusal-through-a-
// promotion safety case that upstream has no counterpart for.
//
// These are the KV-OFFLOAD W4 correctness gates for the TIERING MANAGER:
//   * a disk->CPU promotion is a RETRY this step and a HIT the next, and the
//     promoted bytes are BYTE-IDENTICAL to what was stored (the silent-wrong-
//     output guard: a wrong promoted block would produce wrong tokens);
//   * a block evicted (demoted) from the CPU tier survives on disk and promotes
//     back byte-identically;
//   * reset_cache clears the CPU tier but DELIBERATELY leaves the disk tier
//     intact (a persisted cache survives a prefix-cache reset);
//   * a FRESH manager opening the same directory (a process restart) promotes;
//   * a promotion RE-VERIFIES identity and REFUSES a foreign/corrupt disk block,
//     treating it as absent — never trusting the lower tier blindly.
//
// GATE 9 (disk hygiene): every case bounds its root_dir and removes it.
#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_offload/base.h"
#include "vllm/v1/kv_offload/cache_identity.h"
#include "vllm/v1/kv_offload/cpu_manager.h"
#include "vllm/v1/kv_offload/fs_tier.h"
#include "vllm/v1/kv_offload/tiering_manager.h"

using namespace vllm::v1::kv_offload;  // NOLINT(build/namespaces)
using vllm::v1::BlockHash;

namespace {

class TempDir {
 public:
  explicit TempDir(const std::string& tag) {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("vllmcpp_kvtier_" + tag + "_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::string str() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

constexpr int64_t kPage = 512;

OffloadKey Key(int i) {
  BlockHash h(32, '\0');
  h[0] = static_cast<char>(i & 0xff);
  h[1] = static_cast<char>((i >> 8) & 0xff);
  h[2] = static_cast<char>(0xA5);  // make every page's bytes distinctive
  return make_offload_key(h, 0);
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
  id.block_size = 16;
  id.hash_block_size = 16;
  id.kv_dtype = "bf16";
  id.none_hash_hex = "00";
  return id;
}

// Deterministic per-block payload: page filled with a function of the key so a
// byte mismatch after a promotion is detectable, and a "token" derived from it
// is stable.
std::vector<uint8_t> Payload(const OffloadKey& key) {
  std::vector<uint8_t> p(kPage);
  for (int64_t i = 0; i < kPage; ++i) {
    p[static_cast<size_t>(i)] =
        static_cast<uint8_t>(static_cast<unsigned char>(key[0]) + i * 7 + 13);
  }
  return p;
}

// A tiering manager over `cpu_blocks` CPU slots and a disk tier at `root`.
struct Fixture {
  std::unique_ptr<HeapPrimaryByteView> view;
  std::unique_ptr<TieringOffloadingManager> mgr;
  std::string fs_base;

  Fixture(const std::string& root, int64_t cpu_blocks,
          const CacheIdentity& id) {
    view = std::make_unique<HeapPrimaryByteView>(cpu_blocks, kPage);
    auto primary = std::make_unique<CPUOffloadingManager>(cpu_blocks, "lru");
    FileSystemTierOptions opts;
    opts.root_dir = root;
    opts.identity = id;
    auto secondary = std::make_unique<FileSystemTier>(opts);
    fs_base = secondary->base_path();
    mgr = std::make_unique<TieringOffloadingManager>(
        std::move(primary), std::move(secondary), *view);
  }

  // Simulate a GPU->CPU->disk store: reserve a CPU slot, fill it, publish +
  // cascade, then drive the cascade job to completion.
  void store(const OffloadKey& key, const std::vector<uint8_t>& payload,
             const ReqContext& ctx) {
    auto out = mgr->prepare_store({key}, ctx);
    REQUIRE(out.has_value());
    REQUIRE(out->store_spec != nullptr);
    auto* spec = dynamic_cast<CPULoadStoreSpec*>(out->store_spec.get());
    REQUIRE(spec != nullptr);
    REQUIRE(spec->block_ids.size() == 1);
    std::memcpy(view->slot(spec->block_ids[0]), payload.data(), payload.size());
    mgr->complete_store({key}, ctx, /*success=*/true);
    mgr->drain_jobs();  // finalize the CPU->disk cascade
  }

  // Drive a promotion to completion and read the promoted CPU bytes back.
  // Returns false if the block did not become a HIT (refused / absent).
  bool promote_and_read(const OffloadKey& key, const ReqContext& ctx,
                        std::vector<uint8_t>* out_bytes) {
    // Step N: lookup should MISS the CPU tier and RETRY on the disk hit.
    const LookupResult r = mgr->lookup(key, ctx);
    if (r == LookupResult::kMiss) return false;  // refused/absent
    CHECK(r == LookupResult::kRetry);
    mgr->on_schedule_end();  // flush the deferred disk->CPU load
    mgr->drain_jobs();       // finalize the promotion
    // Step N+1: the promoted block is now a HIT.
    if (mgr->lookup(key, ctx) != LookupResult::kHit) return false;
    auto spec = mgr->prepare_load({key}, ctx);
    auto* cpu = dynamic_cast<CPULoadStoreSpec*>(spec.get());
    REQUIRE(cpu != nullptr);
    REQUIRE(cpu->block_ids.size() == 1);
    out_bytes->assign(kPage, 0);
    std::memcpy(out_bytes->data(), view->slot(cpu->block_ids[0]), kPage);
    mgr->complete_load({key}, ctx);
    return true;
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Promotion: a disk hit is RETRY this step, HIT the next, and byte-identical.
// ---------------------------------------------------------------------------
TEST_CASE("Tiering: disk->CPU promotion is RETRY then HIT, byte-identical") {
  TempDir dir("promote");
  Fixture fx(dir.str(), /*cpu_blocks=*/4, BaseIdentity());
  const ReqContext ctx{"req-0"};
  const OffloadKey k = Key(1);
  const std::vector<uint8_t> payload = Payload(k);

  fx.mgr->on_new_request(ctx);
  fx.store(k, payload, ctx);

  // Evict from the CPU tier (simulates memory pressure / a fresh step) by a
  // prefix-cache reset — which must NOT touch the disk tier.
  fx.mgr->reset_cache();
  // Prove the promotion reads from DISK, not stale CPU memory, by scribbling the
  // whole primary byte arena first.
  for (int64_t b = 0; b < 4; ++b) std::memset(fx.view->slot(b), 0xEE, kPage);

  std::vector<uint8_t> got;
  REQUIRE(fx.promote_and_read(k, ctx, &got));
  CHECK(got == payload);  // BYTE-IDENTICAL to the cold store
  CHECK(fx.mgr->get_stats().promotions == 1);
  CHECK(fx.mgr->get_stats().refusals == 0);
}

// ---------------------------------------------------------------------------
// Demotion under pressure: a block evicted from the CPU tier survives on disk
// and promotes back byte-identically. Token derived from the bytes matches.
// ---------------------------------------------------------------------------
TEST_CASE("Tiering: a CPU-evicted block survives on disk and promotes back") {
  TempDir dir("demote");
  Fixture fx(dir.str(), /*cpu_blocks=*/2, BaseIdentity());
  const ReqContext ctx{"req-0"};
  fx.mgr->on_new_request(ctx);

  const OffloadKey k1 = Key(11), k2 = Key(12), k3 = Key(13);
  fx.store(k1, Payload(k1), ctx);
  fx.store(k2, Payload(k2), ctx);
  // Storing a third block into a 2-slot CPU tier EVICTS the LRU (k1) from CPU —
  // but k1 was already cascaded to disk, so it is recoverable.
  fx.store(k3, Payload(k3), ctx);

  // k1 is gone from the CPU tier (a plain miss would be wrong; it is a disk hit).
  std::memset(fx.view->slot(0), 0xEE, kPage);
  std::memset(fx.view->slot(1), 0xEE, kPage);
  std::vector<uint8_t> got;
  REQUIRE(fx.promote_and_read(k1, ctx, &got));
  CHECK(got == Payload(k1));  // demotion + promotion preserved every byte

  // A token derived from the bytes is identical to a cold run's.
  auto token = [](const std::vector<uint8_t>& b) {
    uint32_t s = 0;
    for (uint8_t x : b) s = s * 131 + x;
    return s;
  };
  CHECK(token(got) == token(Payload(k1)));
  CHECK(fx.mgr->get_stats().fs_num_blocks == 3);  // all three on disk
}

// ---------------------------------------------------------------------------
// Reset ordering: reset_cache clears the CPU tier but the disk tier SURVIVES.
// ---------------------------------------------------------------------------
TEST_CASE("Tiering: reset_cache clears CPU but the disk tier survives") {
  TempDir dir("reset");
  const CacheIdentity id = BaseIdentity();
  Fixture fx(dir.str(), /*cpu_blocks=*/4, id);
  const ReqContext ctx{"req-0"};
  fx.mgr->on_new_request(ctx);
  const OffloadKey k = Key(21);
  fx.store(k, Payload(k), ctx);
  CHECK(fx.mgr->get_stats().fs_num_blocks == 1);

  fx.mgr->reset_cache();
  // The disk tier is untouched: still one block, and a lookup is a disk hit
  // (RETRY), not a miss.
  CHECK(fx.mgr->get_stats().fs_num_blocks == 1);
  CHECK(fx.mgr->lookup(k, ctx) == LookupResult::kRetry);
}

// ---------------------------------------------------------------------------
// Restart: a FRESH manager opening the same directory promotes the persisted
// block byte-identically (cross-process persistence).
// ---------------------------------------------------------------------------
TEST_CASE("Tiering: a fresh manager on the same directory promotes (restart)") {
  TempDir dir("restart");
  const CacheIdentity id = BaseIdentity();
  const OffloadKey k = Key(31);
  const std::vector<uint8_t> payload = Payload(k);

  {  // "process 1": store then tear the manager down.
    Fixture fx1(dir.str(), /*cpu_blocks=*/4, id);
    const ReqContext ctx{"req-0"};
    fx1.mgr->on_new_request(ctx);
    fx1.store(k, payload, ctx);
  }

  // "process 2": a brand-new manager over the SAME disk directory.
  Fixture fx2(dir.str(), /*cpu_blocks=*/4, id);
  const ReqContext ctx{"req-1"};
  fx2.mgr->on_new_request(ctx);
  std::vector<uint8_t> got;
  REQUIRE(fx2.promote_and_read(k, ctx, &got));
  CHECK(got == payload);
  CHECK(fx2.mgr->get_stats().promotions == 1);
}

// ---------------------------------------------------------------------------
// Identity refusal THROUGH a promotion: a corrupt/foreign disk block is REFUSED
// (unlinked, treated as absent), never promoted and trusted.
// ---------------------------------------------------------------------------
TEST_CASE("Tiering: a promotion REFUSES a corrupt disk block, never trusts it") {
  TempDir dir("refuse");
  Fixture fx(dir.str(), /*cpu_blocks=*/4, BaseIdentity());
  const ReqContext ctx{"req-0"};
  fx.mgr->on_new_request(ctx);
  const OffloadKey k = Key(41);
  fx.store(k, Payload(k), ctx);
  fx.mgr->reset_cache();  // clear CPU; the block lives only on disk now

  // Corrupt the block file's header on disk (mimics a foreign / torn block). The
  // per-block verified header must catch it and REFUSE on load.
  // Files live under <base_path>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin, i.e. in
  // a sibling of base_path(); recurse from the tier root to find them.
  std::filesystem::path bin;
  for (auto& e : std::filesystem::recursive_directory_iterator(dir.str())) {
    if (e.is_regular_file() && e.path().extension() == ".bin") {
      bin = e.path();
      break;
    }
  }
  REQUIRE(!bin.empty());
  {
    std::fstream f(bin, std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(f.is_open());
    f.seekp(0);
    const char garbage[16] = {'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',
                              'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'};
    f.write(garbage, sizeof(garbage));
  }

  // The disk still SEES a file (existence is the index), so the lookup RETRYs and
  // a promotion is initiated — but the load re-verifies identity and refuses.
  std::vector<uint8_t> got;
  CHECK_FALSE(fx.promote_and_read(k, ctx, &got));  // never became a HIT
  CHECK(fx.mgr->get_stats().refusals == 1);
  // The offending file was unlinked (self-healing), so the next lookup is a
  // clean miss.
  CHECK(fx.mgr->lookup(k, ctx) == LookupResult::kMiss);
}
