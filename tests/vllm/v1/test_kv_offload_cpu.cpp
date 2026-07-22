// Ported from: tests/v1/kv_offload/cpu/test_manager.py @ e24d1b24 (the LRU and
// ARC cases, prepare_store/complete_store, the not-ready sentinel, the atomic
// evict) and tests/v1/kv_offload/cpu/test_gpu_worker.py (the transfer cases,
// run here over the host-resident path).
#include <doctest/doctest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_offload/base.h"
#include "vllm/v1/kv_offload/cache_policy.h"
#include "vllm/v1/kv_offload/cpu_manager.h"
#include "vllm/v1/kv_offload/kv_block_transfer.h"
#include "vt/backend.h"

using namespace vllm::v1::kv_offload;  // NOLINT(build/namespaces)
using vllm::v1::BlockHash;

namespace {

// A distinct 32-byte digest per index, packed with a group id exactly as the
// production key encoding does.
OffloadKey Key(int i, uint32_t group = 0) {
  BlockHash h(32, '\0');
  h[0] = static_cast<char>(i & 0xff);
  h[1] = static_cast<char>((i >> 8) & 0xff);
  return make_offload_key(h, group);
}

const ReqContext kCtx{"req-0"};

}  // namespace

// --- OffloadKey: the encoding that must already match upstream ---------------

TEST_CASE("OffloadKey packs a 4-byte BIG-ENDIAN group id, byte-identically") {
  // Upstream: block_hash + group_idx.to_bytes(4, "big") (base.py:35-37). Our
  // BlockHashWithGroupId already packs this way, which is what makes an offload
  // tier possible without a key conversion layer.
  BlockHash h(32, 'x');
  const OffloadKey k = make_offload_key(h, 0x01020304u);
  REQUIRE(k.size() == 36);
  CHECK(static_cast<unsigned char>(k[32]) == 0x01);
  CHECK(static_cast<unsigned char>(k[33]) == 0x02);
  CHECK(static_cast<unsigned char>(k[34]) == 0x03);
  CHECK(static_cast<unsigned char>(k[35]) == 0x04);
  CHECK(get_offload_block_hash(k) == h);
  CHECK(get_offload_group_idx(k) == 0x01020304u);
}

// --- CachePolicy contracts ---------------------------------------------------

TEST_CASE("BlockStatus starts NOT READY with the ref_cnt == -1 sentinel") {
  // cpu/policies/base.py:20-25. Readiness is not a separate flag; a port with
  // a `bool ready` beside a non-negative count desynchronizes.
  BlockStatus b(7);
  CHECK(b.ref_cnt == -1);
  CHECK_FALSE(b.is_ready());
  b.ref_cnt = 0;
  CHECK(b.is_ready());
}

TEST_CASE("make_cache_policy rejects an unknown policy name") {
  CHECK_THROWS_AS(make_cache_policy("lfu", 8), std::invalid_argument);
  CHECK_NOTHROW(make_cache_policy("lru", 8));
  CHECK_NOTHROW(make_cache_policy("arc", 8));
}

TEST_CASE("evict is ATOMIC: an unsatisfiable request changes NOTHING") {
  // cpu/policies/base.py:70-73 — a partial eviction is wrong, because the
  // caller uses the all-or-nothing answer to decide whether to skip the store.
  for (const char* name : {"lru", "arc"}) {
    CAPTURE(name);
    auto policy = make_cache_policy(name, 4);
    for (int i = 0; i < 3; ++i) {
      BlockStatus b(i);
      b.ref_cnt = 0;  // idle => evictable
      policy->insert(Key(i), b);
    }
    // Asking for more than exist must fail and leave all three in place.
    OffloadKeySet none;
    CHECK_FALSE(policy->evict(5, none).has_value());
    for (int i = 0; i < 3; ++i) {
      CHECK(policy->get(Key(i)) != nullptr);
    }
    // Protected keys reduce the candidate pool the same way.
    OffloadKeySet all{Key(0), Key(1), Key(2)};
    CHECK_FALSE(policy->evict(1, all).has_value());
    for (int i = 0; i < 3; ++i) {
      CHECK(policy->get(Key(i)) != nullptr);
    }
    // A satisfiable request succeeds.
    auto evicted = policy->evict(2, none);
    REQUIRE(evicted.has_value());
    CHECK(evicted->size() == 2);
  }
}

TEST_CASE("evict(0) is a no-op that SUCCEEDS") {
  auto policy = make_cache_policy("lru", 4);
  OffloadKeySet none;
  auto r = policy->evict(0, none);
  REQUIRE(r.has_value());
  CHECK(r->empty());
}

TEST_CASE("LRU evicts least-recently-used, and touch refreshes recency") {
  auto policy = make_cache_policy("lru", 4);
  for (int i = 0; i < 3; ++i) {
    BlockStatus b(i);
    b.ref_cnt = 0;
    policy->insert(Key(i), b);
  }
  // Touch key 0 so it is no longer the oldest.
  policy->touch({Key(0)});
  OffloadKeySet none;
  auto evicted = policy->evict(1, none);
  REQUIRE(evicted.has_value());
  CHECK(evicted->front().first == Key(1));
}

TEST_CASE("LRU never evicts a PINNED block") {
  auto policy = make_cache_policy("lru", 4);
  BlockStatus pinned(0);
  pinned.ref_cnt = 1;  // a load is in flight
  policy->insert(Key(0), pinned);
  BlockStatus idle(1);
  idle.ref_cnt = 0;
  policy->insert(Key(1), idle);
  OffloadKeySet none;
  auto evicted = policy->evict(1, none);
  REQUIRE(evicted.has_value());
  CHECK(evicted->front().first == Key(1));
  CHECK(policy->get(Key(0)) != nullptr);
}

TEST_CASE("ARC promotes a re-touched block from T1 to T2") {
  auto policy = make_cache_policy("arc", 4);
  for (int i = 0; i < 2; ++i) {
    BlockStatus b(i);
    b.ref_cnt = 0;
    policy->insert(Key(i), b);
  }
  // Key(0) is touched -> promoted to T2 (frequent). Key(1) stays in T1
  // (recent), so it is the first eviction candidate while T1 is over target.
  policy->touch({Key(0)});
  OffloadKeySet none;
  auto evicted = policy->evict(1, none);
  REQUIRE(evicted.has_value());
  CHECK(evicted->front().first == Key(1));
}

TEST_CASE("ARC does NOT promote a block that is still being stored") {
  // arc.py:78-84 — a not-ready block was only just PREPARED for storing, so
  // touching it is not a genuine second access.
  auto policy = make_cache_policy("arc", 4);
  BlockStatus storing(0);  // ref_cnt == -1, not ready
  policy->insert(Key(0), storing);
  policy->touch({Key(0)});
  BlockStatus* b = policy->get(Key(0));
  REQUIRE(b != nullptr);
  CHECK_FALSE(b->is_ready());
}

// --- CPUOffloadingManager -----------------------------------------------------

TEST_CASE("a stored block is HIT_PENDING until complete_store publishes it") {
  CPUOffloadingManager mgr(/*num_blocks=*/4);
  CHECK(mgr.lookup(Key(0), kCtx) == LookupResult::kMiss);

  auto prep = mgr.prepare_store({Key(0)}, kCtx);
  REQUIRE(prep.has_value());
  CHECK(prep->keys_to_store.size() == 1);
  CHECK(std::string(prep->store_spec->medium()) == "CPU");
  // NOT a hit yet: reading it now would read a half-written block.
  CHECK(mgr.lookup(Key(0), kCtx) == LookupResult::kHitPending);

  mgr.complete_store({Key(0)}, kCtx, /*success=*/true);
  CHECK(mgr.lookup(Key(0), kCtx) == LookupResult::kHit);
}

TEST_CASE("a FAILED store releases the block back to the pool") {
  CPUOffloadingManager mgr(/*num_blocks=*/2);
  auto prep = mgr.prepare_store({Key(0), Key(1)}, kCtx);
  REQUIRE(prep.has_value());
  CHECK(mgr.num_free_blocks() == 0);
  mgr.complete_store({Key(0), Key(1)}, kCtx, /*success=*/false);
  CHECK(mgr.lookup(Key(0), kCtx) == LookupResult::kMiss);
  CHECK(mgr.num_free_blocks() == 2);
}

TEST_CASE("prepare_store SKIPS blocks that are already stored") {
  CPUOffloadingManager mgr(/*num_blocks=*/4);
  auto first = mgr.prepare_store({Key(0)}, kCtx);
  REQUIRE(first.has_value());
  mgr.complete_store({Key(0)}, kCtx);
  auto second = mgr.prepare_store({Key(0)}, kCtx);
  REQUIRE(second.has_value());
  CHECK(second->keys_to_store.empty());
}

TEST_CASE("prepare_store returning NOTHING is a control path, not an error") {
  // cpu/manager.py:192-194 — "eviction could not be satisfied, skip the store".
  // Every block is pinned by an in-flight load, so nothing can be evicted.
  CPUOffloadingManager mgr(/*num_blocks=*/2);
  auto prep = mgr.prepare_store({Key(0), Key(1)}, kCtx);
  REQUIRE(prep.has_value());
  mgr.complete_store({Key(0), Key(1)}, kCtx);
  mgr.prepare_load({Key(0), Key(1)}, kCtx);  // pin both

  auto blocked = mgr.prepare_store({Key(2)}, kCtx);
  CHECK_FALSE(blocked.has_value());

  // Once the loads complete the blocks are evictable again and the store fits.
  mgr.complete_load({Key(0), Key(1)}, kCtx);
  auto ok = mgr.prepare_store({Key(2)}, kCtx);
  REQUIRE(ok.has_value());
  CHECK(ok->keys_to_store.size() == 1);
  CHECK(ok->evicted_keys.size() == 1);
}

TEST_CASE("a block from the SAME batch is never evicted to make room for it") {
  // cpu/manager.py:186-190 — `protected = set(keys)`: a block that was already
  // stored must still be in the cache when prepare_store returns.
  CPUOffloadingManager mgr(/*num_blocks=*/2);
  auto p = mgr.prepare_store({Key(0)}, kCtx);
  REQUIRE(p.has_value());
  mgr.complete_store({Key(0)}, kCtx);

  // Ask to store {Key(0) (already there), Key(1), Key(2)} — capacity 2, so one
  // eviction is needed, and it must NOT be Key(0).
  auto p2 = mgr.prepare_store({Key(0), Key(1), Key(2)}, kCtx);
  if (p2.has_value()) {
    for (const OffloadKey& k : p2->evicted_keys) {
      CHECK(k != Key(0));
    }
  }
  CHECK(mgr.lookup(Key(0), kCtx) != LookupResult::kMiss);
}

TEST_CASE("prepare_load pins, complete_load releases") {
  CPUOffloadingManager mgr(/*num_blocks=*/2);
  auto p = mgr.prepare_store({Key(0)}, kCtx);
  REQUIRE(p.has_value());
  mgr.complete_store({Key(0)}, kCtx);

  auto spec = mgr.prepare_load({Key(0)}, kCtx);
  auto* block_ids = dynamic_cast<CPULoadStoreSpec*>(spec.get());
  REQUIRE(block_ids != nullptr);
  CHECK(block_ids->block_ids.size() == 1);
  mgr.complete_load({Key(0)}, kCtx);
  // Releasing twice is a contract violation, not a silent no-op.
  CHECK_THROWS(mgr.complete_load({Key(0)}, kCtx));
}

TEST_CASE("store_threshold defers a store until the block has been seen N times") {
  CPUOffloadingManager mgr(/*num_blocks=*/4, "lru", /*enable_events=*/false,
                           /*store_threshold=*/2);
  // First lookup: count 1, below the threshold -> the store is skipped.
  mgr.lookup(Key(0), kCtx);
  auto first = mgr.prepare_store({Key(0)}, kCtx);
  REQUIRE(first.has_value());
  CHECK(first->keys_to_store.empty());
  CHECK(mgr.get_stats().stores_skipped == 1);

  // Second lookup: count 2, threshold met -> the store proceeds.
  mgr.lookup(Key(0), kCtx);
  auto second = mgr.prepare_store({Key(0)}, kCtx);
  REQUIRE(second.has_value());
  CHECK(second->keys_to_store.size() == 1);
}

TEST_CASE("events are emitted only when enabled") {
  CPUOffloadingManager off(/*num_blocks=*/2);
  auto p = off.prepare_store({Key(0)}, kCtx);
  REQUIRE(p.has_value());
  off.complete_store({Key(0)}, kCtx);
  CHECK(off.take_events().empty());

  CPUOffloadingManager on(/*num_blocks=*/2, "lru", /*enable_events=*/true);
  auto q = on.prepare_store({Key(0)}, kCtx);
  REQUIRE(q.has_value());
  on.complete_store({Key(0)}, kCtx);
  auto events = on.take_events();
  REQUIRE(events.size() == 1);
  CHECK_FALSE(events[0].removed);
  CHECK(events[0].medium == "CPU");
  CHECK(on.take_events().empty());  // draining is destructive
}

TEST_CASE("reset_cache drops everything and restores the full pool") {
  CPUOffloadingManager mgr(/*num_blocks=*/4);
  auto p = mgr.prepare_store({Key(0), Key(1)}, kCtx);
  REQUIRE(p.has_value());
  mgr.complete_store({Key(0), Key(1)}, kCtx);
  mgr.reset_cache();
  CHECK(mgr.lookup(Key(0), kCtx) == LookupResult::kMiss);
  CHECK(mgr.num_free_blocks() == 4);
}

// --- the transfer half --------------------------------------------------------

TEST_CASE("device<->host KV block transfer round-trips byte-exactly") {
  // tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer, run over the
  // host-resident KV path so it gates on CPU. The page is OPAQUE BYTES, which
  // is exactly why the same code serves full attention and MLA.
  const vt::Device device{vt::DeviceType::kCPU, 0};
  vt::Queue queue = vt::CreateQueue(device);
  const size_t kLayers = 2;
  const size_t kDeviceBlocks = 8;
  const size_t kHostBlocks = 4;
  const size_t kPage = 256;

  std::vector<std::vector<uint8_t>> device_mem(
      kLayers, std::vector<uint8_t>(kDeviceBlocks * kPage, 0));
  std::vector<KVLayerRegion> layers;
  for (size_t l = 0; l < kLayers; ++l) {
    for (size_t i = 0; i < device_mem[l].size(); ++i) {
      device_mem[l][i] = static_cast<uint8_t>((l * 31 + i * 7) & 0xff);
    }
    layers.push_back(KVLayerRegion{device_mem[l].data(), kPage, kDeviceBlocks});
  }
  const std::vector<std::vector<uint8_t>> original = device_mem;

  CPUBackingStore store(device, kLayers, kHostBlocks, kPage);
  KVBlockTransferWorker worker(device, queue, layers, store);

  // Store device blocks {2, 5} into host slots {0, 1}.
  TransferJob store_job;
  store_job.direction = TransferDirection::kDeviceToHost;
  store_job.device_block_ids = {2, 5};
  store_job.host_block_ids = {0, 1};
  const int64_t id = worker.submit(store_job);
  worker.wait(id);

  for (size_t l = 0; l < kLayers; ++l) {
    CHECK(std::memcmp(store.slot(l, 0), original[l].data() + 2 * kPage, kPage) ==
          0);
    CHECK(std::memcmp(store.slot(l, 1), original[l].data() + 5 * kPage, kPage) ==
          0);
  }

  // Scribble over the device blocks, then restore them from the host slots.
  for (size_t l = 0; l < kLayers; ++l) {
    std::memset(device_mem[l].data() + 2 * kPage, 0xAB, kPage);
    std::memset(device_mem[l].data() + 5 * kPage, 0xCD, kPage);
  }
  TransferJob load_job;
  load_job.direction = TransferDirection::kHostToDevice;
  load_job.device_block_ids = {2, 5};
  load_job.host_block_ids = {0, 1};
  worker.wait(worker.submit(load_job));

  for (size_t l = 0; l < kLayers; ++l) {
    CHECK(device_mem[l] == original[l]);
  }
  vt::DestroyQueue(queue);
}

TEST_CASE("a transfer whose page size disagrees with the store is REFUSED") {
  // A silent mismatch would transfer a partial page. Page size is a per-spec
  // quantity (full attention and MLA differ by more than a factor of 2), so it
  // must agree exactly rather than be clamped.
  const vt::Device device{vt::DeviceType::kCPU, 0};
  vt::Queue queue = vt::CreateQueue(device);
  std::vector<uint8_t> mem(4 * 128, 0);
  std::vector<KVLayerRegion> layers{KVLayerRegion{mem.data(), 128, 4}};
  CPUBackingStore store(device, 1, 2, /*page_size_bytes=*/256);
  CHECK_THROWS_AS(KVBlockTransferWorker(device, queue, layers, store),
                  std::invalid_argument);
  vt::DestroyQueue(queue);
}

TEST_CASE("an out-of-range device block id is REFUSED, not clamped") {
  const vt::Device device{vt::DeviceType::kCPU, 0};
  vt::Queue queue = vt::CreateQueue(device);
  std::vector<uint8_t> mem(2 * 64, 0);
  std::vector<KVLayerRegion> layers{KVLayerRegion{mem.data(), 64, 2}};
  CPUBackingStore store(device, 1, 2, 64);
  KVBlockTransferWorker worker(device, queue, layers, store);
  TransferJob job;
  job.direction = TransferDirection::kDeviceToHost;
  job.device_block_ids = {99};
  job.host_block_ids = {0};
  CHECK_THROWS_AS(worker.submit(job), std::out_of_range);
  vt::DestroyQueue(queue);
}
