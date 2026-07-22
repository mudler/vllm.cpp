// Ported from: vllm/v1/kv_offload/tiering/fs/manager.py:64-217 @ e24d1b24
//               vllm/v1/kv_offload/tiering/fs/thread_pool.py:41-180
//
// Scope: the DISK (`fs`) secondary tier — the user's headline ask. One raw file
// per block under a content-addressed path, no container and no index
// (EXISTENCE IS THE INDEX), reached through the CPU primary tier because a
// secondary tier may never touch device memory
// (vllm/v1/kv_offload/tiering/base.py:52-64).
//
// PORTED: the naming, the byte path, the atomic publish, the self-healing
// delete, the batched lookup, and the dual-queue read/write thread pool in
// which each worker can drain the OTHER queue so neither reads nor writes can
// starve (thread_pool.py:153-180).
//
// TWO PLACES WE DELIBERATELY EXCEED UPSTREAM
// (.agents/specs/kv-persistence-lmcache.md §B2, §B3):
//
//  * §B2 — A VERIFIED IDENTITY HEADER. Upstream's only identity mechanism is a
//    12-hex path digest, and the config.json it writes is NEVER READ
//    (file_mapper.py:16,122-126; tiering/fs/manager.py:131-137; no reader
//    exists anywhere in the repo). We read config.json back field by field at
//    tier open AND verify a per-block header on every single open, and a
//    mismatch REFUSES rather than warning. See cache_identity.h.
//
//  * §B3 — A BOUNDED, EVICTING TIER. Upstream's fs tier implements NEITHER
//    capacity accounting NOR eviction (`FileSystemTierManager` has no evict at
//    all; contrast cpu/manager.py:169-237), so files accumulate until something
//    external reclaims them. On a box with ~16 GiB free that presents as
//    unrelated bogus test failures rather than as a full disk — the recorded
//    ENOSPC lesson. We add a byte budget enforced over the SAME `CachePolicy`
//    seam the CPU tier uses. This is safe to add where upstream has nothing
//    because eviction is SEMANTICALLY TRANSPARENT: dropping a cached block
//    loses a hit, never changes a token.
#ifndef VLLM_V1_KV_OFFLOAD_FS_TIER_H_
#define VLLM_V1_KV_OFFLOAD_FS_TIER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/v1/kv_offload/base.h"
#include "vllm/v1/kv_offload/cache_identity.h"
#include "vllm/v1/kv_offload/fs_io.h"

namespace vllm::v1::kv_offload {

struct FileSystemTierOptions {
  // The tier root. Every test MUST bound this to a temporary directory and
  // clean it up (gate 9 — an unbounded tier filling the box shows up as
  // unrelated failures).
  std::string root_dir;
  // The full cache identity; determines the directory and every refusal.
  CacheIdentity identity;
  // Byte budget for the tier's payloads. 0 means UNBOUNDED, i.e. exactly
  // upstream's behaviour — available deliberately so a faithful-parity arm can
  // be measured, but NOT the default.
  int64_t capacity_bytes = 0;
  // Replacement policy for the bounded case: "lru" or "arc".
  std::string eviction_policy = "lru";
  int num_read_threads = 2;
  int num_write_threads = 2;
};

// Result of a batched existence probe.
using LookupResults = std::vector<bool>;

class FileSystemTier {
 public:
  // Opens (creating if absent) the tier directory. THROWS std::runtime_error
  // when an existing directory's config.json disagrees with `identity` on any
  // field — a refusal, never a warning.
  explicit FileSystemTier(FileSystemTierOptions options);
  ~FileSystemTier();
  FileSystemTier(const FileSystemTier&) = delete;
  FileSystemTier& operator=(const FileSystemTier&) = delete;

  // Batched existence probe. Upstream needs a GIL-releasing C extension to do
  // this without stalling the interpreter (csrc/fs_io.cpp:23,40); with no GIL
  // this is an ordinary parallel loop (§B1).
  LookupResults lookup(const std::vector<OffloadKey>& keys) const;

  // Store one block synchronously. `payload_size` MUST equal the identity's
  // page_size_bytes. Skips blocks already present. May evict when a byte budget
  // is configured.
  void store(const OffloadKey& key, const void* payload, size_t payload_size);

  // Load one block synchronously. Returns false on a plain miss. THROWS on any
  // identity/format/truncation failure, having first UNLINKED the offending
  // file so the next lookup is a clean miss.
  bool load(const OffloadKey& key, void* out, size_t out_capacity);

  // Asynchronous forms over the dual-queue pool. `submit_*` returns a job id;
  // `wait` blocks for one job and rethrows whatever it threw.
  int64_t submit_store(const OffloadKey& key, const void* payload,
                       size_t payload_size);
  int64_t submit_load(const OffloadKey& key, void* out, size_t out_capacity);
  // Returns the load's hit/miss result (always true for a store).
  bool wait(int64_t job_id);

  // Drop every block this tier holds (files included).
  void reset_cache();

  // --- observability ---------------------------------------------------------
  int64_t num_blocks() const;
  int64_t bytes_used() const;
  int64_t capacity_bytes() const;
  // Blocks evicted by the byte budget since construction (§B3 evidence).
  int64_t num_evicted() const;
  const std::string& base_path() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm::v1::kv_offload

#endif  // VLLM_V1_KV_OFFLOAD_FS_TIER_H_
