// Ported from: vllm/distributed/kv_transfer/kv_connector/v1/base.py:453-561
//               (the SCHEDULER-side hooks of KVConnectorBase_V1) @ e24d1b24
//               vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
//               (OffloadingConnectorScheduler) @ e24d1b24
//
// Scope (KV-OFFLOAD W4, the connector/scheduler HALF): how a connector
// participates in the scheduler's block accounting so a cross-request or
// restarted-process prefix HIT actually loads from a tier and SHORTCUTS PREFILL.
// This is where offload becomes a real speedup rather than just storage.
//
// WE PORT THE SEMANTICS, NOT THE PLUGIN ABI. We are pure C++ with no Python
// connector ABI; the dynamic `importlib` module path, the msgspec metadata
// structs and the `SupportsHMA` isinstance dispatch are Python-incidental and
// are NOT reproduced (recorded deviation — the C++ analogue is compile-time
// wiring). What IS load-bearing for correctness and is mirrored EXACTLY:
//
//   * get_num_new_matched_tokens has THREE states (base.py:468-478). A hit
//     count, `std::nullopt` = "not ready, re-ask next step" (the promotion is
//     still running in a lower tier), and an async-load flag. Treating the
//     nullopt as zero would spin or serve a partial prefix. The nullopt arises
//     from the tiering manager's RETRY/HIT_PENDING while a disk->CPU promotion
//     is in flight (scheduler.py:744-750 pops + defers + continues).
//   * BLOCK-HASH AGREEMENT (base.py:536-549). The offload keys are the SAME
//     Request::block_hashes the prefix cache uses, strided by the
//     hash_block_size_factor (offloading/scheduler.py:270-284): offloaded block
//     j keys on the LAST fine-grained hash of its factor-group,
//     block_hashes[hbsf*j + hbsf-1]. Today hbsf == 1 (block_size ==
//     hash_block_size), so the stride is identity, but the offset math is kept
//     so a >1 factor is a config change, not a code change.
//   * LOAD-BEFORE-COMPUTE ORDERING. update_state_after_alloc records a load of
//     the hit keys INTO the freshly-allocated GPU blocks; the worker must finish
//     that load before those tokens are treated as computed. The connector
//     early-outs when num_external_tokens == 0 (offloading/scheduler.py:698-699),
//     which is why the scheduler's second (post-load) call is a no-op.
//   * build_connector_meta RESETS the connector's per-step batch state
//     (base.py:516-517).
//
// DEVIATION, recorded: W4 ships the SYNCHRONOUS-load shape (async flag always
// false, mirroring LMCacheConnectorV1 which "always returns (n, False)",
// lmcache_connector.py:230-254). The genuinely asynchronous part — the disk->CPU
// promotion — is handled by the tiering manager's RETRY/re-ask across a step
// boundary; the CPU->GPU leg is synchronous within the worker phase. The
// cross-step WAITING_FOR_REMOTE_KVS async-GPU-load buffer (scheduler.py:948-969)
// is NOT ported in W4; it is a W5 generalization behind the same nullopt seam.
#ifndef VLLM_V1_KV_OFFLOAD_KV_CONNECTOR_H_
#define VLLM_V1_KV_OFFLOAD_KV_CONNECTOR_H_

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/v1/kv_offload/base.h"
#include "vllm/v1/kv_offload/tiering_manager.h"

namespace vllm::v1 {
struct Request;
}  // namespace vllm::v1

namespace vllm::v1::kv_offload {

// Result of get_num_new_matched_tokens (base.py:453-486).
struct MatchResult {
  // std::nullopt is the THIRD state: "not ready, re-ask next step". A present
  // value is the number of EXTERNAL tokens matched beyond num_computed_tokens.
  std::optional<int> num_matched_tokens;
  // When true the tokens load asynchronously between steps (must be false when
  // num_matched_tokens is 0). W4 is always false (see the header deviation).
  bool load_async = false;
};

// A load the worker must execute before the loaded tokens count as computed:
// the tier bytes for `keys` go into GPU blocks `gpu_block_ids` (same order).
struct ConnectorLoadJob {
  std::string req_id;
  std::vector<OffloadKey> keys;
  std::vector<int> gpu_block_ids;
};

// The SCHEDULER-facing half of a KV connector. W4 ships ONE concrete
// implementation (OffloadingConnector); the full 7-method ABI with compile-time
// registration and KVTransferConfig is the W5 generalization behind this seam.
// The scheduler holds a non-owning pointer and calls these; a null pointer means
// no connector and ZERO behaviour change (offloading is opt-in and inert off).
class KVConnectorScheduler {
 public:
  virtual ~KVConnectorScheduler() = default;

  // base.py:453-486. MUST be side-effect free in the idempotent sense: it may
  // record which keys were hit (overwritten each call) but re-calling with the
  // same arguments must yield the same answer.
  virtual MatchResult get_num_new_matched_tokens(const Request& request,
                                                 int num_computed_tokens) = 0;

  // base.py:488-507. `block_ids` are the request's per-group allocated GPU
  // blocks. May be called twice per request under async load; the ext==0
  // early-out makes the second call a no-op.
  virtual void update_state_after_alloc(
      const Request& request, const std::vector<std::vector<int>>& block_ids,
      int num_external_tokens) = 0;

  // base.py:509-522. Drains and RESETS the per-step batch state, returning the
  // loads the worker must perform this step.
  virtual std::vector<ConnectorLoadJob> build_connector_meta() = 0;

  // base.py:542-561 / scheduler.py:2341-2371. Called once before a request's
  // blocks are freed. block_ids are ignored by the offloading connector (it
  // tracks its own), so the SupportsHMA single-vs-all-groups split collapses to
  // one method — the Python-incidental part.
  virtual void request_finished(const Request& request) = 0;

  // Once per step, after build_connector_meta: flush deferred promotions and
  // poll transfers (tiering manager on_schedule_end). Separate hook because the
  // scheduler calls it unconditionally at the end of schedule().
  virtual void on_schedule_end() = 0;
};

struct OffloadingConnectorConfig {
  // Which KV cache group is offloaded. W4 offloads the full-attention group
  // (default 0); GDN/Mamba recurrent state is not block-addressed and is out of
  // scope (§Risks R9).
  uint32_t group_idx = 0;
  // Tokens per offloaded block = hash_block_size_factor * hash_block_size. Today
  // hbsf == 1, so this equals the scheduler block size.
  int offload_block_tokens = 16;
  int hash_block_size_factor = 1;
  // The offload policy: block-level (skip prefix hits) or request-level.
  OffloadPolicy policy = OffloadPolicy::kBlockLevel;
  // offload_prompt_only (kv_offloading_usage.md:64-82, default true): only the
  // prompt's blocks are stored, never generated tokens.
  bool offload_prompt_only = true;
};

// The tiered-offload connector, scheduler side. Drives ONE
// TieringOffloadingManager (owned externally) over Request::block_hashes.
class OffloadingConnector final : public KVConnectorScheduler {
 public:
  OffloadingConnector(TieringOffloadingManager& manager,
                      OffloadingConnectorConfig config);

  MatchResult get_num_new_matched_tokens(const Request& request,
                                         int num_computed_tokens) override;
  void update_state_after_alloc(
      const Request& request, const std::vector<std::vector<int>>& block_ids,
      int num_external_tokens) override;
  std::vector<ConnectorLoadJob> build_connector_meta() override;
  void request_finished(const Request& request) override;
  void on_schedule_end() override;

  // Record that a request should have its computed blocks offloaded (called when
  // a request is first seen). Mirrors on_new_request registration.
  void on_new_request(const Request& request);

  // The offload keys for the request's blocks fully covered by
  // num_computed_tokens (clamped to the prompt when offload_prompt_only). These
  // are the keys the worker's GPU->CPU->disk store path writes; exposed so the
  // correctness harness can drive the store (manager prepare_store/complete_store)
  // deterministically. Pure function of block_hashes — does not touch the manager.
  std::vector<OffloadKey> store_keys(const Request& request,
                                     int num_computed_tokens) const;

  TieringOffloadingManager& manager() { return manager_; }
  const OffloadingConnectorConfig& config() const { return config_; }

 private:
  // Stride Request::block_hashes into the offloaded-block key at index j.
  std::optional<OffloadKey> offload_key(const Request& request, int j) const;

  TieringOffloadingManager& manager_;
  OffloadingConnectorConfig config_;
  // Per-request hit keys recorded by the last get_num_new_matched_tokens (used
  // by update_state_after_alloc to build the load). Idempotent.
  std::unordered_map<std::string, std::vector<OffloadKey>> hit_keys_;
  // The current step's accumulated loads, drained + reset by
  // build_connector_meta.
  std::vector<ConnectorLoadJob> batch_loads_;
};

}  // namespace vllm::v1::kv_offload

#endif  // VLLM_V1_KV_OFFLOAD_KV_CONNECTOR_H_
