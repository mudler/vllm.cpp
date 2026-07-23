// Ported from: vllm/distributed/kv_transfer/kv_connector/v1/base.py @ e24d1b24
//   - KVConnectorBase_V1 (:171) — the abstract connector ABI
//   - KVConnectorRole (:124), SupportsHMA (:85-117)
//   - vllm/distributed/kv_transfer/kv_connector/factory.py:27-125 — the registry
//   - vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
//     (OffloadingConnectorScheduler) — the ONE concrete connector W4 shipped
//
// Scope (KV-CONNECTORS W5): GENERALIZE W4's single connector into a clean
// abstract base so multiple backends (the landed disk-offload tier now; the
// LMCache lm:// client next, over the SAME seam) implement ONE ABI. This is the
// connector analogue of the `LinearMethod` seam: POLICY (which connector) is
// config-selected via KVTransferConfig + the compile-time KVConnectorFactory;
// the connector implements the transport.
//
// WE PORT THE SEMANTICS, NOT THE PLUGIN ABI. We are pure C++ with no Python
// connector ABI; the dynamic `importlib` module path (factory.py:96-123), the
// msgspec metadata structs and the `SupportsHMA` isinstance dispatch are
// Python-incidental and are NOT reproduced (recorded deviation — the C++ analogue
// is compile-time registration, mirroring REGISTER_VLLM_MODEL).
//
// LOAD-BEARING vs NO-OP for our SYNCHRONOUS runner (stated explicitly, as the
// spec's port map requires):
//
//   LOAD-BEARING FOR CORRECTNESS (scheduler side — every concrete connector MUST
//   implement these; getting any wrong changes tokens):
//     * get_num_new_matched_tokens (base.py:454) — THREE states: a hit count,
//       std::nullopt = "not ready, re-ask next step" (a lower-tier promotion is
//       in flight), and the async-load flag. Treating nullopt as 0 spins or
//       serves a partial prefix (§Risks R5).
//     * update_state_after_alloc (base.py:489) — records the LOAD of the hit keys
//       INTO the freshly-allocated GPU blocks (load-before-compute). May be
//       called twice under async load; the ext==0 early-out makes the 2nd a no-op.
//     * build_connector_meta (base.py:510) — drains + RESETS the per-step batch
//       state, emitting the loads the worker must run this step.
//     * request_finished (base.py:542) — returning delay_free==true transfers
//       block-freeing OWNERSHIP to the connector until get_finished() reports the
//       id (§Risks R6). SupportsHMA (base.py:85) routes the multi-group form
//       (request_finished_all_groups) — our gate models are two-group hybrids
//       (§Risks R7), so a connector declares supports_hma() and the base's
//       single-group path folds into it, mirroring scheduler.py:2340-2371.
//
//   NO-OP HOOKS for our synchronous runner (defaulted in the base; a connector
//   overrides only if it genuinely transfers off-thread):
//     * register_kv_caches (base.py:251) — the offloading connector binds its
//       byte view directly at construction; the base default is a no-op.
//     * start_load_kv (base.py:293) — our runner executes the build_connector_meta
//       loads directly in the worker phase rather than via a forward-context hook.
//     * wait_for_layer_load / save_kv_layer (base.py:311,325) — LAYERWISE
//       pipelining forces PIECEWISE cudagraphs and would forfeit our decode-graph
//       win; NOT SCHEDULED (§Risks R8). Defaulted no-op.
//     * wait_for_save (base.py:347) — synchronous: the store completes within
//       on_schedule_end, nothing is outstanding across the layer loop.
//     * get_finished (base.py:357) — synchronous connectors have nothing pending
//       across steps, so the default returns an empty set.
//
// DEVIATION, recorded: W4/W5 ship the SYNCHRONOUS-load shape (async flag always
// false, mirroring LMCacheConnectorV1 which "always returns (n, False)"). The
// genuinely asynchronous part — the disk->CPU promotion — is handled by the
// tiering manager's RETRY/re-ask across a step boundary behind the nullopt seam;
// the cross-step WAITING_FOR_REMOTE_KVS GPU-load buffer (scheduler.py:948-969) is
// not wired, and the nullopt third state is the seam a future async connector
// (e.g. NIXL) fills without a shape change.
#ifndef VLLM_V1_KV_OFFLOAD_KV_CONNECTOR_H_
#define VLLM_V1_KV_OFFLOAD_KV_CONNECTOR_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/config/kv_transfer.h"
#include "vllm/v1/kv_offload/base.h"
#include "vllm/v1/kv_offload/cache_identity.h"
#include "vllm/v1/kv_offload/tiering_manager.h"

namespace vllm::v1 {
struct Request;
}  // namespace vllm::v1

namespace vllm::v1::kv_offload {

// Upstream: class KVConnectorRole (base.py:124-130). Two instances of one class,
// one per process, never sharing memory.
enum class KVConnectorRole {
  kScheduler,  // running in the scheduler process
  kWorker,     // running in the worker process
};

// Result of get_num_new_matched_tokens (base.py:453-486).
struct MatchResult {
  // std::nullopt is the THIRD state: "not ready, re-ask next step". A present
  // value is the number of EXTERNAL tokens matched beyond num_computed_tokens.
  std::optional<int> num_matched_tokens;
  // When true the tokens load asynchronously between steps (must be false when
  // num_matched_tokens is 0). The synchronous connectors are always false.
  bool load_async = false;
};

// Result of request_finished (base.py:542-561). delay_free==true transfers
// block-freeing ownership to the connector until get_finished() reports the id.
struct RequestFinishedResult {
  bool delay_free = false;
  // Optional KVTransferParams echoed into the request output (base.py returns a
  // dict|None). Placeholder mirror; the disk/lmcache connectors return none.
  std::optional<std::map<std::string, std::string>> kv_transfer_params;
};

// A load the worker must execute before the loaded tokens count as computed:
// the tier bytes for `keys` go into GPU blocks `gpu_block_ids` (same order).
struct ConnectorLoadJob {
  std::string req_id;
  std::vector<OffloadKey> keys;
  std::vector<int> gpu_block_ids;
};

// The abstract KV connector ABI (mirrors KVConnectorBase_V1, base.py:171). ONE
// class spans both roles; a given instance runs in exactly one (role()). The
// scheduler holds a non-owning pointer to the scheduler-role instance and calls
// the scheduler-side methods; a null pointer means NO connector and ZERO
// behaviour change (offloading is opt-in and inert off).
class KVConnector {
 public:
  virtual ~KVConnector() = default;

  KVConnectorRole role() const { return role_; }

  // ---- SCHEDULER-side ABI (load-bearing; pure virtual) ----------------------

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

  // ---- request-finished / deferred-free ownership ---------------------------

  // True iff this connector supports the hybrid memory allocator (multi KV cache
  // group). base.py:85,117. Our gate models are two-group hybrids (§Risks R7), so
  // a connector that participates in their block lifecycle returns true and
  // implements request_finished_all_groups.
  virtual bool supports_hma() const { return false; }

  // base.py:542-561 / scheduler.py:2340-2369 (the non-HMA path). Called once
  // before a request's blocks are freed, for the SINGLE KV cache group. The base
  // default forwards to request_finished_all_groups with one group, so a
  // connector may implement either.
  virtual RequestFinishedResult request_finished(
      const Request& request, const std::vector<int>& block_ids) {
    return request_finished_all_groups(request, {block_ids});
  }

  // base.py:93-116 / scheduler.py:2371 (the SupportsHMA path). Called once when a
  // request finishes for ALL groups, before its blocks are freed per group. The
  // base default is inert (no deferred free).
  virtual RequestFinishedResult request_finished_all_groups(
      const Request& /*request*/,
      const std::vector<std::vector<int>>& /*block_ids_per_group*/) {
    return RequestFinishedResult{};
  }

  // ---- WORKER-side hooks (no-op for our synchronous runner; see header) ------

  // base.py:251. The offloading connector binds its byte view at construction;
  // the default is a no-op.
  virtual void register_kv_caches() {}
  // base.py:293. Our runner executes the build_connector_meta loads directly.
  virtual void start_load_kv() {}
  // base.py:311. LAYERWISE — NOT SCHEDULED (§Risks R8). No-op.
  virtual void wait_for_layer_load(const std::string& /*layer_name*/) {}
  // base.py:325. LAYERWISE — NOT SCHEDULED (§Risks R8). No-op.
  virtual void save_kv_layer(const std::string& /*layer_name*/) {}
  // base.py:347. Synchronous: the store completes in on_schedule_end. No-op.
  virtual void wait_for_save() {}
  // base.py:357. Ids of requests whose async save/load finished this step.
  // Synchronous connectors have nothing pending across steps -> empty.
  virtual std::vector<std::string> get_finished() { return {}; }

  // ---- per-step end hook (scheduler drives it unconditionally) --------------

  // Once per step, after build_connector_meta: flush deferred promotions and poll
  // transfers (tiering manager on_schedule_end). Base default is a no-op.
  virtual void on_schedule_end() {}

 protected:
  explicit KVConnector(KVConnectorRole role) : role_(role) {}

 private:
  KVConnectorRole role_;
};

// Runtime resources the engine supplies to a connector builder. A connector that
// needs none ignores them. Mirrors vLLM passing (vllm_config, role,
// kv_cache_config) to the connector ctor (factory.py:43-77).
struct KVConnectorContext {
  const KVTransferConfig* config = nullptr;
  KVConnectorRole role = KVConnectorRole::kScheduler;
  // Page geometry of the offloaded KV cache group (from its spec).
  std::int64_t page_size_bytes = 0;
  int block_size = 16;
  // The model identity written into (and verified against) the disk tier's
  // header (§B2). Non-null for a disk-offload connector.
  const CacheIdentity* identity = nullptr;
};

// A connector builder: constructs the concrete connector (and any resources it
// owns) from the context. Plain function pointer, mirroring the model factory
// table. Returns nullptr if the connector cannot be built for this role/context.
using KVConnectorBuilder =
    std::unique_ptr<KVConnector> (*)(const KVConnectorContext&);

// The compile-time connector registry (mirrors KVConnectorFactory,
// factory.py:27-125). Registration is by a static KVConnectorRegistrar from each
// connector's own TU (REGISTER_KV_CONNECTOR), replacing the Python lazy import.
class KVConnectorFactory {
 public:
  // factory.py:31-40. Duplicate registration is rejected.
  static void Register(const std::string& name, KVConnectorBuilder builder);

  // factory.py:43-77 + 96-125. Selection is by config: an absent connector
  // (empty kv_connector / not is_kv_transfer_instance) yields nullptr — the
  // default-OFF inert path. An unknown name throws.
  static std::unique_ptr<KVConnector> Create(const KVConnectorContext& ctx);

  static bool IsRegistered(const std::string& name);
  static std::vector<std::string> RegisteredNames();
};

// Static-init helper; used only through REGISTER_KV_CONNECTOR.
struct KVConnectorRegistrar {
  KVConnectorRegistrar(const std::string& name, KVConnectorBuilder builder) {
    KVConnectorFactory::Register(name, builder);
  }
};

// Registers one connector's builder from its own TU. Place at namespace scope.
#define REGISTER_KV_CONNECTOR(unique_tag, connector_name, builder_fn)      \
  namespace {                                                              \
  const ::vllm::v1::kv_offload::KVConnectorRegistrar                       \
      kv_connector_registrar_##unique_tag((connector_name), (builder_fn)); \
  } /* namespace */

struct OffloadingConnectorConfig {
  // Which KV cache group is offloaded. W4 offloads the full-attention group
  // (default 0); GDN/Mamba recurrent state is not block-addressed (§Risks R9).
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

// The tiered-offload connector (mirrors OffloadingConnector,
// offloading_connector.py:46 — `class OffloadingConnector(KVConnectorBase_V1,
// SupportsHMA)`). Drives ONE TieringOffloadingManager over Request::block_hashes.
// Two construction modes:
//   * BORROWING: OffloadingConnector(manager, config) — the manager/view are
//     owned elsewhere (the restart-hit correctness harness).
//   * OWNING: CreateFromConfig(ctx) — builds and owns the whole CPU+disk tiering
//     stack from KVTransferConfig extra_config; this is the factory path.
class OffloadingConnector final : public KVConnector {
 public:
  OffloadingConnector(TieringOffloadingManager& manager,
                      OffloadingConnectorConfig config);
  ~OffloadingConnector() override;

  // The factory builder (registered under "OffloadingConnector"). Reads root_dir
  // (required), num_cpu_blocks, group_idx, offload_block_tokens,
  // offload_prompt_only, eviction_policy from ctx.config->kv_connector_extra_config
  // and ctx (page_size_bytes, block_size, identity). Builds an OWNING connector.
  static std::unique_ptr<KVConnector> CreateFromConfig(
      const KVConnectorContext& ctx);

  MatchResult get_num_new_matched_tokens(const Request& request,
                                         int num_computed_tokens) override;
  void update_state_after_alloc(
      const Request& request, const std::vector<std::vector<int>>& block_ids,
      int num_external_tokens) override;
  std::vector<ConnectorLoadJob> build_connector_meta() override;

  // The offloading connector participates in a hybrid (multi-group) block
  // lifecycle (§Risks R7). It tracks its own keys, so block_ids are ignored and
  // the single-vs-all-groups split collapses (the Python-incidental part).
  bool supports_hma() const override { return true; }
  RequestFinishedResult request_finished_all_groups(
      const Request& request,
      const std::vector<std::vector<int>>& block_ids_per_group) override;

  void on_schedule_end() override;

  // Record that a request should have its computed blocks offloaded (called when
  // a request is first seen). Mirrors on_new_request registration.
  void on_new_request(const Request& request);

  // The offload keys for the request's blocks fully covered by
  // num_computed_tokens (clamped to the prompt when offload_prompt_only). These
  // are the keys the worker's GPU->CPU->disk store path writes; exposed so the
  // correctness harness can drive the store deterministically. Pure function of
  // block_hashes — does not touch the manager.
  std::vector<OffloadKey> store_keys(const Request& request,
                                     int num_computed_tokens) const;

  TieringOffloadingManager& manager() { return manager_; }
  const OffloadingConnectorConfig& config() const { return config_; }

 private:
  // Owning constructor: takes ownership of the byte view + tiering manager.
  OffloadingConnector(std::unique_ptr<PrimaryByteView> owned_view,
                      std::unique_ptr<TieringOffloadingManager> owned_manager,
                      OffloadingConnectorConfig config);

  // Stride Request::block_hashes into the offloaded-block key at index j.
  std::optional<OffloadKey> offload_key(const Request& request, int j) const;

  // Non-null only for the OWNING path; owned before manager_ so they outlive it.
  std::unique_ptr<PrimaryByteView> owned_view_;
  std::unique_ptr<TieringOffloadingManager> owned_manager_;

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
