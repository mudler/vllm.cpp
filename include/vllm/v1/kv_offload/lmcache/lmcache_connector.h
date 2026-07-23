// Ported from: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py
//              (LMCacheConnectorV1) @ e24d1b24 — the vLLM-side calling
//              convention the connector honours:
//   - get_num_new_matched_tokens ALWAYS returns (n, False): LMCache is
//     SYNCHRONOUS, never async-loads (lmcache_connector.py:230-259).
//   - update_state_after_alloc DROPS the `blocks` argument — LMCache computes
//     the flat slot = block_id*block_size+offset itself (lmcache_connector.py:
//     261-268 / vllm_v1_adapter.py:368-376).
//   - build_connector_meta RESETS the per-step state (lmcache_connector.py:
//     270-286).
//   - every worker hook is pass-through (lmcache_connector.py:120-210).
//
// AND the W5 base: this is a NEW KVConnector subclass over the landed abstract
// ABI (include/vllm/v1/kv_offload/kv_connector.h @ KV-CONNECTORS W5), exactly as
// OffloadingConnector is (kv_connector.cpp:73). It DRIVES the landed W2
// LmcacheRemoteClient (remote_client.h) over the lm:// MODE-1 wire, using the W1
// TokenHasher (blake3 rolling chunk hash) + CacheEngineKey codec + the KV_2LTD
// repack (memory_format.h).
//
// WHAT WE ARE (and are not) IN THIS INCREMENT.
//   * SCHEDULER side: real. get_num_new_matched_tokens computes the request's
//     rolling chunk hashes, builds the CacheEngineKey per chunk and Exist-probes
//     the REMOTE lm:// store for the longest cached prefix beyond the local
//     prefix, so a repeated/restarted prefix HIT shortcuts prefill through the
//     real scheduler. Synchronous -> (n, false).
//   * WORKER side: StoreChunk (PUT the KV_2LTD chunk on the first pass) and
//     LoadChunk (GET + unpack a hit chunk, byte-identical). These are the store
//     and load the worker's GPU->host repack drives; the connector-level
//     round-trip test drives them on synthetic KV, exactly as the disk tier's
//     e2e drives the tiering manager on synthetic KV.
//
// DEVIATION vs upstream chunk keying (recorded). Upstream LMCache keys on a
// rolling blake3 with chunk_size 256 AND injects multimodal hashes into the
// token tensor (vllm_v1_adapter.py:344-352). Our connector keys on the SAME
// rolling-blake3 family (token_hasher.h) with chunk_tokens == block_size so the
// scheduler stays block-aligned, over the plain token ids. That makes OUR
// store<->lookup<->load self-consistent (the connector-level round-trip gate).
// AGREEING BIT-FOR-BIT with a Python vLLM+LMCache PEER additionally requires
// chunk_size 256 and vLLM's own token hash + mm-hash injection — the
// key-agreement gate (spec gate 3), which is the REMAINING full-model e2e step.
#ifndef VLLM_V1_KV_OFFLOAD_LMCACHE_LMCACHE_CONNECTOR_H_
#define VLLM_V1_KV_OFFLOAD_LMCACHE_LMCACHE_CONNECTOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/v1/kv_offload/kv_connector.h"
#include "vllm/v1/kv_offload/lmcache/cache_engine_key.h"
#include "vllm/v1/kv_offload/lmcache/memory_format.h"
#include "vllm/v1/kv_offload/lmcache/remote_client.h"
#include "vllm/v1/kv_offload/lmcache/remote_protocol.h"
#include "vllm/v1/kv_offload/lmcache/token_hasher.h"

namespace vllm::v1 {
struct Request;
}  // namespace vllm::v1

namespace vllm::v1::kv_offload::lmcache {

// The identity + geometry a connector needs to build CacheEngineKeys and repack
// KV chunks. Read from KVConnectorContext (CacheIdentity + geometry) and
// kv_connector_extra_config in CreateFromConfig; also constructed directly by
// the connector-level round-trip test.
struct LmcacheConnectorConfig {
  // CacheEngineKey identity fields (utils.py:449-457):
  // model_name@world_size@worker_id@chunk_hash_hex@dtype.
  std::string model_name = "vllm.cpp";
  int64_t world_size = 1;
  int64_t worker_id = 0;
  // The dtype STRING that goes in the key (independent of the payload dtype in
  // the header). Also the payload element dtype for the KV_2LTD chunk.
  Dtype dtype = Dtype::kBFloat16;

  // Tokens per LMCache chunk. Kept == the scheduler block_size so external
  // matches stay block-aligned (see the header deviation note). The rolling
  // blake3 hashes complete chunks of exactly this many tokens.
  int chunk_tokens = 16;

  // KV_2LTD chunk geometry: [2, num_layers, chunk_tokens, hidden_dim].
  int num_layers = 0;
  int hidden_dim = 0;             // num_kv_heads * head_size
  std::size_t elem_size = 2;      // bytes per KV element (dtype width)

  // offload_prompt_only (kv_offloading_usage.md:64-82): only prompt blocks are
  // stored, never generated tokens.
  bool offload_prompt_only = true;

  // The per-chunk KV_2LTD layout implied by this config.
  Kv2ltdLayout ChunkLayout() const {
    Kv2ltdLayout l;
    l.num_layers = num_layers;
    l.num_tokens = chunk_tokens;
    l.hidden_dim = hidden_dim;
    l.elem_size = elem_size;
    return l;
  }
};

// A recorded load: the remote chunk keys for a request's external prefix and the
// GPU blocks they load into (same order). The connector tracks its own typed
// loads (LMCache keys are strings, not the disk tier's OffloadKey), so
// build_connector_meta returns the empty base form (the scheduler discards it,
// scheduler.cpp:456) and the worker drains these via TakeConnectorLoads.
struct LmcacheLoadJob {
  std::string req_id;
  std::vector<std::string> keys;   // CacheEngineKey.to_string() per chunk
  std::vector<int> gpu_block_ids;  // one GPU block per chunk, same order
};

// The LMCache lm:// (MODE-1) KV connector. A KVConnector subclass driving the
// W2 LmcacheRemoteClient. Synchronous (async flag always false), opt-in,
// default-off inert (selected only by KVTransferConfig{kv_connector =
// "LMCacheConnector"}; a null connector is unchanged behaviour).
class LMCacheConnector final : public KVConnector {
 public:
  LMCacheConnector(LmcacheConnectorConfig config,
                   LmcacheClientConfig client_config);
  ~LMCacheConnector() override;

  // factory.py builder registered under "LMCacheConnector". Reads host/port/
  // hash_algo/model_name/world_size/worker_id/chunk_tokens from
  // ctx.config->kv_connector_extra_config, and the KV_2LTD geometry
  // (num_layers/hidden_dim/elem_size) from the CacheIdentity + block_size. Does
  // NOT connect eagerly (a freshly-spawned server may not be listening yet); the
  // first scheduler/worker op connects with the client's retry/backoff.
  static std::unique_ptr<KVConnector> CreateFromConfig(
      const KVConnectorContext& ctx);

  // ---- scheduler-side ABI ---------------------------------------------------

  // lmcache_connector.py:230-259 (always (n, False)). Computes the request's
  // rolling chunk hashes, Exist-probes the remote store for the longest cached
  // prefix beyond num_computed_tokens, records the hit keys, returns the token
  // count. Synchronous -> load_async false.
  MatchResult get_num_new_matched_tokens(const Request& request,
                                         int num_computed_tokens) override;

  // lmcache_connector.py:261-268 (drops `blocks` upstream; we record the GPU
  // blocks the external prefix loads into, our house convention, so the worker
  // can drive LoadChunk). ext==0 early-out makes the 2nd call a no-op.
  void update_state_after_alloc(
      const Request& request, const std::vector<std::vector<int>>& block_ids,
      int num_external_tokens) override;

  // lmcache_connector.py:270-286 (RESET per-step state). Returns the empty base
  // form; the worker drains the typed loads via TakeConnectorLoads.
  std::vector<ConnectorLoadJob> build_connector_meta() override;

  // LMCacheConnectorV1 is NOT SupportsHMA (factory guard asserts one KV cache
  // group for it). Synchronous store => no deferred free.
  RequestFinishedResult request_finished(
      const Request& request, const std::vector<int>& block_ids) override;

  // ---- worker-side store/load (driven by the worker / round-trip harness) ----

  // Open the TCP connection if not already open (client retry/backoff). Safe to
  // call repeatedly.
  void EnsureConnected();
  LmcacheRemoteClient& client() { return *client_; }
  const LmcacheConnectorConfig& config() const { return config_; }

  // Rolling blake3 chunk hashes (folded to uint64) for every COMPLETE chunk of
  // `chunk_tokens` in `token_ids` (token_hasher.h ComputeChunkHashes +
  // FoldToUint64). Bit-exact with LMCache's TokenHasher family.
  std::vector<uint64_t> ChunkFolds(const std::vector<int32_t>& token_ids) const;

  // CacheEngineKey string for a folded chunk hash (utils.py:449-457).
  std::string ChunkKey(uint64_t chunk_fold) const;

  // The chunk key strings for the request's prompt-prefix chunks fully covered
  // by num_computed_tokens (clamped to the prompt when offload_prompt_only).
  // Pure function of the token ids; used by the worker store path.
  std::vector<std::string> store_keys(const Request& request,
                                      int num_computed_tokens) const;

  // Worker STORE: PUT the KV_2LTD chunk (k_planes[l]/v_planes[l] are the
  // contiguous [chunk_tokens, hidden_dim] bytes of layer l) under `key`.
  void StoreChunk(const std::string& key,
                  const std::vector<std::string>& k_planes,
                  const std::vector<std::string>& v_planes);

  // Worker LOAD: GET + unpack a hit chunk into per-layer K/V planes. Returns
  // false if absent (a genuine miss). THROWS if the returned payload's byte
  // count / shape disagrees with our chunk layout — a FOREIGN block (wrong
  // model/dtype/layout) is REFUSED, never decoded as ours (identity safety,
  // spec gate 5; the refusal lives in LmcacheRemoteClient::GetKv2ltd).
  bool LoadChunk(const std::string& key, std::vector<std::string>* k_planes,
                 std::vector<std::string>* v_planes);

  // Drain the typed per-step loads recorded by update_state_after_alloc.
  std::vector<LmcacheLoadJob> TakeConnectorLoads();

 private:
  LmcacheConnectorConfig config_;
  std::unique_ptr<LmcacheRemoteClient> client_;
  TokenHasher hasher_;

  // req_id -> the chunk key strings the last get_num_new_matched_tokens hit
  // (used by update_state_after_alloc; idempotent overwrite).
  std::unordered_map<std::string, std::vector<std::string>> hit_keys_;
  // This step's accumulated typed loads, drained by TakeConnectorLoads.
  std::vector<LmcacheLoadJob> batch_loads_;
};

}  // namespace vllm::v1::kv_offload::lmcache

#endif  // VLLM_V1_KV_OFFLOAD_LMCACHE_LMCACHE_CONNECTOR_H_
