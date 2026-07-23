// Ported from: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py
//               (LMCacheConnectorV1) @ e24d1b24 — the vLLM-side calling
//               convention (synchronous (n, False); drops `blocks`; resets on
//               build_connector_meta; pass-through worker hooks), AND the W5
//               abstract base (kv_connector.h) + the OffloadingConnector shape
//               (kv_connector.cpp:73-323) as the in-repo reference connector.
// See include/vllm/v1/kv_offload/lmcache/lmcache_connector.h for scope + the
// recorded chunk-keying deviation (chunk_tokens == block_size; key-agreement
// with a Python peer is the remaining full-model e2e step).
#include "vllm/v1/kv_offload/lmcache/lmcache_connector.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "vllm/v1/request.h"

namespace vllm::v1::kv_offload::lmcache {

LMCacheConnector::LMCacheConnector(LmcacheConnectorConfig config,
                                   LmcacheClientConfig client_config)
    : KVConnector(KVConnectorRole::kScheduler),
      config_(std::move(config)),
      client_(std::make_unique<LmcacheRemoteClient>(std::move(client_config))),
      hasher_(config_.chunk_tokens) {
  if (config_.chunk_tokens <= 0) {
    throw std::runtime_error("LMCacheConnector: chunk_tokens must be > 0");
  }
  // W4: the peer-agreeing key derivation (vLLM sha256_cbor over chunk_tokens),
  // byte-identical to a real Python vLLM+LMCache peer (chunked_token_database.h).
  if (config_.key_mode == LmcacheConnectorConfig::KeyMode::kVllmSha256Cbor) {
    peer_db_ = std::make_unique<ChunkedTokenDatabase>(
        config_.chunk_tokens,
        ChunkedTokenDatabase::NoneHashFromSeed(config_.none_hash_seed),
        config_.save_unfull_chunk, PreCachingHash::kSha256Cbor);
  }
}

LMCacheConnector::~LMCacheConnector() = default;

void LMCacheConnector::EnsureConnected() {
  if (!client_->connected()) {
    client_->Connect();
  }
}

std::vector<uint64_t> LMCacheConnector::ChunkFolds(
    const std::vector<int32_t>& token_ids) const {
  // W4 peer-agreeing path: the vLLM sha256_cbor ChunkedTokenDatabase produces
  // the exact folded uint64 chunk hashes a real Python peer keys on.
  if (peer_db_ != nullptr) {
    std::vector<uint64_t> folds;
    for (const ChunkedTokenDatabase::Entry& e : peer_db_->ProcessTokens(token_ids)) {
      folds.push_back(e.chunk_hash);
    }
    return folds;
  }
  // W3 self-consistent path: the rolling blake3 wants unsigned token ids
  // (struct.pack ">I").
  std::vector<uint32_t> toks(token_ids.size());
  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    toks[i] = static_cast<uint32_t>(token_ids[i]);
  }
  const std::vector<std::string> digests = hasher_.ComputeChunkHashes(toks);
  std::vector<uint64_t> folds;
  folds.reserve(digests.size());
  for (const std::string& d : digests) {
    folds.push_back(TokenHasher::FoldToUint64(d));
  }
  return folds;
}

std::string LMCacheConnector::ChunkKey(uint64_t chunk_fold) const {
  CacheEngineKey key;
  key.model_name = config_.model_name;
  key.world_size = config_.world_size;
  key.worker_id = config_.worker_id;
  key.chunk_hash = chunk_fold;
  key.dtype = config_.dtype;
  return key.ToString();
}

// lmcache_connector.py:230-259 — ALWAYS (n, False). Compute the request's
// rolling chunk hashes, Exist-probe the remote store for the longest cached
// prefix beyond the local prefix, record the hit keys.
MatchResult LMCacheConnector::get_num_new_matched_tokens(
    const Request& request, int num_computed_tokens) {
  const int chunk = config_.chunk_tokens;
  if (chunk <= 0) return MatchResult{0, false};
  EnsureConnected();

  const std::vector<int32_t> tokens = request.AllTokenIds();
  const std::vector<uint64_t> folds = ChunkFolds(tokens);

  // The first chunk NOT already covered by the local prefix cache.
  const int start_chunk = num_computed_tokens / chunk;

  // Recompute AT LEAST ONE token (mirror the num_tokens-1 rule / OffloadingC.
  // get_num_new_matched_tokens: keep the external match to whole chunks so total
  // computed <= NumTokens()-1, guaranteeing the scheduler has a token to run).
  const int max_ext_tokens =
      std::max(0, (request.NumTokens() - 1) - num_computed_tokens);
  const int max_hit_chunks = max_ext_tokens / chunk;

  std::vector<std::string> hit;
  for (int j = start_chunk;
       static_cast<int>(hit.size()) < max_hit_chunks &&
       j < static_cast<int>(folds.size());
       ++j) {
    const std::string key = ChunkKey(folds[static_cast<std::size_t>(j)]);
    if (!client_->Exist(key)) break;  // the consecutive prefix run ends
    hit.push_back(key);
  }

  hit_keys_[request.request_id] = hit;
  const int num_hit_tokens = static_cast<int>(hit.size()) * chunk;
  // LMCache is synchronous: the async flag is always false.
  return MatchResult{num_hit_tokens, false};
}

// lmcache_connector.py:261-268 (drops `blocks` upstream; we record the GPU
// blocks so the worker can drive LoadChunk). ext==0 early-out => 2nd call no-op.
void LMCacheConnector::update_state_after_alloc(
    const Request& request, const std::vector<std::vector<int>>& block_ids,
    int num_external_tokens) {
  if (num_external_tokens == 0) return;

  auto it = hit_keys_.find(request.request_id);
  if (it == hit_keys_.end() || it->second.empty()) return;
  const std::vector<std::string>& keys = it->second;

  // The external prefix occupies the GPU blocks immediately AFTER the locally
  // computed ones (group 0). num_external_tokens == keys.size()*chunk_tokens.
  if (block_ids.empty()) return;
  const std::vector<int>& group_blocks = block_ids[0];
  const int chunk = config_.chunk_tokens;
  const int n_local = request.num_computed_tokens / chunk;
  const int n_ext = static_cast<int>(keys.size());
  if (n_local + n_ext > static_cast<int>(group_blocks.size())) return;

  LmcacheLoadJob job;
  job.req_id = request.request_id;
  job.keys = keys;
  job.gpu_block_ids.assign(group_blocks.begin() + n_local,
                           group_blocks.begin() + n_local + n_ext);
  batch_loads_.push_back(std::move(job));
}

// lmcache_connector.py:270-286 — RESET per-step state. The scheduler discards
// the return (scheduler.cpp:456); the worker drains the typed loads via
// TakeConnectorLoads. Returning the empty base form keeps the ABI honest.
std::vector<ConnectorLoadJob> LMCacheConnector::build_connector_meta() {
  return {};
}

RequestFinishedResult LMCacheConnector::request_finished(
    const Request& request, const std::vector<int>& /*block_ids*/) {
  // Synchronous store: no deferred free (delay_free stays false).
  hit_keys_.erase(request.request_id);
  return RequestFinishedResult{};
}

std::vector<std::string> LMCacheConnector::store_keys(
    const Request& request, int num_computed_tokens) const {
  const int chunk = config_.chunk_tokens;
  if (chunk <= 0) return {};
  int limit_tokens = num_computed_tokens;
  if (config_.offload_prompt_only) {
    limit_tokens = std::min(limit_tokens, request.num_prompt_tokens);
  }
  const int num_full_chunks = limit_tokens / chunk;
  const std::vector<uint64_t> folds = ChunkFolds(request.AllTokenIds());
  std::vector<std::string> keys;
  keys.reserve(static_cast<std::size_t>(std::max(0, num_full_chunks)));
  for (int j = 0; j < num_full_chunks && j < static_cast<int>(folds.size());
       ++j) {
    keys.push_back(ChunkKey(folds[static_cast<std::size_t>(j)]));
  }
  return keys;
}

void LMCacheConnector::StoreChunk(const std::string& key,
                                  const std::vector<std::string>& k_planes,
                                  const std::vector<std::string>& v_planes) {
  EnsureConnected();
  client_->PutKv2ltd(key, config_.ChunkLayout(), k_planes, v_planes,
                     config_.dtype);
}

bool LMCacheConnector::LoadChunk(const std::string& key,
                                 std::vector<std::string>* k_planes,
                                 std::vector<std::string>* v_planes) {
  EnsureConnected();
  // GetKv2ltd refuses (throws) a payload whose byte count / shape disagrees with
  // our chunk layout — a foreign block is never decoded as ours (gate 5).
  return client_->GetKv2ltd(key, config_.ChunkLayout(), k_planes, v_planes);
}

std::vector<LmcacheLoadJob> LMCacheConnector::TakeConnectorLoads() {
  std::vector<LmcacheLoadJob> out;
  out.swap(batch_loads_);
  return out;
}

namespace {
int64_t extra_int(const KVTransferConfig& cfg, const std::string& key,
                  int64_t dflt) {
  const std::string v = cfg.get_from_extra_config(key, "");
  if (v.empty()) return dflt;
  return static_cast<int64_t>(std::stoll(v));
}
}  // namespace

std::unique_ptr<KVConnector> LMCacheConnector::CreateFromConfig(
    const KVConnectorContext& ctx) {
  if (ctx.config == nullptr) {
    throw std::runtime_error("LMCacheConnector: null KVTransferConfig");
  }
  const KVTransferConfig& cfg = *ctx.config;

  LmcacheClientConfig client_cfg = LmcacheClientConfig::FromEnv();
  const std::string host = cfg.get_from_extra_config("host", "");
  if (!host.empty()) client_cfg.host = host;
  const std::string port = cfg.get_from_extra_config("port", "");
  if (!port.empty()) client_cfg.port = static_cast<int>(std::stoll(port));
  // hash_algo selects the KEY derivation. "vllm" (or the explicit
  // "sha256_cbor") is the W4 PEER-AGREEING path: vLLM's own sha256_cbor over
  // chunk_size 256, byte-identical to a real Python vLLM+LMCache peer. "blake3"
  // is W3's self-consistent path. Default stays blake3 for back-compat.
  const std::string algo = cfg.get_from_extra_config("hash_algo", "");
  LmcacheConnectorConfig conn_cfg;
  if (algo == "blake3") {
    client_cfg.hash_algo = LmcacheClientConfig::HashAlgo::kBlake3;
    conn_cfg.key_mode = LmcacheConnectorConfig::KeyMode::kBlake3BlockAligned;
  } else if (algo == "vllm" || algo == "sha256_cbor") {
    client_cfg.hash_algo = LmcacheClientConfig::HashAlgo::kVllm;
    conn_cfg.key_mode = LmcacheConnectorConfig::KeyMode::kVllmSha256Cbor;
  } else if (!algo.empty()) {
    throw std::runtime_error(
        "LMCacheConnector: hash_algo must be 'blake3', 'vllm', or 'sha256_cbor'");
  }

  // Peer mode chunks at LMCache's default 256 unless overridden; blake3 mode
  // stays block-aligned (chunk_tokens == block_size).
  const bool peer =
      conn_cfg.key_mode == LmcacheConnectorConfig::KeyMode::kVllmSha256Cbor;
  conn_cfg.chunk_tokens = static_cast<int>(
      extra_int(cfg, "chunk_tokens", peer ? 256 : ctx.block_size));
  conn_cfg.none_hash_seed = cfg.get_from_extra_config("none_hash_seed", "0");
  conn_cfg.save_unfull_chunk =
      cfg.get_from_extra_config("save_unfull_chunk", "false") == "true";
  conn_cfg.world_size = extra_int(cfg, "world_size", 1);
  conn_cfg.worker_id = extra_int(cfg, "worker_id", 0);
  conn_cfg.offload_prompt_only =
      cfg.get_from_extra_config("offload_prompt_only", "true") != "false";

  // Identity + KV_2LTD geometry from the CacheIdentity (as the disk connector
  // takes it from ctx.identity). num_layers / hidden_dim / elem_size / dtype
  // come from the model's KV cache spec; a mismatch makes the key MISS (the
  // identity is folded into the key's model_name + dtype).
  if (ctx.identity != nullptr) {
    conn_cfg.model_name = ctx.identity->model_name;
    conn_cfg.num_layers = ctx.identity->num_hidden_layers;
    conn_cfg.hidden_dim = ctx.identity->num_kv_heads * ctx.identity->head_size;
    const std::string& kd = ctx.identity->kv_dtype;
    if (kd == "bf16" || kd == "bfloat16") {
      conn_cfg.dtype = Dtype::kBFloat16;
      conn_cfg.elem_size = 2;
    } else if (kd == "f16" || kd == "half" || kd == "float16") {
      conn_cfg.dtype = Dtype::kFloat16;
      conn_cfg.elem_size = 2;
    } else if (kd == "f32" || kd == "float" || kd == "float32") {
      conn_cfg.dtype = Dtype::kFloat32;
      conn_cfg.elem_size = 4;
    }
  }
  // Explicit overrides win (for a GPU-free geometry the test controls).
  conn_cfg.num_layers =
      static_cast<int>(extra_int(cfg, "num_layers", conn_cfg.num_layers));
  conn_cfg.hidden_dim =
      static_cast<int>(extra_int(cfg, "hidden_dim", conn_cfg.hidden_dim));
  const std::string model = cfg.get_from_extra_config("model_name", "");
  if (!model.empty()) conn_cfg.model_name = model;

  return std::unique_ptr<KVConnector>(
      new LMCacheConnector(std::move(conn_cfg), std::move(client_cfg)));
}

}  // namespace vllm::v1::kv_offload::lmcache

// Self-register the LMCache lm:// connector (mirrors factory.py registering
// "LMCacheConnectorV1"). Compile-time registration replaces vLLM's dynamic
// importlib path (recorded W5 deviation). Selected by
// KVTransferConfig{kv_connector = "LMCacheConnector"}.
REGISTER_KV_CONNECTOR(
    lmcache, "LMCacheConnector",
    &::vllm::v1::kv_offload::lmcache::LMCacheConnector::CreateFromConfig)
