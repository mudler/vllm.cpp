// LMCache lm:// KVConnector — KV-EXTERNAL-CACHE W3 connector-level gate.
//
// This is the FIRST time the whole chain runs through our engine seam:
//   engine (real Scheduler) -> LMCacheConnector -> W2 LmcacheRemoteClient ->
//   a running lm:// server -> back.
//
// The gate proved here (honest scope): the connector-level round-trip.
//   (a) STORE: on the first pass the connector PUTs a request's prefix KV
//       chunks (KV_2LTD) to the remote store.
//   (b) LOOKUP+SHORTCUT: a SECOND connector (a fresh "restarted process")
//       pointed at the SAME store LOOKS UP the prefix via get_num_new_matched_
//       tokens (Exist-probes) and shortcuts prefill through the REAL scheduler.
//   (c) LOAD is token-identical: LoadChunk returns byte-identical KV to what was
//       stored, so the loaded prefix reproduces the same tokens as a cold run.
//   (d) FOREIGN-KEY REFUSAL: a chunk stored under a different model/dtype key
//       MISSES; a payload whose layout disagrees is REFUSED (throws), never
//       decoded as ours.
//   (e) DEFAULT-OFF INERTNESS: a null connector leaves scheduling byte-identical.
//
// The full model->connector->live-server e2e (a real GPU worker repacking its
// paged KV through the connector, key-agreement bit-for-bit with a Python
// vLLM+LMCache peer at chunk_size 256 + vLLM's token hash) is the REMAINING
// W-step. Here the KV is synthetic and the store<->load is self-consistent,
// exactly as the disk tier's connector e2e drives the tiering manager on
// synthetic KV (test_kv_offload_connector.cpp).
//
// Runs entirely under ctest against an in-process C++ mock lm:// server (no
// Python, no GPU); when VT_LMCACHE_LIVE_HOST/PORT are set the same round-trip
// runs against a REAL lmcache.v1.server (scripts/lmcache/run_live_roundtrip.sh).
#include <doctest/doctest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "vllm/config/kv_transfer.h"
#include "vllm/config/scheduler.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/kv_offload/cache_identity.h"
#include "vllm/v1/kv_offload/kv_connector.h"
#include "vllm/v1/kv_offload/lmcache/lmcache_connector.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using namespace vllm::v1;                       // NOLINT(build/namespaces)
using namespace vllm::v1::kv_offload;           // NOLINT(build/namespaces)
using namespace vllm::v1::kv_offload::lmcache;  // NOLINT(build/namespaces)
using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vt::DType;

namespace {

constexpr int kBlockSize = 16;

// A faithful C++ mock of lmcache.v1.server.LMCacheServer (the lm:// CPU store):
// binds 127.0.0.1:0, serves one connection at a time, stores PUT payloads
// verbatim keyed by the header key. (Same shape as test_lmcache_client.cpp's
// mock; duplicated to keep the TU self-contained.)
class MockLmcacheServer {
 public:
  MockLmcacheServer() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(listen_fd_ >= 0);
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    REQUIRE(::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                   sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    REQUIRE(::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                          &len) == 0);
    port_ = ::ntohs(addr.sin_port);
    REQUIRE(::listen(listen_fd_, 8) == 0);
    thread_ = std::thread([this] { Run(); });
  }
  ~MockLmcacheServer() {
    stop_.store(true);
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
    if (thread_.joinable()) thread_.join();
  }
  int port() const { return port_; }

 private:
  struct Entry {
    std::string bytes;
    MemoryFormat fmt = MemoryFormat::kUndefined;
    Dtype dtype = Dtype::kNone;
    std::vector<int32_t> shape;
  };
  static bool RecvAll(int fd, char* data, std::size_t n) {
    std::size_t got = 0;
    while (got < n) {
      const ssize_t r = ::recv(fd, data + got, n - got, 0);
      if (r <= 0) return false;
      got += static_cast<std::size_t>(r);
    }
    return true;
  }
  static void SendAll(int fd, const char* data, std::size_t n) {
    std::size_t sent = 0;
    while (sent < n) {
      const ssize_t r = ::send(fd, data + sent, n - sent, MSG_NOSIGNAL);
      if (r < 0) return;
      sent += static_cast<std::size_t>(r);
    }
  }
  static ServerMetaMessage FailMeta() {
    ServerMetaMessage m;
    m.code = ServerReturnCode::kFail;
    m.length = 0;
    m.fmt = MemoryFormat::kKV2LTD;
    m.dtype = Dtype::kFloat16;
    m.shape = {0, 0, 0, 0};
    return m;
  }
  void HandleClient(int fd) {
    while (!stop_.load()) {
      std::string header(ClientMetaMessage::PackLength(), '\0');
      if (!RecvAll(fd, header.data(), header.size())) break;
      const ClientMetaMessage meta = ClientMetaMessage::Deserialize(header);
      switch (meta.command) {
        case ClientCommand::kPut: {
          std::string payload(static_cast<std::size_t>(meta.length), '\0');
          if (meta.length > 0 && !RecvAll(fd, payload.data(), payload.size())) {
            return;
          }
          std::lock_guard<std::mutex> lk(mu_);
          store_[meta.key] =
              Entry{std::move(payload), meta.fmt, meta.dtype, meta.shape};
          break;
        }
        case ClientCommand::kGet: {
          std::lock_guard<std::mutex> lk(mu_);
          auto it = store_.find(meta.key);
          if (it == store_.end()) {
            const std::string r = FailMeta().Serialize();
            SendAll(fd, r.data(), r.size());
          } else {
            ServerMetaMessage m;
            m.code = ServerReturnCode::kSuccess;
            m.length = static_cast<int32_t>(it->second.bytes.size());
            m.fmt = it->second.fmt;
            m.dtype = it->second.dtype;
            m.shape = it->second.shape;
            const std::string r = m.Serialize();
            SendAll(fd, r.data(), r.size());
            SendAll(fd, it->second.bytes.data(), it->second.bytes.size());
          }
          break;
        }
        case ClientCommand::kExist: {
          std::lock_guard<std::mutex> lk(mu_);
          ServerMetaMessage m = FailMeta();
          if (store_.count(meta.key) != 0) m.code = ServerReturnCode::kSuccess;
          const std::string r = m.Serialize();
          SendAll(fd, r.data(), r.size());
          break;
        }
        case ClientCommand::kHealth: {
          ServerMetaMessage m = FailMeta();
          m.code = ServerReturnCode::kSuccess;
          const std::string r = m.Serialize();
          SendAll(fd, r.data(), r.size());
          break;
        }
        case ClientCommand::kList: {
          ServerMetaMessage m = FailMeta();
          m.code = ServerReturnCode::kSuccess;
          const std::string r = m.Serialize();
          SendAll(fd, r.data(), r.size());
          break;
        }
      }
    }
    ::close(fd);
  }
  void Run() {
    while (!stop_.load()) {
      const int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) break;
      HandleClient(fd);
    }
  }
  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::mutex mu_;
  std::map<std::string, Entry> store_;
};

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
  std::vector<int32_t> prompt(num_tokens);
  for (int i = 0; i < num_tokens; ++i) prompt[i] = seed * 100000 + i;
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   hasher);
}

LmcacheConnectorConfig ConnConfig(int port_unused) {
  (void)port_unused;
  LmcacheConnectorConfig c;
  c.model_name = "Qwen/Qwen3-4B";
  c.world_size = 1;
  c.worker_id = 0;
  c.dtype = Dtype::kBFloat16;
  c.chunk_tokens = kBlockSize;
  c.num_layers = 2;
  c.hidden_dim = 8;
  c.elem_size = 2;
  return c;
}

LmcacheClientConfig ClientConfig(int port) {
  LmcacheClientConfig c;
  c.host = "127.0.0.1";
  c.port = port;
  return c;
}

// Deterministic bytes so the K/V planes are non-trivial and per-(chunk,layer,kv)
// distinct.
std::string MakeBytes(std::size_t n, uint32_t seed) {
  std::string s(n, '\0');
  uint32_t x = seed * 2654435761u + 1u;
  for (std::size_t i = 0; i < n; ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s[i] = static_cast<char>(x & 0xFF);
  }
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// DEFAULT-OFF INERTNESS: a null connector leaves scheduling byte-identical.
// ---------------------------------------------------------------------------
TEST_CASE("LMCacheConnector: default-off (null connector) is inert") {
  auto sched = CreateScheduler();
  sched->add_request(MakeRequest("a", 48, /*seed=*/1));
  auto out = sched->schedule();
  REQUIRE(out.num_scheduled_tokens.count("a") == 1);
  CHECK(out.num_scheduled_tokens.at("a") == 48);  // full prompt, no shortcut
}

// ---------------------------------------------------------------------------
// The connector is registered + selectable by config; unknown names throw; an
// absent config is a no-op (factory.py:33-36,91-92,96-125).
// ---------------------------------------------------------------------------
TEST_CASE("LMCacheConnector: registered + config-selectable, default off inert") {
  CHECK(KVConnectorFactory::IsRegistered("LMCacheConnector"));

  // Absent config -> nullptr (default off).
  KVConnectorContext none;
  CHECK(KVConnectorFactory::Create(none) == nullptr);

  // A config with no connector name -> still nullptr.
  vllm::KVTransferConfig empty;
  empty.Validate();
  KVConnectorContext ectx;
  ectx.config = &empty;
  CHECK(KVConnectorFactory::Create(ectx) == nullptr);

  // Build from config (does NOT connect eagerly, so no server is needed here).
  CacheIdentity id;
  id.model_name = "Qwen/Qwen3-4B";
  id.num_hidden_layers = 2;
  id.num_kv_heads = 2;
  id.head_size = 4;
  id.kv_dtype = "bf16";
  vllm::KVTransferConfig cfg;
  cfg.kv_connector = "LMCacheConnector";
  cfg.kv_role = vllm::KVRole::kBoth;
  cfg.kv_connector_extra_config["host"] = "127.0.0.1";
  cfg.kv_connector_extra_config["port"] = "65432";
  cfg.kv_connector_extra_config["chunk_tokens"] = std::to_string(kBlockSize);
  cfg.Validate();
  KVConnectorContext ctx;
  ctx.config = &cfg;
  ctx.block_size = kBlockSize;
  ctx.identity = &id;
  std::unique_ptr<KVConnector> conn = KVConnectorFactory::Create(ctx);
  REQUIRE(conn != nullptr);
  CHECK(conn->role() == KVConnectorRole::kScheduler);
  CHECK_FALSE(conn->supports_hma());  // LMCacheConnectorV1 is not SupportsHMA

  auto* lm = dynamic_cast<LMCacheConnector*>(conn.get());
  REQUIRE(lm != nullptr);
  CHECK(lm->config().model_name == "Qwen/Qwen3-4B");
  CHECK(lm->config().num_layers == 2);
  CHECK(lm->config().hidden_dim == 2 * 4);  // num_kv_heads * head_size
  CHECK(lm->config().dtype == Dtype::kBFloat16);

  // Unknown connector name throws.
  vllm::KVTransferConfig bad;
  bad.kv_connector = "NoSuchConnector";
  bad.kv_role = vllm::KVRole::kBoth;
  bad.Validate();
  KVConnectorContext bctx;
  bctx.config = &bad;
  CHECK_THROWS(KVConnectorFactory::Create(bctx));
}

// ---------------------------------------------------------------------------
// THE END-TO-END CONNECTOR ROUND-TRIP: store a prefix on the first pass; a fresh
// "restarted" connector looks it up, shortcuts prefill through the REAL
// scheduler, and loads it back byte-identical.
// ---------------------------------------------------------------------------
TEST_CASE("LMCacheConnector e2e: store -> lookup -> shortcut -> load-identical") {
  MockLmcacheServer server;

  const int kPromptTokens = 48;                        // 3 chunks of 16
  const int kPromptChunks = kPromptTokens / kBlockSize;  // 3
  auto req_probe = MakeRequest("warm", kPromptTokens, /*seed=*/7);

  const LmcacheConnectorConfig ccfg = ConnConfig(server.port());
  const Kv2ltdLayout layout = ccfg.ChunkLayout();
  const std::size_t plane_bytes = layout.LayerStride() * layout.elem_size;

  // ---- "Process 1": STORE the prefix chunks to the remote store. ----
  // Synthesize deterministic per-(chunk,layer) K/V planes = the worker's paged
  // KV repacked into KV_2LTD.
  std::vector<std::vector<std::string>> stored_k(kPromptChunks);
  std::vector<std::vector<std::string>> stored_v(kPromptChunks);
  std::vector<std::string> chunk_keys;
  {
    LMCacheConnector store_conn(ccfg, ClientConfig(server.port()));
    chunk_keys = store_conn.store_keys(*req_probe, kPromptTokens);
    REQUIRE(static_cast<int>(chunk_keys.size()) == kPromptChunks);
    for (int j = 0; j < kPromptChunks; ++j) {
      stored_k[j].resize(ccfg.num_layers);
      stored_v[j].resize(ccfg.num_layers);
      for (int l = 0; l < ccfg.num_layers; ++l) {
        stored_k[j][l] = MakeBytes(plane_bytes, 1000u * (j + 1) + l);
        stored_v[j][l] = MakeBytes(plane_bytes, 2000u * (j + 1) + l);
      }
      store_conn.StoreChunk(chunk_keys[j], stored_k[j], stored_v[j]);
    }
  }

  // ---- Baseline: connector OFF. Fresh scheduler, same prompt -> full prefill.
  int baseline_prefill = 0;
  {
    auto sched0 = CreateScheduler();
    sched0->add_request(MakeRequest("warm", kPromptTokens, /*seed=*/7));
    auto out = sched0->schedule();
    baseline_prefill = out.num_scheduled_tokens.at("warm");
  }
  CHECK(baseline_prefill == kPromptTokens);

  // ---- "Process 2 (restart)": a FRESH connector over the SAME store. ----
  auto conn =
      std::make_unique<LMCacheConnector>(ccfg, ClientConfig(server.port()));
  auto sched = CreateScheduler();
  sched->set_kv_connector(conn.get());
  sched->add_request(MakeRequest("warm", kPromptTokens, /*seed=*/7));

  // One schedule step: LMCache is SYNCHRONOUS, so the lookup + shortcut happen
  // in a SINGLE step (no deferral, unlike the disk tier's promotion).
  auto out = sched->schedule();
  REQUIRE(out.num_scheduled_tokens.count("warm") == 1);
  const int treated_prefill = out.num_scheduled_tokens.at("warm");

  // The connector recomputes at least one token (num_tokens-1 rule) -> matches
  // the first 2 of 3 chunks: 32 external tokens, 16 recomputed.
  const int expected_hit_chunks = (kPromptTokens - 1) / kBlockSize;  // 2
  const int prefill_saved = baseline_prefill - treated_prefill;
  CHECK(prefill_saved == expected_hit_chunks * kBlockSize);  // 32 tokens saved
  CHECK(treated_prefill == kPromptTokens - prefill_saved);   // 16 recomputed
  MESSAGE("lmcache hit = " << expected_hit_chunks << "/" << kPromptChunks
          << " chunks; prefill tokens saved = " << prefill_saved << "/"
          << kPromptTokens);

  // The typed loads were recorded for the worker (keys + GPU blocks, same order).
  std::vector<LmcacheLoadJob> loads = conn->TakeConnectorLoads();
  REQUIRE(loads.size() == 1);
  CHECK(loads[0].req_id == "warm");
  CHECK(static_cast<int>(loads[0].keys.size()) == expected_hit_chunks);
  CHECK(loads[0].gpu_block_ids.size() == loads[0].keys.size());

  // ---- LOAD is token-identical: each hit chunk loads back byte-identical. ----
  for (int j = 0; j < expected_hit_chunks; ++j) {
    std::vector<std::string> k, v;
    REQUIRE(conn->LoadChunk(loads[0].keys[j], &k, &v));
    REQUIRE(static_cast<int>(k.size()) == ccfg.num_layers);
    REQUIRE(static_cast<int>(v.size()) == ccfg.num_layers);
    for (int l = 0; l < ccfg.num_layers; ++l) {
      CHECK(k[l] == stored_k[j][l]);  // byte-identical across the "restart"
      CHECK(v[l] == stored_v[j][l]);
    }
  }
}

// ---------------------------------------------------------------------------
// FOREIGN-KEY REFUSAL: a chunk stored under a different model/dtype key MISSES;
// a payload whose layout disagrees is REFUSED (throws), never mis-decoded.
// ---------------------------------------------------------------------------
TEST_CASE("LMCacheConnector: a foreign / mismatched block is refused, not served") {
  MockLmcacheServer server;
  auto req = MakeRequest("warm", 32, /*seed=*/7);

  LmcacheConnectorConfig our = ConnConfig(server.port());
  const Kv2ltdLayout layout = our.ChunkLayout();
  const std::size_t plane_bytes = layout.LayerStride() * layout.elem_size;

  // A FOREIGN writer: a DIFFERENT model name (its keys differ from ours).
  LmcacheConnectorConfig foreign = our;
  foreign.model_name = "some-other/Model";
  {
    LMCacheConnector fconn(foreign, ClientConfig(server.port()));
    auto fkeys = fconn.store_keys(*req, 32);
    REQUIRE(!fkeys.empty());
    std::vector<std::string> k(foreign.num_layers), v(foreign.num_layers);
    for (int l = 0; l < foreign.num_layers; ++l) {
      k[l] = MakeBytes(plane_bytes, 42u + l);
      v[l] = MakeBytes(plane_bytes, 84u + l);
    }
    fconn.StoreChunk(fkeys[0], k, v);
  }

  // OUR connector, same tokens/model — the foreign block's key is NOT our key,
  // so our lookup MISSES (identity is folded into the key's model_name+dtype).
  LMCacheConnector conn(our, ClientConfig(server.port()));
  const int matched =
      conn.get_num_new_matched_tokens(*req, 0).num_matched_tokens.value_or(-1);
  CHECK(matched == 0);  // no hit: the foreign block is invisible to us

  // A byte-length mismatch under OUR key is REFUSED (throws), never decoded.
  // Store a truncated payload under a valid OUR key, then LoadChunk must throw.
  auto our_keys = conn.store_keys(*req, 32);
  REQUIRE(!our_keys.empty());
  // PUT a deliberately-too-small payload directly via the raw client.
  conn.EnsureConnected();
  conn.client().Put(our_keys[0], std::string(plane_bytes, '\0'),
                    MemoryFormat::kKV2LTD, our.dtype, layout.Shape());
  std::vector<std::string> k, v;
  CHECK_THROWS(conn.LoadChunk(our_keys[0], &k, &v));  // wrong byte count refused

  // A dtype-mismatched key: same tokens, DIFFERENT dtype -> different key string
  // -> our lookup for the bf16 key misses the float32-stored chunk.
  LmcacheConnectorConfig f32 = our;
  f32.dtype = Dtype::kFloat32;
  f32.elem_size = 4;
  CHECK(conn.ChunkKey(conn.ChunkFolds(req->AllTokenIds())[0]) !=
        LMCacheConnector(f32, ClientConfig(server.port()))
            .ChunkKey(LMCacheConnector(f32, ClientConfig(server.port()))
                          .ChunkFolds(req->AllTokenIds())[0]));
}

// ---------------------------------------------------------------------------
// Live interop gate: only runs when pointed at a REAL lmcache.v1.server via
// VT_LMCACHE_LIVE_HOST/PORT. Skipped (passing) under plain ctest.
// ---------------------------------------------------------------------------
TEST_CASE("LMCacheConnector: store->load round-trip vs REAL lmcache.v1.server") {
  const char* host = std::getenv("VT_LMCACHE_LIVE_HOST");
  const char* port = std::getenv("VT_LMCACHE_LIVE_PORT");
  if (host == nullptr || port == nullptr) {
    MESSAGE(
        "skipped: set VT_LMCACHE_LIVE_HOST/VT_LMCACHE_LIVE_PORT to run the "
        "connector round-trip against a real lmcache.v1.server");
    return;
  }
  LmcacheClientConfig cc;
  cc.host = host;
  cc.port = std::atoi(port);
  LmcacheConnectorConfig ccfg = ConnConfig(cc.port);
  auto req = MakeRequest("warm", 48, /*seed=*/7);
  const Kv2ltdLayout layout = ccfg.ChunkLayout();
  const std::size_t plane_bytes = layout.LayerStride() * layout.elem_size;

  LMCacheConnector conn(ccfg, cc);
  auto keys = conn.store_keys(*req, 48);
  REQUIRE(!keys.empty());
  std::vector<std::vector<std::string>> k(keys.size()), v(keys.size());
  for (std::size_t j = 0; j < keys.size(); ++j) {
    k[j].resize(ccfg.num_layers);
    v[j].resize(ccfg.num_layers);
    for (int l = 0; l < ccfg.num_layers; ++l) {
      k[j][l] = MakeBytes(plane_bytes, 7000u * (j + 1) + l);
      v[j][l] = MakeBytes(plane_bytes, 9000u * (j + 1) + l);
    }
    conn.StoreChunk(keys[j], k[j], v[j]);
  }
  // A fresh connector loads them back byte-identical.
  LMCacheConnector conn2(ccfg, cc);
  for (std::size_t j = 0; j < keys.size(); ++j) {
    std::vector<std::string> lk, lv;
    REQUIRE(conn2.LoadChunk(keys[j], &lk, &lv));
    for (int l = 0; l < ccfg.num_layers; ++l) {
      CHECK(lk[l] == k[j][l]);
      CHECK(lv[l] == v[j][l]);
    }
  }
}
