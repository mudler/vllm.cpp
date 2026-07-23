// Ported from: lmcache/v1/storage_backend/connector/lm_connector.py:28-177
//              (LMCServerConnector, the lm:// client loop) and the matching
//              server loop lmcache/v1/server/__main__.py:34-135 @ LMCache
//              8570aad.
//
// A blocking POSIX-socket client that speaks the LMCache MODE-1 (lm:// remote
// store) wire: for each operation it sends the 186-byte ClientMetaMessage
// header (plus the raw KV payload for PUT), then reads the 36-byte
// ServerMetaMessage header (plus the raw payload for GET).  The wire codec
// itself lives in remote_protocol.h (W1, byte-exact vs the real Python codec);
// this file is only the transport + framing loop.
//
// TCP IS A STREAM: recv()/send() may transfer fewer bytes than requested, so
// SendAll/RecvAll loop until the full framed length is moved (mirrors the
// server's receive_all, __main__.py:34-41, and the client's recv_into loop,
// lm_connector.py:75-79).  A short read (peer closed mid-frame) is an error,
// exactly as the server treats it (__main__.py:39 `return None` -> break).
//
// ENDIANNESS: the lm:// struct is native-endian (protocol.py:238, no byte-order
// prefix); remote_protocol.cpp encodes little-endian explicitly, so client and
// server must share byte order (true for every realistic co-located/LAN lm://
// deployment; see the remote_protocol.h R3 note).
#ifndef VLLM_V1_KV_OFFLOAD_LMCACHE_REMOTE_CLIENT_H_
#define VLLM_V1_KV_OFFLOAD_LMCACHE_REMOTE_CLIENT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/v1/kv_offload/lmcache/cache_engine_key.h"
#include "vllm/v1/kv_offload/lmcache/memory_format.h"
#include "vllm/v1/kv_offload/lmcache/remote_protocol.h"

namespace vllm::v1::kv_offload::lmcache {

// Host/port + the hash-algorithm selection.  The algorithm must match the
// peer's `pre_caching_hash_algorithm` (LMCache config): the default LMCache
// TokenDatabase keys on vLLM's own token hash unless configured `blake3`.  W1
// provides both the blake3 rolling hash (TokenHasher) and the uint64 fold; the
// selection is carried here so W3's KVConnector can build keys the server will
// find.  For the raw PUT/GET/EXIST transport the algorithm is irrelevant (the
// caller supplies the key string), but the field is part of the client config.
struct LmcacheClientConfig {
  enum class HashAlgo {
    kBlake3,  // LMCache's own rolling blake3 chunk hash (pre_caching_hash=blake3)
    kVllm,    // vLLM's default token hash (TokenDatabase default)
  };

  std::string host = "127.0.0.1";
  int port = 65432;
  HashAlgo hash_algo = HashAlgo::kBlake3;
  // Connect() retries this many times with a short backoff before giving up
  // (a freshly-spawned server may not have called listen() yet).
  int connect_retries = 20;

  // Reads VT_LMCACHE_HOST, VT_LMCACHE_PORT, VT_LMCACHE_HASH_ALGO
  // ("blake3"|"vllm") from the environment; unset fields keep the defaults.
  static LmcacheClientConfig FromEnv();
};

// A single blocking connection to an lmcache.v1.server LMCacheServer.
// Non-copyable, non-movable (owns a socket fd).  Not thread-safe: the upstream
// connector serializes ops behind an async lock (lm_connector.py:54,86,123),
// and so must callers of a single client instance.
class LmcacheRemoteClient {
 public:
  explicit LmcacheRemoteClient(LmcacheClientConfig config = {});
  ~LmcacheRemoteClient();

  LmcacheRemoteClient(const LmcacheRemoteClient&) = delete;
  LmcacheRemoteClient& operator=(const LmcacheRemoteClient&) = delete;

  // Open the TCP connection (lm_connector.py:47-48).  Throws std::runtime_error
  // after config.connect_retries failed attempts.
  void Connect();
  bool connected() const { return fd_ >= 0; }
  // Close the socket (lm_connector.py:173-176).  Idempotent.
  void Close();

  const LmcacheClientConfig& config() const { return config_; }

  // Result of a successful GET: the raw stored bytes plus the self-describing
  // header the server echoed back (fmt/dtype/shape).
  struct GetResult {
    std::string bytes;
    MemoryFormat fmt = MemoryFormat::kUndefined;
    Dtype dtype = Dtype::kNone;
    std::vector<int32_t> shape;
  };

  // PUT(key, kv_bytes): send the header (command=PUT, length=kv_bytes.size(),
  // fmt/dtype/shape describing the payload) then the raw payload
  // (lm_connector.py:111-136).  The server stores the bytes verbatim under the
  // key (__main__.py:53-57).  No server reply for PUT.
  void Put(const std::string& key, std::string_view kv_bytes, MemoryFormat fmt,
           Dtype dtype, const std::vector<int32_t>& shape);

  // GET(key): send the header, read the 36-byte reply; on SUCCESS read the
  // reply.length payload bytes; on FAIL return nullopt ("absent", never
  // garbage) (lm_connector.py:140-166, server __main__.py:64-94).
  std::optional<GetResult> Get(const std::string& key);

  // EXIST(key): SUCCESS -> true, FAIL -> false (lm_connector.py:83-100,
  // server __main__.py:96-111).
  bool Exist(const std::string& key);

  // HEALTH: SUCCESS -> true (server __main__.py:112-122).
  bool Health();

  // LIST: the intended protocol is a SUCCESS header carrying `length` bytes of
  // newline-joined keys (server __main__.py:124-131 — currently COMMENTED OUT
  // upstream, so a real lmcache.v1.server never replies and this call would
  // block; our in-repo mock server implements it).  Returns the split key list.
  std::vector<std::string> List();

  // KV_2LTD convenience: repack per-layer K/V planes into a [2,L,T,D] chunk
  // (memory_format.h PackKv2ltd) and PUT it, so what we ship is exactly the
  // MemoryFormat::KV_2LTD layout LMCache stores.  `k_planes[l]`/`v_planes[l]`
  // are the contiguous [num_tokens, hidden_dim] bytes of layer l.
  void PutKv2ltd(const std::string& key, const Kv2ltdLayout& layout,
                 const std::vector<std::string>& k_planes,
                 const std::vector<std::string>& v_planes, Dtype dtype);

  // Inverse: GET a KV_2LTD chunk and UnpackKv2ltd it back to per-layer K/V
  // planes.  Returns false if the key is absent.  Throws if the returned
  // shape/length does not match the expected `layout` (identity safety, gate 5:
  // a chunk stored for a different model/layout must not decode as our block).
  bool GetKv2ltd(const std::string& key, const Kv2ltdLayout& layout,
                 std::vector<std::string>* k_planes,
                 std::vector<std::string>* v_planes);

 private:
  // Loop until all n bytes are written / read; throw on error or peer EOF.
  void SendAll(const char* data, std::size_t n);
  void RecvAll(char* data, std::size_t n);

  LmcacheClientConfig config_;
  int fd_ = -1;
};

}  // namespace vllm::v1::kv_offload::lmcache

#endif  // VLLM_V1_KV_OFFLOAD_LMCACHE_REMOTE_CLIENT_H_
