// LMCache MODE-1 (lm://) blocking TCP client — KV-EXTERNAL-CACHE W2 round-trip
// gate.
//
// The whole point of W2: our C++ LmcacheRemoteClient interoperates with a real
// lmcache.v1.server over the lm:// wire — PUT bytes, GET them back
// byte-identical, EXIST true/absent, a GET of an absent key returns absent (not
// garbage).
//
// This binary provides TWO round-trip harnesses over the SAME client:
//   1. An in-process C++ mock server that speaks the exact 186/36-byte protocol
//      (a faithful re-expression of lmcache/v1/server/__main__.py:34-135 using
//      OUR codec).  Always runs under ctest — self-contained, no Python.
//   2. When VT_LMCACHE_LIVE_HOST / VT_LMCACHE_LIVE_PORT are set, the identical
//      sequence is driven against a REAL running lmcache.v1.server.  This is the
//      live interop gate; see scripts/lmcache/run_live_roundtrip.sh.
//
// Ports the interop intent of the (Python) lm-connector round-trip tests; there
// is no direct upstream C++ analogue (spec "Tests to port": a NEW interop e2e).
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
#include <string>
#include <thread>
#include <vector>

#include "vllm/v1/kv_offload/lmcache/memory_format.h"
#include "vllm/v1/kv_offload/lmcache/remote_client.h"
#include "vllm/v1/kv_offload/lmcache/remote_protocol.h"

using namespace vllm::v1::kv_offload::lmcache;  // NOLINT(build/namespaces)

namespace {

// A faithful C++ mock of lmcache.v1.server.LMCacheServer (the lm:// CPU store).
// Binds 127.0.0.1:0 (an ephemeral port), serves one connection at a time in a
// background thread, and stores PUT payloads verbatim keyed by the header key.
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
    addr.sin_port = 0;  // let the kernel pick a free port
    REQUIRE(::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                   sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    REQUIRE(::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                          &len) == 0);
    port_ = ::ntohs(addr.sin_port);
    REQUIRE(::listen(listen_fd_, 4) == 0);
    thread_ = std::thread([this] { Run(); });
  }

  ~MockLmcacheServer() {
    stop_.store(true);
    // Unblock accept() by closing the listen socket.
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
    if (thread_.joinable()) {
      thread_.join();
    }
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
      if (r <= 0) {
        return false;  // EOF or error -> short frame
      }
      got += static_cast<std::size_t>(r);
    }
    return true;
  }

  static void SendAll(int fd, const char* data, std::size_t n) {
    std::size_t sent = 0;
    while (sent < n) {
      const ssize_t r = ::send(fd, data + sent, n - sent, MSG_NOSIGNAL);
      if (r < 0) {
        return;
      }
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

  static ServerMetaMessage OkEmptyMeta() {
    ServerMetaMessage m = FailMeta();
    m.code = ServerReturnCode::kSuccess;
    return m;
  }

  void HandleClient(int fd) {
    while (!stop_.load()) {
      std::string header(ClientMetaMessage::PackLength(), '\0');
      if (!RecvAll(fd, header.data(), header.size())) {
        break;  // client disconnected (server __main__.py:48-49)
      }
      const ClientMetaMessage meta = ClientMetaMessage::Deserialize(header);
      switch (meta.command) {
        case ClientCommand::kPut: {
          std::string payload(static_cast<std::size_t>(meta.length), '\0');
          if (meta.length > 0 &&
              !RecvAll(fd, payload.data(), payload.size())) {
            return;
          }
          store_[meta.key] =
              Entry{std::move(payload), meta.fmt, meta.dtype, meta.shape};
          break;  // PUT has no reply
        }
        case ClientCommand::kGet: {
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
          ServerMetaMessage m = FailMeta();
          if (store_.count(meta.key) != 0) {
            m.code = ServerReturnCode::kSuccess;
          }
          const std::string r = m.Serialize();
          SendAll(fd, r.data(), r.size());
          break;
        }
        case ClientCommand::kHealth: {
          const std::string r = OkEmptyMeta().Serialize();
          SendAll(fd, r.data(), r.size());
          break;
        }
        case ClientCommand::kList: {
          std::string joined;
          bool first = true;
          for (const auto& kv : store_) {
            if (!first) joined.push_back('\n');
            joined += kv.first;
            first = false;
          }
          ServerMetaMessage m = OkEmptyMeta();
          m.length = static_cast<int32_t>(joined.size());
          const std::string r = m.Serialize();
          SendAll(fd, r.data(), r.size());
          if (!joined.empty()) {
            SendAll(fd, joined.data(), joined.size());
          }
          break;
        }
      }
    }
    ::close(fd);
  }

  void Run() {
    while (!stop_.load()) {
      const int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) {
        break;  // listen socket closed -> shutting down
      }
      HandleClient(fd);
    }
  }

  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::map<std::string, Entry> store_;
};

// Deterministic pseudo-random-ish bytes so payloads are non-trivial.
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

// The full round-trip contract, run against ANY connected client (mock or real
// server).  This is the gate.
void RunRoundTrip(LmcacheRemoteClient* client) {
  // HEALTH.
  CHECK(client->Health());

  const std::string model_key = "meta-llama/Meta@2@0@deadbeef@bfloat16";

  // A GET / EXIST before any PUT: absent, not garbage.
  CHECK_FALSE(client->Exist(model_key));
  CHECK_FALSE(client->Get(model_key).has_value());

  // PUT then GET byte-identical, across several sizes/dtypes.
  struct Case {
    std::string key;
    std::size_t size;
    MemoryFormat fmt;
    Dtype dtype;
    std::vector<int32_t> shape;
  };
  // NOTE: every key string must be a valid LMCache CacheEngineKey
  // (model@world@worker@HEX_hash@dtype) — the real server parse_cache_key's
  // EVERY header, so the chunk-hash must be hex and the dtype one of
  // {half,bfloat16,float,double,uint8,fp8_e4m3fn,fp8_e5m2,...} (the key's dtype
  // string is independent of the payload dtype in the header).
  const std::vector<Case> cases = {
      {"m@1@0@0001@bfloat16", 1, MemoryFormat::kKV2LTD, Dtype::kBFloat16,
       {2, 1, 1, 1}},
      {"m@1@0@0002@half", 4096, MemoryFormat::kKV2LTD, Dtype::kFloat16,
       {2, 4, 16, 32}},
      {"m@1@0@0003@float", 65537, MemoryFormat::kKV2LTD, Dtype::kFloat32,
       {2, 8, 64, 32}},  // > 64 KiB: forces multiple recv() chunks
  };
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    const std::string payload = MakeBytes(c.size, static_cast<uint32_t>(i + 1));
    client->Put(c.key, payload, c.fmt, c.dtype, c.shape);
    CHECK(client->Exist(c.key));
    auto got = client->Get(c.key);
    REQUIRE(got.has_value());
    INFO("case " << c.key << " size " << c.size);
    CHECK(got->bytes.size() == payload.size());
    CHECK(got->bytes == payload);  // byte-identical
    CHECK(got->dtype == c.dtype);
    CHECK(got->fmt == c.fmt);
  }

  // A differing (valid but never-stored) key still misses.
  CHECK_FALSE(client->Exist("m@1@0@deadc0de@float"));
  CHECK_FALSE(client->Get("m@1@0@deadc0de@float").has_value());

  // KV_2LTD repack: paged-KV per-layer planes -> [2,L,T,D] chunk -> PUT -> GET
  // -> unpack -> byte-identical planes.  This proves what we ship is exactly the
  // MemoryFormat LMCache stores and what we read back reconstructs our block.
  {
    Kv2ltdLayout layout;
    layout.num_layers = 3;
    layout.num_tokens = 5;
    layout.hidden_dim = 8;
    layout.elem_size = 2;  // bf16
    const std::size_t plane_bytes = layout.LayerStride() * layout.elem_size;
    std::vector<std::string> k_planes(layout.num_layers);
    std::vector<std::string> v_planes(layout.num_layers);
    for (int l = 0; l < layout.num_layers; ++l) {
      k_planes[l] = MakeBytes(plane_bytes, 100u + static_cast<uint32_t>(l));
      v_planes[l] = MakeBytes(plane_bytes, 200u + static_cast<uint32_t>(l));
    }
    const std::string kkey = "m@1@0@aa11bb22@bfloat16";
    client->PutKv2ltd(kkey, layout, k_planes, v_planes, Dtype::kBFloat16);

    std::vector<std::string> k2, v2;
    REQUIRE(client->GetKv2ltd(kkey, layout, &k2, &v2));
    REQUIRE(k2.size() == k_planes.size());
    REQUIRE(v2.size() == v_planes.size());
    for (int l = 0; l < layout.num_layers; ++l) {
      CHECK(k2[l] == k_planes[l]);
      CHECK(v2[l] == v_planes[l]);
    }
    // GetKv2ltd on an absent (valid) key -> false, not a throw.
    std::vector<std::string> k3, v3;
    CHECK_FALSE(client->GetKv2ltd("m@1@0@f00d@bfloat16", layout, &k3, &v3));
  }
}

}  // namespace

TEST_CASE("lmcache LmcacheRemoteClient round-trip vs in-process mock server") {
  MockLmcacheServer server;
  LmcacheClientConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = server.port();
  LmcacheRemoteClient client(cfg);
  client.Connect();
  REQUIRE(client.connected());
  RunRoundTrip(&client);

  // LIST is implemented by the mock (the real server currently no-ops it):
  // after the round-trip the store is non-empty.
  const std::vector<std::string> keys = client.List();
  CHECK(keys.size() >= 1);
}

TEST_CASE("lmcache LmcacheRemoteClient config from env") {
  // Defaults.
  ::unsetenv("VT_LMCACHE_HOST");
  ::unsetenv("VT_LMCACHE_PORT");
  ::unsetenv("VT_LMCACHE_HASH_ALGO");
  {
    const LmcacheClientConfig c = LmcacheClientConfig::FromEnv();
    CHECK(c.host == "127.0.0.1");
    CHECK(c.port == 65432);
    CHECK(c.hash_algo == LmcacheClientConfig::HashAlgo::kBlake3);
  }
  ::setenv("VT_LMCACHE_HOST", "example.internal", 1);
  ::setenv("VT_LMCACHE_PORT", "5555", 1);
  ::setenv("VT_LMCACHE_HASH_ALGO", "vllm", 1);
  {
    const LmcacheClientConfig c = LmcacheClientConfig::FromEnv();
    CHECK(c.host == "example.internal");
    CHECK(c.port == 5555);
    CHECK(c.hash_algo == LmcacheClientConfig::HashAlgo::kVllm);
  }
  ::unsetenv("VT_LMCACHE_HOST");
  ::unsetenv("VT_LMCACHE_PORT");
  ::unsetenv("VT_LMCACHE_HASH_ALGO");
}

// Live interop gate: only runs when pointed at a REAL lmcache.v1.server via
// VT_LMCACHE_LIVE_HOST / VT_LMCACHE_LIVE_PORT.  Skipped (passing, no checks)
// under plain ctest.
TEST_CASE("lmcache LmcacheRemoteClient round-trip vs REAL lmcache.v1.server") {
  const char* host = std::getenv("VT_LMCACHE_LIVE_HOST");
  const char* port = std::getenv("VT_LMCACHE_LIVE_PORT");
  if (host == nullptr || port == nullptr) {
    MESSAGE(
        "skipped: set VT_LMCACHE_LIVE_HOST/VT_LMCACHE_LIVE_PORT to run against "
        "a real lmcache.v1.server");
    return;
  }
  LmcacheClientConfig cfg;
  cfg.host = host;
  cfg.port = std::atoi(port);
  LmcacheRemoteClient client(cfg);
  client.Connect();
  REQUIRE(client.connected());
  RunRoundTrip(&client);

  // Bidirectional wire interop with a REAL LMCache codec on the Python side
  // (scripts/lmcache/run_live_roundtrip.sh drives it; the payload hex is the
  // single source of truth passed via the environment):
  auto hex_decode = [](const std::string& hx) {
    std::string out;
    auto nib = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      return 0;
    };
    for (std::size_t i = 0; i + 1 < hx.size(); i += 2) {
      out.push_back(static_cast<char>((nib(hx[i]) << 4) | nib(hx[i + 1])));
    }
    return out;
  };
  //  * C++ -> Python: PUT the given payload; the Python real-codec client GETs
  //    it afterwards and byte-compares.
  const char* cpp_key = std::getenv("VT_LMCACHE_CPP_KEY");
  const char* cpp_hex = std::getenv("VT_LMCACHE_CPP_HEX");
  if (cpp_key != nullptr && cpp_hex != nullptr) {
    client.Put(cpp_key, hex_decode(cpp_hex), MemoryFormat::kKV2LTD,
               Dtype::kFloat16, {2, 1, 1, 1});
  }
  //  * Python -> C++: read what the Python real-codec client already PUT and
  //    byte-compare.
  const char* py_key = std::getenv("VT_LMCACHE_PY_KEY");
  const char* py_hex = std::getenv("VT_LMCACHE_PY_HEX");
  if (py_key != nullptr && py_hex != nullptr) {
    auto got = client.Get(py_key);
    REQUIRE_MESSAGE(got.has_value(),
                    "Python-written key not found by C++ client");
    CHECK(got->bytes == hex_decode(py_hex));  // Python PUT == C++ GET, byte-exact
  }
}
