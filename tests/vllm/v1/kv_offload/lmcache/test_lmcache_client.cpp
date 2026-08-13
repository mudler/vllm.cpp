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

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "vllm/support/test_platform_compat.h"
#include "vllm/v1/kv_offload/lmcache/memory_format.h"
#include "vllm/v1/kv_offload/lmcache/remote_client.h"
#include "vllm/v1/kv_offload/lmcache/remote_protocol.h"

using namespace vllm::v1::kv_offload::lmcache;  // NOLINT(build/namespaces)

namespace {

#if defined(_WIN32)
using TestSocket = SOCKET;
constexpr TestSocket kInvalidTestSocket = INVALID_SOCKET;
void EnsureTestSockets() {
  static std::once_flag once;
  std::call_once(once, [] {
    WSADATA data{};
    REQUIRE(WSAStartup(MAKEWORD(2, 2), &data) == 0);
  });
}
void CloseTestSocket(TestSocket socket) { closesocket(socket); }
void ShutdownTestSocket(TestSocket socket) { ::shutdown(socket, SD_BOTH); }
using TestSocketLength = int;
int TestSend(TestSocket socket, const char* data, std::size_t size) {
  return ::send(socket, data, static_cast<int>(size), 0);
}
int TestRecv(TestSocket socket, char* data, std::size_t size) {
  return ::recv(socket, data, static_cast<int>(size), 0);
}
void SetTestEnv(const char* name, const char* value) {
  REQUIRE(_putenv_s(name, value == nullptr ? "" : value) == 0);
}
#else
using TestSocket = int;
constexpr TestSocket kInvalidTestSocket = -1;
void EnsureTestSockets() {}
void CloseTestSocket(TestSocket socket) { ::close(socket); }
void ShutdownTestSocket(TestSocket socket) { ::shutdown(socket, SHUT_RDWR); }
using TestSocketLength = socklen_t;
ssize_t TestSend(TestSocket socket, const char* data, std::size_t size) {
  return ::send(socket, data, size, MSG_NOSIGNAL);
}
ssize_t TestRecv(TestSocket socket, char* data, std::size_t size) {
  return ::recv(socket, data, size, 0);
}
void SetTestEnv(const char* name, const char* value) {
  if (value == nullptr) {
    REQUIRE(::unsetenv(name) == 0);
  } else {
    REQUIRE(::setenv(name, value, 1) == 0);
  }
}
#endif

// A faithful C++ mock of lmcache.v1.server.LMCacheServer (the lm:// CPU store).
// Binds 127.0.0.1:0 (an ephemeral port), serves one connection at a time in a
// background thread, and stores PUT payloads verbatim keyed by the header key.
class MockLmcacheServer {
 public:
  explicit MockLmcacheServer(bool close_immediately = false)
      : close_immediately_(close_immediately) {
    EnsureTestSockets();
    const TestSocket fd = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(fd != kInvalidTestSocket);
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
#if defined(_WIN32)
                 reinterpret_cast<const char*>(&one),
#else
                 &one,
#endif
                 static_cast<int>(sizeof(one)));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    // htonl/ntohs are called UNQUALIFIED on purpose: they are functions in
    // glibc but function-like MACROS in the Darwin SDK (<sys/_endian.h>), and
    // `::htonl` does not compile against a macro.
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  // let the kernel pick a free port
    REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&addr),
                   static_cast<TestSocketLength>(sizeof(addr))) == 0);
    TestSocketLength len = static_cast<TestSocketLength>(sizeof(addr));
    REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
    port_ = ntohs(addr.sin_port);
    REQUIRE(::listen(fd, 4) == 0);
    // Publish only once the socket is ready; from the next line on the
    // descriptor is shared with the accept thread.
    listen_fd_.store(fd);
    thread_ = std::thread([this] { Run(); });
  }

  ~MockLmcacheServer() {
    stop_.store(true);
    // Unblock accept() by closing the listen socket. This runs BEFORE the join,
    // so the handoff has to be atomic with the accept thread's read.
    const TestSocket fd = listen_fd_.exchange(kInvalidTestSocket);
    if (fd != kInvalidTestSocket) {
      ShutdownTestSocket(fd);
      CloseTestSocket(fd);
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

  static bool RecvAll(TestSocket fd, char* data, std::size_t n) {
    std::size_t got = 0;
    while (got < n) {
      const auto r = TestRecv(fd, data + got, n - got);
      if (r <= 0) {
        return false;  // EOF or error -> short frame
      }
      got += static_cast<std::size_t>(r);
    }
    return true;
  }

  static void SendAll(TestSocket fd, const char* data, std::size_t n) {
    std::size_t sent = 0;
    while (sent < n) {
      const auto r = TestSend(fd, data + sent, n - sent);
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

  void HandleClient(TestSocket fd) {
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
    CloseTestSocket(fd);
  }

  void Run() {
    while (!stop_.load()) {
      const TestSocket listen_fd = listen_fd_.load();
      if (listen_fd == kInvalidTestSocket) {
        break;
      }
      // The destructor may close this descriptor between the load and accept;
      // accept then fails, which is the intended shutdown path.
      const TestSocket fd = ::accept(listen_fd, nullptr, nullptr);
      if (fd == kInvalidTestSocket) {
        break;  // listen socket closed -> shutting down
      }
      if (close_immediately_) {
        CloseTestSocket(fd);
        break;
      }
      HandleClient(fd);
    }
  }

  std::atomic<TestSocket> listen_fd_{kInvalidTestSocket};
  int port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::map<std::string, Entry> store_;
  bool close_immediately_ = false;
};

class OneReplyServer {
 public:
  OneReplyServer(std::string reply, std::string payload = {})
      : reply_(std::move(reply)), payload_(std::move(payload)) {
    EnsureTestSockets();
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(listen_fd_ != kInvalidTestSocket);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                   static_cast<TestSocketLength>(sizeof(addr))) == 0);
    TestSocketLength length = static_cast<TestSocketLength>(sizeof(addr));
    REQUIRE(::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                          &length) == 0);
    port_ = ntohs(addr.sin_port);
    REQUIRE(::listen(listen_fd_, 1) == 0);
    thread_ = std::thread([this] {
      TestSocket client = ::accept(listen_fd_, nullptr, nullptr);
      if (client == kInvalidTestSocket) return;
      std::string request(ClientMetaMessage::PackLength(), '\0');
      if (RecvRequest(client, request.data(), request.size())) {
        SendReply(client, reply_.data(), reply_.size());
        SendReply(client, payload_.data(), payload_.size());
      }
      CloseTestSocket(client);
    });
  }

  ~OneReplyServer() {
    ShutdownTestSocket(listen_fd_);
    CloseTestSocket(listen_fd_);
    if (thread_.joinable()) thread_.join();
  }
  int port() const { return port_; }

 private:
  static bool RecvRequest(TestSocket fd, char* data, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
      const auto count = TestRecv(fd, data + offset, size - offset);
      if (count <= 0) return false;
      offset += static_cast<std::size_t>(count);
    }
    return true;
  }
  static void SendReply(TestSocket fd, const char* data, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
      const auto count = TestSend(fd, data + offset, size - offset);
      if (count <= 0) return;
      offset += static_cast<std::size_t>(count);
    }
  }
  TestSocket listen_fd_ = kInvalidTestSocket;
  int port_ = 0;
  std::string reply_;
  std::string payload_;
  std::thread thread_;
};

LmcacheClientConfig OneReplyConfig(const OneReplyServer& server) {
  LmcacheClientConfig config;
  config.host = "127.0.0.1";
  config.port = server.port();
  config.connect_retries = 1;
  return config;
}

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
  EnsureTestSockets();
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

TEST_CASE("lmcache peer close invalidates the owned socket") {
  MockLmcacheServer server(/*close_immediately=*/true);
  LmcacheClientConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = server.port();
  cfg.connect_retries = 1;
  LmcacheRemoteClient client(cfg);
  client.Connect();
  REQUIRE(client.connected());
  CHECK_THROWS(client.Health());
  CHECK_FALSE(client.connected());
  client.Close();
  CHECK_FALSE(client.connected());
}

TEST_CASE("lmcache malformed response headers close the connection") {
  ServerMetaMessage meta;
  meta.code = ServerReturnCode::kSuccess;
  meta.length = -1;
  meta.fmt = MemoryFormat::kKV2LTD;
  meta.dtype = Dtype::kFloat16;
  meta.shape = {2, 1, 1, 1};
  OneReplyServer server(meta.Serialize());
  LmcacheRemoteClient client(OneReplyConfig(server));
  client.Connect();
  CHECK_THROWS_WITH_AS(client.Get("m@1@0@1@half"),
                       doctest::Contains("negative response length"),
                       std::runtime_error);
  CHECK_FALSE(client.connected());
}

TEST_CASE("lmcache invalid return code and decode failure close connection") {
  ServerMetaMessage meta;
  meta.code = static_cast<ServerReturnCode>(201);
  meta.length = 0;
  meta.fmt = MemoryFormat::kKV2LTD;
  meta.dtype = Dtype::kFloat16;
  meta.shape = {0, 0, 0, 0};
  {
    OneReplyServer server(meta.Serialize());
    LmcacheRemoteClient client(OneReplyConfig(server));
    client.Connect();
    CHECK_THROWS_WITH_AS(client.Health(), doctest::Contains("return code"),
                         std::runtime_error);
    CHECK_FALSE(client.connected());
  }
  std::string invalid_dtype = meta.Serialize();
  invalid_dtype[12] = static_cast<char>(99);
  {
    OneReplyServer server(invalid_dtype);
    LmcacheRemoteClient client(OneReplyConfig(server));
    client.Connect();
    CHECK_THROWS(client.Health());
    CHECK_FALSE(client.connected());
  }
}

TEST_CASE("lmcache response lengths match the requested operation") {
  ServerMetaMessage meta;
  meta.code = ServerReturnCode::kFail;
  meta.length = 1;
  meta.fmt = MemoryFormat::kKV2LTD;
  meta.dtype = Dtype::kFloat16;
  meta.shape = {0, 0, 0, 0};
  OneReplyServer server(meta.Serialize(), "x");
  LmcacheRemoteClient client(OneReplyConfig(server));
  client.Connect();
  CHECK_THROWS_WITH_AS(client.Exist("m@1@0@1@half"),
                       doctest::Contains("error response length"),
                       std::runtime_error);
  CHECK_FALSE(client.connected());
}

TEST_CASE("lmcache successful response payload contracts close the connection") {
  ServerMetaMessage meta;
  meta.code = ServerReturnCode::kSuccess;
  meta.length = 0;
  meta.fmt = MemoryFormat::kKV2LTD;
  meta.dtype = Dtype::kFloat16;
  meta.shape = {2, 1, 1, 1};

  SUBCASE("GET requires a non-empty payload") {
    OneReplyServer server(meta.Serialize());
    LmcacheRemoteClient client(OneReplyConfig(server));
    client.Connect();
    CHECK_THROWS_WITH_AS(client.Get("m@1@0@1@half"),
                         doctest::Contains("payload is missing"),
                         std::runtime_error);
    CHECK_FALSE(client.connected());
  }

  SUBCASE("EXIST forbids a response payload") {
    meta.length = 1;
    OneReplyServer server(meta.Serialize(), "x");
    LmcacheRemoteClient client(OneReplyConfig(server));
    client.Connect();
    CHECK_THROWS_WITH_AS(client.Exist("m@1@0@1@half"),
                         doctest::Contains("length must be zero"),
                         std::runtime_error);
    CHECK_FALSE(client.connected());
  }

  SUBCASE("HEALTH forbids a response payload") {
    meta.length = 1;
    OneReplyServer server(meta.Serialize(), "x");
    LmcacheRemoteClient client(OneReplyConfig(server));
    client.Connect();
    CHECK_THROWS_WITH_AS(client.Health(),
                         doctest::Contains("length must be zero"),
                         std::runtime_error);
    CHECK_FALSE(client.connected());
  }
}

TEST_CASE("lmcache LIST rejects malformed key payloads and closes the connection") {
  ServerMetaMessage meta;
  meta.code = ServerReturnCode::kSuccess;
  meta.fmt = MemoryFormat::kKV2LTD;
  meta.dtype = Dtype::kFloat16;
  meta.shape = {0, 0, 0, 0};

  for (const std::string& payload : {
           std::string("m@1@0@1@half\n\nm@1@0@2@half"),
           std::string("m@1@0@1@half\n"),
           std::string("not-a-cache-engine-key"),
       }) {
    CAPTURE(payload);
    meta.length = static_cast<int32_t>(payload.size());
    OneReplyServer server(meta.Serialize(), payload);
    LmcacheRemoteClient client(OneReplyConfig(server));
    client.Connect();
    CHECK_THROWS(client.List());
    CHECK_FALSE(client.connected());
  }
}

TEST_CASE("lmcache oversized PUT is rejected before the signed wire cast") {
  LmcacheRemoteClient client;
  static const char byte = 0;
  const std::string_view oversized(
      &byte, static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) + 1);
  CHECK_THROWS_WITH_AS(
      client.Put("m@1@0@1@half", oversized, MemoryFormat::kKV2LTD,
                 Dtype::kFloat16, {2, 1, 1, 1}),
      doctest::Contains("INT32_MAX"), std::invalid_argument);
}

TEST_CASE("lmcache LmcacheRemoteClient config from env") {
  // Defaults.
  SetTestEnv("VT_LMCACHE_HOST", nullptr);
  SetTestEnv("VT_LMCACHE_PORT", nullptr);
  SetTestEnv("VT_LMCACHE_HASH_ALGO", nullptr);
  {
    const LmcacheClientConfig c = LmcacheClientConfig::FromEnv();
    CHECK(c.host == "127.0.0.1");
    CHECK(c.port == 65432);
    CHECK(c.hash_algo == LmcacheClientConfig::HashAlgo::kBlake3);
  }
  SetTestEnv("VT_LMCACHE_HOST", "example.internal");
  SetTestEnv("VT_LMCACHE_PORT", "5555");
  SetTestEnv("VT_LMCACHE_HASH_ALGO", "vllm");
  {
    const LmcacheClientConfig c = LmcacheClientConfig::FromEnv();
    CHECK(c.host == "example.internal");
    CHECK(c.port == 5555);
    CHECK(c.hash_algo == LmcacheClientConfig::HashAlgo::kVllm);
  }
  SetTestEnv("VT_LMCACHE_HOST", nullptr);
  SetTestEnv("VT_LMCACHE_PORT", nullptr);
  SetTestEnv("VT_LMCACHE_HASH_ALGO", nullptr);
}

// Live interop gate: only runs when pointed at a REAL lmcache.v1.server via
// VT_LMCACHE_LIVE_HOST / VT_LMCACHE_LIVE_PORT.  Skipped (passing, no checks)
// under plain ctest.
TEST_CASE("lmcache LmcacheRemoteClient round-trip vs REAL lmcache.v1.server") {
  EnsureTestSockets();
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
