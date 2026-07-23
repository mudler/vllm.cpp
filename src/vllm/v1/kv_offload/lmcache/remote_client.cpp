// Ported from: lmcache/v1/storage_backend/connector/lm_connector.py:28-177
//              + lmcache/v1/server/__main__.py:34-135 @ LMCache 8570aad.
#include "vllm/v1/kv_offload/lmcache/remote_client.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace vllm::v1::kv_offload::lmcache {

LmcacheClientConfig LmcacheClientConfig::FromEnv() {
  LmcacheClientConfig cfg;
  if (const char* h = std::getenv("VT_LMCACHE_HOST"); h != nullptr && *h) {
    cfg.host = h;
  }
  if (const char* p = std::getenv("VT_LMCACHE_PORT"); p != nullptr && *p) {
    cfg.port = std::atoi(p);
  }
  if (const char* a = std::getenv("VT_LMCACHE_HASH_ALGO"); a != nullptr && *a) {
    const std::string s = a;
    if (s == "blake3") {
      cfg.hash_algo = HashAlgo::kBlake3;
    } else if (s == "vllm") {
      cfg.hash_algo = HashAlgo::kVllm;
    } else {
      throw std::invalid_argument(
          "VT_LMCACHE_HASH_ALGO must be 'blake3' or 'vllm'");
    }
  }
  return cfg;
}

LmcacheRemoteClient::LmcacheRemoteClient(LmcacheClientConfig config)
    : config_(std::move(config)) {}

LmcacheRemoteClient::~LmcacheRemoteClient() { Close(); }

void LmcacheRemoteClient::Close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

void LmcacheRemoteClient::Connect() {
  if (fd_ >= 0) {
    return;
  }
  // socket.socket(AF_INET, SOCK_STREAM); socket.connect((host, port))
  // (lm_connector.py:47-48).  Resolve host via getaddrinfo so both a numeric
  // "127.0.0.1" and a name work.
  const std::string port_str = std::to_string(config_.port);
  addrinfo hints{};
  hints.ai_family = AF_INET;  // lm:// server binds AF_INET (__main__.py:30)
  hints.ai_socktype = SOCK_STREAM;

  std::string last_error;
  for (int attempt = 0; attempt < config_.connect_retries; ++attempt) {
    addrinfo* res = nullptr;
    const int gai =
        ::getaddrinfo(config_.host.c_str(), port_str.c_str(), &hints, &res);
    if (gai != 0) {
      last_error = std::string("getaddrinfo: ") + ::gai_strerror(gai);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    int fd = -1;
    for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
      fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
      if (fd < 0) {
        last_error = std::string("socket: ") + std::strerror(errno);
        continue;
      }
      if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
        break;  // connected
      }
      last_error = std::string("connect: ") + std::strerror(errno);
      ::close(fd);
      fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd >= 0) {
      // Disable Nagle: our frames are small headers followed by a large
      // payload; batching the header with the payload is fine, but we never
      // want the header stalled waiting for more data.
      int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      fd_ = fd;
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error("LmcacheRemoteClient::Connect: failed to connect to " +
                           config_.host + ":" + port_str + " (" + last_error +
                           ")");
}

void LmcacheRemoteClient::SendAll(const char* data, std::size_t n) {
  if (fd_ < 0) {
    throw std::runtime_error("LmcacheRemoteClient::SendAll: not connected");
  }
  std::size_t sent = 0;
  while (sent < n) {
    // MSG_NOSIGNAL: a broken pipe returns EPIPE instead of raising SIGPIPE.
    const ssize_t r = ::send(fd_, data + sent, n - sent, MSG_NOSIGNAL);
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("LmcacheRemoteClient::SendAll: ") +
                               std::strerror(errno));
    }
    sent += static_cast<std::size_t>(r);
  }
}

void LmcacheRemoteClient::RecvAll(char* data, std::size_t n) {
  if (fd_ < 0) {
    throw std::runtime_error("LmcacheRemoteClient::RecvAll: not connected");
  }
  std::size_t got = 0;
  while (got < n) {
    const ssize_t r = ::recv(fd_, data + got, n - got, 0);
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("LmcacheRemoteClient::RecvAll: ") +
                               std::strerror(errno));
    }
    if (r == 0) {
      // Peer closed mid-frame: a short read is an error (mirrors the server's
      // receive_all returning None on a truncated frame, __main__.py:38-39).
      throw std::runtime_error(
          "LmcacheRemoteClient::RecvAll: connection closed mid-frame");
    }
    got += static_cast<std::size_t>(r);
  }
}

void LmcacheRemoteClient::Put(const std::string& key, std::string_view kv_bytes,
                              MemoryFormat fmt, Dtype dtype,
                              const std::vector<int32_t>& shape) {
  // ClientMetaMessage(PUT, key, len(kv_bytes), fmt, dtype, shape) then the raw
  // payload (lm_connector.py:126-136).
  ClientMetaMessage msg;
  msg.command = ClientCommand::kPut;
  msg.key = key;
  msg.length = static_cast<int32_t>(kv_bytes.size());
  msg.fmt = fmt;
  msg.dtype = dtype;
  msg.shape = shape;
  const std::string header = msg.Serialize();
  SendAll(header.data(), header.size());
  SendAll(kv_bytes.data(), kv_bytes.size());
}

std::optional<LmcacheRemoteClient::GetResult> LmcacheRemoteClient::Get(
    const std::string& key) {
  // GET header uses the placeholder fmt/dtype/shape the upstream client sends
  // (MemoryFormat(1), float16, [0,0,0,0]) — the server ignores them for GET
  // and keys only on `key` (lm_connector.py:146-155, __main__.py:64-66).
  ClientMetaMessage msg;
  msg.command = ClientCommand::kGet;
  msg.key = key;
  msg.length = 0;
  msg.fmt = MemoryFormat::kKV2LTD;
  msg.dtype = Dtype::kFloat16;
  msg.shape = {0, 0, 0, 0};
  const std::string header = msg.Serialize();
  SendAll(header.data(), header.size());

  std::string reply(ServerMetaMessage::PackLength(), '\0');
  RecvAll(reply.data(), reply.size());
  const ServerMetaMessage meta = ServerMetaMessage::Deserialize(reply);
  if (meta.code != ServerReturnCode::kSuccess) {
    return std::nullopt;  // absent
  }
  GetResult out;
  out.fmt = meta.fmt;
  out.dtype = meta.dtype;
  out.shape = meta.shape;
  out.bytes.resize(static_cast<std::size_t>(meta.length));
  if (meta.length > 0) {
    RecvAll(out.bytes.data(), out.bytes.size());
  }
  return out;
}

bool LmcacheRemoteClient::Exist(const std::string& key) {
  ClientMetaMessage msg;
  msg.command = ClientCommand::kExist;
  msg.key = key;
  msg.length = 0;
  msg.fmt = MemoryFormat::kKV2LTD;
  msg.dtype = Dtype::kFloat16;
  msg.shape = {0, 0, 0, 0};
  const std::string header = msg.Serialize();
  SendAll(header.data(), header.size());

  std::string reply(ServerMetaMessage::PackLength(), '\0');
  RecvAll(reply.data(), reply.size());
  return ServerMetaMessage::Deserialize(reply).code == ServerReturnCode::kSuccess;
}

// The lm:// server deserializes EVERY header via parse_cache_key BEFORE the
// command switch (server __main__.py:50), so even HEALTH/LIST must carry a
// syntactically valid CacheEngineKey string (>=5 @-parts, a hex chunk-hash, a
// known dtype, and a non-digit 6th part).  A benign placeholder satisfies it.
namespace {
constexpr const char* kPlaceholderKey = "__vt_health__@0@0@0@half";
}  // namespace

bool LmcacheRemoteClient::Health() {
  ClientMetaMessage msg;
  msg.command = ClientCommand::kHealth;
  msg.key = kPlaceholderKey;
  msg.length = 0;
  msg.fmt = MemoryFormat::kKV2LTD;
  msg.dtype = Dtype::kFloat16;
  msg.shape = {0, 0, 0, 0};
  const std::string header = msg.Serialize();
  SendAll(header.data(), header.size());

  std::string reply(ServerMetaMessage::PackLength(), '\0');
  RecvAll(reply.data(), reply.size());
  return ServerMetaMessage::Deserialize(reply).code == ServerReturnCode::kSuccess;
}

std::vector<std::string> LmcacheRemoteClient::List() {
  ClientMetaMessage msg;
  msg.command = ClientCommand::kList;
  msg.key = kPlaceholderKey;
  msg.length = 0;
  msg.fmt = MemoryFormat::kKV2LTD;
  msg.dtype = Dtype::kFloat16;
  msg.shape = {0, 0, 0, 0};
  const std::string header = msg.Serialize();
  SendAll(header.data(), header.size());

  std::string reply(ServerMetaMessage::PackLength(), '\0');
  RecvAll(reply.data(), reply.size());
  const ServerMetaMessage meta = ServerMetaMessage::Deserialize(reply);
  std::vector<std::string> keys;
  if (meta.code != ServerReturnCode::kSuccess || meta.length == 0) {
    return keys;
  }
  std::string data(static_cast<std::size_t>(meta.length), '\0');
  RecvAll(data.data(), data.size());
  // Newline-joined keys (server __main__.py:127 `"\n".join(keys)`).
  std::size_t start = 0;
  while (start <= data.size()) {
    const std::size_t nl = data.find('\n', start);
    if (nl == std::string::npos) {
      if (start < data.size()) {
        keys.emplace_back(data.substr(start));
      }
      break;
    }
    keys.emplace_back(data.substr(start, nl - start));
    start = nl + 1;
  }
  return keys;
}

void LmcacheRemoteClient::PutKv2ltd(const std::string& key,
                                    const Kv2ltdLayout& layout,
                                    const std::vector<std::string>& k_planes,
                                    const std::vector<std::string>& v_planes,
                                    Dtype dtype) {
  const std::string packed = PackKv2ltd(layout, k_planes, v_planes);
  Put(key, packed, MemoryFormat::kKV2LTD, dtype, layout.Shape());
}

bool LmcacheRemoteClient::GetKv2ltd(const std::string& key,
                                    const Kv2ltdLayout& layout,
                                    std::vector<std::string>* k_planes,
                                    std::vector<std::string>* v_planes) {
  std::optional<GetResult> got = Get(key);
  if (!got.has_value()) {
    return false;
  }
  // Identity safety (gate 5): refuse a payload whose length/shape does not
  // match the layout we asked for, rather than mis-decoding another model's
  // bytes as our block.
  if (got->bytes.size() != layout.NumBytes()) {
    throw std::runtime_error(
        "LmcacheRemoteClient::GetKv2ltd: payload byte count " +
        std::to_string(got->bytes.size()) + " != expected " +
        std::to_string(layout.NumBytes()));
  }
  const std::vector<int32_t> want = layout.Shape();
  if (!got->shape.empty() && got->shape != want) {
    throw std::runtime_error(
        "LmcacheRemoteClient::GetKv2ltd: shape mismatch for key " + key);
  }
  UnpackKv2ltd(layout, got->bytes, k_planes, v_planes);
  return true;
}

}  // namespace vllm::v1::kv_offload::lmcache
