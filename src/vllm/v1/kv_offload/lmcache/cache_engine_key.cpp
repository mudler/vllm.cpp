// Ported from: lmcache/utils.py:398-561 (CacheEngineKey) @ LMCache 8570aad.
#include "vllm/v1/kv_offload/lmcache/cache_engine_key.h"

#include <cstdio>
#include <stdexcept>

namespace vllm::v1::kv_offload::lmcache {
namespace {

std::vector<std::string> SplitOn(std::string_view s, char sep) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (true) {
    std::size_t pos = s.find(sep, start);
    if (pos == std::string_view::npos) {
      parts.emplace_back(s.substr(start));
      break;
    }
    parts.emplace_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  return parts;
}

int64_t ParseInt(const std::string& s) {
  std::size_t consumed = 0;
  const long long v = std::stoll(s, &consumed, 10);
  if (consumed != s.size()) {
    throw std::invalid_argument("CacheEngineKey: invalid integer field: " + s);
  }
  return static_cast<int64_t>(v);
}

}  // namespace

std::string CacheEngineKey::ChunkHashHex() const {
  // Python f"{chunk_hash:x}" — lowercase hex, minimal digits, "0" for zero.
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%llx",
                static_cast<unsigned long long>(chunk_hash));
  return std::string(buf);
}

std::string CacheEngineKey::ToString() const {
  std::string s = model_name;
  s.push_back('@');
  s += std::to_string(world_size);
  s.push_back('@');
  s += std::to_string(worker_id);
  s.push_back('@');
  s += ChunkHashHex();
  s.push_back('@');
  s += DtypeToStr(dtype);
  // utils.py:454-456 — optional tags as "@name%value".
  for (const auto& [k, v] : tags) {
    s.push_back('@');
    s += k;
    s.push_back('%');
    s += v;
  }
  return s;
}

CacheEngineKey CacheEngineKey::FromString(std::string_view s) {
  // utils.py:489-509.
  const std::vector<std::string> parts = SplitOn(s, '@');
  if (parts.size() < 5) {
    throw std::invalid_argument("CacheEngineKey: invalid key string");
  }
  CacheEngineKey key;
  key.model_name = parts[0];
  key.world_size = ParseInt(parts[1]);
  key.worker_id = ParseInt(parts[2]);
  {
    std::size_t consumed = 0;
    key.chunk_hash = std::stoull(parts[3], &consumed, 16);
    if (consumed != parts[3].size()) {
      throw std::invalid_argument("CacheEngineKey: invalid chunk_hash hex");
    }
  }
  key.dtype = StrToDtype(parts[4]);
  for (std::size_t i = 5; i < parts.size(); ++i) {
    const std::string& kv = parts[i];
    std::size_t pct = kv.find('%');
    if (pct == std::string::npos) {
      throw std::invalid_argument("CacheEngineKey: invalid tag");
    }
    key.tags.emplace_back(kv.substr(0, pct), kv.substr(pct + 1));
  }
  return key;
}

}  // namespace vllm::v1::kv_offload::lmcache
