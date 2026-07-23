// LMCache MODE-1 (lm://) wire codec — KV-EXTERNAL-CACHE W1 correctness gate.
//
// The whole point of W1: our C++ wire bytes must be byte/bit-IDENTICAL to what
// the real Python LMCache codec produces.  The fixtures in
// tests/fixtures/lmcache/lmcache_fixtures.json were captured from:
//   * stdlib `struct` (which IS LMCache's framing codec, protocol.py) for the
//     ClientMetaMessage / ServerMetaMessage / CacheEngineKey vectors,
//   * the real `blake3` PyPI package (the exact package token_hasher.py imports)
//     for the rolling-hash vectors,
//   * numpy for the KV_2LTD [2,L,T,D] contiguous layout,
// by scripts/lmcache/gen_lmcache_fixtures.py.  A single divergent byte fails.
//
// Ports the wire-conformance intent of lmcache-src tests/**/test_protocol*.py,
// test_lm_connector*.py and test_token_hasher*.py (spec: Tests to port).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/v1/kv_offload/lmcache/cache_engine_key.h"
#include "vllm/v1/kv_offload/lmcache/memory_format.h"
#include "vllm/v1/kv_offload/lmcache/remote_protocol.h"
#include "vllm/v1/kv_offload/lmcache/token_hasher.h"

using nlohmann::json;
using namespace vllm::v1::kv_offload::lmcache;  // NOLINT(build/namespaces)

namespace {

std::string HexDecode(const std::string& hex) {
  std::string out;
  out.reserve(hex.size() / 2);
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  };
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back(static_cast<char>((nib(hex[i]) << 4) | nib(hex[i + 1])));
  }
  return out;
}

json LoadFixtures() {
  const std::string path =
      std::string(TEST_FIXTURES_DIR) + "/lmcache/lmcache_fixtures.json";
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "cannot open fixtures: " << path);
  std::stringstream ss;
  ss << f.rdbuf();
  return json::parse(ss.str());
}

MemoryFormat FmtFromName(const std::string& n) {
  if (n == "UNDEFINED") return MemoryFormat::kUndefined;
  if (n == "KV_2LTD") return MemoryFormat::kKV2LTD;
  if (n == "KV_T2D") return MemoryFormat::kKVT2D;
  if (n == "KV_2TD") return MemoryFormat::kKV2TD;
  if (n == "BINARY") return MemoryFormat::kBinary;
  if (n == "BINARY_BUFFER") return MemoryFormat::kBinaryBuffer;
  if (n == "KV_MLA_FMT") return MemoryFormat::kKVMLAFmt;
  if (n == "EC_TD") return MemoryFormat::kECTD;
  if (n == "HS_TD") return MemoryFormat::kHSTD;
  FAIL("unknown fmt name " << n);
  return MemoryFormat::kUndefined;
}

// Fixtures encode dtype as the torch dtype NAME (Python side); map to our enum.
Dtype DtypeFromJson(const json& j) {
  if (j.is_null()) return Dtype::kNone;
  const std::string n = j.get<std::string>();
  if (n == "float16") return Dtype::kFloat16;
  if (n == "bfloat16") return Dtype::kBFloat16;
  if (n == "float32") return Dtype::kFloat32;
  if (n == "float64") return Dtype::kFloat64;
  if (n == "uint8") return Dtype::kUint8;
  if (n == "float8_e4m3fn") return Dtype::kFloat8E4M3FN;
  if (n == "float8_e5m2") return Dtype::kFloat8E5M2;
  FAIL("unknown dtype name " << n);
  return Dtype::kNone;
}

Location LocationFromJson(const json& j) {
  if (j.is_null()) return Location::kNone;
  const std::string n = j.get<std::string>();
  if (n == "LocalCPUBackend") return Location::kLocalCPUBackend;
  if (n == "LocalDiskBackend") return Location::kLocalDiskBackend;
  FAIL("unknown location " << n);
  return Location::kNone;
}

ClientCommand CommandFromInt(int v) {
  return static_cast<ClientCommand>(v);
}

}  // namespace

TEST_CASE("lmcache ClientMetaMessage serialize == Python struct.pack bytes") {
  const json fx = LoadFixtures();
  for (const auto& c : fx["client_meta"]) {
    ClientMetaMessage m;
    m.command = CommandFromInt(c["command"].get<int>());
    m.key = c["key"].get<std::string>();
    m.length = c["length"].get<int32_t>();
    m.fmt = FmtFromName(c["fmt"].get<std::string>());
    m.dtype = DtypeFromJson(c["dtype"]);
    m.shape = c["shape"].get<std::vector<int32_t>>();
    m.location = LocationFromJson(c["location"]);

    const std::string got = m.Serialize();
    const std::string want = HexDecode(c["bytes_hex"].get<std::string>());
    CHECK(got.size() == 186);
    CHECK(want.size() == 186);
    INFO("case " << c["name"].get<std::string>());
    CHECK(got == want);

    // Round-trip: deserialize the Python bytes, re-serialize == identity.
    const ClientMetaMessage back = ClientMetaMessage::Deserialize(want);
    CHECK(back.key == m.key);
    CHECK(back.Serialize() == want);
  }
}

TEST_CASE("lmcache ServerMetaMessage serialize == Python struct.pack bytes") {
  const json fx = LoadFixtures();
  for (const auto& c : fx["server_meta"]) {
    ServerMetaMessage m;
    m.code = static_cast<ServerReturnCode>(c["code"].get<int>());
    m.length = c["length"].get<int32_t>();
    m.fmt = FmtFromName(c["fmt"].get<std::string>());
    m.dtype = DtypeFromJson(c["dtype"]);
    m.shape = c["shape"].get<std::vector<int32_t>>();
    m.location = LocationFromJson(c["location"]);

    const std::string got = m.Serialize();
    const std::string want = HexDecode(c["bytes_hex"].get<std::string>());
    INFO("case " << c["name"].get<std::string>());
    CHECK(got.size() == 36);
    CHECK(got == want);
    CHECK(ServerMetaMessage::Deserialize(want).Serialize() == want);
  }
}

TEST_CASE("lmcache CacheEngineKey.to_string == Python + round-trip") {
  const json fx = LoadFixtures();
  for (const auto& c : fx["cache_key"]) {
    CacheEngineKey k;
    k.model_name = c["model"].get<std::string>();
    k.world_size = c["world"].get<int64_t>();
    k.worker_id = c["worker"].get<int64_t>();
    k.chunk_hash = c["chunk_hash"].get<uint64_t>();
    k.dtype = DtypeFromJson(c["dtype"]);

    const std::string want = c["string"].get<std::string>();
    INFO("key " << want);
    CHECK(k.ChunkHashHex() == c["chunk_hash_hex"].get<std::string>());
    CHECK(k.ToString() == want);

    // from_string(to_string) is identity on all fields.
    const CacheEngineKey rt = CacheEngineKey::FromString(want);
    CHECK(rt.model_name == k.model_name);
    CHECK(rt.world_size == k.world_size);
    CHECK(rt.worker_id == k.worker_id);
    CHECK(rt.chunk_hash == k.chunk_hash);
    CHECK(rt.ToString() == want);
  }
}

TEST_CASE("lmcache CacheEngineKey tags round-trip") {
  CacheEngineKey k;
  k.model_name = "m";
  k.world_size = 1;
  k.worker_id = 0;
  k.chunk_hash = 0xabcdef;
  k.dtype = Dtype::kBFloat16;
  k.tags = {{"lora", "v1"}, {"prio", "hi"}};
  const std::string s = k.ToString();
  CHECK(s == "m@1@0@abcdef@bfloat16@lora%v1@prio%hi");
  const CacheEngineKey rt = CacheEngineKey::FromString(s);
  CHECK(rt.tags.size() == 2);
  CHECK(rt.ToString() == s);
}

TEST_CASE("lmcache blake3 rolling token hash bit-exact vs Python `blake3`") {
  const json fx = LoadFixtures();
  const json& b = fx["blake3"];
  TokenHasher hasher(256);

  // none_hash
  CHECK(TokenHasher::ToHex(hasher.none_hash()) ==
        b["none_hash_hex"].get<std::string>());

  // single-chunk hashes with the none prefix
  for (const auto& s : b["single"]) {
    const std::vector<uint32_t> toks = s["tokens"].get<std::vector<uint32_t>>();
    CHECK(TokenHasher::ToHex(hasher.HashTokens(toks)) ==
          s["digest_hex"].get<std::string>());
  }

  // explicit rolling pair: prefix = first digest
  {
    const auto c0 = b["rolling_pair"]["chunk0"].get<std::vector<uint32_t>>();
    const auto c1 = b["rolling_pair"]["chunk1"].get<std::vector<uint32_t>>();
    const std::string d0 = hasher.HashTokens(c0);
    CHECK(TokenHasher::ToHex(d0) ==
          b["rolling_pair"]["digest0_hex"].get<std::string>());
    const std::string d1 = hasher.HashTokens(c1, d0);
    CHECK(TokenHasher::ToHex(d1) ==
          b["rolling_pair"]["digest1_hex"].get<std::string>());
  }

  // compute_chunk_hashes at several chunk sizes (incl. the real default 256)
  for (const auto& cc : b["chunk_hashes"]) {
    const int cs = cc["chunk_size"].get<int>();
    const auto toks = cc["tokens"].get<std::vector<uint32_t>>();
    TokenHasher h(cs);
    const std::vector<std::string> got = h.ComputeChunkHashes(toks);
    const auto want = cc["chunk_hashes_hex"].get<std::vector<std::string>>();
    INFO("chunk_size " << cs << " ntokens " << toks.size());
    REQUIRE(got.size() == want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      CHECK(TokenHasher::ToHex(got[i]) == want[i]);
    }
  }

  // fold-to-uint64 (token_database _normalize_hash_to_int)
  for (const auto& f : b["fold_to_int"]) {
    const auto toks = f["tokens"].get<std::vector<uint32_t>>();
    const std::string d = hasher.HashTokens(toks);
    CHECK(TokenHasher::ToHex(d) == f["digest_hex"].get<std::string>());
    CHECK(TokenHasher::FoldToUint64(d) == f["folded_uint64"].get<uint64_t>());
  }
}

TEST_CASE("lmcache KV_2LTD [2,L,T,D] repack byte-exact + round-trip") {
  const json fx = LoadFixtures();
  for (const auto& c : fx["kv_2ltd"]) {
    const int L = c["num_layers"].get<int>();
    const int T = c["num_tokens"].get<int>();
    const int D = c["hidden_dim"].get<int>();
    Kv2ltdLayout layout;
    layout.num_layers = L;
    layout.num_tokens = T;
    layout.hidden_dim = D;
    layout.elem_size = sizeof(float);  // fixture is float32

    const std::string packed_want = HexDecode(c["packed_le_hex"].get<std::string>());
    REQUIRE(packed_want.size() == layout.NumBytes());

    // Build the per-(kv,layer) planes independently: element (kv,l,t,d) of the
    // fixture arange array has value == its flat index.  Packing these must
    // reproduce numpy's C-contiguous [2,L,T,D] bytes exactly.
    const std::size_t plane_elems = static_cast<std::size_t>(T) * D;
    std::vector<std::string> k_planes(L), v_planes(L);
    for (int kv = 0; kv < 2; ++kv) {
      for (int l = 0; l < L; ++l) {
        std::string plane;
        plane.resize(plane_elems * sizeof(float));
        for (std::size_t e = 0; e < plane_elems; ++e) {
          const std::size_t flat =
              (static_cast<std::size_t>(kv) * L + l) * plane_elems + e;
          const float val = static_cast<float>(flat);
          std::memcpy(plane.data() + e * sizeof(float), &val, sizeof(float));
        }
        (kv == 0 ? k_planes : v_planes)[static_cast<std::size_t>(l)] =
            std::move(plane);
      }
    }

    const std::string packed_got = PackKv2ltd(layout, k_planes, v_planes);
    INFO("KV_2LTD L=" << L << " T=" << T << " D=" << D);
    CHECK(packed_got == packed_want);

    // ByteOffset / stride order: value at (kv,l,t,d) == flat index.
    for (int kv = 0; kv < 2; ++kv) {
      for (int l = 0; l < L; ++l) {
        for (int t = 0; t < T; ++t) {
          for (int d = 0; d < D; ++d) {
            float v = 0.0f;
            std::memcpy(&v, packed_got.data() + layout.ByteOffset(kv, l, t, d),
                        sizeof(float));
            const std::size_t flat =
                ((static_cast<std::size_t>(kv) * L + l) * T + t) * D + d;
            CHECK(v == doctest::Approx(static_cast<float>(flat)));
          }
        }
      }
    }

    // Round-trip: unpack then repack is identity.
    std::vector<std::string> k2, v2;
    UnpackKv2ltd(layout, packed_got, &k2, &v2);
    CHECK(PackKv2ltd(layout, k2, v2) == packed_want);
  }
}
