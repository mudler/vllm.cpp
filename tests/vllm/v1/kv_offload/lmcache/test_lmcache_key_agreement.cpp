// LMCache MODE-1 (lm://) KEY AGREEMENT vs a REAL Python vLLM+LMCache peer —
// KV-EXTERNAL-CACHE W4 crux gate.
//
// This is THE gate the W3 connector named as remaining: our C++ CacheEngineKey
// strings must be BYTE-IDENTICAL to what a real lmcache `ChunkedTokenDatabase`
// (token_database.py:298-449 @ 8570aad) computes for the same tokens, at
// chunk_size 256, with vLLM's own `sha256_cbor` hash + `init_none_hash`
// (vllm/utils/hashing.py:43, kv_cache_utils.py:99-114 @ e24d1b24). A single
// divergent byte = zero cache hits against the peer.
//
// Fixtures in tests/fixtures/lmcache/key_agreement_fixtures.json are dumped by
// scripts/lmcache/gen_key_agreement_fixtures.py, which drives the REAL lmcache
// ChunkedTokenDatabase.process_tokens() UNMODIFIED (see that script's header).
// Ports the intent of lmcache-src tests/**/test_token_database*.py (the chunked
// prefix-hash key vectors) as a cross-implementation conformance oracle.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/v1/kv_offload/lmcache/cache_engine_key.h"
#include "vllm/v1/kv_offload/lmcache/chunked_token_database.h"
#include "vllm/v1/kv_offload/lmcache/lmcache_connector.h"
#include "vllm/v1/kv_offload/lmcache/remote_client.h"
#include "vllm/v1/kv_offload/lmcache/remote_protocol.h"

using nlohmann::json;
using namespace vllm::v1::kv_offload::lmcache;  // NOLINT(build/namespaces)

namespace {

json LoadFixtures() {
  const std::string path =
      std::string(TEST_FIXTURES_DIR) + "/lmcache/key_agreement_fixtures.json";
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "cannot open key-agreement fixtures: " << path);
  std::stringstream ss;
  ss << f.rdbuf();
  return json::parse(ss.str());
}

Dtype DtypeFromName(const std::string& n) {
  if (n == "bfloat16") return Dtype::kBFloat16;
  if (n == "half") return Dtype::kFloat16;
  if (n == "float") return Dtype::kFloat32;
  FAIL("unknown dtype in fixture: " << n);
  return Dtype::kNone;
}

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

}  // namespace

// The NONE_HASH bootstrap must match the peer's init_none_hash(sha256_cbor) with
// PYTHONHASHSEED=0 (kv_cache_utils.py:113-114): fold8(sha256_cbor("0")).
TEST_CASE("lmcache key-agreement: NONE_HASH matches the peer") {
  const json fx = LoadFixtures();
  const uint64_t expect = fx["cases"][0]["none_hash"].get<uint64_t>();
  CHECK(expect == 0x4e1195df020de59eULL);  // frozen: real peer, PYTHONHASHSEED=0
  CHECK(ChunkedTokenDatabase::NoneHashFromSeed("0") == expect);
}

// Every fixture case: the C++ ChunkedTokenDatabase reproduces the peer's chunk
// boundaries, folded chunk hashes, AND full CacheEngineKey strings byte-for-byte.
TEST_CASE("lmcache key-agreement: CacheEngineKey strings == real peer") {
  const json fx = LoadFixtures();
  int total_entries = 0;
  for (const json& c : fx["cases"]) {
    const std::string name = c["name"].get<std::string>();
    CAPTURE(name);
    const int chunk_size = c["chunk_size"].get<int>();
    const bool save_unfull = c["save_unfull_chunk"].get<bool>();
    const uint64_t none_hash = c["none_hash"].get<uint64_t>();
    const std::string seed = c["none_hash_seed"].get<std::string>();

    // NONE_HASH re-derivation from the seed matches the peer's dumped value.
    CHECK(ChunkedTokenDatabase::NoneHashFromSeed(seed) == none_hash);

    std::vector<int32_t> tokens;
    for (const json& t : c["tokens"]) tokens.push_back(t.get<int32_t>());

    ChunkedTokenDatabase db(chunk_size, none_hash, save_unfull,
                            PreCachingHash::kSha256Cbor);
    const std::vector<ChunkedTokenDatabase::Entry> entries =
        db.ProcessTokens(tokens);

    const json& want = c["entries"];
    REQUIRE(entries.size() == want.size());

    const Dtype dtype = DtypeFromName(c["dtype"].get<std::string>());
    const std::string model = c["model_name"].get<std::string>();
    const int64_t world = c["world_size"].get<int64_t>();
    const int64_t worker = c["worker_id"].get<int64_t>();
    const std::vector<std::string> keys =
        db.ProcessKeys(tokens, model, world, worker, dtype);
    REQUIRE(keys.size() == want.size());

    for (std::size_t i = 0; i < entries.size(); ++i) {
      CAPTURE(i);
      CHECK(entries[i].start == want[i]["start"].get<int>());
      CHECK(entries[i].end == want[i]["end"].get<int>());
      // The folded hash, its lowercase-hex, and the full key string all agree.
      const uint64_t want_hash =
          std::stoull(want[i]["chunk_hash_hex"].get<std::string>(), nullptr, 16);
      CHECK(entries[i].chunk_hash == want_hash);

      CacheEngineKey key;
      key.model_name = model;
      key.world_size = world;
      key.worker_id = worker;
      key.chunk_hash = entries[i].chunk_hash;
      key.dtype = dtype;
      CHECK(key.ChunkHashHex() == want[i]["chunk_hash_hex"].get<std::string>());
      CHECK(key.ToString() == want[i]["key"].get<std::string>());
      CHECK(keys[i] == want[i]["key"].get<std::string>());
      ++total_entries;
    }
  }
  MESSAGE("key-agreement entries checked: " << total_entries);
  CHECK(total_entries > 0);
}

// The LMCacheConnector itself, in kVllmSha256Cbor mode, produces the SAME
// peer-agreeing keys through its ChunkKey path (the wiring, not just the raw
// hasher). Uses the default-256 / bf16 / Llama-3.1-8B case.
TEST_CASE("lmcache key-agreement: LMCacheConnector peer mode == real peer") {
  const json fx = LoadFixtures();
  const json* two = nullptr;
  for (const json& c : fx["cases"]) {
    if (c["name"] == "two_chunks_512") two = &c;
  }
  REQUIRE(two != nullptr);

  LmcacheConnectorConfig cfg;
  cfg.key_mode = LmcacheConnectorConfig::KeyMode::kVllmSha256Cbor;
  cfg.none_hash_seed = (*two)["none_hash_seed"].get<std::string>();
  cfg.save_unfull_chunk = (*two)["save_unfull_chunk"].get<bool>();
  cfg.chunk_tokens = (*two)["chunk_size"].get<int>();
  cfg.model_name = (*two)["model_name"].get<std::string>();
  cfg.world_size = (*two)["world_size"].get<int64_t>();
  cfg.worker_id = (*two)["worker_id"].get<int64_t>();
  cfg.dtype = DtypeFromName((*two)["dtype"].get<std::string>());

  // Point the client at a definitely-closed port; we never Connect() here.
  LmcacheClientConfig client_cfg;
  client_cfg.port = 1;
  LMCacheConnector conn(cfg, client_cfg);

  std::vector<int32_t> tokens;
  for (const json& t : (*two)["tokens"]) tokens.push_back(t.get<int32_t>());
  const std::vector<uint64_t> folds = conn.ChunkFolds(tokens);
  const json& want = (*two)["entries"];
  REQUIRE(folds.size() == want.size());
  for (std::size_t i = 0; i < folds.size(); ++i) {
    const std::string key = conn.ChunkKey(folds[i]);
    CHECK(key == want[i]["key"].get<std::string>());
  }
}

// LIVE key-agreement OVER THE WIRE (peer -> us): opt-in, gated on
// VT_LMCACHE_LIVE_SPEC (path to the JSON scripts/lmcache/lm_key_interop.py wrote
// after a REAL lmcache ChunkedTokenDatabase PUT to a real lmcache.v1.server).
// We independently re-derive the key from the SAME tokens with our C++
// ChunkedTokenDatabase, ASSERT it byte-matches the peer's key, then GET the
// peer-written KV bytes and ASSERT they are byte-identical — the key-agreement
// made real over the actual wire. Driven by scripts/lmcache/run_key_interop.sh.
TEST_CASE("lmcache key-agreement: LIVE peer->us over the wire") {
  const char* spec_path = std::getenv("VT_LMCACHE_LIVE_SPEC");
  if (spec_path == nullptr || *spec_path == '\0') {
    MESSAGE("skipped (set VT_LMCACHE_LIVE_SPEC to run vs a real peer)");
    return;
  }
  std::ifstream f(spec_path);
  REQUIRE_MESSAGE(f.good(), "cannot open live spec: " << spec_path);
  std::stringstream ss;
  ss << f.rdbuf();
  const json spec = json::parse(ss.str());

  std::vector<int32_t> tokens;
  for (const json& t : spec["tokens"]) tokens.push_back(t.get<int32_t>());
  const int chunk = spec["chunk"].get<int>();
  const Dtype dtype = DtypeFromName(spec["dtype"].get<std::string>());

  // Independent C++ derivation of the peer key.
  ChunkedTokenDatabase db(chunk, ChunkedTokenDatabase::NoneHashFromSeed("0"),
                          /*save_unfull_chunk=*/false,
                          PreCachingHash::kSha256Cbor);
  const std::vector<std::string> keys =
      db.ProcessKeys(tokens, spec["model"].get<std::string>(),
                     spec["world"].get<int64_t>(),
                     spec["worker"].get<int64_t>(), dtype);
  REQUIRE(!keys.empty());
  const std::string our_key = keys[0];
  const std::string peer_key = spec["key"].get<std::string>();
  MESSAGE("our key : " << our_key);
  MESSAGE("peer key: " << peer_key);
  CHECK(our_key == peer_key);  // key agreement over the wire

  // GET the peer-written bytes with OUR key and prove byte-identity.
  LmcacheClientConfig client_cfg;
  client_cfg.host = spec["host"].get<std::string>();
  client_cfg.port = spec["port"].get<int>();
  LmcacheRemoteClient client(client_cfg);
  client.Connect();
  const auto got = client.Get(our_key);
  REQUIRE_MESSAGE(got.has_value(), "peer-written key not found by our client");
  const std::string want = HexDecode(spec["payload_hex"].get<std::string>());
  CHECK(got->bytes.size() == want.size());
  CHECK(got->bytes == want);
  client.Close();
}
