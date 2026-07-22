// Ported from: tests/v1/kv_offload/tiering/test_fs_tier.py and
// tests/v1/kv_offload/test_file_mapper.py @ e24d1b24, PLUS the per-field
// identity-refusal matrix that upstream has no counterpart for (upstream writes
// a config.json it never reads — see include/vllm/v1/kv_offload/cache_identity.h).
//
// GATE 9 (disk hygiene): every case bounds its root_dir to a unique temporary
// directory and removes it. An unbounded disk tier filling the box presents as
// unrelated bogus failures elsewhere, which is the recorded ENOSPC lesson.
#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_offload/base.h"
#include "vllm/v1/kv_offload/cache_identity.h"
#include "vllm/v1/kv_offload/fs_io.h"
#include "vllm/v1/kv_offload/fs_tier.h"

using namespace vllm::v1::kv_offload;  // NOLINT(build/namespaces)
using vllm::v1::BlockHash;

namespace {

// A self-cleaning temporary root. Bounded by construction (gate 9).
class TempDir {
 public:
  explicit TempDir(const std::string& tag) {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("vllmcpp_kvoff_" + tag + "_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::string str() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

OffloadKey Key(int i, uint32_t group = 0) {
  BlockHash h(32, '\0');
  h[0] = static_cast<char>(i & 0xff);
  h[1] = static_cast<char>((i >> 8) & 0xff);
  return make_offload_key(h, group);
}

constexpr int64_t kPage = 512;

// A fully populated, VALID identity. Every refusal test mutates exactly one
// field of this.
CacheIdentity BaseIdentity() {
  CacheIdentity id;
  id.model_name = "Qwen/Qwen3-4B";
  id.model_type = "qwen3";
  id.architectures = {"Qwen3ForCausalLM"};
  id.hf_config_digest = "deadbeefcafe";
  id.weight_quantization = "none";
  id.checkpoint_fingerprint = "sha256:aaaa";
  id.num_hidden_layers = 36;
  id.num_kv_heads = 8;
  id.head_size = 128;
  id.head_size_v = 128;
  id.sliding_window = -1;
  id.rope_config = "theta=1000000,dim=128,scaling=none";
  id.max_position_embeddings = 40960;
  id.kv_cache_spec_kind = "full_attention";
  id.page_size_bytes = kPage;
  id.block_size = 16;
  id.hash_block_size = 16;
  id.kv_dtype = "bf16";
  id.kv_quant_mode = "none";
  id.hash_algo = "sha256_cbor";
  id.none_hash_hex = std::string(64, 'a');
  id.none_hash_source = "default";
  return id;
}

std::vector<uint8_t> Payload(uint8_t fill) {
  std::vector<uint8_t> p(kPage);
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = static_cast<uint8_t>(fill + i);
  }
  return p;
}

FileSystemTierOptions Options(const TempDir& dir, CacheIdentity id,
                              int64_t capacity_bytes = 0) {
  FileSystemTierOptions o;
  o.root_dir = dir.str();
  o.identity = std::move(id);
  o.capacity_bytes = capacity_bytes;
  return o;
}

}  // namespace

// --- identity plumbing --------------------------------------------------------

TEST_CASE("the canonical identity JSON round-trips and its digest is stable") {
  const CacheIdentity id = BaseIdentity();
  const std::string json = id.ToCanonicalJson();
  const CacheIdentity back = CacheIdentity::FromCanonicalJson(json);
  CHECK_FALSE(CacheIdentity::FirstMismatch(id, back).has_value());
  CHECK(back.ToCanonicalJson() == json);
  CHECK(id.Digest().size() == 32);
  CHECK(id.ShortDigestHex().size() == 12);
}

TEST_CASE("an INCOMPLETE identity is rejected at construction") {
  // An empty weight_quantization silently reading as "unquantized" is exactly
  // the silent-wrong-output hazard the header exists to prevent, so it is a
  // hard error rather than a default.
  CacheIdentity id = BaseIdentity();
  id.weight_quantization.clear();
  auto bad = id.Validate();
  REQUIRE(bad.has_value());
  CHECK(*bad == "weight_quantization");

  TempDir dir("incomplete");
  CHECK_THROWS_AS(FileMapper(dir.str(), id), std::runtime_error);
}

TEST_CASE("none_hash_source is recorded but NOT compared") {
  // The same seed VALUE reached via PYTHONHASHSEED or via the built-in default
  // describes the same cache; refusing on the route would be a false positive.
  CacheIdentity a = BaseIdentity();
  CacheIdentity b = BaseIdentity();
  b.none_hash_source = "PYTHONHASHSEED";
  CHECK_FALSE(CacheIdentity::FirstMismatch(a, b).has_value());
}

TEST_CASE("the path layout mirrors upstream's fan-out") {
  // file_mapper.py:112-120: <base>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin
  TempDir dir("layout");
  FileMapper mapper(dir.str(), BaseIdentity());
  const std::string path = mapper.file_name(Key(0, /*group=*/3));
  CHECK(path.find("_r0/") != std::string::npos);
  CHECK(path.find("_g3/") != std::string::npos);
  CHECK(path.substr(path.size() - 4) == ".bin");
  // '/' in a HuggingFace id must not nest directories.
  CHECK(mapper.base_path().find("Qwen_Qwen3-4B_") != std::string::npos);
}

// --- round trip ---------------------------------------------------------------

TEST_CASE("a stored block round-trips BYTE-IDENTICALLY") {
  TempDir dir("roundtrip");
  FileSystemTier tier(Options(dir, BaseIdentity()));
  const auto payload = Payload(0x11);

  tier.store(Key(1), payload.data(), payload.size());
  CHECK(tier.num_blocks() == 1);
  CHECK(tier.lookup({Key(1), Key(2)}) == LookupResults{true, false});

  std::vector<uint8_t> out(kPage, 0);
  CHECK(tier.load(Key(1), out.data(), out.size()));
  CHECK(out == payload);
}

TEST_CASE("MLA's rank-3 page shape round-trips through the same code path") {
  // §Risks R9: MLA stores ONE 576-wide latent with num_kv_heads == 1 and NO V
  // tensor, while full attention is rank-4 with an interleaved [K|V] page. The
  // IO layer must interpret NEITHER — the payload is opaque bytes of exactly
  // page_size_bytes and the spec kind lives in the identity.
  TempDir dir("mla");
  CacheIdentity id = BaseIdentity();
  id.kv_cache_spec_kind = "mla";
  id.num_kv_heads = 1;
  id.head_size = 576;
  id.head_size_v = 0;
  id.page_size_bytes = 16 * 1 * 576 * 2;  // block_size * heads * dim * bf16
  FileSystemTier tier(Options(dir, id));

  std::vector<uint8_t> payload(id.page_size_bytes);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>(i * 3);
  }
  tier.store(Key(7), payload.data(), payload.size());
  std::vector<uint8_t> out(payload.size(), 0);
  CHECK(tier.load(Key(7), out.data(), out.size()));
  CHECK(out == payload);
}

TEST_CASE("a missing block is a plain MISS, not an error") {
  TempDir dir("miss");
  FileSystemTier tier(Options(dir, BaseIdentity()));
  std::vector<uint8_t> out(kPage, 0);
  CHECK_FALSE(tier.load(Key(42), out.data(), out.size()));
}

TEST_CASE("storing the same block twice is an existence-skip, not a rewrite") {
  TempDir dir("skip");
  FileSystemTier tier(Options(dir, BaseIdentity()));
  const auto payload = Payload(0x22);
  tier.store(Key(1), payload.data(), payload.size());
  tier.store(Key(1), payload.data(), payload.size());
  CHECK(tier.num_blocks() == 1);
}

TEST_CASE("a payload whose size disagrees with the identity is REFUSED") {
  TempDir dir("size");
  FileSystemTier tier(Options(dir, BaseIdentity()));
  std::vector<uint8_t> wrong(kPage / 2, 0);
  CHECK_THROWS_AS(tier.store(Key(1), wrong.data(), wrong.size()),
                  std::runtime_error);
}

TEST_CASE("the async path round-trips over the dual-queue pool") {
  TempDir dir("async");
  FileSystemTier tier(Options(dir, BaseIdentity()));
  const auto payload = Payload(0x33);
  std::vector<int64_t> jobs;
  std::vector<std::vector<uint8_t>> payloads;
  for (int i = 0; i < 16; ++i) {
    payloads.push_back(Payload(static_cast<uint8_t>(i)));
  }
  for (int i = 0; i < 16; ++i) {
    jobs.push_back(tier.submit_store(Key(i), payloads[i].data(), kPage));
  }
  for (int64_t j : jobs) {
    CHECK(tier.wait(j));
  }
  CHECK(tier.num_blocks() == 16);

  std::vector<std::vector<uint8_t>> outs(16, std::vector<uint8_t>(kPage, 0));
  jobs.clear();
  for (int i = 0; i < 16; ++i) {
    jobs.push_back(tier.submit_load(Key(i), outs[i].data(), kPage));
  }
  for (size_t i = 0; i < jobs.size(); ++i) {
    CHECK(tier.wait(jobs[i]));
    CHECK(outs[i] == payloads[i]);
  }
  (void)payload;
}

// --- corruption / self-healing ------------------------------------------------

TEST_CASE("a TRUNCATED block file is refused and SELF-HEALED") {
  TempDir dir("truncated");
  const CacheIdentity id = BaseIdentity();
  FileMapper mapper(dir.str(), id);
  const std::string path = mapper.file_name(Key(1));
  {
    FileSystemTier tier(Options(dir, id));
    const auto payload = Payload(0x44);
    tier.store(Key(1), payload.data(), payload.size());
  }
  // Chop the payload in half.
  std::filesystem::resize_file(path, kBlockHeaderBytes + kPage / 2);

  FileSystemTier tier(Options(dir, id));
  std::vector<uint8_t> out(kPage, 0);
  // A short read is an ERROR, never a partial block.
  CHECK_THROWS_AS(tier.load(Key(1), out.data(), out.size()),
                  std::runtime_error);
  // Self-healing: the unreadable file is gone, so the next attempt is a clean
  // miss rather than a repeating throw.
  CHECK_FALSE(std::filesystem::exists(path));
  CHECK_FALSE(tier.load(Key(1), out.data(), out.size()));
}

TEST_CASE("a file with a foreign magic is refused and removed") {
  TempDir dir("magic");
  const CacheIdentity id = BaseIdentity();
  FileMapper mapper(dir.str(), id);
  const std::string path = mapper.file_name(Key(1));
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  {
    std::ofstream out(path, std::ios::binary);
    std::vector<char> junk(kBlockHeaderBytes + kPage, 'Z');
    out.write(junk.data(), static_cast<std::streamsize>(junk.size()));
  }
  FileSystemTier tier(Options(dir, id));
  std::vector<uint8_t> buf(kPage, 0);
  CHECK_THROWS_AS(tier.load(Key(1), buf.data(), buf.size()),
                  std::runtime_error);
  CHECK_FALSE(std::filesystem::exists(path));
}

TEST_CASE("a MISFILED block (right path, wrong key in the header) is refused") {
  TempDir dir("misfiled");
  const CacheIdentity id = BaseIdentity();
  FileMapper mapper(dir.str(), id);
  const auto payload = Payload(0x55);

  // Write a block whose header claims Key(2) into Key(1)'s path.
  BlockFileHeader header;
  header.payload_size = kPage;
  header.identity_digest = id.Digest();
  header.key = Key(2);
  store_block(mapper.file_name(Key(1)), header, payload.data(), payload.size());

  FileSystemTier tier(Options(dir, id));
  std::vector<uint8_t> out(kPage, 0);
  CHECK_THROWS_AS(tier.load(Key(1), out.data(), out.size()),
                  std::runtime_error);
}

// --- THE IDENTITY REFUSAL MATRIX ----------------------------------------------

namespace {

// THE DANGEROUS CASE, staged directly. Write a block under identity
// `written`, then place that exact file where a tier running under `read_as`
// would look for the same key, and try to read it. This is what a shared
// cache directory, a restored backup or a re-pointed --kv-offload-dir does in
// the field.
//
// The directory digest alone does NOT cover this: it merely puts the two runs
// in different folders. The property under test is that even when the bytes
// ARE in front of us, the PER-BLOCK VERIFIED HEADER refuses them. Upstream has
// no equivalent check at all — its config.json is never read, so at this point
// it would hand the engine another model's KV and produce plausible wrong
// tokens.
//
// Returns true when the read REFUSED.
bool RefusesUnderChangedIdentity(const CacheIdentity& written,
                                 const CacheIdentity& read_as,
                                 const TempDir& dir) {
  FileMapper writer_map(dir.str(), written);
  FileMapper reader_map(dir.str(), read_as);
  // A changed field must at minimum change the directory, so the two runs
  // cannot even collide by accident.
  REQUIRE(writer_map.base_path() != reader_map.base_path());

  {
    FileSystemTier tier(Options(dir, written));
    std::vector<uint8_t> payload(written.page_size_bytes, 0x66);
    tier.store(Key(1), payload.data(), payload.size());
  }

  // Stage the foreign block exactly where the reader will look.
  const std::string src = writer_map.file_name(Key(1));
  const std::string dst = reader_map.file_name(Key(1));
  std::filesystem::create_directories(std::filesystem::path(dst).parent_path());
  std::filesystem::copy_file(src, dst,
                             std::filesystem::copy_options::overwrite_existing);

  try {
    FileSystemTier tier(Options(dir, read_as));
    std::vector<uint8_t> out(read_as.page_size_bytes, 0);
    const bool hit = tier.load(Key(1), out.data(), out.size());
    // Reading a foreign block successfully is the failure this whole feature
    // exists to prevent.
    if (hit) {
      return false;
    }
    // A miss would also be safe, but it must not happen here: the file IS
    // present, so a silent miss would mean the check never ran.
    return false;
  } catch (const std::runtime_error&) {
    return true;
  }
}

}  // namespace

TEST_CASE("an UNCHANGED identity loads (the positive control)") {
  // Without this, every refusal below could be an artefact of a check that
  // rejects everything. Same identity, separate tier objects (i.e. a restart):
  // the block loads, byte-exactly.
  TempDir dir("identity_same");
  std::vector<uint8_t> payload(kPage, 0x66);
  {
    FileSystemTier tier(Options(dir, BaseIdentity()));
    tier.store(Key(1), payload.data(), payload.size());
  }
  FileSystemTier reopened(Options(dir, BaseIdentity()));
  std::vector<uint8_t> out(kPage, 0);
  CHECK(reopened.load(Key(1), out.data(), out.size()));
  CHECK(out == payload);
}

TEST_CASE("identity refusal: ONE NEGATIVE CASE PER FIELD") {
  // THE SAFETY GATE. A restored cache opened against a different model, dtype,
  // rope configuration, quantization, block size, spec kind or chain seed must
  // REFUSE — never silently produce plausible wrong tokens. Upstream's digest
  // covers none of the last four.
  struct Case {
    const char* name;
    void (*mutate)(CacheIdentity&);
  };
  const std::vector<Case> cases = {
      {"model_name", [](CacheIdentity& i) { i.model_name = "other/model"; }},
      {"model_type", [](CacheIdentity& i) { i.model_type = "llama"; }},
      {"architectures",
       [](CacheIdentity& i) { i.architectures = {"LlamaForCausalLM"}; }},
      {"hf_config_digest",
       [](CacheIdentity& i) { i.hf_config_digest = "0000"; }},
      {"weight_quantization",
       [](CacheIdentity& i) { i.weight_quantization = "nvfp4"; }},
      {"checkpoint_fingerprint",
       [](CacheIdentity& i) { i.checkpoint_fingerprint = "sha256:bbbb"; }},
      {"num_hidden_layers",
       [](CacheIdentity& i) { i.num_hidden_layers = 48; }},
      {"num_kv_heads", [](CacheIdentity& i) { i.num_kv_heads = 4; }},
      {"head_size", [](CacheIdentity& i) { i.head_size = 64; }},
      {"head_size_v", [](CacheIdentity& i) { i.head_size_v = 64; }},
      {"sliding_window", [](CacheIdentity& i) { i.sliding_window = 4096; }},
      {"rope_config",
       [](CacheIdentity& i) { i.rope_config = "theta=500000,dim=128"; }},
      {"max_position_embeddings",
       [](CacheIdentity& i) { i.max_position_embeddings = 8192; }},
      {"kv_cache_spec_kind",
       [](CacheIdentity& i) { i.kv_cache_spec_kind = "sliding_window"; }},
      {"block_size", [](CacheIdentity& i) { i.block_size = 32; }},
      {"hash_block_size", [](CacheIdentity& i) { i.hash_block_size = 32; }},
      {"kv_dtype", [](CacheIdentity& i) { i.kv_dtype = "fp16"; }},
      {"kv_quant_mode", [](CacheIdentity& i) { i.kv_quant_mode = "fp8"; }},
      {"hash_algo", [](CacheIdentity& i) { i.hash_algo = "sha256"; }},
      {"none_hash_hex",
       [](CacheIdentity& i) { i.none_hash_hex = std::string(64, 'b'); }},
      {"format_version", [](CacheIdentity& i) { i.format_version = 99; }},
      {"inference_engine",
       [](CacheIdentity& i) { i.inference_engine = "vllm"; }},
      {"tp_size", [](CacheIdentity& i) { i.tp_size = 2; }},
      {"pp_size", [](CacheIdentity& i) { i.pp_size = 2; }},
      {"pcp_size", [](CacheIdentity& i) { i.pcp_size = 2; }},
      {"dcp_size", [](CacheIdentity& i) { i.dcp_size = 2; }},
      {"rank", [](CacheIdentity& i) { i.rank = 1; }},
  };

  for (const Case& c : cases) {
    CAPTURE(c.name);
    // 1. FirstMismatch names exactly this field.
    CacheIdentity changed = BaseIdentity();
    c.mutate(changed);
    auto mismatch = CacheIdentity::FirstMismatch(BaseIdentity(), changed);
    REQUIRE(mismatch.has_value());
    CHECK(*mismatch == c.name);

    // 2. And an end-to-end open REFUSES.
    TempDir dir(std::string("identity_") + c.name);
    CHECK(RefusesUnderChangedIdentity(BaseIdentity(), changed, dir));
  }
}

TEST_CASE("page_size_bytes: a changed page size is REFUSED") {
  // Handled separately because the payload length changes with it.
  CacheIdentity changed = BaseIdentity();
  changed.page_size_bytes = kPage * 2;
  auto mismatch = CacheIdentity::FirstMismatch(BaseIdentity(), changed);
  REQUIRE(mismatch.has_value());
  CHECK(*mismatch == "page_size_bytes");
  TempDir dir("identity_page_size");
  CHECK(RefusesUnderChangedIdentity(BaseIdentity(), changed, dir));
}

TEST_CASE("the refusal names the offending field") {
  TempDir dir("named");
  {
    FileSystemTier tier(Options(dir, BaseIdentity()));
  }
  CacheIdentity changed = BaseIdentity();
  changed.kv_dtype = "fp16";
  // Same directory digest is impossible here (the digest covers kv_dtype), so
  // force the collision by writing the changed config into the SAME base path
  // to prove the config.json comparison — not just the path digest — is what
  // refuses.
  FileMapper original(dir.str(), BaseIdentity());
  FileMapper other(dir.str(), changed);
  REQUIRE(original.base_path() != other.base_path());
  // Overwrite the original directory's config.json with the changed identity;
  // re-opening under the ORIGINAL identity must now refuse, naming kv_dtype.
  {
    std::ofstream out(original.config_file_path(), std::ios::trunc);
    out << changed.ToCanonicalJson();
  }
  try {
    FileMapper reopened(dir.str(), BaseIdentity());
    reopened.OpenOrCreate();
    FAIL("expected a refusal");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    CHECK(msg.find("kv_dtype") != std::string::npos);
    CHECK(msg.find("REFUSING") != std::string::npos);
  }
}

// --- the bounded tier (§B3, beyond upstream) ----------------------------------

TEST_CASE("a byte budget is respected, and eviction is transparent") {
  // Upstream's fs tier has NO capacity accounting and NO eviction, so it grows
  // until the filesystem fills. Ours holds a budget; dropping a cached block
  // loses a hit, never changes a token.
  TempDir dir("bounded");
  const int64_t bytes_per_block = kPage + static_cast<int64_t>(kBlockHeaderBytes);
  FileSystemTier tier(Options(dir, BaseIdentity(),
                              /*capacity_bytes=*/4 * bytes_per_block));
  CHECK(tier.capacity_bytes() == 4 * bytes_per_block);

  std::vector<std::vector<uint8_t>> payloads;
  for (int i = 0; i < 10; ++i) {
    payloads.push_back(Payload(static_cast<uint8_t>(i)));
    tier.store(Key(i), payloads[i].data(), kPage);
    CHECK(tier.num_blocks() <= 4);
    CHECK(tier.bytes_used() <= tier.capacity_bytes());
  }
  CHECK(tier.num_evicted() == 6);

  // The MOST RECENT blocks survived; the evicted ones are clean misses, and
  // whatever is still there is byte-exact.
  std::vector<uint8_t> out(kPage, 0);
  CHECK(tier.load(Key(9), out.data(), out.size()));
  CHECK(out == payloads[9]);
  CHECK_FALSE(tier.load(Key(0), out.data(), out.size()));
}

TEST_CASE("a capacity smaller than one block is rejected outright") {
  TempDir dir("tiny");
  CHECK_THROWS_AS(FileSystemTier(Options(dir, BaseIdentity(),
                                         /*capacity_bytes=*/16)),
                  std::runtime_error);
}

TEST_CASE("the budget is honoured ACROSS A RESTART") {
  // The in-memory index is rebuilt from what is on disk, so a restart cannot
  // start from a false zero and blow past the budget.
  TempDir dir("restart_budget");
  const int64_t bytes_per_block = kPage + static_cast<int64_t>(kBlockHeaderBytes);
  const auto opts = [&] {
    return Options(dir, BaseIdentity(), /*capacity_bytes=*/4 * bytes_per_block);
  };
  {
    FileSystemTier tier(opts());
    for (int i = 0; i < 4; ++i) {
      auto p = Payload(static_cast<uint8_t>(i));
      tier.store(Key(i), p.data(), kPage);
    }
    CHECK(tier.num_blocks() == 4);
  }
  FileSystemTier reopened(opts());
  CHECK(reopened.num_blocks() == 4);
  auto p = Payload(0x77);
  reopened.store(Key(100), p.data(), kPage);
  CHECK(reopened.num_blocks() == 4);
  CHECK(reopened.num_evicted() == 1);
}

TEST_CASE("A CACHE WRITTEN BY ONE PROCESS IS HIT BY ANOTHER") {
  // The end-to-end payoff of the deterministic-hash fix. Blocks written under
  // real, chained block hashes are found and read back byte-exactly by a
  // freshly constructed tier — which is what a restart is. Before the
  // determinism fix the keys themselves would have differed and this would
  // score 0%.
  TempDir dir("crossproc");
  vllm::v1::init_none_hash(vllm::v1::sha256_cbor);

  CacheIdentity id = BaseIdentity();
  {
    // Record the chain seed in the identity, exactly as production must.
    std::string hex;
    static const char* kDigits = "0123456789abcdef";
    for (unsigned char c : vllm::v1::NONE_HASH) {
      hex.push_back(kDigits[c >> 4]);
      hex.push_back(kDigits[c & 0x0f]);
    }
    id.none_hash_hex = hex;
  }

  // Real chained block hashes over a fixed corpus.
  std::vector<OffloadKey> keys;
  std::optional<BlockHash> parent;
  for (int b = 0; b < 6; ++b) {
    std::vector<int32_t> tokens;
    for (int i = 0; i < 16; ++i) {
      tokens.push_back(b * 16 + i);
    }
    BlockHash h = vllm::v1::hash_block_tokens(vllm::v1::sha256_cbor, parent,
                                              tokens, std::nullopt);
    keys.push_back(make_offload_key(h, 0));
    parent = h;
  }

  std::vector<std::vector<uint8_t>> payloads;
  for (size_t i = 0; i < keys.size(); ++i) {
    payloads.push_back(Payload(static_cast<uint8_t>(i * 17)));
  }
  {
    FileSystemTier writer(Options(dir, id));
    for (size_t i = 0; i < keys.size(); ++i) {
      writer.store(keys[i], payloads[i].data(), kPage);
    }
  }

  FileSystemTier reader(Options(dir, id));
  const LookupResults found = reader.lookup(keys);
  int hits = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!found[i]) {
      continue;
    }
    std::vector<uint8_t> out(kPage, 0);
    CHECK(reader.load(keys[i], out.data(), out.size()));
    CHECK(out == payloads[i]);
    ++hits;
  }
  MESSAGE("MEASURED disk-tier hit rate on restart: "
          << hits << "/" << keys.size());
  CHECK(hits == static_cast<int>(keys.size()));
}

TEST_CASE("reset_cache drops every file") {
  TempDir dir("reset");
  FileSystemTier tier(Options(dir, BaseIdentity()));
  auto p = Payload(0x88);
  tier.store(Key(1), p.data(), kPage);
  CHECK(tier.num_blocks() == 1);
  tier.reset_cache();
  CHECK(tier.num_blocks() == 0);
  std::vector<uint8_t> out(kPage, 0);
  CHECK_FALSE(tier.load(Key(1), out.data(), out.size()));
}

// --- SKIP-marked, with the owning row -----------------------------------------

TEST_CASE("SlidingWindowSpec persistence" * doctest::skip()) {
  // SKIPPED — KV-SLIDING-WINDOW-SPEC. SlidingWindowSpec is registered
  // (src/vllm/v1/kv_cache_spec_registry.cpp:77) and unit-tested, but NO model
  // in this tree constructs one, so there is no production shape to gate a
  // persistence claim against. Upstream's offload path has real
  // sliding-window-specific behaviour (suffix-group lookup,
  // offloading/scheduler.py:464; test_swa_alignment_skip:1452), all of which
  // becomes live only once a model builds the spec.
  FAIL("unreachable");
}

TEST_CASE("quantized-KV persistence" * doctest::skip()) {
  // SKIPPED — KV-FP8 / KV-NVFP4-TURBO. Quantized KV cannot be persisted today
  // at all: every spec's real_page_size_bytes() THROWS when
  // kv_quant_mode != kNone (src/vllm/v1/kv_cache_interface.cpp:18-20,25,49,64,77),
  // so there is no page to write. The identity records kv_quant_mode precisely
  // so a future FP8/NVFP4 KV cache can never silently read bf16 files — the
  // refusal case for that field IS covered above.
  FAIL("unreachable");
}

TEST_CASE("GDN / Mamba recurrent-state persistence" * doctest::skip()) {
  // SKIPPED — out of scope, by design. GDN state is NOT addressed by a block
  // id: column 0 of a GDN group's block-table row is rewritten to a compact
  // per-request state slot keyed on req_id
  // (src/vllm/v1/worker/gpu/runner.cpp:602-653). Recurrent state is a different
  // persistence problem from paged KV and belongs to KV-MAMBA-ALIGN, not here.
  FAIL("unreachable");
}
