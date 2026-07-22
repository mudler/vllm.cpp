// Ported from / grounded in: vllm/v1/core/kv_cache_utils.py:99-114 @ e24d1b24
// and vllm/docs/features/kv_offloading_usage.md:115-121 (the cross-process
// sharing requirement).
//
// THE GATE: block hashes must be identical across SEPARATELY LAUNCHED
// PROCESSES. That is the whole point — an in-process loop proves nothing,
// because the defect being fixed was a per-PROCESS random seed
// (std::random_device at kv_cache_utils.cpp) that every in-process comparison
// would happily agree with.
//
// This test therefore RE-EXECS ITSELF via /proc/self/exe with a doctest
// --test-case filter, captures the child's printed hash chain, and compares it
// against a second, independently launched child. Any per-process randomness in
// the seed makes the two differ.
#include <doctest/doctest.h>

#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/request.h"

using vllm::v1::BlockHash;
using vllm::v1::hash_block_tokens;
using vllm::v1::init_none_hash;
using vllm::v1::NONE_HASH;
using vllm::v1::none_hash_provenance;
using vllm::v1::NoneHashSeedSource;
using vllm::v1::sha256_cbor;

namespace {

std::string ToHex(const std::string& raw) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(raw.size() * 2);
  for (unsigned char c : raw) {
    out.push_back(kDigits[c >> 4]);
    out.push_back(kDigits[c & 0x0f]);
  }
  return out;
}

// A short CHAINED hash sequence over a fixed corpus. Chaining matters: the
// defect propagates from NONE_HASH through every subsequent block, so a chain
// is a strictly stronger probe than one hash.
std::string HashChainHex() {
  std::string out = ToHex(NONE_HASH);
  std::vector<int32_t> tokens;
  std::optional<BlockHash> parent;
  for (int block = 0; block < 4; ++block) {
    tokens.clear();
    for (int i = 0; i < 16; ++i) {
      tokens.push_back(block * 16 + i);
    }
    BlockHash h = hash_block_tokens(sha256_cbor, parent, tokens, std::nullopt);
    out += ":" + ToHex(h);
    parent = h;
  }
  return out;
}

// Run this same test binary again, in a fresh process, with `env_prefix`
// prepended, and return the marker line the child printed.
std::string RunChild(const std::string& env_prefix,
                     const std::string& test_case) {
  // Resolve our own path HERE, in the parent: popen runs the command under
  // /bin/sh, so a literal /proc/self/exe inside it would resolve to the shell.
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';
  // --no-skip is required: the child case is skip-decorated so a normal run
  // never executes it.
  const std::string cmd = env_prefix + " " + std::string(exe) +
                          " --no-skip --test-case='" + test_case +
                          "' 2>/dev/null";
  FILE* pipe = ::popen(cmd.c_str(), "r");
  REQUIRE(pipe != nullptr);
  std::string output;
  std::array<char, 4096> buf{};
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) !=
         nullptr) {
    output += buf.data();
  }
  ::pclose(pipe);

  // Extract the marker line.
  const std::string marker = "HASHCHAIN=";
  const size_t at = output.find(marker);
  INFO("child output: " << output);
  REQUIRE(at != std::string::npos);
  const size_t start = at + marker.size();
  const size_t end = output.find('\n', start);
  return output.substr(start, end == std::string::npos ? end : end - start);
}

}  // namespace

// The CHILD case. It is filtered out of a normal run (doctest skips
// `* [.]`-decorated cases unless explicitly named), so it only executes when a
// parent re-execs it by name.
TEST_CASE("none_hash_child_emits_chain" * doctest::skip()) {
  init_none_hash(sha256_cbor);
  std::printf("HASHCHAIN=%s\n", HashChainHex().c_str());
  std::fflush(stdout);
}

TEST_CASE("init_none_hash is deterministic by DEFAULT (no configuration)") {
  init_none_hash(sha256_cbor);
  CHECK(none_hash_provenance().source == NoneHashSeedSource::kDefault);
  CHECK(none_hash_provenance().seed ==
        std::string(vllm::v1::kDefaultNoneHashSeed));
  // The seeded path hashes the seed text, so NONE_HASH is a 32-byte digest —
  // never the 32 raw random bytes the old unseeded branch produced.
  CHECK(NONE_HASH.size() == 32);
  CHECK(NONE_HASH == sha256_cbor(vllm::v1::CborValue::Text(
                         std::string(vllm::v1::kDefaultNoneHashSeed))));
}

// THE BINDING GATE for KV persistence: two independently launched PROCESSES,
// zero configuration, byte-identical hash chains. Before this change each
// process filled NONE_HASH from std::random_device, so this comparison failed
// and any content-addressed disk tier scored 0% on restart.
TEST_CASE("block hashes are byte-identical across SEPARATE processes") {
  const std::string a = RunChild("", "none_hash_child_emits_chain");
  const std::string b = RunChild("", "none_hash_child_emits_chain");
  CHECK_FALSE(a.empty());
  CHECK(a == b);
}

// The upstream escape hatch, mirrored: PYTHONHASHSEED still works, and a
// DIFFERENT value must produce a DIFFERENT chain (otherwise the seed is being
// ignored).
TEST_CASE("PYTHONHASHSEED is honoured across processes, and changes the chain") {
  const std::string zero_a =
      RunChild("PYTHONHASHSEED=0", "none_hash_child_emits_chain");
  const std::string zero_b =
      RunChild("PYTHONHASHSEED=0", "none_hash_child_emits_chain");
  const std::string one =
      RunChild("PYTHONHASHSEED=1", "none_hash_child_emits_chain");
  CHECK(zero_a == zero_b);
  CHECK(zero_a != one);
  // And a seeded run differs from the unconfigured default.
  const std::string dflt = RunChild("", "none_hash_child_emits_chain");
  CHECK(zero_a != dflt);
}

// Our own escape hatch takes PRIORITY over PYTHONHASHSEED.
TEST_CASE("VLLM_PREFIX_CACHING_HASH_SEED overrides PYTHONHASHSEED") {
  const std::string ours = RunChild(
      "VLLM_PREFIX_CACHING_HASH_SEED=abc PYTHONHASHSEED=0",
      "none_hash_child_emits_chain");
  const std::string ours_only = RunChild("VLLM_PREFIX_CACHING_HASH_SEED=abc",
                                         "none_hash_child_emits_chain");
  const std::string python_only =
      RunChild("PYTHONHASHSEED=0", "none_hash_child_emits_chain");
  CHECK(ours == ours_only);
  CHECK(ours != python_only);
}

// Upstream's os.urandom(32) behaviour remains reachable, as an OPT-IN — and it
// must genuinely be random, i.e. two processes must DISAGREE. This is the
// negative control that proves the default really is doing something.
TEST_CASE("VLLM_PREFIX_CACHING_HASH_SEED=random restores per-process randomness") {
  const std::string a = RunChild("VLLM_PREFIX_CACHING_HASH_SEED=random",
                                 "none_hash_child_emits_chain");
  const std::string b = RunChild("VLLM_PREFIX_CACHING_HASH_SEED=random",
                                 "none_hash_child_emits_chain");
  CHECK(a != b);
}

TEST_CASE("an explicit seed argument still wins, and is recorded as such") {
  init_none_hash(sha256_cbor, std::string("explicit-seed"));
  CHECK(none_hash_provenance().source == NoneHashSeedSource::kExplicit);
  CHECK(none_hash_provenance().seed == "explicit-seed");
  const BlockHash with_explicit = NONE_HASH;
  init_none_hash(sha256_cbor);
  CHECK(NONE_HASH != with_explicit);
  // Restore the default for any test that runs after this one in-process.
  CHECK(none_hash_provenance().source == NoneHashSeedSource::kDefault);
}

TEST_CASE("an empty env value counts as UNSET (not as an empty seed)") {
  // An exported-but-empty variable is the same "not configured" state as an
  // absent one; treating "" as a seed would silently give a third chain.
  const std::string empty_env =
      RunChild("VLLM_PREFIX_CACHING_HASH_SEED=", "none_hash_child_emits_chain");
  const std::string unset = RunChild("", "none_hash_child_emits_chain");
  CHECK(empty_env == unset);
}
