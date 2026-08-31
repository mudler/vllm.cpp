// GLM-5.3 (`GlmMoeDsaForCausalLM`) — G2, the structural loader gate.
//
// Spec `.agents/specs/glm-dsa-latest-deepseek.md` §3.6 G2 and §3.7 W7, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// ─── WHAT THIS SUITE IS FOR ──────────────────────────────────────────────────
// §3.6 states G2 in one sentence: "every tensor in the real `UD-IQ1_S` shards
// is enumerated and accounted: 1809 == 1809, zero unaccounted, and every
// `ggml_type` in the file is one this tree can decode."
//
// The reason it exists as its own gate, rather than being left to the load, is
// that a tensor this port does not CLAIM is weight that is silently absent from
// the model. It is not a crash and it is not a shape error: the layer simply
// runs with whatever the default-constructed operand held, which for a norm is
// zeros and for a projection is an empty buffer the very next call refuses --
// or, worse, for one of the 285 broadcast indexer tensors, is a tensor that
// SHOULD be dropped and would look identical to one that was forgotten. There
// is no end-to-end token gate on this row and there cannot be one on this fleet
// (spec O1: vLLM implements this architecture and cannot fit its 703.74 GiB on
// any device this project reaches), so the accounting is the only thing that
// can see the difference.
//
// ─── WHY IT IS SPLIT IN TWO ──────────────────────────────────────────────────
// The first two cases need no artifact at all. They run the claim-set
// enumerator against the checkpoint's own committed `config.json` and check the
// arithmetic that the third case then confirms against the file. They run in
// CI, on every change, for free.
//
// The third case needs the 201.83 GiB artifact and is skipped unless
// `VT_GLM_DSA_GGUF` names its first shard. It reads only the HEADERS -- 1809
// `tensor_info` records across six shards, about 9.5 MB of the file -- and
// never touches a byte of payload, so it costs seconds rather than the
// half-hour a load costs. That is deliberate: the expensive gate is the load
// itself and it belongs under an `rc` lease, while THIS one is cheap enough to
// run beside every change that touches the name map.
//
// ─── THE NUMBERS ARE MEASURED, NOT QUOTED ────────────────────────────────────
// Every constant below was recomputed from the six published shard headers on
// 2026-08-30 at revision `346b3591c7f28d1a23716f97a065ecf12ec14771`, by parsing
// the GGUF headers directly, and they reproduce §3.4's census exactly: 1809
// tensors, 216,705,819,648 payload bytes = 201.823 GiB, 228 expert towers at
// 187.312 GiB, 1581 resident tensors at 14.511 GiB. The per-type split
// reproduces too, IQ4_XS's four tensors included.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "glm_moe_dsa_config_glm53.inc"

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/glm_moe_dsa.h"
#include "vllm/transformers_utils/hf_config.h"

namespace {

using vllm::GgufFile;
using vllm::GgufTensorInfo;
using vllm::GlmMoeDsaParams;

// The published artifact, re-measured 2026-08-30 from the six shard headers.
constexpr int64_t kFileTensors = 1809;
constexpr int64_t kExpertTowers = 228;
constexpr int64_t kResidentTensors = 1581;
constexpr int64_t kBackboneLayers = 78;
constexpr int64_t kMtpBlockTensors = 27;   // every tensor of `blk.78.`
constexpr int64_t kClaimedTensors = 1782;  // 1809 - 27
constexpr int64_t kFullIndexerLayers = 21;
constexpr int64_t kBroadcastIndexerDropped = 285;  // (78 - 21) * 5
// Payload bytes, expert and resident. The three sum, and they are stated
// separately so a change that moved one would not be absorbed by the total.
constexpr int64_t kExpertBytes = 201125265408;    // 187.312 GiB
constexpr int64_t kResidentBytes = 15580554240;   // 14.511 GiB
constexpr int64_t kTotalBytes = 216705819648;     // 201.823 GiB
// The uniform slot size the host store must be able to hold: the largest
// per-expert slice in the file, which is an IQ4_XS `ffn_down_exps` row block.
// §3.3 predicts 6,684,672 B = 6.375 MiB.
constexpr int64_t kLargestExpertSlice = 6684672;

const char* kStreamedSuffix = "_exps.weight";

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The checkpoint's own config, committed verbatim as
// `glm_moe_dsa_config_glm53.inc` at revision
// `935644c05e76fc198714f4cca449fd8b970ff6d7`.
vllm::HfConfig Glm53Config() {
  vllm::HfConfig c;
  const nlohmann::json j =
      nlohmann::json::parse(glm_moe_dsa_fixture::kGlm53ConfigJson);
  c.model_type = j.at("model_type").get<std::string>();
  c.architectures = j.at("architectures").get<std::vector<std::string>>();
  c.hidden_size = j.at("hidden_size").get<int64_t>();
  c.num_hidden_layers = j.at("num_hidden_layers").get<int64_t>();
  c.num_attention_heads = j.at("num_attention_heads").get<int64_t>();
  c.vocab_size = j.at("vocab_size").get<int64_t>();
  c.intermediate_size = j.at("intermediate_size").get<int64_t>();
  c.rms_norm_eps = j.at("rms_norm_eps").get<double>();
  c.max_position_embeddings = j.at("max_position_embeddings").get<int64_t>();
  c.raw = j;
  return c;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// 1. The claim set, against the checkpoint's own config. No artifact needed.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa G2: the claim set covers the backbone and excludes the MTP block") {
  const GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(Glm53Config());
  REQUIRE(p.num_hidden_layers == kBackboneLayers);
  REQUIRE(vllm::GlmMoeDsaFullIndexerLayerCount(p) == kFullIndexerLayers);

  const std::vector<std::string> claimed =
      vllm::EnumerateGlmMoeDsaGgufTensors(p);

  // The enumerator must not repeat itself: a duplicated name would inflate the
  // claim count and could hide a genuinely unclaimed tensor behind an arithmetic
  // coincidence.
  const std::set<std::string> unique(claimed.begin(), claimed.end());
  CHECK(unique.size() == claimed.size());

  // The headline. 1782 claimed + 27 in the MTP block = the file's 1809.
  CHECK(static_cast<int64_t>(claimed.size()) == kClaimedTensors);
  CHECK(kClaimedTensors + kMtpBlockTensors == kFileTensors);

  // Nothing claimed may name a block past the backbone. `block_count` is 79 and
  // `num_hidden_layers` is 78; claiming `blk.78.*` would build a 79th decoder
  // layer out of the multi-token-prediction block, and nothing downstream would
  // refuse it.
  for (const std::string& n : claimed) {
    CHECK(n.rfind("blk.78.", 0) != 0);
    CHECK(n.find("nextn") == std::string::npos);
  }

  // The three towers are claimed on every MoE layer and on no dense one.
  int64_t towers = 0;
  int64_t dense_mlp = 0;
  for (const std::string& n : claimed) {
    if (EndsWith(n, kStreamedSuffix)) ++towers;
    if (EndsWith(n, ".ffn_gate.weight")) ++dense_mlp;
  }
  // 75 MoE backbone layers x 3. The file's 228 includes the MTP block's three,
  // which this port drops -- so the claim is 225 and the difference is exactly
  // the tail.
  CHECK(towers == 225);
  CHECK(towers + 3 == kExpertTowers);
  CHECK(dense_mlp == p.first_k_dense_replace);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Spec D3, made arithmetic. No artifact needed.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa G2: the broadcast indexer surplus is 57 layers x 5 tensors") {
  const GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(Glm53Config());
  const int64_t full = vllm::GlmMoeDsaFullIndexerLayerCount(p);
  const int64_t shared = p.num_hidden_layers - full;
  CHECK(shared == 57);
  CHECK(shared * 5 == kBroadcastIndexerDropped);

  // The claim set still names all five on every backbone block, because the
  // published file SHIPS them on every block and an unclaimed tensor is an
  // accounting failure. Dropping them is a decision the loader records, not an
  // omission from the claim.
  const std::vector<std::string> claimed =
      vllm::EnumerateGlmMoeDsaGgufTensors(p);
  int64_t indexer_named = 0;
  for (const std::string& n : claimed)
    if (n.find(".indexer.") != std::string::npos) ++indexer_named;
  CHECK(indexer_named == p.num_hidden_layers * 5);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. The real shard headers. Skipped unless the artifact is staged.
//
//    VT_GLM_DSA_GGUF=/path/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf
//
//    `GgufFile::Open` is split-aware: naming shard 1 opens all six and merges
//    their tensor tables, which is what makes 1809 the number this case sees.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa G2: the real UD-IQ1_S shards are fully enumerated and accounted" *
          doctest::skip(std::getenv("VT_GLM_DSA_GGUF") == nullptr)) {
  const char* path = std::getenv("VT_GLM_DSA_GGUF");
  REQUIRE(path != nullptr);
  const GgufFile g = GgufFile::Open(path);

  // (a) The file is the one this gate was measured against. A different arm of
  //     the same repository has a different census, and reporting THIS case
  //     green on THAT file would be a measurement of some other artifact.
  const vllm::GgufValue* arch = g.FindKv("general.architecture");
  REQUIRE(arch != nullptr);
  CHECK(std::get<std::string>(arch->v) == std::string("glm-dsa"));
  CHECK(static_cast<int64_t>(g.Tensors().size()) == kFileTensors);

  // (b) Every ggml type in the file is one this tree can SIZE. `GgufFile::Open`
  //     already throws "gguf: unknown ggml type id N" at file open for a type
  //     without traits, so reaching this line proves it for all 1809; the census
  //     below is what proves we reached the types we expect rather than some
  //     other arm's.
  std::map<std::string, int64_t> by_type;
  int64_t expert_tensors = 0, resident_tensors = 0;
  int64_t expert_bytes = 0, resident_bytes = 0, total_bytes = 0;
  int64_t largest_slice = 0;
  for (const GgufTensorInfo& t : g.Tensors()) {
    const vllm::GgmlTypeTraits& tr = vllm::GgmlTraits(t.ggml_type);
    by_type[tr.name] += 1;
    total_bytes += static_cast<int64_t>(t.nbytes);
    if (EndsWith(t.name, kStreamedSuffix)) {
      ++expert_tensors;
      expert_bytes += static_cast<int64_t>(t.nbytes);
      // A stacked tower is [E, out, in]; one expert is `out` whole rows.
      REQUIRE(t.shape.size() == 3);
      const int64_t per_expert = static_cast<int64_t>(t.nbytes) / t.shape[0];
      largest_slice = std::max(largest_slice, per_expert);
    } else {
      ++resident_tensors;
      resident_bytes += static_cast<int64_t>(t.nbytes);
    }
  }

  // (c) The two classes, and the bytes each carries. This is §3.3's residency
  //     plan checked against the file rather than against the spec.
  CHECK(expert_tensors == kExpertTowers);
  CHECK(resident_tensors == kResidentTensors);
  CHECK(expert_bytes == kExpertBytes);
  CHECK(resident_bytes == kResidentBytes);
  CHECK(total_bytes == kTotalBytes);
  CHECK(expert_bytes + resident_bytes == total_bytes);

  // (d) The per-type census, §3.4's table. IQ4_XS's four tensors are named
  //     explicitly: they are the four `blk.{8,75,76,77}.ffn_down_exps.weight`
  //     towers whose missing keep-quant `vec_dot` blocked this whole row until
  //     `2e9f4d88d` (spec O2), and they are the largest slice in the file, so
  //     they set the uniform slot size for all 1800-odd slots.
  CHECK(by_type["IQ3_XXS"] == 71);
  CHECK(by_type["IQ1_S"] == 106);
  CHECK(by_type["IQ2_XXS"] == 44);
  CHECK(by_type["Q5_K"] == 312);
  CHECK(by_type["IQ4_XS"] == 4);
  CHECK(by_type["Q8_0"] == 476);
  CHECK(by_type["Q2_K"] == 2);
  CHECK(by_type["Q3_K"] == 1);
  CHECK(by_type["Q6_K"] == 82);
  CHECK(by_type["Q4_K"] == 2);
  CHECK(by_type["F32"] == 709);
  // No type outside that list. A new one would be a re-quantized artifact under
  // an unchanged name, which is precisely the thing a repo id alone cannot pin.
  CHECK(by_type.size() == 11);

  CHECK(largest_slice == kLargestExpertSlice);

  // (e) THE GATE ITSELF: every tensor in the file is claimed by this port, or
  //     belongs to the MTP block it deliberately drops. Zero unaccounted.
  const GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(Glm53Config());
  const std::vector<std::string> claimed =
      vllm::EnumerateGlmMoeDsaGgufTensors(p);
  const std::set<std::string> claim_set(claimed.begin(), claimed.end());

  std::vector<std::string> unaccounted;
  int64_t accounted = 0, mtp = 0;
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name.rfind("blk.78.", 0) == 0) {
      ++mtp;
      continue;
    }
    if (claim_set.count(t.name) != 0) {
      ++accounted;
      continue;
    }
    unaccounted.push_back(t.name);
  }
  for (const std::string& n : unaccounted) INFO("unaccounted: " << n);
  CHECK(unaccounted.empty());
  CHECK(accounted == kClaimedTensors);
  CHECK(mtp == kMtpBlockTensors);
  CHECK(accounted + mtp == kFileTensors);

  // (f) And the converse: every name this port claims exists in the file. The
  //     two directions are different failures. (e) catches weight that is
  //     silently absent from the model; this catches a claim the loader would
  //     then throw on at `GgufFile::Get`, which is a refusal rather than a
  //     wrong answer but is still better found here than 20 minutes into a
  //     201.83 GiB load under a lease.
  std::vector<std::string> missing;
  std::set<std::string> file_names;
  for (const GgufTensorInfo& t : g.Tensors()) file_names.insert(t.name);
  for (const std::string& n : claimed)
    if (file_names.count(n) == 0) missing.push_back(n);
  for (const std::string& n : missing) INFO("claimed but absent: " << n);
  CHECK(missing.empty());
}
