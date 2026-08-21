// `QUANT-QWEN38-27B-NVFP4-ARM` W5 — the LOAD side of the SECOND 27B NVFP4
// artifact: `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`
// @ `36f717a22990e82c54c1d48ee77c491b87825680`.
// Issue https://github.com/mudler/vllm.cpp/issues/821, campaign
// https://github.com/mudler/vllm.cpp/issues/1574, spec
// `.agents/specs/qwen38-27b-quant-arms.md`.
//
// WHY A SECOND FILE, AND NOT A WIDENING OF W4's GATE. W4 gates
// `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d...`: a COMPRESSED-TENSORS
// `mixed-precision` checkpoint, 1968 index names, `config_groups` with regex
// `targets` and a 303-entry `ignore`, a per-CHANNEL FP8 weight scale, DYNAMIC
// per-token activations, an FP8 `lm_head`, and zero `input_scale`. This one
// shares the model and nothing else:
//
//   quant_method        modelopt          (not compressed-tensors)
//   quant_algo          MIXED_PRECISION   (not format "mixed-precision")
//   membership          `quantized_layers`, 401 EXACT module names
//   ignore              EMPTY — the vision tower and the MTP head are
//                       unquantized because nothing NAMES them, which is a
//                       different fact from being excluded
//   FP8 half            208 modules, weight_scale F32 [] and input_scale
//                       F32 [] — per-TENSOR and STATIC
//   NVFP4 half          193 modules, W4A16_NVFP4 group_size 16, weight-only:
//                       weight U8 + weight_scale F8_E4M3 + weight_scale_2 F32
//   lm_head             W4A16_NVFP4 (unsloth's is FP8)
//   MLP boundary        NONE — all 64 layers are NVFP4 (unsloth splits at 56)
//   index               2001 names over FOUR shards (unsloth: 1968 over two)
//
// A gate that accepted both name sets by loosening would have stopped measuring
// either, so W4's 1968-name accounting is untouched and this file carries this
// artifact's own numbers, derived from its own four headers.
//
// The four things gated here:
//
//   (1) THE NAME INDEX IS ACCOUNTED, BOTH DIRECTIONS. All 2001 names of the
//       shipped `model.safetensors.index.json` weight map classify into exactly
//       one bucket, nothing is unaccounted, and the buckets sum to 2001. A
//       tensor family nobody classified is the shape that goes missing
//       silently.
//   (2) THE PER-SCHEME COMPOSITION is the checkpoint's own: 256 FP8 modules /
//       720 tensors, 193 W4A16_NVFP4 modules / 579 tensors, 536 unlisted
//       modules / 702 tensors. Reading this checkpoint as uniformly NVFP4 — its
//       repo name says NVFP4 — is numerically plausible and token-invisible
//       while moving the wrong bytes. The FP8 side is 256 and not the 208 the
//       map names, because the 48 `...layers.<i>.linear_attn` CONTAINERS
//       resolve to their first quantized child through upstream's strategy-3
//       prefix scan and carry two tensors each; the case below asserts the 208
//       and the 48 SEPARATELY, so no count here is right for the wrong reason.
//   (3) THE PRODUCTION LOADER READS THE CONFIG. `LoadQwen3_5Dense` routes each
//       projection by probing tensor NAMES. That probe cannot be wrong about
//       the bytes and it can be wrong about the CHECKPOINT, so the declared
//       algorithm is now cross-checked against the shipped spelling and a
//       disagreement is refused BY NAME, in both directions.
//   (4) A DECLARED ALGORITHM WITH NO LOADER IS REFUSED BY NAME rather than read
//       as something plausible.
//
// HERMETIC. Every case reads the four COMMITTED manifests and the two COMMITTED
// config documents, so CI needs no NAS file and no network. The live arm at the
// bottom is env-gated and SKIPS LOUDLY when its variable is unset.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/quantization/modelopt_mixed_precision.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"

#include "qwen38_27b_modelopt_mtp_s1_manifest.inc"
#include "qwen38_27b_modelopt_mtp_s2_manifest.inc"
#include "qwen38_27b_modelopt_mtp_s3_manifest.inc"
#include "qwen38_27b_modelopt_mtp_s4_manifest.inc"

namespace mo = vllm::layers::modelopt;

namespace {

// ── The artifact, restated from its own bytes ────────────────────────────────
//
// Every literal here was derived by range-reading each shard's OWN safetensors
// header, not copied from a record of it. For each shard
// `8 + header_len + max(data_offsets[1])` equals the size the hub reports,
// which is the semantic verification `AGENTS.md` requires instead of a remote
// hash. A locally computed sha256 is UNPAID: these bytes are not mirrored where
// this session can reach them, and `## Owed` in the spec says so.
struct Shard {
  const char* file;
  int64_t bytes;
  int64_t header_len;
  int tensors;
};

constexpr Shard kShards[] = {
    {"model-00001-of-00004.safetensors", 9965644108LL, 118392, 970},
    {"model-00002-of-00004.safetensors", 9985743924LL, 122832, 976},
    {"model-00003-of-00004.safetensors", 1120886516LL, 4784, 40},
    {"model-00004-of-00004.safetensors", 849400592LL, 1800, 15},
};

// `metadata.total_size` of the shipped `model.safetensors.index.json`. It is
// the sum of the TENSOR PAYLOADS, not of the file sizes: the difference is
// exactly the four headers and their four 8-byte length prefixes, which is
// asserted below rather than asserted away.
constexpr int64_t kIndexTotalSize = 21921427300LL;
constexpr int64_t kIndexNameCount = 2001;

std::string FixtureDir() {
#ifdef QWEN38_27B_MODELOPT_MTP_FIXTURE_DIR
  return QWEN38_27B_MODELOPT_MTP_FIXTURE_DIR;
#else
  return "tests/vllm/models/fixtures/qwen38_27b_modelopt_mtp";
#endif
}

const nlohmann::json& Fixture(const char* file) {
  static std::map<std::string, nlohmann::json>* cache =
      new std::map<std::string, nlohmann::json>();
  auto it = cache->find(file);
  if (it != cache->end()) return it->second;
  std::ifstream in(FixtureDir() + "/" + file);
  REQUIRE_MESSAGE(in.good(), "cannot open the committed fixture " << file);
  nlohmann::json j;
  in >> j;
  return cache->emplace(file, std::move(j)).first->second;
}

const nlohmann::json& ReleasedConfig() { return Fixture("config.json"); }
const nlohmann::json& ReleasedQuant() {
  return ReleasedConfig().at("quantization_config");
}
const nlohmann::json& ReleasedHfQuantConfig() {
  return Fixture("hf_quant_config.json");
}

const mo::MixedPrecisionConfig& ReleasedMixed() {
  static const mo::MixedPrecisionConfig* c = new mo::MixedPrecisionConfig(
      mo::MixedPrecisionConfig::Parse(ReleasedQuant()));
  return *c;
}

struct ManifestTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  const char* shard;
};

template <class T>
void Append(std::vector<ManifestTensor>* out, const T* rows, std::size_t n,
            const char* shard) {
  for (std::size_t i = 0; i < n; ++i) {
    ManifestTensor m;
    m.name = rows[i].name;
    m.dtype = rows[i].dtype;
    for (int d = 0; d < rows[i].rank; ++d) m.shape.push_back(rows[i].shape[d]);
    m.shard = shard;
    out->push_back(std::move(m));
  }
}

// The row count is DERIVED from the array, never read from the `...TensorCount`
// literal beside it. Both live in the same generated `.inc`, and the literal is
// hand-maintained: iterating a count LARGER than the array is an out-of-bounds
// read, which is a SIGSEGV inside case 1 that aborts the binary with the other
// cases never run — a crash reads as neither a pass nor a fail. Deriving makes
// that drift an ordinary red instead, and case 1 asserts the two agree so the
// literal is still measured rather than merely bypassed.
const std::vector<ManifestTensor>& Manifest() {
  static const std::vector<ManifestTensor>* all = [] {
    auto* v = new std::vector<ManifestTensor>();
    Append(v, vllm_test::kQwen38_27bModeloptMtpS1Tensors,
           std::size(vllm_test::kQwen38_27bModeloptMtpS1Tensors),
           kShards[0].file);
    Append(v, vllm_test::kQwen38_27bModeloptMtpS2Tensors,
           std::size(vllm_test::kQwen38_27bModeloptMtpS2Tensors),
           kShards[1].file);
    Append(v, vllm_test::kQwen38_27bModeloptMtpS3Tensors,
           std::size(vllm_test::kQwen38_27bModeloptMtpS3Tensors),
           kShards[2].file);
    Append(v, vllm_test::kQwen38_27bModeloptMtpS4Tensors,
           std::size(vllm_test::kQwen38_27bModeloptMtpS4Tensors),
           kShards[3].file);
    return v;
  }();
  return *all;
}

// The split is the PRODUCTION one. A suffix it does not know is a defect this
// gate must SEE rather than one it reproduces with a private copy.
using mo::SplitOperand;

// The bucket a NAME lands in: the `quant_algo` its module resolves to through
// the shipped `quantized_layers` map, "KV_CACHE_SCALE" for the two operands
// that belong to the KV cache rather than to any Linear, and "UNCLASSIFIED" for
// a name whose family the resolver has never seen.
std::string BucketOf(const std::string& name) {
  std::string module;
  std::string suffix;
  if (!SplitOperand(name, &module, &suffix)) return "UNCLASSIFIED";
  if (mo::IsKvCacheScaleSuffix(suffix)) return "KV_CACHE_SCALE";
  const mo::ModuleQuant q = ReleasedMixed().Resolve(module);
  return q.Quantized() ? mo::QuantAlgoName(q.algo) : std::string("UNLISTED");
}

bool Names(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

// ── (1) the committed manifests ARE the shipped index ───────────────────────
TEST_CASE("Qwen3.8-27B-NVFP4-MTP: the four committed manifests account for all 2001 index names") {
  CHECK(vllm_test::kQwen38_27bModeloptMtpS1TensorCount == kShards[0].tensors);
  CHECK(vllm_test::kQwen38_27bModeloptMtpS2TensorCount == kShards[1].tensors);
  CHECK(vllm_test::kQwen38_27bModeloptMtpS3TensorCount == kShards[2].tensors);
  CHECK(vllm_test::kQwen38_27bModeloptMtpS4TensorCount == kShards[3].tensors);
  // ... and the hand-maintained literal agrees with the array it is written
  // beside. `Manifest()` derives the row count from the array, so a literal
  // that drifted upward no longer reads out of bounds; this is what still
  // MEASURES the literal instead of routing around it.
  CHECK(vllm_test::kQwen38_27bModeloptMtpS1TensorCount ==
        static_cast<int64_t>(std::size(vllm_test::kQwen38_27bModeloptMtpS1Tensors)));
  CHECK(vllm_test::kQwen38_27bModeloptMtpS2TensorCount ==
        static_cast<int64_t>(std::size(vllm_test::kQwen38_27bModeloptMtpS2Tensors)));
  CHECK(vllm_test::kQwen38_27bModeloptMtpS3TensorCount ==
        static_cast<int64_t>(std::size(vllm_test::kQwen38_27bModeloptMtpS3Tensors)));
  CHECK(vllm_test::kQwen38_27bModeloptMtpS4TensorCount ==
        static_cast<int64_t>(std::size(vllm_test::kQwen38_27bModeloptMtpS4Tensors)));
  REQUIRE(Manifest().size() == static_cast<std::size_t>(kIndexNameCount));

  // Every name distinct, across ALL FOUR shards. A weight map cannot name one
  // tensor twice, and a duplicated manifest row would inflate every count below
  // without failing any of them.
  std::set<std::string> unique;
  for (const ManifestTensor& t : Manifest()) unique.insert(t.name);
  CHECK(unique.size() == static_cast<std::size_t>(kIndexNameCount));

  // The whole-checkpoint dtype histogram. The 609 F32 are 208 `input_scale` +
  // 208 FP8 `weight_scale` + 193 `weight_scale_2`, every one of them a SCALAR
  // — which is the fact that makes this artifact loadable where the sibling's
  // per-channel BF16 scale is not.
  std::map<std::string, int> dtypes;
  for (const ManifestTensor& t : Manifest()) dtypes[t.dtype] += 1;
  CHECK(dtypes["F32"] == 609);
  CHECK(dtypes["BF16"] == 798);
  CHECK(dtypes["F8_E4M3"] == 401);
  CHECK(dtypes["U8"] == 193);
  CHECK(dtypes.size() == 4);

  // Every name carries a recognised operand suffix. A name that does not is not
  // "other": it is a family the production splitter has never seen, and reading
  // it as unquantized is the silent-dequant direction.
  std::vector<std::string> unclassified;
  for (const ManifestTensor& t : Manifest()) {
    std::string module;
    std::string suffix;
    if (!SplitOperand(t.name, &module, &suffix)) unclassified.push_back(t.name);
  }
  INFO("unclassified names: " << unclassified.size());
  CHECK(unclassified.empty());

  // `metadata.total_size` is the sum of the tensor PAYLOADS, so the four file
  // sizes exceed it by exactly the four headers plus their 8-byte prefixes.
  int64_t file_bytes = 0;
  int64_t header_bytes = 0;
  int tensor_total = 0;
  for (const Shard& s : kShards) {
    file_bytes += s.bytes;
    header_bytes += 8 + s.header_len;
    tensor_total += s.tensors;
  }
  CHECK(tensor_total == kIndexNameCount);
  CHECK(file_bytes - header_bytes == kIndexTotalSize);
}

// ── (2) the per-scheme composition ──────────────────────────────────────────
TEST_CASE("Qwen3.8-27B-NVFP4-MTP: every shipped name resolves to exactly one ModelOpt scheme") {
  const nlohmann::json& q = ReleasedQuant();
  // The SELECTION hook, which is what tells a ModelOpt `quantization_config`
  // apart from a compressed-tensors one sitting in the same field.
  REQUIRE(mo::MixedPrecisionConfig::IsMixedPrecision(q));
  CHECK(q.at("quant_method").get<std::string>() == "modelopt");
  CHECK(q.at("quant_algo").get<std::string>() == "MIXED_PRECISION");
  CHECK(q.at("producer").at("name").get<std::string>() == "modelopt");

  const mo::MixedPrecisionConfig& c = ReleasedMixed();
  CHECK(c.num_quantized_layers() == 401);
  // The `ignore` list is EMPTY. Nothing here is excluded; the vision tower, the
  // MTP head, the norms and the embeddings are unquantized because no entry
  // NAMES them, which resolves through a different code path than an exclusion
  // and is a different statement about the checkpoint.
  CHECK(c.exclude_modules().empty());
  // No entry disagrees about the block size, so the seeding order upstream's
  // `_from_config` depends on (modelopt.py:2365-2377) cannot change the answer
  // here — asserted rather than assumed, because the seed is order-sensitive.
  CHECK(c.group_size() == 16);

  std::map<std::string, int> tensors;
  std::map<std::string, std::set<std::string>> modules;
  for (const ManifestTensor& t : Manifest()) {
    const std::string bucket = BucketOf(t.name);
    tensors[bucket] += 1;
    std::string module;
    std::string suffix;
    if (SplitOperand(t.name, &module, &suffix)) modules[bucket].insert(module);
  }

  // FP8 W8A8, per-tensor and STATIC: 48*3 GDN projections (in_proj_qkv,
  // in_proj_z, out_proj) + 16*4 attention projections = 208 Linears the map
  // NAMES, plus the 48 `...layers.<i>.linear_attn` CONTAINERS that own no
  // Linear weight and resolve to their first quantized child through upstream's
  // strategy-3 prefix scan (modelopt.py:2449-2453). 256 modules and
  // 208*3 + 48*2 = 720 tensors, the two extra per container being `A_log` and
  // `dt_bias`. The 48 are separated below rather than averaged into the 208,
  // because a count that is right for the wrong reason is the failure this
  // whole file exists to prevent.
  CHECK(modules["FP8"].size() == 256);
  CHECK(tensors["FP8"] == 720);
  // W4A16_NVFP4 group_size 16, WEIGHT-ONLY: 64*3 MLP projections + lm_head =
  // 193 modules, each shipping weight + weight_scale + weight_scale_2 = 579
  // tensors. Note the 64: unlike the compressed-tensors sibling there is NO
  // layer-56 boundary, so a number carried over from that artifact is wrong
  // here. No container resolves here, because no `mlp` module owns an operand
  // of its own.
  CHECK(modules["W4A16_NVFP4"].size() == 193);
  CHECK(tensors["W4A16_NVFP4"] == 579);
  // Named by no `quantized_layers` entry and reached by no prefix scan: the
  // norms, conv1d, the embedding table, the 27 vision blocks and the merger,
  // and the whole 15-tensor MTP head.
  CHECK(modules["UNLISTED"].size() == 536);
  CHECK(tensors["UNLISTED"] == 702);
  // This artifact ships NO kv-cache scales at all, which is the half of the FP8
  // KV story `KV-FP8` (#1593) owns; the other half is asserted below.
  CHECK(tensors.count("KV_CACHE_SCALE") == 0);

  // `std::map::operator[]` INSERTS, so the absence assertion is a `count`:
  // reading `tensors["UNCLASSIFIED"]` would create the bucket it is asserting
  // does not exist, and the size check below would then read one more.
  CHECK(tensors.count("UNCLASSIFIED") == 0);
  int sum = 0;
  for (const auto& kv : tensors) sum += kv.second;
  CHECK(sum == kIndexNameCount);
  CHECK(tensors.size() == 3);

  int all_modules = 0;
  for (const auto& kv : modules) all_modules += static_cast<int>(kv.second.size());
  CHECK(all_modules == 985);

  // The split the 256 hides, and the one the loader's cross-check actually
  // guards. A WEIGHT-BEARING module is one that ships `weight` or
  // `weight_packed`; there are exactly 937 of them, one per `.weight` name, and
  // they are the Linears `quantized_layers` is written against. The other 48
  // are containers, and `Resolution` says so: `kDirect` for a module the map
  // NAMES, `kPrefix` for one reached by the scan.
  std::map<std::string, int> weight_bearing;
  int direct = 0;
  int prefix = 0;
  int containers = 0;
  for (const auto& kv : modules) {
    for (const std::string& m : kv.second) {
      const mo::ModuleQuant r = ReleasedMixed().Resolve(m);
      if (r.how == mo::Resolution::kDirect) ++direct;
      if (r.how == mo::Resolution::kPrefix) ++prefix;
      const bool has_weight =
          std::any_of(Manifest().begin(), Manifest().end(),
                      [&m](const ManifestTensor& t) {
                        return t.name == m + ".weight" ||
                               t.name == m + ".weight_packed";
                      });
      if (has_weight) {
        weight_bearing[kv.first] += 1;
      } else {
        ++containers;
      }
    }
  }
  CHECK(direct == 401);
  CHECK(prefix == 48);
  CHECK(containers == 48);
  CHECK(weight_bearing["FP8"] == 208);
  CHECK(weight_bearing["W4A16_NVFP4"] == 193);
  CHECK(weight_bearing["UNLISTED"] == 536);
  CHECK(weight_bearing["FP8"] + weight_bearing["W4A16_NVFP4"] +
            weight_bearing["UNLISTED"] ==
        937);
}

// ── the module families, so a count cannot be right for the wrong reason ────
TEST_CASE("Qwen3.8-27B-NVFP4-MTP: the FP8 and NVFP4 halves are the towers the config names") {
  const mo::MixedPrecisionConfig& c = ReleasedMixed();
  const std::string p = "model.language_model.layers.";

  // Every one of the 64 layers' MLP is NVFP4, INCLUDING 56-63 — the layers the
  // compressed-tensors sibling puts on the FP8 side.
  for (int l : {0, 1, 27, 55, 56, 60, 63}) {
    for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
      CAPTURE(l);
      CAPTURE(proj);
      const mo::ModuleQuant q =
          c.Resolve(p + std::to_string(l) + ".mlp." + proj);
      CHECK(q.algo == mo::QuantAlgo::kW4A16Nvfp4);
      CHECK(q.how == mo::Resolution::kDirect);
      CHECK(q.group_size == 16);
    }
  }
  // `lm_head` is NVFP4 here. The sibling artifact's head is FP8, and the two
  // 27B publishers have never agreed about the output head.
  const mo::ModuleQuant head = c.Resolve("lm_head");
  CHECK(head.algo == mo::QuantAlgo::kW4A16Nvfp4);
  CHECK(head.group_size == 16);

  // The GDN block is split EXACTLY where the map splits it: the three
  // projections it names are FP8 and the two low-rank ones are unlisted. A
  // resolver that treated `linear_attn.*` as one unit gets it wrong in both
  // directions, and this tree's `IsQwen27QuantizedLinear` still says every
  // `.linear_attn.in_proj_*` is unquantized — true of the 3.6 unsloth artifact
  // and false here. It has no production caller; `## Owed` names it.
  for (int l : {0, 1, 62}) {
    CAPTURE(l);
    const std::string la = p + std::to_string(l) + ".linear_attn";
    for (const char* proj : {".in_proj_qkv", ".in_proj_z", ".out_proj"}) {
      CAPTURE(proj);
      CHECK(c.Resolve(la + proj).algo == mo::QuantAlgo::kFp8);
    }
    CHECK_FALSE(c.Resolve(la + ".in_proj_a").Quantized());
    CHECK_FALSE(c.Resolve(la + ".in_proj_b").Quantized());
    CHECK_FALSE(c.Resolve(la + ".norm").Quantized());
    CHECK_FALSE(c.Resolve(la + ".conv1d").Quantized());
  }
  // Layer 3 is a full-attention layer and its four projections are FP8; layer 0
  // is a GDN layer and has none.
  for (const char* proj : {".q_proj", ".k_proj", ".v_proj", ".o_proj"}) {
    CAPTURE(proj);
    CHECK(c.Resolve(p + "3.self_attn" + proj).algo == mo::QuantAlgo::kFp8);
  }

  // Unquantized because NOTHING NAMES THEM, with an empty `ignore` list. The
  // sibling reaches the same answer for the vision tower through a 303-entry
  // exclusion, so a reader who assumes an exclusion mechanism here is wrong.
  for (const char* m : {"model.visual.blocks.0.attn.qkv",
                        "model.visual.blocks.26.mlp.linear_fc2",
                        "model.visual.merger.linear_fc1",
                        "model.visual.patch_embed.proj",
                        "mtp.fc", "mtp.layers.0.self_attn.q_proj",
                        "mtp.layers.0.mlp.gate_proj",
                        "model.language_model.embed_tokens"}) {
    CAPTURE(m);
    const mo::ModuleQuant q = c.Resolve(m);
    CHECK_FALSE(q.Quantized());
    CHECK(q.how == mo::Resolution::kUnlisted);
  }
}

// ── the MTP head this artifact ships, and the FP8 KV half it does not ───────
TEST_CASE("Qwen3.8-27B-NVFP4-MTP: the MTP shard is 15 unquantized bf16 tensors") {
  int mtp = 0;
  for (const ManifestTensor& t : Manifest()) {
    if (t.name.rfind("mtp.", 0) != 0) continue;
    ++mtp;
    CAPTURE(t.name);
    // The head is its own shard AND it is the only thing in it.
    CHECK(std::string(t.shard) == std::string(kShards[3].file));
    CHECK(t.dtype == "BF16");
    CHECK(BucketOf(t.name) == "UNLISTED");
  }
  CHECK(mtp == kShards[3].tensors);
}

TEST_CASE("Qwen3.8-27B-NVFP4-MTP: the FP8 KV scheme is declared where NO production path reads it") {
  // `hf_quant_config.json` asks for an FP8 KV cache.
  const nlohmann::json& hq = ReleasedHfQuantConfig().at("quantization");
  CHECK(hq.at("quant_algo").get<std::string>() == "MIXED_PRECISION");
  CHECK(hq.at("kv_cache_quant_algo").get<std::string>() == "FP8");
  // `config.json`'s `quantization_config` — the ONLY document any production
  // path in this tree reads — declares no KV scheme in either spelling, so the
  // parsed config carries none and no loader can act on it.
  CHECK_FALSE(ReleasedQuant().contains("kv_cache_quant_algo"));
  CHECK_FALSE(ReleasedQuant().contains("kv_cache_scheme"));
  CHECK(ReleasedMixed().kv_cache_quant_algo().empty());
  // And the checkpoint ships no scale for it either way.
  int kv = 0;
  for (const ManifestTensor& t : Manifest()) {
    if (BucketOf(t.name) == "KV_CACHE_SCALE") ++kv;
  }
  CHECK(kv == 0);
  // Owned by `KV-FP8` (#1593), listed under `## Owed` in
  // `.agents/specs/qwen38-27b-quant-arms.md`. Refusing it here would refuse
  // `nvidia/Qwen3.6-27B-NVFP4`, a gate model this tree loads today, which
  // declares `kv_cache_scheme` in its `config.json` and also ships no scales.
}

// ── the vLLM loader log says MXFP8, and the CHECKPOINT says otherwise ───────
//
// Running the artifact under a vLLM image prints three `Detected ModelOpt ...`
// warnings, and one of them is `modelopt.py:1676 Detected ModelOpt MXFP8
// checkpoint`. It is not a statement about this checkpoint. At the pinned
// oracle `5559679229bc961848b121ccdeaa8fa5d79bec98`,
// `ModelOptMixedPrecisionConfig._from_config` (modelopt.py:2371-2400)
// UNCONDITIONALLY constructs all FOUR candidate sub-configs — `ModelOptFp8Config`,
// two `ModelOptNvFp4Config` (NVFP4 and W4A16_NVFP4) and `ModelOptMxFp8Config` —
// and each one warns from its own `__init__` (`:386`, `:1035` twice, `:1708`).
// So the lines report what was CONSTRUCTED, not what was selected. SELECTION is
// `ModelOptMxFp8Config.override_quantization_method` (`:1724-1731`), which
// returns `modelopt_mxfp8` only when the extracted algo string CONTAINS
// "MXFP8", and per-module dispatch is
// `ModelOptMixedPrecisionConfig.get_quant_method` (`:2525-2535`), which hands
// out `mxfp8_config` only for a module whose `quantized_layers` entry SAYS
// MXFP8.
//
// This case is the checkpoint's half of that argument, so a future reader does
// not build an MXFP8 arm on the strength of a log line.
TEST_CASE("Qwen3.8-27B-NVFP4-MTP: nothing in this checkpoint is MXFP8, whatever the loader log says") {
  const nlohmann::json& layers = ReleasedQuant().at("quantized_layers");
  int fp8 = 0;
  int nvfp4 = 0;
  for (auto it = layers.begin(); it != layers.end(); ++it) {
    const std::string algo = it.value().at("quant_algo").get<std::string>();
    CAPTURE(it.key());
    CHECK(algo != "MXFP8");
    if (algo == "FP8") ++fp8;
    if (algo == "W4A16_NVFP4") ++nvfp4;
  }
  CHECK(fp8 == 208);
  CHECK(nvfp4 == 193);
  CHECK(fp8 + nvfp4 == static_cast<int>(layers.size()));
  // MXFP8 stores an E8M0 block scale per 32 elements. This checkpoint ships no
  // E8M0 tensor at all, and its only non-scalar scales are the 193 NVFP4 group
  // scales at F8_E4M3 [out, in/16]. Every FP8 scale is a rank-0 F32.
  int e8m0 = 0;
  int non_scalar_scales = 0;
  int fp8_scalar_scales = 0;
  for (const ManifestTensor& t : Manifest()) {
    if (t.dtype.find("E8M0") != std::string::npos) ++e8m0;
    const bool is_scale = t.name.size() > 6 &&
                          t.name.find("_scale") != std::string::npos;
    if (!is_scale) continue;
    if (!t.shape.empty()) {
      ++non_scalar_scales;
      CHECK(t.dtype == "F8_E4M3");
      CHECK(t.shape.size() == 2);
    } else {
      ++fp8_scalar_scales;
      CHECK(t.dtype == "F32");
    }
  }
  CHECK(e8m0 == 0);
  CHECK(non_scalar_scales == 193);
  // 208 `weight_scale` + 208 `input_scale` + 193 `weight_scale_2`.
  CHECK(fp8_scalar_scales == 609);
}

// ── the production loader ────────────────────────────────────────────────────
//
// Everything below drives `vllm::LoadQwen3_5Dense` — the loader every consumer
// of a Qwen3.5-family safetensors checkpoint arrives through — on a ONE-LAYER
// synthetic checkpoint carrying this artifact's REAL module names and a
// `quantization_config` built from its REAL one. The tensors are tiny; the
// declaration is the shipped shape, so what is exercised is the resolution the
// shipped file asks for and not a fixture's paraphrase of it.
namespace {

struct Spec {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype = "BF16";
};

int64_t Numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (const int64_t d : s) n *= d;
  return n;
}

std::size_t ElemSize(const std::string& dtype) {
  if (dtype == "BF16") return 2;
  if (dtype == "F32") return 4;
  return 1;  // U8 / F8_E4M3
}

std::string U64Le(uint64_t v) {
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i)
    out[static_cast<std::size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xffu);
  return out;
}

// A whole safetensors blob from `specs`, filled with finite positive values so
// a scale the loader reads is never zero and never a NaN.
std::string BuildSafetensors(const std::vector<Spec>& specs) {
  std::string header = "{";
  std::string body;
  uint64_t offset = 0;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    const int64_t n = Numel(specs[i].shape);
    const std::string& dtype = specs[i].dtype;
    const auto nbytes = static_cast<uint64_t>(n) * ElemSize(dtype);
    if (i != 0) header += ",";
    header += "\"" + specs[i].name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
    for (std::size_t d = 0; d < specs[i].shape.size(); ++d) {
      if (d != 0) header += ",";
      header += std::to_string(specs[i].shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + nbytes) + "]}";
    offset += nbytes;
    const std::size_t at = body.size();
    body.resize(at + static_cast<std::size_t>(nbytes));
    char* dst = body.data() + at;
    if (dtype == "BF16") {
      for (int64_t e = 0; e < n; ++e) {
        const auto v = static_cast<uint16_t>(0x3c00u + ((i * 7 + e * 3) & 0x3ffu));
        std::memcpy(dst + e * 2, &v, 2);
      }
    } else if (dtype == "F32") {
      for (int64_t e = 0; e < n; ++e) {
        const float v = 0.125F * static_cast<float>((i * 5 + e * 3) % 7 + 1);
        std::memcpy(dst + e * 4, &v, 4);
      }
    } else {
      for (int64_t e = 0; e < n; ++e)
        dst[e] = static_cast<char>((i * 37 + e * 7) & 0x7f);
    }
  }
  header += "}";
  return U64Le(header.size()) + header + body;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_qwen38_modelopt_mtp_" + std::to_string(counter++) +
              ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// NVFP4 packs two elements per byte in groups of 16, so every width used as an
// in_dim must be a multiple of 16. `o_proj` reads `kQ`, so kQ=8 refused with
// "NVFP4 in_dim must be a multiple of 16" before any scheme was resolved --
// which is a shape refusal wearing the answer this file is asking for.
constexpr int64_t kH = 32;  // hidden
constexpr int64_t kFfn = 64;
constexpr int64_t kHead = 4;
constexpr int64_t kQ = 32;
constexpr int64_t kKv = 16;

const char* kPrefix = "model.language_model.";

// The ModelOpt NVFP4 spelling: a U8 `weight`, an F8 group scale, and the F32
// scalar `weight_scale_2` that IS the global scale (compressed-tensors spells
// the same operand `weight_global_scale` and stores its reciprocal). This
// artifact ships NO `input_scale` on an NVFP4 projection, because its NVFP4
// half is W4A16.
void AppendModeloptNvfp4(std::vector<Spec>& out, const std::string& proj,
                         int64_t n, int64_t k) {
  out.push_back({proj + ".weight", {n, k / 2}, "U8"});
  out.push_back({proj + ".weight_scale", {n, k / 16}, "F8_E4M3"});
  out.push_back({proj + ".weight_scale_2", {}, "F32"});
}

// The compressed-tensors NVFP4 spelling, used only by the disagreement case.
void AppendCtNvfp4(std::vector<Spec>& out, const std::string& proj, int64_t n,
                   int64_t k) {
  out.push_back({proj + ".weight_packed", {n, k / 2}, "U8"});
  out.push_back({proj + ".weight_scale", {n, k / 16}, "F8_E4M3"});
  out.push_back({proj + ".weight_global_scale", {}, "F32"});
  out.push_back({proj + ".input_global_scale", {}, "F32"});
}

// The per-tensor STATIC FP8 spelling this artifact ships on all 208 of its FP8
// modules: an F8 weight beside two F32 SCALARS. Shape `[]`, exactly as the
// header shows, which is what `ReadF32Scalar` accepts.
void AppendStaticFp8(std::vector<Spec>& out, const std::string& proj, int64_t n,
                     int64_t k) {
  out.push_back({proj + ".weight", {n, k}, "F8_E4M3"});
  out.push_back({proj + ".weight_scale", {}, "F32"});
  out.push_back({proj + ".input_scale", {}, "F32"});
}

enum class AttnArm { kStaticFp8, kBf16, kCtNvfp4 };

// The MLP half varies for the same reason the attention half does. The config
// declares all three MLP projections `W4A16_NVFP4`, so `kStaticFp8` and `kBf16`
// are the two ways a checkpoint can ship something else under that declaration
// — and that declaration covers 193 of this artifact's 401 modules, 48% of it.
enum class MlpArm { kModeloptNvfp4, kStaticFp8, kBf16 };

// A one-layer full-attention backbone in this artifact's spelling: a static-FP8
// attention tower and a W4A16 NVFP4 MLP.
std::vector<Spec> OneLayerSpecs(AttnArm attn,
                                MlpArm mlp_arm = MlpArm::kModeloptNvfp4) {
  const std::string p = kPrefix;
  const std::string l = p + "layers.0.";
  const std::string sa = l + "self_attn.";
  const std::string mlp = l + "mlp.";
  std::vector<Spec> s = {
      {p + "embed_tokens.weight", {6, kH}},
      {p + "norm.weight", {kH}},
      {l + "input_layernorm.weight", {kH}},
      {l + "post_attention_layernorm.weight", {kH}},
      {sa + "q_norm.weight", {kHead}},
      {sa + "k_norm.weight", {kHead}},
  };
  const std::pair<const char*, std::pair<int64_t, int64_t>> projs[] = {
      {"q_proj", {kQ, kH}},
      {"k_proj", {kKv, kH}},
      {"v_proj", {kKv, kH}},
      {"o_proj", {kH, kQ}},
  };
  for (const auto& pr : projs) {
    const std::string name = sa + pr.first;
    switch (attn) {
      case AttnArm::kStaticFp8:
        AppendStaticFp8(s, name, pr.second.first, pr.second.second);
        break;
      case AttnArm::kBf16:
        s.push_back({name + ".weight", {pr.second.first, pr.second.second}});
        break;
      case AttnArm::kCtNvfp4:
        AppendCtNvfp4(s, name, pr.second.first, pr.second.second);
        break;
    }
  }
  const std::pair<const char*, std::pair<int64_t, int64_t>> mlp_projs[] = {
      {"gate_proj", {kFfn, kH}},
      {"up_proj", {kFfn, kH}},
      {"down_proj", {kH, kFfn}},
  };
  for (const auto& pr : mlp_projs) {
    const std::string name = mlp + pr.first;
    switch (mlp_arm) {
      case MlpArm::kModeloptNvfp4:
        AppendModeloptNvfp4(s, name, pr.second.first, pr.second.second);
        break;
      case MlpArm::kStaticFp8:
        AppendStaticFp8(s, name, pr.second.first, pr.second.second);
        break;
      case MlpArm::kBf16:
        s.push_back({name + ".weight", {pr.second.first, pr.second.second}});
        break;
    }
  }
  return s;
}

// The shipped `quantization_config`, narrowed to the one layer the synthetic
// checkpoint has. `quantized_layers` names every module exactly, so keeping the
// 401 entries of a 64-layer model would leave 397 of them naming tensors this
// checkpoint does not ship — which says nothing about the module names it does.
nlohmann::json OneLayerQuantConfig() {
  nlohmann::json q = nlohmann::json::object();
  q["quant_method"] = "modelopt";
  q["quant_algo"] = "MIXED_PRECISION";
  q["producer"] = ReleasedQuant().at("producer");
  q["ignore"] = nlohmann::json::array();
  nlohmann::json layers = nlohmann::json::object();
  const std::string l = std::string(kPrefix) + "layers.0.";
  // The FP8 entries carry no `group_size`, exactly as the shipped map does.
  for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
    layers[l + "self_attn." + proj] = {{"quant_algo", "FP8"}};
  }
  for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
    layers[l + "mlp." + proj] = {{"quant_algo", "W4A16_NVFP4"},
                                 {"group_size", 16}};
  }
  q["quantized_layers"] = layers;
  return q;
}

vllm::HfConfig OneLayerConfig(const nlohmann::json& quant) {
  vllm::HfConfig config;
  config.model_type = "qwen3_5_text";
  config.hidden_size = kH;
  config.num_hidden_layers = 1;
  config.layer_types = {"full_attention"};
  config.raw = nlohmann::json::object();
  config.raw["quantization_config"] = quant;
  return config;
}

// `e.what()`, or "" when the load returned. "Threw something" and "threw THIS"
// are different results and only one of them is evidence.
std::string LoadFailure(const std::vector<Spec>& specs,
                        const vllm::HfConfig& config) {
  const TempFile file(BuildSafetensors(specs));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  try {
    const vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5Dense(shards, config);
    (void)w;
    return std::string();
  } catch (const std::exception& e) {
    return e.what();
  }
}

}  // namespace

// ── (3a) the artifact's own spelling LOADS, and stays quantized ─────────────
//
// The negative control for every refusal below. Without it, each of them is
// satisfied by a loader that refuses every ModelOpt checkpoint.
TEST_CASE("Qwen3.8-27B-NVFP4-MTP loader: the shipped spelling loads, unrefused, and stays quantized") {
  const vllm::HfConfig config = OneLayerConfig(OneLayerQuantConfig());
  const std::vector<Spec> specs = OneLayerSpecs(AttnArm::kStaticFp8);
  const std::string message = LoadFailure(specs, config);
  INFO("refusal was: " << message);
  REQUIRE(message.empty());

  const TempFile file(BuildSafetensors(specs));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  const vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5Dense(shards, config);
  REQUIRE(w.layers.size() == 1);

  // The MLP is fp4-RESIDENT and the attention tower is fp8-RESIDENT, not
  // dequantized to bf16. A silent dequant passes every token gate and defeats
  // the whole point of a quantized arm, so the assertion is on the memory
  // format rather than on a value.
  const auto& mlp = w.layers[0].mlp;
  REQUIRE_FALSE(mlp.gate_proj_fp4.packed.Empty());
  CHECK(mlp.gate_proj_fp4.n == kFfn);
  CHECK(mlp.gate_proj_fp4.k == kH);
  CHECK(mlp.gate_proj_fp4.packed.shape[1] == kH / 2);
  CHECK(mlp.gate_up_proj.Empty());
  CHECK(mlp.down_proj.Empty());
  // W4A16, not W4A4: this artifact ships no activation scale on an NVFP4
  // projection, so `alpha` stays zero and the weight routes to the
  // weight-only dispatcher.
  CHECK(mlp.gate_proj_fp4.alpha == 0.0F);

  const auto& attn = w.layers[0].attn;
  CHECK(attn.q_proj.Empty());
  CHECK(attn.q_proj_fp4.packed.Empty());
  CHECK_FALSE(attn.q_proj_fp8.Empty());
  CHECK_FALSE(attn.o_proj_fp8.Empty());
  CHECK(attn.q_proj_fp8.n == kQ);
  CHECK(attn.q_proj_fp8.k == kH);
  // `alpha` folds BOTH per-tensor scalars once at load, which is only possible
  // because this artifact's input scale is STATIC.
  CHECK(attn.q_proj_fp8.alpha ==
        attn.q_proj_fp8.input_scale * attn.q_proj_fp8.weight_scale);
  CHECK(attn.q_proj_fp8.alpha > 0.0F);
}

// ── (3b) a config/tensor disagreement is refused BY NAME, both directions ───
TEST_CASE("Qwen3.8-27B-NVFP4-MTP loader: a module declared FP8 that ships an NVFP4 spelling is refused") {
  // The config says the attention tower is FP8; the tensors spell NVFP4. The
  // name probe takes the NVFP4 arm and never notices, which is the silent half
  // of this defect class: the load succeeds, the tokens are plausible, and the
  // tower moves four bits where the producer wrote eight.
  const std::string message = LoadFailure(OneLayerSpecs(AttnArm::kCtNvfp4),
                                          OneLayerConfig(OneLayerQuantConfig()));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "self_attn.q_proj"));
  CHECK(Names(message, "modelopt MIXED_PRECISION"));
  CHECK(Names(message, "FP8"));
  CHECK(Names(message, "weight_packed"));
  // It is a statement about a DISAGREEMENT, not about a missing tensor: the
  // checkpoint is complete, it is just not the one the config describes.
  CHECK_FALSE(Names(message, "tensor not found"));
}

TEST_CASE("Qwen3.8-27B-NVFP4-MTP loader: a module declared FP8 that ships plain bf16 is refused") {
  const std::string message = LoadFailure(OneLayerSpecs(AttnArm::kBf16),
                                          OneLayerConfig(OneLayerQuantConfig()));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "self_attn.q_proj"));
  CHECK(Names(message, "FP8"));
  CHECK(Names(message, "input_scale"));
  // The whole spelling it DOES ship is named, so the reader is not left to
  // diff two lists in their head.
  CHECK(Names(message, "they ship weight"));
  CHECK_FALSE(Names(message, "tensor not found"));
}

// The NVFP4 direction, and the one that carries 193 of this artifact's 401
// declared modules — 48% of it, including `lm_head`. Its refusal is ONE branch
// in `Refusal`, and until this case existed that branch could be deleted with
// the whole file still green: every other case here exercises the FP8, the
// unquantized, the MXFP8 or the unimplemented-algo arm, and the sibling-shape
// case pins only the direction in which the NVFP4 branch must NOT fire.
// ONE CASE PER SHAPE, not one case with two `SUBCASE`s. A `REQUIRE` throws and
// aborts the whole TEST_CASE, so a first subcase that reds takes every later
// subcase in the same case with it and the report shows one failure where there
// were two — which is the "a case that never ran reads as a pass" shape.
TEST_CASE("Qwen3.8-27B-NVFP4-MTP loader: a module declared W4A16_NVFP4 that ships the static FP8 spelling is refused") {
  // The config declares the whole MLP `W4A16_NVFP4`; the tensors spell static
  // FP8. `IsNvfp4Projection` sees no `weight_scale_2` and no `weight_packed`,
  // so the name probe falls to the `F8_E4M3` dtype arm and loads the MLP at
  // eight bits where the producer wrote four — a load that succeeds, and whose
  // tokens are plausible.
  const std::string message =
      LoadFailure(OneLayerSpecs(AttnArm::kStaticFp8, MlpArm::kStaticFp8),
                  OneLayerConfig(OneLayerQuantConfig()));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "modelopt MIXED_PRECISION"));
  CHECK(Names(message, "mlp.gate_proj"));
  CHECK(Names(message, "W4A16_NVFP4"));
  // The message names the spelling this build DOES read, both ways of spelling
  // it, and the one the checkpoint actually ships.
  CHECK(Names(message, "weight_scale_2"));
  CHECK(Names(message, "weight_global_scale"));
  CHECK(Names(message, "they ship weight + weight_scale + input_scale"));
  // It is a DISAGREEMENT, not a missing tensor: the checkpoint is complete.
  CHECK_FALSE(Names(message, "tensor not found"));
  // And the attention tower still agrees, so only the MLP is named.
  CHECK_FALSE(Names(message, "self_attn.q_proj"));
}

TEST_CASE("Qwen3.8-27B-NVFP4-MTP loader: a module declared W4A16_NVFP4 that ships plain bf16 is refused") {
  // The silent-dequant direction on the NVFP4 half: 193 modules read at sixteen
  // bits because nothing named the four.
  const std::string message =
      LoadFailure(OneLayerSpecs(AttnArm::kStaticFp8, MlpArm::kBf16),
                  OneLayerConfig(OneLayerQuantConfig()));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "modelopt MIXED_PRECISION"));
  CHECK(Names(message, "mlp.down_proj"));
  CHECK(Names(message, "W4A16_NVFP4"));
  CHECK(Names(message, "they ship weight."));
  CHECK_FALSE(Names(message, "tensor not found"));
}

TEST_CASE("Qwen3.8-27B-NVFP4-MTP loader: a module the config does NOT quantize that ships scales is refused") {
  // The other direction, and the one the ModelOpt resolver exists for: reading
  // a checkpoint as uniformly quantized because its repo name says NVFP4.
  nlohmann::json quant = OneLayerQuantConfig();
  quant["quantized_layers"].erase(std::string(kPrefix) +
                                  "layers.0.self_attn.q_proj");
  const std::string message =
      LoadFailure(OneLayerSpecs(AttnArm::kStaticFp8), OneLayerConfig(quant));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "self_attn.q_proj"));
  CHECK(Names(message, "UNQUANTIZED"));
  CHECK(Names(message, "does not quantize"));
  // And it names ONLY that module: the other three attention projections are
  // still declared FP8 and still agree.
  CHECK_FALSE(Names(message, "self_attn.k_proj"));
  CHECK_FALSE(Names(message, "tensor not found"));
}

// ── (4) a declared algorithm with no loader is refused BY NAME ──────────────
TEST_CASE("Qwen3.8-27B-NVFP4-MTP loader: a declared MXFP8 module is refused by name") {
  nlohmann::json quant = OneLayerQuantConfig();
  quant["quantized_layers"][std::string(kPrefix) +
                            "layers.0.self_attn.v_proj"] = {
      {"quant_algo", "MXFP8"}};
  const std::string message =
      LoadFailure(OneLayerSpecs(AttnArm::kStaticFp8), OneLayerConfig(quant));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "self_attn.v_proj"));
  CHECK(Names(message, "MXFP8"));
  CHECK_FALSE(Names(message, "tensor not found"));
}

TEST_CASE("Qwen3.8-27B-NVFP4-MTP loader: a quant_algo this consumer does not implement is refused by name") {
  // `FP8_PER_CHANNEL_PER_TOKEN` IS a real ModelOpt algorithm
  // (modelopt.py:105-120) that this consumer does not implement, and the
  // resolver refuses it rather than falling through to an unquantized read the
  // way upstream's `get_quant_method` does.
  nlohmann::json quant = OneLayerQuantConfig();
  quant["quantized_layers"][std::string(kPrefix) +
                            "layers.0.self_attn.k_proj"] = {
      {"quant_algo", "FP8_PER_CHANNEL_PER_TOKEN"}};
  const std::string message =
      LoadFailure(OneLayerSpecs(AttnArm::kStaticFp8), OneLayerConfig(quant));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "self_attn.k_proj"));
  CHECK(Names(message, "FP8_PER_CHANNEL_PER_TOKEN"));
}

// ── the OTHER ModelOpt checkpoint this loader sees must keep loading ────────
//
// `nvidia/Qwen3.6-27B-NVFP4` @ `0893e1606ff3d5f97a441f405d5fc541a6bdf404` is the
// FP8-tower gate model of #466 (`tests/parity/hf_snapshot.h`), and it declares
// the same `quant_method: "modelopt"`, the same `quant_algo: "MIXED_PRECISION"`
// and the same 401-entry split of 208 `FP8` and 193 `W4A16_NVFP4`. It reaches
// the new call site too, so "does this refusal fire on a checkpoint that loads
// today" is a question about that checkpoint and not about this one.
//
// It differs from the artifact above in exactly three ways, read from its own
// `config.json` and `model.safetensors.index.json` on 2026-08-21, and each of
// the three is a way the refusal could have fired:
//
//   1. `exclude_modules` is `["mtp*", "mtp.layers.0*"]`, matched by `fnmatch`
//      rather than exactly, where this artifact's `ignore` is empty.
//   2. Its 193 NVFP4 modules ALSO ship an `input_scale` (2194 index names
//      against 2001), which is the operand `VT_MODELOPT_W4A4=1` reads.
//   3. Its `config.json` declares a `kv_cache_scheme` and it ships ZERO
//      `k_scale`/`v_scale` tensors for it.
//
// This case rebuilds that SHAPE — not that checkpoint, whose 2194 names are not
// committed here — and asserts the refusal stays silent on all three. A refusal
// that fired on any of them would refuse a gate model every recorded 27B NVFP4
// ratio was taken on.
TEST_CASE("modelopt MIXED_PRECISION: the sibling gate model's shape is not refused") {
  const std::string l = std::string(kPrefix) + "layers.0.";
  nlohmann::json q = OneLayerQuantConfig();
  // (1) wildcard exclusions rather than an empty list.
  q["ignore"] = nlohmann::json::array({"mtp*", "mtp.layers.0*"});
  // (3) a kv_cache_scheme in `config.json`, which `Parse` reads into
  //     `kv_cache_quant_algo` for the flat shape.
  q["kv_cache_scheme"] = {{"dynamic", false}, {"num_bits", 8}, {"type", "float"}};

  std::vector<std::string> names;
  for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
    const std::string m = l + "self_attn." + proj;
    names.push_back(m + ".weight");
    names.push_back(m + ".weight_scale");
    names.push_back(m + ".input_scale");
  }
  for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
    const std::string m = l + "mlp." + proj;
    names.push_back(m + ".weight");
    names.push_back(m + ".weight_scale");
    names.push_back(m + ".weight_scale_2");
    // (2) the extra activation scale the sibling ships on every NVFP4 module.
    names.push_back(m + ".input_scale");
  }
  // The excluded MTP head, and a plain bf16 tower nothing names.
  names.push_back("mtp.fc.weight");
  names.push_back("mtp.layers.0.self_attn.q_proj.weight");
  names.push_back(l + "input_layernorm.weight");
  names.push_back(l + "linear_attn.A_log");
  names.push_back(l + "linear_attn.dt_bias");
  // The KV scales neither artifact ships today, on the module a ModelOpt export
  // attaches them to. The case below is where a KV scale on a WEIGHT-BEARING
  // module is asserted; here they are only carried so this shape stays the
  // sibling's, and so a resolver that refused the suffix outright would red.
  names.push_back(l + "self_attn.k_scale");
  names.push_back(l + "self_attn.v_scale");

  const std::string refusal = mo::RefusalForQuantizationConfig(q, names);
  INFO("refusal was: " << refusal);
  CHECK(refusal.empty());

  // The kv-cache algorithm IS parsed, so the silence above is a decision and
  // not a blind spot: `KV-FP8` (#1593) owns consuming it.
  const mo::MixedPrecisionConfig parsed = mo::MixedPrecisionConfig::Parse(q);
  CHECK(parsed.kv_cache_quant_algo() == "FP8");
  // And the wildcard exclusion really does exclude, so (1) was exercised.
  CHECK(parsed.Resolve("mtp.fc").how == mo::Resolution::kExcluded);
  CHECK_FALSE(parsed.Resolve(l + "mlp.gate_proj").how == mo::Resolution::kExcluded);
}

// ── a family the splitter has never seen is REFUSED, not skipped ────────────
//
// `SplitOperand` documents that `false` means "this resolver has never seen the
// family, and the caller must NOT read that as unquantized". A `continue` in
// `Refusal` read it as exactly that: the name belonged to no module, so nothing
// cross-checked it in either direction and the checkpoint loaded silently —
// which is the state this whole file exists to close, arriving through the one
// door nobody watched.
// One case per shape, for the reason argued above the NVFP4 pair: a `REQUIRE`
// aborts its whole TEST_CASE, so a subcase that reds hides the ones after it.
TEST_CASE("modelopt MIXED_PRECISION: an AWQ/GPTQ triple no probe reads is refused, not skipped") {
  // `qweight` / `qzeros` / `scales` is the AWQ/GPTQ triple this tree already
  // names elsewhere (`minimax_music3_quant.cpp:106`), and no probe in this
  // loader looks for any of the three. Before this arm the checkpoint LOADED:
  // its declared-FP8 tower took the FP8 path, and three tensors carrying a
  // different quantization of the same projection went unread and unmentioned.
  const std::string l = std::string(kPrefix) + "layers.0.";
  std::vector<Spec> specs = OneLayerSpecs(AttnArm::kStaticFp8);
  const std::string m = l + "self_attn.q_proj";
  specs.push_back({m + ".qweight", {kQ, kH / 8}, "F32"});
  specs.push_back({m + ".qzeros", {kQ / 8, kH / 8}, "F32"});
  specs.push_back({m + ".scales", {kQ, 1}});
  const std::string message =
      LoadFailure(specs, OneLayerConfig(OneLayerQuantConfig()));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "modelopt MIXED_PRECISION"));
  CHECK(Names(message, "qweight"));
  CHECK(Names(message, "qzeros"));
  CHECK(Names(message, "scales"));
  CHECK(Names(message, "never seen"));
  CHECK(Names(message, "3 shipped tensor name(s)"));
}

TEST_CASE("modelopt MIXED_PRECISION: a block-wise FP8 scale on a module nothing names is refused") {
  // `.weight_scale_inv` is the block-wise FP8 spelling, and the honest note is
  // that in THIS loader it does not reach the arm above: the block-FP8 guard
  // two rungs earlier in `LoadQwen3_5Dense` refuses it first when the config
  // declares no `weight_block_size`. That guard belongs to one loader and this
  // resolver is a shared header, so the family is still one it has never seen,
  // and a module that ships only it is cross-checked in NEITHER direction —
  // `weight` alone reads as "declared unquantized, ships nothing quantized",
  // which is the silent answer.
  const std::string l = std::string(kPrefix) + "layers.0.";
  const nlohmann::json q = OneLayerQuantConfig();
  std::vector<std::string> names;
  const std::string m = l + "linear_attn.out_proj";
  names.push_back(m + ".weight");
  names.push_back(m + ".weight_scale_inv");
  const std::string refusal = mo::RefusalForQuantizationConfig(q, names);
  INFO("refusal was: " << refusal);
  REQUIRE_FALSE(refusal.empty());
  CHECK(Names(refusal, "weight_scale_inv"));
  CHECK(Names(refusal, "1 shipped tensor name(s)"));
  CHECK(Names(refusal, "never seen"));
}

TEST_CASE("modelopt MIXED_PRECISION: the pinned artifact classifies every name it ships") {
  // The negative control, and the reason the refusal above is safe to add: it
  // cannot fire on the checkpoint this file is written against. The sibling
  // `nvidia/Qwen3.6-27B-NVFP4` is the other one that reaches this resolver, and
  // its 2194 names classify too — read from its own index on 2026-08-21 and
  // recorded above `OperandSuffixes`, since its manifest is not committed here.
  std::vector<std::string> names;
  for (const ManifestTensor& t : Manifest()) names.push_back(t.name);
  REQUIRE(names.size() == static_cast<std::size_t>(kIndexNameCount));
  CHECK(mo::RefusalForQuantizationConfig(ReleasedQuant(), names).empty());
}

// ── a KV-cache scale is recorded, and it NEVER decides a refusal ────────────
//
// `kv_cache_quant_algo` is a SIBLING of `quantized_layers` (modelopt.py:294,
// :306-314), so a `k_scale` says nothing about how a module's WEIGHTS are
// stored. `Refusal` used to state that by skipping the suffix before any module
// was built, which changed no verdict and so asserted nothing — deleting the
// skip left every case green. The statement now lives in one branch:
// `ModuleOperands::Add` RECORDS the scale and `AnyQuantOperand` leaves it out.
TEST_CASE("modelopt MIXED_PRECISION: a KV-cache scale on an UNLISTED weight-bearing module is not refused") {
  // The direction `AnyQuantOperand` decides: a bf16 attention tower the config
  // does not quantize, shipping the KV scales a `kv_cache_scheme` asks for.
  // Counting `k_scale` as a quantized spelling would refuse it, and
  // `nvidia/Qwen3.6-27B-NVFP4` already declares that scheme.
  const std::string l = std::string(kPrefix) + "layers.0.";
  nlohmann::json unlisted = OneLayerQuantConfig();
  const std::string m = l + "self_attn.k_proj";
  unlisted["quantized_layers"].erase(m);
  std::vector<std::string> names;
  names.push_back(m + ".weight");
  names.push_back(m + ".k_scale");
  names.push_back(m + ".v_scale");
  const std::string refusal = mo::RefusalForQuantizationConfig(unlisted, names);
  INFO("refusal was: " << refusal);
  CHECK(refusal.empty());
  // ... and the module really is unquantized, so the silence is the
  // unquantized-direction branch staying quiet rather than a module the
  // resolver never reached.
  CHECK_FALSE(mo::MixedPrecisionConfig::Parse(unlisted).Resolve(m).Quantized());
}

TEST_CASE("modelopt MIXED_PRECISION: a KV-cache scale is RECORDED, and the refusal names it") {
  // The other half: the scale is recorded rather than dropped, so it appears in
  // the spelling the refusal prints. An `Add` that ignored the suffix would
  // report "they ship weight" and hide what the file actually carries.
  const std::string l = std::string(kPrefix) + "layers.0.";
  const nlohmann::json q = OneLayerQuantConfig();
  std::vector<std::string> names;
  const std::string m = l + "self_attn.q_proj";
  names.push_back(m + ".weight");
  names.push_back(m + ".k_scale");
  const std::string refusal = mo::RefusalForQuantizationConfig(q, names);
  INFO("refusal was: " << refusal);
  REQUIRE_FALSE(refusal.empty());
  CHECK(Names(refusal, "self_attn.q_proj"));
  CHECK(Names(refusal, "FP8"));
  CHECK(Names(refusal, "they ship weight + k_scale"));
}

// ── a checkpoint that is NOT ModelOpt is untouched by any of this ───────────
TEST_CASE("Qwen3.8-27B-NVFP4-MTP loader: a non-modelopt quantization_config is not read by the ModelOpt arm") {
  // The selection hook, exercised through the production entry point: the
  // compressed-tensors sibling's `quant_method` is not `modelopt`, so the
  // ModelOpt refusal reads nothing and answers "". Without this a caller could
  // not tell "agreed" from "never looked".
  nlohmann::json quant = OneLayerQuantConfig();
  quant["quant_method"] = "compressed-tensors";
  std::vector<std::string> names;
  for (const ManifestTensor& t : Manifest()) names.push_back(t.name);
  CHECK(mo::RefusalForQuantizationConfig(quant, names).empty());
  // ... while the same names against the shipped ModelOpt config also agree,
  // which is what makes the line above a selection result rather than a tie.
  CHECK(mo::RefusalForQuantizationConfig(ReleasedQuant(), names).empty());
}

// ── the LIVE arm, env-gated ─────────────────────────────────────────────────
//
// The committed manifests are only as good as the files they were captured
// from, and this model family has been re-quantized in place under an unchanged
// name twice. This case re-reads the mirrored artifact's own headers and
// compares them to the manifests name by name.
//
//   VLLM_CPP_QWEN38_27B_MODELOPT_MTP_DIR=/path/to/mirrored/checkpoint
//   ./build/tests/test_qwen38_27b_modelopt_mtp_arm
//
// The artifact is pinned with its four sizes in `docs/USAGE.md`.
TEST_CASE("Qwen3.8-27B-NVFP4-MTP live: the mirrored headers still ARE the committed manifests") {
  const char* dir = std::getenv("VLLM_CPP_QWEN38_27B_MODELOPT_MTP_DIR");
  if (dir == nullptr) {
    MESSAGE(
        "SKIPPED: set VLLM_CPP_QWEN38_27B_MODELOPT_MTP_DIR to the mirrored "
        "r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 directory to re-verify the four "
        "committed manifests and the committed config against the shipped "
        "bytes (docs/USAGE.md pins the artifact)");
    return;
  }
  const std::filesystem::path root(dir);
  MESSAGE("resolved artifact directory: " << root.string());

  std::map<std::string, std::pair<std::string, std::vector<int64_t>>> live;
  for (const Shard& s : kShards) {
    const std::filesystem::path shard = root / s.file;
    REQUIRE_MESSAGE(std::filesystem::exists(shard),
                    "no " << s.file << " under " << dir);
    std::error_code ec;
    CHECK(static_cast<int64_t>(std::filesystem::file_size(shard, ec)) == s.bytes);
    REQUIRE_FALSE(ec);

    std::ifstream in(shard.string(), std::ios::binary);
    REQUIRE(in.good());
    char len_le[8];
    in.read(len_le, 8);
    uint64_t header_len = 0;
    for (int i = 7; i >= 0; --i)
      header_len = (header_len << 8) | static_cast<uint8_t>(len_le[i]);
    CHECK(header_len == static_cast<uint64_t>(s.header_len));
    std::string payload(static_cast<std::size_t>(header_len), '\0');
    in.read(payload.data(), static_cast<std::streamsize>(header_len));
    REQUIRE(in.good());
    const nlohmann::json header = nlohmann::json::parse(payload);

    int64_t data_end = 0;
    int tensors = 0;
    for (auto it = header.begin(); it != header.end(); ++it) {
      if (it.key() == "__metadata__") continue;
      const auto offs = it.value().at("data_offsets");
      data_end = std::max<int64_t>(data_end, offs[1].get<int64_t>());
      live[it.key()] = {it.value().at("dtype").get<std::string>(),
                        it.value().at("shape").get<std::vector<int64_t>>()};
      ++tensors;
    }
    CHECK(tensors == s.tensors);
    // The semantic verification `AGENTS.md` requires in place of a remote hash.
    CHECK(8 + static_cast<int64_t>(header_len) + data_end == s.bytes);
  }
  CHECK(live.size() == static_cast<std::size_t>(kIndexNameCount));

  int mismatches = 0;
  for (const ManifestTensor& t : Manifest()) {
    const auto it = live.find(t.name);
    if (it == live.end() || it->second.first != t.dtype ||
        it->second.second != t.shape) {
      if (mismatches < 5) MESSAGE("manifest/live mismatch: " << t.name);
      ++mismatches;
    }
  }
  CHECK(mismatches == 0);

  // The refusal the PRODUCTION resolver raises over the SHIPPED names and the
  // SHIPPED config: this artifact agrees with itself, so the answer is "".
  std::vector<std::string> live_names;
  live_names.reserve(live.size());
  for (const auto& kv : live) live_names.push_back(kv.first);
  const std::string live_refusal =
      mo::RefusalForQuantizationConfig(ReleasedConfig().at("quantization_config"),
                                       live_names);
  MESSAGE("live refusal: " << live_refusal);
  CHECK(live_refusal.empty());

  // The committed fixtures are the shipped documents, byte for byte.
  for (const char* file : {"config.json", "hf_quant_config.json"}) {
    const std::filesystem::path doc = root / file;
    if (!std::filesystem::exists(doc)) continue;
    std::ifstream live_doc(doc.string());
    nlohmann::json shipped;
    live_doc >> shipped;
    CAPTURE(file);
    CHECK(shipped == Fixture(file));
  }
}
