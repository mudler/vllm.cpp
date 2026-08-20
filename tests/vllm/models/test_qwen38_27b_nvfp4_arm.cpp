// `QUANT-QWEN38-27B-NVFP4-ARM` W4 — the compressed-tensors `mixed-precision`
// accounting and scheme-resolution gate for `unsloth/Qwen3.8-27B-NVFP4`
// @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`.
// Issue https://github.com/mudler/vllm.cpp/issues/821, spec
// `.agents/specs/qwen38-27b-quant-arms.md`.
//
// WHAT THIS ARTIFACT IS. Its repo name says NVFP4 and its
// `quantization_config.format` says `"mixed-precision"`. Layers 0-55's MLP is
// NVFP4 W4A4 (`nvfp4-pack-quantized`, group_size 16); layers 56-63's MLP, the
// whole attention tower, the GDN input/output projections and `lm_head` are FP8
// W8A8 with a PER-OUTPUT-CHANNEL weight scale and DYNAMIC per-token activations
// (`float-quantized`). Same module names on both sides of the layer-56
// boundary, so no per-projection dtype probe can tell them apart: the split is
// a REGEX OVER LAYER INDICES and lives only in the config.
//
// The five things gated here, and why each is a defect a token gate cannot see:
//
//   (1) THE NAME INDEX IS ACCOUNTED, BOTH DIRECTIONS. All 1968 names of the
//       checkpoint's `model.safetensors.index.json` weight map — 1953 in
//       `model.safetensors` and 15 in `model_mtp.safetensors` — classify into
//       exactly one scheme, nothing is unaccounted, and the buckets sum to
//       1968. A tensor family nobody classified is the shape that goes
//       missing silently.
//   (2) THE PER-SCHEME COMPOSITION is the checkpoint's own, not a guess: 233
//       FP8 modules, 168 NVFP4 modules, 317 ignored and 267 unmatched. Reading
//       this checkpoint as UNIFORMLY NVFP4, or as uniformly FP8, is
//       numerically plausible and token-invisible while moving the wrong
//       bytes.
//   (3) THE LAYER-56 BOUNDARY resolves by the ORDER of `config_groups`, which
//       is upstream's first-match rule (`utils.py:169-172`). Both groups'
//       targets match layer 60's `gate_proj`; only the order decides.
//   (4) THE `ignore` LIST IS EXACT-MATCH, NOT PREFIX. `...layers.<i>.linear_attn`
//       is on it and `...linear_attn.in_proj_qkv` is NOT — the GDN block is
//       ignored in one direction and quantized in the other, and a resolver
//       that treats `linear_attn.*` as one unit gets it wrong both ways.
//   (5) THE FP8 ARM IS REFUSED BY NAME, through the production loader. Before
//       this row `LoadQwen3_5Dense` reported `tensor not found:
//       ...in_proj_qkv.input_scale` — a sentence about a checkpoint that is
//       complete, for a scheme this build has no loader for.
//
// HERMETIC. Every case above reads the COMMITTED manifests and the COMMITTED
// `config.json`, so CI needs no NAS file and no network. The live arm at the
// bottom is env-gated and SKIPS LOUDLY when its variable is unset.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_config.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"

#include "qwen38_27b_nvfp4_manifest.inc"
#include "qwen38_27b_nvfp4_mtp_manifest.inc"

namespace ct = vllm::layers::compressed_tensors;

namespace {

// ── The artifact, restated from its own bytes ────────────────────────────────
//
// Every literal here was re-derived by parsing the mirrored file's header, not
// copied from a record of it. `data_offsets[1]`'s maximum plus the 8-byte
// length prefix plus the header equals the file size in both shards, which is
// the semantic verification `AGENTS.md` requires instead of a remote hash.
constexpr int64_t kMainShardBytes = 22568192096LL;
constexpr int64_t kMtpShardBytes = 849400392LL;
// `metadata.total_size` of `model.safetensors.index.json`, which for this
// artifact is the sum of the two file sizes.
constexpr int64_t kIndexTotalSize = 23417592488LL;
constexpr int64_t kIndexNameCount = 1968;

std::string FixtureDir() {
#ifdef QWEN38_27B_NVFP4_FIXTURE_DIR
  return QWEN38_27B_NVFP4_FIXTURE_DIR;
#else
  return "tests/vllm/models/fixtures/qwen38_27b_nvfp4";
#endif
}

const nlohmann::json& ReleasedConfig() {
  static const nlohmann::json* doc = [] {
    std::ifstream in(FixtureDir() + "/config.json");
    REQUIRE_MESSAGE(in.good(), "cannot open the committed config.json fixture");
    auto* j = new nlohmann::json();
    in >> *j;
    return j;
  }();
  return *doc;
}

const ct::Config& ReleasedCtConfig() {
  static const ct::Config* c =
      new ct::Config(ct::Config::FromHfConfigRaw(ReleasedConfig()));
  return *c;
}

// One entry of the two committed manifests, unified.
struct ManifestTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  const char* shard;
};

const std::vector<ManifestTensor>& Manifest() {
  static const std::vector<ManifestTensor>* all = [] {
    auto* v = new std::vector<ManifestTensor>();
    for (const auto& t : vllm_test::kQwen38_27bNvfp4Tensors) {
      ManifestTensor m;
      m.name = t.name;
      m.dtype = t.dtype;
      for (int i = 0; i < t.rank; ++i) m.shape.push_back(t.shape[i]);
      m.shard = "model.safetensors";
      v->push_back(std::move(m));
    }
    for (const auto& t : vllm_test::kQwen38_27bNvfp4MtpTensors) {
      ManifestTensor m;
      m.name = t.name;
      m.dtype = t.dtype;
      for (int i = 0; i < t.rank; ++i) m.shape.push_back(t.shape[i]);
      m.shard = "model_mtp.safetensors";
      v->push_back(std::move(m));
    }
    return v;
  }();
  return *all;
}

// A tensor name minus its operand suffix is the MODULE name the config's
// `targets` and `ignore` entries are written against. The split is the
// PRODUCTION one (`ct::SplitOperand`), not a paraphrase of it: the loader
// resolves the shipped names through that function, so a suffix it does not
// know is a defect this gate must see rather than one it reproduces.
using ct::SplitOperand;

// The bucket a NAME lands in. `k_scale`/`v_scale` belong to the sibling
// `kv_cache_scheme`, not to any config group, so they are their own bucket
// rather than being resolved through `targets` — which is what upstream does
// too (`compressed_tensors.py:262` reads it separately from `config_groups`).
std::string BucketOf(const std::string& name) {
  std::string module;
  std::string suffix;
  if (!SplitOperand(name, &module, &suffix)) return "UNCLASSIFIED";
  if (suffix == "k_scale" || suffix == "v_scale") return "KV_CACHE_SCALE";
  const ct::ModuleScheme s = ReleasedCtConfig().Resolve(module);
  if (s.ignored) return "IGNORED";
  if (!s.matched) return "UNMATCHED";
  return s.group;
}

}  // namespace

// ── (1) the committed manifests ARE the shipped index ───────────────────────
TEST_CASE("Qwen3.8-27B-NVFP4: the two committed manifests account for all 1968 index names") {
  CHECK(vllm_test::kQwen38_27bNvfp4TensorCount == 1953);
  CHECK(vllm_test::kQwen38_27bNvfp4MtpTensorCount == 15);
  CHECK(vllm_test::kQwen38_27bNvfp4TensorCount +
            vllm_test::kQwen38_27bNvfp4MtpTensorCount ==
        kIndexNameCount);
  REQUIRE(Manifest().size() == static_cast<size_t>(kIndexNameCount));

  // Every name distinct, across BOTH shards. A weight map cannot name one
  // tensor twice, and a duplicated manifest row would inflate every count
  // below without failing any of them.
  std::set<std::string> unique;
  for (const ManifestTensor& t : Manifest()) unique.insert(t.name);
  CHECK(unique.size() == static_cast<size_t>(kIndexNameCount));

  // The whole-checkpoint dtype histogram, re-derived from the header. The
  // shipped `model.safetensors` is F32 336 / BF16 1048 / F8_E4M3 401 / U8 168,
  // and the MTP shard is 15 BF16.
  std::map<std::string, int> dtypes;
  for (const ManifestTensor& t : Manifest()) dtypes[t.dtype] += 1;
  CHECK(dtypes["F32"] == 336);
  CHECK(dtypes["BF16"] == 1048 + 15);
  CHECK(dtypes["F8_E4M3"] == 401);
  CHECK(dtypes["U8"] == 168);
  CHECK(dtypes.size() == 4);

  // Every name carries a recognised operand suffix. A name that does not is
  // not "other", it is a family this gate has never seen and cannot classify.
  std::vector<std::string> unclassified;
  for (const ManifestTensor& t : Manifest()) {
    std::string module;
    std::string suffix;
    if (!SplitOperand(t.name, &module, &suffix)) unclassified.push_back(t.name);
  }
  INFO("unclassified names: " << unclassified.size());
  CHECK(unclassified.empty());

  CHECK(kMainShardBytes + kMtpShardBytes == kIndexTotalSize);
}

// ── (2) the per-scheme composition ──────────────────────────────────────────
TEST_CASE("Qwen3.8-27B-NVFP4: every shipped name resolves to exactly one compressed-tensors scheme") {
  const ct::Config& c = ReleasedCtConfig();
  REQUIRE(c.declared());
  CHECK(c.quant_method() == "compressed-tensors");
  CHECK(c.format() == "mixed-precision");
  CHECK(c.group_count() == 2);
  CHECK(c.ignore_count() == 303);

  std::map<std::string, int> tensors;
  std::map<std::string, std::set<std::string>> modules;
  for (const ManifestTensor& t : Manifest()) {
    const std::string bucket = BucketOf(t.name);
    tensors[bucket] += 1;
    std::string module;
    std::string suffix;
    if (SplitOperand(t.name, &module, &suffix)) modules[bucket].insert(module);
  }

  // FP8 W8A8, `float-quantized`: 48*3 GDN projections + 16*4 attention
  // projections + 8*3 MLP projections (layers 56-63) + lm_head = 233 modules,
  // each shipping `weight` + `weight_scale` = 466 tensors.
  CHECK(modules["group_0"].size() == 233);
  CHECK(tensors["group_0"] == 466);
  // NVFP4 W4A4, `nvfp4-pack-quantized`: 56*3 = 168 MLP projections, each
  // shipping weight_packed + weight_scale + weight_global_scale +
  // input_global_scale = 672 tensors.
  CHECK(modules["group_1"].size() == 168);
  CHECK(tensors["group_1"] == 672);
  // On the `ignore` list: the GDN in_proj_a/in_proj_b/norm and the layer's own
  // `linear_attn` scalars, the 27 vision blocks, the merger, and the whole MTP
  // head via `re:^mtp.*`.
  CHECK(modules["IGNORED"].size() == 317);
  CHECK(tensors["IGNORED"] == 475);
  // Matched no target and is not on the ignore list: norms, conv1d, the
  // embedding table and the patch/position embeddings. Unquantized because the
  // config never names them, which is a DIFFERENT fact from being ignored — a
  // resolver that ignored everything would show 792 here and 0 above.
  CHECK(modules["UNMATCHED"].size() == 267);
  CHECK(tensors["UNMATCHED"] == 323);
  // The `kv_cache_scheme` scales: one k_scale + one v_scale on each of the 16
  // full-attention layers.
  CHECK(tensors["KV_CACHE_SCALE"] == 32);
  CHECK(modules["KV_CACHE_SCALE"].size() == 16);

  // `std::map::operator[]` INSERTS, so the absence assertion is a `count`:
  // reading `tensors["UNCLASSIFIED"]` would create the sixth bucket it is
  // asserting does not exist, and the size check below would then read 6.
  CHECK(tensors.count("UNCLASSIFIED") == 0);
  int sum = 0;
  for (const auto& kv : tensors) sum += kv.second;
  CHECK(sum == kIndexNameCount);
  CHECK(tensors.size() == 5);

  // The kv_cache_scheme itself, read rather than assumed: 8-bit float,
  // per-tensor, static, symmetric.
  const ct::KvCacheScheme& kv = c.kv_cache_scheme();
  CHECK(kv.present);
  CHECK(kv.num_bits == 8);
  CHECK(kv.type == "float");
  CHECK(kv.strategy == "tensor");
  CHECK_FALSE(kv.dynamic);
  CHECK(kv.symmetric);
}

// ── (3) the layer-56 boundary, and the ORDER that decides it ────────────────
TEST_CASE("Qwen3.8-27B-NVFP4: the MLP layer-56 boundary resolves by config_groups ORDER") {
  const ct::Config& c = ReleasedCtConfig();
  const std::string p = "model.language_model.layers.";

  for (int l : {0, 1, 27, 54, 55}) {
    const ct::ModuleScheme s = c.Resolve(p + std::to_string(l) + ".mlp.gate_proj");
    CAPTURE(l);
    CHECK(s.group == "group_1");
    CHECK(s.kind == ct::SchemeKind::kNvfp4W4A4);
    CHECK(s.weights.strategy == "tensor_group");
    CHECK(s.weights.group_size == 16);
    CHECK(s.weights.actorder == "static");
    CHECK(s.input_activations.dynamic_str == "local");
    CHECK(s.unsupported_reason.empty());
  }
  for (int l : {56, 57, 60, 63}) {
    for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
      const ct::ModuleScheme s =
          c.Resolve(p + std::to_string(l) + ".mlp." + proj);
      CAPTURE(l);
      CAPTURE(proj);
      CHECK(s.group == "group_0");
      // BOTH groups' targets match this name. `group_0` wins only because it
      // is declared first and upstream takes the first match.
      CHECK(s.target.find("56|57|58|59|60|61|62|63") != std::string::npos);
      CHECK(s.kind == ct::SchemeKind::kUnsupported);
      CHECK(s.weights.strategy == "channel");
      CHECK(s.input_activations.dynamic);
      CHECK(s.input_activations.strategy == "token");
    }
  }
  // The negative control for the boundary: layer 5's `gate_proj` also matches
  // `group_0`'s FIRST target list only if the layer regex is wrong.
  CHECK(c.Resolve(p + "5.mlp.gate_proj").group == "group_1");
  // And a layer index that merely CONTAINS 56 is not layer 56.
  CHECK(c.Resolve(p + "6.mlp.gate_proj").group == "group_1");
}

// ── (4) `ignore` is EXACT MATCH, not a prefix ───────────────────────────────
TEST_CASE("Qwen3.8-27B-NVFP4: the ignore list splits the GDN block exactly where the config does") {
  const ct::Config& c = ReleasedCtConfig();
  const std::string la = "model.language_model.layers.0.linear_attn";

  // Ignored: the module itself (which owns A_log / dt_bias), its norm, and the
  // two low-rank input projections.
  CHECK(c.Resolve(la).ignored);
  CHECK(c.Resolve(la + ".norm").ignored);
  CHECK(c.Resolve(la + ".in_proj_a").ignored);
  CHECK(c.Resolve(la + ".in_proj_b").ignored);
  // NOT ignored, and quantized: the three the `group_0` target names. A prefix
  // match on `...linear_attn` would swallow all five.
  for (const char* proj : {".in_proj_qkv", ".in_proj_z", ".out_proj"}) {
    CAPTURE(proj);
    const ct::ModuleScheme s = c.Resolve(la + proj);
    CHECK_FALSE(s.ignored);
    CHECK(s.group == "group_0");
    CHECK(s.weights.num_bits == 8);
    CHECK(s.weights.strategy == "channel");
  }
  // `conv1d` is neither: no target names it and no ignore entry does.
  const ct::ModuleScheme conv = c.Resolve(la + ".conv1d");
  CHECK_FALSE(conv.ignored);
  CHECK_FALSE(conv.matched);
  // The whole MTP head, through the ONE regex entry on the ignore list.
  CHECK(c.Resolve("mtp.layers.0.self_attn.q_proj").ignored);
  CHECK(c.Resolve("mtp.fc").ignored);
  // `lm_head` is a `group_0` target and is NOT ignored.
  const ct::ModuleScheme head = c.Resolve("lm_head");
  CHECK_FALSE(head.ignored);
  CHECK(head.group == "group_0");
}

// ── upstream's own edge case: an ATTENTION-only config group is DROPPED ─────
//
// `from_config` (compressed_tensors.py:230-246) removes a config group whose
// targets are exactly one `*Attention` entry, because attention quantization on
// its own is coupled to the KV-cache scheme rather than to a linear method. The
// 27B artifact declares no such group, so this is ported for the mirror rather
// than for the artifact: keeping the group would refuse a checkpoint that
// upstream loads, and the divergence would be silent because no shipped tensor
// name would change.
TEST_CASE("compressed-tensors: a config group whose ONE target is *Attention is dropped, as upstream drops it") {
  nlohmann::json q = ReleasedConfig().at("quantization_config");
  q["config_groups"]["group_2"] = q["config_groups"]["group_0"];
  q["config_groups"]["group_2"]["targets"] = nlohmann::json::array({"Qwen3Attention"});
  const ct::Config c = ct::Config::FromQuantizationConfig(q);
  // Two groups, not three: `group_2` never enters the table.
  CHECK(c.group_count() == 2);
  CHECK_FALSE(c.Resolve("Qwen3Attention").matched);
  // A MULTI-target group ending in `Attention` is NOT dropped — upstream's
  // condition is `len(targets) == 1 AND targets[0].endswith("Attention")`, and
  // widening it to "any target ends with Attention" would drop real groups.
  nlohmann::json q2 = ReleasedConfig().at("quantization_config");
  q2["config_groups"]["group_2"] = q2["config_groups"]["group_0"];
  q2["config_groups"]["group_2"]["targets"] =
      nlohmann::json::array({"Qwen3Attention", "re:.*\\.foo_proj$"});
  const ct::Config c2 = ct::Config::FromQuantizationConfig(q2);
  CHECK(c2.group_count() == 3);
  CHECK(c2.Resolve("Qwen3Attention").group == "group_2");
}

// ── the production loader ────────────────────────────────────────────────────
//
// Everything below drives `vllm::LoadQwen3_5Dense` — the loader every consumer
// of a Qwen3.5-family safetensors checkpoint arrives through — on a ONE-LAYER
// synthetic checkpoint that carries this artifact's REAL `quantization_config`
// and this artifact's REAL module names. The tensors are tiny; the config is
// the shipped one, so what is exercised is the resolution the shipped file asks
// for and not a fixture's paraphrase of it.
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

size_t ElemSize(const std::string& dtype) {
  if (dtype == "BF16") return 2;
  if (dtype == "F32") return 4;
  return 1;  // U8 / F8_E4M3
}

std::string U64Le(uint64_t v) {
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i) out[static_cast<size_t>(i)] =
      static_cast<char>((v >> (8 * i)) & 0xffu);
  return out;
}

// A whole safetensors blob from `specs`, filled with finite positive values so
// a scale the loader reads is never a NaN.
std::string BuildSafetensors(const std::vector<Spec>& specs) {
  std::string header = "{";
  std::string body;
  uint64_t offset = 0;
  for (size_t i = 0; i < specs.size(); ++i) {
    const int64_t n = Numel(specs[i].shape);
    const std::string& dtype = specs[i].dtype;
    const auto nbytes = static_cast<uint64_t>(n) * ElemSize(dtype);
    if (i != 0) header += ",";
    header += "\"" + specs[i].name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
    for (size_t d = 0; d < specs[i].shape.size(); ++d) {
      if (d != 0) header += ",";
      header += std::to_string(specs[i].shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + nbytes) + "]}";
    offset += nbytes;
    const size_t at = body.size();
    body.resize(at + static_cast<size_t>(nbytes));
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
             ("vllm_qwen38_nvfp4_" + std::to_string(counter++) + ".safetensors"))
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

// NVFP4 packs two elements per byte in groups of 16, so BOTH widths that
// appear as an in_dim must be multiples of 16: `gate_proj`/`up_proj` read
// `kH` and `down_proj` reads `kFfn`. At kH=8 the loader refused with
// "NVFP4 in_dim must be a multiple of 16" before any scheme was resolved.
constexpr int64_t kH = 32;   // hidden
constexpr int64_t kFfn = 64;
constexpr int64_t kHead = 4;
constexpr int64_t kQ = 8;
constexpr int64_t kKv = 4;

const char* kPrefix = "model.language_model.";

// The NVFP4 (`group_1`) spelling of one projection: packed nibbles, an F8
// group scale, and the two F32 global scales this artifact ships on every
// NVFP4 projection.
void AppendNvfp4(std::vector<Spec>& out, const std::string& proj, int64_t n,
                 int64_t k) {
  out.push_back({proj + ".weight_packed", {n, k / 2}, "U8"});
  out.push_back({proj + ".weight_scale", {n, k / 16}, "F8_E4M3"});
  out.push_back({proj + ".weight_global_scale", {1}, "F32"});
  out.push_back({proj + ".input_global_scale", {1}, "F32"});
}

// The FP8 (`group_0`) spelling: an F8 weight beside a PER-OUTPUT-CHANNEL BF16
// scale, and NO `input_scale` — the activation scheme is dynamic, so the
// checkpoint has none to ship. Exactly what the artifact's header shows.
void AppendFp8Channel(std::vector<Spec>& out, const std::string& proj,
                      int64_t n, int64_t k) {
  out.push_back({proj + ".weight", {n, k}, "F8_E4M3"});
  out.push_back({proj + ".weight_scale", {n, 1}, "BF16"});
}

// A one-layer full-attention backbone. `attn_fp8` picks the `group_0` spelling
// for the attention tower; the MLP is always the `group_1` NVFP4 spelling,
// because layer 0 is on the NVFP4 side of the boundary.
std::vector<Spec> OneLayerSpecs(bool attn_fp8, bool kv_scales) {
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
  if (attn_fp8) {
    AppendFp8Channel(s, sa + "q_proj", kQ, kH);
    AppendFp8Channel(s, sa + "k_proj", kKv, kH);
    AppendFp8Channel(s, sa + "v_proj", kKv, kH);
    AppendFp8Channel(s, sa + "o_proj", kH, kQ);
  } else {
    s.push_back({sa + "q_proj.weight", {kQ, kH}});
    s.push_back({sa + "k_proj.weight", {kKv, kH}});
    s.push_back({sa + "v_proj.weight", {kKv, kH}});
    s.push_back({sa + "o_proj.weight", {kH, kQ}});
  }
  if (kv_scales) {
    s.push_back({sa + "k_scale", {1}, "BF16"});
    s.push_back({sa + "v_scale", {1}, "BF16"});
  }
  AppendNvfp4(s, mlp + "gate_proj", kFfn, kH);
  AppendNvfp4(s, mlp + "up_proj", kFfn, kH);
  AppendNvfp4(s, mlp + "down_proj", kH, kFfn);
  return s;
}

// The shipped `quantization_config`, optionally narrowed. Narrowing is how a
// case isolates ONE refusal: with every group present the FP8 tower refuses
// first and nothing downstream of it is ever observed.
nlohmann::json QuantConfig(bool keep_group_0, bool keep_kv) {
  nlohmann::json q = ReleasedConfig().at("quantization_config");
  if (!keep_group_0) q["config_groups"].erase("group_0");
  if (!keep_kv) q.erase("kv_cache_scheme");
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

bool Names(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

// ── (5) the FP8 arm is refused BY NAME, through the production loader ───────
TEST_CASE("Qwen3.8-27B-NVFP4 loader: the per-channel dynamic FP8 group is refused by name") {
  const std::string message = LoadFailure(
      OneLayerSpecs(/*attn_fp8=*/true, /*kv_scales=*/false),
      OneLayerConfig(QuantConfig(/*keep_group_0=*/true, /*keep_kv=*/false)));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());

  // It names the projection, the group, the format, and BOTH missing pieces.
  CHECK(Names(message, "self_attn.q_proj"));
  CHECK(Names(message, "group_0"));
  CHECK(Names(message, "float-quantized"));
  CHECK(Names(message, "channel"));
  CHECK(Names(message, "dynamic"));
  CHECK(Names(message, "compressed-tensors"));
  // And it does NOT report a missing tensor. Before this row the loader said
  // `tensor not found: ...input_scale`, which is a statement about a
  // checkpoint that is complete: this artifact ships ZERO `*.input_scale`
  // because its activation scheme is dynamic, exactly as the config declares.
  CHECK_FALSE(Names(message, "tensor not found"));
}

// ── (6) `k_scale` / `v_scale` are refused BY NAME, not ignored ──────────────
TEST_CASE("Qwen3.8-27B-NVFP4 loader: the kv_cache_scheme scales are refused by name") {
  // `group_0` removed so the attention tower is unquantized bf16 and this case
  // observes the KV refusal rather than the FP8 one.
  const std::string message = LoadFailure(
      OneLayerSpecs(/*attn_fp8=*/false, /*kv_scales=*/true),
      OneLayerConfig(QuantConfig(/*keep_group_0=*/false, /*keep_kv=*/true)));
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "k_scale"));
  CHECK(Names(message, "v_scale"));
  CHECK(Names(message, "kv_cache_scheme"));
  // Silently ignoring a declared KV-cache quantization scheme is the defect
  // class a token gate cannot see: the tokens still match while the cache
  // holds twice the bytes at the wrong precision.
  CHECK_FALSE(Names(message, "tensor not found"));
}

// ── the negative control: the NVFP4 half still LOADS ────────────────────────
//
// Without this, cases (5) and (6) are satisfied by a loader that refuses every
// compressed-tensors checkpoint. The NVFP4 group is the half of this artifact
// that already works, and it must keep working.
TEST_CASE("Qwen3.8-27B-NVFP4 loader: the NVFP4 group still loads, unrefused") {
  const vllm::HfConfig config =
      OneLayerConfig(QuantConfig(/*keep_group_0=*/false, /*keep_kv=*/false));
  const std::vector<Spec> specs =
      OneLayerSpecs(/*attn_fp8=*/false, /*kv_scales=*/false);
  const std::string message = LoadFailure(specs, config);
  INFO("refusal was: " << message);
  REQUIRE(message.empty());

  const TempFile file(BuildSafetensors(specs));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  const vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5Dense(shards, config);
  REQUIRE(w.layers.size() == 1);
  // The MLP is fp4-RESIDENT, not dequantized to bf16. A silent dequant passes
  // every token gate and defeats the whole point of a quantized arm, so the
  // assertion is on the memory format: packed nibbles, one byte per two
  // elements, and the bf16 slot left empty.
  const auto& mlp = w.layers[0].mlp;
  REQUIRE_FALSE(mlp.gate_proj_fp4.packed.Empty());
  CHECK(mlp.gate_proj_fp4.n == kFfn);
  CHECK(mlp.gate_proj_fp4.k == kH);
  CHECK(mlp.gate_proj_fp4.packed.shape[1] == kH / 2);
  CHECK(mlp.gate_proj_fp4.scale.shape[1] == kH / 16 + (kH % 16 != 0 ? 1 : 0));
  CHECK(mlp.gate_up_proj.Empty());
  CHECK(mlp.down_proj.Empty());
}

// ── the LIVE arm, env-gated ─────────────────────────────────────────────────
//
// The committed manifests are only as good as the file they were captured
// from. This case re-reads the mirrored artifact's own header and compares it
// to the manifest name by name, so a re-quantization under the same name — this
// publisher has done it twice in this family — fails here instead of being
// carried forward silently.
//
//   VLLM_CPP_QWEN38_27B_NVFP4_DIR=/mnt/nas_share/checkpoints/qwen3.8-27b-nvfp4
//   ./build/tests/test_qwen38_27b_nvfp4_arm
//
// The artifact is pinned with its size and sha256 in `docs/USAGE.md`.
TEST_CASE("Qwen3.8-27B-NVFP4 live: the mirrored header still IS the committed manifest") {
  const char* dir = std::getenv("VLLM_CPP_QWEN38_27B_NVFP4_DIR");
  if (dir == nullptr) {
    MESSAGE(
        "SKIPPED: set VLLM_CPP_QWEN38_27B_NVFP4_DIR to the mirrored "
        "unsloth/Qwen3.8-27B-NVFP4 directory to re-verify the committed "
        "manifest and config against the shipped bytes (docs/USAGE.md pins "
        "the artifact)");
    return;
  }
  const std::filesystem::path root(dir);
  const std::filesystem::path shard = root / "model.safetensors";
  REQUIRE_MESSAGE(std::filesystem::exists(shard),
                  "no model.safetensors under " << dir);
  MESSAGE("resolved artifact directory: " << root.string());

  std::error_code ec;
  const auto size = static_cast<int64_t>(std::filesystem::file_size(shard, ec));
  REQUIRE_FALSE(ec);
  CHECK(size == kMainShardBytes);

  std::ifstream in(shard.string(), std::ios::binary);
  REQUIRE(in.good());
  char len_le[8];
  in.read(len_le, 8);
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i)
    header_len = (header_len << 8) | static_cast<uint8_t>(len_le[i]);
  CHECK(header_len == 251128);
  std::string payload(static_cast<size_t>(header_len), '\0');
  in.read(payload.data(), static_cast<std::streamsize>(header_len));
  REQUIRE(in.good());
  const nlohmann::json header = nlohmann::json::parse(payload);

  int64_t data_end = 0;
  std::map<std::string, std::pair<std::string, std::vector<int64_t>>> live;
  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto offs = it.value().at("data_offsets");
    data_end = std::max<int64_t>(data_end, offs[1].get<int64_t>());
    live[it.key()] = {it.value().at("dtype").get<std::string>(),
                      it.value().at("shape").get<std::vector<int64_t>>()};
  }
  // The semantic verification `AGENTS.md` requires in place of a remote hash.
  CHECK(8 + static_cast<int64_t>(header_len) + data_end == kMainShardBytes);
  CHECK(live.size() == static_cast<size_t>(vllm_test::kQwen38_27bNvfp4TensorCount));

  int mismatches = 0;
  for (const ManifestTensor& t : Manifest()) {
    if (std::string(t.shard) != "model.safetensors") continue;
    const auto it = live.find(t.name);
    if (it == live.end() || it->second.first != t.dtype ||
        it->second.second != t.shape) {
      if (mismatches < 5) MESSAGE("manifest/live mismatch: " << t.name);
      ++mismatches;
    }
  }
  CHECK(mismatches == 0);

  // The refusal the PRODUCTION loader raises, derived from the shipped names
  // and the shipped config rather than from a one-layer synthetic. Cases (5)
  // and (6) prove the loader reaches this text; this proves the real artifact
  // is what produces it, and that the 233 is the checkpoint's own count.
  std::vector<std::string> live_names;
  live_names.reserve(live.size());
  for (const auto& kv : live) live_names.push_back(kv.first);
  const std::string live_refusal =
      ct::RefusalForHfConfigRaw(ReleasedConfig(), live_names);
  MESSAGE("live refusal: " << live_refusal);
  CHECK(Names(live_refusal, "233 module(s) resolve to config_groups \"group_0\""));
  CHECK(Names(live_refusal, "float-quantized"));
  CHECK(Names(live_refusal, "lm_head"));
  CHECK_FALSE(Names(live_refusal, "tensor not found"));

  // The committed config fixture is the shipped one, byte for byte.
  const std::filesystem::path cfg = root / "config.json";
  if (std::filesystem::exists(cfg)) {
    std::ifstream live_cfg(cfg.string());
    nlohmann::json shipped;
    live_cfg >> shipped;
    CHECK(shipped == ReleasedConfig());
  }
}
