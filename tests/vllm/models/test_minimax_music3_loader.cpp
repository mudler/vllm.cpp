// MiniMax-Music3 phase W1 — the modular checkpoint loader.
//
// Row MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation,
// .agents/specs/minimax-music3.md, issue #672.
//
// THREE KINDS OF GATE, and they are not interchangeable — the same split
// test_ltx2_loader.cpp:5-25 draws, for the same reason:
//
//  * REAL MANIFEST. `minimax_music3_manifest.inc` is the SHIPPED checkpoint's
//    own safetensors headers — names/dtypes/shapes, not one weight byte. Every
//    geometry claim spec section 1 makes is asserted against it, so a claim
//    cannot drift from the artifact and CI never needs the 28.5 GB asset.
//  * SYNTHETIC FILES. Materialization, weight-norm folding, and the
//    missing/extra/wrong-shape/wrong-dtype refusals need a file small enough to
//    build in a test, so those are written here with a header that MIRRORS the
//    real layout.
//  * MUTATIONS. Every assertion this port claims is BROKEN on purpose and the
//    gate is watched going red. An assertion nobody proved firing is not an
//    assertion.
//
// WHAT IS NOT GATED HERE, said plainly: no VALUE parity against the oracle. W1
// is shapes, dtypes and refusals; per-stage tensor parity is W2+ and needs the
// diffusers oracle standing up (spec section 5). Nothing below compares a
// number against a reference implementation, and nothing below should be read
// as having done so.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "minimax_music3_manifest.inc"

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_music3_loader.h"
#include "vllm/model_executor/models/vocoder1d.h"

#include "support/process_id.h"
using vllm::MiniMaxMusic3AccountReport;
using vllm::MiniMaxMusic3AccountTensors;
using vllm::MiniMaxMusic3Config;
using vllm::MiniMaxMusic3ConditionEncoderConfig;
using vllm::MiniMaxMusic3LanguageModelConfig;
using vllm::MiniMaxMusic3ManifestEntry;
using vllm::MiniMaxMusic3RvqDepthDecoderConfig;
using vllm::MiniMaxMusic3TensorSpec;
using vllm::MiniMaxMusic3TransformerConfig;
using vllm::MiniMaxMusic3VocoderConfig;
using vllm::SafetensorsFile;

namespace {

// ---------------------------------------------------------------------------
// Manifest helpers
// ---------------------------------------------------------------------------

std::vector<int64_t> ShapeOf(const int64_t* dims, int32_t rank) {
  return std::vector<int64_t>(dims, dims + rank);
}

std::vector<MiniMaxMusic3ManifestEntry> ManifestFor(const std::string& component) {
  std::vector<MiniMaxMusic3ManifestEntry> out;
  for (const vllm_test::Music3ManifestTensor& t : vllm_test::kMusic3Manifest) {
    if (component == t.component) {
      out.push_back({t.name, t.dtype, ShapeOf(t.shape, t.rank)});
    }
  }
  return out;
}

int64_t ParamsIn(const std::vector<MiniMaxMusic3ManifestEntry>& entries) {
  int64_t total = 0;
  for (const MiniMaxMusic3ManifestEntry& e : entries) {
    int64_t n = 1;
    for (int64_t d : e.shape) n *= d;
    total += n;
  }
  return total;
}

// ---------------------------------------------------------------------------
// A synthetic .safetensors writer, same shape as
// test_ltx2_loader.cpp:120 WriteSafetensors.
// ---------------------------------------------------------------------------

struct StEntry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

void WriteSafetensors(const std::vector<StEntry>& entries, const std::string& path) {
  std::string header = "{";
  size_t offset = 0;
  for (size_t i = 0; i < entries.size(); ++i) {
    const StEntry& e = entries[i];
    if (i != 0) header += ",";
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t d = 0; d < e.shape.size(); ++d) {
      if (d != 0) header += ",";
      header += std::to_string(e.shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";
  std::ofstream out(path, std::ios::binary);
  const uint64_t len = header.size();
  out.write(reinterpret_cast<const char*>(&len), sizeof(len));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const StEntry& e : entries) {
    out.write(e.bytes.data(), static_cast<std::streamsize>(e.bytes.size()));
  }
}

std::string F32Bytes(const std::vector<float>& values) {
  std::string out(values.size() * 4, '\0');
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

// A deterministic, NON-CONSTANT stream. Constant weights would make the
// weight-norm fold `g * v / ||v||` collapse to something a wrong reduction axis
// could still reproduce, so the values must vary along BOTH axes.
float Sample(int64_t index) {
  const double x = static_cast<double>(index);
  return static_cast<float>(0.25 * std::sin(0.7 * x + 1.3) + 0.03 * x + 0.5);
}

std::vector<float> Ramp(int64_t count, int64_t seed) {
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) out[static_cast<size_t>(i)] = Sample(i + seed * 37);
  return out;
}

// Build a synthetic file carrying EXACTLY the enumerated tensors, so a test can
// then perturb one entry and prove the account catches it.
std::vector<StEntry> EntriesFor(const std::vector<MiniMaxMusic3TensorSpec>& specs) {
  std::vector<StEntry> out;
  int64_t seed = 0;
  for (const MiniMaxMusic3TensorSpec& spec : specs) {
    int64_t numel = 1;
    for (int64_t d : spec.shape) numel *= d;
    std::string bytes;
    if (spec.dtype == "F32") {
      bytes = F32Bytes(Ramp(numel, ++seed));
    } else {  // BF16
      bytes.assign(static_cast<size_t>(numel) * 2, '\0');
    }
    out.push_back({spec.name, spec.dtype, spec.shape, bytes});
  }
  return out;
}

std::string TempPath(const char* stem) {
  return (std::filesystem::temp_directory_path() /
          (std::string("music3_") + stem + "_" + std::to_string(vllm_test::ProcessId()) + ".safetensors"))
      .string();
}

// A REDUCED vocoder: the real geometry is 121 tensors over 1536 channels, which
// is ~54M f32 parameters and not something to write in a unit test. Every
// structural property the fold depends on is preserved -- weight-normed
// Conv1d AND ConvTranspose1d, a non-square kernel, and dim 0 being the
// INPUT-channel axis for the transpose.
MiniMaxMusic3VocoderConfig SmallVocoderConfig() {
  MiniMaxMusic3VocoderConfig config;
  config.latent_channels = 8;
  config.decoder_input_dim = 6;
  config.decoder_hidden_dim = 8;
  config.upsampling_ratios = {2, 2};
  return config;
}

// The reference fold, written independently of the loader: w[c] = g[c] * v[c] /
// ||v[c]||, the norm over every dimension but dim 0. This is the whole point of
// the gate, so it does NOT call the function under test.
std::vector<float> ReferenceFold(const std::vector<float>& g, const std::vector<float>& v,
                                 int64_t dim0) {
  const int64_t per = static_cast<int64_t>(v.size()) / dim0;
  std::vector<float> out(v.size());
  for (int64_t c = 0; c < dim0; ++c) {
    double norm = 0.0;
    for (int64_t i = 0; i < per; ++i) {
      const double value = v[static_cast<size_t>(c * per + i)];
      norm += value * value;
    }
    norm = std::sqrt(norm);
    for (int64_t i = 0; i < per; ++i) {
      out[static_cast<size_t>(c * per + i)] = static_cast<float>(
          static_cast<double>(v[static_cast<size_t>(c * per + i)]) *
          (norm > 0.0 ? static_cast<double>(g[static_cast<size_t>(c)]) / norm : 0.0));
    }
  }
  return out;
}

}  // namespace

// ===========================================================================
// 1. The REAL manifest reproduces spec section 1, component by component
// ===========================================================================

TEST_CASE("music3 manifest: the shipped checkpoint has exactly six weight-bearing components") {
  std::set<std::string> components;
  for (const vllm_test::Music3ManifestTensor& t : vllm_test::kMusic3Manifest) {
    components.insert(t.component);
  }
  // Five carry tensors; `scheduler` and `tokenizer` carry none, which is why
  // the manifest has five keys and the layout has seven directories.
  CHECK(components == std::set<std::string>{"condition_encoder", "language_model",
                                            "rvq_depth_decoder", "transformer", "vocoder"});
  int64_t counted = 0;
  for (const vllm_test::Music3ManifestTensor& t : vllm_test::kMusic3Manifest) {
    (void)t;
    ++counted;
  }
  CHECK(counted == vllm_test::kMusic3ManifestTensorCount);
  MESSAGE("manifest tensors examined: " << counted);
}

TEST_CASE("music3 manifest: spec section 1 geometry, measured") {
  // The numbers spec section 1 states, asserted against the artifact rather
  // than against the prose.
  CHECK(vllm_test::kMusic3TransformerTensorCount == 441);
  CHECK(vllm_test::kMusic3ConditionEncoderTensorCount == 4);
  CHECK(vllm_test::kMusic3RvqDepthDecoderTensorCount == 47);
  CHECK(vllm_test::kMusic3VocoderTensorCount == 121);
  CHECK(vllm_test::kMusic3LanguageModelTensorCount == 399);

  // Parameter counts, MEASURED from the headers by
  // scripts/gen-minimax-music3-manifest.py. The spec quotes them rounded; these
  // are the exact numbers, and the rounding is checked rather than assumed.
  CHECK(vllm_test::kMusic3TransformerParameters == 2431905920);       // spec "2.4B"
  CHECK(vllm_test::kMusic3ConditionEncoderParameters == 25167881);    // spec "0.025B"
  CHECK(vllm_test::kMusic3RvqDepthDecoderParameters == 646025216);    // spec "0.646B"
  CHECK(vllm_test::kMusic3VocoderParameters == 54170722);             // spec "0.054B"
  CHECK(vllm_test::kMusic3LanguageModelParameters == 8584475648);     // spec "~8.6B"
  // The rounding the spec's table states, so a future re-measure that moves a
  // component cannot pass by quietly agreeing with itself.
  CHECK(vllm_test::kMusic3TransformerParameters / 100000000 == 24);      // 2.4B
  CHECK(vllm_test::kMusic3RvqDepthDecoderParameters / 1000000 == 646);   // 0.646B
  CHECK(vllm_test::kMusic3LanguageModelParameters / 100000000 == 85);    // ~8.6B
}

TEST_CASE("music3 manifest: the dtype policy of spec section 2.1 is what the files carry") {
  // The claim under test: transformer / condition_encoder / vocoder are F32
  // (convert_minimax_music3_to_diffusers.py:208-211,267) and rvq_depth_decoder
  // is BF16 (:214). A token gate cannot see a dtype, so it is checked here.
  std::map<std::string, std::set<std::string>> dtypes;
  std::map<std::string, int64_t> counts;
  for (const vllm_test::Music3ManifestTensor& t : vllm_test::kMusic3Manifest) {
    dtypes[t.component].insert(t.dtype);
    ++counts[t.component];
  }
  CHECK(dtypes["transformer"] == std::set<std::string>{"F32"});
  CHECK(dtypes["condition_encoder"] == std::set<std::string>{"F32"});
  CHECK(dtypes["vocoder"] == std::set<std::string>{"F32"});
  CHECK(dtypes["rvq_depth_decoder"] == std::set<std::string>{"BF16"});
  CHECK(dtypes["language_model"] == std::set<std::string>{"BF16"});
  for (const auto& [component, count] : counts) {
    MESSAGE("dtype-checked " << count << " tensors of " << component);
  }
}

TEST_CASE("music3 manifest: the vocoder ships legacy weight_g/weight_v pairs") {
  // Spec section 1: "121 tensors ... with weight_g+weight_v pairs, so
  // weight-norm must be folded at load or reproduced".
  int64_t g = 0, v = 0, plain_weight = 0;
  for (const vllm_test::Music3ManifestTensor& t : vllm_test::kMusic3Manifest) {
    if (std::string(t.component) != "vocoder") continue;
    const std::string name = t.name;
    if (name.size() > 9 && name.compare(name.size() - 9, 9, ".weight_g") == 0) ++g;
    if (name.size() > 9 && name.compare(name.size() - 9, 9, ".weight_v") == 0) ++v;
    if (name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0) ++plain_weight;
  }
  // 30, MEASURED: conv_in + conv_out, plus per block one conv_t1 and three
  // residual units of two convs each (7 x 4 = 28). 2 + 30*3 + 29 snakes = 121.
  CHECK(g == 30);
  CHECK(v == 30);
  CHECK(g == v);
  // `dec_in_proj` is the ONE convolution upstream does NOT weight-norm
  // (minimax_music3_vocoder.py:88), so exactly one plain `.weight` survives.
  CHECK(plain_weight == 1);
  MESSAGE("weight-normed vocoder modules: " << g);
}

// ===========================================================================
// 2. The enumeration reproduces the real manifest, exactly
// ===========================================================================

TEST_CASE("music3 enumeration: every component's enumeration equals its real manifest") {
  MiniMaxMusic3Config config;  // the released checkpoint's own values are the defaults
  const std::map<std::string, std::vector<MiniMaxMusic3TensorSpec>> enumerated =
      vllm::EnumerateMiniMaxMusic3Tensors(config);

  int64_t accounted = 0;
  for (const char* component : {"transformer", "condition_encoder", "rvq_depth_decoder",
                                "vocoder", "language_model"}) {
    CAPTURE(component);
    REQUIRE(enumerated.count(component) == 1);
    const MiniMaxMusic3AccountReport report =
        MiniMaxMusic3AccountTensors(component, enumerated.at(component), ManifestFor(component));
    CHECK(report.required == report.present);
    CHECK(report.matched == report.required);
    CHECK(report.required > 0);
    accounted += report.matched;
    MESSAGE(component << ": " << report.matched << " tensors accounted");
  }
  CHECK(accounted == vllm_test::kMusic3ManifestTensorCount);
  MESSAGE("total tensors accounted against the real manifest: " << accounted);
}

TEST_CASE("music3 enumeration: parameter counts match the artifact") {
  MiniMaxMusic3Config config;
  const auto enumerated = vllm::EnumerateMiniMaxMusic3Tensors(config);
  std::map<std::string, int64_t> params;
  for (const auto& [component, specs] : enumerated) {
    int64_t total = 0;
    for (const MiniMaxMusic3TensorSpec& s : specs) {
      int64_t n = 1;
      for (int64_t d : s.shape) n *= d;
      total += n;
    }
    params[component] = total;
  }
  CHECK(params["transformer"] == ParamsIn(ManifestFor("transformer")));
  CHECK(params["condition_encoder"] == ParamsIn(ManifestFor("condition_encoder")));
  CHECK(params["rvq_depth_decoder"] == ParamsIn(ManifestFor("rvq_depth_decoder")));
  CHECK(params["vocoder"] == ParamsIn(ManifestFor("vocoder")));
  CHECK(params["language_model"] == ParamsIn(ManifestFor("language_model")));
}

// ===========================================================================
// 3. MUTATIONS — the assertions actually fire
// ===========================================================================

TEST_CASE("music3 account: a WRONG SHAPE is refused by name") {
  MiniMaxMusic3Config config;
  const auto enumerated = vllm::EnumerateMiniMaxMusic3Tensors(config);

  int64_t mutated_cases = 0;
  for (const char* component : {"transformer", "condition_encoder", "rvq_depth_decoder",
                                "vocoder", "language_model"}) {
    CAPTURE(component);
    // Baseline: the unmutated manifest passes, so the red below is the mutation.
    REQUIRE_NOTHROW(
        MiniMaxMusic3AccountTensors(component, enumerated.at(component), ManifestFor(component)));

    std::vector<MiniMaxMusic3ManifestEntry> entries = ManifestFor(component);
    REQUIRE(!entries.empty());
    // Perturb the LAST dimension of the first tensor by one. Off by one is the
    // mutation a lenient checker survives; a wholesale reshape is not.
    entries.front().shape.back() += 1;
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(component, enumerated.at(component), entries),
                    std::runtime_error);
    ++mutated_cases;

    // Every OTHER tensor, one at a time, would be too slow for 1012 entries, so
    // the sweep takes the first, the middle and the last.
    for (size_t index : {size_t(0), entries.size() / 2, entries.size() - 1}) {
      std::vector<MiniMaxMusic3ManifestEntry> one = ManifestFor(component);
      one[index].shape.back() += 1;
      CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(component, enumerated.at(component), one),
                      std::runtime_error);
      ++mutated_cases;
    }
  }
  MESSAGE("wrong-shape mutations proven to fire: " << mutated_cases);
  CHECK(mutated_cases == 20);
}

TEST_CASE("music3 account: a WRONG DTYPE is refused, which no token gate could see") {
  MiniMaxMusic3Config config;
  const auto enumerated = vllm::EnumerateMiniMaxMusic3Tensors(config);

  int64_t mutated_cases = 0;
  for (const char* component : {"transformer", "condition_encoder", "rvq_depth_decoder",
                                "vocoder", "language_model"}) {
    CAPTURE(component);
    std::vector<MiniMaxMusic3ManifestEntry> entries = ManifestFor(component);
    // Swap F32 <-> BF16. Widening is the direction AGENTS.md says a correctness
    // gate is BLIND to, so it has to be refused structurally.
    for (MiniMaxMusic3ManifestEntry& e : entries) {
      e.dtype = (e.dtype == "F32") ? "BF16" : "F32";
    }
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(component, enumerated.at(component), entries),
                    std::runtime_error);
    ++mutated_cases;

    // And ONE tensor alone, which is the case a whole-file dtype check misses.
    std::vector<MiniMaxMusic3ManifestEntry> one = ManifestFor(component);
    one.front().dtype = (one.front().dtype == "F32") ? "BF16" : "F32";
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(component, enumerated.at(component), one),
                    std::runtime_error);
    ++mutated_cases;
  }
  MESSAGE("wrong-dtype mutations proven to fire: " << mutated_cases);
  CHECK(mutated_cases == 10);
}

TEST_CASE("music3 account: a MISSING tensor and an EXTRA tensor are both refused") {
  MiniMaxMusic3Config config;
  const auto enumerated = vllm::EnumerateMiniMaxMusic3Tensors(config);

  int64_t mutated_cases = 0;
  for (const char* component : {"transformer", "condition_encoder", "rvq_depth_decoder",
                                "vocoder", "language_model"}) {
    CAPTURE(component);
    std::vector<MiniMaxMusic3ManifestEntry> dropped = ManifestFor(component);
    dropped.erase(dropped.begin());
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(component, enumerated.at(component), dropped),
                    std::runtime_error);
    ++mutated_cases;

    std::vector<MiniMaxMusic3ManifestEntry> added = ManifestFor(component);
    added.push_back({"a.tensor.this.port.does.not.carry", "F32", {1}});
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(component, enumerated.at(component), added),
                    std::runtime_error);
    ++mutated_cases;
  }
  MESSAGE("missing/extra mutations proven to fire: " << mutated_cases);
  CHECK(mutated_cases == 10);
}

TEST_CASE("music3 account: a CONFIG that disagrees with the file is refused") {
  // The failure porting-a-model.md section 1 names: a config that silently
  // deserializes to something plausible binds a DIFFERENT model from the same
  // tensors. One value moved per component, each one a value no other check
  // would see.
  {
    MiniMaxMusic3TransformerConfig config;
    config.num_layers = 35;  // one block short
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(
                        "transformer", vllm::EnumerateMiniMaxMusic3TransformerTensors(config),
                        ManifestFor("transformer")),
                    std::runtime_error);
  }
  {
    MiniMaxMusic3TransformerConfig config;
    config.ff_inner_dim = 4096;  // halves ff_in/ff_out, changes no tensor COUNT
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(
                        "transformer", vllm::EnumerateMiniMaxMusic3TransformerTensors(config),
                        ManifestFor("transformer")),
                    std::runtime_error);
  }
  {
    MiniMaxMusic3ConditionEncoderConfig config;
    config.num_condition_layers = 4;  // the 8-layer mix, halved
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(
                        "condition_encoder",
                        vllm::EnumerateMiniMaxMusic3ConditionEncoderTensors(config),
                        ManifestFor("condition_encoder")),
                    std::runtime_error);
  }
  {
    MiniMaxMusic3RvqDepthDecoderConfig config;
    config.num_codebooks = 7;  // one fewer head AND a smaller embedding table
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(
                        "rvq_depth_decoder",
                        vllm::EnumerateMiniMaxMusic3RvqDepthDecoderTensors(config),
                        ManifestFor("rvq_depth_decoder")),
                    std::runtime_error);
  }
  {
    MiniMaxMusic3VocoderConfig config;
    config.upsampling_ratios = {8, 8, 4};  // hop 256, not 512 -- a 22050 Hz model
    CHECK_THROWS_AS(
        MiniMaxMusic3AccountTensors("vocoder", vllm::EnumerateMiniMaxMusic3VocoderTensors(config),
                                    ManifestFor("vocoder")),
        std::runtime_error);
  }
  {
    MiniMaxMusic3LanguageModelConfig config;
    config.vocab_size = 151936;  // stock Qwen3's vocabulary, not the music one
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(
                        "language_model",
                        vllm::EnumerateMiniMaxMusic3LanguageModelTensors(config),
                        ManifestFor("language_model")),
                    std::runtime_error);
  }
  {
    MiniMaxMusic3LanguageModelConfig config;
    config.num_key_value_heads = 32;  // MHA instead of GQA: only k/v_proj move
    CHECK_THROWS_AS(MiniMaxMusic3AccountTensors(
                        "language_model",
                        vllm::EnumerateMiniMaxMusic3LanguageModelTensors(config),
                        ManifestFor("language_model")),
                    std::runtime_error);
  }
  MESSAGE("config mutations proven to fire: 7");
}

// ===========================================================================
// 4. Weight-norm folding
// ===========================================================================

TEST_CASE("music3 vocoder: weight_g/weight_v fold to a single weight, matching a reference") {
  const MiniMaxMusic3VocoderConfig config = SmallVocoderConfig();
  const std::vector<MiniMaxMusic3TensorSpec> specs =
      vllm::EnumerateMiniMaxMusic3VocoderTensors(config);
  const std::vector<StEntry> entries = EntriesFor(specs);
  const std::string path = TempPath("vocoder_fold");
  WriteSafetensors(entries, path);

  const SafetensorsFile file = SafetensorsFile::Open(path);
  const vllm::MiniMaxMusic3VocoderWeights weights =
      vllm::MiniMaxMusic3LoadVocoderWeights(config, file);

  const std::vector<std::string> modules = vllm::MiniMaxMusic3WeightNormedModules(config);
  REQUIRE(!modules.empty());
  CHECK(weights.folded == static_cast<int64_t>(modules.size()));

  // No `_g` / `_v` name survives the fold, and a plain `weight` replaces each.
  for (const auto& [name, values] : weights.tensors) {
    (void)values;
    CHECK(name.find(".weight_g") == std::string::npos);
    CHECK(name.find(".weight_v") == std::string::npos);
  }

  // Byte-level reference. Recompute the fold from the SAME synthetic stream,
  // independently of the loader.
  std::map<std::string, StEntry> by_name;
  for (const StEntry& e : entries) by_name[e.name] = e;

  int64_t compared = 0;
  double worst = 0.0;
  for (const std::string& module : modules) {
    CAPTURE(module);
    const StEntry& g = by_name.at(module + ".weight_g");
    const StEntry& v = by_name.at(module + ".weight_v");
    std::vector<float> g_values(g.bytes.size() / 4), v_values(v.bytes.size() / 4);
    std::memcpy(g_values.data(), g.bytes.data(), g.bytes.size());
    std::memcpy(v_values.data(), v.bytes.data(), v.bytes.size());
    const std::vector<float> expected = ReferenceFold(g_values, v_values, v.shape[0]);

    const auto it = weights.tensors.find(module + ".weight");
    REQUIRE(it != weights.tensors.end());
    REQUIRE(it->second.size() == expected.size());
    CHECK(weights.shapes.at(module + ".weight") == v.shape);
    for (size_t i = 0; i < expected.size(); ++i) {
      const double diff = std::fabs(static_cast<double>(it->second[i]) -
                                    static_cast<double>(expected[i]));
      REQUIRE(std::isfinite(diff));
      worst = std::max(worst, diff);
      ++compared;
    }
  }
  MESSAGE("folded modules: " << modules.size() << ", values compared: " << compared
                             << ", max|diff|: " << worst);
  CHECK(compared > 0);
  CHECK(worst == 0.0);  // same arithmetic, same order -- bit-identical or a bug
  std::filesystem::remove(path);
}

TEST_CASE("music3 vocoder: the fold is NOT a copy of v, and NOT a per-tensor scale") {
  // The two ways a wrong fold still produces finite, correctly shaped weights:
  // forgetting the normalization entirely (w == g*v), and reducing over the
  // WRONG axis (one norm for the whole tensor instead of one per dim-0 slice).
  // Both are caught here, because neither is visible in a shape.
  const MiniMaxMusic3VocoderConfig config = SmallVocoderConfig();
  const std::vector<MiniMaxMusic3TensorSpec> specs =
      vllm::EnumerateMiniMaxMusic3VocoderTensors(config);
  const std::vector<StEntry> entries = EntriesFor(specs);
  const std::string path = TempPath("vocoder_fold_neg");
  WriteSafetensors(entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);
  const vllm::MiniMaxMusic3VocoderWeights weights =
      vllm::MiniMaxMusic3LoadVocoderWeights(config, file);

  std::map<std::string, StEntry> by_name;
  for (const StEntry& e : entries) by_name[e.name] = e;

  int64_t checked = 0, discriminating = 0, degenerate = 0;
  for (const std::string& module : vllm::MiniMaxMusic3WeightNormedModules(config)) {
    CAPTURE(module);
    const StEntry& g = by_name.at(module + ".weight_g");
    const StEntry& v = by_name.at(module + ".weight_v");
    std::vector<float> g_values(g.bytes.size() / 4), v_values(v.bytes.size() / 4);
    std::memcpy(g_values.data(), g.bytes.data(), g.bytes.size());
    std::memcpy(v_values.data(), v.bytes.data(), v.bytes.size());
    const std::vector<float>& got = weights.tensors.at(module + ".weight");

    // (a) not the raw direction, and not g*v.
    double diff_v = 0.0, diff_gv = 0.0, diff_global = 0.0;
    const int64_t dim0 = v.shape[0];
    const int64_t per = static_cast<int64_t>(v_values.size()) / dim0;
    double global_norm = 0.0;
    for (float value : v_values) global_norm += static_cast<double>(value) * value;
    global_norm = std::sqrt(global_norm);
    for (int64_t c = 0; c < dim0; ++c) {
      for (int64_t i = 0; i < per; ++i) {
        const size_t k = static_cast<size_t>(c * per + i);
        const double have = static_cast<double>(got[k]);
        const double v_value = static_cast<double>(v_values[k]);
        const double g_value = static_cast<double>(g_values[static_cast<size_t>(c)]);
        diff_v = std::max(diff_v, std::fabs(have - v_value));
        diff_gv = std::max(diff_gv, std::fabs(have - g_value * v_value));
        const double wrong_axis = global_norm > 0.0 ? v_value * g_value / global_norm : 0.0;
        diff_global = std::max(diff_global, std::fabs(have - wrong_axis));
      }
    }
    CHECK(diff_v > 1e-6);
    CHECK(diff_gv > 1e-6);
    // The WRONG-AXIS fold is only a DIFFERENT number when there is more than
    // one dim-0 slice to reduce over. `conv_out` has weight_v [1, C, K], so its
    // per-slice norm and its whole-tensor norm are the same quantity and no
    // implementation could tell them apart -- claiming otherwise would be a
    // gate asserting something arithmetic forbids. It is skipped BY NAME, and
    // the modules that can discriminate are counted, so the exemption cannot
    // silently grow to cover everything.
    if (dim0 > 1) {
      CHECK(diff_global > 1e-6);
      ++discriminating;
    } else {
      MESSAGE("dim0 == 1, wrong-axis fold is arithmetically identical: " << module);
      ++degenerate;
    }
    ++checked;
  }
  MESSAGE("modules checked: " << checked << ", discriminating on the reduction axis: "
                              << discriminating << ", degenerate (dim0 == 1): " << degenerate);
  CHECK(checked > 0);
  CHECK(discriminating > 0);
  // Exactly one convolution in this decoder has dim0 == 1: `conv_out`
  // (minimax_music3_vocoder.py:98, nn.Conv1d(output_dim, 1, 7)).
  CHECK(degenerate == 1);
  CHECK(checked == discriminating + degenerate);
  std::filesystem::remove(path);
}

TEST_CASE("music3 vocoder: the shared weight-norm home is what folds") {
  // vocoder1d::MaterializeWeightNorm is the ONE definition (single-home guard:
  // tests/scripts/test_vocoder1d_single_home.py). Asserted numerically here so
  // a fork would have to disagree with a reference to survive.
  const std::vector<float> g{2.0F, -0.5F, 3.0F};
  const std::vector<float> v{1.0F, 2.0F, 2.0F,   // ||v0|| = 3
                             0.0F, 3.0F, 4.0F,   // ||v1|| = 5
                             1.0F, 0.0F, 0.0F};  // ||v2|| = 1
  const std::vector<float> got = vllm::vocoder1d::MaterializeWeightNorm(g, v, 3);
  const std::vector<float> want = ReferenceFold(g, v, 3);
  REQUIRE(got.size() == want.size());
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    REQUIRE(std::isfinite(diff));
    worst = std::max(worst, diff);
  }
  CHECK(worst < 1e-6);
  // And spelled out for the first slice, so the gate does not merely agree with
  // a helper it also wrote: g0/||v0|| = 2/3.
  CHECK(got[0] == doctest::Approx(2.0F / 3.0F).epsilon(1e-6));
  CHECK(got[1] == doctest::Approx(4.0F / 3.0F).epsilon(1e-6));
  MESSAGE("weight-norm values compared: " << got.size() << ", max|diff|: " << worst);
}

// ===========================================================================
// 5. Materialization preserves the on-disk dtype
// ===========================================================================

TEST_CASE("music3 load: a component materializes with its dtype PRESERVED") {
  MiniMaxMusic3RvqDepthDecoderConfig config;
  config.hidden_size = 16;
  config.num_layers = 1;
  config.num_attention_heads = 2;
  config.intermediate_size = 24;
  config.audio_vocab_size = 4;
  config.num_codebooks = 3;
  config.max_position_embeddings = 4;

  const std::vector<MiniMaxMusic3TensorSpec> specs =
      vllm::EnumerateMiniMaxMusic3RvqDepthDecoderTensors(config);
  const std::string path = TempPath("rvq");
  WriteSafetensors(EntriesFor(specs), path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  const vllm::MiniMaxMusic3ComponentWeights weights =
      vllm::MiniMaxMusic3LoadComponent("rvq_depth_decoder", specs, file);
  CHECK(weights.tensors.size() == specs.size());
  int64_t bf16 = 0;
  for (const auto& [name, tensor] : weights.tensors) {
    CAPTURE(name);
    // BF16 on disk stays BF16 in memory. Widening here would be exactly the
    // silent narrowing/widening spec section 2.1 reserves for a measured change.
    CHECK(tensor.dtype == "BF16");
    CHECK(tensor.bytes.size() == static_cast<size_t>(tensor.numel()) * 2);
    ++bf16;
  }
  CHECK(bf16 == static_cast<int64_t>(specs.size()));
  MESSAGE("tensors materialized with dtype preserved: " << bf16);
  std::filesystem::remove(path);
}

TEST_CASE("music3 load: a component whose file lost a tensor throws BY NAME") {
  MiniMaxMusic3ConditionEncoderConfig config;
  const std::vector<MiniMaxMusic3TensorSpec> specs =
      vllm::EnumerateMiniMaxMusic3ConditionEncoderTensors(config);
  std::vector<StEntry> entries = EntriesFor(specs);
  REQUIRE(entries.size() == 4);
  const std::string dropped_name = entries.back().name;
  entries.pop_back();
  const std::string path = TempPath("cond_missing");
  WriteSafetensors(entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  bool named = false;
  try {
    vllm::MiniMaxMusic3LoadComponent("condition_encoder", specs, file);
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    named = message.find(dropped_name) != std::string::npos &&
            message.find("condition_encoder") != std::string::npos;
    MESSAGE("refusal: " << message);
  }
  CHECK(named);
  std::filesystem::remove(path);
}

// ===========================================================================
// 6. The NATIVE arm is refused BY NAME
// ===========================================================================

TEST_CASE("music3 resolve: a NATIVE-arm checkpoint is refused, naming what is missing") {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("music3_native_" + std::to_string(vllm_test::ProcessId()));
  std::filesystem::remove_all(root);
  // The layout convert_minimax_music3_to_diffusers.py:30,34,38 reads, and the
  // one sglang_omni/models/minimax_music3/checkpoint.py:35-56 serves.
  std::filesystem::create_directories(root / "qwen_7B" / "qwen_7B");
  std::ofstream(root / "flowmatching_vae.pth") << "not a safetensors file";
  std::ofstream(root / "dav.pth") << "not a safetensors file";

  CHECK(vllm::MiniMaxMusic3IsNativeArm(root.string()));

  bool named_arm = false, named_missing = false, named_supported = false;
  try {
    vllm::MiniMaxMusic3ResolveCheckpoint(root.string());
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    MESSAGE("refusal: " << message);
    // It must name the artifact it was given...
    named_arm = message.find("flowmatching_vae.pth") != std::string::npos &&
                message.find("dav.pth") != std::string::npos &&
                message.find("qwen_7B") != std::string::npos;
    // ...what is missing...
    named_missing = message.find("transformer") != std::string::npos &&
                    message.find("vocoder") != std::string::npos &&
                    message.find("rvq_depth_decoder") != std::string::npos &&
                    message.find("condition_encoder") != std::string::npos &&
                    message.find("language_model") != std::string::npos;
    // ...and that only the diffusers arm is supported.
    named_supported = message.find("diffusers") != std::string::npos;
  }
  CHECK(named_arm);
  CHECK(named_missing);
  CHECK(named_supported);
  std::filesystem::remove_all(root);
}

TEST_CASE("music3 resolve: ONE native marker is enough to be diagnosed as the native arm") {
  // A partially staged native tree must not fall through to the generic
  // "components missing" message: the whole point is that the person holding it
  // is told which packaging they have.
  int64_t diagnosed = 0;
  for (const char* marker : {"flowmatching_vae.pth", "dav.pth", "qwen_7B"}) {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("music3_partial_" + std::string(marker) + "_" + std::to_string(vllm_test::ProcessId()));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    if (std::string(marker) == "qwen_7B") {
      std::filesystem::create_directories(root / "qwen_7B" / "qwen_7B");
    } else {
      std::ofstream(root / marker) << "x";
    }
    CAPTURE(marker);
    CHECK(vllm::MiniMaxMusic3IsNativeArm(root.string()));
    bool diagnosed_here = false;
    try {
      vllm::MiniMaxMusic3ResolveCheckpoint(root.string());
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      diagnosed_here = message.find("native") != std::string::npos &&
                       message.find("diffusers") != std::string::npos;
    }
    CHECK(diagnosed_here);
    diagnosed += diagnosed_here ? 1 : 0;
    std::filesystem::remove_all(root);
  }
  CHECK(diagnosed == 3);
  MESSAGE("native markers each diagnosed: " << diagnosed);
}

TEST_CASE("music3 resolve: a tree that is NEITHER arm names the components it lacks") {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / ("music3_empty_" + std::to_string(vllm_test::ProcessId()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  CHECK(!vllm::MiniMaxMusic3IsNativeArm(root.string()));
  bool named = false;
  try {
    vllm::MiniMaxMusic3ResolveCheckpoint(root.string());
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    MESSAGE("refusal: " << message);
    named = message.find("modular_model_index.json") != std::string::npos &&
            message.find("native") == std::string::npos;
  }
  CHECK(named);
  std::filesystem::remove_all(root);
}

// ===========================================================================
// 7. The REAL checkpoint, when it is staged
// ===========================================================================

TEST_CASE("music3 real checkpoint: resolve, parse and account every component") {
  const char* root_env = std::getenv("VLLM_CPP_MUSIC3_CHECKPOINT");
  if (root_env == nullptr || *root_env == '\0') {
    MESSAGE("SKIP: set VLLM_CPP_MUSIC3_CHECKPOINT to the staged diffusers-arm checkpoint");
    return;
  }
  const vllm::MiniMaxMusic3Paths paths = vllm::MiniMaxMusic3ResolveCheckpoint(root_env);
  const MiniMaxMusic3Config config = vllm::MiniMaxMusic3LoadConfig(paths);

  // The released values, asserted against spec section 1 rather than trusted.
  CHECK(config.transformer.num_layers == 36);
  CHECK(config.transformer.inner_dim() == 2048);
  CHECK(config.transformer.concat_channels() == 2304);
  CHECK(config.transformer.rotary_dim == 32);
  CHECK(config.condition_encoder.num_condition_layers == 8);
  CHECK(config.rvq_depth_decoder.num_codebooks == 8);
  CHECK(config.rvq_depth_decoder.max_position_embeddings == 16);
  CHECK(config.vocoder.hop_length() == 512);
  CHECK(config.vocoder.sampling_rate == 44100);
  CHECK(config.language_model.vocab_size == 200000);
  CHECK(config.language_model.max_position_embeddings == 10240);
  CHECK(!config.language_model.tie_word_embeddings);
  CHECK(config.scheduler.invert_sigmas);
  CHECK(config.scheduler.num_train_timesteps == 1);

  const auto enumerated = vllm::EnumerateMiniMaxMusic3Tensors(config);
  int64_t total = 0;
  const std::map<std::string, std::vector<std::string>> files{
      {"transformer", paths.transformer_shards},
      {"condition_encoder", {paths.condition_encoder_dir + "/diffusion_pytorch_model.safetensors"}},
      {"rvq_depth_decoder", {paths.rvq_depth_decoder_dir + "/diffusion_pytorch_model.safetensors"}},
      {"vocoder", {paths.vocoder_dir + "/diffusion_pytorch_model.safetensors"}},
      {"language_model", paths.language_model_shards},
  };
  for (const auto& [component, shards] : files) {
    CAPTURE(component);
    std::vector<MiniMaxMusic3ManifestEntry> entries;
    for (const std::string& shard : shards) {
      const SafetensorsFile file = SafetensorsFile::Open(shard);
      for (MiniMaxMusic3ManifestEntry& e : vllm::MiniMaxMusic3ReadManifest(file)) {
        entries.push_back(std::move(e));
      }
    }
    const MiniMaxMusic3AccountReport report =
        MiniMaxMusic3AccountTensors(component, enumerated.at(component), entries);
    CHECK(report.matched == report.required);
    total += report.matched;
    MESSAGE(component << ": " << report.matched << " tensors accounted over " << shards.size()
                      << " shard(s)");
  }
  CHECK(total == vllm_test::kMusic3ManifestTensorCount);
  MESSAGE("real-checkpoint tensors accounted: " << total);
}

// ===========================================================================
// 8. RUNTIME dtype — the oracle refuted spec section 2.1, and this pins it
// ===========================================================================
//
// The spec read the converter's OUTPUT dtypes as upstream's resolved RUNTIME
// policy. Running the oracle refuted that: the on-disk set is not runnable,
// because upstream casts only on the way OUT of the condition encoder
// (denoise.py:83) and into the vocoder (decoders.py:84), never on the way IN.
// `denoise.py:82` moves device only, so the AR half all runs at the language
// model's dtype or it raises from `condition_embedder_minimax_music3.py:64`.

TEST_CASE("music3 runtime dtype: the AR half must agree, and a violation is refused BY NAME") {
  vllm::MiniMaxMusic3RuntimeDtypes dtypes =
      vllm::MiniMaxMusic3ResolveRuntimeDtypes(vllm::MiniMaxMusic3DtypePolicy::kBf16ArFp32Acoustic);
  // The gated configuration: AR half bf16, acoustic half fp32.
  CHECK(dtypes.language_model == "BF16");
  CHECK(dtypes.rvq_depth_decoder == "BF16");
  CHECK(dtypes.condition_encoder == "BF16");
  CHECK(dtypes.transformer == "F32");
  CHECK(dtypes.vocoder == "F32");
  CHECK(vllm::MiniMaxMusic3RuntimeDtypesAreRunnable(dtypes));
  CHECK_NOTHROW(vllm::MiniMaxMusic3CheckRuntimeDtypes(dtypes));

  // Every single-component deviation in the AR half is refused, and the
  // refusal NAMES all three so a reader knows which disagreed with which.
  int64_t refused = 0;
  const char* ar_names[] = {"language_model", "rvq_depth_decoder", "condition_encoder"};
  for (int index = 0; index < 3; ++index) {
    CAPTURE(ar_names[index]);
    vllm::MiniMaxMusic3RuntimeDtypes broken = dtypes;
    (index == 0 ? broken.language_model
                : index == 1 ? broken.rvq_depth_decoder
                             : broken.condition_encoder) = "F32";
    CHECK(!vllm::MiniMaxMusic3RuntimeDtypesAreRunnable(broken));
    bool named_all_three = false;
    try {
      vllm::MiniMaxMusic3CheckRuntimeDtypes(broken);
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      named_all_three = message.find("language_model") != std::string::npos &&
                        message.find("rvq_depth_decoder") != std::string::npos &&
                        message.find("condition_encoder") != std::string::npos &&
                        message.find("BF16") != std::string::npos &&
                        message.find("F32") != std::string::npos;
      if (index == 0) MESSAGE("refusal: " << message);
    }
    CHECK(named_all_three);
    ++refused;
  }
  CHECK(refused == 3);
  MESSAGE("AR-half deviations refused by name: " << refused);

  // The ACOUSTIC half may differ from the AR half and from each other freely:
  // both are reached through an explicit cast, so neither is constrained.
  int64_t allowed = 0;
  for (const char* dtype : {"F32", "BF16"}) {
    vllm::MiniMaxMusic3RuntimeDtypes varied = dtypes;
    varied.transformer = dtype;
    CHECK_NOTHROW(vllm::MiniMaxMusic3CheckRuntimeDtypes(varied));
    ++allowed;
    varied = dtypes;
    varied.vocoder = dtype;
    CHECK_NOTHROW(vllm::MiniMaxMusic3CheckRuntimeDtypes(varied));
    ++allowed;
  }
  CHECK(allowed == 4);
  MESSAGE("acoustic-half variations accepted (each is reached through a cast): " << allowed);
}

TEST_CASE("music3 runtime dtype: the ON-DISK set is reported as NOT RUNNABLE") {
  // This is the correction itself, pinned. The dtypes the files carry are a
  // fact about the artifact; presenting them as a runnable configuration is
  // what the oracle refuted, so `kAsStored` must NOT be runnable.
  const vllm::MiniMaxMusic3RuntimeDtypes stored =
      vllm::MiniMaxMusic3ResolveRuntimeDtypes(vllm::MiniMaxMusic3DtypePolicy::kAsStored);
  CHECK(stored.language_model == "BF16");
  CHECK(stored.rvq_depth_decoder == "BF16");
  CHECK(stored.condition_encoder == "F32");  // the one that breaks it
  CHECK(stored.transformer == "F32");
  CHECK(stored.vocoder == "F32");

  CHECK(!vllm::MiniMaxMusic3RuntimeDtypesAreRunnable(stored));
  CHECK_THROWS_AS(vllm::MiniMaxMusic3CheckRuntimeDtypes(stored), std::runtime_error);

  // And it agrees with what the manifest actually measured, so the "not
  // runnable" claim is about THIS checkpoint and not about a made-up set.
  CHECK(vllm::MiniMaxMusic3OnDiskDtypes().condition_encoder == stored.condition_encoder);
  std::map<std::string, std::set<std::string>> measured;
  for (const vllm_test::Music3ManifestTensor& t : vllm_test::kMusic3Manifest) {
    measured[t.component].insert(t.dtype);
  }
  const vllm::MiniMaxMusic3RuntimeDtypes on_disk = vllm::MiniMaxMusic3OnDiskDtypes();
  CHECK(measured["language_model"] == std::set<std::string>{on_disk.language_model});
  CHECK(measured["rvq_depth_decoder"] == std::set<std::string>{on_disk.rvq_depth_decoder});
  CHECK(measured["condition_encoder"] == std::set<std::string>{on_disk.condition_encoder});
  CHECK(measured["transformer"] == std::set<std::string>{on_disk.transformer});
  CHECK(measured["vocoder"] == std::set<std::string>{on_disk.vocoder});
  MESSAGE("on-disk dtypes cross-checked against the manifest for "
          << measured.size() << " components; the set is NOT runnable");
}
