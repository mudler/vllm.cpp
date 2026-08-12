// EXHAUSTIVE arm of the ModelOpt MIXED_PRECISION resolver gate: the REAL
// 1.3 MB config.json of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4,
// all 5981 quantized_layers entries and all 72 ignore entries.
//
// Weights are NOT needed — only config.json is read — but the file lives with
// the 20 GiB checkpoint on shared storage, so this arm is opt-in and SKIPS
// LOUDLY (CTest SKIP_RETURN_CODE 77) when the checkpoint is not staged. It is
// a separate binary from the always-on curated gate precisely so that skipping
// here can never mask that one.
//
// Row MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm W1, issue #517.
// Upstream anchors: see test_modelopt_mixed_precision.cpp.
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/quantization/modelopt_mixed_precision.h"

using vllm::layers::modelopt::MixedPrecisionConfig;
using vllm::layers::modelopt::ModuleQuant;
using vllm::layers::modelopt::QuantAlgo;
using vllm::layers::modelopt::Resolution;

namespace {

// A gate that CANNOT RUN must never report success. Returning early out of a
// TEST_CASE prints "0 passed | 0 failed" + "Status: SUCCESS!" and exits 0,
// which is indistinguishable in a log from a real pass (issue #463). Exiting
// 77 makes CTest report **Skipped** and stops any `&&` chain.
[[noreturn]] void SkipGate(const std::string& why) {
  std::fprintf(stderr,
               "\n*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "*** test_modelopt_mixed_precision_checkpoint: %s\n\n",
               why.c_str());
  std::fflush(stderr);
  std::exit(77);
}

// $CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4/<file> — the path
// .agents/specs/nemotron-h-model.md §0 records the checkpoint is staged at.
//
// HOW TO MAKE THIS ARM RUN. `CHECKPOINT_ROOT` is a `.env` key, and `.env` is
// not exported into a login shell by anything: `.env.example` and
// `.agents/environment.md` document the loader as `set -a; . ./.env; set +a`,
// and that IS the repo's convention — there is no CTest-side mechanism that
// reads `.env`, and the only other checkpoint-gated helper here
// (`tests/parity/hf_snapshot.h`) resolves a Hugging Face cache under `$HOME`
// rather than a NAS-staged directory, so it does not apply. A plain `ctest`
// from a shell that has not sourced `.env` therefore SKIPS this arm, which is
// correct behavior and not a pass — the banner below names the exact export.
std::string CheckpointFile(const char* filename) {
  const char* root = std::getenv("CHECKPOINT_ROOT");
  if (root == nullptr || *root == '\0') {
    SkipGate(
        std::string("CHECKPOINT_ROOT is unset, so ") + filename +
        " cannot be located.\n"
        "*** To RUN this arm, load the repo env first — `.env.example` and\n"
        "***   .agents/environment.md document exactly this:\n"
        "***       set -a; . ./.env; set +a\n"
        "***   then re-run ctest. Equivalently, export it for one run:\n"
        "***       CHECKPOINT_ROOT=<shared checkpoint dir> ctest -R "
        "test_modelopt_mixed_precision_checkpoint\n"
        "***   The arm needs only\n"
        "***   $CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4/{config,"
        "hf_quant_config}.json — no weights, no GPU");
  }
  const std::filesystem::path p =
      std::filesystem::path(root) / "nemotron-3.5-lightning-30b-nvfp4" / filename;
  std::error_code ec;
  if (!std::filesystem::exists(p, ec)) {
    SkipGate("not staged: " + p.string());
  }
  return p.string();
}

nlohmann::ordered_json LoadJson(const char* filename) {
  const std::string path = CheckpointFile(filename);
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  return nlohmann::ordered_json::parse(f);
}

// config.json -> quantization_config: the FLAT (compressed-tensors style)
// shape, `{"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", ...}`.
nlohmann::ordered_json LoadQuantizationConfig() {
  nlohmann::ordered_json doc = LoadJson("config.json");
  REQUIRE_MESSAGE(doc.contains("quantization_config"),
                  "no quantization_config in config.json");
  return doc.at("quantization_config");
}

}  // namespace

TEST_CASE("modelopt-mixed[ckpt]: every real entry resolves, histogram exact") {
  const nlohmann::ordered_json qc = LoadQuantizationConfig();

  // Assert the FIXTURE is the checkpoint this gate claims to cover before
  // asserting anything about it (oracle-identity discipline: a repo silently
  // re-quantized under the same name has cost this project a campaign).
  REQUIRE(MixedPrecisionConfig::IsMixedPrecision(qc));
  const MixedPrecisionConfig c = MixedPrecisionConfig::Parse(qc);
  REQUIRE(c.num_quantized_layers() == 5981);
  REQUIRE(c.exclude_modules().size() == 72);
  CHECK(c.group_size() == 16);
  CHECK(c.kv_cache_quant_algo() == "FP8");

  std::map<std::string, int> histogram;
  const auto& ql = qc.at("quantized_layers");
  for (auto it = ql.begin(); it != ql.end(); ++it) {
    const std::string name = it.key();
    const ModuleQuant m = c.Resolve(name);
    CAPTURE(name);
    // A listed entry must resolve to a quantized algorithm. "Unquantized" here
    // would mean the loader silently dequantizes a quantized tensor — correct
    // numerics, wrong bytes, invisible to a token gate.
    REQUIRE(m.Quantized());
    CHECK(m.how == Resolution::kDirect);
    histogram[QuantAlgoName(m.algo)] += 1;
  }

  CHECK(histogram.size() == 2);
  CHECK(histogram["W4A16_NVFP4"] == 5935);
  CHECK(histogram["FP8"] == 46);

  // Every one of the 72 ignore entries is excluded, and therefore unquantized.
  for (const auto& entry : qc.at("ignore")) {
    const std::string name = entry.get<std::string>();
    CAPTURE(name);
    CHECK(c.IsLayerExcluded(name));
    CHECK(c.Resolve(name).algo == QuantAlgo::kUnquantized);
  }
}

TEST_CASE("modelopt-mixed[ckpt]: the module prefixes the loader will actually ask for") {
  const MixedPrecisionConfig c = MixedPrecisionConfig::Parse(LoadQuantizationConfig());

  // Routed + shared experts resolve through the PREFIX strategy: the layer
  // module is "...mixer.experts", the map lists "...mixer.experts.<i>.<proj>".
  for (int layer : {1, 3, 6, 51}) {
    const std::string p =
        "backbone.layers." + std::to_string(layer) + ".mixer.experts";
    CAPTURE(p);
    const ModuleQuant m = c.Resolve(p);
    CHECK(m.algo == QuantAlgo::kW4A16Nvfp4);
    CHECK(m.how == Resolution::kPrefix);
    CHECK(m.group_size == 16);
  }

  // Mamba projections are FP8 — the polarity trap this row exists to catch:
  // the repo NAME says NVFP4 and 5935 of 5981 entries are, but these 46 are not.
  for (int layer : {0, 2, 50}) {
    const std::string p =
        "backbone.layers." + std::to_string(layer) + ".mixer.in_proj";
    CAPTURE(p);
    CHECK(c.Resolve(p).algo == QuantAlgo::kFp8);
    CHECK(c.Resolve(p).group_size == 0);
  }

  // Attention (layers 5/12/19/26/33/42), conv1d, gates and embeddings are bf16.
  for (const char* p : {"backbone.layers.5.mixer.q_proj",
                        "backbone.layers.42.mixer.o_proj",
                        "backbone.layers.0.mixer.conv1d",
                        "backbone.layers.1.mixer.gate", "backbone.embeddings"}) {
    CAPTURE(p);
    CHECK(c.Resolve(p).how == Resolution::kExcluded);
  }

  // lm_head is a BARE key in the map; the loader's prefix carries no "model.".
  CHECK(c.Resolve("lm_head").algo == QuantAlgo::kW4A16Nvfp4);
  CHECK(c.Resolve("model.lm_head").algo == QuantAlgo::kW4A16Nvfp4);

  // The whole MTP head is covered by the "mtp*" wildcard.
  CHECK(c.IsLayerExcluded("mtp.layers.0.mixer.up_proj"));
  CHECK(c.IsLayerExcluded("mtp.layers.0.eh_proj"));
}

// The checkpoint ships its quantization config TWICE: once flat inside
// config.json's `quantization_config`, and once in the standalone
// `hf_quant_config.json` that ModelOpt actually writes and that
// `get_config_filenames()` (modelopt.py:265-267) names. Only the first shape
// was gated, and the second is the one upstream's own file list points at.
//
// Its top-level key set is exactly {"producer", "quantization"} — there is NO
// `quant_method` anywhere in the file. `from_config` (modelopt.py:282-367)
// does not want one: it dispatches on the SHAPE and reads `quant_algo` out of
// the nested section. Borrowing the SELECTION hook's `quant_method`
// precondition (`_extract_modelopt_quant_algo`, :245-263, used only by
// `override_quantization_method`) made `Parse` REFUSE this file outright while
// a verbatim upstream transcription resolved all 5981 entries from it.
TEST_CASE("modelopt-mixed[ckpt]: the standalone hf_quant_config.json parses") {
  const nlohmann::ordered_json hq = LoadJson("hf_quant_config.json");

  // Assert the SHAPE this case exists for, so a re-quantized publish that
  // changed it fails here rather than quietly retargeting the case.
  REQUIRE(hq.is_object());
  REQUIRE(hq.contains("quantization"));
  REQUIRE_FALSE(hq.contains("quant_method"));
  REQUIRE_FALSE(hq.at("quantization").contains("quant_method"));
  CHECK(hq.at("quantization").at("quant_algo") == "MIXED_PRECISION");

  // DETECTION still refuses it, and that is upstream's behavior, not a bug:
  // `_extract_modelopt_quant_algo` returns None for a config naming no
  // quantizer. The override hook is for `config.json`'s `quantization_config`,
  // where several vendors share one field and the name is what tells them apart.
  CHECK_FALSE(MixedPrecisionConfig::IsMixedPrecision(hq));

  // PARSING must accept it — this is the file the loader will be handed.
  const MixedPrecisionConfig c = MixedPrecisionConfig::Parse(hq);
  CHECK(c.num_quantized_layers() == 5981);
  CHECK(c.exclude_modules().size() == 72);
  CHECK(c.group_size() == 16);
  // The nested shape spells kv cache as a plain algo STRING
  // (`kv_cache_quant_algo`), where the flat shape uses a `kv_cache_scheme`
  // dict. Both must land on the same answer.
  CHECK(c.kv_cache_quant_algo() == "FP8");

  // ABSOLUTE expectations first. Comparing the two shapes to each other proves
  // consistency, not correctness — both arms call the same resolver, so a
  // wrong answer agrees with itself perfectly.
  CHECK(c.Resolve("backbone.layers.0.mixer.in_proj").algo == QuantAlgo::kFp8);
  CHECK(c.Resolve("backbone.layers.0.mixer.in_proj").group_size == 0);
  CHECK(c.Resolve("backbone.layers.1.mixer.experts").algo ==
        QuantAlgo::kW4A16Nvfp4);
  CHECK(c.Resolve("backbone.layers.1.mixer.experts").how == Resolution::kPrefix);
  CHECK(c.Resolve("backbone.layers.1.mixer.experts").group_size == 16);
  CHECK(c.Resolve("backbone.layers.5.mixer.q_proj").how == Resolution::kExcluded);
  CHECK(c.Resolve("lm_head").algo == QuantAlgo::kW4A16Nvfp4);
  CHECK(c.IsLayerExcluded("mtp.layers.0.eh_proj"));

  // ...and only THEN that the flat shape agrees module for module.
  const MixedPrecisionConfig flat =
      MixedPrecisionConfig::Parse(LoadQuantizationConfig());
  for (const char* p : {"backbone.layers.0.mixer.in_proj",
                        "backbone.layers.1.mixer.experts",
                        "backbone.layers.5.mixer.q_proj", "lm_head",
                        "mtp.layers.0.eh_proj"}) {
    CAPTURE(p);
    const ModuleQuant a = c.Resolve(p);
    const ModuleQuant b = flat.Resolve(p);
    CHECK(a.algo == b.algo);
    CHECK(a.how == b.how);
    CHECK(a.group_size == b.group_size);
  }
}
