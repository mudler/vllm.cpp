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

#include "hf_snapshot.h"
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

// `<snapshot>/<file>`, where the snapshot is resolved by the SINGLE pinned
// resolver `parity::Nemotron35LightningSnapshot()` (tests/parity/hf_snapshot.h).
//
// LOW-3 (#517). This arm used to read `CHECKPOINT_ROOT` itself and join the
// staging directory name by hand, which meant two env vars —
// `VT_NEMOTRON35_SNAPSHOT` there, `CHECKPOINT_ROOT` here — reached one
// checkpoint and neither refused a revision the goldens were not captured
// against. Both spellings now go through that resolver.
//
// GATE-SNAPSHOT-CONTENT-PIN (#569). That resolver gates the staged `local_dir`
// on the per-file `.cache/huggingface/download/<file>.metadata` sidecars, whose
// first line is the revision the bytes were downloaded at, so a re-download
// landing a different revision under the identical path SKIPS here rather than
// being silently substituted. It used to gate on the presence of
// `.cache/huggingface/trees/<revision>.json`, which records only that the
// revision was fetched here once and is never deleted when a later one lands.
// The resolver now hands back the reason it refused, and the banner below prints
// it verbatim -- a skip that does not say WHICH file named WHICH revision gets
// re-run instead of investigated.
//
// HOW TO MAKE THIS ARM RUN. `CHECKPOINT_ROOT` is a `.env` key, and `.env` is not
// exported into a login shell by anything: `.env.example:8` documents the loader
// verbatim as `set -a; . ./.env; set +a`, and `.agents/environment.md:16` points
// at that file. (An earlier version of this comment credited the loader line to
// `.agents/environment.md` as well; it is not there — LOW-3.) There is no
// CTest-side mechanism that reads `.env`, so a plain `ctest` from a shell that
// has not sourced it SKIPS this arm, which is correct behavior and not a pass —
// the banner below names the exact export.
std::string CheckpointFile(const char* filename) {
  std::string why;
  const std::string snapshot = parity::Nemotron35LightningSnapshot(&why);
  if (snapshot.empty()) {
    SkipGate(
        std::string("no pinned Nemotron-3.5-Lightning snapshot, so ") +
        filename + " cannot be located.\n*** REFUSED BECAUSE: " + why +
        "\n"
        "*** To RUN this arm, load the repo env first — `.env.example:8`\n"
        "***   documents exactly this:\n"
        "***       set -a; . ./.env; set +a\n"
        "***   then re-run ctest. Equivalently, export it for one run:\n"
        "***       CHECKPOINT_ROOT=<shared checkpoint dir> ctest -R "
        "test_modelopt_mixed_precision_checkpoint\n"
        "***   The arm needs only\n"
        "***   $CHECKPOINT_ROOT/" +
        parity::kNemotron35LightningLocalDirName +
        "/{config,hf_quant_config}.json — no weights, no GPU.\n"
        "***   A staged directory that IS present skips too unless EVERY file\n"
        "***   in it records revision " +
        parity::kNemotron35LightningNvfP4Revision +
        "\n"
        "***   on line 1 of its .cache/huggingface/download/<file>.metadata —\n"
        "***   the revision the committed goldens were captured against.");
  }
  const std::filesystem::path p = std::filesystem::path(snapshot) / filename;
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
  //
  // `const std::string`, NOT `const char*`, and that is load-bearing: doctest
  // 2.5.2 has no stringifier for a `const char*` VARIABLE, so `CAPTURE(p)` on
  // one prints `logged: p := 1` and a failure here would name none of the five
  // prefixes (LOW-2). The two loops above already spell it `std::string`.
  for (const std::string p : {"backbone.layers.5.mixer.q_proj",
                              "backbone.layers.42.mixer.o_proj",
                              "backbone.layers.0.mixer.conv1d",
                              "backbone.layers.1.mixer.gate",
                              "backbone.embeddings"}) {
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
  // `const std::string` for the same reason as the loop above: a `const char*`
  // capture prints `1` and this loop compares FIVE prefixes (LOW-2).
  for (const std::string p : {"backbone.layers.0.mixer.in_proj",
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
