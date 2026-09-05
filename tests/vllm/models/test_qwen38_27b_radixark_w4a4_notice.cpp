// QUANT-QWEN38-27B-NVFP4-ARM W7 (#2760): a ModelOpt checkpoint that DECLARES
// `quant_algo: "NVFP4"` -- 4-bit weights AND 4-bit ACTIVATIONS -- and that this
// build then executes W4A16, weight-only, HAS TO SAY SO.
//
// THE DEFECT THIS FILE GATES. `RadixArk/Qwen3.8-27B-NVFP4` @
// `554ebba9b5f1b79dc11246341960360e6ef05ef4` -- the target
// `pangoleen/qwen3.8-27b-dgx-spark-dflash2` serves, and the artifact
// `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` was derived from -- declares
// `{"quant_algo": "NVFP4", "group_size": 16}` on all 193 of its NVFP4 modules,
// declares `config_groups.group_1.input_activations` 4-bit static, and ships
// `<module>.input_scale` on every one of them, `lm_head` included. W5's
// `layers::modelopt::Refusal` accepts that and is right to: its question is
// whether the shipped SPELLING matches the declaration, and it does. The arm is
// then chosen by `VT_MODELOPT_W4A4`, which defaults to 0, so the load takes the
// weight-only W4A16 dispatcher and PRINTS NOTHING.
//
// WHY NO EXISTING GATE COULD CATCH IT. The weight bytes swept per decode step
// are identical either way -- the same E2M1 nibbles and the same fp8-e4m3
// group-16 scales -- so no throughput axis moves and no roofline changes. What
// differs is the ACTIVATION path and the numerics, and W4A16 of a W4A4
// checkpoint is numerically plausible, so the tokens still match. That is the
// class `AGENTS.md` §"Inherit vLLM defaults" names: the gate passes and the path
// is wrong. Only a diagnostic can close it, so the assertions below are on the
// diagnostic. Every HERMETIC case enters through `vllm::LoadQwen3_5Dense` with
// STDERR CAPTURED -- never on the notice function alone, because a test that
// calls the function directly stays green when the production call site is
// deleted, and that is the mutation this file is built to fail. The env-gated
// case 7 is the one exception, deliberately: it reads the real artifact's
// HEADERS, so there are no weight bytes for a loader to read, and what it adds
// is the COUNTS over the real 2194 names rather than the reachability the other
// seven already hold.
//
// HERMETIC. Cases 1-6 and 8 build a one-layer synthetic checkpoint carrying this
// artifact's REAL module names and a `quantization_config` built from its REAL
// one, so CI needs no NAS file and no network. The live arm at the bottom is
// env-gated on VLLM_CPP_QWEN38_27B_RADIXARK_DIR and SKIPS LOUDLY when unset; it
// reads the staged `config.json` and the three shards' safetensors HEADERS
// only, never a weight byte.
#include <doctest/doctest.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/quantization/modelopt_mixed_precision.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"

namespace mo = vllm::layers::modelopt;

namespace {

// ── stderr, redirected to a file for the duration of a load ─────────────────
//
// The notice is a PRINT, so the only way to assert it reached an operator is to
// read what the process wrote. Same shape as
// `tests/vllm/multimodal/test_render_phase_log.cpp`.
class StderrCapture {
 public:
  StderrCapture() {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_radixark_w4a4_stderr_" + std::to_string(counter++) + ".txt"))
                .string();
    std::fflush(stderr);
    saved_ = ::dup(STDERR_FILENO);
    sink_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (saved_ >= 0 && sink_ >= 0) ok_ = ::dup2(sink_, STDERR_FILENO) >= 0;
  }
  ~StderrCapture() {
    Restore();
    ::unlink(path_.c_str());
  }
  StderrCapture(const StderrCapture&) = delete;
  StderrCapture& operator=(const StderrCapture&) = delete;

  bool ok() const { return ok_; }

  // Restores the real stderr and returns everything written while it was ours.
  std::string Take() {
    Restore();
    std::ifstream in(path_, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
  }

 private:
  void Restore() {
    if (saved_ < 0) return;
    std::fflush(stderr);
    ::dup2(saved_, STDERR_FILENO);
    ::close(saved_);
    if (sink_ >= 0) ::close(sink_);
    saved_ = -1;
    sink_ = -1;
  }
  std::string path_;
  int saved_ = -1;
  int sink_ = -1;
  bool ok_ = false;
};

// One environment variable, restored on the way out. `ModelOptW4A4OptIn` reads
// `VT_MODELOPT_W4A4` on every call rather than latching it, so a case really can
// move it; leaking it into the next case would make case order a hidden input.
class ScopedEnv {
 public:
  ScopedEnv(std::string key, const char* value) : key_(std::move(key)) {
    const char* old = std::getenv(key_.c_str());
    if (old != nullptr) {
      had_ = true;
      old_ = old;
    }
    if (value == nullptr) {
      ::unsetenv(key_.c_str());
    } else {
      ::setenv(key_.c_str(), value, 1);
    }
  }
  ~ScopedEnv() {
    if (had_) {
      ::setenv(key_.c_str(), old_.c_str(), 1);
    } else {
      ::unsetenv(key_.c_str());
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string key_;
  std::string old_;
  bool had_ = false;
};

// ── a one-layer checkpoint in this artifact's own spelling ──────────────────

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

// A whole safetensors blob from `specs`, filled with finite positive values so a
// scale the loader reads is never zero and never a NaN.
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
             ("vllm_radixark_w4a4_" + std::to_string(counter++) + ".safetensors"))
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
// in_dim is a multiple of 16.
constexpr int64_t kH = 32;
constexpr int64_t kFfn = 64;
constexpr int64_t kHead = 4;
constexpr int64_t kQ = 32;
constexpr int64_t kKv = 16;

const char* kPrefix = "model.language_model.";

// The ModelOpt NVFP4 spelling this artifact ships: a U8 `weight`, an F8 group
// scale, the F32 scalar `weight_scale_2`, and -- the field that makes this
// artifact different from `r0b0tlab`'s -- the F32 scalar `input_scale`, the
// ACTIVATION divisor a W4A4 arm would read.
void AppendModeloptNvfp4(std::vector<Spec>& out, const std::string& proj, int64_t n,
                         int64_t k, bool with_input_scale) {
  out.push_back({proj + ".weight", {n, k / 2}, "U8"});
  out.push_back({proj + ".weight_scale", {n, k / 16}, "F8_E4M3"});
  out.push_back({proj + ".weight_scale_2", {}, "F32"});
  if (with_input_scale) out.push_back({proj + ".input_scale", {}, "F32"});
}

// The compressed-tensors NVFP4 spelling. `Refusal` accepts it under a ModelOpt
// declaration (its `kNvfp4`/`kW4A16Nvfp4` case takes `ops.CtNvfp4()`), and
// `LoadNvfp4AnyNaming` routes on `weight_packed` and reaches `LoadCtNvfp4Raw`,
// which reads `input_global_scale` UNCONDITIONALLY. Such a module really is
// W4A4, so a declaration of `NVFP4` over it is not a divergence.
void AppendCtNvfp4(std::vector<Spec>& out, const std::string& proj, int64_t n,
                   int64_t k) {
  out.push_back({proj + ".weight_packed", {n, k / 2}, "U8"});
  out.push_back({proj + ".weight_scale", {n, k / 16}, "F8_E4M3"});
  out.push_back({proj + ".weight_global_scale", {}, "F32"});
  out.push_back({proj + ".input_global_scale", {}, "F32"});
}

// The per-tensor STATIC FP8 spelling the 208 FP8 modules ship.
void AppendStaticFp8(std::vector<Spec>& out, const std::string& proj, int64_t n,
                     int64_t k) {
  out.push_back({proj + ".weight", {n, k}, "F8_E4M3"});
  out.push_back({proj + ".weight_scale", {}, "F32"});
  out.push_back({proj + ".input_scale", {}, "F32"});
}

enum class MlpNvfp4Spelling { kModelopt, kCompressedTensors };

// A one-layer full-attention backbone in this artifact's spelling: a static-FP8
// attention tower and an NVFP4 MLP that does or does not ship its divisor.
std::vector<Spec> OneLayerSpecs(
    bool nvfp4_input_scale,
    MlpNvfp4Spelling spelling = MlpNvfp4Spelling::kModelopt) {
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
  for (const auto& pr : projs)
    AppendStaticFp8(s, sa + pr.first, pr.second.first, pr.second.second);
  const std::pair<const char*, std::pair<int64_t, int64_t>> mlp_projs[] = {
      {"gate_proj", {kFfn, kH}},
      {"up_proj", {kFfn, kH}},
      {"down_proj", {kH, kFfn}},
  };
  for (const auto& pr : mlp_projs) {
    if (spelling == MlpNvfp4Spelling::kCompressedTensors) {
      AppendCtNvfp4(s, mlp + pr.first, pr.second.first, pr.second.second);
    } else {
      AppendModeloptNvfp4(s, mlp + pr.first, pr.second.first, pr.second.second,
                          nvfp4_input_scale);
    }
  }
  return s;
}

// The shipped `quantization_config`, narrowed to the one layer the synthetic
// checkpoint has and carrying the field under test: `nvfp4_algo` is `"NVFP4"`
// for `RadixArk` and `"W4A16_NVFP4"` for `r0b0tlab`, which is the ONLY
// difference between the two declarations that decides an activation arm.
nlohmann::json OneLayerQuantConfig(const char* nvfp4_algo) {
  nlohmann::json q = nlohmann::json::object();
  q["quant_method"] = "modelopt";
  q["quant_algo"] = "MIXED_PRECISION";
  q["ignore"] = nlohmann::json::array();
  nlohmann::json layers = nlohmann::json::object();
  const std::string l = std::string(kPrefix) + "layers.0.";
  for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"})
    layers[l + "self_attn." + proj] = {{"quant_algo", "FP8"}};
  for (const char* proj : {"gate_proj", "up_proj", "down_proj"})
    layers[l + "mlp." + proj] = {{"quant_algo", nvfp4_algo}, {"group_size", 16}};
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

// Everything one case needs: LOAD through the production entry point, with
// stderr captured, and hand back what the load wrote. A load that throws is a
// different result from a load that printed nothing, so the throw is reported
// rather than swallowed.
struct LoadResult {
  std::string stderr_text;
  std::string threw;
  bool loaded = false;
};

LoadResult LoadCapturingStderr(const std::vector<Spec>& specs,
                               const vllm::HfConfig& config) {
  const TempFile file(BuildSafetensors(specs));
  LoadResult r;
  StderrCapture cap;
  REQUIRE(cap.ok());
  try {
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(file.path()));
    const vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5Dense(shards, config);
    (void)w;
    r.loaded = true;
  } catch (const std::exception& e) {
    r.threw = e.what();
  }
  r.stderr_text = cap.Take();
  return r;
}

bool Mentions(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

// ── (1) THE DEFECT: declared NVFP4, executed W4A16, and it now SPEAKS ────────
//
// This is the case that was RED before W7: the load succeeded and wrote nothing
// at all to stderr.
TEST_CASE("RadixArk W4A4: a checkpoint declaring NVFP4 that we run W4A16 says so at load") {
  const ScopedEnv off("VT_MODELOPT_W4A4", nullptr);
  const vllm::HfConfig config = OneLayerConfig(OneLayerQuantConfig("NVFP4"));
  const LoadResult r = LoadCapturingStderr(OneLayerSpecs(/*nvfp4_input_scale=*/true),
                                           config);
  INFO("threw: " << r.threw);
  REQUIRE(r.threw.empty());
  REQUIRE(r.loaded);

  // The checkpoint LOADS. A notice is not a refusal, and a refusal here would
  // refuse a published artifact for a divergence this tree chose.
  INFO("stderr was: " << r.stderr_text);
  REQUIRE_FALSE(r.stderr_text.empty());

  // Both algorithms, in the operator's own vocabulary. "NVFP4" alone says
  // nothing, because the arm we took is also spelled with those five characters.
  CHECK(Mentions(r.stderr_text, "NVFP4"));
  CHECK(Mentions(r.stderr_text, "W4A16"));
  CHECK(Mentions(r.stderr_text, "4-bit weights AND 4-bit ACTIVATIONS"));
  // The count is the three MLP projections this one-layer checkpoint declares.
  CHECK(Mentions(r.stderr_text, "on 3 module(s)"));
  CHECK(Mentions(r.stderr_text, "3 of them ship the activation divisor"));
  // The operand and the knob, so a reader can act rather than only worry.
  CHECK(Mentions(r.stderr_text, "input_scale"));
  CHECK(Mentions(r.stderr_text, "VT_MODELOPT_W4A4 is unset"));
  // WHY IT MATTERS, in the message rather than in a spec the operator does not
  // have open: no throughput axis moves and a token gate cannot see it.
  CHECK(Mentions(r.stderr_text, "no throughput axis moves"));
  CHECK(Mentions(r.stderr_text, "a token gate cannot see"));
  // The module names, so the count reconciles against the file.
  CHECK(Mentions(r.stderr_text, "model.language_model.layers.0.mlp.gate_proj"));
  CHECK(Mentions(r.stderr_text, "model.language_model.layers.0.mlp.down_proj"));
  // An output head is NOT carved out of the count. An earlier draft of this
  // message said W4A16 was correct for a head whatever the declaration says,
  // and the pin falsifies it: `ModelOptMixedPrecisionConfig` dispatches a
  // `ParallelLMHead` on `quant_algo` like any other Linear, with no head
  // branch, so a head declared NVFP4 resolves upstream to the fp4-ACTIVATION
  // method too.
  CHECK(Mentions(r.stderr_text, "no output-head carve-out"));
  // The owner. Visible debt with no owner is a complaint.
  CHECK(Mentions(r.stderr_text, "QUANT-QWEN38-27B-NVFP4-ARM"));
  CHECK(Mentions(r.stderr_text, "#2760"));
}

// ── (2) THE NEGATIVE CONTROL that makes case 1 mean something ───────────────
//
// `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` declares `W4A16_NVFP4` and ships NO
// `input_scale` on an NVFP4 module. It gets exactly the arm it asked for, so it
// must stay SILENT. Without this case, case 1 is satisfied by a build that
// prints the notice for every ModelOpt checkpoint.
TEST_CASE("RadixArk W4A4: a checkpoint declaring W4A16_NVFP4 and given W4A16 stays silent") {
  const ScopedEnv off("VT_MODELOPT_W4A4", nullptr);
  const vllm::HfConfig config = OneLayerConfig(OneLayerQuantConfig("W4A16_NVFP4"));
  const LoadResult r = LoadCapturingStderr(OneLayerSpecs(/*nvfp4_input_scale=*/false),
                                           config);
  INFO("threw: " << r.threw);
  REQUIRE(r.threw.empty());
  REQUIRE(r.loaded);
  INFO("stderr was: " << r.stderr_text);
  CHECK_FALSE(Mentions(r.stderr_text, "#2760"));
  CHECK_FALSE(Mentions(r.stderr_text, "4-bit weights AND 4-bit ACTIVATIONS"));
}

// ── (3) and the W4A16 declaration stays silent even WITH the divisor ────────
//
// `nvidia/Qwen3.6-27B-NVFP4` @ `0893e160` is this shape: it declares all 193 of
// its NVFP4 modules `W4A16_NVFP4` and ships an `input_scale` on every one. It is
// the gate model every recorded 27B NVFP4 ratio was taken on, and it is NOT
// divergent -- weight-only is what it asked for -- so a notice here would fire on
// the one checkpoint this tree measures most.
TEST_CASE("RadixArk W4A4: W4A16_NVFP4 with an input_scale shipped is still not a divergence") {
  const ScopedEnv off("VT_MODELOPT_W4A4", nullptr);
  const vllm::HfConfig config = OneLayerConfig(OneLayerQuantConfig("W4A16_NVFP4"));
  const LoadResult r = LoadCapturingStderr(OneLayerSpecs(/*nvfp4_input_scale=*/true),
                                           config);
  INFO("threw: " << r.threw);
  REQUIRE(r.threw.empty());
  INFO("stderr was: " << r.stderr_text);
  CHECK_FALSE(Mentions(r.stderr_text, "#2760"));
}

// ── (4) the knob really does silence it, because then the arm really moves ──
TEST_CASE("RadixArk W4A4: VT_MODELOPT_W4A4=1 with the divisor shipped takes the declared arm and says nothing") {
  const ScopedEnv on("VT_MODELOPT_W4A4", "1");
  const vllm::HfConfig config = OneLayerConfig(OneLayerQuantConfig("NVFP4"));
  const LoadResult r = LoadCapturingStderr(OneLayerSpecs(/*nvfp4_input_scale=*/true),
                                           config);
  INFO("threw: " << r.threw);
  REQUIRE(r.threw.empty());
  REQUIRE(r.loaded);
  INFO("stderr was: " << r.stderr_text);
  CHECK_FALSE(Mentions(r.stderr_text, "#2760"));
}

// ── (5) the knob is NOT the predicate: no divisor means no W4A4 arm ─────────
//
// A build that reported on `VT_MODELOPT_W4A4` alone would call this silenced.
// The arm is still weight-only, because `LoadNvfp4AnyNaming` needs BOTH the knob
// and a shipped `input_scale` before it sets `alpha`, so the divergence is still
// live and still has to speak.
TEST_CASE("RadixArk W4A4: declared NVFP4 with NO divisor shipped still diverges under VT_MODELOPT_W4A4=1") {
  const ScopedEnv on("VT_MODELOPT_W4A4", "1");
  const vllm::HfConfig config = OneLayerConfig(OneLayerQuantConfig("NVFP4"));
  const LoadResult r = LoadCapturingStderr(OneLayerSpecs(/*nvfp4_input_scale=*/false),
                                           config);
  INFO("threw: " << r.threw);
  REQUIRE(r.threw.empty());
  INFO("stderr was: " << r.stderr_text);
  REQUIRE(Mentions(r.stderr_text, "#2760"));
  CHECK(Mentions(r.stderr_text, "on 3 module(s)"));
  CHECK(Mentions(r.stderr_text, "0 of them ship the activation divisor"));
  CHECK(Mentions(r.stderr_text, "could not take that arm even with VT_MODELOPT_W4A4=1"));
}

// ── (6) a checkpoint that declares no ModelOpt config reads NOTHING ─────────
//
// The blast radius. Every non-ModelOpt checkpoint answers "" without parsing a
// thing, so nothing that loaded before W7 prints anything new.
TEST_CASE("RadixArk W4A4: a checkpoint with no ModelOpt quantization_config prints nothing") {
  const ScopedEnv off("VT_MODELOPT_W4A4", nullptr);
  vllm::HfConfig config = OneLayerConfig(OneLayerQuantConfig("NVFP4"));
  // compressed-tensors sits in the same field; `IsMixedPrecision` is how a
  // loader tells them apart, mirroring
  // `ModelOptMixedPrecisionConfig.override_quantization_method`, which claims a
  // config only for `quant_algo == "MIXED_PRECISION"` (modelopt.py:2225-2231 @
  // the parity pin `e126687a9a828d513c01a07cd69f025f27d63280`).
  config.raw["quantization_config"]["quant_method"] = "compressed-tensors";
  const LoadResult r = LoadCapturingStderr(OneLayerSpecs(/*nvfp4_input_scale=*/true),
                                           config);
  INFO("stderr was: " << r.stderr_text);
  CHECK_FALSE(Mentions(r.stderr_text, "#2760"));
}

// ── (7) THE REAL ARTIFACT, env-gated ────────────────────────────────────────
//
// Headers and `config.json` only, never a weight byte. Set
// VLLM_CPP_QWEN38_27B_RADIXARK_DIR to the staged
// `RadixArk/Qwen3.8-27B-NVFP4` @ `554ebba9` directory.
TEST_CASE("RadixArk W4A4: the REAL config and the REAL 2194 tensor names name 193 modules") {
  const char* dir = std::getenv("VLLM_CPP_QWEN38_27B_RADIXARK_DIR");
  if (dir == nullptr || dir[0] == '\0') {
    MESSAGE(
        "SKIPPED: set VLLM_CPP_QWEN38_27B_RADIXARK_DIR to the staged "
        "RadixArk/Qwen3.8-27B-NVFP4 @ 554ebba9 directory to run this case");
    CHECK(true);
    return;
  }
  const std::filesystem::path root(dir);
  std::ifstream cfg(root / "config.json");
  REQUIRE(cfg.good());
  const nlohmann::json doc = nlohmann::json::parse(cfg);
  REQUIRE(doc.contains("quantization_config"));
  const nlohmann::json& quant = doc.at("quantization_config");

  std::vector<std::string> names;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    const std::string fn = entry.path().filename().string();
    if (fn.rfind("model-", 0) != 0 || entry.path().extension() != ".safetensors")
      continue;
    const vllm::SafetensorsFile shard = vllm::SafetensorsFile::Open(entry.path().string());
    for (const std::string& n : shard.Names()) names.push_back(n);
  }
  CHECK(names.size() == 2194);

  const std::string notice =
      mo::ActivationArmNoticeForQuantizationConfig(quant, names,
                                                   /*w4a4_opt_in=*/false);
  INFO("notice was: " << notice);
  REQUIRE_FALSE(notice.empty());
  CHECK(notice.find("on 193 module(s)") != std::string::npos);
  CHECK(notice.find("193 of them ship the activation divisor") != std::string::npos);
  CHECK(notice.find("#2760") != std::string::npos);

  // The other half of the same fact: with the knob on, every one of those 193
  // modules really would take the fp4-activation arm, so nothing is reported.
  CHECK(mo::ActivationArmNoticeForQuantizationConfig(quant, names,
                                                     /*w4a4_opt_in=*/true)
            .empty());
}

// ── (8) the compressed-tensors spelling really IS W4A4, so it stays silent ───
//
// The exemption that keeps this notice honest about the arm rather than about
// the declaration. `LoadCtNvfp4Raw` reads `<proj>.input_global_scale`
// UNCONDITIONALLY and folds it into `alpha`, so a module shipping that spelling
// takes the fp4-ACTIVATION GEMM whatever `VT_MODELOPT_W4A4` says -- it got the
// W4A4 arm its declaration asked for, and reporting it would be false.
TEST_CASE("RadixArk W4A4: a compressed-tensors NVFP4 module under an NVFP4 declaration is NOT divergent") {
  const ScopedEnv off("VT_MODELOPT_W4A4", nullptr);
  const vllm::HfConfig config = OneLayerConfig(OneLayerQuantConfig("NVFP4"));
  const LoadResult r = LoadCapturingStderr(
      OneLayerSpecs(/*nvfp4_input_scale=*/false, MlpNvfp4Spelling::kCompressedTensors),
      config);
  INFO("threw: " << r.threw);
  REQUIRE(r.threw.empty());
  REQUIRE(r.loaded);
  INFO("stderr was: " << r.stderr_text);
  CHECK_FALSE(Mentions(r.stderr_text, "#2760"));
}
