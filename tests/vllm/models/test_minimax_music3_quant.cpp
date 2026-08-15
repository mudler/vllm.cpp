// MiniMax-Music3 phase W7 — the QUANTIZED ARMS.
//
// Row MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation,
// .agents/specs/minimax-music3.md phase W7, issue #672.
//
// AGENTS.md: "A model port covers the quantized arms, not just bf16. GGUF
// k-quants in particular are a standing requirement... An arm that is not
// implemented is refused with a message naming the missing piece and recorded
// as owed -- never left to be discovered later."
//
// This file gates the REFUSALS. There is no quantized MiniMax-Music3 checkpoint
// staged on this box, so nothing here loads a quantized weight or compares a
// quantized value: what it proves is that every quantized arm a user can
// actually obtain today is DIAGNOSED BY FORMAT and refused with a message
// naming the arm, the evidence in the artifact, the missing piece and the
// phase that owes it -- instead of the generic "missing transformer,
// condition_encoder, ..." a GGUF directory gets today, or the shape/dtype
// mismatch on an arbitrary tensor a quantized safetensors component gets.
//
// The formats enumerated here are not hypothetical. Each is a checkpoint that
// exists on HuggingFace for THIS model as of 2026-08-14 (spec section 9).
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/minimax_music3_loader.h"
#include "vllm/model_executor/models/minimax_music3_quant.h"

using vllm::MiniMaxMusic3ManifestEntry;
using vllm::MiniMaxMusic3QuantFinding;
using vllm::MiniMaxMusic3QuantFormat;

namespace {

// A scratch directory that removes itself, so a failing CHECK cannot leave a
// tree behind for the next run to trip over.
class ScratchDir {
 public:
  explicit ScratchDir(const std::string& tag) {
    // A random suffix rather than the pid: the scratchpad is shared across
    // sessions here, and two concurrent runs colliding on one directory is a
    // red bought from the environment rather than from the code.
    static std::random_device entropy;
    static const uint64_t run = (static_cast<uint64_t>(entropy()) << 32) ^ entropy();
    static int64_t counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("music3_quant_" + tag + "_" + std::to_string(run) + "_" +
             std::to_string(counter++));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }
  ~ScratchDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  const std::filesystem::path& path() const { return path_; }
  std::string str() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

void WriteFile(const std::filesystem::path& path, const std::string& body) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  out << body;
}

// The four bytes every GGUF file starts with, so a fixture is a GGUF by its
// CONTENT and not only by its name (gguf_reader.cpp reads the same magic).
const std::string kGgufMagic = "GGUF";

// What a refusal must always carry, whatever the format: the model, the fact
// that the arm is not implemented, the phase that owes it, and the issue.
struct RefusalCheck {
  bool named_model = false;
  bool named_format = false;
  bool named_evidence = false;
  bool named_owed = false;
  bool named_issue = false;
  bool named_supported_arm = false;
  std::string message;

  bool All() const {
    return named_model && named_format && named_evidence && named_owed && named_issue &&
           named_supported_arm;
  }
};

RefusalCheck Inspect(const std::string& message, const std::string& format,
                     const std::string& evidence) {
  RefusalCheck out;
  out.message = message;
  out.named_model = message.find("minimax_music3") != std::string::npos;
  out.named_format = message.find(format) != std::string::npos;
  out.named_evidence = message.find(evidence) != std::string::npos;
  // "owed" is the word AGENTS.md uses, and W7 is the phase that owes it.
  out.named_owed = message.find("owed") != std::string::npos &&
                   message.find("W7") != std::string::npos;
  out.named_issue = message.find("#672") != std::string::npos;
  // The user has to be told what WOULD work, or the refusal is a dead end.
  out.named_supported_arm = message.find("bf16") != std::string::npos &&
                            message.find("diffusers") != std::string::npos;
  return out;
}

}  // namespace

// ===========================================================================
// 1. Tree-level detection: the artifact a user actually downloads
// ===========================================================================

TEST_CASE("music3 quant: a GGUF directory is diagnosed as GGUF, not as missing components") {
  // `audio-cpp/MiniMax-Music3-GGUF` ships one GGUF per component. Pointed at
  // this loader today, the tree has no `transformer/` directory and no
  // `modular_model_index.json`, so it falls into the generic
  // "is not a diffusers-arm MiniMax-Music3 checkpoint" refusal -- which names
  // seven directories the user does not have and never says the word GGUF.
  ScratchDir root("gguf_tree");
  WriteFile(root.path() / "language_model_q4_k.gguf", kGgufMagic);
  WriteFile(root.path() / "transformer_q4_k.gguf", kGgufMagic);
  WriteFile(root.path() / "rvq_depth_decoder_q4_k.gguf", kGgufMagic);
  WriteFile(root.path() / "condition_encoder.gguf", kGgufMagic);
  WriteFile(root.path() / "vocoder.gguf", kGgufMagic);

  const MiniMaxMusic3QuantFinding finding = vllm::MiniMaxMusic3DetectQuantTree(root.str());
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kGguf);
  // A detector that cannot say how many things it looked at has not reported.
  CHECK(finding.examined >= 5);
  CHECK(finding.matched == 5);
  MESSAGE("gguf tree: examined " << finding.examined << " entries, " << finding.matched
                                 << " carried the marker, evidence " << finding.evidence);

  RefusalCheck check;
  try {
    vllm::MiniMaxMusic3ResolveCheckpoint(root.str());
    FAIL("a GGUF tree must not resolve");
  } catch (const std::runtime_error& error) {
    check = Inspect(error.what(), "GGUF", "transformer_q4_k.gguf");
    MESSAGE("refusal: " << check.message);
  }
  CHECK(check.named_model);
  CHECK(check.named_format);
  CHECK(check.named_evidence);
  CHECK(check.named_owed);
  CHECK(check.named_issue);
  CHECK(check.named_supported_arm);
  // It must NOT tell the user their GGUF checkpoint is missing seven diffusers
  // component directories. That message is true and useless.
  CHECK(check.message.find("is not a diffusers-arm") == std::string::npos);
}

TEST_CASE("music3 quant: a SINGLE ComfyUI DiT .gguf beside nothing else is still diagnosed") {
  // `Abiray/MiniMax-Music3-GGUF` and `realrebelai/MiniMax-Music-3_GGUFs` ship
  // the 2.46B DiT ALONE, ComfyUI-style. One file is enough evidence.
  ScratchDir root("gguf_single");
  WriteFile(root.path() / "MiniMax-Music3-DiT-Q4_K_M.gguf", kGgufMagic);

  const MiniMaxMusic3QuantFinding finding = vllm::MiniMaxMusic3DetectQuantTree(root.str());
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kGguf);
  CHECK(finding.matched == 1);

  bool refused = false;
  try {
    vllm::MiniMaxMusic3ResolveCheckpoint(root.str());
  } catch (const std::runtime_error& error) {
    refused = Inspect(error.what(), "GGUF", "MiniMax-Music3-DiT-Q4_K_M.gguf").All();
  }
  CHECK(refused);
}

TEST_CASE("music3 quant: a GGUF nested one directory down is found") {
  // `scragnog/MiniMax-Music3-GGUF` and the ComfyUI repacks nest under
  // `diffusion_models/` and `text_encoders/`. A detector that only stats the
  // root would miss every one of them.
  ScratchDir root("gguf_nested");
  WriteFile(root.path() / "diffusion_models" / "minimax_music3_dit_Q4_K_M.gguf", kGgufMagic);
  WriteFile(root.path() / "text_encoders" / "mm3-lm-Q4_K_M.gguf", kGgufMagic);

  const MiniMaxMusic3QuantFinding finding = vllm::MiniMaxMusic3DetectQuantTree(root.str());
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kGguf);
  CHECK(finding.matched == 2);
  MESSAGE("nested gguf evidence: " << finding.evidence);
}

TEST_CASE("music3 quant: the SUPPORTED bf16/fp32 tree is NOT diagnosed as quantized") {
  // The refusal must be silent on the arm that works. A detector that fires on
  // the gated checkpoint would refuse every real load, and a test suite that
  // never checked the negative would not notice.
  ScratchDir root("clean_tree");
  for (const char* component : {"transformer", "condition_encoder", "rvq_depth_decoder",
                                "vocoder", "language_model", "scheduler", "tokenizer"}) {
    std::filesystem::create_directories(root.path() / component);
  }
  WriteFile(root.path() / "modular_model_index.json", "{}");
  WriteFile(root.path() / "transformer" / "diffusion_pytorch_model.safetensors", "not real");

  const MiniMaxMusic3QuantFinding finding = vllm::MiniMaxMusic3DetectQuantTree(root.str());
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kNone);
  CHECK(finding.matched == 0);
  CHECK(finding.examined > 0);
  MESSAGE("clean tree: examined " << finding.examined << ", matched " << finding.matched);
}

TEST_CASE("music3 quant: the NATIVE arm keeps its OWN refusal, not the quant one") {
  // Spec section 2's native-arm refusal is a different diagnosis and must not
  // be swallowed: `.pth` is not a quantization format.
  ScratchDir root("native");
  std::filesystem::create_directories(root.path() / "qwen_7B" / "qwen_7B");
  WriteFile(root.path() / "flowmatching_vae.pth", "x");
  WriteFile(root.path() / "dav.pth", "x");

  CHECK(vllm::MiniMaxMusic3DetectQuantTree(root.str()).format ==
        MiniMaxMusic3QuantFormat::kNone);
  bool named_native = false;
  try {
    vllm::MiniMaxMusic3ResolveCheckpoint(root.str());
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    named_native = message.find("NATIVE arm") != std::string::npos &&
                   message.find("GGUF") == std::string::npos;
  }
  CHECK(named_native);
}

// ===========================================================================
// 2. Manifest-level detection: a diffusers-SHAPED tree whose weights are not
// ===========================================================================

TEST_CASE("music3 quant: an NVFP4 component is named NVFP4, not a shape mismatch") {
  // The NVFP4 triple this project already consumes elsewhere
  // (minimax_h3_nvfp4.cpp:7-11): U8 packed [out, in/2], F8_E4M3 group scale
  // [out, in/16], F32 global. Accounted against the bf16 contract today it
  // produces "expected shape [2048, 2048], file has [2048, 1024]" on whichever
  // tensor sorts first -- true, and no help at all.
  const std::vector<MiniMaxMusic3ManifestEntry> entries{
      {"proj.weight", "U8", {2048, 2048}},
      {"proj.weight_scale", "F8_E4M3", {2048, 256}},
      {"proj.weight_scale_2", "F32", {1}},
      {"layer_scale", "F32", {8}},
  };
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("condition_encoder", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kNvfp4);
  CHECK(finding.component == "condition_encoder");
  CHECK(finding.examined == 4);
  CHECK(finding.matched == 1);

  RefusalCheck check;
  try {
    vllm::MiniMaxMusic3CheckQuantArm(finding);
    FAIL("an NVFP4 manifest must be refused");
  } catch (const std::runtime_error& error) {
    check = Inspect(error.what(), "NVFP4", "proj.weight_scale_2");
    MESSAGE("refusal: " << check.message);
  }
  CHECK(check.All());
  CHECK(check.message.find("condition_encoder") != std::string::npos);
}

TEST_CASE("music3 quant: `matched` counts EVERY marked tensor, not just the first") {
  // Added after a mutation stayed GREEN. Setting `out.matched = 1` unconditionally
  // passed every other case in this file, because each of them happens to carry
  // exactly one marker -- so the count was reported but never discriminated, and
  // a refusal saying "1 of 400" on a fully quantized checkpoint would have read
  // as an isolated stray tensor. The number is part of the evidence, so it gets
  // a case that can only pass if it is real.
  std::vector<MiniMaxMusic3ManifestEntry> entries;
  for (int layer = 0; layer < 36; ++layer) {
    const std::string base = "layers." + std::to_string(layer) + ".self_attn.q_proj.";
    entries.push_back({base + "weight", "U8", {4096, 2048}});
    entries.push_back({base + "weight_scale", "F8_E4M3", {4096, 256}});
    entries.push_back({base + "weight_scale_2", "F32", {1}});
  }
  entries.push_back({"norm.weight", "BF16", {4096}});

  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("language_model", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kNvfp4);
  CHECK(finding.examined == 109);
  CHECK(finding.matched == 36);
  // Evidence is capped so a 400-tensor refusal stays readable, and the cap is
  // ANNOUNCED rather than silently truncating.
  CHECK(finding.evidence.find("(+30 more)") != std::string::npos);
  const std::string message = vllm::MiniMaxMusic3QuantRefusal(finding);
  CHECK(message.find("36 of 109") != std::string::npos);
  MESSAGE("matched " << finding.matched << " of " << finding.examined
                     << " examined; evidence " << finding.evidence);
}

TEST_CASE("music3 quant: an MXFP4 compressed-tensors component is named MXFP4") {
  const std::vector<MiniMaxMusic3ManifestEntry> entries{
      {"blocks.0.ff.net.0.weight_packed", "U8", {8192, 1024}},
      {"blocks.0.ff.net.0.weight_scale", "U8", {8192, 64}},
  };
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("transformer", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kMxfp4);
  CHECK(finding.matched == 1);
  bool refused = false;
  try {
    vllm::MiniMaxMusic3CheckQuantArm(finding);
  } catch (const std::runtime_error& error) {
    refused = Inspect(error.what(), "MXFP4", "weight_packed").All();
  }
  CHECK(refused);
}

TEST_CASE("music3 quant: an AWQ/GPTQ component is named by its qweight triple") {
  const std::vector<MiniMaxMusic3ManifestEntry> entries{
      {"layers.0.self_attn.q_proj.qweight", "I32", {4096, 512}},
      {"layers.0.self_attn.q_proj.qzeros", "I32", {32, 512}},
      {"layers.0.self_attn.q_proj.scales", "F16", {32, 4096}},
  };
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("language_model", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kAwqGptq);
  bool refused = false;
  try {
    vllm::MiniMaxMusic3CheckQuantArm(finding);
  } catch (const std::runtime_error& error) {
    refused = Inspect(error.what(), "AWQ", "qweight").All();
  }
  CHECK(refused);
}

TEST_CASE("music3 quant: an FP8 component is named by its DTYPE, which no name carries") {
  // fp8 is the case a name-based detector alone cannot see: the tensor is still
  // `<module>.weight` at its true [out, in] shape, and only the dtype moved.
  // Against the bf16 contract that surfaces as W1's DTYPE refusal, which says
  // "expected BF16, file has F8_E4M3" and never says fp8 or that an fp8 arm is
  // owed.
  const std::vector<MiniMaxMusic3ManifestEntry> entries{
      {"layers.0.mlp.gate_proj.weight", "F8_E4M3", {12288, 4096}},
      {"layers.0.input_layernorm.weight", "BF16", {4096}},
  };
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("language_model", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kFp8);
  CHECK(finding.matched == 1);
  bool refused = false;
  try {
    vllm::MiniMaxMusic3CheckQuantArm(finding);
  } catch (const std::runtime_error& error) {
    refused = Inspect(error.what(), "FP8", "layers.0.mlp.gate_proj.weight").All();
  }
  CHECK(refused);
}

TEST_CASE("music3 quant: an INT8 component is named INT8 (Comfy-Org's int8-convrot / w4a8)") {
  // `Comfy-Org/MiniMax-Music-3` ships `..._int8_convrot`, and
  // `NidAll/MiniMax-Music3-W4A8` / `dummy9996/...-w4a8-bf16-comfyui` ship w4a8.
  // Neither is fp8; a refusal that said "fp8" would be wrong.
  const std::vector<MiniMaxMusic3ManifestEntry> entries{
      {"blocks.0.attn.to_q.weight", "I8", {2048, 2048}},
      {"blocks.0.attn.to_q.weight_scale", "F32", {2048}},
  };
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("transformer", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kInt8);
  bool refused = false;
  try {
    vllm::MiniMaxMusic3CheckQuantArm(finding);
  } catch (const std::runtime_error& error) {
    refused = Inspect(error.what(), "INT8", "blocks.0.attn.to_q.weight").All();
  }
  CHECK(refused);
}

TEST_CASE("music3 quant: a bitsandbytes component is named by its absmax sidecar") {
  const std::vector<MiniMaxMusic3ManifestEntry> entries{
      {"layers.0.mlp.up_proj.weight", "U8", {25165824, 1}},
      {"layers.0.mlp.up_proj.weight.absmax", "U8", {393216, 1}},
      {"layers.0.mlp.up_proj.weight.quant_map", "F32", {16}},
  };
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("language_model", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kBitsAndBytes);
  bool refused = false;
  try {
    vllm::MiniMaxMusic3CheckQuantArm(finding);
  } catch (const std::runtime_error& error) {
    refused = Inspect(error.what(), "bitsandbytes", "absmax").All();
  }
  CHECK(refused);
}

TEST_CASE("music3 quant: a bare weight_scale names the CANDIDATES rather than picking one") {
  // A `weight_scale` with no `weight_scale_2`, no `weight_packed` and a
  // non-U8 parent could be NVFP4-without-the-global, a compressed-tensors
  // block scheme, or a per-channel int8 scale. The detector must NOT choose:
  // guessing here is exactly what ltx2_loader.h:232-268 records as the failure
  // that produces a finite, correctly shaped, WRONG result.
  const std::vector<MiniMaxMusic3ManifestEntry> entries{
      {"blocks.0.ff.net.2.weight", "BF16", {2048, 8192}},
      {"blocks.0.ff.net.2.weight_scale", "F32", {2048, 512}},
  };
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("transformer", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kUnknownScheme);
  bool refused = false;
  std::string message;
  try {
    vllm::MiniMaxMusic3CheckQuantArm(finding);
  } catch (const std::runtime_error& error) {
    message = error.what();
    refused = Inspect(message, "weight_scale", "blocks.0.ff.net.2.weight_scale").All();
  }
  CHECK(refused);
  // Named as candidates, plural, and not resolved to one.
  CHECK(message.find("NVFP4") != std::string::npos);
  CHECK(message.find("MXFP4") != std::string::npos);
  MESSAGE("refusal: " << message);
}

TEST_CASE("music3 quant: the SUPPORTED dtypes are NOT diagnosed as quantized") {
  // The real checkpoint's own dtypes, per spec section 2.1: BF16 on the AR
  // half, F32 on the acoustic half. Neither may fire.
  const std::vector<MiniMaxMusic3ManifestEntry> entries{
      {"layer_scale", "F32", {8}},
      {"layer_weight_logits", "F32", {8}},
      {"proj.weight", "F32", {2048, 4096, 1}},
      {"proj.bias", "F32", {2048}},
      {"layers.0.self_attn.q_proj.weight", "BF16", {4096, 4096}},
      {"conv_in.weight_g", "F32", {1536, 1, 1}},
      {"conv_in.weight_v", "F32", {1536, 1024, 7}},
  };
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("condition_encoder", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kNone);
  CHECK(finding.matched == 0);
  CHECK(finding.examined == 7);
  // And it does not throw.
  vllm::MiniMaxMusic3CheckQuantArm(finding);
  MESSAGE("supported dtypes: examined " << finding.examined << ", matched " << finding.matched);
}

TEST_CASE("music3 quant: weight_g/weight_v do NOT read as a quantization sidecar") {
  // The vocoder's 30 legacy weight-norm pairs end in `_g` and `_v` and are a
  // PARAMETERIZATION, not a quantization. A detector keying on "a sidecar
  // beside a weight" would refuse the shipped vocoder outright.
  const std::vector<MiniMaxMusic3ManifestEntry> entries{
      {"decoder.model.1.block.1.block.1.weight_g", "F32", {1536, 1, 1}},
      {"decoder.model.1.block.1.block.1.weight_v", "F32", {1536, 1536, 7}},
      {"decoder.model.1.block.0.alpha", "F32", {1, 1536, 1}},
  };
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantManifest("vocoder", entries);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kNone);
  CHECK(finding.matched == 0);
}

// ===========================================================================
// 3. Config-level detection: the declaration, before a byte is read
// ===========================================================================

TEST_CASE("music3 quant: a config.json quantization_config is refused by its quant_method") {
  struct Case {
    const char* method;
    const char* expect;
  };
  const std::vector<Case> cases{
      {"compressed-tensors", "compressed-tensors"},
      {"fp8", "fp8"},
      {"awq", "awq"},
      {"gptq", "gptq"},
      {"bitsandbytes_4bit", "bitsandbytes_4bit"},
      {"modelopt", "modelopt"},
  };
  int64_t refused = 0;
  for (const Case& c : cases) {
    CAPTURE(c.method);
    const std::string body = std::string(R"({"_class_name": "MiniMaxMusic3Transformer1DModel",
      "quantization_config": {"quant_method": ")") +
                             c.method + R"("}})";
    const MiniMaxMusic3QuantFinding finding =
        vllm::MiniMaxMusic3DetectQuantConfig("transformer", body);
    CHECK(finding.format != MiniMaxMusic3QuantFormat::kNone);
    CHECK(finding.matched == 1);
    bool here = false;
    try {
      vllm::MiniMaxMusic3CheckQuantArm(finding);
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      here = message.find(c.expect) != std::string::npos &&
             message.find("quantization_config") != std::string::npos &&
             message.find("owed") != std::string::npos &&
             message.find("#672") != std::string::npos;
      if (!here) MESSAGE("unhelpful refusal for " << c.method << ": " << message);
    }
    CHECK(here);
    refused += here ? 1 : 0;
  }
  CHECK(refused == static_cast<int64_t>(cases.size()));
  MESSAGE("quant_method values refused by name: " << refused << " of " << cases.size());
}

TEST_CASE("music3 quant: an MLX config's `quantization` block is refused by name") {
  // `ddalcu/...-MLX-Serve-8bit`, `vanch007/...-MLX-8bit`, `elishabjm/...-MLX`
  // (4-bit) exist for this model. MLX writes `quantization: {group_size, bits}`
  // rather than `quantization_config.quant_method`, so a detector that only
  // read the latter would pass an MLX tree straight into the bf16 contract.
  const std::string body =
      R"({"_class_name": "MiniMaxMusic3Transformer1DModel",
          "quantization": {"group_size": 64, "bits": 4}})";
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantConfig("transformer", body);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kMlx);
  bool refused = false;
  std::string message;
  try {
    vllm::MiniMaxMusic3CheckQuantArm(finding);
  } catch (const std::runtime_error& error) {
    message = error.what();
    refused = Inspect(message, "MLX", "quantization").All();
  }
  CHECK(refused);
  MESSAGE("refusal: " << message);
}

TEST_CASE("music3 quant: the REAL configs carry no quantization declaration") {
  // The shipped transformer config, abridged to its real keys. A false
  // positive here would refuse the only arm that works.
  const std::string body = R"({
    "_class_name": "MiniMaxMusic3Transformer1DModel",
    "_diffusers_version": "0.40.0.dev0",
    "in_channels": 128, "condition_dim": 2048, "num_layers": 36,
    "num_attention_heads": 32, "attention_head_dim": 64, "ff_inner_dim": 8192,
    "rotary_dim": 32, "fourier_embedding_dim": 256})";
  const MiniMaxMusic3QuantFinding finding =
      vllm::MiniMaxMusic3DetectQuantConfig("transformer", body);
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kNone);
  CHECK(finding.matched == 0);
  vllm::MiniMaxMusic3CheckQuantArm(finding);
}

TEST_CASE("music3 quant: a null quantization_config is ABSENT, not a refusal") {
  // transformers writes `"quantization_config": null` on an unquantized save.
  // Tri-state: absent and null both mean "not quantized"
  // (.agents/porting-a-model.md section 1).
  for (const char* body : {R"({"quantization_config": null})",
                           R"({"quantization": null})"}) {
    CAPTURE(body);
    const MiniMaxMusic3QuantFinding finding =
        vllm::MiniMaxMusic3DetectQuantConfig("transformer", body);
    CHECK(finding.format == MiniMaxMusic3QuantFormat::kNone);
  }
}

// ===========================================================================
// 3b. The three HOOKS, exercised through the loader's own entry points
// ===========================================================================
//
// Sections 1-3 gate the DETECTORS. A detector nobody called would pass every
// one of them while a user still got the old message, so each of the three
// places the loader consults a detector is exercised here through the public
// entry point a caller actually uses.

TEST_CASE("music3 quant hook: MiniMaxMusic3AccountTensors refuses NVFP4 before shape") {
  // The hook that matters most. Accounted against the bf16 contract WITHOUT it,
  // this exact manifest refuses on `layer_scale` -- "has shape [8] but the
  // config implies [1]" -- which is a tensor that is not quantized, is not
  // wrong, and has nothing to do with the problem. It is simply the name that
  // sorts first.
  const vllm::MiniMaxMusic3ConditionEncoderConfig config;
  const std::vector<vllm::MiniMaxMusic3TensorSpec> required =
      vllm::EnumerateMiniMaxMusic3ConditionEncoderTensors(config);
  const std::vector<MiniMaxMusic3ManifestEntry> present{
      {"proj.weight", "U8", {2048, 2048, 1}},
      {"proj.weight_scale", "F8_E4M3", {2048, 256}},
      {"proj.weight_scale_2", "F32", {1}},
      {"proj.bias", "F32", {2048}},
      {"layer_scale", "F32", {8}},
      {"layer_weight_logits", "F32", {8}},
  };
  RefusalCheck check;
  try {
    vllm::MiniMaxMusic3AccountTensors("condition_encoder", required, present);
    FAIL("an NVFP4 manifest must not account clean");
  } catch (const std::runtime_error& error) {
    check = Inspect(error.what(), "NVFP4", "proj.weight_scale_2");
    MESSAGE("refusal: " << check.message);
  }
  CHECK(check.All());
  // And specifically NOT the misleading diagnosis it produced before W7.
  CHECK(check.message.find("layer_scale has shape") == std::string::npos);
}

TEST_CASE("music3 quant hook: MiniMaxMusic3AccountTensors refuses FP8 before the dtype message") {
  // W1's dtype refusal is true -- "expected BF16, file has F8_E4M3" -- and says
  // nothing about fp8 being a quantized ARM that is owed.
  const std::vector<vllm::MiniMaxMusic3TensorSpec> required{
      {"layers.0.mlp.gate_proj.weight", "BF16", {12288, 4096}}};
  const std::vector<MiniMaxMusic3ManifestEntry> present{
      {"layers.0.mlp.gate_proj.weight", "F8_E4M3", {12288, 4096}}};
  RefusalCheck check;
  try {
    vllm::MiniMaxMusic3AccountTensors("language_model", required, present);
    FAIL("an fp8 manifest must not account clean");
  } catch (const std::runtime_error& error) {
    check = Inspect(error.what(), "FP8", "layers.0.mlp.gate_proj.weight");
  }
  CHECK(check.All());
  CHECK(check.message.find("a narrowing or widening is a MEASURED change") ==
        std::string::npos);
}

TEST_CASE("music3 quant hook: MiniMaxMusic3AccountTensors still accounts the REAL contract") {
  // The hook must be invisible to the arm that works. This is the negative that
  // stops the fix from refusing every genuine load.
  const vllm::MiniMaxMusic3ConditionEncoderConfig config;
  const std::vector<vllm::MiniMaxMusic3TensorSpec> required =
      vllm::EnumerateMiniMaxMusic3ConditionEncoderTensors(config);
  std::vector<MiniMaxMusic3ManifestEntry> present;
  for (const vllm::MiniMaxMusic3TensorSpec& spec : required) {
    present.push_back({spec.name, spec.dtype, spec.shape});
  }
  const vllm::MiniMaxMusic3AccountReport report =
      vllm::MiniMaxMusic3AccountTensors("condition_encoder", required, present);
  CHECK(report.matched == report.required);
  CHECK(report.present == report.required);
  CHECK(report.required == 4);
  MESSAGE("clean account: " << report.matched << " of " << report.required << " matched");
}

TEST_CASE("music3 quant hook: a quantized COMPONENT CONFIG is caught at resolve time") {
  // A diffusers-SHAPED tree whose transformer declares compressed-tensors. The
  // tree detector's config sweep is what catches this before a single tensor is
  // accounted, and before MiniMaxMusic3LoadConfig parses a geometry key.
  ScratchDir root("quantized_config_tree");
  for (const char* component : {"transformer", "condition_encoder", "rvq_depth_decoder",
                                "vocoder", "language_model", "scheduler", "tokenizer"}) {
    std::filesystem::create_directories(root.path() / component);
  }
  WriteFile(root.path() / "modular_model_index.json", "{}");
  WriteFile(root.path() / "transformer" / "config.json",
            R"({"_class_name": "MiniMaxMusic3Transformer1DModel",
                "quantization_config": {"quant_method": "compressed-tensors"}})");

  const MiniMaxMusic3QuantFinding finding = vllm::MiniMaxMusic3DetectQuantTree(root.str());
  CHECK(finding.format == MiniMaxMusic3QuantFormat::kCompressedTensors);
  CHECK(finding.evidence.find("transformer/config.json") != std::string::npos);

  RefusalCheck check;
  try {
    vllm::MiniMaxMusic3ResolveCheckpoint(root.str());
    FAIL("a compressed-tensors tree must not resolve");
  } catch (const std::runtime_error& error) {
    check = Inspect(error.what(), "compressed-tensors", "quant_method");
    MESSAGE("refusal: " << check.message);
  }
  CHECK(check.All());
}

TEST_CASE("music3 quant hook: MiniMaxMusic3LoadConfig refuses a quantized component config") {
  // The third hook, reached directly. A caller that assembles
  // `MiniMaxMusic3Paths` itself bypasses the tree sweep, so the config parse
  // carries its own check rather than trusting an earlier one to have run.
  ScratchDir root("loadconfig");
  std::filesystem::create_directories(root.path() / "transformer");
  WriteFile(root.path() / "transformer" / "config.json",
            R"({"_class_name": "MiniMaxMusic3Transformer1DModel",
                "quantization_config": {"quant_method": "awq"},
                "in_channels": 128, "condition_dim": 2048, "num_layers": 36,
                "num_attention_heads": 32, "attention_head_dim": 64,
                "ff_inner_dim": 8192, "rotary_dim": 32,
                "fourier_embedding_dim": 256})");
  vllm::MiniMaxMusic3Paths paths;
  paths.root = root.str();
  paths.transformer_dir = (root.path() / "transformer").string();

  RefusalCheck check;
  try {
    vllm::MiniMaxMusic3LoadConfig(paths);
    FAIL("an AWQ transformer config must not parse");
  } catch (const std::runtime_error& error) {
    check = Inspect(error.what(), "AWQ", "awq");
    MESSAGE("refusal: " << check.message);
  }
  CHECK(check.All());
  // The component is named from the config's own parent directory.
  CHECK(check.message.find("transformer") != std::string::npos);
}

// ===========================================================================
// 3c. The GGUF ARM's lineage refusals, on SYNTHETIC files
// ===========================================================================
//
// Added after mutation QM4 -- neutering the loader's lineage guard to
// `if (false)` -- STAYED GREEN. The reason was a genuine coverage hole rather
// than a bad mutation: the only GGUF the suite had was the real audio-cpp one,
// which IS the accepted lineage, so removing the check that rejects the other
// two changed nothing observable. Spec section 9.2 records that there are three
// mutually incompatible published lineages and that `general.architecture`
// CANNOT separate them, so a loader that stopped checking would bind an `mm3` or
// ComfyUI file by position -- finite, correctly shaped, wrong. These fixtures
// are the smallest GGUFs that prove the guard is load-bearing, and they need no
// checkpoint and no network.

namespace {

// Build a minimal VALID GGUF v3 in memory: header, string metadata, one small
// F32 tensor. Only the metadata is under test, but the file has to PARSE before
// the lineage check can reject it -- a malformed file would be refused for the
// wrong reason and would prove nothing about the guard.
std::string BuildTinyGguf(const std::vector<std::pair<std::string, std::string>>& metadata) {
  std::string out;
  auto put_u32 = [&](uint32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); };
  auto put_u64 = [&](uint64_t v) { out.append(reinterpret_cast<const char*>(&v), 8); };
  auto put_str = [&](const std::string& s) {
    put_u64(s.size());
    out += s;
  };
  out += "GGUF";
  put_u32(3);                // version
  put_u64(1);                // tensor_count
  put_u64(metadata.size());  // metadata_kv_count
  for (const auto& kv : metadata) {
    put_str(kv.first);
    put_u32(8);  // GGUF value type 8 == string
    put_str(kv.second);
  }
  put_str("norm.weight");
  put_u32(1);   // n_dims
  put_u64(32);  // dims[0]
  put_u32(0);   // ggml type 0 == F32
  put_u64(0);   // offset into the data section
  while (out.size() % 32 != 0) out += '\0';  // default GGUF alignment
  out.append(32 * sizeof(float), '\0');
  return out;
}

}  // namespace

TEST_CASE("music3 gguf arm: a NON-audio-cpp lineage is refused, naming what is missing") {
  struct Case {
    const char* label;
    std::vector<std::pair<std::string, std::string>> metadata;
  };
  const std::vector<Case> cases{
      // The scragnog lineage: self-describing, but renamed with fused QKV.
      {"mm3", {{"general.architecture", "mm3"}, {"mm3.model", "MiniMax-Music3"}}},
      // The ComfyUI lineage, whose architecture key collides with real Wan GGUFs.
      {"wan", {{"general.architecture", "wan"}}},
      // A GGUF of the language-model half: a different component entirely.
      {"qwen3", {{"general.architecture", "qwen3"}}},
      // The right ARCHITECTURE string with no family key -- the case that shows
      // why `general.architecture` is not the discriminator.
      {"bare_audiocpp", {{"general.architecture", "audiocpp"}}},
  };
  int64_t refused = 0;
  for (const Case& c : cases) {
    CAPTURE(c.label);
    ScratchDir dir(std::string("lineage_") + c.label);
    const std::filesystem::path path = dir.path() / "candidate.gguf";
    WriteFile(path, BuildTinyGguf(c.metadata));

    const vllm::GgufFile file = vllm::GgufFile::Open(path.string());
    CHECK_FALSE(vllm::MiniMaxMusic3GgufIsNativeLineage(file));

    const vllm::MiniMaxMusic3RvqDepthDecoderConfig config;
    vllm::MiniMaxMusic3GgufLoadReport report;
    bool here = false;
    try {
      vllm::MiniMaxMusic3LoadRvqDepthDecoderFromGguf(config, file, &report);
      FAIL("a non-audio-cpp GGUF must not load");
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      here = message.find("audio-cpp lineage") != std::string::npos &&
             message.find("audiocpp.model_spec.family") != std::string::npos &&
             message.find("#672") != std::string::npos &&
             // It must name the OTHER lineages and why each is unreadable, or
             // the person holding one has to re-derive it.
             message.find("mm3") != std::string::npos &&
             message.find("rename table") != std::string::npos &&
             message.find("ComfyUI") != std::string::npos;
      if (!here) MESSAGE("unhelpful refusal for " << c.label << ": " << message);
    }
    CHECK(here);
    refused += here ? 1 : 0;
  }
  CHECK(refused == static_cast<int64_t>(cases.size()));
  MESSAGE("non-native lineages refused by name: " << refused << " of " << cases.size());
}

TEST_CASE("music3 gguf arm: the right family with a RENAMED layout is still refused") {
  // `tensor_name_format` is a SEPARATE guard from the family, and it has to be:
  // a future audio-cpp export that renamed its tensors would carry the right
  // family and bind to the wrong weights by position.
  ScratchDir dir("renamed");
  const std::filesystem::path path = dir.path() / "renamed.gguf";
  WriteFile(path, BuildTinyGguf({{"general.architecture", "audiocpp"},
                                 {"audiocpp.model_spec.family", "minimax_music3"},
                                 {"audiocpp.tensor_name_format", "flattened"}}));
  const vllm::GgufFile file = vllm::GgufFile::Open(path.string());
  // The FAMILY is accepted...
  CHECK(vllm::MiniMaxMusic3GgufIsNativeLineage(file));
  // ...and the NAME FORMAT is what refuses.
  const vllm::MiniMaxMusic3RvqDepthDecoderConfig config;
  vllm::MiniMaxMusic3GgufLoadReport report;
  bool refused = false;
  std::string message;
  try {
    vllm::MiniMaxMusic3LoadRvqDepthDecoderFromGguf(config, file, &report);
  } catch (const std::runtime_error& error) {
    message = error.what();
    refused = message.find("tensor_name_format") != std::string::npos &&
              message.find("flattened") != std::string::npos &&
              message.find("native") != std::string::npos &&
              message.find("#672") != std::string::npos;
  }
  CHECK(refused);
  MESSAGE("refusal: " << message);
}

TEST_CASE("music3 gguf arm: a native GGUF MISSING tensors is refused, not zero-filled") {
  // The accounting contract, on a file that passes BOTH lineage guards. A
  // one-tensor GGUF owes 47.
  ScratchDir dir("short");
  const std::filesystem::path path = dir.path() / "short.gguf";
  WriteFile(path, BuildTinyGguf({{"general.architecture", "audiocpp"},
                                 {"audiocpp.model_spec.family", "minimax_music3"},
                                 {"audiocpp.tensor_name_format", "native"}}));
  const vllm::GgufFile file = vllm::GgufFile::Open(path.string());
  CHECK(vllm::MiniMaxMusic3GgufIsNativeLineage(file));
  const vllm::MiniMaxMusic3RvqDepthDecoderConfig config;
  vllm::MiniMaxMusic3GgufLoadReport report;
  bool refused = false;
  std::string message;
  try {
    vllm::MiniMaxMusic3LoadRvqDepthDecoderFromGguf(config, file, &report);
  } catch (const std::runtime_error& error) {
    message = error.what();
    refused = message.find("is MISSING") != std::string::npos &&
              message.find("rvq_depth_decoder gguf") != std::string::npos &&
              message.find("zero-filled") != std::string::npos;
  }
  CHECK(refused);
  MESSAGE("refusal: " << message);
}

// ===========================================================================
// 4. Every format has a name, and the enumeration is closed
// ===========================================================================

TEST_CASE("music3 quant: every format has a distinct, non-empty name") {
  const std::vector<MiniMaxMusic3QuantFormat> all{
      MiniMaxMusic3QuantFormat::kNone,          MiniMaxMusic3QuantFormat::kGguf,
      MiniMaxMusic3QuantFormat::kNvfp4,         MiniMaxMusic3QuantFormat::kMxfp4,
      MiniMaxMusic3QuantFormat::kFp8,           MiniMaxMusic3QuantFormat::kInt8,
      MiniMaxMusic3QuantFormat::kAwqGptq,       MiniMaxMusic3QuantFormat::kBitsAndBytes,
      MiniMaxMusic3QuantFormat::kMlx,           MiniMaxMusic3QuantFormat::kCompressedTensors,
      MiniMaxMusic3QuantFormat::kUnknownScheme,
  };
  std::set<std::string> names;
  for (const MiniMaxMusic3QuantFormat format : all) {
    const std::string name = vllm::MiniMaxMusic3QuantFormatName(format);
    CHECK(!name.empty());
    names.insert(name);
  }
  CHECK(names.size() == all.size());
  MESSAGE("distinct format names: " << names.size() << " of " << all.size());
}

TEST_CASE("music3 quant: kNone never throws and every OTHER format always does") {
  const std::vector<MiniMaxMusic3QuantFormat> quantized{
      MiniMaxMusic3QuantFormat::kGguf,          MiniMaxMusic3QuantFormat::kNvfp4,
      MiniMaxMusic3QuantFormat::kMxfp4,         MiniMaxMusic3QuantFormat::kFp8,
      MiniMaxMusic3QuantFormat::kInt8,          MiniMaxMusic3QuantFormat::kAwqGptq,
      MiniMaxMusic3QuantFormat::kBitsAndBytes,  MiniMaxMusic3QuantFormat::kMlx,
      MiniMaxMusic3QuantFormat::kCompressedTensors,
      MiniMaxMusic3QuantFormat::kUnknownScheme,
  };
  int64_t threw = 0;
  for (const MiniMaxMusic3QuantFormat format : quantized) {
    MiniMaxMusic3QuantFinding finding;
    finding.format = format;
    finding.component = "transformer";
    finding.evidence = "some.tensor";
    finding.matched = 1;
    finding.examined = 1;
    CAPTURE(vllm::MiniMaxMusic3QuantFormatName(format));
    bool here = false;
    try {
      vllm::MiniMaxMusic3CheckQuantArm(finding);
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      here = message.find(vllm::MiniMaxMusic3QuantFormatName(format)) != std::string::npos &&
             message.find("owed") != std::string::npos &&
             message.find("W7") != std::string::npos &&
             message.find("#672") != std::string::npos;
    }
    CHECK(here);
    threw += here ? 1 : 0;
  }
  CHECK(threw == static_cast<int64_t>(quantized.size()));
  MESSAGE("formats that refuse by name: " << threw << " of " << quantized.size());

  MiniMaxMusic3QuantFinding none;
  none.format = MiniMaxMusic3QuantFormat::kNone;
  vllm::MiniMaxMusic3CheckQuantArm(none);  // must not throw
}
