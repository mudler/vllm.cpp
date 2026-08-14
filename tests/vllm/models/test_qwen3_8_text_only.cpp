// Qwen3.5-family TEXT-ONLY arms: `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`.
//
// AHEAD-OF-PIN FORWARD PORT of upstream PR #50210 (`ad5d29db7`). Our parity pin
// is `555967922`, whose registry.py carries only the `ForConditionalGeneration`
// strings, so this file is deliberately anchored on a POST-PIN upstream head and
// says so; it does not advance the pin.
//
//   upstream vllm/model_executor/models/registry.py:202-203 @ `ad5d29db7`
//     "Qwen3_5ForCausalLM":    ("qwen3_5", "Qwen3_5ForCausalLM"),
//     "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),
//   upstream vllm/model_executor/models/qwen3_5.py:439-449 @ `ad5d29db7`
//     `Qwen3_5ForCausalLM` IS `Qwen3_5ForCausalLMBase` unchanged; the MoE arm is
//     that same base plus `set_moe_parameters()` — one backbone, two arms.
//   upstream vllm/model_executor/models/qwen3_5.py:296-300 @ `ad5d29db7`
//     WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
//     — `model.` is CANONICAL and the VL-prefixed spelling is its accepted alias.
//
// WHAT THIS FILE DOES NOT CLAIM. `Qwen/Qwen3.8-2.4T-A95B` cannot be executed on
// this hardware (2.4T bf16 ≈ 4.8 TB, the FP8 variant ≈ 2.4 TB, GB10 has 128 GB
// unified and no smaller Qwen3.8 sibling exists), so there is NO token gate for
// that checkpoint and none is implied here. These cases pin architecture
// dispatch, flat-config resolution and weight-namespace resolution — nothing
// about generated tokens. See .agents/specs/qwen38-text-only.md §Gates.
//
// WHAT IT DOES PIN, and why each shape was chosen:
//   1. dispatch          — both architecture strings resolve to the EXISTING
//                          dense/MoE factories with the text `_ModelInfo`.
//   2. config            — the PUBLISHED config.json, committed verbatim as
//                          fixtures/qwen3_8_2_4t_a95b/config.json.
//   3. namespace probe   — clean / VL / vision-inclusive / mtp / mixed / empty.
//   4. dense LOADER      — two synthetic checkpoints, byte-identical payloads,
//                          only the namespace differing, loaded through the
//                          production `LoadQwen3_5Dense`.
//   4b. MoE LOADER       — the same proof through `LoadQwen3_5Moe`, on BOTH
//                          expert-residency paths (eager, and the DEFERRED
//                          `load_layer_experts` closure actually driven). The
//                          MoE arm is the one this row exists for and it
//                          threads the prefix through three sites the dense
//                          loader has no analogue of.
//   5. inertness         — the VL spelling stays the per-layer seam DEFAULT.
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"

using vllm::HfConfig;
using vllm::ModelRegistration;
using vllm::ModelRegistry;

namespace {

HfConfig ArchConfig(std::vector<std::string> architectures) {
  HfConfig config;
  config.architectures = std::move(architectures);
  return config;
}

// ---------------------------------------------------------------------------
// Synthetic checkpoint plumbing (same shape as tests/vllm/test_load_direct_upload
// .cpp: a real safetensors file on disk, opened through the production reader).
// ---------------------------------------------------------------------------

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes, const char* tag) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_qwen3_8_" + std::string(tag) + "_" +
              std::to_string(counter++) + ".safetensors"))
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

// One tiny synthetic tensor description: name + shape + safetensors dtype
// string, filled with a deterministic per-tensor pattern so a wrong binding
// shows up in the VALUES, not only in a name. The dense arm is all-BF16 (the
// default); the MoE arm needs the ONE per-class quantization `LoadQwen3_5Moe`
// can read — per-tensor FP8 attention and PER-EXPERT NVFP4 projections/head —
// because that loader hard-requires each of those dtypes
// (qwen3_5_weights.cpp:385-463) and carries no bf16 and no stacked branch.
//
// That is NOT what the published Qwen3.5-family MoE repos ship. Both
// `Qwen/Qwen3.8-2.4T-A95B` and `Qwen/Qwen3.6-35B-A3B` publish 3D-STACKED,
// UNQUANTIZED experts and zero `weight_scale` tensors; the layout below is what
// an NVFP4 requant (e.g. `nvidia/Qwen3.6-35B-A3B-NVFP4`, which is what our gated
// 35B row actually reads) ships. The published layout is a NOT-IMPLEMENTED arm
// and is pinned as a REFUSAL further down, not as a load.
struct Spec {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype = "BF16";
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  return n;
}

// Byte width of the safetensors dtypes this file emits. Must agree with the
// reader's own table (safetensors_reader.cpp:41-48), which REJECTS a header
// whose shape times dtype width does not equal its data_offsets span.
size_t ElemSize(const std::string& dtype) {
  if (dtype == "BF16") return 2;
  if (dtype == "F32") return 4;
  return 1;  // U8 (packed fp4 codes) / F8_E4M3
}

// Builds a whole safetensors file from `specs`. The bytes of a tensor depend
// ONLY on its position in `specs`, so two files built from the same specs with
// different NAMES carry byte-identical payloads — which is what lets the
// namespace test compare two loads for byte equality.
std::string BuildSafetensors(const std::vector<Spec>& specs) {
  std::string header = "{";
  std::string body;
  uint64_t offset = 0;
  for (size_t i = 0; i < specs.size(); ++i) {
    const int64_t n = Numel(specs[i].shape);
    const std::string& dtype = specs[i].dtype;
    const size_t elem = ElemSize(dtype);
    const auto nbytes = static_cast<uint64_t>(n) * elem;
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
      for (size_t e = 0; e < static_cast<size_t>(n); ++e) {
        // A finite, distinct bf16 per (tensor index, element index).
        const auto v = static_cast<uint16_t>(0x3d00 + ((i * 37 + e * 7) & 0x1ff));
        std::memcpy(dst + e * 2, &v, 2);
      }
    } else if (dtype == "F32") {
      for (size_t e = 0; e < static_cast<size_t>(n); ++e) {
        // A finite, POSITIVE, distinct f32 — these are the per-tensor
        // weight_scale / input_scale / weight_scale_2 scalars, and the loader
        // multiplies them into `alpha`, so a NaN would make an equality
        // assertion on the derived scalar meaningless.
        const float v = 0.125F * static_cast<float>((i * 5 + e * 3) % 7 + 1);
        std::memcpy(dst + e * 4, &v, 4);
      }
    } else {
      for (size_t e = 0; e < static_cast<size_t>(n); ++e) {
        dst[e] = static_cast<char>((i * 37 + e * 7) & 0xff);
      }
    }
  }
  header += "}";
  return U64Le(header.size()) + header + body;
}

// The full backbone tensor list of a ONE-layer full-attention Qwen3.5 dense
// checkpoint under prefix `p`, plus the top-level tied-head case (no lm_head).
// Names verified against the published `Qwen/Qwen3.8-2.4T-A95B` and
// `Qwen/Qwen3.6-35B-A3B` safetensors indices: identical modulo the prefix.
std::vector<Spec> DenseOneLayerSpecs(const std::string& p) {
  const std::string l = p + "layers.0.";
  const std::string sa = l + "self_attn.";
  const std::string mlp = l + "mlp.";
  constexpr int64_t kHidden = 8;
  constexpr int64_t kFfn = 16;
  constexpr int64_t kHeadDim = 4;
  constexpr int64_t kQ = 8;   // 2 heads x 4
  constexpr int64_t kKv = 4;  // 1 head  x 4
  return {
      {p + "embed_tokens.weight", {6, kHidden}},
      {p + "norm.weight", {kHidden}},
      {l + "input_layernorm.weight", {kHidden}},
      {l + "post_attention_layernorm.weight", {kHidden}},
      {sa + "q_proj.weight", {kQ, kHidden}},
      {sa + "k_proj.weight", {kKv, kHidden}},
      {sa + "v_proj.weight", {kKv, kHidden}},
      {sa + "o_proj.weight", {kHidden, kQ}},
      {sa + "q_norm.weight", {kHeadDim}},
      {sa + "k_norm.weight", {kHeadDim}},
      {mlp + "gate_proj.weight", {kFfn, kHidden}},
      {mlp + "up_proj.weight", {kFfn, kHidden}},
      {mlp + "down_proj.weight", {kHidden, kFfn}},
  };
}

// The tensor payload of a safetensors blob: everything after the 8-byte header
// length and the JSON header itself.
std::string Payload(const std::string& file) {
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) {
    header_len = (header_len << 8) | static_cast<uint8_t>(file[static_cast<size_t>(i)]);
  }
  return file.substr(8 + static_cast<size_t>(header_len));
}

std::vector<std::string> NamesOf(const std::vector<Spec>& specs) {
  std::vector<std::string> names;
  names.reserve(specs.size());
  for (const Spec& s : specs) names.push_back(s.name);
  return names;
}

HfConfig OneLayerDenseConfig() {
  HfConfig config;
  config.model_type = "qwen3_5_text";
  config.hidden_size = 8;
  config.num_hidden_layers = 1;
  config.layer_types = {"full_attention"};
  return config;
}

// Byte-for-byte comparison of a loaded tensor pair.
void CheckSameBytes(const vllm::OwnedTensor& a, const vllm::OwnedTensor& b,
                    const char* what) {
  CAPTURE(what);
  REQUIRE(a.rank == b.rank);
  for (int i = 0; i < a.rank; ++i) CHECK(a.shape[i] == b.shape[i]);
  REQUIRE(a.bytes.size() == b.bytes.size());
  REQUIRE(a.bytes.size() > 0);
  CHECK(std::memcmp(a.bytes.data(), b.bytes.data(), a.bytes.size()) == 0);
}

// ---------------------------------------------------------------------------
// MoE arm. `Qwen3_5MoeForCausalLM` is the architecture this row exists for, and
// the MoE loader threads the resolved prefix through THREE further sites the
// dense loader has no analogue of: the per-layer base (LoadLayerImpl,
// qwen3_5_weights.cpp:560-561), the top-level embed/norm pair
// (LoadQwen3_5Moe:677-678), and — only on the shards-owner path — the DEFERRED
// per-layer routed-expert closure (:698-709), which captures the prefix BY
// VALUE and runs long after the resolution frame is gone.
//
// The quantization per weight class is the ONLY scheme `LoadQwen3_5Moe` reads
// (qwen3_5_weights.h:1-13): bf16 embeds/norms/router/shared-gate, per-tensor
// FP8 attention, PER-EXPERT NVFP4 routed experts + shared expert + lm_head.
// NVFP4 requires K % 16 == 0, which is what fixes the toy hidden size at 16.
//
// This is the NVFP4-requant layout (`nvidia/Qwen3.6-35B-A3B-NVFP4` and the
// like), NOT the layout of the published `Qwen/Qwen3.6-35B-A3B` or
// `Qwen/Qwen3.8-2.4T-A95B` repos, which are stacked and unquantized. What
// follows therefore proves the prefix is threaded correctly through the arm we
// implement; the published arm is a refusal, pinned separately.
// ---------------------------------------------------------------------------
constexpr int64_t kMoeHidden = 16;
constexpr int64_t kMoeInter = 16;
constexpr int64_t kMoeVocab = 6;
constexpr int64_t kMoeExperts = 2;
constexpr int64_t kMoeHeadDim = 4;
constexpr int64_t kMoeQ = 8;   // 2 heads x 4
constexpr int64_t kMoeKv = 4;  // 1 head  x 4

// Per-tensor FP8 (W8A8) projection: weight + weight_scale + input_scale. Both
// scalars are emitted so the fixture loads on EITHER arm of DenseNativeEnabled()
// (fp8-resident on a CUDA+cutlass build, dequant-to-bf16 otherwise).
void AppendFp8(std::vector<Spec>& out, const std::string& proj, int64_t out_dim,
               int64_t in_dim) {
  out.push_back({proj + ".weight", {out_dim, in_dim}, "F8_E4M3"});
  out.push_back({proj + ".weight_scale", {1}, "F32"});
  out.push_back({proj + ".input_scale", {1}, "F32"});
}

// NVFP4 W4A16 projection: packed [N, K/2] U8 codes + [N, K/16] fp8-e4m3 group
// scales + the f32 per-tensor global.
void AppendNvfp4(std::vector<Spec>& out, const std::string& proj,
                 int64_t out_dim, int64_t in_dim) {
  out.push_back({proj + ".weight", {out_dim, in_dim / 2}, "U8"});
  out.push_back({proj + ".weight_scale", {out_dim, in_dim / 16}, "F8_E4M3"});
  out.push_back({proj + ".weight_scale_2", {1}, "F32"});
}

// The full tensor list of a ONE-layer, full-attention, two-expert Qwen3.5 MoE
// checkpoint under backbone prefix `p`. `lm_head` is deliberately TOP-LEVEL
// (unprefixed) in both spellings, exactly as both published indices have it.
std::vector<Spec> MoeOneLayerSpecs(const std::string& p) {
  const std::string l = p + "layers.0.";
  const std::string sa = l + "self_attn.";
  const std::string mlp = l + "mlp.";
  std::vector<Spec> s{
      {p + "embed_tokens.weight", {kMoeVocab, kMoeHidden}},
      {p + "norm.weight", {kMoeHidden}},
      {l + "input_layernorm.weight", {kMoeHidden}},
      {l + "post_attention_layernorm.weight", {kMoeHidden}},
  };
  AppendFp8(s, sa + "q_proj", kMoeQ, kMoeHidden);
  AppendFp8(s, sa + "k_proj", kMoeKv, kMoeHidden);
  AppendFp8(s, sa + "v_proj", kMoeKv, kMoeHidden);
  AppendFp8(s, sa + "o_proj", kMoeHidden, kMoeQ);
  s.push_back({sa + "q_norm.weight", {kMoeHeadDim}});
  s.push_back({sa + "k_norm.weight", {kMoeHeadDim}});
  // Router + shared-expert gate (bf16, transposed at load).
  s.push_back({mlp + "gate.weight", {kMoeExperts, kMoeHidden}});
  s.push_back({mlp + "shared_expert_gate.weight", {1, kMoeHidden}});
  for (int64_t e = 0; e < kMoeExperts; ++e) {
    const std::string ex = mlp + "experts." + std::to_string(e) + ".";
    AppendNvfp4(s, ex + "gate_proj", kMoeInter, kMoeHidden);
    AppendNvfp4(s, ex + "up_proj", kMoeInter, kMoeHidden);
    AppendNvfp4(s, ex + "down_proj", kMoeHidden, kMoeInter);
  }
  const std::string se = mlp + "shared_expert.";
  AppendNvfp4(s, se + "gate_proj", kMoeInter, kMoeHidden);
  AppendNvfp4(s, se + "up_proj", kMoeInter, kMoeHidden);
  AppendNvfp4(s, se + "down_proj", kMoeHidden, kMoeInter);
  AppendNvfp4(s, "lm_head", kMoeVocab, kMoeHidden);
  // THE MTP DRAFT HEAD, AND WHY IT IS IN THE *SUPPORTED* FIXTURE. Read off the
  // REAL gated index `nvidia/Qwen3.6-35B-A3B-NVFP4/model.safetensors.index.json`
  // (124,468 tensors, fetched 2026-08-12): the checkpoint this loader is built
  // for DOES carry the 3-D STACKED expert spelling — but only under the
  // top-level `mtp.` prefix, as exactly these two names. Its 92 backbone layers
  // are per-expert NVFP4 throughout (zero stacked names under
  // `model.language_model.layers.`, zero expert `.weight` without a `_scale`
  // sibling), and `lm_head.weight_scale` / `lm_head.weight_scale_2` are both
  // present, so the checkpoint loads.
  //
  // It loads ONLY because `CheckMoeExpertLayoutSupported` scans names under
  // `<backbone>layers.` (qwen3_5_weights.cpp:633,638) and `mtp.` is neither
  // `model.layers.` nor `model.language_model.layers.`. That exclusion is
  // LOAD-BEARING, not incidental: broadening the scan to every `.mlp.experts.`
  // name would refuse the one checkpoint we gate today, on a CUDA-only load
  // path, with the whole CPU suite still green. These two entries put the real
  // index's shape into the fixture so that regression is CPU-visible — see the
  // dedicated subcase in case 4c. `LoadQwen3_5Moe` never requests `mtp.*`
  // (LoadQwen3_5MTP loads the draft head separately, only under speculative
  // decoding), so they are inert to every other assertion here.
  const std::string mtp = "mtp.layers.0.mlp.";
  s.push_back({mtp + "experts.gate_up_proj",
               {kMoeExperts, 2 * kMoeInter, kMoeHidden}});
  s.push_back({mtp + "experts.down_proj", {kMoeExperts, kMoeHidden, kMoeInter}});
  return s;
}

// ---------------------------------------------------------------------------
// The PUBLISHED Qwen3.5-family MoE layouts, which `LoadQwen3_5Moe` does NOT
// implement. Read off the live safetensors indices on 2026-08-12:
//
//   Qwen/Qwen3.8-2.4T-A95B   1609 tensors, 92x `model.layers.N.mlp.experts
//                            .gate_up_proj` + 92x `.down_proj` (3-D STACKED),
//                            ZERO `weight_scale`, ZERO `input_scale`,
//                            `lm_head.weight` alone.
//   Qwen/Qwen3.6-35B-A3B     1045 tensors, same stacked spelling under the VL
//                            prefix, ZERO `weight_scale`.
//
// `LoadMoeExpertsInto` (qwen3_5_weights.cpp:519-530) reads only per-expert
// `experts.<e>.<proj>` through `LoadNvfp4Raw` (:433-462), which hard-requires
// U8 `.weight` + F8_E4M3 `.weight_scale` + `.weight_scale_2`. There is no
// stacked branch and no bf16 branch, unlike `gemma4_weights.cpp:326`, which
// dispatches between layouts. Our gated 35B row reads the REQUANTIZED
// `nvidia/Qwen3.6-35B-A3B-NVFP4`; this loader has never read a published Qwen
// bf16 MoE repo. AGENTS.md requires an unimplemented arm be "refused with a
// message naming the missing piece" — these fixtures are that gate.
// ---------------------------------------------------------------------------

// Everything a published Qwen3.5 MoE checkpoint carries EXCEPT the routed
// experts: all bf16, no scale tensors anywhere, `lm_head.weight` top-level.
std::vector<Spec> PublishedMoeCoreSpecs(const std::string& p) {
  const std::string l = p + "layers.0.";
  const std::string sa = l + "self_attn.";
  const std::string mlp = l + "mlp.";
  return {
      {p + "embed_tokens.weight", {kMoeVocab, kMoeHidden}},
      {p + "norm.weight", {kMoeHidden}},
      {l + "input_layernorm.weight", {kMoeHidden}},
      {l + "post_attention_layernorm.weight", {kMoeHidden}},
      {sa + "q_proj.weight", {kMoeQ, kMoeHidden}},
      {sa + "k_proj.weight", {kMoeKv, kMoeHidden}},
      {sa + "v_proj.weight", {kMoeKv, kMoeHidden}},
      {sa + "o_proj.weight", {kMoeHidden, kMoeQ}},
      {sa + "q_norm.weight", {kMoeHeadDim}},
      {sa + "k_norm.weight", {kMoeHeadDim}},
      {mlp + "gate.weight", {kMoeExperts, kMoeHidden}},
      {mlp + "shared_expert.gate_proj.weight", {kMoeInter, kMoeHidden}},
      {mlp + "shared_expert.up_proj.weight", {kMoeInter, kMoeHidden}},
      {mlp + "shared_expert.down_proj.weight", {kMoeHidden, kMoeInter}},
      {mlp + "shared_expert_gate.weight", {1, kMoeHidden}},
      {"lm_head.weight", {kMoeVocab, kMoeHidden}},
  };
}

// The published shape: ONE 3-D tensor per layer holding ALL experts.
std::vector<Spec> PublishedStackedMoeSpecs(const std::string& p) {
  std::vector<Spec> s = PublishedMoeCoreSpecs(p);
  const std::string mlp = p + "layers.0.mlp.";
  s.push_back({mlp + "experts.gate_up_proj",
               {kMoeExperts, 2 * kMoeInter, kMoeHidden}});
  s.push_back({mlp + "experts.down_proj", {kMoeExperts, kMoeHidden, kMoeInter}});
  return s;
}

// The per-expert spelling this loader DOES resolve, but unquantized: plain bf16
// `<e>.<proj>.weight` with no `.weight_scale` beside it.
std::vector<Spec> UnquantizedPerExpertMoeSpecs(const std::string& p) {
  std::vector<Spec> s = PublishedMoeCoreSpecs(p);
  const std::string mlp = p + "layers.0.mlp.";
  for (int64_t e = 0; e < kMoeExperts; ++e) {
    const std::string ex = mlp + "experts." + std::to_string(e) + ".";
    s.push_back({ex + "gate_proj.weight", {kMoeInter, kMoeHidden}});
    s.push_back({ex + "up_proj.weight", {kMoeInter, kMoeHidden}});
    s.push_back({ex + "down_proj.weight", {kMoeHidden, kMoeInter}});
  }
  return s;
}

// Per-expert NVFP4 everywhere — the supported arm — but a plain bf16 head, the
// spelling the DENSE loader accepts (`LoadLmHeadAnyDtype`) and the MoE loader
// does not.
std::vector<Spec> MoeSpecsWithBf16LmHead(const std::string& p) {
  std::vector<Spec> s;
  for (const Spec& x : MoeOneLayerSpecs(p)) {
    if (x.name.rfind("lm_head.", 0) != 0) s.push_back(x);
  }
  s.push_back({"lm_head.weight", {kMoeVocab, kMoeHidden}});
  return s;
}

// Runs `fn` and returns the `what()` of whatever it threw, or "" if it returned
// normally. A refusal is only useful if it NAMES the missing piece, which
// CHECK_THROWS_AS cannot see.
std::string CaptureThrow(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool Mentions(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

// A TensorResolver + presence probe over ONE synthetic safetensors file, so a
// per-layer public seam can be driven exactly as the full loaders drive it —
// with the backbone-prefix argument OMITTED, which is the only way its DEFAULT
// is observable at all.
class ShardBag {
 public:
  ShardBag(const std::vector<Spec>& specs, const char* tag)
      : file_(BuildSafetensors(specs), tag),
        shard_(vllm::SafetensorsFile::Open(file_.path())) {}
  ShardBag(const ShardBag&) = delete;
  ShardBag& operator=(const ShardBag&) = delete;

  vllm::TensorResolver Resolver() const {
    const vllm::SafetensorsFile* shard = &shard_;
    return [shard](const std::string& name) -> const vllm::StTensor& {
      // Same message shape `LoadQwen3_5Moe`'s own resolver produces, so a miss
      // is reported by the NAME that was looked for.
      for (const std::string& have : shard->Names()) {
        if (have == name) return shard->Get(name);
      }
      throw std::runtime_error("qwen3_5 weights: tensor not found: " + name);
    };
  }

  std::function<bool(const std::string&)> Has() const {
    const vllm::SafetensorsFile* shard = &shard_;
    return [shard](const std::string& name) {
      for (const std::string& have : shard->Names()) {
        if (have == name) return true;
      }
      return false;
    };
  }

 private:
  TempFile file_;
  vllm::SafetensorsFile shard_;
};

HfConfig OneLayerMoeConfig() {
  HfConfig config;
  config.model_type = "qwen3_5_moe_text";
  config.hidden_size = kMoeHidden;
  config.num_hidden_layers = 1;
  config.layer_types = {"full_attention"};
  config.num_experts = kMoeExperts;
  return config;
}

void CheckSameNvfp4(const vllm::Nvfp4Weight& a, const vllm::Nvfp4Weight& b,
                    const char* what) {
  CAPTURE(what);
  CHECK(a.n == b.n);
  CHECK(a.k == b.k);
  CHECK(a.scale2 == b.scale2);
  CheckSameBytes(a.packed, b.packed, what);
  CheckSameBytes(a.scale, b.scale, what);
}

void CheckSameFp8(const vllm::Fp8Weight& a, const vllm::Fp8Weight& b,
                  const char* what) {
  CAPTURE(what);
  CHECK(a.n == b.n);
  CHECK(a.k == b.k);
  CHECK(a.weight_scale == b.weight_scale);
  CHECK(a.input_scale == b.input_scale);
  CHECK(a.alpha == b.alpha);
  CheckSameBytes(a.packed, b.packed, what);
}

// Which arm of the attention projections the loader fills depends on the BUILD,
// not on this test: DenseNativeEnabled() keeps them fp8-resident on a
// CUDA+cutlass build and dequants them to bf16 otherwise. Assert on whichever
// this build populated, and REQUIRE that exactly one of the two is populated.
void CheckSameAttn(const vllm::FullAttnLayerWeights& a,
                   const vllm::FullAttnLayerWeights& b) {
  const bool fp8 = !a.q_proj_fp8.Empty();
  REQUIRE(fp8 == !b.q_proj_fp8.Empty());
  REQUIRE(fp8 == a.q_proj.Empty());
  REQUIRE(fp8 == b.q_proj.Empty());
  if (fp8) {
    CheckSameFp8(a.q_proj_fp8, b.q_proj_fp8, "q_proj fp8");
    CheckSameFp8(a.k_proj_fp8, b.k_proj_fp8, "k_proj fp8");
    CheckSameFp8(a.v_proj_fp8, b.v_proj_fp8, "v_proj fp8");
    CheckSameFp8(a.o_proj_fp8, b.o_proj_fp8, "o_proj fp8");
  } else {
    CheckSameBytes(a.q_proj, b.q_proj, "q_proj");
    CheckSameBytes(a.k_proj, b.k_proj, "k_proj");
    CheckSameBytes(a.v_proj, b.v_proj, "v_proj");
    CheckSameBytes(a.o_proj, b.o_proj, "o_proj");
  }
  CheckSameBytes(a.q_norm, b.q_norm, "q_norm");
  CheckSameBytes(a.k_norm, b.k_norm, "k_norm");
}

void CheckSameMoeBlock(const vllm::MoeBlockWeights& a,
                       const vllm::MoeBlockWeights& b) {
  CheckSameBytes(a.router_gate, b.router_gate, "mlp.gate");
  CheckSameBytes(a.shared_gate, b.shared_gate, "mlp.shared_expert_gate");
  CheckSameNvfp4(a.shared_gate_proj_fp4, b.shared_gate_proj_fp4,
                 "shared_expert.gate_proj");
  CheckSameNvfp4(a.shared_up_proj_fp4, b.shared_up_proj_fp4,
                 "shared_expert.up_proj");
  CheckSameNvfp4(a.shared_down_proj_fp4, b.shared_down_proj_fp4,
                 "shared_expert.down_proj");
  REQUIRE(a.expert_gate_fp4.size() == static_cast<size_t>(kMoeExperts));
  REQUIRE(b.expert_gate_fp4.size() == a.expert_gate_fp4.size());
  REQUIRE(a.expert_up_fp4.size() == a.expert_gate_fp4.size());
  REQUIRE(a.expert_down_fp4.size() == a.expert_gate_fp4.size());
  for (size_t e = 0; e < a.expert_gate_fp4.size(); ++e) {
    CAPTURE(e);
    CheckSameNvfp4(a.expert_gate_fp4[e], b.expert_gate_fp4[e], "expert gate");
    CheckSameNvfp4(a.expert_up_fp4[e], b.expert_up_fp4[e], "expert up");
    CheckSameNvfp4(a.expert_down_fp4[e], b.expert_down_fp4[e], "expert down");
  }
}

void CheckSameMoeModel(const vllm::Qwen3_5MoeWeights& a,
                       const vllm::Qwen3_5MoeWeights& b) {
  REQUIRE(a.layers.size() == b.layers.size());
  REQUIRE(a.layers.size() == 1u);
  CheckSameBytes(a.embed_tokens, b.embed_tokens, "embed_tokens");
  CheckSameBytes(a.final_norm, b.final_norm, "final_norm");
  CheckSameNvfp4(a.lm_head_fp4, b.lm_head_fp4, "lm_head");
  for (size_t l = 0; l < a.layers.size(); ++l) {
    CAPTURE(l);
    const vllm::Qwen3_5MoeLayerWeights& x = a.layers[l];
    const vllm::Qwen3_5MoeLayerWeights& y = b.layers[l];
    CHECK_FALSE(x.is_linear_attention);
    CHECK_FALSE(y.is_linear_attention);
    CheckSameBytes(x.input_layernorm, y.input_layernorm, "input_layernorm");
    CheckSameBytes(x.post_attention_layernorm, y.post_attention_layernorm,
                   "post_attention_layernorm");
    CheckSameAttn(x.attn, y.attn);
    CheckSameMoeBlock(x.moe, y.moe);
  }
}

}  // namespace

// ===========================================================================
// 1. Architecture dispatch. Upstream registers both text-only arms against the
//    SAME `qwen3_5` module (registry.py:202-203 @ `ad5d29db7`), so ours must
//    resolve to the SAME factories the ForConditionalGeneration wrappers use —
//    a second factory would be a fork of a backbone we already gate.
// ===========================================================================
TEST_CASE("qwen3_8: both text-only architecture strings resolve to the Qwen3.5 factories") {
  const HfConfig moe_config = ArchConfig({"Qwen3_5MoeForCausalLM"});
  const ModelRegistration& moe = ModelRegistry::Resolve(moe_config);
  CHECK(moe.architecture == "Qwen3_5MoeForCausalLM");
  CHECK(moe.factory ==
        vllm::RegistrationFor("Qwen3_5MoeForConditionalGeneration").factory);
  CHECK_FALSE(moe.factory->is_dense_model);

  const HfConfig dense_config = ArchConfig({"Qwen3_5ForCausalLM"});
  const ModelRegistration& dense = ModelRegistry::Resolve(dense_config);
  CHECK(dense.architecture == "Qwen3_5ForCausalLM");
  CHECK(dense.factory ==
        vllm::RegistrationFor("Qwen3_5ForConditionalGeneration").factory);
  CHECK(dense.factory->is_dense_model);

  // Upstream's `Qwen3_5ForCausalLMBase` inherits IsHybrid + HasInnerState but
  // NOT SupportsMultiModal (qwen3_5.py:287-296 @ `ad5d29db7`): these are the
  // TEXT arms, and their multimodal wrappers are separate registrations. Same
  // convention as `KimiLinearForCausalLM` — hybrid yes, multimodal no.
  for (const ModelRegistration* registration : {&moe, &dense}) {
    CAPTURE(registration->architecture);
    CHECK(registration->info.is_text_generation_model);
    CHECK(registration->info.is_hybrid);
    CHECK_FALSE(registration->info.supports_multimodal);
    CHECK_FALSE(registration->info.is_pooling_model);
  }
}

// ===========================================================================
// 2. Config resolution on the REAL flat 3.8 config — the PUBLISHED DOCUMENT,
//    not a hand-written approximation of it.
//
//    `tests/vllm/models/fixtures/qwen3_8_2_4t_a95b/config.json` is
//    https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/config.json,
//    committed VERBATIM (md5 303dc59227f1d03afc941646e8df3132, fetched
//    2026-08-12). It declares `model_type: qwen3_5_moe_text` at the TOP level
//    with no `text_config` wrapper, no `vision_config` and no `mrope_section`,
//    so the composite-wrapper path our 27B/35B checkpoints take does not apply.
//
//    Three things a paraphrase got wrong and this document does not: the rope
//    knobs live in a NESTED `rope_parameters` block (so the parse actually
//    reaches hf_config.cpp's rope_parameters branch, LooksLikeNestedRopeParameters
//    included, rather than the no-block early return); `layer_types` is present
//    and 92 entries long, which BOTH loaders hard-require to equal
//    num_hidden_layers; and the dtype key is transformers 4.57.3's `dtype`, not
//    the legacy `torch_dtype`.
// ===========================================================================
TEST_CASE("qwen3_8: the PUBLISHED 2.4T text config resolves through the shared Qwen3.5 path") {
  const HfConfig config = vllm::LoadHfConfig(QWEN3_8_CONFIG_FIXTURE);

  // The architecture the registry will be asked for, straight off a flat doc.
  REQUIRE(config.architectures.size() == 1);
  CHECK(config.architectures[0] == "Qwen3_5MoeForCausalLM");
  CHECK(config.model_type == "qwen3_5_moe_text");
  CHECK_NOTHROW(ModelRegistry::Resolve(config));

  // Scale: every knob that differs from the 35B is CONFIG, never a constant.
  CHECK(config.hidden_size == 8192);
  CHECK(config.num_hidden_layers == 92);
  CHECK(config.num_attention_heads == 64);
  CHECK(config.num_key_value_heads == 4);
  CHECK(config.head_dim == 256);
  CHECK(config.vocab_size == 248320);
  CHECK(config.num_experts == 512);
  CHECK(config.num_experts_per_tok == 10);
  CHECK(config.moe_intermediate_size == 2048);
  CHECK(config.shared_expert_intermediate_size == 2048);
  CHECK(config.linear_num_key_heads == 16);
  CHECK(config.linear_num_value_heads == 128);
  CHECK(config.linear_key_head_dim == 128);
  CHECK(config.linear_value_head_dim == 128);
  CHECK(config.linear_conv_kernel_dim == 4);
  CHECK(config.rope_theta == doctest::Approx(1e7).scale(0.0));

  // `layer_types` is a HARD requirement of both Qwen3.5 loaders
  // (qwen3_5_weights.cpp:660-663, qwen3_5_dense_weights.cpp) — they refuse a
  // checkpoint whose list does not equal num_hidden_layers. The published
  // pattern is [linear, linear, linear, full] x 23, i.e.
  // full_attention_interval 4, the same interleave as the 35B.
  REQUIRE(static_cast<int64_t>(config.layer_types.size()) ==
          config.num_hidden_layers);
  REQUIRE(config.layer_types.size() == 92u);
  for (size_t i = 0; i < config.layer_types.size(); ++i) {
    CAPTURE(i);
    CHECK(config.layer_types[i] ==
          ((i % 4 == 3) ? "full_attention" : "linear_attention"));
  }

  // The rope block is NESTED and flat-valued, so ParseRopeParameters takes the
  // rope_parameters branch, LooksLikeNestedRopeParameters is false (its values
  // are scalars, not per-layer-type objects), and the block's own
  // partial_rotary_factor/rope_theta win. The 0.25 also appears at top level,
  // and IsQwen35Family would default it to 0.25 in any case — all three agree,
  // which is why rotary_dim is 0.25 * 256.
  CHECK(config.has_rope_parameters);
  CHECK(config.rope_parameters.rope_type == "default");
  CHECK(config.rope_parameters.rope_theta == doctest::Approx(1e7).scale(0.0));
  CHECK(config.rope_parameters.partial_rotary_factor ==
        doctest::Approx(0.25).scale(0.0));
  CHECK(config.rotary_dim == 64);

  // A text-only checkpoint has no vision tower and no MRoPE sections. Both are
  // ABSENT rather than empty-but-present, and neither may be synthesized.
  CHECK(config.raw.find("vision_config") == config.raw.end());
  CHECK(config.raw.find("text_config") == config.raw.end());
  CHECK(config.rope_parameters.mrope_section.empty());

  // DELIBERATELY UNCONSUMED, PINNED SO IT CANNOT DRIFT SILENTLY. transformers
  // 4.57.3 writes the model dtype as `dtype`; hf_config.cpp:520-522 reads only
  // the legacy `torch_dtype`, so `cfg.torch_dtype` is EMPTY on this document.
  // That is inert today — nothing in the tree reads `HfConfig::torch_dtype` —
  // and consuming `dtype` is a behavior change on EVERY model, so it is
  // recorded as a tracked deviation (porting-inventory.md §9 deviation 17)
  // rather than smuggled in on this row. This assertion is the tripwire: it
  // fails the day someone teaches hf_config the new spelling, forcing the
  // deviation to be discharged rather than forgotten.
  CHECK(config.raw.at("dtype") == "bfloat16");
  CHECK(config.raw.find("torch_dtype") == config.raw.end());
  CHECK(config.torch_dtype.empty());

  // The published document has no top-level `intermediate_size` — the MoE
  // widths are `moe_intermediate_size` / `shared_expert_intermediate_size`.
  CHECK(config.raw.find("intermediate_size") == config.raw.end());
}

// ===========================================================================
// 3. Weight-namespace resolution. Upstream normalizes with ONE WeightsMapper
//    (qwen3_5.py:296-300 @ `ad5d29db7`); we mirror that with ONE resolution per
//    checkpoint rather than a per-lookup fallback, because a per-lookup fallback
//    would let a checkpoint bind half its tensors from each namespace and still
//    appear to load.
// ===========================================================================
TEST_CASE("qwen3_8: the backbone weight namespace is resolved ONCE per checkpoint") {
  SUBCASE("a clean `model.` index resolves to the canonical namespace") {
    CHECK(vllm::ResolveQwen3_5BackbonePrefix(
              NamesOf(DenseOneLayerSpecs("model."))) == "model.");
  }

  SUBCASE("a VL-prefixed index resolves to `model.language_model.`") {
    CHECK(vllm::ResolveQwen3_5BackbonePrefix(
              NamesOf(DenseOneLayerSpecs("model.language_model."))) ==
          "model.language_model.");
  }

  SUBCASE("a VISION-INCLUSIVE VL checkpoint is still the VL namespace") {
    // The 27B/35B vision-inclusive checkpoints carry `model.visual.*` NEXT TO
    // `model.language_model.*`. `model.visual.` is not a backbone spelling, so
    // it must never be mistaken for the canonical `model.` namespace and turn a
    // checkpoint we gate today into a refusal.
    std::vector<std::string> names = NamesOf(DenseOneLayerSpecs("model.language_model."));
    names.push_back("model.visual.patch_embed.proj.weight");
    names.push_back("model.visual.blocks.0.attn.qkv.weight");
    names.push_back("lm_head.weight");
    CHECK(vllm::ResolveQwen3_5BackbonePrefix(names) == "model.language_model.");
  }

  SUBCASE("the optional `mtp.*` draft head does not decide the namespace") {
    std::vector<std::string> names = NamesOf(DenseOneLayerSpecs("model."));
    names.push_back("mtp.fc.weight");
    names.push_back("mtp.layers.0.input_layernorm.weight");
    CHECK(vllm::ResolveQwen3_5BackbonePrefix(names) == "model.");
  }

  SUBCASE("a MIXED index is REFUSED, not silently half-loaded") {
    std::vector<std::string> names = NamesOf(DenseOneLayerSpecs("model."));
    names.push_back("model.language_model.layers.1.input_layernorm.weight");
    CHECK_THROWS_AS(vllm::ResolveQwen3_5BackbonePrefix(names),
                    std::runtime_error);
  }

  SUBCASE("an index with NEITHER namespace is refused") {
    const std::vector<std::string> names{"lm_head.weight", "mtp.fc.weight"};
    CHECK_THROWS_AS(vllm::ResolveQwen3_5BackbonePrefix(names),
                    std::runtime_error);
  }
}

// ===========================================================================
// 4. The loader must USE the resolved prefix. Two synthetic checkpoints with
//    byte-identical payloads and only the namespace differing must produce
//    byte-identical weights — which no amount of name-mapping unit testing can
//    show on its own.
// ===========================================================================
TEST_CASE("qwen3_8: the dense loader reads the SAME weights through either namespace") {
  const std::vector<Spec> vl = DenseOneLayerSpecs("model.language_model.");
  const std::vector<Spec> flat = DenseOneLayerSpecs("model.");
  REQUIRE(vl.size() == flat.size());

  const std::string vl_bytes = BuildSafetensors(vl);
  const std::string flat_bytes = BuildSafetensors(flat);
  // Same specs in the same order => IDENTICAL payloads, only the names differ.
  // Assert that here, so a later byte-equality of the two loads cannot be
  // satisfied by two identically-WRONG reads of two different payloads.
  REQUIRE(Payload(vl_bytes) == Payload(flat_bytes));
  REQUIRE(vl_bytes != flat_bytes);
  const TempFile vl_file(vl_bytes, "vl");
  const TempFile flat_file(flat_bytes, "flat");

  std::vector<vllm::SafetensorsFile> vl_shards;
  vl_shards.push_back(vllm::SafetensorsFile::Open(vl_file.path()));
  std::vector<vllm::SafetensorsFile> flat_shards;
  flat_shards.push_back(vllm::SafetensorsFile::Open(flat_file.path()));

  const HfConfig config = OneLayerDenseConfig();
  const vllm::Qwen3_5DenseWeights from_vl =
      vllm::LoadQwen3_5Dense(vl_shards, config);
  const vllm::Qwen3_5DenseWeights from_flat =
      vllm::LoadQwen3_5Dense(flat_shards, config);

  REQUIRE(from_vl.layers.size() == 1);
  REQUIRE(from_flat.layers.size() == 1);
  // No lm_head in either index => both tie the head to the embedding table.
  CHECK(from_vl.tied_lm_head);
  CHECK(from_flat.tied_lm_head);

  CheckSameBytes(from_vl.embed_tokens, from_flat.embed_tokens, "embed_tokens");
  CheckSameBytes(from_vl.final_norm, from_flat.final_norm, "final_norm");
  const vllm::Qwen3_5DenseLayerWeights& a = from_vl.layers[0];
  const vllm::Qwen3_5DenseLayerWeights& b = from_flat.layers[0];
  CHECK_FALSE(a.is_linear_attention);
  CHECK_FALSE(b.is_linear_attention);
  CheckSameBytes(a.input_layernorm, b.input_layernorm, "input_layernorm");
  CheckSameBytes(a.post_attention_layernorm, b.post_attention_layernorm,
                 "post_attention_layernorm");
  CheckSameBytes(a.attn.q_proj, b.attn.q_proj, "q_proj");
  CheckSameBytes(a.attn.k_proj, b.attn.k_proj, "k_proj");
  CheckSameBytes(a.attn.v_proj, b.attn.v_proj, "v_proj");
  CheckSameBytes(a.attn.o_proj, b.attn.o_proj, "o_proj");
  CheckSameBytes(a.attn.q_norm, b.attn.q_norm, "q_norm");
  CheckSameBytes(a.attn.k_norm, b.attn.k_norm, "k_norm");
  CheckSameBytes(a.mlp.gate_up_proj, b.mlp.gate_up_proj, "gate_up_proj");
  CheckSameBytes(a.mlp.down_proj, b.mlp.down_proj, "down_proj");
}

// ===========================================================================
// 4b. THE SAME PROOF FOR THE MoE ARM — the architecture this row exists for.
//     `LoadQwen3_5Moe` threads the resolved prefix through three sites the
//     dense loader has no analogue of, and the third of them lives in a
//     closure that runs AFTER the resolving frame has returned. Two synthetic
//     one-layer MoE checkpoints with byte-identical payloads and only the
//     namespace differing must load to byte-identical weights on BOTH expert
//     residency paths:
//       * shards_owner == nullptr  -> routed experts loaded EAGERLY at load;
//       * shards_owner != nullptr  -> routed experts DEFERRED behind
//         `load_layer_experts`, which is then driven explicitly so the closure
//         body actually executes (an installed-but-never-called closure pins
//         nothing).
//     Reverting any of the three sites to the hardcoded `model.language_model.`
//     literal makes the flat load throw `tensor not found`.
// ===========================================================================
TEST_CASE("qwen3_8: the MoE loader reads the SAME weights through either namespace") {
  const std::vector<Spec> vl = MoeOneLayerSpecs("model.language_model.");
  const std::vector<Spec> flat = MoeOneLayerSpecs("model.");
  REQUIRE(vl.size() == flat.size());

  const std::string vl_bytes = BuildSafetensors(vl);
  const std::string flat_bytes = BuildSafetensors(flat);
  // Same specs in the same order => IDENTICAL payloads, only the names differ,
  // so a later byte-equality of the two loads cannot be satisfied by two
  // identically-WRONG reads of two different payloads.
  REQUIRE(Payload(vl_bytes) == Payload(flat_bytes));
  REQUIRE(vl_bytes != flat_bytes);
  const TempFile vl_file(vl_bytes, "moe_vl");
  const TempFile flat_file(flat_bytes, "moe_flat");
  const HfConfig config = OneLayerMoeConfig();

  SUBCASE("EAGER experts (no shards owner)") {
    std::vector<vllm::SafetensorsFile> vl_shards;
    vl_shards.push_back(vllm::SafetensorsFile::Open(vl_file.path()));
    std::vector<vllm::SafetensorsFile> flat_shards;
    flat_shards.push_back(vllm::SafetensorsFile::Open(flat_file.path()));

    const vllm::Qwen3_5MoeWeights from_vl =
        vllm::LoadQwen3_5Moe(vl_shards, config);
    const vllm::Qwen3_5MoeWeights from_flat =
        vllm::LoadQwen3_5Moe(flat_shards, config);

    // No owner => no streaming closure, and the routed experts are already here.
    CHECK_FALSE(static_cast<bool>(from_vl.load_layer_experts));
    CHECK_FALSE(static_cast<bool>(from_flat.load_layer_experts));
    REQUIRE(from_flat.layers.size() == 1u);
    REQUIRE(from_flat.layers[0].moe.expert_gate_fp4.size() ==
            static_cast<size_t>(kMoeExperts));
    CheckSameMoeModel(from_vl, from_flat);
  }

  SUBCASE("DEFERRED experts (shards owner) — the closure runs") {
    auto vl_owner = std::make_shared<std::vector<vllm::SafetensorsFile>>();
    vl_owner->push_back(vllm::SafetensorsFile::Open(vl_file.path()));
    auto flat_owner = std::make_shared<std::vector<vllm::SafetensorsFile>>();
    flat_owner->push_back(vllm::SafetensorsFile::Open(flat_file.path()));

    vllm::Qwen3_5MoeWeights from_vl =
        vllm::LoadQwen3_5Moe(*vl_owner, config, vl_owner);
    vllm::Qwen3_5MoeWeights from_flat =
        vllm::LoadQwen3_5Moe(*flat_owner, config, flat_owner);

    // Deferred precondition: the closure is installed and NOTHING routed is
    // resident yet, so what the next four lines compare is what it produced.
    REQUIRE(static_cast<bool>(from_vl.load_layer_experts));
    REQUIRE(static_cast<bool>(from_flat.load_layer_experts));
    REQUIRE(from_vl.layers.size() == 1u);
    REQUIRE(from_flat.layers.size() == 1u);
    CHECK(from_vl.layers[0].moe.expert_gate_fp4.empty());
    CHECK(from_flat.layers[0].moe.expert_gate_fp4.empty());

    // Move first, exactly as the real load moves the weights into the
    // LoadedModel before PrepareMarlinResident drives the closure.
    vllm::Qwen3_5MoeWeights vl_moved = std::move(from_vl);
    vllm::Qwen3_5MoeWeights flat_moved = std::move(from_flat);
    vl_moved.load_layer_experts(0, vl_moved.layers[0].moe);
    flat_moved.load_layer_experts(0, flat_moved.layers[0].moe);

    REQUIRE(flat_moved.layers[0].moe.expert_gate_fp4.size() ==
            static_cast<size_t>(kMoeExperts));
    CheckSameMoeModel(vl_moved, flat_moved);
  }
}

// ===========================================================================
// 4c. THE PUBLISHED MoE EXPERT LAYOUT IS AN UNIMPLEMENTED ARM, AND IS REFUSED
//     BY NAME. Registering the architecture and resolving the namespace does
//     NOT make `Qwen/Qwen3.8-2.4T-A95B` loadable: its routed experts are 3-D
//     STACKED and it carries no quantization scales at all, while
//     `LoadQwen3_5Moe` reads only per-expert NVFP4. Before this gate the load
//     died at `LoadNvfp4Raw(get, "lm_head")` with
//
//       qwen3_5 weights: expected U8 for lm_head.weight
//
//     which is indistinguishable from a corrupt or truncated checkpoint.
//     AGENTS.md: an arm that is not implemented "is refused with a message
//     naming the missing piece ... never left to be discovered later", and this
//     row's spec §Stop conditions says the same. NOTE this adds a REFUSAL only:
//     stacked/bf16 MoE expert loading is OWED and needs its own spec, RED-first
//     test and NVFP4 inertness proof.
// ===========================================================================
TEST_CASE("qwen3_8: the published stacked/unquantized MoE arm is REFUSED, and the message names it") {
  auto load = [](const std::vector<Spec>& specs, const char* tag) {
    return CaptureThrow([&specs, tag] {
      const TempFile file(BuildSafetensors(specs), tag);
      std::vector<vllm::SafetensorsFile> shards;
      shards.push_back(vllm::SafetensorsFile::Open(file.path()));
      vllm::LoadQwen3_5Moe(shards, OneLayerMoeConfig());
    });
  };

  // Every refusal must say WHAT is missing and WHAT would be required — a bare
  // "unsupported" is the failure mode this case exists to prevent.
  auto names_the_requirement = [](const std::string& message) {
    CAPTURE(message);
    CHECK(Mentions(message, "qwen3_5 weights"));
    CHECK(Mentions(message, "not implemented"));
    CHECK(Mentions(message, "per-expert NVFP4"));
    CHECK(Mentions(message, "weight_scale"));
    // The two failures it must NOT degrade into: a raw dtype complaint about a
    // tensor the reader never reached, or a bare lookup miss.
    CHECK_FALSE(Mentions(message, "expected U8 for lm_head.weight"));
    CHECK_FALSE(Mentions(message, "tensor not found"));
  };

  SUBCASE("3-D stacked experts, flat namespace — the Qwen3.8-2.4T-A95B shape") {
    const std::string message = load(PublishedStackedMoeSpecs("model."), "stacked_flat");
    CAPTURE(message);
    CHECK(Mentions(message, "model.layers.0.mlp.experts.gate_up_proj"));
    CHECK(Mentions(message, "stacked"));
    names_the_requirement(message);
  }

  SUBCASE("3-D stacked experts, VL namespace — the Qwen3.6-35B-A3B shape") {
    // The published 35B repo is stacked too; only the NVFP4 REQUANT loads. So
    // this refusal is not a text-only-arm quirk — the MoE loader has never read
    // a published Qwen bf16 MoE checkpoint under EITHER spelling.
    const std::string message =
        load(PublishedStackedMoeSpecs("model.language_model."), "stacked_vl");
    CAPTURE(message);
    CHECK(Mentions(message,
                   "model.language_model.layers.0.mlp.experts.gate_up_proj"));
    CHECK(Mentions(message, "stacked"));
    names_the_requirement(message);
  }

  SUBCASE("per-expert but UNQUANTIZED experts") {
    const std::string message =
        load(UnquantizedPerExpertMoeSpecs("model."), "unquant_experts");
    CAPTURE(message);
    CHECK(Mentions(message, "model.layers.0.mlp.experts.0.gate_proj.weight"));
    CHECK(Mentions(message, "unquantized"));
    names_the_requirement(message);
  }

  SUBCASE("NVFP4 experts but an UNQUANTIZED lm_head") {
    // The dense arm accepts a bf16 head (`LoadLmHeadAnyDtype`,
    // qwen3_5_dense_weights.cpp:215-233); the MoE arm hard-requires NVFP4.
    const std::string message =
        load(MoeSpecsWithBf16LmHead("model."), "bf16_lmhead");
    CAPTURE(message);
    CHECK(Mentions(message, "lm_head.weight_scale"));
    CHECK(Mentions(message, "unquantized"));
    names_the_requirement(message);
  }

  SUBCASE("the SUPPORTED per-expert NVFP4 layout is untouched by the check") {
    // Inertness in the same case as the refusals, so the gate cannot be made
    // green by refusing everything: the arm we DO implement still loads, under
    // both namespaces.
    CHECK(load(MoeOneLayerSpecs("model."), "inert_flat").empty());
    CHECK(load(MoeOneLayerSpecs("model.language_model."), "inert_vl").empty());
  }

  SUBCASE("the gated checkpoint's `mtp.` STACKED experts do not trip it either") {
    // The one checkpoint this arm is actually gated on —
    // `nvidia/Qwen3.6-35B-A3B-NVFP4`, 124,468 tensors in its published
    // safetensors index — DOES contain the exact stacked spelling the first
    // refusal rejects, as `mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}`.
    // It loads only because the scan is anchored at `<backbone>layers.`
    // (qwen3_5_weights.cpp:633,638) and `mtp.` is under neither backbone
    // spelling. Nothing else in this suite pins that, so a later edit
    // broadening the scan to every `.mlp.experts.` name would refuse the
    // checkpoint we gate — on a CUDA-only load path, with the whole CPU suite
    // green. This subcase is that CPU-visible pin.
    auto carries = [](const std::vector<Spec>& specs, const std::string& name) {
      int hits = 0;
      for (const Spec& x : specs) {
        if (x.name == name) ++hits;
      }
      return hits;
    };
    // The fixture must still BE the real index's shape; a fixture that quietly
    // lost these names would leave the two loads below proving nothing.
    for (const char* p : {"model.", "model.language_model."}) {
      CAPTURE(p);
      const std::vector<Spec> specs = MoeOneLayerSpecs(p);
      REQUIRE(carries(specs, "mtp.layers.0.mlp.experts.gate_up_proj") == 1);
      REQUIRE(carries(specs, "mtp.layers.0.mlp.experts.down_proj") == 1);
      // ...and they must be OUTSIDE the scanned namespace, which is the whole
      // reason the refusal stays silent on them.
      REQUIRE(carries(specs, std::string(p) +
                                 "layers.0.mlp.experts.gate_up_proj") == 0);
    }
    CHECK(load(MoeOneLayerSpecs("model."), "mtp_stacked_flat").empty());
    CHECK(load(MoeOneLayerSpecs("model.language_model."), "mtp_stacked_vl")
              .empty());
  }
}

// ===========================================================================
// 5. INERTNESS of the gated rows. 27B / 35B / Coder are VL-prefixed
//    checkpoints; the per-layer public seams keep the VL prefix as their
//    DEFAULT, so every existing caller is unchanged by construction.
//
//    ASSERTING THE TWO CONSTANTS IS NOT THAT PROOF. Flipping both header
//    defaults from `kQwen3_5VlBackbonePrefix` to `kQwen3_5TextBackbonePrefix`
//    left this case entirely green (review finding F7), because the default
//    ARGUMENT is a third fact neither constant pins. So the seams are driven
//    below with the prefix argument OMITTED, exactly as every 27B/35B/Coder
//    caller drives them.
// ===========================================================================
TEST_CASE("qwen3_8: the VL prefix stays the default for the gated 27B/35B checkpoints") {
  // The two spellings are named constants, not literals scattered per lookup.
  CHECK(std::string(vllm::kQwen3_5VlBackbonePrefix) == "model.language_model.");
  CHECK(std::string(vllm::kQwen3_5TextBackbonePrefix) == "model.");

  // A 35B-shaped VL index. `...mlp.experts.gate_up_proj` is the PUBLISHED
  // stacked spelling, which the loader refuses (case 4c) — it appears here
  // because the namespace probe must ignore it either way: only
  // `<prefix>embed_tokens.weight`, `<prefix>norm.weight` and `<prefix>layers.`
  // vote, so what the expert tensors are called cannot change the answer.
  std::vector<std::string> names{
      "model.language_model.embed_tokens.weight",
      "model.language_model.norm.weight",
      "model.language_model.layers.0.input_layernorm.weight",
      "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
      "model.language_model.layers.0.mlp.experts.gate_up_proj",
      "model.language_model.layers.0.mlp.shared_expert_gate.weight",
      "model.language_model.layers.3.self_attn.q_proj.weight",
      "lm_head.weight",
  };
  CHECK(vllm::ResolveQwen3_5BackbonePrefix(names) == "model.language_model.");

  SUBCASE("LoadQwen3_5MoeLayer defaults to the VL spelling") {
    const ShardBag vl(MoeOneLayerSpecs("model.language_model."), "seam_moe_vl");
    const ShardBag flat(MoeOneLayerSpecs("model."), "seam_moe_flat");
    // Prefix argument OMITTED on both calls. The VL layer must bind...
    CHECK(CaptureThrow([&vl] {
            (void)vllm::LoadQwen3_5MoeLayer(vl.Resolver(), "full_attention", 0,
                                            kMoeExperts);
          }).empty());
    // ...and the flat one must fail looking for the VL name, which is what the
    // default actually being the VL spelling MEANS.
    const std::string message = CaptureThrow([&flat] {
      (void)vllm::LoadQwen3_5MoeLayer(flat.Resolver(), "full_attention", 0,
                                      kMoeExperts);
    });
    CAPTURE(message);
    CHECK(Mentions(message,
                   "model.language_model.layers.0.input_layernorm.weight"));
  }

  SUBCASE("LoadQwen3_5DenseLayer defaults to the VL spelling") {
    const ShardBag vl(DenseOneLayerSpecs("model.language_model."), "seam_dn_vl");
    const ShardBag flat(DenseOneLayerSpecs("model."), "seam_dn_flat");
    CHECK(CaptureThrow([&vl] {
            (void)vllm::LoadQwen3_5DenseLayer(vl.Resolver(), vl.Has(),
                                              "full_attention", 0);
          }).empty());
    const std::string message = CaptureThrow([&flat] {
      (void)vllm::LoadQwen3_5DenseLayer(flat.Resolver(), flat.Has(),
                                        "full_attention", 0);
    });
    CAPTURE(message);
    CHECK(Mentions(message,
                   "model.language_model.layers.0.input_layernorm.weight"));
  }
}
