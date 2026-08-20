// SPEC-DFLASH2 W2 (#1314) — the DFlash2 draft: its config, its grouped dynamic
// convolution weights, and the forward that runs them.
//
// BEYOND-PIN throughout. Upstream is
// `vllm/model_executor/models/qwen3_dflash2.py` @ vllm-project/vllm#52816 head
// `19c9351904df4c63042671bc67a866ca48dc7d6f`; the parity pin `555967922` does
// not carry the architecture and this row does not advance it.
//
// PART 1 — the config builder, which is `## Owed` O3 and O4 of the row's spec.
// `MakeQwen3DFlashDraftConfig` could not parse EITHER published DFlash2
// `config.json` at all: it did `c.at("rope_theta")` and `c.at("block_size")`
// while `z-lab/Qwen3.8-27B-DFlash2` nests them as `rope_parameters.rope_theta`
// and `dflash_config.block_size` and declares neither at the top level, so both
// `at` calls threw before any DFlash2 mechanism could be reached. It also did
// `c.at("layer_types")`, which `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` does not
// declare, while upstream reads `getattr(config, "layer_types", None)`
// (`qwen3_dflash.py:134` and `:66` @ that head). The configs embedded below are
// the PUBLISHED files BYTE-FOR-BYTE, each with the sha256 OF THE EMBEDDED
// LITERAL recorded beside it, so the gate does not depend on a checkout being
// present and the recorded hash describes what the compiler actually sees.
//
// The `attention_sink_bias` refusal is the other half of O4 and is not
// bookkeeping. Upstream reads `dflash_config.attention_sink_bias` and passes a
// per-head sink into its `Attention` (`qwen3_dflash.py:309-313` and `:240-257` @
// that head); this lane has no attention sink at all. Landing the `layer_types`
// fallback ALONE would let MiMo's draft parse and load with the sinks silently
// absent -- acceptance-only, token-invisible, which is the exact class this row
// exists to remove. So the key is refused BY NAME.
//
// PART 2 — the grouped convolution inside the draft forward. The op itself is
// gated in tests/vt/test_ops_dflash2_grouped_conv.cpp against upstream's own
// sequential reference; what is gated here is that the DRAFT MODEL runs it, on
// weights the PRODUCTION loader read off a real on-disk checkpoint, and that
// deleting the production call site turns this suite red.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cstring>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;
using nlohmann::json;
using vllm::HfConfig;
using vllm::MakeQwen3DFlashDraftConfig;
using vllm::OwnedTensor;
using vllm::Qwen3DFlashLayerAttnMode;
using vllm::Qwen3DFlashModel;
using vllm::Qwen3DFlashWeights;
using vllm::ResolveQwen3DFlashAttnModes;

namespace {

// `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`,
// config.json VERBATIM (1239 bytes, sha256
// 873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980, read on
// 2026-08-19). Kept whole rather than reduced: what this case gates is that the
// PUBLISHED document parses, and a reduced copy would only prove that a document
// this test wrote parses.
//
// The sha256 hashes THE LITERAL BELOW, trailing newline included, so the claim
// is checkable from this file alone rather than only against a copy on a share.
// The three literals in this file were all re-fetched and re-hashed on
// 2026-08-20; one of them (`kMimoDflashConfig`) had been re-indented and had
// lost a key, and it is now the published bytes.
constexpr const char* kQwen38Dflash2Config = R"JSON({
  "architectures": [
    "DFlash2DraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "is_causal": false,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [
      5,
      19,
      33,
      47,
      61
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 17408,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 262144,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 64,
  "pad_token_id": 248044,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.15.0",
  "use_cache": true,
  "use_sliding_window": true,
  "vocab_size": 248320
}
)JSON";

// `z-lab/Muse-Glimmer-30B-DFlash2` @ `b54ffdd11fa9cfe2af370012e5763d492c904128`,
// config.json VERBATIM (1326 bytes, sha256
// cb684d6f688a22619a63ea1debe7d30c139c195bf3141fd86a763763ab34b5d9 over the
// literal below, read on 2026-08-19 and re-verified 2026-08-20). The SECOND
// published DFlash2 checkpoint, and the one that makes
// #1327 a correction rather than a note: `block_size` 16 against the 27B's 8,
// and `output_multiplier`/`final_logit_softcapping` SET rather than defaulted.
constexpr const char* kMuseGlimmerDflash2Config = R"JSON({
  "architectures": [
    "DFlash2DraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 200000,
  "is_causal": false,
  "dflash_config": {
    "block_size": 16,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "final_logit_softcapping": 20.0,
    "mask_token_id": 201818,
    "output_multiplier": 0.19611613513818404,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [
      1,
      13,
      25,
      37,
      49
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": 200001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 6656,
  "initializer_range": 0.02,
  "intermediate_size": 19968,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 131072,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 52,
  "pad_token_id": 200018,
  "rms_norm_eps": 1e-05,
  "rope_parameters": {
    "rope_theta": 500000.0,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.15.0",
  "use_cache": false,
  "use_sliding_window": true,
  "vocab_size": 202048
}
)JSON";

// `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` @
// `b754e6c86008bdb5cc901308dda5a38173ec7276`, `dflash/config.json` VERBATIM
// (1251 bytes, sha256
// 2ed5a998f5f57e00a9fe14d2b3e767f06e49462a97eb09d80c927e112a585c9e over the
// literal below, re-fetched 2026-08-20). A DFlash1 draft, present here for O4:
// it is the ONLY published draft that declares no `layer_types`, and it is also
// the only one that declares `attention_sink_bias`.
//
// This copy was NOT verbatim when W2 landed. It had been re-indented to two
// spaces and had dropped `auto_map`, so the recorded sha256 hashed a file that
// was not in this repository and nothing here hashed what the test parsed. The
// published bytes are in, and the two facts the case turns on -- no
// `layer_types`, `attention_sink_bias` present -- are unchanged by the repair.
constexpr const char* kMimoDflashConfig = R"JSON({
    "architectures": [
        "DFlashDraftModel"
    ],
    "model_type": "qwen3",
    "auto_map": {
        "AutoModel": "dflash.DFlashDraftModel"
    },
    "hidden_size": 6144,
    "intermediate_size": 16384,
    "num_hidden_layers": 5,
    "num_attention_heads": 128,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "v_head_dim": 128,
    "partial_rotary_factor": 0.5,
    "block_size": 8,
    "dflash_config": {
        "target_layer_ids": [
            0,
            15,
            31,
            47,
            69
        ],
        "mask_token_id": 151669,
        "num_anchors": 4096,
        "block_size": 8,
        "loss_decay_gamma": 7.0,
        "use_swa": true,
        "swa_window_size": 1024,
        "backbone_rotary_base": 5000000,
        "attention_value_scale": 0.612,
        "attention_sink_bias": true
    },
    "num_target_layers": 70,
    "vocab_size": 152064,
    "max_position_embeddings": 262144,
    "rope_theta": 10000,
    "sliding_window": 1024,
    "rms_norm_eps": 1e-05,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "tie_word_embeddings": false,
    "use_cache": true
})JSON";

}  // namespace

TEST_CASE("dflash2 config: the published Qwen3.8-27B DFlash2 config.json PARSES (O3)") {
  const HfConfig c = MakeQwen3DFlashDraftConfig(json::parse(kQwen38Dflash2Config));
  CHECK(c.hidden_size == 5120);
  CHECK(c.num_hidden_layers == 5);
  CHECK(c.vocab_size == 248320);
  // The nested spellings, which the flat reads could not see. `transformers`
  // moved RoPE settings under `rope_parameters`, so this is a FALLBACK and not a
  // replacement: DFlash1 checkpoints still carry the flat `rope_theta`.
  CHECK(c.rope_theta == doctest::Approx(1e7));
  REQUIRE(c.raw.contains("block_size"));
  CHECK(c.raw.at("block_size").get<int64_t>() == 8);
  // The DFlash2 conv geometry the W2 forward needs, carried through dflash_config.
  REQUIRE(c.raw.contains("dflash_config"));
  CHECK(c.raw.at("dflash_config").at("conv_kernel_size").get<int64_t>() == 2);
  CHECK(c.raw.at("dflash_config").at("conv_group_size").get<int64_t>() == 16);
  // W1's rule, now reachable for the first time on a checkpoint that declares it.
  REQUIRE(c.raw.contains("is_causal"));
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 5);
  for (size_t i = 0; i < modes.size(); ++i) {
    CAPTURE(i);
    CHECK_FALSE(modes[i].causal);
    CHECK(modes[i].sliding_window == 2048);
  }
}

TEST_CASE("dflash2 config: the published Muse-Glimmer-30B DFlash2 config.json PARSES (#1327)") {
  const HfConfig c = MakeQwen3DFlashDraftConfig(json::parse(kMuseGlimmerDflash2Config));
  CHECK(c.hidden_size == 6656);
  CHECK(c.rope_theta == doctest::Approx(5e5));
  REQUIRE(c.raw.contains("block_size"));
  CHECK(c.raw.at("block_size").get<int64_t>() == 16);
  // hidden 6656 / conv_group_size 16 = 416 groups, so kernel_projection is
  // 2*taps*num_groups = 1664 wide. The 27B's is 1280. The two shapes are what
  // makes "gate at BOTH published blocks" a real requirement rather than a
  // parameter sweep.
  CHECK(c.raw.at("dflash_config").at("conv_group_size").get<int64_t>() == 16);
  CHECK(c.hidden_size % c.raw.at("dflash_config").at("conv_group_size").get<int64_t>() == 0);
  // The two scalars #1327 corrects the spec about: they are CHECKPOINT-exercised,
  // not synthetic, and both are applied to candidate VALUES before the selector
  // scores them, so a wrong one reorders the top-K and moves acceptance without
  // raising. W3 consumes them; W2 asserts they survive the parse.
  CHECK(c.raw.at("dflash_config").at("output_multiplier").get<double>() ==
        doctest::Approx(0.19611613513818404));
  CHECK(c.raw.at("dflash_config").at("final_logit_softcapping").get<double>() ==
        doctest::Approx(20.0));
}

TEST_CASE("dflash draft config: an ABSENT layer_types is upstream's None (O4)") {
  // `getattr(config, "layer_types", None)` (qwen3_dflash.py:134 and :66 @ the PR
  // head). As shipped this threw `[json.exception.out_of_range.403] key
  // 'layer_types' not found` before any causality could be resolved, which is why
  // #1366's `use_swa` repair was UNREACHED at its own merge commit.
  json doc = json::parse(kMimoDflashConfig);
  doc["dflash_config"].erase("attention_sink_bias");  // refused separately, below
  const HfConfig c = MakeQwen3DFlashDraftConfig(doc);
  CHECK(c.layer_types.empty());
  // Upstream's own `_resolve_layer_attention` docstring row: `layer_types=None`
  // + `use_swa=True` -> SWA window, causal FALSE. Before #1366 this engine
  // answered causal TRUE on every layer.
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 5);
  for (size_t i = 0; i < modes.size(); ++i) {
    CAPTURE(i);
    CHECK_FALSE(modes[i].causal);
    CHECK(modes[i].sliding_window == 1024);
  }
}

TEST_CASE("dflash draft config: attention_sink_bias is REFUSED BY NAME (O4)") {
  // Upstream reads it and passes a per-head sink into its Attention; this lane
  // has none. Parsing the config WITHOUT refusing would convert a loud
  // `key 'layer_types' not found` into a draft that loads with the sinks
  // silently absent -- acceptance-only and invisible to a token gate.
  const json doc = json::parse(kMimoDflashConfig);
  CHECK_THROWS_WITH_AS(MakeQwen3DFlashDraftConfig(doc),
                       doctest::Contains("attention_sink_bias"), std::exception);
  // A FALSE value is upstream's default and is not refused: `dflash_config.get(
  // "attention_sink_bias", ...)` falsy means no sink parameter is created at all.
  json off = doc;
  off["dflash_config"]["attention_sink_bias"] = false;
  CHECK_NOTHROW(MakeQwen3DFlashDraftConfig(off));
}

TEST_CASE("dflash draft config: the DFlash1 flat spellings are UNCHANGED") {
  // The fallbacks must not move a published DFlash1 draft. `z-lab/Qwen3.6-27B-DFlash`
  // declares a flat `rope_theta`, a flat `block_size` and a full `layer_types`,
  // and every one of them must still win.
  json doc = json::object();
  doc["hidden_size"] = 5120;
  doc["num_attention_heads"] = 32;
  doc["num_key_value_heads"] = 8;
  doc["head_dim"] = 128;
  doc["rope_theta"] = 1e7;
  doc["intermediate_size"] = 25600;
  doc["vocab_size"] = 248320;
  doc["num_hidden_layers"] = 2;
  doc["rms_norm_eps"] = 1e-6;
  doc["sliding_window"] = 2048;
  doc["layer_types"] = json::array({"sliding_attention", "full_attention"});
  doc["block_size"] = 17;
  doc["dflash_config"] = json::object();
  doc["dflash_config"]["mask_token_id"] = 248070;
  doc["dflash_config"]["target_layer_ids"] = json::array({5, 19});
  // A nested rope_parameters must NOT override a declared flat rope_theta on a
  // document that carries both: upstream reads one resolved `config.rope_theta`.
  const HfConfig c = MakeQwen3DFlashDraftConfig(doc);
  CHECK(c.rope_theta == doctest::Approx(1e7));
  CHECK(c.raw.at("block_size").get<int64_t>() == 17);
  REQUIRE(c.layer_types.size() == 2);
  CHECK(c.layer_types[0] == "sliding_attention");
  CHECK(c.layer_types[1] == "full_attention");
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 2);
  CHECK(modes[0].causal);
  CHECK_FALSE(modes[1].causal);
}

// ===========================================================================
// PART 2 — the conv INSIDE the draft, loaded by the production weight loader
// off a real on-disk safetensors checkpoint and run by the production forward.
//
// REACHABILITY (.agents/reachability.md). The chain a user arrives through is
//   LoadedEngine::FromModelDir / ResolveSpecConfig
//     -> LoadDflashDraft            (src/vllm/entrypoints/model_loader.cpp)
//       -> vllm::LoadQwen3DFlash    (reads attention_conv/mlp_conv, sets
//                                    conv_taps/conv_group_size; the loader then
//                                    overwrites conv_block_size with 1 + k)
//     -> GPUModelRunner::propose_drafts_block
//       -> Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV
//         -> the layer body -> DflashConvPrepare / DflashConvFinish
//                           -> vt::DFlashGroupedConv
//       -> RefuseDflash2CandidateSelector   (W3's boundary, by name)
//
// The cases below enter at `vllm::LoadQwen3DFlash` over a real safetensors file
// -- the same function the loader calls, on the same tensor names the published
// checkpoint uses -- and then at the model forward, which is what
// `propose_drafts_block` calls. The reachability MUTATION deletes the
// `DflashConvPrepare`/`DflashConvFinish` call sites in the layer body and this
// suite must redden.
//
// WHY AN IDENTITY CONV IS THE RIGHT PROBE. With `taps = 1`, `base_kernel` all
// ones and `kernel_projection` all zeros, the conv is EXACTLY the identity in
// bf16: `bf16(1 + 0) = 1` and `bf16(1 * x) = x`, with no tap to mask. So a
// DFlash2 draft carrying that conv must produce BIT-IDENTICAL logits to the same
// draft with no conv at all. That separates "the conv is wired into the right
// places and perturbs nothing it should not" from "the conv is wired somewhere",
// which a tolerance-based comparison cannot.

namespace {

// Minimal safetensors writer: header length (u64 LE) + header JSON + payload,
// which is the whole format. Mirrors the one in
// tests/vllm/multimodal/ltx2_video_fixture.h rather than adding a dependency on
// it, because that header carries an entire LTX-2.5 fixture with it.
struct StEntry {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<uint16_t> bf16;
};

void WriteSafetensors(const std::vector<StEntry>& entries, const std::string& path) {
  json header = json::object();
  size_t offset = 0;
  for (const StEntry& e : entries) {
    const size_t nbytes = e.bf16.size() * sizeof(uint16_t);
    header[e.name] = {{"dtype", "BF16"},
                      {"shape", e.shape},
                      {"data_offsets", json::array({offset, offset + nbytes})}};
    offset += nbytes;
  }
  const std::string hs = header.dump();
  std::ofstream out(path, std::ios::binary);
  uint64_t hlen = hs.size();
  out.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
  out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  for (const StEntry& e : entries)
    out.write(reinterpret_cast<const char*>(e.bf16.data()),
              static_cast<std::streamsize>(e.bf16.size() * sizeof(uint16_t)));
}

// Deterministic bf16 fill: value(i) = amp * sin(seed + 0.7*i), the same shape of
// generator tests/vllm/models/test_qwen3_dflash_forward.cpp uses.
std::vector<uint16_t> Fill(int64_t n, double seed, double amp) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] =
        vt::F32ToBF16(static_cast<float>(amp * std::sin(seed + 0.7 * static_cast<double>(i))));
  return v;
}

std::vector<uint16_t> Const(int64_t n, float value) {
  return std::vector<uint16_t>(static_cast<size_t>(n), vt::F32ToBF16(value));
}

struct Dims {
  int64_t H = 8, Hq = 2, Hkv = 1, Dh = 4, I = 6, vocab = 8, layers = 2, taps_fc = 2;
  int64_t conv_taps = 0;      // 0 = a DFlash1 checkpoint (no conv tensors at all)
  int64_t conv_group = 4;
  int64_t block = 8;          // the conv's query block, 1 + k
  // Which conv is non-identity, and on which side. An identity conv is
  // taps=1/base=1/projection=0; a non-identity one gets a real projection.
  bool attn_conv_active = false;
  bool mlp_conv_active = false;
  int active_side = -1;       // -1 = both sides active; 0 = prepare only; 1 = finish only
  // `base_kernel[side]`, per side. Both 1.0 with a zero projection is the exact
  // bf16 identity; making the two DIFFER is what separates "the model passes
  // side 0 to prepare and side 1 to finish" from "the model passes a side".
  float base_side0 = 1.0f;
  float base_side1 = 1.0f;
};

// A scratch directory holding one model.safetensors, removed on scope exit.
class ScratchCkpt {
 public:
  explicit ScratchCkpt(const std::vector<StEntry>& entries) {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
           ("vllmcpp_dflash2_ckpt_" + std::to_string(counter++) + "_" +
            std::to_string(static_cast<long long>(::getpid())));
    fs::create_directories(dir_);
    WriteSafetensors(entries, (dir_ / "model.safetensors").string());
  }
  ~ScratchCkpt() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  ScratchCkpt(const ScratchCkpt&) = delete;
  ScratchCkpt& operator=(const ScratchCkpt&) = delete;
  std::string shard() const { return (dir_ / "model.safetensors").string(); }

 private:
  fs::path dir_;
};

std::vector<StEntry> DraftEntries(const Dims& dm) {
  const int64_t qdim = dm.Hq * dm.Dh, kdim = dm.Hkv * dm.Dh;
  std::vector<StEntry> e;
  e.push_back({"embed_tokens.weight", {dm.vocab, dm.H}, Fill(dm.vocab * dm.H, 0.1, 0.3)});
  e.push_back({"fc.weight", {dm.H, dm.H * dm.taps_fc}, Fill(dm.H * dm.H * dm.taps_fc, 0.2, 0.2)});
  e.push_back({"hidden_norm.weight", {dm.H}, Fill(dm.H, 0.3, 0.5)});
  e.push_back({"norm.weight", {dm.H}, Fill(dm.H, 0.4, 0.5)});
  e.push_back({"lm_head.weight", {dm.vocab, dm.H}, Fill(dm.vocab * dm.H, 0.5, 0.3)});
  for (int64_t l = 0; l < dm.layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    const double s = 1.0 + static_cast<double>(l);
    e.push_back({b + "input_layernorm.weight", {dm.H}, Fill(dm.H, s + 0.1, 0.6)});
    e.push_back({b + "post_attention_layernorm.weight", {dm.H}, Fill(dm.H, s + 0.2, 0.6)});
    e.push_back({b + "self_attn.q_proj.weight", {qdim, dm.H}, Fill(qdim * dm.H, s + 0.3, 0.25)});
    e.push_back({b + "self_attn.k_proj.weight", {kdim, dm.H}, Fill(kdim * dm.H, s + 0.4, 0.25)});
    e.push_back({b + "self_attn.v_proj.weight", {kdim, dm.H}, Fill(kdim * dm.H, s + 0.5, 0.25)});
    e.push_back({b + "self_attn.o_proj.weight", {dm.H, qdim}, Fill(dm.H * qdim, s + 0.6, 0.25)});
    e.push_back({b + "self_attn.q_norm.weight", {dm.Dh}, Fill(dm.Dh, s + 0.7, 0.7)});
    e.push_back({b + "self_attn.k_norm.weight", {dm.Dh}, Fill(dm.Dh, s + 0.8, 0.7)});
    e.push_back({b + "mlp.gate_proj.weight", {dm.I, dm.H}, Fill(dm.I * dm.H, s + 0.9, 0.25)});
    e.push_back({b + "mlp.up_proj.weight", {dm.I, dm.H}, Fill(dm.I * dm.H, s + 1.1, 0.25)});
    e.push_back({b + "mlp.down_proj.weight", {dm.H, dm.I}, Fill(dm.H * dm.I, s + 1.2, 0.25)});
    if (dm.conv_taps > 0) {
      const int64_t groups = dm.H / dm.conv_group;
      const int64_t proj_out = 2 * dm.conv_taps * groups;
      for (int which = 0; which < 2; ++which) {
        const std::string cp = b + (which == 0 ? "attention_conv." : "mlp_conv.");
        const bool active = which == 0 ? dm.attn_conv_active : dm.mlp_conv_active;
        // base_kernel is [2 SIDES, taps, H] and is ALL ONES: with a zero
        // projection that is the exact bf16 identity at taps=1, and with a real
        // projection it is upstream's `base + delta` with base 1.
        // base_kernel[side][tap][channel]. Tap 0 carries the side's scale; every
        // HIGHER tap is 1 when this conv is active and ZERO when it is not.
        //
        // The zero matters, and getting it wrong once already produced a gate
        // that could not tell one missing call site from none: with base 1 on
        // every tap and a zero projection, a taps=2 conv is `x[i] + x[i-1]` and
        // NOT the identity, so an "inactive" conv still moved the logits and an
        // "attention_conv only" arm was quietly exercising both convs.
        std::vector<uint16_t> base;
        for (int side = 0; side < 2; ++side) {
          const float scale = side == 0 ? dm.base_side0 : dm.base_side1;
          for (int64_t t = 0; t < dm.conv_taps; ++t) {
            const std::vector<uint16_t> row =
                Const(dm.H, t == 0 ? scale : (active ? 1.0f : 0.0f));
            base.insert(base.end(), row.begin(), row.end());
          }
        }
        e.push_back({cp + "base_kernel", {2, dm.conv_taps, dm.H}, base});
        std::vector<uint16_t> proj(static_cast<size_t>(proj_out * dm.H), vt::F32ToBF16(0.0f));
        if (active) {
          const std::vector<uint16_t> live = Fill(proj_out * dm.H, s + 2.0 + which, 0.4);
          for (int64_t r = 0; r < proj_out; ++r) {
            // Row r of the projection produces coefficient
            // [side = r / (taps*groups)][tap][group]. Zeroing the rows of the
            // side this case does not exercise is what makes "prepare only" and
            // "finish only" separable.
            const int64_t side = r / (dm.conv_taps * groups);
            if (dm.active_side >= 0 && side != dm.active_side) continue;
            for (int64_t c = 0; c < dm.H; ++c)
              proj[static_cast<size_t>(r * dm.H + c)] = live[static_cast<size_t>(r * dm.H + c)];
          }
        }
        e.push_back({cp + "kernel_projection.weight", {proj_out, dm.H}, proj});
      }
    }
  }
  return e;
}

HfConfig DraftConfig(const Dims& dm) {
  HfConfig c;
  c.hidden_size = dm.H;
  c.num_attention_heads = dm.Hq;
  c.num_key_value_heads = dm.Hkv;
  c.head_dim = dm.Dh;
  c.rotary_dim = dm.Dh;
  c.rope_theta = 1e7;
  c.intermediate_size = dm.I;
  c.vocab_size = dm.vocab;
  c.num_hidden_layers = dm.layers;
  c.rms_norm_eps = 1e-6;
  c.sliding_window = 2048;
  c.layer_types = std::vector<std::string>(static_cast<size_t>(dm.layers), "sliding_attention");
  c.raw = nlohmann::json::object();
  c.raw["dflash_config"] = json::object();
  c.raw["dflash_config"]["mask_token_id"] = 7;
  if (dm.conv_taps > 0) {
    c.raw["dflash_config"]["conv_kernel_size"] = dm.conv_taps;
    c.raw["dflash_config"]["conv_group_size"] = dm.conv_group;
    c.raw["dflash_config"]["block_size"] = dm.block;
  }
  c.raw["block_size"] = dm.block;
  c.raw["is_causal"] = false;
  return c;
}

// Load through the PRODUCTION weight loader, off a real safetensors file.
Qwen3DFlashWeights LoadDraft(const ScratchCkpt& ck, const Dims& dm, const HfConfig& c) {
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  Qwen3DFlashWeights w = vllm::LoadQwen3DFlash(shards, c, dm.taps_fc, /*mask_token_id=*/7);
  // What the loader does with the resolved k (src/vllm/entrypoints/model_loader.cpp,
  // LoadDflashDraft): the conv's block is 1 + k and not the checkpoint key.
  if (w.IsDflash2()) w.conv_block_size = dm.block;
  return w;
}

// One draft forward over a single uniform (1+k) query block, through the
// context-free body. `T` is the block, so `cu` is {0, T}.
std::vector<float> Forward(const Qwen3DFlashWeights& w, const HfConfig& c, int64_t T) {
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  return Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, w, c, q);
}

// The same block through the CONTEXT-AWARE body -- the one
// `DflashProposeBlock` and the runner's gathered path call -- with an empty
// context, so the two bodies are exercised on identical inputs.
std::vector<float> ForwardWithContext(const Qwen3DFlashWeights& w, const HfConfig& c, int64_t T) {
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 0};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  return Qwen3DFlashModel::ForwardBlockLogitsWithContext({}, {}, ctx_cu, ids, pos, cu, w, c, q);
}

bool BitEqual(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

TEST_CASE("dflash2 weights: the production loader reads the conv tensors off a real shard") {
  Dims dm;
  dm.conv_taps = 2;
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  CHECK(w.IsDflash2());
  CHECK(w.conv_taps == 2);
  CHECK(w.conv_group_size == 4);
  REQUIRE(w.layers.size() == 2);
  for (size_t l = 0; l < w.layers.size(); ++l) {
    CAPTURE(l);
    CHECK_FALSE(w.layers[l].attention_conv.Empty());
    CHECK_FALSE(w.layers[l].mlp_conv.Empty());
    // [2 SIDES, taps, H] -- dim 0 is prepare/finish, NOT a tap.
    CHECK(w.layers[l].attention_conv.base_kernel.rank == 3);
    CHECK(w.layers[l].attention_conv.base_kernel.shape[0] == 2);
    CHECK(w.layers[l].attention_conv.base_kernel.shape[1] == dm.conv_taps);
    CHECK(w.layers[l].attention_conv.base_kernel.shape[2] == dm.H);
    // 2 * taps * num_groups = 2*2*(8/4) = 8, the shape upstream's
    // kernel_projection produces (1280 at the published 27B's 5120/16).
    CHECK(w.layers[l].mlp_conv.kernel_projection.shape[0] == 2 * dm.conv_taps * (dm.H / dm.conv_group));
    CHECK(w.layers[l].mlp_conv.kernel_projection.shape[1] == dm.H);
  }
}

TEST_CASE("dflash1 weights: a draft with no conv keys stays a DFlash1 draft") {
  // The inertness half. A DFlash1 checkpoint declares no conv key and ships no
  // conv tensor, so `IsDflash2()` is false and every layer body runs the
  // pre-W2 op sequence -- which the identity case below turns into a
  // BIT-IDENTICAL assertion rather than an argument.
  Dims dm;  // conv_taps 0
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  CHECK_FALSE(w.IsDflash2());
  CHECK(w.conv_taps == 0);
  CHECK(w.layers[0].attention_conv.Empty());
  CHECK(w.layers[0].mlp_conv.Empty());
}

TEST_CASE("dflash2 forward: an IDENTITY conv is BIT-IDENTICAL to no conv at all") {
  // taps=1, base_kernel all ones, kernel_projection all zeros. In bf16 that is
  // exactly `out = 1 * x`, with no tap to mask -- so a DFlash2 draft carrying it
  // must reproduce the DFlash1 logits BIT-FOR-BIT. This is what separates
  // "correctly placed and transparent" from "applied somewhere".
  for (int64_t block : {int64_t{8}, int64_t{16}}) {
    CAPTURE(block);
    Dims base;
    base.block = block;
    Dims ident = base;
    ident.conv_taps = 1;
    const ScratchCkpt ck0(DraftEntries(base));
    const ScratchCkpt ck1(DraftEntries(ident));
    const std::vector<float> without = Forward(LoadDraft(ck0, base, DraftConfig(base)),
                                               DraftConfig(base), block);
    const std::vector<float> with = Forward(LoadDraft(ck1, ident, DraftConfig(ident)),
                                            DraftConfig(ident), block);
    REQUIRE(without.size() == with.size());
    CHECK(BitEqual(without, with));
  }
}

TEST_CASE("dflash2 forward: the conv is LOAD-BEARING at block 8 and at block 16") {
  // The reachability assertion. A real (non-identity) conv must move the draft
  // logits at BOTH published block shapes: `z-lab/Qwen3.8-27B-DFlash2` ships
  // block 8 and `z-lab/Muse-Glimmer-30B-DFlash2` ships block 16. Deleting the
  // DflashConvPrepare/DflashConvFinish call sites in the layer body makes each
  // comparison below EQUAL and this case red.
  for (int64_t block : {int64_t{8}, int64_t{16}}) {
    CAPTURE(block);
    Dims base;
    base.block = block;
    Dims live = base;
    live.conv_taps = 2;
    live.attn_conv_active = true;
    live.mlp_conv_active = true;
    const ScratchCkpt ck0(DraftEntries(base));
    const ScratchCkpt ck1(DraftEntries(live));
    const std::vector<float> without = Forward(LoadDraft(ck0, base, DraftConfig(base)),
                                               DraftConfig(base), block);
    const std::vector<float> with = Forward(LoadDraft(ck1, live, DraftConfig(live)),
                                            DraftConfig(live), block);
    REQUIRE(without.size() == with.size());
    CHECK_FALSE(BitEqual(without, with));
    // And the CONTEXT-AWARE body -- a separate layer body, separately wired --
    // must move too. Its DFlash1 output is its own baseline, because the two
    // bodies build the attention differently.
    const std::vector<float> ctx_without =
        ForwardWithContext(LoadDraft(ck0, base, DraftConfig(base)), DraftConfig(base), block);
    const std::vector<float> ctx_with =
        ForwardWithContext(LoadDraft(ck1, live, DraftConfig(live)), DraftConfig(live), block);
    REQUIRE(ctx_without.size() == ctx_with.size());
    CHECK_FALSE(BitEqual(ctx_without, ctx_with));
  }
}

TEST_CASE("dflash2 forward: the prepare side and the finish side land in DIFFERENT places") {
  // `base_kernel` dim 0 is prepare/finish. Driving only side 0 convolves the
  // sublayer INPUT; driving only side 1 convolves its OUTPUT. If the side index
  // were ignored -- or if both calls read the same half -- these two would agree.
  Dims prep;
  prep.conv_taps = 2;
  prep.attn_conv_active = true;
  prep.mlp_conv_active = true;
  prep.active_side = 0;
  Dims fin = prep;
  fin.active_side = 1;
  const ScratchCkpt ck0(DraftEntries(prep));
  const ScratchCkpt ck1(DraftEntries(fin));
  const std::vector<float> a = Forward(LoadDraft(ck0, prep, DraftConfig(prep)),
                                       DraftConfig(prep), prep.block);
  const std::vector<float> b = Forward(LoadDraft(ck1, fin, DraftConfig(fin)),
                                       DraftConfig(fin), fin.block);
  REQUIRE(a.size() == b.size());
  CHECK_FALSE(BitEqual(a, b));
}

TEST_CASE("dflash2 forward: attention_conv and mlp_conv wrap DIFFERENT sublayers") {
  // Two convs, one per sublayer, and upstream builds them with identical
  // arguments -- so nothing but the CALL SITE distinguishes them. Wiring both to
  // the attention (or both to the MLP) would make these two runs agree.
  Dims attn;
  attn.conv_taps = 2;
  attn.attn_conv_active = true;
  Dims mlp = attn;
  mlp.attn_conv_active = false;
  mlp.mlp_conv_active = true;
  const ScratchCkpt ck0(DraftEntries(attn));
  const ScratchCkpt ck1(DraftEntries(mlp));
  const std::vector<float> a = Forward(LoadDraft(ck0, attn, DraftConfig(attn)),
                                       DraftConfig(attn), attn.block);
  const std::vector<float> b = Forward(LoadDraft(ck1, mlp, DraftConfig(mlp)),
                                       DraftConfig(mlp), mlp.block);
  REQUIRE(a.size() == b.size());
  CHECK_FALSE(BitEqual(a, b));
}

TEST_CASE("dflash2 forward: a ragged query block is REFUSED rather than mis-masked") {
  // The conv masks its taps by `row index mod conv_block_size`, which is the
  // intra-block offset only while every request block is contiguous and
  // block-aligned. A ragged batch would mask the WRONG taps and be invisible:
  // the verify is lossless, so only acceptance would move.
  Dims dm;
  dm.conv_taps = 2;
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  std::vector<int32_t> ids(12), pos(12);
  for (int i = 0; i < 12; ++i) {
    ids[static_cast<size_t>(i)] = i % 8;
    pos[static_cast<size_t>(i)] = i;
  }
  const std::vector<int32_t> ragged = {0, 5, 12};  // neither block is 8 rows
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  CHECK_THROWS_WITH_AS(Qwen3DFlashModel::ForwardBlockLogits(ids, pos, ragged, w, c, q),
                       doctest::Contains("conv_block_size"), std::exception);
}

TEST_CASE("dflash2 propose: the conv RUNS and THEN the selector refuses by name") {
  // The ORDER is the claim. `DflashProposeBlock` runs the draft block forward --
  // grouped convolution and all -- BEFORE anything samples. So a DFlash2 draft
  // reaching here has already executed every line of W2, and what it is refused
  // for is the candidate selector alone.
  //
  // Refusing before the forward would satisfy a "DFlash2 is refused" assertion
  // while leaving the whole convolution unreachable from any production entry
  // point, which is what .agents/reachability.md calls the test-only driver.
  // Deleting the `RefuseDflash2CandidateSelector` call inside `DflashProposeBlock`
  // turns this case red.
  //
  // WHAT THIS CASE DOES NOT PROVE, stated because W2's first review found the
  // claim overstated here: `DflashProposeBlock` itself has NO caller outside
  // `tests/` at this commit, so the refusal call this case gates is the
  // test-reachable one. The refusal a user arrives through is the identical call
  // in `GPUModelRunner::propose_drafts_block`, and deleting THAT one leaves every
  // suite in this repository green. It is `## Owed` O7 and it belongs to W4. The
  // CONVOLUTION is a different matter and is genuinely production-reached: see
  // the ForwardBlockLogitsWithDeviceKV cases below, whose call sites redden this
  // suite one at a time.
  Dims dm;
  dm.conv_taps = 2;
  dm.attn_conv_active = true;
  dm.mlp_conv_active = true;
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  REQUIRE(w.IsDflash2());

  const int64_t T = dm.block;  // one request, one (1+k) query block
  std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 0};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::string what;
  try {
    (void)vllm::v1::DflashProposeBlock(w, c, {}, {}, ctx_cu, ids, pos, cu,
                                       /*num_reqs=*/1, /*k=*/static_cast<int>(T - 1), q);
    FAIL("expected the candidate-selector refusal");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("CANDIDATE SELECTOR") != std::string::npos);
  CHECK(what.find("convolution IS implemented") != std::string::npos);
  CHECK(what.find("#1314") != std::string::npos);
}

TEST_CASE("dflash1 propose: a DFlash1 draft still proposes through the same entry") {
  // The instrument's precondition again: the refusal above must be about DFlash2
  // and not about `DflashProposeBlock`. The same call on a DFlash1 draft returns
  // k tokens per request.
  Dims dm;  // conv_taps 0
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  REQUIRE_FALSE(w.IsDflash2());
  const int64_t T = dm.block;
  std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 0};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::v1::DflashProposeResult r = vllm::v1::DflashProposeBlock(
      w, c, {}, {}, ctx_cu, ids, pos, cu, 1, static_cast<int>(T - 1), q);
  REQUIRE(r.draft_token_ids.size() == 1);
  CHECK(r.draft_token_ids[0].size() == static_cast<size_t>(T - 1));
}

TEST_CASE("dflash2 forward: prepare reads base_kernel[0] and finish reads base_kernel[1]") {
  // `base_kernel` dim 0 is the SIDE. With taps=1 and a ZERO projection the conv
  // is exactly `out = base[side] * x` in bf16, so the two sides are separable by
  // a single scalar each and nothing else in the model changes.
  //
  //   A  base = (1, 1)  -> the exact identity
  //   B  base = (1, 2)  -> only the FINISH side scales
  //   C  base = (2, 1)  -> only the PREPARE side scales
  //
  // A model that passed side 0 to both calls would make B the identity and B == A;
  // one that passed side 1 to both would make C the identity. Comparing B and C
  // to A pins BOTH directions, which comparing them only to each other does not.
  Dims a;
  a.conv_taps = 1;
  Dims b = a;
  b.base_side1 = 2.0f;
  Dims cc = a;
  cc.base_side0 = 2.0f;
  const ScratchCkpt cka(DraftEntries(a));
  const ScratchCkpt ckb(DraftEntries(b));
  const ScratchCkpt ckc(DraftEntries(cc));
  const std::vector<float> la = Forward(LoadDraft(cka, a, DraftConfig(a)), DraftConfig(a), a.block);
  const std::vector<float> lb = Forward(LoadDraft(ckb, b, DraftConfig(b)), DraftConfig(b), b.block);
  const std::vector<float> lc = Forward(LoadDraft(ckc, cc, DraftConfig(cc)), DraftConfig(cc), cc.block);
  CHECK_FALSE(BitEqual(la, lb));  // the FINISH side must reach the model
  CHECK_FALSE(BitEqual(la, lc));  // the PREPARE side must reach the model
  CHECK_FALSE(BitEqual(lb, lc));  // and they must land in different places
}

namespace {

// The DEVICE-KV body -- `ForwardBlockLogitsWithDeviceKV`, which is what
// `GPUModelRunner::propose_drafts_block` calls in production -- over one request
// with a two-row context appended through the production append path.
std::vector<float> ForwardDeviceKV(const Qwen3DFlashWeights& w, const HfConfig& c, int64_t T) {
  std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(2 + i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 2};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::vector<float> ctx(static_cast<size_t>(2 * c.hidden_size));
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] = 0.2f * static_cast<float>(std::sin(0.13 * static_cast<double>(i) + 0.4));
  auto store = Qwen3DFlashModel::MakeDeviceKVStore(c, q);
  Qwen3DFlashModel::AppendContextKVDevice(*store, ctx, {0, 1}, w, c, q);
  std::vector<vllm::DflashDeviceKVStore*> stores = {store.get()};
  return Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(stores, ctx_cu, ids, pos, cu, w, c, q);
}

}  // namespace

TEST_CASE("dflash2 forward: EACH conv reaches EACH layer body separately") {
  // Three bodies carry the DFlash draft's layer loop --
  // `ForwardBlockLogits` (context-free), `ForwardWithCtxKVDev` (context-aware,
  // what `DflashProposeBlock` calls) and `ForwardPagedBody` (the paged store,
  // what the runner's production decode path reaches through
  // `ForwardBlockLogitsWithDeviceKV`) -- and each has its OWN four call sites.
  //
  // A case that activates BOTH convs cannot tell one missing call site from
  // none: the other conv still moves the logits and the comparison still passes.
  // That is exactly what a mutation deleting only the context-aware body's
  // attention_conv proved. So each conv is driven ALONE, through each body.
  struct Arm {
    const char* name;
    bool attn;
    bool mlp;
  };
  const Arm arms[] = {{"attention_conv only", true, false}, {"mlp_conv only", false, true}};
  for (const Arm& arm : arms) {
    CAPTURE(arm.name);
    Dims base;
    Dims live = base;
    live.conv_taps = 2;
    live.attn_conv_active = arm.attn;
    live.mlp_conv_active = arm.mlp;
    const ScratchCkpt ck0(DraftEntries(base));
    const ScratchCkpt ck1(DraftEntries(live));
    const Qwen3DFlashWeights w0 = LoadDraft(ck0, base, DraftConfig(base));
    const Qwen3DFlashWeights w1 = LoadDraft(ck1, live, DraftConfig(live));
    const HfConfig c0 = DraftConfig(base);
    const HfConfig c1 = DraftConfig(live);
    CHECK_FALSE(BitEqual(Forward(w0, c0, base.block), Forward(w1, c1, live.block)));
    CHECK_FALSE(BitEqual(ForwardWithContext(w0, c0, base.block),
                         ForwardWithContext(w1, c1, live.block)));
    CHECK_FALSE(BitEqual(ForwardDeviceKV(w0, c0, base.block),
                         ForwardDeviceKV(w1, c1, live.block)));
  }
}
