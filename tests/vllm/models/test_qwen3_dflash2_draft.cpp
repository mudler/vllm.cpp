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
#include "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h"
#include "vllm/model_executor/models/qwen3_dflash2.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "../gguf_builder.h"
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
//       -> vllm::v1::Dflash2SelectCandidates (W3)
//       -> vllm::v1::Dflash2WalkPath -> vt::Dflash2PathWalk (W4) -> the drafts
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
  // SPEC-DFLASH2 W3 (#1314): the candidate selector. Present whenever
  // `conv_taps > 0`, because a DFlash2 checkpoint carries BOTH mechanisms and
  // the loader refuses a draft that declares only one.
  int64_t sel_rank = 4;
  int64_t sel_top_k = 3;
  // The two OUTPUT SCALARS. `z-lab/Qwen3.8-27B-DFlash2` declares NEITHER and
  // takes 1.0 / disabled; `z-lab/Muse-Glimmer-30B-DFlash2` declares
  // 0.19611613513818404 and 20.0 (#1327, `## Risks/decisions` D9). A negative
  // value here means "do not declare the key at all", which is the 27B's shape
  // and the one a defaults-only gate silently measures.
  double output_multiplier = -1.0;
  double final_logit_softcapping = -1.0;
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
  if (dm.conv_taps > 0) {
    // SPEC-DFLASH2 W3 (#1314): the selector's three tensors, under the exact
    // names the published checkpoint stores them under. The two codebooks are
    // filled with DIFFERENT generators, so a load that swapped them (they share
    // a shape) produces different scores rather than the same ones.
    e.push_back({"candidate_selector.hidden_projection.weight", {dm.sel_rank, dm.H},
                 Fill(dm.sel_rank * dm.H, 3.3, 0.35)});
    e.push_back({"candidate_selector.predecessor_codebook", {dm.vocab, dm.sel_rank},
                 Fill(dm.vocab * dm.sel_rank, 4.4, 0.45)});
    e.push_back({"candidate_selector.successor_codebook", {dm.vocab, dm.sel_rank},
                 Fill(dm.vocab * dm.sel_rank, 5.5, 0.55)});
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
    c.raw["dflash_config"]["selector_rank"] = dm.sel_rank;
    c.raw["dflash_config"]["selector_top_k"] = dm.sel_top_k;
    // Declared only when the case asks for it: the 27B draft declares neither,
    // and a gate built from that draft alone measures the default path.
    if (dm.output_multiplier >= 0.0)
      c.raw["dflash_config"]["output_multiplier"] = dm.output_multiplier;
    if (dm.final_logit_softcapping >= 0.0)
      c.raw["dflash_config"]["final_logit_softcapping"] = dm.final_logit_softcapping;
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

// ===========================================================================
// PART 3 — the CANDIDATE SELECTOR (SPEC-DFLASH2 W3, #1314).
//
// REACHABILITY (.agents/reachability.md). The chain a user arrives through is
//   LoadedEngine::FromModelDir / ResolveSpecConfig
//     -> LoadDflashDraft            (src/vllm/entrypoints/model_loader.cpp)
//       -> vllm::LoadQwen3DFlash    (reads candidate_selector.*, sets
//                                    selector_rank/top_k and the two scalars)
//     -> GPUModelRunner::propose_drafts_block
//       -> Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV (+ final_out)
//       -> vllm::v1::Dflash2SelectCandidates
//            -> Qwen3DFlash2Model::ComputeCandidates  -> vt::TopKValuesIndices
//            -> Qwen3DFlash2Model::SelectorEdgeScores -> vt::MatmulBT,
//                                                        vt::Dflash2SelectorEdges
//       -> vllm::v1::Dflash2WalkPath              (W4, and it IS the draft)
//
// The cases below enter at `vllm::LoadQwen3DFlash` over a real safetensors file
// and at `Dflash2SelectCandidates`, which is the SAME function the runner calls
// -- W3 collapses W2's two copies of that step into one, so there is no
// test-only duplicate to gate by mistake.
namespace {

// A DFlash2 draft with a selector, at the two scalar settings the two published
// checkpoints ship.
Dims Dflash2Dims(bool muse_glimmer_scalars) {
  Dims dm;
  dm.conv_taps = 2;
  dm.attn_conv_active = true;
  dm.mlp_conv_active = true;
  if (muse_glimmer_scalars) {
    // `z-lab/Muse-Glimmer-30B-DFlash2`'s own values (#1327).
    dm.output_multiplier = 0.19611613513818404;
    dm.final_logit_softcapping = 20.0;
  }
  return dm;
}

std::string VecStr(const std::vector<int32_t>& v) {
  std::string out;
  for (int32_t x : v) {
    out += std::to_string(x);
    out += ' ';
  }
  return out;
}

// A deterministic synthetic block, so the selector's inputs are chosen by this
// file rather than by whatever the forward happens to produce. `[P*(1+k), W]`.
std::vector<float> Ramp(int64_t rows, int64_t width, double seed) {
  std::vector<float> v(static_cast<size_t>(rows * width));
  for (int64_t i = 0; i < rows * width; ++i)
    v[static_cast<size_t>(i)] =
        static_cast<float>(2.0 * std::sin(seed + 0.37 * static_cast<double>(i)));
  return v;
}

}  // namespace

TEST_CASE("dflash2 weights: the loader does NOT seed conv_block_size from the checkpoint") {
  // SPEC-DFLASH2 W4 (#1314), spec `## Owed` O5 item 1. Through W3
  // `LoadQwen3DFlash` copied `dflash_config.block_size` into `conv_block_size`
  // so a direct caller had "a usable value". That is what made
  // `LoadDflashDraft`'s own `draft->weights.conv_block_size = draft->k + 1;`
  // ungateable: deleting it left the checkpoint's PLAUSIBLE default behind, the
  // conv masked its taps against the wrong block, and the symptom is
  // acceptance-only and token-invisible -- W2 measured every suite staying green
  // under exactly that mutation.
  //
  // The seed is gone. A DFlash2 draft loaded and forwarded WITHOUT the resolved
  // `k` now refuses BY NAME instead of running against the wrong block, so the
  // silent-wrong has no shape left rather than a gate that has to reach an
  // anonymous-namespace function. This case pins that polarity in both
  // directions: unset refuses, set runs.
  const Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  Qwen3DFlashWeights raw = vllm::LoadQwen3DFlash(shards, c, dm.taps_fc,
                                                 /*mask_token_id=*/7);
  REQUIRE(raw.IsDflash2());
  // The precondition that makes this case non-vacuous: the checkpoint DOES carry
  // a block_size, and it is a value the conv could plausibly have used.
  REQUIRE(c.raw.contains("block_size"));
  CHECK(c.raw.at("block_size").get<int64_t>() == dm.block);
  CHECK(raw.conv_block_size == 0);

  const int64_t T = dm.block;
  std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  CHECK_THROWS_WITH_AS(Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, raw, c, q),
                       doctest::Contains("conv_block_size"), std::exception);
  // And with the resolved block -- what `LoadDflashDraft` writes -- it runs.
  raw.conv_block_size = dm.block;
  CHECK_NOTHROW(
      (void)Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, raw, c, q));
  // A DFlash1 draft is untouched: it has no conv, so the field is meaningless
  // and the forward never reads it.
  const Dims one;  // conv_taps 0
  const ScratchCkpt ck1(DraftEntries(one));
  const HfConfig c1 = DraftConfig(one);
  std::vector<vllm::SafetensorsFile> shards1;
  shards1.push_back(vllm::SafetensorsFile::Open(ck1.shard()));
  const Qwen3DFlashWeights w1 = vllm::LoadQwen3DFlash(shards1, c1, one.taps_fc,
                                                      /*mask_token_id=*/7);
  REQUIRE_FALSE(w1.IsDflash2());
  CHECK(w1.conv_block_size == 0);
  CHECK_NOTHROW(
      (void)Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, w1, c1, q));
}

TEST_CASE("dflash2 propose: the conv, the SELECTOR and the WALK all run, and it DRAFTS") {
  // The ORDER is the claim, and as of W4 the END of it is a draft rather than a
  // refusal. `DflashProposeBlock` runs the draft block forward -- grouped
  // convolution and all -- then `Dflash2SelectCandidates`, then
  // `Dflash2WalkPath`, whose output IS the returned draft. A DFlash2 draft
  // reaching here executes every line of W2, W3 and W4.
  //
  // WHY "IT RETURNED TOKENS" IS NOT THE ASSERTION. Deleting the DFlash2 branch
  // entirely also returns tokens -- the DFlash1 per-slot argmax over the block
  // logits -- and they are well-formed, the verify is lossless, and only
  // acceptance falls. That is the whole reason W1, W2 and W3 refused rather than
  // fell back. So this case asserts two things an argmax fallback cannot do:
  //
  //   1. every drafted id is one of the SELECTOR's own candidates for its step.
  //      The candidate set is `selector_top_k` = 3 wide over a vocabulary of 8,
  //      so a full-vocab argmax lands outside it most of the time. (This is a
  //      consistency check against the selector, and its value is exactly that
  //      discrimination, not correctness of the selector itself -- which
  //      test_ops_dflash2_selector_edges.cpp and the D9 cases below carry.)
  //   2. the drafts MOVE when D9's output scalars move. Those scalars touch the
  //      candidate VALUES and nothing else in the engine, so the block forward,
  //      the convolution and the block logits are identical between the two arms
  //      and the DFlash1 argmax would answer identically for both.
  //
  // And structurally: the argmax arm is entered on EMPTINESS and guarded by
  // `RefuseDflash1ArgmaxOnDflash2Block`, so deleting the walk's call site throws
  // by name instead of drafting quietly.
  const Dims plain = Dflash2Dims(false);
  const Dims muse = Dflash2Dims(true);
  const ScratchCkpt ckp(DraftEntries(plain));
  const ScratchCkpt ckm(DraftEntries(muse));
  const HfConfig cp = DraftConfig(plain);
  const HfConfig cm = DraftConfig(muse);
  const Qwen3DFlashWeights wp = LoadDraft(ckp, plain, cp);
  const Qwen3DFlashWeights wm = LoadDraft(ckm, muse, cm);
  REQUIRE(wp.IsDflash2());
  REQUIRE(wm.IsDflash2());

  const int64_t T = plain.block;  // one request, one (1+k) query block
  const int k = static_cast<int>(T - 1);
  std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % cp.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 0};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  const vllm::v1::DflashProposeResult got =
      vllm::v1::DflashProposeBlock(wp, cp, {}, {}, ctx_cu, ids, pos, cu,
                                   /*num_reqs=*/1, k, q);
  REQUIRE(got.draft_token_ids.size() == 1);
  REQUIRE(got.draft_token_ids[0].size() == static_cast<size_t>(k));
  for (int32_t id : got.draft_token_ids[0]) {
    CHECK(id >= 0);
    CHECK(id < static_cast<int32_t>(cp.vocab_size));
  }

  // (1) MEMBERSHIP in the selector's candidate set, step by step. The block
  // logits and hidden come from the SAME forward the propose ran, so the
  // candidate set below is the one the walk chose from.
  std::vector<float> block_hidden;
  const std::vector<float> block_logits = Qwen3DFlashModel::ForwardBlockLogits(
      ids, pos, cu, wp, cp, q, nullptr, &block_hidden);
  const std::vector<int32_t> anchors = {ids[0]};
  const vllm::v1::Dflash2ProposeState sel = vllm::v1::Dflash2SelectCandidates(
      block_logits, block_hidden, anchors, /*num_reqs=*/1, k, wp, cp, q);
  REQUIRE(sel.top_k == 3);
  int outside_top_k_ids = 0;
  for (int step = 0; step < k; ++step) {
    bool member = false;
    for (int64_t c = 0; c < sel.top_k; ++c) {
      if (sel.candidates.ids[static_cast<size_t>(step * sel.top_k + c)] ==
          got.draft_token_ids[0][static_cast<size_t>(step)])
        member = true;
    }
    INFO("step ", step, " drafted ", got.draft_token_ids[0][static_cast<size_t>(step)]);
    CHECK(member);
    // The instrument's precondition: the candidate set must be a PROPER subset
    // of the vocabulary, or "member" is satisfied by everything.
    CHECK(sel.top_k < cp.vocab_size);
    if (got.draft_token_ids[0][static_cast<size_t>(step)] >= sel.top_k)
      ++outside_top_k_ids;
  }
  // ...and the ids are not the identity permutation of the slots, so membership
  // is a statement about token ids rather than about small integers.
  INFO("drafted ids at or above top_k: ", outside_top_k_ids, " of ", k);
  CHECK(outside_top_k_ids > 0);

  // (2) THE DRAFT IS NOT THE DFlash1 ARGMAX'S DRAFT, over the SAME block logits.
  // This is the discriminator the whole case exists for, and it is direct: the
  // fallback the guard protects against is exactly this function, on exactly
  // this input, and its answer is computed here rather than argued about.
  const std::vector<std::vector<int32_t>> argmax_drafts =
      vllm::v1::SampleDflashBlockDrafts(block_logits, /*num_reqs=*/1, k,
                                        wp.draft_vocab_size);
  REQUIRE(argmax_drafts.size() == 1);
  int argmax_agreements = 0;
  for (int step = 0; step < k; ++step)
    if (argmax_drafts[0][static_cast<size_t>(step)] ==
        got.draft_token_ids[0][static_cast<size_t>(step)])
      ++argmax_agreements;
  // MEASURED on 2026-08-20 at this fixture: walk `3 2 3 7 7 7 7` against argmax
  // `2 2 2 7 7 7 7` -- 5 of the 7 slots agree, 2 do not. The margin is logged
  // rather than assumed, because a fixture whose two answers coincided would
  // make this assertion pass for the wrong reason and look identical from here.
  INFO("walk ", VecStr(got.draft_token_ids[0]), " argmax ",
       VecStr(argmax_drafts[0]), " agreeing slots ", argmax_agreements, " of ", k);
  CHECK(argmax_drafts[0] != got.draft_token_ids[0]);

  // (3) The scalars are wired into this entry too. D9's own gate below measures
  // how often they flip the lattice's ORDER (8 of 45 predecessor slots over a
  // sweep of blocks); the walk visits only `k` of those slots per block, so
  // whether a given block's DRAFT moves is a property of the block. Asserted as
  // a MEASURED sweep with the count logged rather than as a single block that
  // might coincide.
  int blocks_whose_draft_moved = 0, blocks_run = 0;
  for (int shift = 0; shift < 6; ++shift) {
    std::vector<int32_t> sids(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i)
      sids[static_cast<size_t>(i)] =
          static_cast<int32_t>((i * 3 + shift) % cp.vocab_size);
    const vllm::v1::DflashProposeResult a = vllm::v1::DflashProposeBlock(
        wp, cp, {}, {}, ctx_cu, sids, pos, cu, /*num_reqs=*/1, k, q);
    const vllm::v1::DflashProposeResult bmuse = vllm::v1::DflashProposeBlock(
        wm, cm, {}, {}, ctx_cu, sids, pos, cu, /*num_reqs=*/1, k, q);
    ++blocks_run;
    if (a.draft_token_ids[0] != bmuse.draft_token_ids[0]) ++blocks_whose_draft_moved;
    // The instrument's own precondition, per block: the same arm against itself
    // must never differ, or the counter above measures nondeterminism.
    const vllm::v1::DflashProposeResult again = vllm::v1::DflashProposeBlock(
        wp, cp, {}, {}, ctx_cu, sids, pos, cu, /*num_reqs=*/1, k, q);
    CHECK(again.draft_token_ids[0] == a.draft_token_ids[0]);
  }
  // MEASURED on 2026-08-20 at this fixture: 5 of 6 blocks moved.
  INFO("blocks whose draft moved with the scalars: ", blocks_whose_draft_moved,
       " of ", blocks_run);
  CHECK(blocks_whose_draft_moved > 0);
  // The precondition for THAT comparison: the block LOGITS are identical between
  // the two arms, so any difference cannot come from the forward. The scalars
  // live only in the config's `dflash_config`; the weight fixtures are generated
  // from the same seeds.
  std::vector<float> muse_hidden;
  const std::vector<float> muse_logits = Qwen3DFlashModel::ForwardBlockLogits(
      ids, pos, cu, wm, cm, q, nullptr, &muse_hidden);
  CHECK(muse_logits == block_logits);
  CHECK(muse_hidden == block_hidden);
}

TEST_CASE("dflash2 weights: the production loader reads the SELECTOR off a real shard") {
  const Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  REQUIRE(w.IsDflash2());
  const vllm::Dflash2SelectorWeights& sel = w.candidate_selector;
  CHECK_FALSE(sel.Empty());
  CHECK(sel.rank == dm.sel_rank);
  CHECK(sel.top_k == dm.sel_top_k);
  // [rank, H] and NOT [H, rank]: on the published 27B rank 256 differs from
  // every other axis, so a transposed load is caught by the shape and not by a
  // wrong answer.
  CHECK(sel.hidden_projection.rank == 2);
  CHECK(sel.hidden_projection.shape[0] == dm.sel_rank);
  CHECK(sel.hidden_projection.shape[1] == dm.H);
  for (const vllm::OwnedTensor* book :
       {&sel.predecessor_codebook, &sel.successor_codebook}) {
    CHECK(book->rank == 2);
    CHECK(book->shape[0] == dm.vocab);
    CHECK(book->shape[1] == dm.sel_rank);
  }
  // WHICH codebook is which, asserted against the fixture's own bytes. The two
  // share a shape, so a loader that swapped them loads cleanly and scores every
  // transition with the roles reversed -- acceptance-only and token-invisible,
  // this row's signature defect. Swapping the two `LoadBf16Direct` calls left
  // every suite in this repository GREEN until this assertion existed; it was
  // found by running the mutation, not by reading the loader.
  {
    const std::vector<uint16_t> want_pred = Fill(dm.vocab * dm.sel_rank, 4.4, 0.45);
    const std::vector<uint16_t> want_succ = Fill(dm.vocab * dm.sel_rank, 5.5, 0.55);
    REQUIRE(want_pred != want_succ);  // the fixture must be able to tell them apart
    REQUIRE(sel.predecessor_codebook.bytes.size() == want_pred.size() * sizeof(uint16_t));
    CHECK(std::memcmp(sel.predecessor_codebook.bytes.data(), want_pred.data(),
                      want_pred.size() * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(sel.successor_codebook.bytes.data(), want_succ.data(),
                      want_succ.size() * sizeof(uint16_t)) == 0);
  }
  // The 27B draft declares NEITHER scalar, so both take upstream's default and
  // the softcap is DISABLED (0 means "no cap", never "cap at zero").
  CHECK(sel.output_multiplier == 1.0f);
  CHECK(sel.final_logit_softcapping == 0.0f);

  // A DFlash1 draft has no selector at all.
  Dims d1;  // conv_taps 0
  const ScratchCkpt ck1(DraftEntries(d1));
  const Qwen3DFlashWeights w1 = LoadDraft(ck1, d1, DraftConfig(d1));
  CHECK(w1.candidate_selector.Empty());
}

TEST_CASE("dflash2 config: a draft declaring the conv keys and NO selector is refused") {
  // Both mechanisms or neither. A checkpoint that declared only the conv would
  // otherwise load with a selector-shaped hole and be discovered at the first
  // propose.
  //
  // THE MESSAGE IS ASSERTED, not just the throw. A bare CHECK_THROWS here passed
  // with the guard REMOVED -- found by the mutation pass, not by reading -- because
  // an absent `selector_rank` leaves `rank` at 0 and the hidden_projection shape
  // check then throws for an unrelated reason. A gate that cannot tell the
  // refusal it means from an incidental one measures nothing.
  Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  HfConfig c = DraftConfig(dm);
  c.raw["dflash_config"].erase("selector_rank");
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  std::string what;
  try {
    (void)vllm::LoadQwen3DFlash(shards, c, dm.taps_fc, /*mask_token_id=*/7);
    FAIL("expected a refusal for conv keys without selector keys");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("selector_rank") != std::string::npos);
  CHECK(what.find("both mechanisms") != std::string::npos);
}

TEST_CASE("dflash2 weights: a MIS-SHAPED selector tensor is refused at load") {
  // The shape assertions are the only thing separating a correct load from a
  // transposed one, and nothing exercised them until the mutation pass removed
  // them and every suite stayed green. On the published 27B the codebooks are
  // [248320, 256] and the projection [256, 5120], so a transposed codebook is a
  // 254 MB tensor that still loads and still produces finite scores.
  Dims dm = Dflash2Dims(false);
  std::vector<StEntry> entries = DraftEntries(dm);
  for (StEntry& e : entries) {
    if (e.name == "candidate_selector.predecessor_codebook") {
      std::swap(e.shape[0], e.shape[1]);  // [rank, vocab] instead of [vocab, rank]
    }
  }
  const ScratchCkpt ck(entries);
  const HfConfig c = DraftConfig(dm);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  std::string what;
  try {
    (void)vllm::LoadQwen3DFlash(shards, c, dm.taps_fc, /*mask_token_id=*/7);
    FAIL("expected a refusal for a transposed codebook");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("codebook must be") != std::string::npos);
}

TEST_CASE("dflash2 selector: Muse Glimmer's OUTPUT SCALARS are read and applied") {
  // `## Risks/decisions` D9. `z-lab/Qwen3.8-27B-DFlash2` sets NEITHER scalar, so
  // a port that reads them with a default passes every gate built from that
  // draft alone -- such a gate measures the default path and reports it as
  // coverage. `z-lab/Muse-Glimmer-30B-DFlash2` sets output_multiplier
  // 0.19611613513818404 and final_logit_softcapping 20.0, and BOTH are applied
  // to the candidate VALUES before the selector scores them, so a wrong value
  // reweights the lattice against the codebook term and moves acceptance without
  // raising anything.
  const Dims plain = Dflash2Dims(false);
  const Dims muse = Dflash2Dims(true);
  const ScratchCkpt ckp(DraftEntries(plain));
  const ScratchCkpt ckm(DraftEntries(muse));
  const HfConfig cp = DraftConfig(plain);
  const HfConfig cm = DraftConfig(muse);
  const Qwen3DFlashWeights wp = LoadDraft(ckp, plain, cp);
  const Qwen3DFlashWeights wm = LoadDraft(ckm, muse, cm);
  CHECK(wp.candidate_selector.output_multiplier == 1.0f);
  CHECK(wp.candidate_selector.final_logit_softcapping == 0.0f);
  CHECK(wm.candidate_selector.output_multiplier == doctest::Approx(0.19611613513818404));
  CHECK(wm.candidate_selector.final_logit_softcapping == 20.0f);

  // The same logits through both. `ComputeCandidates` must return the SAME ids
  // (a positive multiplier and tanh are both monotone, so the top-K SET does not
  // move) and DIFFERENT values.
  const int64_t rows = 4;
  const std::vector<float> logits = Ramp(rows, plain.vocab, 0.9);
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::Dflash2CandidateSet a =
      vllm::Qwen3DFlash2Model::ComputeCandidates(logits, rows, plain.vocab, wp, q);
  const vllm::Dflash2CandidateSet b =
      vllm::Qwen3DFlash2Model::ComputeCandidates(logits, rows, muse.vocab, wm, q);
  REQUIRE(a.ids.size() == b.ids.size());
  CHECK(a.ids == b.ids);
  bool any_value_moved = false;
  for (size_t i = 0; i < a.values.size(); ++i) {
    // Upstream's order, and it is load-bearing: multiply FIRST, then cap the
    // SCALED value. Capping first would cap a differently-scaled number.
    const float want = static_cast<float>(
        std::tanh(static_cast<double>(a.values[i]) * 0.19611613513818404 / 20.0) * 20.0);
    INFO("slot ", i, " default ", a.values[i], " muse ", b.values[i]);
    CHECK(b.values[i] == doctest::Approx(want).epsilon(1e-6));
    any_value_moved = any_value_moved || b.values[i] != a.values[i];
  }
  CHECK(any_value_moved);
  // The default arm is the RAW top-k logit, unscaled and uncapped -- which is
  // what makes "the scalars are applied" separable from "some scalar is applied".
  for (size_t i = 0; i < a.values.size(); ++i) CHECK(std::isfinite(a.values[i]));
  CHECK(a.values != b.values);
}

TEST_CASE("dflash2 selector: the SCALARS reweight the lattice against the codebook term") {
  // The acceptance-moving half of D9, made executable. `output_multiplier` and
  // the softcap scale the UNARY term only; the codebook contraction is
  // untouched. So the two arms do not differ by a monotone rescale of the edge
  // scores -- the ORDER of the children under some predecessor changes, which is
  // precisely what moves which path the W4 walk takes.
  const Dims plain = Dflash2Dims(false);
  const Dims muse = Dflash2Dims(true);
  const ScratchCkpt ckp(DraftEntries(plain));
  const ScratchCkpt ckm(DraftEntries(muse));
  const Qwen3DFlashWeights wp = LoadDraft(ckp, plain, DraftConfig(plain));
  const Qwen3DFlashWeights wm = LoadDraft(ckm, muse, DraftConfig(muse));
  const HfConfig cp = DraftConfig(plain);
  const HfConfig cm = DraftConfig(muse);

  const int P = 1, k = 3;
  const int64_t nq = k + 1;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  const auto count_flips = [](const vllm::v1::Dflash2ProposeState& x,
                              const vllm::v1::Dflash2ProposeState& y) {
    const int64_t K = x.top_k;
    int flips = 0;
    for (int64_t l = 0; l < x.num_steps; ++l) {
      for (int64_t p = 0; p < K; ++p) {
        const size_t base = static_cast<size_t>((l * K + p) * K);
        int64_t ax = 0, ay = 0;
        for (int64_t c = 1; c < K; ++c) {
          if (x.edge_scores[base + static_cast<size_t>(c)] >
              x.edge_scores[base + static_cast<size_t>(ax)])
            ax = c;
          if (y.edge_scores[base + static_cast<size_t>(c)] >
              y.edge_scores[base + static_cast<size_t>(ay)])
            ay = c;
        }
        if (ax != ay) ++flips;
      }
    }
    return flips;
  };
  // SWEEP, and the reason it is a sweep. A single block gives `num_steps * K` =
  // 3 * 3 = 9 predecessor slots to compare, and W3's fresh review measured that
  // exactly ONE of those nine flips. `> 0` on a margin of 1-of-9 is non-vacuous
  // but thin: it is one arithmetic accident away from a gate that passes for the
  // wrong reason, and a reader has no way to see how thin it is. So the same
  // comparison runs over several blocks -- different anchors, different logit and
  // hidden ramps -- and the assertion is on the TOTAL, with the per-block counts
  // logged. The floor is on the aggregate rather than on any one block, because
  // which block flips is a property of the synthetic ramp and not of the port.
  struct Block {
    int32_t anchor;
    double logit_seed, hidden_seed;
  };
  const Block blocks[] = {{2, 0.9, 1.7}, {0, 0.31, 2.9}, {5, 1.55, 0.44},
                          {7, 2.2, 3.1}, {3, 0.05, 1.05}};
  int total_flips = 0, total_slots = 0, blocks_that_flipped = 0;
  for (const Block& blk : blocks) {
    const std::vector<float> bl = Ramp(P * nq, plain.vocab, blk.logit_seed);
    const std::vector<float> bh = Ramp(P * nq, plain.H, blk.hidden_seed);
    const std::vector<int32_t> anchors = {blk.anchor};
    const vllm::v1::Dflash2ProposeState sp = vllm::v1::Dflash2SelectCandidates(
        bl, bh, anchors, P, k, wp, cp, q);
    const vllm::v1::Dflash2ProposeState sm = vllm::v1::Dflash2SelectCandidates(
        bl, bh, anchors, P, k, wm, cm, q);
    REQUIRE(sp.edge_scores.size() == sm.edge_scores.size());
    // The SCORES always move; whether the ORDER moves is the question below.
    CHECK(sp.edge_scores != sm.edge_scores);
    const int flips = count_flips(sp, sm);
    // The per-block INFO stays live for the precondition CHECKs below, which are
    // in this same scope. It used to be followed by `CHECK(flips >= 0)`, which
    // asserted nothing an `int` counter could ever violate and existed only to
    // give the INFO something to attach to; the assertion that carries the block
    // is `count_flips(sp, sp) == 0` below, and the ones that carry the sweep are
    // the aggregate floors after the loop.
    INFO("anchor ", blk.anchor, " argmax flips ", flips, " of ",
         sp.num_steps * sp.top_k);
    total_flips += flips;
    total_slots += static_cast<int>(sp.num_steps * sp.top_k);
    if (flips > 0) ++blocks_that_flipped;
    // THE INSTRUMENT'S OWN PRECONDITION, per block. The counter has to be able
    // to read zero, or a floor on it measures nothing: comparing an arm with
    // ITSELF must flip nothing at all.
    CHECK(count_flips(sp, sp) == 0);
    CHECK(count_flips(sm, sm) == 0);
  }
  INFO("total argmax flips ", total_flips, " over ", total_slots, " slots in ",
       blocks_that_flipped, " of 5 blocks");
  // MEASURED 2026-08-20 on this fixture: 8 flips over 45 slots, in 4 of the 5
  // blocks -- against 1 of 9 in one block before the sweep. The slot count is
  // pinned exactly, because it is a shape and a change to it means the sweep
  // stopped measuring what this comment says. The flip floors are stated as
  // floors well under the measured 8 and 4, so a rounding-level change does not
  // red the suite for a reason that is not a defect, while a port that stopped
  // reordering children altogether still cannot pass.
  CHECK(total_slots == 45);
  CHECK(total_flips >= 3);
  CHECK(blocks_that_flipped >= 2);
}

TEST_CASE("dflash2 selector: the org-vocab shard indices rebase the candidates") {
  // Upstream reads both off `lm_head.shard_indices`: `num_org_vocab_padding`
  // forces that many trailing columns to -inf BEFORE the top-k, and
  // `org_vocab_start_index` is added to every surviving id AFTER it, rebasing a
  // shard's column space into the full vocabulary. BOTH are 0 on every path this
  // engine ships -- the DFlash lane's lm_head is the raw unpadded checkpoint
  // tensor and there is no vocab-parallel sharding -- so they are gated
  // SYNTHETICALLY here rather than claimed as checkpoint coverage, which is the
  // posture `## Upstream chain` already records for the output scalars. An id
  // that is not rebased indexes the wrong codebook row and moves acceptance
  // without raising anything, so the arithmetic has to be right the day a
  // sharded head arrives.
  const Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, DraftConfig(dm));
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t rows = 3;
  const std::vector<float> logits = Ramp(rows, dm.vocab, 0.9);

  const vllm::Dflash2CandidateSet base =
      vllm::Qwen3DFlash2Model::ComputeCandidates(logits, rows, dm.vocab, w, q);
  vllm::Dflash2CandidateArgs shifted;
  shifted.org_vocab_start_index = 5;
  const vllm::Dflash2CandidateSet moved = vllm::Qwen3DFlash2Model::ComputeCandidates(
      logits, rows, dm.vocab, w, q, shifted);
  REQUIRE(base.ids.size() == moved.ids.size());
  for (size_t i = 0; i < base.ids.size(); ++i) {
    INFO("slot ", i);
    CHECK(moved.ids[i] == base.ids[i] + 5);
  }
  // The VALUES are untouched by a rebase, which is what separates "the offset is
  // added to the ids" from "the offset moved the search".
  CHECK(moved.values == base.values);

  // And the padding: the columns it excludes must actually change the answer, or
  // the parameter is being exercised without being tested. `dm.vocab` is 8, so
  // padding 4 leaves columns 0..3.
  vllm::Dflash2CandidateArgs padded;
  padded.num_org_vocab_padding = 4;
  const vllm::Dflash2CandidateSet capped = vllm::Qwen3DFlash2Model::ComputeCandidates(
      logits, rows, dm.vocab, w, q, padded);
  for (int64_t id : capped.ids) CHECK(id < dm.vocab - 4);
  CHECK(capped.ids != base.ids);
}

TEST_CASE("dflash2 selector: the SAMPLE rows are +1..+k, never the anchor row") {
  // Upstream gathers `last_hidden_states[sample_indices]` -- the k MASK
  // positions. Row +0 carries a verified token and predicts nothing. An
  // off-by-one that started at the anchor row would still produce k steps of
  // finite scores, and no token gate could see it.
  //
  // The probe is a block whose rows have DISJOINT argmax columns, so the
  // candidate id of each step names the row it came from.
  const Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, DraftConfig(dm));
  const HfConfig c = DraftConfig(dm);
  const int P = 1, k = 3;
  const int64_t nq = k + 1, V = dm.vocab;
  std::vector<float> block_logits(static_cast<size_t>(P * nq * V), 0.0f);
  for (int64_t r = 0; r < nq; ++r)
    block_logits[static_cast<size_t>(r * V + r)] = 10.0f;  // row r peaks at column r
  const std::vector<float> block_hidden = Ramp(P * nq, dm.H, 1.7);
  const std::vector<int32_t> anchors = {5};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::v1::Dflash2ProposeState st = vllm::v1::Dflash2SelectCandidates(
      block_logits, block_hidden, anchors, P, k, w, c, q);
  REQUIRE(st.num_steps == k);
  const int64_t K = st.top_k;
  for (int64_t j = 0; j < k; ++j) {
    INFO("step ", j);
    // Step j's TOP candidate is column 1+j: the mask row, not the anchor row.
    CHECK(st.candidates.ids[static_cast<size_t>(j * K)] == 1 + j);
  }
  // And column 0 -- the anchor row's peak -- is never a step's top candidate.
  for (int64_t j = 0; j < k; ++j)
    CHECK(st.candidates.ids[static_cast<size_t>(j * K)] != 0);
}

TEST_CASE("dflash2 selector: a DEQUANTIZED target lm_head is REFUSED by name") {
  // Upstream's LM-head guard, in the WIDE form vllm#52883 landed. The selector's
  // whole input is the target head's EXACT top-K, and a GGUF target's
  // `output.weight` reaches this lane DEQUANTIZED to bf16
  // (`LoadGgufSharedEmbedAndHeadBf16`), which is a different candidate set with
  // no visible symptom.
  const Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  Qwen3DFlashWeights w = LoadDraft(ck, dm, DraftConfig(dm));
  CHECK_FALSE(w.lm_head_dequantized);
  CHECK_NOTHROW(vllm::RefuseQuantizedDflash2LmHead(w));

  w.lm_head_dequantized = true;
  std::string what;
  try {
    vllm::RefuseQuantizedDflash2LmHead(w);
    FAIL("expected a refusal for a dequantized target lm_head");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("QUANTIZED") != std::string::npos);
  CHECK(what.find("unquantized target LM head") != std::string::npos);
  CHECK(what.find("UnquantizedLinearMethod") != std::string::npos);
  CHECK(what.find("acceptance falls") != std::string::npos);

  // The guard is on the DFlash2 path only: a DFlash1 draft off a GGUF target is
  // unchanged, and this lane has shipped that combination since SPEC-DFLASH-GGUF.
  Dims d1;  // conv_taps 0
  const ScratchCkpt ck1(DraftEntries(d1));
  Qwen3DFlashWeights w1 = LoadDraft(ck1, d1, DraftConfig(d1));
  w1.lm_head_dequantized = true;
  CHECK_NOTHROW(vllm::RefuseQuantizedDflash2LmHead(w1));

  // And it fires from the production entry, not only when called directly.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<float> logits = Ramp(2, dm.vocab, 0.9);
  CHECK_THROWS(vllm::Qwen3DFlash2Model::ComputeCandidates(logits, 2, dm.vocab, w, q));
}

TEST_CASE("dflash2 selector: input_embedding_scale is REFUSED BY NAME") {
  // The third scalar. Upstream applies it inside `embed_input_ids`; NEITHER
  // published DFlash2 draft declares it, so implementing it would land three
  // unreachable call sites and ignoring it would run a quietly different model
  // on the first checkpoint that sets it. The polarity is W2's for
  // `attention_sink_bias`: upstream's own default (1.0) is a no-op and is not
  // refused. Spec `## Owed` O9.
  Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));

  HfConfig ok = DraftConfig(dm);
  ok.raw["dflash_config"]["input_embedding_scale"] = 1.0;
  CHECK_NOTHROW(vllm::LoadQwen3DFlash(shards, ok, dm.taps_fc, /*mask_token_id=*/7));

  HfConfig bad = DraftConfig(dm);
  bad.raw["dflash_config"]["input_embedding_scale"] = 0.5;
  std::string what;
  try {
    (void)vllm::LoadQwen3DFlash(shards, bad, dm.taps_fc, /*mask_token_id=*/7);
    FAIL("expected a refusal for a declared input_embedding_scale != 1.0");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("input_embedding_scale") != std::string::npos);
  CHECK(what.find("O9") != std::string::npos);
}

TEST_CASE("dflash2 selector: a GGUF target's QUANTIZED lm_head is CARRIED as dequantized") {
  // The other half of D12, and the half a mutation proved was ungated: the guard
  // above can only fire if something SETS `lm_head_dequantized`, and the only
  // thing that can is the GGUF shared-head loader. Making it always report
  // `false` left every suite in this repository green.
  //
  // A GGUF target's `output.weight` is normally block-quantized, and
  // `LoadGgufSharedEmbedAndHeadBf16` dequantizes it to bf16 for the draft. A
  // scalar ggml type packs ONE element per "block"; anything else was
  // dequantized. Both arms are asserted, because a carry that always reported
  // `true` would satisfy the DFlash2 guard while refusing every bf16 GGUF target
  // the DFlash1 lane has shipped since `SPEC-DFLASH-GGUF`.
  constexpr int64_t V = 32, H = 32;  // Q8_0 blocks are 32 elements / 34 bytes
  const std::string bf16_rows(static_cast<size_t>(V * H) * 2, '\x3c');
  const auto build = [&](uint32_t head_type, const std::string& head_data) {
    gguf_test::GgufModelBuilder b;
    b.AddKv(gguf_test::StrKv("general.architecture", "qwen3"));
    b.AddTensor("token_embd.weight", {static_cast<uint64_t>(H), static_cast<uint64_t>(V)},
                /*BF16=*/30, bf16_rows);
    b.AddTensor("output.weight", {static_cast<uint64_t>(H), static_cast<uint64_t>(V)},
                head_type, head_data);
    return b.Build();
  };

  {  // BF16 head: read as dense floats, nothing dequantized.
    const gguf_test::TempFile f(build(30, bf16_rows));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    vllm::OwnedTensor embed, head;
    bool quantized = true;  // deliberately the WRONG default, so a loader that
                            // never writes the flag cannot pass this arm
    vllm::LoadGgufSharedEmbedAndHeadBf16(g, &embed, &head, &quantized);
    CHECK_FALSE(quantized);
  }
  {  // Q8_0 head: 34 bytes per 32-element block, dequantized on the way in.
    const int64_t blocks = V * H / 32;
    std::string q8(static_cast<size_t>(blocks) * 34, '\0');
    for (int64_t i = 0; i < blocks; ++i) {
      q8[static_cast<size_t>(i * 34)] = '\x00';      // scale, fp16 low byte
      q8[static_cast<size_t>(i * 34 + 1)] = '\x3c';  // fp16 1.0
    }
    const gguf_test::TempFile f(build(/*Q8_0=*/8, q8));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    vllm::OwnedTensor embed, head;
    bool quantized = false;
    vllm::LoadGgufSharedEmbedAndHeadBf16(g, &embed, &head, &quantized);
    CHECK(quantized);
  }
}
