// SPEC-DFLASH2 W1 (#1314) — the DFlash causality rule, and the top-level
// `is_causal` that has to win before the legacy layer default is consulted.
//
// PORT. vLLM `tests/v1/spec_decode/test_dflash_causality.py` as edited by
// vllm-project/vllm#52816 at head `19c9351904df4c63042671bc67a866ca48dc7d6f`,
// against `vllm/model_executor/models/qwen3_dflash.py:58-67` at that head. The
// upstream file exercises `_dflash_layer_causal` and `dflash_has_any_non_causal`
// off a `SimpleNamespace`; this engine resolves the same decision once per draft
// load in `ResolveQwen3DFlashAttnModes`, so every upstream row is driven through
// that function with the same inputs and the same expectations. The upstream
// rows that only exercise the fc input size and the eagle aux-layer ids belong to
// other functions and are not part of this port.
//
// WHY THE RULE MATTERS HERE, and not only upstream. `z-lab/Qwen3.8-27B-DFlash2`
// declares all five layers `sliding_attention` AND `is_causal false`. Under the
// legacy rule alone — causal iff the layer is `sliding_attention`, unless
// `dflash_config.causal` overrides — every one of those layers runs CAUSAL. The
// draft still emits plausible tokens, a token gate against our own output sees
// nothing, and only ACCEPTANCE moves, which the lossless verify hides. That is
// the single defect in this row that raises nothing
// (.agents/specs/dflash2-spec-decode.md D4).
//
// THE INERTNESS HALF IS AS LOAD-BEARING AS THE NEW HALF. No published DFlash1
// checkpoint carries `is_causal`, so their resolution must come out of this
// change byte-for-byte unchanged; the z-lab 27B DFlash1 shape is asserted below
// for exactly that reason.
#include "vllm/model_executor/models/qwen3_dflash.h"

#include <doctest/doctest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../../gguf_builder.h"

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_dflash_gguf.h"
#include "vllm/transformers_utils/hf_config.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::MakeDflashGgufConfig;
using vllm::MakeQwen3DFlashDraftConfig;
using vllm::Qwen3DFlashLayerAttnMode;
using vllm::ResolveQwen3DFlashAttnModes;

namespace {

// The upstream `_config(...)` helper. `layer_types` empty stands for upstream's
// `None` and for its `[]`; both take the same fallback branch here and upstream
// (`layer_types is None or ...` / `bool(layer_types)`), and the two upstream rows
// that separate them are therefore one row here.
//
// A sliding layer needs a window, or `ResolveQwen3DFlashAttnModes` refuses the
// config outright (qwen3_dflash.py:136-144 @ the PR head). Upstream's
// `SimpleNamespace` never supplies one because its test calls only
// `_dflash_layer_causal`, which does not read it. 2048 is the z-lab 27B value.
HfConfig Config(int64_t num_hidden_layers, const std::vector<std::string>& layer_types,
                const json& dflash_config = json::object(),
                const json& is_causal = json()) {
  HfConfig c;
  c.num_hidden_layers = num_hidden_layers;
  c.head_dim = 128;
  c.sliding_window = 2048;
  c.layer_types = layer_types;
  c.raw = json::object();
  c.raw["dflash_config"] = dflash_config;
  if (!is_causal.is_null()) c.raw["is_causal"] = is_causal;
  return c;
}

json CausalOverride(bool causal) {
  json d = json::object();
  d["causal"] = causal;
  return d;
}

// `dflash_config.use_swa`, which forces SWA onto every layer -- including an
// all-full `layer_types` and an absent one -- and which upstream's causality
// fallback deliberately does NOT read (#1366).
json UseSwa() {
  json d = json::object();
  d["use_swa"] = true;
  return d;
}

// Upstream `dflash_has_any_non_causal` (qwen3_dflash.py:68-74), which is the
// value its parametrized table asserts: whether the draft needs a
// non-causal-capable backend at all.
bool AnyNonCausal(const std::vector<Qwen3DFlashLayerAttnMode>& modes) {
  for (const Qwen3DFlashLayerAttnMode& m : modes) {
    if (!m.causal) return true;
  }
  return false;
}

bool AnyNonCausal(const HfConfig& c) {
  return AnyNonCausal(ResolveQwen3DFlashAttnModes(c));
}

// A minimal `dflash`-arch GGUF drafter carrying only the keys
// MakeDflashGgufConfig reads. `causal` selects whether the DFlash2-only
// `dflash.attention.causal` KV is written at all, which is the GGUF spelling of
// the HF top-level `is_causal`. The sliding-window pattern is all-true, which is
// what the published `z-lab/Qwen3.8-27B-DFlash2-GGUF` carries and is exactly the
// shape under which the legacy rule answers CAUSAL for every layer.
std::string DflashGgufBytes(const std::optional<bool>& causal) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "dflash"));
  b.AddKv(gguf_test::U32Kv("dflash.block_count", 5));
  b.AddKv(gguf_test::U32Kv("dflash.embedding_length", 5120));
  b.AddKv(gguf_test::U32Kv("dflash.feed_forward_length", 17408));
  b.AddKv(gguf_test::U32Kv("dflash.attention.head_count", 32));
  b.AddKv(gguf_test::U32Kv("dflash.attention.head_count_kv", 8));
  b.AddKv(gguf_test::U32Kv("dflash.attention.key_length", 128));
  b.AddKv(gguf_test::F32Kv("dflash.rope.freq_base", 1e7f));
  b.AddKv(gguf_test::F32Kv("dflash.attention.layer_norm_rms_epsilon", 1e-6f));
  b.AddKv(gguf_test::U32Kv("dflash.attention.sliding_window", 2048));
  b.AddKv(gguf_test::BoolArrayKv("dflash.attention.sliding_window_pattern",
                                 {true, true, true, true, true}));
  b.AddKv(gguf_test::U32Kv("dflash.block_size", 8));
  b.AddKv(gguf_test::I32ArrayKv("dflash.target_layers", {6, 20, 34, 48, 62}));
  b.AddKv(gguf_test::U32Kv("tokenizer.ggml.mask_token_id", 248070));
  if (causal.has_value()) {
    b.AddKv(gguf_test::BoolKv("dflash.attention.causal", *causal));
  }
  return b.Build();
}

// The same drafter with `dflash.attention.causal` written as a U32 rather than
// a bool, which is what a converter with no boolean KV type emits. `KvI64`
// already takes it; this fixture is what proves the HF arm agrees (#1366).
std::string DflashGgufBytesNumericCausal(uint32_t causal) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "dflash"));
  b.AddKv(gguf_test::U32Kv("dflash.block_count", 5));
  b.AddKv(gguf_test::U32Kv("dflash.embedding_length", 5120));
  b.AddKv(gguf_test::U32Kv("dflash.feed_forward_length", 17408));
  b.AddKv(gguf_test::U32Kv("dflash.attention.head_count", 32));
  b.AddKv(gguf_test::U32Kv("dflash.attention.head_count_kv", 8));
  b.AddKv(gguf_test::U32Kv("dflash.attention.key_length", 128));
  b.AddKv(gguf_test::F32Kv("dflash.rope.freq_base", 1e7f));
  b.AddKv(gguf_test::F32Kv("dflash.attention.layer_norm_rms_epsilon", 1e-6f));
  b.AddKv(gguf_test::U32Kv("dflash.attention.sliding_window", 2048));
  b.AddKv(gguf_test::BoolArrayKv("dflash.attention.sliding_window_pattern",
                                 {true, true, true, true, true}));
  b.AddKv(gguf_test::U32Kv("dflash.block_size", 8));
  b.AddKv(gguf_test::I32ArrayKv("dflash.target_layers", {6, 20, 34, 48, 62}));
  b.AddKv(gguf_test::U32Kv("tokenizer.ggml.mask_token_id", 248070));
  b.AddKv(gguf_test::U32Kv("dflash.attention.causal", causal));
  return b.Build();
}

}  // namespace

TEST_CASE("dflash causality: the upstream branch table") {
  // Every row of the parametrize list at the PR head, in upstream's order.
  // `dflash_config.causal` forces causality on every layer, ignoring layer_types.
  CHECK_FALSE(AnyNonCausal(
      Config(2, {"full_attention", "full_attention"}, CausalOverride(true))));
  // ... and forces non-causal on every layer.
  CHECK(AnyNonCausal(
      Config(2, {"sliding_attention", "sliding_attention"}, CausalOverride(false))));
  // DFlash2 stores the explicit attention semantics at the TOP level. Both rows
  // are RED under the legacy rule: the first resolves to all-causal and the
  // second to all-non-causal, which is the inverse of what the config says.
  CHECK(AnyNonCausal(Config(2, {"sliding_attention", "sliding_attention"},
                            json::object(), json(false))));
  CHECK_FALSE(AnyNonCausal(
      Config(2, {"full_attention", "full_attention"}, json::object(), json(true))));
  // SWA-derived: a full-attention layer is non-causal.
  CHECK(AnyNonCausal(Config(2, {"sliding_attention", "full_attention"})));
  // SWA-derived: all-sliding is fully causal.
  CHECK_FALSE(AnyNonCausal(Config(2, {"sliding_attention", "sliding_attention"})));
  // No layer_types -> the non-causal fallback.
  CHECK(AnyNonCausal(Config(2, {})));
}

TEST_CASE("dflash causality: resolution is PER LAYER without an override") {
  // Upstream test_dflash_layer_causal_is_per_layer.
  const std::vector<Qwen3DFlashLayerAttnMode> modes =
      ResolveQwen3DFlashAttnModes(Config(2, {"sliding_attention", "full_attention"}));
  REQUIRE(modes.size() == 2);
  CHECK(modes[0].causal);
  CHECK_FALSE(modes[1].causal);
}

TEST_CASE("dflash causality: a top-level is_causal covers EVERY layer") {
  // Upstream test_dflash_layer_causal_honors_top_level_override. The mixed
  // layer_types is the point: the legacy rule answers differently per layer, and
  // the top-level value must flatten both.
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(
      Config(2, {"sliding_attention", "full_attention"}, json::object(), json(false)));
  REQUIRE(modes.size() == 2);
  CHECK_FALSE(modes[0].causal);
  CHECK_FALSE(modes[1].causal);
}

TEST_CASE("dflash causality: is_causal takes precedence over dflash_config.causal") {
  // The ORDER, which no upstream row pins because no upstream row sets both.
  // Upstream reads `is_causal` and RETURNS before it ever looks at the legacy
  // override (qwen3_dflash.py:59-64 @ the PR head), so a port that consulted
  // `dflash_config.causal` first would pass every row above and still be wrong.
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(
      Config(2, {"sliding_attention", "sliding_attention"}, CausalOverride(true),
             json(false)));
  REQUIRE(modes.size() == 2);
  CHECK_FALSE(modes[0].causal);
  CHECK_FALSE(modes[1].causal);
}

TEST_CASE("dflash causality: the published DFlash2 config runs NO layer causal") {
  // `z-lab/Qwen3.8-27B-DFlash2` reduced to the keys the resolution reads: five
  // layers, all `sliding_attention`, `is_causal false`. This is the case the row
  // exists for. Under the legacy rule all five come back CAUSAL, the draft still
  // produces tokens, and only acceptance falls.
  const HfConfig c = Config(5,
                            {"sliding_attention", "sliding_attention",
                             "sliding_attention", "sliding_attention",
                             "sliding_attention"},
                            json::object(), json(false));
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 5);
  for (size_t i = 0; i < modes.size(); ++i) {
    CAPTURE(i);
    CHECK_FALSE(modes[i].causal);
    // The WINDOW is a separate axis and this change must not touch it. Upstream
    // resolves `(sliding_window, causal)` as two independent answers
    // (_resolve_layer_attention), so a resolution that let `is_causal` also zero
    // the window would be wrong in a way no causality assertion could see.
    CHECK(modes[i].sliding_window == 2048);
  }
}

TEST_CASE("dflash causality: a DFlash1 checkpoint resolves EXACTLY as before") {
  // `z-lab/Qwen3.6-27B-DFlash`: four sliding layers plus one full, and no
  // `is_causal` anywhere. Upstream leaves this resolution untouched in the same
  // commit that adds the new precedence, and so must we —
  // .agents/specs/dflash2-spec-decode.md D4 asserts it explicitly.
  const HfConfig c = Config(5, {"sliding_attention", "sliding_attention",
                                "sliding_attention", "sliding_attention",
                                "full_attention"});
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 5);
  for (size_t i = 0; i < 4; ++i) {
    CAPTURE(i);
    CHECK(modes[i].causal);
    CHECK(modes[i].sliding_window == 2048);
  }
  CHECK_FALSE(modes[4].causal);
  CHECK(modes[4].sliding_window == 0);
}

TEST_CASE("dflash causality: use_swa makes a layer SLIDING and never CAUSAL (#1366)") {
  // The two rows of upstream's own `_resolve_layer_attention` docstring table
  // that its parametrize list never exercises (qwen3_dflash.py:109-133 @ the PR
  // head). The fallback resolves causality from the RAW `layer_types` and not
  // from the resolved layer type:
  //
  //     layer_types = getattr(config, "layer_types", None)
  //     return bool(layer_types) and layer_types[layer_idx] == _SLIDING_ATTENTION
  //
  // `use_swa` therefore moves the WINDOW and never the causality. The table
  // states it directly -- `layer_types=None` + `use_swa=True` -> causal False --
  // and names `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` as that shape ("sets
  // use_swa, assumes non-causal".)
  //
  // This is a DFlash1 divergence, found by the fresh reviewer of SPEC-DFLASH2 W1
  // and fixed in flow (#1366). It is the same acceptance-only, token-invisible
  // failure class this row exists to remove, one arm over: such a draft ran
  // every layer CAUSAL here and NON-causal upstream, the verify stayed lossless,
  // and only acceptance moved. Both rows assert the WINDOW too, because the
  // resolved layer type is genuinely sliding in both and a fix that reached the
  // window instead of the causality would be wrong in a way no causality
  // assertion could see.
  SUBCASE("layer_types absent + use_swa -> sliding, and NON-causal") {
    const std::vector<Qwen3DFlashLayerAttnMode> modes =
        ResolveQwen3DFlashAttnModes(Config(2, {}, UseSwa()));
    REQUIRE(modes.size() == 2);
    CHECK_FALSE(modes[0].causal);
    CHECK_FALSE(modes[1].causal);
    CHECK(modes[0].sliding_window == 2048);
    CHECK(modes[1].sliding_window == 2048);
  }

  SUBCASE("all-full layer_types + use_swa -> sliding, and NON-causal") {
    // The same divergence by the other branch of the `is_sliding` rule:
    // `use_swa` forces SWA over an all-full `layer_types`, and `layer_types[i]`
    // is still `full_attention`, so upstream answers non-causal.
    const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(
        Config(2, {"full_attention", "full_attention"}, UseSwa()));
    REQUIRE(modes.size() == 2);
    CHECK_FALSE(modes[0].causal);
    CHECK_FALSE(modes[1].causal);
    CHECK(modes[0].sliding_window == 2048);
    CHECK(modes[1].sliding_window == 2048);
  }

  SUBCASE("use_swa is still OVERRIDDEN by an explicit value") {
    // The fallback is the LAST arm, so neither explicit spelling loses its
    // precedence to this repair.
    CHECK_FALSE(AnyNonCausal(Config(2, {}, UseSwa(), json(true))));
    json d = UseSwa();
    d["causal"] = true;
    CHECK_FALSE(AnyNonCausal(Config(2, {}, d)));
  }
}

TEST_CASE("dflash causality: a NUMERIC is_causal is honoured, as upstream's bool() is (#1366)") {
  // Upstream tests PRESENCE and then coerces -- `if is_causal is not None:
  // return bool(is_causal)` (qwen3_dflash.py:59-61 @ the PR head) -- so a
  // config that writes the key as 0 or 1 is honoured there. Requiring a JSON
  // boolean made `"is_causal": 0` fall through to the legacy rule in SILENCE,
  // which is the same invisible-acceptance shape one more time.
  //
  // It also made the two containers disagree with each other. The GGUF arm
  // reads `dflash.attention.causal` through `KvI64`, which takes every integer
  // width AND bool and names its own error on anything else, so a numeric
  // spelling already worked there. The HF arm now matches it on both halves:
  // numbers are honoured, and a type neither container can coerce is REFUSED by
  // name rather than dropped.
  SUBCASE("is_causal 0 -> non-causal, over an all-sliding layer_types") {
    CHECK(AnyNonCausal(Config(2, {"sliding_attention", "sliding_attention"},
                              json::object(), json(0))));
  }
  SUBCASE("is_causal 1 -> causal, over an all-full layer_types") {
    CHECK_FALSE(AnyNonCausal(
        Config(2, {"full_attention", "full_attention"}, json::object(), json(1))));
  }
  SUBCASE("dflash_config.causal 0 -> the same coercion, one arm down") {
    json d = json::object();
    d["causal"] = 0;
    CHECK(AnyNonCausal(Config(2, {"sliding_attention", "sliding_attention"}, d)));
  }
  SUBCASE("a type NEITHER container can coerce is refused BY NAME") {
    // BY NAME, and asserted as such: a bare CHECK_THROWS passes on the
    // nlohmann `type_error` a coercion attempt raises on its own, so it would
    // hold whether or not this engine refuses anything deliberately. The
    // message has to carry the KEY, which is the whole difference between a
    // refusal and an accident. The GGUF arm's `KvI64` names its key the same
    // way.
    std::string message;
    try {
      ResolveQwen3DFlashAttnModes(
          Config(2, {"full_attention", "full_attention"}, json::object(),
                 json("false")));
      FAIL("an uncoercible is_causal was ACCEPTED");
    } catch (const std::exception& e) {
      message = e.what();
    }
    CHECK(message.find("is_causal") != std::string::npos);
    CHECK(message.find("boolean or a number") != std::string::npos);
  }
  SUBCASE("the two containers agree on a numeric spelling") {
    // HF `is_causal: 0` and GGUF `dflash.attention.causal = 0` (a u32, which is
    // what a converter that has no bool type writes) must reach the same answer.
    const gguf_test::TempFile file(DflashGgufBytesNumericCausal(0));
    const vllm::GgufFile g = vllm::GgufFile::Open(file.path());
    const HfConfig gc = MakeDflashGgufConfig(g);
    REQUIRE(gc.raw.contains("is_causal"));
    const std::vector<Qwen3DFlashLayerAttnMode> gmodes = ResolveQwen3DFlashAttnModes(gc);
    REQUIRE(gmodes.size() == 5);

    const HfConfig hc = Config(5,
                               {"sliding_attention", "sliding_attention",
                                "sliding_attention", "sliding_attention",
                                "sliding_attention"},
                               json::object(), json(0));
    const std::vector<Qwen3DFlashLayerAttnMode> hmodes = ResolveQwen3DFlashAttnModes(hc);
    REQUIRE(hmodes.size() == 5);
    for (size_t i = 0; i < 5; ++i) {
      CAPTURE(i);
      CHECK_FALSE(gmodes[i].causal);
      CHECK(gmodes[i].causal == hmodes[i].causal);
    }
  }
}

TEST_CASE("dflash draft config: a NUMERIC is_causal is CARRIED too (#1366)") {
  // The builder gates the key on the same predicate the resolution does, so a
  // spelling one accepts and the other drops cannot exist.
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
  doc["layer_types"] = json::array({"sliding_attention", "sliding_attention"});
  doc["block_size"] = 8;
  doc["dflash_config"] = json::object();
  doc["dflash_config"]["mask_token_id"] = 248070;
  doc["dflash_config"]["target_layer_ids"] = json::array({5, 19});
  doc["is_causal"] = 0;

  const HfConfig c = MakeQwen3DFlashDraftConfig(doc);
  REQUIRE(c.raw.contains("is_causal"));
  CHECK(AnyNonCausal(c));
}

TEST_CASE("dflash draft config: is_causal is CARRIED off the draft's config.json") {
  // The plumbing half. `MakeQwen3DFlashDraftConfig` is what the loader builds a
  // draft `HfConfig` with (src/vllm/entrypoints/model_loader.cpp, LoadDflashDraft),
  // and it copies named keys rather than the whole document — so a resolution
  // that reads `is_causal` and a builder that drops it agree on nothing. Upstream
  // gets this for free because it reads the key off a HuggingFace config object.
  json doc = json::object();
  doc["hidden_size"] = 5120;
  doc["num_attention_heads"] = 32;
  doc["num_key_value_heads"] = 8;
  doc["head_dim"] = 128;
  doc["rope_theta"] = 1e7;
  doc["intermediate_size"] = 25600;
  doc["vocab_size"] = 248320;
  doc["num_hidden_layers"] = 5;
  doc["rms_norm_eps"] = 1e-6;
  doc["sliding_window"] = 2048;
  doc["layer_types"] = json::array({"sliding_attention", "sliding_attention",
                                    "sliding_attention", "sliding_attention",
                                    "sliding_attention"});
  doc["block_size"] = 8;
  doc["dflash_config"] = json::object();
  doc["dflash_config"]["mask_token_id"] = 248070;
  doc["dflash_config"]["target_layer_ids"] = json::array({5, 19, 33, 47, 61});

  SUBCASE("declared false -> every layer non-causal") {
    doc["is_causal"] = false;
    const HfConfig c = MakeQwen3DFlashDraftConfig(doc);
    REQUIRE(c.raw.contains("is_causal"));
    CHECK(AnyNonCausal(c));
    const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
    REQUIRE(modes.size() == 5);
    for (size_t i = 0; i < modes.size(); ++i) {
      CAPTURE(i);
      CHECK_FALSE(modes[i].causal);
    }
  }

  SUBCASE("absent -> the legacy rule, unchanged") {
    const HfConfig c = MakeQwen3DFlashDraftConfig(doc);
    CHECK_FALSE(c.raw.contains("is_causal"));
    CHECK_FALSE(AnyNonCausal(c));
  }
}

TEST_CASE("dflash causality: a GGUF drafter's attention.causal is READ") {
  // The GGUF axis of the same rule. `z-lab/Qwen3.8-27B-DFlash2-GGUF` @
  // `57ab3265056d4024870b0621cfc2c127537020ed` declares
  // `dflash.attention.causal = false` beside an all-true
  // `dflash.attention.sliding_window_pattern`, so the pattern-derived rule alone
  // answers CAUSAL for all five layers -- the same wrong answer, reached by a
  // different spelling, on a checkpoint that is published and loads today.
  //
  // The KV is the GGUF spelling of the HF top-level `is_causal`, so it resolves
  // in the same precedence and is carried through the same `config.raw` key.
  const gguf_test::TempFile file(DflashGgufBytes(false));
  const vllm::GgufFile g = vllm::GgufFile::Open(file.path());
  const HfConfig c = MakeDflashGgufConfig(g);
  REQUIRE(c.raw.contains("is_causal"));
  CHECK_FALSE(c.raw.at("is_causal").get<bool>());

  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 5);
  for (size_t i = 0; i < modes.size(); ++i) {
    CAPTURE(i);
    CHECK_FALSE(modes[i].causal);
    CHECK(modes[i].sliding_window == 2048);
  }
}

TEST_CASE("dflash causality: a DFlash1 GGUF without the KV is UNCHANGED") {
  // The inertness half, and it is not hypothetical: the shipped DFlash1 drafter
  // `muse-glimmer-30b-gguf/dflash-kquant.gguf` carries no
  // `dflash.attention.causal` and an all-true sliding pattern, so every layer is
  // causal and must stay causal. A resolution that defaulted the absent KV to
  // false would silently flip that drafter to bidirectional.
  const gguf_test::TempFile file(DflashGgufBytes(std::nullopt));
  const vllm::GgufFile g = vllm::GgufFile::Open(file.path());
  const HfConfig c = MakeDflashGgufConfig(g);
  CHECK_FALSE(c.raw.contains("is_causal"));

  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 5);
  for (size_t i = 0; i < modes.size(); ++i) {
    CAPTURE(i);
    CHECK(modes[i].causal);
    CHECK(modes[i].sliding_window == 2048);
  }
}

TEST_CASE("REAL published GGUF drafters resolve causality as they must") {
  // ASSET-GATED, paired with the synthetic cases above for the same reason: the
  // fixture encodes what the artifact was READ to contain, and only the artifact
  // can confirm it. It reports the skip rather than passing silently.
  const char* dflash2 = std::getenv("VLLM_DFLASH2_GGUF_MODEL");
  const char* dflash1 = std::getenv("VLLM_DFLASH_GGUF_MODEL");
  if (dflash2 == nullptr && dflash1 == nullptr) {
    MESSAGE("asset-gated: neither VLLM_DFLASH2_GGUF_MODEL nor "
            "VLLM_DFLASH_GGUF_MODEL is set; 0 real-artifact assertions ran");
    return;
  }
  if (dflash2 != nullptr) {
    const vllm::GgufFile g = vllm::GgufFile::Open(dflash2);
    const HfConfig c = MakeDflashGgufConfig(g);
    REQUIRE(c.raw.contains("is_causal"));
    CHECK_FALSE(c.raw.at("is_causal").get<bool>());
    const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
    REQUIRE(!modes.empty());
    for (size_t i = 0; i < modes.size(); ++i) {
      CAPTURE(i);
      CHECK_FALSE(modes[i].causal);
    }
  }
  if (dflash1 != nullptr) {
    // The shipped drafter must come out of this change unchanged. It declares
    // no `dflash.attention.causal`, so the pattern-derived rule still answers.
    const vllm::GgufFile g = vllm::GgufFile::Open(dflash1);
    const HfConfig c = MakeDflashGgufConfig(g);
    CHECK_FALSE(c.raw.contains("is_causal"));
  }
}
