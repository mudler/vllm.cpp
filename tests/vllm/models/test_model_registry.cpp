// Ported from:
//   vllm/tests/models/test_registry.py:31-103,151-159
//   vllm/tests/models/test_initialization.py:50-197
// at e24d1b24fe96a56ba8b0d653efa076d03eb95d6c.
// Negative cases additionally pin registry.py:_raise_for_unsupported:1051-1082,
// which has no direct upstream test at this commit.
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/opt.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using vllm::HfConfig;
using vllm::ModelRegistration;
using vllm::ModelRegistry;
using vllm::UnsupportedModelInfo;

namespace {

std::span<const std::string> ArchSpan(const std::vector<std::string>& archs) {
  return std::span<const std::string>(archs);
}

HfConfig Config(std::vector<std::string> architectures) {
  HfConfig config;
  config.architectures = std::move(architectures);
  return config;
}

}  // namespace

TEST_CASE("registry_imports: every registered architecture has a complete factory") {
  const auto registrations = ModelRegistry::Registrations();
  REQUIRE(registrations.size() == 6);

  for (const ModelRegistration& registration : registrations) {
    CAPTURE(registration.architecture);
    REQUIRE(registration.factory != nullptr);
    CHECK(registration.factory->parse_config != nullptr);
    CHECK(registration.factory->load_weights != nullptr);
    CHECK(registration.factory->prepare != nullptr);
    CHECK(registration.factory->forward != nullptr);
    CHECK(registration.factory->make_kv_cache != nullptr);

    const HfConfig config = Config({std::string(registration.architecture)});
    CHECK(&ModelRegistry::Resolve(config) == &registration);
  }
}

TEST_CASE("self_registration: every arch self-registers from its own TU") {
  // With the fixed kRegistrations array replaced by REGISTER_VLLM_MODEL
  // static-init self-registration (qwen3_5_dense.cpp + qwen3_5_moe.cpp each
  // register themselves), the process-global registry must still contain both
  // implemented architectures with a complete factory — proving the static
  // Registrars ran and populated the shared ModelFactory registry.
  const auto registrations = ModelRegistry::Registrations();
  const auto has_arch = [&](std::string_view arch) {
    for (const ModelRegistration& r : registrations) {
      if (r.architecture == arch) return r.factory != nullptr;
    }
    return false;
  };
  CHECK(has_arch("Qwen3ForCausalLM"));
  CHECK(has_arch("Qwen3MoeForCausalLM"));
  // The CROSS-FAMILY canary: OPT is not a Qwen variant at all, and it still
  // needed nothing but its own TU + one REGISTER_VLLM_MODEL line.
  CHECK(has_arch("OPTForCausalLM"));
  CHECK(has_arch("Qwen3_5ForConditionalGeneration"));
  CHECK(has_arch("Qwen3_5MoeForConditionalGeneration"));
  // MLA campaign W7: the DeepSeek-V2 MLA model, likewise one new TU + one
  // REGISTER_VLLM_MODEL line and ZERO edit to a shared array.
  CHECK(has_arch("DeepseekV2ForCausalLM"));

  // Registration arrival order across TUs is unspecified under C++ static init,
  // so the registry imposes a stable canonical sort by architecture name. This
  // makes SupportedArchs()/error-message order deterministic. Byte order puts
  // "Qwen3ForCausalLM" first ('F' (0x46) < 'M' (0x4D) < '_' (0x5F) after the
  // "Qwen3" prefix), then the full-attention MoE "Qwen3MoeForCausalLM", then the
  // two hybrid Qwen3.5 wrappers; resolution is order-independent.
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  REQUIRE(supported.size() == 6);
  CHECK(std::is_sorted(supported.begin(), supported.end()));
  // MLA campaign W7 adds the first non-Qwen/non-OPT family; 'D' sorts first.
  CHECK(supported.front() == "DeepseekV2ForCausalLM");
  CHECK(supported[1] == "OPTForCausalLM");  // 'O' (0x4F) < 'Q' (0x51)
  CHECK(supported[2] == "Qwen3ForCausalLM");
  CHECK(supported[3] == "Qwen3MoeForCausalLM");
  CHECK(supported.back() == "Qwen3_5MoeForConditionalGeneration");

  // The dense/MoE scheduler policy split survives the per-variant TU move.
  std::unique_ptr<vllm::LoadedModel> dense =
      vllm::MakeQwen3_5DenseLoadedModel(vllm::Qwen3_5DenseWeights{});
  CHECK(ModelRegistry::IsDenseModel(*dense));
}

TEST_CASE("registry_model_property: Qwen registrations match pinned _ModelInfo") {
  for (const ModelRegistration& registration : ModelRegistry::Registrations()) {
    CAPTURE(registration.architecture);
    CHECK(registration.info.is_text_generation_model);
    CHECK_FALSE(registration.info.is_pooling_model);
    CHECK_FALSE(registration.info.has_inner_state);
    CHECK(registration.info.score_type == "bi-encoder");
    if (registration.architecture == "Qwen3ForCausalLM" ||
        registration.architecture == "Qwen3MoeForCausalLM" ||
        registration.architecture == "DeepseekV2ForCausalLM" ||
        registration.architecture == "OPTForCausalLM") {
      // Pure text-only full-attention arch (dense OR MoE): NOT hybrid (no GDN),
      // NOT multimodal (no vision tower).
      CHECK_FALSE(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
    } else {
      // The outer Qwen3.5 multimodal wrappers inherit IsHybrid but not
      // HasInnerState; their inner language-model classes carry HasInnerState.
      CHECK(registration.info.is_hybrid);
      CHECK(registration.info.supports_multimodal);
    }
  }
}

TEST_CASE("resolve_model_cls: Qwen3ForCausalLM resolves to the dense additive factory") {
  // W0 of the first additive-model bring-up: the new TU (qwen3_dense.cpp)
  // self-registers "Qwen3ForCausalLM" with ZERO edit to a shared array. It must
  // resolve to a complete factory whose KV-cache builder is present (the runner
  // W1 generalization consumes its full-attention-only KVCacheConfig).
  const HfConfig config = Config({"Qwen3ForCausalLM"});
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "Qwen3ForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  CHECK(reg.factory->is_dense_model);

  // The pure-dense KV-cache spec is full-attention-ONLY: exactly one FA group,
  // NO MambaSpec/GDN group (the topology that forces the runner generalization).
  HfConfig kv_config = config;
  kv_config.num_key_value_heads = 8;
  kv_config.head_dim = 128;
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(kv_config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
}

TEST_CASE("resolve_model_cls: OPTForCausalLM resolves to the cross-family dense factory") {
  // The CROSS-FAMILY additivity canary (registry.py:176). OPT shares no
  // attention preamble, no norm, no activation and no positional scheme with
  // any Qwen variant in the tree — and the model-registry seam still absorbed
  // it as ONE new TU carrying its own REGISTER_VLLM_MODEL line, with ZERO edit
  // to a shared registration array.
  const HfConfig config = Config({"OPTForCausalLM"});
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "OPTForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  CHECK(reg.factory->is_dense_model);

  // Pure full attention: one FA group, no MambaSpec/GDN group. OPT predates
  // GQA, so num_key_value_heads == num_attention_heads (12 x 64 for opt-125m).
  HfConfig kv_config = config;
  kv_config.num_key_value_heads = 12;
  kv_config.head_dim = 64;
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(kv_config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
}

TEST_CASE("parse_config: OPT reads its family fields out of the raw config doc") {
  // OPT names its FFN width `ffn_dim`, not `intermediate_size`, and carries five
  // switches our typed HfConfig does not model — so its config hook reads
  // `HfConfig::raw`. That escape hatch is what let a new family land without
  // widening the shared HfConfig POD (spec "seams that held" S2).
  HfConfig config = Config({"OPTForCausalLM"});
  config.hidden_size = 768;
  config.num_attention_heads = 12;
  config.max_position_embeddings = 2048;
  config.raw = nlohmann::json{{"ffn_dim", 3072},
                              {"word_embed_proj_dim", 768},
                              {"do_layer_norm_before", true},
                              {"activation_function", "relu"}};
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE_NOTHROW(reg.factory->parse_config(config));

  const vllm::OPTConfigExtras extras = vllm::GetOPTConfigExtras(config);
  CHECK(extras.ffn_dim == 3072);
  CHECK(extras.word_embed_proj_dim == 768);
  CHECK(extras.do_layer_norm_before);
  // Keys facebook/opt-125m omits fall back to the transformers OPTConfig
  // defaults, not to zero.
  CHECK(extras.enable_bias);
  CHECK(extras.layer_norm_elementwise_affine);
  CHECK(extras.tie_word_embeddings);
  CHECK_FALSE(extras.remove_final_layer_norm);

  // A missing ffn_dim is a hard error rather than a silently-zero FFN.
  HfConfig bad = config;
  bad.raw.erase("ffn_dim");
  CHECK_THROWS(reg.factory->parse_config(bad));

  // The two upstream variants we have no checkpoint to gate are REJECTED
  // loudly rather than shipped untested (spec R2).
  HfConfig proj = config;
  proj.raw["word_embed_proj_dim"] = 512;
  CHECK_THROWS(reg.factory->parse_config(proj));
  HfConfig act = config;
  act.raw["activation_function"] = "gelu";
  CHECK_THROWS(reg.factory->parse_config(act));
}

TEST_CASE("resolve_model_cls: Qwen3MoeForCausalLM resolves to the full-attention MoE factory") {
  // W0 of the first full-attention MoE bring-up: the new TU
  // (qwen3_moe_registry.cpp) self-registers "Qwen3MoeForCausalLM" with ZERO edit
  // to a shared array. It must resolve to a complete factory whose KV-cache
  // builder is present and, unlike the 35B hybrid MoE, is full-attention ONLY.
  const HfConfig config = Config({"Qwen3MoeForCausalLM"});
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "Qwen3MoeForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  // MoE, not dense: the scheduler-policy split flags it non-dense.
  CHECK_FALSE(reg.factory->is_dense_model);

  // Full-attention MoE KV-cache spec: exactly one FA group, NO MambaSpec/GDN
  // group (Qwen3-Coder has no linear-attention layers — the runner's
  // full-attention-only path covers it with zero runner change).
  HfConfig kv_config = config;
  kv_config.num_key_value_heads = 4;
  kv_config.head_dim = 128;
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(kv_config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
}

TEST_CASE("runtime quant capability distinguishes true W4A4 from dense BF16") {
  vllm::Qwen3_5DenseWeights bf16;
  std::unique_ptr<vllm::LoadedModel> bf16_model =
      vllm::MakeQwen3_5DenseLoadedModel(std::move(bf16));
  CHECK_FALSE(bf16_model->uses_nvfp4_w4a4());

  vllm::Qwen3_5DenseWeights w4a4;
  w4a4.layers.resize(1);
  w4a4.layers.front().mlp.gate_proj_fp4.alpha = 1.0F;
  std::unique_ptr<vllm::LoadedModel> w4a4_model =
      vllm::MakeQwen3_5DenseLoadedModel(std::move(w4a4));
  CHECK(w4a4_model->uses_nvfp4_w4a4());
}

TEST_CASE("safetensors model source propagates the selected load queue") {
  std::vector<vllm::SafetensorsFile> shards;
  vt::Queue queue;
  vllm::ModelSource borrowed =
      vllm::ModelSource::FromSafetensors(shards, &queue);
  CHECK(borrowed.safetensors == &shards);
  CHECK(borrowed.load_queue == &queue);

  auto owned_shards =
      std::make_shared<const std::vector<vllm::SafetensorsFile>>();
  vllm::ModelSource owned =
      vllm::ModelSource::FromSafetensorsOwned(owned_shards, &queue);
  CHECK(owned.safetensors == owned_shards.get());
  CHECK(owned.safetensors_owned == owned_shards);
  CHECK(owned.load_queue == &queue);
}

TEST_CASE("Qwen3.5 KV spec mirrors independent conv and SSM cache dtypes") {
  HfConfig config = Config({"Qwen3_5ForConditionalGeneration"});
  config.num_key_value_heads = 2;
  config.head_dim = 8;
  config.linear_num_key_heads = 2;
  config.linear_num_value_heads = 4;
  config.linear_key_head_dim = 8;
  config.linear_value_head_dim = 8;
  config.linear_conv_kernel_dim = 4;
  config.mamba_ssm_dtype = "float32";

  vllm::Qwen3_5DenseWeights weights;
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::MakeQwen3_5DenseLoadedModel(std::move(weights));
  const vllm::v1::KVCacheConfig kv =
      ModelRegistry::MakeKVCache(*model, config, /*block_size=*/16,
                                 /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 2);
  const auto* mamba = dynamic_cast<const vllm::v1::MambaSpec*>(
      kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(mamba != nullptr);
  REQUIRE(mamba->shapes.size() == 2);
  REQUIRE(mamba->dtypes.size() == 2);

  // Upstream MambaStateShape/DtypeCalculator order is conv, then temporal.
  CHECK(mamba->shapes[0] == std::vector<int64_t>{64, 3});
  CHECK(mamba->shapes[1] == std::vector<int64_t>{4, 8, 8});
  CHECK(mamba->dtypes[0] == vt::DType::kBF16);
  CHECK(mamba->dtypes[1] == vt::DType::kF32);

  // The runner allocates exactly one conv row plus one temporal row per state
  // slot. Keep external-cache page planning byte-identical to that layout.
  const int64_t runtime_row_bytes =
      64 * 3 * static_cast<int64_t>(vt::SizeOf(vt::DType::kBF16)) +
      4 * 8 * 8 * static_cast<int64_t>(vt::SizeOf(vt::DType::kF32));
  CHECK(mamba->page_size_bytes() == runtime_row_bytes);
}

TEST_CASE("Qwen3.5 SSM cache dtype accepts upstream torch aliases exactly") {
  HfConfig config;
  const auto resolve = [&](const char* value) {
    config.mamba_ssm_dtype = value;
    return vllm::detail::ResolveMambaSsmCacheDType(
        config, vt::DType::kBF16);
  };

  CHECK(resolve("") == vt::DType::kBF16);
  CHECK(resolve("auto") == vt::DType::kBF16);
  CHECK(resolve("float32") == vt::DType::kF32);
  CHECK(resolve("float") == vt::DType::kF32);
  CHECK(resolve("float16") == vt::DType::kF16);
  CHECK(resolve("half") == vt::DType::kF16);
  CHECK(resolve("bfloat16") == vt::DType::kBF16);
  CHECK_THROWS_WITH_AS(resolve("fp16"),
                       doctest::Contains("unsupported mamba_ssm_dtype"),
                       std::runtime_error);
}

TEST_CASE("hf_registry_coverage: every registration has an example config fixture") {
  // C++ fixture registry for the currently implemented subset. Keep this list
  // alias-for-alias with the central ordered table, mirroring HF_EXAMPLE_MODELS.
  constexpr std::array<std::string_view, 6> kExampleConfigArchitectures{
      "DeepseekV2ForCausalLM",
      "OPTForCausalLM",
      "Qwen3ForCausalLM",
      "Qwen3MoeForCausalLM",
      "Qwen3_5ForConditionalGeneration",
      "Qwen3_5MoeForConditionalGeneration",
  };
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  REQUIRE(supported.size() == kExampleConfigArchitectures.size());
  for (size_t i = 0; i < supported.size(); ++i) {
    CHECK(supported[i] == kExampleConfigArchitectures[i]);
  }
}

TEST_CASE("resolve_model_cls: architecture list order selects first registered entry") {
  HfConfig config = Config({"NotImplemented", "Qwen3_5MoeForConditionalGeneration",
                            "Qwen3_5ForConditionalGeneration"});
  CHECK(ModelRegistry::Resolve(config).architecture ==
        "Qwen3_5MoeForConditionalGeneration");

  config = Config({"Qwen3_5ForConditionalGeneration",
                   "Qwen3_5MoeForConditionalGeneration"});
  CHECK(ModelRegistry::Resolve(config).architecture ==
        "Qwen3_5ForConditionalGeneration");
}

TEST_CASE("resolve_model_cls: empty architecture list matches pinned oracle") {
  const HfConfig config = Config({});
  CHECK_THROWS_WITH_AS(ModelRegistry::Resolve(config),
                       "No model architectures are specified",
                       std::runtime_error);
}

TEST_CASE("raise_for_unsupported: registered inspection failure exact message") {
  const std::vector<std::string> architectures{
      "Qwen3_5ForConditionalGeneration"};
  const std::vector<std::string_view> supported{
      "Qwen3_5ForConditionalGeneration",
      "Qwen3_5MoeForConditionalGeneration"};
  CHECK_THROWS_WITH_AS(
      ModelRegistry::RaiseForUnsupported(
          ArchSpan(architectures), std::span<const std::string_view>(supported)),
      "Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be "
      "inspected. Please check the logs for more details.",
      std::runtime_error);
}

TEST_CASE("raise_for_unsupported: previously-supported branch exact message") {
  const HfConfig config = Config({"MotifForCausalLM"});
  CHECK_THROWS_WITH_AS(
      ModelRegistry::Resolve(config),
      "Model architecture MotifForCausalLM was supported in vLLM until "
      "v0.10.2, and is not supported anymore. Please use an older version of "
      "vLLM if you want to use this model architecture.",
      std::runtime_error);
}

TEST_CASE("raise_for_unsupported: out-of-tree branch exact message") {
  const HfConfig config = Config({"BartModel"});
  CHECK_THROWS_WITH_AS(
      ModelRegistry::Resolve(config),
      "Model architecture BartModel is not supported in-tree anymore. Please "
      "install the plugin at https://github.com/vllm-project/bart-plugin if "
      "you want to use this model architecture.",
      std::runtime_error);
}

TEST_CASE("raise_for_unsupported: subset default message and order match oracle") {
  const HfConfig unknown = Config({"LlamaForCausalLM"});
  CHECK_THROWS_WITH_AS(
      ModelRegistry::Resolve(unknown),
      "Model architectures ['LlamaForCausalLM'] are not supported for now. "
      "Supported architectures: "
      "dict_keys(['DeepseekV2ForCausalLM', 'OPTForCausalLM', 'Qwen3ForCausalLM', "
      "'Qwen3MoeForCausalLM', "
      "'Qwen3_5ForConditionalGeneration', "
      "'Qwen3_5MoeForConditionalGeneration'])",
      std::runtime_error);

  const HfConfig multiple = Config({"UnknownA", "UnknownB"});
  CHECK_THROWS_WITH_AS(
      ModelRegistry::Resolve(multiple),
      "Model architectures ['UnknownA', 'UnknownB'] are not supported for now. "
      "Supported architectures: "
      "dict_keys(['DeepseekV2ForCausalLM', 'OPTForCausalLM', 'Qwen3ForCausalLM', "
      "'Qwen3MoeForCausalLM', "
      "'Qwen3_5ForConditionalGeneration', "
      "'Qwen3_5MoeForConditionalGeneration'])",
      std::runtime_error);
}

TEST_CASE("previously-supported table is the complete pinned 32-entry table") {
  constexpr std::array<std::pair<std::string_view, std::string_view>, 32>
      kExpected{{
          {"MotifForCausalLM", "0.10.2"},
          {"Phi3SmallForCausalLM", "0.9.2"},
          {"Phi4FlashForCausalLM", "0.10.2"},
          {"Phi4MultimodalForCausalLM", "0.12.0"},
          {"JAISLMHeadModel", "0.22.0"},
          {"ErnieModel", "0.23.0"},
          {"ErnieForSequenceClassification", "0.23.0"},
          {"ErnieForTokenClassification", "0.23.0"},
          {"InternLM2VEForCausalLM", "0.23.0"},
          {"QWenLMHeadModel", "0.23.0"},
          {"QwenVLForConditionalGeneration", "0.23.0"},
          {"InternLMForCausalLM", "0.23.0"},
          {"DonutForConditionalGeneration", "0.10.2"},
          {"MllamaForConditionalGeneration", "0.10.2"},
          {"XverseForCausalLM", "0.23.0"},
          {"Dots1ForCausalLM", "0.23.0"},
          {"BambaForCausalLM", "0.23.0"},
          {"MiniMaxForCausalLM", "0.23.0"},
          {"MiniMaxText01ForCausalLM", "0.23.0"},
          {"MiniMaxM1ForCausalLM", "0.23.0"},
          {"MiniMaxVL01ForConditionalGeneration", "0.23.0"},
          {"BaiChuanForCausalLM", "0.23.0"},
          {"BaichuanForCausalLM", "0.23.0"},
          {"AquilaModel", "0.24.0"},
          {"AquilaForCausalLM", "0.24.0"},
          {"Grok1ModelForCausalLM", "0.24.0"},
          {"Grok1ForCausalLM", "0.24.0"},
          {"TarsierForConditionalGeneration", "0.24.0"},
          {"Tarsier2ForConditionalGeneration", "0.23.0"},
          {"MantisForConditionalGeneration", "0.24.0"},
          {"MusicFlamingoForConditionalGeneration", "0.24.0"},
          {"AyaVisionForConditionalGeneration", "0.24.0"},
      }};
  const std::span<const UnsupportedModelInfo> actual =
      ModelRegistry::PreviouslySupportedModels();
  REQUIRE(actual.size() == kExpected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    CHECK(actual[i].architecture == kExpected[i].first);
    CHECK(actual[i].detail == kExpected[i].second);
  }
}

TEST_CASE("out-of-tree table is the complete pinned four-entry table") {
  constexpr std::array<std::string_view, 4> kExpected{
      "BartModel", "BartForConditionalGeneration",
      "Florence2ForConditionalGeneration", "MBartForConditionalGeneration"};
  const std::span<const UnsupportedModelInfo> actual =
      ModelRegistry::OutOfTreeSupportedModels();
  REQUIRE(actual.size() == kExpected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    CHECK(actual[i].architecture == kExpected[i]);
    CHECK(actual[i].detail ==
          "https://github.com/vllm-project/bart-plugin");
  }
}

TEST_CASE("can_initialize SKIP: MODEL-FACTORY-registry: no second family yet" *
          doctest::skip()) {
  // Ported test_initialization.py:50-197. The registry seam and both existing
  // Qwen variants are exercised by the loaded-engine CPU suites; a genuinely
  // distinct second-family dummy initialization belongs to the Llama leaf.
  MESSAGE("SKIP MODEL-FACTORY-registry: no second family yet");
}
