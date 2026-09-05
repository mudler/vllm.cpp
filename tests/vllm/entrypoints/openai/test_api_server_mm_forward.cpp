// ENG-MM-INPUT-PIPELINE P2 (#2379) — THE SERVED IMAGE REQUEST, end to end.
//
// One OpenAI `image_url` chat request travels the whole production chain on a
// CPU queue over the real serving stack:
//
//   ApiServer::handle_chat_completions
//     -> OpenAIServingChat (the multimodal chat seam: marker injection,
//        tokenize, Qwen3VLImageProcessor, placeholder EXPANSION, mm_features)
//     -> AsyncLLM::generate(MultiModalInputs)   -> EngineCore
//     -> Scheduler::schedule                    (encoder admission + budget)
//     -> Executor -> GPUModelRunner::execute_model
//        (the encoder step runs the VISION TOWER, the gather slices its rows,
//         the model's embed hook merges them, `.mm` is set)
//     -> ModelRegistry::Forward  -> ForwardQwen3VLForConditionalGeneration
//
// WHY THIS FILE EXISTS RATHER THAN A UNIT TEST. Every piece of the multimodal
// path already had unit coverage before this row and the capability was still
// unreachable: `EncoderCacheManager` was fully ported and constructed nowhere,
// and a grep over the 4600-line runner for `mm_features|MultiModalForwardInput|
// \.mm = ` returned zero. A test that constructs the runner's mm state by hand
// would have passed against that tree. This one cannot: it enters through the
// HTTP dispatch and asserts the tower ran.
//
// THE WEIGHTS ARE SYNTHETIC AND THE TOKENS ARE NOT CHECKED. There is no golden
// here and there cannot be one on a random tiny checkpoint; the token-exact
// closing gate against real Qwen3-VL-4B weights needs a GPU and is owned by
// `MM-SERVE-E2E`. What this file gates is REACHABILITY and the shape of what
// flows: which stage ran, on how many rows, and that a text request through the
// same server is untouched.
//
// THE SEAM IS INSTALLED BY THE PRODUCTION FUNCTION, NOT BY THIS FILE (#2475).
// Until ENG-MM-INPUT-PIPELINE's dispatch wave this harness called
// `oai::MakeQwen3VLImageChatFn` itself, which is exactly why #2408 item 5 could
// delete the server's `set_multimodal_chat_fn(...)` and watch this suite stay
// green: the test was re-implementing the thing it was supposed to gate. It now
// calls `oai::InstallMultiModalChatSeam` — the one function `server_main.cpp`
// calls — against a temporary model directory carrying real
// `preprocessor_config.json` and `config.json` files, so the processor is
// loaded from disk on the production path rather than handed in pre-built.
//
// The tiny vision tower is built directly rather than loaded, because
// `LoadQwen3VLWeights` hard-codes the 4B tower geometry (hidden 1024, depth 24,
// 2304 position embeddings, ~300M parameters) and a unit test cannot synthesise
// a checkpoint for it. `Qwen3VLVisionConfig` is a plain struct and
// `MakeQwen3VLLoadedModel` takes in-memory weights, so the geometry is chosen
// here and the tower is the SAME `Qwen3VLVisionForward` production runs.
#include "vllm/entrypoints/openai/api_server.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/multimodal.h"
#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"
#include "vllm/multimodal/processing/context.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/dtype.h"

namespace oai = vllm::entrypoints::openai;
using nlohmann::json;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::SchedulerConfig;
using oai::ApiServer;
using oai::ChatMessage;
using oai::OpenAIServingChat;
using oai::OpenAIServingCompletion;
using oai::OpenAIServingModels;
using vllm::tok::Tokenizer;
using vllm::v1::AsyncLLM;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::GPUModelRunner;
using vllm::v1::init_none_hash;
using vllm::v1::InputProcessor;
using vllm::v1::KVCacheConfig;
using vllm::v1::OutputProcessor;
using vllm::v1::Scheduler;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

// ─── Geometry ───────────────────────────────────────────────────────────────
// head_dim is 128 because `MropeArgs` (qwen3_vl.cpp:157) hard-codes
// `mrope_section = {24, 20, 20}`, which sums to 64 == head_dim / 2. A narrower
// head would need a config-driven section split, which is a different change.
constexpr int64_t kHidden = 128, kInter = 64, kLayers = 2;
// The vocabulary is DENSE and the config's `vocab_size` equals it exactly: the
// logprobs path detokenizes the top-k alternatives, so any id the sampler can
// pick has to have a token. A `vocab_size` larger than the tokenizer's table
// answers a `top_logprobs` request with "tokenizer: unknown token id 24".
constexpr int64_t kHeads = 2, kKvHeads = 1, kHeadDim = 128, kVocab = 17;
constexpr int kBlockSize = 16, kNumBlocks = 128, kMaxModelLen = 256;

// The vision tokens, as ADDED tokens in the fixture tokenizer. Their ids are the
// processor's `image_token_id` / `vision_{start,end}_token_id`, which is what
// makes the marker string the chat seam injects tokenize to exactly one
// placeholder id that `ExpandImagePlaceholders` can expand.
constexpr int32_t kVisionStartId = 14, kImagePadId = 15, kVisionEndId = 16;

// A 64x64 RGB image over a 16-pixel patch and a 2x2 spatial merge: grid (1,4,4)
// = 16 patches, and 16 / (2*2) = FOUR placeholder tokens. Four rather than one,
// so a mis-sized masked scatter is visible.
constexpr int64_t kImageSide = 64;
constexpr int64_t kExpectedImageTokens = 4;

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  t.bytes.resize(static_cast<size_t>(n) * 2);
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (int64_t i = 0; i < n; ++i)
    p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  return t;
}

std::vector<uint16_t> Bf16Vec(int64_t n, uint64_t seed) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  return v;
}
std::vector<float> F32Vec(int64_t n, uint64_t seed) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = RandV(seed + static_cast<uint64_t>(i));
  return v;
}

HfConfig MakeConfig() {
  HfConfig c;
  c.architectures = {"Qwen3VLForConditionalGeneration"};
  c.hidden_size = kHidden;
  c.intermediate_size = kInter;
  c.num_hidden_layers = kLayers;
  c.num_attention_heads = kHeads;
  c.num_key_value_heads = kKvHeads;
  c.head_dim = kHeadDim;
  c.vocab_size = kVocab;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 5e6;
  c.max_position_embeddings = kMaxModelLen;
  c.raw = json::object();
  return c;
}

vllm::Qwen3DenseWeights MakeTextWeights(const HfConfig& c) {
  vllm::Qwen3DenseWeights w;
  // Qwen3-VL-4B ties its head (qwen3_vl.cpp:128), and the VL forward applies the
  // tied projection as `embed_tokens^T` — so an lm_head here would be dead.
  w.tie_word_embeddings = true;
  w.attention_bias = false;
  w.embed_tokens = MakeOwned(DType::kBF16, {c.vocab_size, kHidden}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {kHidden}, 12);
  for (int64_t l = 0; l < kLayers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3DenseLayerWeights lw;
    lw.input_layernorm = MakeOwned(DType::kBF16, {kHidden}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {kHidden}, s + 2);
    lw.attn.qkv_proj = MakeOwned(
        DType::kBF16, {kHeads * kHeadDim + 2 * kKvHeads * kHeadDim, kHidden}, s + 10);
    lw.attn.o_proj = MakeOwned(DType::kBF16, {kHidden, kHeads * kHeadDim}, s + 20);
    lw.attn.q_norm = MakeOwned(DType::kBF16, {kHeadDim}, s + 30);
    lw.attn.k_norm = MakeOwned(DType::kBF16, {kHeadDim}, s + 40);
    lw.mlp.gate_up_proj = MakeOwned(DType::kBF16, {2 * kInter, kHidden}, s + 50);
    lw.mlp.down_proj = MakeOwned(DType::kBF16, {kHidden, kInter}, s + 60);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

vllm::multimodal::Qwen3VLVisionConfig MakeVisionConfig() {
  vllm::multimodal::Qwen3VLVisionConfig v;
  v.hidden_size = 16;
  v.num_heads = 2;
  v.depth = 2;
  v.intermediate_size = 32;
  v.out_hidden_size = kHidden;   // the tower projects into the TEXT hidden space
  v.patch_size = 16;
  v.temporal_patch_size = 2;
  v.spatial_merge_size = 2;
  v.num_position_embeddings = 16;  // 4x4 grid side, exactly the fixture grid
  v.in_channels = 3;
  v.deepstack_visual_indexes = {0};  // ONE level, so the DeepStack half is live
  v.norm_eps = 1e-6f;
  return v;
}

vllm::multimodal::VisionMergerWeights MakeMerger(
    const vllm::multimodal::Qwen3VLVisionConfig& v, bool postshuffle,
    uint64_t seed) {
  const int64_t ctx = v.hidden_size;
  const int64_t four = 4 * ctx;  // merge_unit() * context_dim
  vllm::multimodal::VisionMergerWeights m;
  m.use_postshuffle_norm = postshuffle;
  const int64_t norm_dim = postshuffle ? four : ctx;
  m.norm_w = Bf16Vec(norm_dim, seed + 1);
  m.norm_b = Bf16Vec(norm_dim, seed + 2);
  m.fc1_w = Bf16Vec(four * four, seed + 3);
  m.fc1_b = Bf16Vec(four, seed + 4);
  m.fc2_w = Bf16Vec(v.out_hidden_size * four, seed + 5);
  m.fc2_b = Bf16Vec(v.out_hidden_size, seed + 6);
  return m;
}

vllm::multimodal::Qwen3VLVisionWeights MakeVisionWeights(
    const vllm::multimodal::Qwen3VLVisionConfig& v) {
  vllm::multimodal::Qwen3VLVisionWeights w;
  const int64_t patch_in = v.in_channels * v.temporal_patch_size * v.patch_size *
                           v.patch_size;
  w.patch_proj_w = Bf16Vec(v.hidden_size * patch_in, 5000);
  w.patch_proj_b = Bf16Vec(v.hidden_size, 5001);
  // f32 DELIBERATELY: the position table is interpolated on the host in f32
  // (see the field note in qwen3_vl_vision.h).
  w.pos_embed_w = F32Vec(v.num_position_embeddings * v.hidden_size, 5002);
  for (int64_t b = 0; b < v.depth; ++b) {
    const uint64_t s = 6000 + static_cast<uint64_t>(b) * 900;
    vllm::multimodal::VisionBlockWeights blk;
    blk.norm1_w = Bf16Vec(v.hidden_size, s + 1);
    blk.norm1_b = Bf16Vec(v.hidden_size, s + 2);
    blk.norm2_w = Bf16Vec(v.hidden_size, s + 3);
    blk.norm2_b = Bf16Vec(v.hidden_size, s + 4);
    blk.qkv_w = Bf16Vec(3 * v.hidden_size * v.hidden_size, s + 5);
    blk.qkv_b = Bf16Vec(3 * v.hidden_size, s + 6);
    blk.proj_w = Bf16Vec(v.hidden_size * v.hidden_size, s + 7);
    blk.proj_b = Bf16Vec(v.hidden_size, s + 8);
    blk.fc1_w = Bf16Vec(v.intermediate_size * v.hidden_size, s + 9);
    blk.fc1_b = Bf16Vec(v.intermediate_size, s + 10);
    blk.fc2_w = Bf16Vec(v.hidden_size * v.intermediate_size, s + 11);
    blk.fc2_b = Bf16Vec(v.hidden_size, s + 12);
    w.blocks.push_back(std::move(blk));
  }
  w.merger = MakeMerger(v, /*postshuffle=*/false, 8000);
  for (size_t i = 0; i < v.deepstack_visual_indexes.size(); ++i) {
    // The DeepStack mergers use the POSTSHUFFLE norm (qwen3_vl.py) — the
    // asymmetry with the main merger above is real and load-bearing.
    w.deepstack_mergers.push_back(
        MakeMerger(v, /*postshuffle=*/true, 9000 + 100 * static_cast<uint64_t>(i)));
  }
  return w;
}

vllm::Qwen3VLWeights MakeVlWeights(const HfConfig& c) {
  vllm::Qwen3VLWeights w;
  w.text = MakeTextWeights(c);
  w.vision_cfg = MakeVisionConfig();
  w.vision = MakeVisionWeights(w.vision_cfg);
  w.vision_loaded = true;
  w.vision_skipped = false;
  return w;
}

KVCacheConfig MakeKvConfig(const HfConfig& c) {
  // The same single full-attention group MakeQwen3VLForConditionalGenerationKVCache
  // publishes for this architecture.
  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<FullAttentionSpec>(kBlockSize,
                                          static_cast<int>(c.num_key_value_heads),
                                          static_cast<int>(c.head_dim),
                                          DType::kBF16));
  return kv;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// A BPE fixture whose ADDED tokens are the three vision markers, so the string
// the chat seam injects tokenizes to exactly [30, 31, 32].
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_mmfwd_tok_" + std::to_string(counter++) + ".json")).string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", kVisionStartId}, {"content", "<|vision_start|>"}, {"special", true}},
       {{"id", kImagePadId}, {"content", "<|image_pad|>"}, {"special", true}},
       {{"id", kVisionEndId}, {"content", "<|vision_end|>"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  // The same Split + ByteLevel sequence the other fixtures use; the loader
  // requires exactly one Split pre-tokenizer.
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},   {"l", 2},    {"o", 3},    {"w", 4},
                {"r", 5},   {"d", 6},   {"Ġ", 7},    {"1", 8},    {"2", 9},
                {"ll", 10}, {"he", 11}, {"llo", 12}, {"hello", 13}};
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array({json::array({"l", "l"}),
                                          json::array({"h", "e"}),
                                          json::array({"ll", "o"}),
                                          json::array({"he", "llo"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

// The chat prompt seam: concatenate the rendered contents. The mm seam has
// already replaced each image part with the marker string, so this is where the
// markers enter the prompt.
std::string ConcatChatPrompt(
    const std::vector<ChatMessage>& messages, bool,
    const std::vector<oai::ChatCompletionToolsParam>&,
    const nlohmann::ordered_json&) {
  std::string p;
  for (const ChatMessage& m : messages)
    if (m.content.has_value()) p += *m.content;
  return p;
}

vllm::multimodal::Qwen3VLProcessorConfig MakeProcessorConfig() {
  vllm::multimodal::Qwen3VLProcessorConfig p;
  p.patch_size = 16;
  p.temporal_patch_size = 2;
  p.merge_size = 2;
  // The 4B defaults are 65536..16777216 pixels, which would force the fixture
  // image up to 256x256 and a 196 KB base64 body for no extra coverage.
  p.min_pixels = 1024;
  p.max_pixels = 1 << 24;
  p.image_token_id = kImagePadId;
  p.vision_start_token_id = kVisionStartId;
  p.vision_end_token_id = kVisionEndId;
  p.model_id = "tiny-qwen3-vl";
  return p;
}

// The raw-RGB passthrough codec, the same shape server_main.cpp installs: a
// square HxWx3 buffer straight through, a container format refused by name.
oai::ImageCodecFn RawRgbCodec() {
  return [](const oai::DecodedMedia& media) -> oai::DecodedImageRgb {
    REQUIRE(media.media_type == "image/x-raw-rgb");
    const std::size_t px = media.bytes.size() / 3;
    const auto side = static_cast<int64_t>(std::llround(std::sqrt(
        static_cast<double>(px))));
    REQUIRE(static_cast<std::size_t>(side * side * 3) == media.bytes.size());
    oai::DecodedImageRgb out;
    out.rgb = media.bytes;
    out.height = side;
    out.width = side;
    return out;
  };
}

// ─── The model DIRECTORY the production factory reads ───────────────────────
//
// `MakeQwen3VLChatSeam` loads its processor config from
// `<model_dir>/preprocessor_config.json` + `<model_dir>/config.json`, exactly as
// the server does. These two documents reproduce `MakeProcessorConfig()`'s
// geometry on disk, so the seam the production install builds is the same
// processor the hand-wired harness used to pass in. `config.json` deliberately
// carries NO `vision_config`: the loader lets that sub-key override merge and
// patch size (qwen3vl_processor.cpp:53-59), and the fixture's values live in
// `preprocessor_config.json`.
json Qwen3VLPreprocessorJson() {
  return json{{"patch_size", 16},
              {"temporal_patch_size", 2},
              {"merge_size", 2},
              {"size", {{"shortest_edge", 1024}, {"longest_edge", 1 << 24}}}};
}

json Qwen3VLConfigJson() {
  return json{{"image_token_id", kImagePadId},
              {"vision_start_token_id", kVisionStartId},
              {"vision_end_token_id", kVisionEndId}};
}

// A throwaway checkpoint directory holding those two documents. Removed on
// destruction so a ctest run leaves nothing behind.
struct TempModelDir {
  std::filesystem::path path;

  TempModelDir() {
    static int counter = 0;
    static const unsigned salt = std::random_device{}();
    path = std::filesystem::temp_directory_path() /
           ("vllm_mmchat_model_" + std::to_string(salt) + "_" +
            std::to_string(counter++));
    std::filesystem::create_directories(path);
    std::ofstream(path / "preprocessor_config.json", std::ios::binary)
        << Qwen3VLPreprocessorJson().dump();
    std::ofstream(path / "config.json", std::ios::binary)
        << Qwen3VLConfigJson().dump();
  }
  ~TempModelDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  TempModelDir(const TempModelDir&) = delete;
  TempModelDir& operator=(const TempModelDir&) = delete;

  std::string dir() const { return path.string(); }
  std::string config_path() const { return (path / "config.json").string(); }
};

// The architecture the model registry resolves this fixture's weights to, and
// the one Qwen3-VL's chat factory is registered under.
constexpr const char* kQwen3VLArch = "Qwen3VLForConditionalGeneration";
// An architecture NOTHING registers a chat seam for. It stands in for the next
// multimodal model to reach this seam before its own factory exists; the cases
// below assert its premise rather than assume it.
constexpr const char* kUnregisteredMmArch = "UnregisteredMmForConditionalGeneration";
// An architecture whose factory IS registered and returns an EMPTY seam. It
// stands in for the mistake a second architecture's author makes once: build
// the processor, fill in `detail` and `allowed_limits`, and forget `chat_fn`.
// See the registration below `}  // namespace`.
constexpr const char* kEmptySeamMmArch = "EmptySeamMmForConditionalGeneration";

std::string EncodeBase64(const std::vector<uint8_t>& raw) {
  static const char* kAlpha =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  size_t i = 0;
  for (; i + 2 < raw.size(); i += 3) {
    const uint32_t v = (uint32_t(raw[i]) << 16) | (uint32_t(raw[i + 1]) << 8) |
                       uint32_t(raw[i + 2]);
    out += kAlpha[(v >> 18) & 63];
    out += kAlpha[(v >> 12) & 63];
    out += kAlpha[(v >> 6) & 63];
    out += kAlpha[v & 63];
  }
  if (i + 1 == raw.size()) {
    const uint32_t v = uint32_t(raw[i]) << 16;
    out += kAlpha[(v >> 18) & 63];
    out += kAlpha[(v >> 12) & 63];
    out += "==";
  } else if (i + 2 == raw.size()) {
    const uint32_t v = (uint32_t(raw[i]) << 16) | (uint32_t(raw[i + 1]) << 8);
    out += kAlpha[(v >> 18) & 63];
    out += kAlpha[(v >> 12) & 63];
    out += kAlpha[(v >> 6) & 63];
    out += '=';
  }
  return out;
}

std::string ImageDataUri(int variant = 0) {
  std::vector<uint8_t> rgb(static_cast<size_t>(kImageSide * kImageSide * 3));
  if (variant == 0) {
    for (size_t i = 0; i < rgb.size(); ++i)
      rgb[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
  } else {
    // A DIFFERENT image, not a shifted one: a smooth vertical ramp against the
    // first one's high-frequency sawtooth, so the two disagree in every patch
    // rather than in a phase.
    for (size_t i = 0; i < rgb.size(); ++i)
      rgb[i] = static_cast<uint8_t>((i / (kImageSide * 3)) * 4 & 0xFF);
  }
  return "data:image/x-raw-rgb;base64," + EncodeBase64(rgb);
}

std::string ChatBodyWithImage(int max_tokens, int variant = 0) {
  const json body = {
      {"model", "test-model"},
      {"messages",
       json::array({{{"role", "user"},
                     {"content",
                      json::array({{{"type", "image_url"},
                                    {"image_url",
                                     {{"url", ImageDataUri(variant)}}}},
                                   {{"type", "text"}, {"text", "hello"}}})}}})},
      {"max_completion_tokens", max_tokens},
      {"temperature", 0.0}};
  return body.dump();
}

std::string CompletionText(const ApiServer::DispatchResult& r) {
  const json j = json::parse(r.body);
  return j.at("choices").at(0).at("message").at("content").get<std::string>();
}

// The same body, plus per-token logprobs. Comparing the LOGPROBS rather than the
// sampled text is what makes the "different images give a different forward"
// case an instrument instead of a coin flip: on a random tiny checkpoint the
// argmax over 64 vocabulary entries is saturated and does not move for a small
// change in the hidden state, while the float logprobs move for ANY change.
std::string ChatBodyWithImageLogprobs(int max_tokens, int variant) {
  const json body = {
      {"model", "test-model"},
      {"messages",
       json::array({{{"role", "user"},
                     {"content",
                      json::array({{{"type", "image_url"},
                                    {"image_url",
                                     {{"url", ImageDataUri(variant)}}}},
                                   {{"type", "text"}, {"text", "hello"}}})}}})},
      {"max_completion_tokens", max_tokens},
      {"logprobs", true},
      {"top_logprobs", 3},
      {"temperature", 0.0}};
  return body.dump();
}

// The whole production serving stack over the tiny Qwen3-VL, on a CPU queue.
// Structurally identical to `ServerHarness` in test_api_server.cpp; it differs
// only in taking a registry `LoadedModel` (the generic runner constructor)
// instead of concrete Qwen3.5-MoE weights, because the multimodal seam lives on
// the REGISTRATION.
struct MmServerHarness {
  MmServerHarness(const HfConfig& c, vllm::LoadedModel& model, const Tokenizer& tok)
      : scheduler(MakeSchedulerConfig(), MakeKvConfig(c), kBlockSize,
                  /*enable_caching=*/true),
        // ONE request per step: the registered Qwen3-VL forward returns only the
        // last token's logits and refuses a batched step by name (see
        // qwen3_vl_registry.cpp). The server config below matches.
        runner(c, model, MakeKvConfig(c), Q(), /*max_num_reqs=*/1, kMaxModelLen,
               kMaxModelLen * 4),
        executor(runner),
        input_processor(tok, c),
        output_processor(&tok),
        async_engine(input_processor, scheduler, executor, output_processor,
                     Hasher()),
        models("test-model"),
        completion(async_engine, "test-model"),
        chat(async_engine, "test-model", &ConcatChatPrompt, "hermes"),
        server(completion, chat, models, "9.9.9") {}

  static SchedulerConfig MakeSchedulerConfig() {
    SchedulerConfig cfg;
    cfg.max_num_seqs = 1;
    cfg.max_num_batched_tokens = kMaxModelLen * 4;
    // Chunked prefill is ON, exactly as the production server runs it — but it
    // never CHUNKS here, and this harness therefore does NOT gate the
    // straddling-item clamp. The budget above is kMaxModelLen * 4 == 1024
    // against a prompt of seven tokens, so every step covers the whole prompt
    // in one piece and the clamp is never reached: deleting the budget/cache
    // clamp in `Scheduler` leaves every case in this file green. What gates the
    // clamp is test_scheduler's pair of truncation cases ("an unschedulable
    // encoder input TRUNCATES the chunk before its span" and "the encoder
    // CACHE, not only the budget, truncates"). The flag stays ON because it is
    // the production value, not because this file measures it.
    cfg.enable_chunked_prefill = true;
    cfg.max_model_len = kMaxModelLen;
    cfg.watermark = 0.0;
    cfg.max_num_encoder_input_tokens = kMaxModelLen * 4;
    cfg.encoder_cache_size = kMaxModelLen * 4;
    return cfg;
  }
  static vllm::v1::BlockHasher Hasher() {
    static bool init = false;
    if (!init) {
      init_none_hash(sha256_cbor);
      init = true;
    }
    return get_request_block_hasher(kBlockSize, sha256_cbor);
  }

  // THE PRODUCTION INSTALL, not a copy of it. `server_main.cpp` builds the same
  // context and calls the same function; what it adds on top is reading the
  // architecture and the multimodal declaration off `LoadedEngine` and pointing
  // `model_dir` at the real checkpoint. Anything this method proves about the
  // dispatch, the server gets for free. What it does NOT reach is that addition
  // in `main`: the eight-field context assembly plus `LoadedEngine::
  // architecture()` and `is_multimodal_model()`. Both remain mutable to green,
  // and `## Risks` in `.agents/specs/mm-chat-seam-registry.md` records the two
  // mutations that prove it and why a server-binary test cannot close it here.
  //
  // `log` is captured rather than sent to stderr so a case can assert on WHAT
  // the install announced, which is the only externally visible difference
  // between the installed and the refusing arm at startup.
  oai::MultiModalChatInstall install_multimodal_seam(
      std::string_view architecture, bool is_multimodal_model,
      const TempModelDir& model, std::ostream& log) {
    oai::MultiModalChatContext ctx;
    ctx.architecture = architecture;
    ctx.model_dir = model.dir();
    ctx.config_path = model.config_path();
    ctx.served_model_name = "tiny-qwen3-vl";
    ctx.tokenizer = &Fixture();
    ctx.prompt_fn = &ConcatChatPrompt;
    ctx.codec = RawRgbCodec();
    ctx.mm_config = &mm_cfg;
    return oai::InstallMultiModalChatSeam(chat, is_multimodal_model, ctx, log);
  }

  // Declared FIRST so it outlives `chat`: the seam's `BaseProcessingInfo` holds
  // it by reference (context.h:105), exactly as the engine's own config is held
  // in production.
  vllm::MultiModalConfig mm_cfg;
  Scheduler scheduler;
  GPUModelRunner runner;
  Executor executor;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  AsyncLLM async_engine;
  OpenAIServingModels models;
  OpenAIServingCompletion completion;
  OpenAIServingChat chat;
  ApiServer server;
};

}  // namespace

// ─── A REGISTERED FACTORY THAT PRODUCES NOTHING ─────────────────────────────
// The null `make_seam` POINTER is checked; the empty `chat_fn` it can RETURN is
// the same hole one step later, and it fails silently rather than loudly.
// `chat.set_multimodal_chat_fn(<empty std::function>)` leaves
// `serving_chat.cpp:699`'s `if (mm_chat_fn_)` false, so the image request is
// answered from the TEXT path while the install reports success and logs "seam
// wired for architecture 'X'". #2475's defect, restored under a success log.
//
// Registered through the production macro from this translation unit, exactly
// as an architecture registers its real one, so the case below enters
// `InstallMultiModalChatSeam` through the same dispatch the server uses rather
// than by handing it a hand-built seam.
namespace vllm::entrypoints::openai {
namespace {
MultiModalChatSeam MakeEmptyMultiModalChatSeam(const MultiModalChatContext&) {
  MultiModalChatSeam seam;
  // Everything EXCEPT the one field that matters, which is what makes this the
  // realistic mistake rather than a contrived one: the install has plenty to
  // print and still nothing to call.
  seam.allowed_limits["image"] = 1;
  seam.detail = "a factory that filled in everything but chat_fn";
  return seam;
}
}  // namespace
REGISTER_VLLM_MM_CHAT(empty_seam_fixture, kEmptySeamMmArch,
                      &MakeEmptyMultiModalChatSeam)
}  // namespace vllm::entrypoints::openai

// ---------------------------------------------------------------------------
// 1. THE REGISTRATION declares the seam. Without these three hooks
//    `ModelRegistry::SupportsMmInputs` is false and the runner's whole
//    multimodal arm is never entered, whatever the scheduler sends.
// ---------------------------------------------------------------------------
TEST_CASE("mm forward: the Qwen3-VL registration declares the encode/embed/mrope hooks") {
  // The vector is named rather than braced inline: `Resolve` is overloaded on
  // HfConfig too, and a braced initializer picks that one.
  const std::vector<std::string> arch = {"Qwen3VLForConditionalGeneration"};
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(arch);
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.info.supports_multimodal);
  CHECK(reg.factory->encode_mm != nullptr);
  CHECK(reg.factory->embed_mm != nullptr);
  CHECK(reg.factory->mrope_prompt_positions != nullptr);

  // ...and the predicate the runner reads is derived from them, not stored.
  const HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeVlWeights(c);
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  CHECK(vllm::ModelRegistry::SupportsMmInputs(*model));
  CHECK(vllm::ModelRegistry::UsesMrope(*model));
}

// ---------------------------------------------------------------------------
// 1b. THE CHAT SEAM IS REGISTERED BY ARCHITECTURE (#2475). The serving half of
//     the same split: the model registration says the model can EMBED image
//     features, this registration says the server can BUILD them from an
//     `image_url` part. Both are keyed on the architecture, and an architecture
//     nobody registered resolves to NOTHING rather than to the first entry.
// ---------------------------------------------------------------------------
TEST_CASE("mm chat registry: the seam is keyed on the architecture, and an unregistered one resolves to nothing") {
  const oai::MultiModalChatRegistration* qwen =
      oai::MultiModalChatRegistry::Find(kQwen3VLArch);
  REQUIRE(qwen != nullptr);
  CHECK(qwen->architecture == kQwen3VLArch);
  CHECK(qwen->make_seam != nullptr);

  // The premise of the refusal cases below, asserted rather than assumed.
  CHECK(oai::MultiModalChatRegistry::Find(kUnregisteredMmArch) == nullptr);

  // The registration is reached through the static library's --whole-archive,
  // so a link that dropped `mm_chat_qwen3vl.cpp` reads as an EMPTY registry
  // rather than as a subtly wrong one.
  const std::vector<std::string_view> archs =
      oai::MultiModalChatRegistry::SupportedArchs();
  CHECK(std::find(archs.begin(), archs.end(), std::string_view(kQwen3VLArch)) !=
        archs.end());

  // The on-disk fixture reproduces the geometry the rest of this file assumes.
  // The production factory loads its processor from those two files, so a
  // directory that loaded to a DIFFERENT config would move
  // `kExpectedImageTokens` and quietly make every case below measure something
  // else.
  const TempModelDir model_dir;
  const vllm::multimodal::Qwen3VLProcessorConfig on_disk =
      vllm::multimodal::LoadQwen3VLProcessorConfig(
          (model_dir.path / "preprocessor_config.json").string(),
          model_dir.config_path(), "tiny-qwen3-vl");
  const vllm::multimodal::Qwen3VLProcessorConfig in_memory =
      MakeProcessorConfig();
  CHECK(on_disk.patch_size == in_memory.patch_size);
  CHECK(on_disk.temporal_patch_size == in_memory.temporal_patch_size);
  CHECK(on_disk.merge_size == in_memory.merge_size);
  CHECK(on_disk.min_pixels == in_memory.min_pixels);
  CHECK(on_disk.max_pixels == in_memory.max_pixels);
  CHECK(on_disk.image_token_id == in_memory.image_token_id);
  CHECK(on_disk.vision_start_token_id == in_memory.vision_start_token_id);
  CHECK(on_disk.vision_end_token_id == in_memory.vision_end_token_id);
  CHECK(on_disk.model_id == in_memory.model_id);

  // And the dispatch refuses BY NAME rather than substituting a processor,
  // which is the whole of #2475 (mirrors registry.py:182-185).
  oai::MultiModalChatContext ctx;
  ctx.architecture = kUnregisteredMmArch;
  bool threw = false;
  try {
    (void)oai::MultiModalChatRegistry::MakeSeam(ctx);
  } catch (const std::exception& e) {
    threw = true;
    const std::string what = e.what();
    INFO("what: ", what);
    CHECK(what.find(kUnregisteredMmArch) != std::string::npos);
    CHECK(what.find("no multimodal CHAT seam is registered") !=
          std::string::npos);
  }
  CHECK(threw);
}

// ---------------------------------------------------------------------------
// 2. THE SERVED REQUEST. One image chat request, through HTTP dispatch.
// ---------------------------------------------------------------------------
TEST_CASE("mm forward: a served image chat request reaches the model forward") {
  const HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeVlWeights(c);
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  vt::Queue q = Q();
  vllm::ModelRegistry::Prepare(*model, c, q);

  const TempModelDir model_dir;
  MmServerHarness h(c, *model, Fixture());
  std::ostringstream install_log;
  REQUIRE(h.install_multimodal_seam(kQwen3VLArch, /*is_multimodal_model=*/true,
                                    model_dir, install_log) ==
          oai::MultiModalChatInstall::kInstalled);

  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(ChatBodyWithImage(/*max_tokens=*/3));

  // A 500 here is the interesting failure, and it is what this whole row is
  // about: before the runner carried mm, `ForwardQwen3VL` refused every step
  // with "requires multimodal inputs (ModelForwardInput.mm)".
  INFO("body: ", r.body);
  REQUIRE(r.status == 200);
  const json j = json::parse(r.body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("usage").at("completion_tokens") == 3);
  // The prompt the engine actually ran is the EXPANDED one: vision_start + FOUR
  // image tokens + vision_end + "hello". A seam that dropped the expansion would
  // report 3 prompt tokens and still answer 200.
  CHECK(j.at("usage").at("prompt_tokens") == 3 + kExpectedImageTokens);
}

// ---------------------------------------------------------------------------
// 3. THE TEXT PATH on the SAME server is unchanged — and the M-RoPE-only
//    Qwen3-VL forward refuses a step with no multimodal request, which is what
//    makes the runner's "any request in the batch carries mm_features"
//    predicate load-bearing rather than an optimisation.
// ---------------------------------------------------------------------------
TEST_CASE("mm forward: two image requests on one server both answer, and share nothing wrong") {
  const HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeVlWeights(c);
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  vt::Queue q = Q();
  vllm::ModelRegistry::Prepare(*model, c, q);

  const TempModelDir model_dir;
  MmServerHarness h(c, *model, Fixture());
  std::ostringstream install_log;
  REQUIRE(h.install_multimodal_seam(kQwen3VLArch, /*is_multimodal_model=*/true,
                                    model_dir, install_log) ==
          oai::MultiModalChatInstall::kInstalled);

  // The SAME image twice. The second request hits the encoder cache by mm_hash,
  // so the tower runs once; both must still answer with the same shape.
  for (int i = 0; i < 2; ++i) {
    const ApiServer::DispatchResult r =
        h.server.handle_chat_completions(ChatBodyWithImage(/*max_tokens=*/2));
    INFO("attempt ", i, " body: ", r.body);
    REQUIRE(r.status == 200);
    const json j = json::parse(r.body);
    CHECK(j.at("usage").at("completion_tokens") == 2);
    CHECK(j.at("usage").at("prompt_tokens") == 3 + kExpectedImageTokens);
  }
}

// ---------------------------------------------------------------------------
// 4. THE TOWER'S OUTPUT REACHES THE FORWARD — two DIFFERENT images, one prompt,
//    two different answers.
//
//    Case 2 proves the request arrives. It does NOT prove the pixels do: an
//    encoder hook that returned a correctly SHAPED constant would satisfy every
//    assertion there, and so would a merge that spliced the token embedding it
//    was about to overwrite. This case is the one that separates them, and it
//    is why deleting the `Qwen3VLVisionForward` call is a mutation with
//    something to detect rather than only a shape check to trip.
//
//    Different mm_hash on each leg, so the encoder cache cannot answer the
//    second from the first.
// ---------------------------------------------------------------------------
TEST_CASE("mm forward: two DIFFERENT images give two different completions") {
  const HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeVlWeights(c);
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  vt::Queue q = Q();
  vllm::ModelRegistry::Prepare(*model, c, q);

  const TempModelDir model_dir;
  MmServerHarness h(c, *model, Fixture());
  std::ostringstream install_log;
  REQUIRE(h.install_multimodal_seam(kQwen3VLArch, /*is_multimodal_model=*/true,
                                    model_dir, install_log) ==
          oai::MultiModalChatInstall::kInstalled);

  const ApiServer::DispatchResult a =
      h.server.handle_chat_completions(ChatBodyWithImageLogprobs(4, /*variant=*/0));
  const ApiServer::DispatchResult b =
      h.server.handle_chat_completions(ChatBodyWithImageLogprobs(4, /*variant=*/1));
  INFO("a: ", a.body);
  INFO("b: ", b.body);
  REQUIRE(a.status == 200);
  REQUIRE(b.status == 200);

  // Same prompt, same weights, greedy sampling: the ONLY difference between the
  // two forwards is the pixels. The logprobs of the FIRST generated token
  // therefore have to differ, and if they do not, the pixels did not reach the
  // forward.
  const json ja = json::parse(a.body);
  const json jb = json::parse(b.body);
  const json& la = ja.at("choices").at(0).at("logprobs").at("content").at(0);
  const json& lb = jb.at("choices").at(0).at("logprobs").at("content").at(0);
  MESSAGE("image A logprob0 ", la.dump());
  MESSAGE("image B logprob0 ", lb.dump());
  CHECK(la != lb);

  // Recorded, not asserted: on a random tiny checkpoint the SAMPLED text does
  // not have to move — the argmax over 64 vocabulary entries is saturated. The
  // logprob comparison above is the sensitive half; this line exists so a
  // reader is not surprised by two identical completions beside a passing case.
  MESSAGE("image A text '", CompletionText(a), "'  image B text '",
          CompletionText(b), "'");
}

// ---------------------------------------------------------------------------
// 5. THE WRONG ARCHITECTURE GETS NOTHING (#2475), through the same HTTP
//    dispatch.
//
//    This is the case the old install could not have. Its gate was the
//    EXISTENCE of `<model_dir>/preprocessor_config.json`, and this model
//    directory has one — so an install that keys on the file, or a registry
//    lookup that ignores its key and returns the first entry, builds Qwen3-VL's
//    processor here and answers 200. Only a lookup that actually reads the
//    architecture refuses.
//
//    The refusal is the SECOND half of the requirement: never fall through to
//    another model's processor, and never be swallowed into the text path. A
//    200 on this body means one of those two happened.
// ---------------------------------------------------------------------------
TEST_CASE("mm chat registry: an architecture with no registered seam REFUSES the image request by name") {
  const HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeVlWeights(c);
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  vt::Queue q = Q();
  vllm::ModelRegistry::Prepare(*model, c, q);

  // A populated Qwen3-VL model directory, deliberately: the file is present and
  // must not be what decides.
  const TempModelDir model_dir;
  MmServerHarness h(c, *model, Fixture());
  std::ostringstream install_log;
  REQUIRE(h.install_multimodal_seam(kUnregisteredMmArch,
                                    /*is_multimodal_model=*/true, model_dir,
                                    install_log) ==
          oai::MultiModalChatInstall::kRefusing);
  INFO("install log: ", install_log.str());
  CHECK(install_log.str().find(kUnregisteredMmArch) != std::string::npos);

  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(ChatBodyWithImage(/*max_tokens=*/3));
  INFO("body: ", r.body);
  // 200 here is the defect: it means the request was served, either by
  // Qwen3-VL's processor or by the text path with the image dropped.
  CHECK(r.status == 400);
  const json j = json::parse(r.body);
  const std::string message =
      j.at("error").at("message").get<std::string>();
  INFO("message: ", message);
  CHECK(message.find(kUnregisteredMmArch) != std::string::npos);
  CHECK(message.find("multimodal input is not available") != std::string::npos);
  // And it names the missing part, so the reader of the log knows what to
  // register rather than only that something failed.
  CHECK(message.find("REGISTER_VLLM_MM_CHAT") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 6. THE REFUSAL IS SCOPED TO MULTIMODAL PARTS. A refusing seam that fired on
//    every request would take the text path down with the images, which is a
//    different way of breaking the server than the one this row is fixing.
//
//    The tiny Qwen3-VL forward refuses a step with no multimodal request by
//    name, so a text request on THIS harness does not reach 200. What is
//    asserted is that whatever it reaches, it is not the seam's refusal.
// ---------------------------------------------------------------------------
TEST_CASE("mm chat registry: the refusing seam leaves a text request alone") {
  const HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeVlWeights(c);
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  vt::Queue q = Q();
  vllm::ModelRegistry::Prepare(*model, c, q);

  const TempModelDir model_dir;
  MmServerHarness h(c, *model, Fixture());
  std::ostringstream install_log;
  REQUIRE(h.install_multimodal_seam(kUnregisteredMmArch,
                                    /*is_multimodal_model=*/true, model_dir,
                                    install_log) ==
          oai::MultiModalChatInstall::kRefusing);

  const json body = {{"model", "test-model"},
                     {"messages", json::array({{{"role", "user"},
                                                {"content", "hello"}}})},
                     {"max_completion_tokens", 2},
                     {"temperature", 0.0}};
  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(body.dump());
  INFO("body: ", r.body);
  // The seam let it through: what refuses is the tiny fixture's OWN registered
  // forward, several layers further in, and its message says so by name. That
  // is a stronger statement than the absence of the seam's message, because it
  // says how far the request got rather than only which error it avoided.
  CHECK(r.body.find("multimodal input is not available") == std::string::npos);
  CHECK(r.body.find("requires multimodal inputs (ModelForwardInput.mm)") !=
        std::string::npos);
}

// ---------------------------------------------------------------------------
// 7. A TEXT-ONLY ARCHITECTURE INSTALLS NOTHING (registry.py:109-110). The old
//    install had no way to know this: it asked the filesystem, so a text-only
//    checkpoint that happened to ship a `preprocessor_config.json` went into
//    the Qwen3-VL branch and out through the swallowed catch.
//
//    An install that dropped the declaration and keyed on the file alone would
//    report kInstalled here, because this directory has the file and the
//    architecture is the registered one.
// ---------------------------------------------------------------------------
TEST_CASE("mm chat registry: an architecture that declares no multimodal support installs no seam") {
  const HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeVlWeights(c);
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  vt::Queue q = Q();
  vllm::ModelRegistry::Prepare(*model, c, q);

  const TempModelDir model_dir;
  MmServerHarness h(c, *model, Fixture());
  std::ostringstream install_log;
  CHECK(h.install_multimodal_seam(kQwen3VLArch, /*is_multimodal_model=*/false,
                                  model_dir, install_log) ==
        oai::MultiModalChatInstall::kTextOnlyModel);
  // Nothing to announce: this is the ordinary text-only server, not a failure.
  CHECK(install_log.str().empty());
}

// ---------------------------------------------------------------------------
// 8. A REGISTERED FACTORY THAT RETURNS AN EMPTY SEAM IS A REFUSAL, NOT A
//    SUCCESS (#2475, fresh-review F2). `MakeSeam` checked that `make_seam` was
//    not null and then trusted whatever it returned. A seam whose `chat_fn` is
//    an empty `std::function` installs as NOTHING: `mm_chat_fn_` is falsy, the
//    image request takes the text path, and the startup log says
//    "multimodal chat seam wired for architecture 'X'".
//
//    Two assertions, because either alone is satisfiable by the wrong tree.
//    The install must not report success — a case that only checked the
//    dispatch would pass on a server that reported `kInstalled` and happened to
//    fail later. And the REQUEST must not reach the text path — a case that
//    only checked the enum would pass on an install that returned `kRefusing`
//    and installed nothing.
//
//    The second assertion is what makes this a reachability case rather than an
//    enum case: on the unguarded tree the body carries the tiny fixture's own
//    forward refusal, "requires multimodal inputs (ModelForwardInput.mm)",
//    which is the text path saying out loud that the image was dropped.
// ---------------------------------------------------------------------------
TEST_CASE("mm chat registry: a registered factory that returns an EMPTY seam refuses, and never installs as success") {
  const HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeVlWeights(c);
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  vt::Queue q = Q();
  vllm::ModelRegistry::Prepare(*model, c, q);

  // The premise: this architecture IS registered, and its factory IS callable.
  // Without this the case could pass for the unregistered-architecture reason
  // that case 5 already covers.
  const oai::MultiModalChatRegistration* empty =
      oai::MultiModalChatRegistry::Find(kEmptySeamMmArch);
  REQUIRE(empty != nullptr);
  REQUIRE(empty->make_seam != nullptr);

  const TempModelDir model_dir;
  MmServerHarness h(c, *model, Fixture());
  std::ostringstream install_log;
  const oai::MultiModalChatInstall outcome = h.install_multimodal_seam(
      kEmptySeamMmArch, /*is_multimodal_model=*/true, model_dir, install_log);
  INFO("install log: ", install_log.str());
  CHECK(outcome != oai::MultiModalChatInstall::kInstalled);
  CHECK(outcome == oai::MultiModalChatInstall::kRefusing);
  // And it announced the refusal rather than a wiring. The success log is the
  // half that makes the defect invisible to an operator reading startup.
  CHECK(install_log.str().find("UNAVAILABLE") != std::string::npos);
  CHECK(install_log.str().find(kEmptySeamMmArch) != std::string::npos);
  CHECK(install_log.str().find("seam wired") == std::string::npos);

  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(ChatBodyWithImage(/*max_tokens=*/3));
  INFO("body: ", r.body);
  CHECK(r.status == 400);
  CHECK(r.body.find("multimodal input is not available") != std::string::npos);
  CHECK(r.body.find(kEmptySeamMmArch) != std::string::npos);
  CHECK(r.body.find("no callable chat function") != std::string::npos);
  // THE FALL-THROUGH ASSERTION. This string is the fixture forward's own
  // refusal of a step carrying no multimodal request (the same one case 6
  // asserts a TEXT request reaches). Seeing it here means the image request was
  // answered from the text path with the image dropped, which on a model that
  // also serves text is a 200 with no error at all.
  CHECK(r.body.find("requires multimodal inputs (ModelForwardInput.mm)") ==
        std::string::npos);
}
