// MODEL-MM-deepseek-v4 W5 (#2411) — CAN A USER SEND AN IMAGE?
//
// W4 made an image reach `ModelRegistry::Forward`. It entered through
// `ModelRegistry::Load`, `EncodeMm`, `EmbedMm` and `Forward`, and every one of
// those is a component seam: the REQUEST path above them was still unwired, and
// the row's spec listed `EncodeDeepSeekV4Messages`,
// `DeepSeekV4ImageProcessor::ProcessImage` and `PrepareDeepSeekV4Inputs` under
// `## Owed` as reached by nothing.
//
// This suite enters through the two production surfaces a user actually arrives
// at, and through nothing else:
//
//   `MultiModalChatRegistry::MakeSeam`   the per-architecture dispatch
//   `InstallMultiModalChatSeam`          the ONE production install
//                                        (`server_main.cpp`, `vllm_c.cpp`)
//
// It never calls `MakeDeepSeekV4ChatSeam` by name and never constructs
// `DeepSeekV4ImageProcessor`. Where it needs to say what an answer SHOULD be it
// builds an oracle, which is a different job: an oracle that agrees with the
// production seam proves the seam ran the same composition, and an oracle that
// IS the seam proves nothing.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "deepseek_v4_lang_gguf_fixture.h"
#include "deepseek_v4_mmproj_fixture.h"
#include "vllm/config/multimodal.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/multimodal/deepseek_v4_processor.h"
#include "vllm/multimodal/hasher.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/input_processor.h"  // InputValidationError

namespace oai = vllm::entrypoints::openai;
namespace mm = vllm::multimodal;

namespace {

using json = nlohmann::ordered_json;

constexpr const char* kArch = "DeepseekV4ForCausalLM";

// The architecture's own vocabulary size in this suite. The processor writes
// `vocab_size + type` at every image position, so this number is what makes a
// sentinel identifier OUT OF VOCABULARY, and the fixture tokenizer below is
// built to exactly it.
constexpr int32_t kVocabSize = 16;

// ─── The tokenizer ──────────────────────────────────────────────────────────
//
// A real `vllm::tok::Tokenizer`, because the seam resolves the image
// placeholder to an id BY STRING through the tokenizer it was handed, and the
// whole claim "the marker this encoder emits is the id the expansion counts" is
// only true if one object does both.
//
// The added tokens are the pinned DeepSeek chat template's own markers plus the
// image placeholder. The plain vocabulary is three letters and the byte-level
// newline, which is all the prompt text below needs; `\n\n` between content
// blocks is `RenderContentBlocks`'s own separator, so it has to encode.
constexpr int32_t kImageTokenId = 4;

// The one pre-tokenizer both fixtures below use. It is the GPT-2 byte-level
// split this tree's loader accepts; a shorter pattern is rejected by name.
json ByteLevelPreTokenizer() {
  return json{
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
}

vllm::tok::Tokenizer BuildTokenizer() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_dsv4_mmchat_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array({
      {{"id", 0}, {"content", "<｜begin▁of▁sentence｜>"}, {"special", true}},
      {{"id", 1}, {"content", "<｜end▁of▁sentence｜>"}, {"special", true}},
      {{"id", 2}, {"content", "<｜User｜>"}, {"special", true}},
      {{"id", 3}, {"content", "<｜Assistant｜>"}, {"special", true}},
      {{"id", kImageTokenId},
       {"content", mm::kDeepSeekV4ImagePlaceholder},
       {"special", true}},
      {{"id", 5}, {"content", "</think>"}, {"special", true}},
      {{"id", 6}, {"content", "<think>"}, {"special", true}},
  });
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = ByteLevelPreTokenizer();
  json vocab = json::object();
  vocab["a"] = 7;
  vocab["b"] = 8;
  vocab["c"] = 9;
  vocab[vllm::tok::MapBytesToUnicode("\n")] = 10;
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array()}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

const vllm::tok::Tokenizer& Tok() {
  static const vllm::tok::Tokenizer t = BuildTokenizer();
  return t;
}

// ─── The context the SERVER fills in ────────────────────────────────────────
//
// Field for field what `server_main.cpp` assigns, so a factory that reads
// something the server does not supply fails here rather than in production.
struct Ctx {
  vllm::HfConfig config;
  vllm::MultiModalConfig mm_config;
  oai::MultiModalChatContext ctx;

  explicit Ctx(bool with_mmproj = true) {
    config.vocab_size = kVocabSize;
    config.hidden_size = 32;
    ctx.architecture = kArch;
    ctx.model_dir = "/nonexistent/deepseek-v4-flash-vision";
    ctx.config_path = ctx.model_dir + "/config.json";
    ctx.served_model_name = "deepseek-v4-flash-vision";
    ctx.tokenizer = &Tok();
    // The server's chat renderer. This architecture does NOT use it (see the
    // seam's own header for why), and a factory that silently started to would
    // be caught by this: it throws.
    ctx.prompt_fn = [](const std::vector<oai::ChatMessage>&, bool,
                       const std::vector<oai::ChatCompletionToolsParam>&,
                       const json&) -> std::string {
      throw std::runtime_error(
          "the DeepSeek-V4 seam must render with its own pinned encoder");
    };
    ctx.codec = oai::DefaultImageCodec();
    ctx.config = &config;
    ctx.mm_config = &mm_config;
    if (with_mmproj) ctx.mmproj_path = "/nonexistent/mmproj-BF16.gguf";
  }
};

// ─── Requests ───────────────────────────────────────────────────────────────

oai::ChatContentPart TextPart(const std::string& text) {
  oai::ChatContentPart p;
  p.type = "text";
  p.text = text;
  return p;
}

// A raw-RGB data URI, which is the ONE container the server's codec decodes.
// `side` is the square side; `seed` changes the bytes so two images are
// distinguishable by content and not only by position.
std::string RawRgbDataUri(int64_t side, int seed) {
  static const char* kB64 =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<uint8_t> rgb(static_cast<size_t>(side * side * 3));
  for (size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = static_cast<uint8_t>((i * 7 + static_cast<size_t>(seed) * 53) %
                                  251);
  }
  std::string out;
  for (size_t i = 0; i < rgb.size(); i += 3) {
    const uint32_t v = (static_cast<uint32_t>(rgb[i]) << 16) |
                       (i + 1 < rgb.size()
                            ? static_cast<uint32_t>(rgb[i + 1]) << 8
                            : 0U) |
                       (i + 2 < rgb.size() ? static_cast<uint32_t>(rgb[i + 2])
                                           : 0U);
    out.push_back(kB64[(v >> 18) & 63]);
    out.push_back(kB64[(v >> 12) & 63]);
    out.push_back(i + 1 < rgb.size() ? kB64[(v >> 6) & 63] : '=');
    out.push_back(i + 2 < rgb.size() ? kB64[v & 63] : '=');
  }
  return "data:image/x-raw-rgb;base64," + out;
}

oai::ChatContentPart ImagePart(int64_t side, int seed) {
  oai::ChatContentPart p;
  p.type = "image_url";
  p.url = RawRgbDataUri(side, seed);
  return p;
}

oai::ChatMessage UserWith(std::vector<oai::ChatContentPart> parts) {
  oai::ChatMessage m;
  m.role = "user";
  m.content_parts = std::move(parts);
  return m;
}

[[maybe_unused]] std::string Threw(const std::function<void()>& body) {
  try {
    body();
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) THE REGISTRATION. Before W5 `Find(kArch)` was null, so the server's
//     install answered every DeepSeek image request with a REFUSING seam.
// ---------------------------------------------------------------------------
TEST_CASE("dsv4 mm chat: the architecture has a registered chat seam") {
  const oai::MultiModalChatRegistration* reg =
      oai::MultiModalChatRegistry::Find(kArch);
  REQUIRE(reg != nullptr);
  CHECK(reg->architecture == kArch);
  CHECK(reg->make_seam != nullptr);

  // Reached through the static library's --whole-archive, so a link that
  // dropped the translation unit reads as an EMPTY registry rather than as a
  // subtly wrong one.
  const std::vector<std::string_view> archs =
      oai::MultiModalChatRegistry::SupportedArchs();
  CHECK(std::find(archs.begin(), archs.end(), std::string_view(kArch)) !=
        archs.end());
}

// ---------------------------------------------------------------------------
// (2) THE REQUEST PATH, entered through the registry's own dispatch.
//
//     Everything W1 landed and nothing reached is on this line: the pinned
//     `encode_messages` renders the prompt and places the placeholder, the
//     tokenizer resolves it to one id, `ProcessImage` preprocesses the bytes
//     and `PrepareDeepSeekV4Inputs` expands the placeholder into the block of
//     `vocab_size + type` sentinels the W4 forward consumes.
// ---------------------------------------------------------------------------

namespace {

// The processor's own geometry, so the oracle below is a statement about the
// PINNED numbers rather than a copy of whatever the seam happened to build.
// `min_pixels` is 384*384, so a smaller image is scaled UP to it; both sides
// here are already at or above it and stay where they are.
constexpr int64_t kSideA = 384;  // -> 392 padded -> 28x28 patches -> 10x10 cells
constexpr int64_t kSideB = 560;  // -> 560        -> 40x40 patches -> 14x14 cells

// The block one image occupies at a given prompt offset, computed from the
// PINNED `build_image_block` rather than from the seam's answer.
int64_t OracleBlockLength(int64_t n_llm, int64_t offset) {
  return static_cast<int64_t>(
      mm::BuildDeepSeekV4ImageBlock(n_llm, n_llm, offset).types.size());
}

std::vector<uint8_t> RawRgb(int64_t side, int seed) {
  std::vector<uint8_t> rgb(static_cast<size_t>(side * side * 3));
  for (size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = static_cast<uint8_t>((i * 7 + static_cast<size_t>(seed) * 53) %
                                  251);
  }
  return rgb;
}

}  // namespace

TEST_CASE("dsv4 mm chat: one image reaches MultiModalInputs through the registry") {
  Ctx c;
  const oai::MultiModalChatSeam seam =
      oai::MultiModalChatRegistry::MakeSeam(c.ctx);
  REQUIRE(seam.chat_fn);

  // A text-only conversation NEVER enters the multimodal path, and the seam
  // says so by declining. Without this the claim below could be satisfied by a
  // seam that rewrote every request.
  oai::ChatMessage text;
  text.role = "user";
  text.content = std::string("a");
  CHECK_FALSE(seam.chat_fn({text}).has_value());

  const std::optional<mm::MultiModalInputs> mm =
      seam.chat_fn({UserWith({TextPart("a"), ImagePart(kSideA, 1)})});
  REQUIRE(mm.has_value());
  REQUIRE(mm->mm_features.size() == 1);

  const mm::MultiModalFeatureSpec& f = mm->mm_features[0];
  CHECK(f.modality == "image");
  REQUIRE(f.data != nullptr);
  // The PINNED preprocessor ran: 384 is already at `min_pixels`, so it pads to
  // the next multiple of 14 and yields a 28x28 patch grid at a 3*14*14 feature
  // width. A seam that skipped `ProcessImage` could not produce these.
  CHECK(f.data->image_grid_thw == std::array<int64_t, 3>{1, 28, 28});
  CHECK(f.data->num_patches == 28 * 28);
  CHECK(f.data->patch_feature_dim == 3 * 14 * 14);

  // The PINNED prompt encoder ran: `<bos><User>` then the joined content, so
  // the placeholder sits after the leading text and the block replaces it in
  // place. The oracle is `build_image_block` at the same offset.
  CHECK(f.offset == 5);
  CHECK(static_cast<int64_t>(f.length) == OracleBlockLength(10, 5));

  // ...and the span really is out-of-vocabulary sentinel identifiers, which is
  // the property `ForwardDeepseekV4ForCausalLM` depends on: a forward that
  // ignored `inputs_embeds` refuses rather than answering.
  for (int i = 0; i < f.length; ++i) {
    const int32_t id = mm->prompt_token_ids[static_cast<size_t>(f.offset + i)];
    CHECK(id >= kVocabSize);
    CHECK(id < kVocabSize + 5);
  }
  // Everything outside the span is ordinary vocabulary, so the expansion did
  // not smear over the prompt.
  for (size_t i = 0; i < mm->prompt_token_ids.size(); ++i) {
    if (static_cast<int>(i) >= f.offset &&
        static_cast<int>(i) < f.offset + f.length) {
      continue;
    }
    CHECK(mm->prompt_token_ids[i] < kVocabSize);
  }
  // The key, which the scheduler and both encoder caches are keyed on. It is
  // the shared hasher's digest over the RAW bytes this request carried,
  // namespaced by the served model name, and it is non-empty.
  const std::vector<uint8_t> rgb = RawRgb(kSideA, 1);
  const std::string content = mm::MultiModalHasher::HashImageRGB(
      c.ctx.served_model_name, rgb.data(), kSideA, kSideA);
  CHECK_FALSE(content.empty());
  CHECK(f.mm_hash.rfind(content, 0) == 0);
}

// ---------------------------------------------------------------------------
// (3) TWO INTERLEAVED IMAGES, AND WHICH ONE LANDED WHERE.
//
//     The pinned encoder and the model author's own example both take several
//     images in source order, so the count is the easy half. The half that
//     matters is the ORDER: image 2 must land in the second placeholder, and a
//     seam that swapped the two would still emit two features, two spans and
//     two plausible blocks.
//
//     THREE THINGS MAKE A SWAP VISIBLE HERE, and one alone would not:
//       * the two images have DIFFERENT GRIDS (10x10 cells against 14x14), so
//         a swap changes both span LENGTHS;
//       * they have DIFFERENT CONTENT, so a swap changes both KEYS;
//       * they are separated by TEXT, so a swap that preserved the lengths
//         would still move the text between them.
//     A fixture with two identical images could express none of the three.
// ---------------------------------------------------------------------------
TEST_CASE("dsv4 mm chat: two interleaved images land in source order") {
  Ctx c;
  const oai::MultiModalChatSeam seam =
      oai::MultiModalChatRegistry::MakeSeam(c.ctx);
  REQUIRE(seam.chat_fn);

  // "a" <imgA> "b" <imgB> "c" -- `RenderContentBlocks` joins the blocks with
  // "\n\n", so the rendered prompt is
  //   <bos><User> a \n\n <img> \n\n b \n\n <img> \n\n c <Assistant></think>
  // and the fixture tokenizer gives each of those exactly one id.
  const std::optional<mm::MultiModalInputs> mm = seam.chat_fn(
      {UserWith({TextPart("a"), ImagePart(kSideA, 1), TextPart("b"),
                 ImagePart(kSideB, 2), TextPart("c")})});
  REQUIRE(mm.has_value());
  REQUIRE(mm->mm_features.size() == 2);

  const mm::MultiModalFeatureSpec& f0 = mm->mm_features[0];
  const mm::MultiModalFeatureSpec& f1 = mm->mm_features[1];

  // THE PREMISE, asserted rather than assumed: the two images really are
  // distinguishable. Without this the three claims below could all hold on a
  // fixture where a swap is a no-op.
  REQUIRE(f0.data != nullptr);
  REQUIRE(f1.data != nullptr);
  REQUIRE(f0.data->image_grid_thw != f1.data->image_grid_thw);
  REQUIRE(f0.length != f1.length);
  REQUIRE(f0.mm_hash != f1.mm_hash);

  // (a) CONTENT. Feature 0 carries the bytes of the FIRST `image_url` part and
  //     feature 1 the second, keyed by the shared hasher over the raw request
  //     bytes. This is the assertion a swap fails first.
  const std::vector<uint8_t> rgb_a = RawRgb(kSideA, 1);
  const std::vector<uint8_t> rgb_b = RawRgb(kSideB, 2);
  const std::string hash_a = mm::MultiModalHasher::HashImageRGB(
      c.ctx.served_model_name, rgb_a.data(), kSideA, kSideA);
  const std::string hash_b = mm::MultiModalHasher::HashImageRGB(
      c.ctx.served_model_name, rgb_b.data(), kSideB, kSideB);
  CHECK(f0.mm_hash.rfind(hash_a, 0) == 0);
  CHECK(f1.mm_hash.rfind(hash_b, 0) == 0);

  // (b) GEOMETRY. The first is the 384-wide image (28x28 patches, 10x10 cells)
  //     and the second the 560-wide one (40x40 patches, 14x14 cells).
  CHECK(f0.data->image_grid_thw == std::array<int64_t, 3>{1, 28, 28});
  CHECK(f1.data->image_grid_thw == std::array<int64_t, 3>{1, 40, 40});

  // (c) POSITION. The placeholders sit at prompt indices 5 and 11 --
  //     `<bos> <User> a \n \n <img> \n \n b \n \n <img>` -- so the first span
  //     opens at 5, and the second opens five ordinary tokens after the first
  //     span closes. `build_image_block` at each offset is the oracle for both
  //     lengths, and its answer differs at the two offsets because
  //     `compress_pad` reads the start position.
  CHECK(f0.offset == 5);
  CHECK(static_cast<int64_t>(f0.length) == OracleBlockLength(10, 5));
  const int second_offset = f0.offset + f0.length + 5;
  CHECK(f1.offset == second_offset);
  CHECK(static_cast<int64_t>(f1.length) ==
        OracleBlockLength(14, second_offset));

  // ...and the five tokens BETWEEN the two spans are the rendered text, in
  // order: newline, newline, "b", newline, newline. A swap that happened to
  // preserve the two lengths would still have to move these.
  const std::vector<int32_t> between(
      mm->prompt_token_ids.begin() + f0.offset + f0.length,
      mm->prompt_token_ids.begin() + f1.offset);
  CHECK(between == std::vector<int32_t>{10, 10, 8, 10, 10});

  // Both spans are out-of-vocabulary sentinels and they do not overlap.
  CHECK(f0.offset + f0.length <= f1.offset);
  for (const mm::MultiModalFeatureSpec* f : {&f0, &f1}) {
    for (int i = 0; i < f->length; ++i) {
      CHECK(mm->prompt_token_ids[static_cast<size_t>(f->offset + i)] >=
            kVocabSize);
    }
  }
}

// A CONVERSATION, not a single turn. `MessageToJson` has two arms -- a
// bare-string `content` and a content-part array -- and the pinned encoder
// renders the whole history around the image. A seam that dropped the earlier
// turns, or that fed the encoder only the message carrying the image, would
// still produce a well-formed block at a plausible offset.
TEST_CASE("dsv4 mm chat: an earlier bare-string turn survives into the prompt") {
  Ctx c;
  const oai::MultiModalChatSeam seam =
      oai::MultiModalChatRegistry::MakeSeam(c.ctx);

  oai::ChatMessage user0;
  user0.role = "user";
  user0.content = std::string("b");
  oai::ChatMessage assistant;
  assistant.role = "assistant";
  assistant.content = std::string("c");

  const std::optional<mm::MultiModalInputs> mm = seam.chat_fn(
      {user0, assistant, UserWith({TextPart("a"), ImagePart(kSideA, 1)})});
  REQUIRE(mm.has_value());
  REQUIRE(mm->mm_features.size() == 1);

  // The first turn's "b" (id 8) and the assistant's "c" (id 9) are both in the
  // prompt, and both BEFORE the image span. The single-turn case above puts the
  // span at offset 5; here the two earlier turns push it further out, which is
  // what a dropped history could not do.
  const std::vector<int32_t>& ids = mm->prompt_token_ids;
  const auto pos = [&](int32_t id) {
    return std::find(ids.begin(), ids.end(), id) - ids.begin();
  };
  CHECK(std::count(ids.begin(), ids.end(), 8) == 1);
  CHECK(std::count(ids.begin(), ids.end(), 9) == 1);
  CHECK(pos(8) < mm->mm_features[0].offset);
  CHECK(pos(9) < mm->mm_features[0].offset);
  CHECK(mm->mm_features[0].offset > 5);
}

// ---------------------------------------------------------------------------
// (4) THE CEILING COMES FROM `MultiModalConfig`, and this seam declares none.
//
//     `MakeQwen3VLImageChatFn` caps image at 1 because its body locates ONE
//     part; the spec forbids a lower hard-coded ceiling here, so the honest
//     number is "unlimited" and every limit a user meets is the engine's.
// ---------------------------------------------------------------------------
TEST_CASE("dsv4 mm chat: the image ceiling is the engine's, not this seam's") {
  {
    // The DEFAULT engine: no `--limit-mm-per-prompt`, so the fold leaves
    // upstream's own per-modality default and three images are served.
    Ctx c;
    const oai::MultiModalChatSeam seam =
        oai::MultiModalChatRegistry::MakeSeam(c.ctx);
    REQUIRE(seam.allowed_limits.count("image") == 1);
    CHECK(seam.allowed_limits.at("image") > 1);
    const std::optional<mm::MultiModalInputs> mm = seam.chat_fn(
        {UserWith({ImagePart(kSideA, 1), TextPart("a"), ImagePart(kSideA, 2),
                   TextPart("b"), ImagePart(kSideB, 3)})});
    REQUIRE(mm.has_value());
    CHECK(mm->mm_features.size() == 3);
    // Three DISTINCT keys, so the scheduler runs the tower three times. Two of
    // the three are the same GRID, which is exactly the pair a length check
    // could not tell apart.
    CHECK(mm->mm_features[0].mm_hash != mm->mm_features[1].mm_hash);
    CHECK(mm->mm_features[1].mm_hash != mm->mm_features[2].mm_hash);
    CHECK(mm->mm_features[0].mm_hash != mm->mm_features[2].mm_hash);
  }
  {
    // `--limit-mm-per-prompt image=2`: the fold takes the engine's number and
    // the third image is REFUSED with upstream's own message, as HTTP 400.
    Ctx c;
    c.mm_config.limit_per_prompt["image"] = 2;
    const oai::MultiModalChatSeam seam =
        oai::MultiModalChatRegistry::MakeSeam(c.ctx);
    CHECK(seam.allowed_limits.at("image") == 2);
    const std::string what = Threw([&] {
      (void)seam.chat_fn({UserWith({ImagePart(kSideA, 1), ImagePart(kSideA, 2),
                                    ImagePart(kSideA, 3)})});
    });
    INFO("what: ", what);
    CHECK(what.find("At most 2 image(s)") != std::string::npos);
    // Two still pass, so the number is a LIMIT and not a refusal of the
    // multi-image arm.
    CHECK(seam.chat_fn({UserWith({ImagePart(kSideA, 1), ImagePart(kSideA, 2)})})
              ->mm_features.size() == 2);
  }
  {
    // `--language-model-only` drives every modality to 0, so the first image is
    // refused before anything is decoded.
    Ctx c;
    c.mm_config.language_model_only = true;
    const oai::MultiModalChatSeam seam =
        oai::MultiModalChatRegistry::MakeSeam(c.ctx);
    CHECK(seam.allowed_limits.at("image") == 0);
    const std::string what = Threw(
        [&] { (void)seam.chat_fn({UserWith({ImagePart(kSideA, 1)})}); });
    INFO("what: ", what);
    CHECK(what.find("At most 0 image(s)") != std::string::npos);
  }
  {
    // Every other modality is ABSENT from the declared map, which
    // `context.py:414-415` reads as limit 0. DeepSeek-V4-Flash-Vision is an
    // image-only model and `EncodeMmDeepseekV4ForCausalLM` refuses any other
    // modality by name; this is the same statement one component earlier.
    Ctx c;
    const oai::MultiModalChatSeam seam =
        oai::MultiModalChatRegistry::MakeSeam(c.ctx);
    CHECK(seam.allowed_limits.count("audio") == 0);
    CHECK(seam.allowed_limits.count("video") == 0);
    oai::ChatContentPart audio;
    audio.type = "input_audio";
    const std::string what =
        Threw([&] { (void)seam.chat_fn({UserWith({audio})}); });
    INFO("what: ", what);
    CHECK(what.find("audio") != std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// (5) THE CONTAINER CODEC AND THE URI SCHEME ARE REFUSED, AND A USER CAN TELL.
//
//     No PNG/JPEG decoder is vendored and no http(s) fetch exists. Both are
//     NAMED MM-SERVE residuals belonging to the LIBRARY rather than to this
//     architecture, so this wave refuses them rather than implementing a
//     cross-model capability under a model row.
//
//     What it does own is the STATUS. `DefaultImageCodec` and `DecodeDataUri`
//     both throw `std::runtime_error`, and `api_server.cpp:373` maps that to
//     HTTP 500 "InternalServerError" -- so a `data:image/png;base64,...` body
//     read as a server fault rather than as a request this server cannot
//     serve. This seam re-throws them as `InputValidationError`, the type
//     `api_server.cpp:357-360` maps to 400, with the residual's own message
//     intact.
// ---------------------------------------------------------------------------
TEST_CASE("dsv4 mm chat: a PNG or an http(s) image is refused as a CLIENT error") {
  Ctx c;
  const oai::MultiModalChatSeam seam =
      oai::MultiModalChatRegistry::MakeSeam(c.ctx);

  const auto refuse = [&](const std::string& url) {
    oai::ChatContentPart p;
    p.type = "image_url";
    p.url = url;
    CHECK_THROWS_AS((void)seam.chat_fn({UserWith({p})}),
                    vllm::v1::InputValidationError);
    return Threw([&] { (void)seam.chat_fn({UserWith({p})}); });
  };

  // (a) A container format. The message names the missing part, which is what
  //     AGENTS.md asks of an unimplemented arm.
  const std::string png = refuse("data:image/png;base64,iVBORw0KGgo=");
  INFO("png: ", png);
  CHECK(png.find("PNG/JPEG") != std::string::npos);
  CHECK(png.find("image/x-raw-rgb") != std::string::npos);

  // (b) An http(s) URL never reaches the codec: the fetch is its own residual
  //     and `DecodeDataUri` names it.
  const std::string http = refuse("https://example.invalid/cat.jpg");
  INFO("http: ", http);
  CHECK(http.find("data: URI") != std::string::npos);

  // (c) A raw-RGB payload that is not a square buffer is a client error too,
  //     and it is the codec's own message rather than a generic failure. Six
  //     bytes are two pixels, and no square HxWx3 buffer has that extent.
  //     (Three bytes WOULD be a valid 1x1 image, which is why the payload is
  //     eight base64 characters and not four.)
  const std::string ragged = refuse("data:image/x-raw-rgb;base64,AAAAAAAA");
  INFO("ragged: ", ragged);
  CHECK(ragged.find("square") != std::string::npos);

  // (d) ...and the RIGHT container still works, so (a)-(c) are refusals of the
  //     unimplemented arms and not of images.
  CHECK(seam.chat_fn({UserWith({ImagePart(kSideA, 1)})}).has_value());
}

// ---------------------------------------------------------------------------
// (6) WHAT THE FACTORY REFUSES AT INSTALL, and why each one is at install.
//
//     `InstallMultiModalChatSeam` catches a throwing factory and installs a
//     REFUSING seam: HTTP 400 naming the architecture, text path untouched.
//     Every condition below is therefore answered before the engine's busy
//     loop can meet it, which is the difference between one 400 and every
//     later request -- text ones included -- becoming a 500.
// ---------------------------------------------------------------------------
TEST_CASE("dsv4 mm chat: the factory refuses an install it cannot serve") {
  {
    // NO SECOND FILE. `DeepseekV4ForCausalLM` names both the text checkpoint
    // and the Flash-Vision one, so the architecture cannot answer this and
    // `--mmproj` is the only thing that can.
    Ctx c(/*with_mmproj=*/false);
    const std::string what =
        Threw([&] { (void)oai::MultiModalChatRegistry::MakeSeam(c.ctx); });
    INFO("what: ", what);
    CHECK(what.find("--mmproj") != std::string::npos);
    CHECK(what.find("deepseek4v") != std::string::npos);
    CHECK(what.find("2411") != std::string::npos);
  }
  {
    // NO PLACEHOLDER TOKEN. The encoder writes the string at every image
    // position and the expansion counts the id it resolves to, so a default
    // would be a guess that surfaces as an image-count mismatch naming the
    // wrong thing.
    Ctx c;
    const vllm::tok::Tokenizer bare = [] {
      // A tokenizer with the template markers but NOT the image placeholder.
      static int counter = 0;
      const std::string path =
          (std::filesystem::temp_directory_path() /
           ("vllm_dsv4_mmchat_bare_" + std::to_string(counter++) + ".json"))
              .string();
      json doc;
      doc["version"] = "1.0";
      doc["added_tokens"] = json::array(
          {{{"id", 0}, {"content", "<｜User｜>"}, {"special", true}}});
      doc["normalizer"] = nullptr;
      doc["pre_tokenizer"] = ByteLevelPreTokenizer();
      doc["model"] = {{"type", "BPE"},
                      {"ignore_merges", false},
                      {"vocab", json{{"a", 1}}},
                      {"merges", json::array()}};
      std::ofstream(path, std::ios::binary) << doc.dump();
      vllm::tok::Tokenizer t = vllm::tok::Tokenizer::FromHfJson(path);
      std::remove(path.c_str());
      return t;
    }();
    c.ctx.tokenizer = &bare;
    const std::string what =
        Threw([&] { (void)oai::MultiModalChatRegistry::MakeSeam(c.ctx); });
    INFO("what: ", what);
    CHECK(what.find("deepseek_image") != std::string::npos);
    CHECK(what.find("added token") != std::string::npos);
  }
  {
    // NO RESOLVED VOCABULARY SIZE. Every image position is `vocab_size + type`,
    // so a zero would put the sentinels at 0..4 -- INSIDE the vocabulary --
    // and the merge would splice image rows over real tokens with no shape
    // error anywhere.
    Ctx c;
    c.config.vocab_size = 0;
    const std::string what =
        Threw([&] { (void)oai::MultiModalChatRegistry::MakeSeam(c.ctx); });
    INFO("what: ", what);
    CHECK(what.find("vocab_size") != std::string::npos);
  }
  {
    // AN INCOMPLETE CONTEXT is refused by name rather than dereferenced.
    Ctx c;
    c.ctx.config = nullptr;
    CHECK(Threw([&] { (void)oai::MultiModalChatRegistry::MakeSeam(c.ctx); })
              .find("install context is incomplete") != std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// (7) THE SERVER SURFACE. `InstallMultiModalChatSeam` is the ONE production
//     caller of `set_multimodal_chat_fn`, and `create_chat_completion` is what
//     an HTTP request reaches. Everything above this case tests the seam BODY;
//     this one tests that a user arrives at it.
//
//     It enters through `LoadedEngine::FromModelDir` -- the entry point every
//     server and command line takes for a `.gguf` argument -- with a `--mmproj`
//     second file, and drives a chat request carrying two `image_url` parts
//     through the real `OpenAIServingChat`.
// ---------------------------------------------------------------------------
namespace {

// The projector geometry, at the PINNED processor's patch size. `output` must
// be the language model's hidden width (the aligner's rows go straight into the
// residual stream) and `patch` must be 14, because the seam's processor is the
// pinned one and `EncodeMmDeepseekV4ForCausalLM` refuses a feature width the
// projector does not want.
dsv4_mmproj_test::Dims ServerProjDims() {
  dsv4_mmproj_test::Dims d;
  d.output = dsv4_lang_test::kH;
  d.patch = 14;
  return d;
}

dsv4_mmproj_test::Options ServerProjOptions() {
  dsv4_mmproj_test::Options o;
  o.fold_exponents = 7;  // this suite RUNS the tower
  return o;
}

// ONE SERVED REQUEST, on its own engine. A failed step stops `AsyncLLM`, so a
// second request on the same engine reports "submitted to a stopped AsyncLLM"
// and any comparison across the two would measure the ORDER rather than the
// paths. Each call therefore builds the whole production stack again.
struct Served {
  oai::MultiModalChatInstall install = oai::MultiModalChatInstall::kTextOnlyModel;
  std::string install_log;
  std::string error;   // empty when the engine answered
  int prompt_tokens = 0;
  std::string role;
};

Served ServeOnce(std::vector<oai::ChatMessage> messages) {
  // SYNCHRONOUS SCHEDULING, and it is load-bearing rather than tidy. With the
  // default asynchronous scheduler this engine dies non-deterministically in
  // `GPUModelRunner::gather_block_table` -- observed on a ONE-TOKEN TEXT prompt
  // as often as on an image one, and swapping between runs of the same binary,
  // so it is neither a multimodal condition nor a prompt-length one. A gate
  // that reports a different failure each run measures the scheduler, not the
  // seam. The instability itself is recorded under `## Owed`.
  setenv("VT_ASYNC_SCHED", "0", /*overwrite=*/1);
  gguf_test::TempFile lang(dsv4_lang_test::BuildDeepseek4Gguf(
      /*vision=*/true, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512, /*with_tokenizer=*/true));
  gguf_test::TempFile proj(
      dsv4_mmproj_test::Build(ServerProjDims(), ServerProjOptions()));

  vllm::entrypoints::EngineParams params;
  params.mmproj_path = proj.path();
  // OFF, and not incidentally. This architecture's KV topology gives the block
  // pool a hash-block size that differs from its block size, and
  // `BlockPool::cache_full_blocks` refuses that combination by name. That is a
  // prefix-cache gap outside this row; leaving it on kills the engine's busy
  // loop before anything here can be measured.
  params.enable_prefix_caching = false;
  // The fixture GGUF carries no `deepseek4.context_length`, so the engine would
  // resolve `max_model_len = 0` and `InputBatch`'s per-request token row would
  // have no width at all.
  params.max_model_len = 1024;

  Served out;
  std::unique_ptr<vllm::entrypoints::LoadedEngine> engine =
      vllm::entrypoints::LoadedEngine::FromModelDir(lang.path(), params);
  REQUIRE(engine != nullptr);
  CHECK(engine->architecture() == kArch);
  CHECK(engine->is_multimodal_model());

  oai::OpenAIServingChat chat(
      engine->async_engine(), "deepseek-v4-flash-vision",
      [](const std::vector<oai::ChatMessage>& ms, bool,
         const std::vector<oai::ChatCompletionToolsParam>&,
         const json&) -> std::string {
        // The TEXT path's renderer. An image request never reaches it: this
        // architecture renders with its own pinned encoder.
        std::string t;
        for (const oai::ChatMessage& m : ms) t += m.content.value_or(std::string());
        return t;
      });

  // THE PRODUCTION INSTALL, field for field as `server_main.cpp` fills it in.
  oai::MultiModalChatContext ctx;
  ctx.architecture = std::string(engine->architecture());
  ctx.model_dir = std::filesystem::path(lang.path()).parent_path().string();
  ctx.config_path = ctx.model_dir + "/config.json";  // a .gguf has none
  ctx.served_model_name = "deepseek-v4-flash-vision";
  ctx.tokenizer = &engine->tokenizer();
  ctx.prompt_fn = [](const std::vector<oai::ChatMessage>&, bool,
                     const std::vector<oai::ChatCompletionToolsParam>&,
                     const json&) -> std::string { return std::string(); };
  ctx.codec = oai::DefaultImageCodec();
  ctx.mm_config = &engine->mm_config();
  ctx.config = &engine->config();
  ctx.mmproj_path = params.mmproj_path;
  std::ostringstream log;
  out.install = oai::InstallMultiModalChatSeam(
      chat, engine->is_multimodal_model(), ctx, log);
  out.install_log = log.str();

  oai::ChatCompletionRequest req;
  req.messages = std::move(messages);
  req.max_completion_tokens = 2;
  req.temperature = 0.0;
  req.stream = false;
  oai::ChatCompletionResult result;
  out.error = Threw([&] { result = chat.create_chat_completion(req); });
  if (out.error.empty() && result.response.has_value() &&
      !result.response->choices.empty()) {
    out.prompt_tokens = result.response->usage.prompt_tokens;
    out.role = result.response->choices[0].message.role;
  }
  return out;
}

}  // namespace

TEST_CASE("dsv4 mm chat: two images reach the server through the production install") {
  const Served image_run = ServeOnce(
      {UserWith({TextPart("a"), ImagePart(kSideA, 1), TextPart("b"),
                 ImagePart(kSideB, 2)})});

  // (a) THE INSTALL. `InstallMultiModalChatSeam` is the ONE production caller
  //     of `set_multimodal_chat_fn`, and it reached the DeepSeek factory. NOT
  //     `kRefusing`, which is what this architecture got before W5 registered
  //     one: `Find("DeepseekV4ForCausalLM")` was null, so the install caught
  //     `RaiseForUnregistered` and wired a seam that answered every image
  //     request with HTTP 400.
  CHECK(image_run.install == oai::MultiModalChatInstall::kInstalled);
  INFO("install log: ", image_run.install_log);
  CHECK(image_run.install_log.find("DeepSeek-V4") != std::string::npos);
  CHECK(image_run.install_log.find("deepseek4v") != std::string::npos);

  // (b) WHERE THE SERVED IMAGE REQUEST GETS TO, and this is the wave's own
  //     reachability claim at the server surface.
  //
  //     The message is the REGISTERED FORWARD's own named W7-device residual,
  //     raised inside `deepseek_v4.cpp`. So the request travelled
  //     `create_chat_completion` -> the installed seam -> `AsyncLLM` ->
  //     `Scheduler` -> `GPUModelRunner::execute_model` ->
  //     `ModelRegistry::Forward`, and was refused THERE. Nothing short of the
  //     registered forward can produce it, which is what makes it evidence
  //     rather than a disappointment: `DeepseekV4Model::ForwardDevice` is what
  //     the runner's gather-logits path reaches for EVERY request on this
  //     architecture, and a CPU build carries no V4 device kernels. Serving
  //     this architecture on a device is W7-CUDA's and issue #2411 owns it.
  //
  //     A generated answer is therefore not available here, and the case
  //     upgrades itself to one the moment the engine can produce it.
  //
  //     ONLY THE IMAGE REQUEST IS DRIVEN. Earlier versions of this case also
  //     served a one-token and a 260-token TEXT prompt on their own engines,
  //     to attribute the stop. Both are UNSTABLE on this synthetic checkpoint:
  //     the same binary segfaults in `InputBatch::add_request` on roughly half
  //     of its runs and otherwise dies in `GPUModelRunner::gather_block_table`,
  //     while the multimodal request reaches the forward on every run of eight.
  //     A flaky probe in a gate measures the scheduler rather than the seam, so
  //     the text instability is recorded in the row's spec under `## Owed` with
  //     that measurement instead of being asserted here.
  MESSAGE("image: " << (image_run.error.empty() ? std::string("served")
                                                : image_run.error));
  if (image_run.error.empty()) {
    // The engine answers. Then the multimodal claim is the PROMPT the request
    // was expanded to: two image blocks of ~120 sentinel tokens each, not the
    // four content parts a seam-less path would have rendered.
    CHECK(image_run.role == "assistant");
    CHECK(image_run.prompt_tokens > 200);
  } else {
    CHECK(image_run.error.find("deepseek_v4.cpp") != std::string::npos);
    CHECK(image_run.error.find("W7-device") != std::string::npos);
  }
}
