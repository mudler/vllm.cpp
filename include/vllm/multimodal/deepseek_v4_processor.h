// DeepSeek-V4 Flash Vision prompt and image processor.
//
// Ported from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp at revision
// 86f746b36186f0e567729a5c06a8c918caba82a9:
//   encoding/encoding_dsv4.py::{parse_tagged_text,process_image_messages,
//     encode_messages}
//   inference/image_processor.py::{grid_tokens,solve_resize_ratio,safe_resize,
//     load_image,build_image_block,prepare_vl_inputs}
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/multimodal/inputs.h"

namespace vllm::multimodal {

inline constexpr const char* kDeepSeekV4ImagePlaceholder =
    "<｜deepseek_image｜>";

enum DeepSeekV4ImageTokenType : int64_t {
  kImageStart = 0,
  kImagePad = 1,
  kImage = 2,
  kImageNewLine = 3,
  kImageEnd = 4,
};

struct DeepSeekV4EncodedPrompt {
  std::string prompt;
  std::vector<nlohmann::ordered_json> images;
};

// Converts compact <image>path</image> syntax to OpenAI image_url content
// blocks. A string without tags is returned as a JSON string, matching the
// pinned Python union return.
nlohmann::ordered_json ParseDeepSeekV4TaggedText(const std::string& text);

// Canonical prompt encoder. `messages` and `context` use OpenAI message JSON so
// nested tool_result content remains representable. Only images from `messages`
// are returned; context images affect prior prompt state but are not re-submitted.
DeepSeekV4EncodedPrompt EncodeDeepSeekV4Messages(
    const nlohmann::ordered_json& messages, const std::string& thinking_mode,
    const nlohmann::ordered_json& context = nlohmann::ordered_json::array(),
    bool drop_thinking = true, bool add_default_bos_token = true,
    const std::string& reasoning_effort = "low");

struct DeepSeekV4GridTokens {
  int64_t n_llm_h = 0;
  int64_t n_llm_w = 0;
  int64_t num_tokens = 0;
};

struct DeepSeekV4Resize {
  int64_t n_llm_h = 0;
  int64_t n_llm_w = 0;
  int64_t height = 0;
  int64_t width = 0;
  int64_t num_tokens = 0;
};

DeepSeekV4GridTokens GridTokens(int64_t height, int64_t width,
                                int64_t patch_size,
                                int64_t downsample_ratio);
DeepSeekV4Resize SolveResizeRatio(int64_t height, int64_t width,
                                  int64_t patch_size,
                                  int64_t downsample_ratio,
                                  int64_t max_image_tokens);
DeepSeekV4Resize SafeResize(int64_t height, int64_t width,
                            int64_t best_height, int64_t best_width,
                            int64_t patch_size, int64_t downsample_ratio,
                            int64_t max_image_tokens);

struct DeepSeekV4ImageBlock {
  std::vector<int64_t> types;
  std::vector<int64_t> permutation;
};

DeepSeekV4ImageBlock BuildDeepSeekV4ImageBlock(int64_t n_llm_h,
                                               int64_t n_llm_w,
                                               int64_t start_position);

struct DeepSeekV4ProcessorConfig {
  int64_t patch_size = 14;
  int64_t downsample_ratio = 3;
  int64_t max_image_tokens = 384;
  int64_t min_pixels = 147456;
  int64_t max_width_height_ratio = 8;
  int32_t vocab_size = 129280;
  // The mm-hash NAMESPACE, `MultiModalHasher.hash_kwargs(model_id=...)`
  // (hasher.py:153, processing/inputs.py:62). Qwen3-VL carries the same field
  // for the same reason: two engines serving different checkpoints must not
  // collide in a cache keyed on the digest.
  std::string model_id;
};

class DeepSeekV4ImageProcessor {
 public:
  explicit DeepSeekV4ImageProcessor(DeepSeekV4ProcessorConfig config);

  const DeepSeekV4ProcessorConfig& config() const { return config_; }

  // RGB input is exact contiguous HWC uint8 data. The returned ImageKwargs
  // carries [n_vit_h*n_vit_w, 3*patch_size*patch_size] BF16 patch rows and grid
  // [1,h,w].
  ImageKwargs ProcessImage(std::span<const uint8_t> rgb, int64_t height,
                           int64_t width) const;

  // The CONTENT key for one image, over the same bytes `ProcessImage` reads:
  // `hash_kwargs(model_id=<model_id>, image=<PIL RGB>)`, exactly as
  // `Qwen3VLImageProcessor::HashImage` computes it. It is only HALF of the key
  // a DeepSeek feature needs -- see `MakeDeepSeekV4MmHash` for the other half
  // and for why the content alone is not enough here.
  std::string HashImage(std::span<const uint8_t> rgb, int64_t height,
                        int64_t width) const;

 private:
  DeepSeekV4ProcessorConfig config_;
};

// One processed image together with the CONTENT half of its cache key.
//
// The two travel as one value because a `MultiModalFeatureSpec` built without a
// key is not merely incomplete, it is WRONG in a way nothing downstream can
// see: `Scheduler::try_schedule_encoder_inputs` skips a second feature whose
// `mm_hash` it has already scheduled this step, and both the scheduler's
// `EncoderCacheManager` and the runner's `encoder_cache_` are process-global
// maps keyed on that string alone. Two images sharing the empty string are
// therefore ONE image to all three, and the second placeholder is filled with
// the first image's rows. So there is no default: a caller has to say what the
// image was.
struct DeepSeekV4ImageItem {
  std::shared_ptr<ImageKwargs> kwargs;
  // `DeepSeekV4ImageProcessor::HashImage` over the same RGB bytes.
  std::string content_hash;
};

// The FULL feature key: the image content, its aligner grid, and the block's
// leading compression padding.
//
// THE CONTENT IS NOT ENOUGH, and this is specific to this architecture.
// `BuildDeepSeekV4ImageBlock` derives `compress_pad = 3 - start_position % 4`,
// so ONE image placed at two different prompt offsets produces blocks of two
// different LENGTHS and two different row sequences. The encoder output is a
// function of (content, grid, start_position mod 4) and of nothing else, which
// is exactly what this key spells. A content-only key -- the shape Qwen3-VL can
// afford, because its expansion has no start-position term -- would let the
// second occurrence of one image read the first occurrence's cached rows and
// splice a block of the wrong length under the sentinels.
std::string MakeDeepSeekV4MmHash(const std::string& content_hash,
                                 int64_t n_llm_h, int64_t n_llm_w,
                                 int64_t compress_pad);

// Expands every placeholder to vocab_size + image-type sentinel ids and records
// each span in the existing shared MultiModalInputs/MultiModalFeatureSpec types.
// Every emitted feature carries `MakeDeepSeekV4MmHash` over its own item; an
// item with an empty `content_hash` is REFUSED by name rather than emitted with
// a key that collides with every other one.
MultiModalInputs PrepareDeepSeekV4Inputs(
    const std::vector<int32_t>& prompt_token_ids, int32_t image_token_id,
    const std::vector<DeepSeekV4ImageItem>& images,
    const DeepSeekV4ProcessorConfig& config);

}  // namespace vllm::multimodal
