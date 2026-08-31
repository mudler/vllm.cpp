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

 private:
  DeepSeekV4ProcessorConfig config_;
};

// Expands every placeholder to vocab_size + image-type sentinel ids and records
// each span in the existing shared MultiModalInputs/MultiModalFeatureSpec types.
MultiModalInputs PrepareDeepSeekV4Inputs(
    const std::vector<int32_t>& prompt_token_ids, int32_t image_token_id,
    const std::vector<std::shared_ptr<ImageKwargs>>& images,
    const DeepSeekV4ProcessorConfig& config);

}  // namespace vllm::multimodal
