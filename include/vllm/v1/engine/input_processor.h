// Ported from: vllm/v1/engine/input_processor.py @ e24d1b24
// (InputProcessor.__init__ + process_inputs, the T0 text path).
//
// Scope (M1.8 Task 2): turn a text prompt + SamplingParams into an
// EngineCoreRequest the EngineCore (Task 1) can schedule. This is the T0 slice of
// upstream InputProcessor.process_inputs (input_processor.py:242-385):
//   _validate_params  -> RUN SamplingParams::PostInit()/Verify()  (see below)
//   preprocess/tokenize -> tokenizer.Encode(prompt) -> prompt_token_ids
//   sampling_params.clone() (implicit: params passed by value)
//   default max_tokens = model_config.max_model_len - len(prompt)  when unset
//   update_from_generation_config(generation_config, eos_token_id)  (:323)
//   update_from_tokenizer(tokenizer)                                (:328)
//   -> EngineCoreRequest(request_id, prompt_token_ids, sampling_params, ...)
//
// THE M1.1 CARRY THIS CLOSES: our SamplingParams (M1.1) deferred __post_init__
// to "the constructing unit — M1.8". The InputProcessor IS that unit:
// ValidateParams runs PostInit() (which normalizes in place AND runs Verify()),
// mirroring upstream, where __post_init__ ran at SamplingParams construction and
// process_inputs then calls params.verify(model_config, ...).
//
// DEVIATIONS vs the pinned API (recorded, use OUR names):
//   - __init__ takes VllmConfig (from which it pulls model_config,
//     generation_config_fields and a renderer holding the tokenizer). We hold a
//     tokenizer + HfConfig reference directly (the T0 deps), deriving
//     max_model_len from HfConfig.max_position_embeddings and the primary eos +
//     secondary eos ids from HfConfig.raw["eos_token_id"] (int OR list) with a
//     Tokenizer::EosId() fallback. HfConfig has no max_model_len override
//     (rope-scaling etc.), so max_position_embeddings stands in for it at T0.
//   - process_inputs signature reordered to (request_id, prompt, params,
//     arrival_time): only the text prompt + SamplingParams path is kept.
//   - update_from_generation_config: our SamplingParams dropped the
//     _all_stop_token_ids field (M1.1 — the detokenizer computes its own stop
//     buffer), so the only observable T0 effect is setting eos_token_id and
//     merging the SECONDARY eos ids into stop_token_ids (both gated on
//     ignore_eos), matching sampling_params.py:627-655.
//   - update_from_tokenizer is a no-op stub: upstream only processes bad_words
//     there (sampling_params.py:657), and bad_words is a deferred SamplingParams
//     field (M1.1).
//
// DEFERRED (marked; matches upstream so re-adding is mechanical): dict/EngineInput
// prompts, prompt_embeds, encoder/decoder split, multimodal (mm_features),
// pooling (PoolingParams), LoRA, data_parallel_rank validation, request-id
// randomization (assign_request_id), _validate_model_inputs (prompt-length /
// out-of-vocab checks), current_platform.validate_request, trace_headers,
// priority, resumable.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/types.h"

namespace vllm::tok {
class Tokenizer;  // vllm/tokenizer/tokenizer.h
}

namespace vllm::v1 {

class InputProcessor {
 public:
  // __init__ (T0 deps): the tokenizer + HfConfig the processor reads. The
  // TOKENIZER must outlive the InputProcessor; the HfConfig need not — it is
  // fully consumed here (max_model_len + eos ids are derived up front and the
  // config is not retained).
  InputProcessor(const tok::Tokenizer& tokenizer, const HfConfig& config);

  // process_inputs (text path): validate + tokenize + build the request.
  // `params` is taken BY VALUE (upstream clones it); PostInit()/eos-wiring
  // mutate the local copy, never the caller's. Throws std::runtime_error via
  // SamplingParams::Verify() on invalid params. arrival_time defaults to the
  // wall clock (upstream time.time()). `priority` (upstream process_inputs
  // priority arg) is carried onto EngineCoreRequest.priority; default 0.
  EngineCoreRequest process_inputs(
      const std::string& request_id, const std::string& prompt,
      SamplingParams params,
      std::optional<double> arrival_time = std::nullopt,
      int priority = 0) const;

 private:
  // _validate_params: runs SamplingParams::PostInit() (normalize + Verify) —
  // this closes the M1.1 deferred-__post_init__ carry.
  void ValidateParams(SamplingParams& params) const;
  // update_from_generation_config (T0 subset: eos_token_id + secondary stop ids).
  void UpdateFromGenerationConfig(SamplingParams& params) const;
  // update_from_tokenizer (T0 no-op: bad_words is deferred).
  void UpdateFromTokenizer(SamplingParams& params) const;

  const tok::Tokenizer& tokenizer_;
  // NOTE: no `const HfConfig&` member. The constructor consumes the HfConfig
  // ENTIRELY at construction — max_model_len_, eos_token_id_ and
  // generation_config_eos_ids_ below are everything this class needs from it —
  // so the reference it used to hold was never read. Clang's
  // -Wunused-private-field flagged it while building on macOS
  // (BACKEND-METAL-MLX W0). Removed rather than suppressed, which also drops a
  // lifetime obligation: the HfConfig no longer has to outlive the processor.
  // model_config.max_model_len (T0: HfConfig.max_position_embeddings).
  int64_t max_model_len_ = 0;
  // The primary eos id (renderer.get_eos_token_id()): the eos_ids list head, or
  // the tokenizer's eos, or unset.
  std::optional<int32_t> eos_token_id_;
  // generation_config["eos_token_id"] as a list (int is a 1-element list).
  std::vector<int32_t> generation_config_eos_ids_;
};

}  // namespace vllm::v1
