// Qwen3-VL (`Qwen3VLForConditionalGeneration`) — M2c e2e image->text wire-up.
//
// The FORKED VL decode: it reuses the landed Qwen3-dense text backbone
// (Qwen3DenseWeights / dense_attn AttnBlock machinery) but forks the forward on
// three points that the plain-dense path does NOT do:
//   1. inputs_embeds path — embed text ids, then masked-scatter the vision tower's
//      merger output (Qwen3VLMergeMultimodal) into the image-placeholder rows
//      before the first decoder layer (instead of the pure embed-from-ids path).
//   2. 3-section MRoPE positions — vt::RopeFromCache over positions [3,T] +
//      mrope_section=[24,20,20] interleaved (the existing mrope path; NOT a new
//      kernel), instead of 1-D RoPE.
//   3. DeepStack injection — add the tower's 3 multiscale merger outputs
//      (Qwen3VLComputeDeepstack -> [L,T,H]) to the hidden stream after decoder
//      layers 0/1/2 (qwen3_vl.py:1589-1594).
//
// Everything else (per-head q/k RMSNorm, paged FA2, SwiGLU MLP, tied lm_head) is
// the byte-identical landed dense path. The vision tower is the M2a
// Qwen3VLVisionForward; the merge/rope-index/deepstack index math are the M2b
// host helpers (qwen3_vl_text.h). This TU is ADDITIVE — it does NOT touch the
// shared dense forward / model runner / registry, so the text SACRED gates are
// byte-identical by construction.
//
// Ported from vllm/model_executor/models/qwen3_vl.py @ e24d1b24:
//   Qwen3VLForConditionalGeneration.load_weights (:2905), get_input_embeddings /
//   forward (:2843), Qwen3LLMModel.forward deepstack (:1589-1594);
//   _get_mrope_input_positions (:2567); the language_model.* / visual.* remap.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/config/multimodal.h"                     // #607 L3 MultiModalConfig
#include "vllm/model_executor/models/qwen3.h"             // Qwen3DenseWeights, PagedKvCache
#include "vllm/model_executor/models/qwen3_vl_vision.h"    // Qwen3VLVisionWeights/Config
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"

namespace vllm {

class SafetensorsFile;
class LoadedModel;  // MM-ENGINE-FORWARD: the registered mm-forward driver target.
struct ForwardLogits;  // RUNNER-ROUTE: on-device logits carrier (qwen3_5.h).

namespace v1 {
struct CommonAttentionMetadata;
}  // namespace v1

// The full Qwen3-VL model weights: the plain-dense text backbone (bf16, tied
// lm_head) under the `model.language_model.*` prefix + the M2a vision tower under
// `model.visual.*`.
struct Qwen3VLWeights {
  Qwen3DenseWeights text;
  multimodal::Qwen3VLVisionWeights vision;
  multimodal::Qwen3VLVisionConfig vision_cfg;

  // #607 L3, the TOWER SKIP. `vision_loaded` is the symmetric partner of Muse
  // Glimmer's `MuseGlimmerVisionTower::loaded`; `vision_skipped` says the load
  // deliberately did not read `model.visual.*` because every modality the tower
  // serves was at limit 0 (interfaces.py:288-293). `vision_cfg` is populated
  // EITHER WAY — that is the construct half of construct-without-initialise.
  //
  // NOTE, and it is a gap this row records rather than repairs: nothing in
  // `src/` reads `Qwen3VLWeights::vision` today. The three consumers are
  // hardware e2e tests. So the flags exist for symmetry and for the gate, and
  // the tower they describe has no production consumer to guard yet.
  bool vision_loaded = false;
  bool vision_skipped = false;
};

// Load `Qwen3VLForConditionalGeneration` (Qwen3-VL-4B, BF16) safetensors. The
// text half remaps `model.language_model.*` onto the landed Qwen3-dense loader
// helpers (LoadMergedBf16RawNK etc.); the vision half loads `model.visual.*` into
// the M2a tower weights (bf16 widened to f32, matching the M2a dump). `config` is
// the text_config-resolved HfConfig (hidden 2560, 36 layers, 32 heads, head_dim
// 128, kv 8, vocab 151936, rope_theta 5e6, tied).
//
// `mm_config` (#607 L3) is the engine's multimodal input limits, BORROWED. When
// image AND video are both at limit 0 the tower's geometry is still resolved but
// `model.visual.*` is never read, mirroring `_mark_tower_model`'s
// `no_init_weights` over `torch.device("meta")` (interfaces.py:288-293,
// utils.py:762). NULL, the default, loads the tower exactly as before.
Qwen3VLWeights LoadQwen3VLWeights(const std::vector<SafetensorsFile>& shards,
                                  const HfConfig& config,
                                  const MultiModalConfig* mm_config = nullptr);

// Load ONLY the vision tower (model.visual.*) into the M2a tower weights (bf16
// widened to f32, matching the M2a dump). Shared by the 4B VL loader above and
// the 27B (Qwen3.6, Qwen3_5ForConditionalGeneration) GDN-hybrid VL path, which
// supplies its own vision config (depth 27, hidden 1152, out_hidden 5120, EMPTY
// deepstack_visual_indexes ⇒ NO deepstack mergers). `vision_cfg.depth` drives the
// block count and `vision_cfg.deepstack_visual_indexes` how many
// deepstack_merger_list.* mergers are read (none for the 27B). The 27B LLM
// backbone is loaded separately by LoadQwen3_5Dense (GDN-hybrid).
multimodal::Qwen3VLVisionWeights LoadQwen3VLVisionWeights(
    const std::vector<SafetensorsFile>& shards,
    const multimodal::Qwen3VLVisionConfig& vision_cfg);

// Single-image, single-sequence GREEDY image->text generation (the M2c gate
// driver). Runs the FULL forked forward: embed(prompt_ids) + merge(mm_embeds) ->
// MRoPE prefill with DeepStack inject at layers 0/1/2 -> greedy argmax -> paged
// decode continuation (MRoPE decode positions, no deepstack). Returns the
// generated token ids (length <= max_new_tokens; stops on eos_token_id).
//
// prompt_ids : the placeholder-expanded model input ids (image_token_id repeated
//              N times at the image span).
// mm_main    : the tower merger output [N, H_text] (== tower_out[:, :out_hidden]),
//              host f32; scattered into the image rows.
// mm_deepstack: the tower multiscale output [N, L*H_text] (== tower_out[:,
//              out_hidden:]), host f32; L = num deepstack levels (3).
// grid_thw   : the LLM-grid source (t,h,w) for get_rope_index.
std::vector<int32_t> Qwen3VLGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::vector<float>& mm_deepstack, int64_t num_deepstack_levels,
    const std::array<int64_t, 3>& grid_thw, int32_t image_token_id,
    int32_t eos_token_id, const Qwen3VLWeights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens);

// Single-VIDEO, single-sequence GREEDY video->text generation (the M3c gate
// driver). Identical forked forward to the image driver above; the two video
// differences are (a) the merge mask is on video_token_id (not image_token_id) and
// (b) the MRoPE prefill positions come from Qwen3VLGetRopeIndexVideo, which scans
// the timestamp-interleaved, per-frame placeholder structure for grid_t frames.
//
// prompt_ids  : the placeholder-expanded model input ids (per-frame timestamp
//               tokens + vision_start + video_token*Nf + vision_end, x grid_t).
// mm_main     : the tower merger output [N, H_text] over ALL video tokens
//               (N = grid_t*(h/merge)*(w/merge)); scattered into the video rows.
// mm_deepstack: the tower multiscale output [N, L*H_text] (L levels; 4B has 3).
// grid_thw    : the video (t,h,w) patch grid for get_rope_index.
std::vector<int32_t> Qwen3VLGenerateGreedyVideo(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::vector<float>& mm_deepstack, int64_t num_deepstack_levels,
    const std::array<int64_t, 3>& grid_thw, int32_t video_token_id,
    int32_t vision_start_token_id, int32_t vision_end_token_id,
    int32_t eos_token_id, const Qwen3VLWeights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens);

// ── MM-ENGINE-FORWARD: the registered engine mm-forward seam ────────────────
//
// A model-owned persistent bf16 cos|sin MRoPE cache (absolute positions 0..8191),
// built ONCE and reused across every forward step. Both the standalone
// Qwen3VLGenerateGreedy driver and the registered forward build it with IDENTICAL
// RopeArgs + Pmax ⇒ bit-identical caches (the ops are deterministic). `storage`
// owns the device buffer (freed on last reference); `tensor` is the view.
struct Qwen3VLCosSinCache {
  std::shared_ptr<void> storage;
  vt::Tensor tensor;  // [8192, rotary_dim] bf16
};

// Build the persistent cos|sin cache (see Qwen3VLCosSinCache). Shared by
// VLGenerateCore and the registered Qwen3-VL LoadedModel (Prepare).
Qwen3VLCosSinCache Qwen3VLMakeCosSinCache(vt::Queue& queue, const HfConfig& config);

// One registered forward STEP: given the already-merged host bf16 embeddings
// [num_tokens*hidden], the 3-D MRoPE positions [3*num_tokens], the (possibly empty)
// DeepStack [levels*num_tokens*hidden], the persistent cos|sin cache, the step
// attention metadata, and the persistent paged KV, run one forked VL forward and
// return the LAST row's logits [vocab] (host f32). This is the EXACT step the M2c
// Qwen3VLGenerateGreedy driver runs (VLForwardLastLogits), exposed so the ENGINE
// registered forward (ForwardQwen3VL) and the standalone driver are numerically
// identical by construction.
std::vector<float> Qwen3VLForwardStepLastLogits(
    vt::Queue& queue, const Qwen3DenseWeights& weights_text, const HfConfig& config,
    const std::vector<uint16_t>& inputs_embeds_bf16,
    const std::vector<int32_t>& positions3, int64_t num_tokens,
    const std::vector<uint16_t>& deepstack_bf16, int64_t deepstack_levels,
    const vt::Tensor& cos_sin_cache_bf16, const v1::CommonAttentionMetadata& meta,
    const std::vector<PagedKvCache>& attn_kv);

// RUNNER-ROUTE: the ON-DEVICE variant of the step above. Identical forward, but the
// last-token [1, vocab] f32 logits stay resident on device — returned as a
// ForwardLogits::on_device() carrier (pool-backed, mirrors qwen3_dense.cpp
// ForwardDevice / WrapDeviceLogits) so the runner / greedy loop samples straight
// off device (vt::GreedyArgmax) with no full-vocab D2H. The registered forward
// (ForwardQwen3VL) returns this on the gather_logits path.
ForwardLogits Qwen3VLForwardStepLastLogitsDevice(
    vt::Queue& queue, const Qwen3DenseWeights& weights_text, const HfConfig& config,
    const std::vector<uint16_t>& inputs_embeds_bf16,
    const std::vector<int32_t>& positions3, int64_t num_tokens,
    const std::vector<uint16_t>& deepstack_bf16, int64_t deepstack_levels,
    const vt::Tensor& cos_sin_cache_bf16, const v1::CommonAttentionMetadata& meta,
    const std::vector<PagedKvCache>& attn_kv);

// Registered-path greedy image->text generation: identical forked forward to
// Qwen3VLGenerateGreedy, but EVERY decoder step is driven through
// ModelRegistry::Forward(model, input) with input.mm carrying the merged
// embeddings / 3-D MRoPE positions / DeepStack — i.e. it exercises the ENGINE
// registered mm-forward (ForwardQwen3VL). Given the same (prompt_ids, tower
// outputs, weights) it returns token-identical output to Qwen3VLGenerateGreedy
// (shared VLGenerateCore step core). `model` must be the registered Qwen3-VL
// LoadedModel loaded from `weights`. This is the entry the MM-SERVE engine seam
// uses and the engine-mm-forward token-exact gate drives.
std::vector<int32_t> Qwen3VLGenerateGreedyViaRegistry(
    LoadedModel& model, const std::vector<int32_t>& prompt_ids,
    const std::vector<float>& mm_main, const std::vector<float>& mm_deepstack,
    int64_t num_deepstack_levels, const std::array<int64_t, 3>& grid_thw,
    int32_t image_token_id, int32_t eos_token_id, const Qwen3VLWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens);

// Compatibility adapter (mirrors MakeQwen3_5DenseLoadedModel): wrap already-loaded
// Qwen3-VL weights in the registered LoadedModel so a caller that owns the weights
// (the M2c/registry e2e gate, the MM-SERVE seam) can drive ModelRegistry::Forward
// without re-reading the checkpoint. The returned model OWNS the moved weights.
std::unique_ptr<LoadedModel> MakeQwen3VLLoadedModel(Qwen3VLWeights weights);

// Borrowing adapter: the returned model does NOT own `weights` (it must outlive
// the model). Used by the registry e2e gate + the MM-SERVE seam, which keep the
// loaded weights alive to ALSO drive the host embed/merge in
// Qwen3VLGenerateGreedyViaRegistry — so the model and the driver share ONE
// Qwen3VLWeights (no multi-GB copy on the unified-memory box).
std::unique_ptr<LoadedModel> BorrowQwen3VLLoadedModel(const Qwen3VLWeights& weights);

}  // namespace vllm
