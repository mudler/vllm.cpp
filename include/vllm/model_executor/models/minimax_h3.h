// MiniMax-H3 (`MiniMaxAI/MiniMax-H3`) — the omni-modal video+audio DIFFUSION
// transformer, ported from vLLM-Omni (vllm-project/vllm-omni). This is the
// project's FIRST diffusion architecture: H3 is not an autoregressive decoder, so
// it has no KV cache, no sampler, and no logits. One request runs a fixed 50-step
// flow-matching denoise loop in which the DiT is forwarded ONCE per step over the
// WHOLE packed sequence, and the resulting latents are decoded to frames + a
// stereo waveform by two VAEs.
//
// ─── HONESTY (up front) ──────────────────────────────────────────────────────
// The real checkpoint is ~354 GB (33.1B DiT + a Qwen3-VL-32B-derived encoder + a
// video VAE + an audio VAE) and its validated serving config is 4x NVIDIA B300 at
// ~133 GB peak per rank. That does not fit this project's hardware (one GB10, 119
// GiB UNIFIED), so there is NO end-to-end token/frame gate for H3 on this box and
// none is claimed. What IS gated here is exact: the layout math, the scheduler,
// and the DiT forward are compared against the UPSTREAM vLLM-Omni modules
// executed at reduced dimensions (scripts/gen-minimax-h3-goldens.py). Structure
// and math are proven; end-to-end generation is hardware-blocked. Full lifecycle,
// component inventory, and the remaining bricks: .agents/specs/minimax-h3.md.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: vllm-project/vllm-omni, vllm_omni/diffusion/models/minimax_h3/
//   OURS                                  <-  UPSTREAM
//   MiniMaxH3DitParams                    <-  minimax_h3_transformer.py:47-78
//                                             (MiniMaxH3DiTArchConfig.from_mapping)
//   MiniMaxH3PatchifyVideoLatent          <-  packed_tokens.py:23-41
//   MiniMaxH3UnpatchifyVideoTokens        <-  packed_tokens.py:44-70
//   MiniMaxH3PackAudioLatent              <-  packed_tokens.py:73-85
//   MiniMaxH3UnpackAudioTokens            <-  packed_tokens.py:88-106
//   BuildMiniMaxH3PackedSequence          <-  packed_sequence.py:116-239
//   BuildMiniMaxH3PackedSequenceRef2va    <-  packed_sequence.py:290-557
//   MiniMaxH3RfVToX0                      <-  scheduling_...euler_ancestral.py:49-69
//   MiniMaxH3EulerEta0Step                <-  scheduling_...euler_ancestral.py:72-102
//   MiniMaxH3DitForward                   <-  minimax_h3_transformer.py:986-1102
//   EnumerateMiniMaxH3DitTensors          <-  minimax_h3_transformer.py:906-922
//   MiniMaxH3ReorderGroupedQkv            <-  minimax_h3_transformer.py:139-168
//   MiniMaxH3DenoiseLoop                  <-  denoise_loop.py:129-239
//   MiniMaxH3TimeShiftSigmas / shape plan <-  time_request.py:5-61,
//                                            pipeline_minimax_h3.py:121-122,
//                                            207-222, 374-434
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/qwen3_5_weights.h"  // Nvfp4Weight (fp4 arm)
#include "vllm/model_executor/models/qwen3_vl_vision.h"  // encoder vision tower reuse
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

// ---------------------------------------------------------------------------
// Architecture
// ---------------------------------------------------------------------------

// MiniMaxH3DiTArchConfig (minimax_h3_transformer.py:47-78). Defaults are the
// SHIPPED MiniMax-H3 geometry; `ParseMiniMaxH3DitParams` overrides any key the
// checkpoint's transformer config actually carries, exactly like `from_mapping`.
struct MiniMaxH3DitParams {
  int64_t num_layers = 50;
  int64_t token_refiner_num_layers = 2;
  int64_t hidden_size = 5376;
  int64_t num_attention_heads = 56;  // MHA: num_kv_heads == num_attention_heads
  int64_t attention_head_dim = 128;
  int64_t ffn_hidden_size = 14336;
  int64_t latents_dim = 24;        // video VAE latent channels
  int64_t audio_latents_dim = 32;  // audio VAE latent channels
  int64_t patch_size_t = 1;
  int64_t patch_size_h = 2;
  int64_t patch_size_w = 2;
  int64_t text_dim = 5120;  // H3-Encoder hidden width (Qwen3-VL layer 50)
  int64_t timestep_input_dim = 256;
  int64_t time_embed_hidden_size = 5376;
  int64_t time_embed_dim = 2688;
  int64_t adaln_out_features = 18 * 5376;      // 6 vectors x 3 modalities x H
  int64_t final_adaln_out_features = 2 * 5376;  // 2 vectors x 1 modality x H
  int64_t rope_inv_freq_len = 16;
  // PRUNED ("curve-form") checkpoints: 0 = the shipped unpruned model, > 0 = the
  // row count of an `adaln_t_table` buffer that REPLACES the time embedder
  // (comfy/ldm/minimax/model.py:409, 419, 428-432). In that form `time_embed_dim`
  // is the curve width (8 in every published pruned checkpoint), not 2688, and
  // `timestep_input_dim` / `time_embed_hidden_size` are unused. See spec 8.20.
  int64_t adaln_curve_grid = 0;
  double norm_eps = 1e-5;
  double qk_norm_eps = 1e-5;
  double final_norm_eps = 1e-5;

  // Derived. video_row_width is the packed video token width
  // (latents_dim * patch volume; 24*1*2*2 = 96 at shipped scale).
  int64_t video_row_width() const {
    return latents_dim * patch_size_t * patch_size_h * patch_size_w;
  }
  // 3D RoPE rotates 6*rope_inv_freq_len of attention_head_dim dims
  // (minimax_h3_transformer.py:207-230; 96 of 128 at shipped scale).
  int64_t rope_rot_dim() const { return 6 * rope_inv_freq_len; }
  // comfy/ldm/minimax/model.py:419 `use_adaln_curves`.
  bool use_adaln_curves() const { return adaln_curve_grid > 0; }
};

// AdaLN modality count: token tags are -1 padding and 0/1/2 for video/text/audio
// (minimax_h3_transformer.py:103-106).
inline constexpr int64_t kMiniMaxH3AdalnModalityNum = 3;

// Packed-sequence token tags (packed_sequence.py:218-221).
inline constexpr int64_t kMiniMaxH3TagPadding = -1;
inline constexpr int64_t kMiniMaxH3TagVideo = 0;
inline constexpr int64_t kMiniMaxH3TagText = 1;
inline constexpr int64_t kMiniMaxH3TagAudio = 2;

// Packed-sequence placeholder input ids (packed_sequence.py:27-35).
inline constexpr int64_t kMiniMaxH3TextId = -5;
inline constexpr int64_t kMiniMaxH3ImgVidCondId = -11;
inline constexpr int64_t kMiniMaxH3AudioRefCondId = -17;
inline constexpr int64_t kMiniMaxH3AudioFirstId = -15;
inline constexpr int64_t kMiniMaxH3AudioId = -14;
inline constexpr int64_t kMiniMaxH3VideoFirstId = -3;
inline constexpr int64_t kMiniMaxH3VideoId = -2;
inline constexpr int64_t kMiniMaxH3VideoLastId = -4;
inline constexpr int64_t kMiniMaxH3PadId = -1;

// Condition anchor timesteps (denoise_loop.py:22-24).
inline constexpr double kMiniMaxH3ImgVidCondTimestep = 0.999;
inline constexpr double kMiniMaxH3AudioRefCondTimestep = 1.0;

// Parse the checkpoint transformer config. Mirrors `from_mapping`: unknown keys
// are ignored, present keys override, and `patch_size` must carry three values.
MiniMaxH3DitParams ParseMiniMaxH3DitParams(const nlohmann::json& config);

// ---------------------------------------------------------------------------
// Latent <-> packed-token conversion (packed_tokens.py). Host-side layout math on
// f32 rows; these move bytes, they do not compute, so they stay off the device.
// ---------------------------------------------------------------------------

// [B,C,T,H,W] -> [B*t*h*w, C*pt*ph*pw] (packed_tokens.py:23-41).
std::vector<float> MiniMaxH3PatchifyVideoLatent(const std::vector<float>& latent, int64_t batch,
                                                int64_t channels, int64_t full_t, int64_t full_h,
                                                int64_t full_w, int64_t patch_t, int64_t patch_h,
                                                int64_t patch_w);

// Inverse of the above (packed_tokens.py:44-70). `rows` is [N, C*pt*ph*pw].
std::vector<float> MiniMaxH3UnpatchifyVideoTokens(const std::vector<float>& rows, int64_t t,
                                                  int64_t h, int64_t w, int64_t channels,
                                                  int64_t patch_t, int64_t patch_h, int64_t patch_w);

// [audio_channel, latent_dim, T] -> [audio_channel*T, latent_dim]
// (packed_tokens.py:73-85).
std::vector<float> MiniMaxH3PackAudioLatent(const std::vector<float>& latent, int64_t audio_channel,
                                            int64_t latent_dim, int64_t steps);

// Inverse (packed_tokens.py:88-106). `rows` is [audio_t, latent_dim].
std::vector<float> MiniMaxH3UnpackAudioTokens(const std::vector<float>& rows, int64_t audio_t,
                                              int64_t audio_channel, int64_t latent_dim);

// ---------------------------------------------------------------------------
// Packed sequence (packed_sequence.py)
// ---------------------------------------------------------------------------

// The structural fields of one CFG branch's packed sequence. Layout:
//   [text L | imgvid_cond C | audio A | video_target V | pad P]
// The used length is padded up to a multiple of 64 and the padding becomes a
// SECOND attention document, so attention never crosses into it.
struct MiniMaxH3PackedSequence {
  int64_t seq_len = 0;
  std::vector<int64_t> input_ids;    // [seq_len] placeholder ids
  std::vector<uint8_t> image_mask;   // [seq_len]
  std::vector<uint8_t> audio_mask;   // [seq_len]
  std::vector<int64_t> img_pos;      // cond rows then target rows
  std::vector<int64_t> audio_pos;    // reference rows then target rows
  std::vector<int64_t> text_pos;     // [0, text_len)
  std::vector<uint8_t> update_mask;  // per img_pos row: is it a denoise target?
  std::vector<uint8_t> audio_update_mask;  // per audio_pos row (ref2va only)
  // [seq_len, 3] (t, h, w) grid in FP64. Kept as double on purpose: the grid is
  // built by fp64 accumulations whose last ulp feeds RoPE, and upstream keeps two
  // deliberately DIFFERENT summation orders for it (see packed_sequence.py:101-113).
  std::vector<double> img_position_ids;
  std::vector<int64_t> token_tags;   // [seq_len]
  std::vector<int32_t> cu_seqlens;   // {0, used, seq_len}
  std::vector<int32_t> document_id;  // [seq_len]; 1 on the padding document
};

// fl2va / t2va layout (packed_sequence.py:116-239). `keyframe_frame_indices` must
// be one of {}, {0}, {-1}, {0,-1} and requires `frame_count` when non-empty.
MiniMaxH3PackedSequence BuildMiniMaxH3PackedSequence(int64_t text_len, int64_t latent_t,
                                                     int64_t latent_h, int64_t latent_w,
                                                     int64_t audio_t, int64_t audio_channel,
                                                     bool include_keyframe_cond,
                                                     const std::vector<int64_t>& keyframe_frame_indices,
                                                     int64_t frame_count);

// One ref2va reference block (packed_sequence.py:290-313).
struct MiniMaxH3RefBlock {
  enum class Kind { kImage, kAudio, kVideoAudio };
  Kind kind = Kind::kImage;
  int64_t ref_audio_t = 0;
  int64_t latent_t = 0;
  int64_t latent_h = 0;
  int64_t latent_w = 0;
};

// General ref2va-family layout (packed_sequence.py:290-557).
MiniMaxH3PackedSequence BuildMiniMaxH3PackedSequenceRef2va(
    int64_t text_len, int64_t latent_t, int64_t latent_h, int64_t latent_w, int64_t audio_t,
    const std::vector<MiniMaxH3RefBlock>& ref_blocks, int64_t audio_channel);

// ---------------------------------------------------------------------------
// Request planning (time_request.py + pipeline_minimax_h3.py shape resolution)
// ---------------------------------------------------------------------------

// Output frame rate is FIXED (pipeline_minimax_h3.py:83).
inline constexpr int64_t kMiniMaxH3Fps = 24;
// Reference-image rescale target (pipeline_minimax_h3.py:87-88).
inline constexpr int64_t kMiniMaxH3ReferenceImageShortEdge = 2048;
inline constexpr int64_t kMiniMaxH3ReferenceImageMultiple = 32;
// Default frame counts when the request names neither duration nor num_frames
// (pipeline_minimax_h3.py:405).
inline constexpr int64_t kMiniMaxH3DefaultFramesT2va = 209;
inline constexpr int64_t kMiniMaxH3DefaultFramesRef2va = 124;
// Default sigma shift scales (pipeline_minimax_h3.py:283-285).
inline constexpr double kMiniMaxH3DefaultVideoShift = 12.0;
inline constexpr double kMiniMaxH3DefaultAudioShift = 3.0;
inline constexpr int64_t kMiniMaxH3DefaultSteps = 50;

// Snap up to the 17n+5 frame boundary (time_request.py:5-12).
int64_t MiniMaxH3AlignFrameCount(int64_t frame_count);
// frame_count -> video latent T (time_request.py:15-18).
int64_t MiniMaxH3VideoLatentT(int64_t frame_count);
// Inverse; T must be 1 or match 5n+2 (time_request.py:21-26).
int64_t MiniMaxH3FrameCountFromVideoLatentT(int64_t out_t);
// Audio latents run at 40 Hz (time_request.py:29-31).
int64_t MiniMaxH3AudioLatentT(double duration_seconds);
// `max(multiple, round(value/multiple)*multiple)` (pipeline_minimax_h3.py:121-122).
int64_t MiniMaxH3AlignMultiple(double value, int64_t multiple);
// Reference image rescale to a 2048 short edge on a 32 grid
// (pipeline_minimax_h3.py:207-222). Returns {width, height}.
std::pair<int64_t, int64_t> MiniMaxH3ReferenceImageShape(int64_t width, int64_t height);

// The rectified-flow time-shifted sigma schedule (time_request.py:34-61):
// base = linspace(1, 0, num_steps); sigma = s*base / (1 + (s-1)*base); duplicate
// consecutive values are collapsed and a terminal 0 is appended when needed.
std::vector<double> MiniMaxH3TimeShiftSigmas(int64_t num_steps, double shift_scale);

// --- reference-video input geometry (reference_video.py) ---
// Only the PURE-MATH half is ported; probe/transcode/decode shell out to ffmpeg
// and share the MP4-muxing dependency decision (spec section 5.2).
inline constexpr int64_t kMiniMaxH3RefVideoShortEdge = 768;
inline constexpr int64_t kMiniMaxH3RefVideoMaxPixels = 768 * 1344;
inline constexpr int64_t kMiniMaxH3RefVideoCanvasMultiple = 32;
inline constexpr int64_t kMiniMaxH3QwenVideoSampleFps = 2;
inline constexpr int64_t kMiniMaxH3QwenTemporalPatch = 2;

// Reference-video canvas: aspect clamp -> 768 short edge -> max-pixel rescale ->
// nearest multiple of 32. Returns {width, height}.
std::pair<int64_t, int64_t> MiniMaxH3ReferenceVideoShape(int64_t width, int64_t height);

struct MiniMaxH3ReferenceVideoSchedule {
  std::vector<int64_t> indices;          // source frame indices to extract
  std::vector<double> block_timestamps;  // one per temporal patch
};

// 24 FPS resampled to the 2 FPS Qwen video rate (duplicate indices dropped), then
// timestamps averaged per temporal patch with the tail padded by REPEATING the last.
MiniMaxH3ReferenceVideoSchedule MiniMaxH3ReferenceVideoFrameSchedule(int64_t frame_count);

// The resolved generation plan for one request.
struct MiniMaxH3ShapePlan {
  int64_t height = 0;
  int64_t width = 0;
  int64_t num_frames = 0;
  int64_t latent_t = 0;
  int64_t audio_t = 0;
};

// `_resolve_shape` (pipeline_minimax_h3.py:393-434). Pass `duration_seconds <= 0`
// and `requested_frames <= 1` to take the per-task default; pass
// `height`/`width` <= 0 to take the aspect-derived default (which needs the
// keyframe aspect for fl2va, hence `image_width`/`image_height`).
MiniMaxH3ShapePlan MiniMaxH3ResolveShape(const std::string& task, double duration_seconds,
                                         int64_t requested_frames, int64_t height, int64_t width,
                                         int64_t image_width, int64_t image_height);

// `_resolve_task` (pipeline_minimax_h3.py:374-391). `requested` may be empty.
std::string MiniMaxH3ResolveTask(const std::string& requested, const std::string& partition,
                                 bool has_image, const std::vector<std::string>& supported_tasks);

// ---------------------------------------------------------------------------
// Checkpoint PARTITION / task guard (the #77 follow-up)
// ---------------------------------------------------------------------------
struct MiniMaxH3T2vaRequest;  // defined below; MiniMaxH3TaskOfRequest takes it by ref
// MiniMax-H3 ships as TWO independently-served DiT partitions and the task MUST
// match the partition upstream loads (recipes/MiniMaxAI/MiniMax-H3.md:50-51,289;
// pipeline_minimax_h3.py:374-391 raises on a mismatch):
//
//   FL2VA  serves {t2va, fl2va}
//   Ref2VA serves {ref2va}
//
// The #70/#74 white render was exactly this: t2va run on the Ref2VA NVFP4
// checkpoint — an out-of-distribution task/partition combination upstream
// rejects. Our driver silently accepted it. This mirrors the raise.
//
// PARTITION DETECTION. The release safetensors carries the served-task set in
// `model_index.json` -> `_minimax_h3` -> {"partition","tasks"}
// (pipeline_minimax_h3.py:279-282). Community GGUF/NVFP4 redistributions STRIP
// that block, and there is NO structural fallback: the FL2VA and Ref2VA DiTs are
// byte-structurally identical — the two real manifests carry the SAME 535 base
// tensor names AND the SAME shapes (ref2va prepends reference rows through the
// SAME video/audio_patch_proj, adding no tensor), so nothing in a stripped file
// discriminates the partition. When the block is gone the partition must be
// DECLARED (an explicit --partition), never guessed.
struct MiniMaxH3PartitionInfo {
  // false => no partition metadata was supplied. The guard is then a no-op: it
  // gates the entry points that LOAD a checkpoint (driver/server), not the pure
  // pipeline-math unit tests that build a request by hand. Every resolver below
  // sets it true.
  bool declared = false;
  // The served partition ("fl2va" | "ref2va"), or "" when a stripped file gave no
  // way to know it. Mirrors `self.partition` (pipeline_minimax_h3.py:281).
  std::string partition;
  // The served-task set. Mirrors `self.supported_tasks` (pipeline:282). Empty when
  // the partition is unknown — then EVERY task is ambiguous and refused.
  std::vector<std::string> supported_tasks;
};

// Read `_minimax_h3.{partition,tasks}` from a parsed model_index.json, mirroring
// pipeline_minimax_h3.py:279-282 exactly. Always sets `declared` true; a missing
// `_minimax_h3` block (or missing keys) yields {true,"",{}}, which the guard
// treats as an unknown partition and refuses rather than guesses.
MiniMaxH3PartitionInfo MiniMaxH3PartitionFromModelIndex(const nlohmann::json& model_index);

// Build partition info from an explicit --partition override, grounded in the
// recipe's one-server-one-partition split (recipes/MiniMaxAI/MiniMax-H3.md:50-51):
// "fl2va" serves {t2va, fl2va}, "ref2va" serves {ref2va}. An empty string yields
// the unknown/declared state (the guard then refuses every task); any other
// non-empty value is refused as an invalid partition name.
MiniMaxH3PartitionInfo MiniMaxH3PartitionFromFlag(const std::string& partition);

// The task a built request ENCODES: non-empty `ref_blocks` => "ref2va", else
// non-empty `keyframe_frame_indices` => "fl2va", else "t2va". The two conditioning
// fields are mutually exclusive (packed layout enforces it), so this is the same
// key the pipeline dispatch already branches on.
std::string MiniMaxH3TaskOfRequest(const MiniMaxH3T2vaRequest& request);

// The task/partition guard — the raise half of `_resolve_task`
// (pipeline_minimax_h3.py:387-390). Throws when `task` is not served by the
// partition ("checkpoint partition 'ref2va' supports {ref2va}, got task='t2va'").
// When the partition is UNKNOWN (stripped community file, no --partition) EVERY
// task is ambiguous — t2va/fl2va are valid only on FL2VA, ref2va only on Ref2VA,
// and the two DiTs are indistinguishable — so it refuses and names the recipe
// lines. A no-op when `info.declared` is false.
void MiniMaxH3CheckTaskPartition(const std::string& task, const MiniMaxH3PartitionInfo& info);

// ---------------------------------------------------------------------------
// Flow-matching scheduler (scheduling_minimax_h3_euler_ancestral.py)
// ---------------------------------------------------------------------------

// x0 = xt + (1 - t) * v (scheduling:49-69). Rectified-flow velocity -> clean sample.
std::vector<float> MiniMaxH3RfVToX0(const std::vector<float>& xt, const std::vector<float>& v,
                                    double timestep);

// ---------------------------------------------------------------------------
// Presentation token tags (presentation.py) — the fl2va vision-span override
//
// The denoise loop requires `token_tags` to already carry the presentation's
// vision spans. A vision block is `<|vision_start|> + pad*count + <|vision_end|>`
// and the WHOLE block — markers included — is tagged VIDEO; tagging only the pads
// shifts every AdaLN modulation index after it. Tokenization stays with the caller.
// ---------------------------------------------------------------------------

struct MiniMaxH3PresentationSpan {
  enum class Kind { kText, kVision };
  Kind kind = Kind::kText;
  int64_t length = 0;  // token count of this span
};

// Token length of a vision block carrying `pad_count` image/video pads.
int64_t MiniMaxH3VisionBlockTokenLength(int64_t pad_count);

// AdaLN token tags aligned with a presentation's span layout.
std::vector<int64_t> MiniMaxH3BuildPresentationTokenTags(
    const std::vector<MiniMaxH3PresentationSpan>& spans);

// ---------------------------------------------------------------------------
// Condition-noise augmentation (condition_noise.py) — fl2va / ref2va
// ---------------------------------------------------------------------------

// Packed row widths (denoise_loop.py:26-28).
inline constexpr int64_t kMiniMaxH3VideoRowWidth = 96;
inline constexpr int64_t kMiniMaxH3AudioRowWidth = 32;
// Channel-major packed audio condition rows are always stereo.
inline constexpr int64_t kMiniMaxH3AudioCondChannels = 2;

// out = noise_aug*clean + (1 - noise_aug)*noise, over packed condition rows.
// `condition_shapes` is a flat list of (latent_t, latent_h, latent_w) triples in
// packed visual-condition order. NOISE IS AN INPUT (see the .cpp for why).
std::vector<float> MiniMaxH3ImgvidCondNoiseAug(const std::vector<float>& clean_rows,
                                               const std::vector<int64_t>& condition_shapes,
                                               int64_t target_latent_t,
                                               int64_t imgvid_cond_num_frames, double noise_aug,
                                               const std::vector<float>& noise_rows);

// The audio-side equivalent; `condition_audio_t` is the latent T of each
// audio-bearing condition in request order.
std::vector<float> MiniMaxH3AudioCondNoiseAug(const std::vector<float>& clean_rows,
                                              const std::vector<int64_t>& condition_audio_t,
                                              double noise_aug,
                                              const std::vector<float>& noise_rows);

// Ancestral Euler with eta = 0 (scheduling:72-102):
//   out = r * state + (1 - r) * denoised,  r = sigma_next / sigma_curr.
// sigma_curr == 0 is the terminal step and returns `state` unchanged.
std::vector<float> MiniMaxH3EulerEta0Step(const std::vector<float>& state,
                                          const std::vector<float>& denoised, double sigma_curr,
                                          double sigma_next);

// ---------------------------------------------------------------------------
// DiT weights + forward
// ---------------------------------------------------------------------------

// Checkpoint tensor names in load order, with their expected shapes. The H3 DiT
// loads by EXACT checkpoint name (minimax_h3_transformer.py:906-922), so this
// enumeration IS the weight contract and is gated structurally without the
// checkpoint.
struct MiniMaxH3TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  // The 12 latent/timestep/output params and the RoPE buffer stay FP32 after load
  // (minimax_h3_transformer.py:85-101, 898-904); everything else is BF16.
  bool fp32 = false;
};

std::vector<MiniMaxH3TensorSpec> EnumerateMiniMaxH3DitTensors(const MiniMaxH3DitParams& params);

// --- GGUF arm (minimax_h3_gguf.cpp) ---
// The ComfyUI-format H3 GGUFs keep the checkpoint's own parameter names, so the
// name map is the IDENTITY against the contract above; only the SHAPES need
// resolving. GGUF `ne` is reversed relative to torch, EXCEPT where ComfyUI had to
// reshape a tensor for quant-block alignment, in which case the true torch shape
// is recorded in `comfy.gguf.orig_shape.<name>` and is used verbatim.
std::vector<int64_t> MiniMaxH3GgufLogicalShape(const std::vector<int64_t>& gguf_dims,
                                               const std::vector<int64_t>& orig_shape);

// A ComfyUI GGUF carries no transformer config, so the SHAPES are the config.
MiniMaxH3DitParams ParseMiniMaxH3DitParamsFromGgufManifest(
    const std::vector<MiniMaxH3TensorSpec>& manifest);

class GgufFile;
// Names + logical shapes + fp32-island flags read out of a GGUF.
std::vector<MiniMaxH3TensorSpec> EnumerateMiniMaxH3GgufTensors(const GgufFile& file);

// ---------------------------------------------------------------------------
// Audio VAE decoder (minimax_h3_audio_vae.cpp)
//
// H3's two VAEs are checkpoint REMOTE CODE loaded under `trust_remote_code`
// (vae.py:41-53); vLLM-Omni only ADAPTS them, so a no-Python engine must
// REIMPLEMENT them. This is the audio side: a DAC-lineage BigVGAN vocoder at
// 32 kHz / 2 channels, gated against the checkpoint's own modules by
// scripts/gen-minimax-h3-audio-vae-goldens.py.
// ---------------------------------------------------------------------------

inline constexpr int64_t kMiniMaxH3AudioSampleRate = 32000;
inline constexpr int64_t kMiniMaxH3AudioChannels = 2;

struct MiniMaxH3AudioVaeConfig {
  int64_t num_mels = 2048;                 // == DacAudioVAE latent_dim
  int64_t upsample_initial_channel = 1024;  // decoder_dim
  std::vector<int64_t> upsample_rates = {5, 5, 2, 2, 2, 2, 2};
  std::vector<int64_t> upsample_kernel_sizes = {9, 9, 4, 4, 4, 4, 4};
  std::vector<int64_t> resblock_kernel_sizes = {3, 7, 11};
  std::vector<std::vector<int64_t>> resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  bool use_tanh_at_final = false;  // H3 CLAMPS instead
  bool use_bias_at_final = false;
  bool snake_logscale = true;
};

// Parameters keyed by their torch state_dict name, so the checkpoint's own
// naming IS the contract (`conv_pre.parametrizations.weight.original0`, ...).
struct MiniMaxH3AudioVaeWeights {
  std::map<std::string, std::vector<float>> tensors;

  const std::vector<float>& Get(const std::string& name) const;
  bool Has(const std::string& name) const { return tensors.count(name) != 0; }
};

// ---------------------------------------------------------------------------
// THE SHARED 1-D BigVGAN PRIMITIVES.
//
// These are published here, next to MiniMaxH3KaiserSincFilter1d and
// MiniMaxH3GroupNorm3d, because MiniMax-H3 is not their only caller: LTX-2.5's
// audio VAE (ltx2_audio_vae.cpp) descends from the same BigVGAN lineage and needs
// exactly this arithmetic. They used to be TU-private to
// minimax_h3_audio_vae.cpp, which forced LTX to stand up a second copy — and a
// second copy of the alias-free trim geometry in particular is the kind of
// duplicate that goes wrong quietly: a fix to the pad/trim arithmetic lands in
// one file, the other keeps its own green gate, and the two audio VAEs disagree
// with nothing to say so. AGENTS.md §"Shared seams" says to extend the seam
// rather than hand-roll a parallel path, so the seam is extended here and there
// is now ONE implementation, gated by BOTH suites.
//
// The `MiniMaxH3` prefix records where they were ported first, not who may call
// them. Signals are CHANNEL-MAJOR [C, T] throughout.
// ---------------------------------------------------------------------------

// One 1-D convolution over [C_in, T] with stride/dilation/groups. Weight is
// [C_out, C_in/groups, K]; the input must ALREADY be padded. Accumulates in
// double, and reports the produced length through `out_len`.
std::vector<float> MiniMaxH3Conv1d(const std::vector<float>& in, int64_t in_channels,
                                   int64_t in_len, const std::vector<float>& weight,
                                   const std::vector<float>* bias, int64_t out_channels,
                                   int64_t kernel, int64_t stride, int64_t dilation,
                                   int64_t groups, int64_t* out_len);

// torch.nn.functional.conv_transpose1d over [C_in, T]. Weight is
// [C_in, C_out/groups, K]; output length is (T-1)*stride - 2*padding + K.
std::vector<float> MiniMaxH3ConvTranspose1d(const std::vector<float>& in, int64_t in_channels,
                                            int64_t in_len, const std::vector<float>& weight,
                                            const std::vector<float>* bias, int64_t out_channels,
                                            int64_t kernel, int64_t stride, int64_t padding,
                                            int64_t groups, int64_t* out_len);

// F.pad along the time axis. `replicate` selects mode="replicate"; false is the
// zero pad an ordinary nn.Conv1d `padding=` argument performs.
std::vector<float> MiniMaxH3Pad1d(const std::vector<float>& in, int64_t channels, int64_t in_len,
                                  int64_t left, int64_t right, bool replicate, int64_t* out_len);

// The stabilizing epsilon in Snake/SnakeBeta's reciprocal, named so it can be
// pinned: upstream writes `1.0 / (beta + 1e-9)` on both sides of this port's
// lineage — LTX-2.5 at audio_vae/vocoder.py:198 (Snake) and :221 (SnakeBeta), and
// MiniMax-H3 in its BigVGAN activation. Mutation proves no reduced-dimension
// golden can tell 1e-9 from 0.0, because beta is O(1) there and never small
// enough for the term to matter; the value still decides whether a real
// checkpoint whose learned beta approaches zero divides or explodes. It is
// therefore held by a source-anchored constant assertion, not by a tensor
// comparison.
inline constexpr double kMiniMaxH3SnakeEps = 1e-9;

// Snake / SnakeBeta: x + (b + kMiniMaxH3SnakeEps)^-1 * sin^2(a * x), in place.
// A null `beta` selects plain Snake, which reuses ALPHA as the reciprocal scale
// (LTX-2.5 vocoder.py:198); a non-null one selects SnakeBeta (vocoder.py:221),
// which is what every MiniMax-H3 checkpoint carries. `logscale` exponentiates
// both, which is how the parameters are stored.
void MiniMaxH3SnakeActivation(std::vector<float>& x, int64_t channels, int64_t length,
                              const std::vector<float>& alpha, const std::vector<float>* beta,
                              bool logscale);

// The anti-aliased activation, `Activation1d`: upsample by `ratio` -> Snake(Beta)
// -> downsample by `ratio`, both through the kaiser-sinc window with REPLICATE
// padding. MiniMax-H3 reaches it through dac_alias_free_act.py +
// dac_alias_free_resample.py; LTX-2.5 through vocoder.py:104-184. The trim
// geometry is the fragile part and the reason this is shared rather than copied.
//
// Build() computes the window once; Apply() is const and may be reused.
struct MiniMaxH3AliasFreeActivation1d {
  int64_t ratio = 2;
  int64_t kernel_size = 12;
  std::vector<float> filter;

  void Build();

  std::vector<float> Apply(const std::vector<float>& in, int64_t channels, int64_t in_len,
                           const std::vector<float>& alpha, const std::vector<float>* beta,
                           bool logscale, int64_t* out_len) const;
};

// kaiser_sinc_filter1d (dac_alias_free_filter.py:26-60) — built at load time, never
// read from the checkpoint.
std::vector<float> MiniMaxH3KaiserSincFilter1d(double cutoff, double half_width,
                                               int64_t kernel_size);

// torch weight_norm: w = g * v / ||v||, norm over every dim except dim 0. Every
// conv in this decoder is weight-normalized, so the checkpoint stores (g, v).
std::vector<float> MiniMaxH3MaterializeWeightNorm(const std::vector<float>& g,
                                                  const std::vector<float>& v,
                                                  int64_t out_channels);

struct StTensor;
class SafetensorsFile;

// Read an unquantized safetensors tensor into f32, whatever its storage dtype
// (F32 / BF16 / F16). Shared by every MiniMax-H3 checkpoint loader.
std::vector<float> MiniMaxH3ReadSafetensorF32(const StTensor& tensor);

// Materialize the audio VAE decoder's weights from the SHIPPED checkpoint
// (FL2VA/audio_vae/model.safetensors).
//
// The file does NOT use the names the decoder reads, and the two mismatches are
// exactly the kind that fail silently, so both are gated against the REAL
// 1087-tensor manifest:
//
//  1. WEIGHT-NORM SPELLING. The checkpoint stores torch's LEGACY weight_norm pair
//     `weight_g` / `weight_v`. The decoder (and the generator that produced its
//     goldens, running a modern torch) reads the PARAMETRIZATION spelling
//     `parametrizations.weight.original0` / `original1`. Same tensors, different
//     era of torch; the loader renames them.
//  2. PREFIX. Every BigVGAN tensor is under `decoder.`, but `dec_in_proj.*` — the
//     Conv1d that runs BEFORE BigVGAN — is at the top level.
//
// The anti-aliasing `.filter` tensors are SKIPPED: those kaiser-sinc filters are
// COMPUTED at load (MiniMaxH3KaiserSincFilter1d), never read from the checkpoint.
// The file also carries the audio ENCODER (`encoder.*`), which generation does not
// need; it is ignored rather than loaded.
MiniMaxH3AudioVaeWeights LoadMiniMaxH3AudioVaeWeights(const SafetensorsFile& file);

// Decode one channel of audio latents to a waveform in [-1, 1]. When the weights
// carry `dec_in_proj` (the checkpoint's Conv1d k=1 from vae_latent_channels to
// num_mels, applied before BigVGAN — dac_audio_vae.py:218-231) the input is
// [vae_latent_channels, frames]; otherwise it is already [num_mels, frames].
std::vector<float> MiniMaxH3AudioVaeDecode(const MiniMaxH3AudioVaeConfig& config,
                                           const MiniMaxH3AudioVaeWeights& weights,
                                           const std::vector<float>& latent, int64_t frames,
                                           int64_t* out_samples);

// ---------------------------------------------------------------------------
// Audio VAE ENCODER (minimax_h3_audio_vae.cpp)
//
// The ANALYSIS half of the same DAC-lineage VAE, and the piece a REFERENCE AUDIO
// needs: ref2va conditions on rows produced from a supplied waveform, so without
// it a `kAudio` block (or a `kVideoAudio` block with `ref_audio_t > 0`) can only
// be refused.
//
// The encode path is not a method on the shipped module — `DacAudioVAE` exposes
// only `decode` (dac_audio_vae.py:211-225). vLLM-Omni composes it by hand
// (vae.py:317-325), and this mirrors that composition exactly:
//
//   preprocess  right-pad to a multiple of hop_length   (dac_audio_vae.py:201-209)
//     -> Encoder     strided conv stack                 (dac_audio_vae.py:90-117)
//     -> pre_block   AttnProjection, when attn_proj     (dac_attn_proj.py:69-88)
//     -> mean_proj   Conv1d(attn_proj_dim -> latent, 1) (dac_audio_vae.py:157)
//
// `mean_proj` and never `logs_proj`: conditioning takes the distribution MEAN, so
// the same reference always produces the same rows — the same rule the video side
// follows in MiniMaxH3VideoVaeEncodeToLatent.
// ---------------------------------------------------------------------------

struct MiniMaxH3AudioVaeEncoderConfig {
  // The SHIPPED geometry (FL2VA/audio_vae/metadata.json + config.yaml), confirmed
  // by the real 1087-tensor manifest.
  int64_t encoder_dim = 64;                            // d_model of the first conv
  std::vector<int64_t> encoder_rates = {2, 4, 4, 5, 5};  // hop_length = 800
  int64_t latent_dim = 2048;                           // d_latent, the last conv's width
  int64_t vae_latent_channels = 32;                    // mean_proj's output width
  bool attn_proj = true;
  int64_t attn_proj_heads = 8;
  double layer_norm_eps = 1e-5;  // torch nn.LayerNorm default

  // np.prod(encoder_rates) (dac_audio_vae.py:148).
  int64_t hop_length() const;
  // dac_audio_vae.py:151-155 — vae_latent_channels when it divides latent_dim,
  // else the next power of two.
  int64_t attn_proj_dim() const;
};

// Encoder.forward (dac_audio_vae.py:116-117) over a MONO waveform, INCLUDING the
// `preprocess` right-pad. Returns [latent_dim, frames]; `frames` is
// padded_samples / hop_length.
std::vector<float> MiniMaxH3AudioVaeEncoderForward(const MiniMaxH3AudioVaeEncoderConfig& config,
                                                   const MiniMaxH3AudioVaeWeights& weights,
                                                   const std::vector<float>& samples,
                                                   int64_t sample_count, int64_t* out_frames);

// AttnProjection.forward (dac_attn_proj.py:85-88) over [tokens, latent_dim] rows,
// returning [tokens, attn_proj_dim]. The narrowing branch it takes is the one the
// checkpoint ships (in_dim > out_dim): causal attention, MEAN over heads, then an
// adaptive average pool down to out_dim.
std::vector<float> MiniMaxH3AudioVaeAttnProjection(const MiniMaxH3AudioVaeEncoderConfig& config,
                                                   const MiniMaxH3AudioVaeWeights& weights,
                                                   const std::vector<float>& tokens,
                                                   int64_t token_count);

// The whole encode of ONE channel: preprocess -> Encoder -> pre_block -> mean_proj.
// Returns the latent MEAN as [vae_latent_channels, frames].
std::vector<float> MiniMaxH3AudioVaeEncodeToLatent(const MiniMaxH3AudioVaeEncoderConfig& config,
                                                   const MiniMaxH3AudioVaeWeights& weights,
                                                   const std::vector<float>& samples,
                                                   int64_t sample_count, int64_t* out_frames);

// vae.py:293-340 end to end: a multi-channel waveform (channel-major, mono is
// REPEATED up to `kMiniMaxH3AudioChannels`) becomes NORMALIZED packed condition
// rows [channels * frames, vae_latent_channels], channel-major — the layout
// BuildMiniMaxH3PackedSequenceRef2va gives an audio-bearing block. `out_audio_t`
// receives `frames`, which is the block's `ref_audio_t`.
//
// `latents_mean` / `latents_std` come from audio_vae/config.json; passing both
// empty skips the normalization (useful in unit tests).
std::vector<float> MiniMaxH3AudioVaeEncodeToRows(const MiniMaxH3AudioVaeEncoderConfig& config,
                                                 const MiniMaxH3AudioVaeWeights& weights,
                                                 const std::vector<float>& waveform,
                                                 int64_t channels, int64_t samples_per_channel,
                                                 const std::vector<float>& latents_mean,
                                                 const std::vector<float>& latents_std,
                                                 int64_t* out_audio_t);

// The ENCODER half of FL2VA/audio_vae/model.safetensors — the half
// LoadMiniMaxH3AudioVaeWeights deliberately skips. `encoder.` is stripped (so the
// forward reads bare `block.N...`), while `pre_block.*` and `mean_proj.*` are kept
// verbatim because they sit at the top level of the file. `logs_proj.*` is NOT
// loaded: conditioning uses the mean, never a sample. The same three weight-norm
// spellings the decoder loader accepts are accepted here.
MiniMaxH3AudioVaeWeights LoadMiniMaxH3AudioVaeEncoderWeights(const SafetensorsFile& file);

// ---------------------------------------------------------------------------
// Video VAE decoder (minimax_h3_video_vae.cpp)
//
// The real 560-tensor manifest splits the video VAE cleanly: the ENCODER is the
// 3D CNN, but the DECODER — the half generation needs — is a 36-block TRANSFORMER.
// This is that block, the repeated unit.
// ---------------------------------------------------------------------------

struct MiniMaxH3VideoVaeBlockConfig {
  int64_t dim = 0;       // embed_dim
  int64_t heads = 0;
  int64_t dim_head = 0;
  int64_t ff_inner = 0;  // w1 emits 2 * ff_inner ([gate | up])
  double eps = 1e-5;
};

// One decoder TransformerBlock (base_module.py:200-281), fp32:
//   h += scale1 * Attention(RMSNorm(h));  h += scale2 * GatedSiLU_FF(RMSNorm(h))
// `scale1`/`scale2` are learned PER-CHANNEL vectors. NOTE the qkv layout is
// PER-HEAD INTERLEAVED ([head][q|k|v]), unlike the DiT's [q_all|k_all|v_all].
// Parameters are looked up by their torch state_dict names under `prefix`.
// `rope_cos`/`rope_sin` are per-TOKEN [seq, rot_dim] (shared across heads) and may
// be null for the no-RoPE path.
std::vector<float> MiniMaxH3VideoVaeBlockForward(const MiniMaxH3VideoVaeBlockConfig& config,
                                                 const MiniMaxH3AudioVaeWeights& weights,
                                                 const std::string& prefix,
                                                 const std::vector<float>& hidden, int64_t seq,
                                                 const float* rope_cos = nullptr,
                                                 const float* rope_sin = nullptr,
                                                 int64_t rot_dim = 0);

// The whole ViT3D decoder (vae_vit.py:216-365). Real hyperparameters from the
// checkpoint's `vit_decoder_kwargs`: 36 layers, 32 heads x 64, RMS norms, qk RMS
// norm WITHOUT affine, gated SiLU, rope_theta 100.0, rope_dim_ratio 0.75.
struct MiniMaxH3VideoVaeDecoderConfig {
  MiniMaxH3VideoVaeBlockConfig block;
  int64_t num_layers = 36;
  int64_t in_channels = 24;   // video latent channels
  int64_t out_channels = 3;   // RGB
  int64_t patch_size = 16;
  int64_t patch_size_t = 4;
  int64_t num_register_tokens = 4;
  int64_t rope_apply_dim = 48;  // int(dim_head * rope_dim_ratio)
  double rope_theta = 100.0;

  // TEMPORAL CHUNKING (klvae.py decode_temporal). The video decode is NOT one
  // pass over the whole latent: upstream feeds the ViT `tokens_chunk_size +
  // token_overlap` temporal tokens at a time. Since the decoder's RoPE is
  // LENGTH-NORMALIZED over the grid it is handed, the temporal extent is part of
  // the input -- a 12-token pass gives every token a position the model never saw.
  // Spatial tiling is a RUNTIME choice, not a checkpoint property: the wrapper
  // reads `vae_decoder_tiling` from caller-supplied config, and neither the VAE
  // config.json nor the source config.json carries a tiling key. Upstream's chunk
  // loop calls `_adaptive_decode`, so tiling composes WITH temporal chunking
  // rather than replacing it. Default ON because it is a no-op below one tile
  // (the tiled path falls through bit-identically) and measurably better above.
  bool decoder_tiling = true;
  int64_t clip_length = 17;   // config `clip_length`
  int64_t token_drop = 3;     // config `token_drop`
  int64_t vae_ratio_t = 4;    // prod(temporal_downsample_factors)

  int64_t tokens_chunk_size() const {
    return (clip_length + vae_ratio_t - 1) / vae_ratio_t;  // ceil
  }
  int64_t frame_pre_padding() const {
    return ((-clip_length) % vae_ratio_t + vae_ratio_t) % vae_ratio_t;
  }
  int64_t token_overlap() const {
    const int64_t c = tokens_chunk_size();
    return c == 0 ? 0 : ((-token_drop) % c + c) % c;
  }
  int64_t frame_overlap() const {
    return std::max<int64_t>(token_overlap() * vae_ratio_t - frame_pre_padding(), 0);
  }
};

// The per-channel latent statistics both VAEs ship in their config.json. The
// pipeline DENORMALIZES with these before decoding; getting them from the config
// (rather than a caller's hardcoded table) is what makes a real run reproducible.
struct MiniMaxH3LatentStats {
  std::vector<float> mean, std_dev;
};

// Parse `audio_vae/config.json` (AutoencoderKLMiniMaxH3Audio) and `vae/config.json`
// (AutoencoderKLMiniMaxH3) into the configs the decoders take, plus their latent
// statistics. Absent keys keep the SHIPPED defaults, matching
// ParseMiniMaxH3DitParams' contract.
MiniMaxH3AudioVaeConfig ParseMiniMaxH3AudioVaeConfig(const nlohmann::json& config,
                                                     MiniMaxH3LatentStats* stats = nullptr);
MiniMaxH3VideoVaeDecoderConfig ParseMiniMaxH3VideoVaeDecoderConfig(
    const nlohmann::json& config, MiniMaxH3LatentStats* stats = nullptr);

struct MiniMaxH3VideoFrameShape {
  int64_t channels = 0, t = 0, h = 0, w = 0;
};

// 3D RoPE tables for one latent grid (RotaryEmbeddingND + create_token_ids).
void MiniMaxH3VideoVaeRope(int64_t latent_t, int64_t latent_h, int64_t latent_w,
                           int64_t num_suffix, int64_t rope_apply_dim, double rope_theta,
                           std::vector<float>* cos_out, std::vector<float>* sin_out);

// --- video-VAE 3D-CNN ENCODER primitives (conv.py, norm.py, vae_cnn.py) ---
// The decoder is a ViT; the ENCODER is this 3D CNN, needed for image/video
// CONDITIONING (fl2va keyframes, ref2va references), not for output frames.

struct MiniMaxH3Conv3dSpec {
  int64_t in_channels = 0, out_channels = 0;
  int64_t t = 0, h = 0, w = 0;
  int64_t kernel_t = 3, kernel_h = 3, kernel_w = 3;
  int64_t pad_t = 1, pad_h = 1, pad_w = 1;
  int64_t stride_t = 1, stride_h = 1, stride_w = 1;
  bool causal = true;  // all temporal padding on the LEFT
};

struct MiniMaxH3Downsample3dConfig {
  int64_t in_channels = 0, out_channels = 0;
  int64_t t = 0, h = 0, w = 0;
  int64_t time_stride = 1;
  int64_t space_stride = 2;
};

// Downsample3D: an ASYMMETRIC one-pixel pad on the right/bottom when the spatial
// stride is 2, then a strided causal conv with padding (1, 0, 0).
std::vector<float> MiniMaxH3Downsample3d(const std::vector<float>& x,
                                         const MiniMaxH3Downsample3dConfig& config,
                                         const std::vector<float>& weight,
                                         const std::vector<float>& bias);

// Causal Conv3d with `reflect` spatial padding; [C,T,H,W] in and out.
std::vector<float> MiniMaxH3CausalConv3d(const std::vector<float>& in,
                                         const MiniMaxH3Conv3dSpec& spec,
                                         const std::vector<float>& weight,
                                         const std::vector<float>* bias);

// GroupNorm whose statistics span the group's channels AND all of time/space.
void MiniMaxH3GroupNorm3d(std::vector<float>& x, int64_t channels, int64_t spatial,
                          int64_t num_groups, const std::vector<float>& weight,
                          const std::vector<float>& bias, double eps);

struct MiniMaxH3ResnetBlock3dConfig {
  int64_t in_channels = 0, out_channels = 0;
  int64_t t = 0, h = 0, w = 0;
  int64_t num_groups = 32;
  double eps = 1e-6;
};

// norm -> SiLU -> conv -> norm -> SiLU -> conv (+ 1x1x1 shortcut when C changes).
std::vector<float> MiniMaxH3ResnetBlock3dForward(const MiniMaxH3ResnetBlock3dConfig& config,
                                                 const MiniMaxH3AudioVaeWeights& weights,
                                                 const std::string& prefix,
                                                 const std::vector<float>& x);

struct MiniMaxH3EncoderFcn3dConfig {
  int64_t ch = 128;
  std::vector<int64_t> ch_mult = {1, 2, 2, 4, 4, 8};
  std::vector<int64_t> space_down = {2, 2, 2, 2, 1, 1};
  std::vector<int64_t> time_down = {1, 2, 2, 1, 1, 1};
  int64_t num_res_blocks = 2;
  int64_t in_channels = 3;
  int64_t z_channels = 24;
  int64_t t = 0, h = 0, w = 0;
  int64_t num_groups = 32;
  double eps = 1e-6;
};

// The whole 3D-CNN encoder level loop. Conditioning-only: a t2va generation path
// never calls it.
std::vector<float> MiniMaxH3EncoderFcn3dForward(const MiniMaxH3EncoderFcn3dConfig& config,
                                                const MiniMaxH3AudioVaeWeights& weights,
                                                const std::vector<float>& x,
                                                MiniMaxH3VideoFrameShape* out_shape);

// Materialize the VIDEO VAE decoder's weights from the shipped checkpoint
// (FL2VA/video_vae/model.safetensors, the gated 560-tensor manifest).
//
// Simpler than the audio VAE: no weight-norm spelling change. The only mapping is
// the `decoder.` prefix, which the ViT3D decoder's own names do not carry. The
// `encoder.*` half (the 3D CNN, conditioning-only) is ignored, and `quant_conv.*`
// belongs to the ENCODER side.
//
// `post_quant_conv.*` IS kept, under its own name, because it is a real step this
// port had not needed before: see MiniMaxH3VideoVaePostQuantConv.
MiniMaxH3AudioVaeWeights LoadMiniMaxH3VideoVaeDecoderWeights(const SafetensorsFile& file);

// The ENCODER half of the same file, for CONDITIONING (fl2va keyframes, ref2va
// references). Strips `encoder.` and KEEPS `quant_conv.*`, which is the encoder's
// output stage rather than the decoder's.
MiniMaxH3AudioVaeWeights LoadMiniMaxH3VideoVaeEncoderWeights(const SafetensorsFile& file);

// Encode frames [in_channels, T, H, W] in [-1, 1] to the conditioning latent
// [z_channels, T', H', W']. Takes the distribution MEAN, not a sample: a sample
// would make one reference image condition differently on every run.
std::vector<float> MiniMaxH3VideoVaeEncodeToLatent(const MiniMaxH3EncoderFcn3dConfig& config,
                                                   const MiniMaxH3AudioVaeWeights& weights,
                                                   const std::vector<float>& frames,
                                                   MiniMaxH3VideoFrameShape* out_shape);

// The AutoencoderKL wrapper's `post_quant_conv` — a Conv3d(latent_ch -> latent_ch,
// kernel 1x1x1), i.e. a per-position channel MIX plus bias, applied to the latent
// BEFORE the decoder proper.
//
// It sits OUTSIDE ViT3DDecoder, which is why the decoder gate (8.9e-8 vs the
// checkpoint's own ViT3DDecoder, whose first op is x_embedder) never covered it and
// nothing in this port applied it. Loading the tensor without applying it would
// have been the worst outcome: a decode that runs, looks reasonable, and is wrong.
//
// `latent` is [channels, t, h, w] contiguous; the result has the same shape.
std::vector<float> MiniMaxH3VideoVaePostQuantConv(const MiniMaxH3AudioVaeWeights& weights,
                                                  const std::vector<float>& latent,
                                                  int64_t channels, int64_t elems_per_channel);

// --- spatial tiling (klvae.py:192-250) ---
// Shipped config: tile_size 256, tile_overlap_min 64, vae_ratio 16.
inline constexpr int64_t kMiniMaxH3VaeTileSize = 256;
inline constexpr int64_t kMiniMaxH3VaeTileOverlapMin = 64;
inline constexpr int64_t kMiniMaxH3VaeRatio = 16;   // prod(space_down)
inline constexpr int64_t kMiniMaxH3VaeRatioT = 4;   // prod(time_down)

struct MiniMaxH3TilePlan {
  std::vector<int64_t> starts;
  std::vector<int64_t> lengths;
  std::vector<int64_t> overlaps;  // one per seam; empty for a single tile
};

// The tile plan along one axis: smallest tile count whose minimum overlaps still
// cover the input, with the leftover slack distributed in whole `vae_ratio` units
// ROUND-ROBIN across the seams.
MiniMaxH3TilePlan MiniMaxH3SplitTiles(int64_t input_len, int64_t tile_size,
                                      int64_t tile_overlap_min, int64_t vae_ratio);

// Linear cross-fade of the last `blend_extent` elements of `a` into the first
// `blend_extent` of `b`, followed by the remainder of `b`.
std::vector<float> MiniMaxH3BlendTiles(const std::vector<float>& a, const std::vector<float>& b,
                                       int64_t blend_extent);

// The video VAE works in IMAGENET-NORMALIZED pixel space, NOT in [-1, 1].
//
// Upstream's wrapper de-normalizes on the way out (`dec*std + mean`, clamp to
// [0,1], then map to [-1,1] -- comfy/ldm/minimax/vae.py:693) and normalizes on the
// way in (`(x+1)/2 - mean) / std` -- vae.py:659). Both sit OUTSIDE ViT3DDecoder,
// which is why the decoder's own 1.19e-07 gate never covered them, exactly like
// post_quant_conv before them.
//
// Skipping the output step feeds ImageNet-normalized values to a writer expecting
// [-1, 1]: the per-channel means differ (0.485/0.456/0.406) so it casts colour, and
// the std of ~0.22 means the true dynamic range is compressed ~4.4x, which reads as
// dark and low-contrast.
inline constexpr float kMiniMaxH3ImagenetMean[3] = {0.485F, 0.456F, 0.406F};
inline constexpr float kMiniMaxH3ImagenetStd[3] = {0.229F, 0.224F, 0.225F};

// Decoder output (ImageNet-normalized, [C,T,H,W]) -> [-1, 1] pixels, in place.
void MiniMaxH3VideoDenormalizePixels(std::vector<float>& frames, int64_t channels,
                                     int64_t per_channel);

// Pixels in [-1, 1] ([C,T,H,W]) -> the ImageNet-normalized space the VAE ENCODER
// expects, in place. The inverse of the above, for reference/keyframe conditioning.
void MiniMaxH3VideoNormalizePixels(std::vector<float>& frames, int64_t channels,
                                   int64_t per_channel);

// Decode a video latent [in_channels, T, H, W] to frames [out_channels, T*pt, H*ps, W*ps].
std::vector<float> MiniMaxH3VideoVaeDecode(const MiniMaxH3VideoVaeDecoderConfig& config,
                                           const MiniMaxH3AudioVaeWeights& weights,
                                           const std::vector<float>& latent, int64_t latent_t,
                                           int64_t latent_h, int64_t latent_w,
                                           MiniMaxH3VideoFrameShape* out_shape);

// --- the DEVICE-RESIDENT video VAE decoder (minimax_h3_video_vae_device.cpp) ---
//
// Same graph as MiniMaxH3VideoVaeDecode, routed to the tuned shared ops with every
// activation on the device. The portable one is a scalar reference; on a 36-layer
// ViT3D it is what made a 256x256 decode time out.
//
// Staging performs two EXACT weight rearrangements (a per-head-interleaved to_qkv
// row permutation, and a fold of each branch's learned per-channel scale into the
// preceding projection) so the whole decoder needs NO new kernels. Held to the
// decoder's tolerance gate rather than bit-equality with the reference: the shared
// ops accumulate in f32 where the reference accumulates in double, and the scale
// fold reassociates that branch's rounding.
struct MiniMaxH3VideoVaeDeviceBlock {
  vt::Tensor norm1, norm2;
  vt::Tensor qkv_weight, qkv_bias;  // rows reordered to [q_all | k_all | v_all]
  vt::Tensor out_weight, out_bias;  // scale1 folded in
  vt::Tensor w1_weight, w1_bias;
  vt::Tensor w2_weight, w2_bias;  // scale2 folded in
};

struct MiniMaxH3VideoVaeDeviceWeights {
  std::vector<std::shared_ptr<void>> storage;  // owns the device allocations
  std::vector<MiniMaxH3VideoVaeDeviceBlock> blocks;  // views into `storage`
  vt::Tensor x_embedder_weight, x_embedder_bias;
  vt::Tensor register_tokens;
  vt::Tensor norm_out_weight, norm_out_bias;
  bool has_norm_out_bias = false;
  vt::Tensor proj_out_weight, proj_out_bias;
  vt::Tensor qk_norm_ones;  // the qk RMSNorm is elementwise_affine=False
};

// Stage the decoder onto `queue`'s device. Every weight must be host-resident f32.
MiniMaxH3VideoVaeDeviceWeights StageMiniMaxH3VideoVaeWeights(
    vt::Queue& queue, const MiniMaxH3VideoVaeDecoderConfig& config,
    const MiniMaxH3AudioVaeWeights& weights);

std::vector<float> MiniMaxH3VideoVaeDecodeDevice(vt::Device device,
                                                 const MiniMaxH3VideoVaeDecoderConfig& config,
                                                 const MiniMaxH3VideoVaeDeviceWeights& staged,
                                                 const std::vector<float>& latent, int64_t latent_t,
                                                 int64_t latent_h, int64_t latent_w,
                                                 MiniMaxH3VideoFrameShape* out_shape);

// The same decode UNDER UPSTREAM'S TILE PLAN, which is what a real canvas needs.
//
// Tiling is not an optional memory strategy here: the ViT3D's RoPE coordinates are
// LENGTH-NORMALIZED over the grid it is handed, so the grid extent is part of the
// input. Upstream always decodes 256-pixel tiles (16 latent units). Decoding a
// larger canvas in one pass gives every token a position the model never saw, and
// shows up as a grid of small squares over otherwise correct frames.
//
// Falls through to the untiled decode, bit for bit, when the plan is a single tile
// (canvas <= tile_size), which is why the reduced-dimension gates never saw this.
// The video decode as upstream actually performs it: TEMPORAL CHUNKS.
//
// `decode_base` routes video through `decode_temporal`, never a single pass. The
// ViT sees `tokens_chunk_size + token_overlap` temporal tokens at a time (7 for
// the shipped config), and since its RoPE is LENGTH-NORMALIZED over the grid it is
// handed, that extent is part of the input -- a whole-latent pass gives every
// token a temporal position the model never saw.
//
// `target_frames` is the request's frame count; upstream center-crops to it
// (trim_output). Pass 0 to keep every decoded frame.
std::vector<float> MiniMaxH3VideoVaeDecodeTemporalDevice(
    vt::Device device, const MiniMaxH3VideoVaeDecoderConfig& config,
    const MiniMaxH3VideoVaeDeviceWeights& staged, const std::vector<float>& latent,
    int64_t latent_t, int64_t latent_h, int64_t latent_w, int64_t target_frames,
    MiniMaxH3VideoFrameShape* out_shape);

std::vector<float> MiniMaxH3VideoVaeDecodeTiledDevice(
    vt::Device device, const MiniMaxH3VideoVaeDecoderConfig& config,
    const MiniMaxH3VideoVaeDeviceWeights& staged, const std::vector<float>& latent,
    int64_t latent_t, int64_t latent_h, int64_t latent_w, MiniMaxH3VideoFrameShape* out_shape);

// ---------------------------------------------------------------------------
// H3-Encoder text tower (minimax_h3_encoder.cpp)
//
// A Qwen3-VL text model with three H3-specific deltas: only the first
// `selected_layer` (50) decoder layers run, the output is UNNORMALIZED (no final
// RMSNorm), and DeepStack visual features are added at the visual token positions
// after each of the first `deepstack.size()` layers.
// ---------------------------------------------------------------------------

inline constexpr int64_t kMiniMaxH3EncoderSelectedLayer = 50;
inline constexpr int64_t kMiniMaxH3EncoderHiddenDim = 5120;

struct MiniMaxH3EncoderConfig {
  int64_t hidden_size = kMiniMaxH3EncoderHiddenDim;
  int64_t num_hidden_layers = 64;
  int64_t selected_layer = kMiniMaxH3EncoderSelectedLayer;
  int64_t num_attention_heads = 40;
  int64_t num_key_value_heads = 8;
  int64_t head_dim = 128;
  int64_t intermediate_size = 17408;
  double rms_norm_eps = 1e-6;
  double rope_theta = 5000000.0;
  std::vector<int64_t> mrope_section = {24, 20, 20};
};

// num_layers = min(config.num_hidden_layers, selected_layer).
int64_t MiniMaxH3EncoderNumLayers(int64_t config_num_hidden_layers, int64_t selected_layer);

// Interleaved M-RoPE cos/sin ([seq, head_dim]) from [3, seq] (t, h, w) positions.
void MiniMaxH3EncoderMrope(const int64_t* positions, int64_t seq, int64_t head_dim,
                           double rope_theta, const std::vector<int64_t>& mrope_section,
                           std::vector<float>* cos_out, std::vector<float>* sin_out);

// The truncated, UNNORMALIZED text tower. `deepstack[i]` is [num_visual, hidden]
// and may be empty; `visual_pos_mask` is [seq] and is required when it is not.
std::vector<float> MiniMaxH3EncoderTextForward(const MiniMaxH3EncoderConfig& config,
                                               const MiniMaxH3AudioVaeWeights& weights,
                                               const std::vector<float>& inputs_embeds,
                                               const int64_t* positions, int64_t seq,
                                               const uint8_t* visual_pos_mask,
                                               const std::vector<std::vector<float>>& deepstack);

// The H3-Encoder in KEEP-QUANT form, materialized from a ComfyUI-format GGUF
// (qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf, 902 tensors, all Q4_K).
//
// This exists because the encoder is 32B: the safetensors loader materializes f32,
// which is ~128 GB and does not fit the box we test on. Keeping the projections in
// their ggml blocks holds the tower at ~14.6 GB, and — as with the DiT — the ggml
// block-quant GEMM carries NO arch gate, so it runs natively where FP4 cannot.
//
// The two fusions the forward needs (q/k/v -> qkv_proj, gate/up -> gate_up_proj)
// are done on the QUANTIZED BYTES. That is sound precisely because ggml rows are
// independent: a row is a whole number of blocks (every K here is a multiple of
// 256), so concatenating whole rows of q, k and v yields a valid block-quant tensor
// whose rows are [q_all | k_all | v_all]. No dequantize/requantize round trip.
struct MiniMaxH3EncoderQuantWeights {
  MiniMaxH3EncoderConfig config;
  std::map<std::string, std::vector<uint8_t>> quant_storage;  // ggml block bytes
  std::map<std::string, std::vector<float>> storage;          // f32 norms/biases
  std::map<std::string, vt::Tensor> views;                    // what the forward binds
  // The ggml type id each kept tensor was stored as. Needed because dequantizing a
  // SINGLE embedding row goes through the ggml entry point, which is keyed by that
  // id rather than by vt::DType.
  std::map<std::string, uint32_t> ggml_type;

  const vt::Tensor& Get(const std::string& name) const;
  bool Has(const std::string& name) const { return views.count(name) != 0; }
};

// `max_layers` truncates the text tower to H3's own min(num_hidden_layers, 50);
// 0 keeps every layer. The GGUF prefixes differ from the safetensors ones
// (`model.layers.N.` here vs `model.language_model.layers.N.` there, and `visual.`
// vs `model.visual.`), which is itself gated.
// Gather embedding rows for `ids` from the (block-quant) `embed_tokens` table.
//
// The table is the single largest tensor in the encoder — [151936, 5120], ~1.5 GB
// even quantized — and a prompt touches a few dozen rows. So this dequantizes ONLY
// the requested rows rather than the table: ggml rows are independent block
// sequences, so a row can be decoded from its own bytes alone.
//
// Returns [ids.size(), hidden_size] f32, which is what the encoder forward takes as
// `inputs_embeds`.
std::vector<float> MiniMaxH3EncoderEmbedTokens(const MiniMaxH3EncoderQuantWeights& weights,
                                               const std::vector<int32_t>& ids);

// The keep-quant encoder staged onto a device: the block bytes uploaded verbatim,
// the f32 norms uploaded as-is. Staged ONCE — a prompt is encoded per request but
// the tower does not change.
struct MiniMaxH3EncoderDeviceWeights {
  std::vector<std::shared_ptr<void>> storage;
  std::map<std::string, vt::Tensor> views;

  const vt::Tensor& Get(const std::string& name) const;
  bool Has(const std::string& name) const { return views.count(name) != 0; }
};

MiniMaxH3EncoderDeviceWeights StageMiniMaxH3EncoderWeights(
    vt::Queue& queue, const MiniMaxH3EncoderQuantWeights& host);

// The DEVICE-resident encoder text tower — H3's conditioning path.
//
// Same graph as MiniMaxH3EncoderTextForward, with the three H3 deltas intact:
// layer truncation to min(num_hidden_layers, selected_layer), the UNNORMALIZED
// output (no final RMSNorm), and DeepStack injection left to the caller.
//
// Reuses the shared ops throughout — the projections go through vt::MatmulBT,
// which dispatches kMatmulBTQuant on the block weights, so a 32B tower runs from
// its ggml blocks with no dequantization. `positions` is [3, seq] (the M-RoPE
// temporal/height/width axes); for a pure text prompt all three are the token
// index. Returns [seq, hidden] f32.
//
// DeepStack (image/video conditioning): `visual_pos_mask` is [seq] (1 at each
// visual-token row) and `deepstack` is one [num_visual, hidden] block per tap.
// After each of the FIRST `len(deepstack)` decoder layers, each block is ADDED to
// the masked rows — the device mirror of MiniMaxH3EncoderTextForward's DeepStack
// and of upstream `MiniMaxH3Qwen3VLTextModel._deepstack_process` (encoder.py:770-800,
// `hidden_states[visual_pos_masks] += visual_embeds`). Text-only prompts pass the
// defaults (no mask, no blocks) and the path is byte-identical to before. The
// MERGED-feature masked_scatter into `inputs_embeds` is the CALLER's job (upstream
// `_encode` does it on inputs_embeds before the tower runs); this forward consumes
// the already-scattered stream, exactly like the host reference.
//
// A BF16 projection is WIDENED to f32 on the device immediately before its GEMM,
// into a scratch buffer reused across layers. That is not a precision choice —
// bf16 -> f32 is EXACT — it is what makes the unquantized (bf16 safetensors) arm
// runnable at all: the activations here are f32 and `vt::MatmulBT` rejects a mixed
// (f32 activation, bf16 weight) pair, while staging the tower as f32 would double
// a 48.8 GiB residency to 97.5 GiB on a 122 GiB UNIFIED pool. Widening per layer
// costs ONE layer's worth of scratch (~2 GiB) and leaves the GEMM inputs
// bit-identical to what an f32-staged tower would have fed it — so the Q4_K_M and
// bf16 arms differ in their WEIGHT BYTES and nothing else.
std::vector<float> MiniMaxH3EncoderTextForwardDevice(
    vt::Queue& queue, const MiniMaxH3EncoderConfig& config,
    const MiniMaxH3EncoderDeviceWeights& weights, const std::vector<float>& inputs_embeds,
    const int64_t* positions, int64_t seq, const uint8_t* visual_pos_mask = nullptr,
    const std::vector<std::vector<float>>& deepstack = {});

MiniMaxH3EncoderQuantWeights LoadMiniMaxH3EncoderFromGguf(const GgufFile& file,
                                                          int64_t max_layers = 0);

// ---------------------------------------------------------------------------
// Encoder VISION tower (image/video conditioning) — REUSE of the shared Qwen3-VL
// front end (`multimodal::Qwen3VLVisionForward` + `Qwen3VLImageProcessor`).
//
// The H3 encoder is a fine-tuned Qwen3-VL, so its ViT is the SAME architecture the
// project already ports; only the config differs. `MiniMaxH3EncoderVisionConfig`
// returns that config, measured from the real encoder GGUF + the shared
// Qwen3.6-27B vision config (state.md 2026-07-25): hidden 1152 / 16 heads / depth
// 27 / intermediate 4304 / out_hidden 5120 / patch 16 / temporal 2 / merge 2 /
// num_position_embeddings 2304. H3 differs from the 27B by carrying 3 REAL
// DeepStack mergers (`visual.deepstack_merger_list.{0,1,2}`). The one value the
// weights-only ComfyUI GGUF does NOT carry is `deepstack_visual_indexes` (WHICH
// text/vision layers the DeepStack taps sit after); the default here is inferred
// (evenly spaced) and must be confirmed against the upstream vision_config for a
// bit-correct DeepStack inject.
multimodal::Qwen3VLVisionConfig MiniMaxH3EncoderVisionConfig();

// Load the encoder GGUF's `visual.*` tower into the shared f32 weight struct the
// Qwen3-VL front end consumes. Mirrors the safetensors `LoadQwen3VLVisionWeights`
// (`qwen3_vl.cpp`) but (a) strips the `visual.` prefix and (b) DEQUANTIZES every
// tensor to f32 via `DequantGgufRowToF32` — the encoder GGUF stores the ViT in
// Q4_K/Q5_K (blocks) with F16 patch_embed/pos_embed. The ComfyUI export reshapes a
// non-256-aligned row (hidden 1152) to `ne0=256`; dequantizing the whole flat
// buffer preserves the row-major `[out,in]` order the tower reads (it indexes every
// weight as a flat buffer with dims taken from the config), so no reshape metadata
// is needed. `cfg.depth` / `cfg.deepstack_visual_indexes.size()` drive the loop.
multimodal::Qwen3VLVisionWeights LoadQwen3VLVisionFromGguf(
    const GgufFile& file, const multimodal::Qwen3VLVisionConfig& cfg);

// ---------------------------------------------------------------------------
// H3-Encoder from the ORIGINAL bf16 release — 14 safetensors shards, 63 GB
// (minimax_h3_encoder_sharded.cpp)
//
// WHY: every H3 render so far conditioned on a Q4_K_M encoder, and nobody had
// measured what that quantization does to the conditioning tensor. Answering it
// needs the SAME prompt encoded by the unquantized tower, which ships as 14
// shards — and `--encoder` only ever accepted a GGUF.
//
// This is a LOADER, not a second forward: it fills the same `views` map
// `MiniMaxH3EncoderDeviceWeights` already binds, over bf16 data instead of ggml
// blocks, and `MiniMaxH3EncoderTextForwardDevice` runs unchanged.
//
// The name map is the one already gated for `LoadMiniMaxH3EncoderWeights`:
// `model.language_model.layers.N.` -> `layers.N.`, with q/k/v and gate/up FUSED
// by row concatenation ([q|k|v], [gate|up]) because the forward slices them that
// way. `model.language_model.norm.weight` and `lm_head.weight` are deliberately
// NOT bound — H3 reads the UNNORMALIZED truncated output.
// ---------------------------------------------------------------------------
class MiniMaxH3ShardedCheckpoint;

// The encoder geometry implied by the shard index's SHAPES alone — no payload is
// read, so this is safe on a checkpoint far larger than RAM and is the answer to
// "do the GGUF and bf16 arms agree on geometry?" without loading either.
// `max_layers` truncates the text tower exactly as the GGUF loader's does.
MiniMaxH3EncoderConfig MiniMaxH3EncoderConfigFromShards(const MiniMaxH3ShardedCheckpoint& ckpt,
                                                        int64_t max_layers = 0);

// Gather `ids`' embedding rows STRAIGHT out of the mmap'd shard holding
// `model.language_model.embed_tokens.weight`. The table is [151936, 5120] — 1.6 GB
// even in bf16 — and a prompt touches a few dozen rows, so nothing is materialized:
// this is the safetensors twin of MiniMaxH3EncoderEmbedTokens' per-row dequantize.
// Returns [ids.size(), hidden] f32.
std::vector<float> MiniMaxH3EncoderEmbedTokensFromShards(const MiniMaxH3ShardedCheckpoint& ckpt,
                                                         const std::vector<int32_t>& ids);

// "This loader actually RAN" counters. A green suite over a path that silently
// fell back to the GGUF loader is a failure mode this codebase has hit before, so
// the streamer is OBSERVABLE and the gate asserts on it. Mirrors
// MiniMaxH3ShardStreamStats.
struct MiniMaxH3EncoderShardStreamStats {
  uint64_t shards_opened = 0;      // shards the checkpoint resolved to
  uint64_t layers_streamed = 0;    // text-tower layers bound
  uint64_t tensors_streamed = 0;   // device views produced
  uint64_t direct_uploads = 0;     // uploaded straight from the mmap, NO host copy
  uint64_t converted_uploads = 0;  // needed one host dtype conversion first
  uint64_t fused_groups = 0;       // qkv / gate_up concatenations done ON DEVICE
  uint64_t bytes_uploaded = 0;     // total device bytes staged
  uint64_t host_peak_bytes = 0;    // largest host conversion buffer alive at once
};

inline MiniMaxH3EncoderShardStreamStats& MutableMiniMaxH3EncoderShardStreamStats() {
  static MiniMaxH3EncoderShardStreamStats s;
  return s;
}
inline MiniMaxH3EncoderShardStreamStats GetMiniMaxH3EncoderShardStreamStats() {
  return MutableMiniMaxH3EncoderShardStreamStats();
}
inline void ResetMiniMaxH3EncoderShardStreamStats() {
  MutableMiniMaxH3EncoderShardStreamStats() = MiniMaxH3EncoderShardStreamStats{};
}

// ★ Stream the bf16 tower STRAIGHT ONTO THE DEVICE, one tensor at a time.
//
// It MUST stream. The box has 122 GiB of UNIFIED memory (host and device share
// ONE pool) and a previous non-streaming H3 loader was OOM-KILLED at anon-rss
// 125 GB. Here the projections stay BF16 on the device (~48.8 GiB for the 50
// layers H3 actually runs, against 97.5 GiB as f32) and are uploaded DIRECTLY out
// of the read-only mmap — a bf16 shard tensor bound for a bf16 device slot needs
// no host buffer at all. Only the norms are widened, and those are [5120] each.
// Each source range goes to MaybeReleaseSourcePages the moment its copy returns.
//
// The FUSIONS are done on the DEVICE: one allocation per fused group, with q, k
// and v uploaded into its row offsets. That keeps the "no host copy" property
// through the one transform this loader performs.
MiniMaxH3EncoderDeviceWeights StreamMiniMaxH3EncoderShardsToDevice(
    vt::Queue& queue, const MiniMaxH3ShardedCheckpoint& ckpt, int64_t max_layers = 0,
    MiniMaxH3EncoderConfig* out_config = nullptr);

// Materialize the H3-Encoder (FL2VA/text_encoder, 14 shards / 1058 tensors) into
// the name map both encoder forwards read.
//
// Unlike the VAE loaders this one TRANSFORMS tensors, it does not merely rename
// them. Two fusions, because the port consumes what vLLM consumes rather than what
// HF ships:
//   * self_attn.{q,k,v}_proj  ->  self_attn.qkv_proj   (row concat, [q|k|v] order)
//   * mlp.{gate,up}_proj      ->  mlp.gate_up_proj     (row concat, [gate|up])
// The VISION tower needs no fusion: HF already ships `attn.qkv` fused, and
// `mlp.linear_fc{1,2}` already match.
//
// Prefixes: `model.language_model.` -> `layers.N....`, `model.visual.` -> stripped.
//
// `max_layers` truncates the text tower, which is H3's own behaviour
// (min(num_hidden_layers, 50) — the file ships 64). 0 keeps every layer.
//
// DELIBERATELY NOT LOADED: `model.language_model.norm.weight`. H3 reads the
// UNNORMALIZED layer-49 output — no final RMSNorm — and loading that tensor would
// imply it is applied. `lm_head.weight` is a logits head the encoder never uses.
MiniMaxH3AudioVaeWeights LoadMiniMaxH3EncoderWeights(const std::vector<SafetensorsFile>& shards,
                                                     int64_t max_layers = 0);

// One vision-tower block (encoder.py:417-481) — the repeated unit of the ViT.
// Unlike the text tower it uses LayerNorm (with bias), a [q_all|k_all|v_all] qkv
// layout, fp32 rotary, NON-CAUSAL attention segmented by `cu_seqlens`, and the
// TANH-approximate GELU.
struct MiniMaxH3VisionBlockConfig {
  int64_t hidden_size = 1152;
  int64_t num_heads = 16;
  int64_t intermediate_size = 4304;
  double eps = 1e-6;
};

std::vector<float> MiniMaxH3VisionBlockForward(const MiniMaxH3VisionBlockConfig& config,
                                               const MiniMaxH3AudioVaeWeights& weights,
                                               const std::string& prefix,
                                               const std::vector<float>& hidden, int64_t seq,
                                               const float* cos, const float* sin,
                                               const int32_t* cu_seqlens, int64_t num_segments);

// The whole vision tower (encoder.py:483-600).
struct MiniMaxH3VisionTowerConfig {
  MiniMaxH3VisionBlockConfig block;
  int64_t depth = 27;
  int64_t patch_size = 16;
  int64_t temporal_patch_size = 2;
  int64_t in_channels = 3;
  int64_t spatial_merge_size = 2;
  int64_t out_hidden_size = 5120;
  int64_t num_position_embeddings = 2304;  // must be a perfect square
  double rope_theta = 10000.0;
  std::vector<int64_t> deepstack_visual_indexes;
};

struct MiniMaxH3VisionTowerResult {
  std::vector<float> merged;                    // [tokens/merge^2, out_hidden_size]
  std::vector<std::vector<float>> deepstack;    // one per deepstack index
};

// Bilinear resample of the learned position grid, then spatial-merge permute.
std::vector<float> MiniMaxH3VisionPosEmbedInterpolate(const std::vector<float>& pos_embed_table,
                                                      int64_t num_grid_per_side, int64_t dim,
                                                      const std::vector<int64_t>& grid_thw,
                                                      int64_t merge_size);

// 2D rotary frequencies in spatial-merge order.
std::vector<float> MiniMaxH3VisionRotary(const std::vector<int64_t>& grid_thw, int64_t merge_size,
                                         int64_t rotary_dim, double theta);

// `patches` is [tokens, in_channels * temporal_patch * patch * patch]; `grid_thw`
// is a flat list of (t, h, w) triples, one per image/video.
MiniMaxH3VisionTowerResult MiniMaxH3VisionTowerForward(const MiniMaxH3VisionTowerConfig& config,
                                                       const MiniMaxH3AudioVaeWeights& weights,
                                                       const std::vector<float>& patches,
                                                       const std::vector<int64_t>& grid_thw);

// The checkpoint stores qkv GROUPED per query group as [q_per_group, k, v]; the
// fused qkv projection wants [q_all, k_all, v_all] (minimax_h3_transformer.py:
// 139-168). H3 is MHA, so heads_per_group == 1.
std::vector<float> MiniMaxH3ReorderGroupedQkv(const std::vector<float>& weight,
                                              int64_t num_query_groups, int64_t heads_per_group,
                                              int64_t head_dim, int64_t in_features);

// PRUNED (curve-form) timestep embedding — comfy/ldm/minimax/model.py:612-615.
//
//   pos = t.clamp(0, 1) * (grid - 1)
//   i0  = floor(pos).clamp(max = grid - 2)
//   out = lerp(table[i0], table[i0 + 1], pos - i0)
//
// The max-clamp on i0 is load-bearing: it keeps t = 1.0 on the LAST interval
// instead of reading table[grid], and the input clamp makes an out-of-range
// timestep (H3 pins condition rows near 1) hold the curve's end value rather
// than extrapolate. Returns [m, dim] row-major fp32.
std::vector<float> MiniMaxH3AdalnCurveEmbed(const float* table, int64_t grid, int64_t dim,
                                            const float* timesteps, int64_t m);

// Non-owning views of every DiT parameter, in the shape the forward consumes.
struct MiniMaxH3DitBlockWeights {
  vt::Tensor norm1;      // [H]
  vt::Tensor norm2;      // [H]
  vt::Tensor qkv_proj;   // [3*heads*Dh, H]
  vt::Tensor q_norm;     // [Dh]
  vt::Tensor k_norm;     // [Dh]
  vt::Tensor out_proj;   // [H, heads*Dh]
  vt::Tensor fc1;        // [2*ffn, H] as [gate; up]
  vt::Tensor fc2;        // [H, ffn]
  vt::Tensor adaln_w;    // [expand*modality*H, time_embed_dim] (blocks only)
  vt::Tensor adaln_b;    // [expand*modality*H]

  // W-FP4a: the fp4 SPEED arm. When the DiT is loaded fp4-RESIDENT (the NVFP4
  // checkpoint kept packed instead of dequantized), these carry the U8-packed
  // E2M1 weight + its E4M3 group-16 scale + f32 global for each quantized
  // projection, and the matching bf16 `vt::Tensor` above is left Empty(). The
  // device forward routes a non-Empty() fp4 weight through
  // dense_nvfp4::MatmulNvfp4W4A16D (Marlin W4A16 on sm_121, the SAME kernel the
  // Laguna routed-expert + dense-Qwen3 NVFP4 arms use) instead of vt::MatmulBT;
  // adaln_b keeps its bias `vt::Tensor` because the fp4 GEMM carries no bias.
  Nvfp4Weight qkv_fp4, out_fp4, fc1_fp4, fc2_fp4, adaln_fp4;
};

struct MiniMaxH3DitWeights {
  vt::Tensor video_patch_proj_w, video_patch_proj_b;
  vt::Tensor audio_patch_proj_w, audio_patch_proj_b;
  vt::Tensor condition_proj_w, condition_proj_b;
  // Unpruned form only: the sinusoidal time embedder's two projections. Empty()
  // when the checkpoint is the pruned/curve form, which carries `adaln_t_table`
  // instead (they are mutually exclusive, and the loader enforces that).
  vt::Tensor time_proj_in_w, time_proj_in_b;
  vt::Tensor time_proj_out_w, time_proj_out_b;
  // PRUNED form only: [adaln_curve_grid, time_embed_dim] fp32, and — like
  // rope.inv_freq — HOST-resident on every path, because the interpolation runs
  // on the host over M <= 4 rows before any kernel does.
  vt::Tensor adaln_t_table;
  vt::Tensor rope_inv_freq;  // [rope_inv_freq_len], fp32
  std::vector<MiniMaxH3DitBlockWeights> refiner;  // no adaln, no rope
  vt::Tensor refiner_final_norm;
  std::vector<MiniMaxH3DitBlockWeights> blocks;
  vt::Tensor final_norm;
  vt::Tensor final_adaln_w, final_adaln_b;
  vt::Tensor video_out_w, video_out_b;
  vt::Tensor audio_out_w, audio_out_b;

  // W-FP4a fp4 SPEED arm (see MiniMaxH3DitBlockWeights): the two quantized
  // projections outside the block loop. condition_proj carries a bias
  // (condition_proj_b, kept above); final_adaln carries final_adaln_b.
  Nvfp4Weight condition_fp4, final_adaln_fp4;
};

// A GGUF-loaded DiT: owned dequantized buffers plus the views the forward takes.
// `storage` must outlive `weights` (the views are non-owning).
struct MiniMaxH3GgufDit {
  MiniMaxH3DitParams params;
  std::map<std::string, std::vector<float>> storage;  // dequantized tensors (f32)
  // KEEP-QUANT residency: the tensor's ggml bytes VERBATIM, plus the vt block
  // dtype they encode. A name appears in exactly one of `storage` /
  // `quant_storage`; the view binder checks both, so a keep-quant load and a
  // dequantizing load produce the same weight-view struct.
  std::map<std::string, std::vector<uint8_t>> quant_storage;
  std::map<std::string, vt::DType> quant_dtype;
  std::map<std::string, uint32_t> quant_ggml_type;  // for a later dequant
  // bf16 bit patterns, when the DiT is loaded for a bf16 GEMM (see the loader's
  // `bf16` flag). Mutually exclusive with `storage` per tensor.
  std::map<std::string, std::vector<uint16_t>> bf16_storage;
  std::map<std::string, std::vector<int64_t>> shapes;
  MiniMaxH3DitWeights weights;
};

// Materialize the DiT from a ComfyUI-format GGUF: derive the geometry from the
// manifest, dequantize every tensor to f32 through the shared GGUF dequant path
// (so the Q2_K/Q3_K/Q4_K families the H3 GGUFs use are covered), and bind the
// forward's views. Missing tensors throw by name rather than reading as zeros.
// `keep_quant` leaves every ELIGIBLE 2-D projection in its ggml block encoding
// instead of dequantizing it to f32. Eligibility is the shared rule
// (KeepQuantDType + K a whole number of blocks); norms, biases and anything
// unported still dequantize, so the forward sees a uniform weight struct either
// way. vt::MatmulBT dispatches a block-typed weight to kMatmulBTQuant on its own
// (ops.cpp:146), so NO call site in the forward changes -- which is exactly why
// this is a loader+staging change rather than a forward rewrite.
//
// This is the arm that makes a quantized H3 run cheap on hardware WITHOUT fp4
// tensor cores: the block-quant GEMM carries no arch gate (cuda_quant_dot.cu has
// no #if at all), unlike every cutlass/marlin/fp4 path.
MiniMaxH3GgufDit LoadMiniMaxH3DitFromGguf(const GgufFile& file, bool keep_quant = false);

// Load the DiT dequantized straight to BF16 — never materializing f32.
//
// This is the THROUGHPUT configuration, and it is what ComfyUI-GGUF effectively does
// (it dequantizes to bf16 and calls F.linear rather than computing in-quant). Our
// in-quant GEMM measured ~103 GFLOP/s because it re-streams the whole weight set per
// SEQUENCE ROW; a bf16 weight goes through the tuned cuBLASLt MatmulBT instead.
//
// It is also the only way this fits: keeping blocks leaves the AdaLN projections
// ineligible (K=2688 is not a whole number of 256-element Q3_K blocks) and they
// dequantize to ~52 GB of f32. Straight to bf16 the whole DiT is ~33 GB.
MiniMaxH3GgufDit LoadMiniMaxH3DitFromGgufBf16(const GgufFile& file);
// Bind the forward's views onto a MiniMaxH3GgufDit's owned buffers. Shared by the
// GGUF and NVFP4 arms: both land on the SAME weight contract.
void BindMiniMaxH3DitViews(MiniMaxH3GgufDit* out);

class SafetensorsFile;
// Materialize the DiT from an NVFP4 safetensors checkpoint. Quantized projections
// carry the compressed-tensors triple (U8 packed FP4 + E4M3 group-16
// `weight_scale` + F32 scalar `weight_scale_2`) and are dequantized through the
// project's existing NVFP4 path; the fp32/bf16 islands are read as-is.
MiniMaxH3GgufDit LoadMiniMaxH3DitFromNvfp4(const SafetensorsFile& file);

// The community `lilcheaty/MiniMax-H3-NVFP4` checkpoints (metadata converted_by
// "Star Ultimate Model Converter Pro") pack the two fp4 elements per byte in the
// OPPOSITE nibble order to the modelopt standard our DequantNvfp4ToBf16 and the
// Marlin W4A16 path assume: element 2i is in the HIGH nibble, 2i+1 in the LOW.
// Read low-first, every adjacent fp4 pair is swapped, scrambling each projection
// matrix so the DiT cannot denoise and every render grids. Verified against the
// coherent FL2VA GGUF (same base weights: islands byte-identical): reading the
// file low-first gives elementwise corr 0.000 vs the GGUF; high-first gives 100%
// sign-agreement over 115M+ weights and corr rising 0.85->0.94 with |w| (the pure
// NVFP4-vs-Q3_K quant-noise floor). Swapping the two nibbles of every packed byte
// at load turns the file's high-first bytes into the standard low-first bytes both
// the bf16 dequant and the Marlin fp4-resident path expect -- one transform fixes
// BOTH arms. Scoped to the H3 NVFP4 loaders; the shared DequantNvfp4ToBf16 stays
// low-first for the modelopt checkpoints (Laguna / DeepSeek-V4 / Qwen3-32B).
// Default ON; VT_H3_NVFP4_LOWNIBBLE=1 opts out (the pre-fix behavior, for A/B).
bool MiniMaxH3Nvfp4HighNibbleFirst();
// Swap the two 4-bit nibbles of every byte in [src, src+n) into dst (sized n).
void MiniMaxH3Nvfp4SwapNibbles(const uint8_t* src, size_t n, uint8_t* dst);

// The per-step inputs of one denoise step (minimax_h3_transformer.py:986-1102).
// Row-major host buffers; the forward stages them onto `device` itself.
struct MiniMaxH3DitInputs {
  int64_t seq_len = 0;
  const float* x = nullptr;            // [seq_len, video_row_width]
  const float* audio_x = nullptr;      // [seq_len, audio_latents_dim]
  const double* img_position_ids = nullptr;  // [seq_len, 3]
  const float* unique_timesteps = nullptr;   // [M]
  int64_t num_unique_timesteps = 0;
  const int64_t* inverse_indices = nullptr;  // [seq_len] -> [0, M)
  const int64_t* token_tags = nullptr;       // [seq_len]
  const float* prompt_embeds = nullptr;      // [text_len, text_dim]
  const int64_t* img_pos = nullptr;
  int64_t num_img_pos = 0;
  const int64_t* audio_pos = nullptr;
  int64_t num_audio_pos = 0;
  const int64_t* text_pos = nullptr;
  int64_t num_text_pos = 0;
  const int64_t* infer_out_pos = nullptr;  // rows the video head reports
  int64_t num_infer_out_pos = 0;
  const uint8_t* update_mask = nullptr;        // [num_infer_out_pos]
  const uint8_t* audio_update_mask = nullptr;  // optional, [num_audio_pos]
  const int32_t* cu_seqlens = nullptr;         // {0, used, seq_len}
  int64_t num_cu_seqlens = 0;
  const int32_t* refiner_cu_seqlens = nullptr;
  int64_t num_refiner_cu_seqlens = 0;
  bool skip_mask_out_condition = false;
};

struct MiniMaxH3DitOutputs {
  std::vector<float> video_logits;  // [num_infer_out_pos, video_row_width]
  std::vector<float> audio_logits;  // [num_audio_pos, audio_latents_dim]
};

// One DiT forward = one denoise step's velocity prediction. `compute_dtype` picks
// the block-stream dtype: kBF16 is the production path (upstream's cast points are
// preserved), kF32 is the parity path the golden suite gates.
// UNPRUNED timestep embedding — the sinusoidal frequency bank (cosine before
// sine) through the two-layer time embedder (minimax_h3_transformer.py:272-285).
// Returns [m, time_embed_dim] row-major fp32. The forward calls this or
// MiniMaxH3AdalnCurveEmbed depending on `params.use_adaln_curves()`; both hand
// back the same shape, which is why nothing downstream branches.
std::vector<float> MiniMaxH3SinusoidalTimeEmbed(vt::Device device, const MiniMaxH3DitParams& params,
                                                const MiniMaxH3DitWeights& weights,
                                                const float* timesteps, int64_t m);

MiniMaxH3DitOutputs MiniMaxH3DitForward(vt::Device device, const MiniMaxH3DitParams& params,
                                        const MiniMaxH3DitWeights& weights,
                                        const MiniMaxH3DitInputs& inputs,
                                        vt::DType compute_dtype);

// --- device-resident forward (brick H3-2b, minimax_h3_device.cpp) -----------
// Owned device copies of every DiT weight, plus the views the device forward
// takes. Upload once, reuse across denoise steps -- the whole point of the
// device path is that the 50-step loop never re-stages weights.
struct MiniMaxH3DitDeviceWeights {
  std::vector<std::shared_ptr<void>> storage;  // owns the device allocations
  // rope.inv_freq is consumed on the HOST (it builds the cos/sin cache before any
  // kernel runs), so it must stay host-resident f32 no matter how the rest is
  // staged. Binding it to device memory segfaults the moment the forward reads it.
  std::vector<float> rope_inv_freq_host;
  // Same rule for the pruned form's curve table: the lerp runs on the host, so
  // the table must stay host-resident whatever the staging path did.
  std::vector<float> adaln_t_table_host;
  MiniMaxH3DitWeights weights;                 // views into `storage`
};

// Stage a host-resident weight set onto `queue`'s device. Every tensor must be
// f32 and host-resident; the returned views live on the device.
// `compute_dtype` picks the WEIGHT storage policy, mirroring upstream's own split
// (MINIMAX_H3_FP32_PARAM_NAMES / _BUFFER_NAMES, minimax_h3_transformer.py:85-101):
//   kF32  -> every weight staged as f32.
//   kBF16 -> the bf16-STORED modules staged as bf16 (condition_proj, every refiner
//            and block norm/attn/mlp/adaln, refiner_final_norm, final_norm,
//            final_adaln), while the fp32 ISLANDS stay f32 (both patch projections,
//            both time-embedder projections, both output heads, rope.inv_freq).
// This is exactly what the bf16 GOLDEN was generated with (the generator's
// to_bf16_weights), so the bf16 device path now matches upstream's weight dtypes
// as well as its activation cast points.
// Stream the DiT from a GGUF STRAIGHT ONTO THE DEVICE as bf16, one tensor at a time.
//
// This exists because peak memory, not total, is what kills a run on a UNIFIED-memory
// box: loading to host bf16 (~33 GB) and then staging to device (~33 GB more) holds
// BOTH at once, and 66 GB of a 122 GiB pool alongside the VAEs and page cache wedged
// the machine hard enough to stop answering ping. Dequantizing and uploading tensor
// by tensor — freeing each host buffer before the next — keeps the peak at the device
// copy plus ONE tensor.
//
// `params` is recovered from the manifest exactly as the loaders do.
MiniMaxH3DitDeviceWeights StreamMiniMaxH3DitToDeviceBf16(vt::Queue& queue, const GgufFile& file,
                                                         MiniMaxH3DitParams* out_params);

// The NVFP4 twin of the above. The non-streaming LoadMiniMaxH3DitFromNvfp4
// materializes every weight as host f32 (~132 GB for this checkpoint) and is an
// OOM kill on a 122 GiB unified box; it stays as the reference loader, and this
// is what a real run uses.
MiniMaxH3DitDeviceWeights StreamMiniMaxH3Nvfp4ToDeviceBf16(vt::Queue& queue,
                                                           const SafetensorsFile& file,
                                                           MiniMaxH3DitParams* out_params = nullptr);

// W-FP4a — the fp4 SPEED twin of the streamer above. Instead of dequantizing each
// U8 projection to bf16 and running vt::MatmulBT, this keeps the compressed-tensors
// FP4 triple RESIDENT (an Nvfp4Weight per quantized projection) so the device
// forward routes those GEMMs through dense_nvfp4::MatmulNvfp4W4A16D — the Marlin
// W4A16 grouped GEMM (sm_121a native FP4 tensor cores) that the Laguna routed-expert
// and dense-Qwen3 NVFP4 arms already use. The fp32 ISLANDS (both patch projections,
// the time embedder, both output heads, rope.inv_freq) and the norms/biases stay
// exactly as the bf16 streamer stages them. The packed weight bytes are held host-
// resident on each Nvfp4Weight and uploaded (once, lazily) by the dispatcher's
// resident-repack on first forward, then freed — so peak device memory is ~1/4 of
// the bf16-dequant arm (the DiT is ~16 GB packed vs ~66 GB bf16). Correctness of
// the Marlin kernel itself is gated on CUDA by test_ops_nvfp4_matmul /
// test_linear_method; this arm is a loader+dispatch wiring, adding NO quant code.
MiniMaxH3DitDeviceWeights StreamMiniMaxH3Nvfp4ToDeviceFp4(vt::Queue& queue,
                                                          const SafetensorsFile& file,
                                                          MiniMaxH3DitParams* out_params = nullptr);

// ---------------------------------------------------------------------------
// The ORIGINAL bf16 release: 13 safetensors shards, 66.3 GB
// (minimax_h3_sharded.cpp + the streamer in minimax_h3_device.cpp)
// ---------------------------------------------------------------------------
//
// Every H3 render so far used a QUANTIZED DiT, and H3 is unusually
// quantization-sensitive (Q3_K_M -> Q4_K_M alone turned a murky lattice into a
// photoreal frame; ComfyUI PR 15298 traces it to the partial split-half RoPE
// producing channel-wise magnitude outliers that corrupt even INT8). Answering
// "what does FULL PRECISION look like?" needs the original release, and every
// DiT loader before this one took a SINGLE file.

// The upstream `MINIMAX_H3_FP32_PARAM_NAMES` / `_BUFFER_NAMES` split
// (minimax_h3_transformer.py:85-101): both patch projections, both time-embedder
// projections, both output heads and `rope.inv_freq` stay FP32 even in a bf16
// stream. This is LOAD-BEARING, not a precision nicety — their activations are
// f32 and `vt::MatmulBT` REJECTS a mixed (f32 activation, bf16 weight) pair, so a
// tensor on the wrong side of this line fails loudly at the first island GEMM.
// Single-sourced here because all four staging paths must agree on it.
bool MiniMaxH3IsFp32IslandTensor(const std::string& name);

// A MULTI-SHARD safetensors checkpoint, resolved through its
// `model.safetensors.index.json` weight map. The index is USED, never guessed
// around: a tensor the index names but whose shard does not contain it throws BY
// NAME instead of being silently skipped (a skipped weight reads as zeros later,
// which is a plausible-looking render rather than an error).
//
// Shape mirrors `LoadMiniMaxH3EncoderWeights(const std::vector<SafetensorsFile>&,
// ...)` — one index over every shard, so a tensor is found wherever it lives.
// Every shard is mmap'd read-only; nothing is materialized at Open() time, which
// is why this is safe on a checkpoint far larger than RAM.
class MiniMaxH3ShardedCheckpoint {
 public:
  // `dir` holds the shards and their index. `model.safetensors.index.json` is the
  // name the H3 release ships; `diffusion_pytorch_model.safetensors.index.json`
  // (the diffusers spelling) is accepted as well. Throws naming `dir` when
  // neither exists.
  static MiniMaxH3ShardedCheckpoint Open(const std::string& dir);
  // Whether `path` is a directory holding one of those indexes — the test a
  // caller with a single `--dit` flag uses to tell a directory from a file.
  static bool IsShardedDir(const std::string& path);

  MiniMaxH3ShardedCheckpoint();
  ~MiniMaxH3ShardedCheckpoint();
  MiniMaxH3ShardedCheckpoint(MiniMaxH3ShardedCheckpoint&&) noexcept;
  MiniMaxH3ShardedCheckpoint& operator=(MiniMaxH3ShardedCheckpoint&&) noexcept;
  MiniMaxH3ShardedCheckpoint(const MiniMaxH3ShardedCheckpoint&) = delete;
  MiniMaxH3ShardedCheckpoint& operator=(const MiniMaxH3ShardedCheckpoint&) = delete;

  // Every tensor the index names, in index order.
  const std::vector<std::string>& Names() const;
  bool Has(const std::string& name) const;
  // Throws BY NAME when the index does not name `name`.
  const StTensor& Get(const std::string& name) const;
  // The shard FILENAME `name` was resolved to — the gateable answer to "did this
  // tensor come out of the right shard?".
  const std::string& ShardOf(const std::string& name) const;
  const std::vector<std::string>& ShardFiles() const;
  const std::string& IndexPath() const;
  size_t ShardCount() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Names + shapes (+ the fp32-island flag) over every shard, the same manifest the
// GGUF and NVFP4 arms build — so `ParseMiniMaxH3DitParamsFromGgufManifest` derives
// the geometry from SHAPES ALONE here too, and a sharded checkpoint and a
// single-file one that hold the same tensors produce IDENTICAL params.
std::vector<MiniMaxH3TensorSpec> EnumerateMiniMaxH3ShardedTensors(
    const MiniMaxH3ShardedCheckpoint& ckpt);

// The REFERENCE (non-streaming) multi-shard loader: materialize every tensor as
// host f32 and bind the forward's views, exactly as LoadMiniMaxH3DitFromNvfp4
// does for the single-file NVFP4 arm. It is the comparison baseline the streamer
// is gated against and the CPU path for small checkpoints; on the REAL 66.3 GB
// release it would need ~132 GB of host f32 and must not be used — that is what
// StreamMiniMaxH3ShardedToDeviceBf16 exists for.
MiniMaxH3GgufDit LoadMiniMaxH3DitFromShards(const MiniMaxH3ShardedCheckpoint& ckpt);

// ★ Stream the ORIGINAL bf16 DiT from its shards STRAIGHT ONTO THE DEVICE, one
// tensor at a time — the multi-shard twin of StreamMiniMaxH3Nvfp4ToDeviceBf16.
//
// It MUST stream. 66.3 GB cannot be materialized on the host and then staged: the
// box has 122 GiB of UNIFIED memory (host and device share ONE pool), and the
// non-streaming NVFP4 loader was already OOM-KILLED at anon-rss 125 GB doing
// exactly that. Peak host anonymous memory here is ONE tensor's conversion buffer
// (freed before the next), and for the real release it is very nearly ZERO: a
// BF16-on-disk tensor bound for a bf16 device slot is uploaded DIRECTLY out of
// the read-only mmap with no host buffer at all, and each source range is
// released (MaybeReleaseSourcePages) the moment its copy returns, so the page
// cache does not accumulate against the same pool the weights live in either.
//
// The fp32 ISLANDS (see MiniMaxH3IsFp32IslandTensor) stay f32; everything else is
// staged bf16. `rope.inv_freq` stays HOST-resident on the returned struct.
MiniMaxH3DitDeviceWeights StreamMiniMaxH3ShardedToDeviceBf16(
    vt::Queue& queue, const MiniMaxH3ShardedCheckpoint& ckpt,
    MiniMaxH3DitParams* out_params = nullptr);

// --- "this loader actually RAN" counters -----------------------------------
// A green suite over a path that silently fell back to another loader is a
// failure mode this codebase has hit before (a kernel guarded to `num_reqs == 1`
// shipped never-executing while the suite stayed green), so the streamer is made
// OBSERVABLE and the gate asserts on it. Mirrors dense_nvfp4_gemm.h's
// Nvfp4W4A16Stats.
struct MiniMaxH3ShardStreamStats {
  uint64_t shards_opened = 0;       // shards the checkpoint resolved to
  uint64_t tensors_streamed = 0;    // tensors uploaded to the device
  uint64_t direct_uploads = 0;      // uploaded straight from the mmap, NO host copy
  uint64_t converted_uploads = 0;   // needed one host dtype conversion first
  uint64_t host_resident = 0;       // kept on the HOST on purpose (rope.inv_freq)
  uint64_t bytes_uploaded = 0;      // total device bytes staged
  uint64_t host_peak_bytes = 0;     // largest host conversion buffer alive at once
};

inline MiniMaxH3ShardStreamStats& MutableMiniMaxH3ShardStreamStats() {
  static MiniMaxH3ShardStreamStats s;
  return s;
}
inline MiniMaxH3ShardStreamStats GetMiniMaxH3ShardStreamStats() {
  return MutableMiniMaxH3ShardStreamStats();
}
inline void ResetMiniMaxH3ShardStreamStats() {
  MutableMiniMaxH3ShardStreamStats() = MiniMaxH3ShardStreamStats{};
}

MiniMaxH3DitDeviceWeights StageMiniMaxH3DitWeights(vt::Queue& queue,
                                                   const MiniMaxH3DitParams& params,
                                                   const MiniMaxH3DitWeights& host,
                                                   vt::DType compute_dtype = vt::DType::kF32);

// The same staging, but DEQUANTIZING block-quant weights to bf16 on the way up
// instead of keeping their blocks.
//
// This is a THROUGHPUT trade, and the measurement behind it is stark: the keep-quant
// GEMM achieves ~103 GFLOP/s on the DiT, which makes a full-quality render a
// ~12-day job. Dequantized to bf16 the projections go through the tuned cuBLASLt
// MatmulBT instead. It costs memory — the DiT is ~33 GB in bf16 against 15.6 GB
// kept-quant — which is affordable where 145 GB of f32 was not.
//
// `gguf` supplies the ggml type ids the block bytes were stored as; a weight with no
// entry is uploaded unchanged.
MiniMaxH3DitDeviceWeights StageMiniMaxH3DitWeightsDequantBf16(
    vt::Queue& queue, const MiniMaxH3DitParams& params, const MiniMaxH3GgufDit& gguf);

// The DEVICE-RESIDENT DiT forward: same graph as MiniMaxH3DitForward, but every
// activation stays in device memory for the whole block stack -- only the inputs
// go up and the selected output rows come back.
//
// NOT bit-identical to the CPU reference, and deliberately so: it reuses the
// tuned SHARED vt:: ops (notably vt::RmsNorm, whose reduction is f32 where the
// reference accumulates in double -- f32 is what upstream torch does). It is
// gated against the SAME upstream goldens, at the same tolerance.
//
// `weights` must be device-resident on `queue`'s device (see
// StageMiniMaxH3DitWeights). compute_dtype must be kF32 for now; the bf16 stream
// policy is a follow-up (see .agents/specs/minimax-h3.md).
MiniMaxH3DitOutputs MiniMaxH3DitForwardDevice(vt::Queue& queue,
                                              const MiniMaxH3DitParams& params,
                                              const MiniMaxH3DitWeights& weights,
                                              const MiniMaxH3DitInputs& inputs,
                                              vt::DType compute_dtype);


// ---------------------------------------------------------------------------
// Denoise loop (denoise_loop.py:129-239)
// ---------------------------------------------------------------------------

// Static per-branch state: the packed layout plus the fixed forward inputs.
struct MiniMaxH3DenoiseBranch {
  MiniMaxH3PackedSequence packed;
  std::vector<float> text_embeddings;  // [text_len, text_dim]
  std::vector<int64_t> token_tags;     // seq_len, with fl2va vision overrides applied
};

// Runs the CFG-distilled loop: one positive forward per step, video and audio
// target rows chained through the Euler-eta0 update while pinned condition rows
// are reset to their anchors every step. Returns the final (video, audio) rows.
struct MiniMaxH3DenoiseResult {
  std::vector<float> video_rows;  // [num_img_pos, video_row_width]
  std::vector<float> audio_rows;  // [num_audio_pos, audio_latents_dim]
};

MiniMaxH3DenoiseResult MiniMaxH3DenoiseLoop(
    vt::Device device, const MiniMaxH3DitParams& params, const MiniMaxH3DitWeights& weights,
    const MiniMaxH3DenoiseBranch& branch, const std::vector<float>& initial_video_rows,
    const std::vector<float>& initial_audio_rows, const std::vector<float>& keyframe_cond_rows,
    const std::vector<float>& audio_ref_rows, const std::vector<double>& sigmas_video,
    const std::vector<double>& sigmas_audio, vt::DType compute_dtype,
    const MiniMaxH3DitDeviceWeights* prestaged = nullptr);


// ---------------------------------------------------------------------------
// t2va pipeline assembly (minimax_h3_pipeline.cpp)
//
// The wiring from prompt embeddings to frames + waveform. Every stage it calls is
// separately ported and gated; this composes them.
// ---------------------------------------------------------------------------

struct MiniMaxH3T2vaRequest {
  int64_t text_len = 0;
  int64_t latent_t = 0, latent_h = 0, latent_w = 0;
  int64_t audio_t = 0;
  int64_t audio_channel = kMiniMaxH3AudioChannels;
  // The frame count the request asked for. The temporal decode produces whole
  // chunks and upstream center-crops to this (trim_output); 0 keeps every frame.
  int64_t num_frames = 0;
  int64_t num_steps = kMiniMaxH3DefaultSteps;

  // The served checkpoint partition, for the task/partition guard mirrored from
  // upstream `_resolve_task` (pipeline_minimax_h3.py:374-391). MiniMaxH3GenerateT2va
  // refuses a task the partition does not serve (t2va/fl2va need FL2VA, ref2va needs
  // Ref2VA). Default-constructed (`declared=false`) leaves the guard INACTIVE, so a
  // direct caller building a request by hand is unaffected; the checkpoint-loading
  // entry points (driver/server) fill it via MiniMaxH3PartitionFromFlag /
  // MiniMaxH3PartitionFromModelIndex. See the #77 follow-up.
  MiniMaxH3PartitionInfo partition;

  // --- fl2va KEYFRAME CONDITIONING (empty => plain t2va) ---
  // Which generated frames the supplied keyframes pin. Upstream allows exactly
  // {}, {0}, {-1} or {0, -1}: first frame, last frame, or both.
  std::vector<int64_t> keyframe_frame_indices;
  // The conditioning rows, already VAE-encoded and patchified into the packed
  // layout -- see MiniMaxH3EncodeKeyframeCondRows.
  std::vector<float> keyframe_cond_rows;
  // Condition-noise augmentation: out = aug*clean + (1-aug)*noise. 1.0 pins the
  // keyframe exactly; upstream lowers it to let the model deviate.
  double imgvid_noise_aug = 1.0;

  // --- ref2va REFERENCE BLOCKS (empty => no references) ---
  // Mutually exclusive with keyframe_frame_indices: fl2va pins frames of the
  // OUTPUT, ref2va prepends whole reference blocks. When non-empty the packed
  // layout is built by BuildMiniMaxH3PackedSequenceRef2va and the visual rows come
  // from the same pinned-condition mechanism keyframes use.
  std::vector<MiniMaxH3RefBlock> ref_blocks;
  // The AUDIO conditioning rows for the audio-bearing reference blocks, in block
  // order and channel-major within a block -- see MiniMaxH3EncodeReferenceAudio.
  // Exactly `sum(ref_audio_t) * audio_channel` rows of `audio_latents_dim`; a
  // block that claims audio rows without supplying them is refused, because the
  // layout would grow around rows nothing ever wrote.
  std::vector<float> audio_ref_rows;
  double video_shift = kMiniMaxH3DefaultVideoShift;
  double audio_shift = kMiniMaxH3DefaultAudioShift;
  // Per-channel latent statistics from each VAE's config.json; empty skips the
  // denormalization (useful in unit tests).
  std::vector<float> video_latents_mean, video_latents_std;
  std::vector<float> audio_latents_mean, audio_latents_std;
};

struct MiniMaxH3T2vaResult {
  std::vector<float> frames;  // [C, T, H, W]
  MiniMaxH3VideoFrameShape frame_shape;
  std::vector<float> waveform;  // channel-major, audio_samples_per_channel each
  int64_t audio_channels = 0;
  int64_t audio_samples_per_channel = 0;
  int64_t sample_rate = 0;
};

// Run the whole t2va path. NOISE IS AN INPUT: upstream seeds a torch CPU
// generator, and reproducing torch's RNG bit-exactly decides WHICH sample you get
// rather than whether the pipeline is right, so the caller supplies it.
// --- video output (minimax_h3_mux.cpp) ---
// `/v1/videos` returns MP4. Upstream shells out to ffmpeg; this LIBRARY does not
// spawn processes (no precedent in src/vllm/), so it produces the ARTIFACTS and
// BUILDS the argv, and the example/server layer performs the invocation.

// One frame as binary PPM (P6, 8-bit RGB); ffmpeg reads it natively.
std::string MiniMaxH3WritePpmFrame(const std::vector<float>& frames,
                                   const MiniMaxH3VideoFrameShape& shape, int64_t frame_index);

struct MiniMaxH3MuxRequest {
  std::string frame_pattern;  // printf-style, e.g. ".../frame_%06d.ppm"
  std::string audio_path;     // WAV path; empty for a silent clip
  std::string output_path;    // .mp4
  int64_t fps = kMiniMaxH3Fps;
  int64_t crf = 18;
};

// The argv the server layer runs. Pure string assembly — never spawns anything.
std::vector<std::string> MiniMaxH3BuildMp4MuxArgs(const MiniMaxH3MuxRequest& request);

// Serialize a decoded waveform as RIFF/WAVE 16-bit PCM. The VAE's output is
// CHANNEL-MAJOR; this interleaves it. Dependency-free, and required under either
// outcome of the open MP4/muxer dependency decision.
std::string MiniMaxH3WriteWav(const std::vector<float>& waveform, int64_t channels,
                              int64_t samples_per_channel, int64_t sample_rate);

// The inverse, for REFERENCE AUDIO: parse 16-bit PCM RIFF/WAVE into the same
// CHANNEL-MAJOR float layout, mono REPEATED up to `want_channels` and anything
// wider truncated (vae.py:305-313). `want_sample_rate` > 0 REFUSES a mismatch
// rather than resampling — encoding at the wrong rate would shift every latent
// frame, and this library has no audio dependency to resample with.
std::vector<float> MiniMaxH3ReadWav(const std::string& bytes, int64_t want_channels,
                                    int64_t want_sample_rate, int64_t* out_samples_per_channel);

MiniMaxH3T2vaResult MiniMaxH3GenerateT2va(vt::Device device, const MiniMaxH3T2vaRequest& request,
                                          const MiniMaxH3DitParams& dit_params,
                                          const MiniMaxH3DitWeights& dit_weights,
                                          const MiniMaxH3VideoVaeDecoderConfig& video_config,
                                          const MiniMaxH3AudioVaeWeights& video_weights,
                                          const MiniMaxH3AudioVaeConfig& audio_config,
                                          const MiniMaxH3AudioVaeWeights& audio_weights,
                                          const std::vector<float>& prompt_embeds,
                                          const std::vector<float>& initial_video_rows,
                                          const std::vector<float>& initial_audio_rows,
                                          vt::DType compute_dtype,
                                          // Weights already staged on `device`. Staging the DiT
                                          // costs tens of seconds, so a driver or server stages
                                          // ONCE and passes it here rather than per generation.
                                          // Null stages internally, as before.
                                          const MiniMaxH3DitDeviceWeights* prestaged = nullptr);

// The DENOISE half of `MiniMaxH3GenerateT2va` on its own: packed layout, the two
// sigma schedules, and the step loop, stopping before the VAEs.
//
// `MiniMaxH3GenerateT2va` is implemented in terms of this, so there is one copy of
// the layout and schedule logic rather than two. Exposed separately because the DiT
// and the VAE decoders have very different cost profiles, and a caller measuring or
// debugging the step loop should not have to load, or find memory for, either VAE.
// Encode supplied KEYFRAME IMAGES (each [in_channels, H, W] in [-1, 1]) into the
// packed conditioning rows the denoise loop pins. Each image is encoded as a
// one-frame causal clip, then patchified with the DiT's patch volume so the rows
// are interchangeable with the video rows the loop carries.
// Encode REFERENCE IMAGES for ref2va: the same VAE encode + patchify as keyframes,
// but it also reports the per-image latent geometry, because a ref2va block must
// declare the shape it occupies in the packed layout.
//
// Returns the packed condition rows; `out_blocks` receives one kImage block per
// image, in the same order.
// Encode a REFERENCE VIDEO (a clip, [in_channels, T, H, W] in [-1, 1]) for ref2va.
//
// Needs no new porting beyond the image path: the 3D CNN is CAUSAL in time, so a
// clip is the same call with t > 1. Emits a kVideoAudio block -- the only kind that
// carries a temporal extent, since kImage counts exactly one frame however large
// its latent_t -- with `ref_audio_t = 0`, i.e. a SILENT video reference, which is
// the part that can be honoured without the (unported) audio-VAE encoder.
std::vector<float> MiniMaxH3EncodeReferenceVideo(
    const MiniMaxH3EncoderFcn3dConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const MiniMaxH3DitParams& dit_params,
    const std::vector<float>& frames, int64_t frame_count, int64_t frame_h, int64_t frame_w,
    MiniMaxH3RefBlock* out_block);

std::vector<float> MiniMaxH3EncodeReferenceImages(
    const MiniMaxH3EncoderFcn3dConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const MiniMaxH3DitParams& dit_params,
    const std::vector<std::vector<float>>& images, int64_t image_h, int64_t image_w,
    std::vector<MiniMaxH3RefBlock>* out_blocks);

// Encode a REFERENCE WAVEFORM (channel-major, already at
// kMiniMaxH3AudioSampleRate) into the packed audio conditioning rows, and report
// the kAudio block it occupies. The audio counterpart of
// MiniMaxH3EncodeReferenceImages, and what makes an audio-bearing ref2va request
// expressible: the rows belong in MiniMaxH3T2vaRequest::audio_ref_rows.
//
// For a VIDEO+AUDIO reference, encode the clip with MiniMaxH3EncodeReferenceVideo
// and copy the `ref_audio_t` this reports onto that block: one kVideoAudio block
// then carries both, which is the layout packed_sequence.py builds.
std::vector<float> MiniMaxH3EncodeReferenceAudio(
    const MiniMaxH3AudioVaeEncoderConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const std::vector<float>& waveform,
    int64_t channels, int64_t samples_per_channel, const std::vector<float>& latents_mean,
    const std::vector<float>& latents_std, double noise_aug,
    const std::vector<float>& noise_rows, MiniMaxH3RefBlock* out_block);

std::vector<float> MiniMaxH3EncodeKeyframeCondRows(
    const MiniMaxH3EncoderFcn3dConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const MiniMaxH3DitParams& dit_params,
    const std::vector<std::vector<float>>& images, int64_t image_h, int64_t image_w,
    int64_t target_latent_t, double noise_aug, const std::vector<float>& noise_rows);

MiniMaxH3DenoiseResult MiniMaxH3DenoiseT2va(vt::Device device, const MiniMaxH3T2vaRequest& request,
                                            const MiniMaxH3DitParams& dit_params,
                                            const MiniMaxH3DitWeights& dit_weights,
                                            const std::vector<float>& prompt_embeds,
                                            const std::vector<float>& initial_video_rows,
                                            const std::vector<float>& initial_audio_rows,
                                            vt::DType compute_dtype,
                                            const MiniMaxH3DitDeviceWeights* prestaged = nullptr);

}  // namespace vllm
