// A TINY, COMPLETE dots3-note checkpoint on disk — language tower + DENSE
// vision tower — shared by the W6a tower gate and the W6a served-request gate
// (#2512).
//
// WHY A SHARED HEADER AND NOT A COPY. The two gates must agree on the geometry
// byte-for-byte: the tower gate measures the arithmetic at a grid, and the
// server gate asserts the PLACEHOLDER COUNT that same grid implies. Two
// hand-typed copies of "8x8 image, patch 2, merge 2 -> 4 placeholder tokens" are
// one edit away from disagreeing, and the disagreement would show up as a
// passing tower gate beside a server gate measuring a different model.
//
// WHY IT IS NEW RATHER THAN EXTRACTED FROM `test_dots3_note_attn.cpp`. That
// file's checkpoint builder is 5867 lines deep in one anonymous namespace and
// carries W3's double-precision attention reference with it. Lifting it would be
// a refactor of four bricks' evidence in a brick that is adding a tower. This
// header builds only what a LOAD needs, which is much less.
//
// The language geometry mirrors `test_dots3_note_attn.cpp`'s device bench and
// for the same measured reason (its review finding F1): `q_lora` 3 and
// `kv_lora` 2 over `hidden` 16 give the two §4-trap-5 rescales sqrt(16/3) and
// sqrt(16/2), which are DIFFERENT from each other and both far from 1, so a
// dropped or swapped rescale cannot hide.
#ifndef VLLM_TESTS_DOTS3_NOTE_TINY_FIXTURE_H_
#define VLLM_TESTS_DOTS3_NOTE_TINY_FIXTURE_H_

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "vt/dtype.h"

namespace dots3_tiny {

// ── the geometry ────────────────────────────────────────────────────────────
//
// Every dimension is the smallest that still exercises the branch it stands
// for:
//   * vision `head_dim` 8, so the 2-D rope has TWO frequencies per spatial axis
//     (`head_dim/2 = 4` split into a height half and a width half,
//     vision.py:518-519 @ 9035151d6). At `head_dim` 4 there would be one
//     frequency per axis and a swapped height/width axis could not show.
//   * TWO dense vision blocks, so a block that read the previous block's
//     weights would move the answer.
//   * a 4x4 patch grid over a 2x2 merge, so the adapter folds FOUR rows into
//     one and produces FOUR merger rows. One row would make a mis-sized scatter
//     invisible.
//   * `intermediate_size` 6 against `embed_dim` 16, so the SwiGLU is not square
//     and a transposed gate/up merge refuses on shape instead of computing.
struct TinySpec {
  // text tower
  int64_t hidden = 16;
  int64_t heads = 2;
  int64_t qk_nope = 4;
  int64_t qk_rope = 4;
  int64_t v_head = 8;
  int64_t q_lora = 3;
  int64_t kv_lora = 2;
  int64_t layers = 2;
  int64_t vocab = 17;
  int64_t inter = 10;
  int64_t max_pos = 64;
  int64_t index_topk = 32;
  int64_t index_n_heads = 2;
  int64_t index_head_dim = 6;
  double rope_theta = 137.0;
  double rms_eps = 1e-3;

  // vision tower
  bool with_vision = true;
  int64_t v_embed = 16;
  int64_t v_heads = 2;
  int64_t v_layers = 2;
  int64_t v_inter = 6;
  int64_t v_patch = 2;
  int64_t v_merge = 2;
  int64_t v_channels = 3;
  int64_t v_temporal = 1;
  double v_rms_eps = 1e-3;
  // `pyramid_num_routed` written into the config. All-negative means "no MoE at
  // all"; a case that wants a PYRAMID block sets one entry positive, and W6b
  // computes it instead of refusing.
  std::vector<int64_t> v_pyramid{-1, -1};
  // The routed expert width, `moe_intermediate_size` of the `vision_config`.
  // Deliberately NOT `v_inter`: an expert whose width equalled the dense
  // SwiGLU's would let a routed block read a dense block's operand and still
  // pass every shape check.
  int64_t v_moe_inter = 4;
  // `capacity_factor`. `topk = min(int(capacity_factor), num_routed)`
  // (vision.py:190 @ 9035151d6), so 2 selects TWO experts — the released value.
  double v_capacity_factor = 2.0;
  // The processor's PIXEL BUDGET, written into `preprocessor_config.json`.
  // The defaults leave `Dots3NoteResizedSize` free to round each side to a
  // multiple of `factor` with no clamp, which is what the conformant fixture
  // image wants.
  //
  // W6c's DOWNSCALE cases lower `p_max_pixels` to `kBudgetMaxPixels` instead of
  // shipping a large image, because the clamp is the only way to force a resize
  // RATIO big enough for PIL's support scaling to dominate. Rounding alone
  // never can: `out = round(in / factor) * factor` moves a side by less than
  // `factor`, so the largest downscale it can produce at all is
  // `1.5 - 1/factor`, and on this fixture's `factor` of 4 the sizes it actually
  // produces are UPSCALES, where `filterscale = max(1, in/out)` is exactly 1
  // and the resampler is the four-tap kernel it is NOT there for.
  // `kBigImageH` x `kBigImageW` under that budget is a 6x downscale on both
  // axes -- see the constants below.
  int64_t p_min_pixels = 16;
  int64_t p_max_pixels = 1 << 20;
  std::string v_router_scoring_func = "sigmoid";
  double v_router_scale = 1.0;
  // The AMPLITUDE of the generated `mlp.router_bias` values, and the ONLY
  // reason it is settable. Two checkpoints that differ in this and in nothing
  // else are the served gate's handle on the router: every other tensor is
  // drawn from the same seed stream, so a difference in what the server answers
  // can only have come through the routing decision.
  double v_router_bias_amp = 0.4;
  // The ROUTER'S OWN SEED OFFSET, and the only reason it is settable. It is
  // added to the two seeds that draw `mlp.gate_weight` and `mlp.router_bias`
  // and to nothing else, and the shared `next()` stream still advances once per
  // tensor, so changing it moves the ROUTING DECISION and leaves every other
  // tensor in the checkpoint byte-for-byte where it was.
  //
  // WHY IT IS NOT 0. At 0 this fixture's 16 tokens routed 13 to {1,2} and 3 to
  // {2,3}: expert 0 was NEVER selected, expert 2 was in every set, and the
  // set assertion's discriminating population was three tokens. Expert 0 is the
  // index an off-by-one lands on, so a fixture that never selects it is the
  // weakest possible arrangement of the strongest assertion in the gate. The
  // value below was SEARCHED over the tower this fixture actually builds — the
  // search and the result are spec section 4.12.9 — for the first offset whose
  // 16 tokens select every expert, produce more than two of the six possible
  // pairs, and keep the minimum decision margin above the gate's 1e-3 bound.
  uint64_t v_router_seed_nudge = 42;
  std::string v_adapter_type = "patch_merger";
  bool v_pre_pixel_shuffle = true;
  bool v_post_norm = true;
  bool v_use_qk_norm = true;
  bool v_use_bias = false;
  bool v_is_causal = false;
  // THE TWO ADAPTER KEYS A CONFORMANT CHECKPOINT TIES TO SOMETHING ELSE, and
  // the only reason either is settable. `adapter_out_dim` is the TEXT hidden
  // size because the encoder's rows are scattered into the prompt; and
  // `adapter_merge_size` is `spatial_merge_size` because the PROMPT side
  // expands placeholders by that key while the tower folds by this one. A case
  // that wants the install-time refusal for a checkpoint whose keys disagree
  // sets one of these positive. Zero means "follow the conformant value".
  int64_t v_adapter_out_override = 0;
  int64_t v_adapter_merge_override = 0;

  // ── the AUDIO tower (W7a, #2703) ───────────────────────────────────────────
  //
  // Every dimension is the smallest that still exercises the branch it stands
  // for, and three of them are load-bearing for a specific defect:
  //
  //   * `a_mels` 16 folds to `freq_after` 2 (16 -> 8 -> 4 -> 2), NOT 1. At 1 the
  //     regroup `x.permute(0,3,1,2).reshape(B, T, C*F)` has a single frequency
  //     per channel and a channel-major-versus-frequency-major mix-up cannot
  //     show. At 2 it can.
  //   * `a_d_model` 16 over `a_heads` 2 gives head_dim 8 and, at
  //     `partial_rotary_factor` 0.5, a rotary_dim of 4 — TWO frequencies, so
  //     `theta` matters, and FOUR untouched trailing dims, so a port that
  //     rotated the whole head moves the answer.
  //   * `a_ffn` 6 against `a_d_model` 16 makes `fc1` [12, 16] and `fc2` [16, 6]:
  //     not square, and not the same width as anything in the vision tower, so
  //     a transposed or mis-merged gate/up operand refuses on shape.
  //
  // `a_chunk_seconds` 1 makes the padded chunk 16000 samples and 100 mel
  // frames, which is what lets a served request carry a real WAV. The FIXTURE
  // CLIP is half of that (`kAudioSamples`), and §4.14 records why that matters:
  // it leaves 50 valid mel frames of 100, so all four temporal-mask stages have
  // something to zero AND every one of them reaches a KEPT token. See
  // `kAudioTokens`.
  bool with_audio = false;
  int64_t a_d_model = 16;
  int64_t a_heads = 2;
  int64_t a_layers = 2;
  int64_t a_ffn = 6;
  int64_t a_mels = 16;
  int64_t a_dhs = 6;        // downsample_hidden_size
  int64_t a_chunk_seconds = 1;
  int64_t a_max_source_positions = 64;
  double a_partial_rotary_factor = 0.5;
  double a_rope_theta = 97.0;   // not 10000: a wrong theta has to MOVE the answer
  // The adapter lands in the TEXT hidden space; anything else cannot be
  // scattered into the prompt. Zero means "follow the conformant value".
  int64_t a_adapter_out_override = 0;
  // The arms W7a refuses BY NAME, settable so a case can build a checkpoint
  // that selects one and assert the refusal names it.
  bool a_use_causal = false;
  bool a_use_conv2d_stem = true;
  bool a_use_rms_norm = true;
  bool a_use_rope = true;
  int64_t a_merge_factor = 1;
  std::string a_encoder_type = "dots";
  std::string a_activation_function = "swiglu";

  int64_t a_head_dim() const { return a_d_model / a_heads; }
  int64_t a_rotary_dim() const {
    const int64_t r = static_cast<int64_t>(
        static_cast<double>(a_head_dim()) * a_partial_rotary_factor);
    return (r / 2) * 2;
  }
  int64_t a_freq_after() const {
    int64_t f = a_mels;
    for (int i = 0; i < 3; ++i) f = (f + 1) / 2;
    return f;
  }
  int64_t a_fc1_out() const {
    return a_activation_function == "swiglu" ? 2 * a_ffn : a_ffn;
  }
  int64_t a_adapter_out() const {
    return a_adapter_out_override > 0 ? a_adapter_out_override : hidden;
  }
  int64_t a_chunk_samples() const { return a_chunk_seconds * 16000; }
  int64_t a_chunk_mel_frames() const { return a_chunk_seconds * 100; }

  int64_t v_head_dim() const { return v_embed / v_heads; }
  int64_t v_patch_row() const {
    return v_channels * v_temporal * v_patch * v_patch;
  }
  int64_t v_adapter_merge() const {
    return v_adapter_merge_override > 0 ? v_adapter_merge_override : v_merge;
  }
  int64_t v_merged_dim() const {
    return v_embed * v_adapter_merge() * v_adapter_merge();
  }
  // The adapter lands in the TEXT hidden space; anything else cannot be
  // scattered into the prompt.
  int64_t v_adapter_out() const {
    return v_adapter_out_override > 0 ? v_adapter_out_override : hidden;
  }
  int64_t qk_head_dim() const { return qk_nope + qk_rope; }
};

// The three vision marker ids. They are the tokenizer fixture's ADDED tokens in
// the server gate, and `config.json`'s `image_token_id` / `image_start_token_id`
// / `image_end_token_id` here, so the marker string the chat seam injects
// tokenizes to exactly one `<|imgpad|>` id the expansion can expand.
inline constexpr int32_t kImgStartId = 14;
inline constexpr int32_t kImgPadId = 15;
inline constexpr int32_t kImgEndId = 16;

// An 8x8 RGB image over a 2-pixel patch and a 2x2 merge: `factor` is 4, the
// image needs no resize, the grid is (1, 4, 4) = 16 patches, and
// 16 / (2*2) = FOUR placeholder tokens.
inline constexpr int64_t kImageSide = 8;
inline constexpr int64_t kExpectedImageTokens = 4;

// The three AUDIO marker ids (W7a, #2703). They are the tokenizer fixture's
// ADDED tokens in the server gate, and they are DELIBERATELY NOT CONSECUTIVE IN
// start/pad/end ORDER: the released `dots-studio/dots3-note-prev` carries
// `<|audio_comp_start|>` 151718, `<|audio_comp_end|>` 151719 and
// `<|audio_comp_pad|>` 151720 — start, END, pad — so a port that guessed
// "start, start+1, start+2" would swap the pad and end markers. The fixture
// reproduces that ORDER so the guess is wrong here too.
inline constexpr int32_t kAudStartId = 17;
inline constexpr int32_t kAudEndId = 18;
inline constexpr int32_t kAudPadId = 19;

// The fixture clip: 8000 samples at 16 kHz = 0.5 s, against a 1-second chunk.
// That ratio is the point of it, and §4.14 works it through: 8000 // 160 = 50
// valid mel frames of the padded 100, halving to 25, 13 and 7, while
// `ceil(8000 / 1280)` is 7 tokens against a stem output of 13. So the padded
// tail is HALF the chunk, all four mask stages have something to zero, and each
// stage reaches a KEPT token through the 3x3 receptive fields — stage 3 through
// token 6 directly, stage 2 through conv3's read of t=13, stage 1 through
// conv2's read of t=25, stage 0 through conv1's read of mel t=50.
//
// A clip filling the whole chunk would mask NOTHING and the strongest assertion
// in the audio gate would be measuring an identity.
inline constexpr int64_t kAudioSamples = 8000;
inline constexpr int64_t kAudioTokens = 7;
inline constexpr int64_t kAudioStemFrames = 13;

// ── W7b (#2797): the MULTI-CHUNK geometry ──────────────────────────────────
//
// The default `a_chunk_seconds = 1` above CANNOT carry a multi-chunk case, and
// that is a property of the arithmetic rather than of the fixture: 16000 is not
// a whole number of 1280-sample token strides, so the tower's per-segment sum
// `sum_i ceil(seg_i / 1280)` and the prompt side's one `ceil(total / 1280)`
// disagree past one chunk (spec §4.15.3) and the port refuses BY NAME. Two
// seconds is the smallest chunk that divides — `16000 * cs % 1280 == 0` iff
// `cs` is EVEN — and it is what the multi-chunk cases set.
//
//   chunk = 32000 samples = 200 mel frames = 25 stem rows = 25 token strides
//   clip  = 80000 samples = 5 s = 2.5 chunks -> 32000, 32000, 16000
//   rows  = 25, 25, 13 = 63 = ceil(80000 / 1280)
//
// THREE chunks so a reversal is not a swap of two halves, and a SHORT last one
// so the truncation back to `token_len` and the temporal mask on a partly
// padded chunk are both exercised. A clip that were an exact multiple of the
// chunk would mask nothing on its last chunk and truncate nothing.
inline constexpr int64_t kAudioLongChunkSeconds = 2;
inline constexpr int64_t kAudioLongSamples = 80000;
inline constexpr int64_t kAudioLongChunks = 3;
inline constexpr int64_t kAudioLongTokens = 63;
inline constexpr int64_t kAudioLongFullChunkTokens = 25;
inline constexpr int64_t kAudioLongLastChunkTokens = 13;

// ── deterministic values ────────────────────────────────────────────────────
inline uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Values already ROUNDED to the bf16 the checkpoint stores, so a comparison
// against a double reference measures the FORWARD rather than the weights'
// storage width.
inline std::vector<double> Values(int64_t n, uint64_t seed, double amp,
                                  double bias = 0.0) {
  std::vector<double> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const double u =
        static_cast<double>(Mix(seed + static_cast<uint64_t>(i)) >> 40) /
        static_cast<double>(1 << 24);
    const float f = static_cast<float>((u * 2.0 - 1.0) * amp + bias);
    v[static_cast<size_t>(i)] =
        static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(f)));
  }
  return v;
}

// ── safetensors ─────────────────────────────────────────────────────────────
struct StOut {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<double> values;  // already rounded to `dtype`
  std::string dtype = "BF16";
};

inline void WriteSafetensors(const std::vector<StOut>& entries,
                             const std::string& path) {
  nlohmann::json header = nlohmann::json::object();
  size_t off = 0;
  for (const StOut& e : entries) {
    size_t n = 1;
    for (int64_t s : e.shape) n *= static_cast<size_t>(s);
    const size_t w = e.dtype == "F32" ? 4u : 2u;
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {off, off + n * w}}};
    off += n * w;
  }
  const std::string hs = header.dump();
  std::ofstream out(path, std::ios::binary);
  const uint64_t hlen = hs.size();
  out.write(reinterpret_cast<const char*>(&hlen), 8);
  out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  for (const StOut& e : entries) {
    for (double v : e.values) {
      if (e.dtype == "F32") {
        const float f = static_cast<float>(v);
        out.write(reinterpret_cast<const char*>(&f), 4);
      } else {
        const uint16_t b = vt::F32ToBF16(static_cast<float>(v));
        out.write(reinterpret_cast<const char*>(&b), 2);
      }
    }
  }
}

// ── the config document ─────────────────────────────────────────────────────
//
// Built on the COMMITTED released `config.json`, with the geometry overridden.
// That is deliberate: all 36 keys `ParseDots3NoteParams` requires, and the whole
// W1 validation, still apply to this fixture, so a tiny config cannot pass
// through a hole the released one would have hit.
inline nlohmann::json TinyConfigDoc(const std::string& fixture_dir,
                                    const TinySpec& s) {
  nlohmann::json d;
  {
    std::ifstream in(fixture_dir + "/config.json");
    in >> d;
  }
  d["hidden_size"] = s.hidden;
  d["num_hidden_layers"] = s.layers;
  nlohmann::json lt = nlohmann::json::array();
  for (int64_t i = 0; i < s.layers; ++i) lt.push_back("full_attention");
  d["layer_types"] = lt;
  d["num_attention_heads"] = s.heads;
  d["num_key_value_heads"] = s.heads;
  d["qk_nope_head_dim"] = s.qk_nope;
  d["qk_rope_head_dim"] = s.qk_rope;
  d["v_head_dim"] = s.v_head;
  d["q_lora_rank"] = s.q_lora;
  d["kv_lora_rank"] = s.kv_lora;
  d["rope_theta"] = s.rope_theta;
  d["rms_norm_eps"] = s.rms_eps;
  d["max_position_embeddings"] = s.max_pos;
  d["index_n_heads"] = s.index_n_heads;
  d["index_head_dim"] = s.index_head_dim;
  d["index_topk"] = s.index_topk;
  d["indexer_rope_interleave"] = true;
  // The SWA geometry is required by the parse even with zero sliding layers;
  // `swa_kv_lora_rank == kv_lora_rank` keeps the PHYSICAL latent row equal to
  // the logical one.
  d["swa_num_attention_heads"] = 1;
  d["swa_num_key_value_heads"] = 1;
  d["swa_q_lora_rank"] = s.q_lora;
  d["swa_kv_lora_rank"] = s.kv_lora;
  d["swa_qk_nope_head_dim"] = s.qk_nope;
  d["swa_qk_rope_head_dim"] = s.qk_rope;
  d["swa_v_head_dim"] = s.v_head;
  d["vocab_size"] = s.vocab;
  d["intermediate_size"] = s.inter;
  d["moe_intermediate_size"] = 6;
  d["n_routed_experts"] = 4;
  d["num_experts_per_tok"] = 2;
  d["first_k_dense_replace"] = s.layers;   // every layer DENSE
  d["num_nextn_predict_layers"] = 0;       // no MTP tail (W10 owns it)
  d["tie_word_embeddings"] = false;
  // The three image marker ids, so the processor can resolve them from
  // `config.json` the way a converted checkpoint carries them.
  d["image_token_id"] = kImgPadId;
  d["image_start_token_id"] = kImgStartId;
  d["image_end_token_id"] = kImgEndId;

  if (s.with_vision) {
    nlohmann::json v = nlohmann::json::object();
    v["embed_dim"] = s.v_embed;
    v["hidden_size"] = s.hidden;
    v["intermediate_size"] = s.v_inter;
    v["moe_intermediate_size"] = s.v_moe_inter;
    v["num_hidden_layers"] = s.v_layers;
    v["num_attention_heads"] = s.v_heads;
    v["num_channels"] = s.v_channels;
    v["patch_size"] = s.v_patch;
    v["spatial_merge_size"] = s.v_merge;
    v["temporal_patch_size"] = s.v_temporal;
    v["rms_norm_eps"] = s.v_rms_eps;
    v["use_bias"] = s.v_use_bias;
    v["use_qk_norm"] = s.v_use_qk_norm;
    v["is_causal"] = s.v_is_causal;
    v["post_norm"] = s.v_post_norm;
    v["pre_pixel_shuffle"] = s.v_pre_pixel_shuffle;
    v["pyramid_num_routed"] = s.v_pyramid;
    v["capacity_factor"] = s.v_capacity_factor;
    v["router_scoring_func"] = s.v_router_scoring_func;
    v["router_scale"] = s.v_router_scale;
    v["adapter_type"] = s.v_adapter_type;
    v["adapter_in_dim"] = s.v_embed;
    v["adapter_out_dim"] = s.v_adapter_out();
    v["adapter_merge_size"] = s.v_adapter_merge();
    d["vision_config"] = v;
  } else {
    d.erase("vision_config");
  }
  // The AUDIO tower (W7a, #2703). A spec that does not ask for one ERASES the
  // key, which is the state every case before W7a was in: `audio_config`
  // absent means upstream builds no audio tower either
  // (`nvidia/multimodal.py:119-126` @ `9035151d6`), so nothing is owed and the
  // chat seam's ceiling does not declare "audio".
  if (s.with_audio) {
    nlohmann::json a = nlohmann::json::object();
    nlohmann::json wc = nlohmann::json::object();
    wc["d_model"] = s.a_d_model;
    wc["encoder_attention_heads"] = s.a_heads;
    wc["encoder_layers"] = s.a_layers;
    wc["encoder_ffn_dim"] = s.a_ffn;
    wc["num_mel_bins"] = s.a_mels;
    wc["max_source_positions"] = s.a_max_source_positions;
    wc["activation_function"] = s.a_activation_function;
    a["whisper_config"] = wc;
    a["encoder_type"] = s.a_encoder_type;
    a["use_conv2d_stem"] = s.a_use_conv2d_stem;
    a["use_rope"] = s.a_use_rope;
    a["use_rms_norm"] = s.a_use_rms_norm;
    a["use_causal"] = s.a_use_causal;
    a["downsample_hidden_size"] = s.a_dhs;
    a["merge_factor"] = s.a_merge_factor;
    a["chunk_seconds"] = s.a_chunk_seconds;
    a["sampling_rate"] = 16000;
    a["whisper_adapter_in_dim"] = s.a_d_model;
    a["whisper_adapter_out_dim"] = s.a_adapter_out();
    // `rope_parameters` (`nvidia/audio.py:53-60` @ `9035151d6`). WRITTEN, not
    // defaulted: without it `partial_rotary_factor` and `rope_theta` fall back
    // to the struct's own values and a case that thinks it is testing a chosen
    // theta is testing 10000.0 instead. The first version of this fixture
    // omitted the key and the rope case caught it.
    nlohmann::json rp = nlohmann::json::object();
    rp["partial_rotary_factor"] = s.a_partial_rotary_factor;
    rp["rope_theta"] = s.a_rope_theta;
    a["rope_parameters"] = rp;
    a["audio_comp_start"] = "<|audio_comp_start|>";
    a["audio_comp_span"] = "<|audio_comp_pad|>";
    a["audio_comp_end"] = "<|audio_comp_end|>";
    // The three MEASURED-DEAD knobs, written in because the released config
    // writes them: `Dots3NoteAudioConfig` copies all three onto the
    // `WhisperConfig` (`nvidia/audio.py:160-165` @ `9035151d6`) and nothing in
    // `audio_encoder.py` ever reads one. A fixture that omitted them could not
    // show that they are inert.
    a["conv_chunksize"] = 500;
    a["conv_bucket_step"] = 10;
    a["conv_bucket_max_elements"] = 20000;
    d["audio_config"] = a;
  } else {
    d.erase("audio_config");
  }
  return d;
}

// The `preprocessor_config.json` the chat seam's factory reads. Its geometry
// reproduces the `vision_config`'s, and the DEFAULT `min_pixels`/`max_pixels`
// leave `Dots3NoteResizedSize` the IDENTITY on the square fixture image. Since
// W6c a non-conformant image is RESIZED rather than refused, so a case is free
// to move either bound or to hand `ProcessImage` a size that is not a multiple
// of `factor`.
inline nlohmann::json TinyPreprocessorDoc(const TinySpec& s) {
  return nlohmann::json{
      {"patch_size", s.v_patch},
      {"temporal_patch_size", s.v_temporal},
      {"merge_size", s.v_merge},
      {"pre_pixel_shuffle", s.v_pre_pixel_shuffle},
      {"image_mean", {0.5, 0.45, 0.4}},
      {"image_std", {0.25, 0.3, 0.35}},
      {"min_pixels", s.p_min_pixels},
      {"max_pixels", s.p_max_pixels}};
}

inline nlohmann::json TinyAddedTokensDoc() {
  // The AUDIO markers are ALWAYS written, whether or not the spec asks for an
  // audio tower, because upstream reads this file for the IMAGE marker and a
  // tokenizer's added tokens do not depend on which towers a config declares.
  // The chat seam resolves the audio ids from the TOKENIZER rather than from
  // here (`common/processor.py:757-760` @ `9035151d6`); this document exists so
  // the image processor's own loader can find its three.
  return nlohmann::json{{"<|img|>", kImgStartId},
                        {"<|imgpad|>", kImgPadId},
                        {"<|endofimg|>", kImgEndId},
                        {"<|audio_comp_start|>", kAudStartId},
                        {"<|audio_comp_end|>", kAudEndId},
                        {"<|audio_comp_pad|>", kAudPadId}};
}

// ── the tensors ─────────────────────────────────────────────────────────────
//
// The LANGUAGE half is exactly what `EnumerateDots3NoteTensors` claims for a
// config whose every layer is full-attention with a dense MLP, and the VISION
// half is exactly what `EnumerateDots3NoteVisionTensors` claims for an all-dense
// tower. A name this list gets wrong refuses the load by name rather than being
// skipped, which is what makes the fixture self-checking.
// The AUDIO half of the fixture checkpoint (W7a, #2703) — exactly what
// `EnumerateDots3NoteAudioTensors` claims, and nothing else. A name this list
// gets wrong refuses the load BY NAME rather than being skipped, which is what
// makes the fixture self-checking, and the two places it is easiest to get
// wrong are both here: `conv_out` ships NO bias (`audio_encoder.py:479` builds
// it `bias=False`) and `k_proj` ships NO bias (`:221` builds it `bias=False`
// while q, v and out take the default True). Adding either would make every
// real checkpoint refuse.
//
// The generators are PASSED IN rather than rebuilt, so the audio tensors draw
// from the SAME `next()` stream the language and vision halves do: two specs
// that differ only in `with_audio` therefore differ in the audio tensors and in
// nothing else that a shared counter could shift.
template <typename Next, typename Norm, typename Proj>
inline void AppendTinyAudioEntries(const TinySpec& s, std::vector<StOut>* out,
                                   Next&& next, Norm&& norm, Proj&& proj) {
  if (!s.with_audio) return;
  std::vector<StOut>& e = *out;
  const int64_t D = s.a_d_model, F = s.a_ffn, dhs = s.a_dhs;
  const std::string se = "audio_encoder.dots_encoder.speech_encoder.";
  e.push_back({se + "conv2d1.weight", {dhs, 1, 3, 3}, proj(dhs * 9)});
  e.push_back({se + "conv2d1.bias", {dhs}, Values(dhs, next(), 0.2)});
  e.push_back({se + "conv2d2.weight", {dhs, dhs, 3, 3}, proj(dhs * dhs * 9)});
  e.push_back({se + "conv2d2.bias", {dhs}, Values(dhs, next(), 0.2)});
  e.push_back({se + "conv2d3.weight", {dhs, dhs, 3, 3}, proj(dhs * dhs * 9)});
  e.push_back({se + "conv2d3.bias", {dhs}, Values(dhs, next(), 0.2)});
  const int64_t CF = dhs * s.a_freq_after();
  e.push_back({se + "conv_out.weight", {D, CF}, proj(D * CF)});
  for (int64_t l = 0; l < s.a_layers; ++l) {
    const std::string p = se + "layers." + std::to_string(l) + ".";
    e.push_back({p + "self_attn_layer_norm.weight", {D}, norm(D)});
    e.push_back({p + "self_attn.q_proj.weight", {D, D}, proj(D * D)});
    e.push_back({p + "self_attn.q_proj.bias", {D}, Values(D, next(), 0.2)});
    e.push_back({p + "self_attn.k_proj.weight", {D, D}, proj(D * D)});
    e.push_back({p + "self_attn.v_proj.weight", {D, D}, proj(D * D)});
    e.push_back({p + "self_attn.v_proj.bias", {D}, Values(D, next(), 0.2)});
    e.push_back({p + "self_attn.out_proj.weight", {D, D}, proj(D * D)});
    e.push_back({p + "self_attn.out_proj.bias", {D}, Values(D, next(), 0.2)});
    e.push_back({p + "final_layer_norm.weight", {D}, norm(D)});
    // The PACKED SwiGLU pair, gate then up, plus the bias that made W7a extend
    // `layers::MlpGateUpMethodBase`. AMPLITUDE 0.5 rather than 0.1: a bias too
    // small to move `silu(gate) * up` would leave a gate on a dropped bias
    // measuring nothing.
    e.push_back({p + "fc1.weight", {s.a_fc1_out(), D}, proj(s.a_fc1_out() * D)});
    e.push_back({p + "fc1.bias", {s.a_fc1_out()},
                 Values(s.a_fc1_out(), next(), 0.5)});
    e.push_back({p + "fc2.weight", {D, F}, proj(D * F)});
    e.push_back({p + "fc2.bias", {D}, Values(D, next(), 0.5)});
  }
  e.push_back({se + "layer_norm.weight", {D}, norm(D)});
  const std::string ad = "audio_encoder.audio_adapter.proj.";
  const int64_t AO = s.a_adapter_out();
  // `proj.0` is a LAYERNORM, so it ships a weight AND a bias, both [in_dim].
  e.push_back({ad + "0.weight", {D}, norm(D)});
  e.push_back({ad + "0.bias", {D}, Values(D, next(), 0.1)});
  e.push_back({ad + "1.weight", {AO, D}, proj(AO * D)});
  e.push_back({ad + "1.bias", {AO}, Values(AO, next(), 0.1)});
  e.push_back({ad + "3.weight", {AO, AO}, proj(AO * AO)});
  e.push_back({ad + "3.bias", {AO}, Values(AO, next(), 0.1)});
}

inline std::vector<StOut> TinyEntries(const TinySpec& s, uint64_t seed = 7) {
  const int64_t H = s.hidden, N = s.heads, QK = s.qk_head_dim();
  std::vector<StOut> e;
  uint64_t k = seed;
  const auto next = [&k]() { return (k += 0x9E37ULL) * 1000003ULL; };
  // A norm weight sits around 1.0; a projection sits around 0.
  const auto norm = [&](int64_t n) { return Values(n, next(), 0.25, 1.0); };
  const auto proj = [&](int64_t n) { return Values(n, next(), 0.5); };

  e.push_back({"model.embed_tokens.weight", {s.vocab, H}, proj(s.vocab * H)});
  e.push_back({"model.norm.weight", {H}, norm(H)});
  e.push_back({"lm_head.weight", {s.vocab, H}, proj(s.vocab * H)});
  for (int64_t l = 0; l < s.layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    const std::string sa = p + "self_attn.";
    e.push_back({p + "input_layernorm.weight", {H}, norm(H)});
    e.push_back({p + "post_attention_layernorm.weight", {H}, norm(H)});
    e.push_back({sa + "q_a_proj.weight", {s.q_lora, H}, proj(s.q_lora * H)});
    e.push_back({sa + "q_a_layernorm.weight", {s.q_lora}, norm(s.q_lora)});
    e.push_back({sa + "q_b_proj.weight", {N * QK, s.q_lora},
                 proj(N * QK * s.q_lora)});
    e.push_back({sa + "kv_a_proj_with_mqa.weight",
                 {s.kv_lora + s.qk_rope, H}, proj((s.kv_lora + s.qk_rope) * H)});
    e.push_back({sa + "kv_a_layernorm.weight", {s.kv_lora}, norm(s.kv_lora)});
    e.push_back({sa + "kv_b_proj.weight",
                 {N * (s.qk_nope + s.v_head), s.kv_lora},
                 proj(N * (s.qk_nope + s.v_head) * s.kv_lora)});
    e.push_back({sa + "o_proj.weight", {H, N * s.v_head}, proj(H * N * s.v_head)});
    e.push_back({sa + "g_proj.weight", {N, H}, proj(N * H)});
    e.push_back({sa + "k_rope_only_layernorm.weight", {s.qk_rope},
                 norm(s.qk_rope)});
    e.push_back({sa + "indexer.wq_b.weight",
                 {s.index_n_heads * s.index_head_dim, s.q_lora},
                 proj(s.index_n_heads * s.index_head_dim * s.q_lora)});
    e.push_back({sa + "indexer.wk.weight", {s.index_head_dim, H},
                 proj(s.index_head_dim * H)});
    e.push_back({sa + "indexer.k_norm.weight", {s.index_head_dim},
                 norm(s.index_head_dim)});
    e.push_back({sa + "indexer.k_norm.bias", {s.index_head_dim},
                 Values(s.index_head_dim, next(), 0.1)});
    e.push_back({sa + "indexer.weights_proj.weight", {s.index_n_heads, H},
                 proj(s.index_n_heads * H)});
    e.push_back({p + "mlp.gate_proj.weight", {s.inter, H}, proj(s.inter * H)});
    e.push_back({p + "mlp.up_proj.weight", {s.inter, H}, proj(s.inter * H)});
    e.push_back({p + "mlp.down_proj.weight", {H, s.inter}, proj(H * s.inter)});
  }

  if (!s.with_vision) {
    AppendTinyAudioEntries(s, &e, next, norm, proj);
    return e;
  }

  const int64_t E = s.v_embed, VI = s.v_inter, D = s.v_head_dim();
  const std::string vp = "vision_encoder.";
  e.push_back({vp + "patch_embed.proj.weight",
               {E, s.v_channels, s.v_patch, s.v_patch},
               proj(E * s.v_patch_row())});
  e.push_back({vp + "patch_embed.proj.bias", {E}, Values(E, next(), 0.2)});
  e.push_back({vp + "patch_embed.norm.weight", {E}, norm(E)});
  for (int64_t b = 0; b < s.v_layers; ++b) {
    const std::string pre = vp + "blocks." + std::to_string(b) + ".";
    const bool moe = b < static_cast<int64_t>(s.v_pyramid.size()) &&
                     s.v_pyramid[static_cast<size_t>(b)] > 0;
    e.push_back({pre + "norm_1.weight", {E}, norm(E)});
    e.push_back({pre + "norm_2.weight", {E}, norm(E)});
    e.push_back({pre + "attn.qkv.weight", {3 * E, E}, proj(3 * E * E)});
    e.push_back({pre + "attn.proj.weight", {E, E}, proj(E * E)});
    if (s.v_use_qk_norm) {
      e.push_back({pre + "attn.q_norm.weight", {D}, norm(D)});
      e.push_back({pre + "attn.k_norm.weight", {D}, norm(D)});
    }
    if (moe) {
      // The PYRAMID block's own state dict: `mlp.gate_weight` BF16 [ne, E] and
      // `mlp.router_bias` **F32** [ne], which is the dtype upstream registers
      // (vision.py:152-154 @ 9035151d6) and the one dtype in this fixture that
      // is not BF16. Writing it BF16 here is what a re-typed-router mutation
      // does, and the loader refuses it by name.
      const int64_t ne = s.v_pyramid[static_cast<size_t>(b)];
      const int64_t mi = s.v_moe_inter;
      e.push_back({pre + "mlp.gate_weight", {ne, E},
                   Values(ne * E, next() + s.v_router_seed_nudge, 0.5)});
      // AMPLITUDE 0.4, not 0.1. The router bias shifts the SELECTION score
      // against sigmoid probabilities that all sit in (0, 1) and mostly near
      // 0.5, so a bias too small to reorder them would leave the fixture's
      // selection identical to the unbiased one — and a gate on a tower whose
      // bias does nothing cannot see a dropped bias.
      e.push_back({pre + "mlp.router_bias", {ne},
                   Values(ne, next() + s.v_router_seed_nudge,
                          s.v_router_bias_amp),
                   "F32"});
      for (int64_t x = 0; x < ne; ++x) {
        const std::string ep = pre + "mlp.experts." + std::to_string(x) + ".";
        e.push_back({ep + "fc1.weight", {mi, E}, proj(mi * E)});
        e.push_back({ep + "fc2.weight", {E, mi}, proj(E * mi)});
        e.push_back({ep + "fc3.weight", {mi, E}, proj(mi * E)});
      }
    } else {
      e.push_back({pre + "mlp.fc1.weight", {VI, E}, proj(VI * E)});
      e.push_back({pre + "mlp.fc2.weight", {E, VI}, proj(E * VI)});
      e.push_back({pre + "mlp.fc3.weight", {VI, E}, proj(VI * E)});
    }
  }
  if (s.v_post_norm) {
    e.push_back({vp + "post_trunk_norm.weight", {E}, norm(E)});
  }
  const int64_t M = s.v_merged_dim(), O = s.v_adapter_out();
  // THE TWO ADAPTERS HAVE DIFFERENT STATE DICTS AND DIFFERENT SHAPES.
  // `PatchMergerAdapter` is ln_q(in_dim) + mlp.0 [M,M] + mlp.2 [O,M]
  // (vision.py:481-486 @ 9035151d6); `PixelShuffleAdapter` is proj.0(merged_dim)
  // + proj.1 [O,M] + proj.3 [O,O] (vision.py:432-437).
  if (s.v_adapter_type == "pixel_shuffle_mlp") {
    e.push_back({vp + "adapter.proj.0.weight", {M}, norm(M)});
    e.push_back({vp + "adapter.proj.0.bias", {M}, Values(M, next(), 0.1)});
    e.push_back({vp + "adapter.proj.1.weight", {O, M}, proj(O * M)});
    e.push_back({vp + "adapter.proj.1.bias", {O}, Values(O, next(), 0.1)});
    e.push_back({vp + "adapter.proj.3.weight", {O, O}, proj(O * O)});
    e.push_back({vp + "adapter.proj.3.bias", {O}, Values(O, next(), 0.1)});
    AppendTinyAudioEntries(s, &e, next, norm, proj);
    return e;
  }
  e.push_back({vp + "adapter.ln_q.weight", {E}, norm(E)});
  e.push_back({vp + "adapter.ln_q.bias", {E}, Values(E, next(), 0.1)});
  e.push_back({vp + "adapter.mlp.0.weight", {M, M}, proj(M * M)});
  e.push_back({vp + "adapter.mlp.0.bias", {M}, Values(M, next(), 0.1)});
  e.push_back({vp + "adapter.mlp.2.weight", {O, M}, proj(O * M)});
  e.push_back({vp + "adapter.mlp.2.bias", {O}, Values(O, next(), 0.1)});
  AppendTinyAudioEntries(s, &e, next, norm, proj);
  return e;
}

// ── the checkpoint DIRECTORY ────────────────────────────────────────────────
//
// A complete model directory: `config.json`, `model.safetensors`,
// `preprocessor_config.json` and `added_tokens.json`. The last two are what the
// PRODUCTION chat factory reads, so the server gate loads its processor off
// disk on the production path rather than being handed one pre-built.
class TinyCheckpoint {
 public:
  TinyCheckpoint(const std::string& fixture_dir, const TinySpec& spec,
                 uint64_t seed = 7)
      : entries_(TinyEntries(spec, seed)) {
    static int counter = 0;
    static const unsigned salt = std::random_device{}();
    dir_ = std::filesystem::temp_directory_path() /
           ("dots3_note_tiny_" + std::to_string(salt) + "_" +
            std::to_string(counter++));
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / "config.json", std::ios::binary)
        << TinyConfigDoc(fixture_dir, spec).dump();
    std::ofstream(dir_ / "preprocessor_config.json", std::ios::binary)
        << TinyPreprocessorDoc(spec).dump();
    std::ofstream(dir_ / "added_tokens.json", std::ios::binary)
        << TinyAddedTokensDoc().dump();
    WriteSafetensors(entries_, (dir_ / "model.safetensors").string());
  }
  ~TinyCheckpoint() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  TinyCheckpoint(const TinyCheckpoint&) = delete;
  TinyCheckpoint& operator=(const TinyCheckpoint&) = delete;

  std::string dir() const { return dir_.string(); }
  std::string config_path() const { return (dir_ / "config.json").string(); }
  std::string weights_path() const {
    return (dir_ / "model.safetensors").string();
  }
  const std::vector<StOut>& entries() const { return entries_; }
  // The bf16-rounded values of one tensor, by name. Used by the tower gate to
  // drive its DOUBLE reference from the SAME bytes the loader read, so the
  // comparison measures the forward and not a second copy of the weights.
  const std::vector<double>& value_of(const std::string& name) const {
    for (const StOut& e : entries_)
      if (e.name == name) return e.values;
    // AN UNKNOWN NAME IS A TEST DEFECT AND SAYS SO. Returning an empty vector
    // made a typo or a renamed tensor read as a tensor of zero elements, which
    // `ref::Tower` then INDEXES — undefined behaviour whose most likely shape
    // is a reference driven from garbage that the gate compares against
    // itself. An instrument must be able to report its own failure.
    throw std::runtime_error(
        "dots3_tiny::TinyCheckpoint::value_of: this fixture checkpoint has no "
        "tensor named '" + name + "'");
  }

 private:
  std::filesystem::path dir_;
  std::vector<StOut> entries_;
};

// The fixture image: HWC uint8, `kImageSide` square. `variant` picks a
// genuinely DIFFERENT image rather than a shifted one — a high-frequency
// sawtooth against a smooth vertical ramp — so the two disagree in every patch.
inline std::vector<uint8_t> FixtureImage(int variant) {
  std::vector<uint8_t> rgb(static_cast<size_t>(kImageSide * kImageSide * 3));
  for (size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = variant == 0
                 ? static_cast<uint8_t>((i * 37 + 11) & 0xFF)
                 : static_cast<uint8_t>(((i / (kImageSide * 3)) * 29) & 0xFF);
  }
  return rgb;
}

// The NON-CONFORMANT fixture image (W6c, #2537): 6 rows by 14 columns, so
// NEITHER side is a multiple of `factor` = 4 and the two sides differ. A square
// probe cannot see an axis swap or a transposed loop, which is the defect a
// resampler is most likely to carry and the one a shape check never reports.
//
// `Dots3NoteResizedSize(6, 14, 4, 16, 1<<20)` rounds each side independently:
// `round(6/4) = 2` -> 8 rows and `round(14/4) = 4` -> 16 columns, so the target
// is 8x16 and the axes stay distinguishable on BOTH sides of the resize. The
// grid is then (1, 4, 8) = 32 patches and 32 / (2*2) = EIGHT placeholder tokens.
inline constexpr int64_t kOddImageH = 6;
inline constexpr int64_t kOddImageW = 14;
inline constexpr int64_t kOddResizedH = 8;
inline constexpr int64_t kOddResizedW = 16;
inline constexpr int64_t kOddExpectedImageTokens = 8;

// THE BUDGET-DOWNSCALED GEOMETRY (W6c, #2537), and the reason it exists.
//
// Every resize the fixture constants above produce is an UPSCALE on both axes,
// because `Dots3NoteResizedSize` rounds each side to a multiple of `factor` and
// nothing else. On an upscale `filterscale = max(1, in/out)` is 1, the support stays
// 2.0, and PIL's resampler degenerates to exactly the textbook four-tap cubic
// that `pil_resize.h` spends thirty lines arguing it is not: measured, 6x14 ->
// 8x16 is BYTE-IDENTICAL with the support scaling deleted. So the regime the
// file exists for -- and the regime essentially every real request lands in,
// since `factor` is 28 on the released checkpoint -- had no case reaching
// `ProcessImage` at all.
//
// `kBudgetMaxPixels` is what forces it. `Dots3NoteResizedSize(24, 96, 4, 16,
// 64)`: both sides already multiples of 4, so `rh * rw = 2304 > 64` takes the
// `max_pixels` arm, `beta = sqrt(2304 / 64) = 6`, and each side floors to
// `24/6 = 4` and `96/6 = 16`. That is a 6x downscale on BOTH axes, a
// `filterscale` of 6, and a 25-tap window per output pixel. The grid is
// (1, 2, 8) = 16 patches and 16 / (2*2) = FOUR placeholder tokens, and the
// 4x16 result is itself conformant, so it can be served back as its own
// pre-resized twin.
//
// The sides stay UNEQUAL through the whole chain (24 != 96, 4 != 16), so an
// axis swap cannot survive this geometry either.
inline constexpr int64_t kBudgetMaxPixels = 64;
inline constexpr int64_t kBigImageH = 24;
inline constexpr int64_t kBigImageW = 96;
inline constexpr int64_t kBigResizedH = 4;
inline constexpr int64_t kBigResizedW = 16;
inline constexpr int64_t kBigExpectedImageTokens = 4;

// An HWC uint8 RGB image whose value depends on BOTH coordinates and on the
// channel, so a swapped axis, a dropped channel and a half-pixel shift each
// change the bytes. `variant` picks a genuinely different image, exactly as
// `FixtureImage` does.
inline std::vector<uint8_t> FixtureImageHW(int64_t h, int64_t w, int variant) {
  std::vector<uint8_t> rgb(static_cast<size_t>(h * w * 3));
  for (int64_t y = 0; y < h; ++y) {
    for (int64_t x = 0; x < w; ++x) {
      for (int64_t c = 0; c < 3; ++c) {
        const size_t i = static_cast<size_t>((y * w + x) * 3 + c);
        rgb[i] = variant == 0
                     ? static_cast<uint8_t>((y * 37 + x * 11 + c * 61) & 0xFF)
                     : static_cast<uint8_t>((y * 5 + x * 23 + c * 97) & 0xFF);
      }
    }
  }
  return rgb;
}

// ── the fixture AUDIO clip (W7a, #2703) ─────────────────────────────────────
//
// `kAudioSamples` int16 samples at 16 kHz, mono. `variant` picks a genuinely
// DIFFERENT waveform rather than a shifted one — a chirp whose frequency
// sweeps against a two-tone beat — so the two disagree in every mel band and in
// every frame. A shifted copy of one signal would leave a tower that ignored
// its input looking correct on the LOGPROB case, which is the case that carries
// this brick.
inline std::vector<int16_t> FixtureAudioPcm16(int variant) {
  std::vector<int16_t> pcm(static_cast<size_t>(kAudioSamples));
  for (int64_t i = 0; i < kAudioSamples; ++i) {
    const double t = static_cast<double>(i) / 16000.0;
    double v;
    if (variant == 0) {
      // A linear chirp, 200 Hz -> 3200 Hz over the clip.
      const double f = 200.0 + 6000.0 * t;
      v = 0.6 * std::sin(2.0 * 3.14159265358979323846 * f * t);
    } else {
      // Two fixed tones that beat against each other.
      v = 0.35 * std::sin(2.0 * 3.14159265358979323846 * 440.0 * t) +
          0.35 * std::sin(2.0 * 3.14159265358979323846 * 523.25 * t);
    }
    const double clipped = v < -1.0 ? -1.0 : (v > 0.999 ? 0.999 : v);
    pcm[static_cast<size_t>(i)] =
        static_cast<int16_t>(std::lround(clipped * 32767.0));
  }
  return pcm;
}

// The MULTI-CHUNK clip (W7b, #2797): `n` samples of a signal whose content
// changes across the whole clip, so two different chunks of it are two
// different waveforms and a concatenation in the wrong ORDER cannot alias with
// the right one.
//
// A SEPARATE GENERATOR rather than a longer `FixtureAudioPcm16`, deliberately:
// that one's chirp sweeps 200 Hz -> 3200 Hz over ITS 0.5 s, so stretching it to
// 5 s would either change the 0.5 s clip every W7a case is gated on or sweep
// past Nyquist. This one scales the sweep to the clip it is asked for.
inline std::vector<int16_t> FixtureAudioPcm16Long(int variant, int64_t n) {
  std::vector<int16_t> pcm(static_cast<size_t>(n));
  const double dur = static_cast<double>(n) / 16000.0;
  for (int64_t i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / 16000.0;
    double v;
    if (variant == 0) {
      // A linear chirp, 200 Hz -> 3200 Hz over the WHOLE clip, so every chunk
      // sits in a different band.
      const double f = 200.0 + 3000.0 * t / dur;
      v = 0.6 * std::sin(2.0 * 3.14159265358979323846 * f * t);
    } else {
      // The sweep run the other way, plus a fixed tone: a genuinely different
      // signal in every chunk and not a shifted copy of variant 0.
      const double f = 3200.0 - 3000.0 * t / dur;
      v = 0.4 * std::sin(2.0 * 3.14159265358979323846 * f * t) +
          0.3 * std::sin(2.0 * 3.14159265358979323846 * 261.63 * t);
    }
    const double clipped = v < -1.0 ? -1.0 : (v > 0.999 ? 0.999 : v);
    pcm[static_cast<size_t>(i)] =
        static_cast<int16_t>(std::lround(clipped * 32767.0));
  }
  return pcm;
}

// Any PCM16 buffer as a canonical little-endian MONO RIFF/WAVE file. Factored
// out of `FixtureAudioWav` below by W7b so the multi-chunk clip reaches the
// served path through the SAME writer, byte for byte, rather than a second one.
inline std::vector<uint8_t> FixtureWavFromPcm16(const std::vector<int16_t>& pcm,
                                                int sample_rate = 16000,
                                                int channels = 1) {
  const uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * 2);
  std::vector<uint8_t> out;
  const auto put_str = [&out](const char* s4) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(s4[i]));
  };
  const auto put_u32 = [&out](uint32_t v) {
    for (int i = 0; i < 4; ++i)
      out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  };
  const auto put_u16 = [&out](uint16_t v) {
    for (int i = 0; i < 2; ++i)
      out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  };
  put_str("RIFF");
  put_u32(36 + data_bytes);
  put_str("WAVE");
  put_str("fmt ");
  put_u32(16);
  put_u16(1);  // PCM
  put_u16(static_cast<uint16_t>(channels));
  put_u32(static_cast<uint32_t>(sample_rate));
  put_u32(static_cast<uint32_t>(sample_rate * channels * 2));  // byte rate
  put_u16(static_cast<uint16_t>(channels * 2));                // block align
  put_u16(16);                                                 // bits
  put_str("data");
  put_u32(data_bytes);
  for (int16_t v : pcm) put_u16(static_cast<uint16_t>(v));
  return out;
}

// The 0.5 s clip as that same container, which is the ONE this port decodes
// (`DecodeWavPcm16Mono`). `sample_rate` and `channels` are parameters so a case
// can build a deliberately non-conformant file and assert the refusal names
// W7c.
inline std::vector<uint8_t> FixtureAudioWav(int variant, int sample_rate = 16000,
                                            int channels = 1) {
  return FixtureWavFromPcm16(FixtureAudioPcm16(variant), sample_rate, channels);
}

// The MULTI-CHUNK clip as the same container (W7b, #2797).
inline std::vector<uint8_t> FixtureAudioWavLong(int variant, int64_t n) {
  return FixtureWavFromPcm16(FixtureAudioPcm16Long(variant, n));
}

// ── the SAME clip at ANOTHER sample rate (W7c-2, #2828) ─────────────────────
//
// `FixtureAudioPcm16` samples its chirp at 16000 for `kAudioSamples` frames,
// which is 0.5 s. This samples THE SAME CONTINUOUS SIGNAL, from the same
// closed form, at `sample_rate` for the same 0.5 s — so `sample_rate` frames
// per second of the identical waveform and NOT a stretched or shifted copy.
//
// WHY THE DURATION IS HELD AND NOT THE FRAME COUNT. The resampled length is
// `ceil(n * 16000 / sample_rate)`, and holding the duration makes that exactly
// `kAudioSamples` for every rate that divides evenly into the 0.5 s — 44100
// gives 22050 frames and `ceil(22050 * 160 / 441) == 8000`. So a resampled
// request expands the SAME 7-token placeholder span as the mono clip every
// other audio case serves, and the served case can compare the two directly.
//
// THAT TOKEN COUNT IS THE ASSERTION A NO-OP RESAMPLE CANNOT SURVIVE: an
// unresampled 22050-frame waveform expands `ceil(22050 / 1280)` = 18
// placeholders, not 7, and it does so without any reference to the resampler's
// values.
inline std::vector<int16_t> FixtureAudioPcm16AtRate(int variant,
                                                    int sample_rate) {
  const int64_t n = kAudioSamples * sample_rate / 16000;
  std::vector<int16_t> pcm(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(sample_rate);
    double v;
    if (variant == 0) {
      const double f = 200.0 + 6000.0 * t;
      v = 0.6 * std::sin(2.0 * 3.14159265358979323846 * f * t);
    } else {
      v = 0.35 * std::sin(2.0 * 3.14159265358979323846 * 440.0 * t) +
          0.35 * std::sin(2.0 * 3.14159265358979323846 * 523.25 * t);
    }
    const double clipped = v < -1.0 ? -1.0 : (v > 0.999 ? 0.999 : v);
    pcm[static_cast<size_t>(i)] =
        static_cast<int16_t>(std::lround(clipped * 32767.0));
  }
  return pcm;
}

// That clip as a PCM16 RIFF/WAVE file whose `fmt ` chunk DECLARES the rate,
// through the same writer every other audio case uses.
inline std::vector<uint8_t> FixtureAudioWavAtRate(int variant, int sample_rate) {
  return FixtureWavFromPcm16(FixtureAudioPcm16AtRate(variant, sample_rate),
                             sample_rate, 1);
}

// ...and as the decoder would produce it, `int16 / 32768.0`.
inline std::vector<float> FixtureAudioF32AtRate(int variant, int sample_rate) {
  const std::vector<int16_t> pcm = FixtureAudioPcm16AtRate(variant, sample_rate);
  std::vector<float> out(pcm.size());
  for (size_t i = 0; i < pcm.size(); ++i)
    out[i] = static_cast<float>(pcm[i]) / 32768.0f;
  return out;
}

// ── the STEREO clip (W7c-1, #2813) ──────────────────────────────────────────
//
// TWO GENUINELY DIFFERENT CHANNELS WHOSE MEAN IS EXACTLY VARIANT 0. Left is
// `m + d` and right is `m - d`, where `m` is `FixtureAudioPcm16(0)` — the chirp
// every W7a case already serves — and `d` is a QUARTER of the two-tone beat of
// variant 1. So `(L[i] + R[i]) / 2 == m[i]` for every i, with no rounding at
// all: the sum is `2 * m[i]`, an even integer.
//
// WHY THAT CONSTRUCTION AND NOT TWO ARBITRARY SIGNALS. The mean of two
// arbitrary int16 channels is not an int16, so it could not be fed back through
// a WAV to compare against. This one makes the EXPECTED mono waveform a clip
// the suite already knows how to send, which turns "the mean is right" into an
// equality between two SERVED requests rather than into a tolerance.
//
// NO CLIPPING IS POSSIBLE, which matters because a clip would break the
// equality silently: |m| <= 0.6 * 32767 = 19660 and |d| <= 0.7 * 32767 / 4 =
// 5734, so |m +- d| <= 25394, well inside int16.
inline void FixtureAudioPcm16StereoChannels(std::vector<int16_t>* left,
                                            std::vector<int16_t>* right) {
  const std::vector<int16_t> m = FixtureAudioPcm16(0);
  const std::vector<int16_t> beat = FixtureAudioPcm16(1);
  left->resize(m.size());
  right->resize(m.size());
  for (size_t i = 0; i < m.size(); ++i) {
    const int d = static_cast<int>(beat[i]) / 4;
    (*left)[i] = static_cast<int16_t>(static_cast<int>(m[i]) + d);
    (*right)[i] = static_cast<int16_t>(static_cast<int>(m[i]) - d);
  }
}

// Those two channels INTERLEAVED, which is the frame order a `data` chunk
// carries: L0 R0 L1 R1 ... The buffer is 2N int16 for N frames, so the file has
// the SAME frame count as the mono clip and expands to the same placeholders.
inline std::vector<int16_t> FixtureAudioPcm16StereoInterleaved() {
  std::vector<int16_t> l, r;
  FixtureAudioPcm16StereoChannels(&l, &r);
  std::vector<int16_t> out(l.size() * 2);
  for (size_t i = 0; i < l.size(); ++i) {
    out[2 * i] = l[i];
    out[2 * i + 1] = r[i];
  }
  return out;
}

// The stereo clip as a 2-channel PCM16 RIFF/WAVE file, through the SAME writer
// every other audio case uses.
inline std::vector<uint8_t> FixtureAudioWavStereo() {
  return FixtureWavFromPcm16(FixtureAudioPcm16StereoInterleaved(), 16000,
                             /*channels=*/2);
}

// The clip as the DECODER would produce it: `int16 / 32768.0`. Used by the
// tower gate so its reference is driven from the same quantized samples the
// served path sees, rather than from the float signal before rounding.
inline std::vector<float> FixtureAudioF32(int variant) {
  const std::vector<int16_t> pcm = FixtureAudioPcm16(variant);
  std::vector<float> out(pcm.size());
  for (size_t i = 0; i < pcm.size(); ++i)
    out[i] = static_cast<float>(pcm[i]) / 32768.0f;
  return out;
}

// The MULTI-CHUNK clip as the decoder would produce it (W7b, #2797).
inline std::vector<float> FixtureAudioLongF32(int variant, int64_t n) {
  const std::vector<int16_t> pcm = FixtureAudioPcm16Long(variant, n);
  std::vector<float> out(pcm.size());
  for (size_t i = 0; i < pcm.size(); ++i)
    out[i] = static_cast<float>(pcm[i]) / 32768.0f;
  return out;
}

}  // namespace dots3_tiny

#endif  // VLLM_TESTS_DOTS3_NOTE_TINY_FIXTURE_H_
