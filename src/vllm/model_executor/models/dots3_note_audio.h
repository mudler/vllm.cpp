// dots3-note AUDIO tower — the `dots` Whisper-variant speech encoder (W7a,
// #2703).
//
// Ported from vLLM read in the local clone `~/_git/vllm` at **`9035151d6`**,
// the merge of vllm#51255 that added the architecture. `dots3_note` does not
// exist at our parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, so that
// SHA is written beside every anchor here and an anchor with no revision is a
// line number read in the wrong tree. §2.5 of the spec records that eight of
// this file's anchors were `+9` before W7a re-measured them.
//
//   vllm/models/dots3_note/nvidia/audio_encoder.py @ 9035151d6 (736 lines)
//     RMSNorm                      :30    -> vt::RmsNorm (eps 1e-6, and see
//                                           the rounding note below)
//     swiglu                       :42    -> vt::SiluAndMul (gate-then-up)
//     RotaryEmbedding              :47    -> Dots3NoteAudioRopeCache
//     rotate_half                  :140   -> vt::RopeFromCache, NeoX
//     apply_rotary_pos_emb         :146   -> the same, at rotary_dim
//     WhisperAttention             :196   -> the q/v/out-bias-and-NOT-k block
//       forward_flash_attn         :226   -> vt::AttentionDenseFlash
//     WhisperEncoderLayer          :310   -> Dots3NoteAudioLayerWeights
//       forward                    :338   -> the pre-norm block
//     DotsSpeechEncoder            :428   -> Dots3NoteAudioForward
//       __init__ (the stem)        :437
//       _temporal_mask             :529   -> the four-stage mask
//       _conv2d_stem_one_chunk     :535
//       _forward_conv2d_stem       :564
//       forward                    :611
//   vllm/models/dots3_note/nvidia/audio.py @ 9035151d6 (305 lines)
//     Dots3NoteAudioConfig         :26    -> Dots3NoteAudioParams
//     DotsEncoderWithMask          :150
//       encode_waveform            :193   -> the caller's slice to num_tokens
//     AudioAdapter                 :237   -> the LayerNorm + 2 Linears
//     Dots3NoteAudioModel          :251
//
// WHY THIS IS A NEW FILE AND NOT AN EXTENSION OF `whisper_audio.h`. Upstream
// itself forks `modeling_whisper.py` into a separate file rather than
// parameterising Whisper (`audio_encoder.py:6-7` says so), and the deltas are
// structural rather than parametric: RMSNorm instead of LayerNorm, a PACKED
// SwiGLU MLP instead of GELU, partial RoPE instead of a fixed additive
// sinusoid, and a three-layer Conv2d stem with a four-stage temporal mask
// instead of two Conv1ds. `whisper_audio.h` was read before this file was
// written, and what it contributed is the block SKELETON and the exact
// q/v/out-bias-and-NOT-k convention — both mirrored here — plus the `vt` ops
// this file routes through.
//
// THE CONV2D STEM IS COMPOSED AS im2col + `vt::MatmulBT`, AND THAT IS ONE
// EXACT TRACKED EXCEPTION TO THE `vt::Conv2d` SEAM (AGENTS.md, "Shared seams").
// `vt::Conv2d` exists, and `src/vt/cpu/cpu_conv2d.cpp:111` is the ONLY
// `RegisterOp(OpId::kConv2d, ...)` in the tree — there is no CUDA provider — so
// a stem routed through it would resolve no op on a CUDA queue, which is every
// host that could ever serve this model. Shipping a capability nobody can run
// on a GPU is worse than composing it out of ops that exist. The in-tree
// precedent is `include/vllm/model_executor/models/whisper_audio.h:33`, which
// composes its Conv1d stem the same way and says "no new CUDA kernel". The gap
// itself is a `vt` gap that outlives this row and is
// [#2709](https://github.com/mudler/vllm.cpp/issues/2709).
//
// THE ONE DELIBERATE FORMULA DIFFERENCE, named rather than left to be found —
// and it is the SAME one `dots3_note_vision.h` records. Upstream's `RMSNorm`
// (`audio_encoder.py:36-39`) is `x * rsqrt(var + eps)` then `self.weight * x`,
// with the intermediate in the ACTIVATION dtype; `vt::RmsNorm` keeps f32
// through the weight multiply and rounds once on the store. Using the shared op
// is the seam rule; the difference is one bf16 rounding step, it is what the
// gate's tolerance carries, and the in-test double-precision reference mirrors
// UPSTREAM rather than this file so the difference is measured rather than
// defined away.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_DOTS3_NOTE_AUDIO_H_
#define VLLM_MODEL_EXECUTOR_MODELS_DOTS3_NOTE_AUDIO_H_

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vt/backend.h"

namespace vllm {

class SafetensorsFile;
struct HfConfig;
struct Dots3NoteTensor;

// `Dots3NoteAudioConfig.__init__` (`audio.py:26-60`) folded together with the
// `WhisperConfig` it builds (`audio.py:153-165`), reduced to the fields this
// arm reads plus the ones it REFUSES on. Every default is upstream's own, so a
// config that omits a key gets what upstream's constructor would have given it.
struct Dots3NoteAudioParams {
  // False when `config.json` carries no `audio_config`. Upstream builds no
  // `Dots3NoteAudioModel` in that case (`multimodal.py:119-126`).
  bool present = false;

  // `whisper_config` (`audio.py:29`), the keys the released checkpoint sets.
  int64_t d_model = 1280;
  int64_t num_heads = 20;              // encoder_attention_heads
  int64_t num_layers = 32;             // encoder_layers
  int64_t ffn_dim = 5120;              // encoder_ffn_dim
  int64_t num_mel_bins = 128;
  int64_t max_source_positions = 6000;
  std::string activation_function = "swiglu";

  // The `audio_config` keys proper.
  std::string encoder_type = "dots";   // audio.py:28
  bool use_conv2d_stem = true;         // audio.py:42
  bool use_rope = true;                // audio.py:43
  bool use_rms_norm = true;            // audio.py:44
  bool use_causal = false;             // audio.py:45
  // NOT an `audio_config` key upstream — `DotsSpeechEncoder` reads it off the
  // `WhisperConfig` with `getattr(..., False)` (`audio_encoder.py:442`) and
  // `Dots3NoteAudioConfig` never sets it, so it is always False for a dots3
  // checkpoint. It is a field here so the refusal can name it: a
  // `whisper_config` that set it would silently select a different stem.
  bool use_latent_input = false;
  int64_t downsample_hidden_size = 480;  // audio.py:46
  int64_t merge_factor = 1;              // audio.py:40
  // `chunk_seconds` (`audio.py:41`), carried on the TOWER and not only on the
  // processor because `DotsEncoderWithMask` keeps it too (`audio.py:169-171`)
  // and it is what upstream's `mel.shape[1] == self.chunk_mel_frames` assert
  // (`:215`) compares against. W7b (#2797) made that assert executable here:
  // once the mel is a STACK of chunks, a caller that hands the tower the whole
  // stack instead of one chunk produces correctly-shaped output, and only this
  // number can tell the two apart.
  int64_t chunk_seconds = 60;
  int64_t adapter_in_dim = 1280;         // whisper_adapter_in_dim, audio.py:30
  int64_t adapter_out_dim = 5120;        // whisper_adapter_out_dim, audio.py:33

  // `rope_parameters` (`audio.py:53-60`).
  double partial_rotary_factor = 0.5;
  double rope_theta = 10000.0;
  std::string rope_type = "default";

  // MEASURED DEAD UPSTREAM, not deferred. `Dots3NoteAudioConfig` reads all
  // three (`audio.py:47`, `:51-52`) and copies them onto the `WhisperConfig`
  // (`audio.py:160-165`), and then NOTHING in `audio_encoder.py` reads any of
  // them: `_forward_conv2d_stem` (`:564-583`) calls `_conv2d_stem_one_chunk`
  // ONCE, on the whole tensor. They are recorded so a reader who finds them in
  // the released `config.json` learns they are inert rather than owed — a
  // "deferred" label here would invent a debt no brick could ever discharge.
  int64_t conv_chunksize = 500;
  int64_t conv_bucket_step = 0;
  int64_t conv_bucket_max_elements = 0;

  // The TEXT tower's `hidden_size`, copied from the LANGUAGE config at parse,
  // for the same reason `Dots3NoteVisionParams::text_hidden_size` exists: the
  // encoder hook compares `adapter_out_dim` against `config.hidden_size`, so
  // the refusal has to be able to ask the same question.
  int64_t text_hidden_size = 0;

  // `nn.LayerNorm` for the ADAPTER's `proj.0` (`audio.py:241`), torch's own
  // default eps. NOT the RMSNorm eps below.
  double layer_norm_eps = 1e-5;
  // `RMSNorm(dim, eps=1e-6)` (`audio_encoder.py:31`), the default every
  // `norm_cls(self.embed_dim)` call takes (`audio_encoder.py:322-323`, `:336`,
  // `:515-516`).
  double rms_norm_eps = 1e-6;

  // `chunk_mel_frames` (`audio.py:79-81`): `chunk_seconds * 100`, which is also
  // `chunk_samples / hop_length` at 16 kHz and 160.
  int64_t chunk_mel_frames() const { return chunk_seconds * 100; }

  int64_t head_dim() const { return d_model / num_heads; }
  // `rotary_dim = int(head_dim * partial_rotary_factor)`, then rounded DOWN to
  // even to keep the cos/sin pairs aligned (`audio_encoder.py:60-62`).
  int64_t rotary_dim() const {
    const int64_t r =
        static_cast<int64_t>(static_cast<double>(head_dim()) *
                             partial_rotary_factor);
    return (r / 2) * 2;
  }
  // `freq_after` (`audio_encoder.py:476-478`): the mel axis after three
  // stride-2 layers with padding 1. 128 -> 64 -> 32 -> 16 on the released
  // geometry, which is why `conv_out` is [1280, 480 * 16 = 7680].
  int64_t freq_after() const {
    int64_t f = num_mel_bins;
    for (int i = 0; i < 3; ++i) f = (f + 1) / 2;
    return f;
  }
  int64_t conv_out_in_dim() const { return downsample_hidden_size * freq_after(); }
  // `fc1_out = ffn_dim * 2 if use_swiglu` (`audio_encoder.py:333`).
  int64_t fc1_out() const {
    return activation_function == "swiglu" ? 2 * ffn_dim : ffn_dim;
  }
};

// Resolve + validate `config.json`'s `audio_config`. Returns `present=false`
// when the key is absent. Throws (VT_CHECK) naming the key on a value no
// dots3-note tower can have; a value that IS owed to a later brick is reported
// by `Dots3NoteAudioRefusal` instead, because a checkpoint whose tower is owed
// must still LOAD its language half.
Dots3NoteAudioParams ParseDots3NoteAudioParams(const HfConfig& config);

// Why the audio tower cannot be materialized, or "" when it can. Names ONE
// thing — the first unrepresentable feature — and the brick or issue that owes
// it. A non-empty answer leaves all 430 `audio_encoder.*` tensors in the
// accounting's existing `audio` deferral bucket, exactly as before W7a, so
// every W2 count assertion is unchanged.
std::string Dots3NoteAudioRefusal(const Dots3NoteAudioParams& a,
                                  const std::string& quant_method,
                                  const std::vector<int64_t>& weight_block_size);

// The same answer from a CONFIG alone, for a caller that holds a checkpoint
// directory and no loaded model — the multimodal CHAT seam is that caller.
//
// A refusal raised from `encode_mm` is FATAL: it is thrown inside the engine's
// busy loop, which stops `AsyncLLM` and turns every later request, TEXT ONES
// INCLUDED, into a 500. That was measured on this row's served-request gate
// (see `mm_chat_dots3note.cpp:232-240`). Asking at INSTALL turns the same
// answer into a REFUSING seam: HTTP 400 naming the architecture and the reason,
// with the text path untouched.
std::string Dots3NoteAudioRefusalFor(const HfConfig& config);

// One encoder block's weights, by the names the checkpoint ships
// (`audio_encoder.dots_encoder.speech_encoder.layers.{L}.*`). Every tensor is
// BF16 on disk and BF16 here: the released `model-audio.safetensors` carries
// 430 BF16 tensors and NOT ONE F32, measured in the committed shard index.
struct Dots3NoteAudioLayerWeights {
  // RMSNorm — weight only, no bias (`audio_encoder.py:30-39`).
  OwnedTensor attn_norm;   // self_attn_layer_norm.weight [D]
  OwnedTensor final_norm;  // final_layer_norm.weight     [D]

  // WHISPER'S BIAS CONVENTION, and the obvious thing to get wrong:
  // `k_proj` is built `bias=False` and the other three `bias=bias` with `bias`
  // defaulting True (`audio_encoder.py:221-224`). The released checkpoint
  // agrees exactly — 32 each of `q_proj.bias`, `v_proj.bias` and
  // `out_proj.bias`, and NO `k_proj.bias`. A port that gave k a bias would
  // refuse the load on a missing tensor; one that dropped q's would compute.
  OwnedTensor q_w, q_b;    // [D, D], [D]
  OwnedTensor k_w;         // [D, D] — NO bias
  OwnedTensor v_w, v_b;
  OwnedTensor out_w, out_b;

  // `fc1` is ALREADY the merged gate|up pair: `nn.Linear(d_model, 2 * ffn_dim)`
  // under `activation_function == "swiglu"` (`audio_encoder.py:333-334`), and
  // `swiglu` chunks it gate-then-up (`:42-44`), which is `vt::SiluAndMul`'s own
  // order. So it rides `layers::MlpGateUpMethodBase` — through the BIAS arm
  // W7a added to that seam, because the checkpoint ships `fc1.bias [10240]`.
  OwnedTensor fc1_w, fc1_b;  // [2 * ffn_dim, D], [2 * ffn_dim]
  OwnedTensor fc2_w, fc2_b;  // [D, ffn_dim], [D]
};

struct Dots3NoteAudioWeights {
  bool present = false;

  // The stem. `conv2d1` takes ONE input channel — the mel matrix is unsqueezed
  // to [B, 1, n_mels, T] (`audio_encoder.py:568`) — and the other two take
  // `downsample_hidden_size`. `conv_out` is `bias=False`
  // (`audio_encoder.py:479`), which the checkpoint confirms by shipping no
  // `conv_out.bias`.
  OwnedTensor conv1_w, conv1_b;  // [dhs, 1, 3, 3], [dhs]
  OwnedTensor conv2_w, conv2_b;  // [dhs, dhs, 3, 3], [dhs]
  OwnedTensor conv3_w, conv3_b;
  OwnedTensor conv_out_w;        // [d_model, dhs * freq_after] — NO bias

  std::vector<Dots3NoteAudioLayerWeights> layers;
  OwnedTensor final_norm;  // speech_encoder.layer_norm.weight [D]

  // `AudioAdapter.proj` is `Sequential(LayerNorm(in), Linear(in, out), GELU(),
  // Linear(out, out))` (`audio.py:240-245`), so the indices are 0, 1, 3 and
  // index 2 is the activation, which ships no tensor. `proj.0` is a LAYERNORM
  // and therefore carries a BIAS as well as a weight — #2703 describes the
  // adapter as "1280 to 5120" without saying that, and a reader who took
  // `proj.0` for a Linear would look for a [5120, 1280] weight and find [1280].
  OwnedTensor adapter_ln_w, adapter_ln_b;  // proj.0 [in_dim] x2
  OwnedTensor adapter_fc1_w, adapter_fc1_b;  // proj.1 [out, in], [out]
  OwnedTensor adapter_fc2_w, adapter_fc2_b;  // proj.3 [out, out], [out]
};

// Every `audio_encoder.*` name the tower claims, with its named consumer. Over
// the released checkpoint this is all 430 audio tensors.
std::vector<Dots3NoteTensor> EnumerateDots3NoteAudioTensors(
    const Dots3NoteAudioParams& a);

// Read the tower out of `shards`. REFUSES BY NAME on the first tensor whose
// shape disagrees with the config. Only called when `Dots3NoteAudioRefusal` is
// empty.
Dots3NoteAudioWeights MaterializeDots3NoteAudio(
    const std::vector<SafetensorsFile>& shards,
    const Dots3NoteAudioParams& a);

// `RotaryEmbedding.get_cos_sin` (`audio_encoder.py:85-137`) folded into the
// `[T, rotary_dim]` = [cos(rd/2) | sin(rd/2)] cache `vt::RopeFromCache`
// consumes. f32 host precompute, deterministic. Positions are `arange(T)`
// (`audio_encoder.py:644-646`), so this takes a length rather than a position
// list.
std::vector<float> Dots3NoteAudioRopeCache(int64_t num_positions,
                                           const Dots3NoteAudioParams& a);

// Optional intermediate capture, for the unit gate only. Production passes
// nullptr and pays nothing.
struct Dots3NoteAudioCapture {
  std::vector<float> rope_cache;    // [T, rotary_dim]
  std::vector<float> masked_mel;    // [n_mels, n_frames] after the stage-0 mask
  std::vector<float> stem_out;      // [T_stem, d_model] after conv_out
  std::vector<float> block0_out;    // [T, d_model] after encoder layer 0
  std::vector<float> trunk_out;     // [T, d_model] after the final RMSNorm
  // The four `valid` lengths the temporal mask uses, in stage order:
  // {mel, after conv1, after conv2, after conv3}. Captured because a mask that
  // halved with the wrong rounding would still zero SOMETHING.
  std::vector<int64_t> valid_lens;
};

// THE TOWER. `mel` is the PADDED log-mel `[n_mels * chunk_mel_frames]` host
// f32, exactly what `Dots3NoteAudioProcessor::ProcessWaveform` returns.
// `num_samples` is the waveform's own length BEFORE padding — the tower derives
// `valid_mel_lens = num_samples / hop_length` from it
// (`audio_encoder.py:570-574`) — and `num_tokens` is the placeholder run
// length, `ceil(num_samples / token_stride)`.
//
// Returns `[num_tokens, adapter_out_dim]` host f32.
//
// `num_tokens` AND `valid_mel_lens` ARE TWO DIFFERENT NUMBERS and neither can
// be derived from the other. Halving `num_samples / 160` three times with
// `(n + 1) / 2` is not `ceil(num_samples / 1280)`: at 1281 samples the first
// gives 1 and the second gives 2, so the span includes one stem row the mask
// zeroed. That is upstream's own behaviour (`audio.py:210-212` computes the
// token length from the SAMPLE count while `audio_encoder.py:570-574` derives
// the mask from it independently) and it is reproduced rather than tidied.
std::vector<float> Dots3NoteAudioForward(const std::vector<float>& mel,
                                         int64_t num_samples,
                                         int64_t num_tokens, int64_t hop_length,
                                         const Dots3NoteAudioWeights& w,
                                         const Dots3NoteAudioParams& a,
                                         vt::Backend& backend,
                                         Dots3NoteAudioCapture* capture = nullptr);

// THE MULTI-CHUNK TOWER (W7b, #2797) — `DotsEncoderWithMask.encode_waveform`'s
// encoder half, `nvidia/audio.py:220-234` @ `9035151d6`. THIS IS THE PRODUCTION
// ENTRY POINT from W7b on; `Dots3NoteAudioForward` above is one chunk of it.
//
// `mels` is the STACKED `[num_chunks, n_mels, chunk_mel_frames]` host f32
// `Dots3NoteAudioProcessor::ProcessWaveform` returns (`torch.stack`, `:220`),
// `chunk_num_samples` is upstream's `audio_sample_lens` and `chunk_num_tokens`
// its `token_lens` (`:217-218`). Returns
// `[sum(chunk_num_tokens), adapter_out_dim]` host f32: each chunk's first
// `token_len * merge_factor` rows, concatenated IN ORDER (`:229-234`).
//
// A LOOP IS UPSTREAM'S BATCHED CALL, and this is the one claim worth reading
// before changing it. Upstream stacks the mels and makes ONE
// `speech_encoder(...)` call with `input_seq_lens` (`:225-227`), and inside it:
//   * the stem masks each batch element from ITS OWN `valid_mel_lens`
//     (`audio_encoder.py:570-577`) and Conv2d is batch-independent;
//   * the varlen PACK builds `cu_seqlens` from `input_seq_lens.cumsum`
//     (`:674-677`), so each chunk is its own bidirectional attention window;
//   * the packed rope positions are `arange(S)` gathered by the valid mask
//     (`:679-685`), so every chunk RESTARTS at position 0;
//   * the UNPACK (`:711-719`), the final `layer_norm` (`:721`) and the whole
//     adapter (`audio.py:240-248`) are row-wise, and the zero rows the unpack
//     writes are sliced away before anything reads them.
// So the chunks never interact, and `k` calls to the single-chunk tower compute
// the same numbers as one call over a batch of `k`. What the batch buys
// upstream is one kernel launch instead of `k`; this row claims no performance
// axis. At `num_chunks == 1` this is W7a's path with W7a's arguments, byte for
// byte.
//
// `captures`, when non-null, is resized to one capture PER CHUNK, so a gate can
// read each chunk's four mask stages rather than only the first one's.
std::vector<float> Dots3NoteAudioForwardChunks(
    const std::vector<float>& mels,
    const std::vector<int64_t>& chunk_num_samples,
    const std::vector<int64_t>& chunk_num_tokens, int64_t hop_length,
    const Dots3NoteAudioWeights& w, const Dots3NoteAudioParams& a,
    vt::Backend& backend,
    std::vector<Dots3NoteAudioCapture>* captures = nullptr);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_DOTS3_NOTE_AUDIO_H_
