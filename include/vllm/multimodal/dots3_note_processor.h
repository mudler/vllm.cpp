// dots3-note IMAGE processor (W6a, #2512).
//
// Ported from `vllm/models/dots3_note/common/processor.py` read in `~/_git/vllm`
// at **`9035151d6`** — the merge of vllm#51255. `dots3_note` does not exist at
// our parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, so every anchor
// here names that SHA; upstream has already moved under this row once.
//
//   IMAGE_START / IMAGE_PAD / IMAGE_END      :41-43
//   Dots3NoteImageProcessor.__init__         :63-79
//   .factor                                  :83-84
//   ._round_by_factor / _ceil / _floor       :86-96
//   .resized_size                            :97-146
//   .preprocess                              :147-218
//
// WHY THIS IS NOT `Qwen3VLImageProcessor` WITH DIFFERENT NUMBERS. Three things
// differ and each is silent when wrong:
//
//   1. `resized_size` is NOT `smart_resize`. Upstream dots3 rounds each side
//      INDEPENDENTLY to a multiple of `factor` and only then applies the pixel
//      budget (`processed.py:139-146`), where `smart_resize`
//      (transformers `image_processing_qwen2_vl.py:62`) does the same rounding
//      but with different guards and a different min-pixel branch. The two
//      agree on many images and disagree on some, and a disagreement moves the
//      GRID, which moves the placeholder count, which changes the prompt.
//   2. `image_mean` / `image_std` are PER-CHANNEL lists, not the single scalar
//      Qwen3-VL's 0.5/0.5 collapses to.
//   3. The patch row order is selected by `pre_pixel_shuffle`. TRUE is the
//      2x2-grouped order (which happens to be byte-identical to Qwen3-VL's
//      merge-grouped patchify); FALSE is plain row-major. The released
//      checkpoint sets TRUE, and the tower's RoPE position builder reads the
//      SAME flag — so a processor and a tower that disagree on it produce a
//      well-shaped, wrong answer.
//
// The placeholder EXPANSION is shared rather than re-written:
// `multimodal::ExpandImagePlaceholders` already takes the image token id, the
// merge size and the grids, and dots3's rule is upstream's same
// `grid.prod(-1) // merge_size**2` (`multimodal.py:151-155` @ `9035151d6`).
#ifndef VLLM_MULTIMODAL_DOTS3_NOTE_PROCESSOR_H_
#define VLLM_MULTIMODAL_DOTS3_NOTE_PROCESSOR_H_

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vllm/multimodal/audio_processor.h"
#include "vllm/multimodal/inputs.h"

namespace vllm::multimodal {

// The subset of `preprocessor_config.json` + `config.json` the image path
// needs. Defaults are upstream's own where upstream has one; the three token
// ids have none upstream (they come from `added_tokens.json`, read at
// `multimodal.py:82-90` @ `9035151d6`) and are therefore REQUIRED by the
// loader below rather than defaulted to a number this port invented.
struct Dots3NoteProcessorConfig {
  int patch_size = 14;
  int temporal_patch_size = 1;
  int merge_size = 2;  // == vision_config.spatial_merge_size
  bool pre_pixel_shuffle = true;
  // Per channel, in the checkpoint's own order (R, G, B).
  std::array<double, 3> image_mean{0.5, 0.5, 0.5};
  std::array<double, 3> image_std{0.5, 0.5, 0.5};
  double rescale_factor = 1.0 / 255.0;
  int64_t min_pixels = 3136;
  int64_t max_pixels = 12845056;

  // `<|img|>` / `<|imgpad|>` / `<|endofimg|>` in the checkpoint's tokenizer.
  int32_t image_token_id = -1;
  int32_t image_start_token_id = -1;
  int32_t image_end_token_id = -1;

  std::string model_id = "dots-studio/dots3-note-prev";  // for the mm-hash
};

// Load from the two HF json documents. THROWS BY NAME when the three image
// token ids cannot be resolved: a processor that guessed them would inject a
// marker the tokenizer maps to something else, and the request would be served
// as text with the image dropped.
Dots3NoteProcessorConfig LoadDots3NoteProcessorConfig(
    const std::string& preprocessor_config_json_path,
    const std::string& config_json_path, const std::string& model_id);

// `Dots3NoteImageProcessor.resized_size` (`common/processor.py:97-146` @
// `9035151d6`) — the height/width both divisible by `factor` whose product lies
// in `[min_pixels, max_pixels]`. Throws upstream's two refusals: a side under
// `factor / 4`, and an aspect ratio over 200.
//
// This is the BODY of upstream's `resized_size` and not its whole signature.
// The per-request `detail` string, the `image_details[detail]` override table
// and the explicit `target_height` / `target_width` arguments
// (`common/processor.py:97-119`) are request parsing, owed to W8 and tracked by
// issue #2645; no released `preprocessor_config.json` carries the table and the
// default `detail` resolves to the config pair this signature already takes.
std::array<int64_t, 2> Dots3NoteResizedSize(int64_t height, int64_t width,
                                            int64_t factor, int64_t min_pixels,
                                            int64_t max_pixels);

class Dots3NoteImageProcessor {
 public:
  explicit Dots3NoteImageProcessor(Dots3NoteProcessorConfig cfg)
      : cfg_(std::move(cfg)) {}

  const Dots3NoteProcessorConfig& config() const { return cfg_; }

  int64_t factor() const {
    return static_cast<int64_t>(cfg_.patch_size) * cfg_.merge_size;
  }

  // Preprocess ONE RGB image (HWC uint8, height*width*3) into
  // `pixel_values [num_patches, channel*temporal*patch*patch]` +
  // `image_grid_thw`, resizing it to `Dots3NoteResizedSize` first with PIL's
  // BICUBIC resampler (`PilResizeBicubicRgb`, W6c / #2537) exactly as
  // `processor.py:174` @ `9035151d6` does. An image whose sides are already
  // multiples of `factor` skips the resample, which is upstream's own short
  // circuit and not a special case here.
  //
  // Still refuses, and these are upstream's own `ValueError`s rather than
  // gaps: a side under `factor / 4`, an aspect ratio over 200, and a patch grid
  // that does not group into whole `merge x merge` blocks under
  // `pre_pixel_shuffle`.
  ImageKwargs ProcessImage(const uint8_t* rgb, int64_t height,
                           int64_t width) const;

  std::string HashImage(const uint8_t* rgb, int64_t height,
                        int64_t width) const;

 private:
  Dots3NoteProcessorConfig cfg_;
};

// ─── THE AUDIO PROCESSOR (W7a, #2703; the chunk loop W7b, #2797) ────────────
//
// Ported from `vllm/models/dots3_note/nvidia/audio.py` @ `9035151d6`:
//   SAMPLE_RATE / N_FFT / HOP_LENGTH               :15-17
//   DEFAULT_CHUNK_LENGTH_S / _CONV_TEMPORAL_STRIDE :18-19
//   Dots3NoteAudioConfig.__init__                  :26-60
//   .conv_temporal_stride / .token_stride          :67-73
//   .chunk_samples / .chunk_mel_frames             :75-81
//   pad_or_trim                                    :84-93
//   _mel_filters                                   :96-107
//   log_mel_spectrogram                            :117-126
//   compute_audio_token_length (DEAD upstream)     :129-147
//   encode_waveform's segment loop (W7b)           :193-218
//   the per-segment token count                    :210-212
// and from `common/processor.py` @ `9035151d6`:
//   AUDIO_START / AUDIO_PAD / AUDIO_END            :44-46
//   _HOP_LENGTH                                    :49
//   the stride and `ceil(length / stride)` rule    :762-771
//
// THE FRONT END IS NOT RE-WRITTEN HERE, and that is the point of this class.
// `log_mel_spectrogram` (`audio.py:117-126`) is Whisper's verbatim, and
// `WhisperAudioProcessor::ProcessWaveform`
// (src/vllm/multimodal/audio_processor.cpp:211-319) already is that function in
// double precision: reflect pad, dropped last frame, PERIODIC Hann, POWER
// spectrogram, `clamp(1e-10)` + `log10`, GLOBAL-max `-8` floor, `(x+4)/4`. The
// only dots3 deltas are CONFIG — `chunk_length_s` 30 -> 60, `n_mels` 80 -> 128 —
// so this class CONFIGURES that processor rather than writing a second one
// (AGENTS.md, "Shared seams": never a parallel path). The mel bank comes from
// the shared `MelFilterBankSlaney` seam, which reproduces the committed
// `voxtral_mel_filters_f32.bin` bit-for-bit; see mel_filter_bank.h.
//
// WHAT THIS CLASS ADDS ON TOP: the `chunk_seconds` SEGMENT LOOP (W7b, #2797,
// `audio.py:193-218`), `pad_or_trim` to `chunk_samples` per segment, the
// `ceil(num_samples / token_stride)` token count, the VALID mel-frame count the
// tower's temporal mask needs, the SAMPLE-RATE conversion (W7c-2, #2828, through
// the shared `ResampleAudioScipy` seam), and the refusal §4.15.3 owns.
struct Dots3NoteAudioProcessorConfig {
  // False when `config.json` carries no `audio_config`. Upstream builds no
  // `Dots3NoteAudioModel` in that case (`multimodal.py:119-126` @ `9035151d6`).
  bool present = false;

  int sampling_rate = 16000;   // `audio_config.sampling_rate`, audio.py:36
  int chunk_seconds = 60;      // `chunk_seconds`, audio.py:41
  int merge_factor = 1;        // `merge_factor`, audio.py:40
  int n_mels = 128;            // `whisper_config.num_mel_bins`
  // N_FFT and HOP_LENGTH are MODULE CONSTANTS upstream (`audio.py:16-17`), not
  // config keys. They are fields here so the front end can be driven at another
  // geometry by a test, and they are NOT read from `config.json`.
  int n_fft = 400;
  int hop_length = 160;
  // `conv_temporal_stride` (`audio.py:67-69`): 8 with the conv2d stem, 2 with
  // the conv1d one. The conv1d stem is refused by name, so this is 8 in
  // practice; it is a field because it is what `token_stride` multiplies.
  int conv_temporal_stride = 8;

  // `audio_comp_start` / `audio_comp_span` / `audio_comp_end`
  // (`audio.py:37-39`). The MIDDLE one is spelled `span` upstream and is the
  // PAD token; the naming is upstream's and is kept so the two can be diffed.
  std::string audio_comp_start = "<|audio_comp_start|>";
  std::string audio_comp_span = "<|audio_comp_pad|>";
  std::string audio_comp_end = "<|audio_comp_end|>";

  // Resolved from the TOKENIZER, never from `config.json`, and -1 until they
  // are. `processor.py:758-760` reads `vocab[AUDIO_START]` and friends off the
  // tokenizer; `multimodal.py:82-89` reads `added_tokens.json` off the
  // checkpoint. Both are the tokenizer's added tokens, and resolving from the
  // LIVE tokenizer is the one that makes "the marker string encodes to this id"
  // true by construction rather than by agreement between two files.
  //
  // MEASURED on `dots-studio/dots3-note-prev` @
  // `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b`: start 151718, END 151719, PAD
  // 151720. Note the ORDER — a port that assumed the three were consecutive in
  // start/pad/end order would swap pad and end and produce a well-formed prompt
  // that no shape check could reject.
  int32_t audio_token_id = -1;        // <|audio_comp_pad|>
  int32_t audio_start_token_id = -1;  // <|audio_comp_start|>
  int32_t audio_end_token_id = -1;    // <|audio_comp_end|>

  std::string model_id = "dots-studio/dots3-note-prev";  // for the mm-hash

  // `chunk_samples` (`audio.py:75-77`) = 60 * 16000 = 960000.
  int64_t chunk_samples() const {
    return static_cast<int64_t>(chunk_seconds) * sampling_rate;
  }
  // `token_stride` (`audio.py:71-73`) = 160 * 8 * 1 = 1280.
  int64_t token_stride() const {
    return static_cast<int64_t>(hop_length) * conv_temporal_stride *
           merge_factor;
  }
  // `chunk_mel_frames` (`audio.py:79-81`) = 60 * 100 = 6000, which is also
  // `chunk_samples / hop_length` and the assert upstream makes at
  // `audio.py:215`.
  int chunk_mel_frames() const { return chunk_seconds * 100; }
  int num_freq_bins() const { return 1 + n_fft / 2; }
};

// Read `audio_config` out of `config.json`. Returns `present=false` when the key
// is absent. Leaves the three token ids at -1: only a tokenizer can resolve
// them, and `Dots3NoteResolveAudioTokenIds` below is where that happens.
Dots3NoteAudioProcessorConfig LoadDots3NoteAudioProcessorConfig(
    const std::string& config_json_path, const std::string& model_id);

// Resolve the three `<|audio_comp_*|>` marker ids from the TOKENIZER'S added
// tokens BY STRING — `vocab[AUDIO_START]` and friends, `processor.py:757-760` @
// `9035151d6`. `lookup` answers with the added-token id for a marker string, or
// -1 when the tokenizer does not carry it.
//
// THROWS BY NAME on -1 rather than defaulting. The three ids are NOT
// consecutive in the order a reader expects: the released checkpoint has
// start 151718, END 151719, PAD 151720, so a port that guessed "start, start+1,
// start+2" would swap the pad and end markers and build a prompt that is
// well-formed, wrongly ordered, and invisible to every shape check.
//
// A CALLBACK RATHER THAN A `Tokenizer&`, so that `vllm::multimodal` does not
// depend on `vllm::tok` for three string lookups and so the refusal can be
// gated without building a tokenizer.
void Dots3NoteResolveAudioTokenIds(
    Dots3NoteAudioProcessorConfig* cfg,
    const std::function<int32_t(const std::string&)>& lookup);

// Why the audio tower cannot be served under this `audio_config`, or "" when it
// can. Names ONE thing and the brick that owes it, exactly as
// `Dots3NoteVisionRefusal` does, and for the same measured reason: a refusal
// raised from inside `encode_mm` runs in the engine's busy loop and turns every
// LATER request, text ones included, into a 500.
std::string Dots3NoteAudioProcessorRefusal(
    const Dots3NoteAudioProcessorConfig& cfg);

// The output is the shared `AudioKwargs`, with the PADDED
// `[n_mels, chunk_mel_frames]` mel `DotsSpeechEncoder.forward` takes
// (`audio_encoder.py:611-620`) plus the two lengths W7a added to that struct —
// `num_samples` (`audio_sample_lens`, `audio.py:218`, which the stem's temporal
// mask is derived from) and `num_tokens` (the placeholder run length). See
// `inputs.h` for why those two cannot be derived from each other.

class Dots3NoteAudioProcessor {
 public:
  explicit Dots3NoteAudioProcessor(Dots3NoteAudioProcessorConfig cfg);

  const Dots3NoteAudioProcessorConfig& config() const { return cfg_; }
  // The [num_freq_bins, n_mels] slaney bank this processor was built with,
  // exposed so the gate can compare it against `voxtral_mel_filters_f32.bin`
  // through the same object the front end actually uses.
  const std::vector<float>& mel_filters() const;

  // ONE `chunk_seconds` segment of the waveform, as
  // `DotsEncoderWithMask.encode_waveform` slices it (`audio.py:196-212` @
  // `9035151d6`). `length` is the segment's OWN sample count — `chunk_samples`
  // for every segment but the last — and `num_tokens` is
  // `NumAudioTokens(length)`, upstream's per-segment `token_len` (`:210-212`).
  struct AudioChunk {
    int64_t start = 0;
    int64_t length = 0;
    int64_t num_tokens = 0;
  };

  // The SEGMENTATION on its own (W7b, #2797), exposed because it is the seam a
  // gate can drive without a front end and because the tower and the prompt
  // side must read the SAME geometry rather than two copies of the loop.
  // Upstream's `while time_step * SAMPLE_RATE < audio_waveform.shape[0]`
  // (`audio.py:196-203`), so no segment is ever EMPTY — which is why the
  // per-segment count can be `NumAudioTokens` unchanged; see its note.
  std::vector<AudioChunk> SegmentWaveform(int64_t num_samples) const;

  // `encode_waveform`'s front-end half (`audio.py:193-218`): slice into
  // `chunk_seconds` segments, and for each one `pad_or_trim` to
  // `chunk_samples`, take the log-mel and assert `mel.shape[1] ==
  // chunk_mel_frames`. The mels are STACKED (`:220`), so `input_features` is
  // `[num_chunks, n_mels, chunk_mel_frames]` and `chunk_num_samples` /
  // `chunk_num_tokens` carry upstream's `audio_sample_lens` / `token_lens`.
  //
  // THE LOG-MEL IS TAKEN PER SEGMENT AND THAT IS NOT AN OPTIMISATION.
  // `log_mel_spectrogram` floors at `log_spec.max() - 8.0` (`audio.py:124`), a
  // GLOBAL max over the tensor it is handed, and upstream hands it one padded
  // segment at a time. One pass over the whole waveform would use one max for
  // every chunk and shift the quietest bands of the quietest chunk.
  //
  // RESAMPLES a `sample_rate` that is not `cfg.sampling_rate` (W7c-2, #2828)
  // through `vllm::multimodal::ResampleAudioScipy`, which is upstream's own
  // `"scipy"` `AudioResampler` arm and NOT its `"pyav"` default; that choice is
  // a recorded divergence and the seam header carries the reason. At the target
  // rate the call returns its input unchanged, so nothing at 16 kHz moves.
  //
  // REFUSES BY NAME, before any arithmetic:
  //   * a non-positive rate, or a reduced polyphase ratio past
  //     `kMaxPolyphaseRate` (from inside the resample seam)
  //   * more than one chunk on a config whose `chunk_samples` is not a whole
  //     number of `token_stride`s. That is the ONE thing segmentation cannot
  //     make safe: the tower produces `sum_i ceil(seg_i / stride)` rows
  //     (`audio.py:129-147`) and the prompt side expands one
  //     `ceil(total / stride)` span (`processor.py:771`), and those two are
  //     equal for every waveform exactly when `chunk_samples % token_stride ==
  //     0`. The released config satisfies it (960000 = 750 * 1280) and so does
  //     every EVEN `chunk_seconds` at 16 kHz. A SINGLE-chunk waveform is served
  //     either way, because a one-segment sum is `ceil(n / stride)` on both
  //     sides — which is why this is a per-request refusal and not an
  //     install-time one.
  //
  // HANDS BACK THE RESAMPLED BUFFER when `resampled_out` is not null, so that a
  // caller which also needs the encoder-cache key does not pay for the resample
  // a second time (PR #2842 F2). It is left EMPTY when no resample happened, which
  // is exactly the case in which the caller's own pointer is already the
  // waveform the tower consumed. Nothing about the returned `AudioKwargs`
  // depends on the argument.
  AudioKwargs ProcessWaveform(const float* samples, int64_t num_samples,
                              int sample_rate,
                              std::vector<float>* resampled_out = nullptr) const;

  // `ceil(num_samples / token_stride)`, exposed because the chat seam needs the
  // placeholder count and the encoder needs the row count and they must be the
  // same function rather than two copies of the same formula.
  //
  // W7b DELIBERATELY DID NOT CHANGE THIS, and #2797 records the check.
  // Upstream's per-segment form is `(n - 1) // stride + 1` (`audio.py:210-212`)
  // and it is algebraically identical to `ceil(n / stride)` for every
  // `n >= 1`. The two differ at `n == 0`, and only in C++: Python's `//`
  // floors, so `(0-1)//1280 + 1 == 0`, while C++ integer division truncates
  // toward zero, so a LITERAL transcription of upstream's expression returns 1
  // — one phantom token for an empty segment. `SegmentWaveform` never emits an
  // empty segment, so the two forms agree everywhere this port calls them; the
  // identity was ported, not the characters.
  int64_t NumAudioTokens(int64_t num_samples) const;

  std::string HashAudio(const float* samples, int64_t num_samples) const;

  // The encoder-cache key for a request that names its OWN sample rate
  // (W7c-2, #2828). It hashes the RESAMPLED waveform, which is what the tower
  // consumes.
  //
  // THE TWO-ARGUMENT FORM IS NOT SAFE FOR A MULTI-RATE CALLER, and W7c-2
  // created that hazard by serving more than one rate. A file carrying N PCM16
  // samples at 16000 Hz and a file carrying the identical N samples at
  // 44100 Hz decode to identical float buffers and hash identically under it,
  // while their features differ. `mm_hash` is a cross-request encoder-cache key
  // (`EncoderCacheManager::cached_`), so that is a hit that serves the wrong
  // audio. Every caller that has a request rate in hand must use this overload;
  // the two-argument one stays for the callers that are single-rate by
  // construction.
  //
  // `resampled` IS AN ANSWER, NOT A HINT: when it is not null it must be the
  // buffer `ProcessWaveform` filled for THIS waveform and THIS rate, and it is
  // hashed as-is. `RouteDots3NoteAudioWav` passes it because it has just called
  // `ProcessWaveform`, and that is what makes the served path resample ONCE
  // rather than twice — the second resample was a 1220.7 MB allocation on the
  // request measured in §4.17.10. A caller that does not hold the buffer passes
  // nothing and this resamples for itself, which is the only behaviour that
  // ever existed and is what the unit suite drives.
  std::string HashAudio(const float* samples, int64_t num_samples,
                        int sample_rate,
                        const std::vector<float>* resampled = nullptr) const;

 private:
  Dots3NoteAudioProcessorConfig cfg_;
  // Held by value rather than rebuilt per request: the front end is otherwise
  // stateless, and the bank is 201 x 128 floats of pure config.
  std::shared_ptr<const WhisperAudioProcessor> front_end_;
};

}  // namespace vllm::multimodal

#endif  // VLLM_MULTIMODAL_DOTS3_NOTE_PROCESSOR_H_
