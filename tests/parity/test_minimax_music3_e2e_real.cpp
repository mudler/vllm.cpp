// MiniMax-Music3 — W6 END TO END against the REAL checkpoint (#672).
//
// The companion of tests/vllm/models/test_minimax_music3_speech.cpp, which
// gates the seam and the request contract without an asset. This file drives
// the SHIPPED weights through W6's own composition — the denoise loop, the
// decode + stitch, the clamp and the WAV the delivery path emits — against the
// oracle's committed capture.
//
// ─── WHY THE ENTRY POINT IS A GOLDEN AND NOT A PROMPT ───────────────────────
//
// A request's waveform can NEVER equal `waveform.npy`, twice over and for two
// independent structural reasons:
//
//   1. the AR half draws every code with `torch.multinomial` against a seeded
//      `torch.Generator` (encoders.py:94-103) — spec §5 withdrew the token gate
//      for exactly this, and `rvq_codes.npy` is a seeded SAMPLE;
//   2. the denoise loop's INITIAL LATENTS are `randn_tensor(...)` from the same
//      generator (denoise.py:117-121).
//
// So "prompt in, compare the WAV" is not a gate that exists to be written, and
// claiming one would be claiming a property of torch's RNG. What IS comparable
// is this pipeline driven from the capture's OWN recorded inputs, which is why
// `Music3NoiseSource` is a parameter of `Music3DenoiseChunks`: the engine
// supplies a seeded normal draw, and this file supplies
// `denoise_first_sample_in.npy`.
//
// ─── WHAT EACH CASE GATES ───────────────────────────────────────────────────
//
//   delivery   vocoder_input_chunk0 [128, 86] -> Music3DecodeChunks -> the
//              44100 Hz STEREO waveform, against waveform.npy [2, 44032];
//              88064 values. Then MiniMaxH3WriteWav -> the RIFF header's rate,
//              channel count and bit depth, and the int16 payload BIT-EXACTLY
//              against the quantization of the golden itself.
//   full tail  frame_hiddens [25, 32768] + the capture's own initial latents
//              -> condition mix -> 4 Euler steps of the 2.4B DiT under CFG ->
//              vocoder -> waveform.npy. Opt-in behind VLLM_CPP_MUSIC3_DIT: it
//              is EIGHT 2.4B fp32 host forwards.
//
// A CORRELATION COEFFICIENT IS NOT A GATE HERE (AGENTS.md; spec §5). Pearson is
// scale-invariant, so a uniformly scaled waveform passes it while the song is
// wrong. Every bound below is on absolute and relative error, the mean absolute
// error is bounded too — a max-relative bound alone is satisfied by an
// implementation off by the bound EVERYWHERE — and every comparison reports how
// many values it examined.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "npy.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/model_executor/models/minimax_music3_acoustic.h"
#include "vllm/model_executor/models/minimax_music3_loader.h"
#include "vllm/model_executor/models/minimax_music3_speech.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;
namespace m3 = vllm::models::music3;

namespace {

// ─── The capture's own facts (manifest.json), asserted rather than assumed ──
constexpr int64_t kLatentChannels = 128;
constexpr int64_t kLatentLength = 86;
constexpr int64_t kConditionDim = 2048;
constexpr int64_t kWaveformChannels = 2;
constexpr int64_t kWaveformSamples = 44032;  // 86 * 512
constexpr int64_t kFrames = 25;
constexpr int64_t kFrameHiddenRow = 8 * 4096;
constexpr int64_t kDenoiseSteps = 4;
constexpr int64_t kSampleRate = 44100;

// ─── The bounds ─────────────────────────────────────────────────────────────
//
// The DELIVERY case's bounds are W5's, unchanged and for the same reason: the
// stage it exercises IS the vocoder, and this file adds only the crop (inert at
// one window), the clamp (inert on a tanh-bounded waveform) and the channel
// split. The measured control recorded in
// tests/parity/test_minimax_music3_acoustic_real.cpp is upstream's own
// `MiniMaxMusic3Vocoder` reproducing the golden AGAINST ITSELF under
// `torch.set_num_threads(1)` — 88064 values, 1.911% bit-identical, mean|d|
// 3.015e-08, max|d| 3.576e-07 — with this arm at mean|d| 3.191e-08 and max|d|
// 3.176e-07, INSIDE the control. Reusing the numbers rather than re-deriving
// them is the point: a second, looser bound on the same tensor would be slack
// nobody measured.
constexpr double kVocRelTol = 1e-4;
constexpr double kVocAbsFloor = 2e-6;
constexpr double kVocMeanAbsTol = 1.5e-7;

// The FULL TAIL's bounds are the DiT's, propagated, and they are NOT the
// vocoder's. The measured DiT arm is max|d| 2.837e-05 on the velocity per
// recorded step; four Euler steps at dt = 0.25 accumulate at most
// sum(dt * |dv|) ~ 3e-05 on the latents, and the vocoder is a 0.054B
// convolution stack over them.
//
// MEASURED on this box, 2026-08-14, this arm against the committed capture,
// running the whole tail (condition mix -> 4 guided DiT steps -> vocoder):
//
//   latents vs vocoder_input_chunk0  11008 values, 155 bit-identical,
//                                    max|d| 2.396e-05, mean|d| 2.375e-06,
//                                    reference max|x| 14.601
//   waveform vs waveform.npy         88064 values, 329 bit-identical,
//                                    max|d| 4.523e-06, mean|d| 1.225e-07,
//                                    reference max|x| 0.306322
//
// The latent numbers land exactly where the propagation says they should, and
// that is the point: they are the DiT's own per-step error carried by four
// Euler steps, NOT a new error the composition introduced.
//
// Each bound below is set from THAT measurement with under an order of
// magnitude of headroom, so none can be satisfied by an implementation
// materially worse than the one measured. An earlier revision of this file used
// 5e-4 / 5e-5, which the measurement showed to be ~100x slack — a bound nobody
// measured is not a bound.
//
// The RELATIVE bound is not the binding one on the waveform: max rel reaches
// 0.126 at samples whose magnitude is near zero, so a relative bound loose
// enough to admit those would admit anything. The ABSOLUTE floor is what binds,
// exactly as it does for the vocoder in the W4/W5 gate.
constexpr double kLatentRelTol = 1e-5;     // at |x| 14.6 that allows 1.46e-04
constexpr double kLatentAbsFloor = 5e-5;   // measured max|d| 2.396e-05
constexpr double kLatentMeanAbsTol = 5e-6; // measured mean|d| 2.375e-06
constexpr double kTailRelTol = 1e-4;       // at |x| 0.306 that allows 3.06e-05
constexpr double kTailAbsFloor = 1e-5;     // measured max|d| 4.523e-06
constexpr double kTailMeanAbsTol = 3e-7;   // measured mean|d| 1.225e-07

std::string CheckpointRoot() {
  if (const char* direct = std::getenv("VLLM_CPP_MUSIC3_CHECKPOINT")) return direct;
  if (const char* root = std::getenv("CHECKPOINT_ROOT")) {
    return (fs::path(root) / "minimax-music3").string();
  }
  return {};
}

std::string GoldensDir() { return std::string(MUSIC3_GOLDENS_DIR); }

const char* const kNeededGoldens[] = {
    "frame_hiddens.npy", "condition_chunk0.npy", "denoise_first_sample_in.npy",
    "vocoder_input_chunk0.npy", "waveform.npy",
};

std::string MissingReason() {
  std::error_code ec;
  for (const char* golden : kNeededGoldens) {
    if (!fs::is_regular_file(fs::path(GoldensDir()) / golden, ec)) {
      return std::string("golden ") + golden + " is absent under " + GoldensDir();
    }
  }
  const std::string root = CheckpointRoot();
  if (root.empty()) return "VLLM_CPP_MUSIC3_CHECKPOINT / CHECKPOINT_ROOT is unset";
  if (!fs::is_directory(root, ec)) return "checkpoint directory " + root + " is absent";
  return {};
}

bool SkipIfMissing(const char* what) {
  const std::string reason = MissingReason();
  if (reason.empty()) return false;
  std::printf("[SKIP] %s: %s\n", what, reason.c_str());
  MESSAGE("SKIPPED (" << reason << ")");
  return true;
}

bool SkipIfDitNotRequested(const char* what) {
  const char* flag = std::getenv("VLLM_CPP_MUSIC3_DIT");
  if (flag != nullptr && std::string(flag) != "0" && !std::string(flag).empty()) return false;
  std::printf("[SKIP] %s: VLLM_CPP_MUSIC3_DIT is unset (eight 2.4B fp32 host forwards)\n", what);
  MESSAGE("SKIPPED (VLLM_CPP_MUSIC3_DIT is unset)");
  return true;
}

// Row-major float32 from a golden; `condition_chunk0.npy` is stored FORTRAN
// order, so the transpose keys off the reader's flag rather than a guess.
std::vector<float> LoadF32Npy(const std::string& name, std::vector<int64_t>* shape) {
  const parity::NpyArray array =
      parity::LoadNpy((fs::path(GoldensDir()) / name).string(), /*allow_fortran_order=*/true);
  REQUIRE_MESSAGE(array.dtype == "<f4",
                  "golden " << name << " must be float32, is " << array.dtype);
  *shape = array.shape;
  const size_t count = array.data.size() / sizeof(float);
  std::vector<float> raw(count);
  std::memcpy(raw.data(), array.data.data(), array.data.size());
  if (!array.fortran_order) return raw;
  REQUIRE_MESSAGE(array.shape.size() == 2, "only a 2-D fortran-order golden is handled: " << name);
  const int64_t rows = array.shape[0];
  const int64_t cols = array.shape[1];
  std::vector<float> out(count);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(r * cols + c)] = raw[static_cast<size_t>(c * rows + r)];
    }
  }
  return out;
}

struct Report {
  int64_t compared = 0;
  int64_t identical = 0;
  int64_t outside = 0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double max_rel = 0.0;
  double ref_absmax = 0.0;
  int64_t first_bad = -1;
};

Report Compare(const std::vector<float>& got, const std::vector<float>& want, double rel_tol,
               double abs_floor) {
  Report report;
  REQUIRE(got.size() == want.size());
  double sum = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double a = got[i];
    const double b = want[i];
    const double diff = std::abs(a - b);
    ++report.compared;
    if (a == b) ++report.identical;
    sum += diff;
    report.max_abs = std::max(report.max_abs, diff);
    report.ref_absmax = std::max(report.ref_absmax, std::abs(b));
    const double magnitude = std::max(std::abs(a), std::abs(b));
    if (magnitude > 0.0) report.max_rel = std::max(report.max_rel, diff / magnitude);
    if (!(diff <= std::max(abs_floor, rel_tol * magnitude))) {
      if (report.outside == 0) report.first_bad = static_cast<int64_t>(i);
      ++report.outside;
    }
  }
  report.mean_abs = report.compared > 0 ? sum / static_cast<double>(report.compared) : 0.0;
  return report;
}

void ReportInto(const std::string& what, const Report& report) {
  MESSAGE(what << ": " << report.compared << " values compared, " << report.identical
               << " bit-identical, " << report.outside << " outside tolerance, max|d| "
               << report.max_abs << ", mean|d| " << report.mean_abs << ", max rel "
               << report.max_rel << ", reference max|x| " << report.ref_absmax
               << (report.first_bad >= 0
                       ? ", first outside at index " + std::to_string(report.first_bad)
                       : std::string()));
}

vllm::MiniMaxMusic3Paths Paths() {
  return vllm::MiniMaxMusic3ResolveCheckpoint(CheckpointRoot());
}

std::vector<float> ReadF32(const vllm::StTensor& tensor, const std::string& name) {
  REQUIRE_MESSAGE(tensor.dtype == "F32",
                  "acoustic tensor " << name << " must be F32 (spec 2.1), is " << tensor.dtype);
  std::vector<float> out(tensor.nbytes / sizeof(float));
  std::memcpy(out.data(), tensor.data, tensor.nbytes);
  return out;
}

// The AR half runs bf16, so an F32 file is ROUNDED rather than widened.
std::vector<float> AtRuntimeDtype(const vllm::StTensor& tensor) {
  const size_t count = tensor.nbytes / (tensor.dtype == "F32" ? 4 : 2);
  std::vector<float> out(count);
  if (tensor.dtype == "F32") {
    std::memcpy(out.data(), tensor.data, tensor.nbytes);
    for (float& value : out) value = vt::BF16ToF32(vt::F32ToBF16(value));
  } else if (tensor.dtype == "BF16") {
    const auto* raw = reinterpret_cast<const uint16_t*>(tensor.data);
    for (size_t i = 0; i < count; ++i) out[i] = vt::BF16ToF32(raw[i]);
  } else {
    FAIL("unexpected checkpoint dtype " << tensor.dtype);
  }
  return out;
}

m3::VocoderWeights LoadVocoder(const vllm::MiniMaxMusic3VocoderConfig& config) {
  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(
      (fs::path(Paths().vocoder_dir) / "diffusion_pytorch_model.safetensors").string());
  const vllm::MiniMaxMusic3VocoderWeights loaded =
      vllm::MiniMaxMusic3LoadVocoderWeights(config, file);
  REQUIRE(loaded.folded == 30);
  return m3::VocoderWeightsFromLoader(config, loaded);
}

m3::ConditionMixWeights LoadCondition() {
  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(
      (fs::path(Paths().condition_encoder_dir) / "diffusion_pytorch_model.safetensors").string());
  m3::ConditionMixWeights weights;
  weights.layer_weight_logits = AtRuntimeDtype(file.Get("layer_weight_logits"));
  weights.layer_scale = AtRuntimeDtype(file.Get("layer_scale"));
  weights.proj_weight = AtRuntimeDtype(file.Get("proj.weight"));
  weights.proj_bias = AtRuntimeDtype(file.Get("proj.bias"));
  return weights;
}

m3::DitWeights LoadDit(const vllm::MiniMaxMusic3TransformerConfig& config) {
  const vllm::MiniMaxMusic3Paths paths = Paths();
  REQUIRE_MESSAGE(!paths.transformer_shards.empty(), "transformer has no shards");
  std::map<std::string, std::vector<float>> tensors;
  for (const std::string& shard : paths.transformer_shards) {
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(shard);
    for (const std::string& name : file.Names()) tensors[name] = ReadF32(file.Get(name), name);
  }
  return m3::DitWeightsFromTensors(config, tensors);
}

// A little-endian reader over the RIFF header the delivery path writes.
uint32_t ReadU32(const std::string& wav, size_t offset) {
  return static_cast<uint32_t>(static_cast<unsigned char>(wav[offset])) |
         (static_cast<uint32_t>(static_cast<unsigned char>(wav[offset + 1])) << 8) |
         (static_cast<uint32_t>(static_cast<unsigned char>(wav[offset + 2])) << 16) |
         (static_cast<uint32_t>(static_cast<unsigned char>(wav[offset + 3])) << 24);
}
uint16_t ReadU16(const std::string& wav, size_t offset) {
  return static_cast<uint16_t>(static_cast<unsigned char>(wav[offset]) |
                               (static_cast<unsigned char>(wav[offset + 1]) << 8));
}

}  // namespace

// ---------------------------------------------------------------------------
// The DELIVERY path: latents -> waveform -> the WAV a request receives
// ---------------------------------------------------------------------------

TEST_CASE("music3 e2e real: the decode reaches waveform.npy at 44100 stereo") {
  if (SkipIfMissing("music3 e2e decode")) return;
  const vllm::MiniMaxMusic3Config config = vllm::MiniMaxMusic3LoadConfig(Paths());
  // spec §1.1, read from the CHECKPOINT rather than assumed by this file.
  CHECK(config.vocoder.sampling_rate == kSampleRate);
  CHECK(config.vocoder.hop_length() == 512);
  CHECK(config.vocoder.latent_channels == kLatentChannels);

  std::vector<int64_t> shape;
  const std::vector<float> latents = LoadF32Npy("vocoder_input_chunk0.npy", &shape);
  REQUIRE(shape.size() == 2);
  REQUIRE(shape[0] == kLatentChannels);
  REQUIRE(shape[1] == kLatentLength);
  const std::vector<float> want = LoadF32Npy("waveform.npy", &shape);
  REQUIRE(shape.size() == 2);
  REQUIRE(shape[0] == kWaveformChannels);
  REQUIRE(shape[1] == kWaveformSamples);

  const m3::VocoderWeights weights = LoadVocoder(config.vocoder);
  int64_t samples = 0;
  const std::vector<float> got =
      m3::Music3DecodeChunks({latents}, config.vocoder, weights, &samples);
  CHECK(samples == kWaveformSamples);
  REQUIRE(static_cast<int64_t>(got.size()) == kWaveformChannels * kWaveformSamples);

  const Report report = Compare(got, want, kVocRelTol, kVocAbsFloor);
  ReportInto("decode + stitch vs waveform.npy", report);
  CHECK(report.compared == kWaveformChannels * kWaveformSamples);
  CHECK(report.outside == 0);
  CHECK(report.mean_abs <= kVocMeanAbsTol);
  // THE STEREO FOLD IS NOT SYMMETRIC. The first 64 latent channels become the
  // LEFT channel and the second 64 the right; interleaving them instead gives a
  // correctly shaped, correctly ranged, wrong waveform. So the two channels are
  // asserted to DIFFER — a decode that emitted one channel twice would satisfy
  // every bound above.
  int64_t channels_differ = 0;
  for (int64_t i = 0; i < kWaveformSamples; ++i) {
    if (got[static_cast<size_t>(i)] != got[static_cast<size_t>(kWaveformSamples + i)]) {
      ++channels_differ;
    }
  }
  MESSAGE("stereo: " << channels_differ << " of " << kWaveformSamples
                     << " sample positions differ between L and R");
  CHECK(channels_differ > kWaveformSamples / 2);
}

TEST_CASE("music3 e2e real: the WAV a request receives is 44100 Hz 16-bit STEREO") {
  if (SkipIfMissing("music3 e2e wav")) return;
  std::vector<int64_t> shape;
  const std::vector<float> waveform = LoadF32Npy("waveform.npy", &shape);
  REQUIRE(shape[0] == kWaveformChannels);
  REQUIRE(shape[1] == kWaveformSamples);

  // The SHARED writer, not a Music3 one: MiniMaxH3WriteWav already serves H3's
  // audio VAE and LTX-2.5 (src/vllm/multimodal/ltx2_video.cpp:1537), so W6 adds
  // no second RIFF encoder to keep in step with it.
  const std::string wav =
      vllm::MiniMaxH3WriteWav(waveform, kWaveformChannels, kWaveformSamples, kSampleRate);

  REQUIRE(wav.size() == static_cast<size_t>(44 + 2 * kWaveformChannels * kWaveformSamples));
  CHECK(wav.compare(0, 4, "RIFF") == 0);
  CHECK(wav.compare(8, 4, "WAVE") == 0);
  CHECK(ReadU16(wav, 20) == 1);                      // PCM
  CHECK(ReadU16(wav, 22) == kWaveformChannels);      // STEREO, not mono
  CHECK(ReadU32(wav, 24) == kSampleRate);            // 44100, spec §1.1's rate
  CHECK(ReadU16(wav, 32) == 2 * kWaveformChannels);  // block align
  CHECK(ReadU16(wav, 34) == 16);                     // bits per sample
  CHECK(ReadU32(wav, 40) == static_cast<uint32_t>(2 * kWaveformChannels * kWaveformSamples));

  // The payload, BIT-EXACTLY against the quantization of the golden itself.
  // There is no reduction here, so there is nothing to round differently and a
  // tolerance would be slack for no reason. This is also what catches a
  // channel-major payload written straight through: WAV is INTERLEAVED.
  int64_t checked = 0;
  int64_t mismatched = 0;
  for (int64_t s = 0; s < kWaveformSamples; ++s) {
    for (int64_t c = 0; c < kWaveformChannels; ++c) {
      const float value = std::min(
          1.0f, std::max(-1.0f, waveform[static_cast<size_t>(c * kWaveformSamples + s)]));
      const int16_t expect = static_cast<int16_t>(std::lround(value * 32767.0f));
      const int16_t got =
          static_cast<int16_t>(ReadU16(wav, static_cast<size_t>(44 + 2 * (s * kWaveformChannels + c))));
      if (got != expect) ++mismatched;
      ++checked;
    }
  }
  MESSAGE("wav payload: " << checked << " int16 samples checked, " << mismatched
                          << " mismatched");
  CHECK(checked == kWaveformChannels * kWaveformSamples);
  CHECK(mismatched == 0);
}

// ---------------------------------------------------------------------------
// The condition mix the tail starts from, so a tail failure is attributable
// ---------------------------------------------------------------------------

TEST_CASE("music3 e2e real: the tail's first stage still reproduces condition_chunk0") {
  if (SkipIfMissing("music3 e2e condition")) return;
  std::vector<int64_t> shape;
  const std::vector<float> frame_hiddens = LoadF32Npy("frame_hiddens.npy", &shape);
  REQUIRE(shape.size() == 2);
  REQUIRE(shape[0] == kFrames);
  REQUIRE(shape[1] == kFrameHiddenRow);
  const std::vector<float> want = LoadF32Npy("condition_chunk0.npy", &shape);
  REQUIRE(shape[0] == kLatentLength);
  REQUIRE(shape[1] == kConditionDim);

  m3::ConditionMixConfig mix;
  const std::vector<m3::Music3Chunk> plan = m3::Music3ChunkPlan(kFrames, mix);
  REQUIRE(plan.size() == 1);
  CHECK(plan[0].latent_length == kLatentLength);

  const std::vector<float> got =
      m3::ConditionMix(frame_hiddens, kFrames, mix, LoadCondition(), m3::ArCompute::kBFloat16);
  // W3's own bound, at bf16: no value beyond one bf16 ULP-or-2^-7 of the golden.
  const Report report = Compare(got, want, 1.0 / 128.0, 1.0 / 128.0);
  ReportInto("condition mix (the tail's entry) vs condition_chunk0", report);
  CHECK(report.compared == kLatentLength * kConditionDim);
  CHECK(report.outside == 0);
}

// ---------------------------------------------------------------------------
// The FULL TAIL: frame hiddens + the capture's own noise -> the waveform
// ---------------------------------------------------------------------------

TEST_CASE("music3 e2e real: the whole tail reaches waveform.npy from frame_hiddens") {
  if (SkipIfMissing("music3 e2e full tail")) return;
  if (SkipIfDitNotRequested("music3 e2e full tail")) return;

  const vllm::MiniMaxMusic3Config config = vllm::MiniMaxMusic3LoadConfig(Paths());
  std::vector<int64_t> shape;
  const std::vector<float> frame_hiddens = LoadF32Npy("frame_hiddens.npy", &shape);
  REQUIRE(shape[0] == kFrames);
  REQUIRE(shape[1] == kFrameHiddenRow);
  const std::vector<float> initial = LoadF32Npy("denoise_first_sample_in.npy", &shape);
  REQUIRE(shape[0] == kLatentChannels);
  REQUIRE(shape[1] == kLatentLength);
  const std::vector<float> want = LoadF32Npy("waveform.npy", &shape);

  m3::Music3AcousticWeights weights;
  weights.condition = LoadCondition();
  weights.vocoder = LoadVocoder(config.vocoder);
  weights.dit = LoadDit(config.transformer);

  m3::Music3DenoiseOptions options;
  options.num_inference_steps = kDenoiseSteps;  // the capture's, not the 30 default
  options.guidance_scale = m3::kDitGuidanceScale;

  // THE CAPTURE'S OWN INITIAL LATENTS. Upstream drew them from a seeded torch
  // generator; supplying them is the only way this loop is comparable at all.
  int64_t draws = 0;
  const m3::Music3NoiseSource noise = [&initial, &draws](int64_t channels, int64_t length,
                                                        int64_t chunk_index) {
    REQUIRE(chunk_index == 0);  // 25 frames is ONE window
    REQUIRE(channels == kLatentChannels);
    REQUIRE(length == kLatentLength);
    ++draws;
    return initial;
  };

  const std::vector<std::vector<float>> chunks =
      m3::Music3DenoiseChunks(frame_hiddens, kFrames, config, weights, options, noise);
  REQUIRE(chunks.size() == 1);
  CHECK(draws == 1);
  REQUIRE(static_cast<int64_t>(chunks[0].size()) == kLatentChannels * kLatentLength);

  // The stage handoff, on the way past: the last Euler step's output IS the
  // vocoder's input (decoders.py:84 casts and nothing else touches it).
  const std::vector<float> vocoder_input = LoadF32Npy("vocoder_input_chunk0.npy", &shape);
  const Report latent_report =
      Compare(chunks[0], vocoder_input, kLatentRelTol, kLatentAbsFloor);
  ReportInto("denoised latents vs vocoder_input_chunk0", latent_report);
  CHECK(latent_report.compared == kLatentChannels * kLatentLength);
  CHECK(latent_report.outside == 0);
  CHECK(latent_report.mean_abs <= kLatentMeanAbsTol);

  int64_t samples = 0;
  const std::vector<float> got =
      m3::Music3DecodeChunks(chunks, config.vocoder, weights.vocoder, &samples);
  CHECK(samples == kWaveformSamples);
  const Report report = Compare(got, want, kTailRelTol, kTailAbsFloor);
  ReportInto("full tail vs waveform.npy", report);
  CHECK(report.compared == kWaveformChannels * kWaveformSamples);
  CHECK(report.outside == 0);
  CHECK(report.mean_abs <= kTailMeanAbsTol);

  // And the WAV the delivery path would hand the request.
  const std::string wav = vllm::MiniMaxH3WriteWav(got, kWaveformChannels, samples, kSampleRate);
  CHECK(ReadU32(wav, 24) == kSampleRate);
  CHECK(ReadU16(wav, 22) == kWaveformChannels);
  MESSAGE("full tail WAV: " << wav.size() << " bytes, " << samples << " frames per channel, "
                            << (static_cast<double>(samples) / kSampleRate) << " s");
}
