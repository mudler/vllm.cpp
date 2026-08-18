// MiniMax-H3 video+audio generation seam — implementation. See the header for
// the contract; every stage below is the ABSORBED pre-fold assembly
// (examples/minimax_h3_gen/main.cpp:687-1288 and the /v1/videos twin
// examples/server/main.cpp:743-1096 @ fc636c76), byte-gated on the committed
// fold fixture (tests/vllm/models/test_minimax_h3_video_fold.cpp).
//
// The library SPAWNS NOTHING here: directories are created with
// std::filesystem (the pre-fold examples shelled out to `mkdir -p`, which a
// library must not), artifacts are written with plain file IO, and the ffmpeg
// invocation stays with the caller as `mux_argv`.
#include "vllm/multimodal/minimax_h3_video.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/video_api.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/platforms/interface.h"  // CurrentPlatform() — which accelerator, if any
#include "vllm/tokenizer/tokenizer.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm::multimodal {
namespace {

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("minimax_h3 video: cannot open " + path);
  nlohmann::json j;
  in >> j;
  return j;
}

std::string ReadFileBytes(const std::string& field, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(field + ": cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::vector<float> ReadF32File(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("minimax_h3 video: cannot open " + path);
  const std::streamsize bytes = in.tellg();
  if (bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
    throw std::runtime_error(path + ": size is not a whole number of f32 values");
  }
  in.seekg(0);
  std::vector<float> out(static_cast<size_t>(bytes) / sizeof(float));
  in.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

void WriteFileBytes(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("minimax_h3 video: cannot write " + path);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) throw std::runtime_error("minimax_h3 video: short write " + path);
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Decode a binary PPM (P6) into [3, H, W] floats in [-1, 1] — the layout the
// H3 video-VAE encoder takes. From BYTES so one decoder serves a path and an
// inline data: URL (absorbed from the pre-fold server's DecodePpmChw; the
// pre-fold example's ReadPpmAsChw was the file-path twin). PPM is the only
// still-image container this tree reads — no PNG/JPEG codec is vendored, the
// same NAMED residual the chat multimodal path carries.
std::vector<float> DecodePpmChw(const std::string& field, const std::string& bytes,
                                int64_t* out_h, int64_t* out_w) {
  std::istringstream in(bytes, std::ios::binary);
  std::string magic;
  in >> magic;
  if (magic != "P6") {
    throw std::runtime_error(field +
                             ": not a binary PPM (P6); no PNG/JPEG codec is vendored, "
                             "so a reference image must be supplied as binary PPM");
  }
  auto next_int = [&]() {
    int v = 0;
    while (in >> std::ws, in.peek() == '#') {
      std::string skip;
      std::getline(in, skip);
    }
    in >> v;
    return v;
  };
  const int w = next_int(), h = next_int(), maxv = next_int();
  if (w <= 0 || h <= 0 || maxv <= 0) throw std::runtime_error(field + ": bad PPM header");
  in.get();  // the single whitespace byte before the payload
  std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
  in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  if (!in) throw std::runtime_error(field + ": truncated PPM payload");
  std::vector<float> chw(rgb.size());
  const int64_t plane = static_cast<int64_t>(w) * h;
  for (int64_t i = 0; i < plane; ++i) {
    for (int64_t c = 0; c < 3; ++c) {
      chw[static_cast<size_t>(c * plane + i)] =
          static_cast<float>(rgb[static_cast<size_t>(i * 3 + c)]) / (maxv * 0.5f) - 1.0f;
    }
  }
  if (out_h != nullptr) *out_h = h;
  if (out_w != nullptr) *out_w = w;
  return chw;
}

std::vector<float> ReadPpm(const std::string& field, const std::string& path,
                           const std::string& inline_bytes, int64_t* out_h, int64_t* out_w) {
  if (!inline_bytes.empty()) return DecodePpmChw(field, inline_bytes, out_h, out_w);
  return DecodePpmChw(field, ReadFileBytes(field, path), out_h, out_w);
}

// A ref2va VIDEO reference: DIR/frame_%06d.ppm — the exact layout this seam
// WRITES, so one run's frames chain into the next request. Returns
// [C, T, H, W] in [-1, 1] (absorbed from both pre-fold copies).
std::vector<float> ReadReferenceClipChw(const std::string& dir, int64_t* out_t, int64_t* out_h,
                                        int64_t* out_w) {
  std::vector<float> per_frame;  // frame-major [T][C,H,W]
  int64_t frames = 0, fh = 0, fw = 0;
  for (int64_t k = 0;; ++k) {
    char name[512];
    std::snprintf(name, sizeof(name), "%s/frame_%06lld.ppm", dir.c_str(),
                  static_cast<long long>(k));
    std::ifstream probe(name, std::ios::binary);
    if (!probe) break;
    const std::string bytes((std::istreambuf_iterator<char>(probe)),
                            std::istreambuf_iterator<char>());
    int64_t h = 0, w = 0;
    const std::vector<float> frame = DecodePpmChw("reference video frame", bytes, &h, &w);
    if (frames == 0) {
      fh = h;
      fw = w;
    }
    if (h != fh || w != fw) {
      throw std::runtime_error(
          "reference video: every frame_%06d.ppm must have the same size");
    }
    per_frame.insert(per_frame.end(), frame.begin(), frame.end());
    ++frames;
  }
  if (frames == 0) {
    throw std::runtime_error("reference video: no frame_%06d.ppm files in " + dir);
  }
  std::vector<float> chw(per_frame.size());
  const int64_t plane = fh * fw;
  for (int64_t c = 0; c < 3; ++c) {
    for (int64_t k = 0; k < frames; ++k) {
      for (int64_t e = 0; e < plane; ++e) {
        chw[static_cast<size_t>((c * frames + k) * plane + e)] =
            per_frame[static_cast<size_t>(k * 3 * plane + c * plane + e)];
      }
    }
  }
  *out_t = frames;
  *out_h = fh;
  *out_w = fw;
  return chw;
}

uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return z;
}

// The deterministic noise draw — BYTE-EXACT to the pre-fold example
// (main.cpp:1159-1195): a splitmix64 stream per modality, Box-Muller Gaussian
// by default (upstream draws torch.randn; mirror policy), VT_H3_GAUSSIAN_NOISE=0
// keeps the legacy uniform draw for the A/B. This deliberately does NOT
// reproduce torch's RNG bit-exactly: matching it decides WHICH sample you get,
// not whether the pipeline is right.
void FillNoise(std::vector<float>& out, uint64_t seed) {
  const char* gn = std::getenv("VT_H3_GAUSSIAN_NOISE");
  const bool gaussian = !(gn != nullptr && gn[0] == '0');
  uint64_t x = seed;
  auto u01 = [&x]() {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (z >> 11) * 0x1.0p-53;  // [0,1)
  };
  for (size_t i = 0; i < out.size(); ++i) {
    if (gaussian) {
      double u1 = u01(), u2 = u01();
      if (u1 < 1e-12) u1 = 1e-12;
      out[i] = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                                  std::cos(2.0 * 3.14159265358979323846 * u2));
    } else {
      out[i] = static_cast<float>(u01() * 2.0 - 1.0);  // uniform [-1,1]
    }
  }
}

}  // namespace

// ── the engine ───────────────────────────────────────────────────────────────

// ── where this engine runs (#659, #660) ──────────────────────────────────────
//
// `device` is the public video ABI's selector: 0 is the CPU, 1 is "the
// accelerator". WHICH accelerator is the PLATFORM's question, not this model
// file's — the same question `ltx2_video.cpp:609-657` (the comment through the
// end of the capability refusal) and the auto arm of
// `src/vllm/entrypoints/model_loader.cpp::SelectQueueForModel` ask.
//
// This used to be `static_cast<vt::DeviceType>(device)`, which is not a mapping
// at all. It reads the ABI selector AS AN ENUM VALUE and is correct only for as
// long as `kCUDA` stays 1 in include/vt/device.h — reorder that enum and every
// H3 device-1 load silently re-points at a different backend. It also named CUDA
// without writing `kCUDA`, so the DSR ratchet's token grep scored it ZERO while
// the test asserting the very same mapping spelled the token honestly and WAS
// counted: the gate read the confession and missed the act (#660). The
// `dev_cast` bucket in scripts/check-device-leakage.py now sees it.
//
// Three questions, the same three the seam answers everywhere else:
//   1. is there an accelerator at all (`CurrentPlatform().device_type()`),
//   2. is a backend registered for it (`vt::TryGetBackend`),
//   3. can that backend actually run THIS model
//      (`Platform::supports_model_architecture`, interface.h:263) — a PARTIAL
//      backend (Metal 15/75 ops, Tenstorrent) must be able to decline by name
//      rather than be handed a queue and die inside a kernel bind (#659).
//
// The architecture key is the family string, because that is this lane's stable
// registry name (`VideoModelParams::family`); the diffusion engines are not
// reached through ModelRegistry's HF `architectures` entry.
//
// On a CUDA box all three pass and this resolves EXACTLY the device the cast
// did. On a CPU-only build device 1 is now REFUSED here instead of returning
// kCUDA and failing one step later in `vt::GetBackend(kCUDA)`, and the refusal
// says which piece is missing.
vt::DeviceType MiniMaxH3VideoDeviceType(int32_t device) {
  if (device != 0 && device != 1) {
    throw std::runtime_error(
        "minimax_h3 video: device must be 0 (cpu) or 1 (the accelerator this build "
        "resolves)");
  }
  if (device == 0) return vt::DeviceType::kCPU;

  const vllm::platforms::Platform& platform = vllm::platforms::CurrentPlatform();
  const vt::DeviceType accelerator = platform.device_type();
  if (accelerator == vt::DeviceType::kCPU || vt::TryGetBackend(accelerator) == nullptr) {
    throw std::runtime_error(
        "minimax_h3 video: device 1 asks for an accelerator, but no accelerator backend "
        "is registered in this build (the platform seam resolves to '" +
        std::string(vt::DeviceTypeName(accelerator)) +
        "'). Refusing rather than naming a device this build cannot run on.");
  }
  if (!platform.supports_model_architecture(kMiniMaxH3VideoFamily)) {
    throw std::runtime_error(
        "minimax_h3 video: device 1 resolves to platform '" +
        std::string(vt::DeviceTypeName(accelerator)) + "', and that platform DECLINES the "
        "architecture '" + std::string(kMiniMaxH3VideoFamily) +
        "' (Platform::supports_model_architecture): it is a PARTIAL backend that has not "
        "registered the kernels this model needs. The build is partial, not broken. "
        "Refusing by name rather than binding a queue that would die inside a kernel bind.");
  }
  return accelerator;
}

struct MiniMaxH3VideoEngine::Impl {
  MiniMaxH3VideoModelParams params;
  vt::Device device{};
  MiniMaxH3GgufDit dit;  // .params always; .weights only for host arms
  MiniMaxH3DitDeviceWeights streamed;
  MiniMaxH3DitDeviceWeights staged;
  const MiniMaxH3DitDeviceWeights* prestaged = nullptr;

  MiniMaxH3VideoVaeDecoderConfig video_cfg;
  MiniMaxH3LatentStats video_stats;
  MiniMaxH3AudioVaeConfig audio_cfg;
  MiniMaxH3LatentStats audio_stats;
  MiniMaxH3AudioVaeWeights video_weights, audio_weights;

  std::vector<float> prompt_embeds;  // fallback conditioning
  MiniMaxH3PartitionInfo partition_info;

  // Encoder tower (optional), staged ONCE — staging the 32B tower costs
  // ~162 s, so per-request staging would dominate every generation.
  bool has_encoder = false;
  MiniMaxH3EncoderConfig enc_config;
  MiniMaxH3EncoderQuantWeights enc_host;                  // GGUF arm
  std::optional<MiniMaxH3ShardedCheckpoint> enc_shards;   // bf16-shard arm
  MiniMaxH3EncoderDeviceWeights enc_staged;
  std::unique_ptr<tok::Tokenizer> tokenizer;
  vt::Queue enc_queue{};

  // The two VAEs' ENCODER halves, for the reference modalities. Loaded LAZILY
  // and ONCE each: a text-to-video engine must not pay for weights it never
  // uses, and one that does must not reload per request.
  bool video_encoder_loaded = false, audio_encoder_loaded = false;
  MiniMaxH3AudioVaeWeights video_encoder_weights, audio_encoder_weights;
  MiniMaxH3EncoderFcn3dConfig video_encoder_cfg;
  MiniMaxH3AudioVaeEncoderConfig audio_encoder_cfg;

  // Generate() is serialized: the staged weights and lazy encoder halves are
  // shared state, and one H3 render saturates the device anyway.
  std::mutex mutex;

  std::vector<float> EncodePrompt(const std::string& prompt);
  void EnsureVideoEncoder();
  void EnsureAudioEncoder();
};

MiniMaxH3VideoEngine::MiniMaxH3VideoEngine() = default;
MiniMaxH3VideoEngine::MiniMaxH3VideoEngine(MiniMaxH3VideoEngine&&) noexcept = default;
MiniMaxH3VideoEngine& MiniMaxH3VideoEngine::operator=(MiniMaxH3VideoEngine&&) noexcept = default;
MiniMaxH3VideoEngine::~MiniMaxH3VideoEngine() = default;

std::string MiniMaxH3VideoEngine::family() const { return kMiniMaxH3VideoFamily; }

vt::Device MiniMaxH3VideoEngine::device() const { return impl_->device; }
bool MiniMaxH3VideoEngine::has_encoder() const { return impl_->has_encoder; }
bool MiniMaxH3VideoEngine::has_prompt_embeds() const { return !impl_->prompt_embeds.empty(); }

std::unique_ptr<MiniMaxH3VideoEngine> MiniMaxH3VideoEngine::Load(
    const MiniMaxH3VideoModelParams& params) {
  if (params.dit_path.empty()) {
    throw std::runtime_error("minimax_h3 video: dit_path is required");
  }
  const vt::DeviceType device_type = MiniMaxH3VideoDeviceType(params.device);
  auto engine = std::unique_ptr<MiniMaxH3VideoEngine>(new MiniMaxH3VideoEngine());
  engine->impl_ = std::make_unique<Impl>();
  Impl& im = *engine->impl_;
  im.params = params;

  // The CUDA context is created BEFORE any weight is read: on a unified-memory
  // box the driver's reservation must land before the pool fills (the GB10
  // load recipe). This throws — loudly — when no CUDA backend is registered.
  vt::Queue stream_queue{};
  if (params.device == 1) {
    stream_queue = vt::GetBackend(device_type).CreateQueue();
    im.device = stream_queue.device;
  }

  // ── 1. DiT: the four loader arms of the pre-fold driver ────────────────────
  // Refuse a TEXT-model directory LOUDLY before any loader guesses: an H3 DiT
  // is a GGUF file, an NVFP4 safetensors file, or the original bf16 release's
  // shard DIRECTORY (model.safetensors.index.json). A config.json directory is
  // a text/transcription checkpoint and belongs to vllm_engine_load.
  if (std::filesystem::is_directory(params.dit_path) &&
      !MiniMaxH3ShardedCheckpoint::IsShardedDir(params.dit_path)) {
    throw std::runtime_error(
        "minimax_h3 video: '" + params.dit_path +
        "' is a directory without model.safetensors.index.json — not an H3 DiT "
        "checkpoint. Expected a GGUF file, an NVFP4 safetensors file, or the bf16 "
        "shard directory; a text-generation model loads through vllm_engine_load "
        "instead");
  }
  const bool cuda = im.device.type != vt::DeviceType::kCPU;
  bool have_streamed = false;
  if (EndsWith(params.dit_path, ".gguf")) {
    const GgufFile f = GgufFile::Open(params.dit_path);
    if (params.dequant_bf16 != 0 && cuda) {
      // STREAM straight onto the device: dequantize + upload one tensor at a
      // time so the host copy never accumulates (peak kills unified memory).
      im.streamed = StreamMiniMaxH3DitToDeviceBf16(stream_queue, f, &im.dit.params);
      have_streamed = true;
    } else if (params.dequant_bf16 != 0) {
      im.dit = LoadMiniMaxH3DitFromGgufBf16(f);
    } else {
      im.dit = LoadMiniMaxH3DitFromGguf(f, /*keep_quant=*/true);
    }
  } else if (MiniMaxH3ShardedCheckpoint::IsShardedDir(params.dit_path)) {
    const MiniMaxH3ShardedCheckpoint ckpt = MiniMaxH3ShardedCheckpoint::Open(params.dit_path);
    if (cuda) {
      im.streamed = StreamMiniMaxH3ShardedToDeviceBf16(stream_queue, ckpt, &im.dit.params);
      have_streamed = true;
    } else {
      // Host f32 reference path — usable only on a reduced checkpoint.
      im.dit = LoadMiniMaxH3DitFromShards(ckpt);
    }
  } else {
    const SafetensorsFile f = SafetensorsFile::Open(params.dit_path);
    if (cuda) {
      // fp4_resident keeps the packed FP4 on device (~1/4 the bf16 footprint)
      // and routes quantized projections through the Marlin W4A16 GEMM; the
      // default bf16 stream dequantizes on the way up.
      im.streamed = params.fp4_resident != 0
                        ? StreamMiniMaxH3Nvfp4ToDeviceFp4(stream_queue, f, &im.dit.params)
                        : StreamMiniMaxH3Nvfp4ToDeviceBf16(stream_queue, f, &im.dit.params);
      have_streamed = true;
    } else {
      im.dit = LoadMiniMaxH3DitFromNvfp4(f);
    }
  }

  // ── 2. VAEs + their configs (the configs carry the latent statistics) ──────
  if (!params.video_vae_config_path.empty()) {
    im.video_cfg =
        ParseMiniMaxH3VideoVaeDecoderConfig(ReadJson(params.video_vae_config_path), &im.video_stats);
  }
  if (!params.audio_vae_config_path.empty()) {
    im.audio_cfg = ParseMiniMaxH3AudioVaeConfig(ReadJson(params.audio_vae_config_path), &im.audio_stats);
  }
  if (params.video_vae_path.empty() || params.audio_vae_path.empty()) {
    throw std::runtime_error(
        "minimax_h3 video: video_vae_path and audio_vae_path are required (a full "
        "render decodes both modalities)");
  }
  {
    const SafetensorsFile f = SafetensorsFile::Open(params.video_vae_path);
    im.video_weights = LoadMiniMaxH3VideoVaeDecoderWeights(f);
  }
  {
    const SafetensorsFile f = SafetensorsFile::Open(params.audio_vae_path);
    im.audio_weights = LoadMiniMaxH3AudioVaeWeights(f);
  }

  // ── 3. fallback conditioning + partition guard input ───────────────────────
  if (!params.prompt_embeds_path.empty()) {
    im.prompt_embeds = ReadF32File(params.prompt_embeds_path);
  }
  im.partition_info = MiniMaxH3PartitionFromFlag(params.partition);

  // ── 4. the optional H3-Encoder text tower, staged once ─────────────────────
  if (!params.encoder_path.empty()) {
    im.enc_queue = vt::Queue{im.device, nullptr};
    if (im.device.type != vt::DeviceType::kCPU) {
      im.enc_queue = vt::GetBackend(im.device.type).CreateQueue();
    }
    if (MiniMaxH3ShardedCheckpoint::IsShardedDir(params.encoder_path)) {
      // The ORIGINAL bf16 release's shards: stream to the device, keep the
      // checkpoint open for the per-request embedding gather.
      im.enc_shards.emplace(MiniMaxH3ShardedCheckpoint::Open(params.encoder_path));
      if (params.tokenizer_path.empty()) {
        throw std::runtime_error(
            "minimax_h3 video: tokenizer_path is required with a safetensors-shard encoder");
      }
      im.tokenizer = std::make_unique<tok::Tokenizer>(
          tok::Tokenizer::FromHfJson(params.tokenizer_path));
      im.enc_staged = StreamMiniMaxH3EncoderShardsToDevice(
          im.enc_queue, *im.enc_shards, params.encoder_max_layers, &im.enc_config);
    } else {
      const GgufFile ef = GgufFile::Open(params.encoder_path);
      im.enc_host = LoadMiniMaxH3EncoderFromGguf(ef, params.encoder_max_layers);
      im.enc_config = im.enc_host.config;
      // The ComfyUI-style encoder GGUF is WEIGHTS ONLY — no tokenizer.ggml.*
      // metadata — so the vocab comes from the checkpoint's own tokenizer.json
      // unless the GGUF happens to embed one.
      im.tokenizer = std::make_unique<tok::Tokenizer>(
          params.tokenizer_path.empty() ? tok::Tokenizer::FromGguf(ef)
                                        : tok::Tokenizer::FromHfJson(params.tokenizer_path));
      im.enc_staged = StageMiniMaxH3EncoderWeights(im.enc_queue, im.enc_host);
    }
    im.has_encoder = true;
  }

  // ── 5. stage the DiT once (device arms) ────────────────────────────────────
  if (have_streamed) {
    im.prestaged = &im.streamed;
  } else if (cuda) {
    vt::Queue sq = vt::GetBackend(im.device.type).CreateQueue();
    im.staged = StageMiniMaxH3DitWeights(sq, im.dit.params, im.dit.weights, vt::DType::kBF16);
    im.prestaged = &im.staged;
  }
  return engine;
}

// Encode `prompt` with the H3 text tower (text-only M-RoPE: all three axes are
// the token index) — the absorbed EncodeH3Prompt/server-lambda conditioning.
std::vector<float> MiniMaxH3VideoEngine::Impl::EncodePrompt(const std::string& prompt) {
  const std::vector<int32_t> ids = tokenizer->Encode(prompt);
  if (ids.empty()) throw std::runtime_error("minimax_h3 video: the prompt tokenized to nothing");
  // DIAGNOSTIC (env-gated): dump the token ids so the tokenization can be
  // diffed against upstream — the pre-fold driver's VT_H3_DUMP_INPUTS knob.
  if (const char* dd = std::getenv("VT_H3_DUMP_INPUTS")) {
    if (std::FILE* fp = std::fopen((std::string(dd) + "/prompt_token_ids.i32").c_str(), "wb")) {
      std::fwrite(ids.data(), sizeof(int32_t), ids.size(), fp);
      std::fclose(fp);
    }
  }
  std::vector<float> embeds = enc_shards.has_value()
                                  ? MiniMaxH3EncoderEmbedTokensFromShards(*enc_shards, ids)
                                  : MiniMaxH3EncoderEmbedTokens(enc_host, ids);
  const int64_t seq = static_cast<int64_t>(ids.size());
  std::vector<int64_t> pos(static_cast<size_t>(3 * seq));
  for (int64_t a = 0; a < 3; ++a) {
    for (int64_t s = 0; s < seq; ++s) pos[static_cast<size_t>(a * seq + s)] = s;
  }
  return MiniMaxH3EncoderTextForwardDevice(enc_queue, enc_config, enc_staged, embeds, pos.data(),
                                           seq);
}

void MiniMaxH3VideoEngine::Impl::EnsureVideoEncoder() {
  if (video_encoder_loaded) return;
  const SafetensorsFile vf = SafetensorsFile::Open(params.video_vae_path);
  video_encoder_weights = LoadMiniMaxH3VideoVaeEncoderWeights(vf);
  video_encoder_cfg = MiniMaxH3EncoderFcn3dConfig{};
  video_encoder_cfg.z_channels = 2 * dit.params.latents_dim;  // moments: mean | logvar
  video_encoder_loaded = true;
}

void MiniMaxH3VideoEngine::Impl::EnsureAudioEncoder() {
  if (audio_encoder_loaded) return;
  if (params.audio_vae_config_path.empty()) {
    throw std::runtime_error(
        "minimax_h3 video: an audio reference needs audio_vae_config_path (it carries "
        "the latent statistics the reference rows are normalized by)");
  }
  const SafetensorsFile af = SafetensorsFile::Open(params.audio_vae_path);
  audio_encoder_weights = LoadMiniMaxH3AudioVaeEncoderWeights(af);
  audio_encoder_cfg = MiniMaxH3AudioVaeEncoderConfig{};
  audio_encoder_cfg.vae_latent_channels = dit.params.audio_latents_dim;
  audio_encoder_loaded = true;
}

MiniMaxH3VideoResult MiniMaxH3VideoEngine::Generate(const MiniMaxH3VideoGenParams& gen) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> guard(im.mutex);
  const MiniMaxH3DitParams& p = im.dit.params;

  if (gen.output_dir.empty()) {
    throw std::runtime_error("minimax_h3 video: output_dir is required");
  }

  // ── conditioning ───────────────────────────────────────────────────────────
  std::vector<float> conditioning;
  if (im.has_encoder && !gen.prompt.empty()) {
    conditioning = im.EncodePrompt(gen.prompt);
  } else if (!im.prompt_embeds.empty()) {
    conditioning = im.prompt_embeds;
  } else {
    throw std::runtime_error(
        "minimax_h3 video: generation needs conditioning — load with an encoder (to "
        "condition on the prompt) or with prompt_embeds_path");
  }
  if (p.text_dim <= 0 ||
      conditioning.size() % static_cast<size_t>(p.text_dim) != 0) {
    throw std::runtime_error(
        "minimax_h3 video: conditioning size is not a multiple of text_dim");
  }

  // ── reference exclusivity (the packed layout's rule, checked up front) ─────
  const bool has_keyframes = !gen.first_frame_path.empty() || !gen.first_frame_ppm.empty() ||
                             !gen.last_frame_path.empty();
  const bool has_ref2va = !gen.ref_image_paths.empty() || !gen.ref_video_dir.empty() ||
                          !gen.ref_audio_path.empty() || !gen.ref_audio_wav.empty();
  if (has_keyframes && has_ref2va) {
    throw std::runtime_error(
        "minimax_h3 video: fl2va keyframes and ref2va references are exclusive "
        "(minimax_h3_pipeline.cpp:251)");
  }
  if (!gen.ref_video_dir.empty() && !gen.ref_image_paths.empty()) {
    throw std::runtime_error(
        "minimax_h3 video: a reference video is exclusive with reference images");
  }

  // ── the keyframe image (fl2va), read before the plan so its aspect can
  // drive the default canvas (upstream _resolve_shape) ───────────────────────
  std::vector<float> first_chw, last_chw;
  int64_t ref_h = 0, ref_w = 0;
  if (!gen.first_frame_path.empty() || !gen.first_frame_ppm.empty()) {
    first_chw = ReadPpm("first_frame", gen.first_frame_path, gen.first_frame_ppm, &ref_h, &ref_w);
  }
  if (!gen.last_frame_path.empty()) {
    int64_t h2 = 0, w2 = 0;
    last_chw = ReadPpm("last_frame", gen.last_frame_path, "", &h2, &w2);
    if (ref_h == 0) {
      ref_h = h2;
      ref_w = w2;
    } else if (h2 != ref_h || w2 != ref_w) {
      throw std::runtime_error(
          "minimax_h3 video: the first and last frames must have the same size");
    }
  }

  // ── task + shape plan (upstream _resolve_task/_resolve_shape) ──────────────
  const std::string task =
      !gen.task.empty() ? gen.task
                        : (has_keyframes ? "fl2va" : (has_ref2va ? "ref2va" : "t2va"));
  const MiniMaxH3ShapePlan plan = MiniMaxH3ResolveShape(
      task, gen.duration_seconds, gen.num_frames, gen.height, gen.width, ref_w, ref_h);

  MiniMaxH3T2vaRequest request;
  request.partition = im.partition_info;  // #77 guard: GenerateT2va refuses a
                                          // task this partition cannot serve.
  request.latent_t = plan.latent_t;
  request.num_frames = plan.num_frames;
  request.latent_h = plan.height / kMiniMaxH3VaeRatio;
  request.latent_w = plan.width / kMiniMaxH3VaeRatio;
  request.audio_t = plan.audio_t;
  request.audio_channel = kMiniMaxH3AudioChannels;
  if (gen.steps > 0) request.num_steps = gen.steps;
  if (gen.flow_shift > 0.0) request.video_shift = gen.flow_shift;
  if (gen.audio_flow_shift > 0.0) request.audio_shift = gen.audio_flow_shift;
  request.video_latents_mean = im.video_stats.mean;
  request.video_latents_std = im.video_stats.std_dev;
  request.audio_latents_mean = im.audio_stats.mean;
  request.audio_latents_std = im.audio_stats.std_dev;
  request.text_len = static_cast<int64_t>(conditioning.size()) / p.text_dim;

  // ── fl2va KEYFRAME CONDITIONING ────────────────────────────────────────────
  if (has_keyframes) {
    if (ref_h != plan.height || ref_w != plan.width) {
      // No image resampler is vendored, and a mis-sized keyframe would either
      // abort deep in the denoise or pin the wrong latent rows. Say so up
      // front, with the geometry we resolved.
      throw std::runtime_error(
          "minimax_h3 video: the keyframe is " + std::to_string(ref_w) + "x" +
          std::to_string(ref_h) + " but this request resolved to " +
          std::to_string(plan.width) + "x" + std::to_string(plan.height) +
          "; supply it at the output size (no image resampler is vendored)");
    }
    im.EnsureVideoEncoder();
    std::vector<std::vector<float>> imgs;
    std::vector<int64_t> idx;
    if (!first_chw.empty()) {
      imgs.push_back(std::move(first_chw));
      idx.push_back(0);
    }
    if (!last_chw.empty()) {
      imgs.push_back(std::move(last_chw));
      idx.push_back(-1);
    }
    request.keyframe_frame_indices = idx;
    request.imgvid_noise_aug = gen.noise_aug;
    request.keyframe_cond_rows =
        MiniMaxH3EncodeKeyframeCondRows(im.video_encoder_cfg, im.video_encoder_weights, p, imgs,
                                        ref_h, ref_w, request.latent_t, gen.noise_aug, {});
  }

  // ── ref2va REFERENCE BLOCKS ────────────────────────────────────────────────
  if (!gen.ref_video_dir.empty()) {
    im.EnsureVideoEncoder();
    int64_t ct = 0, ch = 0, cw = 0;
    const std::vector<float> clip = ReadReferenceClipChw(gen.ref_video_dir, &ct, &ch, &cw);
    MiniMaxH3RefBlock block{};
    request.keyframe_cond_rows = MiniMaxH3EncodeReferenceVideo(
        im.video_encoder_cfg, im.video_encoder_weights, p, clip, ct, ch, cw, &block);
    // SILENT by construction (ref_audio_t == 0): an audio reference below
    // ATTACHES to this block — the layout packed_sequence.py builds.
    request.ref_blocks = {block};
  } else if (!gen.ref_image_paths.empty()) {
    im.EnsureVideoEncoder();
    std::vector<std::vector<float>> imgs;
    int64_t ih = 0, iw = 0;
    for (const std::string& rp : gen.ref_image_paths) {
      int64_t h2 = 0, w2 = 0;
      imgs.push_back(ReadPpm("ref_image", rp, "", &h2, &w2));
      if (ih == 0) {
        ih = h2;
        iw = w2;
      } else if (h2 != ih || w2 != iw) {
        throw std::runtime_error(
            "minimax_h3 video: every reference image must have the same size");
      }
    }
    std::vector<MiniMaxH3RefBlock> blocks;
    request.keyframe_cond_rows = MiniMaxH3EncodeReferenceImages(
        im.video_encoder_cfg, im.video_encoder_weights, p, imgs, ih, iw, &blocks);
    request.ref_blocks = blocks;
  }
  if (!gen.ref_audio_path.empty() || !gen.ref_audio_wav.empty()) {
    im.EnsureAudioEncoder();
    const std::string wav_bytes = !gen.ref_audio_wav.empty()
                                      ? gen.ref_audio_wav
                                      : ReadFileBytes("ref_audio", gen.ref_audio_path);
    int64_t samples_per_channel = 0;
    const std::vector<float> waveform = MiniMaxH3ReadWav(
        wav_bytes, kMiniMaxH3AudioChannels, kMiniMaxH3AudioSampleRate, &samples_per_channel);
    MiniMaxH3RefBlock audio_block{};
    request.audio_ref_rows = MiniMaxH3EncodeReferenceAudio(
        im.audio_encoder_cfg, im.audio_encoder_weights, waveform, kMiniMaxH3AudioChannels,
        samples_per_channel, im.audio_stats.mean, im.audio_stats.std_dev,
        /*noise_aug=*/1.0, {}, &audio_block);
    if (!request.ref_blocks.empty() &&
        request.ref_blocks[0].kind == MiniMaxH3RefBlock::Kind::kVideoAudio) {
      // A video reference that now HAS sound: one kVideoAudio block carries
      // both, so its ref_audio_t must claim exactly the rows just encoded.
      request.ref_blocks[0].ref_audio_t = audio_block.ref_audio_t;
    } else {
      request.ref_blocks.push_back(audio_block);
    }
  }

  // ── the deterministic noise draw ───────────────────────────────────────────
  const int64_t frame_rows =
      (request.latent_h / p.patch_size_h) * (request.latent_w / p.patch_size_w);
  const int64_t video_rows = request.latent_t * frame_rows;
  const int64_t audio_rows = request.audio_t * request.audio_channel;
  std::vector<float> noise_video(static_cast<size_t>(video_rows * p.video_row_width()));
  std::vector<float> noise_audio(static_cast<size_t>(audio_rows * p.audio_latents_dim));
  // Unseeded: the pre-fold driver's fixed per-modality streams (byte-identical
  // default). Seeded: the video stream takes the seed, the audio stream a
  // splitmix64 derivation of it, so the two modalities stay independent.
  const uint64_t video_seed = gen.has_seed ? gen.seed : 0x5EED1234ULL;
  const uint64_t audio_seed = gen.has_seed ? SplitMix64(gen.seed) : 0x5EED5678ULL;
  FillNoise(noise_video, video_seed);
  FillNoise(noise_audio, audio_seed);

  // ── generate ───────────────────────────────────────────────────────────────
  const MiniMaxH3T2vaResult out = MiniMaxH3GenerateT2va(
      im.device, request, p, im.dit.weights, im.video_cfg, im.video_weights, im.audio_cfg,
      im.audio_weights, conditioning, noise_video, noise_audio, vt::DType::kBF16, im.prestaged);

  // ── artifacts (the library WRITES these, spawns nothing) ───────────────────
  std::error_code ec;
  std::filesystem::create_directories(gen.output_dir, ec);
  if (ec) {
    throw std::runtime_error("minimax_h3 video: cannot create " + gen.output_dir + ": " +
                             ec.message());
  }
  MiniMaxH3VideoResult result;
  result.frame_dir = gen.output_dir;
  for (int64_t f = 0; f < out.frame_shape.t; ++f) {
    char name[512];
    std::snprintf(name, sizeof(name), "%s/frame_%06lld.ppm", gen.output_dir.c_str(),
                  static_cast<long long>(f));
    WriteFileBytes(name, MiniMaxH3WritePpmFrame(out.frames, out.frame_shape, f));
  }
  result.audio_path = gen.output_dir + "/audio.wav";
  WriteFileBytes(result.audio_path,
                 MiniMaxH3WriteWav(out.waveform, out.audio_channels,
                                   out.audio_samples_per_channel, out.sample_rate));
  result.frame_count = out.frame_shape.t;
  result.width = out.frame_shape.w;
  result.height = out.frame_shape.h;
  result.fps = kMiniMaxH3Fps;
  result.sample_rate = out.sample_rate;

  // ── the mux argv the CALLER may exec (the ratified ffmpeg boundary) ────────
  MiniMaxH3MuxRequest mux;
  mux.frame_pattern = gen.output_dir + "/frame_%06d.ppm";
  mux.audio_path = result.audio_path;
  mux.output_path = gen.output_dir + "/video.mp4";
  result.mux_argv = MiniMaxH3BuildMp4MuxArgs(mux);
  result.mux_output_path = mux.output_path;
  return result;
}

// ── the /v1/videos request mapping (library-owned so HTTP and FFI cannot
// drift; absorbed from the pre-fold server lambda). At L1 the mapping itself
// moved to the FAMILY-AGNOSTIC seam (VideoGenParamsFromRequest) — nothing in it
// was H3-specific — and this stays as the H3-typed spelling its callers use, so
// there is one implementation rather than two that can disagree. ─────────────
MiniMaxH3VideoGenParams MiniMaxH3VideoGenParamsFromRequest(
    const ::vllm::openai::VideoRequest& request, const std::string& output_dir) {
  return MiniMaxH3VideoGenParamsFromGeneric(VideoGenParamsFromRequest(request, output_dir));
}

// ── The generalized VideoEngine seam (LTX-2.5 L1) ────────────────────────────
// Adapters ONLY. Every one of these converts and calls the H3-typed member
// above; none of them re-implements a step, which is what keeps the fold gate's
// byte-identity claim true for the generic path as well.

MiniMaxH3VideoModelParams MiniMaxH3VideoModelParamsFromGeneric(const VideoModelParams& params) {
  MiniMaxH3VideoModelParams mp;
  mp.dit_path = params.dit_path;
  mp.encoder_path = params.encoder_path;
  mp.tokenizer_path = params.tokenizer_path;
  mp.video_vae_path = params.video_vae_path;
  mp.video_vae_config_path = params.video_vae_config_path;
  mp.audio_vae_path = params.audio_vae_path;
  mp.audio_vae_config_path = params.audio_vae_config_path;
  mp.prompt_embeds_path = params.prompt_embeds_path;
  // The ONE H3-specific load field. An absent key is the empty string, which is
  // declared-but-unknown — the #77 guard then refuses every full render, which
  // is the same thing an omitted `--partition` has always done.
  mp.partition = VideoExtra(params.extras, "partition");
  mp.device = params.device;
  mp.dequant_bf16 = params.dequant_bf16;
  mp.fp4_resident = params.fp4_resident;
  mp.encoder_max_layers = params.encoder_max_layers;
  return mp;
}

VideoModelParams MiniMaxH3VideoModelParamsToGeneric(const MiniMaxH3VideoModelParams& params) {
  VideoModelParams out;
  out.dit_path = params.dit_path;
  out.encoder_path = params.encoder_path;
  out.tokenizer_path = params.tokenizer_path;
  out.video_vae_path = params.video_vae_path;
  out.video_vae_config_path = params.video_vae_config_path;
  out.audio_vae_path = params.audio_vae_path;
  out.audio_vae_config_path = params.audio_vae_config_path;
  out.prompt_embeds_path = params.prompt_embeds_path;
  out.family = kMiniMaxH3VideoFamily;
  out.device = params.device;
  out.dequant_bf16 = params.dequant_bf16;
  out.fp4_resident = params.fp4_resident;
  out.encoder_max_layers = params.encoder_max_layers;
  // Only set the key when there is something to say: an empty value would read
  // as "declared as the empty partition" to a caller inspecting the map.
  if (!params.partition.empty()) out.extras["partition"] = params.partition;
  return out;
}

MiniMaxH3VideoGenParams MiniMaxH3VideoGenParamsFromGeneric(const VideoGenParams& params) {
  MiniMaxH3VideoGenParams gen;
  gen.prompt = params.prompt;
  gen.task = params.task;
  gen.duration_seconds = params.duration_seconds;
  gen.num_frames = params.num_frames;
  gen.height = params.height;
  gen.width = params.width;
  gen.steps = params.steps;
  gen.flow_shift = params.flow_shift;
  gen.audio_flow_shift = params.audio_flow_shift;
  gen.seed = params.seed;
  gen.has_seed = params.has_seed;
  gen.first_frame_path = params.first_frame_path;
  gen.last_frame_path = params.last_frame_path;
  gen.first_frame_ppm = params.first_frame_ppm;
  gen.noise_aug = params.noise_aug;
  gen.ref_image_paths = params.ref_image_paths;
  gen.ref_video_dir = params.ref_video_dir;
  gen.ref_audio_path = params.ref_audio_path;
  gen.ref_audio_wav = params.ref_audio_wav;
  gen.output_dir = params.output_dir;
  // H3 defines no per-request family knob; an unknown extra is REFUSED rather
  // than ignored, so a caller who mistypes a future LTX key on an H3 engine
  // learns about it instead of silently getting the default render.
  if (!params.extras.empty()) {
    throw std::runtime_error("minimax_h3 video: unknown per-generation extra '" +
                             params.extras.begin()->first + "' (this family defines none)");
  }
  return gen;
}

VideoResult MiniMaxH3VideoResultToGeneric(const MiniMaxH3VideoResult& result) {
  VideoResult out;
  out.frame_dir = result.frame_dir;
  out.audio_path = result.audio_path;
  out.frame_count = result.frame_count;
  out.width = result.width;
  out.height = result.height;
  out.fps = result.fps;
  out.sample_rate = result.sample_rate;
  out.mux_argv = result.mux_argv;
  out.mux_output_path = result.mux_output_path;
  return out;
}

VideoResult MiniMaxH3VideoEngine::Generate(const VideoGenParams& params) {
  return MiniMaxH3VideoResultToGeneric(Generate(MiniMaxH3VideoGenParamsFromGeneric(params)));
}

namespace {

// Does this checkpoint set hold an H3 DiT? The discriminator is the DUAL patch
// projection: H3 packs video and audio into ONE sequence through
// `video_patch_proj` + `audio_patch_proj` (minimax_h3_gguf.cpp:100-103, the
// names the loader binds by), and the four loader arms — ComfyUI GGUF, NVFP4
// safetensors, the bf16 release shards, and the reduced fixtures — all carry
// the checkpoint's own parameter names, so this one test covers every arm.
// LTX-2.5 has neither name (it patchifies through `patchify_proj` and carries
// its audio stream as `audio_ff.*` / `audio_attn*`), so the two cannot collide.
//
// Deliberately NOT keyed on the file extension or on "is it a directory": which
// container a checkpoint was repackaged into says nothing about which model it
// holds, and both families ship GGUF, safetensors and sharded arms.
bool DetectMiniMaxH3Video(const VideoModelParams& params) {
  std::vector<std::string> names;
  std::string why;
  if (!ReadVideoCheckpointTensorNames(params.dit_path, &names, &why)) return false;
  bool video_patch = false, audio_patch = false;
  for (const std::string& n : names) {
    if (n == "video_patch_proj.weight") video_patch = true;
    if (n == "audio_patch_proj.weight") audio_patch = true;
    if (video_patch && audio_patch) return true;
  }
  return false;
}

std::unique_ptr<VideoEngine> LoadMiniMaxH3VideoFamily(const VideoModelParams& params) {
  return MiniMaxH3VideoEngine::Load(MiniMaxH3VideoModelParamsFromGeneric(params));
}

}  // namespace

REGISTER_VLLM_VIDEO_FAMILY(minimax_h3, kMiniMaxH3VideoFamily, DetectMiniMaxH3Video,
                           LoadMiniMaxH3VideoFamily)

}  // namespace vllm::multimodal
