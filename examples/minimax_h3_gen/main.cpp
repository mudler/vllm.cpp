// minimax-h3-gen: the ASSEMBLY driver — open the real checkpoints, run the whole
// t2va path, and write an MP4.
//
// Everything below this line was gated component by component (packed layout, DiT
// forward, both VAE decoders, the encoder towers, the loaders). This is the piece
// that puts them together over REAL files, which is the only way the remaining
// integration questions — config plumbing, latent statistics, shard assembly —
// actually get answered.
//
// It lives in examples/ for the same reason the muxer does: this is where the
// ffmpeg invocation is allowed (developer-ratified 2026-08-03). src/vllm/ builds
// artifacts and argv and spawns nothing.
//
// Usage:
//   minimax-h3-gen --dit <dit.gguf|dit.safetensors|shard-dir/>
//                  # a DIRECTORY holding the original bf16 release's shards plus
//                  # model.safetensors.index.json is accepted wherever a single
//                  # DiT file is; every existing --dit form is unchanged.
//                  --video-vae <video_vae.safetensors> --video-vae-config <config.json>
//                  --audio-vae <audio_vae.safetensors> --audio-vae-config <config.json>
//                  --prompt-embeds <f32.bin>   (rows of text_dim, little-endian f32)
//                  --out <out.mp4>
//                  [--partition fl2va|ref2va]  (REQUIRED for a full render: the
//                    served checkpoint partition — community GGUF/NVFP4 strip it and
//                    the FL2VA/Ref2VA DiTs are indistinguishable. t2va/fl2va need
//                    fl2va, ref2va needs ref2va; recipe:50-51,289)
//                  [--keep-quant] [--steps N] [--frames N] [--height N] [--width N]
//                  [--workdir DIR] [--ffmpeg PATH] [--dry-run]
//
// PROMPT EMBEDDINGS are taken as a file rather than computed here, deliberately:
// the encoder tower needs a tokenizer + a 32B forward, which is its own driver.
// This keeps the assembly question ("do the checkpoints compose into a video?")
// separable from the encoding question.
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/model_executor/models/qwen3_vl_text.h"  // Qwen3VLGetRopeIndex + MmImageSpan
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

int RunFfmpeg(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  const pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("fork failed");
  if (pid == 0) {
    execvp(argv[0], argv.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) throw std::runtime_error("waitpid failed");
  if (WIFSIGNALED(status)) {
    throw std::runtime_error("ffmpeg died on signal " + std::to_string(WTERMSIG(status)));
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  nlohmann::json j;
  in >> j;
  return j;
}

// A binary PPM (P6) reader, so a reference frame can be handed in with no image
// dependency -- the same format this example already WRITES for its own output,
// so a frame from one run can condition the next. Returns [3, H, W] in [-1, 1],
// which is the range the VAE encoder expects.
std::vector<float> ReadPpmAsChw(const std::string& path, int64_t* out_h, int64_t* out_w) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string magic;
  in >> magic;
  if (magic != "P6") throw std::runtime_error(path + ": not a binary PPM (P6)");
  auto next_int = [&]() {
    int v = 0;
    while (in >> std::ws, in.peek() == '#') { std::string skip; std::getline(in, skip); }
    in >> v;
    return v;
  };
  const int w = next_int(), h = next_int(), maxv = next_int();
  if (w <= 0 || h <= 0 || maxv <= 0) throw std::runtime_error(path + ": bad PPM header");
  in.get();  // the single whitespace byte before the payload
  std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
  in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  if (!in) throw std::runtime_error(path + ": truncated PPM payload");
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

// A binary PPM (P6) reader that returns HWC uint8 [0,255] -- the layout the shared
// Qwen3-VL image processor expects (it does its own rescale + 0.5/0.5 normalize).
std::vector<uint8_t> ReadPpmAsHwcU8(const std::string& path, int64_t* out_h, int64_t* out_w) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string magic;
  in >> magic;
  if (magic != "P6") throw std::runtime_error(path + ": not a binary PPM (P6)");
  auto next_int = [&]() {
    int v = 0;
    while (in >> std::ws, in.peek() == '#') { std::string skip; std::getline(in, skip); }
    in >> v;
    return v;
  };
  const int w = next_int(), h = next_int(), maxv = next_int();
  if (w <= 0 || h <= 0 || maxv <= 0) throw std::runtime_error(path + ": bad PPM header");
  in.get();  // the single whitespace byte before the payload
  std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
  in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  if (!in) throw std::runtime_error(path + ": truncated PPM payload");
  if (out_h != nullptr) *out_h = h;
  if (out_w != nullptr) *out_w = w;
  return rgb;
}

// The WAV reader lives in the LIBRARY (MiniMaxH3ReadWav), next to the writer and
// unit-gated with it; this only opens the file. Returns CHANNEL-MAJOR samples in
// [-1, 1], mono repeated up to kMiniMaxH3AudioChannels, and REFUSES a sample rate
// the audio VAE was not trained at rather than silently mis-encoding it.
std::vector<float> ReadWavRef(const std::string& path, int64_t* out_channels,
                              int64_t* out_samples_per_channel) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (out_channels != nullptr) *out_channels = vllm::kMiniMaxH3AudioChannels;
  return vllm::MiniMaxH3ReadWav(bytes, vllm::kMiniMaxH3AudioChannels,
                                vllm::kMiniMaxH3AudioSampleRate, out_samples_per_channel);
}

std::vector<float> ReadF32(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const std::streamsize bytes = in.tellg();
  if (bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
    throw std::runtime_error(path + ": size is not a whole number of f32 values");
  }
  in.seekg(0);
  std::vector<float> out(static_cast<size_t>(bytes) / sizeof(float));
  in.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

void WriteFile(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string Need(int argc, char** argv, int i, const std::string& flag) {
  if (i >= argc) throw std::runtime_error("missing value for " + flag);
  return argv[i];
}

}  // namespace

int main(int argc, char** argv) {
  std::string dit_path, video_vae_path, video_cfg_path, audio_vae_path, audio_cfg_path;
  std::string embeds_path, out_path, workdir = "/tmp/minimax_h3_gen", ffmpeg = "ffmpeg";
  bool keep_quant = false, dry_run = false, dequant_bf16 = false, denoise_only = false;
  bool dump_params = false, fp4_resident = false;
  std::string device_name = "cpu";
  std::string encoder_path, prompt, tokenizer_path, save_embeds_path;
  std::string first_frame_path, last_frame_path;
  std::string decode_latent_path;  // diagnostic: decode a dumped VAE-input latent
  std::string roundtrip_path;      // diagnostic: encode->decode a real image
  std::string prompt_image_path;   // diagnostic: run an image through the vision tower
  std::string cond_image_path;     // route an image through the ENCODER vision path
                                   // (merged scatter into prompt_embeds + DeepStack)
  std::vector<std::string> ref_image_paths;
  std::string ref_video_prefix, ref_audio_path;
  // The served checkpoint PARTITION. Community GGUF/NVFP4 files strip the release
  // model_index.json `_minimax_h3` block, and the FL2VA/Ref2VA DiTs are structurally
  // identical, so it cannot be inferred from the weights — it must be DECLARED. Empty
  // => the guard refuses a full render and tells the user to pass it (the #77 catch).
  std::string partition_flag;
  double imgvid_noise_aug = 1.0;
  int64_t encoder_max_layers = 0;
  int64_t steps = 0, frames = 0, height = 0, width = 0;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string f = argv[i];
      if (f == "--dit") dit_path = Need(argc, argv, ++i, f);
      else if (f == "--video-vae") video_vae_path = Need(argc, argv, ++i, f);
      else if (f == "--video-vae-config") video_cfg_path = Need(argc, argv, ++i, f);
      else if (f == "--audio-vae") audio_vae_path = Need(argc, argv, ++i, f);
      else if (f == "--audio-vae-config") audio_cfg_path = Need(argc, argv, ++i, f);
      else if (f == "--prompt-embeds") embeds_path = Need(argc, argv, ++i, f);
      else if (f == "--out") out_path = Need(argc, argv, ++i, f);
      else if (f == "--workdir") workdir = Need(argc, argv, ++i, f);
      else if (f == "--ffmpeg") ffmpeg = Need(argc, argv, ++i, f);
      else if (f == "--keep-quant") keep_quant = true;
      else if (f == "--dequant-bf16") dequant_bf16 = true;
      else if (f == "--fp4-resident") fp4_resident = true;
      else if (f == "--dry-run") dry_run = true;
      else if (f == "--denoise-only") denoise_only = true;
      else if (f == "--dump-params") dump_params = true;
      else if (f == "--decode-latent") decode_latent_path = Need(argc, argv, ++i, f);
      else if (f == "--roundtrip") roundtrip_path = Need(argc, argv, ++i, f);
      else if (f == "--prompt-image") prompt_image_path = Need(argc, argv, ++i, f);
      else if (f == "--cond-image") cond_image_path = Need(argc, argv, ++i, f);
      else if (f == "--device") device_name = Need(argc, argv, ++i, f);
      else if (f == "--encoder") encoder_path = Need(argc, argv, ++i, f);
      else if (f == "--prompt") prompt = Need(argc, argv, ++i, f);
      else if (f == "--tokenizer") tokenizer_path = Need(argc, argv, ++i, f);
      else if (f == "--save-embeds") save_embeds_path = Need(argc, argv, ++i, f);
      else if (f == "--first-frame") first_frame_path = Need(argc, argv, ++i, f);
      else if (f == "--last-frame") last_frame_path = Need(argc, argv, ++i, f);
      else if (f == "--ref-image") ref_image_paths.push_back(Need(argc, argv, ++i, f));
      else if (f == "--ref-video") ref_video_prefix = Need(argc, argv, ++i, f);
      else if (f == "--ref-audio") ref_audio_path = Need(argc, argv, ++i, f);
      else if (f == "--partition") partition_flag = Need(argc, argv, ++i, f);
      else if (f == "--noise-aug") imgvid_noise_aug = std::stod(Need(argc, argv, ++i, f));
      else if (f == "--encoder-max-layers") encoder_max_layers = std::stoll(Need(argc, argv, ++i, f));
      else if (f == "--steps") steps = std::stoll(Need(argc, argv, ++i, f));
      else if (f == "--frames") frames = std::stoll(Need(argc, argv, ++i, f));
      else if (f == "--height") height = std::stoll(Need(argc, argv, ++i, f));
      else if (f == "--width") width = std::stoll(Need(argc, argv, ++i, f));
      else throw std::runtime_error("unknown argument: " + f);
    }
    // --denoise-only stops after the DiT step loop, so it needs neither VAE nor an
    // output path -- and, just as importantly, does not spend their memory. On a
    // unified-memory box that headroom is the difference between a run and a reboot.
    // --dump-params reads the manifest and exits, so it needs NOTHING but --dit:
    // no VAEs, no conditioning, no output path. Requiring them would make the one
    // tool that works on a checkpoint too large to load unusable on exactly that
    // checkpoint.
    // --prompt-image runs the vision tower ONLY (from --encoder); it needs no DiT/VAE/out.
    const bool vision_probe = !prompt_image_path.empty();
    const bool diag_vae_only = !decode_latent_path.empty() || !roundtrip_path.empty();
    const bool need_vaes = !denoise_only && !dump_params && !diag_vae_only;
    const bool need_cond = !dump_params && !diag_vae_only;
    // --decode-latent / --roundtrip / --prompt-image need NO DiT and NO conditioning
    // (their own blocks validate their inputs); the shared check below would otherwise
    // reject --dit.
    if (!diag_vae_only && !vision_probe &&
        (dit_path.empty() || (need_vaes && (video_vae_path.empty() || audio_vae_path.empty())) ||
         (need_vaes && out_path.empty()) ||
         (need_cond && embeds_path.empty() && (encoder_path.empty() || prompt.empty())))) {
      std::cerr << "usage: minimax-h3-gen --dit <f|shard-dir> --video-vae <f> --audio-vae <f> "
                   "--prompt-embeds <f32.bin> --out <out.mp4> [--video-vae-config <j>] "
                   "[--audio-vae-config <j>] [--keep-quant] [--steps N] [--frames N] "
                   "[--height N] [--width N] [--device cpu|cuda] [--workdir DIR] [--ffmpeg PATH] "
                   "[--dry-run] [--denoise-only] [--dump-params] "
                   "[--first-frame f.ppm] [--last-frame f.ppm] [--noise-aug A] "
                   "[--ref-image f.ppm ...] [--ref-video DIR] [--ref-audio f.wav] "
                   "[--partition fl2va|ref2va]\n";
      return 2;
    }

    // --dump-params reads the MANIFEST ONLY -- names and shapes, no payload -- and
    // prints the geometry those shapes imply. That makes it safe on a checkpoint
    // whose weights do not fit (the NVFP4 reference loader is ~132 GB of host f32),
    // and it is the right tool for asking "do two checkpoints agree on geometry?"
    // without running either.
    if (dump_params) {
      vllm::MiniMaxH3DitParams pr;
      if (EndsWith(dit_path, ".gguf")) {
        const vllm::GgufFile gf = vllm::GgufFile::Open(dit_path);
        pr = vllm::ParseMiniMaxH3DitParamsFromGgufManifest(vllm::EnumerateMiniMaxH3GgufTensors(gf));
      } else if (vllm::MiniMaxH3ShardedCheckpoint::IsShardedDir(dit_path)) {
        // A DIRECTORY of shards + index: the original bf16 release. Manifest only,
        // so this answers "do the 13 shards agree on the geometry the GGUF arm
        // derives?" on a 66.3 GB checkpoint without reading a single weight byte.
        const vllm::MiniMaxH3ShardedCheckpoint ckpt =
            vllm::MiniMaxH3ShardedCheckpoint::Open(dit_path);
        std::cerr << "  " << ckpt.ShardCount() << " shard(s), " << ckpt.Names().size()
                  << " tensors, index " << ckpt.IndexPath() << "\n";
        pr = vllm::ParseMiniMaxH3DitParamsFromGgufManifest(
            vllm::EnumerateMiniMaxH3ShardedTensors(ckpt));
      } else {
        const vllm::SafetensorsFile sf = vllm::SafetensorsFile::Open(dit_path);
        std::vector<vllm::MiniMaxH3TensorSpec> manifest;
        for (const std::string& name : sf.Names()) {
          if ((name.size() > 12 && name.compare(name.size() - 12, 12, "weight_scale") == 0) ||
              (name.size() > 14 && name.compare(name.size() - 14, 14, "weight_scale_2") == 0)) {
            continue;
          }
          const vllm::StTensor& st = sf.Get(name);
          vllm::MiniMaxH3TensorSpec spec;
          spec.name = name;
          spec.shape = st.shape;
          if (st.dtype == "U8") spec.shape = {st.shape[0], st.shape[1] * 2};
          manifest.push_back(std::move(spec));
        }
        pr = vllm::ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
      }
      std::cout << "num_layers=" << pr.num_layers
                << "\ntoken_refiner_num_layers=" << pr.token_refiner_num_layers
                << "\nhidden_size=" << pr.hidden_size
                << "\nnum_attention_heads=" << pr.num_attention_heads
                << "\nattention_head_dim=" << pr.attention_head_dim
                << "\nffn_hidden_size=" << pr.ffn_hidden_size
                << "\nlatents_dim=" << pr.latents_dim
                << "\naudio_latents_dim=" << pr.audio_latents_dim
                << "\npatch_size_t=" << pr.patch_size_t
                << "\npatch_size_h=" << pr.patch_size_h
                << "\npatch_size_w=" << pr.patch_size_w
                << "\ntext_dim=" << pr.text_dim
                << "\ntimestep_input_dim=" << pr.timestep_input_dim
                << "\ntime_embed_hidden_size=" << pr.time_embed_hidden_size
                << "\ntime_embed_dim=" << pr.time_embed_dim
                << "\nadaln_out_features=" << pr.adaln_out_features
                << "\nfinal_adaln_out_features=" << pr.final_adaln_out_features
                << "\nrope_inv_freq_len=" << pr.rope_inv_freq_len
                << "\nvideo_row_width=" << pr.video_row_width()
                << "\nrope_rot_dim=" << pr.rope_rot_dim() << "\n";
      return 0;
    }

    // --decode-latent DIAGNOSTIC: decode a dumped VAE-input latent
    // (VT_H3_DUMP_DIR/vae_input_video_latent.f32) directly, with NO DiT and NO
    // conditioning, on either device. Lets the device ViT3D decoder be compared
    // against the scalar CPU reference (gated vs upstream at 8.9e-8) on the SAME
    // real latent -- the VAE-branch oracle test for the render-coherence bisection.
    if (!decode_latent_path.empty()) {
      if (video_cfg_path.empty() || video_vae_path.empty() || out_path.empty()) {
        throw std::runtime_error(
            "--decode-latent needs --video-vae, --video-vae-config, --out and --width/--height/--frames");
      }
      vllm::MiniMaxH3LatentStats vstats;
      vllm::MiniMaxH3VideoVaeDecoderConfig vcfg =
          vllm::ParseMiniMaxH3VideoVaeDecoderConfig(ReadJson(video_cfg_path), &vstats);
      vllm::SafetensorsFile vfile = vllm::SafetensorsFile::Open(video_vae_path);
      vllm::MiniMaxH3AudioVaeWeights vweights = vllm::LoadMiniMaxH3VideoVaeDecoderWeights(vfile);
      const vllm::MiniMaxH3ShapePlan plan = vllm::MiniMaxH3ResolveShape(
          "t2va", 0.0, frames, height, width, 0, 0);
      const int64_t lt = plan.latent_t, lh = plan.height / vllm::kMiniMaxH3VaeRatio,
                    lw = plan.width / vllm::kMiniMaxH3VaeRatio, ch = vcfg.in_channels;
      const int64_t need = ch * lt * lh * lw;
      std::ifstream lf(decode_latent_path, std::ios::binary);
      if (!lf) throw std::runtime_error("cannot open --decode-latent file");
      std::vector<float> latent(static_cast<size_t>(need));
      lf.read(reinterpret_cast<char*>(latent.data()), need * static_cast<int64_t>(sizeof(float)));
      if (!lf) throw std::runtime_error("--decode-latent file too small for [C,T,H,W]");
      std::cerr << "decode-latent: [" << ch << "," << lt << "," << lh << "," << lw << "] on "
                << device_name << "\n";
      vllm::MiniMaxH3T2vaResult result;
      if (device_name == "cuda") {
        vt::Device dev = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue().device;
        vt::Queue vq = vt::GetBackend(dev.type).CreateQueue();
        const vllm::MiniMaxH3VideoVaeDeviceWeights staged =
            vllm::StageMiniMaxH3VideoVaeWeights(vq, vcfg, vweights);
        result.frames = vllm::MiniMaxH3VideoVaeDecodeTemporalDevice(
            dev, vcfg, staged, latent, lt, lh, lw, plan.num_frames, &result.frame_shape);
      } else {
        result.frames = vllm::MiniMaxH3VideoVaeDecode(vcfg, vweights, latent, lt, lh, lw,
                                                      &result.frame_shape);
      }
      vllm::MiniMaxH3VideoDenormalizePixels(
          result.frames, result.frame_shape.channels,
          result.frame_shape.t * result.frame_shape.h * result.frame_shape.w);
      std::string mkc = "mkdir -p '" + workdir + "'";
      if (std::system(mkc.c_str()) != 0) throw std::runtime_error("cannot create " + workdir);
      for (int64_t fr = 0; fr < result.frame_shape.t; ++fr) {
        char nm[512];
        std::snprintf(nm, sizeof(nm), "%s/frame_%06lld.ppm", workdir.c_str(),
                      static_cast<long long>(fr));
        WriteFile(nm, vllm::MiniMaxH3WritePpmFrame(result.frames, result.frame_shape, fr));
      }
      std::cerr << "decode-latent: wrote " << result.frame_shape.t << " frames to " << workdir
                << "\n";
      return 0;
    }

    // --roundtrip DIAGNOSTIC: encode a real image through the video VAE encoder,
    // apply post_quant_conv, and decode -- a DiT-independent gold-standard test of
    // the decoder. A coherent round-trip proves the decoder works and localizes the
    // render bug to the DiT-produced latent; a grid proves the decoder itself.
    if (!roundtrip_path.empty()) {
      if (video_cfg_path.empty() || video_vae_path.empty()) {
        throw std::runtime_error("--roundtrip needs --video-vae and --video-vae-config");
      }
      vllm::MiniMaxH3LatentStats vstats;
      vllm::MiniMaxH3VideoVaeDecoderConfig vcfg =
          vllm::ParseMiniMaxH3VideoVaeDecoderConfig(ReadJson(video_cfg_path), &vstats);
      vllm::SafetensorsFile vfile = vllm::SafetensorsFile::Open(video_vae_path);
      vllm::MiniMaxH3AudioVaeWeights dec_w = vllm::LoadMiniMaxH3VideoVaeDecoderWeights(vfile);
      vllm::MiniMaxH3AudioVaeWeights enc_w = vllm::LoadMiniMaxH3VideoVaeEncoderWeights(vfile);
      int64_t ih = 0, iw = 0;
      std::vector<float> chw = ReadPpmAsChw(roundtrip_path, &ih, &iw);  // [3,H,W] in [0,1]
      vllm::MiniMaxH3VideoNormalizePixels(chw, 3, ih * iw);             // -> imagenet space
      vllm::MiniMaxH3EncoderFcn3dConfig enc_cfg;
      enc_cfg.z_channels = 2 * vcfg.in_channels;  // moments (mean|logvar)
      enc_cfg.t = 1; enc_cfg.h = ih; enc_cfg.w = iw;
      vllm::MiniMaxH3VideoFrameShape ls{};
      std::vector<float> z = vllm::MiniMaxH3VideoVaeEncodeToLatent(enc_cfg, enc_w, chw, &ls);
      std::cerr << "roundtrip: encoded [" << vcfg.in_channels << "," << ls.t << "," << ls.h << ","
                << ls.w << "]\n";
      const int64_t per = ls.t * ls.h * ls.w;
      // per-channel stats of the ENCODED latent (the in-distribution reference)
      { double s2 = 0; for (float v : z) s2 += double(v) * v;
        std::cerr << "roundtrip: encoded-latent rms=" << std::sqrt(s2 / z.size()) << "\n"; }
      if (const char* dd = std::getenv("VT_H3_DUMP_DIR")) {
        std::string p = std::string(dd) + "/encoder_latent.f32";
        if (std::FILE* fp = std::fopen(p.c_str(), "wb")) {
          std::fwrite(z.data(), sizeof(float), z.size(), fp); std::fclose(fp);
          std::cerr << "roundtrip: dumped encoder latent [" << vcfg.in_channels << "," << ls.t
                    << "," << ls.h << "," << ls.w << "] to " << p << "\n";
        }
      }
      if (dec_w.Has("post_quant_conv.weight")) {
        z = vllm::MiniMaxH3VideoVaePostQuantConv(dec_w, z, vcfg.in_channels, per);
      }
      vllm::MiniMaxH3T2vaResult result;
      if (device_name == "cuda") {
        vt::Device dev = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue().device;
        vt::Queue vq = vt::GetBackend(dev.type).CreateQueue();
        const vllm::MiniMaxH3VideoVaeDeviceWeights staged =
            vllm::StageMiniMaxH3VideoVaeWeights(vq, vcfg, dec_w);
        result.frames = vllm::MiniMaxH3VideoVaeDecodeTemporalDevice(
            dev, vcfg, staged, z, ls.t, ls.h, ls.w, 1, &result.frame_shape);
      } else {
        result.frames = vllm::MiniMaxH3VideoVaeDecode(vcfg, dec_w, z, ls.t, ls.h, ls.w,
                                                      &result.frame_shape);
      }
      vllm::MiniMaxH3VideoDenormalizePixels(
          result.frames, result.frame_shape.channels,
          result.frame_shape.t * result.frame_shape.h * result.frame_shape.w);
      std::string mkc = "mkdir -p '" + workdir + "'";
      if (std::system(mkc.c_str()) != 0) throw std::runtime_error("cannot create " + workdir);
      WriteFile(workdir + "/roundtrip.ppm",
                vllm::MiniMaxH3WritePpmFrame(result.frames, result.frame_shape, 0));
      std::cerr << "roundtrip: wrote " << workdir << "/roundtrip.ppm ("
                << result.frame_shape.w << "x" << result.frame_shape.h << ")\n";
      return 0;
    }

    // --- --prompt-image DIAGNOSTIC: route a real image through the shared Qwen3-VL
    // image processor + the encoder's VISION tower, loaded from the encoder GGUF's
    // visual.* tensors (the piece the record reconciliation found missing, spec §8.8).
    // Reports the conditioning feature stats -- the REAL-WEIGHTS proof that the vision
    // tower now runs. Scattering the merged/deepstack features into the DiT-conditioning
    // path (merge into prompt_embeds + DeepStack inject into the device text tower) is
    // the tracked residual; this probe stops after the tower.
    if (!prompt_image_path.empty()) {
      if (encoder_path.empty())
        throw std::runtime_error("--prompt-image needs --encoder (the vision weights live in it)");
      if (device_name != "cuda")
        throw std::runtime_error(
            "--prompt-image needs --device cuda (the vision tower is device-resident)");
      std::cerr << "loading encoder vision tower from " << encoder_path << "\n";
      const vllm::GgufFile ef = vllm::GgufFile::Open(encoder_path);
      const vllm::multimodal::Qwen3VLVisionConfig vcfg = vllm::MiniMaxH3EncoderVisionConfig();
      const vllm::multimodal::Qwen3VLVisionWeights vw = vllm::LoadQwen3VLVisionFromGguf(ef, vcfg);
      std::cerr << "  vision weights: depth=" << vcfg.depth << " hidden=" << vcfg.hidden_size
                << " heads=" << vcfg.num_heads << " out=" << vcfg.out_hidden_size
                << " deepstack=" << vw.deepstack_mergers.size() << "\n";

      int64_t ih = 0, iw = 0;
      const std::vector<uint8_t> rgb = ReadPpmAsHwcU8(prompt_image_path, &ih, &iw);
      std::cerr << "  image " << iw << "x" << ih << "\n";
      vllm::multimodal::Qwen3VLProcessorConfig pcfg;  // patch16/temporal2/merge2/0.5-norm
      pcfg.merge_size = static_cast<int>(vcfg.spatial_merge_size);
      const vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
      const vllm::multimodal::ImageKwargs kw = proc.ProcessImage(rgb.data(), ih, iw);
      const std::array<int64_t, 3> grid = kw.image_grid_thw;
      const int64_t tokens = grid[0] * grid[1] * grid[2];
      const int64_t merge = vcfg.spatial_merge_size * vcfg.spatial_merge_size;
      std::cerr << "  grid_thw=[" << grid[0] << "," << grid[1] << "," << grid[2]
                << "] tokens=" << tokens << " merged=" << (tokens / merge) << "\n";

      vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCUDA);
      vllm::multimodal::Qwen3VLVisionCapture cap;
      const std::vector<float> tower = vllm::multimodal::Qwen3VLVisionForward(
          kw.pixel_values_bf16, grid, vw, vcfg, backend, &cap);
      const int64_t nm = tokens / merge;
      const int64_t width =
          vcfg.out_hidden_size *
          (1 + static_cast<int64_t>(vcfg.deepstack_visual_indexes.size()));
      auto stats = [&](const std::vector<float>& v, const char* tag) {
        double s = 0, s2 = 0, mx = 0;
        bool fin = true;
        for (float f : v) {
          if (!std::isfinite(f)) fin = false;
          s += f;
          s2 += double(f) * f;
          mx = std::max(mx, std::fabs(static_cast<double>(f)));
        }
        const double n = v.empty() ? 1.0 : static_cast<double>(v.size());
        std::cerr << "  " << tag << ": n=" << v.size() << " finite=" << (fin ? "yes" : "NO")
                  << " mean=" << (s / n) << " rms=" << std::sqrt(s2 / n) << " maxabs=" << mx
                  << "\n";
      };
      std::cerr << "  tower out = [" << nm << ", " << width << "]\n";
      stats(tower, "tower_concat");
      stats(cap.merger_out, "merged");
      for (size_t d = 0; d < cap.deepstack_out.size(); ++d)
        stats(cap.deepstack_out[d], ("deepstack_" + std::to_string(d)).c_str());
      std::cout << "prompt-image: vision tower RAN on real weights; merged=[" << nm << ","
                << vcfg.out_hidden_size << "] + " << cap.deepstack_out.size() << " deepstack\n";
      return 0;
    }

    // --- 1. DiT ---
    std::cerr << "loading DiT " << dit_path << (keep_quant ? " (keep-quant)" : "") << "\n";
    vllm::MiniMaxH3GgufDit dit;
    vllm::MiniMaxH3DitDeviceWeights streamed;
    bool have_streamed = false;
    if (EndsWith(dit_path, ".gguf")) {
      const vllm::GgufFile f = vllm::GgufFile::Open(dit_path);
      // --dequant-bf16 loads STRAIGHT to bf16 (~33 GB). Keeping blocks would leave
      // the AdaLN projections ineligible (K=2688 is not a whole number of 256-element
      // Q3_K blocks) and dequantize them to ~52 GB of f32 — which is what does not fit.
      if (dequant_bf16 && device_name == "cuda") {
        // STREAM straight onto the device: dequantize + upload one tensor at a time
        // so the host copy never accumulates. Peak is what kills a unified-memory
        // box, and load-then-stage holds ~33 GB twice.
        vt::Queue sq = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue();
        const auto t0 = std::chrono::steady_clock::now();
        streamed = vllm::StreamMiniMaxH3DitToDeviceBf16(sq, f, &dit.params);
        have_streamed = true;
        std::cerr << "  streamed DiT -> device (bf16) in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " s\n";
      } else {
        dit = dequant_bf16 ? vllm::LoadMiniMaxH3DitFromGgufBf16(f)
                           : vllm::LoadMiniMaxH3DitFromGguf(f, keep_quant);
      }
    } else if (vllm::MiniMaxH3ShardedCheckpoint::IsShardedDir(dit_path)) {
      // The ORIGINAL bf16 release: a DIRECTORY of shards plus
      // model.safetensors.index.json. Every other --dit form is a single file and
      // keeps working unchanged; this is the only one that can open the
      // full-precision DiT at all, which is what makes the quantization-quality
      // question askable.
      const vllm::MiniMaxH3ShardedCheckpoint ckpt =
          vllm::MiniMaxH3ShardedCheckpoint::Open(dit_path);
      std::cerr << "  " << ckpt.ShardCount() << " shard(s), " << ckpt.Names().size()
                << " tensors (index " << ckpt.IndexPath() << ")\n";
      // Host f32 reference path. ~132 GB on the real release, so it is usable
      // only on a reduced checkpoint; opening the real release on a device needs
      // a streaming loader, which this row does not yet ship.
      dit = vllm::LoadMiniMaxH3DitFromShards(ckpt);
    } else {
      const vllm::SafetensorsFile f = vllm::SafetensorsFile::Open(dit_path);
      if (device_name == "cuda") {
        // Stream NVFP4 straight to the device. The host-f32 reference loader is
        // ~132 GB for this checkpoint and gets OOM-killed during load on a
        // unified-memory box; streaming keeps peak at the device copy plus one
        // tensor.
        vt::Queue sq = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue();
        const auto t0 = std::chrono::steady_clock::now();
        // --fp4-resident keeps the packed FP4 on device (~1/4 the bf16 footprint,
        // ~16 GB vs ~66 GB) and routes every quantized projection through the
        // Marlin W4A16 GEMM on sm_121a; the default bf16 stream dequantizes to bf16.
        streamed = fp4_resident
                       ? vllm::StreamMiniMaxH3Nvfp4ToDeviceFp4(sq, f, &dit.params)
                       : vllm::StreamMiniMaxH3Nvfp4ToDeviceBf16(sq, f, &dit.params);
        have_streamed = true;
        std::cerr << "  streamed NVFP4 DiT -> device (" << (fp4_resident ? "fp4-resident" : "bf16")
                  << ") in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " s\n";
      } else {
        dit = vllm::LoadMiniMaxH3DitFromNvfp4(f);
      }
    }
    std::cerr << "  layers=" << dit.params.num_layers << " hidden=" << dit.params.hidden_size
              << " heads=" << dit.params.num_attention_heads << "\n";

    // --- 1b. optional encoder probe. Loading the 32B tower keep-quant is the
    // precondition for real text conditioning; this reports the geometry it
    // recovered so the loader can be validated against the REAL file. ---
    std::vector<float> encoded_prompt;
    if (!encoder_path.empty()) {
      std::cerr << "loading encoder " << encoder_path << " (keep-quant)\n";
      const vllm::GgufFile ef = vllm::GgufFile::Open(encoder_path);
      const vllm::MiniMaxH3EncoderQuantWeights enc =
          vllm::LoadMiniMaxH3EncoderFromGguf(ef, encoder_max_layers);
      vllm::MiniMaxH3EncoderConfig ec = enc.config;
      std::cerr << "  encoder layers=" << ec.num_hidden_layers << " hidden=" << ec.hidden_size
                << " heads=" << ec.num_attention_heads << " kv_heads=" << ec.num_key_value_heads
                << " head_dim=" << ec.head_dim << " ffn=" << ec.intermediate_size << "\n";
      size_t quant_bytes = 0;
      for (const auto& kv : enc.quant_storage) quant_bytes += kv.second.size();
      std::cerr << "  encoder resident (keep-quant) = " << (quant_bytes / (1024.0 * 1024.0 * 1024.0))
                << " GiB\n";

      if (!prompt.empty()) {
        // The ComfyUI-style encoder GGUF is WEIGHTS ONLY — it carries no
        // `tokenizer.ggml.*` metadata, unlike a llama.cpp export — so the vocab
        // comes from the checkpoint's own tokenizer.json. `--tokenizer` is
        // therefore required with `--prompt` unless the GGUF happens to embed one.
        const vllm::tok::Tokenizer tokenizer =
            tokenizer_path.empty() ? vllm::tok::Tokenizer::FromGguf(ef)
                                   : vllm::tok::Tokenizer::FromHfJson(tokenizer_path);
        std::vector<int32_t> ids = tokenizer.Encode(prompt);
        VT_CHECK(!ids.empty(), "minimax-h3-gen: the prompt tokenized to nothing");
        std::cerr << "  prompt tokens = " << ids.size() << "\n";

        // --cond-image routes a reference image through the ENCODER VISION PATH
        // (upstream _encode, encoder.py:1064-1101): the vision tower's MERGED features
        // masked_scatter into inputs_embeds at the image-pad positions, and its 3
        // DeepStack blocks inject into the first N text layers. This is the residual
        // #86 left open — the vision features now REACH the DiT via prompt_embeds.
        std::vector<uint8_t> visual_mask;             // 1 at each image-pad row
        std::vector<std::vector<float>> deepstack;    // 3 x [nm, hidden] taps
        std::vector<float> merged;                    // [nm, hidden] merged features
        std::array<int64_t, 3> vgrid{1, 0, 0};
        int64_t vsmerge = 2;
        const bool have_vision = !cond_image_path.empty();
        if (have_vision) {
          VT_CHECK(device_name == "cuda",
                   "minimax-h3-gen: --cond-image needs --device cuda (the vision tower is "
                   "device-resident)");
          const vllm::multimodal::Qwen3VLVisionConfig vcfg = vllm::MiniMaxH3EncoderVisionConfig();
          const vllm::multimodal::Qwen3VLVisionWeights vw = vllm::LoadQwen3VLVisionFromGguf(ef, vcfg);
          vsmerge = vcfg.spatial_merge_size;
          int64_t cih = 0, ciw = 0;
          const std::vector<uint8_t> rgb = ReadPpmAsHwcU8(cond_image_path, &cih, &ciw);
          std::cerr << "  cond-image " << ciw << "x" << cih << "\n";
          vllm::multimodal::Qwen3VLProcessorConfig pcfg;  // patch16/temporal2/merge2/0.5-norm
          pcfg.merge_size = static_cast<int>(vsmerge);
          const vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
          const vllm::multimodal::ImageKwargs kw = proc.ProcessImage(rgb.data(), cih, ciw);
          vgrid = kw.image_grid_thw;
          vt::Backend& vbackend = vt::GetBackend(vt::DeviceType::kCUDA);
          vllm::multimodal::Qwen3VLVisionCapture cap;
          (void)vllm::multimodal::Qwen3VLVisionForward(kw.pixel_values_bf16, vgrid, vw, vcfg,
                                                       vbackend, &cap);
          merged = std::move(cap.merger_out);
          deepstack = std::move(cap.deepstack_out);
          const int64_t nm = vgrid[0] * vgrid[1] * vgrid[2] / (vsmerge * vsmerge);
          std::cerr << "  cond-image: grid=[" << vgrid[0] << "," << vgrid[1] << "," << vgrid[2]
                    << "] merged=" << nm << " deepstack=" << deepstack.size() << "\n";
          // Prepend a vision block <vision_start><image_pad><vision_end>, then expand
          // the single image_pad into nm image_token_id copies (the mm processor).
          std::vector<int32_t> pre = {pcfg.vision_start_token_id, pcfg.image_token_id,
                                      pcfg.vision_end_token_id};
          pre.insert(pre.end(), ids.begin(), ids.end());
          std::vector<std::array<int, 2>> placeholders;
          ids = vllm::multimodal::ExpandImagePlaceholders(
              pre, pcfg.image_token_id, static_cast<int>(vsmerge),
              {{vgrid[0], vgrid[1], vgrid[2]}}, &placeholders);
          VT_CHECK(!placeholders.empty(),
                   "minimax-h3-gen: cond-image placeholder expansion produced none");
        }

        // DIAGNOSTIC (env-gated): dump the (possibly expanded) token ids so the
        // tokenization can be diffed against upstream. A template mismatch shifts
        // every text row and feeds the 32B tower a different string.
        if (const char* dd = std::getenv("VT_H3_DUMP_INPUTS")) {
          if (std::FILE* fp = std::fopen((std::string(dd) + "/prompt_token_ids.i32").c_str(), "wb")) {
            std::fwrite(ids.data(), sizeof(int32_t), ids.size(), fp);
            std::fclose(fp);
            std::cerr << "  [h3-dump-inputs] prompt_token_ids.i32 (" << ids.size() << " ids): ";
            for (size_t k = 0; k < ids.size() && k < 64; ++k) std::cerr << ids[k] << " ";
            std::cerr << "\n";
          }
        }

        const int64_t seq = static_cast<int64_t>(ids.size());
        std::vector<float> embeds = vllm::MiniMaxH3EncoderEmbedTokens(enc, ids);
        std::vector<int64_t> pos(static_cast<size_t>(3 * seq));
        if (!have_vision) {
          // Text-only: all three M-RoPE axes are the token index.
          for (int64_t a = 0; a < 3; ++a) {
            for (int64_t s = 0; s < seq; ++s) pos[static_cast<size_t>(a * seq + s)] = s;
          }
        } else {
          // masked_scatter the merged features into the image-pad rows + build the
          // visual position mask (upstream `inputs_embeds.masked_scatter(image_mask,
          // image_embeds)` + `_get_placeholder_mask`).
          const int32_t IMG = vllm::multimodal::Qwen3VLProcessorConfig{}.image_token_id;
          visual_mask.assign(static_cast<size_t>(seq), 0);
          const int64_t hidden = ec.hidden_size;
          int64_t vi = 0, off = -1;
          for (int64_t s = 0; s < seq; ++s) {
            if (ids[static_cast<size_t>(s)] != IMG) continue;
            if (off < 0) off = s;
            visual_mask[static_cast<size_t>(s)] = 1;
            VT_CHECK((vi + 1) * hidden <= static_cast<int64_t>(merged.size()),
                     "minimax-h3-gen: merged vision rows < image-pad tokens");
            for (int64_t i = 0; i < hidden; ++i)
              embeds[static_cast<size_t>(s * hidden + i)] =
                  merged[static_cast<size_t>(vi * hidden + i)];
            ++vi;
          }
          VT_CHECK(vi * hidden == static_cast<int64_t>(merged.size()),
                   "minimax-h3-gen: merged vision rows != image-pad tokens");
          // M-RoPE positions. Qwen3VLGetRopeIndex is byte-equivalent to H3's own
          // _get_rope_index for a single-frame image (t==1): text runs sequentially,
          // the image block takes the 3D vision grid, and the next text advances by
          // max(llm_h, llm_w). Layout is [3, seq] flattened as [axis*seq + s].
          vllm::multimodal::MmImageSpan span{off, {vgrid[0], vgrid[1], vgrid[2]}};
          int64_t delta = 0;
          const std::vector<int32_t> p3 =
              vllm::multimodal::Qwen3VLGetRopeIndex(ids, {span}, vsmerge, &delta);
          VT_CHECK(static_cast<int64_t>(p3.size()) == 3 * seq,
                   "minimax-h3-gen: rope index produced the wrong count");
          for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int64_t>(p3[i]);
        }
        vt::Device enc_dev{};
        if (device_name == "cuda") {
          enc_dev = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue().device;
        }
        vt::Queue eq{enc_dev, nullptr};
        vt::Backend& eb = vt::GetBackend(enc_dev.type);
        if (enc_dev.type != vt::DeviceType::kCPU) eq = eb.CreateQueue();
        const vllm::MiniMaxH3EncoderDeviceWeights staged =
            vllm::StageMiniMaxH3EncoderWeights(eq, enc);
        std::cerr << "  encoding prompt" << (have_vision ? " (vision-conditioned)" : "") << "...\n";
        encoded_prompt = vllm::MiniMaxH3EncoderTextForwardDevice(
            eq, ec, staged, embeds, pos.data(), seq,
            have_vision ? visual_mask.data() : nullptr, deepstack);
        std::cerr << "  conditioning = [" << seq << ", " << ec.hidden_size << "]\n";
        // Persisting the conditioning makes a checkpoint A/B CONTROLLED: two DiTs
        // can then be compared on byte-identical text conditioning instead of two
        // separate encoder runs. It also lets the second run skip the 13 GB tower
        // entirely, which is the difference between fitting and not on a
        // unified-memory box.
        if (!save_embeds_path.empty()) {
          std::ofstream out(save_embeds_path, std::ios::binary);
          if (!out) throw std::runtime_error("cannot write " + save_embeds_path);
          out.write(reinterpret_cast<const char*>(encoded_prompt.data()),
                    static_cast<std::streamsize>(encoded_prompt.size() * sizeof(float)));
          std::cerr << "  saved conditioning -> " << save_embeds_path << "\n";
        }
      }
    }

    // --- 2. VAEs + their configs (the configs carry the latent statistics) ---
    vllm::MiniMaxH3VideoVaeDecoderConfig video_cfg;
    vllm::MiniMaxH3LatentStats video_stats;
    if (!video_cfg_path.empty()) {
      video_cfg = vllm::ParseMiniMaxH3VideoVaeDecoderConfig(ReadJson(video_cfg_path), &video_stats);
    }
    vllm::MiniMaxH3AudioVaeConfig audio_cfg;
    vllm::MiniMaxH3LatentStats audio_stats;
    if (!audio_cfg_path.empty()) {
      audio_cfg = vllm::ParseMiniMaxH3AudioVaeConfig(ReadJson(audio_cfg_path), &audio_stats);
    }
    // The files stay in scope alongside the weights: the loaders read through the
    // mapping, so releasing it early would leave the weights pointing at nothing.
    std::optional<vllm::SafetensorsFile> video_file, audio_file;
    vllm::MiniMaxH3AudioVaeWeights video_weights, audio_weights;
    if (need_vaes) {
      std::cerr << "loading video VAE " << video_vae_path << "\n";
      video_file = vllm::SafetensorsFile::Open(video_vae_path);
      video_weights = vllm::LoadMiniMaxH3VideoVaeDecoderWeights(*video_file);
      std::cerr << "loading audio VAE " << audio_vae_path << "\n";
      audio_file = vllm::SafetensorsFile::Open(audio_vae_path);
      audio_weights = vllm::LoadMiniMaxH3AudioVaeWeights(*audio_file);
    }

    // --- 3. request shape ---
    vllm::MiniMaxH3T2vaRequest request;
    request.text_len = 0;  // set from the embeddings below
    if (steps > 0) request.num_steps = steps;
    // `_resolve_shape` decides frames/canvas/latent_t/audio_t; the latent GRID is
    // that canvas divided by the VAE's spatial ratio (prod(space_down) = 16).
    const vllm::MiniMaxH3ShapePlan plan = vllm::MiniMaxH3ResolveShape(
        "t2va", /*duration_seconds=*/0.0, frames, height, width,
        /*image_width=*/0, /*image_height=*/0);
    request.latent_t = plan.latent_t;
    request.num_frames = plan.num_frames;
    request.latent_h = plan.height / vllm::kMiniMaxH3VaeRatio;
    request.latent_w = plan.width / vllm::kMiniMaxH3VaeRatio;
    request.audio_t = plan.audio_t;
    request.audio_channel = vllm::kMiniMaxH3AudioChannels;
    request.video_latents_mean = video_stats.mean;
    request.video_latents_std = video_stats.std_dev;
    request.audio_latents_mean = audio_stats.mean;
    request.audio_latents_std = audio_stats.std_dev;
    // Declare the served partition so MiniMaxH3GenerateT2va can refuse a task the
    // checkpoint does not serve (the #77 catch: t2va on the Ref2VA arm). An empty
    // --partition is declared-but-unknown; the guard then names the recipe lines and
    // asks for fl2va|ref2va rather than silently rendering the wrong combination.
    request.partition = vllm::MiniMaxH3PartitionFromFlag(partition_flag);

    // --- ref2va VIDEO reference: a CLIP prepended to the sequence. Reads
    // DIR/frame_%06d.ppm, which is exactly what this example WRITES, so a previous
    // run's workdir can be handed straight back in as a reference.
    if (!ref_video_prefix.empty()) {
      VT_CHECK(ref_image_paths.empty() && first_frame_path.empty() && last_frame_path.empty(),
               "minimax-h3-gen: --ref-video is exclusive with --ref-image and --first/--last-frame");
      VT_CHECK(!video_vae_path.empty(), "minimax-h3-gen: --ref-video needs --video-vae");
      const vllm::SafetensorsFile vf = vllm::SafetensorsFile::Open(video_vae_path);
      const vllm::MiniMaxH3AudioVaeWeights enc_w = vllm::LoadMiniMaxH3VideoVaeEncoderWeights(vf);
      vllm::MiniMaxH3EncoderFcn3dConfig enc_cfg;
      enc_cfg.z_channels = 2 * dit.params.latents_dim;

      std::vector<float> clip;
      int64_t ft = 0, fh = 0, fw = 0;
      for (int64_t k = 0;; ++k) {
        char nm[512];
        std::snprintf(nm, sizeof(nm), "%s/frame_%06lld.ppm", ref_video_prefix.c_str(),
                      static_cast<long long>(k));
        std::ifstream probe(nm, std::ios::binary);
        if (!probe) break;
        probe.close();
        int64_t h2 = 0, w2 = 0;
        const std::vector<float> f = ReadPpmAsChw(nm, &h2, &w2);
        if (ft == 0) { fh = h2; fw = w2; }
        VT_CHECK(h2 == fh && w2 == fw,
                 "minimax-h3-gen: every --ref-video frame must have the same size");
        // [C,H,W] per frame -> [C,T,H,W]: append per channel, so build channel-major.
        clip.insert(clip.end(), f.begin(), f.end());
        ++ft;
      }
      VT_CHECK(ft > 0, "minimax-h3-gen: --ref-video found no frame_%06d.ppm files");
      // Re-lay the per-frame [C,H,W] stack into [C,T,H,W].
      std::vector<float> chw(clip.size());
      const int64_t plane = fh * fw;
      for (int64_t c = 0; c < 3; ++c) {
        for (int64_t k = 0; k < ft; ++k) {
          for (int64_t e = 0; e < plane; ++e) {
            chw[static_cast<size_t>((c * ft + k) * plane + e)] =
                clip[static_cast<size_t>(k * 3 * plane + c * plane + e)];
          }
        }
      }
      vllm::MiniMaxH3RefBlock vb{};
      request.keyframe_cond_rows = vllm::MiniMaxH3EncodeReferenceVideo(
          enc_cfg, enc_w, dit.params, chw, ft, fh, fw, &vb);
      request.ref_blocks = {vb};
      std::cerr << "  ref2va: reference VIDEO " << ft << " frames at " << fw << "x" << fh
                << " (silent)\n";
    }

    // --- ref2va REFERENCES: whole reference images prepended to the sequence,
    // as opposed to fl2va which pins frames OF THE OUTPUT. Mutually exclusive.
    if (!ref_image_paths.empty()) {
      VT_CHECK(first_frame_path.empty() && last_frame_path.empty(),
               "minimax-h3-gen: --ref-image (ref2va) and --first/--last-frame (fl2va) are exclusive");
      VT_CHECK(!video_vae_path.empty(), "minimax-h3-gen: --ref-image needs --video-vae");
      const vllm::SafetensorsFile vf = vllm::SafetensorsFile::Open(video_vae_path);
      const vllm::MiniMaxH3AudioVaeWeights enc_w = vllm::LoadMiniMaxH3VideoVaeEncoderWeights(vf);
      vllm::MiniMaxH3EncoderFcn3dConfig enc_cfg;
      enc_cfg.z_channels = 2 * dit.params.latents_dim;

      std::vector<std::vector<float>> imgs;
      int64_t ih = 0, iw = 0;
      for (const std::string& rp : ref_image_paths) {
        int64_t h2 = 0, w2 = 0;
        std::cerr << "loading reference image " << rp << "\n";
        imgs.push_back(ReadPpmAsChw(rp, &h2, &w2));
        if (ih == 0) { ih = h2; iw = w2; }
        VT_CHECK(h2 == ih && w2 == iw,
                 "minimax-h3-gen: every --ref-image must have the same size");
      }
      std::vector<vllm::MiniMaxH3RefBlock> blocks;
      request.keyframe_cond_rows = vllm::MiniMaxH3EncodeReferenceImages(
          enc_cfg, enc_w, dit.params, imgs, ih, iw, &blocks);
      request.ref_blocks = blocks;
      std::cerr << "  ref2va: " << blocks.size() << " reference image(s) at " << iw << "x" << ih
                << "\n";
    }

    // --- ref2va REFERENCE AUDIO: a waveform prepended to the sequence, through
    // the audio VAE's ENCODER half. It ATTACHES to a video reference when there is
    // one (one kVideoAudio block carrying both, which is the layout
    // packed_sequence.py builds), and otherwise stands alone as a kAudio block.
    if (!ref_audio_path.empty()) {
      VT_CHECK(first_frame_path.empty() && last_frame_path.empty(),
               "minimax-h3-gen: --ref-audio (ref2va) and --first/--last-frame (fl2va) are "
               "exclusive");
      VT_CHECK(!audio_vae_path.empty(),
               "minimax-h3-gen: --ref-audio needs --audio-vae (the ENCODER half)");
      VT_CHECK(!audio_cfg_path.empty(),
               "minimax-h3-gen: --ref-audio needs --audio-vae-config (it carries the latent "
               "statistics the reference rows are normalized by)");
      std::cerr << "loading reference audio " << ref_audio_path << "\n";
      int64_t wav_channels = 0, wav_samples = 0;
      const std::vector<float> wave = ReadWavRef(ref_audio_path, &wav_channels, &wav_samples);

      // The audio VAE file is opened again for its ENCODER half; the decoder
      // loader deliberately skips it, so the two halves are separate weight sets.
      const vllm::SafetensorsFile af = vllm::SafetensorsFile::Open(audio_vae_path);
      const vllm::MiniMaxH3AudioVaeWeights aenc = vllm::LoadMiniMaxH3AudioVaeEncoderWeights(af);
      vllm::MiniMaxH3AudioVaeEncoderConfig aenc_cfg;  // the shipped geometry
      aenc_cfg.vae_latent_channels = dit.params.audio_latents_dim;

      vllm::MiniMaxH3RefBlock ab{};
      const std::vector<float> arows = vllm::MiniMaxH3EncodeReferenceAudio(
          aenc_cfg, aenc, wave, wav_channels, wav_samples, audio_stats.mean, audio_stats.std_dev,
          /*noise_aug=*/1.0, {}, &ab);
      request.audio_ref_rows = arows;
      if (request.ref_blocks.size() == 1 &&
          request.ref_blocks[0].kind == vllm::MiniMaxH3RefBlock::Kind::kVideoAudio) {
        // A video reference that now has SOUND: same block, non-zero ref_audio_t.
        request.ref_blocks[0].ref_audio_t = ab.ref_audio_t;
        std::cerr << "  ref2va: reference video now carries audio, " << ab.ref_audio_t
                  << " latent frames\n";
      } else {
        request.ref_blocks.push_back(ab);
        std::cerr << "  ref2va: reference AUDIO " << wav_samples << " samples/channel -> "
                  << ab.ref_audio_t << " latent frames\n";
      }
    }

    // --- fl2va KEYFRAMES: encode the supplied frame(s) into pinned conditioning.
    // Upstream allows exactly {0}, {-1} or {0, -1}: first, last, or both.
    if (!first_frame_path.empty() || !last_frame_path.empty()) {
      VT_CHECK(!video_vae_path.empty(),
               "minimax-h3-gen: --first-frame/--last-frame need --video-vae (the ENCODER half)");
      const vllm::SafetensorsFile vf = vllm::SafetensorsFile::Open(video_vae_path);
      const vllm::MiniMaxH3AudioVaeWeights enc_w = vllm::LoadMiniMaxH3VideoVaeEncoderWeights(vf);
      vllm::MiniMaxH3EncoderFcn3dConfig enc_cfg;
      enc_cfg.z_channels = 2 * dit.params.latents_dim;  // moments: mean | logvar

      std::vector<std::vector<float>> imgs;
      std::vector<int64_t> idx;
      int64_t ih = 0, iw = 0;
      if (!first_frame_path.empty()) {
        std::cerr << "loading first frame " << first_frame_path << "\n";
        imgs.push_back(ReadPpmAsChw(first_frame_path, &ih, &iw));
        idx.push_back(0);
      }
      if (!last_frame_path.empty()) {
        int64_t h2 = 0, w2 = 0;
        std::cerr << "loading last frame " << last_frame_path << "\n";
        imgs.push_back(ReadPpmAsChw(last_frame_path, &h2, &w2));
        if (ih == 0) { ih = h2; iw = w2; }
        VT_CHECK(h2 == ih && w2 == iw,
                 "minimax-h3-gen: the first and last frames must have the same size");
        idx.push_back(-1);
      }
      request.keyframe_frame_indices = idx;
      request.imgvid_noise_aug = imgvid_noise_aug;
      request.keyframe_cond_rows = vllm::MiniMaxH3EncodeKeyframeCondRows(
          enc_cfg, enc_w, dit.params, imgs, ih, iw, request.latent_t, imgvid_noise_aug, {});
      std::cerr << "  keyframe conditioning: " << imgs.size() << " frame(s) at " << iw << "x" << ih
                << " -> " << request.keyframe_cond_rows.size() << " floats\n";
    }

    // Real conditioning when a prompt was encoded; otherwise the supplied file.
    const std::vector<float> prompt_embeds =
        !encoded_prompt.empty() ? encoded_prompt : ReadF32(embeds_path);
    if (dit.params.text_dim <= 0 || prompt_embeds.size() % static_cast<size_t>(dit.params.text_dim) != 0) {
      throw std::runtime_error("--prompt-embeds size is not a multiple of text_dim");
    }
    request.text_len = static_cast<int64_t>(prompt_embeds.size()) / dit.params.text_dim;
    std::cerr << "  text_len=" << request.text_len << " latent=" << request.latent_t << "x"
              << request.latent_h << "x" << request.latent_w << " steps=" << request.num_steps
              << "\n";

    if (dry_run) {
      std::cerr << "--dry-run: everything LOADED and planned; stopping before generation\n";
      return 0;
    }

    // --- 4. generate ---
    // NOISE IS AN INPUT (upstream seeds a torch CPU generator, and reproducing its
    // RNG bit-exactly decides WHICH sample you get, not whether the pipeline is
    // right), so it is drawn here from the shared deterministic stream.
    const int64_t frame_rows = (request.latent_h / dit.params.patch_size_h) *
                               (request.latent_w / dit.params.patch_size_w);
    const int64_t video_rows = request.latent_t * frame_rows;
    const int64_t audio_rows = request.audio_t * request.audio_channel;
    // A splitmix64 stream, seeded so a run is REPRODUCIBLE. This deliberately does
    // NOT reproduce torch's RNG: matching it bit-for-bit decides WHICH sample you
    // get, not whether the pipeline is correct, and pretending otherwise would
    // invite comparing our sample against upstream's as if they should match.
    // A flow-matching model is trained with GAUSSIAN N(0,1) noise at sigma=1
    // (torch.randn); feeding uniform[-1,1] (std 0.577) is out-of-distribution.
    // Upstream draws torch.randn — Gaussian is the DEFAULT (mirror policy);
    // VT_H3_GAUSSIAN_NOISE=0 keeps the legacy uniform draw for the A/B.
    const char* gn = std::getenv("VT_H3_GAUSSIAN_NOISE");
    const bool gaussian = !(gn != nullptr && gn[0] == '0');
    auto fill = [gaussian](std::vector<float>& out, uint64_t seed) {
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
    };
    std::vector<float> noise_video(
        static_cast<size_t>(video_rows * dit.params.video_row_width()));
    std::vector<float> noise_audio(
        static_cast<size_t>(audio_rows * dit.params.audio_latents_dim));
    fill(noise_video, 0x5EED1234ULL);
    fill(noise_audio, 0x5EED5678ULL);

    std::cerr << "generating (" << request.num_steps << " steps)...\n";
    // On a device, the denoise loop stages the DiT weights ONCE and runs every step
    // device-resident. On CPU it uses the portable reference forward, which is a
    // correctness path, not a throughput one.
    vt::Device device{};
    if (device_name == "cuda") {
      device = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue().device;
    } else if (device_name != "cpu") {
      throw std::runtime_error("--device must be cpu or cuda");
    }
    // Stage ONCE here rather than inside the denoise loop. --dequant-bf16 trades
    // memory (DiT ~33 GB bf16 vs 15.6 GB kept-quant) for GEMM throughput: the
    // keep-quant path measured ~103 GFLOP/s, which is what makes a full-quality
    // render a multi-day job.
    vllm::MiniMaxH3DitDeviceWeights staged;
    const vllm::MiniMaxH3DitDeviceWeights* prestaged = nullptr;
    if (have_streamed) {
      prestaged = &streamed;
    } else if (device.type != vt::DeviceType::kCPU) {
      vt::Queue sq = vt::GetBackend(device.type).CreateQueue();
      const auto t0 = std::chrono::steady_clock::now();
      staged = vllm::StageMiniMaxH3DitWeights(sq, dit.params, dit.weights, vt::DType::kBF16);
      std::cerr << "  staged DiT (" << (dequant_bf16 ? "dequant-bf16" : "keep-quant") << ") in "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                << " s\n";
      prestaged = &staged;
    }
    if (denoise_only) {
      // Time the DiT step loop by itself. Reported as an AVERAGE over the requested
      // steps rather than a single step: the first step pays one-off costs (RoPE
      // caches, allocator warm-up) that do not recur, so a one-step run overstates
      // the steady-state cost.
      const auto t0 = std::chrono::steady_clock::now();
      const vllm::MiniMaxH3DenoiseResult denoised = vllm::MiniMaxH3DenoiseT2va(
          device, request, dit.params, dit.weights, prompt_embeds, noise_video, noise_audio,
          vt::DType::kBF16, prestaged);
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const int64_t seq = static_cast<int64_t>(request.text_len) + video_rows + audio_rows;
      // A flow-matching schedule of N sigmas runs N-1 DiT FORWARDS (each step maps
      // sigma[i] -> sigma[i+1]), which is exactly what the `[h3] step i/N-1` trace
      // prints. Dividing the elapsed time by `num_steps` therefore UNDER-reports the
      // per-forward cost by N/(N-1) -- 1.5x at the 3-step setting used for kernel
      // A/Bs, which is large enough to make two measurements look like a regression.
      // Report the per-FORWARD cost, which is what the trace and any kernel
      // comparison are denominated in.
      const int64_t forwards = request.num_steps > 1 ? request.num_steps - 1 : 1;
      std::cout << "denoise-only: seq=" << seq << " steps=" << request.num_steps
                << " forwards=" << forwards << " total=" << elapsed
                << " s  per_forward=" << (elapsed / double(forwards)) << " s\n";
      // Touch the result so the loop cannot be optimized away, and so an all-NaN
      // forward shows up here instead of passing as a fast run.
      double checksum = 0.0;
      for (const float v : denoised.video_rows) checksum += v;
      std::cout << "  video_rows=" << denoised.video_rows.size() << " checksum=" << checksum
                << "\n";
      return 0;
    }

    const vllm::MiniMaxH3T2vaResult result = vllm::MiniMaxH3GenerateT2va(
        device, request, dit.params, dit.weights, video_cfg, video_weights, audio_cfg,
        audio_weights, prompt_embeds, noise_video, noise_audio, vt::DType::kBF16, prestaged);

    // --- 5. artifacts (the LIBRARY builds these; nothing spawns) ---
    std::string mkdir_cmd = "mkdir -p '" + workdir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) throw std::runtime_error("cannot create " + workdir);
    for (int64_t f = 0; f < result.frame_shape.t; ++f) {
      char name[512];
      std::snprintf(name, sizeof(name), "%s/frame_%06lld.ppm", workdir.c_str(),
                    static_cast<long long>(f));
      WriteFile(name, vllm::MiniMaxH3WritePpmFrame(result.frames, result.frame_shape, f));
    }
    const std::string wav_path = workdir + "/audio.wav";
    WriteFile(wav_path, vllm::MiniMaxH3WriteWav(result.waveform, result.audio_channels,
                                                result.audio_samples_per_channel,
                                                result.sample_rate));
    std::cerr << "  wrote " << result.frame_shape.t << " frames + " << wav_path << "\n";

    // --- 6. mux (the ONE process spawn, and it is in examples/ by decision) ---
    vllm::MiniMaxH3MuxRequest mux;
    mux.frame_pattern = workdir + "/frame_%06d.ppm";
    mux.audio_path = wav_path;
    mux.output_path = out_path;
    std::vector<std::string> args = vllm::MiniMaxH3BuildMp4MuxArgs(mux);
    if (!args.empty()) args[0] = ffmpeg;
    const int status = RunFfmpeg(args);
    if (status != 0) {
      std::cerr << "ffmpeg exited " << status << "\n";
      return status;
    }
    std::cout << "wrote " << out_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
