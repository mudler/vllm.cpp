// ltx2-gen — an LTX-2.5 video+audio render, end to end, as a THIN CLIENT of the
// public C ABI (include/vllm.h) and NOTHING else, per the ONE SURFACE directive
// (ARCH-ONE-SURFACE ROW 2). Row MODEL-DIFFUSION-LTX25 phase L9B, issue #435.
//
// WHY A SECOND GENERATION EXAMPLE AND NOT A FLAG ON minimax-h3-gen. The ABI's
// video slice went family-generic at v18: `family` selects the model family and
// two parallel string arrays carry the FAMILY-SPECIFIC load knobs
// (vllm.h:735-747). `minimax-h3-gen` predates that and drives neither, and
// LTX-2.5 CANNOT LOAD without three of them — the audio-stream prompt embeds,
// the DiT config the shipped FP8 checkpoint does not carry, and (for the second
// distilled phase) the latent spatial upsampler. This file exists to name those
// knobs as flags rather than to hand a user a `--extra key=value` grab bag, and
// it is the smallest thing that can drive a real render.
//
// WHAT IT DOES NOT DO. It composes no ffmpeg command line of its own and encodes
// nothing: `vllm_video_mux_argv` builds the argv and this file exec's it, which
// is the ratified process boundary (2026-08-03). It carries no model logic — no
// noise stream, no schedule, no dtype choice — because all of that is the
// library's and a second copy here would be a parallel path.
//
// CONDITIONING HAS TWO SOURCES, and as of phase L13 the first of them is a
// typed prompt. `--encoder` names the Gemma-4 12B text tower and `--prompt`
// carries the words; the tower tokenizes them with its OWN embedded tokenizer,
// runs, projects all 49 hidden states to the two stream widths and hands the
// result to the embeddings connector and then to cross-attention. Before L13
// the tower had no route to the DiT and `--prompt` did not exist here at all.
//
// `--prompt-embeds` + `--audio-prompt-embeds` remain, and remain the only
// conditioning without a tower. Both streams are conditioned or neither:
// LTX-2.5 cross-attends at TWO widths (4096 video, 2048 audio) and one of them
// alone leaves a stream unconditioned, which renders instead of failing.
//
// `--encoder-config` is not optional paperwork. The only shipped LTX-2.5 text
// encoder carries no `__metadata__` at all, so its Gemma config cannot come out
// of the file; the engine refuses rather than defaulting one, because
// `layer_types`, `global_head_dim` and `attention_k_eq_v` resolve a DIFFERENT
// tower out of a byte-identical tensor set.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vllm.h"

namespace {

// The ONE process spawn, in examples/ by decision: the library composed `args`;
// this runs it. No shell — the argv is exec'd directly.
int RunFfmpeg(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  const pid_t pid = fork();
  if (pid < 0) {
    std::fprintf(stderr, "error: fork failed\n");
    return -1;
  }
  if (pid == 0) {
    execvp(argv[0], argv.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::fprintf(stderr, "error: waitpid failed\n");
    return -1;
  }
  if (WIFSIGNALED(status)) {
    std::fprintf(stderr, "error: ffmpeg died on signal %d\n", WTERMSIG(status));
    return -1;
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

const char* Need(int argc, char** argv, int i, const char* flag) {
  if (i >= argc) {
    std::fprintf(stderr, "error: missing value for %s\n", flag);
    std::exit(2);
  }
  return argv[i];
}

[[noreturn]] void Usage(int code) {
  std::fprintf(
      stderr,
      "usage: ltx2-gen --dit <transformer.safetensors> --video-vae <f> --audio-vae <f>\n"
      "                (--encoder <gemma4-te.safetensors> --prompt \"...\"\n"
      "                 | --prompt-embeds <video.f32.bin> --audio-prompt-embeds <audio.f32.bin>)\n"
      "                --workdir DIR [--out <out.mp4>] [--ffmpeg PATH]\n"
      "                [--encoder-config <gemma_config.json>]     REQUIRED when the text\n"
      "                                                           encoder carries no metadata\n"
      "                [--dit-config <transformer-config.json>]   REQUIRED when the DiT\n"
      "                                                           carries no __metadata__\n"
      "                [--model-version 2.5] [--pipeline-kind distilled_two_stage]\n"
      "                [--upsampler <latent-spatial-x2.safetensors>]  phase 2 needs it\n"
      "                [--max-phase N] [--allow-unported]\n"
      "                [--prompt-valid-rows N]   how many embed rows are real tokens\n"
      "                [--frames N] [--width N] [--height N] [--seed N]\n"
      "                [--first-frame <image.ppm>] [--image-crf 0]\n"
      "                [--device cpu|cuda]\n\n"
      "Renders LTX-2.5 (family \"ltx-2.5\") through vllm_video_engine_load +\n"
      "vllm_video_generate.\n\n"
      "CONDITIONING, two ways. With --encoder the Gemma-4 12B text tower is loaded and\n"
      "--prompt is tokenized by the tower's OWN embedded tokenizer, run, projected to\n"
      "the two stream widths and passed through the embeddings connector. The tower is\n"
      "~24 GB of host bf16 and stays resident, because a prompt arrives per request.\n"
      "--encoder-config supplies the Gemma config when the encoder declares none, which\n"
      "the only shipped one does not; without either the load is refused rather than\n"
      "defaulted, since the config decides which layers are full-attention and a wrong\n"
      "one resolves a different tower from the same tensors.\n\n"
      "Without --encoder, conditioning is PROMPT-EMBEDS: both files are rows of\n"
      "little-endian f32, the video one 4096 wide and the audio one 2048, with the\n"
      "SAME row count. Passing --prompt without --encoder is refused rather than\n"
      "silently rendering those embeddings as if they were it.\n\n"
      "Those rows are the EMBEDDINGS CONNECTOR's input, not the DiT's: when the\n"
      "checkpoint carries the two *_embeddings_connector families (both shipped\n"
      "LTX-2.5 DiTs do) they run through it first, with the checkpoint's own\n"
      "weights. The row count must then be a multiple of the connector's learnable\n"
      "register count (128 on the shipped files), and --prompt-valid-rows says how\n"
      "many of them are real: the rest are padding, and padding is REPLACED by the\n"
      "learnable register table rather than ignored.\n\n"
      "IMAGE CONDITIONING (image-to-video). --first-frame takes a binary PPM (P6,\n"
      "maxval 255) and pins latent frame 0 to it: it is decoded, aspect-filled and\n"
      "centre-cropped to each phase's own resolution, VAE-encoded, and written into\n"
      "the clean latent. It needs --image-crf 0, and that is DELIBERATELY not the\n"
      "default. Upstream re-compresses a conditioning image through H.264 at the CRF\n"
      "the checkpoint's generation was trained with, which for LTX-2.5 is 18; that\n"
      "round trip needs libx264 and none is vendored here, so leaving --image-crf out\n"
      "resolves 18 and REFUSES by name. --image-crf 0 is upstream-legal and OUT OF\n"
      "DISTRIBUTION: the model sees pixels it was not trained on. That is a quality\n"
      "cost, and this tool states it rather than turning it on quietly.\n");
  std::exit(code);
}

}  // namespace

int main(int argc, char** argv) {
  vllm_video_model_params mp = vllm_video_model_params_default();
  vllm_video_params vp = vllm_video_params_default();
  std::string workdir = "/tmp/ltx2_gen", out_path, ffmpeg = "ffmpeg", device = "cuda";
  // BORROWED by `vllm_video_generate`, like the extras below, so it is owned
  // here and pointed at only after parsing.
  std::string prompt, first_frame, image_crf;

  // The extras are BORROWED by the load call, so the strings must outlive it.
  // Kept as two parallel vectors of owned strings plus the char* views the ABI
  // takes, built once after parsing.
  std::vector<std::string> extra_keys, extra_values;
  auto SetExtra = [&](const char* key, std::string value) {
    for (size_t i = 0; i < extra_keys.size(); ++i) {
      if (extra_keys[i] == key) {
        extra_values[i] = std::move(value);
        return;
      }
    }
    extra_keys.emplace_back(key);
    extra_values.push_back(std::move(value));
  };

  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    if (f == "--dit") mp.dit_path = Need(argc, argv, ++i, "--dit");
    else if (f == "--video-vae") mp.video_vae_path = Need(argc, argv, ++i, "--video-vae");
    else if (f == "--video-vae-config") mp.video_vae_config_path = Need(argc, argv, ++i, f.c_str());
    else if (f == "--audio-vae") mp.audio_vae_path = Need(argc, argv, ++i, "--audio-vae");
    else if (f == "--audio-vae-config") mp.audio_vae_config_path = Need(argc, argv, ++i, f.c_str());
    else if (f == "--encoder") mp.encoder_path = Need(argc, argv, ++i, "--encoder");
    else if (f == "--encoder-config")
      SetExtra("encoder_config_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--prompt") prompt = Need(argc, argv, ++i, "--prompt");
    else if (f == "--prompt-embeds") mp.prompt_embeds_path = Need(argc, argv, ++i, f.c_str());
    else if (f == "--audio-prompt-embeds")
      SetExtra("audio_prompt_embeds_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--dit-config") SetExtra("dit_config_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--model-version") SetExtra("model_version", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--pipeline-kind") SetExtra("pipeline_kind", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--upsampler") SetExtra("upsampler_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--duration-head")
      SetExtra("duration_head_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--max-phase") SetExtra("max_phase", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--prompt-valid-rows")
      SetExtra("prompt_embeds_valid_rows", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--allow-unported") SetExtra("allow_unported_modules", "1");
    // Image conditioning (row LTX25-IMAGE-COND, issue #644). `--first-frame` is
    // a binary PPM; `--image-crf` is the PER-GENERATION extra, so it rides
    // vp.extra_* rather than mp.extra_*. Only 0 is served, and it is NOT
    // defaulted here — leaving it out lets the engine resolve the checkpoint's
    // own 18 and refuse, which is the point: this CLI must not be the thing that
    // quietly turns an out-of-distribution render on.
    else if (f == "--first-frame") first_frame = Need(argc, argv, ++i, "--first-frame");
    else if (f == "--image-crf") image_crf = Need(argc, argv, ++i, "--image-crf");
    else if (f == "--device") device = Need(argc, argv, ++i, "--device");
    else if (f == "--frames") vp.num_frames = std::atoi(Need(argc, argv, ++i, "--frames"));
    else if (f == "--width") vp.width = std::atoi(Need(argc, argv, ++i, "--width"));
    else if (f == "--height") vp.height = std::atoi(Need(argc, argv, ++i, "--height"));
    else if (f == "--seed") {
      vp.seed = static_cast<uint64_t>(std::strtoull(Need(argc, argv, ++i, "--seed"), nullptr, 10));
      vp.has_seed = 1;
    }
    else if (f == "--workdir") workdir = Need(argc, argv, ++i, "--workdir");
    else if (f == "--out") out_path = Need(argc, argv, ++i, "--out");
    else if (f == "--ffmpeg") ffmpeg = Need(argc, argv, ++i, "--ffmpeg");
    else if (f == "--help" || f == "-h") Usage(0);
    else {
      std::fprintf(stderr, "error: unknown argument: %s\n", f.c_str());
      Usage(2);
    }
  }
  if (mp.dit_path == nullptr) Usage(2);
  if (device == "cuda") mp.device = 1;
  else if (device != "cpu") {
    std::fprintf(stderr, "error: --device must be cpu or cuda\n");
    return 2;
  }
  // DECLARED, never detected. Detection would also resolve this checkpoint, but
  // an explicit family is what makes an FP8-vs-NVFP4 comparison a statement
  // about the two files rather than about what a detector happened to claim.
  mp.family = "ltx-2.5";
  vp.output_dir = workdir.c_str();
  if (!prompt.empty()) vp.prompt = prompt.c_str();
  if (!first_frame.empty()) vp.first_frame = first_frame.c_str();

  // The PER-GENERATION extras are a SEPARATE array from the load-time ones, and
  // conflating them is the whole failure this keeps apart: `image_crf` handed to
  // the load call is an unknown LOAD extra and is refused there, which would
  // read as "the flag does not work" rather than as "it goes on the other call".
  std::vector<std::string> gen_keys, gen_values;
  if (!image_crf.empty()) {
    gen_keys.emplace_back("image_crf");
    gen_values.push_back(image_crf);
  }
  std::vector<const char*> gkeys, gvalues;
  for (size_t i = 0; i < gen_keys.size(); ++i) {
    gkeys.push_back(gen_keys[i].c_str());
    gvalues.push_back(gen_values[i].c_str());
  }
  if (!gkeys.empty()) {
    vp.extra_keys = gkeys.data();
    vp.extra_values = gvalues.data();
    vp.n_extras = static_cast<int32_t>(gkeys.size());
  }

  std::vector<const char*> keys, values;
  keys.reserve(extra_keys.size());
  values.reserve(extra_values.size());
  for (size_t i = 0; i < extra_keys.size(); ++i) {
    keys.push_back(extra_keys[i].c_str());
    values.push_back(extra_values[i].c_str());
  }
  if (!keys.empty()) {
    mp.extra_keys = keys.data();
    mp.extra_values = values.data();
    mp.n_extras = static_cast<int32_t>(keys.size());
  }

  vllm_video_engine* engine = nullptr;
  if (vllm_video_engine_load(&mp, &engine) != VLLM_OK) {
    std::fprintf(stderr, "error: %s\n", vllm_last_error());
    return 1;
  }
  // Which family actually loaded, from the handle rather than from the request:
  // spec §3.1 requires every artifact to name what produced it.
  std::fprintf(stderr, "ltx2-gen: family=%s dit=%s\n", vllm_video_engine_family(engine),
               mp.dit_path);

  vllm_video_result out;
  if (vllm_video_generate(engine, &vp, &out) != VLLM_OK) {
    std::fprintf(stderr, "error: %s\n", vllm_last_error());
    vllm_video_engine_free(engine);
    return 1;
  }
  std::fprintf(stderr, "  wrote %d frames (%dx%d @ %d fps) + %s (%d Hz)\n", out.frame_count,
               out.width, out.height, out.fps, out.audio_path, out.sample_rate);

  int status = 0;
  if (!out_path.empty()) {
    const std::string pattern = std::string(out.frame_dir) + "/frame_%06d.ppm";
    vllm_video_mux_params mx = vllm_video_mux_params_default();
    mx.frames = pattern.c_str();
    mx.audio_path = out.audio_path;
    mx.output_path = out_path.c_str();
    mx.fps = out.fps;  // the RECIPE's frame rate, not the mux default
    char** mux_argv = nullptr;
    int32_t mux_argc = 0;
    if (vllm_video_mux_argv(&mx, &mux_argv, &mux_argc) != VLLM_OK) {
      std::fprintf(stderr, "error: %s\n", vllm_last_error());
      vllm_video_result_free(&out);
      vllm_video_engine_free(engine);
      return 1;
    }
    std::vector<std::string> args(mux_argv, mux_argv + mux_argc);
    if (!args.empty()) args[0] = ffmpeg;
    vllm_video_mux_argv_free(mux_argv, mux_argc);
    status = RunFfmpeg(args);
    if (status == 0) {
      std::printf("wrote %s\n", out_path.c_str());
    } else {
      std::fprintf(stderr, "ffmpeg exited %d\n", status);
    }
  }

  vllm_video_result_free(&out);
  vllm_video_engine_free(engine);
  return status;
}
