// minimax-music3-gen — MiniMax-Music3 text-to-music end to end, as a THIN
// CLIENT of the public C ABI (include/vllm.h) and NOTHING else, per the ONE
// SURFACE directive. It includes no internal header, and every capability it
// reaches is one an embedder gets: vllm_speech_engine_load, the three
// interrogations of the loaded handle, vllm_synthesize, and the RIFF bytes the
// result already carries.
//
// It exists because the model had NO example. `/v1/audio/speech` needed a
// running server plus a `curl`, and the C ABI needed a caller nobody had
// written — so the shortest path to hearing this model was neither. Every other
// generative family in this tree has one (minimax-h3-gen, ltx2-gen), and the
// music family was the exception.
//
//   minimax-music3-gen --model <checkpoint-set-dir> --out <song.wav>
//                      --lyrics <text|@file> [--description <text|@file>]
//                      [--duration SECONDS] [--steps N] [--guidance F]
//                      [--seed N] [--family <name>]
//
// `--model` names the checkpoint SET, not a file: the diffusers arm ships six
// component directories beside a `modular_model_index.json`. `--family` is
// optional — omitted, the family is DETECTED by inspecting the artifact, and a
// directory no registered family claims is refused naming every family tried.
//
// `--lyrics` and `--description` take either literal text or `@path` to read a
// file, because lyrics are multi-line by nature and a shell heredoc inside an
// argv is how a `[Verse]` tag ends up mangled.
//
// WHAT THIS CANNOT DO, refused rather than faked: there is no streaming (the
// whole song exists before the first sample does — upstream has none either),
// no format other than RIFF/WAVE 16-bit PCM (no mp3/opus encoder is vendored),
// and no resample off the family's native 44100 Hz stereo. Those are the same
// refusals `/v1/audio/speech` makes, for the same reasons.
//
// IT IS SLOW ON CPU and no speed number is claimed. The acoustic half is
// upstream's own fp32 and the AR half's host GEMM is a scalar loop written for
// a reproducible reduction order (.agents/specs/minimax-music3.md `## Now`).
// Ask for a short `--duration` and few `--steps` while checking that it works.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vllm.h"

namespace {

// `@path` reads the file; anything else is the literal text. Returns false only
// when a `@path` cannot be read — a literal never fails.
bool ResolveText(const std::string& arg, std::string* out) {
  if (arg.empty() || arg[0] != '@') {
    *out = arg;
    return true;
  }
  const std::string path = arg.substr(1);
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "error: cannot read %s\n", path.c_str());
    return false;
  }
  out->clear();
  char buf[4096];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
  std::fclose(f);
  return true;
}

void Usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model <checkpoint-set-dir> --out <song.wav>\n"
               "          --lyrics <text|@file> [--description <text|@file>]\n"
               "          [--duration SECONDS] [--steps N] [--guidance F]\n"
               "          [--seed N] [--family <name>]\n"
               "\n"
               "  --model      the checkpoint SET directory (six component dirs +\n"
               "               modular_model_index.json), NOT a single file\n"
               "  --lyrics     the sung text, with [Verse]/[Chorus] section tags.\n"
               "               REQUIRED: there is nothing to sing without it\n"
               "  --description  genre, BPM, key, instrumentation, mood. NOT a voice\n"
               "  --duration   seconds of audio; omitted => the family default (60 s)\n"
               "  --steps      denoise steps; omitted => the family default (30)\n"
               "  --guidance   CFG scale; 0 IS legal, so omitted (not 0) means default\n"
               "  --seed       seeds the AR sampling and the initial denoise latents\n"
               "  --family     skip detection and name the family\n",
               argv0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, out_path, family;
  std::string lyrics_arg, description_arg;
  double duration = 0.0;      // <= 0 => family default
  int64_t steps = 0;          // <= 0 => family default
  double guidance = 0.0;      // only honoured when guidance_given
  bool guidance_given = false;  // 0 is a LEGAL guidance scale, so presence is a flag
  int64_t seed = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s takes a value\n", name);
        Usage(argv[0]);
        std::exit(2);
      }
      return argv[++i];
    };
    if (flag == "--model") {
      model = next("--model");
    } else if (flag == "--out") {
      out_path = next("--out");
    } else if (flag == "--lyrics") {
      lyrics_arg = next("--lyrics");
    } else if (flag == "--description" || flag == "--prompt") {
      description_arg = next(flag.c_str());
    } else if (flag == "--duration") {
      duration = std::atof(next("--duration").c_str());
    } else if (flag == "--steps") {
      steps = std::atoll(next("--steps").c_str());
    } else if (flag == "--guidance") {
      guidance = std::atof(next("--guidance").c_str());
      guidance_given = true;
    } else if (flag == "--seed") {
      seed = std::atoll(next("--seed").c_str());
    } else if (flag == "--family") {
      family = next("--family");
    } else if (flag == "--help" || flag == "-h") {
      Usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "error: unknown argument '%s'\n", flag.c_str());
      Usage(argv[0]);
      return 2;
    }
  }
  if (model.empty() || out_path.empty() || lyrics_arg.empty()) {
    std::fprintf(stderr, "error: --model, --out and --lyrics are required\n");
    Usage(argv[0]);
    return 2;
  }

  std::string lyrics, description;
  if (!ResolveText(lyrics_arg, &lyrics)) return 1;
  if (!description_arg.empty() && !ResolveText(description_arg, &description)) return 1;

  vllm_speech_model_params mp = vllm_speech_model_params_default();
  mp.path = model.c_str();
  if (!family.empty()) mp.family = family.c_str();

  const auto load_t0 = std::chrono::steady_clock::now();
  vllm_speech_engine* engine = nullptr;
  if (vllm_speech_engine_load(&mp, &engine) != VLLM_OK) {
    std::fprintf(stderr, "load failed: %s\n", vllm_last_error());
    return 1;
  }
  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count();
  std::fprintf(stderr, "loaded family '%s' at %d Hz in %.1f s\n",
               vllm_speech_engine_family(engine), vllm_speech_engine_sample_rate(engine),
               load_s);

  // Ask BEFORE synthesizing, which is the whole reason this interrogation is on
  // the ABI: a family with no text-only synthesis is a caller-side refusal
  // rather than a job that fails after staging tens of gigabytes.
  if (vllm_speech_engine_requires_reference_audio(engine) != 0) {
    std::fprintf(stderr,
                 "error: family '%s' requires a reference clip, which this example does not "
                 "supply — it drives the text-only music path\n",
                 vllm_speech_engine_family(engine));
    vllm_speech_engine_free(engine);
    return 1;
  }

  vllm_speech_params sp = vllm_speech_params_default();
  sp.lyrics = lyrics.c_str();
  if (!description.empty()) sp.description = description.c_str();
  sp.audio_duration_s = duration;
  sp.num_inference_steps = static_cast<int32_t>(steps);
  if (guidance_given) {
    sp.guidance_scale = guidance;
    sp.has_guidance_scale = 1;
  }
  sp.seed = seed;

  const auto gen_t0 = std::chrono::steady_clock::now();
  vllm_speech_result result;
  const vllm_status st = vllm_synthesize(engine, &sp, &result);
  const double gen_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - gen_t0).count();
  if (st != VLLM_OK) {
    std::fprintf(stderr, "synthesize failed: %s\n", vllm_last_error());
    vllm_speech_engine_free(engine);
    return 1;
  }

  // The result carries the RIFF bytes as well as the float samples, so writing a
  // playable file needs no second encoder here.
  std::FILE* sink = std::fopen(out_path.c_str(), "wb");
  if (sink == nullptr) {
    std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
    vllm_speech_result_free(&result);
    vllm_speech_engine_free(engine);
    return 1;
  }
  const size_t written =
      std::fwrite(result.wav, 1, static_cast<size_t>(result.n_wav), sink);
  std::fclose(sink);
  if (written != static_cast<size_t>(result.n_wav)) {
    std::fprintf(stderr, "error: short write to %s (%zu of %lld bytes)\n", out_path.c_str(),
                 written, static_cast<long long>(result.n_wav));
    vllm_speech_result_free(&result);
    vllm_speech_engine_free(engine);
    return 1;
  }

  // Report what came back rather than what was asked for. A duration is a
  // REQUEST: it resolves to a whole number of 25 Hz autoregressive frames and
  // then to a whole number of latent frames, so the delivered length is
  // quantized and printing the request instead would misreport the file.
  const double seconds = result.sample_rate > 0
                             ? static_cast<double>(result.n_samples) / result.sample_rate
                             : 0.0;
  double peak = 0.0;
  double energy = 0.0;
  const int64_t total = result.n_samples * result.channels;
  for (int64_t i = 0; i < total; ++i) {
    const double v = result.samples[i];
    const double a = v < 0.0 ? -v : v;
    if (a > peak) peak = a;
    energy += v * v;
  }
  const double rms = total > 0 ? std::sqrt(energy / static_cast<double>(total)) : 0.0;
  std::fprintf(stderr,
               "wrote %s: %.3f s, %d Hz, %d channel(s), %lld samples/channel, "
               "RMS %.5f, peak %.5f, %.1f s wall\n",
               out_path.c_str(), seconds, result.sample_rate, result.channels,
               static_cast<long long>(result.n_samples), rms, peak, gen_s);

  vllm_speech_result_free(&result);
  vllm_speech_engine_free(engine);
  return 0;
}
