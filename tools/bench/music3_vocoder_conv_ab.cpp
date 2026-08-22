// A/B driver for MiniMax-Music3's vocoder decode window (#672, #1334).
//
// WHAT IT MEASURES, AND WHAT IT IS NOT. It calls `music3::VocoderDecode` — the
// production entry point `Music3DecodeChunks` calls inside the
// `vocoder.decode_window` profile bracket (minimax_music3_speech.cpp:397-398) —
// at the shipped `MiniMaxMusic3VocoderConfig` geometry, for a sweep of latent
// window lengths. It is therefore that bucket's computation and not a
// reconstruction of it. It is NOT an end-to-end synthesis: the weights are
// synthetic, so nothing here is a claim about audio, and it takes no GPU clock
// window (.agents/benchmarking.md), so nothing here is a per-kernel or
// cross-box figure.
//
// WHY SYNTHETIC WEIGHTS ARE HONEST HERE. The conv kernels' cost does not depend
// on the weight VALUES, with one exception that is handled rather than ignored:
// `vt::ConvTranspose1d` skips an input that compares equal to 0.0, so a
// degenerate all-zero input would measure a path the real one never takes. The
// generator below emits no exact zero, and both arms are seeded identically, so
// the two arms execute the same number of the same operations.
//
// HOW THE ARMS ARE SEPARATED. There is no `#if` here and there is no flag. The
// two arms are two SOURCE TREES that differ in
// `src/vt/cpu/cpu_conv1d_general.cpp` and nothing else, built in two build
// directories, and the harness prints a checksum of the waveform so the run can
// assert that the arms agree BIT FOR BIT while their binaries differ. Equal
// times are noise; equal binaries are identity, and that is what voided the
// first attempt at spec §16.6a.
//
// CI NEVER RUNS THIS. It is an executable rather than a registered test, it
// allocates hundreds of megabytes of synthetic weights, and it spends tens of
// seconds per sweep. It is compiled by the ordinary build so that the one file a
// reader has to compile to reproduce the measurement cannot rot silently.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_music3_acoustic.h"
#include "vllm/model_executor/models/minimax_music3_loader.h"
#include "vllm/model_executor/models/music3_profile.h"

namespace {

using vllm::MiniMaxMusic3VocoderConfig;
using vllm::models::music3::VocoderBlockWeights;
using vllm::models::music3::VocoderResidualUnitWeights;
using vllm::models::music3::VocoderWeights;

// A plain LCG, identical on every box and every arm. Values are spread over a
// few decades and NEVER exactly zero, so the transposed op's zero-skip is off
// the path and both arms perform the same operation count.
struct Lcg {
  uint32_t s;
  float Next() {
    s = s * 1664525U + 1013904223U;
    const double mantissa = static_cast<double>(s >> 8) / 16777216.0 - 0.5;
    const int exponent = static_cast<int>((s >> 4) & 0x7U) - 6;
    double v = mantissa * std::ldexp(1.0, exponent);
    if (v == 0.0) v = 1.0 / 4096.0;
    return static_cast<float>(v);
  }
};

std::vector<float> Fill(Lcg& rng, size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = rng.Next();
  return v;
}

// The shipped geometry, read off the config rather than restated: decoder_hidden
// halves at each of the four upsampling ratios, exactly as `VocoderDecode` walks
// it (minimax_music3_acoustic.cpp::VocoderDecode).
VocoderWeights SyntheticWeights(const MiniMaxMusic3VocoderConfig& config, Lcg& rng) {
  VocoderWeights w;
  const int64_t stream = config.stream_channels();
  w.dec_in_proj_weight = Fill(rng, static_cast<size_t>(config.decoder_input_dim * stream));
  w.dec_in_proj_bias = Fill(rng, static_cast<size_t>(config.decoder_input_dim));
  w.conv_in_weight =
      Fill(rng, static_cast<size_t>(config.decoder_hidden_dim * config.decoder_input_dim * 7));
  w.conv_in_bias = Fill(rng, static_cast<size_t>(config.decoder_hidden_dim));
  int64_t last_output = config.decoder_hidden_dim;
  for (size_t index = 0; index < config.upsampling_ratios.size(); ++index) {
    const int64_t input_dim = config.decoder_hidden_dim >> index;
    const int64_t output_dim = config.decoder_hidden_dim >> (index + 1);
    const int64_t stride = config.upsampling_ratios[index];
    last_output = output_dim;
    VocoderBlockWeights block;
    block.snake1_alpha = Fill(rng, static_cast<size_t>(input_dim));
    block.conv_t1_weight = Fill(rng, static_cast<size_t>(input_dim * output_dim * 2 * stride));
    block.conv_t1_bias = Fill(rng, static_cast<size_t>(output_dim));
    for (int unit = 0; unit < vllm::models::music3::kVocoderResidualUnits; ++unit) {
      VocoderResidualUnitWeights res;
      res.snake1_alpha = Fill(rng, static_cast<size_t>(output_dim));
      res.conv1_weight = Fill(rng, static_cast<size_t>(output_dim * output_dim * 7));
      res.conv1_bias = Fill(rng, static_cast<size_t>(output_dim));
      res.snake2_alpha = Fill(rng, static_cast<size_t>(output_dim));
      res.conv2_weight = Fill(rng, static_cast<size_t>(output_dim * output_dim));
      res.conv2_bias = Fill(rng, static_cast<size_t>(output_dim));
      block.res_units.push_back(std::move(res));
    }
    w.blocks.push_back(std::move(block));
  }
  w.snake_out_alpha = Fill(rng, static_cast<size_t>(last_output));
  w.conv_out_weight = Fill(rng, static_cast<size_t>(last_output * 7));
  w.conv_out_bias = Fill(rng, 1);
  return w;
}

// A bitwise fingerprint of the waveform: FNV-1a over the raw float bytes. A sum
// would let two different waveforms agree; this cannot, and it is what lets the
// two arms be compared for BIT identity rather than for closeness.
uint64_t Fingerprint(const std::vector<float>& v) {
  uint64_t h = 1469598103934665603ULL;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
  for (size_t i = 0; i < v.size() * sizeof(float); ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int64_t> lengths;
  int repeats = 3;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--lengths=", 0) == 0) {
      std::string list = a.substr(10);
      size_t pos = 0;
      while (pos <= list.size()) {
        const size_t comma = list.find(',', pos);
        const std::string one = list.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (!one.empty()) lengths.push_back(std::strtoll(one.c_str(), nullptr, 10));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    } else if (a.rfind("--repeats=", 0) == 0) {
      repeats = static_cast<int>(std::strtol(a.substr(10).c_str(), nullptr, 10));
    } else {
      std::fprintf(stderr, "usage: %s [--lengths=20,40,...] [--repeats=N]\n", argv[0]);
      return 2;
    }
  }
  if (lengths.empty()) lengths = {20, 40, 86, 172, 344};

  const MiniMaxMusic3VocoderConfig config;
  Lcg rng{0x51F0C0DEu};
  const VocoderWeights weights = SyntheticWeights(config, rng);

  std::printf("# MiniMax-Music3 vocoder decode-window A/B (#1334)\n");
  std::printf("# latent_channels=%lld decoder_input_dim=%lld decoder_hidden_dim=%lld hop=%lld\n",
              static_cast<long long>(config.latent_channels),
              static_cast<long long>(config.decoder_input_dim),
              static_cast<long long>(config.decoder_hidden_dim),
              static_cast<long long>(config.hop_length()));
  std::printf("%10s %10s %12s %12s %20s\n", "latents", "repeats", "best_s", "s_per_latent",
              "fingerprint");
  for (const int64_t length : lengths) {
    Lcg lrng{0x1234ABCDu + static_cast<uint32_t>(length)};
    const std::vector<float> latents =
        Fill(lrng, static_cast<size_t>(config.latent_channels * length));
    double best = 1e30;
    uint64_t fp = 0;
    // ARMED ONLY BY THE ENVIRONMENT, and armed per length so the split belongs
    // to the leg it is printed under. With `VLLM_CPP_MUSIC3_PROFILE` unset every
    // bracket is one predicted branch and the timing below is byte for byte the
    // path spec §18.8a measured.
    vllm::models::music3::profile::Begin();
    for (int r = 0; r < repeats; ++r) {
      int64_t samples = 0;
      const auto t0 = std::chrono::steady_clock::now();
      const std::vector<float> wave =
          vllm::models::music3::VocoderDecode(latents, length, config, weights, &samples);
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (dt < best) best = dt;
      const uint64_t here = Fingerprint(wave);
      if (r == 0) {
        fp = here;
      } else if (here != fp) {
        std::fprintf(stderr, "FATAL: the decode is not deterministic within one arm at length %lld\n",
                     static_cast<long long>(length));
        return 3;
      }
    }
    std::printf("%10lld %10d %12.4f %12.6f %#20llx\n", static_cast<long long>(length), repeats,
                best, best / static_cast<double>(length),
                static_cast<unsigned long long>(fp));
    std::fflush(stdout);
    // The split of `vocoder.decode_window` into its leaves, over ALL the
    // repeats rather than the best one -- so it is a share, not a duration
    // comparable to `best_s` above, and the report says so by printing the
    // repeat count.
    {
      char banner[128];
      std::snprintf(banner, sizeof(banner), "vocoder decode, latents=%lld, %d repeats",
                    static_cast<long long>(length), repeats);
      vllm::models::music3::profile::Report(banner);
    }
  }
  return 0;
}
