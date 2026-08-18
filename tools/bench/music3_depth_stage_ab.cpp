// MiniMax-Music3 depth-stage A/B driver (#672) — the source behind
// `.agents/specs/minimax-music3.md` §16.6 and benchmark-record's
// MUSIC3-DEPTH-INCREMENTAL section.
//
// DELIBERATELY NOT A REGISTERED TEST. It allocates 2.5 GB of weights and spends
// seconds per round, so CI must never RUN it; and it is a TWO-BUILD A/B, so one
// target could not express it anyway. CI does COMPILE it, both arms, as the
// OBJECT libraries `vllm_music3_depth_stage_ab_{before,after}` in CMakeLists.txt
// — nothing is linked and no weight is ever allocated, and the file cannot rot
// behind a `LinearNoBias` signature change while still being the only artifact a
// reader can reproduce §16.6's 3.50x from (#1246).
//
// To RUN it, compile it twice and link each against the `libvllm.a` you want to
// compare. The recipe is in a block comment because a `//` line may not end in a
// backslash under `-Werror=comment`, which is a rule this file is now inside.
/*
    # the arm WITHOUT the incremental schedule (any commit before it)
    g++ -O3 -std=c++20 -ffp-contract=off -I<wt>/include -I<wt>/src \
        -I<wt>/build/include -isystem <wt>/third_party \
        tools/bench/music3_depth_stage_ab.cpp -o depthbench_before \
        <wt>/build/libvllm.a <wt>/build/libblake3_vendored.a -lpthread

    # the arm WITH it — the same source, plus -DMUSIC3_AFTER
    g++ ... -DMUSIC3_AFTER ... -o depthbench_after ...

    ./depthbench_before 4 ; ./depthbench_after 4      # alternate, take the MIN
*/
//
// The argument is the number of rounds per process. Take the MINIMUM over rounds
// and alternate the arms: a shared box makes an average a statement about
// somebody else's scheduler (.agents/benchmarking.md).
//
// It prints an FNV-1a fingerprint of the frame's depth hidden states. The two
// arms MUST print the same one — the change it prices is bit-identical, and a
// speedup whose fingerprint moved is a different computation, not a faster one.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_music3_ar.h"

namespace m3 = vllm::models::music3;
using m3::ArCompute;

namespace {

uint32_t g_state = 0x9E3779B9u;
float Draw() {
  g_state = g_state * 1664525u + 1013904223u;
  return static_cast<float>(static_cast<double>(g_state >> 8) / 8388608.0 - 1.0) * 0.02f;
}
std::vector<float> Fill(size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = Draw();
  return v;
}

// FNV-1a over the raw output bytes: the two arms must agree, and a fingerprint
// says so without shipping a golden.
uint64_t Fnv(const std::vector<float>& v) {
  uint64_t h = 1469598103934665603ull;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
  for (size_t i = 0; i < v.size() * sizeof(float); ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? std::atoi(argv[1]) : 5;

  m3::DepthDecoderConfig config;  // the REAL geometry: 4096 / 4 / 16 / 6144 / 8
  const int64_t H = config.hidden_size;
  const int64_t I = config.intermediate_size;

  m3::DepthDecoderWeights w;
  w.audio_embeddings = Fill(static_cast<size_t>(config.audio_vocab_size *
                                                config.residual_codebooks()) * H);
  w.projection = Fill(static_cast<size_t>(H * H));
  w.pos_embedding = Fill(static_cast<size_t>(config.max_position_embeddings * H));
  w.norm = Fill(static_cast<size_t>(H));
  for (int64_t l = 0; l < config.num_layers; ++l) {
    m3::DepthDecoderLayerWeights layer;
    layer.input_layernorm = Fill(static_cast<size_t>(H));
    layer.post_attention_layernorm = Fill(static_cast<size_t>(H));
    layer.to_q = Fill(static_cast<size_t>(H * H));
    layer.to_k = Fill(static_cast<size_t>(H * H));
    layer.to_v = Fill(static_cast<size_t>(H * H));
    layer.to_out = Fill(static_cast<size_t>(H * H));
    layer.gate_proj = Fill(static_cast<size_t>(I * H));
    layer.up_proj = Fill(static_cast<size_t>(I * H));
    layer.down_proj = Fill(static_cast<size_t>(H * I));
    w.layers.push_back(std::move(layer));
  }
  for (int64_t h = 0; h < config.residual_codebooks(); ++h) {
    w.audio_heads.push_back(Fill(static_cast<size_t>(config.audio_vocab_size * H)));
  }

  const std::vector<float> last_cond = Fill(static_cast<size_t>(H));
  const std::vector<float> last_uncond = Fill(static_cast<size_t>(H));
  const std::vector<float> semantic = Fill(static_cast<size_t>(H));
  // A FIXED code per step, so the two arms traverse identical rows.
  const int32_t codes[7] = {11, 222, 333, 444, 555, 666, 777};

  double best = 1e300;
  uint64_t fingerprint = 0;
  for (int r = 0; r < rounds; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> depth_hidden;
    depth_hidden.reserve(static_cast<size_t>(config.residual_codebooks() * H));

#if defined(MUSIC3_AFTER)
    m3::DepthDecoderCache cache;
    std::vector<float> prefix_rows;
    prefix_rows.reserve(static_cast<size_t>(3 * H));
    prefix_rows.insert(prefix_rows.end(), last_cond.begin(), last_cond.end());
    prefix_rows.insert(prefix_rows.end(), last_uncond.begin(), last_uncond.end());
    prefix_rows.insert(prefix_rows.end(), semantic.begin(), semantic.end());
    const std::vector<float> prefix =
        m3::LinearNoBias(prefix_rows, 3, H, w.projection, H, ArCompute::kBFloat16);
    m3::DepthDecoderAppend(std::vector<float>(prefix.begin(), prefix.begin() + 2 * H), 2,
                           config, w, ArCompute::kBFloat16, &cache);
    std::vector<float> next(static_cast<size_t>(2 * H));
    std::copy(prefix.begin() + 2 * H, prefix.begin() + 3 * H, next.begin());
    std::copy(prefix.begin() + 2 * H, prefix.begin() + 3 * H, next.begin() + H);
    for (int64_t index = 1; index < config.num_codebooks; ++index) {
      const std::vector<float> states =
          m3::DepthDecoderAppend(next, 2, config, w, ArCompute::kBFloat16, &cache);
      depth_hidden.insert(depth_hidden.end(), states.begin(), states.begin() + H);
      if (index < config.num_codebooks - 1) {
        const int64_t row = (index - 1) * config.audio_vocab_size + codes[index - 1];
        const size_t at = static_cast<size_t>(row * H);
        const std::vector<float> embed(w.audio_embeddings.begin() + at,
                                       w.audio_embeddings.begin() + at + H);
        const std::vector<float> projected =
            m3::LinearNoBias(embed, 1, H, w.projection, H, ArCompute::kBFloat16);
        std::copy(projected.begin(), projected.end(), next.begin());
        std::copy(projected.begin(), projected.end(), next.begin() + H);
      }
    }
#else
    std::vector<int32_t> fed_back;
    for (int64_t index = 1; index < config.num_codebooks; ++index) {
      const int64_t seq_len = index + 1;
      std::vector<float> rows0;
      for (int row = 0; row < 2; ++row) {
        const std::vector<float>& last = row == 0 ? last_cond : last_uncond;
        const std::vector<float> embeds =
            m3::DepthSequenceEmbeds(last, semantic, fed_back, config, w, ArCompute::kBFloat16);
        const std::vector<float> states =
            m3::DepthDecoderForward(embeds, seq_len, config, w, ArCompute::kBFloat16);
        if (row == 0) rows0.assign(states.end() - H, states.end());
      }
      depth_hidden.insert(depth_hidden.end(), rows0.begin(), rows0.end());
      if (index < config.num_codebooks - 1) fed_back.push_back(codes[index - 1]);
    }
#endif
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fingerprint = Fnv(depth_hidden);
    std::printf("  round %d: %.4f s\n", r, seconds);
    if (seconds < best) best = seconds;
  }
  std::printf("%s best=%.4f s  fnv=%016llx  values=%lld\n",
#if defined(MUSIC3_AFTER)
              "AFTER ",
#else
              "BEFORE",
#endif
              best, static_cast<unsigned long long>(fingerprint),
              static_cast<long long>(config.residual_codebooks() * H));
  return 0;
}
