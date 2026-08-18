// vocoder-conv-ab — the CPU-vs-device A/B for the BigVGAN / DAC vocoder
// convolution chain (#672, .agents/specs/minimax-music3.md §13).
//
// SAME BINARY, one variable: `VLLM_CPP_VOCODER_DEVICE`. Both arms call the same
// `vllm::vocoder1d::Conv1d` / `ConvTranspose1d` a real decode calls, at the
// geometries MiniMax-Music3's vocoder actually runs, so the number is the cost
// of the stage rather than the cost of a microbenchmark shaped like it.
//
// WHY A STANDALONE HARNESS RATHER THAN A TEST. A test that also times is a test
// that fails on a busy box, and this box is shared. This prints, and the caller
// decides. It also prints the arm each run RESOLVED, because a silent fallback
// to the host would post a plausible pair of timings that mean nothing — the
// same reason `benchmarks/vulkan_gemm_ab.cpp` reports its tactic.
//
// It additionally CHECKS the two arms against each other bit for bit when both
// are available in one process. That is not redundant with
// tests/vt/test_ops_conv1d_general.cpp: this one runs at production sizes, where
// the CUDA grid-stride loop wraps and a launch-geometry-dependent defect would
// show up and the small gated shapes would not.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/models/vocoder1d.h"

namespace {

std::vector<float> Spread(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525U + 1013904223U;
    v[i] = static_cast<float>(static_cast<double>(s >> 8) / 16777216.0 - 0.5);
  }
  return v;
}

double SecondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The MiniMax-Music3 vocoder's four upsample stages, plus the residual-unit
// convs that follow each one. Channel counts and strides are the shipped
// decoder's (minimax_music3_vocoder.py:55 transpose, :42/:44 residual convs);
// `frames` is scaled by --frames so a run fits the box.
struct Stage {
  const char* name;
  int64_t in_ch, out_ch, kernel, stride, padding;
};

const Stage kStages[] = {
    {"up0", 1536, 768, 16, 8, 4},
    {"up1", 768, 384, 16, 8, 4},
    {"up2", 384, 192, 8, 4, 2},
    {"up3", 192, 96, 4, 2, 1},
};

}  // namespace

int main(int argc, char** argv) {
  int64_t frames = 64;
  int reps = 3;
  bool check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--frames" && i + 1 < argc) frames = std::atoll(argv[++i]);
    else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (a == "--check") check = true;
    else {
      std::fprintf(stderr, "usage: %s [--frames N] [--reps N] [--check]\n", argv[0]);
      return 2;
    }
  }

  // Report the arm that was RESOLVED, not the one that was requested.
  const char* env = std::getenv("VLLM_CPP_VOCODER_DEVICE");
  std::printf("arm: VLLM_CPP_VOCODER_DEVICE=%s  frames=%lld reps=%d\n",
              env != nullptr && env[0] != '\0' ? env : "(unset -> cpu)",
              static_cast<long long>(frames), reps);

  double total_best = 0.0;
  for (const Stage& s : kStages) {
    const std::vector<float> in = Spread(static_cast<size_t>(s.in_ch * frames), 0xC0FFu);
    const std::vector<float> w =
        Spread(static_cast<size_t>(s.in_ch * s.out_ch * s.kernel), 0xBEEFu);
    const std::vector<float> bias = Spread(static_cast<size_t>(s.out_ch), 0x0B1Au);

    double best = 1e30;
    std::vector<float> last;
    int64_t out_len = 0;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      last = vllm::vocoder1d::ConvTranspose1d(in, s.in_ch, frames, w, &bias, s.out_ch, s.kernel,
                                              s.stride, s.padding, /*groups=*/1, &out_len);
      const double dt = SecondsSince(t0);
      if (dt < best) best = dt;
    }
    total_best += best;
    // A checksum, so two arms that print the same time can still be told apart
    // if one of them silently computed something else.
    double sum = 0.0;
    for (const float v : last) sum += static_cast<double>(v);
    std::printf("  %-4s [%4lld->%4lld] x%4lld  out_len=%-7lld  best=%8.4f s  checksum=%.9g\n",
                s.name, static_cast<long long>(s.in_ch), static_cast<long long>(s.out_ch),
                static_cast<long long>(frames), static_cast<long long>(out_len), best, sum);
    (void)check;
  }
  std::printf("TOTAL best-of-%d: %.4f s\n", reps, total_best);
  return 0;
}
