// LTX-2.5 CONNECTOR GEMM PROBE — row LTX25-CONNECTOR-GEMM,
// .agents/specs/ltx25-connector-gemm.md.
//
// WHAT QUESTION THIS EXISTS TO ANSWER. `LTX25-TEXT-COND-DEVICE` (#2354) split
// `conditioning.connector` and found the render's largest single cost is not
// weight materialization but ARITHMETIC: 224.882 s of host f32 GEMM across the
// two guidance passes, 43.52% of an LTX-2.5 render, at about 37 GFLOP/s. Its
// `## Owed` names the first traceable step as "determine whether `MatmulBT`
// reaches `MatmulOneChunk`'s specialized elementwise kernel on these shapes or
// falls back to `MatmulOneChunkRef`". Reading `BuildTier()` cannot answer that,
// because the tier is resolved at RUNTIME from an environment variable and a
// CPU feature probe. This binary answers it by running.
//
// THREE MODES.
//   --mode tier       print the tier the process resolved, its `mr`, and which
//                     f32 entry points are non-null. The resolver's own output.
//   --mode gemm       vt::MatmulBT at the connector's real shapes, in the two
//                     orientations the tree can express, with a bit-identity
//                     assertion between them BEFORE any rate is printed.
//   --mode connector  Ltx2ConnectorForward itself, at the shipped geometry.
//
// THE SHAPES ARE READ OUT OF THE CHECKPOINT, not assumed. The `__metadata__`
// config of `ltx-2.5-22b-dev-transformer-bf16.safetensors` (the DiT pinned by
// `ltx2_oracle_manifest.json`) gives connector_num_attention_heads = 32,
// connector_attention_head_dim = 128, connector_num_layers = 8,
// audio_connector_attention_head_dim = 64, apply_gated_attention = true,
// num_learnable_registers = 128, rope_type = split, frequencies_precision =
// float64. So video inner_dim = 4096, audio inner_dim = 2048, rows = 1024
// (kLtx2GemmaTokenizerMaxLength), and one RunConnector call is
//   8 layers * 12 * (4096^2 + 2048^2) * 1024 = 2.0615e12 MAC = 4.123 TFLOP,
// which is the 4.2 TFLOP #2354 predicted before the number existed.
//
// RUN RECIPE — DIRECT COMPILE, because a full CMake build needs disk this
// devbox does not have (`/` at 99%; a bare ninja writes 9.4 GiB, and the ENOSPC
// that follows has previously made unrelated checkers emit FALSE policy
// refusals). The CMake target below exists so this file cannot rot behind a
// `vt::MatmulBT` or `Ltx2ConnectorForward` signature change -- the #1246
// failure, where the only named harness for a measurement was compiled by
// nothing -- but the recorded runs come from:
//
//   g++ -std=c++20 -O2 -ffp-contract=off -pthread -Iinclude -Isrc -Ithird_party
//       tools/bench/ltx2_connector_gemm_probe.cpp
//       src/vllm/model_executor/models/ltx2_connector.cpp
//       src/vllm/model_executor/models/ltx2.cpp
//       src/vllm/model_executor/models/ltx2_audio_vae.cpp
//       src/vt/{tensor,dtype,ops,backend,op_provider,arena}.cpp
//       src/vt/cpu/{cpu_ops,cpu_backend,cpu_threadpool,cpu_matmul_elem,...}.cpp
//       -o /tmp/ltx2_connector_gemm_probe
//
// `-ffp-contract=off` is NOT optional: CMakeLists.txt:55 pins it for every TU
// and the bit-identity contracts of the vt CPU kernels depend on it. The exact
// TU list this row compiled, and the perf recipe, are in the spec's `## Outcome`.
//
// CI COMPILES IT AND RUNS NOTHING, like its two siblings in this directory: one
// leg allocates gigabytes and spends tens of seconds of twenty cores.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

// The connector mode links the LTX-2.5 model TUs and their transitive audio-VAE
// closure. `-DLTX2_PROBE_NO_CONNECTOR` builds the tier and gemm modes alone,
// which need only the `vt` runtime -- the shape this row used on a devbox whose
// disk cannot hold that closure. The CMake target below defines nothing, so the
// build compiles the full file and the connector mode cannot rot.
#ifndef LTX2_PROBE_NO_CONNECTOR
#include "vllm/model_executor/models/ltx2_connector.h"
#endif
#include "vt/cpu/cpu_matmul_elem.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// A deterministic, reproducible filler. Not random in any statistical sense and
// it does not need to be: a GEMM's cost does not depend on its values, and
// making the stream deterministic is what lets the bit-identity assertion below
// mean something across processes.
struct Lcg {
  uint64_t s;
  float Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    // [-1, 1) from the top 24 bits, so every value is exactly representable.
    const uint32_t top = static_cast<uint32_t>(s >> 40);
    return static_cast<float>(static_cast<int32_t>(top) - (1 << 23)) /
           static_cast<float>(1 << 23);
  }
};

void Fill(std::vector<float>& v, uint64_t seed) {
  Lcg g{seed};
  for (float& x : v) x = g.Next();
}

// Median of a sample, plus the (max - min) / median spread the project reports
// beside every timing.
struct Stat {
  double median = 0.0;
  double spread = 0.0;
  double best = 0.0;
};

Stat Summarize(std::vector<double> xs) {
  Stat s;
  if (xs.empty()) return s;
  std::sort(xs.begin(), xs.end());
  s.median = xs[xs.size() / 2];
  s.best = xs.front();
  s.spread = s.median > 0.0 ? (xs.back() - xs.front()) / s.median : 0.0;
  return s;
}

// ── mode: tier ──────────────────────────────────────────────────────────────
//
// This is the W1 instrument. It prints what the process RESOLVED, which is the
// one thing `BuildTier()` cannot be read for: the tier comes from
// VT_CPU_MATMUL_TIER plus a CPU feature probe, so the same source resolves
// differently on two machines and on two invocations.
int ModeTier() {
  const vt::cpu::ElemGemmTierTable& t = vt::cpu::ElemGemmTier();
  const int f32 = static_cast<int>(vt::cpu::ElemKind::kF32);
  std::printf("tier_name=%s\n", vt::cpu::ElemGemmTierName());
  std::printf("elem_gemm_use_ref=%d\n", vt::cpu::ElemGemmUseRef() ? 1 : 0);
  std::printf("mr=%d\n", t.mr);
  std::printf("f32_bt=%s f32_nk=%s f32_btm=%s f32_nkm=%s\n",
              t.bt[f32] != nullptr ? "set" : "NULL",
              t.nk[f32] != nullptr ? "set" : "NULL",
              t.btm[f32] != nullptr ? "set" : "NULL",
              t.nkm[f32] != nullptr ? "set" : "NULL");
  vt::cpu::ElemKind kind{};
  std::printf("elem_kind_of_f32=%d\n", vt::cpu::ElemKindOf(vt::DType::kF32, &kind) ? 1 : 0);
  // The BT fast branch in MatmulBTKernel is gated on this flag alone, and the
  // connector's weights are plain `std::vector<float>` views, so it is false
  // there. Printing the eligibility beside it says the flag is UNSET rather
  // than unavailable.
  std::printf("repack_eligible_f32_4096x4096=%d\n",
              vt::cpu::ElemRepackEligible(vt::DType::kF32, 4096, 4096) ? 1 : 0);
  return 0;
}

// ── mode: gemm ──────────────────────────────────────────────────────────────

struct Shape {
  const char* what;
  int64_t n;
  int64_t k;
  int count;  // how many of this shape one RunConnector call performs
};

// One connector layer, per stream: to_q / to_k / to_v / to_out are [dim, dim],
// ff.net.0.proj is [4 dim, dim], ff.net.2 is [dim, 4 dim], and the gated
// attention adds a [heads, dim] logits projection. 8 layers each.
std::vector<Shape> ConnectorShapes() {
  return {
      {"video attn dim x dim", 4096, 4096, 4 * 8},
      {"video ff  4dim x dim", 16384, 4096, 1 * 8},
      {"video ff  dim x 4dim", 4096, 16384, 1 * 8},
      {"video gate heads x dim", 32, 4096, 1 * 8},
      {"audio attn dim x dim", 2048, 2048, 4 * 8},
      {"audio ff  4dim x dim", 8192, 2048, 1 * 8},
      {"audio ff  dim x 4dim", 2048, 8192, 1 * 8},
      {"audio gate heads x dim", 32, 2048, 1 * 8},
  };
}

int ModeGemm(int64_t m, int reps, bool do_repacked) {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::printf("# tier=%s use_ref=%d mr=%d m=%lld reps=%d\n", vt::cpu::ElemGemmTierName(),
              vt::cpu::ElemGemmUseRef() ? 1 : 0, vt::cpu::ElemGemmTier().mr,
              static_cast<long long>(m), reps);
  std::printf("# shape                     N      K   bt_s   bt_GF  kn_s   kn_GF  kn/bt  ident\n");
  double total_bt = 0.0, total_kn = 0.0, total_flop = 0.0;
  for (const Shape& s : ConnectorShapes()) {
    std::vector<float> a(static_cast<size_t>(m * s.k));
    std::vector<float> b(static_cast<size_t>(s.n * s.k));
    Fill(a, 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(s.k));
    Fill(b, 0xBF58476D1CE4E5B9ULL ^ static_cast<uint64_t>(s.n));
    std::vector<float> out_bt(static_cast<size_t>(m * s.n));
    std::vector<float> out_kn(static_cast<size_t>(m * s.n));

    vt::Tensor ta = vt::Tensor::Contiguous(a.data(), vt::DType::kF32,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {m, s.k});
    vt::Tensor tb = vt::Tensor::Contiguous(b.data(), vt::DType::kF32,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {s.n, s.k});
    vt::Tensor to = vt::Tensor::Contiguous(out_bt.data(), vt::DType::kF32,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {m, s.n});

    // THE ARMS ARE INTERLEAVED AND A SAME-ARM CONTROL RUNS BESIDE THEM.
    // A devbox shared with other agents drifts by tens of percent over the
    // seconds one shape takes, so "arm A for three reps, then arm B for three"
    // measures the drift as well as the arms. Interleaving puts both arms in
    // the same window, and the second BT leg is the control that says what the
    // same binary on the same data costs twice -- a ratio inside that control's
    // own spread is not a result (`1997-put-us-on-the-slow-topk-kernel`).
    std::vector<double> bt_times, kn_times, ctl_times;
    std::vector<float> b_kn;
    vt::Tensor tb_kn;
    vt::Tensor to_kn;
    if (do_repacked) {
      // The repack the loader performs for GGUF weights (`ElemRepackWeight`,
      // whose only production caller today is qwen3_5_gguf_weights.cpp). It
      // keeps the [N,K] SHAPE and rewrites the BYTES to [K,N]; `MatmulBTKernel`
      // then presents the tensor as the [K,N] it literally is and takes
      // MatmulChunked<false>.
      b_kn = b;
      vt::cpu::ElemRepackWeight(vt::DType::kF32, reinterpret_cast<uint8_t*>(b_kn.data()), s.n,
                                s.k);
      tb_kn = vt::Tensor::Contiguous(b_kn.data(), vt::DType::kF32,
                                     vt::Device{vt::DeviceType::kCPU, 0}, {s.n, s.k});
      tb_kn.elem_kn_repacked = true;
      to_kn = vt::Tensor::Contiguous(out_kn.data(), vt::DType::kF32,
                                     vt::Device{vt::DeviceType::kCPU, 0}, {m, s.n});
    }
    for (int r = 0; r < reps; ++r) {
      Clock::time_point t0 = Clock::now();
      vt::MatmulBT(q, to, ta, tb);
      bt_times.push_back(Seconds(t0, Clock::now()));
      if (do_repacked) {
        t0 = Clock::now();
        vt::MatmulBT(q, to_kn, ta, tb_kn);
        kn_times.push_back(Seconds(t0, Clock::now()));
      }
      t0 = Clock::now();
      vt::MatmulBT(q, to, ta, tb);
      ctl_times.push_back(Seconds(t0, Clock::now()));
    }
    const Stat bt = Summarize(bt_times);
    const Stat ctl = Summarize(ctl_times);
    Stat kn;
    int identical = -1;
    if (do_repacked) {
      kn = Summarize(kn_times);
      // BEFORE the rate is printed. A faster kernel that computed something
      // else is not a faster kernel, and byte equality is the right assertion
      // here rather than a tolerance: both orientations accumulate each output
      // over K in strict increasing order, so they have no room to differ.
      identical = std::memcmp(out_bt.data(), out_kn.data(),
                              out_bt.size() * sizeof(float)) == 0
                      ? 1
                      : 0;
    }

    const double flop = 2.0 * static_cast<double>(m) * static_cast<double>(s.n) *
                        static_cast<double>(s.k);
    std::printf("%-24s %6lld %6lld %6.3f %7.1f %6.3f %7.1f %6.2f %s\n", s.what,
                static_cast<long long>(s.n), static_cast<long long>(s.k), bt.median,
                flop / bt.median / 1e9, kn.median,
                kn.median > 0 ? flop / kn.median / 1e9 : 0.0,
                kn.median > 0 ? kn.median / bt.median : 0.0,
                identical < 0 ? "-" : (identical ? "byte-equal" : "DIFFER"));
    std::printf("#   spread bt=%.2f%% kn=%.2f%% control(bt#2)=%.3f s ctl/bt=%.3f "
                "count_per_call=%d\n",
                bt.spread * 100.0, kn.spread * 100.0, ctl.median,
                bt.median > 0 ? ctl.median / bt.median : 0.0, s.count);
    if (identical == 0) {
      std::fprintf(stderr, "FATAL: the repacked orientation is not byte-identical at %s\n",
                   s.what);
      return 2;
    }
    total_bt += bt.median * s.count;
    total_kn += kn.median * s.count;
    total_flop += flop * s.count;
  }
  std::printf("\n# one RunConnector call, GEMM only, extrapolated from the medians above\n");
  std::printf("total_gemm_flop=%.4fe12  bt_seconds=%.3f  bt_GFLOPs=%.1f\n", total_flop / 1e12,
              total_bt, total_flop / total_bt / 1e9);
  if (do_repacked) {
    std::printf("                         kn_seconds=%.3f  kn_GFLOPs=%.1f  kn/bt=%.3f\n",
                total_kn, total_flop / total_kn / 1e9, total_kn / total_bt);
  }
  return 0;
}

// ── mode: connector ─────────────────────────────────────────────────────────
#ifndef LTX2_PROBE_NO_CONNECTOR

vllm::Ltx2ConnectorConfig VideoConfig(int64_t layers) {
  vllm::Ltx2ConnectorConfig c;
  c.num_attention_heads = 32;
  c.attention_head_dim = 128;  // inner_dim 4096
  c.num_layers = layers;
  c.positional_embedding_max_pos = {4096};
  c.num_learnable_registers = 128;
  c.rope_type = vllm::Ltx2RopeType::kSplit;
  c.double_precision_rope = true;
  c.apply_gated_attention = true;
  c.ff_bias = true;
  c.prefix = "video.";
  return c;
}

vllm::Ltx2ConnectorConfig AudioConfig(int64_t layers) {
  vllm::Ltx2ConnectorConfig c = VideoConfig(layers);
  c.attention_head_dim = 64;  // inner_dim 2048
  c.prefix = "audio.";
  return c;
}

vllm::Ltx2VaeWeights SynthWeights(const vllm::Ltx2ConnectorConfig& cfg, uint64_t seed) {
  vllm::Ltx2VaeWeights w;
  for (const vllm::Ltx2ConnectorTensorSpec& spec : EnumerateLtx2ConnectorTensors(cfg)) {
    int64_t count = 1;
    for (const int64_t d : spec.shape) count *= d;
    std::vector<float> v(static_cast<size_t>(count));
    Fill(v, seed ^ std::hash<std::string>{}(spec.name));
    // Keep the RMSNorm gains near 1 so the residual stream neither vanishes nor
    // explodes over eight layers. The COST does not depend on the values; this
    // only keeps the run free of inf/NaN, which would be a distraction in a
    // perf profile.
    if (spec.shape.size() == 1) {
      for (float& x : v) x = 1.0f + 0.01f * x;
    } else {
      for (float& x : v) x *= 0.02f;
    }
    w.tensors.emplace(spec.name, std::move(v));
  }
  return w;
}

int ModeConnector(int64_t layers, int64_t rows, int64_t valid, int reps, bool video,
                  bool audio) {
  std::printf("# tier=%s use_ref=%d layers=%lld rows=%lld valid=%lld reps=%d\n",
              vt::cpu::ElemGemmTierName(), vt::cpu::ElemGemmUseRef() ? 1 : 0,
              static_cast<long long>(layers), static_cast<long long>(rows),
              static_cast<long long>(valid), reps);
  std::vector<float> mask(static_cast<size_t>(rows));
  for (int64_t i = 0; i < rows; ++i) mask[static_cast<size_t>(i)] = i < valid ? 0.0f : -3.4e38f;

  auto run = [&](const char* label, const vllm::Ltx2ConnectorConfig& cfg) {
    const int64_t dim = cfg.inner_dim();
    const vllm::Ltx2VaeWeights w = SynthWeights(cfg, 0x1234ULL);
    std::vector<float> x(static_cast<size_t>(rows * dim));
    Fill(x, 0xABCDULL);
    std::vector<double> times;
    for (int r = 0; r < reps; ++r) {
      const Clock::time_point t0 = Clock::now();
      const vllm::Ltx2ConnectorOutput out =
          vllm::Ltx2ConnectorForward(cfg, w, x.data(), mask.data(), 1, rows);
      const double s = Seconds(t0, Clock::now());
      times.push_back(s);
      // Consume the result so no part of the forward can be elided.
      std::printf("#   %s rep=%d %.3f s checksum=%.6f\n", label, r, s,
                  static_cast<double>(out.hidden_states[0] + out.hidden_states.back()));
    }
    const Stat st = Summarize(times);
    // 12 * dim^2 MAC per row per layer: to_q/to_k/to_v/to_out (4) plus the two
    // feed-forward projections (4 dim each). The gate logits are heads*dim and
    // are under 0.1% of that, so they are excluded from the denominator and
    // named rather than folded in.
    const double gemm_flop = 2.0 * 12.0 * static_cast<double>(dim) * static_cast<double>(dim) *
                             static_cast<double>(rows) * static_cast<double>(layers);
    std::printf("%s layers=%lld dim=%lld seconds=%.3f spread=%.2f%% gemm_flop=%.4e "
                "implied_GFLOPs=%.1f\n",
                label, static_cast<long long>(layers), static_cast<long long>(dim), st.median,
                st.spread * 100.0, gemm_flop, gemm_flop / st.median / 1e9);
  };
  if (video) run("video", VideoConfig(layers));
  if (audio) run("audio", AudioConfig(layers));
  return 0;
}

#endif  // LTX2_PROBE_NO_CONNECTOR

// ── mode: attn ──────────────────────────────────────────────────────────────
//
// `Ltx2ConnectorForward` passes an additive bias on every call
// (`args.bias = state.mask.data()`), so every connector block routes to
// `vt::AttentionCross`, never to the unbiased dense op. This mode times that op
// alone at the connector's own shape, and beside it a HOISTED-DTYPE reference
// written here in the probe -- not in product code -- so the headroom can be
// priced before anything is changed.
//
// The hoisted form is the SAME transformation `MatmulOneChunk` already applies
// against `MatmulOneChunkRef`: resolve the element type ONCE instead of per
// element. It keeps every output's accumulation strictly sequential over the
// same indices in the same order, so the probe asserts BYTE equality against
// the shipped kernel rather than a tolerance.
void AttnCrossHoisted(float* out, const float* q, const float* k, const float* v,
                      const float* bias, int64_t tq, int64_t hq, int64_t d, int64_t s,
                      float scale) {
  for (int64_t r = 0; r < hq * tq; ++r) {
    const int64_t h = r / tq;
    const int64_t i = r % tq;
    const int64_t qoff = (i * hq + h) * d;
    std::vector<float> probs(static_cast<size_t>(s));
    std::vector<float> acc(static_cast<size_t>(d), 0.0f);
    float m = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < s; ++j) {
      const float* kr = k + (j * hq + h) * d;
      const float* qr = q + qoff;
      float dot = 0.0f;
      for (int64_t e = 0; e < d; ++e) dot += qr[e] * kr[e];
      dot *= scale;
      if (bias != nullptr) dot += bias[j];
      probs[static_cast<size_t>(j)] = dot;
      if (dot > m) m = dot;
    }
    float denom = 0.0f;
    for (int64_t j = 0; j < s; ++j) {
      const float e = std::exp(probs[static_cast<size_t>(j)] - m);
      probs[static_cast<size_t>(j)] = e;
      denom += e;
    }
    const float inv = 1.0f / denom;
    for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
    for (int64_t j = 0; j < s; ++j) {
      const float p = probs[static_cast<size_t>(j)] * inv;
      const float* vr = v + (j * hq + h) * d;
      for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += p * vr[e];
    }
    for (int64_t e = 0; e < d; ++e) out[qoff + e] = acc[static_cast<size_t>(e)];
  }
}

// The same hoisted kernel over `threads` workers, partitioned on the (head,
// query) pairs exactly as `AttentionCrossKernel`'s own `ForRows` partitions
// them. Each output row is independent and its reduction stays sequential over
// the same indices, so this is bit-exact for the same reason the single-threaded
// form is -- and it is what says how much of the gap is the dtype switch rather
// than the thread count. Written with `std::thread` rather than the vt pool
// because the probe must not change what the shipped kernel's pool is doing
// while it is being measured beside it.
void AttnCrossHoistedMt(float* out, const float* q, const float* k, const float* v,
                        const float* bias, int64_t tq, int64_t hq, int64_t d, int64_t s,
                        float scale, int threads) {
  const int64_t rows = hq * tq;
  const int64_t per = (rows + threads - 1) / threads;
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    const int64_t r0 = std::min(static_cast<int64_t>(t) * per, rows);
    const int64_t r1 = std::min(r0 + per, rows);
    if (r0 >= r1) break;
    pool.emplace_back([=] {
      std::vector<float> probs(static_cast<size_t>(s));
      std::vector<float> acc(static_cast<size_t>(d));
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t h = r / tq;
        const int64_t i = r % tq;
        const int64_t qoff = (i * hq + h) * d;
        const float* qr = q + qoff;
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < s; ++j) {
          const float* kr = k + (j * hq + h) * d;
          float dot = 0.0f;
          for (int64_t e = 0; e < d; ++e) dot += qr[e] * kr[e];
          dot *= scale;
          if (bias != nullptr) dot += bias[j];
          probs[static_cast<size_t>(j)] = dot;
          if (dot > m) m = dot;
        }
        float denom = 0.0f;
        for (int64_t j = 0; j < s; ++j) {
          const float e = std::exp(probs[static_cast<size_t>(j)] - m);
          probs[static_cast<size_t>(j)] = e;
          denom += e;
        }
        const float inv = 1.0f / denom;
        for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
        for (int64_t j = 0; j < s; ++j) {
          const float pw = probs[static_cast<size_t>(j)] * inv;
          const float* vr = v + (j * hq + h) * d;
          for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += pw * vr[e];
        }
        for (int64_t e = 0; e < d; ++e) out[qoff + e] = acc[static_cast<size_t>(e)];
      }
    });
  }
  for (std::thread& t : pool) t.join();
}

int ModeAttn(int64_t tq, int reps) {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vt::Device dev{vt::DeviceType::kCPU, 0};
  std::printf("# tier=%s tq=%lld reps=%d\n", vt::cpu::ElemGemmTierName(),
              static_cast<long long>(tq), reps);
  struct Case { const char* what; int64_t heads; int64_t d; };
  for (const Case& c : std::vector<Case>{{"video", 32, 128}, {"audio", 32, 64}}) {
    const int64_t s = tq;
    std::vector<float> qb(static_cast<size_t>(tq * c.heads * c.d));
    std::vector<float> kb(static_cast<size_t>(s * c.heads * c.d));
    std::vector<float> vb(static_cast<size_t>(s * c.heads * c.d));
    std::vector<float> ob(static_cast<size_t>(tq * c.heads * c.d));
    std::vector<float> ob2(ob.size());
    // The connector's own bias after the register substitution: all zeros, one
    // broadcast row (`Ltx2ConnectorReplaceRegisters` assigns zeros, and
    // `Ltx2ConnectorForward` passes bias_rows = 1).
    std::vector<float> bias(static_cast<size_t>(s), 0.0f);
    Fill(qb, 11); Fill(kb, 22); Fill(vb, 33);
    vt::Tensor tq_t = vt::Tensor::Contiguous(qb.data(), vt::DType::kF32, dev,
                                             {tq, c.heads, c.d});
    vt::Tensor tk_t = vt::Tensor::Contiguous(kb.data(), vt::DType::kF32, dev,
                                             {s, c.heads, c.d});
    vt::Tensor tv_t = vt::Tensor::Contiguous(vb.data(), vt::DType::kF32, dev,
                                             {s, c.heads, c.d});
    vt::Tensor to_t = vt::Tensor::Contiguous(ob.data(), vt::DType::kF32, dev,
                                             {tq, c.heads, c.d});
    vt::Tensor tbias = vt::Tensor::Contiguous(bias.data(), vt::DType::kF32, dev, {1, s});
    vt::AttentionCrossArgs a;
    a.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(c.d)));
    std::vector<double> shipped, hoisted;
    for (int r = 0; r < reps; ++r) {
      Clock::time_point t0 = Clock::now();
      vt::AttentionCross(q, to_t, tq_t, tk_t, tv_t, &tbias, a);
      shipped.push_back(Seconds(t0, Clock::now()));
      t0 = Clock::now();
      AttnCrossHoisted(ob2.data(), qb.data(), kb.data(), vb.data(), bias.data(), tq, c.heads,
                       c.d, s, a.scale);
      hoisted.push_back(Seconds(t0, Clock::now()));
    }
    // THE UNBIASED OP AT THE SAME SHAPE, because "route the all-zero bias to
    // `vt::Attention` instead" is the lever a reader reaches for next and it is
    // not one. After `Ltx2ConnectorReplaceRegisters` the additive mask IS all
    // zeros, so the bias adds nothing -- but `AttentionKernel` reads its
    // operands through the SAME per-element `LoadF32`, so the two cost the same
    // and the routing change would buy nothing. Measured rather than argued,
    // and measured here so the next row does not spend a day on it. (Routing on
    // the VALUES would also be the thing `ltx2.cpp` explicitly refuses: "Route
    // on what the call MEANS, never on what its numbers happen to be.")
    std::vector<double> unbiased;
    {
      std::vector<float> ob3(ob.size());
      vt::Tensor to3 = vt::Tensor::Contiguous(ob3.data(), vt::DType::kF32, dev,
                                              {tq, c.heads, c.d});
      vt::AttentionArgs ua;
      ua.scale = a.scale;
      ua.causal = false;
      for (int r = 0; r < reps; ++r) {
        const Clock::time_point t0 = Clock::now();
        vt::Attention(q, to3, tq_t, tk_t, tv_t, ua);
        unbiased.push_back(Seconds(t0, Clock::now()));
      }
    }
    // The threaded hoisted arm: the SAME transformation at the SAME thread count
    // the shipped kernel uses, which is the number the next row needs.
    std::vector<double> hoisted_mt;
    std::vector<float> ob4(ob.size());
    const int threads = static_cast<int>(std::thread::hardware_concurrency());
    for (int r = 0; r < reps; ++r) {
      const Clock::time_point t0 = Clock::now();
      AttnCrossHoistedMt(ob4.data(), qb.data(), kb.data(), vb.data(), bias.data(), tq, c.heads,
                         c.d, s, a.scale, threads);
      hoisted_mt.push_back(Seconds(t0, Clock::now()));
    }
    const Stat hm = Summarize(hoisted_mt);
    const bool eq_mt = std::memcmp(ob.data(), ob4.data(), ob.size() * sizeof(float)) == 0;
    const Stat un = Summarize(unbiased);
    const Stat sh = Summarize(shipped);
    const Stat ho = Summarize(hoisted);
    const bool eq = std::memcmp(ob.data(), ob2.data(), ob.size() * sizeof(float)) == 0;
    // 2 passes over the score matrix: QK^T and the value-weighted sum.
    const double flop = 2.0 * 2.0 * static_cast<double>(c.heads) * static_cast<double>(tq) *
                        static_cast<double>(s) * static_cast<double>(c.d);
    std::printf("%-6s heads=%lld d=%lld  shipped=%.3f s (%.1f GF, spread %.2f%%)  "
                "hoisted_1thread=%.3f s (%.1f GF)  %s\n",
                c.what, static_cast<long long>(c.heads), static_cast<long long>(c.d), sh.median,
                flop / sh.median / 1e9, sh.spread * 100.0, ho.median, flop / ho.median / 1e9,
                eq ? "byte-equal" : "DIFFER");
    std::printf("#   unbiased vt::Attention at the same shape = %.3f s (%.1f GF), "
                "unbiased/shipped = %.3f\n",
                un.median, flop / un.median / 1e9,
                sh.median > 0 ? un.median / sh.median : 0.0);
    std::printf("#   hoisted on %d threads = %.3f s (%.1f GF), speedup vs shipped = %.2fx, %s\n",
                threads, hm.median, flop / hm.median / 1e9,
                hm.median > 0 ? sh.median / hm.median : 0.0,
                eq_mt ? "byte-equal" : "DIFFER");
    if (!eq_mt) {
      std::fprintf(stderr, "FATAL: the threaded hoisted arm is not byte-identical (%s)\n",
                   c.what);
      return 2;
    }
    if (!eq) {
      std::fprintf(stderr, "FATAL: the hoisted reference is not byte-identical (%s)\n", c.what);
      return 2;
    }
  }
  return 0;
}

const char* ArgValue(int argc, char** argv, const char* flag, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  }
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = ArgValue(argc, argv, "--mode", "tier");
  const int reps = std::atoi(ArgValue(argc, argv, "--reps", "3"));
  const int64_t rows = std::atoll(ArgValue(argc, argv, "--rows", "1024"));
  if (mode == "tier") return ModeTier();
  if (mode == "attn") return ModeAttn(rows, reps);
  if (mode == "gemm") {
    const bool repacked = std::strcmp(ArgValue(argc, argv, "--repacked", "1"), "1") == 0;
    return ModeGemm(rows, reps, repacked);
  }
  if (mode == "connector") {
#ifdef LTX2_PROBE_NO_CONNECTOR
    std::fprintf(stderr, "built with -DLTX2_PROBE_NO_CONNECTOR; the connector mode is absent\n");
    return 3;
#else
    const int64_t layers = std::atoll(ArgValue(argc, argv, "--layers", "2"));
    const int64_t valid = std::atoll(ArgValue(argc, argv, "--valid", "180"));
    const std::string stream = ArgValue(argc, argv, "--stream", "both");
    return ModeConnector(layers, rows, valid, reps, stream != "audio", stream != "video");
#endif
  }
  std::fprintf(stderr, "usage: %s --mode tier|gemm|connector [--reps N] [--rows N]\n", argv[0]);
  return 1;
}
