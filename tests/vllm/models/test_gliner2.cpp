// GLiNER2 boundary head parity gate — GLiNER2.5-multi-v1 NER head (MODEL-GLINER25).
//
// Compared against a restatement of the GLiNER2 library boundary architecture
// (github.com/fastino-ai/GLiNER2, boundary_head package), executed at reduced
// dimensions by scripts/gen-gliner2-goldens.py. Both sides rebuild every
// weight and input from ONE deterministic FNV-1a -> splitmix64 stream, so no
// weight byte is checked in.
//
// The sliding window mask in the boundary attention is the feature this
// architecture gets wrong quietly: without it the model still runs and emits
// plausible boundary states. A perturbation test widens the window to full
// attention and verifies the output changes, rather than leaving the mask to
// be implied by the forward.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "gliner2_goldens.inc"
#include "vllm/model_executor/models/deberta_v2.h"
#include "vllm/model_executor/models/gliner2.h"

namespace {

// The generator's stream, byte-for-byte: values uniform in [-1, 1) derived
// from the tensor NAME alone. Deliberately a local copy, same as
// test_deberta_v2.cpp.
std::vector<float> Rand(const std::string& name, int64_t count, double scale) {
  uint64_t seed = 0xCBF29CE484222325ULL;
  for (const char c : name) {
    seed ^= static_cast<unsigned char>(c);
    seed *= 0x100000001B3ULL;
  }
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    uint64_t x = seed + static_cast<uint64_t>(i);
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double u = static_cast<double>(z >> 11) * 0x1.0p-53;
    out[static_cast<size_t>(i)] = static_cast<float>((u * 2.0 - 1.0) * scale);
  }
  return out;
}

std::vector<float> RandPlusOne(const std::string& name, int64_t count,
                                double scale) {
  std::vector<float> v = Rand(name, count, scale);
  for (float& x : v) x += 1.0F;
  return v;
}

vllm::gliner2::BoundaryParams GoldenParams() {
  vllm::gliner2::BoundaryParams p;
  p.hidden_size = gliner2_goldens::kHidden;
  p.boundary_dim = gliner2_goldens::kBoundaryDim;
  p.boundary_attention_heads = gliner2_goldens::kHeads;
  p.boundary_attention_layers = gliner2_goldens::kAttnLayers;
  p.boundary_attention_window = gliner2_goldens::kAttnWindow;
  p.boundary_refinement_layers = gliner2_goldens::kRefineLayers;
  p.boundary_ffn_multiplier = 2.0;
  p.layer_norm_eps = gliner2_goldens::kLayerNormEps;
  return p;
}

vllm::deberta_v2::CheckpointTensors GoldenCheckpoint(
    const vllm::gliner2::BoundaryParams& p) {
  vllm::deberta_v2::CheckpointTensors t;
  const int64_t H = p.hidden_size;
  const int64_t d = p.boundary_dim;
  const int64_t ffn = p.ffn_dim();
  const int64_t half_ffn = ffn / 2;

  const std::string e = "boundary_head.boundary_encoder.";
  t.Set(e + "left_projection.weight", {d, H},
        Rand(e + "left_projection.weight", d * H, 0.3));
  t.Set(e + "left_projection.bias", {d},
        Rand(e + "left_projection.bias", d, 0.2));
  t.Set(e + "right_projection.weight", {d, H},
        Rand(e + "right_projection.weight", d * H, 0.3));
  t.Set(e + "right_projection.bias", {d},
        Rand(e + "right_projection.bias", d, 0.2));
  t.Set(e + "output_projection.weight", {d, 2 * d},
        Rand(e + "output_projection.weight", d * 2 * d, 0.3));
  t.Set(e + "output_projection.bias", {d},
        Rand(e + "output_projection.bias", d, 0.2));
  t.Set(e + "layer_norm.weight", {d},
        RandPlusOne(e + "layer_norm.weight", d, 0.1));
  t.Set(e + "layer_norm.bias", {d},
        Rand(e + "layer_norm.bias", d, 0.1));
  t.Set(e + "bos_state", {H}, Rand(e + "bos_state", H, 0.3));
  t.Set(e + "eos_state", {H}, Rand(e + "eos_state", H, 0.3));

  for (int64_t i = 0; i < p.boundary_attention_layers; ++i) {
    const std::string b = e + "attention_blocks." + std::to_string(i) + ".";
    t.Set(b + "norm.weight", {d}, RandPlusOne(b + "norm.weight", d, 0.1));
    t.Set(b + "norm.bias", {d}, Rand(b + "norm.bias", d, 0.1));
    t.Set(b + "qkv_projection.weight", {3 * d, d},
          Rand(b + "qkv_projection.weight", 3 * d * d, 0.3));
    t.Set(b + "qkv_projection.bias", {3 * d},
          Rand(b + "qkv_projection.bias", 3 * d, 0.2));
    t.Set(b + "output_projection.weight", {d, d},
          Rand(b + "output_projection.weight", d * d, 0.3));
    t.Set(b + "output_projection.bias", {d},
          Rand(b + "output_projection.bias", d, 0.2));
  }

  for (int64_t i = 0; i < p.boundary_refinement_layers; ++i) {
    const std::string b = e + "refinement_blocks." + std::to_string(i) + ".";
    t.Set(b + "norm.weight", {d}, RandPlusOne(b + "norm.weight", d, 0.1));
    t.Set(b + "norm.bias", {d}, Rand(b + "norm.bias", d, 0.1));
    t.Set(b + "input_projection.weight", {ffn, d},
          Rand(b + "input_projection.weight", ffn * d, 0.3));
    t.Set(b + "input_projection.bias", {ffn},
          Rand(b + "input_projection.bias", ffn, 0.2));
    t.Set(b + "output_projection.weight", {d, half_ffn},
          Rand(b + "output_projection.weight", d * half_ffn, 0.3));
    t.Set(b + "output_projection.bias", {d},
          Rand(b + "output_projection.bias", d, 0.2));
  }

  const std::string q = "boundary_head.boundary_query_head.";
  t.Set(q + "start_boundary_projection.weight", {d, d},
        Rand(q + "start_boundary_projection.weight", d * d, 0.3));
  t.Set(q + "start_boundary_projection.bias", {d},
        Rand(q + "start_boundary_projection.bias", d, 0.2));
  t.Set(q + "end_boundary_projection.weight", {d, d},
        Rand(q + "end_boundary_projection.weight", d * d, 0.3));
  t.Set(q + "end_boundary_projection.bias", {d},
        Rand(q + "end_boundary_projection.bias", d, 0.2));
  t.Set(q + "start_query_projection.weight", {d, H},
        Rand(q + "start_query_projection.weight", d * H, 0.3));
  t.Set(q + "start_query_projection.bias", {d},
        Rand(q + "start_query_projection.bias", d, 0.2));
  t.Set(q + "end_query_projection.weight", {d, H},
        Rand(q + "end_query_projection.weight", d * H, 0.3));
  t.Set(q + "end_query_projection.bias", {d},
        Rand(q + "end_query_projection.bias", d, 0.2));
  t.Set(q + "inside_text_projection.weight", {d, H},
        Rand(q + "inside_text_projection.weight", d * H, 0.3));
  t.Set(q + "inside_text_projection.bias", {d},
        Rand(q + "inside_text_projection.bias", d, 0.2));
  t.Set(q + "inside_query_projection.weight", {d, H},
        Rand(q + "inside_query_projection.weight", d * H, 0.3));
  t.Set(q + "inside_query_projection.bias", {d},
        Rand(q + "inside_query_projection.bias", d, 0.2));

  return t;
}

std::vector<float> GoldenTextStates() {
  return Rand("input.text_states",
              gliner2_goldens::kSeqLen * gliner2_goldens::kHidden, 0.5);
}

std::vector<float> GoldenQueryStates() {
  return Rand("input.query_states",
              gliner2_goldens::kNumQueries * gliner2_goldens::kHidden, 0.5);
}

double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) -
                                       static_cast<double>(want[i])));
  }
  return worst;
}

}  // namespace

TEST_CASE("gliner2 boundary encoder reproduces the upstream boundary states") {
  const vllm::gliner2::BoundaryParams p = GoldenParams();
  const vllm::gliner2::BoundaryHeadWeights hw =
      vllm::gliner2::LoadBoundaryHead(p, GoldenCheckpoint(p));
  const std::vector<float> text = GoldenTextStates();

  const std::vector<float> bs = vllm::gliner2::BoundaryEncoderForward(
      p, hw.encoder, text, gliner2_goldens::kSeqLen);

  const size_t n = static_cast<size_t>(gliner2_goldens::kB * p.boundary_dim);
  REQUIRE(bs.size() == n);
  const double worst = MaxAbsDiff(bs, gliner2_goldens::kBoundaryStates, n);
  INFO("max abs diff vs upstream boundary states: ", worst);
  CHECK(worst < 2e-5);
}

TEST_CASE("gliner2 boundary query head reproduces upstream marginals") {
  const vllm::gliner2::BoundaryParams p = GoldenParams();
  const vllm::gliner2::BoundaryHeadWeights hw =
      vllm::gliner2::LoadBoundaryHead(p, GoldenCheckpoint(p));
  const std::vector<float> text = GoldenTextStates();
  const std::vector<float> queries = GoldenQueryStates();

  const std::vector<float> bs = vllm::gliner2::BoundaryEncoderForward(
      p, hw.encoder, text, gliner2_goldens::kSeqLen);

  const vllm::gliner2::BoundaryMarginals m = vllm::gliner2::BoundaryQueryHeadForward(
      p, hw.query_head, bs, gliner2_goldens::kSeqLen, text, queries,
      gliner2_goldens::kNumQueries);

  const int64_t B = gliner2_goldens::kB;
  const int64_t L = gliner2_goldens::kSeqLen;
  const int64_t Q = gliner2_goldens::kNumQueries;

  {
    const size_t n = static_cast<size_t>(Q * B);
    REQUIRE(m.start_logits.size() == n);
    const double worst = MaxAbsDiff(m.start_logits, gliner2_goldens::kStartLogits, n);
    INFO("max abs diff vs upstream start_logits: ", worst);
    CHECK(worst < 2e-5);
  }
  {
    const size_t n = static_cast<size_t>(Q * B);
    REQUIRE(m.end_logits.size() == n);
    const double worst = MaxAbsDiff(m.end_logits, gliner2_goldens::kEndLogits, n);
    INFO("max abs diff vs upstream end_logits: ", worst);
    CHECK(worst < 2e-5);
  }
  {
    const size_t n = static_cast<size_t>(Q * L);
    REQUIRE(m.inside_logits.size() == n);
    const double worst = MaxAbsDiff(m.inside_logits, gliner2_goldens::kInsideLogits, n);
    INFO("max abs diff vs upstream inside_logits: ", worst);
    CHECK(worst < 2e-5);
  }
  {
    const size_t n = static_cast<size_t>(Q * B);
    REQUIRE(m.inside_prefix.size() == n);
    const double worst = MaxAbsDiff(m.inside_prefix, gliner2_goldens::kInsidePrefix, n);
    INFO("max abs diff vs upstream inside_prefix: ", worst);
    CHECK(worst < 2e-5);
  }
}

TEST_CASE("gliner2 sliding window mask changes the boundary states") {
  // Without the window the model still runs and emits plausible boundary
  // states, so the mask is asserted to matter rather than left to be implied.
  const vllm::gliner2::BoundaryParams p = GoldenParams();
  const vllm::gliner2::BoundaryHeadWeights hw =
      vllm::gliner2::LoadBoundaryHead(p, GoldenCheckpoint(p));
  const std::vector<float> text = GoldenTextStates();

  const std::vector<float> windowed = vllm::gliner2::BoundaryEncoderForward(
      p, hw.encoder, text, gliner2_goldens::kSeqLen);

  vllm::gliner2::BoundaryParams p_full = p;
  p_full.boundary_attention_window = 9999;  // effectively full attention
  const std::vector<float> full = vllm::gliner2::BoundaryEncoderForward(
      p_full, hw.encoder, text, gliner2_goldens::kSeqLen);

  REQUIRE(windowed.size() == full.size());
  double worst = 0.0;
  for (size_t i = 0; i < full.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(windowed[i] - full[i])));
  }
  INFO("max abs diff with vs without sliding window mask: ", worst);
  CHECK(worst > 1e-4);
}
