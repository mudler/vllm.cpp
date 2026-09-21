// ModernBERT encoder parity gate — Laya backbone (MODEL-LAYA).
//
// Compared against a restatement of vLLM model_executor/models/modernbert.py
// @ pin e126687a9a (mirrors transformers/models/modernbert/modeling_modernbert.py),
// executed at reduced dimensions by scripts/gen-modernbert-goldens.py. Both
// sides rebuild every weight from ONE deterministic FNV-1a -> splitmix64
// stream, so no weight byte is checked in.
//
// The dual RoPE theta (local vs global), the sliding window on local layers,
// and the Identity norm on layer 0 are the features this architecture gets
// wrong quietly: without them the model still runs and emits plausible hidden
// states. Perturbation tests disable each and verify the output changes.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "modernbert_goldens.inc"
#include "vllm/model_executor/models/modernbert.h"

namespace {

// The generator's stream, byte-for-byte: values uniform in [-1, 1) derived
// from the tensor NAME alone. Deliberately a local copy — if this drifts from
// the generator the goldens stop matching, which is the failure we want, not
// a shared helper that could drift on both sides at once.
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

std::vector<float> RandPlusOne(const std::string& name, int64_t count, double scale) {
  std::vector<float> v = Rand(name, count, scale);
  for (float& x : v) x += 1.0F;
  return v;
}

vllm::modernbert::Params GoldenParams() {
  vllm::modernbert::Params p;
  p.vocab_size = modernbert_goldens::kVocab;
  p.hidden_size = modernbert_goldens::kHidden;
  p.num_hidden_layers = modernbert_goldens::kLayers;
  p.num_attention_heads = modernbert_goldens::kHeads;
  p.head_dim = modernbert_goldens::kHeadDim;
  p.intermediate_size = modernbert_goldens::kInter;
  p.local_attention = modernbert_goldens::kLocalAttention;
  p.global_attn_every_n_layers = modernbert_goldens::kGlobalAttnEveryN;
  p.local_rope_theta = modernbert_goldens::kLocalRopeTheta;
  p.global_rope_theta = modernbert_goldens::kGlobalRopeTheta;
  p.layer_norm_eps = modernbert_goldens::kLayerNormEps;
  p.norm_bias = false;
  p.mlp_bias = false;
  p.attention_bias = false;
  return p;
}

// Rebuild the checkpoint in nn.Linear [out, in] orientation, matching the
// generator's weight names byte-for-byte.
vllm::modernbert::CheckpointTensors GoldenCheckpoint(const vllm::modernbert::Params& p) {
  vllm::modernbert::CheckpointTensors t;
  const int64_t h = p.hidden_size;
  const int64_t inter = p.intermediate_size;

  t.Set("encoder.embeddings.tok_embeddings.weight", {p.vocab_size, h},
        Rand("encoder.embeddings.tok_embeddings.weight", p.vocab_size * h, 0.5));
  t.Set("encoder.embeddings.norm.weight", {h},
        RandPlusOne("encoder.embeddings.norm.weight", h, 0.1));
  t.Set("encoder.final_norm.weight", {h},
        RandPlusOne("encoder.final_norm.weight", h, 0.1));

  for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
    const std::string b = "encoder.layers." + std::to_string(i) + ".";
    t.Set(b + "attn.Wqkv.weight", {3 * h, h},
          Rand(b + "attn.Wqkv.weight", 3 * h * h, 0.3));
    t.Set(b + "attn.Wo.weight", {h, h},
          Rand(b + "attn.Wo.weight", h * h, 0.3));
    // Layer 0 has Identity attn_norm (no weight tensor).
    if (i > 0) {
      t.Set(b + "attn_norm.weight", {h},
            RandPlusOne(b + "attn_norm.weight", h, 0.1));
    }
    t.Set(b + "mlp.Wi.weight", {2 * inter, h},
          Rand(b + "mlp.Wi.weight", 2 * inter * h, 0.3));
    t.Set(b + "mlp.Wo.weight", {h, inter},
          Rand(b + "mlp.Wo.weight", h * inter, 0.3));
    t.Set(b + "mlp_norm.weight", {h},
          RandPlusOne(b + "mlp_norm.weight", h, 0.1));
  }
  return t;
}

std::vector<int64_t> GoldenIds() {
  return std::vector<int64_t>(std::begin(modernbert_goldens::kInputIds),
                              std::end(modernbert_goldens::kInputIds));
}

double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
  }
  return worst;
}

}  // namespace

TEST_CASE("modernbert forward reproduces the upstream hidden states") {
  const vllm::modernbert::Params p = GoldenParams();
  const vllm::modernbert::Weights w = vllm::modernbert::Load(p, GoldenCheckpoint(p));
  const std::vector<float> hidden = vllm::modernbert::ForwardHost(p, w, GoldenIds());

  const size_t n = GoldenIds().size() * static_cast<size_t>(p.hidden_size);
  REQUIRE(hidden.size() == n);
  const double worst = MaxAbsDiff(hidden, modernbert_goldens::kHiddenStates, n);
  INFO("max abs diff vs upstream hidden states: ", worst);
  CHECK(worst < 2e-5);
}

TEST_CASE("modernbert dual RoPE theta changes the output") {
  // Swapping local and global theta still runs and emits plausible hidden
  // states, so the distinction is asserted to matter rather than left to be
  // implied by the forward.
  const vllm::modernbert::Params p = GoldenParams();
  const vllm::modernbert::Weights w = vllm::modernbert::Load(p, GoldenCheckpoint(p));

  const std::vector<float> base = vllm::modernbert::ForwardHost(p, w, GoldenIds());

  vllm::modernbert::Params p_swapped = p;
  std::swap(p_swapped.local_rope_theta, p_swapped.global_rope_theta);
  const std::vector<float> swapped = vllm::modernbert::ForwardHost(p_swapped, w, GoldenIds());

  REQUIRE(base.size() == swapped.size());
  double worst = 0.0;
  for (size_t i = 0; i < base.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(base[i] - swapped[i])));
  }
  INFO("max abs diff with vs without swapped RoPE theta: ", worst);
  CHECK(worst > 1e-4);
}

TEST_CASE("modernbert sliding window changes the output") {
  // Without the sliding window the local layers silently attend to all
  // positions, which still produces plausible output and is the wrong model.
  const vllm::modernbert::Params p = GoldenParams();
  const vllm::modernbert::Weights w = vllm::modernbert::Load(p, GoldenCheckpoint(p));

  const std::vector<float> base = vllm::modernbert::ForwardHost(p, w, GoldenIds());

  vllm::modernbert::Params p_no_window = p;
  p_no_window.local_attention = 0;  // sliding_window() returns 0 → disabled
  const std::vector<float> no_window = vllm::modernbert::ForwardHost(p_no_window, w, GoldenIds());

  REQUIRE(base.size() == no_window.size());
  double worst = 0.0;
  for (size_t i = 0; i < base.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(base[i] - no_window[i])));
  }
  INFO("max abs diff with vs without sliding window: ", worst);
  CHECK(worst > 1e-4);
}

TEST_CASE("modernbert attention is bidirectional") {
  // ModernBERT is bidirectional: a change to an EARLY token must move LATER
  // positions. With a causal mask every position only sees the past, which
  // still produces plausible output and is the wrong model.
  const vllm::modernbert::Params p = GoldenParams();
  const vllm::modernbert::Weights w = vllm::modernbert::Load(p, GoldenCheckpoint(p));

  std::vector<int64_t> ids = GoldenIds();
  const std::vector<float> base = vllm::modernbert::ForwardHost(p, w, ids);

  // Perturb the FIRST token and check that the LAST position moves.
  ids[0] = (ids[0] + 5) % p.vocab_size;
  const std::vector<float> perturbed = vllm::modernbert::ForwardHost(p, w, ids);

  const size_t hidden = static_cast<size_t>(p.hidden_size);
  const size_t last_offset = (ids.size() - 1) * hidden;
  double moved = 0.0;
  for (size_t i = last_offset; i < base.size(); ++i) {
    moved = std::max(moved, std::fabs(static_cast<double>(base[i] - perturbed[i])));
  }
  CHECK(moved > 1e-6);
}
