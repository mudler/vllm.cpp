// GPT-2 backbone parity gate — IndexTTS-2.5's stage-0 talker (W2, #634).
//
// Compared against a TP=1 restatement of vLLM's
// vllm/model_executor/models/gpt2.py @ 555967922 (the parity pin), executed at
// reduced dimensions by scripts/gen-gpt2-goldens.py. Both sides rebuild every
// weight from ONE deterministic FNV-1a -> splitmix64 stream, so no weight byte
// is checked in.
//
// The goldens carry Conv1D weights in UPSTREAM [in, out] orientation, which is
// how a real checkpoint stores them. A port that skips the transpose
// (gpt2.py:242-254 `_transpose_conv1d`) still runs and still emits plausible
// tokens, so the transpose is gated by a case of its own rather than left to be
// implied by the forward.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "gpt2_goldens.inc"
#include "vllm/model_executor/models/gpt2.h"

namespace {

// The generator's stream, byte-for-byte: values uniform in [-1, 1) derived from
// the tensor NAME alone. Deliberately a local copy — if this drifts from the
// generator the goldens stop matching, which is the failure we want, not a
// shared helper that could drift on both sides at once.
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

vllm::gpt2::Params GoldenParams() {
  vllm::gpt2::Params p;
  p.vocab_size = gpt2_goldens::kVocab;
  p.max_position_embeddings = gpt2_goldens::kPositions;
  p.hidden_size = gpt2_goldens::kHidden;
  p.num_hidden_layers = gpt2_goldens::kLayers;
  p.num_attention_heads = gpt2_goldens::kHeads;
  p.inner_size = gpt2_goldens::kInner;
  p.layer_norm_eps = gpt2_goldens::kLayerNormEps;
  return p;
}

// Rebuild the checkpoint in UPSTREAM orientation: Conv1D 2D weights are
// [in, out]. `Load` is what must transpose them.
vllm::gpt2::CheckpointTensors GoldenCheckpoint(const vllm::gpt2::Params& p) {
  vllm::gpt2::CheckpointTensors t;
  const int64_t h = p.hidden_size;
  t.Set("wte.weight", {p.vocab_size, h}, Rand("wte.weight", p.vocab_size * h, 0.5));
  t.Set("wpe.weight", {p.max_position_embeddings, h},
        Rand("wpe.weight", p.max_position_embeddings * h, 0.25));
  t.Set("ln_f.weight", {h}, RandPlusOne("ln_f.weight", h, 0.1));
  t.Set("ln_f.bias", {h}, Rand("ln_f.bias", h, 0.1));
  for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
    const std::string b = "h." + std::to_string(i) + ".";
    t.Set(b + "ln_1.weight", {h}, RandPlusOne(b + "ln_1.weight", h, 0.1));
    t.Set(b + "ln_1.bias", {h}, Rand(b + "ln_1.bias", h, 0.1));
    t.Set(b + "ln_2.weight", {h}, RandPlusOne(b + "ln_2.weight", h, 0.1));
    t.Set(b + "ln_2.bias", {h}, Rand(b + "ln_2.bias", h, 0.1));
    t.Set(b + "attn.c_attn.weight", {h, 3 * h}, Rand(b + "attn.c_attn.weight", h * 3 * h, 0.3));
    t.Set(b + "attn.c_attn.bias", {3 * h}, Rand(b + "attn.c_attn.bias", 3 * h, 0.2));
    t.Set(b + "attn.c_proj.weight", {h, h}, Rand(b + "attn.c_proj.weight", h * h, 0.3));
    t.Set(b + "attn.c_proj.bias", {h}, Rand(b + "attn.c_proj.bias", h, 0.2));
    t.Set(b + "mlp.c_fc.weight", {h, p.inner_size}, Rand(b + "mlp.c_fc.weight", h * p.inner_size, 0.3));
    t.Set(b + "mlp.c_fc.bias", {p.inner_size}, Rand(b + "mlp.c_fc.bias", p.inner_size, 0.2));
    t.Set(b + "mlp.c_proj.weight", {p.inner_size, h},
          Rand(b + "mlp.c_proj.weight", p.inner_size * h, 0.3));
    t.Set(b + "mlp.c_proj.bias", {h}, Rand(b + "mlp.c_proj.bias", h, 0.2));
  }
  return t;
}

std::vector<int64_t> GoldenIds() {
  return std::vector<int64_t>(std::begin(gpt2_goldens::kInputIds),
                              std::end(gpt2_goldens::kInputIds));
}

std::vector<int64_t> GoldenPositions() {
  return std::vector<int64_t>(std::begin(gpt2_goldens::kPositions0),
                              std::end(gpt2_goldens::kPositions0));
}

double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
  }
  return worst;
}

}  // namespace

TEST_CASE("gpt2 forward reproduces the upstream hidden states") {
  const vllm::gpt2::Params p = GoldenParams();
  const vllm::gpt2::Weights w = vllm::gpt2::Load(p, GoldenCheckpoint(p));
  const std::vector<float> hidden = vllm::gpt2::ForwardHost(p, w, GoldenIds(), GoldenPositions());

  const size_t n = GoldenIds().size() * static_cast<size_t>(p.hidden_size);
  REQUIRE(hidden.size() == n);
  const double worst = MaxAbsDiff(hidden, gpt2_goldens::kHiddenStates, n);
  INFO("max abs diff vs upstream hidden states: ", worst);
  CHECK(worst < 2e-5);
}

TEST_CASE("gpt2 logits reproduce upstream, and the argmax is token-exact") {
  const vllm::gpt2::Params p = GoldenParams();
  const vllm::gpt2::Weights w = vllm::gpt2::Load(p, GoldenCheckpoint(p));
  const std::vector<float> hidden = vllm::gpt2::ForwardHost(p, w, GoldenIds(), GoldenPositions());
  const std::vector<float> logits = vllm::gpt2::LogitsHost(p, w, hidden);

  const size_t seq = GoldenIds().size();
  const size_t n = seq * static_cast<size_t>(p.vocab_size);
  REQUIRE(logits.size() == n);
  const double worst = MaxAbsDiff(logits, gpt2_goldens::kLogits, n);
  INFO("max abs diff vs upstream logits: ", worst);
  CHECK(worst < 5e-5);

  // Token-exactness is the property the talker actually needs: a small logit
  // drift that never changes an argmax is harmless, one that does is a
  // different utterance.
  for (size_t t = 0; t < seq; ++t) {
    int64_t best = 0;
    float best_v = logits[t * static_cast<size_t>(p.vocab_size)];
    for (int64_t v = 1; v < p.vocab_size; ++v) {
      const float cur = logits[t * static_cast<size_t>(p.vocab_size) + static_cast<size_t>(v)];
      if (cur > best_v) {
        best_v = cur;
        best = v;
      }
    }
    CHECK(best == gpt2_goldens::kArgmax[t]);
  }
}

TEST_CASE("gpt2 Load transposes the Conv1D weights") {
  // gpt2.py:242-254. HF stores c_attn/c_proj/c_fc as Conv1D: the 2D weight is
  // [in, out], while the matmul wants [out, in]. Skipping the transpose yields a
  // model that runs and is wrong, so the orientation is asserted directly rather
  // than inferred from the forward.
  const vllm::gpt2::Params p = GoldenParams();
  const vllm::gpt2::CheckpointTensors ckpt = GoldenCheckpoint(p);
  const vllm::gpt2::Weights w = vllm::gpt2::Load(p, ckpt);

  const std::vector<float>& raw = ckpt.Get("h.0.attn.c_attn.weight");
  const std::vector<float>& loaded = w.layers[0].c_attn_weight;
  REQUIRE(loaded.size() == raw.size());

  const int64_t in_dim = p.hidden_size;
  const int64_t out_dim = 3 * p.hidden_size;
  for (int64_t i = 0; i < in_dim; ++i) {
    for (int64_t o = 0; o < out_dim; ++o) {
      // raw is [in, out]; loaded must be [out, in].
      CHECK(loaded[static_cast<size_t>(o * in_dim + i)] ==
            raw[static_cast<size_t>(i * out_dim + o)]);
    }
  }
}

TEST_CASE("gpt2 attention is causal") {
  // A change to the LAST token must not move any earlier position. Without the
  // causal mask every position sees the future, which still produces fluent
  // output and is the wrong model.
  const vllm::gpt2::Params p = GoldenParams();
  const vllm::gpt2::Weights w = vllm::gpt2::Load(p, GoldenCheckpoint(p));

  std::vector<int64_t> ids = GoldenIds();
  const std::vector<int64_t> positions = GoldenPositions();
  const std::vector<float> base = vllm::gpt2::ForwardHost(p, w, ids, positions);

  ids.back() = (ids.back() + 5) % p.vocab_size;
  const std::vector<float> perturbed = vllm::gpt2::ForwardHost(p, w, ids, positions);

  const size_t hidden = static_cast<size_t>(p.hidden_size);
  const size_t prefix = (ids.size() - 1) * hidden;
  for (size_t i = 0; i < prefix; ++i) {
    CHECK(base[i] == perturbed[i]);
  }
  double moved = 0.0;
  for (size_t i = prefix; i < base.size(); ++i) {
    moved = std::max(moved, std::fabs(static_cast<double>(base[i] - perturbed[i])));
  }
  CHECK(moved > 1e-6);  // the last position MUST move, or the test proves nothing
}
