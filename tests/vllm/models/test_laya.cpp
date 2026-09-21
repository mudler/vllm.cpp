// Laya decision head parity gate — System 1 model on ModernBERT (MODEL-LAYA).
//
// Compared against a restatement of laya/common.py DecisionModel.forward
// (convaiinnovations/laya), executed at reduced dimensions by
// scripts/gen-laya-goldens.py. Both sides rebuild every weight from ONE
// deterministic FNV-1a -> splitmix64 stream, so no weight byte is checked in.
//
// The type_emb addition, the key_padding_mask in head attention, the pre-norm
// head layers, and the head layers themselves are the features this head gets
// wrong quietly: without them the model still runs and emits plausible logits.
// Perturbation tests verify each feature changes the output.
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "laya_goldens.inc"
#include "vllm/model_executor/models/laya.h"

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

vllm::laya::Params GoldenParams() {
  vllm::laya::Params p;
  p.hidden_size = laya_goldens::kHidden;
  p.num_heads = laya_goldens::kHeads;
  p.head_layers = laya_goldens::kHeadLayers;
  p.dim_ff = laya_goldens::kDimFf;
  p.n_act = laya_goldens::kNAct;
  p.act_hidden = laya_goldens::kActHidden;
  p.layer_norm_eps = laya_goldens::kLayerNormEps;
  return p;
}

// Rebuild the checkpoint in nn.Linear [out, in] orientation, matching the
// generator's weight names byte-for-byte.
vllm::laya::CheckpointTensors GoldenCheckpoint(const vllm::laya::Params& p) {
  vllm::laya::CheckpointTensors t;
  const int64_t d = p.hidden_size;
  const int64_t ff = p.dim_ff;
  const int64_t act_in = d + 4;

  t.Set("type_emb.weight", {3, d}, Rand("type_emb.weight", 3 * d, 0.3));
  t.Set("temperature", {3}, RandPlusOne("temperature", 3, 0.3));

  for (int64_t i = 0; i < p.head_layers; ++i) {
    const std::string b = "head.layers." + std::to_string(i) + ".";
    t.Set(b + "self_attn.in_proj_weight", {3 * d, d},
          Rand(b + "self_attn.in_proj_weight", 3 * d * d, 0.3));
    t.Set(b + "self_attn.in_proj_bias", {3 * d},
          Rand(b + "self_attn.in_proj_bias", 3 * d, 0.3));
    t.Set(b + "self_attn.out_proj.weight", {d, d},
          Rand(b + "self_attn.out_proj.weight", d * d, 0.3));
    t.Set(b + "self_attn.out_proj.bias", {d},
          Rand(b + "self_attn.out_proj.bias", d, 0.3));
    t.Set(b + "linear1.weight", {ff, d},
          Rand(b + "linear1.weight", ff * d, 0.3));
    t.Set(b + "linear1.bias", {ff},
          Rand(b + "linear1.bias", ff, 0.3));
    t.Set(b + "linear2.weight", {d, ff},
          Rand(b + "linear2.weight", d * ff, 0.3));
    t.Set(b + "linear2.bias", {d},
          Rand(b + "linear2.bias", d, 0.3));
    t.Set(b + "norm1.weight", {d},
          RandPlusOne(b + "norm1.weight", d, 0.1));
    t.Set(b + "norm1.bias", {d},
          Rand(b + "norm1.bias", d, 0.1));
    t.Set(b + "norm2.weight", {d},
          RandPlusOne(b + "norm2.weight", d, 0.1));
    t.Set(b + "norm2.bias", {d},
          Rand(b + "norm2.bias", d, 0.1));
  }

  t.Set("scorer.0.weight", {d}, RandPlusOne("scorer.0.weight", d, 0.1));
  t.Set("scorer.0.bias", {d}, Rand("scorer.0.bias", d, 0.1));
  t.Set("scorer.1.weight", {d, d}, Rand("scorer.1.weight", d * d, 0.3));
  t.Set("scorer.1.bias", {d}, Rand("scorer.1.bias", d, 0.3));
  t.Set("scorer.3.weight", {1, d}, Rand("scorer.3.weight", d, 0.3));
  t.Set("scorer.3.bias", {1}, Rand("scorer.3.bias", 1, 0.3));

  t.Set("act_head.0.weight", {p.act_hidden, act_in},
        Rand("act_head.0.weight", p.act_hidden * act_in, 0.3));
  t.Set("act_head.0.bias", {p.act_hidden},
        Rand("act_head.0.bias", p.act_hidden, 0.3));
  t.Set("act_head.2.weight", {p.n_act, p.act_hidden},
        Rand("act_head.2.weight", p.n_act * p.act_hidden, 0.3));
  t.Set("act_head.2.bias", {p.n_act},
        Rand("act_head.2.bias", p.n_act, 0.3));

  return t;
}

std::vector<float> GoldenHiddenStates() {
  const int64_t seq = laya_goldens::kSeq;
  const int64_t d = laya_goldens::kHidden;
  return Rand("input.hidden_states", seq * d, 0.5);
}

std::vector<int64_t> GoldenAttentionMask() {
  return std::vector<int64_t>(std::begin(laya_goldens::kAttentionMask),
                               std::end(laya_goldens::kAttentionMask));
}

std::vector<int64_t> GoldenMarkerPos() {
  return std::vector<int64_t>(std::begin(laya_goldens::kMarkerPos),
                               std::end(laya_goldens::kMarkerPos));
}

std::vector<int64_t> GoldenMarkerMask() {
  return std::vector<int64_t>(std::begin(laya_goldens::kMarkerMask),
                               std::end(laya_goldens::kMarkerMask));
}

double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
  }
  return worst;
}

}  // namespace

TEST_CASE("laya forward reproduces the upstream logits and act_logits") {
  const vllm::laya::Params p = GoldenParams();
  const vllm::laya::Weights w = vllm::laya::Load(p, GoldenCheckpoint(p));

  const vllm::laya::ForwardOutput out = vllm::laya::ForwardHost(
      p, w, GoldenHiddenStates(), GoldenAttentionMask(),
      GoldenMarkerPos(), GoldenMarkerMask(), laya_goldens::kQtype);

  const size_t k_max = static_cast<size_t>(laya_goldens::kKMax);
  REQUIRE(out.logits.size() == k_max);
  REQUIRE(out.act_logits.size() == static_cast<size_t>(laya_goldens::kNAct));

  const double logit_diff = MaxAbsDiff(out.logits, laya_goldens::kLogits, k_max);
  INFO("max abs diff vs upstream logits: ", logit_diff);
  CHECK(logit_diff < 2e-5);

  const double act_diff = MaxAbsDiff(out.act_logits, laya_goldens::kActLogits,
                                      static_cast<size_t>(laya_goldens::kNAct));
  INFO("max abs diff vs upstream act_logits: ", act_diff);
  CHECK(act_diff < 2e-5);
}

TEST_CASE("laya type_emb addition changes the output") {
  // Skipping the type_emb addition still runs and emits plausible logits,
  // so the distinction is asserted to matter: different qtypes must produce
  // different outputs. If type_emb is skipped, both qtypes are identical.
  const vllm::laya::Params p = GoldenParams();
  const vllm::laya::Weights w = vllm::laya::Load(p, GoldenCheckpoint(p));

  const auto hidden = GoldenHiddenStates();
  const auto mask = GoldenAttentionMask();
  const auto mpos = GoldenMarkerPos();
  const auto mmask = GoldenMarkerMask();

  const vllm::laya::ForwardOutput out0 =
      vllm::laya::ForwardHost(p, w, hidden, mask, mpos, mmask, 0);
  const vllm::laya::ForwardOutput out1 =
      vllm::laya::ForwardHost(p, w, hidden, mask, mpos, mmask, 1);

  REQUIRE(out0.logits.size() == out1.logits.size());
  double worst = 0.0;
  for (size_t i = 0; i < out0.logits.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(out0.logits[i] - out1.logits[i])));
  }
  INFO("max abs diff with qtype=0 vs qtype=1: ", worst);
  CHECK(worst > 1e-4);
}

TEST_CASE("laya key_padding_mask changes the output") {
  // Without the key_padding_mask the padded positions silently contribute to
  // attention, which still produces plausible output and is the wrong model.
  const vllm::laya::Params p = GoldenParams();
  const vllm::laya::Weights w = vllm::laya::Load(p, GoldenCheckpoint(p));

  const auto hidden = GoldenHiddenStates();
  const auto mpos = GoldenMarkerPos();
  const auto mmask = GoldenMarkerMask();

  // Partial mask: positions 5,6,7 are padding.
  std::vector<int64_t> mask_partial = GoldenAttentionMask();
  // Full mask: all positions valid.
  std::vector<int64_t> mask_full(mask_partial.size(), 1);

  const vllm::laya::ForwardOutput out_partial =
      vllm::laya::ForwardHost(p, w, hidden, mask_partial, mpos, mmask, 0);
  const vllm::laya::ForwardOutput out_full =
      vllm::laya::ForwardHost(p, w, hidden, mask_full, mpos, mmask, 0);

  REQUIRE(out_partial.logits.size() == out_full.logits.size());
  double worst = 0.0;
  for (size_t i = 0; i < out_partial.logits.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(out_partial.logits[i] - out_full.logits[i])));
  }
  INFO("max abs diff with partial vs full attention mask: ", worst);
  CHECK(worst > 1e-4);
}

TEST_CASE("laya head layers change the output") {
  // Without the head layers the type_emb addition goes directly to the scorer,
  // which still produces plausible logits and is the wrong model.
  const vllm::laya::Params p = GoldenParams();
  const vllm::laya::CheckpointTensors t = GoldenCheckpoint(p);

  const auto hidden = GoldenHiddenStates();
  const auto mask = GoldenAttentionMask();
  const auto mpos = GoldenMarkerPos();
  const auto mmask = GoldenMarkerMask();

  const vllm::laya::Weights w_full = vllm::laya::Load(p, t);

  vllm::laya::Params p_no_head = p;
  p_no_head.head_layers = 0;
  const vllm::laya::Weights w_no_head = vllm::laya::Load(p_no_head, t);

  const vllm::laya::ForwardOutput out_full =
      vllm::laya::ForwardHost(p, w_full, hidden, mask, mpos, mmask, 0);
  const vllm::laya::ForwardOutput out_no_head =
      vllm::laya::ForwardHost(p_no_head, w_no_head, hidden, mask, mpos, mmask, 0);

  REQUIRE(out_full.logits.size() == out_no_head.logits.size());
  double worst = 0.0;
  for (size_t i = 0; i < out_full.logits.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(out_full.logits[i] - out_no_head.logits[i])));
  }
  INFO("max abs diff with vs without head layers: ", worst);
  CHECK(worst > 1e-4);
}

// --- Temperature scaling tests (Finding 3) ---

TEST_CASE("laya temperature: bucket lookup returns the mapped value") {
  const std::vector<float> temperature = {1.0F, 1.0F, 1.0F};
  const std::map<std::string, float> temperature_by_options = {
      {"choice:3-5", 0.8F},
      {"noul:2", 1.2F},
  };

  // k=4 → bucket "choice:3-5" → 0.8
  CHECK(vllm::laya::TemperatureFor(0 /*choice*/, 4, temperature,
                                    temperature_by_options) == doctest::Approx(0.8F));
  // k=2 → bucket "noul:2" → 1.2
  CHECK(vllm::laya::TemperatureFor(2 /*noul*/, 2, temperature,
                                    temperature_by_options) == doctest::Approx(1.2F));
}

TEST_CASE("laya temperature: falls back to per-type temperature when no bucket matches") {
  const std::vector<float> temperature = {0.5F, 0.7F, 0.9F};
  const std::map<std::string, float> temperature_by_options = {
      {"choice:3-5", 0.8F},  // only choice:3-5 is mapped
  };

  // k=4 with noul → no "noul:3-5" bucket → fallback to temperature[2]=0.9
  CHECK(vllm::laya::TemperatureFor(2 /*noul*/, 4, temperature,
                                    temperature_by_options) == doctest::Approx(0.9F));
  // k=11 with choice → no "choice:11+" bucket → fallback to temperature[0]=0.5
  CHECK(vllm::laya::TemperatureFor(0 /*choice*/, 11, temperature,
                                    temperature_by_options) == doctest::Approx(0.5F));
  // k=2 with score → no "score:2" bucket → fallback to temperature[1]=0.7
  CHECK(vllm::laya::TemperatureFor(1 /*score*/, 2, temperature,
                                    temperature_by_options) == doctest::Approx(0.7F));
}

TEST_CASE("laya temperature: bucket boundaries are correct") {
  const std::vector<float> temperature = {1.0F, 1.0F, 1.0F};
  const std::map<std::string, float> temperature_by_options = {
      {"choice:2", 0.2F},
      {"choice:3-5", 0.4F},
      {"choice:6-10", 0.6F},
      {"choice:11+", 0.8F},
  };

  // k <= 2 → "choice:2"
  CHECK(vllm::laya::TemperatureFor(0, 1, temperature, temperature_by_options) == doctest::Approx(0.2F));
  CHECK(vllm::laya::TemperatureFor(0, 2, temperature, temperature_by_options) == doctest::Approx(0.2F));
  // 3 <= k <= 5 → "choice:3-5"
  CHECK(vllm::laya::TemperatureFor(0, 3, temperature, temperature_by_options) == doctest::Approx(0.4F));
  CHECK(vllm::laya::TemperatureFor(0, 5, temperature, temperature_by_options) == doctest::Approx(0.4F));
  // 6 <= k <= 10 → "choice:6-10"
  CHECK(vllm::laya::TemperatureFor(0, 6, temperature, temperature_by_options) == doctest::Approx(0.6F));
  CHECK(vllm::laya::TemperatureFor(0, 10, temperature, temperature_by_options) == doctest::Approx(0.6F));
  // k > 10 → "choice:11+"
  CHECK(vllm::laya::TemperatureFor(0, 11, temperature, temperature_by_options) == doctest::Approx(0.8F));
}
