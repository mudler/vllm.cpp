// The S2Mel DiT transformer stack against upstream goldens. See dit_stack.h.
#include <cstdint>
#include <string>
#include <vector>

#include "dit_stack_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/dit_stack.h"

namespace {

std::vector<float> Rnd(const std::string& name, size_t n, double scale) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char ch : name) {
    h = (h ^ ch) * 0x100000001B3ULL;
  }
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    h += 0x9E3779B97F4A7C15ULL;
    uint64_t z = h;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double u = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    out[i] = static_cast<float>((u * 2.0 - 1.0) * scale);
  }
  return out;
}

using namespace dit_stack_goldens;

vllm::models::dit_stack::Config Cfg() {
  vllm::models::dit_stack::Config c;
  c.dim = kDim;
  c.heads = kHeads;
  c.head_dim = kHeadDim;
  c.intermediate = kIntermediate;
  c.frames = kFrames;
  c.eps = kEps;
  return c;
}

vllm::models::dit_stack::Weights W() {
  vllm::models::dit_stack::Weights w;
  const size_t D = static_cast<size_t>(kDim);
  const size_t I = static_cast<size_t>(kIntermediate);
  const size_t QKV = static_cast<size_t>((kHeads + 2 * kHeads) * kHeadDim);
  for (int64_t i = 0; i < kDepth; ++i) {
    const std::string p = "stack.layers." + std::to_string(i) + ".";
    vllm::models::dit_stack::LayerWeights l;
    l.block.wqkv = Rnd(p + "attention.wqkv.weight", QKV * D, 0.5);
    l.block.wo = Rnd(p + "attention.wo.weight", D * D, 0.5);
    l.block.w1 = Rnd(p + "feed_forward.w1.weight", I * D, 0.5);
    l.block.w3 = Rnd(p + "feed_forward.w3.weight", I * D, 0.5);
    l.block.w2 = Rnd(p + "feed_forward.w2.weight", D * I, 0.5);
    l.block.attn_proj_w = Rnd(p + "attention_norm.project_layer.weight", 2 * D * D, 0.5);
    l.block.attn_proj_b = Rnd(p + "attention_norm.project_layer.bias", 2 * D, 0.5);
    l.block.attn_norm_w = Rnd(p + "attention_norm.norm.weight", D, 0.5);
    l.block.ffn_proj_w = Rnd(p + "ffn_norm.project_layer.weight", 2 * D * D, 0.5);
    l.block.ffn_proj_b = Rnd(p + "ffn_norm.project_layer.bias", 2 * D, 0.5);
    l.block.ffn_norm_w = Rnd(p + "ffn_norm.norm.weight", D, 0.5);
    l.skip_in_w = Rnd(p + "skip_in_linear.weight", D * 2 * D, 0.5);
    l.skip_in_b = Rnd(p + "skip_in_linear.bias", D, 0.5);
    w.layers.push_back(std::move(l));
  }
  w.norm_proj_w = Rnd("stack.norm.project_layer.weight", 2 * D * D, 0.5);
  w.norm_proj_b = Rnd("stack.norm.project_layer.bias", 2 * D, 0.5);
  w.norm_w = Rnd("stack.norm.norm.weight", D, 0.5);
  return w;
}

}  // namespace

TEST_CASE("the whole DiT stack matches upstream Transformer.forward") {
  const std::vector<float> x = Rnd("stack.x", static_cast<size_t>(kFrames * kDim), 1.0);
  const std::vector<float> c = Rnd("stack.c", static_cast<size_t>(kDim), 1.0);
  const std::vector<float> freqs(kFreqs,
                                 kFreqs + static_cast<size_t>(kFrames * kHeadDim / 2 * 2));

  const std::vector<float> got = vllm::models::dit_stack::Forward(Cfg(), W(), x, c, freqs);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kDim));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(kOut[i]).epsilon(3e-5));
  }
}

TEST_CASE("the skip merge is actually consulted") {
  // Perturbing ONLY a receiving layer's skip_in_linear must move the output. If
  // the routing silently skipped the merge, the stack would still be a valid
  // transformer and this is what separates the two.
  const std::vector<float> x = Rnd("stack.x", static_cast<size_t>(kFrames * kDim), 1.0);
  const std::vector<float> c = Rnd("stack.c", static_cast<size_t>(kDim), 1.0);
  const std::vector<float> freqs(kFreqs,
                                 kFreqs + static_cast<size_t>(kFrames * kHeadDim / 2 * 2));
  const auto base = vllm::models::dit_stack::Forward(Cfg(), W(), x, c, freqs);

  // Depth 5 routes 3 <- 1 and 4 <- 0, so layer 3 is a receiver.
  auto w = W();
  w.layers[3].skip_in_w[0] += 1.0F;
  const auto moved = vllm::models::dit_stack::Forward(Cfg(), w, x, c, freqs);
  CHECK(moved != base);

  // Layer 0 never receives, so ITS skip_in_linear must not matter at all.
  auto w0 = W();
  w0.layers[0].skip_in_w[0] += 1.0F;
  CHECK(vllm::models::dit_stack::Forward(Cfg(), w0, x, c, freqs) == base);
}

TEST_CASE("the final norm is applied") {
  const std::vector<float> x = Rnd("stack.x", static_cast<size_t>(kFrames * kDim), 1.0);
  const std::vector<float> c = Rnd("stack.c", static_cast<size_t>(kDim), 1.0);
  const std::vector<float> freqs(kFreqs,
                                 kFreqs + static_cast<size_t>(kFrames * kHeadDim / 2 * 2));
  const auto base = vllm::models::dit_stack::Forward(Cfg(), W(), x, c, freqs);
  auto w = W();
  w.norm_w[0] += 1.0F;
  CHECK(vllm::models::dit_stack::Forward(Cfg(), w, x, c, freqs) != base);
}
