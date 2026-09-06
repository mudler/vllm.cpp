// The synthetic tiny `deepseek4` LANGUAGE GGUF, shared by the W3B loader gate
// and the W4 reachability gate (row
// `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, issue #2411).
//
// EXTRACTED, NOT REWRITTEN, for the reason the projector fixture beside it was:
// W4 drives the same file through `ModelRegistry::Load` instead of through
// `LoadDeepseekV4FromGguf`, and a second hand-written builder would be a second
// description of the artifact that can drift from the loader while both suites
// stay green. `vision` writes `blk.N.exp_probs_b_vl.bias`, which is what makes
// the same file serve a vision checkpoint and a text one.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vt/dtype.h"

namespace dsv4_lang_test {

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::I32ArrayKv;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

// ─── the GGUF arm's own tiny `deepseek4` file ───────────────────────────────
// Deliberately SMALLER than the W2b suite's fixture: no compressor and no
// indexer on any layer (`compress_ratios` all zero), because the DSA families
// have nothing to do with the router bias and every tensor they add is a tensor
// this suite would have to account for again. One hash layer and two gated
// layers is the whole topology the vision bias interacts with.
constexpr int64_t kH = 32, kVocab = 16;
constexpr int64_t kHeads = 2, kHeadDim = 32, kRope = 8;
constexpr int64_t kQLora = 32, kOLora = 32, kOGroups = 2;
constexpr int64_t kExperts = 4, kUsed = 2, kInter = 32;
constexpr int64_t kHc = 2, kSinkhorn = 3;
constexpr int64_t kLayers = 3, kHashLayers = 1;

inline int64_t Prod(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

inline std::vector<uint64_t> GgmlDims(const std::vector<int64_t>& torch_shape) {
  std::vector<uint64_t> d;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it)
    d.push_back(static_cast<uint64_t>(*it));
  return d;
}

template <typename F>
std::string F32Data(int64_t n, F fill) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float v = fill(i);
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    for (int k = 0; k < 4; ++k) s.push_back(static_cast<char>((bits >> (8 * k)) & 0xff));
  }
  return s;
}

// torch [out,in] (in % 32 == 0) -> Q8_0 blocks (`{ f16 d; int8 qs[32] }`).
template <typename F>
std::string Q8Data(int64_t out, int64_t in, F fill) {
  std::string s;
  for (int64_t o = 0; o < out; ++o) {
    for (int64_t b = 0; b < in / 32; ++b) {
      float amax = 0.0f;
      float x[32];
      for (int j = 0; j < 32; ++j) {
        x[j] = fill(o * in + b * 32 + j);
        amax = std::max(amax, std::fabs(x[j]));
      }
      const float d = amax / 127.0f;
      const uint16_t dh = vt::F32ToF16(d);
      s.push_back(static_cast<char>(dh & 0xff));
      s.push_back(static_cast<char>((dh >> 8) & 0xff));
      for (int j = 0; j < 32; ++j) {
        int q = d > 0.0f ? static_cast<int>(std::lround(x[j] / d)) : 0;
        q = std::max(-127, std::min(127, q));
        s.push_back(static_cast<char>(static_cast<int8_t>(q)));
      }
    }
  }
  return s;
}

inline float WFill(int64_t i) { return 0.05f * static_cast<float>((i % 13) - 6); }

// The TWO biases are filled from DIFFERENT functions on purpose. A loader that
// routed `exp_probs_b_vl` into the text slot (or the reverse) would still put a
// plausible `[E]` vector in every slot, so only distinguishable CONTENTS can
// tell the two apart.
inline float TextBiasFill(int64_t l, int64_t i) {
  return 0.25f + 0.5f * static_cast<float>(l) + 0.125f * static_cast<float>(i);
}
inline float VisionBiasFill(int64_t l, int64_t i) {
  return -0.75f - 0.5f * static_cast<float>(l) - 0.0625f * static_cast<float>(i);
}

inline std::string Blk(int64_t l, const std::string& s) {
  return "blk." + std::to_string(l) + "." + s;
}

// The declared width of each router bias. Both default to `expert_count`, which
// is what every published artifact carries. A case that narrows one is asking
// the loader the question a re-quantized publish under an unchanged name asks:
// a bias emitted at `[E-1]` has to be REFUSED, because the router indexes it by
// expert and a short `std::vector<float>` is read past its end rather than
// caught. The fixture writes the KV `expert_count` from `kExperts` regardless,
// so the file states one width and the tensor another — exactly the disagreement
// the loader is the only thing positioned to see.
struct BiasWidths {
  int64_t text = kExperts;
  int64_t vision = kExperts;
};

// `vision` writes `blk.N.exp_probs_b_vl.bias` on EVERY layer, which is what the
// pinned vision artifact carries; false is the text checkpoint.
// `vision_from` is the first layer that carries `exp_probs_b_vl.bias`. 0 is the
// whole artifact, which is what the pinned build holds; a higher value builds
// the PARTIALLY converted file that llama.cpp's `TENSOR_NOT_REQUIRED` accepts.
// `head_dim` defaults to the deliberately tiny `kHeadDim`, which is what the
// W3B loader gate uses because it calls `LoadDeepseekV4FromGguf` directly. A
// caller that enters through `ModelRegistry::Load` instead has to pass 512:
// `ParseDeepseekV4Config` runs there and refuses every other MLA width by name
// ("only the 512-wide MLA geometry (448 NoPE + 64 RoPE) is scoped"). Every
// attention shape below is DERIVED from this argument, so the two files differ
// in one number rather than in a second builder.
inline std::string BuildDeepseek4Gguf(bool vision, BiasWidths bw = BiasWidths{},
                               int64_t vision_from = 0,
                               int64_t head_dim = kHeadDim,
                               bool with_tokenizer = false,
                               // Multiplies every `exp_probs_b_vl` value. Two
                               // files that differ ONLY in this number are what
                               // a gate needs to ask whether the forward READS
                               // the vision bias, and on which rows.
                               float vision_bias_scale = 1.0f) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "deepseek4"));
  const std::string p = "deepseek4.";
  b.AddKv(U32Kv(p + "embedding_length", kH));
  b.AddKv(U32Kv(p + "block_count", kLayers));
  b.AddKv(U32Kv(p + "attention.head_count", kHeads));
  b.AddKv(U32Kv(p + "attention.head_count_kv", 1));
  b.AddKv(U32Kv(p + "attention.key_length", head_dim));
  b.AddKv(U32Kv(p + "rope.dimension_count", kRope));
  b.AddKv(U32Kv(p + "attention.q_lora_rank", kQLora));
  b.AddKv(U32Kv(p + "attention.output_lora_rank", kOLora));
  b.AddKv(U32Kv(p + "attention.output_group_count", kOGroups));
  b.AddKv(F32Kv(p + "rope.freq_base", 10000.0f));
  b.AddKv(F32Kv(p + "attention.compress_rope_freq_base", 160000.0f));
  b.AddKv(F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-6f));
  b.AddKv(U32Kv(p + "expert_count", kExperts));
  b.AddKv(U32Kv(p + "expert_used_count", kUsed));
  b.AddKv(U32Kv(p + "expert_shared_count", 1));
  b.AddKv(U32Kv(p + "expert_feed_forward_length", kInter));
  b.AddKv(U32Kv(p + "hash_layer_count", kHashLayers));
  b.AddKv(F32Kv(p + "swiglu_clamp", 10.0f));
  b.AddKv(U32Kv(p + "hyper_connection.count", kHc));
  b.AddKv(U32Kv(p + "hyper_connection.sinkhorn_iterations", kSinkhorn));
  b.AddKv(F32Kv(p + "hyper_connection.epsilon", 1e-6f));
  b.AddKv(I32ArrayKv(p + "attention.compress_ratios",
                     std::vector<int32_t>(static_cast<size_t>(kLayers), 0)));
  // `LoadedEngine::FromModelDir` opens the tokenizer between the projector
  // block and `ModelRegistry::Load`, so a fixture without these keys stops
  // there. A caller that needs the loader to get PAST the tokenizer asks for
  // them; the W3B gate does not, and stays byte-identical without them.
  if (with_tokenizer) {
    b.AddKv(StrKv("tokenizer.ggml.model", "gpt2"));
    b.AddKv(StrKv("tokenizer.ggml.pre", "llama-bpe"));
    std::vector<std::string> toks;
    std::vector<int32_t> types;
    for (int64_t i = 0; i < kVocab; ++i) {
      toks.push_back(std::string(1, static_cast<char>('a' + i)));
      types.push_back(1);
    }
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.tokens", toks));
    b.AddKv(I32ArrayKv("tokenizer.ggml.token_type", types));
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.merges", {}));
    b.AddKv(U32Kv("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(kVocab - 1)));
  }

  const auto f32 = [&](const std::string& name, const std::vector<int64_t>& shape) {
    b.AddTensor(name, GgmlDims(shape), /*F32=*/0, F32Data(Prod(shape), WFill));
  };
  const auto q8 = [&](const std::string& name, const std::vector<int64_t>& shape) {
    const int64_t out =
        shape.size() == 3 ? shape[0] * shape[1] : shape[0];
    b.AddTensor(name, GgmlDims(shape), /*Q8_0=*/8, Q8Data(out, shape.back(), WFill));
  };

  const int64_t hcf = (2 + kHc) * kHc;
  f32("token_embd.weight", {kVocab, kH});
  q8("output.weight", {kVocab, kH});
  f32("output_norm.weight", {kH});
  f32("output_hc_base.weight", {kHc});
  f32("output_hc_fn.weight", {kHc, kHc * kH});
  f32("output_hc_scale.weight", {1});

  for (int64_t l = 0; l < kLayers; ++l) {
    q8(Blk(l, "attn_q_a.weight"), {kQLora, kH});
    q8(Blk(l, "attn_q_b.weight"), {kHeads * head_dim, kQLora});
    q8(Blk(l, "attn_kv.weight"), {head_dim, kH});
    q8(Blk(l, "attn_output_a.weight"),
       {kOGroups * kOLora, kHeads * head_dim / kOGroups});
    q8(Blk(l, "attn_output_b.weight"), {kH, kOGroups * kOLora});
    f32(Blk(l, "attn_norm.weight"), {kH});
    f32(Blk(l, "attn_q_a_norm.weight"), {kQLora});
    f32(Blk(l, "attn_kv_a_norm.weight"), {head_dim});
    f32(Blk(l, "attn_sinks.weight"), {kHeads});
    f32(Blk(l, "ffn_norm.weight"), {kH});
    f32(Blk(l, "hc_attn_base.weight"), {hcf});
    f32(Blk(l, "hc_attn_fn.weight"), {hcf, kHc * kH});
    f32(Blk(l, "hc_attn_scale.weight"), {3});
    f32(Blk(l, "hc_ffn_base.weight"), {hcf});
    f32(Blk(l, "hc_ffn_fn.weight"), {hcf, kHc * kH});
    f32(Blk(l, "hc_ffn_scale.weight"), {3});
    q8(Blk(l, "ffn_gate_inp.weight"), {kExperts, kH});
    q8(Blk(l, "ffn_gate_exps.weight"), {kExperts, kInter, kH});
    q8(Blk(l, "ffn_up_exps.weight"), {kExperts, kInter, kH});
    q8(Blk(l, "ffn_down_exps.weight"), {kExperts, kH, kInter});
    q8(Blk(l, "ffn_gate_shexp.weight"), {kInter, kH});
    q8(Blk(l, "ffn_up_shexp.weight"), {kInter, kH});
    q8(Blk(l, "ffn_down_shexp.weight"), {kH, kInter});
    if (l < kHashLayers) {
      b.AddTensor(Blk(l, "ffn_gate_tid2eid.weight"), GgmlDims({kVocab, kUsed}),
                  /*F32=*/0, F32Data(kVocab * kUsed, [](int64_t i) {
                    return static_cast<float>(i % kExperts);
                  }));
    } else {
      b.AddTensor(Blk(l, "exp_probs_b.bias"), GgmlDims({bw.text}), /*F32=*/0,
                  F32Data(bw.text, [l](int64_t i) { return TextBiasFill(l, i); }));
    }
    if (vision && l >= vision_from) {
      b.AddTensor(Blk(l, "exp_probs_b_vl.bias"), GgmlDims({bw.vision}), /*F32=*/0,
                  F32Data(bw.vision, [l, vision_bias_scale](int64_t i) {
                    return vision_bias_scale * VisionBiasFill(l, i);
                  }));
    }
  }
  return b.Build();
}
}  // namespace dsv4_lang_test
