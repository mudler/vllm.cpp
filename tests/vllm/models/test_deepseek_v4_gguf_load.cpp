// DeepSeek-V4-Flash W2b — the `deepseek4` GGUF keep-quant TOWER materialization.
//
// Builds a TINY SYNTHETIC `deepseek4` GGUF (gguf_builder.h) with the REAL tensor
// NAMING convention (`blk.N.*`, the check-dsv4-gguf-namemap.py map) + a REAL
// keep-quant ggml type (Q8_0) at tiny dims, then asserts the loader
// (LoadDeepseekV4FromGguf):
//   (a) ACCOUNTS for EVERY GGUF tensor into a non-empty tower slot — none
//       unmapped (a missing required tensor throws), none leftover (a tensor the
//       map does not cover throws);
//   (b) KEEPS the MW/SEW weights COMPRESSED (block dtype resident, smaller than
//       the dequant-to-f32 image) while the small V/ET tensors dequant;
//   (c) load -> Forward works END-TO-END at tiny shape (finite, deterministic
//       logits over the assembled W7 CPU composition).
// RED-first by construction: a missing tensor, a leftover tensor, or forcing
// dequant-instead-of-keep each changes the outcome.
//
// HONEST 3-state: this tiny-synthetic load->forward is DERIVED + BUILD-VERIFIED.
// The real 91 GiB `UD-IQ2_XXS` load + generate stays W8-final (download + DGX);
// the name-map coverage vs the real 1328-tensor manifest is gated separately by
// scripts/check-dsv4-gguf-namemap.py. The IQ2_XXS/IQ3_XXS/Q2_K keep-quant vec_dot
// is gated by tests/vt/test_ops_quant_dot.cpp — the tower WIRING here is
// quant-type-agnostic (it routes by role via HasQuantDotKernel), so Q8_0 (a real
// keep-quant type, block 32, easy to synthesize with meaningful values) exercises
// the identical routing/residency/accounting path the i-quants take.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/merged_gemm.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using gguf_test::F32ArrayKv;
using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::I32ArrayKv;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

namespace {

// ── tiny geometry (Q8_0 block = 32, so every keep-quant K is a multiple of 32) ──
struct Dims {
  int64_t H = 32, vocab = 16;
  int64_t nh = 2, head_dim = 32, rope = 8;   // nope = 24
  int64_t qlr = 32, olr = 32, o_groups = 2;  // wo_a [o_groups*olr, H], wo_b [H, o_groups*olr]
  int64_t E = 4, used = 2, mi = 32;
  int64_t hc = 2, sinkhorn = 3;
  int64_t inh = 2, ihd = 32, index_topk = 3;
  int64_t n_layer = 4, n_hash = 2;
  // KV-DSV4-MULTICACHE W5 (#2323): the SWA window, in tokens. Parameterized
  // because a window CONFOUNDS the paging gate -- with a window the paged arm
  // legitimately differs from the unwindowed full-recompute path, so a case that
  // wants to isolate the paging must switch it off. 4 keeps every existing case
  // byte-identical.
  int64_t sliding_window = 4;
  std::vector<int32_t> compress_ratios = {0, 4, 2, 4};  // idx {1,3}, comp {1,2,3}
  // The DSA compressor's REAL output width is `coff * head_dim` (ds4: coff==2 for
  // compress_ratio==4). Default 1 collapses it onto head_dim (the historical tiny
  // shape that HID the real-geometry forward bug); a real-ish gate sets it to 2 so
  // the compressor projections are [2*head_dim, H] (comp_width != head_dim).
  int64_t comp_coff = 1;
};

int64_t Prod(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// Little-endian f32 bytes for `n` values from fill(idx).
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

// Encode a torch [out,in] (in % 32 == 0) row-major float image into Q8_0 blocks
// (block_q8_0 = { f16 d; int8 qs[32] } = 34 B), the ggml storage order.
template <typename F>
std::string Q8Data(int64_t out, int64_t in, F fill) {
  VT_CHECK(in % 32 == 0, "Q8Data: in must be a multiple of 32");
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

std::vector<uint64_t> GgmlDims(const std::vector<int64_t>& torch_shape) {
  std::vector<uint64_t> d;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it)
    d.push_back(static_cast<uint64_t>(*it));
  return d;
}

// A smooth, nonzero-amax fill so Q8_0 has a real scale and the forward is finite.
float WFill(int64_t i) { return 0.05f * static_cast<float>((i % 13) - 6); }

template <typename F>
void AddF32(GgufModelBuilder& b, const std::string& name,
            const std::vector<int64_t>& torch_shape, F fill) {
  b.AddTensor(name, GgmlDims(torch_shape), /*F32=*/0, F32Data(Prod(torch_shape), fill));
}
void AddQ8(GgufModelBuilder& b, const std::string& name,
           const std::vector<int64_t>& torch_shape) {
  const int64_t out = torch_shape.size() == 3 ? torch_shape[0] * torch_shape[1]
                                              : torch_shape[0];
  const int64_t in = torch_shape.back();
  b.AddTensor(name, GgmlDims(torch_shape), /*Q8_0=*/8, Q8Data(out, in, WFill));
}

std::string Blk(int64_t l, const std::string& s) {
  return "blk." + std::to_string(l) + "." + s;
}

// Build the tiny synthetic deepseek4 GGUF. `drop` omits one tensor name (RED:
// unaccounted -> throw); `extra` adds a bogus tensor (RED: leftover -> throw).
// `ds4_flavor` reproduces the three real-GGUF quirks the antirez q2-imatrix file
// carries that a tiny synthetic otherwise misses (all verified from the shipped
// header): the clamped-SwiGLU limit is a per-layer f32 ARRAY `swiglu_clamp_exp`
// (not the scalar `swiglu_clamp`); `compress_ratios` is block_count+1 long (the
// trailing MTP/nextn entry); and the hash `ffn_gate_tid2eid` is stored as ggml
// I32 (type 26), not F32.
// `q8_embed` stores `token_embd.weight` BLOCK-QUANTIZED (Q8_0) instead of F32.
// Every published deepseek4 checkpoint does exactly that, and F32 is the ONE
// encoding the shared residency policy still routes `kExpandBf16` for a
// `kEmbeddingTable`, so the F32 default hid the W6a regression (#1989 review
// F1) rather than covering it.
std::string BuildGguf(const Dims& d, const std::string& drop = "",
                      const std::string& extra = "", bool ds4_flavor = false,
                      bool q8_embed = false) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "deepseek4"));
  const std::string p = "deepseek4.";
  b.AddKv(U32Kv(p + "embedding_length", d.H));
  b.AddKv(U32Kv(p + "block_count", d.n_layer));
  b.AddKv(U32Kv(p + "attention.head_count", d.nh));
  b.AddKv(U32Kv(p + "attention.head_count_kv", 1));
  b.AddKv(U32Kv(p + "attention.key_length", d.head_dim));
  b.AddKv(U32Kv(p + "rope.dimension_count", d.rope));
  b.AddKv(U32Kv(p + "attention.q_lora_rank", d.qlr));
  b.AddKv(U32Kv(p + "attention.output_lora_rank", d.olr));
  b.AddKv(U32Kv(p + "attention.output_group_count", d.o_groups));
  b.AddKv(U32Kv(p + "attention.sliding_window", static_cast<uint32_t>(d.sliding_window)));
  b.AddKv(F32Kv(p + "rope.freq_base", 10000.0f));
  b.AddKv(F32Kv(p + "attention.compress_rope_freq_base", 160000.0f));
  b.AddKv(F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-6f));
  b.AddKv(U32Kv(p + "expert_count", d.E));
  b.AddKv(U32Kv(p + "expert_used_count", d.used));
  b.AddKv(U32Kv(p + "expert_shared_count", 1));
  b.AddKv(U32Kv(p + "expert_feed_forward_length", d.mi));
  b.AddKv(U32Kv(p + "hash_layer_count", d.n_hash));
  if (ds4_flavor) {
    std::vector<float> clamp_exp(static_cast<size_t>(d.n_layer), 10.0f);
    b.AddKv(F32ArrayKv(p + "swiglu_clamp_exp", clamp_exp));
  } else {
    b.AddKv(F32Kv(p + "swiglu_clamp", 10.0f));
  }
  b.AddKv(U32Kv(p + "hyper_connection.count", d.hc));
  b.AddKv(U32Kv(p + "hyper_connection.sinkhorn_iterations", d.sinkhorn));
  b.AddKv(F32Kv(p + "hyper_connection.epsilon", 1e-6f));
  b.AddKv(U32Kv(p + "attention.indexer.head_count", d.inh));
  b.AddKv(U32Kv(p + "attention.indexer.key_length", d.ihd));
  b.AddKv(U32Kv(p + "attention.indexer.top_k", d.index_topk));
  {
    std::vector<int32_t> cr = d.compress_ratios;
    if (ds4_flavor) cr.push_back(0);  // trailing MTP/nextn entry -> length n_layer+1
    b.AddKv(I32ArrayKv(p + "attention.compress_ratios", cr));
  }

  const int64_t hcf = (2 + d.hc) * d.hc;  // MHC fn out rows

  auto add = [&](const std::string& name, bool q8,
                 const std::vector<int64_t>& shape) {
    if (name == drop) return;
    if (q8) {
      AddQ8(b, name, shape);
    } else {
      AddF32(b, name, shape, WFill);
    }
  };

  // model level.
  add("token_embd.weight", q8_embed, {d.vocab, d.H});
  add("output.weight", true, {d.vocab, d.H});
  add("output_norm.weight", false, {d.H});
  add("output_hc_base.weight", false, {d.hc});
  add("output_hc_fn.weight", false, {d.hc, d.hc * d.H});
  add("output_hc_scale.weight", false, {1});

  for (int64_t l = 0; l < d.n_layer; ++l) {
    const int64_t cr = d.compress_ratios[static_cast<size_t>(l)];
    const bool has_comp = cr != 0;
    const bool has_idx = cr == 4;
    const bool is_hash = l < d.n_hash;
    // MLA (Q8_0 MW) + norms/sink (F32 V).
    add(Blk(l, "attn_q_a.weight"), true, {d.qlr, d.H});
    add(Blk(l, "attn_q_b.weight"), true, {d.nh * d.head_dim, d.qlr});
    add(Blk(l, "attn_kv.weight"), true, {d.head_dim, d.H});
    add(Blk(l, "attn_output_a.weight"), true, {d.o_groups * d.olr, d.nh * d.head_dim / d.o_groups});
    add(Blk(l, "attn_output_b.weight"), true, {d.H, d.o_groups * d.olr});
    add(Blk(l, "attn_norm.weight"), false, {d.H});
    add(Blk(l, "attn_q_a_norm.weight"), false, {d.qlr});
    add(Blk(l, "attn_kv_a_norm.weight"), false, {d.head_dim});
    add(Blk(l, "attn_sinks.weight"), false, {d.nh});
    add(Blk(l, "ffn_norm.weight"), false, {d.H});
    // MHC per-layer mixing (F32 V).
    add(Blk(l, "hc_attn_base.weight"), false, {hcf});
    add(Blk(l, "hc_attn_fn.weight"), false, {hcf, d.hc * d.H});
    add(Blk(l, "hc_attn_scale.weight"), false, {3});
    add(Blk(l, "hc_ffn_base.weight"), false, {hcf});
    add(Blk(l, "hc_ffn_fn.weight"), false, {hcf, d.hc * d.H});
    add(Blk(l, "hc_ffn_scale.weight"), false, {3});
    // MoE.
    add(Blk(l, "ffn_gate_inp.weight"), true, {d.E, d.H});
    add(Blk(l, "ffn_gate_exps.weight"), true, {d.E, d.mi, d.H});
    add(Blk(l, "ffn_up_exps.weight"), true, {d.E, d.mi, d.H});
    add(Blk(l, "ffn_down_exps.weight"), true, {d.E, d.H, d.mi});
    add(Blk(l, "ffn_gate_shexp.weight"), true, {d.mi, d.H});
    add(Blk(l, "ffn_up_shexp.weight"), true, {d.mi, d.H});
    add(Blk(l, "ffn_down_shexp.weight"), true, {d.H, d.mi});
    if (is_hash) {
      // tid2eid holds expert ids in [0, E). The real GGUF stores it as ggml I32
      // (type 26); the tiny synthetic default uses F32 (rounded to int at load).
      const std::string nm = Blk(l, "ffn_gate_tid2eid.weight");
      if (nm != drop) {
        if (ds4_flavor) {
          const int64_t n = d.vocab * d.used;
          std::string data;
          data.reserve(static_cast<size_t>(n) * 4);
          for (int64_t i = 0; i < n; ++i) {
            const auto v = static_cast<int32_t>(i % d.E);
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            for (int k = 0; k < 4; ++k)
              data.push_back(static_cast<char>((bits >> (8 * k)) & 0xff));
          }
          b.AddTensor(nm, GgmlDims({d.vocab, d.used}), /*I32=*/26, data);
        } else {
          AddF32(b, nm, {d.vocab, d.used},
                 [&](int64_t i) { return static_cast<float>(i % d.E); });
        }
      }
    } else {
      add(Blk(l, "exp_probs_b.bias"), false, {d.E});
    }
    // DSA compressor (compress_ratio != 0) + indexer (== 4). The compressor
    // projects to comp_width = comp_coff * head_dim (real: coff==2); the norm is
    // over head_dim (ds4 `ds4.c:5021`), NOT comp_width.
    if (has_comp) {
      const int64_t cw = d.comp_coff * d.head_dim;
      add(Blk(l, "attn_compressor_ape.weight"), false, {cr, cw});
      add(Blk(l, "attn_compressor_gate.weight"), true, {cw, d.H});
      add(Blk(l, "attn_compressor_kv.weight"), true, {cw, d.H});
      add(Blk(l, "attn_compressor_norm.weight"), false, {d.head_dim});
    }
    if (has_idx) {
      add(Blk(l, "indexer.attn_q_b.weight"), true, {d.inh * d.ihd, d.H});
      add(Blk(l, "indexer.proj.weight"), false, {d.inh, d.H});
      add(Blk(l, "indexer_compressor_ape.weight"), false, {cr, d.ihd});
      add(Blk(l, "indexer_compressor_gate.weight"), true, {d.ihd, d.H});
      add(Blk(l, "indexer_compressor_kv.weight"), true, {d.ihd, d.H});
      add(Blk(l, "indexer_compressor_norm.weight"), false, {d.ihd});
    }
  }
  if (!extra.empty()) AddF32(b, extra, {d.H}, WFill);
  return b.Build();
}

// keep-quant policy (deterministic, independent of the platform probe).
vllm::GgufLoadPolicy KeepPolicy() {
  vllm::GgufLoadPolicy pol;
  pol.keep_quant = true;
  return pol;
}
const vllm::GgufLoadPolicy kExpandAll;  // all defaults -> dequant everything

// Relative L2 between two equal-length logit vectors.
double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double dd = static_cast<double>(a[i]) - b[i];
    num += dd * dd;
    den += static_cast<double>(a[i]) * a[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

TEST_CASE("DeepseekV4ParamsFromGguf resolves the deepseek4 hparams") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::DeepseekV4Params p = vllm::DeepseekV4ParamsFromGguf(g);
  CHECK(p.hidden_size == 32);
  CHECK(p.num_hidden_layers == 4);
  CHECK(p.head_dim == 32);
  CHECK(p.q_lora_rank == 32);
  CHECK(p.o_groups == 2);
  CHECK(p.n_routed_experts == 4);
  CHECK(p.num_experts_per_tok == 2);
  CHECK(p.hc_mult == 2);
  CHECK(p.num_hash_layers == 2);
  CHECK(p.index_topk == 3);
  CHECK(p.vocab_size == 16);
  REQUIRE(p.compress_ratios.size() == 4);
  CHECK(p.compress_ratios[1] == 4);
  CHECK(p.has_indexer(1));
  CHECK(p.has_indexer(3));
  CHECK_FALSE(p.has_indexer(0));
  CHECK(p.has_compressor(2));
  CHECK(p.is_hash_layer(1));
  CHECK_FALSE(p.is_hash_layer(2));
}

TEST_CASE("LoadDeepseekV4FromGguf: accounts for every tensor into a tower slot") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol);

  // Total = 6 top-level + Σ layers (layer0 24, layer1 34, layer2 28, layer3 34)
  // = 126, and the loader consumed EXACTLY the file's tensor set (no leftover).
  CHECK(w.accounted_tensors == 126);
  CHECK(w.accounted_tensors == static_cast<int64_t>(g.Tensors().size()));
  CHECK(w.has_gguf_weights);
  CHECK(w.has_host_weights);
  REQUIRE(w.gguf.layers.size() == 4);
  // Per-layer topology mapped through: layer1 has both indexer + compressor +
  // hash; layer2 gated + compressor, no indexer; layer0 hash, no comp/idx.
  CHECK(w.gguf.layers[0].is_hash);
  CHECK_FALSE(w.gguf.layers[0].has_compressor);
  CHECK(w.gguf.layers[1].has_indexer);
  CHECK(w.gguf.layers[1].has_compressor);
  CHECK_FALSE(w.gguf.layers[2].has_indexer);
  CHECK(w.gguf.layers[2].has_compressor);
  // Non-empty slots (materialized, not left as empty OwnedTensors).
  CHECK_FALSE(w.gguf.embed.Empty());
  CHECK_FALSE(w.gguf.lm_head.Empty());
  CHECK_FALSE(w.gguf.layers[0].wq_a.Empty());
  CHECK_FALSE(w.gguf.layers[0].moe_down_exps.Empty());
  CHECK_FALSE(w.gguf.layers[1].idx_wq_b.Empty());
  CHECK_FALSE(w.gguf.layers[2].comp_wgate.Empty());
}

TEST_CASE("LoadDeepseekV4FromGguf: keep-quant blocks stay COMPRESSED") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);

  // The 256-expert down proj (SEW) keeps its Q8_0 blocks: block dtype resident,
  // shape [E*out, in], nk = true, and byte size = compressed (34 B / 32 elems),
  // strictly smaller than the dequant-to-f32 image (4 B / elem).
  const vllm::OwnedTensor& dn = w.gguf.layers[3].moe_down_exps;
  CHECK(dn.dtype == vt::DType::kQ8_0);
  CHECK(vt::IsBlockQuant(dn.dtype));
  CHECK(dn.nk);
  const int64_t rows = d.E * d.H;   // E*out
  const int64_t k = d.mi;           // in
  CHECK(dn.shape[0] == rows);
  CHECK(dn.shape[1] == k);
  const int64_t elems = rows * k;
  CHECK(static_cast<int64_t>(dn.bytes.size()) == rows * (k / 32) * 34);
  CHECK(static_cast<int64_t>(dn.bytes.size()) < elems * 4);  // < dequant-f32
  // An MLA linear (MW) also keeps quant; a norm (V) is dequant f32.
  CHECK(w.gguf.layers[0].wq_a.dtype == vt::DType::kQ8_0);
  CHECK(w.gguf.lm_head.dtype == vt::DType::kQ8_0);
  CHECK(w.gguf.layers[0].attn_norm.dtype == vt::DType::kF32);
  CHECK(w.gguf.layers[0].attn_sink.dtype == vt::DType::kF32);

  // RED-first: dequant-instead-of-keep (an expand policy) leaves the SAME weight
  // UNCOMPRESSED (bf16), proving keep-quant is what compresses it.
  const vllm::DeepseekV4Weights we =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &kExpandAll);
  CHECK(we.gguf.layers[3].moe_down_exps.dtype == vt::DType::kBF16);
  CHECK(static_cast<int64_t>(we.gguf.layers[3].moe_down_exps.bytes.size()) == elems * 2);
}

// ── #1989 review F1: the gather table's residency is a SHARED policy ─────────
// MODEL-MM-QWEN4-EXP W6a made `GgufTensorRole::kEmbeddingTable` keep-quant
// ELIGIBLE. THREE loaders route that role and only one was updated with it.
// DeepSeek-V4 is one of the two that were not: it gathers `token_embd` as a
// FLAT HOST f32 array (deepseek_v4.cpp:1844 indexes `hw.embed[tok * H + h]`)
// and, when the file is tied, hands the same f32 image to the final projection,
// so it cannot read a table that keeps its ggml blocks. The load THREW
// ("deepseek-v4 gguf: a gather-table tensor must not keep quant blocks:
// token_embd.weight") on every published checkpoint, and the suite stayed green
// only because the one fixture in this file stored `token_embd` as F32 — the
// single encoding that still routes `kExpandBf16` for that role.
//
// The invariant pinned here is stronger than "does not throw": for THIS model
// the gather table's CONTENT may not depend on the residency policy at all, so
// the keep-quant load's table is byte-identical to the expand-everything load's.
TEST_CASE("LoadDeepseekV4FromGguf: a BLOCK-QUANTIZED token_embd still loads") {
  Dims d;
  TempFile f(BuildGguf(d, /*drop=*/"", /*extra=*/"", /*ds4_flavor=*/false,
                       /*q8_embed=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  // Non-vacuity, both halves: the fixture really is block-quantized, AND the
  // shared policy really does elect to KEEP it. Without the second REQUIRE this
  // case would pass on a policy that had quietly stopped keeping gather tables.
  REQUIRE(g.Get("token_embd.weight").ggml_type == 8u);  // ggml Q8_0
  REQUIRE(vllm::PeekRoute(KeepPolicy(), g.Get("token_embd.weight"),
                          vllm::GgufTensorRole::kEmbeddingTable) ==
          vllm::GgufResidency::kKeepQuant);

  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights wk =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  const vllm::DeepseekV4Weights we =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &kExpandAll);

  CHECK(wk.gguf.embed.dtype == vt::DType::kF32);
  REQUIRE(wk.gguf.embed.rank == 2);
  CHECK(wk.gguf.embed.shape[0] == d.vocab);
  CHECK(wk.gguf.embed.shape[1] == d.H);
  REQUIRE(wk.gguf.embed.bytes.size() == we.gguf.embed.bytes.size());
  CHECK(std::memcmp(wk.gguf.embed.bytes.data(), we.gguf.embed.bytes.data(),
                    wk.gguf.embed.bytes.size()) == 0);
  // ...and the table is not silently empty, which would satisfy the equality
  // above without carrying a single weight.
  REQUIRE(wk.host.embed.size() == static_cast<size_t>(d.vocab * d.H));
  CHECK(std::any_of(wk.host.embed.begin(), wk.host.embed.end(),
                    [](float v) { return v != 0.0f; }));
  // The GEMM tower is untouched by the narrowing: an untied head still keeps
  // its blocks, so this is a role-scoped answer and not "keep-quant off".
  CHECK(wk.gguf.lm_head.dtype == vt::DType::kQ8_0);
  CHECK(wk.gguf.layers[0].wq_a.dtype == vt::DType::kQ8_0);

  // Reached from the production forward, not only from the loader.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<int32_t> tokens = {1, 5, 9};
  const std::vector<int32_t> positions = {0, 1, 2};
  const std::vector<float> lk =
      vllm::DeepseekV4ForwardGguf(wk, q, tokens, positions);
  REQUIRE(lk.size() == tokens.size() * static_cast<size_t>(d.vocab));
  for (float v : lk) CHECK(std::isfinite(v));
}

// The TIED file is the second `kEmbeddingTable` call site in this loader, and it
// is a GEMM operand rather than a gather — the review's "the two sites need
// DIFFERENT answers". Both must still resolve to the f32 expansion this model's
// forward consumes.
TEST_CASE("LoadDeepseekV4FromGguf: a TIED block-quantized token_embd still loads") {
  Dims d;
  TempFile f(BuildGguf(d, /*drop=*/"output.weight", /*extra=*/"",
                       /*ds4_flavor=*/false, /*q8_embed=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights wk =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);

  CHECK(wk.gguf.embed.dtype == vt::DType::kF32);
  CHECK(wk.gguf.lm_head.dtype == vt::DType::kF32);
  REQUIRE(wk.gguf.lm_head.bytes.size() == wk.gguf.embed.bytes.size());
  CHECK(std::memcmp(wk.gguf.lm_head.bytes.data(), wk.gguf.embed.bytes.data(),
                    wk.gguf.embed.bytes.size()) == 0);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<int32_t> tokens = {1, 5, 9};
  const std::vector<int32_t> positions = {0, 1, 2};
  const std::vector<float> lk =
      vllm::DeepseekV4ForwardGguf(wk, q, tokens, positions);
  REQUIRE(lk.size() == tokens.size() * static_cast<size_t>(d.vocab));
  for (float v : lk) CHECK(std::isfinite(v));
}

// ── W2C: the forward CONSUMES the keep-quant blocks (no f32 tower) ────────────
// DeepseekV4ForwardGguf runs the whole composition with the big MLA/MoE/lm_head
// GEMMs reading the COMPRESSED `w.gguf` blocks via vt::MatmulBT -> kMatmulBTQuant.
// Gate: (a) it runs (finite, deterministic); (b) keep-quant (Q8_0 vec_dot) agrees
// with the dequant oracle (bf16 MatmulBT, the expand-all load) within a near-tie;
// (c) RED-first — a wrong route (a deliberate miswire) DIVERGES.
TEST_CASE("DeepseekV4ForwardGguf: keep-quant forward runs + near-tie vs dequant (W2C)") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights wk =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  const vllm::DeepseekV4Weights we =  // dequant oracle: bf16 OwnedTensors
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &kExpandAll);

  // Prove the keep-quant tower REALLY kept its blocks and the oracle expanded.
  REQUIRE(wk.gguf.layers[0].wq_a.dtype == vt::DType::kQ8_0);
  REQUIRE(we.gguf.layers[0].wq_a.dtype == vt::DType::kBF16);

  const std::vector<int32_t> tokens = {1, 5, 9};
  const std::vector<int32_t> positions = {0, 1, 2};
  const std::vector<float> lk =
      vllm::DeepseekV4ForwardGguf(wk, q, tokens, positions);  // Q8_0 blocks
  const std::vector<float> le =
      vllm::DeepseekV4ForwardGguf(we, q, tokens, positions);  // bf16 dequant

  REQUIRE(lk.size() == tokens.size() * static_cast<size_t>(d.vocab));
  for (float v : lk) CHECK(std::isfinite(v));

  // (b) NEAR-TIE: the only difference between the two is that the keep path also
  // quantizes the ACTIVATION to the weight's q8_0 vec_dot_type (8-bit, per-32
  // block) before the integer dot; the oracle keeps f32 activations. Over the 4
  // deep layers that compounds, but stays a near-tie. MEASURED RelL2 = 0.0116;
  // the 0.05 band is ~4x margin (routing-flip-robust). See docs/BENCHMARKS.md.
  CHECK(RelL2(lk, le) < 0.05);

  // (a) DETERMINISTIC: the same loaded tower re-runs bit-identically.
  const std::vector<float> lk2 =
      vllm::DeepseekV4ForwardGguf(wk, q, tokens, positions);
  REQUIRE(lk2.size() == lk.size());
  for (size_t i = 0; i < lk.size(); ++i) CHECK(lk2[i] == lk[i]);

  // (c) RED-first: a deliberately-miswired interleave (drop the per-head attention
  // sink) CHANGES the keep-quant output — proving the route is load-bearing, not a
  // constant the quant path happens to reproduce.
  const std::vector<float> lk_miswire = vllm::DeepseekV4ForwardGguf(
      wk, q, tokens, positions, /*logits_indices=*/{}, vllm::V4Miswire::kNoAttnSink);
  CHECK(RelL2(lk, lk_miswire) > 1e-4);
}

// ── Stage 1: incremental-decode KV cache == full-recompute (token-identical) ────
// The MLA latent cache makes decode process ONE new token against cached KV rather
// than re-running the whole forward over the growing context. deck[t] depends only
// on token t + its position (MHC-pre is per-token; no cross-token mixing), so the
// cached latent equals the recomputed one and the greedy sequences MUST match. This
// is a pure equivalence — same tokens, ~ctx x fewer FLOPs. RED-first: a cache that
// forgets its history (reset each step) diverges, proving the cached KV load-bearing.
TEST_CASE("W5: PAGED incremental decode == full-recompute, token for token") {
  // `KV-DSV4-MULTICACHE` W5 (#2323). The end-to-end gate for the paged arm: a
  // greedy generation driven through `DeepseekV4ForwardGgufPaged`, one step at a
  // time over the runner's page layout, must produce the SAME TOKENS as
  // re-passing the whole growing context.
  //
  // THIS IS THE CLAIM THAT MATTERS. The op-level cases prove the attention math
  // and the causal mapping in isolation; this proves the whole step -- latent
  // write into pages, block/slot arithmetic, `kv_base` bookkeeping across steps,
  // and the read back -- composes into the same model. A paging defect that the
  // per-op cases cannot see (a slot written to the wrong block, a `kv_base` that
  // drifts by one after several steps) shows up here as a DIFFERENT TOKEN.
  // NO SLIDING WINDOW, deliberately. With one, the paged arm attends a window on
  // the `compress_ratio <= 1` layers while the full-recompute anchor attends the
  // whole prefix -- they diverge above the window BY DESIGN, and mixing that into
  // this case would hide whether the PAGING is right. The window has its own gate
  // in `test_deepseek_v4_paged_equiv`, against a windowed reference.
  Dims d;
  d.sliding_window = 0;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  REQUIRE(w.has_gguf_weights);
  REQUIRE(w.params.sliding_window == 0);

  const int V = static_cast<int>(d.vocab);
  auto argmax_last = [V](const std::vector<float>& logits) -> int {
    const size_t rows = logits.size() / static_cast<size_t>(V);
    const float* row = logits.data() + (rows - 1) * static_cast<size_t>(V);
    int best = 0;
    for (int j = 1; j < V; ++j)
      if (row[j] > row[best]) best = j;
    return best;
  };

  const std::vector<int32_t> prompt = {1, 5, 9};
  const int N = 6;

  // (A) full-recompute greedy — the anchor.
  std::vector<int32_t> gen_full;
  {
    std::vector<int32_t> tok = prompt;
    for (int s = 0; s < N; ++s) {
      std::vector<int32_t> pos(tok.size());
      for (size_t i = 0; i < tok.size(); ++i) pos[i] = static_cast<int32_t>(i);
      const auto lg = vllm::DeepseekV4ForwardGguf(
          w, q, tok, pos, {static_cast<int32_t>(tok.size() - 1)});
      const int nx = argmax_last(lg);
      gen_full.push_back(nx);
      tok.push_back(nx);
    }
  }

  // (B) PAGED incremental greedy.
  std::vector<int32_t> gen_paged;
  {
    const int64_t nlayers = w.params.num_hidden_layers;
    const int64_t hd = w.params.head_dim;
    const int64_t block_size = 8;
    // Enough pages for the prompt plus every generated token.
    const int64_t max_tokens = static_cast<int64_t>(prompt.size()) + N + 1;
    const int64_t num_blocks = (max_tokens + block_size - 1) / block_size;
    std::vector<std::vector<float>> storage(static_cast<size_t>(nlayers));
    std::vector<vt::Tensor> pages(static_cast<size_t>(nlayers));
    for (int64_t l = 0; l < nlayers; ++l) {
      storage[static_cast<size_t>(l)].assign(
          static_cast<size_t>(num_blocks * block_size * hd), 0.0f);
      pages[static_cast<size_t>(l)] = vt::Tensor::Contiguous(
          storage[static_cast<size_t>(l)].data(), vt::DType::kF32, q.device,
          {num_blocks, block_size, hd});
    }

    int64_t kv_base = 0;
    std::vector<int32_t> step = prompt;   // first step is the whole prompt
    for (int s = 0; s < N; ++s) {
      std::vector<int32_t> pos(step.size());
      for (size_t i = 0; i < step.size(); ++i)
        pos[i] = static_cast<int32_t>(kv_base + static_cast<int64_t>(i));
      const auto lg = vllm::DeepseekV4ForwardGgufPaged(
          w, q, pages, kv_base, step, pos,
          {static_cast<int32_t>(step.size() - 1)});
      const int nx = argmax_last(lg);
      gen_paged.push_back(nx);
      kv_base += static_cast<int64_t>(step.size());
      step = {static_cast<int32_t>(nx)};  // every later step is ONE token
    }
  }

  // TOKEN FOR TOKEN. Not a tolerance: paging must not change a value, so it must
  // not change a decision.
  REQUIRE(gen_paged.size() == gen_full.size());
  CHECK(gen_paged == gen_full);
  // And the generation is NON-DEGENERATE -- an all-same-token run would match
  // trivially and prove nothing about either path.
  bool varied = false;
  for (size_t i = 1; i < gen_full.size(); ++i)
    if (gen_full[i] != gen_full[0]) varied = true;
  CHECK(varied);
}

TEST_CASE("DeepseekV4ForwardGgufCached: incremental decode == full-recompute (Stage 1)") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  REQUIRE(w.has_gguf_weights);

  const int V = static_cast<int>(d.vocab);
  auto argmax_last = [V](const std::vector<float>& logits) -> int {
    const size_t rows = logits.size() / static_cast<size_t>(V);
    const float* row = logits.data() + (rows - 1) * static_cast<size_t>(V);
    int best = 0;
    for (int j = 1; j < V; ++j)
      if (row[j] > row[best]) best = j;
    return best;
  };

  const std::vector<int32_t> prompt = {1, 5, 9};
  const int N = 6;

  // (A) full-recompute greedy — the anchor (re-passes the whole growing context).
  std::vector<int32_t> gen_full;
  {
    std::vector<int32_t> tok = prompt;
    for (int s = 0; s < N; ++s) {
      std::vector<int32_t> pos(tok.size());
      for (size_t i = 0; i < tok.size(); ++i) pos[i] = static_cast<int32_t>(i);
      const auto lg = vllm::DeepseekV4ForwardGguf(
          w, q, tok, pos, {static_cast<int32_t>(tok.size() - 1)});
      const int nx = argmax_last(lg);
      gen_full.push_back(nx);
      tok.push_back(nx);
    }
  }

  // (B) incremental greedy — prefill fills the cache, each decode step is ONE token.
  std::vector<int32_t> gen_inc;
  std::vector<float> prefill_full;   // for a direct bit-equivalence check at prefill
  std::vector<float> prefill_cached;
  {
    vllm::DeepseekV4KvCache cache;
    std::vector<int32_t> pos(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i) pos[i] = static_cast<int32_t>(i);
    auto lg = vllm::DeepseekV4ForwardGgufCached(
        w, q, cache, prompt, pos, {static_cast<int32_t>(prompt.size() - 1)});
    prefill_cached = lg;
    prefill_full = vllm::DeepseekV4ForwardGguf(
        w, q, prompt, pos, {static_cast<int32_t>(prompt.size() - 1)});
    int nx = argmax_last(lg);
    gen_inc.push_back(nx);
    for (int s = 1; s < N; ++s) {
      lg = vllm::DeepseekV4ForwardGgufCached(
          w, q, cache, {nx}, {static_cast<int32_t>(cache.len)}, {0});
      nx = argmax_last(lg);
      gen_inc.push_back(nx);
    }
  }

  // Prefill is the SAME batch both ways → BIT-identical logits.
  REQUIRE(prefill_full.size() == prefill_cached.size());
  for (size_t i = 0; i < prefill_full.size(); ++i) CHECK(prefill_cached[i] == prefill_full[i]);

  // EQUIVALENCE over the whole greedy run: identical token sequences.
  REQUIRE(gen_full.size() == gen_inc.size());
  REQUIRE(gen_inc.size() == static_cast<size_t>(N));  // non-vacuous: real decode steps
  for (size_t i = 0; i < gen_full.size(); ++i) CHECK(gen_inc[i] == gen_full[i]);

  // Cache length tracks the tokens consumed (prompt + N-1 appended decode tokens).
  // (each decode step appends one; prefill appended prompt.size()).

  // RED-first: a cache that FORGETS its history (fresh cache each decode step, pos 0)
  // must DIVERGE from the full-context anchor — the cached KV is load-bearing.
  std::vector<int32_t> gen_broken;
  {
    vllm::DeepseekV4KvCache cache;
    std::vector<int32_t> pos(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i) pos[i] = static_cast<int32_t>(i);
    auto lg = vllm::DeepseekV4ForwardGgufCached(
        w, q, cache, prompt, pos, {static_cast<int32_t>(prompt.size() - 1)});
    int nx = argmax_last(lg);
    gen_broken.push_back(nx);
    for (int s = 1; s < N; ++s) {
      vllm::DeepseekV4KvCache fresh;  // loses all prior KV + causal context
      lg = vllm::DeepseekV4ForwardGgufCached(w, q, fresh, {nx}, {0}, {0});
      nx = argmax_last(lg);
      gen_broken.push_back(nx);
    }
  }
  bool any_diff = false;
  for (size_t i = 0; i < gen_broken.size(); ++i)
    if (gen_broken[i] != gen_full[i]) any_diff = true;
  CHECK(any_diff);
}

// ── Re-scoped Stage 2: grouped keep-quant MoE GEMM == per-expert loop ──────────
// vt::MatmulBTQuantGrouped collapses the routed experts' per-expert kMatmulBTQuant
// matvecs into one call. It is the SAME arithmetic, so grouped output MUST be
// BYTE-IDENTICAL to the per-expert loop over the same stacked block weight.
// RED-first: permuting the expert-index map changes the grouped output.
TEST_CASE("MatmulBTQuantGrouped: grouped == per-expert loop (re-scoped Stage 2)") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  REQUIRE(w.has_gguf_weights);

  // Stacked routed gate weight [E*mi, H], block-quant (Q8_0 keep-quant).
  const vllm::OwnedTensor& wexp = w.gguf.layers[0].moe_gate_exps;
  REQUIRE(vt::IsBlockQuant(wexp.dtype));
  const int64_t N = d.mi, K = d.H, E = wexp.shape[0] / d.mi;
  REQUIRE(E >= 3);

  const int P = 3;
  const std::vector<int32_t> eids = {2, 0, 1};  // selected experts (arbitrary)
  std::vector<float> act(static_cast<size_t>(P) * K);
  for (size_t i = 0; i < act.size(); ++i)
    act[i] = 0.01f * static_cast<float>((i * 7 + 3) % 19) - 0.09f;

  auto row_bytes = vt::RowSizeBytes(wexp.dtype, K);
  // (A) grouped: one vt::MatmulBTQuantGrouped call.
  std::vector<float> og(static_cast<size_t>(P) * N);
  {
    std::vector<int32_t> ids = eids;
    vt::Tensor a = vt::Tensor::Contiguous(act.data(), vt::DType::kF32, q.device, {P, K});
    vt::Tensor o = vt::Tensor::Contiguous(og.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor eid = vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
    vt::Tensor wt = wexp.View();
    vt::MatmulBTQuantGrouped(q, o, a, wt, eid);
  }
  // (B) per-expert: vt::MatmulBTQuant on each expert's row-slice.
  std::vector<float> op(static_cast<size_t>(P) * N);
  for (int p = 0; p < P; ++p) {
    vt::Tensor a = vt::Tensor::Contiguous(act.data() + static_cast<size_t>(p) * K,
                                          vt::DType::kF32, q.device, {1, K});
    vt::Tensor o = vt::Tensor::Contiguous(op.data() + static_cast<size_t>(p) * N,
                                          vt::DType::kF32, q.device, {1, N});
    vt::Tensor wt{};
    wt.data = const_cast<uint8_t*>(wexp.bytes.data()) +
              static_cast<size_t>(eids[p]) * N * row_bytes;
    wt.dtype = wexp.dtype;
    wt.device = q.device;
    wt.rank = 2;
    wt.shape[0] = N;
    wt.shape[1] = K;
    wt.stride[0] = K;
    wt.stride[1] = 1;
    vt::MatmulBTQuant(q, o, a, wt);
  }
  // BYTE-IDENTICAL.
  REQUIRE(og.size() == op.size());
  for (size_t i = 0; i < og.size(); ++i) CHECK(og[i] == op[i]);

  // RED-first: a permuted expert map produces a DIFFERENT grouped result (so the
  // expert-index selection is load-bearing, not incidentally equal).
  std::vector<float> og2(static_cast<size_t>(P) * N);
  {
    std::vector<int32_t> ids = {1, 2, 0};  // permuted
    vt::Tensor a = vt::Tensor::Contiguous(act.data(), vt::DType::kF32, q.device, {P, K});
    vt::Tensor o = vt::Tensor::Contiguous(og2.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor eid = vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
    vt::Tensor wt = wexp.View();
    vt::MatmulBTQuantGrouped(q, o, a, wt, eid);
  }
  bool any_diff = false;
  for (size_t i = 0; i < og.size(); ++i)
    if (og2[i] != og[i]) any_diff = true;
  CHECK(any_diff);
}

// Brick 2 (preq-reuse): the routed gate/up feed the SAME hidden to every expert. A
// 1-row BROADCAST activation (quantize x once, read it for all P) MUST be BYTE-IDENTICAL
// to the per-expert path over P physical copies of that row. RED-first: a genuinely
// per-expert activation (rows differ) is NOT equal to the broadcast of row 0.
TEST_CASE("MatmulBTQuantGrouped: 1-row broadcast == replicated activation") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  REQUIRE(w.has_gguf_weights);
  const vllm::OwnedTensor& wexp = w.gguf.layers[0].moe_gate_exps;
  const int64_t N = d.mi, K = d.H, E = wexp.shape[0] / d.mi;
  REQUIRE(E >= 3);

  const int P = 3;
  std::vector<int32_t> eids = {2, 0, 1};
  std::vector<float> x1(static_cast<size_t>(K));  // ONE shared hidden
  for (size_t i = 0; i < x1.size(); ++i)
    x1[i] = 0.013f * static_cast<float>((i * 5 + 1) % 17) - 0.1f;

  // (A) broadcast: act = {1, K}, out = {P, N}.
  std::vector<float> ob(static_cast<size_t>(P) * N);
  {
    std::vector<int32_t> ids = eids;
    vt::Tensor a = vt::Tensor::Contiguous(x1.data(), vt::DType::kF32, q.device, {1, K});
    vt::Tensor o = vt::Tensor::Contiguous(ob.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor eid = vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
    vt::Tensor wt = wexp.View();
    vt::MatmulBTQuantGrouped(q, o, a, wt, eid);
  }
  // (B) replicated: act = {P, K} with every row a copy of x1.
  std::vector<float> rep(static_cast<size_t>(P) * K);
  for (int p = 0; p < P; ++p) std::copy(x1.begin(), x1.end(), rep.begin() + static_cast<size_t>(p) * K);
  std::vector<float> orr(static_cast<size_t>(P) * N);
  {
    std::vector<int32_t> ids = eids;
    vt::Tensor a = vt::Tensor::Contiguous(rep.data(), vt::DType::kF32, q.device, {P, K});
    vt::Tensor o = vt::Tensor::Contiguous(orr.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor eid = vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
    vt::Tensor wt = wexp.View();
    vt::MatmulBTQuantGrouped(q, o, a, wt, eid);
  }
  REQUIRE(ob.size() == orr.size());
  for (size_t i = 0; i < ob.size(); ++i) CHECK(ob[i] == orr[i]);  // BYTE-IDENTICAL

  // RED-first: distinct per-expert rows are NOT the broadcast of row 0.
  std::vector<float> distinct(static_cast<size_t>(P) * K);
  for (int p = 0; p < P; ++p)
    for (int64_t kk = 0; kk < K; ++kk)
      distinct[static_cast<size_t>(p) * K + kk] = x1[static_cast<size_t>(kk)] + 0.001f * (p + 1);
  std::vector<float> od(static_cast<size_t>(P) * N);
  {
    std::vector<int32_t> ids = eids;
    vt::Tensor a = vt::Tensor::Contiguous(distinct.data(), vt::DType::kF32, q.device, {P, K});
    vt::Tensor o = vt::Tensor::Contiguous(od.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor eid = vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
    vt::Tensor wt = wexp.View();
    vt::MatmulBTQuantGrouped(q, o, a, wt, eid);
  }
  bool any_diff = false;
  for (size_t i = 0; i < ob.size(); ++i)
    if (od[i] != ob[i]) any_diff = true;
  CHECK(any_diff);
}

// ── SHARED-OP promotion: kMoeGateUpSwiGLUGrouped == composite golden ──────────
// The fused routed-MoE gate+up+SwiGLU op (promoted from DeepSeek's private
// MoeDeviceKernels::moe_gate_up_swiglu to a first-class vt:: op) MUST be
// BYTE-IDENTICAL to its Tier-0 composite: two vt::MatmulBTQuantGrouped (gate,up)
// + the elementwise clamped-SwiGLU. RED-first: a different `limit` changes output.
namespace {
// The clamped-SwiGLU the fused kernel / CPU golden apply (α=1, β=0).
float ClampedSwiGLU(float g, float u, float limit) {
  const float gate = std::fmin(g, limit);
  const float up = std::fmin(std::fmax(u, -limit), limit);
  return gate * (1.0f / (1.0f + std::exp(-gate))) * up;
}
}  // namespace

TEST_CASE("MoeGateUpSwiGLUGrouped: fused shared op == 2x grouped GEMM + clamped-SwiGLU") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  REQUIRE(w.has_gguf_weights);
  const vllm::OwnedTensor& gate_exps = w.gguf.layers[0].moe_gate_exps;
  const vllm::OwnedTensor& up_exps = w.gguf.layers[0].moe_up_exps;
  REQUIRE(vt::IsBlockQuant(gate_exps.dtype));
  REQUIRE(gate_exps.dtype == up_exps.dtype);
  const int64_t N = d.mi, K = d.H, E = gate_exps.shape[0] / d.mi;
  REQUIRE(E >= 3);

  const int P = 3;
  const std::vector<int32_t> eids = {2, 0, 1};
  std::vector<float> x1(static_cast<size_t>(K));  // ONE shared hidden (broadcast)
  for (size_t i = 0; i < x1.size(); ++i)
    x1[i] = 0.017f * static_cast<float>((i * 5 + 1) % 17) - 0.11f;
  const float limit = 10.0f;

  auto make_a = [&]() {
    return vt::Tensor::Contiguous(x1.data(), vt::DType::kF32, q.device, {1, K});
  };
  auto make_eid = [&](std::vector<int32_t>& ids) {
    return vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
  };

  // (A) fused shared op.
  std::vector<float> ofused(static_cast<size_t>(P) * N);
  {
    std::vector<int32_t> ids = eids;
    vt::Tensor a = make_a();
    vt::Tensor o = vt::Tensor::Contiguous(ofused.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor eid = make_eid(ids);
    vt::Tensor gw = gate_exps.View(), uw = up_exps.View();
    vt::MoeGateUpSwiGLUGrouped(q, o, a, gw, uw, eid, limit);
  }
  // (B) explicit composite: gate GEMM + up GEMM + ClampedSwiGLU (the golden).
  std::vector<float> gg(static_cast<size_t>(P) * N), uu(static_cast<size_t>(P) * N);
  {
    std::vector<int32_t> ids = eids;
    vt::Tensor a = make_a();
    vt::Tensor eid = make_eid(ids);
    vt::Tensor gw = gate_exps.View(), uw = up_exps.View();
    vt::Tensor og = vt::Tensor::Contiguous(gg.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor ou = vt::Tensor::Contiguous(uu.data(), vt::DType::kF32, q.device, {P, N});
    vt::MatmulBTQuantGrouped(q, og, a, gw, eid);
    vt::MatmulBTQuantGrouped(q, ou, a, uw, eid);
  }
  std::vector<float> ocomp(static_cast<size_t>(P) * N);
  for (size_t i = 0; i < ocomp.size(); ++i) ocomp[i] = ClampedSwiGLU(gg[i], uu[i], limit);

  REQUIRE(ofused.size() == ocomp.size());
  for (size_t i = 0; i < ofused.size(); ++i) CHECK(ofused[i] == ocomp[i]);  // BYTE-IDENTICAL

  // RED-first: a different clamp `limit` produces a DIFFERENT fused output (so the
  // epilogue scalar is load-bearing, not incidentally equal).
  std::vector<float> ofused2(static_cast<size_t>(P) * N);
  {
    std::vector<int32_t> ids = eids;
    vt::Tensor a = make_a();
    vt::Tensor o = vt::Tensor::Contiguous(ofused2.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor eid = make_eid(ids);
    vt::Tensor gw = gate_exps.View(), uw = up_exps.View();
    vt::MoeGateUpSwiGLUGrouped(q, o, a, gw, uw, eid, /*limit=*/0.05f);
  }
  bool any_diff = false;
  for (size_t i = 0; i < ofused.size(); ++i)
    if (ofused2[i] != ofused[i]) any_diff = true;
  CHECK(any_diff);
}

// ── DECLARATIVE DESCRIPTOR: vt::MergedGemm(kKeepQuantGateUpSwiGLU) tiering ────
// The merged-GEMM descriptor must realize BYTE-IDENTICALLY across its two tiers:
// the fast op (force_composite=false) and the Tier-0 standalone-op composite
// (force_composite=true). Both equal the manual gate+up GEMM + ClampedSwiGLU. This
// is the "composite == fused, byte-identical" gate the promotion is judged on.
TEST_CASE("MergedGemm descriptor: fast op == Tier-0 composite (byte-identical)") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  REQUIRE(w.has_gguf_weights);
  const vllm::OwnedTensor& gate_exps = w.gguf.layers[0].moe_gate_exps;
  const vllm::OwnedTensor& up_exps = w.gguf.layers[0].moe_up_exps;
  const int64_t N = d.mi, K = d.H, E = gate_exps.shape[0] / d.mi;
  REQUIRE(E >= 3);

  // The realized descriptor names the promoted shared op as its fast_op.
  REQUIRE(vt::kKeepQuantGateUpSwiGLU.arity == 2);
  REQUIRE(vt::kKeepQuantGateUpSwiGLU.epilogue == vt::MergedEpilogue::kSiluMulClamp);
  REQUIRE(vt::kKeepQuantGateUpSwiGLU.fast_op ==
          static_cast<int>(vt::OpId::kMoeGateUpSwiGLUGrouped));

  const int P = 3;
  const std::vector<int32_t> eids = {1, 2, 0};
  std::vector<float> x1(static_cast<size_t>(K));
  for (size_t i = 0; i < x1.size(); ++i)
    x1[i] = 0.021f * static_cast<float>((i * 3 + 2) % 13) - 0.09f;
  const float limit = 10.0f;

  auto run = [&](bool force_composite, std::vector<float>& out) {
    std::vector<int32_t> ids = eids;
    vt::Tensor a = vt::Tensor::Contiguous(x1.data(), vt::DType::kF32, q.device, {1, K});
    vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor eid = vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
    vt::Tensor gw = gate_exps.View(), uw = up_exps.View();
    vt::MergedGemm(q, vt::kKeepQuantGateUpSwiGLU, o, a, gw, uw, eid, limit, force_composite);
  };
  std::vector<float> ofast(static_cast<size_t>(P) * N), ocomp(static_cast<size_t>(P) * N);
  run(/*force_composite=*/false, ofast);
  run(/*force_composite=*/true, ocomp);
  REQUIRE(ofast.size() == ocomp.size());
  for (size_t i = 0; i < ofast.size(); ++i) CHECK(ofast[i] == ocomp[i]);  // BYTE-IDENTICAL

  // And both equal the fully-manual standalone composite.
  std::vector<float> gg(static_cast<size_t>(P) * N), uu(static_cast<size_t>(P) * N);
  {
    std::vector<int32_t> ids = eids;
    vt::Tensor a = vt::Tensor::Contiguous(x1.data(), vt::DType::kF32, q.device, {1, K});
    vt::Tensor eid = vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
    vt::Tensor gw = gate_exps.View(), uw = up_exps.View();
    vt::Tensor og = vt::Tensor::Contiguous(gg.data(), vt::DType::kF32, q.device, {P, N});
    vt::Tensor ou = vt::Tensor::Contiguous(uu.data(), vt::DType::kF32, q.device, {P, N});
    vt::MatmulBTQuantGrouped(q, og, a, gw, eid);
    vt::MatmulBTQuantGrouped(q, ou, a, uw, eid);
  }
  for (size_t i = 0; i < ofast.size(); ++i)
    CHECK(ofast[i] == ClampedSwiGLU(gg[i], uu[i], limit));
}

// ── W2C: the load is MEMORY-BOUNDED — the ~1 TiB f32 tower is NOT built ────────
// The crux of the brick. The big MLA/MoE/lm_head weights live ONLY in the
// compressed `gguf` tower; the `host` tower holds only the small non-GEMM tensors
// (norms/embed/mixing/hash/ape/sink). A regression that rebuilds the f32 tower
// makes the host bytes EXCEED the f32-expanded weight image — asserted RED-first.
TEST_CASE("LoadDeepseekV4FromGguf: memory-bounded — no f32 expert/linear tower (W2C)") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);

  // (1) The big host slots are EMPTY (the loader also VT_CHECKs this at load).
  CHECK(w.host.lm_head.empty());
  CHECK(w.host.layers[0].wq_a.empty());
  CHECK(w.host.layers[0].wq_b.empty());
  CHECK(w.host.layers[3].wkv.empty());
  CHECK(w.host.layers[3].exp_w1.empty());
  CHECK(w.host.layers[3].exp_w2.empty());
  CHECK(w.host.layers[2].comp_wgate.empty());
  // ...but the keep-quant tower DOES hold them, compressed.
  CHECK_FALSE(w.gguf.layers[0].wq_a.Empty());
  CHECK_FALSE(w.gguf.layers[3].moe_down_exps.Empty());

  const int64_t host_b = vllm::DeepseekV4HostResidentBytes(w);
  const int64_t gguf_b = vllm::DeepseekV4GgufResidentBytes(w);

  // (2) The host tower (small tensors only) is SMALLER than the keep-quant tower
  // that actually holds the weights — i.e. the weights are NOT in host.
  CHECK(host_b < gguf_b);

  // (3) RED-first bound: what the host WOULD be if the big weights were
  // f32-expanded (the bug). Compute that image; assert the real host stays far
  // below it (the bug would make host_b ~ this or larger).
  const int64_t big_f32 =
      // per layer: wq_a + wq_b + wkv + wo_a + wo_b + gate + 3 shared + 3*E experts
      static_cast<int64_t>(d.n_layer) *
          (d.qlr * d.H + d.nh * d.head_dim * d.qlr + d.head_dim * d.H +
           d.o_groups * d.olr * (d.nh * d.head_dim / d.o_groups) +
           d.H * d.o_groups * d.olr + d.E * d.H + 3 * d.mi * d.H +
           3 * d.E * d.mi * d.H) *
          4 +
      d.vocab * d.H * 4;  // lm_head
  CHECK(host_b < big_f32);  // host does NOT hold the f32-expanded big tower

  // (4) PROJECTION to the REAL DeepSeek-V4-Flash config (spec §W0/§W2b): the f32
  // expansion of the 256 routed experts alone OOM-reboots the 119 GiB unified
  // pool, while the keep-quant `UD-IQ2_XXS` checkpoint (~91 GiB) fits with the
  // small f32 host tower + headroom. This is the memory-FEASIBILITY the brick
  // unblocks (the real 91 GiB run stays the operational W8-run).
  const double kGiB = 1024.0 * 1024.0 * 1024.0;
  const int64_t kL = 43, kE = 256, kH = 4096, kMI = 2048;
  const double routed_expert_params =
      3.0 * static_cast<double>(kE) * kMI * kH * kL;          // gate+up+down, ~2.77e11
  const double routed_f32_gib = routed_expert_params * 4.0 / kGiB;  // ~1030 GiB
  CHECK(routed_f32_gib > 119.0);  // the f32 tower OOM-reboots the box (the bug)
  const double kIq2xxsCheckpointGiB = 90.9;  // measured file size (spec §W2b HW table)
  const double kSmallHostGiBUpperBound =
      3.0;  // embed (~2 GiB f32) + norms/MHC/hash/ape (< 1 GiB) at real config
  CHECK(kIq2xxsCheckpointGiB + kSmallHostGiBUpperBound < 119.0);  // memory-FEASIBLE
}

// ── entrypoint wiring (W8-final deliverable #1): a `deepseek4` GGUF must route
//    through the top-level GGUF dispatch onto the registered DeepseekV4ForCausalLM
//    model class, so the CLI/server recognizes the arch (it was qwen-only). ─────
TEST_CASE("DeepseekV4HfConfigFromGguf: deepseek4 GGUF routes to DeepseekV4ForCausalLM") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::DeepseekV4HfConfigFromGguf(g);

  // model_type keeps llama.cpp's GGUF family key; architectures maps onto the
  // registered vLLM model class the top-level dispatch resolves.
  CHECK(c.model_type == "deepseek4");
  REQUIRE(c.architectures.size() == 1);
  CHECK(c.architectures[0] == "DeepseekV4ForCausalLM");
  // Typed geometry republished from the GGUF KV.
  CHECK(c.hidden_size == 32);
  CHECK(c.num_hidden_layers == 4);
  CHECK(c.head_dim == 32);
  CHECK(c.num_attention_heads == 2);
  CHECK(c.vocab_size == 16);
  // The scalars the registry parse hook (ParseDeepseekV4Config) reads from raw.
  CHECK(c.raw.at("hc_mult").get<int64_t>() == 2);
  CHECK(c.raw.at("n_routed_experts").get<int64_t>() == 4);
  CHECK(c.raw.at("num_experts_per_tok").get<int64_t>() == 2);
  CHECK(c.raw.at("scoring_func").get<std::string>() == "sqrtsoftplus");
  CHECK(c.raw.at("expert_dtype").get<std::string>() == "fp4");
  REQUIRE(c.raw.at("compress_ratios").is_array());
  CHECK(c.raw.at("compress_ratios").size() == 4);

  // The registry resolves the mapped architecture to the DeepSeek-V4 factory —
  // i.e. the top-level GGUF dispatch now recognizes a deepseek4 file (the qwen-
  // only HfConfigFromGguf would throw "unexpected architecture 'deepseek4'").
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(c);
  CHECK(reg.architecture == "DeepseekV4ForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK_FALSE(reg.factory->is_dense_model);
  CHECK(reg.factory->load_weights != nullptr);
}

// ── The antirez ds4 `q2-imatrix` real-file quirks LOAD + FORWARD (W8-run) ──────
// The cross-engine oracle (antirez/ds4) shares its q2-imatrix GGUF as the
// apples-to-apples vehicle. Its header (verified via HF HTTP-range) carries three
// representational differences from a naive synthetic that the loader must handle
// or it throws before a single token: (1) `swiglu_clamp_exp` per-layer f32 array
// instead of the scalar `swiglu_clamp` (a missing limit ZEROES every expert —
// ClampedSwiGLU min(gate,0)*clamp(up,0,0)); (2) `compress_ratios` of length
// block_count+1 (trailing MTP entry); (3) `ffn_gate_tid2eid` stored as ggml I32.
// The tensor NAME set is otherwise byte-identical to unsloth's (1328/1328, gated
// by scripts/check-dsv4-gguf-namemap.py). This case proves the loader accepts all
// three and still runs the keep-quant forward finite + deterministic.
TEST_CASE("LoadDeepseekV4FromGguf: antirez ds4 q2-imatrix real-file quirks (W8-run)") {
  Dims d;
  TempFile f(BuildGguf(d, /*drop=*/"", /*extra=*/"", /*ds4_flavor=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());

  // Params resolve: swiglu limit read from the ARRAY (not the absent scalar), and
  // compress_ratios truncated to block_count despite the +1 trailing entry.
  const vllm::DeepseekV4Params p = vllm::DeepseekV4ParamsFromGguf(g);
  CHECK(p.swiglu_limit == 10.0);  // read from swiglu_clamp_exp[0]
  CHECK(static_cast<int64_t>(p.compress_ratios.size()) == p.num_hidden_layers);

  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  CHECK(w.accounted_tensors == static_cast<int64_t>(g.Tensors().size()));
  CHECK(w.has_gguf_weights);
  // The I32 tid2eid dequantized into the hash table (expert ids in [0, E)).
  REQUIRE(w.gguf.layers[0].is_hash);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<int32_t> tokens = {1, 5, 9};
  const std::vector<int32_t> positions = {0, 1, 2};
  const std::vector<float> lk = vllm::DeepseekV4ForwardGguf(w, q, tokens, positions);
  REQUIRE(lk.size() == tokens.size() * static_cast<size_t>(d.vocab));
  for (float v : lk) CHECK(std::isfinite(v));
  // Deterministic re-run (self-consistency, the W8-run correctness gate in miniature).
  const std::vector<float> lk2 = vllm::DeepseekV4ForwardGguf(w, q, tokens, positions);
  REQUIRE(lk2.size() == lk.size());
  for (size_t i = 0; i < lk.size(); ++i) CHECK(lk2[i] == lk[i]);
}

// ── REAL-ISH (non-collapsing) geometry gate — regression guard for the W8-run bug ─
// The historical tiny synthetic collapsed the DSA compressor's output width onto
// `head_dim`, which HID a real-geometry forward bug: the keep-quant GGUF forward
// assumed the compressor projects to `head_dim`, but the REAL DeepSeek-V4 compressor
// projects to `comp_width = 2*head_dim` (ds4 `ds4.c:5016-5021`, `coff==2`). On the
// real 80.7 GB run this threw `keep-quant GEMM: weight shape mismatch: want
// [N=512,K=4096] got [1024,4096]` at layer 2. This case builds the compressor at
// `comp_width = 2*head_dim` (comp_coff=2, i.e. comp_width != head_dim) and asserts
// the keep-quant forward RUNS finite + deterministic — i.e. the real keep-quant run
// takes the DENSE-MLA path (the DSA sparse compressor/indexer at real geometry is a
// named residual, EXACT for seq <= index_topk) instead of the tiny-shape compressor
// that mismatches. RED-first: reverting the dense-fallback gate (running the tiny
// compressor at comp_width != head_dim) makes this THROW the shape mismatch again.
TEST_CASE("DeepseekV4ForwardGguf: real-ish geometry (comp_width=2*head_dim) runs dense (W8-run guard)") {
  Dims d;
  d.comp_coff = 2;  // comp_width = 64 != head_dim = 32 (non-collapsing)
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  CHECK(w.accounted_tensors == static_cast<int64_t>(g.Tensors().size()));
  // The loader accepted the [2*head_dim, H] compressor projections.
  REQUIRE(w.gguf.layers[2].has_compressor);
  REQUIRE(w.gguf.layers[2].comp_wgate.shape[0] == 2 * d.head_dim);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<int32_t> tokens = {1, 5, 9};
  const std::vector<int32_t> positions = {0, 1, 2};
  // Runs (no [N=head_dim] vs [2*head_dim] shape-mismatch crash) + finite.
  const std::vector<float> lk = vllm::DeepseekV4ForwardGguf(w, q, tokens, positions);
  REQUIRE(lk.size() == tokens.size() * static_cast<size_t>(d.vocab));
  for (float v : lk) CHECK(std::isfinite(v));
  const std::vector<float> lk2 = vllm::DeepseekV4ForwardGguf(w, q, tokens, positions);
  for (size_t i = 0; i < lk.size(); ++i) CHECK(lk2[i] == lk[i]);
}

// ── MLA per-head query RMS-norm (coherence-debug #188) ────────────────────────
// DeepSeek-V4 applies a per-head RMS-norm to q AFTER wq_b and BEFORE RoPE (ds4
// `head_rms_norm_inplace`): each head's `head_dim` sub-vector is scaled by
// 1/sqrt(mean(x^2)+eps), NO learnable weight. Our forward previously OMITTED it,
// which made q rel-L2 ~0.96 vs the ds4 oracle at L00 (with a bit-exact input) and
// drove the whole incoherent-generation cascade. Spec-anchored oracle: compute the
// reference by hand from the formula (upstream is the executable spec).
TEST_CASE("DeepseekV4QHeadRmsNormInplace: per-head unit-RMS matches the spec formula") {
  const int64_t n_head = 3, head_dim = 4;
  const float eps = 1e-6f;
  // distinct per-head magnitudes so a MISSING norm leaves distinctly non-unit RMS.
  std::vector<float> q = {1.f, 2.f, 3.f, 4.f,        // head0 (rms=sqrt(7.5)≈2.739)
                          -10.f, 0.f, 10.f, -20.f,   // head1 (large)
                          0.1f, -0.1f, 0.2f, -0.2f}; // head2 (small)
  const std::vector<float> in = q;
  vllm::DeepseekV4QHeadRmsNormInplace(q, n_head, head_dim, eps);
  for (int64_t h = 0; h < n_head; ++h) {
    // (a) matches the hand-computed spec reference (per-head 1/sqrt(mean(x^2)+eps)).
    double ss = 0.0;
    for (int64_t i = 0; i < head_dim; ++i) ss += (double)in[h * head_dim + i] * in[h * head_dim + i];
    const float r = 1.0f / std::sqrt((float)(ss / head_dim) + eps);
    double out_ss = 0.0;
    for (int64_t i = 0; i < head_dim; ++i) {
      CHECK(q[h * head_dim + i] == doctest::Approx(in[h * head_dim + i] * r));
      out_ss += (double)q[h * head_dim + i] * q[h * head_dim + i];
    }
    // (b) SEMANTIC: each head is now unit-RMS (the property the forward relies on) —
    // RED-first: the pre-norm heads (rms 2.74 / 15.8 / 0.16) are NOT unit-RMS, so a
    // forward that skips this op feeds mis-scaled queries into attention.
    CHECK(std::sqrt(out_ss / head_dim) == doctest::Approx(1.0f).epsilon(1e-4));
  }
}

TEST_CASE("LoadDeepseekV4FromGguf: RED-first — unmapped/leftover tensors throw") {
  Dims d;
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  // (a) a MISSING required tensor -> the load throws (unaccounted route).
  {
    TempFile f(BuildGguf(d, /*drop=*/Blk(2, "attn_kv.weight")));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    CHECK_THROWS(vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol));
  }
  // (b) a LEFTOVER tensor the blk.N.* map does not cover -> the load throws.
  {
    TempFile f(BuildGguf(d, /*drop=*/"", /*extra=*/"blk.0.bogus_unmapped.weight"));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    CHECK_THROWS(vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol));
  }
}


TEST_CASE("W-3: `kv_prewritten` attends the pages WITHOUT overwriting them (#1314)") {
  // A DSpark drafter block's KV rows come from the TARGET's tap state, written at
  // the target's positions, while its query comes from the block's own hidden
  // state. Every other path in this tree derives both from the same state, which
  // is why the paged write was unconditional. This is the smallest extension that
  // lets the two differ, and the claim it has to earn is exact: with the flag on,
  // the cache the caller filled must come back BYTE-UNCHANGED.
  Dims d;
  d.sliding_window = 0;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);
  REQUIRE(w.has_gguf_weights);

  const int64_t nlayers = w.params.num_hidden_layers;
  const int64_t hd = w.params.head_dim;
  const int64_t block_size = 8, num_blocks = 4;
  const std::vector<int32_t> step{1, 2, 3};
  std::vector<int32_t> pos(step.size());
  for (size_t i = 0; i < step.size(); ++i) pos[i] = static_cast<int32_t>(i);

  const auto make_pages = [&](std::vector<std::vector<float>>* storage,
                              std::vector<vt::Tensor>* pages) {
    storage->assign(static_cast<size_t>(nlayers), {});
    pages->assign(static_cast<size_t>(nlayers), vt::Tensor{});
    for (int64_t l = 0; l < nlayers; ++l) {
      // A RECOGNISABLE pattern, so "unchanged" is a real comparison rather than a
      // check that zeros stayed zero.
      std::vector<float>& buf = (*storage)[static_cast<size_t>(l)];
      buf.resize(static_cast<size_t>(num_blocks * block_size * hd));
      for (size_t i = 0; i < buf.size(); ++i)
        buf[i] = 0.01f * static_cast<float>((i * 7 + l * 13) % 41) - 0.2f;
      (*pages)[static_cast<size_t>(l)] = vt::Tensor::Contiguous(
          buf.data(), vt::DType::kF32, q.device, {num_blocks, block_size, hd});
    }
  };

  std::vector<std::vector<float>> st_on, st_off;
  std::vector<vt::Tensor> pg_on, pg_off;
  make_pages(&st_on, &pg_on);
  make_pages(&st_off, &pg_off);
  const std::vector<std::vector<float>> before = st_on;

  (void)vllm::DeepseekV4ForwardGgufPaged(w, q, pg_on, /*kv_base=*/0, step, pos,
                                         {static_cast<int32_t>(step.size() - 1)},
                                         /*kv_prewritten=*/true);
  (void)vllm::DeepseekV4ForwardGgufPaged(w, q, pg_off, /*kv_base=*/0, step, pos,
                                         {static_cast<int32_t>(step.size() - 1)},
                                         /*kv_prewritten=*/false);

  // ON: every byte the caller wrote survives.
  for (int64_t l = 0; l < nlayers; ++l) {
    const auto& a = before[static_cast<size_t>(l)];
    const auto& b = st_on[static_cast<size_t>(l)];
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) {
        CAPTURE(l);
        CAPTURE(i);
        REQUIRE(a[i] == b[i]);
      }
    }
  }

  // OFF: the step DID write, or the flag is not load-bearing and the case above
  // would pass against a build that never writes at all.
  bool wrote = false;
  for (int64_t l = 0; l < nlayers && !wrote; ++l) {
    const auto& a = before[static_cast<size_t>(l)];
    const auto& b = st_off[static_cast<size_t>(l)];
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i] != b[i]) { wrote = true; break; }
  }
  CHECK(wrote);
}
