// vllm.cpp ORIGINAL — the `qwen4exp` GGUF weight loader. See
// `qwen4_exp_weights.h` for the two oracles this file answers to, the four
// convert-time transforms it inverts, and why the GGUF arm is the only arm.
//
// Tensor names and metadata keys follow llama.cpp pull request
// [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) at head
// `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, which is the layout the shipped
// `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` file uses. That pull request is
// STILL OPEN; if it renames a key before it merges, this file changes and
// `tests/vllm/models/qwen4_exp_gguf_manifest.inc` is regrown.
#include "vllm/model_executor/models/qwen4_exp_weights.h"

#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"  // OwnGgufQuantBlocks
#include "vllm/model_executor/models/qwen4_exp_ple.h"         // NGramTableLayout
#include "vt/dtype.h"
#include "vt/quant.h"

namespace vllm {
namespace {

std::string Blk(int64_t layer, const char* suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

// `GgufTensorInfo::shape` is already REVERSED into torch row-major order by the
// reader, so a 2-D matmul weight reads [N = out, K = in] — the file's own
// orientation and our `vt::MatmulBT` one.
void RequireShape(const GgufTensorInfo& t, const std::vector<int64_t>& want) {
  bool ok = t.shape.size() == want.size();
  for (size_t i = 0; ok && i < want.size(); ++i) ok = t.shape[i] == want[i];
  if (ok) return;
  std::string got;
  for (size_t i = 0; i < t.shape.size(); ++i)
    got += (i != 0 ? ", " : "") + std::to_string(t.shape[i]);
  std::string exp;
  for (size_t i = 0; i < want.size(); ++i)
    exp += (i != 0 ? ", " : "") + std::to_string(want[i]);
  VT_CHECK(false, "qwen4_exp gguf: shape mismatch for " + t.name + ": got [" +
                      got + "], expected [" + exp + "]");
}

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

OwnedTensor MakeTensor(vt::DType dtype, const std::vector<int64_t>& shape,
                       bool nk, size_t elem_bytes) {
  OwnedTensor o;
  o.dtype = dtype;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "qwen4_exp gguf: rank exceeds kMaxRank");
  for (int i = 0; i < o.rank; ++i) o.shape[i] = shape[i];
  o.nk = nk;
  o.bytes.resize(static_cast<size_t>(Numel(shape)) * elem_bytes);
  return o;
}

OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk) {
  return MakeTensor(vt::DType::kBF16, shape, nk, sizeof(uint16_t));
}

// Dequantize the whole tensor to f32 and hand it back for a value or layout
// rewrite. Every transform in this file goes through f32 rather than through
// bf16, so a rewrite never rounds twice.
std::vector<float> DequantAll(const GgufFile& g, const std::string& name,
                              const std::vector<int64_t>& shape) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, shape);
  const int64_t n = Numel(shape);
  std::vector<float> f = DequantGgufRowToF32(t.ggml_type, t.data, n);
  VT_CHECK(static_cast<int64_t>(f.size()) == n,
           "qwen4_exp gguf: dequant length mismatch for " + name);
  return f;
}

OwnedTensor Bf16From(const std::vector<float>& f,
                     const std::vector<int64_t>& shape, bool nk) {
  OwnedTensor o = MakeBf16(shape, nk);
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (size_t i = 0; i < f.size(); ++i) dst[i] = vt::F32ToBF16(f[i]);
  return o;
}

OwnedTensor F32From(const std::vector<float>& f,
                    const std::vector<int64_t>& shape) {
  OwnedTensor o = MakeTensor(vt::DType::kF32, shape, /*nk=*/false, sizeof(float));
  std::memcpy(o.bytes.data(), f.data(), f.size() * sizeof(float));
  return o;
}

// Dequantize a whole tensor into an owned bf16 buffer in the file's own order.
OwnedTensor ExpandBf16(const GgufFile& g, const std::string& name,
                       const std::vector<int64_t>& shape, bool nk) {
  return Bf16From(DequantAll(g, name, shape), shape, nk);
}

// A [n] norm gamma as bf16.
//
// `unshift` inverts the converter's baked `+1` (transform 1). It is a REQUIRED
// argument with no default on purpose: every call site in this file has to state
// which side of the one exception it is on, and a defaulted parameter is how the
// exception gets inherited by the next tensor added below it.
OwnedTensor LoadNormBf16(const GgufFile& g, const std::string& name, int64_t n,
                         bool unshift) {
  std::vector<float> f = DequantAll(g, name, {n});
  if (unshift) {
    for (float& v : f) v -= 1.0F;
  }
  return Bf16From(f, {n}, /*nk=*/false);
}

const GgufFile* MmapSrc(const GgufFile& g, const GgufLoadPolicy& pol) {
  return pol.mmap_residency ? &g : nullptr;
}

// One standalone [N, K] matmul operand taken VERBATIM: kept as raw ggml blocks
// when the policy routes it there, kept as F16 when it routes it there, expanded
// to bf16 in the file's own [N, K] order otherwise.
OwnedTensor LoadMatmul(const GgufFile& g, const GgufLoadPolicy& pol,
                       const std::string& name, int64_t n, int64_t k) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {n, k});
  const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
  if (r == GgufResidency::kKeepQuant)
    return OwnGgufQuantBlocks(t, n, k, /*row_offset=*/0, MmapSrc(g, pol),
                              pol.quant_repack);
  if (r == GgufResidency::kKeepF16)
    return OwnGgufF16(t, n, k, /*row_offset=*/0, MmapSrc(g, pol), /*nk=*/true,
                      pol.elem_kn_repack);
  return ExpandBf16(g, name, {n, k}, /*nk=*/true);
}

// A 3-D [E, N, K] stacked expert tensor. Each expert slab is a whole number of
// ROWS and therefore a whole number of blocks, so the keep-quant arm is a byte
// range and no block is ever cut — which is what lets a 512-expert IQ1_S tensor
// stay resident at all.
OwnedTensor LoadStackedExperts(const GgufFile& g, const GgufLoadPolicy& pol,
                               const std::string& name, int64_t e, int64_t n,
                               int64_t k) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {e, n, k});
  const GgufResidency r = pol.Route(t, GgufTensorRole::kStackedExpertWeight);
  if (r == GgufResidency::kKeepQuant) {
    // `OwnGgufQuantBlocks` works in rows, so the stack is flattened to
    // [E*N, K] and reshaped back. The bytes are identical either way; only the
    // recorded shape differs, and the consumer slices by expert.
    OwnedTensor o = OwnGgufQuantBlocks(t, e * n, k, /*row_offset=*/0,
                                       MmapSrc(g, pol), pol.quant_repack);
    o.rank = 3;
    o.shape[0] = e;
    o.shape[1] = n;
    o.shape[2] = k;
    return o;
  }
  return ExpandBf16(g, name, {e, n, k}, /*nk=*/true);
}

// ── transform 3: the V-head reorder ─────────────────────────────────────────
//
// `_LinearAttentionVReorderBase` (llama.cpp `conversion/qwen.py`) rewrites the V
// heads of every Gated DeltaNet tensor from HF's GROUPED order into ggml's TILED
// order so `ggml_repeat` can broadcast a K head across its V heads. Grouped head
// `g = k*R + r` is stored at tiled position `t = r*K + k`, where `K` is
// `num_k_heads` and `R` is `num_v_heads / num_k_heads`.
//
// These two functions are the INVERSE, reading source position `t` into
// destination position `g`, and they are deliberately written here rather than
// shared with `qwen3_5_gguf_weights.cpp`: that file's copies live in an
// anonymous namespace with no header, and hoisting them would edit a
// 1745-line translation unit several other rows are working in. The
// duplication is four lines of index arithmetic.
//
// IT IS GATED ON THIS SIDE ONLY, and the earlier version of this sentence
// claimed both. Swapping `g` and `t` in the two functions BELOW reddens
// `test_qwen4_exp_gguf_weights` (2 cases, 41 assertions, mutation M5); the
// same swap applied to the `qwen3_5_gguf_weights.cpp` copy leaves
// `test_gguf_qwen36_loader` (7/7, 555 assertions), `test_model_loader_gguf`
// (7/7), `test_gguf_nvfp4` (14/14) and `test_gguf_keep_quant` (42/42) ALL
// green, because every
// synthetic `qwen35`/`qwen35moe` fixture in this tree is built at
// `ssm.group_count = 2, ssm.time_step_rank = 4` — K == R, where the
// permutation is its OWN INVERSE. Filed as
// [#2081](https://github.com/mudler/vllm.cpp/issues/2081) and NOT fixed here:
// re-shaping a shipped model's fixtures is outside this row's scope. Do not
// restore the "both sides" reading without re-running that mutation.
//
// TWO SHAPES OF FIXTURE CANNOT GATE THIS, and the second one was found the hard
// way. At `num_k_heads == 1` both functions are the IDENTITY (`g = r`,
// `t = r*1 + 0 = r`), so a one-key-head fixture cannot tell a correct un-reorder
// from no un-reorder. And at `num_k_heads == v_per_k` the map is its own
// INVERSE, so a two-by-two fixture cannot tell it from the permutation run
// backwards — the mutation that swaps `g` and `t` below produced byte-identical
// output and left the whole reorder suite green. The gate's fixture is K = 2
// with R = 3, which is neither, and which is also the released model's own
// ratio (16 key heads to 48 value heads).

// Rows [row_off, row_off + num_v*head_rows) of a [rows, cols] row-major buffer.
void ReorderVRows(std::vector<float>& buf, int64_t cols, int64_t row_off,
                  int64_t num_k, int64_t v_per_k, int64_t head_rows) {
  const int64_t num_v = num_k * v_per_k;
  std::vector<float> tmp(static_cast<size_t>(num_v) *
                         static_cast<size_t>(head_rows) *
                         static_cast<size_t>(cols));
  for (int64_t k = 0; k < num_k; ++k) {
    for (int64_t r = 0; r < v_per_k; ++r) {
      const int64_t g = k * v_per_k + r;  // destination (HF grouped)
      const int64_t t = r * num_k + k;    // source (GGUF tiled)
      std::memcpy(tmp.data() + static_cast<size_t>(g * head_rows * cols),
                  buf.data() + static_cast<size_t>((row_off + t * head_rows) * cols),
                  static_cast<size_t>(head_rows * cols) * sizeof(float));
    }
  }
  std::memcpy(buf.data() + static_cast<size_t>(row_off * cols), tmp.data(),
              tmp.size() * sizeof(float));
}

// The full column range [0, cols) of a [rows, cols] row-major buffer, where
// `cols == num_k * v_per_k * head_cols` — `out_proj`, whose reorder is on its
// INPUT dimension.
void ReorderVCols(std::vector<float>& buf, int64_t rows, int64_t cols,
                  int64_t num_k, int64_t v_per_k, int64_t head_cols) {
  std::vector<float> tmp(buf.size());
  for (int64_t row = 0; row < rows; ++row) {
    const float* src = buf.data() + static_cast<size_t>(row * cols);
    float* dst = tmp.data() + static_cast<size_t>(row * cols);
    for (int64_t k = 0; k < num_k; ++k) {
      for (int64_t r = 0; r < v_per_k; ++r) {
        const int64_t g = k * v_per_k + r;
        const int64_t t = r * num_k + k;
        std::memcpy(dst + static_cast<size_t>(g * head_cols),
                    src + static_cast<size_t>(t * head_cols),
                    static_cast<size_t>(head_cols) * sizeof(float));
      }
    }
  }
  buf.swap(tmp);
}

// ── the per-module loads ────────────────────────────────────────────────────

Qwen4ExpGatedResidualWeights LoadGatedResidual(const GgufFile& g,
                                               const GgufLoadPolicy& pol,
                                               const Qwen4ExpParams& p,
                                               const std::string& prefix,
                                               bool has_inject) {
  const int64_t stream = p.stream_width();
  Qwen4ExpGatedResidualWeights w;
  // `hc_norm`. Upstream applies `output * (1.0 + weight)` with `weight`
  // ZERO-initialised, so an unfolded gamma is centred on 0 and a folded one on
  // 1.0 — and the fold is what makes the file's value the multiplier our own
  // `out * weight` grouped norm wants. Corroborated elementwise on three
  // published artifacts during the fresh review of #1988.
  w.hc_norm = LoadNormBf16(g, prefix + "norm.weight", stream, /*unshift=*/true);
  w.down = LoadMatmul(g, pol, prefix + "down.weight", p.hc_lowrank, stream);
  w.up = LoadMatmul(g, pol, prefix + "up.weight", stream, p.hc_lowrank);
  w.has_inject = has_inject;
  if (has_inject) {
    w.inject =
        LoadMatmul(g, pol, prefix + "inject.weight", p.hc_count, stream);
  }
  return w;
}

Qwen4ExpGdnWeights LoadGdn(const GgufFile& g, const GgufLoadPolicy& pol,
                           const Qwen4ExpParams& p, int64_t il) {
  const int64_t h = p.hidden_size;
  const int64_t num_k = p.linear_num_key_heads;
  const int64_t num_v = p.linear_num_value_heads;
  const int64_t dv = p.linear_value_head_dim;
  const int64_t key_dim = p.linear_key_dim();
  const int64_t value_dim = p.linear_value_dim();
  const int64_t conv_dim = p.linear_conv_dim();
  const int64_t kk = p.linear_conv_kernel_dim;
  // `num_v == num_k` is a legal config and means the converter wrote HF order
  // straight through. `ParseQwen4ExpParams` already refused a ragged ratio.
  const bool reorder = num_v != num_k;
  const int64_t v_per_k = num_v / num_k;

  // A reordered projection is LAYOUT-REWRITTEN at load, so it is
  // `kTransformedWeight` and can never keep its blocks — a k-quant superblock
  // spans elements the permutation moves. Without the reorder these are
  // ordinary verbatim GEMM operands.
  //
  // THE COST IS REAL AND IT IS NOT HIDDEN: on the released 16-vs-48 config the
  // reorder is always on, so every Gated DeltaNet projection of all 36 linear
  // layers expands to bf16 at load rather than staying Q5_K/Q6_K. That is the
  // same property `qwen3_5_gguf_weights.cpp` already has for the 27B, and the
  // alternative — permuting whole rows inside the block stream — is only
  // available for the ROW reorders, never for `out_proj`'s COLUMN one.
  Qwen4ExpGdnWeights w;

  // in_proj_qkv <- attn_qkv [conv_dim, H]. Only the TRAILING V rows reorder;
  // the leading `2 * key_dim` q and k rows are untouched.
  {
    const std::string nm = Blk(il, "attn_qkv.weight");
    if (!reorder) {
      w.in_proj_qkv = LoadMatmul(g, pol, nm, conv_dim, h);
    } else {
      std::vector<float> f = DequantAll(g, nm, {conv_dim, h});
      ReorderVRows(f, h, /*row_off=*/2 * key_dim, num_k, v_per_k, dv);
      w.in_proj_qkv = Bf16From(f, {conv_dim, h}, /*nk=*/true);
    }
  }
  // in_proj_z <- attn_gate [value_dim, H]; ALL rows reorder.
  {
    const std::string nm = Blk(il, "attn_gate.weight");
    if (!reorder) {
      w.in_proj_z = LoadMatmul(g, pol, nm, value_dim, h);
    } else {
      std::vector<float> f = DequantAll(g, nm, {value_dim, h});
      ReorderVRows(f, h, /*row_off=*/0, num_k, v_per_k, dv);
      w.in_proj_z = Bf16From(f, {value_dim, h}, /*nk=*/true);
    }
  }
  // in_proj_b <- ssm_beta, in_proj_a <- ssm_alpha, both [num_v, H]. One ROW per
  // V head, so the head "row count" is 1.
  //
  // THE NAMES CROSS. `ssm_beta` is `in_proj_b` and `ssm_alpha` is `in_proj_a`,
  // which is the mapping `qwen3_5_gguf_weights.cpp` already uses for the same
  // converter; swapping them exchanges the delta-rule's gate and step and no
  // shape check can see it.
  for (const auto& [dst, nm] :
       {std::pair<OwnedTensor*, const char*>{&w.in_proj_b, "ssm_beta.weight"},
        std::pair<OwnedTensor*, const char*>{&w.in_proj_a, "ssm_alpha.weight"}}) {
    const std::string full = Blk(il, nm);
    if (!reorder) {
      *dst = LoadMatmul(g, pol, full, num_v, h);
    } else {
      std::vector<float> f = DequantAll(g, full, {num_v, h});
      ReorderVRows(f, h, /*row_off=*/0, num_k, v_per_k, /*head_rows=*/1);
      *dst = Bf16From(f, {num_v, h}, /*nk=*/true);
    }
  }
  // conv1d <- ssm_conv1d [conv_dim, K]. Only the V CHANNELS reorder, and this is
  // NOT a matmul operand: it is a depthwise filter, so it is never transposed
  // and never keep-quant (the released file stores it f32 anyway).
  {
    std::vector<float> f = DequantAll(g, Blk(il, "ssm_conv1d.weight"),
                                      {conv_dim, kk});
    if (reorder) ReorderVRows(f, kk, /*row_off=*/2 * key_dim, num_k, v_per_k, dv);
    w.conv1d = Bf16From(f, {conv_dim, kk}, /*nk=*/false);
  }
  // a_log <- ssm_a. Transform 2: the converter stored `-exp(A_log)`, so recover
  // `A_log = log(-x)`. Refuses a non-negative value BY NAME rather than
  // producing a NaN that first shows up as garbage several layers later.
  {
    std::vector<float> f = DequantAll(g, Blk(il, "ssm_a"), {num_v});
    for (size_t i = 0; i < f.size(); ++i) {
      VT_CHECK(f[i] < 0.0F,
               "qwen4_exp gguf: " + Blk(il, "ssm_a") + "[" +
                   std::to_string(i) +
                   "] must be negative (the converter writes -exp(A_log)), got " +
                   std::to_string(f[i]));
      f[i] = std::log(-f[i]);
    }
    if (reorder) ReorderVRows(f, /*cols=*/1, 0, num_k, v_per_k, 1);
    w.a_log = F32From(f, {num_v});
  }
  // dt_bias <- ssm_dt.bias, value unchanged.
  {
    std::vector<float> f = DequantAll(g, Blk(il, "ssm_dt.bias"), {num_v});
    if (reorder) ReorderVRows(f, /*cols=*/1, 0, num_k, v_per_k, 1);
    w.dt_bias = F32From(f, {num_v});
  }
  // norm_weight <- ssm_norm [value_head_dim]. THE ONE NORM WITH NO FOLD: the
  // inherited rule excludes `linear_attn.norm.weight` by name, and the published
  // artifacts show it unfolded in [0.875, 1.023] while every sibling gamma is
  // centred on 1.0.
  w.norm_weight = LoadNormBf16(g, Blk(il, "ssm_norm.weight"), dv,
                               /*unshift=*/false);
  // out_proj <- ssm_out [H, value_dim]; the V COLUMNS reorder.
  {
    const std::string nm = Blk(il, "ssm_out.weight");
    if (!reorder) {
      w.out_proj = LoadMatmul(g, pol, nm, h, value_dim);
    } else {
      std::vector<float> f = DequantAll(g, nm, {h, value_dim});
      ReorderVCols(f, h, value_dim, num_k, v_per_k, dv);
      w.out_proj = Bf16From(f, {h, value_dim}, /*nk=*/true);
    }
  }
  return w;
}

Qwen4ExpQsaWeights LoadQsa(const GgufFile& g, const GgufLoadPolicy& pol,
                           const Qwen4ExpParams& p, int64_t il) {
  const int64_t h = p.hidden_size;
  const int64_t qdim = p.num_attention_heads * p.head_dim;
  const int64_t kvdim = p.num_key_value_heads * p.head_dim;
  Qwen4ExpQsaWeights w;
  // `* 2`: query and OUTPUT GATE share one projection and are split per head.
  w.q_proj = LoadMatmul(g, pol, Blk(il, "attn_q.weight"), qdim * 2, h);
  w.k_proj = LoadMatmul(g, pol, Blk(il, "attn_k.weight"), kvdim, h);
  w.v_proj = LoadMatmul(g, pol, Blk(il, "attn_v.weight"), kvdim, h);
  w.o_proj = LoadMatmul(g, pol, Blk(il, "attn_output.weight"), h, qdim);
  // Per-HEAD gammas, folded: `q_norm`/`k_norm` end in `norm.weight` and the
  // inherited blanket rule reaches them.
  w.q_norm = LoadNormBf16(g, Blk(il, "attn_q_norm.weight"), p.head_dim,
                          /*unshift=*/true);
  w.k_norm = LoadNormBf16(g, Blk(il, "attn_k_norm.weight"), p.head_dim,
                          /*unshift=*/true);

  const int64_t idx = p.qsa.head_dim;
  w.idx_q_proj = LoadMatmul(g, pol, Blk(il, "indexer.q_proj.weight"),
                            p.qsa.n_heads * idx, h);
  w.idx_k_proj = LoadMatmul(g, pol, Blk(il, "indexer.k_proj.weight"),
                            p.qsa.kv_heads * idx, h);
  // Folded by #27742's OWN early-returning branch, which names
  // `.indexer.q_layernorm.weight` and `.indexer.k_layernorm.weight`
  // explicitly. Same answer as the blanket rule would give; the branch returns
  // first, so the fold is applied exactly once.
  w.idx_q_norm = LoadNormBf16(g, Blk(il, "indexer.q_norm.weight"), idx,
                              /*unshift=*/true);
  w.idx_k_norm = LoadNormBf16(g, Blk(il, "indexer.k_norm.weight"), idx,
                              /*unshift=*/true);
  return w;
}

Qwen4ExpMoeWeights LoadMoe(const GgufFile& g, const GgufLoadPolicy& pol,
                           const Qwen4ExpParams& p, int64_t il) {
  const int64_t h = p.hidden_size;
  const int64_t e = p.num_experts;
  const int64_t mi = p.moe_intermediate_size;
  const int64_t si = p.shared_expert_intermediate_size;
  Qwen4ExpMoeWeights w;
  // The router is f32 in the file and stays f32 in ours: a top-10 selection over
  // 512 experts is decided by differences that bf16 can collapse into a tie, and
  // a tie broken the other way is a different expert set for that token.
  w.router = F32From(DequantAll(g, Blk(il, "ffn_gate_inp.weight"), {e, h}),
                     {e, h});
  // `Linear(H, 1)` squeezed to a vector: the shared expert's sigmoid gate. ONE
  // DIMENSIONAL in the file, and a loader that expected [1, H] would refuse a
  // correct file.
  w.shared_gate =
      F32From(DequantAll(g, Blk(il, "ffn_gate_inp_shexp.weight"), {h}), {h});
  w.gate_exps =
      LoadStackedExperts(g, pol, Blk(il, "ffn_gate_exps.weight"), e, mi, h);
  w.up_exps =
      LoadStackedExperts(g, pol, Blk(il, "ffn_up_exps.weight"), e, mi, h);
  // `down` is [E, H, moe_I]: its reduction dim is the intermediate size, which
  // is 640 on the released config and therefore indivisible by 256 — the reason
  // no K-quant can carry it and the shipped file reaches for IQ4_NL instead.
  w.down_exps =
      LoadStackedExperts(g, pol, Blk(il, "ffn_down_exps.weight"), e, h, mi);
  w.shared_gate_proj =
      LoadMatmul(g, pol, Blk(il, "ffn_gate_shexp.weight"), si, h);
  w.shared_up_proj = LoadMatmul(g, pol, Blk(il, "ffn_up_shexp.weight"), si, h);
  w.shared_down_proj =
      LoadMatmul(g, pol, Blk(il, "ffn_down_shexp.weight"), h, si);
  return w;
}

Qwen4ExpPleWeights LoadPle(const GgufFile& g, const GgufLoadPolicy& pol,
                           const Qwen4ExpParams& p, int64_t il) {
  const int64_t stream = p.stream_width();
  const int64_t embed = p.ple.embed_dim;
  Qwen4ExpPleWeights w;
  w.key_proj = LoadMatmul(g, pol, Blk(il, "ple_key.weight"), stream, embed);
  w.value_proj =
      LoadMatmul(g, pol, Blk(il, "ple_value.weight"), p.hidden_size, embed);
  // All three folded by #27742's explicit branch — and NONE of them would have
  // been reached by the inherited `endswith("norm.weight")` rule, because they
  // are spelled `norm_key` / `norm_query` / `norm_conv`. That is why the branch
  // exists, and why a loader that only mirrors the blanket rule leaves these
  // three at gamma ~ 1 + w instead of w.
  w.norm_key = LoadNormBf16(g, Blk(il, "ple_norm_key.weight"), stream,
                            /*unshift=*/true);
  w.norm_query = LoadNormBf16(g, Blk(il, "ple_norm_query.weight"), stream,
                              /*unshift=*/true);
  w.norm_conv = LoadNormBf16(g, Blk(il, "ple_norm_conv.weight"), stream,
                             /*unshift=*/true);
  // Depthwise, DILATED by `ngram_size`, so the state it drives is
  // `(kernel - 1) * ngram_size` deep — nine columns on the released config, not
  // three. The filter itself is still `kernel` wide.
  w.conv1d = ExpandBf16(g, Blk(il, "ple_conv1d.weight"),
                        {stream, p.ple.conv_kernel_size}, /*nk=*/false);
  return w;
}

bool HasTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors())
    if (t.name == name) return true;
  return false;
}

// The n-gram table's row count.
//
// TWO SOURCES, AND THE ORDER BETWEEN THEM IS THE POINT. A `qwen4exp` GGUF
// STATES the resolved per-head vocabulary sizes
// (`qwen4exp.ple.head_vocab_sizes`) and states NEITHER of the two inputs the
// prime chain needs, because llama.cpp's converter reads them off the
// checkpoint's own buffers instead of re-deriving them. Where the source states
// them they are the authority — they are what the shipped tensor was built
// against — and a loader that re-derived them from a DEFAULTED
// `ngram_vocab_size_base` would compute 40000128 rows for a table that has 128.
//
// Where the source states nothing, which is every config.json, W2's
// `BuildNGramTableLayout` (#1987) derives the chain: the sum of `ngram_heads`
// successive primes after `ngram_vocab_size_base - 1`, rounded up to
// `make_ngram_vocab_size_divisible_by`. That call is this wave's FIRST
// production call site for W2, which until now landed unreached.
//
// The two are cross-checked where both exist, but not here — inside this
// function only one of them is ever available. The gate does it at the RELEASED
// config, where the prime chain has its real inputs and the committed manifest
// has the real tensor: 320001536 both ways or the case reds.
int64_t NgramTableRows(const Qwen4ExpParams& p) {
  const int64_t stated = p.ple.stated_padded_vocab_size();
  if (stated > 0) return stated;

  qwen4_exp::PleGeometry geom;
  geom.hidden_size = p.hidden_size;
  geom.hc_count = p.hc_count;
  geom.ple_embed_dim = p.ple.embed_dim;
  geom.ple_conv_kernel_size = p.ple.conv_kernel_size;
  geom.ngram_size = p.ple.ngram_size;
  geom.heads_per_ngram = p.ple.heads_per_ngram;
  geom.ngram_vocab_size_base = p.ple.ngram_vocab_size_base;
  geom.make_ngram_vocab_size_divisible_by =
      p.ple.make_ngram_vocab_size_divisible_by;
  geom.vocab_size = p.vocab_size;
  geom.eos_token_id = p.eos_token_id;
  geom.seed = p.ple.seed;
  geom.rms_norm_eps = p.rms_norm_eps;
  // The layout varies per PLE layer only through `layer_multipliers`, which does
  // not affect the row count; index 0 is the released checkpoint's only PLE
  // layer.
  return qwen4_exp::BuildNGramTableLayout(geom, /*ple_layer_index=*/0)
      .padded_vocab_size;
}

}  // namespace

std::vector<std::string> EnumerateQwen4ExpGgufTensors(
    const Qwen4ExpParams& params) {
  std::vector<std::string> names;
  names.push_back("token_embd.weight");
  // ENUMERATED UNCONDITIONALLY, and the released checkpoint says
  // `tie_word_embeddings: false`, so it is there. A tied file omits it, and
  // then `accounted_tensors` comes back one short of `enumerated_tensors` —
  // which is a VISIBLE discrepancy rather than a silent one, and is why the
  // load reads tie off the FILE (`HasTensor`) instead of off this list. This
  // function takes only the config and the config cannot know.
  names.push_back("output.weight");
  if (!params.ple.layer_ids_zero_based.empty()) {
    names.push_back("per_layer_token_embd.weight");
  }
  names.push_back("output_hc_norm.weight");
  names.push_back("output_hc_down.weight");
  names.push_back("output_hc_up.weight");

  const std::set<int64_t> ple_layers(params.ple.layer_ids_zero_based.begin(),
                                     params.ple.layer_ids_zero_based.end());
  for (int64_t il = 0; il < params.num_hidden_layers; ++il) {
    for (const char* side : {"attn", "ffn"}) {
      const std::string pfx = std::string("hc_") + side + "_";
      names.push_back(Blk(il, (pfx + "norm.weight").c_str()));
      names.push_back(Blk(il, (pfx + "down.weight").c_str()));
      names.push_back(Blk(il, (pfx + "up.weight").c_str()));
      names.push_back(Blk(il, (pfx + "inject.weight").c_str()));
    }
    names.push_back(Blk(il, "ffn_gate_inp.weight"));
    names.push_back(Blk(il, "ffn_gate_inp_shexp.weight"));
    names.push_back(Blk(il, "ffn_gate_exps.weight"));
    names.push_back(Blk(il, "ffn_up_exps.weight"));
    names.push_back(Blk(il, "ffn_down_exps.weight"));
    names.push_back(Blk(il, "ffn_gate_shexp.weight"));
    names.push_back(Blk(il, "ffn_up_shexp.weight"));
    names.push_back(Blk(il, "ffn_down_shexp.weight"));

    if (params.layer_types[static_cast<size_t>(il)] ==
        Qwen4ExpLayerKind::kLinearAttention) {
      names.push_back(Blk(il, "attn_qkv.weight"));
      names.push_back(Blk(il, "attn_gate.weight"));
      names.push_back(Blk(il, "ssm_a"));
      names.push_back(Blk(il, "ssm_alpha.weight"));
      names.push_back(Blk(il, "ssm_beta.weight"));
      names.push_back(Blk(il, "ssm_conv1d.weight"));
      names.push_back(Blk(il, "ssm_dt.bias"));
      names.push_back(Blk(il, "ssm_norm.weight"));
      names.push_back(Blk(il, "ssm_out.weight"));
    } else {
      names.push_back(Blk(il, "attn_q.weight"));
      names.push_back(Blk(il, "attn_k.weight"));
      names.push_back(Blk(il, "attn_v.weight"));
      names.push_back(Blk(il, "attn_output.weight"));
      names.push_back(Blk(il, "attn_q_norm.weight"));
      names.push_back(Blk(il, "attn_k_norm.weight"));
      names.push_back(Blk(il, "indexer.q_proj.weight"));
      names.push_back(Blk(il, "indexer.k_proj.weight"));
      names.push_back(Blk(il, "indexer.q_norm.weight"));
      names.push_back(Blk(il, "indexer.k_norm.weight"));
    }

    if (ple_layers.count(il) != 0) {
      names.push_back(Blk(il, "ple_key.weight"));
      names.push_back(Blk(il, "ple_value.weight"));
      names.push_back(Blk(il, "ple_norm_key.weight"));
      names.push_back(Blk(il, "ple_norm_query.weight"));
      names.push_back(Blk(il, "ple_norm_conv.weight"));
      names.push_back(Blk(il, "ple_conv1d.weight"));
    }
  }
  return names;
}

Qwen4ExpWeights LoadQwen4ExpFromGguf(const GgufFile& gguf,
                                     const HfConfig& config,
                                     vt::DeviceType device,
                                     const GgufLoadPolicy* policy) {
  const GgufLoadPolicy pol =
      policy != nullptr ? *policy : GgufLoadPolicy::FromEnv(device);

  Qwen4ExpWeights w;
  // The SAME resolver the config hook runs, so a file whose metadata the
  // validator would reject is rejected here too rather than half-loaded.
  w.params = ParseQwen4ExpParams(config);
  const Qwen4ExpParams& p = w.params;

  // ── THE DEVICE GATE, AND IT COMES BEFORE ANY TENSOR I/O (#2083) ───────────
  //
  // `DeviceQuantGatherSupported` is true for `kCPU` and `kCUDA` — the two
  // `Embedding` kernels that decode a block row (the CUDA one since KGATHER,
  // src/vt/cuda/cuda_quant_dequant.cuh). On METAL, VULKAN, ROCM and
  // TENSTORRENT, whose gather kernels each assert a float table by name,
  // `RouteGgufTensor` therefore sends
  // `per_layer_token_embd.weight` to `kExpandBf16` — and on the shipped
  // `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` that tensor is
  // [320001536, 160] IQ4_NL: 26.822 GiB on disk becomes
  // 320001536 * 160 * 2 = 102,400,491,520 bytes = 95.368 GiB of ANONYMOUS host
  // memory, on a box with ~119.6 GiB for everything.
  //
  // NOTHING UPSTREAM OF HERE CAN SEE THAT. The #1123 device-fit guard in
  // `entrypoints/model_loader.cpp` sums the file's ON-DISK bytes, which are
  // 67.554 GiB for this artifact and comfortably inside the budget, so it
  // admits the load and the expansion happens after it. That guard's own
  // comment names the outcome: "Loading for 26 minutes and dying mid-stream is
  // the worst of the available behaviours."
  //
  // So a device with no gather arm is REFUSED BY NAME, naming the missing part,
  // which is what AGENTS.md requires of an unimplemented arm. CUDA was the
  // device this text was written about and it is no longer refused; the guard
  // is unchanged because the OTHER four devices still need it, and because a
  // guard deleted the moment its first caller passes is a guard that stops
  // protecting the next one. It is keyed on the DEVICE and
  // not on the residency this policy happens to resolve: a CPU load with
  // keep-quant off expands the same table, and on any file a CPU can hold that
  // is a correct, supported, small load rather than this one.
  //
  // Only when the config actually NAMES a PLE layer. A `qwen4exp` config with
  // no PLE has no n-gram table, nothing gathers from blocks, and there is
  // nothing to refuse.
  if (!p.ple.layer_ids_zero_based.empty() &&
      !DeviceQuantGatherSupported(device)) {
    throw std::runtime_error(
        std::string("qwen4_exp gguf: `per_layer_token_embd.weight` is the "
                    "n-gram gather table and it must stay block-resident, but "
                    "device '") +
        vt::DeviceTypeName(device) +
        "' has no block-decoding gather kernel, so the table would expand to "
        "bf16 — 95.4 GiB of host memory on the released checkpoint, which the "
        "on-disk device-fit guard (issue #1123) cannot see. CPU and CUDA both "
        "gather blocks; this device's arm is owed (issue #2394). Load this "
        "model with --device cpu — CUDA gathers blocks and will LOAD, but this "
        "model's forward has no CUDA arm yet, so it would fail a step later. "
        "See "
        ".agents/specs/cuda-quant-gather.md and issue #2083.");
  }

  // STRUCTURAL accounting. `enumerated` is what the name map expects for this
  // config; `accounted` is how many of those the file carries. The load below
  // reads through the SAME names, so the two can never drift apart.
  const std::vector<std::string> expected = EnumerateQwen4ExpGgufTensors(p);
  w.enumerated_tensors = static_cast<int64_t>(expected.size());
  std::set<std::string> present;
  for (const GgufTensorInfo& info : gguf.Tensors()) present.insert(info.name);
  for (const std::string& n : expected) {
    if (present.count(n) != 0) ++w.accounted_tensors;
  }

  const int64_t h = p.hidden_size;
  const int64_t v = p.vocab_size;

  // The token table is a plain gather and expands to bf16, as every other token
  // table in this tree does.
  w.embed_tokens = ExpandBf16(gguf, "token_embd.weight", {v, h}, /*nk=*/false);

  // TIE IS READ OFF THE FILE, not off a config key. llama.cpp's writer omits
  // `output.weight` exactly when the head is tied, so the file is the authority
  // and a config that disagreed with it would be the config's error. The shipped
  // artifact carries both, so this model is untied.
  w.tied_word_embeddings = !HasTensor(gguf, "output.weight");
  if (!w.tied_word_embeddings) {
    w.lm_head = LoadMatmul(gguf, pol, "output.weight", v, h);
  }

  // The n-gram table. `kEmbeddingTable` is the role W6a (#1989) made keep-quant
  // eligible, and this is the tensor that change exists for: 51.2 G parameters
  // that would otherwise expand from 28.8 GB of Q4_K to 102.4 GB of bf16 on a
  // ~119.6 GiB box. Since KGATHER the same route holds on CUDA: the device
  // decodes a block row, `DeviceQuantGatherSupported` admits it, and the table
  // stays block-resident on the card instead of expanding to 95.4 GiB of host
  // bf16 — which is what makes a GPU arm of this model possible at all.
  if (!p.ple.layer_ids_zero_based.empty()) {
    const int64_t rows = NgramTableRows(p);
    const int64_t cols = p.ple.head_dim_per_ngram();
    const std::string nm = "per_layer_token_embd.weight";
    const GgufTensorInfo& t = gguf.Get(nm);
    RequireShape(t, {rows, cols});
    const GgufResidency r = pol.Route(t, GgufTensorRole::kEmbeddingTable);
    if (r == GgufResidency::kKeepQuant) {
      w.ngram_table = OwnGgufQuantBlocks(t, rows, cols, /*row_offset=*/0,
                                         MmapSrc(gguf, pol),
                                         /*repack=*/false);
      // A gather table is read row-wise, never dotted, so it must NOT carry the
      // matmul orientation flag: `nk` is what tells a consumer this is a
      // MatmulBT operand.
      w.ngram_table.nk = false;
    } else if (r == GgufResidency::kKeepF16) {
      w.ngram_table = OwnGgufF16(t, rows, cols, /*row_offset=*/0,
                                 MmapSrc(gguf, pol), /*nk=*/false);
    } else {
      w.ngram_table = ExpandBf16(gguf, nm, {rows, cols}, /*nk=*/false);
    }
  }

  // The mixer. `use_combine=false`, so no `inject`, and its `hc_norm` is the
  // LAST normalization in the model — there is no `output_norm.weight` in this
  // architecture and the manifest confirms the absence.
  w.mixer = LoadGatedResidual(gguf, pol, p, "output_hc_", /*has_inject=*/false);

  const std::set<int64_t> ple_layers(p.ple.layer_ids_zero_based.begin(),
                                     p.ple.layer_ids_zero_based.end());
  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t il = 0; il < p.num_hidden_layers; ++il) {
    Qwen4ExpLayerWeights& lw = w.layers[static_cast<size_t>(il)];
    lw.is_linear_attention =
        p.layer_types[static_cast<size_t>(il)] ==
        Qwen4ExpLayerKind::kLinearAttention;
    lw.attn_hc = LoadGatedResidual(gguf, pol, p, Blk(il, "hc_attn_"),
                                   /*has_inject=*/true);
    lw.mlp_hc = LoadGatedResidual(gguf, pol, p, Blk(il, "hc_ffn_"),
                                  /*has_inject=*/true);
    if (lw.is_linear_attention) {
      lw.gdn = LoadGdn(gguf, pol, p, il);
    } else {
      lw.qsa = LoadQsa(gguf, pol, p, il);
    }
    lw.moe = LoadMoe(gguf, pol, p, il);
    lw.has_ple = ple_layers.count(il) != 0;
    if (lw.has_ple) lw.ple = LoadPle(gguf, pol, p, il);
  }
  return w;
}

}  // namespace vllm
