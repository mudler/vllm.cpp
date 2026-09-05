// vllm.cpp ORIGINAL — the `glm5next` GGUF weight loader. See
// `glm5_next_loader.h` for the two oracles this file answers to, the four
// convert-time transforms it inverts, the one it deliberately does NOT inherit,
// and why the GGUF arm is the only arm.
//
// Tensor names and metadata keys follow llama.cpp pull request
// [#27752](https://github.com/ggml-org/llama.cpp/pull/27752) at head
// `8a8d0bcc4d5fdf024c457526245bec4bc3a12adc`, which is the layout the shipped
// `unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL` file uses. That pull request is STILL
// OPEN; if it renames a key before it merges, this file changes and
// `tests/vllm/models/glm5_next_gguf_manifest.inc` is regrown.
#include "vllm/model_executor/models/glm5_next_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/models/glm5_next_diag.h"
#include "vllm/model_executor/models/glm5_next_weights.h"     // the name map
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"  // OwnGgufQuantBlocks
#include "vt/quant.h"  // vt::cpu::QuantRepackActive
#include "vt/dtype.h"

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
  VT_CHECK(false, "glm5_next gguf: shape mismatch for " + t.name + ": got [" +
                      got + "], expected [" + exp + "]");
}

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

bool HasTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors())
    if (t.name == name) return true;
  return false;
}

OwnedTensor MakeTensor(vt::DType dtype, const std::vector<int64_t>& shape,
                       bool nk, size_t elem_bytes) {
  OwnedTensor o;
  o.dtype = dtype;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "glm5_next gguf: rank exceeds kMaxRank");
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
           "glm5_next gguf: dequant length mismatch for " + name);
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
  OwnedTensor o =
      MakeTensor(vt::DType::kF32, shape, /*nk=*/false, sizeof(float));
  std::memcpy(o.bytes.data(), f.data(), f.size() * sizeof(float));
  return o;
}

// Dequantize a whole tensor into an owned bf16 buffer in the file's own order.
OwnedTensor ExpandBf16(const GgufFile& g, const std::string& name,
                       const std::vector<int64_t>& shape, bool nk) {
  return Bf16From(DequantAll(g, name, shape), shape, nk);
}

// THE REPACK IS DECLINED ON THIS ROW, and the constant exists so the three
// call sites below say so once rather than three times.
//
// `GgufLoadPolicy::quant_repack` is `keep_quant && !cpu_ref &&
// vt::cpu::QuantRepackActive() && dev == kCPU` (the device term is #2406), and
// `QuantRepackActive()` is TRUE on every aarch64 i8mm box in this fleet, which
// is where this row runs. It
// permutes an eligible q8_0 weight into the `block_q8_0x4` interleave for the
// CPU i8mm GEMM. **Nothing on this row consumes that layout.** The forward is a
// host f32 reference (`glm5_next_forward.h`): every weight it touches goes
// through `DecodeOwnedTensorToF32` / `DecodeOwnedTensorRowsToF32`, which decode
// blocks with `vt::cpu::BlockToFloat(dtype)`. No `vt::Tensor` is ever built
// over these buffers, so `QuantRepackMatmul` is unreachable from here and the
// repack buys this model nothing while costing it a load-time copy that
// defeats the mmap borrow.
//
// What it did cost was the model. The interleave keeps the dtype at `kQ8_0`
// and the byte count identical, so it passed every check the bridge had and
// the bridge read it as plain blocks: 346 Q8_0 tensors on the published
// artifact -- both mHC mixers on all 45 layers, the whole KDA gate chain, and
// the DSA and indexer projections -- decoded to wrong values with no error,
// and the model emitted token id 0 for every position. Declining the repack is
// the repair; `DecodeOwnedTensorToF32` refusing a repacked buffer by name is
// the guard that keeps it from coming back silently. Issue #2241.
//
// A later wave that gives this row a quantized GEMM should turn this back on
// AT THAT SEAM, and will find the bridge's refusal waiting if it forgets one.
constexpr bool kGlm5NextQuantRepack = false;

// The elementwise sibling, declined for the same reason and stated separately
// because it is a different flag with a different consequence: it transposes an
// F16 or bf16 `[N,K]` weight to `[K,N]` while LEAVING THE SHAPE at `[N,K]`, so
// this bridge's `memcpy`/widen would read the wrong axis. It is opt-in
// (`VT_CPU_ELEM_KN_REPACK=1`) and the published artifact carries no F16 tensor,
// so this is not live today -- which is exactly why it is declined now rather
// than after somebody sets that variable.
constexpr bool kGlm5NextElemKnRepack = false;

const GgufFile* MmapSrc(const GgufFile& g, const GgufLoadPolicy& pol) {
  return pol.mmap_residency ? &g : nullptr;
}

// One [n] norm gamma as bf16.
//
// THERE IS NO `+1` TO INVERT and the header says why: the DeepSeek/GLM
// conversion chain this architecture goes through never folds a norm, unlike
// the Qwen3-Next one the sibling `qwen4exp` loader has to undo. This function
// takes no `unshift` argument for exactly that reason — a defaulted one is how
// the wrong convention gets inherited by the next tensor added below it.
OwnedTensor LoadNormBf16(const GgufFile& g, const std::string& name, int64_t n) {
  return Bf16From(DequantAll(g, name, {n}), {n}, /*nk=*/false);
}

// One [n] vector kept at f32. Every call site is an annotated exception and
// names its reason at the field in `glm5_next_loader.h`.
OwnedTensor LoadVecF32(const GgufFile& g, const std::string& name, int64_t n) {
  return F32From(DequantAll(g, name, {n}), {n});
}

// One standalone [N, K] matmul operand taken VERBATIM: kept as raw ggml blocks
// when the policy routes it there, kept as F16 when it routes it there,
// expanded to bf16 in the file's own [N, K] order otherwise.
OwnedTensor LoadMatmul(const GgufFile& g, const GgufLoadPolicy& pol,
                       const std::string& name, int64_t n, int64_t k) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {n, k});
  const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
  if (r == GgufResidency::kKeepQuant)
    return OwnGgufQuantBlocks(t, n, k, /*row_offset=*/0, MmapSrc(g, pol),
                              kGlm5NextQuantRepack);
  if (r == GgufResidency::kKeepF16)
    return OwnGgufF16(t, n, k, /*row_offset=*/0, MmapSrc(g, pol), /*nk=*/true,
                      kGlm5NextElemKnRepack);
  return ExpandBf16(g, name, {n, k}, /*nk=*/true);
}

// One [N, K] matmul operand held at f32 because upstream computes it at f32.
// `Glm5NextTextTopkRouter` is the only user (`modeling_glm5_next.py:158`), and
// the published file already stores every `ffn_gate_inp.weight` as F32, so this
// is a mirror rather than a widening.
OwnedTensor LoadMatmulF32(const GgufFile& g, const std::string& name, int64_t n,
                          int64_t k) {
  OwnedTensor o = F32From(DequantAll(g, name, {n, k}), {n, k});
  o.nk = true;
  return o;
}

// A 3-D [E, N, K] stacked expert tensor. Each expert slab is a whole number of
// ROWS and therefore a whole number of blocks, so the keep-quant arm is a byte
// range and no block is ever cut — which is what lets a 288-expert IQ2_XS
// tensor stay resident at all.
OwnedTensor LoadStackedExperts(const GgufFile& g, const GgufLoadPolicy& pol,
                               const std::string& name, int64_t e, int64_t n,
                               int64_t k) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {e, n, k});
  const GgufResidency r = pol.Route(t, GgufTensorRole::kStackedExpertWeight);
  if (r == GgufResidency::kKeepQuant) {
    // `OwnGgufQuantBlocks` works in rows, so the stack is flattened to [E*N, K]
    // and reshaped back. The bytes are identical either way; only the recorded
    // shape differs, and the consumer slices by expert.
    OwnedTensor o = OwnGgufQuantBlocks(t, e * n, k, /*row_offset=*/0,
                                       MmapSrc(g, pol), kGlm5NextQuantRepack);
    o.rank = 3;
    o.shape[0] = e;
    o.shape[1] = n;
    o.shape[2] = k;
    return o;
  }
  return ExpandBf16(g, name, {e, n, k}, /*nk=*/true);
}

// A 3-D [E, N, K] tensor that is NOT an expert bank and is never sliced by
// expert: the two absorbed MLA halves, whose leading axis is the attention
// head. `kMatmulWeight` is the role, because that is what they are — and asking
// the policy for `kStackedExpertWeight` here would put an 11-layer,
// 8-MiB-per-layer tensor in the class the expert-streaming seam reasons about.
OwnedTensor LoadHeadStacked(const GgufFile& g, const GgufLoadPolicy& pol,
                            const std::string& name, int64_t h, int64_t n,
                            int64_t k) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {h, n, k});
  const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
  if (r == GgufResidency::kKeepQuant) {
    OwnedTensor o = OwnGgufQuantBlocks(t, h * n, k, /*row_offset=*/0,
                                       MmapSrc(g, pol), kGlm5NextQuantRepack);
    o.rank = 3;
    o.shape[0] = h;
    o.shape[1] = n;
    o.shape[2] = k;
    return o;
  }
  return ExpandBf16(g, name, {h, n, k}, /*nk=*/true);
}

// ── transform 1: `ssm_a = -exp(A_log)`, recovered as `A_log = log(-x)` ───────
//
// `conversion/glm5next.py` at the pinned head:
//   `if name.endswith(".A_log"): data_torch = -torch.exp(data_torch.float())`
//
// The inverse is `log(-x)`, which is defined only for a STRICTLY NEGATIVE `x`.
// That is not a formality here. `exp()` has no zeros and no positive-to-
// negative crossing, so every value the converter wrote is strictly negative,
// and a non-negative one means the file was NOT written by that transform. The
// alternative to refusing is `log` of a non-positive number: `-inf` for zero
// and NaN for a positive, propagated through `exp(A_log)` into every decay in
// the layer, which reads downstream as a dead or diverged sequence and never as
// a bad file.
OwnedTensor LoadALog(const GgufFile& g, const std::string& name,
                     int64_t num_heads) {
  std::vector<float> f = DequantAll(g, name, {num_heads});
  for (size_t i = 0; i < f.size(); ++i) {
    VT_CHECK(f[i] < 0.0F,
             "glm5_next gguf: " + name + "[" + std::to_string(i) + "] is " +
                 std::to_string(f[i]) +
                 ", but the converter writes this tensor as `-exp(A_log)` "
                 "(llama.cpp #27752, conversion/glm5next.py), which is "
                 "strictly negative for every input; a non-negative entry "
                 "means this file states the KDA decay by some other "
                 "convention and `log(-x)` would yield NaN or -inf");
    f[i] = std::log(-f[i]);
  }
  return F32From(f, {num_heads});
}

// ── the per-module loads ────────────────────────────────────────────────────

// `blk.N.hc_{attn,ffn}_{fn,base,scale}.weight` — FLAT on the layer, not under
// an `attn_hc.` prefix. `side` is "attn" or "ffn".
Glm5NextMhcWeights LoadMhc(const GgufFile& g, const GgufLoadPolicy& pol,
                           const Glm5NextParams& p, int64_t il,
                           const char* side) {
  // `mix = (2 + hc_mult) * hc_mult` (`modeling_glm5_next.py:258`) — 24 on the
  // published checkpoint, and it is NOT `hc_mult` and NOT `hc_mult * hc_mult`.
  // The three outputs `pre`, `post` and `comb` are split `[hc, hc, hc * hc]`
  // (`:278`), which is where the `2 +` comes from.
  const int64_t mix = (2 + p.mhc.mult) * p.mhc.mult;
  const int64_t stream = p.residual_stream_width();
  const std::string pfx = std::string("hc_") + side + "_";
  Glm5NextMhcWeights w;
  w.fn = LoadMatmul(g, pol, Blk(il, (pfx + "fn.weight").c_str()), mix, stream);
  w.base = LoadVecF32(g, Blk(il, (pfx + "base.weight").c_str()), mix);
  // `self.scale` is [3] — pre, post and comb, in that order (`:265`, `:281`).
  w.scale = LoadVecF32(g, Blk(il, (pfx + "scale.weight").c_str()), 3);
  return w;
}

// One [qkv_dim, 1, K] depthwise conv, stored [qkv_dim, K].
//
// The middle axis is `nn.Conv1d`'s `in_channels / groups`, which is 1 for a
// fully depthwise conv, and it is REQUIRED to be 1 rather than dropped
// silently: a file whose conv is not depthwise carries a different operator,
// and squeezing a middle axis of 3 would reinterpret its bytes as three times
// as many channels.
OwnedTensor LoadDepthwiseConv(const GgufFile& g, const std::string& name,
                              int64_t channels, int64_t kernel) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {channels, 1, kernel});
  std::vector<float> f =
      DequantGgufRowToF32(t.ggml_type, t.data, channels * kernel);
  VT_CHECK(static_cast<int64_t>(f.size()) == channels * kernel,
           "glm5_next gguf: dequant length mismatch for " + name);
  return Bf16From(f, {channels, kernel}, /*nk=*/false);
}

Glm5NextKdaWeights LoadKda(const GgufFile& g, const GgufLoadPolicy& pol,
                           const Glm5NextParams& p, int64_t il) {
  const int64_t h = p.hidden_size;
  const int64_t heads = p.kda.num_heads;
  const int64_t hd = p.kda.head_dim;
  const int64_t qkv = heads * hd;
  const int64_t kk = p.kda.conv_kernel_dim;

  Glm5NextKdaWeights w;
  w.q_proj = LoadMatmul(g, pol, Blk(il, "attn_q.weight"), qkv, h);
  w.k_proj = LoadMatmul(g, pol, Blk(il, "attn_k.weight"), qkv, h);
  w.v_proj = LoadMatmul(g, pol, Blk(il, "attn_v.weight"), qkv, h);
  w.o_proj = LoadMatmul(g, pol, Blk(il, "attn_output.weight"), h, qkv);
  w.q_conv1d = LoadDepthwiseConv(g, Blk(il, "ssm_conv1d_q.weight"), qkv, kk);
  w.k_conv1d = LoadDepthwiseConv(g, Blk(il, "ssm_conv1d_k.weight"), qkv, kk);
  w.v_conv1d = LoadDepthwiseConv(g, Blk(il, "ssm_conv1d_v.weight"), qkv, kk);
  w.f_a_proj = LoadMatmul(g, pol, Blk(il, "ssm_f_a.weight"), hd, h);
  w.f_b_proj = LoadMatmul(g, pol, Blk(il, "ssm_f_b.weight"), qkv, hd);
  w.g_a_proj = LoadMatmul(g, pol, Blk(il, "ssm_g_a.weight"), hd, h);
  w.g_b_proj = LoadMatmul(g, pol, Blk(il, "ssm_g_b.weight"), qkv, hd);
  // `b_proj` is ONE ROW PER HEAD, not per channel: `nn.Linear(hidden_size,
  // self.num_heads)` (`:618`). A port that sized it `[qkv_dim, hidden]` reads
  // 128x too many rows and the shape check below is what says so.
  w.b_proj = LoadMatmul(g, pol, Blk(il, "ssm_beta.weight"), heads, h);
  w.a_log = LoadALog(g, Blk(il, "ssm_a"), heads);
  // `dt_bias` is `[qkv_dim]` and `A_log` is `[num_heads]` — the forget gate
  // declares them one line apart at DIFFERENT widths (`:314-315`), and they
  // are the pair most likely to be sized alike by mistake.
  w.dt_bias = LoadVecF32(g, Blk(il, "ssm_dt.bias"), qkv);
  w.o_norm = LoadNormBf16(g, Blk(il, "ssm_norm.weight"), hd);
  return w;
}

Glm5NextIndexerWeights LoadIndexer(const GgufFile& g,
                                   const GgufLoadPolicy& pol,
                                   const Glm5NextParams& p, int64_t il) {
  const int64_t h = p.hidden_size;
  const int64_t n_heads = p.indexer.n_heads;
  const int64_t hd = p.indexer.head_dim;
  Glm5NextIndexerWeights w;
  w.wq_b = LoadMatmul(g, pol, Blk(il, "indexer.attn_q_b.weight"), n_heads * hd,
                      p.mla.q_lora_rank);
  w.wk = LoadMatmul(g, pol, Blk(il, "indexer.attn_k.weight"), hd, h);
  w.k_norm_weight = LoadNormBf16(g, Blk(il, "indexer.k_norm.weight"), hd);
  // The BIAS is what makes this a LayerNorm rather than an RMSNorm, so it is
  // required rather than optional: a file without it is a file whose indexer
  // normalization is a different operator.
  w.k_norm_bias = LoadNormBf16(g, Blk(il, "indexer.k_norm.bias"), hd);
  // `weights_proj` is `nn.Linear(hidden_size, self.n_heads)` (`:764`) — one
  // row per INDEXER head (32), not per MLA head (64) and not per channel.
  w.weights_proj = LoadMatmul(g, pol, Blk(il, "indexer.proj.weight"), n_heads, h);
  // `index_kpool_compress_ape` is `[index_kpool, head_dim]` (`:770`) — the
  // POOL width leads, and it is 4 on this checkpoint against a class default of
  // 16, so a loader that defaulted instead of reading is wrong by 4x and the
  // shape check is what catches it.
  w.kpool_ape = LoadMatmul(g, pol, Blk(il, "indexer_compressor_ape.weight"),
                           p.indexer.kpool, hd);
  w.kpool_gate = LoadMatmul(g, pol, Blk(il, "indexer_compressor_gate.weight"),
                            hd, h);
  return w;
}

Glm5NextMlaWeights LoadMla(const GgufFile& g, const GgufLoadPolicy& pol,
                           const Glm5NextParams& p, int64_t il) {
  const int64_t h = p.hidden_size;
  const int64_t heads = p.num_attention_heads;
  const int64_t q_lora = p.mla.q_lora_rank;
  const int64_t kv_lora = p.mla.kv_lora_rank;
  const int64_t qk_nope = p.mla.qk_nope_head_dim;
  const int64_t qk_rope = p.mla.qk_rope_head_dim;
  // `qk_head_dim = qk_rope_head_dim + qk_nope_head_dim`, forced by upstream and
  // never read from the config (`glm5_next.h`, `Glm5NextMlaParams`).
  const int64_t qk_head = qk_nope + qk_rope;
  const int64_t v_head = p.mla.v_head_dim;

  Glm5NextMlaWeights w;
  w.q_a_proj = LoadMatmul(g, pol, Blk(il, "attn_q_a.weight"), q_lora, h);
  w.q_a_layernorm = LoadNormBf16(g, Blk(il, "attn_q_a_norm.weight"), q_lora);
  w.q_b_proj =
      LoadMatmul(g, pol, Blk(il, "attn_q_b.weight"), heads * qk_head, q_lora);
  // `nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim)` (`:1111-1115`).
  // On this NoPE model `qk_rope_head_dim` is 0, so the sum is the latent
  // itself; it is written as a sum anyway so a rotary variant of this geometry
  // reads correctly rather than silently dropping the rope slice.
  w.kv_a_proj_with_mqa =
      LoadMatmul(g, pol, Blk(il, "attn_kv_a_mqa.weight"), kv_lora + qk_rope, h);
  w.kv_a_layernorm = LoadNormBf16(g, Blk(il, "attn_kv_a_norm.weight"), kv_lora);
  // Transform 3. `k_b` was `.transpose(1, 2)`-ed by the converter, so its
  // trailing axis is `qk_nope_head_dim` while `v_b`'s is `kv_lora_rank`. The
  // two are NOT the same shape on this model even though `qk_nope_head_dim`
  // and `v_head_dim` are both 256: `[64, 512, 256]` against `[64, 256, 512]`.
  w.k_b_proj =
      LoadHeadStacked(g, pol, Blk(il, "attn_k_b.weight"), heads, kv_lora, qk_nope);
  w.v_b_proj =
      LoadHeadStacked(g, pol, Blk(il, "attn_v_b.weight"), heads, v_head, kv_lora);
  w.o_proj =
      LoadMatmul(g, pol, Blk(il, "attn_output.weight"), h, heads * v_head);
  w.indexer = LoadIndexer(g, pol, p, il);
  return w;
}

Glm5NextMlpWeights LoadMlp(const GgufFile& g, const GgufLoadPolicy& pol,
                           const Glm5NextParams& p, int64_t il,
                           const char* gate, const char* up, const char* down,
                           int64_t inter) {
  const int64_t h = p.hidden_size;
  Glm5NextMlpWeights w;
  w.gate_proj = LoadMatmul(g, pol, Blk(il, gate), inter, h);
  w.up_proj = LoadMatmul(g, pol, Blk(il, up), inter, h);
  w.down_proj = LoadMatmul(g, pol, Blk(il, down), h, inter);
  return w;
}

Glm5NextMoeWeights LoadMoe(const GgufFile& g, const GgufLoadPolicy& pol,
                           const Glm5NextParams& p, int64_t il) {
  const int64_t h = p.hidden_size;
  const int64_t e = p.moe.n_routed_experts;
  const int64_t mi = p.moe.moe_intermediate_size;
  Glm5NextMoeWeights w;
  w.router = LoadMatmulF32(g, Blk(il, "ffn_gate_inp.weight"), e, h);
  w.e_score_correction_bias = LoadVecF32(g, Blk(il, "exp_probs_b.bias"), e);
  w.gate_exps = LoadStackedExperts(g, pol, Blk(il, "ffn_gate_exps.weight"), e, mi, h);
  w.up_exps = LoadStackedExperts(g, pol, Blk(il, "ffn_up_exps.weight"), e, mi, h);
  w.down_exps = LoadStackedExperts(g, pol, Blk(il, "ffn_down_exps.weight"), e, h, mi);
  // The shared expert is sized `moe_intermediate_size * n_shared_experts`
  // (`:196-198`), which is 2048 * 1 here. It is NOT `intermediate_size`, the
  // 12288 the three dense layers use.
  w.shared = LoadMlp(g, pol, p, il, "ffn_gate_shexp.weight",
                     "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                     mi * p.moe.n_shared_experts);
  return w;
}

}  // namespace

int64_t Glm5NextWeights::num_kda_layers() const {
  int64_t n = 0;
  for (const Glm5NextLayerWeights& l : layers)
    if (l.is_linear_attention) ++n;
  return n;
}

int64_t Glm5NextWeights::num_dsa_layers() const {
  return static_cast<int64_t>(layers.size()) - num_kda_layers();
}

std::vector<std::string> EnumerateGlm5NextGgufTensors(
    const Glm5NextParams& params) {
  // ONE enumeration, not two. `Glm5NextExpectedGgufTensors` is the public name
  // map W1 landed and the converter's counterpart; the load below reads through
  // exactly the names it returns, so a divergence between what this row expects
  // and what the converter writes is a compile-time-shared fact rather than two
  // transcriptions that can drift.
  return Glm5NextExpectedGgufTensors(params);
}

Glm5NextWeights LoadGlm5NextFromGguf(const GgufFile& gguf,
                                     const HfConfig& config,
                                     const GgufLoadPolicy* policy) {
  const GgufLoadPolicy pol =
      policy != nullptr ? *policy : GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);

  // WHAT THIS BOX WOULD HAVE DONE, printed beside what this row does instead.
  // The repack defect (#2241) was reachable only where `QuantRepackActive()` is
  // true, so an investigation that does not record that bit cannot tell a box
  // that never repacked from a repair that worked. This line is the record.
  if (glm5_next::diag::Level() > 0) {
    glm5_next::diag::Banner("LoadGlm5NextFromGguf");
    std::fprintf(stderr,
                 "[glm5-diag] load policy: keep_quant=%d cpu_ref=%d "
                 "policy.quant_repack=%d QuantRepackActive()=%d "
                 "policy.elem_kn_repack=%d mmap_residency=%d | this row uses "
                 "quant_repack=%d elem_kn_repack=%d\n",
                 static_cast<int>(pol.keep_quant), static_cast<int>(pol.cpu_ref),
                 static_cast<int>(pol.quant_repack),
                 static_cast<int>(vt::cpu::QuantRepackActive()),
                 static_cast<int>(pol.elem_kn_repack),
                 static_cast<int>(pol.mmap_residency),
                 static_cast<int>(kGlm5NextQuantRepack),
                 static_cast<int>(kGlm5NextElemKnRepack));
    std::fflush(stderr);
  }

  Glm5NextWeights w;
  // The SAME resolver the config hook runs, so a file whose metadata the
  // validator would reject is rejected here too rather than half-loaded.
  w.params = ParseGlm5NextParams(config);
  const Glm5NextParams& p = w.params;

  // The vision tower is W6's and is not in this container at all: the published
  // artifact ships it as a SEPARATE `mmproj-BF16.gguf` whose
  // `clip.projector_type` is `glm5next`, and llama.cpp #27752 drops the vision
  // tensors at convert time (`filter_tensors`: "text-only for now"). A config
  // that declares a vision tower alongside a text-only file would enumerate
  // tensors this loader would then fail to find, one at a time and by the wrong
  // name, so it is refused up front and by the right name.
  VT_CHECK(!p.has_vision,
           "glm5_next gguf: this config declares a vision tower "
           "(`vision_config`), but the `glm5next` container is TEXT-ONLY — the "
           "published artifact ships its tower as a separate `mmproj` file and "
           "llama.cpp #27752 drops the vision tensors at convert time. The "
           "vision arm is owed (W6). See .agents/specs/glm5-next-flash.md and "
           "issue #1998.");

  // STRUCTURAL accounting. `enumerated` is what the name map expects for this
  // config; `accounted` is how many of those the file carries. The load below
  // reads through the SAME names, so the two can never drift apart.
  const std::vector<std::string> expected = EnumerateGlm5NextGgufTensors(p);
  w.enumerated_tensors = static_cast<int64_t>(expected.size());
  std::set<std::string> present;
  for (const GgufTensorInfo& info : gguf.Tensors()) present.insert(info.name);
  for (const std::string& n : expected) {
    if (present.count(n) != 0) ++w.accounted_tensors;
  }

  // THE MTP BLOCKS, COUNTED AND THEN NOT BUILT.
  //
  // The published artifact's `blk.45` is a DeepSeek-V3-style multi-token-
  // prediction block that the reference discards
  // (`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", ...]`), and
  // `Glm5NextHfConfigFromGguf` already resolved `num_hidden_layers` as
  // `block_count - nextn_predict_layers` = 45. This counts what the FILE
  // carries past that depth rather than trusting the subtraction, so a test can
  // assert that the loader SAW an MTP block and declined to build it — an
  // assertion `layers.size() == 45` cannot make, because it is equally true of
  // a stack built from blocks 1..45. It is a count of TENSORS, not of blocks,
  // so that `enumerated_tensors + mtp_block_tensors_dropped` is the file's
  // whole table and the two numbers can be checked against each other.
  for (const GgufTensorInfo& info : gguf.Tensors()) {
    if (info.name.rfind("blk.", 0) != 0) continue;
    const size_t dot = info.name.find('.', 4);
    if (dot == std::string::npos) continue;
    const std::string digits = info.name.substr(4, dot - 4);
    if (digits.empty() ||
        digits.find_first_not_of("0123456789") != std::string::npos) {
      continue;
    }
    if (std::stoll(digits) >= p.num_hidden_layers) ++w.mtp_block_tensors_dropped;
  }

  const int64_t h = p.hidden_size;
  const int64_t v = p.vocab_size;

  // The token table is a plain gather and expands to bf16, as every other token
  // table in this tree does.
  w.embed_tokens = ExpandBf16(gguf, "token_embd.weight", {v, h}, /*nk=*/false);
  // `Glm5NextTextModel.norm`, the final RMSNorm applied AFTER the unweighted
  // head collapse (`:1493`, `self.norm(self.hc_head(hidden_states))`). The
  // sibling `qwen4exp` has no such tensor and its last normalization is inside
  // the mixer; copying that tail here would drop a norm this model has.
  w.norm = LoadNormBf16(gguf, "output_norm.weight", h);

  // TIE IS READ OFF THE FILE, not off a config key. llama.cpp's writer omits
  // `output.weight` exactly when the head is tied
  // (`DeepseekV2Model.modify_tensors`: "Skipping tied output layer
  // 'lm_head.weight'"), so the file is the authority and a config that
  // disagreed with it would be the config's error. The published artifact
  // carries both, so this model is untied.
  w.tied_word_embeddings = !HasTensor(gguf, "output.weight");
  if (!w.tied_word_embeddings) {
    w.lm_head = LoadMatmul(gguf, pol, "output.weight", v, h);
  }

  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t il = 0; il < p.num_hidden_layers; ++il) {
    Glm5NextLayerWeights& lw = w.layers[static_cast<size_t>(il)];
    const size_t i = static_cast<size_t>(il);
    lw.is_linear_attention =
        p.layer_types[i] == Glm5NextLayerKind::kLinearAttention;
    lw.is_dense_mlp = p.mlp_layer_types[i] == Glm5NextMlpKind::kDense;

    lw.input_layernorm = LoadNormBf16(gguf, Blk(il, "attn_norm.weight"), h);
    lw.post_attention_layernorm =
        LoadNormBf16(gguf, Blk(il, "ffn_norm.weight"), h);
    lw.attn_hc = LoadMhc(gguf, pol, p, il, "attn");
    lw.mlp_hc = LoadMhc(gguf, pol, p, il, "ffn");

    if (lw.is_linear_attention) {
      lw.kda = LoadKda(gguf, pol, p, il);
    } else {
      lw.mla = LoadMla(gguf, pol, p, il);
    }
    if (lw.is_dense_mlp) {
      lw.dense_mlp = LoadMlp(gguf, pol, p, il, "ffn_gate.weight",
                             "ffn_up.weight", "ffn_down.weight",
                             p.intermediate_size);
    } else {
      lw.moe = LoadMoe(gguf, pol, p, il);
    }
  }
  return w;
}

}  // namespace vllm
