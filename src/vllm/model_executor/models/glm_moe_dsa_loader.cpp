// vllm.cpp ORIGINAL — the `glm-dsa` GGUF weight loader (GLM-5.3). W7 of
// `.agents/specs/glm-dsa-latest-deepseek.md` §3.7, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// ─── WHAT THIS FILE IS FOR ───────────────────────────────────────────────────
// The published artifact is `unsloth/GLM-5.3-GGUF UD-IQ1_S` at revision
// `346b3591c7f28d1a23716f97a065ecf12ec14771`: six shards, 216,715,365,893 B =
// 201.83 GiB, 1809 tensors. 97.49% of this model's parameters are routed
// experts, so the question is never whether the weights fit but whether the
// STEP WORKING SET does. That is what splits this loader in two:
//
//   * 228 stacked `blk.<n>.ffn_{gate,up,down}_exps.weight` towers, 187.312 GiB,
//     kept as raw ggml blocks BORROWED from the mmap and handed to the
//     expert-streaming lane a slice at a time;
//   * 1581 per-layer tensors, 14.511 GiB, resident for the whole run.
//
// Both figures are reproduced from the six shard headers by
// `tests/vllm/models/test_glm_moe_dsa_gguf_census.cpp` rather than quoted.
//
// ─── WHY THE TOWERS MUST STAY STACKED ────────────────────────────────────────
// `expert_stream::ExpertSlice` serves expert `e` as a PURE BYTE OFFSET over
// whole rows of the same K (`gguf_expert_span.h:11-16`). That works only on the
// keep-quant stacked `[E, out, in]` form. The safetensors DeepSeek-V2 loader's
// `std::vector<OwnedTensor>` per expert (`deepseek_v2.h:250-259`) would be
// 57,600 host tensors here and has no streaming source at all — which is
// exactly what `GlmMoeDsaSafetensorsRefusal()` cites (spec D1).
//
// ─── THREE THINGS THE PUBLISHED FILE GETS WRONG, AND WHAT IS DONE ABOUT THEM ─
//  1. It declares `indexer.*` on ALL 79 blocks while the source checkpoint
//     ships them on 22. The conversion broadcast the shared layers' weights.
//     This loader reads the SCHEDULE and drops the surplus, counting it — the
//     same posture upstream takes at `deepseek_v2.py:1566-1582`. Spec D3.
//  2. It writes no `glm-dsa.attention.indexer.types` and no freq/offset, so it
//     states its schedule nowhere and `ParseGlmMoeDsaParams` refuses it by
//     name. That refusal is upstream of this file and is NOT weakened here;
//     `scripts/glm-dsa-write-indexer-types.py` repairs the FILE from the model
//     author's own `config.json`. Spec O17.
//  3. `block_count` is 79 and the backbone is 78. Block 78 is the
//     multi-token-prediction block: read, counted, dropped. Spec O5.
//
// ─── THE REPACK IS DECLINED, AND THE REASON IS NOT PERFORMANCE ───────────────
// `GgufLoadPolicy::quant_repack` is `keep_quant && !cpu_ref &&
// vt::cpu::QuantRepackActive()`, and `QuantRepackActive()` is TRUE on every
// aarch64 i8mm box in this fleet — `dgx:gpu0` included. It permutes an eligible
// Q8_0 weight into the `block_q8_0x4` interleave, KEEPING THE DTYPE AND THE
// BYTE COUNT IDENTICAL. That is what makes it dangerous rather than merely
// slow: every shape and type assertion in this tree passes on a repacked
// buffer, so a consumer that reads it as plain blocks reads wrong values with
// no error anywhere. The sibling `MODEL-MM-GLM53-FLASH` row paid for this
// exactly once (#2241): 346 Q8_0 tensors decoded to garbage and the model
// emitted token id 0 for every position, on an artifact from the same
// publisher with the same Q8_0-heavy resident class.
//
// This arm's resident class is 476 Q8_0 tensors, 4.852 GiB — the MLA
// `attn_q_b` / `attn_kv_a_mqa` / `attn_k_b` / `attn_v_b` set and the whole
// indexer projection set, i.e. precisely the tensors whose values decide the
// attention selection. W7's Exclusions carry no speed number, so the lever buys
// this wave nothing, and NO GATE ON THIS ROW CAN SEE THE DIFFERENCE: the token
// gate this row can reach does not exist (spec O1), and a wrong-valued
// selection is finite, plausible and silent. Declining it is therefore the
// conservative direction and it is recorded rather than inherited. A later wave
// that wants the lever should turn it on at a seam that is gated, and should
// expect `ResidentWeight`'s own refusal of a repacked weight at device staging
// to be waiting for it.
//
// (the constant itself is in the anonymous namespace below)

#include "vllm/model_executor/models/glm_moe_dsa.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"  // OwnGgufQuantBlocks
#include "vt/dtype.h"

namespace vllm {
namespace {

// Declined; the 40 lines above this file's includes say why.
constexpr bool kGlmMoeDsaQuantRepack = false;

std::string Blk(int64_t layer, const char* suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// `GgufTensorInfo::shape` is already REVERSED into torch row-major order by the
// reader (`gguf_reader.cpp:443`), so a 2-D matmul weight reads [N = out,
// K = in] — the file's own orientation and `vt::MatmulBT`'s.
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
  VT_CHECK(false, "glm-dsa gguf: shape mismatch for " + t.name + ": got [" +
                      got + "], expected [" + exp + "]");
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
  VT_CHECK(o.rank <= vt::kMaxRank, "glm-dsa gguf: rank exceeds kMaxRank");
  for (int i = 0; i < o.rank; ++i) o.shape[i] = shape[i];
  o.nk = nk;
  o.bytes.resize(static_cast<size_t>(Numel(shape)) * elem_bytes);
  return o;
}

// Dequantize a whole tensor to f32. Every value rewrite in this file goes
// through f32 rather than bf16, so a rewrite never rounds twice.
std::vector<float> DequantAll(const GgufFile& g, const std::string& name,
                              const std::vector<int64_t>& shape) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, shape);
  const int64_t n = Numel(shape);
  std::vector<float> f = DequantGgufRowToF32(t.ggml_type, t.data, n);
  VT_CHECK(static_cast<int64_t>(f.size()) == n,
           "glm-dsa gguf: dequant length mismatch for " + name);
  return f;
}

OwnedTensor Bf16From(const std::vector<float>& f,
                     const std::vector<int64_t>& shape, bool nk) {
  OwnedTensor o = MakeTensor(vt::DType::kBF16, shape, nk, sizeof(uint16_t));
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (size_t i = 0; i < f.size(); ++i) dst[i] = vt::F32ToBF16(f[i]);
  return o;
}

OwnedTensor F32From(const std::vector<float>& f,
                    const std::vector<int64_t>& shape, bool nk) {
  OwnedTensor o = MakeTensor(vt::DType::kF32, shape, nk, sizeof(float));
  std::memcpy(o.bytes.data(), f.data(), f.size() * sizeof(float));
  return o;
}

OwnedTensor ExpandBf16(const GgufFile& g, const std::string& name,
                       const std::vector<int64_t>& shape, bool nk) {
  return Bf16From(DequantAll(g, name, shape), shape, nk);
}

// A tensor the FILE stores at F32 that upstream builds at the MODEL dtype, held
// at bf16 — and the round-trip PROVED rather than assumed (W9, #2214).
//
// llama.cpp's converter writes several small tensors as F32 regardless of the
// source dtype, so an F32 tensor in a GGUF is usually the lossless upcast of a
// bf16 one. "Usually" is not a fact, and the consequence of being wrong here is
// not a tolerance: `indexer.proj.weight` scales the indexer logits and therefore
// decides a discrete top-k, where the error is bimodal and rounding flips a
// selection or does nothing at all. So every value must round-trip through bf16
// EXACTLY, and a file that is genuinely f32 is refused by name instead of being
// silently rounded into the model dtype.
OwnedTensor Bf16FromExactF32(const GgufFile& g, const std::string& name,
                             const std::vector<int64_t>& shape, bool nk,
                             const char* upstream_reason) {
  const std::vector<float> f = DequantAll(g, name, shape);
  for (size_t i = 0; i < f.size(); ++i) {
    const float back = vt::BF16ToF32(vt::F32ToBF16(f[i]));
    if (back == f[i]) continue;
    // NaN never compares equal to itself, so it would fail the test above for
    // the wrong reason. It is also not a weight, so say which it is.
    VT_CHECK(false,
             "glm-dsa gguf: " + name + " element " + std::to_string(i) +
                 " is " + std::to_string(f[i]) +
                 ", which does not round-trip through bf16 (bf16 gives " +
                 std::to_string(back) +
                 "). This port holds it at the MODEL dtype because " +
                 std::string(upstream_reason) +
                 ", and the file's F32 is normally llama.cpp's lossless upcast "
                 "of exactly that bf16 tensor. A genuinely f32 tensor here is a "
                 "different operator, and rounding it would move a DISCRETE "
                 "selection that no tolerance bounds, so it is refused rather "
                 "than served");
  }
  return Bf16From(f, shape, nk);
}

// ─── the post-load absorption (W9, #2214) ────────────────────────────────────
// Upstream's `MLAAttention.process_weights_after_loading`
// (`mla_attention.py:875-962`), placed where `deepseek_v2_weights.cpp:138-153`
// already places it for the safetensors arm.
//
// THE FILE'S TWO HALVES ARE NOT THE SEAM'S TWO HALVES, and the difference is a
// transpose rather than a rename. llama.cpp's `ggml_mul_mat` contracts over
// `ne[0]`, so `attn_k_b` reads `[heads, kv_lora, qk_nope]` and `attn_v_b` reads
// `[heads, v_head, kv_lora]`. `vt::BatchedMatmul` — which is `torch.bmm`, the
// primitive upstream's absorption is expressed in — wants `w_uk_t`
// `[heads, qk_nope, kv_lora]` and `w_uv` `[heads, kv_lora, v_head]`, shares
// "f32 or bf16" across a and b, and requires the innermost dimension to be
// unit-stride. So neither a transposed view nor the Q8_0 bytes are admissible,
// and bf16 with the transpose applied is the only form that is. That is exactly
// upstream's answer and upstream's stated reason (`:876-878`).
//
// The CHECKPOINT-layout `kv_b_proj` the PREFILL arm needs is rebuilt here too,
// because the GGUF does not ship it: per head, rows `[0, qk_nope)` are
// `attn_k_b` transposed and rows `[qk_nope, qk_nope + v_head)` are `attn_v_b`
// verbatim — that half is ALREADY in the checkpoint orientation. `w_uk_t` and
// `w_uv` then come from the SHARED `mla::AbsorbKvBProjBf16` applied to it, so
// there is one absorber in this tree and not two.
void AbsorbMla(const GgufFile& g, const GlmMoeDsaParams& p, int64_t il,
               GlmMoeDsaMlaWeights& w) {
  const int64_t heads = p.num_attention_heads;
  const int64_t kv_lora = p.kv_lora_rank;
  const int64_t qk_nope = p.qk_nope_head_dim;
  const int64_t v_head = p.v_head_dim;
  const int64_t row = qk_nope + v_head;

  // f32 throughout the rewrite, so a value is rounded ONCE, on the store.
  const std::vector<float> kb =
      DequantAll(g, Blk(il, "attn_k_b.weight"), {heads, kv_lora, qk_nope});
  const std::vector<float> vb =
      DequantAll(g, Blk(il, "attn_v_b.weight"), {heads, v_head, kv_lora});

  std::vector<float> kvb(static_cast<size_t>(heads * row * kv_lora), 0.0f);
  for (int64_t h = 0; h < heads; ++h) {
    for (int64_t i = 0; i < qk_nope; ++i) {
      for (int64_t j = 0; j < kv_lora; ++j) {
        // TRANSPOSED: the file has `[h][j][i]`, the checkpoint layout is
        // `[h][i][j]`.
        kvb[static_cast<size_t>((h * row + i) * kv_lora + j)] =
            kb[static_cast<size_t>((h * kv_lora + j) * qk_nope + i)];
      }
    }
    for (int64_t i = 0; i < v_head; ++i) {
      for (int64_t j = 0; j < kv_lora; ++j) {
        // VERBATIM: `attn_v_b` is `[h][i][j]` already.
        kvb[static_cast<size_t>((h * row + qk_nope + i) * kv_lora + j)] =
            vb[static_cast<size_t>((h * v_head + i) * kv_lora + j)];
      }
    }
  }
  w.kv_b_proj = Bf16From(kvb, {heads * row, kv_lora}, /*nk=*/true);

  mla::MlaBlockDims dims = GlmMoeDsaMlaBlockDims(p, il);
  const mla::AbsorbedKvBProj abs = mla::AbsorbKvBProjBf16(
      reinterpret_cast<const uint16_t*>(w.kv_b_proj.bytes.data()), dims);
  VT_CHECK(static_cast<int64_t>(abs.w_uk_t.size()) == heads * qk_nope * kv_lora &&
               static_cast<int64_t>(abs.w_uv.size()) == heads * kv_lora * v_head,
           "glm-dsa gguf: the shared MLA absorber returned unexpected sizes for "
           "block " + std::to_string(il));
  w.w_uk_t = MakeTensor(vt::DType::kBF16, {heads, qk_nope, kv_lora},
                        /*nk=*/false, sizeof(uint16_t));
  std::memcpy(w.w_uk_t.bytes.data(), abs.w_uk_t.data(),
              abs.w_uk_t.size() * sizeof(uint16_t));
  w.w_uv = MakeTensor(vt::DType::kBF16, {heads, kv_lora, v_head},
                      /*nk=*/false, sizeof(uint16_t));
  std::memcpy(w.w_uv.bytes.data(), abs.w_uv.data(),
              abs.w_uv.size() * sizeof(uint16_t));
}

// The file to BORROW kept blocks from, or null to copy them. Borrowing is what
// keeps 187 GiB of towers out of host RAM, and it is also what gives the
// streaming lane an `mmap_fd` / `mmap_file_offset` to `pread` from instead of
// faulting 4 KiB at a time through the mapping.
const GgufFile* MmapSrc(const GgufFile& g, const GgufLoadPolicy& pol) {
  return pol.mmap_residency ? &g : nullptr;
}

// One [n] norm gamma as bf16. There is no `+1` to invert: the DeepSeek/GLM
// conversion chain folds no norm, and this function takes no `unshift`
// argument so the wrong convention cannot be inherited by defaulting.
OwnedTensor LoadNormBf16(const GgufFile& g, const std::string& name, int64_t n) {
  return Bf16From(DequantAll(g, name, {n}), {n}, /*nk=*/false);
}

// One [n] vector kept at f32. Every call site is an annotated exception and
// names its reason.
OwnedTensor LoadVecF32(const GgufFile& g, const std::string& name, int64_t n) {
  return F32From(DequantAll(g, name, {n}), {n}, /*nk=*/false);
}

// One [N, K] matmul operand taken VERBATIM: raw ggml blocks when the policy
// routes it there, native F16 when it routes it there, bf16 in the file's own
// [N, K] order otherwise.
OwnedTensor LoadMatmul(const GgufFile& g, const GgufLoadPolicy& pol,
                       const std::string& name, int64_t n, int64_t k) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {n, k});
  const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
  if (r == GgufResidency::kKeepQuant)
    return OwnGgufQuantBlocks(t, n, k, /*row_offset=*/0, MmapSrc(g, pol),
                              kGlmMoeDsaQuantRepack);
  if (r == GgufResidency::kKeepF16)
    return OwnGgufF16(t, n, k, /*row_offset=*/0, MmapSrc(g, pol), /*nk=*/true,
                      /*elem_kn_repack=*/false);
  return ExpandBf16(g, name, {n, k}, /*nk=*/true);
}

// THE VOCABULARY TABLE, WHICH IS A GATHER AND NOT A GEMM, AND ASKING THE POLICY
// THE WRONG QUESTION ABOUT IT COSTS A REFUSAL AT THE FIRST FORWARD.
//
// `token_embd.weight` was routed through `LoadMatmul` until this line existed,
// so the policy was asked about `kMatmulWeight` — and the two roles differ on
// exactly the axis that matters here. A GEMM weight's device gate is
// `DeviceKeepQuantSupported`, which answers TRUE on CUDA because the CUDA
// backend falls back to the CPU kernel for anything it lacks. A GATHER's gate is
// `DeviceQuantGatherSupported`, which answers true ONLY on the CPU, because
// `EmbeddingKernelCuda` (`cuda_ops.cu`) accepts f32 and bf16 tables and nothing
// else and there is no fallback tier under it. So on a device queue the table
// stayed Q4_K and the first forward threw
// `cuda embedding: unsupported table dtype (f32/bf16 only)` with all 201.83 GiB
// of the model already resident — measured on `thor:gpu0`, 2026-08-31, spec O31.
//
// Asking `kEmbeddingTable` is the whole repair: the shared policy already knows
// the answer and already expands to bf16 on a device queue while keeping the
// blocks on a CPU one. The cost on this checkpoint is exact and is stated rather
// than discovered — `[154880, 6144]` Q4_K becomes 1.772 GiB of bf16 — and it is
// the price of a gather this tree cannot yet do on the device, which
// `gguf_keep_quant.cpp`'s own comment records as owed.
//
// `also_matmul` is the TIED case, where the same bytes are the lm_head GEMM as
// well. The gather's encoding rule needs only a row decoder while a GEMM needs a
// `vec_dot`, so a tied table may only keep its blocks when BOTH roles say so;
// the intersection is taken here rather than left for `vt::MatmulBT` to discover
// at the first forward. The published `unsloth/GLM-5.3-GGUF` arm ships
// `token_embd.weight` and `output.weight` as two separate Q4_K tensors, so it
// takes `also_matmul = false`.
OwnedTensor LoadEmbeddingTable(const GgufFile& g, const GgufLoadPolicy& pol,
                               const std::string& name, int64_t v, int64_t h,
                               bool also_matmul) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {v, h});
  GgufResidency r = pol.Route(t, GgufTensorRole::kEmbeddingTable);
  if (also_matmul && r != GgufResidency::kExpandBf16 &&
      pol.Route(t, GgufTensorRole::kMatmulWeight) != r) {
    r = GgufResidency::kExpandBf16;
  }
  if (r == GgufResidency::kKeepQuant)
    return OwnGgufQuantBlocks(t, v, h, /*row_offset=*/0, MmapSrc(g, pol),
                              kGlmMoeDsaQuantRepack);
  if (r == GgufResidency::kKeepF16)
    return OwnGgufF16(t, v, h, /*row_offset=*/0, MmapSrc(g, pol), /*nk=*/true,
                      /*elem_kn_repack=*/false);
  return ExpandBf16(g, name, {v, h}, /*nk=*/true);
}

// The ROUTER GEMM, held at f32. This is the annotated dtype exception the
// polarity rule requires: `_get_moe_router_dtype` (`deepseek_v2.py:123-133`)
// forces `torch.float32` on `model_type == "glm_moe_dsa"` at `:127`, and the
// published file already stores every `ffn_gate_inp.weight` as F32, so this is
// a mirror rather than a widening. `deepseek_v2.cpp:363` sizes the output
// buffer from the same fact.
OwnedTensor LoadMatmulF32(const GgufFile& g, const std::string& name, int64_t n,
                          int64_t k) {
  return F32From(DequantAll(g, name, {n, k}), {n, k}, /*nk=*/true);
}

// A 3-D [E, N, K] STACKED EXPERT tower — the streamed class. Each expert slab
// is a whole number of rows and therefore a whole number of quant blocks, so
// the keep-quant arm is a byte range and no block is ever cut.
//
// The shape is recorded FLAT as [E*N, K] with `rank == 2`, which is what
// `LoadExpertsStackedKq` (`qwen3_5_gguf_weights.cpp:1267-1282`) does and what
// the streaming lane's only gated client therefore consumes. The bytes are
// identical either way; only the recorded shape differs, and every consumer
// slices by expert with an explicit (N, K, row_offset).
OwnedTensor LoadStackedExperts(const GgufFile& g, const GgufLoadPolicy& pol,
                               const std::string& name, int64_t e, int64_t n,
                               int64_t k) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {e, n, k});
  const GgufResidency r = pol.Route(t, GgufTensorRole::kStackedExpertWeight);
  // A tower that is not kept is a tower that left the streaming lane, and
  // `gguf_device_fit.cpp`'s admission rule is all-or-nothing across the model
  // (one bad tower disqualifies all 228). Refusing HERE names the tensor;
  // letting it through would put 24.000 GiB of bf16 where 6.375 GiB of blocks
  // belong and drop the whole arm out of streaming with no message at all.
  VT_CHECK(r == GgufResidency::kKeepQuant || r == GgufResidency::kKeepF16,
           "glm-dsa gguf: " + name +
               " routed to an EXPAND residency. A routed-expert tower that "
               "expands leaves the streaming lane entirely — "
               "GgufExpertTowersReachSlotLane is all-or-nothing across a "
               "model's `*_exps.weight` set, so one expanded tower disqualifies "
               "all of them — and this model's towers are 187.312 GiB "
               "compressed. Check VT_GGUF_KEEP_QUANT and that this encoding has "
               "a keep-quant vec_dot (vt::cpu::HasQuantDotKernel)");
  // `prefault = false`, and this is the line that decides whether this model
  // STREAMS or only says it does (#2214, spec §3.3, O30). Every other borrowed
  // weight is faulted in at load so a page trap does not land in the timed
  // prefill; a routed-expert tower is the one class where that trade does not
  // exist, because `expert_stream::ExpertSlice` preads each slice into a slot
  // and NEVER reads the tower through the mapping. Prefaulting these 228 towers
  // therefore reads 187.312 GiB off the filesystem at load to populate pages
  // nothing looks at — measured on the real artifact before this line existed:
  // RSS climbed linearly past 48 GiB against a 18.99 GiB resident class, at the
  // filesystem's read rate, with no plateau. It is also the exact shape §3.3
  // refuses to publish, a page-cache number under a streaming label.
  if (r == GgufResidency::kKeepQuant) {
    return OwnGgufQuantBlocks(t, e * n, k, /*row_offset=*/0, MmapSrc(g, pol),
                              kGlmMoeDsaQuantRepack, /*cuda_align=*/false,
                              /*prefault=*/false);
  }
  return OwnGgufF16(t, e * n, k, /*row_offset=*/0, MmapSrc(g, pol), /*nk=*/true,
                    /*elem_kn_repack=*/false, /*prefault=*/false);
}

// A 3-D [H, N, K] tensor that is NOT an expert bank and is never sliced by
// expert: the two absorbed MLA halves, whose leading axis is the attention
// HEAD. The role is `kMatmulWeight`, deliberately — asking the policy for
// `kStackedExpertWeight` here would put these in the class the expert-streaming
// seam reasons about and the lane would try to page an attention weight.
OwnedTensor LoadHeadStacked(const GgufFile& g, const GgufLoadPolicy& pol,
                            const std::string& name, int64_t h, int64_t n,
                            int64_t k) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {h, n, k});
  const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
  if (r == GgufResidency::kKeepQuant) {
    OwnedTensor o = OwnGgufQuantBlocks(t, h * n, k, /*row_offset=*/0,
                                       MmapSrc(g, pol), kGlmMoeDsaQuantRepack);
    o.rank = 3;
    o.shape[0] = h;
    o.shape[1] = n;
    o.shape[2] = k;
    return o;
  }
  if (r == GgufResidency::kKeepF16) {
    OwnedTensor o = OwnGgufF16(t, h * n, k, /*row_offset=*/0, MmapSrc(g, pol),
                               /*nk=*/true, /*elem_kn_repack=*/false);
    o.rank = 3;
    o.shape[0] = h;
    o.shape[1] = n;
    o.shape[2] = k;
    return o;
  }
  return ExpandBf16(g, name, {h, n, k}, /*nk=*/true);
}

// ─── the four per-layer loaders ──────────────────────────────────────────────

GlmMoeDsaIndexerWeights LoadIndexer(const GgufFile& g,
                                    const GgufLoadPolicy& pol,
                                    const GlmMoeDsaParams& p, int64_t il) {
  const int64_t h = p.hidden_size;
  const int64_t nh = p.index_n_heads;
  const int64_t hd = p.index_head_dim;
  GlmMoeDsaIndexerWeights w;
  w.wq_b = LoadMatmul(g, pol, Blk(il, "indexer.attn_q_b.weight"), nh * hd,
                      p.q_lora_rank);
  w.wk = LoadMatmul(g, pol, Blk(il, "indexer.attn_k.weight"), hd, h);
  w.k_norm_weight = LoadNormBf16(g, Blk(il, "indexer.k_norm.weight"), hd);
  // Required, not optional: the bias is what makes this a LayerNorm rather than
  // an RMSNorm, and a file without it describes a different operator.
  w.k_norm_bias = LoadNormBf16(g, Blk(il, "indexer.k_norm.bias"), hd);
  // `weights_proj` is `nn.Linear(hidden_size, n_heads)` — one row per INDEXER
  // head (32), never per MLA head (64) and never per channel. The published
  // file stores it F32, and it is the one operand of the indexer whose scale
  // decides the selection outright, so it stays F32 rather than being rounded
  // into the model dtype.
  // W9 (#2214) NARROWED this from the file's F32 to bf16, and the narrowing is
  // upstream rather than a convenience. `wk_weights_proj` is a
  // `MergedColumnParallelLinear` with no `params_dtype`
  // (`deepseek_v2.py:700-707`), so upstream computes it at the MODEL dtype; the
  // shared MLA block does the same (`mla_attention.cpp`, the indexer's
  // `vt::MatmulBT(iw, hidden, indexer_weights_proj)` writes `dt`), and
  // `vt::MatmulBT` needs its two float operands to agree. W2's note that this
  // tensor "decides the selection outright" is why the round-trip is PROVED
  // here instead of the value being rounded on trust.
  w.weights_proj =
      Bf16FromExactF32(g, Blk(il, "indexer.proj.weight"), {nh, h}, /*nk=*/true,
                       "upstream's `wk_weights_proj` carries no params_dtype "
                       "(deepseek_v2.py:700-707) and is therefore the model "
                       "dtype, which the shared MLA block's indexer GEMM also "
                       "requires");
  return w;
}

GlmMoeDsaMlaWeights LoadMla(const GgufFile& g, const GgufLoadPolicy& pol,
                            const GlmMoeDsaParams& p, int64_t il,
                            bool with_indexer) {
  const int64_t h = p.hidden_size;
  const int64_t heads = p.num_attention_heads;
  const int64_t q_lora = p.q_lora_rank;
  const int64_t kv_lora = p.kv_lora_rank;
  const int64_t qk_nope = p.qk_nope_head_dim;
  const int64_t qk_rope = p.qk_rope_head_dim;
  const int64_t qk_head = qk_nope + qk_rope;
  const int64_t v_head = p.v_head_dim;

  GlmMoeDsaMlaWeights w;
  w.q_a_proj = LoadMatmul(g, pol, Blk(il, "attn_q_a.weight"), q_lora, h);
  w.q_a_layernorm = LoadNormBf16(g, Blk(il, "attn_q_a_norm.weight"), q_lora);
  w.q_b_proj =
      LoadMatmul(g, pol, Blk(il, "attn_q_b.weight"), heads * qk_head, q_lora);
  w.kv_a_proj_with_mqa =
      LoadMatmul(g, pol, Blk(il, "attn_kv_a_mqa.weight"), kv_lora + qk_rope, h);
  w.kv_a_layernorm = LoadNormBf16(g, Blk(il, "attn_kv_a_norm.weight"), kv_lora);
  // ALREADY ABSORBED AND ALREADY TRANSPOSED by the converter, so no
  // `AbsorbKvBProjBf16` runs on this arm. `k_b`'s trailing axis is
  // `qk_nope_head_dim` while `v_b`'s is `kv_lora_rank`; the two are NOT the
  // same shape even though both trailing dims happen to be 256 elsewhere on
  // this checkpoint — [64, 512, 192] against [64, 256, 512].
  w.k_b_proj = LoadHeadStacked(g, pol, Blk(il, "attn_k_b.weight"), heads,
                               kv_lora, qk_nope);
  w.v_b_proj = LoadHeadStacked(g, pol, Blk(il, "attn_v_b.weight"), heads,
                               v_head, kv_lora);
  w.o_proj =
      LoadMatmul(g, pol, Blk(il, "attn_output.weight"), h, heads * v_head);
  // A `kShared` layer runs NO indexer (`deepseek_v2.py:1134-1135`) and attends
  // through the preceding full layer's selection. Its five `indexer.*` tensors
  // exist in this file only because the conversion broadcast them (spec D3);
  // they are counted as dropped by the caller and never read.
  if (with_indexer) w.indexer = LoadIndexer(g, pol, p, il);
  // Upstream's `process_weights_after_loading`, run here for the same reason
  // `deepseek_v2_weights.cpp:138-153` runs it in ITS loader: the absorbed forms
  // are what the decode arm consumes, and they are a function of the weights
  // rather than of the step. See `AbsorbMla`.
  AbsorbMla(g, p, il, w);
  return w;
}

GlmMoeDsaMlpWeights LoadMlp(const GgufFile& g, const GgufLoadPolicy& pol,
                            const GlmMoeDsaParams& p, int64_t il,
                            const char* gate, const char* up, const char* down,
                            int64_t inter) {
  const int64_t h = p.hidden_size;
  GlmMoeDsaMlpWeights w;
  w.gate_proj = LoadMatmul(g, pol, Blk(il, gate), inter, h);
  w.up_proj = LoadMatmul(g, pol, Blk(il, up), inter, h);
  w.down_proj = LoadMatmul(g, pol, Blk(il, down), h, inter);
  return w;
}

GlmMoeDsaMoeWeights LoadMoe(const GgufFile& g, const GgufLoadPolicy& pol,
                            const GlmMoeDsaParams& p, int64_t il) {
  const int64_t h = p.hidden_size;
  const int64_t e = p.n_routed_experts;
  const int64_t mi = p.moe_intermediate_size;
  GlmMoeDsaMoeWeights w;
  w.router = LoadMatmulF32(g, Blk(il, "ffn_gate_inp.weight"), e, h);
  // `e_score_correction_bias`, the noaux_tc router's per-expert bias. F32 in
  // the file and f32 in memory: it is ADDED to the sigmoid scores before the
  // top-k, so rounding it into bf16 would move the selection, which is a
  // discrete outcome no tolerance bounds.
  if (p.has_e_score_correction_bias) {
    w.e_score_correction_bias = LoadVecF32(g, Blk(il, "exp_probs_b.bias"), e);
  }
  w.gate_exps =
      LoadStackedExperts(g, pol, Blk(il, "ffn_gate_exps.weight"), e, mi, h);
  w.up_exps =
      LoadStackedExperts(g, pol, Blk(il, "ffn_up_exps.weight"), e, mi, h);
  w.down_exps =
      LoadStackedExperts(g, pol, Blk(il, "ffn_down_exps.weight"), e, h, mi);
  // The shared expert is `moe_intermediate_size * n_shared_experts` (2048 * 1),
  // NOT `intermediate_size` (12288, which blocks 0-2 use).
  w.shared = LoadMlp(g, pol, p, il, "ffn_gate_shexp.weight",
                     "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                     mi * p.n_shared_experts);
  return w;
}

}  // namespace

// ─── the claim set ───────────────────────────────────────────────────────────

std::vector<std::string> EnumerateGlmMoeDsaGgufTensors(
    const GlmMoeDsaParams& p) {
  std::vector<std::string> out;
  out.emplace_back("token_embd.weight");
  out.emplace_back("output_norm.weight");
  // `output.weight` is OPTIONAL: a tied checkpoint omits it. It is claimed
  // here because the published artifact ships it, and the loader reads the tie
  // off the file rather than out of the config.
  out.emplace_back("output.weight");
  for (int64_t il = 0; il < p.num_hidden_layers; ++il) {
    out.push_back(Blk(il, "attn_norm.weight"));
    out.push_back(Blk(il, "ffn_norm.weight"));
    out.push_back(Blk(il, "attn_q_a.weight"));
    out.push_back(Blk(il, "attn_q_a_norm.weight"));
    out.push_back(Blk(il, "attn_q_b.weight"));
    out.push_back(Blk(il, "attn_kv_a_mqa.weight"));
    out.push_back(Blk(il, "attn_kv_a_norm.weight"));
    out.push_back(Blk(il, "attn_k_b.weight"));
    out.push_back(Blk(il, "attn_v_b.weight"));
    out.push_back(Blk(il, "attn_output.weight"));
    // Claimed on EVERY block, because the published file ships them on every
    // block; the `kShared` ones are dropped at load, not left unaccounted.
    out.push_back(Blk(il, "indexer.attn_q_b.weight"));
    out.push_back(Blk(il, "indexer.attn_k.weight"));
    out.push_back(Blk(il, "indexer.k_norm.weight"));
    out.push_back(Blk(il, "indexer.k_norm.bias"));
    out.push_back(Blk(il, "indexer.proj.weight"));
    if (p.is_moe_layer(il)) {
      out.push_back(Blk(il, "ffn_gate_inp.weight"));
      if (p.has_e_score_correction_bias)
        out.push_back(Blk(il, "exp_probs_b.bias"));
      out.push_back(Blk(il, "ffn_gate_exps.weight"));
      out.push_back(Blk(il, "ffn_up_exps.weight"));
      out.push_back(Blk(il, "ffn_down_exps.weight"));
      if (p.n_shared_experts > 0) {
        out.push_back(Blk(il, "ffn_gate_shexp.weight"));
        out.push_back(Blk(il, "ffn_up_shexp.weight"));
        out.push_back(Blk(il, "ffn_down_shexp.weight"));
      }
    } else {
      out.push_back(Blk(il, "ffn_gate.weight"));
      out.push_back(Blk(il, "ffn_up.weight"));
      out.push_back(Blk(il, "ffn_down.weight"));
    }
  }
  return out;
}

// ─── the load ────────────────────────────────────────────────────────────────

GlmMoeDsaWeights LoadGlmMoeDsaFromGguf(const GgufFile& gguf,
                                       const HfConfig& config,
                                       const GgufLoadPolicy* policy) {
  const GgufLoadPolicy pol =
      policy != nullptr ? *policy : GgufLoadPolicy::FromEnv();

  GlmMoeDsaWeights w;
  // ONE validator for both sources: a `config.json` and a `glm-dsa` GGUF header
  // descend through the same `ParseGlmMoeDsaParams`, which is where the indexer
  // schedule is resolved and where a file that states none is refused.
  w.params = ParseGlmMoeDsaParams(config);
  const GlmMoeDsaParams& p = w.params;
  VT_CHECK(p.num_hidden_layers > 0,
           "glm-dsa: num_hidden_layers must be positive");

  w.file_tensors = static_cast<int64_t>(gguf.Tensors().size());

  // ── the MTP tail, read and dropped (spec O5) ──
  // `num_nextn_predict_layers` blocks sit past the backbone. There is no MTP
  // drafter in this tree at all, so they are COUNTED and DROPPED rather than
  // silently skipped: a reader of the accounting can see that 26 tensors went
  // somewhere deliberate. This is what `allow_mtp_tail` means on this arm.
  const int64_t backbone = p.num_hidden_layers;
  for (const GgufTensorInfo& t : gguf.Tensors()) {
    if (t.name.rfind("blk.", 0) != 0) continue;
    const size_t dot = t.name.find('.', 4);
    if (dot == std::string::npos) continue;
    const std::string idx = t.name.substr(4, dot - 4);
    if (idx.empty() ||
        idx.find_first_not_of("0123456789") != std::string::npos)
      continue;
    if (std::stoll(idx) >= backbone) ++w.mtp_block_tensors_dropped;
  }

  // ── model-level ──
  const int64_t h = p.hidden_size;
  const int64_t v = p.vocab_size;
  // The tie is read OFF THE FILE. A converter is free to materialize either
  // shape regardless of what the source config's `tie_word_embeddings` says,
  // and the published artifact ships both `token_embd.weight` and
  // `output.weight` as separate Q4_K tensors. It is resolved BEFORE the table is
  // loaded, because a tied table is also a GEMM weight and that changes which
  // residencies it may take.
  const bool tied = !HasTensor(gguf, "output.weight");
  w.embed_tokens = LoadEmbeddingTable(gguf, pol, "token_embd.weight", v, h,
                                      /*also_matmul=*/tied);
  w.final_norm = LoadNormBf16(gguf, "output_norm.weight", h);
  if (!tied) w.lm_head = LoadMatmul(gguf, pol, "output.weight", v, h);

  // ── ONE shared rope [cos|sin] cache for every layer ──
  // Upstream shares a single `get_rope` module instance across layers. Built in
  // the forward dtype (bf16) so `vt::RopeFromCache` reads it directly.
  // `rope_type` is `default` on this checkpoint and `ParseGlmMoeDsaParams`
  // refuses every other value, so `yarn` is false by a CHECKED fact.
  {
    const int64_t rows = p.max_position_embeddings;
    const int64_t rot = p.qk_rope_head_dim;
    VT_CHECK(rows > 0 && rot > 0,
             "glm-dsa: the rope cache needs a positive "
             "max_position_embeddings and qk_rope_head_dim");
    mla::DeepseekYarnRopeParams rope{};
    rope.base = p.rope_theta;
    rope.rotary_dim = rot;
    rope.yarn = false;
    const std::vector<float> cache =
        mla::BuildDeepseekRopeCosSinCache(rope, rows);
    w.rope_cos_sin_cache = MakeTensor(vt::DType::kBF16, {rows, rot},
                                      /*nk=*/false, sizeof(uint16_t));
    auto* dst = reinterpret_cast<uint16_t*>(w.rope_cos_sin_cache.bytes.data());
    for (size_t i = 0; i < cache.size(); ++i) dst[i] = vt::F32ToBF16(cache[i]);
  }

  // ── the backbone ──
  w.layers.resize(static_cast<size_t>(backbone));
  for (int64_t il = 0; il < backbone; ++il) {
    GlmMoeDsaLayerWeights& lw = w.layers[static_cast<size_t>(il)];
    lw.input_layernorm = LoadNormBf16(gguf, Blk(il, "attn_norm.weight"), h);
    lw.post_attention_layernorm =
        LoadNormBf16(gguf, Blk(il, "ffn_norm.weight"), h);

    const bool full =
        p.indexer_types[static_cast<size_t>(il)] == GlmMoeDsaIndexerKind::kFull;
    lw.attn = LoadMla(gguf, pol, p, il, /*with_indexer=*/full);
    if (!full) {
      // Spec D3, made countable. The five broadcast tensors on this block are
      // present in the file and are not read; the count is what a gate asserts
      // against `57 shared layers x 5`.
      w.broadcast_indexer_tensors_dropped += 5;
    }

    lw.is_moe = p.mlp_layer_types[static_cast<size_t>(il)] ==
                GlmMoeDsaMlpKind::kSparse;
    // The two statements of the same fact must agree. `ParseGlmMoeDsaParams`
    // already refuses a config whose `mlp_layer_types` disagrees with
    // `first_k_dense_replace`; this re-asserts it at the point the weights are
    // chosen, because reading the wrong branch here loads a dense MLP's three
    // tensors out of a block that ships expert towers and the shape check would
    // be the only thing that noticed.
    VT_CHECK(lw.is_moe == p.is_moe_layer(il),
             "glm-dsa gguf: layer " + std::to_string(il) +
                 " is `" +
                 (lw.is_moe ? std::string("sparse") : std::string("dense")) +
                 "` by mlp_layer_types but `" +
                 (p.is_moe_layer(il) ? std::string("sparse")
                                     : std::string("dense")) +
                 "` by first_k_dense_replace/moe_layer_freq");

    if (lw.is_moe) {
      lw.moe = LoadMoe(gguf, pol, p, il);
    } else {
      lw.dense = LoadMlp(gguf, pol, p, il, "ffn_gate.weight", "ffn_up.weight",
                         "ffn_down.weight", p.intermediate_size);
    }
  }

  // ── accounting ──
  // Every tensor in the file is enumerated and accounted, or the load refuses.
  // This is the structural half of G2 (spec §3.6) evaluated on the real file
  // rather than on its header: a name the port does not claim is a tensor whose
  // weights are silently not in the model, and no token gate on this row could
  // ever see that (spec O1 — there is no token gate on this row).
  {
    const std::vector<std::string> claimed = EnumerateGlmMoeDsaGgufTensors(p);
    const std::set<std::string> claim_set(claimed.begin(), claimed.end());
    std::vector<std::string> unaccounted;
    for (const GgufTensorInfo& t : gguf.Tensors()) {
      if (claim_set.count(t.name) != 0) {
        ++w.accounted_tensors;
        continue;
      }
      if (t.name.rfind("blk.", 0) == 0) {
        const size_t dot = t.name.find('.', 4);
        if (dot != std::string::npos) {
          const std::string idx = t.name.substr(4, dot - 4);
          if (!idx.empty() &&
              idx.find_first_not_of("0123456789") == std::string::npos &&
              std::stoll(idx) >= backbone) {
            continue;  // already counted as an MTP-tail drop
          }
        }
      }
      if (unaccounted.size() < 8) unaccounted.push_back(t.name);
    }
    const int64_t total = w.accounted_tensors + w.mtp_block_tensors_dropped;
    if (total != w.file_tensors) {
      std::string sample;
      for (const std::string& n : unaccounted) sample += "\n    " + n;
      VT_CHECK(false,
               "glm-dsa gguf: " + std::to_string(w.file_tensors) +
                   " tensors in the file but " +
                   std::to_string(w.accounted_tensors) +
                   " accounted plus " +
                   std::to_string(w.mtp_block_tensors_dropped) +
                   " dropped from the multi-token-prediction tail = " +
                   std::to_string(total) +
                   ". A tensor this port does not claim is weight that is "
                   "silently absent from the model. First unaccounted names:" +
                   sample);
    }
  }

  // Every layer went through `AbsorbMla`, so the forward's precondition holds.
  // Set LAST, after the accounting refuses a file this port cannot account for,
  // so a partially built model can never read as prepared.
  w.absorbed = true;
  return w;
}

}  // namespace vllm
