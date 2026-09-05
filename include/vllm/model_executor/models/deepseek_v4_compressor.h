// DeepSeek-V4-Flash W4 — the DSA COMPRESSOR forward + the fp8_ds_mla KV-cache
// state read/write layout, as portable host (CPU) reference implementations.
// This is the second half of the DSA sparse-attention stack (W3 landed the
// Lightning-Indexer SELECTION + the 512-wide MLA output seams in
// deepseek_v4_dsa.{h,cpp}); W4 owns the two things that turn the selected /
// windowed KV into the compressed latent the MLA reads, and how that latent is
// cached across steps:
//
//   (A) the DSA COMPRESSOR forward — the softmax-weighted window POOL that
//       compresses `(1+overlap)*compress_ratio` KV-state rows into one
//       compressed latent, plus the fused save-time APE add and the trailing
//       RMSNorm (DeepseekCompressor, compressor.py; the compute kernel
//       common/ops/fused_compress_quant_cache.py:198-218 + save_partial_states.py
//       :92-101), and
//   (B) the fp8_ds_mla KV-CACHE STATE layout — how the compressed latent is
//       written to / read from the paged cache: the 512-wide head split into a
//       448-wide NoPE part quantized to FP8 with per-64 UE8M0 (power-of-two)
//       block scales and a 64-wide RoPE part stored bf16, at a 576-byte token
//       stride with a padded 8-byte scale region (fused_compress_quant_cache.py
//       :220-297 store side; the read/dequant side confirmed against SGLang
//       dsv4/dequant_k_cache.py:12-14,:122-136).
//
// WHY host/CPU reference (honest scope): the full DeepSeek-V4-Flash forward is a
// multi-Spark campaign — the checkpoint is 156.7 GiB (does not fit one GB10, see
// deepseek_v4.h) and the forward also needs MHC (W5) + the sqrtsoftplus/hash MoE
// (W6), none ported yet. So W4 lands + UNIT-GATES these primitives against
// hand-derived small cases with literal expected numbers verified from the vLLM
// source AND from-first-principles double-precision references on randomized
// shapes (tests/vllm/models/test_deepseek_v4_compressor.cpp), rather than a
// full-model dumped-oracle rel-L2 (the fixed-config 167B arch cannot be built at
// a tiny shape). The eventual GPU forward (W7) ports the SAME math into a CUDA
// kernel; this file pins the numerics portably so the kernel port has an oracle.
//
// SACRED-inert: additive TU only. It does NOT touch the DeepSeek-V2 MLA path
// (mla_attention.{h,cpp}, cuda_mla_attn.cu) — V4's 512-wide geometry + fp8_ds_mla
// KV stay a V4-specific path and the shared-mla extraction is a NAMED W7 seam.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922) ───────
//   OURS                          <-  UPSTREAM (vllm/, @ 0.26.0.dev0)  [+ SGLang v0.5.15]
//   CompressorSaveScoreApe        <-  common/ops/save_partial_states.py:92-101
//                                     (score_state = score + ape[position %
//                                      compress_ratio]); SGLang dsv4/compressor.py
//                                     forward_compress ape-add
//   CompressorPoolNorm            <-  common/ops/fused_compress_quant_cache.py
//                                     :198-218 (softmax over the window (dim=0,
//                                      per head-dim column) → weighted sum of kv →
//                                      RMSNorm); SGLang dsv4/fused_compress_triton.py
//                                     _fused_ape_pool_norm_rope_kernel:57-95
//   MakeFp8DsMlaLayout            <-  compressor.py:307-309 (_quant_block=64,
//                                     _token_stride=nope+rope*2=576, _scale_dim=
//                                     nope//64 + 1 = 8); SGLang dsv4/dequant_k_cache.py
//                                     :12-18 (DIM_NOPE=448, TILE_SIZE=64)
//   Fp8DsMlaEncodeToken           <-  fused_compress_quant_cache.py:238-297
//                                     (bf16 round → per-64-block absmax(≥1e-4) →
//                                      exponent=ceil(log2(absmax/448)) → inv_scale=
//                                      2^-exp → clamp[-448,448] → e4m3; scale byte
//                                      = exp+127; rope → bf16)
//   Fp8DsMlaDecodeToken           <-  SGLang dsv4/dequant_k_cache.py:122-136
//                                     (nope = e4m3->f32 * 2^(scale_byte-127);
//                                      rope = bf16->f32) — the paged-KV READ side
//   MakeFp8DsMlaPageLayout        <-  cache_utils.py:59-63 (the region-split block
//                                     comment) + kv_cache_interface.py:401-403
//                                     (storage_block_size*584) + :345-351
//                                     (_apply_alignment_padding, round_up 576)
//   Fp8DsMlaStoreToken            <-  cache_utils.py:36-159
//                                     (quantize_and_insert_k_kernel: :77-78 the
//                                      slot<0 skip, :92 data offset, :96-98 the
//                                      scale-region offset, :146 the 7 scale
//                                      bytes, :148-149 the zero pad byte,
//                                      :155-159 the bf16 rope store)
//   Fp8DsMlaLoadToken             <-  cache_utils.py:228-341
//                                     (_dequantize_and_gather_k_kernel: :281-292
//                                      the same addressing, :306 the fp8 bytes,
//                                      :319 the scale byte, :333-339 the rope);
//                                      it reads n_quant_blocks=7 (`:385`), one
//                                      FEWER than the store writes
#pragma once

#include <cstdint>
#include <vector>

namespace vllm::deepseek_v4 {

// ── (A) DSA Compressor forward ───────────────────────────────────────────────

// Save-time APE fusion (save_partial_states.py:92-101). The compressor stores,
// per token, its score row PLUS an absolute-position embedding row selected by
// `position % compress_ratio`:
//     score_state[t, d] = score[t, d] + ape[positions[t] % compress_ratio, d]
// (The kv half is stored verbatim; only the score half gets the APE add, which
// is why it is fused into the state write.) `score` is [num_tokens, width]
// row-major, `ape` is [compress_ratio, width] row-major; returns [num_tokens,
// width] row-major.
std::vector<float> CompressorSaveScoreApe(const std::vector<float>& score,
                                          const std::vector<float>& ape,
                                          const std::vector<int64_t>& positions,
                                          int64_t num_tokens, int64_t width,
                                          int64_t compress_ratio);

// The compressor POOL + RMSNorm (fused_compress_quant_cache.py:198-218). At a
// compress boundary the kernel gathers a window of `window` KV-state rows and:
//   - softmax over the window (dim=0), INDEPENDENTLY PER head-dim column d:
//         w[i, d] = softmax_i( score[i, d] )   (masked rows excluded)
//   - weighted sum:   compressed[d] = sum_i kv[i, d] * w[i, d]
//   - RMSNorm (fp32): var = mean_d(compressed[d]^2);
//                     normed[d] = compressed[d] * rsqrt(var + eps) * rms_weight[d]
// `valid[i] == 0` masks row i (out-of-range window position, pos < 0): its score
// is treated as -inf (contributes 0 to the softmax) and its kv as 0. The
// per-column softmax is the load-bearing nuance — each head-dim channel pools the
// window with its OWN weights, not one shared attention weight per row.
//
//   kv          : [window, head_dim] row-major
//   score       : [window, head_dim] row-major (the stored score_state, APE added)
//   valid       : [window]           1 = real row, 0 = masked
//   rms_weight  : [head_dim]
// Returns the compressed, normalized latent [head_dim].
std::vector<float> CompressorPoolNorm(const std::vector<float>& kv,
                                      const std::vector<float>& score,
                                      const std::vector<uint8_t>& valid,
                                      const std::vector<float>& rms_weight,
                                      float eps, int64_t window,
                                      int64_t head_dim);

// ── (B) fp8_ds_mla KV-cache state layout ─────────────────────────────────────

// The paged fp8_ds_mla token layout (compressor.py:307-309;
// SGLang dequant_k_cache.py:12-18). Per token the compressed head_dim splits
// into a NoPE part (FP8 e4m3, per-`quant_block` UE8M0 power-of-two scales) and a
// RoPE part (bf16). Byte geometry per token: `token_stride_bytes` =
// nope_head_dim*1 + rope_head_dim*2; the UE8M0 scale region carries
// `n_nope_blocks` real bytes padded to `scale_dim` (7 real + 1 pad for V4).
struct Fp8DsMlaLayout {
  int64_t nope_head_dim;      // 448 (DeepseekV4)
  int64_t rope_head_dim;      // 64
  int64_t quant_block;        // 64  (UE8M0 block size over the NoPE part)
  int64_t n_nope_blocks;      // nope_head_dim / quant_block (7)
  int64_t token_stride_bytes; // nope*1 + rope*2 (576)
  int64_t scale_dim;          // n_nope_blocks + 1 (8 = 7 real + 1 pad)
};
Fp8DsMlaLayout MakeFp8DsMlaLayout(int64_t nope_head_dim, int64_t rope_head_dim,
                                  int64_t quant_block);

// One token's fp8_ds_mla cache state (the bytes actually stored in the page).
struct Fp8DsMlaToken {
  std::vector<uint8_t> nope_fp8;      // [nope_head_dim]  e4m3 bytes
  std::vector<uint8_t> scale_ue8m0;   // [n_nope_blocks]  encoded exponent+127
  std::vector<uint16_t> rope_bf16;    // [rope_head_dim]  bf16 bit patterns
};

// Encode a compressed latent `head` ([nope_head_dim + rope_head_dim], the NoPE
// part first) into the fp8_ds_mla cache state (fused_compress_quant_cache.py
// :238-297 store side). Per NoPE block of `quant_block`:
//     q         = bf16_round(head[..])           (kernel casts fp32->bf16->fp32)
//     absmax    = max(1e-4, max_block |q|)
//     exponent  = ceil(log2(absmax / 448))
//     inv_scale = 2^(-exponent)
//     byte      = e4m3( clamp(q * inv_scale, -448, 448) )
//     scale     = clamp(exponent + 127, 0, 255)  (UE8M0)
// The RoPE part is stored bf16 verbatim (the forward RoPE that precedes it on GPU
// reuses our decoupled-RoPE machinery — a W3/W7 seam — so the layout gate feeds
// an already-rotated rope part and this reference only round-trips the bytes).
Fp8DsMlaToken Fp8DsMlaEncodeToken(const std::vector<float>& head,
                                  const Fp8DsMlaLayout& layout);

// The paged-KV READ (dequant): reverse of the encode
// (SGLang dequant_k_cache.py:122-136):
//     nope[d] = e4m3_to_f32(byte[d]) * 2^(scale_byte[block(d)] - 127)
//     rope[j] = bf16_to_f32(rope_bf16[j])
// Returns the reconstructed latent [nope_head_dim + rope_head_dim].
std::vector<float> Fp8DsMlaDecodeToken(const Fp8DsMlaToken& token,
                                       const Fp8DsMlaLayout& layout);

// ── (B2) The paged fp8_ds_mla BLOCK layout — `KV-DSV4-MULTICACHE` W8 slice 1 ──
//
// `Fp8DsMlaLayout` above is the geometry of ONE token. This is the geometry of a
// PAGE, and the two are not the same shape, because the block is REGION-SPLIT
// (`common/ops/cache_utils.py:59-63`, verbatim):
//
//     K Cache block layout (block_size=64 tokens):
//     - [0, 64*576):              Token data, each token has 448 fp8 + 128 bf16
//     - [64*576, 64*576 + 64*8):  Scales, each token has 8 uint8 scales
//     - [64*576 + 64*8, block_stride): Padding
//
// A token's scale bytes therefore live in a DIFFERENT region of the block from
// its data bytes, so the page is NOT a rank-3 `(num_blocks, block_size, width)`
// tensor and `vt::ConcatAndCacheMla` — which requires exactly that — cannot
// write it. See `.agents/specs/kv-dsv4-multicache.md` `### W8 design`.
//
// UNREACHED, and declared so under AGENTS.md "Nothing lands dead": this slice is
// the shared host reference that W8 slices 2 (`kConcatAndCacheDsMla`), 3 (the
// dequant gather) and 5 (the CUDA kernels) are each a port of. Nothing in the
// tree calls it yet; the owning row is `KV-DSV4-MULTICACHE`, the issue is #2455,
// and the spec lists it under `## Owed`.
struct Fp8DsMlaPageLayout {
  Fp8DsMlaLayout token;         // the per-token geometry (576 data + 8 scale).
  int64_t block_size;           // STORAGE rows per block (`storage_block_size()`).
  int64_t token_data_size;      // = token.token_stride_bytes (576) — `cache_utils.py:92`.
  int64_t scale_region_offset;  // = block_size * token_data_size — `:96-98`.
  int64_t scale_dim;            // = token.scale_dim (8 = 7 real + 1 pad) — `:148-149`.
  int64_t alignment_bytes;      // upstream's literal 576 (`attention.py:642`,
                                // `sparse_swa.py:99`). It coincides with
                                // `token_data_size` at V4's geometry; it is
                                // upstream's own constant, NOT derived from it.
  int64_t real_block_bytes;     // = block_size * 584 — `kv_cache_interface.py:401-403`.
  int64_t padded_block_bytes;   // = round_up(real, alignment) — `:345-351`.
};

// `block_size` is the STORAGE block size (upstream's `cache_block_size` kernel
// argument, i.e. `block_size // compress_ratio`), not the logical block size.
Fp8DsMlaPageLayout MakeFp8DsMlaPageLayout(const Fp8DsMlaLayout& layout,
                                          int64_t block_size);

// Write one token into a block, mirroring `quantize_and_insert_k_kernel`
// (`cache_utils.py:36-159`) with the quantization already done by
// `Fp8DsMlaEncodeToken`:
//   data  at `pos_in_block * token_data_size`      — nope bytes then rope bf16
//                                                    (`:92`, `:100-101`, `:155-159`)
//   scale at `scale_region_offset + pos_in_block * scale_dim`
//                                                    (`:96-98`, `:146`)
// plus an explicit ZERO pad byte at scale index `n_nope_blocks` (`:148-149`).
// A negative `pos_in_block` is upstream's `slot_idx == -1` skip (`:77-78`) and
// leaves the block byte-for-byte untouched.
void Fp8DsMlaStoreToken(uint8_t* block_base, const Fp8DsMlaPageLayout& page,
                        int64_t pos_in_block, const Fp8DsMlaToken& token);

// The exact inverse, mirroring `_dequantize_and_gather_k_kernel`'s addressing
// (`cache_utils.py:281-292`, `:306`, `:319`, `:333-339`).
//
// ASYMMETRY, and it is upstream's: the store writes `scale_dim` (8) scale bytes,
// the read consumes `n_quant_blocks = 7` (`cache_utils.py:385`). The 8th byte is
// written and never read, so the returned `scale_ue8m0` has `n_nope_blocks`
// entries and the pad byte is dropped here exactly as the gather drops it.
Fp8DsMlaToken Fp8DsMlaLoadToken(const uint8_t* block_base,
                                const Fp8DsMlaPageLayout& page,
                                int64_t pos_in_block);


// `MODEL-DSV4-DSA-COMPOSE` W1 (#2286) — ONE COMPRESSOR STEP, state machine and all.
//
// The individual pieces are already gated: `CompressorSaveScoreApe` adds the
// position-wrapped APE, `CompressorPoolNorm` does the per-column softmax pool and
// the RMSNorm. What was NOT gated is the CYCLE that drives them, and the cycle is
// where a compressor goes wrong: it is stateful across steps, so an error shows
// up several tokens later as a plausible value rather than immediately.
//
// Per step it appends each token's `(kv, score + ape)` to the layer's state, and
// at every COMPRESS BOUNDARY -- `(position + 1) % compress_ratio == 0`,
// `fused_compress_quant_cache.py:164-166` -- it pools the window that just closed
// into one compressed row.
//
// COMPRESSOR-ONLY (`coff == 1`) SHAPE. `overlap` is `compress_ratio == 4`, so a
// `compress_ratio == 128` layer gathers exactly `compress_ratio` rows ending at
// the boundary and needs no `head_offset` role selection. The overlapped shape is
// W3's.
//
//   state_kv/state_score  [n_state * head_dim] — appended in place, caller-owned
//   kv/score              [num_tokens * head_dim] this step's rows
//   positions             [num_tokens] GLOBAL positions
//   returns               the compressed rows emitted this step, [k * head_dim]
//                         (k = the number of boundaries this step crossed)
std::vector<float> CompressorStepCycle(std::vector<float>* state_kv,
                                       std::vector<float>* state_score,
                                       const std::vector<float>& kv,
                                       const std::vector<float>& score,
                                       const std::vector<float>& ape,
                                       const std::vector<int64_t>& positions,
                                       const std::vector<float>& rms_weight, float eps,
                                       int64_t compress_ratio, int64_t head_dim,
                                       // THE POOLED ROW IS ROTATED. The kernel that
                                       // writes the compressed cache applies GPT-J
                                       // RoPE to the ROPE TAIL only, unconditionally,
                                       // at `compressed_pos = (position /
                                       // compress_ratio) * compress_ratio` -- the
                                       // WINDOW'S BASE position, not the emitting
                                       // token's
                                       // (`fused_compress_quant_cache.py:272-297`).
                                       // `rope_dim == 0` leaves the row unrotated,
                                       // which is what every gate written before this
                                       // parameter existed expects.
                                       int64_t rope_dim = 0,
                                       double rope_theta = 10000.0,
                                       // `coff = 1 + (compress_ratio == 4)`
                                       // (compressor.py:247-248). At 2 the kv/score
                                       // rows are `coff*head_dim` wide and a window
                                       // position's ROLE picks which half it reads.
                                       int64_t coff = 1);

}  // namespace vllm::deepseek_v4
