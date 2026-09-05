// DeepSeek-V4-Flash W4 primitives — host reference implementations.
// See deepseek_v4_compressor.h for the full port map (file:line on both sides).
#include "vllm/model_executor/models/deepseek_v4_compressor.h"

#include "vllm/model_executor/models/deepseek_v4_rope.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"  // F32ToF8E4M3, kFloat8E4M3Max
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // F8E4M3ToF32
#include "vt/dtype.h"  // VT_CHECK, BF16ToF32, F32ToBF16

namespace vllm::deepseek_v4 {

std::vector<float> CompressorSaveScoreApe(const std::vector<float>& score,
                                          const std::vector<float>& ape,
                                          const std::vector<int64_t>& positions,
                                          int64_t num_tokens, int64_t width,
                                          int64_t compress_ratio) {
  VT_CHECK(width > 0 && compress_ratio > 0, "bad compressor dims");
  VT_CHECK(static_cast<int64_t>(score.size()) == num_tokens * width,
           "score size mismatch");
  VT_CHECK(static_cast<int64_t>(ape.size()) == compress_ratio * width,
           "ape size mismatch");
  VT_CHECK(static_cast<int64_t>(positions.size()) == num_tokens,
           "positions size mismatch");

  std::vector<float> out(static_cast<size_t>(num_tokens) * width);
  for (int64_t t = 0; t < num_tokens; ++t) {
    // ape_row = position % compress_ratio (save_partial_states.py:94). Positions
    // are non-negative token positions; guard the C++ sign of % defensively.
    int64_t ape_row = positions[t] % compress_ratio;
    if (ape_row < 0) ape_row += compress_ratio;
    const float* s = &score[t * width];
    const float* a = &ape[ape_row * width];
    float* o = &out[t * width];
    for (int64_t d = 0; d < width; ++d) o[d] = s[d] + a[d];
  }
  return out;
}

std::vector<float> CompressorPoolNorm(const std::vector<float>& kv,
                                      const std::vector<float>& score,
                                      const std::vector<uint8_t>& valid,
                                      const std::vector<float>& rms_weight,
                                      float eps, int64_t window,
                                      int64_t head_dim) {
  VT_CHECK(window > 0 && head_dim > 0, "bad pool dims");
  VT_CHECK(static_cast<int64_t>(kv.size()) == window * head_dim,
           "kv size mismatch");
  VT_CHECK(static_cast<int64_t>(score.size()) == window * head_dim,
           "score size mismatch");
  VT_CHECK(static_cast<int64_t>(valid.size()) == window, "valid size mismatch");
  VT_CHECK(static_cast<int64_t>(rms_weight.size()) == head_dim,
           "rms_weight size mismatch");

  const float kNegInf = -std::numeric_limits<float>::infinity();

  // ── Softmax over the window (dim=0), per head-dim column, then weighted sum.
  // fused_compress_quant_cache.py:199-212: masked rows load score=-inf / kv=0.
  std::vector<float> compressed(static_cast<size_t>(head_dim), 0.0f);
  for (int64_t d = 0; d < head_dim; ++d) {
    // Max over the window for numerical stability.
    float m = kNegInf;
    for (int64_t i = 0; i < window; ++i) {
      const float s = valid[i] ? score[i * head_dim + d] : kNegInf;
      m = std::max(m, s);
    }
    if (m == kNegInf) continue;  // whole column masked -> compressed stays 0
    float denom = 0.0f;
    float acc = 0.0f;
    for (int64_t i = 0; i < window; ++i) {
      if (!valid[i]) continue;
      const float e = std::exp(score[i * head_dim + d] - m);
      denom += e;
      acc += kv[i * head_dim + d] * e;
    }
    compressed[static_cast<size_t>(d)] = acc / denom;
  }

  // ── RMSNorm in fp32 (fused_compress_quant_cache.py:214-218).
  float variance = 0.0f;
  for (int64_t d = 0; d < head_dim; ++d)
    variance += compressed[static_cast<size_t>(d)] * compressed[static_cast<size_t>(d)];
  variance /= static_cast<float>(head_dim);
  const float rrms = 1.0f / std::sqrt(variance + eps);
  std::vector<float> normed(static_cast<size_t>(head_dim));
  for (int64_t d = 0; d < head_dim; ++d)
    normed[static_cast<size_t>(d)] =
        compressed[static_cast<size_t>(d)] * rrms * rms_weight[static_cast<size_t>(d)];
  return normed;
}

Fp8DsMlaLayout MakeFp8DsMlaLayout(int64_t nope_head_dim, int64_t rope_head_dim,
                                  int64_t quant_block) {
  VT_CHECK(quant_block > 0 && nope_head_dim % quant_block == 0,
           "nope_head_dim must be a multiple of quant_block");
  Fp8DsMlaLayout L;
  L.nope_head_dim = nope_head_dim;
  L.rope_head_dim = rope_head_dim;
  L.quant_block = quant_block;
  L.n_nope_blocks = nope_head_dim / quant_block;   // 7 for V4
  // token_stride: nope*1 byte (fp8) + rope*2 bytes (bf16). = 576 for V4.
  L.token_stride_bytes = nope_head_dim + rope_head_dim * 2;
  // scale region: n_nope_blocks real UE8M0 bytes + 1 pad (compressor.py:309).
  L.scale_dim = L.n_nope_blocks + 1;               // 8 for V4
  return L;
}

Fp8DsMlaToken Fp8DsMlaEncodeToken(const std::vector<float>& head,
                                  const Fp8DsMlaLayout& layout) {
  const int64_t D = layout.nope_head_dim + layout.rope_head_dim;
  VT_CHECK(static_cast<int64_t>(head.size()) == D, "head size mismatch");

  Fp8DsMlaToken t;
  t.nope_fp8.assign(static_cast<size_t>(layout.nope_head_dim), 0);
  t.scale_ue8m0.assign(static_cast<size_t>(layout.n_nope_blocks), 0);
  t.rope_bf16.assign(static_cast<size_t>(layout.rope_head_dim), 0);

  const float kInvFp8Max = 1.0f / kFloat8E4M3Max;  // 1/448

  for (int64_t b = 0; b < layout.n_nope_blocks; ++b) {
    const int64_t base = b * layout.quant_block;
    // bf16 round each element (kernel casts fp32 -> bf16 -> fp32), then absmax.
    float absmax = 0.0f;
    for (int64_t j = 0; j < layout.quant_block; ++j) {
      const float q = vt::BF16ToF32(vt::F32ToBF16(head[base + j]));
      absmax = std::max(absmax, std::fabs(q));
    }
    absmax = std::max(absmax, 1e-4f);
    // UE8M0 power-of-two scale: exponent = ceil(log2(absmax / FP8_MAX)).
    const float raw_scale = absmax * kInvFp8Max;
    const float exponent = std::ceil(std::log2(raw_scale));
    const float inv_scale = std::exp2(-exponent);
    for (int64_t j = 0; j < layout.quant_block; ++j) {
      const float q = vt::BF16ToF32(vt::F32ToBF16(head[base + j]));
      float x = q * inv_scale;
      x = std::clamp(x, -kFloat8E4M3Max, kFloat8E4M3Max);
      t.nope_fp8[static_cast<size_t>(base + j)] = F32ToF8E4M3(x);
    }
    // Encoded UE8M0 byte = clamp(exponent + 127, 0, 255).
    float encoded = exponent + 127.0f;
    encoded = std::max(0.0f, std::min(255.0f, encoded));
    t.scale_ue8m0[static_cast<size_t>(b)] = static_cast<uint8_t>(encoded);
  }

  // RoPE part stored bf16 verbatim (already-rotated on GPU; a W3/W7 seam here).
  for (int64_t j = 0; j < layout.rope_head_dim; ++j)
    t.rope_bf16[static_cast<size_t>(j)] =
        vt::F32ToBF16(head[layout.nope_head_dim + j]);
  return t;
}

std::vector<float> Fp8DsMlaDecodeToken(const Fp8DsMlaToken& token,
                                       const Fp8DsMlaLayout& layout) {
  VT_CHECK(static_cast<int64_t>(token.nope_fp8.size()) == layout.nope_head_dim,
           "nope_fp8 size mismatch");
  VT_CHECK(static_cast<int64_t>(token.scale_ue8m0.size()) == layout.n_nope_blocks,
           "scale size mismatch");
  VT_CHECK(static_cast<int64_t>(token.rope_bf16.size()) == layout.rope_head_dim,
           "rope size mismatch");

  const int64_t D = layout.nope_head_dim + layout.rope_head_dim;
  std::vector<float> out(static_cast<size_t>(D), 0.0f);
  for (int64_t b = 0; b < layout.n_nope_blocks; ++b) {
    // scale_pow2 = 2^(scale_byte - 127) (SGLang dequant_k_cache.py:125).
    const float scale_pow2 =
        std::exp2(static_cast<float>(token.scale_ue8m0[static_cast<size_t>(b)]) - 127.0f);
    for (int64_t j = 0; j < layout.quant_block; ++j) {
      const int64_t d = b * layout.quant_block + j;
      out[static_cast<size_t>(d)] =
          F8E4M3ToF32(token.nope_fp8[static_cast<size_t>(d)]) * scale_pow2;
    }
  }
  for (int64_t j = 0; j < layout.rope_head_dim; ++j)
    out[static_cast<size_t>(layout.nope_head_dim + j)] =
        vt::BF16ToF32(token.rope_bf16[static_cast<size_t>(j)]);
  return out;
}

// ── (B2) The paged fp8_ds_mla BLOCK — `KV-DSV4-MULTICACHE` W8 slice 1 ────────
// See the header for the region-split block comment this ports verbatim.

namespace {
// Upstream's `round_up` as `_apply_alignment_padding` uses it
// (`kv_cache_interface.py:345-351`). `alignment` is > 0 by construction here.
int64_t RoundUpTo(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}
// Upstream passes `alignment=576` as a LITERAL on both fp8_ds_mla specs
// (`attention.py:642`, `sparse_swa.py:99`). At V4's geometry it happens to equal
// `token_stride_bytes`; it is upstream's own constant and is NOT derived from
// the token layout, so it is written here as upstream writes it.
constexpr int64_t kFp8DsMlaAlignmentBytes = 576;
}  // namespace

Fp8DsMlaPageLayout MakeFp8DsMlaPageLayout(const Fp8DsMlaLayout& layout,
                                          int64_t block_size) {
  VT_CHECK(block_size > 0, "fp8_ds_mla page: block_size must be > 0");
  VT_CHECK(layout.token_stride_bytes > 0 && layout.scale_dim > 0,
           "fp8_ds_mla page: layout is not initialized "
           "(use MakeFp8DsMlaLayout)");
  Fp8DsMlaPageLayout P;
  P.token = layout;
  P.block_size = block_size;
  // "Token data pointer: token data is stored contiguously at start of block"
  // (`cache_utils.py:90-92`).
  P.token_data_size = layout.token_stride_bytes;  // 576 for V4
  // "Scale pointer: scales are stored after ALL token data in the block"
  // (`cache_utils.py:94-98`): `cache_block_size * token_data_size`.
  P.scale_region_offset = block_size * P.token_data_size;
  P.scale_dim = layout.scale_dim;  // 8 = 7 real + 1 pad
  P.alignment_bytes = kFp8DsMlaAlignmentBytes;
  // 584 = 576 data + 8 scale, per token (`kv_cache_interface.py:401-403`).
  P.real_block_bytes = block_size * (P.token_data_size + P.scale_dim);
  P.padded_block_bytes = RoundUpTo(P.real_block_bytes, P.alignment_bytes);
  return P;
}

void Fp8DsMlaStoreToken(uint8_t* block_base, const Fp8DsMlaPageLayout& page,
                        int64_t pos_in_block, const Fp8DsMlaToken& token) {
  // Upstream's `if slot_idx == -1: return` (`cache_utils.py:77-78`). A padded
  // or masked token writes NOTHING — not a zero row, not a partial row.
  if (pos_in_block < 0) return;
  VT_CHECK(block_base != nullptr, "fp8_ds_mla store: null block");
  VT_CHECK(pos_in_block < page.block_size,
           "fp8_ds_mla store: pos_in_block out of range");
  const Fp8DsMlaLayout& L = page.token;
  VT_CHECK(static_cast<int64_t>(token.nope_fp8.size()) == L.nope_head_dim,
           "fp8_ds_mla store: nope_fp8 size mismatch");
  VT_CHECK(static_cast<int64_t>(token.scale_ue8m0.size()) == L.n_nope_blocks,
           "fp8_ds_mla store: scale size mismatch");
  VT_CHECK(static_cast<int64_t>(token.rope_bf16.size()) == L.rope_head_dim,
           "fp8_ds_mla store: rope size mismatch");

  // token_data_ptr = cache_block_ptr + pos_in_block * token_data_size (`:92`).
  uint8_t* data = block_base + pos_in_block * page.token_data_size;
  // "Token data layout: [0:448] fp8, [448:576] bf16" (`:100-102`).
  std::memcpy(data, token.nope_fp8.data(),
              static_cast<size_t>(L.nope_head_dim));
  // `:155-159`: the rope half is stored through a bf16 pointer, i.e. host-native
  // 16-bit words, verbatim and unquantized.
  std::memcpy(data + L.nope_head_dim, token.rope_bf16.data(),
              static_cast<size_t>(L.rope_head_dim) * sizeof(uint16_t));

  // token_scale_ptr = cache_block_ptr + cache_block_size * token_data_size
  //                   + pos_in_block * scale_dim  (`:96-98`).
  uint8_t* scales =
      block_base + page.scale_region_offset + pos_in_block * page.scale_dim;
  std::memcpy(scales, token.scale_ue8m0.data(),
              static_cast<size_t>(L.n_nope_blocks));
  // "Padding scale at index 7": `tl.store(token_scale_ptr + 7, zeros)` (`:148-149`).
  // Written EXPLICITLY, so the byte is 0 rather than whatever the page held.
  for (int64_t b = L.n_nope_blocks; b < page.scale_dim; ++b)
    scales[static_cast<size_t>(b)] = 0;
}

Fp8DsMlaToken Fp8DsMlaLoadToken(const uint8_t* block_base,
                                const Fp8DsMlaPageLayout& page,
                                int64_t pos_in_block) {
  VT_CHECK(block_base != nullptr, "fp8_ds_mla load: null block");
  VT_CHECK(pos_in_block >= 0 && pos_in_block < page.block_size,
           "fp8_ds_mla load: pos_in_block out of range");
  const Fp8DsMlaLayout& L = page.token;

  Fp8DsMlaToken t;
  t.nope_fp8.resize(static_cast<size_t>(L.nope_head_dim));
  t.scale_ue8m0.resize(static_cast<size_t>(L.n_nope_blocks));
  t.rope_bf16.resize(static_cast<size_t>(L.rope_head_dim));

  // Identical addressing to the store (`cache_utils.py:281-292`).
  const uint8_t* data = block_base + pos_in_block * page.token_data_size;
  std::memcpy(t.nope_fp8.data(), data, static_cast<size_t>(L.nope_head_dim));  // :306
  std::memcpy(t.rope_bf16.data(), data + L.nope_head_dim,
              static_cast<size_t>(L.rope_head_dim) * sizeof(uint16_t));  // :333-339

  // `:319` reads `token_scale_ptr + qblock_idx` for `qblock_idx` in
  // `static_range(n_quant_blocks)` with n_quant_blocks = 7 (`:385`) — ONE FEWER
  // than the store writes. The 8th byte is deliberately not read.
  const uint8_t* scales =
      block_base + page.scale_region_offset + pos_in_block * page.scale_dim;
  std::memcpy(t.scale_ue8m0.data(), scales,
              static_cast<size_t>(L.n_nope_blocks));
  return t;
}


// MODEL-DSV4-DSA-COMPOSE W1 (#2286). See the header for the cycle's contract.
std::vector<float> CompressorStepCycle(std::vector<float>* state_kv,
                                       std::vector<float>* state_score,
                                       const std::vector<float>& kv,
                                       const std::vector<float>& score,
                                       const std::vector<float>& ape,
                                       const std::vector<int64_t>& positions,
                                       const std::vector<float>& rms_weight, float eps,
                                       int64_t compress_ratio, int64_t head_dim,
                                       int64_t rope_dim, double rope_theta,
                                       int64_t coff) {
  VT_CHECK(state_kv != nullptr && state_score != nullptr,
           "CompressorStepCycle: state buffers are required");
  VT_CHECK(compress_ratio > 0 && head_dim > 0,
           "CompressorStepCycle: compress_ratio and head_dim must be > 0");
  VT_CHECK(coff == 1 || coff == 2,
           "CompressorStepCycle: coff is 1 + (compress_ratio == 4) and is 1 or 2 "
           "(compressor.py:247-248)");
  const int64_t T = static_cast<int64_t>(positions.size());
  // At `coff == 2` the projections are DOUBLED: a row carries BOTH roles, and the
  // gather below picks which half each window position reads.
  const int64_t width = coff * head_dim;
  VT_CHECK(static_cast<int64_t>(kv.size()) == T * width &&
               static_cast<int64_t>(score.size()) == T * width,
           "CompressorStepCycle: kv/score must be [num_tokens, coff*head_dim]");
  VT_CHECK(state_kv->size() == state_score->size(),
           "CompressorStepCycle: the two state buffers must stay in lockstep");

  // SAVE: the score carries the position-wrapped APE, the kv does not. Reusing
  // the gated helper rather than re-deriving `position % compress_ratio` here.
  const std::vector<float> scored =
      CompressorSaveScoreApe(score, ape, positions, T, width, compress_ratio);

  std::vector<float> emitted;
  for (int64_t t = 0; t < T; ++t) {
    state_kv->insert(state_kv->end(), kv.begin() + t * width,
                     kv.begin() + (t + 1) * width);
    state_score->insert(state_score->end(), scored.begin() + t * width,
                        scored.begin() + (t + 1) * width);

    // BOUNDARY-ONLY EMISSION. `(position + 1) % compress_ratio == 0` is upstream's
    // gate verbatim; a step that crosses none emits nothing, and a step that
    // crosses several emits several -- which is why prefill and decode need no
    // separate paths here.
    const int64_t pos = positions[static_cast<size_t>(t)];
    if ((pos + 1) % compress_ratio != 0) continue;

    // THE GATHERING WINDOW, 1:1 with `fused_compress_quant_cache.py:169-183`:
    //
    //     start  = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    //     tokens = arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    //     head_offset = (tokens >= COMPRESS_RATIO) * HEAD_SIZE
    //
    // So `coff * compress_ratio` rows are gathered ending at the boundary, and a
    // row's ROLE is its index WITHIN THIS WINDOW, not a property of the row: the
    // first `compress_ratio` positions read the state's low half, the rest read
    // the high half. That is why the tensors are doubled and why the role "is
    // never recoverable from the tensor alone".
    const int64_t have = static_cast<int64_t>(state_kv->size()) / width;
    const int64_t win = coff * compress_ratio;
    std::vector<float> wkv(static_cast<size_t>(win * head_dim), 0.0f);
    std::vector<float> wsc(static_cast<size_t>(win * head_dim), 0.0f);
    std::vector<uint8_t> valid(static_cast<size_t>(win), 0);
    for (int64_t i = 0; i < win; ++i) {
      const int64_t row = have - win + i;  // global row index into the state
      if (row < 0) continue;               // masked: `pos >= 0`, before the start
      valid[static_cast<size_t>(i)] = 1;
      const int64_t head_offset = (i >= compress_ratio) ? head_dim : 0;
      const int64_t base = row * width + head_offset;
      std::copy(state_kv->begin() + base, state_kv->begin() + base + head_dim,
                wkv.begin() + i * head_dim);
      std::copy(state_score->begin() + base, state_score->begin() + base + head_dim,
                wsc.begin() + i * head_dim);
    }
    std::vector<float> pooled =
        CompressorPoolNorm(wkv, wsc, valid, rms_weight, eps, win, head_dim);
    // ROTATE THE POOLED ROW. `fused_compress_quant_cache.py:272-297` applies
    // GPT-J RoPE to the rope tail only, and does so UNCONDITIONALLY -- the
    // `rotate` constructor flag is dead and both compressors pass it. The
    // position is `(position / compress_ratio) * compress_ratio`, the window's
    // BASE rather than the emitting token's, so every row of one window shares a
    // phase.
    if (rope_dim > 0) {
      VT_CHECK(rope_dim % 2 == 0 && rope_dim <= head_dim,
               "CompressorStepCycle: the rope tail is rotated in PAIRS and lies "
               "inside the head");
      const int64_t compressed_pos = (pos / compress_ratio) * compress_ratio;
      RopeInplaceLayer(pooled.data() + (head_dim - rope_dim), rope_dim,
                       compressed_pos, rope_theta, /*freq_scale=*/1.0,
                       /*ext_factor=*/0.0, /*n_ctx_orig=*/0, /*beta_fast=*/0.0,
                       /*beta_slow=*/0.0);
    }
    emitted.insert(emitted.end(), pooled.begin(), pooled.end());
  }
  return emitted;
}

}  // namespace vllm::deepseek_v4
