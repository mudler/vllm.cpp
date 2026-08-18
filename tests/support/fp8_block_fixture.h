// The synthetic block-wise (fine-grained 128x128) FP8 Qwen3.5 dense fixture,
// shared by the #1189 model-level suites.
//
// EXTRACTED VERBATIM from `tests/vllm/model_executor/models/
// test_fp8_block_linear.cpp` (#1189 M4, `281b4bc76`) when M6 needed the same
// model to compare a MERGED projection against a split one. Two copies of a
// fixture drift, and the pair that must agree here is exactly the pair a copy
// would let diverge: M4 asserts one block GEMM per projection and M6 asserts
// the merged count over the SAME model.
//
// Everything below is `inline` and lives in one namespace, so both suites link
// one definition. No case lives here — a fixture that asserts is a suite.
#pragma once

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_fp8_block_gemm.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace fp8_block_fixture {

using vllm::DenseMlpWeights;
using vllm::ForwardLogits;
using vllm::Fp8BlockWeight;
using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::ModelForwardInput;
using vllm::ModelRegistry;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseWeights;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;
namespace block_gemm = vllm::dense_fp8_block;



// ---------------------------------------------------------------------------
// Deterministic small values
// ---------------------------------------------------------------------------

inline uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
inline float RandV(uint64_t seed) {
  const double u = static_cast<double>(Mix(seed) >> 40) /
                   static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

inline OwnedTensor MakeOwned(DType dt, const std::vector<int64_t>& shape,
                      uint64_t seed, bool nk = false) {
  OwnedTensor t;
  t.dtype = dt;
  t.nk = nk;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

inline int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// An INDEPENDENT fp8-e4m3fn decode, unpacked from the bit fields rather than
// copied from a table in `src/`: sign, 4-bit exponent with bias 7, 3-bit
// mantissa, subnormals at exponent 0, and 0x7F/0xFF as the NaN this format
// reserves in place of an infinity. Only the finite values are generated below.
inline double DecodeE4M3(uint8_t byte) {
  const int sign = (byte >> 7) & 1;
  const int exp = (byte >> 3) & 0xF;
  const int man = byte & 0x7;
  double mag = 0.0;
  if (exp == 0) {
    mag = std::ldexp(static_cast<double>(man) / 8.0, -6);
  } else if (exp == 0xF && man == 0x7) {
    mag = std::nan("");
  } else {
    mag = std::ldexp(1.0 + static_cast<double>(man) / 8.0, exp - 7);
  }
  return sign != 0 ? -mag : mag;
}

// A block-wise FP8 weight built by hand: raw fp8 bytes plus an f32 scale grid
// whose values DIFFER along both axes, so a per-tensor collapse, a transposed
// index and a floor tiling each produce a different answer.
inline Fp8BlockWeight MakeFp8Block(int64_t N, int64_t K, int64_t block_n,
                            int64_t block_k, uint64_t seed) {
  Fp8BlockWeight w;
  w.n = N;
  w.k = K;
  w.block_n = block_n;
  w.block_k = block_k;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K;
  w.packed.bytes.resize(static_cast<size_t>(N * K));
  auto* p = reinterpret_cast<uint8_t*>(w.packed.bytes.data());
  for (int64_t i = 0; i < N * K; ++i) {
    // Finite e4m3 values only: exponent 3..10 keeps the magnitude inside
    // [2^-4, 2^4) and never forms the 0x7F NaN.
    const uint64_t r = Mix(seed + static_cast<uint64_t>(i));
    const uint8_t sign = static_cast<uint8_t>((r >> 3) & 1);
    const uint8_t exp = static_cast<uint8_t>(3 + ((r >> 8) % 8));
    const uint8_t man = static_cast<uint8_t>((r >> 16) & 0x7);
    p[static_cast<size_t>(i)] =
        static_cast<uint8_t>((sign << 7) | (exp << 3) | man);
  }
  const int64_t rows = CDiv(N, block_n);
  const int64_t cols = CDiv(K, block_k);
  w.scale.dtype = DType::kF32;
  w.scale.rank = 2;
  w.scale.shape[0] = rows;
  w.scale.shape[1] = cols;
  w.scale.bytes.resize(static_cast<size_t>(rows * cols) * 4);
  auto* s = reinterpret_cast<float*>(w.scale.bytes.data());
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      // Vary along BOTH axes and by more than a rounding: 0.0625 .. 0.5625.
      s[r * cols + c] =
          0.0625F + 0.125F * static_cast<float>((r * 3 + c * 5) % 5);
    }
  }
  return w;
}

// The GEMM half of upstream's apply, written independently of the kernel under
// test: a DIFFERENT loop nest (n outermost), a `double` accumulator, and the
// bit-unpacked decode above. Mirrors `native_w8a8_block_matmul`
// (`tests/kernels/quant_utils.py:145-151` @ `5559679229`): a separate per-K-block
// partial product, scaled by `a_s * b_s`, folded into the running accumulator.
inline std::vector<double> RefBlockGemm(const std::vector<uint8_t>& a_fp8,
                                 const std::vector<float>& a_scale, int64_t M,
                                 int64_t K, const Fp8BlockWeight& w) {
  const int64_t N = w.n;
  const int64_t k_tiles = CDiv(K, w.block_k);
  const int64_t s_cols = w.scale.shape[1];
  const auto* bs = reinterpret_cast<const float*>(w.scale.bytes.data());
  const auto* bp = reinterpret_cast<const uint8_t*>(w.packed.bytes.data());
  std::vector<double> out(static_cast<size_t>(M * N), 0.0);
  for (int64_t n = 0; n < N; ++n) {
    const int64_t nb = n / w.block_n;
    for (int64_t m = 0; m < M; ++m) {
      double acc = 0.0;
      for (int64_t kt = 0; kt < k_tiles; ++kt) {
        const int64_t k0 = kt * w.block_k;
        const int64_t k1 = std::min(k0 + w.block_k, K);
        double part = 0.0;
        for (int64_t k = k0; k < k1; ++k) {
          part += DecodeE4M3(a_fp8[static_cast<size_t>(m * K + k)]) *
                  DecodeE4M3(bp[static_cast<size_t>(n * K + k)]);
        }
        acc += part * (static_cast<double>(a_scale[static_cast<size_t>(
                           m * k_tiles + kt)]) *
                       static_cast<double>(bs[nb * s_cols + kt]));
      }
      out[static_cast<size_t>(m * N + n)] = acc;
    }
  }
  return out;
}

inline vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

inline std::vector<float> ToF32(const vt::Tensor& t, int64_t numel) {
  std::vector<float> out(static_cast<size_t>(numel));
  if (t.dtype == DType::kF32) {
    std::memcpy(out.data(), t.data, static_cast<size_t>(numel) * 4);
  } else {
    REQUIRE(t.dtype == DType::kBF16);
    const auto* p = reinterpret_cast<const uint16_t*>(t.data);
    for (int64_t i = 0; i < numel; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
  }
  return out;
}

inline int64_t CountNonZero(const std::vector<float>& v) {
  int64_t n = 0;
  for (const float x : v)
    if (x != 0.0F) ++n;
  return n;
}

// ---------------------------------------------------------------------------
// A runnable block-wise Qwen3.5 dense model
// ---------------------------------------------------------------------------
//
// Every K a projection presents to the activation quant is a multiple of 128,
// because `vt::QuantFp8Group` refuses any other K and upstream asserts the same
// (`fp8_utils.py:596-599`).
//
// One N axis is deliberately NOT a multiple of 128. The GDN `in_proj_qkv` is
// 192 wide, so its scale grid's first axis is a `cdiv` with a short final
// block, which is legal (`fp8_utils.py:935-936`) and is the shape a floor
// tiling gets wrong. It is the GDN projection rather than `k_proj` because the
// attention q/k/v are ONE `QKVParallelLinear` upstream, and upstream refuses a
// merged block-quant linear whose non-final partition is ragged
// (`fp8_utils.py:1229-1244`); a 64-wide `k_proj` beside a 256-wide `q_proj` is
// a geometry vLLM cannot load, so the fixture does not ship one. See
// `.agents/specs/model-fp8-block-merged.md`.
constexpr int64_t kBlock = 128;

inline HfConfig BlockConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 128;
  c.num_hidden_layers = 2;
  c.vocab_size = 32;
  c.num_attention_heads = 2;   // Hq*Dh == 128, so o_proj's K is quantizable
  c.num_key_value_heads = 2;   // kv_n == 128, one whole block
  c.head_dim = 64;
  c.layer_types = {"linear_attention", "full_attention"};
  c.intermediate_size = 128;
  c.num_experts = 0;
  c.linear_num_key_heads = 1;
  c.linear_num_value_heads = 2;
  c.linear_key_head_dim = 32;    // conv_dim == 192, a RAGGED N
  c.linear_value_head_dim = 64;  // value_dim == 128, so out_proj's K is too
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 32;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  return c;
}

// The block-scaled GEMMs one forward dispatches over this config's two layers.
//
//   layer 0 (GDN):  in_proj_qkv, in_proj_z, out_proj        3
//                   merged gate_up, down_proj               2
//   layer 1 (attn): merged qkv, o_proj                      2
//                   merged gate_up, down_proj               2
//
// It is 9 rather than the 13 QUANTIZED PROJECTIONS, because vLLM runs `gate`
// and `up` as one MergedColumnParallelLinear and `q`/`k`/`v` as one
// QKVParallelLinear, and #1189 M6 does too
// (`.agents/specs/model-fp8-block-merged.md`). The merged GEMM is
// byte-identical to the split ones it replaces, so this count is the ONLY
// instrument that can tell the two topologies apart.
constexpr int64_t kBlockProjectionsPerForward = 9;

// `arm` false builds the same model with plain BF16 projections — the negative
// control, and the twin G5 perturbs.
inline Qwen3_5DenseWeights BlockWeights(const HfConfig& c, bool block_arm,
                                 float scale_bump = 1.0F) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size, I = c.intermediate_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);

  // One projection, either arm. The block weight is [N,K]; the bf16 fallback
  // keeps the Matmul-B [K,N] orientation the synthetic dense tests use.
  const auto proj = [&](OwnedTensor& bf16, Fp8BlockWeight& blk, int64_t N,
                        int64_t K, uint64_t seed) {
    if (block_arm) {
      blk = MakeFp8Block(N, K, kBlock, kBlock, seed);
    } else {
      bf16 = MakeOwned(DType::kBF16, {K, N}, seed);
    }
  };

  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      proj(lw.gdn.in_proj_qkv, lw.gdn.in_proj_qkv_fp8_block, conv_dim, H, s + 10);
      proj(lw.gdn.in_proj_z, lw.gdn.in_proj_z_fp8_block, value_dim, H, s + 20);
      proj(lw.gdn.out_proj, lw.gdn.out_proj_fp8_block, H, value_dim, s + 90);
      // b/a are small and this checkpoint family leaves them unquantized.
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
    } else {
      proj(lw.attn.q_proj, lw.attn.q_proj_fp8_block, 2 * Hq * Dh, H, s + 10);
      proj(lw.attn.k_proj, lw.attn.k_proj_fp8_block, Hkv * Dh, H, s + 20);
      proj(lw.attn.v_proj, lw.attn.v_proj_fp8_block, Hkv * Dh, H, s + 30);
      proj(lw.attn.o_proj, lw.attn.o_proj_fp8_block, H, Hq * Dh, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    proj(lw.mlp.gate_proj, lw.mlp.gate_proj_fp8_block, I, H, s + 501);
    proj(lw.mlp.up_proj, lw.mlp.up_proj_fp8_block, I, H, s + 502);
    proj(lw.mlp.down_proj, lw.mlp.down_proj_fp8_block, H, I, s + 503);
    w.layers.push_back(std::move(lw));
  }

  // G5's instrument: multiply ONE block of ONE projection's scale grid. A
  // per-tensor collapse to element (0,0), an epilogue-folded alpha, and a
  // transposed scale index are each blind to it.
  if (block_arm && scale_bump != 1.0F) {
    Fp8BlockWeight& q = w.layers[1].attn.q_proj_fp8_block;
    auto* s = reinterpret_cast<float*>(q.scale.bytes.data());
    s[q.scale.shape[1]] *= scale_bump;  // row 1, column 0 — never element (0,0)
  }
  return w;
}

struct CachePool {
  const HfConfig& c;
  int64_t num_blocks;
  int64_t block_size;
  DType kv_dtype;
  std::vector<std::vector<float>> full_attn_buf;
  std::vector<std::vector<float>> gdn_ssm_buf;
  std::vector<std::vector<float>> gdn_conv_buf;
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;

  // `kv` is the paged KV cache's STORAGE dtype, and bf16 is the production
  // default (`include/vllm/v1/kv_cache_dtype.h`) as well as the only dtype this
  // arm can use: the block method emits a bf16 V at `v_proj` -- upstream's
  // `out_dtype`, the model dtype -- and `vt::ReshapeAndCache`'s auto path
  // requires k/v and the cache to share one dtype, mirroring upstream's own
  // `reshape_and_cache_flash`. An f32 cache (`VT_KV_CACHE_F32=1`) therefore
  // refuses EVERY bf16 arm, not only this one; that is issue #1249, it predates
  // this row, and the row's spec lists it under `## Owed`.
  CachePool(const HfConfig& cfg, int64_t nb, int64_t bs,
            DType kv = DType::kBF16)
      : c(cfg), num_blocks(nb), block_size(bs), kv_dtype(kv) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                  Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
    const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      if (c.layer_types[static_cast<size_t>(l)] == "linear_attention") {
        gdn_ssm_buf.emplace_back(static_cast<size_t>(nb * Hv * Dv * Dk), 0.0F);
        gdn_conv_buf.emplace_back(
            static_cast<size_t>(nb * conv_dim * (Kw - 1)), 0.0F);
      } else {
        // Sized in floats; a bf16 cache needs half the bytes and the buffer is
        // simply larger than it has to be.
        full_attn_buf.emplace_back(
            static_cast<size_t>(nb * 2 * bs * Hkv * Dh), 0.0F);
      }
    }
    const vt::Device cpu{vt::DeviceType::kCPU, 0};
    for (auto& b : full_attn_buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = kv_dtype;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
    for (size_t g = 0; g < gdn_ssm_buf.size(); ++g) {
      GdnStateCache gs;
      gs.ssm_state = vt::Tensor::Contiguous(gdn_ssm_buf[g].data(), DType::kF32,
                                            cpu, {num_blocks, Hv, Dv, Dk});
      gs.conv_state = vt::Tensor::Contiguous(gdn_conv_buf[g].data(),
                                             DType::kF32, cpu,
                                             {num_blocks, conv_dim, Kw - 1});
      gdn_state.push_back(gs);
    }
  }
};

inline CommonAttentionMetadata PrefillAttnMeta(int64_t T,
                                        const std::vector<int32_t>& blocks,
                                        int64_t block_size) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = static_cast<int>(blocks.size());
  m.block_table_tensor = blocks;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t blk = blocks[static_cast<size_t>(t / block_size)];
    m.slot_mapping.push_back(blk * block_size + t % block_size);
  }
  m.causal = true;
  return m;
}

inline GDNAttentionMetadata PrefillGdnMeta(int64_t T) {
  GDNAttentionMetadata g;
  g.num_prefills = 1;
  g.num_prefill_tokens = static_cast<int>(T);
  g.num_decodes = 0;
  g.num_decode_tokens = 0;
  g.num_actual_tokens = static_cast<int>(T);
  g.has_initial_state = std::vector<uint8_t>{0};
  g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_state_indices = std::vector<int32_t>{0};
  g.prefill_has_initial_state = std::vector<uint8_t>{0};
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}

// Drive the type-erased registry forward once and return the [T, vocab] logits.
// `counted` receives how many block-scaled GEMMs the forward dispatched.
inline std::vector<float> RegistryForward(const HfConfig& c,
                                   const Qwen3_5DenseWeights& w,
                                   uint64_t* counted,
                                   DType kv_dtype = DType::kBF16) {
  const int64_t T = 4;
  const std::vector<int32_t> ids = {5, 9, 2, 17};
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  const std::vector<int32_t> logits_indices;
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/8, kv_dtype);
  const CommonAttentionMetadata am = PrefillAttnMeta(T, {0}, 8);
  const GDNAttentionMetadata gm = PrefillGdnMeta(T);
  vt::Queue q = Q();
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  ModelForwardInput in{ids, pos, am, gm, pool.attn_kv, pool.gdn_state, c, q,
                       logits_indices};
  in.num_reqs = 1;
  // The HOST logits route, so the case can compare values without a device
  // read-back. Byte-identical arithmetic to the gathered one; only the sink
  // differs (`ForwardQwen3_5Dense`).
  in.gather_logits = false;
  const uint64_t before = block_gemm::BlockGemmCount();
  const ForwardLogits out = ModelRegistry::Forward(*model, in);
  if (counted != nullptr) *counted = block_gemm::BlockGemmCount() - before;
  REQUIRE(out.host.size() == static_cast<size_t>(T * c.vocab_size));
  return out.host;
}

inline bool AllFinite(const std::vector<float>& v) {
  for (const float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

}  // namespace fp8_block_fixture
