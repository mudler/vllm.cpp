// MODEL-FP8-BLOCK-LINEAR — #1189 milestone M4, spec
// `.agents/specs/model-fp8-block-linear.md`.
//
// A block-wise (fine-grained 128x128) FP8 checkpoint RUNS: the linear method,
// the shared compute body it wraps, and the ten Qwen3.5 dense projections that
// reach it.
//
// The load half is M3's and is gated by `test_fp8_block_weight_load.cpp`. What
// is new here is the CONSUMPTION, so the cases are built around three questions
// a numeric comparison alone cannot answer:
//
//   1. does a PRODUCTION entry point reach the arm at all (G3, the dispatch
//      counter through `ModelRegistry::Forward`);
//   2. is the weight still `N*K` fp8 bytes at the GEMM boundary, or did
//      something dequantize it to bf16 on the way — which is numerically BETTER
//      and therefore invisible to every correctness gate (G4);
//   3. is the scale really per BLOCK, or did it collapse to one number (G5,
//      which perturbs ONE block and requires the logits to move).
//
// No checkpoint download, no GPU, no snapshot. `vt::MatmulFp8BlockScaled` is a
// CPU correctness reference (M2) and that is what makes this gateable here; the
// CUDA kernel is #1189 M5 and until it lands a CUDA device refuses this
// checkpoint by name at `Prepare`, which G6 asserts.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/quantization/fp8.h"
#include "vllm/model_executor/layers/quantization/fp8_block.h"
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

namespace {

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
namespace layers = vllm::layers;
namespace block_gemm = vllm::dense_fp8_block;

// ---------------------------------------------------------------------------
// Deterministic small values
// ---------------------------------------------------------------------------

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u = static_cast<double>(Mix(seed) >> 40) /
                   static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

OwnedTensor MakeOwned(DType dt, const std::vector<int64_t>& shape,
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

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// An INDEPENDENT fp8-e4m3fn decode, unpacked from the bit fields rather than
// copied from a table in `src/`: sign, 4-bit exponent with bias 7, 3-bit
// mantissa, subnormals at exponent 0, and 0x7F/0xFF as the NaN this format
// reserves in place of an infinity. Only the finite values are generated below.
double DecodeE4M3(uint8_t byte) {
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
Fp8BlockWeight MakeFp8Block(int64_t N, int64_t K, int64_t block_n,
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
std::vector<double> RefBlockGemm(const std::vector<uint8_t>& a_fp8,
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

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

std::vector<float> ToF32(const vt::Tensor& t, int64_t numel) {
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

int64_t CountNonZero(const std::vector<float>& v) {
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
// (`fp8_utils.py:596-599`). The N axes are deliberately NOT all multiples of
// 128: `k_proj`/`v_proj` are 64 wide, so the scale grid's first axis is a
// `cdiv` with a short final block, which is legal (`fp8_utils.py:935-936`) and
// is the shape a floor tiling gets wrong.
constexpr int64_t kBlock = 128;

HfConfig BlockConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 128;
  c.num_hidden_layers = 2;
  c.vocab_size = 32;
  c.num_attention_heads = 2;   // Hq*Dh == 128, so o_proj's K is quantizable
  c.num_key_value_heads = 1;   // kv_n == 64, a RAGGED N
  c.head_dim = 64;
  c.layer_types = {"linear_attention", "full_attention"};
  c.intermediate_size = 128;
  c.num_experts = 0;
  c.linear_num_key_heads = 1;
  c.linear_num_value_heads = 2;
  c.linear_key_head_dim = 64;
  c.linear_value_head_dim = 64;  // value_dim == 128, so out_proj's K is too
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 32;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  return c;
}

// The ten projections a block-wise `Qwen3_5ForConditionalGeneration` quantizes,
// per this config's two layers: 3 GDN + 3 MLP on layer 0, 4 attn + 3 MLP on
// layer 1.
constexpr int64_t kBlockProjectionsPerForward = 13;

// `arm` false builds the same model with plain BF16 projections — the negative
// control, and the twin G5 perturbs.
Qwen3_5DenseWeights BlockWeights(const HfConfig& c, bool block_arm,
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

CommonAttentionMetadata PrefillAttnMeta(int64_t T,
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

GDNAttentionMetadata PrefillGdnMeta(int64_t T) {
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
std::vector<float> RegistryForward(const HfConfig& c,
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

bool AllFinite(const std::vector<float>& v) {
  for (const float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

}  // namespace

// G1 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: the factory selects the block method by weight presence") {
  OwnedTensor bf16 = MakeOwned(DType::kBF16, {128, 128}, 3, /*nk=*/true);
  const Fp8BlockWeight empty;
  const Fp8BlockWeight block = MakeFp8Block(128, 128, kBlock, kBlock, 31);
  REQUIRE(empty.Empty());
  REQUIRE_FALSE(block.Empty());

  // get_quant_method analogue (`fp8.py:297-298`): `weight_block_size is not
  // None` is the whole dispatch, resolved ONCE from the populated weight.
  auto m_bf16 = layers::MakeLinearMethod(bf16, empty);
  CHECK(std::string(m_bf16->Name()) == "bf16-unquantized");

  auto m_block = layers::MakeLinearMethod(bf16, block);
  CHECK(std::string(m_block->Name()) == "fp8-w8a8-block");
  // The NAME alone would pass for a method that returned it from the wrong
  // class, so the concrete type is pinned too.
  CHECK(dynamic_cast<const layers::Fp8BlockLinearMethod*>(m_block.get()) !=
        nullptr);

  // The overloads coexist and are chosen by the weight TYPE, not by a runtime
  // probe: the per-tensor factory is in scope here and an `Fp8Weight` argument
  // still reaches it rather than this one.
  const vllm::Fp8Weight per_tensor;
  auto m_pt = layers::MakeLinearMethod(bf16, per_tensor);
  CHECK(std::string(m_pt->Name()) == "bf16-unquantized");
}

// G2 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: Apply is upstream's quant-then-block-GEMM, in f32, stored to out_dtype") {
  // block_n != block_k on purpose. Both are 128 in every checkpoint in play, so
  // a body that swapped the two arguments would be invisible at 128x128; here
  // the scale grid is [cdiv(N,64), cdiv(K,128)] and a swap misindexes it.
  const int64_t M = 3, K = 256, N = 192;
  const int64_t bn = 64, bk = 128;
  const Fp8BlockWeight w = MakeFp8Block(N, K, bn, bk, 77);
  REQUIRE(w.scale.shape[0] == CDiv(N, bn));
  REQUIRE(w.scale.shape[1] == CDiv(K, bk));

  OwnedTensor xw = MakeOwned(DType::kBF16, {M, K}, 5);
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());

  auto method = layers::MakeLinearMethod(MakeOwned(DType::kBF16, {N, K}, 1), w);
  REQUIRE(std::string(method->Name()) == "fp8-w8a8-block");

  SUBCASE("bf16 out, which is upstream's torch.get_default_dtype()") {
    vllm::dense_attn::DBuf out = method->Apply(d, x.t(), DType::kBF16);
    REQUIRE(out.t().dtype == DType::kBF16);
    REQUIRE(out.t().shape[0] == M);
    REQUIRE(out.t().shape[1] == N);

    // The activation half, run through the gated M1 op, so the reference below
    // measures the GEMM and the composition rather than re-deriving the e4m3
    // encoder a byte-exact gate already owns.
    vllm::dense_attn::DBuf a_fp8(d, DType::kI8, {M, K});
    vllm::dense_attn::DBuf a_scale(d, DType::kF32, {M, K / bk});
    vt::QuantFp8Group(q, a_fp8.t(), a_scale.t(), x.t(), static_cast<int>(bk));
    std::vector<uint8_t> a_bytes(static_cast<size_t>(M * K));
    std::memcpy(a_bytes.data(), a_fp8.t().data, a_bytes.size());
    std::vector<float> a_s(static_cast<size_t>(M * (K / bk)));
    std::memcpy(a_s.data(), a_scale.t().data, a_s.size() * 4);

    const std::vector<double> ref = RefBlockGemm(a_bytes, a_s, M, K, w);
    const std::vector<float> got = ToF32(out.t(), M * N);

    // bf16 carries 8 significand bits, so the store rounds at ~2^-9 relative.
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double denom = std::max(std::abs(ref[i]), 1e-6);
      worst = std::max(worst, std::abs(got[i] - ref[i]) / denom);
    }
    CHECK(worst < 8e-3);
    // A vacuity guard: an all-zero output satisfies every relative comparison
    // whose reference is also zero, and this one would not be.
    CHECK(CountNonZero(got) == static_cast<int64_t>(got.size()));
  }

  SUBCASE("f32 out is the same value, stored wider") {
    vllm::dense_attn::DBuf out = method->Apply(d, x.t(), DType::kF32);
    REQUIRE(out.t().dtype == DType::kF32);
    const std::vector<float> got = ToF32(out.t(), M * N);
    CHECK(CountNonZero(got) == static_cast<int64_t>(got.size()));

    vllm::dense_attn::DBuf a_fp8(d, DType::kI8, {M, K});
    vllm::dense_attn::DBuf a_scale(d, DType::kF32, {M, K / bk});
    vt::QuantFp8Group(q, a_fp8.t(), a_scale.t(), x.t(), static_cast<int>(bk));
    std::vector<uint8_t> a_bytes(static_cast<size_t>(M * K));
    std::memcpy(a_bytes.data(), a_fp8.t().data, a_bytes.size());
    std::vector<float> a_s(static_cast<size_t>(M * (K / bk)));
    std::memcpy(a_s.data(), a_scale.t().data, a_s.size() * 4);
    const std::vector<double> ref = RefBlockGemm(a_bytes, a_s, M, K, w);
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double denom = std::max(std::abs(ref[i]), 1e-6);
      worst = std::max(worst, std::abs(got[i] - ref[i]) / denom);
    }
    // f32 accumulation against a double reference: two orders tighter than the
    // bf16 store above, which is the whole point of asserting both.
    CHECK(worst < 5e-5);
  }
}

// G3 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: ModelRegistry::Forward REACHES the block arm at every projection") {
  // The reachability case (`.agents/reachability.md`). It enters at
  // `ModelRegistry::Forward` — a production entry point — and not at the linear
  // method, so deleting a production call site in the dense forward reds it.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/true);

  uint64_t dispatched = 0;
  const std::vector<float> logits = RegistryForward(c, w, &dispatched);

  // The instrument a token gate cannot fake: the forward dispatched exactly one
  // block-scaled GEMM per quantized projection. A count of 0 means nothing
  // reached the arm; a count below 13 means a projection silently took another.
  CHECK(dispatched == static_cast<uint64_t>(kBlockProjectionsPerForward));
  CHECK(AllFinite(logits));
  CHECK(CountNonZero(logits) == static_cast<int64_t>(logits.size()));

  // The f32 KV cache (`VT_KV_CACHE_F32=1`) is NOT exercised here and cannot be:
  // this arm emits a bf16 V at `v_proj`, which is upstream's `out_dtype` and the
  // model dtype, and `vt::ReshapeAndCache`'s auto path requires one shared
  // dtype exactly as upstream's `reshape_and_cache_flash` does. That refusal
  // hits every bf16 arm in the tree rather than this row's, so it is issue
  // #1249 and the row's spec lists it under `## Owed` instead of being widened
  // into M4.
}

// G4 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: the weight stays fp8 bytes plus a scale grid, not a bf16 expansion") {
  // A silent dequant to bf16 is numerically BETTER than the quantized path and
  // therefore invisible to every value comparison in this file. Only the byte
  // counts can see it.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/true);
  const Fp8BlockWeight& q = w.layers[1].attn.q_proj_fp8_block;
  const int64_t N = q.n, K = q.k;
  REQUIRE(N == 256);
  REQUIRE(K == 128);

  CHECK_MESSAGE(q.packed.bytes.size() == static_cast<size_t>(N * K),
                "one fp8 byte per element; a bf16 expansion would be "
                    << (2 * N * K));
  CHECK(q.packed.dtype == DType::kI8);
  CHECK(q.scale.dtype == DType::kF32);
  CHECK(q.scale.bytes.size() ==
        static_cast<size_t>(CDiv(N, kBlock) * CDiv(K, kBlock)) * 4);
  // The ragged axis: `k_proj` is 64 wide against a 128 block, so its scale grid
  // has ONE row. A floor tiling would give zero and index out of the grid.
  const Fp8BlockWeight& k = w.layers[1].attn.k_proj_fp8_block;
  CHECK(k.n == 64);
  CHECK(k.scale.shape[0] == 1);
  CHECK(k.scale.bytes.size() == static_cast<size_t>(1 * CDiv(k.k, kBlock)) * 4);

  // And the resident upload keeps both, at the same widths, on the device the
  // GEMM reads them from.
  vt::Queue qq = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, qq};
  const vt::Tensor packed = block_gemm::ResidentFp8BlockPacked(d, q);
  const vt::Tensor scale = block_gemm::ResidentFp8BlockScale(d, q);
  CHECK(packed.dtype == DType::kI8);
  CHECK(packed.shape[0] == N);
  CHECK(packed.shape[1] == K);
  CHECK(scale.dtype == DType::kF32);
  CHECK(scale.shape[0] == CDiv(N, kBlock));
  CHECK(scale.shape[1] == CDiv(K, kBlock));
}

// G5 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: ONE perturbed block scale moves the model's logits") {
  // The issue's gate design, point 2: a token gate cannot see a scale
  // perturbation (`tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp:43-45`
  // records x1.02 and x1.10 both producing 16/16 identical tokens). This is the
  // instrument that can. The bumped element is row 1, column 0 of ONE
  // projection's grid, so a per-tensor collapse to (0,0), an epilogue-folded
  // alpha and a transposed scale index are each blind to it.
  const HfConfig c = BlockConfig();
  const std::vector<float> base =
      RegistryForward(c, BlockWeights(c, true, 1.0F), nullptr);
  const std::vector<float> bumped =
      RegistryForward(c, BlockWeights(c, true, 1.10F), nullptr);
  REQUIRE(base.size() == bumped.size());

  double worst = 0.0;
  for (size_t i = 0; i < base.size(); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(base[i]) - bumped[i]));
  CHECK_MESSAGE(worst > 1e-4,
                "a x1.10 bump on one 128x128 block moved the logits by "
                    << worst);

  // The control: the SAME construction with no bump is bit-reproducible, so the
  // difference above is the scale and not run-to-run noise.
  const std::vector<float> again =
      RegistryForward(c, BlockWeights(c, true, 1.0F), nullptr);
  CHECK(std::memcmp(base.data(), again.data(), base.size() * 4) == 0);
}

// G6 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: a device with no block-scaled GEMM refuses at Prepare, by name") {
  // M3 refused every LOADED block weight because nothing could read one. That
  // is narrowed, not deleted: the refusal now asks the PREPARE queue's device
  // whether it has a kernel. CUDA does not until #1189 M5, and a user finding
  // that out at the first GEMM — after a graph capture — is the failure this
  // prevents.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/true);

  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, vt::DeviceType::kCPU));
  REQUIRE_FALSE(
      vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, vt::DeviceType::kCUDA));

  std::string message;
  try {
    vllm::RefuseUnrunnableQwen3_5DenseFp8Block(w, vt::DeviceType::kCUDA);
  } catch (const std::exception& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("block-wise") != std::string::npos);
  CHECK(message.find("in_proj_qkv") != std::string::npos);
  CHECK(message.find("cuda") != std::string::npos);
  CHECK(message.find("1189") != std::string::npos);

  // On a device that HAS the kernel it is inert. Without this the case passes
  // for a refusal that never stopped refusing.
  CHECK_NOTHROW(
      vllm::RefuseUnrunnableQwen3_5DenseFp8Block(w, vt::DeviceType::kCPU));
  // And a checkpoint with no block weight is inert on every device.
  const Qwen3_5DenseWeights plain = BlockWeights(c, /*block_arm=*/false);
  CHECK_NOTHROW(
      vllm::RefuseUnrunnableQwen3_5DenseFp8Block(plain, vt::DeviceType::kCUDA));

  // Through the PRODUCTION call site, not just the function: deleting the call
  // in `PrepareQwen3_5Dense` must red this. The queue names a CUDA device and
  // carries no handle, which is safe precisely BECAUSE the refusal runs first
  // in that function — before `PrepareLmHeadResident` or anything else touches
  // a backend. If the refusal is ever moved below them, this stops being a
  // clean throw and says so.
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  vt::Queue cuda_q{vt::Device{vt::DeviceType::kCUDA, 0}, nullptr};
  std::string routed;
  try {
    ModelRegistry::Prepare(*model, c, cuda_q);
  } catch (const std::exception& e) {
    routed = e.what();
  }
  REQUIRE_FALSE(routed.empty());
  CHECK(routed.find("block-wise") != std::string::npos);
  CHECK(routed.find("cuda") != std::string::npos);
  CHECK(routed.find("1189") != std::string::npos);

  // The same call site on a device that CAN run it does not refuse.
  std::unique_ptr<vllm::LoadedModel> cpu_model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  vt::Queue cpu_q = Q();
  CHECK_NOTHROW(ModelRegistry::Prepare(*cpu_model, c, cpu_q));
}

// G7 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: the bf16 arm is unchanged and dispatches no block GEMM") {
  // Without this the gate passes for a forward that routed EVERY projection
  // through the block arm.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/false);
  uint64_t dispatched = 0;
  const std::vector<float> logits = RegistryForward(c, w, &dispatched);
  CHECK(dispatched == 0);
  CHECK(AllFinite(logits));
  CHECK(CountNonZero(logits) == static_cast<int64_t>(logits.size()));
}

// G8 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: the body refuses what it cannot compute, by name") {
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  SUBCASE("a K the activation quant cannot group") {
    // Upstream asserts the same divisibility (`fp8_utils.py:596-599`). The
    // refusal is the BODY's, so it names the shape and the block rather than
    // surfacing from inside vt one frame deeper.
    const int64_t M = 2, K = 192, N = 128;
    const Fp8BlockWeight w = MakeFp8Block(N, K, kBlock, kBlock, 9);
    vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K});
    auto method = layers::MakeLinearMethod(OwnedTensor{}, w);
    CHECK_THROWS_WITH_AS(method->Apply(d, x.t(), DType::kBF16),
                         doctest::Contains("192"), std::runtime_error);
  }

  SUBCASE("an activation whose K disagrees with the weight's") {
    const int64_t M = 2, K = 256, N = 128;
    const Fp8BlockWeight w = MakeFp8Block(N, 128, kBlock, kBlock, 9);
    vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K});
    auto method = layers::MakeLinearMethod(OwnedTensor{}, w);
    CHECK_THROWS_AS(method->Apply(d, x.t(), DType::kBF16), std::runtime_error);
  }
}
