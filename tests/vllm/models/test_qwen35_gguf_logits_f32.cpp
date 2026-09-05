// #2534 — the GGUF keep-quant lm_head must emit logits with f32 RESOLUTION.
//
// The Q4_K_M token gate against llama.cpp `b10451` loses six near-ties whose
// margins are 0.027 to 0.178 logits. The measured cause is not error magnitude
// (our per-step delta sits INSIDE the oracle's own kernel-schedule noise band)
// but RESOLUTION: 288 of 288 of our top-1 logits landed exactly on the bf16
// grid, whose ULP is 0.125 at the magnitude 16-32 those logits carry, so five
// of the six contested gaps are below the smallest non-zero difference we can
// represent and six steps are EXACT TIES in our arithmetic
// (docs/bench-evidence/qwen38-27b-q4km-logit-delta-20260902.md).
//
// A block-quantized head reached `MatmulBf16LogitsF32D` only because
// `OwnGgufQuantBlocks` sets `nk = true` for a LAYOUT reason, so it inherited a
// rule written for the tied BF16 torch-Linear head. These cases pin the routing
// that fixes it, in the property the measurement named.
//
// WHAT IS COMPARED, in words, because a grid test that only looks at one arm
// cannot tell "we widened the head" from "these numbers happen to be coarse":
// every case runs TWO heads over the SAME model and the SAME tokens, differing
// only in the head's storage — a block-quant head (Q4_K, and Q6_K, which is what
// a Q4_K_M file stores `output.weight` as) against a BF16 head, both `nk = true`
// so both took the bf16 helper before this change. The block-quant head's logits
// must be OFF the bf16 grid; the BF16 head's must stay ON it, because vLLM's
// model-dtype rule still governs that arm and this change must not move it.
//
// Synthetic weights, CPU only, no checkpoint and no GPU.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::ForwardLogits;
using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::ModelForwardInput;
using vllm::ModelRegistry;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseModel;
using vllm::Qwen3_5DenseWeights;
using vllm::DenseMlpWeights;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

namespace {

// ggml type ids (ggml/include/ggml.h:390-432), the ids the GGUF header carries.
constexpr uint32_t kGgmlQ4_K = 12, kGgmlQ6_K = 14;

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

float RandV(uint64_t seed) {
  const double u = static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

// The BF16 head in the GGUF `expand_nk` / tied-torch-Linear orientation: the
// file's own [N = vocab, K = H] order with nk = true. This is the arm vLLM's
// model-dtype rule governs and the arm this change must NOT move.
OwnedTensor MakeBf16HeadNK(int64_t vocab, int64_t H, uint64_t seed) {
  OwnedTensor t = MakeOwned(DType::kBF16, {vocab, H}, seed);
  t.nk = true;
  return t;
}

// xorshift32 — deterministic across platforms and compilers, so a failure is
// reproducible from the seed alone.
uint32_t NextRand(uint32_t* s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

// The GGUF KEEP-QUANT head: raw ggml block payload in the file's own [N, K]
// order with nk = true, which is exactly what
// `qwen3_5_gguf_weights.cpp::OwnGgufQuantBlocks` produces from a mapped GGUF.
//
// The payload is RANDOM BYTES with the f16 exponent MSB cleared on every odd
// byte, the generator `tests/vllm/test_gguf_keep_quant.cpp` already uses for the
// same encodings. Nothing in this tree ENCODES a k-quant — `from_float` is
// populated only for the two activation encodings Q8_0 and Q8_K
// (`include/vt/quant.h`, `src/vt/cpu/cpu_quant_traits.cpp`) — because llama.cpp
// quantizes offline and we only ever read. `CheckFinite` below asserts the bytes
// decoded to real numbers, because a resolution gate over NaN would prove
// nothing.
OwnedTensor MakeKeptQuantHead(DType q, uint32_t ggml_type, int64_t vocab,
                              int64_t H, uint32_t seed) {
  const size_t row_bytes = vt::RowSizeBytes(q, H);
  OwnedTensor t;
  t.dtype = q;
  t.rank = 2;
  t.shape[0] = vocab;
  t.shape[1] = H;
  t.nk = true;
  t.bytes.resize(row_bytes * static_cast<size_t>(vocab));
  uint32_t s = seed | 1U;
  for (size_t i = 0; i < t.bytes.size(); ++i) {
    auto b = static_cast<uint8_t>(NextRand(&s) & 0xFFU);
    if ((i % 2) == 1) b &= 0xBFU;  // clear the f16 exponent MSB -> finite
    t.bytes.data()[i] = b;
  }
  const std::vector<float> decoded = vllm::DequantGgufRowToF32(
      ggml_type, t.bytes.data(), vocab * H);
  REQUIRE(decoded.size() == static_cast<size_t>(vocab * H));
  for (float v : decoded) REQUIRE(std::isfinite(v));
  return t;
}

// A value is ON the bf16 grid when a bf16 round trip returns it unchanged. Every
// logit a bf16-output GEMM produces is on it by construction, whatever the f32
// container it was later widened into.
bool OnBf16Grid(float v) { return vt::BF16ToF32(vt::F32ToBF16(v)) == v; }

int64_t CountOffGrid(const float* p, size_t n) {
  int64_t off = 0;
  for (size_t i = 0; i < n; ++i)
    if (!OnBf16Grid(p[i])) ++off;
  return off;
}

// H = 256 so a K-quant super-block (256 elements) divides K exactly, which is
// what makes a Q4_K / Q6_K head representable at all. Otherwise the 27B-shaped
// small dense config: layer_types [LA, LA, LA, FA], no experts, GQA ratio 3.
HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 256;
  c.num_hidden_layers = 4;
  c.vocab_size = 40;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  return c;
}

DenseMlpWeights MakeMlp(const HfConfig& c, uint64_t s) {
  DenseMlpWeights m;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  m.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 1);
  m.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 2);
  m.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 3);
  return m;
}

// Everything except the head; the caller installs the head under test so the two
// arms differ in exactly one tensor.
Qwen3_5DenseWeights MakeTrunk(const HfConfig& c) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention = (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.mlp = MakeMlp(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

struct CachePool {
  const HfConfig& c;
  int64_t num_blocks;
  int64_t block_size;
  std::vector<std::vector<float>> full_attn_buf;
  std::vector<std::vector<float>> gdn_ssm_buf;
  std::vector<std::vector<float>> gdn_conv_buf;
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;

  CachePool(const HfConfig& cfg, int64_t nb, int64_t bs)
      : c(cfg), num_blocks(nb), block_size(bs) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                  Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
    const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      if (c.layer_types[static_cast<size_t>(l)] == "linear_attention") {
        gdn_ssm_buf.emplace_back(static_cast<size_t>(nb * Hv * Dv * Dk), 0.0f);
        gdn_conv_buf.emplace_back(static_cast<size_t>(nb * conv_dim * (Kw - 1)), 0.0f);
      } else {
        full_attn_buf.emplace_back(static_cast<size_t>(nb * 2 * bs * Hkv * Dh), 0.0f);
      }
    }
    const vt::Device cpu{vt::DeviceType::kCPU, 0};
    for (auto& b : full_attn_buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
    for (size_t g = 0; g < gdn_ssm_buf.size(); ++g) {
      GdnStateCache gs;
      gs.ssm_state = vt::Tensor::Contiguous(gdn_ssm_buf[g].data(), DType::kF32, cpu,
                                            {num_blocks, Hv, Dv, Dk});
      gs.conv_state = vt::Tensor::Contiguous(gdn_conv_buf[g].data(), DType::kF32, cpu,
                                             {num_blocks, conv_dim, Kw - 1});
      gdn_state.push_back(gs);
    }
  }
};

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

CommonAttentionMetadata PrefillAttnMeta(int64_t T, const std::vector<int32_t>& blocks,
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
  const auto conv = vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}

const std::vector<int32_t> kIds = {5, 9, 2, 31, 17, 3};
const std::vector<int32_t> kPos = {0, 1, 2, 3, 4, 5};

// The PRODUCTION entry point: the type-erased registry forward, on its default
// configuration. A head that only the reference ForwardDense could reach would
// prove the class works and not that anything reaches it.
std::vector<float> RegistryLogits(const HfConfig& c, const Qwen3_5DenseWeights& w) {
  CachePool pool(c, /*num_blocks=*/8, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillAttnMeta(6, {0, 1}, 8);
  const GDNAttentionMetadata gm = PrefillGdnMeta(6);
  vt::Queue q = Q();
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3_5DenseLoadedModel(w);
  const std::vector<int32_t> logits_indices;
  ModelForwardInput in{kIds, kPos, am, gm, pool.attn_kv, pool.gdn_state, c, q,
                       logits_indices};
  in.num_reqs = 1;
  const ForwardLogits out = ModelRegistry::Forward(*model, in);
  // The default forward returns DEVICE logits (VT_LOGITS_GATHER). On the CPU
  // backend that buffer is host memory, so it is read in place; the host arm is
  // handled too so this helper does not depend on which one the default takes.
  const size_t n = static_cast<size_t>(6) * static_cast<size_t>(c.vocab_size);
  if (out.on_device()) {
    REQUIRE(out.rows == 6);
    REQUIRE(out.vocab == c.vocab_size);
    REQUIRE(out.device_tensor.dtype == DType::kF32);
    REQUIRE(out.device_tensor.data != nullptr);
    const auto* p = static_cast<const float*>(out.device_tensor.data);
    return std::vector<float>(p, p + n);
  }
  REQUIRE(out.host.size() == n);
  return out.host;
}

// Both arms of one case, reported so the comparison is readable in the output
// rather than only in this source file.
void CheckHeadResolution(DType q, uint32_t ggml_type, const char* label) {
  const HfConfig c = MakeConfig();
  const int64_t H = c.hidden_size, V = c.vocab_size;

  Qwen3_5DenseWeights kept = MakeTrunk(c);
  kept.lm_head = MakeKeptQuantHead(q, ggml_type, V, H, 4242);
  REQUIRE(vt::IsBlockQuant(kept.lm_head.dtype));
  REQUIRE(kept.lm_head.nk);

  Qwen3_5DenseWeights bf16 = MakeTrunk(c);
  bf16.lm_head = MakeBf16HeadNK(V, H, 4242);
  REQUIRE_FALSE(vt::IsBlockQuant(bf16.lm_head.dtype));
  REQUIRE(bf16.lm_head.nk);

  const std::vector<float> kept_logits = RegistryLogits(c, kept);
  const std::vector<float> bf16_logits = RegistryLogits(c, bf16);
  const size_t n = kept_logits.size();
  REQUIRE(n == bf16_logits.size());

  const int64_t kept_off = CountOffGrid(kept_logits.data(), n);
  const int64_t bf16_off = CountOffGrid(bf16_logits.data(), n);
  // std::string, not the bare `const char*`: doctest stringifies a char pointer
  // as a BOOL, which prints "head=1" and loses the label the report needs.
  INFO("head=" << std::string(label) << " compared a BLOCK-QUANT head against a BF16 head "
                            "over the same trunk and the same tokens; logits off "
                            "the bf16 grid: block-quant "
                << kept_off << "/" << n << ", bf16 " << bf16_off << "/" << n);

  // The answer must be a real GEMM before "off the grid" means anything: an
  // all-zero logit vector is on the grid and would read as a clean FAIL for the
  // wrong reason.
  bool nonzero = false;
  for (size_t i = 0; i < n; ++i) nonzero = nonzero || kept_logits[i] != 0.0F;
  REQUIRE(nonzero);

  // THE ASSERTION (#2534). A bf16-output GEMM puts every logit on the grid by
  // construction, so any appreciable off-grid count proves the head wrote f32.
  // The bound is 3/4 rather than all of them because a genuine f32 product may
  // land on the grid by chance at a rate of about 2^-16.
  CHECK(kept_off * 4 >= static_cast<int64_t>(n) * 3);

  // The CONTROL, and the reason this is a comparison and not a coarse-number
  // check: the BF16 head still follows vLLM's model-dtype rule
  // (logits_processor.py:99-136), so its logits stay entirely on the grid.
  CHECK(bf16_off == 0);
}

}  // namespace

TEST_CASE("qwen3_5 gguf head: a Q4_K keep-quant lm_head emits f32-resolution logits") {
  CheckHeadResolution(DType::kQ4_K, kGgmlQ4_K, "Q4_K");
}

// Q6_K because that is what a Q4_K_M file stores `output.weight` as: the gate's
// own artifact is Q4_K 294 / Q6_K 67 / Q5_K 48 / Q8_0 1.
TEST_CASE("qwen3_5 gguf head: a Q6_K keep-quant lm_head emits f32-resolution logits") {
  CheckHeadResolution(DType::kQ6_K, kGgmlQ6_K, "Q6_K");
}

// The eager reference forward is the OTHER call site of DenseLogitsF32D
// (ForwardDense), and it must not keep the old rounding after the paged arms
// lose it — the routing lives in one function precisely so it cannot.
TEST_CASE("qwen3_5 gguf head: the eager dense forward widens the same head") {
  const HfConfig c = MakeConfig();
  Qwen3_5DenseWeights w = MakeTrunk(c);
  w.lm_head = MakeKeptQuantHead(DType::kQ4_K, kGgmlQ4_K, c.vocab_size, c.hidden_size, 4242);
  vt::Queue q = Q();
  const std::vector<float> logits = Qwen3_5DenseModel::ForwardDense(kIds, kPos, w, c, q);
  REQUIRE(logits.size() == static_cast<size_t>(kIds.size()) * c.vocab_size);
  const int64_t off = CountOffGrid(logits.data(), logits.size());
  INFO("ForwardDense with a Q4_K head: " << off << "/" << logits.size()
                                         << " logits off the bf16 grid");
  CHECK(off * 4 >= static_cast<int64_t>(logits.size()) * 3);
}
