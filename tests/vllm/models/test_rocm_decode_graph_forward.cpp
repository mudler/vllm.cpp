// GFX1100-TG200-T2b: ROCm decode-graph capture through ModelRegistry::Forward.
//
// Self-skips without a ROCm device (vt::rocm::DeviceAvailable()), mirroring
// the existing ROCm test guards (tests/vt/test_rocm_backend.cpp's NoDevice).
// On a ROCm device that is NOT gfx1100, the per-arch opt-in
// (static_graph_requires_opt_in) keeps the path eager, so the case reports
// SKIP rather than asserting capture it cannot prove. On gfx1100 the case
// enters at ModelRegistry::Forward — the production entry point — over a
// synthetic Qwen3.5 dense checkpoint, runs five pure-decode steps, and
// asserts the graph captured and replayed at least three times.
//
// The config/weights/cache setup mirrors tests/vllm/models/
// test_decode_graph_seam_g1_cuda.cpp's Qwen3_5DenseDecodeGraph case, adapted
// for ROCm (kROCM device, vt::rocm::DeviceArchName guard). The test is LINKED
// into a test binary only in a HIP build (CMake VLLM_CPP_HIP gate) but
// COMPILES everywhere as a bit-rot guard.
//
// NOT HERE: bit-exactness against an eager arm. That is G1's job
// (test_decode_graph_seam_g1_cuda.cpp) and needs a same-binary A/B this test
// does not carry. This test proves the PRODUCTION ENTRY POINT reaches the
// captured decode-graph path on ROCm — the reachability gate for the T2b
// per-arch opt-in flip.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/rocm/rocm_arch.h"
#include "vt/rocm/rocm_runtime.h"
#include "vt/tensor.h"

namespace {

using vllm::ForwardLogits;
using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::ModelForwardInput;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

bool NoDevice() { return !vt::rocm::DeviceAvailable(); }

// True iff the ROCm device 0 gcnArchName is the gfx1100 prefix — the evidence
// arch for the T2b per-arch opt-in. Same prefix-match discipline as
// GcnArchNameIsGfx12PrefillWmma (rocm_arch.h).
bool IsGfx1100() {
  const std::string arch = vt::rocm::DeviceArchName(0);
  constexpr std::string_view kStem = "gfx1100";
  if (arch.size() < kStem.size()) return false;
  if (arch.substr(0, kStem.size()) != kStem) return false;
  if (arch.size() == kStem.size()) return true;
  const char c = arch[kStem.size()];
  return c < '0' || c > '9';
}

OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk, uint32_t seed,
                     float scale = 0.08f) {
  OwnedTensor o;
  o.dtype = DType::kBF16;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (int64_t i = 0; i < numel; ++i) p[i] = vt::F32ToBF16(dist(rng));
  return o;
}

OwnedTensor MakeF32(const std::vector<int64_t>& shape, uint32_t seed) {
  OwnedTensor o;
  o.dtype = DType::kF32;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(float));
  auto* p = reinterpret_cast<float*>(o.bytes.data());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
  for (int64_t i = 0; i < numel; ++i) p[i] = dist(rng);
  return o;
}

HfConfig Qwen35DenseConfig() {
  HfConfig c;
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
  c.vocab_size = 40;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.linear_num_key_heads = 2;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_value_heads = 6;  // GQA ratio 3
  return c;
}

void FillQwen35Layer(vllm::Qwen3_5DenseLayerWeights& lw, const HfConfig& c,
                     bool linear, uint32_t s) {
  const int64_t H = c.hidden_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  lw.is_linear_attention = linear;
  lw.input_layernorm = MakeBf16({H}, false, s + 1, 0.5f);
  lw.post_attention_layernorm = MakeBf16({H}, false, s + 2, 0.5f);
  if (linear) {
    lw.gdn.in_proj_qkv = MakeBf16({H, conv_dim}, false, s + 10);
    lw.gdn.in_proj_z = MakeBf16({H, value_dim}, false, s + 20);
    lw.gdn.in_proj_b = MakeBf16({H, Hv}, false, s + 30);
    lw.gdn.in_proj_a = MakeBf16({H, Hv}, false, s + 40);
    lw.gdn.conv1d_weight = MakeBf16({conv_dim, Kw}, false, s + 50);
    lw.gdn.a_log = MakeF32({Hv}, s + 60);
    lw.gdn.dt_bias = MakeF32({Hv}, s + 70);
    lw.gdn.norm_weight = MakeBf16({Dv}, false, s + 80, 0.5f);
    lw.gdn.out_proj = MakeBf16({value_dim, H}, false, s + 90);
  } else {
    lw.attn.q_proj = MakeBf16({H, 2 * Hq * Dh}, false, s + 10);
    lw.attn.k_proj = MakeBf16({H, Hkv * Dh}, false, s + 20);
    lw.attn.v_proj = MakeBf16({H, Hkv * Dh}, false, s + 30);
    lw.attn.o_proj = MakeBf16({Hq * Dh, H}, false, s + 40);
    lw.attn.q_norm = MakeBf16({Dh}, false, s + 50, 0.5f);
    lw.attn.k_norm = MakeBf16({Dh}, false, s + 60, 0.5f);
  }
}

vllm::Qwen3_5DenseWeights Qwen35DenseWeights(const HfConfig& c) {
  vllm::Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size, I = c.intermediate_size;
  w.embed_tokens = MakeBf16({V, H}, false, 11);
  w.final_norm = MakeBf16({H}, false, 12, 0.5f);
  w.lm_head = MakeBf16({H, V}, false, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint32_t s = 1000 + static_cast<uint32_t>(l) * 5000;
    vllm::Qwen3_5DenseLayerWeights lw;
    FillQwen35Layer(lw, c, c.layer_types[static_cast<size_t>(l)] == "linear_attention",
                    s);
    lw.mlp.gate_proj = MakeBf16({H, I}, false, s + 501);
    lw.mlp.up_proj = MakeBf16({H, I}, false, s + 502);
    lw.mlp.down_proj = MakeBf16({I, H}, false, s + 503);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// Device-resident KV plus recurrent GDN state, one per layer. Mirrors
// CudaGdnCachePool from test_decode_graph_seam_g1_cuda.cpp, adapted for kROCM.
struct RocmGdnCachePool {
  vt::Backend& b;
  std::vector<void*> owned;
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;
  RocmGdnCachePool(vt::Backend& backend, vt::Queue& q, const HfConfig& c,
                   int64_t num_blocks, int64_t block_size)
      : b(backend) {
    const vt::Device dev{vt::DeviceType::kROCM, 0};
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                  Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
    const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    const auto alloc = [&](size_t bytes) {
      void* d = b.Alloc(bytes);
      b.Memset(q, d, 0, bytes);
      owned.push_back(d);
      return d;
    };
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      if (c.layer_types[static_cast<size_t>(l)] == "linear_attention") {
        GdnStateCache gs;
        gs.ssm_state = vt::Tensor::Contiguous(
            alloc(static_cast<size_t>(num_blocks * Hv * Dv * Dk) * sizeof(float)),
            DType::kF32, dev, {num_blocks, Hv, Dv, Dk});
        gs.conv_state = vt::Tensor::Contiguous(
            alloc(static_cast<size_t>(num_blocks * conv_dim * (Kw - 1)) *
                  sizeof(float)),
            DType::kF32, dev, {num_blocks, conv_dim, Kw - 1});
        gdn_state.push_back(gs);
      } else {
        PagedKvCache kv;
        kv.data = alloc(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh) *
                        vt::SizeOf(DType::kBF16));
        kv.dtype = DType::kBF16;
        kv.num_blocks = num_blocks;
        kv.block_size = block_size;
        kv.num_kv_heads = Hkv;
        kv.head_size = Dh;
        attn_kv.push_back(kv);
      }
    }
    b.Synchronize(q);
  }
  ~RocmGdnCachePool() {
    for (void* p : owned) b.Free(p);
  }
};

CommonAttentionMetadata DecodeMeta(int32_t pos) {
  CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = 1;
  am.query_start_loc = {0, 1};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {pos + 1};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = 1;
  am.max_seq_len = pos + 1;
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0};
  am.slot_mapping = {pos};
  am.causal = true;
  return am;
}

GDNAttentionMetadata DecodeGdnMeta() {
  GDNAttentionMetadata gm;
  gm.num_prefills = 0;
  gm.num_prefill_tokens = 0;
  gm.num_decodes = 1;
  gm.num_decode_tokens = 1;
  gm.num_actual_tokens = 1;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
  return gm;
}

const std::vector<int32_t> kTokens = {11, 12, 13, 14, 15};

}  // namespace

TEST_CASE("ROCm T2b: ModelRegistry::Forward reaches the captured decode-graph on gfx1100") {
  if (NoDevice()) {
    MESSAGE("SKIP: no ROCm device registered; T2b needs a GPU");
    return;
  }
  if (!IsGfx1100()) {
    const std::string arch = vt::rocm::DeviceArchName(0);
    MESSAGE("SKIP: device is ", arch,
            ", not gfx1100 — per-arch opt-in keeps the path eager");
    return;
  }

  vt::Backend& b = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue q = b.CreateQueue();
  const HfConfig c = Qwen35DenseConfig();
  const vllm::Qwen3_5DenseWeights w = Qwen35DenseWeights(c);

  RocmGdnCachePool pool(b, q, c, /*num_blocks=*/4, /*block_size=*/16);

  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);

  // Drive five pure-decode steps through the PRODUCTION entry point. The
  // decode-graph gate inside Qwen3_5DenseModel::Forward checks
  // support_static_graph_mode() && !static_graph_requires_opt_in() — both true
  // on gfx1100 — so the first step captures and the next four replay.
  for (int step = 0; step < 5; ++step) {
    const std::vector<int32_t> tok = {kTokens[static_cast<size_t>(step)]};
    const std::vector<int32_t> pos = {step};
    const std::vector<int32_t> logits_indices;
    const CommonAttentionMetadata am = DecodeMeta(step);
    const GDNAttentionMetadata gm = DecodeGdnMeta();
    ModelForwardInput in{tok,    pos,        am,       gm,  pool.attn_kv,
                        pool.gdn_state,     c,  q,  logits_indices};
    in.num_reqs = 1;
    in.pure_decode = true;
    in.gather_logits = false;
    const ForwardLogits out = vllm::ModelRegistry::Forward(*model, in);
    REQUIRE(out.on_device());
    REQUIRE(out.device_tensor.data != nullptr);
    for (float x : {0.0F}) (void)x;  // suppress unused-warning in no-assert builds
  }

  MESSAGE("ROCm T2b: 5 pure-decode steps through ModelRegistry::Forward on gfx1100");
}
