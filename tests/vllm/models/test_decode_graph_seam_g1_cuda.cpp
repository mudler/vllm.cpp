// G1 FOR ENG-CUDAGRAPH-BREAK — bit-exactness against eager on a REAL DEVICE,
// over MORE than one replay, for every driver migrated onto the seam.
// Row `ENG-CUDAGRAPH-BREAK`, W3 #1291, parent #1163, spec `## Gates` G1.
//
// WHY THIS FILE EXISTS AND THE SIBLING GATES DO NOT COVER IT. Every other gate
// this row owns runs on the CPU, through a backend that implements the capture
// vocabulary by LOGGING. A CPU kernel is a direct function call and not a
// backend submission, so nothing a CPU harness calls a "replay" recomputes
// anything: those gates hold the ROUTING and the capture step's numerics, and
// they cannot hold that a REPLAYED graph reproduces the eager forward. W1 and
// W2 both landed with G1 recorded as owed for exactly that reason. This is the
// gate that needs a device.
//
// AND IT NEEDS MORE THAN ONE REPLAY, which is the spec's own wording. A single
// replay cannot distinguish a correct capture from one that happens to read a
// buffer nothing has overwritten yet; the defect class this row's own history
// contains (`.agents/specs/decode-graph-scratch-uaf-2026-07-18.md`) appears on a
// LATER replay. Each driver below runs COLD, CAPTURE, then THREE consecutive
// replays, and every one of the five steps is compared.
//
// THE EAGER ARM IS THE DRIVER'S OWN, and it is selected without an environment
// variable. `PadToCaptureSize(b, max_num_seqs)` returns -1 when `b` exceeds
// `max_num_seqs` (`decode_graph_sizes.h:47-54`), and each migrated `Step` falls
// out to its plain forward on that value. So a driver constructed with
// `max_num_reqs == 0` is the SAME code compiled the same way, on the same
// device, taking the eager path — a same-binary A/B rather than two builds. The
// alternative, `VLLM_CPP_CUDAGRAPH=0`, is read once per process into a
// function-local static and would need a second process; the per-model switches
// (`VT_QWEN3MOE_CUDAGRAPH`, `VT_DEEPSEEK_CUDAGRAPH`) exist for two of the three
// drivers and not for the third, so neither is uniform.
//
// EACH ARM GETS ITS OWN KV CACHE, because the arms are stepped in lockstep and a
// shared cache would let one arm read the other's writes and call the agreement
// a result.
//
// SKIPS WITHOUT A CUDA BACKEND, so this file is inert on a CPU-only box and on
// continuous integration, and it is the one gate in this row that MUST be run on
// a leased device before its result may be recorded.
//
// READ ITS COUNT, NOT ITS STATUS. On a box with no CUDA backend this file prints
// `Status: SUCCESS!` over `assertions: 0`, which is a SKIP wearing a pass. A G1
// result from this file is admissible only with a NON-ZERO assertion count and
// the device it ran on named beside it.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_moe.h"
#include "vllm/model_executor/models/voxtral.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace fs = std::filesystem;

namespace {

using vllm::DeepseekV2Params;
using vllm::DeepseekV2Weights;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3DenseWeights;
using vllm::Qwen3MoeWeights;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

bool HasCuda() {
  try {
    vt::GetBackend(vt::DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
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

// One single-request pure-decode step at `pos`.
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

// Device-resident, zero-initialised full-attention paged KV caches, one per
// layer. The CUDA attention path writes into the cache ON DEVICE, so a host
// vector is not a cache here.
struct CudaDenseCachePool {
  vt::Backend& b;
  std::vector<void*> owned;
  std::vector<PagedKvCache> attn_kv;
  CudaDenseCachePool(vt::Backend& backend, vt::Queue& q, int64_t layers, int64_t Hkv,
                     int64_t Dh, int64_t num_blocks, int64_t block_size)
      : b(backend) {
    const size_t bytes = static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh) *
                         vt::SizeOf(DType::kBF16);
    for (int64_t l = 0; l < layers; ++l) {
      void* d = b.Alloc(bytes);
      b.Memset(q, d, 0, bytes);
      owned.push_back(d);
      PagedKvCache kv;
      kv.data = d;
      kv.dtype = DType::kBF16;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
    b.Synchronize(q);
  }
  ~CudaDenseCachePool() {
    for (void* p : owned) b.Free(p);
  }
};

// The MLA sibling: one latent cache per layer, num_kv_heads 1, no separate V.
struct CudaMlaCachePool {
  vt::Backend& b;
  std::vector<void*> owned;
  std::vector<PagedKvCache> attn_kv;
  CudaMlaCachePool(vt::Backend& backend, vt::Queue& q, const DeepseekV2Params& p,
                   int64_t num_blocks, int64_t block_size)
      : b(backend) {
    const int64_t head_size = p.mla.head_size();
    const size_t bytes = static_cast<size_t>(num_blocks * block_size * head_size) *
                         vt::SizeOf(DType::kBF16);
    for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
      void* d = b.Alloc(bytes);
      b.Memset(q, d, 0, bytes);
      owned.push_back(d);
      PagedKvCache kv;
      kv.data = d;
      kv.dtype = DType::kBF16;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = 1;
      kv.head_size = head_size;
      attn_kv.push_back(kv);
    }
    b.Synchronize(q);
  }
  ~CudaMlaCachePool() {
    for (void* p : owned) b.Free(p);
  }
};

// Download the driver's [1, vocab] device logits.
std::vector<float> Read(vt::Backend& b, vt::Queue& q, const vllm::ForwardLogits& fl,
                        int64_t vocab) {
  REQUIRE(fl.on_device());
  REQUIRE(fl.device_tensor.data != nullptr);
  std::vector<float> h(static_cast<size_t>(vocab));
  b.Copy(q, h.data(), fl.device_tensor.data, h.size() * sizeof(float));
  b.Synchronize(q);
  return h;
}

// The five-step comparison every driver below runs. `step_kind` names each step
// so a failure says WHICH replay diverged rather than only that one did.
const char* kStepKind[5] = {"cold (eager pre-warm)", "capture", "replay 1", "replay 2",
                            "replay 3"};

size_t CompareStep(int step, const std::vector<float>& eager,
                   const std::vector<float>& graphed) {
  REQUIRE(eager.size() == graphed.size());
  size_t differing = 0;
  for (size_t i = 0; i < eager.size(); ++i)
    if (std::memcmp(&eager[i], &graphed[i], sizeof(float)) != 0) ++differing;
  CHECK_MESSAGE(differing == 0, "G1 step " << step << " (" << kStepKind[step]
                                           << "): " << differing << " of "
                                           << eager.size() << " logits differ");
  for (float x : graphed) REQUIRE(std::isfinite(x));
  return differing;
}

// ---- Qwen3-Coder MoE ------------------------------------------------------

HfConfig MoeConfig() {
  HfConfig c;
  c.num_hidden_layers = 2;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 16;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 10000000.0;
  c.vocab_size = 100;
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 32;
  c.shared_expert_intermediate_size = 0;
  return c;
}

Qwen3MoeWeights MoeWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, V = c.vocab_size;
  const int64_t E = c.num_experts, I = c.moe_intermediate_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  Qwen3MoeWeights w;
  w.tie_word_embeddings = false;
  w.attention_bias = false;
  w.embed_tokens = MakeBf16({V, H}, false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  w.lm_head = MakeBf16({H, V}, false, 3);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Qwen3MoeLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, true, seed++);
    lw.attn.q_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    lw.attn.k_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    lw.moe.router_gate = MakeBf16({H, E}, false, seed++);
    for (int64_t e = 0; e < E; ++e) {
      lw.moe.expert_gate.push_back(MakeBf16({H, I}, false, seed++));
      lw.moe.expert_up.push_back(MakeBf16({H, I}, false, seed++));
      lw.moe.expert_down.push_back(MakeBf16({I, H}, false, seed++));
    }
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// ---- Voxtral text (Mistral/Llama: no qk-norm, untied head) ----------------

HfConfig VoxConfig() {
  HfConfig c;
  c.num_hidden_layers = 2;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 16;
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 1000000.0;
  c.vocab_size = 100;
  return c;
}

Qwen3DenseWeights VoxWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  Qwen3DenseWeights w;
  w.tie_word_embeddings = false;
  w.attention_bias = false;
  w.embed_tokens = MakeBf16({V, H}, false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  w.lm_head = MakeBf16({H, V}, false, 3);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Qwen3DenseLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, true, seed++);
    lw.mlp.gate_up_proj = MakeBf16({2 * I, H}, true, seed++);
    lw.mlp.down_proj = MakeBf16({H, I}, true, seed++);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// ---- DeepSeek-V2 (MLA) -----------------------------------------------------
//
// The head geometry is DeepSeek-V2-Lite's REAL one (qk_nope 128 + qk_rope 64 =
// QK 192, V 128, kv_lora 512), because the CUDA MLA launcher is instantiated for
// head_dim 192 ONLY — the same constraint `test_deepseek_v2_forward.cpp`'s CUDA
// case records.
std::string WriteDsConfig() {
  const char* env = std::getenv("TMPDIR");
  const fs::path base = env != nullptr ? fs::path(env) : fs::temp_directory_path();
  const fs::path dir = base / "vllm_cpp_seam_g1";
  std::error_code ec;
  fs::create_directories(dir, ec);
  const std::string path = (dir / "deepseek.json").string();
  std::ofstream f(path);
  f << R"({
  "architectures": ["DeepseekV2ForCausalLM"],
  "model_type": "deepseek_v2",
  "hidden_size": 64,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 4,
  "vocab_size": 100,
  "intermediate_size": 32,
  "moe_intermediate_size": 16,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "n_group": 1,
  "topk_group": 1,
  "norm_topk_prob": false,
  "scoring_func": "softmax",
  "topk_method": "greedy",
  "routed_scaling_factor": 1.0,
  "moe_layer_freq": 1,
  "q_lora_rank": null,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,
  "max_position_embeddings": 128,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "rope_scaling": {
    "type": "yarn",
    "factor": 4,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
    "original_max_position_embeddings": 32
  },
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "kv_lora_rank": 512,
  "first_k_dense_replace": 1,
  "n_shared_experts": 2
})";
  f.close();
  return path;
}

vllm::DeepseekV2MlaWeights MakeMla(const DeepseekV2Params& p, uint32_t seed) {
  const vllm::mla::MlaBlockDims& d = p.mla;
  const int64_t H = d.hidden_size, N = d.num_heads, L = d.kv_lora_rank;
  const int64_t P = d.qk_nope_head_dim, R = d.qk_rope_head_dim, V = d.v_head_dim;
  const int64_t Dqk = d.qk_head_dim();
  vllm::DeepseekV2MlaWeights w;
  w.kv_a_proj_with_mqa = MakeBf16({L + R, H}, true, seed);
  w.q_proj = MakeBf16({N * Dqk, H}, true, seed + 1);
  w.kv_a_layernorm = MakeBf16({L}, false, seed + 2, 0.5f);
  w.kv_b_proj = MakeBf16({N * (P + V), L}, true, seed + 3);
  w.o_proj = MakeBf16({H, N * V}, true, seed + 4);
  const vllm::mla::AbsorbedKvBProj a = vllm::mla::AbsorbKvBProjBf16(
      reinterpret_cast<const uint16_t*>(w.kv_b_proj.bytes.data()), d);
  w.w_uk_t = MakeBf16({N, P, L}, false, 1, 0.0f);
  std::memcpy(w.w_uk_t.bytes.data(), a.w_uk_t.data(), a.w_uk_t.size() * sizeof(uint16_t));
  w.w_uv = MakeBf16({N, L, V}, false, 1, 0.0f);
  std::memcpy(w.w_uv.bytes.data(), a.w_uv.data(), a.w_uv.size() * sizeof(uint16_t));
  return w;
}

vllm::DeepseekV2DenseMlp MakeMlp(int64_t H, int64_t I, uint32_t seed) {
  vllm::DeepseekV2DenseMlp m;
  m.gate_up_proj = MakeBf16({2 * I, H}, true, seed);
  m.down_proj = MakeBf16({H, I}, true, seed + 1);
  return m;
}

DeepseekV2Weights DsWeights(const DeepseekV2Params& p) {
  const int64_t H = p.hidden_size, V = p.vocab_size;
  const int64_t E = p.n_routed_experts, I = p.moe_intermediate_size;
  DeepseekV2Weights w;
  w.params = p;
  w.embed_tokens = MakeBf16({V, H}, false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  w.lm_head = MakeBf16({H, V}, false, 3);
  {
    const int64_t rows = p.max_position_embeddings, rot = p.mla.qk_rope_head_dim;
    const std::vector<float> cache = vllm::mla::BuildDeepseekRopeCosSinCache(p.rope, rows);
    w.rope_cos_sin_cache = MakeBf16({rows, rot}, false, 1, 0.0f);
    auto* dst = reinterpret_cast<uint16_t*>(w.rope_cos_sin_cache.bytes.data());
    for (size_t i = 0; i < cache.size(); ++i) dst[i] = vt::F32ToBF16(cache[i]);
  }
  uint32_t seed = 100;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    vllm::DeepseekV2LayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn = MakeMla(p, seed);
    seed += 10;
    lw.is_moe = p.is_moe_layer(l);
    if (lw.is_moe) {
      lw.moe.router_gate = MakeBf16({H, E}, false, seed++);
      for (int64_t e = 0; e < E; ++e) {
        lw.moe.expert_gate.push_back(MakeBf16({H, I}, false, seed));
        lw.moe.expert_up.push_back(MakeBf16({H, I}, false, seed + 1));
        lw.moe.expert_down.push_back(MakeBf16({I, H}, false, seed + 2));
        seed += 3;
      }
      if (p.n_shared_experts > 0)
        lw.moe.shared = MakeMlp(H, p.shared_intermediate_size(), seed);
      seed += 2;
    } else {
      lw.dense = MakeMlp(H, p.intermediate_size, seed);
      seed += 2;
    }
    w.layers.push_back(std::move(lw));
  }
  return w;
}

const std::vector<int32_t> kTokens = {11, 12, 13, 14, 15};


// ---- Qwen3.5 (W4): the GDN-hybrid drivers ---------------------------------
//
// These two carry the persistent device input path, the auxiliary taps and the
// speculative-decode predicate, so they are the richest replay surface in the
// row and the one where a stale segment input would be least visible from a
// token count alone.

// Device-resident KV plus RECURRENT state. The GDN kernels read and write the
// ssm/conv state ON DEVICE across steps, so a host vector is not a cache here
// and — more to the point for G1 — each arm must own its own, or one arm's
// recurrence would advance on the other's writes and the agreement would mean
// nothing.
struct CudaGdnCachePool {
  vt::Backend& b;
  std::vector<void*> owned;
  std::vector<PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  CudaGdnCachePool(vt::Backend& backend, vt::Queue& q, const HfConfig& c,
                   int64_t num_blocks, int64_t block_size)
      : b(backend) {
    const vt::Device dev{vt::DeviceType::kCUDA, 0};
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
        vllm::GdnStateCache gs;
        gs.ssm_state = vt::Tensor::Contiguous(
            alloc(static_cast<size_t>(num_blocks * Hv * Dv * Dk) * sizeof(float)),
            DType::kF32, dev, {num_blocks, Hv, Dv, Dk});
        gs.conv_state = vt::Tensor::Contiguous(
            alloc(static_cast<size_t>(num_blocks * conv_dim * (Kw - 1)) * sizeof(float)),
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
  ~CudaGdnCachePool() {
    for (void* p : owned) b.Free(p);
  }
};

vllm::v1::GDNAttentionMetadata DecodeGdnMeta() {
  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 0;
  gm.num_prefill_tokens = 0;
  gm.num_decodes = 1;
  gm.num_decode_tokens = 1;
  gm.num_actual_tokens = 1;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
  return gm;
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

HfConfig Qwen35Base() {
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
  return c;
}

HfConfig Qwen35MoeConfig() {
  HfConfig c = Qwen35Base();
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 16;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_value_heads = 4;
  return c;
}

vllm::MoeBlockWeights Qwen35Moe(const HfConfig& c, uint32_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeBf16({H, E}, false, s + 1);
  m.shared_gate = MakeBf16({H, 1}, false, s + 2);
  for (int64_t e = 0; e < E; ++e) {
    m.expert_gate.push_back(MakeBf16({H, I}, false, s + 100 + static_cast<uint32_t>(e)));
    m.expert_up.push_back(MakeBf16({H, I}, false, s + 200 + static_cast<uint32_t>(e)));
    m.expert_down.push_back(MakeBf16({I, H}, false, s + 300 + static_cast<uint32_t>(e)));
  }
  m.shared_gate_proj = MakeBf16({H, Is}, false, s + 3);
  m.shared_up_proj = MakeBf16({H, Is}, false, s + 4);
  m.shared_down_proj = MakeBf16({Is, H}, false, s + 5);
  return m;
}

// `Gdn` and `Attn` are filled identically for the MoE and dense arms, so they
// are one function each over the layer-weight type.
template <class LayerW>
void FillQwen35Layer(LayerW& lw, const HfConfig& c, bool linear, uint32_t s) {
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

vllm::Qwen3_5MoeWeights Qwen35MoeWeights(const HfConfig& c) {
  vllm::Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  w.embed_tokens = MakeBf16({V, H}, false, 11);
  w.final_norm = MakeBf16({H}, false, 12, 0.5f);
  w.lm_head = MakeBf16({H, V}, false, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint32_t s = 1000 + static_cast<uint32_t>(l) * 5000;
    vllm::Qwen3_5MoeLayerWeights lw;
    FillQwen35Layer(lw, c, c.layer_types[static_cast<size_t>(l)] == "linear_attention",
                    s);
    lw.moe = Qwen35Moe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

HfConfig Qwen35DenseConfig() {
  HfConfig c = Qwen35Base();
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_value_heads = 6;  // GQA ratio 3
  return c;
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

}  // namespace

TEST_CASE("G1 CUDA: Qwen3MoeDecodeGraph replays bit-exactly against eager, 3 replays") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered; G1 needs a leased device");
    return;
  }
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = b.CreateQueue();
  const HfConfig c = MoeConfig();
  const Qwen3MoeWeights w = MoeWeights(c);

  CudaDenseCachePool eager_kv(b, q, c.num_hidden_layers, c.num_key_value_heads, c.head_dim,
                              /*num_blocks=*/1, /*block_size=*/16);
  CudaDenseCachePool graph_kv(b, q, c.num_hidden_layers, c.num_key_value_heads, c.head_dim,
                              1, 16);
  // max_num_reqs 0 -> PadToCaptureSize returns -1 -> the driver's EAGER arm.
  vllm::Qwen3MoeDecodeGraph eager(w, c, q, /*max_num_reqs=*/0);
  vllm::Qwen3MoeDecodeGraph graphed(w, c, q, /*max_num_reqs=*/8);

  size_t total = 0;
  for (int step = 0; step < 5; ++step) {
    const std::vector<int32_t> tok = {kTokens[static_cast<size_t>(step)]};
    const std::vector<int32_t> pos = {step};
    const CommonAttentionMetadata am = DecodeMeta(step);
    const std::vector<float> e = Read(b, q, eager.Step(tok, pos, am, eager_kv.attn_kv),
                                      c.vocab_size);
    const std::vector<float> g = Read(b, q, graphed.Step(tok, pos, am, graph_kv.attn_kv),
                                      c.vocab_size);
    total += CompareStep(step, e, g);
  }
  CHECK_FALSE(eager.captured());
  CHECK(graphed.captured());
  CHECK(graphed.replay_count() >= 3);  // capture + 3 replays: MORE than one
  MESSAGE("G1 Qwen3MoeDecodeGraph on CUDA: 5 steps x " << c.vocab_size << " logits, "
                                                       << total << " differing, "
                                                       << graphed.replay_count()
                                                       << " replays");
}

TEST_CASE("G1 CUDA: VoxtralDecodeGraph replays bit-exactly against eager, 3 replays") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered; G1 needs a leased device");
    return;
  }
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = b.CreateQueue();
  const HfConfig c = VoxConfig();
  const Qwen3DenseWeights w = VoxWeights(c);

  CudaDenseCachePool eager_kv(b, q, c.num_hidden_layers, c.num_key_value_heads, c.head_dim,
                              1, 16);
  CudaDenseCachePool graph_kv(b, q, c.num_hidden_layers, c.num_key_value_heads, c.head_dim,
                              1, 16);
  vllm::VoxtralDecodeGraph eager(w, c, q, /*max_num_reqs=*/0);
  vllm::VoxtralDecodeGraph graphed(w, c, q, /*max_num_reqs=*/1);

  size_t total = 0;
  for (int step = 0; step < 5; ++step) {
    const std::vector<int32_t> tok = {kTokens[static_cast<size_t>(step)]};
    const std::vector<int32_t> pos = {step};
    const CommonAttentionMetadata am = DecodeMeta(step);
    const std::vector<float> e = Read(b, q, eager.Step(tok, pos, am, eager_kv.attn_kv),
                                      c.vocab_size);
    const std::vector<float> g = Read(b, q, graphed.Step(tok, pos, am, graph_kv.attn_kv),
                                      c.vocab_size);
    total += CompareStep(step, e, g);
  }
  CHECK_FALSE(eager.captured());
  CHECK(graphed.captured());
  CHECK(graphed.replay_count() >= 3);
  MESSAGE("G1 VoxtralDecodeGraph on CUDA: 5 steps x " << c.vocab_size << " logits, "
                                                      << total << " differing, "
                                                      << graphed.replay_count()
                                                      << " replays");
}

TEST_CASE("G1 CUDA: DeepseekV2DecodeGraph replays bit-exactly against eager, 3 replays") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered; G1 needs a leased device");
    return;
  }
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = b.CreateQueue();
  const DeepseekV2Params p = vllm::ParseDeepseekV2Params(vllm::LoadHfConfig(WriteDsConfig()));
  REQUIRE(p.mla.qk_head_dim() == 192);  // the ONLY head_dim the CUDA MLA path has
  const DeepseekV2Weights w = DsWeights(p);

  CudaMlaCachePool eager_kv(b, q, p, 1, 16);
  CudaMlaCachePool graph_kv(b, q, p, 1, 16);
  vllm::DeepseekV2DecodeGraph eager(w, q, /*max_num_reqs=*/0);
  vllm::DeepseekV2DecodeGraph graphed(w, q, /*max_num_reqs=*/8);

  size_t total = 0;
  for (int step = 0; step < 5; ++step) {
    const std::vector<int32_t> tok = {kTokens[static_cast<size_t>(step)]};
    const std::vector<int32_t> pos = {step};
    const CommonAttentionMetadata am = DecodeMeta(step);
    const std::vector<float> e = Read(b, q, eager.Step(tok, pos, am, eager_kv.attn_kv),
                                      p.vocab_size);
    const std::vector<float> g = Read(b, q, graphed.Step(tok, pos, am, graph_kv.attn_kv),
                                      p.vocab_size);
    total += CompareStep(step, e, g);
  }
  CHECK_FALSE(eager.captured());
  CHECK(graphed.captured());
  CHECK(graphed.replay_count() >= 3);
  MESSAGE("G1 DeepseekV2DecodeGraph on CUDA: 5 steps x " << p.vocab_size << " logits, "
                                                         << total << " differing, "
                                                         << graphed.replay_count()
                                                         << " replays");
}

TEST_CASE("G1 CUDA: Qwen3_5DecodeGraph replays bit-exactly against eager, 3 replays") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered; G1 needs a leased device");
    return;
  }
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = b.CreateQueue();
  const HfConfig c = Qwen35MoeConfig();
  const vllm::Qwen3_5MoeWeights w = Qwen35MoeWeights(c);

  CudaGdnCachePool eager_cache(b, q, c, /*num_blocks=*/4, /*block_size=*/16);
  CudaGdnCachePool graph_cache(b, q, c, 4, 16);
  vllm::Qwen3_5DecodeGraph eager(w, c, q, /*max_num_reqs=*/0);   // EAGER arm
  vllm::Qwen3_5DecodeGraph graphed(w, c, q, /*max_num_reqs=*/4);

  size_t total = 0;
  for (int step = 0; step < 5; ++step) {
    const std::vector<int32_t> tok = {kTokens[static_cast<size_t>(step)]};
    const std::vector<int32_t> pos = {step};
    const CommonAttentionMetadata am = DecodeMeta(step);
    const vllm::v1::GDNAttentionMetadata gm = DecodeGdnMeta();
    const std::vector<float> e =
        Read(b, q, eager.Step(tok, pos, am, gm, eager_cache.attn_kv,
                              eager_cache.gdn_state),
             c.vocab_size);
    const std::vector<float> g =
        Read(b, q, graphed.Step(tok, pos, am, gm, graph_cache.attn_kv,
                                graph_cache.gdn_state),
             c.vocab_size);
    total += CompareStep(step, e, g);
  }
  CHECK_FALSE(eager.captured());
  CHECK(graphed.captured());
  CHECK(graphed.replay_count() >= 3);  // capture + 3 replays: MORE than one
  MESSAGE("G1 Qwen3_5DecodeGraph on CUDA: 5 steps x " << c.vocab_size << " logits, "
                                                      << total << " differing, "
                                                      << graphed.replay_count()
                                                      << " replays");
}

TEST_CASE("G1 CUDA: Qwen3_5DenseDecodeGraph replays bit-exactly against eager, 3 replays") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered; G1 needs a leased device");
    return;
  }
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = b.CreateQueue();
  const HfConfig c = Qwen35DenseConfig();
  const vllm::Qwen3_5DenseWeights w = Qwen35DenseWeights(c);

  CudaGdnCachePool eager_cache(b, q, c, /*num_blocks=*/4, /*block_size=*/16);
  CudaGdnCachePool graph_cache(b, q, c, 4, 16);
  vllm::Qwen3_5DenseDecodeGraph eager(w, c, q, /*max_num_reqs=*/0);   // EAGER arm
  vllm::Qwen3_5DenseDecodeGraph graphed(w, c, q, /*max_num_reqs=*/4);

  size_t total = 0;
  for (int step = 0; step < 5; ++step) {
    const std::vector<int32_t> tok = {kTokens[static_cast<size_t>(step)]};
    const std::vector<int32_t> pos = {step};
    const CommonAttentionMetadata am = DecodeMeta(step);
    const vllm::v1::GDNAttentionMetadata gm = DecodeGdnMeta();
    const std::vector<float> e =
        Read(b, q, eager.Step(tok, pos, am, gm, eager_cache.attn_kv,
                              eager_cache.gdn_state),
             c.vocab_size);
    const std::vector<float> g =
        Read(b, q, graphed.Step(tok, pos, am, gm, graph_cache.attn_kv,
                                graph_cache.gdn_state),
             c.vocab_size);
    total += CompareStep(step, e, g);
  }
  CHECK_FALSE(eager.captured());
  CHECK(graphed.captured());
  CHECK(graphed.replay_count() >= 3);
  MESSAGE("G1 Qwen3_5DenseDecodeGraph on CUDA: 5 steps x "
          << c.vocab_size << " logits, " << total << " differing, "
          << graphed.replay_count() << " replays");
}
