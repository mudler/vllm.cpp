// MoE vision tower (issue #891, .agents/specs/moe-vision-tower.md) — the CPU
// gates for the two things that were missing on `Qwen3_5MoeForConditionalGenera
// tion`: the tower is no longer dropped at load, and the forward actually
// CONSUMES it.
//
// What these gates establish, and what they deliberately do NOT:
//
//   * LOADER — `Qwen/Qwen3.6-35B-A3B` ships 333 `model.visual.*` tensors that
//     `LoadQwen3_5Moe` does not read. `LoadQwen3_5MoeVision` reads them through
//     the SHARED `LoadQwen3VLVisionWeights` the dense 27B arm already gates, and
//     REFUSES BY NAME when a checkpoint carries none — instead of silently
//     producing a text-only model that answers image prompts from text alone.
//     Gated here on synthetic shards (a real 27-block tower is ~1.6 GB f32).
//
//   * FORWARD INVOCATION — the risk the spec names first: "if the tower loads
//     but is never invoked, image input degenerates to text-only and still
//     produces plausible tokens". A green suite cannot see that unless something
//     asserts the merger rows CHANGE the output. Two gates do:
//       (a) two different `mm_main` tensors must produce different token
//           streams (kills "never invoked" / "output ignored");
//       (b) SWAPPING two `mm_main` rows must change the token stream (kills a
//           scatter that ignores the row->position mapping).
//     Both run the real `Qwen3_5MoeVLGenerateGreedy` over a small synthetic MoE
//     model on CPU, i.e. the same driver the 35B runs.
//
//   * NOT ESTABLISHED HERE: token-EXACTNESS. Nothing on CPU compares against the
//     pinned oracle, so these gates prove the tower is loaded and invoked and
//     that its values propagate — never that the values are right. That is the
//     binding image/video token-exact gate at 35B, and it is separate.
#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5MoeWeights;
using vllm::SafetensorsFile;
using vt::DType;

namespace {

// splitmix64-based small deterministic values in [-0.08, 0.08) — the same
// generator tests/vllm/models/test_qwen35_paged_forward.cpp uses, so the
// synthetic model is the one the paged-forward gate already exercises.
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
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

// A small GDN-hybrid MoE config with the 35B's MRoPE shape: 3 mrope sections,
// interleaved, summing to rotary_dim/2 (the 35B is [11,11,10] over rotary_dim
// 64; this is [1,1,0] over rotary_dim 4). Everything else is the synthetic
// paged-forward config.
HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
  c.vocab_size = 40;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 16;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  c.rope_parameters.mrope_interleaved = true;
  c.rope_parameters.mrope_section = {1, 1, 0};
  return c;
}

// A ROPE-SENSITIVE variant of the config above: 3 of the 4 layers are
// full_attention, so 3 layers apply rope (the shipped hybrid ratio rotates in 1
// layer out of 4, and GDN layers carry no rope at all), and the whole head dim
// rotates. Used by the MRoPE gate, where the property under test IS the rope
// angle. One GDN layer is kept so the hybrid state path still runs.
HfConfig MakeConfigAllAttn() {
  HfConfig c = MakeConfig();
  c.layer_types = {"linear_attention", "full_attention", "full_attention",
                   "full_attention"};
  c.rotary_dim = 8;                          // == head_dim
  c.rope_parameters.mrope_section = {2, 1, 1};  // sums to rotary_dim/2
  return c;
}

vllm::MoeBlockWeights MakeMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(DType::kBF16, {H, 1}, s + 2);
  for (int64_t e = 0; e < E; ++e) {
    m.expert_gate.push_back(MakeOwned(DType::kBF16, {H, I}, s + 100 + e * 7));
    m.expert_up.push_back(MakeOwned(DType::kBF16, {H, I}, s + 200 + e * 7));
    m.expert_down.push_back(MakeOwned(DType::kBF16, {I, H}, s + 300 + e * 7));
  }
  m.shared_gate_proj = MakeOwned(DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(DType::kBF16, {Is, H}, s + 5);
  return m;
}

Qwen3_5MoeWeights MakeWeights(const HfConfig& c) {
  Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5MoeLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
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
    lw.moe = MakeMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// Tower merger output stand-in: [N, H] host f32, the shape/dtype
// `Qwen3VLVisionForward` returns.
//
// THE EXACT REDUCTION these gates lean on. `mm_main` row r is set to the model's
// OWN embedding row for a chosen token id, and the visual grid is chosen so the
// LLM grid is 1x1x1 per visual token. Both halves then collapse:
//   * the masked scatter writes embed_tokens[k] into the visual row, so
//     inputs_embeds is EXACTLY embed(prompt with the visual token replaced by k);
//   * `Qwen3VLGetRopeIndex[Video]` gives a 1x1x1 visual token the running-max
//     position on all three axes, so the MRoPE positions [3,T] are the plain
//     sequential 1-D positions broadcast, and `delta` is 0.
// So the whole forked VL forward must reproduce a plain TEXT greedy run over the
// substituted prompt, token for token. That is an EXACT equality, not a "the
// output moved" heuristic — and every defect this row can have breaks it: a
// tower that is never invoked leaves the placeholder id's own embedding there, a
// scatter at the wrong row lands it elsewhere, a wrong MRoPE cache changes the
// rope angle, a wrong decode delta shifts every generated position.
std::vector<float> EmbeddingRowF32(const Qwen3_5MoeWeights& w, int64_t token_id,
                                   int64_t h) {
  const auto* p = reinterpret_cast<const uint16_t*>(w.embed_tokens.bytes.data());
  std::vector<float> row(static_cast<size_t>(h));
  for (int64_t j = 0; j < h; ++j)
    row[static_cast<size_t>(j)] =
        vt::BF16ToF32(p[static_cast<size_t>(token_id * h + j)]);
  return row;
}

// ── Plain TEXT greedy reference ──────────────────────────────────────────────
// Prefill + single-token decode through `Qwen3_5Model::Forward` over the SAME
// cache shapes/dtypes the VL core allocates (bf16 paged KV, f32 GDN ssm state,
// bf16 GDN conv state, one block, one sequence), with 1-D sequential positions.
// This is the "no multimodal input at all" path, and the equality above is
// against it.
std::vector<int32_t> TextGreedy(const std::vector<int32_t>& prompt_ids,
                                const Qwen3_5MoeWeights& w, const HfConfig& c,
                                vt::Queue& q, int max_new_tokens) {
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());
  const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads;
  const int64_t Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim;
  const int64_t Kw = c.linear_conv_kernel_dim;
  const int64_t conv_dim = 2 * Hk * Dk + Hv * Dv, conv_len = Kw - 1;
  const int64_t block_size = T0 + max_new_tokens + 8;
  const vt::Device dev{vt::DeviceType::kCPU, 0};

  std::vector<std::vector<uint16_t>> kv_buf;
  std::vector<std::vector<float>> ssm_buf;
  std::vector<std::vector<uint16_t>> conv_buf;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    if (c.layer_types[static_cast<size_t>(l)] == "linear_attention") {
      ssm_buf.emplace_back(static_cast<size_t>(Hv * Dv * Dk), 0.0F);
      conv_buf.emplace_back(static_cast<size_t>(conv_dim * conv_len), 0);
    } else {
      kv_buf.emplace_back(static_cast<size_t>(2 * block_size * Hkv * Dh), 0);
    }
  }
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  for (auto& b : kv_buf) {
    vllm::PagedKvCache kv;
    kv.data = b.data();
    kv.dtype = DType::kBF16;
    kv.num_blocks = 1;
    kv.block_size = block_size;
    kv.num_kv_heads = Hkv;
    kv.head_size = Dh;
    attn_kv.push_back(kv);
  }
  for (size_t g = 0; g < ssm_buf.size(); ++g) {
    vllm::GdnStateCache gs;
    gs.ssm_state = vt::Tensor::Contiguous(ssm_buf[g].data(), DType::kF32, dev,
                                          {1, Hv, Dv, Dk});
    gs.conv_state = vt::Tensor::Contiguous(conv_buf[g].data(), DType::kBF16, dev,
                                           {1, conv_dim, conv_len});
    gdn_state.push_back(gs);
  }

  auto attn_meta = [&](int64_t qlen, int64_t context) {
    vllm::v1::CommonAttentionMetadata m;
    m.num_reqs = 1;
    m.num_actual_tokens = static_cast<int>(qlen);
    m.query_start_loc = {0, static_cast<int32_t>(qlen)};
    m.query_start_loc_cpu = m.query_start_loc;
    m.seq_lens = {static_cast<int32_t>(context + qlen)};
    m.seq_lens_cpu = m.seq_lens;
    m.max_query_len = static_cast<int>(qlen);
    m.max_seq_len = static_cast<int>(context + qlen);
    m.block_table_num_cols = 1;
    m.block_table_tensor = {0};
    for (int64_t t = 0; t < qlen; ++t) m.slot_mapping.push_back(context + t);
    m.causal = true;
    return m;
  };

  std::vector<int32_t> generated;
  const int64_t vocab = c.vocab_size;
  auto argmax_last = [&](const std::vector<float>& logits) {
    const size_t rows = logits.size() / static_cast<size_t>(vocab);
    const float* row = logits.data() + (rows - 1) * static_cast<size_t>(vocab);
    int32_t best = 0;
    for (int64_t v = 1; v < vocab; ++v)
      if (row[v] > row[best]) best = static_cast<int32_t>(v);
    return best;
  };

  {
    std::vector<int32_t> pos(static_cast<size_t>(T0));
    for (int64_t t = 0; t < T0; ++t) pos[static_cast<size_t>(t)] =
        static_cast<int32_t>(t);
    vllm::v1::GDNAttentionMetadata g;
    g.num_prefills = 1;
    g.num_prefill_tokens = static_cast<int>(T0);
    g.num_decodes = 0;
    g.num_decode_tokens = 0;
    g.num_actual_tokens = static_cast<int>(T0);
    g.has_initial_state = std::vector<uint8_t>{0};
    g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    g.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T0)};
    g.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T0)};
    g.prefill_state_indices = std::vector<int32_t>{0};
    g.prefill_has_initial_state = std::vector<uint8_t>{0};
    const vllm::v1::CausalConv1dMetadata conv =
        vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
    g.batch_ptr = conv.batch_ptr;
    g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    const std::vector<float> logits = vllm::Qwen3_5Model::Forward(
        prompt_ids, pos, attn_meta(T0, 0), g, attn_kv, gdn_state, w, c, q,
        {static_cast<int32_t>(T0 - 1)});
    generated.push_back(argmax_last(logits));
  }
  for (int step = 1; step < max_new_tokens; ++step) {
    const int64_t abs_idx = T0 + (step - 1);
    vllm::v1::GDNAttentionMetadata g;
    g.num_prefills = 0;
    g.num_prefill_tokens = 0;
    g.num_decodes = 1;
    g.num_decode_tokens = 1;
    g.num_actual_tokens = 1;
    g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    g.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
    const std::vector<int32_t> one = {generated.back()};
    const std::vector<int32_t> pos = {static_cast<int32_t>(abs_idx)};
    const std::vector<float> logits = vllm::Qwen3_5Model::Forward(
        one, pos, attn_meta(1, abs_idx), g, attn_kv, gdn_state, w, c, q, {});
    generated.push_back(argmax_last(logits));
  }
  return generated;
}

// ── Synthetic safetensors shard (loader gates) ───────────────────────────────
// Minimal well-formed file: 8-byte little-endian header length + JSON header +
// the data blob. One f32 scalar per named tensor is enough — the refusal gate
// fires on tensor NAMES, before a single weight byte is read.
std::string WriteShard(const std::string& path,
                       const std::vector<std::string>& names) {
  std::string json = "{";
  for (size_t i = 0; i < names.size(); ++i) {
    if (i != 0) json += ",";
    json += "\"" + names[i] + "\":{\"dtype\":\"F32\",\"shape\":[1],"
            "\"data_offsets\":[" + std::to_string(i * 4) + "," +
            std::to_string((i + 1) * 4) + "]}";
  }
  json += "}";
  while (json.size() % 8 != 0) json += " ";
  std::string out(8, '\0');
  const uint64_t n = json.size();
  for (int b = 0; b < 8; ++b) out[static_cast<size_t>(b)] =
      static_cast<char>((n >> (8 * b)) & 0xFF);
  out += json;
  out.append(names.size() * 4, '\0');
  FILE* f = std::fopen(path.c_str(), "wb");
  REQUIRE(f != nullptr);
  std::fwrite(out.data(), 1, out.size(), f);
  std::fclose(f);
  return path;
}

std::string TmpDir() {
  const char* t = std::getenv("TMPDIR");
  return std::string(t != nullptr ? t : "/tmp");
}

}  // namespace

// ── LOADER: the tower is no longer dropped, and its ABSENCE is a refusal ─────

TEST_CASE("qwen3_5_moe_vision_config_mirrors_the_checkpoint_vision_config") {
  HfConfig c = MakeConfig();
  c.hidden_size = 2048;  // Qwen/Qwen3.6-35B-A3B text hidden
  const vllm::multimodal::Qwen3VLVisionConfig v = vllm::Qwen3_5MoeVisionConfig(c);
  // config.json::vision_config of Qwen/Qwen3.6-35B-A3B.
  CHECK(v.depth == 27);
  CHECK(v.hidden_size == 1152);
  CHECK(v.num_heads == 16);
  CHECK(v.intermediate_size == 4304);
  CHECK(v.patch_size == 16);
  CHECK(v.temporal_patch_size == 2);
  CHECK(v.spatial_merge_size == 2);
  CHECK(v.num_position_embeddings == 2304);
  CHECK(v.in_channels == 3);
  // deepstack_visual_indexes: [] — the deepstack path is compiled out upstream
  // for this family (qwen3_vl.py:1709-1716).
  CHECK(v.deepstack_visual_indexes.empty());
  // The merger writes into the text residual stream, so the tower's output width
  // IS the text hidden size. This is the ONE field that differs between the 35B
  // MoE (2048) and the 27B dense (5120) arms.
  CHECK(v.out_hidden_size == 2048);
  c.hidden_size = 5120;
  CHECK(vllm::Qwen3_5MoeVisionConfig(c).out_hidden_size == 5120);
}

TEST_CASE("qwen3_5_moe_vision_absent_visual_tensors_are_refused_by_name") {
  const std::string p = TmpDir() + "/vllmcpp_moe_novisual.safetensors";
  WriteShard(p, {"model.language_model.embed_tokens.weight",
                 "model.language_model.norm.weight", "lm_head.weight"});
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(p));

  CHECK_FALSE(vllm::HasQwen3_5MoeVisionTower(shards));

  // The whole point: a text-only checkpoint must NOT quietly yield an empty
  // tower. It must name what is missing.
  std::string msg;
  bool threw = false;
  try {
    (void)vllm::LoadQwen3_5MoeVision(shards, MakeConfig());
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  REQUIRE(threw);
  CHECK(msg.find("model.visual.") != std::string::npos);
  CHECK(msg.find("patch_embed") != std::string::npos);
  CHECK(msg.find("merger") != std::string::npos);
  std::remove(p.c_str());
}

TEST_CASE("qwen3_5_moe_vision_present_visual_tensors_are_seen_not_dropped") {
  const std::string p = TmpDir() + "/vllmcpp_moe_withvisual.safetensors";
  // One real `model.visual.*` name from Qwen/Qwen3.6-35B-A3B's index.
  WriteShard(p, {"model.language_model.embed_tokens.weight",
                 "model.visual.blocks.0.attn.proj.bias"});
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(p));

  CHECK(vllm::HasQwen3_5MoeVisionTower(shards));

  // The tower is now REACHED: this checkpoint no longer trips the "no tower"
  // refusal, so the load proceeds into the shared reader and fails on the FIRST
  // tower tensor it actually needs — a different, named error. (A complete
  // synthetic 27-block tower is ~1.6 GB f32; the real tower load is gated on
  // hardware against Qwen/Qwen3.6-35B-A3B's 333 tensors.)
  std::string msg;
  try {
    (void)vllm::LoadQwen3_5MoeVision(shards, MakeConfig());
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE(!msg.empty());
  CHECK(msg.find("carries NO `model.visual.*` tensors") == std::string::npos);
  CHECK(msg.find("patch_embed.proj.weight") != std::string::npos);
  std::remove(p.c_str());
}

// ── FORWARD: the tower output is actually CONSUMED, at the right rows ────────

TEST_CASE("qwen3_5_moe_vl_image_forward_is_the_text_forward_over_the_tower_row") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t H = c.hidden_size;
  const int32_t kImg = 30;      // the image placeholder id
  constexpr int kSteps = 6;

  // ONE image token, grid (1,2,2) -> a 1x1x1 LLM grid -> MRoPE degenerates to
  // the plain 1-D positions (see EmbeddingRowF32's note).
  const std::array<int64_t, 3> grid = {1, 2, 2};
  auto prompt_with = [&](int32_t mid) {
    return std::vector<int32_t>{1, 2, 3, mid, 5, 6};
  };
  const std::vector<int32_t> prompt_img = prompt_with(kImg);
  const std::vector<int32_t> text_placeholder =
      TextGreedy(prompt_img, w, c, q, kSteps);

  // Pick the two substituted tokens BY SEARCH rather than by hardcoding, and
  // require that a discriminating pair exists. A greedy argmax over a small
  // synthetic model is a coarse instrument: two substitutions that happen to
  // decode identically would make the equality below vacuous, and a hardcoded
  // pair silently becomes vacuous the day the synthetic weights change.
  int32_t kSub = -1, kSub2 = -1;
  std::vector<int32_t> text_sub, text_sub2;
  for (int32_t k = 0; k < static_cast<int32_t>(c.vocab_size) && kSub2 < 0; ++k) {
    if (k == kImg) continue;
    const std::vector<int32_t> t = TextGreedy(prompt_with(k), w, c, q, kSteps);
    if (t == text_placeholder) continue;
    if (kSub < 0) {
      kSub = k;
      text_sub = t;
    } else if (t != text_sub) {
      kSub2 = k;
      text_sub2 = t;
    }
  }
  REQUIRE(kSub >= 0);
  REQUIRE(kSub2 >= 0);
  MESSAGE("substituted ids: kSub=" << kSub << " kSub2=" << kSub2);
  REQUIRE(text_sub != text_sub2);
  REQUIRE(text_sub != text_placeholder);

  const std::vector<int32_t> vl_sub = vllm::Qwen3_5MoeVLGenerateGreedy(
      prompt_img, EmbeddingRowF32(w, kSub, H), grid, kImg, /*eos_token_id=*/-1,
      w, c, q, kSteps);
  const std::vector<int32_t> vl_sub2 = vllm::Qwen3_5MoeVLGenerateGreedy(
      prompt_img, EmbeddingRowF32(w, kSub2, H), grid, kImg, -1, w, c, q, kSteps);

  // THE gate. Feeding the tower row for `kSub` must reproduce the text run over
  // the prompt with `kSub` in that position, token for token.
  //
  // A tower that loads but is NEVER INVOKED — the spec's first named risk —
  // would leave the placeholder's own embedding in that row and so reproduce
  // `text_placeholder` instead, while still emitting perfectly plausible tokens.
  CHECK(vl_sub == text_sub);
  CHECK(vl_sub2 == text_sub2);
  CHECK(vl_sub != text_placeholder);
}

TEST_CASE("qwen3_5_moe_vl_image_forward_uses_MRoPE_positions_not_plain_1d") {
  const HfConfig c = MakeConfigAllAttn();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t H = c.hidden_size;
  const int32_t kImg = 30, kSub = 16;
  constexpr int kSteps = 6;

  // The COMPLEMENT of the degenerate case above. Grid (1,16,16) -> an 8x8 LLM
  // grid, so the 64 image tokens carry MRoPE (t,h,w) positions in 0..7 per axis
  // where plain 1-D positions would run 0..63, and the decode continuation is
  // shifted by `mrope_position_delta`. Feeding the SAME substituted embedding in
  // every visual row isolates the positions: the only thing that can still
  // differ from the text run is the rope.
  //
  // This is the gate that fails if the injected MRoPE cos|sin cache is ignored
  // and the 1-D cache is built from `positions` instead — the degenerate 1x1x1
  // case above cannot see that, because there the two caches coincide.
  const std::array<int64_t, 3> grid = {1, 16, 16};
  const int64_t n_vis = (grid[1] / 2) * (grid[2] / 2);  // 64
  std::vector<int32_t> prompt_img = {1, 2, 3};
  std::vector<int32_t> prompt_sub = {1, 2, 3};
  for (int64_t i = 0; i < n_vis; ++i) {
    prompt_img.push_back(kImg);
    prompt_sub.push_back(kSub);
  }
  prompt_img.push_back(5);
  prompt_sub.push_back(5);

  const std::vector<float> row = EmbeddingRowF32(w, kSub, H);
  std::vector<float> mm;
  for (int64_t i = 0; i < n_vis; ++i) mm.insert(mm.end(), row.begin(), row.end());

  const std::vector<int32_t> vl =
      vllm::Qwen3_5MoeVLGenerateGreedy(prompt_img, mm, grid, kImg,
                                       /*eos_token_id=*/-1, w, c, q, kSteps);
  const std::vector<int32_t> text_1d = TextGreedy(prompt_sub, w, c, q, kSteps);
  REQUIRE(vl.size() == static_cast<size_t>(kSteps));
  CHECK(vl != text_1d);
}

TEST_CASE("qwen3_5_moe_vl_video_forward_scatters_each_frame_row_in_order") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t H = c.hidden_size;
  const int32_t kVid = 31, kStart = 28, kEnd = 29;
  const int32_t kA = 7, kB = 21;
  constexpr int kSteps = 8;

  // Two frames, ONE video token each (grid (2,2,2) -> per frame (2/2)*(2/2)=1),
  // in the timestamp-interleaved placeholder structure. Each frame's LLM grid is
  // 1x1x1, so the video MRoPE scan degenerates to the plain 1-D positions too —
  // and there are now TWO tower rows, so their ORDER is observable.
  const std::array<int64_t, 3> grid = {2, 2, 2};
  const std::vector<int32_t> prompt_vid = {1, kStart, kVid, kEnd,
                                           2, kStart, kVid, kEnd, 3};
  auto substituted = [&](int32_t first, int32_t second) {
    return std::vector<int32_t>{1, kStart, first, kEnd, 2, kStart, second, kEnd, 3};
  };

  const std::vector<int32_t> text_ab =
      TextGreedy(substituted(kA, kB), w, c, q, kSteps);
  const std::vector<int32_t> text_ba =
      TextGreedy(substituted(kB, kA), w, c, q, kSteps);
  REQUIRE(text_ab != text_ba);  // the order must be observable at all

  std::vector<float> mm_ab = EmbeddingRowF32(w, kA, H);
  const std::vector<float> row_b = EmbeddingRowF32(w, kB, H);
  mm_ab.insert(mm_ab.end(), row_b.begin(), row_b.end());
  std::vector<float> mm_ba = EmbeddingRowF32(w, kB, H);
  const std::vector<float> row_a = EmbeddingRowF32(w, kA, H);
  mm_ba.insert(mm_ba.end(), row_a.begin(), row_a.end());

  const std::vector<int32_t> vl_ab = vllm::Qwen3_5MoeVLGenerateGreedyVideo(
      prompt_vid, mm_ab, grid, kVid, kStart, kEnd, /*eos_token_id=*/-1, w, c, q,
      kSteps);
  const std::vector<int32_t> vl_ba = vllm::Qwen3_5MoeVLGenerateGreedyVideo(
      prompt_vid, mm_ba, grid, kVid, kStart, kEnd, -1, w, c, q, kSteps);

  // Row 0 must land on frame 0's video position and row 1 on frame 1's. A
  // scatter that broadcast one row, or wrote them in the wrong order, satisfies
  // neither equality.
  CHECK(vl_ab == text_ab);
  CHECK(vl_ba == text_ba);
}

TEST_CASE("qwen3_5_moe_vl_forward_refuses_a_mismatched_tower_row_count") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int32_t kImg = 30;
  const std::vector<int32_t> prompt_ids = {1, 2, 3, kImg, kImg, kImg, kImg, 5};
  const std::array<int64_t, 3> grid = {1, 4, 4};

  // 3 rows for 4 image tokens: silently scattering 3 and leaving the 4th as the
  // placeholder embedding is exactly the degenerate-to-text failure.
  std::vector<float> short_mm;
  for (int32_t k = 0; k < 3; ++k) {
    const std::vector<float> row = EmbeddingRowF32(w, k, c.hidden_size);
    short_mm.insert(short_mm.end(), row.begin(), row.end());
  }
  CHECK_THROWS_AS((void)vllm::Qwen3_5MoeVLGenerateGreedy(
                      prompt_ids, short_mm, grid, kImg, -1, w, c, q, 2),
                  std::exception);

  // No image token at all is likewise refused rather than silently text-only.
  const std::vector<int32_t> text_only = {1, 2, 3, 4, 5};
  CHECK_THROWS_AS((void)vllm::Qwen3_5MoeVLGenerateGreedy(
                      text_only, EmbeddingRowF32(w, 1, c.hidden_size), grid,
                      kImg, -1, w, c, q, 2),
                  std::exception);

  // And a video prompt with no video token.
  CHECK_THROWS_AS((void)vllm::Qwen3_5MoeVLGenerateGreedyVideo(
                      text_only, EmbeddingRowF32(w, 1, c.hidden_size), grid, 31,
                      28, 29, -1, w, c, q, 2),
                  std::exception);
}
