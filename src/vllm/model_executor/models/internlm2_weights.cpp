// Weight loader for InternLM2 (`InternLM2ForCausalLM`, internlm2-chat-1_8b, BF16).
// Loads the checkpoint safetensors into the SHARED dense container
// (Qwen3DenseWeights, qwen3.h) via the SHARED dense_weight_loaders.h helpers. The
// ONLY delta over the Llama/Granite loaders is (a) the InternLM2 name map and
// (b) the fused `wqkv` DE-INTERLEAVE — every other projection reuses the shared
// merged/direct/transposed BF16 helpers VERBATIM.
//
// Grounding: vllm/model_executor/models/internlm2.py @ e24d1b24 —
//   - InternLM2Attention.wqkv: ONE QKVParallelLinear(head_dim, total_num_heads,
//     total_num_kv_heads) (:126-135), bias=False. Its output rows are grouped by
//     kv-head: for each of num_kv_heads groups, {key_value_groups query heads, 1
//     key head, 1 value head}, each head head_dim rows.
//   - InternLM2Attention.split_qkv (:158-176, single-GPU path): reshape the
//     projection to [seq, total_num_kv_heads, key_value_groups+2, head_dim] then
//     torch.split(..., [key_value_groups, 1, 1], dim=-2) -> q, k, v, each
//     flattened back to [seq, q_size|kv_size|kv_size]. This loader inverts that
//     on the WEIGHT rows so the merged qkv_proj is the plain [q|k|v]-row operand
//     the shared dense AttnBlock consumes (the projection output then splits by
//     QkvSplit(q_size, kv_size, kv_size), byte-identical to vLLM's split_qkv).
//   - InternLM2MLP: merged gate_up_proj <- [w1, w3] (hf_to_vllm_mapper :278-283),
//     SiluAndMul, down_proj = w2 (:236-247).
//   - hf_to_vllm_mapper name map: model.tok_embeddings, attention.wqkv,
//     attention.wo, feed_forward.{w1,w2,w3}, attention_norm, ffn_norm, output.
//   - tie_word_embeddings: false for the chat checkpoints (a separate
//     `output.weight` lm_head); the loader loads it (Matmul-B layout) when untied.
#include "vllm/model_executor/models/internlm2.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::MakeOwned;

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

// De-interleave InternLM2's fused `wqkv` weight into the plain [q|k|v]-row merged
// qkv_proj (raw-NK bf16 [Hq*Dh + 2*Hkv*Dh, H], nk=true) that the shared dense
// AttnBlock consumes. Inverts vLLM InternLM2Attention.split_qkv (internlm2.py
// :158-176) at the WEIGHT-ROW level for the single-GPU (TP=1) case.
//
// On-disk wqkv is [Hkv*(groups+2)*Dh, H] with rows grouped by kv-head g:
//   g*(groups+2)*Dh + [0 .. groups*Dh)         -> the g query heads (kv-group g)
//   g*(groups+2)*Dh + [groups*Dh .. (groups+1)*Dh) -> the g key head
//   g*(groups+2)*Dh + [(groups+1)*Dh .. (groups+2)*Dh) -> the g value head
// Output merged rows are [ q(all Hq heads, in head order) | k(all Hkv) | v(all
// Hkv) ], which is exactly what QkvSplit(q_size, kv_size, kv_size) then re-splits
// — and the standard GQA head->kv-head map (head i -> i/groups) is preserved.
// RED-test hook (default OFF, byte-inert): VT_INTERNLM2_WRONG_SPLIT=1 makes the
// loader treat `wqkv` as an ALREADY-CONCATENATED [q|k|v] weight (the plain
// non-interleaved split every OTHER dense family uses) instead of de-interleaving
// the kv-group layout. That is the WRONG split for InternLM2 — it scrambles
// query/key/value heads — and the SACRED gate must CATCH it (root divergence
// > 0.5 nats). Used to prove the interleaved de-split is the load-bearing delta.
bool WrongSplitRed() {
  const char* e = std::getenv("VT_INTERNLM2_WRONG_SPLIT");
  return e != nullptr && e[0] == '1';
}

OwnedTensor DeinterleaveWqkv(const StTensor& wqkv, const HfConfig& cfg,
                             const std::string& name) {
  VT_CHECK(wqkv.dtype == "BF16", "internlm2: expected BF16 for " + name);
  VT_CHECK(wqkv.shape.size() == 2, "internlm2: expected 2-D wqkv for " + name);
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int64_t H = wqkv.shape[1];
  VT_CHECK(Hq > 0 && Hkv > 0 && Dh > 0, "internlm2: bad head config");
  VT_CHECK(Hq % Hkv == 0, "internlm2: num_attention_heads must be a multiple of "
                          "num_key_value_heads");
  const int64_t groups = Hq / Hkv;
  const int64_t q_rows = Hq * Dh;
  const int64_t kv_rows = Hkv * Dh;
  const int64_t out_rows = q_rows + 2 * kv_rows;
  VT_CHECK(wqkv.shape[0] == Hkv * (groups + 2) * Dh,
           "internlm2: wqkv row count " + std::to_string(wqkv.shape[0]) +
               " != num_kv_heads*(kv_groups+2)*head_dim for " + name);
  const size_t expected =
      static_cast<size_t>(wqkv.shape[0]) * static_cast<size_t>(H) * sizeof(uint16_t);
  VT_CHECK(wqkv.nbytes == expected, "internlm2: byte-size mismatch for " + name);

  OwnedTensor merged = MakeOwned(vt::DType::kBF16, {out_rows, H});
  // `src` stays BYTE-typed: it points into the mmap'd safetensors payload, whose
  // per-tensor offset is the running total of everything ahead of it and so need
  // not be even, and a `const uint16_t*` onto an odd byte is undefined to form
  // (issue #627). Every read below is a bulk memcpy, so byte offsets — the same
  // arithmetic scaled by sizeof(uint16_t) — copy the identical bytes.
  const uint8_t* src = wqkv.data;
  auto* dst = reinterpret_cast<uint16_t*>(merged.bytes.data());
  const size_t row = static_cast<size_t>(H);        // elems per output row
  const size_t row_bytes = row * sizeof(uint16_t);  // bytes per output row
  if (WrongSplitRed()) {
    // RED path: copy wqkv straight through (naive [q|k|v] concat, NO
    // de-interleave) — the WRONG split; heads end up scrambled.
    std::memcpy(dst, src, static_cast<size_t>(out_rows) * row_bytes);
    MaybeReleaseSourcePages(wqkv.data, wqkv.nbytes);
    merged.nk = true;
    return merged;
  }
  for (int64_t g = 0; g < Hkv; ++g) {
    const int64_t block = g * (groups + 2) * Dh;  // first wqkv row of kv-group g
    // q: groups*Dh rows -> q section at g*groups*Dh
    std::memcpy(dst + static_cast<size_t>(g * groups * Dh) * row,
                src + static_cast<size_t>(block) * row_bytes,
                static_cast<size_t>(groups * Dh) * row_bytes);
    // k: Dh rows -> k section (after q_rows) at g*Dh
    std::memcpy(dst + static_cast<size_t>(q_rows + g * Dh) * row,
                src + static_cast<size_t>(block + groups * Dh) * row_bytes,
                static_cast<size_t>(Dh) * row_bytes);
    // v: Dh rows -> v section (after q_rows+kv_rows) at g*Dh
    std::memcpy(dst + static_cast<size_t>(q_rows + kv_rows + g * Dh) * row,
                src + static_cast<size_t>(block + (groups + 1) * Dh) * row_bytes,
                static_cast<size_t>(Dh) * row_bytes);
  }
  MaybeReleaseSourcePages(wqkv.data, wqkv.nbytes);
  merged.nk = true;  // raw [N=out_rows, K=H] for vt::MatmulBT
  return merged;
}

Qwen3DenseLayerWeights LoadInternLM2Layer(const TensorResolver& get,
                                          const HfConfig& config, int64_t layer) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string at = base + "attention.";
  const std::string ff = base + "feed_forward.";

  Qwen3DenseLayerWeights w;
  w.input_layernorm = LoadBf16Direct(get, base + "attention_norm.weight");
  w.post_attention_layernorm = LoadBf16Direct(get, base + "ffn_norm.weight");

  // The ONE delta: fused wqkv de-interleaved into merged [q|k|v]-row qkv_proj.
  w.attn.qkv_proj = DeinterleaveWqkv(get(at + "wqkv.weight"), config,
                                     at + "wqkv.weight");
  w.attn.o_proj = LoadMergedBf16RawNK(get, {at + "wo.weight"});
  // InternLM2 has NO per-head q/k RMSNorm and NO qkv bias (bias=false).

  w.mlp.gate_up_proj =
      LoadMergedBf16RawNK(get, {ff + "w1.weight", ff + "w3.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {ff + "w2.weight"});
  return w;
}

}  // namespace

InternLM2Weights LoadInternLM2ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "internlm2: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "internlm2: num_hidden_layers must be positive");

  InternLM2Weights w;
  // internlm2 chat checkpoints are UNTIED (separate output.weight).
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", false);
  w.attention_bias = false;  // InternLM2 wqkv/wo carry no bias (config bias=false).

  w.embed_tokens = LoadBf16Direct(get, "model.tok_embeddings.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "output.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadInternLM2Layer(get, config, l));
  return w;
}

}  // namespace vllm
