// H3-Encoder — the ORIGINAL bf16 Qwen3-VL-32B tower, 14 safetensors shards, 63 GB.
//
// WHY THIS EXISTS. Every H3 render so far conditioned on the Q4_K_M encoder, and
// nobody had ever measured what that quantization does to the conditioning tensor
// the DiT actually consumes. The question needs the SAME prompt encoded by the
// UNQUANTIZED tower — and `--encoder` only ever accepted a single GGUF, while the
// bf16 release ships 14 shards. This file is the missing half.
//
// It is a LOADER, not a second forward. `MiniMaxH3EncoderDeviceWeights` binds a
// plain `std::map<std::string, vt::Tensor>`; the GGUF arm fills it with ggml
// blocks, this one fills it with bf16, and `MiniMaxH3EncoderTextForwardDevice`
// runs unchanged over either. Everything downstream of the weight bytes — the
// M-RoPE, the per-head q/k norms, the causal GQA attention, the f32 activations,
// the truncation to min(num_hidden_layers, 50), the UNNORMALIZED output — is
// literally the same code, which is what makes the two arms comparable at all.
//
// THE NAME MAP IS NOT RE-DERIVED. It is the one already gated for the in-tree
// `LoadMiniMaxH3EncoderWeights(const std::vector<SafetensorsFile>&, ...)`:
//   model.language_model.layers.N.*  ->  layers.N.*
//   self_attn.{q,k,v}_proj           ->  self_attn.qkv_proj   ([q|k|v] rows)
//   mlp.{gate,up}_proj               ->  mlp.gate_up_proj     ([gate|up] rows)
// `model.language_model.norm.weight` and `lm_head.weight` are deliberately NOT
// bound: H3 reads the UNNORMALIZED layer-49 output, and binding the final norm
// would imply it is applied. Shard resolution goes through
// `MiniMaxH3ShardedCheckpoint`, i.e. the checkpoint's own index, never a scan.
//
// MEMORY IS THE DESIGN CONSTRAINT. The box has 122 GiB of UNIFIED memory (host
// and device share ONE pool) and a previous non-streaming H3 loader was
// OOM-KILLED at anon-rss 125 GB. So: the projections stay BF16 on the device
// (48.8 GiB for the 50 layers H3 runs, against 97.5 GiB as f32) and are uploaded
// DIRECTLY out of the read-only mmap with no host buffer at all; the row
// concatenations are done ON THE DEVICE by uploading each member into its offset
// of one allocation; and every source range is released the moment its copy
// returns. Only the norms — [5120] each — are widened on the host, because
// vt::RmsNorm takes an f32 weight.
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

const char* const kLmPrefix = "model.language_model.";

// One destination view and the safetensors tensors that make it. `srcs.size() > 1`
// means a ROW CONCATENATION, and the order is load-bearing: the forward slices
// qkv_proj at [0,q) / [q,q+kv) / [q+kv,...), so any other order silently feeds
// keys into the query path — which still runs, and is still wrong.
struct EncoderShardGroup {
  std::string dst;
  std::vector<std::string> srcs;
  bool norm = false;  // f32 on the device (vt::RmsNorm takes an f32 weight)
};

std::string LayerSrc(int64_t layer, const std::string& leaf) {
  return std::string(kLmPrefix) + "layers." + std::to_string(layer) + "." + leaf;
}

// The whole name map in ONE place, so the geometry deriver, the streamer and the
// gate all read the same plan. `max_layers` truncates the text tower, which is
// H3's own behaviour (min(num_hidden_layers, 50) — the release ships 64); 0 keeps
// every layer the index names.
std::vector<EncoderShardGroup> PlanEncoderShardGroups(const MiniMaxH3ShardedCheckpoint& ckpt,
                                                      int64_t max_layers, int64_t* out_layers) {
  std::vector<EncoderShardGroup> plan;
  int64_t layers = 0;
  for (int64_t layer = 0;; ++layer) {
    if (!ckpt.Has(LayerSrc(layer, "input_layernorm.weight"))) break;
    if (max_layers > 0 && layer >= max_layers) break;
    const std::string dst = "layers." + std::to_string(layer) + ".";
    plan.push_back({dst + "input_layernorm.weight", {LayerSrc(layer, "input_layernorm.weight")},
                    true});
    plan.push_back({dst + "post_attention_layernorm.weight",
                    {LayerSrc(layer, "post_attention_layernorm.weight")}, true});
    plan.push_back({dst + "self_attn.q_norm.weight", {LayerSrc(layer, "self_attn.q_norm.weight")},
                    true});
    plan.push_back({dst + "self_attn.k_norm.weight", {LayerSrc(layer, "self_attn.k_norm.weight")},
                    true});
    plan.push_back({dst + "self_attn.qkv_proj.weight",
                    {LayerSrc(layer, "self_attn.q_proj.weight"),
                     LayerSrc(layer, "self_attn.k_proj.weight"),
                     LayerSrc(layer, "self_attn.v_proj.weight")},
                    false});
    plan.push_back({dst + "self_attn.o_proj.weight", {LayerSrc(layer, "self_attn.o_proj.weight")},
                    false});
    plan.push_back({dst + "mlp.gate_up_proj.weight",
                    {LayerSrc(layer, "mlp.gate_proj.weight"), LayerSrc(layer, "mlp.up_proj.weight")},
                    false});
    plan.push_back({dst + "mlp.down_proj.weight", {LayerSrc(layer, "mlp.down_proj.weight")}, false});
    ++layers;
  }
  VT_CHECK(layers > 0,
           "minimax_h3 encoder shards: no text-tower layers were found (expected "
           "model.language_model.layers.0.input_layernorm.weight)");
  if (out_layers != nullptr) *out_layers = layers;
  return plan;
}

// A rank-2 [rows, K] projection's shape, checked.
void ProjShape(const MiniMaxH3ShardedCheckpoint& ckpt, const std::string& name, int64_t* rows,
               int64_t* k) {
  const StTensor& t = ckpt.Get(name);
  VT_CHECK(t.shape.size() == 2,
           "minimax_h3 encoder shards: '" + name + "' is not a rank-2 projection");
  *rows = t.shape[0];
  *k = t.shape[1];
}

}  // namespace

MiniMaxH3EncoderConfig MiniMaxH3EncoderConfigFromShards(const MiniMaxH3ShardedCheckpoint& ckpt,
                                                        int64_t max_layers) {
  int64_t layers = 0;
  (void)PlanEncoderShardGroups(ckpt, max_layers, &layers);

  MiniMaxH3EncoderConfig config;
  config.num_hidden_layers = layers;

  // head_dim comes from q_norm, which is [head_dim] — the same recovery the GGUF
  // loader does, so the two arms cannot disagree on geometry by construction.
  const StTensor& qn = ckpt.Get(LayerSrc(0, "self_attn.q_norm.weight"));
  VT_CHECK(qn.shape.size() == 1, "minimax_h3 encoder shards: q_norm must be rank-1 [head_dim]");
  config.head_dim = qn.shape[0];

  int64_t q_rows = 0, k_rows = 0, hidden = 0, kv_k = 0;
  ProjShape(ckpt, LayerSrc(0, "self_attn.q_proj.weight"), &q_rows, &hidden);
  ProjShape(ckpt, LayerSrc(0, "self_attn.k_proj.weight"), &k_rows, &kv_k);
  VT_CHECK(kv_k == hidden, "minimax_h3 encoder shards: q_proj and k_proj disagree on K");
  config.hidden_size = hidden;
  VT_CHECK(config.head_dim > 0 && q_rows % config.head_dim == 0 && k_rows % config.head_dim == 0,
           "minimax_h3 encoder shards: projection rows are not a multiple of head_dim");
  config.num_attention_heads = q_rows / config.head_dim;
  config.num_key_value_heads = k_rows / config.head_dim;

  int64_t ffn = 0, gate_k = 0;
  ProjShape(ckpt, LayerSrc(0, "mlp.gate_proj.weight"), &ffn, &gate_k);
  VT_CHECK(gate_k == hidden, "minimax_h3 encoder shards: gate_proj K is not hidden_size");
  config.intermediate_size = ffn;
  return config;
}

std::vector<float> MiniMaxH3EncoderEmbedTokensFromShards(const MiniMaxH3ShardedCheckpoint& ckpt,
                                                         const std::vector<int32_t>& ids) {
  const std::string name = std::string(kLmPrefix) + "embed_tokens.weight";
  VT_CHECK(ckpt.Has(name),
           "minimax_h3 encoder shards: checkpoint has no " + name);
  const StTensor& table = ckpt.Get(name);
  VT_CHECK(table.shape.size() == 2,
           "minimax_h3 encoder shards: embed_tokens must be [vocab, hidden]");
  const int64_t vocab = table.shape[0], hidden = table.shape[1];

  VT_CHECK(vocab > 0 && table.nbytes % static_cast<size_t>(vocab) == 0,
           "minimax_h3 encoder shards: embed_tokens byte span does not divide by its rows");
  const size_t row_bytes = table.nbytes / static_cast<size_t>(vocab);

  // ROW AT A TIME out of the mmap. The table is the single largest tensor in the
  // checkpoint ([151936, 5120]) and a prompt touches a few dozen rows, so
  // materializing it to gather them would be the expensive way to the same answer.
  // Each row is handed to the SHARED, already-gated dtype converter as a one-row
  // view rather than getting its own bf16/f16 decode here.
  std::vector<float> out(ids.size() * static_cast<size_t>(hidden));
  for (size_t i = 0; i < ids.size(); ++i) {
    const int64_t id = ids[i];
    VT_CHECK(id >= 0 && id < vocab,
             "minimax_h3 encoder shards: token id out of vocabulary range");
    StTensor row;
    row.dtype = table.dtype;
    row.shape = {hidden};
    row.data = table.data + static_cast<size_t>(id) * row_bytes;
    row.nbytes = row_bytes;
    const std::vector<float> values = MiniMaxH3ReadSafetensorF32(row);
    VT_CHECK(static_cast<int64_t>(values.size()) == hidden,
             "minimax_h3 encoder shards: an embedding row decoded to the wrong width");
    std::memcpy(out.data() + i * static_cast<size_t>(hidden), values.data(),
                static_cast<size_t>(hidden) * sizeof(float));
  }
  return out;
}

MiniMaxH3EncoderDeviceWeights StreamMiniMaxH3EncoderShardsToDevice(
    vt::Queue& queue, const MiniMaxH3ShardedCheckpoint& ckpt, int64_t max_layers,
    MiniMaxH3EncoderConfig* out_config) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  int64_t layers = 0;
  const std::vector<EncoderShardGroup> plan = PlanEncoderShardGroups(ckpt, max_layers, &layers);
  if (out_config != nullptr) *out_config = MiniMaxH3EncoderConfigFromShards(ckpt, max_layers);

  MiniMaxH3EncoderShardStreamStats& stats = MutableMiniMaxH3EncoderShardStreamStats();
  stats = MiniMaxH3EncoderShardStreamStats{};
  stats.shards_opened = static_cast<uint64_t>(ckpt.ShardCount());
  stats.layers_streamed = static_cast<uint64_t>(layers);

  MiniMaxH3EncoderDeviceWeights out;
  for (const EncoderShardGroup& group : plan) {
    // Shape of the destination: rows are SUMMED over the group, K is shared.
    int64_t rows = 0, k = -1;
    bool all_bf16 = true;
    std::vector<int64_t> member_rows;
    member_rows.reserve(group.srcs.size());
    for (const std::string& src : group.srcs) {
      const StTensor& t = ckpt.Get(src);
      VT_CHECK(!t.shape.empty(), "minimax_h3 encoder shards: '" + src + "' has no shape");
      const int64_t r = t.shape[0];
      const int64_t kk = t.shape.size() == 2 ? t.shape[1] : -1;
      if (k == -1) {
        k = kk;
      } else {
        VT_CHECK(kk == k, "minimax_h3 encoder shards: fused group '" + group.dst +
                              "' disagrees on K");
      }
      if (t.dtype != "BF16") all_bf16 = false;
      member_rows.push_back(r);
      rows += r;
    }

    // Norms go to f32 (vt::RmsNorm's contract). Projections stay BF16 when the
    // shards store bf16 — the whole point, since that is the 48.8 GiB residency
    // and the no-host-copy upload — and are widened to f32 only if the shards
    // store something else, so no precision is ever ROUNDED AWAY by this loader.
    const vt::DType dtype = (group.norm || !all_bf16) ? vt::DType::kF32 : vt::DType::kBF16;
    const size_t elem = vt::SizeOf(dtype);
    const int64_t numel = k >= 0 ? rows * k : rows;
    const size_t bytes = static_cast<size_t>(numel) * elem;

    void* base = backend.Alloc(bytes);
    std::shared_ptr<void> owner(base, [&backend](void* p) { backend.Free(p); });

    size_t offset = 0;
    for (size_t m = 0; m < group.srcs.size(); ++m) {
      const StTensor& t = ckpt.Get(group.srcs[m]);
      const size_t member_numel =
          static_cast<size_t>(member_rows[m]) * static_cast<size_t>(k >= 0 ? k : 1);
      const size_t member_bytes = member_numel * elem;
      VT_CHECK(offset + member_bytes <= bytes,
               "minimax_h3 encoder shards: '" + group.dst + "' member overruns its allocation");
      // DECLARED OUTSIDE the branch on purpose: vt::Backend::Copy is
      // cudaMemcpyAsync, so a conversion buffer that dies at the end of its own
      // block is a use-after-free the stream may or may not have read yet. This
      // codebase has been bitten by exactly that (uploads from function-local
      // temporaries), so the buffer outlives the Synchronize below.
      std::vector<float> host;
      if (dtype == vt::DType::kBF16) {
        // The on-disk bytes ARE the device bytes: no host buffer exists at any
        // point, which is what keeps peak host memory flat across a 63 GB load.
        VT_CHECK(t.nbytes == member_bytes,
                 "minimax_h3 encoder shards: '" + group.srcs[m] +
                     "' byte span does not match its shape");
        backend.Copy(queue, static_cast<uint8_t*>(base) + offset, t.data, member_bytes);
        ++stats.direct_uploads;
      } else {
        host = MiniMaxH3ReadSafetensorF32(t);
        VT_CHECK(host.size() == member_numel,
                 "minimax_h3 encoder shards: '" + group.srcs[m] +
                     "' read produced the wrong element count");
        backend.Copy(queue, static_cast<uint8_t*>(base) + offset, host.data(), member_bytes);
        ++stats.converted_uploads;
        const size_t host_bytes = host.size() * sizeof(float);
        if (host_bytes > stats.host_peak_bytes) stats.host_peak_bytes = host_bytes;
      }
      // The SOURCE — mmap range or conversion buffer — must stay live until the
      // copy has landed, and `host` must not be freed before this returns.
      backend.Synchronize(queue);
      MaybeReleaseSourcePages(t.data, t.nbytes);
      offset += member_bytes;
      stats.bytes_uploaded += member_bytes;
    }
    VT_CHECK(offset == bytes,
             "minimax_h3 encoder shards: '" + group.dst + "' did not fill its allocation");
    if (group.srcs.size() > 1) ++stats.fused_groups;

    std::vector<int64_t> shape;
    if (k >= 0) {
      shape = {rows, k};
    } else {
      shape = {rows};
    }
    out.views[group.dst] = dense_attn::MakeTensor(base, dtype, queue.device, shape);
    out.storage.push_back(std::move(owner));
    ++stats.tensors_streamed;
  }
  backend.Synchronize(queue);
  return out;
}

}  // namespace vllm
