// GLM-5.3-Flash W5b-1 — the `OwnedTensor` -> host f32 bridge. See
// `glm5_next_bridge.h` for the residency decision and its arithmetic.
#include "vllm/model_executor/models/glm5_next_bridge.h"

#include <cstring>
#include <stdexcept>

#include "vt/dtype.h"
#include "vt/quant.h"

namespace vllm::glm5_next {
namespace {

[[noreturn]] void Fail(const std::string& what, const std::string& why) {
  throw std::runtime_error("glm5_next bridge: `" + what + "` " + why);
}

int64_t Numel(const OwnedTensor& t) {
  if (t.rank <= 0) return 0;
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
  return n;
}

std::string ShapeStr(const OwnedTensor& t) {
  std::string s = "[";
  for (int i = 0; i < t.rank; ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(t.shape[i]);
  }
  return s + "]";
}

std::string WantStr(const std::vector<int64_t>& want) {
  std::string s = "[";
  for (size_t i = 0; i < want.size(); ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(want[i]);
  }
  return s + "]";
}

// Decode `t` and require its shape to be exactly `want`. The shape check comes
// FIRST: a tensor that decodes to the right element COUNT at the wrong rank is
// the silent case, and `k_b_proj` at `[H, qk_nope, kv_lora]` instead of
// `[H, kv_lora, qk_nope]` is exactly it whenever the two widths coincide.
std::vector<float> DecodeShaped(const OwnedTensor& t, const std::string& what,
                                const std::vector<int64_t>& want,
                                int64_t byte_ceiling) {
  if (static_cast<size_t>(t.rank) != want.size()) {
    Fail(what, "has rank " + std::to_string(t.rank) + ", expected " +
                   std::to_string(want.size()) + " for shape " + WantStr(want));
  }
  for (size_t i = 0; i < want.size(); ++i) {
    if (t.shape[i] != want[i]) {
      Fail(what, "has shape " + ShapeStr(t) + ", expected " + WantStr(want));
    }
  }
  return DecodeOwnedTensorToF32(t, what, byte_ceiling);
}

}  // namespace

int64_t HostF32Bytes(const OwnedTensor& t) {
  return Numel(t) * static_cast<int64_t>(sizeof(float));
}

int64_t BridgedDsaLayerF32Bytes(const MlaDims& d, const IndexerDims& id) {
  d.Validate();
  id.Validate();
  const int64_t mla =
      d.q_lora_rank * d.hidden_size                              // q_a_proj
      + d.q_lora_rank                                            // q_a_layernorm
      + d.num_heads * d.qk_head_dim() * d.q_lora_rank            // q_b_proj
      + (d.kv_lora_rank + d.qk_rope_head_dim) * d.hidden_size    // kv_a_proj
      + d.kv_lora_rank                                           // kv_a_layernorm
      + d.num_heads * d.kv_lora_rank * d.qk_nope_head_dim        // k_b_proj
      + d.num_heads * d.v_head_dim * d.kv_lora_rank              // v_b_proj
      + d.hidden_size * d.num_heads * d.v_head_dim;              // o_proj
  const int64_t idx = id.n_heads * id.head_dim * id.q_lora_rank  // wq_b
                      + id.head_dim * id.hidden_size             // wk
                      + id.head_dim                              // k_norm.weight
                      + id.head_dim                              // k_norm.bias
                      + id.n_heads * id.hidden_size              // weights_proj
                      + id.index_kpool * id.head_dim             // kpool_ape
                      + id.head_dim * id.hidden_size;            // kpool_gate
  return (mla + idx) * static_cast<int64_t>(sizeof(float));
}

std::vector<float> DecodeOwnedTensorToF32(const OwnedTensor& t,
                                          const std::string& what,
                                          int64_t byte_ceiling) {
  const int64_t numel = Numel(t);
  if (t.rank <= 0 || numel <= 0) {
    Fail(what, "has no elements (rank " + std::to_string(t.rank) +
                   ", shape " + ShapeStr(t) + "); the loader never filled it");
  }
  // The ceiling is checked BEFORE the allocation, from the shape alone. A
  // check after the fact would already have taken the 9 GiB it exists to
  // refuse.
  const int64_t want_bytes = HostF32Bytes(t);
  if (want_bytes > byte_ceiling) {
    Fail(what, "would need " + std::to_string(want_bytes) +
                   " bytes as host f32, over this bridge's " +
                   std::to_string(byte_ceiling) +
                   "-byte ceiling. The bridge mirrors ONE DSA layer's "
                   "attention weights; a tensor this large is an expert bank, "
                   "and decoding the tower does not fit any device this "
                   "project reaches (see glm5_next_bridge.h)");
  }
  if (t.host_released) {
    Fail(what, "had its host bytes released; a device-resident weight cannot "
               "be bridged to a host f32 reference");
  }
  if (t.bytes.empty()) Fail(what, "carries no bytes");

  std::vector<float> out(static_cast<size_t>(numel));
  const uint8_t* src = t.bytes.data();

  if (vt::IsBlockQuant(t.dtype)) {
    const int64_t elems = vt::BlockElems(t.dtype);
    if (numel % elems != 0) {
      Fail(what, std::string("has ") + std::to_string(numel) +
                     " elements, which is not a whole number of " +
                     vt::Name(t.dtype) + " blocks of " + std::to_string(elems));
    }
    const size_t need = vt::RowSizeBytes(t.dtype, numel);
    if (t.bytes.size() != need) {
      Fail(what, std::string("holds ") + std::to_string(t.bytes.size()) +
                     " bytes, but " + std::to_string(numel) + " " +
                     vt::Name(t.dtype) + " elements need " +
                     std::to_string(need));
    }
    const vt::cpu::ToFloatFn to_float = vt::cpu::BlockToFloat(t.dtype);
    if (to_float == nullptr) {
      // Not a hypothetical: this build gained IQ2_XS and IQ4_XS only at #2245,
      // and the refusal is what keeps a missing decoder from reading as zeros.
      Fail(what, std::string("is ") + vt::Name(t.dtype) +
                     ", which this build has no `BlockToFloat` decoder for");
    }
    to_float(src, out.data(), numel);
    return out;
  }

  const size_t need = static_cast<size_t>(numel) * vt::SizeOf(t.dtype);
  if (t.bytes.size() != need) {
    Fail(what, std::string("holds ") + std::to_string(t.bytes.size()) +
                   " bytes, but " + std::to_string(numel) + " " +
                   vt::Name(t.dtype) + " elements need " +
                   std::to_string(need));
  }
  switch (t.dtype) {
    case vt::DType::kF32:
      std::memcpy(out.data(), src, need);
      return out;
    case vt::DType::kBF16: {
      const auto* p = reinterpret_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < numel; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
      return out;
    }
    case vt::DType::kF16: {
      const auto* p = reinterpret_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < numel; ++i) out[static_cast<size_t>(i)] = vt::F16ToF32(p[i]);
      return out;
    }
    default:
      Fail(what, std::string("is ") + vt::Name(t.dtype) +
                     ", which is not a float encoding this bridge can widen");
  }
}

IndexerWeights BridgedDsaLayer::IndexerView() const {
  IndexerWeights w;
  w.wq_b = idx_wq_b.data();
  w.wk = idx_wk.data();
  w.k_norm_weight = idx_k_norm_weight.data();
  w.k_norm_bias = idx_k_norm_bias.data();
  w.weights_proj = idx_weights_proj.data();
  w.kpool_ape = idx_kpool_ape.data();
  w.kpool_gate = idx_kpool_gate.data();
  return w;
}

BridgedDsaLayer BridgeDsaLayer(const Glm5NextMlaWeights& src, const MlaDims& d,
                               const IndexerDims& id, int64_t byte_ceiling) {
  d.Validate();
  id.Validate();
  if (id.hidden_size != d.hidden_size || id.q_lora_rank != d.q_lora_rank) {
    throw std::runtime_error(
        "glm5_next bridge: the indexer dims disagree with the MLA dims "
        "(hidden_size " +
        std::to_string(id.hidden_size) + " vs " + std::to_string(d.hidden_size) +
        ", q_lora_rank " + std::to_string(id.q_lora_rank) + " vs " +
        std::to_string(d.q_lora_rank) + "); they come from ONE config");
  }

  BridgedDsaLayer out;
  out.mla.q_a_proj = DecodeShaped(src.q_a_proj, "q_a_proj",
                                  {d.q_lora_rank, d.hidden_size}, byte_ceiling);
  out.mla.q_a_layernorm = DecodeShaped(src.q_a_layernorm, "q_a_layernorm",
                                       {d.q_lora_rank}, byte_ceiling);
  out.mla.q_b_proj =
      DecodeShaped(src.q_b_proj, "q_b_proj",
                   {d.num_heads * d.qk_head_dim(), d.q_lora_rank}, byte_ceiling);
  out.mla.kv_a_proj_with_mqa = DecodeShaped(
      src.kv_a_proj_with_mqa, "kv_a_proj_with_mqa",
      {d.kv_lora_rank + d.qk_rope_head_dim, d.hidden_size}, byte_ceiling);
  out.mla.kv_a_layernorm = DecodeShaped(src.kv_a_layernorm, "kv_a_layernorm",
                                        {d.kv_lora_rank}, byte_ceiling);
  // The two absorbed halves, each at the CONVERTER's own shape. The k half is
  // transposed and the v half is not, so these two `want` vectors are the
  // whole of trap 1 and are deliberately not derived from one another.
  out.mla.k_b_proj =
      DecodeShaped(src.k_b_proj, "k_b_proj",
                   {d.num_heads, d.kv_lora_rank, d.qk_nope_head_dim}, byte_ceiling);
  out.mla.v_b_proj =
      DecodeShaped(src.v_b_proj, "v_b_proj",
                   {d.num_heads, d.v_head_dim, d.kv_lora_rank}, byte_ceiling);
  out.mla.o_proj =
      DecodeShaped(src.o_proj, "o_proj",
                   {d.hidden_size, d.num_heads * d.v_head_dim}, byte_ceiling);

  const Glm5NextIndexerWeights& ix = src.indexer;
  out.idx_wq_b = DecodeShaped(ix.wq_b, "indexer.wq_b",
                              {id.n_heads * id.head_dim, id.q_lora_rank},
                              byte_ceiling);
  out.idx_wk = DecodeShaped(ix.wk, "indexer.wk", {id.head_dim, id.hidden_size},
                            byte_ceiling);
  out.idx_k_norm_weight = DecodeShaped(ix.k_norm_weight, "indexer.k_norm.weight",
                                       {id.head_dim}, byte_ceiling);
  // The BIAS is what makes `k_norm` a LayerNorm and not an RMSNorm; a bridge
  // that dropped it would run and be wrong by a constant per channel.
  out.idx_k_norm_bias = DecodeShaped(ix.k_norm_bias, "indexer.k_norm.bias",
                                     {id.head_dim}, byte_ceiling);
  out.idx_weights_proj = DecodeShaped(ix.weights_proj, "indexer.weights_proj",
                                      {id.n_heads, id.hidden_size}, byte_ceiling);
  out.idx_kpool_ape =
      DecodeShaped(ix.kpool_ape, "indexer.index_kpool_compress_ape",
                   {id.index_kpool, id.head_dim}, byte_ceiling);
  out.idx_kpool_gate =
      DecodeShaped(ix.kpool_gate, "indexer.index_kpool_compress_gate",
                   {id.head_dim, id.hidden_size}, byte_ceiling);

  // MEASURED from the decoded buffers, not predicted from the dims: the two
  // agreeing is what makes `BridgedDsaLayerF32Bytes` a budget a caller can
  // trust before it allocates.
  const auto add = [&out](const std::vector<float>& v) {
    out.host_f32_bytes +=
        static_cast<int64_t>(v.size()) * static_cast<int64_t>(sizeof(float));
  };
  add(out.mla.q_a_proj);
  add(out.mla.q_a_layernorm);
  add(out.mla.q_b_proj);
  add(out.mla.kv_a_proj_with_mqa);
  add(out.mla.kv_a_layernorm);
  add(out.mla.k_b_proj);
  add(out.mla.v_b_proj);
  add(out.mla.o_proj);
  add(out.idx_wq_b);
  add(out.idx_wk);
  add(out.idx_k_norm_weight);
  add(out.idx_k_norm_bias);
  add(out.idx_weights_proj);
  add(out.idx_kpool_ape);
  add(out.idx_kpool_gate);
  return out;
}

}  // namespace vllm::glm5_next
