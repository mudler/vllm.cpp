// GLM-5.3-Flash W5b-1 — the `OwnedTensor` -> host f32 bridge, and W5b-2b's
// four remaining arms plus the per-expert source. See `glm5_next_bridge.h` for
// the residency decision and its arithmetic, and `glm5_next_moe.h` for the
// per-expert half of it.
#include "vllm/model_executor/models/glm5_next_bridge.h"

#include <algorithm>
#include <cstddef>
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

int64_t HostF32RowBytes(const OwnedTensor& t) {
  if (t.rank <= 0 || t.shape[0] <= 0) return 0;
  return (Numel(t) / t.shape[0]) * static_cast<int64_t>(sizeof(float));
}

std::vector<float> DecodeOwnedTensorRowsToF32(const OwnedTensor& t,
                                              const std::string& what,
                                              int64_t first_row, int64_t num_rows,
                                              int64_t byte_ceiling) {
  const int64_t numel = Numel(t);
  if (t.rank <= 0 || numel <= 0) {
    Fail(what, "has no elements (rank " + std::to_string(t.rank) + ", shape " +
                   ShapeStr(t) + "); the loader never filled it");
  }
  const int64_t rows = t.shape[0];
  if (rows <= 0) Fail(what, "has a zero-length leading axis; there is no row to slice");
  if (first_row < 0 || num_rows <= 0 || first_row + num_rows > rows) {
    Fail(what, "was asked for rows [" + std::to_string(first_row) + ", " +
                   std::to_string(first_row + num_rows) + ") of " +
                   std::to_string(rows) + " in shape " + ShapeStr(t));
  }
  const int64_t row_elems = numel / rows;
  const int64_t want_elems = row_elems * num_rows;
  // The CEILING IS THE RANGE'S, checked before anything is allocated and
  // against the SAME `byte_ceiling` a whole-tensor decode is checked against.
  // Asking for every row is therefore refused by exactly the arithmetic that
  // refuses the whole tensor, so this function is not a way around it.
  const int64_t want_bytes = want_elems * static_cast<int64_t>(sizeof(float));
  if (want_bytes > byte_ceiling) {
    Fail(what, "rows [" + std::to_string(first_row) + ", " +
                   std::to_string(first_row + num_rows) + ") would need " +
                   std::to_string(want_bytes) +
                   " bytes as host f32, over this bridge's " +
                   std::to_string(byte_ceiling) +
                   "-byte ceiling. Decode FEWER rows: the whole point of a row "
                   "range is that a caller never holds a bank (see "
                   "glm5_next_bridge.h and glm5_next_moe.h)");
  }
  if (t.host_released) {
    Fail(what, "had its host bytes released; a device-resident weight cannot be "
               "bridged to a host f32 reference");
  }
  if (t.bytes.empty()) Fail(what, "carries no bytes");
  // A LAYOUT THIS DECODER CANNOT READ IS REFUSED BY NAME, because it cannot be
  // detected any other way. `OwnGgufQuantBlocks` permutes an eligible q8_0
  // weight into the `block_q8_0x4` i8mm interleave when
  // `GgufLoadPolicy::quant_repack` is set -- which it is on every aarch64 i8mm
  // box this row runs on -- and it leaves the DTYPE at `kQ8_0` and the byte
  // count UNCHANGED, since `sizeof(BlockQ8_0x4) == 4 * sizeof(BlockQ8_0)`. So
  // every span and block check below passes, `BlockToFloat(kQ8_0)` reads the
  // interleave as plain blocks, and this function returns finite, plausible,
  // WRONG floats. The same holds for the CUDA coalesced-load layout and for the
  // elementwise `[N,K] -> [K,N]` transpose, whose own field comment says a
  // consumer that ignores it "would read garbage". This bridge is exactly such
  // a consumer. See `test_glm5_next_bridge.cpp` and issue #2241.
  if (t.repacked || t.q8_0_aligned || t.elem_kn_repacked) {
    Fail(what, std::string("was repacked at load into a layout this host f32 "
                           "bridge cannot read (") +
                   (t.repacked ? "repacked " : "") +
                   (t.q8_0_aligned ? "q8_0_aligned " : "") +
                   (t.elem_kn_repacked ? "elem_kn_repacked " : "") +
                   "set). The bytes are a permutation, the dtype and the byte "
                   "count are unchanged, and decoding them as plain blocks "
                   "returns wrong values with no error. The loader must not "
                   "request a repack for a weight this bridge decodes");
  }

  std::vector<float> out(static_cast<size_t>(want_elems));
  if (vt::IsBlockQuant(t.dtype)) {
    const int64_t elems = vt::BlockElems(t.dtype);
    // A ROW that is not a whole number of blocks would make every slice after
    // the first start MID-BLOCK. A decoder handed a misaligned base does not
    // fail: it reads the next block's scale and returns plausible values from
    // the wrong quantization, which is the silent shape this whole file exists
    // to refuse.
    if (row_elems % elems != 0) {
      Fail(what, std::string("has ") + std::to_string(row_elems) +
                     " elements per row, which is not a whole number of " +
                     vt::Name(t.dtype) + " blocks of " + std::to_string(elems) +
                     "; a row slice would start mid-block");
    }
    const size_t need = vt::RowSizeBytes(t.dtype, numel);
    if (t.bytes.size() != need) {
      Fail(what, std::string("holds ") + std::to_string(t.bytes.size()) +
                     " bytes, but " + std::to_string(numel) + " " +
                     vt::Name(t.dtype) + " elements need " + std::to_string(need));
    }
    const vt::cpu::ToFloatFn to_float = vt::cpu::BlockToFloat(t.dtype);
    if (to_float == nullptr) {
      Fail(what, std::string("is ") + vt::Name(t.dtype) +
                     ", which this build has no `BlockToFloat` decoder for");
    }
    to_float(t.bytes.data() + vt::RowSizeBytes(t.dtype, first_row * row_elems),
             out.data(), want_elems);
    return out;
  }

  const size_t esz = vt::SizeOf(t.dtype);
  const size_t need = static_cast<size_t>(numel) * esz;
  if (t.bytes.size() != need) {
    Fail(what, std::string("holds ") + std::to_string(t.bytes.size()) +
                   " bytes, but " + std::to_string(numel) + " " +
                   vt::Name(t.dtype) + " elements need " + std::to_string(need));
  }
  const uint8_t* src = t.bytes.data() + static_cast<size_t>(first_row * row_elems) * esz;
  switch (t.dtype) {
    case vt::DType::kF32:
      std::memcpy(out.data(), src, static_cast<size_t>(want_elems) * esz);
      return out;
    case vt::DType::kBF16: {
      const auto* p = reinterpret_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < want_elems; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
      return out;
    }
    case vt::DType::kF16: {
      const auto* p = reinterpret_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < want_elems; ++i) out[static_cast<size_t>(i)] = vt::F16ToF32(p[i]);
      return out;
    }
    default:
      Fail(what, std::string("is ") + vt::Name(t.dtype) +
                     ", which is not a float encoding this bridge can widen");
  }
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
  // A LAYOUT THIS DECODER CANNOT READ IS REFUSED BY NAME, because it cannot be
  // detected any other way. `OwnGgufQuantBlocks` permutes an eligible q8_0
  // weight into the `block_q8_0x4` i8mm interleave when
  // `GgufLoadPolicy::quant_repack` is set -- which it is on every aarch64 i8mm
  // box this row runs on -- and it leaves the DTYPE at `kQ8_0` and the byte
  // count UNCHANGED, since `sizeof(BlockQ8_0x4) == 4 * sizeof(BlockQ8_0)`. So
  // every span and block check below passes, `BlockToFloat(kQ8_0)` reads the
  // interleave as plain blocks, and this function returns finite, plausible,
  // WRONG floats. The same holds for the CUDA coalesced-load layout and for the
  // elementwise `[N,K] -> [K,N]` transpose, whose own field comment says a
  // consumer that ignores it "would read garbage". This bridge is exactly such
  // a consumer. See `test_glm5_next_bridge.cpp` and issue #2241.
  if (t.repacked || t.q8_0_aligned || t.elem_kn_repacked) {
    Fail(what, std::string("was repacked at load into a layout this host f32 "
                           "bridge cannot read (") +
                   (t.repacked ? "repacked " : "") +
                   (t.q8_0_aligned ? "q8_0_aligned " : "") +
                   (t.elem_kn_repacked ? "elem_kn_repacked " : "") +
                   "set). The bytes are a permutation, the dtype and the byte "
                   "count are unchanged, and decoding them as plain blocks "
                   "returns wrong values with no error. The loader must not "
                   "request a repack for a weight this bridge decodes");
  }

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

// ─── the other four arms (W5b-2b) ────────────────────────────────────────────

glm5_next_kda::Glm5NextKdaLayerWeights BridgeKdaLayer(
    const Glm5NextKdaWeights& src, const glm5_next_kda::Glm5NextKdaDims& d,
    int64_t byte_ceiling) {
  if (d.hidden_size <= 0 || d.num_heads <= 0 || d.head_dim <= 0 ||
      d.conv_kernel_size <= 0) {
    throw std::runtime_error(
        "glm5_next bridge: the KDA dims are incomplete (hidden_size " +
        std::to_string(d.hidden_size) + ", num_heads " +
        std::to_string(d.num_heads) + ", head_dim " + std::to_string(d.head_dim) +
        ", conv_kernel_size " + std::to_string(d.conv_kernel_size) +
        "); they come from ONE config");
  }
  const int64_t H = d.hidden_size;
  const int64_t qkv = d.qkv_dim();
  const int64_t hd = d.head_dim;
  const int64_t heads = d.num_heads;
  const int64_t kk = d.conv_kernel_size;

  glm5_next_kda::Glm5NextKdaLayerWeights out;
  out.q_proj = DecodeShaped(src.q_proj, "kda.q_proj", {qkv, H}, byte_ceiling);
  out.k_proj = DecodeShaped(src.k_proj, "kda.k_proj", {qkv, H}, byte_ceiling);
  out.v_proj = DecodeShaped(src.v_proj, "kda.v_proj", {qkv, H}, byte_ceiling);
  // The THREE convs stay separate here and are concatenated inside the layer,
  // in the q, k, v order the reference's single grouped `nn.Conv1d` declares.
  // The loader already dropped the file's middle `[C, 1, K]` axis, so these are
  // rank 2 and a rank-3 tensor is a named refusal rather than a silent reshape.
  out.q_conv1d = DecodeShaped(src.q_conv1d, "kda.q_conv1d", {qkv, kk}, byte_ceiling);
  out.k_conv1d = DecodeShaped(src.k_conv1d, "kda.k_conv1d", {qkv, kk}, byte_ceiling);
  out.v_conv1d = DecodeShaped(src.v_conv1d, "kda.v_conv1d", {qkv, kk}, byte_ceiling);
  out.f_a_proj = DecodeShaped(src.f_a_proj, "kda.f_a_proj", {hd, H}, byte_ceiling);
  out.f_b_proj = DecodeShaped(src.f_b_proj, "kda.f_b_proj", {qkv, hd}, byte_ceiling);
  out.g_a_proj = DecodeShaped(src.g_a_proj, "kda.g_a_proj", {hd, H}, byte_ceiling);
  out.g_b_proj = DecodeShaped(src.g_b_proj, "kda.g_b_proj", {qkv, hd}, byte_ceiling);
  // ONE ROW PER HEAD (`:618`), not per channel. `[qkv, H]` is shape-valid for a
  // port that read it as a per-channel projection and is 128x too many rows.
  out.b_proj = DecodeShaped(src.b_proj, "kda.b_proj", {heads, H}, byte_ceiling);
  out.a_log = DecodeShaped(src.a_log, "kda.a_log", {heads}, byte_ceiling);
  out.dt_bias = DecodeShaped(src.dt_bias, "kda.dt_bias", {qkv}, byte_ceiling);
  out.o_norm = DecodeShaped(src.o_norm, "kda.o_norm", {hd}, byte_ceiling);
  out.o_proj = DecodeShaped(src.o_proj, "kda.o_proj", {H, qkv}, byte_ceiling);
  return out;
}

DenseMlpWeights BridgeMlp(const Glm5NextMlpWeights& src, int64_t hidden,
                          int64_t intermediate, const std::string& what,
                          int64_t byte_ceiling) {
  if (hidden <= 0 || intermediate <= 0) {
    throw std::runtime_error("glm5_next bridge: `" + what +
                             "` needs positive dims; got hidden " +
                             std::to_string(hidden) + " intermediate " +
                             std::to_string(intermediate));
  }
  DenseMlpWeights out;
  out.gate_proj = DecodeShaped(src.gate_proj, what + ".gate_proj",
                               {intermediate, hidden}, byte_ceiling);
  out.up_proj = DecodeShaped(src.up_proj, what + ".up_proj",
                             {intermediate, hidden}, byte_ceiling);
  out.down_proj = DecodeShaped(src.down_proj, what + ".down_proj",
                               {hidden, intermediate}, byte_ceiling);
  return out;
}

HcSite BridgeMhcSite(const Glm5NextMhcWeights& src, const Glm5NextMhcParams& mhc,
                     int64_t hidden, const std::string& what,
                     int64_t byte_ceiling) {
  if (mhc.mult <= 0 || hidden <= 0) {
    throw std::runtime_error(
        "glm5_next bridge: `" + what + "` needs a positive hc_mult and hidden; got " +
        std::to_string(mhc.mult) + " and " + std::to_string(hidden));
  }
  // `mix = (2 + hc_mult) * hc_mult` is NONLINEAR in `hc_mult`, so a port that
  // read the width as `hc_mult * hc_mult` or as `2 * hc_mult` agrees at no
  // published value and the shape check is what says so.
  const int64_t mix = (2 + mhc.mult) * mhc.mult;
  HcSite out;
  out.fn = DecodeShaped(src.fn, what + ".fn", {mix, mhc.mult * hidden}, byte_ceiling);
  out.base = DecodeShaped(src.base, what + ".base", {mix}, byte_ceiling);
  // [3] — the pre, post and comb gains, in that order.
  out.scale = DecodeShaped(src.scale, what + ".scale", {3}, byte_ceiling);
  return out;
}

// ─── the MoE arm: the router resident, the banks NEVER ───────────────────────

GgufExpertSource::GgufExpertSource(const Glm5NextMoeWeights& src, const MoeDims& d,
                                   const std::string& what, int64_t byte_ceiling)
    : src_(&src), d_(d), what_(what), byte_ceiling_(byte_ceiling) {
  d_.Validate();
}

void GgufExpertSource::Expert(int64_t e, std::vector<float>& gate_up,
                              std::vector<float>& down) {
  if (e < 0 || e >= d_.n_routed_experts) {
    throw std::runtime_error("glm5_next bridge: `" + what_ +
                             "` was asked for expert " + std::to_string(e) +
                             ", which is outside [0, " +
                             std::to_string(d_.n_routed_experts) + ")");
  }
  decoded_.push_back(e);
  const int64_t I = d_.moe_intermediate_size;
  const int64_t H = d_.hidden_size;

  // ONE row of each bank, and the two halves are FUSED here into the
  // `[2I, H]` gate-first row the seam declares. The file stores `gate_exps` and
  // `up_exps` separately (`Glm5NextMoeWeights`), so the fusion is this bridge's
  // and not the loader's, and doing it per expert is what keeps the peak at one
  // expert instead of one bank.
  const std::vector<float> g = DecodeOwnedTensorRowsToF32(
      src_->gate_exps, what_ + ".gate_exps", e, 1, byte_ceiling_);
  const std::vector<float> u = DecodeOwnedTensorRowsToF32(
      src_->up_exps, what_ + ".up_exps", e, 1, byte_ceiling_);
  if (static_cast<int64_t>(g.size()) != I * H ||
      static_cast<int64_t>(u.size()) != I * H) {
    throw std::runtime_error(
        "glm5_next bridge: `" + what_ + "` expert " + std::to_string(e) +
        " decoded " + std::to_string(g.size()) + " gate and " +
        std::to_string(u.size()) + " up floats, expected " +
        std::to_string(I * H) + " each ([" + std::to_string(I) + ", " +
        std::to_string(H) + "])");
  }
  gate_up.resize(static_cast<size_t>(2 * I * H));
  std::copy(g.begin(), g.end(), gate_up.begin());
  std::copy(u.begin(), u.end(), gate_up.begin() + static_cast<ptrdiff_t>(I * H));

  down = DecodeOwnedTensorRowsToF32(src_->down_exps, what_ + ".down_exps", e, 1,
                                    byte_ceiling_);
  if (static_cast<int64_t>(down.size()) != H * I) {
    throw std::runtime_error(
        "glm5_next bridge: `" + what_ + "` expert " + std::to_string(e) +
        " decoded " + std::to_string(down.size()) + " down floats, expected " +
        std::to_string(H * I) + " ([" + std::to_string(H) + ", " +
        std::to_string(I) + "])");
  }
}

// ─── W9a: the keep-quant expert banks, admitted or declined or refused ───────
//
// THREE outcomes, and collapsing them to two is how this goes wrong.
//
//   admitted  -> returns true, fills `*out`. The three banks are block-quantized,
//                gate and up share a dtype, none is repacked, and the shapes are
//                the stacked towers the seam declares.
//   declined  -> returns false, `*out` untouched. The banks are NOT block-quantized,
//                which is the loader's own bf16 expansion route (`LoadStackedExperts`
//                falls to `ExpandBf16` when the policy says so). That is a designed
//                residency and not a fault, so the f32 arm runs and says nothing.
//   refused   -> THROWS. The banks ARE block-quantized and the seam cannot represent
//                them. Silently falling back here would be the failure this row's
//                headers keep naming: correct tokens at f32-decode speed, on a path
//                nobody asked for, which no token gate can see.
//
// THE REPACK CHECK IS NOT DEFENSIVE. `Tensor::repacked` marks the i8mm block
// reordering, which preserves BOTH the dtype and the byte count — so a repacked
// bank fed to a kernel that does not expect it produces finite, wrongly-ordered
// numbers, and every shape and dtype assertion in this file passes. This model
// declines the repack at load (`kGlm5NextQuantRepack = false`,
// `glm5_next_loader.cpp:145`), and this check is what makes that a fact the seam
// depends on rather than a constant a later edit can flip silently.
bool AdmitMoeQuantBanks(const Glm5NextMoeWeights& src, const MoeDims& d,
                        const std::string& what, MoeQuantBanks* out) {
  const OwnedTensor* banks[3] = {&src.gate_exps, &src.up_exps, &src.down_exps};
  const char* names[3] = {".gate_exps", ".up_exps", ".down_exps"};

  // Declined: not a keep-quant residency at all. All three move together,
  // because `LoadStackedExperts` routes them with one policy call each and a
  // split would mean the seam ran on some layers and not others.
  int quant = 0;
  for (int i = 0; i < 3; ++i) {
    if (banks[i]->host_released) return false;  // staged to a device, W9b's
    if (vt::IsBlockQuant(banks[i]->dtype)) ++quant;
  }
  if (quant == 0) return false;
  if (quant != 3) {
    throw std::runtime_error(
        "glm5_next bridge: `" + what + "` has " + std::to_string(quant) +
        " of its 3 routed-expert banks block-quantized and the rest expanded. "
        "The grouped keep-quant seam takes all three or none, and a partial "
        "state means the load policy routed one bank differently from its "
        "siblings. Route all three the same way; see glm5_next_moe.h.");
  }

  for (int i = 0; i < 3; ++i) {
    // All THREE shape-preserving transforms, not just the i8mm one. Each
    // rewrites the buffer while leaving the dtype and the byte count alone:
    // `repacked` is the i8mm q8_0 interleave, `q8_0_aligned` is the CUDA
    // coalesced [all qs | all scales] layout, and `elem_kn_repacked` transposes
    // to [K, N] while the recorded shape stays [N, K]. A bank carrying any of
    // them is bytes the seam would read in the wrong order.
    if (banks[i]->repacked || banks[i]->q8_0_aligned ||
        banks[i]->elem_kn_repacked) {
      throw std::runtime_error(
          "glm5_next bridge: `" + what + names[i] +
          "` carries a load-time REPACK marker (repacked=" +
          std::to_string(static_cast<int>(banks[i]->repacked)) +
          " q8_0_aligned=" + std::to_string(static_cast<int>(banks[i]->q8_0_aligned)) +
          " elem_kn_repacked=" +
          std::to_string(static_cast<int>(banks[i]->elem_kn_repacked)) +
          "), and the grouped keep-quant seam "
          "reads canonical block order. A repack preserves the dtype AND the "
          "byte count, so nothing else in this bridge can detect it and the "
          "result would be finite and wrong. This model declines the repack at "
          "load (kGlm5NextQuantRepack, glm5_next_loader.cpp); if that changed, "
          "the seam has to change with it.");
    }
  }

  if (src.gate_exps.dtype != src.up_exps.dtype) {
    throw std::runtime_error(
        "glm5_next bridge: `" + what + "` stores gate_exps as " +
        std::string(vt::Name(src.gate_exps.dtype)) + " and up_exps as " +
        std::string(vt::Name(src.up_exps.dtype)) +
        ". `vt::MoeGateUpSwiGLUGrouped` quantizes the activation ONCE and dots "
        "both towers with it, so it requires one dtype for the pair and cannot "
        "represent this checkpoint. Measured across all 43 sparse blocks of "
        "unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL there are 0 such layers, so a "
        "checkpoint that reaches this message is a NEW mixing pattern and needs "
        "the split-GEMM shape rather than a silent f32 fallback.");
  }

  const int64_t E = d.n_routed_experts;
  const int64_t I = d.moe_intermediate_size;
  const int64_t H = d.hidden_size;
  // `[E, N, K]` as `LoadStackedExperts` records it; the bytes are `[E * N, K]`,
  // which is the stacked tower both ops declare. The rank change is a view.
  const int64_t want[3][3] = {{E, I, H}, {E, I, H}, {E, H, I}};
  for (int i = 0; i < 3; ++i) {
    const OwnedTensor& b = *banks[i];
    if (b.rank != 3 || b.shape[0] != want[i][0] || b.shape[1] != want[i][1] ||
        b.shape[2] != want[i][2]) {
      throw std::runtime_error(
          "glm5_next bridge: `" + what + names[i] + "` is rank " +
          std::to_string(b.rank) + " [" + std::to_string(b.shape[0]) + ", " +
          std::to_string(b.shape[1]) + ", " + std::to_string(b.shape[2]) +
          "], expected the stacked [" + std::to_string(want[i][0]) + ", " +
          std::to_string(want[i][1]) + ", " + std::to_string(want[i][2]) + "]");
    }
  }

  // Collapse `[E, N, K]` to `[E * N, K]` through `vt::Tensor::View`, which
  // asserts contiguity and that the element count is unchanged, rather than by
  // hand-editing strides. It rebuilds the tensor with `Contiguous(data, dtype,
  // device, shape)`, so the block dtype and the pointer carry and the strides
  // are recomputed in the same unit `OwnedTensor::View()` used. It drops
  // `repacked`, which is safe here only because a repacked bank was already
  // refused above.
  out->gate = src.gate_exps.View().View({E * I, H});
  out->up = src.up_exps.View().View({E * I, H});
  out->down = src.down_exps.View().View({E * H, I});
  // W9c-3a: and the tensors those views were taken over, for the device arm.
  // `View()` produces a host tensor whose `device` is CPU and cannot be
  // retagged onto a GPU (`minimax_h3_device.cpp:339-342`), so the device arm
  // uploads from the SOURCE instead and memoizes the allocation on the source's
  // own `d_dev`. These three pointers are what make that upload happen once per
  // MODEL rather than once per layer per step, because this whole struct is
  // rebuilt on every layer of every step by `Glm5NextGgufLayerSource`.
  //
  // They are borrowed and they point into `Glm5NextWeights`, which outlives
  // every `MoeForward` call, exactly as `expert_source` does.
  out->gate_src = &src.gate_exps;
  out->up_src = &src.up_exps;
  out->down_src = &src.down_exps;
  return true;
}

MoeLayerWeights BridgeMoeLayer(const Glm5NextMoeWeights& src, const MoeDims& d,
                               const std::string& what, int64_t byte_ceiling) {
  d.Validate();
  MoeLayerWeights out;
  // f32 in the file and f32 here: `F.linear(hidden.float(), weight.float())`
  // (`:158`) is upstream's own dtype, not a widening this bridge chose.
  out.router_weight = DecodeShaped(src.router, what + ".router",
                                   {d.n_routed_experts, d.hidden_size}, byte_ceiling);
  out.e_score_correction_bias =
      DecodeShaped(src.e_score_correction_bias, what + ".e_score_correction_bias",
                   {d.n_routed_experts}, byte_ceiling);
  out.shared = BridgeMlp(src.shared, d.hidden_size, d.shared_intermediate_size(),
                         what + ".shared", byte_ceiling);
  // `expert_gate_up` and `expert_down` STAY EMPTY. Filling them is 27.0 GiB per
  // sparse layer in f32 against a ~119.63 GiB box, and
  // `kBridgeTensorF32ByteCeiling` refuses the first 9.0 GiB bank by name before
  // any of it is allocated. The caller sets `expert_source`; `MoeForward`
  // refuses a layer that has neither.
  //
  // W9a: and when the banks kept their blocks, hand `MoeForward` a view of them
  // as well. It prefers this arm, and the `expert_source` the caller still sets
  // stays reachable below it as the parity operand.
  out.has_quant_banks = AdmitMoeQuantBanks(src, d, what, &out.quant_banks);
  return out;
}

}  // namespace vllm::glm5_next
