// vllm.cpp ORIGINAL; see qwen3_5_gguf_weights.h. GGUF tensor names + metadata
// keys mirror llama.cpp @ 237ad9b (qwen35moe/qwen3next: src/llama-arch.cpp,
// src/models/qwen35moe.cpp); the convert-time value transforms this file
// INVERTS to recover raw-HF weights are in conversion/qwen.py.
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vt/dtype.h"

namespace vllm {

OwnedTensor OwnGgufQuantBlocks(const GgufTensorInfo& tensor, int64_t n,
                               int64_t k, int64_t row_offset) {
  vt::DType dt = vt::DType::kF32;
  VT_CHECK(KeepQuantDType(tensor.ggml_type, &dt),
           "qwen3_5 gguf: keep-quant on a non-keep-quant encoding for " +
               tensor.name);
  VT_CHECK(n > 0 && k > 0 && row_offset >= 0,
           "qwen3_5 gguf: bad keep-quant slice for " + tensor.name);
  // Throws when k is not a whole number of blocks (ggml_row_size contract).
  const size_t row_bytes = vt::RowSizeBytes(dt, k);
  const size_t begin = static_cast<size_t>(row_offset) * row_bytes;
  const size_t bytes = static_cast<size_t>(n) * row_bytes;
  VT_CHECK(begin + bytes <= tensor.nbytes,
           "qwen3_5 gguf: keep-quant slice exceeds the tensor span for " +
               tensor.name);

  OwnedTensor o;
  o.dtype = dt;
  o.rank = 2;
  o.shape[0] = n;  // N = out features (ggml src0 rows)
  o.shape[1] = k;  // K = in features
  // GGUF disk order [out, in] IS the MatmulBT [N, K] orientation: no transpose.
  o.nk = true;
  o.bytes.assign(tensor.data + begin, tensor.data + begin + bytes);
  return o;
}

namespace {

// --- small helpers -------------------------------------------------------

bool HasTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

int64_t ShapeNumel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "qwen3_5 gguf: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

// src bf16 [rows, cols] -> dst bf16 [cols, rows].
void TransposeBf16(const uint16_t* src, int64_t rows, int64_t cols,
                   uint16_t* dst) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* src_row = src + r * cols;
    for (int64_t c = 0; c < cols; ++c) dst[c * rows + r] = src_row[c];
  }
}

// Reorder the V-head rows in [row_off, row_off + num_v*head_rows) of a
// [rows, cols] row-major buffer from GGUF tiled order back to HF grouped order
// (inverse of conversion/qwen.py _reorder_v_heads). grouped head g=k*R+r reads
// tiled head t=r*K+k (R = num_v_per_k, K = num_k_heads).
template <typename T>
void ReorderVRows(std::vector<T>& buf, int64_t cols, int64_t row_off,
                  int64_t num_k, int64_t num_v_per_k, int64_t head_rows) {
  const int64_t num_v = num_k * num_v_per_k;
  const int64_t head_stride = head_rows * cols;
  std::vector<T> seg(static_cast<size_t>(num_v) * head_stride);
  T* base = buf.data() + row_off * cols;
  for (int64_t k = 0; k < num_k; ++k) {
    for (int64_t r = 0; r < num_v_per_k; ++r) {
      const int64_t g = k * num_v_per_k + r;
      const int64_t t = r * num_k + k;
      std::memcpy(seg.data() + g * head_stride, base + t * head_stride,
                  static_cast<size_t>(head_stride) * sizeof(T));
    }
  }
  std::memcpy(base, seg.data(), seg.size() * sizeof(T));
}

// Reorder the full column range [0, cols) of a [rows, cols] row-major buffer
// (cols = num_v * head_cols) from GGUF tiled to HF grouped order (out_proj).
template <typename T>
void ReorderVCols(std::vector<T>& buf, int64_t rows, int64_t cols,
                  int64_t num_k, int64_t num_v_per_k, int64_t head_cols) {
  std::vector<int64_t> src_col(static_cast<size_t>(cols));
  for (int64_t k = 0; k < num_k; ++k) {
    for (int64_t r = 0; r < num_v_per_k; ++r) {
      const int64_t g = k * num_v_per_k + r;  // grouped head (dst)
      const int64_t t = r * num_k + k;        // tiled head (src)
      for (int64_t h = 0; h < head_cols; ++h) {
        src_col[static_cast<size_t>(g * head_cols + h)] = t * head_cols + h;
      }
    }
  }
  std::vector<T> out(buf.size());
  for (int64_t rr = 0; rr < rows; ++rr) {
    const T* in_row = buf.data() + rr * cols;
    T* out_row = out.data() + rr * cols;
    for (int64_t c = 0; c < cols; ++c) out_row[c] = in_row[src_col[c]];
  }
  buf.swap(out);
}

// Dequant a named GGUF tensor to bf16 bit patterns (natural [out, in] row-major
// order, ne0 = in fastest). Returns the flat buffer; `info` is filled in.
std::vector<uint16_t> DqBf16(const GgufFile& g, const std::string& name,
                             const GgufTensorInfo** info) {
  const GgufTensorInfo& t = g.Get(name);
  *info = &t;
  return DequantGgufRowToBf16(t.ggml_type, t.data, ShapeNumel(t.shape));
}

std::vector<float> DqF32(const GgufFile& g, const std::string& name,
                         const GgufTensorInfo** info) {
  const GgufTensorInfo& t = g.Get(name);
  *info = &t;
  return DequantGgufRowToF32(t.ggml_type, t.data, ShapeNumel(t.shape));
}

// bf16 tensor copied verbatim with `shape` (dequant, then own the bytes).
OwnedTensor OwnBf16(const GgufFile& g, const std::string& name,
                    const std::vector<int64_t>& shape) {
  const GgufTensorInfo* t = nullptr;
  std::vector<uint16_t> dq = DqBf16(g, name, &t);
  VT_CHECK(ShapeNumel(shape) == static_cast<int64_t>(dq.size()),
           "qwen3_5 gguf: element-count mismatch for " + name);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  std::memcpy(o.bytes.data(), dq.data(), dq.size() * sizeof(uint16_t));
  return o;
}

// bf16 [out, in] -> owned bf16 [in, out] (Matmul-B layout).
OwnedTensor OwnBf16T(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo* t = nullptr;
  std::vector<uint16_t> dq = DqBf16(g, name, &t);
  VT_CHECK(t->shape.size() == 2, "qwen3_5 gguf: expected 2-D weight " + name);
  const int64_t out_dim = t->shape[0];
  const int64_t in_dim = t->shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(dq.data(), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// RMSNorm weight stored (w + 1) by the convert -> owned bf16 [n] holding raw w.
OwnedTensor OwnNormMinus1(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo* t = nullptr;
  std::vector<float> dq = DqF32(g, name, &t);
  OwnedTensor o = MakeOwned(vt::DType::kBF16,
                            {static_cast<int64_t>(dq.size())});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (size_t i = 0; i < dq.size(); ++i) dst[i] = vt::F32ToBF16(dq[i] - 1.0F);
  return o;
}

// --- residency routing (L3) ----------------------------------------------
//
// EVERY tensor this loader touches passes through GgufLoadPolicy::Route with
// an explicit role, so the policy's audit hook observes the complete tensor
// list and no tensor can reach a residency by omission.

// A 2-D GEMM weight taken verbatim from the file. keep-quant -> raw blocks in
// the file's [N, K] order (nk = true); otherwise TODAY'S EXACT path — dequant
// to bf16 and transpose to Matmul-B [K, N] — byte for byte unchanged.
OwnedTensor OwnMatmulWeight(const GgufFile& g, const std::string& name,
                            const GgufLoadPolicy& pol) {
  const GgufTensorInfo& t = g.Get(name);
  if (pol.Route(t, GgufTensorRole::kMatmulWeight) ==
      GgufResidency::kKeepQuant) {
    VT_CHECK(t.shape.size() == 2, "qwen3_5 gguf: expected 2-D weight " + name);
    return OwnGgufQuantBlocks(t, t.shape[0], t.shape[1]);
  }
  return OwnBf16T(g, name);
}

// Route a tensor that can NEVER keep its blocks (a value/layout rewrite, a
// gather table, a conv filter or a 1-D vector) and assert the policy agrees.
// This is the runtime half of the totality contract: the audit hook still sees
// the tensor, and a policy that ever tried to keep such a tensor quantized
// would fail LOUDLY here rather than silently skip its transform.
void RequireExpand(const GgufLoadPolicy& pol, const GgufFile& g,
                   const std::string& name, GgufTensorRole role) {
  VT_CHECK(pol.Route(g.Get(name), role) == GgufResidency::kExpandBf16,
           std::string("qwen3_5 gguf: a ") + Name(role) +
               " tensor must not keep quant blocks: " + name);
}

// --- config --------------------------------------------------------------

int64_t KvInt(const GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI64: return std::get<int64_t>(v.v);
    case kGgufBool: return std::get<bool>(v.v) ? 1 : 0;
    default:
      throw std::runtime_error("qwen3_5 gguf: key " + key +
                               " is not an integer");
  }
}

double KvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(KvInt(v, key));
}

int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "qwen3_5 gguf: missing metadata key " + key);
  return KvInt(*v, key);
}

double ReqFloat(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "qwen3_5 gguf: missing metadata key " + key);
  return KvFloat(*v, key);
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v ? KvInt(*v, key) : dflt;
}

}  // namespace

HfConfig HfConfigFromGguf(const GgufFile& gguf) {
  const GgufValue* arch_v = gguf.FindKv("general.architecture");
  VT_CHECK(arch_v != nullptr && arch_v->TypeId() == kGgufString,
           "qwen3_5 gguf: general.architecture must be a string");
  const std::string arch = std::get<std::string>(arch_v->v);
  VT_CHECK(arch == "qwen35moe" || arch == "qwen3next" || arch == "qwen35",
           "qwen3_5 gguf: unexpected architecture '" + arch + "'");
  const std::string p = arch + ".";

  HfConfig c;
  c.model_type = arch;
  // `general.architecture` is llama.cpp's GGUF family key, not the HuggingFace
  // model-class ID consumed by vLLM's ModelRegistry. Map it onto the canonical
  // registered architecture (dense `qwen35` -> the 27B-family dense wrapper,
  // the MoE keys -> the MoE wrapper) while retaining the GGUF key in
  // model_type for metadata lookup.
  c.architectures = {arch == "qwen35" ? "Qwen3_5ForConditionalGeneration"
                                      : "Qwen3_5MoeForConditionalGeneration"};

  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  const int64_t block_count = ReqInt(gguf, p + "block_count");
  const int64_t nextn = OptInt(gguf, p + "nextn_predict_layers", 0);
  c.num_hidden_layers = block_count - nextn;

  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads =
      OptInt(gguf, p + "attention.head_count_kv", c.num_attention_heads);
  c.head_dim = OptInt(gguf, p + "attention.key_length",
                      c.num_attention_heads > 0
                          ? c.hidden_size / c.num_attention_heads
                          : 0);

  // vocab_size: prefer the kv, else read token_embd's leading (out) dim.
  const GgufValue* vocab_kv = gguf.FindKv(p + "vocab_size");
  c.vocab_size = vocab_kv ? KvInt(*vocab_kv, p + "vocab_size")
                          : gguf.Get("token_embd.weight").shape[0];

  // MoE.
  c.num_experts = OptInt(gguf, p + "expert_count", 0);
  c.num_experts_per_tok = OptInt(gguf, p + "expert_used_count", 0);
  c.moe_intermediate_size =
      OptInt(gguf, p + "expert_feed_forward_length", 0);
  c.shared_expert_intermediate_size =
      OptInt(gguf, p + "expert_shared_feed_forward_length", 0);
  c.intermediate_size = OptInt(gguf, p + "feed_forward_length", 0);

  // Gated DeltaNet (llama.cpp qwen35moe: head_k/v_dim = ssm.state_size,
  // n_k_heads = ssm.group_count, n_v_heads = ssm.time_step_rank).
  c.linear_num_key_heads = OptInt(gguf, p + "ssm.group_count", 0);
  c.linear_num_value_heads = OptInt(gguf, p + "ssm.time_step_rank", 0);
  const int64_t ssm_state = OptInt(gguf, p + "ssm.state_size", 0);
  c.linear_key_head_dim = ssm_state;
  c.linear_value_head_dim = ssm_state;
  c.linear_conv_kernel_dim = OptInt(gguf, p + "ssm.conv_kernel", 0);
  // The GGUF metadata does not currently carry HF's mamba_ssm_dtype, but this
  // loader is specific to the Qwen3.5/3.6 family whose recurrent cache contract
  // is FP32. Preserve it when reconstructing HfConfig so safetensors and GGUF
  // allocate the same BF16-conv/FP32-SSM state pair.
  c.mamba_ssm_dtype = "float32";

  // RoPE / norm / context.
  const GgufValue* freq = gguf.FindKv(p + "rope.freq_base");
  c.rope_theta = freq ? KvFloat(*freq, p + "rope.freq_base") : 10000.0;
  c.rotary_dim = OptInt(gguf, p + "rope.dimension_count", 0);
  c.rms_norm_eps = ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 0);
  c.torch_dtype = "bfloat16";

  // layer_types: explicit recurrent-layers bool array, else the
  // full_attention_interval pattern (every interval-th layer is full attn).
  c.layer_types.reserve(static_cast<size_t>(c.num_hidden_layers));
  const GgufValue* recr = gguf.FindKv(p + "attention.recurrent_layers");
  if (recr != nullptr && recr->TypeId() == kGgufArray) {
    const GgufArray& arr = std::get<GgufArray>(recr->v);
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      const bool is_recr =
          l < static_cast<int64_t>(arr.elems.size()) &&
          KvInt(arr.elems[static_cast<size_t>(l)], "recurrent_layers") != 0;
      c.layer_types.emplace_back(is_recr ? "linear_attention"
                                         : "full_attention");
    }
  } else {
    const int64_t interval =
        OptInt(gguf, p + "full_attention_interval", 4);
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      const bool is_recr = interval <= 0 || ((l + 1) % interval != 0);
      c.layer_types.emplace_back(is_recr ? "linear_attention"
                                         : "full_attention");
    }
  }
  return c;
}

// --- weights -------------------------------------------------------------

namespace {

std::string Blk(int64_t il, const std::string& suffix) {
  return "blk." + std::to_string(il) + "." + suffix;
}

GdnLayerWeights LoadGdnGguf(const GgufFile& g, int64_t il, const HfConfig& c,
                            const GgufLoadPolicy& pol) {
  const int64_t num_k = c.linear_num_key_heads;
  const int64_t num_v = c.linear_num_value_heads;
  const int64_t dv = c.linear_value_head_dim;
  const int64_t key_dim = num_k * c.linear_key_head_dim;
  const bool reorder = num_v != num_k && num_k > 0 && (num_v % num_k) == 0;
  const int64_t rpk = num_k > 0 ? num_v / num_k : 1;  // num_v_per_k
  // When the V-head reorder is active these projections are LAYOUT-rewritten
  // at load, so they are kTransformedWeight and can never keep their blocks;
  // without it they are ordinary verbatim GEMM weights. (out_proj's reorder
  // permutes COLUMNS, which live inside a block, so it is unconditionally
  // block-unsafe when active — same rule, stated per tensor below.)
  const GgufTensorRole proj_role = reorder
                                       ? GgufTensorRole::kTransformedWeight
                                       : GgufTensorRole::kMatmulWeight;

  GdnLayerWeights gdn;

  // in_proj_qkv <- attn_qkv [conv_dim, H]; only the trailing V rows reorder.
  {
    const std::string nm = Blk(il, "attn_qkv.weight");
    if (pol.Route(g.Get(nm), proj_role) == GgufResidency::kKeepQuant) {
      const GgufTensorInfo& ti = g.Get(nm);
      gdn.in_proj_qkv = OwnGgufQuantBlocks(ti, ti.shape[0], ti.shape[1]);
    } else {
      const GgufTensorInfo* t = nullptr;
      std::vector<uint16_t> dq = DqBf16(g, nm, &t);
      const int64_t out_dim = t->shape[0];
      const int64_t in_dim = t->shape[1];
      if (reorder) ReorderVRows(dq, in_dim, /*row_off=*/2 * key_dim, num_k, rpk, dv);
      gdn.in_proj_qkv = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
      TransposeBf16(dq.data(), out_dim, in_dim,
                    reinterpret_cast<uint16_t*>(gdn.in_proj_qkv.bytes.data()));
    }
  }
  // in_proj_z <- attn_gate [value_dim, H]; all rows are V.
  {
    const std::string nm = Blk(il, "attn_gate.weight");
    if (pol.Route(g.Get(nm), proj_role) == GgufResidency::kKeepQuant) {
      const GgufTensorInfo& ti = g.Get(nm);
      gdn.in_proj_z = OwnGgufQuantBlocks(ti, ti.shape[0], ti.shape[1]);
    } else {
      const GgufTensorInfo* t = nullptr;
      std::vector<uint16_t> dq = DqBf16(g, nm, &t);
      const int64_t out_dim = t->shape[0];
      const int64_t in_dim = t->shape[1];
      if (reorder) ReorderVRows(dq, in_dim, 0, num_k, rpk, dv);
      gdn.in_proj_z = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
      TransposeBf16(dq.data(), out_dim, in_dim,
                    reinterpret_cast<uint16_t*>(gdn.in_proj_z.bytes.data()));
    }
  }
  // in_proj_b <- ssm_beta, in_proj_a <- ssm_alpha [num_v, H]; rows are V heads.
  for (auto* pr : {&gdn.in_proj_b, &gdn.in_proj_a}) {
    const std::string nm =
        Blk(il, pr == &gdn.in_proj_b ? "ssm_beta.weight" : "ssm_alpha.weight");
    if (pol.Route(g.Get(nm), proj_role) == GgufResidency::kKeepQuant) {
      const GgufTensorInfo& ti = g.Get(nm);
      *pr = OwnGgufQuantBlocks(ti, ti.shape[0], ti.shape[1]);
      continue;
    }
    const GgufTensorInfo* t = nullptr;
    std::vector<uint16_t> dq = DqBf16(g, nm, &t);
    const int64_t out_dim = t->shape[0];
    const int64_t in_dim = t->shape[1];
    if (reorder) ReorderVRows(dq, in_dim, 0, num_k, rpk, 1);
    *pr = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
    TransposeBf16(dq.data(), out_dim, in_dim,
                  reinterpret_cast<uint16_t*>(pr->bytes.data()));
  }
  // conv1d <- ssm_conv1d [conv_dim, K]; only V channels reorder. NOT transposed.
  {
    RequireExpand(pol, g, Blk(il, "ssm_conv1d.weight"),
                  GgufTensorRole::kConvWeight);
    const GgufTensorInfo* t = nullptr;
    std::vector<uint16_t> dq = DqBf16(g, Blk(il, "ssm_conv1d.weight"), &t);
    VT_CHECK(t->shape.size() == 2, "qwen3_5 gguf: ssm_conv1d must be 2-D");
    const int64_t conv_dim = t->shape[0];
    const int64_t kk = t->shape[1];
    if (reorder) ReorderVRows(dq, kk, /*row_off=*/2 * key_dim, num_k, rpk, dv);
    gdn.conv1d_weight = MakeOwned(vt::DType::kBF16, {conv_dim, kk});
    std::memcpy(gdn.conv1d_weight.bytes.data(), dq.data(),
                dq.size() * sizeof(uint16_t));
  }
  // out_proj <- ssm_out [H, value_dim]; reorder V columns, then transpose.
  // The COLUMN reorder cuts across block boundaries, so when it is active this
  // tensor is kTransformedWeight and must expand.
  {
    const std::string nm = Blk(il, "ssm_out.weight");
    if (pol.Route(g.Get(nm), proj_role) == GgufResidency::kKeepQuant) {
      const GgufTensorInfo& ti = g.Get(nm);
      gdn.out_proj = OwnGgufQuantBlocks(ti, ti.shape[0], ti.shape[1]);
    } else {
      const GgufTensorInfo* t = nullptr;
      std::vector<uint16_t> dq = DqBf16(g, nm, &t);
      const int64_t out_dim = t->shape[0];
      const int64_t in_dim = t->shape[1];
      if (reorder) ReorderVCols(dq, out_dim, in_dim, num_k, rpk, dv);
      gdn.out_proj = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
      TransposeBf16(dq.data(), out_dim, in_dim,
                    reinterpret_cast<uint16_t*>(gdn.out_proj.bytes.data()));
    }
  }
  // a_log <- ssm_a = -exp(A_log): recover A_log = log(-value). f32 [num_v].
  {
    RequireExpand(pol, g, Blk(il, "ssm_a"),
                  GgufTensorRole::kTransformedWeight);
    const GgufTensorInfo* t = nullptr;
    std::vector<float> dq = DqF32(g, Blk(il, "ssm_a"), &t);
    if (reorder) ReorderVRows(dq, 1, 0, num_k, rpk, 1);
    gdn.a_log = MakeOwned(vt::DType::kF32, {static_cast<int64_t>(dq.size())});
    auto* dst = reinterpret_cast<float*>(gdn.a_log.bytes.data());
    for (size_t i = 0; i < dq.size(); ++i) dst[i] = std::log(-dq[i]);
  }
  // dt_bias <- ssm_dt.bias (unchanged value). f32 [num_v].
  {
    RequireExpand(pol, g, Blk(il, "ssm_dt.bias"), GgufTensorRole::kVector);
    const GgufTensorInfo* t = nullptr;
    std::vector<float> dq = DqF32(g, Blk(il, "ssm_dt.bias"), &t);
    if (reorder) ReorderVRows(dq, 1, 0, num_k, rpk, 1);
    gdn.dt_bias = MakeOwned(vt::DType::kF32, {static_cast<int64_t>(dq.size())});
    std::memcpy(gdn.dt_bias.bytes.data(), dq.data(), dq.size() * sizeof(float));
  }
  // norm_weight <- ssm_norm: raw HF (convert does NOT add 1 here). bf16 [Dv].
  RequireExpand(pol, g, Blk(il, "ssm_norm.weight"), GgufTensorRole::kVector);
  gdn.norm_weight = OwnBf16(g, Blk(il, "ssm_norm.weight"), {dv});
  return gdn;
}

FullAttnLayerWeights LoadAttnGguf(const GgufFile& g, int64_t il,
                                  const GgufLoadPolicy& pol) {
  FullAttnLayerWeights a;
  a.q_proj = OwnMatmulWeight(g, Blk(il, "attn_q.weight"), pol);
  a.k_proj = OwnMatmulWeight(g, Blk(il, "attn_k.weight"), pol);
  a.v_proj = OwnMatmulWeight(g, Blk(il, "attn_v.weight"), pol);
  a.o_proj = OwnMatmulWeight(g, Blk(il, "attn_output.weight"), pol);
  // (w - 1) rewrite: a VALUE transform, so never keep-quant.
  RequireExpand(pol, g, Blk(il, "attn_q_norm.weight"),
                GgufTensorRole::kTransformedWeight);
  a.q_norm = OwnNormMinus1(g, Blk(il, "attn_q_norm.weight"));
  RequireExpand(pol, g, Blk(il, "attn_k_norm.weight"),
                GgufTensorRole::kTransformedWeight);
  a.k_norm = OwnNormMinus1(g, Blk(il, "attn_k_norm.weight"));
  return a;
}

// Split a stacked expert tensor "blk.N.<stem>" (GGUF torch shape [E, out, in])
// into E owned bf16 [in, out] (transposed to Matmul-B layout).
std::vector<OwnedTensor> LoadExpertsT(const GgufFile& g, int64_t il,
                                      const std::string& stem,
                                      int64_t num_experts,
                                      const GgufLoadPolicy& pol) {
  const std::string name = Blk(il, stem);
  const GgufTensorInfo& ti = g.Get(name);
  VT_CHECK(ti.shape.size() == 3 && ti.shape[0] == num_experts,
           "qwen3_5 gguf: expected [E,out,in] expert tensor " + stem);
  const int64_t out_dim = ti.shape[1];
  const int64_t in_dim = ti.shape[2];

  std::vector<OwnedTensor> experts;
  experts.reserve(static_cast<size_t>(num_experts));

  if (pol.Route(ti, GgufTensorRole::kStackedExpertWeight) ==
      GgufResidency::kKeepQuant) {
    // Each expert occupies `out_dim` WHOLE rows, i.e. a whole number of
    // blocks, so the split is a byte range and no block is ever cut.
    for (int64_t e = 0; e < num_experts; ++e) {
      experts.push_back(
          OwnGgufQuantBlocks(ti, out_dim, in_dim, /*row_offset=*/e * out_dim));
    }
    return experts;
  }

  const GgufTensorInfo* t = nullptr;
  std::vector<uint16_t> dq = DqBf16(g, name, &t);
  const int64_t per = out_dim * in_dim;
  for (int64_t e = 0; e < num_experts; ++e) {
    OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
    TransposeBf16(dq.data() + e * per, out_dim, in_dim,
                  reinterpret_cast<uint16_t*>(o.bytes.data()));
    experts.push_back(std::move(o));
  }
  return experts;
}

MoeBlockWeights LoadMoeGguf(const GgufFile& g, int64_t il, const HfConfig& c,
                            const GgufLoadPolicy& pol) {
  MoeBlockWeights m;
  m.router_gate = OwnMatmulWeight(g, Blk(il, "ffn_gate_inp.weight"), pol);
  // shared gate: 1-D [H] in GGUF -> [H, 1] (matches the safetensors [H,1]).
  RequireExpand(pol, g, Blk(il, "ffn_gate_inp_shexp.weight"),
                GgufTensorRole::kVector);
  m.shared_gate =
      OwnBf16(g, Blk(il, "ffn_gate_inp_shexp.weight"), {c.hidden_size, 1});
  m.expert_gate =
      LoadExpertsT(g, il, "ffn_gate_exps.weight", c.num_experts, pol);
  m.expert_up = LoadExpertsT(g, il, "ffn_up_exps.weight", c.num_experts, pol);
  m.expert_down =
      LoadExpertsT(g, il, "ffn_down_exps.weight", c.num_experts, pol);
  m.shared_gate_proj = OwnMatmulWeight(g, Blk(il, "ffn_gate_shexp.weight"), pol);
  m.shared_up_proj = OwnMatmulWeight(g, Blk(il, "ffn_up_shexp.weight"), pol);
  m.shared_down_proj = OwnMatmulWeight(g, Blk(il, "ffn_down_shexp.weight"), pol);
  return m;
}

}  // namespace

Qwen3_5MoeWeights LoadQwen3_5MoeFromGguf(const GgufFile& gguf,
                                         const HfConfig& config,
                                         const GgufLoadPolicy* policy) {
  // Null policy => the process environment, whose defaults reproduce the
  // historical all-bf16 expansion byte for byte.
  const GgufLoadPolicy env_policy = GgufLoadPolicy::FromEnv();
  const GgufLoadPolicy& pol = policy != nullptr ? *policy : env_policy;
  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 gguf: layer_types size must equal num_hidden_layers");
  VT_CHECK(config.num_experts > 0,
           "qwen3_5 gguf: num_experts must be > 0 for the MoE model");

  Qwen3_5MoeWeights w;
  // embed_tokens [vocab, H] (NOT transposed): a GATHER table, never a GEMM.
  RequireExpand(pol, gguf, "token_embd.weight",
                GgufTensorRole::kEmbeddingTable);
  w.embed_tokens =
      OwnBf16(gguf, "token_embd.weight", gguf.Get("token_embd.weight").shape);
  RequireExpand(pol, gguf, "output_norm.weight",
                GgufTensorRole::kTransformedWeight);
  w.final_norm = OwnNormMinus1(gguf, "output_norm.weight");
  // lm_head [H, vocab] (+T). Tied-embedding GGUFs omit output.weight; then the
  // head is the transposed token_embd (same [vocab,H] source), as llama.cpp
  // does (TENSOR_DUPLICATED from token_embd). When tied, token_embd is routed
  // TWICE — once as the gather table, once as this GEMM weight — which is the
  // correct answer for both uses.
  w.lm_head = OwnMatmulWeight(gguf,
                              HasTensor(gguf, "output.weight")
                                  ? "output.weight"
                                  : "token_embd.weight",
                              pol);

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t il = 0; il < config.num_hidden_layers; ++il) {
    Qwen3_5MoeLayerWeights layer;
    RequireExpand(pol, gguf, Blk(il, "attn_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.input_layernorm = OwnNormMinus1(gguf, Blk(il, "attn_norm.weight"));
    RequireExpand(pol, gguf, Blk(il, "post_attention_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.post_attention_layernorm =
        OwnNormMinus1(gguf, Blk(il, "post_attention_norm.weight"));
    const std::string& lt = config.layer_types[static_cast<size_t>(il)];
    if (lt == "linear_attention") {
      layer.is_linear_attention = true;
      layer.gdn = LoadGdnGguf(gguf, il, config, pol);
    } else if (lt == "full_attention") {
      layer.is_linear_attention = false;
      layer.attn = LoadAttnGguf(gguf, il, pol);
    } else {
      VT_CHECK(false, "qwen3_5 gguf: unknown layer_type " + lt);
    }
    layer.moe = LoadMoeGguf(gguf, il, config, pol);
    w.layers.push_back(std::move(layer));
  }
  return w;
}

Qwen3_5DenseWeights LoadQwen3_5DenseFromGguf(const GgufFile& gguf,
                                             const HfConfig& config,
                                             const GgufLoadPolicy* policy) {
  // Null policy => the process environment, whose defaults reproduce the
  // historical all-bf16 expansion byte for byte.
  const GgufLoadPolicy env_policy = GgufLoadPolicy::FromEnv();
  const GgufLoadPolicy& pol = policy != nullptr ? *policy : env_policy;
  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 gguf: layer_types size must equal num_hidden_layers");
  VT_CHECK(config.num_experts == 0,
           "qwen3_5 gguf: num_experts must be 0 for the dense model");

  Qwen3_5DenseWeights w;
  RequireExpand(pol, gguf, "token_embd.weight",
                GgufTensorRole::kEmbeddingTable);
  w.embed_tokens =
      OwnBf16(gguf, "token_embd.weight", gguf.Get("token_embd.weight").shape);
  RequireExpand(pol, gguf, "output_norm.weight",
                GgufTensorRole::kTransformedWeight);
  w.final_norm = OwnNormMinus1(gguf, "output_norm.weight");
  // Tied-embedding GGUFs (the 2B) omit output.weight; the head is then the
  // transposed token_embd, as llama.cpp does (TENSOR_DUPLICATED).
  w.lm_head = OwnMatmulWeight(gguf,
                              HasTensor(gguf, "output.weight")
                                  ? "output.weight"
                                  : "token_embd.weight",
                              pol);

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t il = 0; il < config.num_hidden_layers; ++il) {
    Qwen3_5DenseLayerWeights layer;
    RequireExpand(pol, gguf, Blk(il, "attn_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.input_layernorm = OwnNormMinus1(gguf, Blk(il, "attn_norm.weight"));
    RequireExpand(pol, gguf, Blk(il, "post_attention_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.post_attention_layernorm =
        OwnNormMinus1(gguf, Blk(il, "post_attention_norm.weight"));
    const std::string& lt = config.layer_types[static_cast<size_t>(il)];
    if (lt == "linear_attention") {
      layer.is_linear_attention = true;
      layer.gdn = LoadGdnGguf(gguf, il, config, pol);
    } else if (lt == "full_attention") {
      layer.is_linear_attention = false;
      layer.attn = LoadAttnGguf(gguf, il, pol);
    } else {
      VT_CHECK(false, "qwen3_5 gguf: unknown layer_type " + lt);
    }
    // Dense SwiGLU MLP (bf16 fields; the fp4 variants stay empty).
    layer.mlp.gate_proj = OwnMatmulWeight(gguf, Blk(il, "ffn_gate.weight"), pol);
    layer.mlp.up_proj = OwnMatmulWeight(gguf, Blk(il, "ffn_up.weight"), pol);
    layer.mlp.down_proj = OwnMatmulWeight(gguf, Blk(il, "ffn_down.weight"), pol);
    w.layers.push_back(std::move(layer));
  }
  return w;
}

}  // namespace vllm
