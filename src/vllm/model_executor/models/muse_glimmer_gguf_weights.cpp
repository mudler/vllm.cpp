// vllm.cpp ORIGINAL; see muse_glimmer_gguf_weights.h for the port record, the
// four convert-time transforms this file inverts (each verified element-wise
// against the released bf16 safetensors), and the residency rationale.
//
// GGUF tensor names + metadata keys mirror llama.cpp's `muse-glimmer` arch
// (ggml-org/llama.cpp#26841, merged 2026-08-10): `token_embd`, `output`,
// `output_norm`, `blk.%d.attn_{norm,q,k,v,output,gate,q_norm,k_norm}`,
// `blk.%d.{post_attention_norm,ffn_norm,post_ffw_norm}`,
// `blk.%d.ffn_{gate,up,down}`, and LLM_KV_* under the `muse-glimmer.` prefix.
// llama.cpp is a SECONDARY reference only — never the correctness oracle and
// never a speed denominator (.agents/specs/muse-glimmer.md §0).
#include "vllm/model_executor/models/muse_glimmer_gguf_weights.h"

#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"  // OwnGgufQuantBlocks
#include "vt/dtype.h"
#include "vt/quant.h"

namespace vllm {
namespace {

constexpr const char* kPrefix = "muse-glimmer.";

// ── metadata readers ─────────────────────────────────────────────────────────

int64_t AsInt(const GgufValue& v, const std::string& key) {
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
    default: break;
  }
  VT_CHECK(false, "muse_glimmer gguf: key " + key + " is not an integer");
  return 0;
}

double AsDouble(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(AsInt(v, key));
}

int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "muse_glimmer gguf: missing required key '" + key + "'");
  return AsInt(*v, key);
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t fallback) {
  const GgufValue* v = g.FindKv(key);
  return v == nullptr ? fallback : AsInt(*v, key);
}

double OptDouble(const GgufFile& g, const std::string& key, double fallback) {
  const GgufValue* v = g.FindKv(key);
  return v == nullptr ? fallback : AsDouble(*v, key);
}

bool HasTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors())
    if (t.name == name) return true;
  return false;
}

std::string Blk(int64_t layer, const char* suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

// ── shape / value helpers ────────────────────────────────────────────────────

// `GgufTensorInfo::shape` is already REVERSED into torch row-major order by the
// reader, so a 2-D matmul weight reads [N = out, K = in] — the file's own
// orientation and our MatmulBT one.
void RequireShape(const GgufTensorInfo& t, const std::vector<int64_t>& want) {
  bool ok = t.shape.size() == want.size();
  for (size_t i = 0; ok && i < want.size(); ++i) ok = t.shape[i] == want[i];
  if (ok) return;
  std::string got;
  for (size_t i = 0; i < t.shape.size(); ++i)
    got += (i ? ", " : "") + std::to_string(t.shape[i]);
  std::string exp;
  for (size_t i = 0; i < want.size(); ++i)
    exp += (i ? ", " : "") + std::to_string(want[i]);
  VT_CHECK(false, "muse_glimmer gguf: shape mismatch for " + t.name + ": got [" +
                      got + "], expected [" + exp + "]");
}

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk) {
  OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "muse_glimmer gguf: rank exceeds kMaxRank");
  for (int i = 0; i < o.rank; ++i) o.shape[i] = shape[i];
  o.nk = nk;
  o.bytes.resize(static_cast<size_t>(Numel(shape)) * sizeof(uint16_t));
  return o;
}

// Dequantize a whole tensor into an owned bf16 buffer in the file's own order.
OwnedTensor ExpandBf16(const GgufFile& g, const std::string& name,
                       const std::vector<int64_t>& shape, bool nk) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, shape);
  const int64_t n = Numel(shape);
  const std::vector<uint16_t> dq = DequantGgufRowToBf16(t.ggml_type, t.data, n);
  VT_CHECK(static_cast<int64_t>(dq.size()) == n,
           "muse_glimmer gguf: dequant length mismatch for " + name);
  OwnedTensor o = MakeBf16(shape, nk);
  std::memcpy(o.bytes.data(), dq.data(), dq.size() * sizeof(uint16_t));
  return o;
}

// The UNTIED head is the one weight the forward consumes through `vt::Matmul` in
// Matmul-B [K = H, N = vocab] order, so it is transposed at load exactly as the
// safetensors path's LoadBf16Transposed does. A block encoding cannot be
// transposed without requantizing, which is why this one always expands.
OwnedTensor ExpandBf16Transposed(const GgufFile& g, const std::string& name,
                                 int64_t rows, int64_t cols) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {rows, cols});
  const std::vector<uint16_t> dq =
      DequantGgufRowToBf16(t.ggml_type, t.data, rows * cols);
  OwnedTensor o = MakeBf16({cols, rows}, /*nk=*/false);
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* src = dq.data() + r * cols;
    for (int64_t c = 0; c < cols; ++c) dst[c * rows + r] = src[c];
  }
  return o;
}

// A [n] norm vector as bf16. `unshift` inverts the converter's baked `+1`
// (transform 1). The FINAL norm takes no offset in the model and is stored raw,
// so it passes `unshift = false` — an over-eager blanket un-shift is exactly the
// bug the gate's final-norm case catches.
OwnedTensor LoadNormBf16(const GgufFile& g, const std::string& name, int64_t n,
                         bool unshift) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {n});
  const std::vector<float> f = DequantGgufRowToF32(t.ggml_type, t.data, n);
  OwnedTensor o = MakeBf16({n}, /*nk=*/false);
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t i = 0; i < n; ++i)
    dst[i] = vt::F32ToBF16(unshift ? f[static_cast<size_t>(i)] - 1.0f
                                   : f[static_cast<size_t>(i)]);
  return o;
}

// ── residency ────────────────────────────────────────────────────────────────

const GgufFile* MmapSrc(const GgufFile& g, const GgufLoadPolicy& pol) {
  return pol.mmap_residency ? &g : nullptr;
}

// One standalone [N, K] matmul operand: kept as raw ggml blocks when the policy
// routes it there, expanded to bf16 in the file's own [N, K] order otherwise.
OwnedTensor LoadMatmul(const GgufFile& g, const GgufLoadPolicy& pol,
                       const std::string& name, int64_t n, int64_t k) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {n, k});
  const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
  if (r == GgufResidency::kKeepQuant)
    return OwnGgufQuantBlocks(t, n, k, /*row_offset=*/0, MmapSrc(g, pol),
                              pol.quant_repack);
  if (r == GgufResidency::kKeepF16)
    return OwnGgufF16(t, n, k, /*row_offset=*/0, MmapSrc(g, pol), /*nk=*/true,
                      pol.elem_kn_repack);
  return ExpandBf16(g, name, {n, k}, /*nk=*/true);
}

// ── the converter's Q/K RoPE row permutation (transform 4) ───────────────────

// llama.cpp's converter stores `attn_q` / `attn_k` in ggml's INTERLEAVED-RoPE row
// order, not HF's. `LlamaModel.permute` (llama.cpp `conversion/llama.py:163-169`,
// which the `muse-glimmer` converter in ggml-org/llama.cpp#26841 inherits) does,
// per head:
//
//     w.reshape(n_head, 2, head_dim/2, K).swapaxes(1, 2).reshape(n_head*head_dim, K)
//
// so GGUF row `h*Dh + b*2 + a` holds HF row `h*Dh + a*(Dh/2) + b`. That is the
// weight-side half of ggml's `rope_norm`, which rotates ADJACENT channel pairs
// (2i, 2i+1); HF — and our `vt::RopeNeox` — rotates HALF-OFFSET pairs
// (i, i + Dh/2). `attn_k` is permuted with `n_head_kv`, `attn_q` with `n_head`.
//
// Our forward consumes HF order, so the loader must UN-permute. The inverse of
// the map above sends destination (HF) row `i` within a head to source (GGUF) row
// `(i % (Dh/2)) * 2 + i / (Dh/2)`.
//
// VERIFIED against the released bf16 checkpoint on 2026-08-11, layers 0, 3, 25
// and 51, for both `attn_q` (32 heads) and `attn_k` (2 kv heads): read verbatim
// the GGUF disagrees with `meta-models/Muse-Glimmer-30B` at mean relative error
// ~1.40 (i.e. unrelated numbers), and read through this map it agrees at ~0.077 —
// exactly the Q4_K quantization noise every other tensor in the file shows.
// `attn_v`, `attn_output` and both MLP shards are NOT permuted upstream and match
// verbatim, which is why only q and k carry a head count here.
//
// This is invisible without RoPE: a permutation applied to BOTH q and k leaves
// q·k unchanged, so the 13 NoPE layers are correct either way. On the 39 RoPE
// layers it rotates the wrong channel pairs, which is why the model loaded,
// ran, and emitted degenerate text instead of failing (issue #359).
int64_t QkUnpermuteSrcRow(int64_t dst_row, int64_t head_dim) {
  const int64_t half = head_dim / 2;
  const int64_t h = dst_row / head_dim;
  const int64_t i = dst_row % head_dim;
  return h * head_dim + (i % half) * 2 + i / half;
}

// One shard of a merged operand. `rope_heads > 0` marks a shard that carries the
// converter's Q/K row permutation and must be un-permuted on the way in; 0 means
// the shard is taken verbatim.
struct MergedShard {
  std::string name;
  int64_t n = 0;
  int64_t rope_heads = 0;
};

// A MERGED [sum(n_i), K] operand built from several file tensors that share K.
//
// The forward wants ONE tensor for qkv (rows q|k|v) and one for gate_up (rows
// gate|up), and the row ORDER is load-bearing: a k shard landing where q is
// expected permutes attention silently.
//
// Kept as a BLOCK CONCAT only when every shard is routed to kKeepQuant AND all
// shards carry the SAME ggml type — a k-quant row is a whole number of
// superblocks, so appending one tensor's rows to another's is a byte
// concatenation and nothing is requantized. Two different block encodings cannot
// share one tensor (one dtype, one row size), so a heterogeneous set expands to
// bf16 instead. That is not a preference: on the released 17 GB file
// `attn_v` is Q6_K while `attn_q`/`attn_k` are Q4_K, so the qkv trio expands and
// the homogeneous ffn_gate|ffn_up pair stays quantized.
OwnedTensor LoadMerged(const GgufFile& g, const GgufLoadPolicy& pol,
                       const std::vector<MergedShard>& shards, int64_t k) {
  VT_CHECK(!shards.empty(), "muse_glimmer gguf: empty merged operand");
  int64_t total_n = 0;
  bool keep = true;
  uint32_t type0 = 0;
  for (size_t i = 0; i < shards.size(); ++i) {
    const GgufTensorInfo& t = g.Get(shards[i].name);
    RequireShape(t, {shards[i].n, k});
    VT_CHECK(shards[i].rope_heads == 0 ||
                 (shards[i].n % shards[i].rope_heads == 0 &&
                  (shards[i].n / shards[i].rope_heads) % 2 == 0),
             "muse_glimmer gguf: " + shards[i].name +
                 " cannot carry the converter's Q/K RoPE row permutation: its row "
                 "count is not an even head_dim times the head count");
    total_n += shards[i].n;
    if (i == 0) type0 = t.ggml_type;
    keep = keep && t.ggml_type == type0 &&
           pol.Route(t, GgufTensorRole::kMatmulWeight) == GgufResidency::kKeepQuant;
  }

  if (keep) {
    vt::DType dt = vt::DType::kF32;
    VT_CHECK(KeepQuantDType(type0, &dt),
             "muse_glimmer gguf: keep-quant routed a non-keep-quant encoding");
    const size_t row_bytes = vt::RowSizeBytes(dt, k);
    OwnedTensor o;
    o.dtype = dt;
    o.rank = 2;
    o.shape[0] = total_n;
    o.shape[1] = k;
    o.nk = true;
    std::vector<uint8_t> buf;
    buf.reserve(static_cast<size_t>(total_n) * row_bytes);
    for (const MergedShard& s : shards) {
      const GgufTensorInfo& t = g.Get(s.name);
      const size_t bytes = static_cast<size_t>(s.n) * row_bytes;
      VT_CHECK(bytes <= t.nbytes,
               "muse_glimmer gguf: merged slice exceeds the tensor span for " + s.name);
      if (s.rope_heads > 0) {
        // Transform 4, on the KEPT-QUANT path. A k-quant row is a whole number of
        // superblocks, so reordering WHOLE rows never touches a block boundary and
        // nothing is requantized — the same property that makes the concat itself
        // a byte operation.
        const int64_t head_dim = s.n / s.rope_heads;
        for (int64_t r = 0; r < s.n; ++r) {
          const uint8_t* src =
              t.data + static_cast<size_t>(QkUnpermuteSrcRow(r, head_dim)) * row_bytes;
          buf.insert(buf.end(), src, src + row_bytes);
        }
      } else {
        buf.insert(buf.end(), t.data, t.data + bytes);
      }
      // The merged buffer is now the authority for these bytes; the file pages
      // they came from are read-once (llama.cpp `unmap_fragment`, L5).
      if (pol.mmap_residency) g.DropSpanResidency(t.data, bytes);
    }
    o.bytes.assign(buf.data(), buf.data() + buf.size());
    // CIQ G7 applies to the WHOLE merged operand, once, not per shard. It is a
    // q8_0-only byte permutation, so it never fires on a k-quant file.
    if (pol.quant_repack && vt::cpu::QuantRepackEligible(dt, total_n, k)) {
      vt::cpu::QuantRepackWeight(dt, o.bytes.data(), total_n, k);
      o.repacked = true;
    }
    return o;
  }

  OwnedTensor o = MakeBf16({total_n, k}, /*nk=*/true);
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  int64_t row = 0;
  for (const MergedShard& s : shards) {
    const GgufTensorInfo& t = g.Get(s.name);
    const std::vector<uint16_t> dq =
        DequantGgufRowToBf16(t.ggml_type, t.data, s.n * k);
    VT_CHECK(static_cast<int64_t>(dq.size()) == s.n * k,
             "muse_glimmer gguf: dequant length mismatch for " + s.name);
    if (s.rope_heads > 0) {
      // Transform 4, on the DEQUANTIZED path — the one the released 17 GB file
      // actually takes, because its qkv trio is heterogeneous (Q6_K v vs Q4_K q/k).
      const int64_t head_dim = s.n / s.rope_heads;
      for (int64_t r = 0; r < s.n; ++r)
        std::memcpy(dst + (row + r) * k,
                    dq.data() + QkUnpermuteSrcRow(r, head_dim) * k,
                    static_cast<size_t>(k) * sizeof(uint16_t));
    } else {
      std::memcpy(dst + row * k, dq.data(), dq.size() * sizeof(uint16_t));
    }
    row += s.n;
  }
  return o;
}

// ── the folded weightless QK-norms (transform 2) ─────────────────────────────

// Reads one F32/F16 [head_dim] vector and requires it to be a single constant,
// returning that constant. A NON-constant vector is a genuinely weighted
// per-channel QK-norm, which this architecture does not have; taking its mean
// (or its first element) would build a plausible-looking wrong model, so this
// throws naming the tensor instead.
float ConstantVector(const GgufFile& g, const std::string& name, int64_t n) {
  const GgufTensorInfo& t = g.Get(name);
  RequireShape(t, {n});
  const std::vector<float> f = DequantGgufRowToF32(t.ggml_type, t.data, n);
  for (int64_t i = 1; i < n; ++i)
    VT_CHECK(f[static_cast<size_t>(i)] == f[0],
             "muse_glimmer gguf: " + name +
                 " is not a constant vector. Muse Glimmer's per-head QK-norm is "
                 "WEIGHTLESS (muse_glimmer.py:1121) and llama.cpp materializes it "
                 "as a constant (ones on the key side, the query pre-scale on the "
                 "query side); a per-channel weight here is a norm this "
                 "architecture does not have and is refused rather than averaged.");
  return f[0];
}

}  // namespace

// ── public surface ───────────────────────────────────────────────────────────

bool IsMuseGlimmerGguf(const GgufFile& gguf) {
  const GgufValue* a = gguf.FindKv("general.architecture");
  return a != nullptr && a->TypeId() == kGgufString &&
         std::get<std::string>(a->v) == kMuseGlimmerGgufArch;
}

double MuseGlimmerGgufQueryPreScale(const GgufFile& gguf, int64_t num_layers,
                                    int64_t head_dim) {
  VT_CHECK(num_layers > 0 && head_dim > 0,
           "muse_glimmer gguf: bad geometry for the query pre-scale recovery");
  double scale = 0.0;
  for (int64_t l = 0; l < num_layers; ++l) {
    const float q = ConstantVector(gguf, Blk(l, "attn_q_norm.weight"), head_dim);
    const float k = ConstantVector(gguf, Blk(l, "attn_k_norm.weight"), head_dim);
    VT_CHECK(k == 1.0f,
             "muse_glimmer gguf: " + Blk(l, "attn_k_norm.weight") +
                 " is not all ones. The key-side QK-norm is weightless, so the "
                 "converter writes the identity there; a different constant means "
                 "a scaling this forward would drop.");
    VT_CHECK(q > 0.0f && std::isfinite(q),
             "muse_glimmer gguf: " + Blk(l, "attn_q_norm.weight") +
                 " carries a non-positive query pre-scale");
    if (l == 0) scale = q;
    VT_CHECK(static_cast<double>(q) == scale,
             "muse_glimmer gguf: layer " + std::to_string(l) +
                 " disagrees with layer 0 about the folded query pre-scale; only a "
                 "single model-wide `scale_query_by` is supported");
  }
  return scale;
}

HfConfig MuseGlimmerHfConfigFromGguf(const GgufFile& gguf) {
  VT_CHECK(IsMuseGlimmerGguf(gguf),
           "muse_glimmer gguf: general.architecture must be 'muse-glimmer'");
  const std::string p = kPrefix;

  HfConfig c;
  c.model_type = "muse_glimmer";
  // A text-tower GGUF carries no perception encoder (that is the separate
  // mmproj file), so it announces the CAUSAL-LM architecture. The registry maps
  // both Muse strings onto one factory, so either resolves.
  c.architectures = {"MuseGlimmerForCausalLM"};

  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_hidden_layers = ReqInt(gguf, p + "block_count");
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads =
      OptInt(gguf, p + "attention.head_count_kv", c.num_attention_heads);
  c.head_dim = OptInt(gguf, p + "attention.key_length",
                      c.num_attention_heads > 0
                          ? c.hidden_size / c.num_attention_heads
                          : 0);
  c.intermediate_size = OptInt(gguf, p + "feed_forward_length", 0);
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 0);
  c.rope_theta = OptDouble(gguf, p + "rope.freq_base", 500000.0);
  c.rms_norm_eps = OptDouble(gguf, p + "attention.layer_norm_rms_epsilon", 1e-6);
  c.torch_dtype = "bfloat16";
  // vocab_size: prefer the kv, else read token_embd's leading (out) dim.
  c.vocab_size = OptInt(gguf, p + "vocab_size", 0);
  if (c.vocab_size == 0) c.vocab_size = gguf.Get("token_embd.weight").shape[0];

  const int64_t L = c.num_hidden_layers;
  const int64_t sliding_window = OptInt(gguf, p + "attention.sliding_window", 0);

  // Transform 3: `attention.sliding_window_pattern` IS the iRoPE mask.
  // true  => RoPE AND sliding window   => no_rope_layers[i] = 1
  // false => NoPE AND full attention   => no_rope_layers[i] = 0
  // Absent => upstream's backward-counted default, which only HAPPENS to agree
  // with the released 52-layer schedule; deriving from the file when the file
  // says so is what keeps a differently-scheduled checkpoint from being split
  // into wrong-RoPE, wrong-window layers that still emit fluent text.
  std::vector<int64_t> no_rope;
  const GgufValue* pat = gguf.FindKv(p + "attention.sliding_window_pattern");
  if (pat != nullptr && pat->TypeId() == kGgufArray) {
    const GgufArray& arr = std::get<GgufArray>(pat->v);
    VT_CHECK(static_cast<int64_t>(arr.elems.size()) == L,
             "muse_glimmer gguf: attention.sliding_window_pattern length must "
             "equal block_count");
    for (const GgufValue& e : arr.elems)
      no_rope.push_back(AsInt(e, "sliding_window_pattern") != 0 ? 1 : 0);
  } else {
    no_rope = DefaultMuseGlimmerNoRopeLayers(L);
  }

  // Transform 2: the query pre-scale lives in the folded attn_q_norm, not in a
  // metadata key. A file without those tensors has no QK-norm at all.
  const bool has_qk_norm = HasTensor(gguf, Blk(0, "attn_q_norm.weight"));
  const double scale_query_by =
      has_qk_norm ? MuseGlimmerGgufQueryPreScale(gguf, L, c.head_dim) : 1.0;

  // The head is TIED exactly when the file omits `output.weight`, which is
  // llama.cpp's TENSOR_DUPLICATED convention. The released 30B ships one, so it
  // is untied — reading this backwards would silently run the embedding table as
  // the head.
  const bool tied = !HasTensor(gguf, "output.weight");

  nlohmann::json text = nlohmann::json::object();
  text["vocab_size"] = c.vocab_size;
  text["hidden_size"] = c.hidden_size;
  text["intermediate_size"] = c.intermediate_size;
  text["num_hidden_layers"] = L;
  text["num_attention_heads"] = c.num_attention_heads;
  text["num_key_value_heads"] = c.num_key_value_heads;
  text["head_dim"] = c.head_dim;
  text["max_position_embeddings"] = c.max_position_embeddings;
  text["sliding_window"] = sliding_window;
  text["tie_word_embeddings"] = tied;
  text["rms_norm_eps"] = c.rms_norm_eps;
  // NAMED RESIDUAL: the GGUF carries no post-norm epsilon, so the post-norms
  // fall back to `rms_norm_eps` (1e-5 vs the safetensors config's 1e-8). See the
  // header — the difference is ~5e-6 relative inside 1/sqrt(ms + eps), two
  // orders of magnitude below bf16's spacing, so it is not representable in the
  // activation dtype. Recorded, not hidden.
  text["hidden_activation"] = "silu";
  text["rope_parameters"] = nlohmann::json{{"rope_theta", c.rope_theta}};
  text["no_rope_layers"] = no_rope;
  text["output_multiplier"] = OptDouble(gguf, p + "logit_scale", 1.0);
  text["final_logit_softcapping"] =
      OptDouble(gguf, p + "final_logit_softcapping", 0.0);
  text["use_qk_norm"] = has_qk_norm;
  // The attention output gate is present exactly when the file ships attn_gate.
  text["use_attn_output_gate"] = HasTensor(gguf, Blk(0, "attn_gate.weight"));
  // An EXPLICIT scale_query_by wins outright in ResolveMuseGlimmerQueryPreScale,
  // which is what we want: the value is already folded (post-1/sqrt(head_dim)),
  // so the magnitude disambiguation must not run on it.
  text["scale_query_by"] = scale_query_by;

  c.raw = nlohmann::json::object();
  c.raw["architectures"] = c.architectures;
  c.raw["model_type"] = "muse_glimmer";
  c.raw["text_config"] = std::move(text);
  // No `vision_config`: the perception encoder is a separate mmproj file, which
  // this loader refuses by name (MuseGlimmerRefuseMmproj).
  return c;
}

bool MuseGlimmerGgufTensorName(const std::string& canonical, std::string* out) {
  if (canonical == "model.embed_tokens.weight") {
    *out = "token_embd.weight";
    return true;
  }
  if (canonical == "model.norm.weight") {
    *out = "output_norm.weight";
    return true;
  }
  if (canonical == "lm_head.weight") {
    *out = "output.weight";
    return true;
  }
  static constexpr char kLayers[] = "model.layers.";
  if (canonical.rfind(kLayers, 0) != 0) return false;  // vision_* lives elsewhere
  const size_t begin = sizeof(kLayers) - 1;
  const size_t dot = canonical.find('.', begin);
  if (dot == std::string::npos) return false;
  const std::string idx = canonical.substr(begin, dot - begin);
  const std::string rest = canonical.substr(dot + 1);

  // The attention OUTPUT GATE and the MLP gate share a suffix on the HF side and
  // MUST land on different GGUF names; the table keys on the full canonical
  // suffix, so the two can never collide.
  static const std::unordered_map<std::string, std::string> kSuffix = {
      {"input_layernorm.weight", "attn_norm.weight"},
      {"post_attention_layernorm.weight", "post_attention_norm.weight"},
      {"pre_feedforward_layernorm.weight", "ffn_norm.weight"},
      {"post_feedforward_layernorm.weight", "post_ffw_norm.weight"},
      {"self_attn.q_proj.weight", "attn_q.weight"},
      {"self_attn.k_proj.weight", "attn_k.weight"},
      {"self_attn.v_proj.weight", "attn_v.weight"},
      {"self_attn.o_proj.weight", "attn_output.weight"},
      {"self_attn.output_gate_proj.weight", "attn_gate.weight"},
      {"mlp.gate_proj.weight", "ffn_gate.weight"},
      {"mlp.up_proj.weight", "ffn_up.weight"},
      {"mlp.down_proj.weight", "ffn_down.weight"},
  };
  const auto it = kSuffix.find(rest);
  if (it == kSuffix.end()) return false;
  *out = "blk." + idx + "." + it->second;
  return true;
}

std::vector<std::string> EnumerateMuseGlimmerGgufTensors(
    const MuseGlimmerParams& params) {
  const MuseGlimmerTextParams& t = params.text;
  std::vector<std::string> names;
  names.push_back("token_embd.weight");
  for (int64_t l = 0; l < t.num_hidden_layers; ++l) {
    names.push_back(Blk(l, "attn_norm.weight"));
    names.push_back(Blk(l, "post_attention_norm.weight"));
    names.push_back(Blk(l, "ffn_norm.weight"));
    names.push_back(Blk(l, "post_ffw_norm.weight"));
    names.push_back(Blk(l, "attn_q.weight"));
    names.push_back(Blk(l, "attn_k.weight"));
    names.push_back(Blk(l, "attn_v.weight"));
    names.push_back(Blk(l, "attn_output.weight"));
    if (t.use_attn_output_gate) names.push_back(Blk(l, "attn_gate.weight"));
    // GGUF-ONLY: what our side realizes WITHOUT a tensor. ggml has no weightless
    // RMSNorm, so the converter materializes the identity (k) and the folded
    // query pre-scale (q). Enumerating them is what lets the structural gate
    // demand zero unaccounted tensors instead of tolerating strangers.
    if (t.use_qk_norm) {
      names.push_back(Blk(l, "attn_q_norm.weight"));
      names.push_back(Blk(l, "attn_k_norm.weight"));
    }
    names.push_back(Blk(l, "ffn_gate.weight"));
    names.push_back(Blk(l, "ffn_up.weight"));
    names.push_back(Blk(l, "ffn_down.weight"));
  }
  names.push_back("output_norm.weight");
  if (!t.tie_word_embeddings) names.push_back("output.weight");
  return names;
}

MuseGlimmerWeights LoadMuseGlimmerFromGguf(const GgufFile& gguf,
                                           const HfConfig& config,
                                           const GgufLoadPolicy* policy) {
  const GgufLoadPolicy pol = policy != nullptr ? *policy : GgufLoadPolicy::FromEnv();

  MuseGlimmerWeights w;
  w.params = ParseMuseGlimmerParams(config);
  const MuseGlimmerTextParams& t = w.params.text;

  // STRUCTURAL accounting, the same shape the safetensors loader reports: the
  // enumerated GGUF names against what the file actually carries. The load below
  // reads through the SAME names, so accounting and materialization can never
  // disagree.
  const std::vector<std::string> expected = EnumerateMuseGlimmerGgufTensors(w.params);
  w.enumerated_tensors = static_cast<int64_t>(expected.size());
  std::set<std::string> present;
  for (const GgufTensorInfo& info : gguf.Tensors()) present.insert(info.name);
  for (const std::string& n : expected)
    if (present.count(n) != 0) ++w.accounted_tensors;

  const int64_t H = t.hidden_size;
  const int64_t V = t.vocab_size;
  const int64_t I = t.intermediate_size;
  const int64_t qdim = t.num_attention_heads * t.head_dim;
  const int64_t kdim = t.num_key_value_heads * t.head_dim;

  // A [vocab, H] GATHER table, not a GEMM operand — the embedding kernel reads
  // it row-wise, so it expands to bf16 like every other GGUF embedding in the
  // tree (llama.cpp likewise dequantizes embedding rows on the fly).
  w.embed_tokens = ExpandBf16(gguf, "token_embd.weight", {V, H}, /*nk=*/false);
  // NO un-shift: the final norm carries no `+1` offset in the model.
  w.final_norm = LoadNormBf16(gguf, "output_norm.weight", H, /*unshift=*/false);
  if (!t.tie_word_embeddings)
    w.lm_head = ExpandBf16Transposed(gguf, "output.weight", V, H);

  w.layers.reserve(static_cast<size_t>(t.num_hidden_layers));
  for (int64_t l = 0; l < t.num_hidden_layers; ++l) {
    MuseGlimmerLayerWeights lw;
    // Transform 1: the four sandwich norms are stored PRE-OFFSET (w_hf + 1).
    lw.input_layernorm = LoadNormBf16(gguf, Blk(l, "attn_norm.weight"), H, true);
    lw.post_attention_layernorm =
        LoadNormBf16(gguf, Blk(l, "post_attention_norm.weight"), H, true);
    lw.pre_feedforward_layernorm =
        LoadNormBf16(gguf, Blk(l, "ffn_norm.weight"), H, true);
    lw.post_feedforward_layernorm =
        LoadNormBf16(gguf, Blk(l, "post_ffw_norm.weight"), H, true);

    // Transform 4: `attn_q` and `attn_k` are stored in ggml's interleaved-RoPE row
    // order and are UN-permuted here (with `n_head` and `n_head_kv` respectively,
    // mirroring llama.cpp's `permute(w, n_head, n_head_kv)`); `attn_v` is stored
    // verbatim and takes no head count.
    lw.attn.qkv_proj =
        LoadMerged(gguf, pol,
                   {{Blk(l, "attn_q.weight"), qdim, t.num_attention_heads},
                    {Blk(l, "attn_k.weight"), kdim, t.num_key_value_heads},
                    {Blk(l, "attn_v.weight"), kdim, 0}},
                   H);
    lw.attn.o_proj = LoadMatmul(gguf, pol, Blk(l, "attn_output.weight"), H, qdim);
    if (t.use_attn_output_gate)
      lw.attn.output_gate_proj =
          LoadMatmul(gguf, pol, Blk(l, "attn_gate.weight"), qdim, H);

    lw.mlp.gate_up_proj =
        LoadMerged(gguf, pol,
                   {{Blk(l, "ffn_gate.weight"), I, 0}, {Blk(l, "ffn_up.weight"), I, 0}},
                   H);
    lw.mlp.down_proj = LoadMatmul(gguf, pol, Blk(l, "ffn_down.weight"), H, I);
    w.layers.push_back(std::move(lw));
  }
  w.text_loaded = true;
  // `vision.loaded` stays false: a text-tower GGUF ships no perception encoder,
  // and the mmproj that would carry one cannot build ours (see below). The mm
  // forward already refuses BY NAME on an unloaded tower rather than reading
  // empty vectors.
  return w;
}

[[noreturn]] void MuseGlimmerRefuseMmproj() {
  throw std::runtime_error(
      "MuseGlimmer GGUF: the released mmproj perception encoder cannot be loaded. "
      "Its `v.patch_embd.weight` is ggml ne [14, 14, 3, 1536] = torch "
      "[1536, 3*14*14] = [1536, 588], but `conv1_linear` needs "
      "patch_temporal * 3 * patch_size^2 = 2*3*14*14 = 1176 input features (the "
      "bf16 safetensors ships exactly [1536, 1176]). The patch_temporal axis is "
      "ABSENT from the mmproj file, so the temporal half of the patch embedding "
      "does not exist to be loaded; every other tower tensor maps cleanly. Use "
      "the safetensors checkpoint for image and video until the llama.cpp "
      "converter emits the full patch_temporal-by-3-by-patch^2 weight.");
}

}  // namespace vllm
