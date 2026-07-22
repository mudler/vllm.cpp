// Ported from: vllm/v1/attention/backend.py @ e24d1b24
//
// Scope (M1.6 Task 1): the per-step attention metadata + the attention backend
// interface the paged-attention path (Task 2/3) and the GDN metadata (Task 4)
// build on. Behavioral only: no CUDA, no model. The concrete paged backend
// (get_impl_cls / get_builder_cls / the real forward) is Task 3.
//
// ─── DEVICE-TENSOR-AS-HOST-ARRAY DEVIATION ──────────────────────────────────
// Upstream keeps every CommonAttentionMetadata index as a torch.Tensor, with a
// GPU copy AND a CPU copy for many of them (query_start_loc + query_start_loc_cpu,
// seq_lens + the deprecated _seq_lens_cpu / seq_lens_cpu property). At T0 both
// the "device" and "_cpu" variants are plain host std::vectors — device
// placement is the runner's concern (same deviation prepare_inputs.h records for
// StepInputs). We keep BOTH fields so Task-3+ can wire the device copy without
// reshaping the struct; today they alias the same host data.
//
// ─── DEFERRED upstream fields (marked; T0 never exercises) ──────────────────
//   * FastPrefillAttentionBuilder: logits_indices_padded / num_logits_indices.
//   * CrossAttentionBuilder / encoder-decoder: encoder_seq_lens(_cpu).
//   * Decode context parallelism (dcp): dcp_local_seq_lens(_cpu).
//   * DeepSeek V4 sparse: positions, mm_req_doc_ranges, is_prefilling,
//     seq_lens_cpu_upper_bound, rswa_prefix_lens.
//   * The deprecated lazy _seq_lens_cpu accessor and the unpadded()/replace()
//     spec-decode helpers. The host-equivalent num_computed_tokens_cpu array is
//     ported because chunked-local virtual batching consumes and rewrites it.
//   * AttentionBackend: the whole supports_*/validate_configuration capability
//     surface, get_kv_cache_stride_order / get_kv_cache_block_dim, cudagraph
//     capture/drafting hooks, MLA / sparse-MLA impls. AttentionCGSupport itself
//     is ported for the chunked-local wrapper. Ported here: the T0 core
//     (CommonAttentionMetadata, the AttentionBackend/AttentionImpl/
//     AttentionMetadataBuilder ABCs, get_name + get_kv_cache_shape, the forward
//     signature).
#ifndef VLLM_V1_ATTENTION_BACKEND_H_
#define VLLM_V1_ATTENTION_BACKEND_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/v1/worker/gpu/prepare_inputs.h"
#include "vt/ops.h"

namespace vllm::v1 {

// Upstream AttentionType (str Enum). Only DECODER is exercised at T0; the
// others are carried for fidelity / attn_type routing (Task 3+).
enum class AttentionType {
  kDecoder,        // "decoder"
  kEncoder,        // "encoder"
  kEncoderOnly,    // "encoder_only"
  kEncoderDecoder  // "encoder_decoder"
};

// Cudagraph support level advertised by an attention metadata builder.
// Values preserve upstream ordering; chunked-local explicitly returns kNever.
enum class AttentionCGSupport {
  kNever = 0,
  kUniformSingleTokenDecode = 1,
  kUniformBatch = 2,
  kAlways = 3,
};

// Marker base for per-backend attention metadata (upstream `class
// AttentionMetadata: pass`). The concrete backend metadata (Task 3) derives it.
struct AttentionMetadata {
  virtual ~AttentionMetadata() = default;
};

// Per-batch attention metadata, shared across layers and backends.
// AttentionMetadataBuilder instances use it to construct per-layer metadata.
// (Upstream: @dataclass CommonAttentionMetadata — the T0 field subset.)
//
// NOTE: both the device fields and their `_cpu` counterparts are host arrays at
// T0 (see the header deviation note). block_table_tensor is a 2D
// [num_reqs, block_table_num_cols] table flattened row-major (matching
// BlockTable's flat CpuGpuBuffer layout).
struct CommonAttentionMetadata {
  // (num_reqs + 1,) the start location of each request in the query stream.
  std::vector<int32_t> query_start_loc;
  std::vector<int32_t> query_start_loc_cpu;

  // (num_reqs,) the number of computed tokens for each request (context length
  // after this step). `seq_lens` is the device copy; `seq_lens_cpu` the CPU one.
  // (Upstream's CPU copy is the deprecated `_seq_lens_cpu` / `seq_lens_cpu`
  // property; we name it plainly per the M1.6 brief.)
  std::vector<int32_t> seq_lens;
  std::vector<int32_t> seq_lens_cpu;

  // (num_reqs,) number of tokens computed before this step. Upstream exposes
  // this as the deprecated lazy num_computed_tokens_cpu property; the local
  // host-array representation stores the derived value eagerly.
  std::vector<int32_t> num_computed_tokens_cpu;

  // Number of requests.
  int num_reqs = 0;
  // Total number of tokens in the batch (may be padded; see upstream TODO).
  int num_actual_tokens = 0;
  // Longest query in the batch (= max scheduled tokens per request).
  int max_query_len = 0;
  // Longest context length in the batch (may be an upper bound).
  int max_seq_len = 0;

  // (num_reqs, block_table_num_cols) block-id table, flattened row-major.
  std::vector<int32_t> block_table_tensor;
  int block_table_num_cols = 0;

  // (num_actual_tokens,) flat KV-cache slot id per token.
  std::vector<int64_t> slot_mapping;

  // Whether attention is causal (upstream also allows a per-request tensor;
  // T0 uses the scalar form only).
  bool causal = true;

  // Upstream: self.seq_lens.shape[0].
  int batch_size() const { return static_cast<int>(seq_lens.size()); }

  // Upstream: query_start_loc[1:] - query_start_loc[:-1]. "Naive" because it
  // assumes a query ends where the next one starts.
  std::vector<int32_t> naive_query_lens() const;
};

// Build a CommonAttentionMetadata from the M1.5 step-inputs + a block table.
// Mirrors upstream `create_common_attn_metadata` (tests/v1/attention/utils.py):
// the query_start_loc / seq_lens come straight from StepInputs, num_reqs /
// num_actual_tokens / max_query_len / max_seq_len are DERIVED here
// (num_actual_tokens = query_start_loc.back(); max_query_len = max query length;
// max_seq_len = max seq_lens). slot_mapping is taken from KV-cache group
// `kv_cache_group_id` (upstream computes one slot mapping per group).
//
// block_table_flat is the [num_reqs, block_table_num_cols] block-id table,
// flattened row-major (e.g. BlockTable::get_cpu_tensor() truncated to num_reqs
// rows). Requires num_reqs > 0.
CommonAttentionMetadata MakeCommonAttentionMetadata(
    const StepInputs& step, const std::vector<int32_t>& block_table_flat,
    int block_table_num_cols, bool causal = true, int kv_cache_group_id = 0);

// Minimal attention layer view (upstream `class AttentionLayer(Protocol)`): the
// per-layer q/k/v scale floats the impl reads in forward. T0 carries only the
// float scales the correctness path needs; the tensor scales / _prob_scale are
// deferred (fp8 KV cache is out of T0 scope).
struct AttentionLayer {
  float q_scale = 1.0f;  // upstream _q_scale_float
  float k_scale = 1.0f;  // upstream _k_scale_float
  float v_scale = 1.0f;  // upstream _v_scale_float
  // Per-layer backend-neutral window resolved by the generic attention layer.
  // nullopt is full attention; model-specific code must not reinterpret it.
  std::optional<vt::AttentionWindow> window_size = std::nullopt;
};

// Base class for attention implementations (upstream AttentionImpl, flattened
// from AttentionImplBase). The concrete paged impl is Task 3. Carries the
// common attributes every impl has (num_heads / head_size / scale) and the
// forward contract.
class AttentionImpl {
 public:
  virtual ~AttentionImpl() = default;

  // Common attributes (upstream AttentionImplBase required attributes).
  int num_heads = 0;
  int head_size = 0;
  float scale = 1.0f;
  std::string kv_cache_dtype = "auto";

  // forward(layer, query, key, value, kv_cache, attn_metadata, output, ...).
  // Writes the attention result into `output` (upstream also returns it; the
  // in-place `output` arg is the load-bearing one). output_scale /
  // output_block_scale are the fused-output-quant hooks (fp8/nvfp4), unused at
  // T0 — pass nullptr.
  virtual void forward(const AttentionLayer& layer, const vt::Tensor& query,
                       const vt::Tensor& key, const vt::Tensor& value,
                       const vt::Tensor& kv_cache,
                       const AttentionMetadata& attn_metadata,
                       vt::Tensor& output,
                       const vt::Tensor* output_scale = nullptr,
                       const vt::Tensor* output_block_scale = nullptr) = 0;
};

// Builds per-backend attention metadata from the common metadata (upstream
// AttentionMetadataBuilder, ABC generic over the metadata type M). The concrete
// backend-specific builder is Task 3 (full-attn) / Task 4 (GDN).
//
// build_for_cudagraph_capture / build_for_drafting / update_block_table /
// use_cascade_attention are deferred (see header note).
template <typename M>
class AttentionMetadataBuilder {
 public:
  virtual ~AttentionMetadataBuilder() = default;

  // Central method that builds attention metadata. common_prefix_len is the
  // length of the batch's common prefix (cascade attention; 0 at T0). fast_build
  // prioritizes build speed over execution speed (spec decode); unused at T0.
  virtual M build(int common_prefix_len,
                  const CommonAttentionMetadata& common_attn_metadata,
                  bool fast_build = false) = 0;
};

// Abstract attention backend (upstream AttentionBackend). T0 contract: get_name
// + get_kv_cache_shape. The get_impl_cls / get_builder_cls factories are Task 3
// (the base returns nullptr / is realized per backend via the templated
// AttentionMetadataBuilder).
class AttentionBackend {
 public:
  virtual ~AttentionBackend() = default;

  virtual std::string get_name() const = 0;

  // The logical KV-cache tensor shape for one layer (upstream
  // get_kv_cache_shape). cache_dtype_str selects the quant layout ("auto" is
  // the only T0 mode).
  virtual std::vector<int64_t> get_kv_cache_shape(
      int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
      int64_t head_size, const std::string& cache_dtype_str = "auto") const = 0;

  // Factory hook for the attention impl (upstream get_impl_cls returns the impl
  // CLASS; here a factory returning an instance). Task-3 backends override; the
  // base returns nullptr.
  virtual std::unique_ptr<AttentionImpl> get_impl_cls() const { return nullptr; }

  // ─── Capability predicates consumed by the SELECTOR (W2) ──────────────────
  // Ported from vllm/v1/attention/backend.py:307-360 `validate_configuration`,
  // the two checks that decide selection on GB10:
  //   * `is_mla()` — mla_attention.py:1240-1242 `MLACommonBackend.is_mla()`
  //     returns True; every dense backend inherits False. Must EQUAL the
  //     request's `use_mla`.
  //   * `is_sparse()` — True only for the DSA / sparse-MLA family
  //     (flashinfer_mla_sparse.py:67). Must EQUAL the request's `use_sparse`.
  //     This is precisely why FLASHINFER_MLA_SPARSE_SM120 is filtered out of
  //     GB10's two-entry MLA list for a dense request, leaving TRITON_MLA — the
  //     behavior OBSERVED from the vLLM 0.25.0 oracle at W0.
  // Defaults are the dense/non-sparse answer, so every backend registered before
  // W2 keeps its exact selection behavior.
  virtual bool is_mla() const { return false; }
  virtual bool is_sparse() const { return false; }
};

// The T0 concrete full-attention backend. Ports the FlashAttention V1 paged KV
// layout — the mainstream vLLM V1 layout the gate models use — from
// vllm/v1/attention/backends/flash_attn.py::get_kv_cache_shape @ e24d1b24:
//     (num_blocks, 2, block_size, num_kv_heads, head_size)
// The "2" (dim 1) splits K and V; num_blocks is the OUTERMOST dim.
//
// NB: this differs from the CPU backend (cpu_attn.py), which uses an HND
// concatenated layout (num_blocks, num_kv_heads, block_size, 2*head_size).
class FlashAttentionBackend final : public AttentionBackend {
 public:
  static constexpr const char* kName = "FLASH_ATTN";

  std::string get_name() const override { return kName; }

  std::vector<int64_t> get_kv_cache_shape(
      int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
      int64_t head_size,
      const std::string& cache_dtype_str = "auto") const override;
};

// The dense-MLA backend, and the ONLY one reachable on GB10 — read from
// vllm/platforms/cuda.py:129-133 (sm_12x → [TRITON_MLA,
// FLASHINFER_MLA_SPARSE_SM120]) and OBSERVED at W0 from the vLLM 0.25.0 oracle
// on sm_121 ("Using TRITON_MLA attention backend out of potential backends:
// ['TRITON_MLA']"). Ported from
// vllm/v1/attention/backends/mla/triton_mla.py:81 TritonMLABackend, whose
// shape/flags come from its base
// vllm/model_executor/layers/attention/mla_attention.py:1206 MLACommonBackend.
//
// W2 landed the NAME + the selection-relevant capability surface (that is what
// makes `use_mla=true` resolve to TRITON_MLA). W4 fills `get_impl_cls()` with
// `TritonMLAImpl` below — the two-stage split-KV MQA decode over the latent.
// The PREFILL half is W5, and `TritonMLAImpl::forward` says so explicitly rather
// than silently producing wrong numbers.
class TritonMLABackend final : public AttentionBackend {
 public:
  static constexpr const char* kName = "TRITON_MLA";

  std::string get_name() const override { return kName; }

  // mla_attention.py:1216-1224 MLACommonBackend.get_kv_cache_shape:
  //     (num_blocks, block_size, head_size)
  // THREE dimensions — no leading K/V axis, because MLA caches ONE latent row
  // per token (kv_lora_rank + qk_rope_head_dim == 576) and reconstructs V from
  // it. `num_kv_heads` is accepted and IGNORED (`:1219`: "assumed to be 1 for
  // MLA"); we assert it rather than ignore it silently.
  std::vector<int64_t> get_kv_cache_shape(
      int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
      int64_t head_size,
      const std::string& cache_dtype_str = "auto") const override;

  // mla_attention.py:1240-1242 is_mla() -> True; TritonMLABackend is the DENSE
  // MLA backend, so is_sparse() keeps the base False (that False is exactly what
  // makes FLASHINFER_MLA_SPARSE_SM120 lose and TRITON_MLA win on GB10).
  bool is_mla() const override { return true; }

  // triton_mla.py:100-103 supports_block_size — block_size % 16 == 0. Exposed as
  // a plain predicate (the full supports_* surface is still deferred, see the
  // header note) because get_kv_cache_shape enforces it.
  static bool supports_block_size(int64_t block_size) { return block_size % 16 == 0; }

  // W4: the MLA decode impl. Upstream `get_impl_cls` returns the CLASS
  // (triton_mla.py:126-128 -> TritonMLAImpl); here a factory returning an
  // instance whose num_heads / head_size / scale the layer fills in, exactly as
  // upstream's layer passes them to the constructor.
  std::unique_ptr<AttentionImpl> get_impl_cls() const override;
};

// Decode-side attention metadata for the MLA backends. Upstream this is
// `MLACommonMetadata` / `MLACommonDecodeMetadata`
// (vllm/model_executor/layers/attention/mla_attention.py); W4 carries EXACTLY the
// fields `TritonMLAImpl.forward_mqa` reads —
//   * `attn_metadata.decode.block_table` and `.seq_lens` (triton_mla.py:245-246),
//   * `attn_metadata.max_seq_len`, which sizes the split heuristic (`:214-216`).
// Both tensors are DEVICE tensors here (not the host arrays
// CommonAttentionMetadata carries): the decode kernel reads them on the GPU with
// no host round-trip, which is what keeps the path CUDA-graph capturable.
struct MLACommonMetadata : AttentionMetadata {
  vt::Tensor block_table;  // [num_reqs, max_blocks] i32, device
  vt::Tensor seq_lens;     // [num_reqs] i32, device
  // Host-known max over seq_lens. 0 => the impl falls back to 1 split.
  int max_seq_len = 0;
  // 0 => `_compute_num_kv_splits` (triton_mla.py:40-47). 1 forces the
  // batch-invariant single-split reduction (`:212-213`).
  int num_kv_splits = 0;
};

// The dense-MLA attention impl. Ported from
// vllm/v1/attention/backends/mla/triton_mla.py:134 `TritonMLAImpl` @ e24d1b24,
// whose decode entry point is `:189 forward_mqa`. W4 implements the DECODE half
// (`vt::MlaDecodeAttention`); the prefill half is W5.
//
// Upstream's constructor REJECTS `alibi_slopes`, `sliding_window` and
// `logits_soft_cap` (`:165-171`) and any non-decoder attention type
// (`:173-179`). Our AttentionImpl surface carries only `window_size` of those,
// on the per-layer AttentionLayer, so forward() rejects a set window there — the
// same refusal at the only place we can see it.
class TritonMLAImpl final : public AttentionImpl {
 public:
  // triton_mla.py:135 `can_return_lse_for_decode = True`.
  static constexpr bool kCanReturnLseForDecode = true;

  // DEVIATION (recorded): upstream launches on torch's AMBIENT CUDA stream, which
  // a C++ port has no equivalent of. The runner sets this to its per-step queue
  // before calling forward; nullptr means the DEFAULT stream, which is correct
  // but serializing — fine for unit tests, wrong for the hot path. W7 wires the
  // runner's queue in when the DeepSeek-V2 forward lands.
  vt::Queue* queue = nullptr;

  // The DECODE entry point — the 1:1 counterpart of `forward_mqa`
  // (triton_mla.py:189-260). `q` is the already-concatenated
  // [num_reqs, num_heads, kv_lora_rank + qk_rope_head_dim] query (upstream
  // concatenates a tuple at `:200-201`); `kv_c_and_k_pe_cache` is the 3-D MLA
  // cache; `out` is [num_reqs, num_heads, kv_lora_rank] and `lse` (optional) is
  // [num_reqs, num_heads] f32.
  //
  // NOT weight absorption: folding W_UK into the query and un-projecting the
  // output with W_UV is W6. This method takes the query already in latent space
  // and returns the output still in latent space, exactly like `forward_mqa`.
  void forward_mqa(const AttentionLayer& layer, const vt::Tensor& q,
                   const vt::Tensor& kv_c_and_k_pe_cache, const MLACommonMetadata& metadata,
                   vt::Tensor& out, vt::Tensor* lse = nullptr) const;

  // The generic AttentionImpl entry. `key`/`value` are unused: MLA's cache write
  // is the separate `vt::ConcatAndCacheMla` (W3), never a (k, v) pair. Routes to
  // forward_mqa; a prefill-shaped batch throws with the W5 reason rather than
  // silently producing wrong numbers.
  void forward(const AttentionLayer& layer, const vt::Tensor& query, const vt::Tensor& key,
               const vt::Tensor& value, const vt::Tensor& kv_cache,
               const AttentionMetadata& attn_metadata, vt::Tensor& output,
               const vt::Tensor* output_scale = nullptr,
               const vt::Tensor* output_block_scale = nullptr) override;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ATTENTION_BACKEND_H_
