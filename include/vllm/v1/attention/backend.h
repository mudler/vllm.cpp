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
//   * The deprecated _seq_lens_cpu / _num_computed_tokens_cpu accessors and the
//     unpadded()/replace() spec-decode helpers.
//   * AttentionBackend: the whole supports_*/validate_configuration capability
//     surface, get_kv_cache_stride_order / get_kv_cache_block_dim, cudagraph
//     support (AttentionCGSupport / build_for_cudagraph_capture / drafting),
//     MLA / sparse-MLA impls. Ported here: the T0 core (CommonAttentionMetadata,
//     the AttentionBackend/AttentionImpl/AttentionMetadataBuilder ABCs,
//     get_name + get_kv_cache_shape, the forward signature).
#ifndef VLLM_V1_ATTENTION_BACKEND_H_
#define VLLM_V1_ATTENTION_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/v1/worker/gpu/prepare_inputs.h"
#include "vt/tensor.h"

namespace vllm::v1 {

// Upstream AttentionType (str Enum). Only DECODER is exercised at T0; the
// others are carried for fidelity / attn_type routing (Task 3+).
enum class AttentionType {
  kDecoder,        // "decoder"
  kEncoder,        // "encoder"
  kEncoderOnly,    // "encoder_only"
  kEncoderDecoder  // "encoder_decoder"
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

}  // namespace vllm::v1

#endif  // VLLM_V1_ATTENTION_BACKEND_H_
