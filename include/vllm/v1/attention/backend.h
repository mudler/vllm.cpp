// Ported from: vllm/v1/attention/backend.py @ pin 5559679229 (re-anchored from
// `e24d1b24`, the pin retired at W5; `validate_configuration` is `:320-393` and
// the `supports_*` predicates `:154-317` there).
//
// ONE anchor is deliberately NOT advanced: FlashAttentionBackend::get_kv_cache_shape
// below still mirrors flash_attn.py @ `e24d1b24`. At `5559679229` upstream returns
// (num_blocks, num_kv_heads, block_size, 2 * head_size) instead. That is the KV
// memory format the whole engine allocates and every paged-attention kernel
// reads, so re-anchoring it is a kernel campaign rather than a comment edit.
// Named here rather than left to be discovered; owner: BACKEND-ATTN-REGISTRY.
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
//   * AttentionBackend: get_kv_cache_stride_order / get_kv_cache_block_dim,
//     cudagraph capture/drafting hooks, MLA / sparse-MLA impls.
//     AttentionCGSupport itself is ported for the chunked-local wrapper.
//     The supports_*/validate_configuration capability surface is NO LONGER
//     deferred — it is ported below (issue #1332 M1).
#ifndef VLLM_V1_ATTENTION_BACKEND_H_
#define VLLM_V1_ATTENTION_BACKEND_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/platforms/interface.h"
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

// Upstream's AttentionType IS its string value (backend.py:38-46 is a class of
// `str` constants), and `supports_attn_type` takes that string (`:292-298`).
// `platforms::AttnSelectorConfig::attn_type` therefore carries the string, and
// this is the conversion. Returns the exact upstream spelling.
const char* AttentionTypeName(AttentionType type);

// The `CacheDType` name (vllm/config/cache.py:19-36) for a KV-cache storage
// dtype, which is what `supports_kv_cache_dtype` matches against.
//
// Upstream's "auto" means "the model dtype, unquantized" (`cache.py:77-78`), and
// an f32/f16/bf16 KV cache is exactly that — so all three map to "auto" rather
// than to their own `CacheDType` spellings, which upstream uses only when the
// user pins the cache to a dtype the model does not have. `vt::DType` carries no
// fp8 member yet, so the quantized names are unreachable from here; when it
// does, this is the one place the mapping goes.
const char* KvCacheDTypeName(vt::DType dtype);

// vllm/utils/torch_utils.py:75-80 `is_quantized_kv_cache`, over the same name.
bool IsQuantizedKvCacheName(const std::string& kv_cache_dtype);

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

  // ─── The declared-capability surface (backend.py:154-393) ─────────────────
  //
  // READ THIS BEFORE TRUSTING A GREEN RESULT FROM IT (issue #1332). Every
  // predicate below describes what a backend CLAIMS. None of them can see what
  // the shipped binary contains. Upstream's
  // FlashAttentionBackend.supports_compute_capability is `>= (8,0)`; it returned
  // true on a GB10 (capability 12,1) whose FA2 fatbin holds sm_80 SASS plus
  // compute_80 PTX and nothing else, and every launch then failed a driver JIT
  // with cudaErrorUnsupportedPtxVersion. `grep -rn get_arch_list vllm/` returns
  // zero hits: vLLM never asks what its own fatbins contain, and neither does
  // this. Measured on that board through the pinned oracle, same wheel and same
  // prompt: asking for FLASHINFER generates text and exits 0, while the default
  // — which resolves FLASH_ATTN, upstream's priority 0 for this device — dies at
  // the FIRST attention call. The priority-0 choice is unrunnable and the
  // priority-1 choice works, and every predicate below passes BOTH.
  //
  // The invariant #1332 states is that no backend may be declared valid on
  // the strength of a property of the DEVICE alone, and satisfying it needs the
  // build-derived compiled-arch manifest (M2) and the launch probe (M3). This
  // layer is NECESSARY AND NOT SUFFICIENT, and a reader who takes it for a
  // runnability check will reproduce exactly the failure that opened the issue.
  //
  // Structure mirrors upstream: static declarations returning bool, collected by
  // validate_configuration into a list of reason strings; an EMPTY list means
  // valid. Ours are virtual member functions because our backends are instances
  // rather than classes (MakeAttentionBackend), which is the same adaptation
  // get_impl_cls above already records.

  // backend.py:155-156 / :59-64 — the declared value lists. A backend overrides
  // the list, not the predicate, wherever upstream does.
  virtual std::vector<vt::DType> supported_dtypes() const {
    return {vt::DType::kF16, vt::DType::kBF16};  // backend.py:59
  }
  virtual std::vector<std::string> supported_kv_cache_dtypes() const {
    return {"auto", "float16", "bfloat16"};  // backend.py:60-64
  }
  virtual std::vector<int> get_supported_head_sizes() const { return {}; }
  // backend.py:69-71 — upstream returns [MultipleOf(1)]. Every entry here IS a
  // MultipleOf: supports_block_size accepts any multiple of any entry, which is
  // upstream's hybrid_blocks rule at :184-192, and MultipleOf(1) accepts all.
  virtual std::vector<int> get_supported_kernel_block_sizes() const { return {1}; }

  // backend.py:158-161.
  virtual bool supports_head_size(int head_size) const;
  // backend.py:163-165.
  virtual bool supports_dtype(vt::DType dtype) const;
  // backend.py:167-173 — an EMPTY name is upstream's `None`, accepted outright.
  virtual bool supports_kv_cache_dtype(const std::string& kv_cache_dtype) const;
  // backend.py:175-192 — 0 is upstream's `None`, accepted outright.
  virtual bool supports_block_size(int block_size) const;

  // backend.py:237-298 — the plain feature flags, upstream defaults.
  virtual bool supports_sink() const { return false; }            // :241-243
  virtual bool supports_mm_prefix() const { return false; }        // :249-251
  virtual bool supports_per_head_quant_scales() const { return false; }  // :257-259
  virtual bool supports_sliding_window() const { return false; }   // :261-263
  virtual bool supports_non_causal() const { return false; }       // :265-274
  virtual bool supports_batch_invariance() const { return false; } // :276-278
  virtual bool supports_kv_connector() const { return true; }      // :280-282
  virtual bool supports_pcp() const { return false; }              // :284-289
  // :291-298 — the base supports DECODER only.
  virtual bool supports_attn_type(const std::string& attn_type) const {
    return attn_type == AttentionTypeName(AttentionType::kDecoder);
  }
  // :300-302.
  virtual bool supports_compute_capability(
      const platforms::DeviceCapability& capability) const {
    (void)capability;
    return true;
  }

  // :304-317 — the cross-field rule a per-field predicate cannot express.
  // Returns the reason when the COMBINATION is invalid, nullopt when it is fine.
  //
  // ADAPTATION: upstream passes nine positional arguments, every one of them a
  // field of the AttentionSelectorConfig the caller already holds
  // (cuda.py:381-384 splats that config). We pass the config itself, so a new
  // upstream field does not change nine signatures.
  virtual std::optional<std::string> supports_combination(
      const platforms::AttnSelectorConfig& cfg,
      const platforms::DeviceCapability& capability) const {
    (void)cfg;
    (void)capability;
    return std::nullopt;
  }

  // backend.py:319-393. Collects one reason per failed predicate and returns
  // them ALL: upstream builds a list and prints the whole list (cuda.py:416-420,
  // :432-446), because a message that stopped at the first failure would send a
  // reader chasing one cause of four. An EMPTY result means the backend is valid
  // for this request.
  //
  // The reason STRINGS are upstream's, byte for byte, so a refusal here and a
  // refusal from the oracle read the same.
  virtual std::vector<std::string> validate_configuration(
      const platforms::AttnSelectorConfig& cfg,
      const platforms::DeviceCapability& capability) const;
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

  // ─── Capability overrides, ported from flash_attn.py:72-239 ──────────────
  // flash_attn.py:74-80.
  std::vector<std::string> supported_kv_cache_dtypes() const override {
    return {"auto", "float16", "bfloat16", "fp8", "fp8_e4m3"};
  }
  // flash_attn.py:82-84 — MultipleOf(16).
  std::vector<int> get_supported_kernel_block_sizes() const override { return {16}; }
  // flash_attn.py:170-178. Upstream raises the ceiling to 512 when FlashAttention
  // v4 resolves (`is_fa_version_supported(4)`); this tree ships FA2 only
  // (BACKEND-CUDA-COMP-FA records FA3/FA4 as unported), so the 512 arm is
  // unreachable and is deliberately not written as an always-false branch.
  bool supports_head_size(int head_size) const override {
    if (head_size % 8 != 0) return false;
    return head_size <= 256;
  }
  // flash_attn.py:200-202. SEE THE WARNING ON THE BASE CLASS: this is the exact
  // predicate that returned true on a GB10 whose FA2 fatbin had no sm_121 code
  // (issue #1332). It is upstream's, and it is not a runnability check.
  bool supports_compute_capability(
      const platforms::DeviceCapability& capability) const override {
    return capability.major > 8 || (capability.major == 8 && capability.minor >= 0);
  }
  bool supports_sliding_window() const override { return true; }   // :98-100
  bool supports_batch_invariance() const override { return true; }  // :102-104
  bool supports_non_causal() const override { return true; }        // :106-108
  // :110-118 — FlashAttention serves every attention type.
  bool supports_attn_type(const std::string& attn_type) const override {
    return attn_type == AttentionTypeName(AttentionType::kDecoder) ||
           attn_type == AttentionTypeName(AttentionType::kEncoder) ||
           attn_type == AttentionTypeName(AttentionType::kEncoderOnly) ||
           attn_type == AttentionTypeName(AttentionType::kEncoderDecoder);
  }
  // :120-123 `get_flash_attn_version() >= 3`. This tree ships FA2, so the answer
  // is upstream's own answer for FA2 rather than a divergence — and it is what
  // makes the ported test_per_head_quant_scales case assert a refusal.
  bool supports_per_head_quant_scales() const override { return false; }
  // :204-239. Only the sink rule is expressible here: the fp8-KV and mm_prefix
  // rules both key on a resolved FA version >= 3, which this tree never reaches,
  // and supported_kv_cache_dtypes / supports_mm_prefix already refuse them.
  std::optional<std::string> supports_combination(
      const platforms::AttnSelectorConfig& cfg,
      const platforms::DeviceCapability& capability) const override {
    if (cfg.has_sink && capability.present() && capability.major < 9) {
      return std::string("sink not supported on compute capability < 9.0");
    }
    return std::nullopt;
  }
};

// The dense ROCm paged-attention backend (issue #41 M3). Upstream ROCM_ATTN
// (vllm/v1/attention/backends/rocm_attn.py) is the FlashAttention-family
// backend the ROCm platform prefers on non-AITER boards, and its NAME is what
// rocm.py:407-441 `_get_backend_priorities` puts first in the dense branch
// (verified at pin 555967922). The concrete kernel is selected at the vt::
// op-table level (vt::PagedAttention -> GetOp(kPagedAttention, kROCM),
// registered in src/vt/rocm/rocm_ops.hip) — exactly the FlashAttention
// division of labour on CUDA. The host metadata here is device-agnostic, so the
// class lives alongside FLASH_ATTN and self-registers for kROCM in backend.cpp
// (same footing as the kMETAL / kVULKAN / kTENSTORRENT rows).
//
// --- KV-LAYOUT DEVIATION (one exact tracked exception — recorded decision,
// issue #41 M3, spec rocm-attn-backend.md §3): ---
// Upstream rocm_attn.py:247-256 (`get_kv_cache_shape`) returns
// (2, num_blocks, block_size, num_kv_heads, head_size) — K/V split OUTERMOST —
// and rocm.py:521-522 says it outright: "ROCM_ATTN still uses a legacy
// attention layout (KV is the outer dimension)". Our ROCm paged-attn kernel
// (src/vt/rocm/rocm_paged_attn.hip) is a port of the CPU/CUDA pair and reads
// the SAME NHD layout FlashAttentionBackend::get_kv_cache_shape allocates
// (num_blocks, 2, block_size, num_kv_heads, head_size), indexed by TENSOR
// strides (kc_blk/kc_pg/kc_hd) — the identical precondition the Metal and
// Vulkan legs document at their registration lines. Registering the NAME
// ROCM_ATTN therefore inverts upstream's defining property: upstream's
// K/V-outermost cache cannot exist here because the engine allocates NHD and
// the local kernel reads NHD. The alternatives were (a) register FLASH_ATTN
// for kROCM, as Metal/Vulkan/Tenstorrent do, leaving "ROCM_ATTN" in the
// priority list as a permanently-skipped placeholder, or (b) keep the name and
// record the deviation as ONE exact tracked exception. We chose (b): the name
// identifies the KERNEL FAMILY that actually runs (the ROCm paged-attn kernel
// behind kPagedAttention/kROCM), the selection log then reports ROCM_ATTN on
// real silicon, and the deviation is what this record + spec pins. When (if) a
// real upstream-layout ROCm attention kernel lands, this shape flips with it.
//
// --- KV-CONNECTOR GUARD (why upstream's `use_kv_connector` gate does not
// apply — recorded, spec §4): ---
// Upstream appends ROCM_ATTN to the priority list only `if not use_kv_connector`
// (rocm.py:432-433), because connector transfer semantics are unvalidated for
// its ASYMMETRIC native K/V views. We ship a KV connector (LMCache,
// include/vllm/v1/kv_offload/kv_connector.h), but our registered shape is the
// SHARED SYMMETRIC NHD layout — the same one FLASH_ATTN (which upstream does
// use with connectors) allocates — so the asymmetric-view premise of the guard
// does not exist for this registration. The guard therefore needs no
// AttnSelectorConfig field here; if a future ROCm kernel adopts upstream's
// asymmetric layout, this registration flips shape AND the guard becomes
// load-bearing (that is the tracked-exception escape hatch).
class RocmAttentionBackend final : public AttentionBackend {
 public:
  static constexpr const char* kName = "ROCM_ATTN";

  std::string get_name() const override { return kName; }

  // rocm_attn.py:181-190 — MultipleOf(16), for the same reason
  // get_kv_cache_shape below refuses block_size % 16 != 0: the native ROCm
  // paged-attn kernel is LDS-bound to 16/32. DECLARING it is what lets the
  // registry refuse an unsupported block size while SELECTING a backend
  // (validate_configuration's "block_size not supported"), rather than letting
  // a selected backend throw at allocation. Omitting it left the base
  // MultipleOf(1) in place, so this backend advertised every block size and
  // then refused most of them (#1608).
  std::vector<int> get_supported_kernel_block_sizes() const override { return {16}; }

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

  // triton_mla.py:100-103 supports_block_size — block_size % 16 == 0. Declared as
  // the SUPPORTED-SIZE LIST rather than as an overridden predicate, which is
  // upstream's shape and which the base supports_block_size (backend.py:175-192)
  // turns into the same `% 16 == 0` answer. Was a `static` member before #1332;
  // a static of that name would have HIDDEN the base virtual, so a caller
  // through an AttentionBackend& and a caller through a TritonMLABackend& would
  // have got different answers.
  std::vector<int> get_supported_kernel_block_sizes() const override { return {16}; }

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
