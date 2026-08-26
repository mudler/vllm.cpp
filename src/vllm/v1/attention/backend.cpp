// Ported from: vllm/v1/attention/backend.py @ pin 5559679229
// (get_kv_cache_shape from vllm/v1/attention/backends/flash_attn.py @ e24d1b24 —
// deliberately NOT re-anchored; see the divergence note at the top of backend.h)
#include "vllm/v1/attention/backend.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "vllm/v1/attention/registry.h"

namespace vllm::v1 {

std::vector<int32_t> CommonAttentionMetadata::naive_query_lens() const {
  std::vector<int32_t> lens;
  if (query_start_loc.size() < 2) {
    return lens;
  }
  lens.reserve(query_start_loc.size() - 1);
  for (size_t i = 1; i < query_start_loc.size(); ++i) {
    lens.push_back(query_start_loc[i] - query_start_loc[i - 1]);
  }
  return lens;
}

CommonAttentionMetadata MakeCommonAttentionMetadata(
    const StepInputs& step, const std::vector<int32_t>& block_table_flat,
    int block_table_num_cols, bool causal, int kv_cache_group_id) {
  if (step.seq_lens.empty()) {
    throw std::invalid_argument(
        "MakeCommonAttentionMetadata requires num_reqs > 0");
  }

  CommonAttentionMetadata cam;
  // query_start_loc / seq_lens come straight from the step inputs; the device
  // and _cpu copies alias the same host data at T0 (header deviation note).
  cam.query_start_loc = step.query_start_loc;
  cam.query_start_loc_cpu = step.query_start_loc;
  cam.seq_lens = step.seq_lens;
  cam.seq_lens_cpu = step.seq_lens;

  cam.num_reqs = static_cast<int>(step.seq_lens.size());
  // num_actual_tokens = total tokens = query_start_loc[-1].
  cam.num_actual_tokens =
      step.query_start_loc.empty() ? 0 : step.query_start_loc.back();

  // max_query_len = longest query = max over per-request query lengths
  // (derivable from the step inputs; equals max num_scheduled_tokens).
  cam.max_query_len = 0;
  const std::vector<int32_t> query_lens = cam.naive_query_lens();
  cam.num_computed_tokens_cpu.reserve(query_lens.size());
  for (size_t i = 0; i < query_lens.size(); ++i) {
    const int32_t len = query_lens[i];
    cam.max_query_len = std::max(cam.max_query_len, len);
    cam.num_computed_tokens_cpu.push_back(cam.seq_lens_cpu[i] - len);
  }
  // max_seq_len = longest context length in the batch.
  cam.max_seq_len =
      *std::max_element(step.seq_lens.begin(), step.seq_lens.end());

  cam.block_table_tensor = block_table_flat;
  cam.block_table_num_cols = block_table_num_cols;

  // slot_mapping: one per KV cache group upstream; select the requested group.
  const auto group = static_cast<size_t>(kv_cache_group_id);
  if (group < step.slot_mapping.size()) {
    cam.slot_mapping = step.slot_mapping[group];
  }

  cam.causal = causal;
  return cam;
}

const char* AttentionTypeName(AttentionType type) {
  // backend.py:38-46 — the exact upstream string values.
  switch (type) {
    case AttentionType::kDecoder:
      return "decoder";
    case AttentionType::kEncoder:
      return "encoder";
    case AttentionType::kEncoderOnly:
      return "encoder_only";
    case AttentionType::kEncoderDecoder:
      return "encoder_decoder";
  }
  return "decoder";
}

// vllm/config/cache.py:19-36 CacheDType. See the header for why f32/f16/bf16 all
// map to "auto".
const char* KvCacheDTypeName(vt::DType dtype) {
  switch (dtype) {
    case vt::DType::kF32:
    case vt::DType::kF16:
    case vt::DType::kBF16:
      return "auto";
    default:
      // No other vt::DType is a KV-cache storage dtype in this tree. "auto" is
      // the unquantized answer, which is what a caller with an unexpected dtype
      // needs the selector to keep doing until the mapping above grows a row.
      return "auto";
  }
}

// vllm/utils/torch_utils.py:75-80.
bool IsQuantizedKvCacheName(const std::string& kv_cache_dtype) {
  const auto ends_with = [&](const std::string& suffix) {
    return kv_cache_dtype.size() >= suffix.size() &&
           kv_cache_dtype.compare(kv_cache_dtype.size() - suffix.size(),
                                  suffix.size(), suffix) == 0;
  };
  return kv_cache_dtype.rfind("fp8", 0) == 0 || ends_with("per_token_head") ||
         kv_cache_dtype == "nvfp4";
}

// backend.py:158-161 — an EMPTY supported list means "no constraint".
bool AttentionBackend::supports_head_size(int head_size) const {
  const std::vector<int> supported = get_supported_head_sizes();
  if (supported.empty()) return true;
  return std::find(supported.begin(), supported.end(), head_size) != supported.end();
}

// backend.py:163-165.
bool AttentionBackend::supports_dtype(vt::DType dtype) const {
  const std::vector<vt::DType> supported = supported_dtypes();
  return std::find(supported.begin(), supported.end(), dtype) != supported.end();
}

// backend.py:167-173. An EMPTY name is upstream's `None`.
bool AttentionBackend::supports_kv_cache_dtype(
    const std::string& kv_cache_dtype) const {
  if (kv_cache_dtype.empty()) return true;
  const std::vector<std::string> supported = supported_kv_cache_dtypes();
  if (supported.empty()) return true;
  return std::find(supported.begin(), supported.end(), kv_cache_dtype) !=
         supported.end();
}

// backend.py:175-192. Every entry of get_supported_kernel_block_sizes() is an
// upstream MultipleOf, so a framework block size is supported when it is a
// multiple of ANY declared size — upstream's hybrid_blocks rule at :187-191,
// which is why 32 passes a backend declaring MultipleOf(16) and 8 does not.
bool AttentionBackend::supports_block_size(int block_size) const {
  if (block_size == 0) return true;  // upstream `None`
  const std::vector<int> supported = get_supported_kernel_block_sizes();
  if (supported.empty()) return true;
  for (const int size : supported) {
    if (size != 0 && block_size % size == 0) return true;
  }
  return false;
}

// backend.py:319-393, in upstream's order, with upstream's reason strings.
std::vector<std::string> AttentionBackend::validate_configuration(
    const platforms::AttnSelectorConfig& cfg,
    const platforms::DeviceCapability& capability) const {
  std::vector<std::string> invalid_reasons;
  if (!supports_head_size(cfg.head_size)) {
    invalid_reasons.emplace_back("head_size not supported");
  }
  if (!supports_dtype(cfg.dtype)) {
    invalid_reasons.emplace_back("dtype not supported");
  }
  if (!supports_kv_cache_dtype(cfg.kv_cache_dtype)) {
    invalid_reasons.emplace_back("kv_cache_dtype not supported");
  }
  if (!supports_block_size(cfg.block_size)) {
    invalid_reasons.emplace_back("block_size not supported");
  }
  if (cfg.use_mm_prefix && !supports_mm_prefix()) {
    invalid_reasons.emplace_back(
        "partial multimodal token full attention not supported");
  }
  if (cfg.use_mla != is_mla()) {
    invalid_reasons.emplace_back(cfg.use_mla ? "MLA not supported"
                                             : "non-MLA not supported");
  }
  if (cfg.has_sink && !supports_sink()) {
    invalid_reasons.emplace_back("attention sinks not supported");
  }
  if (cfg.use_sparse != is_sparse()) {
    invalid_reasons.emplace_back(cfg.use_sparse ? "sparse not supported"
                                                : "non-sparse not supported");
  }
  if (cfg.use_per_head_quant_scales && !supports_per_head_quant_scales()) {
    invalid_reasons.emplace_back("per-head quant scales not supported");
  }
  // backend.py:366-367, with upstream's own PRECONDITION rather than a new rule:
  // CudaPlatform.get_attn_backend_cls asserts `device_capability is not None`
  // before it calls this (cuda.py:403-404), and CpuPlatform has a separate
  // selector that never reaches it (cpu.py:75-87). Our selector is shared across
  // every DeviceType, so the guard is where upstream's caller-side assert lands.
  // Without it, FLASH_ATTN — which this tree also registers for
  // kCPU/kMETAL/kVULKAN/kTENSTORRENT — would be refused on every one of them by
  // a rule about NVIDIA compute capability.
  //
  // #1823: this comment used to argue that `present()` is "already false for
  // every platform that cannot answer the question". THAT WAS NOT TRUE, and it
  // was not true of two platforms at once — Metal answered with the Apple GPU
  // family and Vulkan with the Vulkan API version, so an SM-8.0 bar was compared
  // against numbers that have nothing to do with SM versions. `present()` is a
  // guard on WHETHER a platform answers, and it can say nothing about the UNIT.
  // The unit is a contract on the value, stated on
  // Platform::get_device_capability (include/vllm/platforms/interface.h) and
  // gated for every registered platform by
  // tests/vllm/platforms/test_platform.cpp. This line is correct only because
  // that contract holds.
  if (capability.present() && !supports_compute_capability(capability)) {
    invalid_reasons.emplace_back("compute capability not supported");
  }
  if (!supports_attn_type(cfg.attn_type)) {
    invalid_reasons.emplace_back("attention type " + cfg.attn_type +
                                 " not supported");
  }
  if (cfg.has_sliding_window && !supports_sliding_window()) {
    invalid_reasons.emplace_back("sliding window not supported");
  }
  if (cfg.use_non_causal && !supports_non_causal()) {
    invalid_reasons.emplace_back("non-causal attention not supported");
  }
  if (cfg.use_batch_invariant && !supports_batch_invariance()) {
    invalid_reasons.emplace_back("batch invariance not supported");
  }
  if (cfg.use_kv_connector && !supports_kv_connector()) {
    invalid_reasons.emplace_back("KV connector not supported");
  }
  if (cfg.use_pcp && !supports_pcp()) {
    invalid_reasons.emplace_back("PCP not supported");
  }
  if (const std::optional<std::string> combination =
          supports_combination(cfg, capability);
      combination.has_value()) {
    invalid_reasons.push_back(*combination);
  }
  return invalid_reasons;
}

std::vector<int64_t> FlashAttentionBackend::get_kv_cache_shape(
    int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
    int64_t head_size, const std::string& /*cache_dtype_str*/) const {
  // flash_attn.py: block size must be a multiple of 16.
  if (block_size % 16 != 0) {
    throw std::invalid_argument("Block size must be a multiple of 16.");
  }
  return {num_blocks, 2, block_size, num_kv_heads, head_size};
}

std::vector<int64_t> RocmAttentionBackend::get_kv_cache_shape(
    int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
    int64_t head_size, const std::string& /*cache_dtype_str*/) const {
  // rocm_attn.py:249-251 enforces block_size % 16 == 0 (the native ROCm
  // paged-attn kernel supports LDS-bound block sizes 16/32). The SHAPE is the
  // shared NHD layout, not upstream's (2, num_blocks, ...) — see the class
  // comment in backend.h (KV-LAYOUT DEVIATION). Deliberately identical to
  // FlashAttentionBackend::get_kv_cache_shape above; kept as its own function
  // so the deviation reads in one place.
  if (block_size % 16 != 0) {
    throw std::invalid_argument("Block size must be a multiple of 16.");
  }
  return {num_blocks, 2, block_size, num_kv_heads, head_size};
}

std::vector<int64_t> TritonMLABackend::get_kv_cache_shape(
    int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
    int64_t head_size, const std::string& /*cache_dtype_str*/) const {
  // triton_mla.py:100-103 supports_block_size — now the base predicate reading
  // this class's get_supported_kernel_block_sizes() == {16}.
  if (!supports_block_size(static_cast<int>(block_size))) {
    throw std::invalid_argument("Block size must be a multiple of 16.");
  }
  // mla_attention.py:1219 — "num_kv_heads ... assumed to be 1 for MLA". Upstream
  // ignores the argument; we REFUSE a non-1 value so a caller that wired a
  // GQA-shaped spec into an MLA layer fails loudly instead of allocating a cache
  // the MQA decode cannot read.
  if (num_kv_heads != 1) {
    throw std::invalid_argument(
        "MLA kv cache requires num_kv_heads == 1 (the latent is one head).");
  }
  // THREE dims: no K/V axis (mla_attention.py:1216-1224).
  return {num_blocks, block_size, head_size};
}

std::unique_ptr<AttentionImpl> TritonMLABackend::get_impl_cls() const {
  // triton_mla.py:126-128 `get_impl_cls() -> type[TritonMLAImpl]`. Upstream
  // returns the CLASS and the layer constructs it with num_heads/head_size/scale;
  // we return the instance and the layer fills those public fields.
  return std::make_unique<TritonMLAImpl>();
}

void TritonMLAImpl::forward_mqa(const AttentionLayer& layer, const vt::Tensor& q,
                                const vt::Tensor& kv_c_and_k_pe_cache,
                                const MLACommonMetadata& metadata, vt::Tensor& out,
                                vt::Tensor* lse) const {
  // triton_mla.py:165-171 — the impl rejects a sliding window outright.
  if (layer.window_size.has_value()) {
    throw std::invalid_argument(
        "TritonMLAImpl: sliding window is not supported by MLA "
        "(triton_mla.py:165-171).");
  }
  // `:190` assert kv_c_and_k_pe_cache.numel() > 0
  if (kv_c_and_k_pe_cache.rank != 3 || kv_c_and_k_pe_cache.shape[0] <= 0) {
    throw std::invalid_argument(
        "TritonMLAImpl::forward_mqa: kv cache must be the 3-D MLA cache "
        "(num_blocks, block_size, kv_lora_rank + qk_rope_head_dim).");
  }
  if (scale <= 0.0f) {
    throw std::invalid_argument(
        "TritonMLAImpl::forward_mqa: scale must be set (it carries the mscale^2 "
        "correction for the DeepSeek YaRN configs).");
  }
  vt::MlaDecodeAttentionArgs args;
  args.scale = scale;  // `:253` self.scale
  args.num_kv_splits = metadata.num_kv_splits;
  args.max_seq_len = metadata.max_seq_len;
  // `:242-259` decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, o, lse,
  //   block_table, seq_lens, attn_logits, num_kv_splits, scale, PAGE_SIZE, ...)
  // — the two "K" and "V" arguments are the SAME buffer, which our single
  // kv_cache argument expresses directly.
  vt::Queue default_stream{q.device, nullptr};
  vt::Queue& qq = queue != nullptr ? *queue : default_stream;
  vt::MlaDecodeAttention(qq, out, lse, q, kv_c_and_k_pe_cache, metadata.block_table,
                         metadata.seq_lens, args);
}

void TritonMLAImpl::forward(const AttentionLayer& layer, const vt::Tensor& query,
                            const vt::Tensor& /*key*/, const vt::Tensor& /*value*/,
                            const vt::Tensor& kv_cache, const AttentionMetadata& attn_metadata,
                            vt::Tensor& output, const vt::Tensor* output_scale,
                            const vt::Tensor* output_block_scale) {
  if (output_scale != nullptr || output_block_scale != nullptr) {
    throw std::invalid_argument(
        "TritonMLAImpl: fused output quantization is not ported "
        "(backend_supports_prefill_query_quantization() is False on sm_12x, "
        "mla_attention.py:1382-1385).");
  }
  const auto* md = dynamic_cast<const MLACommonMetadata*>(&attn_metadata);
  if (md == nullptr) {
    throw std::invalid_argument(
        "TritonMLAImpl: attn_metadata must be an MLACommonMetadata.");
  }
  // Decode is one query token per request: query.shape[0] must equal the number
  // of requests the metadata describes. Anything longer is a prefill / chunked
  // batch, which is W5 (the FLASH_ATTN MLA prefill backend), not W4.
  if (query.rank != 3 || md->seq_lens.rank != 1 || query.shape[0] != md->seq_lens.shape[0]) {
    throw std::invalid_argument(
        "TritonMLAImpl::forward: only the pure-DECODE shape is implemented "
        "(one query token per request). MLA prefill is campaign step W5 — see "
        ".agents/specs/mla-deepseek-campaign.md.");
  }
  forward_mqa(layer, query, kv_cache, *md, output, nullptr);
}

namespace {
// TRITON_MLA self-registers for CUDA only — it is a CUDA-capability backend
// (cuda.py:129-133) and there is no CPU MLA backend upstream at the pin, so a
// CPU MLA request correctly finds nothing and throws. Registering the NAME is
// what lets SelectAttentionBackendName resolve use_mla=true on GB10 today; the
// impl arrives at W4/W6.
const AttentionBackendRegistrar kTritonMlaCuda{
    vt::DeviceType::kCUDA, TritonMLABackend::kName,
    []() -> std::unique_ptr<AttentionBackend> {
      return std::make_unique<TritonMLABackend>();
    }};

// FLASH_ATTN self-registers for the device types whose paged-attention KV cache
// uses its NHD (num_blocks,2,block,H,D) layout: CUDA (the gate) and, per the
// cpu_paged_attn.cpp deviation, CPU. Mirrors upstream
// @register_backend(AttentionBackendEnum.FLASH_ATTN) (registry.py) — self-
// registration keeps the selection DATA-driven (registry + platform priority),
// so adding a backend never edits the selector. The backend is device-agnostic
// host metadata (get_kv_cache_shape), so registration lives here, not in a
// device-gated TU. Retained past the linker via the vllm --whole-archive
// INTERFACE option (same as the CPU/CUDA platform + model registrars).
AttentionBackendFactory MakeFlashAttentionBackend = []() -> std::unique_ptr<AttentionBackend> {
  return std::make_unique<FlashAttentionBackend>();
};
// ...and kROCM (issue #41 M3). The ROCm paged-attn kernel
// (src/vt/rocm/rocm_paged_attn.hip) reads and writes the SAME NHD layout
// get_kv_cache_shape allocates — the shared layout is the precondition for this
// line, exactly as the kMETAL / kVULKAN / kTENSTORRENT rows document. The
// factory returns the ROCM_ATTN class (not the FlashAttention one): the name a
// backend answers to IS its identity here, and the test asserts the constructed
// backend's get_name() == "ROCM_ATTN". RocmPlatform::get_attn_backend_priority
// is what decides whether the name is reached (rocm.py:424-434 puts ROCM_ATTN
// first on non-AITER boards).
AttentionBackendFactory MakeRocmAttentionBackend = []() -> std::unique_ptr<AttentionBackend> {
  return std::make_unique<RocmAttentionBackend>();
};
const AttentionBackendRegistrar kRocmAttn{vt::DeviceType::kROCM,
                                          RocmAttentionBackend::kName,
                                          MakeRocmAttentionBackend};
const AttentionBackendRegistrar kFlashAttnCuda{vt::DeviceType::kCUDA,
                                               FlashAttentionBackend::kName,
                                               MakeFlashAttentionBackend};
const AttentionBackendRegistrar kFlashAttnCpu{vt::DeviceType::kCPU,
                                              FlashAttentionBackend::kName,
                                              MakeFlashAttentionBackend};
// ...and kMETAL (BACKEND-METAL-MLX work row M3a). The Metal kPagedAttention /
// kReshapeAndCache kernels (src/vt/metal/metal_ops.mm) read and write the SAME
// NHD (num_blocks, 2, block_size, num_kv_heads, head_size) layout
// get_kv_cache_shape allocates, so this is a NAME registration only — not one
// line of the backend's host metadata is device-specific. Registering here
// rather than in a Metal-gated TU is deliberate and matches the CUDA/CPU rows:
// the shape/metadata logic is device-agnostic, and MetalPlatform::
// get_attn_backend_priority is what decides whether the name is ever reached.
const AttentionBackendRegistrar kFlashAttnMetal{vt::DeviceType::kMETAL,
                                                FlashAttentionBackend::kName,
                                                MakeFlashAttentionBackend};
// ...and kVULKAN, on exactly the same footing (BACKEND-VULKAN work row VK-B). The
// Vulkan kPagedAttention / kReshapeAndCache kernels
// (src/vt/vulkan/shaders/vt_paged_attn.comp, vt_reshape_and_cache.comp) are ports
// of the CPU pair and read and write the SAME NHD
// (num_blocks, 2, block_size, num_kv_heads, head_size) layout get_kv_cache_shape
// allocates -- that shared layout is the actual precondition for this line, not
// the device name. So this is a NAME registration only, and
// VulkanPlatform::get_attn_backend_priority is what decides whether it is reached.
const AttentionBackendRegistrar kFlashAttnVulkan{vt::DeviceType::kVULKAN,
                                                 FlashAttentionBackend::kName,
                                                 MakeFlashAttentionBackend};
// ...and kTENSTORRENT (BACKEND-TENSTORRENT). Host-staged kPagedAttention /
// kReshapeAndCache use the same NHD layout get_kv_cache_shape allocates —
// name registration only; TenstorrentPlatform::get_attn_backend_priority is
// what decides whether the name is reached.
const AttentionBackendRegistrar kFlashAttnTenstorrent{vt::DeviceType::kTENSTORRENT,
                                                      FlashAttentionBackend::kName,
                                                      MakeFlashAttentionBackend};
}  // namespace

}  // namespace vllm::v1
