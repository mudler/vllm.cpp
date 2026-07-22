// Ported from: vllm/v1/attention/backend.py @ e24d1b24
// (get_kv_cache_shape from vllm/v1/attention/backends/flash_attn.py @ e24d1b24)
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

std::vector<int64_t> FlashAttentionBackend::get_kv_cache_shape(
    int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
    int64_t head_size, const std::string& /*cache_dtype_str*/) const {
  // flash_attn.py: block size must be a multiple of 16.
  if (block_size % 16 != 0) {
    throw std::invalid_argument("Block size must be a multiple of 16.");
  }
  return {num_blocks, 2, block_size, num_kv_heads, head_size};
}

std::vector<int64_t> TritonMLABackend::get_kv_cache_shape(
    int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
    int64_t head_size, const std::string& /*cache_dtype_str*/) const {
  // triton_mla.py:100-103 supports_block_size.
  if (!supports_block_size(block_size)) {
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
const AttentionBackendRegistrar kFlashAttnCuda{vt::DeviceType::kCUDA,
                                               FlashAttentionBackend::kName,
                                               MakeFlashAttentionBackend};
const AttentionBackendRegistrar kFlashAttnCpu{vt::DeviceType::kCPU,
                                              FlashAttentionBackend::kName,
                                              MakeFlashAttentionBackend};
}  // namespace

}  // namespace vllm::v1
