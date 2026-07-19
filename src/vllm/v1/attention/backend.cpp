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

namespace {
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
