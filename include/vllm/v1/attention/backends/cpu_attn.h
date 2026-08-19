// Ported from: vllm/v1/attention/backends/cpu_attn.py @ pin 5559679229
//   (CPUAttentionBackend, :39-110 — the CAPABILITY surface and the name.)
//
// Why this file exists, and what it fixes (issue #1371).
//
// `CpuPlatform::get_attn_backend_priority` has always returned
// {CPU_ATTN, FLASH_ATTN}, mirroring cpu.py:75-87, where CPU_ATTN is the ONLY
// answer upstream ever gives on a CPU. Until now the first name was never
// registered, so every CPU selection fell through to FLASH_ATTN — the backend
// whose NHD KV layout our CPU paged-attention kernel happens to share. That
// fallthrough was recorded as behavior-preserving, and it was, right up to the
// moment FLASH_ATTN grew a capability surface: #1332 M1 (369ea7fd4) ported
// `validate_configuration`, and with it flash_attn.py's `head_size % 8 == 0`
// rule. A CPU request whose head size is not a multiple of 8 then matched NO
// registered backend at all and `SelectAttentionBackendName` threw out of
// `GPUModelRunner::initialize_kv_cache`. The FLASH_ATTN rule is correct and is
// untouched here — it describes the FA2 kernel. The defect was that CPU had
// nothing else to fall back TO, because its own backend was a name in a list.
//
// So this registers it. CPU selection now answers CPU_ATTN, which is what
// upstream answers, and FLASH_ATTN stays registered for kCPU as the second
// entry of the same priority list.
//
// ─── TWO RECORDED DEVIATIONS FROM cpu_attn.py ──────────────────────────────
//
// 1. `get_kv_cache_shape` returns the NHD 5-dim shape
//    (num_blocks, 2, block_size, num_kv_heads, head_size), NOT upstream's HND
//    (num_blocks, num_kv_heads, block_size, 2 * head_size) at cpu_attn.py:94-104.
//    This is the SAME long-standing deviation the CPU kernel header already
//    records at src/vt/cpu/cpu_paged_attn.cpp:5-8: our CPU paged-attention
//    kernel reads the FlashAttention NHD layout by tensor strides. The shape a
//    backend reports is checked against the geometry the engine actually
//    allocates (`vllm::v1::CheckKvCacheShape`), so reporting upstream's HND
//    here would fail every CPU run at init. The name a backend answers to is
//    its identity in this registry — the same footing ROCM_ATTN is registered
//    on in backend.h.
//
// 2. `get_supported_head_sizes` declares NO constraint, where cpu_attn.py:58-60
//    lists [32, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512]. That list
//    describes upstream's CPU kernel, which dispatches through fixed-width
//    vectorized templates. Ours does not: src/vt/cpu/cpu_paged_attn.cpp is a
//    per-(request, token, q-head) scalar loop with a two-pass f32 softmax, and
//    the ONLY value it specializes on is the K/V cache dtype. It has no
//    head-size specialization to miss, and it demonstrably runs head sizes
//    outside upstream's list. Declaring upstream's list would refuse work this
//    binary performs correctly — which is the same class of untrue capability
//    claim #1332 was opened for, pointed the other way.
#ifndef VLLM_V1_ATTENTION_BACKENDS_CPU_ATTN_H_
#define VLLM_V1_ATTENTION_BACKENDS_CPU_ATTN_H_

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/v1/attention/backend.h"
#include "vt/dtype.h"

namespace vllm::v1 {

class CpuAttentionBackend final : public AttentionBackend {
 public:
  // cpu_attn.py:63-65.
  static constexpr const char* kName = "CPU_ATTN";

  std::string get_name() const override { return kName; }

  // DEVIATION 1 in the file header — NHD, not upstream's HND.
  std::vector<int64_t> get_kv_cache_shape(
      int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
      int64_t head_size,
      const std::string& cache_dtype_str = "auto") const override;

  // ─── Capability overrides, ported from cpu_attn.py:39-83 ─────────────────
  // cpu_attn.py:42-46 — the CPU backend serves f32 as well as the two half
  // dtypes, and our CPU kernel widens every arm to f32 internally anyway.
  std::vector<vt::DType> supported_dtypes() const override {
    return {vt::DType::kF16, vt::DType::kBF16, vt::DType::kF32};
  }
  // cpu_attn.py:47-52, minus "fp8_e5m2". The KV-FP8 arm of our CPU kernel is
  // e4m3 only (`KvKind::kFp8` -> LoadKvFp8E4M3, cpu_paged_attn.cpp), so e5m2 is
  // omitted rather than claimed. FlashAttentionBackend's list is trimmed for
  // exactly the same reason.
  std::vector<std::string> supported_kv_cache_dtypes() const override {
    return {"auto", "fp8", "fp8_e4m3"};
  }
  // cpu_attn.py:54-56 — MultipleOf(16).
  std::vector<int> get_supported_kernel_block_sizes() const override { return {16}; }
  // DEVIATION 2 in the file header. An empty list is "no constraint"
  // (backend.py:158-161), which is what our scalar CPU kernel actually offers.
  std::vector<int> get_supported_head_sizes() const override { return {}; }
  // cpu_attn.py:66-68.
  bool supports_non_causal() const override { return true; }
  // cpu_attn.py:70-72.
  bool supports_sliding_window() const override { return true; }
  // cpu_attn.py:74-83 — decoder, encoder, encoder-only and encoder-decoder.
  bool supports_attn_type(const std::string& attn_type) const override {
    return attn_type == AttentionTypeName(AttentionType::kDecoder) ||
           attn_type == AttentionTypeName(AttentionType::kEncoder) ||
           attn_type == AttentionTypeName(AttentionType::kEncoderOnly) ||
           attn_type == AttentionTypeName(AttentionType::kEncoderDecoder);
  }
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ATTENTION_BACKENDS_CPU_ATTN_H_
