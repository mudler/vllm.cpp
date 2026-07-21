// The CUDA attention-backend priority TABLE — a faithful, complete port of
// vllm/platforms/cuda.py:84-176 `_get_backend_priorities` @ pin e24d1b24 (BOTH
// the MLA branch `:93-142` and the non-MLA branch `:143-166`), plus
// vllm/v1/attention/backends/mla/prefill/selector.py:47-76
// `_get_mla_prefill_backend_priorities`.
//
// It lives in a HEADER, not inside the CUDA-only cuda.cpp TU, for two reasons:
//   * it is pure DATA + a lookup — no CUDA, no device — so the CPU test tier can
//     assert the REAL table instead of a hand-copied duplicate (the pre-W2
//     FakeCudaPlatform in test_attn_backend_registry.cpp was exactly such a
//     duplicate, with a "keep in sync" comment for a maintainer to forget);
//   * a new CUDA architecture, or a re-order, is then a one-ROW edit in one
//     place, with no selector and no platform-class change.
#ifndef VLLM_PLATFORMS_CUDA_ATTN_PRIORITY_H_
#define VLLM_PLATFORMS_CUDA_ATTN_PRIORITY_H_

#include <string>
#include <vector>

#include "vllm/platforms/interface.h"

namespace vllm::platforms {

// Each upstream `if device_capability.major == N` arm is ONE ROW below, keyed on
// (use_mla, major). `kAnyMajor` is the upstream `else` arm and is matched only
// after an exact-major row misses.
constexpr int kAnyMajor = -1;

// cuda.py:94-115 — on sm_100 ONLY, the two SPARSE-MLA entries swap order:
// FlashInfer is preferred for an fp8 KV cache, and for bf16 KV at low head
// counts (`num_heads <= 16`, because FlashMLA pads); otherwise FlashMLA leads.
// Every other row's order is fixed, so this is a per-row policy field rather
// than a branch in the lookup.
enum class SparseTailOrder {
  kFixed,          // the row's list is already in final order
  kSm100Adaptive,  // cuda.py:96-115 — append the two sparse entries, order per cfg
};

struct AttnPriorityRow {
  bool use_mla;
  int major;  // exact compute-capability major, or kAnyMajor for upstream `else`
  std::vector<const char*> backends;
  SparseTailOrder sparse_tail;
};

// Upstream names are the AttentionBackendEnum MEMBER names
// (vllm/v1/attention/backends/registry.py:53-99), which is what our registry is
// keyed on and what SelectAttentionBackendName walks.
inline const std::vector<AttnPriorityRow>& AttnPriorityTable() {
  static const std::vector<AttnPriorityRow> table = {
      // ── MLA ────────────────────────────────────────────────────────────────
      // cuda.py:117-131 — sm_100 (Blackwell datacenter). The two sparse entries
      // are appended by SparseTailOrder::kSm100Adaptive.
      {true, 10,
       {"FLASHINFER_MLA", "TOKENSPEED_MLA", "CUTLASS_MLA", "FLASH_ATTN_MLA",
        "FLASHMLA", "TRITON_MLA"},
       SparseTailOrder::kSm100Adaptive},
      // cuda.py:129-133 — sm_12x, i.e. OUR hardware (GB10 / sm_121 == major 12).
      // Exactly two entries. TRITON_MLA is the dense-MLA decode backend; the
      // FlashInfer sparse SM120 entry is present but is filtered out by the
      // selector's is_sparse() check for a dense request — CONFIRMED at W0 by
      // the oracle logging `Using TRITON_MLA attention backend out of potential
      // backends: ['TRITON_MLA']`. Leaving the sparse name IN the row is
      // deliberate: it is the DSA / sparse-MLA SEAM. When that campaign
      // registers its backend with `is_sparse() == true`, a `use_sparse=true`
      // request selects it here with NO edit to this table or to the selector.
      {true, 12, {"TRITON_MLA", "FLASHINFER_MLA_SPARSE_SM120"}, SparseTailOrder::kFixed},
      // cuda.py:134-142 — sm_90 and older.
      {true, kAnyMajor,
       {"FLASH_ATTN_MLA", "FLASHMLA", "FLASHINFER_MLA", "TRITON_MLA",
        "FLASH_ATTN_MLA_SPARSE", "FLASHMLA_SPARSE"},
       SparseTailOrder::kFixed},
      // ── non-MLA (unchanged behavior; previously an inline if-chain) ────────
      // cuda.py:144-152 — sm_100.
      {false, 10,
       {"FLASHINFER", "FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION", "TURBOQUANT"},
       SparseTailOrder::kFixed},
      // cuda.py:153-166 — everything else, INCLUDING GB10 sm_121 (major 12).
      {false, kAnyMajor,
       {"FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION", "TURBOQUANT"},
       SparseTailOrder::kFixed},
  };
  return table;
}

inline std::vector<std::string> LookupAttnPriority(int major,
                                                   const AttnSelectorConfig& cfg) {
  const AttnPriorityRow* match = nullptr;
  const AttnPriorityRow* fallback = nullptr;
  for (const AttnPriorityRow& row : AttnPriorityTable()) {
    if (row.use_mla != cfg.use_mla) continue;
    if (row.major == major) {
      match = &row;
      break;
    }
    if (row.major == kAnyMajor) fallback = &row;
  }
  if (match == nullptr) match = fallback;
  if (match == nullptr) return {};

  std::vector<std::string> out;
  out.reserve(match->backends.size() + 2);
  for (const char* name : match->backends) out.emplace_back(name);
  if (match->sparse_tail == SparseTailOrder::kSm100Adaptive) {
    // cuda.py:96-115: FlashInfer first for a quantized (fp8) KV cache, or for
    // bf16 KV when num_heads <= 16 (FlashMLA pads at low head counts); else
    // FlashMLA first. `num_heads == 0` is upstream's `None` (unknown).
    const bool prefer_flashinfer =
        cfg.quantized_kv_cache || (cfg.num_heads > 0 && cfg.num_heads <= 16);
    if (prefer_flashinfer) {
      out.emplace_back("FLASHINFER_MLA_SPARSE");
      out.emplace_back("FLASHMLA_SPARSE");
    } else {
      out.emplace_back("FLASHMLA_SPARSE");
      out.emplace_back("FLASHINFER_MLA_SPARSE");
    }
  }
  return out;
}

// vllm/v1/attention/backends/mla/prefill/selector.py:47-76
// `_get_mla_prefill_backend_priorities`. The ROCm arm (`:60-65`) is not ported
// (no ROCm platform yet) and is recorded here rather than silently dropped.
// OBSERVED at W0 on GB10: `Using FLASH_ATTN MLA prefill backend.`
inline std::vector<std::string> LookupMlaPrefillPriority(int major) {
  if (major == 10) {  // selector.py:66-72 (Blackwell datacenter)
    return {"FLASH_ATTN", "TRTLLM_RAGGED", "FLASHINFER", "TOKENSPEED_MLA"};
  }
  return {"FLASH_ATTN"};  // selector.py:73-76 — sm_90/older AND sm_12x (GB10)
}

}  // namespace vllm::platforms

#endif  // VLLM_PLATFORMS_CUDA_ATTN_PRIORITY_H_
