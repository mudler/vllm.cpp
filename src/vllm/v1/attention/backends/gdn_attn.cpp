// Ported from: vllm/v1/attention/backends/gdn_attn.py @ e24d1b24
#include "vllm/v1/attention/backends/gdn_attn.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "vllm/v1/attention/registry.h"

namespace vllm::v1 {

// Ported from utils.py::split_decodes_and_prefills @ e24d1b24 (lines 564-633),
// T0 subset: require_uniform=False, treat_short_extends_as_decodes=True.
std::tuple<int, int, int, int> SplitDecodesAndPrefills(
    const CommonAttentionMetadata& m, int decode_threshold) {
  const int max_query_len = m.max_query_len;
  const int num_reqs = m.num_reqs;
  const int num_tokens = m.num_actual_tokens;
  const std::vector<int32_t>& qsl = m.query_start_loc_cpu;

  // treat_short_extends_as_decodes (default): a batch whose longest query is
  // within the threshold is all decodes (utils.py:599-604).
  if (max_query_len <= decode_threshold) {
    return {num_reqs, 0, num_tokens, 0};
  }

  const std::vector<int32_t> query_lens = m.naive_query_lens();
  // First request is not a decode ⇒ no decodes (utils.py:607-609). This is the
  // decode-first ordering contract: a prefill at the front means the batch is
  // all prefills.
  if (!query_lens.empty() && query_lens[0] > decode_threshold) {
    return {0, num_reqs, 0, num_tokens};
  }

  // is_prefill[i] = query_lens[i] > decode_threshold; first_prefill = argmax
  // (first True) (utils.py:619-632).
  int first_prefill = -1;
  for (int i = 0; i < static_cast<int>(query_lens.size()); ++i) {
    if (query_lens[i] > decode_threshold) {
      first_prefill = i;
      break;
    }
  }
  if (first_prefill < 0) {
    // No prefill found ⇒ all decodes (utils.py:625-626).
    return {num_reqs, 0, num_tokens, 0};
  }

  const int num_decodes = first_prefill;
  const int num_prefills = num_reqs - num_decodes;
  const int num_decode_tokens = qsl[first_prefill];
  const int num_prefill_tokens = num_tokens - num_decode_tokens;
  return {num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens};
}

GDNAttentionMetadata GDNAttentionMetadataBuilder::build(
    int /*common_prefix_len*/, const CommonAttentionMetadata& m,
    bool /*fast_build*/) {
  GDNAttentionMetadata meta;
  meta.num_actual_tokens = m.num_actual_tokens;

  // context_lens = num_computed_tokens = seq_lens - query_lens
  // (backend.py::compute_num_computed_tokens; gdn_attn.py:180).
  const std::vector<int32_t> query_lens = m.naive_query_lens();
  std::vector<int32_t> context_lens(static_cast<size_t>(m.num_reqs), 0);
  for (int r = 0; r < m.num_reqs; ++r) {
    context_lens[static_cast<size_t>(r)] = m.seq_lens_cpu[static_cast<size_t>(r)] -
                                           query_lens[static_cast<size_t>(r)];
  }

  // Per-request GDN mamba-state block id = block_table column 0
  // (gdn_attn.py:219; mamba_get_block_table_tensor identity for the "all"/"none"
  // cache modes — align-mode gather is deferred, T0 uses column 0 directly).
  std::vector<int32_t> state_indices(static_cast<size_t>(m.num_reqs), 0);
  for (int r = 0; r < m.num_reqs; ++r) {
    const size_t off = static_cast<size_t>(r) *
                       static_cast<size_t>(m.block_table_num_cols);
    if (off < m.block_table_tensor.size()) {
      state_indices[static_cast<size_t>(r)] = m.block_table_tensor[off];
    }
  }

  // ── Non-spec (T0) path (gdn_attn.py:211-223). Spec-decode is deferred: T0 has
  // no speculative config, so spec_sequence_masks is always None upstream and we
  // take this branch unconditionally. ──
  const auto [num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens] =
      SplitDecodesAndPrefills(m, /*decode_threshold=*/1);
  meta.num_decodes = num_decodes;
  meta.num_prefills = num_prefills;
  meta.num_decode_tokens = num_decode_tokens;
  meta.num_prefill_tokens = num_prefill_tokens;
  meta.num_spec_decodes = 0;
  meta.num_spec_decode_tokens = 0;

  meta.non_spec_state_indices_tensor = state_indices;
  meta.non_spec_query_start_loc = m.query_start_loc;

  // ── Prefill-kernel inputs + has_initial_state (gdn_attn.py:333-405). Only
  // computed when there is prefill work; decode-only batches leave
  // has_initial_state == None (a decode always continues a sequence). ──
  if (num_prefills > 0) {
    // has_initial_state (full batch, decode-first order) = context_lens > 0.
    std::vector<uint8_t> has_initial_state(static_cast<size_t>(m.num_reqs), 0);
    for (int r = 0; r < m.num_reqs; ++r) {
      has_initial_state[static_cast<size_t>(r)] =
          context_lens[static_cast<size_t>(r)] > 0 ? 1 : 0;
    }
    meta.has_initial_state = has_initial_state;

    if (num_decodes > 0) {
      // MIXED batch: peel the leading decodes off and rebase the prefill-only
      // cu_seqlens / state indices / mask (gdn_attn.py:340-354, 400-401).
      std::vector<int32_t> pqsl;
      pqsl.reserve(m.query_start_loc.size() - static_cast<size_t>(num_decodes));
      for (size_t i = static_cast<size_t>(num_decodes);
           i < m.query_start_loc.size(); ++i) {
        pqsl.push_back(m.query_start_loc[i] - num_decode_tokens);
      }
      meta.prefill_query_start_loc = pqsl;

      std::vector<int32_t> psi(state_indices.begin() +
                                   static_cast<std::ptrdiff_t>(num_decodes),
                               state_indices.end());
      meta.prefill_state_indices = psi;

      std::vector<uint8_t> phis(has_initial_state.begin() +
                                    static_cast<std::ptrdiff_t>(num_decodes),
                                has_initial_state.end());
      meta.prefill_has_initial_state = phis;
    } else {
      // Prefill-only: the whole non-spec sub-batch is the prefill sub-batch
      // (gdn_attn.py:352-354, 403).
      meta.prefill_query_start_loc = m.query_start_loc;
      meta.prefill_state_indices = state_indices;
      meta.prefill_has_initial_state = has_initial_state;
    }
  }
  // else: has_initial_state / prefill_* stay nullopt (gdn_attn.py:405).

  // Spec-decode fields stay nullopt (DEFERRED T0 stub — see header).
  return meta;
}

std::vector<int64_t> GDNAttentionBackend::get_kv_cache_shape(
    int64_t /*num_blocks*/, int64_t /*block_size*/, int64_t /*num_kv_heads*/,
    int64_t /*head_size*/, const std::string& /*cache_dtype_str*/) const {
  // GDN is a state-space backend: its cache is the mamba state (shape from
  // MambaSpec, not a paged KV cache). Upstream GDNAttentionBackend intentionally
  // does not implement get_kv_cache_shape.
  throw std::logic_error(
      "GDN_ATTN is an SSM backend; the mamba-state shape comes from MambaSpec, "
      "not get_kv_cache_shape.");
}

namespace {
// GDN_ATTN self-registers for discoverability in the attention-backend registry
// (mirrors upstream @register_backend(AttentionBackendEnum.GDN_ATTN),
// registry.py:180). It is an SSM (mamba) backend selected PER LAYER by the model
// architecture (the linear-attention layers), NOT via the paged-attention
// priority walk — so it is intentionally absent from the CUDA/CPU
// get_attn_backend_priority lists. Registration makes it constructible through
// MakeAttentionBackend(kCUDA/kCPU, "GDN_ATTN") without an inline code edit.
AttentionBackendFactory MakeGDNAttentionBackend = []() -> std::unique_ptr<AttentionBackend> {
  return std::make_unique<GDNAttentionBackend>();
};
const AttentionBackendRegistrar kGdnAttnCuda{vt::DeviceType::kCUDA,
                                             GDNAttentionBackend::kName,
                                             MakeGDNAttentionBackend};
const AttentionBackendRegistrar kGdnAttnCpu{vt::DeviceType::kCPU,
                                            GDNAttentionBackend::kName,
                                            MakeGDNAttentionBackend};
}  // namespace

}  // namespace vllm::v1
