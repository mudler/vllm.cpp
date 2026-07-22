// MLA prefill + chunked-context — the CPU REFERENCES for vt::MlaPrefillAttention,
// vt::GatherMlaCache and vt::MergeAttnStates (MLA campaign W5).
//
// Ported from (both sides cited per the ground-every-impl rule, @ pin e24d1b24):
//
//   vt::MlaPrefillAttention
//     * SEMANTICS: vllm/v1/attention/backends/mla/prefill/flash_attn.py:153-248
//       (`FlashAttnPrefillBackend._flash_attn_varlen_diff_headdims` +
//       `run_prefill_new_tokens` (causal) + `run_prefill_context_chunk`
//       (non-causal)) — the ONLY MLA prefill backend reachable on sm_121
//       (`mla/prefill/selector.py:66-76`), OBSERVED selected at W0.
//     * The V ZERO-PAD to the QK width and the output slice-back
//       (`flash_attn.py:164-168`, `:196-197`) live in the CUDA launcher; this
//       reference computes at the true widths, which is the same number.
//     * NUMERICS: a plain TWO-PASS softmax (running max over the row, then the
//       exp-sum, then the weighted sum) — DELIBERATELY a different algorithm
//       from the streaming online-softmax FlashAttention uses, so a bug in the
//       streaming rescale cannot hide behind a matching reference. This is the
//       same independence rule the W4 decode oracle follows.
//
//   vt::GatherMlaCache
//     * vllm/csrc/libtorch_stable/cache_kernels.cu:992-1064
//       (`vllm::gather_and_maybe_dequant_cache`), host wrapper `:1099-1157`.
//       The per-token index arithmetic below is that kernel's `:1013-1031`
//       transcribed scalar-wise.
//
//   vt::MergeAttnStates
//     * vllm/csrc/libtorch_stable/attention/merge_attn_states.cu:18-192
//       (`vllm::merge_attn_states_kernel`) — including BOTH edge cases:
//       `+inf` normalized to `-inf` (`:97-98`) and the both-`-inf` case that
//       would otherwise produce NaN, which emits the PREFIX output (`:100-134`).
//
// The chunked-context DRIVER that composes these three
// (`mla_attention.py:2094-2199 _compute_prefill_context` +
// `:2344-2425 forward_mha`) is device-agnostic C++ and lives in
// include/vllm/model_executor/layers/attention/mla_chunked_context.h.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

float LoadF(const void* base, DType dt, int64_t i) {
  switch (dt) {
    case DType::kF32: return static_cast<const float*>(base)[i];
    case DType::kBF16: return BF16ToF32(static_cast<const uint16_t*>(base)[i]);
    case DType::kF16: return F16ToF32(static_cast<const uint16_t*>(base)[i]);
    default: VT_CHECK(false, "cpu mla prefill: unsupported float dtype"); return 0.0f;
  }
}

void StoreF(void* base, DType dt, int64_t i, float v) {
  switch (dt) {
    case DType::kF32: static_cast<float*>(base)[i] = v; break;
    case DType::kBF16: static_cast<uint16_t*>(base)[i] = F32ToBF16(v); break;
    case DType::kF16: static_cast<uint16_t*>(base)[i] = F32ToF16(v); break;
    default: VT_CHECK(false, "cpu mla prefill: unsupported float dtype");
  }
}

// ─── vt::MlaPrefillAttention ────────────────────────────────────────────────
void MlaPrefillAttentionKernel(Queue&, Tensor& out, Tensor* lse, const Tensor& query,
                               const Tensor& key, const Tensor& value,
                               const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k,
                               const MlaPrefillAttentionArgs& args) {
  const int64_t num_reqs = cu_seqlens_q.shape[0] - 1;
  const int64_t num_heads = query.shape[1];
  const int64_t qk_head_dim = query.shape[2];
  const int64_t v_head_dim = value.shape[2];
  const float scale = args.scale;

  const int32_t* qsl = cu_seqlens_q.Ptr<int32_t>();
  const int32_t* ksl = cu_seqlens_k.Ptr<int32_t>();

  std::vector<float> logits;
  std::vector<float> acc(static_cast<size_t>(v_head_dim));

  for (int64_t b = 0; b < num_reqs; ++b) {
    const int64_t q_begin = qsl[b];
    const int64_t q_end = qsl[b + 1];
    const int64_t k_begin = ksl[b];
    const int64_t k_end = ksl[b + 1];
    const int64_t len_q = q_end - q_begin;
    const int64_t len_k = k_end - k_begin;
    VT_CHECK(len_q >= 0 && len_k >= 0, "cpu mla_prefill_attention: negative cu_seqlens span");
    if (len_q == 0) continue;
    // FlashAttention's BOTTOM-RIGHT causal alignment: query i sees keys
    // j <= i + (len_k - len_q). Identical to the paged-attention convention we
    // already use (ops.h PagedAttentionArgs::window_size comment).
    const int64_t causal_shift = len_k - len_q;

    for (int64_t iq = 0; iq < len_q; ++iq) {
      const int64_t t = q_begin + iq;
      const int64_t visible =
          args.causal ? std::min<int64_t>(len_k, std::max<int64_t>(0, iq + causal_shift + 1))
                      : len_k;
      for (int64_t h = 0; h < num_heads; ++h) {
        const int64_t q_off = t * query.stride[0] + h * query.stride[1];

        // PASS 1 — the raw logits and their max.
        logits.assign(static_cast<size_t>(visible), 0.0f);
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < visible; ++j) {
          const int64_t k_off = (k_begin + j) * key.stride[0] + h * key.stride[1];
          float dot = 0.0f;
          for (int64_t d = 0; d < qk_head_dim; ++d) {
            dot += LoadF(query.data, query.dtype, q_off + d) *
                   LoadF(key.data, key.dtype, k_off + d);
          }
          dot *= scale;
          logits[static_cast<size_t>(j)] = dot;
          m = std::max(m, dot);
        }

        // PASS 2 — the exp-sum, then the weighted sum. No running rescale.
        float l = 0.0f;
        for (int64_t j = 0; j < visible; ++j) {
          const float p = std::exp(logits[static_cast<size_t>(j)] - m);
          logits[static_cast<size_t>(j)] = p;
          l += p;
        }
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int64_t j = 0; j < visible; ++j) {
          const int64_t v_off = (k_begin + j) * value.stride[0] + h * value.stride[1];
          const float p = logits[static_cast<size_t>(j)];
          for (int64_t d = 0; d < v_head_dim; ++d) {
            acc[static_cast<size_t>(d)] += p * LoadF(value.data, value.dtype, v_off + d);
          }
        }

        const int64_t o_off = t * out.stride[0] + h * out.stride[1];
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int64_t d = 0; d < v_head_dim; ++d) {
          StoreF(out.data, out.dtype, o_off + d, acc[static_cast<size_t>(d)] * inv);
        }
        if (lse != nullptr) {
          // FlashAttention's varlen LSE layout is [num_heads, total_q]; a row
          // with no visible keys is -inf, which is what MergeAttnStates'
          // both-(-inf) branch is written for.
          lse->Ptr<float>()[h * lse->stride[0] + t] =
              l > 0.0f ? m + std::log(l) : -std::numeric_limits<float>::infinity();
        }
      }
    }
  }
}

// ─── vt::GatherMlaCache ─────────────────────────────────────────────────────
void GatherMlaCacheKernel(Queue&, Tensor& dst, const Tensor& src_cache,
                          const Tensor& block_table, const Tensor& cu_seq_lens,
                          const Tensor& token_to_seq, const Tensor* seq_starts,
                          int64_t num_tokens) {
  const int64_t block_size = src_cache.shape[1];
  const int64_t head_dim = src_cache.shape[2];
  const int64_t max_blocks = block_table.shape[1];
  const int32_t* bt = block_table.Ptr<int32_t>();
  const int32_t* cu = cu_seq_lens.Ptr<int32_t>();
  const int32_t* t2s = token_to_seq.Ptr<int32_t>();
  const int32_t* starts = seq_starts != nullptr ? seq_starts->Ptr<int32_t>() : nullptr;

  // cache_kernels.cu:1013-1031, scalar-wise.
  for (int64_t token_id = 0; token_id < num_tokens; ++token_id) {
    const int64_t batch_id = t2s[token_id];
    const int64_t batch_start = cu[batch_id];
    const int64_t batch_end = cu[batch_id + 1];
    if (token_id >= batch_end) continue;  // upstream `:1019` early-out
    int64_t batch_offset = token_id - batch_start;
    if (starts != nullptr) batch_offset += starts[batch_id];
    const int64_t block_table_id = batch_offset / block_size;
    const int64_t slot_id = batch_offset % block_size;
    VT_CHECK(block_table_id < max_blocks,
             "cpu gather_mla_cache: chunk offset exceeds the block_table row");
    const int64_t block_id = bt[batch_id * block_table.stride[0] + block_table_id];
    const int64_t cache_off = block_id * src_cache.stride[0] + slot_id * src_cache.stride[1];
    const int64_t dst_off = token_id * dst.stride[0];
    for (int64_t d = 0; d < head_dim; ++d) {
      StoreF(dst.data, dst.dtype, dst_off + d,
             LoadF(src_cache.data, src_cache.dtype, cache_off + d));
    }
  }
}

// ─── vt::MergeAttnStates ────────────────────────────────────────────────────
void MergeAttnStatesKernel(Queue&, Tensor& output, Tensor* output_lse,
                           const Tensor& prefix_output, const Tensor& prefix_lse,
                           const Tensor& suffix_output, const Tensor& suffix_lse,
                           int64_t prefill_tokens_with_context) {
  const int64_t num_tokens = output.shape[0];
  const int64_t num_heads = output.shape[1];
  const int64_t head_size = output.shape[2];
  const int64_t prefix_num_tokens =
      prefill_tokens_with_context < 0 ? num_tokens : prefill_tokens_with_context;

  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t h = 0; h < num_heads; ++h) {
      const int64_t src_off = t * prefix_output.stride[0] + h * prefix_output.stride[1];
      const int64_t suf_off = t * suffix_output.stride[0] + h * suffix_output.stride[1];
      const int64_t dst_off = t * output.stride[0] + h * output.stride[1];

      // `:66-89` — tokens past the with-context prefix take the suffix verbatim.
      if (t >= prefix_num_tokens) {
        for (int64_t d = 0; d < head_size; ++d) {
          StoreF(output.data, output.dtype, dst_off + d,
                 LoadF(suffix_output.data, suffix_output.dtype, suf_off + d));
        }
        if (output_lse != nullptr) {
          output_lse->Ptr<float>()[h * output_lse->stride[0] + t] =
              suffix_lse.Ptr<float>()[h * suffix_lse.stride[0] + t];
        }
        continue;
      }

      float p_lse = prefix_lse.Ptr<float>()[h * prefix_lse.stride[0] + t];
      float s_lse = suffix_lse.Ptr<float>()[h * suffix_lse.stride[0] + t];
      // `:97-98` — a +inf LSE is normalized to -inf before anything else.
      if (std::isinf(p_lse)) p_lse = -std::numeric_limits<float>::infinity();
      if (std::isinf(s_lse)) s_lse = -std::numeric_limits<float>::infinity();
      const float max_lse = std::fmax(p_lse, s_lse);

      // `:100-134` — both -inf would make the merge 0/0; emit the prefix.
      if (std::isinf(max_lse)) {
        for (int64_t d = 0; d < head_size; ++d) {
          StoreF(output.data, output.dtype, dst_off + d,
                 LoadF(prefix_output.data, prefix_output.dtype, src_off + d));
        }
        if (output_lse != nullptr) {
          output_lse->Ptr<float>()[h * output_lse->stride[0] + t] = max_lse;
        }
        continue;
      }

      const float p_se = std::exp(p_lse - max_lse);
      const float s_se = std::exp(s_lse - max_lse);
      const float out_se = p_se + s_se;
      const float p_scale = p_se / out_se;
      const float s_scale = s_se / out_se;
      for (int64_t d = 0; d < head_size; ++d) {
        const float p_out = LoadF(prefix_output.data, prefix_output.dtype, src_off + d);
        const float s_out = LoadF(suffix_output.data, suffix_output.dtype, suf_off + d);
        StoreF(output.data, output.dtype, dst_off + d, p_out * p_scale + s_out * s_scale);
      }
      if (output_lse != nullptr) {
        output_lse->Ptr<float>()[h * output_lse->stride[0] + t] = std::log(out_se) + max_lse;
      }
    }
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMlaPrefillAttention, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<MlaPrefillAttentionFn>(&MlaPrefillAttentionKernel)));
    RegisterOp(OpId::kGatherMlaCache, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GatherMlaCacheFn>(&GatherMlaCacheKernel)));
    RegisterOp(OpId::kMergeAttnStates, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MergeAttnStatesFn>(&MergeAttnStatesKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
