// MLA decode attention — the CPU REFERENCE for vt::MlaDecodeAttention
// (MLA campaign W4).
//
// Ported from (both sides cited per the ground-every-impl rule):
//   * SEMANTICS: vllm/v1/attention/backends/mla/triton_mla.py:189-260
//     (`TritonMLAImpl.forward_mqa`) @ pin e24d1b24 — the MQA decode over the
//     compressed latent, which passes ONE buffer as both K and V
//     (`kv_c_and_k_pe_cache` / `kv_c_and_k_pe_cache[..., :kv_lora_rank]`,
//     `:236-244`) with `is_mla=True`.
//   * NUMERICS: vllm/csrc/cpu/mla_decode.cpp `mla_decode_kvcache_cpu_impl`
//     @ e24d1b24 — upstream's own CPU MLA decode: per (request, head) it walks
//     the block table, accumulates a running max + running exp-sum + a running
//     f32 output accumulator over the gathered latent rows, and rescales on
//     every new max. Upstream's `head_dim`/`v_head_dim` split is exactly ours:
//     the QK dot spans the FULL head_dim (576) while V is the LEADING
//     v_head_dim (512) slice of the same row.
//   * The upstream TEST that pins this behaviour is
//     vllm/tests/kernels/attention/test_mla_decode_cpu.py:13-33 (`ref_mla`,
//     "gather and flatten KV-cache", `v = kv[:, :, :v_head_dim]`,
//     `F.scaled_dot_product_attention(..., enable_gqa=True)`), ported in
//     tests/vt/test_ops_mla_attn.cpp.
//
// DELIBERATELY NOT the two-stage split-KV form. The CUDA impl
// (src/vt/cuda/cuda_mla_attn.cu) ports the split+combine Triton pair; this CPU
// side is the INDEPENDENT single-pass oracle it is gated against, so a bug in
// the split schedule cannot hide behind a matching reference. Accumulation is
// f32 throughout (upstream's CPU kernel accumulates in f32 too), so CPU and CUDA
// agree to f32 online-softmax tolerance, not bit-for-bit — the split schedule
// changes the summation ORDER by construction.
//
// A request with seq_len == 0 has no keys at all; upstream's Triton stage 2
// would divide by e_sum == 0. Our contract (validated in ops.cpp) is
// seq_lens >= 0, and both impls agree to write ZEROS for such a row rather than
// NaN — recorded as a deviation because upstream never reaches the case (a
// scheduled decode request always has >= 1 computed token).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

// Read one element of a float tensor of dtype `dt` at flat element offset `i`.
float LoadF(const void* base, DType dt, int64_t i) {
  switch (dt) {
    case DType::kF32: return static_cast<const float*>(base)[i];
    case DType::kBF16: return BF16ToF32(static_cast<const uint16_t*>(base)[i]);
    case DType::kF16: return F16ToF32(static_cast<const uint16_t*>(base)[i]);
    default: VT_CHECK(false, "cpu mla_decode_attention: unsupported float dtype"); return 0.0f;
  }
}

void StoreF(void* base, DType dt, int64_t i, float v) {
  switch (dt) {
    case DType::kF32: static_cast<float*>(base)[i] = v; break;
    case DType::kBF16: static_cast<uint16_t*>(base)[i] = F32ToBF16(v); break;
    case DType::kF16: static_cast<uint16_t*>(base)[i] = F32ToF16(v); break;
    default: VT_CHECK(false, "cpu mla_decode_attention: unsupported float dtype");
  }
}

void MlaDecodeAttentionKernel(Queue&, Tensor& out, Tensor* lse, const Tensor& query,
                              const Tensor& kv_cache, const Tensor& block_table,
                              const Tensor& seq_lens, const MlaDecodeAttentionArgs& args) {
  const int64_t batch = query.shape[0];
  const int64_t heads = query.shape[1];
  const int64_t head_size = query.shape[2];      // D  (576)
  const int64_t v_head_dim = out.shape[2];       // Dv (512)
  const int64_t block_size = kv_cache.shape[1];
  const int64_t max_blocks = block_table.shape[1];
  const float scale = args.scale;

  const int32_t* seq = seq_lens.Ptr<int32_t>();
  const int32_t* bt = block_table.Ptr<int32_t>();

  std::vector<float> acc(static_cast<size_t>(v_head_dim));
  std::vector<float> q_row(static_cast<size_t>(head_size));

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t seq_len = seq[b];
    for (int64_t h = 0; h < heads; ++h) {
      const int64_t q_off = b * query.stride[0] + h * query.stride[1];
      for (int64_t d = 0; d < head_size; ++d) q_row[static_cast<size_t>(d)] = LoadF(query.data, query.dtype, q_off + d);

      // Running online softmax (mla_decode.cpp: `max_val` / `acc_lse` / `acc_out`).
      float m = -std::numeric_limits<float>::infinity();
      float l = 0.0f;
      std::fill(acc.begin(), acc.end(), 0.0f);

      for (int64_t j = 0; j < seq_len; ++j) {
        const int64_t blk_slot = j / block_size;
        VT_CHECK(blk_slot < max_blocks,
                 "cpu mla_decode_attention: seq_len exceeds the block_table row");
        const int64_t blk = bt[b * block_table.stride[0] + blk_slot];
        const int64_t entry = blk * kv_cache.stride[0] + (j % block_size) * kv_cache.stride[1];

        float qk = 0.0f;
        for (int64_t d = 0; d < head_size; ++d) {
          qk += q_row[static_cast<size_t>(d)] * LoadF(kv_cache.data, kv_cache.dtype, entry + d);
        }
        qk *= scale;

        const float m_new = std::max(m, qk);
        const float rescale = std::isinf(m) ? 0.0f : std::exp(m - m_new);
        const float p = std::exp(qk - m_new);
        for (int64_t d = 0; d < v_head_dim; ++d) {
          acc[static_cast<size_t>(d)] =
              acc[static_cast<size_t>(d)] * rescale +
              p * LoadF(kv_cache.data, kv_cache.dtype, entry + d);
        }
        l = l * rescale + p;
        m = m_new;
      }

      const int64_t o_off = b * out.stride[0] + h * out.stride[1];
      const float inv = l > 0.0f ? 1.0f / l : 0.0f;
      for (int64_t d = 0; d < v_head_dim; ++d) {
        StoreF(out.data, out.dtype, o_off + d, acc[static_cast<size_t>(d)] * inv);
      }
      if (lse != nullptr) {
        // triton_decode_attention.py:636 `lse_val = e_max + tl.log(e_sum)`.
        lse->Ptr<float>()[b * lse->stride[0] + h] =
            l > 0.0f ? m + std::log(l) : -std::numeric_limits<float>::infinity();
      }
    }
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(
        OpId::kMlaDecodeAttention, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<MlaDecodeAttentionFn>(&MlaDecodeAttentionKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
