// CPU kernels for Qwen Sparse Attention (Qwen4-Exp / `Qwen3.8-Flash-Next`) —
// `vt::Qwen4ExpQsaCompress` and `vt::Qwen4ExpQsaGatherAttention`.
// Row MODEL-MM-QWEN4-EXP W5b-4 (#2167), spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// This row splits its oracles by developer direction (spec `## Oracles`):
// transformers supplies the ALGORITHM, vLLM supplies the OP FORM. vLLM has never
// registered `qwen4_exp` at any revision, so there is no vLLM kernel to mirror
// for QSA itself; what vLLM supplies is the shape of the DeepSeek-V4 indexer
// lane, which QSA matches on nine independent structural points including the
// literal `compress_ratio == 4`.
//
//   ALGORITHM  transformers v5.16.0 (the lane pin),
//              `models/qwen4_exp/modeling_qwen4_exp.py`
//                ::Qwen4ExpTextRMSNorm             (:167-179)
//                ::apply_rotary_pos_emb            (:566-604)
//                ::Qwen4ExpTextQSAIndexer.forward  (:677-717)
//   OP FORM    vLLM @ origin/main 6a5e8f5979,
//              `models/deepseek_v4/common/ops/fused_compress_quant_cache.py`
//                the TRITON head_dim=128 `_fused_kv_compress_norm_rope_insert_
//                indexer_attn` (:677-830) — boundary predicate :729-731, gather
//                window :735-736, paged store :783-795, block-start RoPE :816.
//                SCAFFOLDING ONLY: its pool is a LEARNED softmax over an
//                OVERLAPPING window (:769), driven by a score channel this
//                checkpoint has no tensor for, and the CuteDSL C4 variant
//                refuses `overlap=False` at compile time.
//
// The landed HOST reference for the same arithmetic is
// `src/vllm/model_executor/models/qwen4_exp_qsa.{h,cpp}` (W4, #1991). These
// kernels are gated against THE SAME lane-pinned goldens that reference answers
// to (`tests/vllm/models/test_qwen4_exp_qsa_device.cpp`), so the two arms answer
// to one oracle instead of to each other.
//
// ─── WHAT IS DELIBERATELY NOT IN THIS FILE ────────────────────────────────────
// The MQA block SCORE and the per-query top-k. `vt::DsaIndexerLogits` computes
// `sum_h fold[t,h] * ReLU(dot(q[t,h,:], k[s,:]))` over a one-KV-head MQA cache
// with a per-query `[win_start, win_end)` window, and with `weights == 1`,
// `q_scale == null` and `n_head_scale == 1` its fold collapses to the single
// constant `softmax_scale`, which is QSA's `1 / sqrt(index_head_dim)` — QSA has
// neither DeepSeek-V4's learned `weights_proj` nor its `n_head ** -0.5`.
// `vt::DsaTopkSelect` is the same all-select-below-k, ties-to-the-lower-index,
// ASCENDING-emission top-k, over the block axis instead of the token axis.
// Writing a QSA-private copy of either beside them would be the parallel path
// AGENTS.md "Shared seams" forbids, exactly as `cpu_dsa_indexer.cpp` says of
// `k_norm` and the leading-slice rope it declines to re-implement.
//
// ─── PRECISION, AND WHY THE ROUNDING FLAG IS NOT THE HOUSE CONTRACT ───────────
// Every reduction here accumulates in f32, in the host reference's order,
// because that is the order the goldens were dumped in.
//
// The house contract elsewhere is "widen on load, compute in f32, round ONCE on
// the store", and `Qwen4ExpQsaCompressArgs::round_intermediates_to_bf16` breaks
// it on purpose. Upstream narrows in the middle three times in eleven lines —
// `.to(raw_keys.dtype)` after the mean pool, `.type_as(x)` closing the RMS norm,
// and a bf16 elementwise RoPE that rounds per operation — and here that rounding
// is LOAD-BEARING rather than cosmetic: the mean pool is the one place a bf16
// round-trip changes which four raw keys a state can represent. An op that could
// not express it could not be gated against a bf16 oracle at all, which is why
// this is a flag and not the silent divergence `cpu_qwen4_exp.cpp` records for
// the gated residual. FALSE is the f32 arm and is the house contract.
//
// ─── W5d-3 (#2249 item 2): THE PAGED ADDRESS MODE ────────────────────────────
// `Qwen4ExpQsaGatherAttentionKernel` below serves two cache shapes, and the fork
// is four lines: the resolution of ONE key/value row address. The engine
// allocates this model's QSA K/V as a PAGED `FullAttentionSpec` group
// (`MakeQwen4ExpKVCache`), so the contiguous arm alone could serve nothing a
// runner hands a forward; the paged arm mirrors vLLM's paged read exactly as
// `vt::PagedAttention` states it (`block = block_table[j / block_size]`,
// `offset = j % block_size`). Everything else — the expansion, the ascending
// visit order, the two softmax passes, the f32 accumulation — is one body, so
// the two arms cannot drift apart the way two kernels would.
//
// A CUDA ARM IS OWED, NOT WRITTEN. It cannot be gated on a CPU-only host and an
// ungated kernel is worse than an absent one; nothing here registers for any
// device but kCPU, so the dispatcher refuses by name on every other one rather
// than silently falling back. That arm additionally owes a DEVICE-side
// `keys_visited` counter and its copy-back — the host pointer below is a host
// kernel's instrument and cannot survive a launch.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType, the op declarations

namespace vt::cpu {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Local dtype accessors, the `cpu_layernorm.cpp` / `cpu_qwen4_exp.cpp`
// arrangement: `cpu_ops.cpp`'s are file-static there and hoisting them would
// edit a translation unit several other rows are working in.
float LoadF32At(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "qwen4_exp_qsa: unsupported input dtype"); return 0.0f;
  }
}

void StoreF32At(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kF16: t.Ptr<uint16_t>()[i] = F32ToF16(v); break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "qwen4_exp_qsa: unsupported output dtype");
  }
}

// One bf16 round trip. Upstream's `.to(raw_keys.dtype)` / `.type_as(x)` on a
// bf16 model path, and a bf16 elementwise op's store, both do exactly this.
// `F32ToBF16` rounds to nearest even, which is what torch does.
inline float MaybeBf16(float x, bool round) {
  return round ? BF16ToF32(F32ToBF16(x)) : x;
}

// `Qwen4ExpTextQSAIndexer`'s pooled-key build (modeling_qwen4_exp.py:677-688):
// mean-pool a NON-overlapping window of `compress_ratio` raw keys, round back to
// the cache dtype, `k_layernorm` the POOLED key, then RoPE it at the position of
// the block's FIRST token.
[[maybe_unused]] void Qwen4ExpQsaCompressKernel(Queue&, Tensor& block_keys,
                                                const Tensor& raw_keys,
                                                const Tensor& k_norm_weight, const Tensor& cos,
                                                const Tensor& sin,
                                                const Qwen4ExpQsaCompressArgs& args) {
  const int64_t D = raw_keys.shape[1];
  const int64_t CR = args.compress_ratio;
  const int64_t nb = raw_keys.shape[0] / CR;
  const int64_t rot = args.rotary_dim;
  const int64_t half = rot / 2;
  const bool round = args.round_intermediates_to_bf16;
  const float* cosp = cos.Ptr<float>();
  const float* sinp = sin.Ptr<float>();

  std::vector<float> pooled(static_cast<size_t>(D));
  std::vector<float> normed(static_cast<size_t>(D));
  for (int64_t b = 0; b < nb; ++b) {
    // 1. THE POOL IS AN UNWEIGHTED MEAN OVER A WINDOW THAT DOES NOT OVERLAP.
    //    This is the one place DeepSeek-V4's compressor must NOT be copied. The
    //    `/ CR` is invisible to almost everything downstream, because
    //    `k_layernorm` runs on the pooled key and RMSNorm is scale-invariant
    //    whenever its epsilon is negligible against the mean square; the gate
    //    for it is a hand-derived case at an eps that dominates.
    for (int64_t d = 0; d < D; ++d) {
      float acc = 0.0f;
      for (int64_t i = 0; i < CR; ++i) acc += LoadF32At(raw_keys, (b * CR + i) * D + d);
      pooled[static_cast<size_t>(d)] = MaybeBf16(acc / static_cast<float>(CR), round);
    }

    // 2. `k_layernorm` ON THE POOLED KEY. The weight polarity is upstream's
    //    `out * (1.0 + weight)` with a ZERO-initialised weight, NOT vLLM's
    //    `out * weight` with a ones-initialised one: dropping the `+ 1` turns
    //    the norm's weighting off entirely and is invisible on a freshly built
    //    module. The sum of squares is f32 in ascending order, the host
    //    reference's order and the order the goldens were dumped in.
    float ss = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
      ss += pooled[static_cast<size_t>(d)] * pooled[static_cast<size_t>(d)];
    }
    // The epsilon is INSIDE the rsqrt, added to the MEAN SQUARE
    // (modeling_qwen4_exp.py:170). Added to the norm instead it is a different
    // function, and at the model's real eps of 1e-6 the difference hides.
    const float rrms = 1.0f / std::sqrt(ss / static_cast<float>(D) + args.eps);
    for (int64_t d = 0; d < D; ++d) {
      const float w = LoadF32At(k_norm_weight, d);
      normed[static_cast<size_t>(d)] =
          MaybeBf16(pooled[static_cast<size_t>(d)] * rrms * (1.0f + w), round);
    }

    // 3. RoPE AT THE BLOCK'S FIRST TOKEN. Upstream reads it as
    //    `group_starts = block_token_indices[:, 0]`; the Triton kernel computes
    //    the same value as `compressed_pos = (position // CR) * CR`. Taking the
    //    block's LAST position instead is a silent one-block phase error that no
    //    shape check can see. The rotation is NeoX `rotate_half` over the
    //    LEADING `rotary_dim` dims — DeepSeek-V4's indexer ropes a TRAILING
    //    contiguous span with adjacent-pair GPT-J pairing, so the halves are
    //    swapped end for end AND the pairing convention differs.
    const int64_t pos = b * CR;
    for (int64_t d = 0; d < rot; ++d) {
      const float rot_src = d < half ? -normed[static_cast<size_t>(d + half)]
                                     : normed[static_cast<size_t>(d - half)];
      // Two products then a sum. On a bf16 tensor each of those three ops stores
      // a bf16, so each rounds; folding them into one f32 expression would drift
      // from the oracle.
      const float a = MaybeBf16(normed[static_cast<size_t>(d)] * cosp[pos * rot + d], round);
      const float c = MaybeBf16(rot_src * sinp[pos * rot + d], round);
      StoreF32At(block_keys, b * D + d, MaybeBf16(a + c, round));
    }
    // The NoPE dims trail and are concatenated back untouched.
    for (int64_t d = rot; d < D; ++d) {
      StoreF32At(block_keys, b * D + d, normed[static_cast<size_t>(d)]);
    }
  }
}

// THE GATHER CONSUMER. Dense GQA over ONLY the gathered rows, batched over query
// tokens.
[[maybe_unused]] void Qwen4ExpQsaGatherAttentionKernel(Queue&, Tensor& out, const Tensor& query,
                                                       const Tensor& key, const Tensor& value,
                                                       const Tensor& block_ids,
                                                       const Tensor& kv_lens,
                                                       const Qwen4ExpQsaAttnArgs& args) {
  const int64_t T = query.shape[0];
  const int64_t HQ = query.shape[1];
  const int64_t DH = query.shape[2];
  // THE PAGED ADDRESS MODE (W5d-3, #2249 item 2). It is the SAME body: the
  // expansion, the ascending visit order, the two softmax passes and the f32
  // accumulation below are shared, and the only thing that forks is the
  // resolution of one key/value row address. A second kernel would be the
  // parallel path AGENTS.md "Shared seams" forbids, and it would have to be
  // kept bit-identical to this one by hand.
  const bool paged = args.kv_block_table != nullptr;
  const int64_t HKV = key.shape[paged ? 2 : 1];
  // How many logical token rows the cache can address. Contiguous: its row
  // count. Paged: pages named by the table times the page height — NOT the
  // physical page count, because the table may name a subset in any order.
  const int64_t page_size = args.kv_block_size;
  const int32_t* pages = paged ? args.kv_block_table->Ptr<int32_t>() : nullptr;
  const int64_t num_pages_named = paged ? args.kv_block_table->shape[1] : 0;
  const int64_t max_kv = paged ? num_pages_named * page_size : key.shape[0];
  const int64_t topk = block_ids.shape[1];
  const int64_t CR = args.compress_ratio;
  const int64_t groups = HQ / HKV;
  const int32_t* ids = block_ids.Ptr<int32_t>();
  const int32_t* lens = kv_lens.Ptr<int32_t>();

  // The one address that differs between the two arms, mirroring vLLM's paged
  // read (`vt::PagedAttention`'s own semantics line, ported from
  // `flash_attn.py::FlashAttentionImpl.forward`):
  //   block = block_table[j / block_size], offset = j % block_size,
  //   K = k_cache[block, offset, g, :]
  // The row itself is contiguous in both arms (the dispatcher checks
  // `stride[3] == 1` for the paged views), so the caller reads `DH` running
  // elements from the returned base either way.
  auto RowBase = [&](const Tensor& c, int64_t p, int64_t kvh) -> int64_t {
    if (!paged) return (p * HKV + kvh) * DH;
    const int64_t page = pages[p / page_size];
    VT_CHECK(page >= 0 && page < c.shape[0],
             "qwen4_exp_qsa_gather_attention: kv_block_table names physical page " +
                 std::to_string(page) + ", outside the " + std::to_string(c.shape[0]) +
                 " pages the cache holds");
    return page * c.stride[0] + (p % page_size) * c.stride[1] + kvh * c.stride[2];
  };

  // THE KEY-ROW READ COUNT, taken AT the read and nowhere else. An earlier
  // revision of the host reference assigned it from the selection, which
  // restates the index buffer instead of measuring the loop, and a body that
  // dot-products every one of the `kv_len` cached rows kept reporting the sparse
  // number with the suite green (W4 fresh review, mutation M22c). Counting at
  // the read makes THIS body's number a function of THIS body's walk.
  //
  // AND THAT IS AS FAR AS IT GOES, which is measured rather than assumed. W5b-4
  // mutation M11 is a dense masked walk that ALSO reports `sel.size() * 2` per
  // head, and it passed 10 cases and 4167 assertions: a counter a kernel writes
  // cannot convict the kernel that writes it, because a mask-shaped port changes
  // the loop and the counter together — that is what a mask-shaped port is. No
  // value comparison convicts it either, since `exp(-inf - m)` is exactly +0.
  // The gate that does is an observable of the WALK: the suite runs this op over
  // a cache whose UNSELECTED rows are NaN, which a gather never addresses and a
  // mask multiplies by a zero weight into `0.0f * NaN` = NaN. This counter's job
  // is the ratio it reports for an honest body, not the conviction of a
  // dishonest one; mutation M11b (the same dense walk with the increment left
  // here) is what says the counter is nonetheless live.
  int64_t reads = 0;

  std::vector<int64_t> sel;
  std::vector<float> qrow(static_cast<size_t>(DH));
  for (int64_t t = 0; t < T; ++t) {
    const int64_t kv_len = lens[t];
    VT_CHECK(kv_len >= 0 && kv_len <= max_kv,
             "qwen4_exp_qsa_gather_attention: kv_lens[" + std::to_string(t) + "] is " +
                 std::to_string(kv_len) + ", past the " + std::to_string(max_kv) +
                 " rows the cache holds");
    const int64_t complete = kv_len / CR;

    // THE EXPANSION. Selected block `b` IS tokens [CR*b, CR*b + CR). It is
    // computed here, per query token, as ADDRESSES — never handed in as a
    // `[T, token_budget + compress_ratio - 1]` token buffer, which at the
    // released config would be 8 KiB a token to say what four multiplications
    // of a block id say.
    sel.clear();
    int64_t prev = -1;
    for (int64_t j = 0; j < topk; ++j) {
      const int64_t b = ids[t * topk + j];
      if (b < 0) break;  // -1 terminates; the padding is not a block
      // ASCENDING is load-bearing, not cosmetic: a gather's visit order IS the
      // softmax's reduction order, and ascending is what makes a sub-budget
      // gather reduce over exactly the dense sequence and so be BIT-identical to
      // dense attention rather than merely close. `vt::DsaTopkSelect` emits
      // exactly this order.
      VT_CHECK(b > prev && b < complete,
               "qwen4_exp_qsa_gather_attention: selected block " + std::to_string(b) +
                   " for query token " + std::to_string(t) +
                   " must be ASCENDING and inside the " + std::to_string(complete) +
                   " COMPLETE blocks visible at kv_len " + std::to_string(kv_len));
      prev = b;
      for (int64_t i = 0; i < CR; ++i) sel.push_back(b * CR + i);
    }
    // The incomplete trailing block is ALWAYS attended, whatever the scores
    // said. It is why the index buffer upstream is `budget + compress_ratio - 1`
    // wide and not `budget`, and it writes no state to the side cache.
    for (int64_t p = complete * CR; p < kv_len; ++p) sel.push_back(p);
    VT_CHECK(!sel.empty() || kv_len == 0,
             "qwen4_exp_qsa_gather_attention: a causal query attends at least itself");

    for (int64_t h = 0; h < HQ; ++h) {
      const int64_t kvh = h / groups;
      for (int64_t d = 0; d < DH; ++d) {
        qrow[static_cast<size_t>(d)] = LoadF32At(query, (t * HQ + h) * DH + d);
      }
      // Pass 1: the max, over the GATHERED rows only.
      float m = kNegInf;
      for (int64_t p : sel) {
        ++reads;
        float dot = 0.0f;
        const int64_t base = RowBase(key, p, kvh);
        for (int64_t d = 0; d < DH; ++d) {
          dot += qrow[static_cast<size_t>(d)] * LoadF32At(key, base + d);
        }
        m = std::max(m, dot * args.scale);
      }
      // Pass 2: the softmax weights and the value reduction, ascending.
      float denom = 0.0f;
      std::vector<float> acc(static_cast<size_t>(DH), 0.0f);
      for (int64_t p : sel) {
        ++reads;
        float dot = 0.0f;
        const int64_t kbase = RowBase(key, p, kvh);
        for (int64_t d = 0; d < DH; ++d) {
          dot += qrow[static_cast<size_t>(d)] * LoadF32At(key, kbase + d);
        }
        const float w = std::exp(dot * args.scale - m);
        denom += w;
        const int64_t vbase = RowBase(value, p, kvh);
        for (int64_t d = 0; d < DH; ++d) {
          acc[static_cast<size_t>(d)] += w * LoadF32At(value, vbase + d);
        }
      }
      for (int64_t d = 0; d < DH; ++d) {
        StoreF32At(out, (t * HQ + h) * DH + d, acc[static_cast<size_t>(d)] / denom);
      }
    }
  }
  if (args.keys_visited != nullptr) *args.keys_visited = reads;
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQwen4ExpQsaCompress, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpQsaCompressFn>(&Qwen4ExpQsaCompressKernel)));
    RegisterOp(OpId::kQwen4ExpQsaGatherAttention, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Qwen4ExpQsaGatherAttentionFn>(
                   &Qwen4ExpQsaGatherAttentionKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
