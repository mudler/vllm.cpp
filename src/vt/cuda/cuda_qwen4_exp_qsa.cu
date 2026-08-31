// CUDA kernels for Qwen Sparse Attention (Qwen4-Exp / `Qwen3.8-Flash-Next`) —
// `vt::Qwen4ExpQsaCompress` and `vt::Qwen4ExpQsaGatherAttention`.
// Row MODEL-MM-QWEN4-EXP W6-CUDA-B, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (`src/vt/cuda/cuda_qwen4_exp.cu`, `src/vt/cuda/cuda_qwen4_exp_ple.cu`): no
// existing kernel TU and no shared op array is edited. The file name mirrors its
// CPU sibling `src/vt/cpu/cpu_qwen4_exp_qsa.cpp` one for one, as CMakeLists.txt
// already pairs the `cpu_qwen4_exp*.cpp` units with `cuda_qwen4_exp*.cu`.
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// The CPU arms in `src/vt/cpu/cpu_qwen4_exp_qsa.cpp`, index for index. Those are
// themselves the port of, with the row's oracle split recorded in the spec's
// `## Oracles` (transformers supplies the ALGORITHM, vLLM the OP FORM):
//
//   ALGORITHM  transformers v5.16.0 (the lane pin,
//              `.agents/oracles/transformers.md`),
//              `models/qwen4_exp/modeling_qwen4_exp.py`
//                ::Qwen4ExpTextRMSNorm             (:167-179)
//                ::apply_rotary_pos_emb            (:566-604)
//                ::Qwen4ExpTextQSAIndexer.forward  (:677-717)
//   OP FORM    vLLM @ origin/main 6a5e8f5979,
//              `models/deepseek_v4/common/ops/fused_compress_quant_cache.py`
//                the TRITON head_dim=128 `_fused_kv_compress_norm_rope_insert_
//                indexer_attn` (:677-830) — boundary predicate :729-731, gather
//                window :735-736, paged store :783-795, block-start RoPE :816.
//                SCAFFOLDING ONLY, for the reason the CPU sibling's header
//                gives: DSv4's pool is a LEARNED softmax over an OVERLAPPING
//                window driven by a score channel this checkpoint has no tensor
//                for. vLLM has never registered `qwen4_exp` at any revision.
//
// The KERNEL STRUCTURE (`Check`, `AsStream`, the runtime dtype tag, the
// unconditional `Registrar`) is mirrored from `src/vt/cuda/cuda_qwen4_exp_ple.cu`.
//
// ─── WHAT IS DELIBERATELY NOT IN THIS FILE, unchanged from the CPU sibling ────
// The MQA block SCORE and the per-query top-k. `vt::DsaIndexerLogits` and
// `vt::DsaTopkSelect` already are those two functions and both have CUDA arms
// (`src/vt/cuda/cuda_dsa_indexer.cu`); writing a QSA-private copy beside them
// would be the parallel path AGENTS.md "Shared seams" forbids.
//
// ─── THE PRECISION CONTRACT, WHICH IS THE CPU ARM'S AND NOT A NEW ONE ────────
// The host provider is pinned to `-ffp-contract=off` (CMakeLists.txt:41-56) and
// nvcc's `-fmad` defaults to ON and is NOT pinned, so every multiply-add that
// must match the host is spelled `__fmul_rn` / `__fadd_rn` here — the measure
// `cuda_conv1d_general.cu:138-141` and `cuda_qwen4_exp.cu` already take. The
// divides are `__fdiv_rn` / `__frcp_rn` and the roots are `sqrtf`; IEEE-754
// requires both correctly rounded, nvcc's defaults (`-prec-div=true`,
// `-prec-sqrt=true`) keep them so, and glibc gives the same guarantee, so those
// agree by the standard rather than by an intrinsic.
//
//   `vt::Qwen4ExpQsaCompress` IS BYTE-IDENTICAL to its CPU arm and its gate is a
//   `memcmp`, in BOTH arms of `round_intermediates_to_bf16`. It contains no
//   transcendental, every reduction runs in the host's ascending f32 order, and
//   `__float2bfloat16` is round-to-nearest-even exactly as the host `F32ToBF16`
//   is. A tolerance would be the wrong instrument: a rope applied at the block's
//   LAST position instead of its first, or a pool over an OVERLAPPING window,
//   lands inside any epsilon anyone would write.
//
//   `vt::Qwen4ExpQsaGatherAttention` CANNOT reach that, and exactly one function
//   is why: `exp`. The CPU arm calls `std::exp` on a float; this arm calls
//   `expf`; CUDA documents up to 2 ulp for `expf` while glibc's is correctly
//   rounded. Every other operation on both paths is IEEE-exact or spelled with
//   an `_rn` intrinsic and every reduction runs in the CPU arm's order, so the
//   two arms are expected to differ only where `exp` moved. The gate MEASURES
//   that difference and reports it rather than holding a bound chosen to pass —
//   the position `cuda_qwen4_exp_ple.cu` takes for the same reason.
//
// ─── THE FOUR ORDERS THE GATHER ARM PRESERVES ────────────────────────────────
// The CPU arm's header calls the ascending visit order "load-bearing, not
// cosmetic: a gather's visit order IS the softmax's reduction order, and
// ascending is what makes a sub-budget gather reduce over exactly the dense
// sequence and so be BIT-identical to dense attention rather than merely close".
// This arm keeps all four:
//
//   1. the DOT `sum_d q[d]*k[d]` is sequential ascending f32 on ONE thread,
//      never tree-reduced;
//   2. the MAX over the gathered rows is split across threads and block-reduced,
//      which is exact and order-free — `max` is associative and commutative in
//      IEEE arithmetic and it is the one reduction here that may be parallelised;
//   3. the DENOMINATOR `sum_p w` is accumulated ascending by one thread;
//   4. the VALUE accumulation `acc[d] += w * v[d]` runs ascending over `p` and is
//      parallel across `d` only, which each thread owns alone.
//
// THE COST IS STATED RATHER THAN LEFT TO BE FOUND. Pass 2's dot is sequential on
// thread 0, so a block spends `|sel| * head_dim` dependent f32 operations on one
// lane — ~525k per (token, head) at the released config. The lever that removes
// it is named and NOT taken here: a deterministic tree reduction over `d` would
// PRESERVE the gather-vs-dense property, because the dot's order would then
// depend on `head_dim` alone and be identical in the sub-budget and the dense
// run, while breaking the CPU-vs-CUDA relation the sequential dot keeps. That
// trade is declined on the run that first puts this kernel on a device, because
// it would replace a measured `max|diff|` with an argued one. The spec's
// `## Owed` records it as a SPEED item with its condition.
//
// ─── TWO REFUSALS THIS ARM IMPOSES THAT THE CPU ARM DOES NOT ─────────────────
// `head_dim > 1024` IS REFUSED BY NAME. One thread block serves one
// (token, head) and each thread owns one output dim, so the block cannot be
// wider than the device maximum. `head_dim` is 256 in the released config.
//
// A MALFORMED SELECTION POISONS THE ROW WITH NaN rather than being refused by
// name. The CPU arm `VT_CHECK`s that each block id is ASCENDING and inside the
// complete-block count, that `kv_lens[t]` fits the cache, and that a paged
// entry names a physical page the cache holds. A device kernel cannot throw, and
// `block_ids` / `kv_lens` / `kv_block_table` are DEVICE-resident on a CUDA
// queue, so the host dispatcher cannot read them either — the position
// `cuda_qwen4_exp_ple.cu` already takes for `query_start_loc`, whose wrapper
// checks run `if (q.device.type == kCPU)` only. This arm makes those
// preconditions the caller's AND refuses to read out of range: a violating row
// reads NOTHING out of bounds and writes NaN across its whole output. NaN over a
// clamp or a skip, because both of those return a plausible tensor, and
// `MaxAbsDiff` returns +infinity on any non-finite operand (#449) so a poisoned
// row cannot read as a match in any gate in this tree.
//
// ─── `keys_visited` IS A DEVICE COUNTER NOW, AND THAT DEBT IS DISCHARGED ─────
// `Qwen4ExpQsaAttnArgs::keys_visited` says in its own words: "A host pointer, on
// the `GdnArgs::query_start_loc_host` precedent; a CUDA arm owes a device-side
// counter and its copy-back." This is that. It is INCREMENTED AT THE READ in a
// per-thread counter, block-reduced and `atomicAdd`ed once per block, never
// assigned from the selection — the distinction the args comment spends a
// paragraph on, because W4's first fresh review found a counter set from
// `sel.size()` under which a dense walk still reported the sparse figure. The
// device slot is allocated, zeroed, read back and freed ONLY when the caller
// passed the instrument, so the production path (`keys_visited == nullptr`)
// takes no allocation and no stream synchronisation.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// <math.h>, not <cmath>: CUDA's math headers declare `expf`, `sqrtf` and
// `fmaxf` in the GLOBAL namespace.
#include <math.h>

#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType, the op declarations

// ─── THE GATHER'S ERROR BOUND HAS A COMPILER PREMISE, SO PIN IT ──────────────
// `tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp` gates this file's
// gather against its CPU arm with a bound DERIVED from CUDA's documented 2 ulp
// for `expf`. Under `-use_fast_math`, nvcc maps `expf` to `__expf`, whose error
// is about 2^-21.4 -- far outside 2 ulp -- and the derivation silently becomes
// wrong while the gate keeps reporting a ratio it can no longer justify.
//
// The premise holds today: the only `-use_fast_math` in the build is a
// `set_property(SOURCE ...)` that does not name this translation unit. But a
// GLOBAL one has been considered in-tree, and a numeric contract that depends on
// a compiler flag nobody is watching is exactly the silent-wrong-answer shape
// this row keeps meeting. So it is a compile error rather than a comment.
#if defined(__CUDA_FAST_MATH__)
#error "cuda_qwen4_exp_qsa.cu: -use_fast_math maps expf to __expf (~2^-21.4), which breaks the 2-ulp premise the gather's derived arm-vs-arm bound rests on. Either exclude this TU from fast math or re-derive the bound in tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp."
#endif

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// The device's maximum block width. Both kernels give one thread one element of
// the head dim, so this is the head_dim ceiling the wrappers refuse above.
constexpr int64_t kMaxHeadDim = 1024;

// Local dtype tag + accessors, the `cuda_qwen4_exp_ple.cu` arrangement: a local
// copy rather than a hoist, because hoisting would edit a translation unit
// several other rows are working in. `__float2bfloat16` / `__float2half` are
// round-to-nearest-even, identical to the host `vt::F32ToBF16` / `vt::F32ToF16`,
// so this reads as the CPU arm's `LoadF32At` / `StoreF32At` line for line.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* op, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false, std::string("cuda ") + op + ": unsupported " + what +
                          " dtype (f32/f16/bf16)");
      return DTag::kF32;
  }
}

__device__ inline float LoadAt(const void* p, DTag tag, int64_t i) {
  switch (tag) {
    case DTag::kF32: return static_cast<const float*>(p)[i];
    case DTag::kF16: return __half2float(static_cast<const __half*>(p)[i]);
    default: return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
  }
}

__device__ inline void StoreAt(void* p, DTag tag, int64_t i, float v) {
  switch (tag) {
    case DTag::kF32: static_cast<float*>(p)[i] = v; break;
    case DTag::kF16: static_cast<__half*>(p)[i] = __float2half(v); break;
    default: static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(v); break;
  }
}

// One bf16 round trip, the CPU arm's `MaybeBf16`. Upstream's `.to(raw_keys.
// dtype)` / `.type_as(x)` on a bf16 model path, and a bf16 elementwise op's
// store, each do exactly this.
__device__ inline float MaybeBf16(float x, bool round) {
  return round ? __bfloat162float(__float2bfloat16(x)) : x;
}

// Round the block width up to a whole warp so no reduction below has a partial
// final warp to special-case.
unsigned BlockWidthFor(int64_t head_dim) {
  const int64_t w = ((head_dim + 31) / 32) * 32;
  return static_cast<unsigned>(w < 32 ? 32 : w);
}

// ---------------------------------------------------------------------------
// vt::Qwen4ExpQsaCompress — `Qwen4ExpTextQSAIndexer`'s pooled-key build
// (modeling_qwen4_exp.py:677-688): mean-pool a NON-overlapping window of
// `compress_ratio` raw keys, round back to the cache dtype, `k_layernorm` the
// POOLED key, then RoPE it at the position of the block's FIRST token.
//
// ONE BLOCK PER COMPRESSED BLOCK, one thread per head dim. The pool, the norm's
// multiply and the rope are independent per `d` and run in parallel; the SUM OF
// SQUARES is the one reduction and it is walked SEQUENTIALLY ASCENDING BY THREAD
// 0, in the host's order, because that is the order the goldens were dumped in
// and a tree reduction is a different one. `head_dim` is 128 at the released
// config, so that serial stretch is 128 dependent adds against a block that has
// just done `compress_ratio * head_dim` loads in parallel.
__global__ void QsaCompressKernel(void* block_keys, DTag bk_tag, const void* raw_keys,
                                  DTag raw_tag, const void* k_norm_w, DTag w_tag,
                                  const float* cosp, const float* sinp, int64_t nb, int64_t D,
                                  int64_t CR, int64_t rot, float eps, bool round) {
  extern __shared__ float s_compress[];
  float* s_pooled = s_compress;          // [D]
  float* s_normed = s_compress + D;      // [D]
  __shared__ float s_rrms;
  const int64_t rot_half = rot / 2;  // NOT `half`: cuda_fp16.h typedefs that
  const float d_f = static_cast<float>(D);
  const float cr_f = static_cast<float>(CR);

  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    __syncthreads();  // every thread is done reading the PREVIOUS block's shared

    // 1. THE POOL IS AN UNWEIGHTED MEAN OVER A WINDOW THAT DOES NOT OVERLAP.
    //    This is the one place DeepSeek-V4's compressor must NOT be copied. Each
    //    thread owns one `d` and walks the window ascending, which is the host's
    //    order for the only reduction that is per-`d`.
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
      float acc = 0.0f;
      for (int64_t i = 0; i < CR; ++i) {
        acc = __fadd_rn(acc, LoadAt(raw_keys, raw_tag, (b * CR + i) * D + d));
      }
      s_pooled[d] = MaybeBf16(__fdiv_rn(acc, cr_f), round);
    }
    __syncthreads();

    // 2. `k_layernorm` ON THE POOLED KEY. The weight polarity is upstream's
    //    `out * (1.0 + weight)` with a ZERO-initialised weight, NOT vLLM's
    //    `out * weight` with a ones-initialised one. The sum of squares is f32
    //    in ASCENDING order — the host reference's order and the order the
    //    goldens were dumped in — so thread 0 walks it rather than the block
    //    reducing it.
    if (threadIdx.x == 0) {
      float ss = 0.0f;
      for (int64_t d = 0; d < D; ++d) {
        ss = __fadd_rn(ss, __fmul_rn(s_pooled[d], s_pooled[d]));
      }
      // The epsilon is INSIDE the rsqrt, added to the MEAN SQUARE
      // (modeling_qwen4_exp.py:170). Added to the norm instead it is a different
      // function, and at the model's real eps of 1e-6 the difference hides.
      s_rrms = __frcp_rn(sqrtf(__fadd_rn(__fdiv_rn(ss, d_f), eps)));
    }
    __syncthreads();
    const float rrms = s_rrms;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
      const float w = LoadAt(k_norm_w, w_tag, d);
      s_normed[d] = MaybeBf16(__fmul_rn(__fmul_rn(s_pooled[d], rrms), __fadd_rn(1.0f, w)),
                              round);
    }
    __syncthreads();

    // 3. RoPE AT THE BLOCK'S FIRST TOKEN. Upstream reads it as
    //    `group_starts = block_token_indices[:, 0]`; the Triton kernel computes
    //    the same value as `compressed_pos = (position // CR) * CR`. Taking the
    //    block's LAST position instead is a silent one-block phase error no
    //    shape check can see. The rotation is NeoX `rotate_half` over the
    //    LEADING `rotary_dim` dims — DeepSeek-V4's indexer ropes a TRAILING
    //    contiguous span with adjacent-pair GPT-J pairing, so the halves are
    //    swapped end for end AND the pairing convention differs.
    const int64_t pos = b * CR;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
      if (d < rot) {
        const float rot_src =
            d < rot_half ? -s_normed[d + rot_half] : s_normed[d - rot_half];
        // Two products then a sum. On a bf16 tensor each of those three ops
        // stores a bf16, so each rounds; folding them into one expression would
        // drift from the oracle.
        const float a = MaybeBf16(__fmul_rn(s_normed[d], cosp[pos * rot + d]), round);
        const float c = MaybeBf16(__fmul_rn(rot_src, sinp[pos * rot + d]), round);
        StoreAt(block_keys, bk_tag, b * D + d, MaybeBf16(__fadd_rn(a, c), round));
      } else {
        // The NoPE dims trail and are concatenated back untouched.
        StoreAt(block_keys, bk_tag, b * D + d, s_normed[d]);
      }
    }
  }
}

void Qwen4ExpQsaCompressKernelCuda(Queue& q, Tensor& block_keys, const Tensor& raw_keys,
                                   const Tensor& k_norm_weight, const Tensor& cos,
                                   const Tensor& sin,
                                   const Qwen4ExpQsaCompressArgs& args) {
  constexpr const char* kOp = "qwen4_exp_qsa_compress";
  const int64_t D = raw_keys.shape[1];
  const int64_t CR = args.compress_ratio;
  const int64_t nb = raw_keys.shape[0] / CR;
  if (nb == 0 || D == 0) return;  // empty-work early return, before the launch
  VT_CHECK(D <= kMaxHeadDim,
           std::string("cuda ") + kOp + ": index head_dim " + std::to_string(D) +
               " exceeds the " + std::to_string(kMaxHeadDim) +
               " this arm can give one thread each; the CPU arm has no such limit");
  const DTag bk_tag = TagOf(block_keys.dtype, kOp, "block_keys");
  const DTag raw_tag = TagOf(raw_keys.dtype, kOp, "raw_keys");
  const DTag w_tag = TagOf(k_norm_weight.dtype, kOp, "k_layernorm weight");
  const unsigned width = BlockWidthFor(D);
  const unsigned grid = static_cast<unsigned>(nb < 4096 ? nb : 4096);
  const size_t shared = static_cast<size_t>(2 * D) * sizeof(float);
  QsaCompressKernel<<<grid, width, shared, AsStream(q)>>>(
      block_keys.data, bk_tag, raw_keys.data, raw_tag, k_norm_weight.data, w_tag,
      cos.Ptr<float>(), sin.Ptr<float>(), nb, D, CR, args.rotary_dim, args.eps,
      args.round_intermediates_to_bf16);
  Check(cudaGetLastError(), "qwen4_exp_qsa_compress launch");
}

// ---------------------------------------------------------------------------
// vt::Qwen4ExpQsaGatherAttention — dense GQA over ONLY the gathered rows.
//
// ONE BLOCK PER (query token, query head); one thread per head dim. The two
// address modes are ONE body, as they are on the host: the only thing that forks
// is the resolution of one key/value row address, and a second kernel would be
// the parallel path AGENTS.md "Shared seams" forbids AND would have to be kept
// bit-identical to this one by hand.
//
// How many rows the tile below carries between thread 0 and the block. It exists
// to amortise `__syncthreads`, NOT to change any order: thread 0 still walks `s`
// ASCENDING and still accumulates `denom` ascending, and the block still applies
// the tile's weights to `acc[d]` in ascending `s`. With `|sel|` up to 2051 at the
// released config this turns ~4100 barrier pairs into ~130.
constexpr int kSelTile = 32;

// The one address that differs between the two arms, mirroring vLLM's paged read
// (`vt::PagedAttention`'s own semantics line, ported from
// `flash_attn.py::FlashAttentionImpl.forward`):
//   block = block_table[j / block_size], offset = j % block_size,
//   K = k_cache[block, offset, g, :]
// The row itself is contiguous in both arms (the dispatcher checks
// `stride[3] == 1` for the paged views), so the caller reads `DH` running
// elements from the returned base either way. Returns -1 when the table names a
// page the cache does not hold, which is the CPU arm's `VT_CHECK` turned into a
// value a device kernel can act on: the caller reads NOTHING and poisons the row.
__device__ inline int64_t RowBase(bool paged, const int32_t* pages, int64_t page_size,
                                  int64_t num_phys_pages, const int64_t* strides, int64_t HKV,
                                  int64_t DH, int64_t p, int64_t kvh) {
  if (!paged) return (p * HKV + kvh) * DH;
  const int64_t page = pages[p / page_size];
  if (page < 0 || page >= num_phys_pages) return -1;
  return page * strides[0] + (p % page_size) * strides[1] + kvh * strides[2];
}

__global__ void QsaGatherAttentionKernel(
    void* out, DTag out_tag, const void* query, DTag q_tag, const void* key, DTag k_tag,
    const void* value, DTag v_tag, const int32_t* ids, const int32_t* lens, int64_t T,
    int64_t HQ, int64_t DH, int64_t HKV, int64_t groups, int64_t topk, int64_t CR, float scale,
    bool paged, const int32_t* pages, int64_t page_size, int64_t num_phys_pages,
    int64_t max_kv, int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t v_s0, int64_t v_s1,
    int64_t v_s2, unsigned long long* visited) {
  extern __shared__ float s_attn[];
  float* s_q = s_attn;                       // [DH] the query row
  float* s_red = s_attn + DH;                // [blockDim/32] warp partials
  float* s_wtile = s_red + (blockDim.x / 32);  // [kSelTile]
  __shared__ int64_t s_ptile[kSelTile];
  __shared__ float s_m;
  __shared__ float s_denom;
  __shared__ int s_bad;
  __shared__ int64_t s_nsel_blocks;  // how many block ids the row selected
  __shared__ int64_t s_selcount;     // |sel|, expansions + ragged tail
  __shared__ int64_t s_complete;

  const int64_t k_str[3] = {k_s0, k_s1, k_s2};
  const int64_t v_str[3] = {v_s0, v_s1, v_s2};
  const int64_t pairs = T * HQ;
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x >> 5;
  const unsigned warps = blockDim.x >> 5;

  for (int64_t pair = blockIdx.x; pair < pairs; pair += gridDim.x) {
    __syncthreads();
    const int64_t t = pair / HQ;
    const int64_t h = pair - t * HQ;
    const int64_t kvh = h / groups;
    unsigned long long reads = 0;  // COUNTED AT THE READ; see the header

    if (threadIdx.x == 0) {
      s_bad = 0;
      const int64_t kv_len = static_cast<int64_t>(lens[t]);
      // The CPU arm's `VT_CHECK(kv_len >= 0 && kv_len <= max_kv, ...)`. A device
      // kernel cannot throw, so the row is poisoned instead.
      if (kv_len < 0 || kv_len > max_kv) s_bad = 1;
      const int64_t complete = (kv_len < 0 ? 0 : kv_len) / CR;
      // THE EXPANSION. Selected block `b` IS tokens [CR*b, CR*b + CR). It is
      // resolved as ADDRESSES — never handed in as a
      // `[T, token_budget + compress_ratio - 1]` token buffer, which at the
      // released config would be 8 KiB a token to say what four multiplications
      // of a block id say. Only the COUNT is materialised here; the mapping
      // `s -> p` below is arithmetic.
      int64_t nsel = 0;
      int64_t prev = -1;
      for (int64_t j = 0; j < topk; ++j) {
        const int64_t b = static_cast<int64_t>(ids[t * topk + j]);
        if (b < 0) break;  // -1 terminates; the padding is not a block
        // ASCENDING is load-bearing, not cosmetic: a gather's visit order IS the
        // softmax's reduction order. `vt::DsaTopkSelect` emits exactly this
        // order, and the CPU arm refuses anything else by name.
        if (!(b > prev && b < complete)) { s_bad = 1; break; }
        prev = b;
        ++nsel;
      }
      const int64_t tail = (kv_len < 0 ? 0 : kv_len) - complete * CR;
      s_nsel_blocks = nsel;
      s_complete = complete;
      s_selcount = nsel * CR + tail;
      // "a causal query attends at least itself" — the CPU arm's last check.
      if (s_selcount == 0 && kv_len != 0) s_bad = 1;
    }
    __syncthreads();
    if (s_bad != 0) {
      // POISON THE WHOLE ROW. Nothing out of range was read to get here.
      for (int64_t d = threadIdx.x; d < DH; d += blockDim.x) {
        StoreAt(out, out_tag, (t * HQ + h) * DH + d, nanf(""));
      }
      continue;
    }
    const int64_t selcount = s_selcount;
    const int64_t nsel_blocks = s_nsel_blocks;
    const int64_t complete = s_complete;

    for (int64_t d = threadIdx.x; d < DH; d += blockDim.x) {
      s_q[d] = LoadAt(query, q_tag, (t * HQ + h) * DH + d);
    }
    __syncthreads();

    // Pass 1: the max, over the GATHERED rows only. `max` is associative and
    // commutative in IEEE arithmetic, so this is the ONE reduction here that may
    // be split across threads without changing the answer by a bit. Each
    // thread's DOT is still sequential ascending f32.
    float local_m = -INFINITY;  // the CPU arm's kNegInf
    for (int64_t s = threadIdx.x; s < selcount; s += blockDim.x) {
      int64_t p;
      if (s < nsel_blocks * CR) {
        p = static_cast<int64_t>(ids[t * topk + s / CR]) * CR + (s % CR);
      } else {
        // The incomplete trailing block is ALWAYS attended, whatever the scores
        // said. It is why the index buffer upstream is
        // `budget + compress_ratio - 1` wide and not `budget`.
        p = complete * CR + (s - nsel_blocks * CR);
      }
      const int64_t base = RowBase(paged, pages, page_size, num_phys_pages, k_str, HKV, DH, p,
                                   kvh);
      if (base < 0) { s_bad = 1; continue; }
      ++reads;
      float dot = 0.0f;
      for (int64_t d = 0; d < DH; ++d) {
        dot = __fadd_rn(dot, __fmul_rn(s_q[d], LoadAt(key, k_tag, base + d)));
      }
      local_m = fmaxf(local_m, __fmul_rn(dot, scale));
    }
    // Warp then block reduction of the max. Exact at every step.
    for (int off = 16; off > 0; off >>= 1) {
      local_m = fmaxf(local_m, __shfl_down_sync(0xffffffffu, local_m, off));
    }
    if (lane == 0) s_red[warp] = local_m;
    __syncthreads();
    if (threadIdx.x == 0) {
      float m = s_red[0];
      for (unsigned wi = 1; wi < warps; ++wi) m = fmaxf(m, s_red[wi]);
      s_m = m;
    }
    __syncthreads();
    if (s_bad != 0) {
      for (int64_t d = threadIdx.x; d < DH; d += blockDim.x) {
        StoreAt(out, out_tag, (t * HQ + h) * DH + d, nanf(""));
      }
      continue;
    }
    const float m = s_m;

    // Pass 2: the softmax weights and the value reduction, ASCENDING. Thread 0
    // walks `s` in order and accumulates `denom` in order; the block applies each
    // tile's weights to `acc[d]` in the same order. `kSelTile` amortises the
    // barriers and changes no order at all.
    float acc = 0.0f;  // this thread's acc[d], d == threadIdx.x
    if (threadIdx.x == 0) s_denom = 0.0f;
    __syncthreads();
    for (int64_t s0 = 0; s0 < selcount; s0 += kSelTile) {
      const int64_t n = (selcount - s0) < kSelTile ? (selcount - s0) : kSelTile;
      if (threadIdx.x == 0) {
        float denom = s_denom;
        for (int64_t u = 0; u < n; ++u) {
          const int64_t s = s0 + u;
          int64_t p;
          if (s < nsel_blocks * CR) {
            p = static_cast<int64_t>(ids[t * topk + s / CR]) * CR + (s % CR);
          } else {
            p = complete * CR + (s - nsel_blocks * CR);
          }
          const int64_t base = RowBase(paged, pages, page_size, num_phys_pages, k_str, HKV, DH,
                                       p, kvh);
          if (base < 0) { s_bad = 1; s_wtile[u] = 0.0f; s_ptile[u] = -1; continue; }
          ++reads;
          float dot = 0.0f;
          for (int64_t d = 0; d < DH; ++d) {
            dot = __fadd_rn(dot, __fmul_rn(s_q[d], LoadAt(key, k_tag, base + d)));
          }
          const float w = expf(__fsub_rn(__fmul_rn(dot, scale), m));
          denom = __fadd_rn(denom, w);
          s_wtile[u] = w;
          s_ptile[u] = p;
        }
        s_denom = denom;
      }
      __syncthreads();
      if (threadIdx.x < DH) {
        const int64_t d = threadIdx.x;
        for (int64_t u = 0; u < n; ++u) {
          const int64_t p = s_ptile[u];
          if (p < 0) continue;
          const int64_t vbase = RowBase(paged, pages, page_size, num_phys_pages, v_str, HKV,
                                        DH, p, kvh);
          if (vbase < 0) continue;
          acc = __fadd_rn(acc, __fmul_rn(s_wtile[u], LoadAt(value, v_tag, vbase + d)));
        }
      }
      __syncthreads();
    }
    if (s_bad != 0) {
      for (int64_t d = threadIdx.x; d < DH; d += blockDim.x) {
        StoreAt(out, out_tag, (t * HQ + h) * DH + d, nanf(""));
      }
      continue;
    }
    const float denom = s_denom;
    if (threadIdx.x < DH) {
      StoreAt(out, out_tag, (t * HQ + h) * DH + threadIdx.x, __fdiv_rn(acc, denom));
    }

    // The counter, block-reduced then ONE atomic per block. Every increment
    // above sits inside a read loop; nothing here is derived from the selection.
    if (visited != nullptr) {
      for (int off = 16; off > 0; off >>= 1) {
        reads += __shfl_down_sync(0xffffffffu, reads, off);
      }
      if (lane == 0) atomicAdd(visited, reads);
    }
  }
}

void Qwen4ExpQsaGatherAttentionKernelCuda(Queue& q, Tensor& out, const Tensor& query,
                                          const Tensor& key, const Tensor& value,
                                          const Tensor& block_ids, const Tensor& kv_lens,
                                          const Qwen4ExpQsaAttnArgs& args) {
  constexpr const char* kOp = "qwen4_exp_qsa_gather_attention";
  const bool paged = args.kv_block_table != nullptr;
  const int64_t T = query.shape[0];
  const int64_t HQ = query.shape[1];
  const int64_t DH = query.shape[2];
  const int64_t HKV = key.shape[paged ? 2 : 1];
  const int64_t pairs = T * HQ;
  if (pairs == 0 || DH == 0) return;  // empty-work early return
  VT_CHECK(DH <= kMaxHeadDim,
           std::string("cuda ") + kOp + ": head_dim " + std::to_string(DH) + " exceeds the " +
               std::to_string(kMaxHeadDim) +
               " this arm can give one thread each; the CPU arm has no such limit");
  const int64_t page_size = args.kv_block_size;
  const int64_t num_pages_named = paged ? args.kv_block_table->shape[1] : 0;
  // How many logical token rows the cache can address. Contiguous: its row count.
  // Paged: pages NAMED by the table times the page height — not the physical
  // page count, because the table may name a subset in any order.
  const int64_t max_kv = paged ? num_pages_named * page_size : key.shape[0];
  const DTag out_tag = TagOf(out.dtype, kOp, "out");
  const DTag q_tag = TagOf(query.dtype, kOp, "query");
  const DTag k_tag = TagOf(key.dtype, kOp, "key");
  const DTag v_tag = TagOf(value.dtype, kOp, "value");

  // The device counter and its copy-back, allocated ONLY when the caller passed
  // the instrument. `keys_visited == nullptr` is the production path and takes
  // neither the allocation nor the synchronise.
  unsigned long long* d_visited = nullptr;
  if (args.keys_visited != nullptr) {
    Check(cudaMalloc(&d_visited, sizeof(unsigned long long)),
          "qwen4_exp_qsa_gather_attention keys_visited alloc");
    Check(cudaMemsetAsync(d_visited, 0, sizeof(unsigned long long), AsStream(q)),
          "qwen4_exp_qsa_gather_attention keys_visited zero");
  }

  const unsigned width = BlockWidthFor(DH);
  const unsigned grid = static_cast<unsigned>(pairs < 4096 ? pairs : 4096);
  const size_t shared =
      (static_cast<size_t>(DH) + width / 32 + kSelTile) * sizeof(float);
  QsaGatherAttentionKernel<<<grid, width, shared, AsStream(q)>>>(
      out.data, out_tag, query.data, q_tag, key.data, k_tag, value.data, v_tag,
      block_ids.Ptr<int32_t>(), kv_lens.Ptr<int32_t>(), T, HQ, DH, HKV, HQ / HKV,
      block_ids.shape[1], args.compress_ratio, args.scale, paged,
      paged ? args.kv_block_table->Ptr<int32_t>() : nullptr, page_size,
      paged ? key.shape[0] : 0, max_kv, key.stride[0], key.stride[1], key.stride[2],
      value.stride[0], value.stride[1], value.stride[2], d_visited);
  Check(cudaGetLastError(), "qwen4_exp_qsa_gather_attention launch");

  if (args.keys_visited != nullptr) {
    unsigned long long host = 0;
    Check(cudaMemcpyAsync(&host, d_visited, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost, AsStream(q)),
          "qwen4_exp_qsa_gather_attention keys_visited copy-back");
    Check(cudaStreamSynchronize(AsStream(q)),
          "qwen4_exp_qsa_gather_attention keys_visited sync");
    Check(cudaFree(d_visited), "qwen4_exp_qsa_gather_attention keys_visited free");
    *args.keys_visited = static_cast<int64_t>(host);
  }
}

// Registers the CUDA Qwen4-Exp QSA kernels during static init (pre-main, like
// cuda_layernorm.cu's Registrar). Filling the op table is harmless on a machine
// without a GPU: the kCUDA backend never registers there, so no CUDA queue can
// exist to dispatch with. UNCONDITIONAL and at preprocessor depth 0 — see the
// sibling TU's Registrar comment for what `scripts/check-cuda-op-arch-gate.py`
// requires and why.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQwen4ExpQsaCompress, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpQsaCompressFn>(&Qwen4ExpQsaCompressKernelCuda)));
    RegisterOp(OpId::kQwen4ExpQsaGatherAttention, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Qwen4ExpQsaGatherAttentionFn>(
                   &Qwen4ExpQsaGatherAttentionKernelCuda)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cuda
