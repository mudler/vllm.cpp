// ROCm/HIP arm of Qwen Sparse Attention (Qwen4-Exp / `Qwen3.8-Flash-Next`) —
// `vt::Qwen4ExpQsaCompress` and `vt::Qwen4ExpQsaGatherAttention`.
// Row MODEL-MM-QWEN4-EXP, W4 of `.agents/specs/qwen4-exp-rocm-ops.md`,
// issue ISSUE-LOCAL-01M2A1DTCZQVAH7M193XT9PN2V.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (`src/vt/rocm/rocm_qwen4_exp.hip` from W1, `src/vt/rocm/rocm_qwen4_exp_ple.hip`
// from W3): no existing kernel TU and no shared op array is edited, and in
// particular `rocm_ops.hip` is not, because that file is a surface every ROCm
// row writes and AGENTS.md "Records" calls such a file a lock. The file name
// mirrors its CUDA sibling `src/vt/cuda/cuda_qwen4_exp_qsa.cu` and its CPU
// sibling `src/vt/cpu/cpu_qwen4_exp_qsa.cpp` one for one.
//
// ─── WHAT vLLM's AMD BACKEND SAYS ABOUT THESE TWO ────────────────────────────
// Spec D3 makes vLLM's first-party AMD backend the mirror wherever it defines
// behaviour, and unlike W3's PLE pair (D3d: `amd/ple_layer.py` carries ZERO
// Triton kernels) `amd/ops/qsa.py` at the pin `e126687a9a` carries SIX, and
// diverges from `nvidia/ops/qsa.py` by 268 diff lines. So the mirror was read
// kernel by kernel before this file was written, and spec D3e records the
// result in full. In one line, because a reader of this file will ask:
//
//   **THE DIVERGENCE IS REAL AND IT IS NOT IN EITHER OF THESE TWO OPS.**
//
//   - `_compress_qsa_groups_kernel`, which is `vt::Qwen4ExpQsaCompress`'s pool,
//     is BYTE-IDENTICAL between `amd/ops/qsa.py` and `nvidia/ops/qsa.py`. The
//     only change in its neighbourhood is the wrapper's RuntimeError text
//     ("QSA CUDA compression requires Triton" -> "QSA compression requires a GPU
//     and Triton").
//   - `_qsa_sparse_paged_gqa_splitk_kernel`, which is
//     `vt::Qwen4ExpQsaGatherAttention`'s consumer, is BYTE-IDENTICAL too. Its
//     LAUNCH differs by one line — `num_stages = 1 if current_platform.is_rocm()
//     else 2`, with vLLM's own comment "gfx942 and gfx950 have a 64 KiB LDS
//     limit. One software-pipelining stage keeps the wide TP4 tile within that
//     shared-memory budget." That is a Triton software-pipelining depth, i.e. an
//     LDS budget for a tile this arm does not have (this kernel's dynamic shared
//     footprint is `(head_dim + blockDim + 32) * 4` bytes, 5.3 KiB at the
//     released head_dim of 256), so there is no behaviour in it to mirror.
//   - The 268 diff lines are almost entirely `_qsa_mqa_paged_kernel`, the block
//     SCORING kernel, where AMD drops the tensor-core `tl.dot` over a padded
//     head axis for a per-head `tl.sum(keys * query)` loop — and the top-k call,
//     where AMD routes to `ops.top_k_per_row_decode` instead of
//     `torch.ops._C.{cooperative,persistent}_topk`. **Those two are
//     `vt::DsaIndexerLogits` and `vt::DsaTopkSelect` in this tree**, which
//     `qwen4_exp_qsa_block.cpp:391,403` COMPOSES the indexer from and which this
//     file does not own — and which have NO ROCm arm at all. The spec's
//     `## Owed` records that, because it is why the forward still refuses on
//     this board after W4.
//
// So D3's "where they agree, port ours" applies to both ops in this file, and
// the donor is `src/vt/cuda/cuda_qwen4_exp_qsa.cu`, transcribed.
//
// ─── WHAT THE TRANSCRIPTION CHANGED, AND WHY EACH CHANGE ─────────────────────
// The HIP spelling table is W3's: `cuda_runtime.h` -> `hip/hip_runtime.h`,
// `cuda_bf16.h` -> `hip/hip_bf16.h`, `__nv_bfloat16` -> `__hip_bfloat16`,
// `cudaStream_t` -> `hipStream_t`, `cudaGetLastError` -> `hipGetLastError`,
// `cudaMalloc`/`cudaMemsetAsync`/`cudaMemcpyAsync`/`cudaStreamSynchronize`/
// `cudaFree` -> their `hip` spellings, `vt::cuda` -> `vt::rocm`. Beyond that,
// exactly THREE things differ from the donor and none of them is arithmetic:
//
// 1. **NO CROSS-LANE PRIMITIVE, ANYWHERE.** The donor reduces the softmax max
//    and the read counter with `__shfl_down_sync(0xffffffffu, ...)`, which is a
//    32-lane assumption wearing a mask argument. This tree targets gfx1151
//    (RDNA3.5, wavefront 32) and CDNA parts (wavefront 64) from one source, and
//    a hard-coded 32 is wrong on the second. Both reductions are replaced with
//    forms that have NO wavefront width in them: the max goes through a
//    `[blockDim]` shared array reduced by thread 0, and the counter through one
//    LDS `atomicAdd` per thread plus one global `atomicAdd` per block.
//    **NEITHER CHANGES AN ANSWER.** `max` is associative and commutative in IEEE
//    arithmetic (the donor's own header says so, and it is why the max is the
//    ONE reduction here that may be reassociated), and the counter is an integer
//    sum. Every reduction that is order-sensitive — the dot, the denominator,
//    the value accumulation — is still walked in the CPU arm's ascending order
//    by one thread, untouched.
// 2. **`nanf("")` -> `__int_as_float(0x7fc00000)`.** The donor poisons a
//    malformed row with a quiet NaN. The bit pattern is spelled directly rather
//    than through the libm entry point so the poison value cannot depend on a
//    device-library detail; `0x7fc00000` IS the f32 quiet NaN the host's
//    `nanf("")` returns, and `MaxAbsDiff` returns +infinity on any non-finite
//    operand (#449) either way.
// 3. **The block width rounds up to 64, not 32.** A multiple of 64 is a whole
//    number of wavefronts on BOTH families. Nothing in this file reads
//    `warpSize`, so this is a launch-shape choice and not a correctness one; it
//    is stated because the donor's `BlockWidthFor` rounds to 32 and a reader
//    diffing the two files will see the constant change.
//
// ─── PRECISION: WHAT IS BYTE-IDENTICAL AND WHAT IS NOT ───────────────────────
// Inherited from the donor's header, which inherits it from the CPU arm:
//
//   `vt::Qwen4ExpQsaCompress` contains NO transcendental. Its only non-exact
//   operations are `sqrtf` and the `__frcp_rn` over it, and IEEE-754 requires
//   both correctly rounded — glibc gives the same guarantee, and this build pins
//   `-ffp-contract=off` for HIP (CMakeLists.txt:414) so no fused multiply-add
//   re-associates anything either. Every reduction runs in the host's ascending
//   f32 order and `__float2bfloat16` is round-to-nearest-even exactly as the
//   host `vt::F32ToBF16` is. So this arm is EXPECTED to be bit-identical to its
//   CPU oracle, and the gate MEASURES that (it reports the differing-byte count
//   beside the NMSE) rather than asserting it from this paragraph.
//
//   `vt::Qwen4ExpQsaGatherAttention` CANNOT reach that and exactly one function
//   is why: `expf`. The CPU arm calls `std::exp` on a float, which glibc rounds
//   correctly; ROCm's device `expf` carries its own (documented, small) bound.
//   Every other operation on both paths is IEEE-exact or spelled with an `_rn`
//   intrinsic and every order-sensitive reduction runs in the CPU arm's order,
//   so the two arms are expected to differ only where `exp` moved. Its gate is
//   the cross-device NMSE band, `5e-4`, and it reports the measured value beside
//   the verdict.
//
// The `_rn` intrinsics survive the port for the reason W1's and W3's headers
// give: the build pins `-ffp-contract=off` today, but the file must stay
// diffable against its donor line for line, and an arm whose numerics depend on
// a compile option staying set changes answer when somebody adds a flag.
//
// ─── THE FOUR ORDERS THE GATHER ARM PRESERVES, UNCHANGED FROM THE DONOR ──────
//   1. the DOT `sum_d q[d]*k[d]` is sequential ascending f32 on ONE thread,
//      never tree-reduced;
//   2. the MAX over the gathered rows is split across threads and block-reduced,
//      which is exact and order-free;
//   3. the DENOMINATOR `sum_p w` is accumulated ascending by one thread;
//   4. the VALUE accumulation `acc[d] += w * v[d]` runs ascending over `p` and is
//      parallel across `d` only, which each thread owns alone.
//
// ─── TWO REFUSALS THIS ARM IMPOSES THAT THE CPU ARM DOES NOT ─────────────────
// `head_dim > 1024` IS REFUSED BY NAME: one block serves one (token, head) and
// each thread owns one output dim. A MALFORMED SELECTION POISONS THE ROW WITH
// NaN rather than being refused by name, because a device kernel cannot throw
// and `block_ids` / `kv_lens` / `kv_block_table` are device-resident on a ROCm
// queue. Both are the donor's positions and the reasoning is in its header at
// length; neither is new here.
//
// ─── NO MFMA, NO DP4A, NO WAVEFRONT ASSUMPTION ──────────────────────────────
// Neither kernel uses a matrix intrinsic, a dot-product intrinsic or a
// cross-lane primitive. MFMA is CDNA-only and RDNA has no equivalent, so a port
// that reached for one would not build for gfx1151 at all; nothing here would
// have reached for one, because the only tensor-shaped operation in the pair is
// a `head_dim`-long dot that must stay in the host's sequential order anyway.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// <math.h>, not <cmath>: the device math functions this file uses (`expf`,
// `sqrtf`, `fmaxf`) are declared in the GLOBAL namespace by the HIP math
// headers, and both siblings spell the include the same way.
#include <math.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType, the op declarations

namespace vt::rocm {
namespace {

void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " + hipGetErrorString(err));
  }
}

hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

// The device's maximum block width. Both kernels give one thread one element of
// the head dim, so this is the head_dim ceiling the wrappers refuse above.
constexpr int64_t kMaxHeadDim = 1024;

// Local dtype tag + accessors, the donor's arrangement: a local copy rather than
// a hoist, because hoisting would edit a translation unit several other rows are
// working in — the shared-file lock AGENTS.md "Records" names.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* op, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false, std::string("rocm ") + op + ": unsupported " + what +
                          " dtype (f32/f16/bf16)");
      return DTag::kF32;
  }
}

__device__ inline float LoadAt(const void* p, DTag tag, int64_t i) {
  switch (tag) {
    case DTag::kF32: return static_cast<const float*>(p)[i];
    case DTag::kF16: return __half2float(static_cast<const __half*>(p)[i]);
    default: return __bfloat162float(static_cast<const __hip_bfloat16*>(p)[i]);
  }
}

__device__ inline void StoreAt(void* p, DTag tag, int64_t i, float v) {
  switch (tag) {
    case DTag::kF32: static_cast<float*>(p)[i] = v; break;
    case DTag::kF16: static_cast<__half*>(p)[i] = __float2half(v); break;
    default: static_cast<__hip_bfloat16*>(p)[i] = __float2bfloat16(v); break;
  }
}

// The poison value, spelled as its bit pattern — see the header's change 2.
__device__ inline float QuietNan() { return __int_as_float(0x7fc00000); }

// One bf16 round trip, the CPU arm's `MaybeBf16`. Upstream's `.to(raw_keys.
// dtype)` / `.type_as(x)` on a bf16 model path, and a bf16 elementwise op's
// store, each do exactly this.
__device__ inline float MaybeBf16(float x, bool round) {
  return round ? __bfloat162float(__float2bfloat16(x)) : x;
}

// Round the block width up to a whole WAVEFRONT on both families — 64 is a whole
// number of wavefronts on RDNA (32) and on CDNA (64). Nothing below reads
// `warpSize`; this is a launch shape, not a correctness premise.
unsigned BlockWidthFor(int64_t head_dim) {
  const int64_t w = ((head_dim + 63) / 64) * 64;
  return static_cast<unsigned>(w < 64 ? 64 : w);
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
// and a tree reduction is a different one.
__global__ void QsaCompressKernel(void* block_keys, DTag bk_tag, const void* raw_keys,
                                  DTag raw_tag, const void* k_norm_w, DTag w_tag,
                                  const float* cosp, const float* sinp, int64_t nb, int64_t D,
                                  int64_t CR, int64_t rot, float eps, bool round) {
  extern __shared__ float s_compress[];
  float* s_pooled = s_compress;      // [D]
  float* s_normed = s_compress + D;  // [D]
  __shared__ float s_rrms;
  const int64_t rot_half = rot / 2;  // NOT `half`: hip_fp16.h typedefs that
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
      // function, and at the model's real eps of 1e-6 the difference hides
      // unless the fixture separates the row scales — which is what W1 had to
      // learn on `vt::RmsNormGroup` (spec D3b) and what this op's gate does too.
      s_rrms = __frcp_rn(sqrtf(__fadd_rn(__fdiv_rn(ss, d_f), eps)));
    }
    __syncthreads();
    const float rrms = s_rrms;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
      const float w = LoadAt(k_norm_w, w_tag, d);
      s_normed[d] =
          MaybeBf16(__fmul_rn(__fmul_rn(s_pooled[d], rrms), __fadd_rn(1.0f, w)), round);
    }
    __syncthreads();

    // 3. RoPE AT THE BLOCK'S FIRST TOKEN. Upstream reads it as
    //    `group_starts = block_token_indices[:, 0]`; the Triton kernel computes
    //    the same value as `compressed_pos = (position // CR) * CR`. Taking the
    //    block's LAST position instead is a silent one-block phase error no
    //    shape check can see. The rotation is NeoX `rotate_half` over the
    //    LEADING `rotary_dim` dims.
    const int64_t pos = b * CR;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
      if (d < rot) {
        const float rot_src = d < rot_half ? -s_normed[d + rot_half] : s_normed[d - rot_half];
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

void Qwen4ExpQsaCompressKernelRocm(Queue& q, Tensor& block_keys, const Tensor& raw_keys,
                                   const Tensor& k_norm_weight, const Tensor& cos,
                                   const Tensor& sin,
                                   const Qwen4ExpQsaCompressArgs& args) {
  constexpr const char* kOp = "qwen4_exp_qsa_compress";
  const int64_t D = raw_keys.shape[1];
  const int64_t CR = args.compress_ratio;
  const int64_t nb = raw_keys.shape[0] / CR;
  if (nb == 0 || D == 0) return;  // empty-work early return, before the launch
  VT_CHECK(D <= kMaxHeadDim,
           std::string("rocm ") + kOp + ": index head_dim " + std::to_string(D) +
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
  Check(hipGetLastError(), "qwen4_exp_qsa_compress launch");
}

// ---------------------------------------------------------------------------
// vt::Qwen4ExpQsaGatherAttention — dense GQA over ONLY the gathered rows.
//
// ONE BLOCK PER (query token, query head); one thread per head dim. The two
// address modes are ONE body, as they are on the host and in the donor: the only
// thing that forks is the resolution of one key/value row address.
//
// How many rows the tile below carries between thread 0 and the block. It exists
// to amortise `__syncthreads`, NOT to change any order.
constexpr int kSelTile = 32;

// The one address that differs between the two arms, mirroring vLLM's paged read
//   block = block_table[j / block_size], offset = j % block_size,
//   K = k_cache[block, offset, g, :]
// Returns -1 when the table names a page the cache does not hold, which is the
// CPU arm's `VT_CHECK` turned into a value a device kernel can act on: the caller
// reads NOTHING and poisons the row.
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
  float* s_q = s_attn;                          // [DH] the query row
  float* s_red = s_attn + DH;                   // [blockDim] per-THREAD max partials
  float* s_wtile = s_red + blockDim.x;          // [kSelTile]
  __shared__ int64_t s_ptile[kSelTile];
  __shared__ float s_m;
  __shared__ float s_denom;
  __shared__ int s_bad;
  __shared__ int64_t s_nsel_blocks;  // how many block ids the row selected
  __shared__ int64_t s_selcount;     // |sel|, expansions + ragged tail
  __shared__ int64_t s_complete;
  // The read counter's per-block accumulator. An LDS atomic rather than the
  // donor's `__shfl_down_sync` chain, because that chain hard-codes 32 lanes;
  // an integer sum is order-free, so this is the same number by a different
  // route. See the header's change 1.
  __shared__ unsigned long long s_visited;

  const int64_t k_str[3] = {k_s0, k_s1, k_s2};
  const int64_t v_str[3] = {v_s0, v_s1, v_s2};
  const int64_t pairs = T * HQ;

  for (int64_t pair = blockIdx.x; pair < pairs; pair += gridDim.x) {
    __syncthreads();
    const int64_t t = pair / HQ;
    const int64_t h = pair - t * HQ;
    const int64_t kvh = h / groups;
    unsigned long long reads = 0;  // COUNTED AT THE READ; see the header

    if (threadIdx.x == 0) {
      s_bad = 0;
      s_visited = 0;
      const int64_t kv_len = static_cast<int64_t>(lens[t]);
      // The CPU arm's `VT_CHECK(kv_len >= 0 && kv_len <= max_kv, ...)`. A device
      // kernel cannot throw, so the row is poisoned instead.
      if (kv_len < 0 || kv_len > max_kv) s_bad = 1;
      const int64_t complete = (kv_len < 0 ? 0 : kv_len) / CR;
      // THE EXPANSION. Selected block `b` IS tokens [CR*b, CR*b + CR). It is
      // resolved as ADDRESSES — never handed in as a token buffer. Only the
      // COUNT is materialised here; the mapping `s -> p` below is arithmetic.
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
        StoreAt(out, out_tag, (t * HQ + h) * DH + d, QuietNan());
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
      const int64_t base =
          RowBase(paged, pages, page_size, num_phys_pages, k_str, HKV, DH, p, kvh);
      if (base < 0) { s_bad = 1; continue; }
      ++reads;
      float dot = 0.0f;
      for (int64_t d = 0; d < DH; ++d) {
        dot = __fadd_rn(dot, __fmul_rn(s_q[d], LoadAt(key, k_tag, base + d)));
      }
      local_m = fmaxf(local_m, __fmul_rn(dot, scale));
    }
    // The block reduction of the max, with NO wavefront width in it: every
    // thread publishes its partial and thread 0 folds them. Exact at every step.
    s_red[threadIdx.x] = local_m;
    __syncthreads();
    if (threadIdx.x == 0) {
      float m = s_red[0];
      for (unsigned i = 1; i < blockDim.x; ++i) m = fmaxf(m, s_red[i]);
      s_m = m;
    }
    __syncthreads();
    if (s_bad != 0) {
      for (int64_t d = threadIdx.x; d < DH; d += blockDim.x) {
        StoreAt(out, out_tag, (t * HQ + h) * DH + d, QuietNan());
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
          const int64_t base =
              RowBase(paged, pages, page_size, num_phys_pages, k_str, HKV, DH, p, kvh);
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
          const int64_t vbase =
              RowBase(paged, pages, page_size, num_phys_pages, v_str, HKV, DH, p, kvh);
          if (vbase < 0) continue;
          acc = __fadd_rn(acc, __fmul_rn(s_wtile[u], LoadAt(value, v_tag, vbase + d)));
        }
      }
      __syncthreads();
    }
    if (s_bad != 0) {
      for (int64_t d = threadIdx.x; d < DH; d += blockDim.x) {
        StoreAt(out, out_tag, (t * HQ + h) * DH + d, QuietNan());
      }
      continue;
    }
    const float denom = s_denom;
    if (threadIdx.x < DH) {
      StoreAt(out, out_tag, (t * HQ + h) * DH + threadIdx.x, __fdiv_rn(acc, denom));
    }

    // The counter: one LDS atomic per thread, then ONE global atomic per block.
    // Every increment above sits inside a read loop; nothing here is derived
    // from the selection.
    if (visited != nullptr) {
      atomicAdd(&s_visited, reads);
      __syncthreads();
      if (threadIdx.x == 0) atomicAdd(visited, s_visited);
    }
  }
}

void Qwen4ExpQsaGatherAttentionKernelRocm(Queue& q, Tensor& out, const Tensor& query,
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
           std::string("rocm ") + kOp + ": head_dim " + std::to_string(DH) + " exceeds the " +
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
    Check(hipMalloc(&d_visited, sizeof(unsigned long long)),
          "qwen4_exp_qsa_gather_attention keys_visited alloc");
    Check(hipMemsetAsync(d_visited, 0, sizeof(unsigned long long), AsStream(q)),
          "qwen4_exp_qsa_gather_attention keys_visited zero");
  }

  const unsigned width = BlockWidthFor(DH);
  const unsigned grid = static_cast<unsigned>(pairs < 4096 ? pairs : 4096);
  // [DH query row | blockDim max partials | kSelTile weights]. The per-THREAD
  // partial array is what replaces the donor's per-WARP one; at the released
  // head_dim of 256 that is 256 + 256 + 32 floats = 2.1 KiB, and at the 1024
  // ceiling 8.25 KiB, both far inside the 64 KiB LDS budget vLLM's own AMD
  // comment cites.
  const size_t shared = (static_cast<size_t>(DH) + width + kSelTile) * sizeof(float);
  QsaGatherAttentionKernel<<<grid, width, shared, AsStream(q)>>>(
      out.data, out_tag, query.data, q_tag, key.data, k_tag, value.data, v_tag,
      block_ids.Ptr<int32_t>(), kv_lens.Ptr<int32_t>(), T, HQ, DH, HKV, HQ / HKV,
      block_ids.shape[1], args.compress_ratio, args.scale, paged,
      paged ? args.kv_block_table->Ptr<int32_t>() : nullptr, page_size,
      paged ? key.shape[0] : 0, max_kv, key.stride[0], key.stride[1], key.stride[2],
      value.stride[0], value.stride[1], value.stride[2], d_visited);
  Check(hipGetLastError(), "qwen4_exp_qsa_gather_attention launch");

  if (args.keys_visited != nullptr) {
    unsigned long long host = 0;
    Check(hipMemcpyAsync(&host, d_visited, sizeof(unsigned long long),
                         hipMemcpyDeviceToHost, AsStream(q)),
          "qwen4_exp_qsa_gather_attention keys_visited copy-back");
    Check(hipStreamSynchronize(AsStream(q)),
          "qwen4_exp_qsa_gather_attention keys_visited sync");
    Check(hipFree(d_visited), "qwen4_exp_qsa_gather_attention keys_visited free");
    *args.keys_visited = static_cast<int64_t>(host);
  }
}

// Registers the ROCm arms during static init, the pattern every `rocm_*.hip`
// registrar in this directory uses. UNCONDITIONAL and at preprocessor depth 0:
// an op registered from a feature-gated TU disappears on a build that does not
// compile the feature, the resolver falls through, and on `gfx1151` there is
// nothing to fall through TO — the portable reference tier is not installed
// there (spec D2), so the call refuses by name instead. That refusal is the
// intended failure and a conditional registration would turn it into a silent
// absence.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQwen4ExpQsaCompress, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpQsaCompressFn>(&Qwen4ExpQsaCompressKernelRocm)));
    RegisterOp(OpId::kQwen4ExpQsaGatherAttention, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<Qwen4ExpQsaGatherAttentionFn>(
                   &Qwen4ExpQsaGatherAttentionKernelRocm)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::rocm
