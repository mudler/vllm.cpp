// Shared BLOCK-WISE (fine-grained) **FP8 W8A8** dense GEMM glue — the block-wise
// sibling of dense_fp8_gemm.h, and the half of MODEL-FP8-BLOCK-LINEAR (#1189
// milestone M4, spec .agents/specs/model-fp8-block-linear.md) that computes.
//
// UPSTREAM CHAIN (ported FROM, ground-every-impl rule), pinned vLLM
// `5559679229bc961848b121ccdeaa8fa5d79bec98`:
//   * the dispatch, whole:  vllm/model_executor/layers/quantization/fp8.py:297-298
//       `self.block_quant = self.weight_block_size is not None`
//   * the kernel selection: fp8.py:387-393 -> kernels/linear/__init__.py:580-600
//   * THE APPLY, WHOLE:     kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:97-135
//       `q_input, input_scale = self.quant_fp8(input_2d)` then
//       `self.apply_block_scaled_mm(A=q_input, B=weight, As=input_scale,
//                                   Bs=weight_scale)`
//   * the activation quant:  BlockScaledMMLinearKernel.py:53-58 — DYNAMIC, per
//       token, group (1, 128); a static scheme is refused at :62-70
//   * the CUTLASS arm:      kernels/linear/scaled_mm/cutlass.py:312-326
//       `ops.cutlass_scaled_mm(A, B.T, out_dtype, scale_a=As, scale_b=Bs.T)`
//   * the OUTPUT DTYPE:     BlockScaledMMLinearKernel.py:104 reads
//       `self.config.out_dtype`, which is `Fp8LinearMethod.out_dtype` =
//       `torch.get_default_dtype()` (fp8.py:284, :391-392) — the MODEL dtype,
//       bf16 for `Qwen/Qwen3.8-27B-FP8`. Nothing here is f32 by default, and
//       every wired call site's dtype is recorded in the row's spec, because a
//       token gate cannot see a dtype that is too wide (.agents/porting.md).
//   * no bias:              csrc/.../c3x/scaled_mm_helper.hpp:54 refuses one on
//       this path, so this mirrors a refusal rather than deferring a feature.
//
// Both halves already exist as `vt` ops and are gated on their own:
// `vt::QuantFp8Group` (#1189 M1, `ad5f175e7`,
// .agents/specs/vt-quant-fp8-group.md) and `vt::MatmulFp8BlockScaled` (#1189
// M2, `770e49486`, .agents/specs/vt-matmul-fp8-block-ref.md). This header is
// the composition and nothing else: no new arithmetic lands here.
//
// TEMPLATED ON Dev/DBuf ON PURPOSE, for the reason dense_fp8_gemm.h records at
// length: `src/vllm/model_executor/models/qwen3_5.cpp` carries its OWN
// anonymous-namespace `Dev`/`DBuf` (qwen3_5.cpp:669), distinct TYPES from
// `dense_attn::Dev`/`DBuf` even though the layouts match, and unifying the two
// glue families is a separate refactor dense_nvfp4_gemm.h already records as
// deferred. A non-template header would have to be re-typed into qwen3_5.cpp,
// which is exactly the hand-rolled parallel path AGENTS.md §"Shared seams"
// forbids. One definition, two instantiations, no copy.
//
// DEVICE REACH — CPU ONLY, and that is inherited from the op table rather than
// chosen here. `vt::MatmulFp8BlockScaled` is a CPU correctness reference that
// makes no speed claim; the mainloop-scaled CUTLASS kernel for `sm_121a` is
// #1189 milestone M5. `RefuseUnrunnableQwen3_5DenseFp8Block` asks the same
// question at `ModelRegistry::Prepare` so a CUDA user is told BEFORE a graph is
// captured rather than at the first GEMM.
#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev/DBuf/MakeTensor
#include "vllm/model_executor/models/qwen3_5_weights.h"    // Fp8BlockWeight
#include "vt/backend.h"
#include "vt/device.h"  // DeviceTypeName
#include "vt/dtype.h"   // VT_CHECK
#include "vt/merged_gemm.h"  // MergedGemmGroup, kFp8Block*
#include "vt/ops.h"

namespace vllm {
namespace dense_fp8_block {

using vt::Backend;
using vt::DType;
using vt::Tensor;

// How many block-scaled GEMMs this process has dispatched.
//
// An instrument, and a deliberate one. #1189's gate design records the
// measurement that forced it: a x1.02 and a x1.10 scale perturbation on the
// per-tensor fp8 tower were demonstrably reached and still produced 16/16
// identical tokens (tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp:43-45).
// A token gate therefore cannot answer "did anything reach this arm", and a
// green forward over a projection that silently took the bf16 fallback looks
// exactly like a green forward that took this one. The counter can answer it,
// and tests/vllm/model_executor/models/test_fp8_block_linear.cpp G3 asserts the
// exact per-forward count through `ModelRegistry::Forward`.
//
// One relaxed increment per GEMM — roughly ten per layer per step, against a
// GEMM each — so it is not a hot-path cost. The function-local static in an
// inline function is one object across every translation unit ([basic.def.odr]).
inline std::atomic<uint64_t>& BlockGemmCounter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}
inline uint64_t BlockGemmCount() {
  return BlockGemmCounter().load(std::memory_order_relaxed);
}

// Is there a block-wise FP8 arm on this device?
//
// BOTH ops, not one. `vt::QuantFp8Group` has a CUDA arm and
// `vt::MatmulFp8BlockScaled` does not (M5 owns it), so asking about the quant
// alone would answer yes on CUDA and then fail one frame deeper with a message
// about the wrong op.
inline bool BlockFp8Runnable(vt::DeviceType device) {
  return vt::OpRegistered(vt::OpId::kQuantFp8Group, device) &&
         vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, device);
}

// Device-resident view over an Fp8BlockWeight's raw fp8 [N,K] bytes, uploaded
// ONCE (lazily) and reused across every forward step — the same seam
// `ResidentFp8` and `ResidentNvfp4` use. The shared_ptr in the (const) weight
// owns the device buffer for the model lifetime.
//
// The bytes are the checkpoint's, verbatim: one fp8-e4m3fn byte per element, in
// the original torch [N=out_features, K=in_features] orientation. Nothing here
// dequantizes, transposes or re-lays-out, and the byte count is asserted by
// test_fp8_block_linear.cpp G4 because a silent dequant to bf16 is numerically
// BETTER than the quantized path and therefore invisible to every value
// comparison in the tree.
template <class DevT>
inline Tensor ResidentFp8BlockPacked(DevT d, const Fp8BlockWeight& w) {
  if (!w.d_packed) {
    const size_t pb = w.packed.bytes.size();
    void* p = d.b.Alloc(pb);
    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);
    Backend* bk = &d.b;
    w.d_packed = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  return dense_attn::MakeTensor(w.d_packed.get(), DType::kI8, d.q.device,
                                {w.n, w.k});
}

// Device-resident view over the f32 [cdiv(N,block_n), cdiv(K,block_k)] scale
// grid. `cdiv` on BOTH axes, as upstream allocates it
// (utils/fp8_utils.py:1283-1296) and as the GEMM's contract demands
// (fp8_utils.py:935-936), so a short final block is legal and works.
template <class DevT>
inline Tensor ResidentFp8BlockScale(DevT d, const Fp8BlockWeight& w) {
  VT_CHECK(w.scale.dtype == DType::kF32 && w.scale.rank == 2,
           "block-wise FP8: the weight scale must be a 2-D f32 grid");
  if (!w.d_scale) {
    const size_t sb = w.scale.bytes.size();
    void* p = d.b.Alloc(sb);
    d.b.Copy(d.q, p, w.scale.bytes.data(), sb);
    Backend* bk = &d.b;
    w.d_scale = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  return dense_attn::MakeTensor(w.d_scale.get(), DType::kF32, d.q.device,
                                {w.scale.shape[0], w.scale.shape[1]});
}

// y[M,N] = x[M,K] @ dequant(w).T through the block-wise W8A8 fp8 path:
// upstream's apply, in upstream's order.
//
//   a_fp8   [M, K]                       i8   vt::QuantFp8Group(x, block_k)
//   a_scale [M, K / block_k]             f32  emitted by the same call
//   w_fp8   [N, K]                       i8   resident, uploaded once
//   w_scale [cdiv(N,bn), cdiv(K,bk)]     f32  resident, uploaded once
//   out     [M, N]                       out_dtype
//
// The scales apply in the GEMM MAINLOOP, once per K-block, into an f32
// accumulator — not in an epilogue. That is `vt::MatmulFp8BlockScaled`'s
// contract and it is a correctness constraint rather than an optimisation: an
// epilogue has exactly one degree of freedom per output element and this scheme
// has `cdiv(K, block_k)` of them. See include/vt/ops.h and
// .agents/specs/vt-matmul-fp8-block-ref.md.
//
// `out_dtype` is the CALLER's, and every wired call site passes what upstream's
// `out_dtype` is at that site — the model dtype, bf16 — rather than the f32 the
// per-tensor fp8 arm beside two of them happens to use. The f32 accumulation
// and the cast at the store live inside the GEMM, as they do upstream.
template <class DBufT, class DevT>
inline DBufT MatmulFp8BlockScaledD(DevT d, const Tensor& x,
                                   const Fp8BlockWeight& w, DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  VT_CHECK(!w.Empty() && w.block_n > 0 && w.block_k > 0,
           "MatmulFp8BlockScaledD: the block-wise FP8 weight is empty or "
           "carries no block geometry");
  VT_CHECK(K == w.k,
           "MatmulFp8BlockScaledD: activation K " + std::to_string(K) +
               " does not match the block-wise FP8 weight's in_features " +
               std::to_string(w.k));
  // Refused HERE rather than one frame deeper inside vt::QuantFp8Group, so the
  // message carries the shape a reader can act on. Upstream asserts the same
  // divisibility on the ACTIVATION (utils/fp8_utils.py:596-599) while tiling
  // the WEIGHT with cdiv (fp8_utils.py:930-936); that asymmetry is upstream's,
  // not ours.
  VT_CHECK(K % w.block_k == 0,
           "MatmulFp8BlockScaledD: the activation's in_features " +
               std::to_string(K) +
               " is not a multiple of the quantization block's k " +
               std::to_string(w.block_k) +
               ", which the dynamic per-token per-group activation quant "
               "requires (vllm utils/fp8_utils.py:596-599)");
  VT_CHECK(BlockFp8Runnable(d.q.device.type),
           std::string("MatmulFp8BlockScaledD: no block-wise (fine-grained) "
                       "FP8 kernel on device '") +
               vt::DeviceTypeName(d.q.device.type) +
               "'. The CPU reference arm is what exists today; the "
               "mainloop-scaled CUTLASS kernel is milestone M5 of "
               "https://github.com/mudler/vllm.cpp/issues/1189");

  DBufT a_fp8(d, DType::kI8, {M, K});
  DBufT a_scale(d, DType::kF32, {M, K / w.block_k});
  vt::QuantFp8Group(d.q, a_fp8.t(), a_scale.t(), x,
                    static_cast<int>(w.block_k));
  const Tensor w_fp8 = ResidentFp8BlockPacked(d, w);
  const Tensor w_scale = ResidentFp8BlockScale(d, w);
  DBufT dout(d, out_dtype, {M, N});
  vt::MatmulFp8BlockScaled(d.q, dout.t(), a_fp8.t(), a_scale.t(), w_fp8,
                           w_scale, static_cast<int>(w.block_n),
                           static_cast<int>(w.block_k));
  BlockGemmCounter().fetch_add(1, std::memory_order_relaxed);
  return dout;
}


// ---------------------------------------------------------------------------
// MERGED groups — MODEL-FP8-BLOCK-MERGED (#1189 milestone M6, spec
// `.agents/specs/model-fp8-block-merged.md`)
// ---------------------------------------------------------------------------
//
// vLLM runs `gate_proj`/`up_proj` as ONE MergedColumnParallelLinear and
// `q`/`k`/`v` as ONE QKVParallelLinear -- the model declares both in its
// `packed_modules_mapping` (`models/qwen3_5.py:288-298` @ `5559679229`), and
// the two loaders are `layers/linear.py:660` and `:1021`. The block arm can do the same by simply concatenating the
// shards along N, which the per-tensor fp8 arm cannot: a per-tensor shard folds
// its own scalar alpha and two scalars do not concatenate, so that arm runs at
// alpha=1 and applies a per-output-column alpha vector afterwards. A block
// scale is indexed by `n / block_n`, so merged row `offset_j + l` of shard `j`
// lands in block `offset_j / block_n + l / block_n` exactly when `offset_j` is
// a multiple of `block_n`, and the merged grid is then the row-concatenation of
// the shard grids.

// One shard of a merged group. The NAME travels with the pointer because every
// refusal below has to say WHICH projection is wrong, and a caller that passes
// three anonymous pointers cannot be told that.
struct Fp8BlockShard {
  const Fp8BlockWeight* w;
  const char* name;
};

inline int64_t Fp8BlockCDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// May this group be N-concatenated at all?
//
// THE GUARD THIS FILE EXISTS FOR. Upstream's `validate_fp8_block_shape`
// (`utils/fp8_utils.py:1229-1244`) requires every partition of a merged
// block-quant linear EXCEPT THE LAST to be a multiple of `block_n`, and
// `adjust_block_scale_shard` (`layers/linear.py:86-95`) ceil-divides the shard
// offset and the shard size INDEPENDENTLY, which is consistent only under that
// rule. Upstream also carries an escape hatch that disables the validation when
// any partition is ragged (`linear.py:532-557`, read at `fp8_utils.py:1207`),
// but the hatch only MOVES the failure: run upstream's own function over a
// QKVParallelLinear with q=256, k=64, v=64 at block_n=128 and the `v` shard
// slices rows [3,4) of a scale parameter that has 3 rows.
//
// So there is no correct merged GEMM for a ragged NON-final shard, and this
// refuses one by name rather than splitting silently or slicing silently. A
// ragged FINAL shard is fine, and upstream allows exactly that case.
inline void CheckFp8BlockMergeable(const vt::MergedGemmGroup& desc,
                                   const char* group,
                                   const Fp8BlockShard* shards, int count) {
  VT_CHECK(shards != nullptr && count >= 2,
           std::string("block-wise FP8 merged '") + group +
               "': a merged group needs at least two shards");
  VT_CHECK(count == desc.arity,
           std::string("block-wise FP8 merged '") + group + "': " +
               std::to_string(count) + " shards against descriptor '" +
               desc.name + "' arity " + std::to_string(desc.arity));
  const Fp8BlockWeight& first = *shards[0].w;
  for (int i = 0; i < count; ++i) {
    const Fp8BlockShard& sh = shards[i];
    const Fp8BlockWeight& w = *sh.w;
    const std::string where =
        std::string("block-wise FP8 merged '") + group + "': shard '" +
        sh.name + "' ";
    VT_CHECK(!w.Empty() && w.block_n > 0 && w.block_k > 0,
             where + "is empty or carries no block geometry");
    VT_CHECK(w.k == first.k,
             where + "has in_features " + std::to_string(w.k) +
                 " where shard '" + shards[0].name + "' has " +
                 std::to_string(first.k) +
                 "; a merged linear has ONE input");
    VT_CHECK(w.block_n == first.block_n && w.block_k == first.block_k,
             where + "has block geometry " + std::to_string(w.block_n) + "x" +
                 std::to_string(w.block_k) + " where shard '" +
                 shards[0].name + "' has " + std::to_string(first.block_n) +
                 "x" + std::to_string(first.block_k));
    VT_CHECK(w.scale.rank == 2 && w.scale.dtype == DType::kF32,
             where + "does not carry a 2-D f32 scale grid");
    VT_CHECK(w.scale.shape[0] == Fp8BlockCDiv(w.n, w.block_n) &&
                 w.scale.shape[1] == Fp8BlockCDiv(w.k, w.block_k),
             where + "carries a scale grid that is not cdiv(N,block_n) by "
                     "cdiv(K,block_k)");
    // An elementwise gated epilogue reads the merged output as two EQUAL
    // halves (vt::SiluAndMul splits [M,2D] at D), so an unequal pair would
    // silently mis-split. Upstream's own gate_up is `output_sizes = [I, I]`.
    VT_CHECK(desc.epilogue != vt::MergedEpilogue::kSiluMulClamp ||
                 w.n == first.n,
             where + "has out_features " + std::to_string(w.n) +
                 " where shard '" + shards[0].name + "' has " +
                 std::to_string(first.n) +
                 "; a gated merged linear splits its output into two EQUAL "
                 "halves");
    // Every shard but the LAST has to start the next one on a block boundary.
    if (i + 1 < count) {
      VT_CHECK(
          w.n % first.block_n == 0,
          where + "has out_features " + std::to_string(w.n) +
              ", which is not a multiple of the quantization block's n " +
              std::to_string(first.block_n) +
              ". Only the LAST shard of a merged block-quant linear may be "
              "ragged (vllm utils/fp8_utils.py:1229-1244); an earlier ragged "
              "shard makes the concatenated scale grid disagree with the "
              "merged operand's block rows, so the merge is refused rather "
              "than mis-sliced");
    }
  }
}

// The N-concatenated operand, as tensor views over one device buffer pair.
struct Fp8BlockMergedView {
  Tensor packed;  // i8  [sum N_i, K]
  Tensor scale;   // f32 [sum cdiv(N_i, block_n), cdiv(K, block_k)]
  int64_t n_total = 0;
  int64_t k = 0;
  int64_t block_n = 0;
  int64_t block_k = 0;
};

// Build (lazily, ONCE) the merged operand and keep it resident. Mirrors
// `ResidentFp8Qkv` and `ResidentNvfp4Qkv`: the shards' bytes are copied
// back-to-back and the per-shard residents are then never built, so the merged
// arm costs no duplicate device bytes.
template <class DevT>
inline Fp8BlockMergedView ResidentFp8BlockMerged(
    DevT d, const vt::MergedGemmGroup& desc, const char* group,
    const Fp8BlockShard* shards, int count, const Fp8BlockMergedResident& r) {
  CheckFp8BlockMergeable(desc, group, shards, count);
  const Fp8BlockWeight& first = *shards[0].w;
  int64_t n_total = 0;
  int64_t scale_rows = 0;
  size_t packed_bytes = 0;
  size_t scale_bytes = 0;
  for (int i = 0; i < count; ++i) {
    const Fp8BlockWeight& w = *shards[i].w;
    VT_CHECK(w.packed.bytes.size() == static_cast<size_t>(w.n * w.k),
             std::string("block-wise FP8 merged '") + group + "': shard '" +
                 shards[i].name +
                 "' does not carry exactly one fp8 byte per element");
    n_total += w.n;
    scale_rows += w.scale.shape[0];
    packed_bytes += w.packed.bytes.size();
    scale_bytes += w.scale.bytes.size();
  }
  // The property the whole merge rests on, asserted rather than assumed: the
  // concatenated grid is the grid the merged operand's own N would allocate.
  VT_CHECK(scale_rows == Fp8BlockCDiv(n_total, first.block_n),
           std::string("block-wise FP8 merged '") + group +
               "': the concatenated scale grid has " +
               std::to_string(scale_rows) + " rows where the merged N " +
               std::to_string(n_total) + " needs " +
               std::to_string(Fp8BlockCDiv(n_total, first.block_n)));

  if (!r.d_packed || !r.d_scale) {
    VT_CHECK(!r.d_packed && !r.d_scale,
             std::string("block-wise FP8 merged '") + group +
                 "': partial resident state");
    Backend* bk = &d.b;
    void* pp = d.b.Alloc(packed_bytes);
    std::shared_ptr<void> packed_owner(pp,
                                       [bk](void* q) { bk->Free(q); });
    auto* pdst = static_cast<uint8_t*>(pp);
    for (int i = 0; i < count; ++i) {
      const auto& src = shards[i].w->packed.bytes;
      d.b.Copy(d.q, pdst, src.data(), src.size());
      pdst += src.size();
    }
    void* sp = d.b.Alloc(scale_bytes);
    std::shared_ptr<void> scale_owner(sp, [bk](void* q) { bk->Free(q); });
    auto* sdst = static_cast<uint8_t*>(sp);
    for (int i = 0; i < count; ++i) {
      const auto& src = shards[i].w->scale.bytes;
      d.b.Copy(d.q, sdst, src.data(), src.size());
      sdst += src.size();
    }
    r.d_packed = std::move(packed_owner);
    r.d_scale = std::move(scale_owner);
  }

  Fp8BlockMergedView v;
  v.packed = dense_attn::MakeTensor(r.d_packed.get(), DType::kI8, d.q.device,
                                    {n_total, first.k});
  v.scale = dense_attn::MakeTensor(r.d_scale.get(), DType::kF32, d.q.device,
                                   {scale_rows, first.scale.shape[1]});
  v.n_total = n_total;
  v.k = first.k;
  v.block_n = first.block_n;
  v.block_k = first.block_k;
  return v;
}

// The preconditions every merged group shares, refused HERE rather than one
// frame deeper inside `vt::QuantFp8Group` so the message names the projection
// group a reader can act on. Upstream asserts the same divisibility on the
// ACTIVATION (`utils/fp8_utils.py:596-599`) while tiling the WEIGHT with cdiv
// (`fp8_utils.py:930-936`); that asymmetry is upstream's, not ours.
template <class DevT>
inline void CheckFp8BlockMergedActivation(DevT d, const char* group,
                                          const Tensor& x,
                                          const Fp8BlockMergedView& v) {
  const std::string where = std::string("block-wise FP8 merged '") + group + "': ";
  const int64_t K = x.shape[1];
  VT_CHECK(K == v.k, where + "activation K " + std::to_string(K) +
                         " does not match the merged weight's in_features " +
                         std::to_string(v.k));
  VT_CHECK(K % v.block_k == 0,
           where + "the activation's in_features " + std::to_string(K) +
               " is not a multiple of the quantization block's k " +
               std::to_string(v.block_k) +
               ", which the dynamic per-token per-group activation quant "
               "requires (vllm utils/fp8_utils.py:596-599)");
  VT_CHECK(BlockFp8Runnable(d.q.device.type),
           where + "no block-wise (fine-grained) FP8 kernel on device '" +
               vt::DeviceTypeName(d.q.device.type) +
               "'. The CPU reference arm is what exists today; the "
               "mainloop-scaled CUTLASS kernel is milestone M5 of "
               "https://github.com/mudler/vllm.cpp/issues/1189");
}

// y[M, sum N_i] = x[M,K] @ dequant(concat(w_i)).T — ONE block-scaled GEMM over
// the merged operand, the caller slicing the output into its logical shards.
// This is the QKV shape (`vt::kFp8BlockQkv`, epilogue kNone).
template <class DBufT, class DevT>
inline DBufT MatmulFp8BlockMergedD(DevT d, const Tensor& x,
                                   const Fp8BlockMergedView& v, DType out_dtype,
                                   const vt::MergedGemmGroup& desc) {
  const int64_t M = x.shape[0], K = x.shape[1];
  CheckFp8BlockMergedActivation(d, desc.name, x, v);
  DBufT a_fp8(d, DType::kI8, {M, K});
  DBufT a_scale(d, DType::kF32, {M, K / v.block_k});
  vt::QuantFp8Group(d.q, a_fp8.t(), a_scale.t(), x,
                    static_cast<int>(v.block_k));
  DBufT out(d, out_dtype, {M, v.n_total});
  vt::MergedGemmFp8Block(d.q, desc, out.t(), /*out=*/nullptr, a_fp8.t(),
                         a_scale.t(), v.packed, v.scale,
                         static_cast<int>(v.block_n),
                         static_cast<int>(v.block_k),
                         /*epilogue_scalar=*/0.0F);
  BlockGemmCounter().fetch_add(1, std::memory_order_relaxed);
  return out;
}

// silu(gate) * up as [M, sum N_i / 2] — ONE merged gate_up GEMM plus the SwiGLU
// tail (`vt::kFp8BlockGateUpSwiGLU`). This is what
// `layers::Fp8BlockMlpGateUpMethod` and the dense MLP both run.
template <class DBufT, class DevT>
inline DBufT Fp8BlockGateUpSwiGLUD(DevT d, const Tensor& x,
                                   const Fp8BlockMergedView& v,
                                   DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1];
  VT_CHECK(v.n_total % 2 == 0,
           "block-wise FP8 merged 'gate_up_proj': the merged N must be even");
  CheckFp8BlockMergedActivation(d, "gate_up_proj", x, v);
  DBufT a_fp8(d, DType::kI8, {M, K});
  DBufT a_scale(d, DType::kF32, {M, K / v.block_k});
  vt::QuantFp8Group(d.q, a_fp8.t(), a_scale.t(), x,
                    static_cast<int>(v.block_k));
  DBufT gate_up(d, out_dtype, {M, v.n_total});
  DBufT act(d, out_dtype, {M, v.n_total / 2});
  vt::MergedGemmFp8Block(d.q, vt::kFp8BlockGateUpSwiGLU, gate_up.t(), &act.t(),
                         a_fp8.t(), a_scale.t(), v.packed, v.scale,
                         static_cast<int>(v.block_n),
                         static_cast<int>(v.block_k),
                         std::numeric_limits<float>::infinity());
  BlockGemmCounter().fetch_add(1, std::memory_order_relaxed);
  return act;
}

}  // namespace dense_fp8_block
}  // namespace vllm
