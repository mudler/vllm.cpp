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
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev/DBuf/MakeTensor
#include "vllm/model_executor/models/qwen3_5_weights.h"    // Fp8BlockWeight
#include "vt/backend.h"
#include "vt/device.h"  // DeviceTypeName
#include "vt/dtype.h"   // VT_CHECK
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

}  // namespace dense_fp8_block
}  // namespace vllm
