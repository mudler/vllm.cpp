// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt {

constexpr int kMaxRank = 4;

// Non-owning tensor view. Strides are in ELEMENTS (torch convention).
struct Tensor {
  void* data = nullptr;
  DType dtype = DType::kF32;
  Device device;
  int rank = 0;
  int64_t shape[kMaxRank] = {0, 0, 0, 0};
  int64_t stride[kMaxRank] = {0, 0, 0, 0};

  // CIQ G7: this is a block-quant weight whose bytes were REPACKED at load into
  // the CPU i8mm SIMD/cache-friendly interleave (q8_0 -> block_q8_0x4). Storage
  // only — set by the GGUF keep-quant loader, consumed by `kMatmulBTQuant`,
  // which then dispatches the repacked gemm/gemv. Ignored on every other op and
  // device (a repacked weight only ever reaches the CPU quant GEMM). The total
  // byte count and [N,K] shape are unchanged; only the block interleave differs.
  bool repacked = false;

  // Brick 4 (DeepSeek-V4 last-mile): this is a Q8_0 weight whose bytes were
  // REPACKED at load into the CUDA coalesced-load layout — the 32 int8 `qs` of
  // every block deinterleaved into one 16-byte-aligned contiguous section
  // (`[all qs | all scales]`) so a warp lane reads them via aligned `int4`
  // (128-bit) loads instead of the 34-byte in-place block's 2-byte reads.
  // Storage only — set by the GGUF keep-quant loader, consumed by the CUDA
  // `kMatmulBTQuant` Q8_0 path. Same total byte count + [N,K] shape; only the
  // block byte order differs. Mutually exclusive with `repacked` (CPU i8mm).
  bool q8_0_aligned = false;

  // KERNEL-GEMM-CPU-TILED lever 2: this is a [N,K] elementwise (f32/f16/bf16)
  // weight whose BYTES were transposed at load into [K,N]. Storage only, and
  // the SHAPE still reads [N,K] so `vt::MatmulBT`'s contract is unchanged for
  // every caller. Set by the loader via `vt::cpu::ElemRepackWeight`, consumed
  // by `MatmulBTKernel`, which then presents the buffer as the [K,N] tensor it
  // literally is and takes the transpose-free `nk`/`nkm` micro-kernels.
  //
  // Bit-exactness: the two orientations produce BYTE-IDENTICAL results for the
  // same logical weight (both accumulate each output over K in strict
  // increasing order; measured on dgx at every conformer shape), so this is a
  // pure layout choice and never a numerical one. Ignored on every other op and
  // on non-CPU devices. Mutually exclusive with the block-quant flags above,
  // which apply to block dtypes this one never does.
  bool elem_kn_repacked = false;

  // Weight-only value contract for ordinary Matmul/MatmulBT B and Embedding
  // table operands. Storage remains F16; ROCm converts to BF16 or F32 values
  // before arithmetic. Unset preserves exact storage values.
  std::optional<DType> weight_value_dtype;

  static Tensor Contiguous(void* data, DType dtype, Device device,
                           std::initializer_list<int64_t> shape);

  int64_t Numel() const;
  bool IsContiguous() const;
  size_t Bytes() const;
  Tensor View(std::initializer_list<int64_t> new_shape) const;
  Tensor Slice(int dim, int64_t start, int64_t stop) const;
  int64_t Offset(std::initializer_list<int64_t> idx) const;

  template <typename T>
  T* Ptr() const {
    return static_cast<T*>(data);
  }
};

}  // namespace vt
