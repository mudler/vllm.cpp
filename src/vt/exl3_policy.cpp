// EXL3 kernel-shape policy — MODEL-DSV4-EXL3 W2b, host tier.
//
// PORTED 1:1 FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//   exllamav3_ext/quant/exl3_kernel_map.cuh:53-60  the shape + geometry macros
//   exllamav3_ext/quant/exl3_kernel_map.cu:23-91   select_gemm_shape,
//                                                  exl3_gemm_shape_compat
//   exllamav3_ext/quant/exl3_kernel_map.cu:153-160 the empty-block clamp
//   exllamav3_ext/quant/exl3_devctx.cu:32-46       the compute-capability bucket
//
// WHY THIS IS A .cpp AND NOT PART OF THE .cu. The table decides which kernel a
// shape gets, and it is pure integer arithmetic over (cc, m, k, n, bits). Left
// inside the CUDA translation unit it would be checkable only on a machine with
// a GPU, and a table nobody can check is a table that drifts. Here it is gated
// by `tests/vt/test_exl3_gemm.cpp` on any host, and the CUDA launcher calls
// exactly these functions rather than a second copy of them.
#include <string>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt {
namespace {

// exl3_kernel_map.cuh:53-56, in the macro's own order:
//   TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES
// plus EXL3_GEMM_BLOCKDIM (:60), whose slot 0 is upstream's unused `nullptr`
// entry. Index 0 exists so `shape_idx` stays 1-based exactly as upstream's
// instance arrays are.
constexpr Exl3GemmShape kShapes[5] = {
    {0, 0, 0, 0, 0, 0},
    {16, 16, 128, 6, 5, 256},
    {16, 32, 128, 4, 3, 512},
    {16, 32, 256, 4, 3, 512},
    {16, 16, 512, 4, 3, 256},
};

}  // namespace

Exl3Cc Exl3CcFromSm(int sm_major, int sm_minor) {
  // exl3_devctx.cu:39-43, verbatim, including the `major >= 8 && minor >= 9`
  // Ada test that reads the MINOR only after the major has qualified.
  if (sm_major >= 10) return Exl3Cc::kBlackwell;
  if (sm_major >= 9) return Exl3Cc::kHopper;
  if (sm_major >= 8 && sm_minor >= 9) return Exl3Cc::kAda;
  if (sm_major >= 8) return Exl3Cc::kAmpere;
  return Exl3Cc::kOld;
}

int Exl3GemmNumShapes() { return 4; }  // EXL3_GEMM_NUM_SHAPES (exl3_kernel_map.cuh:62)

Exl3GemmShape Exl3GemmShapeParams(int shape_idx) {
  VT_CHECK(shape_idx >= 0 && shape_idx <= Exl3GemmNumShapes(),
           "exl3_gemm: shape index out of range (0.." +
               std::to_string(Exl3GemmNumShapes()) +
               "); got " + std::to_string(shape_idx));
  return kShapes[shape_idx];
}

int Exl3SelectGemmShape(Exl3Cc cc, int size_m, int size_k, int size_n, int bits,
                        bool multi) {
  // exl3_kernel_map.cu:23-75. `size_m` is upstream's parameter and upstream
  // reads it nowhere in the body; it is kept so the port reads against its
  // anchor line for line.
  (void)size_m;
  const bool mod_256 = (size_n % 256 == 0);
  const bool mod_512 = (size_n % 512 == 0);
  const int K = bits;

  switch (cc) {
    case Exl3Cc::kOld:
    case Exl3Cc::kAmpere:
      if (mod_256 && K <= 4) {
        if (size_n <= 2048 || size_k <= 2048) return 2;
        return 3;
      }
      if (mod_256 && size_n < 4096) return size_k > 8192 ? 3 : 2;
      if (mod_512 && (size_n * size_k) > (4096 * 4096) && K <= 6) return 4;
      if (mod_256) return 3;
      return 2;

    case Exl3Cc::kAda:
      if (mod_256 && K <= 3) {
        if (size_k <= 2048 && !multi) return 2;
        if (size_n < 4096 && size_k <= 12288) return 2;
        return 3;
      }
      if (size_n <= 16384) return 2;
      if (mod_512 && size_n >= 32768) return 4;
      if (mod_256) return 3;
      return 2;

    case Exl3Cc::kHopper:
    case Exl3Cc::kBlackwell:
      if ((K == 4 || K == 2) && !multi) {
        if (size_k <= 2048) return 1;
      }
      if (K >= 7) {
        if (mod_256 && size_n <= 8192) return size_k > 32768 ? 3 : 2;
        if (mod_512 && size_n > 32768) return 4;
        return 2;
      }
      if (mod_256 && size_n <= 4096) return size_k > 8192 && K >= 3 ? 3 : 2;
      if (mod_512 && size_n > 16384) return 4;
      if (mod_256) return 3;
      return 2;
  }
  return 0;
}

bool Exl3GemmShapeCompat(int shape_idx, int size_k, int size_n) {
  // exl3_kernel_map.cu:86-91. Upstream also takes `size_m` and `K` and reads
  // neither; both are dropped rather than carried as parameters `-Werror` would
  // then need told about. Index 0 has no geometry and is compatible with
  // nothing, which is upstream's `nullptr` slot expressed as a predicate.
  const Exl3GemmShape s = Exl3GemmShapeParams(shape_idx);
  if (s.tile_k == 0 || s.tile_n == 0) return false;
  return (size_k % s.tile_k == 0) && (size_n % s.tile_n == 0);
}

int Exl3GemmNumSms(int shape_idx, int size_k, int size_n, int device_sms) {
  // exl3_kernel_map.cu:153-160: *num_sms = MAX(MIN(max_slices, *num_sms), 1),
  // with max_slices = size_k / tilesize_k * size_n / tilesize_n. The clamp is
  // what keeps a small problem from launching blocks with no tile to take, and
  // the trailing MAX(.., 1) is what keeps a launch from asking for zero.
  const Exl3GemmShape s = Exl3GemmShapeParams(shape_idx);
  VT_CHECK(s.tile_k > 0 && s.tile_n > 0,
           "exl3_gemm: shape " + std::to_string(shape_idx) + " has no geometry");
  const int max_slices = (size_k / s.tile_k) * (size_n / s.tile_n);
  const int lo = max_slices < device_sms ? max_slices : device_sms;
  return lo > 1 ? lo : 1;
}

}  // namespace vt
