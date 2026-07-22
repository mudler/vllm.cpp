// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#define VT_CHECK(cond, msg)                                                       \
  do {                                                                            \
    if (!(cond)) {                                                                \
      throw std::runtime_error(std::string("vt: ") + (msg) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                   \
    }                                                                             \
  } while (0)

namespace vt {

// Storage dtypes. The first six are ELEMENTWISE (one scalar per element, a
// well-defined SizeOf); the trailing entries are BLOCK-QUANTIZED ggml
// encodings, where a fixed group of `BlockElems` scalars shares one packed
// `BlockBytes` record and there is NO per-element size.
//
// Block dtypes live in the enum (rather than a parallel `QuantTensor`) to
// mirror ggml's "the type carries the layout" model: a `Tensor` stays
// self-describing and quant dispatch keys on `b.dtype` instead of forking
// every op signature (spec `.agents/specs/gguf-compute-in-quant-gemm.md`
// § Risks/decisions). The consequence is enforced here: block dtypes are
// STORAGE-ONLY — `SizeOf` on one is a `VT_CHECK` failure, so any elementwise
// kernel that reaches one fails loudly instead of reading garbage.
//
// Ids/geometry mirror llama.cpp @ 237ad9b96:
//   ggml/include/ggml.h:390-432 (enum ggml_type)
//   ggml/src/ggml-common.h:242-245 (block_q8_0), :305-310 (block_q3_K),
//     :317-327 (block_q4_K), :334-345 (block_q5_K), :352-357 (block_q6_K),
//     :361-365 (block_q8_K)
// kQ8_K is ACTIVATION-ONLY: it is the `vec_dot_type` of the K-quants and never
// appears as a weight/storage type in a GGUF file.
enum class DType : uint8_t {
  kF32,
  kF16,
  kBF16,
  kI8,
  kI32,
  kI64,
  // --- block-quantized (storage-only) ---
  kQ4_0,
  kQ8_0,
  kQ3_K,
  kQ4_K,
  kQ5_K,
  kQ6_K,
  kQ8_K,
};

// Bytes per ELEMENT. Throws for block-quantized dtypes (they have no
// per-element size) — see IsBlockQuant/BlockBytes/RowSizeBytes.
size_t SizeOf(DType dtype);
const char* Name(DType dtype);

// True for the ggml block-quantized encodings above.
bool IsBlockQuant(DType dtype);

// Block geometry for a block-quantized dtype (throws for elementwise dtypes).
// `BlockElems` = ggml's blck_size, `BlockBytes` = ggml's type_size.
int64_t BlockElems(DType dtype);
int64_t BlockBytes(DType dtype);

// ggml_row_size (ggml/src/ggml.c): bytes occupied by `k` contiguous elements.
// `k` must be a whole number of blocks — rows are whole blocks, which is also
// the keep-quant eligibility rule for a GEMM weight (K % BlockElems == 0).
// Defined for elementwise dtypes too (k * SizeOf) so callers stay uniform.
size_t RowSizeBytes(DType dtype, int64_t k);

// The ggml type id (ggml.h:390-432) a block dtype corresponds to, so callers
// can cross-check against the GGUF reader's independent `GgmlTraits` table.
uint32_t GgmlTypeId(DType dtype);

// Inverse of GgmlTypeId for the block dtypes we execute; returns false when
// the id is not one of them (F32/F16/BF16 and every unported encoding).
bool BlockDTypeFromGgmlTypeId(uint32_t ggml_type, DType* out);

float F16ToF32(uint16_t h);
uint16_t F32ToF16(float f);
float BF16ToF32(uint16_t b);
uint16_t F32ToBF16(float f);

}  // namespace vt
