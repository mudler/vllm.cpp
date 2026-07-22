// vllm.cpp ORIGINAL — GGUF keep-quantized residency policy. See
// gguf_keep_quant.h for the upstream (llama.cpp @ 237ad9b96) anchors and the
// totality contract.
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

#include <cstdlib>
#include <cstring>

#include "vt/quant.h"

namespace vllm {
namespace {

// -1 means "this role never keeps its blocks"; otherwise the K (in-features)
// dimension whose block alignment decides eligibility.
//
// NOTE the deliberate absence of a `default:` label: a new GgufTensorRole that
// forgets to state its residency is a -Werror=switch BUILD failure, which is
// the compile-time half of the totality proof (the unit tests are the other).
int64_t KeepQuantKDim(GgufTensorRole role, const std::vector<int64_t>& shape) {
  switch (role) {
    case GgufTensorRole::kMatmulWeight:
      // [out, in]; ggml's src0 orientation, our MatmulBT [N, K] with nk=true.
      return shape.size() == 2 ? shape[1] : -1;
    case GgufTensorRole::kStackedExpertWeight:
      // [E, out, in]; each expert slice is whole rows of the same K.
      return shape.size() == 3 ? shape[2] : -1;
    case GgufTensorRole::kTransformedWeight:
    case GgufTensorRole::kEmbeddingTable:
    case GgufTensorRole::kConvWeight:
    case GgufTensorRole::kVector:
      return -1;
  }
  return -1;  // unreachable for a valid enumerator (see -Wswitch above)
}

bool EnvOn(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return false;
  return !(std::strcmp(v, "") == 0 || std::strcmp(v, "0") == 0 ||
           std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}

}  // namespace

const char* Name(GgufTensorRole role) {
  switch (role) {
    case GgufTensorRole::kMatmulWeight: return "matmul_weight";
    case GgufTensorRole::kStackedExpertWeight: return "stacked_expert_weight";
    case GgufTensorRole::kTransformedWeight: return "transformed_weight";
    case GgufTensorRole::kEmbeddingTable: return "embedding_table";
    case GgufTensorRole::kConvWeight: return "conv_weight";
    case GgufTensorRole::kVector: return "vector";
  }
  return "?";
}

const char* Name(GgufResidency residency) {
  switch (residency) {
    case GgufResidency::kExpandBf16: return "expand_bf16";
    case GgufResidency::kKeepQuant: return "keep_quant";
  }
  return "?";
}

bool KeepQuantDType(uint32_t ggml_type, vt::DType* out) {
  vt::DType dt = vt::DType::kF32;
  if (!vt::BlockDTypeFromGgmlTypeId(ggml_type, &dt)) return false;
  // Q8_K is the K-quants' ACTIVATION encoding; it never appears as a file
  // weight type and has no vec_dot, so it is not keep-quant capable.
  if (!vt::cpu::HasQuantDotKernel(dt)) return false;
  if (out != nullptr) *out = dt;
  return true;
}

GgufResidency RouteGgufTensor(bool keep_quant, bool cpu_ref,
                              GgufTensorRole role, uint32_t ggml_type,
                              const std::vector<int64_t>& shape) {
  // The oracle switch wins over everything (spec gate 2).
  if (cpu_ref || !keep_quant) return GgufResidency::kExpandBf16;

  const int64_t k = KeepQuantKDim(role, shape);
  if (k <= 0) return GgufResidency::kExpandBf16;

  vt::DType dt = vt::DType::kF32;
  if (!KeepQuantDType(ggml_type, &dt)) return GgufResidency::kExpandBf16;

  // ggml_row_size's precondition: a row is a whole number of blocks. A weight
  // whose K is ragged cannot be dotted block-wise, so it expands.
  if (k % vt::BlockElems(dt) != 0) return GgufResidency::kExpandBf16;

  return GgufResidency::kKeepQuant;
}

GgufLoadPolicy GgufLoadPolicy::FromEnv() {
  GgufLoadPolicy p;
  p.cpu_ref = EnvOn("VT_CPU_REF");
  p.keep_quant = EnvOn("VT_GGUF_KEEP_QUANT");
  return p;
}

GgufResidency GgufLoadPolicy::Route(const GgufTensorInfo& tensor,
                                    GgufTensorRole role) const {
  const GgufResidency r =
      RouteGgufTensor(keep_quant, cpu_ref, role, tensor.ggml_type, tensor.shape);
  if (audit) audit(tensor.name, role, r);
  return r;
}

}  // namespace vllm
