// vllm.cpp ORIGINAL — GGUF keep-quantized residency policy. See
// gguf_keep_quant.h for the upstream (llama.cpp @ 237ad9b96) anchors and the
// totality contract.
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

#include <cstdlib>
#include <cstring>

#include "vllm/platforms/interface.h"
#include "vt/ops.h"
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

// Tri-state: unset -> `fallback`; "0"/"false"/"off"/"" -> false; else true.
bool EnvOnOr(const char* name, bool fallback) {
  return std::getenv(name) == nullptr ? fallback : EnvOn(name);
}

}  // namespace

bool GgufQuantComputeAvailable() {
  return vt::OpRegistered(vt::OpId::kMatmulBTQuant,
                   vllm::platforms::CurrentPlatform().device_type());
}

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
  // CIQ G4 flipped this default: keep-quant is ON wherever the running device
  // can execute the quantized GEMM. VT_GGUF_KEEP_QUANT is the two-way
  // override that survives the flip (=0 is the opt-out the spec promised).
  p.keep_quant = EnvOnOr("VT_GGUF_KEEP_QUANT", GgufQuantComputeAvailable());
  // The orientation win rides the same availability condition, and the oracle
  // switch turns it off with everything else so VT_CPU_REF=1 reproduces the
  // historical load byte for byte.
  p.expand_nk = p.keep_quant && !p.cpu_ref;
  // L5. Both ride the same availability condition as the residency they refine,
  // and both are forced off by the oracle switch, so VT_CPU_REF=1 keeps
  // reproducing the historical load byte for byte and allocation for allocation.
  p.mmap_residency = EnvOnOr("VT_GGUF_MMAP", p.keep_quant) && !p.cpu_ref;
  p.share_tied_head = EnvOnOr("VT_GGUF_SHARE_TIED_HEAD", p.expand_nk) && p.expand_nk;
  // GDN split-projection orientation. Rides expand_nk (so VT_CPU_REF=1
  // reproduces the historical transpose); VT_GGUF_GDN_NK=0 is the narrow
  // same-binary A/B that reverts only the GDN projections to [K, N].
  p.gdn_expand_nk = EnvOnOr("VT_GGUF_GDN_NK", p.expand_nk) && p.expand_nk;
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
