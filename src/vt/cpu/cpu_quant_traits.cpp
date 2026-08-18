// See include/vt/quant.h for the upstream anchor and the population status of
// each column. Ported from llama.cpp @ 237ad9b96
// ggml/src/ggml-cpu/ggml-cpu.c:211-406 (`type_traits_cpu[]`).
#include "vt/quant.h"

#include <string>

namespace vt::cpu {
namespace {

// One row per block dtype, in the same order and with the same fields as
// upstream's designated-initializer table. `vec_dot_type` is the dispatch fact
// this row asserts: which activation encoding the weight type is dotted
// against. `nrows` is 1 on the generic tier everywhere; upstream's
// `#if defined(__ARM_FEATURE_MATMUL_INT8) nrows = 2` variants for Q4_0/Q8_0/
// Q4_K/Q6_K (ggml-cpu.c:234-238, :266-270, :304-308, :320-324) arrive with the
// i8mm mmla kernels in work row G6 — enabling nrows==2 without them would be a
// silent correctness bug, so the generic tier pins 1.
QuantTypeTraits MakeTraits(DType dtype, DType vec_dot_type) {
  QuantTypeTraits t;
  t.to_float = BlockToFloat(dtype);        // cpu_quant_dequant.cpp (G1)
  t.from_float = BlockFromFloat(dtype);    // cpu_quant_act.cpp   (G2)
  t.vec_dot = BlockVecDot(dtype);          // cpu_quant_dot.cpp   (G3)
  t.vec_dot_type = vec_dot_type;
  t.nrows = 1;
  return t;
}

const QuantTypeTraits* FindQuantTraits(DType dtype) {
  switch (dtype) {
    // ggml-cpu.c:230-239 — Q4_0 -> Q8_0 activations.
    case DType::kQ4_0: {
      static const QuantTypeTraits t = MakeTraits(DType::kQ4_0, DType::kQ8_0);
      return &t;
    }
    // ggml-cpu.c:262-271 — Q8_0 -> Q8_0 activations.
    case DType::kQ8_0: {
      static const QuantTypeTraits t = MakeTraits(DType::kQ8_0, DType::kQ8_0);
      return &t;
    }
    // ggml-cpu.c:283-288 — Q2_K -> Q8_K activations (DeepSeek-V4 W8).
    case DType::kQ2_K: {
      static const QuantTypeTraits t = MakeTraits(DType::kQ2_K, DType::kQ8_K);
      return &t;
    }
    // ggml-cpu.c:295-300 — Q3_K -> Q8_K activations.
    case DType::kQ3_K: {
      static const QuantTypeTraits t = MakeTraits(DType::kQ3_K, DType::kQ8_K);
      return &t;
    }
    // ggml-cpu.c:301-310 — Q4_K -> Q8_K activations.
    case DType::kQ4_K: {
      static const QuantTypeTraits t = MakeTraits(DType::kQ4_K, DType::kQ8_K);
      return &t;
    }
    // ggml-cpu.c:311-316 — Q5_K -> Q8_K activations.
    case DType::kQ5_K: {
      static const QuantTypeTraits t = MakeTraits(DType::kQ5_K, DType::kQ8_K);
      return &t;
    }
    // ggml-cpu.c:317-326 — Q6_K -> Q8_K activations.
    case DType::kQ6_K: {
      static const QuantTypeTraits t = MakeTraits(DType::kQ6_K, DType::kQ8_K);
      return &t;
    }
    // Q8_K is the K-quant ACTIVATION encoding. Upstream gives it no
    // `type_traits_cpu` row at all (it is produced by `from_float` on the
    // src1 side and consumed by the weight type's vec_dot, never dispatched
    // on); we carry a row solely so its `to_float` is reachable for tests.
    case DType::kQ8_K: {
      static const QuantTypeTraits t = MakeTraits(DType::kQ8_K, DType::kQ8_K);
      return &t;
    }
    // ggml-cpu.c IQ rows — IQ2_XXS/IQ3_XXS -> Q8_K activations (DeepSeek-V4 W8).
    // These are the ~2-3-bit codebook encodings the single-Spark DeepSeek-V4
    // GGUF vehicle's routed experts use; keeping them keep-quant is the memory
    // enabler (they carry no `from_float` — nothing quantizes INTO them).
    case DType::kIQ2_XXS: {
      static const QuantTypeTraits t = MakeTraits(DType::kIQ2_XXS, DType::kQ8_K);
      return &t;
    }
    case DType::kIQ3_XXS: {
      static const QuantTypeTraits t = MakeTraits(DType::kIQ3_XXS, DType::kQ8_K);
      return &t;
    }
    // ggml-cpu.c:352-357 — IQ2_S -> Q8_K activations. The UD-IQ2_M ffn_gate/up
    // routed-expert slabs; keep-quant against Q8_K (no from_float into it).
    case DType::kIQ2_S: {
      static const QuantTypeTraits t = MakeTraits(DType::kIQ2_S, DType::kQ8_K);
      return &t;
    }
    // IQ1_S (1.5625 bpw) is 96.92 % of Qwen3.8-2.4T-A95B UD-IQ1_S; keep-quant
    // against Q8_K, no from_float into it (nothing quantizes INTO a codebook).
    case DType::kIQ1_S: {
      static const QuantTypeTraits t = MakeTraits(DType::kIQ1_S, DType::kQ8_K);
      return &t;
    }
    // IQ1_XXXS (1.1875 bpw) is 96.92 % of Qwen3.8-2.4T-A95B UD-Q1_0. Same
    // keep-quant contract as IQ1_S, from the fork oracle `llama-cpp-unsloth`.
    case DType::kIQ1_XXXS: {
      static const QuantTypeTraits t =
          MakeTraits(DType::kIQ1_XXXS, DType::kQ8_K);
      return &t;
    }
    // ggml-cpu.c:277-282 — MXFP4 -> Q8_0 activations (NOT Q8_K: MXFP4's 32-elem
    // blocks pair with the legacy 32-elem Q8_0 encoding). The UD-IQ2_M ffn_down
    // routed-expert slabs; keep-quant, no from_float into it.
    case DType::kMXFP4: {
      static const QuantTypeTraits t = MakeTraits(DType::kMXFP4, DType::kQ8_0);
      return &t;
    }
    default:
      return nullptr;
  }
}

}  // namespace

const QuantTypeTraits& QuantTraits(DType dtype) {
  const QuantTypeTraits* t = FindQuantTraits(dtype);
  VT_CHECK(t != nullptr, std::string("QuantTraits: no CPU quant traits row for "
                                     "dtype ") + Name(dtype));
  return *t;
}

bool HasQuantDotKernel(DType dtype) {
  const QuantTypeTraits* t = FindQuantTraits(dtype);
  if (t == nullptr || t->vec_dot == nullptr) return false;
  const QuantTypeTraits* act = FindQuantTraits(t->vec_dot_type);
  return act != nullptr && act->from_float != nullptr;
}

}  // namespace vt::cpu
