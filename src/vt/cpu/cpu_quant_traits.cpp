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
    // llama.cpp @ b10451 ggml-cpu.c:259-264 — Q5_0 -> Q8_0 activations. Added
    // for `qwen4exp`, whose 640-wide expert K rules out every K-quant and whose
    // `-Q4_K_M` recipe therefore lands on Q5_0 through llama.cpp's own
    // `tensor_type_fallback`. No `from_float`: nothing here quantizes an
    // activation INTO Q5_0 (upstream's row does carry one, and porting the
    // encoder would be dead code — see the k-quant encoders, deliberately
    // unported for the same reason).
    case DType::kQ5_0: {
      static const QuantTypeTraits t = MakeTraits(DType::kQ5_0, DType::kQ8_0);
      return &t;
    }
    // llama.cpp @ b10451 ggml-cpu.c:379-384 — IQ4_NL -> Q8_0 activations (NOT
    // Q8_K: IQ4_NL's block is 32 elements, so it pairs with the legacy 32-element
    // activation encoding, exactly like MXFP4). This is the encoding the shipped
    // Qwen3.8-Flash-Next UD-IQ1_S uses for all 48 `ffn_down_exps` AND for the
    // 20M-entry n-gram table; keep-quant is the memory enabler for both.
    case DType::kIQ4_NL: {
      static const QuantTypeTraits t = MakeTraits(DType::kIQ4_NL, DType::kQ8_0);
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
    // llama.cpp @ b10451 ggml-cpu.c:342-347 — IQ2_XS -> Q8_K activations. 82 of
    // the staged `unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL` artifact's 1412
    // tensors are IQ2_XS, 53.33 GiB on disk against 369.00 GiB as bf16: this
    // row is what keeps them compressed. No `from_float` (upstream's row has
    // none either — nothing quantizes an activation INTO a codebook).
    case DType::kIQ2_XS: {
      static const QuantTypeTraits t = MakeTraits(DType::kIQ2_XS, DType::kQ8_K);
      return &t;
    }
    // llama.cpp @ b10451 ggml-cpu.c:385-390 — IQ4_XS -> Q8_K activations. NOT
    // the Q8_0 of its codebook sibling IQ4_NL (:379-384): IQ4_XS reuses
    // `kvalues_iq4nl` but its block is a 256-element super-block, so it pairs
    // with the 256-element activation encoding. Upstream's row does carry a
    // `from_float` (`quantize_row_iq4_xs`); porting it would be dead code here
    // for the same reason the k-quant encoders are unported.
    case DType::kIQ4_XS: {
      static const QuantTypeTraits t = MakeTraits(DType::kIQ4_XS, DType::kQ8_K);
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
    // TQ2_0/TQ1_0 (Vulkan-native ternary keep-quant): 256-element blocks dotted
    // against Q8_K activations. No CPU vec_dot — the keep-quant dot is
    // Vulkan-only. The CPU path uses the dequant-composite fallback
    // (BlockToFloat -> f32 matmul), which is the reference oracle the Vulkan
    // tests compare against. HasQuantDotKernel returns false because vec_dot
    // is nullptr, so MatmulBTQuantKernel takes the to_float branch.
    case DType::kTQ2_0: {
      static const QuantTypeTraits t = MakeTraits(DType::kTQ2_0, DType::kQ8_K);
      return &t;
    }
    case DType::kTQ1_0: {
      static const QuantTypeTraits t = MakeTraits(DType::kTQ1_0, DType::kQ8_K);
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
