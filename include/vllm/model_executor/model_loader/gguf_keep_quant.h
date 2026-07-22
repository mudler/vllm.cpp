// vllm.cpp ORIGINAL — GGUF keep-quantized residency POLICY (leaf spec
// .agents/specs/gguf-keep-quant-loader.md, work rows L2/L3). No upstream vLLM
// mirror; the behavior mirrors llama.cpp @ 237ad9b96, where a weight KEEPS the
// ggml_type it has in the file from load to GEMM:
//   src/llama-model-loader.cpp:1047  create_tensor() keeps the file's type
//   src/llama-model-loader.cpp:1385  load_data_for() copies raw block bytes
//   ggml/src/ggml.c                  ggml_row_size — rows are WHOLE blocks,
//                                    which is the eligibility rule below
//
// This header owns the DECISION only (which tensor keeps its blocks, which is
// expanded to bf16 at load); the residency buffers themselves are built by
// OwnGgufQuantBlocks in models/qwen3_5_gguf_weights.h, keeping the
// model_loader layer free of any dependency on the model weight structs.
//
// TOTALITY IS THE CONTRACT. Every tensor the GGUF loader touches is routed
// through RouteGgufTensor with an EXPLICIT role; there is no silent default.
// The role switch carries no `default:` label, so adding a role without
// deciding its residency is a -Werror=switch BUILD failure, and the optional
// `audit` hook lets a test prove that the set of routed tensors is exactly the
// set of tensors in the file.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"

namespace vllm {

// What a GGUF tensor IS to the model, which is what decides whether its ggml
// blocks can survive to the GEMM. The distinction that matters is whether the
// loader must look INSIDE a block: any value rewrite or column permutation
// does, so those tensors can never stay quantized.
enum class GgufTensorRole {
  // 2-D [out, in] GEMM weight taken verbatim from the file (attention q/k/v/o,
  // dense ffn gate/up/down, router gate, shared-expert projections, lm_head,
  // and the GDN projections when no V-head reorder is required).
  kMatmulWeight,
  // 3-D [E, out, in] stacked expert GEMM weights. Each expert slice is a whole
  // number of ROWS, hence a whole number of blocks, so the split is a byte
  // range — no block is ever cut.
  kStackedExpertWeight,
  // A tensor whose VALUE or intra-row LAYOUT is rewritten at load: the (w - 1)
  // RMSNorm rewrite, ssm_a = log(-x), and the V-head reorders (the out_proj
  // reorder permutes COLUMNS, which live inside blocks). Never keep-quant.
  kTransformedWeight,
  // Embedding table — a gather, not a GEMM (llama.cpp likewise dequantizes
  // embedding rows on the fly). A quantized-gather op is a follow-up row.
  kEmbeddingTable,
  // conv1d filter [conv_dim, K]: consumed by the depthwise conv, not a GEMM.
  kConvWeight,
  // 1-D norm weight / bias / scale / gate vector.
  kVector,
};

// The two possible fates of a tensor at load.
enum class GgufResidency {
  // Today's behavior: DequantGgufRowToBf16/ToF32 at load, bf16 owned tensor.
  kExpandBf16,
  // L2: the file's ggml blocks are copied verbatim into an owned buffer and
  // the weight stays block-typed ([N, K], nk = true).
  kKeepQuant,
};

const char* Name(GgufTensorRole role);
const char* Name(GgufResidency residency);

// Called once per tensor the loader routes, with the decision it made. Used by
// the routing tests to prove total coverage of a real file's tensor list.
using GgufRoutingAudit =
    std::function<void(const std::string& name, GgufTensorRole role,
                       GgufResidency residency)>;

// True when `ggml_type` is one of the six encodings that can currently stay
// resident (Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K), writing its vt block dtype to
// `*out`. False for the unquantized types, for Q8_K (activation-only, never a
// file weight type) and for every unported encoding.
bool KeepQuantDType(uint32_t ggml_type, vt::DType* out);

// The pure decision. `shape` is the GGUF reader's torch-order shape.
// kKeepQuant requires ALL of: keep_quant enabled, cpu_ref off, a role whose
// bytes are taken verbatim, the expected rank, a supported encoding with an
// executable vec_dot kernel, and K a whole number of blocks. Anything else is
// kExpandBf16 — the decision is total and never throws.
GgufResidency RouteGgufTensor(bool keep_quant, bool cpu_ref,
                              GgufTensorRole role, uint32_t ggml_type,
                              const std::vector<int64_t>& shape);

// Loader-wide residency policy.
struct GgufLoadPolicy {
  // Master switch for keep-quant residency. DEFAULT OFF: until CIQ work row
  // G4 routes the model's MatmulBT call sites onto kMatmulBTQuant, nothing can
  // CONSUME a block-typed weight, so enabling it by default would break the
  // forward rather than speed it up. G4 flips this default; VT_GGUF_KEEP_QUANT
  // =1 opts in today (that is what the residency/losslessness tests use).
  bool keep_quant = false;
  // VT_CPU_REF=1 — the parity ORACLE switch (spec gate 2). Forces the full
  // dequant-to-bf16 load path regardless of `keep_quant`, so the bit-stable
  // reference numerics stay reachable once keep-quant becomes the default.
  bool cpu_ref = false;
  // Optional observer; null in production.
  GgufRoutingAudit audit;

  // Reads VT_CPU_REF and VT_GGUF_KEEP_QUANT. A variable set to "0", "false",
  // "off" or empty is OFF; any other value is ON.
  static GgufLoadPolicy FromEnv();

  // Route one tensor and notify `audit`. This is the ONLY entry point the
  // loader uses, so every routed tensor is observable.
  GgufResidency Route(const GgufTensorInfo& tensor, GgufTensorRole role) const;
};

}  // namespace vllm
