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

// The possible fates of a tensor at load.
enum class GgufResidency {
  // Today's behavior: DequantGgufRowToBf16/ToF32 at load, bf16 owned tensor.
  kExpandBf16,
  // L2: the file's ggml blocks are copied verbatim into an owned buffer and
  // the weight stays block-typed ([N, K], nk = true).
  kKeepQuant,
  // L6: an F16 file weight stays F16, resident in the file's own [N, K] order
  // (nk = true), consumed by the elementwise f16 GEMM directly instead of being
  // dequantized (RE-rounded) to bf16 at load. Mirrors llama.cpp, which keeps
  // f16 weights resident and computes on them with ggml_vec_dot_f16
  // (src/llama-model-loader.cpp:1385 mmap `load_data_for`; ggml-cpu/vec.cpp:264
  // f16 mul_mat). The residency change is byte-for-byte lossless (the SAME f16
  // bytes, read not re-encoded); the COMPUTE dtype changes bf16 -> f16 input, so
  // tokens may move — and toward the file's f16 truth, see the leaf spec's
  // correctness section.
  kKeepF16,
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

// True when `ggml_type` is F16 (ggml type id 1) — the one native-float encoding
// L6 keeps resident. F32/BF16 weights are not kept native: F32 file tensors are
// only ever norms/vectors (value-transformed), and no published Qwen GGUF stores
// a bf16 matmul weight.
bool KeepF16DType(uint32_t ggml_type);

// The pure decision. `shape` is the GGUF reader's torch-order shape.
// kKeepQuant requires ALL of: keep_quant enabled, cpu_ref off, a role whose
// bytes are taken verbatim, the expected rank, a supported encoding with an
// executable vec_dot kernel, and K a whole number of blocks.
// kKeepF16 requires ALL of: keep_f16 enabled, cpu_ref off, the tensor NOT
// keep-quant eligible, a role whose bytes are taken verbatim (incl. the F16
// embedding table, which is a plain gather), and an F16 encoding. Anything else
// is kExpandBf16 — the decision is total and never throws.
GgufResidency RouteGgufTensor(bool keep_quant, bool keep_f16, bool cpu_ref,
                              GgufTensorRole role, uint32_t ggml_type,
                              const std::vector<int64_t>& shape);

// True when the device this process will actually run the forward on can
// execute `OpId::kMatmulBTQuant` — i.e. when a block-typed weight has a
// consumer. This is the condition CIQ G4 flipped the keep-quant default onto:
// today only the CPU backend registers the quantized GEMM, so a CUDA build
// keeps expanding to bf16 (CUDA GGUF compute-in-quant is a future backend
// row), and the day that kernel is registered for another device this default
// follows it with no edit here.
bool GgufQuantComputeAvailable();

// Loader-wide residency policy.
struct GgufLoadPolicy {
  // Master switch for keep-quant residency. The STRUCT default stays false so
  // a default-constructed policy is the historical all-expand load; the
  // PRODUCTION default is decided by FromEnv(), which turns it ON wherever
  // GgufQuantComputeAvailable() holds (CIQ G4). VT_GGUF_KEEP_QUANT=1 forces it
  // on, =0/off/false forces it off (the opt-out).
  bool keep_quant = false;
  // When a matmul weight CANNOT keep its blocks (an f16/f32 file tensor, or an
  // encoding without a vec_dot) but the quantized path is otherwise active,
  // expand it to bf16 in the file's OWN [N, K] order with nk = true instead of
  // transposing it to Matmul-B [K, N]. GGUF's disk order is already [N, K], so
  // the transpose was extra load-time work to reach the SLOWER kernel:
  // `kMatmulBT` reads the weight row contiguously while `kMatmul` strides by N
  // down the K loop, measured at 1.3-3.0x on aarch64
  // (specs/cpu-llamacpp-floor-remeasure-2026-07-22.md lever 2). The CPU
  // kernels differ ONLY in that weight offset — `MatmulOneChunk<kBT>`
  // (cpu_ops.cpp:70-83) keeps the same sequential f32 K reduction — so the
  // result is bit-identical to the transposed load. It is tied to the same
  // availability condition as keep_quant (and forced off by cpu_ref) because
  // on a device whose GEMM is not that shared kernel — cuBLASLt picks its algo
  // from the operand layout — orientation is NOT numerically free.
  bool expand_nk = false;
  // L6 — keep-f16 residency (OPT-IN, VT_GGUF_KEEP_F16=1; default OFF). An F16
  // matmul weight (and the F16 embedding table) STAYS F16 resident in the file's
  // own [N, K] order (nk = true) rather than being dequantized to bf16 at load,
  // and the elementwise GEMM consumes the f16 rows directly (LoadF32/
  // WidenRowToF32 already widen f16; the SIMD tiers have an f16 vec_dot). This
  // mirrors llama.cpp, which keeps f16 weights resident and computes on them
  // (ggml_vec_dot_f16), so the compute is CLOSER to the file's f16 truth than the
  // lossy bf16 re-expansion (empirically byte-identical greedy tokens on the 2B
  // bench file — no movement). It is DEFAULT OFF because the binding measurement
  // REFUTED the premise that it closes the CPU RSS gap: L5's page-release already
  // recovered the f16 second materialization, so keep-f16 only swaps an anonymous
  // bf16 buffer for equal-size file-backed f16 pages (RSS-neutral, -52 MB) while
  // moving the f16 first-touch faults into the timed prefill (TTFT regresses
  // below the llama.cpp floor). The remaining RSS gap is the engine's anonymous
  // activation/KV workspace, not weights. It rides expand_nk (CPU-only, off under
  // cpu_ref). See docs/BENCHMARKS.md and the leaf spec §L6.
  bool keep_f16 = false;
  // VT_CPU_REF=1 — the parity ORACLE switch (spec gate 2). Forces the full
  // dequant-to-bf16 load path regardless of `keep_quant`, so the bit-stable
  // reference numerics stay reachable once keep-quant becomes the default.
  bool cpu_ref = false;
  // L5 — mmap residency. A weight that KEEPS its blocks is consumed IN PLACE
  // out of the GGUF's read-only mapping instead of being copied into an owned
  // buffer, holding a refcount on the mapping (GgufMapping) for its lifetime.
  // llama.cpp does exactly this when `use_mmap` is set
  // (src/llama-model-loader.cpp:1385 `load_data_for`, which points the tensor at
  // `mapping->addr + w.offs` rather than `read_raw`-ing into it @ 237ad9b96);
  // L2 deliberately started with the copy, and this is the recorded follow-up.
  // Byte-exactness is structural: the SAME file bytes in the SAME [N, K] order,
  // read rather than memcpy'd first. Rides the keep_quant condition (there is
  // nothing to borrow when everything expands) and is forced off by cpu_ref.
  // VT_GGUF_MMAP=0 is the opt-out, kept so the copy arm stays A/B-able.
  bool mmap_residency = false;
  // L5 — tied-head sharing. When the file omits `output.weight`, the head IS
  // `token_embd.weight` (llama.cpp calls this TENSOR_DUPLICATED), so the model
  // materializes ONE vocab matrix and both the gather table and the lm_head GEMM
  // weight view it. Only possible when the head expands in the file's own
  // [N, K] order (expand_nk), because that is the case in which the head's bytes
  // are elementwise IDENTICAL to the embedding table's; the transposing
  // Matmul-B path builds genuinely different bytes and shares nothing.
  bool share_tied_head = false;
  // GDN split-projection orientation (CPU prefill lever, 2026-07-23). The GDN
  // linear-attention layers' expanded in_proj_qkv / in_proj_z / in_proj_b /
  // in_proj_a and out_proj were the ONE family of matmul weights that
  // LoadGdnGguf still transposed into Matmul-B [K, N] (nk = false) after G4
  // gave every OTHER expanded weight the [N, K] treatment via expand_nk — a
  // fresh op-dispatch profile of the current binary put those 4x18 = 72 GEMMs
  // at 17.9 % of prefill, running the N-striding `kMatmul` instead of the
  // M-blocked `kMatmulBT`. This flag keeps them in the file's own [N, K] order
  // (nk = true) exactly as expand_nk does for the rest, which is bit-identical
  // for the same reason (the CPU kernels differ only in the weight offset, same
  // sequential f32 K reduction). Rides expand_nk (so cpu_ref reproduces the
  // historical transpose path), with VT_GGUF_GDN_NK=0 the narrow same-binary
  // A/B opt-out that reverts ONLY the GDN projections.
  bool gdn_expand_nk = false;
  // CIQ G7 — repack-at-load for the q8_0 quant GEMM. When on, a KEEP-QUANT q8_0
  // weight whose bytes are copied resident is REPACKED once into the CPU i8mm
  // interleave (block_q8_0x4) so `kMatmulBTQuant` runs the pre-shuffled
  // gemm/gemv with no in-register row shuffles — the profile's #1 CPU prefill
  // lever (kMatmulBTQuant = 55 % of prefill). Repacking is a byte permutation
  // and the gemm folds the scale in the same order with a non-fused MAC, so it
  // is BIT-IDENTICAL to the non-repacked path. Rides keep_quant AND
  // vt::cpu::QuantRepackActive() (i8mm present, not disabled by
  // VT_CPU_QUANT_REPACK=0), and is forced off by cpu_ref. Because the transform
  // mutates the buffer it selects the COPY residency for the tensors it touches
  // (an mmap borrow is read-only); every other tensor keeps its chosen
  // residency. VT_CPU_QUANT_REPACK=0 is the same-binary A/B opt-out (it also
  // turns off the kernel probe, so the two can never disagree).
  bool quant_repack = false;
  // Optional observer; null in production.
  GgufRoutingAudit audit;

  // Reads VT_CPU_REF and VT_GGUF_KEEP_QUANT. A variable set to "0", "false",
  // "off" or empty is OFF; any other value is ON. VT_GGUF_KEEP_QUANT UNSET
  // means "decide by GgufQuantComputeAvailable()" — the G4 default.
  static GgufLoadPolicy FromEnv();

  // Route one tensor and notify `audit`. This is the ONLY entry point the
  // loader uses, so every routed tensor is observable.
  GgufResidency Route(const GgufTensorInfo& tensor, GgufTensorRole role) const;
};

}  // namespace vllm
