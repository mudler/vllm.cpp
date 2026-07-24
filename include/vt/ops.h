// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "vt/fused_recipe.h"
#include "vt/op_provider.h"
#include "vt/tensor.h"

namespace vt {

// Upstream-compatible vllm::ScalarType IDs without importing the Marlin-only
// scalar_type.hpp into the backend-neutral public surface. The bit packing is
// ported from csrc/core/scalar_type.hpp:80-151 @ e24d1b24. Storage DType and
// semantic type are deliberately separate: DType::kI8 never guesses whether
// its bytes contain int8, FP4, FP8, or a block scale.
using ScalarTypeId = int64_t;

namespace scalar_type {

enum class NanRepr : uint8_t { kNone = 0, kIeee754 = 1, kExtendedRangeMaxMin = 2 };

constexpr ScalarTypeId Make(uint8_t exponent, uint8_t mantissa, bool is_signed,
                            int32_t bias, bool finite_values_only, NanRepr nan_repr) {
  const uint64_t bias_bits = static_cast<uint32_t>(bias);
  return static_cast<ScalarTypeId>(
      static_cast<uint64_t>(exponent) |
      (static_cast<uint64_t>(mantissa) << 8) |
      (static_cast<uint64_t>(is_signed) << 16) |
      (bias_bits << 17) |
      (static_cast<uint64_t>(finite_values_only) << 49) |
      (static_cast<uint64_t>(nan_repr) << 50));
}

inline constexpr ScalarTypeId kF32 = Make(8, 23, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kF16 = Make(5, 10, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kBF16 = Make(8, 7, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kI8 = Make(0, 7, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kI32 = Make(0, 31, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kI64 = Make(0, 63, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kI4 = Make(0, 3, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kU4 = Make(0, 4, false, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kFE2M1f = Make(2, 1, true, 0, true, NanRepr::kNone);
inline constexpr ScalarTypeId kFE4M3fn =
    Make(4, 3, true, 0, true, NanRepr::kExtendedRangeMaxMin);
inline constexpr ScalarTypeId kFE8M0fnu =
    Make(8, 0, false, 0, true, NanRepr::kExtendedRangeMaxMin);

}  // namespace scalar_type

ScalarTypeId ToScalarType(DType dtype);

enum class KernelLayout : uint8_t {
  kStrided = 0,
  kPackedTwoFp4PerByte,
  kBlockScaleLinear,
  kBlockScaleSwizzled,
  kMarlinInterleaved,
};

// Explicit output layout for dynamic NVFP4 activation block scales. Keep this
// separate from KernelLayout: it is an op argument which selects how a producer
// addresses its output, not metadata inferred from a Tensor's shape. Aligned
// linear and CUTLASS-swizzled buffers can have the same dimensions.
enum class Fp4ScaleLayout : uint8_t {
  kLinear = 0,
  kCutlassSwizzled,
};

struct KernelTensorDesc {
  void* data = nullptr;
  DType storage_dtype = DType::kF32;
  ScalarTypeId scalar_type = vt::scalar_type::kF32;
  Device device;
  int rank = 0;
  int64_t shape[kMaxRank] = {0, 0, 0, 0};
  int64_t stride[kMaxRank] = {0, 0, 0, 0};
  KernelLayout layout = KernelLayout::kStrided;
};

KernelTensorDesc Describe(const Tensor& tensor, ScalarTypeId scalar_type,
                          KernelLayout layout);

enum class OpId : uint8_t {
  kMatmul,
  kRmsNorm,
  kSiluAndMul,
  kRopeNeox,
  kEmbedding,
  kCausalConv1dFwd,
  kCausalConv1dUpdate,
  kL2Norm,
  kRmsNormGated,
  kGdnPrefill,
  kGdnDecode,
  kGdnPackedDecode,
  kMoeRouterTopK,
  kMoeCombine,
  kAttention,
  kReshapeAndCache,
  kConcatAndCacheMla,
  kMlaDecodeAttention,
  kMlaPrefillAttention,
  kGatherMlaCache,
  kMergeAttnStates,
  kPagedAttention,
  kApplyTemperature,
  kGreedyArgmax,
  kApplyTopKTopP,
  kComputeProbs,
  kComputeLogprobs,
  kRandomSample,
  kApplyPenalties,
  kApplyMinP,
  kApplyLogitBias,
  kApplyTokenMask,
  kApplyAllowedTokenIds,
  kMatmulNvfp4,
  kScaledFp4Quant,
  kSiluMulFp4Quant,
  kSiluAndMulFp4Quant,
  kSigmoidGateFp4Quant,
  kMatmulNvfp4Fp4,
  kMatmulNvfp4Cutlass,
  kMatmulFp8Cutlass,
  kMatmulFp8CublasLt,
  kQuantFp8Static,
  kSwizzleBlockscale,
  kMoeGroupedGemmNvfp4,
  kMoeSiluMul,
  kCastBf16,
  kCastF32,
  kMulColVecF32,
  kAttnGateSplit,
  kSigmoidGateBf16,
  kGdnGBeta,
  kGdnConvSplit,
  kQkvSplit,
  kSharedExpertGate,
  kMoeCombineGate,
  kMoeGroupedGemmNvfp4Marlin,
  kGdnPostConv,
  kRopeCosSinCache,
  kAttnQkNormRopeGate,
  kFusedChain,
  kRmsNormQuantFp8,
  kRmsNormGatedQuantFp8,
  kMatmulBT,
  // kMatmulBT with a BLOCK-QUANTIZED [N,K] weight kept in its native ggml
  // encoding — llama.cpp's `ggml_compute_forward_mul_mat`
  // (ggml/src/ggml-cpu/ggml-cpu.c:1245-1443 @ 237ad9b96).
  kMatmulBTQuant,
  // W0-only raw-signature probe for the shared drop-in adapter boundary. It is
  // not a production kernel-family migration.
  kDropinProbe,
  // In-place base/MRoPE rotation from a supplied dtype-specific global cache.
  kRopeFromCache,
  // Indexed persistent GDN cache boundary: one launch replaces the former
  // per-row copies plus a separate BF16<->F32 cast.
  kGdnStateGather,
  kGdnStateScatter,
  // BF16 grouped-MoE GEMM: the dtype-native analog of kMoeGroupedGemmNvfp4 (no
  // fp4 decode). Powers the Qwen3-Coder (Qwen3MoeForCausalLM) fast bf16 MoE path.
  kMoeGroupedGemmBf16,
  // Cross-family dense primitives introduced by the OPT (`OPTForCausalLM`)
  // bring-up — the pre-RMSNorm/pre-SwiGLU transformer vocabulary every
  // non-Qwen family needs. All three mirror torch/vLLM semantics exactly:
  //   kLayerNorm — `nn.LayerNorm` (mean+variance normalization with a BIAS
  //                term), as used by opt.py:146-148,164-166,248-251. Distinct
  //                from kRmsNorm, which subtracts no mean and has no bias.
  //   kRelu      — `get_act_fn("relu")` (opt.py:156), i.e. ReLU rather than the
  //                SwiGLU every Qwen model uses.
  //   kAdd       — elementwise add, plus the rank-1 row-BROADCAST form that
  //                applies a `nn.Linear` bias (opt.py:90-104,149-163: OPT's
  //                q/k/v/out/fc1/fc2 all carry `enable_bias` bias terms, which
  //                the bias-free Qwen projections never needed).
  kLayerNorm,
  kRelu,
  kAdd,
  // Batched dense GEMM (`torch.bmm`). The primitive MLA weight absorption is
  // expressed in — mla_attention.py:789 (q-side W_UK fold) and :1034 (W_UV
  // v-up-projection). See vt::BatchedMatmul.
  kBatchedMatmul,
  // MLA nope|rope head concatenation — upstream `concat_mla_q`
  // (csrc/libtorch_stable/concat_mla_q.cuh) and `_concat_k_nope_k_pe`
  // (mla_attention.py:2063-2092). See vt::ConcatMlaNopeRope.
  kConcatMlaNopeRope,
  // Gemma GeGLU activation: gelu_pytorch_tanh(gate) * up — the tanh-approx GELU
  // on the gate half elementwise-multiplied by the up half. The GeGLU analog of
  // kSiluAndMul, mirroring vLLM GeluAndMul(approximate="tanh") (activation.py).
  // NEW for the Gemma family (Gemma 1/2/3/4 MLP).
  kGeluAndMul,
  // Elementwise multiply by a runtime scalar: out[i] = x[i] * scalar (f32
  // compute, out-dtype store). The Gemma embedding normalizer
  // `embed_tokens(ids) * sqrt(hidden_size)` (gemma3.py:328-341). Additive; no
  // Qwen/Llama/OPT/GLM model sets it.
  kMulScalar,
  // Logit soft-cap: out[i] = cap * tanh(x[i] / cap) (f32 compute, out-dtype
  // store). The Gemma-2 final logit soft-cap (gemma2.py:344-345,
  // LogitsProcessor(soft_cap=final_logit_softcapping)) — a monotone squashing of
  // the logits. NEW for the Gemma-2 family. Additive; default-unused otherwise.
  kSoftCap,
  kCount
};

enum class WorkspaceSlot : uint8_t {
  kWorkspace = 0,
  kOutput,
  kLse,
  kSemaphore,
  kDeviceScalar0,
  kDeviceScalar1,
};

enum class WorkspaceInit : uint8_t {
  kUninitialized = 0,
  kZeroOnFirstUse,
  kZeroEachUse,
};

struct WorkspaceKey {
  Device device;
  uint64_t queue_id = 0;
  uintptr_t native_handle = 0;
  OpId op = OpId::kMatmul;
  WorkspaceSlot slot = WorkspaceSlot::kWorkspace;

  friend bool operator==(const WorkspaceKey& a, const WorkspaceKey& b) {
    return a.device == b.device && a.queue_id == b.queue_id &&
           a.native_handle == b.native_handle && a.op == b.op && a.slot == b.slot;
  }
};

WorkspaceKey MakeWorkspaceKey(const Queue& q, OpId op, WorkspaceSlot slot);

struct DropinProbeArgs {
  ScalarTypeId scalar_type = vt::scalar_type::kF32;
  KernelLayout layout = KernelLayout::kStrided;
  size_t workspace_bytes = sizeof(uint32_t);
  size_t workspace_alignment = alignof(uint32_t);
  WorkspaceInit workspace_init = WorkspaceInit::kZeroEachUse;
  WorkspaceSlot workspace_slot = WorkspaceSlot::kWorkspace;
  WorkspaceSlot scalar_slot = WorkspaceSlot::kDeviceScalar0;
  float scalar = 0.0f;
};

struct RmsNormArgs {
  float eps = 1e-6f;
  bool gemma = false;  // weight applied as (1 + w), GemmaRMSNorm style
};

// torch `nn.LayerNorm` arguments (opt.py:146-148,164-166,248-251 construct it
// with the default eps=1e-5 and `elementwise_affine=config.
// layer_norm_elementwise_affine`). Unlike RmsNormArgs there is no gemma variant:
// LayerNorm subtracts the mean and applies weight AND bias.
struct LayerNormArgs {
  float eps = 1e-5f;
};

struct RopeArgs {
  float base = 10000.0f;
  int rotary_dim = 0;  // <= head_dim; even
  bool is_neox_style = true;
  // Empty (all zero) for 1-D RoPE. For positions[3,T], the entries are the
  // temporal/height/width counts in the half-rotary frequency dimension.
  std::array<int32_t, 3> mrope_section = {0, 0, 0};
  bool mrope_interleaved = false;

  // Llama-3 rope frequency rescaling (rope_type=="llama3", e.g. Llama-3.2). When
  // llama3_scaling_factor <= 0 (the default) NO rescale is applied and the RoPE
  // is byte-identical to plain RoPE — so every existing caller (Qwen, the gate
  // models) that leaves these zero is UNCHANGED. When set, the base inv_freq
  // (base^(-2i/rotary_dim)) is rescaled per frequency by a piecewise low/high
  // wavelength interpolation, mirroring vLLM Llama3RotaryEmbedding._compute_inv_freq
  // (vllm/model_executor/layers/rotary_embedding/llama3_rope.py:33-54). Consumed
  // by RopeNeox + RopeCosSinCache (the cache feeds RopeFromCache, so no extra
  // field is needed there).
  float llama3_scaling_factor = 0.0f;    // rope_scaling "factor" (0 => disabled)
  float llama3_low_freq_factor = 0.0f;   // rope_scaling "low_freq_factor"
  float llama3_high_freq_factor = 0.0f;  // rope_scaling "high_freq_factor"
  float llama3_orig_max_position = 0.0f;  // "original_max_position_embeddings"
};

// GDN op args (.agents/specs/gdn-semantics.md is the formula reference; sections
// cited on each op below).
struct CausalConv1dArgs {
  // Upstream `activation` is "silu"/"swish" (→ silu) or None (→ identity);
  // Qwen GDN always uses silu (gdn-semantics.md §2).
  bool silu_activation = true;
};

struct L2NormArgs {
  float eps = 1e-6f;  // upstream default (gdn-semantics.md §4)
};

struct RmsNormGatedArgs {
  float eps = 1e-6f;
  // Gate activation: silu by default; "sigmoid" allowed by upstream
  // output_gate_type (gdn-semantics.md §5). norm_before_gate=True and
  // group_size=None (the only configuration Qwen GDN uses) are baked in.
  bool sigmoid_gate = false;
};

struct GdnArgs {
  // q scale, applied to q only after l2norm; upstream default Dk^-0.5
  // (gdn-semantics.md §1). Must be set explicitly (> 0).
  float scale = 0.0f;
  // OPTIONAL host-resident query_start_loc[N+1] (same values as the device
  // `query_start_loc` tensor). When set, the CUDA chunked-prefill path builds
  // its chunk layout from these host values + a device meta-kernel, avoiding
  // the per-layer D2H copy + cudaStreamSynchronize (the prefill host-tax — it
  // forced host↔GPU lockstep every GDN layer, ~67% GPU-idle). nullptr => the
  // path falls back to the D2H+sync (op tests / callers without host qsl).
  // Mirrors the decode StepDevInputs device-resident metadata pattern.
  const int32_t* query_start_loc_host = nullptr;
};

// Dense causal attention args (.agents/specs/qwen36-forward-notes.md §5 is the
// formula reference — Qwen3NextAttention's core scaled-dot-product).
struct AttentionArgs {
  // Softmax scale, applied to the qk dot product. Upstream sets it to
  // head_dim^-0.5 (Qwen3NextAttention.scaling). Must be set explicitly (> 0).
  float scale = 0.0f;
  // Causal masking: key position j attends only when j <= query position i.
  // Always true for the M0.9 decoder path (bidirectional is a M1.6+ concern).
  bool causal = true;
};

// Backend-neutral local-attention window, matching FlashAttention's
// `window_size=(left, right)` convention. The bounds are inclusive distances
// from the bottom-right-aligned absolute query position: (W-1, 0) is a causal
// decoder window of W tokens and (W-1, W-1) is the symmetric encoder form.
// Full attention is represented by std::nullopt on PagedAttentionArgs, never by
// a backend-specific sentinel pair.
struct AttentionWindow {
  int32_t left = 0;
  int32_t right = 0;
};

// Paged attention args (M1.6). Same softmax convention as AttentionArgs — the
// paged op generalizes the dense M0.9 attention to the varlen/batched/paged
// case and MUST agree with it on the single-sequence contiguous read.
struct PagedAttentionArgs {
  // Softmax scale, applied to the qk dot product (upstream FlashAttentionImpl
  // self.scale = head_size^-0.5). Must be set explicitly (> 0).
  float scale = 0.0f;
  // Causal masking: a query token at absolute position p attends only to key
  // positions j <= p. True for the decoder path; non-causal carried for
  // fidelity (matches AttentionArgs.causal).
  bool causal = true;
  // OPTIONAL local-attention bounds. For an absolute query position p, visible
  // keys are intersected with [p-left, p+right] after the causal/full bound is
  // applied. Query positions use FlashAttention's bottom-right alignment:
  // p = seq_len - query_len + local_query_index. std::nullopt preserves the
  // existing full causal/non-causal behavior exactly.
  std::optional<AttentionWindow> window_size = std::nullopt;
  // OPTIONAL attention logit soft-cap (vLLM Attention(logits_soft_cap=...),
  // gemma2.py:202 attn_logit_softcapping). When > 0 each pre-softmax score S is
  // replaced by cap * tanh(S / cap) before the online softmax. 0.0 (default)
  // leaves the plain scaled-dot path byte-identical — every existing model uses
  // the default, so this is diff-inert for them. Gemma-2/4 set it (50.0).
  float logits_soft_cap = 0.0f;
  // OPTIONAL host-resident query_start_loc[num_reqs+1] (same values as the
  // device `query_start_loc` tensor). When set, the CUDA prefill flash/WMMA
  // launchers size the per-request query-tile grid from these host values and
  // build the device tile array with a device meta-kernel, avoiding the
  // per-layer D2H copy + cudaStreamSynchronize that drained the pipeline every
  // full-attention prefill layer (~10-12 syncs/step; prefill only 43.7%
  // GPU-busy). nullptr => the launcher falls back to the D2H+sync (op unit tests
  // / callers without a host qsl). Mirrors GdnArgs::query_start_loc_host and the
  // decode StepDevInputs device-resident metadata pattern.
  const int32_t* query_start_loc_host = nullptr;
  // OPTIONAL host-known max context length in this batch (max over the device
  // `seq_lens` values = CommonAttentionMetadata::max_seq_len; an upper bound is
  // safe — it only sizes grids/rounded dims, per-request geometry stays on the
  // device values). When > 0 the FA-2 prefill launcher sizes its grid without a
  // device read (companion to query_start_loc_host). 0 => that launcher falls
  // back to the D2H+sync.
  int32_t max_seq_len = 0;
};

// Arguments for vt::MlaDecodeAttention (MLA campaign W4). Mirrors the scalar
// arguments `TritonMLAImpl.forward_mqa` passes to `decode_attention_fwd`
// (vllm/v1/attention/backends/mla/triton_mla.py:242-259 @ e24d1b24).
struct MlaDecodeAttentionArgs {
  // `self.scale` — for DeepSeek this is head_dim^-0.5 TIMES the YaRN mscale^2
  // correction (mla_attention.py computes it once and hands it to the kernel as
  // a plain float; the kernel itself knows nothing about mscale). Must be > 0.
  float scale = 0.0f;
  // NUM_KV_SPLITS. 0 (the default) => the impl computes it exactly like
  // `_compute_num_kv_splits` (triton_mla.py:40-47):
  //     min(next_pow2(max(1, max_seq_len // 512)), sm_count * 2)
  // from `max_seq_len` below. 1 forces the single-split (batch-invariant)
  // reduction upstream uses under VLLM_BATCH_INVARIANT (triton_mla.py:212-213).
  int32_t num_kv_splits = 0;
  // Host-known max over `seq_lens` (CommonAttentionMetadata::max_seq_len). Only
  // used to derive `num_kv_splits` when that is 0; an upper bound is safe. When
  // both are 0 the impl falls back to 1 split.
  int32_t max_seq_len = 0;
};

// Arguments for vt::MlaPrefillAttention (MLA campaign W5). Mirrors the scalar
// arguments `FlashAttnPrefillBackend` passes to `flash_attn_varlen_func`
// (vllm/v1/attention/backends/mla/prefill/flash_attn.py:205-248 @ e24d1b24).
struct MlaPrefillAttentionArgs {
  // `self.scale` — head_dim^-0.5 TIMES the YaRN mscale^2 correction for
  // DeepSeek, handed to the kernel as a plain float (`flash_attn.py:222,245`).
  float scale = 0.0f;
  // `causal=True` for the NEW-TOKENS call (`flash_attn.py:223`), `causal=False`
  // for every CONTEXT-CHUNK call (`:246`, "Context is unmasked"). Causal here is
  // FlashAttention's BOTTOM-RIGHT alignment: query index i of a request whose
  // query length is Lq and key length is Lk sees keys j <= i + (Lk - Lq).
  bool causal = true;
  // Host-known max over the per-request query / key lengths
  // (`max_seqlen_q` / `max_seqlen_k`, `flash_attn.py:220-221,243-244`). Used for
  // GRID SIZING and the rounded dims only — the per-request geometry reads the
  // DEVICE cu_seqlens, so an UPPER BOUND is safe. 0 => the launcher falls back
  // to a small D2H + sync (op unit tests / callers without host lengths), the
  // same fallback the FA-2 paged prefill launcher uses.
  int32_t max_seqlen_q = 0;
  int32_t max_seqlen_k = 0;
};

// Router SCORING function. softmax over all E is the Qwen3.6 / DeepSeek-V2
// behavior; sigmoid (elementwise, NOT normalized across experts) is what
// DeepSeek-V3 / R1 use with `topk_method == "noaux_tc"`. Mirrors
// vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:110-117.
enum class MoeScoringFunc {
  kSoftmax,  // scores = softmax(gating_output, dim=-1)      (:111-112)
  kSigmoid,  // scores = gating_output.sigmoid()             (:113-114)
};

// MoE router top-k args (.agents/specs/moe-semantics.md §3 is the formula reference).
//
// The four fields below `renormalize` are the W3 GROUPED-TOPK (`noaux_tc`)
// extension, ported from grouped_topk_router.py:80-161 @ pin e24d1b24. They are
// ADDITIVE: every field defaults to the pre-W3 behavior, and `num_expert_group
// == 0` selects the original ungrouped softmax+top-k path VERBATIM (a separate
// kernel — the existing one is not touched), so the 27B / 35B / Coder / dense
// routers stay byte-identical.
struct MoeRouterTopKArgs {
  // Number of experts selected per token (top_k = num_experts_per_tok).
  int top_k = 0;
  // renormalize = norm_topk_prob (True for Qwen3.6, moe-semantics.md §1/§3):
  // divide the k selected softmax probs by their sum (denom>0 guard).
  bool renormalize = true;

  // --- grouped-topk (`noaux_tc`) extension ---------------------------------
  // scoring_func: softmax (V2 / Qwen) vs sigmoid (V3 / R1).
  MoeScoringFunc scoring_func = MoeScoringFunc::kSoftmax;
  // num_expert_group == config.n_group. 0 DISABLES grouping entirely (the
  // pre-W3 path). When > 0 it must divide num_experts exactly.
  int num_expert_group = 0;
  // topk_group == config.topk_group: how many expert GROUPS survive the
  // first-level mask. Must be in [1, num_expert_group] when grouping is on.
  int topk_group = 0;
  // routed_scaling_factor: a final multiply on the routing weights
  // (grouped_topk_router.py:159-160; deepseek_v2.py:288). 1.0 == no-op.
  float routed_scaling_factor = 1.0f;
};

// Kernel registration contract. Backends register one kernel per (OpId,
// DeviceType); the kernel's signature must match the alias for its op
// exactly. Register with a static_cast against the alias so signature drift
// is a compile error:
//   RegisterOp(OpId::kMatmul, DeviceType::kCPU,
//              reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
// The public op functions below validate arguments, then dispatch through
// these types. A kernel that does not support a validated dtype combination
// must throw loudly, never silently truncate.
using MatmulFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using MatmulNvfp4Fn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, float);
using ScaledFp4QuantFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, float, Fp4ScaleLayout);
using SiluMulFp4QuantFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&, float,
             Fp4ScaleLayout);
using SiluAndMulFp4QuantFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, float, Fp4ScaleLayout);
using SigmoidGateFp4QuantFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&, float,
             Fp4ScaleLayout);
using MatmulNvfp4Fp4Fn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, float);
using MatmulNvfp4CutlassFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor*, float);
using MatmulFp8CutlassFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, float);
using MatmulFp8CublasLtFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, float);
using QuantFp8StaticFn = void (*)(Queue&, Tensor&, const Tensor&, float);
using RmsNormQuantFp8Fn = void (*)(Queue&, Tensor& /*out_fp8*/, Tensor* /*out_bf16*/,
                                   const Tensor& /*x*/, const Tensor& /*weight*/,
                                   const RmsNormArgs&, Tensor* /*residual*/, float /*input_scale*/);
using RmsNormGatedQuantFp8Fn = void (*)(Queue&, Tensor& /*out_fp8*/, const Tensor& /*x*/,
                                        const Tensor& /*gate*/, const Tensor& /*weight*/,
                                        const RmsNormGatedArgs&, float /*input_scale*/);
using SwizzleBlockscaleFn = void (*)(Queue&, Tensor&, const Tensor&);
// vt::BatchedMatmul (`torch.bmm`) — same shape as MatmulFn but rank-3 and
// stride-driven; a distinct alias so registrations read unambiguously.
using BatchedMatmulFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
// vt::ConcatMlaNopeRope — out[..., :Dn] = nope, out[..., Dn:] = rope.
using ConcatMlaNopeRopeFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using MoeGroupedGemmNvfp4Fn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*, const Tensor&,
             const Tensor&, const Tensor&);
using MoeGroupedGemmBf16Fn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*, const Tensor&);
// Marlin NVFP4 W4A16 grouped-MoE GEMM (lift of vLLM moe_wna16_marlin_gemm; see
// MoeGroupedGemmNvfp4Marlin below). Scalar params travel in MoeMarlinArgs.
struct MoeMarlinArgs {
  int moe_block_size = 0;  // vLLM moe_align_block_size block (16..64, or 8)
  int top_k = 0;
  int size_m = 0;  // number of tokens (rows of `a`)
  int size_n = 0;  // output features
  int size_k = 0;  // input features (contraction; multiple of 16)
  bool mul_topk_weights = false;  // fold topk_weights into the output (down proj)
};
using MoeGroupedGemmNvfp4MarlinFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, Tensor&,
             const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MoeMarlinArgs&);
using MoeSiluMulFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
// --- Qwen3.6 elementwise "glue" ops (M0.9 forward). These replace host-side
// loops so the decode step can run entirely on-device (CUDA-graph capture).
// All math in f32; dims are inferred from the tensor shapes (no args structs).
using CastBf16Fn = void (*)(Queue&, Tensor&, const Tensor&);
using CastF32Fn = void (*)(Queue&, Tensor&, const Tensor&);
using MulColVecF32Fn = void (*)(Queue&, Tensor&, const Tensor&);
using AttnGateSplitFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&);
using SigmoidGateBf16Fn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using GdnGBetaFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            const Tensor&);
using GdnConvSplitFn = void (*)(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&);
using QkvSplitFn = void (*)(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&);
// Fused GDN post-conv prep (mirror of fla fused_gdn_prefill_post_conv):
// conv-split + q/k l2norm + g/beta gating in ONE launch. eps travels in
// L2NormArgs (the q/k l2norm eps; softplus threshold 20 baked in as in GdnGBeta).
using GdnPostConvFn = void (*)(Queue&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, const Tensor&,
                               const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                               const L2NormArgs&);
// Per-step RoPE cos|sin cache fill (fused-attn-preamble prep): cos_sin[T,rot] f32
// from positions[T] (RopeArgs.base/rotary_dim). Cols [0,rot/2)=cos, [rot/2,rot)=sin.
using RopeCosSinCacheFn = void (*)(Queue&, Tensor&, const Tensor&, const RopeArgs&);
// Fused full-attention preamble (split q|gate + gemma qk-RMSNorm + partial NeoX
// RoPE-from-cache + gate passthrough) in ONE launch. See AttnQkNormRopeGate below.
using AttnQkNormRopeGateFn =
    void (*)(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor&, const Tensor&, const RmsNormArgs&, const RopeArgs&);
using SharedExpertGateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using RmsNormFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const RmsNormArgs&, Tensor*);
using SiluAndMulFn = void (*)(Queue&, Tensor&, const Tensor&);
using GeluAndMulFn = void (*)(Queue&, Tensor&, const Tensor&);
using MulScalarFn = void (*)(Queue&, Tensor&, const Tensor&, double);
using SoftCapFn = void (*)(Queue&, Tensor&, const Tensor&, double);
using LayerNormFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor*, const Tensor*,
                             const LayerNormArgs&);
using ReluFn = void (*)(Queue&, Tensor&, const Tensor&);
using AddFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using EmbeddingFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using RopeFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const RopeArgs&);
using RopeFromCacheFn = void (*)(Queue&, Tensor&, Tensor*, const Tensor&,
                                 const Tensor&, const RopeArgs&);
using CausalConv1dFwdFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*,
                                   Tensor&, const Tensor&, const Tensor&,
                                   const CausalConv1dArgs&);
using CausalConv1dUpdateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                      const Tensor*, Tensor&, const Tensor*,
                                      const CausalConv1dArgs&);
using L2NormFn = void (*)(Queue&, Tensor&, const Tensor&, const L2NormArgs&);
using RmsNormGatedFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                const RmsNormGatedArgs&);
using GdnPrefillFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, Tensor&, const Tensor&,
                              const GdnArgs&);
using GdnDecodeFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, const Tensor&, Tensor&, const Tensor*,
                             const GdnArgs&);
using GdnPackedDecodeFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor&, const Tensor&, Tensor&, const Tensor&,
             const GdnArgs&);
using GdnStateGatherFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*);
using GdnStateScatterFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using MoeRouterTopKFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&,
                                 const MoeRouterTopKArgs&, const Tensor*);
using MoeCombineFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*);
using MoeCombineGateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&);
using AttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const AttentionArgs&);
using ReshapeAndCacheFn = void (*)(Queue&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                   const Tensor&);
using ConcatAndCacheMlaFn =
    void (*)(Queue&, const Tensor&, const Tensor&, Tensor&, const Tensor&);
using MlaDecodeAttentionFn = void (*)(Queue&, Tensor&, Tensor*, const Tensor&, const Tensor&,
                                      const Tensor&, const Tensor&,
                                      const MlaDecodeAttentionArgs&);
using MlaPrefillAttentionFn = void (*)(Queue&, Tensor&, Tensor*, const Tensor&, const Tensor&,
                                       const Tensor&, const Tensor&, const Tensor&,
                                       const MlaPrefillAttentionArgs&);
using GatherMlaCacheFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor*, int64_t);
using MergeAttnStatesFn = void (*)(Queue&, Tensor&, Tensor*, const Tensor&, const Tensor&,
                                   const Tensor&, const Tensor&, int64_t);
using PagedAttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&, const Tensor&,
                                  const PagedAttentionArgs&);
// --- V1 sampling ops (M1.7 Task 2). See the sampling-op section at the bottom.
using ApplyTemperatureFn = void (*)(Queue&, Tensor&, const Tensor&, bool);
using GreedyArgmaxFn = void (*)(Queue&, Tensor&, const Tensor&);
using ApplyTopKTopPFn = void (*)(Queue&, Tensor&, const Tensor*, const Tensor*);
using ComputeProbsFn = void (*)(Queue&, Tensor&, const Tensor&);
using ComputeLogprobsFn = void (*)(Queue&, Tensor&, const Tensor&);
using RandomSampleFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
// --- V1 penalty / mask / builtin-proc ops (M1.7 Task 3). See the section at the
// bottom of this header for the full contracts.
using ApplyPenaltiesFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&, const Tensor&);
using ApplyMinPFn = void (*)(Queue&, Tensor&, const Tensor&);
using ApplyLogitBiasFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&);
using ApplyTokenMaskFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using ApplyAllowedTokenIdsFn = void (*)(Queue&, Tensor&, const Tensor&);
// --- Fused declarative recipe (TDR; the portable fusion framework). The
// registered per-backend op is the Tier-1 single-pass INTERPRETER, over the
// canonical (out, x, weight, residual) 4-operand shape it serves (kFusedAddRmsNorm
// and any future all-elementwise recipe). Tier-0 composite is device-agnostic and
// lives in ops.cpp (it walks the recipe dispatching each opcode to the standalone
// vt:: op, so it is byte-exact to the unfused sequence — see FusedChain below).
using FusedChainFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, Tensor*, const FusedRecipe&, float);
using DropinProbeFn = void (*)(Queue&, Tensor&, const Tensor&, const DropinProbeArgs&);

// `RegisterOp`, `GetOp`, `OpRegistered` and the acceleration-provider seam they
// now dispatch through are declared in vt/op_provider.h, included at the top of
// this header. Their signatures and semantics are unchanged: RegisterOp still
// installs one kernel for (OpId, DeviceType) — it is simply the priority-0
// "vt-native" provider now, and a SECOND provider (MLX on Metal, cuBLASLt on
// CUDA, llama.cpp on CPU/Vulkan) can coexist deterministically instead of
// silently overwriting it under unspecified static-init order.

template <typename Fn>
void RegisterTypedOp(OpId op, DeviceType device, Fn fn) {
  static_assert(std::is_pointer_v<Fn> && std::is_function_v<std::remove_pointer_t<Fn>>,
                "registered op must be a function pointer");
  RegisterOp(op, device, reinterpret_cast<void*>(fn));
}

template <typename Fn>
Fn GetTypedOp(OpId op, DeviceType device) {
  static_assert(std::is_pointer_v<Fn> && std::is_function_v<std::remove_pointer_t<Fn>>,
                "looked-up op must be a function pointer");
  return reinterpret_cast<Fn>(GetOp(op, device));
}

// Contract: out must not alias any input tensor (RopeNeox is in-place by design).

// out[M,N] = a[M,K] @ b[K,N]; a/b float dtypes (f32/f16/bf16), out f32 or
// bf16, f32 accumulation, all contiguous, same device.
void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// Test-only ABI probe: a tiny adapter binds Queue/Tensor metadata to a raw
// pointer/shape/stride/semantic-type/workspace/stream launcher. Production
// families migrate independently after this spine is gated.
void DropinProbe(Queue& q, Tensor& out, const Tensor& in,
                 const DropinProbeArgs& args);

// out[M,N] = a[M,K] @ b^T with b [N,K] row-major — the torch Linear weight
// orientation, K contiguous in BOTH operands (the "TN" GEMM). This is the
// layout vLLM's F.linear hits for its bf16 projections (GDN in_proj_qkvz /
// in_proj_ba, qwen3_next.py packed_modules_mapping @ e24d1b24): on GB10 it
// selects the fast `nvjet_sm121_tst_..._TNNN` cuBLASLt kernels (~1.3x the
// per-token rate of our row-major x row-major kMatmul, which cuBLASLt serves
// with slower `NNNN`/sm80-cutlass kernels — measured 2026-07-10, 27B prefill:
// in_proj site ours 2.29 vs vLLM 1.80 us/tok). Same f32 accumulation and
// dtype contract as Matmul: a/b bf16 (or f32), out f32 or bf16, contiguous,
// same device. NOT bit-identical to kMatmul on the transposed weight (the
// cuBLASLt algo — and so the K-reduction split — differs); token-exact gates
// decide call-site adoption.
void MatmulBT(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// --- Compute-in-quant GEMM (QUANT-GGUF-CIQ-GEMM) ----------------------------
// out[M,N] = a[M,K] @ b^T where the WEIGHT `b` is [N,K] row-major kept in its
// native ggml BLOCK encoding (a block `DType`), never expanded to bf16. The
// 1:1 counterpart of llama.cpp's `ggml_compute_forward_mul_mat`
// (ggml/src/ggml-cpu/ggml-cpu.c:1245-1443 @ 237ad9b96): GGUF's on-disk
// [out_features, in_features] row-major order IS ggml's src0 layout and IS
// this `[N,K]` orientation, so keep-quant needs no transpose (block rows
// cannot be transposed without requantizing).
//
// Because `b` is block-typed, its `Tensor.shape` is in ELEMENTS but its bytes
// are `RowSizeBytes(b.dtype, K)` per row; `b.stride` is not meaningful and
// `b` must be block-contiguous. K must be a whole number of blocks.
//
// Dispatch (mirrors ggml, and is why this is a separate OpId rather than a
// dtype branch inside MatmulBT): when the weight type has both a `vec_dot`
// kernel and its `vec_dot_type`'s activation quantizer, the activation is
// quantized once and each output is one integer block-dot. Until those land
// (work rows G2/G3) the CPU kernel runs the GENERIC COMPOSITE fallback —
// decode the weight row to f32 via the traits table's `to_float` and take the
// f32 dot — which is numerically the dequant-to-f32 reference the ported
// MUL_MAT tests measure the quantized path against.
void MatmulBTQuant(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// --- Batched dense GEMM (MLA campaign W6) -----------------------------------
// out[G,M,N] = a[G,M,K] @ b[G,K,N] — one independent row-major GEMM per batch
// entry. The 1:1 counterpart of `torch.bmm`, which is the primitive MLA WEIGHT
// ABSORPTION is expressed in upstream:
//   * vllm/model_executor/layers/attention/mla_attention.py:789
//       torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
//     — (N,B,P) x (N,P,L) -> (N,B,L), folding W_UK into the decode QUERY so the
//     MQA runs directly against the 576-wide cached latent;
//   * mla_attention.py:1034 (`_v_up_proj`)
//       torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
//     — (N,B,L) x (N,L,V) -> (N,B,V), un-projecting the latent-space attention
//     output back to `v_head_dim`.
// Upstream's absorption is therefore a LOAD-TIME weight transform plus these two
// batched GEMMs — not a fused kernel (the spike's §2.2 finding), which is why a
// portable port needs exactly this one new primitive.
//
// WHAT ACTUALLY RUNS UPSTREAM, per the whole-chain rule: `torch.bmm` on a CUDA
// bf16 tensor dispatches to ATen's `baddbmm_out_cuda_impl`, which for a
// non-broadcast, equal-strided batch calls cuBLAS
// `gemmStridedBatchedEx` (CUDA_R_16BF operands, CUBLAS_COMPUTE_32F). Our CUDA
// impl is the cuBLASLt strided-batched form of exactly that GEMM, reusing the
// same handle/workspace as vt::Matmul; the CPU impl is the sequential-over-K f32
// reference. No flashinfer / cutlass / TRT-LLM variant participates: the
// ROCm-only aiter fp8/fp4 bmm branches (`:766-776`, `:1024-1032`) are the only
// alternatives upstream has and they are unreachable on CUDA.
//
// STRIDE-DRIVEN on every operand. Both upstream call sites pass TRANSPOSED VIEWS
// (`mqa_q_nope = q_nope.transpose(0,1)`, `out.transpose(0,1)`), i.e. tensors
// whose batch axis is NOT the outermost storage axis, so a contiguity-only op
// would force two extra copies per layer per step. Only the innermost dimension
// must be unit-stride; `stride[0]` (batch) and `stride[1]` (row) are free.
//
// a/b share f32 or bf16; out is f32 or bf16; accumulation is f32. G/M/N may be 0
// (no-op); K == 0 zero-fills, mirroring an empty contraction.
void BatchedMatmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// --- MLA nope|rope head concatenation (MLA campaign W6) ---------------------
//   out[t, h, 0 : Dn)       = nope[t, h, :]
//   out[t, h, Dn : Dn + Dr) = rope[t, (Dr broadcast head), :]
//
// MLA has TWO places that build a head by concatenating a "nope" part with the
// decoupled rope part, and upstream implements both as a pre-allocated output
// plus direct copies rather than a `torch.cat` over an expanded non-contiguous
// tensor (which is why this is an op and not a view):
//
//   1. DECODE q — `torch.cat([q_nope, q_pe], dim=-1)`
//      (vllm/v1/attention/backends/mla/triton_mla.py:200-201) building the
//      576-wide MQA query out of the ABSORBED `ql_nope` (512, the transposed
//      output of the W_UK bmm — NON-CONTIGUOUS) and `q_pe` (64). vLLM ships a
//      dedicated csrc kernel for exactly this, `concat_mla_q`
//      (csrc/libtorch_stable/concat_mla_q.cuh `ConcatMLAQKernel`, host wrapper
//      csrc/libtorch_stable/cache_kernels.cu:1555-1600, bound at
//      torch_bindings.cpp:841,905 and reached from _custom_ops.py:2696-2708);
//      it is stride-driven on the token and head axes precisely so the
//      transposed bmm output needs no `.contiguous()`.
//   2. PREFILL k — `_concat_k_nope_k_pe` (mla_attention.py:2063-2092),
//      concatenating the materialized per-head `k_nope` (qk_nope_head_dim) with
//      the SINGLE-head `k_pe` (qk_rope_head_dim) BROADCAST across all heads.
//      Its docstring states the reason verbatim: "avoids the performance penalty
//      of torch.cat with expanded non-contiguous tensors by pre-allocating the
//      output and using direct copies".
//
// This op is the single generalization of both: stride-driven on the token and
// head axes of every operand, the nope/rope widths taken from the SHAPES rather
// than a compile-time template, and `rope.shape[1] == 1` with `out.shape[1] > 1`
// meaning the head-BROADCAST form case 2 needs. Only the innermost dimension
// must be unit-stride, exactly as upstream asserts
// (cache_kernels.cu:1572-1577). All three tensors share one f32/bf16/f16 dtype.
//
// DEVIATION (recorded, same shape as W5's MergeAttnStates note): upstream's
// kernel is 128/256-bit vectorized and instantiated only for NOPE_DIM=512 /
// rope 64 (`concat_mla_q.cuh:13,21-24,50-53`). Ours is SCALAR and width-generic
// so it serves the prefill K concat (nope 128) as well; the arithmetic is a pure
// copy either way, so the results are identical. Vectorization is a W9 concern.
void ConcatMlaNopeRope(Queue& q, Tensor& out, const Tensor& nope, const Tensor& rope);

// out[M,N] = act[M,K] @ dequant(w).T  — the modelopt W4A16_NVFP4 dequant-GEMM
// (M2.2a). The NVFP4 weight is read DIRECTLY from device memory and dequantized
// on the fly in the kernel (no host bf16 weight materialization); it is the
// drop-in equivalent of Matmul(act, DequantNvfp4ToBf16(w).T) but with the fp4
// weight kept resident on-device.
//
// The weight is a torch Linear weight [N=out_features, K=in_features] in the
// modelopt W4A16_NVFP4 layout (identical decode to
// vllm::DequantNvfp4ToBf16 — the authoritative reference):
//   weight_packed [N, K/2]  i8 bytes: two 4-bit E2M1 (fp4) codes per byte,
//                           low nibble = input elem 2i, high nibble = 2i+1;
//                           nibble bit 3 is the sign, bits 0..2 index the
//                           E2M1 magnitude LUT {0,.5,1,1.5,2,3,4,6}.
//   weight_scale  [N, K/16] i8 bytes: one IEEE fp8-e4m3fn scale per 16-elem
//                           input group (LINEAR layout, multiply not reciprocal).
//   weight_scale_2          per-tensor f32 global scale (amax/2688), multiplied.
// Group scale = f32(weight_scale[n, k/16]) * weight_scale_2 (f32), then the
// dequanted weight is ROUNDED TO BF16 before the multiply — bit-for-bit the
// value DequantNvfp4ToBf16 stores — so the two paths differ only in K-reduction
// order (matmul tolerance), not in the per-element product. These are IEEE
// fp8-e4m3fn scales: the GGUF killgate fork's UE4M3 x0.5 LUT trap does NOT apply.
//
// act [M,K] f32/bf16, out [M,N] f32/bf16, f32 accumulation. K must be a
// multiple of 16. CUDA only (no CPU kernel registered — the CPU reference path
// is DequantNvfp4ToBf16 + Matmul).
void MatmulNvfp4(Queue& q, Tensor& out, const Tensor& act, const Tensor& weight_packed,
                 const Tensor& weight_scale, float weight_scale_2);

// --- TRUE W4A4 (fp4 activations x fp4 weights) — the 27B path (notes §7). Mirror
// of vllm's dynamic activation quant + cutlass_scaled_fp4_mm_sm120a.
//
// ScaledFp4Quant (mirror vllm scaled_fp4_quant): dynamically per-token, per-16-
// group quantizes a bf16/f32 activation [M,K] to fp4, emitting the two streams
// the fp4xfp4 GEMM consumes:
//   out_packed [M, K/2]  i8  low-nibble-first E2M1 (a_fp4)
//   out_scale  linear [M,K/16], or CUTLASS-swizzled
//              [round_up(M,128),round_up(K/16,4)] i8 fp8-e4m3fn block scales
//              (a_scale_fp8, RAW — the GEMM folds 1/input_global_scale into
//              `alpha`). The swizzled operation initializes every padded byte
//              to zero; backends may do that in the producer body or as a
//              capture-safe pre-zero immediately before it.
// `input_global_scale_inv` is the ON-DISK activation divisor (2688/amax_act) used
// DIRECTLY. K a multiple of 16. Math = vllm cvt_warp_fp16_to_fp4 (notes §7.2) /
// the CPU vllm::RefScaledFp4Quant. CPU + CUDA.
void ScaledFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& x,
                    float input_global_scale_inv,
                    Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// SiluMulFp4Quant (mirror vllm ActivationQuantFusionPass / silu_and_mul_nvfp4_quant):
// FUSES silu(gate)*up with the NVFP4 activation quant into one kernel, removing the
// bf16 intermediate that the unfused MoeSiluMul(->bf16 [M,I]) + ScaledFp4Quant path
// writes+reads (a memory-traffic win on the prefill). gate/up are [M,I] (our
// two-input MoeSiluMul form). Outputs match ScaledFp4Quant's selected linear or
// CUTLASS-swizzled scale-layout contract:
//   out_packed [M, I/2] i8
// BIT-IDENTICAL to MoeSiluMul(gate,up -> bf16) then ScaledFp4Quant(bf16): the
// silu*up value is rounded through bf16 before quant. I a multiple of 16. CPU+CUDA
// (CPU fallback = the composite sequence).
void SiluMulFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& gate,
                     const Tensor& up, float input_global_scale_inv,
                     Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// SiluAndMulFp4Quant is the exact one-input vLLM custom-op form. `gate_up` is
// contiguous [M,2I], with gate then up along the inner dimension. It fuses the
// BF16 SiluAndMul rounding boundary with ScaledFp4Quant and emits the same
// packed/scale streams as SiluMulFp4Quant, without materializing [M,I]. CPU +
// CUDA; FP16 is tracked as the separate NVFP4 W4 breadth leaf.
void SiluAndMulFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale,
                        const Tensor& gate_up, float input_global_scale_inv,
                        Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// SigmoidGateFp4Quant (mirror vllm Inductor triton_poi_fused_mul_scaled_fp4_quant
// _sigmoid_view): FUSES the full-attention sigmoid output gate — attn*sigmoid(gate)
// — with the NVFP4 activation quant of the o_proj into one kernel, removing the
// bf16 intermediate that the unfused SigmoidGateBf16(->bf16 [M,K]) + ScaledFp4Quant
// path writes+reads. attn is [M,K] f32 OR bf16 (the FA-2 prefill hands bf16; the
// upcast is exact), gate is [M,K] f32 (sigmoid input must not be rounded). Outputs
// match ScaledFp4Quant's selected linear or CUTLASS-swizzled scale layout:
//   out_packed [M, K/2] i8 ; out_scale linear [M, K/16] or swizzled.
// BIT-IDENTICAL to SigmoidGateBf16(attn,gate -> bf16) then ScaledFp4Quant(bf16):
// the attn*sigmoid(gate) value is rounded through bf16 before quant. K a multiple
// of 16. CPU+CUDA (CPU fallback = the composite sequence).
void SigmoidGateFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale,
                         const Tensor& attn, const Tensor& gate,
                         float input_global_scale_inv,
                         Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// MatmulNvfp4Fp4 (mirror vllm cutlass_scaled_fp4_mm / ..._sm120a; notes §7.3):
//   out[m,n] = alpha * Σ_k ( a_fp4[m,k]·a_scale_fp8[m,k/16] )
//                            · ( b_fp4[n,k]·b_scale_fp8[n,k/16] )
// a_packed [M,K/2] / a_scale [M,K/16] are ScaledFp4Quant's outputs; b_packed
// [N,K/2] / b_scale [N,K/16] are the on-disk weight_packed / weight_scale (RAW
// fp8, LINEAR — no swizzle: we own producer and consumer). `alpha` folds both
// on-disk globals: alpha = (1/input_divisor)·(1/weight_divisor). f32 accumulation;
// out [M,N] f32/bf16. This is the exact vllm::RunNvfp4Emulation numeric result.
// K a multiple of 16. CPU + CUDA.
void MatmulNvfp4Fp4(Queue& q, Tensor& out, const Tensor& a_packed, const Tensor& a_scale,
                    const Tensor& b_packed, const Tensor& b_scale, float alpha);

// SwizzleBlockscale (mirror vllm swizzle_blockscale, nvfp4_utils.py:13-53): pad a
// LINEAR fp8-e4m3 block-scale [rows, groups] to [round_up(rows,128),
// round_up(groups,4)] and block-interleave into the atom layout the cutlass
// sm120a fp4 GEMM reads (Sm1xxBlkScaledConfig::tile_atom_to_shape_SF{A,B}). Both
// tensors i8 (raw fp8 bytes). Used once per weight (B_sf, at load) and per step
// (A_sf, after ScaledFp4Quant). CUDA (+ CPU reference).
void SwizzleBlockscale(Queue& q, Tensor& out_swizzled, const Tensor& in_linear);

// MatmulNvfp4Cutlass (lift of vllm cutlass_scaled_fp4_mm_sm120a — the near-peak
// Blackwell block-scaled fp4xfp4 GEMM). Same numeric contract as MatmulNvfp4Fp4
// but the two fp8 block-scale streams MUST be pre-swizzled (SwizzleBlockscale):
//   a_sf_sw [round_up(M,128), round_up(K/16,4)], b_sf_sw [round_up(N,128), ...].
// a_packed [M,K/2], b_packed [N,K/2] raw fp4 (e2m1x2). alpha = (1/input_divisor)
// ·(1/weight_divisor). The tensor overload mirrors vLLM/FlashInfer: alpha is a
// resident contiguous CUDA f32 scalar and its pointer passes straight into the
// CUTLASS epilogue. The float overload retains host-scalar staging as an exact
// compatibility/diagnostic path. out [M,N] bf16. CUDA-only (sm120a). K,N % 32 == 0.
void MatmulNvfp4Cutlass(Queue& q, Tensor& out, const Tensor& a_packed,
                        const Tensor& a_sf_sw, const Tensor& b_packed,
                        const Tensor& b_sf_sw, const Tensor& alpha);
void MatmulNvfp4Cutlass(Queue& q, Tensor& out, const Tensor& a_packed, const Tensor& a_sf_sw,
                        const Tensor& b_packed, const Tensor& b_sf_sw, float alpha);

// --- Per-tensor FP8 W8A8 (the 35B attn q/k/v/o + GDN in_proj_qkv/z/out_proj).
// Mirror of vLLM's static-scale fp8 linear: static per-tensor activation quant +
// cutlass_scaled_mm_sm120_fp8. The checkpoint stores weight F8_E4M3 + a f32
// per-tensor weight_scale + a f32 per-tensor input_scale (both applied directly:
// dequant(w)=f8(w)*weight_scale, dequant(a)=f8(a)*input_scale).

// QuantFp8Static (mirror vLLM static_scaled_fp8_quant, is_scale_inverted=False):
//   out_fp8[i] = fp8_e4m3( clamp(x[i] / input_scale, -448, 448) )   // RNE hw cvt
// Static per-tensor scale (NOT dynamic/per-token). x [M,K] f32/bf16, out [M,K]
// i8 (raw fp8-e4m3fn bytes). CUDA only (the 35B W8A8 path is CUDA-resident).
void QuantFp8Static(Queue& q, Tensor& out_fp8, const Tensor& x, float input_scale);

// RmsNormQuantFp8 (fused fp8 RMSNorm -> static per-tensor activation quant). One
// HBM pass mirrors vLLM's Inductor `fused_add_rms_norm_static_fp8_quant`
// (vllm/compilation/passes/fusion/rms_quant_fusion.py:124) — the RMSNorm producer
// that directly emits the fp8 activation so the standalone QuantFp8Static pass (and
// its bf16 round-trip) disappears. The activation is quantized ONCE and shared by
// every projection that reads it (the fp8 analog of the fp4 quantize-once), so
// callers feed the single fp8 to all shared GEMMs (see MatmulFp8CutlassPreQuantD).
//   res += x  (rounded to res dtype, as fused_add_rms_norm);
//   n = rmsnorm(res, weight)  (gemma -> weight applied as 1+w);
//   b = bf16(n);  out_fp8[i] = fp8_e4m3( b * (1/input_scale) )   // RNE hw cvt
// BIT-IDENTICAL to RmsNorm(bf16 out, x, weight, {eps,gemma}, res) followed by
// QuantFp8Static(out_fp8, that bf16, input_scale) — the bf16-intermediate form the
// current path already rounds through (the RMSNorm output IS bf16 before the quant).
// out_fp8 [T,H] i8 (raw fp8-e4m3fn bytes). out_bf16 optional [T,H] bf16 (the normed
// activation, emitted only when a bf16 consumer of it coexists at the site — e.g.
// the GDN in_proj_a/b; nullptr for full-attn q/k/v where nothing reads it). x [T,H]
// / weight [H] float; residual optional [T,H] f32/bf16 (in/out). CUDA + CPU.
void RmsNormQuantFp8(Queue& q, Tensor& out_fp8, Tensor* out_bf16, const Tensor& x,
                     const Tensor& weight, const RmsNormArgs& args, Tensor* residual,
                     float input_scale);

// RmsNormGatedQuantFp8 (fused gated-RMSNorm -> static per-tensor activation quant).
// The gated-norm analog of RmsNormQuantFp8: the GDN gated-RMSNorm producer emits the
// fp8 activation DIRECTLY, so the standalone QuantFp8Static pass (and the bf16
// round-trip that the gated norm otherwise writes then the quant re-reads) disappear.
// Mirrors vLLM's Inductor fusion of the gated-RMSNorm epilogue with the fp8 activation
// quant of the following RowParallelLinear (fla layernorm_guard.py RMSNormGated ->
// out_proj W8A8), the gated sibling of rms_quant_fusion.py's static-fp8 fusion.
//   var = mean(x^2 over last dim);  n = x * (1/sqrt(var+eps)) * w * act(z);
//   b = bf16(n);  out_fp8[i] = fp8_e4m3( b * (1/input_scale) )   // RNE hw cvt
// BIT-IDENTICAL to RmsNormGated(bf16 out, x, gate, weight, args) followed by
// QuantFp8Static(out_fp8, that bf16, input_scale): the fp8 is taken from the SAME
// bf16-rounded value the split path already rounds through, and the variance reduction
// ORDER is the shipped RmsNormGated order (fast d==128 path bit-identical to shipped).
// out_fp8 same shape as x (rank-2 [rows,D] or rank-3 [T,Hv,D]) i8 (raw fp8-e4m3fn
// bytes). x/weight float; gate may carry a padded outer (token) stride. CUDA + CPU.
void RmsNormGatedQuantFp8(Queue& q, Tensor& out_fp8, const Tensor& x, const Tensor& gate,
                          const Tensor& weight, const RmsNormGatedArgs& args, float input_scale);

// MatmulFp8Cutlass (lift of vLLM cutlass_scaled_mm_sm120_fp8 — the per-tensor
// W8A8 fp8 GEMM vLLM selects on sm120/GB10). Same math as vLLM's ScaledEpilogue
//   out[M,N] = scale_a * (scale_b * (A_fp8 @ B_fp8^T))
// but the two PER-TENSOR static scales are folded into one scalar
//   alpha = input_scale * weight_scale
// (identical for per-tensor scalars; a single fused f32 multiply vs vLLM's
// sequential scale_a·(scale_b·acc) — within fp8 tolerance, ported deviation).
// a_fp8 [M,K] (= QuantFp8Static output), b_fp8 [N,K] the on-disk raw fp8-e4m3fn
// weight (K contiguous). out [M,N] bf16 (cutlass epilogue) or f32 (via cast).
// K,N multiples of 16 (128-bit fp8 alignment). CUDA-only (sm120a).
void MatmulFp8Cutlass(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                      float alpha);

// MatmulFp8CublasLt — cuBLASLt FP8 (e4m3) dense GEMM, the native equivalent of
// vLLM's cuBLASLt fp8 path (the `nvjet_sm121_qqtst_*` / `qq*` kernels that
// torch._scaled_mm / cublasLt select for the 35B's fp8 dense projections on
// GB10/sm_121a). Same host surface + same math as MatmulFp8Cutlass — both fold
// the two per-tensor static scales into one f32 alpha (= input_scale*weight_
// scale) applied to the fp32 accumulator:
//   out[M,N] = alpha * (A_fp8[M,K] @ B_fp8[N,K]^T)
// but routed through cublasLtMatmul (CUBLAS_COMPUTE_32F, e4m3 A/B, f32 scale)
// instead of the cutlass sm120 kernel. cuBLASLt fp8 requires the TN layout
// (contraction K contiguous for both operands) — a_fp8 [M,K] and b_fp8 [N,K]
// row-major already satisfy it. a_fp8 [M,K] (= QuantFp8Static output), b_fp8
// [N,K] the on-disk raw fp8-e4m3fn weight (K contiguous). out [M,N] bf16 or f32
// (cublasLt writes the requested output type directly). K,N multiples of 16.
// CUDA-only. Falls back to the cutlass fp8 GEMM if cublasLt has no fp8 heuristic
// for a given shape (keeps the correctness gate robust on odd M).
void MatmulFp8CublasLt(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                       float alpha);

// --- Fused MoE grouped NVFP4 GEMM (M2.4). One kernel launch computes the expert
// projection for ALL (token, activated-expert) pairs at once, instead of the
// per-expert loop of tiny MatmulNvfp4 launches (the launch-overhead-bound decode
// bottleneck). For each output row p (a (token,slot) pair):
//   out[p, :] = act[row_map ? row_map[p] : p, :] @ dequant(W[expert_ids[p]]).T
// where W[e] is the modelopt W4A16_NVFP4 weight [N=out, K=in] of expert e (same
// on-the-fly decode as vt::MatmulNvfp4, bit-for-bit vllm::DequantNvfp4ToBf16).
// The E per-expert weights are passed as DEVICE POINTER ARRAYS (the fp4-resident
// buffers, M2.2b) indexed by expert id — no weight gather/copy:
//   out          [P, N] f32/bf16 (P = num (token,expert) pairs)
//   act          [R, K] f32/bf16 (R rows; gate/up read the token hidden [T,H],
//                down reads the per-pair silu output [P,I])
//   expert_ids   [P] i32  (device; the router's top-k indices, [T,top_k] viewed
//                as [P] — pair p = token p/top_k, slot p%top_k)
//   row_map      [P] i32  (device) or nullptr: act row for output row p (nullptr
//                => identity p; gate/up pass token-of-pair = p/top_k)
//   packed_ptrs  [E] i64  (device) each entry = (uintptr_t) of expert e's packed
//                [N,K/2] i8 buffer
//   scale_ptrs   [E] i64  (device) each entry = (uintptr_t) of expert e's scale
//                [N,K/16] i8 buffer
//   scale2s      [E] f32  (device) per-expert weight_scale_2
// K = act.shape[1] (multiple of 16), N = out.shape[1], P = out.shape[0]. f32
// accumulation, per-row bit-identical to the per-expert MatmulNvfp4. CUDA only
// (the CPU MoE path keeps the per-expert dequant+matmul reference).
void MoeGroupedGemmNvfp4(Queue& q, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                         const Tensor* row_map, const Tensor& packed_ptrs,
                         const Tensor& scale_ptrs, const Tensor& scale2s);

// BF16 grouped-MoE GEMM (the fast bf16 MoE path, Qwen3-Coder Qwen3MoeForCausalLM).
// out[p, n] = sum_k act[row(p), k] * W_e[k, n], where e = expert_ids[p], W_e =
// weight_ptrs[e] is a bf16 [K, N] (Matmul-B / loader-transposed) weight, and
// row(p) = row_map ? row_map[p] : p. Structurally identical scheduling to
// MoeGroupedGemmNvfp4 (naive one-thread-per-output for small/decode P; expert-
// counting-sort + bf16 WMMA tensor-core tiles for large/prefill P) — the fp4
// on-the-fly decode is replaced by a direct bf16 weight read. f32 accumulation,
// out f32 (gate/up, matching the reference MatmulF32) or bf16 (down). CUDA only
// (the CPU/GGUF MoE path keeps the per-expert MatmulBf16 reference).
//   act          [*, K] bf16 (row(p) selects the source row)
//   expert_ids   [P] i32 (device) — per-pair expert id (router top-k, viewed [P])
//   row_map      [P] i32 (device) or nullptr — pair p -> source act row
//   weight_ptrs  [E] i64 (device) — each entry = (uintptr_t) of expert e's bf16
//                [K, N] weight buffer
// K = act.shape[1], N = out.shape[1], P = out.shape[0].
void MoeGroupedGemmBf16(Queue& q, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                        const Tensor* row_map, const Tensor& weight_ptrs);

// MoeGroupedGemmNvfp4Marlin (lift of vLLM moe_wna16_marlin_gemm, ops.cu:543 —
// the Marlin W4A16 kernel vLLM selects for the 35B's NVFP4 MoE experts). One
// launch computes the grouped expert projection over all padded (token,expert)
// blocks. Inputs mirror the NVFP4 branch: b_type=kFE2M1f + s_type=kFE4M3fn,
// group size 16, bf16 activation/output.
//   c            [size_m*top_k, size_n] bf16 (out; the per-(token,slot) result)
//   a            [size_m, size_k]        bf16 (token hidden)
//   b_q_weight   [E, size_k/16, size_n*8/pack] i32 — Marlin-interleaved fp4
//                (from the load-time gptq_marlin_moe_repack)
//   b_scales     [E, size_k/16, size_n]  fp8-e4m3 (processed:
//                marlin_permute_scales + nvfp4_marlin_process_scales)
//   global_scale [E]                     f32 (nvfp4_marlin_process_global_scale)
//   workspace    [>= sms*4 or per-tile]  i32 (zeroed reduction locks)
//   sorted_token_ids / expert_ids / num_tokens_past_padded / topk_weights:
//                the moe_align_block_size outputs (int32 / int32 / int32 / f32).
// CUDA-only (Blackwell sm_12xa; needs the vendored Marlin TUs, VT_MARLIN_NVFP4).
void MoeGroupedGemmNvfp4Marlin(Queue& q, Tensor& c, const Tensor& a, const Tensor& b_q_weight,
                               const Tensor& b_scales, const Tensor& global_scale,
                               Tensor& workspace, const Tensor& sorted_token_ids,
                               const Tensor& expert_ids, const Tensor& num_tokens_past_padded,
                               const Tensor& topk_weights, const MoeMarlinArgs& args);

// out[R,I] = silu(gate[R,I]) * up[R,I]  (moe-semantics.md §4; the fused-MoE
// element-wise activation between the grouped gate/up and down GEMMs). gate/up
// f32 or bf16, out f32/bf16; silu/mul computed in f32, rounded on store. Unlike
// vt::SiluAndMul (single [T,2D] input), this takes the two separately-produced
// projections so no concat/copy is needed. CPU + CUDA.
void MoeSiluMul(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up);

// out[T,H] = x[T,H] / sqrt(mean(x^2) + eps) * w  (or *(1+w) when gemma);
// out f32 or bf16 (computed in f32, rounded on store).
// With residual != nullptr (f32 OR bf16 [T,H]): residual += x first (new residual
// stream), and that sum is what gets normalized (upstream fused_add_rms_norm).
// The variance/normalize accumulation is always f32; the residual load/store dtype
// follows the tensor. A bf16 residual mirrors vLLM's bf16 model dtype (the add is
// rounded to bf16 before the f32 variance, matching csrc fused_add_rms_norm); a f32
// residual keeps full precision across layers (the byte-identical previous path).
// Note: unlike upstream forward_native, the standard path keeps full f32 precision
// (no x.to(weight.dtype) rounding before the weight multiply); parity tests vs
// upstream bf16 need bf16-eps tolerance on the non-gemma path.
void RmsNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
             const RmsNormArgs& args, Tensor* residual = nullptr);

// --- Fused declarative recipe (TDR; see include/vt/recipes.h and
// .agents/specs/portable-fusion-framework.md). A recipe (a backend-agnostic
// constexpr FusedRecipe) is realized in tiers selected by VT_FUSED_TIER:
//   Tier 0 (default, composite): a device-agnostic walker (ops.cpp) dispatches
//     each opcode to the already-registered standalone vt:: op — BYTE-EXACT to the
//     unfused sequence the model hand-calls (kRmsNorm->RmsNorm, kSiluMul->
//     MoeSiluMul, kQuantFp4->ScaledFp4Quant, ...). Correct on every backend that
//     registers the constituent ops; the fp8 quant terminal is CUDA-only.
//   Tier 1 (interpreter): a single-pass backend kernel for Tier-1-able recipes
//     (all steps elementwise/kRmsNorm, e.g. kFusedAddRmsNorm); the perf tier.
//
// Runtime scalars travel in FusedParams (structural constants like gemma live in
// the recipe). Tensors are bound positionally to the recipe's indexed operand
// table via FusedBinding (op[i] is the tensor for operand slot i; nullptr for an
// absent optional slot). Intermediate slots (a bf16 norm/activation result the
// next step quantizes) are caller-bound scratch — exactly as the unfused sequence
// materializes them.

// Physical tensors bound to a recipe's indexed operand table (op[i] == slot i).
struct FusedBinding {
  Tensor* op[kMaxFusedOperands] = {};
  int n = 0;
};

// Runtime scalars a recipe's opcodes consume (structural constants stay in the
// recipe). eps: RMSNorm/gated epsilon. quant_scale: the fp8 input_scale OR the
// fp4 input_global_scale_inv. fp4_layout: ScaledFp4Quant scale layout. rope: the
// kRope / kAttnQkNormRopeGate RoPE args.
struct FusedParams {
  float eps = 1e-6f;
  float quant_scale = 1.0f;
  Fp4ScaleLayout fp4_layout = Fp4ScaleLayout::kLinear;
  RopeArgs rope{};
};

// General entry: realize `recipe` over the bound operands with `params`.
//
// Realization order (W2): (1) if the recipe carries a fast_op (a bespoke single-
// launch fused kernel, e.g. kSiluMulFp4Quant) AND that OpId is registered on the
// device, dispatch to it — the SAME fast kernel the model called directly before
// migration, so the migration is perf-neutral by construction; (2) else, for a
// Tier-1-able recipe with VT_FUSED_TIER=1, the interpreter kernel; (3) else the
// Tier-0 composite. Every tier is byte-exact to the others per the §5 discipline.
void FusedChain(Queue& q, const FusedRecipe& recipe, const FusedBinding& binding,
                const FusedParams& params);

// Force the Tier-0 composite realization (the standalone-op-sequence oracle),
// bypassing any fast_op / interpreter tier. This is the byte-exact golden the fast
// realization is validated against; the parity tests call it to assert
// fast == composite == the unfused sequence. Production code uses FusedChain.
void FusedChainComposite(Queue& q, const FusedRecipe& recipe, const FusedBinding& binding,
                         const FusedParams& params);

// Narrow overload for the canonical (out, x, weight, residual) 4-operand shape —
// the W0-adopted kFusedAddRmsNorm site. Binds [x, weight, residual, out] and
// forwards to the general entry; bit-identical to RmsNorm(out, x, weight,
// {eps, gemma=true}, residual) for kFusedAddRmsNorm.
void FusedChain(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, Tensor* residual,
                const FusedRecipe& recipe, float eps);

// --- W2 convenience overloads: keep each migrated model call site a SINGLE
// FusedChain call (the bespoke fused-op call → one declarative recipe dispatch).
// Each binds the recipe's indexed operand table positionally and forwards to the
// general entry, which dispatches to the recipe's fast_op realization. The Tier-0
// composite intermediate slot (tmp_bf16) is bound nullptr — these sites feed the
// fast realization (which never materializes it); FusedChainComposite validates
// the composite separately with caller-provided scratch.

// Fp4-activation-quant shape (kSiluMulFp4Quant, kSigmoidGateFp4Quant): two float
// inputs a,b -> out_packed[M,K/2] + out_scale. Binds [a, b, nullptr, packed, scale].
void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_packed, Tensor& out_scale,
                const Tensor& a, const Tensor& b, float quant_scale,
                Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// RmsNorm->fp8 shape (kRmsNormQuantFp8): residual-add + gemma-RMSNorm -> static
// fp8, with an optional bf16 normed output. Binds [x, weight, residual, out_bf16,
// out_fp8]; residual/out_bf16 may be nullptr (matching the bespoke op contract).
void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_fp8, Tensor* out_bf16,
                const Tensor& x, const Tensor& weight, Tensor* residual, float eps,
                float input_scale);

// Gated-RmsNorm->fp8 shape (kRmsNormGatedQuantFp8): gated-RMSNorm -> static fp8.
// Binds [x, gate, weight, nullptr, out_fp8].
void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_fp8, const Tensor& x,
                const Tensor& gate, const Tensor& weight, float eps, float input_scale);

// Fused-attention-preamble MACRO shape (kAttnQkNormRopeGate): binds the recipe's
// fixed 8-operand table [qgate, kf, q_norm, k_norm, cos_sin, q_out, k_out, gate_out]
// and forwards to the general entry. This recipe has NO fast_op — its Tier-0
// composite already dispatches the whole preamble to the single vt::AttnQkNormRopeGate
// kernel, so the migration is perf-neutral by construction (same one launch).
void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& q_out, Tensor& k_out,
                Tensor& gate_out, const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                const Tensor& k_norm, const Tensor& cos_sin, float eps, const RopeArgs& rope);

// out[T,D] = silu(x[:, :D]) * x[:, D:], x is [T, 2D]; out f32 or bf16.
// Note: computes in f32 (upstream forward_native computes in x's dtype); bf16 parity tests need bf16-eps tolerance.
void SiluAndMul(Queue& q, Tensor& out, const Tensor& x);

// out[T,D] = gelu_tanh(x[:, :D]) * x[:, D:], x is [T, 2D]; out f32 or bf16.
// gelu_tanh(a) = 0.5*a*(1 + tanh(sqrt(2/pi)*(a + 0.044715*a^3))) — the exact
// `gelu_pytorch_tanh` / F.gelu(approximate="tanh"), computed in f32 then stored.
// Mirrors vLLM GeluAndMul(approximate="tanh") (activation.py NewGELU math). The
// GeGLU analog of vt::SiluAndMul; the one genuinely-new Gemma compute kernel.
void GeluAndMul(Queue& q, Tensor& out, const Tensor& x);

// out[i] = x[i] * scalar, elementwise, computed in f32 and rounded to out's
// dtype (f32/bf16). `scalar` is a runtime double (pass the bf16-rounded value
// for a bf16-exact match of a torch bf16-scalar multiply). Powers the Gemma
// embedding normalizer `embed_tokens(ids) * sqrt(hidden_size)` (gemma3.py:341).
void MulScalar(Queue& q, Tensor& out, const Tensor& x, double scalar);

// out[i] = cap * tanh(x[i] / cap), elementwise, computed in f32 and rounded to
// out's dtype. The Gemma-2 final logit soft-cap
// (LogitsProcessor(soft_cap=final_logit_softcapping), gemma2.py:344-345): a
// monotone squashing that leaves greedy argmax invariant but is applied for
// faithfulness. `cap` must be > 0. Shapes must match (same rank/extent).
void SoftCap(Queue& q, Tensor& out, const Tensor& x, double cap);

// out[..,D] = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias — torch
// `nn.LayerNorm` over the LAST dim (ported from ATen `native_layer_norm` /
// `vectorized_layer_norm_kernel`, the kernel `nn.LayerNorm` dispatches to for a
// CUDA half/bfloat16 input). `weight`/`bias` are optional rank-1 [D] tensors:
// both null == `elementwise_affine=False`. `var` is the BIASED (1/N) variance,
// as torch uses. x/out f32 or bf16; the mean/variance accumulation, the
// normalization and the affine are all computed in f32 and rounded once on
// store — matching torch's `acc_type<bfloat16> == float` contract, so a bf16
// LayerNorm here rounds exactly where torch's does.
//
// This is the mean-subtracting, bias-carrying sibling of vt::RmsNorm. It is
// what every pre-Llama-era family (OPT, GPT-2, BLOOM, ...) normalizes with;
// vllm/model_executor/models/opt.py:146-148,164-166,248-251. CPU + CUDA.
void LayerNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
               const Tensor* bias, const LayerNormArgs& args);

// out[i] = max(x[i], 0) — torch `F.relu`, i.e. vLLM `get_act_fn("relu")`
// (vllm/model_executor/layers/activation.py) as selected by OPT's
// `config.activation_function` (opt.py:156). Computed in f32, rounded on store;
// `out` may alias `x` (in-place). x/out f32 or bf16. CPU + CUDA.
void Relu(Queue& q, Tensor& out, const Tensor& x);

// out = a + b, in two shapes:
//   ELEMENTWISE  — b has a's exact shape (OPT's `residual + hidden_states`
//                  residual joins, opt.py:178,191, and the
//                  `inputs_embeds + pos_embeds` embedding join, opt.py:279).
//   ROW-BROADCAST — b is rank-1 [D] matching a's LAST dim, applied to every row
//                  (a `nn.Linear` bias term: OPT's q/k/v/out_proj/fc1/fc2 all
//                  carry one under `config.enable_bias`, opt.py:90-104,149-163).
// Computed in f32, rounded on store; `out` may alias `a` (in-place). All of
// a/b/out f32 or bf16. CPU + CUDA.
void Add(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// out[T,H] = table[ids[t], :]; ids i32/i64, bounds-checked; out f32 or bf16.
// CUDA note (M0.6 decision): ids live on the device, so the CUDA kernel clamps
// bad ids for the gather (no OOB read), records the first bad id in a device
// flag, and the wrapper synchronizes the stream and throws before returning.
// CUDA Embedding is therefore synchronizing for now — correctness-grade;
// revisit for full async when the model runner needs it (M0.9/M2).
void Embedding(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids);

// In-place partial NeoX RoPE on q[T,Hq,D] and k[T,Hk,D], positions[T].
// q/k dtype f32 or bf16 (same dtype for both); rotation computed in f32,
// rounded back on store for bf16.
void RopeNeox(Queue& q, Tensor& q_states, Tensor& k_states, const Tensor& positions,
              const RopeArgs& args);

// In-place partial RoPE from a supplied global cos|sin cache. Mirrors pinned
// vLLM's rotary_embedding custom op (base.py:160-252; _custom_ops.py:200-225)
// and MRotaryEmbedding's 3-axis selection (mrope.py:14-187,263-375).
//
// q [T,Hq,D], optional k [T,Hk,D], and cache [P,rotary_dim] share f32 or bf16.
// positions is [T] for ordinary/text RoPE or [3,T] for MRoPE. For the latter,
// args.mrope_section must sum to rotary_dim/2; contiguous and interleaved T/H/W
// layouts use the exact pinned selection rules. args.is_neox_style selects
// half-split NeoX or adjacent-pair GPT-J rotation. Formula construction is not
// part of this op: the hot path only gathers cache values and rotates.
//
// STRIDE-DRIVEN q/k (relaxed at MLA campaign W6; positions/cache stay
// contiguous). Only the innermost dimension must be unit-stride. DeepSeek's
// DECOUPLED RoPE rotates the TRAILING qk_rope_head_dim slice of each query head
// (`q[..., qk_nope_head_dim:]`, mla.py:160-167) with `is_neox_style=False`, and
// its k_pe is the trailing column block of the single fused
// `kv_a_proj_with_mqa` output (deepseek_v2.py:511) — both are strided views, so
// a contiguity requirement would cost two copies per layer per step. For a
// CONTIGUOUS tensor the strided offsets are integer-identical to the pre-W6
// (token * heads + head) * head_dim formula, so every existing caller (the
// Qwen3 dense/MoE preamble) is bit-identical by construction.
void RopeFromCache(Queue& q, Tensor& q_states, Tensor* k_states,
                   const Tensor& positions, const Tensor& cos_sin_cache,
                   const RopeArgs& args);

// --- Fused full-attention preamble (default-OFF prefill lever; mirror of vLLM's
// fused_qk_rmsnorm_rope / fla fused_qk_norm_rope.py:95-102, which reads a
// precomputed cos_sin_cache with ZERO in-kernel transcendentals). Two ops:
//
// RopeCosSinCache: precompute the batch's cos|sin ONCE per step so the fused
// preamble kernel below does no per-element pow/cos/sin (the current RopeNeox
// recomputes them in DOUBLE per element, per head, per layer). Fills
// cos_sin[T, rotary_dim] f32 from positions[T]: for token t, pair i in
// [0, rotary_dim/2):
//   freq  = base^(-2i/rotary_dim)            (double, matching RopeNeox)
//   angle = positions[t] * freq             (double)
//   cos_sin[t, i]              = (f32) cos(angle)
//   cos_sin[t, rotary_dim/2+i] = (f32) sin(angle)
// The double-precision angle + f32 cast reproduce RopeNeox's per-element c/sn
// BIT-FOR-BIT, so the cached rotation is token-exact vs the inline path.
// positions i32/i64; cos_sin f32 [T, rotary_dim]. CPU + CUDA.
void RopeCosSinCache(Queue& q, Tensor& cos_sin, const Tensor& positions, const RopeArgs& args);

// AttnQkNormRopeGate: the fused full-attention preamble — one launch replacing the
// AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox four-kernel chain, removing
// their f32 HBM intermediate round-trips. Grid (T, Hq+Hkv), one block per
// (token, head); the block reduces over Dh (gemma-RMSNorm), then applies partial
// NeoX RoPE reading the precomputed cos_sin cache. Bit-for-bit equal to composing
// the four ops when the outputs are f32 (the wired default): the gemma RMSNorm
// (f32 variance, weight as (1+w)) and the RoPE (x*c - y*sn / x*sn + y*c from the
// same f32 c/sn) are the identical f32 arithmetic; only launch-count and HBM
// traffic change. The output dtype is templated (f32 keeps the PagedAttention f32
// query contract; bf16 halves the writes but rounds q/k/gate — a GPU-gated A/B).
//   qgate   [T, Hq*2*Dh]  f32/bf16 — the fused q|gate projection (per head [q|gate])
//   kf      [T, Hkv*Dh]   f32/bf16 — the k projection
//   q_norm/k_norm [Dh]    f32      — the per-head gemma-RMSNorm weights
//   cos_sin [T, rot]      f32      — RopeCosSinCache output (rot = rope_args.rotary_dim)
//   q_out   [T, Hq, Dh]   f32/bf16 — gemma-RMSNorm'd + RoPE'd q
//   k_out   [T, Hkv, Dh]  f32/bf16 — gemma-RMSNorm'd + RoPE'd k
//   gate_out[T, Hq, Dh]   f32/bf16 — the raw gate half (passthrough, no norm/rope)
// norm_args: eps + gemma (must be gemma=true for Qwen). rope_args: rotary_dim (the
// partial rotation width; base is unused — the cache is precomputed). q_out/k_out
// share one dtype; gate_out matches it OR stays f32 while q/k are bf16 (the FA-2
// prefill combo: bf16 q/k feed FA-2 + the bf16 KV-cache write — each store is the
// RN round of the same f32 value, bit-identical to f32-out + CastBf16 — while
// sigmoid(gate) keeps the un-rounded f32 gate). qgate/kf share one dtype. CPU + CUDA.
void AttnQkNormRopeGate(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                        const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                        const Tensor& k_norm, const Tensor& cos_sin,
                        const RmsNormArgs& norm_args, const RopeArgs& rope_args);

// --- GDN (Gated DeltaNet) ops. Formula reference: .agents/specs/gdn-semantics.md.
// All GDN state tensors are caller-allocated f32 and updated IN PLACE
// (upstream computes states in f32 and rounds to the cache dtype on store —
// that rounding point is M0.9 layer assembly, gdn-semantics.md §1).

// Varlen causal conv over the token stream (gdn-semantics.md §2, upstream
// causal_conv1d_fn). x[T,C] token-major, weight[C,K], optional bias[C]
// (nullptr = no bias; Qwen GDN conv has bias=False), conv_state[N,C,K-1] f32
// in/out (per-sequence slices, gathered — cache_indices/NULL-block handling is
// M0.9), query_start_loc[N+1] i32 cumulative token offsets (seq s spans
// [qsl[s], qsl[s+1])), has_initial_state[N] i8/i32 (0/1; upstream bool).
//   out[c,t] = act(bias[c] + sum_j w[c,j] * window[j]), window[j] = x token
//   t-(K-1-j), falling back to conv_state (if has_initial_state) or 0 for
//   tokens before the sequence start. w[:,K-1] multiplies the current token.
// State write-back: last K-1 RAW x tokens (pre-activation), left-padded with
// zeros (no init state) or shifted old state when T < K-1.
// x [T,C] may be a padded-row (inner-contiguous, outer stride >= C) view — the
// merged qkvz projection feeds mixed_qkv = mixed_qkvz[:, :conv_dim] without a
// copy; out/weight/conv_state stay contiguous.
void CausalConv1dFwd(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, Tensor& conv_state, const Tensor& query_start_loc,
                     const Tensor& has_initial_state, const CausalConv1dArgs& args);

// Single-token conv step (gdn-semantics.md §3, upstream causal_conv1d_update
// seqlen==1 path). x[B,C] one token per sequence, conv_state[B,C,K-1] f32
// in/out. Read-old-then-roll:
//   out[c] = act(bias[c] + sum_j w[c,j] * [conv_state[c,:], x[c]][j])
//   conv_state[c,:] <- [conv_state[c,1:], x[c]]   (raw x)
// conv_state_indices (optional; mirrors mamba causal_conv1d_update conv_state_indices /
// cache_indices): when non-null, row bt reads/writes the persistent cache slot
// conv_state_indices[bt] (conv_state is then the FULL [num_slots,C,K-1] cache), so the
// caller need not gather/scatter per-request rows. When null, conv_state is compact
// [batch,C,K-1] and row == bt. x [B,C] may be a padded-row (inner-contiguous)
// view of the merged qkvz output; out stays contiguous.
void CausalConv1dUpdate(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                        const Tensor* bias, Tensor& conv_state, const CausalConv1dArgs& args,
                        const Tensor* conv_state_indices = nullptr);

// Rowwise l2 normalization over the LAST dim (gdn-semantics.md §4, upstream
// l2norm_fwd): y = x * rsqrt(sum(x^2) + eps). Plain SUM, not mean — this is
// not an rmsnorm. x/out rank 2 or 3 ([rows, D] or [T, H, D]); f32 math.
void L2Norm(Queue& q, Tensor& out, const Tensor& x, const L2NormArgs& args);

// Gated rmsnorm (gdn-semantics.md §5, upstream RMSNormGated with
// norm_before_gate=True, group_size=None, no bias):
//   var = mean(x^2 over last dim);  out = x * rsqrt(var + eps) * w * act(z)
// x/gate/out rank-2 [rows,D] or rank-3 [T,Hv,D], weight [D]; normalization is
// over the LAST dim either way; act = silu (or sigmoid, args.sigmoid_gate).
// x/out stay contiguous; the gate may carry a padded outer (token) stride with
// contiguous inner dims — the merged qkvz projection's z = mixed_qkvz[:,
// conv_dim:] slice viewed as [T,Hv,Dv] (qwen_gdn_linear_attn.py:929-936).
void RmsNormGated(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                  const Tensor& weight, const RmsNormGatedArgs& args);

// Gated-delta-rule recurrence over varlen prefill sequences
// (gdn-semantics.md §7, upstream fused_recurrent_gated_delta_rule — the
// pinned sequential statement of chunk_gated_delta_rule). q_in/k[T,Hk,Dk]
// ALREADY l2-normalized (upstream prefill normalizes in fused_post_conv_prep,
// §4), v[T,Hv,Dv], g/beta[T,Hv] f32 (log-space decay / sigmoid(b), derived
// upstream per §6), state[N,Hv,Dv,Dk] f32 in/out (zeros for fresh sequences),
// query_start_loc[N+1] i32. GQA broadcast: v-head hv reads q/k head
// hv / (Hv/Hk); Hv must be a multiple of Hk. Per token:
//   q' = q * scale;  S *= exp(g[hv]);  v' = (v - S @ k) * beta[hv];
//   S += outer(v', k);  out = S @ q'
// k is NOT scaled. All arithmetic f32.
void GdnPrefill(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                const Tensor& g, const Tensor& beta, Tensor& state,
                const Tensor& query_start_loc, const GdnArgs& args);

// Single-token gated-delta-rule step, one token per sequence
// (gdn-semantics.md §7 decode path). Same math as GdnPrefill with T == B and
// state[B,Hv,Dv,Dk] row b for token b. q_in/k must be l2-normalized by the
// caller (vt::L2Norm) — upstream fuses the l2norm and the §6 g/beta gating
// into the decode kernel; the decomposition is exact (gdn-semantics.md §4,
// §9). g/beta derivation from raw a/b/A_log/dt_bias is M0.9.
// state_idx (optional; mirrors fla fused_recurrent_gated_delta_rule ssm_state_indices):
// when non-null, row bt reads/writes the persistent cache slot state_idx[bt] (state is
// then the FULL [num_slots,Hv,Dv,Dk] cache), so the caller need not gather/scatter
// per-request state rows. When null, state is compact [batch,Hv,Dv,Dk] and row == bt.
void GdnDecode(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
               const Tensor& g, const Tensor& beta, Tensor& state, const GdnArgs& args,
               const Tensor* state_idx = nullptr);

// Pure non-spec packed decode, ported from vLLM v0.25.0
// vllm/model_executor/layers/fla/ops/fused_recurrent.py:255-478 @ 702f4814.
// mixed_qkv [B, 2*Hk*Dk + Hv*Dv] and a/b [B,Hv] are last-dimension
// contiguous and may have padded outer row strides. A_log/dt_bias [Hv], out
// [B,Hv,Dv], state [slots,Hv,Dv,Dk], state_idx [B]. mixed_qkv/a/b/out share
// one FP16/BF16/F32 activation dtype. State has an independent floating cache
// dtype, and A_log/dt_bias may independently use any floating dtype; this is
// the real Qwen3.6 contract (BF16 activations/output, FP32 SSM state, FP32
// A_log, BF16 dt_bias). The op normalizes raw q/k in F32, rounds sigmoid(b)
// through b.dtype, applies args.scale to q, and updates state in place. Local
// cache ABI: state_idx < 0
// zeros/skips the row; slot 0 is valid. Live CUDA indices are engine metadata
// and must be unique and in range before upload; the kernel also bounds-checks
// every slot without adding a capture-breaking host synchronization. CPU calls
// validate both properties directly.
void GdnPackedDecode(Queue& q, Tensor& out, const Tensor& mixed_qkv,
                     const Tensor& a, const Tensor& b, const Tensor& a_log,
                     const Tensor& dt_bias, Tensor& state,
                     const Tensor& state_idx, const GdnArgs& args);

// Indexed persistent-state cache boundary used by GDN mixed prefill. `cache`
// is [num_slots,...] f32 or bf16, `state_idx` is i32 [N], and `working` is the
// compact f32 [N,...] state consumed by CausalConv1dFwd/GdnPrefill. Gather
// fuses cache indexing + BF16->F32 conversion in one launch; optional
// has_initial_state (i8 or i32 [N]) zeros fresh rows while gathering. Scatter
// performs the inverse indexed F32->cache-dtype store in one launch. Cache rows
// not named by state_idx are untouched.
void GdnStateGather(Queue& q, Tensor& working, const Tensor& cache,
                    const Tensor& state_idx,
                    const Tensor* has_initial_state = nullptr);
void GdnStateScatter(Queue& q, Tensor& cache, const Tensor& working,
                     const Tensor& state_idx);

// --- MoE (sparse mixture-of-experts) ops. Formula reference:
// .agents/specs/moe-semantics.md. The expert MLP itself is NOT an op — it is composed
// in the layer/runner from Matmul + SiluAndMul (§4). These two ops cover the
// pieces composition cannot express: the router top-k/normalize and the
// weighted scatter-combine.

// Router top-k (moe-semantics.md §3, upstream topk_softmax + fused_topk).
// logits [T,E] any float dtype; softmax computed in f32 over ALL E experts,
// greedy top-k (weights emitted in descending order per token), lowest expert
// index wins ties, then optional renormalize (divide the k probs by their sum,
// denom<=0 -> 1 guard). weights [T,top_k] f32, indices [T,top_k] i32.
//
// --- GROUPED-TOPK / `noaux_tc` (W3) ----------------------------------------
// When `args.num_expert_group > 0` this instead runs the DeepSeek grouped-topk
// router, a 1:1 port of
// vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:80-161 @
// pin e24d1b24 (the `native_impl`/`forward_native` path — the fully fused
// `ops.grouped_topk` CUDA path at `:28-70` is an OPTIMIZATION of the same
// formula, gated on `VLLM_USE_FUSED_MOE_GROUPED_TOPK` + sigmoid + a bias, and is
// not a different result):
//   1. scores = softmax(logits, -1) | sigmoid(logits)          (:110-117)
//   2. with a bias: original_scores = scores; scores += bias    (:120-124)
//        group_score[g] = SUM of the TOP-2 scores in group g    (:124-126)
//      without a bias:
//        group_score[g] = MAX score in group g                  (:128-131)
//   3. keep the top `topk_group` groups, mask the rest to -inf  (:133-145)
//   4. top-k over the masked scores; with a bias the WEIGHT is read from the
//      UNBIASED `original_scores` at the selected ids — the biased score selects,
//      the unbiased score weights                               (:147-150)
//   5. optional renormalize (:156-157), then routed_scaling_factor (:159-160).
//
// `e_score_correction_bias` [E] f32 is upstream's optional learned gate bias
// (deepseek_v2.py:313-318 — created ONLY for `topk_method == "noaux_tc"`, hence
// nullptr for DeepSeek-V2-Lite and for every Qwen model). Passing it with
// `num_expert_group == 0` is rejected: upstream only ever reaches the bias
// asymmetry through the grouped path.
//
// DETERMINISM DEVIATION (recorded): upstream uses `torch.topk`, whose tie order
// is unspecified (`sorted=` is only forced under VLLM_BATCH_INVARIANT). We keep
// our house convention — greedy selection with a strict `>` scan over ascending
// index, so the LOWEST expert index wins an exact tie — for BOTH the group
// selection and the expert selection. This is the same rule the ungrouped router
// already uses, and it is what makes CPU and CUDA agree bit-for-bit.
void MoeRouterTopK(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                   const MoeRouterTopKArgs& args,
                   const Tensor* e_score_correction_bias = nullptr);

// Weighted scatter-combine of the per-expert outputs (moe-semantics.md §4/§6).
//   out[t,:] = sum_j weights[t,j] * expert_out[t,j,:]   (f32 accumulation)
//              + shared[t,:]                            (when shared != nullptr)
// expert_out [T,K,H] any float dtype (the K per-slot expert MLP outputs for
// token t), weights [T,K] f32 (router weights, §3), optional shared [T,H] any
// float dtype (the shared-expert term, §5). out [T,H] f32 or bf16. The routed
// f32 sum is stored at out's dtype; the shared term is added in that same store
// (§6 combine order: shared_output + routed_output). The activation-dtype
// rounding of the routed sum before the shared add is carried by the caller
// materializing expert_out/shared in the activation dtype.
void MoeCombine(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                const Tensor* shared = nullptr);

// --- Fused MoE combine + shared-expert gate (MoE glue fusion). Equivalent to
// SharedExpertGate(shared=bf16(sigmoid(gl)*sd)) followed by MoeCombine(...,shared),
// but in ONE launch: the shared term is gated inline (sigmoid(gl[t])*sd[t,c],
// rounded through bf16 exactly as SharedExpertGate's store, then re-added in f32)
// and folded into the top-k weighted reduction. Removes the separate
// SharedExpertGate launch and the shared [T,H] global round-trip. Bit-identical
// to the two-kernel path. sd [T,H] f32, gl [T,1] f32.
void MoeCombineGate(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                    const Tensor& sd, const Tensor& gl);

// --- Dense causal attention (M0.9). Formula reference:
// .agents/specs/qwen36-forward-notes.md §5 (pinned Qwen3NextAttention core).
//
// Causal scaled-dot-product attention with GQA broadcast over a single packed
// sequence. query [T,Hq,D], key/value [T,Hk,D], out [T,Hq,D]; Hq a multiple of
// Hk (q-head h reads kv-head h / (Hq/Hk)). q/k arrive ALREADY qk-normed and
// RoPE'd (compose vt::RmsNorm + vt::RopeNeox upstream); v is raw. The output
// gate (sigmoid) is applied by the caller (it is elementwise on the projection
// split, not attention math). Per q-head h (kv-head g), query i:
//   s[j] = scale * (query[i,h] · key[j,g])   for j <= i (causal), else -inf
//   p    = softmax_j(s)                       (f32, max-subtracted)
//   out[i,h] = Σ_j p[j] * value[j,g]
// f32 or bf16 in, f32/bf16 out; all softmax/accumulation math in f32.
void Attention(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
               const Tensor& value, const AttentionArgs& args);

// --- Paged KV-cache write (M1.6). Semantics ported from the FlashAttention
// path of vllm/csrc/.../cache_kernels.cu::reshape_and_cache_flash @ e24d1b24;
// the NHD cache layout is the one FlashAttentionBackend::get_kv_cache_shape
// allocates (num_blocks, 2, block_size, num_kv_heads, head_size) — NOT the HND
// cpu_attn layout.
//
// Writes each new per-token K/V into the paged cache at its slot id.
//   k / v          [num_tokens, num_kv_heads, head_size]   (source rows)
//   k_cache/v_cache[num_blocks, block_size, num_kv_heads, head_size]  (the two
//                  dim-1 slices of the flash cache; written in place)
//   slot_mapping   [num_slots] i64  (num_slots <= num_tokens; the tail of k/v is
//                  CUDA-graph padding and is ignored)
// For token t with slot s = slot_mapping[t]: block = s / block_size,
// offset = s % block_size, and k[t] is copied to k_cache[block, offset, :, :]
// (all kv-heads, all head_size). A slot s < 0 skips token t (padding). The copy
// is a raw element copy (the "auto" cache path: cache dtype == k/v dtype, no
// fp8 scaling — fp8 KV cache is out of T0 scope).
void ReshapeAndCache(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                     Tensor& v_cache, const Tensor& slot_mapping);

// --- MLA paged KV-cache write (W3). The MLA counterpart of ReshapeAndCache.
// Ported 1:1 from vllm/csrc/libtorch_stable/cache_kernels.cu:401-442
// (`concat_and_cache_mla_kernel`) + its host wrapper `:842-905`
// (`concat_and_cache_mla`) @ pin e24d1b24, reached from
// vllm/_custom_ops.py:2532 and called at vllm/v1/attention/backend.py:995,1075.
//
// WHAT ACTUALLY RUNS UPSTREAM (verified, not assumed — AGENTS.md "ground every
// check in the whole execution chain"): unlike the GEMM/attention families,
// this one is NOT delegated to a dependency. `concat_and_cache_mla` binds
// straight to `torch.ops._C_cache_ops.concat_and_cache_mla`
// (`_custom_ops.py:2540-2542`), registered from vLLM's OWN
// csrc/libtorch_stable/torch_bindings.cpp over the kernel above. There is no
// flashinfer / cutlass / TRT-LLM variant in the dense-bf16 path — the only
// alternative in that TU is `concat_and_cache_ds_mla_kernel` (`:445+`), which is
// the fp8_ds_mla 656-byte V3.2 layout, out of campaign scope. The one thing that
// can displace it is the COMPILE-TIME fusion pass
// vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py:40, which folds
// RoPE into `concat_and_cache_mla_rope_fused` (`_custom_ops.py:2545`) — same
// math, one launch; our fusion-catalog analogue is deferred to W9, exactly as
// the campaign spec sequences it (unfused byte-exact first).
//
// MLA caches ONE row per token: the compressed latent CONCATENATED with the
// decoupled rope part, `kv_lora_rank + qk_rope_head_dim` wide (512 + 64 == 576
// for every DeepSeek variant and for Kimi Linear's MLA layers), with
// num_kv_heads == 1 and NO separate V — V is reconstructed from the same latent
// via W_UV at decode. That is why the cache is 3-D
// (num_blocks, block_size, head_size; mla_attention.py:1216-1224) and why
// ReshapeAndCache's (k, v, k_cache, v_cache) signature cannot express this write.
//
//   kv_c         [num_tokens, kv_lora_rank]        (post-`kv_a_layernorm` latent)
//   k_pe         [num_tokens, qk_rope_head_dim]    (the shared single-head rope part)
//   kv_cache     [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
//   slot_mapping [num_slots] i64  (num_slots <= num_tokens; the tail of kv_c/k_pe
//                is CUDA-graph padding and is ignored — upstream uses
//                slot_mapping.size(0) as the token count, `:855-863`)
// For token t with slot s = slot_mapping[t]: block = s / block_size,
// offset = s % block_size; kv_c[t] lands at columns [0, kv_lora_rank) of that
// entry and k_pe[t] at [kv_lora_rank, kv_lora_rank + pe_dim). A slot s < 0 skips
// token t (padding). Indexing is driven by the tensor STRIDES (upstream reads
// kv_cache.stride(0)/stride(1) and kv_c/k_pe.stride(0)), so a strided cache view
// or a split-projection source view is handled without a copy.
// The "auto" path only: cache dtype == kv_c dtype, no fp8 scaling — the
// `fp8_ds_mla` / int4 layouts are out of scope and are REFUSED loudly rather
// than silently mis-written.
void ConcatAndCacheMla(Queue& q, const Tensor& kv_c, const Tensor& k_pe, Tensor& kv_cache,
                       const Tensor& slot_mapping);

// --- MLA decode attention (MLA campaign W4) ---------------------------------
// The MQA decode half of Multi-head Latent Attention: every one of the Hq query
// heads attends to the SAME single-head compressed latent row in the 3-D MLA
// cache, and V is the leading `v_head_dim` slice of that same row — there is no
// V tensor. Upstream calls this `forward_mqa`
// (vllm/v1/attention/backends/mla/triton_mla.py:189-260 @ e24d1b24), which hands
// the SAME buffer to `decode_attention_fwd` twice
// (`kv_c_and_k_pe_cache` as K, `kv_c_and_k_pe_cache[..., :kv_lora_rank]` as V,
// `:236-244`) with `is_mla=True` and `layer._k_scale` used for BOTH k_scale and
// v_scale (`:256-257`). vt::PagedAttention's (k_cache, v_cache) signature cannot
// express that, which is why this is its own op.
//
//   out          [B, Hq, Dv]           (Dv == kv_lora_rank == 512 for DeepSeek)
//   lse          [B, Hq] or nullptr    (upstream `can_return_lse_for_decode`)
//   query        [B, Hq, D]            (D == kv_lora_rank + qk_rope_head_dim == 576)
//   kv_cache     [num_blocks, block_size, D]   — the 3-D MLA cache
//                (mla_attention.py:1216-1224), i.e. exactly what
//                vt::ConcatAndCacheMla writes
//   block_table  [B, max_blocks_per_seq] i32   (upstream `Req_to_tokens`)
//   seq_lens     [B] i32                       (upstream `B_Seqlen`)
//
// SEMANTICS (the numbers, independent of the split schedule) — for request b and
// head h, over key positions j in [0, seq_lens[b]) with
// entry = kv_cache[block_table[b, j / block_size], j % block_size, :]:
//     qk[j] = scale * dot(query[b,h,:], entry[:])          (the FULL D == 576)
//     p     = softmax_j(qk)                                (f32, online/streaming)
//     out[b,h,:]  = sum_j p[j] * entry[:Dv]                (V is the Dv PREFIX)
//     lse[b,h]    = log(sum_j exp(qk[j] - max)) + max
// There is NO causal mask: decode has exactly one query token per request whose
// position is seq_lens[b]-1, so the whole context is visible. `logit_cap` is 0
// on every reachable MLA path (TritonMLAImpl rejects `logits_soft_cap`,
// triton_mla.py:165-171) and is therefore not ported.
//
// TWO-STAGE SPLIT-KV (the CUDA impl; ported from the upstream Triton pair
// `_fwd_grouped_kernel_stage1` triton_decode_attention.py:278-458 and
// `_fwd_kernel_stage2` `:575-639`). Stage 1 partitions [0, seq_len) into
// `num_kv_splits` contiguous chunks of `cdiv(seq_len, num_kv_splits)` and emits
// one NORMALIZED partial `acc/e_sum` plus its `e_max + log(e_sum)` per split;
// stage 2 merges them with the same online-softmax rescale. A split whose
// [start, end) is empty is SKIPPED by both stages (upstream `:361`, `:610`), so
// the scratch row for it is never written and never read.
//
// DETERMINISM: stage 2 merges splits in a fixed ASCENDING split order (upstream
// `:607` `for split_kv_id in range(0, NUM_KV_SPLITS)`) — no atomicAdd anywhere,
// so the result is run-to-run bit-reproducible for a fixed `num_kv_splits`, our
// house convention. Changing `num_kv_splits` changes the f32 summation order and
// therefore may change the last bits; that is upstream's behavior too.
//
// dtypes: query/kv_cache/out share one float dtype (f32 or bf16 — upstream's
// `supported_dtypes` are fp16/bf16, `triton_mla.py:82`); all accumulation is f32.
// The fp8 KV-cache branch (`if k.dtype.is_fp8()`, `:390-391`) is out of scope and
// refused, exactly as the W3 cache write refuses `fp8_ds_mla`.
void MlaDecodeAttention(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                        const Tensor& kv_cache, const Tensor& block_table,
                        const Tensor& seq_lens, const MlaDecodeAttentionArgs& args);

// --- MLA prefill attention (MLA campaign W5) --------------------------------
// The MHA prefill half of Multi-head Latent Attention — upstream's
// "Compute Friendly Approach" (mla_attention.py:66-89): `kv_b_proj` has already
// MATERIALIZED per-head K `[total_k, H, qk_nope+qk_rope]` and V
// `[total_k, H, v_head_dim]` from the latent, so this is an ordinary varlen MHA
// with an ASYMMETRIC pair of head dims — QK 192 (128 nope + 64 rope) and V 128
// for every DeepSeek variant — over CONTIGUOUS (not paged) k/v.
//
// WHAT ACTUALLY RUNS UPSTREAM ON GB10 (OBSERVED at W0, not inferred: the oracle
// logs `Using FLASH_ATTN MLA prefill backend` on sm_121). The MLA prefill
// selector gives major != 10 the single entry `[FLASH_ATTN]`
// (vllm/v1/attention/backends/mla/prefill/selector.py:66-76) and hard-raises if
// it is unavailable (`:191-194`) — there is no fallback below FA on sm_121.
// That backend is
// vllm/v1/attention/backends/mla/prefill/flash_attn.py:40 FlashAttnPrefillBackend,
// whose two entry points are `:205 run_prefill_new_tokens` (causal, over the new
// tokens' own K/V) and `:229 run_prefill_context_chunk` (NON-causal, over one
// gathered context chunk), both funnelled through
// `:153 _flash_attn_varlen_diff_headdims` into `flash_attn_varlen_func`.
//
// V ZERO-PADDING IS UPSTREAM BEHAVIOUR, ported exactly. `requires_v_padding` is
// TRUE on GB10 (`flash_attn.py:88-99` clears it only for FA3-on-SM90 or FA4), so
// upstream pads V from v_head_dim to the QK head dim with ZEROS
// (`:164-168 torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)`)
// and slices the output back to v_head_dim afterwards (`:196-197`). We do the
// same, inside the op: the caller passes V at its true width and gets `out` at
// its true width; the padded staging buffer is an implementation detail. Zero
// padding is exact — the padded output columns are sum_j p[j]*0 == 0 — so the
// slice loses nothing.
//
//   out          [total_q, H, Dv]     (Dv == v_head_dim == 128)
//   lse          [H, total_q] f32 or nullptr — upstream's UNPADDED varlen LSE
//                layout (`Flash_fwd_params::unpadded_lse`), which is what
//                vt::MergeAttnStates consumes. `return_softmax_lse` is True for
//                every context chunk and for the new-tokens call WHEN there is
//                context to merge with (`mla_attention.py:2385`).
//   query        [total_q, H, Dqk]    (Dqk == qk_nope_head_dim + qk_rope_head_dim == 192)
//   key          [total_k, H, Dqk]
//   value        [total_k, H, Dv]
//   cu_seqlens_q [B+1] i32            (`_prefill_metadata.query_start_loc`)
//   cu_seqlens_k [B+1] i32            (the SAME tensor for the new-tokens call,
//                `chunked_context.cu_seq_lens[chunk_idx]` for a context chunk —
//                flash_attn.py:218-219 vs :241-242)
//
// H is the QUERY head count and is also the K/V head count: MLA prefill is
// genuinely multi-head on both sides (the latent has been up-projected), so
// there is no GQA grouping here — mla_attention.py:315-318 records the shape as
// K `[S, 128, 192]` / V `[S, 128, 128]` with 128 KV heads.
//
// dtypes: query/key/value/out share one float dtype. The CUDA path is bf16 only
// (the vendored FA-2 instantiations); the CPU reference also accepts f32/f16.
// The fp8-prefill branch (`use_fp8_prefill`, mla_attention.py:2360-2379) needs
// `is_device_capability_family(100)` (`mla_attention.py:1382-1385`) and is
// therefore UNREACHABLE on GB10 — not ported, refused by the dtype check.
void MlaPrefillAttention(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                         const Tensor& key, const Tensor& value, const Tensor& cu_seqlens_q,
                         const Tensor& cu_seqlens_k, const MlaPrefillAttentionArgs& args);

// --- MLA chunked-context cache gather (MLA campaign W5) ---------------------
// Ported 1:1 from vllm/csrc/libtorch_stable/cache_kernels.cu:992-1064
// (`vllm::gather_and_maybe_dequant_cache`) + its host wrapper `:1099-1157`,
// reached from vllm/_custom_ops.py and called at
// mla_attention.py:2119-2129 (`ops.gather_and_maybe_dequant_cache`) — the FIRST
// step of every chunked-context iteration.
//
// WHAT ACTUALLY RUNS UPSTREAM (whole-chain rule): like the W3 cache write this
// is vLLM's OWN csrc kernel, not a dependency. The sibling `cp_gather_cache`
// (`:1237`) is the fp8/DCP path (`mla_attention.py:2132-2139`), out of scope.
//
// It gathers, for each prefill request b, the context rows
// [seq_starts[b], seq_starts[b] + chunk_seq_lens[b]) out of the PAGED 3-D MLA
// cache into a CONTIGUOUS varlen workspace laid out by cu_seq_lens:
//
//   dst          [>= num_tokens, D]                the chunk workspace
//   src_cache    [num_blocks, block_size, D]       the 3-D MLA cache
//   block_table  [B, max_blocks] i32
//   cu_seq_lens  [B+1] i32   cumulative per-request token counts IN THIS CHUNK
//   token_to_seq [>= num_tokens] i32  back-map token index -> request (`:1015`)
//   seq_starts   [B] i32 or nullptr — the chunk's start offset within each
//                request's context (`chunked_context.starts[i]`). Upstream
//                rounds `max_context_chunk` DOWN to a multiple of page_size
//                (mla_attention.py:1687-1690) precisely because this kernel
//                indexes `(seq_starts[b] + within_chunk) / block_size` into the
//                block table; we mirror the requirement and REFUSE a
//                non-page-aligned start rather than silently mis-gather.
//   num_tokens   the chunk's total token count (`chunk_total_token[i]`)
//
// Everything is STRIDE-driven exactly as upstream (`:1150-1153` reads
// block_table.stride(0), src_cache.stride(0)/stride(1), dst.stride(0)), so a
// per-layer cache slice and a workspace slice both work copy-free. The fp8
// dequant branch (`kv_dt != kAuto`) is out of scope and refused.
void GatherMlaCache(Queue& q, Tensor& dst, const Tensor& src_cache, const Tensor& block_table,
                    const Tensor& cu_seq_lens, const Tensor& token_to_seq,
                    const Tensor* seq_starts, int64_t num_tokens);

// --- Attention-state LSE merge (MLA campaign W5) ----------------------------
// Ported 1:1 from vllm/csrc/libtorch_stable/attention/merge_attn_states.cu:18-192
// (`vllm::merge_attn_states_kernel`), reached from
// vllm/v1/attention/ops/merge_attn_states.py:9 — which selects the CUDA kernel
// on CUDA for f32/f16/bf16 with a head_size divisible by the 128-bit pack
// (`:59-77`), i.e. exactly our case, and only falls back to the Triton
// transcription otherwise. Call sites: mla_attention.py:2188-2195 (merging
// consecutive CONTEXT CHUNKS) and `:2413-2420` (merging the whole context result
// with the new-tokens result).
//
// Implements §2.2 of https://www.arxiv.org/pdf/2501.01005: two partial softmax
// attention results over DISJOINT key sets are combined by their log-sum-exps.
//   m = max(p_lse, s_lse); ps = exp(p_lse-m); ss = exp(s_lse-m); t = ps+ss
//   out = prefix_out * (ps/t) + suffix_out * (ss/t)
//   out_lse = log(t) + m
// Arithmetic is f32 regardless of the tensor dtype (upstream converts through
// `to_float`/`from_float`, `:158-171`).
//
//   output       [num_tokens, H, Dv]
//   output_lse   [H, num_tokens] f32 or nullptr
//   prefix_output/suffix_output [num_tokens, H, Dv]
//   prefix_lse/suffix_lse       [H, num_tokens] f32
//   prefill_tokens_with_context — tokens at index >= this take the SUFFIX
//     output verbatim with no merge (`:66-89`); < 0 means "all tokens have
//     context", upstream's `prefill_tokens_with_context=None`.
//
// TWO EDGE CASES PORTED VERBATIM, both of which a naive merge gets wrong:
//   * `+inf` LSE is normalized to `-inf` first (`:97-98`);
//   * when BOTH are `-inf` (a chunk in which a request has no keys at all —
//     upstream's comment at `:100-106` describes exactly the chunked-prefill
//     situation that produces it) the merge would be 0/0 => NaN, so upstream
//     emits the PREFIX output and `-inf` LSE instead. We do the same.
// The fp8-output branch (`USE_FP8_OUTPUT`) is out of scope and not ported.
void MergeAttnStates(Queue& q, Tensor& output, Tensor* output_lse, const Tensor& prefix_output,
                     const Tensor& prefix_lse, const Tensor& suffix_output,
                     const Tensor& suffix_lse, int64_t prefill_tokens_with_context);

// --- Paged attention (M1.6). Correctness-grade varlen prefill + paged decode.
// Semantics ported from the FlashAttention path
// vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl.forward @
// e24d1b24 (causal GQA softmax over the paged K/V; scale = self.scale). The
// cache read is the NHD layout FlashAttentionBackend::get_kv_cache_shape
// allocates (num_blocks, 2, block_size, num_kv_heads, head_size) — NOT cpu_attn's
// HND arithmetic — and is driven by k_cache/v_cache STRIDES (the two dim-1 unbind
// slices; block stride 2*bs*H*D). This GENERALIZES the dense M0.9 vt::Attention:
// on a single contiguous sequence the two ops agree.
//
// For each query token t (found via query_start_loc: token t belongs to request
// r with query_start_loc[r] <= t < query_start_loc[r+1]) at absolute position
// p = (seq_lens[r] - query_len_r) + (t - query_start_loc[r]) (query_len_r =
// query_start_loc[r+1] - query_start_loc[r]; seq_lens[r] is the total context
// INCLUDING this chunk), and q-head h (kv-head g = h / (Hq/Hk)):
//   for key position j in 0..p (causal), intersected with
//   [p-window.left,p+window.right] when args.window_size is present —
//   block = block_table[r, j / block_size],
//   offset = j % block_size, K = k_cache[block, offset, g, :], V likewise —
//   s[j] = scale * (query[t,h] · K);  out[t,h,:] = Σ_j softmax(s)_j * V.
// Softmax/accumulation in f32 (max-subtracted). The current step's K/V must
// already be in the cache (compose vt::ReshapeAndCache upstream), so no separate
// key/value args — the read is entirely from the paged cache.
//
//   query          [num_actual_tokens, num_q_heads, head_size]  (contiguous)
//   out            same shape/dtype as query (contiguous)
//   k_cache/v_cache[num_blocks, block_size, num_kv_heads, head_size]  (the two
//                  dim-1 slices of the flash cache; STRIDED, read-only)
//   block_table    [num_reqs, max_blocks] i32  (Task 1's block_table_tensor)
//   seq_lens       [num_reqs] i32              (context length incl. this step)
//   query_start_loc[num_reqs + 1] i32          (cumulative query offsets)
void PagedAttention(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                    const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                    const Tensor& query_start_loc, const PagedAttentionArgs& args);

// --- V1 sampling ops (M1.7 Task 2). Ported from
// vllm/v1/sample/ops/topk_topp_sampler.py + vllm/v1/sample/sampler.py @ e24d1b24.
// The Sampler pipeline (M1.7 Task 4) composes these over the model's final
// logits `[num_reqs, vocab_size]` (row-major f32). Every op is correctness-grade
// CPU + CUDA and indexes contiguous rows; greedy_argmax is the bit-exact parity
// primitive (matches torch.argmax's lowest-index tie-break).

// _SAMPLING_EPS (sampler.py:17). Greedy rows carry temperature < this; the
// temperature guard and the greedy/random where-merge both key off it.
constexpr float kSamplingEps = 1e-5f;

// apply_temperature (sampler.py::Sampler.apply_temperature). In-place per-row
// `logits[i] /= temp[i]`. When `!all_random`, greedy rows (temp < eps) would
// divide by ~0, so temp is first replaced per-row with `where(temp<eps, 1.0,
// temp)` (upstream comment: "Avoid division by zero if there are greedy
// requests"). logits [num_reqs, vocab] f32 in place; temp [num_reqs] f32.
void ApplyTemperature(Queue& q, Tensor& logits, const Tensor& temp, bool all_random);

// greedy_sample (sampler.py::Sampler.greedy_sample) = argmax(logits, dim=-1).
// LOWEST-INDEX tie-break: torch.argmax returns the FIRST occurrence of the max,
// so a strict `>` scan (keep the first max) is bit-exact vs torch on f32 logits.
// This is the M0-exit parity gate primitive. token_ids [num_reqs] i64 (torch
// argmax returns int64); logits [num_reqs, vocab] f32.
void GreedyArgmax(Queue& q, Tensor& token_ids, const Tensor& logits);

// apply_top_k_top_p (topk_topp_sampler.py::apply_top_k_top_p_pytorch, the CPU
// allow_cpu_sync path). Masks non-top-k / non-top-p logits to -inf IN PLACE.
// k [num_reqs] i32 (nullptr => skip top-k), p [num_reqs] f32 (nullptr => skip
// top-p); per-row. When p is nullptr and k given, uses the no-sort
// apply_top_k_only fast path; otherwise the sort-based path (top-k threshold =
// the k-th largest, ties at the threshold kept; top-p = smallest tail whose
// cumulative prob >= p, at-least-one). logits [num_reqs, vocab] f32.
void ApplyTopKTopP(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p);

// probs = softmax(logits, dim=-1) in f32 (forward_native's
// `logits.softmax(dim=-1, dtype=torch.float32)`; numerically stable, row-max
// subtracted). probs/logits [num_reqs, vocab] f32.
void ComputeProbs(Queue& q, Tensor& probs, const Tensor& logits);

// compute_logprobs (sampler.py::Sampler.compute_logprobs) =
// log_softmax(logits, dim=-1) in f32 (row-max subtracted, log-sum-exp).
// logprobs/logits [num_reqs, vocab] f32.
void ComputeLogprobs(Queue& q, Tensor& logprobs, const Tensor& logits);

// random_sample (topk_topp_sampler.py::random_sample +
// sample_with_exponential_noise): exponential-noise gumbel-max. For each element
// draw q ~ Exponential(1); pick `argmax_j(probs[i,j] / q[i,j])`. Per-request
// seeded RNG: `seeds[i]` is the resolved per-row seed (the Sampler picks the
// per-request override from `SamplingMetadata.generators` or the batch default —
// M1.7 Task 4). token_ids [num_reqs] i64; probs [num_reqs, vocab] f32;
// seeds [num_reqs] i64.
//
// RNG DEVIATION (recorded, T1 carry): the Exponential(1) draws come from a
// deterministic splitmix64 hash of (seed, row, vocab_index) mapped through the
// inverse-CDF q = -log(U), U in (0,1). This is distribution-correct (the
// exponential race gives P(argmax) == softmax, validated at large N) and
// reproducible under a fixed seed, but is NOT bit-exact vs torch's Philox4x32
// `exponential_()` — exact random-sampling parity vs torch is the documented
// M1.7 deferral. Greedy stays bit-exact; random is validated by algorithm +
// determinism + distribution.
void RandomSample(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds);

// --- V1 penalty / mask / builtin-proc ops (M1.7 Task 3). Ported from
// vllm/model_executor/layers/utils.py (apply_penalties), vllm/_custom_ops.py
// (apply_repetition_penalties), vllm/v1/sample/ops/bad_words.py, and
// vllm/v1/sample/logits_processor/builtin.py @ e24d1b24. The Sampler pipeline
// (M1.7 Task 4) composes these over the model's final logits [num_reqs, vocab]
// (row-major f32). Every op is correctness-grade CPU + CUDA. The higher-level
// ported entry points (src/vllm/v1/sample/ops/{penalties,bad_words}.{h,cpp},
// src/vllm/v1/sample/logits_processor/builtin.{h,cpp}) build the per-request
// derived tensors from the host SamplingMetadata and call these ops.

// apply_penalties (utils.py::apply_penalties + _custom_ops.py::
// apply_repetition_penalties_torch). Fuses the repetition penalty and the
// frequency / presence subtractions into a single elementwise pass over the
// pre-computed masks/counts. Per element (row i, col j):
//   penalty = (prompt_mask[i,j] || output_mask[i,j]) ? rep[i] : 1.0
//   logits *= (logits > 0) ? 1/penalty : penalty            (repetition)
//   logits -= frequency_penalties[i] * output_bin_counts[i,j]
//   logits -= presence_penalties[i] * output_mask[i,j]
// prompt_mask / output_mask [num_reqs, vocab] i8 (0/1), output_bin_counts
// [num_reqs, vocab] i32, frequency/presence/repetition [num_reqs] f32. In place.
void ApplyPenalties(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                    const Tensor& output_bin_counts, const Tensor& output_mask,
                    const Tensor& frequency_penalties, const Tensor& presence_penalties,
                    const Tensor& repetition_penalties);

// apply (builtin.py::MinPLogitsProcessor.apply). Per row: probs = softmax(logits),
// pmax = max_j probs; mask probs < min_p[i] * pmax to -inf. Rows with min_p[i]==0
// are unaffected (threshold 0). IS argmax-invariant (the max-prob token survives).
// logits [num_reqs, vocab] f32 in place; min_p [num_reqs] f32.
void ApplyMinP(Queue& q, Tensor& logits, const Tensor& min_p);

// apply (builtin.py::LogitBiasLogitsProcessor.apply) — the sparse scatter-add
// `logits[(rows, cols)] += biases`. rows/cols [m] i32 (flattened (req, token)
// pairs), biases [m] f32. NOT argmax-invariant. In place.
void ApplyLogitBias(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols,
                    const Tensor& biases);

// Sparse scatter of -inf at the (rows[k], cols[k]) positions — the primitive
// behind builtin.py::MinTokensLogitsProcessor.apply (index_put_ of -inf over the
// stop-token slice) AND bad_words.py::apply_bad_words (block the final n-gram
// token). rows/cols [m] i32. In place.
void ApplyTokenMask(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols);

// sampler.py:396-397 `logits.masked_fill_(allowed_token_ids_mask, -inf)`. Sets
// logits[i,j] = -inf wherever mask[i,j] != 0. The mask is TRUE for tokens to
// EXCLUDE (gpu_input_batch.py fills the row True then clears the allowed ids to
// False). mask [num_reqs, vocab] i8. In place.
void ApplyAllowedTokenIds(Queue& q, Tensor& logits, const Tensor& mask);

// --- Qwen3.6 elementwise "glue" ops (M0.9 forward). These fuse the small
// host-side reshape/split/activation loops that sit between the big ops in the
// decode forward, so the whole decode step stays on-device (CUDA-graph
// capturable). All arithmetic is f32; every dimension is inferred from the
// tensor shapes (no args structs). CPU + CUDA.

// out[i] = F32ToBF16(in[i]); out bf16, in f32, same element count. The plain
// f32 -> bf16 activation-dtype cast used before feeding a bf16-consuming op.
void CastBf16(Queue& q, Tensor& out, const Tensor& in);

// out[i] = f32(in[i]); out f32, in bf16, same element count. The bf16 -> f32
// upcast used to expose a bf16-only GEMM (Marlin) as an f32 result, matching the
// value the bf16 output rounds to (mirror of the cutlass f32-output scratch cast).
void CastF32(Queue& q, Tensor& out, const Tensor& in);

// In-place per-output-column scale: x[m,n] *= col[n], with x an F32 [M,N]
// (row-major, inner-contiguous rows; row stride may be padded) and col an F32
// [N] contiguous broadcast vector. The load-time-free realization of a merged
// per-tensor-fp8 projection's per-shard dequant: one fp8 GEMM over the
// N-concatenated weight is run with alpha=1 (raw f32 accumulation), then this
// applies each output column's folded scalar (= input_scale * that shard's
// weight_scale) in f32 — byte-identical to the separate per-shard GEMMs when the
// GEMM's accumulation matches (the alpha multiply is the same IEEE f32 op cuBLASLt
// would fold). CPU + CUDA. (Mirrors the fp4 merge's per-column block-scale
// concatenation, qwen3_5.cpp ResidentNvfp4Qkv.)
void MulColVecF32(Queue& q, Tensor& x, const Tensor& col);

// Splits the fused q/gate attention projection into its two halves. qgate is
// [T, Hq*2*Dh] contiguous, laid out per (t,hq) as [q(Dh) | gate(Dh)]; q_out and
// gate_out are [T,Hq,Dh]. For t in [0,T), hq in [0,Hq):
//   q_out[t,hq,:]    = qgate row t at offset (hq*2*Dh)      .. +Dh
//   gate_out[t,hq,:] = qgate row t at offset (hq*2*Dh + Dh) .. +2*Dh
// T/Hq/Dh are inferred from q_out's shape. All f32.
void AttnGateSplit(Queue& q, Tensor& q_out, Tensor& gate_out, const Tensor& qgate);

// out[i] = F32ToBF16(attn[i] * sigmoid(gate[i])), sigmoid(x)=1/(1+exp(-x)); out
// bf16, attn f32 OR bf16 (the FA-2 prefill path hands bf16 attention out; the
// upcast is exact so bf16-attn is bit-identical to f32-attn holding the same
// values), gate f32 (sigmoid input must not be rounded), same element count.
// The sigmoid output-gate applied to the attention result before the o_proj
// (elementwise on the projection split).
void SigmoidGateBf16(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate);

// Derives the GDN per-head decay g and gate beta from the raw projections
// (gdn-semantics.md §6). g_out/beta_out/araw/braw are [T,Hv]; a_log/dt_bias are
// [Hv]. For t in [0,T), hv in [0,Hv), idx=t*Hv+hv:
//   x  = araw[idx] + dt_bias[hv]
//   sp = softplus(x) = (x > 20) ? x : log1p(exp(x))    (threshold 20)
//   g_out[idx]    = -exp(a_log[hv]) * sp
//   beta_out[idx] = sigmoid(braw[idx])
// T/Hv inferred from g_out's shape. g/beta/a_log/dt_bias are f32. araw/braw
// share either f32 or bf16 dtype and may be inner-contiguous row-strided views
// (the merged `[b,a]` projection has row stride 2*Hv); kernels upcast on load.
void GdnGBeta(Queue& q, Tensor& g_out, Tensor& beta_out, const Tensor& araw, const Tensor& braw,
              const Tensor& a_log, const Tensor& dt_bias);

// Splits the GDN mixed-qkv conv output into its q/k/v parts. conv is
// [T, conv_dim] contiguous, laid out per row as [q(key_dim) | k(key_dim) |
// v(value_dim)], conv_dim = 2*key_dim + value_dim. q_out/k_out are [T,key_dim],
// v_out is [T,value_dim]; for each row t:
//   q_out row = conv row [0, key_dim);  k_out row = conv row [key_dim, 2*key_dim);
//   v_out row = conv row [2*key_dim, 2*key_dim + value_dim)
// T = conv.shape[0]; key_dim = q_out.Numel()/T, value_dim = v_out.Numel()/T
// (rows treated row-major). All f32.
void GdnConvSplit(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& conv);

// Splits a merged QKVParallelLinear projection into its q/k/v parts with
// INDEPENDENT head dims (GQA: q_dim != k_dim, unlike the GDN GdnConvSplit which
// assumes q_dim == k_dim). qkv is [T, q_dim + k_dim + v_dim] contiguous, laid
// out per row as [q(q_dim) | k(k_dim) | v(v_dim)]; q_out [T,q_dim], k_out
// [T,k_dim], v_out [T,v_dim], all contiguous, same dtype as qkv (f32 or bf16).
// For each row t: q_out row = qkv row [0,q_dim); k_out row = qkv row
// [q_dim, q_dim+k_dim); v_out row = qkv row [q_dim+k_dim, q_dim+k_dim+v_dim).
// q_dim/k_dim/v_dim are inferred from q_out/k_out/v_out.Numel()/T. Mirrors
// vLLM's qkv_proj output split (qwen3.py Qwen3Attention: one qkv GEMM then
// .split([q_size, kv_size, kv_size], dim=-1)). CPU + CUDA.
void QkvSplit(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv);

// Fused GDN post-conv preparation (launch-count fusion; mirror of upstream
// vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py
// _fused_post_conv_kernel — "split → l2norm*2 → gating" in a single kernel,
// grid (L, H+HV)). Replaces the GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta
// four-launch chain with one launch, bit-for-bit equal to composing them:
//   [q|k|v]     = split(conv[T, 2*key_dim+value_dim])   (GdnConvSplit)
//   q_out,k_out = l2norm(q), l2norm(k) over Dk           (L2Norm, args.eps)
//   v_out       = v                                      (copy)
//   g_out,beta_out from araw/braw + a_log/dt_bias        (GdnGBeta, §6)
// q_out/k_out [T,Hk,Dk] (l2-normed), v_out [T,Hv,Dv], g_out/beta_out [T,Hv];
// conv [T, 2*Hk*Dk + Hv*Dv]; araw/braw [T,Hv]; a_log/dt_bias [Hv]. araw/braw
// share f32 or bf16 dtype and may have a padded row stride; all gate math is f32.
void GdnPostConv(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                 Tensor& beta_out, const Tensor& conv, const Tensor& araw, const Tensor& braw,
                 const Tensor& a_log, const Tensor& dt_bias, const L2NormArgs& args);

// out[t,c] = F32ToBF16(sigmoid(gl[t]) * sd[t*H+c]); out bf16 [T,H], sd f32
// [T,H], gl f32 with T elements (shape [T] or [T,1]). The shared-expert
// sigmoid gate (moe-semantics.md §5), applied per token to the shared MLP
// output. T inferred from out.shape[0], H = out.Numel()/T.
void SharedExpertGate(Queue& q, Tensor& out, const Tensor& sd, const Tensor& gl);

}  // namespace vt
