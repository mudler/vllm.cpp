// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "vt/fp8_kv.h"
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
  kCausalConv1dSpecUpdate,
  kL2Norm,
  kRmsNormGated,
  kGdnPrefill,
  kGdnDecode,
  kGdnSpecDecode,
  kGdnPackedDecode,
  kKdaGatedDeltaRule,
  kKdaChunkPrefill,
  kMoeRouterTopK,
  kMoeCombine,
  kAttention,
  kAttentionCross,
  kAttentionDenseFast,
  kAttentionDenseFlash,
  kAttentionDenseFa2,
  kDFlashBlockAttention,
  kDFlashPagedBlockAttention,
  kDFlashGroupedConv,
  kDflash2SelectorEdges,
  kDflash2PathWalk,
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
  kTopKValuesIndices,
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
  // Gate-FREE sibling: per-head standard RMSNorm(q)+RMSNorm(k)+partial NeoX RoPE,
  // the Qwen3-DENSE preamble. Backends may register it as kAttnQkNormRope's
  // fast_op; those that do not keep the byte-exact composite automatically.
  kAttnQkNormRope,
  kFusedChain,
  kRmsNormQuantFp8,
  kRmsNormGatedQuantFp8,
  kMatmulBT,
  // kMatmulBT with a BLOCK-QUANTIZED [N,K] weight kept in its native ggml
  // encoding — llama.cpp's `ggml_compute_forward_mul_mat`
  // (ggml/src/ggml-cpu/ggml-cpu.c:1245-1443 @ 237ad9b96).
  kMatmulBTQuant,
  // GROUPED keep-quant GEMM over an expert-index list: out[p,:] = act[p,:] .
  // weight[expert_ids[p]*N .. +N]. Collapses the DeepSeek-V4 MoE's 6 routed
  // experts x {gate,up,down} = 18 tiny T=1 kMatmulBTQuant matvecs/layer into 3
  // grouped kernels (fewer host launches + higher GPU occupancy). Same arithmetic
  // as the per-expert loop, so BYTE-IDENTICAL on the CPU provider (which loops the
  // kMatmulBTQuant kernel per group). Mirrors kMoeGroupedGemmBf16's expert-index
  // structure for the keep-quant (IQ2_XXS/Q2_K/Q8_0) tower.
  kMatmulBTQuantGrouped,
  // W0-only raw-signature probe for the shared drop-in adapter boundary. It is
  // not a production kernel-family migration.
  kDropinProbe,
  // In-place base/MRoPE rotation from a supplied dtype-specific global cache.
  kRopeFromCache,
  // Indexed persistent GDN cache boundary: one launch replaces the former
  // per-row copies plus a separate BF16<->F32 cast.
  kGdnStateGather,
  kGdnStateScatter,
  // Row gather/scatter over dim 0 (torch index_select / index_copy_). Additive
  // op powering the MIXED spec+non-spec GDN batch split/merge (SPEC-MTP): gather
  // the per-token conv inputs into compact spec / non-spec working buffers and
  // scatter the per-group core outputs back to their original row positions.
  kIndexSelect,
  kIndexCopy,
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
  // Elementwise GELU activations (NEW for the Qwen3-VL vision tower). Both are
  // plain (non-gated) elementwise ops, may alias in-place, f32 compute →
  // out-dtype store.
  //   kGeluTanh — `gelu_pytorch_tanh` / F.gelu(approximate="tanh"): the vision
  //               MLP act_fn (qwen3_vl.py::Qwen3_VisionMLP, hidden_act
  //               "gelu_pytorch_tanh"). Same constant as kGeluAndMul's gate.
  //   kGeluErf  — exact erf GELU / nn.GELU(): the patch-merger act_fn
  //               (qwen3_vl.py::Qwen3_VisionPatchMerger, self.act_fn=nn.GELU()).
  kGeluTanh,
  kGeluErf,
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
  // Greedy speculative-decoding rejection sampling over the EXPANDED verify
  // logits `[Σ(1+k_i), vocab]` — the accept-iff-draft-equals-target-argmax rule
  // plus the bonus/replacement token. Mirrors the greedy branch of vLLM's
  // `_rejection_kernel` (vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:
  // 564-585,628) + the greedy short-circuit of `_resample_kernel` (:846-861) and
  // `_insert_resampled_kernel` (:828-841). See vt::GreedyRejectionSample.
  // Additive: nothing on the non-speculative path calls it.
  kGreedyRejectionSample,
  // --- Collective transport ops (BACKEND-DISTRIBUTED-COMM W2) -----------------
  // The vt::Communicator collectives, deferred from W1 (a direct method was the
  // cleaner W1 gate; W2 routes them through OpProvider so a backend SUPPLIES the
  // transport — CPU in-process reduce, NCCL on kCUDA, MLX-ring on kMETAL). Each
  // is dispatched on the queue's DeviceType and hands the bound Communicator its
  // device-specific data plane. Mirrors DeviceCommunicatorBase.all_reduce:215 /
  // all_gather:219 / send:321 / recv:328 (base_device_communicator.py). See
  // vt::CommAllReduceFn (include/vt/communicator.h). Additive: nothing on the
  // world_size==1 single-GPU path dispatches them (the collective returns before
  // the lookup — parallel_state.py:638 bypass).
  kAllReduce,
  kAllGather,
  kSend,
  kRecv,
  // --- DeepSeek-V4-Flash device kernels (W7-device) --------------------------
  // The four NEW V4 op families' CUDA kernels (MHC Sinkhorn+pre/post+head, DSA
  // Lightning-Indexer+seams, Compressor pool+fp8_ds_mla KV, sqrtsoftplus/hash
  // router+clamped SwiGLU), registered through the OpProvider seam so
  // DeepseekV4Model::ForwardDevice can dispatch them. Each OpId's `fn` points at
  // a family kernels-struct (deepseek_v4_device.h) of typed device launchers,
  // each a 1:1 CUDA port of the landed portable HOST reference
  // (deepseek_v4_{mhc,dsa,compressor,moe}.{h,cpp}) it is unit-gated against. The
  // 512-wide MLA attention + expert grouped-GEMM REUSE the existing kernels
  // (kMlaDecodeAttention / kMoeGroupedGemmNvfp4) and are NOT re-ported. Additive:
  // nothing outside the V4 device forward dispatches them.
  kDeepseekV4Mhc,
  kDeepseekV4Dsa,
  kDeepseekV4Compressor,
  kDeepseekV4Moe,
  // fp8 KV-cache STORE (KV-FP8 W1). The fp8 sibling of kReshapeAndCache: the
  // paged K/V cache pages are 1-byte fp8-e4m3fn (DType::kI8 storage) and the
  // write quantizes each K/V element as Quantize(hp / k_scale|v_scale). Mirrors
  // reshape_and_cache_flash's fp8 branch (cache_kernels.cu:314-401,
  // CopyWithScaleOp :241-252). Kept a SEPARATE op so every float-cache caller
  // stays byte-identical. Appended before kCount so no existing op's id shifts.
  kReshapeAndCacheFp8,
  // SHARED fused routed-MoE gate+up+SwiGLU keep-quant epilogue. Promoted from the
  // DeepSeek-V4-private MoeDeviceKernels::moe_gate_up_swiglu seam
  // (deepseek_v4_device.h) into a first-class vt:: op so EVERY keep-quant MoE arch
  // inherits the tuned single-launch kernel — the contraction-tier sibling of the
  // grouped keep-quant GEMM kMatmulBTQuantGrouped. ONE launch computes, per
  // (expert-slot p, mid-row j): gate=gate_w[e,j]·xq, up=up_w[e,j]·xq (shared Q8_K
  // act, broadcast), then adown[p*N+j] = silu(min(gate,limit))·clamp(up,±limit) —
  // gate/up never touch HBM. BIT-IDENTICAL to {2× kMatmulBTQuantGrouped +
  // clamped-SwiGLU}; the CPU provider runs exactly that composite as the golden.
  // Appended before kCount so no existing op's id shifts.
  kMoeGateUpSwiGLUGrouped,
  // SHARED fused MLA norm-rope (Tier-A5 fold; ground: DeepSeek-V4-private
  // NormRopeRowsKernel, deepseek_v4.cpp Brick-7). ONE launch over the merged
  // kv_a projection row [T, off+rot] computes BOTH DeepSeek-MLA decoupled-rope
  // halves: latent_out[t,:off] = RmsNorm(x[t,:off]) with norm_weight (the
  // kv_a_layernorm over kv_lora_rank), and pe_out[t,:rot] = RopeFromCache-rotate
  // x[t,off:off+rot] (the UNNORMED, UNWEIGHTED decoupled k_pe). The two halves
  // are DISJOINT dims (latent normed, rope part not), so this is BIT-IDENTICAL
  // to {vt::RmsNorm(x[:,:off]); vt::RopeFromCache(x[:,off:])} — the CPU provider
  // runs exactly that composite as the golden. NOTE this reads the precomputed
  // cos|sin CACHE (like the MLA rope it replaces), NOT ds4's in-kernel analytic
  // recompute — DeepSeek-V2/kimi rope from a cache, so the cache path is what is
  // byte-exact here; ds4's own NormRopeRows stays its per-head analytic form.
  // Appended before kCount so no existing op's id shifts.
  kFusedNormRope,
  // SHARED fused BF16 grouped-MoE gate+up+SwiGLU (Tier-A4 fold). The bf16-native
  // arm of the routed-MoE gate+up+SwiGLU family (keep-quant arm is
  // kMoeGateUpSwiGLUGrouped): ONE vt entry replaces the {gate grouped GEMM; up
  // grouped GEMM; SiluAndMul} triplet the bf16 grouped-MoE archs (Qwen3-Coder,
  // DeepSeek-V2, kimi) ran. gate_w/up_w travel as the SAME per-expert bf16
  // device-pointer arrays [E] i64 kMoeGroupedGemmBf16 consumes (NOT a contiguous
  // [E*N,K] tensor — the bf16 experts are separate resident allocations), plus
  // the optional pair->token row_map. In the decode/non-WMMA regime the fused
  // kernels compute gate+up in ONE grouped launch (reusing the exact split-K
  // sequential-k accumulation) and reduce+SwiGLU in a second, dropping the two
  // f32 [P,I] HBM round-trips; in the WMMA regime it reuses LaunchGroupedBf16
  // twice + the byte-identical silu-mul. BIT-IDENTICAL to {2x kMoeGroupedGemmBf16
  // (f32 out) + kMoeSiluMul (bf16 out)} in every regime — that composite is the
  // golden the A/B unit test gates against. CUDA-only (like kMoeGroupedGemmBf16).
  // Appended before kCount so no existing op's id shifts.
  kMoeGroupedGemmBf16GateUpSilu,
  // Laguna-S-2.1 device-resident-decode glue table (the 5 small host ops the
  // NVFP4/Marlin arm still ran on the host: sequential RMSNorm, partial-NeoX RoPE,
  // GQA T=1 decode attention, per-head softplus out-gate, sigmoid-noaux top-k).
  // Registered on kCUDA by cuda_laguna.cu; resolved via laguna::LagunaDevice().
  // BYTE-EXACT (sequential reductions) to the host Laguna forward. Additive: only
  // LagunaForwardResidentDecode dispatches it. Appended before kCount (no id shift).
  kLaguna,
  // DENSE Marlin W4A16 GEMM (lift of vLLM's own dense marlin.cu marlin_gemm; see
  // MarlinDenseGemm below). Byte-preserving replacement for the single-expert
  // MoE-marlin route the dense E=1 NVFP4/MXFP4 projections use today — direct-A,
  // tile-per-CTA, vLLM's own dense fp32-C_tmp reduce (no par regrouping ULP).
  // CUDA-only (Blackwell sm_12xa; vendored dense marlin TUs, VT_MARLIN_NVFP4).
  // Appended before kCount (no existing op's id shifts).
  kMarlinDenseGemm,
  // MiniMax-H3 DiT device-resident-forward glue table (brick H3-2b). Only the 3
  // small ops the shared vt:: surface does NOT already cover: the two indexed
  // AdaLN modulates and plain elementwise SiLU. Everything else in the DiT
  // forward reuses tuned shared ops (kMatmulBT, kRmsNorm, kQkvSplit, kSiluAndMul,
  // kAdd, kIndexSelect, kIndexCopy, kRopeFromCache, kDFlashBlockAttention), so
  // this table stays deliberately tiny. Registered on BOTH kCPU and kCUDA
  // (cpu_ops.cpp / cuda_minimax_h3.cu) so the device forward is exercised in CPU
  // CI too; resolved via minimax_h3::MiniMaxH3Device(). Additive: only
  // MiniMaxH3DitForwardDevice dispatches it. Appended before kCount (no id shift).
  kMiniMaxH3,
  // --- Conformer / FastConformer audio-encoder kernels (spike
  // .agents/specs/parakeet-conformer-encoder.md rows P1/P2/P3). Three primitives
  // the tree had no device op for, each mirroring a transformers 5.3.0
  // `transformers/models/parakeet/modeling_parakeet.py` module and, where the
  // structure is identical, vLLM's own native conformer
  // (`vllm/model_executor/models/conformer_encoder.py`). See vt::Conv2d,
  // vt::DepthwiseConv1d and vt::AttentionRelPos below for the exact contracts.
  // Additive: nothing outside the audio-encoder path dispatches them. Appended
  // before kCount so no existing op's id shifts.
  kConv2d,
  kDepthwiseConv1d,
  kAttentionRelPos,
  // PERF-FP8-ALPHA-FOLD (.agents/specs/perf-fp8-alpha-fold.md, #402 §3 "Lever
  // B"): the vector-alpha form of kMatmulFp8CublasLt — a per-output-column f32
  // alpha applied INSIDE the cuBLASLt epilogue instead of by a second
  // full-tensor pass. A distinct id, not a parameter of the existing op,
  // because the pointer mode lives on the matmul DESCRIPTOR and therefore
  // participates in plan/algo selection. Appended before kCount so no existing
  // op's id shifts.
  kMatmulFp8CublasLtAlphaVec,
  // --- Mamba2 / SSD selective-scan core (.agents/specs/mamba2-ssd.md W1,
  // #496). The three numerical objects the GDN arm of KERNEL-SSM-MAMBA never
  // built. SSD is NOT the gated delta rule: there is no `(I - beta*k*k^T)`
  // removal term, the decay is a diagonal `exp(A*dt)` driven by `A_log`/`dt`,
  // and `B`/`C` are shared across `n_groups` head groups. See
  // vt::Mamba2ChunkScan / vt::Mamba2StateUpdate / vt::RmsNormGatedGroup below
  // for the exact contracts and their upstream anchors. Additive: nothing
  // outside the Mamba2 path dispatches them. Appended before kCount so no
  // existing op's id shifts.
  kMamba2ChunkScan,
  kMamba2StateUpdate,
  kRmsNormGatedGroup,
  // LTX-2.5 DiT device-resident-forward glue table (phase L8). Only the seven
  // small ops the shared vt:: surface does NOT already cover: the AdaLN table
  // lookup, the AdaZero affine, the gated residual accumulate, the per-head
  // attention gate, LTX's split/interleaved RoPE, the output head's
  // table+embedded affine, and plain ungated SiLU. Everything else in the DiT
  // forward reuses tuned shared ops (kMatmulBT, kRmsNorm, kLayerNorm, kGeluTanh,
  // kAdd, kAttention, kAttentionCross), so this table stays deliberately small.
  // Registered on BOTH kCPU and kCUDA (cpu_ltx2.cpp / cuda_ltx2.cu) so the
  // device forward is exercised in CPU CI too; resolved via ltx2::Ltx2Device().
  // Additive: only Ltx2DitForwardDevice dispatches it. Appended before kCount
  // so no existing op's id shifts.
  kLtx2,
  // The NON-GATED MoE activation: out = relu(x)^2, the whole epilogue of a
  // NemotronH expert (models/nemotron_h.py:227 activation_without_mul("relu2")
  // -> MoEActivation.RELU2_NO_MUL). Sibling of kMoeSiluMul with ONE input
  // instead of two, because a non-gated expert has no gate half to multiply by
  // (nemotron_h.py:220 ckpt_names=("up_proj","down_proj","")). See vt::MoeRelu2.
  // Appended before kCount so no existing op's id shifts.
  kMoeRelu2,
  // --- BigVGAN / DAC vocoder 1-D convolutions (#672,
  // .agents/specs/minimax-music3.md §11.4). The GENERAL grouped `nn.Conv1d` and
  // the transposed `nn.ConvTranspose1d` that the three `vocoder1d` consumers
  // (MiniMax-Music3, MiniMax-H3's audio VAE, IndexTTS-2.5, plus LTX-2.5's audio
  // VAE) are built from. Two reasons these are new ids rather than parameters of
  // kDepthwiseConv1d:
  //   * `vt` had NO transposed 1-D convolution of any kind. kCausalConv1dFwd is
  //     causal/stateful/SiLU-folded and kDepthwiseConv1d is centre-padded and
  //     depthwise; neither can express a scatter that GROWS the time axis.
  //   * the bias seeding and the zero-skip differ and are part of the contract,
  //     not implementation detail — see vt::ConvTranspose1d clause (3), where
  //     the skip decides the SIGN of a zero output cell. The ACCUMULATOR WIDTH
  //     used to differ too and no longer does: kDepthwiseConv1d accumulates in
  //     f32 and since #1474 so do these, because f32 is what torch accumulates
  //     a float convolution in. Either one's width still moves a shipped
  //     model's numerics, so they remain SIBLINGS and kDepthwiseConv1d is
  //     untouched.
  // See vt::Conv1d / vt::ConvTranspose1d below for the exact contracts.
  // Appended before kCount so no existing op's id shifts.
  kConv1d,
  kConvTranspose1d,
  // --- Block-wise FP8 (VT-QUANT-FP8-GROUP, #1189 milestone M1). The DYNAMIC
  // per-token, per-group fp8 activation quant that a 128x128 block-scaled FP8
  // GEMM consumes. It is not a parameter of kQuantFp8Static: that op takes ONE
  // static per-tensor scale from the checkpoint and emits no scale tensor at
  // all, while this one derives a scale per (row, group) at run time and emits
  // an f32 [M, K/group_size] second output. The nearest existing shape is
  // kScaledFp4Quant, which is dynamic per-token with a 2-D scale as well.
  // See vt::QuantFp8Group below for the contract.
  // Appended before kCount so no existing op's id shifts.
  kQuantFp8Group,
  // --- Block-wise FP8 (VT-MATMUL-FP8-BLOCK-REF, #1189 milestone M2). The
  // 128x128 block-scaled fp8 GEMM that consumes kQuantFp8Group's output. It is
  // NOT a parameter of kMatmulFp8Cutlass and cannot be: that op folds ONE
  // scalar alpha into the epilogue, which has exactly one degree of freedom per
  // output element, while this scheme has cdiv(K, block_k) of them and applies
  // them in the MAINLOOP. See vt::MatmulFp8BlockScaled below for the contract.
  // Appended before kCount so no existing op's id shifts.
  kMatmulFp8BlockScaled,
  // --- General 3-D convolution (LTX25-DEVICE-RESIDENCY W5, #1007). The
  // primitive `vt` has never had on ANY device, and the reason the LTX-2.5
  // video VAE decode ran on the host while every oracle runs it GPU-resident.
  // It is NOT a parameter of kConv2d, and the difference is the ACCUMULATION
  // ORDER rather than the rank: kConv2d keeps one f32 accumulator over the whole
  // (ic, kh, kw) sweep, while this op keeps one f32 PARTIAL PER INPUT CHANNEL —
  // which is what torch's blocked-GEMM f32 convolution does, and what every
  // committed LTX-2.5 video VAE golden was taken through. Same sibling
  // relationship, and the same reason, as kConv1d vs kDepthwiseConv1d above.
  // See vt::Conv3d below for the contract. Appended before kCount so no
  // existing op's id shifts.
  kConv3d,
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
  // Optional exact upstream prefill work descriptor, both i32 [num_programs]
  // on the queue device. Entry p owns sequence batch_ptr[p] and its
  // token_chunk_offset_ptr[p]-th 8-token chunk. CUDA consumes these when the
  // default CUDA path is selected; VT_CONV_EXACT_CHUNKS=0 restores the legacy mapping and
  // CPU keeps its scalar reference.
  const Tensor* batch_ptr = nullptr;
  const Tensor* token_chunk_offset_ptr = nullptr;
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

// Silu-gated GROUP RMS norm args (Mixer2RMSNormGated, mamba_mixer2.py:69-149).
// SIBLING of RmsNormGatedArgs, not a mode of it: the GDN/KDA gated norm reduces
// over the WHOLE row with an optional sigmoid gate, this one always uses SILU and
// reduces over `group_size = hidden / n_groups` slices (:136-141).
struct RmsNormGatedGroupArgs {
  float eps = 1e-6f;  // Mixer2RMSNormGated default (mamba_mixer2.py:76)
  // full_n_groups. group_size = hidden / n_groups must divide the last dim
  // (mamba_mixer2.py:80). n_groups == 1 degenerates to a whole-row RMS norm,
  // which is exactly upstream's `self.n_groups == 1` branch (:120-131).
  int64_t n_groups = 1;
  // W1 lands tp_world_size == 1 only. Any other value is REFUSED with a message
  // naming `extra_groups_for_head_shards` (mamba_utils.py:187) rather than
  // silently computing a wrong split (mamba2-ssd.md §7).
  int64_t tp_world_size = 1;
};

// Mamba2 SSD args, shared by the chunked prefill scan and the decode state
// update (ssd_combined.py:27-235, mamba_ssm.py:497+).
struct Mamba2Args {
  // Physical chunk length. MUST be an integer power of two — upstream asserts it
  // (`is_int_pow_2`, ssd_combined.py:48). Ignored by Mamba2StateUpdate.
  int64_t chunk_size = 0;
  // Whether dt goes through softplus before the decay (`dt_softplus`). Upstream
  // guards it as `dt <= 20 ? softplus(dt) : dt` (ssd_chunk_state.py:94,
  // mamba_kernels.hpp:177). mamba_mixer2.py:1097 passes True.
  bool dt_softplus = false;
  // dt_limit, applied AFTER softplus (`tl.clamp(dt, dt_min, dt_max)`,
  // ssd_chunk_state.py:96). Upstream default is (0.0, +inf).
  // Ignored by Mamba2StateUpdate: selective_state_update has no dt_limit.
  float dt_min = 0.0f;
  float dt_max = std::numeric_limits<float>::infinity();
  // W1 lands tp_world_size == 1 only; see RmsNormGatedGroupArgs::tp_world_size.
  int64_t tp_world_size = 1;
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

// Dense NON-CAUSAL CROSS attention args. vt::Attention requires key/value to
// carry the SAME token count as query (ops.cpp: "query/key/value token count must
// match"), which no cross-attention can satisfy: LTX-2.5's text cross-attention
// (transformer.py:113 `attn2`) and its audio<->video cross-attention
// (transformer.py:154 `audio_to_video_attn`, :166 `video_to_audio_attn`) each
// project queries from one stream and keys/values from another, so Tq != S by
// construction. This is that seam, kept SEPARATE from vt::Attention so every
// existing self-attention call stays byte-identical.
//
// query [Tq,Hq,D], key/value [S,Hkv,D], out [Tq,Hq,D]; Hq a multiple of Hkv
// (GQA broadcast, exactly as vt::Attention). Query i attends to EVERY key j in
// [0,S) — bidirectional, no causal mask; the only masking is the optional
// additive `bias` (see vt::AttentionCross). f32 softmax with max subtraction.
struct AttentionCrossArgs {
  // Softmax scale applied to the qk dot product. torch SDPA's default is
  // E**-0.5 with E = query.size(-1) = head_dim, which is what every LTX
  // attention gets (attention.py:98). Must be set explicitly (> 0).
  float scale = 0.0f;
};

// --- Conformer / FastConformer audio-encoder op args (spike
// .agents/specs/parakeet-conformer-encoder.md P1/P2/P3). ------------------------

// torch `nn.Conv2d` arguments. Mirrors the constructor keywords 1:1 so a reader
// of `ParakeetEncoderSubsamplingConv2D.__init__`
// (transformers 5.3.0 transformers/models/parakeet/modeling_parakeet.py:357-390)
// can map every field by name. `groups == in_channels == out_channels` is the
// DEPTHWISE form that module's inner layers use (:377-386); `groups == 1` with
// a 1x1 kernel is its pointwise layer (:388).
struct Conv2dArgs {
  int64_t stride_h = 1;
  int64_t stride_w = 1;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t dilation_h = 1;
  int64_t dilation_w = 1;
  int64_t groups = 1;
};

// --- General 3-D convolution args (LTX25-DEVICE-RESIDENCY W5, #1007). --------

// torch `nn.Conv3d` arguments, in `(T, H, W)` axis order. Field names mirror the
// constructor keywords 1:1, exactly as Conv2dArgs mirrors `nn.Conv2d`'s, so a
// reader of `CausalConv3d.__init__` (Lightricks/LTX-2 @ fd4ded7f2,
// packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:267-302) can map
// every field by name.
//
// `padding` here is ZERO padding on both sides of an axis, and that is the whole
// of what this op does: a non-zero `padding_mode` — LTX's `reflect` and
// `replicate` — is realised by the CALLER materialising the padded volume and
// passing pad 0, which is what torch itself does (`nn.Conv3d` with a non-`zeros`
// `padding_mode` runs `F.pad` and then a zero-padded convolution). The LTX
// decoder's asymmetric CAUSAL temporal pad is materialised the same way, and for
// the same reason: upstream materialises it too, with a `torch.concatenate`
// (convolution.py:305-311).
struct Conv3dArgs {
  int64_t stride_t = 1;
  int64_t stride_h = 1;
  int64_t stride_w = 1;
  int64_t pad_t = 0;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t dilation_t = 1;
  int64_t dilation_h = 1;
  int64_t dilation_w = 1;
  int64_t groups = 1;
};

// torch `nn.Conv1d(C, C, K, stride, padding, groups=C)` arguments — the
// NON-CAUSAL depthwise conv1d of `ParakeetEncoderConvolutionModule`
// (modeling_parakeet.py:116, ctor :138-146; padding = (K-1)//2 at :136). Not to
// be confused with CausalConv1dArgs, which drives the Mamba/GDN CAUSAL conv and
// carries a persistent conv_state; this op is stateless and centre-padded.
struct DepthwiseConv1dArgs {
  int64_t stride = 1;
  int64_t padding = 0;
  int64_t dilation = 1;
};

// --- BigVGAN / DAC vocoder 1-D convolution args (#672). --------------------

// torch `nn.Conv1d` arguments, the GENERAL grouped form. Field names mirror the
// constructor keywords 1:1. `groups == 1` is the dense conv the vocoder's
// conv_pre/conv_post/1x1 projections use; `groups == C_in == C_out` is the
// depthwise form the alias-free low-pass filter uses.
//
// NOT to be confused with DepthwiseConv1dArgs, which drives the conformer's
// f32-accumulate depthwise op — see OpId::kConv1d for why the two are siblings.
struct Conv1dArgs {
  int64_t stride = 1;
  int64_t padding = 0;  // ZERO padding on BOTH sides; out-of-range taps skipped
  int64_t dilation = 1;
  int64_t groups = 1;
};

// torch `nn.ConvTranspose1d` arguments. Mirrors the constructor keywords 1:1.
// `padding` here CROPS (torch's "dilation * (kernel_size - 1) - padding"
// implicit zero-padding on both sides), and `output_padding` appends to the
// right only.
struct ConvTranspose1dArgs {
  int64_t stride = 1;
  int64_t padding = 0;
  int64_t output_padding = 0;
  int64_t dilation = 1;
  int64_t groups = 1;
};

// Transformer-XL relative-position self-attention args — the conformer
// attention of `ParakeetEncoderAttention` (modeling_parakeet.py:259, forward
// :302-347, `_rel_shift` :349-355) and of vLLM's own native
// `RelPosMultiHeadAttention` (conformer_encoder.py:170, forward :188-217,
// `_rel_shift` :179-186). See vt::AttentionRelPos for the full formula.
struct AttentionRelPosArgs {
  // Softmax scale. Upstream sets it to head_dim^-0.5 (ParakeetEncoderAttention
  // .scaling :271; RelPosMultiHeadAttention.scale :174).
  float scale = 0.0f;
  // WHERE the scale is applied, the one arithmetic difference between the two
  // upstreams. false (default) = HF's form: `matrix_bd *= scaling` (:322) and
  // `attn_weights = q@k^T * scaling + matrix_bd` (eager_attention_forward :247),
  // i.e. `s = ac*scale + bd*scale`. true = vLLM's native form:
  // `attn_scores = (matrix_ac + matrix_bd); attn_scores.mul_(self.scale)`
  // (conformer_encoder.py:212-213), i.e. `s = (ac + bd) * scale`. The two agree
  // in exact arithmetic and differ in f32 rounding, so the flag is exposed
  // rather than chosen, and each upstream gets its own byte-exact path.
  bool scale_after_sum = false;
};

// Arguments for vt::DFlashBlockAttention — the DFlash draft's IN-BLOCK attention
// (SPEC-DFLASH D2, DF-DRAFT-MODEL; the project's FIRST bidirectional/non-causal
// attention primitive). Ported from the semantics of DFlashQwen3Attention +
// _resolve_layer_attention (vllm/model_executor/models/qwen3_dflash.py:86-146,
// 149-263 @ 555967922) and grounded in flashinfer's non-causal attention path
// (the #48167 Blackwell non-causal kernel now in-pin). This op computes attention
// for the uniform (1+k)-token QUERY block of each request over the keys IN THAT
// SAME block only (the context K/V is pre-inserted separately by the D3
// context-KV precompute; D2 isolates the block forward). Kept a SEPARATE op from
// vt::Attention / vt::PagedAttention so every CAUSAL model stays byte-identical.
//
// Per-request block boundaries come from `cu_seqlens` (host, length num_reqs+1):
// request r owns query/key rows [cu_seqlens[r], cu_seqlens[r+1]). Each query i in
// the block attends to keys j in the same block subject to:
//   - full-attention layer  (causal=false): ALL j in the block (BIDIRECTIONAL);
//   - sliding-window layer   (causal=true):  j <= i AND j >= i-(window-1)
//     (window<=0 means plain causal). Positions are the intra-block offsets, which
//     matches DFlash's contiguous (1+k) block; the z-lab 27B window (2048) >> 17
//     so the SWA layer degenerates to plain causal over the block — the mask still
//     computes the true window bound for fidelity to other DFlash checkpoints.
// f32 softmax accumulation (max-subtracted), matching vLLM. GQA broadcast as in
// vt::Attention. query [T,Hq,D], key/value [T,Hkv,D], out [T,Hq,D], T = ΣblockLen.
struct DFlashBlockAttentionArgs {
  float scale = 0.0f;              // head_dim^-0.5 (DFlashQwen3Attention.scaling)
  bool causal = false;            // per-layer: false=full(non-causal), true=SWA
  int64_t sliding_window = 0;     // SWA window (>0); 0 = full causal when causal
  const int32_t* cu_seqlens = nullptr;  // host, length num_reqs+1 (block bounds)
  int num_reqs = 1;               // number of query blocks
};

// SPEC-DFLASH D12 Part B — CAPTURE-SAFE paged variant of DFlashBlockAttention.
// The (1+k) block queries attend over [persistent PAGED context ; their own (1+k)
// block] exactly as DFlashBlockAttention does over a materialized [context; block]
// combined buffer, but the growing context enters as DATA (a paged K/V cache +
// per-request block_table + seq_lens) instead of a variable-size combined buffer —
// so the launch grid is STATIC over the fixed (1+k)*num_reqs query rows and every
// metadata input is a persistent DEVICE tensor read in place (NO function-local
// host upload, the cudagraph-capture-bakes-stack-addresses UAF class the eager
// DFlashBlockAttention launcher had). Same f32 online-softmax recurrence + the D2
// in-block mask (full/non-causal or causal-SWA), applied over the COMBINED index
// (context rows [0,C_r) then block rows [C_r, C_r+blen_r)); the context is always
// position-ordered (ascending), so the combined mask matches the materialized one
// bit-for-bit. CUDA is capture-safe; CPU is the reference; a unit test cross-checks
// both against DFlashBlockAttention over an explicit combined buffer.
struct DFlashPagedBlockAttentionArgs {
  float scale = 0.0f;             // head_dim^-0.5
  bool causal = false;            // false=full(non-causal); true=causal-SWA
  int64_t sliding_window = 0;     // SWA window (>0); 0 = full causal when causal
  int num_reqs = 1;               // number of (1+k) query blocks
  int64_t block_size = 0;         // rows per paged context page (>0)
};

// Arguments for vt::DFlashGroupedConv — the DFlash2 draft's GROUPED DYNAMIC
// DEPTHWISE CONVOLUTION, wrapped around each attention and each MLP sublayer
// (SPEC-DFLASH2 W2, #1314).
//
// BEYOND-PIN. Ported from `_grouped_conv` and `DFlashGroupedConv`
// (vllm/model_executor/models/qwen3_dflash2.py @ vllm-project/vllm#52816 head
// `19c9351904df4c63042671bc67a866ca48dc7d6f`); the parity pin `555967922` does
// not carry the architecture at all and this op does NOT advance it.
//
// The math, in upstream's own terms:
//
//   out[i,c] = sum_{t=0..taps-1} (base[side,t,c] + delta[i,side,t,g(c)]) * x[i-t,c]
//
// with `g(c) = c / group_size` the channel's GROUP, and tap `t` contributing only
// where `(i mod block_size) >= t`. That mask is what makes the op a BLOCK
// convolution rather than a sequence one: a proposal position sees the positions
// before it inside its own (1+k) query block and NOTHING across the block
// boundary, which is how a DFlash2 draft gets causal structure without another
// backbone pass. `i` is the GLOBAL row index, exactly as upstream's
// `torch.arange(hidden_states.shape[0])` is, and it is the right one here for the
// same reason it is there: every request's block is contiguous and
// block_size-aligned, so `i mod block_size` IS the intra-block offset.
//
// Upstream computes that mask on a POWER-OF-TWO block as `position & (block-1)`
// and otherwise as `position % block`. Both arms are mirrored, and both are
// gated: the two published DFlash2 checkpoints ship block 8 and block 16 (both
// power-of-two, `z-lab/Qwen3.8-27B-DFlash2` and `z-lab/Muse-Glimmer-30B-DFlash2`)
// and upstream's own reference test parametrises 5 to reach the modulo arm.
//
// `block_size` is `1 + num_speculative_tokens`, NOT `dflash_config.block_size` —
// upstream sizes the conv by the QUERY block (the bonus token plus the mask
// tokens) rather than by the checkpoint key, which only supplies that value's
// default (`DFlash2Qwen3DecoderLayer.__init__` @ that head).
//
// SIDES. `base_kernel` is `[2, taps, hidden]` and dim 0 is the SIDE — 0 =
// `prepare` (before the sublayer), 1 = `finish` (after it) — NOT a tap. One
// projection of the sublayer input produces BOTH sides' deltas
// (`kernel_projection`: hidden -> 2*taps*num_groups), so `coefficients` is the
// same buffer for both calls and `side` selects the half. Passing the whole
// buffer rather than a slice mirrors upstream's `coefficients[:, side]` view
// without materializing a copy of a non-contiguous slice.
//
// ACCUMULATION. Every intermediate is rounded to the tensor dtype after each
// step, because upstream's chain materializes bf16 tensors at each one
// (`base + delta`, `coefficients * blocks`, `output += ...`). This is elementwise
// with no reduction-order freedom, so the CPU reference and the CUDA kernel are
// specified BIT-IDENTICAL rather than within an envelope.
//
// WHAT IS ACTUALLY GATED, because the two halves of that sentence are not
// equally proven. The per-step POLICY is pinned on CPU in bf16 by
// `tests/vt/test_ops_dflash2_grouped_conv.cpp` — one hand-computed case with
// literal expectations that differ from the rounded-once-at-the-end answer in
// six of eight outputs, plus three shapes asserted bit-exact against a reference
// that rounds where UPSTREAM materializes. On f32 this rounding is the identity
// by construction, so no f32 case can see it and none is claimed to. The CPU ==
// CUDA half is NOT proven: that case exists and is written to run, but it has
// never compiled on a host without `nvcc` and reports `no CUDA backend;
// skipping`. See `## Owed` O6 of `.agents/specs/dflash2-spec-decode.md`.
struct DFlashGroupedConvArgs {
  int64_t block_size = 0;   // 1 + num_speculative_tokens (the query block)
  int64_t taps = 0;         // dflash_config.conv_kernel_size
  int64_t num_groups = 0;   // hidden_size / conv_group_size
  int64_t group_size = 0;   // dflash_config.conv_group_size
  int64_t side = 0;         // 0 = prepare, 1 = finish (selects base/coefficient half)
};

// Arguments for vt::Dflash2SelectorEdges — the DFlash2 CANDIDATE SELECTOR's edge
// lattice (SPEC-DFLASH2 W3, #1314).
//
// BEYOND-PIN. Ported from `_score_edges` and `CandidateSelector.forward`
// (vllm/model_executor/models/qwen3_dflash2.py:208-276 @ vllm-project/vllm#52816
// head `66e5414c6d75a8529473d977f7458c140bbab8a0`); the parity pin `555967922`
// does not carry the architecture at all and this op does NOT advance it. The
// PR head MOVED from `19c9351904df4c63042671bc67a866ca48dc7d6f` on 2026-08-19
// (#1404); `_score_edges` and `CandidateSelector` are BYTE-IDENTICAL at the two
// heads, and only the enclosing model's `set_model_tag` and the LM-head guard
// changed, so this op cites the NEW head and its math is unaffected.
//
// The math, in upstream's own terms:
//
//   edge(b, l, p, c) = unary[b,l,c]
//                    + sum_r (pred_codebook[pid(b,l,p), r] * hidden[b,l,r])
//                            * succ_codebook[cand[b,l,c], r]
//
// where `pid(b,l,p)` is the PREDECESSOR token of slot `p` at step `l`: the
// request's verified ANCHOR token at `l == 0` (the same token for every `p`,
// upstream's `anchor_token_ids[:, None, None].expand(-1, 1, top_k)`), and
// `cand[b, l-1, p]` at every later step. So `p` indexes the previous step's K
// candidates and `c` this step's K candidates, which is exactly the transition
// lattice the W4 path walk consumes.
//
// `unary` is the candidate VALUE from `compute_candidates` — the target head's
// top-K logit AFTER `output_multiplier` and `final_logit_softcapping` — and it
// is f32 there and f32 here. Upstream broadcasts it over `p`
// (`unary_logits[:, :, None]`), so it is a per-CHILD bias and not a per-edge one.
//
// ACCUMULATION AND DTYPE. Upstream materializes two bf16 tensors inside the
// einsum chain and then promotes: `predecessors * hidden[:, :, None]` is a bf16
// tensor, the einsum's own output is a bf16 tensor, and `unary_logits + <bf16>`
// promotes to f32 by torch's own type-promotion rule. This op mirrors that
// placement exactly: the elementwise product rounds to the codebook dtype, the
// rank reduction accumulates in f32 and rounds ONCE to the codebook dtype, and
// only then is the f32 `unary` added. On f32 codebooks both roundings are the
// identity, so the policy is observable ONLY in bf16 — which is where it is
// gated, for the reason `## Owed` O6 records for the W2 convolution.
//
// The rank reduction IS a reduction, so unlike vt::DFlashGroupedConv this op is
// NOT specified bit-identical across backends: a CUDA mirror is free to reduce
// in a different order and is gated within an f32 envelope. That is a real
// difference from W2 and is stated here rather than inherited by analogy.
struct Dflash2SelectorEdgesArgs {
  int64_t top_k = 0;  // dflash_config.selector_top_k (16 on both published drafts)
};

// Arguments for vt::Dflash2PathWalk — the DFlash2 candidate selector's PATH WALK
// (SPEC-DFLASH2 W4, #1314).
//
// BEYOND-PIN. Ported from `_selector_walk_kernel`
// (vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:16-79 @
// vllm-project/vllm#52816 head `66e5414c6d75a8529473d977f7458c140bbab8a0`).
//
// WHAT IT IS. The selector scores every (predecessor, child) transition of every
// step; the walk turns that lattice into k tokens. It starts at the verified
// anchor -- which the lattice already carries as EVERY predecessor slot of step
// 0, so the walk enters at slot 0 -- takes the best child, and reads the NEXT
// step's block at the predecessor row it just chose. The slot-to-slot dependency
// is the whole shape of the problem: step l cannot start until step l-1 has
// picked, so `L` steps are inherently sequential.
//
// ITS SHAPE IS UPSTREAM'S SHAPE, and that is a decision rather than a detail.
// Upstream runs ONE program per request: the K scores of a slot stay in
// registers and the step loop lives INSIDE the program, so `L` steps cost one
// launch rather than `L`. Spec `## Risks/decisions` D3 requires the walk to be
// on device from the first landing, because the identical sequential shape in
// DSpark shipped host-side and measured 28% of the 27B draft step
// (https://github.com/mudler/vllm.cpp/issues/436) before
// `SampleSequentialDevice` moved it. The CUDA arm here is one block per request
// with a shuffle argmax; the CPU arm is the authoritative reference.
//
// GREEDY IS THE ONLY ARM, and it is upstream's greedy arm exactly. At the moved
// head the walk's two hand-written branches collapse into one
// `gumbel_noised_argmax` call whose temperature argument is
// `temperature if SAMPLE_PROBABILISTIC else 0.0`, and `SAMPLE_PROBABILISTIC` is
// `self.draft_logits is not None`, which is set only for
// `draft_sample_method == "probabilistic"`. At temperature 0 that helper divides
// by nothing, adds no noise, and returns `tl.max(logits, axis=0,
// return_indices=True)`; the `USE_FP64` cast it may apply is order-preserving on
// fp32 inputs and cannot move a greedy answer. This engine refuses
// `draft_sample_method: "probabilistic"` BY NAME at
// `vllm::ParseSpeculativeConfigJson` (src/vllm/config/speculative.cpp) and
// verifies accept-iff-equal, so no configuration can reach the noised arm and
// nothing could consume the realized proposal distribution it exists to feed.
// Landing that arm would be landing code no entry point can reach; it is owed,
// with its layout, at `## Owed` of .agents/specs/dflash2-spec-decode.md.
//
// THE TIE-BREAK AND THE -inf ROW ARE PART OF THE CONTRACT. `tl.max(...,
// return_indices=True)` resolves a tie to the LOWEST index, and upstream loads
// masked lanes as -inf, so a row that is entirely -inf resolves to index 0
// rather than to "no index". Both are pinned by literal cases and both arms
// implement them identically: the reduction is seeded at -inf with slot index
// `top_k`, keeps a candidate only when it STRICTLY exceeds the running best (so
// a NaN never wins, on either arm), and collapses a `top_k` seed back to 0. A
// different tie rule would pick a different PREDECESSOR for the next step, so it
// moves the whole remaining path and not one token.
//
// UNLIKE vt::Dflash2SelectorEdges this op is specified BIT-EXACT across
// backends. It performs no arithmetic -- only comparisons and one gather -- so
// there is no reduction order for a backend to differ in.
//
// STRICTNESS IS THE WHOLE OF THAT CLAIM, and it was not free. W4's fresh review
// measured the two arms apart on a NaN-bearing row: the CUDA per-lane scan
// carried `|| (v == best && j < slot)` beside its strict `>`, a clause that is
// unreachable once a lane has claimed anything (`j` only ascends) and whose one
// effect was at the SEED -- a lane holding -inf compared equal to the -inf seed
// and claimed a slot the CPU arm refuses. `[NaN,-inf]` read cpu 0 / cuda 1,
// `[NaN,NaN,-inf]` read cpu 0 / cuda 2, and every NaN-free row agreed,
// including the all -inf row and the forced tie group, so no fixture without a
// NaN could see it. The clause is now DELETED rather than this claim narrowed,
// and the lower-slot tie rule stays where it is needed: the cross-lane
// butterfly, which combines lane winners out of slot order. BOTH HALVES OF
// THAT DELETION ARE NOW COVERED, and #1518 corrects an earlier sentence here
// saying it had never been compiled or run on a device. It COMPILES on every
// pull request: `src/vt/cuda/cuda_ops.cu` is in the CUDA source list
// (CMakeLists.txt, `target_sources(vllm PRIVATE ...)` under `if(VLLM_CPP_CUDA)`)
// that CI's `build-cuda-fat` job builds for ten architectures
// (`80;86;87;89;90a;100a;103a;110;120a;121a`). And it has RUN: the operator
// executed this suite on `dgx:gpu0` (GB10, sm_121a) at the W4 merge commit,
// reporting 83 assertions on device against 49 on CPU, `Status: SUCCESS!`, with
// zero `no CUDA backend; skipping` lines -- the increment includes the NaN row
// chained into the parity fixture, which is the row that measured the
// divergence. What remains owed at `## Owed` O11 of
// .agents/specs/dflash2-spec-decode.md is the rest of that entry, not this
// deletion; the AUTHORING HOST still has no `nvcc` and still skips the case
// locally. NO SHIPPED PATH FEEDS THIS OP A NaN:
// the lattice comes from vt::Dflash2SelectorEdges over a target LM head, so the
// row that measured the difference is synthetic, and it is a gap in the reach
// of the contract rather than in any draft a user can obtain.
struct Dflash2PathWalkArgs {
  int64_t top_k = 0;  // dflash_config.selector_top_k (16 on both published drafts)
};

// Arguments for vt::TopKValuesIndices — the vocabulary top-k that EMITS the
// surviving (id, value) pairs (SPEC-DFLASH2 W3 / D2, #1314).
//
// BEYOND-PIN. Ported from `_topk`
// (vllm/model_executor/models/qwen3_dflash2.py:60-64 @ vllm-project/vllm#52816
// head `66e5414c6d75a8529473d977f7458c140bbab8a0`), which is `torch.topk(scores,
// k, dim=-1)` off CUDA and FlashInfer's radix `top_k(..., sorted=True,
// deterministic=True)` on it.
//
// WHY NOT FLASHINFER'S RADIX KERNEL. Spec `## Risks/decisions` D2: `topk.cuh` is
// 3380 lines of general kernel (multi-CTA, deterministic mode, three tie-break
// modes, dynamic shared-memory sizing) for a shape that is fixed and small here
// — K = 16 over a 248320 vocabulary for `num_reqs * k` rows, about 224 at
// concurrency 32. The CUDA arm instead extends the sort-free block-cooperative
// pivot-bracket threshold search this repository already carries and gates
// (`src/vt/cuda/cuda_sample.cu::ApplyTopKTopPRowKernel`, ported from the SAME
// FlashInfer `TopK/TopPRenormProb` approach) so that it COMPACTS and ORDERS the
// survivors instead of masking below the k-th largest.
//
// TIE-BREAK IS PART OF THE CONTRACT, not an implementation detail. The threshold
// search finds an exact array VALUE, so it keeps whole tie groups atomically and
// can leave more than k survivors; something then has to choose among equals.
// This op returns exactly k pairs ordered by DESCENDING value with ties broken
// by ASCENDING index, which is `torch.topk`'s CPU order and what FlashInfer's
// `deterministic=True` exists to provide. A backend that broke ties differently
// would reorder the selector's candidate slots and move acceptance without
// raising anything, so the CPU reference pins it and the CUDA arm mirrors it.
//
// NaN IS THE ONE POINT WHERE THE TWO ARMS ARE NOT EQUAL, and this is stated here
// because a contract that claimed otherwise would be a claim no shipped backend
// delivers. The CPU arm orders NaN FIRST, which is `torch.topk(largest=True)`'s
// own answer, and it is stated rather than left to the sort: leaving it implicit
// made the CPU comparator an intransitive equivalence and therefore undefined
// behaviour, not merely an unusual result. The CUDA arm DOES NOT ORDER NaN
// FIRST and cannot as written -- `TopKValuesIndicesRowKernel`'s pivot bracket
// uses `fmaxf`/`fminf`, which return the non-NaN operand, and its survivor pass
// tests `r[j] > thr`, which is false for a NaN, so the search can never select
// one. Measured on a GB10 on 2026-08-20 rather than argued:
// [#1489](https://github.com/mudler/vllm.cpp/issues/1489), where the direct
// cross-arm comparison read `gpu.indices[0] == cpu.indices[0]` as `2 == 1`.
// Reconciling the kernel to NaN-first is owed to that issue; until it lands the
// device gate is scoped to the rows the kernel implements. NO SHIPPED PATH FEEDS
// THIS OP A NaN LOGIT -- the candidate values come from a target LM head -- so
// the row that pins the CPU order is synthetic in the same sense the padding row
// is, and the asymmetry is a gap in the contract's reach rather than in any
// output a user can obtain.
//
// `num_org_vocab_padding` mirrors upstream's
// `lm_head.shard_indices.num_org_vocab_padding`: that many columns at the END of
// each row are forced to -inf BEFORE the search, so a padded head can never
// contribute a candidate. It is 0 on every path this engine ships today (the
// DFlash lane's lm_head is the raw unpadded checkpoint tensor and there is no
// vocab-parallel sharding), and it is implemented and gated synthetically rather
// than claimed as checkpoint coverage — the same posture `## Upstream chain`
// records for the three output scalars. Upstream's companion
// `org_vocab_start_index` is applied by the CALLER
// (`vllm::Qwen3DFlash2Model::ComputeCandidates`), not here, because it is an
// id-space rebase of the result rather than a property of the search.
struct TopKValuesIndicesArgs {
  int64_t k = 0;                        // dflash_config.selector_top_k
  int64_t num_org_vocab_padding = 0;    // trailing columns forced to -inf first
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
  // OPTIONAL fp8 KV-cache read (KV-FP8 W1). kAuto (default) => the cache holds
  // the model float dtype and is read directly — every existing caller is
  // byte-identical. When != kAuto the K/V cache pages are 1-byte fp8 (DType::kI8
  // storage) and each read is DEQUANTIZED as Dequant(fp8) * k_scale|v_scale
  // before entering the f32 softmax, mirroring the fp8 attention read path
  // (scaled_vec_conversion<float,uint8_t>, quant_utils.cuh:302-308). k_scale /
  // v_scale are the per-tensor scales from BaseKVCacheMethod (kv_cache.py:108-191)
  // — 1.0 is the uncalibrated default. Per-head scales are a later brick.
  Fp8KVCacheDataType kv_cache_dtype = Fp8KVCacheDataType::kAuto;
  float k_scale = 1.0f;
  float v_scale = 1.0f;
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
// The trailing bool is `claims_splitk1_premise` — see MatmulFp8CublasLt below.
using MatmulFp8CublasLtFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, float, bool);
// Same operands as MatmulFp8CublasLtFn, but the trailing scalar alpha becomes a
// device f32 [N] vector — one folded alpha per OUTPUT COLUMN.
using MatmulFp8CublasLtAlphaVecFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor& /*alpha_vec*/, bool);
using QuantFp8StaticFn = void (*)(Queue&, Tensor&, const Tensor&, float);
// Two outputs, because the scale is computed rather than supplied: the fp8 bytes
// and the f32 [M, K/group_size] per-group scale.
using QuantFp8GroupFn = void (*)(Queue&, Tensor& /*out_fp8*/, Tensor& /*out_scale*/,
                                 const Tensor& /*x*/, int /*group_size*/);
// Two scale streams and the block geometry, where the per-tensor fp8 GEMMs above
// carry one scalar alpha. The geometry is not a convenience: it is what selects
// which scale pair each K-block multiplies by.
using MatmulFp8BlockScaledFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*a_fp8*/,
                                        const Tensor& /*a_scale*/, const Tensor& /*b_fp8*/,
                                        const Tensor& /*b_scale*/, int /*block_n*/,
                                        int /*block_k*/);
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
// kMoeGroupedGemmBf16GateUpSilu: out[P,N] bf16 = silu(gate)*up, gate/up = grouped
// bf16 GEMM of act[T|P,K] against per-expert weight-pointer arrays gate_ptrs/
// up_ptrs [E] i64, expert_ids[P] i32, optional row_map[P] i32. Same convention as
// MoeGroupedGemmBf16 with a SECOND weight-pointer array + the fused SwiGLU epilogue.
using MoeGroupedGemmBf16GateUpSiluFn =
    void (*)(Queue&, Tensor& /*out*/, const Tensor& /*act*/, const Tensor& /*expert_ids*/,
             const Tensor* /*row_map*/, const Tensor& /*gate_ptrs*/, const Tensor& /*up_ptrs*/);
// kMatmulBTQuantGrouped: out[P,N], act[P,K] (f32/bf16), weight[E*N,K] block-quant,
// expert_ids[P] i32 — weight row for (p,n) is expert_ids[p]*N + n.
using MatmulBTQuantGroupedFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&);
// kMoeGateUpSwiGLUGrouped: out[P,N] f32 adown, act[Pa,K] (Pa==1 broadcast),
// gate_w/up_w[E*N,K] SAME block-quant dtype, expert_ids[P] i32, float limit. See
// vt::MoeGateUpSwiGLUGrouped / OpId::kMoeGateUpSwiGLUGrouped.
using MoeGateUpSwiGLUGroupedFn =
    void (*)(Queue&, Tensor& /*out*/, const Tensor& /*act*/, const Tensor& /*gate_w*/,
             const Tensor& /*up_w*/, const Tensor& /*expert_ids*/, float /*limit*/);
// Marlin NVFP4 W4A16 grouped-MoE GEMM (lift of vLLM moe_wna16_marlin_gemm; see
// MoeGroupedGemmNvfp4Marlin below). Scalar params travel in MoeMarlinArgs.
struct MoeMarlinArgs {
  int moe_block_size = 0;  // vLLM moe_align_block_size block (16..64, or 8)
  int top_k = 0;
  int size_m = 0;  // number of tokens (rows of `a`)
  int size_n = 0;  // output features
  int size_k = 0;  // input features (contraction; multiple of 16)
  bool mul_topk_weights = false;  // fold topk_weights into the output (down proj)
  // Block-scale format selector. Default = NVFP4 (fp8-e4m3 scales, group 16,
  // per-tensor global scale). group_size 32 + mxfp4=true selects the MXFP4 path
  // (E8M0/UE8M0 scales => s_type kFE8M0fnu, group_blocks 2, NO global scale;
  // the `global_scale` tensor is ignored). Mirrors vLLM's is_nvfp4 branch.
  int group_size = 16;
  bool mxfp4 = false;
};
using MoeGroupedGemmNvfp4MarlinFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, Tensor&,
             const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MoeMarlinArgs&);
// DENSE Marlin W4A16 GEMM (lift of vLLM's own dense marlin_gemm; see
// MarlinDenseGemm below). Scalar params travel in MarlinDenseArgs. Unlike the
// MoE path there is NO moe_align gather (sorted_token_ids/expert_ids/top_k):
// `a` is a plain [size_m, size_k] contiguous activation (lda = size_k).
struct MarlinDenseArgs {
  int size_m = 0;  // number of tokens (rows of `a`)
  int size_n = 0;  // output features
  int size_k = 0;  // input features (contraction; multiple of 16)
  // Block-scale format selector, identical semantics to MoeMarlinArgs: default =
  // NVFP4 (fp8-e4m3 scales, group 16, per-tensor global scale). group_size 32 +
  // mxfp4=true selects the MXFP4 path (E8M0 scales, group_blocks 2, NO global
  // scale; the `global_scale` tensor is ignored). Mirrors vLLM's is_nvfp4 branch.
  int group_size = 16;
  bool mxfp4 = false;
};
using MarlinDenseGemmFn =
    void (*)(Queue&, Tensor& /*c*/, const Tensor& /*a*/, const Tensor& /*b_q_weight*/,
             const Tensor& /*b_scales*/, const Tensor& /*global_scale*/, Tensor& /*workspace*/,
             const MarlinDenseArgs&);
using MoeSiluMulFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
// kMoeRelu2: out[i] = relu(x[i])^2 — the NON-GATED MoE activation (one input).
using MoeRelu2Fn = void (*)(Queue&, Tensor&, const Tensor&);
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
// Gate-free fused preamble (kAttnQkNormRope): q3/k3 are normed and rotated IN
// PLACE, so the op takes no separate outputs. See the recipe in recipes.h.
using AttnQkNormRopeFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor&, const RmsNormArgs&, const RopeArgs&);
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
// Fused MLA norm-rope (kFusedNormRope): latent RmsNorm + decoupled-pe
// RopeFromCache over one merged kv_a row, in ONE launch. See vt::FusedNormRope.
using FusedNormRopeFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, const RmsNormArgs&,
                                 const RopeArgs&);
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
using GdnSpecDecodeFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, const GdnArgs&);
using CausalConv1dSpecUpdateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                          const Tensor*, Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const CausalConv1dArgs&);
using GdnPackedDecodeFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor&, const Tensor&, Tensor&, const Tensor&,
             const GdnArgs&);
// Per-k-channel-decay gated-delta recurrence (KDA). Same shape as GdnPrefillFn;
// the ONLY difference is g is [T,Hv,Dk] (per-channel) not [T,Hv] (per-head).
using KdaGatedDeltaRuleFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&, Tensor&, const Tensor&,
                                     const GdnArgs&);
// KDA CHUNK-PREFILL: the chunked (WY-representation) forward of the SAME
// per-K-channel gated-delta linear attention as KdaGatedDeltaRule, but processing
// the whole prompt in BT=64 chunks through the vendored FLA Triton-AOT cubins
// (vLLM's actual prefill kernels) instead of the token-sequential recurrence.
// Takes the RAW gate projection g_raw + a_log + dt_bias (the gate is fused
// on-device by kda_gate_cumsum), NOT a pre-gated per-channel decay.
using KdaChunkPrefillFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                   const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                   Tensor&, const Tensor&, const GdnArgs&);
// Mamba2 SSD varlen chunked prefill scan (vt::Mamba2ChunkScan). NOT a shape of
// GdnPrefillFn: the SSD recurrence has no delta-removal term and carries the
// per-chunk decay/state-passing metadata upstream builds in mamba2_attn.py.
using Mamba2ChunkScanFn = void (*)(Queue&, Tensor& /*out*/, Tensor& /*final_states*/,
                                   const Tensor& /*x*/, const Tensor& /*dt*/,
                                   const Tensor& /*A*/, const Tensor& /*B*/,
                                   const Tensor& /*C*/, const Tensor* /*D*/,
                                   const Tensor* /*z*/, const Tensor* /*dt_bias*/,
                                   const Tensor* /*initial_states*/,
                                   const Tensor& /*cu_seqlens*/,
                                   const Tensor& /*cu_chunk_seqlens*/,
                                   const Tensor& /*last_chunk_indices*/,
                                   const Tensor& /*seq_idx*/, const Mamba2Args&);
// Mamba2 single-token selective state update (vt::Mamba2StateUpdate).
using Mamba2StateUpdateFn = void (*)(Queue&, Tensor& /*out*/, Tensor& /*state*/,
                                     const Tensor& /*x*/, const Tensor& /*dt*/,
                                     const Tensor& /*A*/, const Tensor& /*B*/,
                                     const Tensor& /*C*/, const Tensor* /*D*/,
                                     const Tensor* /*z*/, const Tensor* /*dt_bias*/,
                                     const Tensor* /*state_indices*/, const Mamba2Args&);
// Silu-gated group RMS norm (vt::RmsNormGatedGroup). `weight` is nullable:
// upstream skips the parameter entirely when use_rms_norm is False.
using RmsNormGatedGroupFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/,
                                     const Tensor& /*gate*/, const Tensor* /*weight*/,
                                     const RmsNormGatedGroupArgs&);
using GdnStateGatherFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*);
using GdnStateScatterFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using IndexSelectFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using IndexCopyFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using MoeRouterTopKFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&,
                                 const MoeRouterTopKArgs&, const Tensor*);
// The trailing float is `routed_scale` — the routed_scaling_factor applied to
// the ROUTED sum before the shared term is added (see vt::MoeCombine).
using MoeCombineFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*, float);
using MoeCombineGateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&);
using AttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const AttentionArgs&);
using AttentionCrossFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor* /*bias*/, const AttentionCrossArgs&);
// Conformer / FastConformer audio-encoder kernels (spike P1/P2/P3).
using Conv2dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/, const Tensor& /*weight*/,
                          const Tensor* /*bias*/, const Conv2dArgs&);
using DepthwiseConv1dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/,
                                   const Tensor& /*weight*/, const Tensor* /*bias*/,
                                   const DepthwiseConv1dArgs&);
// General 3-D convolution (#1007).
using Conv3dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/, const Tensor& /*weight*/,
                          const Tensor* /*bias*/, const Conv3dArgs&);
// BigVGAN / DAC vocoder 1-D convolutions (#672).
using Conv1dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/, const Tensor& /*weight*/,
                          const Tensor* /*bias*/, const Conv1dArgs&);
using ConvTranspose1dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/,
                                   const Tensor& /*weight*/, const Tensor* /*bias*/,
                                   const ConvTranspose1dArgs&);
using AttentionRelPosFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*query*/,
                                   const Tensor& /*key*/, const Tensor& /*value*/,
                                   const Tensor& /*rel_key*/, const Tensor* /*bias_u*/,
                                   const Tensor* /*bias_v*/, const Tensor* /*key_mask*/,
                                   const AttentionRelPosArgs&);
using DFlashBlockAttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const DFlashBlockAttentionArgs&);
using DFlashPagedBlockAttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                             const Tensor&, const Tensor&, const Tensor&,
                                             const Tensor&, const Tensor&, const Tensor&,
                                             const DFlashPagedBlockAttentionArgs&);
using DFlashGroupedConvFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                     const Tensor&, const DFlashGroupedConvArgs&);
using Dflash2SelectorEdgesFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Dflash2SelectorEdgesArgs&);
using Dflash2PathWalkFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                   const Dflash2PathWalkArgs&);
using TopKValuesIndicesFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&,
                                     const TopKValuesIndicesArgs&);
using ReshapeAndCacheFn = void (*)(Queue&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                   const Tensor&);
// fp8 KV-cache store (KV-FP8 W1). k_cache/v_cache are 1-byte fp8 (DType::kI8);
// each element is stored as Quantize(hp / k_scale|v_scale). `kind` selects the
// fp8 interpretation (kFp8E4M3 landed; kFp8E5M2 is a later brick).
using ReshapeAndCacheFp8Fn = void (*)(Queue&, const Tensor& /*k*/, const Tensor& /*v*/,
                                      Tensor& /*k_cache*/, Tensor& /*v_cache*/,
                                      const Tensor& /*slot_mapping*/, Fp8KVCacheDataType /*kind*/,
                                      float /*k_scale*/, float /*v_scale*/);
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
// --- Greedy spec-decode rejection sampling (SPEC-REJECTION I3).
using GreedyRejectionSampleFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&,
                                         const Tensor&);
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

// vt::MatmulBTQuantGrouped — grouped keep-quant GEMM over an expert-index list.
// out[P,N] where out[p,:] = act[p,:] . weight[expert_ids[p]*N .. +N] (the block
// row-slice of the stacked expert weight [E*N,K]). Collapses the DeepSeek-V4 MoE's
// per-expert kMatmulBTQuant matvecs into ONE launch. The CPU provider loops
// kMatmulBTQuant per group ⇒ BYTE-IDENTICAL to the per-expert path; the CUDA
// provider runs one warp-per-(p,n) grouped kernel (fewer launches, higher
// occupancy). act f32/bf16, weight Q8_K-family block-quant, expert_ids i32.
void MatmulBTQuantGrouped(Queue& q, Tensor& out, const Tensor& act,
                          const Tensor& weight, const Tensor& expert_ids);

// vt::MoeGateUpSwiGLUGrouped — SHARED fused routed-MoE gate+up+SwiGLU keep-quant
// epilogue (OpId::kMoeGateUpSwiGLUGrouped). out[P,N] f32 = per (p,j):
//   gate = min(FinalFactor·(gate_w[e,j]·xq),  limit)
//   up   = clamp(FinalFactor·(up_w[e,j]·xq), -limit, limit)
//   out[p,j] = gate·sigmoid(gate)·up          (clamped SwiGLU, α=1, β=0)
// where e = expert_ids[p] and xq is act[p] (or act[0] broadcast when Pa==1) quantized
// to Q8_K ONCE. gate_w/up_w are the stacked [E*N,K] expert towers in the SAME CUDA
// keep-quant dtype. The CUDA provider is one warp-per-(p,j) fused launch (gate/up stay
// in registers). The CPU provider runs the BYTE-EXACT composite: two
// vt::MatmulBTQuantGrouped (gate,up) + the clamped-SwiGLU elementwise — so it is the
// golden the fused kernel is gated against. limit=+inf reduces to plain silu(gate)·up
// (a standard SwiGLU MLP with no clamp). Promoted from DeepSeek-V4's private
// MoeDeviceKernels::moe_gate_up_swiglu so any keep-quant MoE arch inherits it.
void MoeGateUpSwiGLUGrouped(Queue& q, Tensor& out, const Tensor& act, const Tensor& gate_w,
                            const Tensor& up_w, const Tensor& expert_ids, float limit);

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

// QuantFp8Static (mirror vLLM static_scaled_fp8_quant):
//   inv = 1/input_scale;  out_fp8[i] = fp8_e4m3( clamp(x[i] * inv, -448, 448) )
// RNE convert. The scale is applied as a RECIPROCAL MULTIPLY, not a divide, and
// the reciprocal is formed ONCE outside the elementwise loop — that is what
// upstream ships: `x = val * scale` under is_scale_inverted=true
// (csrc/quantization/w8a8/fp8/common.cuh:62, clamp at :68) with the inverse formed
// by the caller (csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:31,
// `1.0f / scale[...]`). DO NOT "correct" the kernels to a divide to match a
// prose formula: `x/s` and `x*(1/s)` differ by up to one f32 ulp, and near an
// e4m3 tie that ulp changes the emitted byte on a default-ON 35B path.
// Static per-tensor scale (NOT dynamic/per-token). x [M,K] f32/bf16, out [M,K]
// i8 (raw fp8-e4m3fn bytes). CUDA + CPU (the CPU arm is the portable reference
// that makes the fp8 seam testable without a GPU, #468).
void QuantFp8Static(Queue& q, Tensor& out_fp8, const Tensor& x, float input_scale);

// --- Block-wise FP8 (VT-QUANT-FP8-GROUP, #1189 M1,
// .agents/specs/vt-quant-fp8-group.md). QuantFp8Group is the DYNAMIC per-token,
// per-group sibling of QuantFp8Static: the scale is derived from the data, once
// per contiguous run of `group_size` elements inside a row, and written out.
//
//   amax = max(1e-10, max |x_f32| over the group)
//   y_s  = amax / 448.0f
//   out_fp8[i] = fp8_e4m3( min(max(x_f32[i] / y_s, -448.0f), 448.0f) )
//   out_scale[row, group] = y_s
//
// x [M,K] f32/bf16, out_fp8 [M,K] i8 (raw fp8-e4m3fn bytes), out_scale
// [M, K/group_size] F32 — f32 because upstream allocates it f32 and the GEMM
// that consumes it multiplies in f32 (fp8_utils.py:629-631). K must be a
// multiple of group_size; the op refuses any other K by name, as upstream
// asserts at fp8_utils.py:596-599. CPU + CUDA.
//
// WHICH UPSTREAM ARM THIS MIRRORS, because there are two and they DISAGREE.
// `per_token_group_quant_fp8` reads like a Triton kernel with a C++ fast path
// and it is the other way round: on a CUDA-alike platform with a contiguous
// input it calls the C++ custom op and RETURNS (fp8_utils.py:635-650), so the
// Triton kernel never executes there. The executing kernel is
// csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu:
//   :47  float local_absmax = eps           — eps SEEDS the reduction, so an
//                                             all-zero group cannot divide by 0
//   :53  fmaxf(local_absmax, fabsf((float)src))
//   :68  float y_s = local_absmax / max_8bit                    — a DIVIDE
//   :85  fminf(fmaxf((float)src / y_s, min_8bit), max_8bit)     — a DIVIDE
//   :86  DST_DTYPE(q)                       — hardware e4m3 RNE, saturating
// DO NOT "correct" either divide into a hoisted reciprocal multiply to match
// QuantFp8Static's form. That form is right for QuantFp8Static because upstream
// ships it there (common.cuh:62 with the reciprocal formed by the caller); here
// upstream ships a divide, the scale changes per group so nothing is
// loop-invariant, and the Triton fallback's `_absmax * (1.0 / fp8_max)`
// (fp8_utils.py:145) carries an upstream comment naming the 1-ULP gap it opens.
// One f32 ulp before an e4m3 round changes the emitted byte near a tie, and
// upstream's own test cannot see it: it compares values at rtol=0.15 and the
// scale at rtol=1e-5 (test_block_fp8.py:112-115). Only a byte comparison can,
// which is what tests/vt/test_ops_quant_fp8_group_cpu.cpp G1 is.
//
// The column-major and TMA-aligned scale layouts (fp8_utils.py:610-628) and the
// `use_ue8m0` DeepGEMM scale rounding (per_token_group_quant.cu:69-71) are NOT
// implemented. Both are recorded under `## Owed` in the row's spec; no consumer
// in this tree can read either yet, and upstream excludes the target model from
// DeepGEMM on family 120 (vllm/utils/deep_gemm.py:27-46).
void QuantFp8Group(Queue& q, Tensor& out_fp8, Tensor& out_scale, const Tensor& x,
                   int group_size);

// MatmulFp8BlockScaled (VT-MATMUL-FP8-BLOCK-REF, #1189 M2,
// .agents/specs/vt-matmul-fp8-block-ref.md) — the 128x128 block-scaled fp8 GEMM,
// mirroring native_w8a8_block_matmul (tests/kernels/quant_utils.py:91-154):
//
//   for each (m, n):
//     acc = 0                                              f32
//     for kt in [0, cdiv(K, block_k)):
//       part = 0                                           f32, a SEPARATE register
//       for k in the k-tile:  part += f8(a[m,k]) * f8(b[n,k])
//       acc += part * ( a_scale[m, kt] * b_scale[n / block_n, kt] )
//     out[m, n] = acc                                      stored to out's dtype
//
// THE SCALES APPLY IN THE MAINLOOP, ONCE PER K-BLOCK, INTO AN F32 ACCUMULATOR —
// NOT IN THE EPILOGUE, and that is a correctness constraint rather than an
// optimisation choice. MatmulFp8Cutlass above folds one scalar alpha after the
// whole K reduction. An epilogue has exactly ONE degree of freedom per output
// element; this scheme has cdiv(K, block_k) of them. An epilogue-only
// application therefore cannot express a per-K-block scale AT ALL, which is why
// this is a separate op. DO NOT "simplify" `part` away into `acc`: that IS the
// epilogue form, and tests/vt/test_ops_matmul_fp8_block_cpu.cpp G4 is built so
// that no single-alpha implementation can pass it.
//
// WHICH UPSTREAM ARM THIS MIRRORS. The Triton kernel at fp8_utils.py:826-836 is
// not what executes on the target architecture; CUTLASS is
// (vllm/model_executor/kernels/linear/__init__.py:355-377 ranks it third,
// DeepGEMM is auto-disabled for qwen3_5_text on family 120 at
// vllm/utils/deep_gemm.py:27-46, and Marlin is excluded at cc >= 89). Unlike the
// QuantFp8Group case above, the two AGREE: csrc/.../c3x/
// scaled_mm_blockwise_sm120_fp8_dispatch.cuh:56-58,218-235 hands both scale
// pointers to the MAINLOOP arguments over an `ElementAccumulator = float`, and
// cutlass 4.5.0's sm120_mma_tma_blockwise_scaling.hpp:714-717 is literally
// `accum(i) += tmp_accum(i) * tCrScaleAViewAsC(i) * tCrScaleBViewAsC(i)`.
// CUTLASS associates the two scale multiplies left to right where the reference
// forms their product first (quant_utils.py:150-151); the difference is at most
// one f32 ULP per K-block and upstream's own gate admits it, comparing the two
// at rel_diff < 0.001 (test_block_fp8.py:194-200). We mirror the reference's
// association, because this op IS the reference port and the CUDA kernel that
// #1189 milestone M5 lands will be measured against it.
//
// SHAPES, with CEIL on every tiling, so a ragged final block is legal and must
// work (upstream asserts exactly this at fp8_utils.py:935-936):
//   a_fp8   [M,K]                              i8, raw fp8-e4m3fn bytes
//   a_scale [M, cdiv(K, block_k)]              F32
//   b_fp8   [N,K]                              i8, raw fp8-e4m3fn bytes
//   b_scale [cdiv(N, block_n), cdiv(K, block_k)] F32
//   out     [M,N]                              f32 or bf16
// The scales are f32 because upstream refuses any other dtype on this path
// (csrc/.../c3x/scaled_mm_helper.hpp:15-18) and the accumulator is f32.
// a_scale's K axis is a CEIL too, so this op accepts a K that vt::QuantFp8Group
// would refuse; that asymmetry is upstream's own (fp8_utils.py:930 uses cdiv
// where fp8_utils.py:596-599 demands divisibility).
//
// No bias: upstream refuses one outright on the blockwise path
// (scaled_mm_helper.hpp:54), so this mirrors a refusal rather than deferring a
// feature.
//
// CPU only. A CORRECTNESS REFERENCE, NOT A PERFORMANCE PATH — it is the
// numerical oracle #1189 milestone M5's CUTLASS kernel is measured against, and
// it makes no speed claim. M5 owns the CUDA arm.
void MatmulFp8BlockScaled(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& a_scale,
                          const Tensor& b_fp8, const Tensor& b_scale, int block_n,
                          int block_k);

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
// K,N multiples of 16 (128-bit fp8 alignment). CUDA (sm120a) + a CPU CORRECTNESS
// REFERENCE (f32 accumulate, naive triple loop — no speed claim, and no
// production model routes through it; it exists so the fp8 seam resolves on a
// CPU queue and can be gated without a GPU, #468). The CPU arm is EXPECTED to
// agree with the CUDA kernel to fp8/bf16 tolerance and NOT byte-for-byte, because
// the CUDA arm reduces K in tensor-core order and rounds its epilogue through
// bf16 — but that agreement is DECLARED AND OWED, not measured. No committed run
// has compared the two arms: gate G2 of .agents/specs/vt-fp8-w8a8-cpu-arm.md is
// PENDING for want of a GPU. Treat the tolerance above as the claim to be tested,
// not as a result.
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
//
// `claims_splitk1_premise` (default FALSE) is how a caller opts INTO a stricter
// contract than the op otherwise offers. Pass it only when this GEMM's bf16 `out`
// is asserted to be byte-equivalent to the SAME GEMM's f32 `out` — a pure
// store-width narrowing over one ordered f32 reduction. That claim requires the
// selected cuBLASLt plan to run at splitK=1 (a split-K sums per-split partials
// in an order the f32 arm never used, and f32 addition is not associative), so
// the implementation verifies it and REFUSES otherwise.
//
// It is FALSE by default because that claim is unusual. An ordinary bf16 `out`
// — what every `o_proj` / `out_proj` fp8 projection asks for, on a default-ON
// path — is just an output dtype: split-K is correct for it, it is compared
// against nothing, and it is never checked. Do not set this flag merely because
// `out` is bf16; set it when you are asserting equivalence with an f32 arm.
void MatmulFp8CublasLt(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                       float alpha, bool claims_splitk1_premise = false);

// MatmulFp8CublasLtAlphaVec — the SAME fp8 GEMM with a per-output-COLUMN alpha:
//   out[m,n] = alpha_vec[n] * (A_fp8[M,K] @ B_fp8[N,K]^T)[m,n]
// `alpha_vec` is f32 [N], contiguous, on the queue device; `out` is f32 OR bf16
// [M,N]. This is the form an N-CONCATENATED operand needs when its shards carry
// different folded alphas (input_scale * that shard's weight_scale), which no
// single host scalar can express. Mirrors the tensor-alpha overload the NVFP4
// CUTLASS path already took (.agents/specs/nvfp4-device-alpha.md).
//
// The op is TOTAL: when VT_FP8_ALPHA_VEC_EPILOGUE=1 AND the heuristic returns an
// algo whose CUBLASLT_ALGO_CAP_POINTER_MODE_MASK advertises
// ALPHA_DEVICE_VECTOR_BETA_ZERO, the alpha is applied in the cuBLASLt epilogue
// (one launch). Otherwise it runs the GEMM at alpha=1 and applies the vector
// with vt::MulColVecF32 — the two-launch form this seam shipped with, byte for
// byte. Callers therefore never branch on the toggle or the driver's capability;
// they express the per-column alpha ONCE, here. CUDA-only.
//
// A bf16 `out` (PERF-FP8-ALPHA-FOLD / #417) is what vLLM emits for this
// projection (ModelOptFp8LinearMethod's out_dtype is the model dtype,
// modelopt.py:458 @ the pin). It halves the bytes the column pass moves — the
// dominant cost, since that pass is a full read-modify-write measured at 77% of
// the device's peak bandwidth. It is NOT value-neutral: the GEMM's f32
// accumulator is rounded to bf16 before the alpha multiply instead of after it,
// so callers opt in rather than defaulting to it.
//
// A bf16 `out` always takes the TWO-LAUNCH arm regardless of the toggle. At bf16
// the epilogue would round once where the fallback rounds twice, so admitting it
// would make VT_FP8_ALPHA_VEC_EPILOGUE change VALUES rather than just speed; the
// toggle is kept a pure performance A/B at every dtype.
//
// `claims_splitk1_premise` carries exactly the meaning it has on
// MatmulFp8CublasLt above, and reaches the same check: a bf16 `out` always takes
// the two-launch arm, whose GEMM IS MatmulFp8CublasLt.
void MatmulFp8CublasLtAlphaVec(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                               const Tensor& alpha_vec, bool claims_splitk1_premise = false);

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

// vt::MoeGroupedGemmBf16GateUpSilu — SHARED fused BF16 grouped-MoE gate+up+SwiGLU
// (Tier-A4 fold, OpId::kMoeGroupedGemmBf16GateUpSilu). ONE vt entry replaces the
// {MoeGroupedGemmBf16(gate); MoeGroupedGemmBf16(up); MoeSiluMul} triplet the bf16
// grouped-MoE archs (Qwen3-Coder, DeepSeek-V2, kimi) ran:
//   out[p, j] = silu(gate[p, j]) * up[p, j]           (bf16 store)
// where gate[p, j] = sum_k act[row(p), k] * gate_W_e[k, j],
//       up[p, j]   = sum_k act[row(p), k] * up_W_e[k, j],
//       e = expert_ids[p], row(p) = row_map ? row_map[p] : p, and gate_W_e/up_W_e
// are gate_ptrs[e]/up_ptrs[e] (bf16 [K, N] Matmul-B weights). Same argument
// convention as MoeGroupedGemmBf16 with a SECOND weight-pointer array. BIT-IDENTICAL
// to that composite in every launch regime — the decode/non-WMMA path fuses the two
// GEMMs into one grouped launch + a reduce+SwiGLU launch (reusing the exact split-K
// sequential-k accumulation and the MoeSiluMul math); the WMMA path reuses the
// grouped-GEMM dispatch twice + the identical silu-mul. CUDA only.
//   out          [P, N] bf16 (the per-(token,slot) silu(gate)*up)
//   act          [*, K] bf16
//   expert_ids   [P] i32 (device)
//   row_map      [P] i32 (device) or nullptr
//   gate_ptrs    [E] i64 (device) — gate expert weight pointers (bf16 [K, N])
//   up_ptrs      [E] i64 (device) — up   expert weight pointers (bf16 [K, N])
void MoeGroupedGemmBf16GateUpSilu(Queue& q, Tensor& out, const Tensor& act,
                                  const Tensor& expert_ids, const Tensor* row_map,
                                  const Tensor& gate_ptrs, const Tensor& up_ptrs);

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

// MarlinDenseGemm (lift of vLLM's DENSE marlin_gemm, marlin.cu:545 -> marlin_mm
// at :326 — the byte-preserving dense W4A16 kernel vLLM itself ships for a16
// weight-only linears). One launch computes y = a @ dequant(b).T with vLLM's own
// direct-A, tile-per-CTA layout and dense fp32-C_tmp reduce — NOT the MoE
// single-expert route, whose par regrouping of the reduce costs one bf16 ULP.
//   c            [size_m, size_n]         bf16 (out)
//   a            [size_m, size_k]         bf16 (token hidden; contiguous, lda=size_k)
//   b_q_weight   [size_k/16, size_n*8/pack] i32 — Marlin-interleaved fp4 (SAME
//                repack as the MoE path: marlin_permute; a shim is added only if
//                a layout divergence is proven — see dense_nvfp4_gemm.h)
//   b_scales     [size_k/group_size, size_n] fp8 (processed marlin scales)
//   global_scale [1]                      f32 (nvfp4 only; ignored for mxfp4)
//   workspace    [>= sms]                 i32 (zeroed reduction locks)
// CUDA-only (Blackwell sm_12xa; needs the vendored dense Marlin TUs, VT_MARLIN_NVFP4).
void MarlinDenseGemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& b_q_weight,
                     const Tensor& b_scales, const Tensor& global_scale, Tensor& workspace,
                     const MarlinDenseArgs& args);

// out[R,I] = silu(gate[R,I]) * up[R,I]  (moe-semantics.md §4; the fused-MoE
// element-wise activation between the grouped gate/up and down GEMMs). gate/up
// f32 or bf16, out f32/bf16; silu/mul computed in f32, rounded on store. Unlike
// vt::SiluAndMul (single [T,2D] input), this takes the two separately-produced
// projections so no concat/copy is needed. CPU + CUDA.
void MoeSiluMul(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up);

// out[R,I] = relu(x[R,I])^2 — the NON-GATED MoE activation, and the whole
// epilogue of a NemotronH expert. Mirror of vLLM's `ReLUSquaredActivation`
// (layers/activation.py:609-628, forward_native = torch.square(F.relu(x))) as
// reached through the fused-MoE path: `activation_without_mul("relu2")` ->
// `MoEActivation.RELU2_NO_MUL` -> `apply_moe_activation`'s
// `F.relu(input, inplace=True); torch.square(input, out=output)`
// (layers/fused_moe/activation.py:34 `RELU2_NO_MUL`, :98
// `activation_without_mul`, and the :184 RELU2_NO_MUL branch; `:33` is
// GELU_TANH_NO_MUL, the neighbouring enumerator).
//
// Why this is NOT a MergedGemmGroup epilogue: a NON-gated expert has no gate
// half to merge with (nemotron_h.py:220 `ckpt_names=("up_proj","down_proj","")`
// — the empty third entry IS the absent gate). There is exactly ONE projection,
// so the expert is the EXISTING grouped GEMM plus this activation, exactly as
// the gated bf16 archs are kMoeGroupedGemmBf16 + kMoeSiluMul. See
// merged_gemm.h's note on the non-gated family.
//
// DTYPE/ROUNDING ORDER is the mirrored part, not an implementation detail:
// upstream's kernel (csrc/libtorch_stable/activation_kernels.cu:673-678)
// widens to f32, clamps at zero in f32, squares in f32 and rounds ONCE on the
// store — so a bf16 input with an f32 output keeps the FULL f32 square. x f32
// or bf16, out f32/bf16. CPU + CUDA.
void MoeRelu2(Queue& q, Tensor& out, const Tensor& x);

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

// Elementwise GELU (NEW, Qwen3-VL vision tower). out[i] = gelu(x[i]); `out` may
// alias `x`. x/out f32 or bf16, computed in f32. GeluTanh is the tanh-approx
// (`gelu_pytorch_tanh`, vision MLP); GeluErf is exact erf (`nn.GELU()`, merger).
void GeluTanh(Queue& q, Tensor& out, const Tensor& x);
void GeluErf(Queue& q, Tensor& out, const Tensor& x);

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

// FusedNormRope (kFusedNormRope): the fused DeepSeek-MLA norm-rope — one launch
// replacing the {RmsNorm(kv_c latent) ; RopeFromCache(k_pe)} pair over the merged
// kv_a projection output. Grid is one block per token; each block RMS-reduces the
// leading latent slice x[t, 0:off) and writes latent_out, then rotates the trailing
// decoupled-rope slice x[t, off:off+rot) from the precomputed cos|sin cache and
// writes pe_out. The two halves address DISJOINT dims, so it is BIT-FOR-BIT equal
// to composing the two standalone ops:
//   x         [T, off+rot] f32/bf16 — merged kv_a output (contiguous; off = norm_weight length)
//   norm_weight [off]      f32/bf16 — kv_a_layernorm weight (over kv_lora_rank)
//   positions [T]          i32/i64  — rope positions
//   cos_sin_cache [P, rot] f32      — the rope cache (cols [0,rot/2)=cos, [rot/2,rot)=sin)
//   latent_out[T, off]     f32/bf16 — RmsNorm(x[:, :off]) (NOT roped)
//   pe_out    [T, rot]     f32/bf16 — RopeFromCache-rotated x[:, off:] (NOT normed, single vector)
// norm_args: eps + gemma (gemma=false for DeepSeek). rope_args: rotary_dim (=rot) +
// is_neox_style (DeepSeek-V2/V3 use the GPT-J adjacent-pair form, is_neox_style=false).
// CPU + CUDA. Additive: only the DeepSeek-V2/kimi MLA block dispatches it.
void FusedNormRope(Queue& q, Tensor& latent_out, Tensor& pe_out, const Tensor& x,
                   const Tensor& norm_weight, const Tensor& positions,
                   const Tensor& cos_sin_cache, const RmsNormArgs& norm_args,
                   const RopeArgs& rope_args);

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

// SPECULATIVE multi-token conv step (SPEC-MTP I4). Ported from
// vllm/model_executor/layers/mamba/ops/causal_conv1d.py @ e24d1b24
// (_causal_conv1d_update_kernel IS_SPEC_DECODING + IS_VARLEN branches
// :835-1067, wrapper `state_len = width - 1 + (seqlen - 1)` :1181-1184), as
// called by qwen_gdn_linear_attn.py:1344-1356.
//
// A speculative request submits `1 + k` query tokens per step, but only
// `num_accepted_tokens` of the PREVIOUS step's tokens were actually kept. The
// conv window must therefore advance by exactly the accepted count, NOT by
// 1 + k. Upstream does that WITHOUT any rollback copy, by widening the conv
// state row to `state_len = (K-1) + k` taps and treating it as a sliding
// window: the read offset for this step is `num_accepted_tokens[i] - 1`.
//
//   x            [num_tokens, C] varlen token stream of the SPEC rows only,
//                request i occupying [cu_seqlens[i], cu_seqlens[i+1]).
//                May be a padded-row (inner-contiguous) view. Written in place
//                is NOT done here — `out` is a separate [num_tokens, C] buffer.
//   conv_state   [num_slots, C, state_len] FULL persistent cache, updated in
//                place. state_len must be (K-1) + max_query_len - 1.
//   conv_state_indices [num_reqs] — the slot of each spec row. This is
//                spec_state_indices_tensor COLUMN 0 only: unlike the SSM state,
//                the conv window needs no per-timestep slots because the whole
//                1+k history fits in the widened row (qwen_gdn_linear_attn.py:1350).
//                Local ABI: index < 0 ⇒ NULL row, skipped (see GdnDecode).
//   num_accepted_tokens [num_reqs] i32, each in [1, max_query_len].
//   cu_seqlens   [num_reqs + 1] i32 (upstream query_start_loc).
//   max_query_len = 1 + k (upstream `max_query_len`, = spec_state_indices.size(-1)).
//
// Semantics, per request i and channel c, with off = num_accepted_tokens[i]-1,
// S = conv_state[slot][c][off : off+K-1] ++ x[i-th row range][c]:
//   out[t][c] = act(bias[c] + sum_{j<K} w[c][j] * S[t + j])
//   conv_state[slot][c] <- S[1 : 1 + state_len]      (left-shift by ONE tap)
// The left-shift-by-one (not by seqlen) is what makes the NEXT step's
// `off = num_accepted - 1` select exactly the window ending at the last
// accepted token — i.e. the conv rollback.
void CausalConv1dSpecUpdate(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                            const Tensor* bias, Tensor& conv_state,
                            const Tensor& conv_state_indices,
                            const Tensor& num_accepted_tokens, const Tensor& cu_seqlens,
                            const CausalConv1dArgs& args);

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

// Kimi Delta Attention (KDA) gated-delta recurrence — the PER-K-CHANNEL-DECAY
// variant of GdnPrefill. Ported 1:1 from FLA's
// fused_recurrent_gated_delta_rule_fwd_kernel with IS_KDA=True (third_party/
// flash_linear_attention/ops/fused_recurrent.py:88-175 @ pin 555967922; the KDA
// wrapper is ops/kda.py:109-146 fused_recurrent_kda). The plain-GDN kernel
// (IS_KDA=False) applies a PER-HEAD scalar decay `b_h *= exp(b_g)`; KDA applies a
// PER-K-CHANNEL decay `b_h *= exp(b_gk[None, :])` — so `g` here is [T,Hv,Dk]
// (one log-decay per K channel of the value head's [Dv,Dk] state), broadcast
// across the Dv rows. Everything else is byte-for-byte GdnPrefill's recurrence
// (decay -> predict -> beta -> rank-1 update -> read-out, all in f32):
//   S[:,k] *= exp(g[hv,k]);  v' = (v - S @ k) * beta[hv];  S += outer(v',k);
//   out = S @ (q*scale)
// The shared GDN kernels are UNTOUCHED (Qwen3.6 27B/35B GDN gate byte-identical);
// KDA lands as this additive per-channel op. q_in/k [T,Hk,Dk] MUST be
// l2-normalized by the caller (as GdnPrefill; upstream fuses it via
// USE_QK_L2NORM_IN_KERNEL, exact per gdn-semantics.md §4). v/out [T,Hv,Dv],
// g [T,Hv,Dk] f32 per-channel log-decay, beta [T,Hv] f32 (per-head sigmoid(b)),
// state [N,Hv,Dv,Dk] f32 in/out (zeros for fresh sequences),
// query_start_loc [N+1] i32. For KDA Hk==Hv; GQA broadcast supported for reuse.
void KdaGatedDeltaRule(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                       const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                       const Tensor& query_start_loc, const GdnArgs& args);

// KDA CHUNK-PREFILL — the chunked forward of KdaGatedDeltaRule, computing the same
// per-K-channel gated-delta linear-attention output but in BT=64 chunks through the
// vendored FLA Triton-AOT cubins (kda_gate_cumsum -> kkt(inter+intra) -> solve_tril
// -> recompute_w_u -> chunk_delta_h -> chunk_gla_o), mirroring vLLM's prefill path
// (kimi_gdn_linear_attn.py:141 chunk_kda_with_fused_gate; decode stays the recurrent
// KdaGatedDeltaRule). Result differs from the recurrence only by chunked-vs-recurrent
// REDUCTION ORDER (not bit-exact). Fires only at the pinned Kimi KDA geometry
// (Hk==Hv==32, Dk==Dv==128) in a CUDA + VLLM_CPP_TRITON build; any other shape or a
// CPU queue transparently falls back to the recurrence. Inputs:
//   q_in/k [T,H,Dk] f32/bf16, L2-normalized by the caller (as KdaGatedDeltaRule);
//   v [T,H,Dv] f32/bf16; out [T,H,Dv] f32/bf16;
//   g_raw [T,H,Dk] f32 — the RAW gate projection (kda_gate_cumsum fuses the gate:
//     -exp(a_log)*softplus(g_raw+dt_bias), chunk-cumsum, RCP_LN2 fold, on device);
//   beta [T,H] f32 — the per-head gate (sigmoid(b), applied directly);
//   a_log [H] f32; dt_bias [H*Dk] f32 (or empty => no bias);
//   state [1,H,Dv,Dk] f32 (fresh zeros in, final written); query_start_loc [N+1] i32.
// args.scale == Dk^-0.5 (baked in the cubins; guarded).
void KdaChunkPrefill(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                     const Tensor& v, const Tensor& g_raw, const Tensor& beta,
                     const Tensor& a_log, const Tensor& dt_bias, Tensor& state,
                     const Tensor& query_start_loc, const GdnArgs& args);

// ─── Mamba2 / SSD (.agents/specs/mamba2-ssd.md, issue #496) ──────────────────
//
// SSD IS NOT THE GATED DELTA RULE. GdnPrefill/KdaGatedDeltaRule above carry a
// delta REMOVAL term `(I - beta*k*k^T)` and a decay that is a per-head (GDN) or
// per-K-channel (KDA) scalar. Mamba2's SSD is a diagonally-decayed gated linear
// recurrence with NO removal term, whose decay `exp(A[h]*dt[t,h])` is driven by
// `A_log`/`dt`, with a `D` skip and `B`/`C` SHARED across `n_groups` head groups
// (head h reads group `h / (nheads/ngroups)`). These are sibling ops, never a
// parameterisation of the GDN kernels (mamba2-ssd.md §0, §7).

// MAMBA2 CHUNKED SSD PREFILL — the varlen entry
// `mamba_chunk_scan_combined_varlen` (ssd_combined.py:157-235) over the 5-stage
// pipeline `_mamba_chunk_scan_combined_fwd` (:27-156), in upstream order:
//   1. `_chunk_cumsum_fwd`  (ssd_chunk_state.py:300-346) — dt bias/softplus/clamp
//      then per-chunk `dA_cumsum[h,c,i] = sum_{j<=i} dt[h,c,j]*A[h]`. Positions
//      past a partial chunk's length hold dt = 0, so `dA_cumsum[h,c,cs-1]` is the
//      chunk's TOTAL decay whatever its length (:96-110).
//   2. `_chunk_state_fwd`   (ssd_chunk_state.py:349-407) — the chunk-local state
//      `sum_i x*B * exp(min(dA_last - dA_i, 0)) * dt`, accumulated in f32
//      (`states_in_fp32=True`, ssd_combined.py:100-102).
//   3. `_state_passing_fwd` (ssd_state_passing.py:99-146) — the inter-chunk
//      recurrence `S_c = exp(dA_last[c]) * S_{c-1} + states[c]`, run per SEQUENCE
//      over the chunk range `last_chunk_indices` derives, seeded from
//      `initial_states`. Note out[c] is the state AFTER chunk c (:90-97).
//   4. `_bmm_chunk_fwd`     (ssd_bmm.py:148-209) — `CB[c,g,i,j] = C_i . B_j`,
//      accumulated and RETURNED in f32 regardless of activation dtype
//      (`output_dtype=torch.float32`, ssd_combined.py:124).
//   5. `_chunk_scan_fwd`    (ssd_chunk_scan.py:216-525) — per token i of chunk c:
//        out = exp(dA_i) * (C_i . S_{c-1})                       // inter-chunk
//            + sum_{j<=i} CB[i,j] * exp(min(dA_i - dA_j,0)) * dt_j * x_j // intra
//            + D * x_i                                           // skip
//        and `out *= z*sigmoid(z)` when z is given. `S_{c-1}` is
//        `initial_states[seq_idx[c]]` when `seq_idx[c] != seq_idx[c-1]` and
//        initial states were supplied, ZEROS when they were not (:236-250,
//        :271-289) — that is what makes a sequence boundary inside a physical
//        chunk correct.
//
// Shapes (varlen, implicit batch 1 — ssd_combined.py:158-215):
//   x            [T,H,P] float          dt      [T,H] float
//   A            [H] f32 (negative)     B,C     [T,G,N] float, H % G == 0
//   D            optional [H] or [H,P] f32       z  optional [T,H,P] float
//   dt_bias      optional [H] f32
//   initial_states optional [S,H,P,N] f32/bf16 (:79, :194)
//   cu_seqlens   [S+1] i32              seq_idx [nchunks] i32 — PER CHUNK, not
//                                       per token (:60-61, :189)
//   cu_chunk_seqlens [nchunks+1] i32    last_chunk_indices [S] i32
//   out          [T,H,P] f32/bf16 (pre-allocated, written in place)
//   final_states [S,H,P,N] f32/bf16 — `varlen_states`, the state after each
//                sequence's LAST chunk (:154). Its dtype is the SEPARATE
//                `state_dtype` knob (:46,119,176), NOT the activation dtype.
// `args.chunk_size` must be a power of two (:48). All arithmetic is f32; the
// device tile-precision downcasts inside the Triton dots are a W2 concern.
void Mamba2ChunkScan(Queue& q, Tensor& out, Tensor& final_states, const Tensor& x,
                     const Tensor& dt, const Tensor& A, const Tensor& B, const Tensor& C,
                     const Tensor* D, const Tensor* z, const Tensor* dt_bias,
                     const Tensor* initial_states, const Tensor& cu_seqlens,
                     const Tensor& cu_chunk_seqlens, const Tensor& last_chunk_indices,
                     const Tensor& seq_idx, const Mamba2Args& args);

// MAMBA2 SINGLE-TOKEN SELECTIVE STATE UPDATE — the decode path,
// `selective_state_update` (mamba_ssm.py:497+) as mamba_mixer2.py:1087 calls it.
// Per row b and head h (group g = h / (H/G)), with dt SCALAR PER HEAD — the
// `tie_hdim` shape Mamba2 always uses, and the only one upstream's own CPU
// kernel implements (csrc/cpu/mamba_kernels.hpp:104-250, and the
// `current_platform.is_cpu()` skips in tests/kernels/mamba/test_mamba_ssm.py):
//   dt = softplus(dt[b,h] + dt_bias[h])            // guarded dt<=20, :177
//   s  = s * exp(A[h]*dt) + B[b,g,:] * x[b,h,p] * dt
//   out[b,h,p] = sum_n s[n]*C[b,g,n] + D[h]*x[b,h,p]   (then *= silu(z))
//
//   state   [S,H,P,N] f32/bf16 — the FULL cache when state_indices is given
//           (mamba2_state_dtype's ssm_dtype is its own knob, mamba_utils.py:73-81),
//           else compact [Nb,H,P,N] with row b == b. Updated IN PLACE.
//   x [Nb,H,P] float, dt [Nb,H] float, A [H] f32, B/C [Nb,G,N] float,
//   D optional [H] f32, z optional [Nb,H,P] float, dt_bias optional [H] f32,
//   out [Nb,H,P] f32/bf16.
//   state_indices optional [Nb] i32 (upstream `state_batch_indices`). LOCAL ABI:
//           index < 0 is the NULL row — its cache slot is left untouched and its
//           output row is zeroed, exactly as GdnDecode/CausalConv1dSpecUpdate
//           already model it (ops.h GdnDecode, cpu_ops.cpp GdnDecodeKernel).
//           Upstream's sentinel is `NULL_BLOCK_ID = 0`
//           (v1/attention/backends/utils.py:46) and it leaves the padded output
//           rows UNDEFINED (`continue`, mamba_kernels.hpp:147); the caller maps
//           its padding onto the local negative sentinel, and zeroing is strictly
//           more defined than what upstream promises.
void Mamba2StateUpdate(Queue& q, Tensor& out, Tensor& state, const Tensor& x,
                       const Tensor& dt, const Tensor& A, const Tensor& B, const Tensor& C,
                       const Tensor* D, const Tensor* z, const Tensor* dt_bias,
                       const Tensor* state_indices, const Mamba2Args& args);

// SILU-GATED GROUP RMS NORM — `Mixer2RMSNormGated.forward_native`
// (mamba_mixer2.py:100-149). A SIBLING of RmsNormGated, not a mode of it: that
// one is the GDN/KDA gate (sigmoid or silu) over the WHOLE row; this one always
// SILU-gates and reduces the variance over `group_size = hidden / n_groups`
// slices. Both the activation and the reduction extent differ (mamba2-ssd.md §0).
//   v      = x * silu(f32(gate))                                     (:114)
//   out    = weight * dtype(x)( v * rsqrt(mean(v^2 over its group) + eps) )
//            (:136-141 grouped, :127-131 the n_groups == 1 whole-row branch,
//             :149 the `self.weight * x.to(input_dtype)` cast point)
// x/gate/out are rank 2 [rows,Hd] or rank 3 [T,H,D]; the LAST dim is the hidden
// dim and every leading dim is a row (`*prefix_dims, hidden_dim`, :136).
// `weight` is [Hd] float, or NULLPTR for `use_rms_norm=False`, where upstream
// registers no parameter at all and returns just the gated value (:94-96,
// :115-116). args.n_groups must divide the last dim.
void RmsNormGatedGroup(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                       const Tensor* weight, const RmsNormGatedGroupArgs& args);

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

// SPECULATIVE gated-delta-rule step (SPEC-MTP I4) — the multi-token (T>1),
// slot-snapshotting variant of GdnDecode. Ported from
// vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py @ e24d1b24
// (fused_recurrent_gated_delta_rule_fwd_kernel, IS_VARLEN :66-72 +
// IS_SPEC_DECODING initial-state select :103-116 + INPLACE_FINAL_STATE
// per-timestep snapshot :156-166), as called by
// qwen_gdn_linear_attn.py:1455-1476.
//
// THE MECHANISM (why there is no rollback code anywhere). A speculative step
// runs 1+k query positions for a request, but a later verify may reject some of
// them. Instead of undoing the state, the kernel WRITES A SNAPSHOT PER
// TIMESTEP into a per-request array of k+1 state slots, and the NEXT step
// SELECTS its initial slot by the accepted count:
//
//   initial state  = state[ state_indices[i][ num_accepted_tokens[i] - 1 ] ]
//   after token t  → stored to state[ state_indices[i][t] ]      (t = 0..T-1)
//
// Rejecting at position j is therefore nothing but reading slot j next step.
// Because the recurrence is strictly sequential and slot j is written with the
// state after exactly j+1 tokens, the surviving state is BIT-IDENTICAL to
// running only the accepted prefix — no copy, no undo, no drift.
//
//   q_in/k       [num_tokens, Hk, Dk] ALREADY l2-normalized (like GdnDecode)
//   v            [num_tokens, Hv, Dv]
//   g/beta       [num_tokens, Hv] f32 (log-decay / sigmoid(b))
//   state        [num_slots, Hv, Dv, Dk] FULL persistent cache, in place
//   cu_seqlens   [num_reqs + 1] i32 — request i owns tokens
//                [cu_seqlens[i], cu_seqlens[i+1]) of the (already gathered)
//                spec token stream. Upstream spec_query_start_loc.
//   state_indices [num_reqs, num_cols] i32 ROW-MAJOR, num_cols == k+1.
//                Upstream spec_state_indices_tensor. Local ABI: an entry < 0 is
//                a NULL slot — a NULL INITIAL slot skips the request entirely
//                (output zeroed), a NULL timestep slot skips only that store.
//   num_accepted_tokens [num_reqs] i32, each in [1, num_cols].
void GdnSpecDecode(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                   const Tensor& g, const Tensor& beta, Tensor& state, const Tensor& cu_seqlens,
                   const Tensor& state_indices, const Tensor& num_accepted_tokens,
                   const GdnArgs& args);

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

// Row gather over dim 0 (torch `index_select(0, idx)`): out[i, ...] =
// in[idx[i], ...]. `idx` is i32 [M] on the same device; out is [M, D...]
// contiguous; in is [N, D...] and MAY carry an outer row stride (a padded /
// row-strided packed view), but its inner dims must be contiguous. in/out share
// dtype (any elementwise dtype). Powers the MIXED spec+non-spec GDN split
// (qwen_gdn_linear_attn.py:1334-1335,1407-1408).
void IndexSelect(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx);

// Row scatter over dim 0 (torch `index_copy_(0, idx, src)`): out[idx[i], ...] =
// in[i, ...]. `idx` is i32 [M]; in is [M, D...] contiguous; out is [N, D...]
// contiguous. Rows of out not named by idx are untouched. Merges the per-group
// GDN core outputs back into their original positions
// (qwen_gdn_linear_attn.py:1570-1571).
void IndexCopy(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx);

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
//   out[t,:] = routed_scale * sum_j weights[t,j] * expert_out[t,j,:]  (f32 accum)
//              + shared[t,:]                            (when shared != nullptr)
// expert_out [T,K,H] any float dtype (the K per-slot expert MLP outputs for
// token t), weights [T,K] f32 (router weights, §3), optional shared [T,H] any
// float dtype (the shared-expert term, §5). out [T,H] f32 or bf16. The routed
// f32 sum is stored at out's dtype; the shared term is added in that same store
// (§6 combine order: shared_output + routed_output). The activation-dtype
// rounding of the routed sum before the shared add is carried by the caller
// materializing expert_out/shared in the activation dtype.
//
// `routed_scale` is upstream's `apply_routed_scale_to_output=True` arm
// (layers/fused_moe/runner/moe_runner.py:390-407, :402-406 `fused_output *=
// routed_scaling_factor`, then :722-725 `result = shared_output + fused_output`).
// It multiplies the ROUTED sum ONLY — the shared-expert term is added unscaled,
// which is the whole point of the flag and the error a token gate catches late.
// The DEFAULT 1.0f is the `apply_routed_scale_to_output=False` polarity every
// landed caller uses, where the factor is instead folded into the router weights
// by MoeRouterTopKArgs::routed_scaling_factor (layer.py:291-300 forces the
// router's factor to 1.0 exactly when this one is not).
void MoeCombine(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                const Tensor* shared = nullptr, float routed_scale = 1.0f);

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

// --- Dense non-causal CROSS attention (LTX-2.5 L2). See AttentionCrossArgs for
// why vt::Attention cannot serve it. Per q-head h (kv-head g = h/(Hq/Hkv)),
// query i in [0,Tq), keys j in [0,S):
//   s[j] = scale * (query[i,h] · key[j,g]) + bias[i or 0, j]
//   p    = softmax_j(s)                       (f32, max-subtracted)
//   out[i,h] = Σ_j p[j] * value[j,g]
// `bias` is an OPTIONAL rank-2 additive score bias [Tq, S] or [1, S] (the
// broadcast key-only form a padding mask produces), f32, on the same device;
// nullptr means no bias. It is additive in torch SDPA's sense — upstream builds
// it as `(mask - 1) * finfo.max` for the prompt mask (transformer_args.py:204)
// and as log-space attenuation for the self-attention strength mask (:232-237),
// so a fully masked key reaches the softmax as a large negative number, NOT as
// -inf, and an all-masked row degenerates to a uniform average exactly as torch's
// does. All softmax/accumulation math is f32.
//
// BACKENDS, recorded so it cannot be discovered later: this op ships with a CPU
// kernel (`AttentionCrossKernel`, src/vt/cpu/cpu_ops.cpp) AND — since phase L8,
// 2026-08-12 — a NATIVE CUDA kernel (src/vt/cuda/cuda_attention_cross.cu). An
// earlier revision of this paragraph recorded the CUDA one as OWED "alongside the
// LTX-2.5 device-resident forward, which is the first caller that would need it";
// that caller arrived and so did the kernel, in the same change, and a CUDA
// device now has a real provider rather than a refusal or a unified-memory
// fallback to the host.
//
// The CUDA kernel is a structural port of `AttentionDenseFlashKernel`
// (src/vt/cuda/cuda_ops.cu), generalized on the three axes AttentionCrossArgs
// exists for. It uses the online-softmax recurrence where the CPU kernel uses the
// explicit three-pass max/exp/normalize, so the two agree to f32 summation-order
// slack and are NOT bit-identical — the same relationship `AttentionDenseFast`
// already has with `AttentionKernel`.
//
// A device with NO provider — kXPU, say — still refuses through `GetOp`, naming
// this op via `vt::OpName` rather than falling back. Callers route on what a call
// MEANS, not on whether its numbers happen to be square, so that refusal is
// deterministic per call site instead of per prompt length — see Ltx2Attention
// (src/vllm/model_executor/models/ltx2.cpp).
void AttentionCross(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                    const Tensor& value, const Tensor* bias, const AttentionCrossArgs& args);

// --- Conformer / FastConformer audio-encoder kernels -------------------------
// Spike: .agents/specs/parakeet-conformer-encoder.md (rows P1/P2/P3). Upstream
// mirror: transformers 5.3.0 transformers/models/parakeet/modeling_parakeet.py,
// which is what vLLM itself runs (vllm/model_executor/models/parakeet.py:37,62
// delegate to `from transformers import ParakeetEncoder`); the structurally
// identical vLLM-NATIVE conformer at
// vllm/model_executor/models/conformer_encoder.py is cited alongside where it
// agrees, and its one arithmetic divergence is a flag (AttentionRelPosArgs).
//
// P1 — torch `nn.Conv2d`, the encoder front end.
//   out    [N, Cout, Hout, Wout]
//   x      [N, Cin,  Hin,  Win ]
//   weight [Cout, Cin/groups, KH, KW]   (torch's exact parameter layout)
//   bias   optional rank-1 [Cout]
// Hout = (Hin + 2*pad_h - dilation_h*(KH-1) - 1)/stride_h + 1, likewise Wout.
// `groups` must divide both Cin and Cout; output channel oc reads input group
// oc/(Cout/groups). Zero padding: taps outside [0,Hin)x[0,Win) are SKIPPED.
// Per output element the reduction is a SINGLE f32 accumulator walked strictly
// in (ic, kh, kw) order with the bias added LAST, so the result does not depend
// on the thread count and is byte-reproducible; the in-test scalar reference in
// tests/vt/test_ops_conv2d.cpp gates exactly that order.
//
// This is the op `ParakeetEncoderSubsamplingConv2D` (modeling_parakeet.py:357)
// is built from: a dense 1->C k/stride/pad conv (:369-371), then per extra
// stage a DEPTHWISE C->C conv (groups=C, :377-386) and a POINTWISE 1x1 conv
// (:388). vLLM's native `Conv2dSubsampling` (conformer_encoder.py:18) is the
// same stack. It also REPLACES the host `std::vector<float>` Conv2dK3S2P1 loop
// in src/vllm/model_executor/models/gemma4_audio.cpp:92, which stays as an
// independent correctness reference for that model's prefix.
// x/weight/bias f32/f16/bf16; out f32/f16/bf16; all math in f32.
void Conv2d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
            const Conv2dArgs& args);

// --- General 3-D convolution (LTX25-DEVICE-RESIDENCY W5, #1007) --------------
// Spec: .agents/specs/ltx25-device-residency.md `## W5`. Upstream mirror:
// torch `nn.Conv3d` as `CausalConv3d` instantiates it — Lightricks/LTX-2 @
// fd4ded7f2, packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:292-302,
// called once at :312, which is the WHOLE of the LTX-2.5 video VAE's arithmetic.
//
//   out    [Cout, Tout, Hout, Wout]
//   x      [Cin,  Tin,  Hin,  Win ]
//   weight [Cout * Cin/groups, KT, KH, KW]
//   bias   optional rank-1 [Cout]
// Tout = (Tin + 2*pad_t - dilation_t*(KT-1) - 1)/stride_t + 1, likewise H and W.
// `groups` must divide both Cin and Cout; output channel oc reads input group
// oc/(Cout/groups). Zero padding: taps outside the input extent are SKIPPED.
//
// BATCH 1, DELIBERATELY. torch's tensor is [N, Cin, T, H, W] and this one drops
// the N axis, because `vt::Tensor` caps rank at 4 (include/vt/tensor.h:12) and
// raising that cap changes the one struct every op in the tree passes. Batch 1
// is what the video VAE decodes. A caller that needs N > 1 gets a refusal by
// name from the wrapper, never a silently folded axis.
//
// THE WEIGHT'S TWO LEADING AXES ARE MERGED, for the same rank reason: torch's
// [Cout, Cin/groups, KT, KH, KW] is rank 5, and [Cout * Cin/groups, KT, KH, KW]
// is the same bytes in the same order. `Cout` comes from `out` and `Cin/groups`
// from `x` and `args.groups`, so nothing is lost and the wrapper CHECKS the
// product rather than trusting it.
//
// ACCUMULATION ORDER — CONTRACT, NOT DETAIL, AND NOT kConv2d's. Per output
// element: one f32 accumulator seeded with the bias, then ONE f32 PARTIAL PER
// INPUT CHANNEL walked strictly in (kt, kh, kw) order and added into it. That is
// what torch's f32 convolution does — it is a blocked GEMM and sums one partial
// per channel block — and it is what every committed LTX-2.5 video VAE golden
// was taken through. kConv2d's single flat accumulator is a DIFFERENT number:
// src/vllm/model_executor/models/ltx2_video_vae.cpp measures the flat order
// pushing the non-causal tiled golden to 5.00679e-06 against a 5e-06 tolerance.
// The two ops are siblings for exactly the reason kConv1d and kDepthwiseConv1d
// are, and neither may be re-pointed at the other.
//
// The order is why the CPU and CUDA arms are BYTE-IDENTICAL rather than close:
// the device kernel walks the same sweep with __fmul_rn/__fadd_rn, and the host
// side is already pinned against FMA contraction by -ffp-contract=off
// (CMakeLists.txt). tests/vt/test_ops_conv3d.cpp gates the order against an
// independent in-test scalar reference.
//
// x/weight/bias f32/f16/bf16; out f32/f16/bf16; all math in f32.
void Conv3d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
            const Conv3dArgs& args);

// P2 — NON-CAUSAL depthwise `nn.Conv1d(C, C, K, stride, padding, groups=C)`, the
// conformer convolution module's temporal mixer.
//   out    [N, C, Lout]
//   x      [N, C, Lin]
//   weight [C, 1, K] (torch's depthwise parameter shape) or [C, K]
//   bias   optional rank-1 [C]
// Lout = (Lin + 2*padding - dilation*(K-1) - 1)/stride + 1. Zero padding: taps
// outside [0,Lin) are SKIPPED. Per output element a SINGLE f32 accumulator over
// k in increasing order, bias added LAST — thread-count independent and
// byte-reproducible, gated in tests/vt/test_ops_conv1d_depthwise.cpp.
//
// Upstream: `ParakeetEncoderConvolutionModule.depthwise_conv`
// (modeling_parakeet.py:116, constructed :138-146 with padding=(K-1)//2 :136,
// applied :180); vLLM native `ConformerConvolution.depthwise_conv`
// (conformer_encoder.py:229). DELIBERATELY a sibling of, never a modification
// of, vt::CausalConv1dFwd (ops.h kCausalConv1dFwd): that op is causal, carries a
// persistent conv_state across steps and folds a SiLU; this one is centre-padded,
// stateless and activation-free, exactly as the conformer module wants.
// x/weight/bias f32/f16/bf16; out f32/f16/bf16; all math in f32.
void DepthwiseConv1d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, const DepthwiseConv1dArgs& args);

// --- BigVGAN / DAC vocoder 1-D convolutions (#672, spec
// .agents/specs/minimax-music3.md §11.4). ------------------------------------
//
// These two are the whole convolution vocabulary of `vllm::vocoder1d`, the
// shared 1-D BigVGAN core that MiniMax-Music3's vocoder, MiniMax-H3's audio VAE,
// LTX-2.5's audio VAE and IndexTTS-2.5 all decode through
// (include/vllm/model_executor/models/vocoder1d.h). The transposed one is 88.5 %
// of MiniMax-Music3's acoustic-half profile and `vt` had no op of any kind that
// could express it, which is why the whole stage had no device kernel to route
// to.
//
// Upstream mirror: `torch.nn.functional.conv1d` / `conv_transpose1d` as the
// checkpoints instantiate them —
//   * MiniMax-Music3 `minimax_music3_vocoder.py:42,44,55,89,98`
//     (`nn.Conv1d` at :42/:44/:89/:98, `nn.ConvTranspose1d` at :55),
//   * LTX-2.5 `audio_vae/vocoder.py:104-184` (the alias-free resample pair) and
//     its BigVGAN conv_pre/conv_post,
//   * MiniMax-H3's DAC audio VAE (`dac_alias_free_resample.py`).
//
//   Conv1d              out    [N, Cout, Lout]
//                       x      [N, Cin,  Lin ]
//                       weight [Cout, Cin/groups, K]        (torch's layout)
//                       bias   optional rank-1 [Cout]
//     Lout = (Lin + 2*padding - dilation*(K-1) - 1)/stride + 1
//
//   ConvTranspose1d     out    [N, Cout, Lout]
//                       x      [N, Cin,  Lin ]
//                       weight [Cin, Cout/groups, K]        (torch's layout —
//                              note dim 0 is the INPUT channel here)
//                       bias   optional rank-1 [Cout]
//     Lout = (Lin-1)*stride - 2*padding + dilation*(K-1) + 1 + output_padding
//
// `groups` must divide both Cin and Cout. Zero padding is realised by SKIPPING
// out-of-range taps.
//
// THE NUMERIC CONTRACT, which is the load-bearing part.
//
// (1) Every output element owns ONE f32 accumulator. Not f64. That is what
//     torch accumulates a float convolution in, MEASURED rather than read: on a
//     27-tap `[+1e8, 0.1 x 25, -1e8]` probe, where an f32 accumulator lands on
//     exactly 0.0 in ANY summation order and an f64 one lands near 2.5,
//     `F.conv1d` returns 0.0 at f32 and at bf16 and `F.conv_transpose1d`
//     returns 0.0 at f32 (torch 2.11.0+cu130). vLLM owns neither op at the
//     parity pin `555967922` — it drops the vocoder it would otherwise own,
//     `qwen3_omni_moe_thinker.py:1975` — so torch is the reference here through
//     the per-consumer secondary oracles, and where vLLM DOES own a convolution
//     it states the same polarity itself (`csrc/cpu/mamba_kernels.hpp`,
//     "Accumulate in float32 for precision").
//
//     THIS WAS f64 UNTIL VT-CONV1D-F32-ACC (#1474), justified by a claim that
//     did not hold. The host reference these replaced did accumulate in double
//     (`src/vllm/model_executor/models/vocoder1d.cpp` @ 8fa405bb7), but the
//     goldens were never taken through it at that width: all four consumers'
//     generators run torch in f32 (`scripts/gen-bigvgan-goldens.py:48` builds
//     f64 and then calls `.float()`; `gen-ltx2-vae-goldens.py:223,234` and
//     `gen-minimax-music3-acoustic-goldens.py:81,134` cast with
//     `astype(np.float32)`), so this op was WIDER than the oracle its own
//     goldens came from. The clause also cited `tests/parity/goldens/`, a
//     directory that contains no vocoder, BigVGAN, LTX-2.5 VAE, FVQ or
//     general-conv1d golden at all — they are `.inc` headers beside their tests
//     (`tests/vllm/models/bigvgan_goldens.inc`, `ltx2_vae_goldens.inc`,
//     `minimax_music3_acoustic_goldens.inc`). An uncheckable citation is how
//     the first claim survived. Narrowing moved the port TOWARD its goldens:
//     over 194 arms, 182 unchanged, 10 improved, 2 one ULP worse and three or
//     more decimal orders inside their bounds.
//     .agents/specs/vt-conv1d-f32-accumulator.md.
//
// (2) THE VISIT ORDER IS PINNED, not merely the value.
//     Conv1d accumulates over (ic ascending, k ascending) with the bias seeded
//     FIRST — `acc = bias; for ic: for k: acc += x*w`.
//     ConvTranspose1d accumulates over (ic ascending, then input position t
//     ascending, taking the single tap k with `t*stride + k*dilation == p`) with
//     the bias added LAST. That is the sequence of additions the host scatter
//     performed into each destination cell, which is what lets the gather-form
//     kernels here be BIT-IDENTICAL to each other rather than merely close. The
//     ORDER is the host loop's; since #1474 the WIDTH is not, so the identity
//     holds between the providers here and not against that loop.
//
// (3) ConvTranspose1d SKIPS an input whose value compares equal to 0.0, exactly
//     as the host loop did. That is not an optimisation that may be dropped: it
//     decides the sign of a zero output cell, because (-0.0) + (+0.0) == +0.0
//     while (-0.0) alone is -0.0.
//
// Parallelism partitions OUTPUT elements only, so results do not depend on the
// thread count or the launch geometry. Gated in
// tests/vt/test_ops_conv1d_general.cpp and tests/vllm/models/test_host_parallel.cpp
// against in-test serial references that walk the declared order. Those
// references were verbatim copies of the pre-op host loop until #1474 narrowed
// them in lockstep with the kernels, so what they now assert is that the ORDER
// did not move, not that the arithmetic matches a historical loop. Clause (1)
// is what gates the WIDTH, and it is gated against torch's own answer rather
// than against a copy of ourselves — `tests/vllm/models/test_host_parallel.cpp`,
// `vocoder1d Conv1d / ConvTranspose1d accumulates in f32, which is what torch
// does`, entering through the production `vllm::vocoder1d::*` entry point.
//
// (4) THE CUDA PROVIDER IS BYTE-IDENTICAL TO THE CPU ONE, not merely close.
//     Both are one f32 accumulator per output element walked in the order
//     above; the host is compiled `-ffp-contract=off` (CMakeLists.txt:40-56)
//     and the device kernel uses `__fmul_rn`/`__fadd_rn`, so every operation on
//     both arms is an IEEE single multiply or add with round-to-nearest-even on
//     the same values in the same sequence. The gate asserts `memcmp` equality,
//     not a tolerance. **UNVERIFIED at the narrowed width**: #1474 had no CUDA
//     toolkit and no lease, so this clause rests on the construction and not on
//     a run. Owed, and named as owed:
//     .agents/specs/vt-conv1d-f32-accumulator.md §7.
//
// x/weight/bias/out are f32 only. f16/bf16 are REFUSED with a message naming
// the gap rather than silently widened — no consumer has them and no golden
// covers them (owed: .agents/specs/minimax-music3.md §11.4).
void Conv1d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
            const Conv1dArgs& args);

void ConvTranspose1d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, const ConvTranspose1dArgs& args);

// Output length for the given input length and args — the single definition of
// torch's shape arithmetic, so a caller sizing an output buffer and the op
// validating it can never disagree. Returns <= 0 when the geometry is empty.
int64_t Conv1dOutLength(int64_t in_len, int64_t kernel, const Conv1dArgs& args);
int64_t ConvTranspose1dOutLength(int64_t in_len, int64_t kernel, const ConvTranspose1dArgs& args);

// P3 — Transformer-XL relative-position ENCODER self-attention. No KV cache, no
// paging, no RoPE, non-causal: every existing vt attention path is a decoder
// path and none of them can express this.
//   out      [T, Hq,  D]
//   query    [T, Hq,  D]   raw q projection (bias_u/bias_v added here)
//   key      [T, Hkv, D]   Hq a multiple of Hkv (GQA broadcast, as vt::Attention)
//   value    [T, Hkv, D]
//   rel_key  [P, Hq,  D]   P == 2*T-1, the projected relative-position keys
//   bias_u   optional [Hq, D]  global CONTENT bias   (term (c))
//   bias_v   optional [Hq, D]  global POSITION bias  (term (d))
//   key_mask optional rank-1 [T] i8/i32, 1 = valid key, 0 = masked to -inf
// Per q-head h (kv-head g = h/(Hq/Hkv)), query i, key j:
//   ac = Σ_d (query[i,h,d] + bias_u[h,d]) * key[j,g,d]              (a)+(c)
//   bd = Σ_d (query[i,h,d] + bias_v[h,d]) * rel_key[T-1-i+j, h, d]  (b)+(d)
//   s[j] = scale*ac + scale*bd     (or (ac+bd)*scale, see scale_after_sum)
//   p    = softmax_j(s)            (f32, max-subtracted)
//   out[i,h] = Σ_j p[j] * value[j,g]
// The `T-1-i+j` index IS the `_rel_shift` of both upstreams, closed-form. Proof:
// _rel_shift left-pads the [T, 2T-1] matrix_bd by one column, reinterprets it as
// [2T, T], drops the first row and reinterprets as [T, 2T-1]; element (i,p) then
// carries flat index i*(2T-1)+p+T, whose (row, col) in the [T, 2T] padded view is
// (i, T-i+p) because 1 <= T-i+p <= 2T-1 for all i,p in [0,T). Column c>=1 of the
// padded view is original column c-1, so shifted(i,p) = raw(i, T-1-i+p); the
// `[..., :T]` truncation then leaves p == j. HF (modeling_parakeet.py:349-355,
// truncated :320) and vLLM native (conformer_encoder.py:179-186, truncated to
// T2//2+1 == T) perform the identical map, so the closed form serves both.
//
// Deviation, recorded: upstream MATERIALIZES query+bias_u and query+bias_v as
// activation-dtype tensors (:295-300 / conformer_encoder.py:205-206); we add the
// bias inside the kernel in f32, which is at least as accurate but rounds
// differently for a bf16/f16 activation. Pass bias_u/bias_v == nullptr and a
// pre-biased query to reproduce upstream's rounding exactly.
//
// A query row whose every key is masked yields ZEROS rather than upstream's NaN
// (softmax over an all -inf row); such a row is itself padding and its output is
// discarded. Recorded deviation, asserted in tests/vt/test_ops_attn_relpos.cpp.
// Reductions are strictly sequential per output element => thread-count
// independent and byte-reproducible. f32/f16/bf16 in, f32/f16/bf16 out; all
// softmax and accumulation math in f32.
void AttentionRelPos(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                     const Tensor& value, const Tensor& rel_key, const Tensor* bias_u,
                     const Tensor* bias_v, const Tensor* key_mask,
                     const AttentionRelPosArgs& args);

// Same contract and math as Attention (dense full/causal, GQA broadcast, f32
// online-softmax), but the CUDA impl is WARP-scoped (one warp per (query,head),
// __shfl head_dim reduction, register accumulator, no __syncthreads) — the fast
// path for small head_dim dense attention (the Qwen3-VL vision tower). NOT
// bit-identical to Attention (different head_dim partial-sum grouping), same f32
// math; adopt per token-exact gate. On CPU it dispatches to the SAME kernel as
// Attention (byte-identical there). Separate op so kAttention stays untouched.
void AttentionDenseFast(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                        const Tensor& value, const AttentionArgs& args);

// Same contract and math as AttentionDenseFast, but the CUDA impl is FLASH-TILED:
// a block of query-warps SHARES each streamed K/V tile out of shared memory (classic
// FlashAttention K/V tiling, ported in structure from src/vt/cuda/flash_attn/src/
// flash_fwd_kernel.h), eliminating the O(t^2) redundant global K/V reads that make
// AttentionDenseFast memory-bound on long non-causal contexts (the Whisper audio
// encoder, 1500 frames × 32 layers). The per-warp online-softmax arithmetic and its
// order are UNCHANGED from AttentionDenseFast, so the CUDA output is BIT-IDENTICAL to
// it (K/V bytes merely sourced from shared memory) ⇒ token-identical by construction.
// On CPU it dispatches to the SAME kernel as Attention (byte-identical). Separate op
// so kAttention / kAttentionDenseFast stay untouched.
void AttentionDenseFlash(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                         const Tensor& value, const AttentionArgs& args);

// Same contract as AttentionDenseFlash, but the CUDA impl runs the VENDORED
// FlashAttention-2 forward (src/vt/cuda/flash_attn/) on its tensor cores instead of a
// scalar per-warp recurrence — the kernel vLLM itself dispatches for dense non-causal
// encoder self-attention (vllm/model_executor/models/whisper.py
// WhisperEncoderAttention:255 -> forward:298-317 -> flash_attn_varlen_func).
//
// NOT bit-identical to AttentionDenseFast/Flash: `mma.sync` reassociates the QK^T and
// PV reductions, so results differ within the bf16 envelope and adoption is gated on
// the token-exact / ratified near-tie gate, never assumed. The FA-2 fast path applies
// only to bf16, head_dim 64, non-causal, MHA (h_k == h) on CUDA with the vendored
// kernels compiled; every other shape falls back to kAttentionDenseFlash, which is why
// this op is safe to call generically. On CPU it dispatches to the SAME kernel as
// Attention. Separate op so kAttention / kAttentionDenseFast / kAttentionDenseFlash
// stay untouched.
void AttentionDenseFa2(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                       const Tensor& value, const AttentionArgs& args);

// DFlash in-block attention (SPEC-DFLASH D2, DF-DRAFT-MODEL) — the project's FIRST
// non-causal / bidirectional attention primitive. See DFlashBlockAttentionArgs for
// the full contract. Per request block [cu_seqlens[r], cu_seqlens[r+1]), each query
// attends to keys IN THAT BLOCK: BIDIRECTIONAL when args.causal is false (the
// DFlash full-attention layers, qwen3_dflash.py _resolve_layer_attention -> causal
// False), causal-within-window when args.causal is true (the SWA layers). f32
// online softmax, GQA broadcast. A SEPARATE op from vt::Attention/PagedAttention so
// the causal decoder path used by every other model is byte-identical. CUDA impl
// mirrors AttentionKernel's block-reduction recurrence; CPU is the reference.
void DFlashBlockAttention(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                          const Tensor& value, const DFlashBlockAttentionArgs& args);

// DFlash PAGED in-block attention (SPEC-DFLASH D12 Part B) — the capture-safe form
// of DFlashBlockAttention. See DFlashPagedBlockAttentionArgs. Tensors:
//   out          [Nq, Hq, D]           block-query outputs (Nq = sum of block lens)
//   query        [Nq, Hq, D]           the (1+k)*num_reqs block queries
//   block_key    [Nq, Hkv, D]          the block K (same rows as query)
//   block_value  [Nq, Hkv, D]          the block V
//   ctx_key      [num_pages, block_size, Hkv, D]   PAGED context K (normed+RoPE'd)
//   ctx_value    [num_pages, block_size, Hkv, D]   PAGED context V (raw)
//   cu_seqlens   [num_reqs+1] i32       block-query row bounds (device)
//   seq_lens     [num_reqs]   i32       per-request context length C_r (device)
//   block_table  [num_reqs, max_pages] i32  logical-page -> physical-page (device)
// Every metadata tensor is read in place on the op's device (device for CUDA =
// capture-safe; host for CPU). GQA broadcast; f32 online softmax. Bit-identical to
// DFlashBlockAttention over the materialized [context; block] combined buffer.
void DFlashPagedBlockAttention(Queue& q, Tensor& out, const Tensor& query,
                               const Tensor& block_key, const Tensor& block_value,
                               const Tensor& ctx_key, const Tensor& ctx_value,
                               const Tensor& cu_seqlens, const Tensor& seq_lens,
                               const Tensor& block_table,
                               const DFlashPagedBlockAttentionArgs& args);

// DFlash2 grouped dynamic depthwise convolution (SPEC-DFLASH2 W2, #1314). See
// DFlashGroupedConvArgs for the contract and the upstream anchor. Tensors:
//   out           [T, H]                       the convolved sublayer stream
//   x             [T, H]                       the sublayer input/output stream
//   coefficients  [T, sides, taps, num_groups] the per-position kernel DELTAS
//   base          [sides, taps, H]             the checkpoint's base_kernel
// with H == num_groups * group_size and `args.side` in [0, sides). All four share
// one float dtype (bf16 on every published checkpoint). CPU is the authoritative
// reference; CUDA mirrors it bit-for-bit.
void DFlashGroupedConv(Queue& q, Tensor& out, const Tensor& x, const Tensor& coefficients,
                       const Tensor& base, const DFlashGroupedConvArgs& args);

// DFlash2 candidate-selector edge lattice (SPEC-DFLASH2 W3, #1314). See
// Dflash2SelectorEdgesArgs for the contract and the upstream anchor. Tensors:
//   scores          [B, L, K, K] f32   edge(b,l,predecessor,child)
//   pred_codebook   [V, R]             candidate_selector.predecessor_codebook
//   succ_codebook   [V, R]             candidate_selector.successor_codebook
//   candidate_ids   [B, L, K] i64      compute_candidates' ids (target vocab)
//   unary           [B, L, K] f32      compute_candidates' values
//   hidden          [B, L, R]          hidden_projection(final hidden states)
//   anchors         [B] i64            each request's verified anchor token
// The two codebooks and `hidden` share one float dtype (bf16 on every published
// checkpoint); `scores` and `unary` are always f32. CPU is the authoritative
// reference.
void Dflash2SelectorEdges(Queue& q, Tensor& scores, const Tensor& pred_codebook,
                          const Tensor& succ_codebook, const Tensor& candidate_ids,
                          const Tensor& unary, const Tensor& hidden, const Tensor& anchors,
                          const Dflash2SelectorEdgesArgs& args);

// DFlash2 candidate-selector PATH WALK (SPEC-DFLASH2 W4, #1314). See
// Dflash2PathWalkArgs for the contract, the tie-break and the upstream anchor.
// Tensors:
//   tokens        [B, L] i64        the drafted token per (request, step)
//   scores        [B, L, K, K] f32  vt::Dflash2SelectorEdges' output
//   candidate_ids [B, L, K] i64     compute_candidates' ids (target vocab)
// One request is one program: the step loop is INSIDE it, because step l reads
// the predecessor row step l-1 chose.
void Dflash2PathWalk(Queue& q, Tensor& tokens, const Tensor& scores,
                     const Tensor& candidate_ids, const Dflash2PathWalkArgs& args);

// Top-k that EMITS the surviving (id, value) pairs (SPEC-DFLASH2 W3 / D2,
// #1314). See TopKValuesIndicesArgs for the contract, the tie-break and the
// upstream anchor. Tensors:
//   values   [rows, k] f32   the k largest logits of each row, DESCENDING
//   indices  [rows, k] i64   their column indices, ties broken ASCENDING
//   logits   [rows, V] f32   read-only (unlike vt::ApplyTopKTopP, which masks
//                            its input in place)
// `args.k` must be in [1, V - args.num_org_vocab_padding].
void TopKValuesIndices(Queue& q, Tensor& values, Tensor& indices, const Tensor& logits,
                       const TopKValuesIndicesArgs& args);

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

// --- fp8 KV-cache write (KV-FP8 W1). The fp8 sibling of ReshapeAndCache: the
// K/V cache pages are 1-byte fp8 (DType::kI8 storage, `kind` = the fp8
// interpretation) and each element is quantized as Quantize(hp / k_scale) /
// Quantize(hp / v_scale). Mirrors the fp8 branch of vLLM reshape_and_cache_flash
// (csrc/libtorch_stable/cache_kernels.cu:314-401 + CopyWithScaleOp :241-252 +
// the fp8::scaled_convert scale convention, quant_utils.cuh:296-308) @ pin
// 555967922. k_scale/v_scale are the per-tensor scales BaseKVCacheMethod loads
// from the checkpoint (kv_cache.py:108-191); both must be > 0. Same shape/stride
// contract as ReshapeAndCache; the ONLY difference is the fp8 store. CPU-only in
// W1 (the CUDA fp8-KV store kernel is a named later brick); kFp8E5M2 CPU compute
// is likewise a later brick.
void ReshapeAndCacheFp8(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                        Tensor& v_cache, const Tensor& slot_mapping, Fp8KVCacheDataType kind,
                        float k_scale, float v_scale);

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

// --- Greedy spec-decode rejection sampling (SPEC-REJECTION I3) --------------
// Ported from vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py @ e24d1b24:
// the `is_greedy` branch of `_rejection_kernel` (:564-585 accept + :628 store of
// the accepted length), plus the greedy short-circuits of `_resample_kernel`
// (:846-849) and `_insert_resampled_kernel` (:828-841 num_sampled += 1 and the
// bonus-token argmax insert). The per-row `target_argmax` this needs is upstream's
// `_compute_global_target_argmax` over `_compute_local_logits_stats_kernel`'s
// per-vocab-block partials (:923-946).
//
// THE GREEDY ACCEPT RULE (the whole correctness story — see the header note in
// include/vllm/v1/spec_decode/rejection_sampler.h):
//   For request r the expanded logit rows are [cu[r], cu[r+1]); k_r = cu[r+1] -
//   cu[r] - 1 draft positions plus one bonus position. Walking i = 0..k_r-1 while
//   still accepting:
//     * `target_argmax = argmax(logits[cu[r]+i])`
//     * `draft = draft_sampled[cu[r]+i+1]`   (NOTE the +1: draft token i is the
//       INPUT id at the NEXT expanded row, mirroring :534)
//     * accept iff `draft == target_argmax`; store `draft` when accepted, else
//       store `target_argmax` (the replacement) and STOP accepting.
//   If all k_r accepted, store `argmax(logits[cu[r]+k_r])` (the bonus).
//   num_sampled[r] = accepted_length + 1 (always ≥ 1: a rejection still emits the
//   target argmax at the mismatch position).
// A placeholder draft id of -1 can never equal an argmax (≥ 0), so it is rejected
// with no out-of-bounds read — mirroring the upstream `-1` padding contract.
//
//   logits        [num_logits, vocab] f32   the EXPANDED verify logits
//   draft_sampled [num_logits] i32          input ids gathered at logits_indices
//   cu_num_logits [num_reqs + 1] i32        per-request expanded-row offsets
//   sampled       [num_reqs, max_num_logits] i32  OUT; positions past
//                 num_sampled[r] are filled with -1 (PLACEHOLDER_TOKEN_ID,
//                 vllm/v1/sample/rejection_sampler.py:30). Upstream leaves them
//                 uninitialized (`new_empty`); we fill so the buffer is
//                 deterministic and the ported legacy-sampler assertions read
//                 directly. Recorded deviation.
//   num_sampled   [num_reqs] i32            OUT; accepted_length + 1
//
// Argmax tie-break is LOWEST INDEX (torch.argmax), identical to vt::GreedyArgmax,
// so a k=0 request reduces EXACTLY to the non-speculative greedy sampler.
void GreedyRejectionSample(Queue& q, Tensor& sampled, Tensor& num_sampled,
                           const Tensor& logits, const Tensor& draft_sampled,
                           const Tensor& cu_num_logits);

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

// In-place per-output-column scale: x[m,n] *= col[n], with x an F32 or BF16
// [M,N] (row-major, inner-contiguous rows; row stride may be padded) and col an
// F32 [N] contiguous broadcast vector. The load-time-free realization of a merged
// per-tensor-fp8 projection's per-shard dequant: one fp8 GEMM over the
// N-concatenated weight is run with alpha=1 (raw f32 accumulation), then this
// applies each output column's folded scalar (= input_scale * that shard's
// weight_scale) in f32 — byte-identical to the separate per-shard GEMMs when the
// GEMM's accumulation matches (the alpha multiply is the same IEEE f32 op cuBLASLt
// would fold). CPU + CUDA. (Mirrors the fp4 merge's per-column block-scale
// concatenation, qwen3_5.cpp ResidentNvfp4Qkv.)
//
// The MULTIPLY is f32 on both x dtypes, and `col` is always f32 — x's dtype is
// the STORE width only. A bf16 x (PERF-FP8-ALPHA-FOLD / #417) halves the bytes
// this read-modify-write moves, which is its whole cost: it is bandwidth-bound,
// measured at 209.5 GB/s = 77% of the GB10's peak over the merged FP8 GDN
// in_proj output. It also rounds the product, so the byte-identity above holds
// only for the f32 arm; a bf16 x is the vLLM-faithful width (its fp8 linear
// emits the model dtype) but is a real value change and is opt-in at the caller.
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
