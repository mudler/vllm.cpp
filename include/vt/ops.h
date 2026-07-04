// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include "vt/tensor.h"

namespace vt {

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
  kMoeRouterTopK,
  kMoeCombine,
  kAttention,
  kReshapeAndCache,
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
  kMoeGroupedGemmNvfp4,
  kMoeSiluMul,
  kCastBf16,
  kAttnGateSplit,
  kSigmoidGateBf16,
  kGdnGBeta,
  kGdnConvSplit,
  kSharedExpertGate,
  kCount
};

struct RmsNormArgs {
  float eps = 1e-6f;
  bool gemma = false;  // weight applied as (1 + w), GemmaRMSNorm style
};

struct RopeArgs {
  float base = 10000.0f;
  int rotary_dim = 0;  // <= head_dim; even
};

// GDN op args (.agents/gdn-semantics.md is the formula reference; sections
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
};

// Dense causal attention args (.agents/qwen36-forward-notes.md §5 is the
// formula reference — Qwen3NextAttention's core scaled-dot-product).
struct AttentionArgs {
  // Softmax scale, applied to the qk dot product. Upstream sets it to
  // head_dim^-0.5 (Qwen3NextAttention.scaling). Must be set explicitly (> 0).
  float scale = 0.0f;
  // Causal masking: key position j attends only when j <= query position i.
  // Always true for the M0.9 decoder path (bidirectional is a M1.6+ concern).
  bool causal = true;
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
};

// MoE router top-k args (.agents/moe-semantics.md §3 is the formula reference).
struct MoeRouterTopKArgs {
  // Number of experts selected per token (top_k = num_experts_per_tok).
  int top_k = 0;
  // renormalize = norm_topk_prob (True for Qwen3.6, moe-semantics.md §1/§3):
  // divide the k selected softmax probs by their sum (denom>0 guard).
  bool renormalize = true;
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
using MoeGroupedGemmNvfp4Fn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*, const Tensor&,
             const Tensor&, const Tensor&);
using MoeSiluMulFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
// --- Qwen3.6 elementwise "glue" ops (M0.9 forward). These replace host-side
// loops so the decode step can run entirely on-device (CUDA-graph capture).
// All math in f32; dims are inferred from the tensor shapes (no args structs).
using CastBf16Fn = void (*)(Queue&, Tensor&, const Tensor&);
using AttnGateSplitFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&);
using SigmoidGateBf16Fn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using GdnGBetaFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            const Tensor&);
using GdnConvSplitFn = void (*)(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&);
using SharedExpertGateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using RmsNormFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const RmsNormArgs&, Tensor*);
using SiluAndMulFn = void (*)(Queue&, Tensor&, const Tensor&);
using EmbeddingFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using RopeFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const RopeArgs&);
using CausalConv1dFwdFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*,
                                   Tensor&, const Tensor&, const Tensor&,
                                   const CausalConv1dArgs&);
using CausalConv1dUpdateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                      const Tensor*, Tensor&, const CausalConv1dArgs&);
using L2NormFn = void (*)(Queue&, Tensor&, const Tensor&, const L2NormArgs&);
using RmsNormGatedFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                const RmsNormGatedArgs&);
using GdnPrefillFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, Tensor&, const Tensor&,
                              const GdnArgs&);
using GdnDecodeFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, const Tensor&, Tensor&, const GdnArgs&);
using MoeRouterTopKFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const MoeRouterTopKArgs&);
using MoeCombineFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*);
using AttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const AttentionArgs&);
using ReshapeAndCacheFn = void (*)(Queue&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                   const Tensor&);
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

void RegisterOp(OpId op, DeviceType device, void* fn);
void* GetOp(OpId op, DeviceType device);

// Contract: out must not alias any input tensor (RopeNeox is in-place by design).

// out[M,N] = a[M,K] @ b[K,N]; a/b float dtypes (f32/f16/bf16), out f32 or
// bf16, f32 accumulation, all contiguous, same device.
void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

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

// out[R,I] = silu(gate[R,I]) * up[R,I]  (moe-semantics.md §4; the fused-MoE
// element-wise activation between the grouped gate/up and down GEMMs). gate/up
// f32 or bf16, out f32/bf16; silu/mul computed in f32, rounded on store. Unlike
// vt::SiluAndMul (single [T,2D] input), this takes the two separately-produced
// projections so no concat/copy is needed. CPU + CUDA.
void MoeSiluMul(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up);

// out[T,H] = x[T,H] / sqrt(mean(x^2) + eps) * w  (or *(1+w) when gemma);
// out f32 or bf16 (computed in f32, rounded on store).
// With residual != nullptr (f32 [T,H]): residual += x first (new residual
// stream), and that sum is what gets normalized (upstream fused_add_rms_norm).
// The residual stream stays f32-only by design — full precision is kept
// across layers even when out is bf16.
// Note: unlike upstream forward_native, the standard path keeps full f32 precision
// (no x.to(weight.dtype) rounding before the weight multiply); parity tests vs
// upstream bf16 need bf16-eps tolerance on the non-gemma path.
void RmsNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
             const RmsNormArgs& args, Tensor* residual = nullptr);

// out[T,D] = silu(x[:, :D]) * x[:, D:], x is [T, 2D]; out f32 or bf16.
// Note: computes in f32 (upstream forward_native computes in x's dtype); bf16 parity tests need bf16-eps tolerance.
void SiluAndMul(Queue& q, Tensor& out, const Tensor& x);

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

// --- GDN (Gated DeltaNet) ops. Formula reference: .agents/gdn-semantics.md.
// All GDN state tensors are caller-allocated f32 and updated IN PLACE
// (upstream computes states in f32 and rounds to the cache dtype on store —
// that rounding point is M0.9 layer assembly, gdn-semantics.md §1).

// Varlen causal conv over the token stream (gdn-semantics.md §2, upstream
// causal_conv1d_fn). x[T,C] token-major, weight[C,K], optional bias[C]
// (nullptr = no bias; Qwen GDN conv has bias=False), conv_state[N,C,K-1] f32
// in/out (per-sequence slices, gathered — cache_indices/NULL-block handling is
// M0.9), query_start_loc[N+1] i32 cumulative token offsets (seq s spans
// [qsl[s], qsl[s+1])), has_initial_state[N] i32 (0/1).
//   out[c,t] = act(bias[c] + sum_j w[c,j] * window[j]), window[j] = x token
//   t-(K-1-j), falling back to conv_state (if has_initial_state) or 0 for
//   tokens before the sequence start. w[:,K-1] multiplies the current token.
// State write-back: last K-1 RAW x tokens (pre-activation), left-padded with
// zeros (no init state) or shifted old state when T < K-1.
void CausalConv1dFwd(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, Tensor& conv_state, const Tensor& query_start_loc,
                     const Tensor& has_initial_state, const CausalConv1dArgs& args);

// Single-token conv step (gdn-semantics.md §3, upstream causal_conv1d_update
// seqlen==1 path). x[B,C] one token per sequence, conv_state[B,C,K-1] f32
// in/out. Read-old-then-roll:
//   out[c] = act(bias[c] + sum_j w[c,j] * [conv_state[c,:], x[c]][j])
//   conv_state[c,:] <- [conv_state[c,1:], x[c]]   (raw x)
void CausalConv1dUpdate(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                        const Tensor* bias, Tensor& conv_state, const CausalConv1dArgs& args);

// Rowwise l2 normalization over the LAST dim (gdn-semantics.md §4, upstream
// l2norm_fwd): y = x * rsqrt(sum(x^2) + eps). Plain SUM, not mean — this is
// not an rmsnorm. x/out rank 2 or 3 ([rows, D] or [T, H, D]); f32 math.
void L2Norm(Queue& q, Tensor& out, const Tensor& x, const L2NormArgs& args);

// Gated rmsnorm (gdn-semantics.md §5, upstream RMSNormGated with
// norm_before_gate=True, group_size=None, no bias):
//   var = mean(x^2 over last dim);  out = x * rsqrt(var + eps) * w * act(z)
// x/gate/out [T,D], weight [D]; act = silu (or sigmoid, args.sigmoid_gate).
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
void GdnDecode(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
               const Tensor& g, const Tensor& beta, Tensor& state, const GdnArgs& args);

// --- MoE (sparse mixture-of-experts) ops. Formula reference:
// .agents/moe-semantics.md. The expert MLP itself is NOT an op — it is composed
// in the layer/runner from Matmul + SiluAndMul (§4). These two ops cover the
// pieces composition cannot express: the router top-k/normalize and the
// weighted scatter-combine.

// Router top-k (moe-semantics.md §3, upstream topk_softmax + fused_topk).
// logits [T,E] any float dtype; softmax computed in f32 over ALL E experts,
// greedy top-k (weights emitted in descending order per token), lowest expert
// index wins ties, then optional renormalize (divide the k probs by their sum,
// denom<=0 -> 1 guard). weights [T,top_k] f32, indices [T,top_k] i32.
void MoeRouterTopK(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                   const MoeRouterTopKArgs& args);

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

// --- Dense causal attention (M0.9). Formula reference:
// .agents/qwen36-forward-notes.md §5 (pinned Qwen3NextAttention core).
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
//   for key position j in 0..p (causal) — block = block_table[r, j / block_size],
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

// Splits the fused q/gate attention projection into its two halves. qgate is
// [T, Hq*2*Dh] contiguous, laid out per (t,hq) as [q(Dh) | gate(Dh)]; q_out and
// gate_out are [T,Hq,Dh]. For t in [0,T), hq in [0,Hq):
//   q_out[t,hq,:]    = qgate row t at offset (hq*2*Dh)      .. +Dh
//   gate_out[t,hq,:] = qgate row t at offset (hq*2*Dh + Dh) .. +2*Dh
// T/Hq/Dh are inferred from q_out's shape. All f32.
void AttnGateSplit(Queue& q, Tensor& q_out, Tensor& gate_out, const Tensor& qgate);

// out[i] = F32ToBF16(attn[i] * sigmoid(gate[i])), sigmoid(x)=1/(1+exp(-x)); out
// bf16, attn/gate f32, same element count. The sigmoid output-gate applied to
// the attention result before the o_proj (elementwise on the projection split).
void SigmoidGateBf16(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate);

// Derives the GDN per-head decay g and gate beta from the raw projections
// (gdn-semantics.md §6). g_out/beta_out/araw/braw are [T,Hv]; a_log/dt_bias are
// [Hv]. For t in [0,T), hv in [0,Hv), idx=t*Hv+hv:
//   x  = araw[idx] + dt_bias[hv]
//   sp = softplus(x) = (x > 20) ? x : log1p(exp(x))    (threshold 20)
//   g_out[idx]    = -exp(a_log[hv]) * sp
//   beta_out[idx] = sigmoid(braw[idx])
// T/Hv inferred from g_out's shape. All f32.
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

// out[t,c] = F32ToBF16(sigmoid(gl[t]) * sd[t*H+c]); out bf16 [T,H], sd f32
// [T,H], gl f32 with T elements (shape [T] or [T,1]). The shared-expert
// sigmoid gate (moe-semantics.md §5), applied per token to the shared MLP
// output. T inferred from out.shape[0], H = out.Numel()/T.
void SharedExpertGate(Queue& q, Tensor& out, const Tensor& sd, const Tensor& gl);

}  // namespace vt
