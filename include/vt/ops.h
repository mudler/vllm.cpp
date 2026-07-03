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

void RegisterOp(OpId op, DeviceType device, void* fn);
void* GetOp(OpId op, DeviceType device);

// Contract: out must not alias any input tensor (RopeNeox is in-place by design).

// out[M,N] = a[M,K] @ b[K,N]; a/b float dtypes (f32/f16/bf16), out f32 or
// bf16, f32 accumulation, all contiguous, same device.
void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

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

}  // namespace vt
