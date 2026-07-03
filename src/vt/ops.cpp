// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/ops.h"

#include <array>

namespace vt {

namespace {
using OpTable =
    std::array<std::array<void*, kNumDeviceTypes>, static_cast<size_t>(OpId::kCount)>;
OpTable& Table() {
  static OpTable table{};
  return table;
}

bool IsFloat(DType d) { return d == DType::kF32 || d == DType::kF16 || d == DType::kBF16; }
bool IsOutFloat(DType d) { return d == DType::kF32 || d == DType::kBF16; }
}  // namespace

void RegisterOp(OpId op, DeviceType device, void* fn) {
  VT_CHECK(static_cast<size_t>(op) < static_cast<size_t>(OpId::kCount), "invalid op id");
  VT_CHECK(static_cast<size_t>(device) < kNumDeviceTypes, "invalid device type");
  Table()[static_cast<size_t>(op)][static_cast<size_t>(device)] = fn;
}

void* GetOp(OpId op, DeviceType device) {
  VT_CHECK(static_cast<size_t>(device) < kNumDeviceTypes, "invalid device type");
  void* fn = Table()[static_cast<size_t>(op)][static_cast<size_t>(device)];
  VT_CHECK(fn != nullptr, std::string("no kernel for op ") +
                              std::to_string(static_cast<int>(op)) + " on device type " +
                              std::to_string(static_cast<int>(device)));
  return fn;
}

void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2, "matmul: rank-2 tensors required");
  VT_CHECK(a.shape[1] == b.shape[0], "matmul: inner dims mismatch");
  VT_CHECK(out.shape[0] == a.shape[0] && out.shape[1] == b.shape[1],
           "matmul: output shape mismatch");
  VT_CHECK(IsFloat(a.dtype) && IsFloat(b.dtype) && IsOutFloat(out.dtype),
           "matmul: float inputs and f32/bf16 output required");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "matmul: contiguous tensors required");
  VT_CHECK(a.device == b.device && a.device == out.device && a.device == q.device,
           "matmul: device mismatch");
  reinterpret_cast<MatmulFn>(GetOp(OpId::kMatmul, q.device.type))(q, out, a, b);
}

void RmsNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
             const RmsNormArgs& args, Tensor* residual) {
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 1, "rmsnorm: x/out rank-2, w rank-1");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1], "rmsnorm: shape mismatch");
  VT_CHECK(weight.shape[0] == x.shape[1], "rmsnorm: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsOutFloat(out.dtype),
           "rmsnorm: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "rmsnorm: contiguous required");
  if (residual != nullptr) {
    VT_CHECK(residual->dtype == DType::kF32 && residual->rank == 2 &&
                 residual->shape[0] == x.shape[0] && residual->shape[1] == x.shape[1] &&
                 residual->IsContiguous() && residual->device == x.device,
             "rmsnorm: residual must be f32 [T,H] contiguous on x's device");
  }
  VT_CHECK(x.device == out.device && weight.device == x.device && x.device == q.device,
           "rmsnorm: device mismatch (x/out/weight/queue)");
  reinterpret_cast<RmsNormFn>(GetOp(OpId::kRmsNorm, q.device.type))(q, out, x, weight, args,
                                                                    residual);
}

void SiluAndMul(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2, "silu_and_mul: rank-2 required");
  VT_CHECK(x.shape[1] % 2 == 0, "silu_and_mul: inner dim must be even");
  VT_CHECK(out.shape[0] == x.shape[0] && out.shape[1] == x.shape[1] / 2,
           "silu_and_mul: output shape mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "silu_and_mul: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "silu_and_mul: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "silu_and_mul: device mismatch");
  reinterpret_cast<SiluAndMulFn>(GetOp(OpId::kSiluAndMul, q.device.type))(q, out, x);
}

void Embedding(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids) {
  VT_CHECK(table.rank == 2 && ids.rank == 1 && out.rank == 2, "embedding: bad ranks");
  VT_CHECK(out.shape[0] == ids.shape[0] && out.shape[1] == table.shape[1],
           "embedding: output shape mismatch");
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64, "embedding: ids i32/i64");
  VT_CHECK(IsFloat(table.dtype) && IsOutFloat(out.dtype),
           "embedding: float table, f32/bf16 out");
  VT_CHECK(table.IsContiguous() && ids.IsContiguous() && out.IsContiguous(),
           "embedding: contiguous required");
  VT_CHECK(table.device == out.device && ids.device == table.device && table.device == q.device,
           "embedding: device mismatch (table/out/ids/queue)");
  reinterpret_cast<EmbeddingFn>(GetOp(OpId::kEmbedding, q.device.type))(q, out, table, ids);
}

void RopeNeox(Queue& q, Tensor& q_states, Tensor& k_states, const Tensor& positions,
              const RopeArgs& args) {
  VT_CHECK(q_states.rank == 3 && k_states.rank == 3, "rope: q/k rank-3 [T,H,D]");
  VT_CHECK(q_states.shape[0] == k_states.shape[0] && q_states.shape[2] == k_states.shape[2],
           "rope: q/k token count and head_dim must match");
  VT_CHECK(positions.rank == 1 && positions.shape[0] == q_states.shape[0],
           "rope: positions[T] mismatch");
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "rope: positions i32/i64");
  VT_CHECK(IsOutFloat(q_states.dtype) && k_states.dtype == q_states.dtype,
           "rope: q/k must be f32 or bf16, same dtype");
  VT_CHECK(args.rotary_dim > 0 && args.rotary_dim % 2 == 0 &&
               args.rotary_dim <= q_states.shape[2],
           "rope: rotary_dim must be even and <= head_dim");
  VT_CHECK(q_states.IsContiguous() && k_states.IsContiguous() && positions.IsContiguous(),
           "rope: contiguous required");
  VT_CHECK(q_states.device == q.device && k_states.device == q.device &&
               positions.device == q.device,
           "rope: device mismatch (q/k/positions/queue)");
  reinterpret_cast<RopeFn>(GetOp(OpId::kRopeNeox, q.device.type))(q, q_states, k_states,
                                                                  positions, args);
}

namespace {
// Shared shape/dtype/device validation for the two conv ops. x/out [T,C],
// weight [C,K], optional bias [C], conv_state [N,C,K-1] f32.
void CheckConvCommon(const Queue& q, const Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, const Tensor& conv_state, const char* name) {
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 2 && conv_state.rank == 3,
           std::string(name) + ": x/out [T,C], weight [C,K], conv_state [N,C,K-1]");
  const int64_t c = x.shape[1], k = weight.shape[1];
  VT_CHECK(out.shape[0] == x.shape[0] && out.shape[1] == c,
           std::string(name) + ": out shape must match x");
  VT_CHECK(weight.shape[0] == c, std::string(name) + ": weight channel dim mismatch");
  VT_CHECK(k >= 1, std::string(name) + ": kernel width must be >= 1");
  VT_CHECK(conv_state.shape[1] == c && conv_state.shape[2] == k - 1,
           std::string(name) + ": conv_state must be [N,C,K-1]");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsOutFloat(out.dtype),
           std::string(name) + ": float x/weight, f32/bf16 out");
  VT_CHECK(conv_state.dtype == DType::kF32,
           std::string(name) + ": conv_state must be f32 (in/out, in place)");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous() &&
               conv_state.IsContiguous(),
           std::string(name) + ": contiguous required");
  VT_CHECK(x.device == q.device && out.device == q.device && weight.device == q.device &&
               conv_state.device == q.device,
           std::string(name) + ": device mismatch (x/out/weight/conv_state/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 1 && bias->shape[0] == c && IsFloat(bias->dtype) &&
                 bias->IsContiguous() && bias->device == q.device,
             std::string(name) + ": bias must be float [C] contiguous on the queue device");
  }
}

// Shared validation for the two delta-rule ops. q_in/k [T,Hk,Dk], v [T,Hv,Dv],
// g/beta [T,Hv] f32, state [N,Hv,Dv,Dk] f32, out [T,Hv,Dv].
void CheckGdnCommon(const Queue& q, const Tensor& out, const Tensor& q_in, const Tensor& k,
                    const Tensor& v, const Tensor& g, const Tensor& beta, const Tensor& state,
                    const GdnArgs& args, const char* name) {
  VT_CHECK(q_in.rank == 3 && k.rank == 3 && v.rank == 3 && out.rank == 3 && g.rank == 2 &&
               beta.rank == 2 && state.rank == 4,
           std::string(name) +
               ": q/k [T,Hk,Dk], v/out [T,Hv,Dv], g/beta [T,Hv], state [N,Hv,Dv,Dk]");
  const int64_t t = q_in.shape[0], hk = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv = v.shape[1], dv = v.shape[2];
  VT_CHECK(k.shape[0] == t && k.shape[1] == hk && k.shape[2] == dk,
           std::string(name) + ": k shape must match q");
  VT_CHECK(v.shape[0] == t, std::string(name) + ": v token count must match q");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hv && out.shape[2] == dv,
           std::string(name) + ": out must be [T,Hv,Dv]");
  VT_CHECK(g.shape[0] == t && g.shape[1] == hv && beta.shape[0] == t && beta.shape[1] == hv,
           std::string(name) + ": g/beta must be [T,Hv]");
  VT_CHECK(hk >= 1 && hv % hk == 0,
           std::string(name) + ": Hv must be a multiple of Hk (GQA broadcast)");
  VT_CHECK(state.shape[1] == hv && state.shape[2] == dv && state.shape[3] == dk,
           std::string(name) + ": state must be [N,Hv,Dv,Dk]");
  VT_CHECK(IsFloat(q_in.dtype) && IsFloat(k.dtype) && IsFloat(v.dtype) &&
               IsOutFloat(out.dtype),
           std::string(name) + ": float q/k/v, f32/bf16 out");
  VT_CHECK(g.dtype == DType::kF32 && beta.dtype == DType::kF32,
           std::string(name) + ": g/beta must be f32 (upstream keeps them f32)");
  VT_CHECK(state.dtype == DType::kF32,
           std::string(name) + ": state must be f32 (in/out, in place)");
  VT_CHECK(q_in.IsContiguous() && k.IsContiguous() && v.IsContiguous() && out.IsContiguous() &&
               g.IsContiguous() && beta.IsContiguous() && state.IsContiguous(),
           std::string(name) + ": contiguous required");
  VT_CHECK(q_in.device == q.device && k.device == q.device && v.device == q.device &&
               out.device == q.device && g.device == q.device && beta.device == q.device &&
               state.device == q.device,
           std::string(name) + ": device mismatch (q/k/v/out/g/beta/state/queue)");
  VT_CHECK(args.scale > 0.0f, std::string(name) + ": args.scale must be set (> 0)");
}

void CheckI32Meta(const Queue& q, const Tensor& t, int64_t expect_len, const char* name,
                  const char* what) {
  VT_CHECK(t.rank == 1 && t.shape[0] == expect_len && t.dtype == DType::kI32 &&
               t.IsContiguous() && t.device == q.device,
           std::string(name) + ": " + what + " must be i32 [" + std::to_string(expect_len) +
               "] contiguous on the queue device");
}
}  // namespace

void CausalConv1dFwd(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, Tensor& conv_state, const Tensor& query_start_loc,
                     const Tensor& has_initial_state, const CausalConv1dArgs& args) {
  CheckConvCommon(q, out, x, weight, bias, conv_state, "causal_conv1d_fwd");
  const int64_t n = conv_state.shape[0];
  CheckI32Meta(q, query_start_loc, n + 1, "causal_conv1d_fwd", "query_start_loc");
  CheckI32Meta(q, has_initial_state, n, "causal_conv1d_fwd", "has_initial_state");
  reinterpret_cast<CausalConv1dFwdFn>(GetOp(OpId::kCausalConv1dFwd, q.device.type))(
      q, out, x, weight, bias, conv_state, query_start_loc, has_initial_state, args);
}

void CausalConv1dUpdate(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                        const Tensor* bias, Tensor& conv_state, const CausalConv1dArgs& args) {
  CheckConvCommon(q, out, x, weight, bias, conv_state, "causal_conv1d_update");
  VT_CHECK(conv_state.shape[0] == x.shape[0],
           "causal_conv1d_update: one conv_state row per token required");
  reinterpret_cast<CausalConv1dUpdateFn>(GetOp(OpId::kCausalConv1dUpdate, q.device.type))(
      q, out, x, weight, bias, conv_state, args);
}

void L2Norm(Queue& q, Tensor& out, const Tensor& x, const L2NormArgs& args) {
  VT_CHECK(x.rank == 2 || x.rank == 3, "l2norm: rank 2 or 3 required");
  VT_CHECK(out.rank == x.rank, "l2norm: out rank must match x");
  for (int d = 0; d < x.rank; ++d)
    VT_CHECK(out.shape[d] == x.shape[d], "l2norm: out shape must match x");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "l2norm: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "l2norm: contiguous required");
  VT_CHECK(x.device == q.device && out.device == q.device,
           "l2norm: device mismatch (x/out/queue)");
  reinterpret_cast<L2NormFn>(GetOp(OpId::kL2Norm, q.device.type))(q, out, x, args);
}

void RmsNormGated(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                  const Tensor& weight, const RmsNormGatedArgs& args) {
  VT_CHECK(x.rank == 2 && gate.rank == 2 && out.rank == 2 && weight.rank == 1,
           "rmsnorm_gated: x/gate/out rank-2, weight rank-1");
  VT_CHECK(gate.shape[0] == x.shape[0] && gate.shape[1] == x.shape[1] &&
               out.shape[0] == x.shape[0] && out.shape[1] == x.shape[1],
           "rmsnorm_gated: x/gate/out shapes must match");
  VT_CHECK(weight.shape[0] == x.shape[1], "rmsnorm_gated: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(gate.dtype) && IsFloat(weight.dtype) &&
               IsOutFloat(out.dtype),
           "rmsnorm_gated: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && gate.IsContiguous() && weight.IsContiguous() &&
               out.IsContiguous(),
           "rmsnorm_gated: contiguous required");
  VT_CHECK(x.device == q.device && gate.device == q.device && weight.device == q.device &&
               out.device == q.device,
           "rmsnorm_gated: device mismatch (x/gate/weight/out/queue)");
  reinterpret_cast<RmsNormGatedFn>(GetOp(OpId::kRmsNormGated, q.device.type))(q, out, x, gate,
                                                                              weight, args);
}

void GdnPrefill(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                const Tensor& g, const Tensor& beta, Tensor& state,
                const Tensor& query_start_loc, const GdnArgs& args) {
  CheckGdnCommon(q, out, q_in, k, v, g, beta, state, args, "gdn_prefill");
  CheckI32Meta(q, query_start_loc, state.shape[0] + 1, "gdn_prefill", "query_start_loc");
  reinterpret_cast<GdnPrefillFn>(GetOp(OpId::kGdnPrefill, q.device.type))(
      q, out, q_in, k, v, g, beta, state, query_start_loc, args);
}

void GdnDecode(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
               const Tensor& g, const Tensor& beta, Tensor& state, const GdnArgs& args) {
  CheckGdnCommon(q, out, q_in, k, v, g, beta, state, args, "gdn_decode");
  VT_CHECK(state.shape[0] == q_in.shape[0],
           "gdn_decode: one state row per token required (single-token sequences)");
  reinterpret_cast<GdnDecodeFn>(GetOp(OpId::kGdnDecode, q.device.type))(q, out, q_in, k, v, g,
                                                                        beta, state, args);
}

void MoeRouterTopK(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                   const MoeRouterTopKArgs& args) {
  VT_CHECK(logits.rank == 2 && weights.rank == 2 && indices.rank == 2,
           "moe_router_topk: logits/weights/indices rank-2");
  const int64_t t = logits.shape[0], e = logits.shape[1];
  VT_CHECK(args.top_k >= 1 && args.top_k <= e,
           "moe_router_topk: top_k must be in [1, num_experts]");
  VT_CHECK(weights.shape[0] == t && weights.shape[1] == args.top_k,
           "moe_router_topk: weights must be [T, top_k]");
  VT_CHECK(indices.shape[0] == t && indices.shape[1] == args.top_k,
           "moe_router_topk: indices must be [T, top_k]");
  VT_CHECK(IsFloat(logits.dtype), "moe_router_topk: logits must be a float dtype");
  VT_CHECK(weights.dtype == DType::kF32, "moe_router_topk: weights must be f32");
  VT_CHECK(indices.dtype == DType::kI32, "moe_router_topk: indices must be i32");
  VT_CHECK(logits.IsContiguous() && weights.IsContiguous() && indices.IsContiguous(),
           "moe_router_topk: contiguous required");
  VT_CHECK(logits.device == q.device && weights.device == q.device &&
               indices.device == q.device,
           "moe_router_topk: device mismatch (logits/weights/indices/queue)");
  reinterpret_cast<MoeRouterTopKFn>(GetOp(OpId::kMoeRouterTopK, q.device.type))(
      q, weights, indices, logits, args);
}

void MoeCombine(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                const Tensor* shared) {
  VT_CHECK(expert_out.rank == 3 && weights.rank == 2 && out.rank == 2,
           "moe_combine: expert_out [T,K,H], weights [T,K], out [T,H]");
  const int64_t t = out.shape[0], h = out.shape[1], k = weights.shape[1];
  VT_CHECK(expert_out.shape[0] == t && expert_out.shape[1] == k && expert_out.shape[2] == h,
           "moe_combine: expert_out must be [T,K,H] matching out/weights");
  VT_CHECK(weights.shape[0] == t, "moe_combine: weights token count must match out");
  VT_CHECK(IsFloat(expert_out.dtype) && IsOutFloat(out.dtype),
           "moe_combine: float expert_out, f32/bf16 out");
  VT_CHECK(weights.dtype == DType::kF32, "moe_combine: weights must be f32");
  VT_CHECK(expert_out.IsContiguous() && weights.IsContiguous() && out.IsContiguous(),
           "moe_combine: contiguous required");
  VT_CHECK(expert_out.device == q.device && weights.device == q.device &&
               out.device == q.device,
           "moe_combine: device mismatch (expert_out/weights/out/queue)");
  if (shared != nullptr) {
    VT_CHECK(shared->rank == 2 && shared->shape[0] == t && shared->shape[1] == h &&
                 IsFloat(shared->dtype) && shared->IsContiguous() &&
                 shared->device == q.device,
             "moe_combine: shared must be float [T,H] contiguous on the queue device");
  }
  reinterpret_cast<MoeCombineFn>(GetOp(OpId::kMoeCombine, q.device.type))(q, out, expert_out,
                                                                          weights, shared);
}

void Attention(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
               const Tensor& value, const AttentionArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3,
           "attention: query/key/value/out rank-3 [T,H,D]");
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  VT_CHECK(key.shape[0] == t && value.shape[0] == t,
           "attention: query/key/value token count must match");
  VT_CHECK(key.shape[2] == d && value.shape[2] == d,
           "attention: key/value head_dim must match query");
  VT_CHECK(value.shape[1] == hk, "attention: key/value must share the kv-head count");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hq && out.shape[2] == d,
           "attention: out must be [T,Hq,D] matching query");
  VT_CHECK(hk >= 1 && hq >= 1 && hq % hk == 0,
           "attention: Hq must be a positive multiple of Hk (GQA broadcast)");
  VT_CHECK(args.scale > 0.0f, "attention: scale must be set (> 0), e.g. head_dim^-0.5");
  VT_CHECK(IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype,
           "attention: query/key/value must share one float dtype");
  VT_CHECK(IsOutFloat(out.dtype), "attention: out must be f32 or bf16");
  VT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() &&
               out.IsContiguous(),
           "attention: contiguous tensors required");
  VT_CHECK(query.device == q.device && key.device == q.device && value.device == q.device &&
               out.device == q.device,
           "attention: device mismatch (query/key/value/out/queue)");
  reinterpret_cast<AttentionFn>(GetOp(OpId::kAttention, q.device.type))(q, out, query, key,
                                                                        value, args);
}

void ReshapeAndCache(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                     Tensor& v_cache, const Tensor& slot_mapping) {
  VT_CHECK(k.rank == 3 && v.rank == 3,
           "reshape_and_cache: k/v must be rank-3 [num_tokens,num_kv_heads,head_size]");
  VT_CHECK(k_cache.rank == 4 && v_cache.rank == 4,
           "reshape_and_cache: k_cache/v_cache must be rank-4 "
           "[num_blocks,block_size,num_kv_heads,head_size]");
  VT_CHECK(slot_mapping.rank == 1, "reshape_and_cache: slot_mapping must be rank-1 [num_slots]");
  const int64_t num_kv_heads = k.shape[1], head_size = k.shape[2];
  VT_CHECK(v.shape[0] == k.shape[0] && v.shape[1] == num_kv_heads && v.shape[2] == head_size,
           "reshape_and_cache: k and v must share [num_tokens,num_kv_heads,head_size]");
  VT_CHECK(k_cache.shape[2] == num_kv_heads && k_cache.shape[3] == head_size,
           "reshape_and_cache: k_cache num_kv_heads/head_size must match k");
  VT_CHECK(v_cache.shape[0] == k_cache.shape[0] && v_cache.shape[1] == k_cache.shape[1] &&
               v_cache.shape[2] == k_cache.shape[2] && v_cache.shape[3] == k_cache.shape[3],
           "reshape_and_cache: k_cache and v_cache must share shape");
  // Upstream uses slot_mapping.size(0) as the token count: k/v may carry extra
  // trailing rows (CUDA-graph padding) that are ignored.
  VT_CHECK(k.shape[0] >= slot_mapping.shape[0],
           "reshape_and_cache: num_tokens (k.shape[0]) must be >= slot_mapping length");
  VT_CHECK(IsFloat(k.dtype) && k.dtype == v.dtype && k_cache.dtype == k.dtype &&
               v_cache.dtype == k.dtype,
           "reshape_and_cache: k/v/k_cache/v_cache must share one float dtype (auto cache path)");
  VT_CHECK(slot_mapping.dtype == DType::kI64, "reshape_and_cache: slot_mapping must be i64");
  // The paged KV cache is ONE (num_blocks, 2, block_size, H, D) allocation;
  // k_cache/v_cache are the two dim-1 unbind slices, i.e. rank-4 STRIDED views
  // (block stride 2*bs*H*D, not bs*H*D). We therefore must NOT require the cache
  // to be contiguous — indexing is driven by k_cache/v_cache strides (mirroring
  // pinned cache_kernels.cu::reshape_and_cache_flash, which reads block/page/
  // head strides from key_cache.stride(0/1/2)). We only require what the copy
  // actually needs: the innermost element access is well-defined (elem stride 1)
  // and the per-token page is dense (head stride == head_size, i.e. dim-2/3
  // packed), which holds for the NHD unbind slice. The input k/v rows and
  // slot_mapping must be contiguous (upstream reads k/v inner packed, applying
  // only key.stride(0) for the token, and indexes slot_mapping directly).
  VT_CHECK(k.IsContiguous() && v.IsContiguous() && slot_mapping.IsContiguous(),
           "reshape_and_cache: k/v inputs and slot_mapping must be contiguous");
  VT_CHECK(k_cache.stride[3] == 1 && v_cache.stride[3] == 1,
           "reshape_and_cache: k_cache/v_cache innermost (head_size) stride must be 1");
  VT_CHECK(k_cache.stride[2] == head_size && v_cache.stride[2] == head_size,
           "reshape_and_cache: k_cache/v_cache page must be head-contiguous "
           "(stride[2] == head_size) — the NHD unbind-slice layout");
  VT_CHECK(k.device == q.device && v.device == q.device && k_cache.device == q.device &&
               v_cache.device == q.device && slot_mapping.device == q.device,
           "reshape_and_cache: device mismatch (k/v/k_cache/v_cache/slot_mapping/queue)");
  reinterpret_cast<ReshapeAndCacheFn>(GetOp(OpId::kReshapeAndCache, q.device.type))(
      q, k, v, k_cache, v_cache, slot_mapping);
}

}  // namespace vt
