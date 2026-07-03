// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include "vt/tensor.h"

namespace vt {

enum class OpId : uint8_t { kMatmul, kRmsNorm, kSiluAndMul, kRopeNeox, kEmbedding, kCount };

void RegisterOp(OpId op, DeviceType device, void* fn);
void* GetOp(OpId op, DeviceType device);

// Contract: out must not alias any input tensor (RopeNeox is in-place by design).

// out[M,N] = a[M,K] @ b[K,N]; a/b float dtypes (f32/f16/bf16), out f32,
// f32 accumulation, all contiguous, same device.
void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

struct RmsNormArgs {
  float eps = 1e-6f;
  bool gemma = false;  // weight applied as (1 + w), GemmaRMSNorm style
};

// out[T,H] = x[T,H] / sqrt(mean(x^2) + eps) * w  (or *(1+w) when gemma).
// With residual != nullptr (f32 [T,H]): residual += x first (new residual
// stream), and that sum is what gets normalized (upstream fused_add_rms_norm).
// Note: unlike upstream forward_native, the standard path keeps full f32 precision
// (no x.to(weight.dtype) rounding before the weight multiply); parity tests vs
// upstream bf16 need bf16-eps tolerance on the non-gemma path.
void RmsNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
             const RmsNormArgs& args, Tensor* residual = nullptr);

// out[T,D] = silu(x[:, :D]) * x[:, D:], x is [T, 2D].
// Note: computes in f32 (upstream forward_native computes in x's dtype); bf16 parity tests need bf16-eps tolerance.
void SiluAndMul(Queue& q, Tensor& out, const Tensor& x);

// out[T,H] = table[ids[t], :]; ids i32/i64, bounds-checked.
void Embedding(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids);

struct RopeArgs {
  float base = 10000.0f;
  int rotary_dim = 0;  // <= head_dim; even
};

// In-place partial NeoX RoPE on q[T,Hq,D] and k[T,Hk,D] (f32), positions[T].
void RopeNeox(Queue& q, Tensor& q_states, Tensor& k_states, const Tensor& positions,
              const RopeArgs& args);

}  // namespace vt
