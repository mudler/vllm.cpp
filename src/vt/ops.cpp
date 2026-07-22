// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/ops.h"

#include <array>
#include <vector>

namespace vt {

namespace {
// The op table itself moved to src/vt/op_provider.cpp — it is now the
// acceleration-provider registry (include/vt/op_provider.h,
// .agents/specs/metal-mlx-reuse-study.md §6), not a flat one-slot-per-device
// array with silent last-writer-wins. `RegisterOp`, `GetOp` and `OpRegistered`
// keep their exact signatures and semantics, so every one of the ~70 op wrappers
// below is UNCHANGED: the seam is inserted at `GetOp` and picked up by all of
// them at once with zero call-site edits.
bool IsFloat(DType d) { return d == DType::kF32 || d == DType::kF16 || d == DType::kBF16; }
bool IsOutFloat(DType d) { return d == DType::kF32 || d == DType::kBF16; }
}  // namespace

ScalarTypeId ToScalarType(DType dtype) {
  switch (dtype) {
    case DType::kF32: return scalar_type::kF32;
    case DType::kF16: return scalar_type::kF16;
    case DType::kBF16: return scalar_type::kBF16;
    case DType::kI8: return scalar_type::kI8;
    case DType::kI32: return scalar_type::kI32;
    case DType::kI64: return scalar_type::kI64;
    // Block-quantized encodings have no single scalar type: a block mixes
    // scales and packed codes. Kernels consume them through the quant traits
    // table, never through a KernelTensorDesc scalar type.
    case DType::kQ4_0:
    case DType::kQ8_0:
    case DType::kQ3_K:
    case DType::kQ4_K:
    case DType::kQ5_K:
    case DType::kQ6_K:
    case DType::kQ8_K:
      break;
  }
  VT_CHECK(false, "unsupported storage dtype for scalar-type conversion");
  return scalar_type::kF32;
}

KernelTensorDesc Describe(const Tensor& tensor, ScalarTypeId semantic_type,
                          KernelLayout layout) {
  VT_CHECK(tensor.rank >= 1 && tensor.rank <= kMaxRank,
           "kernel tensor descriptor rank out of range");
  VT_CHECK(tensor.data != nullptr, "kernel tensor descriptor requires non-null data");
  for (int d = 0; d < tensor.rank; ++d) {
    VT_CHECK(tensor.shape[d] > 0, "kernel tensor descriptor requires positive dimensions");
    VT_CHECK(tensor.stride[d] >= 0, "kernel tensor descriptor rejects negative strides");
  }

  switch (layout) {
    case KernelLayout::kStrided:
      VT_CHECK(semantic_type == ToScalarType(tensor.dtype),
               "strided layout semantic type must match its storage dtype");
      break;
    case KernelLayout::kPackedTwoFp4PerByte:
      VT_CHECK(tensor.dtype == DType::kI8 && semantic_type == scalar_type::kFE2M1f,
               "packed-two-fp4 layout requires i8 storage with explicit FE2M1 semantics");
      break;
    case KernelLayout::kBlockScaleLinear:
    case KernelLayout::kBlockScaleSwizzled:
      VT_CHECK(tensor.dtype == DType::kI8 &&
                   (semantic_type == scalar_type::kFE4M3fn ||
                    semantic_type == scalar_type::kFE8M0fnu),
               "block-scale layout requires i8 storage with explicit FP8 scale semantics");
      break;
    case KernelLayout::kMarlinInterleaved:
      VT_CHECK(tensor.dtype == DType::kI8 &&
                   (semantic_type == scalar_type::kFE2M1f ||
                    semantic_type == scalar_type::kI4 || semantic_type == scalar_type::kU4),
               "Marlin layout requires i8 storage with an explicit 4-bit semantic type");
      break;
  }

  KernelTensorDesc desc;
  desc.data = tensor.data;
  desc.storage_dtype = tensor.dtype;
  desc.scalar_type = semantic_type;
  desc.device = tensor.device;
  desc.rank = tensor.rank;
  desc.layout = layout;
  for (int d = 0; d < kMaxRank; ++d) {
    desc.shape[d] = tensor.shape[d];
    desc.stride[d] = tensor.stride[d];
  }
  return desc;
}

WorkspaceKey MakeWorkspaceKey(const Queue& q, OpId op, WorkspaceSlot slot) {
  VT_CHECK(q.id != 0, "workspace key requires a live queue identity");
  return WorkspaceKey{q.device, q.id, reinterpret_cast<uintptr_t>(q.handle), op, slot};
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

void DropinProbe(Queue& q, Tensor& out, const Tensor& in,
                 const DropinProbeArgs& args) {
  VT_CHECK(q.id != 0, "dropin_probe: live queue required");
  VT_CHECK(in.rank == 2 && out.rank == 2, "dropin_probe: rank-2 tensors required");
  VT_CHECK(in.shape[0] == out.shape[0] && in.shape[1] == out.shape[1],
           "dropin_probe: input/output shape mismatch");
  VT_CHECK(in.device == q.device && out.device == q.device,
           "dropin_probe: input/output/queue device mismatch");
  VT_CHECK(args.workspace_slot != args.scalar_slot,
           "dropin_probe: workspace and scalar slots must not alias");
  VT_CHECK(args.workspace_bytes >= sizeof(uint32_t),
           "dropin_probe: workspace must hold the raw-launch marker");
  (void)Describe(in, args.scalar_type, args.layout);
  (void)Describe(out, args.scalar_type, args.layout);
  VT_CHECK(args.scalar_type == scalar_type::kF32 && args.layout == KernelLayout::kStrided,
           "dropin_probe: W0 raw kernel supports f32 strided tensors only");
  GetTypedOp<DropinProbeFn>(OpId::kDropinProbe, q.device.type)(q, out, in, args);
}

void MatmulBT(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  // GGUF compute-in-quant (QUANT-GGUF-CIQ-GEMM work row G4). A block-quantized
  // weight is NOT an elementwise tensor — it has no per-element stride and
  // cannot be read by kMatmulBT — but it IS in exactly the [N, K] orientation
  // this entry point defines, which is ggml's src0 layout and GGUF's disk
  // order (ggml-cpu.c:1245-1443). Dispatching it to kMatmulBTQuant here is
  // what routes the MODEL: every matmul helper already sends an `nk == true`
  // weight to MatmulBT (qwen3_5.cpp:1067,1078 device-in/out and :743,760
  // host), so the keep-quant loader's block-typed OwnedTensor reaches the
  // quantized GEMM with no signature, call-site or forward change. Elementwise
  // weights fall through to the unchanged validation + kMatmulBT below, so
  // every safetensors path is bit-identical by construction.
  if (IsBlockQuant(b.dtype)) {
    MatmulBTQuant(q, out, a, b);
    return;
  }
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2, "matmul_bt: rank-2 tensors required");
  VT_CHECK(a.shape[1] == b.shape[1], "matmul_bt: inner dims mismatch (b is [N,K])");
  VT_CHECK(out.shape[0] == a.shape[0] && out.shape[1] == b.shape[0],
           "matmul_bt: output shape mismatch");
  VT_CHECK(IsFloat(a.dtype) && IsFloat(b.dtype) && IsOutFloat(out.dtype),
           "matmul_bt: float inputs and f32/bf16 output required");
  // The ACTIVATION may be ROW-STRIDED (relaxed at MLA campaign W6): upstream's
  // `kv_b_proj(kv_c)` inside `_compute_prefill_context`
  // (mla_attention.py:2141-2160) is applied to a COLUMN SLICE of the 576-wide
  // chunked-prefill workspace, i.e. a torch view whose row stride is 576 while
  // K is 512 — and `F.linear` accepts exactly that. Only the innermost dim must
  // be packed; for a CONTIGUOUS activation the row stride IS K, so every
  // existing caller passes byte-identical arguments (same cuBLASLt ld, so the
  // same algo, so bit-identical results).
  VT_CHECK(a.stride[1] == 1 && a.stride[0] >= a.shape[1],
           "matmul_bt: activation rows must be packed (innermost stride 1) and "
           "non-overlapping");
  VT_CHECK(b.IsContiguous() && out.IsContiguous(),
           "matmul_bt: contiguous weight and output required");
  VT_CHECK(a.device == b.device && a.device == out.device && a.device == q.device,
           "matmul_bt: device mismatch");
  reinterpret_cast<MatmulFn>(GetOp(OpId::kMatmulBT, q.device.type))(q, out, a, b);
}

// vt::MatmulBTQuant — see ops.h. Validation mirrors MatmulBT except for the
// weight, whose block layout replaces the elementwise stride contract.
void MatmulBTQuant(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "matmul_bt_quant: rank-2 tensors required");
  VT_CHECK(a.shape[1] == b.shape[1],
           "matmul_bt_quant: inner dims mismatch (b is [N,K])");
  VT_CHECK(out.shape[0] == a.shape[0] && out.shape[1] == b.shape[0],
           "matmul_bt_quant: output shape mismatch");
  VT_CHECK(IsBlockQuant(b.dtype),
           "matmul_bt_quant: weight must be a block-quantized dtype (use "
           "MatmulBT for elementwise weights)");
  VT_CHECK(IsFloat(a.dtype) && IsOutFloat(out.dtype),
           "matmul_bt_quant: float activation and f32/bf16 output required");
  // ggml_row_size asserts the row is whole blocks; a GEMM weight whose K is
  // not block-aligned is not keep-quant eligible in the first place.
  VT_CHECK(b.shape[1] % BlockElems(b.dtype) == 0,
           "matmul_bt_quant: K must be a whole number of weight blocks");
  // Same relaxed activation contract as MatmulBT: only the innermost dim must
  // be packed, so a column slice of a wider workspace is consumed as-is.
  VT_CHECK(a.stride[1] == 1 && a.stride[0] >= a.shape[1],
           "matmul_bt_quant: activation rows must be packed (innermost stride "
           "1) and non-overlapping");
  VT_CHECK(out.IsContiguous(), "matmul_bt_quant: contiguous output required");
  VT_CHECK(a.device == b.device && a.device == out.device && a.device == q.device,
           "matmul_bt_quant: device mismatch");
  reinterpret_cast<MatmulFn>(GetOp(OpId::kMatmulBTQuant, q.device.type))(q, out, a, b);
}

// vt::BatchedMatmul — `torch.bmm` (mla_attention.py:789 q-side W_UK absorption,
// :1034 W_UV v-up-projection). Stride-driven on every operand because BOTH
// upstream call sites pass transposed views; only the innermost dim must be
// unit-stride.
void BatchedMatmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 3 && b.rank == 3 && out.rank == 3,
           "batched_matmul: rank-3 tensors required (out[G,M,N] = a[G,M,K] @ b[G,K,N])");
  VT_CHECK(a.shape[0] == b.shape[0] && a.shape[0] == out.shape[0],
           "batched_matmul: batch dim mismatch");
  VT_CHECK(a.shape[2] == b.shape[1], "batched_matmul: inner dims mismatch");
  VT_CHECK(out.shape[1] == a.shape[1] && out.shape[2] == b.shape[2],
           "batched_matmul: output shape mismatch");
  VT_CHECK(IsFloat(a.dtype) && IsFloat(b.dtype) && IsOutFloat(out.dtype),
           "batched_matmul: float inputs and f32/bf16 output required");
  VT_CHECK(a.dtype == b.dtype, "batched_matmul: a/b dtype must match");
  // Only the innermost stride is constrained; the batch/row strides are free so
  // a transposed view (upstream's `q_nope.transpose(0,1)` / `out.transpose(0,1)`)
  // is consumed without a copy.
  VT_CHECK(a.stride[2] == 1 && b.stride[2] == 1 && out.stride[2] == 1,
           "batched_matmul: innermost dimension must be unit-stride");
  VT_CHECK(a.stride[1] >= a.shape[2] && b.stride[1] >= b.shape[2] &&
               out.stride[1] >= out.shape[2],
           "batched_matmul: row stride must not overlap the next row");
  VT_CHECK(a.device == b.device && a.device == out.device && a.device == q.device,
           "batched_matmul: device mismatch");
  GetTypedOp<BatchedMatmulFn>(OpId::kBatchedMatmul, q.device.type)(q, out, a, b);
}

// vt::ConcatMlaNopeRope — the generalization of upstream's two MLA head-concat
// sites: `concat_mla_q` (cache_kernels.cu:1555-1600, the decode 512+64 query)
// and `_concat_k_nope_k_pe` (mla_attention.py:2063-2092, the prefill 128+64 key
// with a head-BROADCAST rope part). Stride checks mirror
// cache_kernels.cu:1572-1577.
void ConcatMlaNopeRope(Queue& q, Tensor& out, const Tensor& nope, const Tensor& rope) {
  VT_CHECK(out.rank == 3 && nope.rank == 3 && rope.rank == 3,
           "concat_mla_nope_rope: rank-3 [tokens, heads, dim] tensors required");
  const int64_t tokens = out.shape[0], heads = out.shape[1];
  const int64_t dn = nope.shape[2], dr = rope.shape[2];
  VT_CHECK(nope.shape[0] == tokens && rope.shape[0] == tokens,
           "concat_mla_nope_rope: token count mismatch");
  VT_CHECK(nope.shape[1] == heads, "concat_mla_nope_rope: nope head count mismatch");
  VT_CHECK(rope.shape[1] == heads || rope.shape[1] == 1,
           "concat_mla_nope_rope: rope must carry `heads` heads or exactly 1 "
           "(the single shared k_pe head, broadcast — mla_attention.py:2063-2092)");
  VT_CHECK(out.shape[2] == dn + dr,
           "concat_mla_nope_rope: out last dim must be nope_dim + rope_dim");
  VT_CHECK(dn > 0 && dr > 0, "concat_mla_nope_rope: both parts must be non-empty");
  VT_CHECK(out.dtype == nope.dtype && out.dtype == rope.dtype,
           "concat_mla_nope_rope: all tensors must share one dtype");
  VT_CHECK(IsOutFloat(out.dtype) || out.dtype == DType::kF16,
           "concat_mla_nope_rope: f32/bf16/f16 only");
  VT_CHECK(out.stride[2] == 1 && nope.stride[2] == 1 && rope.stride[2] == 1,
           "concat_mla_nope_rope: innermost dimension must be unit-stride "
           "(upstream cache_kernels.cu:1572-1577)");
  VT_CHECK(out.device == q.device && nope.device == q.device && rope.device == q.device,
           "concat_mla_nope_rope: device mismatch");
  if (tokens == 0 || heads == 0) return;  // `if (num_tokens == 0) return;` (:1584)
  GetTypedOp<ConcatMlaNopeRopeFn>(OpId::kConcatMlaNopeRope, q.device.type)(q, out, nope, rope);
}

void MatmulNvfp4(Queue& q, Tensor& out, const Tensor& act, const Tensor& weight_packed,
                 const Tensor& weight_scale, float weight_scale_2) {
  VT_CHECK(act.rank == 2 && weight_packed.rank == 2 && weight_scale.rank == 2 && out.rank == 2,
           "matmul_nvfp4: act/weight_packed/weight_scale/out must be rank-2");
  const int64_t m = act.shape[0], k = act.shape[1], n = weight_packed.shape[0];
  VT_CHECK(k % 16 == 0, "matmul_nvfp4: K (act inner dim) must be a multiple of 16");
  VT_CHECK(weight_packed.shape[1] == k / 2,
           "matmul_nvfp4: weight_packed must be [N, K/2] (two fp4 codes per byte)");
  VT_CHECK(weight_scale.shape[0] == n && weight_scale.shape[1] == k / 16,
           "matmul_nvfp4: weight_scale must be [N, K/16] (one fp8 scale per 16-elem group)");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_nvfp4: out must be [M, N]");
  VT_CHECK(IsFloat(act.dtype) && IsOutFloat(out.dtype),
           "matmul_nvfp4: float act, f32/bf16 out");
  VT_CHECK(weight_packed.dtype == DType::kI8 && weight_scale.dtype == DType::kI8,
           "matmul_nvfp4: weight_packed/weight_scale must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(act.IsContiguous() && weight_packed.IsContiguous() && weight_scale.IsContiguous() &&
               out.IsContiguous(),
           "matmul_nvfp4: contiguous tensors required");
  VT_CHECK(act.device == q.device && weight_packed.device == q.device &&
               weight_scale.device == q.device && out.device == q.device,
           "matmul_nvfp4: device mismatch (act/weight_packed/weight_scale/out/queue)");
  reinterpret_cast<MatmulNvfp4Fn>(GetOp(OpId::kMatmulNvfp4, q.device.type))(
      q, out, act, weight_packed, weight_scale, weight_scale_2);
}

void ScaledFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& x,
                    float input_global_scale_inv, Fp4ScaleLayout scale_layout) {
  VT_CHECK(x.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "scaled_fp4_quant: x/out_packed/out_scale must be rank-2");
  const int64_t m = x.shape[0], k = x.shape[1];
  VT_CHECK(k % 16 == 0, "scaled_fp4_quant: K (inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == k / 2,
           "scaled_fp4_quant: out_packed must be [M, K/2]");
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  if (scale_layout == Fp4ScaleLayout::kLinear) {
    VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == k / 16,
             "scaled_fp4_quant: linear out_scale must be [M, K/16]");
  } else {
    VT_CHECK(scale_layout == Fp4ScaleLayout::kCutlassSwizzled,
             "scaled_fp4_quant: invalid scale layout");
    VT_CHECK(out_scale.shape[0] == round_up(m, 128) &&
                 out_scale.shape[1] == round_up(k / 16, 4),
             "scaled_fp4_quant: swizzled out_scale must be "
             "[round_up(M,128), round_up(K/16,4)]");
  }
  VT_CHECK(IsFloat(x.dtype), "scaled_fp4_quant: float x required");
  VT_CHECK(out_packed.dtype == DType::kI8 && out_scale.dtype == DType::kI8,
           "scaled_fp4_quant: out_packed/out_scale must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(x.IsContiguous() && out_packed.IsContiguous() && out_scale.IsContiguous(),
           "scaled_fp4_quant: contiguous tensors required");
  VT_CHECK(x.device == q.device && out_packed.device == q.device && out_scale.device == q.device,
           "scaled_fp4_quant: device mismatch (x/out_packed/out_scale/queue)");
  reinterpret_cast<ScaledFp4QuantFn>(GetOp(OpId::kScaledFp4Quant, q.device.type))(
      q, out_packed, out_scale, x, input_global_scale_inv, scale_layout);
}

void SiluMulFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& gate,
                     const Tensor& up, float input_global_scale_inv,
                     Fp4ScaleLayout scale_layout) {
  VT_CHECK(gate.rank == 2 && up.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "silu_mul_fp4_quant: gate/up/out_packed/out_scale must be rank-2");
  const int64_t m = gate.shape[0], i = gate.shape[1];
  VT_CHECK(up.shape[0] == m && up.shape[1] == i, "silu_mul_fp4_quant: gate/up shape mismatch");
  VT_CHECK(i % 16 == 0, "silu_mul_fp4_quant: I (inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == i / 2,
           "silu_mul_fp4_quant: out_packed must be [M, I/2]");
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  if (scale_layout == Fp4ScaleLayout::kLinear) {
    VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == i / 16,
             "silu_mul_fp4_quant: linear out_scale must be [M, I/16]");
  } else {
    VT_CHECK(scale_layout == Fp4ScaleLayout::kCutlassSwizzled,
             "silu_mul_fp4_quant: invalid scale layout");
    VT_CHECK(out_scale.shape[0] == round_up(m, 128) &&
                 out_scale.shape[1] == round_up(i / 16, 4),
             "silu_mul_fp4_quant: swizzled out_scale must be "
             "[round_up(M,128), round_up(I/16,4)]");
  }
  VT_CHECK(IsFloat(gate.dtype) && gate.dtype == up.dtype,
           "silu_mul_fp4_quant: gate/up must be the same float dtype");
  VT_CHECK(out_packed.dtype == DType::kI8 && out_scale.dtype == DType::kI8,
           "silu_mul_fp4_quant: out_packed/out_scale must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(gate.IsContiguous() && up.IsContiguous() && out_packed.IsContiguous() &&
               out_scale.IsContiguous(),
           "silu_mul_fp4_quant: contiguous tensors required");
  VT_CHECK(gate.device == q.device && up.device == q.device && out_packed.device == q.device &&
               out_scale.device == q.device,
           "silu_mul_fp4_quant: device mismatch");
  reinterpret_cast<SiluMulFp4QuantFn>(GetOp(OpId::kSiluMulFp4Quant, q.device.type))(
      q, out_packed, out_scale, gate, up, input_global_scale_inv, scale_layout);
}

void SiluAndMulFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale,
                        const Tensor& gate_up, float input_global_scale_inv,
                        Fp4ScaleLayout scale_layout) {
  VT_CHECK(gate_up.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "silu_and_mul_fp4_quant: gate_up/out_packed/out_scale must be rank-2");
  const int64_t m = gate_up.shape[0];
  VT_CHECK(gate_up.shape[1] % 2 == 0,
           "silu_and_mul_fp4_quant: gate_up inner dim must be even");
  const int64_t i = gate_up.shape[1] / 2;
  VT_CHECK(i % 16 == 0,
           "silu_and_mul_fp4_quant: I (half inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == i / 2,
           "silu_and_mul_fp4_quant: out_packed must be [M, I/2]");
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  if (scale_layout == Fp4ScaleLayout::kLinear) {
    VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == i / 16,
             "silu_and_mul_fp4_quant: linear out_scale must be [M, I/16]");
  } else {
    VT_CHECK(scale_layout == Fp4ScaleLayout::kCutlassSwizzled,
             "silu_and_mul_fp4_quant: invalid scale layout");
    VT_CHECK(out_scale.shape[0] == round_up(m, 128) &&
                 out_scale.shape[1] == round_up(i / 16, 4),
             "silu_and_mul_fp4_quant: swizzled out_scale must be "
             "[round_up(M,128), round_up(I/16,4)]");
  }
  VT_CHECK(gate_up.dtype == DType::kF32 || gate_up.dtype == DType::kBF16,
           "silu_and_mul_fp4_quant: gate_up must be f32 or bf16");
  VT_CHECK(out_packed.dtype == DType::kI8 && out_scale.dtype == DType::kI8,
           "silu_and_mul_fp4_quant: outputs must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(gate_up.IsContiguous() && out_packed.IsContiguous() &&
               out_scale.IsContiguous(),
           "silu_and_mul_fp4_quant: contiguous tensors required");
  VT_CHECK(gate_up.device == q.device && out_packed.device == q.device &&
               out_scale.device == q.device,
           "silu_and_mul_fp4_quant: device mismatch");
  reinterpret_cast<SiluAndMulFp4QuantFn>(
      GetOp(OpId::kSiluAndMulFp4Quant, q.device.type))(
      q, out_packed, out_scale, gate_up, input_global_scale_inv, scale_layout);
}

void SigmoidGateFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale,
                         const Tensor& attn, const Tensor& gate,
                         float input_global_scale_inv, Fp4ScaleLayout scale_layout) {
  VT_CHECK(attn.rank == 2 && gate.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "sigmoid_gate_fp4_quant: attn/gate/out_packed/out_scale must be rank-2");
  const int64_t m = attn.shape[0], i = attn.shape[1];
  VT_CHECK(gate.shape[0] == m && gate.shape[1] == i,
           "sigmoid_gate_fp4_quant: attn/gate shape mismatch");
  VT_CHECK(i % 16 == 0, "sigmoid_gate_fp4_quant: K (inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == i / 2,
           "sigmoid_gate_fp4_quant: out_packed must be [M, K/2]");
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  if (scale_layout == Fp4ScaleLayout::kLinear) {
    VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == i / 16,
             "sigmoid_gate_fp4_quant: linear out_scale must be [M, K/16]");
  } else {
    VT_CHECK(scale_layout == Fp4ScaleLayout::kCutlassSwizzled,
             "sigmoid_gate_fp4_quant: invalid scale layout");
    VT_CHECK(out_scale.shape[0] == round_up(m, 128) &&
                 out_scale.shape[1] == round_up(i / 16, 4),
             "sigmoid_gate_fp4_quant: swizzled out_scale must be "
             "[round_up(M,128), round_up(K/16,4)]");
  }
  VT_CHECK(attn.dtype == DType::kF32 || attn.dtype == DType::kBF16,
           "sigmoid_gate_fp4_quant: attn must be f32 or bf16");
  VT_CHECK(gate.dtype == DType::kF32,
           "sigmoid_gate_fp4_quant: gate must be f32 (sigmoid input unrounded)");
  VT_CHECK(out_packed.dtype == DType::kI8 && out_scale.dtype == DType::kI8,
           "sigmoid_gate_fp4_quant: out_packed/out_scale must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(attn.IsContiguous() && gate.IsContiguous() && out_packed.IsContiguous() &&
               out_scale.IsContiguous(),
           "sigmoid_gate_fp4_quant: contiguous tensors required");
  VT_CHECK(attn.device == q.device && gate.device == q.device &&
               out_packed.device == q.device && out_scale.device == q.device,
           "sigmoid_gate_fp4_quant: device mismatch");
  reinterpret_cast<SigmoidGateFp4QuantFn>(GetOp(OpId::kSigmoidGateFp4Quant, q.device.type))(
      q, out_packed, out_scale, attn, gate, input_global_scale_inv, scale_layout);
}

void MatmulNvfp4Fp4(Queue& q, Tensor& out, const Tensor& a_packed, const Tensor& a_scale,
                    const Tensor& b_packed, const Tensor& b_scale, float alpha) {
  VT_CHECK(out.rank == 2 && a_packed.rank == 2 && a_scale.rank == 2 && b_packed.rank == 2 &&
               b_scale.rank == 2,
           "matmul_nvfp4_fp4: all tensors must be rank-2");
  const int64_t m = a_packed.shape[0], k = a_packed.shape[1] * 2, n = b_packed.shape[0];
  VT_CHECK(k % 16 == 0, "matmul_nvfp4_fp4: K (inner dim) must be a multiple of 16");
  VT_CHECK(a_scale.shape[0] == m && a_scale.shape[1] == k / 16,
           "matmul_nvfp4_fp4: a_scale must be [M, K/16]");
  VT_CHECK(b_packed.shape[1] == k / 2,
           "matmul_nvfp4_fp4: b_packed must be [N, K/2] (K matches a_packed)");
  VT_CHECK(b_scale.shape[0] == n && b_scale.shape[1] == k / 16,
           "matmul_nvfp4_fp4: b_scale must be [N, K/16]");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_nvfp4_fp4: out must be [M, N]");
  VT_CHECK(IsOutFloat(out.dtype), "matmul_nvfp4_fp4: f32/bf16 out");
  VT_CHECK(a_packed.dtype == DType::kI8 && a_scale.dtype == DType::kI8 &&
               b_packed.dtype == DType::kI8 && b_scale.dtype == DType::kI8,
           "matmul_nvfp4_fp4: packed/scale operands must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(out.IsContiguous() && a_packed.IsContiguous() && a_scale.IsContiguous() &&
               b_packed.IsContiguous() && b_scale.IsContiguous(),
           "matmul_nvfp4_fp4: contiguous tensors required");
  VT_CHECK(out.device == q.device && a_packed.device == q.device && a_scale.device == q.device &&
               b_packed.device == q.device && b_scale.device == q.device,
           "matmul_nvfp4_fp4: device mismatch");
  reinterpret_cast<MatmulNvfp4Fp4Fn>(GetOp(OpId::kMatmulNvfp4Fp4, q.device.type))(
      q, out, a_packed, a_scale, b_packed, b_scale, alpha);
}

void SwizzleBlockscale(Queue& q, Tensor& out_swizzled, const Tensor& in_linear) {
  VT_CHECK(in_linear.rank == 2 && out_swizzled.rank == 2,
           "swizzle_blockscale: rank-2 tensors required");
  const int64_t rows = in_linear.shape[0], cols = in_linear.shape[1];
  auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  VT_CHECK(out_swizzled.shape[0] == round_up(rows, 128) &&
               out_swizzled.shape[1] == round_up(cols, 4),
           "swizzle_blockscale: out must be [round_up(rows,128), round_up(cols,4)]");
  VT_CHECK(in_linear.dtype == DType::kI8 && out_swizzled.dtype == DType::kI8,
           "swizzle_blockscale: i8 (raw fp8) operands required");
  VT_CHECK(in_linear.IsContiguous() && out_swizzled.IsContiguous(),
           "swizzle_blockscale: contiguous tensors required");
  VT_CHECK(in_linear.device == q.device && out_swizzled.device == q.device,
           "swizzle_blockscale: device mismatch");
  reinterpret_cast<SwizzleBlockscaleFn>(GetOp(OpId::kSwizzleBlockscale, q.device.type))(
      q, out_swizzled, in_linear);
}

namespace {
void ValidateMatmulNvfp4Cutlass(Queue& q, Tensor& out,
                                const Tensor& a_packed,
                                const Tensor& a_sf_sw,
                                const Tensor& b_packed,
                                const Tensor& b_sf_sw) {
  VT_CHECK(out.rank == 2 && a_packed.rank == 2 && a_sf_sw.rank == 2 && b_packed.rank == 2 &&
               b_sf_sw.rank == 2,
           "matmul_nvfp4_cutlass: all tensors must be rank-2");
  const int64_t m = a_packed.shape[0], k = a_packed.shape[1] * 2, n = b_packed.shape[0];
  VT_CHECK(k % 32 == 0 && n % 32 == 0, "matmul_nvfp4_cutlass: K and N must be multiples of 32");
  VT_CHECK(b_packed.shape[1] == k / 2,
           "matmul_nvfp4_cutlass: b_packed must be [N, K/2] (K matches a_packed)");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_nvfp4_cutlass: out must be [M, N]");
  VT_CHECK(out.dtype == DType::kBF16 || out.dtype == DType::kF32,
           "matmul_nvfp4_cutlass: out must be bf16 or f32 (bf16 epilogue, f32 via cast)");
  VT_CHECK(a_packed.dtype == DType::kI8 && a_sf_sw.dtype == DType::kI8 &&
               b_packed.dtype == DType::kI8 && b_sf_sw.dtype == DType::kI8,
           "matmul_nvfp4_cutlass: packed/scale operands must be i8 (raw fp4/fp8 bytes)");
  auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  VT_CHECK(a_sf_sw.shape[0] == round_up(m, 128) && a_sf_sw.shape[1] == round_up(k / 16, 4),
           "matmul_nvfp4_cutlass: a_sf must be swizzled [round_up(M,128), round_up(K/16,4)]");
  VT_CHECK(b_sf_sw.shape[0] == round_up(n, 128) && b_sf_sw.shape[1] == round_up(k / 16, 4),
           "matmul_nvfp4_cutlass: b_sf must be swizzled [round_up(N,128), round_up(K/16,4)]");
  VT_CHECK(out.device == q.device && a_packed.device == q.device && a_sf_sw.device == q.device &&
               b_packed.device == q.device && b_sf_sw.device == q.device,
           "matmul_nvfp4_cutlass: device mismatch");
}

void DispatchMatmulNvfp4Cutlass(Queue& q, Tensor& out,
                                const Tensor& a_packed,
                                const Tensor& a_sf_sw,
                                const Tensor& b_packed,
                                const Tensor& b_sf_sw,
                                const Tensor* alpha_device,
                                float alpha_host) {
  reinterpret_cast<MatmulNvfp4CutlassFn>(GetOp(OpId::kMatmulNvfp4Cutlass, q.device.type))(
      q, out, a_packed, a_sf_sw, b_packed, b_sf_sw, alpha_device,
      alpha_host);
}
}  // namespace

void MatmulNvfp4Cutlass(Queue& q, Tensor& out, const Tensor& a_packed,
                        const Tensor& a_sf_sw, const Tensor& b_packed,
                        const Tensor& b_sf_sw, const Tensor& alpha) {
  ValidateMatmulNvfp4Cutlass(q, out, a_packed, a_sf_sw, b_packed, b_sf_sw);
  VT_CHECK(alpha.rank == 0 || alpha.rank == 1,
           "matmul_nvfp4_cutlass: alpha must be a rank-0 or rank-1 scalar tensor");
  VT_CHECK(alpha.Numel() == 1,
           "matmul_nvfp4_cutlass: alpha must contain exactly one element");
  VT_CHECK(alpha.dtype == DType::kF32,
           "matmul_nvfp4_cutlass: alpha must be f32");
  VT_CHECK(alpha.data != nullptr,
           "matmul_nvfp4_cutlass: alpha must have non-null storage");
  VT_CHECK(alpha.IsContiguous(),
           "matmul_nvfp4_cutlass: alpha must be contiguous");
  VT_CHECK(alpha.device == q.device,
           "matmul_nvfp4_cutlass: alpha device mismatch");
  DispatchMatmulNvfp4Cutlass(q, out, a_packed, a_sf_sw, b_packed,
                             b_sf_sw, &alpha, 0.0F);
}

void MatmulNvfp4Cutlass(Queue& q, Tensor& out, const Tensor& a_packed,
                        const Tensor& a_sf_sw, const Tensor& b_packed,
                        const Tensor& b_sf_sw, float alpha) {
  ValidateMatmulNvfp4Cutlass(q, out, a_packed, a_sf_sw, b_packed, b_sf_sw);
  DispatchMatmulNvfp4Cutlass(q, out, a_packed, a_sf_sw, b_packed,
                             b_sf_sw, nullptr, alpha);
}
void QuantFp8Static(Queue& q, Tensor& out_fp8, const Tensor& x, float input_scale) {
  VT_CHECK(x.rank == 2 && out_fp8.rank == 2, "quant_fp8_static: x/out must be rank-2");
  VT_CHECK(out_fp8.shape[0] == x.shape[0] && out_fp8.shape[1] == x.shape[1],
           "quant_fp8_static: out must match x shape [M,K]");
  VT_CHECK(IsFloat(x.dtype), "quant_fp8_static: float x (f32/bf16) required");
  VT_CHECK(out_fp8.dtype == DType::kI8, "quant_fp8_static: out must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(x.IsContiguous() && out_fp8.IsContiguous(),
           "quant_fp8_static: contiguous tensors required");
  VT_CHECK(x.device == q.device && out_fp8.device == q.device,
           "quant_fp8_static: device mismatch (x/out/queue)");
  reinterpret_cast<QuantFp8StaticFn>(GetOp(OpId::kQuantFp8Static, q.device.type))(q, out_fp8, x,
                                                                                  input_scale);
}
void RmsNormQuantFp8(Queue& q, Tensor& out_fp8, Tensor* out_bf16, const Tensor& x,
                     const Tensor& weight, const RmsNormArgs& args, Tensor* residual,
                     float input_scale) {
  VT_CHECK(x.rank == 2 && out_fp8.rank == 2 && weight.rank == 1,
           "rmsnorm_quant_fp8: x/out_fp8 rank-2, weight rank-1");
  VT_CHECK(x.shape[0] == out_fp8.shape[0] && x.shape[1] == out_fp8.shape[1],
           "rmsnorm_quant_fp8: out_fp8 must match x shape [T,H]");
  VT_CHECK(weight.shape[0] == x.shape[1], "rmsnorm_quant_fp8: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype), "rmsnorm_quant_fp8: float x/weight required");
  VT_CHECK(out_fp8.dtype == DType::kI8,
           "rmsnorm_quant_fp8: out_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(x.IsContiguous() && out_fp8.IsContiguous() && weight.IsContiguous(),
           "rmsnorm_quant_fp8: contiguous tensors required");
  if (out_bf16 != nullptr) {
    VT_CHECK(out_bf16->dtype == DType::kBF16 && out_bf16->rank == 2 &&
                 out_bf16->shape[0] == x.shape[0] && out_bf16->shape[1] == x.shape[1] &&
                 out_bf16->IsContiguous() && out_bf16->device == x.device,
             "rmsnorm_quant_fp8: out_bf16 must be bf16 [T,H] contiguous on x's device");
  }
  if (residual != nullptr) {
    VT_CHECK((residual->dtype == DType::kF32 || residual->dtype == DType::kBF16) &&
                 residual->rank == 2 && residual->shape[0] == x.shape[0] &&
                 residual->shape[1] == x.shape[1] && residual->IsContiguous() &&
                 residual->device == x.device,
             "rmsnorm_quant_fp8: residual must be f32/bf16 [T,H] contiguous on x's device");
  }
  VT_CHECK(x.device == out_fp8.device && weight.device == x.device && x.device == q.device,
           "rmsnorm_quant_fp8: device mismatch (x/out_fp8/weight/queue)");
  reinterpret_cast<RmsNormQuantFp8Fn>(GetOp(OpId::kRmsNormQuantFp8, q.device.type))(
      q, out_fp8, out_bf16, x, weight, args, residual, input_scale);
}
void RmsNormGatedQuantFp8(Queue& q, Tensor& out_fp8, const Tensor& x, const Tensor& gate,
                          const Tensor& weight, const RmsNormGatedArgs& args, float input_scale) {
  const int64_t d = x.rank == 0 ? 0 : x.shape[x.rank - 1];
  VT_CHECK(weight.rank == 1 && weight.shape[0] == d,
           "rmsnorm_gated_quant_fp8: weight must be rank-1 [D] matching x's last dim");
  VT_CHECK(out_fp8.rank == x.rank, "rmsnorm_gated_quant_fp8: out_fp8 rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out_fp8.shape[i] == x.shape[i],
             "rmsnorm_gated_quant_fp8: out_fp8 shape must match x");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsFloat(gate.dtype),
           "rmsnorm_gated_quant_fp8: float x/gate/weight required");
  VT_CHECK(gate.dtype == x.dtype && weight.dtype == x.dtype,
           "rmsnorm_gated_quant_fp8: gate/weight dtype must match x");
  VT_CHECK(out_fp8.dtype == DType::kI8,
           "rmsnorm_gated_quant_fp8: out_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(x.IsContiguous() && out_fp8.IsContiguous() && weight.IsContiguous(),
           "rmsnorm_gated_quant_fp8: contiguous x/out_fp8/weight required");
  VT_CHECK(x.device == out_fp8.device && weight.device == x.device && gate.device == x.device &&
               x.device == q.device,
           "rmsnorm_gated_quant_fp8: device mismatch (x/out_fp8/gate/weight/queue)");
  reinterpret_cast<RmsNormGatedQuantFp8Fn>(GetOp(OpId::kRmsNormGatedQuantFp8, q.device.type))(
      q, out_fp8, x, gate, weight, args, input_scale);
}
void MatmulFp8Cutlass(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                      float alpha) {
  VT_CHECK(out.rank == 2 && a_fp8.rank == 2 && b_fp8.rank == 2,
           "matmul_fp8_cutlass: all tensors must be rank-2");
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1], n = b_fp8.shape[0];
  VT_CHECK(k % 16 == 0 && n % 16 == 0, "matmul_fp8_cutlass: K and N must be multiples of 16");
  VT_CHECK(b_fp8.shape[1] == k, "matmul_fp8_cutlass: b_fp8 must be [N, K] (K matches a_fp8)");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_fp8_cutlass: out must be [M, N]");
  VT_CHECK(out.dtype == DType::kBF16 || out.dtype == DType::kF32,
           "matmul_fp8_cutlass: out must be bf16 or f32 (bf16 epilogue, f32 via cast)");
  VT_CHECK(a_fp8.dtype == DType::kI8 && b_fp8.dtype == DType::kI8,
           "matmul_fp8_cutlass: a_fp8/b_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(out.IsContiguous() && a_fp8.IsContiguous() && b_fp8.IsContiguous(),
           "matmul_fp8_cutlass: contiguous tensors required");
  VT_CHECK(out.device == q.device && a_fp8.device == q.device && b_fp8.device == q.device,
           "matmul_fp8_cutlass: device mismatch");
  reinterpret_cast<MatmulFp8CutlassFn>(GetOp(OpId::kMatmulFp8Cutlass, q.device.type))(
      q, out, a_fp8, b_fp8, alpha);
}
void MatmulFp8CublasLt(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                       float alpha) {
  // Same argument contract as MatmulFp8Cutlass (drop-in fp8 dense GEMM).
  VT_CHECK(out.rank == 2 && a_fp8.rank == 2 && b_fp8.rank == 2,
           "matmul_fp8_cublaslt: all tensors must be rank-2");
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1], n = b_fp8.shape[0];
  VT_CHECK(k % 16 == 0 && n % 16 == 0, "matmul_fp8_cublaslt: K and N must be multiples of 16");
  VT_CHECK(b_fp8.shape[1] == k, "matmul_fp8_cublaslt: b_fp8 must be [N, K] (K matches a_fp8)");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_fp8_cublaslt: out must be [M, N]");
  VT_CHECK(out.dtype == DType::kBF16 || out.dtype == DType::kF32,
           "matmul_fp8_cublaslt: out must be bf16 or f32");
  VT_CHECK(a_fp8.dtype == DType::kI8 && b_fp8.dtype == DType::kI8,
           "matmul_fp8_cublaslt: a_fp8/b_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(out.IsContiguous() && a_fp8.IsContiguous() && b_fp8.IsContiguous(),
           "matmul_fp8_cublaslt: contiguous tensors required");
  VT_CHECK(out.device == q.device && a_fp8.device == q.device && b_fp8.device == q.device,
           "matmul_fp8_cublaslt: device mismatch");
  reinterpret_cast<MatmulFp8CublasLtFn>(GetOp(OpId::kMatmulFp8CublasLt, q.device.type))(
      q, out, a_fp8, b_fp8, alpha);
}

void MoeGroupedGemmNvfp4(Queue& q, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                         const Tensor* row_map, const Tensor& packed_ptrs,
                         const Tensor& scale_ptrs, const Tensor& scale2s) {
  VT_CHECK(out.rank == 2 && act.rank == 2, "moe_grouped_gemm_nvfp4: out/act must be rank-2");
  const int64_t p = out.shape[0], k = act.shape[1], e = scale2s.shape[0];
  VT_CHECK(k % 16 == 0, "moe_grouped_gemm_nvfp4: K (act inner dim) must be a multiple of 16");
  VT_CHECK(expert_ids.Numel() == p,
           "moe_grouped_gemm_nvfp4: expert_ids must have P entries (one per out row)");
  VT_CHECK(expert_ids.dtype == DType::kI32, "moe_grouped_gemm_nvfp4: expert_ids must be i32");
  VT_CHECK(packed_ptrs.Numel() == e && scale_ptrs.Numel() == e,
           "moe_grouped_gemm_nvfp4: packed_ptrs/scale_ptrs must have E entries");
  VT_CHECK(packed_ptrs.dtype == DType::kI64 && scale_ptrs.dtype == DType::kI64,
           "moe_grouped_gemm_nvfp4: packed_ptrs/scale_ptrs must be i64 (device pointers)");
  VT_CHECK(scale2s.dtype == DType::kF32, "moe_grouped_gemm_nvfp4: scale2s must be f32");
  VT_CHECK(IsFloat(act.dtype) && IsOutFloat(out.dtype),
           "moe_grouped_gemm_nvfp4: float act, f32/bf16 out");
  VT_CHECK(act.IsContiguous() && out.IsContiguous() && expert_ids.IsContiguous() &&
               packed_ptrs.IsContiguous() && scale_ptrs.IsContiguous() && scale2s.IsContiguous(),
           "moe_grouped_gemm_nvfp4: contiguous tensors required");
  VT_CHECK(act.device == q.device && out.device == q.device && expert_ids.device == q.device &&
               packed_ptrs.device == q.device && scale_ptrs.device == q.device &&
               scale2s.device == q.device,
           "moe_grouped_gemm_nvfp4: device mismatch");
  if (row_map != nullptr) {
    VT_CHECK(row_map->Numel() == p && row_map->dtype == DType::kI32 && row_map->IsContiguous() &&
                 row_map->device == q.device,
             "moe_grouped_gemm_nvfp4: row_map must be contiguous i32 [P] on the queue device");
  }
  reinterpret_cast<MoeGroupedGemmNvfp4Fn>(GetOp(OpId::kMoeGroupedGemmNvfp4, q.device.type))(
      q, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s);
}

void MoeGroupedGemmBf16(Queue& q, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                        const Tensor* row_map, const Tensor& weight_ptrs) {
  VT_CHECK(out.rank == 2 && act.rank == 2, "moe_grouped_gemm_bf16: out/act must be rank-2");
  const int64_t p = out.shape[0], e = weight_ptrs.shape[0];
  VT_CHECK(act.dtype == DType::kBF16, "moe_grouped_gemm_bf16: act must be bf16");
  VT_CHECK(IsOutFloat(out.dtype), "moe_grouped_gemm_bf16: out must be f32/bf16");
  VT_CHECK(expert_ids.Numel() == p,
           "moe_grouped_gemm_bf16: expert_ids must have P entries (one per out row)");
  VT_CHECK(expert_ids.dtype == DType::kI32, "moe_grouped_gemm_bf16: expert_ids must be i32");
  VT_CHECK(weight_ptrs.Numel() == e && weight_ptrs.dtype == DType::kI64,
           "moe_grouped_gemm_bf16: weight_ptrs must be i64 [E] (device pointers)");
  VT_CHECK(act.IsContiguous() && out.IsContiguous() && expert_ids.IsContiguous() &&
               weight_ptrs.IsContiguous(),
           "moe_grouped_gemm_bf16: contiguous tensors required");
  VT_CHECK(act.device == q.device && out.device == q.device && expert_ids.device == q.device &&
               weight_ptrs.device == q.device,
           "moe_grouped_gemm_bf16: device mismatch");
  if (row_map != nullptr) {
    VT_CHECK(row_map->Numel() == p && row_map->dtype == DType::kI32 && row_map->IsContiguous() &&
                 row_map->device == q.device,
             "moe_grouped_gemm_bf16: row_map must be contiguous i32 [P] on the queue device");
  }
  reinterpret_cast<MoeGroupedGemmBf16Fn>(GetOp(OpId::kMoeGroupedGemmBf16, q.device.type))(
      q, out, act, expert_ids, row_map, weight_ptrs);
}

void MoeGroupedGemmNvfp4Marlin(Queue& q, Tensor& c, const Tensor& a, const Tensor& b_q_weight,
                               const Tensor& b_scales, const Tensor& global_scale,
                               Tensor& workspace, const Tensor& sorted_token_ids,
                               const Tensor& expert_ids, const Tensor& num_tokens_past_padded,
                               const Tensor& topk_weights, const MoeMarlinArgs& args) {
  VT_CHECK(a.rank == 2 && c.rank == 2, "moe_marlin: a/c must be rank-2");
  VT_CHECK(a.dtype == DType::kBF16 && c.dtype == DType::kBF16, "moe_marlin: a/c must be bf16");
  VT_CHECK(args.size_k % 16 == 0, "moe_marlin: size_k must be a multiple of 16 (group size)");
  VT_CHECK(a.shape[0] == args.size_m && a.shape[1] == args.size_k,
           "moe_marlin: a shape must be [size_m, size_k]");
  VT_CHECK(b_q_weight.rank == 3, "moe_marlin: b_q_weight must be rank-3 [E, K/16, N*8/pack]");
  VT_CHECK(expert_ids.dtype == DType::kI32 && sorted_token_ids.dtype == DType::kI32 &&
               num_tokens_past_padded.dtype == DType::kI32,
           "moe_marlin: align tensors must be i32");
  VT_CHECK(global_scale.dtype == DType::kF32 && topk_weights.dtype == DType::kF32,
           "moe_marlin: global_scale/topk_weights must be f32");
  VT_CHECK(workspace.dtype == DType::kI32, "moe_marlin: workspace must be i32 (reduction locks)");
  reinterpret_cast<MoeGroupedGemmNvfp4MarlinFn>(
      GetOp(OpId::kMoeGroupedGemmNvfp4Marlin, q.device.type))(
      q, c, a, b_q_weight, b_scales, global_scale, workspace, sorted_token_ids, expert_ids,
      num_tokens_past_padded, topk_weights, args);
}

void MoeSiluMul(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up) {
  VT_CHECK(gate.Numel() == out.Numel() && up.Numel() == out.Numel(),
           "moe_silu_mul: out/gate/up must have the same element count");
  VT_CHECK(IsFloat(gate.dtype) && IsFloat(up.dtype) && IsOutFloat(out.dtype),
           "moe_silu_mul: float gate/up, f32/bf16 out");
  VT_CHECK(out.IsContiguous() && gate.IsContiguous() && up.IsContiguous(),
           "moe_silu_mul: contiguous tensors required");
  VT_CHECK(out.device == q.device && gate.device == q.device && up.device == q.device,
           "moe_silu_mul: device mismatch (out/gate/up/queue)");
  reinterpret_cast<MoeSiluMulFn>(GetOp(OpId::kMoeSiluMul, q.device.type))(q, out, gate, up);
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
    VT_CHECK((residual->dtype == DType::kF32 || residual->dtype == DType::kBF16) &&
                 residual->rank == 2 &&
                 residual->shape[0] == x.shape[0] && residual->shape[1] == x.shape[1] &&
                 residual->IsContiguous() && residual->device == x.device,
             "rmsnorm: residual must be f32/bf16 [T,H] contiguous on x's device");
  }
  VT_CHECK(x.device == out.device && weight.device == x.device && x.device == q.device,
           "rmsnorm: device mismatch (x/out/weight/queue)");
  reinterpret_cast<RmsNormFn>(GetOp(OpId::kRmsNorm, q.device.type))(q, out, x, weight, args,
                                                                    residual);
}

namespace {

// Fetch the tensor bound to operand slot `idx`, checked non-null.
Tensor* FusedOp(const FusedBinding& b, uint8_t idx, const char* what) {
  VT_CHECK(idx < b.n, "fused_chain: operand index out of range");
  VT_CHECK(b.op[idx] != nullptr, what);
  return b.op[idx];
}

// Tier-0 composite: walk the recipe DISPATCHING each opcode to the already-
// registered standalone vt:: op. Device-agnostic — every op self-dispatches on
// q.device, so the same walker realizes CPU and CUDA. Byte-exact by construction
// to the unfused standalone-op sequence the model hand-calls (that IS the golden).
// The residual-add idiom (kAdd writing the residual) folds into the following
// norm's RmsNorm(residual) call — the only form whose f32 add-then-normalize the
// standalone op reproduces bit-for-bit.
void FusedChainCompositeImpl(Queue& q, const FusedRecipe& r, const FusedBinding& b,
                             const FusedParams& p) {
  Tensor* add_x = nullptr;    // pending residual-add: x operand
  Tensor* add_res = nullptr;  // pending residual-add: residual operand (also the out)
  bool add_pending = false;
  for (int s = 0; s < r.n; ++s) {
    const FStep& st = r.steps[s];
    switch (st.op) {
      case FOp::kAdd:
        // Residual-add producing the residual stream: fold into the next kRmsNorm.
        VT_CHECK(st.nin == 2 && st.out == st.in[1],
                 "fused_chain composite: kAdd must be residual-add (out==in[1])");
        add_x = FusedOp(b, st.in[0], "fused_chain: null add input");
        add_res = FusedOp(b, st.out, "fused_chain: null residual");
        add_pending = true;
        break;
      case FOp::kRmsNorm: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null rmsnorm out");
        Tensor* w = FusedOp(b, st.in[1], "fused_chain: null rmsnorm weight");
        if (add_pending) {
          RmsNorm(q, *out, *add_x, *w, RmsNormArgs{p.eps, st.gemma}, add_res);
          add_pending = false;
        } else {
          Tensor* a = FusedOp(b, st.in[0], "fused_chain: null rmsnorm input");
          RmsNorm(q, *out, *a, *w, RmsNormArgs{p.eps, st.gemma}, nullptr);
        }
        break;
      }
      case FOp::kRmsNormGated: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null gated-norm out");
        Tensor* x = FusedOp(b, st.in[0], "fused_chain: null gated-norm x");
        Tensor* gate = FusedOp(b, st.in[1], "fused_chain: null gated-norm gate");
        Tensor* w = FusedOp(b, st.in[2], "fused_chain: null gated-norm weight");
        RmsNormGated(q, *out, *x, *gate, *w, RmsNormGatedArgs{p.eps, st.sigmoid_gate});
        break;
      }
      case FOp::kSiluMul: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null silu_mul out");
        Tensor* gate = FusedOp(b, st.in[0], "fused_chain: null silu_mul gate");
        Tensor* up = FusedOp(b, st.in[1], "fused_chain: null silu_mul up");
        MoeSiluMul(q, *out, *gate, *up);
        break;
      }
      case FOp::kSigmoidGate: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null sigmoid_gate out");
        Tensor* attn = FusedOp(b, st.in[0], "fused_chain: null sigmoid_gate attn");
        Tensor* gate = FusedOp(b, st.in[1], "fused_chain: null sigmoid_gate gate");
        SigmoidGateBf16(q, *out, *attn, *gate);
        break;
      }
      case FOp::kRope: {
        Tensor* qs = FusedOp(b, st.out, "fused_chain: null rope q");
        Tensor* cos_sin = FusedOp(b, st.in[1], "fused_chain: null rope cos_sin cache");
        Tensor* pos = FusedOp(b, st.in[2], "fused_chain: null rope positions");
        Tensor* ks = (st.out2 == kNoOperand) ? nullptr : b.op[st.out2];
        RopeFromCache(q, *qs, ks, *pos, *cos_sin, p.rope);
        break;
      }
      case FOp::kQuantFp8: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null fp8 out");
        Tensor* a = FusedOp(b, st.in[0], "fused_chain: null fp8 input");
        QuantFp8Static(q, *out, *a, p.quant_scale);
        break;
      }
      case FOp::kQuantFp4: {
        Tensor* packed = FusedOp(b, st.out, "fused_chain: null fp4 packed out");
        VT_CHECK(st.out2 != kNoOperand, "fused_chain: kQuantFp4 needs an out_scale (out2)");
        Tensor* scale = FusedOp(b, st.out2, "fused_chain: null fp4 scale out");
        Tensor* a = FusedOp(b, st.in[0], "fused_chain: null fp4 input");
        ScaledFp4Quant(q, *packed, *scale, *a, p.quant_scale, p.fp4_layout);
        break;
      }
      case FOp::kAttnQkNormRopeGate: {
        // Composite MACRO: dispatch the whole fused preamble to the one standalone
        // op. Operand order is fixed (recipes.h): [qgate, kf, q_norm, k_norm,
        // cos_sin, q_out, k_out, gate_out].
        VT_CHECK(r.n_operands == 8 && b.n == 8, "fused_chain: attn preamble needs 8 operands");
        AttnQkNormRopeGate(q, *FusedOp(b, 5, "q_out"), *FusedOp(b, 6, "k_out"),
                           *FusedOp(b, 7, "gate_out"), *FusedOp(b, 0, "qgate"),
                           *FusedOp(b, 1, "kf"), *FusedOp(b, 2, "q_norm"),
                           *FusedOp(b, 3, "k_norm"), *FusedOp(b, 4, "cos_sin"),
                           RmsNormArgs{p.eps, st.gemma}, p.rope);
        break;
      }
      case FOp::kMul:
      case FOp::kSilu:
      case FOp::kSigmoid:
        VT_CHECK(false,
                 "fused_chain composite: granular kMul/kSilu/kSigmoid have no standalone op "
                 "(Tier-1 vocabulary only)");
        break;
    }
  }
  VT_CHECK(!add_pending, "fused_chain composite: residual-add without a following rmsnorm");
}

// FAST realization (W2): dispatch a recipe bound to a bespoke single-launch fused
// kernel (recipe.fast_op) to that kernel via its existing vt:: wrapper. One switch
// case per fast realization — the additive surface (O(1), mirrors the composite's
// per-opcode switch). Each case unpacks the recipe's indexed operand table into the
// wrapper's arguments; the wrapper self-dispatches on q.device. Byte-exact to the
// composite by construction (it IS the kernel the composite is validated against).
void DispatchFusedFast(Queue& q, const FusedRecipe& r, const FusedBinding& b,
                       const FusedParams& p) {
  switch (static_cast<OpId>(r.fast_op)) {
    case OpId::kRmsNormQuantFp8: {
      // operands: 0=x, 1=weight, 2=residual?, 3=tmp_bf16 (optional out_bf16), 4=out_fp8
      Tensor* out_fp8 = FusedOp(b, 4, "fused_chain fast: null fp8 out");
      Tensor* x = FusedOp(b, 0, "fused_chain fast: null x");
      Tensor* w = FusedOp(b, 1, "fused_chain fast: null weight");
      Tensor* residual = b.op[2];  // optional
      Tensor* out_bf16 = b.op[3];  // optional (bf16 normed activation consumer)
      RmsNormQuantFp8(q, *out_fp8, out_bf16, *x, *w, RmsNormArgs{p.eps, r.steps[1].gemma},
                      residual, p.quant_scale);
      break;
    }
    case OpId::kRmsNormGatedQuantFp8: {
      // operands: 0=x, 1=gate, 2=weight, 3=tmp_bf16 (unused), 4=out_fp8
      Tensor* out_fp8 = FusedOp(b, 4, "fused_chain fast: null gated fp8 out");
      Tensor* x = FusedOp(b, 0, "fused_chain fast: null gated x");
      Tensor* gate = FusedOp(b, 1, "fused_chain fast: null gated gate");
      Tensor* w = FusedOp(b, 2, "fused_chain fast: null gated weight");
      RmsNormGatedQuantFp8(q, *out_fp8, *x, *gate, *w,
                           RmsNormGatedArgs{p.eps, r.steps[0].sigmoid_gate}, p.quant_scale);
      break;
    }
    case OpId::kSiluMulFp4Quant: {
      // operands: 0=gate, 1=up, 2=tmp_bf16 (unused), 3=out_packed, 4=out_scale
      Tensor* packed = FusedOp(b, 3, "fused_chain fast: null silu_mul packed");
      Tensor* scale = FusedOp(b, 4, "fused_chain fast: null silu_mul scale");
      Tensor* gate = FusedOp(b, 0, "fused_chain fast: null silu_mul gate");
      Tensor* up = FusedOp(b, 1, "fused_chain fast: null silu_mul up");
      SiluMulFp4Quant(q, *packed, *scale, *gate, *up, p.quant_scale, p.fp4_layout);
      break;
    }
    case OpId::kSigmoidGateFp4Quant: {
      // operands: 0=attn, 1=gate, 2=tmp_bf16 (unused), 3=out_packed, 4=out_scale
      Tensor* packed = FusedOp(b, 3, "fused_chain fast: null sigmoid_gate packed");
      Tensor* scale = FusedOp(b, 4, "fused_chain fast: null sigmoid_gate scale");
      Tensor* attn = FusedOp(b, 0, "fused_chain fast: null sigmoid_gate attn");
      Tensor* gate = FusedOp(b, 1, "fused_chain fast: null sigmoid_gate gate");
      SigmoidGateFp4Quant(q, *packed, *scale, *attn, *gate, p.quant_scale, p.fp4_layout);
      break;
    }
    default:
      VT_CHECK(false, "fused_chain: recipe.fast_op has no fast-realization case");
  }
}

}  // namespace

void FusedChainComposite(Queue& q, const FusedRecipe& recipe, const FusedBinding& binding,
                         const FusedParams& params) {
  VT_CHECK(recipe.n >= 1 && recipe.n <= kMaxFusedSteps, "fused_chain: empty/oversized recipe");
  VT_CHECK(recipe.n_operands == binding.n && binding.n <= kMaxFusedOperands,
           "fused_chain: binding size must match recipe operand count");
  FusedChainCompositeImpl(q, recipe, binding, params);
}

void FusedChain(Queue& q, const FusedRecipe& recipe, const FusedBinding& binding,
                const FusedParams& params) {
  VT_CHECK(recipe.n >= 1 && recipe.n <= kMaxFusedSteps, "fused_chain: empty/oversized recipe");
  VT_CHECK(recipe.n_operands == binding.n && binding.n <= kMaxFusedOperands,
           "fused_chain: binding size must match recipe operand count");

  // FAST realization: a recipe bound to an existing bespoke fused kernel dispatches
  // to it whenever the backend provides that OpId — the SAME single-launch kernel the
  // model called directly before W2 migration, so this is perf-neutral by
  // construction (no extra kernel, no getenv, no allocation on the path). A backend
  // that lacks the fast kernel falls through to the byte-exact composite below.
  if (recipe.fast_op != kNoFastOp &&
      OpRegistered(static_cast<OpId>(recipe.fast_op), q.device.type)) {
    DispatchFusedFast(q, recipe, binding, params);
    return;
  }

  // Tier-1 interpreter path: only for Tier-1-able recipes (all elementwise/rms)
  // over the canonical operand order [x, weight, residual?, out]. Everything else
  // (quant/rope/gated/attn) runs through the byte-exact composite.
  if (RecipeIsTier1Able(recipe) && FusedTier() == 1) {
    Tensor* x = FusedOp(binding, 0, "fused_chain: null x");
    Tensor* weight = FusedOp(binding, 1, "fused_chain: null weight");
    Tensor* residual =
        (binding.n >= 3 && recipe.operands[2].kind == FKind::kResidual) ? binding.op[2] : nullptr;
    Tensor* out = FusedOp(binding, static_cast<uint8_t>(recipe.steps[recipe.n - 1].out),
                          "fused_chain: null out");
    reinterpret_cast<FusedChainFn>(GetOp(OpId::kFusedChain, q.device.type))(
        q, *out, *x, *weight, residual, recipe, params.eps);
    return;
  }
  FusedChainComposite(q, recipe, binding, params);
}

void FusedChain(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, Tensor* residual,
                const FusedRecipe& recipe, float eps) {
  // Canonical 4-operand shape (the W0-adopted kFusedAddRmsNorm site): x/out [T,H],
  // weight [H], optional residual [T,H] f32/bf16. Validate at the chokepoint so a
  // bad call fails here, not inside a kernel, then bind [x, weight, residual, out]
  // and forward to the general entry.
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 1,
           "fused_chain: x/out rank-2, weight rank-1");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "fused_chain: out shape must match x");
  VT_CHECK(weight.shape[0] == x.shape[1], "fused_chain: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsOutFloat(out.dtype),
           "fused_chain: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "fused_chain: contiguous required");
  if (residual != nullptr) {
    VT_CHECK((residual->dtype == DType::kF32 || residual->dtype == DType::kBF16) &&
                 residual->rank == 2 && residual->shape[0] == x.shape[0] &&
                 residual->shape[1] == x.shape[1] && residual->IsContiguous() &&
                 residual->device == x.device,
             "fused_chain: residual must be f32/bf16 [T,H] contiguous on x's device");
  }
  VT_CHECK(x.device == out.device && weight.device == x.device && x.device == q.device,
           "fused_chain: device mismatch (x/out/weight/queue)");
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&x);
  binding.op[1] = const_cast<Tensor*>(&weight);
  binding.op[2] = residual;
  binding.op[3] = &out;
  binding.n = 4;
  FusedChain(q, recipe, binding, FusedParams{eps, 1.0f, Fp4ScaleLayout::kLinear, RopeArgs{}});
}

void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_packed, Tensor& out_scale,
                const Tensor& a, const Tensor& b, float quant_scale,
                Fp4ScaleLayout scale_layout) {
  // Fp4-activation-quant shape (kSiluMulFp4Quant, kSigmoidGateFp4Quant): bind the
  // recipe's [a, b, tmp_bf16, out_packed, out_scale] table (tmp_bf16 = nullptr; the
  // fast realization never materializes it) and dispatch to the recipe's fast_op.
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&a);
  binding.op[1] = const_cast<Tensor*>(&b);
  binding.op[2] = nullptr;  // tmp_bf16 (composite-only scratch)
  binding.op[3] = &out_packed;
  binding.op[4] = &out_scale;
  binding.n = 5;
  FusedChain(q, recipe, binding, FusedParams{1e-6f, quant_scale, scale_layout, RopeArgs{}});
}

void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_fp8, Tensor* out_bf16,
                const Tensor& x, const Tensor& weight, Tensor* residual, float eps,
                float input_scale) {
  // RmsNorm->fp8 shape (kRmsNormQuantFp8): bind [x, weight, residual, out_bf16,
  // out_fp8]. residual / out_bf16 pass through as-is (may be nullptr), matching the
  // bespoke RmsNormQuantFp8 contract; tmp_bf16 slot IS the optional out_bf16 here.
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&x);
  binding.op[1] = const_cast<Tensor*>(&weight);
  binding.op[2] = residual;
  binding.op[3] = out_bf16;
  binding.op[4] = &out_fp8;
  binding.n = 5;
  FusedChain(q, recipe, binding, FusedParams{eps, input_scale, Fp4ScaleLayout::kLinear, RopeArgs{}});
}

void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_fp8, const Tensor& x,
                const Tensor& gate, const Tensor& weight, float eps, float input_scale) {
  // Gated-RmsNorm->fp8 shape (kRmsNormGatedQuantFp8): bind [x, gate, weight,
  // tmp_bf16, out_fp8] (tmp_bf16 = nullptr; fast realization skips it).
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&x);
  binding.op[1] = const_cast<Tensor*>(&gate);
  binding.op[2] = const_cast<Tensor*>(&weight);
  binding.op[3] = nullptr;  // tmp_bf16 (composite-only scratch)
  binding.op[4] = &out_fp8;
  binding.n = 5;
  FusedChain(q, recipe, binding, FusedParams{eps, input_scale, Fp4ScaleLayout::kLinear, RopeArgs{}});
}

void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& q_out, Tensor& k_out,
                Tensor& gate_out, const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                const Tensor& k_norm, const Tensor& cos_sin, float eps, const RopeArgs& rope) {
  // Attn-preamble MACRO shape (kAttnQkNormRopeGate): bind the fixed operand table
  // [qgate, kf, q_norm, k_norm, cos_sin, q_out, k_out, gate_out]. No fast_op — the
  // composite macro dispatches to the single vt::AttnQkNormRopeGate kernel.
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&qgate);
  binding.op[1] = const_cast<Tensor*>(&kf);
  binding.op[2] = const_cast<Tensor*>(&q_norm);
  binding.op[3] = const_cast<Tensor*>(&k_norm);
  binding.op[4] = const_cast<Tensor*>(&cos_sin);
  binding.op[5] = &q_out;
  binding.op[6] = &k_out;
  binding.op[7] = &gate_out;
  binding.n = 8;
  FusedChain(q, recipe, binding, FusedParams{eps, 1.0f, Fp4ScaleLayout::kLinear, rope});
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

void LayerNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
               const Tensor* bias, const LayerNormArgs& args) {
  VT_CHECK(x.rank >= 1 && out.rank == x.rank, "layer_norm: out rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out.shape[i] == x.shape[i], "layer_norm: out shape must match x");
  const int64_t d = x.shape[x.rank - 1];
  VT_CHECK(d > 0, "layer_norm: last dim must be non-empty");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "layer_norm: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "layer_norm: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "layer_norm: device mismatch");
  for (const Tensor* p : {weight, bias}) {
    if (p == nullptr) continue;
    VT_CHECK(p->rank == 1 && p->shape[0] == d,
             "layer_norm: weight/bias must be rank-1 [D] matching x's last dim");
    VT_CHECK(IsFloat(p->dtype), "layer_norm: float weight/bias required");
    VT_CHECK(p->IsContiguous() && p->device == x.device,
             "layer_norm: weight/bias must be contiguous on x's device");
  }
  VT_CHECK(args.eps >= 0.0f, "layer_norm: eps must be non-negative");
  reinterpret_cast<LayerNormFn>(GetOp(OpId::kLayerNorm, q.device.type))(q, out, x, weight, bias,
                                                                       args);
}

void Relu(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(out.rank == x.rank, "relu: out rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out.shape[i] == x.shape[i], "relu: out shape must match x");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "relu: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "relu: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "relu: device mismatch");
  reinterpret_cast<ReluFn>(GetOp(OpId::kRelu, q.device.type))(q, out, x);
}

void Add(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank >= 1 && out.rank == a.rank, "add: out rank must match a");
  for (int i = 0; i < a.rank; ++i)
    VT_CHECK(out.shape[i] == a.shape[i], "add: out shape must match a");
  const int64_t d = a.shape[a.rank - 1];
  // Two accepted b shapes: a's exact shape (elementwise) or rank-1 [D] over the
  // last dim (a nn.Linear bias row-broadcast).
  const bool broadcast = b.rank == 1 && a.rank != 1;
  if (broadcast) {
    VT_CHECK(b.shape[0] == d, "add: rank-1 b must match a's last dim (bias broadcast)");
  } else {
    VT_CHECK(b.rank == a.rank, "add: b must match a's rank or be rank-1 [D]");
    for (int i = 0; i < a.rank; ++i)
      VT_CHECK(b.shape[i] == a.shape[i], "add: b shape must match a");
  }
  VT_CHECK(IsFloat(a.dtype) && IsFloat(b.dtype) && IsOutFloat(out.dtype),
           "add: float in, f32/bf16 out");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "add: contiguous required");
  VT_CHECK(a.device == out.device && b.device == a.device && a.device == q.device,
           "add: device mismatch");
  reinterpret_cast<AddFn>(GetOp(OpId::kAdd, q.device.type))(q, out, a, b);
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

void RopeFromCache(Queue& q, Tensor& q_states, Tensor* k_states,
                   const Tensor& positions, const Tensor& cos_sin_cache,
                   const RopeArgs& args) {
  VT_CHECK(q_states.rank == 3, "rope_from_cache: q must be rank-3 [T,H,D]");
  const int64_t tokens = q_states.shape[0];
  const int64_t head_dim = q_states.shape[2];
  if (k_states != nullptr) {
    VT_CHECK(k_states->rank == 3 && k_states->shape[0] == tokens &&
                 k_states->shape[2] == head_dim,
             "rope_from_cache: k must be [T,Hk,D] matching q");
    VT_CHECK(k_states->dtype == q_states.dtype,
             "rope_from_cache: q/k dtype mismatch");
    // MLA campaign W6: only the innermost dim must be unit-stride. DeepSeek's
    // k_pe is the trailing column block of the single fused
    // `kv_a_proj_with_mqa` output (deepseek_v2.py:511, mla.py:155-162), i.e. a
    // STRIDED view; requiring contiguity would force a copy per layer per step.
    VT_CHECK(k_states->stride[2] == 1,
             "rope_from_cache: k innermost dimension must be unit-stride");
    VT_CHECK(k_states->device == q.device,
             "rope_from_cache: k/queue device mismatch");
  }
  VT_CHECK(positions.rank == 1 || positions.rank == 2,
           "rope_from_cache: positions must be [T] or [3,T]");
  if (positions.rank == 1) {
    VT_CHECK(positions.shape[0] == tokens,
             "rope_from_cache: positions[T] mismatch");
  } else {
    VT_CHECK(positions.shape[0] == 3 && positions.shape[1] == tokens,
             "rope_from_cache: MRoPE positions must be [3,T]");
    int64_t section_sum = 0;
    for (int32_t section : args.mrope_section) {
      VT_CHECK(section >= 0,
               "rope_from_cache: negative mrope_section entry");
      section_sum += section;
    }
    VT_CHECK(section_sum == args.rotary_dim / 2,
             "rope_from_cache: mrope_section must sum to rotary_dim/2");
  }
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "rope_from_cache: positions must be i32/i64");
  VT_CHECK(cos_sin_cache.rank == 2 && cos_sin_cache.shape[0] > 0 &&
               cos_sin_cache.shape[1] == args.rotary_dim,
           "rope_from_cache: cache must be [P,rotary_dim]");
  VT_CHECK(IsOutFloat(q_states.dtype) &&
               cos_sin_cache.dtype == q_states.dtype,
           "rope_from_cache: q/k/cache must share f32 or bf16 dtype");
  VT_CHECK(args.rotary_dim > 0 && args.rotary_dim % 2 == 0 &&
               args.rotary_dim <= head_dim,
           "rope_from_cache: rotary_dim must be even and <= head_dim");
  // q likewise only needs a unit-stride innermost dim: DeepSeek rotates the
  // TRAILING qk_rope_head_dim slice of each query head
  // (mla.py:160-167 passes `q[..., qk_nope_head_dim:]`), a strided view of the
  // [T, num_heads, qk_head_dim] query. For contiguous tensors the kernels'
  // strided offsets are integer-identical to the pre-W6 formula.
  VT_CHECK(q_states.stride[2] == 1,
           "rope_from_cache: q innermost dimension must be unit-stride");
  VT_CHECK(positions.IsContiguous() && cos_sin_cache.IsContiguous(),
           "rope_from_cache: contiguous positions/cache required");
  VT_CHECK(q_states.device == q.device && positions.device == q.device &&
               cos_sin_cache.device == q.device,
           "rope_from_cache: q/positions/cache/queue device mismatch");
  reinterpret_cast<RopeFromCacheFn>(
      GetOp(OpId::kRopeFromCache, q.device.type))(
      q, q_states, k_states, positions, cos_sin_cache, args);
}

void RopeCosSinCache(Queue& q, Tensor& cos_sin, const Tensor& positions, const RopeArgs& args) {
  VT_CHECK(cos_sin.rank == 2, "rope_cos_sin_cache: cos_sin rank-2 [T, rotary_dim]");
  VT_CHECK(cos_sin.dtype == DType::kF32, "rope_cos_sin_cache: cos_sin must be f32");
  VT_CHECK(positions.rank == 1 && positions.shape[0] == cos_sin.shape[0],
           "rope_cos_sin_cache: positions[T] must match cos_sin leading dim");
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "rope_cos_sin_cache: positions i32/i64");
  VT_CHECK(args.rotary_dim > 0 && args.rotary_dim % 2 == 0 && cos_sin.shape[1] == args.rotary_dim,
           "rope_cos_sin_cache: cos_sin second dim must equal an even rotary_dim");
  VT_CHECK(cos_sin.IsContiguous() && positions.IsContiguous(),
           "rope_cos_sin_cache: contiguous required");
  VT_CHECK(cos_sin.device == q.device && positions.device == q.device,
           "rope_cos_sin_cache: device mismatch (cos_sin/positions/queue)");
  reinterpret_cast<RopeCosSinCacheFn>(GetOp(OpId::kRopeCosSinCache, q.device.type))(q, cos_sin,
                                                                                    positions, args);
}

void AttnQkNormRopeGate(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                        const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                        const Tensor& k_norm, const Tensor& cos_sin,
                        const RmsNormArgs& norm_args, const RopeArgs& rope_args) {
  VT_CHECK(q_out.rank == 3 && k_out.rank == 3 && gate_out.rank == 3,
           "attn_qk_norm_rope_gate: q_out/k_out/gate_out rank-3 [T,H,Dh]");
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  VT_CHECK(k_out.shape[0] == t && k_out.shape[2] == dh, "attn_qk_norm_rope_gate: k_out [T,Hkv,Dh]");
  VT_CHECK(gate_out.shape[0] == t && gate_out.shape[1] == hq && gate_out.shape[2] == dh,
           "attn_qk_norm_rope_gate: gate_out must match q_out [T,Hq,Dh]");
  VT_CHECK(qgate.rank == 2 && qgate.shape[0] == t && qgate.shape[1] == hq * 2 * dh,
           "attn_qk_norm_rope_gate: qgate must be [T, Hq*2*Dh]");
  VT_CHECK(kf.rank == 2 && kf.shape[0] == t && kf.shape[1] == hkv * dh,
           "attn_qk_norm_rope_gate: kf must be [T, Hkv*Dh]");
  VT_CHECK(q_norm.rank == 1 && q_norm.shape[0] == dh && k_norm.rank == 1 && k_norm.shape[0] == dh,
           "attn_qk_norm_rope_gate: q_norm/k_norm must be [Dh]");
  VT_CHECK(cos_sin.rank == 2 && cos_sin.shape[0] == t && cos_sin.shape[1] == rope_args.rotary_dim,
           "attn_qk_norm_rope_gate: cos_sin must be [T, rotary_dim]");
  VT_CHECK(rope_args.rotary_dim > 0 && rope_args.rotary_dim % 2 == 0 && rope_args.rotary_dim <= dh,
           "attn_qk_norm_rope_gate: rotary_dim must be even and <= Dh");
  // q/k share one out dtype; the gate may additionally stay f32 while q/k are
  // bf16 (the FA-2 prefill combo: bf16 q feeds FA-2 and bf16 k feeds the bf16
  // KV-cache write, but sigmoid(gate) must see the un-rounded f32 gate). All
  // kernel math is f32 either way; a bf16 store is the RN round of the same
  // value, so mixed out is bit-identical to f32-out + CastBf16 on q/k.
  VT_CHECK(IsOutFloat(q_out.dtype) && k_out.dtype == q_out.dtype &&
               (gate_out.dtype == q_out.dtype ||
                (q_out.dtype == DType::kBF16 && gate_out.dtype == DType::kF32)),
           "attn_qk_norm_rope_gate: q/k/gate out f32 or bf16 (gate f32 allowed with bf16 q/k)");
  VT_CHECK(IsFloat(qgate.dtype) && kf.dtype == qgate.dtype,
           "attn_qk_norm_rope_gate: qgate/kf float, same dtype");
  VT_CHECK(q_norm.dtype == DType::kF32 && k_norm.dtype == DType::kF32 &&
               cos_sin.dtype == DType::kF32,
           "attn_qk_norm_rope_gate: q_norm/k_norm/cos_sin must be f32");
  // QKVParallelLinear returns torch.split-style logical views: every Q/K row
  // is inner-contiguous, but stride(0) remains Q+K+V. Consume that layout
  // directly instead of adding split-copy kernels.
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() &&
               gate_out.IsContiguous() && qgate.stride[1] == 1 &&
               qgate.stride[0] >= qgate.shape[1] && kf.stride[1] == 1 &&
               kf.stride[0] >= kf.shape[1] && q_norm.IsContiguous() &&
               k_norm.IsContiguous() && cos_sin.IsContiguous(),
           "attn_qk_norm_rope_gate: outputs/weights/cache must be contiguous; "
           "qgate/kf rows must be inner-contiguous");
  VT_CHECK(q_out.device == q.device && k_out.device == q.device && gate_out.device == q.device &&
               qgate.device == q.device && kf.device == q.device && q_norm.device == q.device &&
               k_norm.device == q.device && cos_sin.device == q.device,
           "attn_qk_norm_rope_gate: device mismatch");
  reinterpret_cast<AttnQkNormRopeGateFn>(GetOp(OpId::kAttnQkNormRopeGate, q.device.type))(
      q, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm, cos_sin, norm_args, rope_args);
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
  VT_CHECK(conv_state.dtype == DType::kF32 ||
               (conv_state.dtype == DType::kBF16 && q.device.type == DeviceType::kCUDA),
           std::string(name) +
               ": conv_state must be f32, or bf16 on CUDA (in/out, in place; bf16 = "
               "vLLM default mamba_cache_dtype, read/written in f32 registers)");
  // out/weight/conv_state stay fully contiguous. x may be a padded-row
  // (inner-contiguous, outer stride >= C) view: the merged qkvz projection feeds
  // the conv its mixed_qkv = mixed_qkvz[:, :conv_dim] slice without any copy
  // (qwen_gdn_linear_attn.py:929-936). The kernels honor x.stride[0] directly.
  VT_CHECK(x.stride[1] == 1 && x.stride[0] >= c && out.IsContiguous() &&
               weight.IsContiguous() && conv_state.IsContiguous(),
           std::string(name) +
               ": out/weight/conv_state contiguous; x rows inner-contiguous "
               "(padded outer stride allowed)");
  VT_CHECK(x.device == q.device && out.device == q.device && weight.device == q.device &&
               conv_state.device == q.device,
           std::string(name) + ": device mismatch (x/out/weight/conv_state/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 1 && bias->shape[0] == c && IsFloat(bias->dtype) &&
                 bias->IsContiguous() && bias->device == q.device,
             std::string(name) + ": bias must be float [C] contiguous on the queue device");
  }
}

// Shared validation for the two decomposed delta-rule ops. q_in/k
// [T,Hk,Dk], v [T,Hv,Dv], g/beta [T,Hv] f32, state [N,Hv,Dv,Dk]
// fp16/bf16/fp32 on CUDA (independent Mamba temporal-cache dtype), out
// [T,Hv,Dv]. CPU keeps the f32 recurrence reference.
void CheckGdnCommon(const Queue& q, const Tensor& out, const Tensor& q_in, const Tensor& k,
                    const Tensor& v, const Tensor& g, const Tensor& beta, const Tensor& state,
                    const GdnArgs& args, const char* name,
                    bool allow_compressed_state) {
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
  if (allow_compressed_state) {
    VT_CHECK(state.dtype == DType::kF32 ||
                 ((state.dtype == DType::kF16 ||
                   state.dtype == DType::kBF16) &&
                  q.device.type == DeviceType::kCUDA),
             std::string(name) +
                 ": state must be f32, or fp16/bf16 on CUDA (in/out, in place; "
                 "read/written in f32 registers)");
  } else {
    VT_CHECK(state.dtype == DType::kF32,
             std::string(name) +
                 ": state must be f32; gather compressed cache rows first");
  }
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

void CheckBoolMeta(const Queue& q, const Tensor& t, int64_t expect_len,
                   const char* name, const char* what) {
  VT_CHECK(t.rank == 1 && t.shape[0] == expect_len &&
               (t.dtype == DType::kI8 || t.dtype == DType::kI32) &&
               t.IsContiguous() && t.device == q.device,
           std::string(name) + ": " + what + " must be i8/i32 [" +
               std::to_string(expect_len) + "] contiguous on the queue device");
}
}  // namespace

void CausalConv1dFwd(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, Tensor& conv_state, const Tensor& query_start_loc,
                     const Tensor& has_initial_state, const CausalConv1dArgs& args) {
  CheckConvCommon(q, out, x, weight, bias, conv_state, "causal_conv1d_fwd");
  const int64_t n = conv_state.shape[0];
  CheckI32Meta(q, query_start_loc, n + 1, "causal_conv1d_fwd", "query_start_loc");
  CheckBoolMeta(q, has_initial_state, n, "causal_conv1d_fwd", "has_initial_state");
  reinterpret_cast<CausalConv1dFwdFn>(GetOp(OpId::kCausalConv1dFwd, q.device.type))(
      q, out, x, weight, bias, conv_state, query_start_loc, has_initial_state, args);
}

void CausalConv1dUpdate(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                        const Tensor* bias, Tensor& conv_state, const CausalConv1dArgs& args,
                        const Tensor* conv_state_indices) {
  CheckConvCommon(q, out, x, weight, bias, conv_state, "causal_conv1d_update");
  if (conv_state_indices == nullptr) {
    VT_CHECK(conv_state.shape[0] == x.shape[0],
             "causal_conv1d_update: one conv_state row per token required");
  } else {
    // In-place indexed path: conv_state is the FULL cache; one index per token.
    CheckI32Meta(q, *conv_state_indices, x.shape[0], "causal_conv1d_update",
                 "conv_state_indices");
    VT_CHECK(conv_state.shape[0] >= x.shape[0],
             "causal_conv1d_update: indexed cache must have >= batch rows");
  }
  reinterpret_cast<CausalConv1dUpdateFn>(GetOp(OpId::kCausalConv1dUpdate, q.device.type))(
      q, out, x, weight, bias, conv_state, conv_state_indices, args);
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
  // Rank-2 [rows, D] (contiguous split path) OR rank-3 [T, Hv, D] (merged qkvz
  // path): the gate is the padded-row z = mixed_qkvz[:, conv_dim:] slice viewed
  // as [T, Hv, Dv], so its inner [Hv,Dv] block stays contiguous while the token
  // stride may exceed Hv*Dv (qwen_gdn_linear_attn.py:929-936). Normalization is
  // always over the LAST dim; rows = numel/D either way.
  VT_CHECK((x.rank == 2 || x.rank == 3) && gate.rank == x.rank &&
               out.rank == x.rank && weight.rank == 1,
           "rmsnorm_gated: x/gate/out rank-2 or rank-3, weight rank-1");
  for (int dd = 0; dd < x.rank; ++dd)
    VT_CHECK(gate.shape[dd] == x.shape[dd] && out.shape[dd] == x.shape[dd],
             "rmsnorm_gated: x/gate/out shapes must match");
  const int64_t d = x.shape[x.rank - 1];
  VT_CHECK(weight.shape[0] == d, "rmsnorm_gated: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(gate.dtype) && IsFloat(weight.dtype) &&
               IsOutFloat(out.dtype),
           "rmsnorm_gated: float in, f32/bf16 out");
  // x/out/weight stay contiguous; the gate may carry a padded outer (token)
  // stride with all inner dims contiguous.
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "rmsnorm_gated: x/out/weight contiguous required");
  bool gate_inner = true;
  int64_t gate_span = 1;
  for (int dd = gate.rank - 1; dd >= 1; --dd) {
    gate_inner = gate_inner && gate.stride[dd] == gate_span;
    gate_span *= gate.shape[dd];
  }
  gate_inner = gate_inner && gate.stride[0] >= gate_span;
  VT_CHECK(gate_inner,
           "rmsnorm_gated: gate rows inner-contiguous (padded outer stride "
           "allowed)");
  VT_CHECK(x.device == q.device && gate.device == q.device && weight.device == q.device &&
               out.device == q.device,
           "rmsnorm_gated: device mismatch (x/gate/weight/out/queue)");
  reinterpret_cast<RmsNormGatedFn>(GetOp(OpId::kRmsNormGated, q.device.type))(q, out, x, gate,
                                                                              weight, args);
}

void GdnPrefill(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                const Tensor& g, const Tensor& beta, Tensor& state,
                const Tensor& query_start_loc, const GdnArgs& args) {
  CheckGdnCommon(q, out, q_in, k, v, g, beta, state, args, "gdn_prefill",
                 /*allow_compressed_state=*/false);
  CheckI32Meta(q, query_start_loc, state.shape[0] + 1, "gdn_prefill", "query_start_loc");
  reinterpret_cast<GdnPrefillFn>(GetOp(OpId::kGdnPrefill, q.device.type))(
      q, out, q_in, k, v, g, beta, state, query_start_loc, args);
}

void GdnDecode(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
               const Tensor& g, const Tensor& beta, Tensor& state, const GdnArgs& args,
               const Tensor* state_idx) {
  CheckGdnCommon(q, out, q_in, k, v, g, beta, state, args, "gdn_decode",
                 /*allow_compressed_state=*/true);
  if (state_idx == nullptr) {
    VT_CHECK(state.shape[0] == q_in.shape[0],
             "gdn_decode: one state row per token required (single-token sequences)");
  } else {
    // In-place indexed path: state is the FULL cache; one slot index per token.
    CheckI32Meta(q, *state_idx, q_in.shape[0], "gdn_decode", "state_idx");
    VT_CHECK(state.shape[0] >= q_in.shape[0],
             "gdn_decode: indexed cache must have >= batch rows");
  }
  reinterpret_cast<GdnDecodeFn>(GetOp(OpId::kGdnDecode, q.device.type))(
      q, out, q_in, k, v, g, beta, state, state_idx, args);
}

void GdnPackedDecode(Queue& q, Tensor& out, const Tensor& mixed_qkv,
                     const Tensor& a, const Tensor& b, const Tensor& a_log,
                     const Tensor& dt_bias, Tensor& state,
                     const Tensor& state_idx, const GdnArgs& args) {
  constexpr const char* name = "gdn_packed_decode";
  VT_CHECK(mixed_qkv.rank == 2 && a.rank == 2 && b.rank == 2 &&
               a_log.rank == 1 && dt_bias.rank == 1 && out.rank == 3 &&
               state.rank == 4,
           "gdn_packed_decode: mixed_qkv/a/b rank-2, A_log/dt_bias rank-1, "
           "out rank-3, state rank-4 required");
  const int64_t batch = mixed_qkv.shape[0];
  const int64_t hv = state.shape[1];
  const int64_t dv = state.shape[2];
  const int64_t dk = state.shape[3];
  VT_CHECK(a.shape[0] == batch && b.shape[0] == batch && a.shape[1] == hv &&
               b.shape[1] == hv,
           "gdn_packed_decode: a/b must be [B,Hv]");
  VT_CHECK(a_log.shape[0] == hv && dt_bias.shape[0] == hv,
           "gdn_packed_decode: A_log/dt_bias must be [Hv]");
  VT_CHECK(out.shape[0] == batch && out.shape[1] == hv && out.shape[2] == dv,
           "gdn_packed_decode: out must be [B,Hv,Dv]");
  CheckI32Meta(q, state_idx, batch, name, "state_idx");

  const int64_t qk_dim = mixed_qkv.shape[1] - hv * dv;
  VT_CHECK(qk_dim > 0 && qk_dim % 2 == 0,
           "gdn_packed_decode: invalid packed mixed_qkv width");
  const int64_t q_dim = qk_dim / 2;
  VT_CHECK(dk > 0 && q_dim % dk == 0,
           "gdn_packed_decode: packed Q width must be divisible by Dk");
  const int64_t hk = q_dim / dk;
  VT_CHECK(hk > 0 && hv % hk == 0,
           "gdn_packed_decode: Hv must be a multiple of inferred Hk");

  VT_CHECK(IsFloat(mixed_qkv.dtype) && a.dtype == mixed_qkv.dtype &&
               b.dtype == mixed_qkv.dtype && out.dtype == mixed_qkv.dtype,
           "gdn_packed_decode: mixed_qkv/a/b/out must share FP16/BF16/F32 dtype");
  VT_CHECK(IsFloat(state.dtype),
           "gdn_packed_decode: state must use an independent FP16/BF16/F32 dtype");
  VT_CHECK(IsFloat(a_log.dtype) && IsFloat(dt_bias.dtype),
           "gdn_packed_decode: A_log/dt_bias must each use a floating dtype");
  VT_CHECK(mixed_qkv.stride[1] == 1 &&
               mixed_qkv.stride[0] >= mixed_qkv.shape[1] && a.stride[1] == 1 &&
               a.stride[0] >= hv && b.stride[1] == 1 && b.stride[0] >= hv,
           "gdn_packed_decode: mixed_qkv/a/b require inner-contiguous non-overlapping rows");
  VT_CHECK(a_log.IsContiguous() && dt_bias.IsContiguous() && out.IsContiguous() &&
               state.IsContiguous(),
           "gdn_packed_decode: A_log/dt_bias/out/state must be contiguous");
  VT_CHECK(mixed_qkv.device == q.device && a.device == q.device &&
               b.device == q.device && a_log.device == q.device &&
               dt_bias.device == q.device && out.device == q.device &&
               state.device == q.device,
           "gdn_packed_decode: device mismatch");
  VT_CHECK(args.scale > 0.0f,
           "gdn_packed_decode: args.scale must be set (> 0)");

  // CPU can validate the index values without synchronization. CUDA metadata
  // is engine-owned and remains device-resident; its kernel independently
  // bounds-checks each slot before dereferencing it.
  if (q.device.type == DeviceType::kCPU) {
    std::vector<uint8_t> seen(static_cast<size_t>(state.shape[0]), 0);
    const int32_t* idx = state_idx.Ptr<int32_t>();
    for (int64_t i = 0; i < batch; ++i) {
      if (idx[i] < 0) continue;
      VT_CHECK(idx[i] < state.shape[0],
               "gdn_packed_decode: state_idx out of range");
      VT_CHECK(seen[static_cast<size_t>(idx[i])] == 0,
               "gdn_packed_decode: duplicate live state_idx");
      seen[static_cast<size_t>(idx[i])] = 1;
    }
  }

  reinterpret_cast<GdnPackedDecodeFn>(
      GetOp(OpId::kGdnPackedDecode, q.device.type))(
      q, out, mixed_qkv, a, b, a_log, dt_bias, state, state_idx, args);
}

namespace {
void CheckGdnStateIo(const Queue& q, const Tensor& working,
                     const Tensor& cache, const Tensor& state_idx,
                     const char* name) {
  VT_CHECK(cache.rank >= 2 && cache.rank <= kMaxRank,
           std::string(name) + ": cache rank must be in [2,4]");
  VT_CHECK(working.rank == cache.rank,
           std::string(name) + ": working/cache ranks must match");
  VT_CHECK(state_idx.rank == 1 && state_idx.dtype == DType::kI32 &&
               state_idx.IsContiguous(),
           std::string(name) + ": state_idx must be contiguous i32 [N]");
  VT_CHECK(working.shape[0] == state_idx.shape[0],
           std::string(name) + ": working rows must match state_idx");
  for (int d = 1; d < cache.rank; ++d) {
    VT_CHECK(working.shape[d] == cache.shape[d],
             std::string(name) + ": working/cache row shapes must match");
  }
  VT_CHECK(working.dtype == DType::kF32,
           std::string(name) + ": working state must be f32");
  VT_CHECK(cache.dtype == DType::kF32 || cache.dtype == DType::kF16 ||
               cache.dtype == DType::kBF16,
           std::string(name) + ": cache must be fp16, bf16, or f32");
  VT_CHECK(working.IsContiguous() && cache.IsContiguous(),
           std::string(name) + ": working/cache must be contiguous");
  VT_CHECK(working.device == q.device && cache.device == q.device &&
               state_idx.device == q.device,
           std::string(name) + ": device mismatch");
}
}  // namespace

void GdnStateGather(Queue& q, Tensor& working, const Tensor& cache,
                    const Tensor& state_idx,
                    const Tensor* has_initial_state) {
  CheckGdnStateIo(q, working, cache, state_idx, "gdn_state_gather");
  if (has_initial_state != nullptr) {
    VT_CHECK(has_initial_state->rank == 1 &&
                 has_initial_state->shape[0] == state_idx.shape[0] &&
                 (has_initial_state->dtype == DType::kI8 ||
                  has_initial_state->dtype == DType::kI32) &&
                 has_initial_state->IsContiguous() &&
                 has_initial_state->device == q.device,
             "gdn_state_gather: has_initial_state must be contiguous i8/i32 [N] on device");
  }
  reinterpret_cast<GdnStateGatherFn>(
      GetOp(OpId::kGdnStateGather, q.device.type))(
      q, working, cache, state_idx, has_initial_state);
}

void GdnStateScatter(Queue& q, Tensor& cache, const Tensor& working,
                     const Tensor& state_idx) {
  CheckGdnStateIo(q, working, cache, state_idx, "gdn_state_scatter");
  reinterpret_cast<GdnStateScatterFn>(
      GetOp(OpId::kGdnStateScatter, q.device.type))(
      q, cache, working, state_idx);
}

void MoeRouterTopK(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                   const MoeRouterTopKArgs& args,
                   const Tensor* e_score_correction_bias) {
  VT_CHECK(logits.rank == 2 && weights.rank == 2 && indices.rank == 2,
           "moe_router_topk: logits/weights/indices rank-2");
  const int64_t t = logits.shape[0], e = logits.shape[1];
  // --- grouped-topk (`noaux_tc`) argument contract (W3). num_expert_group == 0
  // is the pre-W3 ungrouped path and must reject every grouped-only knob, so an
  // args struct that was half-filled fails loudly instead of silently ignoring
  // the grouping the caller asked for.
  if (args.num_expert_group > 0) {
    VT_CHECK(e % args.num_expert_group == 0,
             "moe_router_topk: num_experts must be divisible by num_expert_group");
    VT_CHECK(args.topk_group >= 1 && args.topk_group <= args.num_expert_group,
             "moe_router_topk: topk_group must be in [1, num_expert_group]");
    // grouped_topk_router.py masks all but topk_group groups, so at most
    // topk_group * (E / n_group) experts remain selectable.
    VT_CHECK(static_cast<int64_t>(args.top_k) <=
                 static_cast<int64_t>(args.topk_group) * (e / args.num_expert_group),
             "moe_router_topk: top_k exceeds the experts surviving the group mask");
  } else {
    VT_CHECK(args.topk_group == 0,
             "moe_router_topk: topk_group requires num_expert_group > 0");
    VT_CHECK(args.scoring_func == MoeScoringFunc::kSoftmax,
             "moe_router_topk: sigmoid scoring is only defined on the grouped "
             "(noaux_tc) path");
    VT_CHECK(args.routed_scaling_factor == 1.0f,
             "moe_router_topk: routed_scaling_factor requires num_expert_group > 0");
    VT_CHECK(e_score_correction_bias == nullptr,
             "moe_router_topk: e_score_correction_bias requires num_expert_group > 0");
  }
  if (e_score_correction_bias != nullptr) {
    const Tensor& bias = *e_score_correction_bias;
    VT_CHECK(bias.rank == 1 && bias.shape[0] == e,
             "moe_router_topk: e_score_correction_bias must be [num_experts]");
    VT_CHECK(bias.dtype == DType::kF32,
             "moe_router_topk: e_score_correction_bias must be f32");
    VT_CHECK(bias.IsContiguous(),
             "moe_router_topk: e_score_correction_bias must be contiguous");
    VT_CHECK(bias.device == q.device,
             "moe_router_topk: device mismatch (e_score_correction_bias/queue)");
  }
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
      q, weights, indices, logits, args, e_score_correction_bias);
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

void MoeCombineGate(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                    const Tensor& sd, const Tensor& gl) {
  VT_CHECK(expert_out.rank == 3 && weights.rank == 2 && out.rank == 2,
           "moe_combine_gate: expert_out [T,K,H], weights [T,K], out [T,H]");
  const int64_t t = out.shape[0], h = out.shape[1], k = weights.shape[1];
  VT_CHECK(expert_out.shape[0] == t && expert_out.shape[1] == k && expert_out.shape[2] == h,
           "moe_combine_gate: expert_out must be [T,K,H] matching out/weights");
  VT_CHECK(weights.shape[0] == t, "moe_combine_gate: weights token count must match out");
  VT_CHECK(IsFloat(expert_out.dtype) && IsOutFloat(out.dtype),
           "moe_combine_gate: float expert_out, f32/bf16 out");
  VT_CHECK(weights.dtype == DType::kF32, "moe_combine_gate: weights must be f32");
  VT_CHECK(sd.dtype == DType::kF32 && sd.rank == 2 && sd.shape[0] == t && sd.shape[1] == h,
           "moe_combine_gate: sd must be f32 [T,H]");
  VT_CHECK(gl.dtype == DType::kF32 && gl.Numel() == t,
           "moe_combine_gate: gl must be f32 with T elements");
  VT_CHECK(expert_out.IsContiguous() && weights.IsContiguous() && out.IsContiguous() &&
               sd.IsContiguous() && gl.IsContiguous(),
           "moe_combine_gate: contiguous required");
  VT_CHECK(expert_out.device == q.device && weights.device == q.device &&
               out.device == q.device && sd.device == q.device && gl.device == q.device,
           "moe_combine_gate: device mismatch");
  reinterpret_cast<MoeCombineGateFn>(GetOp(OpId::kMoeCombineGate, q.device.type))(
      q, out, expert_out, weights, sd, gl);
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
  // packed), which holds for the NHD unbind slice. Input K/V may likewise be
  // torch.split-style QKVParallelLinear views: each [H,D] token page is packed,
  // while stride(0) still spans Q+K+V. The kernels already consume the explicit
  // token strides, matching upstream reshape_and_cache_flash.
  VT_CHECK(k.stride[2] == 1 && v.stride[2] == 1 &&
               k.stride[1] == head_size && v.stride[1] == head_size &&
               k.stride[0] >= num_kv_heads * head_size &&
               v.stride[0] >= num_kv_heads * head_size &&
               slot_mapping.IsContiguous(),
           "reshape_and_cache: k/v token pages must be inner-contiguous and "
           "slot_mapping contiguous");
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

void ConcatAndCacheMla(Queue& q, const Tensor& kv_c, const Tensor& k_pe, Tensor& kv_cache,
                       const Tensor& slot_mapping) {
  VT_CHECK(kv_c.rank == 2 && k_pe.rank == 2,
           "concat_and_cache_mla: kv_c/k_pe must be rank-2 "
           "[num_tokens, kv_lora_rank] / [num_tokens, qk_rope_head_dim]");
  VT_CHECK(kv_cache.rank == 3,
           "concat_and_cache_mla: kv_cache must be rank-3 "
           "[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim] — MLA has "
           "NO K/V axis (mla_attention.py:1216-1224)");
  VT_CHECK(slot_mapping.rank == 1,
           "concat_and_cache_mla: slot_mapping must be rank-1 [num_slots]");
  const int64_t kv_lora_rank = kv_c.shape[1];
  const int64_t pe_dim = k_pe.shape[1];
  VT_CHECK(kv_c.shape[0] == k_pe.shape[0],
           "concat_and_cache_mla: kv_c and k_pe must share num_tokens");
  // Upstream's host-side contract, cache_kernels.cu:876
  // (`STD_TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim)`).
  VT_CHECK(kv_cache.shape[2] == kv_lora_rank + pe_dim,
           "concat_and_cache_mla: kv_cache entry width must equal "
           "kv_lora_rank + qk_rope_head_dim");
  VT_CHECK(kv_lora_rank > 0 && pe_dim > 0,
           "concat_and_cache_mla: kv_lora_rank and qk_rope_head_dim must be > 0");
  // Upstream uses slot_mapping.size(0) as the token count (`:855-863`): kv_c/k_pe
  // may carry extra trailing rows (CUDA-graph padding) that are ignored.
  VT_CHECK(kv_c.shape[0] >= slot_mapping.shape[0],
           "concat_and_cache_mla: num_tokens must be >= slot_mapping length");
  // The "auto" cache path ONLY. fp8_ds_mla (the 656-byte V3.2 layout,
  // cache_kernels.cu:866-875) and int4 are out of campaign scope, so a mismatched
  // cache dtype is REFUSED rather than silently mis-written.
  VT_CHECK(IsFloat(kv_c.dtype) && kv_c.dtype == k_pe.dtype && kv_cache.dtype == kv_c.dtype,
           "concat_and_cache_mla: kv_c/k_pe/kv_cache must share one float dtype "
           "(the auto cache path; fp8_ds_mla is out of scope)");
  VT_CHECK(slot_mapping.dtype == DType::kI64,
           "concat_and_cache_mla: slot_mapping must be i64");
  // Indexing is stride-driven (upstream reads kv_cache.stride(0)/stride(1) and
  // kv_c/k_pe.stride(0)), so the cache need not be contiguous — a strided view
  // and a split-projection source row are both fine. We require only what the
  // copy actually needs: unit innermost stride on every operand.
  VT_CHECK(kv_c.stride[1] == 1 && k_pe.stride[1] == 1 && kv_cache.stride[2] == 1 &&
               slot_mapping.IsContiguous(),
           "concat_and_cache_mla: kv_c/k_pe/kv_cache innermost stride must be 1 "
           "and slot_mapping contiguous");
  VT_CHECK(kv_c.stride[0] >= kv_lora_rank && k_pe.stride[0] >= pe_dim,
           "concat_and_cache_mla: kv_c/k_pe token rows must not overlap");
  VT_CHECK(kv_c.device == q.device && k_pe.device == q.device &&
               kv_cache.device == q.device && slot_mapping.device == q.device,
           "concat_and_cache_mla: device mismatch "
           "(kv_c/k_pe/kv_cache/slot_mapping/queue)");
  reinterpret_cast<ConcatAndCacheMlaFn>(GetOp(OpId::kConcatAndCacheMla, q.device.type))(
      q, kv_c, k_pe, kv_cache, slot_mapping);
}

void MlaDecodeAttention(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                        const Tensor& kv_cache, const Tensor& block_table,
                        const Tensor& seq_lens, const MlaDecodeAttentionArgs& args) {
  VT_CHECK(query.rank == 3 && out.rank == 3,
           "mla_decode_attention: query/out must be rank-3 "
           "[batch, num_q_heads, head_size] / [batch, num_q_heads, v_head_dim]");
  // The MLA cache has NO K/V axis and NO head axis (mla_attention.py:1216-1224).
  VT_CHECK(kv_cache.rank == 3,
           "mla_decode_attention: kv_cache must be rank-3 "
           "[num_blocks, block_size, head_size] — MLA has no K/V and no head axis");
  VT_CHECK(block_table.rank == 2, "mla_decode_attention: block_table rank-2 [batch, max_blocks]");
  VT_CHECK(seq_lens.rank == 1, "mla_decode_attention: seq_lens rank-1 [batch]");
  const int64_t batch = query.shape[0];
  const int64_t num_q_heads = query.shape[1];
  const int64_t head_size = query.shape[2];
  const int64_t v_head_dim = out.shape[2];
  VT_CHECK(batch > 0 && num_q_heads > 0 && head_size > 0 && v_head_dim > 0,
           "mla_decode_attention: batch/num_q_heads/head_size/v_head_dim must be > 0");
  VT_CHECK(out.shape[0] == batch && out.shape[1] == num_q_heads,
           "mla_decode_attention: out must be [batch, num_q_heads, v_head_dim]");
  VT_CHECK(kv_cache.shape[2] == head_size,
           "mla_decode_attention: kv_cache entry width must equal query head_size "
           "(kv_lora_rank + qk_rope_head_dim)");
  // V is the LEADING v_head_dim slice of the same latent row (triton_mla.py:236,
  // `kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]`).
  VT_CHECK(v_head_dim <= head_size,
           "mla_decode_attention: v_head_dim must be <= head_size (V is the leading "
           "kv_lora_rank slice of the SAME cache row)");
  VT_CHECK(kv_cache.shape[1] > 0, "mla_decode_attention: block_size must be > 0");
  VT_CHECK(block_table.shape[0] == batch && seq_lens.shape[0] == batch,
           "mla_decode_attention: block_table/seq_lens must have `batch` rows");
  VT_CHECK(block_table.dtype == DType::kI32 && seq_lens.dtype == DType::kI32,
           "mla_decode_attention: block_table/seq_lens must be i32");
  VT_CHECK(IsFloat(query.dtype) && kv_cache.dtype == query.dtype && out.dtype == query.dtype,
           "mla_decode_attention: query/kv_cache/out must share one float dtype "
           "(the auto cache path; the fp8 KV-cache branch is out of scope)");
  VT_CHECK(args.scale > 0.0f, "mla_decode_attention: args.scale must be > 0");
  VT_CHECK(args.num_kv_splits >= 0, "mla_decode_attention: args.num_kv_splits must be >= 0");
  // Indexing is stride-driven on the leading dims (a cross-layer cache view has
  // gaps — cf. upstream `_page_stride`, triton_decode_attention.py:59-65), so we
  // require only unit innermost strides.
  VT_CHECK(query.stride[2] == 1 && out.stride[2] == 1 && kv_cache.stride[2] == 1,
           "mla_decode_attention: query/out/kv_cache innermost stride must be 1");
  VT_CHECK(block_table.stride[1] == 1 && seq_lens.IsContiguous(),
           "mla_decode_attention: block_table rows must be contiguous and seq_lens contiguous");
  if (lse != nullptr) {
    VT_CHECK(lse->rank == 2 && lse->shape[0] == batch && lse->shape[1] == num_q_heads,
             "mla_decode_attention: lse must be rank-2 [batch, num_q_heads]");
    VT_CHECK(lse->dtype == DType::kF32,
             "mla_decode_attention: lse must be f32 (we keep the LSE in the "
             "accumulation dtype; upstream stores it in q.dtype)");
    VT_CHECK(lse->stride[1] == 1, "mla_decode_attention: lse innermost stride must be 1");
    VT_CHECK(lse->device == q.device, "mla_decode_attention: lse device mismatch");
  }
  VT_CHECK(out.device == q.device && query.device == q.device && kv_cache.device == q.device &&
               block_table.device == q.device && seq_lens.device == q.device,
           "mla_decode_attention: device mismatch "
           "(out/query/kv_cache/block_table/seq_lens/queue)");
  reinterpret_cast<MlaDecodeAttentionFn>(GetOp(OpId::kMlaDecodeAttention, q.device.type))(
      q, out, lse, query, kv_cache, block_table, seq_lens, args);
}

void MlaPrefillAttention(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                         const Tensor& key, const Tensor& value, const Tensor& cu_seqlens_q,
                         const Tensor& cu_seqlens_k, const MlaPrefillAttentionArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3,
           "mla_prefill_attention: query/key/value/out must be rank-3 "
           "[total_q|total_k, num_heads, head_dim]");
  VT_CHECK(cu_seqlens_q.rank == 1 && cu_seqlens_k.rank == 1,
           "mla_prefill_attention: cu_seqlens_q/cu_seqlens_k must be rank-1 [num_reqs+1]");
  const int64_t total_q = query.shape[0];
  const int64_t total_k = key.shape[0];
  const int64_t num_heads = query.shape[1];
  const int64_t qk_head_dim = query.shape[2];
  const int64_t v_head_dim = value.shape[2];
  VT_CHECK(num_heads > 0 && qk_head_dim > 0 && v_head_dim > 0,
           "mla_prefill_attention: num_heads/qk_head_dim/v_head_dim must be > 0");
  VT_CHECK(key.shape[1] == num_heads && value.shape[1] == num_heads,
           "mla_prefill_attention: key/value must carry the SAME head count as query "
           "(MLA prefill is multi-head on both sides — the latent is up-projected; "
           "mla_attention.py:315-318)");
  VT_CHECK(key.shape[2] == qk_head_dim,
           "mla_prefill_attention: key head_dim must equal query head_dim "
           "(qk_nope_head_dim + qk_rope_head_dim)");
  VT_CHECK(value.shape[0] == total_k,
           "mla_prefill_attention: key and value must share total_k");
  VT_CHECK(out.shape[0] == total_q && out.shape[1] == num_heads && out.shape[2] == v_head_dim,
           "mla_prefill_attention: out must be [total_q, num_heads, v_head_dim]");
  // Upstream pads V UP to the QK width (flash_attn.py:164-168), so V may never
  // be WIDER than QK. For DeepSeek v_head_dim 128 <= qk_head_dim 192.
  VT_CHECK(v_head_dim <= qk_head_dim,
           "mla_prefill_attention: v_head_dim must be <= qk_head_dim (upstream ZERO-PADS "
           "V up to the QK width, flash_attn.py:164-168)");
  VT_CHECK(cu_seqlens_q.shape[0] >= 2 && cu_seqlens_q.shape[0] == cu_seqlens_k.shape[0],
           "mla_prefill_attention: cu_seqlens_q/cu_seqlens_k must both be [num_reqs+1]");
  VT_CHECK(cu_seqlens_q.dtype == DType::kI32 && cu_seqlens_k.dtype == DType::kI32,
           "mla_prefill_attention: cu_seqlens_q/cu_seqlens_k must be i32");
  VT_CHECK(cu_seqlens_q.IsContiguous() && cu_seqlens_k.IsContiguous(),
           "mla_prefill_attention: cu_seqlens_q/cu_seqlens_k must be contiguous");
  VT_CHECK(IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype &&
               out.dtype == query.dtype,
           "mla_prefill_attention: query/key/value/out must share one float dtype "
           "(the fp8-prefill branch needs device-capability family 100 and is "
           "UNREACHABLE on sm_121 — mla_attention.py:1382-1385)");
  VT_CHECK(args.scale > 0.0f, "mla_prefill_attention: args.scale must be > 0");
  VT_CHECK(args.max_seqlen_q >= 0 && args.max_seqlen_k >= 0,
           "mla_prefill_attention: args.max_seqlen_q/max_seqlen_k must be >= 0");
  // Stride-driven on the token/head axes (a workspace slice is a strided view);
  // the innermost head_dim must be packed.
  VT_CHECK(query.stride[2] == 1 && key.stride[2] == 1 && value.stride[2] == 1 &&
               out.stride[2] == 1,
           "mla_prefill_attention: query/key/value/out innermost stride must be 1");
  if (lse != nullptr) {
    // UNPADDED varlen LSE, upstream's [num_heads, total_q] layout.
    VT_CHECK(lse->rank == 2 && lse->shape[0] == num_heads && lse->shape[1] == total_q,
             "mla_prefill_attention: lse must be rank-2 [num_heads, total_q] (upstream's "
             "unpadded varlen LSE layout)");
    VT_CHECK(lse->dtype == DType::kF32, "mla_prefill_attention: lse must be f32");
    VT_CHECK(lse->stride[1] == 1, "mla_prefill_attention: lse innermost stride must be 1");
    VT_CHECK(lse->device == q.device, "mla_prefill_attention: lse device mismatch");
  }
  VT_CHECK(out.device == q.device && query.device == q.device && key.device == q.device &&
               value.device == q.device && cu_seqlens_q.device == q.device &&
               cu_seqlens_k.device == q.device,
           "mla_prefill_attention: device mismatch "
           "(out/query/key/value/cu_seqlens_q/cu_seqlens_k/queue)");
  reinterpret_cast<MlaPrefillAttentionFn>(GetOp(OpId::kMlaPrefillAttention, q.device.type))(
      q, out, lse, query, key, value, cu_seqlens_q, cu_seqlens_k, args);
}

void GatherMlaCache(Queue& q, Tensor& dst, const Tensor& src_cache, const Tensor& block_table,
                    const Tensor& cu_seq_lens, const Tensor& token_to_seq,
                    const Tensor* seq_starts, int64_t num_tokens) {
  VT_CHECK(dst.rank == 2, "gather_mla_cache: dst must be rank-2 [tot_tokens, head_dim]");
  VT_CHECK(src_cache.rank == 3,
           "gather_mla_cache: src_cache must be rank-3 [num_blocks, block_size, head_dim] "
           "— the 3-D MLA cache (mla_attention.py:1216-1224)");
  VT_CHECK(block_table.rank == 2, "gather_mla_cache: block_table rank-2 [batch, max_blocks]");
  VT_CHECK(cu_seq_lens.rank == 1, "gather_mla_cache: cu_seq_lens rank-1 [batch+1]");
  VT_CHECK(token_to_seq.rank == 1, "gather_mla_cache: token_to_seq rank-1 [max_tokens]");
  const int64_t batch = block_table.shape[0];
  const int64_t head_dim = src_cache.shape[2];
  VT_CHECK(batch > 0 && head_dim > 0, "gather_mla_cache: batch/head_dim must be > 0");
  VT_CHECK(dst.shape[1] == head_dim,
           "gather_mla_cache: dst entry width must equal the cache entry width");
  VT_CHECK(cu_seq_lens.shape[0] == batch + 1,
           "gather_mla_cache: cu_seq_lens must be [batch+1]");
  VT_CHECK(num_tokens >= 0, "gather_mla_cache: num_tokens must be >= 0");
  VT_CHECK(dst.shape[0] >= num_tokens, "gather_mla_cache: dst must hold num_tokens rows");
  VT_CHECK(token_to_seq.shape[0] >= num_tokens,
           "gather_mla_cache: token_to_seq must hold num_tokens entries");
  VT_CHECK(src_cache.shape[1] > 0, "gather_mla_cache: block_size must be > 0");
  VT_CHECK(block_table.dtype == DType::kI32 && cu_seq_lens.dtype == DType::kI32 &&
               token_to_seq.dtype == DType::kI32,
           "gather_mla_cache: block_table/cu_seq_lens/token_to_seq must be i32");
  // The auto path only — the fp8 dequant branch (cache_kernels.cu:1039-1047) is
  // out of campaign scope, exactly as the W3 cache write refuses fp8_ds_mla.
  VT_CHECK(IsFloat(dst.dtype) && src_cache.dtype == dst.dtype,
           "gather_mla_cache: dst/src_cache must share one float dtype (the auto path; "
           "the fp8 dequant branch is out of scope)");
  VT_CHECK(dst.stride[1] == 1 && src_cache.stride[2] == 1,
           "gather_mla_cache: dst/src_cache innermost stride must be 1");
  VT_CHECK(block_table.stride[1] == 1 && cu_seq_lens.IsContiguous() &&
               token_to_seq.IsContiguous(),
           "gather_mla_cache: block_table rows / cu_seq_lens / token_to_seq must be contiguous");
  VT_CHECK(dst.device == q.device && src_cache.device == q.device &&
               block_table.device == q.device && cu_seq_lens.device == q.device &&
               token_to_seq.device == q.device,
           "gather_mla_cache: device mismatch");
  if (seq_starts != nullptr) {
    VT_CHECK(seq_starts->rank == 1 && seq_starts->shape[0] == batch,
             "gather_mla_cache: seq_starts must be rank-1 [batch]");
    VT_CHECK(seq_starts->dtype == DType::kI32, "gather_mla_cache: seq_starts must be i32");
    VT_CHECK(seq_starts->IsContiguous(), "gather_mla_cache: seq_starts must be contiguous");
    VT_CHECK(seq_starts->device == q.device, "gather_mla_cache: seq_starts device mismatch");
  }
  reinterpret_cast<GatherMlaCacheFn>(GetOp(OpId::kGatherMlaCache, q.device.type))(
      q, dst, src_cache, block_table, cu_seq_lens, token_to_seq, seq_starts, num_tokens);
}

void MergeAttnStates(Queue& q, Tensor& output, Tensor* output_lse, const Tensor& prefix_output,
                     const Tensor& prefix_lse, const Tensor& suffix_output,
                     const Tensor& suffix_lse, int64_t prefill_tokens_with_context) {
  VT_CHECK(output.rank == 3 && prefix_output.rank == 3 && suffix_output.rank == 3,
           "merge_attn_states: output/prefix_output/suffix_output must be rank-3 "
           "[num_tokens, num_heads, head_size]");
  VT_CHECK(prefix_lse.rank == 2 && suffix_lse.rank == 2,
           "merge_attn_states: prefix_lse/suffix_lse must be rank-2 [num_heads, num_tokens]");
  const int64_t num_tokens = output.shape[0];
  const int64_t num_heads = output.shape[1];
  const int64_t head_size = output.shape[2];
  VT_CHECK(num_heads > 0 && head_size > 0,
           "merge_attn_states: num_heads/head_size must be > 0");
  VT_CHECK(prefix_output.shape[0] == num_tokens && prefix_output.shape[1] == num_heads &&
               prefix_output.shape[2] == head_size,
           "merge_attn_states: prefix_output shape must match output");
  VT_CHECK(suffix_output.shape[0] == num_tokens && suffix_output.shape[1] == num_heads &&
               suffix_output.shape[2] == head_size,
           "merge_attn_states: suffix_output shape must match output");
  VT_CHECK(prefix_lse.shape[0] == num_heads && prefix_lse.shape[1] == num_tokens &&
               suffix_lse.shape[0] == num_heads && suffix_lse.shape[1] == num_tokens,
           "merge_attn_states: prefix_lse/suffix_lse must be [num_heads, num_tokens]");
  VT_CHECK(prefix_lse.dtype == DType::kF32 && suffix_lse.dtype == DType::kF32,
           "merge_attn_states: prefix_lse/suffix_lse must be f32");
  VT_CHECK(IsFloat(output.dtype) && prefix_output.dtype == output.dtype &&
               suffix_output.dtype == output.dtype,
           "merge_attn_states: output/prefix_output/suffix_output must share one float dtype "
           "(the USE_FP8_OUTPUT branch is out of scope)");
  VT_CHECK(output.stride[2] == 1 && prefix_output.stride[2] == 1 &&
               suffix_output.stride[2] == 1 && prefix_lse.stride[1] == 1 &&
               suffix_lse.stride[1] == 1,
           "merge_attn_states: innermost strides must be 1");
  VT_CHECK(output.device == q.device && prefix_output.device == q.device &&
               prefix_lse.device == q.device && suffix_output.device == q.device &&
               suffix_lse.device == q.device,
           "merge_attn_states: device mismatch");
  if (output_lse != nullptr) {
    VT_CHECK(output_lse->rank == 2 && output_lse->shape[0] == num_heads &&
                 output_lse->shape[1] == num_tokens,
             "merge_attn_states: output_lse must be [num_heads, num_tokens]");
    VT_CHECK(output_lse->dtype == DType::kF32, "merge_attn_states: output_lse must be f32");
    VT_CHECK(output_lse->stride[1] == 1,
             "merge_attn_states: output_lse innermost stride must be 1");
    VT_CHECK(output_lse->device == q.device, "merge_attn_states: output_lse device mismatch");
  }
  reinterpret_cast<MergeAttnStatesFn>(GetOp(OpId::kMergeAttnStates, q.device.type))(
      q, output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse,
      prefill_tokens_with_context);
}

void PagedAttention(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                    const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                    const Tensor& query_start_loc, const PagedAttentionArgs& args) {
  VT_CHECK(query.rank == 3 && out.rank == 3,
           "paged_attention: query/out rank-3 [num_actual_tokens,num_q_heads,head_size]");
  VT_CHECK(k_cache.rank == 4 && v_cache.rank == 4,
           "paged_attention: k_cache/v_cache rank-4 "
           "[num_blocks,block_size,num_kv_heads,head_size]");
  const int64_t num_tokens = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t num_kv_heads = k_cache.shape[2], head_size = k_cache.shape[3];
  VT_CHECK(out.shape[0] == num_tokens && out.shape[1] == hq && out.shape[2] == d,
           "paged_attention: out must match query shape");
  VT_CHECK(d == head_size, "paged_attention: query head_size must match the cache head_size");
  VT_CHECK(v_cache.shape[0] == k_cache.shape[0] && v_cache.shape[1] == k_cache.shape[1] &&
               v_cache.shape[2] == num_kv_heads && v_cache.shape[3] == head_size,
           "paged_attention: k_cache and v_cache must share shape");
  VT_CHECK(hq >= 1 && num_kv_heads >= 1 && hq % num_kv_heads == 0,
           "paged_attention: num_q_heads must be a positive multiple of num_kv_heads (GQA)");
  VT_CHECK(args.scale > 0.0f, "paged_attention: scale must be set (> 0), e.g. head_size^-0.5");
  if (args.window_size.has_value()) {
    VT_CHECK(args.window_size->left >= 0 && args.window_size->right >= 0,
             "paged_attention: window_size left/right must both be >= 0");
  }
  VT_CHECK(IsFloat(query.dtype) && IsOutFloat(out.dtype),
           "paged_attention: float query, f32/bf16 out");
  // The KV cache may be a DIFFERENT float dtype than the query (Phase-1 bf16 KV
  // cache: f32 query · bf16 cache — the kernel converts bf16 cache reads to f32
  // and accumulates in f32). Require only that K and V share one float dtype.
  VT_CHECK(IsFloat(k_cache.dtype) && k_cache.dtype == v_cache.dtype,
           "paged_attention: k_cache/v_cache must share one float dtype");
  // metadata: block_table [num_reqs, max_blocks] i32, seq_lens [num_reqs] i32,
  // query_start_loc [num_reqs+1] i32.
  VT_CHECK(seq_lens.rank == 1 && seq_lens.dtype == DType::kI32,
           "paged_attention: seq_lens must be i32 [num_reqs]");
  const int64_t num_reqs = seq_lens.shape[0];
  VT_CHECK(num_reqs >= 1, "paged_attention: num_reqs must be >= 1");
  VT_CHECK(block_table.rank == 2 && block_table.shape[0] == num_reqs &&
               block_table.dtype == DType::kI32,
           "paged_attention: block_table must be i32 [num_reqs, max_blocks]");
  VT_CHECK(query_start_loc.rank == 1 && query_start_loc.shape[0] == num_reqs + 1 &&
               query_start_loc.dtype == DType::kI32,
           "paged_attention: query_start_loc must be i32 [num_reqs+1]");
  // query/out contiguous; seq_lens/query_start_loc contiguous. The cache and
  // block_table are read via strides (the cache is the strided NHD unbind slice),
  // but the per-token page must be head-contiguous (elem stride 1, head stride
  // head_size) — same guarantee reshape_and_cache relies on.
  VT_CHECK(query.IsContiguous() && out.IsContiguous() && seq_lens.IsContiguous() &&
               query_start_loc.IsContiguous(),
           "paged_attention: query/out/seq_lens/query_start_loc must be contiguous");
  VT_CHECK(k_cache.stride[3] == 1 && v_cache.stride[3] == 1,
           "paged_attention: k_cache/v_cache innermost (head_size) stride must be 1");
  VT_CHECK(k_cache.stride[2] == head_size && v_cache.stride[2] == head_size,
           "paged_attention: k_cache/v_cache page must be head-contiguous "
           "(stride[2] == head_size) — the NHD unbind-slice layout");
  VT_CHECK(query.device == q.device && out.device == q.device && k_cache.device == q.device &&
               v_cache.device == q.device && block_table.device == q.device &&
               seq_lens.device == q.device && query_start_loc.device == q.device,
           "paged_attention: device mismatch (query/out/cache/block_table/seq_lens/"
           "query_start_loc/queue)");
  reinterpret_cast<PagedAttentionFn>(GetOp(OpId::kPagedAttention, q.device.type))(
      q, out, query, k_cache, v_cache, block_table, seq_lens, query_start_loc, args);
}

namespace {
// Shared checks for the sampling ops: logits [num_reqs, vocab] f32, contiguous,
// on the queue device. Returns num_reqs for downstream per-row-metadata checks.
int64_t CheckSamplingLogits(const Queue& q, const Tensor& logits, const char* name) {
  VT_CHECK(logits.rank == 2, std::string(name) + ": logits must be rank-2 [num_reqs, vocab]");
  VT_CHECK(logits.dtype == DType::kF32, std::string(name) + ": logits must be f32");
  VT_CHECK(logits.IsContiguous(), std::string(name) + ": logits must be contiguous");
  VT_CHECK(logits.device == q.device, std::string(name) + ": logits device must match queue");
  return logits.shape[0];
}
}  // namespace

void ApplyTemperature(Queue& q, Tensor& logits, const Tensor& temp, bool all_random) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_temperature");
  VT_CHECK(temp.rank == 1 && temp.shape[0] == n && temp.dtype == DType::kF32 &&
               temp.IsContiguous() && temp.device == q.device,
           "apply_temperature: temp must be f32 [num_reqs] contiguous on the queue device");
  reinterpret_cast<ApplyTemperatureFn>(GetOp(OpId::kApplyTemperature, q.device.type))(
      q, logits, temp, all_random);
}

void GreedyArgmax(Queue& q, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = CheckSamplingLogits(q, logits, "greedy_argmax");
  VT_CHECK(token_ids.rank == 1 && token_ids.shape[0] == n && token_ids.dtype == DType::kI64 &&
               token_ids.IsContiguous() && token_ids.device == q.device,
           "greedy_argmax: token_ids must be i64 [num_reqs] contiguous on the queue device");
  reinterpret_cast<GreedyArgmaxFn>(GetOp(OpId::kGreedyArgmax, q.device.type))(q, token_ids,
                                                                              logits);
}

void ApplyTopKTopP(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_top_k_top_p");
  // Both None => no-op (upstream apply_top_k_top_p returns logits unchanged).
  if (k == nullptr && p == nullptr) return;
  if (k != nullptr) {
    VT_CHECK(k->rank == 1 && k->shape[0] == n && k->dtype == DType::kI32 && k->IsContiguous() &&
                 k->device == q.device,
             "apply_top_k_top_p: k must be i32 [num_reqs] contiguous on the queue device");
  }
  if (p != nullptr) {
    VT_CHECK(p->rank == 1 && p->shape[0] == n && p->dtype == DType::kF32 && p->IsContiguous() &&
                 p->device == q.device,
             "apply_top_k_top_p: p must be f32 [num_reqs] contiguous on the queue device");
  }
  reinterpret_cast<ApplyTopKTopPFn>(GetOp(OpId::kApplyTopKTopP, q.device.type))(q, logits, k, p);
}

void ComputeProbs(Queue& q, Tensor& probs, const Tensor& logits) {
  const int64_t n = CheckSamplingLogits(q, logits, "compute_probs");
  VT_CHECK(probs.rank == 2 && probs.shape[0] == n && probs.shape[1] == logits.shape[1] &&
               probs.dtype == DType::kF32 && probs.IsContiguous() && probs.device == q.device,
           "compute_probs: probs must be f32 [num_reqs, vocab] contiguous matching logits");
  reinterpret_cast<ComputeProbsFn>(GetOp(OpId::kComputeProbs, q.device.type))(q, probs, logits);
}

void ComputeLogprobs(Queue& q, Tensor& logprobs, const Tensor& logits) {
  const int64_t n = CheckSamplingLogits(q, logits, "compute_logprobs");
  VT_CHECK(logprobs.rank == 2 && logprobs.shape[0] == n && logprobs.shape[1] == logits.shape[1] &&
               logprobs.dtype == DType::kF32 && logprobs.IsContiguous() &&
               logprobs.device == q.device,
           "compute_logprobs: logprobs must be f32 [num_reqs, vocab] contiguous matching logits");
  reinterpret_cast<ComputeLogprobsFn>(GetOp(OpId::kComputeLogprobs, q.device.type))(q, logprobs,
                                                                                    logits);
}

void RandomSample(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds) {
  VT_CHECK(probs.rank == 2, "random_sample: probs must be rank-2 [num_reqs, vocab]");
  VT_CHECK(probs.dtype == DType::kF32, "random_sample: probs must be f32");
  VT_CHECK(probs.IsContiguous(), "random_sample: probs must be contiguous");
  VT_CHECK(probs.device == q.device, "random_sample: probs device must match queue");
  const int64_t n = probs.shape[0];
  VT_CHECK(token_ids.rank == 1 && token_ids.shape[0] == n && token_ids.dtype == DType::kI64 &&
               token_ids.IsContiguous() && token_ids.device == q.device,
           "random_sample: token_ids must be i64 [num_reqs] contiguous on the queue device");
  VT_CHECK(seeds.rank == 1 && seeds.shape[0] == n && seeds.dtype == DType::kI64 &&
               seeds.IsContiguous() && seeds.device == q.device,
           "random_sample: seeds must be i64 [num_reqs] contiguous on the queue device");
  reinterpret_cast<RandomSampleFn>(GetOp(OpId::kRandomSample, q.device.type))(q, token_ids, probs,
                                                                              seeds);
}

namespace {
// [num_reqs, vocab] tensor of a required dtype, contiguous, on the queue device.
void CheckSamplingMatrix(const Queue& q, const Tensor& t, int64_t n, int64_t v, DType dt,
                         const char* name, const char* what) {
  VT_CHECK(t.rank == 2 && t.shape[0] == n && t.shape[1] == v && t.dtype == dt &&
               t.IsContiguous() && t.device == q.device,
           std::string(name) + ": " + what + " must be [num_reqs, vocab] of the expected dtype, "
                                             "contiguous on the queue device");
}
// [num_reqs] f32 vector.
void CheckSamplingVec(const Queue& q, const Tensor& t, int64_t n, const char* name,
                      const char* what) {
  VT_CHECK(t.rank == 1 && t.shape[0] == n && t.dtype == DType::kF32 && t.IsContiguous() &&
               t.device == q.device,
           std::string(name) + ": " + what + " must be f32 [num_reqs] contiguous on the device");
}
}  // namespace

void ApplyPenalties(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                    const Tensor& output_bin_counts, const Tensor& output_mask,
                    const Tensor& frequency_penalties, const Tensor& presence_penalties,
                    const Tensor& repetition_penalties) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_penalties");
  const int64_t v = logits.shape[1];
  CheckSamplingMatrix(q, prompt_mask, n, v, DType::kI8, "apply_penalties", "prompt_mask");
  CheckSamplingMatrix(q, output_mask, n, v, DType::kI8, "apply_penalties", "output_mask");
  CheckSamplingMatrix(q, output_bin_counts, n, v, DType::kI32, "apply_penalties",
                      "output_bin_counts");
  CheckSamplingVec(q, frequency_penalties, n, "apply_penalties", "frequency_penalties");
  CheckSamplingVec(q, presence_penalties, n, "apply_penalties", "presence_penalties");
  CheckSamplingVec(q, repetition_penalties, n, "apply_penalties", "repetition_penalties");
  reinterpret_cast<ApplyPenaltiesFn>(GetOp(OpId::kApplyPenalties, q.device.type))(
      q, logits, prompt_mask, output_bin_counts, output_mask, frequency_penalties,
      presence_penalties, repetition_penalties);
}

void ApplyMinP(Queue& q, Tensor& logits, const Tensor& min_p) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_min_p");
  CheckSamplingVec(q, min_p, n, "apply_min_p", "min_p");
  reinterpret_cast<ApplyMinPFn>(GetOp(OpId::kApplyMinP, q.device.type))(q, logits, min_p);
}

namespace {
// The (rows, cols) pair-list shape shared by ApplyLogitBias / ApplyTokenMask.
void CheckPairList(const Queue& q, const Tensor& rows, const Tensor& cols, const char* name) {
  VT_CHECK(rows.rank == 1 && cols.rank == 1 && rows.shape[0] == cols.shape[0],
           std::string(name) + ": rows and cols must be equal-length rank-1 [m]");
  VT_CHECK(rows.dtype == DType::kI32 && cols.dtype == DType::kI32,
           std::string(name) + ": rows/cols must be i32");
  VT_CHECK(rows.IsContiguous() && cols.IsContiguous() && rows.device == q.device &&
               cols.device == q.device,
           std::string(name) + ": rows/cols must be contiguous on the queue device");
}
}  // namespace

void ApplyLogitBias(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols,
                    const Tensor& biases) {
  CheckSamplingLogits(q, logits, "apply_logit_bias");
  CheckPairList(q, rows, cols, "apply_logit_bias");
  VT_CHECK(biases.rank == 1 && biases.shape[0] == rows.shape[0] && biases.dtype == DType::kF32 &&
               biases.IsContiguous() && biases.device == q.device,
           "apply_logit_bias: biases must be f32 [m] contiguous on the queue device");
  reinterpret_cast<ApplyLogitBiasFn>(GetOp(OpId::kApplyLogitBias, q.device.type))(q, logits, rows,
                                                                                  cols, biases);
}

void ApplyTokenMask(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols) {
  CheckSamplingLogits(q, logits, "apply_token_mask");
  CheckPairList(q, rows, cols, "apply_token_mask");
  reinterpret_cast<ApplyTokenMaskFn>(GetOp(OpId::kApplyTokenMask, q.device.type))(q, logits, rows,
                                                                                  cols);
}

void ApplyAllowedTokenIds(Queue& q, Tensor& logits, const Tensor& mask) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_allowed_token_ids");
  CheckSamplingMatrix(q, mask, n, logits.shape[1], DType::kI8, "apply_allowed_token_ids", "mask");
  reinterpret_cast<ApplyAllowedTokenIdsFn>(GetOp(OpId::kApplyAllowedTokenIds, q.device.type))(
      q, logits, mask);
}

// --- Qwen3.6 elementwise "glue" ops (M0.9 forward). ------------------------

void CastBf16(Queue& q, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kBF16, "cast_bf16: out must be bf16");
  VT_CHECK(in.dtype == DType::kF32, "cast_bf16: in must be f32");
  VT_CHECK(out.Numel() == in.Numel(), "cast_bf16: out/in must have the same element count");
  // Input may be a torch.split-style packed view (the merged QKV path): each
  // logical row is dense while the row stride spans the parent Q+K+V tensor.
  // Symmetric with CastF32; strided inputs only arise on CUDA (the merge is
  // CUDA-only), where the kernel honors the row stride.
  int64_t inner = 1;
  bool inner_contiguous = true;
  for (int dim = in.rank - 1; dim >= 1; --dim) {
    inner_contiguous = inner_contiguous && in.stride[dim] == inner;
    inner *= in.shape[dim];
  }
  inner_contiguous = inner_contiguous && in.rank >= 1 && in.stride[0] >= inner;
  VT_CHECK(out.IsContiguous() && inner_contiguous,
           "cast_bf16: out must be contiguous and input rows inner-contiguous");
  VT_CHECK(out.device == q.device && in.device == q.device,
           "cast_bf16: device mismatch (out/in/queue)");
  reinterpret_cast<CastBf16Fn>(GetOp(OpId::kCastBf16, q.device.type))(q, out, in);
}

void CastF32(Queue& q, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kF32, "cast_f32: out must be f32");
  VT_CHECK(in.dtype == DType::kBF16, "cast_f32: in must be bf16");
  VT_CHECK(out.Numel() == in.Numel(), "cast_f32: out/in must have the same element count");
  int64_t inner = 1;
  bool inner_contiguous = true;
  for (int dim = in.rank - 1; dim >= 1; --dim) {
    inner_contiguous = inner_contiguous && in.stride[dim] == inner;
    inner *= in.shape[dim];
  }
  inner_contiguous = inner_contiguous && in.rank >= 1 &&
                     in.stride[0] >= inner;
  VT_CHECK(out.IsContiguous() && inner_contiguous,
           "cast_f32: out must be contiguous and input rows inner-contiguous");
  VT_CHECK(out.device == q.device && in.device == q.device,
           "cast_f32: device mismatch (out/in/queue)");
  reinterpret_cast<CastF32Fn>(GetOp(OpId::kCastF32, q.device.type))(q, out, in);
}

void MulColVecF32(Queue& q, Tensor& x, const Tensor& col) {
  VT_CHECK(x.dtype == DType::kF32, "mul_col_vec_f32: x must be f32");
  VT_CHECK(col.dtype == DType::kF32, "mul_col_vec_f32: col must be f32");
  VT_CHECK(x.rank == 2, "mul_col_vec_f32: x must be rank-2 [M,N]");
  VT_CHECK(col.rank == 1, "mul_col_vec_f32: col must be rank-1 [N]");
  VT_CHECK(col.shape[0] == x.shape[1],
           "mul_col_vec_f32: col length must equal x columns");
  VT_CHECK(x.stride[1] == 1 && x.stride[0] >= x.shape[1],
           "mul_col_vec_f32: x rows must be inner-contiguous");
  VT_CHECK(col.IsContiguous(), "mul_col_vec_f32: col must be contiguous");
  VT_CHECK(x.device == q.device && col.device == q.device,
           "mul_col_vec_f32: device mismatch (x/col/queue)");
  reinterpret_cast<MulColVecF32Fn>(GetOp(OpId::kMulColVecF32, q.device.type))(q, x, col);
}

void AttnGateSplit(Queue& q, Tensor& q_out, Tensor& gate_out, const Tensor& qgate) {
  VT_CHECK(q_out.rank == 3 && gate_out.rank == 3, "attn_gate_split: q_out/gate_out rank-3 [T,Hq,Dh]");
  VT_CHECK(qgate.rank == 2, "attn_gate_split: qgate rank-2 [T, Hq*2*Dh]");
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  VT_CHECK(gate_out.shape[0] == t && gate_out.shape[1] == hq && gate_out.shape[2] == dh,
           "attn_gate_split: gate_out must match q_out [T,Hq,Dh]");
  VT_CHECK(qgate.shape[0] == t && qgate.shape[1] == hq * 2 * dh,
           "attn_gate_split: qgate must be [T, Hq*2*Dh]");
  VT_CHECK(q_out.dtype == DType::kF32 && gate_out.dtype == DType::kF32,
           "attn_gate_split: q_out/gate_out must be f32");
  VT_CHECK(qgate.dtype == DType::kF32 || qgate.dtype == DType::kBF16,
           "attn_gate_split: qgate must be f32 or bf16 (bf16 = VT_BF16_GEMM_OUT q_proj)");
  VT_CHECK(q_out.IsContiguous() && gate_out.IsContiguous() && qgate.IsContiguous(),
           "attn_gate_split: contiguous required");
  VT_CHECK(q_out.device == q.device && gate_out.device == q.device && qgate.device == q.device,
           "attn_gate_split: device mismatch (q_out/gate_out/qgate/queue)");
  reinterpret_cast<AttnGateSplitFn>(GetOp(OpId::kAttnGateSplit, q.device.type))(q, q_out, gate_out,
                                                                                qgate);
}

void SigmoidGateBf16(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate) {
  VT_CHECK(out.dtype == DType::kBF16, "sigmoid_gate_bf16: out must be bf16");
  // attn may be bf16 (the FA-2 prefill path outputs bf16 attention); it is
  // upcast to f32 inside the kernel (exact), so bf16-attn is bit-identical to
  // f32-attn holding the same bf16-representable values. The gate stays f32
  // (sigmoid input must not be rounded).
  VT_CHECK((attn.dtype == DType::kF32 || attn.dtype == DType::kBF16) &&
               gate.dtype == DType::kF32,
           "sigmoid_gate_bf16: attn must be f32/bf16, gate f32");
  VT_CHECK(out.Numel() == attn.Numel() && out.Numel() == gate.Numel(),
           "sigmoid_gate_bf16: out/attn/gate must have the same element count");
  VT_CHECK(out.IsContiguous() && attn.IsContiguous() && gate.IsContiguous(),
           "sigmoid_gate_bf16: contiguous required");
  VT_CHECK(out.device == q.device && attn.device == q.device && gate.device == q.device,
           "sigmoid_gate_bf16: device mismatch (out/attn/gate/queue)");
  reinterpret_cast<SigmoidGateBf16Fn>(GetOp(OpId::kSigmoidGateBf16, q.device.type))(q, out, attn,
                                                                                    gate);
}

void GdnGBeta(Queue& q, Tensor& g_out, Tensor& beta_out, const Tensor& araw, const Tensor& braw,
              const Tensor& a_log, const Tensor& dt_bias) {
  VT_CHECK(g_out.rank == 2 && beta_out.rank == 2 && araw.rank == 2 && braw.rank == 2,
           "gdn_g_beta: g_out/beta_out/araw/braw rank-2 [T,Hv]");
  const int64_t t = g_out.shape[0], hv = g_out.shape[1];
  VT_CHECK(beta_out.shape[0] == t && beta_out.shape[1] == hv && araw.shape[0] == t &&
               araw.shape[1] == hv && braw.shape[0] == t && braw.shape[1] == hv,
           "gdn_g_beta: g_out/beta_out/araw/braw must all be [T,Hv]");
  VT_CHECK(a_log.rank == 1 && a_log.shape[0] == hv && dt_bias.rank == 1 && dt_bias.shape[0] == hv,
           "gdn_g_beta: a_log/dt_bias must be [Hv]");
  VT_CHECK(g_out.dtype == DType::kF32 && beta_out.dtype == DType::kF32 &&
               (araw.dtype == DType::kF32 || araw.dtype == DType::kBF16) &&
               braw.dtype == araw.dtype && a_log.dtype == DType::kF32 &&
               dt_bias.dtype == DType::kF32,
           "gdn_g_beta: g/beta/a_log/dt_bias f32; a/b must share f32 or bf16");
  VT_CHECK(g_out.IsContiguous() && beta_out.IsContiguous() &&
               araw.stride[1] == 1 && braw.stride[1] == 1 &&
               araw.stride[0] >= hv && braw.stride[0] >= hv &&
               a_log.IsContiguous() && dt_bias.IsContiguous(),
           "gdn_g_beta: outputs/vectors contiguous; a/b inner-contiguous row views required");
  VT_CHECK(g_out.device == q.device && beta_out.device == q.device && araw.device == q.device &&
               braw.device == q.device && a_log.device == q.device && dt_bias.device == q.device,
           "gdn_g_beta: device mismatch (g_out/beta_out/araw/braw/a_log/dt_bias/queue)");
  reinterpret_cast<GdnGBetaFn>(GetOp(OpId::kGdnGBeta, q.device.type))(q, g_out, beta_out, araw,
                                                                      braw, a_log, dt_bias);
}

void GdnConvSplit(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& conv) {
  VT_CHECK(conv.rank == 2, "gdn_conv_split: conv rank-2 [T, conv_dim]");
  const int64_t t = conv.shape[0];
  VT_CHECK(t > 0, "gdn_conv_split: T must be > 0");
  VT_CHECK(q_out.Numel() % t == 0 && k_out.Numel() % t == 0 && v_out.Numel() % t == 0,
           "gdn_conv_split: q_out/k_out/v_out element counts must be divisible by T");
  const int64_t key_dim = q_out.Numel() / t, value_dim = v_out.Numel() / t;
  VT_CHECK(k_out.Numel() / t == key_dim, "gdn_conv_split: q_out and k_out must share key_dim");
  VT_CHECK(conv.shape[1] == 2 * key_dim + value_dim,
           "gdn_conv_split: conv_dim must be 2*key_dim + value_dim");
  // q/k/v out may be f32 OR bf16 (coupled GDN bf16 path); conv may be f32 OR bf16
  // (input-side bf16 GDN path, VT_GDN_IN_BF16) — the kernel upcasts via Load().
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype && v_out.dtype == q_out.dtype &&
               (conv.dtype == DType::kF32 || conv.dtype == DType::kBF16),
           "gdn_conv_split: q/k/v out f32 or bf16 (same dtype), conv f32 or bf16");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               conv.IsContiguous(),
           "gdn_conv_split: contiguous required");
  VT_CHECK(q_out.device == q.device && k_out.device == q.device && v_out.device == q.device &&
               conv.device == q.device,
           "gdn_conv_split: device mismatch (q_out/k_out/v_out/conv/queue)");
  reinterpret_cast<GdnConvSplitFn>(GetOp(OpId::kGdnConvSplit, q.device.type))(q, q_out, k_out,
                                                                              v_out, conv);
}

void QkvSplit(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  VT_CHECK(qkv.rank == 2, "qkv_split: qkv rank-2 [T, q_dim+k_dim+v_dim]");
  const int64_t t = qkv.shape[0];
  VT_CHECK(t > 0, "qkv_split: T must be > 0");
  VT_CHECK(q_out.Numel() % t == 0 && k_out.Numel() % t == 0 && v_out.Numel() % t == 0,
           "qkv_split: q_out/k_out/v_out element counts must be divisible by T");
  const int64_t q_dim = q_out.Numel() / t, k_dim = k_out.Numel() / t, v_dim = v_out.Numel() / t;
  VT_CHECK(qkv.shape[1] == q_dim + k_dim + v_dim,
           "qkv_split: qkv inner dim must be q_dim + k_dim + v_dim");
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype && v_out.dtype == q_out.dtype &&
               qkv.dtype == q_out.dtype,
           "qkv_split: q/k/v out and qkv must share dtype (f32 or bf16)");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               qkv.IsContiguous(),
           "qkv_split: contiguous required");
  VT_CHECK(q_out.device == q.device && k_out.device == q.device && v_out.device == q.device &&
               qkv.device == q.device,
           "qkv_split: device mismatch (q_out/k_out/v_out/qkv/queue)");
  reinterpret_cast<QkvSplitFn>(GetOp(OpId::kQkvSplit, q.device.type))(q, q_out, k_out, v_out, qkv);
}

void GdnPostConv(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                 Tensor& beta_out, const Tensor& conv, const Tensor& araw, const Tensor& braw,
                 const Tensor& a_log, const Tensor& dt_bias, const L2NormArgs& args) {
  // Fusion of GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta; validation is the
  // union of those four ops (same shape/dtype/device/contiguity contracts).
  VT_CHECK(conv.rank == 2, "gdn_post_conv: conv rank-2 [T, conv_dim]");
  VT_CHECK(q_out.rank == 3 && k_out.rank == 3 && v_out.rank == 3,
           "gdn_post_conv: q_out/k_out/v_out rank-3 [T,H,D]");
  const int64_t t = conv.shape[0];
  VT_CHECK(t > 0, "gdn_post_conv: T must be > 0");
  VT_CHECK(q_out.shape[0] == t && k_out.shape[0] == t && v_out.shape[0] == t,
           "gdn_post_conv: q_out/k_out/v_out leading dim must be T");
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  VT_CHECK(k_out.shape[1] == hk && k_out.shape[2] == dk,
           "gdn_post_conv: q_out and k_out must share [T,Hk,Dk]");
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  VT_CHECK(conv.shape[1] == 2 * key_dim + value_dim,
           "gdn_post_conv: conv_dim must be 2*key_dim + value_dim");
  VT_CHECK(g_out.rank == 2 && beta_out.rank == 2 && araw.rank == 2 && braw.rank == 2,
           "gdn_post_conv: g_out/beta_out/araw/braw rank-2 [T,Hv]");
  VT_CHECK(g_out.shape[0] == t && g_out.shape[1] == hv && beta_out.shape[0] == t &&
               beta_out.shape[1] == hv && araw.shape[0] == t && araw.shape[1] == hv &&
               braw.shape[0] == t && braw.shape[1] == hv,
           "gdn_post_conv: g_out/beta_out/araw/braw must all be [T,Hv]");
  VT_CHECK(a_log.rank == 1 && a_log.shape[0] == hv && dt_bias.rank == 1 && dt_bias.shape[0] == hv,
           "gdn_post_conv: a_log/dt_bias must be [Hv]");
  // q/k/v activations may be f32 OR bf16 (coupled GDN bf16 path, VT_GDN_BF16):
  // bf16 feeds the WMMA chunk-scan as native bf16 fragments. All three must share
  // one dtype (the scan requires q.dtype==k.dtype==v.dtype). g/beta and the
  // araw/braw may share f32 OR bf16 dtype. vLLM's packed BA projection emits
  // model-dtype bf16 and slicing it produces row-strided inner-contiguous
  // views; kernels upcast those loads while g/beta remain f32. conv may be f32
  // OR bf16 (input-side bf16 GDN path, VT_GDN_IN_BF16).
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype && v_out.dtype == q_out.dtype,
           "gdn_post_conv: q_out/k_out/v_out must be f32 or bf16, same dtype");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "gdn_post_conv: conv must be f32 or bf16");
  VT_CHECK(g_out.dtype == DType::kF32 && beta_out.dtype == DType::kF32 &&
               (araw.dtype == DType::kF32 || araw.dtype == DType::kBF16) &&
               braw.dtype == araw.dtype && a_log.dtype == DType::kF32 &&
               dt_bias.dtype == DType::kF32,
           "gdn_post_conv: g/beta/a_log/dt_bias f32; a/b must share f32 or bf16");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               g_out.IsContiguous() && beta_out.IsContiguous() && conv.IsContiguous() &&
               araw.stride[1] == 1 && braw.stride[1] == 1 &&
               araw.stride[0] >= hv && braw.stride[0] >= hv &&
               a_log.IsContiguous() && dt_bias.IsContiguous(),
           "gdn_post_conv: outputs/conv/vectors contiguous; a/b inner-contiguous row views required");
  VT_CHECK(q_out.device == q.device && k_out.device == q.device && v_out.device == q.device &&
               g_out.device == q.device && beta_out.device == q.device && conv.device == q.device &&
               araw.device == q.device && braw.device == q.device && a_log.device == q.device &&
               dt_bias.device == q.device,
           "gdn_post_conv: device mismatch");
  reinterpret_cast<GdnPostConvFn>(GetOp(OpId::kGdnPostConv, q.device.type))(
      q, q_out, k_out, v_out, g_out, beta_out, conv, araw, braw, a_log, dt_bias, args);
}

void SharedExpertGate(Queue& q, Tensor& out, const Tensor& sd, const Tensor& gl) {
  VT_CHECK(out.rank == 2, "shared_expert_gate: out rank-2 [T,H]");
  const int64_t t = out.shape[0], h = out.shape[1];
  VT_CHECK(out.dtype == DType::kBF16, "shared_expert_gate: out must be bf16");
  VT_CHECK(sd.dtype == DType::kF32 && gl.dtype == DType::kF32,
           "shared_expert_gate: sd/gl must be f32");
  VT_CHECK(sd.Numel() == t * h, "shared_expert_gate: sd must have T*H elements matching out");
  VT_CHECK(gl.Numel() == t, "shared_expert_gate: gl must have T elements (one gate per token)");
  VT_CHECK(out.IsContiguous() && sd.IsContiguous() && gl.IsContiguous(),
           "shared_expert_gate: contiguous required");
  VT_CHECK(out.device == q.device && sd.device == q.device && gl.device == q.device,
           "shared_expert_gate: device mismatch (out/sd/gl/queue)");
  reinterpret_cast<SharedExpertGateFn>(GetOp(OpId::kSharedExpertGate, q.device.type))(q, out, sd,
                                                                                      gl);
}

}  // namespace vt
