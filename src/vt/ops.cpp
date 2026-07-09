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
                    float input_global_scale_inv) {
  VT_CHECK(x.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "scaled_fp4_quant: x/out_packed/out_scale must be rank-2");
  const int64_t m = x.shape[0], k = x.shape[1];
  VT_CHECK(k % 16 == 0, "scaled_fp4_quant: K (inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == k / 2,
           "scaled_fp4_quant: out_packed must be [M, K/2]");
  VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == k / 16,
           "scaled_fp4_quant: out_scale must be [M, K/16]");
  VT_CHECK(IsFloat(x.dtype), "scaled_fp4_quant: float x required");
  VT_CHECK(out_packed.dtype == DType::kI8 && out_scale.dtype == DType::kI8,
           "scaled_fp4_quant: out_packed/out_scale must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(x.IsContiguous() && out_packed.IsContiguous() && out_scale.IsContiguous(),
           "scaled_fp4_quant: contiguous tensors required");
  VT_CHECK(x.device == q.device && out_packed.device == q.device && out_scale.device == q.device,
           "scaled_fp4_quant: device mismatch (x/out_packed/out_scale/queue)");
  reinterpret_cast<ScaledFp4QuantFn>(GetOp(OpId::kScaledFp4Quant, q.device.type))(
      q, out_packed, out_scale, x, input_global_scale_inv);
}

void SiluMulFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& gate,
                     const Tensor& up, float input_global_scale_inv) {
  VT_CHECK(gate.rank == 2 && up.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "silu_mul_fp4_quant: gate/up/out_packed/out_scale must be rank-2");
  const int64_t m = gate.shape[0], i = gate.shape[1];
  VT_CHECK(up.shape[0] == m && up.shape[1] == i, "silu_mul_fp4_quant: gate/up shape mismatch");
  VT_CHECK(i % 16 == 0, "silu_mul_fp4_quant: I (inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == i / 2,
           "silu_mul_fp4_quant: out_packed must be [M, I/2]");
  VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == i / 16,
           "silu_mul_fp4_quant: out_scale must be [M, I/16]");
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
      q, out_packed, out_scale, gate, up, input_global_scale_inv);
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

void MatmulNvfp4Cutlass(Queue& q, Tensor& out, const Tensor& a_packed, const Tensor& a_sf_sw,
                        const Tensor& b_packed, const Tensor& b_sf_sw, float alpha) {
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
  reinterpret_cast<MatmulNvfp4CutlassFn>(GetOp(OpId::kMatmulNvfp4Cutlass, q.device.type))(
      q, out, a_packed, a_sf_sw, b_packed, b_sf_sw, alpha);
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

void FusedChain(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, Tensor* residual,
                const FusedRecipe& recipe, float eps) {
  // Phase-0 operand contract mirrors RmsNorm(residual): x/out [T,H], weight [H],
  // optional residual [T,H] f32/bf16. Recipe structure is validated by the
  // backend realization; here we gate the tensor shapes/dtypes/devices so a bad
  // call fails at the chokepoint, not inside a kernel.
  VT_CHECK(recipe.n >= 1 && recipe.n <= kMaxFusedSteps, "fused_chain: empty/oversized recipe");
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
  reinterpret_cast<FusedChainFn>(GetOp(OpId::kFusedChain, q.device.type))(q, out, x, weight,
                                                                          residual, recipe, eps);
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
  VT_CHECK(IsOutFloat(q_out.dtype) && k_out.dtype == q_out.dtype && gate_out.dtype == q_out.dtype,
           "attn_qk_norm_rope_gate: q/k/gate out f32 or bf16 (same dtype)");
  VT_CHECK(IsFloat(qgate.dtype) && kf.dtype == qgate.dtype,
           "attn_qk_norm_rope_gate: qgate/kf float, same dtype");
  VT_CHECK(q_norm.dtype == DType::kF32 && k_norm.dtype == DType::kF32 &&
               cos_sin.dtype == DType::kF32,
           "attn_qk_norm_rope_gate: q_norm/k_norm/cos_sin must be f32");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && gate_out.IsContiguous() &&
               qgate.IsContiguous() && kf.IsContiguous() && q_norm.IsContiguous() &&
               k_norm.IsContiguous() && cos_sin.IsContiguous(),
           "attn_qk_norm_rope_gate: contiguous required");
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
  VT_CHECK(state.dtype == DType::kF32 ||
               (state.dtype == DType::kBF16 && q.device.type == DeviceType::kCUDA),
           std::string(name) +
               ": state must be f32, or bf16 on CUDA (in/out, in place; bf16 = vLLM "
               "default mamba_ssm_cache_dtype, read/written in f32 registers)");
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
               const Tensor& g, const Tensor& beta, Tensor& state, const GdnArgs& args,
               const Tensor* state_idx) {
  CheckGdnCommon(q, out, q_in, k, v, g, beta, state, args, "gdn_decode");
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
  VT_CHECK(out.IsContiguous() && in.IsContiguous(), "cast_bf16: contiguous required");
  VT_CHECK(out.device == q.device && in.device == q.device,
           "cast_bf16: device mismatch (out/in/queue)");
  reinterpret_cast<CastBf16Fn>(GetOp(OpId::kCastBf16, q.device.type))(q, out, in);
}

void CastF32(Queue& q, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kF32, "cast_f32: out must be f32");
  VT_CHECK(in.dtype == DType::kBF16, "cast_f32: in must be bf16");
  VT_CHECK(out.Numel() == in.Numel(), "cast_f32: out/in must have the same element count");
  VT_CHECK(out.IsContiguous() && in.IsContiguous(), "cast_f32: contiguous required");
  VT_CHECK(out.device == q.device && in.device == q.device,
           "cast_f32: device mismatch (out/in/queue)");
  reinterpret_cast<CastF32Fn>(GetOp(OpId::kCastF32, q.device.type))(q, out, in);
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
  VT_CHECK(attn.dtype == DType::kF32 && gate.dtype == DType::kF32,
           "sigmoid_gate_bf16: attn/gate must be f32");
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
               araw.dtype == DType::kF32 && braw.dtype == DType::kF32 &&
               a_log.dtype == DType::kF32 && dt_bias.dtype == DType::kF32,
           "gdn_g_beta: all tensors must be f32");
  VT_CHECK(g_out.IsContiguous() && beta_out.IsContiguous() && araw.IsContiguous() &&
               braw.IsContiguous() && a_log.IsContiguous() && dt_bias.IsContiguous(),
           "gdn_g_beta: contiguous required");
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
  // araw/braw/a_log/dt_bias inputs stay f32 — FLA keeps the gates f32. conv may
  // be f32 OR bf16 (input-side bf16 GDN path, VT_GDN_IN_BF16): the conv-output
  // activation halves its read traffic; the kernel upcasts to f32 (Load()).
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype && v_out.dtype == q_out.dtype,
           "gdn_post_conv: q_out/k_out/v_out must be f32 or bf16, same dtype");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "gdn_post_conv: conv must be f32 or bf16");
  VT_CHECK(g_out.dtype == DType::kF32 && beta_out.dtype == DType::kF32 &&
               araw.dtype == DType::kF32 && braw.dtype == DType::kF32 &&
               a_log.dtype == DType::kF32 && dt_bias.dtype == DType::kF32,
           "gdn_post_conv: g/beta/araw/braw/a_log/dt_bias must be f32");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               g_out.IsContiguous() && beta_out.IsContiguous() && conv.IsContiguous() &&
               araw.IsContiguous() && braw.IsContiguous() && a_log.IsContiguous() &&
               dt_bias.IsContiguous(),
           "gdn_post_conv: contiguous required");
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
