# triton_kernels/rmsnorm.py
#
# Gemma RMSNorm, f32 accumulate — the PROOF kernel for the Triton AOT toolchain
# (cmake/TritonAOT.cmake). It is deliberately an ALTERNATIVE realization of an op
# we already have in hand-C++ (vt::RmsNorm / RmsNormRowKernel in
# src/vt/cuda/cuda_ops.cu) so it is A/B-able and token-exact-checkable.
#
# Mirrors vLLM GemmaRMSNorm.forward_native
#   (vllm/model_executor/layers/layernorm.py:129-157):
#       weight = self.weight.float() + 1.0
#       out    = x * rsqrt(mean(x^2) + eps) * weight
#   i.e.  out = x * rsqrt(mean(x^2) + eps) * (1 + w), variance accumulated in f32.
# This matches RmsNormRowKernel with gemma=true (one block per row, f32 reduction,
# then v * inv * (w + 1)).
#
# Layout: x/out are [n_rows, n_cols] row-major; w is [n_cols]. One Triton program
# handles one row (grid = (n_rows, 1, 1)); it reduces the whole row of n_cols
# elements (n_cols <= BLOCK_SIZE) with a masked tail, so any row width works as
# long as it fits BLOCK_SIZE. n_rows is passed only so the launch grid can be
# expressed as "n_rows,1,1"; it is not read in the body.
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd(
    out_ptr,   # *fp32  [n_rows, n_cols]  output
    x_ptr,     # *fp32  [n_rows, n_cols]  input
    w_ptr,     # *fp32  [n_cols]          raw weight (the +1 is applied here)
    n_rows,    # i32    number of rows (drives the launch grid only)
    n_cols,    # i32    row width (H)
    eps,       # fp32   variance epsilon
    BLOCK_SIZE: tl.constexpr,  # power-of-two >= n_cols
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * inv * (1.0 + w)
    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)
