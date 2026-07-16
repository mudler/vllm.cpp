# triton_kernels/fused_recurrent_packed_decode.py
#
# GDN packed pure-decode recurrence (fused_recurrent_gated_delta_rule packed
# decode), the SANCTIONED CUDA-only Triton AOT fast-path for the PROVEN
# codegen-bound kernel (see .agents/discipline.md "SANCTIONED EXCEPTION",
# .agents/mission.md, porting-inventory.md §9). Compiled to a cubin at BUILD time
# (cmake/TritonAOT.cmake, gated VLLM_CPP_TRITON=OFF by default) and embedded in
# libvllm; the RUNTIME is Triton/Python-free (cubin via the CUDA driver API). The
# portable hand-C++ CUDA kernel (GdnPackedDecodeKernel in src/vt/cuda/cuda_gdn.cu)
# + the CPU reference are PRESERVED as the fallback and remain the DEFAULT even
# when VLLM_CPP_TRITON=ON (the Triton decode path is opt-in behind
# VT_GDN_PACKED_DECODE_TRITON).
#
# WHY AOT (MEASURED, 2026-07-16, root dgx:~/work/vllm.cpp-gdn-recurrence/phase1):
# the identical register-resident [BV=32, BK=128] fp32 state tile compiles to
# REG:205 STACK:0 LOCAL:0 (ZERO spills) under Triton/ptxas, but the byte-for-byte
# hand-CUDA port (GdnPackedDecodeRegTileKernel, `float sh[128]` + #pragma unroll)
# forces NVCC to REG:255 (the hard ceiling) + STACK:48 (spills to local) — fatal
# for this state-bandwidth-bound decode (the naive port measured 700.5 tok/s vs
# 793.6 legacy at c16, and FAILED the oracle boundary, on DGX 54f0541). The gap
# is register allocation / codegen, not algorithm/structure (the structure was
# ported 1:1). This is exactly the codegen-bound case the sanctioned exception
# covers; the sibling GDN chunk/WY kernels already took this lane.
#
# Ported VERBATIM FROM (vLLM 0.25.0 oracle venv @ pin
# 702f4814fe54fabff350d43cb753ae3e47c0c276):
#   vllm/model_executor/layers/fla/ops/fused_recurrent.py:256-336
#     @triton.jit fused_recurrent_gated_delta_rule_packed_decode_kernel
#     (launch :439-478: num_warps=1, num_stages=3, grid=(cdiv(V,BV), B*HV))
#   (upstream flash-linear-attention; MIT). The kernel BODY is byte-for-byte the
#   FLA source. The ONLY adaptations for AOT compilation are:
#     (1) the runtime `scale` arg is REMOVED and pinned to K**-0.5 (the GDN
#         q-scale). Triton's AOT launcher mis-packs an fp32 scalar as an 8-byte
#         double (the kernel reads 4 bytes -> garbage), so a runtime float scalar
#         is unusable (identical constraint to chunk_o.py); the model always
#         passes scale == Dk^-0.5 (qwen3_5.cpp: scale = 1/sqrt(Dk)) and the
#         dispatch (TryTritonPackedDecode) guards args.scale matches before firing.
#     (2) the constexpr strides/dims (stride_*, H, HV, K, V, BK, BV,
#         SOFTPLUS_THRESHOLD, USE_QK_L2NORM_IN_KERNEL) are PINNED per-shape via the
#         triton.tools.compile SIGNATURE (see cmake/TritonAOTKernels.cmake) to the
#         27B gate-model packed-decode call site; the launcher guards every stride
#         and dim before firing and otherwise falls back to the hand kernel.
#     (3) one trailing runtime scalar `NBH` (= B*HV, the number of
#         (sequence, value-head) pairs) is appended: the FLA launch grid is
#         (cdiv(V,BV), B*HV) but B (the sequence count) is NOT a kernel argument,
#         so the grid-y extent is carried in `NBH` for the AOT launcher's baked
#         grid expression (grid-x = cdiv(V,BV) = 4 is baked). `NBH` is unused by
#         the kernel body (dead arg -> eliminated -> the compute codegen is
#         identical to FLA's). Mirrors chunk_o.py's `NT` and chunk_delta_h.py's
#         `NH` grid carriers.
#     (4) STATE-INDEX ABI ADAPTER (documented in .agents/specs/gdn-packed-decode.md
#         "State-index adapter"): FLA skips `state_idx <= 0` (vLLM reserves cache
#         slot 0 as NULL_BLOCK_ID). Our compact per-sequence state pool uses
#         `slot < 0 == skip, slot 0 == valid`, so the skip test is `state_idx < 0`.
#         This is a cache-ABI translation matching the hand kernel and CPU
#         reference, NOT a recurrence deviation; it does not affect the boundary
#         oracle (indices there are >= 1).
#
# Pinned config for the vllm.cpp 27B GDN packed pure-decode call site
# (qwen3_5.cpp GdnPackedDecode; mirror fused_recurrent.py packed decode):
#   H=16 (Hk), HV=48 (Hv), K=128 (Dk), V=128 (Dv), BK=128, BV=32,
#   SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=1.
# Buffer layout is a 1:1 drop-in (verified stride-for-stride against the FLA
# pointer arithmetic and our hand GdnPackedDecodeKernel):
#   mixed_qkv=[B, 2*H*K + HV*V]=[B,10240] bf16 row-contiguous (stride 10240),
#   a=[B,HV] bf16 / b=[B,HV] bf16 (the merged BA view: b at col 0, a at col HV,
#     each row stride 2*HV=96), A_log=[HV] f32 / dt_bias=[HV] f32 (loader upcasts),
#   o=[B,HV,V] bf16 (stride HV*V then V), h0=ht=state=[slots,HV,V,K] f32
#     (stride HV*V*K=786432), ssm_state_indices=[B] i32.
import triton
import triton.language as tl

# fla/ops/op.py: exp (FLA_USE_FAST_OPS=0 default -> tl.exp).
exp = tl.exp


@triton.jit(do_not_specialize=["NBH"])
def fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    NBH,  # AOT grid-carrier (= B*HV): grid-y extent; unused by the body. See header.
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    # scale = Dk^-0.5 (GDN q-scale), pinned since Triton AOT can't take an fp32
    # runtime scalar (see header adaptation (1)).
    scale = K ** -0.5

    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    # Skip if state index is invalid. vllm.cpp cache ABI: slot < 0 == skip,
    # slot 0 == valid (adaptation (4); FLA upstream skips <= 0).
    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
