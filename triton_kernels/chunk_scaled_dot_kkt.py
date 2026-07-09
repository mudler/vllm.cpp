# triton_kernels/chunk_scaled_dot_kkt.py
#
# GDN WY-representation step 1/3 (scaled dot K@Kᵀ), part of the SANCTIONED
# CUDA-only Triton AOT fast-path for the WU (recompute_w_u) pipeline (see
# .agents/discipline.md "SANCTIONED EXCEPTION", porting-inventory.md §9). Compiled
# to a cubin at BUILD time (cmake/TritonAOT.cmake, gated VLLM_CPP_TRITON=OFF by
# default); the RUNTIME is Triton/Python-free. The portable hand-C++ fused kernel
# (GdnChunkWUWmmaVecKernel in src/vt/cuda/cuda_gdn.cu — which fuses this + solve_tril
# + recompute_w_u) + the CPU reference are PRESERVED as the fallback and remain the
# default when VLLM_CPP_TRITON=OFF.
#
# Ported VERBATIM FROM (vLLM 0.24.0 oracle venv):
#   vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py:36-100
#     @triton.jit chunk_scaled_dot_kkt_fwd_kernel
#   (upstream flash-linear-attention; MIT). Adaptations for AOT only:
#     (1) @triton.heuristics / @triton.autotune removed — flags (USE_G,IS_VARLEN)
#         and dims (H,Hg,K,BT,BK) PINNED per-shape via the compile SIGNATURE.
#     (2) trailing runtime scalar `NT` (= total chunks) appended as the grid-x
#         carrier: FLA grid is (NT, B*H); B*H == H (varlen packing has B=1) is the
#         constexpr grid-y (baked per spec). `NT` is unused by the body (dead arg).
# Pinned: USE_G=1, IS_VARLEN=1, BK=64 (K=128 -> 2 iters). Output A is f32
# [T,H,BT] = beta*K*Kᵀ*exp(g_diff), strictly-lower-tri masked (the raw matrix that
# solve_tril inverts).
import triton
import triton.language as tl

exp = tl.exp


@triton.jit(do_not_specialize=["T", "NT"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    NT,  # AOT grid-carrier (= total chunks): grid-x extent; unused by the body.
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
