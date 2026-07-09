# triton_kernels/wy_fast.py
#
# GDN WY-representation step 3/3 (recompute w,u), the BIGGEST GDN chunk kernel and
# part of the SANCTIONED CUDA-only Triton AOT fast-path for the WU pipeline (see
# .agents/discipline.md "SANCTIONED EXCEPTION", porting-inventory.md §9). Compiled
# to a cubin at BUILD time (cmake/TritonAOT.cmake, gated VLLM_CPP_TRITON=OFF by
# default); the RUNTIME is Triton/Python-free. The portable hand-C++ fused kernel
# (GdnChunkWUWmmaVecKernel in src/vt/cuda/cuda_gdn.cu — fuses kkt + solve_tril +
# recompute_w_u) + the CPU reference are PRESERVED as the fallback and remain the
# default when VLLM_CPP_TRITON=OFF.
#
# Ported VERBATIM FROM (vLLM 0.24.0 oracle venv):
#   vllm/model_executor/layers/fla/ops/wy_fast.py:29-120
#     @triton.jit recompute_w_u_fwd_kernel
#   (upstream flash-linear-attention; MIT). Adaptations for AOT only:
#     (1) @triton.heuristics / @triton.autotune removed — H,Hg,K,V,BT,BK,BV,IS_VARLEN
#         PINNED per-shape via the compile SIGNATURE (winning num_warps/num_stages
#         there). FLA hardcodes BK=64, BV=64 (recompute_w_u_fwd), pinned identically.
#     (2) trailing runtime scalar `NT` (= total chunks) appended as grid-x carrier:
#         FLA grid is (NT, B*H); B*H == H (varlen packing has B=1) is the constexpr
#         grid-y (baked per spec). `NT` is unused by the body (dead arg).
# Inputs: k=[T,Hg,K] bf16, v(=ORIGINAL v, not v_new)=[T,H,V] bf16, beta=[T,H] f32,
# A(=Ai from solve_tril)=[T,H,BT] bf16, g(=gcum within-chunk cumsum)=[T,H] f32.
# Outputs: w=[T,H,K] bf16 = A@(beta*exp(g)*k), u=[T,H,V] bf16 = A@(beta*v).
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "NT"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    NT,  # AOT grid-carrier (= total chunks): grid-x extent; unused by the body.
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
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
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
