# triton_kernels/chunk_o.py
#
# GDN chunked output kernel (chunk_o), the SANCTIONED CUDA-only Triton AOT
# fast-path for the PROVEN codegen-bound kernel (see .agents/discipline.md
# "SANCTIONED EXCEPTION", .agents/mission.md, porting-inventory.md §9). Compiled
# to a cubin at BUILD time (cmake/TritonAOT.cmake, gated VLLM_CPP_TRITON=OFF by
# default) and embedded in libvllm; the RUNTIME is Triton/Python-free (cubin via
# the CUDA driver API). The portable hand-C++ CUDA kernel (GdnChunkOWmmaKernel in
# src/vt/cuda/cuda_gdn.cu) + the CPU reference are PRESERVED as the fallback and
# remain the default when VLLM_CPP_TRITON=OFF.
#
# Ported VERBATIM FROM (vLLM 0.24.0 oracle venv):
#   vllm/model_executor/layers/fla/ops/chunk_o.py:47-160
#     @triton.jit chunk_fwd_kernel_o
#   (upstream flash-linear-attention; MIT). The kernel BODY is byte-for-byte the
#   FLA source. The ONLY adaptations for AOT compilation are:
#     (1) the @triton.heuristics / @triton.autotune decorators are removed — the
#         flags (USE_G,IS_VARLEN) and dims (H,Hg,K,V,BT,BK,BV) are PINNED per-shape
#         via the triton.tools.compile SIGNATURE (see CMakeLists.txt), and the
#         winning (BK,BV,num_warps,num_stages) is selected there, exactly what the
#         autotuner would pick for the gate shape;
#     (2) one trailing runtime scalar `NT` (= total number of chunks across all
#         sequences) is appended: the FLA launch grid is (cdiv(V,BV), NT, B*H) but
#         NT is NOT a kernel argument, so the grid-y extent is carried in `NT` for
#         the AOT launcher's baked grid expression. `B*H` == H because our varlen
#         packing has B=1 (single packed [T,H,*] tensor + cu_seqlens), so grid-z is
#         the constexpr H (baked per spec). `NT` is unused by the kernel body (dead
#         arg → eliminated → the compute codegen is identical to FLA's).
#     (3) the runtime `scale` arg is REMOVED and pinned to Dk^-0.5 (= K**-0.5, the
#         GDN q-scale; K=128 pinned). Triton's AOT launcher mis-packs an fp32 scalar
#         as an 8-byte double (the kernel reads 4 bytes → garbage), so a runtime
#         float scalar is unusable here; the model always passes scale == Dk^-0.5
#         (qwen3_5.cpp: scale = 1/sqrt(Dk)) and the dispatch (TryTritonChunkO) guards
#         that args.scale matches before firing, so pinning it is exact.
#
# Pinned flags for the vllm.cpp GDN chunked-prefill call site: USE_G=1,
# IS_VARLEN=1. Buffer layout is a 1:1 drop-in (verified stride-for-stride against
# the FLA pointer arithmetic and our hand GdnChunkOWmmaKernel):
#   q=[T,Hg,K] bf16, k=[T,Hg,K] bf16, v(=v_new)=[T,H,V] bf16,
#   h(=hstate snapshot)=[NT,H,V,K] bf16, g(=gcum within-chunk cumsum)=[T,H] f32,
#   o(=out)=[T,H,V] f32 or bf16, selected by the AOT signature. The bf16
#   specialization mirrors FLA's empty_like(v) / output-buffer dtype and is
#   evaluated before VT_GDN_OUT_BF16 can become a default. cu_seqlens=[N+1] i32,
#   chunk_indices=[NT,2] i32 (per global chunk: (i_n, i_t_local)).
import triton
import triton.language as tl

# fla/ops/op.py: exp (FLA_USE_FAST_OPS=0 default -> tl.exp).
exp = tl.exp


@triton.jit(do_not_specialize=["T", "NT"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    NT,  # AOT grid-carrier (= total chunks): grid-y extent; unused by the body.
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # scale = Dk^-0.5 (GDN q-scale), pinned since Triton AOT can't take an fp32
    # scalar arg reliably; K == Dk == 128 for the gate shape (see header note 3).
    scale = K ** -0.5
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, tl.trans(b_h))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
