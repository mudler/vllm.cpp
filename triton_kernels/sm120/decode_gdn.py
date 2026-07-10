# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# Adapted from vLLM 0.24.0:
# vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py
# fused_sigmoid_gating_delta_rule_update_kernel. The project computes g and
# beta in its preceding fused post-conv kernel, so this AOT specialization starts
# at the recurrent update while retaining vLLM/FLA's register-tiled [BV, BK]
# state realization.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def gdn_decode_kernel(
    q,
    k,
    v,
    g,
    beta,
    out,
    state,
    state_indices,
    N,
    HK: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_hk = i_hv // (HV // HK)

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    if state_idx < 0:
        return

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mask_h = mask_v[:, None]

    p_q = q + (i_n * HK + i_hk) * K + o_k
    p_k = k + (i_n * HK + i_hk) * K + o_k
    p_v = v + (i_n * HV + i_hv) * V + o_v
    p_h = (
        state
        + (state_idx * HV + i_hv) * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )

    b_q = tl.load(p_q).to(tl.float32) * (K ** -0.5)
    b_k = tl.load(p_k).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    b_h *= tl.exp(tl.load(g + i_n * HV + i_hv).to(tl.float32))
    b_v -= tl.sum(b_h * b_k[None, :], axis=1)
    b_v *= tl.load(beta + i_n * HV + i_hv).to(tl.float32)
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], axis=1)

    tl.store(out + (i_n * HV + i_hv) * V + o_v, b_o, mask=mask_v)
    tl.store(p_h, b_h, mask=mask_h)
