#!/usr/bin/env python3
"""Standalone upstream Marlin MoE driver for #442.

Drives vLLM's OWN `moe_wna16_marlin_gemm` on the 35B-A3B decode shapes with NO
EngineCore, NO multiprocessing and NO model load, so that `ncu` can attach --
which is the blocker recorded in the spec (upstream's engine will not initialise
under ncu in either replay mode).

Shapes come from nvidia/Qwen3.6-35B-A3B-NVFP4: hidden 2048, moe_intermediate
512, E=256, top_k=8. Marlin's runtime is data-independent, so random weights
measure the same kernel the model does.

RUN IT UNDER THE BOX LOCK: `flock "${GPU_LOCK:-$HOME/gpu.lock}" ...` -- the repo's
ONE mutex (#777). Any other path coordinates with nothing, and `flock` succeeds
on it either way, so the divergence is silent. `nvidia-smi` reporting no compute
apps does not mean the GPU is unreserved -- a holder may be between phases -- so
check `fuser -v "${GPU_LOCK:-$HOME/gpu.lock}"` first. Absolute timings taken
unlocked are upper bounds; only interleaved RATIOS survive contention.
"""

import argparse
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=9, help="tokens (1 + k spec draft)")
    ap.add_argument("--iters", type=int, default=80)  # matches the C++ arm
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--arm", choices=["gate_up", "down", "both"], default="both")
    ap.add_argument("--bps", type=int, default=-1,
                    help="blocks_per_sm passed to the kernel (-1 = auto)")
    ap.add_argument("--experts", type=int, default=0,
                    help="draw routing from only this many distinct experts "
                         "(0 = all E); controls occupied block count")
    args = ap.parse_args()

    import vllm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as f4
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.scalar_type import scalar_types

    # ORACLE IDENTITY. A rollback venv silently invalidates every number here;
    # this repo has already measured a rollback three times believing it was the
    # pin. Abort rather than measure the wrong build.
    want = "555967922"
    if want not in vllm.__version__:
        print(f"ABORT: vllm {vllm.__version__} is not the pinned {want}")
        return 2
    print(f"vllm={vllm.__version__} torch={torch.__version__}", flush=True)

    dev = torch.device("cuda")
    torch.cuda.init()

    E, K, N, TOPK = 256, 2048, 512, 8
    GROUP = 16
    BLOCK_M = 8
    M = args.m
    dtype = torch.bfloat16
    quant_type = scalar_types.float4_e2m1f

    perm = torch.empty(0, dtype=torch.int, device=dev)

    def build(size_n: int, size_k: int):
        """Repacked weights + processed scales for one grouped GEMM.

        Mirrors prepare_nvfp4_moe_layer_for_marlin's permute_scales exactly --
        per-expert marlin_permute_scales on the transposed scale, one shared
        scale_factor across experts, then the S0E5M3 conversion.
        """
        w = torch.randint(0, 255, (E, size_n, size_k // 2), dtype=torch.uint8,
                          device=dev)
        wq = f4._repack_marlin_experts(w, size_n, size_k, perm, False)

        # Positive scales, as NVFP4 weight scales always are.
        s = (torch.rand((E, size_n, size_k // GROUP), dtype=dtype, device=dev)
             * 0.5 + 0.5)
        factor = f4._nvfp4_compute_scale_factor(s, dtype)
        per_expert = []
        for i in range(E):
            ms = f4.marlin_permute_scales(s=s[i].T, size_k=size_k,
                                          size_n=size_n, group_size=GROUP,
                                          is_a_8bit=False)
            ms, _ = f4.nvfp4_marlin_process_scales(ms, scale_factor=factor,
                                                   a_dtype=dtype)
            per_expert.append(ms)
        s_proc = torch.cat([x.unsqueeze(0) for x in per_expert], 0)

        s2 = torch.ones(E, dtype=torch.float32, device=dev)
        g_proc = f4.nvfp4_marlin_process_global_scale(s2, dtype) / factor
        return wq, s_proc, g_proc

    print("building weights...", flush=True)
    w1, w1s, w1g = build(2 * N, K)     # gate_up: size_n = 2*512, size_k = 2048
    w2, w2s, w2g = build(K, N)         # down:    size_n = 2048,  size_k = 512
    workspace = marlin_make_workspace_new(dev, 4)

    pool = args.experts if args.experts > 0 else E
    # SEEDED: routing dominates the time, so every configuration under
    # comparison must see the IDENTICAL draw or the sweep measures the draw.
    torch.manual_seed(20260813)
    topk_ids = torch.randint(0, pool, (M, TOPK), dtype=torch.int32, device=dev)
    topk_w = torch.rand((M, TOPK), dtype=torch.float32, device=dev)
    sorted_ids, expert_ids, num_past = moe_align_block_size(
        topk_ids, BLOCK_M, E, None)
    distinct = int(torch.unique(topk_ids).numel())

    a1 = torch.randn((M, K), dtype=dtype, device=dev)
    c1 = torch.empty((M * TOPK, 2 * N), dtype=dtype, device=dev)
    a2 = torch.randn((M * TOPK, N), dtype=dtype, device=dev)
    c2 = torch.empty((M * TOPK, K), dtype=dtype, device=dev)

    ops = torch.ops._moe_C

    def gate_up():
        return ops.moe_wna16_marlin_gemm(
            a1, c1, w1, None, w1s, None, w1g, None, None, None, workspace,
            sorted_ids, expert_ids, num_past, topk_w,
            BLOCK_M, TOPK, False, quant_type.id, M, 2 * N, K,
            True, False, True, False, -1, -1, args.bps)

    def down():
        return ops.moe_wna16_marlin_gemm(
            a2, c2, w2, None, w2s, None, w2g, None, None, None, workspace,
            sorted_ids, expert_ids, num_past, topk_w,
            BLOCK_M, 1, True, quant_type.id, M * TOPK, K, N,
            True, False, True, False, -1, -1, args.bps)

    arms = {"gate_up": gate_up, "down": down}
    if args.arm != "both":
        arms = {args.arm: arms[args.arm]}

    for name, fn in arms.items():
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(args.iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        us = st.elapsed_time(en) * 1000.0 / args.iters
        blocks = int(num_past.item()) // BLOCK_M
        print(f"UPSTREAM {name} bps={args.bps} M={M} pool={pool} distinct={distinct} blocks={blocks} "
              f"us_per_call={us:.3f} us_per_block={us/max(blocks,1):.4f}",
              flush=True)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
