#!/usr/bin/env python3
# ROAD-V1-D4-APC W3 — capture the vLLM 0.26.0.dev0 oracle greedy continuation for
# the Qwen3-4B AUTOMATIC-PREFIX-CACHING shared-prefix workload, with prefix caching
# ENABLED (vLLM's dense default), so the goldens ARE the vLLM-APC-ON reference.
#
# The workload (a long common PREFIX + N varied suffixes) is defined ONCE, in
# tests/parity/test_qwen3_apc_e2e.cpp. That test, run under VT_DUMP_IDS=1 with the
# goldens absent, writes the byte-identical prompt strings to
# tests/parity/goldens/qwen3_apc_4b/prompt_%02d.txt (and our APC-ON token ids to
# our_ids.i32). THIS script reads those prompt files, so there is no prompt
# duplication and no drift risk between the C++ gate and the Python oracle.
#
# Two golden artifacts (see [[near-tie-distributional-gate]]):
#   greedy_ids.npy   [N, T]     i32 — run 0 point golden (strict count report).
#   greedy_dist.npy  [N, T, K]  i32 — the per-(prompt,position) SET of tokens vLLM
#                                     emits across K greedy runs. The near-tie
#                                     DISTRIBUTIONAL gate PASSES when OUR APC-ON
#                                     token is a MEMBER of vLLM's observed set at
#                                     that step. Short runs (early EOS) pad -1.
#
# Qwen3-4B bf16 greedy is a near-tie decoder (its kernels' reduction order flips
# greedy at tiny top1<->top2 gaps); the honest "mirror vLLM" bar is membership in
# vLLM's actual observed K-run distribution, strict where vLLM is deterministic.
#
# Run on dgx with the oracle venv; PATH must include the venv bin (flashinfer JIT):
#   flock "${GPU_LOCK:-$HOME/gpu.lock}" \
#   env PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     $HOME/venvs/vllm-oracle/bin/python scripts/qwen3-apc-oracle-capture.py \
#       --model Qwen/Qwen3-4B --runs 5 --gpu-mem 0.40
import argparse
import glob
import os
import sys

import numpy as np


def default_out_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "parity", "goldens", "qwen3_apc_4b",
    )


def load_prompts(out_dir):
    """Read prompt_%02d.txt (raw bytes, may contain newlines) in index order."""
    files = sorted(glob.glob(os.path.join(out_dir, "prompt_*.txt")))
    if not files:
        sys.exit(f"no prompt_*.txt in {out_dir}; run the W3 test under VT_DUMP_IDS=1 "
                 f"first (bootstrap dump)")
    prompts = []
    for f in files:
        with open(f, "rb") as fh:
            prompts.append(fh.read().decode("utf-8"))
    return prompts


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-4B"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--runs", type=int, default=int(os.environ.get("QWEN3_RUNS", "5")),
                    help="K greedy runs to build the observed distribution (K>=1)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float,
                    default=float(os.environ.get("QWEN3_GPU_MEM", "0.40")),
                    help="gpu_memory_utilization (GB10 unified memory — keep low)")
    return ap.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    out_dir = args.out_dir or default_out_dir()
    prompts = load_prompts(out_dir)
    N = len(prompts)
    T = args.max_tokens
    K = max(1, args.runs)
    sp = SamplingParams(temperature=0.0, max_tokens=T)

    # Capture vLLM's greedy tokens under BOTH APC modes (enable_prefix_caching
    # False and True), K runs each, and UNION them into greedy_dist's last axis.
    # This is the fair near-tie reference: a Qwen3-4B bf16 near-tie may resolve one
    # way with the full-prefill attention kernel (APC off) and the other with the
    # cached partial-prefill kernel (APC on) — exactly the ON/OFF sensitivity we
    # score OUR two arms against. The union is what "a token vLLM produces" means.
    dist = np.full((N, T, 2 * K), -1, dtype="<i4")
    run0 = np.zeros((N, T), dtype="<i4")  # first APC-off run (reference greedy).
    slot = 0
    for apc in (False, True):
        print(f"\n=== capturing {args.model}  APC={'ON' if apc else 'OFF'}  "
              f"N={N} T={T} K={K} (per-prompt batch=1) ===")
        llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
                  enable_prefix_caching=apc, gpu_memory_utilization=args.gpu_mem)
        for k in range(K):
            for i, p in enumerate(prompts):
                o = llm.generate([p], sp)[0]
                ids = list(o.outputs[0].token_ids)
                if len(ids) < T:
                    print(f"NOTE apc={apc} run{k} prompt[{i}] produced {len(ids)} "
                          f"tokens (< {T})", file=sys.stderr)
                for j in range(min(T, len(ids))):
                    dist[i, j, slot + k] = ids[j]
                if not apc and k == 0:
                    for j in range(min(T, len(ids))):
                        run0[i, j] = ids[j]
                    print(f"prompt[{i}] (len {len(o.prompt_token_ids)} tok) -> {ids}  "
                          f"({o.outputs[0].text!r})")
        del llm
        slot += K

    np.save(os.path.join(out_dir, "greedy_ids.npy"), run0)
    np.save(os.path.join(out_dir, "greedy_dist.npy"), dist)

    KT = 2 * K
    print(f"\n=== determinism report: {args.model}  N={N} T={T} K={KT} (both modes) ===")
    deterministic = True
    total_multi_pos = 0
    for i in range(N):
        seqs = {tuple(int(x) for x in dist[i, :, k]) for k in range(KT)}
        multi = [j for j in range(T)
                 if len({int(dist[i, j, k]) for k in range(KT)} - {-1}) > 1]
        total_multi_pos += len(multi)
        if len(seqs) > 1:
            deterministic = False
            observed = {j: sorted({int(dist[i, j, k]) for k in range(KT)} - {-1})
                        for j in multi}
            print(f"  prompt[{i}] NON-DET across modes: near-tie positions {observed}")
        else:
            print(f"  prompt[{i}] deterministic (1 sequence over {KT} runs, both modes)")
    print(f"=== {'ALL DETERMINISTIC' if deterministic else 'NON-DETERMINISTIC'} "
          f"over {KT} runs (APC off+on); {total_multi_pos} multi-member (prompt,pos) "
          f"cells ===")
    print(f"wrote {out_dir}/greedy_ids.npy {run0.shape} + "
          f"{out_dir}/greedy_dist.npy {dist.shape}")


if __name__ == "__main__":
    main()
