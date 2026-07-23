#!/usr/bin/env python3
# ADDITIVE-MODEL W4 — capture the vLLM 0.25.0 oracle greedy continuation for a
# Qwen3 DENSE checkpoint on the standard prompt battery.
#
# Two golden artifacts are produced (see [[near-tie-distributional-gate]]):
#   greedy_ids.npy   [N, T]     i32 — run 0 (the point golden; the STRICT gate on
#                                     BIGGER dense models where vLLM is deterministic).
#   greedy_dist.npy  [N, T, K]  i32 — the per-(prompt,position) SET of tokens vLLM
#                                     emits across K greedy runs. The near-tie
#                                     DISTRIBUTIONAL gate (Qwen3-0.6B) PASSES when
#                                     OUR greedy token at each step is a MEMBER of
#                                     vLLM's observed set at that step. Short runs
#                                     (early EOS) are right-padded with -1.
#
# vLLM 0.25.0's OWN bf16 greedy is NON-DETERMINISTIC on near-ties (Qwen3-0.6B,
# FLASH_ATTN, head_dim 128): its kernels' reduction order varies run-to-run and
# flips greedy at top1<->top2 gaps <=~0.125 nats. So the "16/16 vs a FIXED golden"
# bar is ill-posed there; the honest "mirror vLLM" bar is membership in vLLM's
# actual observed K-run distribution. On a BIGGER dense model (Qwen3-4B) vLLM is
# expected deterministic -> the point golden gives a well-posed STRICT gate; this
# script's determinism report tells you which.
#
# The prompt list MUST match tests/parity/test_qwen3_paged_engine.cpp::Prompts()
# exactly (goldens and gate never drift). Run on dgx with the oracle venv; PATH
# must include ~/venvs/vllm-oracle/bin (flashinfer's ninja JIT):
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/qwen3-oracle-capture.py \
#       --model Qwen/Qwen3-0.6B --runs 10 --out-dir <goldens>/qwen3_greedy_0_6b
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/qwen3-oracle-capture.py \
#       --model Qwen/Qwen3-4B  --runs 3  --out-dir <goldens>/qwen3_greedy_4b
import argparse
import os
import sys
import numpy as np

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "In the beginning God created",
    "The quick brown fox jumps over",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "The theory of relativity was developed by",
    "To be or not to be, that is",
    "The largest planet in our solar system is",
    "Machine learning is a subfield of",
    "The mitochondria is the powerhouse of",
    "Roses are red, violets are",
    "The first president of the United States was",
    "E equals m c",
    "A journey of a thousand miles begins with",
    "The chemical symbol for gold is",
]


def default_out_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "parity", "goldens", "qwen3_greedy_0_6b",
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--out-dir", default=None,
                    help="golden dir (default tests/parity/goldens/qwen3_greedy_0_6b)")
    ap.add_argument("--runs", type=int, default=int(os.environ.get("QWEN3_RUNS", "10")),
                    help="K greedy runs to build the observed distribution (K>=1)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float,
                    default=float(os.environ.get("QWEN3_GPU_MEM", "0.9")),
                    help="gpu_memory_utilization (GB10 unified memory reboots at "
                         "high values — pass 0.40 on dgx)")
    ap.add_argument("--per-prompt", action="store_true",
                    help="decode each prompt in its OWN generate() call (batch size 1) "
                         "to match the paged-engine gate's single-request decode "
                         "regime; otherwise all prompts are batched in one call")
    return ap.parse_args()


def generate_all(llm, sp, per_prompt):
    """Return {prompt: token_ids} for one greedy pass over PROMPTS."""
    from_ids = {}
    if per_prompt:
        for p in PROMPTS:
            o = llm.generate([p], sp)[0]
            from_ids[p] = o
    else:
        for o in llm.generate(PROMPTS, sp):
            from_ids[o.prompt] = o
    return from_ids


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    out_dir = args.out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    N = len(PROMPTS)
    T = args.max_tokens
    K = max(1, args.runs)

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem)
    sp = SamplingParams(temperature=0.0, max_tokens=T)

    # runs[k][i] = list of token ids for prompt i on run k (padded to T with -1).
    dist = np.full((N, T, K), -1, dtype="<i4")
    run0 = np.zeros((N, T), dtype="<i4")
    print(f"capture regime: {'PER-PROMPT (batch=1)' if args.per_prompt else 'BATCHED'}")
    for k in range(K):
        by_prompt = generate_all(llm, sp, args.per_prompt)
        for i, p in enumerate(PROMPTS):
            o = by_prompt[p]
            ids = list(o.outputs[0].token_ids)
            if len(ids) < T:
                print(f"NOTE run{k} prompt[{i}] produced {len(ids)} tokens (< {T}); "
                      f"padding with -1", file=sys.stderr)
            for j in range(min(T, len(ids))):
                dist[i, j, k] = ids[j]
            if k == 0:
                for j in range(min(T, len(ids))):
                    run0[i, j] = ids[j]
                np.array(o.prompt_token_ids, dtype="<i4").tofile(
                    os.path.join(out_dir, f"p{i}_prompt.i32"))
                print(f"prompt[{i}] {p!r} -> {ids}  ({o.outputs[0].text!r})")

    np.save(os.path.join(out_dir, "greedy_ids.npy"), run0)
    np.save(os.path.join(out_dir, "greedy_dist.npy"), dist)

    # ---- determinism report -------------------------------------------------
    print(f"\n=== determinism report: {args.model}  N={N} T={T} K={K} ===")
    deterministic = True
    total_multi_pos = 0
    for i in range(N):
        # distinct full sequences across the K runs
        seqs = {tuple(int(x) for x in dist[i, :, k]) for k in range(K)}
        # per-position distinct-token count
        multi = [j for j in range(T) if len({int(dist[i, j, k]) for k in range(K)}) > 1]
        total_multi_pos += len(multi)
        if len(seqs) > 1:
            deterministic = False
            print(f"  prompt[{i}] NON-DET: {len(seqs)} distinct sequences; "
                  f"near-tie positions {multi}")
            for j in multi:
                observed = sorted({int(dist[i, j, k]) for k in range(K)})
                print(f"      pos {j:2d}: observed tokens {observed}")
        else:
            print(f"  prompt[{i}] deterministic (1 sequence over {K} runs)")
    print(f"=== {'ALL DETERMINISTIC' if deterministic else 'NON-DETERMINISTIC'} "
          f"over K={K}; {total_multi_pos} multi-member (prompt,pos) cells ===")
    print(f"wrote {out_dir}/greedy_ids.npy {run0.shape} + "
          f"{out_dir}/greedy_dist.npy {dist.shape}")


if __name__ == "__main__":
    main()
