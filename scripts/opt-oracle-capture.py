#!/usr/bin/env python3
# ADDITIVE-MODEL W4 (OPT-125m, `OPTForCausalLM` — the CROSS-FAMILY additivity
# canary) — capture the vLLM 0.25.0 oracle greedy continuation on the OPT prompt
# battery AND measure vLLM's OWN self-determinism (K greedy runs), which SELECTS
# the gate (see [[near-tie-distributional-gate]]):
#
#   * vLLM self-consistent (deterministic) over K runs -> the paged-engine gate
#     holds US to STRICT token-exact vs greedy_ids.npy (the real bar).
#   * vLLM self-DIVERGES at bf16 near-ties -> the strict-where-well-posed
#     near-tie-robust gate (opt-neartie-gap.py supplies the teacher-forced band).
#
# Golden artifacts (mirrors scripts/qwen3coder-oracle-capture.py):
#   greedy_ids.npy   [N, T]     i32 — run 0, the point golden (STRICT bar).
#   greedy_dist.npy  [N, T, K]  i32 — the per-(prompt,pos) SET vLLM emits across
#                                     K greedy runs (-1 right-pads short runs).
#   p{i}_prompt.i32  [Li]       i32 — vLLM's tokenization of prompt i.
#
# DTYPE (spec .agents/specs/sweep-opt-125m.md D1): facebook/opt-125m ships
# `torch_dtype: float16`, but vllm.cpp's CUDA compute path is BF16/F32 (kF16 is
# unimplemented outside cuda_gdn.cu). `--dtype bfloat16` is a first-class vLLM
# production mode, so BOTH arms run bf16 — an apples-to-apples oracle. Our
# checkpoint is materialized bf16 by scripts/opt-materialize-checkpoint.py with
# the SAME single fp16->bf16 rounding vLLM applies at load.
#
# The PROMPTS list MUST match tests/vllm/models/test_opt_paged_engine.cpp
# ::Prompts() exactly (goldens and gate never drift). Run on dgx with the oracle
# venv; PATH must include ~/venvs/vllm-oracle/bin:
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/opt-oracle-capture.py \
#       --model facebook/opt-125m --runs 5 \
#       --out-dir tests/parity/goldens/opt_greedy
import argparse
import os
import sys

import numpy as np

# Small general battery. OPT-125m is a 2022 base LM with no chat template, so the
# prompts are plain completion stems (no instruction phrasing).
PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "In 1969, humans first walked on",
    "Water boils at a temperature of",
]


def default_out_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "parity", "goldens", "opt_greedy",
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("OPT_MODEL", "facebook/opt-125m"))
    ap.add_argument("--out-dir", default=None,
                    help="golden dir (default tests/parity/goldens/opt_greedy)")
    ap.add_argument("--runs", type=int, default=int(os.environ.get("OPT_RUNS", "5")),
                    help="K greedy runs to build the observed distribution (K>=5)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("OPT_GPU_UTIL", "0.20")))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("OPT_MAX_LEN", "2048")))
    return ap.parse_args()


def generate_per_prompt(llm, sp):
    """Return {prompt: RequestOutput} decoding each prompt in its OWN batch=1
    generate() call — matches the paged-engine gate's single-request regime."""
    return {p: llm.generate([p], sp)[0] for p in PROMPTS}


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    out_dir = args.out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    N = len(PROMPTS)
    T = args.max_tokens
    K = max(1, args.runs)

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=T)

    dist = np.full((N, T, K), -1, dtype="<i4")
    run0 = np.zeros((N, T), dtype="<i4")
    print(f"model={args.model}  N={N} T={T} K={K}  regime=PER-PROMPT(batch=1)")
    for k in range(K):
        by_prompt = generate_per_prompt(llm, sp)
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

    # ---- determinism report (SELECTS the gate) ------------------------------
    print(f"\n=== vLLM SELF-DETERMINISM report: {args.model}  N={N} T={T} K={K} ===")
    deterministic = True
    total_multi_pos = 0
    for i in range(N):
        seqs = {tuple(int(x) for x in dist[i, :, k]) for k in range(K)}
        multi = [j for j in range(T)
                 if len({int(dist[i, j, k]) for k in range(K)}) > 1]
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
    verdict = "ALL DETERMINISTIC -> STRICT token-exact gate" if deterministic \
        else "NON-DETERMINISTIC -> near-tie-robust gate"
    print(f"=== {verdict}  (K={K}; {total_multi_pos} multi-member (prompt,pos) cells) ===")
    print(f"wrote {out_dir}/greedy_ids.npy {run0.shape} + "
          f"{out_dir}/greedy_dist.npy {dist.shape}")


if __name__ == "__main__":
    main()
