#!/usr/bin/env python3
# GLM/DSA G1 (GLM-4.7-Flash, `Glm4MoeLiteForCausalLM` — the SECOND MLA model in
# this tree, and the first to exercise the q_lora query branch AND the `noaux_tc`
# grouped router end-to-end). Mirror of scripts/deepseek-v2-oracle-capture.py.
#
# Capture the vLLM 0.25.0 oracle greedy continuation on the prompt battery AND
# re-confirm vLLM's OWN self-determinism (K greedy runs), which SELECTS the gate
# form ([[near-tie-distributional-gate]]):
#   * vLLM deterministic over K runs -> the paged-engine gate holds US to STRICT
#     token-exact vs greedy_ids.npy. A 31.2B MoE is well above the small-dense
#     near-tie regime, so STRICT is the expectation.
#   * vLLM self-diverges at bf16 near-ties -> the strict-where-well-posed
#     near-tie-robust gate (glm4-moe-lite-neartie-gap.py supplies the band).
#
# Golden artifacts (tests/parity/goldens/glm4_moe_lite_greedy/):
#   greedy_ids.npy   [N,T]    i32 — run 0, the point golden (the STRICT bar).
#   greedy_dist.npy  [N,T,K]  i32 — per-(prompt,pos) SET across K greedy runs.
#   p{i}_prompt.i32  [Li]     i32 — vLLM's TOKENIZATION of prompt i (first-class:
#                                   a silently-wrong BOS scores 0/N while emitting
#                                   fluent text — the OPT lesson).
#
# MEMORY — READ THIS. GB10's ~119 GiB is UNIFIED memory: weights, the gmu
# reservation, profiling activations and the checkpoint page cache all draw on the
# SAME pool, and a large request has HARD-REBOOTED this box. GLM-4.7-Flash weights
# are ~58 GiB, so gmu must be high enough to hold them (~0.49 of 119 GiB) yet not
# so high it reboots the box (0.85 rebooted the 27B). Default 0.72 with a SMALL
# profiling batch (max_num_batched_tokens/max_num_seqs) and moe_backend=triton.
# vLLM auto-selects the FlashInfer CUTLASS unquantized MoE backend whose expert
# workspace OOM-reboots GB10 (the DeepSeek-V2 campaign hit it five times); `triton`
# is a first-class vLLM backend computing the SAME bf16 grouped MoE with a small
# footprint, and MUST be used consistently by capture and teacher-forcing.
#
# Run ALONE on dgx (never co-scheduled with another big model), oracle venv, PATH
# including ~/venvs/vllm-oracle/bin (FlashInfer JIT needs `ninja`):
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/glm4-moe-lite-oracle-capture.py --runs 5
import argparse
import os
import sys

import numpy as np

# MUST match tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp::Prompts() and
# scripts/glm4-moe-lite-neartie-gap.py::PROMPTS exactly (goldens + gate never drift).
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "Q: What is 17 * 23?\nA:",
    "The three laws of robotics are",
    "Once upon a time, in a land far away,",
    "The chemical symbol for gold is",
    "To be or not to be, that is",
]


def default_out_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "parity", "goldens", "glm4_moe_lite_greedy",
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",
                    default=os.environ.get("GLM47_MODEL", "zai-org/GLM-4.7-Flash"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--runs", type=int, default=int(os.environ.get("GLM47_RUNS", "5")))
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("GLM47_GPU_UTIL", "0.72")))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("GLM47_MAX_LEN", "2048")))
    ap.add_argument("--max-num-batched-tokens", type=int,
                    default=int(os.environ.get("GLM47_MAX_BATCHED", "2048")))
    ap.add_argument("--max-num-seqs", type=int,
                    default=int(os.environ.get("GLM47_MAX_SEQS", "4")))
    ap.add_argument("--moe-backend",
                    default=os.environ.get("GLM47_MOE_BACKEND", "triton"))
    return ap.parse_args()


def generate_per_prompt(llm, sp):
    """One prompt per batch=1 generate() call — the determinism-sensitive regime
    the paged-engine gate also runs in (batching re-orders reductions)."""
    return {p: llm.generate([p], sp)[0] for p in PROMPTS}


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    out_dir = args.out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    N = len(PROMPTS)
    T = args.max_tokens
    K = max(1, args.runs)

    llm_kwargs = dict(model=args.model, dtype="bfloat16", enforce_eager=True,
                      trust_remote_code=True,
                      gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len,
                      max_num_batched_tokens=args.max_num_batched_tokens,
                      max_num_seqs=args.max_num_seqs)
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
    print(f"moe_backend={args.moe_backend or 'auto'}")
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=T)

    dist = np.full((N, T, K), -1, dtype="<i4")
    run0 = np.zeros((N, T), dtype="<i4")
    print(f"model={args.model}  N={N} T={T} K={K}  regime=PER-PROMPT(batch=1)  "
          f"gpu_memory_utilization={args.gpu_mem_util}")
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
                print(f"prompt[{i}] {p!r}")
                print(f"    prompt_ids={list(o.prompt_token_ids)}")
                print(f"    -> {ids}  ({o.outputs[0].text!r})")

    np.save(os.path.join(out_dir, "greedy_ids.npy"), run0)
    np.save(os.path.join(out_dir, "greedy_dist.npy"), dist)

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
        else "NON-DETERMINISTIC -> re-derive the gate (do NOT assume the band)"
    print(f"=== {verdict}  (K={K}; {total_multi_pos} multi-member (prompt,pos) cells) ===")
    print(f"wrote {out_dir}/greedy_ids.npy {run0.shape} + "
          f"{out_dir}/greedy_dist.npy {dist.shape}")


if __name__ == "__main__":
    main()
