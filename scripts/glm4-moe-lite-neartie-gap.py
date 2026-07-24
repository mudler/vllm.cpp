#!/usr/bin/env python3
# GLM/DSA G1 (GLM-4.7-Flash, `Glm4MoeLiteForCausalLM`) — the TEACHER-FORCED
# DIVERGENCE DIAGNOSTIC. Mirror of scripts/deepseek-v2-neartie-gap.py.
#
# The gate is STRICT TOKEN-EXACT wherever vLLM is deterministic. This is NOT a way
# to soften it: it is the ratified diagnostic run WHEN our engine diverges from a
# deterministic vLLM — teacher-force vLLM on OUR exact generated sequence and ask,
# per position, how many nats vLLM's OWN argmax beats OUR token by, GIVEN OUR
# PREFIX. gap 0 = our token IS vLLM's argmax (downstream of an earlier bf16 tie);
# gap <= 0.5 nats = a bf16 near-tie vLLM cannot separate; gap large / our token
# outside vLLM's top-K = a REAL forward divergence (fix the model, do not widen a
# band).
#
# Emits into --golden-dir:
#   our_ids.npy           [N,T] i32 — OUR engine's exact greedy tokens (anchor).
#   neartie_gap_mnats.npy [N,T] i32 — vLLM's teacher-forced gap in MILLI-nats.
# Input: our_ids.i32 in --golden-dir ([N,T] i32 raw dump of our engine's tokens).
#
# MEMORY: same UNIFIED-memory constraints and moe_backend=triton substitution as
# the capture (both arms MUST see one oracle configuration). Run ALONE on dgx,
# oracle venv, PATH including ~/venvs/vllm-oracle/bin.
import argparse
import os

import numpy as np

# MUST match test_glm4_moe_lite_paged_engine.cpp::Prompts() and the capture.
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
OUTSIDE_TOPK_MNATS = 99_999_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.72)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--moe-backend", default="triton")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    N, T = len(PROMPTS), args.max_tokens
    our = np.fromfile(os.path.join(args.golden_dir, "our_ids.i32"),
                      dtype="<i4").reshape(N, T)
    greedy = np.load(os.path.join(args.golden_dir, "greedy_ids.npy"))

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

    gap_mnats = np.zeros((N, T), dtype="<i4")
    print(f"=== teacher-forced gap: {args.model} (vLLM's logits on OUR prefix) ===")
    max_gap = 0.0
    worst = None
    n_div = 0
    for i in range(N):
        prompt_ids = np.fromfile(
            os.path.join(args.golden_dir, f"p{i}_prompt.i32"), dtype="<i4").tolist()
        our_ids = [int(x) for x in our[i]]
        full = list(prompt_ids) + our_ids
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk)
        out = llm.generate({"prompt_token_ids": full}, sp)[0]
        plp = out.prompt_logprobs
        P = len(prompt_ids)
        for j in range(T):
            pos = P + j
            d = plp[pos] or {}
            arg_tid = max(d, key=lambda k: d[k].logprob) if d else -1
            arg_lp = d[arg_tid].logprob if d else 0.0
            our_tid = our_ids[j]
            if our_tid in d:
                gap = max(0.0, arg_lp - d[our_tid].logprob)
                gap_mnats[i, j] = int(round(gap * 1000.0))
                if gap > max_gap:
                    max_gap, worst = gap, (i, j, gap)
            else:
                gap_mnats[i, j] = OUTSIDE_TOPK_MNATS
                print(f"  p{i:2d} tok{j:2d}: OUR TOKEN {our_tid} OUTSIDE vLLM "
                      f"top-{args.topk} (REAL divergence)")
            if our_ids[j] != int(greedy[i, j]):
                n_div += 1
                print(f"  p{i:2d} tok{j:2d}: our={our_tid} "
                      f"vLLM_greedy={int(greedy[i,j])} vLLM_argmax={arg_tid} "
                      f"gap={gap_mnats[i,j]/1000.0:.4f} nats")

    np.save(os.path.join(args.golden_dir, "our_ids.npy"), our)
    np.save(os.path.join(args.golden_dir, "neartie_gap_mnats.npy"), gap_mnats)
    print(f"=== {n_div} token-divergent positions vs vLLM greedy; "
          f"max teacher-forced gap {max_gap:.4f} nats (worst {worst}) ===")
    print(f"wrote {args.golden_dir}/our_ids.npy + neartie_gap_mnats.npy "
          f"{gap_mnats.shape}")


if __name__ == "__main__":
    main()
