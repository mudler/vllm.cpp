#!/usr/bin/env python3
# ADDITIVE-MODEL W4 (Qwen3-Coder-30B-A3B, Qwen3MoeForCausalLM) — capture the
# vLLM 0.25.0 oracle greedy continuation on the Qwen3-Coder prompt battery AND
# measure vLLM's OWN self-determinism (K greedy runs), which SELECTS the gate
# (see [[near-tie-distributional-gate]]):
#
#   * vLLM self-consistent (deterministic) over K runs  -> the paged-engine gate
#     holds US to STRICT token-exact vs greedy_ids.npy (the real bar; an A3B MoE
#     is expected deterministic like the 35B, which passes strict 315/315).
#   * vLLM self-DIVERGES at bf16 near-ties -> the DISTRIBUTIONAL gate: OUR token
#     at each (prompt,pos) must be a MEMBER of vLLM's observed K-run set at that
#     cell (greedy_dist.npy). The determinism report below documents which.
#
# Two golden artifacts (mirrors scripts/qwen3-oracle-capture.py):
#   greedy_ids.npy   [N, T]     i32 — run 0, the point golden (STRICT bar).
#   greedy_dist.npy  [N, T, K]  i32 — the per-(prompt,pos) SET vLLM emits across
#                                     K greedy runs (-1 right-pads short runs).
#   p{i}_prompt.i32  [Li]       i32 — vLLM's tokenization of prompt i (the paged
#                                     engine's own tokenization is cross-checked).
#
# The PROMPTS list MUST match tests/vllm/models/test_qwen3coder_paged_engine.cpp
# ::Prompts() exactly (goldens and gate never drift). Run on dgx with the oracle
# venv; PATH must include ~/venvs/vllm-oracle/bin (flashinfer's ninja JIT):
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/qwen3coder-oracle-capture.py \
#       --model Qwen/Qwen3-Coder-30B-A3B-Instruct --runs 5 \
#       --out-dir tests/parity/goldens/qwen3coder_greedy
import argparse
import os
import sys
import numpy as np

# Small general/code-flavored battery — bounded decode over the SLOW bf16
# per-expert reference loop is fine (correctness is speed-independent). The set
# is deliberately compact (the A3B MoE reference path is slow on our side).
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time,",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "import numpy as np",
]


def default_out_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "parity", "goldens", "qwen3coder_greedy",
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get(
        "QWEN3CODER_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct"))
    ap.add_argument("--out-dir", default=None,
                    help="golden dir (default tests/parity/goldens/qwen3coder_greedy)")
    ap.add_argument("--runs", type=int,
                    default=int(os.environ.get("QWEN3CODER_RUNS", "5")),
                    help="K greedy runs to build the observed distribution (K>=5)")
    ap.add_argument("--max-tokens", type=int, default=16)
    # Conservative memory settings for the 30B bf16 model on GB10's 128 GiB
    # UNIFIED memory: ~57 GiB weights + the ~57 GiB checkpoint page-cache share
    # the same pool, so vLLM's default 0.9 KV-util OOM-kills at profiling. A low
    # util + small max_model_len caps KV (the capture needs only a few short
    # sequences) and leaves headroom for the reclaimable page cache.
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("QWEN3CODER_GPU_UTIL", "0.55")))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("QWEN3CODER_MAX_LEN", "8192")))
    # vLLM's V1 profiling dummy-run allocates activations for max_num_batched_tokens
    # tokens; on the 128-expert MoE that spike (on top of 57 GiB weights + the
    # checkpoint page-cache) OOM-kills the process. A small batch bounds it — the
    # capture only ever runs 1 short sequence at a time.
    ap.add_argument("--max-num-batched-tokens", type=int,
                    default=int(os.environ.get("QWEN3CODER_MAX_BATCHED", "1024")))
    ap.add_argument("--max-num-seqs", type=int,
                    default=int(os.environ.get("QWEN3CODER_MAX_SEQS", "4")))
    # vLLM unquantized-MoE backend. The auto-selected FlashInfer CUTLASS backend
    # allocates a very large workspace for 128 experts that exceeds GB10's memory
    # headroom (57 GiB weights on a 119 GiB unified pool) and OOM-kills at the
    # first forward. "triton" is a first-class vLLM backend computing the SAME bf16
    # grouped MoE with a small footprint — a legitimate 0.25.0 oracle. "" keeps
    # vLLM's auto selection.
    ap.add_argument("--moe-backend",
                    default=os.environ.get("QWEN3CODER_MOE_BACKEND", "triton"))
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

    llm_kwargs = dict(model=args.model, dtype="bfloat16", enforce_eager=True,
                      gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len,
                      max_num_batched_tokens=args.max_num_batched_tokens,
                      max_num_seqs=args.max_num_seqs)
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
    print(f"moe_backend={args.moe_backend or 'auto'}")
    llm = LLM(**llm_kwargs)
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
        else "NON-DETERMINISTIC -> DISTRIBUTIONAL (set-membership) gate"
    print(f"=== {verdict}  (K={K}; {total_multi_pos} multi-member (prompt,pos) cells) ===")
    print(f"wrote {out_dir}/greedy_ids.npy {run0.shape} + "
          f"{out_dir}/greedy_dist.npy {dist.shape}")


if __name__ == "__main__":
    main()
