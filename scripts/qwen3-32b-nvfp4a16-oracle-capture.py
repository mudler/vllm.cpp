#!/usr/bin/env python3
# ADDITIVE-QUANT W4 (Qwen3-32B-NVFP4A16, `Qwen3ForCausalLM` + compressed-tensors
# NVFP4A16 / W4A16 — the QUANT-SCHEME additivity proof on an already-done dense
# forward) — capture the vLLM 0.25.0 oracle greedy continuation on the prompt
# battery AND measure vLLM's OWN self-determinism (K greedy runs), which SELECTS
# the gate (see [[near-tie-distributional-gate]]):
#
#   * vLLM self-consistent (deterministic) over K runs -> the paged-engine gate
#     holds US to STRICT token-exact vs greedy_ids.npy (the real bar; a 64L
#     32B dense is expected deterministic, as OPT-125m and Qwen3-Coder were).
#   * vLLM self-DIVERGES at bf16 near-ties -> the strict-where-well-posed
#     near-tie-robust gate over greedy_dist.npy set membership.
#
# Golden artifacts (mirrors scripts/qwen3coder-oracle-capture.py):
#   greedy_ids.npy   [N, T]     i32 — run 0, the point golden (STRICT bar).
#   greedy_dist.npy  [N, T, K]  i32 — the per-(prompt,pos) SET vLLM emits across
#                                     K greedy runs (-1 right-pads short runs).
#   p{i}_prompt.i32  [Li]       i32 — vLLM's tokenization of prompt i. COMMITTED
#                                     as a golden: on OPT a BOS bug scored 0/6
#                                     while emitting fluent English, and the
#                                     committed tokenization golden is what
#                                     caught it in one run.
#
# CHECKPOINT: RedHatAI/Qwen3-32B-NVFP4A16 — 64L dense, hidden 5120, 64 Q heads /
# 8 KV heads, head_dim 128, intermediate 25600, vocab 151936. Its
# quantization_config is compressed-tensors `nvfp4-pack-quantized`, group_size
# 16, num_bits 4, type float, strategy tensor_group, input_activations null
# (=> W4A16: NVFP4 weights, BF16 activations), ignore ["lm_head"].
#
# The PROMPTS list MUST match
# tests/vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp::Prompts() exactly
# (goldens and gate never drift). Run on dgx with the oracle venv; PATH must
# include ~/venvs/vllm-oracle/bin (flashinfer's ninja JIT):
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/qwen3-32b-nvfp4a16-oracle-capture.py \
#       --model RedHatAI/Qwen3-32B-NVFP4A16 --runs 5 \
#       --out-dir tests/parity/goldens/qwen3_32b_nvfp4a16_greedy
import argparse
import os
import sys

import numpy as np

# Small general battery — the SAME stems as the Qwen3-Coder row where they
# overlap, so cross-model divergences are comparable. Plain completion stems (no
# chat template): the gate exercises the base LM continuation path.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time,",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "In 1969, humans first walked on",
]


def default_out_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "parity", "goldens", "qwen3_32b_nvfp4a16_greedy",
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get(
        "QWEN3_32B_NVFP4A16_MODEL", "RedHatAI/Qwen3-32B-NVFP4A16"))
    ap.add_argument("--out-dir", default=None,
                    help="golden dir (default tests/parity/goldens/"
                         "qwen3_32b_nvfp4a16_greedy)")
    ap.add_argument("--runs", type=int,
                    default=int(os.environ.get("QWEN3_32B_NVFP4A16_RUNS", "5")),
                    help="K greedy runs to build the observed distribution (K>=5)")
    ap.add_argument("--max-tokens", type=int, default=16)
    # ~17 GiB of NVFP4 weights + the checkpoint page-cache share GB10's 128 GiB
    # UNIFIED pool; a modest util + small max_model_len caps KV (the capture runs
    # only a few short sequences) and leaves reclaimable headroom.
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("QWEN3_32B_NVFP4A16_GPU_UTIL", "0.55")))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("QWEN3_32B_NVFP4A16_MAX_LEN", "8192")))
    ap.add_argument("--max-num-batched-tokens", type=int,
                    default=int(os.environ.get("QWEN3_32B_NVFP4A16_MAX_BATCHED", "1024")))
    ap.add_argument("--max-num-seqs", type=int,
                    default=int(os.environ.get("QWEN3_32B_NVFP4A16_MAX_SEQS", "4")))
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
              max_model_len=args.max_model_len,
              max_num_batched_tokens=args.max_num_batched_tokens,
              max_num_seqs=args.max_num_seqs)
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
                print(f"prompt[{i}] {p!r} tok={list(o.prompt_token_ids)} -> {ids}  "
                      f"({o.outputs[0].text!r})")

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
