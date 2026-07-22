#!/usr/bin/env python3
# MLA CAMPAIGN W8 (DeepSeek-V2-Lite, `DeepseekV2ForCausalLM` — the FIRST MLA model
# in this tree) — capture the vLLM 0.25.0 oracle greedy continuation on the
# DeepSeek prompt battery AND re-confirm vLLM's OWN self-determinism (K greedy
# runs), which SELECTS the gate (see [[near-tie-distributional-gate]]):
#
#   * vLLM self-consistent (deterministic) over K runs -> the paged-engine gate
#     holds US to STRICT token-exact vs greedy_ids.npy (the real bar).
#   * vLLM self-DIVERGES at bf16 near-ties -> the strict-where-well-posed
#     near-tie-robust gate (deepseek-v2-neartie-gap.py supplies the band).
#
# W0 ALREADY MEASURED THIS, and the answer is STRICT. Campaign spike
# `.agents/specs/mla-deepseek-campaign.md` §"Work breakdown" W0(e): at **batch=1**
# — the regime the SACRED gate actually runs in — vLLM's own greedy is
# DETERMINISTIC on **8/8** prompts (`ALL_PROMPTS_DETERMINISTIC: True`,
# `~/scratch_mla_w1/w0_probe_b1.{py,log,json}` on dgx, K=5, 32 tokens). A FIRST
# probe that generated all 8 prompts in ONE BATCH reported false self-inconsistency
# on 3/8 — that is the known BATCHING artifact (the same one the Qwen3-dense razor
# hit; batched generation re-orders reductions), NOT non-determinism. This script
# therefore generates **one prompt per generate() call** (`generate_per_prompt`),
# exactly like every other oracle capture in this tree, and re-confirms the K-run
# determinism into the committed golden so the gate can re-assert it at run time.
# Do NOT re-derive a weaker gate from a batched probe.
#
# Golden artifacts (mirrors scripts/opt-oracle-capture.py):
#   greedy_ids.npy   [N, T]     i32 — run 0, the point golden (the STRICT bar).
#   greedy_dist.npy  [N, T, K]  i32 — the per-(prompt,pos) SET vLLM emits across
#                                     K greedy runs (-1 right-pads short runs).
#   p{i}_prompt.i32  [Li]       i32 — vLLM's TOKENIZATION of prompt i. This is a
#                                     first-class golden, not a nicety: on OPT a
#                                     silently-unapplied BOS scored 0/6 while
#                                     emitting fluent English, and the committed
#                                     tokenization golden is what caught it in one
#                                     run. DeepSeek ships its own tokenizer
#                                     (LlamaTokenizerFast over a 100k BPE vocab
#                                     with `add_bos_token: true`, bos=100000), so
#                                     the same class of bug is live here.
#
# MEMORY (GB10 is UNIFIED memory — a large request is HOST RAM and has rebooted
# the box): `gpu_memory_utilization=0.40` is MANDATORY for the oracle here, and is
# the same value W0's determinism probe used, so the capture regime matches the
# measurement that selected the gate.
#
# The PROMPTS list MUST match tests/vllm/models/test_deepseek_v2_paged_engine.cpp
# ::Prompts() exactly (goldens and gate never drift), and is the SAME battery W0
# proved deterministic. Run on dgx with the oracle venv; PATH must include
# ~/venvs/vllm-oracle/bin (the FlashInfer JIT needs `ninja` on PATH or engine-core
# init dies in determine_available_memory):
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/deepseek-v2-oracle-capture.py \
#       --runs 5 --out-dir tests/parity/goldens/deepseek_v2_greedy
import argparse
import os
import sys

import numpy as np

# The compact battery W0 proved deterministic at batch=1 (8/8, K=5). Kept
# byte-identical here, in the gate, and in the teacher-forcing script.
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
        "tests", "parity", "goldens", "deepseek_v2_greedy",
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",
                    default=os.environ.get("DSV2_MODEL", "deepseek-ai/DeepSeek-V2-Lite"))
    ap.add_argument("--out-dir", default=None,
                    help="golden dir (default tests/parity/goldens/deepseek_v2_greedy)")
    ap.add_argument("--runs", type=int, default=int(os.environ.get("DSV2_RUNS", "5")),
                    help="K greedy runs to build the observed distribution (K>=5)")
    ap.add_argument("--max-tokens", type=int, default=16)
    # 0.40 is NOT tunable upward here: GB10's 119 GiB is UNIFIED memory, so a
    # larger request is host RAM. See the module docstring.
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("DSV2_GPU_UTIL", "0.40")))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("DSV2_MAX_LEN", "2048")))
    # MEASURED (W8, 2026-07-22): the ENGINE-INIT MEMORY PROFILING forward — not
    # generation — is what kills this box. Two capture attempts at
    # max_model_len=4096 with vLLM's default token budget REBOOTED dgx at exactly
    # the same point (right after model load / sampler init, inside
    # determine_available_memory), once with a build running alongside and once
    # completely alone. GB10's 119 GiB is UNIFIED, so the 29.35 GiB of weights,
    # the 0.40 reservation, the profiling activations and the 30 GB checkpoint
    # page cache all draw on the SAME pool. Capping the profiling batch is the
    # lever that does not touch the mandated gpu_memory_utilization=0.40; the
    # battery's longest prompt is ~20 tokens generating 16, so 2048 is ample and
    # nothing about the greedy result depends on it (DeepSeek's YaRN scaling
    # reads `original_max_position_embeddings`/`factor` from config.json, never
    # from max_model_len).
    ap.add_argument("--max-num-batched-tokens", type=int,
                    default=int(os.environ.get("DSV2_MAX_BATCHED", "2048")))
    ap.add_argument("--max-num-seqs", type=int,
                    default=int(os.environ.get("DSV2_MAX_SEQS", "4")))
    # …and capping the profiling batch turned out NOT to be the lever. A THIRD
    # attempt, alone on a freshly-rebooted box with an empty page cache and
    # max_num_batched_tokens=2048, died at the SAME point. The actual cause is
    # the one the Qwen3-Coder bring-up already documented: vLLM auto-selects the
    # **FlashInfer CUTLASS unquantized MoE** backend (the startup log prints it,
    # out of ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON',
    # 'BATCHED_TRITON']), whose expert workspace allocation is what OOM-kills a
    # large MoE on GB10's unified pool. `triton` is a FIRST-CLASS vLLM backend
    # computing the SAME bf16 grouped MoE with a small footprint — the exact
    # same legitimate-oracle substitution scripts/qwen3coder-*.py make, and it
    # must be used CONSISTENTLY by the capture and any teacher-forcing run so
    # both arms see one oracle configuration. Recorded as a deviation in the
    # campaign spike.
    ap.add_argument("--moe-backend", default=os.environ.get("DSV2_MOE_BACKEND",
                                                            "triton"))
    return ap.parse_args()


def generate_per_prompt(llm, sp):
    """Return {prompt: RequestOutput} decoding each prompt in its OWN batch=1
    generate() call. This is THE determinism-sensitive detail: batching all
    prompts into one call re-orders reductions and manufactures false
    self-inconsistency (W0 hit exactly that on 3/8 prompts). batch=1 is also the
    regime the paged-engine gate runs in."""
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
        else "NON-DETERMINISTIC -> re-derive the gate (do NOT assume the band)"
    print(f"=== {verdict}  (K={K}; {total_multi_pos} multi-member (prompt,pos) cells) ===")
    print(f"wrote {out_dir}/greedy_ids.npy {run0.shape} + "
          f"{out_dir}/greedy_dist.npy {dist.shape}")


if __name__ == "__main__":
    main()
