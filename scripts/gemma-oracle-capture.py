#!/usr/bin/env python3
# Sweep W4/W5 (Gemma-2 `Gemma2ForCausalLM`, Gemma-1 `GemmaForCausalLM`) — generalized
# oracle-capture: run the vLLM 0.25.0 greedy continuation on the shared Gemma prompt
# battery, measure vLLM's OWN per-prompt (batch=1) self-determinism over K runs (which
# SELECTS the gate form, [[near-tie-distributional-gate]]), VERIFY BOS handling vs the
# oracle, and print an embeddable C++ table for the tokenizer-free SACRED MODEL gate
# (tests/vllm/models/test_gemma{2,}_forward.cpp) between --tag markers.
#
#   * vLLM deterministic per-prompt over K runs -> STRICT token-exact gate (singleton
#     sets). * vLLM self-diverges at a bf16 near-tie -> that (prompt,pos) becomes a
#     multi-member SET; our argmax must be a MEMBER (near-tie-robust). The report says
#     which. Gemma prepends BOS (bos_token_id=2); this asserts prompt_token_ids[0]==2
#     for every prompt (the OPT lesson). The C++ table embeds vLLM's EXACT prompt token
#     ids so the gate is tokenizer-free.
#
# Run on dgx with the oracle venv, ALONE, under flock $HOME/gpu.lock:
#   HF_TOKEN=... HF_HUB_DISABLE_XET=1 \
#     ~/venvs/vllm-oracle/bin/python scripts/gemma-oracle-capture.py \
#       --model google/gemma-2-2b-it --tag GEMMA2 --runs 5 --max-tokens 8
import argparse
import os
import sys

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "The chemical symbol for gold is",
    "The first president of the United States was",
    "Roses are red, violets are",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="marker prefix, e.g. GEMMA2 or GEMMA1")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=2048)
    return ap.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    N, T, K = len(PROMPTS), args.max_tokens, max(1, args.runs)
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=T)

    dist = [[set() for _ in range(T)] for _ in range(N)]
    prompt_ids = [None] * N
    for k in range(K):
        for i, p in enumerate(PROMPTS):
            o = llm.generate([p], sp)[0]
            ids = list(o.outputs[0].token_ids)
            if k == 0:
                prompt_ids[i] = list(o.prompt_token_ids)
            for j in range(min(T, len(ids))):
                dist[i][j].add(int(ids[j]))

    print("=== BOS verification (Gemma bos_token_id=2) ===")
    bos_ok = True
    for i, p in enumerate(PROMPTS):
        first = prompt_ids[i][0] if prompt_ids[i] else None
        ok = first == 2
        bos_ok = bos_ok and ok
        print(f"  p{i} {p!r}: prompt_ids[0]={first} {'OK' if ok else 'FAIL (expected BOS 2)'}")
    print(f"BOS: {'ALL PREPEND BOS 2' if bos_ok else 'MISMATCH — investigate before gating'}")

    print(f"\n=== vLLM SELF-DETERMINISM: {args.model}  N={N} T={T} K={K} (per-prompt batch=1) ===")
    multi = 0
    for i in range(N):
        cells = [j for j in range(T) if len(dist[i][j]) > 1]
        multi += len(cells)
        if cells:
            print(f"  p{i} near-tie cells {cells}: " +
                  ", ".join(f"pos{j}{sorted(dist[i][j])}" for j in cells))
        else:
            print(f"  p{i} deterministic ({T} singleton cells)")
    verdict = ("ALL DETERMINISTIC -> STRICT token-exact gate" if multi == 0
               else f"{multi} multi-member cells -> near-tie-robust set-membership gate")
    print(f"=== {verdict} ===\n")

    print(f"// {args.tag}_ORACLE_GREEDY_BEGIN (vLLM 0.25.0 greedy, " + args.model + ")")
    for i, p in enumerate(PROMPTS):
        pr = ", ".join(str(x) for x in prompt_ids[i])
        conts = []
        for j in range(T):
            s = sorted(dist[i][j])
            conts.append("{" + ", ".join(str(x) for x in s) + "}")
        print("    {{" + pr + "},  // " + repr(p))
        print("     {" + ", ".join(conts) + "}},")
    print(f"// {args.tag}_ORACLE_GREEDY_END")
    if not bos_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
