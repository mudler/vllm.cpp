#!/usr/bin/env python3
# ADDITIVE-MODEL W4 — the near-tie-robust CORRECTNESS golden (see
# [[near-tie-distributional-gate]]).
#
# vLLM 0.25.0 greedy is DETERMINISTIC per-prompt (batch=1), but its one-shot
# PREFILL argmax disagrees with its incremental DECODE at bf16 near-ties, and two
# independent bf16 decoders (ours vs vLLM) may resolve a near-tie either way. The
# honest "mirror vLLM" bar is therefore: is OUR token, given OUR exact prefix, one
# that vLLM's OWN logits cannot confidently separate from vLLM's argmax? This
# script TEACHER-FORCES vLLM on OUR engine's exact generated sequence (our_ids.i32,
# dumped by the gate with VT_DUMP_IDS=1) and records, per position, the gap in
# nats between vLLM's argmax logprob and OUR token's logprob. A tiny gap
# (<=~0.5 nats) is a bf16 near-tie (structurally correct); a large gap (or our
# token outside vLLM's top-K) is a REAL forward divergence the gate must fail on.
#
# Emits, into --golden-dir:
#   our_ids.npy           [N, T] i32  — OUR engine's exact greedy tokens (anchor).
#   neartie_gap_mnats.npy [N, T] i32  — vLLM's teacher-forced gap in MILLI-nats for
#                                       OUR token (0 = our token IS vLLM's argmax;
#                                       99_999_000 = our token outside vLLM's top-K
#                                       => real divergence, gate fails).
# Run on dgx with the oracle venv (PATH incl. ~/venvs/vllm-oracle/bin):
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH ~/venvs/vllm-oracle/bin/python \
#     scripts/qwen3-neartie-gap.py --model Qwen/Qwen3-4B \
#       --golden-dir tests/parity/goldens/qwen3_greedy_4b
import argparse, os, sys
import numpy as np

PROMPTS = [
    "The capital of France is", "Once upon a time,", "In the beginning God created",
    "The quick brown fox jumps over", "def fibonacci(n):",
    "Water boils at a temperature of", "The theory of relativity was developed by",
    "To be or not to be, that is", "The largest planet in our solar system is",
    "Machine learning is a subfield of", "The mitochondria is the powerhouse of",
    "Roses are red, violets are", "The first president of the United States was",
    "E equals m c", "A journey of a thousand miles begins with",
    "The chemical symbol for gold is",
]
OUTSIDE_TOPK_MNATS = 99_999_000  # our token not even in vLLM's top-K => real bug

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    N, T = len(PROMPTS), args.max_tokens
    our = np.fromfile(os.path.join(args.golden_dir, "our_ids.i32"),
                      dtype="<i4").reshape(N, T)
    greedy = np.load(os.path.join(args.golden_dir, "greedy_ids.npy"))

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True)

    gap_mnats = np.zeros((N, T), dtype="<i4")
    print(f"=== teacher-forced near-tie gap: {args.model} (OUR prefix) ===")
    max_gap = 0.0
    worst = None
    n_div = 0
    for i in range(N):
        # Use the EXACT prompt tokenization the oracle greedy capture used
        # (p{i}_prompt.i32) so the teacher-forced prefix aligns bit-for-bit.
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
                print(f"  p{i:2d} tok{j:2d}: OUR TOKEN {our_tid} OUTSIDE vLLM top-{args.topk}"
                      f" (REAL divergence)")
            if our_ids[j] != int(greedy[i, j]):
                n_div += 1
                print(f"  p{i:2d} tok{j:2d}: our={our_tid} vLLM_greedy={int(greedy[i,j])}"
                      f" vLLM_argmax={arg_tid} gap={gap_mnats[i,j]/1000.0:.4f} nats")

    np.save(os.path.join(args.golden_dir, "our_ids.npy"), our)
    np.save(os.path.join(args.golden_dir, "neartie_gap_mnats.npy"), gap_mnats)
    print(f"=== {n_div} token-divergent positions vs vLLM greedy; "
          f"max near-tie gap {max_gap:.4f} nats (worst {worst}) ===")
    print(f"wrote {args.golden_dir}/our_ids.npy + neartie_gap_mnats.npy {gap_mnats.shape}")

if __name__ == "__main__":
    main()
