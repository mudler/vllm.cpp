#!/usr/bin/env python3
# ADDITIVE-MODEL W4 (Qwen3-Coder-30B-A3B) — the near-tie-robust CORRECTNESS golden
# (see [[near-tie-distributional-gate]]); mirror of scripts/qwen3-neartie-gap.py.
#
# vLLM 0.25.0 greedy on Qwen3-Coder is DETERMINISTIC per-prompt (the capture
# script's K=5 report shows 0 multi-valued cells), but at bf16 near-ties two
# independent bf16 decoders (ours vs vLLM) can resolve a next-token tie either way
# and vLLM's own one-shot PREFILL argmax disagrees with its incremental DECODE. The
# honest "mirror vLLM" bar is therefore: is OUR token, GIVEN OUR EXACT PREFIX, one
# that vLLM's OWN logits cannot confidently separate from vLLM's argmax? This script
# TEACHER-FORCES vLLM on OUR engine's exact generated sequence (our_ids.i32, dumped
# by the gate under VT_DUMP_IDS=1) and records, per position, the gap in nats
# between vLLM's argmax logprob and OUR token's logprob. A tiny gap (<=0.5 nats) is
# a bf16 near-tie (structurally correct); a large gap (or our token outside vLLM's
# top-K) is a REAL forward divergence the gate must fail on.
#
# Emits, into --golden-dir:
#   our_ids.npy           [N,T] i32 — OUR engine's exact greedy tokens (anchor).
#   neartie_gap_mnats.npy [N,T] i32 — vLLM's teacher-forced gap in MILLI-nats for OUR
#                                     token (0 = our token IS vLLM's argmax;
#                                     99_999_000 = our token outside vLLM's top-K).
#
# vLLM MoE backend + memory: the auto-selected FlashInfer CUTLASS unquantized-MoE
# backend allocates a workspace for 128 experts that OOM-kills the 57 GiB model on
# GB10's 119 GiB unified pool. "triton" is a first-class vLLM backend computing the
# SAME bf16 grouped MoE with a small footprint — the same legitimate 0.25.0 oracle
# the capture script used. Run on dgx with the oracle venv (PATH incl.
# ~/venvs/vllm-oracle/bin), ideally with scripts/... evict.py evicting the
# checkpoint page-cache alongside:
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH ~/venvs/vllm-oracle/bin/python \
#     scripts/qwen3coder-neartie-gap.py --golden-dir tests/parity/goldens/qwen3coder_greedy
import argparse, os
import numpy as np

# MUST match tests/vllm/models/test_qwen3coder_paged_engine.cpp::Prompts() and
# scripts/qwen3coder-oracle-capture.py::PROMPTS exactly.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time,",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "import numpy as np",
]
OUTSIDE_TOPK_MNATS = 99_999_000  # our token not even in vLLM's top-K => real bug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.58)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    ap.add_argument("--max-num-seqs", type=int, default=2)
    ap.add_argument("--moe-backend", default="triton")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    N, T = len(PROMPTS), args.max_tokens
    our = np.fromfile(os.path.join(args.golden_dir, "our_ids.i32"),
                      dtype="<i4").reshape(N, T)
    greedy = np.load(os.path.join(args.golden_dir, "greedy_ids.npy"))

    llm_kwargs = dict(model=args.model, dtype="bfloat16", enforce_eager=True,
                      gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len,
                      max_num_batched_tokens=args.max_num_batched_tokens,
                      max_num_seqs=args.max_num_seqs)
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
    print(f"moe_backend={args.moe_backend or 'auto'}")
    llm = LLM(**llm_kwargs)

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
