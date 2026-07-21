#!/usr/bin/env python3
# ADDITIVE-QUANT W4 DIAGNOSIS (Qwen3-32B-NVFP4A16, compressed-tensors W4A16) —
# the ratified TEACHER-FORCING isolation. Mirror of
# scripts/qwen3coder-neartie-gap.py / scripts/qwen3-neartie-gap.py.
#
# WHY. The strict gate scores 4/6 with both divergences at token 1 (the first
# DECODE step) while every PREFILL argmax is exact. The question the record must
# answer is which of two things that is:
#   (a) a genuine forward/numerics DEFECT of ours — vLLM's own argmax GIVEN OUR
#       EXACT PREFIX is far from our token (large nats gap, or our token outside
#       vLLM's top-K);
#   (b) accumulated bf16 near-tie drift — our token IS vLLM's teacher-forced
#       argmax, or is separated from it by a gap indistinguishable from bf16
#       rounding.
# This script feeds vLLM OUR exact generated sequence (our_ids.i32, dumped by the
# gate under VT_DUMP_IDS=1) as prompt_token_ids and reads prompt_logprobs, giving
# vLLM's own per-position distribution on OUR prefix.
#
# NOTE specific to THIS row: because every prefill argmax is exact, the prefix at
# the divergent token-1 positions is BIT-IDENTICAL between us and vLLM. So the
# gap reported at those positions is not a prefix-drift artefact — it is a direct
# comparison of our decode-step logit ordering against vLLM's at the same state.
#
# Emits, into --golden-dir:
#   our_ids.npy           [N,T] i32 — OUR engine's exact greedy tokens.
#   neartie_gap_mnats.npy [N,T] i32 — vLLM's teacher-forced gap in MILLI-nats for
#                                     OUR token (0 = our token IS vLLM's argmax;
#                                     99_999_000 = outside vLLM's top-K).
#
# GB10's 119 GiB is UNIFIED (host RAM): keep --gpu-mem-util LOW (0.40 default).
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH ~/venvs/vllm-oracle/bin/python \
#     scripts/qwen3-32b-nvfp4a16-neartie-gap.py \
#       --golden-dir tests/parity/goldens/qwen3_32b_nvfp4a16_greedy
import argparse
import os

import numpy as np

# MUST match tests/vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp::Prompts()
# and scripts/qwen3-32b-nvfp4a16-oracle-capture.py::PROMPTS exactly.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time,",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "In 1969, humans first walked on",
]
OUTSIDE_TOPK_MNATS = 99_999_000  # our token not even in vLLM's top-K => real bug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RedHatAI/Qwen3-32B-NVFP4A16")
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    ap.add_argument("--max-num-seqs", type=int, default=2)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    N, T = len(PROMPTS), args.max_tokens
    our = np.fromfile(os.path.join(args.golden_dir, "our_ids.i32"),
                      dtype="<i4").reshape(N, T)
    greedy = np.load(os.path.join(args.golden_dir, "greedy_ids.npy"))

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len,
              max_num_batched_tokens=args.max_num_batched_tokens,
              max_num_seqs=args.max_num_seqs)

    gap_mnats = np.zeros((N, T), dtype="<i4")
    print(f"=== teacher-forced near-tie gap: {args.model} (OUR prefix) ===")
    max_gap = 0.0
    worst = None
    n_div = 0
    for i in range(N):
        # Use the EXACT prompt tokenization the oracle greedy capture used so the
        # teacher-forced prefix aligns bit-for-bit.
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
            gap = 0.0
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
                print(f"  DIVERGENT p{i:2d} tok{j:2d}: our={our_tid} "
                      f"vLLM_greedy={int(greedy[i,j])} vLLM_tf_argmax={arg_tid} "
                      f"gap={gap_mnats[i,j]/1000.0:.6f} nats")
                top = sorted(d.items(), key=lambda kv: -kv[1].logprob)[:5]
                for tid, lp in top:
                    print(f"      top: {tid:8d} lp={lp.logprob:+.6f} "
                          f"{lp.decoded_token!r}")
        print(f"  p{i}: max gap over T={T} is "
              f"{max(gap_mnats[i])/1000.0:.6f} nats")

    np.save(os.path.join(args.golden_dir, "our_ids.npy"), our)
    np.save(os.path.join(args.golden_dir, "neartie_gap_mnats.npy"), gap_mnats)
    print(f"=== {n_div} token-divergent positions vs vLLM greedy; "
          f"max near-tie gap {max_gap:.6f} nats (worst {worst}) ===")
    print(f"wrote {args.golden_dir}/our_ids.npy + neartie_gap_mnats.npy "
          f"{gap_mnats.shape}")


if __name__ == "__main__":
    main()
