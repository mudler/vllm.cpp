#!/usr/bin/env python3
# ROAD-V1-D4-APC W3 — the near-tie-robust CORRECTNESS golden for the Qwen3-4B APC
# shared-prefix workload (see [[near-tie-distributional-gate]] and
# scripts/qwen3-neartie-gap.py, the sibling this mirrors).
#
# Qwen3-4B bf16 greedy is a near-tie decoder, and a prefix cache changes the
# attention KERNEL PATH for the uncached tail, so APC-ON and APC-OFF can each
# resolve a genuine near-tie either way — exactly as vLLM's OWN prefill-vs-decode
# (and its own APC-on-vs-off) greedy does. The honest "mirror vLLM" bar is: given
# OUR exact prefix, does vLLM's OWN logits place OUR token within a tiny gap of
# vLLM's argmax? This script TEACHER-FORCES vLLM on OUR engine's exact generated
# sequences (BOTH arms: our_ids_on.i32 / our_ids_off.i32, dumped by the W3 gate
# under VT_DUMP_IDS=1) and records, per position and per arm, the gap in nats
# between vLLM's argmax logprob and OUR token's logprob.
#
# Emits, into --golden-dir (tests/parity/goldens/qwen3_apc_4b):
#   our_ids_on.npy  / our_ids_off.npy   [N,T] i32 — OUR exact greedy tokens (anchors).
#   neartie_gap_on.npy / neartie_gap_off.npy [N,T] i32 — vLLM teacher-forced gap in
#       MILLI-nats for OUR token (0 = our token IS vLLM's argmax; 99_999_000 =
#       outside vLLM's top-K => a REAL forward divergence the gate fails on).
#
# Run on dgx with a working oracle venv (0.25.0-stage is fine; Qwen3-4B is
# byte-stable across the 0.25->0.26 pin):
#   flock "${GPU_LOCK:-$HOME/gpu.lock}" env PATH=$V/bin:$PATH $V/bin/python \
#     scripts/qwen3-apc-neartie-gap.py --model Qwen/Qwen3-4B \
#       --golden-dir tests/parity/goldens/qwen3_apc_4b
import argparse
import glob
import os

import numpy as np

OUTSIDE_TOPK_MNATS = 99_999_000  # our token not even in vLLM's top-K => real bug


def load_prompts(gdir):
    files = sorted(glob.glob(os.path.join(gdir, "prompt_*.txt")))
    return [open(f, "rb").read().decode("utf-8") for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    gdir = args.golden_dir
    prompts = load_prompts(gdir)
    N, T = len(prompts), args.max_tokens
    on = np.fromfile(os.path.join(gdir, "our_ids_on.i32"), dtype="<i4").reshape(N, T)
    off = np.fromfile(os.path.join(gdir, "our_ids_off.i32"), dtype="<i4").reshape(N, T)

    # enable_prefix_caching is irrelevant to teacher-forced logprobs (max_tokens=1,
    # single forward); keep it OFF so the returned prompt_logprobs are one clean
    # dense prefill of the whole forced sequence.
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              enable_prefix_caching=False, gpu_memory_utilization=args.gpu_mem_util)

    def gaps_for(arm_ids, tag):
        out_ids = np.zeros((N, T), dtype="<i4")
        gap_mnats = np.zeros((N, T), dtype="<i4")
        max_gap, worst, n_out = 0.0, None, 0
        for i in range(N):
            # Exact prompt tokenization vLLM uses on this string.
            pid = llm.generate([prompts[i]],
                               SamplingParams(temperature=0.0, max_tokens=1))[0].prompt_token_ids
            arm = [int(x) for x in arm_ids[i]]
            out_ids[i] = arm_ids[i]
            full = list(pid) + arm
            o = llm.generate({"prompt_token_ids": full},
                             SamplingParams(temperature=0.0, max_tokens=1,
                                            prompt_logprobs=args.topk))[0]
            plp, P = o.prompt_logprobs, len(pid)
            for j in range(T):
                d = plp[P + j] or {}
                if not d:
                    continue
                arg = max(d, key=lambda k: d[k].logprob)
                tid = arm[j]
                if tid in d:
                    gap = max(0.0, d[arg].logprob - d[tid].logprob)
                    gap_mnats[i, j] = int(round(gap * 1000.0))
                    if gap > max_gap:
                        max_gap, worst = gap, (i, j, gap)
                else:
                    gap_mnats[i, j] = OUTSIDE_TOPK_MNATS
                    n_out += 1
                    print(f"  [{tag}] p{i} tok{j}: OUR TOKEN {tid} OUTSIDE vLLM "
                          f"top-{args.topk} (REAL divergence)")
        np.save(os.path.join(gdir, f"our_ids_{tag}.npy"), out_ids)
        np.save(os.path.join(gdir, f"neartie_gap_{tag}.npy"), gap_mnats)
        print(f"=== {tag}: max near-tie gap {max_gap:.4f} nats (worst {worst}); "
              f"{n_out} outside-top-{args.topk} ===")
        return max_gap, n_out

    print(f"=== teacher-forced near-tie gap (both arms): {args.model} ===")
    on_gap, on_out = gaps_for(on, "on")
    off_gap, off_out = gaps_for(off, "off")
    print(f"wrote our_ids_{{on,off}}.npy + neartie_gap_{{on,off}}.npy to {gdir}")
    print(f"SUMMARY: APC-ON max gap {on_gap:.4f} nats ({on_out} real-div); "
          f"APC-OFF max gap {off_gap:.4f} nats ({off_out} real-div)")


if __name__ == "__main__":
    main()
