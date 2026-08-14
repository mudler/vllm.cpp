#!/usr/bin/env python3
# DFlash D8 — ACCEPTANCE RCA (spec .agents/specs/dflash-spec-decode.md §0 D8).
#
# The decisive question: our DFlash-ON accepts ~2.49 draft-tok/step vs vLLM ~3.13
# (D7). Is that deficit (a) a FIXED per-step draft-quality gap (our draft proposes
# systematically worse blocks on identical context -> REDUCIBLE if a bug), or
# (b) TRAJECTORY DIVERGENCE that COMPOUNDS with generation length (a single bf16
# near-tie flip decorrelates the two greedy trajectories; each engine then decodes
# a different, differently-draftable text -> bf16-IRREDUCIBLE, D6 RCA)?
#
# SIGNATURE TEST: measure vLLM-DFlash-ON's OWN acceptance_len on the 4 committed
# golden prompts as a function of generation length L in {8,16,32,64,128,256}.
#   * A FIXED per-step deficit is length-INVARIANT (shows already at L=32, where
#     D5 measured our acceptance ~= vLLM's on all 4 prompts).
#   * TRAJECTORY divergence GROWS the ours-vs-vLLM gap with L (D5 gap~0 at L=32
#     vs D7 gap~20% at L=256 already hints this, but on DIFFERENT prompt sets;
#     this pins it on the SAME prompts, both engines).
# Also dumps vLLM's per-length acceptance so the C++ vllm-bench arm (run separately
# on the SAME dataset + lengths) is directly comparable. Both metrics normalized
# to ACCEPTED DRAFT-TOK/STEP: vLLM = acceptance_len - 1; ours = accept_rate * k.
#
# Run (dgx, flock, oracle venv, mm-off + gpu_util 0.30, VLLM_USE_V2_MODEL_RUNNER=1):
#   flock "${GPU_LOCK:-$HOME/gpu.lock}" env PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     VLLM_USE_V2_MODEL_RUNNER=1 ~/venvs/vllm-oracle/bin/python \
#     scripts/spec/d8_acceptance_rca.py --dataset /tmp/dflash_golden4.json --out /tmp/d8_vllm_rca.json
import argparse
import json
import os
import sys
import time

os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def metric_val(metrics, name):
    for m in metrics:
        if getattr(m, "name", None) == name:
            if hasattr(m, "value"):
                return getattr(m, "value")
            return sum(getattr(m, "values", []) or [0])
    return None


def accepted_per_step(llm, base_drafts, base_accepted):
    """Return (drafts_delta, accepted_delta, accepted_draft_tok_per_step) since base."""
    m = llm.get_metrics()
    nd = metric_val(m, "vllm:spec_decode_num_drafts") or 0
    na = metric_val(m, "vllm:spec_decode_num_accepted_tokens") or 0
    dd = nd - base_drafts
    ad = na - base_accepted
    aps = (ad / dd) if dd else 0.0
    return nd, na, dd, ad, aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="unsloth/Qwen3.6-27B-NVFP4")
    ap.add_argument("--draft", default="z-lab/Qwen3.6-27B-DFlash")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--lengths", default="8,16,32,64,128,256")
    a = ap.parse_args()

    lengths = [int(x) for x in a.lengths.split(",")]
    prompts = [e["conversations"][0]["value"] for e in json.load(open(a.dataset))]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=a.target,
        speculative_config={
            "method": "dflash", "model": a.draft,
            "num_speculative_tokens": a.k, "max_model_len": a.max_model_len,
        },
        gpu_memory_utilization=a.gpu_mem_util,
        max_model_len=a.max_model_len, max_num_seqs=1, max_num_batched_tokens=a.max_model_len,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=False, disable_log_stats=False,
    )
    # warmup (excluded)
    llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))

    records = []
    bd, ba = 0, 0
    m0 = llm.get_metrics()
    bd = metric_val(m0, "vllm:spec_decode_num_drafts") or 0
    ba = metric_val(m0, "vllm:spec_decode_num_accepted_tokens") or 0
    for L in lengths:
        for pi, p in enumerate(prompts):
            sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=L, ignore_eos=True)
            t0 = time.perf_counter()
            o = llm.generate([p], sp)[0]
            dt = time.perf_counter() - t0
            nd, na, dd, ad, aps = accepted_per_step(llm, bd, ba)
            bd, ba = nd, na
            toks = list(o.outputs[0].token_ids)
            records.append({
                "length": L, "prompt_idx": pi, "prompt": p[:50],
                "n_out": len(toks), "elapsed_s": round(dt, 3),
                "drafts_delta": dd, "accepted_delta": ad,
                "accepted_draft_tok_per_step": round(aps, 4),
                "acceptance_len": round(aps + 1.0, 4),
                "token_ids": toks,
            })
            print(f"[d8] L={L:4d} p{pi} acc/step={aps:.3f} acc_len={aps+1:.3f} "
                  f"n={len(toks)} ({dt:.1f}s)", flush=True)

    # Aggregate accepted-draft-tok/step per length (over the 4 prompts).
    agg = {}
    for L in lengths:
        rs = [r for r in records if r["length"] == L]
        tot_d = sum(r["drafts_delta"] for r in rs)
        tot_a = sum(r["accepted_delta"] for r in rs)
        agg[str(L)] = {
            "accepted_draft_tok_per_step": round(tot_a / tot_d, 4) if tot_d else 0.0,
            "acceptance_len": round(tot_a / tot_d + 1.0, 4) if tot_d else 1.0,
            "total_drafts": tot_d, "total_accepted": tot_a,
        }
    out = {"engine": "vllm-dflash-on-graphed", "target": a.target, "draft": a.draft,
           "k": a.k, "lengths": lengths, "aggregate_by_length": agg, "records": records}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[d8] AGGREGATE accepted-draft-tok/step by length (vLLM-DFlash-ON):")
    for L in lengths:
        print(f"    L={L:4d}: {agg[str(L)]['accepted_draft_tok_per_step']:.3f} "
              f"(acc_len {agg[str(L)]['acceptance_len']:.3f})")
    print(f"[d8] wrote {a.out}")


if __name__ == "__main__":
    main()
