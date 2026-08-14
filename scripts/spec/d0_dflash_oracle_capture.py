#!/usr/bin/env python3
# DFlash D0 — READINESS + ORACLE CAPTURE (spec .agents/specs/dflash-spec-decode.md §0/§6).
#
# THE DECISIVE CHECK this script answers: does the pinned vLLM 0.25.0 oracle
# (`~/venvs/vllm-oracle` -> vllm-oracle-v0.25.0-stage) actually SERVE DFlash
# block-diffusion speculative decoding on sm_121/GB10 end-to-end — i.e. does it
# CONSTRUCT the DFlash draft, run the propose/verify/rejection loop, and emit
# COHERENT tokens (not just AutoConfig-construct the config)? This is the same
# RUN-verify, not config-construct, discipline recorded in
# [[oracle-gateability-model-runs-not-config-constructs]] (the OLMo-3 case where a
# constructible config still aborted at model init). We MUST run the model.
#
# It captures, on a fixed deterministic greedy prompt battery:
#   * --mode spec-on : LLM(target, speculative_config={method:"dflash", model:draft,
#                      num_speculative_tokens:K, ...}) greedy generation, PLUS the
#                      DFlash acceptance metrics (vllm:spec_decode_num_drafts /
#                      _num_accepted_tokens, acceptance_len computed exactly as
#                      tests/v1/e2e/spec_decode/test_spec_decode.py::compute_acceptance_len).
#   * --mode spec-off: the SAME target, SAME prompts, greedy, NO speculative_config
#                      — the baseline for the D-series three-way identity check
#                      (our-DFlash-ON == our-spec-OFF == vLLM-DFlash-ON greedy,
#                      token-for-token; mirrors the MTP I5e gate).
#
# Greedy spec-decode MUST be token-identical to greedy non-spec on the same target;
# comparing the two committed goldens is a first-class D0 self-check (a dead drafter
# still yields identical tokens, so token-identity is necessary but NOT sufficient —
# the acceptance rate captured here is the companion liveness proof, exactly the bug
# class [[mtp-spec-decode-c1-speed-gate-met]] flags).
#
# DFlash config keys are grounded in the pinned oracle:
#   * method "dflash" resolved in vllm/config/speculative.py (`use_dflash`, :1244);
#     `num_speculative_tokens` is REQUIRED (raises if None) and equals the drafted
#     block_size (z-lab 27B/35B block_size=16 -> the e2e fixture passes 16, NOT 15;
#     see test_spec_decode.py::dflash_config :1322-1341).
#   * DFlash forces parallel_drafting=True (speculative.py:910-911) and needs a
#     NON-CAUSAL-capable draft attention backend — "DFlash needs a non-causal-capable
#     backend like FLASH_ATTN" (speculative.py:120). `--draft-attention-backend`
#     lets D0 pin FLASH_ATTN if auto-selection does not pick one on sm_121 (THE soft
#     risk this task resolves).
#
# MEMORY (GB10 119 GiB is UNIFIED -> a large request is HOST RAM and has REBOOTED
# this box three times; see [[gb10-unified-memory-oom-reboots-box]]). Default target
# is the NVFP4 27B (~25 GiB on disk) — the spec's actual parity denominator (§2:
# "draft is bf16 while targets are NVFP4 — exactly the combination the DGX-Spark
# community container runs on GB10") and far safer than the bf16 27B (~69 GiB).
# gpu_memory_utilization is held MODERATE and the workload is short.
#
# Run on dgx with the oracle venv; PATH MUST include ~/venvs/vllm-oracle/bin (the
# FlashInfer JIT needs `ninja` on PATH or engine-core init dies in
# determine_available_memory), under `flock "${GPU_LOCK:-$HOME/gpu.lock}"`:
#   flock "${GPU_LOCK:-$HOME/gpu.lock}" env PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/spec/d0_dflash_oracle_capture.py \
#       --mode spec-on --out-dir tests/parity/goldens/dflash_27b
import argparse
import json
import os
import sys

# Fixed deterministic greedy battery (short prompts -> bounded KV on the unified pool).
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Q: What is 17 * 23?\nA:",
    "The three laws of robotics are",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["spec-on", "spec-off"], required=True)
    ap.add_argument("--target",
                    default=os.environ.get("DFLASH_TARGET",
                                           "unsloth/Qwen3.6-27B-NVFP4"))
    ap.add_argument("--draft",
                    default=os.environ.get("DFLASH_DRAFT",
                                           "z-lab/Qwen3.6-27B-DFlash"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-tokens", type=int, default=32)
    # block_size 16 -> num_speculative_tokens 16 (the e2e fixture value, NOT 15).
    ap.add_argument("--num-spec-tokens", type=int,
                    default=int(os.environ.get("DFLASH_K", "16")))
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("DFLASH_GPU_UTIL", "0.30")))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("DFLASH_MAX_LEN", "4096")))
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    # None => let vLLM auto-select; set FLASH_ATTN to pin the non-causal backend.
    ap.add_argument("--draft-attention-backend",
                    default=os.environ.get("DFLASH_DRAFT_ATTN_BACKEND") or None)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="run eager (default) — set for lower memory / no draft CG")
    ap.add_argument("--use-cudagraph", action="store_true",
                    help="enforce_eager=False (exercise DFlash FULL CG for 1+k)")
    return ap.parse_args()


def default_out_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests", "parity", "goldens", "dflash_27b",
    )


def metric_val(metrics, name):
    for m in metrics:
        if m.name == name:
            return m.value
    return None


def acceptance(metrics, prev):
    """acceptance_len exactly like test_spec_decode.py::compute_acceptance_len."""
    nd = metric_val(metrics, "vllm:spec_decode_num_drafts")
    na = metric_val(metrics, "vllm:spec_decode_num_accepted_tokens")
    if nd is None or na is None:
        return None, nd, na
    if prev is not None:
        nd -= prev[0] or 0
        na -= prev[1] or 0
    if not nd or nd <= 0:
        return 1.0, nd, na
    return 1.0 + (na / nd), nd, na


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    out_dir = args.out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    enforce_eager = not args.use_cudagraph  # eager by default; --use-cudagraph flips
    llm_kwargs = dict(
        model=args.target,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        # DFlash LIVENESS PROOF (the dead-drafter trap): greedy spec-on is token-
        # identical to greedy spec-off even when the drafter accepts ZERO tokens, so
        # coherent output is NOT sufficient — we MUST read the acceptance metrics.
        # The LLM API defaults disable_log_stats=True, which makes get_metrics()
        # raise "Stat logging disabled"; enable it so the spec_decode_num_drafts /
        # _num_accepted_tokens counters are collected (acceptance_len companion).
        disable_log_stats=False,
        # MEMORY (GB10 119 GiB UNIFIED): the NVFP4 27B is a *multimodal*
        # (ForConditionalGeneration) target; leaving the vision tower enabled makes
        # vLLM profile the encoder cache with one max-feature-size dummy image at
        # startup, which spiked the unified pool and HARD-REBOOTED dgx on the first
        # D0-redo attempt (2026-07-26; the exact [[gb10-unified-memory-oom-reboots-box]]
        # hazard). DFlash is a TEXT speculative path, so we disable image+video mm
        # entirely to skip that profiling — the drafter/target text path is unchanged.
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    if args.mode == "spec-on":
        spec = {
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.num_spec_tokens,
            "max_model_len": args.max_model_len,
        }
        if args.draft_attention_backend:
            spec["attention_backend"] = args.draft_attention_backend
        llm_kwargs["speculative_config"] = spec
        print(f"[spec-on] speculative_config={json.dumps(spec)}")
    else:
        print("[spec-off] no speculative_config (baseline greedy)")

    print(f"target={args.target}  enforce_eager={enforce_eager}  "
          f"gpu_memory_utilization={args.gpu_mem_util}  "
          f"max_model_len={args.max_model_len}", flush=True)

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    records = []
    prev = None
    for i, p in enumerate(PROMPTS):
        o = llm.generate([p], sp)[0]
        ids = list(int(t) for t in o.outputs[0].token_ids)
        rec = {
            "prompt": p,
            "prompt_token_ids": [int(t) for t in o.prompt_token_ids],
            "output_token_ids": ids,
            "text": o.outputs[0].text,
        }
        if args.mode == "spec-on":
            try:
                m = llm.get_metrics()
                al, nd, na = acceptance(m, prev)
                prev = (metric_val(m, "vllm:spec_decode_num_drafts"),
                        metric_val(m, "vllm:spec_decode_num_accepted_tokens"))
                rec["acceptance_len"] = al
                rec["num_drafts_delta"] = nd
                rec["num_accepted_delta"] = na
                print(f"prompt[{i}] {p!r} acceptance_len={al} "
                      f"(drafts+{nd} accepted+{na})")
            except Exception as e:  # metrics are a nicety; never fail the capture
                print(f"prompt[{i}] metrics unavailable: {type(e).__name__}: {e}")
        print(f"    prompt_ids={rec['prompt_token_ids']}")
        print(f"    -> {ids}\n    text={o.outputs[0].text!r}", flush=True)
        records.append(rec)

    out = {
        "mode": args.mode,
        "target": args.target,
        "draft": args.draft if args.mode == "spec-on" else None,
        "num_speculative_tokens": args.num_spec_tokens if args.mode == "spec-on" else None,
        "draft_attention_backend": args.draft_attention_backend,
        "enforce_eager": enforce_eager,
        "max_tokens": args.max_tokens,
        "vllm_version": __import__("vllm").__version__,
        "records": records,
    }
    fn = os.path.join(out_dir, f"dflash_27b_{args.mode.replace('-', '_')}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {fn}  ({len(records)} prompts)")


if __name__ == "__main__":
    main()
