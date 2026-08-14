#!/usr/bin/env python3
# SPEC-NGRAM (ROAD-V1-D3) — vLLM-ngram-ON ORACLE CAPTURE.
#
# Captures the pinned vLLM 0.26.0.dev0 oracle's `--speculative-config ngram`
# greedy continuations + acceptance metrics on the 27B NVFP4 target, so the C++
# gate (tests/parity/test_qwen27_ngram_spec_decode.cpp) can assert
#   our-ngram-ON == vLLM-ngram-ON   (STRICT token identity)
# plus MEASURED nonzero acceptance ~ vLLM's (the dead-drafter trap — token
# identity ALONE passes on a drafter that never matches, so acceptance is a
# REQUIRED companion; see [[mtp-spec-decode-c1-speed-gate-met]] / the I5e lesson).
#
# ngram is DRAFT-FREE: the proposer matches the sequence's own suffix n-gram
# (length in [prompt_lookup_min, prompt_lookup_max]) against an earlier
# occurrence and proposes the k tokens that followed it. It works on ANY model
# (no draft checkpoint). Greedy ngram spec-decode is EXACTNESS-PRESERVING (the
# greedy rejection sampler accepts a draft iff it equals the target argmax), so
# vLLM-ngram-ON == vLLM-spec-OFF == our-ngram-ON, token-for-token. The prompts
# below are deliberately REPETITIVE so the suffix-ngram matcher finds matches and
# acceptance is nonzero.
#
# Grounding: vllm/config/speculative.py:734-762 (ngram method + prompt_lookup
# defaults 5/5), vllm/v1/spec_decode/ngram_proposer.py (the matcher). Metrics
# exactly like tests/v1/e2e/spec_decode/test_spec_decode.py::compute_acceptance_len.
#
# Run on dgx with the oracle venv, under flock "${GPU_LOCK:-$HOME/gpu.lock}":
#   flock "${GPU_LOCK:-$HOME/gpu.lock}" env PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/spec/ngram_27b_golden.py \
#       --out-dir tests/parity/goldens/ngram_27b
import argparse
import json
import os

# Deterministic REPETITIVE greedy battery. ngram's design workload is repetitive
# text (it accelerates exactly that), and a strongly-repetitive continuation is
# ALSO bf16-deterministic (the pattern token is unambiguous, no near-tie), so the
# gate is a CLEAN strict our-ngram-ON == vLLM-ngram-ON identity with high
# acceptance. (Factual single-answer prompts are ngram's worst case — no
# repetition, zero acceptance — and separately hit bf16 near-ties where even
# spec-OFF greedy diverges cross-engine; those are deliberately excluded so the
# gate measures ngram correctness, not near-tie roulette.)
PROMPTS = [
    "1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3",
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
    "a b c a b c a b c a b c a b c a b c a b c",
    "10 20 30 40 10 20 30 40 10 20 30 40 10 20 30 40",
    "red green blue red green blue red green blue red green blue red green",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target",
                    default=os.environ.get("NGRAM_TARGET",
                                           "unsloth/Qwen3.6-27B-NVFP4"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--num-spec-tokens", type=int,
                    default=int(os.environ.get("NGRAM_K", "3")))
    ap.add_argument("--prompt-lookup-min", type=int,
                    default=int(os.environ.get("NGRAM_MIN", "2")))
    ap.add_argument("--prompt-lookup-max", type=int,
                    default=int(os.environ.get("NGRAM_MAX", "3")))
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("NGRAM_GPU_UTIL", "0.30")))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("NGRAM_MAX_LEN", "4096")))
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    return ap.parse_args()


def default_out_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests", "parity", "goldens", "ngram_27b",
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

    spec = {
        "method": "ngram",
        "num_speculative_tokens": args.num_spec_tokens,
        "prompt_lookup_min": args.prompt_lookup_min,
        "prompt_lookup_max": args.prompt_lookup_max,
    }
    llm_kwargs = dict(
        model=args.target,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        # Liveness proof needs the spec_decode counters (see the module docstring).
        disable_log_stats=False,
        # GB10 119 GiB UNIFIED: the NVFP4 27B is multimodal; disable the vision
        # tower profiling to avoid the OOM-reboot ([[gb10-unified-memory-oom-reboots-box]]).
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config=spec,
    )
    print(f"[ngram] speculative_config={json.dumps(spec)}  target={args.target}  "
          f"gpu_memory_utilization={args.gpu_mem_util}", flush=True)

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
        except Exception as e:
            print(f"prompt[{i}] metrics unavailable: {type(e).__name__}: {e}")
        print(f"    prompt_ids={rec['prompt_token_ids']}")
        print(f"    -> {ids}\n    text={o.outputs[0].text!r}", flush=True)
        records.append(rec)

    out = {
        "mode": "spec-on",
        "method": "ngram",
        "target": args.target,
        "num_speculative_tokens": args.num_spec_tokens,
        "prompt_lookup_min": args.prompt_lookup_min,
        "prompt_lookup_max": args.prompt_lookup_max,
        "enforce_eager": True,
        "max_tokens": args.max_tokens,
        "vllm_version": __import__("vllm").__version__,
        "records": records,
    }
    fn = os.path.join(out_dir, "ngram_27b_spec_on.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {fn}  ({len(records)} prompts)")


if __name__ == "__main__":
    main()
