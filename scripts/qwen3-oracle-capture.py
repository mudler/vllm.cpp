#!/usr/bin/env python3
# ADDITIVE-MODEL W4 — capture the vLLM 0.25.0 oracle greedy continuation for a
# Qwen3 DENSE checkpoint on the standard prompt battery.
#
# Two golden artifacts are produced (see [[near-tie-distributional-gate]]):
#   greedy_ids.npy   [N, T]     i32 — run 0 (the point golden; the STRICT gate on
#                                     BIGGER dense models where vLLM is deterministic).
#   greedy_dist.npy  [N, T, K]  i32 — the per-(prompt,position) SET of tokens vLLM
#                                     emits across K greedy runs. The near-tie
#                                     DISTRIBUTIONAL gate (Qwen3-0.6B) PASSES when
#                                     OUR greedy token at each step is a MEMBER of
#                                     vLLM's observed set at that step. Short runs
#                                     (early EOS) are right-padded with -1.
#
# vLLM 0.25.0's OWN bf16 greedy is NON-DETERMINISTIC on near-ties (Qwen3-0.6B,
# FLASH_ATTN, head_dim 128): its kernels' reduction order varies run-to-run and
# flips greedy at top1<->top2 gaps <=~0.125 nats. So the "16/16 vs a FIXED golden"
# bar is ill-posed there; the honest "mirror vLLM" bar is membership in vLLM's
# actual observed K-run distribution. On a BIGGER dense model (Qwen3-4B) vLLM is
# expected deterministic -> the point golden gives a well-posed STRICT gate; this
# script's determinism report tells you which.
#
# The prompt list MUST match tests/parity/test_qwen3_paged_engine.cpp::Prompts()
# exactly (goldens and gate never drift). Run on dgx with the oracle venv; PATH
# must include ~/venvs/vllm-oracle/bin (flashinfer's ninja JIT):
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/qwen3-oracle-capture.py \
#       --model Qwen/Qwen3-0.6B --runs 10 --out-dir <goldens>/qwen3_greedy_0_6b
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     ~/venvs/vllm-oracle/bin/python scripts/qwen3-oracle-capture.py \
#       --model Qwen/Qwen3-4B  --runs 3  --out-dir <goldens>/qwen3_greedy_4b
#
# The historical Qwen3 distributional workflow above remains available.
# Qwen3.5 uses verified provenance and at least ten deterministic repeats
# under #2773. See docs/USAGE.md for its capture and launcher inputs.
import argparse
import io
from pathlib import Path
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_oracle_common as common

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "In the beginning God created",
    "The quick brown fox jumps over",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "The theory of relativity was developed by",
    "To be or not to be, that is",
    "The largest planet in our solar system is",
    "Machine learning is a subfield of",
    "The mitochondria is the powerhouse of",
    "Roses are red, violets are",
    "The first president of the United States was",
    "E equals m c",
    "A journey of a thousand miles begins with",
    "The chemical symbol for gold is",
]


def default_out_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "parity", "goldens", "qwen3_greedy_0_6b",
    )


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--out-dir", default=None,
                    help="golden dir (default tests/parity/goldens/qwen3_greedy_0_6b)")
    ap.add_argument("--runs", "--repetitions", type=int, default=int(os.environ.get("QWEN3_RUNS", "10")),
                    help="K greedy runs to build the observed distribution (K>=1)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float,
                    default=float(os.environ.get("QWEN3_GPU_MEM", "0.9")),
                    help="gpu_memory_utilization (GB10 unified memory reboots at "
                         "high values — pass 0.40 on dgx)")
    ap.add_argument("--per-prompt", action="store_true",
                    help="decode each prompt in its OWN generate() call (batch size 1) "
                         "to match the paged-engine gate's single-request decode "
                         "regime; otherwise all prompts are batched in one call")
    common.add_options(ap)
    return common.finish_args(ap, argv)


def parse_args(argv=None):
    return _parse_args(argv)


def _llm_kwargs(args):
    return common.llm_kwargs(args, args.gpu_mem)


def _mode_narration(args):
    return common.mode_narration(args)


def generate_all(llm, sp, per_prompt):
    """Return {prompt: token_ids} for one greedy pass over PROMPTS."""
    from_ids = {}
    if per_prompt:
        for p in PROMPTS:
            o = llm.generate([p], sp)[0]
            from_ids[p] = o
    else:
        for o in llm.generate(PROMPTS, sp):
            from_ids[o.prompt] = o
    return from_ids


def main():
    args = _parse_args()
    prompt_record = common.check_prompts(__file__, PROMPTS)
    model = common.model_identity(args)
    common.strict_inputs(args, model)
    import vllm
    from vllm import LLM, SamplingParams

    strict = common.is_qwen35(model["identity"])
    runtime = common.runtime_identity(vllm, args, strict) if strict else None
    print(_mode_narration(args))
    llm = LLM(**_llm_kwargs(args))
    runtime = runtime or common.runtime_identity(vllm, args, False)
    context = common.resolved_context(args, llm, model, runtime, prompt_record,
                                      1 if args.per_prompt else len(PROMPTS))
    context["tool"] = "qwen3-oracle-capture"
    strict = context["regime"] == "qwen3_5_strict"
    out_dir = Path(args.out_dir or default_out_dir())
    common.require(not strict or not out_dir.exists() or not any(out_dir.iterdir()),
                   f"capture directory is not empty: {out_dir}")
    N, T, K = len(PROMPTS), args.max_tokens, args.runs
    sp = SamplingParams(temperature=0.0, max_tokens=T, seed=args.seed or 0)
    common.record_sampling(context, sp)
    dist = np.full((N, T, K), -1, dtype="<i4")
    prompt_ids = []
    deterministic = True
    print(f"capture regime: {context['regime']}; batch={context['batching']['batch_size']}; repeats={K}")
    for k in range(K):
        by_prompt = generate_all(llm, sp, args.per_prompt)
        common.require(set(by_prompt) == set(PROMPTS), "oracle returned a different prompt set", "STRUCTURE_MISMATCH")
        for i, prompt in enumerate(PROMPTS):
            output = by_prompt[prompt]
            common.require(len(output.outputs) == 1, "oracle returned multiple continuations", "STRUCTURE_MISMATCH")
            ids = list(output.outputs[0].token_ids)
            prefix = list(output.prompt_token_ids)
            common.require(prefix and all(isinstance(x, int) and 0 <= x <= 2147483647 for x in prefix + ids)
                           and len(ids) <= T, "invalid oracle token sequence", "STRUCTURE_MISMATCH")
            dist[i, :len(ids), k] = ids
            if k == 0:
                prompt_ids.append(prefix)
            else:
                deterministic &= prefix == prompt_ids[i] and np.array_equal(dist[i, :, k], dist[i, :, 0])
    context["deterministic"] = bool(deterministic)
    common.require(not strict or deterministic, "oracle tokens or prompt tokenization changed across repeats", "NONDETERMINISTIC")
    common.confirm_inputs(args, vllm, context, __file__, PROMPTS)
    run0 = dist[:, :, 0].copy()
    payloads = {f"p{i}_prompt.i32": np.asarray(ids, dtype="<i4").tobytes()
                for i, ids in enumerate(prompt_ids)}
    for name, array in (("greedy_ids.npy", run0), ("greedy_dist.npy", dist)):
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payloads[name] = buffer.getvalue()
    common.publish(out_dir, payloads, context, "oracle-provenance.json", args.provenance_out,
                   protected_inputs=common.capture_input_paths(args, vllm, context, __file__))
    print(f"wrote {out_dir}; deterministic={bool(deterministic)}; output_sha256={context['output_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (common.CaptureError, OSError, ValueError) as error:
        print(str(error) if isinstance(error, common.CaptureError) else f"ARTIFACT_MISMATCH: {error}", file=sys.stderr)
        raise SystemExit(1)
