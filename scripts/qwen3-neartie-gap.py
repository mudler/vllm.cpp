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
# Run on the gate host with the oracle venv (PATH incl. ${VLLM_ORACLE}/bin):
#   PATH="${VLLM_ORACLE}/bin:$PATH" "${VLLM_ORACLE}/bin/python" \
#     scripts/qwen3-neartie-gap.py --model Qwen/Qwen3-4B \
#       --golden-dir tests/parity/goldens/qwen3_greedy_4b
#
# Qwen3.5 additionally requires a matching strict capture manifest and ten
# identical raw-logprob repeats. See docs/USAGE.md for its provenance inputs.
import argparse, os, sys
import hashlib
import io
import math
from pathlib import Path
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_oracle_common as common

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

def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    # GB10 (dgx.casa) has UNIFIED memory: vLLM's default gpu_memory_utilization=0.9
    # over-commits and REBOOTS the box. 0.40 is the safe ceiling (does not affect the
    # teacher-forced logprobs — only KV-cache capacity, unused here at max_tokens=1).
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--runs", "--repetitions", type=int, default=None)
    common.add_options(ap)
    return common.finish_args(ap, argv)


def _llm_kwargs(args):
    return common.llm_kwargs(args, args.gpu_mem_util)


def _mode_narration(args):
    return common.mode_narration(args)


def validate_capture(context, capture, inputs):
    common.require(capture.get("tool") == "qwen3-oracle-capture"
                   and capture.get("regime") == "qwen3_5_strict"
                   and capture.get("deterministic") is True
                   and isinstance(capture.get("repetitions"), int) and capture["repetitions"] >= 10,
                   "near-tie input is not a deterministic strict capture")
    for key in ("provenance_status", "resolved_model_identity", "prompts_sha256", "sampling",
                "sampling_normalized", "sampling_resolved", "sampling_resolution", "execution_mode", "batching"):
        common.require(capture.get(key) == context[key], f"capture {key} differs")
    for key in ("requested", "resolved"):
        common.require(capture.get("cache", {}).get(key) == context["cache"][key], f"capture cache {key} differs")
    for key in ("requested_revision", "identity", "files", "missing"):
        common.require(capture.get("model", {}).get(key) == context["model"][key], f"capture model {key} differs")
    for key in ("requested_revision", "package_files", "missing", "version"):
        common.require(capture.get("runtime", {}).get(key) == context["runtime"][key], f"capture oracle {key} differs")
    runtime = capture["runtime"]
    revision = runtime.get("revision")
    verification = runtime.get("revision_verification")
    valid_revision = verification == "clean_git_source" and common.full_revision(revision)
    if verification == "installed_version_vcs_prefix":
        # A captured installed revision keeps its observed prefix, independently
        # checked against the current verified full revision and package bytes.
        match = re.search(r"(?:\+|\.)g([0-9a-f]{7,40})(?:[.+-]|$)", str(runtime.get("version")))
        valid_revision = match is not None and revision == match.group(1)
    common.require(valid_revision and context["runtime"]["requested_revision"].startswith(revision),
                   "capture observed oracle revision is missing, unverified, or differs")
    for key, field in (("wheel", "sha256"), ("image", "digest")):
        common.require(capture.get("runtime", {}).get(key, {}).get(field) == context["runtime"][key][field],
                       f"capture oracle {key} differs")
    expected = {"greedy_ids.npy", "greedy_dist.npy", *(f"p{i}_prompt.i32" for i in range(len(PROMPTS)))}
    outputs = capture.get("outputs", {})
    common.require(set(outputs) == expected, "capture output manifest is incomplete")
    common.require(capture.get("output_sha256") == hashlib.sha256(common.json_bytes(outputs)).hexdigest(),
                   "capture output manifest hash differs")
    for name in expected:
        common.require(name in inputs and outputs[name] == {
            "sha256": hashlib.sha256(inputs[name]).hexdigest(), "size": len(inputs[name])},
            f"capture input hash differs: {name}")


def main():
    args = _parse_args()
    prompt_record = common.check_prompts(__file__, PROMPTS)
    model = common.model_identity(args)
    strict = common.is_qwen35(model["identity"])
    if args.runs is None:
        args.runs = 10 if strict else 1
    common.strict_inputs(args, model)
    import vllm
    from vllm import LLM, SamplingParams

    runtime = common.runtime_identity(vllm, args, True) if strict else None
    print(_mode_narration(args))
    llm = LLM(**_llm_kwargs(args))
    runtime = runtime or common.runtime_identity(vllm, args, False)
    context = common.resolved_context(args, llm, model, runtime, prompt_record, 1)
    strict = context["regime"] == "qwen3_5_strict"
    context["tool"] = "qwen3-neartie-gap"
    directory = Path(args.golden_dir)
    N, T = len(PROMPTS), args.max_tokens
    common.record_sampling(context, SamplingParams(temperature=0.0, max_tokens=T, seed=args.seed or 0))
    names = ["our_ids.i32", "greedy_ids.npy", *(f"p{i}_prompt.i32" for i in range(N))]
    if strict:
        names += ["greedy_dist.npy", "oracle-provenance.json"]
    inputs = {name: (directory / name).read_bytes() for name in names}
    if strict:
        capture = common.read_json(directory / "oracle-provenance.json")
        validate_capture(context, capture, inputs)
    common.require(len(inputs["our_ids.i32"]) == N * T * 4, "local token array has the wrong size", "STRUCTURE_MISMATCH")
    our = np.frombuffer(inputs["our_ids.i32"], dtype="<i4").reshape(N, T).copy()
    greedy = np.load(io.BytesIO(inputs["greedy_ids.npy"]), allow_pickle=False)
    common.require(greedy.shape == (N, T) and greedy.dtype == np.dtype("<i4"),
                   "greedy token array has the wrong shape or dtype", "STRUCTURE_MISMATCH")
    common.require(np.all(our >= 0), "teacher forcing needs a complete nonnegative token stream", "STRUCTURE_MISMATCH")
    if strict:
        dist = np.load(io.BytesIO(inputs["greedy_dist.npy"]), allow_pickle=False)
        common.require(dist.shape == (N, T, capture["repetitions"]) and dist.dtype == np.dtype("<i4")
                       and np.all(dist == greedy[:, :, None]), "capture distribution is not deterministic", "NONDETERMINISTIC")
    prefixes = []
    for i in range(N):
        data = inputs[f"p{i}_prompt.i32"]
        common.require(data and len(data) % 4 == 0, "prompt token file is incomplete", "STRUCTURE_MISMATCH")
        tokens = np.frombuffer(data, dtype="<i4").tolist()
        common.require(all(token >= 0 for token in tokens), "prompt contains a negative token", "STRUCTURE_MISMATCH")
        prefixes.append(tokens)
    sp_args = {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": args.topk, "seed": args.seed or 0}
    common.record_sampling(context, SamplingParams(**sp_args), key="teacher_forcing_sampling")
    reference = []
    gap_mnats = np.zeros((N, T), dtype="<i4")
    deterministic = True
    for repeat in range(args.runs):
        for i in range(N):
            full = prefixes[i] + our[i].tolist()
            outputs = llm.generate({"prompt_token_ids": full}, SamplingParams(**sp_args))
            common.require(len(outputs) == 1 and len(outputs[0].prompt_logprobs) == len(full),
                           "teacher-forced logprobs do not cover the exact prefix", "STRUCTURE_MISMATCH")
            logprobs = outputs[0].prompt_logprobs
            observed = []
            for j in range(T):
                values = logprobs[len(prefixes[i]) + j]
                common.require(isinstance(values, dict) and values, "missing teacher-forced logprobs", "STRUCTURE_MISMATCH")
                row = sorted((int(token), float(value.logprob)) for token, value in values.items())
                common.require(all(math.isfinite(value) for _, value in row), "teacher-forced logprob is not finite", "NONFINITE")
                observed.append(row)
                if repeat == 0:
                    lookup = dict(row)
                    token = int(our[i, j])
                    gap = max(lookup.values()) - lookup[token] if token in lookup else None
                    value = int(round(max(0.0, gap) * 1000.0)) if gap is not None else OUTSIDE_TOPK_MNATS
                    common.require(value <= 2147483647, "near-tie gap overflows the output dtype", "STRUCTURE_MISMATCH")
                    gap_mnats[i, j] = value
            if repeat == 0:
                reference.append(observed)
            else:
                deterministic &= observed == reference[i]
    common.require(not strict or deterministic, "teacher-forced logprobs changed before millinat rounding", "NONDETERMINISTIC")
    context["deterministic"] = bool(deterministic)
    context["inputs"] = {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                         for name, data in sorted(inputs.items())}
    context["logprobs_sha256"] = hashlib.sha256(common.json_bytes(reference)).hexdigest()
    for name, data in inputs.items():
        common.require((directory / name).read_bytes() == data, f"input changed during teacher forcing: {name}")
    common.confirm_inputs(args, vllm, context, __file__, PROMPTS)
    payloads = {}
    for name, array in (("our_ids.npy", our), ("neartie_gap_mnats.npy", gap_mnats)):
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payloads[name] = buffer.getvalue()
    protected = common.capture_input_paths(args, vllm, context, __file__)
    protected.update(directory / name for name in inputs)
    common.publish(directory, payloads, context, "neartie-provenance.json", args.provenance_out,
                   protected_inputs=protected)
    print(f"wrote {directory}; teacher forcing uses OUR exact prefix; output_sha256={context['output_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (common.CaptureError, OSError, ValueError) as error:
        print(str(error) if isinstance(error, common.CaptureError) else f"ARTIFACT_MISMATCH: {error}", file=sys.stderr)
        raise SystemExit(1)
