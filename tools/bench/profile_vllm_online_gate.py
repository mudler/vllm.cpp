#!/usr/bin/env python3
"""Capture production-vLLM kernels for the online-gate token shape.

This is the mandated GB10 fallback when nsys breaks V1 EngineCore startup.
The sequence mirrors pinned vLLM's profiler API and example:
``examples/features/profiling/simple_profiling_offline.py:20-40`` and
``vllm/entrypoints/llm.py:787-798`` at e24d1b24.  The process-safe ``main``
guard is intentional because vLLM uses multiprocessing spawn.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import pathlib
import sys
import time

# Keep the committed reproduction command valid when this file is invoked by
# absolute path from a commit-owned evidence directory.  Module execution
# already has the repository root on sys.path; direct script execution does
# not.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.bench.online_gate import (
    INPUT_LEN,
    OUTPUT_LEN,
    TRACE_CONCURRENCY,
    TRACE_PROMPTS,
    TRACE_REPETITIONS,
)
from tools.bench.serve_low_common import (
    HarnessError,
    canonical_json,
    read_jsonl,
    sha256_file,
    write_json_atomic,
)


def load_prompts(corpus: pathlib.Path, batch_size: int):
    if batch_size <= 0:
        raise HarnessError("batch size must be positive")
    rows = list(read_jsonl(corpus))
    if len(rows) < batch_size:
        raise HarnessError(f"{corpus}: found {len(rows)} prompts; need {batch_size}")
    prompts = []
    for index, row in enumerate(rows[:batch_size]):
        token_ids = row.get("prompt_token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != INPUT_LEN:
            raise HarnessError(f"{corpus}:{index + 1}: prompt token IDs are not exact")
        if any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in token_ids
        ):
            raise HarnessError(f"{corpus}:{index + 1}: prompt token ID is not an integer")
        prompts.append(token_ids)
    return prompts


def _output_digest(outputs) -> str:
    digest = hashlib.sha256()
    for output in outputs:
        token_ids = output.outputs[0].token_ids
        if len(token_ids) != OUTPUT_LEN:
            raise HarnessError(
                f"profile generation returned {len(token_ids)} tokens; expected {OUTPUT_LEN}"
            )
        digest.update(canonical_json(token_ids).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def async_scheduling_override(mode: str) -> dict[str, bool]:
    """Translate the diagnostic CLI mode without changing the binding default.

    Omitting ``async_scheduling`` is important for the accepted H1d recipe: it
    exercises vLLM's own default resolution.  Explicit ``on``/``off`` modes are
    reserved for paired denominator probes.
    """

    if mode == "default":
        return {}
    if mode == "on":
        return {"async_scheduling": True}
    if mode == "off":
        return {"async_scheduling": False}
    raise HarnessError(f"unknown async scheduling mode: {mode}")


def run_closed_loop(
    llm,
    prompts,
    sampling,
    max_concurrency: int,
    request_id_base: int,
    final_output_kind=None,
):
    """Run the offline engine with the online client's closed-loop admission.

    Only ``max_concurrency`` requests are resident. Each finished request admits
    one replacement, matching ``vllm bench serve --max-concurrency`` instead of
    preloading all prompts into the scheduler's waiting queue.
    """

    if max_concurrency <= 0 or max_concurrency > len(prompts):
        raise HarnessError("max-concurrency must be in [1, num-prompts]")
    engine = llm.llm_engine
    next_index = 0
    completed = {}

    def submit(index: int) -> None:
        params = copy.copy(sampling)
        if final_output_kind is not None:
            params.output_kind = final_output_kind
        engine.add_request(str(request_id_base + index), prompts[index], params)

    while next_index < min(max_concurrency, len(prompts)):
        submit(next_index)
        next_index += 1

    while engine.has_unfinished_requests():
        for output in engine.step():
            if not output.finished:
                continue
            index = int(output.request_id) - request_id_base
            if index in completed or index < 0 or index >= len(prompts):
                raise HarnessError("closed-loop profiler returned an invalid request id")
            completed[index] = output
            if next_index < len(prompts):
                submit(next_index)
                next_index += 1

    if len(completed) != len(prompts):
        raise HarnessError(
            f"closed-loop profiler completed {len(completed)} prompts; "
            f"expected {len(prompts)}"
        )
    return [completed[index] for index in range(len(prompts))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--profile-dir", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=TRACE_PROMPTS)
    parser.add_argument("--max-concurrency", type=int, default=TRACE_CONCURRENCY)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=TRACE_REPETITIONS)
    parser.add_argument(
        "--async-scheduling",
        choices=("default", "on", "off"),
        default="default",
        help=(
            "keep vLLM's default resolution or explicitly force the paired "
            "async-scheduler diagnostic arm"
        ),
    )
    args = parser.parse_args()
    if not args.model.is_dir():
        raise HarnessError(f"model snapshot is absent: {args.model}")
    if args.repetitions < 2:
        raise HarnessError("profile needs at least two measured repetitions")
    if args.profile_dir.exists() and any(args.profile_dir.iterdir()):
        raise HarnessError(f"refusing to mix profiler output in {args.profile_dir}")
    if args.metadata.exists():
        raise HarnessError(f"refusing to overwrite profiler metadata: {args.metadata}")
    args.profile_dir.mkdir(parents=True, exist_ok=True)

    # Delayed imports keep CPU contract tests independent of vLLM/PyTorch.
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config import ProfilerConfig
    from vllm.sampling_params import RequestOutputKind

    if args.max_num_seqs <= 0:
        raise HarnessError("max-num-seqs must be positive")
    if args.max_concurrency <= 0 or args.max_concurrency > args.max_num_seqs:
        raise HarnessError("max-concurrency must be in [1, max-num-seqs]")
    if args.max_num_batched_tokens < args.max_num_seqs:
        raise HarnessError("max-num-batched-tokens must cover max-num-seqs")
    token_rows = load_prompts(args.corpus, args.num_prompts)
    prompts = [TokensPrompt(prompt_token_ids=row) for row in token_rows]
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=OUTPUT_LEN,
        min_tokens=OUTPUT_LEN,
    )
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        seed=0,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.6,
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(args.profile_dir.resolve()),
            torch_profiler_with_stack=False,
        ),
        **async_scheduling_override(args.async_scheduling),
    )

    measured_digests = []
    llm.start_profile("online-gate")
    try:
        # Keep the exact four-run semantic workload used by the accepted vLLM
        # trace: one 48-prompt warmup plus three measured 48-prompt repetitions.
        # H1d compares only the clean pure-decode GPU annotation windows from
        # this raw trace; its local four-replay probe remains structural and is
        # never treated as an equal-duration timing window.
        warmup = run_closed_loop(
            llm,
            prompts,
            sampling,
            args.max_concurrency,
            request_id_base=0,
            final_output_kind=RequestOutputKind.FINAL_ONLY,
        )
        warmup_digest = _output_digest(warmup)
        for repetition in range(args.repetitions):
            outputs = run_closed_loop(
                llm,
                prompts,
                sampling,
                args.max_concurrency,
                request_id_base=(repetition + 1) * 100_000,
                final_output_kind=RequestOutputKind.FINAL_ONLY,
            )
            measured_digests.append(_output_digest(outputs))
    finally:
        llm.stop_profile()
    output_digests = [warmup_digest, *measured_digests]

    # EngineCore may write the worker trace asynchronously after stop_profile.
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if list(args.profile_dir.rglob("*.pt.trace.json*")):
            break
        time.sleep(1.0)
    result = {
        "admission_mode": "closed-loop",
        "async_scheduling_requested": args.async_scheduling,
        "async_scheduling_resolved": (
            llm.llm_engine.vllm_config.scheduler_config.async_scheduling
        ),
        "enable_prefix_caching": False,
        "max_concurrency": args.max_concurrency,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": llm.llm_engine.model_config.max_model_len,
        "corpus": str(args.corpus),
        "corpus_sha256": sha256_file(args.corpus),
        "input_len": INPUT_LEN,
        "model": str(args.model),
        "output_digest": warmup_digest,
        "output_digests": output_digests,
        "output_digests_equal": len(set(output_digests)) == 1,
        "output_len": OUTPUT_LEN,
        "profile_dir": str(args.profile_dir.resolve()),
        "num_prompts": args.num_prompts,
        "repetitions": args.repetitions,
        "profiled_warmup_prompts": args.num_prompts,
    }
    write_json_atomic(args.metadata, result)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"profile-vllm-online-gate: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
