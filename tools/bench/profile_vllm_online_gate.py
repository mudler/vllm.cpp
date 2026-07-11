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
import hashlib
import pathlib
import time

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
        if any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--profile-dir", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=TRACE_PROMPTS)
    parser.add_argument("--max-num-seqs", type=int, default=TRACE_CONCURRENCY)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=TRACE_REPETITIONS)
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

    if args.max_num_seqs <= 0 or args.max_num_seqs > args.num_prompts:
        raise HarnessError("max-num-seqs must be in [1, num-prompts]")
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
        max_model_len=INPUT_LEN + OUTPUT_LEN,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=0.6,
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(args.profile_dir.resolve()),
            torch_profiler_with_stack=False,
        ),
    )

    measured_digests = []
    llm.start_profile("online-gate")
    try:
        # Ours is captured from server start and its three client invocations
        # each include a full c16 warmup wave. Profile the equivalent 48-prompt
        # (three-wave) warmup here so both traces contain 192 prompt executions
        # before comparing runtime-resolved kernel families.
        warmup = llm.generate(prompts, sampling, use_tqdm=False)
        warmup_digest = _output_digest(warmup)
        for _ in range(args.repetitions):
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
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
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
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
