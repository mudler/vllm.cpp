#!/usr/bin/env python3
"""Run an exact closed-loop vLLM workload with client-observed metrics.

This is the reference-side companion to ``examples/bench/bench_core.h``.  It
uses the same bounded in-flight admission, DELTA outputs, fixed token count and
numpy-style percentile reduction, while retaining generated IDs for the
correctness precondition.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import statistics
import time
from typing import Any


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = p / 100.0 * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "median": percentile(values, 50),
        "p99": percentile(values, 99),
    }


def load_prompt_text(path: pathlib.Path, count: int) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[str] = []
    for row in rows:
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            continue
        value = conversations[0].get("value")
        if isinstance(value, str):
            prompts.append(value)
        if len(prompts) == count:
            break
    if len(prompts) != count:
        raise ValueError(f"dataset has {len(prompts)} valid prompts; need {count}")
    return prompts


def run_closed_loop(llm, prompts, sampling, concurrency: int, request_id_base: int):
    from vllm.sampling_params import RequestOutputKind

    if concurrency <= 0 or concurrency > len(prompts):
        raise ValueError("concurrency must be in [1, num-prompts]")
    engine = llm.llm_engine
    records: dict[int, dict[str, Any]] = {}
    next_index = 0
    completed = 0
    started = time.perf_counter()

    def submit(index: int) -> None:
        params = copy.copy(sampling)
        params.output_kind = RequestOutputKind.DELTA
        records[index] = {
            "arrival_s": time.perf_counter(),
            "first_token_s": None,
            "last_token_s": None,
            "completion_s": None,
            "itls_s": [],
            "output_token_ids": [],
            "core_ttft_s": None,
        }
        engine.add_request(str(request_id_base + index), prompts[index], params)

    while next_index < min(concurrency, len(prompts)):
        submit(next_index)
        next_index += 1

    while engine.has_unfinished_requests():
        for output in engine.step():
            index = int(output.request_id) - request_id_base
            if index not in records:
                raise RuntimeError(f"unexpected request id: {output.request_id}")
            record = records[index]
            if output.outputs and output.outputs[0].token_ids:
                now = time.perf_counter()
                if record["first_token_s"] is None:
                    record["first_token_s"] = now
                else:
                    record["itls_s"].append(now - record["last_token_s"])
                record["last_token_s"] = now
                record["output_token_ids"].extend(output.outputs[0].token_ids)
            if output.finished and record["completion_s"] is None:
                record["completion_s"] = time.perf_counter()
                if output.metrics is not None:
                    record["core_ttft_s"] = output.metrics.first_token_latency
                completed += 1

        while next_index < len(prompts) and next_index - completed < concurrency:
            submit(next_index)
            next_index += 1

    duration = time.perf_counter() - started
    return duration, [records[index] for index in range(len(prompts))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True, type=pathlib.Path)
    parser.add_argument("--metrics-output", required=True, type=pathlib.Path)
    parser.add_argument("--tokens-output", required=True, type=pathlib.Path)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--async-scheduling", choices=("default", "on", "off"), default="default"
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--request-id-base", type=int, default=0)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams, TokensPrompt

    prompt_text = load_prompt_text(args.dataset_path, args.num_prompts)
    scheduler_override: dict[str, bool] = {}
    if args.async_scheduling != "default":
        scheduler_override["async_scheduling"] = args.async_scheduling == "on"
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        seed=0,
        disable_log_stats=False,
        enforce_eager=args.enforce_eager,
        **scheduler_override,
    )
    tokenizer = llm.get_tokenizer()
    prompt_ids = [tokenizer.encode(text) for text in prompt_text]
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.output_len,
        min_tokens=args.output_len,
        ignore_eos=True,
        detokenize=False,
    )
    duration, records = run_closed_loop(
        llm, prompts, sampling, args.max_concurrency, args.request_id_base
    )

    request_rows = []
    all_itls_ms: list[float] = []
    token_ids: list[list[int]] = []
    for index, record in enumerate(records):
        if record["first_token_s"] is None or record["completion_s"] is None:
            raise RuntimeError(f"request {index} has incomplete timing")
        ids = list(record["output_token_ids"])
        if len(ids) != args.output_len:
            raise RuntimeError(
                f"request {index} emitted {len(ids)} tokens; expected {args.output_len}"
            )
        token_ids.append(ids)
        ttft_ms = (record["first_token_s"] - record["arrival_s"]) * 1000.0
        e2el_ms = (record["completion_s"] - record["arrival_s"]) * 1000.0
        tpot_ms = (e2el_ms - ttft_ms) / (len(ids) - 1)
        itls_ms = [value * 1000.0 for value in record["itls_s"]]
        all_itls_ms.extend(itls_ms)
        request_rows.append(
            {
                "index": index,
                "prompt_tokens": len(prompt_ids[index]),
                "output_tokens": len(ids),
                "ttft_ms": ttft_ms,
                "core_ttft_ms": (
                    record["core_ttft_s"] * 1000.0
                    if record["core_ttft_s"] is not None
                    else None
                ),
                "tpot_ms": tpot_ms,
                "e2el_ms": e2el_ms,
                "itls_ms": itls_ms,
            }
        )

    ttft_ms = [row["ttft_ms"] for row in request_rows]
    core_ttft_ms = [
        row["core_ttft_ms"]
        for row in request_rows
        if row["core_ttft_ms"] is not None
    ]
    tpot_ms = [row["tpot_ms"] for row in request_rows]
    e2el_ms = [row["e2el_ms"] for row in request_rows]
    input_tokens = sum(map(len, prompt_ids))
    output_tokens = sum(map(len, token_ids))
    result = {
        "admission": "closed_loop_delta_client_observed",
        "async_scheduling_requested": args.async_scheduling,
        "duration_s": duration,
        "successful_requests": len(records),
        "maximum_request_concurrency": args.max_concurrency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_throughput": len(records) / duration,
        "input_token_throughput": input_tokens / duration,
        "output_token_throughput": output_tokens / duration,
        "total_token_throughput": (input_tokens + output_tokens) / duration,
        "mean_per_stream_decode_rate": 1000.0 / statistics.fmean(tpot_ms),
        "ttft_ms": summarize(ttft_ms),
        "core_ttft_ms": summarize(core_ttft_ms),
        "tpot_ms": summarize(tpot_ms),
        "itl_ms": summarize(all_itls_ms),
        "e2el_ms": summarize(e2el_ms),
        "requests": request_rows,
    }
    args.metrics_output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.tokens_output.write_text(json.dumps(token_ids) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "requests"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
