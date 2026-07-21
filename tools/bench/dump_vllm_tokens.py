#!/usr/bin/env python3
"""Dump greedy vLLM token IDs and optional raw top logits for a workload."""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--top-logits-output", type=Path)
    parser.add_argument("--num-top-logits", type=int, default=8)
    args = parser.parse_args()

    rows = json.loads(args.dataset_path.read_text(encoding="utf-8"))
    prompts = [row["conversations"][0]["value"] for row in rows[: args.num_prompts]]
    if len(prompts) != args.num_prompts:
        raise ValueError("dataset has fewer rows than --num-prompts")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        seed=0,
        logprobs_mode="raw_logits" if args.top_logits_output else "raw_logprobs",
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.output_len,
        ignore_eos=True,
        detokenize=False,
        logprobs=args.num_top_logits if args.top_logits_output else None,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - start
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    args.output.write_text(json.dumps(token_ids) + "\n", encoding="utf-8")

    if args.benchmark_output:
        tokenizer = llm.get_tokenizer()
        input_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)
        output_tokens = sum(len(ids) for ids in token_ids)
        summary = {
            "elapsed_seconds": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "request_throughput": len(outputs) / elapsed,
            "output_token_throughput": output_tokens / elapsed,
            "total_token_throughput": (input_tokens + output_tokens) / elapsed,
            "temperature": args.temperature,
        }
        args.benchmark_output.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))

    if args.top_logits_output:
        traces = []
        for output in outputs:
            request_trace = []
            for position in output.outputs[0].logprobs:
                ranked = sorted(position.items(), key=lambda item: item[1].rank)
                request_trace.append(
                    [
                        {
                            "token_id": token_id,
                            "logit": entry.logprob,
                            "rank": entry.rank,
                        }
                        for token_id, entry in ranked
                    ]
                )
            traces.append(request_trace)
        args.top_logits_output.write_text(json.dumps(traces) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
