#!/usr/bin/env python3
"""Dump greedy vLLM token IDs for a local ShareGPT correctness workload."""

import argparse
import json
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
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
        detokenize=False,
    )
    outputs = llm.generate(prompts, sampling)
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    args.output.write_text(json.dumps(token_ids) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
