#!/usr/bin/env python3
"""Generate deterministic ShareGPT prompts for vLLM/vllm.cpp A/B runs."""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-prompts", required=True, type=int)
    parser.add_argument("--input-len", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--words", nargs="+", default=("hello", "world", "1", "2"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = []
    for index in range(args.num_prompts):
        rng = random.Random(args.seed + index)
        prompt_words = []
        while True:
            prompt_words.extend(rng.choice(args.words) for _ in range(8))
            prompt = " ".join(prompt_words)
            if len(tokenizer.encode(prompt, add_special_tokens=False)) >= args.input_len:
                break
        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": "ok"},
                ]
            }
        )

    args.output.write_text(json.dumps(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
