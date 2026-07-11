#!/usr/bin/env python3
"""Build the deterministic SGLang low-concurrency custom JSONL corpus.

The output shape is consumed by SGLang's pinned ``CustomDataset`` implementation
(``python/sglang/benchmark/datasets/custom.py:54-147`` at commit 28b095c).
Every row retains the canonical token IDs in addition to the two conversation
turns that the unmodified SGLang loader reads.
"""

from __future__ import annotations

import argparse
import pathlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tools.bench.serve_low_common import (
    HarnessError,
    SGLANG_COMMIT,
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class HfJsonTokenizer:
    """Thin runtime adapter; ``tokenizers`` is required only when invoked."""

    def __init__(self, tokenizer_json: pathlib.Path) -> None:
        try:
            from tokenizers import Tokenizer as RustTokenizer
        except ImportError as error:
            raise HarnessError(
                "tokenizers is required to generate a real corpus; run in the "
                "pinned client image or vLLM oracle environment"
            ) from error
        self._tokenizer = RustTokenizer.from_file(str(tokenizer_json))

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text).ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)


@dataclass(frozen=True)
class CorpusConfig:
    model_key: str
    tokenizer_revision: str
    seed: int = 0
    target_input_len: int = 1024
    output_len: int = 128
    requests_per_partition: int = 80
    warmup_requests: int = 80
    concurrencies: tuple[int, ...] = (1, 2, 4, 8, 16)
    repetitions: int = 3
    common_prefix_limit: int = 32


_WORDS = (
    "amber", "atlas", "birch", "cobalt", "delta", "ember", "fjord",
    "garden", "harbor", "indigo", "jasmine", "kernel", "lunar", "meadow",
    "nectar", "onyx", "prairie", "quartz", "river", "saffron", "timber",
    "umber", "violet", "willow", "xenon", "yarrow", "zephyr",
)


def _candidate_text(index: int, seed: int, minimum_words: int) -> str:
    rng = random.Random((seed << 32) ^ index ^ 0x5EED5EED)
    unique = f"request-{index:08x}-{rng.getrandbits(64):016x}"
    words = [unique]
    for position in range(minimum_words):
        word = _WORDS[rng.randrange(len(_WORDS))]
        words.append(f"{word}-{position % 97:02d}-{rng.randrange(1 << 16):04x}")
    return " ".join(words)


def _exact_prompt(tokenizer: Tokenizer, index: int, config: CorpusConfig) -> tuple[str, list[int]]:
    # Produce ample text, then decode prefixes of the tokenizer's own canonical
    # encoding. Re-encoding that text is the acceptance check; no requested
    # length or decode assumption is trusted.
    minimum_words = max(64, config.target_input_len)
    for expansion in range(8):
        text = _candidate_text(index, config.seed, minimum_words * (expansion + 1))
        encoded = tokenizer.encode(text)
        if len(encoded) < config.target_input_len:
            continue
        lower = max(1, config.target_input_len - 64)
        upper = min(len(encoded), config.target_input_len + 64)
        widths = [config.target_input_len]
        widths.extend(
            width
            for delta in range(1, 65)
            for width in (config.target_input_len - delta, config.target_input_len + delta)
            if lower <= width <= upper
        )
        for width in widths:
            prompt = tokenizer.decode(encoded[:width])
            verified = tokenizer.encode(prompt)
            if len(verified) == config.target_input_len:
                return prompt, verified
    raise HarnessError(
        f"could not construct an exact {config.target_input_len}-token prompt "
        f"for corpus row {index}"
    )


def _common_prefix(left: Sequence[int], right: Sequence[int], limit: int) -> int:
    length = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        length += 1
        if length > limit:
            break
    return length


def _validate_global_rows(rows: Sequence[dict], config: CorpusConfig) -> None:
    hashes = [row["prompt_sha256"] for row in rows]
    if len(set(hashes)) != len(hashes):
        raise HarnessError("corpus prompts are not globally disjoint")
    token_rows = sorted(tuple(row["prompt_token_ids"]) for row in rows)
    for left, right in zip(token_rows, token_rows[1:]):
        prefix = _common_prefix(left, right, config.common_prefix_limit)
        if prefix > config.common_prefix_limit:
            raise HarnessError(
                f"corpus common-token prefix {prefix} exceeds "
                f"limit {config.common_prefix_limit}"
            )


def generate_corpus(
    tokenizer: Tokenizer,
    output_dir: pathlib.Path,
    config: CorpusConfig,
    *,
    tokenizer_sha256: str,
) -> dict:
    if config.target_input_len < 2 or config.output_len < 4:
        raise HarnessError("input length must be >=2 and output length must be >=4")
    if config.requests_per_partition <= 0 or config.warmup_requests <= 0:
        raise HarnessError("partition sizes must be positive")
    if not config.concurrencies or any(value <= 0 for value in config.concurrencies):
        raise HarnessError("concurrencies must be non-empty and positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise HarnessError(f"refusing to mix corpus artifacts in non-empty {output_dir}")

    partitions = [("warmup", config.warmup_requests)]
    partitions.extend(
        (f"c{concurrency}-r{repetition}", config.requests_per_partition)
        for repetition in range(1, config.repetitions + 1)
        for concurrency in config.concurrencies
    )
    all_rows: list[dict] = []
    pending_files: list[tuple[str, str, list[dict]]] = []
    next_index = 0
    for partition, count in partitions:
        partition_rows = []
        for _ in range(count):
            prompt, token_ids = _exact_prompt(tokenizer, next_index, config)
            prompt_sha = sha256_bytes(prompt.encode("utf-8"))
            row = {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": "benchmark completion placeholder"},
                ],
                "index": next_index,
                "output_len": config.output_len,
                "partition": partition,
                "prompt_len": len(token_ids),
                "prompt_sha256": prompt_sha,
                "prompt_token_ids": token_ids,
                "tokenizer_revision": config.tokenizer_revision,
            }
            partition_rows.append(row)
            all_rows.append(row)
            next_index += 1
        filename = "warmup.jsonl" if partition == "warmup" else f"{partition}.jsonl"
        pending_files.append((filename, partition, partition_rows))

    # Validate the complete in-memory corpus before exposing any partition.
    _validate_global_rows(all_rows, config)
    files: list[dict] = []
    for filename, partition, partition_rows in pending_files:
        path = output_dir / filename
        write_jsonl_atomic(path, partition_rows)
        files.append(
            {
                "file": filename,
                "partition": partition,
                "requests": len(partition_rows),
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "common_prefix_limit": config.common_prefix_limit,
        "files": files,
        "format": "sglang-custom-conversations-jsonl-v1",
        "model_key": config.model_key,
        "output_len": config.output_len,
        "requests_per_partition": config.requests_per_partition,
        "seed": config.seed,
        "sglang_commit": SGLANG_COMMIT,
        "target_input_len": config.target_input_len,
        "tokenizer_revision": config.tokenizer_revision,
        "tokenizer_sha256": tokenizer_sha256,
        "total_prompts": len(all_rows),
        "warmup_requests": config.warmup_requests,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-json", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--requests-per-partition", type=int, default=80)
    parser.add_argument("--warmup-requests", type=int, default=80)
    parser.add_argument("--concurrencies", default="1,2,4,8,16")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--common-prefix-limit", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    concurrencies = tuple(int(item) for item in args.concurrencies.split(","))
    config = CorpusConfig(
        model_key=args.model_key,
        tokenizer_revision=args.tokenizer_revision,
        seed=args.seed,
        target_input_len=args.target_input_len,
        output_len=args.output_len,
        requests_per_partition=args.requests_per_partition,
        warmup_requests=args.warmup_requests,
        concurrencies=concurrencies,
        repetitions=args.repetitions,
        common_prefix_limit=args.common_prefix_limit,
    )
    tokenizer = HfJsonTokenizer(args.tokenizer_json)
    manifest = generate_corpus(
        tokenizer,
        args.out,
        config,
        tokenizer_sha256=sha256_file(args.tokenizer_json),
    )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
