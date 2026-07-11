"""Shared fail-closed helpers for the SGLang low-concurrency harness.

The benchmark behavior is pinned by
``.agents/specs/cuda-sglang-low-concurrency.md``.  This module deliberately
uses only the Python standard library so its artifact validators run in CPU CI
without installing either benchmark engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

SGLANG_COMMIT = "28b095c01005d4a3a2a5b637b7d028b07fba31b2"
SGLANG_IMAGE = (
    "docker.io/lmsysorg/sglang:v0.5.13-cu130-runtime@"
    "sha256:9631280f57d95503ed64cf3892de72190aafbfe6e58e90718a019fa775113bfb"
)
VLLM_COMMIT = "e24d1b24fe96a56ba8b0d653efa076d03eb95d6c"


class HarnessError(RuntimeError):
    """A benchmark precondition or artifact contract failed."""


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8-safe JSON used by all hashed artifacts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: pathlib.Path, text: str) -> None:
    """Atomically replace *path* without leaving partial evidence files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: pathlib.Path, value: Any) -> None:
    write_text_atomic(path, canonical_json(value) + "\n")


def write_jsonl_atomic(path: pathlib.Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_text_atomic(path, "".join(canonical_json(row) + "\n" for row in rows))


def read_jsonl(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise HarnessError(f"{path}:{line_number}: JSONL row is not an object")
            yield value


def require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HarnessError(f"{field} must be finite")
    return result


def percentile(values: Sequence[float], percent: float) -> float:
    """NumPy-compatible linear percentile without a NumPy dependency."""

    if not values:
        raise HarnessError("cannot calculate a percentile of an empty sequence")
    if not 0.0 <= percent <= 100.0:
        raise HarnessError(f"percentile outside [0, 100]: {percent}")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def coefficient_of_variation(values: Sequence[float]) -> float:
    if not values:
        raise HarnessError("cannot calculate variation of an empty sequence")
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0 if all(value == 0.0 for value in values) else math.inf
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / abs(mean)
