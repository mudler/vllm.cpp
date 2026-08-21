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
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

SGLANG_COMMIT = "28b095c01005d4a3a2a5b637b7d028b07fba31b2"
SGLANG_IMAGE = (
    "docker.io/lmsysorg/sglang:v0.5.13-cu130-runtime@"
    "sha256:9631280f57d95503ed64cf3892de72190aafbfe6e58e90718a019fa775113bfb"
)


class HarnessError(RuntimeError):
    """A benchmark precondition or artifact contract failed."""


# THE PARITY PIN IS READ FROM THE RECORD, NOT DUPLICATED HERE (#520).
#
# AGENTS.md: "Comparisons run against the pinned oracle recorded in
# .agents/upstream-sync.md."  These constants used to be a second copy of that
# record, and the copy drifted: the pin advanced to 555967922 on 2026-07-26 and
# the harness stayed at 0.25.0 / 0.6.13 / 702f4814 until 2026-08-12.  Because
# the check RAISES rather than defaults, the gate spent that window actively
# REFUSING the oracle the record required, so no measurement in it could name a
# compliant denominator even deliberately.  One reader, one record, no copy.
#
# The block is parsed, not the prose.  The pin paragraph names the release
# ("vLLM 0.26.0.dev0"); a running oracle reports 0.23.1rc1.dev1511+g555967922,
# and its distribution metadata appends ".precompiled" where the runtime string
# does not.  A parser over the prose would therefore have produced a constant no
# oracle can ever match -- more code AND wrong.  Only measured strings go in.
_PIN_RECORD = pathlib.Path(__file__).resolve().parents[2] / ".agents" / "upstream-sync.md"
_PIN_BLOCK_RE = re.compile(r"^```parity-pin$\n(.*?)^```$", re.MULTILINE | re.DOTALL)
_PIN_FIELDS = (
    "vllm_commit",
    "vllm_runtime_version",
    "vllm_distribution_version",
    "flashinfer_version",
)


def read_parity_pin(record: pathlib.Path | None = None) -> dict[str, str]:
    """Return the parity pin's exact runtime identity strings.

    Fails closed on every defect -- missing record, missing block, more than one
    block, an unparsable line, a missing or unknown key, a duplicate key.  The
    failure mode of this indirection must be a refusal to measure, never a
    silent default, because a default is exactly the #375 shape: an oracle that
    runs, looks healthy, and is not the one the record names.
    """

    path = _PIN_RECORD if record is None else record
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise HarnessError(f"parity pin record is unreadable: {path}: {error}") from error
    blocks = _PIN_BLOCK_RE.findall(text)
    if len(blocks) != 1:
        raise HarnessError(
            f"{path}: expected exactly one ```parity-pin block, found {len(blocks)}"
        )
    fields: dict[str, str] = {}
    for line in blocks[0].splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise HarnessError(f"{path}: malformed parity-pin line: {line!r}")
        if key not in _PIN_FIELDS:
            raise HarnessError(f"{path}: unknown parity-pin field {key!r}")
        if key in fields:
            raise HarnessError(f"{path}: duplicate parity-pin field {key!r}")
        fields[key] = value
    missing = [name for name in _PIN_FIELDS if name not in fields]
    if missing:
        raise HarnessError(f"{path}: parity-pin block omits {', '.join(missing)}")
    commit = fields["vllm_commit"]
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise HarnessError(f"{path}: vllm_commit must be a full 40-hex SHA, got {commit!r}")
    return fields


_PIN = read_parity_pin()
# The exact source commit of the pinned oracle -- the parity pin itself, no
# longer a separate "executable oracle" commit tracked apart from it.
VLLM_COMMIT = _PIN["vllm_commit"]
# What `vllm.__version__` reports, and what `importlib.metadata` reports.  They
# are DIFFERENT strings on the pin (the editable/precompiled build appends a
# suffix), so a check that equates them cannot pass at any single value.
VLLM_ORACLE_VERSION = _PIN["vllm_runtime_version"]
VLLM_DISTRIBUTION_VERSION = _PIN["vllm_distribution_version"]
FLASHINFER_VERSION = _PIN["flashinfer_version"]

_VERSION_COMMIT_RE = re.compile(r"\+g([0-9a-f]{7,40})")


def assert_oracle_commit(runtime_version: object) -> None:
    """Require *runtime_version* to name the pinned commit.

    A version number alone cannot tell the pin from the preserved rollback: the
    rollback reports a clean "0.25.0", runs, and is deterministic, so it is
    indistinguishable from a correct reference by every check that existed
    before #520.  What it cannot produce is the pin's `+g<sha>` local version
    segment.  Comparing a PREFIX of the recorded 40-hex SHA rather than a fixed
    abbreviation length keeps this true across git's variable auto-abbreviation
    (the pin abbreviates to nine).

    THIS IS DEFENCE IN DEPTH, NOT THE OPERATIVE TERM AT THIS PIN.  The first
    commit message of #520 overstated it as "the check #375 needed"; the
    correction lives here because that message cannot be rewritten.  At all
    three call sites an exact equality against `VLLM_ORACLE_VERSION` runs first,
    and today that constant already CONTAINS `+g555967922` -- so any string that
    passes the equality also passes this function, and it cannot fire in
    production.  What refuses the rollback today is the updated constant.  This
    assertion earns its place when the two come apart: a manifest read off disk
    from another venv or another day, a hand-edited evidence file, or a future
    pin whose recorded version is a plain release number.  `tests/tools/
    test_oracle_pin.py` proves the call sites exist by patching the version
    constant to a release-numbered shape, which is the only input that can reach
    this function while the equality holds.
    """

    text = "" if runtime_version is None else str(runtime_version)
    match = _VERSION_COMMIT_RE.search(text)
    if match is None or not VLLM_COMMIT.startswith(match.group(1)):
        raise HarnessError(
            "vLLM oracle commit drift: "
            f"runtime={text!r} does not name the pinned commit {VLLM_COMMIT!r}. "
            "A version string that merely LOOKS healthy is the #375 failure "
            "mode -- check which venv ~/venvs/vllm-oracle resolves to."
        )


# EVERY BENCHMARK SUBJECT, AND THE EXACT CHECKPOINT ITS KEY MEANS.
#
# A model key is a directory name -- `evidence/corpus/<key>/` and
# `evidence/raw/<key>/<engine>/` -- and `summarize_serve_low.py` aggregates by
# it.  So a key is a SUBJECT IDENTITY and never a size, and two checkpoints
# accepted under one key do not crash: they complete, write plausible raw
# results, and are averaged into one published number for a subject that never
# ran (#1594).  A key that names no checkpoint has the same shape, because
# nothing then says which weights produced the evidence under it.
#
# ONE record, read by every tool that keys on a subject, so that admitting a key
# and pinning its checkpoint cannot be separate edits and cannot drift apart.
# "27" and "27n" are DIFFERENT MODELS rather than two spellings of one, and
# "q38mtp" is a THIRD 27B checkpoint that is neither -- which is why it is
# deliberately not digit-prefixed, so no evidence path can be read as another
# subject's.
MODEL_CHECKPOINTS: dict[str, dict[str, str]] = {
    "27": {
        "repository": "unsloth/Qwen3.6-27B-NVFP4",
        "revision": "890bdef7a42feba6d83b6e17a03315c694112f2a",
    },
    "27n": {
        "repository": "nvidia/Qwen3.6-27B-NVFP4",
        "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    },
    "35": {
        "repository": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "revision": "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
    },
    "q3mxfp4": {
        "repository": "Yi30/Qwen3-8B-MXFP4",
        "revision": "b3e7ab32f7225ca779b3dbf6ef4ecefeb6de9b47",
    },
    # #1574's subject: `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`, a 27B NVFP4
    # checkpoint that is NOT the "27" subject.
    "q38mtp": {
        "repository": "r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121",
        "revision": "36f717a22990e82c54c1d48ee77c491b87825680",
    },
}


def model_checkpoint(model_key: str) -> dict[str, str]:
    """Return the repository and revision *model_key* names, or refuse.

    Fails closed on a key the registry does not carry, and on a registry entry
    that does not pin a full commit ID: a floating revision is a subject whose
    weights can be re-quantized in place under an unchanged name.
    """

    entry = MODEL_CHECKPOINTS.get(model_key)
    if entry is None:
        raise HarnessError(f"unknown model key: {model_key}")
    revision = entry["revision"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise HarnessError(
            f"model key {model_key} does not pin a full checkpoint revision"
        )
    return dict(entry)


def admitted_model_keys(*model_keys: str) -> tuple[str, ...]:
    """Return *model_keys* once each has been proven to name a checkpoint.

    A harness declares the subjects it benches by calling this, so a key cannot
    enter an admitted set -- an argparse `choices`, an evidence tree, a manifest
    -- without the registry saying which weights it means.
    """

    for model_key in model_keys:
        model_checkpoint(model_key)
    return model_keys


def require_model_checkpoint(model_key: str, revision: str, *, field: str) -> None:
    """Require *revision* to be the revision *model_key* names.

    This is the binding #1594 exists for: without it `--model-key` is a free
    label beside a free `--model-revision`, and the pair can disagree while the
    run exits 0.
    """

    expected = model_checkpoint(model_key)["revision"]
    if revision != expected:
        raise HarnessError(
            f"{field}={revision!r} is not the checkpoint model key "
            f"{model_key!r} names ({expected})"
        )


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


def _require_count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessError(f"{field} must be an integer request count")
    if value < 0:
        raise HarnessError(f"{field} must not be negative")
    return value


def require_complete_request_set(
    record: Mapping[str, Any],
    *,
    expected_requests: int | None = None,
    source: str = "benchmark result",
) -> int:
    """Refuse to derive any rate from a request set that is not whole (#931).

    A benchmark client divides tokens by the leg's WALL duration, and that
    duration still contains every second a dead request spent before it failed.
    The Qwen3.8-27B c1 leg read 0.675x output throughput against vLLM while
    median TPOT in the same file read 1.017x in OUR favour; the difference was
    one request of six that never completed. The number was not noisy, it was
    wrong, and it was wrong in a direction nobody can predict from the record.

    So this is a precondition of DERIVING a number, not a property of a
    validator that some caller may or may not have run first. Call it at the
    site that computes the rate.

    Returns the completed count so the caller can use it as the numerator
    instead of an assumed request count.
    """

    completed = _require_count(record.get("completed"), f"{source}: completed")

    declared = record.get("num_prompts")
    if declared is None:
        expected = expected_requests
    else:
        expected = _require_count(declared, f"{source}: num_prompts")
        if expected_requests is not None and expected_requests != expected:
            raise HarnessError(
                f"{source}: num_prompts={expected} contradicts the expected "
                f"request count {expected_requests}"
            )
    if expected is None:
        raise HarnessError(
            f"{source}: cannot establish that the request set is complete -- the "
            "record declares no num_prompts and the caller named no expected "
            "count. An unprovable request set is not a complete one."
        )

    # `failed` is absent from the SGLang schema and present in the pinned vLLM
    # client's. When it is there, an unexamined non-zero value is exactly how
    # #931 stayed invisible, so it voids the leg on its own.
    failed = record.get("failed")
    failed_count = 0 if failed is None else _require_count(failed, f"{source}: failed")
    if completed != expected or failed_count != 0:
        raise HarnessError(
            f"{source}: request set is partial: completed={completed!r}, "
            f"failed={failed_count!r}, expected={expected!r}. No throughput, "
            "latency or memory number may be derived from it."
        )

    errors = record.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        reported = [str(value) for value in errors if value]
        if reported:
            raise HarnessError(
                f"{source}: the request set records {len(reported)} request "
                f"error(s); first: {reported[0][:200]!r}"
            )
    return completed


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
