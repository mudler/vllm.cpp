#!/usr/bin/env python3
"""Orchestrate and validate the pinned SGLang low-concurrency client.

Timed requests are always executed by the unmodified
``sglang.bench_serving`` module at commit 28b095c.  The local wrapper owns only
command construction, fail-closed artifact validation, and untimed native-ID /
OpenAI serving preflights.  Its SSE parsing follows the pinned OpenAI request
function at ``python/sglang/bench_serving.py:232-344``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import http.client
import json
import os
import pathlib
import platform
import subprocess
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any

from tools.bench.serve_low_common import (
    HarnessError,
    SGLANG_COMMIT,
    SGLANG_IMAGE,
    VLLM_COMMIT,
    canonical_json,
    read_jsonl,
    require_number,
    write_json_atomic,
)


@dataclasses.dataclass(frozen=True)
class HttpResult:
    status: int
    body: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class StreamResult:
    emitted_chunks: int
    first_chunk_s: float
    total_s: float
    generated_text: str

    @property
    def spread_s(self) -> float:
        return max(0.0, self.total_s - self.first_chunk_s)


@dataclasses.dataclass(frozen=True)
class BenchRun:
    image: str
    model_repo: pathlib.Path
    model_revision: str
    evidence_root: pathlib.Path
    model_key: str
    engine: str
    base_url: str
    concurrency: int
    repetition: int
    num_prompts: int = 80
    output_len: int = 128

    @property
    def corpus_path(self) -> pathlib.Path:
        return self.evidence_root / "corpus" / self.model_key / (
            f"c{self.concurrency}-r{self.repetition}.jsonl"
        )

    @property
    def output_path(self) -> pathlib.Path:
        return self.evidence_root / "raw" / self.model_key / self.engine / (
            f"c{self.concurrency}-r{self.repetition}.jsonl"
        )


def _connection(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HarnessError(f"unsupported URL: {url}")
    cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = cls(parsed.hostname, parsed.port, timeout=timeout)
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    return connection, path


def post_json(url: str, payload: Mapping[str, Any], timeout: float = 60.0) -> HttpResult:
    connection, path = _connection(url, timeout)
    body = canonical_json(dict(payload)).encode("utf-8")
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read()
        try:
            value = json.loads(raw or b"{}")
        except json.JSONDecodeError as error:
            raise HarnessError(
                f"{url}: HTTP {response.status} returned invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise HarnessError(f"{url}: response is not a JSON object")
        return HttpResult(response.status, value)
    finally:
        connection.close()


def capture_native_output_ids(
    url: str,
    input_ids: Sequence[int],
    *,
    expected_output_len: int = 128,
    oracle_output_ids: Sequence[int] | None = None,
    timeout: float = 900.0,
) -> list[int]:
    result = post_json(
        url,
        {
            "input_ids": list(input_ids),
            "sampling_params": {
                "ignore_eos": True,
                "max_new_tokens": expected_output_len,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "stream": False,
        },
        timeout,
    )
    if result.status != 200:
        raise HarnessError(f"native preflight returned HTTP {result.status}: {result.body}")
    output_ids = result.body.get("output_ids")
    if not isinstance(output_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in output_ids
    ):
        raise HarnessError("native preflight did not return integer output_ids")
    if len(output_ids) != expected_output_len:
        raise HarnessError(
            f"native preflight returned {len(output_ids)} output IDs; "
            f"expected {expected_output_len}"
        )
    if oracle_output_ids is not None and list(oracle_output_ids) != output_ids:
        if len(oracle_output_ids) != len(output_ids):
            raise HarnessError(
                f"native output-ID count {len(output_ids)} differs from oracle "
                f"count {len(oracle_output_ids)}"
            )
        mismatch = next(
            index
            for index, (want, got) in enumerate(zip(oracle_output_ids, output_ids))
            if want != got
        )
        raise HarnessError(
            f"native output IDs differ from the oracle at token {mismatch}: "
            f"{output_ids[mismatch]} != {oracle_output_ids[mismatch]}"
        )
    return output_ids


def validate_openai_usage(
    result: HttpResult,
    *,
    prompt_tokens: int = 1024,
    completion_tokens: int = 128,
) -> dict[str, Any]:
    if result.status != 200:
        raise HarnessError(f"OpenAI preflight returned HTTP {result.status}: {result.body}")
    choices = result.body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise HarnessError("OpenAI preflight requires exactly one choice")
    if choices[0].get("finish_reason") != "length":
        raise HarnessError("OpenAI preflight finish_reason is not 'length'")
    usage = result.body.get("usage")
    expected = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if not isinstance(usage, dict) or any(
        usage.get(key) != value for key, value in expected.items()
    ):
        raise HarnessError(f"OpenAI usage mismatch: got {usage!r}, expected {expected!r}")
    return result.body


def openai_usage_preflight(
    url: str,
    prompt: str,
    *,
    model: str = "gate",
    prompt_tokens: int = 1024,
    completion_tokens: int = 128,
    timeout: float = 900.0,
) -> dict[str, Any]:
    result = post_json(
        url,
        {
            "ignore_eos": True,
            "max_tokens": completion_tokens,
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        timeout,
    )
    return validate_openai_usage(
        result,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def openai_stream_probe(
    url: str,
    prompt: str,
    *,
    model: str = "gate",
    completion_tokens: int = 128,
    minimum_spread_s: float = 0.0,
    timeout: float = 900.0,
) -> StreamResult:
    connection, path = _connection(url, timeout)
    payload = canonical_json(
        {
            "ignore_eos": True,
            "max_tokens": completion_tokens,
            "model": model,
            "prompt": prompt,
            "stream": True,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    ).encode("utf-8")
    start = time.monotonic()
    chunks = 0
    first: float | None = None
    generated: list[str] = []
    try:
        connection.request(
            "POST",
            path,
            body=payload,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            body = response.read().decode("utf-8", errors="replace")
            raise HarnessError(f"stream probe returned HTTP {response.status}: {body}")
        event_data: list[str] = []
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                event_data.append(line[5:].lstrip())
                continue
            if line or not event_data:
                continue
            data = "\n".join(event_data)
            event_data.clear()
            if data == "[DONE]":
                continue
            try:
                value = json.loads(data)
                text = value["choices"][0]["text"]
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                raise HarnessError(f"malformed completion SSE event: {data!r}") from error
            if text:
                now = time.monotonic()
                first = now if first is None else first
                chunks += 1
                generated.append(text)
        end = time.monotonic()
    finally:
        connection.close()
    if first is None:
        raise HarnessError("stream probe returned no token-bearing SSE event")
    result = StreamResult(chunks, first - start, end - start, "".join(generated))
    if result.emitted_chunks != completion_tokens:
        raise HarnessError(
            f"stream probe emitted {result.emitted_chunks} token chunks; "
            f"expected {completion_tokens}"
        )
    if not result.first_chunk_s < result.total_s:
        raise HarnessError("stream probe first chunk did not precede completion")
    if result.spread_s < minimum_spread_s:
        raise HarnessError(
            f"stream probe spread {result.spread_s:.6f}s is below "
            f"the required {minimum_spread_s:.6f}s"
        )
    return result


def run_usage_batch(
    url: str,
    prompts: Sequence[str],
    *,
    max_concurrency: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> list[dict[str, Any]]:
    if max_concurrency <= 0:
        raise HarnessError("max_concurrency must be positive")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [
            pool.submit(
                openai_usage_preflight,
                url,
                prompt,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            for prompt in prompts
        ]
        return [future.result() for future in futures]


def build_bench_command(run: BenchRun) -> list[str]:
    if run.image != SGLANG_IMAGE:
        raise HarnessError("benchmark image does not match the digest pin")
    if run.engine not in {"ours", "sglang", "vllm"}:
        raise HarnessError(f"unknown engine arm: {run.engine}")
    if run.concurrency not in {1, 2, 4, 8, 16} or run.repetition not in {1, 2, 3}:
        raise HarnessError("run is outside the canonical concurrency/repetition grid")
    if not run.corpus_path.is_file():
        raise HarnessError(f"missing corpus partition: {run.corpus_path}")
    if not run.model_repo.is_dir():
        raise HarnessError(f"missing model repository: {run.model_repo}")
    backend = "sglang-oai" if run.engine == "sglang" else "vllm"
    container_corpus = f"/evidence/corpus/{run.model_key}/{run.corpus_path.name}"
    container_output = (
        f"/evidence/raw/{run.model_key}/{run.engine}/{run.output_path.name}"
    )
    return [
        "docker", "run", "--rm", "--network=host", "--pull=never",
        "--mount", f"type=bind,src={run.model_repo},dst=/models/gate,readonly",
        "--mount", f"type=bind,src={run.evidence_root},dst=/evidence",
        run.image,
        "python3", "-m", "sglang.bench_serving",
        "--backend", backend,
        "--base-url", run.base_url,
        "--model", "gate",
        "--tokenizer", f"/models/gate/snapshots/{run.model_revision}",
        "--dataset-name", "custom",
        "--dataset-path", container_corpus,
        "--num-prompts", str(run.num_prompts),
        "--sharegpt-output-len", str(run.output_len),
        "--request-rate", "inf",
        "--max-concurrency", str(run.concurrency),
        "--warmup-requests", "0",
        "--seed", "0",
        "--temperature", "0",
        "--top-p", "1",
        "--extra-request-body", '{"ignore_eos":true}',
        "--output-details",
        "--output-file", container_output,
    ]


def validate_raw_result(
    record: Mapping[str, Any],
    *,
    expected_requests: int = 80,
    prompt_len: int = 1024,
    output_len: int = 128,
    max_concurrency: int | None = None,
) -> None:
    if record.get("completed") != expected_requests:
        raise HarnessError(
            f"completed={record.get('completed')!r}; expected {expected_requests}"
        )
    input_lens = record.get("input_lens")
    output_lens = record.get("output_lens")
    errors = record.get("errors")
    generated_texts = record.get("generated_texts")
    ttfts = record.get("ttfts")
    itls = record.get("itls")
    if not all(
        isinstance(value, list)
        for value in (input_lens, output_lens, errors, generated_texts, ttfts, itls)
    ):
        raise HarnessError("raw result is missing --output-details arrays")
    if len(input_lens) != expected_requests or any(value != prompt_len for value in input_lens):
        raise HarnessError("raw result input lengths do not match the corpus")
    if len(output_lens) != expected_requests or any(value != output_len for value in output_lens):
        raise HarnessError("raw result output lengths are not exact")
    if len(errors) != expected_requests or any(value for value in errors):
        raise HarnessError("raw result contains request errors")
    if len(generated_texts) != expected_requests or any(
        not isinstance(value, str) for value in generated_texts
    ):
        raise HarnessError("raw result did not retain every generated text")
    if len(ttfts) != expected_requests or len(itls) != expected_requests:
        raise HarnessError("raw result timing arrays have the wrong cardinality")
    if any(len(row) != output_len - 1 for row in itls):
        raise HarnessError("raw result ITL arrays do not contain output_len-1 intervals")
    for index, value in enumerate(ttfts):
        if require_number(value, f"ttfts[{index}]") < 0.0:
            raise HarnessError("raw result contains a negative TTFT")
    for row_index, row in enumerate(itls):
        for value_index, value in enumerate(row):
            if require_number(value, f"itls[{row_index}][{value_index}]") < 0.0:
                raise HarnessError("raw result contains a negative ITL")
    if max_concurrency is not None:
        achieved = record.get("max_concurrent_requests")
        if isinstance(achieved, bool) or not isinstance(achieved, int):
            raise HarnessError("raw result has no integer max_concurrent_requests")
        if not 1 <= achieved <= max_concurrency:
            raise HarnessError(
                f"raw peak concurrency {achieved} is outside [1, {max_concurrency}]"
            )


def run_benchmark(run: BenchRun) -> dict[str, Any]:
    command = build_bench_command(run)
    run.output_path.parent.mkdir(parents=True, exist_ok=True)
    if run.output_path.exists():
        raise HarnessError(f"refusing to append to existing raw result: {run.output_path}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise HarnessError(f"pinned benchmark client exited {completed.returncode}")
    records = list(read_jsonl(run.output_path))
    if len(records) != 1:
        raise HarnessError(f"expected one raw result row, found {len(records)}")
    validate_raw_result(
        records[0],
        expected_requests=run.num_prompts,
        output_len=run.output_len,
        max_concurrency=run.concurrency,
    )
    return records[0]


def build_dry_run_manifest(
    *,
    claim_root: pathlib.Path,
    vllm_cpp_sha: str,
    image: str,
) -> dict[str, Any]:
    if image != SGLANG_IMAGE:
        raise HarnessError("dry-run image does not match the digest pin")
    if len(vllm_cpp_sha) != 40 or any(char not in "0123456789abcdef" for char in vllm_cpp_sha):
        raise HarnessError("vllm.cpp SHA must be a full lowercase commit ID")
    return {
        "artifact_root": str(claim_root / "evidence" / vllm_cpp_sha),
        "concurrencies": [1, 2, 4, 8, 16],
        "dry_run": True,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_lock_acquisitions_planned": 1,
        "host": {
            "kernel": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
            "python": platform.python_version(),
        },
        "image": image,
        "models": {
            "27": {
                "repository": "unsloth/Qwen3.6-27B-NVFP4",
                "revision": "890bdef7a42feba6d83b6e17a03315c694112f2a",
            },
            "35": {
                "repository": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                "revision": "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
            },
        },
        "pending_preconditions": [
            "host_idle_proof",
            "image_platform_digest_and_revision",
            "checkpoint_file_manifests",
            "native_output_id_parity",
            "incremental_streaming",
        ],
        "planned_artifacts": [
            "manifest.json", "provision/", "corpus/", "preflight/", "raw/",
            "memory/", "logs/", "nsys/", "summary/",
        ],
        "planned_commands": {
            "client": [
                "docker", "run", "--rm", "--network=host", "--pull=never",
                "--mount", "type=bind,src=<MODEL_REPO>,dst=/models/gate,readonly",
                "--mount", "type=bind,src=<EVIDENCE>,dst=/evidence",
                image,
                "python3", "-m", "sglang.bench_serving",
                "--backend", "<sglang-oai|vllm>",
                "--base-url", "http://127.0.0.1:30000",
                "--model", "gate",
                "--tokenizer", "/models/gate/snapshots/<MODEL_REV>",
                "--dataset-name", "custom",
                "--dataset-path", "/evidence/corpus/<MODEL>/c<C>-r<REP>.jsonl",
                "--num-prompts", "80",
                "--sharegpt-output-len", "128",
                "--request-rate", "inf",
                "--max-concurrency", "<1|2|4|8|16>",
                "--warmup-requests", "0",
                "--seed", "0",
                "--temperature", "0",
                "--top-p", "1",
                "--extra-request-body", '{"ignore_eos":true}',
                "--output-details",
                "--output-file", "/evidence/raw/<MODEL>/<ENGINE>/c<C>-r<REP>.jsonl",
            ],
            "execution_lock": [
                "flock", "/tmp/gpu", "<one-whole-P2-or-P3-campaign-command>"
            ],
            "ours_server": [
                "<VLLM_CPP_BUILD>/examples/server",
                "--model", "<MODEL_SNAPSHOT>",
                "--served-model-name", "gate",
                "--host", "127.0.0.1", "--port", "30000",
                "--block-size", "32", "--num-blocks", "640",
                "--max-model-len", "1152",
            ],
            "provision": ["docker", "pull", image],
            "sglang_server": [
                "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
                "--network=host", "--pull=never", image,
                "python3", "-m", "sglang.launch_server",
                "--model-path", "/models/gate/snapshots/<MODEL_REV>",
                "--served-model-name", "gate",
                "--host", "127.0.0.1", "--port", "30000",
                "--language-only", "--context-length", "1152",
                "--max-running-requests", "16", "--max-total-tokens", "20480",
                "--mem-fraction-static", "0.85", "--stream-interval", "1",
            ],
            "vllm_server": [
                "<VLLM_ORACLE>/bin/vllm", "serve", "<MODEL_SNAPSHOT>",
                "--served-model-name", "gate",
                "--host", "127.0.0.1", "--port", "30000",
                "--language-model-only", "--max-model-len", "1152",
                "--max-num-seqs", "16", "--block-size", "32",
                "--num-gpu-blocks-override", "640",
                "--gpu-memory-utilization", "0.85", "--stream-interval", "1",
            ],
        },
        "pull_under_gpu_lock": False,
        "repetitions": 3,
        "sglang_commit": SGLANG_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "vllm_cpp_sha": vllm_cpp_sha,
    }


def _load_id_list(path: pathlib.Path, field: str) -> list[int]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HarnessError(f"{path}: invalid JSON: {error}") from error
    if isinstance(value, dict):
        value = value.get(field)
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise HarnessError(f"{path}: expected an integer {field} array")
    return value


def _load_corpus_prompts(path: pathlib.Path, limit: int) -> list[str]:
    if limit <= 0:
        raise HarnessError("prompt limit must be positive")
    prompts = []
    for row in read_jsonl(path):
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise HarnessError(f"{path}: row has no conversations")
        first = conversations[0]
        if not isinstance(first, dict):
            raise HarnessError(f"{path}: first conversation is not an object")
        prompt = first.get("content", first.get("value"))
        if not isinstance(prompt, str):
            raise HarnessError(f"{path}: first conversation has no text")
        prompts.append(prompt)
        if len(prompts) == limit:
            break
    if len(prompts) != limit:
        raise HarnessError(f"{path}: found {len(prompts)} prompts; expected {limit}")
    return prompts


def capture_diagnostic_error_body(
    url: str,
    corpus: pathlib.Path,
    output: pathlib.Path,
    *,
    output_len: int = 128,
    model: str = "gate",
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Replay corpus row 0 as a non-streaming completion and persist the body.

    Bounded ``--diagnostic-c16`` reproduction: unlike every timed path this
    keeps the raw response even when the server answers HTTP 500, so the
    root-cause error body of the packed c16 boundary is captured verbatim.
    """

    if output.exists():
        raise HarnessError(f"refusing to overwrite diagnostic error body: {output}")
    prompt = _load_corpus_prompts(corpus, 1)[0]
    result = post_json(
        url,
        {
            "ignore_eos": True,
            "max_tokens": output_len,
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        timeout,
    )
    record = {
        "status": result.status,
        "body": result.body,
        "diagnostic": True,
        "mode": "diagnostic-c16",
    }
    write_json_atomic(output, record)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-raw")
    validate.add_argument("path", type=pathlib.Path)
    validate.add_argument("--requests", type=int, default=80)
    validate.add_argument("--prompt-len", type=int, default=1024)
    validate.add_argument("--output-len", type=int, default=128)
    validate.add_argument("--max-concurrency", type=int)

    bench = subparsers.add_parser("bench")
    bench.add_argument("--image", default=SGLANG_IMAGE)
    bench.add_argument("--model-repo", type=pathlib.Path, required=True)
    bench.add_argument("--model-revision", required=True)
    bench.add_argument("--evidence", type=pathlib.Path, required=True)
    bench.add_argument("--model-key", choices=("27", "35"), required=True)
    bench.add_argument("--engine", choices=("ours", "vllm", "sglang"), required=True)
    bench.add_argument("--base-url", required=True)
    bench.add_argument("--concurrency", type=int, required=True)
    bench.add_argument("--repetition", type=int, required=True)

    native = subparsers.add_parser("native")
    native.add_argument("--url", required=True)
    native.add_argument("--input-ids", type=pathlib.Path, required=True)
    native.add_argument("--oracle-output-ids", type=pathlib.Path)
    native.add_argument("--output-len", type=int, default=128)
    native.add_argument("--output", type=pathlib.Path, required=True)

    usage = subparsers.add_parser("usage")
    usage.add_argument("--url", required=True)
    usage.add_argument("--corpus", type=pathlib.Path, required=True)
    usage.add_argument("--requests", type=int, default=16)
    usage.add_argument("--max-concurrency", type=int, default=16)
    usage.add_argument("--prompt-len", type=int, default=1024)
    usage.add_argument("--output-len", type=int, default=128)
    usage.add_argument("--output", type=pathlib.Path, required=True)

    stream = subparsers.add_parser("stream")
    stream.add_argument("--url", required=True)
    stream.add_argument("--corpus", type=pathlib.Path, required=True)
    stream.add_argument("--output-len", type=int, default=128)
    stream.add_argument("--minimum-spread", type=float, default=0.05)
    stream.add_argument("--output", type=pathlib.Path, required=True)

    diagnostic = subparsers.add_parser("diagnostic-error-body")
    diagnostic.add_argument("--url", required=True)
    diagnostic.add_argument("--corpus", type=pathlib.Path, required=True)
    diagnostic.add_argument("--output-len", type=int, default=128)
    diagnostic.add_argument("--output", type=pathlib.Path, required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--claim-root", type=pathlib.Path, required=True)
    plan.add_argument("--vllm-cpp-sha", required=True)
    plan.add_argument("--image", default=SGLANG_IMAGE)
    plan.add_argument("--output", type=pathlib.Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-raw":
        records = list(read_jsonl(args.path))
        if len(records) != 1:
            raise HarnessError(f"expected one raw result row, found {len(records)}")
        validate_raw_result(
            records[0],
            expected_requests=args.requests,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            max_concurrency=args.max_concurrency,
        )
        return 0
    if args.command == "bench":
        result = run_benchmark(
            BenchRun(
                image=args.image,
                model_repo=args.model_repo,
                model_revision=args.model_revision,
                evidence_root=args.evidence,
                model_key=args.model_key,
                engine=args.engine,
                base_url=args.base_url,
                concurrency=args.concurrency,
                repetition=args.repetition,
            )
        )
    elif args.command == "native":
        if args.output.exists():
            raise HarnessError(f"refusing to overwrite preflight output: {args.output}")
        input_ids = _load_id_list(args.input_ids, "input_ids")
        oracle = (
            _load_id_list(args.oracle_output_ids, "output_ids")
            if args.oracle_output_ids
            else None
        )
        output_ids = capture_native_output_ids(
            args.url,
            input_ids,
            expected_output_len=args.output_len,
            oracle_output_ids=oracle,
        )
        result = {"input_ids": input_ids, "output_ids": output_ids}
        write_json_atomic(args.output, result)
    elif args.command == "usage":
        if args.output.exists():
            raise HarnessError(f"refusing to overwrite preflight output: {args.output}")
        prompts = _load_corpus_prompts(args.corpus, args.requests)
        responses = run_usage_batch(
            args.url,
            prompts,
            max_concurrency=args.max_concurrency,
            prompt_tokens=args.prompt_len,
            completion_tokens=args.output_len,
        )
        result = {"responses": responses}
        write_json_atomic(args.output, result)
    elif args.command == "stream":
        if args.output.exists():
            raise HarnessError(f"refusing to overwrite preflight output: {args.output}")
        prompt = _load_corpus_prompts(args.corpus, 1)[0]
        stream_result = openai_stream_probe(
            args.url,
            prompt,
            completion_tokens=args.output_len,
            minimum_spread_s=args.minimum_spread,
        )
        result = dataclasses.asdict(stream_result)
        write_json_atomic(args.output, result)
    elif args.command == "diagnostic-error-body":
        result = capture_diagnostic_error_body(
            args.url,
            args.corpus,
            args.output,
            output_len=args.output_len,
        )
    else:
        if args.output.exists():
            raise HarnessError(f"refusing to overwrite dry-run manifest: {args.output}")
        if args.output.parent.exists() and any(args.output.parent.iterdir()):
            raise HarnessError(
                f"refusing to mix dry-run artifacts in {args.output.parent}"
            )
        result = build_dry_run_manifest(
            claim_root=args.claim_root,
            vllm_cpp_sha=args.vllm_cpp_sha,
            image=args.image,
        )
        write_json_atomic(args.output, result)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"serve-low: {error}", file=os.sys.stderr)
        raise SystemExit(2) from error
