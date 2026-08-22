#!/usr/bin/env python3
"""Capture OUR side of the SPEC-DFLASH2 speed pairing, through the public ABI.

The oracle arm lives in `tools/bench/dflash2_oracle_capture.py`. This is the
numerator, and it is deliberately a THIN CLIENT of `examples/cli` (`vllm-cli`),
which is itself a thin client of `include/vllm.h`: `AGENTS.md` §Shared seams
requires every shipped capability to be reachable through the ABI, and a
benchmark that reached around it would measure a path no user has.

`vllm-cli` already reports what a throughput axis needs, one line per repetition
on stderr:

    vllm-cli: run=2/5 finish_reason=length prompt_tokens=5 completion_tokens=64 secs=1.234 tok_s=51.863

`--repeat` loads the model ONCE and completes N times, so the legs after the
first are warm. This harness discards leg 1 for that NAMED CAUSE and no other:
`.agents/benchmarking.md` allows discarding a cold leg for a named cause and
never because it is inconvenient, so the discard is recorded in the output
rather than applied silently, and every leg is kept in `legs`.

What it does NOT claim
----------------------
`vllm-cli` reports wall time and completion tokens, so this arm produces exactly
one axis: `output_throughput_tok_s`, the median over the warm legs.

It reports no time to first token and no peak device allocation, so this harness
records neither and `dflash2_speed_harness.axis_rows` renders both `NOT
MEASURED`. `tpot_ms` is left unmeasured for a different reason and deliberately:
wall time over completion tokens is `output_throughput_tok_s` inverted, so
emitting it would fill an axis with the axis above it and hide the fact that the
per-token latency, which excludes prefill, was never measured at all. An
aggregate cannot support a per-request claim, and three of the four axes are
open gaps that a plausible-looking number would close on paper only.

    python3 -m tools.bench.dflash2_our_arm \\
        --binary /workspace/build/bin/vllm-cli \\
        --model /workspace/dflash2/target-gguf \\
        --speculative-config '{"model":"/workspace/dflash2/draft.gguf","num_speculative_tokens":7}' \\
        --repeat 5 --max-tokens 64 \\
        --artifact target=/workspace/dflash2/target.gguf=<sha256> \\
        --lease-id "$RC_LEASE_ID" --our-revision "$(git rev-parse HEAD)" \\
        --our-build-recipe "cmake --preset cuda-release" \\
        --clock-summary evidence/clock-ours.json \\
        --output evidence/our-arm.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from tools.bench.dflash2_oracle_capture import (
    PROMPTS,
    _load_clock,
    _parse_artifact,
    sample_compute_processes,
)
from tools.bench.dflash2_speed_harness import (
    SSE_PING_ENV,
    build_recipe_reasons,
    checkpoint_reasons,
    clock_state_reasons,
    contention_reasons,
    fold_legs,
    leg_reasons,
    model_binding_reasons,
    repeat_reasons,
    require_no_reasons,
    speculative_config_reasons,
    sse_keepalive_reasons,
    workload_fingerprint,
)

#: Re-exported so `our_arm.leg_reasons` and `our_arm.fold_legs` keep naming the
#: rule this arm applies. The rule itself lives in `dflash2_speed_harness`
#: because the ORACLE ARM APPLIES THE SAME ONE: a ratio between a median folded
#: here and a median folded there is a ratio between two statistics.
__all__ = ["fold_legs", "leg_reasons", "parse_legs", "main"]
from tools.bench.serve_low_common import HarnessError, canonical_json, write_json_atomic

#: Positional against `examples/cli/main.cpp`'s own format string. A rename
#: upstream of this regex produces ZERO matches, and zero legs is a refusal
#: below -- never an empty average that reads as a measurement.
LEG_RE = re.compile(
    r"run=(?P<run>\d+)/(?P<of>\d+)\s+finish_reason=(?P<finish>\S+)\s+"
    r"prompt_tokens=(?P<prompt>\d+)\s+completion_tokens=(?P<completion>\d+)\s+"
    r"secs=(?P<secs>[0-9.]+)\s+tok_s=(?P<tok_s>[0-9.]+)"
)


def parse_legs(stderr_text: str) -> list[dict[str, Any]]:
    """Every timing line `vllm-cli` printed, in order."""

    legs: list[dict[str, Any]] = []
    for match in LEG_RE.finditer(stderr_text):
        legs.append(
            {
                "run": int(match.group("run")),
                "finish_reason": match.group("finish"),
                "prompt_tokens": int(match.group("prompt")),
                "completion_tokens": int(match.group("completion")),
                "secs": float(match.group("secs")),
                "tok_s": float(match.group("tok_s")),
            }
        )
    return legs


def run_binary(args: argparse.Namespace, prompt: str) -> str:
    command = [
        args.binary,
        "--model",
        args.model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        "0",
        "--repeat",
        str(args.repeat),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if args.speculative_config:
        command += ["--speculative-config", args.speculative_config]
    if args.device:
        command += ["--device", args.device]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise HarnessError(f"cannot run {args.binary!r}: {error}") from error
    if completed.returncode != 0:
        raise HarnessError(
            f"{args.binary} exited {completed.returncode}; the last stderr line was "
            f"{completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else '(empty)'!r}"
        )
    return completed.stderr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="capture OUR arm of the DFlash2 speed pairing")
    parser.add_argument("--binary", required=True, help="the built vllm-cli, a pure ABI client")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--speculative-config",
        default="",
        help="the drafter JSON `vllm-cli` reads. REQUIRED: this row measures a "
        "speculative decode against a speculative decode, and an absent config "
        "runs a plain one that would still fingerprint-match the oracle's k",
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=7,
        help="the k this run CLAIMS. It is checked against the k inside "
        "--speculative-config, which is the one the binary reads, and a "
        "disagreement is a refusal naming both",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifact", action="append", default=[], metavar="ROLE=PATH=SHA256")
    parser.add_argument("--lease-id", default="")
    parser.add_argument("--our-revision", default="")
    parser.add_argument("--our-build-recipe", default="")
    parser.add_argument("--clock-summary", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument("--assume-compute-processes", type=json.loads, default=None)
    return parser


def precheck(args: argparse.Namespace, env: Mapping[str, str]) -> dict[str, Any]:
    artifacts = [_parse_artifact(item) for item in args.artifact]
    contention = {
        "lease_id": args.lease_id,
        "own_pids": [os.getpid()],
        "compute_processes": (
            list(args.assume_compute_processes)
            if args.assume_compute_processes is not None
            else sample_compute_processes()
        ),
    }
    build = {"revision": args.our_revision, "build_recipe": args.our_build_recipe}
    prompts = list(args.prompt) or list(PROMPTS)
    # THE CONFIG IS THE SOURCE OF TRUTH FOR k, NOT THE FLAG. `--speculative-config`
    # is what `vllm-cli` reads; `--num-speculative-tokens` was a separate integer
    # that went straight into the workload fingerprint, so the record could read
    # k=7 over a binary configured at k=3 -- and, while the config was optional,
    # over a binary running no speculative decoding at all.
    spec_reasons, spec_config = speculative_config_reasons(
        args.speculative_config, args.num_speculative_tokens, label="ours"
    )
    configured_k = spec_config.get("num_speculative_tokens", args.num_speculative_tokens)
    models = {"model": args.model}
    drafter = spec_config.get("model")
    if drafter:
        models["drafter"] = str(drafter)
    reasons: list[str] = []
    reasons += sse_keepalive_reasons(env)
    reasons += checkpoint_reasons(artifacts)
    reasons += contention_reasons(contention)
    reasons += model_binding_reasons(models, artifacts, label="ours")
    reasons += repeat_reasons(args.repeat, label="ours")
    reasons += build_recipe_reasons(build, label="ours")
    reasons += spec_reasons
    require_no_reasons(reasons, what="DFlash2 our-arm capture")
    return {
        "artifacts": artifacts,
        "contention": contention,
        "build": build,
        "models": models,
        "speculative_config": spec_config,
        "env": {SSE_PING_ENV: env[SSE_PING_ENV]},
        "prompts": prompts,
        "workload": {
            "prompts_sha256": workload_fingerprint(prompts),
            "num_prompts": len(prompts),
            "max_tokens": args.max_tokens,
            # RECORDED FROM THE CONFIG THE BINARY READ.
            "num_speculative_tokens": int(configured_k),
            "max_num_seqs": args.max_num_seqs,
            "repeat": args.repeat,
            "concurrency": 1,
            "temperature": 0.0,
            "seed": None,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)
    checked = precheck(args, env)
    if args.precheck_only:
        print(canonical_json({"precheck": "PASS", **checked}))
        return 0

    clock = _load_clock(args.clock_summary)
    require_no_reasons(clock_state_reasons(clock, label="ours"), what="DFlash2 our-arm capture")

    legs: list[dict[str, Any]] = []
    for prompt in checked["prompts"]:
        legs += parse_legs(run_binary(args, prompt))
    require_no_reasons(
        leg_reasons(legs, max_tokens=args.max_tokens), what="DFlash2 our-arm capture"
    )
    record = {**checked, **fold_legs(legs), "clock": clock, "binary": args.binary}
    if args.output is not None:
        write_json_atomic(args.output, record)
    print(canonical_json(record))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HarnessError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(2)
