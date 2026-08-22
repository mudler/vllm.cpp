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
        --lease-id "$RC_JOB_ID" --our-revision "$(git rev-parse HEAD)" \\
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
    ClockWindow,
    _parse_artifact,
    add_clock_arguments,
    clock_evidence_reasons,
    clock_window,
    sample_compute_processes,
)
from tools.bench.dflash2_speed_harness import (
    SSE_PING_ENV,
    build_recipe_reasons,
    checkpoint_reasons,
    clock_state_reasons,
    contention_reasons,
    fold_legs,
    is_warm_leg,
    leg_reasons,
    model_binding_reasons,
    repeat_reasons,
    require_no_reasons,
    speculative_config_reasons,
    sse_keepalive_reasons,
    workload_fingerprint,
)
from tools.bench.gpu_clock_state import (
    build_spanned_clock_record,
    read_boot_id,
    read_sample_stream,
)

#: Re-exported so `our_arm.leg_reasons` and `our_arm.fold_legs` keep naming the
#: rule this arm applies. The rule itself lives in `dflash2_speed_harness`
#: because the ORACLE ARM APPLIES THE SAME ONE: a ratio between a median folded
#: here and a median folded there is a ratio between two statistics.
__all__ = [
    "fold_legs",
    "leg_reasons",
    "legs_with_spans",
    "parse_leg_spans",
    "parse_legs",
    "warm_leg_spans",
    "main",
]
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


#: The leg-boundary marker, positional against the SECOND format string
#: `examples/cli/main.cpp` prints per leg (#1671). Separate from `LEG_RE` on
#: purpose: the timing line is parsed by evidence and readers that predate this
#: marker, so it keeps its bytes and the span arrives on its own line.
#:
#: Epoch seconds, because the sampler stamps `timestamp_utc` and the two have to
#: be comparable without either side guessing a time zone.
LEG_SPAN_RE = re.compile(
    r"run=(?P<run>\d+)/(?P<of>\d+)\s+generate_start_unix=(?P<start>[0-9.]+)\s+"
    r"generate_end_unix=(?P<end>[0-9.]+)"
)


def parse_leg_spans(stderr_text: str) -> dict[int, tuple[float, float]]:
    """Every leg boundary `vllm-cli` marked, keyed by its run number.

    ONE PROCESS AT A TIME. `--repeat N` numbers its legs 1..N and this harness
    launches one process per prompt, so the key is unique within the text of a
    single invocation and would collide across four of them.
    """

    spans: dict[int, tuple[float, float]] = {}
    for match in LEG_SPAN_RE.finditer(stderr_text):
        spans[int(match.group("run"))] = (
            float(match.group("start")),
            float(match.group("end")),
        )
    return spans


def legs_with_spans(stderr_text: str) -> list[dict[str, Any]]:
    """This process's legs, each carrying the instants it generated between.

    A leg with no marker is a REFUSAL and never a leg with no span. The clock
    window is built from these spans, so a leg that cannot be placed in time
    would silently shrink the window that has to describe the measurement --
    and a binary built before the marker existed would produce exactly that,
    quietly, on a leased run nobody wants to repeat.
    """

    legs = parse_legs(stderr_text)
    spans = parse_leg_spans(stderr_text)
    for leg in legs:
        span = spans.get(int(leg["run"]))
        if span is None:
            raise HarnessError(
                f"legs: run {leg['run']} printed a timing line and NO leg-boundary "
                "marker, so the clock cannot be attributed to it. The marker is "
                "`generate_start_unix=`/`generate_end_unix=`, printed by "
                "`examples/cli/main.cpp`; a binary built before it cannot drive "
                "this arm"
            )
        leg["generate_start_unix"], leg["generate_end_unix"] = span
    return legs


def warm_leg_spans(legs: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """The spans of the legs the median is folded from, and ONLY those.

    Not every leg. `fold_legs` discards run 1 of each repetition group for a
    named cause, so a window that kept run 1's span would attribute the number to
    samples taken during work the number excludes. On the 2026-08-22 run those
    four legs were 959.3 s against 93.2 s of warm generation, so this is not a
    rounding decision.

    THE MODEL LOAD IS ALREADY OUTSIDE EVERY SPAN, and not because of this
    function. `examples/cli/main.cpp` brackets `vllm_complete` alone, and
    `vllm_engine_load` runs before the repeat loop, so no leg of this arm has
    ever contained a load. What run 1 does carry is the first graph capture, the
    first KV allocation and the first touch of weights the loader mapped -- which
    is why it is 240 s to a warm leg's 5.8 s and why it is discarded.
    """

    return [
        (float(leg["generate_start_unix"]), float(leg["generate_end_unix"]))
        for leg in legs
        if is_warm_leg(leg)
    ]


def spanned_clock(
    window: ClockWindow, legs: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    """This arm's clock record, restricted to the warm legs it folded.

    READ FROM THE WINDOW THE ARM OWNS. `ClockWindow` starts the sampler after
    the precheck and stops it after the last leg (#1657), so `window.samples` is
    the stream that sampler wrote and nothing else could have written it. It is
    read AFTER the `with` block, and by then `ClockWindow.close` has signalled
    the sampler and waited for the process to exit -- so the file is complete
    and closed, which is strictly stronger than the per-line flush
    `read_sample_stream` tolerates a partial tail for.

    `(None, None)` when NO sampler ran, which is what `--clock-summary` unset
    means. That is not a spanning failure and gets no message of its own:
    `clock_state_reasons` already refuses a `None` record with the words that
    name it.

    Otherwise returns `(record, error)`. A failure is RETURNED rather than
    raised because the legs already cost a lease by the time it can happen, and
    the caller writes the arm record before it judges the clock. Nothing
    quotable is printed either way.
    """

    if window.samples is None:
        return None, None
    try:
        stream = read_sample_stream(window.samples)
        boot = read_boot_id()
        return (
            build_spanned_clock_record(stream, warm_leg_spans(legs), boot_id=boot),
            None,
        )
    except HarnessError as error:
        return None, str(error)


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
    # BOTH clock paths come from here. `--clock-samples` is where THIS arm's
    # own sampler writes its raw stream, and it is also what the arm reads back
    # to restrict the window to its warm legs (#1671) -- one path, one file, no
    # second spelling of the same flag.
    add_clock_arguments(parser)
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
    reasons += clock_evidence_reasons(args, label="ours")
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

    # THE ARM OWNS ITS WINDOW (#1657), AND THE WINDOW IS THE WORK (#1671).
    # Two repairs, one path, and neither is optional.
    #
    # `ClockWindow` starts the sampler HERE rather than in the shell, because
    # `gpu_clock_state` writes its summary when the sampler STOPS: an arm handed
    # a summary before it ran could never read one. It also waits for the first
    # sample, so no leg runs outside the window, and it keeps a stop failure in
    # `close_error` instead of raising it over legs that already cost a lease.
    #
    # The record this arm is JUDGED on is then built from the samples inside the
    # WARM LEG SPANS `vllm-cli` marked, not from the sampler's whole-process
    # summary. Unlike the oracle arm, ours is one `vllm-cli` process per prompt
    # and each one loads its own checkpoint, so the sampler's window necessarily
    # spans four loads: on 2026-08-22 that was 3222 s sampled against 93 s of
    # warm generation, 18.37% busy, and `clock_reasons` refused it -- correctly.
    # Spanning is not a relaxation of that floor. `build_spanned_clock_record`
    # runs `build_clock_record` over a smaller and truer set of samples and
    # `clock_reasons` judges the result unchanged.
    legs: list[dict[str, Any]] = []
    window = clock_window(args)
    with window:
        for prompt in checked["prompts"]:
            # NOT `parse_legs`. A leg with no boundary marker is a refusal,
            # because the window is built out of those boundaries and a leg that
            # cannot be placed in time would shrink it silently.
            legs += legs_with_spans(run_binary(args, prompt))
    require_no_reasons(
        leg_reasons(legs, max_tokens=args.max_tokens), what="DFlash2 our-arm capture"
    )

    clock, span_error = spanned_clock(window, legs)
    # THE SAMPLER'S FAILURE OUTRANKS THE SPANNING READER'S, because it causes
    # it: a sampler that died at stop wrote the truncated stream the spanning
    # then could not use, and the first message names the cause.
    clock_error = window.close_error or span_error
    record = {
        **checked,
        **fold_legs(legs),
        "clock": clock,
        # NULL on a healthy run; the sampler's or the spanning reader's own
        # words when there is no window to judge. `ClockWindow.__exit__` keeps a
        # stop failure rather than raising it, and `spanned_clock` returns
        # rather than raises, so this arm's legs reach the disk either way.
        "clock_error": clock_error,
        "binary": args.binary,
    }
    # WRITTEN BEFORE THE CLOCK IS JUDGED, and nothing quotable is printed until
    # it passes: the refusal precedes the `print`, and `build_speed_result`
    # refuses the same record again through `clock_pairing`.
    if args.output is not None:
        write_json_atomic(args.output, record)
    reasons = clock_state_reasons(clock, label="ours", detail=str(clock_error or ""))
    if clock is not None and window.close_error:
        # THE SPANNED WINDOW CAN SURVIVE A SAMPLER THAT DID NOT, because the raw
        # stream is flushed per line and the summary is written last. It is
        # still refused: a sampler that failed at stop is a sampler whose stream
        # nobody can vouch for, and `clock_state_reasons` cannot say so here
        # because it was handed a record that looks healthy.
        reasons.append(
            f"clock: the ours arm's sampler did not stop cleanly, so the window it "
            f"describes is not established: {window.close_error}"
        )
    require_no_reasons(reasons, what="DFlash2 our-arm capture")
    print(canonical_json(record))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HarnessError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(2)
