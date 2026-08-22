#!/usr/bin/env python3
"""Record and assert the GPU clock a measurement was actually taken at (#543).

On `dgx.casa` (GB10, driver 580.159.03) the SM clock differs **between boots**
and is not throttling — `clocks_throttle_reasons.active = 0x0`, persistence
`Enabled`:

    boot f6bbbfc6   n=61, min 2398 / med 2470 / max 2489   82.1664 ms/step
    boot 2fca2b02   n=50, flat 2190 (max 3003, apps 2418)  88.1000 ms/step

A 12.79% median-clock delta produced +7.22% step time, and the control settles
it: `marlin::Marlin`, 129 calls/step, byte-identical invocation and **no source
change**, moved 45.2845 -> 49.6544 ms/step = **+9.65%**. That drift is larger
than either deficit it was used to rank (`in_proj` +2.97%, `out_proj` +6.28%),
so both are NOT ESTABLISHED.

Nothing in the tree recorded any of it. This module is the one place that
samples the state, writes it down, and refuses a comparison that cannot be
attributed to it — the same shape as the oracle-identity pin (#520), for the
same reason: an environment variable nobody recorded silently reprices every
number.

Standard library only, like `serve_low_common`, so every assertion here runs in
CPU CI with no GPU and no `nvidia-smi`. Sampling needs the binary; the logic
does not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from types import FrameType
from typing import Any

from tools.bench.serve_low_common import HarnessError, canonical_json, write_json_atomic

BOOT_ID_PATH = pathlib.Path("/proc/sys/kernel/random/boot_id")
NVIDIA_SMI = "nvidia-smi"

# The query is ordered, and `parse_query_row` is positional against it: a driver
# that reorders or drops a column produces a length mismatch, which is a
# refusal, rather than a value silently read from the wrong position.
QUERY_FIELDS: tuple[tuple[str, str], ...] = (
    ("index", "index"),
    ("name", "gpu_name"),
    ("driver_version", "driver_version"),
    ("clocks.sm", "sm_clock_mhz"),
    ("clocks.max.sm", "clocks_max_sm_mhz"),
    ("clocks.applications.graphics", "clocks_applications_graphics_mhz"),
    # Driver 580 accepts the `throttle` spelling; newer drivers prefer
    # `clocks_event_reasons.active`. A failed query is a refusal to sample, so a
    # rename surfaces as a loud stop rather than as an absent field.
    ("clocks_throttle_reasons.active", "throttle_reasons_active"),
    ("persistence_mode", "persistence_mode"),
    ("utilization.gpu", "utilization_gpu_pct"),
)
_INTEGER_FIELDS = frozenset(
    {
        "index",
        "sm_clock_mhz",
        "clocks_max_sm_mhz",
        "clocks_applications_graphics_mhz",
        "utilization_gpu_pct",
    }
)
# Held constant for the whole window; a mid-window change means the sampler
# watched two different GPUs or the applications clock moved under it.
STATIC_FIELDS: tuple[str, ...] = (
    "gpu_name",
    "driver_version",
    "clocks_max_sm_mhz",
    "clocks_applications_graphics_mhz",
    "persistence_mode",
)

# --- Thresholds. All four are arguments from the data above, not preferences. -
#
# Within-run spread, 5.0%. The admissible band is bounded on both sides: it must
# ACCEPT the only clean window we have, (2489-2398)/2470 = 3.68%, because a
# threshold that voids our one good measurement is useless; and it must REJECT
# the two probes eight minutes apart inside ONE boot, 2398 against 1781, which
# is ~26% however it is normalized. 5.0 clears the clean observation by ~1.3
# points so a marginally noisier but healthy window is not spuriously voided,
# and sits roughly five times below the failure it exists to catch.
#
# It is deliberately NOT held to the forward criterion the offset below is held
# to. 5.0 x the transfer is 3.77%, above the 2.97% smallest deficit ranked, so a
# leg sitting AT the spread ceiling can carry an artifact larger than that
# deficit. The two rules defend different things and the difference is the
# reason. The offset bounds a SYSTEMATIC difference between the arms -- one arm
# ran at one clock and the other at another, and the whole of it transfers into
# the ratio -- so it must sit under the smallest effect anyone ranks. Spread
# bounds DISPERSION inside one arm's window, which does not transfer that way:
# both arms sweep the same six concurrency points on the same box, so most of
# the dispersion is common, and what survives into the ratio is the difference
# of two medians, which MAX_CROSS_ARM_OFFSET_PCT already bounds at 1.0%. The
# spread rule's job is to detect that a window was not ONE state at all (the 26%
# within-boot disagreement), not to bound a transferable bias. Tightening it to
# satisfy the forward criterion would mean <=3.93%, which sits 0.25 points above
# our only clean capture and would void it on any noisier-but-healthy day -- the
# exact failure the both-sides bound was chosen to avoid. The residual is stated
# rather than hidden: passing spread establishes that each arm was one state, it
# does NOT by itself establish a sub-4% deficit. The offset rule is what
# qualifies the ratio.
MAX_WITHIN_RUN_SPREAD_PCT = 5.0
#
# Cross-arm median offset, 1.0%. For any kernel whose time scales with clock the
# transfer is bounded above by 1.0 -- a 12.79% clock deficit can cost at most
# 12.79% of time -- so a 1.0% offset implies AT MOST a 1.0% effect on physics,
# with no appeal to any measurement. That is already under the 2.97% smallest
# deficit this harness has been used to rank, so a pair inside the threshold
# cannot have had its ranking inverted by clocks. The measured 0.7548 below is
# consistent with that ceiling and sits under it exactly as a partly
# memory-bound kernel should, which is corroboration, not the argument.
MAX_CROSS_ARM_OFFSET_PCT = 1.0
#
# Cross-arm MEAN offset, 1.0. The rule above bounds the two arms' MEDIANS, and a
# median is not the statistic that carries the transfer it is bounding.
# Throughput is an INTEGRAL over the window, so what it sees is the arm's mean
# clock. On the three Qwen3.8-27B bf16 c1 pairings of 2026-08-19
# (`/mnt/nas_share/rc/q38bf16/out/`) `median_offset_pct` reads exactly 0.0000% on
# all three while the MEANS are 0.2521 / 0.1530 / 0.1035 points apart. The
# excursion population is 5.74% of retained busy samples (97 of 1690) and sits
# almost entirely below the median -- 95 of those 97 -- so a median over 155 to
# 246 samples steps straight over the one part of the distribution that does NOT
# cancel between the arms. #1546.
#
# The NUMBER is not a new one, deliberately. It is the same physics ceiling and
# the same forward criterion that place MAX_CROSS_ARM_OFFSET_PCT: the transfer is
# bounded above by 1.0 point of kernel time per point of clock, so a 1.0% mean
# offset implies AT MOST a 1.0% effect on the ratio, which is under the 2.97%
# smallest deficit this harness has been used to rank. On a flat window
# mean == median and the two terms are identical, so every steady capture --
# including the 2026-08-15 pinned series, flat 2184 MHz over n=861 -- scores the
# same on both, and no new class of window is refused there.
#
# It bounds a CROSS-ARM quantity and not a per-arm one. A burden BOTH arms pay
# divides out of the ratio and this term is silent on it; bounding the burden
# each arm carries on its own is a different rule, specced under #1354 and not
# implemented here.
#
# What it does NOT bound is a phase-scoped metric. The excursions are locked to
# the request head, so the mean bounds output throughput and median inter-token
# latency and does NOT bound time-to-first-token, whose repeat CV is 2.20%
# against 0.01% for median inter-token latency in the same files. Argument,
# rejected statistics and evidence: `.agents/specs/clock-cross-arm-mean.md`.
MAX_CROSS_ARM_MEAN_OFFSET_PCT = 1.0
#
# Retained busy samples per window, 30. `spread_pct` over n == 1 is definitionally
# 0.00% -- the BEST score the gate can award -- so without a floor the window the
# sampler barely observed outscores the one it actually watched. Bounded on both
# sides like the spread rule. It must ACCEPT the real windows: the only two
# #543 captured are n=61 and n=50, and 30 sits 40% below the smaller. It must
# REJECT the degenerate window, and 30 is 30x above n == 1. And it reads
# straight off the sampler and the grid rather than off what happens to pass: at
# the driver's `--interval 1` a busy sample is a second of OBSERVED BUSY GPU,
# while the smallest configured leg (online_gate.POINTS_BY_MODEL's four-point
# set: 6+6+12+24 prompts at 128 output tokens = 1920 sequential decode steps)
# would have to average under ~16 ms/step INCLUDING its 48 prefills of 1024
# tokens to finish in 30 busy seconds. #543's own table for this box is 82-88
# ms/step. A leg below this floor is not a fast leg; it is an unobserved one.
MIN_BUSY_SAMPLES = 30
#
# Busy fraction of the window, 0.5. The count floor alone does not catch
# dilution: 30 busy among 3000 idle still clears it, and the reported spread
# still describes 1% of the window. The sampler covers the BENCH LOOP only -- it
# starts after the preflight stream and stops before the after-thermal snapshot
# -- so the non-busy time inside it is the client startup between six
# `online_gate.py bench` invocations, not model load, not cache drops, not
# server start, while the GPU serves 336 requests of 1024-in/128-out across the
# span. Requiring the MAJORITY of the window to be busy is the weakest form of
# the claim the record makes, namely that it describes the measured work. It is
# also the field that betrays a sampler which outlived its leg: an orphan accrues
# idle samples without bound and nothing else in the record notices.
MIN_BUSY_FRACTION = 0.5
#
# Percentage points of kernel time per percentage point of clock, from the one
# cross-boot pair we have: +9.65% marlin over a 12.79% clock offset. The
# step-level transfer was lower (0.565); the larger is used. THIS IS n = 1. It is
# not a gate TERM -- no gate expression evaluates it, which the mutation set
# proves -- but it was a gate PREMISE, because MAX_CROSS_ARM_OFFSET_PCT was
# chosen by multiplying through it. The physics bound above retires it from that
# role: the threshold now holds at the transfer's theoretical ceiling of 1.0, so
# the coefficient only ever REPORTS an estimated effect.
CLOCK_TIME_TRANSFER = (49.6544 / 45.2845 - 1.0) / (2470.0 / 2190.0 - 1.0)

# GpuIdle | ApplicationsClocksSetting | DisplayClockSetting. These say something
# about configuration or occupancy, not about the GPU being held back while it
# worked. Every other bit -- power cap, hardware slowdown, sync boost, thermal,
# power brake -- means the window is not the window the number claims.
BENIGN_THROTTLE_MASK = 0x1 | 0x2 | 0x100
THROTTLE_BIT_NAMES: dict[int, str] = {
    0x1: "GpuIdle",
    0x2: "ApplicationsClocksSetting",
    0x4: "SwPowerCap",
    0x8: "HwSlowdown",
    0x10: "SyncBoost",
    0x20: "SwThermalSlowdown",
    0x40: "HwThermalSlowdown",
    0x80: "HwPowerBrakeSlowdown",
    0x100: "DisplayClockSetting",
}

_REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "boot_id",
    "clocks_applications_graphics_mhz",
    "clocks_max_sm_mhz",
    "driver_version",
    "gpu_name",
    "idle_samples_excluded",
    "persistence_mode",
    "sm_clock_mhz",
    "throttle_reasons_active",
)
_REQUIRED_SUMMARY_FIELDS: tuple[str, ...] = ("n", "min", "median", "max", "spread_pct")

# Both thresholds are INCLUSIVE, and a ratio of two integer clocks lands on them
# only to within double rounding (2020/2000 - 1 == 0.010000000000000009). The
# tolerance is a floating-point artifact allowance, not slack in the rule: it is
# nine orders of magnitude below the smallest effect either threshold defends.
_THRESHOLD_EPSILON = 1e-9


def read_boot_id(path: pathlib.Path = BOOT_ID_PATH) -> str:
    """Return the running kernel's boot id, or refuse.

    This is the only field that can tell two measurements apart when everything
    else -- binary, argv, model, driver -- is identical, which is exactly the
    #543 shape. An absent or empty value is a refusal, never a default: a
    default would make every cross-boot pair look same-boot.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise HarnessError(f"boot id is unreadable: {path}: {error}") from error
    value = text.strip()
    if not value:
        raise HarnessError(f"boot id is empty: {path}")
    return value


def _parse_scalar(raw: str, query_name: str, field: str) -> Any:
    value = raw.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NA", "[NOT SUPPORTED]"}:
        raise HarnessError(f"{query_name} is unavailable: {raw!r}")
    if field in _INTEGER_FIELDS:
        # `--format=csv` appends a unit ("2190 MHz", "97 %"); `nounits` does
        # not. Both are accepted so a fixture captured either way parses.
        token = value.split()[0]
        try:
            return int(token)
        except ValueError as error:
            raise HarnessError(f"{query_name} is not an integer: {raw!r}") from error
    return value


def parse_query_row(row: str) -> dict[str, Any]:
    """Parse one `nvidia-smi --query-gpu` CSV row into a sample."""

    cells = row.split(",")
    if len(cells) != len(QUERY_FIELDS):
        raise HarnessError(
            f"nvidia-smi row expected {len(QUERY_FIELDS)} columns, got {len(cells)}: {row!r}"
        )
    sample: dict[str, Any] = {}
    for (query_name, field), cell in zip(QUERY_FIELDS, cells):
        sample[field] = _parse_scalar(cell, query_name, field)
    sample["throttle_reasons_active"] = _normalize_throttle(
        sample["throttle_reasons_active"]
    )
    return sample


def _normalize_throttle(value: object) -> str:
    text = str(value).strip().lower()
    try:
        bits = int(text, 16)
    except ValueError as error:
        raise HarnessError(
            f"clocks_throttle_reasons.active is not hexadecimal: {value!r}"
        ) from error
    if bits < 0:
        raise HarnessError(f"clocks_throttle_reasons.active is negative: {value!r}")
    return f"0x{bits:016x}"


def query_once(*, smi: str = NVIDIA_SMI, timeout_s: float = 10.0) -> dict[str, Any]:
    """Sample the GPU once. A failed probe refuses; it never returns a guess."""

    argv = [
        smi,
        "--query-gpu=" + ",".join(name for name, _ in QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=timeout_s
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HarnessError(f"nvidia-smi clock probe failed: {error}") from error
    if result.returncode != 0:
        raise HarnessError(
            f"nvidia-smi clock probe exited {result.returncode}: {result.stderr.strip()!r}"
        )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise HarnessError("nvidia-smi clock probe returned no rows")
    if len(rows) > 1:
        # More than one GPU is a real configuration, but this harness records
        # ONE clock per leg and would have to pick. Refuse rather than pick.
        raise HarnessError(
            f"nvidia-smi clock probe returned {len(rows)} GPUs; this harness records one"
        )
    return parse_query_row(rows[0])


def summarize_sm_clocks(values: Sequence[float]) -> dict[str, float]:
    """Return `{n, min, median, max, mean, spread_pct}` over the measured window.

    `mean` is what an INTEGRAL metric was actually clocked at. The median steps
    over the excursion population entirely -- 5.74% of the 2026-08-19 samples,
    95 of 97 of them below their window's median -- and that population is the
    part that does not cancel between two arms. Recording it is what makes
    MAX_CROSS_ARM_MEAN_OFFSET_PCT computable (#1546).
    """

    if not values:
        raise HarnessError("cannot summarize an empty SM-clock window")
    numbers = [float(value) for value in values]
    for number in numbers:
        if not math.isfinite(number) or number <= 0.0:
            raise HarnessError(f"SM clock must be finite and positive, got {number!r}")
    median = statistics.median(numbers)
    return {
        "max": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": median,
        "min": min(numbers),
        "n": len(numbers),
        "spread_pct": (max(numbers) - min(numbers)) / median * 100.0,
    }


def build_clock_record(
    samples: Sequence[Mapping[str, Any]], *, boot_id: str
) -> dict[str, Any]:
    """Reduce a window of samples to the per-leg record.

    Idle samples are EXCLUDED from the statistics and COUNTED, not dropped: a
    clock read while the GPU is doing nothing did not price any work, and the
    timed window necessarily contains the harness's own gaps between
    concurrency points. Excluding them silently would be a lie; a window that is
    entirely idle has nothing to summarize and is refused.
    """

    if not samples:
        raise HarnessError("cannot build a clock record from an empty window")
    if not str(boot_id).strip():
        raise HarnessError("clock record requires a boot id")
    first = samples[0]
    for field in STATIC_FIELDS:
        if field not in first:
            raise HarnessError(f"clock sample omits {field}")
        for sample in samples[1:]:
            if sample.get(field) != first[field]:
                raise HarnessError(
                    f"{field} changed mid-window: {first[field]!r} -> {sample.get(field)!r}"
                )
    busy: list[float] = []
    idle = 0
    throttle: set[str] = set()
    for sample in samples:
        if "sm_clock_mhz" not in sample:
            raise HarnessError("clock sample omits sm_clock_mhz")
        throttle.add(_normalize_throttle(sample.get("throttle_reasons_active", "0x0")))
        utilization = sample.get("utilization_gpu_pct")
        if utilization is None:
            raise HarnessError("clock sample omits utilization_gpu_pct")
        if float(utilization) <= 0.0:
            idle += 1
            continue
        busy.append(float(sample["sm_clock_mhz"]))
    if not busy:
        raise HarnessError(
            f"every one of {len(samples)} clock samples was idle; "
            "there is no window to attribute the measurement to"
        )
    return {
        "boot_id": str(boot_id).strip(),
        "clocks_applications_graphics_mhz": first["clocks_applications_graphics_mhz"],
        "clocks_max_sm_mhz": first["clocks_max_sm_mhz"],
        "driver_version": first["driver_version"],
        "gpu_name": first["gpu_name"],
        "idle_samples_excluded": idle,
        "persistence_mode": first["persistence_mode"],
        "sm_clock_mhz": summarize_sm_clocks(busy),
        "throttle_reasons_active": sorted(throttle),
    }


def sample_instant_s(sample: Mapping[str, Any]) -> float:
    """The wall-clock instant this sample was taken at, in Unix epoch seconds.

    `run_sampler` writes `timestamp_utc` as an AWARE ISO-8601 stamp, and this is
    the only field that can place a sample against a span a DIFFERENT process
    measured. `elapsed_s` cannot: it is relative to the sampler's own start, and
    the arm that marks its legs has no access to that origin.

    A naive stamp is a REFUSAL rather than an assumed UTC. An unknown zone
    shifts every span by the host's offset, which is a whole hour on this fleet
    and moves a window nobody would see was wrong.
    """

    raw = sample.get("timestamp_utc")
    if raw is None:
        raise HarnessError(
            "clock sample omits timestamp_utc, so it cannot be placed in a span"
        )
    text = str(raw).strip()
    # `fromisoformat` takes `Z` from 3.11 on; normalizing keeps this module on
    # the standard library floor the rest of it holds to.
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        moment = dt.datetime.fromisoformat(text)
    except ValueError as error:
        raise HarnessError(
            f"clock sample timestamp_utc is not ISO-8601: {raw!r}: {error}"
        ) from error
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise HarnessError(
            f"clock sample timestamp_utc {raw!r} carries no time zone; an unknown "
            "zone is not UTC and would shift every span by the host's offset"
        )
    return moment.timestamp()


def normalize_spans(spans: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Validate the spans a driver measured its work in, or refuse.

    An EMPTY list is the refusal that matters most. It is what a driver whose
    leg markers did not parse hands over, and falling back to the whole stream
    there would silently restore the very window the busy floor refused -- under
    a record that claims to be spanned.
    """

    if not spans:
        raise HarnessError(
            "a spanned clock record was asked for with no span at all; there is "
            "no window to attribute the measurement to, and the whole stream is "
            "not a fallback"
        )
    normalized: list[tuple[float, float]] = []
    for index, span in enumerate(spans):
        pair = tuple(span)
        if len(pair) != 2:
            raise HarnessError(
                f"clock span {index + 1} is not a (start, end) pair: {span!r}"
            )
        start, end = (float(pair[0]), float(pair[1]))
        if not math.isfinite(start) or not math.isfinite(end):
            raise HarnessError(f"clock span {index + 1} is not finite: {span!r}")
        if end < start:
            raise HarnessError(
                f"clock span {index + 1} ends before it starts: {start!r} -> {end!r}"
            )
        normalized.append((start, end))
    return normalized


def samples_within_spans(
    samples: Sequence[Mapping[str, Any]], spans: Sequence[Sequence[float]]
) -> list[Mapping[str, Any]]:
    """The samples whose instant lies inside the union of `spans`.

    Both ends are inclusive. The sampler ticks on its own interval and the spans
    come from another process's clock, so an exclusive end would drop a sample
    that lands exactly on a boundary for no reason anyone could state.
    """

    bounds = normalize_spans(spans)
    return [
        sample
        for sample in samples
        if any(start <= sample_instant_s(sample) <= end for start, end in bounds)
    ]


def build_spanned_clock_record(
    samples: Sequence[Mapping[str, Any]],
    spans: Sequence[Sequence[float]],
    *,
    boot_id: str,
) -> dict[str, Any]:
    """Reduce a sampled stream to the record for the WORK, not for the process.

    `MIN_BUSY_FRACTION` asks that the retained window describe the measured
    work. A driver that samples across a whole process lifetime does not satisfy
    it by accident: the DFlash2 our-arm run of 2026-08-22 sampled 3222 seconds
    of which 93 were warm generation, and `clock_reasons` refused it at 18.37%
    busy ([#1671](https://github.com/mudler/vllm.cpp/issues/1671)). The refusal
    was correct, so the repair is not to the floor. It is to the WINDOW: a
    driver that can say when its work ran hands the spans over here, and the
    record is built from the samples inside them.

    **Nothing here relaxes a rule.** The result is an ordinary clock record and
    `clock_reasons` judges it unchanged -- the busy fraction, the retained
    count, the spread, the throttle reasons and persistence mode all apply, to a
    smaller and truer set of samples. A span the GPU was idle through is refused
    below; a span too short to observe fails the count floor; and no span at all
    is refused rather than widened back to the stream.

    It reuses `build_clock_record` rather than re-deriving anything, so the
    statistics, the idle accounting and the mid-window field check are the same
    code the unrestricted path runs.
    """

    if not samples:
        raise HarnessError("cannot build a clock record from an empty window")
    bounds = normalize_spans(spans)
    kept = samples_within_spans(samples, bounds)
    if not kept:
        raise HarnessError(
            f"the {len(bounds)} measured span(s) retained none of the "
            f"{len(samples)} clock samples in the stream; the driver's spans and "
            "the sampler's window do not overlap, which is a driver defect and "
            "never a fast window"
        )
    record = build_clock_record(kept, boot_id=boot_id)
    #: WHAT WAS RESTRICTED, beside the record it produced. Without it a reader
    #: cannot tell a window that covered the work from one that covered a
    #: fraction of it, and both score the same on `spread_pct`.
    record["window"] = {
        "retained_samples": len(kept),
        "spanned_s": sum(end - start for start, end in bounds),
        "spans": len(bounds),
        "stream_samples": len(samples),
    }
    return record


def read_sample_stream(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read the raw per-sample stream `run_sampler` writes, or refuse.

    ONE TRUNCATED FINAL LINE IS TOLERATED, and nothing else is. `run_sampler`
    flushes after every sample, so a consumer that reads the file while the
    sampler is still appending sees whole lines and, at worst, a partial tail.
    Refusing that would end a two-hour leased run on the newest sample in the
    file -- which is by construction AFTER the last leg and therefore outside
    every span the caller asked for, so it can cost the measurement nothing.

    A malformed line ANYWHERE ELSE is a refusal. A stream with a hole in the
    middle is not a shorter stream, and a shorter stream is exactly what the
    coverage floors exist to catch.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise HarnessError(f"cannot read the clock sample stream {path}: {error}") from error
    lines = text.splitlines()
    # `splitlines` cannot tell `{...}\n` from `{...}`, and only the latter is a
    # tail still being written. The final line is exempt only when the file does
    # not end in a newline.
    partial_tail_allowed = bool(lines) and not text.endswith("\n")
    samples: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as error:
            if number == len(lines) and partial_tail_allowed:
                break
            raise HarnessError(
                f"clock sample stream {path} line {number} is not JSON: {error}"
            ) from error
        if not isinstance(sample, dict):
            raise HarnessError(
                f"clock sample stream {path} line {number} is not an object"
            )
        samples.append(sample)
    if not samples:
        raise HarnessError(f"clock sample stream {path} holds no sample")
    return samples



def merge_clock_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold one arm's repeated legs into the arm's clock record.

    A ratio is taken between two ARMS, not two legs, so the clock the ratio was
    measured at is the arm's. Folding refuses what it cannot fold: repetitions
    that straddle two boots, or that saw different hardware or a different
    applications clock, are not one arm's measurement and merging them would
    manufacture a window that never existed.

    The merged spread spans every leg. That is deliberate -- an arm whose three
    repetitions individually looked steady but sat at different clocks is
    exactly the case the within-run rule must still catch.
    """

    if not records:
        raise HarnessError("cannot merge an empty set of clock records")
    for index, record in enumerate(records):
        validate_clock_record(record, label=f"leg {index + 1}")
    first = records[0]
    for index, record in enumerate(records[1:], start=2):
        if record["boot_id"] != first["boot_id"]:
            raise HarnessError(
                "arm straddles two boots: leg 1 ran on "
                f"{first['boot_id']} and leg {index} on {record['boot_id']}; "
                "these are not one measurement"
            )
        for field in STATIC_FIELDS:
            if record.get(field) != first.get(field):
                raise HarnessError(
                    f"{field} differs between legs: "
                    f"{first.get(field)!r} vs {record.get(field)!r}"
                )
    medians = [float(record["sm_clock_mhz"]["median"]) for record in records]
    lowest = min(float(record["sm_clock_mhz"]["min"]) for record in records)
    highest = max(float(record["sm_clock_mhz"]["max"]) for record in records)
    median = statistics.median(medians)
    # SAMPLE-COUNT WEIGHTED, because the arm's mean is over the arm's whole
    # retained busy series and the legs are not the same length -- 155 / 155 /
    # 156 at c1 and 245 / 246 / 244 at c8 in the one campaign this fold has run
    # on. An unweighted mean of leg means is a different number and is wrong.
    # A leg with no mean makes the ARM have none rather than a fabricated one, so
    # the refusal in `compare_clock_records` propagates instead of being papered
    # over by the two legs that do carry it.
    leg_means = [_mean_or_none(record) for record in records]
    weighted_mean: float | None = None
    if all(value is not None for value in leg_means):
        counts = [int(record["sm_clock_mhz"]["n"]) for record in records]
        total = sum(counts)
        if total > 0:
            weighted_mean = (
                sum(value * count for value, count in zip(leg_means, counts)) / total
            )
    throttle: set[str] = set()
    for record in records:
        throttle.update(_normalize_throttle(value) for value in record["throttle_reasons_active"])
    return {
        "boot_id": first["boot_id"],
        "clocks_applications_graphics_mhz": first["clocks_applications_graphics_mhz"],
        "clocks_max_sm_mhz": first["clocks_max_sm_mhz"],
        "driver_version": first["driver_version"],
        "gpu_name": first["gpu_name"],
        "idle_samples_excluded": sum(
            int(record["idle_samples_excluded"]) for record in records
        ),
        "legs": len(records),
        "persistence_mode": first["persistence_mode"],
        "sm_clock_mhz": {
            "max": highest,
            **({} if weighted_mean is None else {"mean": weighted_mean}),
            "median": median,
            "min": lowest,
            "n": sum(int(record["sm_clock_mhz"]["n"]) for record in records),
            "spread_pct": (highest - lowest) / median * 100.0,
        },
        "throttle_reasons_active": sorted(throttle),
    }


def validate_clock_record(record: Mapping[str, Any], *, label: str) -> None:
    """Fail closed on every defect a clock record can carry."""

    if not isinstance(record, Mapping):
        raise HarnessError(f"{label} clock record is not an object")
    for field in _REQUIRED_RECORD_FIELDS:
        if field not in record:
            raise HarnessError(f"{label} clock record omits {field}")
    if not str(record["boot_id"]).strip():
        raise HarnessError(f"{label} clock record has an empty boot_id")
    summary = record["sm_clock_mhz"]
    if not isinstance(summary, Mapping):
        raise HarnessError(f"{label} clock record sm_clock_mhz is not an object")
    for field in _REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            raise HarnessError(f"{label} clock record sm_clock_mhz omits {field}")
    for field in ("min", "median", "max"):
        value = summary[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HarnessError(f"{label} clock record sm_clock_mhz.{field} is not numeric")
        if not math.isfinite(float(value)):
            raise HarnessError(f"{label} clock record sm_clock_mhz.{field} is not finite")
        if float(value) <= 0.0:
            raise HarnessError(
                f"{label} clock record sm_clock_mhz.{field} must be positive, got {value!r}"
            )
    if not isinstance(summary["n"], int) or isinstance(summary["n"], bool) or summary["n"] < 1:
        raise HarnessError(f"{label} clock record sm_clock_mhz.n must be a positive count")
    idle = record["idle_samples_excluded"]
    if not isinstance(idle, int) or isinstance(idle, bool) or idle < 0:
        raise HarnessError(
            f"{label} clock record idle_samples_excluded must be a non-negative count"
        )
    if float(summary["min"]) > float(summary["max"]):
        raise HarnessError(f"{label} clock record sm_clock_mhz min exceeds max")
    reasons = record["throttle_reasons_active"]
    if not isinstance(reasons, (list, tuple)) or not reasons:
        raise HarnessError(f"{label} clock record throttle_reasons_active is empty")
    for value in reasons:
        _normalize_throttle(value)


def _throttle_offenders(values: Iterable[str]) -> list[str]:
    offenders: list[str] = []
    for value in values:
        bits = int(_normalize_throttle(value), 16) & ~BENIGN_THROTTLE_MASK
        if not bits:
            continue
        names = [
            name for bit, name in sorted(THROTTLE_BIT_NAMES.items()) if bits & bit
        ] or [f"0x{bits:x}"]
        offenders.extend(names)
    return sorted(set(offenders))


def clock_reasons(record: Mapping[str, Any], *, label: str) -> list[str]:
    """Return the reasons this arm's clock state does not establish a number.

    An empty list means established. Every entry names its arm, so a reader of
    the summary can tell which side voided the ratio without opening evidence.
    """

    try:
        validate_clock_record(record, label=label)
    except HarnessError as error:
        return [f"clock: {error}"]
    reasons: list[str] = []
    # The coverage floors come FIRST because they qualify everything below them:
    # a spread computed over one retained sample is 0.00% and says nothing, and
    # a median computed over 1% of the window describes 1% of the window.
    busy = int(record["sm_clock_mhz"]["n"])
    idle = int(record["idle_samples_excluded"])
    if busy < MIN_BUSY_SAMPLES:
        reasons.append(
            f"clock: {label} retained only {busy} busy SM-clock sample(s) over the "
            f"measured window, below the {MIN_BUSY_SAMPLES} floor; a window this "
            "short cannot establish a spread (over n == 1 the spread is "
            "definitionally 0.00%, the best score the gate awards)"
        )
    observed = busy + idle
    busy_fraction = busy / observed if observed else 0.0
    if busy_fraction < MIN_BUSY_FRACTION:
        reasons.append(
            f"clock: {label} was idle for {idle} of {observed} SM-clock samples "
            f"({busy_fraction * 100.0:.2f}% busy, below the "
            f"{MIN_BUSY_FRACTION * 100.0:.0f}% floor); the retained window does not "
            "describe the measured work"
        )
    spread = float(record["sm_clock_mhz"]["spread_pct"])
    if spread > MAX_WITHIN_RUN_SPREAD_PCT + _THRESHOLD_EPSILON:
        reasons.append(
            f"clock: {label} SM-clock spread over the measured window is "
            f"{spread:.2f}%, above the {MAX_WITHIN_RUN_SPREAD_PCT}% ceiling; "
            "the number is NOT ESTABLISHED"
        )
    offenders = _throttle_offenders(record["throttle_reasons_active"])
    if offenders:
        reasons.append(
            f"clock: {label} was throttled during the measured window "
            f"({', '.join(offenders)}); the number is NOT ESTABLISHED"
        )
    if str(record["persistence_mode"]) != "Enabled":
        reasons.append(
            f"clock: {label} ran with persistence mode "
            f"{record['persistence_mode']!r}, not Enabled"
        )
    return reasons


def compare_clock_records(
    ours: Mapping[str, Any],
    theirs: Mapping[str, Any],
    *,
    ours_label: str = "ours",
    theirs_label: str = "vllm",
    allow_cross_boot: bool = False,
) -> dict[str, Any]:
    """Compare two arms' clock state and size the clock against the effect.

    The returned block is meant to sit NEXT TO the ratio it qualifies: both
    medians, both spreads, the signed offset, and the estimated share of the
    ratio the clock alone explains.

    `allow_cross_boot` waives ONE FIELD, `boot_id`, and nothing else. It exists
    because #545 makes same-boot capture of a four-leg chain unreliable and a
    gate nobody can satisfy is a gate everybody routes around; it converts that
    one refusal into a recorded caveat, and every other rule still applies.

    Because #545 makes the override the NORMAL path, the hardware the two arms
    ran on is asserted here explicitly rather than left to same-boot equality as
    an implicit proxy. Without that, waiving the boot silently waived "same
    machine" too: a GB10 at 3003 MHz max against an H100 at 1980 compared clean,
    with a caveat that said only "different boots".
    """

    reasons = [
        *clock_reasons(ours, label=ours_label),
        *clock_reasons(theirs, label=theirs_label),
    ]
    # STATIC_FIELDS is enforced within a window (`build_clock_record`) and
    # between one arm's legs (`merge_clock_records`); this is the third edge, and
    # the only one the override can reach. It is UNCONDITIONAL: two arms that
    # disagree on the GPU, the driver, the maximum SM clock, the applications
    # clock, or persistence mode are not two arms of one comparison, and no
    # override makes them so.
    for field in STATIC_FIELDS:
        ours_value = ours.get(field)
        theirs_value = theirs.get(field)
        if ours_value != theirs_value:
            reasons.append(
                f"clock: {ours_label} and {theirs_label} report a different {field} "
                f"({ours_value!r} vs {theirs_value!r}); these are not two arms of "
                "one comparison"
            )
    caveats: list[str] = []
    ours_boot = str(ours.get("boot_id", ""))
    theirs_boot = str(theirs.get("boot_id", ""))
    same_boot = bool(ours_boot) and ours_boot == theirs_boot
    cross_boot_override = bool(allow_cross_boot) and not same_boot
    if not same_boot:
        message = (
            f"clock: {ours_label} and {theirs_label} ran on DIFFERENT boots "
            f"({ours_boot or '<absent>'} vs {theirs_boot or '<absent>'}); "
            "cross-boot comparison is what produced the retracted #543 findings"
        )
        (caveats if cross_boot_override else reasons).append(
            message + " -- ACCEPTED under an explicit override" if cross_boot_override
            else message
        )

    ours_median = _median_or_none(ours)
    theirs_median = _median_or_none(theirs)
    offset_pct: float | None = None
    effect_pct: float | None = None
    if ours_median is not None and theirs_median is not None and theirs_median > 0.0:
        offset_pct = (ours_median / theirs_median - 1.0) * 100.0
        effect_pct = offset_pct * CLOCK_TIME_TRANSFER
        if abs(offset_pct) > MAX_CROSS_ARM_OFFSET_PCT + _THRESHOLD_EPSILON:
            reasons.append(
                f"clock: {ours_label} and {theirs_label} median SM-clock offset is "
                f"{offset_pct:+.2f}% (>{MAX_CROSS_ARM_OFFSET_PCT}%), estimated to move "
                f"kernel time by {effect_pct:+.2f}%; the ratio is NOT ESTABLISHED"
            )

    # The same comparison on the statistic an integral metric was actually
    # clocked at. Independent of the rule above on the recorded evidence: the
    # median offset is exactly 0.0000% on all three 2026-08-19 c1 pairings while
    # the mean offset is -0.2521 / -0.1530 / +0.1035 (#1546).
    ours_mean = _mean_or_none(ours)
    theirs_mean = _mean_or_none(theirs)
    mean_offset_pct: float | None = None
    for mean, label in ((ours_mean, ours_label), (theirs_mean, theirs_label)):
        if mean is None:
            # Fail closed. Every record written before this term lacks the field,
            # and skipping the term on those is the absent hook that reads as an
            # armed instrument: the block would print a clean verdict on a
            # quantity nothing examined. Unknown is not absence or success.
            reasons.append(
                f"clock: {label} SM-clock record carries no usable mean, so the "
                "cross-arm MEAN SM-clock comparison cannot be computed; the "
                "record predates that term and the window must be re-recorded"
            )
    if ours_mean is not None and theirs_mean is not None and theirs_mean > 0.0:
        mean_offset_pct = (ours_mean / theirs_mean - 1.0) * 100.0
        if abs(mean_offset_pct) > MAX_CROSS_ARM_MEAN_OFFSET_PCT + _THRESHOLD_EPSILON:
            reasons.append(
                f"clock: {ours_label} and {theirs_label} ran at MEAN SM clocks "
                f"{mean_offset_pct:+.2f}% apart ({ours_mean:.1f} vs {theirs_mean:.1f} "
                f"MHz, >{MAX_CROSS_ARM_MEAN_OFFSET_PCT}%); the excursion population "
                "does not cancel between the arms and the ratio is NOT ESTABLISHED"
            )
    ours_cost_pct = _mean_cost_pct(ours)
    theirs_cost_pct = _mean_cost_pct(theirs)
    # REPORTED, never gated on -- the same demotion `spread_pct` and
    # CLOCK_TIME_TRANSFER carry. It is the quantity #1546 quotes, and §Design of
    # `.agents/specs/clock-cross-arm-mean.md` measures the pairing where it reads
    # exactly 0.0000 while the arms' mean clocks are 2.28% apart: an absolute
    # value discards the sign, and two opposing excursion populations ADD.
    burden_difference_pct: float | None = None
    if ours_cost_pct is not None and theirs_cost_pct is not None:
        burden_difference_pct = abs(ours_cost_pct) - abs(theirs_cost_pct)
    return {
        "allow_cross_boot": bool(allow_cross_boot),
        "caveats": caveats,
        "cross_boot_override": cross_boot_override,
        "estimated_effect_pct": effect_pct,
        "estimated_effect_basis": (
            f"{CLOCK_TIME_TRANSFER:.4f} points of kernel time per point of clock, "
            "measured ONCE (marlin::Marlin 45.2845 -> 49.6544 ms/step over a "
            "12.79% clock offset, #543); reported, never gated on"
        ),
        # Reported beside the two gated offsets so a reader can see WHICH part of
        # each arm's distribution carried a refusal. Not gated on.
        "excursion_burden_difference_pct": burden_difference_pct,
        "mean_offset_pct": mean_offset_pct,
        "median_offset_pct": offset_pct,
        f"{ours_label}_boot_id": ours_boot,
        # How much of the window each side actually observed. Without these two
        # pairs a reader cannot tell a window the sampler watched from one it
        # barely touched, and the latter scores a perfect 0.00% spread.
        f"{ours_label}_busy_samples": _busy_or_none(ours),
        f"{ours_label}_idle_samples_excluded": _idle_or_none(ours),
        f"{ours_label}_mean_cost_pct": ours_cost_pct,
        f"{ours_label}_mean_sm_mhz": ours_mean,
        f"{ours_label}_median_sm_mhz": ours_median,
        f"{ours_label}_spread_pct": _spread_or_none(ours),
        "reasons": reasons,
        "same_boot": same_boot,
        f"{theirs_label}_boot_id": theirs_boot,
        f"{theirs_label}_busy_samples": _busy_or_none(theirs),
        f"{theirs_label}_idle_samples_excluded": _idle_or_none(theirs),
        f"{theirs_label}_mean_cost_pct": theirs_cost_pct,
        f"{theirs_label}_mean_sm_mhz": theirs_mean,
        f"{theirs_label}_median_sm_mhz": theirs_median,
        f"{theirs_label}_spread_pct": _spread_or_none(theirs),
    }


def _summary_field(record: Mapping[str, Any], field: str) -> float | None:
    summary = record.get("sm_clock_mhz")
    if not isinstance(summary, Mapping):
        return None
    value = summary.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median_or_none(record: Mapping[str, Any]) -> float | None:
    return _summary_field(record, "median")


def _spread_or_none(record: Mapping[str, Any]) -> float | None:
    return _summary_field(record, "spread_pct")


def _mean_or_none(record: Mapping[str, Any]) -> float | None:
    """The window's mean SM clock, or None for a record that predates the field.

    A record written before #1546 has no `mean`, and there is no way to recover
    one from `{n, min, median, max}`. `compare_clock_records` turns the None into
    a refusal rather than into a skipped term.
    """

    value = _summary_field(record, "mean")
    return value if value is not None and value > 0.0 else None


def _mean_cost_pct(record: Mapping[str, Any]) -> float | None:
    """`(median - mean) / median * 100`, SIGNED, over the retained busy series.

    Positive means the mean sits BELOW the median, which is what a downward
    excursion population does. Reported and never gated on: the gated quantity is
    the CROSS-ARM mean offset, because two arms carrying the same burden cancel
    and two arms carrying opposite ones add.
    """

    mean = _mean_or_none(record)
    median = _median_or_none(record)
    if mean is None or median is None or median <= 0.0:
        return None
    return (median - mean) / median * 100.0


def _busy_or_none(record: Mapping[str, Any]) -> int | None:
    summary = record.get("sm_clock_mhz")
    if not isinstance(summary, Mapping):
        return None
    value = summary.get("n")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _idle_or_none(record: Mapping[str, Any]) -> int | None:
    value = record.get("idle_samples_excluded")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


# --------------------------------------------------------------------------
# The sampler. Same shape as tools/bench/sample_process_memory.py: a background
# process for the length of the measured window, one artifact of raw samples
# and one summary, and a stop condition the caller already owns.
# --------------------------------------------------------------------------


class _Stop:
    def __init__(self) -> None:
        self.requested = False

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        self.requested = True


def run_sampler(
    *,
    samples_output: pathlib.Path,
    summary_output: pathlib.Path,
    interval_s: float = 1.0,
    max_duration_s: float | None = None,
    smi: str = NVIDIA_SMI,
    boot_id_path: pathlib.Path = BOOT_ID_PATH,
) -> dict[str, Any]:
    if interval_s <= 0.0:
        raise HarnessError("clock sampling interval must be positive")
    if max_duration_s is not None and max_duration_s <= 0.0:
        raise HarnessError("clock sampling max duration must be positive")
    for path in (samples_output, summary_output):
        if path.exists():
            raise HarnessError(f"refusing to overwrite clock evidence: {path}")
    boot_id = read_boot_id(boot_id_path)
    stop = _Stop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    collected: list[dict[str, Any]] = []
    with samples_output.open("w", encoding="utf-8", newline="\n") as sink:
        while True:
            sample = query_once(smi=smi)
            sample["elapsed_s"] = time.monotonic() - start
            sample["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            sink.write(canonical_json(sample) + "\n")
            sink.flush()
            collected.append(sample)
            if stop.requested:
                break
            if max_duration_s is not None and time.monotonic() - start >= max_duration_s:
                break
            time.sleep(interval_s)
            if stop.requested:
                break
    record = build_clock_record(collected, boot_id=boot_id)
    write_json_atomic(summary_output, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser(
        "sample", help="sample the SM clock across a measured window"
    )
    sample.add_argument("--output", type=pathlib.Path, required=True)
    sample.add_argument("--summary", type=pathlib.Path, required=True)
    sample.add_argument("--interval", type=float, default=1.0)
    sample.add_argument("--max-duration", type=float)

    check = subparsers.add_parser(
        "compare", help="compare two recorded arms and print the clock block"
    )
    check.add_argument("--ours", type=pathlib.Path, required=True)
    check.add_argument("--vllm", type=pathlib.Path, required=True)
    check.add_argument(
        "--allow-cross-boot",
        action="store_true",
        help="waive boot IDENTITY (never state); records a caveat, not silence",
    )

    args = parser.parse_args(argv)
    if args.command == "sample":
        record = run_sampler(
            samples_output=args.output,
            summary_output=args.summary,
            interval_s=args.interval,
            max_duration_s=args.max_duration,
        )
        print(canonical_json(record))
        return 0

    # `json` is imported at module scope since `read_sample_stream` needed it;
    # the function-local `import json as _json` this replaced is now redundant.
    comparison = compare_clock_records(
        json.loads(args.ours.read_text(encoding="utf-8")),
        json.loads(args.vllm.read_text(encoding="utf-8")),
        allow_cross_boot=args.allow_cross_boot,
    )
    print(canonical_json(comparison))
    return 0 if not comparison["reasons"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"gpu-clock-state: {error}", file=sys.stderr)
        raise SystemExit(2) from error
