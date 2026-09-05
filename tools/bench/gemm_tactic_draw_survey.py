#!/usr/bin/env python3
"""Take N INDEPENDENT kernel-selection draws on one leased device, and judge them.

WHAT THIS ANSWERS
-----------------
Three filed issues, all of which ask for a MEASUREMENT and none of which asks
for a feature:

* #2750 (`KERNEL-GEMM-BF16`) -- `src/vt/cuda/gemm_plan_cache.h` and
  `src/vt/cuda/fp8_plan_cache.h` both justify caching the cuBLASLt heuristic
  with a PER-PROCESS determinism premise. Nothing has ever measured the
  CROSS-PROCESS question. Every process re-resolves the heuristic, so if the
  answer is "not stable" then every same-binary A/B on that lane carried an
  uncontrolled variable that shows up as spread rather than as an error.
* #2751 (`KERNEL-GEMM-NVFP4-W4A4`) -- our NVFP4 tuner mirrors FlashInfer
  0.6.13's timing recipe (3 warmups / 10 iterations / 5,000 us delay;
  `src/vt/cuda/nvfp4_persistent_cache.h:25-27`). **Read the aggregation
  carefully**: `TimeCandidate` at
  `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:322-363` records ONE `cudaEvent`
  pair spanning ALL TEN iterations and returns `elapsed_ms / kTimingIterations`.
  That is a MEAN, and no per-iteration timing exists anywhere -- there is no
  distribution in the tree to take a median or a minimum of. "Minimum wins"
  describes the argmin ACROSS CANDIDATES at `:758-764`, never a min-of-10.
  One draw per process, kept whole. The draw distribution is known to be WIDE
  in identity (18--33 of 64 shared tactic IDs across paired runs). It has never
  been measured in SPEED.
* #2752 (`KERNEL-GEMM-NVFP4-W4A4`) -- which draw would we ship as a pinned GB10
  artifact. That is blocked on #2751's answer, because "which draw" is only
  arbitrary if the draws are performance-equivalent.

TWO ARMS SELECT BY DIFFERENT RULES, SO A DRAW SPREAD MUST NAME ITS ARM
----------------------------------------------------------------------
`Fp4FullTacticsEnabled()` (`:189-195`) reads
`value == nullptr || value[0] != '0'`, so `VT_FP4_FULL_TACTICS` is **default
ON**: the shipped arm is 32 candidates chosen by pure argmin. The non-default
`=0` arm is 4 candidates AND carries a variance damper (`:770-776`) -- a tactic
displaces the fixed baseline only if it beats it by more than 1%, otherwise the
baseline is kept.

So the two arms answer different questions and their draw spreads are not
comparable. This module refuses an evidence root that mixes them (exit `81`)
rather than reporting a number that does not say which rule produced it.

THE TWO INSTRUMENTS ALREADY EXIST. NEITHER IS BUILT HERE.
---------------------------------------------------------
* `VT_GEMM_ALGO_LOG=1` makes `MaybeLogGemmAlgo`
  (`src/vt/cuda/cuda_matmul.cu:248`) emit ONE stderr line per unique
  (shape, dtype-combo, epilogue) key naming the cuBLASLt algo config.
  `LogOncePerKey` (`src/vt/cuda/gemm_algo_log.h`) dedupes WITHIN a process, so
  diffing those line sets ACROSS fresh processes IS the #2750 experiment.
* `[VT_FP4_CACHE] prepared/complete/selected`
  (`src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:610,657,673`) reports the loaded /
  tuned / rejected / saved counts, the metadata fingerprint, and the WHOLE
  selected plan map, one line per plan. `VT_FP4_AUTOTUNE_CACHE_PATH` points a
  process at its own cache document and `VT_FP4_AUTOTUNE_CACHE_READONLY=1`
  freezes it, which is how N independent draws are collected and then replayed.
* `VT_FP4_AUTOTUNE_VERBOSE=1` adds one `[VT_FP4_AUTOTUNE] ... -> id=%d %s
  (%.1f us)` line per tuned key (`:777-788`). That microsecond figure is
  `timings[chosen] * 1000`, i.e. the CHOSEN candidate's ten-iteration mean, and
  it is the only timing number the tuner exposes. It is a DIAGNOSTIC here and
  never a gate: it is a single mean with no spread, produced by the very
  instrument whose noise is under suspicion, so ranking draws by it would be
  selecting on the measurement in question.

So this module adds no product code and changes no kernel. It runs the existing
instruments, PARSES them, and refuses to report a verdict the evidence does not
carry.

WHY THE JUDGEMENT IS SEPARATE FROM THE DEVICE
---------------------------------------------
Standard library only, and every predicate below is a pure function of text
that a CPU can produce. `gpu_clock_state.py` and `resumable_legs.py` chose that
polarity for the same reason: the half that decides has to be testable without
the hardware whose scarcity is the reason the decision matters.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does NOT choose a draw to ship by ranking draws on the workload the shipped
draw will later be gated on. `.agents/benchmarking.md` and #2751 both refuse
that shape by name: selecting a kernel plan on the same measurement it will
later be scored against is measuring around the harness. `select_shipping_draw`
below therefore returns a draw ONLY when the draws are performance-equivalent,
and then it returns the FIRST draw in draw order -- a rule fixed before any
number was taken, which no measurement can bias. When the draws are NOT
equivalent it returns no draw at all and names #2751 as the blocker, which is
exactly what #2752's own "blocked-by consideration" paragraph asks for.

It also does not run the legs of the SCORING phase. `tools/bench/c8_leg_runner.py`
already owns that -- interleaved plan, append-on-completion ledger, resume,
cross-boot refusal, terminal control -- and a second leg runner in this tree
would be a second set of those rules to keep in agreement.

EXIT CODES, AND WHY THEY ARE NAMED
----------------------------------
A silent instrument reads as a result. A draw phase that produced zero
`[VT_GEMM_ALGO]` lines must not report "stable"; a scoring leg that re-tuned
instead of loading a frozen map measured its own draw and not the one it was
asked about. Every such condition has its own code so a failed run says WHICH
precondition failed, in the exit status, without reading the log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Exit codes. 0 is "the run completed and the verdict is in the report"; the
# verdict itself is NOT encoded here, because "the draws are unstable" is a
# successful measurement, not a harness failure.
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DRAW_FAILED = 60          # a draw process exited non-zero
EXIT_ALGO_SILENT = 70          # zero [VT_GEMM_ALGO] lines: instrument did not run
EXIT_ALGO_NO_BF16 = 71         # lines exist, none with a bf16 input: says nothing about KERNEL-GEMM-BF16
EXIT_FP4_SILENT = 72           # missing [VT_FP4_CACHE] prepared/complete
EXIT_DRAW_NOT_INDEPENDENT = 73  # a "draw" loaded a map instead of tuning one
EXIT_KEYSET_DIFFERS = 74       # draws tuned different plan keys: not comparable
EXIT_FINGERPRINT_DIFFERS = 75  # different device/driver/build identity across draws
EXIT_ALGO_KEYSET_DIFFERS = 76  # processes saw different cuBLASLt shapes: different workloads
EXIT_CACHE_MISSING = 77        # a draw published no cache document
EXIT_LEG_NOT_FROZEN = 78       # a scoring leg tuned instead of loading frozen
EXIT_BINARY_DIFFERS = 79       # two legs are two binaries; only the draw may vary
EXIT_ARM_MIXED = 81            # one evidence root holds both tactic-set arms


# ---------------------------------------------------------------------------
# The two thresholds the speed half is judged against, ONE DEFINITION EACH.
#
# Both are 1.02 and they mean different things, which is written out at length
# in `speed_spread` and in `.agents/specs/nvfp4-persistent-plan-cache.md` W3-F.
# They are two constants rather than one alias BECAUSE they share a value: the
# spec says moving one must not move the other, and a single name would make
# that impossible to obey.
#
# THE POINT OF NAMING THEM IS THAT THE VALUE HAD THREE COPIES. `speed_spread`'s
# keyword default was the only one under test, while `reduce_evidence`'s default
# and the `reduce` subparser's `--noise-ceiling` default carried their own
# literals -- and the subparser's is the copy the spec's published gate command
# actually runs at, because that command passes no `--noise-ceiling`. Widening
# either of the other two left the whole suite green, so the number governing a
# lease was the one nothing asserted. A literal written once cannot drift from
# itself.
DEFAULT_RATIFICATION_BAR = 1.02
DEFAULT_NOISE_CEILING = 1.02


# ---------------------------------------------------------------------------
# Parsing. Every regex below is deliberately UNANCHORED.
# A `^`-anchored grep over this tree's logs has already failed on lines a
# harness had prefixed with a timestamp or an ANSI colour reset, and an anchored
# pattern that matches nothing is indistinguishable from an instrument that
# emitted nothing -- which is the exact confusion the exit codes above exist to
# remove.
# ---------------------------------------------------------------------------
ALGO_TAG = "[VT_GEMM_ALGO]"
FP4_TAG = "[VT_FP4_CACHE]"

_BF16_TAGS = frozenset({"bf16", "BF16", "b16", "16bf"})


def kv_tokens(line: str) -> dict[str, str]:
    """Split a `k=v k=v` diagnostic line into a mapping, parentheses included.

    Written as a tokenizer rather than as one big regex because of two shapes
    the emitted lines actually have, both of which a `(\\S+)`-per-field regex
    reads WRONG rather than failing on:

    1.  **Empty values.** `prepared` prints `native=%s` and `flashinfer=%s`
        straight from a `std::filesystem::path`, and an unset FlashInfer path
        yields the bare token `flashinfer=`. A greedy group swallows the NEXT
        key when that happens.
    2.  **A parenthesised breakdown that REUSES the outer key names.**
        `prepared` prints `loaded=%llu (flashinfer=%llu native=%llu)`, so the
        tokens `flashinfer` and `native` appear TWICE on one line -- once as a
        path and once as a count. A flat last-wins mapping silently replaces
        the path with the count and nothing says so. Tokens inside parentheses
        are therefore stored under an `in_` prefix (`in_flashinfer`,
        `in_native`), and a repeated key keeps its FIRST value, so the outer
        meaning of a name is the one that survives.

    Neither shape is hypothetical: both are in the format strings at
    `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:610`.
    """

    out: dict[str, str] = {}
    depth = 0
    for token in line.split():
        opened = token.startswith("(")
        closed = token.endswith(")")
        inside = depth > 0 or opened
        if opened:
            depth += 1
        stripped = token.strip("()")
        if closed and depth > 0:
            depth -= 1
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if not key:
            continue
        if inside:
            key = "in_" + key
        out.setdefault(key, value)
    return out


def algo_key(fields: Mapping[str, str]) -> str:
    """The dedupe key `LogOncePerKey` uses, rebuilt from the emitted line.

    It must be the SELECTION key and must NOT include the selection itself:
    the whole experiment is whether one key answers with one config across
    processes, and folding `algoId` into the key would make every run agree by
    construction.
    """

    return (
        f"{fields.get('backend', '?')}|m={fields.get('m', '?')}"
        f" n={fields.get('n', '?')} k={fields.get('k', '?')}"
        f"|a={fields.get('a', '?')} b={fields.get('b', '?')}"
        f" c={fields.get('c', '?')}|{fields.get('epilogue', '?')}"
    )


def algo_selection(fields: Mapping[str, str]) -> tuple[str, str, str, str]:
    """The four config values the issue asks about, as a comparable tuple.

    `wsSize` is READ but is not part of this tuple. It is the heuristic's
    workspace estimate, which the same algo can report differently under a
    different workspace budget; folding it in would report a selection change
    that never happened.
    """

    return (
        fields.get("algoId", "?"),
        fields.get("tile", "?"),
        fields.get("stages", "?"),
        fields.get("splitK", "?"),
    )


# The `[VT_GEMM_ALGO]` tag has SIX emit sites and only ONE of them is a
# selection. All six are behind the same `GemmAlgoLogEnabled()`, so a run that
# turns the instrument on gets all six in one stderr:
#
#   src/vt/cuda/cuda_matmul.cu:270  SELECTION        MaybeLogGemmAlgo
#   src/vt/cuda/cuda_matmul.cu:805  REFUSED          MaybeLogFp8PlanRefusal
#   src/vt/cuda/cuda_matmul.cu:842  DENSE-CANDIDATES MaybeLogDenseAlgoCandidates, query failed
#   src/vt/cuda/cuda_matmul.cu:858  DENSE-CANDIDATE  MaybeLogDenseAlgoCandidates, one per candidate
#   src/vt/cuda/cuda_matmul.cu:883  CANDIDATES       MaybeLogFp8AlgoCandidates, query failed
#   src/vt/cuda/cuda_matmul.cu:900  CANDIDATE rank=  MaybeLogFp8AlgoCandidates, one per candidate
#
# The dense dump is reached from the TN-bt lane (`:582`) -- the bf16 lane #2750
# is about -- so these are in the same stderr as the selections, not in some
# other workload. `a=`, `b=`, `c=` and `epilogue=` appear on the SELECTION line
# and on none of the other five, which is what discriminates them; `algoId=` is
# additionally required because it is what the comparison reads.
_SELECTION_DISCRIMINANTS = ("a", "b", "c", "epilogue", "algoId")


def is_algo_selection(fields: Mapping[str, str]) -> bool:
    """Does this parsed `[VT_GEMM_ALGO]` line name a SELECTION?

    A REFUSED line names a plan that was never built. A CANDIDATE line names one
    entry of a heuristic LIST, and rank 0 of a list is not what the matmul ran.
    Reading either as a selection is not a cosmetic parse error: the candidate
    dumps repeat one shape once per ranked candidate, a first-wins map keeps
    rank 0, and two processes whose lists came back in a different order then
    report `UNSTABLE`. That verdict is the one that turns #2750 into a
    benchmark-validity repair and licenses the persistent cuBLASLt cache #2750
    says must be built ONLY in its case 4.

    Stated as a POSITIVE requirement on the selection line's own fields rather
    than as a blacklist of the five diagnostic tags, so a seventh diagnostic
    shape added later is excluded by default. The failure mode of that polarity
    is a selection line that stops carrying one of these fields, which reads as
    an instrument that emitted nothing (exit `70`) -- loud, and refused by name.
    """

    return all(name in fields for name in _SELECTION_DISCRIMINANTS)


def parse_algo_lines(text: str) -> dict[str, dict[str, str]]:
    """Every `[VT_GEMM_ALGO]` SELECTION in one process's stderr, keyed by its key.

    Diagnostic shapes are counted under `_diagnostics` and excluded, because
    "zero `[VT_GEMM_ALGO]` lines" and "zero selections among forty lines" are
    different diagnoses of a failed run and the refusal message says which.

    A duplicate key inside ONE process would mean `LogOncePerKey` failed, so it
    is recorded rather than merged: `_duplicates` counts them and the caller
    reports the count instead of averaging over an instrument that misbehaved.
    That counter is only meaningful once the candidate dumps are excluded --
    they repeat one shape by construction, so before the exclusion it counted
    the instrument working correctly.
    """

    found: dict[str, dict[str, str]] = {}
    duplicates = 0
    diagnostics = 0
    for line in text.splitlines():
        if ALGO_TAG not in line:
            continue
        fields = kv_tokens(line[line.index(ALGO_TAG) + len(ALGO_TAG):])
        if not is_algo_selection(fields):
            diagnostics += 1
            continue
        key = algo_key(fields)
        if key in found:
            duplicates += 1
            continue
        found[key] = dict(fields)
    if duplicates:
        found["_duplicates"] = {"count": str(duplicates)}
    if diagnostics:
        found["_diagnostics"] = {"count": str(diagnostics)}
    return found


def parse_fp4_lines(text: str) -> dict[str, Any]:
    """The `prepared`, `complete` and `selected` half of one process's stderr.

    `selected` is emitted once per plan by `CompletePersistentRuntime`, so the
    map it builds is the process's WHOLE draw and not a sample of it.
    """

    prepared: dict[str, str] | None = None
    complete: dict[str, str] | None = None
    selected: dict[str, int] = {}
    for line in text.splitlines():
        if FP4_TAG not in line:
            continue
        rest = line[line.index(FP4_TAG) + len(FP4_TAG):].strip()
        kind = rest.split(" ", 1)[0] if rest else ""
        fields = kv_tokens(rest)
        if kind == "prepared":
            prepared = fields
        elif kind == "complete":
            complete = fields
        elif kind == "selected":
            try:
                key = f"{int(fields['M'])},{int(fields['N'])},{int(fields['K'])}"
                selected[key] = int(fields["tactic"])
            except (KeyError, ValueError):
                continue
    return {"prepared": prepared, "complete": complete, "selected": selected}


AUTOTUNE_TAG = "[VT_FP4_AUTOTUNE]"

# The selection line is a dedicated regex rather than `kv_tokens`, because it is
# not a `k=v` line: `M=%d(bucket=%u)` fuses two fields into one token and
# `-> id=%d %s (%.1f us)` puts the tactic name and the time outside the k=v
# shape entirely. Unanchored, like every other pattern here.
AUTOTUNE_RE = re.compile(
    r"\[VT_FP4_AUTOTUNE\]\s+set=(?P<set>\S+)\s+M=(?P<m>-?\d+)\(bucket=(?P<bucket>\d+)\)"
    r"\s+N=(?P<n>-?\d+)\s+K=(?P<k>-?\d+)\s+device=(?P<device>-?\d+)\s+sm=(?P<sm>\d+)"
    r"\s+delay_us=(?P<delay>\d+)\s+->\s+id=(?P<id>-?\d+)\s+(?P<name>.+?)"
    r"\s+\((?P<us>[0-9.]+)\s+us\)"
)


def parse_autotune_lines(text: str) -> dict[str, Any]:
    """Every tactic SELECTION the tuner announced, keyed like the plan map.

    The key is `bucket,N,K`, which is exactly what `[VT_FP4_CACHE] selected`
    prints (it emits `plan.m_bucket` under the name `M`), so the two maps join
    without a translation step.

    `us` IS A SINGLE MEAN AND THE CALLER MUST NOT TREAT IT AS A SAMPLE.
    `TimeCandidate` wraps all ten iterations in ONE `cudaEvent` pair and divides,
    so the tuner never holds a per-iteration figure and none can be recovered
    from this line. A median or a minimum over iterations is not a reduction
    this harness -- or a future selector -- can switch to; it would first need
    per-iteration events that do not exist.

    The value recorded is `timings[chosen]`, not `timings[best]`. In the
    non-default `VT_FP4_FULL_TACTICS=0` arm the damper can keep the baseline
    even when another candidate timed faster, so `chosen` and `best` differ
    there and this line reports the one that was installed.

    A key announced twice means the single-flight tuner ran twice for it (a
    lazy miss after the pre-serve warmup). The FIRST is kept and the repeat is
    counted, because silently overwriting would hide the miss.
    """

    found: dict[str, dict[str, Any]] = {}
    sets: set[str] = set()
    repeats = 0
    for line in text.splitlines():
        match = AUTOTUNE_RE.search(line)
        if match is None:
            continue
        sets.add(match.group("set"))
        key = f"{match.group('bucket')},{match.group('n')},{match.group('k')}"
        if key in found:
            repeats += 1
            continue
        found[key] = {
            "tactic_id": int(match.group("id")),
            "name": match.group("name"),
            "mean_us": float(match.group("us")),
            "m": int(match.group("m")),
        }
        found[key]["set"] = match.group("set")
    return {"selections": found, "sets": sorted(sets), "repeat_selections": repeats}


_BENCH_FIELDS = {
    "total_token_throughput": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "output_token_throughput": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "successful_requests": r"Successful requests:\s+([0-9]+)",
    "total_generated_tokens": r"Total generated tokens:\s+([0-9]+)",
    "total_input_tokens": r"Total input tokens:\s+([0-9]+)",
    "duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
}


def parse_bench_report(text: str) -> dict[str, float]:
    """The `vllm-bench` report block, as far as it is present.

    A MISSING field is omitted rather than defaulted. `.agents/benchmarking.md`
    §"Two arms have to BE two arms" turns on the COUNTS -- equal times are
    noise, equal counts are identity -- so a run whose request count did not
    parse must read as absent, never as zero.
    """

    out: dict[str, float] = {}
    for name, pattern in _BENCH_FIELDS.items():
        match = re.search(pattern, text)
        if match:
            try:
                out[name] = float(match.group(1))
            except ValueError:
                continue
    return out


# ---------------------------------------------------------------------------
# The predicates. Pure functions of parsed evidence; no I/O, no device.
# ---------------------------------------------------------------------------
def algo_stability(runs: Mapping[str, Mapping[str, Mapping[str, str]]]) -> dict[str, Any]:
    """#2750: does one selection key answer with ONE cuBLASLt config across processes?

    `runs` maps a run label to that run's `parse_algo_lines` result.

    THE KEY-SET IDENTITY CHECK IS PART OF THE ANSWER, NOT A PRELIMINARY. A run
    that saw fewer shapes ran a different workload, and comparing it against a
    run that saw more would report "stable" over the intersection while hiding
    that the two processes did not do the same thing. So a key present in some
    runs and absent from others is reported as `keys_partial` and makes the
    verdict `INCOMPARABLE`, never `STABLE`.

    The verdict is SCOPED to the keys actually observed and says so. cuBLASLt
    heuristic determinism over four shapes is not a claim about the heuristic.
    """

    labels = sorted(runs)
    key_sets = {label: {k for k in runs[label] if not k.startswith("_")} for label in labels}
    if not labels:
        return {"verdict": "INCOMPARABLE", "reason": "no runs", "runs": 0}

    common: set[str] = set.intersection(*key_sets.values()) if key_sets else set()
    union: set[str] = set.union(*key_sets.values()) if key_sets else set()
    partial = sorted(union - common)

    unstable: dict[str, dict[str, list[str]]] = {}
    for key in sorted(common):
        by_selection: dict[tuple[str, str, str, str], list[str]] = {}
        for label in labels:
            by_selection.setdefault(algo_selection(runs[label][key]), []).append(label)
        if len(by_selection) > 1:
            unstable[key] = {
                "algoId=%s tile=%s stages=%s splitK=%s" % sel: sorted(who)
                for sel, who in sorted(by_selection.items())
            }

    bf16_keys = sorted(
        key for key in common
        if runs[labels[0]][key].get("a", "") in _BF16_TAGS
    )
    duplicates = sum(
        int(runs[label].get("_duplicates", {}).get("count", "0")) for label in labels
    )

    if len(labels) < 2:
        verdict = "INCOMPARABLE"
        reason = "one process cannot answer a cross-process question"
    elif duplicates:
        # A repeated SELECTION key inside one process means the within-process
        # dedupe did not hold, and the parser kept the FIRST line and dropped
        # the rest. The comparison would then run over a first-wins subset of
        # what the process said, which is not a stable run with a footnote.
        verdict = "INCOMPARABLE"
        reason = (
            f"{duplicates} repeated selection line(s) inside a single process: "
            "LogOncePerKey did not dedupe, or two processes' stderr were "
            "concatenated. Only the first line per key was kept, so this "
            "comparison would run over a subset of what was emitted"
        )
    elif partial:
        verdict = "INCOMPARABLE"
        reason = (
            f"{len(partial)} of {len(union)} keys are missing from at least one "
            "run, so the processes did not run the same workload"
        )
    elif not common:
        verdict = "INCOMPARABLE"
        reason = "no key was observed in every run"
    elif unstable:
        verdict = "UNSTABLE"
        reason = (
            f"{len(unstable)} of {len(common)} keys selected more than one "
            "cuBLASLt config across processes"
        )
    else:
        verdict = "STABLE"
        reason = (
            f"all {len(common)} observed keys selected one config across "
            f"{len(labels)} fresh processes"
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "runs": len(labels),
        "keys_common": len(common),
        "keys_partial": partial,
        "keys_bf16_input": len(bf16_keys),
        "bf16_keys": bf16_keys,
        "unstable_keys": unstable,
        "within_process_duplicate_lines": duplicates,
        "scope": (
            "This verdict covers exactly the selection keys this workload "
            "produced. It is not a claim about the cuBLASLt heuristic in "
            "general, and a shape absent from `keys_common` is unmeasured."
        ),
    }


def draw_identity(draws: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """#2751, identity half: how far apart are the tactic maps of N draws?

    Reports the pairwise agreement the way the spec's existing evidence does --
    "shared X of 64 selected tactic IDs" -- plus the per-key count of distinct
    tactics, which is the statistic that says whether the spread is spread over
    a few keys or over all of them.
    """

    labels = sorted(draws)
    if len(labels) < 2:
        return {"verdict": "INCOMPARABLE", "reason": "fewer than two draws",
                "draws": len(labels)}

    key_sets = {label: set(draws[label]) for label in labels}
    common = set.intersection(*key_sets.values())
    union = set.union(*key_sets.values())
    if union - common:
        return {
            "verdict": "INCOMPARABLE",
            "reason": (
                f"{len(union - common)} plan keys are absent from at least one "
                "draw; the draws tuned different key sets, so their tactic maps "
                "are not comparable"
            ),
            "draws": len(labels),
            "keys_partial": sorted(union - common),
        }

    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = sum(1 for key in common if draws[a][key] == draws[b][key])
            pairs.append({"a": a, "b": b, "shared": shared, "of": len(common)})

    per_key = {key: len({draws[label][key] for label in labels}) for key in common}
    unanimous = sorted(key for key, n in per_key.items() if n == 1)
    shared_counts = sorted(pair["shared"] for pair in pairs)

    return {
        "verdict": "MEASURED",
        "draws": len(labels),
        "keys": len(common),
        "pairwise_shared_min": shared_counts[0],
        "pairwise_shared_max": shared_counts[-1],
        "pairwise_shared_median": shared_counts[len(shared_counts) // 2],
        "keys_unanimous": len(unanimous),
        "keys_with_multiple_tactics": len(common) - len(unanimous),
        "max_distinct_tactics_on_one_key": max(per_key.values()) if per_key else 0,
        "pairs": pairs,
    }


def selection_time_spread(draws: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """DIAGNOSTIC ONLY: how far apart the draws' own selection-time means are.

    `draws` maps a draw label to that draw's `parse_autotune_lines` selections.

    **This is not a gate and must never become one.** Three reasons, and each
    alone is sufficient:

    1.  The number is ONE ten-iteration mean per key. `TimeCandidate` records a
        single `cudaEvent` pair around the whole loop, so no per-iteration
        sample exists and no spread, median or minimum is computable from it.
    2.  It is produced by the instrument whose noise is the subject of the
        investigation. Ranking draws by the tuner's own timing would settle the
        question with the measurement under suspicion.
    3.  It is measured on the tuner's synthetic warmup shapes, which is exactly
        the workload the draw was taken on. Scoring a draw on the workload that
        produced it is the shape #2751 and the benchmark protocol refuse.

    What it IS good for: saying whether the draws' installed candidates even
    claimed different times. Two draws that picked different tactic IDs at
    near-identical selection-time means are the "performance-equivalent
    selections differing only in reduction order" hypothesis made visible; a
    large selection-time gap is a reason to look harder at the end-to-end
    numbers, never a substitute for them.
    """

    labels = sorted(draws)
    if len(labels) < 2:
        return {"state": "INCOMPARABLE", "reason": "fewer than two draws"}

    key_sets = {label: set(draws[label]) for label in labels}
    common = set.intersection(*key_sets.values())
    union = set.union(*key_sets.values())
    if not common:
        return {"state": "INCOMPARABLE", "reason": "no key was tuned in every draw"}

    per_key: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    for key in sorted(common):
        means = [draws[label][key]["mean_us"] for label in labels]
        ids = sorted({draws[label][key]["tactic_id"] for label in labels})
        low, high = min(means), max(means)
        ratio = high / low if low > 0 else None
        if ratio is not None:
            ratios.append(ratio)
        per_key[key] = {
            "min_us": low,
            "max_us": high,
            "max_over_min": ratio,
            "distinct_tactic_ids": ids,
        }

    ratios.sort()
    return {
        "state": "DIAGNOSTIC",
        "keys": len(common),
        "keys_missing_from_some_draw": len(union - common),
        "max_over_min_min": ratios[0] if ratios else None,
        "max_over_min_median": ratios[len(ratios) // 2] if ratios else None,
        "max_over_min_max": ratios[-1] if ratios else None,
        "per_key": per_key,
        "not_a_gate": (
            "One ten-iteration MEAN per key, from a single cudaEvent pair; no "
            "per-iteration sample exists in the tree. Measured by the tuner "
            "under suspicion, on the tuner's own warmup shapes. Report it, "
            "never rank draws by it."
        ),
    }


def speed_spread(
    per_draw: Mapping[str, Sequence[float]],
    *,
    ratification_bar: float = DEFAULT_RATIFICATION_BAR,
    noise_ceiling: float = DEFAULT_NOISE_CEILING,
    min_legs: int = 3,
) -> dict[str, Any]:
    """#2751, speed half: is the draw-to-draw spread bigger than the harness noise?

    `per_draw` maps a draw label to its scoring legs' metric values (higher is
    better; the harness feeds total token throughput).

    THE BAR IS ARGUED, NOT PICKED. The frozen-plan steady-state component on
    this lane measured c2 1.0045x / c16 1.0050x and STRICT-FAILED at 39/40
    timing + 1/8 memory (`.agents/specs/nvfp4-persistent-plan-cache.md`
    §"W3-C3 corrected frozen-plan component result"). A draw selector is a
    DELIBERATE DIVERGENCE from the pinned FlashInfer selection method, so it
    has to be worth more than the control that lane already has. The default
    2% bar is four times that control's magnitude and sits above both the 1%
    cross-arm clock-mean rule and the 5% within-run clock-spread ceiling that
    `.agents/benchmarking.md` applies to a lease, where the clock cannot be
    pinned at all.

    THE NOISE FLOOR IS NOT THE WORST DRAW IN THE RUN, AND A RUN THAT DID NOT
    HOLD STILL IS REPEATED RATHER THAN CALLED EQUIVALENT. Two separate rules,
    both learned the hard way on this host:

    1.  `min_legs` is THREE. A within-draw spread over two legs is one
        difference, not an estimate of anything, and the first revision of this
        harness took the run's floor from such a figure at the spec's own
        `--score-reps 2`.
    2.  A draw whose own spread is UNDEFINED -- a non-positive leg, which is
        what a leg that produced no tokens records -- is refused by name. Both
        rules below read a per-draw spread, and a draw missing from that
        population is exempt from both while its mean still decides the ratio.
    3.  `noise_ceiling` refuses the run outright when ANY draw's own repeat
        spread exceeds it. Without that, one restless draw sets a floor that
        certifies every other draw as equivalent -- and `EQUIVALENT` is the
        verdict that closes #2751 as "no divergence warranted" and unblocks
        #2752 to pin a draw. One unchanged binary has read 36.82 and 78.86
        tok/s at c8 on this box (`c8-measurement-admissibility.md`, #2154), so a
        wide within-draw spread is what a box that did not hold still looks
        like, not a rarity. `INCOMPARABLE` here means REPEAT THE RUN.

    The ceiling is its OWN parameter and its own default, deliberately not
    `ratification_bar`. They share the value 1.02 and they mean different
    things: the bar is how much a draw-to-draw gap must be worth before a
    divergence from the mirrored FlashInfer method is arguable, and the ceiling
    is how much repeat noise a run may carry before it can answer that question
    at all. A run whose own repeats are as wide as the effect it was built to
    resolve has not measured the effect. Moving one must never move the other.

    The comparison floor is then the POOLED within-draw spread -- the median of
    the per-draw figures -- rather than the maximum, so one draw at the edge of
    the ceiling cannot set the floor for all of them. Both are reported.

    A spread the pooled WITHIN-draw repeat spread already covers is not a
    spread. It is reported as `EQUIVALENT` and #2751 closes as "no divergence
    warranted", which is the outcome the issue names first.
    """

    labels = sorted(per_draw)
    usable = {label: [v for v in per_draw[label] if v is not None] for label in labels}
    if len(labels) < 2:
        return {"verdict": "INCOMPARABLE", "reason": "fewer than two draws scored"}

    thin = sorted(label for label in labels if len(usable[label]) < min_legs)
    means = {label: sum(v) / len(v) for label, v in usable.items() if v}

    within = {
        label: (max(v) - min(v)) / min(v) if len(v) >= 2 and min(v) > 0 else None
        for label, v in usable.items()
    }
    # A DRAW WHOSE SPREAD IS `None` IS EXEMPT FROM BOTH GUARDS BELOW AND STILL
    # MOVES THE RATIO, so it is refused by name rather than dropped. `within` is
    # `None` when the draw's minimum leg is not positive, and a zero is what a
    # leg records when its run produced no tokens: `parse_bench_report` reads
    # the field rather than omitting it. Dropped, such a draw took no part in
    # the ceiling and no part in the pooled floor while its own depressed mean
    # decided which draw was worst -- so the wildest draw in the run was the one
    # neither guard could see, and the reported spread described the others.
    unspreadable = sorted(
        label for label in labels if usable[label] and within[label] is None
    )
    within_values = sorted(w for w in within.values() if w is not None)
    worst_within = within_values[-1] if within_values else None
    pooled_within = statistics.median(within_values) if within_values else None

    best_label = max(means, key=lambda label: means[label]) if means else None
    worst_label = min(means, key=lambda label: means[label]) if means else None
    ratio = (
        means[best_label] / means[worst_label]
        if best_label is not None and means[worst_label] > 0
        else None
    )

    if len(means) < 2:
        verdict = "INCOMPARABLE"
        reason = "fewer than two draws produced a number"
    elif thin:
        verdict = "INCOMPARABLE"
        reason = (
            f"{len(thin)} draw(s) carry fewer than {min_legs} usable legs, so "
            "their own repeat spread is one difference rather than an estimate "
            "and a draw-to-draw gap cannot be distinguished from it. Three legs "
            "per draw is the minimum this predicate accepts"
        )
    elif unspreadable:
        verdict = "INCOMPARABLE"
        reason = (
            f"{', '.join(unspreadable)}: a scored draw carries a non-positive "
            "leg, so its own repeat spread is undefined. It cannot be measured "
            "against the noise ceiling or contribute to the pooled floor, and a "
            "run judged with that draw exempt would quote a spread over the "
            "OTHER draws while this one set the ratio. Read the leg's own log: "
            "a zero is a leg that produced no tokens, not a slow one"
        )
    elif ratio is None:
        verdict, reason = "INCOMPARABLE", "a scored draw has a non-positive mean"
    elif worst_within is not None and 1.0 + worst_within > noise_ceiling:
        verdict = "INCOMPARABLE"
        reason = (
            f"the worst within-draw repeat spread {1.0 + worst_within:.6f} "
            f"exceeds the {noise_ceiling:.4f}x noise ceiling: this run's own "
            "repeats are as wide as the effect it was asked about, so it cannot "
            "answer the question either way. REPEAT THE RUN"
        )
    elif pooled_within is not None and (ratio - 1.0) <= pooled_within:
        verdict = "EQUIVALENT"
        reason = (
            f"the best/worst draw ratio {ratio:.6f} does not exceed the pooled "
            f"within-draw repeat spread {1.0 + pooled_within:.6f}; this "
            "measurement does not separate the draws"
        )
    elif ratio < ratification_bar:
        verdict = "SEPARATED_BELOW_BAR"
        reason = (
            f"the best/worst draw ratio {ratio:.6f} exceeds the pooled "
            f"within-draw spread but is below the {ratification_bar:.4f}x bar a "
            "divergence from the mirrored FlashInfer selection method must clear"
        )
    else:
        verdict = "ABOVE_BAR"
        reason = (
            f"the best/worst draw ratio {ratio:.6f} clears the "
            f"{ratification_bar:.4f}x bar; #2751 asks for DEVELOPER "
            "RATIFICATION at this point, not for a selector to be written"
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "ratification_bar": ratification_bar,
        "noise_ceiling": noise_ceiling,
        "min_legs_per_draw": min_legs,
        "best_draw": best_label,
        "worst_draw": worst_label,
        "best_mean": means.get(best_label),
        "worst_mean": means.get(worst_label),
        "ratio": ratio,
        "worst_within_draw_spread": (
            None if worst_within is None else 1.0 + worst_within
        ),
        "pooled_within_draw_spread": (
            None if pooled_within is None else 1.0 + pooled_within
        ),
        # BOTH SPREADS ABOVE ARE POOLED OVER THIS LIST'S COMPLEMENT. Reporting
        # them without saying so is how a figure measured over one of two draws
        # reads as the run's own noise.
        "draws_without_a_spread": unspreadable,
        "legs_per_draw": {label: len(v) for label, v in usable.items()},
        "means": means,
    }


def select_shipping_draw(
    spread: Mapping[str, Any], draw_order: Sequence[str]
) -> dict[str, Any]:
    """#2752: which draw, if any, may be committed as the pinned GB10 artifact.

    THE ONLY SELECTION RULE THIS FUNCTION IMPLEMENTS IS "THE FIRST ONE", and
    that is the point. `.agents/benchmarking.md` and #2751 both refuse a
    selector that picks a plan on the workload the plan will later be scored
    on. Ranking N draws on workload S and shipping the winner is that shape
    even when S and the gate differ, because the winner was chosen BY a
    measurement of its own speed.

    So: when the draws are performance-equivalent the choice is arbitrary by
    construction, and an a-priori rule -- draw order, fixed before the first
    number existed -- is the one choice no measurement can bias. When they are
    NOT equivalent, this returns NO draw and names #2751 as the blocker, which
    is what #2752's own "blocked-by consideration" paragraph asks for.
    """

    verdict = spread.get("verdict")
    if verdict == "EQUIVALENT":
        if not draw_order:
            return {"ship": None, "reason": "no draws were taken"}
        return {
            "ship": draw_order[0],
            "rule": "first draw in draw order, fixed before any measurement",
            "reason": (
                "the draws are performance-equivalent on this measurement, so "
                "the pinned artifact is a reproducibility and warmup artifact "
                "and its identity is arbitrary; #2752 may proceed"
            ),
        }
    return {
        "ship": None,
        "reason": (
            f"draw speed verdict is {verdict}; #2752 is blocked on #2751 by its "
            "own blocked-by consideration -- which draw we ship stops being "
            "arbitrary as soon as the draws differ, and picking the fastest on "
            "this workload is the selector both issues refuse"
        ),
    }


# ---------------------------------------------------------------------------
# I/O half: run the draws, assert the preconditions, write evidence as it lands.
# ---------------------------------------------------------------------------
def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, payload: Any) -> None:
    """Write evidence the instant it exists, and fsync it.

    `dgx.casa` has gone down four times in one session (#545). A record held in
    memory until the end of a run is a record the next crash deletes, and the
    resume path below can only skip work whose evidence is already on disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def mirror_evidence(local: pathlib.Path, share: pathlib.Path | None) -> None:
    """Copy the evidence root to the share, best-effort, AFTER EACH DRAW.

    `dgx.casa` has crashed roughly hourly under a long ladder (#545) and the
    draw phase is the long one: N model loads and N tuning passes. Mirroring
    when the phase RETURNS loses every completed draw to a crash inside it,
    which is the crash this harness was shaped around.

    Best-effort on purpose. `/workspace` is CIFS and can be slow, full, or
    briefly absent, and none of those is a reason to throw away a draw that is
    already on the local disk and already has its `DONE` file. A failure to
    mirror costs the resume, never the measurement.
    """

    if share is None:
        return
    try:
        share.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local, share, symlinks=False, dirs_exist_ok=True)
    except OSError as error:  # CIFS, ENOSPC, a share that went away
        print(f"gemm-tactic-draw-survey: mirror to {share} failed: {error}",
              file=sys.stderr)


def draw_command(
    bench: pathlib.Path, model: str, cfg: Mapping[str, Any]
) -> list[str]:
    return [
        str(bench),
        "--model", model,
        "--num-prompts", str(cfg["num_prompts"]),
        "--input-len", str(cfg["input_len"]),
        "--output-len", str(cfg["output_len"]),
        "--concurrency", str(cfg["concurrency"]),
        "--seed", str(cfg["seed"]),
        "--temperature", "0",
        "--max-num-batched-tokens", str(cfg["max_num_batched_tokens"]),
    ]


def synthetic_draw_output(
    index: int, cfg: Mapping[str, Any], *, tactic_set: str = "full"
) -> tuple[str, str, int]:
    """A device-free fixture with the EXACT byte shape of the two instruments.

    This exists so `--dry-run` walks the whole record / resume / precondition /
    reduce path on a laptop, and so the parsers above are exercised against the
    format strings they claim to read rather than against a paraphrase. It is
    labelled `dry_run` in every record it writes and `reduce` prints that label
    at the top of its report, because a fixture that cannot be told apart from a
    measurement is worse than no fixture.

    Draws share a plan-key SET and differ in tactic, which is the shape the real
    evidence has (18--33 of 64 shared IDs).

    ONE of the four cuBLASLt shapes MOVES its selection with the draw index and
    the other three hold still, so the fixture reads `UNSTABLE` with exactly one
    unstable key out of four common keys. That asymmetry is deliberate and it is
    what makes the fixture able to detect a tainted dedupe key. When every draw
    agreed, folding `algoId` into `algo_key` -- mutation M10 -- changed no
    verdict and survived the whole suite; with one shape moving, the id in the
    key turns that shape into a different key per draw and the verdict collapses
    to `INCOMPARABLE` instead of naming the shape. The three still shapes are the
    control: without them the case could not tell a detected instability from a
    fixture that can never read stable.

    A fixture that reports a verdict must never be mistakable for a measurement,
    so every record it writes carries `dry_run: true` and `reduce` prints that
    label at the top of the report.
    """

    shapes = [(1, 3072, 2048), (1, 12288, 2048), (512, 3072, 2048), (512, 2048, 6144)]
    algo = []
    for position, (m, n, k) in enumerate(shapes):
        moves = 1 if position == 1 else 0
        algo.append(
            f"[VT_GEMM_ALGO] backend=cublasLt m={m} n={n} k={k} a=bf16 b=bf16 "
            f"c=bf16 epilogue=rowmajor-NN algoId={21 + (n % 3) + moves * index} "
            f"tile=15 stages=4 splitK=1 wsSize=4194304"
        )
    plans = []
    tuned = []
    for bucket in (1, 8, 64, 512):
        for n, k in ((3072, 2048), (2048, 6144)):
            tactic = (bucket + n + index * 7) % 32
            plans.append(f"[VT_FP4_CACHE] selected M={bucket} N={n} K={k} tactic={tactic}")
            tuned.append(
                f"[VT_FP4_AUTOTUNE] set={tactic_set} M={bucket}(bucket={bucket}) "
                f"N={n} K={k} device=0 sm=121 delay_us=5000 -> id={tactic} "
                f"tactic_{tactic} ({120.0 + index * 0.3:.1f} us), workspace=4194304"
            )
    prepared = (
        "[VT_FP4_CACHE] prepared mode=read-write native=/tmp/dry/autotune_configs.json "
        "flashinfer= loaded=0 (flashinfer=0 native=0) rejected=0 delay_us=5000 "
        f"metadata=dryrunfingerprint selected={len(plans)}"
    )
    complete = (
        f"[VT_FP4_CACHE] complete mode=read-write loaded=0 tuned={len(plans)} "
        f"rejected=0 saved={len(plans)} selected={len(plans)} metadata=dryrunfingerprint"
    )
    stderr = "\n".join(algo + tuned + [prepared, complete] + plans) + "\n"
    stdout = (
        "\n============= vllm.cpp Benchmark Result =============\n"
        f"Successful requests:                       {cfg['num_prompts']}\n"
        f"Total input tokens:                        {cfg['num_prompts'] * cfg['input_len']}\n"
        f"Total generated tokens:                    {cfg['num_prompts'] * cfg['output_len']}\n"
        "Benchmark duration (s):                    10.00\n"
        "Request throughput (req/s):                3.20\n"
        "Output token throughput (tok/s):           204.80\n"
        f"Total token throughput (tok/s):            {1840.0 + index:.2f}\n"
        "Mean TPOT (ms):                            9.77\n"
        "====================================================\n"
    )
    return stdout, stderr, 0


def run_draw(
    index: int,
    evidence: pathlib.Path,
    bench: pathlib.Path,
    model: str,
    cfg: Mapping[str, Any],
    *,
    tactic_set: str = "full",
    dry_run: bool = False,
    mirror: pathlib.Path | None = None,
) -> dict[str, Any]:
    """One fresh process: its own cache path, its own stderr, its own evidence.

    `VT_FP4_AUTOTUNE_CACHE_PATH` names a file that does NOT yet exist, so the
    process tunes all its plans rather than loading somebody else's draw, and
    publishes its own document at `CompletePersistentRuntime`. That is what
    makes the draws INDEPENDENT; reusing one cache path would collect one draw
    N times and report it as N.

    "Does not yet exist" is ENFORCED and not assumed. A draw that published its
    document and then died before writing `DONE` leaves that document behind; on
    the retry the fresh process would LOAD it, tune nothing, and be refused with
    `73` -- whose message says the draw is a copy of an earlier draw, which
    misdescribes a crash and sends the reader looking for a duplicated cache
    path. The unlink is skipped only on the resume path, where `DONE` says the
    draw finished and its document is the evidence #2752 would pin.

    The mirror runs HERE, per draw, rather than when the phase returns.
    """

    label = f"draw{index:02d}"
    home = evidence / "draws" / label
    home.mkdir(parents=True, exist_ok=True)
    done = home / "DONE"
    if done.is_file():
        return json.loads((home / "record.json").read_text(encoding="utf-8"))

    cache = home / "autotune_configs.json"
    cache.unlink(missing_ok=True)
    env = dict(os.environ)
    env["VT_GEMM_ALGO_LOG"] = "1"
    env["VT_FP4_PERSISTENT_CACHE"] = "1"
    env["VT_FP4_AUTOTUNE_CACHE_PATH"] = str(cache)
    env["VT_FP4_AUTOTUNE_CACHE_READONLY"] = "0"
    env.pop("VT_FP4_FLASHINFER_CACHE_PATH", None)
    # The arm is SET, never inherited. `Fp4FullTacticsEnabled` defaults ON, so
    # an unset variable is the 32-candidate pure-argmin arm -- and a draw whose
    # arm was decided by the operator's ambient environment is a draw nobody can
    # attribute. `VT_FP4_AUTOTUNE_VERBOSE` is what emits the per-key selection
    # line the diagnostic reads.
    env["VT_FP4_FULL_TACTICS"] = "1" if tactic_set == "full" else "0"
    env["VT_FP4_AUTOTUNE_VERBOSE"] = "1"

    command = draw_command(bench, model, cfg)
    if dry_run:
        stdout, stderr, rc = synthetic_draw_output(index, cfg, tactic_set=tactic_set)
        cache.write_text(
            json.dumps({"_metadata": {"dry_run": True}, "plans": []}) + "\n",
            encoding="utf-8",
        )
    else:
        proc = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode

    (home / "stdout.log").write_text(stdout, encoding="utf-8")
    (home / "stderr.log").write_text(stderr, encoding="utf-8")

    record: dict[str, Any] = {
        "label": label,
        "rc": rc,
        "command": command,
        "cache_path": str(cache),
        "cache_sha256": sha256_file(cache) if cache.is_file() else None,
        "cache_bytes": cache.stat().st_size if cache.is_file() else 0,
        "tactic_set": tactic_set,
        "algo": parse_algo_lines(stderr),
        "fp4": parse_fp4_lines(stderr),
        "autotune": parse_autotune_lines(stderr),
        "bench": parse_bench_report(stdout + stderr),
        "binary_sha256": (
            "dry-run" if dry_run else (sha256_file(bench) if bench.is_file() else None)
        ),
        "dry_run": bool(dry_run),
    }
    write_json(home / "record.json", record)
    if rc == 0:
        done.write_text("ok\n", encoding="utf-8")
    mirror_evidence(evidence, mirror)
    return record


def check_draw_preconditions(records: Sequence[Mapping[str, Any]]) -> tuple[int, list[str]]:
    """Refuse to report on evidence an instrument did not actually produce.

    Every failure here is a case where the run LOOKS finished. A draw phase with
    zero `[VT_GEMM_ALGO]` lines produces a perfectly consistent, perfectly empty
    stability report that reads as `STABLE` to anyone who does not count the
    keys; a "draw" that loaded somebody else's document tuned nothing and is a
    copy, not a draw. `.agents/verification.md`'s instrument rule is that an
    instrument must be shown to have RUN, and these are that rule made specific
    to the two instruments this harness uses.
    """

    problems: list[str] = []
    if not records:
        return EXIT_USAGE, ["no draws were recorded"]

    for record in records:
        label = record.get("label", "?")
        if record.get("rc") != 0:
            problems.append(f"{label}: exited {record.get('rc')}")
    if problems:
        return EXIT_DRAW_FAILED, problems

    for record in records:
        label = record["label"]
        algo = record.get("algo", {})
        keys = {k for k in algo if not k.startswith("_")}
        if not keys:
            seen = int(algo.get("_diagnostics", {}).get("count", "0"))
            problems.append(
                f"{label}: zero {ALGO_TAG} SELECTION lines ({seen} diagnostic "
                f"{ALGO_TAG} line(s) were seen). With none, VT_GEMM_ALGO_LOG did "
                "not reach MaybeLogGemmAlgo or no cuBLASLt GEMM ran; with some, "
                "the flag reached the candidate dumps and no selection was "
                "logged. A run that logged nothing cannot report that anything "
                "is stable"
            )
    if problems:
        return EXIT_ALGO_SILENT, problems

    for record in records:
        label = record["label"]
        bf16 = [
            k for k, fields in record["algo"].items()
            if not k.startswith("_") and fields.get("a", "") in _BF16_TAGS
        ]
        if not bf16:
            problems.append(
                f"{label}: {ALGO_TAG} lines exist but none has a bf16 input, so "
                "this run says nothing about KERNEL-GEMM-BF16 (#2750). Add a "
                "bf16 arm, or state that the workload does not reach the lane"
            )
    if problems:
        return EXIT_ALGO_NO_BF16, problems

    for record in records:
        label = record["label"]
        fp4 = record.get("fp4") or {}
        if not fp4.get("prepared") or not fp4.get("complete"):
            problems.append(
                f"{label}: missing {FP4_TAG} prepared/complete. The NVFP4 "
                "persistent runtime did not start, so no draw was taken"
            )
        elif not fp4.get("selected"):
            problems.append(f"{label}: {FP4_TAG} reported no selected plans")
    if problems:
        return EXIT_FP4_SILENT, problems

    for record in records:
        label = record["label"]
        complete = record["fp4"]["complete"]
        try:
            tuned = int(complete.get("tuned", "0"))
            loaded = int(complete.get("loaded", "0"))
        except ValueError:
            problems.append(f"{label}: unparsable tuned/loaded counts")
            continue
        if tuned <= 0:
            problems.append(
                f"{label}: tuned={tuned} loaded={loaded}. This process LOADED a "
                "plan map instead of drawing one, so it is a copy of an earlier "
                "draw and not an independent sample"
            )
    if problems:
        return EXIT_DRAW_NOT_INDEPENDENT, problems

    key_sets = {r["label"]: set(r["fp4"]["selected"]) for r in records}
    union = set.union(*key_sets.values())
    common = set.intersection(*key_sets.values())
    if union != common:
        return EXIT_KEYSET_DIFFERS, [
            "the draws tuned different plan-key sets "
            f"({len(union - common)} keys are not in every draw). The tuned set "
            "follows max_num_batched_tokens and the model shapes, so this means "
            "the draws did not run the same configuration and their tactic maps "
            "compare nothing"
        ]

    fingerprints = {r["label"]: r["fp4"]["prepared"].get("metadata") for r in records}
    if len({v for v in fingerprints.values()}) > 1:
        return EXIT_FINGERPRINT_DIFFERS, [
            "the draws report different PersistentCacheMetadataFingerprint "
            f"values: {json.dumps(fingerprints, sort_keys=True)}. Device, "
            "driver, CUTLASS, tactic ABI or build identity moved under the run"
        ]

    algo_sets = {
        r["label"]: {k for k in r["algo"] if not k.startswith("_")} for r in records
    }
    algo_union = set.union(*algo_sets.values())
    algo_common = set.intersection(*algo_sets.values())
    if algo_union != algo_common:
        return EXIT_ALGO_KEYSET_DIFFERS, [
            f"{len(algo_union - algo_common)} cuBLASLt selection keys are absent "
            "from at least one process, so the processes did not execute the "
            "same shapes and a cross-process comparison over the intersection "
            "would hide that"
        ]

    arms = {r.get("tactic_set") for r in records}
    if len(arms) > 1 or None in arms:
        return EXIT_ARM_MIXED, [
            "this evidence root holds more than one tactic-set arm "
            f"({sorted(str(a) for a in arms)}). `VT_FP4_FULL_TACTICS` default-ON "
            "selects 32 candidates by pure argmin and `=0` selects 4 candidates "
            "through a >1% stickiness damper, so the two arms choose by "
            "DIFFERENT RULES. A draw spread pooled across them names no rule and "
            "cannot be interpreted; run one root per arm"
        ]

    binaries = {r.get("binary_sha256") for r in records}
    if len(binaries) > 1 or None in binaries:
        return EXIT_BINARY_DIFFERS, [
            f"the draws did not all run one binary: {sorted(str(b) for b in binaries)}. "
            "Only the draw may vary between these processes"
        ]

    for record in records:
        if not record.get("cache_sha256") or not record.get("cache_bytes"):
            return EXIT_CACHE_MISSING, [
                f"{record['label']}: published no cache document, so the draw "
                "cannot be replayed frozen and #2752 has nothing to pin"
            ]

    return EXIT_OK, []


def check_frozen_leg(text: str, expected_plans: int) -> tuple[bool, str]:
    """A scoring leg must LOAD its draw, never tune a new one.

    This is the control that makes the scoring phase a comparison of draws
    rather than N more draws. `VT_FP4_AUTOTUNE_CACHE_READONLY=1` makes the
    runtime reject a frozen miss before any tuning, so a leg reporting
    `tuned>0` is evidence that the freeze did not take -- and its number is a
    measurement of a draw nobody recorded.
    """

    fp4 = parse_fp4_lines(text)
    complete = fp4.get("complete")
    if not complete:
        return False, f"no {FP4_TAG} complete line: the persistent runtime did not run"
    # A SECOND, INDEPENDENT WITNESS. `tuned=0` is the runtime's own count; this
    # is the tuner's own voice. `VT_FP4_AUTOTUNE_VERBOSE=1` prints one line per
    # tactic SELECTION, so a frozen replay must print none. The two can disagree
    # -- a selection that happened outside the counted population would show
    # here and nowhere else -- and a control with one witness is a control that
    # cannot be cross-checked.
    announced = parse_autotune_lines(text)["selections"]
    if announced:
        return False, (
            f"the tuner announced {len(announced)} tactic selection(s) on a leg "
            "that was supposed to replay a frozen map"
        )
    try:
        tuned = int(complete.get("tuned", "-1"))
        loaded = int(complete.get("loaded", "-1"))
    except ValueError:
        return False, "unparsable tuned/loaded counts on the complete line"
    if tuned != 0:
        return False, f"tuned={tuned}: the leg re-tuned instead of replaying the frozen draw"
    if loaded != expected_plans:
        return False, (
            f"loaded={loaded} but the draw carries {expected_plans} plans: the "
            "leg did not install the whole frozen map"
        )
    return True, f"frozen: tuned=0 loaded={loaded}"


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------
def read_draw_records(evidence: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((evidence / "draws").glob("draw*/record.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def read_score_ledger(path: pathlib.Path, metric: str) -> dict[str, list[float]]:
    """Fold the leg ledger into per-draw value lists.

    A VOID leg -- one that ran and produced no parsable number -- is DROPPED
    from the values but its absence changes `legs_per_draw`, which
    `speed_spread` reads. A harness that silently backfilled it would report a
    confident number over fewer legs than it claims.
    """

    per_draw: dict[str, list[float]] = {}
    if not path.is_file():
        return per_draw
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        arm = rec.get("arm")
        if arm is None or metric not in rec:
            continue
        per_draw.setdefault(arm, []).append(float(rec[metric]))
    return per_draw


def read_frozen_controls(evidence: pathlib.Path) -> list[dict[str, Any]]:
    """Every scoring leg's own frozen-control record, ONE FILE PER LEG.

    `check-frozen --record` writes `score/<arm>-<n>/frozen.json` as the leg
    finishes, pass or fail. A single shared control file would be a surface
    every leg has to write, which is the lock shape the protocol refuses; a
    glob over per-leg files has no writer in common.
    """

    out: list[dict[str, Any]] = []
    for path in sorted((evidence / "score").glob("*/frozen.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            out.append({"leg": path.parent.name, "frozen": False,
                        "why": f"unreadable control record: {error}"})
    return out


def frozen_control_state(
    controls: Sequence[Mapping[str, Any]], legs_by_arm: Mapping[str, int]
) -> dict[str, Any]:
    """Did EVERY leg that contributed a number pass the frozen control?

    Two ways to fail and both are the same defect: a leg that re-tuned measured
    a plan map nobody recorded, and a leg with no control record measured one
    nobody checked. Either way the scoring phase is N MORE DRAWS wearing the
    label of a replay, and its numbers are attributed to the arm they were asked
    about rather than to the map they actually ran.

    THE SECOND CHECK IS A JOIN AND NOT A COUNT. `check-frozen --record` writes
    `score/<arm>-<n>/frozen.json` and labels it `<arm>-<n>`, so a control record
    says which arm it answers for. Comparing totals instead -- N passing records
    against N ledger legs -- lets a record for one arm satisfy a leg of another,
    and an arm whose legs were never checked then reads as checked. The single
    writer this driver has cannot produce that state today, so this is a claim
    the function was making rather than a failure anyone has seen; it is cheaper
    to make the claim true than to weaken it, and the polarity is fail-closed
    either way.
    """

    passing = [c for c in controls if c.get("frozen") is True]
    failing = [c for c in controls if c.get("frozen") is not True]
    legs_in_ledger = sum(legs_by_arm.values())

    def arm_of(control: Mapping[str, Any]) -> str:
        return str(control.get("leg", "?")).rsplit("-", 1)[0]

    passing_by_arm: dict[str, int] = {}
    for control in passing:
        passing_by_arm[arm_of(control)] = passing_by_arm.get(arm_of(control), 0) + 1
    short = sorted(
        f"{arm} ({passing_by_arm.get(arm, 0)} of {count})"
        for arm, count in legs_by_arm.items()
        if passing_by_arm.get(arm, 0) < count
    )

    state = "PASS"
    reason = f"all {legs_in_ledger} scored leg(s) replayed a frozen map"
    if failing:
        state = "REFUSED"
        reason = (
            f"{len(failing)} scoring leg(s) failed the frozen control: "
            + "; ".join(f"{c.get('leg', '?')}: {c.get('why', '?')}" for c in failing)
        )
    elif short:
        state = "REFUSED"
        reason = (
            "these arm(s) contributed numbers from legs that carry no passing "
            "frozen-control record: " + "; ".join(short) + ". A leg whose "
            "control was never recorded measured a plan map nobody checked"
        )
    return {
        "state": state,
        "reason": reason,
        "legs_in_ledger": legs_in_ledger,
        "legs_by_arm": dict(sorted(legs_by_arm.items())),
        "legs_with_a_passing_control": len(passing),
        "passing_controls_by_arm": dict(sorted(passing_by_arm.items())),
        "records": [dict(c) for c in controls],
    }


def reduce_evidence(
    evidence: pathlib.Path,
    *,
    metric: str,
    ratification_bar: float,
    noise_ceiling: float = DEFAULT_NOISE_CEILING,
) -> tuple[int, dict[str, Any]]:
    records = read_draw_records(evidence)
    code, problems = check_draw_preconditions(records)
    report: dict[str, Any] = {
        "evidence": str(evidence),
        "dry_run": any(r.get("dry_run") for r in records),
        # The arm is a TOP-LEVEL field because a draw-spread number that does
        # not say which selection rule produced it is not interpretable, and a
        # report gets read out of the directory it was written in.
        "tactic_set": sorted({r.get("tactic_set") for r in records if r.get("tactic_set")}),
        "draws_recorded": [r.get("label") for r in records],
        "preconditions": {
            "exit_code": code,
            "problems": problems,
            "passed": code == EXIT_OK,
        },
    }
    if code != EXIT_OK:
        report["verdict"] = "REFUSED"
        report["note"] = (
            "No verdict is reported. Every condition above is one where the run "
            "looks finished and the instrument did not run, which reads as a "
            "result if it is not refused by name."
        )
        return code, report

    # The WHOLE parsed map, `_duplicates` and `_diagnostics` included.
    # `algo_stability` already excludes `_`-prefixed names from its key sets, and
    # stripping them here left `within_process_duplicate_lines` hard-wired to 0
    # in every report -- the one witness for `LogOncePerKey` misbehaving, unable
    # to fire.
    report["issue_2750_draw_processes"] = algo_stability(
        {r["label"]: r["algo"] for r in records}
    )
    report["issue_2751_identity"] = draw_identity(
        {r["label"]: r["fp4"]["selected"] for r in records}
    )
    report["issue_2751_selection_time"] = selection_time_spread(
        {r["label"]: (r.get("autotune") or {}).get("selections", {}) for r in records}
    )
    repeats = sum(
        (r.get("autotune") or {}).get("repeat_selections", 0) for r in records
    )
    if repeats:
        report["issue_2751_selection_time"]["lazy_miss_repeat_selections"] = repeats

    per_draw = read_score_ledger(evidence / "score" / "legs.jsonl", metric)
    if per_draw:
        # THE FROZEN CONTROL IS READ BEFORE ANY SPEED VERDICT IS WRITTEN.
        # Without it a report could carry EQUIVALENT and "ship draw00" over legs
        # that re-tuned, and the gate table's `78` would have no path to fire at
        # all -- which is what it had before, because the file this used to read
        # (`score/frozen-checks.json`) had no writer anywhere in the tree.
        controls = read_frozen_controls(evidence)
        legs_by_arm = {arm: len(values) for arm, values in per_draw.items()}
        report["frozen_leg_control"] = frozen_control_state(controls, legs_by_arm)
        if report["frozen_leg_control"]["state"] != "PASS":
            report["issue_2751_speed"] = {
                "verdict": "REFUSED",
                "reason": (
                    "the scoring legs did not all replay a frozen map: "
                    + report["frozen_leg_control"]["reason"]
                ),
            }
            report["issue_2752"] = {
                "ship": None,
                "reason": (
                    "no draw is pinned from legs whose frozen control did not "
                    "pass; the numbers describe a plan map nobody recorded"
                ),
            }
            return EXIT_LEG_NOT_FROZEN, report
        report["issue_2751_speed"] = speed_spread(
            per_draw, ratification_bar=ratification_bar, noise_ceiling=noise_ceiling
        )
        report["issue_2752"] = select_shipping_draw(
            report["issue_2751_speed"], [r["label"] for r in records]
        )
    else:
        report["issue_2751_speed"] = {
            "verdict": "NOT RUN",
            "reason": "no scoring ledger at score/legs.jsonl",
        }
        report["issue_2752"] = {
            "ship": None,
            "reason": "the speed half has not run, so #2752 stays blocked",
        }

    clock = evidence / "score" / "clock-windows.json"
    if clock.is_file():
        report["clock_windows"] = json.loads(clock.read_text(encoding="utf-8"))
    else:
        report["clock_windows"] = {
            "state": "ABSENT",
            "note": (
                "No clock window was recorded, so no speed figure in this report "
                "carries clock attribution (.agents/benchmarking.md). Inside an "
                "rc lease the SM clock can be SAMPLED and not pinned (#1354), so "
                "a pairing may be refused on within-run spread with no lever to "
                "fix it -- that refusal is a result, not a harness bug."
            ),
        }
    return EXIT_OK, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    draw = sub.add_parser("draw", help="take N independent draws in N fresh processes")
    draw.add_argument("--evidence", required=True, type=pathlib.Path)
    draw.add_argument("--bench", required=True, type=pathlib.Path,
                      help="path to the built vllm-bench binary")
    draw.add_argument("--model", required=True,
                      help="checkpoint directory; this harness NEVER defaults one")
    draw.add_argument("--draws", type=int, default=8)
    draw.add_argument("--num-prompts", type=int, default=32)
    draw.add_argument("--input-len", type=int, default=512)
    draw.add_argument("--output-len", type=int, default=64)
    draw.add_argument("--concurrency", type=int, default=2)
    draw.add_argument("--seed", type=int, default=0)
    draw.add_argument("--max-num-batched-tokens", type=int, default=8192)
    draw.add_argument("--tactic-set", choices=("full", "w1"), default="full",
                      help="'full' sets VT_FP4_FULL_TACTICS=1 (32 candidates, pure "
                           "argmin; the shipped default) and 'w1' sets it to 0 "
                           "(4 candidates behind the >1%% stickiness damper). The two "
                           "select by different rules, so ONE ARM PER EVIDENCE ROOT")
    draw.add_argument("--mirror", type=pathlib.Path,
                      help="copy the evidence root here AFTER EACH DRAW. The "
                           "local root is a real filesystem the engine can "
                           "publish its cache document onto; this is what "
                           "survives the box going down mid-phase")
    draw.add_argument("--dry-run", action="store_true",
                      help="walk the resume/record path with no subprocess and no device")

    frozen = sub.add_parser(
        "check-frozen", help="assert one scoring leg replayed a frozen map"
    )
    frozen.add_argument("--log", required=True, type=pathlib.Path)
    frozen.add_argument("--expected-plans", required=True, type=int)
    frozen.add_argument("--record", type=pathlib.Path,
                        help="write this leg's own control record here, PASS OR "
                             "FAIL. `reduce` globs score/*/frozen.json and "
                             "refuses (78) when a leg that contributed a number "
                             "carries no passing record")
    frozen.add_argument("--leg", default="?",
                        help="the leg label recorded in --record")

    reduce_p = sub.add_parser("reduce", help="judge a complete evidence directory")
    reduce_p.add_argument("--evidence", required=True, type=pathlib.Path)
    reduce_p.add_argument("--metric", default="total_token_throughput")
    reduce_p.add_argument("--ratification-bar", type=float,
                          default=DEFAULT_RATIFICATION_BAR)
    reduce_p.add_argument("--noise-ceiling", type=float,
                          default=DEFAULT_NOISE_CEILING,
                          help="the widest WITHIN-draw repeat spread a run may "
                               "carry and still be judged. Its own parameter, "
                               "deliberately not the ratification bar: they "
                               "share a value and mean different things")
    reduce_p.add_argument("--out", type=pathlib.Path)

    args = parser.parse_args(argv)

    if args.command == "draw":
        cfg = {
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        }
        write_json(
            args.evidence / "draw-config.json",
            dict(cfg, tactic_set=args.tactic_set),
        )
        records: list[dict[str, Any]] = []
        for index in range(args.draws):
            record = run_draw(
                index, args.evidence, args.bench, args.model, cfg,
                tactic_set=args.tactic_set, dry_run=args.dry_run,
                mirror=args.mirror,
            )
            records.append(record)
            keys = len([k for k in record.get("algo", {}) if not k.startswith("_")])
            plans = len((record.get("fp4") or {}).get("selected", {}))
            print(
                f"  {record['label']}: rc={record['rc']} algo_keys={keys} "
                f"plans={plans} cache={record.get('cache_sha256')}"
            )
        code, problems = check_draw_preconditions(records)
        for problem in problems:
            print(f"gemm-tactic-draw-survey: {problem}", file=sys.stderr)
        write_json(
            args.evidence / "draw-preconditions.json",
            {"exit_code": code, "problems": problems},
        )
        mirror_evidence(args.evidence, args.mirror)
        return code

    if args.command == "check-frozen":
        text = args.log.read_text(encoding="utf-8", errors="replace")
        ok, why = check_frozen_leg(text, args.expected_plans)
        print(f"frozen-leg: {'OK' if ok else 'REFUSED'}: {why}")
        if args.record:
            # WRITTEN ON BOTH OUTCOMES. A control that records itself only on
            # success makes a refused leg indistinguishable from a leg that
            # never ran, and `reduce` counts records against ledger rows.
            write_json(args.record, {
                "leg": args.leg,
                "frozen": ok,
                "why": why,
                "expected_plans": args.expected_plans,
                "log": str(args.log),
            })
        return EXIT_OK if ok else EXIT_LEG_NOT_FROZEN

    code, report = reduce_evidence(
        args.evidence, metric=args.metric, ratification_bar=args.ratification_bar,
        noise_ceiling=args.noise_ceiling,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
