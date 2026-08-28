#!/usr/bin/env python3
"""A leg ledger that survives the host rebooting under it.

WHY THIS EXISTS
---------------
`dgx.casa` went down FOUR times on 2026-08-28 (#545), and three of those crashes
killed the same experiment: the instance-versus-pass variance test that #2152's
spec names as the blocking question before any further c=8 A/B. Each attempt
restarted from zero because the runner held its results in memory and printed
them at the end. A sequence long enough to answer the question is longer than
this host's MTBF, so it can only finish if it is RESUMABLE.

The second failure this addresses is quieter. Every c=8 reading in this campaign
was taken by an ad-hoc script living only on the box, so nothing recorded which
BOOT a number came from, and two comparisons were built across a reboot before
anyone noticed. `tools/bench/gpu_clock_state.py` has refused cross-boot
comparison since #543; this module refuses to FOLD across one, which is the same
rule applied to a sequence rather than to a pair.

WHAT IT IS, AND IS NOT
----------------------
A ledger, not a runner. It owns the append-only record, the resume decision and
the fold; the caller owns the subprocess. That split is deliberate: the caller's
half needs a GPU and the ledger's half does not, so every rule below is unit
tested on the CPU, which is the polarity `gpu_clock_state.py` chose for the same
reason.

THE RULES
---------
1.  A leg is appended the moment it completes. A crash loses at most the leg in
    flight.
2.  Resume replays the ledger and returns only the legs still owed, in order.
3.  **Folding refuses across a boot change.** A ledger spanning two boots is not
    a population; it is two populations. `fold` names the boots rather than
    averaging them, because the alternative is what produced the retracted #543
    findings and what nearly landed two wrong conclusions here.
4.  A terminal control is a plan entry, not an afterthought. `plan` places the
    opening arm again at the end, and `terminal_check` compares them, so a run
    that drifted says so instead of returning a confident number.
"""

from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any, Iterable, Mapping, Sequence


def plan(arms: Sequence[str], legs_per_arm: int, *, terminal_control: bool = True) -> list[str]:
    """The leg order: arms INTERLEAVED, with the opening arm repeated last.

    Interleaved because a block of one arm followed by a block of the other
    measures the hour as much as the change -- two of four sequences run for
    #2154 self-invalidated on exactly that. The terminal control is what makes a
    drifting box declare itself.
    """

    if not arms:
        return []
    if legs_per_arm < 1:
        raise ValueError("legs_per_arm must be >= 1")
    order: list[str] = []
    for i in range(legs_per_arm):
        for arm in arms:
            order.append(arm)
        del i
    if terminal_control and len(arms) > 1:
        order.append(arms[0])
    return order


def read_ledger(path: pathlib.Path) -> list[dict[str, Any]]:
    """Every completed leg, in completion order. A truncated tail is DROPPED.

    A crash mid-write leaves a partial JSON line. Dropping it is right: the leg
    it describes did not finish, so it has no result, and the alternative is a
    parse error that makes the whole ledger unreadable.
    """

    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn tail from a crash mid-append
        if isinstance(rec, dict) and "arm" in rec:
            out.append(rec)
    return out


def append_leg(path: pathlib.Path, rec: Mapping[str, Any]) -> None:
    """Append one completed leg and flush, so a crash cannot lose it."""

    if "arm" not in rec:
        raise ValueError("a leg record must name its arm")
    if "boot_id" not in rec:
        raise ValueError("a leg record must name the boot it ran on (#545)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()


def remaining(order: Sequence[str], done: Sequence[Mapping[str, Any]]) -> list[str]:
    """The legs still owed. Resume is a replay of the ledger, not a guess."""

    return list(order[len(done):])


def boots(done: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: list[str] = []
    for rec in done:
        b = str(rec.get("boot_id", ""))
        if b and b not in seen:
            seen.append(b)
    return seen


def fold(done: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    """Median per arm, or a REFUSAL naming why.

    Refuses on a boot change rather than averaging across one. It also refuses
    an arm with a single leg, because a lone leg has no spread and this rung has
    already produced 0.0% and 87.7% from one unchanged binary.
    """

    reasons: list[str] = []
    bs = boots(done)
    if len(bs) > 1:
        reasons.append(
            "boot: the ledger spans "
            + str(len(bs))
            + " boots ("
            + ", ".join(b[:8] for b in bs)
            + "). Legs from different boots are two populations, not one; a "
            "byte-identical kernel moved 9.65% across a boot with nothing "
            "throttling (#543)"
        )
    by_arm: dict[str, list[float]] = {}
    for rec in done:
        if metric not in rec:
            continue
        try:
            by_arm.setdefault(str(rec["arm"]), []).append(float(rec[metric]))
        except (TypeError, ValueError):
            continue
    if not by_arm:
        reasons.append(f"metric: no leg carried {metric!r}, so there is nothing to fold")
    for arm, vals in sorted(by_arm.items()):
        if len(vals) < 2:
            reasons.append(
                f"legs: arm {arm!r} has {len(vals)} leg(s). One leg has no spread, and this "
                "rung has produced 0.0% and 87.7% from one unchanged binary (#2154)"
            )
    summary = {
        arm: {
            "median": statistics.median(vals),
            "n": len(vals),
            "min": min(vals),
            "max": max(vals),
            "spread_pct": (100.0 * (max(vals) - min(vals)) / min(vals)) if min(vals) > 0 else None,
        }
        for arm, vals in sorted(by_arm.items())
    }
    return {"reasons": reasons, "arms": summary, "boots": bs, "admissible": not reasons}


def terminal_check(
    done: Sequence[Mapping[str, Any]], metric: str, *, tolerance_pct: float
) -> dict[str, Any]:
    """Did the opening arm still read the same at the END?

    The check `plan`'s trailing entry exists for. If the first and last legs of
    that arm disagree by more than `tolerance_pct`, the run measured the hour and
    no comparison inside it is admissible -- which is how two of this campaign's
    four sequences correctly invalidated themselves.
    """

    legs = [r for r in done if metric in r]
    if len(legs) < 2:
        return {"checked": False, "reason": "fewer than two legs carry the metric"}
    opening_arm = str(legs[0]["arm"])
    same = [r for r in legs if str(r["arm"]) == opening_arm]
    if len(same) < 2:
        return {"checked": False, "reason": f"arm {opening_arm!r} ran once; no terminal control"}
    first, last = float(same[0][metric]), float(same[-1][metric])
    if first == 0:
        return {"checked": False, "reason": "opening leg read zero"}
    drift = 100.0 * abs(last - first) / abs(first)
    return {
        "checked": True,
        "arm": opening_arm,
        "first": first,
        "last": last,
        "drift_pct": drift,
        "ok": drift <= tolerance_pct,
    }
