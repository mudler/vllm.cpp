#!/usr/bin/env python3
"""Drive a repeated c=8 leg sequence through the ledger, and survive a reboot.

WHAT THIS IS FOR
----------------
#2154's severity metric — the fraction of draft blocks that come back all-zero —
ranges 0.0% to 87.7% across runs of ONE unchanged binary. Nothing can be
concluded from it at n=1 per arm, and the instance-versus-pass question #2152
names as blocking needs several legs per arm across several server instances.
That is an hour of legs on a host whose MTBF is shorter than an hour (#545,
four crashes in one session, three of which killed this exact experiment).

`tools/bench/resumable_legs.py` holds the rules. This is its caller: it turns a
plan into subprocess invocations, appends each leg the instant it finishes, and
on restart replays the ledger and runs only what is owed.

WHY IT IS A SEPARATE FILE
-------------------------
The ledger has no I/O and no subprocesses, so every rule in it is unit tested on
the CPU. This file has both, so it carries `--dry-run`, which walks the whole
plan/resume/fold path with a stub command and no device. That is what
`tests/tools/test_c8_leg_runner.py` drives, and it is why the runner can be
gated without a lease.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not know what a leg MEANS. The command it runs and the metric it reads
are arguments, because the c=8 ladder must eventually run through
`scripts/dgx-online-serving.sh` (#2152), and that driver cannot express a
speculative workload yet. Binding this runner to today's ad-hoc script would
make retiring that script harder, not easier.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

from tools.bench.resumable_legs import (
    append_leg,
    fold,
    plan,
    read_ledger,
    remaining,
    terminal_check,
)

BOOT_ID_PATH = pathlib.Path("/proc/sys/kernel/random/boot_id")


def read_boot_id(path: pathlib.Path = BOOT_ID_PATH) -> str:
    """The running kernel's boot id, or empty when it cannot be read.

    Empty is NOT silently tolerated: `append_leg` refuses a record without one,
    so an unreadable boot id stops the run rather than producing legs nobody can
    attribute afterwards.
    """

    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def parse_metric(text: str, pattern: str) -> float | None:
    """Pull one number out of a leg's stdout with the caller's regex.

    Returns None rather than raising, and the caller records the leg as VOID.
    A leg that ran and produced no parsable number is evidence of a broken
    instrument; dropping it silently would leave the fold looking healthy on
    fewer legs than it claims.
    """

    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (IndexError, ValueError):
        return None


def run_one(command: list[str], arm: str, *, dry_run: bool) -> tuple[str, int]:
    if dry_run:
        return (f"arm={arm} metric=1.0\n", 0)
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    return (proc.stdout + proc.stderr, proc.returncode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True, type=pathlib.Path)
    ap.add_argument("--arm", action="append", required=True,
                    help="repeatable; the arms to interleave")
    ap.add_argument("--legs-per-arm", type=int, required=True)
    ap.add_argument("--metric", required=True, help="name recorded on each leg")
    ap.add_argument("--metric-regex", required=True,
                    help="regex with ONE capture group, applied to the leg's output")
    ap.add_argument("--command", required=True,
                    help="shell command; {arm} is substituted")
    ap.add_argument("--terminal-tolerance-pct", type=float, default=6.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="walk the plan with a stub command and no device")
    args = ap.parse_args(argv)

    boot = read_boot_id()
    if not boot and not args.dry_run:
        print("c8-leg-runner: cannot read the boot id, so no leg could be "
              "attributed afterwards (#545). Refusing.", file=sys.stderr)
        return 2
    boot = boot or "dry-run"

    order = plan(args.arm, args.legs_per_arm)
    done = read_ledger(args.ledger)
    owed = remaining(order, done)
    print(f"c8-leg-runner: {len(order)} legs planned, {len(done)} done, {len(owed)} owed")

    for arm in owed:
        cmd = args.command.replace("{arm}", arm)
        out, rc = run_one(["bash", "-lc", cmd], arm, dry_run=args.dry_run)
        value = parse_metric(out, args.metric_regex)
        rec: dict[str, object] = {"arm": arm, "boot_id": read_boot_id() or boot, "rc": rc}
        if value is None:
            rec["void"] = "no parsable metric in the leg's output"
        else:
            rec[args.metric] = value
        append_leg(args.ledger, rec)
        print(f"  {arm}: {args.metric}={value if value is not None else 'VOID'} rc={rc}")

    done = read_ledger(args.ledger)
    summary = fold(done, args.metric)
    control = terminal_check(done, args.metric,
                             tolerance_pct=args.terminal_tolerance_pct)
    print(json.dumps({"fold": summary, "terminal_control": control},
                     indent=2, sort_keys=True))
    if not summary["admissible"]:
        print("c8-leg-runner: NOT ADMISSIBLE — see fold.reasons", file=sys.stderr)
        return 1
    if control.get("checked") and not control.get("ok"):
        print("c8-leg-runner: the terminal control DRIFTED; no comparison in this "
              "run is admissible", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
