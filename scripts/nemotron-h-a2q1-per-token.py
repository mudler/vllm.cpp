#!/usr/bin/env python3
"""Derive the per-output-token time of one A2-Q1 A3 gate run, WITH its terms.

`examples/nemotron_h_gen` reports neither a rate nor a duration: it prints the
engine load time and a `TOKEN MATCH: m/n` line and nothing else. So the per-token
number this row is measured on has to be derived, and a derived number that hides
its terms is how a rate over an unknown denominator comes to be quoted as if it
were measured.

THE WINDOW IS THE DECODE, AND THE CALLER BRACKETS IT. `t0`/`t1` are taken AFTER
the driver prints `engine loaded in Ns`, so the 20.1 GiB load is already outside
them and nothing is subtracted here. An earlier version bracketed the whole
process and subtracted the load, which put a multi-minute GPU-idle phase inside
the same window as the decode -- the same defect as summing prefill and decode
into one profile. The load is still read out of the log and printed, because it
is context for the number, not a term in it.

This prints the window, the load it excludes and the token count it divides by on
separate lines, and it REFUSES rather than printing 0 when a term is missing or
the window is not positive.

THE vLLM RATIO IS QUOTED ONLY ON THE SILICON IT WAS MEASURED ON. 0.014369 s is a
GB10 figure. Printing it beside a Thor per-token number invites a comparison
across two different pieces of silicon, and the first Thor run did exactly that
-- it printed `ratio 54.7x` for a number that was never measured against vLLM on
that box. This is the same defect the busy-fraction reporter carried, and fixing
one surface while leaving its twin is how a wrong comparison survives.

    nemotron-h-a2q1-per-token.py <a3.log> <label> <t0_epoch> <t1_epoch> [arch]
"""

from __future__ import annotations

import re
import sys

# vLLM 0.26.0.dev0 on the same GB10 workload, the denominator this row's gap is
# quoted against. Carried as a constant so the ratio below cannot be computed
# against a number nobody can find the origin of.
VLLM_PER_TOKEN_S = 0.014369


# The arch the vLLM per-token reference was measured on. Any other arch gets the
# number withheld rather than a ratio nobody can defend.
VLLM_REFERENCE_ARCH = "121a"


def main(argv: list[str]) -> int:
    if len(argv) not in (5, 6):
        print(__doc__.strip())
        return 2
    log, label, t0, t1 = argv[1], argv[2], float(argv[3]), float(argv[4])
    arch = argv[5] if len(argv) == 6 else VLLM_REFERENCE_ARCH
    with open(log, errors="replace") as fh:
        text = fh.read()
    decode_s = t1 - t0
    print(f"{label}: decode window {decode_s:.3f} s (the engine load is OUTSIDE it)")

    load = re.search(r"engine loaded in ([0-9.]+)s", text)
    match = re.search(r"TOKEN MATCH: (\d+)/(\d+) over (\d+) prompt", text)
    if load is not None:
        print(f"{label}: engine load {float(load.group(1)):.1f} s, excluded")
    if match is None:
        print(f"{label}: NO TOKEN MATCH LINE -- per-token time NOT derivable, not 0")
        return 0
    matched, compared = int(match.group(1)), int(match.group(2))
    print(f"{label}: tokens compared {compared} ; matched {matched}")
    if compared <= 0:
        print(f"{label}: ZERO tokens compared -- a rate here would divide by nothing")
        return 0
    if decode_s <= 0.0:
        # A non-positive window means the caller's brackets did not span the
        # decode -- the driver exited before the sampler started, or the two
        # timestamps came from different runs. The quotient would be a negative
        # or infinite rate that still formats like a measurement.
        print(f"{label}: decode window {decode_s:.3f} s is not positive -- the "
              f"brackets did not span a decode, so NO per-token time is reported")
        return 0
    per_token = decode_s / compared
    if arch == VLLM_REFERENCE_ARCH:
        note = (f"vLLM {VLLM_PER_TOKEN_S} s; ratio {per_token / VLLM_PER_TOKEN_S:.1f}x")
    else:
        note = (f"NO vLLM ratio for arch {arch}: the {VLLM_PER_TOKEN_S} s reference is "
                f"GB10's, so a ratio against it would compare two different pieces of "
                f"silicon; use the ON/OFF A/B on this box")
    print(f"{label}: per output token {per_token:.6f} s ({note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
