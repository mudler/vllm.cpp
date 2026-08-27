#!/usr/bin/env python3
"""Refold tt_clock_state windows to their busy-only slices and re-judge.

The Blackhole P150 AICLK governor is two-state: 800 MHz idle, pegged at the
claimed max under load. A raw window therefore mixes the pre-open idle head
with pegged compute and refuses on within-run spread — correctly, because
the NVIDIA rule describes quasi-continuous clocks. The honest attribution
for such a platform is the BUSY SLICE: samples whose busy flag was recorded
live at sample time (leg pid holding /dev/tenstorrent fds), a criterion
independent of the outcome clock values. This tool rebuilds those records
from a summary's _t/_aiclks/_busy columns and runs the same judge.

Usage: tt_refold_busy.py SUMMARY.json [SUMMARY.json ...]
Writes one JSON verdict to stdout; exit 0 iff no refusal reasons.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tt_clock_state as ttc  # noqa: E402


def refold(summary: dict) -> dict:
    samples = [
        {"t": t, "aiclk": a, "busy": True}
        for t, a, b in zip(
            summary["_t"], summary["_aiclks"], summary["_busy"]
        )
        if b
    ]
    return ttc.fold(
        samples,
        summary.get("claimed_max_aiclk_mhz"),
        summary.get("claimed_max_provenance"),
        allow_cross_boot=bool(summary.get("_allow_cross_boot")),
    )


def main(argv: list[str]) -> int:
    records = []
    for path in argv:
        s = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        records.append(refold(s))
    result = ttc.judge(records)
    print(json.dumps(result, indent=1))
    return 0 if not result["reasons"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
