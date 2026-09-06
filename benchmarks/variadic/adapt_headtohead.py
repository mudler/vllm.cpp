#!/usr/bin/env python3
"""Recompute the concurrency-1 head-to-head legs through `report.py`.

The predecessor run
(`docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/`) wrote one record
per request with the SAME field definitions this harness uses: `ttft` is the
wall time to the first content-carrying chunk, `itls` are gaps between streamed
chunks, `latency` is client-side, and the token counts come from each server's
`usage`. Its client then reduced them to means and threw the distribution away.

The records did not go anywhere. This converts them into the shape `report.py`
reads, so the published legs get percentiles, a cold-start view and a
tokens-per-chunk figure without touching a GPU.

WHAT IT CANNOT RECOVER. The old client did not record the character length of
the first chunk or of the whole completion, so `first_chunk_tokens` and the
chunk-corrected time to first token are NOT estimable from these files. The
report prints them as absent rather than guessing, which is the point: the raw
tokens-per-chunk figure is measurable from `n_chunks` and `usage`, and the
correction is not.

Request `i = 0` is tagged `warmup`, so the report's warm-only and with-warmup
views separate the cold request from the rest. That is the predecessor's own
stated rule ("Run 1 of every leg is cold and discarded"), which its head-to-head
legs did not apply.

Every prompt in that run is a HumanEval prompt, so every record is band `M`.
"""
import argparse
import glob
import json
import os
import sys


def adapt(path, out_dir, warmup):
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    label = old["summary"]["leg"]                      # e.g. OURS-A
    arm, _, tag = label.partition("-")
    rnd = {"A": 1, "B": 2}.get(tag, 1)

    records = []
    for rec in old["records"]:
        new = {
            "i": rec["i"],
            "id": f"M-{rec['i']:04d}",
            "band": "M",
            "phase": "warmup" if rec["i"] < warmup else "measured",
            "ok": bool(rec.get("ok")),
            "ttft": rec.get("ttft"),
            "latency": rec.get("latency"),
            "n_chunks": rec.get("n_chunks", 0),
            "itls": rec.get("itls", []),
            "usage": rec.get("usage"),
            "accepted_draft_tokens": rec.get("accepted_draft_tokens"),
            # ABSENT, not zero: the old client did not record them.
            "first_chunk_chars": None,
            "total_chars": None,
        }
        if not new["ok"]:
            new["err"] = rec.get("err", "unknown")
        records.append(new)

    ok = [r for r in records if r["ok"]]
    warm_ok = [r for r in ok if r["phase"] == "warmup"]
    meas_ok = [r for r in ok if r["phase"] == "measured"]
    # The old leg has ONE wall clock covering both phases. Split it by summing
    # each phase's latencies, which at concurrency 1 is the whole leg to within
    # the client's own per-request overhead. Stated here rather than hidden: at
    # c = 1 the requests are sequential, so this is exact up to that overhead.
    warm_s = sum(r["latency"] for r in warm_ok)
    total_s = old["summary"]["wall_s"]
    out = {
        "summary": {
            "leg": f"{arm}-r{rnd}-c1",
            "arm": arm,
            "round": rnd,
            "concurrency": 1,
            "warmup_requests": warmup,
            "measured_requests": len(records) - warmup,
            "warm_wall_s": warm_s,
            "measured_wall_s": max(0.0, total_s - warm_s),
            "adapted_from": os.path.basename(path),
            "adapter_note": (
                "recomputed from the 2026-09-03 head-to-head records; the "
                "chunk-character fields were never recorded, so the "
                "chunk-corrected TTFT is absent rather than estimated"),
        },
        "config": old.get("config", {}),
        # The old run had no /metrics scrape on either side.
        "metrics_before": {"available": False, "error": "not recorded"},
        "metrics_after": {"available": False, "error": "not recorded"},
        "records": records,
    }
    dst = os.path.join(out_dir, f"{arm}-r{rnd}-c1.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return dst, len(ok), len(meas_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="the head-to-head out/ directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--warmup", type=int, default=1,
                    help="requests to tag as warmup; the predecessor's own "
                         "stated rule is 1")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    paths = [p for p in sorted(glob.glob(os.path.join(args.in_dir, "*.json")))
             if os.path.basename(p).startswith(("OURS-", "THEIRS-"))]
    if not paths:
        print(f"no leg files under {args.in_dir}", file=sys.stderr)
        return 1
    for p in paths:
        dst, n_ok, n_meas = adapt(p, args.out_dir, args.warmup)
        print(f"{os.path.basename(p)} -> {os.path.basename(dst)} "
              f"({n_ok} ok, {n_meas} measured)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
