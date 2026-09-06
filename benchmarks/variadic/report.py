#!/usr/bin/env python3
"""Turn the leg records into the published tables.

Every number this prints is computed from a file `client.py` wrote, so the
statistics can be recomputed, corrected or extended without touching the GPU.

PERCENTILES. Linear interpolation between order statistics, which is what
`numpy.percentile` does by default and what vLLM's own serving benchmark uses
(vllm/benchmarks/serve.py:739). Implemented here rather than imported so the
report runs on a container with no numpy.

METRIC DEFINITIONS, mirrored from vllm/benchmarks/serve.py:321 BenchmarkMetrics:

    ttft  time to the first content-carrying chunk        serve.py:615
    tpot  (latency - ttft) / (output_tokens - 1)          serve.py:610
    itl   gap between consecutive streamed chunks         serve.py:614
    e2el  client-side request latency                     serve.py:616

`tpot` is the primary inter-token axis and `itl` is secondary, because `tpot`
is normalised by the token count the server reported and is therefore immune to
the two engines streaming at different granularities, while `itl` is not.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

PCTS = (50, 90, 95, 99)


def percentile(values, p):
    """numpy.percentile's default: linear interpolation between order stats."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * p / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[int(idx)]
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)


def spread(values):
    """Relative spread, max over min minus one. The word the tables use."""
    vs = [v for v in values if v is not None and v > 0]
    if len(vs) < 2:
        return None
    return max(vs) / min(vs) - 1.0


def load_legs(paths):
    legs = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                legs.append((path, json.load(f)))
        except Exception as exc:                                # noqa: BLE001
            print(f"UNREADABLE {path}: {exc}", file=sys.stderr)
    return legs


def ok_records(leg, phase):
    return [r for r in leg["records"]
            if r.get("ok") and (phase is None or r["phase"] == phase)]


def axes(recs):
    """The four vLLM axes plus the chunking observables, in milliseconds."""
    ttft, tpot, itl, e2el = [], [], [], []
    ttft_corrected = []
    over_corrected = 0
    toks_per_chunk, first_chunk_tokens, chars_per_token = [], [], []
    out_tokens, in_tokens = [], []
    counted_by = set()
    dropped_no_usage = 0
    for r in recs:
        usage = r.get("usage") or {}
        n_out = usage.get("completion_tokens")
        if not n_out:
            counted_by.add("chunks")
            dropped_no_usage += 1
            continue
        counted_by.add("usage")
        n_out = int(n_out)
        out_tokens.append(n_out)
        in_tokens.append(int(usage.get("prompt_tokens") or 0))
        ttft.append(r["ttft"] * 1000.0)
        e2el.append(r["latency"] * 1000.0)
        itl.extend(x * 1000.0 for x in r.get("itls", []))
        this_tpot = None
        if n_out > 1:
            this_tpot = (r["latency"] - r["ttft"]) / (n_out - 1)
            tpot.append(this_tpot * 1000.0)
        if r.get("n_chunks"):
            toks_per_chunk.append(n_out / r["n_chunks"])
        # ESTIMATE, and labelled as one everywhere it is printed. Streaming
        # carries no per-chunk token count, so the first chunk's token count is
        # inferred from this request's own characters-per-token.
        if r.get("total_chars") and n_out:
            cpt = r["total_chars"] / n_out
            chars_per_token.append(cpt)
            fct = r.get("first_chunk_chars", 0) / cpt if cpt > 0 else 0.0
            first_chunk_tokens.append(fct)
            if this_tpot is not None:
                # REFUSE an over-correction, never floor it. `fct` is an
                # estimate, so `(fct - 1) * tpot` can exceed the measured ttft
                # on a coarse chunker -- at 8 tokens per chunk and a 30 ms tpot
                # that is 210 ms against a 200 ms ttft. Flooring publishes 0.0
                # as if it were a measurement; dropping the record makes the
                # correction absent for it, which is what it is.
                corrected = r["ttft"] - max(0.0, fct - 1.0) * this_tpot
                if corrected > 0.0:
                    ttft_corrected.append(corrected * 1000.0)
                else:
                    over_corrected += 1
    return {
        "n": len(out_tokens), "ttft": ttft, "tpot": tpot, "itl": itl,
        "e2el": e2el, "ttft_corrected": ttft_corrected,
        "out_tokens": out_tokens, "in_tokens": in_tokens,
        "toks_per_chunk": toks_per_chunk,
        "first_chunk_tokens": first_chunk_tokens,
        "chars_per_token": chars_per_token,
        "counted_by": "+".join(sorted(counted_by)) or "none",
        "dropped_no_usage": dropped_no_usage,
        "over_corrected": over_corrected,
        # G-USAGE. A leg is publishable only when EVERY successful request
        # carried a server-side token count. One that did not is refused, not
        # footnoted: its throughput divides the tokens it could count by a wall
        # clock that also contains the requests it could not, which understates
        # the rate with no signal beyond a string in the last column.
        "publishable": dropped_no_usage == 0 and bool(out_tokens),
    }


def cell_key(leg):
    s = leg["summary"]
    return (s.get("arm") or "?", int(s["concurrency"]))


def acceptance(leg):
    """Ours from the /metrics delta, theirs from the usage field. Never zero
    for an engine that exports nothing: absent stays absent."""
    mb = leg.get("metrics_before") or {}
    ma = leg.get("metrics_after") or {}
    # A scrape that FAILED is not a scrape that returned nothing. Ignoring the
    # flag makes a dropped request on a crashy box read as "this engine exports
    # no acceptance", which is a claim about the engine.
    if mb.get("available") and ma.get("available"):
        before = mb.get("counters") or {}
        after = ma.get("counters") or {}
        draft = accepted = None
        for name, val in after.items():
            base = name.split("{")[0]
            # SUM every label series, never keep the last one: one series per
            # model is what this engine emits today and that is not a contract.
            if base.endswith("spec_decode_num_draft_tokens_total"):
                draft = (draft or 0.0) + val - before.get(name, 0.0)
            elif base.endswith("spec_decode_num_accepted_tokens_total"):
                accepted = (accepted or 0.0) + val - before.get(name, 0.0)
        if draft is not None and accepted is not None:
            if draft < 0 or accepted < 0:
                # A counter cannot fall. The server restarted inside the leg, so
                # the delta is not a measurement of it.
                return {"source": "counter reset inside the leg"}
            if draft > 0:
                # ACCEPTED PER OUTPUT TOKEN as well as accepted/proposed. The
                # other engine here exports only the first of those, and two
                # acceptance numbers on different denominators printed in one
                # column is an invitation to divide one by the other.
                out = sum(int((r.get("usage") or {}).get("completion_tokens") or 0)
                          for r in ok_records(leg, "measured"))
                return {"source": "/metrics", "draft": draft,
                        "accepted": accepted, "rate": accepted / draft,
                        "per_output_token": (accepted / out) if out else None}
    acc = [r["accepted_draft_tokens"] for r in ok_records(leg, "measured")
           if r.get("accepted_draft_tokens") is not None]
    if acc:
        out = sum(int((r.get("usage") or {}).get("completion_tokens") or 0)
                  for r in ok_records(leg, "measured"))
        return {"source": "usage", "accepted": sum(acc),
                "per_output_token": (sum(acc) / out) if out else None}
    return {"source": "absent"}


def fmt(v, nd=1):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) \
        else f"{v:.{nd}f}"


def print_histogram(legs, args):
    print("\n## Realised prompt-length histogram, read back from each server's "
          "own usage.prompt_tokens\n")
    edges = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 1 << 30]
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(edges) - 2)]
    labels.append(f">={edges[-2]}")
    per_arm = {}
    per_band = {}
    for _, leg in legs:
        arm = leg["summary"].get("arm") or "?"
        for r in ok_records(leg, "measured"):
            n_in = int((r.get("usage") or {}).get("prompt_tokens") or 0)
            if not n_in:
                continue
            b = next(i for i in range(len(edges) - 1)
                     if edges[i] <= n_in < edges[i + 1])
            per_arm.setdefault(arm, [0] * (len(edges) - 1))[b] += 1
            per_band.setdefault(r["band"], []).append(n_in)
    print("| prompt tokens | " + " | ".join(sorted(per_arm)) + " |")
    print("|---|" + "---|" * len(per_arm))
    for i, lab in enumerate(labels):
        row = [str(per_arm[a][i]) for a in sorted(per_arm)]
        if any(int(x) for x in row):
            print(f"| {lab} | " + " | ".join(row) + " |")
    print("\n| band | n | min | p50 | p90 | max | mean |")
    print("|---|---|---|---|---|---|---|")
    for band in sorted(per_band, key=lambda b: statistics.median(per_band[b])):
        v = per_band[band]
        print(f"| `{band}` | {len(v)} | {min(v)} | {fmt(percentile(v, 50), 0)} "
              f"| {fmt(percentile(v, 90), 0)} | {max(v)} "
              f"| {fmt(statistics.mean(v), 0)} |")


def print_cells(legs, args):
    cells = {}
    for path, leg in legs:
        cells.setdefault(cell_key(leg), []).append((path, leg))

    print("\n## Per leg: throughput and the spread between rounds\n")
    print("| arm | c | round | ok/n | out tok/s | decode-only tok/s "
          "| mean TTFT ms | p95 TTFT ms | wall s | counted by | publishable |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(cells):
        for path, leg in sorted(cells[key],
                                key=lambda pl: pl[1]["summary"]["round"]):
            s = leg["summary"]
            recs = ok_records(leg, "measured")
            a = axes(recs)
            wall = s["measured_wall_s"]
            out_s = sum(a["out_tokens"]) / wall if wall else 0.0
            dec = 1000.0 / statistics.mean(a["tpot"]) if a["tpot"] else None
            print(f"| {key[0]} | {key[1]} | {s['round']} "
                  f"| {len(recs)}/{s['measured_requests']} | {fmt(out_s, 2)} "
                  f"| {fmt(dec, 2)} "
                  f"| {fmt(statistics.mean(a['ttft']) if a['ttft'] else None)} "
                  f"| {fmt(percentile(a['ttft'], 95))} | {fmt(wall)} "
                  f"| {a['counted_by']} "
                  f"| {'yes' if a['publishable'] else 'NO (G-USAGE)'} |")

    print("\n## Round-to-round spread per cell\n")
    print("| arm | c | out tok/s per round | spread | p95 TTFT ms per round "
          "| spread |")
    print("|---|---|---|---|---|---|")
    for key in sorted(cells):
        tps, p95s = [], []
        for _, leg in cells[key]:
            recs = ok_records(leg, "measured")
            a = axes(recs)
            wall = leg["summary"]["measured_wall_s"]
            tps.append(sum(a["out_tokens"]) / wall if wall else 0.0)
            p95s.append(percentile(a["ttft"], 95))
        st, sp = spread(tps), spread(p95s)
        print(f"| {key[0]} | {key[1]} | "
              + " / ".join(fmt(v, 2) for v in tps)
              + f" | {'-' if st is None else f'{100 * st:.1f}%'} | "
              + " / ".join(fmt(v) for v in p95s)
              + f" | {'-' if sp is None else f'{100 * sp:.1f}%'} |")

    for phase, title in (("measured", "warm only, warmup discarded"),
                         (None, "with warmup included")):
        print(f"\n## Percentiles, pooled over both rounds ({title})\n")
        print("Pooled n per cell is both rounds together, so p99 and max are "
              "read off that pool and not off a single leg.\n")
        cols = " | ".join(f"p{p:g}" for p in PCTS)
        print(f"| arm | c | n | axis | {cols} | max | mean |")
        print("|---|---|---|" + "---|" * (len(PCTS) + 3))
        for key in sorted(cells):
            pooled = []
            for _, leg in cells[key]:
                pooled.extend(ok_records(leg, phase))
            a = axes(pooled)
            for axis in ("ttft", "ttft_corrected", "tpot", "itl", "e2el"):
                v = a[axis]
                if not v:
                    continue
                name = axis + (" (est)" if axis == "ttft_corrected" else "")
                print(f"| {key[0]} | {key[1]} | {len(v)} | {name} ms | "
                      + " | ".join(fmt(percentile(v, p)) for p in PCTS)
                      + f" | {fmt(max(v))} | {fmt(statistics.mean(v))} |")

    print("\n## Streaming granularity, and what it does to TTFT\n")
    print("`first chunk tokens` is an ESTIMATE: streaming carries no per-chunk "
          "token count, so it is the first chunk's characters divided by that "
          "request's own characters-per-token.\n")
    print("| arm | c | tokens per chunk | chars per token | first chunk tokens "
          "(est) | p50 TTFT raw ms | p50 TTFT corrected ms (est) | corrections refused |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(cells):
        pooled = []
        for _, leg in cells[key]:
            pooled.extend(ok_records(leg, "measured"))
        a = axes(pooled)
        mean = lambda xs: statistics.mean(xs) if xs else None       # noqa: E731
        print(f"| {key[0]} | {key[1]} | {fmt(mean(a['toks_per_chunk']), 2)} "
              f"| {fmt(mean(a['chars_per_token']), 2)} "
              f"| {fmt(mean(a['first_chunk_tokens']), 2)} "
              f"| {fmt(percentile(a['ttft'], 50))} "
              f"| {fmt(percentile(a['ttft_corrected'], 50))} "
              f"| {a['over_corrected']}/{a['n']} |")

    print("\n## Acceptance\n")
    print("| arm | c | round | source | value |")
    print("|---|---|---|---|---|")
    for key in sorted(cells):
        for _, leg in sorted(cells[key],
                             key=lambda pl: pl[1]["summary"]["round"]):
            acc = acceptance(leg)
            if acc["source"] == "/metrics":
                val = (f"accepted/proposed = {acc['accepted']:.0f}/{acc['draft']:.0f}"
                       f" = {acc['rate']:.3f}, "
                       f"{fmt(acc['per_output_token'], 3)} per output token")
            elif acc["source"] == "usage":
                val = (f"accepted {acc['accepted']:.0f}, "
                       f"{fmt(acc['per_output_token'], 3)} per output token "
                       f"(this engine exports no proposed count)")
            elif acc["source"] == "absent":
                val = "NOT EXPOSED BY THIS ENGINE"
            else:
                val = acc["source"]
            print(f"| {key[0]} | {key[1]} | {leg['summary']['round']} "
                  f"| {acc['source']} | {val} |")

    print("\n## Cold start: what the first request costs\n")
    print("| arm | c | round | warmup TTFT ms, first request | warm p50 TTFT ms "
          "| mean TTFT ms warm | mean TTFT ms with warmup |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(cells):
        for _, leg in sorted(cells[key],
                             key=lambda pl: pl[1]["summary"]["round"]):
            warm = ok_records(leg, "warmup")
            # THE FIRST warmup request, by dispatch order, not the cheapest one.
            # `min` over the ttft values returns whichever warmup request paid
            # the LEAST, which at c > 1 is precisely the one that did not pay the
            # cold start. This column exists to publish that cost.
            first_rec = min(warm, key=lambda r: (r.get("t_dispatch", 0.0),
                                                 r.get("i", 0)), default=None)
            first = first_rec["ttft"] if first_rec is not None else None
            aw = axes(ok_records(leg, "measured"))
            aa = axes(ok_records(leg, None))
            print(f"| {key[0]} | {key[1]} | {leg['summary']['round']} "
                  f"| {fmt(first * 1000 if first is not None else None)} "
                  f"| {fmt(percentile(aw['ttft'], 50))} "
                  f"| {fmt(statistics.mean(aw['ttft']) if aw['ttft'] else None)} "
                  f"| {fmt(statistics.mean(aa['ttft']) if aa['ttft'] else None)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.json",
                    help="leg JSON files written by client.py")
    ap.add_argument("--dir", default=".")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    legs = [(p, leg) for p, leg in load_legs(paths)
            if isinstance(leg, dict) and "records" in leg and "summary" in leg]
    if not legs:
        print("NO LEG PRODUCED A RESULT", file=sys.stderr)
        return 1
    print(f"# Variadic load report over {len(legs)} legs")
    print_histogram(legs, args)
    print_cells(legs, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
