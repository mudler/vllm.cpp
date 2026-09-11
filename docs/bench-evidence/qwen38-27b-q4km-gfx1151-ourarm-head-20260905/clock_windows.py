#!/usr/bin/env python3
"""Reproduce lease-2 clock windows from committed logs, without a GPU."""
import argparse
import gzip
import json
import math
from pathlib import Path
import re
import statistics

WINDOW = re.compile(r"vllm-cli: run=(\d+)/4 generate_start_unix=([\d.]+) generate_end_unix=([\d.]+)")


def summarize(samples):
    if not samples:
        raise ValueError("no clock samples in generation window")
    return {"samples": len(samples),
            "sclk_mhz_mean": statistics.mean(row["sclk_mhz"] for row in samples),
            "busy_percent_mean": statistics.mean(row["busy_percent"] for row in samples)}


def fold(directory):
    legs, all_samples, all_warm = [], [], []
    for count in (64, 128):
        for repetition in range(1, 5):
            tag = f"n{count}-r{repetition}"
            text = (directory / f"{tag}.err.txt").read_text()
            windows = [(int(run), float(start), float(end)) for run, start, end in WINDOW.findall(text)]
            if [run for run, _, _ in windows] != [1, 2, 3, 4]:
                raise ValueError(f"{tag}: incomplete or duplicate generation windows")
            previous_end = -math.inf
            for _, start, end in windows:
                if not math.isfinite(start) or not math.isfinite(end) or start >= end or start <= previous_end:
                    raise ValueError(f"{tag}: invalid or overlapping generation windows")
                previous_end = end
            with gzip.open(directory / f"clock-{tag}.jsonl.gz", "rt") as stream:
                samples = [json.loads(line) for line in stream if line.strip()]
            for sample in samples:
                if not all(isinstance(sample[key], (int, float)) and math.isfinite(sample[key])
                           for key in ("timestamp", "sclk_mhz", "busy_percent")):
                    raise ValueError(f"{tag}: invalid clock sample")
            warm, generations = [], []
            for run, start, end in windows:
                selected = [row for row in samples if start <= row["timestamp"] <= end]
                generations.append({"run": run, "start_unix": start, "end_unix": end, **summarize(selected)})
                if run > 1:
                    warm.extend(selected)
            legs.append({"leg": tag, "whole_leg": summarize(samples), "generations": generations,
                         "warm": summarize(warm)})
            all_samples.extend(samples)
            all_warm.extend(warm)
    return {"evidence_dir": str(directory.resolve()), "token_gate": "FAIL (carried, not remeasured)",
            "window": "inclusive Unix timestamps of warm generations 2 through 4; includes prefill",
            "interpretation": "Busy percentage is sampled GPU activity, not occupancy or a quantitative bound on host idle time.",
            "legs": legs, "pooled_whole_leg": summarize(all_samples), "pooled_warm": summarize(all_warm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(fold(args.evidence_dir), indent=2))


if __name__ == "__main__":
    main()
