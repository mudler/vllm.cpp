#!/usr/bin/env python3
"""Tenstorrent clock-state sampler and judge — the sibling of gpu_clock_state.

Implements the ".agents/benchmarking.md" clock contract for TT boards:
AICLK sampling via `tt-smi -s` JSON snapshots, stop-time summary writing,
and cross-window pairing assertions with the NVIDIA helper's thresholds.
Where the platform has no analog the field is recorded NOT APPLICABLE with
a reason, never dropped silently; throttle-unobservability is carried as a
caveat in every output. See .agents/specs/tt-clock-state.md (#2005).

Exit codes match gpu_clock_state: 0 on success, 1 when refusal reasons exist.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time

#: The board query. `tt-smi -s` snapshots stdout in ~430 ms (measured
#: 2026-08-26, P150 thalia), so --interval 1 is supportable and ~2 Hz is
#: the practical ceiling; ~0.43 of every second is subprocess startup.
TT_SMI = "/home/lu_zero/Sources/tt/.venv/bin/tt-smi"

#: Device node prefix a healthy leg holds open while it measures.
TT_DEV_PREFIX = "/dev/tenstorrent/"

BOOT_ID_PATH = pathlib.Path("/proc/sys/kernel/random/boot_id")

#: Within-run spread ceiling, percentage (gpu_clock_state).
MAX_WITHIN_RUN_SPREAD_PCT = 5.0

#: Cross-arm MEDIAN clock offset ceiling, percentage (gpu_clock_state).
MAX_CROSS_ARM_OFFSET_PCT = 1.0

#: Cross-arm MEAN offset ceiling — a SEPARATE rule, not a restatement: the
#: excursions sit below the median and do not cancel between arms (#1546),
#: and throughput is an integral over the window.
MAX_CROSS_ARM_MEAN_OFFSET_PCT = 1.0

#: Retained busy samples per window (gpu_clock_state). spread_pct over n==1
#: is definitionally 0.00% — the best score a gate can award — so without a
#: floor the unobserved window outscores the observed one.
MIN_BUSY_SAMPLES = 30

#: Busy fraction of the window (gpu_clock_state): 30 busy among 3000 idle
#: still clears the count floor, and this field betrays an orphan sampler.
MIN_BUSY_FRACTION = 0.5


def read_boot_id(path: pathlib.Path = BOOT_ID_PATH) -> str:
    return path.read_text(encoding="utf-8").strip()


def _hexint(value):
    """tt-smi encodes telemetry as hex strings; pass ints through."""
    if isinstance(value, str) and value.startswith("0x"):
        try:
            return int(value, 16)
        except ValueError:
            return None
    if isinstance(value, int):
        return value
    return None


def sample_once(smi_path: str = TT_SMI, timeout_s: float = 10.0) -> dict | None:
    """One tt-smi snapshot -> flattened board-0 sample dict, or None."""
    try:
        proc = subprocess.run(
            [smi_path, "-s"], capture_output=True, text=True, timeout=timeout_s
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    devices = doc.get("device_info") or []
    host_sw = doc.get("host_sw_vers") or {}
    driver = (doc.get("host_info") or {}).get("Driver")
    if not devices:
        return None
    telem = devices[0].get("smbus_telem") or {}
    aiclk = _hexint(telem.get("AICLK"))
    if aiclk is None:
        return None
    return {
        "t": time.time(),
        "aiclk": aiclk,
        "arcclk": _hexint(telem.get("ARCCLK")),
        "axiclk": _hexint(telem.get("AXICLK")),
        "vcore": _hexint(telem.get("VCORE")),
        "tdp": _hexint(telem.get("TDP")),
        "tdc": _hexint(telem.get("TDC")),
        "asic_temp_raw": telem.get("ASIC_TEMPERATURE"),
        "fan_rpm": _hexint(telem.get("FAN_RPM")),
        "board_id": "{}:{}".format(
            telem.get("BOARD_ID_HIGH"), telem.get("BOARD_ID_LOW")
        ),
        "flash_bundle": telem.get("FLASH_BUNDLE_VERSION"),
        "kmd_driver": driver,
        "tt_smi_version": host_sw.get("tt_smi"),
        "umd_version": host_sw.get("tt_umd"),
    }


def pid_holds_tt_device(pid: int) -> bool:
    """True when pid is alive AND holds an fd on a tenstorrent device node."""
    fd_dir = pathlib.Path(f"/proc/{pid}/fd")
    try:
        for fd in fd_dir.iterdir():
            try:
                if str(fd.readlink()).startswith(TT_DEV_PREFIX):
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def fold(samples: list[dict], claimed_max_mhz: int | None = None,
         provenance: str | None = None, allow_cross_boot: bool = False) -> dict:
    """Fold raw samples into the window record shared by judge and tests."""
    aiclks = [s["aiclk"] for s in samples]
    median = statistics.median(aiclks) if aiclks else None
    identity_keys = (
        "board_id", "flash_bundle", "kmd_driver", "tt_smi_version",
        "umd_version",
    )
    rec = {
        "n_samples": len(samples),
        "n": len(aiclks),
        "min": min(aiclks) if aiclks else None,
        "median": median,
        "max": max(aiclks) if aiclks else None,
        "mean": (
            sum(aiclks) / len(aiclks) if aiclks else None
        ),
        "spread_pct": (
            (max(aiclks) - min(aiclks)) / median * 100.0
            if median else 0.0
        ),
        "busy_n": sum(1 for s in samples if s.get("busy")),
        "idle_n": sum(1 for s in samples if not s.get("busy")),
        "busy_fraction": (
            sum(1 for s in samples if s.get("busy")) / len(samples)
            if samples else 0.0
        ),
        "boot_id": read_boot_id(),
        "claimed_max_aiclk_mhz": claimed_max_mhz,
        "claimed_max_provenance": provenance,
        "not_applicable": {
            "applications_clocks_setting": "no TT analog knob exists",
            "persistence_mode": "no TT analog knob exists",
            "throttle_reasons_live": (
                "no live bitmap exposed by tt-smi; THM_LIMIT_*/VDD_LIMITS "
                "are static limits"
            ),
        },
        "context_medians": {},
    }
    last = samples[-1] if samples else {}
    for key in identity_keys:
        rec[key] = last.get(key)
    for key in ("arcclk", "axiclk", "vcore", "tdp", "tdc", "fan_rpm"):
        vals = [s[key] for s in samples if isinstance(s.get(key), int)]
        rec["context_medians"][key] = statistics.median(vals) if vals else None
    rec["_aiclks"] = aiclks
    rec["_t"] = [s["t"] for s in samples]
    rec["_busy"] = [bool(s.get("busy")) for s in samples]
    if allow_cross_boot:
        rec["_allow_cross_boot"] = True
        rec["_waiver_caveat"] = (
            "boot id waived at request; machine identity compared "
            "unconditionally"
        )
    return rec


def clock_state_reasons(record: dict, other: dict | None = None) -> list[str]:
    """Per-window plus cross-window refusal reasons, rule-for-rule."""
    reasons: list[str] = []
    if record.get("median") is None:
        reasons.append(
            f"idle-or-empty window: n={record.get('n')} median=None"
        )
        return reasons
    if record["n"] < MIN_BUSY_SAMPLES:
        reasons.append(
            f"retained busy samples {record['n']} < MIN_BUSY_SAMPLES="
            f"{MIN_BUSY_SAMPLES}"
        )
    if record["busy_fraction"] < MIN_BUSY_FRACTION:
        reasons.append(
            f"busy fraction {record['busy_fraction']:.3f} < MIN_BUSY_FRACTION="
            f"{MIN_BUSY_FRACTION}"
        )
    if record["spread_pct"] > MAX_WITHIN_RUN_SPREAD_PCT:
        reasons.append(
            f"within-run spread {record['spread_pct']:.2f}% > "
            f"{MAX_WITHIN_RUN_SPREAD_PCT}%"
        )
    if other is None:
        return reasons
    # ---- cross-window terms ----
    for key, label in (
        ("board_id", "board id"),
        ("flash_bundle", "firmware bundle"),
        ("kmd_driver", "KMD/driver"),
        ("tt_smi_version", "tt-smi version"),
        ("umd_version", "UMD version"),
    ):
        if record.get(key) != other.get(key):
            reasons.append(
                f"unconditional mismatch {label}: {record.get(key)!r} vs "
                f"{other.get(key)!r} — a waived boot is NOT a waived machine"
            )
    if record["boot_id"] != other["boot_id"]:
        if not (record.get("_allow_cross_boot") or other.get("_allow_cross_boot")):
            reasons.append(
                f"different boot ids: {str(record['boot_id'])[:8]} vs "
                f"{str(other['boot_id'])[:8]}"
            )
        elif "_waiver_caveat" not in record and "_waiver_caveat" not in other:
            reasons.append(
                "cross-boot pair compared WITHOUT the stamped waiver caveat"
            )
    denom_med = min(record["median"], other["median"])
    med_off = abs(record["median"] - other["median"]) / denom_med * 100.0
    denom_mean = min(record["mean"], other["mean"])
    mean_off = abs(record["mean"] - other["mean"]) / denom_mean * 100.0
    if med_off > MAX_CROSS_ARM_OFFSET_PCT:
        reasons.append(
            f"cross-arm median offset {med_off:.2f}% > "
            f"{MAX_CROSS_ARM_OFFSET_PCT}%"
        )
    if mean_off > MAX_CROSS_ARM_MEAN_OFFSET_PCT:
        reasons.append(
            f"cross-arm mean offset {mean_off:.2f}% > "
            f"{MAX_CROSS_ARM_MEAN_OFFSET_PCT}%"
        )
    return reasons


def judge(records: list[dict]) -> dict:
    """Judge windows pairwise (each against the FIRST); refuse on any reason."""
    comparison: dict = {"reasons": [], "pairs": [], "windows": []}
    base = records[0]
    for rec in records:
        rr = {k: v for k, v in rec.items() if not k.startswith("_")}
        rr["throttle_unobservability_caveat"] = (
            "no live TT throttle bitmap exists; a thermal/VDD clamp during "
            "the window is UNOBSERVED, see spec tt-clock-state.md"
        )
        comparison["windows"].append(rr)
    for idx, rec in enumerate(records[1:], start=1):
        rs = clock_state_reasons(rec, base)
        comparison["pairs"].append({"window": idx, "reasons": rs})
        comparison["reasons"].extend(rs)
    solo = clock_state_reasons(records[0])
    comparison["pairs"].insert(0, {"window": 0, "reasons": solo})
    comparison["reasons"] = solo + comparison["reasons"]
    return comparison


def run_sampler(args: argparse.Namespace) -> int:
    """Sample --duration seconds (or until stdin says 'stop'); summarize at STOP.

    The summary exists only after the sampler stops (#1657). A refused or
    empty window still writes its evidence file; the exit code carries the
    verdict.
    """
    samples: list[dict] = []
    deadline = time.monotonic() + args.duration if args.duration else None
    use_stdin_stop = args.duration is None
    while True:
        snap = sample_once(TT_SMI)
        if snap is not None:
            snap["busy"] = (
                pid_holds_tt_device(args.leg_pid) if args.leg_pid else True
            )
            samples.append(snap)
        if deadline is not None and time.monotonic() >= deadline:
            break
        if use_stdin_stop and sys.stdin.readline().strip():
            break
        time.sleep(args.interval)
    rec = fold(samples, args.claimed_max_aiclk_mhz, args.claimed_max_provenance,
               args.allow_cross_boot)
    rec["interval_s"] = args.interval
    rec["leg_pid"] = args.leg_pid
    pathlib.Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {args.out}: n={rec['n']} busy={rec['busy_n']}")
    return 1 if clock_state_reasons(rec) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample", help="sample until stdin closes/--duration")
    sp.add_argument("--out", required=True)
    sp.add_argument("--interval", type=float, default=1.0)
    sp.add_argument("--duration", type=float, default=None)
    sp.add_argument("--leg-pid", type=int, default=None)
    sp.add_argument("--claimed-max-aiclk-mhz", type=int, default=None)
    sp.add_argument("--claimed-max-provenance", default=None)
    sp.add_argument("--allow-cross-boot", action="store_true")

    jp = sub.add_parser("judge", help="judge 2+ window summaries pairwise")
    jp.add_argument("summaries", nargs="+")

    args = ap.parse_args(argv)
    if args.cmd == "sample":
        return run_sampler(args)
    records = []
    for path in args.summaries:
        doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        doc["_path"] = path
        records.append(doc)
    result = judge(records)
    print(json.dumps(result, indent=1))
    return 0 if not result["reasons"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
