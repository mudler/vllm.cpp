#!/usr/bin/env python3
import argparse
import glob
import json
import pathlib
import re
import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--interval", type=float, default=1.0)
args = parser.parse_args()

running = True

def stop(_signum, _frame):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

busy_paths = sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"))
sclk_paths = sorted(glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk"))
boot_id = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
args.output.parent.mkdir(parents=True, exist_ok=True)

with args.output.open("x", encoding="utf-8") as out:
    while running:
        now = time.time()
        busy = None
        if busy_paths:
            try:
                busy = int(pathlib.Path(busy_paths[0]).read_text().strip())
            except (OSError, ValueError):
                busy = None
        sclk_raw = None
        sclk_mhz = None
        if sclk_paths:
            try:
                sclk_raw = pathlib.Path(sclk_paths[0]).read_text().strip()
                active = next((line for line in sclk_raw.splitlines() if "*" in line), "")
                match = re.search(r"([0-9]+)\s*[Mm][Hh][Zz]", active)
                if match:
                    sclk_mhz = int(match.group(1))
            except OSError:
                pass
        out.write(json.dumps({"timestamp": now, "boot_id": boot_id, "busy_percent": busy, "sclk_mhz": sclk_mhz, "sclk_raw": sclk_raw}, sort_keys=True) + "\n")
        out.flush()
        time.sleep(args.interval)
