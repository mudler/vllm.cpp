#!/usr/bin/env bash
# Assert the resource came back after the SGLang serve job was killed mid-leg.
#
# The serve job b9e7709d died at 2026-08-23T22:27:00Z with no teardown, so it
# never printed TEARDOWN_VERDICT and never asserted that the GPU was returned.
# This is that assertion, made separately.
#
# It matches on the VENV PATH, never on a script name. `pkill -f` on a launcher
# name is the recorded failure that stranded an EngineCore holding 23 GB across
# three jobs; a broad match can also kill another session's work.
set -uo pipefail

# ARCHIVE THE OUTPUT. The first run of this script on 2026-08-23 wrote only to
# stdout, so its verdict survived nowhere but the controller's job store and had
# to be recovered later with `rc logs <job-id>`. Its siblings
# `rc-sglang-oracle-lease.sh` install/serve both `tee` into `$W/out/`, and this
# one now does too, so a reap job archives itself on the shared /workspace.
W="${W:-/workspace/sglang-w2}"
OUT="$W/out/reap-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT" || { echo "CANNOT CREATE $OUT -- output would go unarchived"; exit 3; }
exec > >(tee -a "$OUT/job.log") 2>&1
echo "ARCHIVE=$OUT/job.log"

echo "### reap $(date -u)"
hostname
cat /proc/sys/kernel/random/boot_id
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
echo "COMPUTE_APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+')"
# The unambiguous signature: OUR virtual environment's interpreter.
mapfile -t PIDS < <(ps -eo pid,args | awk '/\/tmp\/sgenv\/bin\/python/ && !/awk/ {print $1}')
echo "SGENV_PROCS=${#PIDS[@]}"
if [ "${#PIDS[@]}" -gt 0 ]; then
  ps -o pid,pgid,etime,rss,args -p "$(IFS=,; echo "${PIDS[*]}")" | head -20
  for p in "${PIDS[@]}"; do
    g=$(ps -o pgid= -p "$p" | tr -d ' ')
    echo "KILLING pgid=$g (pid $p)"
    kill -9 -- "-$g" 2>/dev/null
  done
  sleep 10
  mapfile -t LEFT < <(ps -eo pid,args | awk '/\/tmp\/sgenv\/bin\/python/ && !/awk/ {print $1}')
  echo "SGENV_PROCS_AFTER=${#LEFT[@]}"
else
  echo "NOTHING STRANDED: no process is running our venv interpreter."
fi
free -g | head -2 | tail -1
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
du -sh /tmp/sgenv /tmp/ckpt38 2>/dev/null
df -h /tmp | tail -1
echo "DONE_MARKER_SGLANG_W2_REAP"
