#!/usr/bin/env bash
# Submit build_w0.sh to a leased fleet device and stream its output.
#
#   ./run_on_lease.sh [device] > job.log 2>&1
#
# orin:gpu0 has LOCAL storage only and cannot see the NAS, so probe.cu and
# build_w0.sh are staged onto the box by base64 on the `rc run` command line
# rather than through /workspace. Compile-only work needs no particular GPU:
# cuobjdump and nvdisasm read a compiled object and never execute the kernel, so
# -arch=sm_121a needs the toolkit and the flag, not a Blackwell device. Every
# fleet box is aarch64 and the host does not enter the codegen.
#
# The job dies if the submitting client detaches, so redirect this script's
# stdout to the file you intend to keep; the script also leaves a copy in
# /workspace/replayssm-w0b/out on the device.
set -euo pipefail
DEV="${1:-orin:gpu0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RC_SUBMITTER="${RC_SUBMITTER:-claude/mudler-ubuntu-box/replayssm-w0b}"
PB="$(base64 -w0 "$HERE/probe.cu")"
SB="$(base64 -w0 "$HERE/build_w0.sh")"
exec rc run -d "$DEV" --max-runtime 40m --idle-timeout 0 -- bash -lc "
set -u
W=/workspace/replayssm-w0b
mkdir -p \$W
echo $PB | base64 -d > \$W/probe.cu
echo $SB | base64 -d > \$W/build_w0.sh
chmod +x \$W/build_w0.sh
bash \$W/build_w0.sh
"
