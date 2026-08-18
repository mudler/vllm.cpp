#!/usr/bin/env bash
# Submit count_sass.sh to a leased fleet device and stream its output.
#
#   ./run_sass_on_lease.sh [device] > job3_sass.log 2>&1
#
# Same staging constraint as run_on_lease.sh: orin:gpu0 has LOCAL storage only
# and cannot see the NAS, so probe.cu and count_sass.sh go onto the box by
# base64 on the `rc run` command line. Compile-only, so the device choice is
# about toolkit availability, not about the GPU.
#
# The job dies if the submitting client detaches, so redirect this script's
# stdout to the file you intend to keep.
set -euo pipefail
DEV="${1:-orin:gpu0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RC_SUBMITTER="${RC_SUBMITTER:-claude/mudler-ubuntu-box/replayssm-w0b-sass}"
PB="$(base64 -w0 "$HERE/probe.cu")"
SB="$(base64 -w0 "$HERE/count_sass.sh")"
exec rc run -d "$DEV" --max-runtime 30m --idle-timeout 0 -- bash -lc "
set -u
W=/workspace/replayssm-w0b
mkdir -p \$W
echo $PB | base64 -d > \$W/probe.cu
echo $SB | base64 -d > \$W/count_sass.sh
chmod +x \$W/count_sass.sh
bash \$W/count_sass.sh
"
