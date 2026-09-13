#!/usr/bin/env bash
# BACKEND-ROCM wave 2. Lease fe7c11d3 established the blocker: rocprofv3
# --kernel-trace over the 67 GiB qwen4_exp UD-IQ1_S ran 60 MINUTES against an
# unprofiled 76-SECOND control, blew through `timeout 3600` without exiting
# (SIGTERM alone does not end it), and never wrote a CSV. That is the
# 2026-09-07 stall reproducing on kernel-trace alone.
#
# This job separates "the profiler is broken on gfx1151" from "the profiler
# cannot survive THIS artifact", which are different results with different
# next actions. Two models, same binary, same board, profiled and unprofiled:
#   27B  Qwen3.8-27B-Q4_K_M, 17 GiB -- the artifact the 2026-09-07 capture
#        profiled successfully. If it profiles, the instrument works.
#   Q4E  Qwen3.8-Flash-Next UD-IQ1_S, 67 GiB, 12 tokens x 2 -- the smallest
#        profiled window that can still produce a decode-step ranking.
# Every timeout now carries -k, so a profiler that ignores SIGTERM is KILLED
# and the leg ends instead of consuming the lease.
set -u
if [ "${KA_STAGED:-0}" != "1" ]; then
  cp -f "$0" /tmp/kattrib4-staged.sh 2>/dev/null && chmod +x /tmp/kattrib4-staged.sh \
    && exec env KA_STAGED=1 bash /tmp/kattrib4-staged.sh "$@"
  echo "FATAL: could not stage into /tmp"; exit 80
fi
BASE_SHA=cc0e827dd9425ee00f3827e2fc36d2ac46560202
Q4E_DIR=/tmp/ckpt-iq1s
Q4E=$Q4E_DIR/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
Q4E_SHA=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
B27_NAS=/workspace/ckpt/qwen38-27b-q4km/Qwen3.8-27B-Q4_K_M.gguf
B27=/tmp/kattrib-27b/Qwen3.8-27B-Q4_K_M.gguf
B27_SHA=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
W=/workspace/rocm-kernel-attrib
OUT=$W/w2-$(date -u +%Y%m%dT%H%M%SZ)
L=/tmp/kattrib; SRC=$L/src; BLD=$L/build
mkdir -p "$OUT" "$L" || { echo "FATAL mkdir"; exit 81; }
exec 2>&1   # NEVER `exec >file`: rc reaps the job on EOF (lease b4e2330a)
( while true; do echo "[hb] $(date -u +%H:%M:%S) free=$(df -BG --output=avail /tmp|tail -1)"; sleep 120; done ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT INT TERM
say(){ echo; echo "===== $* ====="; date -u +%FT%TZ; }
die(){ echo "FATAL: $1"; echo "JOB_VERDICT=FAIL"; exit "${2:-1}"; }

say "0. IDENTITY"
hostname; echo "RC_JOB_ID=${RC_JOB_ID:-UNSET}"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
free -g|head -2; df -h /tmp|tail -1
[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ] || die "HSA_OVERRIDE_GFX_VERSION set" 91
env | grep -qE '^(HSA_|ROCR_|HIP_|GGML_|VT_)' && { env|grep -E '^(HSA_|ROCR_|HIP_|GGML_|VT_)'; die "inherited env" 91; }
echo "inherited_env NONE   <- VT_ROCM_MANAGED_ALLOC unset, as required"

say "1. ROCm (already provisioned on this boot; reinstall only if gone)"
export DEBIAN_FRONTEND=noninteractive
if [ ! -x /opt/rocm/bin/rocprofv3 ]; then
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq wget gnupg ca-certificates git cmake ninja-build ccache python3 >/dev/null 2>&1
  mkdir -p /etc/apt/keyrings
  wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main" > /etc/apt/sources.list.d/rocm.list
  printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' > /etc/apt/preferences.d/rocm-radeon-first
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq rocm-hip-sdk rocprofiler-sdk hsa-amd-aqlprofile libdw1t64 > "$L/apt2.log" 2>&1 || die "rocm install failed: $(tail -5 "$L/apt2.log")" 84
fi
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
echo "rocm=$(cat /opt/rocm/.info/version)"; rocprofv3 --version 2>&1|grep -i version|head -3

say "2. BUILD (ccache-warm; the tree from lease 067720d8 should survive)"
if [ ! -d "$SRC/.git" ]; then git clone -q https://github.com/mudler/vllm.cpp "$SRC" || die clone 85; fi
git -C "$SRC" fetch -q origin 2>/dev/null; git -C "$SRC" checkout -q "$BASE_SHA" || die checkout 86
[ "$(git -C "$SRC" rev-parse HEAD)" = "$BASE_SHA" ] || die "revision mismatch" 86
echo "source_revision=$(git -C "$SRC" rev-parse HEAD) dirty=$(git -C "$SRC" status --porcelain|wc -l)"
export CCACHE_DIR=/root/ccache CCACHE_REMOTE_STORAGE=file:/workspace/ccache-remote-strix CCACHE_MAXSIZE=20G
mkdir -p "$CCACHE_DIR"
HIPCC=/opt/rocm/llvm/bin/clang++
if [ ! -f "$BLD/CMakeCache.txt" ]; then
  cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_HIP=ON \
    -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 -DCMAKE_HIP_COMPILER="$HIPCC" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
    -DCMAKE_EXE_LINKER_FLAGS="-L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib" -DROCM_PATH=/opt/rocm > "$L/cmake2.log" 2>&1
  grep -q 'ROCm backend: ENABLED for arch(es) \[gfx1151\]' "$L/cmake2.log" || { tail -20 "$L/cmake2.log"; die "gfx1151 not enabled" 92; }
fi
ninja -C "$BLD" -j 4 vllm-cli > "$L/ninja2.log" 2>&1 || { grep -i error "$L/ninja2.log"|head -10; die "build failed" 93; }
CLI=$(find "$BLD" -name vllm-cli -type f -perm -u+x|head -1); [ -n "$CLI" ] || die "no vllm-cli" 93
echo "CLI=$CLI cli_sha256=$(sha256sum "$CLI"|cut -d' ' -f1)"
ldd "$CLI" | grep -q libamdhip64.so.7 || die "no libamdhip64.so.7" 94
echo "not_found_libs=$(ldd "$CLI"|grep -c 'not found')   <- must be 0"

say "3. ARTIFACTS"
[ -f "$Q4E" ] || die "qwen4_exp shard 1 gone from $Q4E_DIR" 95
echo "q4exp_sha256=$(sha256sum "$Q4E"|cut -d' ' -f1)  expect $Q4E_SHA"
[ "$(sha256sum "$Q4E"|cut -d' ' -f1)" = "$Q4E_SHA" ] || die "q4exp sha mismatch" 95
mkdir -p "$(dirname "$B27")"
if [ ! -f "$B27" ] || [ "$(stat -c%s "$B27")" != "$(stat -c%s "$B27_NAS")" ]; then
  T=$(date +%s); cp "$B27_NAS" "$B27" || die "27b stage failed" 95; echo "staged_27b_secs=$(( $(date +%s) - T ))"
fi
echo "b27_sha256=$(sha256sum "$B27"|cut -d' ' -f1)  expect $B27_SHA"
[ "$(sha256sum "$B27"|cut -d' ' -f1)" = "$B27_SHA" ] || die "27b sha mismatch" 95

# leg <tag> <profiled> <model> <ngen> <repeat> <timeout_s>
leg(){
  local tag=$1 prof=$2 model=$3 ngen=$4 rep=$5 tmo=$6
  local D=$L/w2-$tag; rm -rf "$D"; mkdir -p "$D/trace"
  say "LEG[$tag] profiled=$prof model=$(basename "$model") ngen=$ngen repeat=$rep timeout=${tmo}s"
  local PRE=()
  [ "$prof" = 1 ] && PRE=(rocprofv3 --kernel-trace --output-format csv --output-directory "$D/trace" --)
  local S=$(date +%s)
  # -k 60: SIGTERM alone did NOT end rocprofv3 on lease fe7c11d3. Escalate.
  timeout -k 60 --foreground "$tmo" "${PRE[@]}" "$CLI" --model "$model" --device auto \
      --prompt 'The capital of France is' --max-tokens "$ngen" --temperature 0 \
      --max-num-seqs 1 --repeat "$rep" > "$D/stdout.txt" 2> "$D/stderr.txt" </dev/null
  local r=$?
  echo "LEG[$tag] rc=$r wall=$(( $(date +%s) - S ))s   <- the RUN's own status, no pipe"
  case $r in
    0) echo "LEG[$tag] VERDICT: EXITED 0";;
    124) echo "LEG[$tag] VERDICT: TIMED OUT at ${tmo}s (SIGTERM)";;
    137) echo "LEG[$tag] VERDICT: SIGKILLED after the -k grace -- it IGNORED SIGTERM";;
    13[0-9]|1[4-9][0-9]) echo "LEG[$tag] VERDICT: SIGNAL $((r-128)) -- a CRASH";;
    *) echo "LEG[$tag] VERDICT: exited $r";;
  esac
  grep -a "tok_s=" "$D/stderr.txt" | sed "s/^/$tag  /"
  echo "$tag  output: $(head -c 160 "$D/stdout.txt"|tr '\n' ' ')"
  grep -aiE "illegal memory access|HSA_STATUS_ERROR|Memory access fault|core dumped" "$D/stderr.txt"|head -3|sed "s/^/$tag  CRASHSIG /"
  if [ "$prof" = 1 ]; then
    echo -n "$tag  swap_warnings="; grep -aci "swap" "$D/stderr.txt"
    echo "$tag  trace dir contents:"; ls -la "$D/trace" 2>/dev/null | head -10
    local KT=$(find "$D/trace" -name '*kernel_trace.csv' 2>/dev/null|head -1)
    if [ -n "$KT" ]; then
      echo "$tag  kernel_trace=$KT rows=$(( $(wc -l < "$KT") - 1 ))"
      python3 -c "
import csv,collections
rows=list(csv.DictReader(open('$KT')))
c=collections.Counter(r['Kernel_Name'] for r in rows)
print('$tag  top dispatch counts:')
for n,k in c.most_common(15): print('   %7d  %s'%(k,n[:86]))
d=[int(r['End_Timestamp'])-int(r['Start_Timestamp']) for r in rows]
print('$tag  duration sanity: n=%d neg=%d zero=%d min=%d max=%d sum_ms=%.1f'%(len(d),sum(1 for x in d if x<0),sum(1 for x in d if x==0),min(d),max(d),sum(d)/1e6))
" 2>&1 | head -25
      gzip -9 -c "$KT" > "$OUT/w2-$tag-kernel_trace.csv.gz"
    else
      echo "$tag  NO KERNEL TRACE PRODUCED"
    fi
  fi
  gzip -9 -c "$D/stderr.txt" > "$OUT/w2-$tag-stderr.txt.gz"
  cp -f "$D/stdout.txt" "$OUT/w2-$tag-stdout.txt" 2>/dev/null
  return 0
}

say "4. LEGS -- the 27B control first, because it decides which result this is"
leg C27u 0 "$B27" 40 3 900
leg C27p 1 "$B27" 40 3 1800
leg Q4Eu 0 "$Q4E" 12 2 900
leg Q4Ep 1 "$Q4E" 12 2 1800

say "5. ARCHIVE"
{ echo "base_sha=$BASE_SHA"; echo "cli_sha256=$(sha256sum "$CLI"|cut -d' ' -f1)";
  echo "q4exp_sha256=$Q4E_SHA"; echo "b27_sha256=$B27_SHA";
  echo "rocm=$(cat /opt/rocm/.info/version)"; echo "rc_job=${RC_JOB_ID:-UNSET}";
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"; } > "$OUT/IDENTITY.txt"
cat "$OUT/IDENTITY.txt"; ls -la "$OUT"
say "DONE /workspace $OUT"
