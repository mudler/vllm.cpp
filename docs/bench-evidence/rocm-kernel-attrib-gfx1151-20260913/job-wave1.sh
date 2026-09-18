#!/usr/bin/env bash
# BACKEND-ROCM: build a per-kernel decode-time attribution for gfx1151.
#
# THE QUESTION. Qwen3.8-Flash-Next decodes at 5.0-5.3 tok/s on strix:gpu0 and we
# cannot say where that time goes, because we own no per-kernel ROCm timing
# instrument. #3040 records "invalid profiler timestamps" on this board from the
# 2026-09-07 capture and that record has been read since as a blanket refusal.
#
# WHAT THIS JOB DOES. It runs rocprofv3 --kernel-trace ONLY -- the exact leg that
# COMPLETED on 2026-09-07 (lease 25065496, 64 tokens, 85,737 dispatch rows) --
# over a qwen4_exp decode, and pairs it with an unprofiled control on the same
# binary and artifact so the profiler's own distortion is MEASURED, not assumed.
# Every trace-derived number is produced offline afterwards; no in-process probe
# is added, so the profiled binary is byte-identical to the control binary.
#
# NOT CLAIMED: any cross-engine ratio. The qwen4_exp ROCm arm has NO declared
# token-exact gate (ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG). Every figure here is
# this engine against itself.
set -u
if [ "${KA_STAGED:-0}" != "1" ]; then
  cp -f "$0" /tmp/kattrib-staged.sh 2>/dev/null && chmod +x /tmp/kattrib-staged.sh \
    && exec env KA_STAGED=1 bash /tmp/kattrib-staged.sh "$@"
  echo "FATAL: could not stage into /tmp"; exit 80
fi

TAG=kattrib
BASE_SHA=cc0e827dd9425ee00f3827e2fc36d2ac46560202
GGUF=/tmp/ckpt-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
WANT_GGUF=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
W=/workspace/rocm-kernel-attrib
OUT=$W/$(date -u +%Y%m%dT%H%M%SZ)
L=/tmp/$TAG
SRC=$L/src
BLD=$L/build
NGEN=40
REPEAT=4          # run 1 is the lazy-staging warmup and is DISCARDED everywhere

mkdir -p "$OUT" "$L" || { echo "FATAL: mkdir"; exit 81; }
# Output streams to rc's own stdout. An `exec >file` here closes the job's
# stdout, and rc then reports "log stream ended: unexpected EOF" and REAPS the
# job -- measured on lease b4e2330a, which died before it installed anything.
exec 2>&1
( while true; do echo "[hb] $(date -u +%H:%M:%S) free=$(df -BG --output=avail /tmp|tail -1)"; sleep 180; done ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT INT TERM
say(){ echo; echo "===== $* ====="; date -u +%FT%TZ; }
die(){ echo "FATAL: $1"; echo "JOB_VERDICT=FAIL"; exit "${2:-1}"; }

say "0. IDENTITY and INHERITED ENVIRONMENT"
hostname; uname -m; echo "nproc=$(nproc)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "uptime_s=$(cut -d' ' -f1 /proc/uptime)"
free -g|head -2; df -h / /tmp /workspace|tail -4
# The one that would silently measure a different device.
[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ] || die "HSA_OVERRIDE_GFX_VERSION is set" 91
if env | grep -qE '^(HSA_|ROCR_|HIP_|GGML_|VT_)'; then
  env | grep -E '^(HSA_|ROCR_|HIP_|GGML_|VT_)' | sed 's/^/inherited_env /'
  die "an HSA_/ROCR_/HIP_/GGML_/VT_ variable was inherited" 91
fi
echo "inherited_env NONE   <- VT_ROCM_MANAGED_ALLOC is therefore UNSET, as required"
ps -eo pid,etimes,comm --sort=-etimes 2>/dev/null | head -6

say "1. PROVISION ROCm 7.2.4 and the profiler"
export DEBIAN_FRONTEND=noninteractive
T=$(date +%s)
if [ ! -x /opt/rocm/bin/rocprofv3 ]; then
  apt-get update -qq > "$L/apt.log" 2>&1
  apt-get install -y -qq wget gnupg ca-certificates git cmake ninja-build ccache python3 curl >> "$L/apt.log" 2>&1 \
    || die "base apt failed; $(tail -5 "$L/apt.log")" 82
  mkdir -p /etc/apt/keyrings
  wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg 2>>"$L/apt.log"
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main" \
    > /etc/apt/sources.list.d/rocm.list
  # Ubuntu noble ships its OWN rocminfo 5.7.1, rocm-cmake 6.0.0 and hipcc 5.7.1 in
  # universe. Without this pin apt prefers them and rocm-hip-sdk is unsatisfiable:
  # "rocm-hip-runtime : Depends: rocminfo (= 1.0.0.70204) but 5.7.1-3build1 is to
  # be installed" -- measured on lease 4be26d88, which died on exactly that.
  printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
    > /etc/apt/preferences.d/rocm-radeon-first
  apt-get update -qq >> "$L/apt.log" 2>&1 || die "radeon apt update failed; $(tail -5 "$L/apt.log")" 83
  apt-get install -y -qq rocm-hip-sdk rocprofiler-sdk hsa-amd-aqlprofile libdw1t64 \
    >> "$L/apt.log" 2>&1 || die "rocm install failed; $(tail -20 "$L/apt.log")" 84
fi
echo "provision_secs=$(( $(date +%s) - T ))"
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
echo "rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || echo NONE)"
command -v rocprofv3 >/dev/null || die "no rocprofv3 after install" 84
rocprofv3 --version 2>&1 | head -12
rocminfo 2>/dev/null | grep -m2 -E "gfx" || echo "rocminfo: no gfx line"
say "1b. PROFILER SMOKE TEST -- it must trace a trivial process before we trust it"
rm -rf "$L/smoke"; mkdir -p "$L/smoke"
rocprofv3 --kernel-trace --output-format csv --output-directory "$L/smoke" -- /bin/true > "$L/smoke.log" 2>&1
echo "smoke rc=$?"; tail -5 "$L/smoke.log"

say "2. SOURCE at the declared pin"
if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone -q https://github.com/mudler/vllm.cpp "$SRC" 2>"$L/clone.log" || die "clone failed; $(cat "$L/clone.log")" 85
fi
git -C "$SRC" fetch -q origin 2>>"$L/clone.log"
git -C "$SRC" checkout -q "$BASE_SHA" 2>>"$L/clone.log" || die "checkout $BASE_SHA failed" 86
GOT_REV=$(git -C "$SRC" rev-parse HEAD)
echo "source_revision=$GOT_REV"
[ "$GOT_REV" = "$BASE_SHA" ] || die "revision mismatch: got $GOT_REV want $BASE_SHA" 86
echo "source_tree=$(git -C "$SRC" rev-parse HEAD^{tree})"
echo "dirty=$(git -C "$SRC" status --porcelain | wc -l)   <- must be 0"
[ "$(git -C "$SRC" status --porcelain | wc -l)" = "0" ] || die "source tree is dirty" 86

say "3. BUILD vllm-cli for gfx1151"
export CCACHE_DIR=/root/ccache
export CCACHE_REMOTE_STORAGE=file:/workspace/ccache-remote-strix
export CCACHE_MAXSIZE=20G
mkdir -p "$CCACHE_DIR"
ln -sf /dev/null "$CCACHE_DIR/.symlink-probe" 2>/dev/null || die "CCACHE_DIR cannot symlink(2)" 87
rm -f "$CCACHE_DIR/.symlink-probe"; ccache -z >/dev/null 2>&1
HIPCC=""; for c in /opt/rocm/llvm/bin/clang++ /opt/rocm/bin/hipcc; do [ -x "$c" ] && { HIPCC=$c; break; }; done
[ -n "$HIPCC" ] || die "no HIP compiler" 88
echo "HIPCC=$HIPCC"; $HIPCC --version | head -2
T=$(date +%s)
cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER="$HIPCC" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib" \
  -DROCM_PATH=/opt/rocm > "$L/cmake.log" 2>&1
echo "cmake rc=$?  secs=$(( $(date +%s) - T ))"
grep -E "ROCm backend|hipBLAS" "$L/cmake.log" | head -5
grep -q 'ROCm backend: ENABLED for arch(es) \[gfx1151\]' "$L/cmake.log" \
  || { tail -25 "$L/cmake.log"; die "ROCm backend NOT enabled for gfx1151" 92; }
T=$(date +%s)
ninja -C "$BLD" -j 4 vllm-cli > "$L/ninja.log" 2>&1
NR=$?; echo "ninja rc=$NR secs=$(( $(date +%s) - T ))"; tail -4 "$L/ninja.log"
[ $NR -eq 0 ] || { grep -iE "error" "$L/ninja.log" | head -20; die "build failed" 93; }
ccache -s 2>/dev/null | grep -iE "cacheable|hit|miss" | head -6
CLI=$(find "$BLD" -name vllm-cli -type f -perm -u+x | head -1)
[ -n "$CLI" ] || die "no vllm-cli produced" 93
echo "CLI=$CLI"; echo "cli_sha256=$(sha256sum "$CLI"|cut -d' ' -f1)"

say "4. ASSERT THE ARTIFACT'S COMPILED FEATURE SET"
ldd "$CLI" | grep -E "amdhip|hipblas|rocblas|not found" | sed 's/^/  /'
ldd "$CLI" | grep -q "libamdhip64.so.7" || die "binary does not resolve libamdhip64.so.7" 94
echo "not_found_libs=$(ldd "$CLI" | grep -c 'not found')   <- must be 0"
[ "$(ldd "$CLI" | grep -c 'not found')" = "0" ] || die "unresolved libraries" 94
echo "--- archive the build so a re-run need not rebuild ---"
tar czf "$OUT/vllm-cli-gfx1151.tar.gz" -C "$(dirname "$CLI")" "$(basename "$CLI")" 2>/dev/null
ls -la "$OUT/vllm-cli-gfx1151.tar.gz" 2>/dev/null

say "5. ASSERT THE ARTIFACT"
[ -f "$GGUF" ] || die "no $GGUF" 95
GOTG=$(sha256sum "$GGUF"|cut -d' ' -f1); echo "gguf_sha256=$GOTG"
[ "$GOTG" = "$WANT_GGUF" ] || die "gguf sha mismatch, want $WANT_GGUF" 95
echo "shards:"; ls -la /tmp/ckpt-iq1s/

say "6. CLOCK CORRELATION -- so rocprof timestamps can be mapped to the CLI's epoch marks"
python3 - <<'PY'
import time
for _ in range(3):
    a=time.clock_gettime(time.CLOCK_BOOTTIME); b=time.time(); c=time.clock_gettime(time.CLOCK_MONOTONIC)
    print(f"CLOCKPAIR boottime={a:.9f} epoch={b:.9f} monotonic={c:.9f}")
PY

# ---------------------------------------------------------------------------
leg(){ # leg <label> <profiled:0|1>
  local label=$1 prof=$2
  local D=$L/leg-$label; rm -rf "$D"; mkdir -p "$D"
  say "LEG[$label] profiled=$prof  ngen=$NGEN repeat=$REPEAT"
  local PRE=()
  [ "$prof" = "1" ] && PRE=(rocprofv3 --kernel-trace --output-format csv --output-directory "$D/trace" --)
  local S=$(date +%s)
  timeout --foreground 3600 "${PRE[@]}" "$CLI" \
      --model "$GGUF" --device auto --prompt 'The capital of France is' \
      --max-tokens $NGEN --temperature 0 --max-num-seqs 1 --repeat $REPEAT \
      > "$D/stdout.txt" 2> "$D/stderr.txt" </dev/null
  local r=$?
  echo "LEG[$label] rc=$r wall=$(( $(date +%s) - S ))s   <- the RUN's own status, no pipe"
  case $r in
    0) echo "LEG[$label] VERDICT: EXITED 0";;
    124) echo "LEG[$label] VERDICT: TIMED OUT";;
    13[0-9]|1[4-9][0-9]) echo "LEG[$label] VERDICT: SIGNAL $((r-128)) -- a CRASH";;
    *) echo "LEG[$label] VERDICT: exited $r";;
  esac
  echo "LEG[$label] stderr_bytes=$(wc -c < "$D/stderr.txt")  stdout_bytes=$(wc -c < "$D/stdout.txt")"
  echo "--- the per-run timing lines (run 1 is the lazy-staging warmup, DISCARDED) ---"
  grep -a "tok_s=\|generate_start_unix" "$D/stderr.txt" | sed "s/^/$label  /"
  echo "--- completion text ---"; head -c 600 "$D/stdout.txt"; echo
  echo "--- crash signatures ---"
  grep -aiE "illegal memory access|HSA_STATUS_ERROR|Memory access fault|core dumped|aborting" "$D/stderr.txt" | head -4 | sed "s/^/$label  CRASHSIG /"
  if [ "$prof" = "1" ]; then
    echo "--- profiler timestamp-swap warnings (this is #3040's signature) ---"
    echo -n "$label  swap_warnings="; grep -aci "swap\|timestamp" "$D/stderr.txt"
    grep -ai "swap" "$D/stderr.txt" | head -3 | sed "s/^/$label  SWAP /"
    local KT=$(find "$D/trace" -name '*kernel_trace.csv' | head -1)
    if [ -n "$KT" ]; then
      echo "$label  kernel_trace=$KT rows=$(( $(wc -l < "$KT") - 1 )) bytes=$(stat -c%s "$KT")"
      echo "--- top kernel names by DISPATCH COUNT (a count, not a time) ---"
      python3 -c "
import csv,collections,re,sys
rows=list(csv.DictReader(open('$KT')))
c=collections.Counter(r['Kernel_Name'] for r in rows)
for n,k in c.most_common(18):
    print('   %6d  %s'%(k,n[:88]))
d=[int(r['End_Timestamp'])-int(r['Start_Timestamp']) for r in rows]
print('   duration sanity: n=%d neg=%d zero=%d min=%d max=%d'%(len(d),sum(1 for x in d if x<0),sum(1 for x in d if x==0),min(d),max(d)))
"
      gzip -9 "$KT"; cp -f "$KT.gz" "$OUT/leg-$label-kernel_trace.csv.gz"
    else
      echo "$label  NO KERNEL TRACE PRODUCED"; ls -laR "$D/trace" 2>/dev/null | head -20
    fi
  fi
  gzip -9 -c "$D/stderr.txt" > "$OUT/leg-$label-stderr.txt.gz"
  cp -f "$D/stdout.txt" "$OUT/leg-$label-stdout.txt"
  return 0
}

say "7. THE LEGS -- unprofiled first, so the control is not a profiled box's idea of idle"
leg U1 0
leg P1 1
leg U2 0
leg P2 1

say "8. ARCHIVE"
cp -f "$L/cmake.log" "$L/ninja.log" "$L/apt.log" "$OUT/" 2>/dev/null
{ echo "base_sha=$BASE_SHA"; echo "cli_sha256=$(sha256sum "$CLI"|cut -d' ' -f1)";
  echo "gguf_sha256=$GOTG"; echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null)";
  echo "rc_job=${RC_JOB_ID:-UNSET}"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"; } > "$OUT/IDENTITY.txt"
cat "$OUT/IDENTITY.txt"
ls -la "$OUT"
say "DONE $OUT"
