#!/bin/bash
# MODEL-MM-QWEN4-EXP / ISSUE-LOCAL-01M2CJXQMV9R9JGRSKZMW4W21F
# The ROCm HcGroupedNormKernel: baseline, M0 order-gating probe, red-first check,
# the FIX, and the mutation battery. Speed is a SECOND job.
# NEVER `exec >LOG` (rc reaps the job); stream to rc's own stdout.
set -u
FIX=63925eb7f417882c0b682ab850d9135873978fab
BASE=60990ee784101f74f6d1775575e9e89dfb26f73a
KERN=src/vt/rocm/rocm_qwen4_exp.hip
ROOT=/tmp/vllmcpp-hcnorm-$$
OUT=/workspace/q4exp-hcnorm/$(date -u +%Y%m%dT%H%M%SZ)
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
trap 'rm -rf "$ROOT" 2>/dev/null' EXIT
( while sleep 180; do kill -0 $$ 2>/dev/null||exit 0; echo "### hb $(date -u +%H:%M:%S) free=$(df -BG --output=avail /tmp|tail -1)"; done ) & HB=$!
trap 'kill -9 $HB 2>/dev/null; rm -rf "$ROOT" 2>/dev/null' EXIT INT TERM
ulimit -c 0

say IDENTITY; uname -m; nproc; free -g|head -2; df -h /tmp|tail -1
cat /proc/sys/kernel/random/boot_id

say "INSTALL ROCm (pin the vendor repo above noble, or rocm-hip-sdk is unsatisfiable)"
if [ ! -x /opt/rocm/bin/hipcc ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq 2>&1 | tail -2
  apt-get install -y -qq wget gnupg ca-certificates git cmake ninja-build python3 build-essential 2>&1 | tail -3
  mkdir -p /etc/apt/keyrings
  wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg
  UB=$(. /etc/os-release; echo $VERSION_CODENAME)
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/latest $UB main" > /etc/apt/sources.list.d/rocm.list
  printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' > /etc/apt/preferences.d/rocm-radeon-first
  apt-get update -qq 2>&1 | tail -2
  T0=$SECONDS
  apt-get install -y -qq rocm-hip-sdk 2>&1 | tail -5
  echo "rocm-hip-sdk install rc=$? in $((SECONDS-T0))s"
fi
/opt/rocm/bin/hipcc --version 2>&1 | head -3
export LD_LIBRARY_PATH="$(dirname "$(find /opt -name 'libamdhip64.so.7' 2>/dev/null | head -1)")":/opt/rocm/lib
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
rocminfo 2>/dev/null | grep -m2 -E 'gfx|Wavefront' || true

say "CLONE $ROOT"
git clone -q --no-checkout https://github.com/mudler/vllm.cpp "$ROOT" || { echo FATAL clone; exit 90; }
cd "$ROOT" || exit 90
git fetch -q origin "$FIX" "$BASE" 2>&1 | tail -2
git checkout -q "$FIX" || { echo FATAL checkout; exit 91; }
echo "HEAD=$(git rev-parse HEAD)"
git submodule update -q --init --recursive 2>&1 | tail -2

cfg(){ cmake -S "$ROOT" -B "$ROOT/b" -G Ninja -DVLLM_CPP_HIP=ON -DCMAKE_BUILD_TYPE=Release \
        -DROCM_PATH=/opt/rocm -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 2>&1 | tail -4; }
bld(){ ninja -C "$ROOT/b" -j 4 test_backend_cross_device test_qwen4_exp_rocm_reductions 2>&1 | tail -6; }
SEL="$ROOT/scripts/run-doctest-selected.sh"
BINR="$ROOT/b/tests/test_qwen4_exp_rocm_reductions"
BINX="$ROOT/b/tests/test_backend_cross_device"
# ALWAYS report cases AND assertions, and name the binary: this row has shipped
# two selectors that matched nothing and printed Status: SUCCESS! with exit 0.
run(){ local tag="$1" bin="$2" filt="$3"
  say "RUN $tag :: $(basename $bin) -tc='$filt'  md5=$(md5sum $bin|cut -c1-12)"
  "$SEL" "$bin" -tc="$filt" > "$OUT/$tag.log" 2>&1; local rc=$?
  grep -E '^\[MEASURED\]|^\[SKIP\]|test cases:|assertions:|Status:|selected=' "$OUT/$tag.log" | sed "s/^/$tag  /"
  echo "$tag rc=$rc"
  grep -iE 'illegal memory access|HSA_STATUS_ERROR|Memory access fault|core dumped' "$OUT/$tag.log" | head -3 | sed "s/^/$tag  CRASHSIG /"
  return 0; }

mkdir -p "$OUT" || { echo "FATAL cannot write $OUT"; exit 92; }

say "CONFIGURE + BUILD (FIX tree)"; cfg; T0=$SECONDS; bld; echo "build rc=$? in $((SECONDS-T0))s"
[ -x "$BINR" ] || { echo FATAL no rocm reductions binary; exit 93; }

# ── ARM 1: the FIX ──────────────────────────────────────────────────────────
run FIX_battery   "$BINR" '*ROCM W7*'
run FIX_crossdev  "$BINX" '*'
say "FIX DSA control (must be UNMOVED)"; grep -cE 'DSA' "$OUT/FIX_crossdev.log" || true
"$BINX" -tc='*DSA*' > "$OUT/FIX_dsa.log" 2>&1; echo "FIX_dsa rc=$?"
grep -E 'test cases:|assertions:|Status:' "$OUT/FIX_dsa.log" | sed 's/^/FIX_dsa  /'

# ── ARM 2: BASE KERNEL + NEW TESTS ──────────────────────────────────────────
# Red-first: does the new battery convict the pre-change kernel? And M0: does
# ANY committed suite convict a reversed summation order in that kernel?
say "ARM 2 :: restore the BASE kernel under the new tests"
git checkout "$BASE" -- "$KERN"
bld >/dev/null 2>&1; echo "rebuild rc=$?  md5=$(md5sum $BINR|cut -c1-12)"
run BASE_battery  "$BINR" '*ROCM W7*'
run BASE_crossdev "$BINX" '*qwen4_exp*'

say "M0 :: REVERSE the per-group walk in the UNCHANGED kernel"
python3 - "$KERN" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
old="""    double ss = 0.0;
    for (int64_t h = 0; h < H; ++h) {"""
new="""    double ss = 0.0;
    for (int64_t h = H - 1; h >= 0; --h) {"""
assert old in s, "M0 anchor missing"
open(p,'w').write(s.replace(old,new,1)); print("M0 applied")
PY
bld >/dev/null 2>&1; M0MD5=$(md5sum $BINR|cut -c1-12); echo "M0 rebuild rc=$?  md5=$M0MD5"
run M0_crossdev "$BINX" '*qwen4_exp*'
run M0_battery  "$BINR" '*ROCM W7*'
git checkout "$BASE" -- "$KERN"; bld >/dev/null 2>&1
echo "M0 restored md5=$(md5sum $BINR|cut -c1-12) (must differ from $M0MD5)"

# ── ARM 3: the mutation battery on the FIXED kernel ─────────────────────────
git checkout "$FIX" -- "$KERN"
bld >/dev/null 2>&1; FIXMD5=$(md5sum $BINR|cut -c1-12); echo "FIX restored md5=$FIXMD5"
git diff --exit-code >/dev/null 2>&1; echo "tree clean after restore: rc=$?"

mut(){ local tag="$1" py="$2"
  say "MUTATION $tag"
  python3 - "$KERN" <<PY
import sys
p=sys.argv[1]; s=open(p).read()
$py
PY
  [ $? -ne 0 ] && { echo "$tag ANCHOR MISSING -- mutation NOT applied, result VOID"; git checkout "$FIX" -- "$KERN"; return 0; }
  bld >/dev/null 2>&1
  local m=$(md5sum $BINR|cut -c1-12)
  echo "$tag binary md5=$m  (unmutated=$FIXMD5)"
  [ "$m" = "$FIXMD5" ] && echo "$tag BINARY UNCHANGED -- this mutation measured NOTHING"
  run ${tag}_battery "$BINR" '*ROCM W7*'
  git checkout "$FIX" -- "$KERN"
  bld >/dev/null 2>&1
  echo "$tag restored md5=$(md5sum $BINR|cut -c1-12); git diff rc: "; git diff --exit-code -- "$KERN" >/dev/null 2>&1; echo "  $?"
  return 0; }

mut M1 'old="""    if (lane == 0) s_part[wave] = part;"""
new="""    if (lane == 0) s_part[wave] = part;
    if (true) { if (threadIdx.x == 0) s_part[0] = part; }"""
assert old in s
open(p,"w").write(s.replace(old,new,1))'

mut M2 'old="""      const float v = LoadAt(hyper, hyper_tag, base + h);
      part = __fadd_rn(part, __fmul_rn(v, v));
    }"""
new="""      const float v = LoadAt(hyper, hyper_tag, base + h);
      part = __fadd_rn(part, __fmul_rn(v, v));
      h += 1;
    }"""
assert old in s
open(p,"w").write(s.replace(old,new,1))'

mut M3 'old="""    for (int64_t h = static_cast<int64_t>(threadIdx.x); h < H;
         h += static_cast<int64_t>(blockDim.x)) {
      const float v = LoadAt(hyper, hyper_tag, base + h);"""
new="""    for (int64_t h = static_cast<int64_t>(threadIdx.x);
         h + static_cast<int64_t>(blockDim.x) < H;
         h += static_cast<int64_t>(blockDim.x)) {
      const float v = LoadAt(hyper, hyper_tag, base + h);"""
assert old in s
open(p,"w").write(s.replace(old,new,1))'

mut M4 'old="""    for (int64_t h = static_cast<int64_t>(threadIdx.x); h < H;
         h += static_cast<int64_t>(blockDim.x)) {
      const float v = LoadAt(hyper, hyper_tag, base + h);"""
new=old
old2="""  for (int64_t g = static_cast<int64_t>(blockIdx.x); g < total;
       g += static_cast<int64_t>(gridDim.x)) {"""
new2="""  for (int64_t g = static_cast<int64_t>(blockIdx.x); g < total;
       g += total) {"""
assert old2 in s
open(p,"w").write(s.replace(old2,new2,1))'

mut M5 'old="""    const float r = s_r;"""
new="""    const float r = __shfl(s_r, 0);
    (void)0;"""
assert old in s
open(p,"w").write(s.replace(old,new,1))'

mut M6 'old="""    float part = 0.0f;
    for (int64_t h = static_cast<int64_t>(threadIdx.x); h < H;
         h += static_cast<int64_t>(blockDim.x)) {
      const float v = LoadAt(hyper, hyper_tag, base + h);
      part = __fadd_rn(part, __fmul_rn(v, v));
    }"""
new="""    float part = 0.0f;
    { double acc = 0.0;
      for (int64_t h = static_cast<int64_t>(threadIdx.x); h < H;
           h += static_cast<int64_t>(blockDim.x)) {
        const double v = static_cast<double>(LoadAt(hyper, hyper_tag, base + h));
        acc = __dadd_rn(acc, __dmul_rn(v, v));
      }
      part = static_cast<float>(acc); }"""
assert old in s
open(p,"w").write(s.replace(old,new,1))'

say "FINAL restore + re-verify the FIX"
git checkout "$FIX" -- .; bld >/dev/null 2>&1
echo "final md5=$(md5sum $BINR|cut -c1-12) (unmutated=$FIXMD5)"
git status --short | head

say ARCHIVE; ls -la "$OUT"; du -sh "$OUT"
echo "=====HCN_TAR_B64_BEGIN====="; tar czf - -C "$OUT" . | base64 -w 200; echo "=====HCN_TAR_B64_END====="
say "DONE $OUT"
