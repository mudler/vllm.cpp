#!/usr/bin/env bash
# #2944 -- the vllm.cpp arm of the #2921 gfx1151 survey, MEASURED ON MAIN'S HEAD
# with the revision ASSERTED rather than printed.
#
# WHY THIS RUN EXISTS. The published survey (#2940) reports our arm as
# FAULTED, 0 of 4, rc 139. #2933 established the cause: `survey.sh` asserted the
# llama.cpp source manifest and refused to run when it moved, but for our own arm
# it did `[ -x "$VC" ]` and PRINTED the revision. It never built, never fetched
# and never asserted, so it inherited #2511's staging directory and measured that
# campaign's PRE-FIX build (11fed3ba5, an ancestor of the fix 6b97a6800).
#
# So the published fault rate is #2511 reproducing. Whether the CURRENT head
# completes this workload on gfx1151 has never been measured. This job measures
# it, and its whole point is that the assertion below is fail-closed.
#
# WHAT IS NOT CLAIMED: this is a SURVEY arm, not a parity result. TOKEN_GATE=FAIL
# on this path and no deterministic denominator exists on it. The correctness
# state travels with any number this produces.
set -uo pipefail

TAG=strixarm2944
W=/workspace/strix-arm-2933
OUT=$W/out
LOCAL=/tmp/$TAG
SRC=$LOCAL/src
BLD=$LOCAL/build
L=$LOCAL/log
LOUT=$L/legs

# ---- THE DECLARED PIN. Every one of these is asserted, none is derived. ----
WANT_REV=c796fea41f74fe90b8cf78190eeb2a5b4c977449
WANT_TREE=018178f3bf1fca3983d9bc0cfd42f5ea4bf130b1
BUNDLE_SHA=044dec44ebd8f153f25e13bfbeffd45ca7ba33d6342bdc99fdc63e348b6ca2b8
# The exact binary #2933 names as the one the survey actually measured. A build
# that reproduces this hash IS the pre-fix build, whatever the source says.
FORBIDDEN_CLI_SHA=a703b83dd8954ba6dd3cbe82efcd38083c1d55492bbbaecf5c406f7c6efd646f
# Present in the library only since 6b97a6800 narrowed managed allocation.
MARK_FIX='cannot take a recoverable page fault'

GGUF_NAME=Qwen3.8-27B-Q4_K_M.gguf
GGUF_NAS=/workspace/ckpt/qwen38-27b-q4km/$GGUF_NAME
GGUF=$LOCAL/models/$GGUF_NAME
GGUF_SIZE=17106775008
GGUF_SHA=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
SIBLINGS=(/tmp/rocm-strix-q4k/models/$GGUF_NAME /tmp/rocm-strix-managed/models/$GGUF_NAME)

IMG=vllmcpp-rocm-build:7.2.4
P=(podman --storage-driver=vfs --root /tmp/podman-pr66-root-vfs --runroot /tmp/podman-pr66-run-vfs)

# ---- THE DESIGN. N comes from HERE and is handed to the fold. Never tallied. ----
ROUNDS=4          # legs per token count
REPEAT=4          # completions per leg; completion 1 is COLD and is discarded
COLD_RUNS=1
NGEN_A=64         # the survey's own token count
NGEN_B=128        # the second count, so a decode slope can be DERIVED
PROMPT='The capital of France is'
LEG_TIMEOUT=20m
DESIGN_LEGS=$(( ROUNDS * 2 ))

T0=$(date +%s)
DEADLINE_BUILD=$(( T0 + 60*60 ))
DEADLINE_LEGS=$(( T0 + 200*60 ))

mkdir -p "$OUT" "$LOUT" "$L" "$LOCAL/models" || exit 90
exec > >(tee "$L/job.log") 2>&1
( while true; do cp -f "$L/job.log" "$OUT/job.log" 2>/dev/null; sleep 20; done ) &
SYNC=$!
( while true; do echo "[hb] $(date -u +%H:%M:%S)"; sleep 120; done ) &
HB=$!
trap 'kill $SYNC $HB 2>/dev/null; cp -f "$L/job.log" "$OUT/job.log" 2>/dev/null' EXIT

die() { echo "FATAL: $*"; echo "JOB_VERDICT=FAIL"; exit "${2:-1}"; }
say() { echo; echo "===== $* ====="; date -u +%FT%TZ; }

say "0. worker identity, inherited environment, contention state"
hostname; uname -a; echo "nproc=$(nproc)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "uptime_s=$(cut -d' ' -f1 /proc/uptime)"
echo "rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || echo NONE)"
free -g | head -3; df -h / /tmp /workspace | tail -4
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-UNSET}   <- must read UNSET"
[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ] || die "HSA_OVERRIDE_GFX_VERSION is set; this would measure a different device" 91
if env | grep -qE '^(HSA_|ROCR_|PYTORCH_|HIP_|GGML_|VT_)'; then
  env | grep -E '^(HSA_|ROCR_|PYTORCH_|HIP_|GGML_|VT_)' | sed 's/^/inherited_env /'
  die "an HSA_/ROCR_/PYTORCH_/HIP_/GGML_/VT_ variable was inherited into this job" 91
fi
echo "inherited_env NONE"
echo "--- who else is on this box, as this job sees it ---"
ps -eo pid,etimes,comm --sort=-etimes 2>/dev/null | head -8

say "1. the artifact, verified ON the worker before any timing"
sizeof() { stat -c %s "$1" 2>/dev/null || echo 0; }
have_it() { [ "$(sizeof "$1")" = "$GGUF_SIZE" ]; }
if have_it "$GGUF"; then
  echo "already staged worker-locally at $GGUF"
else
  for s in "${SIBLINGS[@]}"; do
    if have_it "$s"; then
      echo "hard-linking a sibling job's staged copy: $s"
      ln -f "$s" "$GGUF" 2>/dev/null || cp -f "$s" "$GGUF"
      break
    fi
  done
fi
if ! have_it "$GGUF"; then
  # A single cp off CIFS lost a whole lease once with a transient read error.
  for a in 1 2 3; do
    echo "staging from the share, attempt $a of 3"
    cp -f "$GGUF_NAS" "$GGUF" && break
    echo "attempt $a failed (local size $(sizeof "$GGUF") of $GGUF_SIZE)"; sleep 30
  done
fi
have_it "$GGUF" || die "stage failed, local size $(sizeof "$GGUF") of $GGUF_SIZE" 20
GOT=$(sha256sum "$GGUF" | cut -d' ' -f1)
echo "gguf_bytes=$(sizeof "$GGUF") gguf_sha256=$GOT"
[ "$GOT" = "$GGUF_SHA" ] || die "artifact sha256 mismatch on the worker" 21

say "2. the tree under test -- ASSERTED, three ways, fail-closed"
printf 'bundle '; sha256sum "$W/repo.bundle"
BGOT=$(sha256sum "$W/repo.bundle" | cut -d' ' -f1)
[ "$BGOT" = "$BUNDLE_SHA" ] || die "the bundle on the share is not the declared bytes" 22
# The clone is UNCONDITIONAL. #2933 happened because a staging directory was
# reused; a tree that is deleted cannot be inherited.
rm -rf "$SRC"
git clone -q "$W/repo.bundle" "$SRC" || die "clone from the bundle failed" 23
git -C "$SRC" checkout -q "$WANT_REV" 2>/dev/null || true
GOT_REV=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo NONE)
GOT_TREE=$(git -C "$SRC" rev-parse 'HEAD^{tree}' 2>/dev/null || echo NONE)
GOT_DIRT=$(git -C "$SRC" status --porcelain 2>/dev/null | wc -l)
echo "want_revision=$WANT_REV"
echo "got_revision=$GOT_REV"
echo "want_tree=$WANT_TREE"
echo "got_tree=$GOT_TREE"
echo "worktree_modified_files=$GOT_DIRT"
[ "$GOT_REV" = "$WANT_REV" ] || die "REVISION ASSERTION FAILED: got $GOT_REV, want $WANT_REV" 24
[ "$GOT_TREE" = "$WANT_TREE" ] || die "TREE ASSERTION FAILED: got $GOT_TREE, want $WANT_TREE" 24
[ "$GOT_DIRT" = "0" ] || die "the checked-out worktree is modified; this is not the declared tree" 24
grep -q "$MARK_FIX" "$SRC/include/vt/rocm/rocm_arch.h" \
  || die "the checked-out SOURCE does not carry the #2511 narrowing" 24
echo "REVISION_ASSERTION=OK  TREE_ASSERTION=OK  CLEAN=OK  SOURCE_CARRIES_FIX=OK"

say "3. build -- from a build directory that did not exist a moment ago"
[ "$(date +%s)" -lt "$DEADLINE_BUILD" ] || die "past the build deadline before starting" 25
# A removed build directory cannot be stale. That is the second half of #2933:
# the survey's source tree was not even wrong, the BUILD it ran was inherited.
rm -rf "$BLD"
BUILD_START=$(date +%s)
echo "build_start_unix=$BUILD_START"
ROUTE=UNSET
if command -v podman >/dev/null 2>&1 && "${P[@]}" image exists "$IMG" 2>/dev/null; then
  ROUTE=podman
else
  ROUTE=native
fi
echo "BUILD_ROUTE=$ROUTE"

if [ "$ROUTE" = podman ]; then
  # The route #2511 and the survey both used, so the toolchain is the one the
  # measurement being superseded was built with.
  "${P[@]}" run --rm --entrypoint bash --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$LOCAL:$LOCAL:rw" -v /workspace/ccache:/workspace/ccache:rw \
    "$IMG" -lc "
      set -uo pipefail
      export CCACHE_DIR=/workspace/ccache
      mkdir -p \"\$CCACHE_DIR\" 2>/dev/null || true
      CCARGS=()
      if command -v ccache >/dev/null 2>&1; then
        CCARGS=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache)
        echo \"ccache=\$(ccache --version | head -1)\"
      else
        echo 'ccache=UNAVAILABLE (building without it)'
      fi
      cmake -S '$SRC' -B '$BLD' \
        -DCMAKE_BUILD_TYPE=Release \
        -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=ON \
        -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 \
        -DROCM_PATH=/opt/rocm \
        -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.4 \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/rocm/lib \
        \"\${CCARGS[@]}\" \
        -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_SERVER=OFF \
        -DVLLM_CPP_BUILD_EXAMPLES=ON -DVLLM_CPP_BUILD_TESTS=OFF \
        || { echo CONFIGURE_FAILED; exit 10; }
      # -j 4: unconstrained parallelism has OOM-rebooted a fleet box.
      cmake --build '$BLD' -j 4 --target vllm-cli || { echo BUILD_FAILED; exit 11; }
      ccache -s 2>/dev/null | head -6
    " > "$L/build.log" 2>&1
  BRC=$?
else
  export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
  export DEBIAN_FRONTEND=noninteractive
  # The native route's known failure is a MISSING dev symlink: cmake's
  # find_library(amdhip64) misses, and vllm-cli fails at link with ~70
  # undefined hip symbols. Repair it, and RECORD the repair.
  if [ ! -e /opt/rocm/lib/libamdhip64.so ]; then
    apt-get install -y -qq --no-install-recommends hip-runtime-amd 2>&1 | tail -2 || true
    if [ ! -e /opt/rocm/lib/libamdhip64.so ]; then
      REAL=$(ls /opt/rocm/lib/libamdhip64.so.* 2>/dev/null | head -1)
      [ -n "$REAL" ] || die "no libamdhip64 of any kind on this node" 88
      ln -sf "$REAL" /opt/rocm/lib/libamdhip64.so
      echo "ENVIRONMENT REPAIR: symlinked /opt/rocm/lib/libamdhip64.so -> $REAL"
    fi
  fi
  if [ ! -d /opt/rocm/lib/cmake/hipblaslt ]; then
    apt-get update -qq && apt-get install -y -qq --no-install-recommends rocm-libs 2>&1 | tail -2 || true
  fi
  # NO ccache on the native route: three runs on this board failed to LINK
  # vllm-cli with it and a ccache-free build of identical source linked rc=0
  # (#2506). The podman route above keeps it, because that is where it works.
  unset CCACHE_DIR CCACHE_REMOTE_STORAGE
  ( cmake -S "$SRC" -B "$BLD" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=ON \
      -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 \
      -DROCM_PATH=/opt/rocm \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_SERVER=OFF \
      -DVLLM_CPP_BUILD_EXAMPLES=ON -DVLLM_CPP_BUILD_TESTS=OFF \
      && cmake --build "$BLD" -j 4 --target vllm-cli ) > "$L/build.log" 2>&1
  BRC=$?
fi
echo "build_rc=$BRC elapsed_s=$(( $(date +%s) - BUILD_START ))"
grep -nE 'ROCm backend|ENABLED for arch' "$L/build.log" | head -4
tail -30 "$L/build.log"
cp -f "$L/build.log" "$OUT/build.log"
[ "$BRC" = 0 ] || die "the build FAILED; no leg of this job may be read as an engine verdict" 25

say "4. the BUILT BYTES -- asserted, not printed"
CLI=$(find "$BLD" -name vllm-cli -type f -perm -u+x 2>/dev/null | head -1)
SO=$(find "$BLD" -maxdepth 2 -name 'libvllm.so*' -type f 2>/dev/null | LC_ALL=C sort | head -1)
[ -n "$CLI" ] || die "the build reported rc=0 and produced no vllm-cli; a missing binary is not a measurement" 26
[ -n "$SO" ]  || die "no libvllm.so in the build tree" 26
echo "CLI=$CLI"; echo "SO=$SO"
printf 'built_vllm_cli '; sha256sum "$CLI"
printf 'built_libvllm '; sha256sum "$SO"
CLI_SHA=$(sha256sum "$CLI" | cut -d' ' -f1)
echo "note: the kernels live in libvllm.so; vllm-cli is a thin client and its hash alone proves little"
# Mtimes: the binaries must post-date the moment this job emptied the directory.
for b in "$CLI" "$SO"; do
  M=$(stat -c %Y "$b")
  echo "mtime_check $b mtime=$M build_start=$BUILD_START"
  [ "$M" -ge "$BUILD_START" ] || die "$b predates this job's build; it was inherited, which is exactly #2933" 27
done
# `grep -a` reads the binary directly; `strings` is binutils and this image does
# not guarantee it -- an instrument whose failure would look like a result.
grep -a -q "$MARK_FIX" "$SO" || die "the NARROWING is absent from the library under test" 28
[ "$CLI_SHA" != "$FORBIDDEN_CLI_SHA" ] || die "this is byte-for-byte the PRE-FIX binary #2933 names" 29
echo "BUILT_BYTES_ASSERTION=OK (post-dates the empty build dir, carries the narrowing, is not the pre-fix binary)"

say "5. the legs: $ROUNDS rounds x 2 token counts = $DESIGN_LEGS legs BY DESIGN"
echo "design: ROUNDS=$ROUNDS REPEAT=$REPEAT COLD_RUNS=$COLD_RUNS NGEN_A=$NGEN_A NGEN_B=$NGEN_B"
echo "design_legs=$DESIGN_LEGS  (the fold is told this; it counts nothing)"
echo "the order of the two token counts rotates by round, so session drift is not read as a difference"

run_leg() {   # run_leg <ngen> <round>
  local ngen=$1 round=$2 tag rc st el cls clk cpid
  tag="n${ngen}-r${round}"
  clk="$LOUT/clock-$tag.jsonl"; rm -f "$clk"
  python3 "$W/amd_clock_sample.py" --output "$clk" --interval 0.25 &
  cpid=$!
  sleep 1
  echo "--- leg $tag start $(date -u +%FT%TZ) ---"
  st=$(date +%s)
  if [ "$ROUTE" = podman ]; then
    timeout --foreground "$LEG_TIMEOUT" "${P[@]}" run --rm --entrypoint "$CLI" \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      -e "LD_LIBRARY_PATH=$(dirname "$SO"):/opt/rocm/lib" -e VT_OP_PROVIDER_STATS=1 \
      -v "$LOCAL:$LOCAL:rw" "$IMG" \
      --model "$GGUF" --prompt "$PROMPT" --max-tokens "$ngen" \
      --temperature 0 --repeat "$REPEAT" --max-num-seqs 1 \
      > "$LOUT/$tag.out" 2> "$LOUT/$tag.err" < /dev/null
    rc=$?
  else
    ( cd "$BLD" && LD_LIBRARY_PATH="$(dirname "$SO"):/opt/rocm/lib" VT_OP_PROVIDER_STATS=1 \
        timeout --foreground "$LEG_TIMEOUT" "$CLI" \
        --model "$GGUF" --prompt "$PROMPT" --max-tokens "$ngen" \
        --temperature 0 --repeat "$REPEAT" --max-num-seqs 1 ) \
      > "$LOUT/$tag.out" 2> "$LOUT/$tag.err" < /dev/null
    rc=$?
  fi
  el=$(( $(date +%s) - st ))
  kill -TERM "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
  echo "$rc" > "$OUT/$tag.rc"
  # ORDER MATTERS, and it has to be FIRST MATCH WINS. A refusal is not a board
  # fault and must never be reported as one. The reference tier is withdrawn on
  # this board since 6b97a6800, so `no kernel for op` is the specific failure
  # this configuration could produce.
  #
  # This was a chain of independent `&& cls=...` assignments, in which the LAST
  # match won, so `GPU Hang` silently OVERRODE `OPREFUSED` -- the exact inversion
  # the comment above forbids. It fires on the case at hand: the #2933 logs carry
  # `Memory access fault` AND `GPU Hang` on the same leg. It also disagreed with
  # fold.py, which returns on first match, so the two instruments would have put
  # different classes on identical bytes.
  if   grep -qa 'no kernel for op'    "$LOUT/$tag.err"; then cls=OPREFUSED
  elif grep -qa 'Memory access fault' "$LOUT/$tag.err"; then cls=MEMFAULT
  elif grep -qa 'GPU Hang'            "$LOUT/$tag.err"; then cls=GPUHANG
  elif [ "$rc" = 124 ];                                 then cls=TIMEOUT
  elif [ "$rc" != 0 ];                                  then cls="OTHER_rc$rc"
  else                                                       cls=OK
  fi
  # Both spellings: the banner says `reference-tier`, the refusal says
  # `reference tier`, and a grep for one reads 0 over the other.
  local reftier; reftier=$(grep -aEc 'reference[- ]tier' "$LOUT/$tag.err")
  local samples; samples=$(wc -l < "$clk" 2>/dev/null || echo 0)
  printf 'LEG %s ngen=%s repeat=%s rc=%s class=%s secs=%s clock_samples=%s reftier_lines=%s\n' \
    "$tag" "$ngen" "$REPEAT" "$rc" "$cls" "$el" "$samples" "$reftier"
  grep -E 'run=[0-9]+/[0-9]+ ' "$LOUT/$tag.err" | sed "s/^/$tag /"
  grep -iE 'GPU Hang|HW Exception|Memory access fault|no kernel for op' "$LOUT/$tag.err" | head -3 | sed "s/^/$tag /"
  cp -f "$LOUT/$tag.err" "$OUT/$tag.err" 2>/dev/null
  cp -f "$LOUT/$tag.out" "$OUT/$tag.out" 2>/dev/null
  cp -f "$clk" "$OUT/clock-$tag.jsonl" 2>/dev/null
}

for r in $(seq 1 "$ROUNDS"); do
  if [ "$(date +%s)" -gt "$DEADLINE_LEGS" ]; then echo "LEG DEADLINE reached before round $r"; break; fi
  echo; echo "--- round $r of $ROUNDS ---"
  if [ $(( r % 2 )) = 1 ]; then
    run_leg "$NGEN_A" "$r"; run_leg "$NGEN_B" "$r"
  else
    run_leg "$NGEN_B" "$r"; run_leg "$NGEN_A" "$r"
  fi
  cp -f "$L/job.log" "$OUT/job.log" 2>/dev/null
done

say "6. the fold"
cp -f "$L/job.log" "$OUT/job.log" 2>/dev/null
python3 "$W/fold.py" --evidence "$OUT" --rounds "$ROUNDS" \
  --ngen-a "$NGEN_A" --ngen-b "$NGEN_B" --repeat "$REPEAT" --cold-runs "$COLD_RUNS" \
  > "$OUT/RESULT.json" 2> "$OUT/fold.err"
FRC=$?
cat "$OUT/RESULT.json"; echo "fold_rc=$FRC"; cat "$OUT/fold.err"

say "7. what a reader must carry away with any number above"
cat <<'CLOSING'
TOKEN_GATE=FAIL. This is a SURVEY arm, not a parity result.
  vllm.cpp ROCm arm vs llama.cpp b10451   3 of 6 prompts divergent
  vllm.cpp ROCm arm vs vLLM (compiled)    5 of 6 prompts divergent
Every divergence is a near-tie at about 0.125 nats, one bf16 ULP. Those counts
are CONSTANTS carried from earlier evidence; no leg of this job re-measured them.
NO DETERMINISTIC DENOMINATOR EXISTS ON THIS PATH.
The `vllm-cli` tok_s figure is WHOLE COMPLETION, not decode. It may not be
divided by `llama-bench -p 0`, which is pure decode.
CLOSING
echo "OUT=$OUT"
echo "=== ARM JOB DONE ==="
