#!/usr/bin/env bash
# Drive the GLM-5.3 (`glm-dsa`) first load on a leased worker. W7 of
# `.agents/specs/glm-dsa-latest-deepseek.md` §3.7, issue
# https://github.com/mudler/vllm.cpp/issues/2214.
#
# W7 WROTE THIS AND DID NOT RUN IT. Both fleet GPUs were busy for the whole wave
# and the artifact was 43% staged when it ended, so nothing below has been
# executed against the real model. It is committed as the recipe the next wave
# runs, not as evidence of a run — that distinction is the whole reason this
# header exists. Whoever runs it first records what actually happened.
#
# Run it through a LEASE, never over ssh:
#
#   rc run -d dgx:gpu0 --max-runtime 240m --timeout 0 -- \
#       bash /workspace/glm53-dsa-first-load.sh
#
# ─── WHAT IT REFUSES TO DO ───────────────────────────────────────────────────
# Every precondition below is checked and reported rather than assumed, because
# each one has a failure mode that looks like a result:
#
#  * `/workspace` unmounted becomes an empty local directory, and a job that
#    writes into it "succeeds" writing nowhere.
#  * A shard that is short reads as a corrupt model. `GgufFile::Open` catches
#    this (it validates every tensor's span against the data section), but by
#    then you have paid for a lease. The sizes and hashes are checked first.
#  * The binary you built and the binary you ran are not the same thing unless
#    something checks. The identity guard covers the executable AND every shared
#    object beside it: grepping only `vllm-cli` has refused a correct build.
#  * A missing `glm-dsa.attention.indexer.types` is a REFUSAL, not a crash, and
#    the message names it. That is spec O17 and the repair is to the file.
#
# ─── IT IS RESUMABLE ─────────────────────────────────────────────────────────
# `dgx.casa` has crashed roughly hourly under long sequences before (#545), and
# this load is expected to take tens of minutes: the sibling 101 GiB model took
# ~26 minutes on this box and this one is twice the size and streams. Every
# stage writes a stamp under $OUT and re-running skips the stages already
# stamped. Nothing here deletes a stamp; delete one by hand to redo a stage.
set -uo pipefail

WS=${WS:-/workspace}
OUT=${OUT:-$WS/glm53-dsa-w7}
CKPT=${CKPT:-$WS/ckpt/GLM-5.3-UD-IQ1_S}
SHARD1=${SHARD1:-$CKPT/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf}
SRC_TGZ=${SRC_TGZ:-$WS/glm53-dsa-w7-src.tar.gz}
BUILD=${BUILD:-/tmp/glm53-dsa-build}      # NOT on CIFS: it holds no symlink
SRC=${SRC:-/tmp/glm53-dsa-src}
JOBS=${JOBS:-4}                            # unconstrained parallelism has OOM-rebooted this box
PROMPT=${PROMPT:-The capital of France is}
MAX_TOKENS=${MAX_TOKENS:-1}                # a FIRST token; a second needs spec O4
SLOTS=${SLOTS:-4096}                       # §3.3; the floor is 228 towers x 8 = 1824

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
die() { log "REFUSED: $*"; exit 1; }
stamp() { [ -f "$OUT/stamp.$1" ]; }
mark()  { : > "$OUT/stamp.$1"; }

# ── 0. the mount, first, because a failed mount is silent ────────────────────
mountpoint -q "$WS" || grep -qs " $WS " /proc/mounts \
  || die "$WS is not a mount point. A failed mount becomes an empty local dir and this job would 'succeed' writing nowhere."
mkdir -p "$OUT" || die "cannot write to $OUT"
log "workspace OK: $(df -h "$WS" | tail -1)"

# ── 1. the artifact: present, complete, and the bytes we think ───────────────
# The published sizes, at revision 346b3591c7f28d1a23716f97a065ecf12ec14771.
declare -a SIZES=(9428677 49968868928 49717743616 0 0 0)
TOTAL_EXPECT=216715365893
[ -d "$CKPT" ] || die "no checkpoint directory at $CKPT"
n=$(ls -1 "$CKPT"/GLM-5.3-UD-IQ1_S-*-of-00006.gguf 2>/dev/null | wc -l)
[ "$n" -eq 6 ] || die "found $n of 6 shards in $CKPT; the arm is 6 shards and a partial set is not a model"
total=$(du -bc "$CKPT"/GLM-5.3-UD-IQ1_S-*-of-00006.gguf | tail -1 | cut -f1)
[ "$total" -eq "$TOTAL_EXPECT" ] \
  || die "the six shards total $total bytes, expected $TOTAL_EXPECT (201.83 GiB). A short shard reads as a corrupt model."
log "artifact OK: 6 shards, $total bytes"

# Hashes. Shards 1 and 2 were verified at fetch time by the staging script; the
# other four are OWED (spec O7) and this is where they get recorded.
if ! stamp sha256; then
  log "hashing six shards (this reads 201.83 GiB and is the slow precondition)"
  ( cd "$CKPT" && sha256sum GLM-5.3-UD-IQ1_S-*-of-00006.gguf ) > "$OUT/shards.sha256" \
    || die "sha256 failed"
  grep -q ff3adab0853dfb00bdf3889ec3f5556196f56b65783115720d57767bbd760dd9 "$OUT/shards.sha256" \
    || die "shard 1's hash does not match the value recorded at fetch time; this is a different file"
  grep -q 659d04cf4fc0b6026944f34c0b590a635803bff06c1775361e28490db7b168f8 "$OUT/shards.sha256" \
    || die "shard 2's hash does not match the value recorded at fetch time"
  mark sha256
fi
log "shard hashes:"; cat "$OUT/shards.sha256"

# ── 2. the indexer schedule, spec O17 ────────────────────────────────────────
# The published file states none, so the loader refuses it BY NAME. Repair the
# FILE from the model author's own config.json; never the loader (spec D3).
FEED_DIR=$CKPT
if [ -n "${GLM53_CONFIG_JSON:-}" ] && [ -f "$GLM53_CONFIG_JSON" ]; then
  if ! stamp indexer_types; then
    log "writing glm-dsa.attention.indexer.types into a DERIVED copy of shard 1"
    mkdir -p "$OUT/derived"
    # Hard-link the five payload shards; only the metadata shard is rewritten.
    for f in "$CKPT"/GLM-5.3-UD-IQ1_S-*-of-00006.gguf; do
      b=$(basename "$f")
      [ "$b" = "$(basename "$SHARD1")" ] && continue
      ln -f "$f" "$OUT/derived/$b" 2>/dev/null || cp -l "$f" "$OUT/derived/$b" || die "cannot link $b"
    done
    python3 "$SRC/scripts/glm-dsa-write-indexer-types.py" \
      --shard "$SHARD1" --from-config "$GLM53_CONFIG_JSON" \
      --out "$OUT/derived/$(basename "$SHARD1")" --force \
      > "$OUT/indexer_types.log" 2>&1 || { cat "$OUT/indexer_types.log"; die "the metadata repair refused"; }
    cat "$OUT/indexer_types.log"
    mark indexer_types
  fi
  FEED_DIR=$OUT/derived
  log "feeding the DERIVED artifact at $FEED_DIR (spec O17). It is NOT unsloth/GLM-5.3-GGUF's shard 1."
else
  log "GLM53_CONFIG_JSON is unset, so the PUBLISHED file is fed as-is."
  log "EXPECT A REFUSAL naming the absent indexer schedule (spec O17). That is a correct result, not a failure."
fi

# ── 3. source and build. /tmp, not CIFS: /workspace holds no symlink ─────────
if ! stamp build; then
  [ -f "$SRC_TGZ" ] || die "no source archive at $SRC_TGZ (git archive it WITHOUT --prefix=src/, which double-nests the repo's own src/)"
  rm -rf "$SRC" && mkdir -p "$SRC" && tar -xzf "$SRC_TGZ" -C "$SRC" || die "source extract failed"
  [ -f "$SRC/CMakeLists.txt" ] || die "$SRC has no CMakeLists.txt at its root -- the archive was made with a --prefix"
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release > "$OUT/cmake.log" 2>&1 \
    || { tail -30 "$OUT/cmake.log"; die "cmake failed"; }
  # -j 4: unconstrained parallelism has OOM-rebooted this box.
  ninja -C "$BUILD" -j "$JOBS" vllm-cli > "$OUT/build.log" 2>&1 \
    || { tail -40 "$OUT/build.log"; die "build failed"; }
  mark build
fi

# ── 4. the binary identity guard, over the executable AND every .so beside it ─
# Grepping only `vllm-cli` has refused a correct build before, because the
# symbol can land in a shared object next to it.
CLI=$BUILD/vllm-cli
[ -x "$CLI" ] || die "no vllm-cli at $CLI"
SENTINEL=${SENTINEL:-GlmMoeDsaForCausalLM}
found=no
for obj in "$CLI" "$(dirname "$CLI")"/*.so*; do
  [ -e "$obj" ] || continue
  if strings -a "$obj" 2>/dev/null | grep -q "$SENTINEL"; then
    log "identity OK: $SENTINEL present in $(basename "$obj")"
    found=yes
  fi
done
[ "$found" = yes ] || die "$SENTINEL is in neither vllm-cli nor any .so beside it; this build does not contain the change under test"
sha256sum "$CLI" | tee "$OUT/binary.sha256"

# ── 5. the load ──────────────────────────────────────────────────────────────
# Streaming ON, and the slot budget above the model's own working set. The
# loader refuses below 228 towers x 8 experts = 1824 slots rather than degrading
# to a 187 GiB random read per token, which would publish a page-cache number
# under a streaming label.
export VT_MOE_EXPERT_STREAM=1
export VT_MOE_EXPERT_STREAM_SLOTS=$SLOTS
log "VT_MOE_EXPERT_STREAM=1 VT_MOE_EXPERT_STREAM_SLOTS=$SLOTS"

# `/usr/bin/time` does not exist here; sample /proc/<pid>/status instead.
# READ THE CAVEAT BEFORE QUOTING THE NUMBER: while the towers are mmap-resident,
# VmHWM tracks PAGE-CACHE PRESSURE and is NOT a residency measurement. The
# sibling row read 99.47, 85.18 and 72.05 GiB for one model on three boxes. It
# is sampled because it is cheap and bounded, not because it answers spec O9.
"$CLI" --model "$FEED_DIR/$(basename "$SHARD1")" \
       --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" \
       > "$OUT/load.stdout" 2> "$OUT/load.stderr" &
pid=$!
hwm=0
while kill -0 "$pid" 2>/dev/null; do
  v=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null)
  [ -n "${v:-}" ] && [ "$v" -gt "$hwm" ] && hwm=$v
  sleep 5
done
wait "$pid"; rc=$?

log "vllm-cli rc=$rc"
log "VmHWM peak = $hwm kB = $(awk -v k="$hwm" 'BEGIN{printf "%.2f", k/1048576}') GiB  (page-cache-inflated; NOT the resident footprint, spec O9)"
echo "--- stdout ---"; cat "$OUT/load.stdout"
echo "--- stderr (tail) ---"; tail -60 "$OUT/load.stderr"
grep -h '\[expert-stream\]' "$OUT/load.stderr" | tail -5 || true
{ echo "rc=$rc"; echo "vmhwm_kb=$hwm"; echo "feed_dir=$FEED_DIR"; echo "slots=$SLOTS"; } > "$OUT/result.env"

# The honest exits. A refusal is a RESULT here and the message is the evidence;
# it is reported rather than smoothed into a pass.
if [ "$rc" -ne 0 ]; then
  log "THE LOAD DID NOT COMPLETE. The message above is the finding -- record it verbatim."
  log "Expected refusals at this wave: the absent indexer schedule (spec O17) if the published file was fed;"
  log "the forward, which is not implemented (spec O21); and CheckDeviceWeightFit charging all 187.312 GiB"
  log "of towers, because kGlmMoeDsaFactory does not set streams_routed_experts (spec O22)."
fi
exit "$rc"
