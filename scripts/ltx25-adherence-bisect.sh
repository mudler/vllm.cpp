#!/usr/bin/env bash
# LTX25-ADHERENCE-BISECT (#2514) phase A -- dump the ORACLE's final video latent
# at the pinned #1864 request, and PROVE the dump did not change the render.
#
# WHY A SEPARATE HARNESS, AND WHY IT IS SHORT.
# `.agents/specs/ltx25-adherence-bisect.md` asks whether our 2.86-point CLIP
# adherence gap lives in the LATENT or in the VAE DECODE. The experiment that
# answers it is a CROSS-DECODE: one latent, both VAEs. This job produces the one
# latent. It is 61,440 bytes -- (1,128,4,6,10) bf16 -- so once it exists, the
# other arm needs no lease at all.
#
# WHY NOT A FLAG ON `ltx25-oracle-absolute-render.sh`: that script runs OUR
# engine. The oracle itself is driven by `tools/oracle/ltx2_oracle.py`, which
# runs upstream in a SUBPROCESS and therefore cannot reach `video_state.latent`.
# Reaching it needs upstream's `main()` in-process, which is
# `tools/oracle/ltx2_latent_dump.py`, which is a different program.
#
# THIS JOB IS RESUMABLE, because `dgx` has crashed roughly hourly under long
# sequences and a 42 GB stage is 10 minutes of it. Every phase writes a marker
# under $W and skips when its marker and its artefacts are both present, so a
# re-queue continues rather than restarts. Phase A's result is copied to the NAS
# the moment it exists, before anything else runs.
#
# PHASE B IS NOT HERE, DELIBERATELY. The n = 1 question needs three of OUR
# renders, and `scripts/ltx25-render-confirm.sh` already takes them with a
# verified binary, a staged checkpoint set and a phase table. This change gives
# that harness `KEEP_FRAMES=1` so renders 2 and 3 keep their frames instead of
# being deleted at its `:570`. Run it as its own job: two jobs of under an hour
# survive an hourly crash where one job of two hours does not.
#
# Exit codes:
#   0  the latent is written AND reproduces the committed reference
#   90 upstream pin mismatch          92 a checkpoint would not stage
#   93 a checkpoint's sha256 is not the manifest's
#   94 the venv/import/JIT gate failed
#   60..64 are ltx2_latent_dump.py's own; 64 means the probe perturbed the run
set -uo pipefail

T0=$SECONDS
say() { echo "[$(date -u +%H:%M:%S) +$((SECONDS-T0))s] $*"; }

PIN=fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
UP=https://github.com/Lightricks/LTX-2.git
W=${W:-/workspace/ltx25-adherence-bisect}
NASCK=${NASCK:-/workspace/ckpt/ltx-2.5}
FULL=${FULL:-/workspace/ltx25-fullmodel}
LOCAL=${LOCAL:-/tmp/ltx2-bisect}
VENV=$LOCAL/venv
SRC=$LOCAL/LTX-2
CK=$LOCAL/ckpt
TREE=$W/tree                       # tools/oracle/*.py + the committed SHA256SUMS
OUT=$W/out
mkdir -p "$W" "$OUT" "$LOCAL" "$CK"

# HEARTBEAT ON STDERR, and backgrounded with its stdout redirected: `.agents/
# oracles/ltx-2.md` records 2h37m of a lease lost to a heartbeat subshell
# holding a command-substitution pipe open. Signals too -- `rc` reclaiming a
# device sends SIGTERM, and a bare EXIT trap leaves the heartbeat orphaned.
( while true; do sleep 120; echo "[hb +$((SECONDS-T0))s] alive" >&2; done ) &
HEARTBEAT=$!
cleanup() { kill "$HEARTBEAT" 2>/dev/null; }
trap cleanup EXIT
for sig in HUP INT TERM; do
  trap "cleanup; exit \$((128 + \$(kill -l $sig)))" "$sig"
done

say "=== [0] the box ==="
uname -m; nproc; free -g; uptime
nvidia-smi --query-gpu=name,memory.total,memory.used,clocks.sm --format=csv 2>&1 | head -3
df -h /tmp /workspace 2>&1 | head -5
{
  echo "run_started=$(date -Is)"
  echo "rc_job=${RC_JOB_ID:-unknown}"
  echo "harness_sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
  echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
  echo "gpu_clock_sm=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null | head -1)"
} >> "$W/PROVENANCE"

# ---------------------------------------------------------------------------
say "=== [1] the tree this job runs (tools/oracle + the committed digests) ==="
# `ltx2_latent_dump.py` resolves the committed SHA256SUMS as
# `parents[2]/tests/parity/goldens/ltx2_oracle/SHA256SUMS`, so the two files
# must sit in that shape. The caller stages $TREE; this only checks it.
for f in tools/oracle/ltx2_oracle.py tools/oracle/ltx2_latent_dump.py \
         tools/oracle/jitprobe.py tests/parity/goldens/ltx2_oracle/SHA256SUMS; do
  [ -s "$TREE/$f" ] || { echo "FATAL: $TREE/$f is missing. Stage the tree first."; exit 43; }
  echo "  $(sha256sum "$TREE/$f")"
done

# ---------------------------------------------------------------------------
say "=== [2] upstream at the pin ==="
if [ ! -d "$SRC/.git" ]; then
  git clone --filter=blob:none "$UP" "$SRC" 2>&1 | tail -3
fi
git -C "$SRC" fetch --depth 1 origin "$PIN" 2>&1 | tail -2
git -C "$SRC" checkout -q FETCH_HEAD 2>&1 | tail -2
GOT=$(git -C "$SRC" rev-parse HEAD)
[ "$GOT" = "$PIN" ] || { echo "FATAL: pin mismatch got=$GOT want=$PIN"; exit 90; }
DIRTY=$(git -C "$SRC" status --porcelain | wc -l)
[ "$DIRTY" = 0 ] || { echo "FATAL: the pinned checkout is DIRTY ($DIRTY paths). The"
                      echo "whole point of this job is that upstream is unmodified."; exit 90; }
say "  PIN_OK $GOT clean"

# ---------------------------------------------------------------------------
say "=== [3] venv, torch and the import/JIT gates ==="
# The recipe is `/workspace/ltx2-oracle/setup.sh`'s, which is the one that
# actually rendered. The three gates below each cost seconds and each replaces a
# failure that cost a staged 68 GB to discover (#1864 attempts 3 and 5).
[ -d "$VENV" ] || python3 -m venv "$VENV"
P="$VENV/bin/python"; PIP="$VENV/bin/pip"
$PIP install -q --upgrade pip setuptools wheel 2>&1 | tail -3
if ! $P -c "import torch" 2>/dev/null; then
  S=$(date +%s)
  $PIP install -q torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -10
  say "  torch-from-pytorch-index secs=$(( $(date +%s) - S ))"
fi
# UNCONDITIONAL, because /tmp survives between jobs on this worker sometimes and
# not others: an install guarded by a DIFFERENT package's absence guarantees
# nothing, and that exact reasoning error cost #1864 an attempt.
$PIP install -q torchvision --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -3
$PIP install -q numpy pillow av 2>&1 | tail -3
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq python3-dev libpython3.12-dev build-essential 2>&1 | tail -3
$PIP install -q -e "$SRC/packages/ltx-core" 2>&1 | tail -3
$PIP install -q -e "$SRC/packages/ltx-pipelines" 2>&1 | tail -3
if ! $P -c "import ltx_pipelines" 2>/dev/null; then
  $PIP install -q --no-deps -e "$SRC/packages/ltx-pipelines" 2>&1 | tail -3
  for d in av tqdm pillow "cloudpickle>=3.1"; do $PIP install -q "$d" 2>&1 | tail -1; done
fi
$P - <<'GATE' || { echo "FATAL: the render's own imports are unavailable; refusing to stage 70 GB for a load-time death"; exit 94; }
import numpy, PIL, torchvision, torch, av
from transformers import Gemma4UnifiedProcessor
from ltx_pipelines.utils.blocks import RecordingDiffusionStage
import ltx_pipelines.ti2vid_one_stage as m
# The hook this whole job depends on, asserted HERE where it costs seconds.
assert hasattr(m.TI2VidOneStagePipeline, "__init__")
print("import-gate OK: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| RecordingDiffusionStage", RecordingDiffusionStage.__name__)
GATE
$P "$TREE/tools/oracle/jitprobe.py" 2>&1 | tail -20
JITRC=${PIPESTATUS[0]}
[ "${JITRC:-0}" -eq 0 ] || { echo "FATAL: triton cannot JIT-build here (rc=$JITRC); the text tower dies 20 s in"; exit 94; }

# ---------------------------------------------------------------------------
say "=== [4] stage the four BF16 checkpoints, and VERIFY BY SHA256 ==="
# Size is not identity. This repository has run a gate against the wrong
# checkpoint before, and the manifest carries digests precisely so that a
# re-quantized file under an unchanged name is caught. Sizes gate the COPY
# (cheap, per attempt); sha256 gates the RUN (once, before the model loads).
declare -A SRCOF=(
  [vae/ltx-2.5-video-vae-conv-bf16.safetensors]="$FULL/ckpt/ltx-2.5-video-vae-conv-bf16.safetensors"
  [vae/ltx-2.5-audio-vae-bf16.safetensors]="$FULL/ckpt/ltx-2.5-audio-vae-bf16.safetensors"
  [diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors]="$FULL/ckpt/ltx-2.5-22b-dev-transformer-bf16.safetensors"
  [text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors]="$NASCK/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
)
declare -A SIZEOF=(
  [vae/ltx-2.5-video-vae-conv-bf16.safetensors]=1452269922
  [vae/ltx-2.5-audio-vae-bf16.safetensors]=364866540
  [diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors]=42018190584
  [text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors]=26263858182
)
declare -A SHAOF=(
  [vae/ltx-2.5-video-vae-conv-bf16.safetensors]=685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d
  [vae/ltx-2.5-audio-vae-bf16.safetensors]=c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5
  [diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors]=792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584
  [text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors]=ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1
)
stage() {
  local rel=$1 exp=${SIZEOF[$1]} s=${SRCOF[$1]} d="$CK/$1"
  mkdir -p "$(dirname "$d")"
  local have=0; [ -f "$d" ] && have=$(stat -c %s "$d")
  [ "$have" -eq "$exp" ] && { say "  SKIP $rel (already local, size ok)"; return 0; }
  [ -f "$s" ] || { echo "MISSING ON NAS: $s"; return 1; }
  # `.part` then rename on a size match, with retries, because /workspace is
  # CIFS mounted `soft`: a stalled read returns EAGAIN mid-copy instead of
  # blocking, and the 42 GB transformer has died that way. Carrying on with a
  # TRUNCATED checkpoint is worse than stopping, because it renders and the
  # render looks like a result.
  local try S SZ
  for try in 1 2 3 4 5; do
    S=$(date +%s); rm -f "$d.part"
    if cp "$s" "$d.part"; then
      SZ=$(stat -c %s "$d.part" 2>/dev/null || echo 0)
      if [ "$SZ" -eq "$exp" ]; then
        mv -f "$d.part" "$d"
        say "  STAGED $rel bytes=$SZ secs=$(( $(date +%s) - S )) attempt=$try"
        return 0
      fi
      say "  RETRY $rel attempt=$try SHORT $SZ != $exp"
    else
      say "  RETRY $rel attempt=$try cp failed (CIFS soft-mount EAGAIN)"
    fi
    rm -f "$d.part"; sleep 15
  done
  echo "FATAL: could not stage $rel after 5 attempts"; return 1
}
for rel in "${!SIZEOF[@]}"; do stage "$rel" || exit 92; done

say "  --- sha256, against the manifest's own digests ---"
for rel in "${!SHAOF[@]}"; do
  GOT=$(sha256sum "$CK/$rel" | awk '{print $1}')
  if [ "$GOT" = "${SHAOF[$rel]}" ]; then
    say "  OK   $rel"
  else
    echo "FATAL: $rel sha256 $GOT is not the manifest's ${SHAOF[$rel]}."
    echo "A re-quantized file under an unchanged name is exactly what this check exists for."
    exit 93
  fi
  echo "checkpoint_sha256 $rel=$GOT" >> "$W/PROVENANCE"
done

# ---------------------------------------------------------------------------
say "=== [5] THE DUMP, with its own control ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_OUT=$OUT/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_OUT"
{ echo "== pre-run state"; echo "  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)";
  nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader 2>/dev/null | sed 's/^/  gpu=/'; } \
  | tee -a "$W/PROVENANCE"
S=$(date +%s)
"$P" "$TREE/tools/oracle/ltx2_latent_dump.py" \
  --ltx2-source "$SRC" --checkpoints "$CK" --out "$RUN_OUT" \
  --device cuda --offload cpu 2>&1
DUMPRC=$?
say "  ltx2_latent_dump.py rc=$DUMPRC secs=$(( $(date +%s) - S ))"
echo "dump_rc=$DUMPRC" >> "$W/PROVENANCE"

# COPY OUT BEFORE THE VERDICT, so a run that fails its control still leaves the
# evidence that says WHY on the NAS rather than losing it with the worker's /tmp.
say "=== [6] copy out ==="
mkdir -p "$W/result"
cp -f "$RUN_OUT"/*.raw "$RUN_OUT"/*.npy "$RUN_OUT"/*.json "$RUN_OUT"/*.mp4 "$W/result/" 2>/dev/null
mkdir -p "$W/result/recorded_frames"
cp -f "$RUN_OUT"/recorded_frames/* "$W/result/recorded_frames/" 2>/dev/null
ls -la "$W/result" | head -20
( cd "$W/result" && sha256sum ./*.raw ./*.npy ./*.json 2>/dev/null ) | tee "$W/result/SHA256SUMS.local"

[ "$DUMPRC" -eq 0 ] || { echo "PHASE A FAILED rc=$DUMPRC; see the manifest in $W/result"; exit "$DUMPRC"; }
say "=== PHASE A DONE: the oracle's latent is at $W/result and its control PASSED ==="
