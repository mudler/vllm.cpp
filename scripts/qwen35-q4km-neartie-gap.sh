#!/bin/bash
# The pinned re-derivation path for the qwen3.5 Q4_K_M Tenstorrent capture
# pair's near-tie gap golden (spec .agents/specs/tenstorrent-keepquant.md,
# Evidence). Teacher-forces the dequantized artifact through transformers
# bf16 on OUR captured ids and rewrites the canonical golden pair.
#
# Why this wrapper exists: the 2026-09-12 off-recipe incident. An ad-hoc
# re-derivation produced f32-grain gap values (62/313/562 mnats — impossible
# under the pinned bf16 recipe, whose gaps sit on the bf16 logit grid) that
# manufactured four band violations and nearly drove a widening of
# kNearTieMnats. The pinned path must be the convenient path, and it must
# assert its own environment, so a silent env drift fails loudly instead of
# writing a plausible-looking golden.
#
# Changing VENV, MODEL_DIR, or the dtype here is an exception: argue for it
# in the commit message.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GDIR="${GDIR:-$ROOT/tests/parity/goldens/qwen35_gguf_q4km}"
VENV="${VENV:-$HOME/Sources/tt/venv-qwen35gap}"
MODEL_DIR="${MODEL_DIR:-/tmp/q4km-dequant}"
IDS="${IDS:-our_ids_tenstorrent_capture.i32}"

fail() { echo "REFUSING: $*" >&2; exit 1; }

[ -x "$VENV/bin/python" ] || fail "oracle venv $VENV/bin/python not found"
[ -f "$MODEL_DIR/model.safetensors" ] || fail \
  "dequant artifact $MODEL_DIR/model.safetensors not found (regenerate with \
test_qwen35_gguf_q4km_dequant_dump, see the spec Evidence section)"
[ -f "$GDIR/$IDS" ] || fail "captured ids $GDIR/$IDS not found"

# The spec pins the oracle env (Evidence: teacher-forced goldens). Refuse to
# write a golden from anything else.
"$VENV/bin/python" - <<EOF || fail "oracle env does not match the spec pin \
(python 3.12, torch 2.7.1+cpu, transformers 5.8.1) — see spec tenstorrent-keepquant.md"
import sys
assert sys.version_info[:2] == (3, 12), sys.version
import torch, transformers
assert torch.__version__ == "2.7.1+cpu", torch.__version__
assert transformers.__version__ == "5.8.1", transformers.__version__
EOF

echo "oracle env: $($VENV/bin/python -c 'import sys,torch,transformers; \
print(sys.version.split()[0], torch.__version__, transformers.__version__)')"
echo "dequant artifact: $(sha256sum "$MODEL_DIR/model.safetensors" | cut -d' ' -f1)"
echo "  (record this hash in the landing commit body; input GGUF pin lives in the spec)"

BASE="${IDS%.i32}"
GAPNAME="neartie_gap_mnats_${BASE#our_ids_}.npy"
"$VENV/bin/python" "$ROOT/scripts/qwen3-neartie-gap-transformers.py" \
  --model "$MODEL_DIR" \
  --golden-dir "$GDIR" \
  --our-ids "$IDS" \
  --ids-name "${BASE}.npy" \
  --gap-name "$GAPNAME"
gap="$GDIR/$GAPNAME"

# Provenance warning (not a gate): the pinned bf16 recipe produces gaps on
# the bf16 logit grid (0.0625 nats = 62.5 mnats steps). A wrong-dtype or
# f32-model golden lights up nearly everywhere. i32 storage leaves BORDERLINE
# values ambiguous: a legitimate grid gap can store 0.5 mnats off its point
# (187.5 -> 188, 62.5 -> 62 by half-to-even), so a value at exactly 0.5 off —
# like the incident's 313 (312.5 + 0.5) — is indistinguishable from a stored
# legitimate one. This check fires on distances beyond that storage boundary
# only; the load-bearing tripwires are the env assertion above and the
# bit-for-bit reproducibility check recorded in the spec.
"$VENV/bin/python" - "$gap" <<'EOF'
import sys
import numpy as np
g = np.load(sys.argv[1]).flatten()
dist = np.abs(g - 62.5 * np.round(g / 62.5))
off = [int(v) for v in g[(dist > 0.501) & (g != 0)]]
if off:
    print(f"WARNING: {len(off)} gross off-grid gap values (e.g. {off[:4]}) — "
          "off-recipe numerics suspected; verify venv/model/dtype before "
          "committing this golden", file=sys.stderr)
print(f"gap golden: max {int(g.max())} mnats, "
      f"{int((g > 500).sum())} cells above the 500-mnat band")
EOF
