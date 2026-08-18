#!/usr/bin/env bash
# #785 P1 d=256 SharedK WMMA product-seam witness.
# GPU HOLD until Researcher names GO. Default exit 78.
#
# A: SHAREDK_WMMA=1, exclusive WMMA kernel.
# B: SHAREDK_WMMA=0, exclusive scalar SharedK.
# Same binary, separate processes. No timing. No d=512. Never :8010/:8012.
set -euo pipefail

if [[ "${VT_785_P1_GPU_GO:-}" != "1" ]]; then
  printf '%s\n' "P1 GPU HOLD: set VT_785_P1_GPU_GO=1 only after Researcher GO"
  exit 78
fi

ROOT="${VT_785_P1_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BUILD="${VT_785_P1_BUILD:-/home/don/.cache/hermes-builds/vllm-785-p0/build-hip}"
OUT="${VT_785_P1_OUT:-/home/don/.cache/hermes-builds/vllm-785-p0/p1-out}"
BIN="${BUILD}/tests/test_ops_paged_attn_sharedk_wmma_p1_gpu"
PARSER="${ROOT}/tests/scripts/parse_785_p1_trace.py"
ROCPROF="${ROCPROFV3:-/opt/rocm/bin/rocprofv3}"
QHASH=27f164e220edf8d37ddfd1783f0c390968b9d2314ff5d89b7aea3ef50f0eba72
KHASH=d803e64df022e1cb2049b06d65fc36da02a79ac09f77831e967237e47d49f7ca
VHASH=910548159fd5a5a478573dac6de96f374014cb0b76dac7389071f265dfe1cae7

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "$BIN" ]] || die "missing GPU test binary: $BIN"
[[ -f "$PARSER" ]] || die "missing parser: $PARSER"
[[ -x "$ROCPROF" ]] || die "missing rocprofv3: $ROCPROF"

rm -rf "$OUT"
mkdir -p "$OUT/A" "$OUT/B"

run_arm() {
  local arm="$1"
  local wmma="$2"
  local dest="$OUT/$arm"
  mkdir -p "$dest/trace" "$dest/art"
  env \
    VT_785_P1_GPU=1 \
    VT_785_P1_OUT="$dest/art" \
    VT_ATTN_PREFILL_SHAREDK_WMMA="$wmma" \
    VT_ATTN_PREFILL_FLASH_SHAREDK=1 \
    VT_ATTN_DECODE_OPT=1 \
    VT_ATTN_DECODE_GQA=1 \
    VT_ROCM_ATTN_CPU_REF=0 \
    VT_CPU_REF=0 \
    "$ROCPROF" --kernel-trace --output-format csv \
      -d "$dest/trace" -o "ktrace" \
      -- "$BIN" --test-case="*product seam*"
  local rc=$?
  if (( rc != 0 )); then
    die "arm $arm binary rc=$rc (fail closed; 77 is skip, not a pass; no retry)"
  fi
  python3 "$PARSER" "$dest/trace" --expect "$arm" \
    | tee "$dest/trace-class.txt"
  local hashes="$dest/art/fixture-hashes.txt"
  [[ -f "$hashes" ]] || die "arm $arm missing fixture-hashes.txt"
  grep -qx "q_bf16=$QHASH" "$hashes" || die "arm $arm Q hash mismatch"
  grep -qx "k_bf16=$KHASH" "$hashes" || die "arm $arm K hash mismatch"
  grep -qx "v_bf16=$VHASH" "$hashes" || die "arm $arm V hash mismatch"
}

run_arm A 1
run_arm B 0

python3 - <<PY
import math, struct, sys
from pathlib import Path
out = Path("$OUT")
def load(p):
    b = Path(p).read_bytes()
    if len(b) % 2:
        print("ERROR: odd bf16 byte length", p, file=sys.stderr)
        sys.exit(1)
    n = len(b)//2
    return [struct.unpack_from("<H", b, 2*i)[0] for i in range(n)]

def bits_to_f32(h):
    return struct.unpack("<f", struct.pack("<I", (h & 0xffff) << 16))[0]

a = [bits_to_f32(x) for x in load(out/"A/art/out.bf16")]
b = [bits_to_f32(x) for x in load(out/"B/art/out.bf16")]
if len(a) != len(b) or not a:
    print("ERROR: A/B output length mismatch or empty", len(a), len(b), file=sys.stderr)
    sys.exit(1)
max_abs = max(abs(x-y) for x,y in zip(a,b))
mean_a = sum(a)/len(a); mean_b = sum(b)/len(b)
num = sum((x-mean_a)*(y-mean_b) for x,y in zip(a,b))
da = math.sqrt(sum((x-mean_a)**2 for x in a))
db = math.sqrt(sum((y-mean_b)**2 for y in b))
corr = 0.0 if da == 0.0 or db == 0.0 else num/(da*db)
print(f"ab_max_abs={max_abs:.8f}")
print(f"ab_corr={corr:.8f}")
print(f"n={len(a)}")
Path(out/"ab-distance.txt").write_text(
    f"ab_max_abs={max_abs:.8f}\nab_corr={corr:.8f}\nn={len(a)}\n")
for arm in ("A","B"):
    met = (out/arm/"art"/"metrics.txt").read_text()
    print(f"--- {arm} metrics ---")
    print(met, end="" if met.endswith("\n") else "\n")
    if "nonfinite=0" not in met:
        print("ERROR: nonfinite in", arm, file=sys.stderr)
        sys.exit(1)
    if "oracle_ok=1" not in met:
        print("ERROR: oracle miss in", arm, file=sys.stderr)
        sys.exit(1)
print("P1 compare complete (no promote)")
PY

printf '%s\n' "P1 arms A and B completed; lab does not self-promote GREEN"
