#!/bin/bash
# MODEL-MM-QWEN4-EXP: WHY is `-ot .*=ROCm0` 13x slower than the -ngl 99 default?
# Landed evidence says the 27,465.95 MiB CPU model buffer is ONE tensor,
# per_layer_token_embd.weight (iq4_nl), and that probe C's ROCm_Host COMPUTE
# buffer grew to 27,467.24 MiB. Discriminate: is the term that one tensor, or is
# it residency/bandwidth/the expert kernel?
set -u
GGUF=/tmp/ckpt-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
WANT_GGUF=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
WANT_BENCH=03bc5011bdafaa82eaeca4f32ce0ecf35f145b2ab20c79d8f0e69502596a1597
OUT=/workspace/q4exp-gfx1151-denominator/ple-$(date -u +%Y%m%dT%H%M%SZ)
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
( while sleep 120; do kill -0 $$ 2>/dev/null||exit 0; echo "### hb $(date -u +%H:%M:%S) free=$(df -BG --output=avail /tmp|tail -1)"; done ) & HB=$!
trap 'kill -9 $HB 2>/dev/null' EXIT INT TERM

say "IDENTITY"; uname -m; nproc; free -g|head -2; df -h /tmp /workspace 2>&1|head -6
cat /proc/sys/kernel/random/boot_id; rocminfo 2>/dev/null | grep -m2 gfx || true

say "LOCATE the pinned llama.cpp build"
BIN=""
for c in /tmp/q4exp-hip-src-*/build/bin; do [ -x "$c/llama-bench" ] && BIN="$c" && break; done
if [ -z "$BIN" ]; then
  say "no live build tree; restoring from /workspace tar"
  ls -la /workspace/q4exp-gfx1151-denominator/ 2>&1 | head -20
  T=$(ls -1 /workspace/q4exp-gfx1151-denominator/*.tar.gz /workspace/q4exp-gfx1151-denominator/*.tgz /workspace/q4exp-gfx1151-denominator/*.tar 2>/dev/null | head -1)
  if [ -z "$T" ]; then echo "FATAL: no build tree and no tar"; exit 91; fi
  D=/tmp/q4exp-ple-restore-$$; mkdir -p "$D"; tar xf "$T" -C "$D"
  for c in $(find "$D" -name llama-bench -type f); do BIN=$(dirname "$c"); break; done
fi
[ -n "$BIN" ] || { echo "FATAL: no llama-bench"; exit 92; }
echo "BIN=$BIN"
GOT=$(sha256sum "$BIN/llama-bench"|cut -d' ' -f1); echo "llama-bench sha256=$GOT"
[ "$GOT" = "$WANT_BENCH" ] || { echo "FATAL: llama-bench sha mismatch, want $WANT_BENCH"; exit 93; }
export LD_LIBRARY_PATH="$BIN:/opt/rocm/lib"
ldd "$BIN/llama-bench" | grep -c "not found" | sed 's/^/not-found-libs: /'

say "ASSERT the artifact"
[ -f "$GGUF" ] || { echo "FATAL: no $GGUF"; exit 94; }
GOTG=$(sha256sum "$GGUF"|cut -d' ' -f1); echo "gguf sha256=$GOTG"
[ "$GOTG" = "$WANT_GGUF" ] || { echo "FATAL: gguf sha mismatch"; exit 95; }

mkdir -p "$OUT" || { echo "FATAL: cannot write $OUT"; exit 96; }

probe(){ local tag="$1"; shift
  say "PROBE $tag :: $*"
  timeout 1800 "$BIN/llama-bench" -m "$GGUF" -p 0 -n 64 -ngl 99 -r 3 -v -o json "$@" \
     > "$OUT/$tag.json" 2> "$OUT/$tag.err" </dev/null
  local rc=$?; echo "$tag rc=$rc"
  grep -E 'model buffer size|compute buffer size|graph splits|graph nodes|cannot be used with preferred|offloaded .*layers' "$OUT/$tag.err" | sed "s/^/$tag  /"
  echo -n "$tag  n_overrides_applied="; grep -c "buffer type overridden" "$OUT/$tag.err"
  grep "buffer type overridden" "$OUT/$tag.err" | grep -c "per_layer_token_embd" | sed "s/^/$tag  ple_overridden=/"
  grep -iE "illegal memory access|HSA_STATUS_ERROR|core dumped" "$OUT/$tag.err" | head -3 | sed "s/^/$tag  CRASHSIG /"
  python3 -c "
import json;d=json.load(open('$OUT/$tag.json'))[0]
print('$tag  RESULT avg_ts=%.4f stddev=%.4f samples=%s'%(d['avg_ts'],d['stddev_ts'],d['samples_ts']))" 2>&1
  [ $rc -ne 0 ] && tail -8 "$OUT/$tag.err"
  return 0
}

# --- the discriminating set -------------------------------------------------
# A  : control, the production default (expect ~25.9)
# P  : move ONLY per_layer_token_embd to the device, nothing else.
#      copy-hypothesis predicts ~1.95 ; H1/H3 predict near-baseline.
# Q  : move EVERYTHING EXCEPT per_layer_token_embd to the device (first-match-wins).
#      copy-hypothesis predicts ~25.9 -> full-GPU residency per se is NOT slow.
# C  : reproduce -ot .*=ROCm0 (expect ~1.95)
# R  : move the 512-expert MoE weights to the HOST. tests H1 / H2 directly.
# S  : move the shared+dense FFN to host, experts stay on device (H1 control)
probe A_ctl
probe P_ple_dev  -ot "per_layer_token_embd=ROCm0"
probe Q_all_but_ple -ot "per_layer_token_embd=CPU" -ot ".*=ROCm0"
probe C_all_dev  -ot ".*=ROCm0"
probe R_exps_host -ot "ffn_(gate|up|down|gate_up)_exps=CPU"
probe A_ctl2
probe P_ple_dev2 -ot "per_layer_token_embd=ROCm0"
probe Q_all_but_ple2 -ot "per_layer_token_embd=CPU" -ot ".*=ROCm0"

say "MECHANISM: scheduler split assignment, 1 token, 1 rep, both placements"
for tag in sched_A sched_P; do
  EXTRA=""; [ "$tag" = "sched_P" ] && EXTRA='-ot per_layer_token_embd=ROCm0'
  GGML_SCHED_DEBUG=2 timeout 900 "$BIN/llama-bench" -m "$GGUF" -p 0 -n 1 -ngl 99 -r 1 -v -o json $EXTRA \
     > "$OUT/$tag.json" 2> "$OUT/$tag.err" </dev/null
  echo "$tag rc=$?  bytes=$(stat -c%s "$OUT/$tag.err")"
  echo "--- $tag: splits / backend of the PLE gather ---"
  grep -nE "^## SPLIT|GET_ROWS|per_layer_token_embd|node #" "$OUT/$tag.err" | grep -iE "split|per_layer|get_rows" | head -25
  grep -E "compute buffer size|graph splits" "$OUT/$tag.err" | sed "s/^/$tag  /"
done

say "ARCHIVE"
gzip -f "$OUT"/*.err
ls -la "$OUT"
echo "=====PLE_TAR_B64_BEGIN====="; tar czf - -C "$OUT" . | base64 -w 200; echo "=====PLE_TAR_B64_END====="
say "DONE $OUT"
