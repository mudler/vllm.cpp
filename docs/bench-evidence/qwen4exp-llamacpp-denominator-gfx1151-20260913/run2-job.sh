CLOCKPY_B64=<base64 of docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902/amd_clock_sample.py, verbatim>
#!/bin/bash
# Denominator, run 2: N = 12 legs is the DESIGN. Plus the tensor-placement
# capture run 1 did not take, and a -n 32 set at the token count the row's other
# (ungated, separately reported) arm used. ONE ENGINE. No ratio.
set -u
BIN=/tmp/q4exp-hip-src-D9K2/build/bin
CK=/tmp/ckpt-iq1s
GGUF=$CK/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
SHARD1_SHA=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
BENCH_SHA=03bc5011bdafaa82eaeca4f32ce0ecf35f145b2ab20c79d8f0e69502596a1597
TAR=/workspace/q4exp-gfx1151-denominator/llamacpp-pr27742-hip-gfx1151-bin.tar.gz
OUT=/workspace/q4exp-gfx1151-denominator/run2-$(date -u +%Y%m%dT%H%M%SZ)
LEGS=12; NGEN=64; REPS=3; NGL=99
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
res(){ echo "RESULT $*"; echo "RESULT $*" >> "$OUT/results.txt"; }
mkdir -p "$OUT"
( while sleep 120; do kill -0 $$ 2>/dev/null || exit 0; echo "### hb $(date -u +%H:%M:%S)"; done ) & HB=$!
trap 'kill -9 $HB 2>/dev/null' EXIT INT TERM
echo "$CLOCKPY_B64" | base64 -d > /tmp/amd_clock_sample.py

say "1. IDENTITY / PRECONDITIONS"
hostname; date -u; nproc
res "HOST $(hostname) date=$(date -u +%FT%TZ) boot=$(cat /proc/sys/kernel/random/boot_id)"
res "LOADAVG_BEFORE $(cut -d' ' -f1-3 /proc/loadavg)"
[ -x "$BIN/llama-bench" ] || { mkdir -p /tmp/q4exp-hip-src-D9K2/build; tar xzf "$TAR" -C /tmp/q4exp-hip-src-D9K2/build; }
G=$(sha256sum "$BIN/llama-bench" | cut -d' ' -f1); res "llama-bench sha256=$G (expect $BENCH_SHA)"
[ "$G" = "$BENCH_SHA" ] || exit 21
S1=$(sha256sum "$GGUF" | cut -d' ' -f1); res "SHARD1 sha256=$S1"
[ "$S1" = "$SHARD1_SHA" ] || exit 30
export LD_LIBRARY_PATH="$BIN:/opt/rocm/lib"

say "2. WHERE THE TENSORS ACTUALLY LANDED -- a -ngl 99 that silently fell back to the CPU would still print a number"
timeout 900 "$BIN/llama-cli" -m "$GGUF" -ngl $NGL -n 4 -p "The capital of France is" -no-cnv --temp 0 --seed 1 -v \
   > /dev/null 2> "$OUT/placement.err" </dev/null
echo "llama-cli placement leg rc=$? (a timeout is expected: the TUI does not exit)"
echo "-- load_tensors / buffer sizes --"
grep -E 'load_tensors|buffer size|offload|n_gpu_layers|ROCm[0-9]+ model buf|CPU model buf|repack' "$OUT/placement.err" | head -30
echo "-- system_info --"; grep -m1 'system_info:' "$OUT/placement.err"
echo "-- KV cache --"; grep -iE 'kv cache|kv_cache|KV buffer' "$OUT/placement.err" | head -10
echo "-- what the oracle declined to load --"; grep -iE 'missing|not found|skipping|unused tensor' "$OUT/placement.err" | head -10
res "PLACEMENT_LINES $(grep -cE 'load_tensors' "$OUT/placement.err")"
grep -E 'load_tensors:.*buffer size' "$OUT/placement.err" | sed 's/^/RESULT PLACEMENT /' | tee -a "$OUT/results.txt"

say "3. THE $LEGS TIMED LEGS (N=$LEGS IS THE DESIGN)"
for L in $(seq 1 $LEGS); do
  python3 /tmp/amd_clock_sample.py --output "$OUT/clock-leg$L.jsonl" --interval 0.25 & CP=$!
  sleep 1; T0=$(date +%s)
  timeout 1800 "$BIN/llama-bench" -m "$GGUF" -p 0 -n $NGEN -ngl $NGL -r $REPS -o json \
     > "$OUT/leg$L.json" 2> "$OUT/leg$L.err" </dev/null
  RC=$?; T1=$(date +%s); kill -TERM $CP 2>/dev/null; wait $CP 2>/dev/null
  echo "$RC" > "$OUT/leg$L.rc"
  A=$(python3 -c "import json;d=json.load(open('$OUT/leg$L.json'));print(d[0]['avg_ts'],d[0]['stddev_ts'],' '.join(str(x) for x in d[0].get('samples_ts',[])))" 2>&1)
  res "LEG $L rc=$RC wall=$((T1-T0))s $A"
  [ $RC -ne 0 ] && { tail -15 "$OUT/leg$L.err"; grep -qi 'illegal\|memory access\|HSA_STATUS' "$OUT/leg$L.err" && res "LEG $L = the gfx1151 illegal-memory-access signature (ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD)"; }
done

say "4. A SECONDARY SET AT n=32, the token count the row's OTHER arm used (reported separately, never divided)"
for L in 1 2 3 4; do
  timeout 1800 "$BIN/llama-bench" -m "$GGUF" -p 0 -n 32 -ngl $NGL -r 3 -o json > "$OUT/n32-leg$L.json" 2>/dev/null </dev/null
  res "N32 LEG $L rc=$? $(python3 -c "import json;d=json.load(open('$OUT/n32-leg$L.json'));print(d[0]['avg_ts'],' '.join(str(x) for x in d[0]['samples_ts']))" 2>&1)"
done

say "5. FOLD"
python3 - "$OUT" $LEGS <<'PY' 2>&1 | tee "$OUT/FOLD.txt"
import json,sys,statistics,pathlib,glob
out=pathlib.Path(sys.argv[1]); n=int(sys.argv[2])
def fold(pat,cnt,label):
    legs=[];reps=[]
    for i in range(1,cnt+1):
        p=out/(pat%i)
        if not p.exists(): legs.append((i,"?",None,None,[])); continue
        d=json.load(open(p))[0]; s=[float(x) for x in d.get("samples_ts",[])]
        legs.append((i,"0",float(d["avg_ts"]),float(d.get("stddev_ts",0)),s)); reps+=s
    ok=[l for l in legs if l[2] is not None]; v=[l[2] for l in ok]
    print("== %s: legs_designed=%d legs_completed=%d"%(label,cnt,len(ok)))
    for i,rc,a,sd,s in legs: print("  leg %2d avg_ts=%s stddev=%s reps=%s"%(i,a,sd,s))
    if not v: print("  NO LEG COMPLETED"); return None
    med=statistics.median(v)
    d={"label":label,"legs_designed":cnt,"legs_completed":len(ok),"median_tok_s":med,
       "mean_tok_s":statistics.mean(v),"min_leg":min(v),"max_leg":max(v),
       "leg_spread_pct":(max(v)-min(v))/med*100,
       "stdev_of_legs":statistics.stdev(v) if len(v)>1 else 0.0,
       "cv_pct":(statistics.stdev(v)/statistics.mean(v)*100) if len(v)>1 else 0.0,
       "median_of_reps":statistics.median(reps),"min_rep":min(reps),"max_rep":max(reps),
       "rep_spread_pct":(max(reps)-min(reps))/statistics.median(reps)*100,"n_reps":len(reps),
       "per_leg":[{"leg":i,"rc":rc,"avg_ts":a,"stddev_ts":sd,"samples":s} for i,rc,a,sd,s in legs]}
    for k in ("median_tok_s","mean_tok_s","min_leg","max_leg","leg_spread_pct","stdev_of_legs","cv_pct","median_of_reps","min_rep","max_rep","rep_spread_pct","n_reps"):
        print("  %s=%s"%(k.upper(),round(d[k],4)))
    return d
a=fold("leg%d.json",n,"n_gen=64, r=3")
b=fold("n32-leg%d.json",4,"n_gen=32, r=3")
json.dump({"primary":a,"secondary_n32":b,"vllmcpp_binaries_built":0,"vllmcpp_legs_run":0,"ratios_computed":0},
          open(out/"RESULT.json","w"),indent=1)
PY
python3 - "$OUT" <<'PY' 2>&1 | tee "$OUT/CLOCK.txt"
import json,glob,statistics,pathlib,sys
out=pathlib.Path(sys.argv[1])
allb=[]
for f in sorted(glob.glob(str(out/"clock-leg*.jsonl")),key=lambda p:int(''.join(c for c in pathlib.Path(p).stem if c.isdigit()))):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    hi=[r["sclk_mhz"] for r in rows if (r.get("busy_percent") or 0)>=50 and r.get("sclk_mhz")]
    bu=[r["sclk_mhz"] for r in rows if (r.get("busy_percent") or 0)>0 and r.get("sclk_mhz")]
    allb+=hi
    print("%s n=%d busy>0=%d busy>=50=%d  sclk(busy>=50) min/med/max=%s/%s/%s spread=%s%%"%(
      pathlib.Path(f).name,len(rows),len(bu),len(hi),
      min(hi) if hi else None,statistics.median(hi) if hi else None,max(hi) if hi else None,
      round((max(hi)-min(hi))/statistics.median(hi)*100,3) if hi else None))
if allb:
    print("POOLED busy>=50 samples=%d min=%s median=%s max=%s spread=%.3f%%"%(
      len(allb),min(allb),statistics.median(allb),max(allb),(max(allb)-min(allb))/statistics.median(allb)*100))
PY
res "LOADAVG_AFTER $(cut -d' ' -f1-3 /proc/loadavg)"

say "6. EMIT THE EVIDENCE TO STDOUT (rc logs age out and the share is not reachable from the dev box)"
gzip -f "$OUT"/*.err 2>/dev/null
tar czf /tmp/q4exp-evidence.tar.gz -C /workspace/q4exp-gfx1151-denominator . 2>/dev/null
ls -la /tmp/q4exp-evidence.tar.gz; sha256sum /tmp/q4exp-evidence.tar.gz
echo "=====EVIDENCE_TAR_B64_BEGIN====="
tar czf - -C "$OUT" --exclude='*.tar.gz' . | base64 -w 200
echo "=====EVIDENCE_TAR_B64_END====="
say "DONE"
