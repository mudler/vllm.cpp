CLOCKPY_B64=<base64 of docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902/amd_clock_sample.py, verbatim>
#!/bin/bash
# The llama.cpp DENOMINATOR for Qwen3.8-Flash-Next UD-IQ1_S on strix:gpu0 (gfx1151).
# ONE ENGINE, MEASURED ALONE. No vllm.cpp leg, no ratio -- AGENTS.md Gates:
# a performance result needs the arm's declared token-exact gate, and qwen4_exp
# has NONE on ROCm. Shape follows docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902.md
set -u
BIN=/tmp/q4exp-hip-src-D9K2/build/bin
CK=/tmp/ckpt-iq1s
S1F=Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
GGUF=$CK/$S1F
SHARD1_SHA=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
BENCH_SHA=03bc5011bdafaa82eaeca4f32ce0ecf35f145b2ab20c79d8f0e69502596a1597
CLI_SHA=8617aa73d9694e5d8f2fc782763f6be4ac7ed9a2bce10bb28bd919b0326650e9
SRV_SHA=222e5c795a87343156979a450ea0c181dc1c65c4dccde745c153fb2f767384c2
PIN=035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
TAR=/workspace/q4exp-gfx1151-denominator/llamacpp-pr27742-hip-gfx1151-bin.tar.gz
OUT=/workspace/q4exp-gfx1151-denominator/run-$(date -u +%Y%m%dT%H%M%SZ)
LEGS=${LEGS:-6}; NGEN=${NGEN:-64}; REPS=${REPS:-3}; NGL=99
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
res(){ echo "RESULT $*"; echo "RESULT $*" >> "$OUT/results.txt"; }
mkdir -p "$OUT"
( while sleep 120; do kill -0 $$ 2>/dev/null || exit 0; echo "### hb $(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg)"; done ) & HB=$!
trap 'kill -9 $HB 2>/dev/null' EXIT INT TERM
echo "$CLOCKPY_B64" | base64 -d > /tmp/amd_clock_sample.py; python3 -c "import ast;ast.parse(open('/tmp/amd_clock_sample.py').read())" && echo "clock sampler parses"

say "1. IDENTITY AND CONTENTION"
hostname; date -u; uname -m; nproc; free -g | head -2
res "HOST $(hostname) arch=$(uname -m) cpus=$(nproc) date=$(date -u +%FT%TZ)"
echo "-- who else is on this box (load avg is the only in-lease witness) --"
cat /proc/loadavg; uptime
ps -eo pcpu,pmem,comm --sort=-pcpu | head -8
res "LOADAVG_BEFORE $(cut -d' ' -f1-3 /proc/loadavg)"
df -h /tmp /workspace
cat /proc/sys/kernel/random/boot_id
res "BOOT_ID $(cat /proc/sys/kernel/random/boot_id)"
echo "-- amdgpu sysfs --"; ls /sys/class/drm/card*/device/gpu_busy_percent /sys/class/drm/card*/device/pp_dpm_sclk 2>&1
cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | head -20

say "2. THE BINARIES: present, or restored from the archived tar"
if [ ! -x "$BIN/llama-bench" ]; then
  echo "build tree gone from /tmp -- restoring from $TAR (tar preserves mode+symlinks; /workspace strips the exec bit and find -type f drops SONAMEs)"
  mkdir -p /tmp/q4exp-hip-src-D9K2/build && tar xzf "$TAR" -C /tmp/q4exp-hip-src-D9K2/build && echo "restore rc=0"
fi
[ -x "$BIN/llama-bench" ] || { res "FATAL: no llama-bench"; exit 20; }
for p in "llama-bench $BENCH_SHA" "llama-cli $CLI_SHA" "llama-server $SRV_SHA"; do
  set -- $p; G=$(sha256sum "$BIN/$1" | cut -d' ' -f1)
  if [ "$G" = "$2" ]; then res "BINARY $1 sha256 MATCHES the build job: $G"; else res "FATAL: $1 sha256 $G != $2"; exit 21; fi
done
export LD_LIBRARY_PATH="$BIN:/opt/rocm/lib"
echo "-- ldd proves the HIP runtime resolves (an unresolved one exits 127 having measured nothing) --"
ldd "$BIN/llama-bench" | grep -Ei 'hip|rocblas|hsa|not found' | tee "$OUT/ldd-llama-bench.txt"
grep -q 'not found' "$OUT/ldd-llama-bench.txt" && { res "FATAL: unresolved library"; exit 22; }
res "LDD libamdhip64.so.7 -> $(ldd "$BIN/llama-bench" | grep -o '/opt/rocm[^ ]*libamdhip64[^ ]*')"
res "QWEN4EXP_STRINGS_IN_LIBLLAMA $(strings -a "$BIN"/libllama.so.0.3.0 | grep -c qwen4exp)"
res "GFX1151_CODE_OBJECTS_IN_LIBGGML_HIP $(strings -a "$BIN"/libggml-hip.so.0.22.0 | grep -c gfx1151)"

say "3. THE ARTIFACT: the same bytes the row's other arm ran"
ls -la "$CK"
for f in "$CK"/*.gguf; do echo "$(stat -c %s "$f")  $f"; done
S1=$(sha256sum "$GGUF" | cut -d' ' -f1)
res "SHARD1 sha256=$S1 (expect $SHARD1_SHA)"
[ "$S1" = "$SHARD1_SHA" ] || { res "FATAL: artifact is not the pinned one"; exit 30; }
res "SHARD SIZES $(stat -c %s "$CK"/*.gguf | tr '\n' ' ')"
res "FILESYSTEM $(df -h "$CK" | tail -1 | tr -s ' ') -- LOCAL disk, not the CIFS share"

say "4. PILOT LEG: does this oracle load and decode this artifact at all, and how long does a load cost"
T0=$(date +%s)
timeout 3600 "$BIN/llama-bench" -m "$GGUF" -p 0 -n 8 -ngl $NGL -r 1 -o json > "$OUT/pilot.json" 2> "$OUT/pilot.err" </dev/null
PRC=$?; T1=$(date +%s)
res "PILOT rc=$PRC wall=$((T1-T0))s"
tail -30 "$OUT/pilot.err"; echo "-- pilot json --"; cat "$OUT/pilot.json"
if [ $PRC -ne 0 ]; then res "FATAL: the oracle does not run this artifact on gfx1151 -- see pilot.err"; cp -f "$OUT/pilot.err" "$OUT/PILOT_FAILED.err"; exit 40; fi
PILOT_WALL=$((T1-T0))
res "PILOT build_commit=$(python3 -c "import json;d=json.load(open('$OUT/pilot.json'));print(d[0].get('build_commit'),d[0].get('build_number'))" 2>&1)"

say "5. THE $LEGS TIMED LEGS: -p 0 -n $NGEN -ngl $NGL -r $REPS -o json, one process each"
# A leg is ONE llama-bench process: one model load, llama-bench's own warmup run,
# then $REPS timed generations. The warmup is what makes this symmetric with the
# other arm's 'exclude the first generation' rule: no leg's reported figure
# contains a cold weight upload.
for L in $(seq 1 $LEGS); do
  say "leg $L / $LEGS"
  python3 /tmp/amd_clock_sample.py --output "$OUT/clock-leg$L.jsonl" --interval 1.0 & CP=$!
  sleep 2
  T0=$(date +%s)
  timeout 3600 "$BIN/llama-bench" -m "$GGUF" -p 0 -n $NGEN -ngl $NGL -r $REPS -o json \
     > "$OUT/leg$L.json" 2> "$OUT/leg$L.err" </dev/null
  RC=$?; T1=$(date +%s)
  kill -TERM $CP 2>/dev/null; wait $CP 2>/dev/null
  echo "$RC" > "$OUT/leg$L.rc"
  AVG=$(python3 -c "import json;d=json.load(open('$OUT/leg$L.json'));print(d[0]['avg_ts'],d[0]['stddev_ts'],' '.join(str(x) for x in d[0].get('samples_ts',[])))" 2>&1)
  res "LEG $L rc=$RC wall=$((T1-T0))s avg_ts/stddev_ts/samples: $AVG"
  if [ $RC -ne 0 ]; then
    echo "-- leg $L stderr tail --"; tail -20 "$OUT/leg$L.err"
    grep -qi 'illegal\|memory access\|HSA_STATUS' "$OUT/leg$L.err" && res "LEG $L matches the gfx1151 illegal-memory-access signature (ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD)"
  fi
done

say "6. COHERENCE AND BUILD IDENTITY FROM llama-server /props"
"$BIN/llama-server" -m "$GGUF" -ngl $NGL --host 127.0.0.1 --port 8188 -c 4096 -np 1 \
   > "$OUT/server.log" 2>&1 </dev/null & SRV=$!
for i in $(seq 1 300); do sleep 5; curl -sf http://127.0.0.1:8188/health >/dev/null 2>&1 && break; kill -0 $SRV 2>/dev/null || break; done
res "SERVER health after $((i*5))s (pid alive: $(kill -0 $SRV 2>/dev/null && echo yes || echo no))"
curl -sf http://127.0.0.1:8188/props -o "$OUT/props.json" && echo "props rc=0"
python3 -c "import json;d=json.load(open('$OUT/props.json'));print('build_info',d.get('build_info'));print('model_ftype',(d.get('model_ftype') or d.get('default_generation_settings',{}).get('model')));print('modalities',d.get('modalities'))" 2>&1 | tee "$OUT/props-extract.txt"
res "PROPS $(tr '\n' ' ' < "$OUT/props-extract.txt")"
curl -sf http://127.0.0.1:8188/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0,"seed":1}' -o "$OUT/completion.json"
echo "completion rc=$?"; cat "$OUT/completion.json"
res "COMPLETION $(python3 -c "import json;d=json.load(open('$OUT/completion.json'));print(repr(d['choices'][0]['text'][:120]))" 2>&1)"
kill -TERM $SRV 2>/dev/null; sleep 5; kill -9 $SRV 2>/dev/null
grep -iE 'build|llm_load_print_meta: model type|arch' "$OUT/server.log" | head -10

say "7. FOLD: median, mean, min, max, spread -- N IS THE DESIGN, never a grep count"
python3 - "$OUT" $LEGS <<'PY' 2>&1 | tee "$OUT/FOLD.txt"
import json,sys,glob,statistics,pathlib
out=pathlib.Path(sys.argv[1]); n=int(sys.argv[2])
legs=[];reps=[]
for i in range(1,n+1):
    p=out/f"leg{i}.json"; rc=(out/f"leg{i}.rc").read_text().strip() if (out/f"leg{i}.rc").exists() else "?"
    if rc!="0" or not p.exists(): legs.append((i,rc,None,None,[])); continue
    d=json.load(open(p))[0]
    s=[float(x) for x in d.get("samples_ts",[])]
    legs.append((i,rc,float(d["avg_ts"]),float(d.get("stddev_ts",0)),s)); reps+=s
ok=[l for l in legs if l[2] is not None]
print("legs_designed=%d legs_completed=%d legs_failed=%d"%(n,len(ok),n-len(ok)))
for i,rc,a,sd,s in legs:
    print("leg %d rc=%s avg_ts=%s stddev=%s reps=%s"%(i,rc,a,sd,s))
if ok:
    v=[l[2] for l in ok]
    med=statistics.median(v)
    print("MEDIAN_OF_LEGS=%.4f"%med); print("MEAN_OF_LEGS=%.4f"%statistics.mean(v))
    print("MIN_LEG=%.4f MAX_LEG=%.4f"%(min(v),max(v)))
    print("LEG_SPREAD_PCT_OF_MEDIAN=%.3f"%((max(v)-min(v))/med*100))
    if reps:
        print("MEDIAN_OF_REPS=%.4f MIN_REP=%.4f MAX_REP=%.4f REP_SPREAD_PCT=%.3f"%(
            statistics.median(reps),min(reps),max(reps),(max(reps)-min(reps))/statistics.median(reps)*100))
    print("N_REPS=%d"%len(reps))
    json.dump({"legs_designed":n,"legs_completed":len(ok),"median_tok_s":med,
      "mean_tok_s":statistics.mean(v),"min_leg":min(v),"max_leg":max(v),
      "leg_spread_pct":(max(v)-min(v))/med*100,"per_leg":[{"leg":i,"rc":rc,"avg_ts":a,"stddev_ts":sd,"samples":s} for i,rc,a,sd,s in legs],
      "vllmcpp_binaries_built":0,"vllmcpp_legs_run":0,"ratios_computed":0},open(out/"RESULT.json","w"),indent=1)
else:
    print("NO LEG COMPLETED -- there is no denominator")
PY
echo "-- clock windows --"
python3 - "$OUT" <<'PY' 2>&1 | tee "$OUT/CLOCK.txt"
import json,glob,sys,statistics,pathlib
out=pathlib.Path(sys.argv[1])
for f in sorted(glob.glob(str(out/"clock-leg*.jsonl"))):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    busy=[r["busy_percent"] for r in rows if r.get("busy_percent") is not None]
    sc=[r["sclk_mhz"] for r in rows if r.get("sclk_mhz") is not None]
    bu=[r for r in rows if (r.get("busy_percent") or 0)>0]
    scb=[r["sclk_mhz"] for r in bu if r.get("sclk_mhz") is not None]
    print("%s samples=%d busy_samples=%d busy_median=%s sclk_all_median=%s sclk_busy_min/med/max=%s/%s/%s spread_pct=%s boot=%s"%(
        pathlib.Path(f).name,len(rows),len(bu),
        statistics.median(busy) if busy else None,
        statistics.median(sc) if sc else None,
        min(scb) if scb else None, statistics.median(scb) if scb else None, max(scb) if scb else None,
        round((max(scb)-min(scb))/statistics.median(scb)*100,3) if scb and statistics.median(scb) else None,
        rows[0]["boot_id"][:8] if rows else None))
PY
res "LOADAVG_AFTER $(cut -d' ' -f1-3 /proc/loadavg)"
say "8. ARCHIVE"
cp -f /tmp/amd_clock_sample.py "$OUT/" 2>/dev/null
echo "$PIN" > "$OUT/LLAMA_PIN"
gzip -f "$OUT"/leg*.err "$OUT/server.log" "$OUT/pilot.err" 2>/dev/null
ls -la "$OUT"
cat "$OUT/results.txt"
say "BENCH DONE"
