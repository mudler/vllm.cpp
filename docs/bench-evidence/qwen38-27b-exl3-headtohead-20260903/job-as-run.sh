#!/usr/bin/env bash
# THEIR ENGINE vs OURS, one box, one client, one workload, interleaved.
#
# The question this job exists to settle: our published page leads with 1.25x
# on the Qwen3.8-27B EXL3 pair, but that ratio is 59.5 tok/s counted decode-only
# against their 47.5 "decode tok/s", a term their card defines nowhere. The SAME
# run of ours reads 45.1 tok/s counted over whole-run wall time. If they count
# with the prefill in, we are at 0.95x -- slower. Prose cannot settle it, so
# this job runs THEIR engine and OURS behind ONE client that computes BOTH
# conventions from the same timings. The convention question then disappears.
#
# NO `set -e` ANYWHERE. `set +e ... set -e` around a guarded command ENABLES
# errexit, and a bare `[ cond ] && cmd` afterwards then kills the job on the
# path where the condition is false. That has silently killed two jobs in this
# campaign. Every conditional below is `if ... then ... fi`.
#
# RESUMABLE: dgx has crashed roughly hourly under load in this campaign, and
# this job is a multi-hour sequence (16 GiB of weights off CIFS, a cold CUDA
# build of our engine, their engine's 129 translation units, four legs). Each
# phase drops a marker under $STATE and each leg writes its JSON as it finishes,
# so a resubmission after a reboot picks up rather than restarting.
set -u

W=/workspace/exl3-headtohead
STATE=$W/state; OUT=$W/out; mkdir -p "$STATE" "$OUT"
SRC=/workspace/qwen38-exl3-bench
CKPT=$SRC/ckpt
DS=$SRC/humaneval-sharegpt.json

SCRATCH=/tmp/exl3h2h
MDL=$SCRATCH/target
DRF=$SCRATCH/draft
OURS_PORT=8801
THEIRS_PORT=8802
NPROMPTS=${NPROMPTS:-164}
OUTLEN=${OUTLEN:-128}
TEMP=${TEMP:-0.6}
TOPP=${TOPP:-0.95}
TOPK=${TOPK:-20}

T0=$(date +%s)
say(){ echo; echo "##### $(date -u +%H:%M:%S) t=$(( $(date +%s) - T0 ))s $*"; }
res(){ echo "RESULT $*"; echo "RESULT $*" >> "$OUT/results.txt"; }
done_mark(){ touch "$STATE/$1"; }
is_done(){ test -f "$STATE/$1"; }

( while true; do sleep 120; echo "[hb] $(date -u +%H:%M:%S) t=$(( $(date +%s) - T0 ))s"; done ) & HB=$!
SERVER_PID=""
kill_server(){
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 30); do
            if kill -0 "$SERVER_PID" 2>/dev/null; then sleep 1; else break; fi
        done
        kill -9 "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        SERVER_PID=""
    fi
}
finish(){ st=$?; kill_server; kill -9 $HB 2>/dev/null
          echo "JOB_EXIT_STATUS=$st"; echo "EXL3_H2H_DONE"; }
trap finish EXIT

say "A. PRECONDITIONS"
res "DEVICE $(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>&1 | head -1)"
res "ARCH $(uname -m)  KERNEL $(uname -r)  BOOT $(cat /proc/sys/kernel/random/boot_id)"
LEASE="${RC_JOB_ID:-${RC_LEASE_ID:-}}"
res "LEASE_ID=$LEASE"
if [ -z "$LEASE" ]; then echo "ABORT: no lease id -- never run GPU work outside a lease"; exit 39; fi
mkdir -p "$SCRATCH"
# Needs: our build tree, a 15.4 GiB target + 1.4 GiB draft, and a torch venv.
FREE=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
res "SCRATCH_FREE_GiB $FREE"
if [ "${FREE:-0}" -lt 80 ]; then echo "ABORT: /tmp has only ${FREE}G, need >=80"; exit 60; fi

say "A2. PROVISION -- no CUDA toolkit is preinstalled in this container"
export DEBIAN_FRONTEND=noninteractive
APT_ARCHIVES=/workspace/dflash2-staged/aptcache; mkdir -p "$APT_ARCHIVES/partial" 2>/dev/null
APT_CACHED="-o Dir::Cache::archives=$APT_ARCHIVES"
apt-get update -qq > /tmp/aptup.log 2>&1; res "APT_UPDATE_RC=$?"
# `patch` and `python3-venv` are this job's own additions: it applies a harness
# patch and builds a virtualenv, neither of which the earlier jobs did.
apt-get install -y -qq $APT_CACHED cmake ninja-build build-essential ccache git \
    curl ca-certificates patch python3-venv python3-dev > /tmp/deps.log 2>&1
res "DEPS_RC=$? (cmake ninja build-essential ccache git curl patch python3-venv python3-dev)"
have_ver(){ /usr/local/cuda/bin/nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1; }
if [ "$(have_ver)" != "13.0" ]; then
    curl -fsSL -o /tmp/ck.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb > /tmp/ck.log 2>&1
    dpkg -i /tmp/ck.deb >> /tmp/ck.log 2>&1
    apt-get update -qq >> /tmp/ck.log 2>&1
    apt-get install -y $APT_CACHED cuda-toolkit-13-0 > /tmp/cuda.log 2>&1
    res "APT_CUDA_RC=$?"
    if [ -x /usr/local/cuda-13.0/bin/nvcc ]; then
        rm -f /usr/local/cuda; ln -sfn /usr/local/cuda-13.0 /usr/local/cuda
    fi
fi
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# The sbsa packages lay the headers and libs under targets/<triple>/, and both
# our CMake and torch's cpp_extension want them at $CUDA_HOME/{include,lib64}.
T=$(ls -d "$CUDA_HOME"/targets/*/ 2>/dev/null | head -1)
if [ -n "$T" ]; then
    if [ ! -e "$CUDA_HOME/include" ]; then ln -sfn "${T}include" "$CUDA_HOME/include"; fi
    if [ ! -e "$CUDA_HOME/lib64" ];   then ln -sfn "${T}lib"     "$CUDA_HOME/lib64"; fi
fi
NV=$(have_ver); res "NVCC=$NV"
if [ "$NV" != "13.0" ]; then echo "ABORT: nvcc '$NV'"; tail -20 /tmp/cuda.log; exit 46; fi
if [ ! -f "$DS" ]; then echo "ABORT: dataset missing at $DS"; exit 40; fi
res "DATASET $(sha256sum "$DS" | cut -c1-24)... $(python3 -c "import json;print(len(json.load(open('$DS'))),'problems')" 2>/dev/null)"

say "B. STAGE THE WEIGHTS OFF CIFS (/workspace holds no symlink; never serve from it)"
if is_done weights; then
    res "WEIGHTS already staged (resumed)"
else
    rm -rf "$MDL" "$DRF"
    cp -rL "$CKPT/target-3.5bpw" "$MDL"; RC1=$?
    cp -rL "$CKPT/draft-dflash2-5.0bpw" "$DRF"; RC2=$?
    res "COPY rc target=$RC1 draft=$RC2"
    if [ $RC1 -ne 0 ] || [ $RC2 -ne 0 ]; then echo "ABORT: weight copy failed"; exit 41; fi
    # Recompute on the device. A repo id is not a pin and a share can rot.
    A=$(sha256sum "$MDL/model-00001-of-00002.safetensors" | cut -d' ' -f1)
    B=$(sha256sum "$MDL/model-00002-of-00002.safetensors" | cut -d' ' -f1)
    C=$(sha256sum "$DRF/model.safetensors" | cut -d' ' -f1)
    res "SHA target1 $A"
    res "SHA target2 $B"
    res "SHA draft   $C"
    EXP1=7b77214fe58ff15fed0b4af55e3cd92f38842b8711886d68954e8071ff8270c6
    EXP2=411c83bb1070b27f3d670fc93e38dca0f17eb66429f64b5706901b12613188b2
    EXP3=6b2e3afc694a343b7f3f0edfe5925e460762fc9ede4699165b577ca0733c8e56
    if [ "$A" = "$EXP1" ] && [ "$B" = "$EXP2" ] && [ "$C" = "$EXP3" ]; then
        res "SHA MATCH: the bytes measured are the pinned artifact"
        done_mark weights
    else
        res "SHA MISMATCH -- staged bytes are NOT the pinned artifact"
        exit 42
    fi
fi

say "C. BUILD OUR ENGINE (pinned tree 5649e07d, the one the published number came from)"
CACHED_BIN=$W/bin/vllm-server      # the CIFS cache, for a resume after a reboot
BIN=$SCRATCH/bin/vllm-server       # what actually RUNS -- never off CIFS
mkdir -p "$SCRATCH/bin"
if is_done ourbuild && [ -x "$CACHED_BIN" ]; then
    # Restore to /tmp and run from there. A measured binary is never executed off
    # the share: /workspace is CIFS, and a network filesystem in the load path is
    # exactly the kind of confound this campaign has been bitten by.
    cp -L "$CACHED_BIN" "$BIN"
    cp -L "$W"/bin/*.so* "$SCRATCH/bin/" 2>/dev/null
    chmod +x "$BIN"
    res "OURS binary restored from the share to $BIN (resumed) md5=$(md5sum "$BIN" | cut -d' ' -f1)"
else
    export CCACHE_DIR=/workspace/ccache CCACHE_MAXSIZE=40G
    export CCACHE_NOHARDLINK=1 CCACHE_TEMPDIR=/tmp/ccache-tmp
    mkdir -p "$CCACHE_TEMPDIR"
    if [ ! -d "$SCRATCH/src" ]; then
        mkdir -p "$SCRATCH/src"
        tar xzf "$SRC/src.tar.gz" -C "$SCRATCH/src"; RCT=$?
        res "SRC untar rc=$RCT (pin $(cat "$SRC/PINNED_SHA"))"
        if [ $RCT -ne 0 ]; then echo "ABORT: source untar failed"; exit 43; fi
    fi
    SRCDIR=$(find "$SCRATCH/src" -maxdepth 2 -name CMakeLists.txt | head -1 | xargs dirname)
    res "SRCDIR $SRCDIR"
    CMAKE_LINE="cmake -S $SRCDIR -B $SCRATCH/build -G Ninja -DVLLM_CPP_CUDA=ON \
-DVLLM_CPP_CUTLASS_FETCH=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a \
-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER_LAUNCHER=ccache \
-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
    res "OURS BUILD RECIPE: $CMAKE_LINE && cmake --build $SCRATCH/build -j 4 --target vllm-server"
    $CMAKE_LINE > /tmp/ours-cmake.log 2>&1
    RCC=$?; res "OURS cmake rc=$RCC"
    if [ $RCC -ne 0 ]; then tail -30 /tmp/ours-cmake.log; exit 44; fi
    # -j 4: unconstrained parallelism has OOM-rebooted this box.
    cmake --build "$SCRATCH/build" -j 4 --target vllm-server > /tmp/ours-build.log 2>&1
    RCB=$?; res "OURS build rc=$RCB"
    if [ $RCB -ne 0 ]; then
        grep -n 'error:' /tmp/ours-build.log | head -20; tail -30 /tmp/ours-build.log; exit 45
    fi
    BUILT=$(find "$SCRATCH/build" -name 'vllm-server' -type f -perm -u+x | head -1)
    if [ -z "$BUILT" ]; then echo "ABORT: no vllm-server produced"; exit 46; fi
    cp -L "$BUILT" "$BIN"
    find "$SCRATCH/build" -name '*.so*' -exec cp -L {} "$SCRATCH/bin/" \; 2>/dev/null
    res "OURS binary $BIN md5=$(md5sum "$BIN" | cut -d' ' -f1)"
    # Cache a copy on the share so a reboot does not cost the whole CUDA build
    # again. The cache is never the thing that runs.
    mkdir -p "$W/bin"
    cp -L "$BIN" "$CACHED_BIN" 2>/dev/null
    cp -L "$SCRATCH"/bin/*.so* "$W/bin/" 2>/dev/null
    done_mark ourbuild
fi

say "D. INSTALL THEIR ENGINE (MiaAI-Lab/exllamav3 @ 63b32f00, OUR pin -- their card pins none)"
# Their source tree, unpacked and patched BEFORE the venv guard: the server runs
# from this tree (tools/serve_openai.py is not part of the installed wheel), so a
# resume that finds a usable venv still needs the tree here.
if [ ! -d "$SCRATCH/exllamav3-fork" ]; then
    tar xzf "$W/exllamav3-fork.tar.gz" -C "$SCRATCH"
    res "THEIRS source untar rc=$? sha256(tarball)=$(sha256sum "$W/exllamav3-fork.tar.gz" | cut -c1-24)..."
fi
if [ ! -f "$SCRATCH/exllamav3-fork/.usage_patched" ]; then
    # THE ONE HARNESS ADAPTATION, and it is confined to their SERVER WRAPPER
    # (tools/serve_openai.py). It touches no file under exllamav3/, so no engine
    # or kernel behaviour changes. Their streaming path already computes the
    # prompt/completion token counts and the Job's accepted_draft_tokens and
    # then DROPS them; their non-streaming path returns the same numbers. The
    # patch forwards them and emits the standard OpenAI
    # stream_options.include_usage terminal chunk -- the exact shape vllm.cpp's
    # own chat stream emits. Without it a streaming client has no token count to
    # divide by on their side and would have to count SSE chunks, which their
    # HOLD_BACK=16 buffering makes emphatically NOT a token count.
    ( cd "$SCRATCH/exllamav3-fork" && patch -p1 < "$W/serve_openai-usage.patch" )
    RCP=$?
    res "THEIRS harness patch (serve_openai usage+acceptance) rc=$RCP"
    if [ $RCP -ne 0 ]; then echo "ABORT: harness patch did not apply"; exit 53; fi
    python3 -m py_compile "$SCRATCH/exllamav3-fork/tools/serve_openai.py"
    RCPY=$?
    res "THEIRS patched server compiles rc=$RCPY"
    if [ $RCPY -ne 0 ]; then echo "ABORT: patched server does not compile"; exit 54; fi
    touch "$SCRATCH/exllamav3-fork/.usage_patched"
fi
VENV=$SCRATCH/venv
PY=$VENV/bin/python
export PIP_CACHE_DIR=/workspace/exl3-headtohead/pipcache
mkdir -p "$PIP_CACHE_DIR"
if [ -x "$PY" ] && "$PY" -c "import torch, exllamav3" 2>/dev/null; then
    res "THEIRS venv already usable (resumed)"
else
    # The venv lives in /tmp, NOT on /workspace: CIFS holds no symlink, and a
    # multi-GB torch tree on a network share is slow enough to distort a load
    # time. It leaks nothing into the next job because /tmp is per-boot. The
    # wheel cache and the built engine wheel DO live on the share, so a rerun
    # after a crash skips the download and the CUDA compile.
    python3 -m venv "$VENV" > /tmp/venv.log 2>&1
    RCV=$?; res "THEIRS venv rc=$RCV"
    if [ $RCV -ne 0 ]; then tail -20 /tmp/venv.log; exit 50; fi
    "$PY" -m pip install --quiet --upgrade pip setuptools wheel packaging typing_extensions \
        > /tmp/pip-boot.log 2>&1
    res "THEIRS pip bootstrap rc=$?"
    # THE BLOCKER THIS JOB WAS TOLD TO EXPECT: an aarch64 CUDA-13 torch wheel.
    "$PY" -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu130 \
        > /tmp/pip-torch.log 2>&1
    RCT=$?; res "THEIRS torch install rc=$RCT"
    if [ $RCT -ne 0 ]; then
        res "THEIRS TORCH FAILED -- this is the publishable answer, recorded verbatim:"
        tail -40 /tmp/pip-torch.log | sed 's/^/  torch| /'
        exit 51
    fi
    res "THEIRS torch $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>&1 | head -1)"
    res "THEIRS torch cuda_available=$("$PY" -c 'import torch;print(torch.cuda.is_available())' 2>&1 | head -1)"
    export TORCH_CUDA_ARCH_LIST="12.0;12.1"   # what their own start.sh sets on aarch64
    export MAX_JOBS=4                          # -j 4; this box has OOM-rebooted
    export GIT_TERMINAL_PROMPTS=0 GIT_ASKPASS=/bin/true
    WHL=$(ls "$W"/wheels/exllamav3-*.whl 2>/dev/null | head -1)
    if [ -n "$WHL" ]; then
        res "THEIRS reusing cached wheel $(basename "$WHL")"
        "$PY" -m pip install "$WHL" > /tmp/pip-exl3.log 2>&1
        RCE=$?
    else
        mkdir -p "$W/wheels"
        "$PY" -m pip wheel --no-build-isolation --no-deps \
            -w "$W/wheels" "$SCRATCH/exllamav3-fork" > /tmp/pip-exl3.log 2>&1
        RCE=$?
        if [ $RCE -eq 0 ]; then
            WHL=$(ls "$W"/wheels/exllamav3-*.whl 2>/dev/null | head -1)
            "$PY" -m pip install --no-build-isolation "$SCRATCH/exllamav3-fork" \
                >> /tmp/pip-exl3.log 2>&1
            RCE=$?
        fi
    fi
    res "THEIRS engine build+install rc=$RCE"
    if [ $RCE -ne 0 ]; then
        res "THEIRS ENGINE FAILED TO BUILD -- publishable answer, recorded verbatim:"
        grep -nE 'error:|Error|FAILED|fatal' /tmp/pip-exl3.log | head -30 | sed 's/^/  exl3| /'
        tail -30 /tmp/pip-exl3.log | sed 's/^/  exl3| /'
        exit 52
    fi
    "$PY" -m pip install --quiet aiohttp huggingface_hub > /tmp/pip-srv.log 2>&1
    res "THEIRS server deps rc=$?"
    res "THEIRS import check: $("$PY" -c 'import exllamav3; print("exllamav3", exllamav3.__version__)' 2>&1 | tail -1)"
    done_mark theirsinstall
fi

say "E. THE LEGS -- interleaved theirs, ours, theirs, ours (a sequential A/B measures drift)"

wait_ready(){  # wait_ready <port> <pid> <label> ; bounded, and checks the process is ALIVE
    local port="$1" pid="$2" label="$3" i
    for i in $(seq 1 900); do
        if ! kill -0 "$pid" 2>/dev/null; then
            res "$label SERVER DIED during startup after ${i}s (NOT a slow result)"
            return 2
        fi
        if curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
            res "$label ready after ${i}s"
            return 0
        fi
        sleep 2
    done
    res "$label NEVER BECAME READY within 1800s (NOT a slow result)"
    return 3
}

run_client(){  # run_client <label> <port> <modelname>
    local label="$1" port="$2" name="$3"
    python3 "$W/client.py" --url "http://127.0.0.1:$port" --model "$name" \
        --dataset "$DS" --num-prompts "$NPROMPTS" --max-tokens "$OUTLEN" \
        --temperature "$TEMP" --top-p "$TOPP" --top-k "$TOPK" --seed 0 \
        --label "$label" --out "$OUT/$label.json" 2>&1 | tee "$OUT/$label.clientlog"
    return "${PIPESTATUS[0]}"
}

leg_recorded(){  # true only when the leg produced SUCCESSFUL requests
    test -f "$1" || return 1
    python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$1'))['summary'].get('ok',0) > 0 else 1)
except Exception: sys.exit(1)
" 2>/dev/null
}

leg_theirs(){  # leg_theirs <label>
    local label="$1"
    if leg_recorded "$OUT/$label.json"; then
        res "$label already recorded with successful requests (resumed)"; return 0
    fi
    kill_server
    # THEIR RECIPE, VERBATIM FROM THE CARD, only the paths and the port changed:
    #   python tools/serve_openai.py --port 8000 -m <target> -dm <draft> -cq nvfp4 -cs 262144
    ( cd "$SCRATCH/exllamav3-fork" && exec "$PY" tools/serve_openai.py \
        --port "$THEIRS_PORT" --host 127.0.0.1 \
        -m "$MDL" -dm "$DRF" -cq nvfp4 -cs 262144 ) \
        > "$OUT/$label.server.log" 2>&1 &
    SERVER_PID=$!
    res "$label server pid=$SERVER_PID (theirs)"
    wait_ready "$THEIRS_PORT" "$SERVER_PID" "$label"
    local rr=$?
    if [ $rr -ne 0 ]; then
        tail -25 "$OUT/$label.server.log" | sed 's/^/  theirs| /'
        kill_server; return 1
    fi
    run_client "$label" "$THEIRS_PORT" "qwen3.8-27b-exl3-3.5bpw-wm"
    res "$label client rc=$?"
    kill_server
    return 0
}

leg_ours(){  # leg_ours <label>
    local label="$1"
    if leg_recorded "$OUT/$label.json"; then
        res "$label already recorded with successful requests (resumed)"; return 0
    fi
    kill_server
    # VT_DFLASH_PAGED=0: the shipped paged draft route faults eagerly (#2274).
    # It is the one difference from their recipe we cannot remove, and the
    # published page already carries it as a limitation.
    LD_LIBRARY_PATH="$SCRATCH/bin:$LD_LIBRARY_PATH" VT_DFLASH_PAGED=0 "$BIN" \
        --model "$MDL" --device cuda --port "$OURS_PORT" --host 127.0.0.1 \
        --max-num-seqs 1 --served-model-name ours --enable-thinking \
        --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRF\",\"num_speculative_tokens\":7}" \
        > "$OUT/$label.server.log" 2>&1 &
    SERVER_PID=$!
    res "$label server pid=$SERVER_PID (ours)"
    wait_ready "$OURS_PORT" "$SERVER_PID" "$label"
    local rr=$?
    if [ $rr -ne 0 ]; then
        tail -25 "$OUT/$label.server.log" | sed 's/^/  ours| /'
        kill_server; return 1
    fi
    run_client "$label" "$OURS_PORT" "ours"
    res "$label client rc=$?"
    kill_server
    return 0
}

leg_theirs THEIRS-A
leg_ours   OURS-A
leg_theirs THEIRS-B
leg_ours   OURS-B

say "F. THE FOUR CELLS"
python3 - <<'PYSUM' 2>&1 | tee -a "$OUT/results.txt"
import glob, json, os
rows = []
for p in sorted(glob.glob("/workspace/exl3-headtohead/out/*.json")):
    try:
        s = json.load(open(p))["summary"]
    except Exception as e:
        print(f"UNREADABLE {p}: {e}"); continue
    rows.append(s)
if not rows:
    print("NO LEG PRODUCED A RESULT")
else:
    hdr = f"{'leg':<10} {'ok/n':>7} {'decode-only':>12} {'whole-run':>10} {'ttft ms':>9} {'out tok':>8} {'wall s':>8}"
    print(hdr); print("-" * len(hdr))
    for s in rows:
        print(f"{s['leg']:<10} {s.get('ok',0)}/{s.get('requests',0):<5} "
              f"{s.get('decode_only_tok_s',0):>12.2f} {s.get('whole_run_tok_s',0):>10.2f} "
              f"{s.get('mean_ttft_ms',0):>9.1f} {s.get('completion_tokens_total',0):>8} "
              f"{s.get('wall_s',0):>8.1f}")
    def best(pref):
        v = [s for s in rows if s['leg'].startswith(pref) and s.get('ok')]
        if not v: return None
        return (sum(x.get('decode_only_tok_s',0) for x in v)/len(v),
                sum(x.get('whole_run_tok_s',0) for x in v)/len(v))
    t, o = best('THEIRS'), best('OURS')
    if t and o:
        print()
        print(f"VERDICT decode-only : ours {o[0]:.2f} vs theirs {t[0]:.2f} = {o[0]/t[0]:.3f}x")
        print(f"VERDICT whole-run   : ours {o[1]:.2f} vs theirs {t[1]:.2f} = {o[1]/t[1]:.3f}x")
    else:
        print("\nVERDICT INCOMPLETE: theirs=%s ours=%s" % (bool(t), bool(o)))
PYSUM

say "G. ACCEPTANCE, from whatever each engine exposes"
# THEIRS: the Job's own accepted_draft_tokens, surfaced by the harness patch.
# OURS: NOT exposed on the server path -- spec_drafts_proposed/accepted live on
# the internal runner and only the bench client reaches them. Reported as absent
# rather than as zero, because zero would read as "the draft never fired".
for L in THEIRS-A THEIRS-B OURS-A OURS-B; do
    if [ -f "$OUT/$L.json" ]; then
        res "$L ACCEPTANCE $(python3 -c "
import json,sys
s=json.load(open('$OUT/$L.json'))['summary']
a=s.get('accepted_draft_tokens_total')
print('NOT EXPOSED BY THIS ENGINE ON THE SERVER PATH' if a is None else
      'accepted=%d per_output_token=%.3f' % (a, s.get('accepted_draft_tokens_per_output_token',0)))
" 2>&1 | tail -1)"
    fi
done

say "Z. SUMMARY"
cat "$OUT/results.txt"
