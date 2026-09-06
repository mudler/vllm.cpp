#!/usr/bin/env bash
# VARIADIC SERVING LOAD: their engine and ours, one client, mixed prompt
# lengths, a concurrency sweep, percentiles.
#
# `.agents/specs/bench-qwen38-exl3-variadic.md`, #2970. The predecessor
# (docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/job-as-run.sh) put
# both engines behind one client at concurrency 1 on one prompt band and
# published means. This job keeps that shape and changes the load.
#
# NO `set -e` ANYWHERE. `set +e ... set -e` around a guarded command ENABLES
# errexit, and a bare `[ cond ] && cmd` afterwards then kills the job on the
# path where the condition is false. That has silently killed two jobs in this
# campaign. Every conditional below is `if ... then ... fi`.
#
# RESUMABLE. `dgx:gpu0` has crashed roughly hourly under load through this
# campaign, and this is a multi-hour sequence. Each phase drops a marker under
# $STATE and each leg writes its JSON as it finishes, so a resubmission after a
# reboot picks up rather than restarting. A crash costs one leg.
#
# PARAMETERISED. Every knob below is an environment variable with a default.
# Nothing about this file is specific to one run; the checkpoint pins are the
# only values it asserts, and they are asserted rather than assumed.
set -u

W=${W:-/workspace/exl3-variadic}
SHARED=${SHARED:-/workspace/exl3-headtohead}   # their fork, wheel and pip cache
SRC=${SRC:-/workspace/qwen38-exl3-bench}       # checkpoints and the src tarball
STATE=$W/state; OUT=$W/out; HARNESS=$W/harness
mkdir -p "$STATE" "$OUT"

CKPT=${CKPT:-$SRC/ckpt}
SCRATCH=${SCRATCH:-/tmp/exl3variadic}
MDL=$SCRATCH/target
DRF=$SCRATCH/draft
OURS_PORT=${OURS_PORT:-8811}
THEIRS_PORT=${THEIRS_PORT:-8812}

# --- the load ---
RUNGS=${RUNGS:-"1 4 8"}
ROUNDS=${ROUNDS:-2}
NPROMPTS=${NPROMPTS:-128}          # MEASURED requests per leg
OUTLEN=${OUTLEN:-192}
TEMP=${TEMP:-0.6}
TOPP=${TOPP:-0.95}
TOPK=${TOPK:-20}
SEED=${SEED:-0}
CORPUS_COUNT=${CORPUS_COUNT:-144}  # >= NPROMPTS + max warmup
CORPUS_SEED=${CORPUS_SEED:-0}

# --- our engine's serving configuration ---
# --max-model-len is EXPLICIT. Left to auto-fit it takes the whole KV pool, and
# the draft context then allocates per concurrent request against that number.
# --num-blocks x block size is the KV pool: 2048 x 32 = 65,536 tokens. Sized
# from the top rung: 8 sequences of (about 3.4k prompt + 192 output) needs about
# 900 blocks, and this model is a HYBRID whose pool ALSO holds each concurrent
# sequence's recurrent state, roughly one page per linear-attention layer, so
# about 384 more. 2048 leaves room over both. It is not set larger because the
# pool is real device memory and a server that cannot allocate it dies at
# startup; `start_ours` raises it and restarts if the clamp bites, and halves it
# and restarts if the server dies allocating it.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
NUM_BLOCKS=${NUM_BLOCKS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_BATCHED=${MAX_BATCHED:-16384}  # our engine has died at 65536 batch tokens
NSPEC=${NSPEC:-7}

EXP_T1=${EXP_T1:-7b77214fe58ff15fed0b4af55e3cd92f38842b8711886d68954e8071ff8270c6}
EXP_T2=${EXP_T2:-411c83bb1070b27f3d670fc93e38dca0f17eb66429f64b5706901b12613188b2}
EXP_DR=${EXP_DR:-6b2e3afc694a343b7f3f0edfe5925e460762fc9ede4699165b577ca0733c8e56}

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
        for _ in $(seq 1 40); do
            if kill -0 "$SERVER_PID" 2>/dev/null; then sleep 1; else break; fi
        done
        kill -9 "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        SERVER_PID=""
        sleep 5     # let the device settle before the next boot
    fi
}
finish(){ st=$?; kill_server; kill -9 $HB 2>/dev/null
          echo "JOB_EXIT_STATUS=$st"; echo "EXL3_VARIADIC_DONE"; }
trap finish EXIT

say "A. PRECONDITIONS"
res "DEVICE $(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>&1 | head -1)"
res "ARCH $(uname -m)  KERNEL $(uname -r)  BOOT $(cat /proc/sys/kernel/random/boot_id)"
LEASE="${RC_JOB_ID:-${RC_LEASE_ID:-}}"
res "LEASE_ID=$LEASE"
if [ -z "$LEASE" ]; then echo "ABORT: no lease id -- never run GPU work outside a lease"; exit 39; fi
mkdir -p "$SCRATCH"
FREE=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
res "SCRATCH_FREE_GiB $FREE"
if [ "${FREE:-0}" -lt 80 ]; then echo "ABORT: /tmp has only ${FREE}G, need >=80"; exit 60; fi
res "LOAD $(cat /proc/loadavg)"

say "A2. PROVISION -- no CUDA toolkit is preinstalled in this container"
export DEBIAN_FRONTEND=noninteractive
APT_ARCHIVES=/workspace/dflash2-staged/aptcache; mkdir -p "$APT_ARCHIVES/partial" 2>/dev/null
APT_CACHED="-o Dir::Cache::archives=$APT_ARCHIVES"
apt-get update -qq > /tmp/aptup.log 2>&1; res "APT_UPDATE_RC=$?"
apt-get install -y -qq $APT_CACHED cmake ninja-build build-essential ccache git \
    curl ca-certificates patch python3-venv python3-dev > /tmp/deps.log 2>&1
res "DEPS_RC=$?"
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
T=$(ls -d "$CUDA_HOME"/targets/*/ 2>/dev/null | head -1)
if [ -n "$T" ]; then
    if [ ! -e "$CUDA_HOME/include" ]; then ln -sfn "${T}include" "$CUDA_HOME/include"; fi
    if [ ! -e "$CUDA_HOME/lib64" ];   then ln -sfn "${T}lib"     "$CUDA_HOME/lib64"; fi
fi
NV=$(have_ver); res "NVCC=$NV"
if [ "$NV" != "13.0" ]; then echo "ABORT: nvcc '$NV'"; tail -20 /tmp/cuda.log; exit 46; fi

say "B. BUILD THE CORPUS -- a pure function of (sources, seed, weights, count)"
CORPUS=$SCRATCH/corpus.json
MANIFEST=$OUT/corpus-manifest.json
python3 "$HARNESS/build_corpus.py" \
    --gsm8k "$W/corpora/gsm8k-test.jsonl" \
    --humaneval "$W/corpora/HumanEval.jsonl" \
    --sonnet "$W/corpora/sonnet.txt" \
    --count "$CORPUS_COUNT" --seed "$CORPUS_SEED" \
    --out "$CORPUS" --manifest "$MANIFEST" > /tmp/corpus.log 2>&1
RCX=$?; res "CORPUS rc=$RCX"
if [ $RCX -ne 0 ]; then tail -20 /tmp/corpus.log; exit 47; fi
res "CORPUS sha256 $(sha256sum "$CORPUS" | cut -d' ' -f1)"
python3 -c "
import json; m=json.load(open('$MANIFEST'))
for k,v in m['sources'].items(): print('RESULT CORPUS_SOURCE %-10s %s' % (k, v['sha256']))
for b,v in sorted(m['realised_chars'].items()): print('RESULT CORPUS_BAND %-3s n=%-3d chars min=%d p50=%d max=%d' % (b, v['n'], v['min'], v['median'], v['max']))
" | tee -a "$OUT/results.txt"

say "C. STAGE THE WEIGHTS OFF CIFS (/workspace holds no symlink; never serve from it)"
if is_done weights && [ -d "$MDL" ] && [ -d "$DRF" ]; then
    res "WEIGHTS already staged in this boot (resumed)"
else
    rm -rf "$MDL" "$DRF"
    cp -rL "$CKPT/target-3.5bpw" "$MDL"; RC1=$?
    cp -rL "$CKPT/draft-dflash2-5.0bpw" "$DRF"; RC2=$?
    res "COPY rc target=$RC1 draft=$RC2"
    if [ $RC1 -ne 0 ] || [ $RC2 -ne 0 ]; then echo "ABORT: weight copy failed"; exit 41; fi
    A=$(sha256sum "$MDL/model-00001-of-00002.safetensors" | cut -d' ' -f1)
    B=$(sha256sum "$MDL/model-00002-of-00002.safetensors" | cut -d' ' -f1)
    C=$(sha256sum "$DRF/model.safetensors" | cut -d' ' -f1)
    res "SHA target1 $A"; res "SHA target2 $B"; res "SHA draft   $C"
    if [ "$A" = "$EXP_T1" ] && [ "$B" = "$EXP_T2" ] && [ "$C" = "$EXP_DR" ]; then
        res "G-BYTES PASS: the bytes measured are the pinned artifact"
        done_mark weights
    else
        res "G-BYTES FAIL: staged bytes are NOT the pinned artifact"
        exit 42
    fi
fi

say "D. BUILD OUR ENGINE"
PIN=$(cat "$W/PINNED_SHA" 2>/dev/null | tr -d '[:space:]')
res "OURS TREE PIN $PIN"
CACHED_BIN=$W/bin/$PIN/vllm-server
BIN=$SCRATCH/bin/vllm-server
mkdir -p "$SCRATCH/bin"
if [ -x "$CACHED_BIN" ]; then
    # Restore to /tmp and run from there. A measured binary is never executed
    # off the share: /workspace is CIFS, and a network filesystem in the load
    # path is exactly the confound this campaign has been bitten by.
    cp -L "$CACHED_BIN" "$BIN"
    cp -L "$W"/bin/"$PIN"/*.so* "$SCRATCH/bin/" 2>/dev/null
    chmod +x "$BIN"
    res "OURS binary restored from the share md5=$(md5sum "$BIN" | cut -d' ' -f1)"
else
    export CCACHE_DIR=/workspace/ccache CCACHE_MAXSIZE=40G
    export CCACHE_NOHARDLINK=1 CCACHE_TEMPDIR=/tmp/ccache-tmp
    mkdir -p "$CCACHE_TEMPDIR"
    if [ ! -d "$SCRATCH/src" ]; then
        mkdir -p "$SCRATCH/src"
        tar xzf "$W/src.tar.gz" -C "$SCRATCH/src"; RCT=$?
        res "SRC untar rc=$RCT (pin $PIN)"
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
    mkdir -p "$W/bin/$PIN"
    cp -L "$BIN" "$CACHED_BIN" 2>/dev/null
    cp -L "$SCRATCH"/bin/*.so* "$W/bin/$PIN/" 2>/dev/null
    # No marker here on purpose. The guard for this phase is the cached binary
    # itself (`[ -x "$CACHED_BIN" ]` above), which is the thing a resume needs;
    # a marker beside it would be a second source of truth that can disagree.
fi

say "E. INSTALL THEIR ENGINE (MiaAI-Lab/exllamav3 @ 63b32f00, OUR pin -- their card pins none)"
if [ ! -d "$SCRATCH/exllamav3-fork" ]; then
    tar xzf "$SHARED/exllamav3-fork.tar.gz" -C "$SCRATCH"
    res "THEIRS source untar rc=$? sha256(tarball)=$(sha256sum "$SHARED/exllamav3-fork.tar.gz" | cut -d' ' -f1)"
fi
if [ ! -f "$SCRATCH/exllamav3-fork/.usage_patched" ]; then
    # THE ONE HARNESS ADAPTATION, confined to their SERVER WRAPPER. It touches
    # no file under exllamav3/, so no engine or kernel behaviour changes. Their
    # streaming path already computes the token counts and the Job's
    # accepted_draft_tokens and then DROPS them; the patch forwards them.
    ( cd "$SCRATCH/exllamav3-fork" && patch -p1 < "$SHARED/serve_openai-usage.patch" )
    RCP=$?; res "THEIRS harness patch rc=$RCP"
    if [ $RCP -ne 0 ]; then echo "ABORT: harness patch did not apply"; exit 53; fi
    python3 -m py_compile "$SCRATCH/exllamav3-fork/tools/serve_openai.py"
    RCPY=$?; res "THEIRS patched server compiles rc=$RCPY"
    if [ $RCPY -ne 0 ]; then echo "ABORT: patched server does not compile"; exit 54; fi
    touch "$SCRATCH/exllamav3-fork/.usage_patched"
fi
VENV=$SCRATCH/venv
PY=$VENV/bin/python
export PIP_CACHE_DIR=$SHARED/pipcache
mkdir -p "$PIP_CACHE_DIR"
if [ -x "$PY" ] && "$PY" -c "import torch, exllamav3" 2>/dev/null; then
    res "THEIRS venv already usable (resumed)"
else
    # The venv lives in /tmp, NOT on /workspace: CIFS holds no symlink, and a
    # multi-GB torch tree on a network share distorts a load time.
    python3 -m venv "$VENV" > /tmp/venv.log 2>&1
    RCV=$?; res "THEIRS venv rc=$RCV"
    if [ $RCV -ne 0 ]; then tail -20 /tmp/venv.log; exit 50; fi
    "$PY" -m pip install --quiet --upgrade pip setuptools wheel packaging typing_extensions > /tmp/pip-boot.log 2>&1
    res "THEIRS pip bootstrap rc=$?"
    "$PY" -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu130 > /tmp/pip-torch.log 2>&1
    RCT=$?; res "THEIRS torch install rc=$RCT"
    if [ $RCT -ne 0 ]; then tail -40 /tmp/pip-torch.log | sed 's/^/  torch| /'; exit 51; fi
    res "THEIRS torch $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>&1 | head -1)"
    export TORCH_CUDA_ARCH_LIST="12.0;12.1"
    export MAX_JOBS=4
    WHL=$(ls "$SHARED"/wheels/exllamav3-*.whl 2>/dev/null | head -1)
    if [ -n "$WHL" ]; then
        res "THEIRS reusing cached wheel $(basename "$WHL") sha256=$(sha256sum "$WHL" | cut -d' ' -f1)"
        "$PY" -m pip install "$WHL" > /tmp/pip-exl3.log 2>&1; RCE=$?
    else
        mkdir -p "$SHARED/wheels"
        "$PY" -m pip wheel --no-build-isolation --no-deps -w "$SHARED/wheels" "$SCRATCH/exllamav3-fork" > /tmp/pip-exl3.log 2>&1
        RCE=$?
        if [ $RCE -eq 0 ]; then
            "$PY" -m pip install --no-build-isolation "$SCRATCH/exllamav3-fork" >> /tmp/pip-exl3.log 2>&1; RCE=$?
        fi
    fi
    res "THEIRS engine build+install rc=$RCE"
    if [ $RCE -ne 0 ]; then
        grep -nE 'error:|Error|FAILED|fatal' /tmp/pip-exl3.log | head -30 | sed 's/^/  exl3| /'
        tail -30 /tmp/pip-exl3.log | sed 's/^/  exl3| /'; exit 52
    fi
    "$PY" -m pip install --quiet aiohttp huggingface_hub > /tmp/pip-srv.log 2>&1
    res "THEIRS server deps rc=$?"
    res "THEIRS import check: $("$PY" -c 'import exllamav3; print("exllamav3", exllamav3.__version__)' 2>&1 | tail -1)"
fi

say "F. SERVER CONTROL"
wait_ready(){  # wait_ready <port> <pid> <label>; bounded, and checks the process is ALIVE
    local port="$1" pid="$2" label="$3" i
    for i in $(seq 1 900); do
        if ! kill -0 "$pid" 2>/dev/null; then
            res "$label SERVER DIED during startup after $((i*2))s (NOT a slow result)"
            return 2
        fi
        if curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
            res "$label ready after $((i*2))s"; return 0
        fi
        sleep 2
    done
    res "$label NEVER BECAME READY within 1800s (NOT a slow result)"; return 3
}

# G-RESOLVED: our server auto-fits max_num_seqs against the KV budget. A leg
# that ran with a resolved value BELOW its rung measured a different engine
# configuration than the one it claims. Read the value back rather than assume
# the flag took.
resolved_seqs(){  # resolved_seqs <serverlog>
    grep -oE 'max_num_seqs (from )?[0-9]+ to [0-9]+' "$1" 2>/dev/null | tail -1 \
        | grep -oE '[0-9]+$'
}

PAGED_ENV=""     # empty means the SHIPPED DEFAULT (paged draft route on)
start_ours_once(){  # start_ours_once <logfile>
    kill_server
    # shellcheck disable=SC2086
    env $PAGED_ENV LD_LIBRARY_PATH="$SCRATCH/bin:$LD_LIBRARY_PATH" "$BIN" \
        --model "$MDL" --device cuda --port "$OURS_PORT" --host 127.0.0.1 \
        --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
        --num-blocks "$NUM_BLOCKS" --max-num-batched-tokens "$MAX_BATCHED" \
        --no-enable-prefix-caching --enable-metrics \
        --served-model-name ours --enable-thinking \
        --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRF\",\"num_speculative_tokens\":$NSPEC}" \
        > "$1" 2>&1 &
    SERVER_PID=$!
    res "OURS server pid=$SERVER_PID paged_env='${PAGED_ENV:-shipped default}'"
    wait_ready "$OURS_PORT" "$SERVER_PID" "OURS"
}

# THIS MODEL IS A HYBRID: 48 of its 64 layers are linear attention, and each
# concurrent sequence owns a slice of the KV pool for its recurrent state as
# well as its KV blocks. The loader therefore CLAMPS max_num_seqs against the
# pool and says so ("INFO recurrent-state budget: reduced max_num_seqs from X to
# Y", model_loader.cpp). The predecessor run was clamped to 1 at 256 blocks.
#
# A leg that ran at a clamped value measured a different engine configuration
# than the rung it claims, so the pool is RAISED and the server RESTARTED rather
# than the leg being published with a footnote. Bounded: two escalations, then
# the rung is refused.
start_ours(){  # start_ours <logfile> [<required concurrency>]
    local slog="$1" need="${2:-$MAX_NUM_SEQS}" tries=0 shrinks=0 rs
    while : ; do
        start_ours_once "$slog"
        if [ $? -ne 0 ]; then
            # A pool that does not fit reads as a server that died at startup.
            # Halve it once and retry, rather than losing the arm to a number
            # this job chose.
            if [ "$shrinks" -lt 1 ] && [ "$NUM_BLOCKS" -gt 512 ]; then
                shrinks=$(( shrinks + 1 ))
                NUM_BLOCKS=$(( NUM_BLOCKS / 2 ))
                res "OURS server did not start at --num-blocks $(( NUM_BLOCKS * 2 )) -- halving to $NUM_BLOCKS and retrying once"
                continue
            fi
            return 1
        fi
        rs=$(resolved_seqs "$slog")
        if [ -z "$rs" ] || [ "$rs" -ge "$need" ]; then
            res "OURS resolved max_num_seqs=${rs:-$MAX_NUM_SEQS} (need $need) at --num-blocks $NUM_BLOCKS"
            return 0
        fi
        tries=$(( tries + 1 ))
        if [ "$tries" -gt 2 ]; then
            res "OURS G-RESOLVED FAIL: max_num_seqs stuck at $rs below $need after $tries escalations (--num-blocks $NUM_BLOCKS)"
            return 0
        fi
        NUM_BLOCKS=$(( NUM_BLOCKS * 4 ))
        res "OURS resolved max_num_seqs=$rs below $need -- raising --num-blocks to $NUM_BLOCKS and restarting"
    done
}
start_theirs(){  # start_theirs <logfile>
    kill_server
    # THEIR RECIPE, VERBATIM FROM THE CARD, only the paths and the port changed.
    ( cd "$SCRATCH/exllamav3-fork" && exec "$PY" tools/serve_openai.py \
        --port "$THEIRS_PORT" --host 127.0.0.1 \
        -m "$MDL" -dm "$DRF" -cq nvfp4 -cs 262144 ) > "$1" 2>&1 &
    SERVER_PID=$!
    res "THEIRS server pid=$SERVER_PID"
    wait_ready "$THEIRS_PORT" "$SERVER_PID" "THEIRS"
}

leg_recorded(){
    test -f "$1" || return 1
    python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$1'))['summary']['headline_warm'].get('ok',0) > 0 else 1)
except Exception: sys.exit(1)
" 2>/dev/null
}

run_leg(){  # run_leg <arm> <round> <c> <port> <modelname> <serverlog>
    local arm="$1" rnd="$2" c="$3" port="$4" name="$5" slog="$6"
    local label="$arm-r$rnd-c$c" out="$OUT/$arm-r$rnd-c$c.json"
    if leg_recorded "$out"; then res "$label already recorded (resumed)"; return 0; fi
    local rs; rs=$(resolved_seqs "$slog")
    if [ -n "$rs" ] && [ "$rs" -lt "$c" ]; then
        res "$label G-RESOLVED FAIL: server resolved max_num_seqs=$rs below rung c=$c -- leg NOT run"
        return 1
    fi
    res "$label CONFIG num_blocks=$NUM_BLOCKS max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_BATCHED paged='${PAGED_ENV:-shipped default}'"
    python3 "$HARNESS/client.py" --url "http://127.0.0.1:$port" --model "$name" \
        --dataset "$CORPUS" --num-prompts "$NPROMPTS" --concurrency "$c" \
        --max-tokens "$OUTLEN" --temperature "$TEMP" --top-p "$TOPP" \
        --top-k "$TOPK" --seed "$SEED" --label "$label" --arm "$arm" \
        --round "$rnd" --out "$out" 2>&1 | tee "$OUT/$label.clientlog"
    res "$label client rc=${PIPESTATUS[0]}"
    return 0
}

say "G. SMOKE PROBE -- does the SHIPPED paged draft route survive the top rung?"
TOP_RUNG=$(for c in $RUNGS; do echo "$c"; done | sort -n | tail -1)
# G-RESOLVED reads back a CLAMP MESSAGE, and model_loader.cpp prints none when
# the configured value is the binding one. So a rung above --max-num-seqs would
# run against a server capped below it and read back an empty string, which
# G-RESOLVED treats as "no clamp". Assert the one case the read-back cannot see.
if [ "$MAX_NUM_SEQS" -lt "$TOP_RUNG" ]; then
    res "ABORT: --max-num-seqs $MAX_NUM_SEQS is below the top rung $TOP_RUNG. The loader prints no clamp line when the flag is the binding value, so G-RESOLVED could not detect this leg running capped."
    exit 62
fi
if is_done pagedprobe && [ -f "$STATE/paged_env" ]; then
    PAGED_ENV=$(cat "$STATE/paged_env")
    res "PAGED PROBE resumed: PAGED_ENV='${PAGED_ENV:-shipped default}'"
else
    start_ours "$OUT/probe.server.log" "$TOP_RUNG"
    if [ $? -ne 0 ]; then
        tail -25 "$OUT/probe.server.log" | sed 's/^/  ours| /'
        res "PAGED PROBE: server did not start on the shipped default"
        kill_server
        PAGED_ENV="VT_DFLASH_PAGED=0"
        start_ours "$OUT/probe2.server.log" "$TOP_RUNG"
        if [ $? -ne 0 ]; then tail -25 "$OUT/probe2.server.log"; exit 61; fi
    fi
    python3 "$HARNESS/client.py" --url "http://127.0.0.1:$OURS_PORT" --model ours \
        --dataset "$CORPUS" --num-prompts 8 --warmup 2 --concurrency "$TOP_RUNG" \
        --max-tokens 64 --temperature "$TEMP" --top-p "$TOPP" --top-k "$TOPK" \
        --seed "$SEED" --label PROBE --arm PROBE --out "$OUT/PROBE.json" \
        > "$OUT/PROBE.clientlog" 2>&1
    RCPB=$?
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then RCPB=99; fi
    res "PAGED PROBE rc=$RCPB on '${PAGED_ENV:-shipped default}'"
    if [ $RCPB -ne 0 ] && [ -z "$PAGED_ENV" ]; then
        res "PAGED PROBE FAILED on the shipped default -- falling back to VT_DFLASH_PAGED=0 for every leg"
        tail -25 "$OUT/probe.server.log" | sed 's/^/  ours| /'
        PAGED_ENV="VT_DFLASH_PAGED=0"
        kill_server
        start_ours "$OUT/probe2.server.log" "$TOP_RUNG"
        python3 "$HARNESS/client.py" --url "http://127.0.0.1:$OURS_PORT" --model ours \
            --dataset "$CORPUS" --num-prompts 8 --warmup 2 --concurrency "$TOP_RUNG" \
            --max-tokens 64 --temperature "$TEMP" --top-p "$TOPP" --top-k "$TOPK" \
            --seed "$SEED" --label PROBE2 --arm PROBE --out "$OUT/PROBE2.json" \
            > "$OUT/PROBE2.clientlog" 2>&1
        res "PAGED PROBE fallback rc=$?"
    fi
    kill_server
    echo "$PAGED_ENV" > "$STATE/paged_env"
    done_mark pagedprobe
fi
res "PAGED ROUTE FOR EVERY LEG: '${PAGED_ENV:-shipped default (paged ON)}'"

say "H. THE LEGS -- rounds interleave the arms AND reverse the rung order"
# Round 2 reverses both the arm order and the rung order, so a monotone drift
# over the session biases each rung in OPPOSITE directions in the two rounds.
# It then shows up as a rung-to-rung disagreement between rounds rather than as
# a shape in the sweep.
asc(){ for c in $RUNGS; do echo "$c"; done | sort -n | tr '\n' ' '; }
desc(){ for c in $RUNGS; do echo "$c"; done | sort -rn | tr '\n' ' '; }

serve_arm(){  # serve_arm <arm> <round> <rung-list>
    local arm="$1" rnd="$2" rungs="$3" c slog
    slog="$OUT/$arm-r$rnd.server.log"
    local all_done=1
    for c in $rungs; do
        if ! leg_recorded "$OUT/$arm-r$rnd-c$c.json"; then all_done=0; fi
    done
    if [ "$all_done" -eq 1 ]; then
        res "$arm round $rnd: every rung already recorded (resumed, server not started)"
        return 0
    fi
    local need=1
    for c in $rungs; do if [ "$c" -gt "$need" ]; then need=$c; fi; done
    if [ "$arm" = "OURS" ]; then start_ours "$slog" "$need"; else start_theirs "$slog"; fi
    if [ $? -ne 0 ]; then
        tail -25 "$slog" | sed "s/^/  $arm| /"; kill_server; return 1
    fi
    if [ "$arm" = "OURS" ]; then
        res "OURS RESOLVED $(grep -iE 'auto-fit|max_num_seqs|draft speculative context' "$slog" | head -3 | tr '\n' ' | ')"
    fi
    for c in $rungs; do
        if [ "$arm" = "OURS" ]; then
            run_leg OURS "$rnd" "$c" "$OURS_PORT" ours "$slog"
        else
            run_leg THEIRS "$rnd" "$c" "$THEIRS_PORT" qwen3.8-27b-exl3-3.5bpw-wm "$slog"
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            res "$arm round $rnd: SERVER DIED after rung c=$c -- remaining rungs of this block are owed"
            SERVER_PID=""
            return 1
        fi
    done
    kill_server
    return 0
}

r=1
while [ "$r" -le "$ROUNDS" ]; do
    if [ $(( r % 2 )) -eq 1 ]; then
        serve_arm THEIRS "$r" "$(asc)"
        serve_arm OURS   "$r" "$(asc)"
    else
        serve_arm OURS   "$r" "$(desc)"
        serve_arm THEIRS "$r" "$(desc)"
    fi
    r=$(( r + 1 ))
done

say "I. THE REPORT"
python3 "$HARNESS/report.py" --dir "$OUT" --glob '*-r*-c*.json' 2>&1 | tee "$OUT/report.md"

say "Z. SUMMARY"
cat "$OUT/results.txt"
