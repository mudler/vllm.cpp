#!/usr/bin/env bash
# The llama.cpp ARM of the MODEL-MM-QWEN4-EXP speed gate (G4), as a concurrency
# ladder.
#
# Row      MODEL-MM-QWEN4-EXP (#1978)
# Oracle   .agents/oracles/llama-cpp-qwen4exp.md, pin 035e2273 (ggml-org PR #27742).
#          No RELEASED llama.cpp names `qwen4exp`, so the stock `llama-cpp` pin
#          cannot supply this denominator.
# Gate     .agents/specs/qwen4-exp-flash-next.md §Gates, G4. The developer's
#          binding words are quoted there: "we should be faster than llama.cpp",
#          "especially at high concurrency". A c=1 result neither confirms nor
#          refutes that, so the LADDER is the headline.
#
# WHAT MAKES THE TWO ARMS COMMENSURABLE.  This script does not time anything
# itself.  Timed requests are issued only by the pinned `vllm bench serve`
# client -- the SAME instrument, the same corpus partitions, the same flags in
# the same order that `tools/bench/online_gate.py:build_client_command` already
# issues against our own server.  Two ratio sets that disagree are frequently
# two harnesses (.agents/benchmarking.md), and the cheapest way not to have that
# argument is to have one client.  This script owns only: server lifecycle, the
# ladder, the mutex boundary, the clock and memory windows, and artifact
# capture.
#
# "The same flags" is an assertion here, not a claim.
# `test_client_flag_list_matches_build_client_command` reads the flag sequence
# out of `build_client_command` and compares it to the invocation below, so the
# two cannot drift apart silently.  The first version of this file said "the
# same flags" and diverged on TWO of them, which is why the claim is now tested:
#
#   * `--num-warmups`.  `OnlineRun.num_warmups` defaults to the CONCURRENCY, so
#     our arm takes 32 warmups at c=32.  This file hardcoded 0, which would have
#     given llama.cpp none -- and `docs/benchmarks/vllm-online-serving.md`
#     records warmup as exactly where two engines differ most.  It now passes
#     the concurrency, as the vllm.cpp arm does.
#   * The CORPUS.  `online_gate.prepare_corpus_views` builds one partition per
#     (concurrency, repetition) and REFUSES if any prompt appears in two of
#     them.  This file passed ONE file to all 18 legs, so with
#     `--disable-shuffle` every repetition replayed the same prompts and warmed
#     llama-server's slot prefix cache in a way our arm never sees.  It now
#     consumes the same `c<C>-r<R>.jsonl` partition set, by the same names.
#
# The spec's G4 wording is "Identical artifact, PROMPTS, token counts, sampling
# and concurrency".  A shared corpus is not identical prompts; it is the same
# prompts three times on one side and three disjoint thirds on the other.
#
# THE LADDER IS NOT INVENTED HERE.  POINTS, INPUT_LEN, OUTPUT_LEN and the three
# repetitions are the published online-serving grid
# (`tools/bench/online_gate.py:59-60,232`, rendered in
# `docs/benchmarks/vllm-online-serving.md`), so a cell of this table can be read
# beside a cell of that one.  The spec asks for c = 1, 4, 8, 16, 32 "at
# minimum"; this is that set plus the c=2 the published grid carries.
#
# WHAT THIS SCRIPT REFUSES TO DO.  It never pins a clock: inside an `rc` lease
# `nvidia-smi -lgc` returns LGC_RC=4 even as root, so the SM clock here is
# SAMPLED and a pairing may be refused on spread with no lever to fix it
# (.agents/benchmarking.md, #1354).  A pinned clock left behind outlives the
# lease and reprices the next holder's measurement, so this script has no code
# path that could leave one.
#
# `set -u` WITHOUT `set -e`.  Every command's status is read explicitly into a
# variable, because `$?` after a pipe reads the status of the LAST element and
# has already made a failing gate read rc=0 four times in one session here.
set -u

readonly E_USAGE=2
readonly E_MODEL_MISSING=10
readonly E_SERVER_MISSING=11
readonly E_CLIENT_MISSING=12
readonly E_WRAPPER_MISSING=13
readonly E_CORPUS_MISSING=14
readonly E_MEM_BUDGET=15
readonly E_LOCK_TIMEOUT=16
readonly E_SERVER_NOT_READY=17
readonly E_LEG_FAILED=18
readonly E_KV_OVER_BUDGET=19
readonly E_CLOCK_UNUSABLE=20
readonly E_KV_UNREPORTED=21

# ---------------------------------------------------------------- the ladder
# (concurrency, num_prompts).  A strict copy of online_gate.POINTS.
readonly LADDER="1:6 2:6 4:12 8:24 16:96 32:192"
readonly INPUT_LEN=1024
readonly OUTPUT_LEN=128
readonly REPETITIONS=3
# The concurrency the server is configured for.  Every leg runs against ONE
# server process so the ladder measures the engine and not N different engines.
readonly MAX_CONCURRENCY=32

# ------------------------------------------------------------- configuration
# EVERY name used in an arithmetic expansion below is given a default HERE.
# `set -u` turns an undefaulted operand into "unbound variable" at the moment
# the leg runs, which is after the lock is held and after the model is loaded --
# the most expensive place in the run to discover a typo.  The rule this file
# keeps: an identifier that appears inside $(( )) is assigned in this block.
MODEL=${MODEL:-}
SERVER=${SERVER:-}
CLIENT=${CLIENT:-}
TOKENIZER=${TOKENIZER:-}
CORPUS_DIR=${CORPUS_DIR:-}
OUT=${OUT:-evi-q4exp-llamacpp}
PORT=${PORT:-8077}
NGL=${NGL:-99}
# Total context across all slots.  llama-server divides -c by -np, so each of
# the 32 slots gets CTX_TOTAL/32 tokens and must hold INPUT_LEN + OUTPUT_LEN.
CTX_TOTAL=${CTX_TOTAL:-49152}
# Measured, never guessed.  A KV formula invented in a harness once asked for
# 128 GiB on a 119 GB box.  0 means "not supplied".
#
# THIS VALUE IS NOT OPTIONAL, and the earlier version of this file was wrong to
# imply that it was.  It said the binding check was the post-launch one against
# llama-server's OWN reported KV size, "which always runs".  Measured against
# this row's own production capture, `llama-server` at THIS PIN prints no such
# line: `.../decode-proof/llama-server.log` is the complete, unfiltered server
# output (the decode proof redirects both streams into it) and it carries no
# `KV self size`, no `llama_kv_cache:` sizing line, and no allocation summary at
# all -- 16 minutes of load between `load_model:` and `threadpool init` with
# nothing in between.  `/props` carries no KV bytes either.  So with
# KV_BYTES_PER_TOKEN=0 NEITHER check has a KV term, and the ladder would size a
# 32-slot context against a 67.5 GiB model on a box that reboots rather than
# swaps.  That is the 128-GiB-on-a-119-GB-box defect wearing a guard.
#
# The harness therefore REFUSES rather than defaulting the term to zero.  Supply
# a value measured on a leased load, or the ladder does not run.
KV_BYTES_PER_TOKEN=${KV_BYTES_PER_TOKEN:-0}
# Share of the box's memory the run may occupy.  GB10 is unified, so the device
# ceiling IS host MemTotal and there is no second pool to fall back on.
MEM_FRACTION=${MEM_FRACTION:-85}
MEM_TOTAL_BYTES=${MEM_TOTAL_BYTES:-0}
# Bounds.  Every wait in this file is bounded, because every wait in this file
# happens while the GPU mutex is held.
LOCK_WAIT_SECONDS=${LOCK_WAIT_SECONDS:-7200}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-1800}
READY_POLL_INTERVAL=${READY_POLL_INTERVAL:-2}
LEG_TIMEOUT_SECONDS=${LEG_TIMEOUT_SECONDS:-3600}
SHUTDOWN_GRACE_SECONDS=${SHUTDOWN_GRACE_SECONDS:-60}
CLOCK_INTERVAL=${CLOCK_INTERVAL:-0.5}
CLOCK_MAX_SECONDS=${CLOCK_MAX_SECONDS:-7200}
MEM_INTERVAL=${MEM_INTERVAL:-1.0}
# `${HOME:-...}`, not `$HOME`.  Under `set -u` a bare $HOME is an unbound
# variable in any environment that does not export it -- an `rc` job, a cron
# entry, a `env -i` reproduction -- and it aborts on line 1 of the defaults
# block, before a single guard has had a chance to name what is wrong.
GPU_LOCK=${GPU_LOCK:-${HOME:-/tmp}/gpu.lock}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# Wrapper binaries, named so `command -v` can assert each one.  A wrapper that
# is absent does not make the wrapped command run unwrapped -- it makes it not
# run at all, and a leg that never ran writes no artifact and looks exactly
# like a leg that ran and produced nothing.
TIMEV_BIN=${TIMEV_BIN:-/usr/bin/time}
TIMEOUT_BIN=${TIMEOUT_BIN:-timeout}
FLOCK_BIN=${FLOCK_BIN:-flock}
PYTHON_BIN=${PYTHON_BIN:-python3}
CURL_BIN=${CURL_BIN:-curl}
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}

MODE=""
SERVER_PID=""
CLOCK_PID=""
MEM_PID=""

usage() {
  cat >&2 <<'EOF'
usage:
  qwen4exp-llamacpp-ladder.sh --self-check
      Run every preflight guard and exit.  Touches no GPU, starts no server,
      takes no lock.  This is the mode the tests drive.
  qwen4exp-llamacpp-ladder.sh --execute
      Take the GPU mutex, start one llama-server, run the ladder, stop.
  qwen4exp-llamacpp-ladder.sh --summarize --out DIR
      Render the ladder table from the legs already in DIR.  Every published
      figure comes from here, never from a human reading a JSON file.

required (env or flag):
  --model PATH        shard 1 of the GGUF; every shard named by it must exist
  --server PATH       llama-server built at the llama-cpp-qwen4exp pin
  --client PATH       the pinned vLLM client (issues every timed request)
  --tokenizer DIR     tokenizer snapshot handed to the client
  --corpus-dir DIR    the frozen corpus VIEW directory: one c<C>-r<R>.jsonl
                      partition per rung and repetition, named exactly as
                      online_gate.prepare_corpus_views writes them
optional:
  --out DIR --port N --ctx-total N --kv-bytes-per-token N --mem-fraction PCT
EOF
}

while (($#)); do
  case "$1" in
    --self-check|--execute|--summarize)
      [[ -z ${MODE} ]] || { echo "choose exactly one mode" >&2; exit "${E_USAGE}"; }
      MODE=${1#--}; shift ;;
    --model) MODEL=${2:?}; shift 2 ;;
    --server) SERVER=${2:?}; shift 2 ;;
    --client) CLIENT=${2:?}; shift 2 ;;
    --tokenizer) TOKENIZER=${2:?}; shift 2 ;;
    --corpus-dir) CORPUS_DIR=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --port) PORT=${2:?}; shift 2 ;;
    --ctx-total) CTX_TOTAL=${2:?}; shift 2 ;;
    --kv-bytes-per-token) KV_BYTES_PER_TOKEN=${2:?}; shift 2 ;;
    --mem-fraction) MEM_FRACTION=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit "${E_USAGE}" ;;
  esac
done
[[ -n ${MODE} ]] || { usage; exit "${E_USAGE}"; }

say() { printf '%s\n' "$*"; }
fail() {  # fail CODE MESSAGE...
  local code
  code=$1
  shift
  printf 'REFUSED(%s): %s\n' "${code}" "$*" >&2
  exit "${code}"
}

# ------------------------------------------------------------------ teardown
# Runs on the success path, on the error path, and on the kill path.  A server
# left running holds the whole box; a sampler left running writes forever.
# Nothing here pins or releases a clock, because nothing here ever pinned one.
teardown() {
  local pid
  for pid in "${CLOCK_PID}" "${MEM_PID}"; do
    [[ -n ${pid} ]] && kill "${pid}" 2>/dev/null
  done
  if [[ -n ${SERVER_PID} ]]; then
    kill "${SERVER_PID}" 2>/dev/null
    local waited=0
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
      [[ ${waited} -ge ${SHUTDOWN_GRACE_SECONDS} ]] && { kill -9 "${SERVER_PID}" 2>/dev/null; break; }
      sleep 1
      waited=$((waited + 1))
    done
  fi
  wait 2>/dev/null
}
trap teardown EXIT INT TERM

# ------------------------------------------------------------- shard discovery
# "-00001-of-00003" is llama.cpp's own split naming.  Deriving the SET from the
# name is what turns "the artifact is missing" into a named refusal instead of a
# load that dies half way through with the shard number in a log nobody reads.
model_shards() {
  local first base dir stem total idx name
  first=$1
  dir=$(dirname "${first}")
  base=$(basename "${first}")
  if [[ ${base} =~ ^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
    stem=${BASH_REMATCH[1]}
    total=$((10#${BASH_REMATCH[3]}))
    for ((idx = 1; idx <= total; idx++)); do
      name=$(printf '%s-%05d-of-%05d.gguf' "${stem}" "${idx}" "${total}")
      printf '%s\n' "${dir}/${name}"
    done
  else
    printf '%s\n' "${first}"
  fi
}

weights_bytes() {
  local total path size
  total=0
  while IFS= read -r path; do
    size=$(stat -c %s "${path}" 2>/dev/null)
    [[ -n ${size} ]] || size=0
    total=$((total + size))
  done < <(model_shards "$1")
  printf '%s\n' "${total}"
}

detect_mem_total_bytes() {
  local reported kb
  if [[ ${MEM_TOTAL_BYTES} -gt 0 ]]; then
    printf '%s nvidia-smi-overridden\n' "${MEM_TOTAL_BYTES}"
    return 0
  fi
  # GB10 reports `[N/A]` for VRAM -- the rc device labels say so -- because the
  # memory is unified.  So the ceiling that binds is host MemTotal, and a
  # harness that trusted nvidia-smi here would size against the string "N/A".
  if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
    reported=$("${NVIDIA_SMI}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
    if [[ -n ${reported} && ${reported} -gt 0 ]]; then
      printf '%s nvidia-smi\n' "$((reported * 1024 * 1024))"
      return 0
    fi
  fi
  kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null)
  [[ -n ${kb} ]] || kb=0
  printf '%s proc-meminfo\n' "$((kb * 1024))"
}

# -------------------------------------------------------------------- guards
preflight() {
  local shard missing bin total source_name budget need kvterm point rep part
  [[ -n ${MODEL} ]] || fail "${E_MODEL_MISSING}" "--model is required and names shard 1 of the GGUF"
  missing=0
  while IFS= read -r shard; do
    if [[ ! -s ${shard} ]]; then
      printf 'missing or empty shard: %s\n' "${shard}" >&2
      missing=$((missing + 1))
    fi
  done < <(model_shards "${MODEL}")
  [[ ${missing} -eq 0 ]] || fail "${E_MODEL_MISSING}" \
    "${missing} shard(s) of ${MODEL} are missing or empty; a truncated shard measures a truncated file, not an oracle"

  [[ -n ${SERVER} && -x ${SERVER} ]] || fail "${E_SERVER_MISSING}" \
    "llama-server is not executable at '${SERVER}'; build it at the llama-cpp-qwen4exp pin first"
  [[ -n ${CLIENT} && -x ${CLIENT} ]] || fail "${E_CLIENT_MISSING}" \
    "the pinned vLLM client is not executable at '${CLIENT}'; it issues every timed request, so without it there is no measurement"
  [[ -n ${TOKENIZER} && -d ${TOKENIZER} ]] || fail "${E_CORPUS_MISSING}" \
    "tokenizer snapshot missing at '${TOKENIZER}'"
  [[ -n ${CORPUS_DIR} && -d ${CORPUS_DIR} ]] || fail "${E_CORPUS_MISSING}" \
    "frozen corpus view directory missing at '${CORPUS_DIR}'"
  # Every one of the 18 partitions, asserted BEFORE the lock and before a 67 GiB
  # load.  A partition discovered missing at rep 3 has already spent the box.
  missing=0
  for point in ${LADDER}; do
    for rep in $(seq 1 "${REPETITIONS}"); do
      part="${CORPUS_DIR}/c${point%%:*}-r${rep}.jsonl"
      if [[ ! -s ${part} ]]; then
        printf 'missing or empty corpus partition: %s\n' "${part}" >&2
        missing=$((missing + 1))
      fi
    done
  done
  [[ ${missing} -eq 0 ]] || fail "${E_CORPUS_MISSING}" \
    "${missing} of $((6 * REPETITIONS)) corpus partitions are missing or empty under '${CORPUS_DIR}'; online_gate.prepare_corpus_views writes one per (concurrency, repetition) and our arm consumes exactly that set"

  # A wrapper that is not on PATH does not degrade the leg, it deletes it.
  for bin in "${TIMEV_BIN}" "${TIMEOUT_BIN}" "${FLOCK_BIN}" "${PYTHON_BIN}" "${CURL_BIN}"; do
    if [[ ${bin} == /* ]]; then
      [[ -x ${bin} ]] || fail "${E_WRAPPER_MISSING}" \
        "wrapper '${bin}' is not executable; a missing wrapper makes the wrapped command not run at all"
    else
      command -v "${bin}" >/dev/null 2>&1 || fail "${E_WRAPPER_MISSING}" \
        "wrapper '${bin}' is not on PATH; a missing wrapper makes the wrapped command not run at all"
    fi
  done
  [[ -f ${REPO_ROOT}/tools/bench/gpu_clock_state.py ]] || fail "${E_WRAPPER_MISSING}" \
    "tools/bench/gpu_clock_state.py not found under REPO_ROOT='${REPO_ROOT}'; .agents/benchmarking.md requires this helper, not a rolled one"
  [[ -f ${REPO_ROOT}/tools/bench/sample_process_memory.py ]] || fail "${E_WRAPPER_MISSING}" \
    "tools/bench/sample_process_memory.py not found under REPO_ROOT='${REPO_ROOT}'"

  [[ ${CTX_TOTAL} -ge $((MAX_CONCURRENCY * (INPUT_LEN + OUTPUT_LEN))) ]] || fail "${E_MEM_BUDGET}" \
    "CTX_TOTAL=${CTX_TOTAL} cannot hold ${MAX_CONCURRENCY} slots of $((INPUT_LEN + OUTPUT_LEN)) tokens; llama-server divides -c by -np"

  read -r total source_name < <(detect_mem_total_bytes)
  [[ ${total} -gt 0 ]] || fail "${E_MEM_BUDGET}" \
    "could not read the box's memory total from ${source_name}; sizing a KV cache against an unknown ceiling is the 128-GiB-on-a-119-GB-box defect"
  budget=$((total / 100 * MEM_FRACTION))
  need=$(weights_bytes "${MODEL}")
  kvterm=$((CTX_TOTAL * KV_BYTES_PER_TOKEN))
  need=$((need + kvterm))
  say "memory total     ${total} bytes (source: ${source_name})"
  say "memory budget    ${budget} bytes (${MEM_FRACTION}%)"
  say "weights + KV     ${need} bytes (KV term ${kvterm}, from KV_BYTES_PER_TOKEN=${KV_BYTES_PER_TOKEN})"
  if [[ ${KV_BYTES_PER_TOKEN} -eq 0 ]]; then
    say "NOTE: KV_BYTES_PER_TOKEN is 0, so this static check covers the WEIGHTS only."
    say "      No formula is invented here.  llama-server at this pin reports no KV"
    say "      size either, so with 0 supplied NOTHING carries a KV term and the"
    say "      post-launch check will REFUSE (${E_KV_UNREPORTED}) rather than pass."
    say "      --self-check still completes: this mode never launches a server."
  fi
  [[ ${need} -le ${budget} ]] || fail "${E_MEM_BUDGET}" \
    "weights+KV ${need} bytes exceeds ${MEM_FRACTION}% of ${total} bytes; lower CTX_TOTAL or pick a smaller arm"
  say "PREFLIGHT OK"
}

# The memory check that runs after the model is resident, and the one place this
# harness was failing OPEN.
#
# It used to grep the server log for `KV self size = N` and, finding nothing,
# set the term to 0 and pass.  llama-server at this pin PRINTS NO SUCH LINE --
# measured on this row's own production capture, not assumed -- so on the real
# denominator the guard had no KV term at all, while the static check next door
# also had none whenever KV_BYTES_PER_TOKEN was left at its 0 default.  Two
# checks, neither of them bounding the cache, in front of a 32-slot 49,152-token
# context on a box that reboots rather than swaps.
#
# The order of preference here is: the ENGINE'S OWN number if it ever prints one
# (measured by the thing that has to live in it), otherwise the operator's
# measured KV_BYTES_PER_TOKEN, otherwise REFUSE.  A silent server and a zero
# term is not a pass.
assert_reported_kv_fits() {
  local log total source_name budget weights kv_mib kv_bytes need origin
  log=$1
  read -r total source_name < <(detect_mem_total_bytes)
  budget=$((total / 100 * MEM_FRACTION))
  weights=$(weights_bytes "${MODEL}")
  # The spellings llama.cpp has used for this line, so a pin that starts
  # printing one is picked up without a code change.  NONE of them occurs at the
  # `llama-cpp-qwen4exp` pin -- measured, see the decode proof's server log.
  kv_mib=$(grep -Eio '(KV self size|KV cache size|kv size) *= *[0-9]+' "${log}" 2>/dev/null | grep -Eo '[0-9]+' | tail -1)
  if [[ -n ${kv_mib} ]]; then
    origin="server-reported"
    kv_bytes=$((kv_mib * 1024 * 1024))
  elif [[ ${KV_BYTES_PER_TOKEN} -gt 0 ]]; then
    origin="KV_BYTES_PER_TOKEN"
    kv_mib=0
    kv_bytes=$((CTX_TOTAL * KV_BYTES_PER_TOKEN))
    say "llama-server reported NO KV size; falling back to the supplied KV_BYTES_PER_TOKEN=${KV_BYTES_PER_TOKEN}"
  else
    fail "${E_KV_UNREPORTED}" \
      "llama-server wrote no KV size to ${log} and KV_BYTES_PER_TOKEN is 0, so NOTHING bounds the KV cache. This is the 128-GiB-on-a-119-GB-box defect: the model is resident, the ladder is about to configure ${CTX_TOTAL} tokens over ${MAX_CONCURRENCY} slots, and no check carries a KV term. Measure the per-token cost on a leased load and pass --kv-bytes-per-token; do not guess one"
  fi
  need=$((weights + kv_bytes))
  say "KV term source      ${origin}"
  say "KV bytes            ${kv_bytes} (${kv_mib} MiB as reported)"
  say "weights+KV          ${need} bytes against budget ${budget} (source: ${source_name})"
  [[ ${need} -le ${budget} ]] || fail "${E_KV_OVER_BUDGET}" \
    "the ${origin} KV term is ${kv_bytes} bytes; with ${weights} bytes of weights that is ${need} bytes, over the ${MEM_FRACTION}% budget of ${total}"
}

wait_for_ready() {  # bounded; the lock is held while this runs
  local waited code
  waited=0
  while :; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      fail "${E_SERVER_NOT_READY}" "llama-server exited before it answered /health; see ${OUT}/server.log"
    fi
    code=$("${CURL_BIN}" -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)
    [[ -n ${code} ]] || code=000
    if [[ ${code} == 200 ]]; then
      say "server READY after ${waited}s"
      return 0
    fi
    if [[ ${waited} -ge ${READY_TIMEOUT_SECONDS} ]]; then
      fail "${E_SERVER_NOT_READY}" \
        "llama-server did not answer /health within ${READY_TIMEOUT_SECONDS}s (last code ${code}); refusing to wait on a held lock"
    fi
    sleep "${READY_POLL_INTERVAL}"
    waited=$((waited + READY_POLL_INTERVAL))
  done
}

start_clock_window() {  # $1 stem
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/bench/gpu_clock_state.py" sample \
    --output "$1.clock.jsonl" --summary "$1.clock.json" \
    --interval "${CLOCK_INTERVAL}" --max-duration "${CLOCK_MAX_SECONDS}" \
    > "$1.clock.log" 2>&1 &
  CLOCK_PID=$!
}

stop_clock_window() {  # $1 stem; the summary exists only once the sampler STOPS
  local waited
  [[ -n ${CLOCK_PID} ]] || return 0
  kill "${CLOCK_PID}" 2>/dev/null
  waited=0
  while kill -0 "${CLOCK_PID}" 2>/dev/null; do
    [[ ${waited} -ge 30 ]] && { kill -9 "${CLOCK_PID}" 2>/dev/null; break; }
    sleep 1
    waited=$((waited + 1))
  done
  CLOCK_PID=""
  # A window that turns out unusable still leaves its evidence on disk and
  # yields no number.  It never discards the leg that already ran.
  if [[ ! -s $1.clock.json ]]; then
    say "CLOCK WINDOW UNUSABLE for $1: no summary written; this leg carries NO clock attribution"
    return "${E_CLOCK_UNUSABLE}"
  fi
  return 0
}

run_leg() {  # run_leg CONCURRENCY PROMPTS REP -> 0 accepted
  local conc prompts rep stem rc
  conc=$1; prompts=$2; rep=$3
  stem="${OUT}/llamacpp-c${conc}-r${rep}"
  say "=== leg c=${conc} prompts=${prompts} rep=${rep} ==="
  start_clock_window "${stem}"
  "${TIMEOUT_BIN}" "${LEG_TIMEOUT_SECONDS}" \
    "${TIMEV_BIN}" -v -o "${stem}.time" \
    "${CLIENT}" bench serve \
      --backend openai \
      --base-url "http://127.0.0.1:${PORT}" \
      --endpoint /v1/completions \
      --model gate \
      --tokenizer "${TOKENIZER}" \
      --dataset-name custom \
      --dataset-path "${CORPUS_DIR}/c${conc}-r${rep}.jsonl" \
      --custom-output-len "${OUTPUT_LEN}" \
      --skip-chat-template \
      --disable-shuffle \
      --num-prompts "${prompts}" \
      --max-concurrency "${conc}" \
      --request-rate inf \
      --num-warmups "${conc}" \
      --ready-check-timeout-sec 0 \
      --seed 0 \
      --ignore-eos \
      --temperature 0 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,99 \
      --save-result --save-detailed \
      --result-dir "${OUT}" \
      --result-filename "$(basename "${stem}").json" \
      --disable-tqdm \
      > "${stem}.stdout" 2> "${stem}.stderr"
  rc=$?
  stop_clock_window "${stem}" || say "  (clock window unusable, recorded)"
  if [[ ${rc} -ne 0 ]]; then
    say "leg c=${conc} rep=${rep} FAILED rc=${rc} (124 = the ${LEG_TIMEOUT_SECONDS}s leg timeout fired)"
    return "${E_LEG_FAILED}"
  fi
  # The wrapper's OWN output is the proof the wrapper ran.  `/usr/bin/time`
  # absent would have made the whole leg not run; `/usr/bin/time` present but
  # silent would leave a result file with no memory axis, and the memory axis is
  # a gate requirement, not a nicety.
  grep -q 'Maximum resident set size' "${stem}.time" 2>/dev/null || {
    say "leg c=${conc} rep=${rep} FAILED: ${stem}.time carries no rusage; the time wrapper did not run"
    return "${E_LEG_FAILED}"
  }
  [[ -s ${stem}.json ]] || {
    say "leg c=${conc} rep=${rep} FAILED: the client wrote no result file"
    return "${E_LEG_FAILED}"
  }
  say "leg c=${conc} rep=${rep} OK"
  return 0
}

# Every published figure comes from HERE, never from a human eye on a JSON file.
# Prefill and decode are reported SEPARATELY because the gate asks for that:
# input length splits them and an aggregate hides which lever moved.
summarize() {
  # `env`, not a bare assignment prefix: LADDER, INPUT_LEN, OUTPUT_LEN and
  # REPETITIONS are `readonly` here on purpose -- they are the published grid --
  # and bash refuses to shadow a readonly name in a command prefix.
  env OUT="${OUT}" LADDER="${LADDER}" REPETITIONS="${REPETITIONS}" \
    INPUT_LEN="${INPUT_LEN}" OUTPUT_LEN="${OUTPUT_LEN}" \
    "${PYTHON_BIN}" - <<'PYEOF'
import json, os, pathlib, statistics, sys

out = pathlib.Path(os.environ["OUT"])
ladder = [tuple(int(x) for x in p.split(":")) for p in os.environ["LADDER"].split()]
reps = int(os.environ["REPETITIONS"])
inlen, outlen = os.environ["INPUT_LEN"], os.environ["OUTPUT_LEN"]

RSS_TAG = "Maximum resident set size (kbytes):"


def rss_kb(path):
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if RSS_TAG in line:
            return int(line.rsplit(":", 1)[1].strip())
    return None


def clock(path):
    if not path.is_file() or not path.stat().st_size:
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


rows, gaps = [], []
for conc, prompts in ladder:
    legs = []
    for rep in range(1, reps + 1):
        stem = out / f"llamacpp-c{conc}-r{rep}"
        result = stem.with_suffix(".json")
        if not result.is_file():
            gaps.append(f"c{conc} rep{rep}: no result file")
            continue
        try:
            record = json.loads(result.read_text())
        except json.JSONDecodeError:
            gaps.append(f"c{conc} rep{rep}: result file is not JSON")
            continue
        if record.get("completed") != prompts:
            gaps.append(
                f"c{conc} rep{rep}: completed={record.get('completed')} of {prompts}"
            )
        legs.append((record, rss_kb(stem.with_suffix(".time")), clock(pathlib.Path(f"{stem}.clock.json"))))
    if not legs:
        rows.append((conc, prompts, None))
        continue
    rows.append((conc, prompts, legs))

lines = [
    "# llama.cpp arm, MODEL-MM-QWEN4-EXP G4 concurrency ladder",
    "",
    f"Oracle pin `035e2273` (ggml-org/llama.cpp PR #27742). {inlen} in / {outlen} out, "
    f"{reps} repetitions, medians of per-rung repetitions. Timed requests issued only "
    "by the pinned `vllm bench serve`, the same client the vllm.cpp arm uses.",
    "",
    "**This table is a DENOMINATOR, not a result.** It carries no vllm.cpp arm and no "
    "ratio. #27742 is mask-only, so a LONG-CONTEXT decode comparison against it is "
    "partly their mask approach and not purely our gather; short-context and "
    "high-concurrency cells are unaffected. State which side ran what.",
    "",
    "**The configuration the vllm.cpp cell beside this one MUST have run.** This "
    "denominator is TEXT-ONLY: `/props` on the server that produced it reports "
    "`\"modalities\":{\"vision\":false,\"video\":false,\"audio\":false}`, while "
    "`MODEL-MM-QWEN4-EXP` is a multimodal port. An arm that loads a vision tower "
    "does strictly more work per request, so a ratio taken against it measures a "
    "configuration difference and reads as a performance one. Pair this table only "
    "with a vllm.cpp run on the SAME `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S "
    "artifact, text-only prompts, no vision or video tower resident, 1,024 in / "
    "128 out, `--disable-shuffle`, the c<C>-r<R> corpus partition for the cell, and "
    "warmups equal to the concurrency. Anything else is a different measurement "
    "wearing this table's row labels.",
    "",
    "| Concurrency | prompts | decode tok/s | total tok/s | median TTFT ms (prefill) | "
    "median TPOT ms (decode) | median ITL ms | peak RSS MiB | SM clock med MHz | spread % |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for conc, prompts, legs in rows:
    if legs is None:
        lines.append(f"| {conc} | {prompts} | MISSING | MISSING | - | - | - | - | - | - |")
        continue
    def axis(key):
        return med([r.get(key) for r, _, _ in legs])
    rss = med([kb for _, kb, _ in legs])
    clocks = [c for _, _, c in legs if c]
    cmed = med([c.get("sm_clock_median_mhz") for c in clocks]) if clocks else None
    cspread = med([c.get("sm_clock_spread_pct") for c in clocks]) if clocks else None
    def fmt(v, digits=2):
        return "-" if v is None else f"{v:.{digits}f}"
    lines.append(
        f"| {conc} | {prompts} | {fmt(axis('output_throughput'))} | "
        f"{fmt(axis('total_token_throughput'))} | {fmt(axis('median_ttft_ms'))} | "
        f"{fmt(axis('median_tpot_ms'))} | {fmt(axis('median_itl_ms'))} | "
        f"{'-' if rss is None else f'{rss / 1024:.1f}'} | {fmt(cmed, 0)} | {fmt(cspread)} |"
    )

if any(c is None for _, _, legs in rows if legs for _, _, c in legs):
    lines += [
        "",
        "**At least one leg carries NO clock attribution.** Inside an `rc` lease the SM "
        "clock can be sampled and not pinned, so a cell without a clock is not quotable "
        "against a cell taken at another clock, and the run states that rather than "
        "hiding it.",
    ]

(out / "ladder-summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
if gaps:
    print("SUMMARY_INCOMPLETE: " + "; ".join(gaps), file=sys.stderr)
    sys.exit(1)
PYEOF
}

execute() {
  local point conc prompts rep rc lock_fd
  mkdir -p "${OUT}" || fail "${E_USAGE}" "cannot create OUT='${OUT}'"

  # The mutex.  Bounded: `flock -w`.  An unbounded wait here is a hang that
  # nobody can see, holding nothing and blocking the next holder's diagnosis.
  # On a FLEET device this runs INSIDE an `rc` lease and never instead of one --
  # the lease decides who gets the box, the mutex serialises this job's work.
  exec {lock_fd}>"${GPU_LOCK}" || fail "${E_LOCK_TIMEOUT}" "cannot open GPU_LOCK='${GPU_LOCK}'"
  "${FLOCK_BIN}" -w "${LOCK_WAIT_SECONDS}" "${lock_fd}" || fail "${E_LOCK_TIMEOUT}" \
    "could not take '${GPU_LOCK}' within ${LOCK_WAIT_SECONDS}s"
  say "GPU mutex held: ${GPU_LOCK} (rc lease: ${RC_JOB_ID:-none detected})"

  "${SERVER}" \
    --model "${MODEL}" \
    --alias gate \
    --host 127.0.0.1 --port "${PORT}" \
    -ngl "${NGL}" \
    -c "${CTX_TOTAL}" \
    -np "${MAX_CONCURRENCY}" \
    --cont-batching \
    --flash-attn on \
    --metrics \
    > "${OUT}/server.log" 2>&1 &
  SERVER_PID=$!
  say "llama-server pid ${SERVER_PID}"
  wait_for_ready
  assert_reported_kv_fits "${OUT}/server.log"

  "${PYTHON_BIN}" "${REPO_ROOT}/tools/bench/sample_process_memory.py" \
    --pid "${SERVER_PID}" --output "${OUT}/server-memory.jsonl" \
    --interval "${MEM_INTERVAL}" --include-gpu \
    > "${OUT}/server-memory.log" 2>&1 &
  MEM_PID=$!

  # Repetition-major, so the ladder is walked three times rather than each rung
  # being repeated three times in place.  A rung that owns one quiet window is
  # not comparable to a rung that owned three.
  for rep in $(seq 1 "${REPETITIONS}"); do
    for point in ${LADDER}; do
      conc=${point%%:*}
      prompts=${point##*:}
      run_leg "${conc}" "${prompts}" "${rep}"
      rc=$?
      [[ ${rc} -eq 0 ]] || fail "${rc}" "leg c=${conc} rep=${rep} did not produce a usable result"
    done
  done
  say "LADDER DONE: ${REPETITIONS} repetitions of ${LADDER}"
  summarize || say "SUMMARY_INCOMPLETE (see stderr); the legs are on disk"
}

if [[ ${MODE} == summarize ]]; then
  [[ -d ${OUT} ]] || fail "${E_USAGE}" "--summarize needs an existing --out directory; '${OUT}' does not exist"
  summarize
  exit $?
fi
say "mode          ${MODE}"
say "ladder        ${LADDER}  (in ${INPUT_LEN} / out ${OUTPUT_LEN}, ${REPETITIONS} reps)"
say "oracle pin    035e22731a7fd70b9854b3a2d64ec68e9b1a45d3 (ggml-org/llama.cpp PR #27742)"
preflight
if [[ ${MODE} == self-check ]]; then
  say "SELF_CHECK_OK"
  exit 0
fi
execute
say "SERIES_DONE"
