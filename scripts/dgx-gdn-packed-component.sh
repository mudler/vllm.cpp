#!/usr/bin/env bash
# Production-build GDN packed-default versus rollback component driver.
# Exact execution contract: .agents/specs/gdn-packed-decode.md G3.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  dgx-gdn-packed-component.sh --dry-run --vllm-cpp-sha SHA
  dgx-gdn-packed-component.sh --execute --snapshot DIR --source-corpus DIR \
    --evidence DIR --build-dir DIR --configure-log FILE --vllm-cpp-sha SHA \
    [--client PATH] [--port N]
  dgx-gdn-packed-component.sh --diagnostic-c16 --snapshot DIR --source-corpus DIR \
    --evidence DIR --build-dir DIR --configure-log FILE --vllm-cpp-sha SHA \
    [--client PATH] [--port N]
EOF
}

mode=""
snapshot=""
source_corpus=""
evidence=""
build_dir=""
configure_log=""
client="${HOME}/venvs/vllm-oracle/bin/vllm"
vllm_cpp_sha=""
port=8001
num_blocks=4736
max_num_seqs=32
max_num_batched_tokens=2048
warmup_label=w0

while (($#)); do
  case "$1" in
    --dry-run|--execute|--diagnostic-c16)
      [[ -z ${mode} ]] || { echo "choose exactly one mode" >&2; exit 2; }
      mode=${1#--}
      shift
      ;;
    --snapshot) snapshot=${2:?}; shift 2 ;;
    --source-corpus) source_corpus=${2:?}; shift 2 ;;
    --evidence) evidence=${2:?}; shift 2 ;;
    --build-dir) build_dir=${2:?}; shift 2 ;;
    --configure-log) configure_log=${2:?}; shift 2 ;;
    --client) client=${2:?}; shift 2 ;;
    --vllm-cpp-sha) vllm_cpp_sha=${2:?}; shift 2 ;;
    --port) port=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n ${mode} ]] || { usage; exit 2; }
[[ -n ${vllm_cpp_sha} ]] || { echo "--vllm-cpp-sha is required" >&2; exit 2; }
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ${mode} == dry-run ]]; then
  exec python3 "${repo_root}/tools/bench/gdn_packed_component.py" plan \
    --vllm-cpp-sha "${vllm_cpp_sha}"
fi

[[ ${mode} == execute || ${mode} == diagnostic-c16 ]] || { usage; exit 2; }
[[ -n ${evidence} ]] || { echo "--evidence is required" >&2; exit 2; }
[[ -n ${snapshot} && -d ${snapshot} ]] || {
  echo "--snapshot must name the pinned Qwen3.6-27B directory" >&2
  exit 2
}
[[ -n ${source_corpus} && -d ${source_corpus} ]] || {
  echo "--source-corpus must name the component corpus directory" >&2
  exit 2
}
[[ -n ${build_dir} && -f ${build_dir}/CMakeCache.txt ]] || {
  echo "--build-dir must name a configured production CMake tree" >&2
  exit 2
}
[[ -n ${configure_log} && -s ${configure_log} ]] || {
  echo "--configure-log must name the non-empty production configure log" >&2
  exit 2
}
[[ -x ${client} ]] || { echo "pinned vLLM client is not executable: ${client}" >&2; exit 2; }
[[ -x /usr/bin/nvidia-smi ]] || { echo "pinned /usr/bin/nvidia-smi is absent" >&2; exit 2; }
expected_corpus="${evidence}/corpus/27"
[[ $(realpath -m "${source_corpus}") == "$(realpath -m "${expected_corpus}")" ]] || {
  echo "--source-corpus must be ${expected_corpus}" >&2
  exit 2
}
[[ $(git -C "${repo_root}" rev-parse HEAD) == "${vllm_cpp_sha}" ]] || {
  echo "worktree HEAD does not match --vllm-cpp-sha" >&2
  exit 2
}
[[ -z $(git -C "${repo_root}" status --porcelain) ]] || {
  echo "vllm.cpp source tree is not completely clean" >&2
  exit 2
}
cmake_home=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${build_dir}/CMakeCache.txt")
[[ -n ${cmake_home} && $(realpath -e "${cmake_home}") == "$(realpath -e "${repo_root}")" ]] || {
  echo "build CMAKE_HOME_DIRECTORY does not match this worktree" >&2
  exit 2
}
grep -Fxq 'VLLM_CPP_BENCH_PROFILE_CONTROL:BOOL=OFF' "${build_dir}/CMakeCache.txt" || {
  echo "component timing requires VLLM_CPP_BENCH_PROFILE_CONTROL=OFF" >&2
  exit 2
}
if [[ ${mode} == diagnostic-c16 ]]; then
  [[ $(basename "${evidence}") == *diagnostic-c16* ]] || {
    echo "--evidence basename must contain diagnostic-c16" >&2
    exit 2
  }
  if compgen -G "${evidence}/component-*.json" >/dev/null 2>&1; then
    echo "diagnostic mode refuses a directory holding component-*.json" >&2
    exit 2
  fi
fi

mkdir -p "${evidence}"
plan="${evidence}/component-plan.json"
run_log="${evidence}/component-run.log"
order_log="${evidence}/component-order.log"
summary="${evidence}/component-summary.json"
manifest="${evidence}/component-manifest.json"
status="${evidence}/component-status.json"
[[ ! -e ${plan} && ! -e ${run_log} && ! -e ${order_log} && ! -e ${summary} && \
   ! -e ${manifest} && ! -e ${status} ]] || {
  echo "refusing to overwrite component evidence in ${evidence}" >&2
  exit 1
}
python3 "${repo_root}/tools/bench/gdn_packed_component.py" plan \
  --vllm-cpp-sha "${vllm_cpp_sha}" --output "${plan}" >/dev/null
exec 3>&1 4>&2
exec >"${run_log}" 2>&1

benchmark_system_path=/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin
benchmark_common_env=(
  "HOME=${HOME}"
  "PYTHONPATH=${repo_root}"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "TMPDIR=/tmp"
  "TZ=UTC"
)
benchmark_clean_env=(
  /usr/bin/env -i
  "${benchmark_common_env[@]}"
  "PATH=${benchmark_system_path}"
)
fixture="${repo_root}/tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
native_plan="${evidence}/native-plan-must-not-exist.json"
[[ -s ${fixture} ]] || { echo "frozen FP4 plan fixture is absent" >&2; exit 2; }
[[ ! -e ${native_plan} ]] || { echo "native plan target must remain absent" >&2; exit 2; }
plan_env=(
  "VT_FP4_AUTOTUNE=1"
  "VT_FP4_AUTOTUNE_CACHE_PATH=${native_plan}"
  "VT_FP4_AUTOTUNE_CACHE_READONLY=1"
  "VT_FP4_AUTOTUNE_DELAY_US=5000"
  "VT_FP4_AUTOTUNE_VERBOSE=1"
  "VT_FP4_FLASHINFER_CACHE_PATH=${fixture}"
  "VT_FP4_FULL_TACTICS=1"
  "VT_FP4_PERSISTENT_CACHE=1"
  "VT_FP4_PLAN_CACHE=1"
  "VT_FP4_PRE_SERVE_WARMUP=1"
)

vllm_python="$(dirname "${client}")/python"
[[ -x ${vllm_python} ]] || { echo "vLLM oracle Python is absent" >&2; exit 2; }
execution_dir="${evidence}/execution"
mkdir -p "${execution_dir}"
oracle_manifest="${execution_dir}/27-oracle.json"
"${benchmark_clean_env[@]}" "${vllm_python}" \
  "${repo_root}/tools/bench/online_gate.py" record-oracle \
  --output "${oracle_manifest}" --client "${client}"

build_command="${execution_dir}/27-build-command.txt"
build_log="${execution_dir}/27-build.log"
build_cmd=(
  cmake --build "${build_dir}"
  --target server test_qwen27_paged_engine
  --parallel "$(nproc)"
)
printf '%q ' "${build_cmd[@]}" >"${build_command}"
printf '\n' >>"${build_command}"
if ! "${build_cmd[@]}" >"${build_log}" 2>&1; then
  cat "${build_log}" >&2
  exit 1
fi

execution_manifest="${execution_dir}/27-component.json"
"${benchmark_clean_env[@]}" "${plan_env[@]}" \
  python3 "${repo_root}/tools/bench/online_gate.py" record-execution \
  --output "${execution_manifest}" \
  --model-key 27 \
  --vllm-cpp-sha "${vllm_cpp_sha}" \
  --build-dir "${build_dir}" \
  --client "${client}" \
  --snapshot "${snapshot}" \
  --configure-log "${configure_log}" \
  --build-command "${build_command}" \
  --build-log "${build_log}" \
  --oracle-manifest "${oracle_manifest}" \
  --port "${port}" \
  --num-blocks "${num_blocks}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --profile-control off

vllm_corpus="${source_corpus}/vllm"
if [[ ! -f ${vllm_corpus}/manifest.json ]]; then
  "${benchmark_clean_env[@]}" python3 \
    "${repo_root}/tools/bench/online_gate.py" prepare-corpus \
    --source "${source_corpus}" --output "${vllm_corpus}" --model-key 27 \
    --repetitions 1 2 3 4 5
fi

spid=""
mpid=""
cleanup_server() {
  if [[ -n ${spid} ]] && kill -0 "${spid}" 2>/dev/null; then
    kill -TERM -- "-${spid}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "${spid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${spid}" 2>/dev/null; then
      kill -KILL -- "-${spid}" 2>/dev/null || true
    fi
    wait "${spid}" 2>/dev/null || true
  fi
  spid=""
  if [[ -n ${mpid} ]]; then
    wait "${mpid}" 2>/dev/null || true
  fi
  mpid=""
}
handle_signal() {
  local status=$1
  exit "${status}"
}
trap cleanup_server EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

record_order() {
  printf '%s\n' "$*" | tee -a "${order_log}"
}

gpu_idle() {
  "${benchmark_clean_env[@]}" python3 \
    "${repo_root}/tools/bench/gdn_packed_component.py" probe-gpu-idle >/dev/null
}

wait_gpu_idle() {
  for _ in $(seq 1 120); do
    gpu_idle && return 0
    sleep 1
  done
  echo "GPU did not return idle" >&2
  return 1
}

mem_available_kib() {
  awk '/^MemAvailable:/{print $2}' /proc/meminfo
}

drop_caches() {
  local output=$1
  "${benchmark_clean_env[@]}" python3 -m tools.bench.drop_file_cache \
    --root "${snapshot}" \
    --root "${source_corpus}" \
    --root "${build_dir}/examples/server" \
    --root "${client}" \
    --output "${output}"
}

wait_ready() {
  local log=$1
  for _ in $(seq 1 360); do
    if curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${spid}" 2>/dev/null; then
      echo "server exited during readiness" >&2
      tail -n 100 "${log}" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "server readiness timed out" >&2
  return 1
}

server_command() {
  local arm=$1
  local -n output=$2
  local packed_value=0
  [[ ${arm} == packed ]] && packed_value=1
  output=(
    "${benchmark_clean_env[@]}"
    "${plan_env[@]}"
    "VT_GDN_PACKED_DECODE=${packed_value}"
    "${build_dir}/examples/server"
    --model "${snapshot}"
    --port "${port}"
    --num-blocks "${num_blocks}"
    --max-num-seqs "${max_num_seqs}"
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --no-enable-prefix-caching
    --served-model-name gate
  )
}

run_model_gate() {
  local arm=$1
  local packed_value=0
  [[ ${arm} == packed ]] && packed_value=1
  local gate_dir="${evidence}/model-gates"
  local gate_log="${gate_dir}/${arm}.log"
  local gate_command="${gate_dir}/${arm}-command.txt"
  local gate_status="${gate_dir}/${arm}.json"
  local gate_binary="${build_dir}/tests/test_qwen27_paged_engine"
  local -a command=(
    "${benchmark_clean_env[@]}"
    "${plan_env[@]}"
    "VT_GDN_PACKED_DECODE=${packed_value}"
    "${gate_binary}"
  )
  mkdir -p "${gate_dir}"
  [[ -x ${gate_binary} ]] || { echo "model-gate binary is not executable" >&2; return 1; }
  printf '%q ' "${command[@]}" >"${gate_command}"
  printf '\n' >>"${gate_command}"
  "${command[@]}" >"${gate_log}" 2>&1
  cat "${gate_log}"
  python3 "${repo_root}/tools/bench/online_gate.py" record-model-gate \
    --output "${gate_status}" --log "${gate_log}" --model-key 27 \
    --test-name test_qwen27_paged_engine --vllm-cpp-sha "${vllm_cpp_sha}"
  wait_gpu_idle
  record_order "model_gate_complete arm=${arm}"
}

run_leg() {
  local concurrency=$1 arm=$2 repetition=$3
  local memory_dir="${evidence}/memory/27/c${concurrency}/${arm}"
  local return_dir="${evidence}/memory-return/27/c${concurrency}/${arm}"
  local cache_dir="${evidence}/cache-drop/27/c${concurrency}/${arm}"
  local log_dir="${evidence}/logs/27/c${concurrency}/${arm}"
  local thermal_dir="${evidence}/thermal/27/c${concurrency}/${arm}"
  local preflight_dir="${evidence}/preflight/27/c${concurrency}/${arm}"
  local before_cache="${cache_dir}/r${repetition}-before.json"
  local after_cache="${cache_dir}/r${repetition}-after.json"
  local log="${log_dir}/r${repetition}-server.log"
  local command_file="${log_dir}/r${repetition}-server-command.txt"
  local preflight_command_file="${preflight_dir}/r${repetition}-stream-command.txt"
  local baseline final
  local -a command preflight_command
  mkdir -p "${memory_dir}" "${return_dir}" "${cache_dir}" "${log_dir}" \
    "${thermal_dir}" "${preflight_dir}"

  gpu_idle || { echo "GPU busy before c${concurrency}/${arm}/r${repetition}" >&2; return 1; }
  drop_caches "${before_cache}"
  baseline=$(mem_available_kib)
  server_command "${arm}" command
  printf '%q ' "${command[@]}" >"${command_file}"
  printf '\n' >>"${command_file}"
  record_order "leg_begin concurrency=${concurrency} arm=${arm} repetition=${repetition}"
  setsid "${command[@]}" >"${log}" 2>&1 &
  spid=$!
  python3 "${repo_root}/tools/bench/sample_process_memory.py" \
    --pid "${spid}" --output "${memory_dir}/r${repetition}.samples.jsonl" \
    --interval 0.1 --include-gpu >"${memory_dir}/r${repetition}.summary.json" &
  mpid=$!
  wait_ready "${log}"
  grep -Fq 'HTTP worker pool 36 fixed' "${log}"
  /usr/bin/nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER \
    >"${thermal_dir}/r${repetition}-before.txt"
  preflight_command=(
    "${benchmark_clean_env[@]}"
    python3 "${repo_root}/tools/bench/run_serve_low.py" stream
    --url "http://127.0.0.1:${port}/v1/completions"
    --corpus "${source_corpus}/c1-r${repetition}.jsonl"
    --output-len 128
    --minimum-spread 0.05
    --output "${preflight_dir}/r${repetition}-stream.json"
  )
  printf '%q ' "${preflight_command[@]}" >"${preflight_command_file}"
  printf '\n' >>"${preflight_command_file}"
  "${preflight_command[@]}"
  "${benchmark_clean_env[@]}" python3 "${repo_root}/tools/bench/online_gate.py" bench \
    --client "${client}" --tokenizer "${snapshot}" --evidence "${evidence}" \
    --model-key 27 --engine ours --base-url "http://127.0.0.1:${port}" \
    --concurrency "${concurrency}" --repetition "${repetition}" \
    --artifact-tag "gdn-${arm}"
  /usr/bin/nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER \
    >"${thermal_dir}/r${repetition}-after.txt"
  cleanup_server
  wait_gpu_idle
  drop_caches "${after_cache}"
  final=$(mem_available_kib)
  python3 "${repo_root}/tools/bench/online_gate.py" record-memory-return \
    --output "${return_dir}/r${repetition}.json" \
    --baseline-kib "${baseline}" --final-kib "${final}" \
    --tolerance-kib 1048576 --before-cache-drop-report "${before_cache}" \
    --after-cache-drop-report "${after_cache}" --gpu-idle
  record_order "leg_end concurrency=${concurrency} arm=${arm} repetition=${repetition}"
}

# Single discarded cold-start warmup leg (one per arm), run FIRST before the
# whole timed series at the smallest concurrency. Its bench output is thrown
# away — the leg exists only to absorb the one-time whole-series cold-first-leg
# draw so it never lands on a timed leg. The bench is failure-tolerant (the
# cold-start HTTP 500 is exactly what we are absorbing).
run_warmup_leg() {
  local arm=$1
  local concurrency=2
  local warmup_dir="${evidence}/warmup/27"
  local log_dir="${warmup_dir}/${arm}-logs"
  local -a command
  mkdir -p "${warmup_dir}" "${log_dir}"
  gpu_idle || { echo "GPU busy before warmup ${arm}" >&2; return 1; }
  drop_caches "${warmup_dir}/${arm}-before-cache.json"
  server_command "${arm}" command
  record_order "warmup_leg_begin arm=${arm} label=${warmup_label}"
  setsid "${command[@]}" >"${log_dir}/server.log" 2>&1 &
  spid=$!
  wait_ready "${log_dir}/server.log"
  "${benchmark_clean_env[@]}" python3 "${repo_root}/tools/bench/online_gate.py" bench \
    --client "${client}" --tokenizer "${snapshot}" --evidence "${evidence}" \
    --model-key 27 --engine ours --base-url "http://127.0.0.1:${port}" \
    --concurrency "${concurrency}" --repetition 1 \
    --artifact-tag "gdn-${arm}-${warmup_label}" || true
  cleanup_server
  wait_gpu_idle
  # The throwaway bench output must never contaminate the timed raw directory.
  rm -f "${evidence}/raw/27/ours/c${concurrency}-r1-gdn-${arm}-${warmup_label}.json"
  printf '{"arm":"%s","label":"%s","discarded":true,"server_log":"%s"}\n' \
    "${arm}" "${warmup_label}" "${log_dir}/server.log" \
    >"${warmup_dir}/w0-${arm}.json"
  record_order "warmup_leg_end arm=${arm} label=${warmup_label}"
}

# >>> diagnostic-c16 flow (bounded reproduction of the c16 packed HTTP 500).
# Packed arm ONLY, concurrency 16, three fresh servers under one GPU lock. No
# model gates, no 2/16 sweep, no component finalization: this checkpoint only
# captures the engine-fatal root cause the timed component leg drops. The
# server carries VT_GDN_DIAG_STEP_LOG=1 so its step geometry is on the record.
diagnostic_server_command() {
  local arm=$1
  local -n diag_out=$2
  local -a base_command
  server_command "${arm}" base_command
  # env -i only consumes VAR=val up to the first non-assignment token, so splice
  # VT_GDN_DIAG_STEP_LOG=1 in immediately BEFORE the server binary by rebuilding
  # the array (never string-substituting the assembled command line).
  diag_out=()
  local spliced=0 token
  for token in "${base_command[@]}"; do
    if [[ ${spliced} -eq 0 && ${token} == "${build_dir}/examples/server" ]]; then
      diag_out+=("VT_GDN_DIAG_STEP_LOG=1")
      spliced=1
    fi
    diag_out+=("${token}")
  done
}

run_diagnostic_leg() {
  local diag_root="${evidence}/diagnostic/c16/packed"
  local log_dir="${diag_root}/logs"
  local thermal_dir="${diag_root}/thermal"
  local preflight_dir="${diag_root}/preflight"
  mkdir -p "${diag_root}" "${log_dir}" "${thermal_dir}" "${preflight_dir}"
  local -a rep_outcomes=()
  local rep
  for rep in 1 2 3; do
    local bench_failed=0
    local log="${log_dir}/r${rep}-server.log"
    local -a command
    gpu_idle || { echo "GPU busy before diagnostic r${rep}" >&2; return 1; }
    drop_caches "${diag_root}/r${rep}-before-cache.json"
    diagnostic_server_command packed command
    printf '%q ' "${command[@]}" >"${log_dir}/r${rep}-server-command.txt"
    printf '\n' >>"${log_dir}/r${rep}-server-command.txt"
    record_order "diagnostic_leg_begin repetition=${rep}"
    setsid "${command[@]}" >"${log}" 2>&1 &
    spid=$!
    wait_ready "${log}"
    /usr/bin/nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER \
      >"${thermal_dir}/r${rep}-before.txt"
    "${benchmark_clean_env[@]}" python3 \
      "${repo_root}/tools/bench/run_serve_low.py" stream \
      --url "http://127.0.0.1:${port}/v1/completions" \
      --corpus "${source_corpus}/c1-r${rep}.jsonl" \
      --output-len 128 --minimum-spread 0.05 \
      --output "${preflight_dir}/r${rep}-stream.json"
    "${benchmark_clean_env[@]}" python3 \
      "${repo_root}/tools/bench/online_gate.py" bench \
      --client "${client}" --tokenizer "${snapshot}" --evidence "${evidence}" \
      --model-key 27 --engine ours --base-url "http://127.0.0.1:${port}" \
      --concurrency 16 --repetition "${rep}" --artifact-tag "gdn-packed" \
      || bench_failed=1
    /usr/bin/nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER \
      >"${thermal_dir}/r${rep}-after.txt"
    if [[ ${bench_failed} -eq 1 ]]; then
      "${benchmark_clean_env[@]}" python3 \
        "${repo_root}/tools/bench/run_serve_low.py" diagnostic-error-body \
        --url "http://127.0.0.1:${port}/v1/completions" \
        --corpus "${source_corpus}/c16-r${rep}.jsonl" \
        --output "${diag_root}/r${rep}-error-body.json"
      rep_outcomes+=("{\"repetition\":${rep},\"bench_failed\":true}")
    else
      rep_outcomes+=("{\"repetition\":${rep},\"bench_failed\":false}")
    fi
    cleanup_server
    wait_gpu_idle
    record_order "diagnostic_leg_end repetition=${rep}"
  done
  diagnostic_reps=$(IFS=,; echo "${rep_outcomes[*]}")
}

if [[ ${mode} == diagnostic-c16 ]]; then
  exec 9>/tmp/gpu
  flock 9
  record_order "gpu_lock_acquired path=/tmp/gpu"
  gpu_idle || { echo "GPU is busy despite acquiring /tmp/gpu" >&2; exit 1; }
  diagnostic_reps=""
  run_diagnostic_leg
  flock -u 9
  record_order "gpu_lock_released path=/tmp/gpu"
  printf '{"diagnostic":true,"mode":"diagnostic-c16","repetitions":[%s]}\n' \
    "${diagnostic_reps}" >"${evidence}/component-diagnostic.json"
  record_order "diagnostic_complete"
  exit 0
fi
# <<< diagnostic-c16 flow

python3 "${repo_root}/tools/bench/gdn_packed_component.py" validate-corpus \
  --evidence "${evidence}" --vllm-cpp-sha "${vllm_cpp_sha}" >/dev/null
record_order "corpus_validated"

exec 9>/tmp/gpu
flock 9
record_order "gpu_lock_acquired path=/tmp/gpu"
gpu_idle || { echo "GPU is busy despite acquiring /tmp/gpu" >&2; exit 1; }
run_model_gate packed
run_model_gate rollback
python3 "${repo_root}/tools/bench/gdn_packed_component.py" validate-model-gates \
  --evidence "${evidence}" --vllm-cpp-sha "${vllm_cpp_sha}" >/dev/null
record_order "model_gates_validated"
# One discarded cold-start warmup pair, run first before the whole timed series.
run_warmup_leg packed
run_warmup_leg rollback
for concurrency in 2 16; do
  run_leg "${concurrency}" packed 1
  run_leg "${concurrency}" rollback 1
  run_leg "${concurrency}" rollback 2
  run_leg "${concurrency}" packed 2
  run_leg "${concurrency}" packed 3
  run_leg "${concurrency}" rollback 3
  run_leg "${concurrency}" rollback 4
  run_leg "${concurrency}" packed 4
  run_leg "${concurrency}" packed 5
  run_leg "${concurrency}" rollback 5
done
record_order "gpu_series_complete"
flock -u 9
record_order "gpu_lock_released path=/tmp/gpu"

exec 1>&3 2>&4
exec 3>&- 4>&-
python3 "${repo_root}/tools/bench/gdn_packed_component.py" finalize \
  --evidence "${evidence}" --vllm-cpp-sha "${vllm_cpp_sha}"
