#!/usr/bin/env bash
# Reproducible SERVE-GATE-ONLINE campaign driver.
#
# The accepted contract is .agents/specs/cuda-online-serving-gate.md. Timed
# requests are issued only by pinned vLLM `bench serve`; this script owns server
# lifecycle, interleaving, the one-model/one-lock boundary, memory return, and
# artifact capture.  --dry-run and --prepare-corpus never acquire /tmp/gpu;
# --trace-only performs the model gate plus the paired trace without the grid.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  dgx-online-serving.sh --dry-run [--claim-root DIR] [--client PATH] [--vllm-cpp-sha SHA]
  dgx-online-serving.sh --prepare-corpus --model 27|35 --source-corpus DIR --evidence DIR
  dgx-online-serving.sh --trace-only --model 27 --snapshot DIR --source-corpus DIR \
    --evidence DIR --build-dir DIR --configure-log FILE [--client PATH] [--port N] \
    [--trace-concurrency 2|16] [--gdn-ba-mode both] [--gdn-packed-mode both]
  dgx-online-serving.sh --execute --model 27|35 --snapshot DIR --source-corpus DIR \
    --evidence DIR --build-dir DIR --configure-log FILE [--client PATH] [--port N]
EOF
}

mode=""
model=""
snapshot=""
source_corpus=""
claim_root="${HOME}/work/vllm.cpp-online-gate"
evidence=""
build_dir=""
configure_log=""
client="${HOME}/venvs/vllm-oracle/bin/vllm"
vllm_cpp_sha=""
port=8001
num_blocks=4736
max_num_seqs=32
max_num_batched_tokens=""
trace_concurrency=16
gdn_ba_mode=""
gdn_packed_mode=""

while (($#)); do
  case "$1" in
    --dry-run|--prepare-corpus|--trace-only|--execute)
      [[ -z ${mode} ]] || { echo "choose exactly one mode" >&2; exit 2; }
      mode=${1#--}
      shift
      ;;
    --model) model=${2:?}; shift 2 ;;
    --snapshot) snapshot=${2:?}; shift 2 ;;
    --source-corpus) source_corpus=${2:?}; shift 2 ;;
    --claim-root) claim_root=${2:?}; shift 2 ;;
    --evidence) evidence=${2:?}; shift 2 ;;
    --build-dir) build_dir=${2:?}; shift 2 ;;
    --configure-log) configure_log=${2:?}; shift 2 ;;
    --client) client=${2:?}; shift 2 ;;
    --vllm-cpp-sha) vllm_cpp_sha=${2:?}; shift 2 ;;
    --port) port=${2:?}; shift 2 ;;
    --num-blocks) num_blocks=${2:?}; shift 2 ;;
    --trace-concurrency) trace_concurrency=${2:?}; shift 2 ;;
    --gdn-ba-mode) gdn_ba_mode=${2:?}; shift 2 ;;
    --gdn-packed-mode) gdn_packed_mode=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n ${mode} ]] || { usage; exit 2; }
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z ${vllm_cpp_sha} ]]; then
  vllm_cpp_sha=$(git -C "${repo_root}" rev-parse HEAD)
fi
if [[ -n ${gdn_ba_mode} &&
      ( ${mode} != trace-only || ${gdn_ba_mode} != both ) ]]; then
  echo "--gdn-ba-mode currently accepts only 'both' with --trace-only" >&2
  exit 2
fi
if [[ -n ${gdn_packed_mode} &&
      ( ${mode} != trace-only || ${gdn_packed_mode} != both ) ]]; then
  echo "--gdn-packed-mode currently accepts only 'both' with --trace-only" >&2
  exit 2
fi
if [[ -n ${gdn_ba_mode} && -n ${gdn_packed_mode} ]]; then
  echo "--gdn-ba-mode and --gdn-packed-mode are mutually exclusive" >&2
  exit 2
fi

if [[ ${mode} == dry-run ]]; then
  evidence_root="${claim_root}/evidence/${vllm_cpp_sha}"
  python3 "${repo_root}/tools/bench/online_gate.py" plan \
    --claim-root "${claim_root}" \
    --vllm-cpp-sha "${vllm_cpp_sha}" \
    --client "${client}" \
    --output "${evidence_root}/manifest.json"
  echo "dry-run manifest: ${evidence_root}/manifest.json" >&2
  exit 0
fi

[[ ${model} == 27 || ${model} == 35 ]] || { echo "--model must be 27 or 35" >&2; exit 2; }
if [[ ${model} == 27 ]]; then
  max_num_batched_tokens=2048
else
  max_num_batched_tokens=8192
fi
[[ -n ${evidence} ]] || { echo "--evidence is required" >&2; exit 2; }
[[ -n ${source_corpus} ]] || { echo "--source-corpus is required" >&2; exit 2; }
expected_source_corpus="${evidence}/corpus/${model}"
[[ $(realpath -m "${source_corpus}") == "$(realpath -m "${expected_source_corpus}")" ]] || {
  echo "--source-corpus must be ${expected_source_corpus} so its hashes stay in evidence" >&2
  exit 2
}

prepare_corpus() {
  local output="${evidence}/corpus/${model}/vllm"
  if [[ -f ${output}/manifest.json ]]; then
    return
  fi
  python3 "${repo_root}/tools/bench/online_gate.py" prepare-corpus \
    --source "${source_corpus}" \
    --output "${output}" \
    --model-key "${model}"
}

if [[ ${mode} == prepare-corpus ]]; then
  prepare_corpus
  exit 0
fi

[[ ${mode} == execute || ${mode} == trace-only ]] || { usage; exit 2; }
# The H1d-era unconditional --execute hold was LIFTED 2026-07-15: its stated
# precondition (H1d/G4 complete; separate production and trace builds) was met
# on 2026-07-13, and the W1D3 closure (b80663a) authorized the fresh
# binding/exact-grid rerun. Timed grids still require a production
# (profile-control-OFF) build via the recorded configure contract.
if [[ ${mode} == trace-only && ${model} != 27 ]]; then
  echo "H1d trace-only control is defined only for the Qwen3.6-27B dense graph; 35B performance remains held" >&2
  exit 2
fi
if [[ ${mode} == trace-only && ${trace_concurrency} != 2 && ${trace_concurrency} != 16 ]]; then
  echo "--trace-concurrency must be 2 or 16" >&2
  exit 2
fi
if [[ -n ${gdn_ba_mode} &&
      ( ${model} != 27 || ${trace_concurrency} != 2 ) ]]; then
  echo "--gdn-ba-mode both is defined only for the 27B c2 trace-only gate" >&2
  exit 2
fi
if [[ -n ${gdn_packed_mode} &&
      ( ${model} != 27 || ${trace_concurrency} != 2 ) ]]; then
  echo "--gdn-packed-mode both is defined only for the 27B c2 trace-only gate" >&2
  exit 2
fi
[[ -n ${snapshot} && -d ${snapshot} ]] || { echo "--snapshot directory is required" >&2; exit 2; }
[[ -n ${build_dir} && -f ${build_dir}/CMakeCache.txt ]] || {
  echo "--build-dir must name a configured CMake build tree" >&2
  exit 2
}
[[ -n ${configure_log} && -s ${configure_log} ]] || {
  echo "--configure-log must name the non-empty log from this build configuration" >&2
  exit 2
}
[[ -x ${client} ]] || { echo "pinned vLLM client is not executable: ${client}" >&2; exit 2; }
[[ $(git -C "${repo_root}" rev-parse HEAD) == "${vllm_cpp_sha}" ]] || {
  echo "worktree HEAD does not match --vllm-cpp-sha" >&2
  exit 2
}
[[ -z $(git -C "${repo_root}" status --porcelain) ]] || {
  echo "vllm.cpp source tree is not completely clean" >&2
  exit 2
}
python3 "${repo_root}/tools/bench/online_gate.py" validate-plan \
  "${evidence}/manifest.json" \
  --vllm-cpp-sha "${vllm_cpp_sha}" >/dev/null
prepare_corpus

# The execution manifest validates this environment before the GPU lock.  Run
# every model-bearing command from a fixed allowlist instead of inheriting an
# operator shell: hidden VT_*/VLLM_CPP_* controls must not alter the graph.
benchmark_system_path=/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin
benchmark_common_env=(
  "HOME=${HOME}"
  "PYTHONPATH=${repo_root}"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "TMPDIR=/tmp"
  "TZ=UTC"
)
benchmark_clean_prefix=(/usr/bin/env -i "${benchmark_common_env[@]}")
benchmark_clean_env=("${benchmark_clean_prefix[@]}" "PATH=${benchmark_system_path}")
h1d_plan_env=(
  "VT_FP4_AUTOTUNE=1"
  "VT_FP4_AUTOTUNE_CACHE_PATH=${evidence}/native-plan-must-not-exist.json"
  "VT_FP4_AUTOTUNE_CACHE_READONLY=1"
  "VT_FP4_AUTOTUNE_DELAY_US=5000"
  "VT_FP4_FLASHINFER_CACHE_PATH=${repo_root}/tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
  "VT_FP4_FULL_TACTICS=1"
  "VT_FP4_PERSISTENT_CACHE=1"
  "VT_FP4_PLAN_CACHE=1"
  "VT_FP4_PRE_SERVE_WARMUP=1"
)
flashinfer_plan_fixture="${repo_root}/tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
native_plan_target="${evidence}/native-plan-must-not-exist.json"
[[ -f ${flashinfer_plan_fixture} ]] || {
  echo "frozen FlashInfer plan fixture is absent" >&2
  exit 2
}
[[ ! -e ${native_plan_target} ]] || {
  echo "native plan target must be absent before execution" >&2
  exit 2
}

execution_dir="${evidence}/execution"
mkdir -p "${execution_dir}"
execution_manifest="${execution_dir}/${model}-trace.json"
if [[ ${model} == 27 ]]; then
  test_name=test_qwen27_paged_engine
else
  test_name=test_qwen36_paged_engine
fi
cmake_home=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${build_dir}/CMakeCache.txt")
[[ -n ${cmake_home} && $(realpath -e "${cmake_home}") == "$(realpath -e "${repo_root}")" ]] || {
  echo "build CMAKE_HOME_DIRECTORY does not match this worktree" >&2
  exit 2
}
vllm_python="$(dirname "${client}")/python"
[[ -x ${vllm_python} ]] || { echo "vLLM oracle Python is absent" >&2; exit 2; }
oracle_manifest="${execution_dir}/${model}-oracle.json"
"${benchmark_clean_env[@]}" "${vllm_python}" \
  "${repo_root}/tools/bench/online_gate.py" record-oracle \
  --output "${oracle_manifest}" \
  --client "${client}"

build_command="${execution_dir}/${model}-build-command.txt"
build_log="${execution_dir}/${model}-build.log"
[[ ! -e ${build_command} && ! -e ${build_log} ]] || {
  echo "refusing to overwrite build provenance for ${model}" >&2
  exit 1
}
build_jobs=$(nproc)
build_cmd=(
  cmake --build "${build_dir}"
  --target server "${test_name}"
  --parallel "${build_jobs}"
)
printf '%q ' "${build_cmd[@]}" >"${build_command}"
printf '\n' >>"${build_command}"
if ! "${build_cmd[@]}" >"${build_log}" 2>&1; then
  cat "${build_log}" >&2
  exit 1
fi
[[ -x ${build_dir}/examples/server ]] || {
  echo "provenance-recorded build did not produce examples/server" >&2
  exit 1
}
# Timed grids record the production (profile-control-OFF) build; the H1d
# trace campaign records the instrumented ON build. The grid summary
# requires profile_control=false for timed evidence.
profile_control_flag=$([[ ${mode} == trace-only ]] && echo on || echo off)
"${benchmark_clean_env[@]}" "${h1d_plan_env[@]}" \
  python3 "${repo_root}/tools/bench/online_gate.py" record-execution \
  --output "${execution_manifest}" \
  --model-key "${model}" \
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
  --profile-control "${profile_control_flag}"

spid=""
mpid=""
profiled_pid=""
profiled_pgid=""
cleanup_server() {
  if [[ -n ${profiled_pid} ]] && kill -0 "${profiled_pid}" 2>/dev/null; then
    if [[ -n ${profiled_pgid} && ${profiled_pgid} == "${profiled_pid}" ]]; then
      kill -TERM -- "-${profiled_pgid}" 2>/dev/null || true
    else
      kill -TERM "${profiled_pid}" 2>/dev/null || true
    fi
    for _ in $(seq 1 30); do
      kill -0 "${profiled_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${profiled_pid}" 2>/dev/null; then
      if [[ -n ${profiled_pgid} && ${profiled_pgid} == "${profiled_pid}" ]]; then
        kill -KILL -- "-${profiled_pgid}" 2>/dev/null || true
      else
        kill -KILL "${profiled_pid}" 2>/dev/null || true
      fi
    fi
  fi
  profiled_pid=""
  profiled_pgid=""
  if [[ -n ${spid} ]] && kill -0 "${spid}" 2>/dev/null; then
    kill -TERM -- "-${spid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
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
trap cleanup_server EXIT INT TERM

mem_available_kib() {
  awk '/^MemAvailable:/{print $2}' /proc/meminfo
}

gpu_idle() {
  [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]') ]]
}

drop_caches() {
  local output=$1
  python3 -m tools.bench.drop_file_cache \
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
      tail -n 80 "${log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "server readiness timed out" >&2
  return 1
}

start_server() {
  local engine=$1 repetition=$2 memory_dir=$3
  local log_dir="${evidence}/logs/${model}/${engine}"
  local log="${log_dir}/r${repetition}-server.log"
  local command_file="${log_dir}/r${repetition}-server-command.txt"
  mkdir -p "${log_dir}"
  [[ ! -e ${log} && ! -e ${command_file} ]] || {
    echo "refusing to overwrite server evidence for ${model}/${engine}/r${repetition}" >&2
    return 1
  }
  local -a server_cmd
  if [[ ${engine} == ours ]]; then
    server_cmd=(
      "${build_dir}/examples/server"
      --model "${snapshot}"
      --port "${port}"
      --num-blocks "${num_blocks}"
      --max-num-seqs "${max_num_seqs}"
      --max-num-batched-tokens "${max_num_batched_tokens}"
      --no-enable-prefix-caching
      --served-model-name gate
    )
  else
    server_cmd=(
      env "PATH=$(dirname "${client}"):${PATH}"
      "${client}" serve "${snapshot}"
      --served-model-name gate
      --gpu-memory-utilization 0.6
      --max-num-seqs "${max_num_seqs}"
      --max-num-batched-tokens "${max_num_batched_tokens}"
      --no-enable-prefix-caching
      --mamba-ssm-cache-dtype float32
      --port "${port}"
    )
  fi
  printf '%q ' "${server_cmd[@]}" >"${command_file}"
  printf '\n' >>"${command_file}"
  setsid "${server_cmd[@]}" >"${log}" 2>&1 &
  spid=$!
  python3 "${repo_root}/tools/bench/sample_process_memory.py" \
    --pid "${spid}" \
    --output "${memory_dir}/r${repetition}.samples.jsonl" \
    --interval 0.1 \
    --include-gpu >"${memory_dir}/r${repetition}.summary.json" &
  mpid=$!
  wait_ready "${log}"
}

run_leg() {
  local engine=$1 repetition=$2
  local baseline final idle_ok=0
  local memory_dir="${evidence}/memory/${model}/${engine}"
  local thermal_dir="${evidence}/thermal/${model}/${engine}"
  local return_dir="${evidence}/memory-return/${model}/${engine}"
  local cache_dir="${evidence}/cache-drop/${model}/${engine}"
  local preflight_dir="${evidence}/preflight/${model}/${engine}"
  local before_cache="${cache_dir}/r${repetition}-before.json"
  local after_cache="${cache_dir}/r${repetition}-after.json"
  mkdir -p \
    "${memory_dir}" "${thermal_dir}" "${return_dir}" "${cache_dir}" "${preflight_dir}"

  gpu_idle || { echo "GPU is not idle before ${model}/${engine}/r${repetition}" >&2; return 1; }
  drop_caches "${before_cache}"
  baseline=$(mem_available_kib)
  start_server "${engine}" "${repetition}" "${memory_dir}"
  nvidia-smi -q -d TEMPERATURE,POWER >"${thermal_dir}/r${repetition}-before.txt"

  python3 "${repo_root}/tools/bench/run_serve_low.py" stream \
    --url "http://127.0.0.1:${port}/v1/completions" \
    --corpus "${source_corpus}/c1-r${repetition}.jsonl" \
    --output-len 128 \
    --minimum-spread 0.05 \
    --output "${preflight_dir}/r${repetition}-stream.json"

  local concurrency
  for concurrency in 1 2 4 8 16 32; do
    kill -0 "${spid}" 2>/dev/null || {
      echo "server died before c${concurrency}" >&2
      return 1
    }
    python3 "${repo_root}/tools/bench/online_gate.py" bench \
      --client "${client}" \
      --tokenizer "${snapshot}" \
      --evidence "${evidence}" \
      --model-key "${model}" \
      --engine "${engine}" \
      --base-url "http://127.0.0.1:${port}" \
      --concurrency "${concurrency}" \
      --repetition "${repetition}"
  done
  nvidia-smi -q -d TEMPERATURE,POWER >"${thermal_dir}/r${repetition}-after.txt"
  cleanup_server

  for _ in $(seq 1 120); do
    if gpu_idle; then
      idle_ok=1
      break
    fi
    sleep 1
  done
  ((idle_ok == 1)) || { echo "GPU did not return after ${model}/${engine}/r${repetition}" >&2; return 1; }
  drop_caches "${after_cache}"
  final=$(mem_available_kib)
  python3 "${repo_root}/tools/bench/online_gate.py" record-memory-return \
    --output "${return_dir}/r${repetition}.json" \
    --baseline-kib "${baseline}" \
    --final-kib "${final}" \
    --tolerance-kib 1048576 \
    --before-cache-drop-report "${before_cache}" \
    --after-cache-drop-report "${after_cache}" \
    --gpu-idle
}

run_paired_traces() {
  local trace_arm=${1:-}
  local trace_dir="${evidence}/trace/${model}"
  local artifact_prefix=""
  local gdn_ba_env_value=""
  local gdn_packed_env_value=""
  case "${trace_arm}" in
    "") ;;
    merged)
      trace_dir="${evidence}/trace/${model}/gdn-ba-merged"
      artifact_prefix="gdn-ba-merged-"
      gdn_ba_env_value=1
      ;;
    split)
      trace_dir="${evidence}/trace/${model}/gdn-ba-split"
      artifact_prefix="gdn-ba-split-"
      gdn_ba_env_value=0
      ;;
    packed)
      trace_dir="${evidence}/trace/${model}/gdn-packed"
      artifact_prefix="gdn-packed-"
      gdn_packed_env_value=1
      ;;
    rollback)
      trace_dir="${evidence}/trace/${model}/gdn-rollback"
      artifact_prefix="gdn-rollback-"
      gdn_packed_env_value=0
      ;;
    *)
      echo "unknown internal GDN trace arm: ${trace_arm}" >&2
      return 2
      ;;
  esac
  local trace_prompts=48
  if [[ ${trace_concurrency} == 2 ]]; then
    trace_prompts=6
  fi
  local vllm_profile_dir="${trace_dir}/vllm-profile"
  local vllm_metadata="${trace_dir}/vllm-profile-metadata.json"
  local vllm_corpus="${evidence}/corpus/${model}/vllm/c${trace_concurrency}-r1.jsonl"
  local vllm_summary="${trace_dir}/vllm-kernels.json"
  local vllm_log="${trace_dir}/vllm-profile.log"
  local vllm_command="${trace_dir}/vllm-profile-command.txt"
  local status="${trace_dir}/status.json"
  if [[ ${trace_concurrency} == 2 ]]; then
    status="${trace_dir}/status-c2.json"
  fi
  local cache_before_ours="${trace_dir}/cache-before-ours.json"
  local cache_between="${trace_dir}/cache-between-engines.json"
  local cache_after_vllm="${trace_dir}/cache-after-vllm.json"
  local -a ours_reps=()
  local -a ours_sqlites=()
  local -a ours_summaries=()
  local -a ours_validations=()
  local -a ours_logs=()
  local -a ours_commands=()
  local -a ours_controls=()
  mkdir -p "${trace_dir}"
  [[ ! -e ${vllm_summary} && ! -e ${status} ]] || {
    echo "refusing to overwrite paired trace evidence for ${model}" >&2
    return 1
  }
  command -v nsys >/dev/null || { echo "nsys is required for our trace" >&2; return 1; }

  gpu_idle || { echo "GPU is not idle before our trace" >&2; return 1; }
  drop_caches "${cache_before_ours}"
  local trace_rep
  for trace_rep in 1 2 3; do
    local ours_prefix="${trace_dir}/ours-r${trace_rep}"
    local ours_log="${trace_dir}/ours-r${trace_rep}-profile.log"
    local ours_command="${trace_dir}/ours-r${trace_rep}-profile-command.txt"
    local ours_control="${trace_dir}/ours-r${trace_rep}-profile-control.json"
    local shutdown_fifo="${ours_prefix}-shutdown.fifo"
    [[ ! -e ${ours_log} && ! -e ${ours_command} && ! -e ${ours_control} &&
       ! -e ${shutdown_fifo} ]] || {
      echo "refusing to overwrite ours trace repetition ${trace_rep}" >&2
      return 1
    }
    local range_index
    for range_index in 1 2 3 4; do
      local range_prefix="${ours_prefix}.${range_index}"
      [[ ! -e ${range_prefix}.nsys-rep && ! -e ${range_prefix}.sqlite &&
         ! -e ${range_prefix}-nsys-validation.json &&
         ! -e ${range_prefix}-cuda_gpu_kern_sum.txt ]] || {
        echo "refusing to overwrite ours trace repetition ${trace_rep} range ${range_index}" >&2
        return 1
      }
    done
    mkfifo --mode=600 "${shutdown_fifo}"
    local -a server_cmd=(
      "${build_dir}/examples/server"
      --model "${snapshot}"
      --port "${port}"
      --num-blocks "${num_blocks}"
      --max-num-seqs "${max_num_seqs}"
      --max-num-batched-tokens "${max_num_batched_tokens}"
      --max-model-len 262144
      --no-enable-prefix-caching
      --cuda-profile-graph-replays 4
    )
    if [[ ${trace_concurrency} == 2 ]]; then
      server_cmd+=(--cuda-profile-graph-batch 2)
    fi
    server_cmd+=(
      --benchmark-shutdown-fifo "${shutdown_fifo}"
      --served-model-name gate
    )
    local -a trace_env=("${h1d_plan_env[@]}")
    if [[ -n ${gdn_ba_env_value} ]]; then
      trace_env+=("VT_GDN_MERGED_BA=${gdn_ba_env_value}")
    fi
    if [[ -n ${gdn_packed_env_value} ]]; then
      trace_env+=("VT_GDN_PACKED_DECODE=${gdn_packed_env_value}")
    fi
    local -a profile_cmd=(
      "${benchmark_clean_env[@]}"
      "${trace_env[@]}"
      nsys profile
      --trace=cuda
      --capture-range=cudaProfilerApi
      --capture-range-end=repeat:4:sync
      --flush-on-cudaprofilerstop=true
      --cuda-flush-interval=0
      --cuda-graph-trace=node:host-only
      --cuda-event-trace=false
      --sample=none
      --cpuctxsw=none
      --stats=false
      --kill=none
      --force-overwrite=true
      --output "${ours_prefix}"
      "${server_cmd[@]}"
    )
    printf '%q ' "${profile_cmd[@]}" >"${ours_command}"
    printf '\n' >>"${ours_command}"
    setsid "${profile_cmd[@]}" >"${ours_log}" 2>&1 &
    spid=$!
    local nsys_pid=${spid}
    wait_ready "${ours_log}"
    local server_pid=""
    for _ in $(seq 1 60); do
      server_pid=$(sed -n 's/^\[VT_CUDA_PROFILE\] ready pid=\([0-9][0-9]*\) signal=SIGUSR2 target_replays=4$/\1/p' "${ours_log}")
      [[ $(wc -w <<<"${server_pid}") -eq 1 ]] && break
      sleep 1
    done
    [[ ${server_pid} =~ ^[0-9]+$ ]] || {
      echo "profiled server did not emit one exact ready marker" >&2
      return 1
    }
    local shutdown_ready_pid=""
    for _ in $(seq 1 60); do
      shutdown_ready_pid=$(sed -n 's/^\[VT_BENCH_SHUTDOWN\] ready pid=\([0-9][0-9]*\) control=fifo$/\1/p' "${ours_log}")
      [[ $(wc -w <<<"${shutdown_ready_pid}") -eq 1 ]] && break
      sleep 1
    done
    [[ ${shutdown_ready_pid} == "${server_pid}" ]] || {
      echo "profiled server did not arm one exact graceful-shutdown waiter" >&2
      return 1
    }
    profiled_pid=${server_pid}
    kill -0 "${server_pid}" 2>/dev/null || {
      echo "profiled server PID is not live" >&2
      return 1
    }
    local server_identity launcher_identity nsys_identity
    local server_ppid server_pgid server_sid
    local launcher_pid launcher_ppid launcher_pgid launcher_sid launcher_comm
    local nsys_pgid nsys_sid
    server_identity=$(ps -o ppid=,pgid=,sid= -p "${server_pid}")
    read -r server_ppid server_pgid server_sid <<<"${server_identity}"
    launcher_pid=${server_ppid}
    launcher_identity=$(ps -o ppid=,pgid=,sid=,comm= -p "${launcher_pid}")
    read -r launcher_ppid launcher_pgid launcher_sid launcher_comm <<<"${launcher_identity}"
    nsys_identity=$(ps -o pgid=,sid= -p "${nsys_pid}")
    read -r nsys_pgid nsys_sid <<<"${nsys_identity}"
    [[ ${nsys_pgid} == "${nsys_pid}" && ${nsys_sid} == "${nsys_pid}" &&
       ${launcher_ppid} == "${nsys_pid}" && ${launcher_pgid} == "${nsys_pid}" &&
       ${launcher_sid} == "${nsys_pid}" && ${launcher_comm} == nsys-launcher &&
       ${server_ppid} == "${launcher_pid}" && ${server_pgid} == "${server_pid}" &&
       ${server_sid} == "${server_pid}" ]] || {
      echo "profiled server does not have the expected nsys -> nsys-launcher -> target ancestry" >&2
      echo "nsys pid=${nsys_pid} pgid=${nsys_pgid} sid=${nsys_sid}; launcher pid=${launcher_pid} ppid=${launcher_ppid} pgid=${launcher_pgid} sid=${launcher_sid} comm=${launcher_comm}; server pid=${server_pid} ppid=${server_ppid} pgid=${server_pgid} sid=${server_sid}" >&2
      return 1
    }
    profiled_pgid=${server_pgid}
    python3 "${repo_root}/tools/bench/online_gate.py" bench \
      --client "${client}" \
      --tokenizer "${snapshot}" \
      --evidence "${evidence}" \
      --model-key "${model}" \
      --engine ours \
      --base-url "http://127.0.0.1:${port}" \
      --concurrency "${trace_concurrency}" \
      --repetition 1 \
      --num-prompts "${trace_prompts}" \
      --artifact-tag "${artifact_prefix}trace${trace_rep}"
    [[ ! -e ${native_plan_target} ]] || {
      echo "forbidden native plan target was created before capture" >&2
      return 1
    }
    kill -USR2 "${server_pid}"
    python3 "${repo_root}/tools/bench/online_gate.py" bench \
      --client "${client}" \
      --tokenizer "${snapshot}" \
      --evidence "${evidence}" \
      --model-key "${model}" \
      --engine ours \
      --base-url "http://127.0.0.1:${port}" \
      --concurrency "${trace_concurrency}" \
      --repetition 1 \
      --num-prompts "${trace_concurrency}" \
      --num-warmups 0 \
      --artifact-tag "${artifact_prefix}trace${trace_rep}-probe"
    local capture_closed=0
    for _ in $(seq 1 60); do
      if grep -q '\[VT_CUDA_PROFILE\] stopped captured_replays=4 graph=0x[0-9a-f][0-9a-f]*$' "${ours_log}"; then
        capture_closed=1
        break
      fi
      kill -0 "${server_pid}" 2>/dev/null || break
      sleep 1
    done
    ((capture_closed == 1)) || {
      echo "profiled server did not close the exact four-replay window" >&2
      return 1
    }
    printf 'Q' >"${shutdown_fifo}"
    local nsys_status=0
    local nsys_exited=0
    for _ in $(seq 1 60); do
      if ! kill -0 "${nsys_pid}" 2>/dev/null; then
        nsys_exited=1
        break
      fi
      sleep 1
    done
    if ((nsys_exited == 0)); then
      echo "nsys did not exit within 60 seconds after graceful target shutdown" >&2
      kill -INT -- "-${nsys_pid}" 2>/dev/null || true
      return 1
    fi
    wait "${nsys_pid}" || nsys_status=$?
    if ((nsys_status != 0)); then
      echo "nsys exited ${nsys_status} for trace repetition ${trace_rep}" >&2
      return 1
    fi
    grep -q '^\[VT_BENCH_SHUTDOWN\] requested control=fifo$' "${ours_log}" || {
      echo "profiled server did not record its graceful-shutdown request" >&2
      return 1
    }
    grep -q '^\[VT_BENCH_SHUTDOWN\] completed control=fifo$' "${ours_log}" || {
      echo "profiled server did not complete graceful shutdown" >&2
      return 1
    }
    rm -- "${shutdown_fifo}"
    [[ ! -e ${shutdown_fifo} ]] || {
      echo "profile shutdown FIFO still exists after target exit" >&2
      return 1
    }
    spid=""
    profiled_pid=""
    profiled_pgid=""
    python3 "${repo_root}/tools/bench/online_gate.py" record-profile-control \
      --output "${ours_control}" \
      --profile-log "${ours_log}" \
      --nsys-pid "${nsys_pid}" \
      --nsys-pgid "${nsys_pgid}" \
      --nsys-sid "${nsys_sid}" \
      --nsys-exit-status "${nsys_status}" \
      --launcher-pid "${launcher_pid}" \
      --launcher-ppid "${launcher_ppid}" \
      --launcher-pgid "${launcher_pgid}" \
      --launcher-sid "${launcher_sid}" \
      --launcher-comm "${launcher_comm}" \
      --server-pid "${server_pid}" \
      --server-ppid "${server_ppid}" \
      --server-pgid "${server_pgid}" \
      --server-sid "${server_sid}" \
      --shutdown-fifo "${shutdown_fifo}" \
      --expected-batch "${trace_concurrency}"
    for range_index in 1 2 3 4; do
      local range_prefix="${ours_prefix}.${range_index}"
      local ours_rep="${range_prefix}.nsys-rep"
      local ours_sqlite="${range_prefix}.sqlite"
      local ours_summary="${range_prefix}-cuda_gpu_kern_sum.txt"
      local ours_validation="${range_prefix}-nsys-validation.json"
      [[ -s ${ours_rep} ]] || {
        echo "nsys did not write ${ours_rep}" >&2
        return 1
      }
      nsys export --type=sqlite --lazy=false --force-overwrite=false \
        --output "${ours_sqlite}" "${ours_rep}" >/dev/null
      [[ -s ${ours_sqlite} ]] || {
        echo "ours Nsight SQLite export is empty for range ${range_index}" >&2
        return 1
      }
      local -a gdn_ba_validation_args=()
      local -a gdn_validation_args=()
      if [[ -n ${gdn_ba_env_value} ]]; then
        gdn_ba_validation_args=(--gdn-ba-mode "${trace_arm}")
      fi
      if [[ -n ${gdn_packed_env_value} ]]; then
        gdn_validation_args=(--gdn-packed-mode "${trace_arm}")
      fi
      python3 "${repo_root}/tools/bench/online_gate.py" validate-nsys-trace \
        --sqlite "${ours_sqlite}" \
        --model-key "${model}" \
        --range-index "${range_index}" \
        --expected-batch "${trace_concurrency}" \
        "${gdn_ba_validation_args[@]}" \
        "${gdn_validation_args[@]}" \
        --output "${ours_validation}"
      python3 "${repo_root}/tools/bench/online_gate.py" summarize-nsys-kernels \
        --sqlite "${ours_sqlite}" \
        --model-key "${model}" \
        --range-index "${range_index}" \
        --expected-batch "${trace_concurrency}" \
        "${gdn_ba_validation_args[@]}" \
        "${gdn_validation_args[@]}" \
        --output "${ours_summary}"
      ours_reps+=("${ours_rep}")
      ours_sqlites+=("${ours_sqlite}")
      ours_summaries+=("${ours_summary}")
      ours_validations+=("${ours_validation}")
    done
    ours_logs+=("${ours_log}")
    ours_commands+=("${ours_command}")
    ours_controls+=("${ours_control}")
    gpu_idle || {
      echo "GPU is not idle after ours trace repetition ${trace_rep}" >&2
      return 1
    }
  done

  gpu_idle || { echo "GPU is not idle before vLLM trace" >&2; return 1; }
  drop_caches "${cache_between}"
  local -a vllm_profile_cmd=(
    "${benchmark_clean_prefix[@]}"
    "PATH=$(dirname "${client}"):${benchmark_system_path}"
    "${vllm_python}" "${repo_root}/tools/bench/profile_vllm_online_gate.py"
    --model "${snapshot}"
    --corpus "${vllm_corpus}"
    --profile-dir "${vllm_profile_dir}"
    --metadata "${vllm_metadata}"
    --num-prompts "${trace_prompts}"
    --max-concurrency "${trace_concurrency}"
    --max-num-seqs "${max_num_seqs}"
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --repetitions 3
  )
  printf '%q ' "${vllm_profile_cmd[@]}" >"${vllm_command}"
  printf '\n' >>"${vllm_command}"
  "${vllm_profile_cmd[@]}" >"${vllm_log}" 2>&1
  if [[ ${trace_concurrency} == 16 ]]; then
    python3 "${repo_root}/tools/bench/summarize_torch_kernels.py" \
      --profile-dir "${vllm_profile_dir}" \
      --model-key "${model}" \
      --output "${vllm_summary}"
  else
    python3 "${repo_root}/tools/bench/summarize_torch_kernels.py" \
      --profile-dir "${vllm_profile_dir}" \
      --output "${vllm_summary}"
  fi
  local vllm_trace
  vllm_trace=$(python3 - "${vllm_summary}" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["selected_trace"])
PY
)
  local -a status_args=(
    --output "${status}"
    --model-key "${model}"
    --vllm-torch-trace "${vllm_trace}"
    --vllm-kernel-summary "${vllm_summary}"
    --vllm-command "${vllm_command}"
    --vllm-profile-log "${vllm_log}"
    --vllm-metadata "${vllm_metadata}"
    --vllm-corpus "${vllm_corpus}"
    --cache-drop-report "${cache_before_ours}"
    --cache-drop-report "${cache_between}"
    --cache-drop-report "${cache_after_vllm}"
    --execution-manifest "${execution_manifest}"
    --vllm-cpp-sha "${vllm_cpp_sha}"
  )
  local trace_index
  for trace_index in $(seq 0 11); do
    status_args+=(
      --ours-nsys-report "${ours_reps[trace_index]}"
      --ours-nsys-sqlite "${ours_sqlites[trace_index]}"
      --ours-nsys-validation "${ours_validations[trace_index]}"
      --ours-kernel-summary "${ours_summaries[trace_index]}"
    )
  done
  for trace_index in 0 1 2; do
    trace_rep=$((trace_index + 1))
    status_args+=(
      --ours-command "${ours_commands[trace_index]}"
      --ours-profile-log "${ours_logs[trace_index]}"
      --ours-profile-control "${ours_controls[trace_index]}"
      --ours-client-result "${evidence}/raw/${model}/ours/c16-r1-trace${trace_rep}.json"
      --ours-client-log "${evidence}/logs/${model}/ours/c16-r1-trace${trace_rep}.log"
      --ours-probe-result "${evidence}/raw/${model}/ours/c16-r1-trace${trace_rep}-probe.json"
      --ours-probe-log "${evidence}/logs/${model}/ours/c16-r1-trace${trace_rep}-probe.log"
    )
  done
  gpu_idle || { echo "GPU did not return after paired traces" >&2; return 1; }
  drop_caches "${cache_after_vllm}"
  if [[ ${trace_concurrency} == 2 ]]; then
    if [[ -n ${gdn_ba_env_value} ]]; then
      echo "c2 GDN BA ${trace_arm} raw paired trace capture complete; status remains PENDING until GDN BA finalization" >&2
    elif [[ -n ${gdn_packed_env_value} ]]; then
      echo "c2 GDN packed ${trace_arm} raw paired trace capture complete; status remains PENDING until GDN packed finalization" >&2
    else
      echo "c2 raw paired trace capture complete; status remains PENDING until low-batch finalization" >&2
    fi
    return 0
  fi
  python3 "${repo_root}/tools/bench/online_gate.py" record-trace-status \
    "${status_args[@]}"
}

# One lock for every ours/vLLM leg of this model, including the model gate.
exec 9>/tmp/gpu
flock 9
gpu_idle || { echo "GPU has a compute owner after acquiring /tmp/gpu" >&2; exit 1; }

gate_dir="${evidence}/preflight/model-gate"
gate_log="${gate_dir}/${model}.log"
gate_status="${gate_dir}/${model}.json"
mkdir -p "${gate_dir}"
[[ ! -e ${gate_log} && ! -e ${gate_status} ]] || {
  echo "refusing to overwrite model-gate evidence for ${model}" >&2
  exit 1
}
if ! "${benchmark_clean_env[@]}" "${h1d_plan_env[@]}" \
  ctest --test-dir "${build_dir}" -R "^${test_name}$" --output-on-failure \
  >"${gate_log}" 2>&1; then
  cat "${gate_log}" >&2
  exit 1
fi
cat "${gate_log}"
python3 "${repo_root}/tools/bench/online_gate.py" record-model-gate \
  --output "${gate_status}" \
  --log "${gate_log}" \
  --model-key "${model}" \
  --test-name "${test_name}" \
  --vllm-cpp-sha "${vllm_cpp_sha}"

if [[ ${mode} == trace-only ]]; then
  if [[ ${gdn_ba_mode} == both ]]; then
    run_paired_traces merged
    run_paired_traces split
    echo "model ${model} GDN BA merged/split node-level paired traces complete" >&2
  elif [[ ${gdn_packed_mode} == both ]]; then
    run_paired_traces packed
    run_paired_traces rollback
    echo "model ${model} GDN packed/rollback node-level paired traces complete" >&2
  else
    run_paired_traces
    echo "model ${model} node-level paired trace complete" >&2
  fi
  exit 0
fi

for repetition in 1 2 3; do
  run_leg ours "${repetition}"
  run_leg vllm "${repetition}"
done
run_paired_traces

model_summary_status=0
python3 "${repo_root}/tools/bench/online_gate_summary.py" \
  --evidence "${evidence}" \
  --model "${model}" || model_summary_status=$?
if ((model_summary_status > 1)); then
  exit "${model_summary_status}"
fi

full_summary_status=0
if [[ -f ${evidence}/summary-27/ratios.json && -f ${evidence}/summary-35/ratios.json ]]; then
  python3 "${repo_root}/tools/bench/online_gate_summary.py" \
    --evidence "${evidence}" || full_summary_status=$?
  if ((full_summary_status > 1)); then
    exit "${full_summary_status}"
  fi
else
  echo "model ${model} summary complete; cross-model summary waits for the other model" >&2
fi
if ((model_summary_status != 0 || full_summary_status != 0)); then
  exit 1
fi
