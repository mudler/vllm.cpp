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
  dgx-online-serving.sh --trace-only --model 27|35 --snapshot DIR --source-corpus DIR \
    --evidence DIR --build-dir DIR [--client PATH] [--port N]
  dgx-online-serving.sh --execute --model 27|35 --snapshot DIR --source-corpus DIR \
    --evidence DIR --build-dir DIR [--client PATH] [--port N]
EOF
}

mode=""
model=""
snapshot=""
source_corpus=""
claim_root="${HOME}/work/vllm.cpp-online-gate"
evidence=""
build_dir=""
client="${HOME}/venvs/vllm-oracle/bin/vllm"
vllm_cpp_sha=""
port=8001
num_blocks=4736
max_num_seqs=32
max_num_batched_tokens=""

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
    --client) client=${2:?}; shift 2 ;;
    --vllm-cpp-sha) vllm_cpp_sha=${2:?}; shift 2 ;;
    --port) port=${2:?}; shift 2 ;;
    --num-blocks) num_blocks=${2:?}; shift 2 ;;
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
[[ -n ${snapshot} && -d ${snapshot} ]] || { echo "--snapshot directory is required" >&2; exit 2; }
[[ -n ${build_dir} && -x ${build_dir}/examples/server ]] || {
  echo "--build-dir must contain examples/server" >&2
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

execution_dir="${evidence}/execution"
mkdir -p "${execution_dir}"
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
"${vllm_python}" "${repo_root}/tools/bench/online_gate.py" record-oracle \
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
python3 "${repo_root}/tools/bench/online_gate.py" record-execution \
  --output "${execution_dir}/${model}.json" \
  --model-key "${model}" \
  --vllm-cpp-sha "${vllm_cpp_sha}" \
  --build-dir "${build_dir}" \
  --client "${client}" \
  --snapshot "${snapshot}" \
  --build-command "${build_command}" \
  --build-log "${build_log}" \
  --oracle-manifest "${oracle_manifest}" \
  --port "${port}" \
  --num-blocks "${num_blocks}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-num-batched-tokens "${max_num_batched_tokens}"

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
  local trace_dir="${evidence}/trace/${model}"
  local vllm_profile_dir="${trace_dir}/vllm-profile"
  local vllm_metadata="${trace_dir}/vllm-profile-metadata.json"
  local vllm_corpus="${evidence}/corpus/${model}/vllm/c16-r1.jsonl"
  local vllm_summary="${trace_dir}/vllm-kernels.json"
  local vllm_log="${trace_dir}/vllm-profile.log"
  local vllm_command="${trace_dir}/vllm-profile-command.txt"
  local status="${trace_dir}/status.json"
  local cache_before_ours="${trace_dir}/cache-before-ours.json"
  local cache_between="${trace_dir}/cache-between-engines.json"
  local cache_after_vllm="${trace_dir}/cache-after-vllm.json"
  local -a ours_reps=()
  local -a ours_sqlites=()
  local -a ours_summaries=()
  local -a ours_validations=()
  local -a ours_logs=()
  local -a ours_commands=()
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
    local ours_rep="${ours_prefix}.nsys-rep"
    local ours_sqlite="${ours_prefix}.sqlite"
    local ours_summary="${trace_dir}/ours-r${trace_rep}-cuda_gpu_kern_sum.txt"
    local ours_validation="${trace_dir}/ours-r${trace_rep}-nsys-validation.json"
    local ours_log="${trace_dir}/ours-r${trace_rep}-profile.log"
    local ours_command="${trace_dir}/ours-r${trace_rep}-profile-command.txt"
    [[ ! -e ${ours_rep} && ! -e ${ours_sqlite} && ! -e ${ours_validation} &&
       ! -e ${ours_summary} && ! -e ${ours_log} && ! -e ${ours_command} ]] || {
      echo "refusing to overwrite ours trace repetition ${trace_rep}" >&2
      return 1
    }
    local -a server_cmd=(
      "${build_dir}/examples/server"
      --model "${snapshot}"
      --port "${port}"
      --num-blocks "${num_blocks}"
      --max-num-seqs "${max_num_seqs}"
      --max-num-batched-tokens "${max_num_batched_tokens}"
      --no-enable-prefix-caching
      --served-model-name gate
    )
    local -a profile_cmd=(
      nsys profile
      --trace=cuda
      --cuda-graph-trace=node
      --cuda-flush-interval=10000
      --sample=none
      --cpuctxsw=none
      --stats=false
      --force-overwrite=true
      --output "${ours_prefix}"
      "${server_cmd[@]}"
    )
    printf '%q ' "${profile_cmd[@]}" >"${ours_command}"
    printf '\n' >>"${ours_command}"
    setsid "${profile_cmd[@]}" >"${ours_log}" 2>&1 &
    spid=$!
    wait_ready "${ours_log}"
    python3 "${repo_root}/tools/bench/online_gate.py" bench \
      --client "${client}" \
      --tokenizer "${snapshot}" \
      --evidence "${evidence}" \
      --model-key "${model}" \
      --engine ours \
      --base-url "http://127.0.0.1:${port}" \
      --concurrency 16 \
      --repetition 1 \
      --num-prompts 48 \
      --artifact-tag "trace${trace_rep}"
    # Nsight 2025.3 recommends a 10-second periodic flush for collections over
    # 30 seconds. Leave one complete idle interval before target shutdown.
    sleep 11
    if ! pkill -TERM -P "${spid}" 2>/dev/null; then
      kill -INT "${spid}" 2>/dev/null || true
    fi
    for _ in $(seq 1 120); do
      kill -0 "${spid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${spid}" 2>/dev/null; then
      kill -INT "${spid}" 2>/dev/null || true
    fi
    wait "${spid}" 2>/dev/null || true
    spid=""
    [[ -s ${ours_rep} ]] || {
      echo "nsys did not write ${ours_rep}" >&2
      return 1
    }
    nsys stats --force-export=true --report cuda_gpu_kern_sum "${ours_rep}" \
      >"${ours_summary}"
    [[ -s ${ours_summary} ]] || {
      echo "ours kernel summary is empty" >&2
      return 1
    }
    [[ -s ${ours_sqlite} ]] || {
      echo "ours Nsight SQLite export is empty" >&2
      return 1
    }
    python3 "${repo_root}/tools/bench/online_gate.py" validate-nsys-trace \
      --sqlite "${ours_sqlite}" \
      --model-key "${model}" \
      --output "${ours_validation}"
    ours_reps+=("${ours_rep}")
    ours_sqlites+=("${ours_sqlite}")
    ours_summaries+=("${ours_summary}")
    ours_validations+=("${ours_validation}")
    ours_logs+=("${ours_log}")
    ours_commands+=("${ours_command}")
    gpu_idle || {
      echo "GPU is not idle after ours trace repetition ${trace_rep}" >&2
      return 1
    }
  done

  gpu_idle || { echo "GPU is not idle before vLLM trace" >&2; return 1; }
  drop_caches "${cache_between}"
  local -a vllm_profile_cmd=(
    env "PATH=$(dirname "${client}"):${PATH}"
    "${vllm_python}" "${repo_root}/tools/bench/profile_vllm_online_gate.py"
    --model "${snapshot}"
    --corpus "${vllm_corpus}"
    --profile-dir "${vllm_profile_dir}"
    --metadata "${vllm_metadata}"
    --num-prompts 48
    --max-concurrency 16
    --max-num-seqs "${max_num_seqs}"
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --repetitions 3
  )
  printf '%q ' "${vllm_profile_cmd[@]}" >"${vllm_command}"
  printf '\n' >>"${vllm_command}"
  "${vllm_profile_cmd[@]}" >"${vllm_log}" 2>&1
  python3 "${repo_root}/tools/bench/summarize_torch_kernels.py" \
    --profile-dir "${vllm_profile_dir}" \
    --output "${vllm_summary}"
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
    --vllm-cpp-sha "${vllm_cpp_sha}"
  )
  local trace_index
  for trace_index in 0 1 2; do
    trace_rep=$((trace_index + 1))
    status_args+=(
      --ours-nsys-report "${ours_reps[trace_index]}"
      --ours-nsys-sqlite "${ours_sqlites[trace_index]}"
      --ours-nsys-validation "${ours_validations[trace_index]}"
      --ours-kernel-summary "${ours_summaries[trace_index]}"
      --ours-command "${ours_commands[trace_index]}"
      --ours-profile-log "${ours_logs[trace_index]}"
      --ours-client-result "${evidence}/raw/${model}/ours/c16-r1-trace${trace_rep}.json"
      --ours-client-log "${evidence}/logs/${model}/ours/c16-r1-trace${trace_rep}.log"
    )
  done
  gpu_idle || { echo "GPU did not return after paired traces" >&2; return 1; }
  drop_caches "${cache_after_vllm}"
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
if ! ctest --test-dir "${build_dir}" -R "^${test_name}$" --output-on-failure \
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
  run_paired_traces
  echo "model ${model} node-level paired trace complete" >&2
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
