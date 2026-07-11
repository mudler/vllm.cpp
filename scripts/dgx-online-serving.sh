#!/usr/bin/env bash
# Reproducible SERVE-GATE-ONLINE campaign driver.
#
# The accepted contract is .agents/specs/cuda-online-serving-gate.md. Timed
# requests are issued only by pinned vLLM `bench serve`; this script owns server
# lifecycle, interleaving, the one-model/one-lock boundary, memory return, and
# artifact capture.  --dry-run and --prepare-corpus never acquire /tmp/gpu.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  dgx-online-serving.sh --dry-run [--claim-root DIR] [--client PATH] [--vllm-cpp-sha SHA]
  dgx-online-serving.sh --prepare-corpus --model 27|35 --source-corpus DIR --evidence DIR
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
    --dry-run|--prepare-corpus|--execute)
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

[[ ${mode} == execute ]] || { usage; exit 2; }
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
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
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
  local baseline final drop_ok=0 idle_ok=0
  local memory_dir="${evidence}/memory/${model}/${engine}"
  local thermal_dir="${evidence}/thermal/${model}/${engine}"
  local return_dir="${evidence}/memory-return/${model}/${engine}"
  local preflight_dir="${evidence}/preflight/${model}/${engine}"
  mkdir -p "${memory_dir}" "${thermal_dir}" "${return_dir}" "${preflight_dir}"

  drop_caches
  baseline=$(mem_available_kib)
  gpu_idle || { echo "GPU is not idle before ${model}/${engine}/r${repetition}" >&2; return 1; }
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

  if drop_caches; then
    drop_ok=1
  fi
  for _ in $(seq 1 120); do
    if gpu_idle; then
      idle_ok=1
      break
    fi
    sleep 1
  done
  final=$(mem_available_kib)
  local -a return_flags=()
  ((drop_ok == 1)) && return_flags+=(--drop-caches-succeeded)
  ((idle_ok == 1)) && return_flags+=(--gpu-idle)
  python3 "${repo_root}/tools/bench/online_gate.py" record-memory-return \
    --output "${return_dir}/r${repetition}.json" \
    --baseline-kib "${baseline}" \
    --final-kib "${final}" \
    --tolerance-kib 1048576 \
    "${return_flags[@]}"
}

run_paired_traces() {
  local trace_dir="${evidence}/trace/${model}"
  local ours_prefix="${trace_dir}/ours"
  local ours_rep="${ours_prefix}.nsys-rep"
  local ours_summary="${trace_dir}/ours-cuda_gpu_kern_sum.txt"
  local ours_log="${trace_dir}/ours-profile.log"
  local ours_command="${trace_dir}/ours-profile-command.txt"
  local vllm_profile_dir="${trace_dir}/vllm-profile"
  local vllm_metadata="${trace_dir}/vllm-profile-metadata.json"
  local vllm_corpus="${evidence}/corpus/${model}/vllm/c16-r1.jsonl"
  local vllm_summary="${trace_dir}/vllm-kernels.json"
  local vllm_log="${trace_dir}/vllm-profile.log"
  local vllm_command="${trace_dir}/vllm-profile-command.txt"
  local status="${trace_dir}/status.json"
  mkdir -p "${trace_dir}"
  [[ ! -e ${ours_rep} && ! -e ${vllm_summary} && ! -e ${status} ]] || {
    echo "refusing to overwrite paired trace evidence for ${model}" >&2
    return 1
  }
  command -v nsys >/dev/null || { echo "nsys is required for our trace" >&2; return 1; }

  drop_caches
  gpu_idle || { echo "GPU is not idle before our trace" >&2; return 1; }
  local -a server_cmd=(
    "${build_dir}/examples/server"
    --model "${snapshot}"
    --port "${port}"
    --num-blocks "${num_blocks}"
    --max-num-seqs "${max_num_seqs}"
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --served-model-name gate
  )
  local -a profile_cmd=(
    nsys profile
    --trace=cuda
    --sample=none
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
  local trace_rep
  for trace_rep in 1 2 3; do
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
  done
  # Stop only the profiled target first so nsys can flush its report.
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
  [[ -s ${ours_summary} ]] || { echo "ours kernel summary is empty" >&2; return 1; }

  drop_caches
  gpu_idle || { echo "GPU is not idle before vLLM trace" >&2; return 1; }
  local -a vllm_profile_cmd=(
    "${vllm_python}" "${repo_root}/tools/bench/profile_vllm_online_gate.py"
    --model "${snapshot}"
    --corpus "${vllm_corpus}"
    --profile-dir "${vllm_profile_dir}"
    --metadata "${vllm_metadata}"
    --num-prompts 48
    --max-num-seqs 16
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
    --ours-nsys-report "${ours_rep}"
    --ours-kernel-summary "${ours_summary}"
    --ours-command "${ours_command}"
    --ours-profile-log "${ours_log}"
    --vllm-torch-trace "${vllm_trace}"
    --vllm-kernel-summary "${vllm_summary}"
    --vllm-command "${vllm_command}"
    --vllm-profile-log "${vllm_log}"
    --vllm-metadata "${vllm_metadata}"
    --vllm-corpus "${vllm_corpus}"
    --vllm-cpp-sha "${vllm_cpp_sha}"
  )
  for trace_rep in 1 2 3; do
    status_args+=(
      --ours-client-result "${evidence}/raw/${model}/ours/c16-r1-trace${trace_rep}.json"
      --ours-client-log "${evidence}/logs/${model}/ours/c16-r1-trace${trace_rep}.log"
    )
  done
  python3 "${repo_root}/tools/bench/online_gate.py" record-trace-status \
    "${status_args[@]}"
  drop_caches
  gpu_idle || { echo "GPU did not return after paired traces" >&2; return 1; }
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

for repetition in 1 2 3; do
  run_leg ours "${repetition}"
  run_leg vllm "${repetition}"
done
run_paired_traces

if [[ -d ${evidence}/raw/27/ours && -d ${evidence}/raw/35/ours ]]; then
  python3 "${repo_root}/tools/bench/online_gate_summary.py" --evidence "${evidence}"
else
  echo "model ${model} series complete; cross-model summary waits for the other model" >&2
fi
