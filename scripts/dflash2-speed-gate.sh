#!/usr/bin/env bash
# The SPEC-DFLASH2 speed gate, end to end, on a leased fleet device.
#
# WHY THIS FILE EXISTS
# --------------------
# `.agents/specs/dflash2-spec-decode.md` `## Owed` O22 and O23 described this
# procedure in prose and committed none of it
# (https://github.com/mudler/vllm.cpp/issues/1562). The DFlash2 speed axis is
# recorded NOT TAKEN, and an unreproducible harness is why a number taken now
# would be worth little. This script is the runnable procedure: it takes the
# clock window, drives both arms on an identical workload, folds them into the
# machine-readable result `.agents/benchmark-record.md` can cite, and refuses
# out loud when a precondition is absent.
#
# THREE RULES THIS FILE OBEYS, EACH FROM A LOST RUN
# -------------------------------------------------
# 1. THE COMPLETION MARKER IS WRITTEN FROM `trap ... EXIT`, NEVER FROM THE
#    SUCCESS PATH. Seven waiters once blocked forever on a marker a dead harness
#    could not write. The marker here carries the exit STATUS, so a waiter reads
#    a verdict and never a presence.
# 2. STATE IS RESTORED FROM THE SAME TRAP. A dead harness once left a source
#    file mutated and unrestored. Nothing here mutates a source file, and the
#    clock sampler, which is the one background process, is stopped from the
#    trap on every path.
# 3. NEVER EDIT THIS FILE WHILE BASH IS EXECUTING IT. Bash re-reads a script
#    from its byte offset, so an edit mid-run runs a spliced program. Run
#    `bash -n` before every run; the gate below does it for you.
#
# THE LEASE, AND WHAT THE LEASE CANNOT DO
# ---------------------------------------
# `dgx:gpu0`, `thor:gpu0` and `orin:gpu0` are fleet devices. Claim one with
# `rc run` or `rc hold`; never `ssh` to one and run work directly, and never
# take `$GPU_LOCK` INSTEAD of a lease -- the fleet cannot see that mutex, and on
# 2026-08-17 two sessions held the same box through the two different mutexes
# and voided a whole speed axis.
#
# Inside a lease `nvidia-smi -lgc` returns LGC_RC=4 even as root (#1354): the
# worker's CapBnd holds no CAP_SYS_ADMIN. So on a fleet device the SM clock is
# SAMPLED and NOT PINNED, this script does not attempt to pin it, and a pairing
# may be refused on within-run spread with no lever to fix it. That is a real
# outcome and it is recorded rather than worked around.
#
# USAGE
#
#   scripts/dflash2-speed-gate.sh --self-check          # bash -n, no GPU, no lease
#   scripts/dflash2-speed-gate.sh \
#       --evidence /workspace/evidence/dflash2-speed \
#       --target   /workspace/ckpt/qwen3.8-27b-hf \
#       --draft    /workspace/dflash2/draft-st \
#       --oracle-commit 3406ec1d0000000000000000000000000000000 \
#       --attention-backend TRITON_ATTN \
#       --artifact target=/workspace/ckpt/qwen3.8-27b-hf/model.safetensors=<sha256> \
#       --our-binary /workspace/build/bin/vllm-cli \
#       --our-model  /workspace/dflash2/target-gguf \
#       --our-artifact our_target=/workspace/dflash2/target-gguf/model.gguf=<sha256> \
#       --our-speculative-config '{"method":"dflash","model":"/workspace/dflash2/draft.gguf","num_speculative_tokens":7}' \
#       --our-build-recipe "cmake --preset cuda-release -DVLLM_CPP_CUDA_ARCH=121a" \
#       --oracle-build-recipe "pip install -e . at <the oracle head>" \
#       --num-speculative-tokens 7
#
# `--artifact` names the ORACLE'S weights and `--our-artifact` names ours; each
# arm refuses when no entry it was given identifies a model it loads.
# `--our-speculative-config` is REQUIRED, and the k inside it is the k recorded.
#
# `--lease-id` defaults to `$RC_LEASE_ID`, which `rc` exports into the job.

set -euo pipefail

MARKER=""
CLOCK_PID=""
SELF_CHECK=0
EVIDENCE=""
TARGET=""
DRAFT=""
ORACLE_COMMIT=""
ORACLE_BUILD_RECIPE=""
ATTENTION_BACKEND=""
OUR_BINARY=""
OUR_BUILD_RECIPE=""
OUR_MODEL=""
OUR_SPECULATIVE_CONFIG=""
LEASE_ID="${RC_LEASE_ID:-}"
CLOCK_INTERVAL="1.0"
REPEAT="5"
# THE DRAFT DEPTH, THREADED TO BOTH ARMS. It was exposed on neither, so this
# gate could only ever run at the oracle capture's default of 7: any other k in
# --our-speculative-config made the two workload fingerprints disagree and the
# summary refused. A refusal is the right failure and it is still a gate that
# cannot sweep k, which is the one knob this row's acceptance length turns on.
NUM_SPECULATIVE_TOKENS="7"
ASSUME_COMPUTE_PROCESSES=""
ARTIFACTS=()
# OUR ARM'S OWN WEIGHTS. The same --artifact list used to go to both arms, while
# the oracle loads an HF checkpoint directory and we load GGUF, so at most one
# arm was ever describing the weights it opened. Each arm now binds its model
# paths to its own artifact entries and refuses when nothing names them.
OUR_ARTIFACTS=()

# THE ONLY EXIT PATH THAT WRITES ANYTHING. Registered before the first action
# that can fail, so a refusal during argument parsing still produces a marker.
finish() {
  status=$?
  if [ -n "${CLOCK_PID}" ] && kill -0 "${CLOCK_PID}" 2>/dev/null; then
    kill -TERM "${CLOCK_PID}" 2>/dev/null || true
    wait "${CLOCK_PID}" 2>/dev/null || true
  fi
  if [ -n "${MARKER}" ]; then
    printf 'status=%s\nfinished_utc=%s\n' \
      "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MARKER}.tmp" || true
    mv -f "${MARKER}.tmp" "${MARKER}" 2>/dev/null || true
  fi
  if [ "${status}" -ne 0 ]; then
    echo "dflash2-speed-gate: REFUSED or FAILED with status ${status}" >&2
  fi
  return "${status}"
}
trap finish EXIT

usage() {
  # THE WHOLE HEADER, bounded by the `set -euo` line rather than by a line
  # number an edit invalidates silently.
  sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --self-check) SELF_CHECK=1; shift ;;
    --evidence) EVIDENCE="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --draft) DRAFT="$2"; shift 2 ;;
    --oracle-commit) ORACLE_COMMIT="$2"; shift 2 ;;
    --oracle-build-recipe) ORACLE_BUILD_RECIPE="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --artifact) ARTIFACTS+=("$2"); shift 2 ;;
    --our-artifact) OUR_ARTIFACTS+=("$2"); shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --num-speculative-tokens) NUM_SPECULATIVE_TOKENS="$2"; shift 2 ;;
    # TEST-ONLY. Stands in for the driver's compute-app query so the CPU suites
    # can drive this script without an `nvidia-smi` on PATH. A REAL run omits
    # it, and the harness then samples the driver itself.
    --assume-compute-processes) ASSUME_COMPUTE_PROCESSES="$2"; shift 2 ;;
    --our-binary) OUR_BINARY="$2"; shift 2 ;;
    --our-build-recipe) OUR_BUILD_RECIPE="$2"; shift 2 ;;
    --our-model) OUR_MODEL="$2"; shift 2 ;;
    --our-speculative-config) OUR_SPECULATIVE_CONFIG="$2"; shift 2 ;;
    --lease-id) LEASE_ID="$2"; shift 2 ;;
    --clock-interval) CLOCK_INTERVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# RULE 3, enforced rather than remembered. A spliced program is not a failed
# run, it is a run whose output nobody can explain.
if ! bash -n "$0"; then
  echo "dflash2-speed-gate: this script does not parse; refusing to run it" >&2
  exit 2
fi
if [ "${SELF_CHECK}" -eq 1 ]; then
  bash -n scripts/dflash2-speed-gate.sh
  # ALL THREE modules. Two of them were listed and `dflash2_our_arm.py` was not,
  # so appending garbage to the arm this gate is the numerator of still printed
  # PASS -- a self-check that does not check the file it drives.
  python3 -c "import ast,sys; [ast.parse(open(p).read(), p) for p in sys.argv[1:]]" \
    tools/bench/dflash2_speed_harness.py \
    tools/bench/dflash2_oracle_capture.py \
    tools/bench/dflash2_our_arm.py
  echo "dflash2-speed-gate: self-check PASS (syntax only; no GPU was touched)"
  exit 0
fi

for required in EVIDENCE TARGET DRAFT ORACLE_COMMIT ATTENTION_BACKEND OUR_BINARY OUR_MODEL; do
  if [ -z "${!required}" ]; then
    echo "dflash2-speed-gate: --${required,,} is required" >&2
    exit 2
  fi
done
if [ "${#ARTIFACTS[@]}" -eq 0 ]; then
  echo "dflash2-speed-gate: at least one --artifact role=path=sha256 is required; a repo id is not a pin" >&2
  exit 2
fi
if [ -z "${LEASE_ID}" ]; then
  echo "dflash2-speed-gate: no lease id. Claim the device with \`rc run\`/\`rc hold\` first; never ssh to a fleet box" >&2
  exit 2
fi
if [ "${#OUR_ARTIFACTS[@]}" -eq 0 ]; then
  echo "dflash2-speed-gate: at least one --our-artifact role=path=sha256 is required. Our arm loads GGUF and the oracle loads an HF checkpoint, so one --artifact list cannot identify both" >&2
  exit 2
fi
# A SPEC-DECODE COMPARISON NEEDS A SPECULATIVE DECODE ON BOTH SIDES. This was
# optional, so our arm could run with no drafter at all and still fingerprint-
# match the oracle's k=7, and the ratio would have measured the feature rather
# than the implementation.
if [ -z "${OUR_SPECULATIVE_CONFIG}" ]; then
  echo "dflash2-speed-gate: --our-speculative-config is required; the oracle arm drafts and a ratio against a plain decode measures the feature, not this row" >&2
  exit 2
fi

mkdir -p "${EVIDENCE}"
MARKER="${EVIDENCE}/COMPLETED"
rm -f "${MARKER}"

# THE KEEPALIVE IS ASSERTED, NOT INHERITED. It defaults to 0 since #931, but a
# default is a claim about a binary and not a record of this run (#577).
export VT_SERVER_SSE_PING_S=0
# The hook must land on the object that drafts; the harness asserts the RESOLVED
# client class as well, because setting the variable is not the same claim.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

OUR_REVISION="$(git rev-parse HEAD)"

artifact_args=()
for artifact in "${ARTIFACTS[@]}"; do
  artifact_args+=(--artifact "${artifact}")
done

our_artifact_args=()
for artifact in "${OUR_ARTIFACTS[@]}"; do
  our_artifact_args+=(--artifact "${artifact}")
done

assume_args=()
if [ -n "${ASSUME_COMPUTE_PROCESSES}" ]; then
  assume_args+=(--assume-compute-processes "${ASSUME_COMPUTE_PROCESSES}")
fi

common_args=(
  --target "${TARGET}"
  --draft "${DRAFT}"
  --oracle-commit "${ORACLE_COMMIT}"
  --oracle-build-recipe "${ORACLE_BUILD_RECIPE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --lease-id "${LEASE_ID}"
  --our-revision "${OUR_REVISION}"
  --our-build-recipe "${OUR_BUILD_RECIPE}"
  # BOTH ARMS REPEAT THE SAME NUMBER OF TIMES. Each folds a median over its warm
  # legs, and two medians over differently sized populations do not divide.
  --repeat "${REPEAT}"
  # BOTH ARMS DECLARE THE SAME k. Our arm reads the value the binary actually
  # got out of --speculative-config and refuses when the two disagree, so this
  # flag is a cross-check on our side and the source of truth on the oracle's.
  --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}"
  "${artifact_args[@]}"
  ${assume_args[@]+"${assume_args[@]}"}
)

# BUILT HERE, not beside the run, so the SAME list is prechecked and executed.
# Our arm carries its own `--artifact` list, its own `--model` and its own
# `--speculative-config`, and `checkpoint_reasons`, `model_binding_reasons` and
# `speculative_config_reasons` are first evaluated on them inside
# `dflash2_our_arm.precheck`. Prechecking only the oracle arm left every one of
# those first evaluated AFTER the oracle's full load and timed run, which is the
# failure the banner below says does not happen here.
our_common_args=(
  --binary "${OUR_BINARY}"
  --model "${OUR_MODEL}"
  --lease-id "${LEASE_ID}"
  --our-revision "${OUR_REVISION}"
  --our-build-recipe "${OUR_BUILD_RECIPE}"
  --repeat "${REPEAT}"
  --speculative-config "${OUR_SPECULATIVE_CONFIG}"
  --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}"
  "${our_artifact_args[@]}"
  ${assume_args[@]+"${assume_args[@]}"}
)

echo "== precheck BOTH ARMS (no GPU work yet; the failure that costs a lease is found before the lease)"
python3 -m tools.bench.dflash2_oracle_capture "${common_args[@]}" --precheck-only \
  > "${EVIDENCE}/precheck.json"
python3 -m tools.bench.dflash2_our_arm "${our_common_args[@]}" --precheck-only \
  > "${EVIDENCE}/precheck-ours.json"

# ONE WINDOW PER ARM, never one window spanning both: a single window cannot
# see the cross-arm offset, and the offset is the term that transfers into the
# ratio (#543). `gpu_clock_state` refuses to overwrite existing evidence, so a
# rerun into a used evidence directory stops rather than silently blending two
# runs.
open_clock_window() {
  python3 -m tools.bench.gpu_clock_state sample \
    --output "${EVIDENCE}/clock-$1-samples.jsonl" \
    --summary "${EVIDENCE}/clock-$1.json" \
    --interval "${CLOCK_INTERVAL}" &
  CLOCK_PID="$!"
}

close_clock_window() {
  if [ -n "${CLOCK_PID}" ]; then
    kill -TERM "${CLOCK_PID}" 2>/dev/null || true
    wait "${CLOCK_PID}" 2>/dev/null || true
    CLOCK_PID=""
  fi
}

echo "== vLLM arm, PRODUCTION configuration (never --enforce-eager)"
echo "   clock is SAMPLED, not pinned: LGC_RC=4 inside a lease even as root (#1354)"
open_clock_window vllm
python3 -m tools.bench.dflash2_oracle_capture "${common_args[@]}" \
  --clock-summary "${EVIDENCE}/clock-vllm.json" \
  --output "${EVIDENCE}/vllm-arm.json" || { close_clock_window; exit 1; }
close_clock_window

echo "== our arm, identical workload, through the public ABI (vllm-cli)"
open_clock_window ours
python3 -m tools.bench.dflash2_our_arm "${our_common_args[@]}" \
  --clock-summary "${EVIDENCE}/clock-ours.json" \
  --output "${EVIDENCE}/our-arm.json" || { close_clock_window; exit 1; }
close_clock_window

echo "== fold into the citable result"
python3 -m tools.bench.dflash2_speed_harness summarize \
  --ours "${EVIDENCE}/our-arm.json" \
  --vllm "${EVIDENCE}/vllm-arm.json" \
  --output "${EVIDENCE}/dflash2-speed.json"

echo "dflash2-speed-gate: result written to ${EVIDENCE}/dflash2-speed.json"
echo "Cite it from .agents/benchmark-record.md; the marker carries the exit status."
