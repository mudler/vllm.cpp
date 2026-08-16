#!/usr/bin/env bash
# One command to run at session start and again before committing.
#
# The record gates are cheap, deterministic and CPU-only, but they were spread
# across a dozen invocations that an agent had to remember individually. A
# forgotten gate becomes a red build after the push, which is the expensive
# moment to find out. This runs all of them and prints the resume digest.
#
#   scripts/agent-preflight.sh              # gates + role + print .agents/NOW.md
#   scripts/agent-preflight.sh --staged     # also check the staged change
#   scripts/agent-preflight.sh --quiet      # gates only, no digest
#   scripts/agent-preflight.sh --no-require-role  # tolerate an undeclared role
#   scripts/agent-preflight.sh --role-only  # ONLY the role gate; NOT a preflight
#
# A session declares a role, and an UNDECLARED one is a failing gate by default:
# the obligation used to live in prose and in an opt-in flag, and neither fired.
# read-only passes a plain run and fails --staged, because staging is writing.
#
# --role-only exists so the mutation suite can EXECUTE that gate instead of
# grepping it: text assertions catch a REWRITE of the default and miss an
# OVERRIDE on a later line. A nested FULL run is impossible -- this script runs
# the very suite that would call it, so it would recurse without bound -- hence
# a mode that runs the role block and stops. It is not an opt-out: it checks the
# ROLE strictly, which --no-require-role does not, and it never prints the "All
# gates green." banner, because it skips every record gate and mutation suite
# --no-require-role runs and so has not earned it. Neither mode is a superset of
# the other; --role-only is narrower and stricter, and says so on stdout.
#
# It never writes anything, so it is always safe to run.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGED=0
QUIET=0
ROLE_ONLY=0
# ON by default: an undeclared session is a FAILING gate. The mutation suite
# anchors on THIS line (`^REQUIRE_ROLE=1$`) and refuses any line-anchored
# assignment of zero, quoted or not, so a silent revert of the default goes red
# however it is spelled. The opt-out in the arg loop is indented and therefore
# not line-anchored, which is what keeps the two distinguishable.
REQUIRE_ROLE=1
for arg in "$@"; do
  case "$arg" in
    --staged) STAGED=1 ;;
    --quiet) QUIET=1 ;;
    --require-role) REQUIRE_ROLE=1 ;;
    --no-require-role) REQUIRE_ROLE=0 ;;
    --role-only) ROLE_ONLY=1 ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

CHECKERS=(
  check-prompt-contract
  check-agent-record
  check-release-binary-contract
  check-release-workflow
  check-windows-release-state
  check-container-matrix
  check-container-workflow
  check-role-discipline
  claim-view
  check-readme-structure
  check-public-doc-tables
  check-model-checklist
  check-supported-models
  check-env-doc
  check-fusion-consistency
  check-fp4-resident-consistency
  check-cuda-op-arch-gate
  check-runner-routing-consistency
  check-surface-coverage
  check-test-registration
  check-snapshot-pins
  check-oracle-pins
  check-now-current
  check-gate-commands
)

SUITES=(
  test_check_prompt_contract
  test_agent_gates
  test_agent_record
  test_check_issue_index_append_only
  test_check_release_binary_contract
  test_release_manifest
  test_release_archive
  test_release_pipeline
  test_check_windows_release_state
  test_release_postpublish_audit
  test_check_container_matrix
  test_check_container_workflow
  test_release_index
  test_release_metadata
  test_release_accelerator_metadata
  test_release_macos_metadata
  test_release_windows_metadata
  test_cpu_release_gates
  test_agent_role
  test_agent_onboard
  test_agent_start
  test_claim_view
  test_upstream_inventory
  test_doc_checkpoint
  test_check_readme_structure
  test_check_public_doc_tables
  test_check_model_checklist
  test_check_supported_models
  test_check_env_doc
  test_checker_text
  test_check_fusion_consistency
  test_check_fp4_resident_consistency
  test_check_cuda_op_arch_gate
  test_check_runner_routing_consistency
  test_check_surface_coverage
  test_check_test_registration
  test_check_snapshot_pins
  test_check_oracle_pins
  test_cpu_x86_llamacpp_floor
  test_audit_live_rows
  test_check_gate_commands
  test_gpu_lock_one_truth
  test_main_baseline
)

failed=()

run() {
  local label="$1"
  shift
  if output=$("$@" 2>&1); then
    printf '  \033[32mok\033[0m   %s\n' "$label"
  else
    printf '  \033[31mFAIL\033[0m %s\n' "$label"
    printf '%s\n' "$output" | sed 's/^/         /' | head -12
    failed+=("$label")
  fi
}

echo "Session role:"
if role_line=$(python3 scripts/agent-role.py show 2>&1); then
  printf '  \033[32mok\033[0m   %s\n' "$role_line"
else
  printf '  \033[33m--\033[0m   %s\n' "$(printf '%s' "$role_line" | head -1)"
  printf '       This session has not declared a role. Run scripts/agent-start.py\n'
  printf '       for the canonical role interview and exact next actions.\n'
  if [ "$REQUIRE_ROLE" -eq 1 ]; then
    failed+=("role-undeclared")
  fi
fi

# read-only PASSES a plain preflight -- that is the point of the third answer.
# It fails --staged, because staging is writing.
if [ "$STAGED" -eq 1 ] && printf '%s' "$role_line" | grep -q 'role=read-only'; then
  printf '  \033[31mFAIL\033[0m read-only sessions do not write. Claim operator or helper first.\n'
  failed+=("read-only-cannot-stage")
fi

if [ "$ROLE_ONLY" -eq 1 ]; then
  echo "ROLE CHECK ONLY -- this is NOT a full preflight; no record gate ran."
  if [ "${#failed[@]}" -ne 0 ]; then
    echo "${#failed[@]} gate(s) failed: ${failed[*]}"
    exit 1
  fi
  exit 0
fi

echo "Record gates:"
for checker in "${CHECKERS[@]}"; do
  case "$checker" in
    # Both default to a REPORT that exits 0 whatever the record says; the gate
    # is the flag. Wiring either without --check installs a gate that cannot
    # fail, which for check-gate-commands is the very defect it classifies.
    claim-view|check-gate-commands) run "$checker" python3 "scripts/$checker.py" --check ;;
    *) run "$checker" python3 "scripts/$checker.py" ;;
  esac
done

run "ready-for-helper" python3 scripts/ready-for-helper.py --check
run "upstream-inventory" python3 scripts/upstream-inventory.py --check
# Every ACTIVE row still has real Git evidence behind it. Wired only AFTER the
# record was repaired (P0), so it never had to be relaxed to pass: a red here
# means the record drifted, never that the gate is too strict.
#
# It is UNCONDITIONAL, and that is the deliberate call. The audit aborts when
# origin/main does not resolve (a clone whose remote is not named `origin`, a
# shallow/detached checkout), so preflight goes RED there rather than skipping.
# A skip would have to survive the "All gates green." banner below, and a green
# preflight that never verified the record is the one unacceptable outcome --
# the same reason audit-live-rows.py refuses to treat absence of information as
# absence of work. The repair is one command and the abort message names it, so
# the red is actionable. Preflight itself must NOT fetch: it is documented above
# as never writing anything, and a gate that mutates refs to make itself pass is
# exactly the shape this protocol forbids.
run "audit-live-rows" python3 scripts/audit-live-rows.py --check

echo "Mutation suites:"
for suite in "${SUITES[@]}"; do
  run "$suite" python3 "tests/scripts/$suite.py"
done
run "trailer suites" python3 -m unittest \
  tests.scripts.test_check_commit_trailers
run "commit style suites" python3 -m unittest \
  tests.scripts.test_check_commit_style

# The COMMITTED range, checked the way CI checks it. Deliberately OUTSIDE the
# --staged block: `--staged` inspects staged paths and is therefore VACUOUS after
# `git commit`, which is when preflight normally runs -- so the obligation went
# unchecked for a whole series and PR #80 landed eight commits that reddened
# documentation-checkpoint on main, where a diff-scoped range is never re-covered.
# Gating this on --staged would reproduce that hole exactly.
if git rev-parse --verify -q origin/main >/dev/null 2>&1 &&
   [ "$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)" -gt 0 ]; then
  echo "Committed range vs origin/main:"
  run "now-current range" python3 scripts/check-now-current.py \
    --base origin/main --head HEAD
  run "doc-checkpoint range" python3 scripts/check-doc-checkpoint.py \
    --base origin/main --head HEAD
  run "issue-index append-only" python3 scripts/check-issue-index-append-only.py \
    --base origin/main --head HEAD
fi

# Trailer enforcement reads only committed Git objects.
if git rev-parse --verify -q origin/main >/dev/null 2>&1 &&
   git merge-base --is-ancestor origin/main HEAD &&
   [ "$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)" -gt 0 ]; then
  echo "Commit trailers vs origin/main:"
  run "commit-trailers" python3 scripts/check-commit-trailers.py \
    --range "origin/main..HEAD"
  run "commit-style" python3 scripts/check-commit-style.py \
    --range "origin/main..HEAD"
fi

if [ "$STAGED" -eq 1 ]; then
  echo "Staged change:"
  run "doc-checkpoint --staged" python3 scripts/check-doc-checkpoint.py --staged
  run "now-current --staged" python3 scripts/check-now-current.py --staged
fi

if [ "${#failed[@]}" -ne 0 ]; then
  echo
  echo "${#failed[@]} gate(s) failed: ${failed[*]}"
  echo "Repair the record. Never weaken a checker to make a transition pass."
  exit 1
fi

echo
echo "All gates green."

if [ "$QUIET" -eq 0 ]; then
  echo
  echo "================ .agents/NOW.md ================"
  cat .agents/NOW.md
fi
