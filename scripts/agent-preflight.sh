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
#   scripts/agent-preflight.sh --require-role  # FAIL if the role is undeclared
#
# It never writes anything, so it is always safe to run.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGED=0
QUIET=0
REQUIRE_ROLE=0
for arg in "$@"; do
  case "$arg" in
    --staged) STAGED=1 ;;
    --quiet) QUIET=1 ;;
    --require-role) REQUIRE_ROLE=1 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

CHECKERS=(
  check-agent-record
  check-role-discipline
  claim-view
  check-readme-structure
  check-public-doc-tables
  check-model-checklist
  check-env-doc
  check-fusion-consistency
  check-runner-routing-consistency
  check-protocol-consistency
  check-state-order
  check-now-current
)

SUITES=(
  test_agent_record
  test_agent_role
  test_claim_view
  test_upstream_inventory
  test_doc_checkpoint
  test_check_readme_structure
  test_check_public_doc_tables
  test_check_model_checklist
  test_check_env_doc
  test_check_fusion_consistency
  test_check_runner_routing_consistency
  test_check_protocol_consistency
  test_check_state_order
  test_check_now_current
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
  printf '       declare it: scripts/agent-role.py claim operator | claim helper --row <ROW-ID>\n'
  if [ "$REQUIRE_ROLE" -eq 1 ]; then
    failed+=("role-undeclared")
  fi
fi

echo "Record gates:"
for checker in "${CHECKERS[@]}"; do
  case "$checker" in
    claim-view) run "$checker" python3 "scripts/$checker.py" --check ;;
    *) run "$checker" python3 "scripts/$checker.py" ;;
  esac
done

run "ready-for-helper" python3 scripts/ready-for-helper.py --check
run "upstream-inventory" python3 scripts/upstream-inventory.py --check

echo "Mutation suites:"
for suite in "${SUITES[@]}"; do
  run "$suite" python3 "tests/scripts/$suite.py"
done

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
