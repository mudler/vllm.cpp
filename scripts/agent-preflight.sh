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
#   scripts/agent-preflight.sh --fail-on-skip  # exit 1 when any gate was SKIPPED
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
# Every gate reports one of THREE states. `ok` ran and passed, `FAIL` ran and
# failed, `SKIP` did not run and says why. "All gates green." needs an empty
# FAIL list AND an empty SKIP list, because a banner over a block that never
# executed is a false report rather than a weaker gate (#998).
#
# The exit status carries only TWO of those states, so a SKIP still exits 0 and
# a reader who wants the third has to read the report. That is right for a human
# on a branch behind main and wrong for a program: `scripts/agent-ready.py` read
# preflight by exit code alone and called a run with two unexecuted trailer gates
# "green". --fail-on-skip is the opt-in for a machine consumer -- it exits 1 when
# anything was skipped, and it does not change what any gate demands or what a
# plain run reports. Ask for it when a skip must not read as success.
#
# It never writes anything, so it is always safe to run.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGED=0
QUIET=0
ROLE_ONLY=0
# OFF by default, and that default is the decision argued in the spec's §3.4: a
# branch behind origin/main is ordinary work, and a gate that fires on ordinary
# work is the defect. The flag exists because the exit status cannot carry the
# third state, so every consumer that reads only the return code reads a SKIP as
# success. There was exactly one such consumer (#998).
FAIL_ON_SKIP=0
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
    --fail-on-skip) FAIL_ON_SKIP=1 ;;
    -h|--help) sed -n '2,43p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

CHECKERS=(
  # FIRST deliberately. Every other gate reads a file for its own reason and
  # measures its own budget, so a table row that is half one branch and half
  # another satisfies all of them (#1417). This one asks whether a merge tool
  # wrote into a tracked file at all, and a reader who sees it fail knows to
  # stop reading the verdicts below.
  check-conflict-markers
  check-prompt-contract
  check-agent-record
  check-release-binary-contract
  check-release-workflow
  check-windows-release-state
  check-container-matrix
  check-container-workflow
  check-build-runtime-deps
  check-role-discipline
  claim-view
  check-readme-structure
  check-quickstart-recipes
  check-benchmark-index
  check-model-checklist
  check-supported-models
  check-env-doc
  check-fusion-consistency
  check-attention-rung-consistency
  check-fp4-resident-consistency
  check-cuda-op-arch-gate
  check-runner-routing-consistency
  check-surface-coverage
  check-test-registration
  check-snapshot-pins
  check-oracle-pins
  check-oracle-denominator-flags
  check-now-current
  check-gate-commands
  check-symbol-anchors
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
  test_check_build_runtime_deps
  test_validate_container_image
  test_release_index
  test_release_metadata
  test_release_accelerator_metadata
  test_release_macos_metadata
  test_release_windows_metadata
  test_cpu_release_gates
  test_agent_role
  test_agent_onboard
  test_agent_start
  test_gate_bringup
  test_env_agnostic_tooling
  test_claim_view
  test_upstream_inventory
  test_check_readme_structure
  test_check_quickstart_recipes
  test_check_benchmark_index
  test_check_model_checklist
  test_check_supported_models
  test_check_env_doc
  test_checker_text
  test_check_fusion_consistency
  test_check_attention_rung_consistency
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
  test_agent_preflight_skip_report
  test_agent_pr_body
  test_check_symbol_anchors
  test_check_oracle_denominator_flags
  test_check_conflict_markers
  test_prepush_checker_names
  test_ab_arms_differ
  test_ltx25_pixel_ab_harness
  test_ltx25_ab_memwatch
)

failed=()
# A gate that did NOT run is a THIRD state (#998). It is not `ok`, because
# nothing was verified, and it is not `FAIL`, because nothing was found wrong.
# Reporting it as neither is what let this script print "All gates green." over
# a block that never executed: three times in one session, twice because
# origin/main moved MID-RUN. See BASE_SHA below and the two range blocks.
skipped=()

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

# The same shape as run(), for a gate this run cannot execute. The REASON prints
# with it and is not optional: a reader who learns only that something did not
# happen cannot tell a stale checkout from a broken one.
skip() {
  local label="$1"
  shift
  printf '  \033[33mSKIP\033[0m %s\n' "$label"
  printf '%s\n' "$*" | sed 's/^/         /'
  skipped+=("$label")
}

# origin/main resolved ONCE, before any gate runs. Every range block below
# compares against this SHA and never against the ref.
#
# The ref is remote-tracking and therefore SHARED by every linked worktree of
# this checkout. A fetch in any other worktree moves it while this run is in
# flight, so `--is-ancestor origin/main HEAD` answered true at the top of the
# run and false by the time the trailer block asked. The trailer gates then
# vanished from a report that still said green, and the `ok` count fell from 76
# to 74 with no other change in the output. Pinning removes that race. Printing
# the SHA tells the reader which revision the verdict is about.
#
# Resolving to a commit (`^{commit}`) rather than to whatever the ref names
# keeps a tag or an annotated object from reaching a gate as a base.
BASE_REF="origin/main"
BASE_SHA="$(git rev-parse --verify -q "${BASE_REF}^{commit}" 2>/dev/null || true)"
BASE_UNRESOLVED="${BASE_REF} does not resolve here, so this run cannot tell which
commits are new. Fetch the remote, or name the remote that carries the base
branch, then rerun. Unknown is not absence."

# The other two questions about the pinned base, asked ONCE and beside the pin,
# with the git exit status kept rather than collapsed.
#
# `--is-ancestor` answers 1 for "no" and 128 for "that question cannot be asked
# here" -- an unborn HEAD (`git checkout --orphan`) is one way to reach it. Both
# used to take the same arm, so the report named a cause ("is not an ancestor of
# HEAD") that was not the cause.
#
# `rev-list --count` used to be spelled `|| echo 0`, which filed a FAILED count
# under the deliberate empty-range exemption of §3.6. An empty range withholds
# nothing and keeps the banner. A count that could not be taken withholds
# everything and must not. Mapping the second onto the first is the same
# unknown-as-success conflation this row exists to remove.
#
# The count keeps stderr OUT of its value, and the value is then validated. The
# repair for `|| echo 0` was first written as `2>&1`, which reintroduced the very
# defect through a narrower door: git can write to stderr AND exit 0, so the
# status catches nothing while the value carries the error text ahead of the
# number. `[ "$RANGE_COUNT" -gt 0 ]` does not evaluate false there, it ERRORS
# with status 2, which reads as false, and both range blocks fell through to
# their empty-range arm while "All gates green." printed over five dropped
# gates. A `.git/objects/info/alternates` naming a path that does not exist is
# one way to reach it: `error: unable to normalize alternate object path: ...`
# on stderr, the count on stdout, exit 0.
#
# Discarding stderr is what makes the VALUE right, and it is also what keeps a
# git that merely warns from costing five gates a SKIP it does not deserve.
# Validating the value is what makes an arm exist at all for a count that is not
# a count: without it, any non-numeric value reaches `-gt` and its status-2 error
# is indistinguishable from "zero commits". The two halves cover different
# failures, so both are here.
#
# Discarding stderr also had a COST, and the cost is paid separately rather than
# argued away. `2>/dev/null` threw the message away along with its influence on
# the value, so an unborn HEAD reported "exited 128 and printed [] on stdout"
# while git had already named the cause in one line. That report is honest and
# not actionable, and this row owes both. The rule is message-NOT-a-value, not
# no-message: the text goes into its own variable, which nothing compares and no
# arm reads, exactly as ANCESTRY_ERROR already did one line above.
ANCESTRY_ERROR=""
ANCESTRY_STATUS=0
RANGE_COUNT=""
RANGE_ERROR=""
RANGE_STATUS=0
if [ -n "$BASE_SHA" ]; then
  # stderr stays merged HERE deliberately. ANCESTRY_ERROR is a MESSAGE and never
  # a value: nothing compares it, and only ANCESTRY_STATUS selects an arm below.
  # Checked rather than assumed -- under the broken-alternates case above this
  # call also writes to stderr and exits 0, and every ancestry arm is unaffected.
  ANCESTRY_ERROR="$(git merge-base --is-ancestor "$BASE_SHA" HEAD 2>&1)"
  ANCESTRY_STATUS=$?
  # The count is asked TWICE, for two different things, and the order matters
  # twice over. `2>&1 >/dev/null` duplicates stderr onto the capture BEFORE
  # stdout leaves for /dev/null, so this call yields the message alone and never
  # the count. The value call runs second so RANGE_STATUS still describes the
  # command the value came from. Nothing selects an arm from RANGE_ERROR.
  RANGE_ERROR="$(git rev-list --count "${BASE_SHA}..HEAD" 2>&1 >/dev/null)"
  RANGE_COUNT="$(git rev-list --count "${BASE_SHA}..HEAD" 2>/dev/null)"
  RANGE_STATUS=$?
fi
# One predicate for "this run holds a usable commit count", so the two range
# blocks below cannot drift apart on what counts as unknown.
RANGE_NUMERIC=1
case "$RANGE_COUNT" in
  '' | *[!0-9]*) RANGE_NUMERIC=0 ;;
esac
ANCESTRY_UNKNOWN="git merge-base --is-ancestor ${BASE_SHA} HEAD exited
${ANCESTRY_STATUS}, so this run could not ask whether the base is behind HEAD:
${ANCESTRY_ERROR}
An unborn HEAD is one way to reach this. Unknown is not a verdict on ancestry."
# The closing sentence stays on ONE line deliberately. The suite asserts it
# verbatim, and a rewrap that splits it across a newline turns a correct report
# into a red gate rather than into a weaker one.
RANGE_UNKNOWN="git rev-list --count ${BASE_SHA}..HEAD exited ${RANGE_STATUS} and
printed [${RANGE_COUNT}] on stdout, which is not a commit count, so this run
could not count the commits under judgement. An unborn HEAD is one way to reach
this, and so is a git that writes an error to stderr and still exits 0. git
wrote this to stderr:
${RANGE_ERROR}
Unknown is not an empty range."

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
    # check-agent-record also carries the record-anchor ratchet
    # (ENG-RECORD-ANCHOR-RATCHET, #632): a citation that names a line no longer
    # holding the symbol beside it fails HERE, on the plain call, and the error
    # names the bucket that moved. Deliberately NOT wired as --report: `run`
    # shows only the first 12 lines of a failure, and the report's offender list
    # would push the error message out of that window. The full list is one
    # command away (`scripts/check-agent-record.py --report`) and CI prints it
    # unconditionally.
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
# THE ONE SUITE HERE WITH A THIRD-PARTY DEPENDENCY (#1612). It exercises
# `scripts/ltx25-render-compare.py`, whose only import beyond the standard
# library is numpy -- the tool reads PPM and WAV by hand precisely so that a
# leased worker needs nothing else. It ran on NO lane at all until now: absent
# from this array, from the enumerated python block in `.github/workflows/ci.yml`
# and from `tests/CMakeLists.txt`, while section 8 of its spec registered it as a
# gate. A gate no lane runs is "nothing lands dead" applied to the instrument.
#
# A missing numpy is a SKIP and never an `ok`: nothing was verified. CI installs
# it, so the lane that must not be silent is not the one that can be.
if python3 -c 'import numpy' >/dev/null 2>&1; then
  run "test_ltx25_render_compare" python3 tests/scripts/test_ltx25_render_compare.py
else
  skip "test_ltx25_render_compare" \
    "numpy is not importable here, and the tool this suite exercises needs it." \
    "CI installs python3-numpy and runs the same suite."
fi
run "trailer suites" python3 -m unittest \
  tests.scripts.test_check_commit_trailers
run "commit style suites" python3 -m unittest \
  tests.scripts.test_check_commit_style
# THE BENCHMARK-TOOL SUITES, which PREFLIGHT never ran until #1646 (#1648).
# `tests/tools/` holds the oracle pin, the clock-state assertions, the
# online-gate client and summary, and the serve-low request-set completeness.
#
# #1646 said no lane ran them at all. That was WRONG and #1648 corrects it:
# `tests/CMakeLists.txt` has registered them as the CTest target
# `test_serve_low_tools` since `e58858a91`, and CI's `build-test-cpu` runs
# `ctest --test-dir build` on every pull request. The claim came from grepping
# for `tests.tools`, the dotted module path, while CMake spells it
# `tests/tools` -- a null grep proving the terms wrong rather than the thing
# absent. The parity ledger's "all tools" citations were therefore citing a
# LIVE suite, not a dead one.
#
# The narrow gap was real and this line closes it: preflight ran none of them,
# so a local pre-edit check missed a red these suites would have caught, and
# only a full C++ configure-and-build surfaced it.
#
# DISCOVERED rather than enumerated. An enumeration is a shared list every new
# suite must edit, which is the record-lock shape AGENTS.md §Records forbids;
# discovery makes the file's existence the registration. Standard library only,
# no GPU and no wheel.
run "tools suites" python3 -m unittest discover -s tests/tools -t . -p "test_*.py"

# The COMMITTED range, checked the way CI checks it. Deliberately OUTSIDE the
# --staged block: `--staged` inspects staged paths and is therefore VACUOUS after
# `git commit`, which is when preflight normally runs -- so the obligation went
# unchecked for a whole series and PR #80 landed eight commits that reddened
# documentation-checkpoint on main, where a diff-scoped range is never re-covered.
# Gating this on --staged would reproduce that hole exactly.
#
# An EMPTY range is not a skipped gate and does not print SKIP. A verdict over
# zero commits withholds nothing, and reporting it as a skip would fire on the
# ordinary session-start run of a freshly cut branch. An UNRESOLVABLE base is
# the other case entirely: there are commits to judge and this run cannot tell
# which, so it reports SKIP and forfeits the banner.
if [ -z "$BASE_SHA" ]; then
  echo "Committed range vs ${BASE_REF}:"
  skip "now-current range" "$BASE_UNRESOLVED"
  skip "doc-checkpoint range" "$BASE_UNRESOLVED"
  skip "issue-index append-only" "$BASE_UNRESOLVED"
elif [ "$RANGE_STATUS" -ne 0 ] || [ "$RANGE_NUMERIC" -eq 0 ]; then
  echo "Committed range vs ${BASE_REF} ${BASE_SHA}:"
  skip "now-current range" "$RANGE_UNKNOWN"
  skip "issue-index append-only" "$RANGE_UNKNOWN"
elif [ "$RANGE_COUNT" -gt 0 ]; then
  echo "Committed range vs ${BASE_REF} ${BASE_SHA}:"
  run "now-current range" python3 scripts/check-now-current.py \
    --base "$BASE_SHA" --head HEAD
  run "issue-index append-only" python3 scripts/check-issue-index-append-only.py \
    --base "$BASE_SHA" --head HEAD
else
  echo "Committed range vs ${BASE_REF} ${BASE_SHA}: empty, HEAD adds no commits."
fi

# Trailer enforcement reads only committed Git objects.
#
# The ancestry arm is the one #998 was filed for. It USED to be spelled as a
# silent `&&` in the condition, so a base that was not an ancestor deleted both
# gates from the report and the run still printed "All gates green.". The guard
# itself stays, because check-commit-style.py refuses a non-ancestor base at
# validate_range (#999 owes that repair, and until it lands, dropping the guard
# would turn every branch behind main RED instead of honest). What changes is
# that the skip now SAYS SO and costs the banner.
if [ -z "$BASE_SHA" ]; then
  echo "Commit trailers vs ${BASE_REF}:"
  skip "commit-trailers" "$BASE_UNRESOLVED"
  skip "commit-style" "$BASE_UNRESOLVED"
elif [ "$ANCESTRY_STATUS" -gt 1 ]; then
  echo "Commit trailers vs ${BASE_REF} ${BASE_SHA}:"
  skip "commit-trailers" "$ANCESTRY_UNKNOWN"
  skip "commit-style" "$ANCESTRY_UNKNOWN"
elif [ "$ANCESTRY_STATUS" -ne 0 ]; then
  echo "Commit trailers vs ${BASE_REF} ${BASE_SHA}:"
  TRAILER_BEHIND="${BASE_REF} ${BASE_SHA} is not an ancestor of HEAD, so this
branch is behind it and the trailer gates did NOT run. Merge ${BASE_REF} and
rerun. Neither gate reported anything about this tree."
  skip "commit-trailers" "$TRAILER_BEHIND"
  skip "commit-style" "$TRAILER_BEHIND"
elif [ "$RANGE_STATUS" -ne 0 ] || [ "$RANGE_NUMERIC" -eq 0 ]; then
  echo "Commit trailers vs ${BASE_REF} ${BASE_SHA}:"
  skip "commit-trailers" "$RANGE_UNKNOWN"
  skip "commit-style" "$RANGE_UNKNOWN"
elif [ "$RANGE_COUNT" -gt 0 ]; then
  echo "Commit trailers vs ${BASE_REF} ${BASE_SHA}:"
  run "commit-trailers" python3 scripts/check-commit-trailers.py \
    --range "${BASE_SHA}..HEAD"
  run "commit-style" python3 scripts/check-commit-style.py \
    --range "${BASE_SHA}..HEAD"
else
  echo "Commit trailers vs ${BASE_REF} ${BASE_SHA}: empty, HEAD adds no commits."
fi

if [ "$STAGED" -eq 1 ]; then
  echo "Staged change:"
  run "now-current --staged" python3 scripts/check-now-current.py --staged
fi

# Printed BEFORE the failure summary and outside it, so a run that both failed
# and skipped reports both facts. They are different facts.
if [ "${#skipped[@]}" -ne 0 ]; then
  echo
  echo "${#skipped[@]} gate(s) SKIPPED: ${skipped[*]}"
  echo "NOT a green preflight: a skipped gate reported nothing about this tree."
  echo "Each reason is printed beside its SKIP above."
fi

if [ "${#failed[@]}" -ne 0 ]; then
  echo
  echo "${#failed[@]} gate(s) failed: ${failed[*]}"
  echo "Repair the record. Never weaken a checker to make a transition pass."
  exit 1
fi

# The opt-in, and the only thing that changes the exit status for a skip. The
# report above already says everything this line says. A caller reaches here
# because it reads the STATUS and not the report, which is what agent-ready.py
# did while printing the word "green" over two gates that never ran.
if [ "${#skipped[@]}" -ne 0 ] && [ "$FAIL_ON_SKIP" -eq 1 ]; then
  echo
  echo "--fail-on-skip: ${#skipped[@]} gate(s) did not run, so this run cannot"
  echo "answer the question that was asked of it. Exit 1 for the caller that"
  echo "reads only the status. Nothing here failed, and nothing here passed."
  exit 1
fi

# Reachable ONLY when both arrays are empty. A skip used to survive this line,
# which made the banner a claim the run had not earned (#998). The DEFAULT exit
# status stays 0 for a skip: a branch behind origin/main is ordinary work, and
# exit 1 would merge "a gate did not run" into the signal that means "a gate ran
# and failed". A caller that cannot read the report asks for --fail-on-skip.
if [ "${#skipped[@]}" -eq 0 ]; then
  echo
  echo "All gates green."
fi

if [ "$QUIET" -eq 0 ]; then
  echo
  echo "================ .agents/NOW.md ================"
  cat .agents/NOW.md
fi
