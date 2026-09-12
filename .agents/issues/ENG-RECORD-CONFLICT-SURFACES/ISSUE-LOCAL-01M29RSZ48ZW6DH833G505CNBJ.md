ID: ISSUE-LOCAL-01M29RSZ48ZW6DH833G505CNBJ
Title: check-pr-size forces every RUNNABLE_BASELINE re-pin through one shared test file
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

scripts/check-pr-size.py:704-722 requires any change touching a governance_checker path to also change that checker's recognized_evidence file with lines > 0 in the same diff. For scripts/check-gate-commands.py that file is tests/scripts/test_check_gate_commands.py. Every row that enters the RUNNABLE_BASELINE population must therefore edit one shared test file, which is the shape AGENTS.md Records names as defective: 'A gate often creates the lock. If a checker requires every change to edit one shared file, the checker is defective. Move the obligation to a per-row surface. Do not delete the obligation.'

Measured, not argued. The same ERROR fires on three already-merged commits of the same shape, so this is repo-wide behaviour rather than one row's mistake:

  d1256c2fd  BACKEND-TENSTORRENT-KEEPQUANT   fires (also on check-agent-record.py)
  2f4001199  QUANT-EXL3-PERF                 fires (also on check-agent-record.py)
  aa7bcc8c6  QUANT-EXL3-MUL1                 fires

Found on PR #3149 (row QUANT-GGUF-IQ4_NL), where the baseline move was genuinely required: deleting the single line 'QUANT-GGUF-IQ4_NL' from RUNNABLE_BASELINE reds 17 tests, including test_the_baseline_matches_the_shipped_record, test_the_baseline_re_pin_is_load_bearing and test_hf_model_download_earns_its_runnable_baseline_entry. So the ratchet is doing its job and the evidence requirement is the part that does not fit.

One further fact a reader needs, and one CORRECTION this issue makes against its own first draft.

The fact: agent-preflight.sh SKIPS check-pr-size for want of --base/--head/--branch, so no local gate surfaces this at all and the first signal is CI. That is how PR #3149 reached a fresh review with it outstanding.

The correction: this issue first asserted that check-pr-size 'prints ERROR and EXITS 0', and that is FALSE. Measured both ways at the same revision:

  python3 scripts/check-pr-size.py ... | tail -2   ->  $? == 0
  python3 scripts/check-pr-size.py ... > file      ->  $? == 1

main() returns 1 when errors is non-empty. The zero came from reading $? after a PIPE, which reports the exit status of tail and not of the checker. The wrong reading was the author's, not the tool's, and it is recorded here rather than quietly deleted because a filed defect that misstates the tool it accuses is worse than no filing. Nothing else in this issue depends on it: the shared-file lock is a property of check-pr-size.py:704-722 and is unaffected by how the process exits.

The gate is also STRICTER THAN THE RULE IT ENFORCES. AGENTS.md says a semantic checker change needs 'a spec, a red-before test OR MUTATION, and green-after evidence'. A mutation satisfies AGENTS.md; check-pr-size accepts only a changed line in one named test file, which a pure data re-pin cannot produce without adding a per-row test class to that shared file.

Not filed with a fix. The repair is a design question: whether a data-only re-pin of a baseline list should count as a semantic checker change at all, whether the evidence surface should become per-row (one file per row, read with a glob, as Records prescribes), and whether the checker should exit non-zero when it prints ERROR. That wants a spec and a red-before case.

## Resolution

-
