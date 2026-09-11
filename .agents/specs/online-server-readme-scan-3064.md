# Include the corrected README in the server artifact scan

Row: `SERVE-GATE-ONLINE`.
Issue: [#3064](https://github.com/mudler/vllm.cpp/issues/3064).
Parent spec: [CUDA online serving gate](cuda-online-serving-gate.md).
Base: `415d17859500caf2a4cac00511820e4f4760e86f`.

## Scope and source

Repair `tests/tools/test_online_gate_server_binary.py` after #3057 corrected
the README commands. Do not edit the README, production code, or other gates.
The server artifact name comes from `examples/CMakeLists.txt`.
No upstream sampling, benchmark, or oracle behavior changes.

`git log -S SCAN_BLOCKED_ON_POLICY` identifies `a7bb94402` as the exception's
introduction. `git log -S build/examples/server -- README.md` identifies
`415d17859` as the correction. The existing debt assertion fails on that
unchanged base because it requires the obsolete README command.

## Design

Remove the resolved README exception and its obsolete debt assertion.
Keep README in the existing root-file scan and preserve every stale-pattern
and legitimate-fallback rule.
Add a regression that injects a stale README command through the file reader
and executes the real repository scanner. The scanner must report the README
path and line. Do not replace the scanner's file discovery in the test.

## Tests and gates

Commit this spec before the regression and implementation.
Run the injected-README regression before removing the exception and retain
the expected failure. Then run the full server-binary suite and all tools tests.
Run `scripts/agent-preflight.sh` and report failures and skips separately.
In a scratch copy, remove README from root-file discovery. The new regression
must fail. Restore the copy and require the focused suite to pass.
The operator obtains fresh review and repeats the applicable gate before merge.

## Risks and stop conditions

Stop if the corrected README contains another actual stale artifact reference.
No blanket exclusion or stale-pattern weakening is permitted.
This repair needs no external compute, assets, or model execution.

## Now

ACTIVE: repair passes focused and full tools suites. Full preflight and fresh
review remain required before landing.

## Outcome

The existing debt assertion failed on unchanged `415d17859`. The new injected
README regression failed before the exception was removed.
After the repair, the server-binary suite passes 20 tests and the full tools
suite passes 799 tests. Removing README from discovery in a scratch copy makes
the new regression fail. Restoring the copied module restores its passing
result and byte equality with the candidate.
The scan retains every stale-artifact pattern and replay-fallback rule.
Removing only the failing debt assertion was rejected because that would leave
the README unscanned. No product default or oracle changes are needed.
