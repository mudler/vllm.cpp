# Spec — the env-doc gate runs in both directions — `ENG-GATE-ENV-DOC`

Issue: [#2389](https://github.com/mudler/vllm.cpp/issues/2389)
Row: `ENG-GATE-ENV-DOC` (engine-matrix)
State: `GATING`

`scripts/check-env-doc.py` computed `scanned - documented - allowlisted`. That
catches a variable READ but not documented. The reverse — DOCUMENTED in the
user-facing table and read by nothing — was unguarded, so a knob could outlive
its implementation indefinitely and nothing would say so. The failure is silent
in the worst way: the operator sets the variable, nothing happens, and the page
still states a default, a formula and tuning advice for it.

Two live instances proved the gap was not theoretical.
`VT_QWEN35_STAGE_MIN_FREE_FRAC` lost its reader when the Qwen3.5/3.6 staging
policy was rewritten to a total-memory rule; its doc row survived, and its only
occurrence in compiled code is a `//` comment. `VT_GEMMA4_MLP_MOE_PARALLEL` was
never wired at all: it existed only in the deferred layer-loop path PR #140 did
not ship, and no tree in this repository has ever contained the name.

## Spike

| Section | Content |
|---|---|
| Scope | IN: the reverse assertion in `scripts/check-env-doc.py` — every variable in a user-facing TABLE of `docs/ENVIRONMENT.md` is read by at least one file CMake compiles, as more than a comment; the `UNREAD_EXCEPTIONS` declared escape, with its stated-reason and staleness guards; the reverse cases in `tests/scripts/test_check_env_doc.py`; the removal of the `VT_GEMMA4_MLP_MOE_PARALLEL` row the new direction reports; the `docs/ENVIRONMENT.md` §"Keeping this reference honest" text that states what the gate now asserts. OUT: the FORWARD direction, whose scan roots, regex and predicate are untouched byte-for-byte, so no existing classification moves; the `VT_QWEN35_STAGE_MIN_FREE_FRAC` doc row, owned by `ENG-WEIGHT-RESIDENCY` / [#2385](https://github.com/mudler/vllm.cpp/issues/2385) and deliberately not edited here; any C++ change (this row touches Python and documents only). |
| Upstream chain | n/a — a local record gate, not an upstream mirror. vLLM has no equivalent surface: its env vars live in one `vllm/envs.py` dict, where the declaration IS the reader, so the divergence this gate catches cannot exist there. The mirrored structure is this repository's own `scripts/check-readme-structure.py` / `scripts/check-fusion-consistency.py` shape: a pure predicate the suite calls directly, plus a `main()` the suite drives end to end. |
| Our baseline | `scripts/check-env-doc.py` before this change: `scan_env_names` over `src/` + `include/`, `documented_names` over the whole of `docs/ENVIRONMENT.md`, `allowlisted_names`, and one predicate `undocumented_env_vars(scanned, documented, allowlisted) = sorted(scanned - documented - allowlisted)`. `tests/scripts/test_check_env_doc.py` carried 8 cases, all of the forward direction. Measured on the tree: 398 scanned names, 186 names in a doc table, 3 of those read by nothing under `src/` + `include/`. |
| Port map | New in `scripts/check-env-doc.py`: `READ_SITE_ROOTS = ("src", "include", "examples")`; `strip_comments(text)` (a literal-aware C/C++ comment stripper); `scan_read_sites(root)` (comment-stripped, quoted-literal, over `READ_SITE_ROOTS`); `table_documented_names(text)` (first column of a markdown table row); `unread_documented_vars(table, read, exceptions)`; `stale_unread_exceptions(...)`; `unreasoned_unread_exceptions(...)`; `UNREAD_EXCEPTIONS`. `main()` runs both directions and accumulates, so one failure does not hide the other. `docs/ENVIRONMENT.md` loses the `VT_GEMMA4_MLP_MOE_PARALLEL` row and gains the prose that says why. Nothing in the forward path is edited. |
| Tests to port | No upstream suite to port; the evidence is this repository's mutation contract. `tests/scripts/test_check_env_doc.py` grows a `UnreadDocumentedVarTests` class of 22 cases: the predicate's four outcomes; the two staleness shapes and the unreasoned-entry shape; four `strip_comments` unit cases including a `//` inside a string literal and a C++14 digit separator; a synthetic-tree case pinning the `strip_comments` CALL SITE inside `scan_read_sites`; the three real-tree complications named on the issue; the table parser versus prose; the shipped-tree sweep; the `VT_GEMMA4_MLP_MOE_PARALLEL` repair; a fabricated dead row; and an end-to-end `main()` case that fails when the reverse call site is deleted. |
| Gates | `python3 scripts/check-env-doc.py` → rc 0, `OK: all 398 production env vars are documented or classified kernel-internal.` / `OK: all 185 env vars documented in a user-facing table are read by compiled code (1 declared exception(s)).` `python3 tests/scripts/test_check_env_doc.py` → `Ran 30 tests` `OK`, rc 0. RED-BEFORE (the checker-evidence contract): the HEAD suite against the BASE checker from `origin/main` → `FAILED (errors=21)`, rc 1, `AttributeError: module 'check_env_doc' has no attribute 'scan_read_sites'`, with `__pycache__` deleted between swaps. Five mutations, each restored byte-for-byte: main()'s reverse call site deleted → 1 failure; `strip_comments` dropped from `scan_read_sites` → 1; `examples` dropped from `READ_SITE_ROOTS` → 2; the stated-reason predicate defeated → 1; the staleness predicate defeated → 2. `python3 scripts/check-agent-record.py` and `python3 tests/scripts/test_agent_record.py` → rc 0. No build was run and none is needed: the change is Python and Markdown. |
| Dependencies | [#2385](https://github.com/mudler/vllm.cpp/issues/2385) / `ENG-WEIGHT-RESIDENCY` owns the `VT_QWEN35_STAGE_MIN_FREE_FRAC` doc row and is removing it; this row holds that name in `UNREAD_EXCEPTIONS` until then and its entry self-clears. `scripts/check-pr-size.py` supplies the checker-evidence contract this change satisfies and is not edited: `scripts/check-env-doc.py` already classifies `governance_checker` and its recognized evidence path already resolves to `tests/scripts/test_check_env_doc.py`. No new file is added, so no path needs classifying. No GPU, no model, no network. |
| Work breakdown | (1) the reverse predicate and its two guard predicates; (2) `strip_comments` and `scan_read_sites`; (3) `table_documented_names`; (4) `UNREAD_EXCEPTIONS` with the #2385 entry; (5) `main()` wiring that accumulates both directions; (6) the 22 suite cases; (7) the RED-before / GREEN-after swap and the five mutations; (8) the `VT_GEMMA4_MLP_MOE_PARALLEL` row removal and the honesty-section rewrite; (9) records — this spec, the matrix row, and the state expectation in `tests/scripts/test_agent_record.py`. One claim, one branch. |
| Risks/decisions | **The escape lives in the checker, not in a new file.** A second `scripts/env-doc-*.txt` would be an unclassified path and one more shared surface; a dict beside the rule keeps the reason next to what it excuses, the way `check-pr-size.py` holds `CHECKER_EVIDENCE_OVERRIDES`. **The escape is self-clearing.** An entry with no reason fails, and so does one whose variable has left the tables or gained a reader, so an exception cannot quietly become permanent — which is the exact failure mode this row exists to close, reintroduced one level up. **`benchmarks/` and `tools/` are NOT read sites.** No CMake target builds them, so counting a read there would let a knob nobody can reach read as live; that is widening scope to make a red green, and it is refused. **The forward direction is untouched.** Stripping comments there would make the scanned set SMALLER, which is a weaker gate, so `scan_env_names` keeps its exact previous body. **Table rows, not prose.** A table row states a default and an effect and is the promise an operator acts on; prose that merely names a variable, and the deferred kernel-internal section that names families, are out of the reverse rule on purpose. |

## Now

`GATING`. Both directions are in `scripts/check-env-doc.py`, green on the tree,
and run by `scripts/agent-preflight.sh` and the CI `agent-record` job. One
declared exception remains, `VT_QWEN35_STAGE_MIN_FREE_FRAC`, and it clears
itself when [#2385](https://github.com/mudler/vllm.cpp/issues/2385) removes the
doc row it names.

## Outcome

**What was measured.** 186 variables sit in a user-facing table of
`docs/ENVIRONMENT.md`. Three of them were read by nothing under `src/` +
`include/`: `VT_BENCH_PRETOKENIZE`, `VT_GEMMA4_MLP_MOE_PARALLEL` and
`VT_QWEN35_STAGE_MIN_FREE_FRAC`. Widening the read sites to the third compiled
tree, `examples/`, resolves the first — it is read in `examples/bench/bench_core.h`,
the `vllm-bench` binary its row scopes it to — and leaves the two the issue
names. Comment stripping changed the count on this tree by zero, because the one
comment-only hit is written in markdown backticks rather than as a C string, and
the quoted-literal regex already excluded it. That is exactly why the stripper
has its own synthetic-tree case: on this tree its call site is not load-bearing,
and the next comment that quotes the name it discusses would pass without it.

**What was rejected, and why.** Adding `benchmarks/` and `tools/` to the read
sites was rejected: neither is `add_subdirectory`'d, so a read there is not a
read anybody can reach, and including them would turn a real dead knob green.
Holding the exceptions in a new `scripts/env-doc-unread-allowlist.txt` was
rejected: it would need a `classify_path` entry in `scripts/check-pr-size.py`,
which would drag that checker's own evidence contract into an unrelated change,
and it would separate the reason from the rule. Editing the
`VT_QWEN35_STAGE_MIN_FREE_FRAC` doc row was rejected: #2385 owns it, and two
branches editing one row is the conflict this repository keeps paying for.
Caveating the `VT_GEMMA4_MLP_MOE_PARALLEL` row instead of deleting it was
rejected: it already carried the caveat "Not wired in this PR tip" and still sat
in the table with a default and an effect, which is the shape of the promise the
gate exists to refuse.

**Why each default has its value.** `READ_SITE_ROOTS` is the CMake target set,
not a convenience list. The reverse rule reads the FIRST column of a table row,
because that is where the doc's own convention puts the variable being promised.
`UNREAD_EXCEPTIONS` starts with exactly one entry, and that entry names the row
and issue that will delete it.
