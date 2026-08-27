# The Windows test gate names the test case it dies in

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#584](https://github.com/mudler/vllm.cpp/issues/584)

Parent specification:
[windows-baseline-coverage.md](windows-baseline-coverage.md), which narrowed
#584 and lists it under `## Owed`. Sibling:
[windows-test-thread-raii.md](windows-test-thread-raii.md), which landed the
in-process half of the reporting repair and explicitly did not claim to cure
#584.

Status: `ACTIVE`. Base `331eda8887e6a5c06244944c328b949b035cce4a`.

## Scope

`test_openai_api_server.exe` fast-fails with `0xC0000409` on both Windows lanes
and names no test case. This spec makes the **harness** name it, because the
**process** provably cannot.

In: `scripts/build-windows-release.ps1` — a localiser that runs a failed doctest
executable one test case per process, printing each case's name from PowerShell
before the child is launched, and its exit code after. In: PowerShell contract
tests for that function, in the script's existing `-ContractTest` mode.

Out: `src/`, `include/`, `tests/`, and the CI workflow. Out: any change to which
tests run in the passing path, and any change to the failure's disposition — the
lane stays red, the localiser runs only after the failure and then rethrows.

This is an **instrument**. It is not a fix for #584, and the issue stays open
when it lands.

## Why the process cannot report this itself

- `__fastfail` bypasses SEH by design, so doctest's Windows handler never runs.
- The vendored doctest contains **no** `std::flush` and no `.flush()` anywhere
  in its reporters (`third_party/doctest/doctest.h`; measured by grep on
  2026-08-27). Its `ConsoleReporter` writes to a buffered `stdout`, which
  `__fastfail` discards. That is why the entire job output is the version banner
  while the server's `std::cerr` lines — unit-buffered, through
  `src/vllm/entrypoints/openai/request_logger.cpp:26` — all arrive.
- So no in-process reporter, listener, or added flush can be trusted to survive
  the death it is trying to describe. The name has to be printed by a process
  that is not the one dying.

## Design

`Invoke-DoctestCaseLocaliser -Program <exe> [-Runner <scriptblock>]`:

1. `--count --order-by=file` gives `N`, the number of cases passing the filters.
2. For `i` in `1..N`:
   - `--list-test-cases --order-by=file --first=$i --last=$i` gives that case's
     **name**. doctest applies the `first`/`last` range check *before* both the
     `count` and the `list_test_cases` branches (`doctest.h:6010-6031`), so the
     index that names a case is the index that runs it. The alignment is a
     property of one loop, not of two orderings agreeing.
   - `Write-Host` the index and name **before** launching, so the line is in the
     job log whatever the child does.
   - Run `--order-by=file --first=$i --last=$i` and record the exit code.
3. Report every index that exited nonzero, by name.

Two outcomes, both answers:

- **Some index dies.** That case is named, and #584 has a source location.
- **No index dies.** Every case passes in isolation, so the fault needs state
  accumulated across cases — port or handle exhaustion, a leaked thread, a
  static destroyed twice. That eliminates the whole single-case class and is not
  a null result.

`-Runner` mirrors `Invoke-Checked`'s injection seam so the function is testable
without a doctest binary. It is invoked as `& $Runner $Program $Arguments` and
returns `@{ Output = <string[]>; ExitCode = <int> }`.

The call site wraps only the test executables, in
`Invoke-CheckedTestExecutable`: run through `Invoke-Checked`, and on a throw run
the localiser and **rethrow**. Swallowing the failure would be the
delete-the-assertion move `AGENTS.md` forbids.

## Risks

- **Cost.** `test_openai_api_server` has 77 `TEST_CASE`s. On failure the
  localiser adds ~77 process launches. It runs only after a failure, so the
  passing path costs nothing, and the lane it runs on is already red.
- **A hanging case.** A case that hangs rather than crashes would hold the job.
  The localiser bounds each child with a wall-clock timeout and reports a
  timeout as its own disposition rather than waiting.
- **A localiser that itself throws** would replace the real failure with its
  own. Every localiser step is inside a `try`, and its own failure is reported
  and then followed by the rethrow of the original.

## Tests and evidence

PowerShell contract tests in the existing `-ContractTest` mode, all runnable on
Linux with `pwsh` and run on both Windows lanes by `ci.yml:1228,1253`:

1. A runner that fails only at index 3 makes the localiser report index 3, and
   only 3.
2. A runner where every index passes makes the localiser report the
   cumulative-state outcome explicitly, not silence.
3. The naming call and the run call for one iteration carry the **same**
   `--first`/`--last` index — the off-by-one that would name the wrong case.
4. `Invoke-CheckedTestExecutable` rethrows after localising: a failing program
   still throws.
5. A localiser whose own enumeration throws does not suppress the original
   failure.

Red-before evidence is captured by running the contract tests against the tree
with the function bodies stubbed out, then again after they are written.

## Gates

- `pwsh -NoProfile -File scripts/build-windows-release.ps1 -ContractTest`
- `scripts/agent-preflight.sh --fail-on-skip`

## Stop conditions

- Stop if the localiser cannot be proven on a real doctest binary; the
  `--first`/`--last` semantics are the load-bearing assumption and are pinned by
  measurement in `## Measured` rather than read from documentation.
- Stop and escalate rather than making the Windows lane green by any route that
  does not name and fix the defect.

## Measured

On `build/tests/test_sampler` (Linux, doctest 2.5.2), 2026-08-27:

- `--count --order-by=file` reports 15.
- `--first=N --last=N` for N in {1, 3, 15} executes exactly the 1st, 3rd and
  15th case listed by `--list-test-cases --order-by=file`, confirmed by
  `--duration` output naming each.
- `--list-test-cases --order-by=file --first=N --last=N` prints exactly that one
  name, for the same N.

End-to-end on Linux with `pwsh` 7.6.5, driving the **committed function text**
(extracted from `scripts/build-windows-release.ps1` by its own AST, so nothing is
transcribed), 2026-08-27:

- Against the real `build/tests/test_sampler`: reads 15, names all 15 correctly,
  runs each in its own process, reports `EveryCasePassed=True`.
- Against a stub that emulates `--count` / `--list-test-cases` and then raises
  `SIGABRT` at index 3: `FAILING CASE 3/5 rc=134 : synthetic case 3`. That is the
  Windows scenario in the shape Linux can raise it — `abort()` reaching the
  parent as a nonzero child status while the child prints nothing.

Nine mutations of the shipped functions, each applied to a scratch copy, verified
as applied, and restored byte-for-byte (sha256 equal after each), all detected by
`-ContractTest` with rc 1: off-by-one on the run index; dropping `--order-by`
from the run call; swallowing the failure instead of rethrowing; treating a
timeout as a pass; a process runner that never reports a timeout; a process
runner that drops the child exit status; running the localiser on a passing
executable; removing the wall-clock bound entirely; and honouring the bound 30 s
LATE, which is the one that separates *reporting* a timeout from *enforcing*
one and is caught by its own assertion rather than by a neighbour's.

Localisation of #584 as it stands at `331eda888`, from jobs `98643239944`
(`windows-msvc-cpu`) and `98643239723` (`windows-msvc-vulkan`), both at
`d8404ff29`:

- Both lanes log exactly 33 `/v1/chat/completions` ingress lines, the same 33
  byte counts in the same order, ending `420`.
- A 420-byte body is `json::dump` of the request built at
  `tests/vllm/entrypoints/openai/test_api_server.cpp:2151-2157` — computed, not
  matched by eye: `max_tokens`, two messages of 200 and 89 `h`, `model`,
  `temperature`. It is posted at `:2158-2159`, inside
  `SUBCASE("/v1/chat/completions: refused on the summed message BYTES")`.
- Job `97120875302` of 2026-08-23 ends on the **same** `body_bytes=420` line,
  four days and two test-file commits earlier.

This **refutes** the window recorded on the issue and in
`.agents/issue-index.md`, which places the fault between the old `:1294` and
`:1325` — the socket-smoke case's teardown. That case is today's `:1721-:1787`;
its request is logged, and the `:2159` request is logged after it. The earlier
reading anchored on the last *chat* line available in a tree where the byte-bound
case did not yet exist and assumed adjacency.

`.agents/issue-index.md` is append-only, so the correction is not made by
editing that row. It lives here and on the issue, which is where a reader of the
row is sent.

The bound this evidence actually supports is `:2159` → `:2965`, the next chat
ingress. That is about thirty test cases, dominated by the `/v1/videos` block,
and it is why the localiser is the deliverable rather than a fix.

## Owed

- The cure for #584 itself. This spec buys the source location, not the repair.
- The same localiser for the other Windows test executables is already covered,
  because the call site wraps the loop rather than one binary. Nothing is owed
  there.

## Now

`ACTIVE`. The proof of this change is a Windows CI run, which the authoring
session could not execute; see the pull request body.
