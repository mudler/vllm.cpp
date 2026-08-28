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

The call site keeps `Invoke-Checked (Join-Path $BuildDir "tests/Release/$test")
@()` **verbatim**, inside a `foreach ($test ...)` loop, and adds a `catch` that
calls `Invoke-DoctestCaseLocaliserSafely` and then `throw`s.

That shape is deliberate. `scripts/check-windows-portability.py:1210,1247` pins
the focused-test stage by exactly those two tokens, and an earlier draft that
routed the call through a new `Invoke-CheckedTestExecutable` broke the checker —
`\bInvoke-Checked\b` does not match `Invoke-CheckedTestExecutable`. Measured on
job `98673557390`: `ERROR: build-windows-release.ps1: missing active focused
tests in PowerShell AST`, which killed the lane before a test binary ran. The
repair is to keep the call site the checker already describes, not to widen the
checker to accept a new name; nothing in `check-windows-portability.py` changes.

`Invoke-DoctestCaseLocaliserSafely` never throws, so the instrument cannot change
the outcome of the gate it describes. The `throw` is at the call site, next to
the `Invoke-Checked` that failed, where a reader sees it — and it is asserted
from this script's own AST rather than transcribed.

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
4. `Invoke-DoctestCaseLocaliserSafely` runs the localiser on the program that
   failed, and a localiser that throws is reported and swallowed.
5. Read off this script's own AST: the focused-test loop runs `Invoke-Checked`,
   calls the localiser, and its `catch` contains a `throw`. Deleting that `throw`
   would localise a fast-failing binary and then wave it through.

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

Twelve mutations of the shipped functions and of the call site, each applied to a scratch copy, verified
as applied, and restored byte-for-byte (sha256 equal after each), all detected by
`-ContractTest` with rc 1: off-by-one on the run index; dropping `--order-by`
from the run call; swallowing the failure instead of rethrowing; treating a
timeout as a pass; a process runner that never reports a timeout; a process
runner that drops the child exit status; running the localiser on a passing
executable; removing the wall-clock bound entirely; and honouring the bound 30 s
LATE, which is the one that separates *reporting* a timeout from *enforcing*
one and is caught by its own assertion rather than by a neighbour's; deleting the
call site's `throw`; deleting its localiser call; and making
`Invoke-DoctestCaseLocaliserSafely` rethrow.

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

From the logs alone the supported bound was `:2159` onwards. The upper end could
not be fixed, and an earlier draft of this spec wrongly proposed `:2965` as one:
the chat call at `:2965` runs on a **transcription-only** server, where
`handle_chat_completions` returns its 500 at
`src/vllm/entrypoints/openai/api_server.cpp:307-311` **before**
`LogHttpIngress` at `:314`, so that call logs nothing and its silence proves
nothing. That is exactly why the instrument, and not a source bound, is the
deliverable.

## Result: the localiser answered on its first Windows run

`windows-msvc-cpu` job `98675485850` and `windows-msvc-vulkan` job
`98675485769`, both at the pull request's merge head, enumerate all 76 cases and
name the **same three** on both lanes, each reproducing the fault in its own
process:

| index | test case | exit |
|---|---|---|
| 56/76 | `api_server: an explicit-cpu device-selected engine serves /v1/completions` | `-1073740791` |
| 57/76 | `api_server: embeddings dispatch — OpenAI shape over the engine path` | `-1073740791` |
| 59/76 | `api_server: embeddings socket smoke; generate routes 404 on the embedding server` | `-1073740791` |

The original failure was rethrown and the job still ended `exit code 1`, so the
lane stayed red.

**The correlation is complete and it names one type.**
`vllm::entrypoints::LoadedEngine` is constructed at exactly two sites in that
file — `tests/vllm/entrypoints/openai/test_api_server.cpp:3101` (case 56) and
`:3146` inside `EmbedHarness` (cases 57 and 59, through
`LoadedEngine::FromModelDir`). Those three cases are precisely the three that
die. Their immediate neighbours are the control:

- 58/76 `embeddings without an embedder is a 500, not a crash` — **passes**, and
  builds no engine.
- 60/76 `/v1/embeddings does not exist on a TEXT server` — **passes**, same.
- 33/76 `socket smoke`, 38/76 `concurrent requests`, and every other socket case
  — **pass**, and use `ServerHarness`, which does not build a `LoadedEngine`.

So the fault is not in the api-server, the sockets, the threading, or the
byte-bound refusal that the log's last line pointed at. **In the
one-case-per-process regime**, every case that builds a `LoadedEngine` fast-fails
and every case that does not passes. The regime is part of the claim, not
scenery: the original single-process run only ever reached case 56, because the
fast-fail killed everything after it, so 57 and 59 are observations that run
could not produce. That is more information rather than a contradiction, and it
is what the instrument was built to buy. The property is measured of the
isolated regime, and a single-process run is not evidence for or against it.

**The evidence names the type, not the member.** It does not separate
construction from destruction. `__fastfail` discards the child's doctest output,
and `LogHttpIngress` at `src/vllm/entrypoints/openai/api_server.cpp:314` is the
only ingress log and fires for `/v1/chat/completions` alone, which none of the
three cases sends, so no case emits a marker. All three also run assertions
after the object exists: case 56 holds `loaded` as a stack local whose destructor
runs at the end of the case, and `EmbedHarness` holds a `shared_ptr` destroyed at
teardown. `~LoadedEngine` is therefore as consistent with `0xC0000409` as the
constructor is. What this run bounds is the `LoadedEngine` **lifetime**.

Two further readings that the enumeration closes:

- 61/76 to 63/76 are the `platform process` and `platform shutdown` cases,
  including `teardown drains an acquired console handler`. All three **pass**, so
  [#1782](https://github.com/mudler/vllm.cpp/issues/1782) is not implicated.
- The full-run log's silence after case 36 is now explained rather than
  suspicious: cases 37 to 55 issue no chat ingress that logs, so the process was
  reaching case 56 all along.

Naming the defect inside `LoadedEngine` is the next row's work and is listed
under `## Owed`. This spec bought the location, and it is not a guess.

## Owed

- **The cure for #584.** The location is now known: the `LoadedEngine` lifetime
  on Windows, with construction and destruction both still in scope, because
  nothing in this run separates them. The repair is a separate row with its own
  spec, because it is a `src/` change in code this spec deliberately does not
  touch.
- **The downstream Windows executables have still never run.**
  `test_openai_api_server` is the first the gate invokes and it still throws, so
  `test_lmcache_client`, `test_kv_offload_fs`, `test_cpu_isa_x86` and
  `test_vulkan_loader` remain unmeasured on Windows. That is pre-existing and is
  not made worse here.

## Now

`ACTIVE`. The instrument landed its purpose on its first complete Windows run:
both lanes name cases 56, 57 and 59 of 76, and the common factor is the
`LoadedEngine` lifetime. #584 stays open for the repair.
