# The Windows api-server gate becomes able to report its own failure

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#584](https://github.com/mudler/vllm.cpp/issues/584)

Parent specification: [windows-baseline-coverage.md](windows-baseline-coverage.md),
which narrowed #584 and listed it under `## Owed`. This spec takes the half that
document names as provable by inspection.

Status: `ACTIVE`. Base `affc2a7fdfaa1a75c6c2b8bacd2e79b2990446f7`.

## Scope

`tests/vllm/entrypoints/openai/test_api_server.cpp` holds joinable
`std::thread` objects across assertions that throw. Convert every such site to a
scoped joiner, so that a failing assertion in that file is reported by doctest
instead of killing the process through `std::terminate`.

In: that one test file. In: an exception barrier on each thread body, because an
exception escaping a thread function is the same `std::terminate` from the other
direction. Out: `src/`, `include/`, the CI workflow, the release script, and any
other test file — the same shape exists elsewhere and is recorded under `## Owed`
rather than fixed here.

This is a **reporting** repair. It is not claimed as the cure for #584, and the
issue stays open when it lands. See `## What this does and does not establish`.

## Platform anchors

vLLM has no Windows lane and no equivalent test, so the anchors are language and
platform contracts rather than an upstream port:

- `[thread.thread.destr]`: `~thread` calls `std::terminate` if the thread is
  joinable. A `std::thread` member destroyed during stack unwinding therefore
  ends the process.
- `[except.handle]/9`: an exception escaping the initial function of a thread
  calls `std::terminate`.
- MSVC implements `abort()` — which `std::terminate` reaches through the default
  handler — as `__fastfail(FAST_FAIL_FATAL_APP_EXIT)`. `__fastfail` raises
  status `0xC0000409` for every fail-fast code and bypasses SEH by design, so
  doctest's Windows handler never runs and its buffered `stdout` is discarded
  unflushed. That is why the whole job output is the doctest version banner.
- `httplib::Server::stop()` (`third_party/httplib/httplib.h:11460`) is a no-op
  while `is_running_` is false, and `listen_internal` sets that flag only after
  it enters the accept loop (`:12027`). A joiner that does not account for this
  can block forever on a server that has not started, which is why the scoped
  server thread waits before it stops.

## Design

Two small types at the top of the test file, above the first `TEST_CASE`.

`ScopedThread` owns one `std::thread`, runs the body inside `try`/`catch`, and
joins in its destructor. The caught exception is stored in an
`std::exception_ptr` and rethrown by the explicit `join()`, which is a
synchronisation point, so the store and the load do not race. The destructor
never rethrows, because a destructor that throws during unwinding is the failure
it exists to prevent. An optional stop action runs before the join so a body
that waits on something can be released.

`ScopedServerThread` is `ScopedThread` for the case that dominates this file: it
serves an `ApiServer` and owns the `stop()` as well as the join. Its stop action
first waits, bounded, for `is_running()`, because `httplib::Server::stop()` does
nothing before the accept loop is up and a naive joiner would convert an
`0xC0000409` into a 180-minute CI timeout — a worse instrument, not a better one.
The bound is the same `500 x 2 ms` the call sites already used to wait for the
server, so a server that never starts costs one second and then joins.

Owning the stop means the explicit `h.server.stop()` lines are removed at each
converted site. Calling `stop()` twice is not equivalent to calling it once: the
second call sees `is_running_` still true while the accept loop unwinds and
`svr_sock_` already exchanged to `INVALID_SOCKET`, which trips
`assert(svr_sock_ != INVALID_SOCKET)` at `httplib.h:11462` on every build that
is not `NDEBUG` — which is the Linux test build. One owner is the only shape
that is correct on both platforms.

## Risks

- **A joiner that hangs is worse than a fast-fail.** Handled above by the
  bounded wait, and by keeping the stop action explicit for the two Windows
  console-handler threads that block on an event.
- **The conversion is mechanical across many sites.** A missed `stop()` removal
  would hang the Linux run rather than pass quietly, so the failure mode of a
  mistake here is loud.
- **The Windows crash may not be a joinable-thread terminate.** Then the lane
  stays red — but red with a name, which is the point.

## Tests and evidence

The instrument cannot assert its own effect on the platform where the effect
matters, because no Windows host is available to this row. What is gated:

1. The file compiles and the suite runs green on Linux, with the doctest case
   count unchanged before and after — the conversion adds no case and removes
   none.
2. A red-first mutation, executed and recorded in `## Outcome`: make one
   assertion inside a scoped-server case fail on purpose, and record what the
   run prints. Before the change the expectation is process death; after it, a
   named doctest failure with a `Status:` line.

The doctest case-count assertion is explicit rather than "it passed", and the
count is asserted non-zero, because a `-tc` filter that matches nothing prints
`SUCCESS!`.

## Measured

Host `mudler-desktop`, Linux, GCC, `cmake -S . -B build-584
-DVLLM_CPP_BUILD_TESTS=ON` — the CI `build-test-cpu` configuration, so `NDEBUG`
is NOT defined and `httplib`'s `assert` is live. Same build directory for every
arm below, target `test_openai_api_server`, `-j 4`. Base
`affc2a7fdfaa1a75c6c2b8bacd2e79b2990446f7`. Zero compiler warnings.

**Case count, unchanged and non-zero.** Both arms report the same totals, so the
conversion added no case and removed none:

| arm | run |
|---|---|
| before | `test cases: 62 \| 62 passed \| 0 failed \| 0 skipped`, `assertions: 733 \| 733 passed \| 0 failed`, `Status: SUCCESS!` |
| after | `test cases: 62 \| 62 passed \| 0 failed \| 0 skipped`, `assertions: 733 \| 733 passed \| 0 failed`, `Status: SUCCESS!` |

**The mutation.** One assertion inside the socket-smoke case is made to fail
while the server thread is still joinable — `CHECK(res->status == 200)` on
`/health` becomes `REQUIRE(res->status == 999)`. `git diff --stat` confirmed one
changed line in each arm, and each arm compiled with rc 0, so neither reading is
a build failure wearing a pass.

| arm | exit | what the run printed |
|---|---|---|
| before | **134** (`SIGABRT`) | `terminate called without an active exception`, then `test case CRASHED: SIGABRT` |
| after | **1** | the named failure and nothing else: `FATAL ERROR: REQUIRE( res->status == 999 ) is NOT correct! values: REQUIRE( 200 == 999 )` |

`terminate called without an active exception` names the mechanism exactly: it
is `~thread` on a joinable thread, not an escaped exception. The `after` arm
exits 1 through the ordinary failure path with no abort at all.

**Why this has been invisible on Linux.** The `before` arm still printed its
assertion, because `SIGABRT` is catchable and doctest's POSIX handler reports it
and flushes. MSVC's `__fastfail` is not catchable and bypasses SEH, so the same
`std::terminate` prints nothing there. The defect is platform-independent; only
its reportability is not, which is why a decade of green Linux runs is not
evidence against it.

## Gates

- `scripts/agent-preflight.sh`
- `cmake --build <dir> --target test_openai_api_server` and the binary's own run
- `python3 scripts/check-commit-style.py`, `check-commit-trailers.py`,
  `check-agent-record.py`, `check-issue-index-append-only.py`, `check-pr-size.py`

## Stop conditions

Stop and report if the build cannot complete in the free disk available; an
`ENOSPC` here makes unrelated checkers emit refusals that read as verdicts about
this diff. Stop rather than widen: converting the same shape in other test files
belongs to its own change.

## What this does and does not establish

Establishes: an assertion failure anywhere in this file is reported. Every
converted thread is joined on every path.

Does not establish: that #584's fast-fail was a joinable-thread terminate. A
`/GS` cookie failure and a CRT invalid-parameter call raise the identical status
and are not excluded by anything in the log. The issue stays open, and this
change is what makes the next Windows run able to answer the question.

## Owed

- The same shape exists outside this file and is not converted here:
  `tests/vllm/entrypoints/openai/test_conformance.cpp:418`,
  `tests/vllm/v1/kv_offload/lmcache/test_lmcache_client.cpp:140,326` and
  `tests/vllm/v1/kv_offload/lmcache/test_lmcache_connector.cpp:106` each join
  under an `if (joinable)` in a destructor or teardown, which is the safe half,
  but no other file was audited for a bare `std::thread` held across an
  assertion. `test_lmcache_client` is one of the four executables the Windows
  gate runs after this one, so it is next in line to be reached at all. Tracked
  by [#584](https://github.com/mudler/vllm.cpp/issues/584) until this file's
  repair lets the lane say what fails next.
