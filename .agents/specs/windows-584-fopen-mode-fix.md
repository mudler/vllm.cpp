# Spec: Windows MSVC 0xC0000409 — fix glibc-only fopen mode in host memory probe

## Row

`WINDOWS-584`

## Issue

#1495 (re-opened; was closed `not_planned` without a fix). Also closes the
`test_openai_api_server` crash tracked by #2403.

## Problem

`vllm::v1::host_available_memory_bytes()` at
`src/vllm/v1/core/kv_cache_utils.cpp:954` opens `/proc/meminfo` with mode
`"re"`. The `e` suffix (O_CLOEXEC) is a glibc extension. MSVC's Universal
CRT does not recognise it as a valid `fopen` mode character and invokes its
invalid-parameter handler, which calls `__fastfail`. The process terminates
with exception code `0xC0000409` (STATUS_STACK_BUFFER_OVERRUN). This
bypasses all C++ and SEH exception handlers, so doctest's Windows crash
handler never fires and no test case name is printed.

The call is reached from the `LoadedEngine` constructor body at
`src/vllm/entrypoints/model_loader.cpp:2308`, unconditionally — before the
`state_needed > 0 && host_available > 0` guard at line 2309. All three
crashing test cases (56, 57, 59) construct `LoadedEngine` directly, so all
hit this path. `ServerHarness` (used by the passing cases) builds
`KVCacheConfig` in the test and never calls `host_available_memory_bytes()`.

This explains why the crash is Windows-only: on Linux, `"re"` is valid and
`/proc/meminfo` exists. On Windows, `"re"` is invalid and crashes before
the `nullptr` return check at line 955 can run.

## Scope

Change `std::fopen("/proc/meminfo", "re")` to `std::fopen("/proc/meminfo", "r")`
in `host_available_memory_bytes()`. The function already handles `nullptr`
return (returns 0, the documented "unknown" value). On Windows,
`/proc/meminfo` does not exist, so `fopen("...","r")` returns `nullptr` and
the function returns 0 — the safe fallback. On Linux, `fopen("...","r")`
works identically; the lost `O_CLOEXEC` is irrelevant for a transient read
that is opened, consumed, and closed within the same function call with no
`fork`/`exec` in between.

Add a portability checker rule to `scripts/check-windows-portability.py` that
rejects `fopen` calls using the glibc-only `"e"` mode suffix, so this class
of bug cannot re-enter.

## Design

### Product code change

`src/vllm/v1/core/kv_cache_utils.cpp:954`:

```
-  std::FILE* f = std::fopen("/proc/meminfo", "re");
+  std::FILE* f = std::fopen("/proc/meminfo", "r");
```

One line. The comment above (lines 944-953) already documents the return-0
fallback for unreadable `/proc/meminfo`, which is exactly the Windows path.

### Checker change

Add a regex to `POSIX_PATTERNS` in `scripts/check-windows-portability.py`
that flags `fopen` calls whose mode string contains the `e` character.

## Risks

- **O_CLOEXEC loss**: irrelevant. The file handle is open for microseconds,
  read, and closed. No `fork`/`exec` can interleave.
- **False positives from the checker**: the `e` suffix is only valid on glibc
  and only useful with `fork`/`exec`. Any `fopen(..., "...e")` in the
  codebase is a Windows portability bug. The pattern is narrow enough.

## Tests

- The existing Windows MSVC CI lane (`windows-msvc-cpu`) is the test: the
  three crashing cases (56, 57, 59) must pass. The case localiser
  (PR #2124) already isolates them.
- A unit test for the portability checker that asserts it rejects a
  `fopen(..., "re")` call and accepts `fopen(..., "r")`.
- Linux `build-test-cpu` must still pass (no regression).

## Gates

- `build-test-cpu` (Linux): green (no regression).
- `windows-msvc-cpu`: the three previously-crashing cases pass.
- `check-windows-portability`: the new rule fires on a mutation that
  reintroduces `fopen(..., "re")`.

## Evidence

- Root cause: `src/vllm/v1/core/kv_cache_utils.cpp:954` — `fopen("re")`.
- Call site: `src/vllm/entrypoints/model_loader.cpp:2308`.
- Issue #1495 body describes the exact mechanism.
- Prior investigation disproved the destructor-throw hypothesis: the crash
  is a UCRT invalid-parameter fast-fail, not a C++ exception.

## Stop conditions

- If changing `"re"` to `"r"` does not fix the Windows crash, the root cause
  is elsewhere. Re-open the investigation.
- If the checker rule produces false positives on legitimate code, narrow
  the pattern.

## Git integration

One pull request (spec + fix + checker + tests).
