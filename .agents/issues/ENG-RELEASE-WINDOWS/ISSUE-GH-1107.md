ID: ISSUE-GH-1107
Title: `scripts/check-windows-portability.py` exits 0 on a tree carrying an unguarded `::setenv` in a test translation unit — measured `Windows portability contract OK`, rc 0, on the unrepaired `test_expert_stream_steps.cpp`. It misses the class twice over, and either miss alone is enough. SCOPE: `check()` builds its `texts` map from `shipped_server_sources(...)`, the sources reachable from the shipped SERVER target, so no file under `tests/` is read for ANY of its rules. VOCABULARY: `POSIX_PATTERNS` names `fork|execvp|waitpid|pipe|read|write|open|close|fsync|pread|pwrite|getpid|stat` plus the `unistd.h`-family includes, and neither `setenv` nor `unsetenv` appears — so even inside the scanned set the call would pass. The second miss is the one that generalises: the three private `_putenv_s` copies [#603](https://github.com/mudler/vllm.cpp/issues/603) records, and the shim it landed, all exist because this is the recurring call, and the checker holding the Windows contract does not know its name. Invisible in CI because `windows-msvc-*` are PR-only with no `main` baseline ([#584](https://github.com/mudler/vllm.cpp/issues/584)) and are currently red inside the product library on [#1068](https://github.com/mudler/vllm.cpp/issues/1068), so a lane failing before it reaches a test TU cannot report a new test-TU failure; the static checker was the only instrument that could have. Found while repairing [#1106](https://github.com/mudler/vllm.cpp/issues/1106) finding 4 and NOT fixed there: a checker semantic change needs its own spec, red-before test and green-after evidence, and widening the scan to `tests/` has to separate a guarded POSIX call from an unguarded one across a large surface, which wants measurement rather than a guess
Row: ENG-RELEASE-WINDOWS
State: UNKNOWN
Kind: bug
GitHub: 1107
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:318`

### Frozen archive evidence

> | [#1107](https://github.com/mudler/vllm.cpp/issues/1107) | `ENG-RELEASE-WINDOWS` | `scripts/check-windows-portability.py` exits 0 on a tree carrying an unguarded `::setenv` in a test translation unit — measured `Windows portability contract OK`, rc 0, on the unrepaired `test_expert_stream_steps.cpp`. It misses the class twice over, and either miss alone is enough. SCOPE: `check()` builds its `texts` map from `shipped_server_sources(...)`, the sources reachable from the shipped SERVER target, so no file under `tests/` is read for ANY of its rules. VOCABULARY: `POSIX_PATTERNS` names `fork\|execvp\|waitpid\|pipe\|read\|write\|open\|close\|fsync\|pread\|pwrite\|getpid\|stat` plus the `unistd.h`-family includes, and neither `setenv` nor `unsetenv` appears — so even inside the scanned set the call would pass. The second miss is the one that generalises: the three private `_putenv_s` copies [#603](https://github.com/mudler/vllm.cpp/issues/603) records, and the shim it landed, all exist because this is the recurring call, and the checker holding the Windows contract does not know its name. Invisible in CI because `windows-msvc-*` are PR-only with no `main` baseline ([#584](https://github.com/mudler/vllm.cpp/issues/584)) and are currently red inside the product library on [#1068](https://github.com/mudler/vllm.cpp/issues/1068), so a lane failing before it reaches a test TU cannot report a new test-TU failure; the static checker was the only instrument that could have. Found while repairing [#1106](https://github.com/mudler/vllm.cpp/issues/1106) finding 4 and NOT fixed there: a checker semantic change needs its own spec, red-before test and green-after evidence, and widening the scan to `tests/` has to separate a guarded POSIX call from an unguarded one across a large surface, which wants measurement rather than a guess | bug |

## Resolution

-
