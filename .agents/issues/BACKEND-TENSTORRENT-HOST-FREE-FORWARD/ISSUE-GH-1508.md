ID: ISSUE-GH-1508
Title: doctest MessageBuilder streams const char* as bool, so test_qwen3_paged_engine prints a wrong committed-anchor value (62901 for 6290) and a useless label
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: CLOSED
Kind: bug
GitHub: 1508
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-21
Closed: 2026-08-21

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Defect
>
> Every `MESSAGE`/`REQUIRE_MESSAGE` in `tests/parity/test_qwen3_paged_engine.cpp` that streams a
> separately-bound `const char*` renders it as `1`, because the pinned doctest
> (`third_party/doctest/doctest.h`) `MessageBuilder` stream has no `const char*` overload and the
> compiler falls back to the implicit pointer-to-`bool` conversion.
>
> Two visible consequences, both reproduced with a 7-line doctest harness against the pinned header
> (`/tmp/dt_charptr.cpp`, 2026-08-20):
>
> 1. `label` (`const char* "qwen3-0.6B"`) prints as `1`, so every log line starts
>    `1: loading via FromModelDir(...)` / `1 anchor drift ...` instead of naming the model.
> 2. The anchor-drift `REQUIRE_MESSAGE` ends with a `const char*` ternary, which prints as a
>    trailing `1` **concatenated onto the last numeric field**:
>
>    ```
>    logged: 1 anchor drift prompt[1] tok=10 engine=14126 committed anchor=62901
>    ```
>
>    The committed anchor is `6290`; the message prints `62901`. Same for
>    `committed anchor=96251` where the golden holds `9625` (prompt[0] tok 5). The value a reader
>    would diff against never existed in any golden file.
>
> This misled the #1476/#1488 investigation into suspecting memory corruption of the loaded golden
> buffers; the buffers were byte-clean (npy v1, 118-byte header, flat[26]=6290 verified at the byte
> level).
>
> ## Reproducer
>
> ```cpp
> #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
> #include "doctest.h"
> TEST_CASE("charptr streaming") {
>     const char* label = "qwen3-0.6B";
>     int32_t committed = 6290;
>     MESSAGE(label << " anchor drift committed=" << committed
>             << (true ? " — re-capture via script" : " — other"));
> }
> // prints: MESSAGE: 1 anchor drift committed=62901
> ```
>
> `std::string` streams correctly (the `snap` path in the same messages prints fine), so the fix is
> to pass `label` as `std::string` and wrap the trailing ternary in `std::string(...)`.
>
> ## Blast radius
>
> Any test that streams a separately-bound `const char*` (variable or ternary) through
> `MESSAGE`/`REQUIRE_MESSAGE`/`CHECK_MESSAGE`. Known instance: `test_qwen3_paged_engine.cpp`
> (`label` at :216/:255/:279/etc., trailing ternary at :341). Found during the #1488 golden
> re-adjudication.
>
> ## Ownership
>
> Filed and fixed in the same flow as the #1488 TT golden refresh (row
> `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`); the fix rides that pull request as its own commit.

## Resolution

GitHub records closing pull request #1514 (https://github.com/mudler/vllm.cpp/pull/1514) merged on 2026-08-21 as commit `49c64bbc819fb8240d00253ef91e06232f9955e1`. GitHub closed issue #1508 on 2026-08-21.
