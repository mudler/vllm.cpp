ID: ISSUE-GH-1626
Title: test_mistral_paged_engine still streams its const char* label as bool (the #1508 class)
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: CLOSED
Kind: bug
GitHub: 1626
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-21
Updated: 2026-08-22
Closed: 2026-08-22

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> `RunGate` in `tests/parity/test_mistral_paged_engine.cpp:102` takes `const char* label` and streams it into doctest `MESSAGE`/`REQUIRE_MESSAGE`, so every label renders as `1`: the R5 dump run printed `1 dumped our token ids -> ...` and the anchor-drift message printed `logged: 1 anchor drift prompt[3] ...`.
>
> This is the same defect #1508 fixed in `test_qwen3_paged_engine.cpp` (doctest's `MessageBuilder` has no `const char*` overload): the fix there was taking `const std::string&`. The Mistral copy of the gate never got it.
>
> ## Fix
>
> One line: change the `label` parameter to `const std::string&` (the call site's string literal converts implicitly). Found during the #1604 golden refresh when the garbled label hid which gate had dumped.
>
> FIXED IN FLOW by the #1604 change that found it.
>
> Owner: row `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`.

## Resolution

GitHub records closing pull request #1630 (https://github.com/mudler/vllm.cpp/pull/1630) merged on 2026-08-22 as commit `333509dc5a788d167cbe34b5482fcfaa1e3c9e61`. GitHub closed issue #1626 on 2026-08-22.
