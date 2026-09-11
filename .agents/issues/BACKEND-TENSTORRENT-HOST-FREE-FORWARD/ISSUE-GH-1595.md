ID: ISSUE-GH-1595
Title: test_qwen3_dflash2_gguf.cpp does not compile under clang-20: ::getpid() used without <unistd.h>
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: CLOSED
Kind: bug
GitHub: 1595
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-21
Updated: 2026-08-21
Closed: 2026-08-21

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Defect
>
> `tests/vllm/models/test_qwen3_dflash2_gguf.cpp:547` calls `::getpid()` but the file never
> includes `<unistd.h>`:
>
> ```
> tests/vllm/models/test_qwen3_dflash2_gguf.cpp:547:53: error: no member named 'getpid' in the global namespace; did you mean 'getpt'?
> ```
>
> The include block (`:51-72`) has the C++ headers, `nlohmann/json.hpp`, `gguf_builder.h` and the
> model headers — no POSIX header, so glibc never leaks `getpid` in under clang-20 + this glibc.
> The file landed on `main` in the SPEC-DFLASH2 GGUF drafter arm (`5702d8f83` /
> `141402e6c`, PRs #1531/#1537 area) and fails to compile as soon as any branch merges current
> `main` (verified byte-identical to `origin/main` `e2a9e035d` in the #1498 worktree; the merge
> itself is clean, the TU is main's).
>
> Sibling tests use a plain `#include <unistd.h>` (`test_kimi_linear_paged.cpp:44`,
> `test_loader_unaligned_offsets.cpp:46`, `test_ltx2_loader.cpp:26`,
> `test_moe_async_device_ids.cpp:80`) — no MSVC guard, this suite is POSIX-only.
>
> ## Impact
>
> `ninja` in a TT-configured worktree (clang-20 toolchain on the Blackhole P150 box) stops at
> `tests/CMakeFiles/test_qwen3_dflash2_gguf.dir`, which blocks rebuilding ANY target after a
> merge of `main` — including the `test_qwen3_paged_engine` / `test_tenstorrent_backend` gates
> the Tenstorrent rows need. Any main-merging branch on a clang host is affected.
>
> ## Ownership
>
> Found while merging `main` into `row/BACKEND-TENSTORRENT-HOST-FREE-F1476` (#1498); fixed
> in-flow on that branch with the one-line missing include.

## Resolution

GitHub records closing pull request #1498 (https://github.com/mudler/vllm.cpp/pull/1498) merged on 2026-08-21 as commit `d27639e71150557a2d9aafaf9c55108c621ac419`. GitHub closed issue #1595 on 2026-08-21.
