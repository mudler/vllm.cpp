ID: ISSUE-GH-1486
Title: test_tenstorrent_backend SIGSEGVs after a fully green doctest summary: static ttnn::Tensor caches destroyed after the device closes
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: CLOSED
Kind: bug
GitHub: 1486
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-28
Closed: 2026-08-28

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Measured on thalia (Blackhole P150 host, aarch64, TT Release build with `CMAKE_PREFIX_PATH` pointing at the tt-metal `build_Release` cmake dirs), at `206afb63` == `origin/main` with **zero local commits**:
>
> ```
> [doctest] test cases:  23 |  23 passed | 0 failed | 0 skipped
> [doctest] assertions: 831 | 831 passed | 0 failed |
> [thalia:...] Failing at address: ...
> [1] libtt_metal.so tt::tt_metal::GraphTracker::is_enabled()
> [2] _ttnncpp.so ttnn::Tensor::deallocate_impl(bool)
> [3] _ttnncpp.so ttnn::Tensor::~Tensor()
> [4-8] std::optional<ttnn::Tensor> destruction chain
> ...exit 139
> ```
>
> The crash is **post-summary static destruction**: `std::optional<ttnn::Tensor>` fields of process-lifetime cache entries (`src/vt/tenstorrent/tenstorrent_ops.cpp:211,396,707,1020,1206` — weight/activation caches) are destroyed after the UMD device drivers have already printed their close lines, and `deallocate_impl` reaches `GraphTracker::is_enabled()` on a torn-down tracker.
>
> **Proven pre-existing by A/B in one build directory** (only `src/vt/tenstorrent/tenstorrent_ops.cpp`, `tenstorrent_device.h`, `src/vllm/model_executor/models/qwen3.cpp` differing — the #1476 fix, which adds no `optional<Tensor>`): stashed (pre-fix) build → 23/23 + exit 139; fixed build → 23/23 + exit 139. Identical signature both arms. Deterministic per run (every run of the suite on this host ends 139).
>
> Impact: `ctest` and any script keying on exit code report the suite FAILED while every test passed; it is also the same signature as the teardown segfault noted in #1476 on `test_qwen3_paged_engine`.
>
> Likely repair direction: clear the TT static caches (or detach their device buffers) before device teardown — e.g. an explicit drain hook called from the device close path — rather than relying on exit-time destruction order.
>
> Found while gating #1476. Owning row `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`, listed under `## Owed` in its spec.

## Resolution

GitHub records closing pull request #2118 (https://github.com/mudler/vllm.cpp/pull/2118) merged on 2026-08-28 as commit `d7d89fd57d43055050ea8a2538b10222134ac0cf`. GitHub closed issue #1486 on 2026-08-28.
