ID: ISSUE-GH-1139
Title: All three upstream anchors on that row (`.agents/engine-matrix.md:112`) point at unrelated code at the current parity pin `555967922`, verified by reading the pinned tree: `vllm/v1/worker/gpu/model_runner.py:504` is inside a `DraftModelSpeculator.set_attn(...)` call, `:647` is a `torch.zeros(...)` argument in a `dummy_run=True` construction, and `vllm/v1/worker/gpu_worker.py:430` is a comment about `max_split_size_mb` inside `load_model`. The startup memory profile the row describes is `GPUWorker.determine_available_memory` (`gpu_worker.py:451-495`, `memory_profiling` at `:491-494` around `profile_run()` at `:495`) and `GPUModelRunner.profile_run` (`gpu/model_runner.py:682`); `model_memory_usage` is recorded AFTER the load at `gpu/model_runner.py:315`, which is why upstream never asks whether the weights will fit and why [#1123](https://github.com/mudler/vllm.cpp/issues/1123) has no upstream counterpart to mirror. Found while repairing [#1136](https://github.com/mudler/vllm.cpp/issues/1136): `gguf_device_fit.h` and `expert-streaming.md` had both COPIED the `:504,647` pair from this row, and both are corrected there, so this row is the surviving source. Filed and not fixed in flow because the fix is one cell in `.agents/engine-matrix.md`, which PR #1119 ([#1110](https://github.com/mudler/vllm.cpp/issues/1110)) is concurrently bumping alongside the hardcoded `ENGINE` count in `scripts/check-agent-record.py` — the record-lock hazard AGENTS.md names, and the reason the repairing session was told to leave both files alone. Most likely cause: correct at the previous `e24d1b24` pin and not reconciled when the pin advanced; whether other `INVENTORIED` rows citing `vllm/v1/worker/gpu/**` share the defect is a wider sweep than one cell
Row: KV-WARMUP-PROFILE
State: UNKNOWN
Kind: bug
GitHub: 1139
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:347`

### Frozen archive evidence

> | [#1139](https://github.com/mudler/vllm.cpp/issues/1139) | `KV-WARMUP-PROFILE` | All three upstream anchors on that row (`.agents/engine-matrix.md:112`) point at unrelated code at the current parity pin `555967922`, verified by reading the pinned tree: `vllm/v1/worker/gpu/model_runner.py:504` is inside a `DraftModelSpeculator.set_attn(...)` call, `:647` is a `torch.zeros(...)` argument in a `dummy_run=True` construction, and `vllm/v1/worker/gpu_worker.py:430` is a comment about `max_split_size_mb` inside `load_model`. The startup memory profile the row describes is `GPUWorker.determine_available_memory` (`gpu_worker.py:451-495`, `memory_profiling` at `:491-494` around `profile_run()` at `:495`) and `GPUModelRunner.profile_run` (`gpu/model_runner.py:682`); `model_memory_usage` is recorded AFTER the load at `gpu/model_runner.py:315`, which is why upstream never asks whether the weights will fit and why [#1123](https://github.com/mudler/vllm.cpp/issues/1123) has no upstream counterpart to mirror. Found while repairing [#1136](https://github.com/mudler/vllm.cpp/issues/1136): `gguf_device_fit.h` and `expert-streaming.md` had both COPIED the `:504,647` pair from this row, and both are corrected there, so this row is the surviving source. Filed and not fixed in flow because the fix is one cell in `.agents/engine-matrix.md`, which PR #1119 ([#1110](https://github.com/mudler/vllm.cpp/issues/1110)) is concurrently bumping alongside the hardcoded `ENGINE` count in `scripts/check-agent-record.py` — the record-lock hazard AGENTS.md names, and the reason the repairing session was told to leave both files alone. Most likely cause: correct at the previous `e24d1b24` pin and not reconciled when the pin advanced; whether other `INVENTORIED` rows citing `vllm/v1/worker/gpu/**` share the defect is a wider sweep than one cell | bug |

## Resolution

-
