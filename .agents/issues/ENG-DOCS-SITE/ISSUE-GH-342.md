ID: ISSUE-GH-342
Title: Public docs drift from shipped CLI, ABI v17, endpoints, registry counts, and benchmark scope
Row: ENG-DOCS-SITE
State: OPEN
Kind: UNKNOWN
GitHub: 342
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-11
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Problem
>
> The user-facing README and usage guide disagree with current source and current keyed public records:
>
> - `README.md` invokes `build/examples/server`, but the CMake target outputs `build/examples/vllm-server`.
> - README and `docs/USAGE.md` describe ABI v10 with 19 exports, while `include/vllm.h` declares ABI v17 and the current public surface.
> - `docs/USAGE.md` omits the conditional `/v1/embeddings` and `/v1/audio/transcriptions` routes implemented in `api_server.cpp`.
> - README says 25+, 28, and 30 architectures in different places, while the registry-bound `docs/FEATURES.md` checker proves 35 registered and 27 gated.
> - README says every architecture is token-gated, although registered scaffold/oracle-blocked rows are explicitly documented.
> - README's global throughput language conflates the passing `unsloth@890bdef7` 27B grid with the current NVIDIA ModelOpt 27B and 35B grids that remain speed-pending.
>
> ## Scope
>
> Create one source-grounded documentation repair:
>
> 1. Correct the quickstart/server executable path.
> 2. Update ABI wording and the ABI evolution table through v17 from `include/vllm.h`.
> 3. Add the shipped conditional embedding and transcription endpoints to `docs/USAGE.md`.
> 4. Make README registry counts and gate wording agree with `docs/FEATURES.md`.
> 5. Scope throughput claims to the exact checkpoint and keep current ModelOpt/35B gaps honest.
> 6. Apply a human-language pass without adding new claims or benchmark numbers.
>
> Do not compact `docs/STATUS.md` or `docs/BENCHMARKS.md` in this issue. Do not change code, checker semantics, lifecycle state, or accepted benchmark values.
>
> ## CPU-only acceptance
>
> - `python3 scripts/check-readme-structure.py`
> - `python3 scripts/check-public-doc-tables.py`
> - `python3 scripts/check-supported-models.py`
> - `python3 scripts/check-surface-coverage.py`
> - Configure/build the CPU examples and prove `build/examples/vllm-server --help`.
> - Stale-string scan finds no old ABI/count/path claims.
>

## Resolution

-
