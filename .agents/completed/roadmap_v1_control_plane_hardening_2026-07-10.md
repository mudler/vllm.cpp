# Roadmap v1 control-plane hardening - 2026-07-10

This completed block audited and repaired the first tabular-roadmap groundwork.
It changed project coordination and CI policy, not runtime feature support.

## Closed scope

| Finding | Resolution |
|---|---|
| Count-only CI admitted malformed lifecycle rows | lifecycle-aware parser now validates semantic fields, evidence, specs, claims, `DONE` ledger/commit closure, portfolio order, counts, links and table shape |
| Checker behavior itself was untested | 13-test mutation suite proves prose-only/empty specs and missing fields/spec identity/claims/evidence/closure are rejected |
| Priority-zero A1/A5 labels were not claimable | stable `SERVE-GATE-ONLINE` and `SERVE-E2E-NIGHTLY` rows; online gate has a complete spike |
| `READY` TP/MTP/DFlash rows did not name IDs or satisfy the spike map | specs now bind exact rows and map scope, chain, baseline, port, tests, gates, dependencies, work and risk; DFlash has non-overlapping claim leaves |
| Model target rows lived in a scoping document | all 323 stable `MODEL-*` claim rows now live in `model-matrix.md`; the spec retains only inventory methodology |
| GGUF compute/presets lacked claim rows; KV/MLX schemas and legacy `DONE` closure were incomplete | added compute/preset umbrellas, normalized owner/upstream/code/evidence/spec fields, and grounded three completed gate slices with ledger plus closing commits |
| Zen CPU and MLX-LM reference surfaces were missing | added `BACKEND-CPU-ZEN` and `BACKEND-GATE-METAL-MLXLM` |
| SGLang comparison was only an umbrella idea | accepted the pinned v0.5.13 leaf; split claimable harness/token-ID/checkpoint preflight from the binding gate, which remains blocked on async streaming and exact equivalence |
| AOT byte reproducibility was overclaimed | narrowed the archived A4 text to source/manifest hash tracking; cross-worktree path normalization stays with `CLAIM-PR3` |

## Frozen inventory

| Area | Stable claim rows |
|---|---:|
| Engine/serving | 83 |
| Model | 323 target/factory/dynamic rows; 353 unique static architecture IDs |
| Quantization | 78 |
| Kernel families | 30 |
| Backends/targets/competitor gates | 51 |

The ordered `ROAD-V1-A`, C1-C9, D1-D4 sequence did not change.

## Validation

- `python3 scripts/check-agent-record.py`
- `python3 tests/scripts/test_agent_record.py`
- `git diff --check`
- full CPU configure/build/ctest recorded in the state/ledger; post-push GitHub
  CI is required before this block's commit is treated as integrated

## Remaining work

Permanent support rows remain live in their owning matrices. The next claims are
the priority-zero online gate diagnosis/rerun, nightly spike, PR #3 closure,
kernel ABI spike, model factory spike, GGUF compute leaf split, and backend
architecture spine. This archive is evidence only and is not load-bearing for
those decisions.
