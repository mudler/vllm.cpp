# Qwen3.8 public documentation refresh

## Scope

Correct public documentation against the code and committed evidence at base
`6db4bef90`. This is an editorial audit, not a feature lifecycle transition.
No model, kernel, test, or benchmark implementation changes are in scope.

## Inventory and design

| Surface | Gap | Required correction | Authority |
|---|---|---|---|
| README News | September EXL3 comparison is absent | Add a compact entry linking the measurement, with its incomplete and ungated limits | `docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md` |
| Benchmark index | Mixed-length comparison reads only Measured | Expose incomplete comparator repetitions in its disposition | Same benchmark detail and committed run records |
| Qwen3.8 model guide | CUDA component paragraph denies a token gate documented earlier | Distinguish component validation from the later model comparison; simplify affected prose | `docs/models/qwen3-8-27b.md`, matching model and FP8 CUDA implementation, recorded gate evidence |

## Verification

Inspect the actual code for every changed behavior claim. Retain the recorded
hardware, oracle revision, workload, and limitations. Do not extrapolate a
sampled EXL3 serving comparison into correctness or speed parity.

Run the existing README, benchmark-index, site, and agent-record checks, plus
staged preflight. No GPU, model download, external compute, or new benchmark
is needed or authorized. Runtime performance and correctness are unchanged.
A fresh agent implements these edits from this committed spec. A fresh reviewer
checks the immutable result against the code and evidence. The operator reruns
the focused gates before pushing to the fork and opening an upstream PR.

## Risks and stop conditions

Do not edit unrelated benchmark history or feature states. Do not replace a
historical measurement with an unmeasured current result. Report any unavailable
gate dependency explicitly. No merge into upstream is authorized.

## Owed

- ISSUE-LOCAL-01M221WAVSK2STJ49Y3WTDSDS3: correct the three public surfaces above.
