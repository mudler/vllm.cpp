# Classify recorded benchmark client logs and their patch

Row: `GATE-PR-SIZE-BENCH-EVIDENCE`
Issue: [#3060](https://github.com/mudler/vllm.cpp/issues/3060)
State: `ACTIVE`
Base: `2add10f31d28381a10c206596f62cd1878bf7205`

This governance row lives in this spec. It adds no product capability or matrix row.

## Scope

Classify the 16 inspected benchmark artifacts listed below as `evidence`.
Change `scripts/check-pr-size.py` and its existing test module.
Preserve every previously classified path and every other checker rule.

The temporary-storage repair belongs to [#3058](https://github.com/mudler/vllm.cpp/issues/3058), row `GATE-PR-SIZE-TEMPDIR`.
This row changes no temporary-storage code, budget, continuous integration exemption, oracle, product, model, provider, or runtime behavior.

## Baseline and source anchors

At the pinned base, `BENCH_EVIDENCE_RUN` omits `.clientlog` and `.patch`.
`classify_path` rejects the 16 files, and `PathClassification.test_every_tracked_and_current_change_path_is_classified` fails.
The unchanged complete module runs 56 tests and fails that one test.

The base tracks 6,213 paths.
The predecessor proof compares 6,214 paths because it also includes the new #3058 spec.
Both populations contain the same 16 refused artifacts.
The current checker gives every common classified path the same result.

The original patch entered history in `f0c24345d`.
Commit `3d62da761` records the completed head-to-head logs.
Commit `5c04f3c6d` records the variadic method, and `a8889f255` records its client output.
The existing benchmark evidence and its measurement claims remain unchanged.

The head-to-head [README](../../docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/README.md) identifies the client output and exact submitted recipe.
Its `job-as-run.sh:306` writes each `.clientlog` through `tee`.
The variadic [README](../../docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/README.md) records the same output role.
Its `job-as-run.sh:406` writes leg logs, and line 438 writes the probe log.
`THEIRS-r2-c8.clientlog` contains two startup lines without a `CLIENT_RESULT`.
An incomplete observation still belongs to evidence. Classification does not validate a measurement.

The 72-line `serve_openai-usage.patch` changes only the external `tools/serve_openai.py` server wrapper.
It records the benchmark adaptation that forwards token counters into the streaming response.
The archived head-to-head recipe applies it at `job-as-run.sh:210`.
The live `benchmarks/variadic/job.sh:240` also applies its staged `$SHARED` copy.
Therefore, absence of consumers outside `docs/` is not a premise for this repair.
This task classifies the recorded patch. It does not revalidate the benchmark adaptation or its oracle.

## Design

Add an explicit set of recorded evidence paths and consult it in `classify_path`'s existing evidence branch.
Keep `BENCH_EVIDENCE_RUN`, its suffix list, canonical-path validation, and classification order unchanged.
Do not add a general `.clientlog` or `.patch` suffix rule.

The set contains only these files under `docs/bench-evidence/`:

| Run directory | Files |
|---|---|
| `qwen38-27b-exl3-headtohead-20260903` | `OURS-A.clientlog`, `OURS-B.clientlog`, `THEIRS-A.clientlog`, `THEIRS-B.clientlog`, `serve_openai-usage.patch` |
| `qwen38-27b-exl3-variadic-20260905` | `OURS-r1-c1.clientlog`, `OURS-r1-c4.clientlog`, `OURS-r1-c8.clientlog`, `OURS-r2-c1.clientlog`, `OURS-r2-c4.clientlog`, `OURS-r2-c8.clientlog`, `PROBE.clientlog`, `THEIRS-r1-c1.clientlog`, `THEIRS-r1-c4.clientlog`, `THEIRS-r1-c8.clientlog`, `THEIRS-r2-c8.clientlog` |

Exact paths express the inspected surface without assigning an evidence class to uninspected patches or future logs.
A new sibling artifact requires its own justified classification.
The `evidence` class still requires reviewed pull-request arrival through the existing `requires_reviewed_pr` rule.

## Tests and upstream applicability

There is no vLLM implementation or upstream test suite for this repository governance classifier.
Extend `tests/scripts/test_check_pr_size.py` without changing existing assertions.
Keep expected paths independent of the new classifier set.

Require the evidence class for every listed artifact, including the incomplete log and the recorded patch.
Reject unknown siblings, another run, nested or flat placements, and noncanonical near misses.
Preserve existing document, recipe, mutable source, checker, and test classifications.
Retain the whole-tree classification sweep.

Delete the new production classification call site, remove individual admitted paths, return a wrong class, and broaden admission in owned copies.
Each focused detector must fail for the intended reason.
Removing only its assertion must recover green, and restoring the detector must recover the intended failure.
Restore source bytes after every mutation and preserve all injected versions and receipts.

## Gates and evidence

Run complete initial preflight before tracked edits and staged preflight before commits.
Use isolated Git configuration and discovery, owned home-cache temporary storage, existing dependency shims, and compiler jobs 4.
Preserve every explicit skip. A zero preflight exit is not blanket green.

Run the unchanged smallest failing test and the unchanged complete checker module before implementation.
Run focused regressions, the complete module, and the effective mutation pairs after implementation.
Compare every base path classification before and after. Only the listed 16 refusals may become `evidence`.
Verify that all other tracked source and artifact bytes remain unchanged.

After committing the implementation, run the checker's complete semantic evidence gate over the exact base and candidate commits.
Use the unchanged base checker as the driver to avoid a candidate certifying itself.
Require the complete candidate module to pass and the same module to fail against the base checker.
Retain both test counts, full output, exact commands, environments, and source hashes.
Run record, commit-style, and trailer checks over the exact candidate range.

On this host, the base checker and three tests explicitly request `/dev/shm`.
Root authorizes a private adapter that redirects only those requests into owned scratch and retains default allocation under owned `TMPDIR`.
The adapter preserves original and effective commands and environments, and changes no tracked source or verdict logic.
This accommodation does not implement or accept #3058.

Evidence lives under the operator's `bench-evidence-classification-3060/implementer` bundle in `current-main-validation`.
The bundle retains the original #3058 failure proof, independent reproductions, source audit, classification joins, and full mutation receipts.
Archive all remaining owned scratch with a complete inventory and full stream verification before cleanup.

## Git integration and work breakdown

Use one pull request under the repository and session default.
Commit this spec before tests and implementation.
Add and run the failing regressions, make the minimum classifier change, and run the declared gates.
A fresh reviewer inspects the immutable candidate and mutates its guarantees.
The operator independently reruns the reviewed gate and owns publication.
The owning pull request closes #3060 when the repair lands.

## Now

`ACTIVE`. The exact artifact classification repair is implemented.
Fresh review and operator verification remain required before landing.

## Risks and stop conditions

A suffix-wide rule would admit uninspected source patches. Exact-path admission avoids that expansion.
Do not confuse an evidence class with a successful benchmark or an approved oracle adaptation.
Stop if another classification must change, a required input binding is missing, or a gate needs unowned scratch.
Do not waive the baseline failure or weaken checker assertions to obtain green.
