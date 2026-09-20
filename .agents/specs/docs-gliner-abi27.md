# Document GLiNER2.5 and ABI 27

Work ID: `DOCS-PUBLIC-REFRESH-20260920`.
Base: `5058268d7`.

## Now

Documentation audit only. No engine, model, or benchmark lifecycle changes.
The README news omits the newly merged GLiNER2.5 surface. The README and C API
reference still call ABI 26 current, although the public header declares 27.

## Scope and design

| Surface | Source | Correction |
|---|---|---|
| README news and library overview | `include/vllm.h:354`, `src/vllm/entrypoints/openai/server_main.cpp:1134` | Announce entity extraction and report ABI 27 |
| C API reference | `include/vllm.h:1126`, `src/capi/vllm_c.cpp:1557` | Describe the blocking NER call, ownership, and non-GLiNER refusal |
| Usage introduction | `docs/USAGE.md:342`, `src/vllm/entrypoints/openai/api_server.cpp:1797` | Simplify the new feature introduction and preserve exact semantics |

Use the existing usage recipe and feature evidence. Link to them rather than
copying their detailed implementation record. Exclude runtime code, new tests,
new measurements, architecture counts, and unrelated editorial rewrites.

## Upstream and dependencies

This change documents local public interfaces. It ports no upstream behavior.
No oracle run or GPU is applicable. Source inspection verifies the names and
semantics. Existing GLiNER evidence defines the limits of any public claim.

## Tests and gates

Run the existing README structure, supported-model, surface-coverage, and
benchmark-index checks. Run their existing mutation suites and git diff checks.
Run preflight and record unavailable tools or baseline failures separately.
A fresh implementer edits the prose from this committed spec. A fresh reviewer
checks the immutable diff against source. The operator repeats focused gates.
No new test is needed for this reversible prose correction.

## Risks and stop conditions

Do not infer GPU readiness, output parity, or speed from a registered endpoint.
Do not invent checkpoint revisions or measurements. Keep unknowns explicit.
Missing source evidence requires narrower wording. No merge is authorized.

## Work breakdown

1. Commit the issue and spec before public edits.
2. Update README, the C API reference, and the GLiNER usage introduction.
3. Review against code, run CPU-only checks, and open a fork pull request.

## Owed

- [ISSUE-LOCAL-01M2YC7SRCE7HTHA62R2DSMDJ7](../issues/_owed/ISSUE-LOCAL-01M2YC7SRCE7HTHA62R2DSMDJ7.md): correct the public GLiNER and ABI overview.

## Implementation evidence

The documentation patch starts from committed spec `a9d6a142b`. It changes
README news and the current ABI number, adds the ABI 27 reference entry, and
shortens the GLiNER introduction. Historical ABI 26 news stays unchanged.

Source checks:

- `include/vllm.h:361` declares ABI 27.
- `include/vllm.h:1126` declares the NER types and blocking call.
- `src/capi/vllm_c.cpp:1559` implements validation, architecture refusal,
  result allocation, and error reporting.
- `src/capi/vllm_c.cpp:1669` frees entity strings and the array, then zeroes
  the caller's result.
- `src/vllm/entrypoints/openai/server_main.cpp:1132` selects the GLiNER server.
- `src/vllm/entrypoints/openai/api_server.cpp:1797` registers `POST /v1/ner`.
- `src/vllm/model_executor/models/gliner2_registry.cpp:168` runs the shared
  NER pipeline with the host encoder and boundary head.
- `docs/FEATURES.md:200` records the existing CPU validation and device gap.

The scoped gate runs with these environment settings:

```sh
export PATH=/tmp/vllm-doc-tools/root/usr/bin:$PATH
export LD_LIBRARY_PATH=/tmp/vllm-doc-tools/root/usr/lib
python3 scripts/check-readme-structure.py
python3 scripts/check-supported-models.py
python3 scripts/check-surface-coverage.py
python3 scripts/check-benchmark-index.py
python3 -m unittest discover -s tests/scripts -p test_check_readme_structure.py
python3 -m unittest discover -s tests/scripts -p test_check_supported_models.py
python3 -m unittest discover -s tests/scripts -p test_check_surface_coverage.py
python3 -m unittest discover -s tests/scripts -p test_check_benchmark_index.py
git diff --check
```

All commands exited 0. The existing suites passed 19, 12, 46, and 6 tests,
respectively. No new tests or runtime behavior require a new negative mutation.
The fresh review checks the documentation guarantees against source.

The initial `bash scripts/agent-preflight.sh` reached baseline failures in
`check-agent-record`, `check-release-workflow`, `check-test-registration`, and
`test_agent_record`. The stale-anchor count was 33 against a baseline of 28.
The operator stopped this duplicate run during mutation suites with exit 130.
Its log is `/tmp/vllm-docs-impl-preflight-before.log`. This incomplete run is
not a passing gate. The operator owns the complete baseline preflight report.
No GPU, oracle run, or benchmark is applicable to this prose-only correction.
