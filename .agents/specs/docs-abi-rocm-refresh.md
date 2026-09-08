# Refresh the ABI and ROCm user instructions

Work ID: `ENG-DOCS-ABI-ROCM-REFRESH`.
Issue: [#3055](https://github.com/mudler/vllm.cpp/issues/3055).
Base: `2add10f31`.

## Now

The documentation audit is scoped. Implementation and independent review are
pending. This is an ad hoc documentation task, not a model or engine lifecycle
transition. The work ID identifies the branch and issue, not a new capability.

## Scope

Correct stale ABI, executable, and ROCm statements in `README.md`,
`docs/BUILD.md`, and `docs/reference/c-api.md`. Simplify the affected prose.
Add September news for the shipped ABI additions and native ROCm EXL3 path.
Keep the LocalAI presentation, capability limits, and evidence links.

Exclude runtime code, new tests, checker semantics, benchmarks, model lifecycle
states, and the GLM and FP8 placement edits in PRs #2374 and #2236.

## Source inventory and design

| Surface | Source | Documentation action |
|---|---|---|
| Current ABI and additions | `include/vllm.h:324` through `include/vllm.h:354` | Correct ABI 23 to 26. Explain KV dtype, speculative acceptance, and sliding-window controls with their limits |
| Server executable | `examples/CMakeLists.txt:91`, `examples/CMakeLists.txt:108` | Use `build/examples/vllm-server` |
| ROCm configuration | `cmake/Dependencies.cmake`, `CMakeLists.txt`, `src/vt/rocm/` | Check real option names, compiler selection, and optimization behavior before editing |
| ROCm native coverage and allocation | `src/vt/rocm/backend.cpp`, `docs/ROCM.md`, `.agents/specs/backend-rocm-exl3.md` | Replace the obsolete one-kernel skeleton description with current coverage and explicit unresolved gates |
| Public C example | `include/vllm.h`, `tests/capi/test_capi.cpp` | Present a complete C program if revising the example. Preserve defaults and cleanup |

## Upstream chain

This change describes local interfaces and existing measurements. It ports no
vLLM behavior. No oracle execution applies. Current source and the existing
row evidence define every statement. Do not infer runtime correctness from a
registration or advertise a new speed ratio.

## Tests and gates

Run CPU-only documentation checks: `check-readme-structure.py`,
`check-supported-models.py`, `check-public-doc-tables.py`,
`check-surface-coverage.py`, and `check-benchmark-index.py` under `scripts/`.
Run the relevant existing mutation suites and `git diff --check`.
Run `scripts/agent-preflight.sh` and report each unavailable host-dependent
check separately. Check changed commands and identifiers against source.
A fresh reviewer verifies the immutable diff and repeats focused checks.
No product test is added for a reversible prose correction.

## Dependencies and risks

Only local CPU tools and GitHub access apply. GPU execution, model downloads,
external hosts, and performance measurement are excluded. Python is available
in an isolated tool directory at `/tmp/vllm-doc-tools/root` on this host.
Unrelated baseline failures remain explicit and cannot justify weaker gates.

## Work breakdown

1. Commit this spec before editing public prose.
2. A fresh implementer edits only the three public pages and this spec.
3. A fresh reviewer checks source grounding, scope, and the documented limits.
4. The coordinator reruns the checks and opens a fork PR against upstream main.

## Stop conditions

Do not promote a capability or invent a measurement. A missing source anchor
requires a narrower statement. Product defects require separate tracked work.
No merge authority is granted by the user's request to open a PR.
