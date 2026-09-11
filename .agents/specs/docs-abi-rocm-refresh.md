# Refresh the ABI and ROCm user instructions

Work ID: `ENG-DOCS-ABI-ROCM-REFRESH`.
Issue: [#3055](https://github.com/mudler/vllm.cpp/issues/3055).
Base: `2add10f31`.

## Now

The scoped public corrections include both README repairs from independent
review: the C ABI version and ROCm hardware row. Fresh review is pending.
Full preflight remains INCOMPLETE. This documentation task changes no model or
engine lifecycle state. The work ID identifies the branch and issue.

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
| ROCm configuration | `CMakeLists.txt:337-455`, `src/vt/rocm/` | Check real option names, compiler selection, and optimization behavior before editing |
| ROCm native coverage and allocation | `src/vt/rocm/rocm_backend.hip`, `docs/ROCM.md`, `.agents/specs/backend-rocm-exl3.md` | Replace the obsolete one-kernel skeleton description with current coverage and explicit unresolved gates |
| Public C example | `include/vllm.h`, `tests/capi/test_capi.cpp` | Present a complete C program if revising the example. Preserve defaults and cleanup |

## Upstream chain

This change describes local interfaces and existing measurements. It ports no
vLLM behavior. No oracle execution applies. Current source and the existing
row evidence define every statement. Do not infer runtime correctness from a
registration or advertise a new speed ratio.

## Tests and gates

Run CPU-only documentation checks: `check-readme-structure.py`,
`check-supported-models.py`,
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

## Implementation evidence

The changes correct ABI 26 and the `vllm-server` executable name. September
news describes shipped controls and native ROCm EXL3 generation without adding
a speed claim. The ROCm instructions retain allocation and hardware limits.
The C example now includes `stdio.h`, `main`, and completion failure cleanup.

Source checks used `include/vllm.h:354`, `include/vllm.h:586-599`,
`include/vllm.h:706-713`, and `include/vllm.h:825-882` for ABI values and semantics.
`src/capi/vllm_c.cpp:535-543`, `:688-689`, `:746-760`, and `:862-890` confirm
parameter defaults, loading, and acceptance queries.
`examples/CMakeLists.txt:91-108` defines the server output name.
`CMakeLists.txt:370-449` defines compiler hints and the Debug optimization floor.
`src/vt/rocm/rocm_ops.hip:190-293` registers the documented native operations.
`src/vt/rocm/rocm_backend.hip:200-235` implements allocation selection.
`backend-rocm-exl3.md` records gfx1151 generation and the unmeasured discrete
and competitive performance gates.

With the isolated Python tool directory from this spec, these commands exit 0:

```sh
python3 scripts/check-readme-structure.py
python3 scripts/check-supported-models.py
python3 scripts/check-surface-coverage.py
python3 scripts/check-benchmark-index.py
python3 -m unittest discover -s tests/scripts -p 'test_check_readme_structure.py'
python3 -m unittest discover -s tests/scripts -p 'test_check_supported_models.py'
python3 -m unittest discover -s tests/scripts -p 'test_check_surface_coverage.py'
python3 -m unittest discover -s tests/scripts -p 'test_check_benchmark_index.py'
git diff --check
```

The four existing suites pass 19, 11, 46, and 6 tests respectively, including
their existing negative mutations. No new test or product mutation applies to
this prose change. The source inventory now names the actual ROCm files.
The retired `check-public-doc-tables.py` is absent and is not a runnable gate.

Full preflight is INCOMPLETE. The coordinator and initial implementer invoked
it, then stopped after baseline failures involving missing CMake and ELF tools,
isolated Python loading errors, and unrelated preflight and report failures.
Those runs do not establish a full pass. The repair runs only the role check
and the focused documentation checks above, not another full preflight.

The coordinator supplied an isolated Tiny C Compiler and musl headers.
The initial implementer extracted the first C block from the reference page.
This compilation command exited 0:

```sh
tcc -B/tmp/vllm-doc-tools/root/usr/lib/tcc \
  -I/tmp/vllm-doc-tools/root/usr/include -Iinclude \
  -c /tmp/vllm-docs-impl-c-api.c -o /tmp/vllm-docs-impl-c-api.o
```

This verifies compilation only. Linking, model generation, GPU execution, and
new performance measurements are excluded.

## Review repair evidence

The README now reports `VLLM_ABI_VERSION 26`, matching `include/vllm.h:354`.
Its hardware row describes native EXL3 generation on gfx1151 and retains the
unverified discrete GPU correctness and competitive performance limits.
The source registers `kCastF16` at `src/vt/rocm/rocm_ops.hip:190` and
`kExl3Gemm` at `src/vt/rocm/rocm_ops.hip:292`. The existing
[`BACKEND-ROCM-EXL3` evidence](backend-rocm-exl3.md) establishes CPU reference
agreement on gfx1151 and states the remaining measurement limits.

The repair uses the same isolated Python environment and repeats all four
checks and the 82 tests above. A text search over the three scoped public
pages finds no residual ABI 23 or ROCm skeleton claim. No GPU runs apply.
