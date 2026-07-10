# Spike: vLLM model-family inventory

**Rows:** every `MODEL-*` row in
[model-matrix.md](../model-matrix.md) at parity pin `e24d1b24fe96`.
**State:** inventory spike accepted; each implementation target still requires
its own leaf spike before `READY`.

## Scope

Enumerate every finite static architecture registration in pinned vLLM, group
aliases only when `(category, module, class)` is identical, preserve stable
claim IDs, and expose the dynamic Transformers route separately. This spike
does not claim that shared layers make a model supported.

## Upstream chain

- Registry and merge order:
  `/home/mudler/_git/vllm/vllm/model_executor/models/registry.py:71-693`.
- Lazy resolution, capabilities, and dynamic Transformers fallback:
  `registry.py:998-1164,1192-1296,1396-1403`.
- Per-target implementation paths are recorded directly in each canonical
  [model row](../model-matrix.md).
- A leaf spike must continue through loaders, quantization, platform dispatch,
  dependency kernels, runtime traces, and the target's upstream tests.

## Our baseline

The local factory accepts only the Qwen3.5 MoE wrapper at
`src/vllm/model_executor/models/registry.cpp:10-20`; the function type at
`include/vllm/model_executor/models/registry.h:18-21` is model-specific.
`LoadedEngine::IsDenseArch` at
`src/vllm/entrypoints/model_loader.cpp:232-237,266-277` can classify an unknown
dense configuration as Qwen3.5 instead of rejecting it. The two Qwen3.5
conditional-generation rows have grounded text-only `PARTIAL` evidence in the
matrix; vision remains absent.

## Port map

| Upstream surface | Local destination |
|---|---|
| `models/registry.py` | `src/vllm/model_executor/models/registry.cpp` plus a task-aware type-erased factory |
| target `models/<module>.py` | matching `src/vllm/model_executor/models/<module>.{h,cpp}` |
| loader/config mapping | `src/vllm/entrypoints/model_loader.cpp` and model weight loaders |
| reusable layer/ops | existing `src/vllm/model_executor/layers/` and `src/vt/`, extended only from a leaf dependency spike |
| model tests and fixtures | `tests/vllm/models/`, `tests/parity/`, and model-specific e2e gates |

## Tests to port

- `tests/models/test_registry.py:31-158`: registration, lazy import,
  capabilities, pipeline flags, and failures.
- `tests/models/registry.py:16-1742`: one traceable task/tokenizer/dtype fixture
  per architecture alias.
- `tests/models/test_initialization.py:50-197`: construction and loading.
- `tests/models/test_transformers.py`: native/generic equivalence and generic
  backend behavior.
- The exact generation, pooling, multimodal, classification, or speculative
  modules named by the target leaf spike. Blocked cases land skipped with the
  stable `MODEL-*` ID and reason.

## Gates

Each target needs config/weight mapping tests, applicable upstream test ports,
real-checkpoint correctness against pinned vLLM, negative unknown-architecture
behavior, backend and quantization coverage, peak memory, and the same-workload
performance protocol. Dynamic dispatch is traced when source inspection cannot
prove which dependency kernel ran.

## Dependencies

The cross-cutting `MODEL-FACTORY-registry` row precedes new native families.
Target rows declare their `KERNEL-*`, `QUANT-*`, `BACKEND-*`, `ATTN-*`, `KV-*`,
and serving dependencies in their leaf spike. Multimodal and pooling targets
also depend on their task-specific runner/API surfaces.

## Work breakdown

1. Close `MODEL-FACTORY-registry` with deterministic reject-unknown behavior.
2. Spike dense shared-stack rows: Llama, Qwen2/3, Mistral, Gemma, Phi.
3. Spike MoE shared-stack rows: Mixtral, Qwen2/3-MoE, GLM4-MoE, OLMoE.
4. Port Qwen3-Next while Qwen3.5 MTP/DFlash proceed independently.
5. Expand hybrid/MLA families, then pooling/classification and multimodal rows.
6. Define the bounded pure-C++ dynamic Transformers compatibility contract.

Claims are per stable matrix row or an explicitly listed non-overlapping row
set. A family umbrella never grants ownership of every target beneath it.

## Risks and decisions

- Registry presence is not runtime support; `DONE` requires local code, tests,
  checkpoint evidence, and backend/performance gates.
- Architecture aliases are grouped only for the exact same upstream target.
- The dynamic Transformers route is capability-driven and therefore unbounded;
  its completion criterion is a tested compatibility contract, not a finite
  count.
- Upstream tests are the executable spec and are ported with the implementation.
