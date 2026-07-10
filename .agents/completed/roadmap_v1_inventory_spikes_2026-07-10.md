# Completed roadmap v1 inventory groundwork

**Closed:** 2026-07-10. This block established the canonical tables and spike
contract needed for independent agents to claim roadmap work. The inventories
remain live because support states change; this archive records only the
finished coverage work and its handoff.

| Closed spike | Coverage result | Live record | First implementation handoff |
|---|---|---|---|
| `INV-POLICY` | portfolio -> area matrix -> spike -> claim -> evidence -> completed hierarchy; stable IDs and CI validation | [coordination](../coordination.md), [workflow](../workflow.md) | claim only `READY` rows in isolated worktrees |
| `INV-MODEL` | 370 category memberships, 353 unique static architecture IDs, 321 category/implementation targets and dynamic Transformers compatibility | [model matrix](../model-matrix.md), [detailed inventory](../specs/model-family-inventory.md) | spike generic architecture-to-factory and reject-unknown behavior |
| `INV-QUANT` | 76 rows across GGUF/llama.cpp, vLLM-native NVIDIA formats and MLX-native formats; recognition, materialization, quant compute, e2e and performance tracked separately | [quantization matrix](../quantization-matrix.md), [coverage spike](../specs/quantization-coverage.md) | spike GGUF compute-in-quant storage and dispatch |
| `INV-KERNEL` | 30 practical families grounded through vLLM and its execution dependencies | [kernel matrix](../kernel-matrix.md), [family spike](../specs/kernel-family-inventory.md) | spike raw-pointer/shape/stride/stream adapter ABI |
| `INV-BACKEND` | 13 CUDA targets, 18 component target rules, 8 platform/ABI rows and 9 native competitor gates | [backend matrix](../backend-matrix.md), [CUDA/backend spike](../specs/cuda-architecture-inventory.md) | spike the common architecture spine, then parallel target ports |
| `INV-COMPETITORS` | vLLM remains universal CUDA oracle; SGLang low-concurrency, llama.cpp CPU/Vulkan/Metal, and oMLX/MLX-LM Apple floors added | [benchmark spike](../specs/competitive-benchmarks.md), [benchmark protocol](../benchmark-protocol.md) | run the isolated DGX SGLang series after the active PR #3 GPU claim |
| `INV-FEATURE-ANCHORS` | legacy cross-cutting feature surface migrated to stable engine/serving rows; 26 code-bearing claims narrowed to their evidenced slices | [engine matrix](../engine-matrix.md), [anchor-backfill spike](../specs/feature-anchor-backfill.md) | agents write the missing leaf spikes before implementation or `DONE` claims |

## Corrections made by the coverage pass

- Only GB10/sm121a gate workloads are built, traced and gated; this is not full
  CUDA-family or full-kernel support.
- Current GGUF execution materializes supported encodings to bf16. It does not
  provide llama.cpp-class compute-in-quant speed.
- The current Qwen3.5/Qwen3.6 wrappers are model-specific text paths, not a
  generic model-family factory and not complete multimodal support.
- Static ModelOpt FP8 and the two gate-specific NVFP4 paths are supported
  slices; generic FP8/NVFP4/MX breadth remains open.
- FlashInfer owns fused Add/RMSNorm+FP4 kernels even though those kernels are
  outside vLLM `csrc`; dependency and runtime traces are part of the inventory.

## Closure rule applied

Every coverage spike produced stable rows, upstream/dependency anchors, tests
to port, gates, dependencies and parallel work ordering. No implementation row
was marked done merely because it was inventoried. Finished spike claims were
removed from the active board; the live matrices now own all future status.
