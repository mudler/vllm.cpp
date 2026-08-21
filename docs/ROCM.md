# Use the ROCm backend

The ROCm backend runs on integrated and discrete AMD GPUs. Contributors have
run the HIP gates on gfx1100, gfx1103, gfx1151, gfx1200, and gfx1201. Gemma
3 1B IT is token-identical to two vLLM ROCm oracles on gfx1200. Qwen3 0.6B has
one deterministic, version-sensitive near tie on that device.

Qwen3.5 0.8B runs through the native GDN stack, but its CPU and ROCm outputs
still have an open correctness gap. The Gemma 4 FP8 MoE and SharedK WMMA path
has runtime evidence on two gfx1201 GPUs. This repository has only CPU link
coverage for that path and no matched vLLM ROCm performance result.

## Build for ROCm

Use a release build. A build without optimization can trigger a ROCm host-call
teardown race on gfx1100.

```sh
cmake -S . -B build-hip \
  -DVLLM_CPP_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_PATH=/opt/rocm
cmake --build build-hip -j
ctest --test-dir build-hip -R 'rocm|cross_device' --output-on-failure
```

Set `ROCM_PATH` to your ROCm or TheRock installation prefix. CMake derives the
compiler root and HIP flags from that prefix. See the [build guide](BUILD.md)
for all ROCm build options.

The TheRock nightly on gfx1103 can print `Status: SUCCESS!` and then hang during
process teardown. This is a known deadlock in `libamdhip64.so.7`. Record the
ROCm build when you encounter it.

## Select the device

The CLI does not accept `--device rocm`. On a system with AMD hardware and no
CUDA device, the default `--device auto` selects ROCm through the platform
priority order.

```sh
VT_OP_PROVIDER_STATS=1 ./build-hip/examples/vllm-cli \
  --model /path/to/a/small/dense/model \
  --prompt "The capital of France is" \
  --max-tokens 8 \
  --temperature 0
```

Run the same command with `--device cpu` to compare greedy tokens.
`VT_OP_PROVIDER_STATS=1` reports native operations and CPU fallbacks.

## Understand fallback behavior

An integrated GPU can use the CPU reference tier when the backend reports
unified memory. On managed-capable integrated devices, the backend uses
`hipMallocManaged` and reports `UnifiedMemory() == true`. A missing native
operation then runs through its CPU implementation.

A discrete GPU cannot use the reference tier. The model runs only when ROCm
registers every operation that the model needs. `GetOp` reports an error for a
missing operation.

Do not use a run with CPU fallbacks as a performance result.
`GetReferenceTierHits()` must return `0` for a valid measurement.

## Current backend surface

| Seam | File | Current state |
|---|---|---|
| Device enum | [`include/vt/device.h`](../include/vt/device.h) | Compiled and routed through the shared device switch |
| Architecture mapping | [`include/vt/rocm/rocm_arch.h`](../include/vt/rocm/rocm_arch.h) | Unit-tested gfx name mapping |
| Runtime backend | [`src/vt/rocm/rocm_backend.hip`](../src/vt/rocm/rocm_backend.hip) | Runs on five gfx architectures; managed allocation still needs an integrated-board rerun |
| Operation table | [`src/vt/rocm/rocm_ops.hip`](../src/vt/rocm/rocm_ops.hip) | 44 distinct registered `OpId` values at the recorded count |
| Kernels | [`src/vt/rocm/`](../src/vt/rocm/) | Dense, GDN, attention, sampling, and the contributor-tested Gemma 4 FP8 MoE path |
| Platform | [`src/vllm/platforms/rocm.cpp`](../src/vllm/platforms/rocm.cpp) | Runtime-verified on five gfx architectures |
| Attention | [`src/vt/rocm/rocm_paged_attn.hip`](../src/vt/rocm/rocm_paged_attn.hip) | Native paged attention and the SharedK WMMA prefill path |
| Build | [`CMakeLists.txt`](../CMakeLists.txt) | `VLLM_CPP_HIP` configuration and build verified on five architectures |
| Tests | [`tests/vt/test_rocm_backend.cpp`](../tests/vt/test_rocm_backend.cpp) | Runtime cases pass; managed-allocation cases remain pending |

Recount registered operations before you quote the total:

```sh
grep -rho 'RegisterOp(OpId::[A-Za-z0-9_]*' src/vt/<backend>/ | sort -u | wc -l
```

## Hardware notes

| Hardware | Architecture | Memory | Current path |
|---|---|---|---|
| Strix Halo | gfx1151 | Unified | Verify managed allocation, then run a small dense model through the reference tier |
| Radeon 780M | gfx1103 | Shared | Use the same reference-tier path with a smaller model |
| Radeon 7900 XTX | gfx1100 | Discrete | Native kernels are required; this class can also host the vLLM ROCm oracle |
| Radeon R9700 | gfx1201 | Discrete | Contributor-tested Gemma 4 FP8 MoE and SharedK WMMA path |
| Radeon RX 9060 XT | gfx1200 | Discrete | Gemma 3 1B IT oracle parity; Qwen3 0.6B has a recorded near tie |

## Features that need a separate AMD implementation

Do not hipify NVIDIA-specific implementations:

- NVFP4, Marlin, and FP4 tensor-core kernels;
- CUTLASS FlashAttention and scaled matrix multiplication kernels;
- vendored Triton AOT cubins;
- NCCL transport;
- cuBLASLt plan caches.

Use the AMD equivalents where applicable. These include MXFP4, Composable
Kernel or AITER, RCCL, and hipBLASLt.

For correctness evidence and implementation history, see the
[ROCm backend specification](../.agents/specs/rocm-backend-w0.md), the
[unified-memory decision](../.agents/specs/rocm-unified-memory-b.md), and the
[gfx1200 correctness record](../.agents/specs/rocm-gfx1200-m2-correctness.md).
