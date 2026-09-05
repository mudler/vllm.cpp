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

## Select Q8_K activation quantization

`VT_ROCM_Q8K_BLOCK` selects the Q8_K activation quantizer used by ROCm dense
and grouped keep-quant operations. When it is unset, a queue device that
resolves to `gfx1100`, including a valid feature-suffix spelling, uses the
cooperative arm. Unset uses the legacy arm on `gfx1200`, `gfx1201`, an unknown
architecture, or architecture-resolution failure. This policy does not imply
runtime validation on `gfx1200` or `gfx1201`.

The parser is strict. Set `VT_ROCM_Q8K_BLOCK=0` for the permanent legacy A/B
arm, or set `VT_ROCM_Q8K_BLOCK=1` to force the cooperative diagnostic arm
regardless of architecture. Explicit `1` is diagnostic-only outside validated
`gfx1100`; it does not make a correctness, performance, or default claim for
another architecture. Any value other than `0` or `1` refuses the operation
instead of choosing an arm silently.

## Understand fallback behavior

An integrated GPU can use the CPU reference tier when the backend reports
unified memory. A missing native operation then runs through its CPU
implementation.

The backend reports unified memory when it allocates with `hipMallocManaged`,
or when the device reports `Integrated` and `PageableMemoryAccess` together. It
allocates with `hipMallocManaged` only on a managed-capable integrated device
that also reports `PageableMemoryAccess = 1`. A device that reports
`PageableMemoryAccess = 0` cannot take a recoverable page fault, so a migratable
allocation on it faults the GPU under load
([#2511](https://github.com/mudler/vllm.cpp/issues/2511) measured 17 faults in
21 runs, against 0 in 21 with plain `hipMalloc`).

**Strix Halo (gfx1151) and Radeon 780M (gfx1103) report
`PageableMemoryAccess = 0`, so they get plain `hipMalloc` and no reference
tier.** They run a model when ROCm registers every operation the model needs,
the same rule a discrete card follows. `GetOp` reports an error for a missing
operation, and on these boards that error also names the attribute, the issue,
and `VT_ROCM_MANAGED_ALLOC=1`, which restores the previous managed behaviour at
that risk. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for the knob.

A discrete GPU cannot use the reference tier. The model runs only when ROCm
registers every operation that the model needs. `GetOp` reports an error for a
missing operation.

Do not use a run with CPU fallbacks as a performance result.
`GetReferenceTierHits()` must return `0` for a valid measurement.

## Current backend surface

| Seam | File | Current state |
|---|---|---|
| Device enum | [`include/vt/device.h`](../include/vt/device.h) | Compiled and routed through the shared device switch |
| Architecture mapping | [`include/vt/rocm/rocm_arch.h`](../include/vt/rocm/rocm_arch.h) | Unit-tested gfx name mapping, and the unit-tested allocator/host-addressability policy table |
| Runtime backend | [`src/vt/rocm/rocm_backend.hip`](../src/vt/rocm/rocm_backend.hip) | Runs on five gfx architectures; managed allocation is measured on gfx1151 and narrowed to devices that can take a recoverable page fault (#2511) |
| Operation table | [`src/vt/rocm/rocm_ops.hip`](../src/vt/rocm/rocm_ops.hip) | One `Registrar` that names every `OpId` this backend serves natively. Recount it with the command below rather than quoting a number from here |
| Kernels | [`src/vt/rocm/`](../src/vt/rocm/) | Dense, GDN, attention, sampling, and the contributor-tested Gemma 4 FP8 MoE path |
| Platform | [`src/vllm/platforms/rocm.cpp`](../src/vllm/platforms/rocm.cpp) | Runtime-verified on five gfx architectures |
| Attention | [`src/vt/rocm/rocm_paged_attn.hip`](../src/vt/rocm/rocm_paged_attn.hip) | Native paged attention and the SharedK WMMA prefill path |
| Build | [`CMakeLists.txt`](../CMakeLists.txt) | `VLLM_CPP_HIP` configuration and build verified on five architectures |
| Tests | [`tests/vt/test_rocm_backend.cpp`](../tests/vt/test_rocm_backend.cpp) | Runtime cases pass; the allocation-path case asserts the #2511 coupling on the board it runs on |

Recount registered operations before you quote the total. The scan must not
depend on where the argument list wraps: several calls in `rocm_ops.hip` break
the line after `RegisterOp(`, and a line-based `grep` never sees `RegisterOp(`
and `OpId::` together on those (#2573). Read the whole file and match across
newlines:

```sh
grep -rhoz 'RegisterOp([[:space:]]*OpId::[A-Za-z0-9_]*[[:space:]]*,[[:space:]]*DeviceType::kROCM' \
    src/vt/rocm/ | tr '\0' '\n' | grep -o 'OpId::[A-Za-z0-9_]*' | sort -u | wc -l
```

Substitute the `DeviceType::` value for another backend. Naming the device in the
pattern is what keeps the count answering the question a reader asked -- the
previous command counted every `RegisterOp(OpId::` line in the directory
regardless of which device it registered for.

## Hardware notes

| Hardware | Architecture | Memory | Current path |
|---|---|---|---|
| Strix Halo | gfx1151 | Unified, `PageableMemoryAccess = 0` | Plain `hipMalloc`, no reference tier: run a model whose every operation ROCm registers natively (#2511) |
| Radeon 780M | gfx1103 | Shared, `PageableMemoryAccess = 0` | Same path as Strix Halo, with a smaller model |
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
