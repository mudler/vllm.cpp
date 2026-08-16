# Building vllm.cpp

vllm.cpp uses CMake (>= 3.24) and a C++20 compiler (gcc 13/14 and clang are
exercised; the tree builds -Werror-clean on gcc 14.2). The core has no ML
dependencies; the OpenAI server uses a vendored header-only HTTP transport
(cpp-httplib). The [README](../README.md) carries the two-line quickstart; this
page is the full build reference.

## Build out-of-source

Every recipe on this page configures into a separate build directory, and that
is a requirement, not a style preference: the example targets are named after
the directories they are built from (`examples/tokenize` builds `tokenize`,
`examples/dump_container` builds `dump_container`, ...), so `cmake .` asks the
linker to write each executable on top of its own source directory and fails
with `cannot open output file <target>: Is a directory` (issue #85).

Configure refuses an in-source build up front and says so. If an earlier attempt
already wrote into the checkout, clear it with
`rm -rf CMakeCache.txt CMakeFiles`.

## CPU build (the correctness / CI reference)

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

The server is ON by default. Example binaries land under `build/examples/`:
`vllm-cli`, `server`, `vllm-bench`, and `tokenize`.

## CUDA build (NVIDIA GB10 / DGX Spark)

```sh
cmake -S . -B build-cuda \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_CUTLASS_FETCH=ON
cmake --build build-cuda -j
```

Triton-AOT cubins for the fast GDN path are **vendored**: Python and Triton are
needed only to regenerate them (`VLLM_CPP_TRITON_REGEN`), never to build or run
them. Because building them needs nothing a CUDA build does not already have,
`VLLM_CPP_TRITON` defaults **ON** for a CUDA build and the line above no longer
carries it; `-DVLLM_CPP_TRITON=OFF` drops back to the hand C++/CUDA kernels,
which stay the always-available fallback.

### CUTLASS: the one external build dependency

CUTLASS (>= 4.5.0) is header-only, and it is the only thing a CUDA build fetches
from the network. It feeds two independent consumers:

- **FlashAttention-2** prefill/decode, on every arch in `8.0 8.6 8.7 8.9 12.0a 12.1a`.
- The **sm_12xa NVFP4 block-scaled GEMM**, on Blackwell only.

Without it the FA2 kernels are not compiled and attention falls back to the
portable path, which is slower. Nothing fails and no test goes red, so it is
worth being deliberate about. Pick one:

```sh
-DVLLM_CPP_CUTLASS_FETCH=ON            # download CUTLASS 4.5.0 (~200 MB, needs network)
-DVLLM_CPP_CUTLASS_DIR=/path/to/cutlass  # reuse a checkout you already have
```

The default is neither, so that a disk- or network-constrained box configures
without surprises. When you skip it on an arch that supports FA2, configure
prints a `CMake Warning` saying so.

Confirm you got it from the configure output:

```
-- CUDA feature fa2: ENABLED for [86]
-- FlashAttention-2 prefill/decode: ENABLED for arch(es) [86] (runtime toggles VT_FA2_PREFILL, VT_FA2_DECODE)
```

The first line means the arch supports FA2; the second means it was actually
built. If only the first appears, CUTLASS was not found.

### Other CUDA families

Set the arch explicitly. It defaults to `121a` (GB10), which will not load on
anything else:

```sh
# Hopper H100/H200
cmake -S . -B build-cuda -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=90a \
  -DVLLM_CPP_CUTLASS_FETCH=ON

# Ampere consumer (RTX 3090 = 86), Ada (89), Jetson Orin (87)
cmake -S . -B build-cuda -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=86 \
  -DVLLM_CPP_CUTLASS_FETCH=ON
```

Of these, only `sm_87` (Orin) and `sm_110` (Thor) have been run on real
hardware. `sm_80/86/89`, `sm_90a` and `sm_100a/103a` are build-verified: they
compile `-Werror`-clean and emit the expected SASS, but no board here has
executed them. See [STATUS.md](STATUS.md) for what that label means and
`.agents/specs/cuda-arch-ampere-fastpath.md` for the per-arch detail. Reports
from those boards are welcome.

## Metal build (Apple Silicon)

Metal is detected automatically on an Apple host with an ObjC++ compiler. The
optional MLX GEMM provider is a separate opt-in and needs an MLX install:

```sh
cmake -S . -B build-metal -DVLLM_CPP_MLX=ON -DMLX_ROOT=/path/to/mlx
cmake --build build-metal -j
```

MLX is shape-gated to prefill, where it wins; it declines the `m < 2` decode
GEMV by design. Note that an MLX build produces a **different greedy sequence**
than the default build (MLX's GEMM is not bit-identical), so goldens must not be
re-anchored to an MLX build. Details in [docs/BENCHMARKS.md](BENCHMARKS.md).

## Vulkan build

Headers are vendored and SPIR-V is committed, so no graphics toolchain is
needed. It is off unless requested:

```sh
cmake -S . -B build-vulkan -DVLLM_CPP_VULKAN=ON
cmake --build build-vulkan -j
```

## Tenstorrent build (Blackhole)

The Tenstorrent backend is opt-in and requires local TT-Metalium and TT-NN
package installations. Point `CMAKE_PREFIX_PATH` at the built tt-metal install
tree. Configure fails instead of silently producing a CPU-only build when
either package is missing.

```sh
cmake -S . -B build-tenstorrent \
  -DVLLM_CPP_TENSTORRENT=ON \
  -DCMAKE_PREFIX_PATH=/path/to/tt-metal/install
cmake --build build-tenstorrent -j
```

The Blackhole lane is correctness-focused. OPT-125m passes its strict 6/6
end-to-end gate. Qwen3-0.6B is selected by the platform and has a device-aware
near-tie gate plus committed goldens, but the full 16x16 rerun is still pending.
There is no binding speed result.

## ROCm build (AMD GPUs) — community-verified W0, blind F6 fix

> The W0 HIP sources compiled clean and passed `ctest -R 'rocm|cross_device'`
> on four community boards — gfx1151, gfx1103, gfx1100, gfx1201
> ([issue #41](https://github.com/mudler/vllm.cpp/issues/41)). The
> unified-memory fix on top of them (approach (b),
> [docs/ROCM.md §3.1](ROCM.md)) was again written with **no AMD GPU or ROCm
> toolchain on any maintainer machine**, so a compile error in it is expected,
> useful, and belongs on #41.

```sh
cmake -S . -B build-hip -DVLLM_CPP_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
ctest --test-dir build-hip -R 'rocm|cross_device'
```

`VLLM_CPP_HIP_ARCHITECTURES` is optional: leave it empty and hipcc targets the
installed GPU, which is what you want when building on the machine you will run
on. The validated names are upstream vLLM's `HIP_SUPPORTED_ARCHS`; anything else
configures with a warning and is passed to hipcc anyway. If ROCm lives outside
`/opt/rocm`, point at it with `-DROCM_PATH=<prefix>`. When `ROCM_PATH` names a
real install, the configure now derives the compiler hints from it
(`CMAKE_HIP_COMPILER_ROCM_ROOT`, `--rocm-path` in `CMAKE_HIP_FLAGS`, the
`ROCM_PATH` environment variable — each only if you have not set it), which is
what makes Arch and TheRock dist-tarball layouts configure without the manual
flags issue #41's gfx1151 report needed.

A build with no `CMAKE_BUILD_TYPE` now floors **HIP device code** at `-O1`
automatically, and says so at configure time. At `-O0` hipcc marks the kernels
as dynamic-stack users, which makes the ROCm runtime start a hostcall listener
the kernels never use, and its teardown handshake can deadlock at process exit —
the tests all pass and then the process never returns
([#132](https://github.com/mudler/vllm.cpp/issues/132)). Setting a build type,
or putting your own `-O` in `CMAKE_HIP_FLAGS`, overrides this and is respected
as-is.

`-DVLLM_CPP_HIP=ON` **fails the configure** when no HIP compiler is found rather
than quietly producing a CPU-only build, for the same reason the CUTLASS note
above exists: a silent downgrade is indistinguishable from success.

What exists today is the W0 skeleton — the `vt::Backend`, the `Platform`, one
registered kernel (RmsNorm), and the tests that gate them — plus the approach-(b)
unified-memory branch for integrated APUs. What that does and
does not get you, and where to start on your specific board, is
[docs/ROCM.md](ROCM.md).

## Nix shells

The checked-in flake pins CMake, Ninja and the CUDA toolchain, so nothing has
to be globally installed:

```sh
# CPU (correctness / CI reference)
nix develop .#default --command cmake -S . -B build-nix-cpu -G Ninja \
  -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
nix develop .#default --command cmake --build build-nix-cpu -j4

# CUDA (set the arch for your GPU)
nix develop .#cuda --command bash -lc \
  'cmake -S . -B build-nix-cuda -G Ninja -DVLLM_CPP_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$CMAKE_CUDA_COMPILER" \
    -DCMAKE_CUDA_HOST_COMPILER="$CMAKE_CUDA_HOST_COMPILER" \
    -DVLLM_CPP_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=RelWithDebInfo'
nix develop .#cuda --command cmake --build build-nix-cuda -j4
```

On NixOS the CUDA shell exports the driver-library path and
`TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` so Triton finds `libcuda` without
`/sbin/ldconfig`.

## CMake options

Read from [`CMakeLists.txt`](../CMakeLists.txt). Defaults shown are the shipped
defaults.

| Option | Default | Purpose |
|---|---|---|
| `VLLM_CPP_CUDA` | `AUTO` | Build the CUDA backend: `ON`, `OFF`, or `AUTO` (on when a CUDA toolchain is found) |
| `VLLM_CPP_CUDA_ARCHITECTURES` | `121a` | Target CUDA arch(s): `121a` (GB10), `120a`/`120a;121a` (consumer Blackwell), and cross-family targets `90a`, `80`/`86`/`87`/`89`, `100a`/`103a`, `110`. The `a` suffix is required for the native fp4 MMA |
| `VLLM_CPP_METAL` | `AUTO` | Build the Metal backend: `ON`, `OFF`, or `AUTO` (on for an Apple host with an ObjC++ compiler) |
| `VLLM_CPP_VULKAN` | `AUTO` (= `OFF`) | Build the Vulkan backend. Opt-in with `-DVLLM_CPP_VULKAN=ON`; headers are vendored and SPIR-V is committed |
| `VLLM_CPP_TENSTORRENT` | `AUTO` (= `OFF`) | Build the Tenstorrent backend. Opt-in with `-DVLLM_CPP_TENSTORRENT=ON`; requires TT-Metalium and TT-NN and fails configure if either package is missing |
| `VLLM_CPP_HIP` | `AUTO` (= `OFF`) | Build the ROCm/HIP backend. Opt-in with `-DVLLM_CPP_HIP=ON`, which fails loudly if no `hipcc` is found. Community W0 builds cover four `gfx` targets; model gates remain open |
| `VLLM_CPP_HIP_ARCHITECTURES` | (empty) | Target `gfx` arch(es), e.g. `gfx1100` or `gfx1100;gfx1151`. Empty means hipcc targets the installed GPU |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation prefix, for a nightly/TheRock install elsewhere |
| `VLLM_CPP_MLX` | `OFF` | Build the optional MLX GEMM provider for Metal (needs `-DMLX_ROOT=<mlx install>`) |
| `MLX_ROOT` | (empty) | Root of an MLX install (`include/` + `lib/`) for `VLLM_CPP_MLX` |
| `VLLM_CPP_SERVER` | `ON` | Build the OpenAI HTTP server (needs `third_party/httplib/httplib.h`; disables itself with a warning if absent) |
| `VLLM_CPP_TRITON` | computed: `ON` for a CUDA build with the vendored trees present, else `OFF` | Consume the vendored per-arch Triton-AOT GDN cubins (CUDA only; no Python needed). It ships ON because the artifacts are pre-generated cubins embedded in plain C — a C compiler is the whole requirement. It declines, with one `STATUS` line naming the condition, when `VLLM_CPP_CUDA` is `OFF`, when `VLLM_CPP_TRITON_REGEN` is `ON`, or when a vendored tree is absent, incomplete, or older than the `triton_kernels/*.py` it was generated from — the same conditions the build itself refuses to consume, checked before the default is set so it can decline instead of failing your configure. Turn it off with `-DVLLM_CPP_TRITON=OFF` |
| `VLLM_CPP_TRITON_REGEN` | `OFF` | Maintainer knob: regenerate the AOT cubins with Python + Triton |
| `VLLM_CPP_CUTLASS_DIR` | `third_party/cutlass` | CUTLASS source root (>= 4.5.0). Feeds the sm120a NVFP4 GEMM **and** FlashAttention-2 on `8.0/8.6/8.7/8.9/12.0a/12.1a`. Absent on an FA2-capable arch, configure warns and FA2 is not built |
| `VLLM_CPP_CUTLASS_FETCH` | `OFF` | FetchContent CUTLASS 4.5.0 if not found locally (~200 MB, needs network) |
| `VLLM_CPP_MARLIN` | `ON` | Build the vendored Marlin NVFP4 W4A16 MoE GEMM (sm_12xa) |
| `VLLM_CPP_BUILD_TESTS` | `ON` | Compile and register ctest targets |
| `VLLM_CPP_BUILD_EXAMPLES` | `ON` | Build the example CLI, server, and bench binaries |
| `VLLM_CPP_BENCH_PROFILE_CONTROL` | `OFF` | Trace-only profiler replay control (never for production timing builds) |

## Backend and hardware state

| Backend | Hardware | State |
|---|---|---|
| CPU | x86-64 and arm64 | Correctness / CI reference; at or ahead of llama.cpp on every GGUF axis, with an Arm i8mm quant-GEMM tier. That verdict's denominator is SUPERSEDED: it was our own fork `237ad9b96`, and the oracle is now stock `b10451` (re-take owed, #1003) |
| CUDA | GB10 / DGX Spark, sm_121a | Gate-model correctness passes; 27B at/above vLLM throughput, 35B prefill-pending. The only runtime-gated CUDA target |
| CUDA | Consumer Blackwell, sm_120a | Build-supported (compiles, emits real sm_120a code, all fast paths resolve) but not runtime-proven here (no such card) |
| CUDA | Hopper, sm_90a | Build-supported; the fast GDN (Triton-AOT) path is build-verified, not runtime-proven here |
| CUDA | Ampere/Ada (sm_80/86/87/89), datacenter Blackwell (sm_100a/103a) | Build-supported; the fast GDN path is build-verified per-arch on sm_80/86/89/100a (plus FA2 on Ampere, sm_100a NVFP4 GEMM), not runtime-gated here. sm_70/sm_75 unsupported (no bf16 tensor cores) |
| CUDA | Jetson Thor, sm_110 | Runtime-verified: the portable bf16 path ran the Llama-3.2-1B greedy gate token-exact on real silicon. Community reports add a 32B NVFP4A16 serving through the portable dequant-GEMM, and CUDA 13.2 passing the CUDA gates. fp8/fp4/CUTLASS/Marlin/FA2 fast paths resolve EMPTY for `110` |
| Metal | Apple Silicon | Two models run end to end and pass correctness; 18 of 75 ops native. Warm b=1 throughput is 95.9% of MLX-LM, or 97.6% with the optional MLX provider gated to prefill (where we are 1.5% ahead). Indicative |
| Vulkan | Portable GPU | `opt-125m` is strict token-exact; Qwen3.6-27B decode matches llama.cpp Vulkan on GB10. See [BENCHMARKS.md](BENCHMARKS.md) for the measured scope |
| Intel XPU | Intel GPUs | Spiked, hardware-blocked |
| ROCm | AMD GPUs | W0 tests passed on four community `gfx` targets; ROCm 6.x build fix landed. Model and oracle gates remain open: [ROCM.md](ROCM.md) |
| Tenstorrent | Blackhole | `ACTIVE`: OPT-125m strict 6/6 on real hardware; Qwen3-0.6B gate wired, full 16x16 rerun and performance path pending |
| ANE | Apple Neural Engine | Post-parity roadmap |

Only GB10 / sm_121a is a runtime-gated CUDA target today. Consumer Blackwell
(`120a`) plus the cross-family targets are build-supported (they compile and emit
real machine code, with the fast GDN path build-verified on several) but unproven
at runtime here (no such board), and non-Apple / non-NVIDIA backends run a subset
of operations. Per-op detail is in the
[backend matrix](../.agents/backend-matrix.md).

## Quantization formats

| Format | State |
|---|---|
| NVFP4 W4A4 / W4A16 | Both gate-model paths run on GB10, token-exact. FP4 tactics match vLLM; Marlin NVFP4 W4A16 grouped-MoE is the 35B expert path |
| compressed-tensors NVFP4A16 (W4A16), dense | Correctness-complete via the Marlin weight-only path; speed not yet measured |
| GGUF F32 / F16 and block quantization | Supported. CPU keeps the supported Q, IQ, and MXFP4 blocks compressed through the matrix multiply. CUDA also keeps the supported Q8_K-activation formats compressed; other formats fall back to expansion or CPU compute. Set `VT_GGUF_KEEP_QUANT=0` to disable the direct path. See [STATUS.md](STATUS.md) for the exact format and backend coverage |
| FP8 (W8A8) | The 35B ModelOpt static per-tensor projection slice is implemented; generic FP8 modes and FP8 KV remain open |
| compressed-tensors MXFP4 (W4A16) | Qwen3 dense weights load and run through the Marlin path on CUDA. Qwen3-8B is correctness-gated and benchmarked against vLLM; c1 passes the speed floor, while c2-c8 remain below it. MXFP8 compute remains open |

## Environment variables

Runtime knobs (op providers, keep-quant, profiling) are documented in
[docs/ENVIRONMENT.md](ENVIRONMENT.md).
