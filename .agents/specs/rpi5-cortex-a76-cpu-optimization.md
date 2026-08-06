# BACKEND-CPU leaf spike: Raspberry Pi 5 Cortex-A76 optimization campaign

Date: 2026-08-06

Row: `BACKEND-CPU`

Claim: draft PR #65, branch `row/BACKEND-CPU`

State at spike: `PARTIAL`, Raspberry Pi 5 correctness and speed `GATING`

## Goal and scope

Run `Qwen3.5-2B-UD-Q8_K_XL.gguf` correctly on a Raspberry Pi 5, then
systematically optimize every CPU loop reached by that fixed workload. Begin
with portable C++, proceed to Cortex-A76-tuned C++/NEON/SDOT, and add AArch64
assembly only where compiler output and hardware counters prove a remaining
instruction-selection, scheduling, register-allocation, spill, addressing or
branching defect.

The campaign covers model load, prefill, decode, sampling, the CPU threadpool
and serving overhead. It does not optimize kernels that the fixed Qwen trace
never reaches, claim other Arm cores, or run vLLM on the Pi. The x86-64 pinned
vLLM/current-engine run supplies correctness goldens; same-file llama.cpp on
the Pi is the performance and memory floor.

## Fixed vehicle and hardware

- Model: `unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-UD-Q8_K_XL.gguf`, 2.83 GB,
  SHA-256 `a53988df91157d78acaf3c95e22db179d13f6236061bdb86576494dc99b1bc3b`.
- Target: Raspberry Pi 5 Model B Rev 1.1, four Cortex-A76 r4p1 cores,
  1.5-2.4 GHz, 64 KiB L1D + 64 KiB L1I and 512 KiB L2 per core, shared
  2 MiB L3, 64-byte cache line, 8 GiB RAM.
- ISA observed from Linux: AArch64, ASIMD/NEON, FP16, RDM and DotProd/SDOT.
  FEAT_I8MM is absent, so the existing MMLA tier must remain unavailable.
- Kernel: Debian Raspberry Pi `6.18.34+rpt-rpi-2712`. The
  `armv8_cortex_a76` PMU is exposed as perf event source type 9 for CPUs 0-3,
  with cycles, retired/speculated instructions, frontend/backend stalls,
  branch, TLB and L1/L2/L3/last-level cache events.
- Native toolchain observed: GCC 14.2, CMake 3.31.6, Ninja and binutils 2.44.
  `perf` and LLVM are not initially installed; direct `perf_event_open` is the
  binding harness path, with `perf`/`llvm-mca` optional cross-checks.

## Build and execution split

The Pi is an execution target, not a build host. `docker/Dockerfile.arm64`
mirrors the repository-family buildx pattern: Ubuntu 24.04 runs as
`linux/arm64` under QEMU, configures a Release CPU-only build, compiles
`vllm-bench` plus `vllm-cpu-kernel-bench`, executes a small quantized-matmul
smoke, and exports only those two binaries:

```sh
docker buildx build --builder pf-arm --platform linux/arm64 \
  -f docker/Dockerfile.arm64 --target export \
  --output type=local,dest=/tmp/vllm-arm64 .
```

This keeps source compilation, build packages and build trees off the Pi. The
exported glibc 2.39 binaries are deployed into a disposable user-owned Pi
directory and execute against the Pi's newer glibc. QEMU proves AArch64 build,
link and instruction execution, but its time and virtual PMU are never accepted
as Cortex-A76 performance evidence. All binding cycles, stalls, cache events,
wall time, frequency, temperature and throttling metadata come from execution
of the exported artifact on the physical Pi.

## Upstream and dependency chain

vLLM is the semantic oracle, not the Pi implementation source:

- `${VLLM_SOURCE}` at `555967922`: `vllm/platforms/cpu.py` owns CPU platform
  selection; `csrc/cpu/torch_bindings.cpp` is its compiled-op entry layer.
- Qwen3.5 model semantics and tests remain those already ported by the
  `Qwen3_5ForConditionalGeneration`/dense GGUF route. No model-specific decode
  driver or kernel is introduced here.

The performance implementation reference is llama.cpp at the project pin
`237ad9b96`:

- `ggml/src/ggml-cpu/ggml-cpu.c`: graph execution, worker pool, task
  partitioning and tensor-trait dispatch.
- `ggml/src/ggml-cpu/quants.c` and `ggml/src/ggml-cpu/arch/arm/quants.c`:
  portable and Arm quantized dot products.
- `ggml/src/ggml-cpu/repack.cpp`: CPU feature-driven Q8 repacking and
  microkernel selection.
- `src/models/qwen35.cpp`: the same architecture/model path used for the
  on-device competitor.

Additional microkernel references are Arm Compute Library, KleidiAI and
XNNPACK. They are design evidence only: code is ported only when licensing,
semantics, layout and tests are reconciled explicitly.

## Current implementation inventory

The current production path is already shared and model-independent:

| Surface | Current anchor | Pi disposition |
|---|---|---|
| Backend and memory | `src/vt/cpu/cpu_backend.cpp` | portable, bring-up first |
| Persistent workers | `src/vt/cpu/cpu_threadpool.cpp` | 1/2/4-core scaling and affinity profiling owed |
| Generic CPU ops | `src/vt/cpu/cpu_ops.cpp` | trace all reached loops, preserve shared op routing |
| F16/BF16/F32 GEMM | `src/vt/cpu/cpu_matmul_elem.cpp` | NEON exists; A76 schedule/roof analysis owed |
| Quant GEMM dispatcher | `src/vt/cpu/cpu_quant_gemm.cpp` | portable path works; Pi-specific dispatch absent |
| Portable quant dots | `src/vt/cpu/cpu_quant_dot.cpp` | correctness fallback |
| Arm quant dots | `src/vt/cpu/cpu_quant_dot_arm.cpp` | i8mm-only fast tier; inert on Pi |
| Q8 repack | `src/vt/cpu/cpu_quant_repack{,_arm}.cpp` | i8mm path inert; DotProd/SDOT alternative owed |
| Paged attention | `src/vt/cpu/cpu_paged_attn.cpp` | correctness and 1/2/4-core scaling owed |
| Qwen3.5 forward | `src/vllm/model_executor/models/qwen3_5*.cpp` | reuse unchanged; no Pi-private forward |
| Provider selection | `include/vt/op_provider.h`, `src/vt/op_provider.cpp` | register A76 variants here with same-binary fallback |
| CPU microbench | `examples/cpu_kernel_bench/main.cpp` | R1 vt-op/PMU substrate CPU-gated; QEMU build and physical-Pi execution green |
| Legacy quant evidence | `examples/quant_gemm_bench/main.cpp` | retained unchanged for prior results |

Every operation observed in the Qwen trace is entered into the experiment
manifest before tuning. Expected families are quant and elementwise GEMM/GEMV,
activation quantization, GDN prefill/decode, paged attention, normalization,
RoPE, elementwise fusion, embedding/logits, sampling, threadpool barriers and
loader page/repack work. The trace, not this expectation, is binding.

## Benchmark harness contract

Generalize the quant-only evidence tool into `vllm-cpu-kernel-bench` with:

- `--op`, `--dtype`, `--m`, `--n`, `--k`, `--threads`, `--variant`,
  `--warmup`, `--iterations`, `--cache={hot,l2,l3,stream}` and
  `--format={text,json}` controls;
- deterministic synthetic inputs plus exact shape/layout fixtures captured
  from the Qwen workload;
- batched timing that keeps timer/syscall cost below 0.1% of measured work;
- direct Linux `perf_event_open`, grouped so cycles/instructions accompany
  each category pass and multiplexing is reported rather than hidden;
- time, cycles, instructions, IPC, frontend/backend stalls, branch misses,
  cache/TLB events, bytes, effective bandwidth, checksum, compiler/flags,
  CPU identity, affinity, frequency, temperature and throttling metadata;
- scopes for kernel, `vt::` operation, transformer block, full prefill, full
  decode and serving request, so every accepted micro-change is checked in
  its enclosing scopes.

Runs pin one worker to one core for single-thread evidence and use CPUs 0-3 for
four-thread evidence. A run is void if the task migrates, the PMU group is
unaccounted-for, throttling is reported, frequency is unstable, another heavy
process overlaps, or output checksums differ.

## Optimization procedure

For each reached hot loop, build a measured speed-limit model before editing:

1. frontend fetch/decode and branch prediction;
2. dispatch/uop and execution-pipe pressure;
3. loop-carried dependencies, accumulator count and scheduler/ROB occupancy;
4. register pressure, spills and address-generation work;
5. load/store issue width, cache-line crossings, cache/TLB misses and DRAM
   bandwidth;
6. synchronization, false sharing, work partition and memory-bandwidth scaling.

Test algorithm/layout/fusion/recomputation first, then blocking, alignment,
load width, unrolling, multiple accumulators, software pipelining, load/compute
interleaving and prefetch distance. GCC optimized/missed reports and disassembly
explain compiler behavior; `llvm-mca` is advisory; Pi PMU evidence is binding.

An intrinsic variant is promoted to assembly only after it plateaus and the
compiler defect is demonstrated. Assembly lives in a separate `.S` file,
follows AAPCS64 and CFI rules, has no core-loop stack spill unless measured and
justified, handles complete aligned tiles, and declines to the tested C++ path
for unsupported features, shapes, alignment and tails.

## Runtime dispatch and compatibility

- Publish Linux CPU capability bits before first CPU op resolution.
- Detect DotProd through HWCAP. Detect the Cortex-A76 part before selecting a
  schedule specialized for that microarchitecture.
- Register optimized implementations through `vt::OpProvider` at a higher
  priority than `vt-native`; a call that cannot serve its dtype/shape declines
  to the cached fallback.
- Keep a same-binary provider-disable control for every measured A/B.
- Never compile the whole library for Cortex-A76. Only provider translation
  units receive `-mcpu=cortex-a76` or the minimum equivalent DotProd ISA flags,
  preserving the generic AArch64 and x86 builds.

## Tests to port and extend

No new vLLM semantic test exists for CPU instruction scheduling. Existing
Qwen3.5 model/parity tests remain the semantic spec. Port and extend the
matching llama.cpp quant/repack cases for any borrowed layout or kernel:

- exact portable-versus-provider results for integer dot/repack paths;
- existing tolerance contracts for floating reductions, without widening;
- M/N/K edge tiles, odd counts, short K, unaligned inputs and every tail;
- feature detection: A76+DotProd selects SDOT, A76 without DotProd and
  non-A76 DotProd take their declared fallbacks, absent i8mm never executes
  MMLA;
- deterministic results at 1/2/4 threads and concurrent submitter coverage;
- x86 and generic AArch64 builds, ASan/UBSan and thread sanitizer where
  supported;
- disassembly audit proving the optimized object contains only its declared
  ISA and assembly hot-loop invariants.

## Correctness and performance gates

1. x86-64 captures fixed prompt IDs, seeds, sampling parameters, 16 greedy
   tokens, selected logits and operation fixtures from the pinned vLLM oracle
   and the current CPU implementation.
2. The QEMU-built portable artifact loads the exact-hash GGUF on the Pi and
   matches 16/16 tokens before any optimized provider is enabled.
3. Every provider matches its portable operation contract and the Pi full
   model repeats the cross-architecture golden.
4. Microbench A/B uses randomized interleaved trials. A retained metric-level
   lever must improve its causal counter with no enclosing-scope regression;
   it stays non-default until combined evidence improves an enclosing scope.
5. Full-model Pi gates use at least three interleaved repetitions for prefill,
   decode, load time and peak memory at 1/2/4 threads. Same-file pinned
   llama.cpp is measured on the same idle Pi. Concurrent serving follows after
   the single-request gate.
6. Close an optimization family only after every inventoried lever category has
   a win/neutral/negative/blocked disposition and the reconciled model predicts
   less than 1% remaining end-to-end gain. A flat wall clock does not erase a
   causal metric win; combinations are re-tested after each bottleneck moves.

## Work breakdown

| W | Item | Entry gate | Exit gate |
|---|---|---|---|
| R0 | Spike, Pi inventory and fixed model recipe | observed hardware facts | this spec + current record surfaces |
| R1 | **CPU-GATED** general CPU kernel/PMU harness | existing quant bench | warning-clean build; JSON/CLI/timer/counter contract; 1/4-thread runs |
| R2 | **GREEN** QEMU-built portable Pi bring-up and x86 goldens | R0 | exact hash, load, 16/16 tokens, operation fixtures |
| R3 | **GREEN** Qwen trace + recursive scope profiling | R1-R2 | reached-loop inventory and binding baseline below |
| R4 | **GREEN** A76 C++/NEON/SDOT provider | R3 ranked evidence | exact operation/model output; 2.4x scalar kernel speedup |
| R5 | **GREEN with named T4 residual** A76 assembly candidate | R4 plateau + proven compiler gap | 3.66-5.08% binding kernel win; recursive model non-regressing |
| R6 | **llama.cpp measured, speed OPEN**; whole-system/thread/serving exhaustion | accepted R4/R5 stack | close measured 2.17x prefill / 1.53x decode gap, M1/T4 and concurrency; retain RSS win |

R1-R3 are the first implementation checkpoint. R4/R5 split into separate
kernel-row PRs if the changed code exceeds the one-row helper size cap; this PR
does not silently absorb unrelated kernel families.

## R2-R3 binding result

The Ubuntu 24.04 buildx/QEMU build completed with GCC 13.3 and its quantized
matmul smoke passed. The two exported AArch64 binaries were hash-gated before
deployment. On the physical Pi, the pinned model matched the x86 current-engine
golden 16/16 tokens (golden file SHA-256 `684f55a32355c0ccb6ce9c987273981f077b9591a46db07aea68561eb6432966`).
The four portable fixture arms also retained their exact checksums: decode
M=1 `0xd6aec014c0050fda` and prefill M=128 `0xa89baff1f3a4e360`, at one and
four threads.

The idle, unthrottled 2.4 GHz Pi baseline measured Q8_0 M=1/N=3072/K=2048 at
1,554,115 ns median (8.10 GFLOP/s) on one core and 742,585 ns (16.94 GFLOP/s)
on four. The M=128 arm measured 197,061,735 ns on one core and 49,890,756 ns
on four (3.95x scaling). The 16-token model arm measured TTFT 1,961.99 ms,
TPOT/ITL 366.91 ms and output throughput 2.14 tok/s. These are portable
baselines, not optimized results or llama.cpp parity claims.

A zero-loss, low-overhead `cycles:u` profile of a 64-token model run ranks the
reached loops: BF16 `Bt16Neon` 57.76%, portable `VecDotQ8_0Q8_0` 20.10%,
thread-ready 6.45% and `F16ToF32` 4.87%. The Q8 dot is selected for R4/R5:
the Pi has DotProd but the only existing Arm quant fast path requires i8mm, so
the R4-R5 checkpoint therefore compared portable, compiler-generated
exact-order SDOT and AAPCS64 assembly in one binary before changing dispatch.

## R4-R5 binding result

The A76 Q8 row is now `GATING`, with its requested assembly/compiler gap
proven. The locally QEMU-built binary passed 20/20 focused cases and 150,258
assertions. On the physical Pi, all operation checksums and all 64 Qwen tokens
match their portable/x86 goldens. The scheduled two-block AAPCS64 leaf beats
GCC's ACLE SDOT loop 3.66% on M=1/T1, 5.08% on M=128/T1 and 3.69% on
M=128/T4, while retiring about 10% fewer instructions. GCC emits a 48-byte
frame and two dependent SDOTs per block; the valid assembly hot path is a
stack-free leaf with two independent block chains.

The recursive model gate is non-regressing: assembly versus compiler SDOT has
1.55% lower median TTFT, neutral TPOT and 0.13% lower E2E. Against the
portable arm it lowers E2E 2.67%. `auto` selects the assembly only on
Cortex-A76+DotProd. M=1/T4 remains a measured negative at −2.43%, so R6 owns
that thread-partition interaction plus BF16 GEMM and concurrency. Full
provenance and raw hashes are in the
[R5 evidence](../../docs/bench-evidence/rpi5-a76-q8-dot-20260806.md).

## R6 same-file llama.cpp checkpoint

The Pi competitor floor is no longer unknown. Against a locally QEMU-built
official llama.cpp b9892 (`ee445f93d`), on the same Q8_K_XL bytes, four cores
and 17-input/64-output shape, vllm.cpp measures 12.81 tok/s prefill, 2.55 tok/s
decode and 2.46 tok/s output-equivalent E2E. llama.cpp measures 27.77, 3.91 and
3.77 tok/s: vllm.cpp ratios 0.461x / 0.653x / 0.653x. llama.cpp is therefore
2.17x faster in prefill and 1.53x faster in decode/E2E, well outside each
arm's sub-1% spread. vllm.cpp wins peak RSS at 2.841 vs 3.747 GiB, 24.2% less.
Both engines produce byte-identical normalized text for a same-prompt 64-token
greedy check.

R6 remains open on speed and concurrency, not memory or correctness. The prior
profile already puts BF16 `Bt16Neon` at 57.76% of the model run, so the next
recursive step is a fresh clean profile of both engines followed by the BF16
GEMM compiler/output audit; the M1/T4 Q8 partition remains a secondary leaf.
The exact b9892 reconstruction caveat, commands, samples and raw hashes are in
the [Pi competitor evidence](../../docs/bench-evidence/rpi5-a76-llamacpp-20260806.md).

## Risks and decisions

- `BACKEND-CPU` already passes a 20-core Arm i8mm Qwen3.5-2B single-stream
  llama.cpp floor. That evidence does not transfer to a four-core A76 without
  i8mm; the Pi now has its own measured open speed gate and never replaces the
  existing 20-core scoreboard.
- The Q8_K_XL file contains substantial f16 weights, so quant GEMM alone cannot
  be assumed dominant. A fresh trace ranks work before each lever.
- Reordered floating reductions may move near ties. Exact integer paths and the
  current floating tolerance/token rules remain binding; speed never widens a
  correctness threshold.
- PMU counter availability is verified, but permissions and event scheduling
  can still reject an event. The harness reports unsupported events and never
  fabricates zero counts.
- Artifact/model deployment into a disposable user-owned directory is approved.
  Source is never compiled on the Pi. Every temporary system setting is
  recorded and restored; no public service is started and vLLM is not installed
  on the Pi.
