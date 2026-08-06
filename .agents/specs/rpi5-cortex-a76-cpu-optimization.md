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
| CPU microbench | `examples/cpu_kernel_bench/main.cpp` | R1 vt-op/PMU substrate CPU-gated; Pi execution pending |
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
2. Pi portable build loads the exact-hash GGUF and matches 16/16 tokens before
   any optimized provider is enabled.
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
| R2 | Portable Pi bring-up and x86 goldens | R0 | exact hash, load, 16/16 tokens, operation fixtures |
| R3 | Qwen trace + recursive scope profiling | R1-R2 | complete reached-loop inventory and binding baseline |
| R4 | A76 C++/NEON/SDOT providers | R3 ranked evidence | op correctness + causal metric win + no enclosing regression |
| R5 | A76 assembly candidates | R4 plateau + proven compiler gap | ABI/disassembly/correctness + recursive A/B |
| R6 | Whole-system/thread/serving exhaustion | accepted R4/R5 stack | all lever dispositions, <1% residual model, llama.cpp floor |

R1-R3 are the first implementation checkpoint. R4/R5 split into separate
kernel-row PRs if the changed code exceeds the one-row helper size cap; this PR
does not silently absorb unrelated kernel families.

## Risks and decisions

- `BACKEND-CPU` already passes a 20-core Arm i8mm Qwen3.5-2B single-stream
  llama.cpp floor. That evidence does not transfer to a four-core A76 without
  i8mm; Pi numbers begin `PENDING` and never replace the existing scoreboard.
- The Q8_K_XL file contains substantial f16 weights, so quant GEMM alone cannot
  be assumed dominant. A fresh trace ranks work before each lever.
- Reordered floating reductions may move near ties. Exact integer paths and the
  current floating tolerance/token rules remain binding; speed never widens a
  correctness threshold.
- PMU counter availability is verified, but permissions and event scheduling
  can still reject an event. The harness reports unsupported events and never
  fabricates zero counts.
- Temporary governor/tool installation and model transfer were approved by the
  user. Every system setting is recorded and restored; no public service is
  started and vLLM is not installed on the Pi.
