# Cortex-A76 BF16 elementwise GEMM: close the Raspberry Pi 5 llama.cpp gap

**Row:** `KERNEL-GEMM-CPU-ELEM-A76` · **issue:**
[#284](https://github.com/mudler/vllm.cpp/issues/284) · **state:** `READY` ·
**parent:** `KERNEL-GEMM-CPU-ELEM` · **target:** Raspberry Pi 5 Cortex-A76
r4p1, four cores, AArch64 NEON + DotProd, no i8mm · **comparison source:**
official llama.cpp b9892 at `ee445f93d8a0a5033a46d1960e901ef5caec9a41`.

## Scope

The goal is to make the production C++/NEON CPU path for the same-file
Qwen3.5-2B Q8_K_XL workload faster than llama.cpp on the Raspberry Pi 5.
Prefill, decode and output-equivalent end-to-end throughput must all exceed
llama.cpp; latency must be lower and the existing peak-RSS win must not regress.

In scope:

- a fresh clean profile of both engines on the identical binding workload;
- a focused BF16 elementwise GEMM microbenchmark in
  `examples/cpu_kernel_bench/main.cpp` with production shapes, threads, PMU
  counters, cache state and same-binary variant control;
- Cortex-A76-specific C++/ACLE scheduling for the `[N,K]` BF16 path in
  `src/vt/cpu/cpu_matmul_elem.cpp`, including tile shape, activation-row
  sharing, load/convert/transpose ordering, prefetch distance, tail handling
  and caller thread partition;
- a separate A76 translation unit and runtime MIDR selection if Cortex-A76
  compile flags improve code without weakening the generic AArch64 build;
- recursive measurement at microkernel, enclosing `kMatmulBT`, prefill/decode
  phase and full-model levels after every retained candidate.

Out of scope:

- CUDA, Vulkan, model-forward, loader, quantization-format or C ABI changes;
- weakening the byte-exact elementwise GEMM contract, changing sampling or
  regenerating goldens;
- compiling on the Pi, installing vLLM there, or timing under QEMU;
- handwritten assembly until the explicit assembly gate below is satisfied;
- treating the already-faster peak RSS result as permission to trade memory
  for speed without measuring the new peak.

## Upstream chain

The behavioral oracle remains pinned vLLM `555967922`; llama.cpp is the speed
competitor for this CPU vehicle. The anchors below were verified in a fresh
official b9892 clone at exact commit `ee445f93d`; reconstruct that commit rather
than relying on a mutable local llama.cpp checkout.

| Component | Exact competitor anchor | What it establishes |
|---|---|---|
| BF16 dot | `ggml/src/ggml-cpu/vec.cpp:139-260` `ggml_vec_dot_bf16` | BF16 widening, independent accumulators and the scalar fallback |
| Chunk worker | `ggml/src/ggml-cpu/ggml-cpu.c:1158-1245` `ggml_compute_forward_mul_mat_one_chunk` | 16x16 outer blocking, row/column traversal and `vec_dot` call shape |
| Type dispatch | `ggml/src/ggml-cpu/ggml-cpu.c:390` | BF16 type selects `ggml_vec_dot_bf16` |
| SIMD vocabulary | `ggml/src/ggml-cpu/simd-mappings.h` | AArch64 load/convert/mul/add spellings used by the competitor build |

Before calling any competitor mechanism decisive, record its resolved build
flags and disassembly on the Pi artifact. Source inspection proposes a lever;
matched profiles and counters establish whether it ran.

## Our baseline and measured gap

Binding evidence is
[`docs/bench-evidence/rpi5-a76-llamacpp-20260806.md`](../../docs/bench-evidence/rpi5-a76-llamacpp-20260806.md).
Both engines use the same Qwen3.5-2B Q8_K_XL GGUF bytes, four A76 cores,
17 input tokens, 64 output tokens and three clean unthrottled repetitions.

| Axis | vllm.cpp | llama.cpp | Ratio / gap |
|---|---:|---:|---:|
| Prefill | 12.81 tok/s | 27.77 tok/s | **0.461x; 2.17x behind** |
| Decode | 2.55 tok/s | 3.91 tok/s | **0.653x; 1.53x behind** |
| Output-equivalent E2E | 2.46 tok/s | 3.77 tok/s | **0.653x; 1.53x behind** |
| E2E latency | 26,018.39 ms | 16,998.49 ms | **1.531x slower** |
| Peak RSS | 2.841 GiB | 3.747 GiB | **0.758x; 24.2% better** |

The prior clean model profile attributes 57.76% of wall time to BF16
`Bt16Neon`, 20.10% to the quantized Q8 dot, 6.45% to thread-ready/synchronism
and 4.87% to `F16ToF32`. That makes BF16 elementwise GEMM the largest measured
lever and prefill the furthest-behind binding metric. The attribution must be
reproduced from the fresh branch before changing the kernel.

Current local anchors:

- `src/vt/cpu/cpu_matmul_elem.cpp:152-186` — `Bt16Neon`, 16 output columns,
  four vector accumulators, four loads + a 4x4 transpose per output group;
- `src/vt/cpu/cpu_matmul_elem.cpp:194-232` — `BtM4Neon`, four activation rows
  sharing weight loads and transposes;
- `src/vt/cpu/cpu_matmul_elem.cpp:559-593` — Arm runtime tier selection;
- `src/vt/cpu/cpu_ops.cpp:106-276` — typed chunk worker and partitioning;
- `tests/vt/test_ops_matmul_elem.cpp` — byte-identity oracle across dtypes,
  layouts, shapes and thread counts;
- `examples/cpu_kernel_bench/main.cpp` — existing PMU-capable quant benchmark,
  which does not yet expose elementwise BF16 GEMM.

## Port map

| Competitor / reference | Local surface | Adaptation |
|---|---|---|
| llama.cpp `ggml_vec_dot_bf16` | `src/vt/cpu/cpu_matmul_elem.cpp` A76 BF16 microkernel | preserve its load-latency-hiding intent, but retain lane-per-output and strict K order instead of horizontal reduction |
| llama.cpp 16x16 `mul_mat` chunk worker | `src/vt/cpu/cpu_ops.cpp` typed elementwise chunk worker | measure its current 16-column and M-block partition before changing ownership or tile shape |
| llama.cpp AArch64 build flags / resolved disassembly | optional additive A76 translation unit + runtime MIDR selector | compile locally under QEMU; execute and count only on the Pi |
| llama.cpp benchmark and `perf` evidence | `examples/cpu_kernel_bench/main.cpp` + indexed Pi evidence | identical shapes, affinity and counters; QEMU timings are void |

## Design and candidate order

The correctness-preserving invariant is one SIMD lane per output column and
strictly sequential accumulation over K. Separate `vmulq_f32` then
`vaddq_f32` stays mandatory: FMA or K-reassociation is not an optimization for
this row because it changes the existing result contract.

Candidates are evaluated one at a time in this order:

1. **Measurement seam.** Add `--op matmul-bt-elem --dtype bf16` fixtures for
   real Qwen shapes and M in `{1,17,128}`, threads in `{1,4}`, hot/cold cache,
   plus named same-binary variants. Capture wall time, cycles, instructions,
   IPC, L1D/L2/refill events and migration/throttle state.
2. **Generated-code audit.** Compile the current NEON function for generic
   AArch64 and `-mcpu=cortex-a76`; compare instruction schedule, spills,
   register pressure, load-use distance, transpose sequence and unrolling.
3. **C++ schedule.** Interleave independent group loads/converts/transposes
   with arithmetic from prior groups; test group-major versus K-major order,
   explicit prefetch distances and MR in `{1,2,3,4}`. Preserve exact K order
   per lane and reject any spill-driven or model-level regression.
4. **Caller partition.** Measure output-tile ownership and false sharing at M=1
   and prefill M. Change chunking only if the enclosing-op counters demonstrate
   a synchronism or cache-locality defect.
5. **Optional packed view.** Consider a load-time/cacheable BF16 tile view only
   if counters prove transpose/load waste remains binding and its full model
   peak RSS stays no worse than llama.cpp and no worse than the accepted local
   baseline without an explicit result disposition.

Each candidate has a production-default arm and an exact same-binary rollback.
Rejected candidates are recorded with the smallest recursive level that
falsified them; they are not stacked into a mixed experiment.

## Assembly gate

Handwritten AArch64 assembly is forbidden while any binding C++ axis remains
at or below llama.cpp. It becomes eligible only after all of these are true:

1. the retained C++/NEON default is byte-exact at unit and model level;
2. vllm.cpp exceeds llama.cpp on prefill, decode and output-equivalent E2E
   throughput, has lower TTFT/E2E latency, and retains its memory floor;
3. the C++ result reproduces on an idle Pi in at least three interleaved
   same-binary repetitions with sub-3% spread;
4. disassembly plus PMU counters identify a remaining compiler scheduling,
   register-allocation or instruction-selection defect that source-level C++
   experiments failed to remove.

If assembly becomes eligible, it is a new work item within this same issue:
first capture the compiler body, write a byte-exact direct kernel mutation
test, implement the smallest AAPCS64 leaf, and compare compiler/assembly in the
same binary. A microbenchmark-only win never justifies selecting assembly.

## Tests to port and extend

| Source / guarantee | Local gate |
|---|---|
| llama.cpp `tests/test-backend-ops.cpp` BF16 MUL_MAT coverage | extend `tests/vt/test_ops_matmul_elem.cpp` only for any new A76 dispatch/tail/tile seam; retain `memcmp` byte identity |
| Existing scalar reference and full 16-bit conversion sweep | all old cases unchanged under `ref`, `portable`, `neon` and any new A76 tier |
| Variant selection | CLI/mechanism test proves each named benchmark arm selects the intended function; mutation of the selector must fail |
| Model correctness | the established 64-token Pi stream is byte-identical across rollback/default and matches the recorded x86/portable tokens |

## Gates

1. **Build/test locally, never on the Pi.** Use the repository-family buildx
   recipe (`pf-arm`, Ubuntu 24.04, GCC 13.3, `linux/arm64`) to build Release
   CPU-only artifacts under QEMU. Run the focused unit and CLI tests under QEMU.
2. **Direct correctness.** `test_ops_matmul_elem` passes byte-for-byte under
   every selectable tier. A deliberate wrong selector or altered accumulation
   order must turn the intended test red, then the exact source bytes are
   restored.
3. **Physical execution.** Copy only built artifacts and indexed evidence into
   a disposable directory on `rich@rpi5fan.lan`. Record SHA-256, CPU/MIDR,
   affinity, governor, frequency, temperature and `vcgencmd get_throttled`
   before/after. QEMU timing is void.
4. **Recursive performance.** Retain a change only when the focused kernel,
   enclosing `kMatmulBT`, phase-level prefill/decode and full model tell a
   coherent story. A lower-level win with an enclosing regression is rejected.
5. **Competitor floor.** Rebuild the exact llama.cpp b9892 artifact with the
   same QEMU toolchain and rerun it on the same Pi/file/workload. Success is
   strictly `>1.0x` throughput and `<1.0x` latency/memory ratio on every
   applicable binding axis; equality remains open.
6. **Regression.** The native CPU gate and project record gates pass. The
   kernel change must be inert on x86 and on non-A76 Arm selection.

## Dependencies

- `KERNEL-GEMM-CPU-ELEM` and `KERNEL-CPU-A76-Q8-DOT` are landed prerequisites.
- The exact GGUF is already present on the Pi; no new model download is owed.
- Docker buildx/QEMU is the compilation environment. The Pi is execution and
  PMU hardware only.
- Physical-Pi deployment is user-authorized for this disposable machine; do
  not change persistent services or system configuration.

## Work breakdown

| W | Deliverable | State |
|---|---|---|
| W0 | Refresh both-engine binding baseline/profile; validate artifact hashes and idle-state evidence | pending |
| W1 | Add elementwise BF16 microbench/PMU fixtures and selection mutation proof | pending |
| W2 | Audit current compiler output and evaluate C++/ACLE schedule/tile/prefetch candidates | pending |
| W3 | Recursively gate retained C++ default through full model vs llama.cpp | pending |
| W4 | Assembly eligibility decision from the explicit gate; implement only if eligible | blocked by W3 |

## Risks and stop conditions

- Qwen's mixed GGUF means an op-level BF16 gain may be Amdahl-limited. Stop a
  candidate when the enclosing phase falsifies it and re-profile before moving
  to a different family.
- Prefetch can improve an isolated hot loop while increasing shared-cache
  pressure. Both hot/cold microbench arms and full-model peak RSS are binding.
- A76 has 32 architectural SIMD registers, but MR=4 plus transient transpose
  vectors can still spill after compiler scheduling. Disassembly is evidence,
  not a post-hoc explanation.
- If the fresh profile no longer names BF16 GEMM as the largest actionable
  lever, stop before code, update this spec with the measured ranking, and
  create/claim the correct row rather than forcing this hypothesis.
- Network, SSH, PMU permission, thermal throttling or unavailable artifacts
  leave the corresponding gate `PENDING`; they never turn into assumed success.

## Outcome

Spec accepted after the inventory checker correctly rejected the new 52nd
kernel row while pinned to 51. The count ratchet moved only after the issue,
row and structured spec existed. Implementation and binding measurement remain
pending.
