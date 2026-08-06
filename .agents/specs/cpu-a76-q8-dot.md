# KERNEL-CPU-A76-Q8-DOT spike

Date: 2026-08-06

Row: `KERNEL-CPU-A76-Q8-DOT`

Claim: `CLAIM-KERNEL-CPU-A76-Q8-DOT`, stacked branch
`row/KERNEL-CPU-A76-Q8-DOT`

State: `SPIKE`

## Scope

Optimize the Q8_0 weight by Q8_0 activation dot used by
`vt::MatmulBTQuant` on Raspberry Pi 5's Cortex-A76. The row owns a portable
fallback, a compiler-generated Arm DotProd/SDOT implementation, a scheduled
AAPCS64 implementation, runtime dispatch, same-binary selection, focused tests,
disassembly evidence and recursive Qwen3.5-2B measurements.

In scope is `nrc == 1` and K divisible by the 32-element Q8_0 block. The
existing i8mm `nrc == 2` path, other quantization formats, BF16 GEMM, model
semantics, the threadpool and non-A76 scheduling are out of scope. DotProd
cores other than A76 may use the compiler implementation after correctness is
proven, but the assembly schedule is selected only for Cortex-A76.

## Upstream chain

- Pinned vLLM `555967922` supplies Qwen3.5 semantics, not this low-level CPU
  kernel. Its CPU platform entry is `vllm/platforms/cpu.py:42-125` and compiled
  ops begin at `csrc/cpu/torch_bindings.cpp:123-139`.
- The project llama.cpp pin `237ad9b96` owns the performance reference:
  `ggml/src/ggml-cpu/quants.c:400` is the portable Q8_0 dot and
  `ggml/src/ggml-cpu/arch/arm/quants.c:1076-1160` is its Arm NEON/DotProd
  implementation. The latter is design evidence, not a license to change the
  project's reduction contract.
- Runtime truth is the R3 physical-Pi trace: 241K `cycles:u` samples, zero
  loss, with portable `VecDotQ8_0Q8_0` at 20.10% and `F16ToF32` at 4.87%.
  The Pi reports ASIMDDP/DotProd and no i8mm, so the scalar dot is genuinely
  selected while the existing Arm fast tier is inert.

## Our baseline

- `src/vt/cpu/cpu_quant_dot.cpp:88-115` executes one scalar integer dot per
  block, converts two binary16 scales through `F16ToF32`, and accumulates one
  float contribution per block in strict order.
- `src/vt/cpu/cpu_quant_gemm.cpp:72-94` calls that function for every output
  element. `src/vt/cpu/cpu_quant_dot_arm.cpp:538-548` only offers i8mm MMLA,
  which the target lacks.
- `tests/vt/test_ops_quant_dot.cpp:559-585` ports llama.cpp's Q8 dot error case;
  the surrounding cases gate fixed reduction, ragged K, GEMM NMSE and
  thread-count determinism.
- `examples/cpu_kernel_bench/main.cpp` supplies exact Q8 fixtures, cache modes,
  1/2/4-core affinity and grouped PMU counters. The portable M=1/N=3072/K=2048
  denominator is 1,554,115 ns on one core and 742,585 ns on four, checksum
  `0xd6aec014c0050fda`.

## Port map

| Source | Local destination | Decision |
|---|---|---|
| llama portable `quants.c:400` | existing `cpu_quant_dot.cpp` | permanent fallback and exact oracle |
| llama Arm DotProd structure | new `cpu_quant_dot_sdot.cpp` | ACLE `vdotq_s32`, exact per-block float order |
| measured A76 schedule | new `cpu_quant_dot_a76.S` | original implementation, AAPCS64, no stack spill in core loop |
| Linux HWCAP/MIDR selection | new C++ TU plus `include/vt/quant.h` | ASIMDDP gate; A76-only assembly; env A/B control |
| quant GEMM dispatch | `cpu_quant_dot.cpp` | selected function replaces only Q8_0 row |
| benchmark variants | `examples/cpu_kernel_bench/main.cpp` | `portable`, `sdot`, `a76-asm`, `auto` |

The optimized implementations return exactly the portable function signature.
They perform both 16-byte SDOTs per Q8 block, horizontally add the four i32
lanes, then apply scales and update the scalar f32 accumulator in original
block order. Integer reassociation is exact; float block order is not changed.

## Tests to port

- Retain the upstream-derived `test-quantize-fns` Q8 dot error case already at
  `tests/vt/test_ops_quant_dot.cpp:559`.
- Extend its Q8 arm to invoke portable, C++ SDOT and assembly directly on
  random, all-zero, signed-extreme and scale-edge blocks; require bit identity
  for finite normal scale cases and explicit IEEE agreement for special scales.
- Gate K={32,64,96,2048}, unaligned-but-valid block bases, `nrc != 1`, ragged K,
  and null/unsupported dispatch behavior.
- Add feature-selection cases: no DotProd -> portable, DotProd non-A76 -> C++,
  A76+DotProd -> assembly, and environment overrides for every variant.
- Keep `MatmulBTQuant` bit-exact across 1/2/4 threads and run ASan/UBSan on the
  generic build. An x86 build must link only the stubs and retain current output.

No vLLM Python test specifies instruction scheduling. Existing Qwen3.5 GGUF
model tests and the pinned 16-token current-engine golden remain the semantic
e2e test.

## Gates

Correctness:

1. x86/generic AArch64 build and current quant suite remain green.
2. QEMU executes every explicit variant and preserves exact fixture checksums.
3. Physical Pi variants preserve both portable checksums and the 16-token x86
   golden. No threshold may be widened to admit a faster answer.

Compiler-gap and ABI:

1. Save `objdump -drC` for portable, C++ SDOT and assembly symbols. The C++
   object must prove SDOT issuance; assembly must prove the intended load/SDOT,
   scale-convert and accumulation schedule, no core-loop stack traffic, only
   caller-saved clobbers, and no instruction above Armv8.2-A+DotProd+FP16.
2. Compare C++ SDOT to assembly in the same binary. Assembly is accepted only
   if physical-Pi wall time and cycles improve outside run noise; beating only
   the scalar baseline is insufficient.

Performance:

1. Use randomized interleaved portable/C++/assembly fixture trials, at least
   seven samples per arm, on an idle unthrottled Pi at recorded frequency.
2. Record median/min wall time, cycles, instructions, IPC, frontend/backend
   stalls and last-level traffic for M=1 and M=128 at one and four threads.
3. Run at least three interleaved 64-token Qwen arms. Assembly must improve the
   enclosing model or be kept opt-in with its kernel-only win recorded. Exact
   output tokens, TTFT/TPOT/ITL, load time, RSS, temperature and throttling are
   recorded. QEMU timings are never evidence.

The final CPU-backend floor against same-file llama.cpp is a later campaign
gate; this row only closes the proven Q8 assembly/compiler gap.

## Dependencies

- `BACKEND-CPU` R2-R3 commit `16c848326`, the pinned Q8_K_XL model and golden.
- Local Docker buildx builder `pf-arm`; QEMU compiles/tests, never benchmarks.
- Disposable execution directory `rich@rpi5fan.lan:~/vllm-cpp-assembly`.
- Linux `getauxval(AT_HWCAP)` and Cortex-A76 implementer/part detection.
- No new library, generated binary or external source dependency. Any borrowed
  code would require a separate license/attribution review; the assembly is
  written for this row from the stated instruction/dataflow contract.

## Work breakdown

| W | Work | Exit |
|---|---|---|
| W0 | spike, row and claim | record checkers green, committed before code |
| W1 | selection seam + explicit benchmark variants | x86 fallback and QEMU smoke green |
| W2 | exact-order C++ SDOT | disassembly contains SDOT; fixture/model exact |
| W3 | scheduled AAPCS64 kernel | ABI/disassembly checks and direct tests green |
| W4 | interleaved Pi PMU A/B | assembly beats C++ in cycles and wall time |
| W5 | recursive Qwen A/B and checkpoint | no output/enclosing regression; docs current |

## Risks and decisions

- The compiler may already schedule the intrinsic loop optimally. In that case
  this assembly candidate is negative and the row remains open or selects a
  different proven compiler gap; an assembly file is not success by itself.
- Q8 blocks are 34 bytes, so scale and payload streams are not naturally
  16-byte aligned. Loads must remain unaligned-safe, and speculative reads may
  not cross the allocated final block.
- Hardware FP16 conversion must match `F16ToF32`, including signed zero,
  infinities and NaNs. If payload handling differs, retain software conversion
  for specials or dispatch those inputs to portable code.
- Unrolling may expose memory-level parallelism but increases register and I-
  cache pressure. PMU plus recursive model A/B, not instruction count alone,
  decides the retained schedule.
- The R3 model is BF16-GEMM dominated. A clear dot-kernel win may be diluted;
  it is still accepted as the requested demonstrated assembly win only when no
  enclosing regression exists, and remains opt-in until model wall time moves.
