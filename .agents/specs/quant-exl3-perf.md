# QUANT-EXL3-PERF — the EXL3 decode-throughput arm set, and the envelope that decides it

Row: `QUANT-EXL3-PERF`
Issues: [#2570](https://github.com/mudler/vllm.cpp/issues/2570) (primary)
Base SHA: `3d045ba1b`
Parent row: [`QUANT-EXL3`](quant-exl3-shared.md)
Sibling row: [`QUANT-EXL3-MUL1`](quant-exl3-mul1.md) — that row ports the FORMAT,
this one owns what it COSTS.
Matrix: [`.agents/quantization-matrix.md`](../quantization-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements
no EXL3** at the pin, so the kernels are mirrored from the registered secondary
oracle [`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (v1.4.3, MIT). The seam is vLLM's;
exllamav3 supplies the trellis kernels only.

## Now

`ACTIVE`. This row exists because no row owned EXL3 PERFORMANCE. `QUANT-EXL3`
owns the format and `QUANT-EXL3-MUL1` owns the `mul1` widths; both are
correctness rows and both say so. #2570 named a throughput gap with no owner,
and an unowned gap is one nobody reruns.

Slice A (the `(3, 2)` GEMV instantiation) and slice B (the `(4, 2)` GEMV KERNEL
PORT) are described below. Everything else is
under `## Evidence

**TWO FINDINGS CHANGE WHAT THIS ROW IS FOR, and they are stated here because
they are what a reader needs before any of the detail below means anything.**
Both are gated in `tests/vt/test_exl3_gemv.cpp` and derived in full under
`## THE WIDE CONFIG IS NOT THE ESCAPE`.

1. **The wide config is NOT the escape from the occupancy ceiling.** Slice A
   predicted it would be, because upstream admits CFG 1 at large `n` for
   `K == 4`. That band is `size_n >= 8192 && size_k <= 4096`, and the SMALLEST
   4-bit `k` in this checkpoint is 5120. It is dead for all 270 bits-4 modules,
   exactly as it is for the 137 bits-3 ones. **All 409 sit on the same
   `size_n / 32 <= narrow_coresident` test**, which is an occupancy query.
2. **What bits 4 buys is LOWER THRESHOLDS, not a second admitting branch.** Its
   270 modules spread over SIX values of `n` where the bits-3 137 have two, so
   the admission ladder is 32, 160, 192, 320, 384, 544 instead of 160 and 544.

3. **AND THE OCCUPANCY IS NOW MEASURED, WHICH SETTLES BOTH.** GB10 reports
   `SM_COUNT=48 MAX_THREADS_PER_SM=1536`, so the narrow config's 512-thread
   blocks ceiling `narrow_coresident` at `floor(1536/512) * 48 = **144**`. Every
   threshold above 144 declines. **34 of 409 modules are admitted — 0.75% of the
   trellis BYTES** — and the measured decode A/B is a null at 1.3% spread with
   `nsys` showing the arm never launches at the default. An earlier draft of
   this section claimed "164 of 409 at `narrow_coresident = 160`"; 160 is not
   reachable on this hardware and that claim is superseded by
   `## THE REASSESSMENT`.

So the row's open question is no longer "does an arm exist for these tensors" --
after slice B one exists for 407 of 409, the same 407 upstream takes -- nor even
"does the envelope admit them", which is now measured as almost never on GB10.
It is whether this arm set is worth its compile cost on a device whose
`MAX_THREADS_PER_SM` ceilings it out. `docs/benchmarks/qwen38-27b-exl3-gb10.md`
carries the same distinction on `main` as of `f8efa5761`.

### SLICE A, host arm, executed 2026-09-03 on `mudler-ubuntu-box`

CPU-only build (`RelWithDebInfo`, no CUDA toolchain on this host), built in
`/dev/shm` because `/` had 8.2 GB free. `BUILD_RC=0` in 432 s;
`ctest -R '^test_exl3_gemv$'` passed.

```
[doctest] test cases:  7 |  7 passed | 0 failed | 0 skipped
[doctest] assertions: 69 | 69 passed | 0 failed |
```

The device tier-3c case SKIPPED, once, and is counted above because it still
asserts on the way out (`CHECK_FALSE(OpRegistered(kExl3Gemm, kCUDA))`). A skip
that asserts nothing reports `assertions: 0`, which reads as a pass; this one
cannot. **So the tier-3c bound and the `(3,1)`-vs-`(3,2)` discrimination check
are UNMEASURED here and are not claimed.** They need the lease.

### Mutation table — slice A, host arm

Each row asserts four things, because three of them have faked a pass in this
tree before: the mutated file's `sha256` CHANGED, it COMPILED, the test binary's
`mtime` MOVED, and the tree RESTORED byte-for-byte.

| Mutation | file | sha256 | built | mtime moved | `TEST_RC` | verdict | restored |
|---|---|---|---|---|---|---|---|
| baseline | — | `db293772…` | — | — | 0 | 69/69 pass | — |
| M3 narrow admission boundary `<=` → `<` | `src/vt/exl3_policy.cpp` | `bd89702425…` | OK | YES | 1 | **RED**, 6 failed | YES |
| M4 `if (K == 3) return -1;` → `return 0;` | `src/vt/exl3_policy.cpp` | `f82c3ece4a…` | OK | YES | 1 | **RED**, 4 failed | YES |

After restore: build `rc=0`, file `sha256` matches the baseline, gate `rc=0`, and
the assertion counts are identical to the baseline line for line.

The two mutations bracket the boundary FROM BOTH SIDES, which is what a discrete
selection gate needs — its error is bimodal, not a tolerance. M3 reds the
ADMITTED side and M4 reds the DECLINED side, and the specific cases that fire
are this row's new ones:

```
M3  :169  Exl3GemvSelectConfig(bw, 1, 5120, 17408, 3, 2, 1, 544) == 0   -> got -1
M3  :171  Exl3GemvSelectConfig(bw, 1, 17408, 5120, 3, 2, 1, 160) == 0   -> got -1
M4  :168  Exl3GemvSelectConfig(bw, 1, 5120, 17408, 3, 2, 1, 543) == -1  -> got 0
M4  :170  Exl3GemvSelectConfig(bw, 1, 17408, 5120, 3, 2, 1, 159) == -1  -> got 0
```

M3 also reds the `(4, 2)` threshold rows, which is the point of recording them
before that arm exists: they are what the port will be measured against.

### SLICE A, device arm — `thor:gpu0`, executed 2026-09-03

NVIDIA Thor, compute capability **11.0**, driver **595.78**, nvcc 13.0, built
`sm_110` from the pinned bundle. `BUILD_RC=0`, so `GemvKernelForArm<3, 2>`
COMPILES; this was its first build anywhere.

```
GATE_RC=0
[doctest] test cases:  7 |  7 passed | 0 failed | 0 skipped
[doctest] assertions: 76 | 76 passed | 0 failed
device case SKIPPED count: 0        <- the device arm RAN; it did not skip
cb 1  tier 3c: relative RMS 5.16027e-04, worst elementwise 0.125
cb 2  tier 3c: relative RMS 5.27980e-04, worst elementwise 0.125
(3,1) vs (3,2) differ in 4095 of 4096 outputs
```

Against the tier-3c bound of `6.0e-3`, the new `(3, 2)` arm measures
`5.2798e-4` — an order of magnitude inside it, and within 2.3% of the
established `(3, 1)` arm's `5.16e-4` on the same fixture. The bound was not
touched.

The discrimination check is what makes those two numbers mean anything: the two
codebooks disagree on **4095 of 4096** outputs. A `cb` threaded wrongly between
them would have produced identical output and two green tolerances.

### Mutation table — slice A, device arm, `thor:gpu0`

| Mutation | sha256 | built | mtime moved | `TEST_RC` | verdict | restored |
|---|---|---|---|---|---|---|
| baseline | — | OK | — | 0 | 76/76 pass, 0 skipped | — |
| M1 `(3,2)` out of the predicate | `6c2dc9f5d5…` | OK | YES | 1 | **RED** | YES |
| M2 `cb == 2` → `GemvKernelForArm<3, 1>` | `283d10a5a8…` | OK | YES | 1 | **RED**, 3 failed | YES |
| after restore | matches baseline | OK | — | 0 | 76/76 pass, 0 failed | — |

**M1 IS THE REPO'S OWN doctest TRAP, CAUGHT.** Its summary line reads
`assertions: 71 | 71 passed | 0 failed` — a clean sweep — while `TEST_RC=1`,
because the case did not fail an assertion, it **THREW**:

```
test_exl3_gemv.cpp:227: ERROR: test case THREW exception:
vt cuda exl3: exl3_gemm was asked to force the m<=8 GEMV arm (force_gemv=1)
but the call is not hard-eligible for it: m=1 k=2048 n=4096 bits=3.
```

Anything reading that assertions line as the verdict would have recorded M1 as
GREEN and concluded the gate could not see the arm's removal. Only the exit code
caught it. That is why every row of this table carries `TEST_RC` and not a
summary line.

M1 is also the CONTROL the one-binary A/B rests on: with `(3, 2)` out of the
predicate the tree is the pre-change product exactly, and it refuses by name.

**M2 fails THREE independent assertions**, which is the confusable pair caught
from three directions at once:

```
:325  CHECK( rel <= 6.0e-3 )                     NOT correct   <- tier 3c
:326  CHECK( worst <= 64.0 * UlpF16(rms_ref) )   NOT correct   <- worst element
:352  CHECK( differing > size/2 )                NOT correct   <- discrimination
```

Worth stating precisely rather than flattering the design: on THIS fixture the
tolerance caught the wrong codebook by itself, so the discrimination check was
not the only line of defence here. It is still the one that GENERALISES. A
tolerance can only see a mis-threaded codebook when the two decodes happen to
diverge by more than the bound on the data at hand; the discrimination check
fails whenever they agree, which is the actual invariant — `(3, 1)` and `(3, 2)`
are different decodes of the same bits and must never produce the same numbers.
On the real fixture they differ in 4095 of 4096 outputs.

The job ends with `RESTORE build rc=0`, `cuda_exl3.cu matches baseline: YES`,
and `GATE AFTER RESTORE rc=0` at `assertions: 76 | 76 passed | 0 failed` — the
same counts as the pre-mutation baseline, line for line. So the two mutations
left no damage, and the green that opens this section is the same green that
closes it.

### THE OCCUPANCY TERM HAS A CEILING, AND IT IS BELOW WHAT THIS CHECKPOINT NEEDS

`DEVICE PROPS SM_COUNT=20 MAX_THREADS_PER_SM=1536 REGS_PER_SM=65536`.

The narrow config launches **512-thread** blocks, so
`blocks_per_sm <= floor(max_threads_per_sm / 512)` REGARDLESS of registers or
shared memory — a thread-budget ceiling no kernel tuning can lift. Therefore

```
narrow_coresident = blocks_per_sm * sm_count
                 <= floor(max_threads_per_sm / 512) * sm_count
```

| device | max_threads/SM | SMs | ceiling | n=17408 needs 544 | n=5120 needs 160 |
|---|---:|---:|---:|---|---|
| Thor sm_110 (MEASURED) | 1536 | 20 | **60** | CANNOT | CANNOT |
| GB10, if 1536 / 48 | 1536 | 48 | 144 | CANNOT | CANNOT |
| GB10, if 2048 / 48 | 2048 | 48 | 192 | **CANNOT** | can |

**CONFIRMED 2026-09-03.** GB10 measured `SM_COUNT=48 MAX_THREADS_PER_SM=1536`,
so the ceiling is `floor(1536/512) * 48 = 144`, and BOTH bits-3 shapes decline —
including the `n = 5120` one this table guessed "can" at 192. The row of this
table that was right is the mechanism; the row that was wrong is the guess about
GB10's threads per SM, which turned out to be 1536 and not 2048. See
`## Device throughput — SLICE A, MEASURED` and `## THE REASSESSMENT`.

**This was a PREDICTION, recorded before the `dgx:gpu0` job ran**, so it could be
falsified rather than fitted afterwards. The `n = 17408` shape — 92 of the 137
bits-3 modules, every `mlp.gate_proj` and `mlp.up_proj` — needs 544, which needs
roughly 136 SMs at 4 blocks each. No part in this fleet is close. So at mode 1
those 92 modules DECLINE on GB10 as certainly as they decline here, and the most
that can be admitted is the 45 `mlp.down_proj` modules at `n = 5120`, and only
if GB10 reaches 4 blocks/SM across at least 40 SMs. The dgx job prints
`SM_COUNT` and `MAX_THREADS_PER_SM`, which settles it.

If the A/B then reads `G1 == G0`, that is not a mystery and not a ceiling: it is
this table. **The next traceable hypothesis is the WIDE config (CFG 1), not the
kernel.** The narrow grid is one block per 32 output columns, which is what
makes a large-`n` shape need a co-resident wave these parts cannot supply;
upstream's envelope admits the wide config at large-`n`/small-`k` only for
`K == 4` (`exl3_gemv.cu:69`). That is the `(4, 2)` port, which is already this
row's largest `## Owed` item — so the measurement and the owed work point at the
same place, from two directions.

### SLICE B, host arm, executed 2026-09-03 on `mudler-ubuntu-box`

CPU-only build (`RelWithDebInfo`, no CUDA toolchain on this host), built in
`/dev/shm` because `/` had 8.2 GB free. `BUILD_RC=0`; `test_exl3_gemv` and
`test_exl3_gemm` both `rc=0`.

```
test_exl3_gemv  [doctest] test cases:   7 |   7 passed | 0 failed | 0 skipped
                [doctest] assertions: 158 | 158 passed | 0 failed
test_exl3_gemm  [doctest] test cases:  15 |  15 passed | 0 failed | 0 skipped
                [doctest] assertions: 210 | 210 passed | 0 failed
python3 scripts/check-pr-size.py --base origin/main --head HEAD  ->  rc 0
```

158 against slice A's 69, and 89 of the difference is the bits-4 envelope table.

`scripts/agent-preflight.sh` exits 0 while printing `NOT a green preflight`, so
the output is read and not the exit code. **Zero gates FAILED.** Five SKIPPED,
every one for `needs arguments preflight does not supply`, and each is chased by
hand rather than left as an unknown:

| Skipped gate | run by hand | result |
|---|---|---|
| `check-pr-size.py` | `--base origin/main --head HEAD` | OK, every path class within budget |
| `check-cpu-isa-build.py` | `--compile-commands` from the local CPU build | OK |
| `check-arm-isa-build.py` | same | FAILS on its own PRECONDITION, not on this change: it is fed an x86 build, where `cpu_quant_dot_sdot.cpp`, `cpu_quant_dot_arm.cpp` and `cpu_quant_repack_arm.cpp` carry no `-march=armv8.2-a` flags because they are not built. None of the three is in this change. It needs an Arm build |
| `check-cuda-fat-gencode.py` | not runnable here | needs `--library` or `--cuobjdump-list` as well, which needs a CUDA build this host cannot produce |
| `check-triton-aot-multiarch.py` | not runnable here | needs `--vendored-root` |

`check-tree-compiles` reports `1 of 1 translation unit(s) in scope compiled`,
which is the second half of the point above: the one TU is the test file. The
CUDA TU is not in scope on a host with no CUDA toolchain.

**NOTHING ON THIS HOST COMPILES `cuda_exl3.cu` AT ALL.** There is no CUDA
toolchain here, so the file is not in the CPU build at any warning level. Every
claim about the ported kernel — that it compiles, that it meets tier 3c, that
the wide config works, that the arm is reached unforced — is a DEVICE claim and
is PENDING until the lease runs. The host arm gates the envelope and nothing
else, and the count above must not be read as covering the port.

### Mutation table — slice B, host arm

| Mutation | file | sha256 | built | mtime moved | `TEST_RC` | verdict | restored |
|---|---|---|---|---|---|---|---|
| baseline | — | `db293772…` | — | — | 0 | 158/158 pass | — |
| H1 narrow admission `<=` → `<` | `src/vt/exl3_policy.cpp` | `bd89702425…` | OK | YES | 1 | **RED**, 11 failed | YES |
| H2 `if (K == 3)` → `if (K == 3 \|\| K == 4)` | `src/vt/exl3_policy.cpp` | `d57f810bccbb…` | OK | YES | 1 | **RED**, 3 failed | YES |

Each row asserts four things, because three of them have faked a pass in this
tree before: the mutated file's `sha256` CHANGED, it COMPILED, the test binary's
`mtime` MOVED, and the tree RESTORED byte-for-byte.

H1 reds the ADMITTED side of every threshold including the eight new bits-4
rows (`:204`). H2 is the bits-4-specific one: it makes `K == 4` decline where
`K == 3` does, and the three cases that fire are exactly the three that read the
WIDE band — `:112`, `:114` and `:222`. That is the point worth having in an
evidence table rather than in prose: the wide band is genuinely gated, and it is
also genuinely unreachable at this checkpoint's shapes, and those are different
statements.

**What the host mutations do NOT cover, stated rather than implied.** The
`s.n / 32 == s.threshold` line inside the bits-4 loop is arithmetic the envelope
also performs, so it gates a typo in the module table and not the envelope. The
gate on the envelope is the two-sided threshold assertion beside it, and H1 is
the evidence that it bites.

### The arm labels had to be TYPED, and the trap is executable in this tree

The four-leg device case is parameterised, so each `MESSAGE` carries the arm's
label — and that label is what the lease's evidence gets read by. The first cut
declared it `const char*`. Under doctest 2.5.2 that decays to BOOL:

```
$ g++ -std=c++20 -Ithird_party dtprobe.cpp -o dtprobe && ./dtprobe -s
dtprobe.cpp:7: MESSAGE: RAW   -> 1
dtprobe.cpp:8: MESSAGE: TYPED -> (4,2) wide
  logged: raw := 1
          typed := (4,2) wide
```

`raw` and `typed` hold the same characters. A string LITERAL streams correctly,
which is why slice A's `MESSAGE("cb ", codebook, …)` was fine and hid the
problem; a decayed pointer does not. Four tier-3c numbers labelled `1`, `1`, `1`
and `1` would have been attributed by assumed loop order, which is how three
measured values were once rotated across three axes in this repository. The
field is `std::string`, and the probe above is why rather than a precaution.

### SLICE B, device arm — `thor:gpu0`, executed 2026-09-03. GREEN.

NVIDIA Thor, compute capability **11.0**, driver **595.78**, nvcc 13.0, built
`sm_110` from the pinned bundle `c67a…`/`b1cf294a2…` at commit `b1bad1a5a`.
`BUILD_RC=0`, so **the ported bits-4 kernel COMPILES** — its first build
anywhere. `SM_COUNT=20 MAX_THREADS_PER_SM=1536 REGS_PER_SM=65536
SMEM_PER_SM=233472`.

```
GATE[default] RC=0        test cases: 7 | 7 passed   assertions: 184 | 184 passed | 0 failed
device case SKIPPED count: 0        <- the device arm RAN
(3,1) narrow  rel RMS 5.16027e-04  worst 0.125    vs cb 2 reference: 1.29746
(3,2) narrow  rel RMS 5.27980e-04  worst 0.125    vs cb 1 reference: 1.60119
(4,2) narrow  rel RMS 5.09719e-04  worst 0.0625   vs cb 1 reference: 1.59095
(4,2) wide    rel RMS 4.54155e-04  worst 0.0625   vs cb 1 reference: 1.60119
(3,1) vs (3,2) differ in 4095 of 4096 outputs
GATE[smem=1]  RC=0, identical numbers        <- the SMEM_STAGE arm too
```

**The new arm is the most accurate of the four**, at `4.5e-4`–`5.1e-4` against
the tier-3c bound of `6.0e-3`, an order of magnitude inside it, and its worst
elementwise error is HALF the bits-3 arms' — which is what a wider codeword
should do. The bound was not touched. The two pre-existing arms report exactly
the numbers slice A recorded, so the port disturbed neither.

`rel_sib` is the discrimination check: every arm sits ~1.3–1.6 relative RMS away
from the SAME width decoded with the other codebook, against `> 0.6` required.
That is the check that generalises to `(4, 2)`, whose confusable partner
`(4, 1)` is deliberately not compiled.

**REACHED UNFORCED, all three narrow arms**: `4096 of 4096 outputs byte-equal to
the forced launch`, with `force_gemv` left at its `-1` default. The arm is on the
production path, not only on the test's.

### Mutation table — slice B, device arm, `thor:gpu0`

| Mutation | sha256 | built | mtime moved | `TEST_RC` | verdict | restored |
|---|---|---|---|---|---|---|
| baseline | — | OK | — | 0 | 184/184, 0 skipped | — |
| M1 `(4,2)` out of the predicate | `02f7154bd7c3…` | OK | YES | 1 | **RED** — and it THREW | YES |
| M2 `GemvKernelForArm<4, 2>` → `<4, 1>` | `c599b7883aa4…` | OK | YES | 1 | **RED**, 6 failed | YES |
| M3 24-lane guard applied at bits 4 | `5417211c13ea…` | OK | YES | 1 | **RED**, 4 failed | YES |
| M4 bits-4 window pair reversed | `96792eff7ff1…` | OK | YES | 1 | **RED**, 4 failed | YES |
| M5 `LSTRIDE` back to literal 24 | `8674798a3bec…` | **compile error** | — | — | **CAUGHT BY `static_assert`** | YES |
| M6 production call site deleted | `69d81746e773…` | OK | YES | 1 | **RED**, exactly 3 failed | YES |
| after restore | matches baseline | OK | — | 0 | 184/184, 0 failed | — |

Every expectation in this table was WRITTEN DOWN BEFORE THE RUN, in this file,
and each is what happened.

**M1 is the doctest trap caught again.** Its summary reads
`assertions: 171 | 171 passed | 0 failed` — a clean sweep — while `TEST_RC=1`,
because the case did not fail an assertion, it THREW:
`exl3_gemm was asked to force the m<=8 GEMV arm (force_gemv=1) but the call is
not hard-eligible for it: m=1 k=2048 n=4096 bits=4`. Reading the assertions line
would have recorded M1 as green.

**M5 failed to COMPILE, and that is the stronger result, not a void mutation.**
The first error is
`cuda_exl3.cu(1385): error: static assertion failed with "exl3 gemv: one warp
load must cover one tile"` — the invariant slice B added, naming itself. A wrong
`LSTRIDE` cannot reach a test run.

**M6 IS THE REACHABILITY PROOF, and its shape is the evidence.** Deleting the
`Exl3GemvTryLaunch` call site reds **exactly three** assertions, and all three
are `CHECK( same == got.size() )` — the unforced legs, one per narrow arm. Every
forced assertion stayed GREEN. So the suite distinguishes "the kernel works"
from "anything reaches it", which is what a gate that only forced the arm could
never do.

### Device throughput — SLICE A, MEASURED on `dgx:gpu0` 2026-09-03: a NULL, with the envelope declining rather than the kernel failing to help

NVIDIA **GB10**, cc **12.1**, driver **580.173.02**, `sm_121a`, commit
`665167c4a`. Real HumanEval, ShareGPT-shaped, `num_prompts=32 output_len=128
temperature=0.6 seed=0 concurrency=1`, `VT_DFLASH_PAGED=0` on every leg alike.
Axis is `Mean per-stream decode rate`, decode only. Three rounds INTERLEAVED on
ONE binary.

| leg | round 1 | round 2 | round 3 | mean |
|---|---|---|---|---|
| `VT_EXL3_GEMV=0` (arm off) | 17.20 | 17.18 | 17.09 | 17.157 |
| `VT_EXL3_GEMV=1` (default, the only production claim) | 17.22 | 17.12 | 17.09 | 17.143 |
| `VT_EXL3_GEMV=2` (diagnostic, never a production number) | 17.02 | 17.05 | 16.99 | 17.020 |

**G1 == G0 to 0.08%.** Total spread across all nine legs is 1.3%. There is no
separation between the arms.

**THE PREDICTION THIS ROW REGISTERED BEFORE THE RUN IS CONFIRMED.**
`## THE OCCUPANCY TERM HAS A CEILING` predicted
`narrow_coresident <= floor(max_threads_per_sm / 512) * sm_count` and said the
dgx job's `SM_COUNT` and `MAX_THREADS_PER_SM` would settle it. Measured:
`SM_COUNT=48 MAX_THREADS_PER_SM=1536`, so the ceiling is
`floor(1536/512) * 48 = 3 * 48 = **144**`. Against the **160** and **544** this
checkpoint's two bits-3 shapes need, **both decline** — including the
`down_proj` shape at 160 that the prediction table listed as the one that might
be admitted. It is not. **No bits-3 module of this artifact takes the arm on
GB10.**

**nsys separates a decline from an ineffective kernel, which is the whole reason
that leg exists:**

```
mode=0   exl3_gemv_kernel rows = 0    exl3_gemm_kernel rows = 7
mode=1   exl3_gemv_kernel rows = 0    exl3_gemm_kernel rows = 7
mode=2   exl3_gemv_kernel rows = 4    exl3_gemm_kernel rows = 5
         exl3_gemv_kernel<(int)3, (bool)1, (int)2, (int)0, (int)1, (bool)0>
```

At the default the GEMV **never launches**. Forced, it launches and decodes
correctly — the template arguments read `bits=3, c_fp32=true, cb=2, MMODE=0,
CFG=1, SMEM_STAGE=false`, so it is the `(3, 2)` arm, in the WIDE config, which
is what mode 2 selects for the `n = 17408` shape. So `G1 == G0` is this row's
table and not a mystery, and the arm is DECLINED rather than ineffective. Those
are different findings and only the trace separates them.

**A NULL IS THE RESULT, AND IT IS NOT A CEILING.** The instantiation is correct,
upstream-faithful, device-gated at tier 3c on two boxes, and worth zero end to
end on GB10 — because upstream's own envelope, ported verbatim, declines every
shape this checkpoint has at this occupancy.

Read `G2` with care and do not quote it as a production number: it is
consistently the slowest leg, in all three rounds (17.02 < 17.20, 17.05 < 17.18,
16.99 < 17.09). Three of three in one direction is `p = 0.125` under a sign
test, so it is NOT evidence, and the effect is inside the 1.3% spread. It is at
most a weak hint that the GEMV would not have won on these shapes anyway, which
is exactly what upstream's envelope asserts by declining them.

**ONE LEG IS UNMEASURED AND IS NOT REPORTED AS A NULL.** The job printed
`DRAFT LEGS SKIPPED: draft absent at /tmp/q38bench/draft or past the time guard.
The m == 8 end of the envelope is therefore UNMEASURED, not measured-as-null.`
Every number above is `m == 1`, `MMODE == 0`. The speculative-decode arm drives
`m` up to 8 and takes `MMODE == 1` with row-guarded fragment loads — a different
compiled kernel and a different point on the envelope. Nothing here measures it.

### Device throughput for `(4, 2)` — NOT MEASURED, and see the reassessment

Queued on `dgx:gpu0` behind other work and did not run. **No throughput number
is claimed for `(4, 2)`.** The reassessment below says what the slice A
measurement already implies about it, which is a different thing from a
measurement of it.


## THE REASSESSMENT: what `narrow_coresident = 144` does to `(4, 2)`

**This section restates a number this row previously published, because the
measurement moved it by a factor of forty.** `## THE WIDE CONFIG IS NOT THE
ESCAPE` said the bits-4 ladder reaches "164 of 409 modules at
`narrow_coresident = 160`". **160 IS NOT REACHABLE ON GB10.** The measured
ceiling is 144, so that sentence describes a device this fleet does not have.

The admission rule is `size_n / 32 <= narrow_coresident`, and on GB10
`narrow_coresident = blocks_per_sm * 48` with `blocks_per_sm <= 3`:

| `blocks_per_sm` | `narrow_coresident` | largest admitted `n` |
|---:|---:|---:|
| 1 | 48 | 1536 |
| 2 | 96 | 3072 |
| 3 (the ceiling) | 144 | 4608 |

**The verdict is the same across that whole range**, which is what makes it
robust rather than a knife edge: the only shape in the checkpoint with
`n <= 4608` is `n = 1024`, and it is admitted even at one block per SM
(`48 >= 32`); the next rung up is `n = 5120` at 160, which even three blocks per
SM cannot reach. So `blocks_per_sm` need not be measured directly to settle
this, and the empty `GEMV RES-USAGE` line in the dgx log costs nothing.

| | modules | share of modules | share of trellis BYTES |
|---|---:|---:|---:|
| admitted at `nc = 144` (MEASURED) | **34 of 409** | 8.3% | **0.75%** |
| admitted at `nc = 160` (this row's earlier claim) | 164 of 409 | 40.1% | 29.8% |

The byte column is the one that decides it, because decode at `m == 1` is
weight-bandwidth-bound and the 34 admitted modules are the SMALLEST in the
artifact: `k = 5120, n = 1024`, 2.6 MB each against 33–45 MB for the projections
that dominate. 34 of them is 89 MB of an 11.0 GiB trellis.

**So on GB10, `(4, 2)` moves 0.75% of the weight traffic to a different kernel,
and the measurement that would detect it has a 1.3% spread.** It cannot show. A
throughput case for landing `(4, 2)` on this hardware does not exist, and this
row will not manufacture one by running the A/B until a favourable draw appears.

**What that does and does not mean:**

- It is NOT a reason to call the port wrong. It is device-gated GREEN on Thor at
  `4.5e-4`–`5.1e-4` against a `6.0e-3` bound in both configs and both smem
  modes, it is upstream-faithful line for line, its production route is proven
  by M6, and six mutations red exactly as pre-registered.
- It is NOT a ceiling claim. The binding term is `MAX_THREADS_PER_SM = 1536`
  against a 512-thread block. A Blackwell part at 2048 threads/SM reaches
  `4 * 48 = 192`, which admits `n <= 6144` — 167 bits-4 modules and 45 bits-3
  ones, 212 of 409. The arm set is not useless; GB10 is the wrong device for it.
- The next traceable hypothesis is therefore the BLOCK SIZE, not the kernel and
  not the width. `narrow_coresident` is ceilinged by `floor(1536/512) = 3`
  purely because CFG 0 launches 512 threads. Upstream's CFG 1 launches 256 and
  would ceiling at 6, but its admission band needs `size_k <= 4096` and this
  checkpoint's smallest 4-bit `k` is 5120, so it is unreachable here — that is
  slice B's first finding, and it is what closes the easy escape. Whether a
  256-thread NARROW variant is admissible is an upstream question this row has
  not asked, and inventing one would be a structure upstream does not have.
- **`m == 8` is still unmeasured.** The draft legs did not run. Everything above
  is `m == 1` / `MMODE == 0`, and the speculative arm compiles a different
  kernel at a different point on the envelope.

**RECOMMENDATION, stated plainly because the coordinator has to decide it.**
Land `(4, 2)` for coverage and correctness, or do not land it, on the cost of 16
more kernels in a fat build for ten architectures — but do not land it for
throughput on GB10, and do not let any record imply that it buys any. The
measured position is 0.75% of weight bytes under a 1.3% spread. If the answer
is "not worth the compile cost today", that is a defensible reading of this
table and the arm should be recorded as ported-and-parked with this section as
the reason, not quietly dropped.

## Owed`, itemised, with the reason it is not closed here.

**No throughput number is claimed by this row yet.** The arm is instantiated and
its numeric gate and A/B are QUEUED on `dgx:gpu0` behind other work. Until that
lease runs, every device claim here is PENDING and is reported as PENDING; a
queued job nobody could gate is a partial result and never a pass.

## The gap, as #2570 states it

Our `m <= 8` EXL3 GEMV instantiated exactly one arm, `(bits = 3, cb = 1)`.
`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` — the #2495 benchmark checkpoint — contains
**zero** tensors of that arm, so the small-m fast path was unreachable on the
model the benchmark is about, one arm at a time and silently, because a declined
GEMV falls through to the regular block-pipelined kernel rather than refusing.

Upstream instantiates seven pairs at the pin
(`exllamav3/exllamav3_ext/quant/exl3_gemv.cu:83-86`):

```c
SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
```

## The census, recomputed for this row rather than inherited

Read from the LOCAL safetensors headers of
`/mnt/nas_share/rc/qwen38-exl3-bench/ckpt/target-3.5bpw`, both shards, on
2026-09-02. 2426 tensors, of which **409 are `.trellis`**, **409 carry a `mul1`
marker and zero carry `mcg`** — so every quantized linear in the artifact is
codebook 2, and `LinearEXL3`'s presence rule (`exl3.py:74-77`) has no other
answer for it.

Bit widths are read from the trellis shape's third axis (`16 * bits`), and every
shape is 128-aligned on both `k` and `n`, so none is excluded by the GEMV's
`size_k % 128 || size_n % 128` hard test:

| bits | k | n | modules | which |
|---:|---:|---:|---:|---|
| 3 | 5120 | 17408 | 92 | `mlp.gate_proj`, `mlp.up_proj` |
| 3 | 17408 | 5120 | 45 | `mlp.down_proj` |
| 4 | 5120 | 1024 | 34 | narrow attention/GDN projections |
| 4 | 5120 | 6144 | 48 | |
| 4 | 5120 | 10240 | 48 | |
| 4 | 5120 | 12288 | 17 | |
| 4 | 5120 | 17408 | 38 | |
| 4 | 6144 | 5120 | 64 | |
| 4 | 10240 | 5120 | 1 | |
| 4 | 17408 | 5120 | 20 | |
| 5 | 6144 | 5120 | 1 | |
| 6 | 5120 | 248320 | 1 | `lm_head` |

Totals `{3: 137, 4: 270, 5: 1, 6: 1}` = 409. This reproduces the count
`QUANT-EXL3-MUL1` slice F recorded, from the artifact and not from a summary of
it, which is the discipline that row's own lesson demands.

## THE INSTANTIATION IS NOT THE ONLY GATE, AND ON GB10 IT IS NOT THE BINDING ONE

This is the finding that shapes the whole row, and #2570 does not contain it.

`Exl3GemvSelectConfig` (`src/vt/exl3_policy.cpp:154-179`) is upstream's
`exl3_gemv_cfg` (`exl3_gemv.cu:46-72`) verbatim, and it returns `-1` to DECLINE.
Read it at the shapes above, on Blackwell (GB10 is `Exl3Cc::kBlackwell`), at the
default `mode == 1`:

```
if (K == 2)                      -> config chosen        (no bits-2 tensor here)
if (K == 3 && cc == kAda)        -> config chosen        (GB10 is NOT Ada)
if (size_n / 32 <= narrow_coresident) -> 0
if (size_k <= 2048 && size_n <= 8192) -> 0               (min k here is 5120)
if (K == 3)                      -> -1                   <-- every bits-3 shape
if (size_n >= 8192 && size_k <= 4096) -> 1               (min k here is 5120)
if (... && cc == kAmpere)        -> 1                     (GB10 is NOT Ampere)
                                 -> -1
```

`narrow_coresident` is `GemvOccupancy(narrow kernel, 512 threads) * num_sms`,
the only term a device supplies. Every other term is fixed by the shape. So on
GB10 the arm is admitted at the default mode **only** where
`n / 32 <= blocks_per_sm * num_sms`:

| bits | k | n | n/32 | admitted at mode 1? |
|---:|---:|---:|---:|---|
| 3 | 5120 | 17408 | 544 | only if `narrow_coresident >= 544` |
| 3 | 17408 | 5120 | 160 | only if `narrow_coresident >= 160` |
| 4 | 5120 | 1024 | 32 | almost certainly YES |
| 4 | 6144 | 5120 | 160 | only if `narrow_coresident >= 160` |
| 4 | 5120 | 6144 | 192 | only if `narrow_coresident >= 192` |
| 4 | 5120 | 10240 | 320 | only if `narrow_coresident >= 320` |
| 4 | 5120 | 12288 | 384 | only if `narrow_coresident >= 384` |
| 4 | 5120 | 17408 | 544 | only if `narrow_coresident >= 544` |

An occupancy of one or two 512-thread blocks per SM on a device with a few dozen
SMs puts `narrow_coresident` in the low hundreds. **The instantiation is
therefore necessary and possibly not sufficient**, and a measurement that only
compares "arm present" against "arm absent" at the default mode cannot tell the
two apart: a zero would be indistinguishable from an envelope that never
admitted the arm.

That confusion has already cost this tree one published number. The comment at
`cuda_exl3.cu:2086` records a `VT_EXL3_GEMV=1` vs `=0` A/B reported as an 8%
GEMV effect **when neither arm could take the GEMV at all** — it ran the same
path twice. This row does not repeat it.

So the measurement carries THREE legs, not two:

- `VT_EXL3_GEMV=0` — the arm is off. On this checkpoint this is byte-identical
  in behaviour to the pre-change binary, because the pre-change predicate
  admitted only `(3, 1)` and the artifact has zero `(3, 1)` tensors. That
  equivalence is asserted by a fourth leg on the PRE-CHANGE binary, not assumed.
- `VT_EXL3_GEMV=1` — the default. Upstream's measured envelope decides, and this
  is the only leg that is a production claim.
- `VT_EXL3_GEMV=2` — upstream's own "use wherever the hard constraints allow"
  testing mode (`exl3_gemv.cu:22`). This leg separates "the envelope declined"
  from "the arm ran and did not help". It is a DIAGNOSTIC and is never reported
  as a production number.

`narrow_coresident` itself is printed from the device, so the table above stops
being a prediction.

**WHAT SEPARATES G0 FROM G1 WHEN THE ENVELOPE DECLINES, exactly.** Read
`Exl3GemvTryLaunch` in order: `force_gemv == 0`, then the arm predicate, then
`Exl3GemvHardEligible`, then the mode, then `GemvKernel`, then `GemvOccupancy`,
then `Exl3GemvSelectConfig`. `VT_EXL3_GEMV=0` returns at the MODE test, which is
before the occupancy query. So if the envelope declines at mode 1, the whole
measured difference between G0 and G1 is one
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` per (device, kernel) pair,
memoised in a static map. That is why `G1 == G0` would be the EXPECTED reading
under a declining envelope rather than a surprise, and why G2 is needed to say
anything about the kernel itself.

The same order is what makes the pre-change binary and `VT_EXL3_GEMV=0`
equivalent on this checkpoint: the old predicate returned false one test EARLIER
than the mode test does, and neither reaches a device call. Both fall through to
`exl3_gemm_kernel` with identical arguments.

## Which tree the device evidence is measured on

The lease job is pinned to the source tarball `sha256
c133a4cbba670dbef6c6daebf4c0e130824519f317af221eaff7ebb1e9e9c5dd`, which is
commit `665167c4a` of `row/QUANT-EXL3-PERF`.

**THIS SENTENCE USED TO CLAIM MORE, AND A MERGE FALSIFIED IT.** It read "every
compiled and tested path is byte-identical between that commit and this row's
head; only `docs/USAGE.md` moved afterwards". That was true when written and
became false when `origin/main` was merged into this branch before landing: 25
paths under `src/`, `include/`, `tests/` and `CMakeLists.txt` differ between
`665167c4a` and the merged head, none of them this row's. The gates stayed green
throughout, which is exactly why the claim had to be re-checked rather than
assumed -- a merge can leave the code correct and the PROSE wrong, and only the
prose is load-bearing for an evidence table.

What is true, and what the evidence actually rests on: **the five files this
row's measurements depend on are byte-identical** between both pins and the
merged head --- `src/vt/cuda/cuda_exl3.cu`, `src/vt/exl3_policy.cpp`,
`include/vt/ops.h`, `tests/vt/test_exl3_gemv.cpp` and
`tests/vt/exl3_fixture.h`. So the device numbers below still name this tree's
kernel, its envelope and its fixture. Anything measured on the pinned trees is a
statement about THOSE trees; re-derive it after any change to those five
files.

The DEVICE NUMERIC GATE runs on `thor:gpu0` from a `git bundle` pinned to
`sha256 8156c0dbe30092ac267ab1749d2386c37ea832d7a2ceed5f3db6d29504d9ebf7`,
which is commit `57e359907`, and the job aborts unless `git rev-parse HEAD`
inside the clone matches. Nothing is pushed to reach that box.

Named here because an evidence table must say which tree produced it, and
because the bundle and tarball hashes are the only identifiers the job can
verify on the box: each aborts rather than build a tree it was not pinned to. A
`git rev-parse` plus an empty porcelain proves nothing on the far side of a
copy.

**Why two boxes.** `Exl3CcFromSm` maps every `sm_major >= 10` to
`Exl3Cc::kBlackwell`, so Thor (11.0) and GB10 (12.1) are the SAME envelope
class and only `narrow_coresident` differs by SM count. That makes Thor a valid
CORRECTNESS box for the tier-3c bound and the codebook discrimination, and it is
NOT a valid throughput box: the 16.7 tok/s baseline is GB10's, and a decode
number from a different part is not comparable to it. The throughput half stays
on `dgx:gpu0` and nothing from Thor is quoted on that axis.

## Scope

**In scope, slice A:** instantiate the GEMV at `(bits = 3, cb = 2)`; extend the
device numeric case to gate both codebooks at tier 3c; measure the end-to-end
decode effect on the real checkpoint on `dgx:gpu0`; record the envelope's actual
verdict per shape.

**In scope, slice B:** PORT the GEMV kernel to `bits == 4` and instantiate
`(bits = 4, cb = 2)` — 270 of the 409 trellis modules, the largest single
population in the #2495 checkpoint and the item #2570 leads with; gate the new
arm at tier 3c in BOTH configs of the envelope; make the eight bits-4 admission
thresholds executable; measure the decode effect on `dgx:gpu0`.

**Out of scope and owed, not silently dropped:** the 2 bpw GEMV arm, `(4, 0)`
and `(4, 1)`, and the fused MoE arm. Each is under `## Owed` with its reason.

Two items that stood in this list when the slice was written are no longer owed
by anyone, and both closed on `main` while this branch was in flight. The
four-shape coverage of the regular kernel's shape table CLOSED under
`QUANT-EXL3` W5 ([#2749](https://github.com/mudler/vllm.cpp/issues/2749)), which
forces all four shapes and gates them from both sides of every threshold.
`kExl3HadR` on ROCm was WITHDRAWN as a false record
([#2757](https://github.com/mudler/vllm.cpp/issues/2757)): ROCm implements the
Hadamard natively in `rocm_exl3.hip` and calls it from `Exl3GemmKernelRocm`, so
it never used the reference tier, and `Exl3HadR128(` has no production caller on
any backend, which is why Vulkan declines to register it too.

## Slice B design — what a width costs that a codebook does not

`cb` reaches exactly one call in the GEMV, the `dq8_regs_*bits<cb>` extractor,
and nothing about the kernel's geometry depends on it. That is why slice A was
one template argument. `bits` is a GEOMETRY argument and reaches four places,
which is why slice B is a port. All four, with the upstream line that defines
each:

| What | Upstream | bits 3 | bits 4 |
|---|---|---|---|
| `LSTRIDE`, uint32 per warp load | `exl3_gemv_kernel.cuh:153` | 24 | 32 |
| the global-load lane guard | `:226-232` | `lane < 24` | every lane |
| the shared-stage store guard | `:265-273` | `lane < 24` | every lane |
| the window extractor | `:87` / `:121` | `dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, …)` | `dq8_regs_4bits<cb>(tp[(lane+31)&31], tp[lane], …)` |

The guards and the extractor are the SAME fact seen twice. A 16x16 tile is
`8 * bits` uint32, so it is 24 words at 3 bpw and 32 at 4. At 4 bpw that is
exactly one word per lane and four nibble-aligned codewords inside each, so the
window pair is always the lane's own word and its left neighbour. At 3 bpw the
codeword boundaries do not align to the uint32 grid at all — the window start
moves by 3 bits per codeword — so every lane computes its own word pair and
funnel offset (`x_src_a`, `x_src_b`, `x_s2`, `:206-216`), and eight lanes of the
warp carry nothing.

`dq8_regs_4bits` is the register form of `dq8_aligned_4bits`
(`exl3_dq.cuh:163-184`), and it needs two immediate-operand PTX forms this tree
did not carry: `shf.r.wrap.b32` and `bfe.u32` with a literal field offset
(`ptx.cuh:314-315`). They are macros upstream and macros here, because the
operand is an ASSEMBLER IMMEDIATE and no runtime argument or template non-type
argument can reach an `asm` string literal. Both are `#undef`d at the end of the
arm so neither name escapes it.

What is NOT ported, and why it is named rather than dropped:

- **2 bpw.** A fourth geometry: `LOADS` halves to `WNT / 2` and ONE loaded word
  carries two tiles, selected by a `(t & 1) << 4` lane base (`:152`,
  `:302-310`). No 2-bit EXL3 artifact has reached this tree, so there is nothing
  to gate it against, and an ungated third geometry in a translation unit the
  fat build compiles for ten architectures is cost without a reader.
- **`(4, 0)` and `(4, 1)`.** The kernel should compile for them -- `cb` is a
  free template argument reaching one call, and `decode_3inst_2<cb>` carries all
  three codebooks -- and `Exl3GemvHardEligible` admits `(4, 0)`. **That is an
  inference for `(4, 0)` and a MEASUREMENT for `(4, 1)`**: no translation unit
  instantiates either, so nothing here proves it, and only the thor job's M2
  mutation compiles `GemvKernelForArm<4, 1>` at all. Do not upgrade the wording
  until M2 reports a build — upstream's `K != 4 && cb == 0`
  refusal exempts exactly this width. They are still not instantiated, because
  16 kernels each that no artifact in this tree can reach is precisely the
  `(3, 0)` mistake recorded at `cuda_exl3.cu`'s GEMV arm-set comment, which
  reached a published 8% number measured on a path neither arm could take.

## THE WIDE CONFIG IS NOT THE ESCAPE, AND THE PREMISE THAT IT WAS IS FALSIFIED

Slice A predicted, before its device job ran, that `narrow_coresident` has a
thread-budget ceiling of `floor(max_threads_per_sm / 512) * sm_count` — measured
at **60** on Thor — against the 544 and 160 this checkpoint's two bits-3 shapes
need. It then named the wide config as the next hypothesis, on the reading that
upstream admits it at large `n` for `K == 4` (`exl3_gemv.cu:69`).

**Read at this checkpoint's actual shapes, that escape does not exist.** The
wide-config band is `size_n >= 8192 && size_k <= 4096`, and the SMALLEST 4-bit
`k` in the artifact is 5120. Every one of the eight bits-4 shapes fails
`size_k <= 4096`, so on Blackwell at mode 1 the branch is dead for all 270
modules, exactly as it is for the 137 bits-3 ones. `K == 4` does avoid the
bits-3 early `return -1;`, so the door stays open one line longer; it leads
nowhere here.

What bits 4 actually buys is LOWER THRESHOLDS on the same narrow branch, because
`size_n / 32` is the only term that moves and the 4-bit population is spread
over six values of `n` rather than two:

| `n` | `n / 32` | bits-4 modules | cumulative admitted |
|---:|---:|---:|---:|
| 1024 | 32 | 34 | 34 |
| 5120 | 160 | 85 | 119 |
| 6144 | 192 | 48 | 167 |
| 10240 | 320 | 48 | 215 |
| 12288 | 384 | 17 | 232 |
| 17408 | 544 | 38 | 270 |

Against the bits-3 side, where `n = 5120` (45 modules) needs 160 and
`n = 17408` (92 modules) needs 544. So at a `narrow_coresident` of 160 the arm
set reaches 119 bits-4 modules plus 45 bits-3 ones, 164 of 409, where slice A
alone reached 45. At 32 it reaches 34 and 0, and below 32 it reaches nothing at
either width.

**Whether GB10 reaches 160 is NOT settled here, and the arithmetic that would
settle it is only half known.** Thor MEASURED `SM_COUNT=20,
MAX_THREADS_PER_SM=1536` in slice A, which ceilings it at 60. GB10's SM count is
recorded in this tree as 48 (`.agents/specs/dspark-spec-decode.md` §"GB10 | 48
SMs, 102400 B shared per SM", and `.agents/specs/moe-shared-aux-stream.md`), but
its `maxThreadsPerMultiProcessor` is recorded NOWHERE and neither is the
`(4, 2)` narrow kernel's actual `blocks_per_sm`, which can be below the thread
ceiling if the kernel is register- or shared-memory-bound. 48 SMs needs
`blocks_per_sm >= 4` to clear 160 — reachable at 2048 threads/SM and not at
1536. The dgx job prints both numbers in section F, which is why that section
runs before the A/B and why its output must be read first.

**This is recorded as a correction, not as a design.** The prediction it
replaces was written into this spec before the measurement and is left standing
above; naming which half of it survived is the point of writing it down first.
`tests/vt/test_exl3_gemv.cpp` asserts every row of the table above from both
sides of its threshold, and asserts that the wide band is unreachable at
`size_k == 4224` and reachable at `4096`, so the paragraph is a gate.

**The occupancy ceiling therefore still binds, and it still binds hardest.**
`narrow_coresident` is `blocks_per_sm * sm_count` with 512-thread blocks; the
`(4, 2)` narrow kernel is a different kernel from the `(3, 2)` one and gets its
own `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, which can be lower than the
thread-budget ceiling if it is register- or shared-memory-bound. Its shared
footprint is the same 20 KB (`sh_red` 16 KB, `sh_stage` 4 KB when staged) and
its prefetch ring is the same `PF * LOADS == 8` words, so no term predicts a
drop — but that is a hypothesis, and only a launch on the device measures it.

## Upstream chain

vLLM implements no EXL3 at the parity pin, so the chain below is the secondary
oracle's, read at `2398c05635fbbad01a0a51dce63c85c6c8a8450e` and cited by
`file:line` as `.agents/porting.md` requires.

| Upstream path | What it defines | Where it lands here |
|---|---|---|
| `exllamav3_ext/quant/exl3_gemv.cu:29-42` | the `EXL3_GEMV` and `EXL3_GEMV_SMEM` knobs | `Exl3GemvParseMode` / `Exl3GemvParseSmemMode`, `src/vt/exl3_policy.cpp` |
| `exllamav3_ext/quant/exl3_gemv.cu:46-72` | `exl3_gemv_cfg`, the narrow/wide/decline envelope | `Exl3GemvSelectConfig`, `src/vt/exl3_policy.cpp:154-179`, verbatim |
| `exllamav3_ext/quant/exl3_gemv.cu:83-86` | `SEL_GRID`, the instantiated `(bits, cb)` arms | `Exl3GemvArmInstantiated` / `GemvKernel`, `src/vt/cuda/cuda_exl3.cu` |
| `exllamav3_ext/quant/exl3_gemv.cu:108-114` | the hard eligibility tests, in upstream's order | `Exl3GemvHardEligible`, `src/vt/exl3_policy.cpp:141-152` |
| `exllamav3_ext/quant/exl3_gemv.cu:171-241` | the direct entry point that ERRORS on an ineligible call | `Exl3GemmArgs::force_gemv`, used by the device gate |
| `exllamav3_ext/quant/exl3_gemv_kernel.cuh:31` | `EXL3_GEMV_MAX_M` | `vt::kExl3GemvMaxM` / `kExl3GemvMaxMDev`, asserted equal |
| `exllamav3_ext/quant/exl3_gemv_kernel.cuh:37-52` | the fp16-accumulate `mma.m16n8k16` fragment | `ptx_mma_ab_h`, `src/vt/cuda/cuda_exl3.cu` |
| `exllamav3_ext/quant/exl3_gemv_kernel.cuh:120-134` | the register-form `dq8` with per-lane funnel alignment | `dq8_regs_3bits<cb>` |
| `exllamav3_ext/quant/exl3_gemv_kernel.cuh:138-402` | the kernel, both configs, both m-modes | `exl3_gemv_kernel`, narrowed to `bits == 3` |
| `exllamav3_ext/quant/codebook.cuh:56-90` | `decode_3inst<cb>` for all three codebooks | `decode_3inst_2<cb>`, landed by `QUANT-EXL3-MUL1` slice A |
| `exllamav3_ext/quant/exl3_gemm.cu:220-236` | a DECLINED GEMV falls through to the shape table | `Exl3GemmKernelCuda`'s `TryGemv` call site |

The chain that EXECUTES on a decode step is: the model's linear method reaches
`vt::Exl3Gemm` through `ModelRegistry::Forward`; the CUDA arm calls
`Exl3GemvTryLaunch`; that reads the arm predicate, then `Exl3GemvHardEligible`,
then the env mode, then an occupancy query, then `Exl3GemvSelectConfig`; a
non-negative config launches `exl3_gemv_kernel` cooperatively and a `-1` falls
through to `exl3_gemm_kernel`. Every one of those is a place the arm can be lost,
which is why the measurement observes the kernel rather than inferring it.

## Our baseline

Measured on `dgx:gpu0` (GB10, `sm_121a`, driver 580.173.02) under an `rc` lease,
prompt `The capital of France is`, 64 tokens, greedy, `--repeat 5` with run 1
discarded as cold, `VT_DFLASH_PAGED=0` (#2274):

| arm | decode tok/s | spread |
|---|---|---|
| target only | 16.670 – 16.796 | 0.75%, two interleaved legs |
| target + DFlash2 draft, k = 7 | 48.446 – 49.079 | two interleaved legs |

Reproduced by this row's own job before anything is compared, on the same
binary, because a sequential A/B on this box measures drift as much as it
measures a change: one unchanged binary has read 36.82 and 78.86 tok/s in a
single session here.

The GEMV's own baseline is that it NEVER RUNS on this checkpoint. Every one of
its 409 trellis modules is codebook 2, and the arm predicate admitted only
`(3, 1)`.

## Port map

| Item | Upstream | Here | Status |
|---|---|---|---|
| `(3, 1)` GEMV arm | `SEL_GRID(3, 1, *)` | `GemvKernelForArm<3, 1>` | landed, `MODEL-DSV4-EXL3` W2c |
| `(3, 2)` GEMV arm | `SEL_GRID(3, 2, false)` | `GemvKernelForArm<3, 2>` | **this row, slice A** |
| `(4, 2)` GEMV arm | `SEL_GRID(4, 2, false)` | `GemvKernelForArm<4, 2>` | **this row, slice B** |
| `dq8_regs_4bits<cb>` | `exl3_gemv_kernel.cuh:86-100` | `dq8_regs_4bits` in `cuda_exl3.cu` | **this row, slice B** |
| `FSHF_IMM`, `BFE16_IMM` | `ptx.cuh:314-315` | `VT_EXL3_FSHF_IMM`, `VT_EXL3_BFE16_IMM`, scoped and `#undef`d | **this row, slice B** |
| per-width `LSTRIDE` and lane guards | `:153`, `:226-232`, `:265-273` | `exl3_gemv_kernel` | **this row, slice B** |
| `(4, 0)`, `(4, 1)` | `SEL_GRID(4, 0/1, false)` | not instantiated; the kernel is expected to compile for them and nothing here builds either | OWED, and it is now one line each |
| `(2, 1)`, `(2, 2)` | `SEL_GRID(2, *, *)` | — | OWED; a third geometry, and no 2-bit artifact has reached this tree |
| `(3, 1)`, `(3, 2)` smem-staged | `SEL_GRID(3, *, true)` | `SMEM_STAGE` template arm, both instantiated | landed |
| the envelope | `exl3_gemv_cfg` | `Exl3GemvSelectConfig` | landed, verbatim, per-K so `(3, 2)` inherits `(3, 1)`'s |
| the hard tests | `exl3_gemv.cu:110-114` | `Exl3GemvHardEligible` | landed, upstream's order |

## Tests to port

Upstream ships no C++ unit test for `exl3_gemv`; it exercises the path from
Python through `exl3_gemv` and `exl3_gemm` with `EXL3_GEMV` set, and its
correctness reference is the same linear evaluated by the regular kernel. The
adaptation is recorded rather than hidden: the envelope is extracted as a pure
function and gated directly, which upstream cannot do because its copy is
`static` inside the `.cu`, and the kernel is gated against this tree's CPU arm,
which `test_exl3_gemm` already gates against an f64 chain built from
DEFINITIONS. That keeps ONE reference for both device arms instead of two that
must agree.

What is preserved from upstream: `EXL3_GEMV_MAX_M`, the mode values and their
`atoi` parsing including the unset defaults, every branch of the envelope, the
`K != 4 && cb == 0` refusal, the 128-alignment tests, and the `force` semantics
of the direct entry point.

## Dependencies

- `QUANT-EXL3` (#2181) — the format, the CPU reference, `had_r_128`, and the
  tier-3c bound this row inherits rather than restates.
- `QUANT-EXL3-MUL1` (#2495) — `decode_3inst_2<2>` and `decode_mul1_product_2`.
  Without codebook 2 in the shared decoder there is no `(3, 2)` arm to
  instantiate; this row adds no decode of its own.
- `.agents/oracles/exllamav3.md` — the pin. Advancing it re-opens every
  `file:line` above.
- An `rc` lease on `dgx:gpu0` for anything device-shaped. There is no local CUDA
  toolchain, so every device claim in this row is PENDING until the lease runs.
- #2274 (`VT_DFLASH_PAGED`) — the workaround the baseline carries, applied to
  every leg alike so it cancels in the ratio.

## Risks/decisions

- **D1. The A/B is env-driven on ONE binary, not two binaries.** Decided,
  because on this checkpoint `VT_EXL3_GEMV=0` and the pre-change binary take the
  same code path: the old predicate admitted only `(3, 1)` and the artifact has
  zero `(3, 1)` tensors. That removes binary drift from the comparison. The
  equivalence is not assumed — a mutation that takes `(3, 2)` back out of the
  predicate rebuilds the pre-change product behaviour and is measured.
- **D2. Three legs, not two.** `VT_EXL3_GEMV=2` is upstream's testing mode and
  is carried as a DIAGNOSTIC so an envelope decline stays distinguishable from a
  null effect. It is never reported as a production number. R: quoting it as one
  would be the 8%-from-nothing failure again.
- **D3. The tier is 3c and it is inherited.** R: a `(3, 2)` arm that misses
  `6.0e-3` is wrong; widening the bound to admit it would delete the only check
  that can see a mis-threaded codebook.
- **R1. The envelope may decline every bits-3 shape on GB10.** Then the arm is
  correct, upstream-faithful and worth nothing on this device, and the next
  hypothesis is the narrow config's occupancy. Recorded in `## Owed` as the
  outcome it would be, not as a failure of the row.
- **R2. The fat build compiles one more kernel set.** `(3, 2)` is 16 more
  kernels in a translation unit the fat build compiles for ten architectures.
  Upstream answers this with a per-`K` compilation-unit split
  (`comp_units/exl3_comp_unit_K_cbX.cu`); this tree has one unit and the split
  stays owed by `QUANT-EXL3-MUL1`, which argued it first.
- **D4. The `QUANT-*EXL3*` sibling ratchet is widened by NAME, not by
  predicate.** `tests/scripts/test_agent_record.py` pins the exact set of
  `QUANT-` rows carrying `EXL3`, and it went RED on this row before its argument
  was written, which is the gate working. The argument is that this row sits on
  a different AXIS rather than being a third scheme: the two siblings answer
  "does this width RUN" and this one answers "what does it COST", and
  `QUANT-EXL3-MUL1`'s own claim file EXCLUDES the GEMV by name, so this row is
  the owner that exclusion implies. A FOURTH row still fails there and must
  argue for itself.
- **D5. `docs/FEATURES.md` is edited, `docs/BENCHMARKS.md` is not.** The GEMV
  arm set is a quantization surface and its row said the mul1 GEMV was owed, so
  that sentence is now false and is repaired. No benchmark ID is added, because
  no number exists yet; the cell says the measurement is PENDING rather than
  omitting it.
- **R3. Measuring on a box that crashes.** `dgx:gpu0` has crashed roughly hourly
  under long ladders. The job prints results incrementally and orders the A/B
  ahead of the mutations, so a crash costs the cheapest evidence rather than the
  most expensive.

## Why `(3, 2)` is an instantiation and `(4, 2)` is a port

`exl3_gemv_kernel` in `src/vt/cuda/cuda_exl3.cu` is already
`template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE>`
and threads `cb` all the way down: the only decode site in it is
`dq8_regs_3bits<cb>`, which calls `decode_3inst_2<cb>`, and that function has
carried all three codebooks since `QUANT-EXL3-MUL1` slice A. `GemvKernelForArm`
is already `template <int BITS, int CB>`. Nothing about the kernel's geometry
depends on `cb`: `LSTRIDE`, `TWORDS`, `FOLD`, `PF` and `LOADS` are all functions
of `bits`, `CFG` and `MMODE`.

So `(3, 2)` changes ONE template argument and ONE predicate. It covers 137 of
the 409 modules.

`(4, 2)` did not, and slice B did that port. **One sentence written here at
slice A was WRONG and is corrected rather than quietly dropped**: it read
"`TWORDS == 32` and they are not equal; the prefetch ring depth `PF`, the fold
cadence `FOLD` and the load count `LOADS` are all tuned around that equality".
`LSTRIDE` and `TWORDS` ARE equal at bits 4 as well — upstream's `LSTRIDE` is
`bits == 3 ? 24 : 32` and `TWORDS` is `8 * bits`, which agree at 24 and at 32 —
and `PF`, `FOLD` and `LOADS` do not depend on `bits` at all at these two widths
(`exl3_gemv_kernel.cuh:145-153`). The kernel now carries
`static_assert(LSTRIDE == TWORDS)` so the corrected claim is executable. What
`bits` actually costs is in `## Slice B design` above: a per-width `LSTRIDE`,
two lane guards that exist only because a bits-3 tile is 24 words, and a
different window extractor. The 2 bpw arm IS the case the wrong sentence
described — there `LOADS` halves and one word carries two tiles — and it stays
unported.

## The envelope is ALREADY ported, and that matters

#2570 asks that any port bring `exl3_gemv.cu:55-71` with it. It is already here,
verbatim, including the `K == 4` wide-config admission and the commented-out
`cc != CC_AMPERE` guard that upstream disabled
(`src/vt/exl3_policy.cpp:154-179`), and `tests/vt/test_exl3_gemv.cpp:101-122`
gates its branches on any machine. Neither slice adds an envelope change,
because there is none to add: upstream's envelope is per-`K`, not per-`cb`, so
`(3, 2)` inherits `(3, 1)`'s exactly and `(4, 2)` inherits `(4, 1)`'s and
`(4, 0)`'s. Slice B ASSERTS that rather than assuming it — the case loops all
eight bits-4 shapes at four occupancies each and requires `(4, 2)`, `(4, 1)` and
`(4, 0)` to return the same config every time. If the envelope ever stops being
per-`K`, this row's "no envelope change" claim fails there and not in a
benchmark.

## The numeric tier is 3c, and it is NOT widened

The GEMV accumulates in `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` and
folds to an f32 pair only every `FOLD` iterations, so an fp16 accumulator
absorbs up to `FOLD * 16` k-elements. `QUANT-EXL3`'s `## W2cd design` W2c-3 sets
its bound at tier 3c, relative RMS `6.0e-3`, rather than tier 3's `1.0e-3`. A
new arm INHERITS that bound. If `(3, 2)` or `(4, 2)` cannot meet `6.0e-3` the
arm is wrong; the bound does not move.

`(3, 1)` and `(3, 2)` are the confusable pair and the reason the device case
gates both. Both are three bits wide, both take the same `dq8` route, both use
the same tile shapes. A `cb` threaded wrongly between them does not fail to
compile and does not change a shape: it decodes with the other codebook's tail
and yields a weight with the right DISTRIBUTION and no correlation to the true
one. The mutation table below makes that failure red.

**`(4, 2)`'s confusable partner is `(4, 1)`, and it is deliberately NOT
compiled**, so slice A's cross-arm device check cannot be repeated for it — two
device outputs need two instantiations. The check that replaces it compares the
device output against the CPU decoder at the SIBLING codebook and requires the
distance to be LARGE: `rel_sib > 100 * 6.0e-3`. Two decodes of the same bits
under different codebooks are uncorrelated, so `rel_sib` is order 1 while `rel`
is order `1e-4`; the constant sits two orders below the one and three above the
other and is not tuned to a measurement. That form generalises to any arm,
including one whose partner is not on the device, and it is applied to all four
legs rather than to the new one alone. Slice A's direct `(3, 1)`-vs-`(3, 2)`
device comparison is KEPT beside it, because it needs no host reference at all.

## Work breakdown

- **A1.** Extend `tests/vt/test_exl3_gemv.cpp`'s device case over `cb` in
  `{1, 2}`, referenced against the CPU arm at the same `cb`. RED first: `(3, 2)`
  declines the arm before A2, so the forced call throws.
- **A2.** Add `(3, 2)` to `Exl3GemvArmInstantiated` and `GemvKernel`, and
  correct the comment block above them, which asserts the arm set is `(3, 1)`
  only.
- **A3.** Add a machine-independent case that reads the ENVELOPE at this
  checkpoint's real shapes, so the table in this spec is executable rather than
  prose.
- **A4.** Measure on `dgx:gpu0` under an `rc` lease: the four legs above,
  interleaved, one binary per arm, `narrow_coresident` printed.
- **B1.** Port `dq8_regs_4bits<cb>` and the two immediate PTX forms it needs.
- **B2.** Generalise `exl3_gemv_kernel` to `bits == 4`: per-width `LSTRIDE`, the
  two 24-lane guards narrowed to bits 3, the funnel constants made bits-3-only,
  the extractor dispatched per width.
- **B3.** Add `(4, 2)` to `Exl3GemvArmInstantiated` and `GemvKernel`, and repair
  the comment block above them, which says `(4, 2)` is owed.
- **B4.** Extend the device case to gate `(4, 2)` in BOTH configs, and give
  every leg a sibling-codebook discrimination that does not need the confusable
  arm to be compiled. RED first: before B3 the forced `(4, 2)` call throws by
  name, and mutation M1 restores exactly that state.
- **B5.** Make the eight bits-4 admission thresholds executable, both sides of
  each, and assert the per-`K` envelope claim across `cb` 0, 1 and 2.
- **B6.** Measure on `dgx:gpu0`: `(4, 2)` on, off, and at mode 2, interleaved on
  one binary, with `nsys --cuda-graph-trace=node` confirming that
  `exl3_gemv_kernel` RAN.

## Tests

- `tests/vt/test_exl3_gemv.cpp` — the envelope cases (any machine) and the
  device tier-3c case (CUDA only, skips loudly and still asserts).
- The device case covers `(3, 1)`, `(3, 2)`, `(4, 2)` narrow and `(4, 2)` wide.
  The two configs are different compiled kernels with different `WK`, `WNT`,
  `PF` and `FOLD`, and gating one would leave the other measured by nothing.
  `force_gemv` drives the envelope at mode 2, whose whole rule is
  `size_n <= 8192 ? 0 : 1`, so `n` selects the config: `n = 4096` is narrow and
  `n = 8320` is wide.
- `SMEM_STAGE` is a THIRD arm per config and is selected by `VT_EXL3_GEMV_SMEM`,
  which `Exl3GemvSmemMode()` caches in a function-local static. It cannot be
  flipped inside one process, so the lease job runs the same binary a second
  time with `VT_EXL3_GEMV_SMEM=1` rather than leaving the staged path ungated.
- The device case is FORCED through `Exl3GemmArgs::force_gemv`, mirroring
  upstream's direct entry point (`exl3_gemv.cu:171-241`). Forcing is what makes
  it a gate rather than a coin flip on an occupancy query: without it a device
  whose envelope declines the shape measures the REGULAR kernel and reports
  tier 3c green.

## Gates

```sh
ctest --test-dir build -R '^test_exl3_gemv$' --output-on-failure
ctest --test-dir build -R '^test_exl3_gemm$' --output-on-failure
scripts/agent-preflight.sh --staged
```

The device arm additionally needs an `rc` lease and a CUDA build:

```sh
rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R '^test_exl3_gemv$' -V
```

A CPU-only green is not a device result and is never reported as one. A doctest
`assertions: 0` line is a skip wearing a pass; read the ctest exit code.

## Owed

- **The 2 bpw GEMV arm.** `LOADS` halves to `WNT / 2` and one loaded word
  carries TWO tiles (`exl3_gemv_kernel.cuh:152`, `:302-310`), so it is a third
  geometry rather than another width of slice B's. No 2-bit EXL3 artifact has
  reached this tree, so nothing would gate it. Named, not attempted.
- **`(4, 0)` and `(4, 1)`.** After slice B the kernel is expected to compile for
  them -- unmeasured, see the note under `## Slice B design` -- and the hard
  tests admit `(4, 0)` — upstream's `K != 4 && cb == 0` refusal exempts
  this width alone — so each is ONE line in `GemvKernel` and one term in the
  predicate. They stay out because no artifact here carries a 4-bit tensor at
  either codebook, and 16 unreachable kernels each in a translation unit the fat
  build compiles for ten architectures is the `(3, 0)` mistake this row's own
  comment block records. Owed with the artifact that needs them.
- **MEASURED, and no longer owed: the envelope DOES decline almost every shape
  on GB10, at both widths.** `narrow_coresident` ceilings at 144
  (`SM_COUNT=48`, `MAX_THREADS_PER_SM=1536`, 512-thread blocks), so 34 of 409
  modules are admitted — 0.75% of the trellis bytes — and the decode A/B is a
  null at 1.3% spread with `nsys` confirming the arm never launches at the
  default. What remains owed is the DECISION this creates, not the measurement:
  see `## THE REASSESSMENT`.

- **`(4, 2)`'s own throughput leg on `dgx:gpu0`.** Queued and did not run. It is
  owed, and the reassessment says what to expect from it rather than standing in
  for it: at 0.75% of weight bytes under a 1.3% spread it cannot resolve, so
  running it would produce a null that means nothing new. The measurement that
  WOULD be worth taking is on a part with `MAX_THREADS_PER_SM >= 2048`, where
  the same arm set admits 212 of 409 modules.

- **`m == 8` / `MMODE == 1` is UNMEASURED at every width.** The dgx job's draft
  legs were skipped and said so rather than reporting a null. Every throughput
  number this row holds is `m == 1`. The speculative-decode arm compiles a
  different kernel with row-guarded fragment loads and sits at a different point
  on the envelope; nothing here touches it.

- **The fused MoE arm is `(3, 1)` only** (`kMoeBits`/`kMoeCb` in
  `cuda_exl3.cu`), and it is CUDA-only. The #2495 checkpoint is dense so nothing
  here reaches it, and it is named so the next MoE EXL3 artifact does not
  rediscover it.
- **`kExl3HadR` on ROCm runs the portable CPU reference tier, not a native
  kernel.** It is correct and it is not fast, and no row owned that either.
- **The regular kernel's shape table is gated at ONE of its four shapes.**
  Read at `src/vt/exl3_policy.cpp:98-110`, the Blackwell arm returns shape 2 at
  the dimensions the suite tests (`k 2048, n 4096, bits 3`: `mod_256 &&
  size_n <= 4096` with `size_k > 8192` false). Shape 4 needs
  `mod_512 && size_n > 16384` and shape 1 needs `(K == 4 || K == 2) && !multi &&
  size_k <= 2048`, so neither is reachable at those dimensions at all.
  `force_shape_idx` already exists, so covering the other three is a test loop
  rather than a port. Owed by `QUANT-EXL3-MUL1`, which owns that table; repeated
  here because the GEMV falls THROUGH to it on every shape the envelope
  declines, which on this checkpoint may be all of them.
- **`bits` 5 and 6 have no GEMV upstream either** (`exl3_gemv.cu:110-111`
  refuses `K < 2 || K > 4`), so the one 5-bit tensor and the 6-bit `lm_head`
  falling to the regular kernel is upstream's behaviour and not a gap. Recorded
  so it is not re-filed.
- **`docs/USAGE.md` owes this checkpoint's file names, sizes, repo and
  REVISION.** Owed by `QUANT-EXL3-MUL1`, which loads it; repeated here because
  this row measures it.

## Stop conditions

- `(3, 2)` and `(3, 1)` agree elementwise on the device case → the case is not
  discriminating; the two codebooks must produce different numbers at the same
  width. Stop and fix the fixture before reading any tolerance.
- The tier-3c bound fails → the arm is wrong. Never widen the bound.
- The `(4, 2)` port needs a kernel structure that is not upstream's → return
  `NEEDS_DECISION` rather than inventing one. It did not: every line of slice B
  has an upstream `file:line` in `## Slice B design`.
- The lease never arrives → report the arm as instantiated-and-ungated, and say
  so. A queued job nobody could gate is a partial result, never a pass.
- An A/B leg shows the GEMV arm never ran → the measurement is void, not a zero.
  Print the envelope's verdict before reading any tok/s.
