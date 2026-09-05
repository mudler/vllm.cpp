# The CUDA router top-k IS stable at an exact tie -- and the geometry that proves it had never been run, 2 September 2026

Wave TIEBREAK of [`MODEL-MM-QWEN4-EXP`](../../.agents/specs/qwen4-exp-flash-next.md),
[#2586](https://github.com/mudler/vllm.cpp/issues/2586).

**The one-line result. The CUDA top-k realises the lowest-index tie-break it
declares.** On bit-identical logits whose top-k boundary is an EXACT bf16 tie,
at `E = 512` `k = 10` -- the geometry `qwen4_exp` routes -- the dispatched CUDA
kernel returns the closed-form correct selection, agrees byte-for-byte with the
serial GPU oracle and with the CPU reference, repeats identically over 32
launches, and gives the same answer for a row run alone and as each of 257 rows
of a batch. **4652 assertions, 0 failed, rc = 0.** Wave MOEDIV's conclusion
([#2552](https://github.com/mudler/vllm.cpp/issues/2552)) stands: the arms'
expert flips are tie-break order under a perturbed input, not an unstable
tie-break, and the remaining gap on the session's token goal stays attributed to
arithmetic.

**And the tree could not have said so before this run.** A mutation that makes
the block kernel's per-thread scan pick the HIGHEST tied index instead of the
lowest is caught by 104 of this file's assertions and by **ZERO assertions of
the standing adversarial sweep**, whose counts are byte-identical before and
after it. The blind spot was real, it was exactly `E > 256`, and it is now
closed.

## 1. What was run, and on what

**Tree.** `64617b150e35e27e2f17353ff1c206935238c80c` -- `origin/main` `4de001dfe`
plus this wave's spec section and one new test file. No product code changed.

**Host.** `thor:gpu0` under an `rc` lease, job
`5f50e5f7-c730-463f-93cc-3293835ed007`, pod `rc-worker-n8smh`, aarch64,
`NVIDIA Thor`, driver 595.78, compute capability 11.0, `cuda_13.0.r13.0` from
the `sbsa` repo, built `sm_110`.

**It is a CUDA build, and that is asserted rather than assumed.**
`CMAKE rc=0`; `BUILD rc=0 objects=586`; 41 `*.cu.o`; and the test binary's own
`ldd` resolves
`libcudart.so.13 => /usr/local/cuda-13.0/targets/sbsa-linux/lib/libcudart.so.13`
and `libcublasLt.so.13`. Every number below came out of that one binary.

**No checkpoint was needed.** The decisive experiment is a synthetic row with a
deliberate tie, so it costs a unit test rather than the 72 GB artifact and its
~2195 s stage.

## 2. Why the standing gate could not answer the question

`tests/vt/test_ops_moe_grouped.cpp` "CUDA moe_router_topk parallel == serial
byte-for-byte (adversarial)" is a real gate. It compares against
`vt::cuda::MoeRouterTopKSerialCuda` byte-for-byte, carries tie storms, and pins
`VT_MOE_ROUTER_WARP` both ways. Two limits keep it away from this question:

1. **It sweeps `E in {32,64,128,256}` only.**
   `MoeRouterWarpValuesPerThread` (`src/vt/cuda/moe_router_warp.h`) returns 0 for
   E = 512, so `qwen4_exp`'s router falls through to
   `MoeRouterTopKKernel<Tin,false>` -- the BLOCK kernel at **2 experts per
   thread**. Nothing in the tree had executed that decomposition.
2. **Its tie patterns are `(e / 4) % 5` and `e < 12 ? 3.0f : 0.0f`**, and at
   `kBlock = 256` a thread holds ONE expert for every E it sweeps. The
   per-thread strided compare therefore never sees a tie, because there is
   nothing on that thread to compare.

## 3. The experiment

Every row has a **closed-form** correct answer, so this gates the semantics
rather than arm-vs-arm agreement: `h` experts at strictly higher distinct
bf16-exact values, a tied set `S` whose members carry one bf16 bit pattern, and a
low floor. The correct top-k is the `h` winners in descending value order
followed by **the `k - h` LOWEST INDICES OF `S`**. That is upstream's own rule,
not a local convention -- see §5.

Six patterns place `S`, because the placement decides WHICH LEVEL of the
reduction has to break the tie: scattered by `(197i + 11) mod E`; the upper half;
the last warp of block 0; lanes 28..31 of every warp; a contiguous block just
under E; and **both strided slots of the same threads**, the only pattern that
can reach the per-thread compare and the only one that needs `E > 256`.
Geometries `E in {256, 512, 1024}`, `k in {8, 10}`, `h in {0, 1, k-1}`, f32 and
bf16 logits, `VT_MOE_ROUTER_WARP` pinned ON and OFF with the pinned state
`REQUIRE`d.

**The two `VT_MOE_ROUTER_WARP` arms are two structures only at E = 256.**
`MoeRouterWarpValuesPerThread` (`moe_router_warp.h:96-98`) returns 0 for every E
outside `{32,64,128,256}`, and `LaunchRouterWarp` (`cuda_moe.cu:564`) returns
`false` on `vpt == 0` before it touches a tensor, so at E = 512 and E = 1024 --
`qwen4_exp`'s own geometry included -- the warp-on arm falls through to the same
`MoeRouterTopKKernel<Tin,false>` the warp-off arm runs. Every count in §4 and §6
is honest, because every assertion really executed; but 36 of the 72 rows at
E > 256 are byte-identical repeats, so §6's `12` per cell is
`2 arms x 3 h x 2 dtypes` of which 6 are the second count of one disagreement,
and COVERAGE above E = 256 is half what the doubled figures imply. Filed as
[#2604](https://github.com/mudler/vllm.cpp/issues/2604) and listed under the
spec's `## Owed`; not repaired here because re-narrowing the sweep would move
every number this document records and needs the lease again.

Indices are compared with `==` and the selection as an ordered vector. **There is
no tolerance anywhere in the file**, because a discrete selection has bimodal
error. The counted property is `tied_seen`: the number of rows whose boundary
carries the SAME bf16 bit pattern on both sides, read off the LOGITS and
therefore independent of every kernel under test, `CHECK`ed equal to the row
count (54 on the CPU case, 108 on the device case). A builder that stopped
producing ties drives it to 0 and the case cannot read green.

## 4. The results

| run | binary | cases | assertions | failed | rc |
|---|---|---|---|---|---|
| baseline | `test_moe_router_tie_stability` | 4 | **4652** | **0** | **0** |
| baseline | `test_ops_moe_grouped` | 14 | 1907 | 1 | 1 |
| baseline | `test_ops_moe` | 9 | 33451 | 0 | 0 |
| mutant | `test_moe_router_tie_stability` | 4 (3 failed) | 4652 | **104** | 1 |
| mutant | `test_ops_moe_grouped` | 14 | 1907 | **1** | 1 |
| mutant | `test_ops_moe` | 9 | 33451 | **0** | 0 |
| restored | `test_moe_router_tie_stability` | 4 | 4652 | **0** | **0** |

`test_ops_moe_grouped`'s single failure is the SAME one in all three runs:
`test_ops_moe_grouped.cpp:1262 CHECK( bitdiff == 0 )` in
"CUDA marlin NVFP4 W4A16 dense M=8: block=8 byte-exact vs block=16". That is the
known Marlin NVFP4 block-size disagreement on this arch
([#962](https://github.com/mudler/vllm.cpp/issues/962)), already recorded in
`.agents/environment.md`. It is not this wave's, it is present at the baseline,
and it is unmoved by the mutation.

**So the answers, in the order they were asked:**

1. **Bit-identical logits at an exact tie produce the same selection on both
   arms**, and both equal the closed-form answer. 54 rows x 2 `VT_MOE_ROUTER_WARP` arms x 2
   dtypes = 216 selections, each checked three ways -- `par == r.expect`,
   `ser == r.expect`, `par == ser` -- all green.
2. **The CUDA arm is deterministic on the tie.** 32 launches of the same bytes
   at each of 9 (geometry, pattern) points: 279 repeat comparisons, all
   identical. And a row's selection is the same alone and at every one of 257
   batch positions: 2313 comparisons, all identical.
3. Therefore the tie-break is genuinely stable, and MOEDIV's reading holds.

## 5. What upstream does at a tie

vLLM DEFINES this tie-break rather than leaving it unspecified, and it does so in
both of its kernels. Read at the parity pin
`5559679229bc961848b121ccdeaa8fa5d79bec98` -- these are pin anchors, not forward
references:

| anchor | what it says |
|---|---|
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu:536-537` | `topkGating`'s butterfly: "We want lower indices to \"win\" in every thread so we break ties this way", `other_max_for_choice == max_val_for_choice && other_expert < expert` |
| `:515-517` | its per-thread scan: "columns with the smallest index are processed first and only updated if > (not >=)" |
| `:707-708` | `case 512: LAUNCH_TOPK(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2)` -- E = 512 IS a registered `topkGating` width, so upstream runs its register-resident kernel at this model's geometry |
| `:186`, `:222`, `:225` | the fallback `moeTopK` reduces with `cub::ArgMax`, whose contract is the same lowest-key-on-tie |
| `:465` | "With 0s, the argmax uses index tie-breaking to pick [0,1,2,...,k-1]" -- upstream states the resulting behaviour outright |

So `lowest index wins` is the specified upstream behaviour, our two arms mirror
it, and §4's closed-form assertion is a mirror of the oracle rather than a local
invention.

## 6. The mutation, and exactly what it proves

One site in `src/vt/cuda/cuda_moe.cu`, the block kernel's per-thread strided
scan:

```
-        if (v > lv) {  // strict `>`, ascending stride -> lowest index at max
+        if (v > lv || (v == lv && li >= 0)) {  // MUTATION highest-index-wins
```

`MUTATION sites found: 1` (an assert, not a grep), `MUT BUILD rc=0 objects=1`,
and the restore rebuilt and returned 4652/4652. The polarity flip is deliberately
NON-destructive -- it never lets a masked `-INFINITY` slot win -- so it changes
the tie-break and nothing else.

**Where the 104 failures land is the whole point:**

| geometry | pattern | `par == r.expect` failures |
|---|---|---|
| E = 512 | 3 (lanes 28..31 of every warp) | 12 |
| E = 512 | 5 (both slots of the same threads) | 12 |
| E = 1024 | 3 | 12 |
| E = 1024 | 5 | 12 |
| **E = 256** | **any** | **0** |

plus the matching 48 `par == ser` failures and 8 in the determinism and batch
cases. **Zero at E = 256**, which is why the standing sweep's counts are
byte-identical with the mutation in place: 1907 assertions, 1906 passed, the one
pre-existing #962 failure, before AND after.

A representative failure names the mechanism:

```
CHECK( par == r.expect ) is NOT correct!
  values: {284, 28, 285, 29, 286, 30, 287, 31, 316, 60}
       == {28,  29, 30,  31, 60,  61, 62,  63, 92,  93}
  g.e := 512   g.k := 10   pattern := 3   h := 0   r.tied.size() := 64
```

Thread 28 owns experts 28 and 284, both tied. The correct kernel reports 28; the
mutant reports 284, and expert 28 then never reaches the warp reduction at all.
This compare does not exist at `E <= 256`.

**The CPU half has its own red-first, and it is the one CI runs everywhere.**
On the authoring host, CPU-only build at the branch head, flipping the CPU
greedy argmax's strict `>` to `>=` at ONE site --
`src/vt/cpu/cpu_ops.cpp:2985`, `if (p[static_cast<size_t>(idx)] > best_v)`,
`grep -c` = 1 -- reds **every** selection assertion in the CPU case:

| run | cases | assertions | passed | failed | rc |
|---|---|---|---|---|---|
| baseline | 1 | 488 | 488 | **0** | 0 |
| mutant | 1 | 488 | 380 | **108** | 1 |
| restored | 1 | 488 | 488 | **0** | 0 |

All 108 failures are `CHECK( ids == r.expect )` and none is a counted-property
assertion, so the mutant is caught by the SELECTION and the row builder is
proved still to have built ties. 108 is the whole population and not a subset:
`h < k` is `REQUIRE`d, so every one of the 54 rows selects at least one member
of the tied set, and `>=` mis-selects on each of them in both dtypes --
`54 x 2 = 108`. An earlier revision of #2595's pull-request body said 90; that
was a stale draft figure from a five-pattern version of the file, and the
shipped file has six patterns.

**Note what the mutant did NOT break: determinism.** Its 279 repeat comparisons
and 2313 batch comparisons all still passed. A wrong tie-break is perfectly
repeatable. Determinism is a necessary condition for the correct answer, never a
sufficient one, and a run that measured only repeatability would have called this
mutant clean.

## 7. What this closes and what it leaves open

**Closes.** #2586, and the untested half of #2552's conclusion. The tie-break is
stable, so the `qwen4_exp` CPU-vs-CUDA token gap is arithmetic, exactly as
PREFILLDIV and MOEDIV said. No fix was needed and none was made; the session
goal's three disagreeing token ids do NOT move, because nothing moved them.

**Leaves open.** The gate against vLLM itself stays OWED for the same reason
MOEDIV recorded: it needs an artifact vLLM can load, and vLLM's GGUF support is
an out-of-tree plugin while every safetensors arm of this model exceeds the
largest fleet box. This run says our two arms agree with each other AND with
upstream's stated rule; it does not run upstream's kernel on upstream's weights.

## 8. The second box measured NOTHING on device, and says so out loud

A corroborating lease ran on `orin:gpu0` (job
`ca79e804-d880-4d24-af3c-ac1b929a9789`, pod `rc-worker-wjwvb`, aarch64, 12 CPUs,
nvcc `cuda-13.3` from the `sbsa` repo, configured `-DVLLM_CPP_CUDA_ARCHITECTURES=87`).
It built: `CMAKE rc=0`, `BUILD rc=0 objects=579`, 34 `*.cu.o`, and the binary's
`ldd` resolves `libcudart.so.13` and `libcublasLt.so.13`. So the tree compiles
for `sm_87` as well as `sm_110`.

**And then no device appeared.** `nvidia-smi` is not on that worker (documented
for Jetson in `.agents/environment.md`), no CUDA backend registered at runtime,
and every device case printed `no CUDA backend registered; skipping` and
returned. Its three runs read:

| run | assertions | rc |
|---|---|---|
| baseline | 488 | 0 |
| **mutant** | **488** | **0** |
| restored | 488 | 0 |

488 is the CPU case alone. **The mutant read GREEN on this box** -- same
assertion count, rc 0 -- with a broken tie-break compiled into the binary,
because the arm that would have caught it never ran. That is the
`assertions: 0 ... SUCCESS!` failure mode wearing a non-zero number.

**And the ASSERTION count alone does not separate the three states -- the CASE
count is the other half.** An earlier revision of this document said the count
was "the only thing" that distinguishes this box from Thor. It distinguishes
this box from Thor; it does NOT distinguish this box from a CPU-only build,
which reads 488 as well:

| state | cases | assertions | rc |
|---|---|---|---|
| CPU-only build — the device cases are `#ifdef`'d out | **1** | 488 | 0 |
| `orin:gpu0` — CUDA build, no device, cases skip and return | **4** | 488 | 0 |
| `thor:gpu0` — CUDA build with a device | **4** | **4652** | 0 |

The CPU-only row is measured on the authoring host at the branch head
(`test cases: 1 | 1 passed`, `assertions: 488 | 488 passed`, rc 0); the middle
row's case count follows from the three cases being compiled in and executing
their skip `return`. Read both numbers, never the status line and never one
number.

**The permissive skip that makes the middle row possible is a tree-wide shape
and is now owed.** `grep -rl 'no CUDA backend registered; skipping' tests/` finds
14 files, 13 of them not this one, and the tree already carries the idiom that
refuses instead: `tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp:149`
reads `VT_REQUIRE_27N_FP8_GATE` and turns absence into a hard FAILURE. Adopting
it across those files is a row of its own, filed as
[#2603](https://github.com/mudler/vllm.cpp/issues/2603) and listed under the
spec's `## Owed`.

`orin:gpu0` therefore corroborates the CPU arm on aarch64 and the `sm_87`
compile, and contributes nothing to §4. Every device number in this document
comes from `thor:gpu0`.

## 9. Re-measured at the MERGED head, because a merge has falsified prose here before

§1-§6 were measured at `64617b150`. That head was then merged with 16 commits of
`origin/main`, and one of them touched a file this binary links:
`src/vt/cuda/cuda_exl3.cu` (the EXL3 (3, 2) GEMM instantiation). Nothing in the
merge touched the router, the test, or `tests/CMakeLists.txt` -- but "nothing
relevant changed" is a claim, and the counts in this document are the thing at
risk, so they were taken again rather than argued for.

**Second lease, merged head `d52bc830fef80735e139819f68897e105fbc0579`:**
`thor:gpu0`, job `186b67e9-4e48-43fb-9424-65735964083c`, pod `rc-worker-n8smh`,
NVIDIA Thor, driver 595.78, cc 11.0, nvcc `cuda_13.0.r13.0`, `sm_110`.
`CMAKE rc=0`, `BUILD rc=0 objects=586`, 41 `*.cu.o`, `libcudart.so.13` resolved.

| measurement | at `64617b150` | at `d52bc830f` |
|---|---|---|
| `test_moe_router_tie_stability` baseline | 4652 / 0 failed / rc 0 | **4652 / 0 failed / rc 0** |
| same, under the mutation | 4652 / **104** failed / rc 1 | 4652 / **104** failed / rc 1 |
| same, restored | 4652 / 0 failed / rc 0 | 4652 / 0 failed / rc 0 |
| `test_ops_moe_grouped` base / mutant | 1907 / 1906 / 1 failed, both | 1907 / 1906 / 1 failed, both |
| `test_ops_moe` base / mutant | 33451 / 0 failed, both | 33451 / 0 failed, both |
| mutant failures at E = 512, patterns 3 / 5 | 12 / 12 | 12 / 12 |
| mutant failures at E = 1024, patterns 3 / 5 | 12 / 12 | 12 / 12 |
| mutant failures at **E = 256** | **0** | **0** |

Every number reproduces. The two structural claims were re-read at the merged
head as well: `tests/vt/test_ops_moe_grouped.cpp:637` still sweeps
`{32, 64, 128, 256}`, and `moe_router_warp.h:98` still returns 0 for E = 512.

One commit lands after this run, `631da56c4`, and it changes exactly one file --
`.agents/specs/qwen4-exp-flash-next.md`, repairing that section's Gates commands
-- so the binary these numbers came from is the binary the branch ships.
