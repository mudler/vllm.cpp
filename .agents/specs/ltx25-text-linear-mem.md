# `LTX25-TEXT-LINEAR-MEM` — #1252 did not cost 26 GiB, and the seam has no memory bound

Row: `LTX25-TEXT-LINEAR-MEM`
Issue: [#1286](https://github.com/mudler/vllm.cpp/issues/1286)
Sibling rows: `LTX25-TEXT-LINEAR-SEAM` (#1252, [spec](ltx25-text-linear-seam.md)),
`LTX25-LORA-FUSE-SEAM` (#1259, [spec](ltx25-lora-fuse-seam.md))

## Scope

[#1286](https://github.com/mudler/vllm.cpp/issues/1286) reports that #1252's
threaded caption projection raises peak host memory from ~79 GiB to 105.85 GiB
on GB10 and aborts a full-model LTX-2.5 render before any denoise step. It names
three candidate causes — the shared threadpool's per-thread tiles, the
`ltx2_text_encoder.cpp` call site's full-size `scaled` copy, or an interaction —
and states plainly that the attribution is not measured and belongs to whoever
takes the row.

This row does that attribution first, and lets the answer choose the fix.

In scope:

- attribute the reported +26 GiB against the retained evidence and against a
  local before/after measurement at the shipped geometry;
- report peak host memory **and** throughput for both arms, because a memory fix
  that costs the 8.57x is a bad trade;
- decide whether a code change is warranted, and make it if so;
- give the shared GEMM seam a **memory bound that a gate can hold**, since the
  tree has none and could not have caught the reported defect had it been real;
- check whether #1259, the sibling seam change that has never run on the full
  model, shares the profile.

Out of scope: the LTX-2.5 speed axis, the unmeasured GB10 per-core ratio `R`
that #1252 carries under `## Owed`, #1254's bf16 add-back, and #1210's
fuse/un-fuse round trip. None becomes reachable from this change.

## Upstream chain

None, and that is a property of the question rather than a gap. Peak resident
footprint is a property of this port's execution strategy: vLLM does not
implement LTX-2.5, vLLM-Omni's recipes stop at 2.3, and the Lightricks
`ltx_core` oracle runs the projection as one `torch.nn.functional.linear` whose
allocator behaviour says nothing about a C++ threadpool's per-worker tiles. The
arithmetic mirror is already decided and unchanged by this row: f32 accumulation
through `vt::MatmulBT`, argued in `ltx2_text_encoder.cpp:64-69` and in
[`ltx25-text-linear-seam.md`](ltx25-text-linear-seam.md).

The parallel-dispatch layer this row measures **is** a 1:1 port, and its anchors
are the ones to read: `src/vt/cpu/cpu_ops.cpp:1-8` and `cpu_threadpool.h` record
llama.cpp (local fork) `ggml/src/ggml-cpu/ggml-cpu.c:1155-1443` @ `237ad9b96` as
the chunk policy's source. `MatmulOneChunk`'s widened-activation buffer is a
recorded *deviation* from that port (`cpu_ops.cpp:99-108`), so its cost is ours
to bound.

## Our baseline

Read out of the tree at `origin/main` `678fc672c`:

1. **The seam allocates exactly one thing per call, and it is thread-local.**
   `src/vt/cpu/cpu_ops.cpp:125` — `static thread_local std::vector<float> af;`,
   resized to `nrows * k` where `nrows = min(16, chunk rows)` and 16 is ggml's
   `blck_1`. Nothing else in `MatmulChunked`, `MatmulOneChunk`,
   `MatmulOneChunkRef` or `vt::MatmulBT` allocates. At the shipped
   `k = 188160` that is `16 x 188160 x 4 = 12.04 MB` per worker, and at
   `hardware_concurrency() = 20` it is **241 MB in total**, not 26 GiB.
2. **`static thread_local` means it is allocated once per worker for the process
   lifetime and never shrinks** — the comment at `cpu_ops.cpp:122-124` says so.
   That is a retention property worth recording; it is not a per-call cost.
3. **The call site's `scaled` copy is pre-existing and unchanged by #1252.**
   `ltx2_text_encoder.cpp:460-465` builds one `std::vector<float>` the size of
   `normed` per `project()` call — at `B*T = 1024`, `flat = 188160` that is
   770.4 MB — and #1252's body records it as deliberately kept on both arms.
   A copy present in the *before* arm cannot explain a *regression*.
4. **The tree has no memory-bounded test at all.** `tests/vt/test_ops_matmul.cpp`
   and `test_ops_matmul_elem.cpp` assert values and byte equality; nothing
   anywhere asserts a peak-RSS or allocation bound. So the reported defect, had
   it been real, would have been invisible to every gate in the repository.

## Port map

| Piece | Where | Change |
|---|---|---|
| Attribution of #1286 | this spec, `## Risks/decisions` | new |
| Local A/B probe (scratch, uncommitted) | scratchpad | new |
| Memory bound for the shared GEMM seam | `tests/vt/test_ops_matmul_mem.cpp` | new |
| Test registration | `tests/CMakeLists.txt` | one line |

No `src/` change is planned. If the measurement contradicts the baseline above,
the fix is chosen from #1286's own list and this table is rewritten before any
code is written.

## Tests to port

None to port: there is no upstream test for a C++ threadpool's peak resident
footprint. The new test is original and its contract is stated where it lives.

It asserts the property the baseline claims and #1286 doubted: **peak RSS growth
across a `vt::MatmulBT` call does not scale with the whole intermediate per
worker.** Concretely, the growth in `VmHWM` attributable to the call, measured
at a large `K` where the per-worker tile is the dominant term, stays under the
documented `nthreads x 16 x K x 4` plus slack — and in particular is nowhere
near `nthreads x rows x K x 4`, which is the shape #1286 hypothesised.

It is red-first by mutation rather than by pre-existing failure, because there
is no defect to start red against. The mutation is the hypothesised defect
itself: widen `af` from the 16-row tile to the chunk's whole row span. Recorded
with `BUILT`, the compiler error count, `git diff --stat`, and the failing
assertion by name.

## Gates

- `test_ops_matmul_mem` — the new bound, plus its mutation.
- `test_ops_matmul`, `test_ops_matmul_elem` — the seam's existing value and
  byte-equality gates, unchanged and still green.
- `test_ltx2_text_encoder` — the caption projection's byte-exactness against the
  recorded goldens. **No tolerance is widened anywhere in this row**; the values
  must not move, and nothing here touches summation order.
- `test_ltx2_lora`, `test_ltx2_loader` — #1259's gates, for the sibling check.
- Full `ctest`, and `scripts/agent-preflight.sh --staged`.

Known-environmental and not chased: `test_engine_core_proc` and `test_async_llm`
starve under `ctest -j` and are re-run serially; `test_cpu_x86_llamacpp_floor`
is #618 under load; `windows-msvc-*` is red on every pull request.

## Dependencies

- The retained GB10 evidence under
  `/mnt/nas_share/rc/ltx25-fullmodel/out/` — both the #1252 run
  (`20260818T220620Z/1024x576-25f/`) and the pre-#1252 run
  (`1024x576-25f/`, 2026-08-18 17:20-18:11), which is what makes a paired
  comparison possible at all.
- `scripts/../runguard.py` on the NAS defines the columns; its
  `runguard.py:236-237,260` fix `used_gib = MemTotal - MemAvailable` and
  `anon_gib = AnonPages`, both **system-wide**, against `rss_gib`, which is the
  child's own. That distinction is what the attribution turns on.
- No GPU and no lease. Allocation sizes are architecture-independent, and the
  local arm is an x86 box.

## Work breakdown

| # | Work | Landable alone |
|---|---|---|
| W1 | Attribute #1286 against the retained paired evidence | yes |
| W2 | Local before/after A/B: peak RSS and throughput at the shipped geometry, swept over threadpool width | yes |
| W3 | Check #1259's profile and either fix in flow or file | yes |
| W4 | The seam's memory bound, red by mutation, green after | yes |

One pull request; no split case applies, and no developer answer is recorded
under `## Git integration`.

## Risks/decisions

**The instrument's own precondition is the first risk, and it is the one that
bites.** #1286 compares two *absolute* peaks of a *system-wide* column. That
comparison is only sound when both runs start from the same occupancy, and
nothing in the issue checks whether they did. This spec's `## Outcome` records
what the check found.

**A scaled probe is never presented as a full-geometry one.** Where the local
arm reduces `rows` or `out_features` to keep a single-threaded arm inside a
sensible wall time, the reduction is stated beside the number, and the
thread-scaling term it isolates is independent of `rows` by construction (`af`
is sized by the 16-row tile, not by the chunk's row count).

**Byte-exactness is not negotiable.** #1252's path is bit-exact against recorded
goldens and its review confirmed a test diff of 150 insertions and zero
deletions. If any candidate fix altered summation order — blocking the GEMM over
`rows` is the one that would — this row stops and puts the question here rather
than deciding it in code. No tolerance is widened to make a red green.

**A memory bound can be flaky.** Peak RSS depends on the allocator, so the test
is written to bound a *difference* at a geometry where the signal (12 MB per
worker) is far above malloc noise, rather than to pin an absolute number. If a
robust bound cannot be written, that is reported as a finding instead of being
weakened until it passes.

## Owed

- [#1317](https://github.com/mudler/vllm.cpp/issues/1317) — the harness compares
  absolute system-wide peaks across runs with different starting occupancy, and
  something held 26.8 GiB on `dgx:gpu0` before the run began. Both halves are
  outside this repository — `runguard.py` and the fleet's job hygiene — so
  neither is fixed in flow. It is what actually aborted the render.
- **No GB10 number.** Every measurement below is from a 20-core Zen 5 under KVM.
  Allocation sizes are architecture-independent, so the attribution carries; a
  throughput ratio does not, and none is claimed for GB10.
- **The seam's tile is retained for the process lifetime.** `af` is a
  `static thread_local` that grows to the largest `K` any GEMM on that worker
  ever saw and never shrinks. At `K = 188160` and 20 workers that is 232 MiB
  held until exit, on a box that may later be tight. It is bounded, deliberate
  and now measured; it is not repaired here, because the buffer exists precisely
  so a worker allocates once, and trading that away is a performance decision
  with its own spec.

## Now

`DONE`. The reported defect does not exist; the seam gained the bound whose
absence let it be believed.

## Outcome

### W1 — the attribution: the +26 GiB is the box, not the change

Both runs are retained, and that is what makes this answerable.
`runguard.py:236-237,260` fixes the compared column: `used_gib` is the
**system-wide** `MemTotal - MemAvailable`, `anon_gib` is system-wide
`AnonPages`, and only `rss_gib` is the child's own.

| | pre-#1252 `1024x576-25f/` | #1252 `20260818T220620Z/1024x576-25f/` |
|---|---:|---:|
| `used_gib` at `t=0` | **4.741** | **31.553** |
| `avail_gib` at `t=0` | 114.890 | 88.078 |
| peak `used_gib` | 79.206 at t=1867.4 s, `cpu=1885.5%` | 105.853 at t=199.2 s, `cpu=1925.2%` |

The box carried **26.812 GiB** before `ltx2-gen` started in the second run.
#1286's regression is `105.853 - 79.206 = 26.647 GiB`. The two agree to
**0.165 GiB**.

Each run's own demand — peak minus its own `t=0` — is the same:

| axis | pre-#1252 | #1252 | delta |
|---|---:|---:|---:|
| `used_gib` peak minus own `t=0` | **74.465** | **74.300** | **-0.165** |
| `anon_gib` peak minus own `t=0` | 38.012 | 38.217 | +0.205 |
| child `rss_gib` at the peak sample | 41.952 | 42.090 | +0.138 |

Three columns, three instruments, all inside ±0.25 GiB, and both peaks sampled
inside a ~1900% CPU stretch rather than in two different phases. On a box as
clean as the first run's, the #1252 binary's own 74.300 GiB would have left
**40.59 GiB** available — above the 12 GiB hard floor and the 8 GiB projection
floor, and consistent with the "never below ~40 GiB" the pre-#1252 runs showed.

Two further readings from the same evidence, recorded because each one on its
own would have been read as supporting #1286:

- During its single-core projection the pre-#1252 run sits **dead flat** at
  `anon = 34.35`, `rss = 33.56`, `used = 74.50` from t=124 s to t=1801 s. The
  #1252 run's threaded stretch sits at `anon = 34.30`, `rss = 30.35`,
  `used = 97.40`. **The same `anon` to 0.15%** — with `used` 23 GiB apart.
- The pre-#1252 run reached t=3017.3 s and was stopped from outside
  (`# TERMINAL signal=15 (supervisor asked to stop)`), still inside
  conditioning with `dit_runs=0`. Neither run ever loaded the DiT, so neither
  peak is a whole-render peak, and neither is presented as one.

Filed as [#1317](https://github.com/mudler/vllm.cpp/issues/1317).

### W2 — the local A/B: the seam costs 0.24 GiB and keeps the 8.4x

Same probe source and flags on both arms, only the `Linear` body differing, at
the shipped `K = 188160` and `out_features = 4096`; peak RSS is `VmHWM` read
before and after the call with every operand already allocated and touched.
20-core Zen 5 under KVM, Release, `-ffp-contract=off`, box **not idle**
(loadavg 5.4-13.3), three replicates, median:

| arm | wall | rate | peak-RSS growth |
|---|---:|---:|---:|
| before, scalar `double` loop | **28.488 s** | 1.732 GMAC/s | **1.0 MiB** |
| after, `vt::MatmulBT` | **3.378 s** | 14.60 GMAC/s | **232 MiB** |

**8.43x, and +231 MiB.** The speedup reproduces #1252's 8.57x; the memory cost
is **0.85% of the 26.6 GiB #1286 attributes to it**, i.e. 117x too small to be
the reported defect.

Swept over threadpool width at the same geometry, the growth is exactly the
per-worker tile and nothing else:

| workers | 1 | 2 | 4 | 8 | 16 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| peak-RSS growth (MiB) | 12.9 | 24.7 | 47.8 | 93.8 | 186.1 | **232.1** |
| per worker (MiB) | 12.9 | 12.3 | 12.0 | 11.7 | 11.6 | 11.6 |

The model predicts `16 x 188160 x 4 = 11.48 MiB` per worker. The measurement
lands on it, and the output checksum is **byte-identical across all six thread
counts**, which is the dispatch determinism contract holding.

At the **full shipped geometry**, `rows = 1024`, both projections, default 20
workers (box heavily loaded, loadavg 15.9 then 28.5, so the wall times are not
a speed claim):

| projection | wall | rate | peak-RSS growth |
|---|---:|---:|---:|
| `1024 x 188160 x 4096` | 57.34 s | 13.76 GMAC/s | **247 MiB** |
| `1024 x 188160 x 2048` | 28.95 s | 13.63 GMAC/s | **239 MiB** |

`rows` moved 16x between the two sweeps and the growth moved by 15 MiB — which
is the output buffer, `1024 x 4096 x 4 = 16.8 MB`, allocated inside the timed
region. The tile does not scale with rows, as designed.

**So the answer to "which of the three" is none of them.** Not the threaded
kernel: 232 MiB at 20 workers. Not the call site: its `scaled` copy is 770 MB
and is present on **both** arms, so it cannot be a regression. Not an
interaction. The measurement was confounded by its baseline.

### W3 — #1259 has the same profile and needs nothing

`Ltx2FuseLoraIntoTensor` routes through `vt::Matmul`, the other member of the
same seam, and `MatmulOneChunk` is one template shared by both orientations —
so the per-worker buffer is `16 x K x 4` there too, with `K = rank`. At the
shipped `4096 x 450 x 4096`, measured with the same allocation counter the new
gate uses:

| workers | 1 | 2 | 4 | 8 | 16 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| bytes requested | 28,864 | 28,864 | 86,464 | 201,664 | 432,064 | **547,264** |

**0.52 MiB at 20 workers**, exactly `19 x 28,800 + 64`. `bs` and `agg` are
unchanged by #1259. It cannot reach the wall #1286 describes, so nothing is
fixed and nothing is filed for it.

### W4 — the bound, and what looking for it exposed

`tests/vt/test_ops_matmul_mem.cpp`. The refutation does not repair the thing the
attribution found: **no gate in this tree bounds the seam's memory.** Every GEMM
gate asserts values or byte equality, so a kernel that allocated a whole
intermediate per worker would have passed all of them on every model until a box
ran out. That is why the bound lands here even though the reported defect is not
real.

**Peak RSS could not carry the gate, and finding that out was the useful part.**
The first draft measured `VmHWM` around `/proc/self/clear_refs`. It read
`growth_bytes = 0` for **every** thread count and **every** row count while
passing every bound — because the operands are freed between measurements,
glibc keeps the arena, and the next tile is served from resident pages. A mute
switch that reports green over a kernel doing anything at all. The gate is
therefore a replaced global `operator new` counting what the seam **asks for**,
which no allocator policy can silence, with a liveness case beside it requiring
the counter to see at least six of eight fresh workers take a tile.

Green, at `K = 65536` so one tile is exactly 4 MiB:

| workers | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| bytes requested | 64 | 4,194,368 | 12,582,976 | 29,360,192 |

`(nthreads - 1)` tiles, because the caller thread's `thread_local` is already
sized. Rows 64 vs 1024 at 4 workers: **12,582,976 both times**, difference 0.

**Red first, by mutation, and the first attempt did not build.** M1 widens `af`
from the 16-row tile to the whole activation — #1286's own hypothesis, written
into the kernel.

- **First attempt: `BUILT=NO`, `compile_err=2`**,
  `cpu_ops.cpp:134:19: error: unused variable 'nrows' [-Werror=unused-variable]`.
  Recorded because a mutation that fails to build reads exactly like a passing
  test.
- **Second attempt: `BUILT=YES`, `compile_err=0`**, `git diff --stat` =
  `src/vt/cpu/cpu_ops.cpp | 2 +-, 1 file changed, 1 insertion(+), 1 deletion(-)`.
  `test_ops_matmul_mem` **exit 1**, `Status: FAILURE!`,
  `2 test cases | 1 passed | 1 failed`, `13 assertions | 7 passed | 6 failed`.
  Failing by name: `CHECK(growth <= Bound(nthreads))` at three worker counts —
  `67,108,928 <= 25,165,824`, `201,326,656 <= 41,943,040`,
  `469,762,112 <= 75,497,472` — and all three of the row-scaling assertions,
  with `rows=1024` reading **1,073,741,888 bytes**, a gibibyte of exactly the
  shape #1286 supposed.
- Restored byte-for-byte: `sha256(cpu_ops.cpp) =`
  `dc39eccdece48879e82be7209e95d182c6ed624eb04389de689fa4b58fe4f1f3` before and
  after. Rebuilt (binary mtime moved 07:15:48 -> 07:16:27, so the green is not a
  stale binary) and green again: **2 cases, 13 assertions, 0 failed, exit 0**.

M1 mutates **`src/vt/cpu/cpu_ops.cpp`**, product code reached from
`ModelRegistry::Forward` and from the LTX-2.5 text tower, which is what makes
the red evidence that the gate measures shipped behaviour rather than a
test-local copy. This row adds no product code, so there is no production call
site to delete — `reachability.md` answers that case directly.

### What is not claimed

- No GB10 measurement of any kind, so #1252's unmeasured per-core ratio `R` is
  untouched and the LTX-2.5 speed axis stays where it was.
- The bound is a **CPU** bound. `vt::MatmulBT` on CUDA/ROCm/Vulkan is not
  covered, and their allocation behaviour is not measured here.
- Nothing establishes what would happen on a full-model render now, because no
  full-model render has been rerun. The claim is that #1252 is not what stopped
  the last one.
