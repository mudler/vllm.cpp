# ROCm chunked H2D through a bounded pinned bounce ring

Row: `MODEL-MM-QWEN4-EXP`
Issue: `ISSUE-LOCAL-01M2BZ5QK4XRETK48CXKSHKRDW` (row-owned, deferred by
`.agents/specs/rocm-host-residency-after-upload.md` § "Deferred, with an issue",
and made REQUIRED by that spec's §6a measurement)

## 1. The defect

`RocmBackend::Copy` is one call:

```cpp
void Copy(Queue& q, void* dst, const void* src, size_t bytes) override {
  Check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, AsStream(q)), "hipMemcpyAsync");
}
```

(`src/vt/rocm/rocm_backend.hip:278-280` at `98e2cd7da`.)

On `strix:gpu0` (gfx1151) that call is handed a **multi-GiB, pageable,
file-backed, CIFS-backed** source: a `PROT_READ MAP_PRIVATE` view of a GGUF
shard on `//192.168.68.102/Data`. To DMA out of a pageable range the ROCr
runtime has to make that range resident and describe it to the KFD, and
`svm_range_set_attr` is where that work lands.

`.agents/specs/rocm-host-residency-after-upload.md` §6a measured the
consequence. The 67.56 GiB `Qwen3.8-Flash-Next UD-IQ1_S` loads with zero op
refusals, stages 29.69 GiB onto the board, and then never produces a token: two
1200 s runs, killed at the deadline, with the compute thread uninterruptible in
`svm_range_set_attr` for 153 of 196 and 148 of 198 `wchan` samples. Releasing
the spent source pages after upload (that spec's fix 1) and stopping the
prefault (fix 2) both landed and neither moved it: host `RssFile` during the
forward is still 20.2-20.4 GiB. That spec's §7 fourth risk names this outcome
and names this change as the consequence.

So the remaining suspect is the SHAPE of the transfer, not the residency of
what it reads.

## 2. Oracle

`llama-cpp` pin `10bf611e5` (b10451), the recorded pin in
`.agents/oracles/llama-cpp.md`. The upload ring is in
`src/llama-model-loader.cpp`:

- `:1440` — `constexpr size_t n_buffers = 4;`
- `:1449` — `const size_t buffer_size = alignment != 1 ? 64 * 1024 * 1024 + 2 * alignment : 1 * 1024 * 1024;`
  under the comment "Buffer size: balance between memory usage and I/O
  efficiency / 64MB works well for NVMe drives".
- `:1496-1516` — allocates `n_buffers` buffers from
  `ggml_backend_dev_host_buffer_type(dev)` (pinned host memory) and one
  `ggml_backend_event_t` per buffer.
- `:1591-1642` — the loop. For each chunk: `ggml_backend_event_synchronize` on
  the buffer about to be reused, fill the buffer, `ggml_backend_tensor_set_async`
  from the pinned buffer to the device, `ggml_backend_event_record`, advance
  `buffer_idx` modulo `n_buffers`.
- `:1664` — the buffers are freed once, after the whole load.

Peak PINNED host residency is therefore `n_buffers * buffer_size` for any model
size. Ours is O(the largest single weight) of pageable range that the driver has
to pin and describe, on every weight.

**ONE HONEST DIFFERENCE, STATED RATHER THAN GLOSSED.** The oracle takes this
ring only when it is NOT using mmap (`:1456-1458`, `if (use_mmap ||
check_tensors) return nullptr;`); with mmap it calls `ggml_backend_tensor_set`
straight from the mapped pointer, which is the same single-shot pageable copy we
do today. So this is not a case where the oracle refuses the shape we ship. What
the oracle supplies is the SHAPE — ring size, buffer size, event discipline, and
the fact that a bounded pinned ring is the sanctioned way to feed a device out of
host storage the device cannot address. We apply it to a source the oracle
reaches by `read()` and we reach by `memcpy` out of a mapping. That adaptation is
the design, and it is why this is its own spec rather than a port.

## 3. Design

A ring of `N` fixed pinned buffers inside `RocmBackend`, lazily created on the
first copy that qualifies.

```
for (off = 0; off < bytes; off += n) {
  n = min(chunk, bytes - off);
  i = next slot
  if (slot[i] has an in-flight chunk) hipEventSynchronize(slot[i].event);
  memcpy(slot[i].host, (const char*)src + off, n);
  hipMemcpyAsync((char*)dst + off, slot[i].host, n, hipMemcpyHostToDevice, stream);
  hipEventRecord(slot[i].event, stream);
}
```

The call still returns with work in flight on the stream, so `Backend::Copy`'s
asynchronous contract is unchanged. What changes is that the SOURCE is fully
consumed by the time `Copy` returns, which is strictly stronger than today.

### 3a. N and the chunk size, justified against the oracle

- **`N = 4`.** The oracle's `n_buffers` at `llama-model-loader.cpp:1440`, taken
  rather than re-derived. Four slots keep one buffer filling while up to three
  drain, which is what a ring buys over a double buffer, and four is the number
  a runtime that loads models for a living ships.
- **`chunk = 64 MiB`.** The oracle's `buffer_size` at `:1449`. We drop its
  `+ 2 * alignment` term, deliberately: that term exists to absorb the
  `read_alignment()` padding of an `O_DIRECT`-style file read (`:1604-1631`
  computes `aligned_offset`, `read_start`, `read_end` and trims the padding back
  off). Our source is an in-memory mapping and our chunk boundaries are exact, so
  there is no padding to absorb and a padded buffer would only waste pinned
  bytes.
- **Total pinned residency: `4 * 64 MiB = 256 MiB`, for any model size.** That
  is the oracle's bound, unchanged.
- **Threshold = one chunk (64 MiB).** Below one chunk the ring degenerates to
  `memcpy` + `hipMemcpyAsync` with no overlap, no second slot ever used, and no
  residency benefit that the driver's own internal staging does not already
  give — it is strictly one extra copy of the bytes. A 4 KiB norm weight must
  not pay a bounce, and at this threshold it does not: it takes the existing
  single call, byte for byte. This boundary is not invented — it is where the
  oracle's own `while` loop stops having more than one iteration.

`VT_ROCM_PINNED_H2D_MIB` overrides the chunk in MiB and `0` disables the path
entirely, restoring `98e2cd7da` behaviour in the same binary. The A/B has to be
available in one binary because the model gate is a 20-minute run on a leased
box and rebuilding between arms is how a build difference gets to masquerade as
the effect.

### 3b. THE REACH, ENUMERATED BEFORE THE CODE

`Backend::Copy` is a shared seam. This change is confined to ONE override of it.
Files touched in `src/` and `include/`:
`src/vt/rocm/rocm_backend.hip` and the new HIP-free
`include/vt/rocm/rocm_pinned_h2d.h`. No other backend's `Copy`, no model, no
loader, no layer.

**Which backends change: ROCm, and only ROCm.** `cuda_backend.cu`,
`cpu_backend.cpp`, `metal_backend.mm`, `vulkan_backend.cpp`, the XPU backend and
the Tenstorrent backend are not edited and their `Copy` is byte-identical. CUDA
is deliberately excluded: `dgx`, `thor` and `orin` produce tokens today, so
nothing there is owed this, and widening to CUDA would put a new host
`cudaEventSynchronize` on a path four measured campaigns depend on. A CUDA arm
is a later row if a CUDA measurement ever asks for one.

**Which ROCm traffic changes.** The staged path is taken only when ALL FIVE
hold, and anything else takes the existing single call unchanged:

1. `bytes >= chunk` (64 MiB by default).
2. `dst` resolves to DEVICE memory. D2H and H2H never bounce, so the sampler's
   pinned-host download and every readback are untouched.
3. `src` is UNREGISTERED host memory. D2D never bounces. A source that is
   already pinned (`hipMemoryTypeHost`) never bounces, because it is already
   DMA-able and the ring exists only to create that property. A MANAGED source
   (`hipMemoryTypeManaged` / `Unified`) never bounces either, because it is
   already device-addressable — which is exactly the `VT_ROCM_MANAGED_ALLOC=1`
   configuration, so that knob's behaviour is unchanged.

   **THIS TERM WAS TRUE OF THE DECISION AND FALSE OF THE PROCESS, AND A REVIEW
   CAUGHT IT.** The first implementation evaluated `ring_available =
   EnsureRing(chunk)` BEFORE `ShouldStageH2D`, so a managed destination still
   built the ring: `VT_ROCM_MANAGED_ALLOC=1` on gfx1151 measured
   `staged=0 direct=3 chunks=0 ring_bytes=268435456`, and the device case
   printed that line while asserting nothing about it. The copy was direct, as
   this term promises, and 256 MiB of pinned host memory was allocated anyway --
   on exactly the boards that never stage, and out of the resource
   `.agents/environment.md` measures as gfx1151's real ceiling. `EnsureRing` is
   now the LAST term evaluated and the managed arm of the device case asserts
   `ring_bytes` is unmoved.
4. The stream is NOT capturing a graph. `hipEventSynchronize` inside a capture
   region aborts the capture. The capture contract documented at
   `rocm_backend.hip:334-348` already forbids "host<->device blocking copies"
   inside the region, so a qualifying copy in there is already a contract
   violation — but it would previously have been a silent one and would now be a
   loud one, and changing WHICH failure a misuse produces is still a change. The
   guard keeps capture byte-identical.
5. The ring resolves (`VT_ROCM_PINNED_H2D_MIB != 0` and the pinned allocation
   succeeded). A failed `hipHostMalloc` falls back to the existing single call
   rather than failing the load: 256 MiB of pinned memory is not worth refusing
   a model over.

**So what actually takes it.** Every H2D weight upload of 64 MiB or more on an
AMD board. That is a WIDER set than the release in
`rocm-host-residency-after-upload.md` §4a, and saying so is the point of this
paragraph: that release needed `mmap_fd >= 0`, so it fired only on the GGUF
keep-quant borrow. This predicate asks only "unregistered host source, big
enough", so it ALSO fires on safetensors borrows and on any heap buffer a loader
hands to `Copy`. On ROCm the families that reach it are therefore §4a's five
(Qwen4-Exp, GLM-MoE-DSA, GLM5-Next, Muse-Glimmer, the Qwen3.5 DFlash draft
head's rebound embedding table) PLUS every safetensors model whose weights clear
64 MiB — gemma, phi, minicpm, olmo2, stablelm, commandr, deepseek_v2, dots3,
nemotron_h, qwen3_vl and the rest — PLUS the EXL3 device loader's trellis
uploads, which §4a excluded for want of `mmap_fd`. The behaviour those models
see is identical bytes, bounded pinned host residency, and one extra host
`memcpy` per 64 MiB.

**What is NOT covered by a test, and is recorded rather than claimed.** The only
family this change is MEASURED on is Qwen4-Exp, on one board. The device case in
§5 enters `RocmBackend::Copy` directly with a large pageable source, which is the
seam every one of those families reaches, so the CALL SITE is gated; no family
but Qwen4-Exp gets an end-to-end run here, for the same reason
`rocm-host-residency-after-upload.md` §5 gives — there is no harness that can
drive a second family's production entry point on a device.
`ISSUE-LOCAL-01M2CKN5516AKE7W2JVDV86Z8X` already owns that gap and this change
does not narrow it.

### 3c. Where the decision lives

The PURE parts — the chunk plan, the ring-reuse order, and the predicate that
decides whether to stage — go in `include/vt/rocm/rocm_pinned_h2d.h`, free of
HIP headers, and are table-tested in the ordinary CPU build. That mirrors
`include/vt/rocm/rocm_arch.h` and `ResolveMemoryPolicy` exactly, and for the
reason that header states about `CapabilityFromGcnArch`: the piece a wrong
answer breaks silently is the piece that must be gated on a runner with no AMD
GPU. `rocm_backend.hip` reads two `hipPointerGetAttributes`, one
`hipStreamIsCapturing`, and calls it.

### 3d. The ring is deliberately never freed

The pinned slots and events are allocated on first use and leaked at process
exit. `RocmBackend` instances live in a function-local `static
std::vector<std::unique_ptr<RocmBackend>>` in the registrar, so a destructor
would run during static destruction, and calling `hipHostFree` / `hipEventDestroy`
after the HIP runtime has begun tearing down is a hazard this file does not
have today (`exec_`, the `hipGraphExec_t`, is likewise never destroyed). 256 MiB
returned to the OS at `exit()` buys nothing and a teardown-order crash costs a
measurement. Stated here because "it leaks" must be a decision on the record,
not something a reader discovers.

## 4. Tests — red first

Commit order is the red: the test commit lands before the implementation commit,
so the failing run is reproducible by building the tree at the test commit.

**`tests/vt/test_rocm_pinned_h2d.cpp`** — new, UNCONDITIONAL (no HIP needed),
registered beside `test_rocm_arch`. Drives the pure header with fakes:

1. **The plan.** `bytes = 200 MiB`, `chunk = 64 MiB` gives 4 chunks of
   64/64/64/8 MiB, contiguous offsets summing to `bytes`, and no chunk larger
   than `chunk`. Mutating the chunk size to the whole buffer produces 1 chunk of
   200 MiB and fails this.
2. **The ring waits before it reuses.** With `N = 4` and 9 chunks, slot 0 is
   waited on before chunks 4 and 8 and NOT before chunk 0. The fake records the
   wait/stage/enqueue order; deleting the wait fails it.
3. **The bytes survive.** The fake stage/enqueue pair reassembles the output and
   it is `memcmp`-identical to the input, over a size that is NOT a multiple of
   the chunk.
4. **The predicate truth table.** Staged only for {unregistered host src, device
   dst, `bytes >= chunk`, not capturing, `chunk != 0`}. Every other row of the
   table is direct. Deleting the size term, the src-kind term, the dst term or
   the capture term each flips a row.

**`tests/vt/test_backend_cross_device.cpp`** — one new case, which is the
REACHABILITY conviction. It enters `RocmBackend::Copy` through
`vt::Backend&`, the production seam, with a 200 MiB pageable `std::vector`
source and a device destination, and asserts:

- the downloaded bytes are `memcmp`-identical to the source (bit-exactness is
  this file's declared bar for a pure copy path);
- `PinnedH2DSnapshot()` shows the copy took the ring — `staged_copies` grew by
  one and `chunks` by four — and that `max_chunk_bytes <= 64 MiB`, which is the
  bounded-host-residency assertion in the form this harness can actually make;
- a SMALL copy on the same backend grows `direct_copies` and not
  `staged_copies`, so the threshold is gated in the same case.

Deleting the staged branch from `rocm_backend.hip` leaves `staged_copies` at 0
and fails this case. That is the mutation that proves the wiring, and it is the
one `AGENTS.md` "Nothing lands dead" asks for.

This case measures nothing on a build with no ROCm device, exactly like every
other case in that file, and the gate line below therefore names the board it
was run on.

## 5. Gates

Every selector names its binary and prints its case and assertion counts. This
row has already shipped two selectors that matched nothing and reported
`Status: SUCCESS!`; a selector that matches nothing is indistinguishable from
one that passes.

Measured on `strix:gpu0` (gfx1151, ROCm 7.2.4) under `rc` job
`e8bf3b66-eb80-4659-8e6e-48167eb6bf60`, from a clean clone built in
`/tmp/vllmcpp-chunked-h2d` with
`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_HIP=ON`
`-DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 -DROCM_PATH=/opt/rocm`
`-DVLLM_CPP_BUILD_TESTS=ON`, `ninja -j 4`, run with
`LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib` — without which the binary exits 127
having measured nothing.

| binary | selector | cases | assertions | result |
|---|---|---|---|---|
| `test_rocm_pinned_h2d` | (none — whole binary) | 9 | 84 | SUCCESS |
| `test_backend_cross_device` | `-tc=*pinned bounce*` | 1 | 8 | SUCCESS |
| `test_backend_cross_device` | (none — whole binary) | 61 | 84841 | SUCCESS |
| `test_backend_cross_device` | `-tc=*DSA*` | 2 | 273 | SUCCESS |

### 5a. THE REPAIR WAVE, AND THE ONE ASSERTION THAT WAS MISSING

A fresh review returned FAIL on the eager `EnsureRing` (§3b term 3) and on the
dead `stream_capturing` field (§6). Both are repaired above. The gate was re-run
on `strix:gpu0` (gfx1151, ROCm 7.2.4) in the same `/tmp/vllmcpp-chunked-h2d`
clone, `LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib`, evidence archived at
`/workspace/vtchunked/repair-20260913-073233/` (`rc` logs age out within a day,
so the share holds them).

**The SHAs below are the ones the worker checked out and the evidence names.**
The branch was rebased onto `origin/main` twice afterwards, so `c794b5dda` is
now `d83c4b1c6` and `33a1eaaf8` is now `a6621c937`. The four files the gate
reads -- `include/vt/rocm/rocm_pinned_h2d.h`, `src/vt/rocm/rocm_backend.hip`,
`tests/vt/test_rocm_pinned_h2d.cpp` and
`tests/vt/test_backend_cross_device.cpp` -- are byte-identical across both
rewrites, and neither `43622bc37` nor `ee0644eab` touches any of them, so the
measurement carries.

**RED FIRST, and the selector that produces it is the one nobody had run.** At
the test commit `c794b5dda` — the new managed-arm assertion, no fix —
`VT_ROCM_MANAGED_ALLOC=1 test_backend_cross_device -tc='*pinned bounce*'`
reported **1 case, 0 passed, 1 failed / 4 assertions, 3 passed, 1 failed**,
`Status: FAILURE!`, exit 1, and printed
`pinned H2D: managed_alloc=1 staged=0 direct=3 chunks=0 max_chunk=0 ring_bytes=268435456`
with `pinned H2D case ran on a ROCm board: 1`. That is the defect: a ring
allocated, never used, on the arm that never stages. Binaries
`fc26b58276f3e6f9a2cda2a550b0596a` (`test_rocm_pinned_h2d`) and
`3cc2b9a5a58c76f02807dd6ad42ac007` (`test_backend_cross_device`).

**GREEN AFTER, at the fix commit `33a1eaaf8`**, binaries proven changed:
`eba82c4b2d73198179ebf29a0fd37461` and `45bacea1a99edb77d508f590532cb03d`. Every
selector prints its counts and its board line.

| binary | selector | cases | assertions | result |
|---|---|---|---|---|
| `test_rocm_pinned_h2d` | (none — whole binary) | 10 | 2007 | SUCCESS |
| `test_rocm_pinned_h2d` | `-tc=*cheap terms*` | 1 | 1923 | SUCCESS |
| `test_backend_cross_device` | `-tc=*pinned bounce*` | 1 | 8 | SUCCESS |
| `test_backend_cross_device` | `-tc=*pinned bounce*`, `VT_ROCM_MANAGED_ALLOC=1` | 1 | 4 | SUCCESS, `ring_bytes=0` |
| `test_backend_cross_device` | (none — whole binary) | 61 | 84841 | SUCCESS |
| `test_backend_cross_device` | `-tc=*DSA*` | 2 | 273 | SUCCESS |

The unit binary grew by one case, "the cheap terms are exactly the decision
minus the allocating term", which walks 480 inputs.

**A review measured that case's FIRST version as a tautology, and it is
replaced.** That version asserted `StagingTermsExceptRing(in) ==
ShouldStageH2D(in)` with `ring_available` held true. The decision is DEFINED as
the helper AND that flag, so the assertion is `X == (X && true)` -- true for any
definition of the helper, deleted terms included. Measured on `strix:gpu0`:
deleting `dst == kDevice` from the helper, and separately deleting
`bytes >= chunk_bytes`, each left that case 1443/1443 SUCCESS. Both mutants died
to the PRE-EXISTING five-term truth table instead (3 and 2 failed assertions
respectively, elsewhere in the binary), so this spec's earlier sentence -- that
deleting `dst == kDevice` "fails it (3 assertions)" -- attributed a whole-binary
count to the case just added. That is the `touched the symbol is not caused the
failure` error, and the sentence is WITHDRAWN.

The case now computes the expected answer in the test file from the four
inputs, as a sequence of refusals rather than as a conjunction, and checks BOTH
expressions against it, so a term deleted from either one fails the case itself.
It is run off-board because the header is HIP-free.

**RE-MEASURED, and the new case now convicts.** Same box, same clone
(`/tmp/vllmcpp-chunked-h2d`), `rc` job `ecf7b0b6-6e6f-4123-b6d2-50d5178a32dd`,
at `8e1f0ac03`. Clean unit binary `32b4cade5a4bba7d2060f6086a681dde`, 10 cases /
2007 assertions SUCCESS, and the case alone 1 / 1923 SUCCESS.

| mutation in `StagingTermsExceptRing` | binary md5 | `-tc=*cheap terms*` | whole unit binary |
|---|---|---|---|
| delete `in.dst == PtrKind::kDevice &&` | `c7f00ede469d373502f961347e5b7361` | **FAILURE**, 1 case failed, first failed assertion at `test_rocm_pinned_h2d.cpp:363` | FAILURE, 2 cases / 4 assertions failed |
| delete `in.bytes >= in.chunk_bytes` | `c1e2d3d36df481a2cc49c7f091c1210b` | **FAILURE**, 1 case failed, same line | FAILURE, 2 cases / 3 assertions failed |

Both restored builds hash back to `32b4cade5a4bba7d2060f6086a681dde` and the
restored tree runs 10 / 2007 SUCCESS, which is what proves the restoration
rather than `git status`. A `REQUIRE` aborts its case, so a mutant run reports
fewer assertions than the clean one; the verdict is the failure, not the count.
The cross-device binary is unmoved at
61 / 84841: the repair adds one assertion to an arm that is SKIPPED in the
default configuration, which is why the managed selector had to be run
explicitly and is now a declared gate line.

**THE MODEL STILL PRODUCES TOKENS AFTER THE REPAIR**, confirmed once rather
than re-measured, because §7's result is accepted and the staging decision is
unchanged for the arm that stages. At `33a1eaaf8`, `/tmp/ckpt-iq1s` (worker-local
shards), `--device auto --max-tokens 32 --temperature 0 --max-num-seqs 1
--repeat 3`: three runs, each `finish_reason=length completion_tokens=32`, at
42.399 s / 6.086 s / 6.074 s, decode **5.258 and 5.269 tok/s**, exit 0. Evidence
at `/workspace/vtchunked/repair-model-20260913-073358/`. `vllm-cli` reports md5
`364f11fddc6392feb79cdc23f8d69cf5`, which is byte-identical to §7's — and that
is NOT a stale build: the executable is a 26,720-byte shim and the code links
through `libvllm.so.0.0.3`, relinked at 07:33:59 from a `rocm_backend.hip.o`
rebuilt at 07:32:52 on the fix commit.

Baseline at `98e2cd7da`: `test_backend_cross_device` whole binary **60 cases /
84833 assertions**, `-tc=*DSA*` **2 / 273**. The whole binary therefore grew by
exactly one case and eight assertions, which is this spec's case and nothing
else, and `-tc=*DSA*` is unmoved.

**RED, at the test commit `e1bf7fd1d`, same binary, same board.**
`-tc=*pinned bounce*` reported **1 case, 0 passed, 1 failed / 8 assertions, 3
passed, 5 failed**, `Status: FAILURE!`, exit 1, and printed
`staged=0 direct=0 chunks=0 max_chunk=0 ring_bytes=0`. At the implementation
commit `81f91b600` the same selector on the same board printed
`staged=1 direct=2 chunks=3 max_chunk=67108864 ring_bytes=268435456` and
`Status: SUCCESS!`. Both runs printed `pinned H2D case ran on a ROCm board: 1`,
so neither is the shape where a selector matched nothing.

**MUTATIONS, on the board, each with the binary proven changed by md5.** The
clean implementation build is `88efb430a22c65c4` (`test_backend_cross_device`)
and `fc26b58276f3e6f9` (`test_rocm_pinned_h2d`).

| mutation | binaries | device selector | unit binary |
|---|---|---|---|
| delete `if (StagedCopy(...)) return;` from `RocmBackend::Copy` | `006c453cce2b25fb` / `8f82b9f94db909aa` | FAILURE, 4 of 8 assertions failed, `staged=0 direct=3 chunks=0 ring_bytes=0` | n/a |
| widen the chunk to the whole buffer | `74949cfa593d0843` / `f3720497fdbe96ca` | FAILURE, SIGSEGV (exit 139) before any assertion | FAILURE, 10 of 19 assertions failed, then SIGSEGV |

The second mutation convicts by CRASH rather than by assertion, and that is
reported as what it is rather than dressed up: with the chunk widened, both the
fake ring's slot buffers and the real pinned slots are one chunk long and the
copy reads past them, so the process dies. It is still caused by the mutation and
still red, and the arm that matters — the first one — fails cleanly on the
instrument.

**The tree was restored byte for byte, and the proof is the binary, not
`git status`.** The rebuild after both mutations produced md5
`88efb430a22c65c4` and `fc26b58276f3e6f9` — identical to the clean
implementation build — and re-ran 9/84 and 1/8 green. `git status` reported one
dirty path after the second mutation; that is the `core` file the SIGSEGV
dumped into the checkout, not a source edit, and the identical binary hashes are
what settle it.

Eight further mutations of the pure header were run off-board before the branch
was pushed, each convicting and each with a distinct binary md5: the chunk size
widened, the slot wait deleted, each of the four predicate terms deleted, the
cross-call in-flight state reset, and the knob parse made to answer 0 on a typo.

```sh
python3 scripts/check-agent-record.py
python3 scripts/check-commit-style.py --range origin/main..HEAD
python3 scripts/check-commit-trailers.py --range origin/main..HEAD
python3 scripts/check-pr-size.py --base origin/main --head HEAD \
  --branch row/MODEL-MM-QWEN4-EXP-ROCM-CHUNKED-H2D
```

**The model gate is the deliverable.** `/workspace/ckpt/qwen4exp-flash-next-iq1s`
shard 1, `examples/vllm-server` with `--device auto`, `VT_ROCM_MANAGED_ALLOC`
unset, on `strix:gpu0` inside an `rc` lease. Does it produce a token? If it does,
that is this row's G3 and the numbers follow — TTFT, prefill and decode tok/s,
load seconds, peak host `VmHWM` / `RssFile` / `RssAnon`, device memory from
`/sys/class/drm/card*/device/mem_info_vram_used` (`rocm-smi` is not on `PATH` in
the leased container), the exact build and run recipe, revisions, artifact sizes,
environment and contention. Generation is repeated at least three times and
reported as a spread, because gfx1151 fails about two of five identical greedy
runs with an illegal GPU memory access
(`ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD`) and one green there is unreplicated.
If it does not produce a token, NO number is recorded and the `wchan`
distribution is reported instead, including whether it moved off
`svm_range_set_attr`.

The artifact's compiled feature set is asserted before it is timed: `ldd` for
`libamdhip64.so.7` and the HIP arch it was built for.

## 6. Risks

- **The stall survives this too.** Then the trigger is neither the residency of
  the source nor the shape of the transfer, and the next hypothesis is the
  allocation side — 29.69 GiB of `hipMalloc` against this board's
  `hipMemGetInfo` total of **96.000 GiB** (`.agents/environment.md:89-92`;
  33,270,497,280 B is the box's HOST RAM, not its VRAM carve), or the CIFS
  mount, which §6a already named as an unseparated confound.
  A negative result with a `wchan` distribution is the reportable outcome, not a
  failure of the change.
- **An extra `memcpy` per 64 MiB slows the load.** Load seconds are recorded on
  both arms of `VT_ROCM_PINNED_H2D_MIB` in one binary, so the cost is measured
  rather than argued.
- **256 MiB of pinned memory on a 31 GiB host.** It is allocated once, it is the
  oracle's own bound, and it replaces an unbounded pinned range the driver was
  creating per copy.
- **A qualifying copy inside a graph capture.** Guarded by term 4, and the guard
  is in the truth table. **A review found that this was only half true of the
  first implementation.** `in.stream_capturing` was hardcoded `false` and a
  separate early `return false` did the work, so the term the truth table gates
  was a value production never supplied -- the program was safe and the
  guarantee was in the wrong place. The probe now feeds the field and the
  predicate IS the guard.

  **WHAT IS STILL NOT MEASURED ON A BOARD, STATED RATHER THAN GLOSSED:**
  `StreamIsCapturing`'s own return value. No case in
  `tests/vt/test_backend_cross_device.cpp` opens a capture region, and one was
  considered and NOT written, because a pageable asynchronous H2D is itself
  illegal inside a capture region -- such a case would measure HIP's refusal
  rather than this guard, and a green would not distinguish the two. The term
  it feeds is gated in the truth table; the probe is not. That is the honest
  extent of it.
- **`hipPointerGetAttributes` leaves a sticky error for an unregistered host
  pointer.** It returns `hipErrorInvalidValue` for one, which is the very answer
  we want, and the last error is cleared with `hipGetLastError()` immediately so
  it cannot poison the next `Check`.

## 7. Outcome — THE MODEL GATE IS MET, AND THE RING IS WHAT MEETS IT

Measured 2026-09-13 on `strix:gpu0` (gfx1151, Radeon 8060S, ROCm 7.2.4, 30 GiB
host, 32 CPUs), box exclusively leased, under `rc` job
`672093bc-932b-4e54-b319-e15529f70256`. Built on the worker in
`/tmp/vllmcpp-chunked-h2d` from a clean clone at `81f91b6005c7e8` with
`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_HIP=ON`
`-DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 -DROCM_PATH=/opt/rocm -DVLLM_CPP_SERVER=ON`,
`ninja -j 4 vllm-cli vllm-server`, run with
`LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib`. `VT_ROCM_MANAGED_ALLOC` unset.

**The artifact was asserted before it was timed.** `ldd` on `vllm-cli` resolves
`libamdhip64.so.7`, `libhsa-runtime64.so.1`, `libhipblaslt.so.1`,
`libhipblas.so.3` and `librocblas.so.5`, all from `/opt/rocm-7.2.4/lib`, and
CMake reported `ROCm backend: ENABLED for arch(es) [gfx1151]`. `vllm-cli` md5
`364f11fddc6392feb79cdc23f8d69cf5`, `vllm-server` md5
`aab04a1549ad3a5a543b1a6012834054`.

Workload: `examples/vllm-cli --device auto --max-tokens 32 --temperature 0
--max-num-seqs 1 --repeat 3 --prompt "Say hello"` over
`/workspace/ckpt/qwen4exp-flash-next-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf`
(shard 1 of 3; 10,946,624 + 49,990,818,368 + 22,544,696,352 bytes).

### The A/B, ONE BINARY, ONE BOOT, THE KNOB READ AT RUNTIME

| | ARM A — ring ON, CIFS | ARM B — `VT_ROCM_PINNED_H2D_MIB=0`, CIFS | ARM C — ring ON, LOCAL copy |
|---|---|---|---|
| token | **YES — 3 x 32, `finish_reason=length`** | **NONE**, killed at the 1200 s deadline (exit 137) | **YES — 3 x 32** |
| wall | 771 s, exited 0 | 1218 s, `killed_at_deadline=1` | **62 s**, exited 0 |
| `[vt load] mmap+header` | 0.568 s | 0.201 s | 0.023 s |
| `[vt load] weights` | 61.187 s | 35.909 s (page cache warm from arm A) | **8.770 s** |
| peak `VmHWM` | 20,714,504 kB (19.75 GiB) | 27,076,580 kB (25.82 GiB) | 26,920,604 kB (25.67 GiB) |
| peak `RssFile` | 14,516,792 kB (13.84 GiB) | 21,093,296 kB (20.12 GiB) | 20,728,476 kB (19.77 GiB) |
| peak `RssAnon` | 6,190,884 kB (5.90 GiB) | 5,807,020 kB (5.54 GiB) | 6,190,900 kB (5.90 GiB) |
| peak device `mem_info_vram_used` | **77,271,658,496 B** | 31,873,912,832 B | **77,271,609,344 B** |
| `wchan` over D-state threads | 115 `folio_wait_bit_common`, 1 `wait_for_response`, **0 `svm_range_set_attr`**, 122 samples | **135 `svm_range_set_attr`**, 39 `folio_wait_bit_common`, 20 `do_mprotect_pkey`, 8 `exit_mm`, 1 `wait_for_response`, 197 samples | 1 `wait_for_response`, 1 unresolved, **0 `svm_range_set_attr`**, 11 samples |

Arm C is arm A with the three shards copied to the worker's local disk first
(`/tmp/ckpt-iq1s`, same bytes, same sizes), so it answers the confound §6a left
open: the checkpoint lives on a CIFS mount and page-cache read wait is not device
work. `VT_ROCM_PINNED_H2D_MIB` is unset in both.

Arm B is `rocm-host-residency-after-upload.md` §6a reproduced to within noise —
27.08 vs 27.25/27.32 GB `VmHWM`, 21.09 vs 21.21/21.34 GB `RssFile`,
31,873,912,832 vs 31,878,860,800/31,880,183,808 B of device memory, and the same
`svm_range_set_attr` majority. That is what makes arm A attributable: the two
arms are the SAME BINARY minutes apart on the same board, and the only thing that
differs is one environment variable this change reads. The ring is the cause.

**The device memory is the tell.** Arm B stops at 29.69 GiB and stays there,
exactly as §6a recorded. Arm A reaches 71.96 GiB, which is the whole checkpoint.
So the wedge was never the model failing to fit; it was the upload never
finishing.

**THERE IS NO DISCREPANCY, AND THIS PARAGRAPH USED TO SAY THERE WAS.** It read
that 71.96 GiB "exceeds the 33.27 GB `hipMemGetInfo` total §6a quotes" and
declined to resolve it. `.agents/environment.md` §"strix" already resolved it:
since the 2026-09-11 firmware change this board reports `mem_info_vram_total`
and `hipMemGetInfo` total of **96.000 GiB**, while 33,270,497,280 B is the
box's HOST RAM. §6a read one for the other and this spec propagated it. Both
specs now say so; 71.96 GiB of 96.000 GiB is an unremarkable number.

### The generations

| arm | run | tokens | seconds | tok/s |
|---|---|---|---|---|
| A (CIFS) | 1 | 32 | 695.871 | 0.046 |
| A | 2 | 32 | 6.048 | 5.291 |
| A | 3 | 32 | 6.069 | 5.273 |
| C (local) | 1 | 32 | 38.438 | 0.833 |
| C | 2 | 32 | 6.067 | 5.274 |
| C | 3 | 32 | 6.067 | 5.275 |

**Run 1 of each arm is not a decode number and must not be quoted as one.**
Weight staging is lazy — `dense_attn::ResidentWeight` uploads on first use,
behind the `d_dev` memo — so run 1 carries the one-time 72 GiB host-to-device
transfer of the whole checkpoint.

**The steady-state number is 5.0-5.3 tok/s.** This spec first wrote
"5.27-5.29 tok/s ... spread max-to-min 0.34%" off FOUR samples across TWO
process launches — 5.291, 5.273, 5.274, 5.275 — and **0.34% understates the
spread of this measurement.** A fresh review launched the process three more
times on the reviewed head and read 5.002, 5.183 and 5.097 tok/s, a 3.6% spread
on its own; the repair wave's confirmation run (§5a) read 5.258 and 5.269. Nine
samples across six independent launches run from **5.002 to 5.291 tok/s**, 5.8%
max-to-min. Quote the range and the launch count; 5.29 is not reproducible to
that precision and must not be written as if it were. The row's
three-repetition rule is satisfied several times over, and it is satisfied
across process boundaries rather than three times inside one handle, which is
the stronger shape. gfx1151 fails about two
runs in five with an illegal GPU memory access
(`ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD`); no arm here hit that signature, and
all six generations completed with `finish_reason=length`.

No TTFT is recorded: `vllm-cli` in blocking mode reports whole-generation
seconds, not first-token latency, and no number is invented from it. What arm C
does give is a cold-start-to-first-answer figure on local storage:
0.023 s header + 8.770 s weights + 38.438 s first generation, 62 s of wall for
the whole process.

### What this does NOT claim

- No throughput COMPARISON. There is no llama.cpp or vLLM denominator here, and
  5.0-5.3 tok/s is this engine's first number on this part, not a ratio.
- No load-time verdict for the ring. Arm A's 61.187 s and arm B's 35.909 s are
  not comparable: arm A read the checkpoint cold off CIFS and arm B read it with
  the page cache already warm. A load-time A/B needs interleaved repeats from a
  dropped cache and was not taken.
- Nothing about any other family. §3b names them; only Qwen4-Exp was run.

## 7a. Where the first generation's time goes — MEASURED, not hypothesised

Arm A spends 695.871 s inside its first generation and 6.05 s inside each
subsequent one, and the obvious misreading is that the ring stages at about
104 MiB/s. It does not. Arm A's `wchan` histogram is 115 of 122 samples in
`folio_wait_bit_common`, which is page-cache read wait on the CIFS mount
(`//192.168.68.102/Data`), not device work.

**Arm C settles it.** The same binary, the same checkpoint, the same ring, with
the three shards copied to local disk first: the first generation takes
**38.438 s instead of 695.871**, an 18.1x reduction, and the D-state histogram
collapses to 11 samples with nothing in it. Weight load falls the same way,
8.770 s against 61.187. So 657 of arm A's 696 s were reading the file over CIFS,
and the ring itself moves the 72 GiB checkpoint onto the board in about 38 s,
which is roughly 1.9 GiB/s.

**§6a's CIFS confound is therefore separated and closed for this row.** It was
never the cause of the wedge — arm B wedges off the same mount arm A succeeds
on — and it accounts for essentially all of the remaining first-generation cost.

## 8. Stop conditions

- The ring cannot be placed without changing `Backend::Copy`'s asynchronous
  contract: STOP and return `NEEDS_DECISION`.
- The change would have to widen past `RocmBackend::Copy` to be effective: STOP
  and return `NEEDS_DECISION` rather than editing a second backend.
- `strix:gpu0` is unreachable or the controller is down: the model gate is
  reported UNVERIFIED. It is never replaced by an `ssh` plus a file mutex the
  fleet cannot see.
