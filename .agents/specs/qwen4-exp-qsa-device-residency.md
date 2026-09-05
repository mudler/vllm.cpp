# The QSA block's three host reads, on a device-resident operand

**Campaign row:** `MODEL-MM-QWEN4-EXP`
**Issue:** [#2421](https://github.com/mudler/vllm.cpp/issues/2421)
**Parent spec:** [`qwen4-exp-flash-next.md`](qwen4-exp-flash-next.md), which carries
this work under `## Owed` as "the QSA CUDA arm must give the translation a
device-side home or argue it away".
**Base:** `origin/main` at `84f6fac0a`. **Rebased from `4ab04afd6`, and the
rebase changed what this row can claim.** At the old base a CUDA `qwen4_exp` step
was unavailable at the LOADER, so the device branch would have landed unreached.
Both keystones have since merged: #2396 made `DeviceQuantGatherSupported` read
`vt::OpRegistered(vt::OpId::kEmbeddingQuant, dev)` instead of `dev == kCPU`, and
#2391 registered the four missing `vt::` `kCUDA` arms. So the refusals this
change removes are now genuinely reachable from `ModelRegistry::Forward` on a
`--device cuda` queue.

## Scope

`src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp` resolves three values on
the host. Each site refuses by name when its tensor is not CPU-resident:

| site | anchor at the base SHA | value read |
|---|---|---|
| `IndexerRows` | `qwen4_exp_qsa_block.cpp:83` | group 2's block table, `[1, pages]` i32 |
| `CheckRopeLayoutsAgree` | `qwen4_exp_qsa_block.cpp:147` | `kRopeProbeRows` rows of the packed cache and of the separate `cos`/`sin` |
| `Qwen4ExpQsaIndex` | `qwen4_exp_qsa_block.cpp:273` | `kv_lens`, `[T]` i32 |

In scope: make the three reads work on a device-resident operand, by copying the
words they read to the host explicitly. Out of scope, and named under `## Owed`
rather than dropped: a device-side page translation, a device-side rope
cross-check, and any throughput claim.

**Not in scope, and deliberately so: the four `vt::` op arms** (#2380 / PR #2391)
and the load-time quantized gather gate (PR #2396). This change is one of three
independent things in the way of a CUDA step and it claims only its own.

## Why refusing is not the answer, and why deleting is not either

The three refusals are correct as written. A host loop that dereferenced a device
pointer would fault or, on a unified-memory device, read plausible bytes and
produce a silently wrong selection — which is the failure this row already paid a
day for once (dropped repack markers, NaN at layer 0, every token id 0, nothing
thrown). So the refusals are not deleted.

They are also not the smallest thing that works. Each of the three values is
O(pages) or O(T) machine words that pick out an INDEX or feed a HOST comparison,
and none of them has a device-side consumer waiting for it:

- the block table's output is a `[n]` i32 row vector that is immediately uploaded
  as the `idx` operand of `vt::IndexSelect` / `vt::IndexCopy`;
- the rope probe feeds `std::fabs` on the host and produces no tensor at all;
- `kv_lens`'s output is `win_start` / `win_end`, two `[T]` i32 vectors that are
  immediately uploaded.

So the honest move is to say the read is a host read, copy the words it reads,
and state the cost. `Backend::Copy` infers its direction from the pointers
(`src/vt/cuda/cuda_backend.cu:116`, `cudaMemcpyDefault`), so one entry point
covers every arm, and `Backend::Synchronize` is what makes the bytes readable.

## Design

One file-local helper, `HostWords<T>(Dev, const Tensor&, offset, count)`. On a
`kCPU` tensor it is a `memcpy` from the tensor's own bytes and enqueues nothing.
On any other device it enqueues one `Backend::Copy` and one
`Backend::Synchronize`. The three call sites pass the exact element range they
read, so the copy's size is visible where the read is.

Three consequential decisions:

1. **`kv_lens` becomes ONE tensor, not two.** At the base SHA `QsaBlockCore`
   builds `kv_lens_host`, wraps it as a CPU `Tensor` for `Qwen4ExpQsaIndex`, AND
   uploads the same bytes as a `DBuf` for `vt::Qwen4ExpQsaGatherAttention`. The
   CPU wrapper is dropped and the `DBuf` is passed to both. On CPU that is
   byte-identical and costs nothing (`DBuf` on a CPU queue is host memory whose
   `device.type` is `kCPU`, so `HostWords` takes the memcpy arm). On a device arm
   it costs one round trip and, crucially, it is what makes the new branch
   REACHED from the production forward instead of exercised only by a test.

2. **The block table stays a `DBuf` in `qwen4_exp_registry.cpp`.** The parent spec
   records a cheaper fix — hand the block a host tensor over the runner's own
   vector — and this change does not take it, for the same reason as (1): a host
   tensor there would make the device branch dead on the production path. The
   registry is not touched.

3. **The block table is downloaded ONCE per block call, not twice.**
   `IndexerRows` is called twice on the paged arm (the scatter at `past_len`, the
   gather at `0`). It becomes a function over an already-resolved host vector, and
   `QsaBlockCore` resolves that vector once.

## Upstream anchors

This change introduces no arithmetic. The three values and their meanings are
unchanged, so the algorithm oracle is unchanged: `transformers` 5.16.0,
`models/qwen4_exp/modeling_qwen4_exp.py`, as the file's own header states. The
anchors the touched code cites — `:684`, `:670-676`, `:693`, `:653`, `:679` —
are untouched and re-verified by the block's existing golden gate, which compares
values against that oracle's captured pre-top-k `scores`.

The residency policy itself is a vLLM mirror in the following sense and no other:
vLLM's own paged-attention wrappers read a block table on the host wherever a
Python-level loop needs one and hand the device the resulting index tensor. This
change is not a port of a named upstream function, and this spec says so rather
than manufacturing an anchor for it.

## Tests

The CPU arm is the behavioural oracle. Three obligations:

1. **CPU stays byte-identical.** `test_qwen4_exp_qsa_block`,
   `test_qwen4_exp_qsa_device`, `test_qwen4_exp_layer_loop` and
   `test_qwen4_exp_runner` must be green and unchanged. On a CPU queue every
   tensor is `kCPU`, so `HostWords` takes the memcpy arm and no value moves.

2. **The device arm agrees with the CPU arm.** A CUDA case that runs the same
   block over the same inputs on both queues, comparing:
   - the SELECTION by SET EQUALITY per query token, plus the printed MARGIN
     between the last selected and the first rejected logit. A top-k error is
     bimodal — it flips or it floors — so a float tolerance on `block_ids` gates
     nothing;
   - the LOGITS and the BLOCK OUTPUT by `max|diff|`, against the parent spec's
     existing relative bounds.

3. **The refusals that remain still fire.** The block still refuses what it
   cannot do; those cases are unchanged and must stay red-on-mutation.

## Reachability

`ModelRegistry::Forward` on a `--device cuda` queue is the entry point. The
mutation is the deletion of the production call site: remove the
`RunQwen4ExpQsaBlockPaged` call in `qwen4_exp_forward.cpp`'s layer loop in a
scratch copy and the focused gate must go RED. Proving applied-ness is by a
COUNTED property, never by a patch exit code.

**This change alone does not make a CUDA step run.** Two other things are in the
way and are owned elsewhere: the four missing `vt::` CUDA op arms (#2380, PR
#2391) and the load-time quantized gather gate (PR #2396). This spec claims only
that the QSA block stops refusing a device-resident operand, and the `## Now`
section below states what a CUDA step actually reaches.

## Stop conditions

Stop and report rather than widening scope if:
- the CUDA forward stops somewhere other than predicted — that finding outranks
  the fix;
- making a read work on device requires a new `vt::` op or a change under
  `src/vt/cuda/` — that is another wave's territory;
- a device-vs-CPU selection difference appears. A selection that moves is a
  correctness defect, never a tolerance to widen.

## Now

Landed on `row/MODEL-MM-QWEN4-EXP-QSADEV`, PR #2422. The device arm is UNMEASURED
— see `## Owed`.

## Evidence

Host: `mudler-ubuntu-box`, x86-64, g++ 13.3.0, `-DCMAKE_BUILD_TYPE=Release`,
Ninja, no CUDA in the build. Tree at `97f51f1ad` plus this spec. Every exit
status below was read from the command's own `$?`, never through a pipe.

### The CPU gate, green and stable

`ninja -j 4 test_qwen4_exp_qsa_block test_qwen4_exp_qsa_device
test_qwen4_exp_layer_loop test_qwen4_exp_runner` → **rc 0**.

| suite | rc | cases | assertions |
|---|---|---|---|
| `test_qwen4_exp_qsa_block` | 0 | 13 / 13 | 5937 / 5937 |
| `test_qwen4_exp_qsa_device` | 0 | 12 / 12 | 4697 / 4697 |
| `test_qwen4_exp_layer_loop` | 0 | 6 / 6 | 309 / 309 |
| `test_qwen4_exp_runner` | 0 | 5 / 5 | 136 / 136 |

Reproduced identically three times. The oracle case reports
`max|diff| = 0.00982457 against a bound of 0.03`, unmoved from the base SHA, which
is what "byte-identical on CPU" predicts: on a CPU queue every operand is `kCPU`,
so `StageHostWords` takes its memcpy arm.

### A FALSE RED, and how it was caught

The FIRST run of this gate was red: `test_qwen4_exp_layer_loop` 4/6, with the
oracle at `max|diff| = 0.33274` and the by-name case reporting `group-2 rows
written = {}`. It was not the change. **The mutation harness's dry runs had been
applied and reverted while `ninja` was still compiling**, so the object for
`qwen4_exp_forward.cpp` was built from a tree that had M1 applied — the
production QSA call site deleted. Both symptoms are M1's signature exactly, which
is what identified it.

It was settled by measurement rather than by argument: the three touched files
were reverted to the base SHA `4ab04afd6`, rebuilt and rerun (`rc 0`, 6/6,
309/309), then restored and rebuilt (`rc 0`, 6/6, 309/309). **Never run a
mutation against a live build.** A build that races a mutation produces a binary
no source in the tree corresponds to, and it reads as a defect in the change.

### The mutations

Applied one at a time, each with a full rebuild and no other build running, each
reverted with `git checkout --` and confirmed byte-for-byte by
`git diff --exit-code` (rc 0). Applied-ness is proven by a COUNTED property, not
by a patch exit code.

| # | mutation | counted property | result |
|---|---|---|---|
| **M1** | REACHABILITY. Delete the production call site — the `RunQwen4ExpQsaBlockPaged` call in `qwen4_exp_forward.cpp`'s layer loop. Not the block. | `count("RunQwen4ExpQsaBlockPaged")` 1 → 0 | `test_qwen4_exp_layer_loop` **RED**, rc 1, 4/6 — including `qwen4_exp layer loop: ModelRegistry::Forward reaches it on a loaded qwen4exp GGUF`. `test_qwen4_exp_runner` stayed GREEN, so that suite does not gate reach into this block. |
| **M2** | Is the new host read on the path at all? Make `StageHostWords`' CPU arm copy nothing. | `count("std::memcpy(dst, src, bytes);")` 1 → 0 | `test_qwen4_exp_qsa_block` **RED** rc 1, 2/13; `test_qwen4_exp_layer_loop` **RED** rc 1, 1/6 |
| **M3** | Does the ROPE probe's staged read reach real data? M2 cannot answer it — with the memcpy gone all three buffers stay zero and zeros agree with zeros. Stage `cos` out of the `sin` table instead. | `count("StageHostWords(d, cos, r * rot, rot,")` 1 → 0 | Both suites **RED**, rc 1, with the intended message: `the PACKED cos_sin cache and the SEPARATE cos/sin tables do not describe the same angles` |

M1 answers `AGENTS.md` "Nothing lands dead" for the BLOCK. It does not answer it
for the device branch, which no queue can reach today; `## Owed` says so.

**The mutations were applied in the reviewed worktree, not a scratch copy, and
that is a deviation from [`reachability.md`](../reachability.md).** A scratch copy
needs its own build tree — CMake caches absolute source paths — and one clean
build of this tree took about 100 minutes on a box at load 130. Each mutation was
reverted and confirmed byte-for-byte before the next, and the final tree is
`git diff --exit-code` clean against `HEAD`.

## The device measurement

`thor:gpu0` (`rc-worker-n8smh`, aarch64 Tegra), inside an `rc` lease, job
`51587748-255c-425f-aabf-59663f24a962`. CUDA toolkit 13.0 apt-installed by the
job; `cmake -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110
-DVLLM_CPP_TRITON=OFF`, `CMAKE_RC=0`, `BUILD_RC=0`, **40 `.cu.o` objects**. Tree
`ec666b716` (= `origin/main` `84f6fac0a` + this branch).

**Two legs, differing ONLY in `qwen4_exp_qsa_block.{cpp,h}`.** Leg B substitutes
`origin/main`'s copies of those two files into the same tree and rebuilds
(`BUILD_B_RC=0`); the counted property is `grep -c kCPU`, **6 on main against 2
here**. That is a real ABSENCE, not a mutation.

### The refusal was real, and it was exactly where predicted

| leg | `test_qwen4_exp_qsa_block` | the block stopped with |
|---|---|---|
| **B** (main's block) | **rc 1**, 12/13, 5939/5940 — `CHECK_FALSE(IsResidencyRefusal(...))` fired | `qwen4_exp qsa block: the two rope layouts are cross-checked on the host, so both must be CPU-resident …` at **`qwen4_exp_qsa_block.cpp:147`** |
| **A** (this branch) | **rc 0**, 13/13, 5940/5940 | `vt: cuda rmsnorm: weight dtype must match x` at `src/vt/cuda/cuda_ops.cu:463` |

So `CheckRopeLayoutsAgree` WAS the first stop on a CUDA queue, as the code read
predicted, and it is gone. **The CUDA-ness of the run is proven by those two
messages themselves** — the leg-B refusal fires only when `device.type != kCPU`,
and leg A's names a `src/vt/cuda/` kernel. `test_cuda_backend` returned 127
because the job built only the two test targets it needed, so that particular
instrument did NOT run and is not offered as evidence.

### But the FORWARD never reaches the block, and that outranks the fix

Both legs, identically, through `ModelRegistry::Forward` on a CUDA queue:

```text
vt: qwen4_exp_gated_residual: block_inject_weight device mismatch
    at src/vt/ops.cpp:2562
```

That is **decoder layer 0's attention hyper-connection** — before PLE, before any
MoE block, and long before layer 3's `qwen_sparse_attention`. The QSA block is
not reached from the forward on CUDA at all, which is why leg A and leg B give
the same answer for that case.
`qwen4_exp_forward.cpp:418` and `:476` pass `lw.{attn,mlp}_hc.inject.View()` raw
while every sibling operand goes through `dense_attn::ResidentWeight`.
[#2449](https://github.com/mudler/vllm.cpp/issues/2449) owns it, and records the
worse shape behind it: `check_projection` skips the device check entirely for a
block-quantized operand, and the released checkpoint stores every inject as Q8_0,
so on the real weights this would be a host pointer handed to a device kernel
rather than a message.

### What is therefore NOT measured

**No CPU-vs-CUDA parity numbers exist.** Tier 2 of the block case never ran,
because the block stops at the `cuda rmsnorm` dtype refusal above. There is no
`max|diff|`, no selection set comparison and no margin from a device. **No token
was produced on CUDA, and none is claimed.**

## Owed

- **(SUPERSEDED BY THE REBASE.)** This entry read "THE DEVICE BRANCH LANDS
  UNREACHED" and was correct at base `4ab04afd6`: the loader threw for any device
  where `DeviceQuantGatherSupported` was false, and that predicate was
  `return dev == vt::DeviceType::kCPU;`. It is kept, struck, because a reader
  scanning for "is this dead code" needs to be sent forward rather than told
  something false. **#2396 replaced that predicate with
  `vt::OpRegistered(vt::OpId::kEmbeddingQuant, dev)` and #2391 registered the
  four missing `vt::` `kCUDA` arms**, so a CUDA queue now reaches this block and
  the `Backend::Copy` arm is on a production path. What remains owed is the
  MEASUREMENT, below.
- **(CLOSED — the device run happened; see `## The device measurement`.)** This
  entry read "NO DEVICE RUN IS RECORDED IN THIS SPEC YET." On a CPU queue every operand is
  `kCPU`, so `StageHostWords` takes its memcpy arm and the `Backend::Copy` arm
  does not run; the CPU gate therefore cannot see the device branch at all, and
  its greenness is evidence of no regression rather than of the fix. The two CUDA
  cases — `test_qwen4_exp_qsa_block.cpp`'s block-level parity case and
  `test_qwen4_exp_layer_loop.cpp`'s `ModelRegistry::Forward` case — each report
  `UNMEASURED` on a build with no CUDA backend, which is what a CPU CI run
  produces. Until a leased device runs them, **the predicted stop-point at
  `CheckRopeLayoutsAgree` is a code read and nothing more.**
- **THE SHIPPED CHECKPOINT WOULD NOT REACH THE QSA LAYER EITHER, for a worse
  reason than a refusal.** `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S stores
  tensors in IQ4_NL and Q5_0, neither of which `IsCudaKeepQuantSupported`
  (`src/vt/cuda/cuda_quant_dot.cu:1736-1770`) admits, and the fallback at
  `:1993-2000` calls the CPU kernel on device pointers. That is a SIGSEGV at
  layer 0's projections rather than a message, and the file's own comment records
  it. [#2419](https://github.com/mudler/vllm.cpp/issues/2419) owns it; this spec
  claims nothing about that path.
- **THE CPU-vs-CUDA PARITY OF THIS BLOCK IS STILL OWED.** The gate is written and
  runs — `test_qwen4_exp_qsa_block.cpp`'s tier 2 compares the output against the
  oracle golden and the SELECTION by SET EQUALITY with its margin printed — but it
  cannot execute until `vt::RmsNorm`'s CUDA arm accepts this block's operands
  (`cuda_ops.cu:463`, "weight dtype must match x"). Until then the device arm is
  proven to be ENTERED and not to be CORRECT, and this spec claims only the first.
- **THE COPY COSTS A QUEUE SYNCHRONIZE PER QSA LAYER PER STEP on a device arm.**
  THREE, counted rather than estimated: `CheckRopeLayoutsAgree` synchronises once
  for its nine staged ranges, `QsaBlockCore` once for group 2's page table on the
  paged arm, and `Qwen4ExpQsaIndex` once for `kv_lens`. On this architecture's
  48 layers, of which 12 are `qwen_sparse_attention`, that is 36 synchronizes per
  step. They are NOT folded into one here: the rope check runs before the cache
  validation that establishes the table's shape, so staging both would reorder
  which refusal a doubly-malformed caller sees, and reordering refusals to save a
  synchronize on a path no step can reach yet is the wrong trade. NOTHING HAS
  MEASURED ANY OF IT, because no CUDA `qwen4_exp` step exists to measure. The fix is the
  entry the parent spec already carries: fold the page resolution into
  `vt::Qwen4ExpQsaCompress`'s own address mode, which removes the table download,
  and give the rope cross-check a device-side home or argue it away. Owed under
  [#2421](https://github.com/mudler/vllm.cpp/issues/2421); no speed claim on this
  row is admissible before it.
- **THE ROPE CROSS-CHECK IS NOT MEMOIZED, and that is deliberate.** The forward
  hands every QSA layer the same three tensors, so a memo keyed on their data
  pointers would turn 12 checks per step into one. It is not taken: the device
  pool recycles pointers, so a different table at a recycled address would pass a
  pointer-keyed memo. That is the mute-switch shape this file already refuses
  once, and paying the check is cheaper than being wrong about it.
- **NO DEVICE ARM OTHER THAN CUDA WAS RUN.** `HostWords` is device-agnostic by
  construction — it reads `device.type` and calls `Backend::Copy` — but ROCm,
  Metal, Vulkan and Tenstorrent are unmeasured here and this spec claims nothing
  about them.
