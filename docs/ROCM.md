# ROCm (AMD GPU) backend — contributor guide

**State today: the W0 skeleton is community-verified on four architectures, and
the F6 unified-memory fix (approach (b)) is committed but unverified.**
[Issue #41](https://github.com/mudler/vllm.cpp/issues/41) board owners compiled
the HIP sources clean and ran the ctest gates on gfx1151 (Strix Halo), gfx1103
(Radeon 780M), gfx1100 (4x 7900 XTX) and gfx1201 (2x R9700) — M0 and M1 MET on
all four, with two runtime-teardown caveats recorded in §7. Their headline
finding, F6: `UnifiedMemory()` probed **false** on the RDNA3 APUs, because
XNACK-less RDNA reports `PageableMemoryAccess=0`, so the zero-kernel reference
tier the whole unified-memory plan rests on did not install and M2 was blocked.
The ratified fix (§3.1) allocates through `hipMallocManaged` on integrated
managed-capable devices so host access is API-guaranteed — written blind like
the skeleton before it, so **the (b) branch owes the same community compile+run
evidence W0 already earned**; a compile error in it is useful data, not a
mistake on your side.

This page exists because several people offered hardware in
[issue #41](https://github.com/mudler/vllm.cpp/issues/41), and it answers the
three questions that decide whether that goes anywhere: what a backend actually
*is* in this codebase, what to write first on the hardware you own, and what
"done" means. The design record behind the skeleton, including what was
deliberately left out, is
[.agents/specs/rocm-backend-w0.md](../.agents/specs/rocm-backend-w0.md); the
unified-memory decision record is
[.agents/specs/rocm-unified-memory-b.md](../.agents/specs/rocm-unified-memory-b.md).

Everything here is checked against the tree on 2026-08-08. Where a number is
counted, the command that counts it is given, because these numbers drift.

## 1. Why ROCm is the cheapest backend to add

Three structural facts, in the order they matter:

1. **The engine never learns about your device.** Scheduler, KV/block manager,
   persistent batch, sampler and serving are backend-agnostic, mirroring
   upstream vLLM. A backend lands as additive files through three seams
   ([.agents/backends.md](../.agents/backends.md)). Adding a platform touches
   one enum and one switch, both in `include/vt/device.h`. That is the whole
   core edit.
2. **Our CUDA kernels are ports of vLLM's `csrc/`, and upstream compiles that
   same `csrc/` for ROCm through a hipify pass.** So ROCm is not the
   Metal/Vulkan situation, where every kernel is written from scratch against a
   foreign API. Most of `src/vt/cuda/` is HIP source that has not been hipified
   yet. Upstream also ships RDNA3-specific kernels in `csrc/rocm/`
   (`q_gemm_rdna3.cu`, `moe_q_gemm_rdna3.cu`, `skinny_gemms.cu`), and every board
   offered in #41 so far is RDNA3.

   One caveat, so nobody loses an afternoon to it: vLLM's own `cmake/hipify.py`
   imports `torch.utils.hipify`, so it is a **torch-dependent** tool and we
   cannot reuse it. Your route is ROCm's `hipify-clang`, or hand-translation.
   `src/vt/rocm/rocm_rmsnorm.hip` is a hand-translation of
   `src/vt/cuda/cuda_ops.cu:96-126` written out deliberately so the two can be
   read side by side as a worked example of what the pass does.
3. **On unified-memory parts, a model can run correctly with zero ROCm
   kernels.** See §3. This is the single biggest lever for getting started, and
   it splits the work by hardware rather than by skill.

## 2. What a backend is, file by file

These now exist. The column that matters is the last one: what has been checked
on a real machine, and what has not.

| Seam | File | Verified? |
|---|---|---|
| Device enum | [`include/vt/device.h`](../include/vt/device.h) | ✅ compiled; the enum forced exactly one switch site tree-wide |
| — | [`include/vt/rocm/rocm_arch.h`](../include/vt/rocm/rocm_arch.h) — gfx name → `(major, minor)`, ported 1:1 from `rocm.py:223` | ✅ **unit-tested**, 40 assertions, no GPU needed |
| Runtime backend | [`src/vt/rocm/rocm_backend.hip`](../src/vt/rocm/rocm_backend.hip) — the 6 `vt::Backend` virtuals | ✅ W0 compiled + ctest-run on gfx1151/1103/1100/1201 (#41) — ❌ the approach-(b) delta is **unbuilt** |
| Op table | [`src/vt/rocm/rocm_ops.hip`](../src/vt/rocm/rocm_ops.hip) — one `RegisterOp` line | ✅ compiled + run on the same four boards |
| Kernel | [`src/vt/rocm/rocm_rmsnorm.hip`](../src/vt/rocm/rocm_rmsnorm.hip) | ✅ NMSE ≤ 5e-4 vs the CPU oracle on all four boards (F5: RDNA is wave32, the wave64 hazard moves to a future gfx9 board) |
| Platform | [`src/vllm/platforms/rocm.cpp`](../src/vllm/platforms/rocm.cpp) — mirrors `vllm/platforms/rocm.py` | ✅ compiles `-Werror` everywhere; ✅ run on the four boards |
| Attention | *(none yet — `get_attn_backend_priority()` returns empty)* | — |
| Build | `VLLM_CPP_HIP` in [`CMakeLists.txt`](../CMakeLists.txt) | ✅ ON-path configure+build on the four boards (Arch/TheRock now auto-hinted, §5 M0); ✅ the OFF path and the fail-without-hipcc path |
| Test | [`tests/vt/test_rocm_backend.cpp`](../tests/vt/test_rocm_backend.cpp) | ✅ W0 cases run green (1044 assertions in the #41 tables) — ❌ the two approach-(b) cases never run |

So the shape is decided and the parts that hold a *decision* are tested; what
you are validating is the API glue. Adding your own op is one line in
`rocm_ops.hip` plus the kernel — no selector, model or runner edit anywhere.

Op coverage as of 2026-08-06 (`OpId` has 106 entries):

| Backend | Registered ops |
|---|---|
| CUDA | 103 |
| CPU | 83 |
| Metal | 19 |
| Vulkan | 8 |
| **ROCm** | **1** (RmsNorm) |

Recount before quoting:

```sh
grep -rho 'RegisterOp(OpId::[A-Za-z0-9_]*' src/vt/<backend>/ | sort -u | wc -l
```

The platform seam is deliberately plain C++ with no device headers: everything
device-specific is reached through the `vt::Backend` virtuals, which is why
`platforms/vulkan.cpp` compiles without a Vulkan header. Do the same, and the
engine-side tree stays free of HIP.

## 3. Correctness before kernels: the reference tier

`include/vt/op_provider.h:186-224` is our equivalent of vLLM's
`CustomOp.forward_native`. An op with no native kernel on your device falls back
to the CPU kernel, registered at a strictly-negative priority so a native kernel
always wins when it exists. A backend that implements **zero** kernels is
therefore correct, just slow.

**The gate is `Backend::UnifiedMemory()`, never `DeviceType`.** A CPU kernel
dereferences host pointers, which is only valid where host and device memory
alias. Consequences:

- **Unified memory** (Strix Halo, RDNA3 iGPU, anything where you report
  `UnifiedMemory() == true` honestly): a model runs end to end as soon as
  §2's first three rows exist. Kernels then replace the fallback one at a time,
  each one a measurable win with no correctness risk.
- **Discrete** (7900 XTX and every dGPU): the tier never installs, and it must
  not. `GetOp` throws on an unregistered op, so a model runs only once the ops
  it needs are registered. Your first milestone is the kernel path.

Two rules that keep this honest: `VT_OP_PROVIDER_STATS=1` prints the first time
each `(op, device)` falls back, and `GetReferenceTierHits()` **must be 0 in any
performance measurement**. A non-zero value means you benchmarked the CPU.

### 3.1 The F6 fix: unified memory true by construction (approach (b))

Issue #41's headline finding (F6, measured on gfx1151, confirmed on gfx1103):
XNACK-less RDNA3 APUs report `hipDeviceAttributeIntegrated=1` but
`hipDeviceAttributePageableMemoryAccess=0`, so the W0 probe (CUDA's own
conjunction, integrated AND pageable) answered `UnifiedMemory() == false` on the
very boards the zero-kernel plan was written for — even though host dereference
of `hipMalloc` memory demonstrably worked there. The attribute answers the
opposite question (device reading pageable host memory) from the one the tier
needs (host reading device allocations); the two coincide on NVIDIA integrated
parts and come apart on RDNA.

The maintainer decision (#41, 2026-08-08) is approach **(b)** — make the claim
true by construction rather than gate on an architectural accident:

> On a device reporting hipDeviceAttributeIntegrated=1 (and ManagedMemory=1 +
> ConcurrentManagedAccess=1), Backend::Alloc in the ROCm backend uses
> hipMallocManaged instead of hipMalloc, and UnifiedMemory() returns true
> exactly then — host access becomes API-guaranteed rather than architecturally
> incidental, which is the standard this gate exists to hold.

What each device class gets:

| Device class | `Backend::Alloc` | `UnifiedMemory()` | Reference tier | M2 |
|---|---|---|---|---|
| Integrated + managed-capable (gfx1151, gfx1103: all three attributes probed 1) | `hipMallocManaged(hipMemAttachGlobal)` | **true**, by construction | installs — a model runs with one native kernel | unblocked |
| Integrated, NOT managed-capable (no known board; the probes default to 0 on error) | `hipMalloc` | false unless the W0 conjunction holds | does not install | blocked — post the probe triple on #41 |
| Discrete (gfx1100, gfx1201, MI50...) | `hipMalloc` — the managed branch is provably dead (`Integrated=0`) | false | never installs (memory-safety gate) | native kernels required, unchanged |

The free path is `hipFree` for both branches: the HIP runtime API documents it
as the release call for `hipMalloc` and `hipMallocManaged` allocations alike,
mirroring `cudaFree`. `Backend::AllocPinned` inherits the base delegation to
`Alloc` (`src/vt/backend.cpp:19`), so pinned blocks ride the same branch and
stay host-accessible — coherent with its contract (`include/vt/backend.h:76-78`).
Introspection for tests and bug reports:
`vt::rocm::ManagedAllocActive(index)` / `IntegratedDevice(index)` in
[`include/vt/rocm/rocm_runtime.h`](../include/vt/rocm/rocm_runtime.h) report
which path the silicon took, and `tests/vt/test_rocm_backend.cpp` gates that the
alloc path and the `UnifiedMemory()` claim move together — including F6's
decisive experiment (kernel writes, host reads back, **no copy**) as a standing
test.

Approach (a) — gate the tier on `Integrated` alone — remains the recorded
fallback **if managed allocations measure slower on gfx1151: measure, don't
assume** (the maintainer decision, verbatim). Decision record:
[.agents/specs/rocm-unified-memory-b.md](../.agents/specs/rocm-unified-memory-b.md).

## 4. Pick your first task from your hardware

| Hardware | Arch | Memory | Start here |
|---|---|---|---|
| Strix Halo / GTR9 Pro 128GB | gfx1151 | unified | M0/M1 **MET** (#41). Now: **verify the §3.1 fix, then M2** (§5.2) — the reference tier means a model runs with no further kernel written. Closest analogue to GB10, so the residency-policy question in §6 is yours |
| Radeon 780M iGPU | gfx1103 | shared | M0/M1 **MET** (#41). Same §5.2 path, smaller models. Best position to find every place a "CUDA" assumption is really an "NVIDIA" assumption. A vLLM-ROCm oracle is unlikely on this board, so M4 stays PENDING there — fine, and to be said rather than papered over |
| 4x 7900 XTX | gfx1100 | discrete | M0/M1 **MET** (#41, with the #132 caveat). Now **the kernel path**, since the reference tier cannot install on a dGPU and a model needs real kernels. The only board class that can host a vLLM-ROCm oracle for M4 and, later, multi-GPU TP — the backend already registers all four at `Device{kROCM, i}`. gfx1201 (2x R9700) is on the same discrete lane via PR #140 |
| RX 9060 XT | gfx1200 | discrete | M0/M1 **MET** independently ([#41](https://github.com/mudler/vllm.cpp/issues/41)). **Gemma-3-1B-it: M4 MET, no caveat** — 48/48 tokens identical against TWO independent real vLLM-ROCm oracles on this board (a prebuilt AMD image and a from-source build at this project's own pinned commit `555967922`). **Qwen3-0.6B: one genuine near-tie prompt, not a defect** — our CPU and ROCm backends split, and so do the two real oracles (2-and-2), each internally deterministic; the reference itself doesn't hold still on this input, which is the strongest possible evidence it's a near-tie-robust-gate case, not a bug on either side ([#269](https://github.com/mudler/vllm.cpp/issues/269), [.agents/specs/rocm-gfx1200-m2-correctness.md](../.agents/specs/rocm-gfx1200-m2-correctness.md)) |

These do not collide. Two people can be on M0/M1/M2 on unified parts while a
third does the hipify pass, and the discrete board is what turns the result into
a gated backend.

## 5. Milestones as concrete PRs

**M0 — build. MET** on gfx1151, gfx1103, gfx1100 and gfx1201 (#41 tables).
Tri-state `VLLM_CPP_HIP`, hipcc detection that fails loudly,
`VLLM_CPP_HIP_ARCHITECTURES`, `ROCM_PATH`. The Arch/TheRock findings F1/F3 are
now absorbed into the configure: when `ROCM_PATH` points at a real install
(default `/opt/rocm`), CMake derives `CMAKE_HIP_COMPILER_ROCM_ROOT`, seeds
`--rocm-path` into `CMAKE_HIP_FLAGS`, and exports `ROCM_PATH` into the
environment — each only when you have not set it yourself. The manual
three-flag workaround from the gfx1151 report
(`-DCMAKE_HIP_COMPILER_ROCM_ROOT=... -DCMAKE_HIP_FLAGS=--rocm-path=...` on top
of `-DROCM_PATH`) is therefore **legacy**: still honoured if passed, no longer
required. F2 (raw `--whole-archive` reaching the clang driver) was downstream
of the unidentified compiler and disappears with F1/F3 — if you still see it,
your compiler identification failed and that configure log is the thing to
post. *The absorption itself is untested on a real Arch/TheRock layout — a
configure log from one, with no manual flags, is wanted evidence on #41.*

**M1 — platform + backend. MET** on the same four boards: `ctest -R
'rocm|cross_device'` green, RmsNorm within NMSE ≤ 5e-4 of the CPU oracle on
real silicon — with two runtime caveats, both teardown-related, in §5.1 below.

**M2 — first model end to end. UNBLOCKED-UNVERIFIED on unified parts** by the
§3.1 fix: assert `ReferenceTierEligible(kROCM)` and run a small dense model.
Acceptance: greedy token parity against the **CPU backend** on the same build,
plus the `VT_OP_PROVIDER_STATS=1` output showing which ops fell back, which is
your kernel to-do list, sorted by real usage rather than by guesswork.

**On a discrete board there is no reference tier (§3), so M2's acceptance bar
is unchanged but its mechanism is not** — every op the model needs must
already be a real kernel; `VT_OP_PROVIDER_STATS=1` reporting `selected=vt-native`
on all of them, with zero fallbacks, *is* the evidence, since a fallback
literally cannot exist to hide behind. **MET on gfx1200** this way ([#269](https://github.com/mudler/vllm.cpp/issues/269)): both models ran greedy,
all-native, token parity vs `--device cpu`.

**Went further than M2 on gfx1200 the same day: two independent real
vLLM-ROCm oracles**, Docker-based — the practical answer to pip-vs-NixOS
friction, since neither needs a PyTorch/vLLM build on the host. Tier 1 is
AMD's prebuilt `rocm/vllm:...gfx120X...` image (fast, vLLM 0.19.1, not this
project's pin). Tier 2 builds this project's exact pinned commit
(`555967922`) from source, inside `rocm/vllm-dev:base` (the same base image
vLLM's own official Dockerfile.rocm uses) — ROCm 7.2.3 in that base matches
this board's native build exactly, compiled clean in ~6.5 minutes.

**Gemma-3-1B-it: 48/48 tokens identical against BOTH oracles** — genuine M4,
no caveat. **Qwen3-0.6B: the two oracles disagree with each other** (0.19.1
matches our ROCm; the exact pin matches our CPU), each internally
deterministic (K=5, 5/5). That is direct proof this specific prompt is a
genuine, version-sensitive near-tie in the reference implementation itself —
not a defect in either of our backends, and not closeable by a strict
token-exact bar on this input. Full story:
[.agents/specs/rocm-gfx1200-m2-correctness.md](../.agents/specs/rocm-gfx1200-m2-correctness.md).

### 5.1 Known runtime issues on the #41 boards

- **TheRock nightly teardown hang (gfx1103).** All three test binaries print
  `Status: SUCCESS!` and then never exit, so `ctest` times out waiting.
  arch-btw's GDB backtrace pins it inside `libamdhip64.so.7` during
  `__cxa_finalize`/`_dl_fini`, waiting on HSA `AsyncEventsLoop` threads stuck
  in an `ioctl` wait — an upstream runtime teardown deadlock in the TheRock
  nightly (`10.1.0a20260731`), not a vllm.cpp bug. Treat "SUCCESS printed, then
  hang" as a PASS of the test body plus this known issue; report the ROCm
  build you saw it on.
- **`-O0` hostcall teardown race ([#132](https://github.com/mudler/vllm.cpp/issues/132),
  gfx1100).** A no-build-type compile leaves the RmsNorm kernels with
  hidden-hostcall metadata at `-O0`; ROCm CLR's listener handshake can then
  deadlock the process finalizer under CPU saturation (intermittent, 14/20 at
  48 threads). Validated avoidance: build with an optimization level
  (`-DCMAKE_BUILD_TYPE=Release`), which removes the hostcall path entirely.

### 5.2 The sequence for board owners, post-F6-fix

Everything below assumes the tree at or after the approach-(b) change. On
Arch/TheRock, no compiler flags beyond `ROCM_PATH` should now be needed — if
that is false, the configure log is finding number one.

```sh
cmake -S . -B build-hip -DVLLM_CPP_HIP=ON -DCMAKE_BUILD_TYPE=Release \
      -DROCM_PATH=/opt/rocm        # or your TheRock dist prefix
cmake --build build-hip -j
ctest --test-dir build-hip -R 'rocm|cross_device' --output-on-failure
```

What to report on #41, in the M0/M1 table shape already in use there:

1. The configure/compile/link results, and whether any manual flag was still
   required (that would mean §5's F1/F3 absorption missed your layout).
2. The new probe triple printed by `test_rocm_backend` — `integrated`,
   `managed-alloc`, `UnifiedMemory()` — plus the pass/fail of the two
   approach-(b) cases ("alloc path and UnifiedMemory() move together",
   "kernel-written value is host-readable with no copy").
3. On an APU, the M2 attempt: a small dense model through the CLI. `--device`
   has no `rocm` literal yet; `auto` (the default) selects ROCm on an AMD box
   with no CUDA via the platform priority walk. E.g.:

   ```sh
   VT_OP_PROVIDER_STATS=1 ./build-hip/examples/vllm-cli \
     --model <a small dense HF model dir> \
     --prompt 'The capital of France is' --max-tokens 8 --temperature 0
   ```

   Post the generated tokens, whether they match the same command with
   `--device cpu`, and the `VT_OP_PROVIDER_STATS` fallback list — that list is
   the prioritized M3 kernel to-do for your board.

**M3 — kernels + attention.** Hipify `src/vt/cuda/` family by family, starting
with what M2's fallback log actually hit: layernorm, rope, activations, glue,
reshape-cache, sampling, then paged attention. Register a ROCm attention backend
and put its name in the platform priority in the same change. For what upstream
selects on your arch, read `_get_backend_priorities` (`rocm.py:407`) and
`get_attn_backend_cls` (`rocm.py:545`): AITER FA is gfx9-only, RDNA3 goes down
the Triton/ROCm attention path.

**M4 — correctness gate.** Greedy token parity against a vLLM-ROCm oracle on the
same hardware, same workload, following
[verification procedure](../.agents/verification.md) and the near-tie methodology. Where
vLLM's own greedy output is non-deterministic, the gate is distributional (ours
inside vLLM's K-run set), not token-exact.

**M5 — speed.** `vllm bench throughput` on the same box, quant-matched, against
the same model. The bar is vLLM, not llama.cpp. Method and honesty rules:
[verification procedure](../.agents/verification.md) and
[docs/BENCHMARKS.md](BENCHMARKS.md).

Each milestone is a PR, or several. Do not stack M3 kernels into one change: one
kernel family per PR, each with its own correctness check, is what keeps review
from becoming the bottleneck.

## 6. What not to port

Do not spend time hipifying these. They are NVIDIA-specific and none of them is
on the path to a working AMD backend:

- NVFP4 (`cuda_matmul_nvfp4*.cu`, the `nvfp4_tactics` family) and Marlin. FP4
  tensor cores are a Blackwell thing; AMD's analogue is MXFP4 on gfx950 and is a
  separate project.
- CUTLASS-backed FA2 and the sm90/sm100 scaled-MM paths. The ROCm equivalents
  are Composable Kernel / AITER, and they are M3-and-later decisions.
- Vendored Triton-AOT cubins. Arch-specific NVIDIA binaries.
- NCCL transport. RCCL is API-compatible, but multi-GPU is post-M5.
- cuBLASLt plan caches. Route GEMM to hipBLASLt and measure before porting any
  caching strategy.

## 7. Working with the record

The project keeps an append-only engineering record under `.agents/`. Two things
matter for an outside contributor:

- **Machine paths in that record are not instructions.** They describe the
  developer's boxes. Yours go in an untracked `.env` (copy
  [`.env.example`](../.env.example), which already has the device-toolchain
  fields a ROCm bring-up needs: toolkit root, compiler, target arch) plus
  `.agents/developer-preferences.md`. A fresh agent session will generate both
  interactively. Register your AMD box as a profile in
  [.agents/environment.md](../.agents/environment.md) so it becomes the named
  gate environment for the ROCm rows.
- **A gate you cannot run stays PENDING.** That is a normal, publishable state.
  Claiming a pass you did not observe is the one thing that is not recoverable.
  The same applies to labels: "build-supported" means it compiles and emits real
  code, "runtime-gated" means a board here executed it. Do not upgrade one to
  the other on inference.

## 8. CI gates your PR will hit

All of these run on pull requests and are cheap to check locally first:

- **`FOLLOWING_AGENTS_PROTOCOL` trailer** on every non-merge commit, asserting
  you read [AGENTS.md](../AGENTS.md). Also add `Assisted-by: <tool>` if an AI
  assistant helped.
- **PR size**: 900 changed lines outside `.agents/`, `docs/`, `scripts/`,
  `tests/scripts/`, `.github/`. Enforced on `row/*` branches, reported on
  others.
- **Documentation checkpoint**: `python3 scripts/check-doc-checkpoint.py --base
  <base> --head <head>`. It validates the **committed** diff, so run it after
  committing, not before.
- **Device-leakage ratchet**: `python3 scripts/check-device-leakage.py`. It
  counts CUDA-specific references in the device-agnostic layer
  (`src/vllm/`, `include/vllm/`) and fails on any increase. It will not object to
  ROCm code under `src/vt/rocm/` or to your platform file; it will object if a
  model TU grows a device branch. Keep ROCm specifics below the seams.
- The full CPU test suite. Run `ctest --test-dir build` before pushing; some
  tests are flaky under `ctest -j` on a loaded box, so re-run failures serially
  before reporting them.

## 9. Asking

Comment on [#41](https://github.com/mudler/vllm.cpp/issues/41) with what you
picked and what you hit. Milestones get split into their own issues once work
starts, so say which one you are taking to avoid two people writing the same
platform file.
