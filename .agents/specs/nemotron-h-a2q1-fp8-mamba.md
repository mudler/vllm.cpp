# A2-Q1 — NemotronH's 23 Mamba2 blocks reach the device on the FP8 W8A8 seam

**Issue:** [#810](https://github.com/mudler/vllm.cpp/issues/810).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)).
**Governing spec:** [`nemotron-h-abi-e2e.md`](nemotron-h-abi-e2e.md) — this file
owns the FP8 half of what that spec's §1 called `A2` and what the A2-R landing
commit (`598226e96`) renamed `A2-Q`.
**Base:** `origin/main` @ `0e1bee42f16b5f3fb3ae5a23869f6fd97bfc037d`.
**Declared dependency:** [#960](https://github.com/mudler/vllm.cpp/issues/960).
A2-Q1 **does not start** until it lands. See §3.
**Pinned oracle:** `${VLLM_SOURCE}` @ `5559679229bc961848b121ccdeaa8fa5d79bec98`
(vLLM 0.26.0.dev0), per [`upstream-sync.md`](../upstream-sync.md).
**Lifecycle at this commit:** unchanged. A spec commit changes no lifecycle
state and owes no `STATUS`/`BENCHMARKS`/`NOW` write; §9 records what the
implementing change owes.

**No product code is written by this spec.** Per the governing spec's §1.4 the
implementation is a separate PR by a different agent, and the reviewer is a
third.

---

## 0. Why this is its own unit

A2-Q was briefed as one unit covering the mamba, MoE and `lm_head` device arms.
Scoping measured it as three independent quantization mechanisms with three
independent failure modes, two of them separately blocked on this project's
sm_110 host. The split into **A2-Q1** (this file, FP8 W8A8) and **A2-Q2**
([`nemotron-h-a2q2-nvfp4-moe-lmhead.md`](nemotron-h-a2q2-nvfp4-moe-lmhead.md),
NVFP4 W4A16) was ratified 2026-08-16.

The decisive reason is not size. It is that the two halves have **different
blockers**: A2-Q1 cannot run on Thor at all until #960 lands, while A2-Q2 can
build and run there but cannot yet be trusted to report a number
([#962](https://github.com/mudler/vllm.cpp/issues/962)). Bundling them makes the
NVFP4 half wait on a build-file fix it does not need.

### 0.1 The ordering constraint that shapes the whole unit

**The mamba block cannot be split.** `mixer.in_proj` produces the fused `zxbcdt`
that the causal conv and the SSD scan both consume
(`mamba_mixer2.py:550`, split points `:692-696`; locally
`nemotron_h.cpp:478-484`). So the block moves to the device **whole, on the FP8
seam, or not at all**. There is no intermediate landing in which the conv is on
the device and `in_proj` is not.

That is why the FP8 shared seam was extracted first
([#940](https://github.com/mudler/vllm.cpp/issues/940), landed `0e1bee42f`):
without `dense_fp8_gemm.h` this block has no device path to build.

---

## 1. Scope

| In A2-Q1 | Out of A2-Q1 |
|---|---|
| the 46 mamba `in_proj`/`out_proj` projections on `dense_fp8_gemm.h` (`ResidentFp8:97`, `MatmulFp8CutlassD:127`) | the NVFP4 MoE and `lm_head` arms — A2-Q2 |
| the device causal conv, SSD scan and silu-gated group RMSNorm for all 23 mamba layers | paging, carried recurrent state, batching — A2-P / A2-B |
| supplementing `NemotronHMambaWeights` with the shared `Fp8Weight` (`qwen3_5_weights.h:273`) beside the existing `NemotronHOwned` | converting the NVFP4 weights to `Nvfp4Weight` — A2-Q2 |
| the per-block numeric gate for the mamba arm on the real checkpoint, both hosts | any throughput number, on any axis |
| fixing #960 | **NO** — A2-Q1 consumes it (§3) |

**Explicitly out:** the MTP head (parent W5), the GGUF k-quant arm (parent W7),
`NemotronHPuzzleForCausalLM`, `n_groups` TP sharding, and any change to the
Qwen3.5 or Kimi-Linear model files.

### 1.1 The conv-state dtype question does NOT reach this unit

The governing spec's §2.7 decides that the **persistent** conv page is bf16 and
that `vt::CausalConv1dFwd` is owed a bf16 conv-state arm. **That obligation
belongs to A2-P, not here.** A2-Q1 is non-paged: its conv state is a transient
per-call buffer, exactly as the host reference's is
(`nemotron_h.cpp:493-498`, whose comment already says so). No persistent page is
read or written, so no reconciliation is owed and none may be smuggled in.

Recorded explicitly because the temptation runs the other way: a unit touching
the conv looks like the place to fix the conv dtype. It is not.

**The persistent conv page IS bf16 today, and the guard is not what holds it
there.** `nemotron_h_registry.cpp:264` allocates the page as
`vt::DType::kBF16`, and `runner.cpp:885-893` sizes the buffer from
`vt::SizeOf(gdn_conv_cache_dtype_)`, so the ALLOCATION fixes the bytes. Only the
transient gather row is f32 (`nemotron_h_device.cpp:1954`, two `DBuf(d,
DType::kF32, …)`), which is `vt::GdnStateGather`'s op contract and matches
upstream's bf16 page. The falsification of "the page was widened to f32" is
therefore correct.

The comment above the paged guard is not. It says the conv page is "never
widened to f32", and the `VT_CHECK` on the next line admits
`kBF16 || kF16 || kF32` (`nemotron_h_device.cpp:1581-1583`). `runner.cpp:643-644`
admits the same three. A `conv_dtype` mutation to f32 would sail past both
checks; nothing but the allocation would stop it, and a token gate cannot see the
doubled bytes. This is a record defect on `main` — the comment and the guard are
both older than this row — and it is listed under §11 rather than repaired here,
because A2-Q1 owns no line of it.

---

## 2. Upstream anchors — `file:line` on both sides, at `555967922`

| Behaviour | Upstream | Ours |
|---|---|---|
| fused `zxbcdt` projection | `mamba_mixer2.py:550` | `nemotron_h.cpp:478` |
| z / xBC / dt split | `mamba_mixer2.py:692-696`, `:583` | `nemotron_h.cpp:482-484` |
| causal conv + silu | `mamba_mixer2.py:832-847` | `nemotron_h.cpp:507-526` |
| x / B / C split | `mamba_mixer2.py:535-543` | `nemotron_h.cpp:529-531` |
| `A = -exp(A_log)`, f32 | `mamba_mixer2.py` (`self.A = -torch.exp(self.A_log.float())`) | `nemotron_h.cpp:539-543` |
| varlen chunk metadata | `mamba2_attn.py:22-88` | `nemotron_h.cpp:559-571` |
| SSD scan, `dt_softplus=True`, `dt_limit=(0.0, inf)`, `z` NOT passed | `mamba_mixer2.py:870-892` | `nemotron_h.cpp:588-598` |
| silu-gated GROUP RMSNorm, `n_groups` (not 1) | `mamba_mixer2.py:478-480`, `:583-585` | `nemotron_h.cpp:608-621` |
| `out_proj` | `mamba_mixer2.py:586` | `nemotron_h.cpp:624` |

**The FP8 apply chain** is already cited on the seam header
(`dense_fp8_gemm.h:20-39`): `compressed_tensors_w8a8_fp8.py:60,201-207` and
`modelopt.py:444,531-537` both land on the same `fp8_linear.apply_weights`;
static per-tensor activation quant is `quant_utils.py:124`
(`kFp8StaticTensorSym`); the folded `alpha = input_scale · weight_scale` mirrors
vLLM's per-tensor `ScaledEpilogue`. A2-Q1 **cites, and does not restate,** that
chain — it is the seam's, not this model's.

---

## 3. The Thor dependency — MEASURED, and it is the reason this unit waits

This section is the part of the spec that cannot be re-derived cheaply. It is
recorded in full because the conclusion inverted twice during scoping.

### 3.1 What resolves on sm_110, and what does not

`cmake/CudaArchFeaturesTest.cmake` pins `marlin-nvfp4` ENABLED for `[110]`
(`:145`) and every other fast-path cell EMPTY (`:133-134`). The naive reading —
"no `cutlass-fp8` on Thor, therefore no fp8 arm" — is **wrong**, and so is the
opposite reading. The measured state is that Thor has *half* an fp8 arm.

**The GEMM half is present and works.** `kMatmulFp8CublasLt` is registered
UNCONDITIONALLY for kCUDA at `src/vt/cuda/cuda_matmul.cu:920` — it is not behind
the cutlass gate. And cuBLASLt genuinely resolves fp8 plans on this silicon.

> **Measurement (2026-08-16).** A standalone probe mirroring `BuildFp8Plan`'s
> exact key (`src/vt/cuda/cuda_matmul.cu:533`) — A = e4m3 op `T`, B = e4m3 op
> `N`, `CUBLAS_COMPUTE_32F`, scale `CUDA_R_32F`, epilogue DEFAULT, 32 MiB
> workspace — run in `nvidia/cuda:13.0.1-devel-ubuntu24.04` on
> `192.168.68.23` with `--runtime=nvidia -e NVIDIA_DISABLE_REQUIRE=1`.
> Device `NVIDIA Thor sm_110`, runtime 13000, driver 13020, cublasLt 130002.
>
> ```
> -- D=bf16 --
>   m=1     n=10304  k=2688   status=0 returned=1  -> FP8 PLAN AVAILABLE
>   m=8     n=10304  k=2688   status=0 returned=1  -> FP8 PLAN AVAILABLE
>   m=512   n=10304  k=2688   status=0 returned=1  -> FP8 PLAN AVAILABLE
>   m=1     n=2688   k=4096   status=0 returned=1  -> FP8 PLAN AVAILABLE
>   m=512   n=2688   k=4096   status=0 returned=1  -> FP8 PLAN AVAILABLE
>   m=64    n=4096   k=4096   status=0 returned=1  -> FP8 PLAN AVAILABLE
> -- D=f32 --   (same six shapes)                  -> FP8 PLAN AVAILABLE x6
> RESULT: 0 of 12 probes had NO fp8 plan
> ```
>
> The six shapes are NemotronH's own: `in_proj` N=10304 K=2688 and `out_proj`
> N=2688 K=4096 at decode (M=1), small-batch and prefill widths, plus a square
> control. **12/12 available, both D dtypes.**

**The activation-quant half is missing, and it is a build accident.**
`kQuantFp8Static`'s only CUDA registration is
`src/vt/cuda/cuda_matmul_fp8_cutlass.cu:376`, and `CMakeLists.txt:1668-1669`
compiles that translation unit **only** when `VT_CUTLASS_FP8_ARCHS` is non-empty
— EMPTY on sm_110. The kernel body, `QuantFp8StaticKernelCuda`
(`cuda_matmul_fp8_cutlass.cu:353-370`), is a plain elementwise
`x * (1/input_scale)` → e4m3 convert with **zero CUTLASS references**. It is
trapped in a CUTLASS-gated TU while its GEMM partner is available on the same
arch.

The consequence is not slowness. `MatmulFp8CutlassD`'s own guard
(`dense_fp8_gemm.h:130`) keys on `kMatmulFp8CublasLt`, which IS registered — so
the guard **passes** on Thor and the code crashes one call later, when
`vt::QuantFp8Static` misses, the resolver installs the portable CPU reference
tier (`src/vt/op_provider.cpp:501`, eligible because `CudaBackend::UnifiedMemory()`
is true, `cuda_backend.cu:111`) and that host kernel dereferences device
pointers. **#960 measured exactly this as a SIGSEGV on Thor**, alongside the
general form in [#844](https://github.com/mudler/vllm.cpp/issues/844).

### 3.2 The decision, and why the fix is not A2-Q1's

> **A2-Q1 declares [#960](https://github.com/mudler/vllm.cpp/issues/960) its
> base and does not start until it lands.** It must not relocate the
> registration itself.

The relocation unblocks the FP8 W8A8 arm on **every** non-CUTLASS CUDA arch, not
only sm_110 and not only NemotronH. That makes it the same shape as #940: a
shared-seam fix whose gate is about the seam, not about the model that needed it
first. Landing it under a NemotronH title would bury a vt/CMake semantics change
and deny it the red-before it deserves. It is dispatched as its own unit.

**Depend on the ISSUE, not on a branch name.** `git ls-remote --heads origin` on
2026-08-16 showed no ref for the dispatched row; it is working locally,
pre-push. A base has to be something git can reach, and today only #960
qualifies.

### 3.3 Everything else the mamba block needs is already native on Thor

Verified, because a second missing registration would change the plan again:

| Op | Registration | TU gated? |
|---|---|---|
| `kCausalConv1dFwd` | `src/vt/cuda/cuda_gdn.cu:6605` | **no** — `cuda_gdn.cu` is an unconditional CUDA source, `CMakeLists.txt:1589` |
| `kMamba2ChunkScan` | `src/vt/cuda/cuda_gdn.cu:6657` | no |
| `kMamba2StateUpdate` | `src/vt/cuda/cuda_gdn.cu` (same registrar) | no |
| `kRmsNormGatedGroup` | `src/vt/cuda/cuda_gdn.cu` (same registrar) | no |

So the governing spec's **R3 block is CLEARED**: #496 W2, the Mamba2 SSD CUDA
arm, landed at `43a6c5518`. Re-verified against the current head rather than
inherited from a sentence, as R3 itself demands.

---

## 4. Design

### 4.1 The weights

`NemotronHMambaWeights` (`nemotron_h_forward.h:212-230`) holds `in_proj` and
`out_proj` as `NemotronHOwned` with `form = kFp8W8A8Static`. A2-Q1 **supplements**
rather than replaces: add `Fp8Weight` fields beside the existing ones, filled by
the loader on the CUDA path, with the `NemotronHOwned` retained for the host
reference arm. This is the tree's existing idiom — `GdnLayerWeights`
(`qwen3_5_weights.h`) already carries "exactly one is filled" pairs.

Replacing outright would delete the host reference's operand and destroy the
comparison the gate is built on.

**`Fp8Weight` carries no `has_input_scale`.** `NemotronHOwned` distinguishes "the
checkpoint shipped 1.0" from "no scale shipped"
(`nemotron_h_forward.h:174`). `Fp8Weight::input_scale` defaults to `0.0F`
(`qwen3_5_weights.h:279`), so the distinction survives without adding a field —
0.0 means unshipped. **Assert it**; do not assume it.

### 4.2 The loader

`LoadFp8` (`nemotron_h_weights.cpp:613-633`) reads the e4m3 bytes, `weight_scale`
and `input_scale` today; the two call sites are `:654-655`. A2-Q1 adds the
`Fp8Weight` population there, computing `alpha = input_scale · weight_scale` at
load exactly as the seam's contract states (`dense_fp8_gemm.h`, `Fp8Weight`
comment at `qwen3_5_weights.h:264-272`).

> **`HostBytesOf` (`nemotron_h_weights.cpp:745`) feeds `rep.host_bytes`, which
> `tests/vllm/models/test_nemotron_h_loader.cpp:310` pins as the exact literal
> `18888922112`.** A copy-preserving supplementation keeps it green. A
> supplementation that *borrows* rather than copies changes it. Either outcome
> is acceptable, but the literal must be re-derived and updated **in the same
> change** with the arithmetic shown, never adjusted to whatever the run
> printed.

### 4.3 The forward

A `NemotronHMamba2MixerDevice` in `nemotron_h_device.cpp`, mirroring the host
arm's op sequence statement for statement, differing only in backend — the
property A2-R established and `scripts/check-fusion-consistency.py` enforces.

The transient conv state and the SSM state stay per-call (§1.1). Device-side
column slicing replaces the host `SliceCols`.

### 4.4 Load accounting will be short, and the report must say so

[#974](https://github.com/mudler/vllm.cpp/issues/974): `ResidentFp8`
(`dense_fp8_gemm.h:97-107`) uploads without `load_stats::AddDeviceUpload` and
without `AdoptDeviceBytesAsHost`, while `ResidentWeight`
(`dense_attn_block.h:197,204`) and `ResidentNvfp4`
(`dense_nvfp4_gemm.h:306,311,316,321`) do both.

**A2-Q1 uses those helpers and does NOT fix this** — it is a device-behaviour
change with its own gate, and `scripts/check-fp4-resident-consistency.py`
currently enforces the obligation for `ResidentNvfp4` alone. **The
implementer's report states that its device-upload total is short by the whole
fp8 tower rather than quoting a number it knows is wrong.**

---

## 5. Gates

### 5.1 The gate

**Per-block numeric equivalence against the host reference on the REAL
checkpoint**, via `NemotronHTrace`: the device mamba arm compared against
`trace.mixer[l]` at **every one of the 23 mamba layers**, plus hybrid-vs-host
token identity. Numeric, not token-only — a token gate cannot see a dequant
fallback, a too-wide dtype or a transposed operand, and this row has been bitten
by all three.

**Hosts:** `dgx.casa` (GB10, sm_121a) and `192.168.68.23` (Thor, sm_110).
Both, per the governing spec's §5.4. The local x86_64 box is a development arm
and a result from it is not a result.

### 5.2 Bands are MEASURED, and the guard is a PROPERTY

Two lessons this row has already paid for, both binding here:

- A bf16 band of `3e-2` sat *above* a `2.11e-2` defect and accepted a fully
  rope'd answer. A2-R's repair is the model: `DevRelFor(dt)`
  (`test_nemotron_h_forward.cpp:1596`) is `1e-5` f32 / `4e-3` bf16, each derived
  from what the two arms actually agree to with the band driven to `1e-9`.
- A "safety factor" guard compared two compile-time constants and observed
  nothing about the running system.

> **Derive every band from a measurement taken in the case. Make the guard a
> property: the perturbed answer, run through the same arithmetic that accepted
> the real one, must come out rejected. No stored twin, no invented safety
> factor.**

### 5.3 Mutations

Applied **alone**, in a scratch copy, rebuilt, run, tree restored to the
**baseline sha** — that restore is the control that catches `shutil.copy2`
preserving mtime so ninja skips the rebuild.

| # | Mutation | Must RED |
|---|---|---|
| Q1-M1 | `input_scale` dropped on the fp8 arm (quant with 1.0) | the mamba numeric gate |
| Q1-M2 | `alpha` folded as `weight_scale` alone | the mamba numeric gate |
| Q1-M3 | one mamba projection left host-side | the mamba numeric gate |
| Q1-M4 | `in_proj` operand transposed (`[K,N]` fed where `[N,K]` is meant) | the mamba numeric gate |
| Q1-M5 | the `zxbcdt` split offsets shifted by one column | the mamba numeric gate |
| Q1-M6 | the device call site deleted, host arm left in place | the gate must go RED — otherwise it measures a class, not a capability (AGENTS.md §"Nothing lands dead") |

**Report per mutation:** the exact `[doctest] test cases:` / `assertions:` /
`Status:` lines, a **non-zero case count**, compile exit AND error count, and a
binary sha256 distinct from baseline.

**Never put a comma in a `TEST_CASE` name** — `-tc` splits on commas, selects
zero cases, prints `SUCCESS!` and exits 0.

### 5.4 Upstream tests owed in the same change

`tests/v1/attention/test_mamba_update_block_table.py:75` and `:178` are owed to
**A2-P**, not here: they gate per-request state indexing, which A2-Q1 does not
create. `tests/v1/worker/test_mamba_utils.py:2136`
(`test_ds_conv_layout_bias_gt_0_byte_equal_to_sd`) is likewise A2-P's, since
§1.1 keeps the persistent conv page out of this unit. **A2-Q1 ports no upstream
test**, and that is a searched result with the paths named, not an omission.

### 5.5 Baseline

Measured 2026-08-16 in a clean detached worktree at
`0e1bee42f16b5f3fb3ae5a23869f6fd97bfc037d`, local x86_64 dev box:
`scripts/agent-preflight.sh` **fully green** (all 26 record gates, including the
seven #873 ones — `check-release-binary-contract`, `check-release-workflow`,
`check-test-registration` — plus both mutation suites). Release build, **0
warnings**, 1441/1441 linked. `ctest -R "nemotron_h|ops_mamba2|ops_fp8|linear_method"`
**13/13 passed**.

**The #873 gates are FIXED.** A red on any of them is the implementer's, not
inherited. Genuinely inherited: `windows-msvc-*` (#584, PR-only),
`test_cpu_x86_llamacpp_floor` under load (`NO_QUIET_WINDOW` exit 4 — the harness
refusing to measure), `test_async_llm` under `ctest -j` (#294 — re-run serially
before calling it a regression), and on GB10 `test_qwen3_5_gdn_spec_routing`
119/123 and `test_linear_method` 83/85 (#907).

---

## 6. ★ G-SAFE stays FULLY intact

All three clauses at `src/vllm/model_executor/models/nemotron_h_registry.cpp:162`
— `input.attn_kv.empty() && input.gdn_state.empty() && input.num_reqs <= 1`.

A2-Q1 creates no paging, no carried state and no batching, so **nothing may be
weakened or narrowed.** A2-P is where it narrows. A reviewer who cannot point at
all three intact returns FAIL.

## 7. A2-Q1 inherits A2-R's UNREACHED posture

`NemotronHDeviceForward` is called only from
`tests/vllm/models/test_nemotron_h_forward.cpp:1805`;
`ForwardNemotronHForCausalLM` still routes to the host `NemotronHForward`
(`nemotron_h_registry.cpp:185-187`). A2-R disclosed this and assigned the wiring
to **A2-P**.

A2-Q1 does not change it, and **its commit body and PR body must say so** —
naming what is unreached, the row that owns the wiring (A2-P) and the issue
(#810), per AGENTS.md §"Nothing lands dead". Silence is not an exception. This
spec lists it under §11 `## Owed`.

---

## 8. Risks

**R1 — #960 does not land, or lands differently.** A2-Q1 stops and reports. It
does not work around the missing registration, and specifically must not add a
NemotronH-local quant path: that is the hand-rolled parallel path AGENTS.md
forbids, and it would leave every other non-CUTLASS arch broken.

**R2 — the fp8 arm silently takes the reference tier on a host nobody checked.**
The tier warns once per (op, device) (`op_provider.cpp:518-520`). **The gate run
must capture stderr and assert the absence of a `[vt reference-tier]` line for
`QuantFp8Static`.** A pass obtained on the portable tier is not a pass for this
arm, and it is invisible in the numbers because the tier is numerically correct.

**R3 — the SSD CUDA arm has never run at NemotronH's shapes on Thor.** #496 W2's
evidence (§8.4 of [`mamba2-ssd.md`](mamba2-ssd.md)) was taken on GB10. Thor is
the portable path. Treat a Thor-only numeric disagreement as a finding about the
SSD arm, not automatically about A2-Q1, and bisect by running
`test_ops_mamba2_ssd` on that host first.

**R4 — `origin/main` moves under a shared checkout.** Merge an immutable SHA and
**re-run the full gate after merging rather than reading the diff** (#818 is
exactly that failure, in this model's own tests). Never force-push.

**R5 — the fixture must be the checkpoint the changed path loads.** Pin revision
`29f2d174`; leave `VT_NEMOTRON35_SNAPSHOT` UNSET so the revision check binds
(`tests/parity/test_hf_snapshot_pinning.cpp:62`) and record the resolved
directory as evidence.

### 8.1 Stop conditions

- #960 has not landed → **stop and report.** Do not implement against a
  described dependency.
- A `quantized_layers` entry names an algorithm we do not implement → refuse by
  name, record as owed, never silently dequantize.
- The gate cannot run on Thor for a reason other than #960 → **`NEEDS_DECISION`**,
  stating the reason; do not land a one-host result as if it were the gate.
- Any temptation to widen a band until it passes → **stop.** §5.2 is the rule
  this row already paid for.

---

## 9. Records owed on landing

A2-Q1 changes lifecycle state, so it owes `docs/STATUS.md`,
`docs/BENCHMARKS.md` (pending, failed or void is a result — silence is not), the
row spec's `## Now`, and the row + checklist entry + rollup in
`.agents/model-matrix.md` in the **same** change
(`scripts/check-model-checklist.py` enforces the rollup). Plus `docs/FEATURES.md`
if the NemotronH row's wording changes, and `docs/USAGE.md` if
`scripts/check-doc-checkpoint.py` classifies anything new as user-facing — run
the checker rather than assuming either way.

It does **not** remove `scripts/runner-routing-allowlist.txt:26`: that entry's
stated removal condition is the device/paged runner path, which is A2-P's.

## 10. Now

**State at this commit:** the device arm is IMPLEMENTED and REACHED, and its
binding measurement is PENDING a lease.

`NemotronHMamba2MixerDevice` (`nemotron_h_device.cpp`) runs the whole block on
the device on the shared FP8 W8A8 seam, and BOTH forwards select it at runtime:
`NemotronHDeviceForward` (non-paged, discarding the recurrence) and
`NemotronHPagedForward` (production, advancing the gathered recurrent rows in
place). The selection is a runtime op-table query plus a weight-form predicate,
never a preprocessor guard, so a dense NemotronH or a device without the fp8 pair
keeps the host bounce.

**Three design points differ from §4 and the reasons are recorded here rather
than left to be re-derived:**

1. **The `Fp8Weight` pair is built ON FIRST DEVICE USE, not by the loader**
   (§4.1/§4.2 put it in the loader). Building it in a `ResidentSlot` the weights
   own is A2-Q2a's newer idiom and it is strictly better here: it does not double
   the 890 MB fp8 tower in host memory at load, it does not upload anything on a
   host-only run, and it leaves `rep.host_bytes` — the literal `18888922112` that
   `test_nemotron_h_loader.cpp:310` pins — untouched, so §4.2's re-derivation
   obligation does not arise at all. The e4m3 staging copy is released as soon as
   `ResidentFp8` has uploaded it, so the peak cost of the conversion is one
   projection.
2. **The upload IS accounted** (§4.4 expected the report to say it was short).
   `dense_fp8::ResidentFp8` still does not call `load_stats::AddDeviceUpload`
   — that is [#974](https://github.com/mudler/vllm.cpp/issues/974), unchanged, and
   the shared header is not touched — so A2-Q1 accounts what IT uploads at the
   site that causes it, exactly as `ResidentWeight` and `ResidentNvfp4` do. The
   counter is then also the instrument the residency case reads, because an arm
   that re-uploaded the tower every step returns identical numbers to one that
   uploads it once.
3. **The paged selection carries an `ssm_dtype == f32` term.**
   `vt::GdnStateGather` widens the page into an f32 working buffer by op
   contract, and the HOST arm then narrows it back to `ssm_dtype` before the
   mixer sees it. On a checkpoint whose `mamba_ssm_cache_dtype` is not f32 the
   two arms would round differently and the per-block numeric gate would be
   comparing two different computations. The released checkpoint resolves f32.

**The comparison is NOT bit-comparable by construction, and §5.2's "measure the
band" is therefore binding rather than cautionary.** The host reference is
W8A16 — `DenseFor` dequantizes the fp8 weight to bf16 and leaves the activation
alone, as `DenseBf16` says outright — while the device arm is W8A8 as vLLM is.
The difference between them is the e4m3 activation quantization, and every band
in `tests/vllm/models/test_nemotron_h_mamba_device.cpp` is measured in the run
against a defect the fixture separates.

### 10.1 Thor (sm_110) RAN the arm, and it answers §3's question

Measured 2026-08-18 under an `rc` lease on `thor:gpu0`, product tree behaviourally
identical to this row's head (the later commits touch tests, scripts and one
comment only). `cmake --build -j 4` returned 0. The feature table read
**`ENABLED for [110]: 1 ; DISABLED cells: 7`** — only `marlin-nvfp4`, with
`cutlass-fp8` and both `scaledmm-c3x` cells DISABLED.

**That configuration is the whole point.** §3.1 measured Thor as having half an
fp8 arm: the GEMM present through `kMatmulFp8CublasLt`, the activation quant
trapped in a CUTLASS-gated TU. #991 moved the registration out. The arm running
here, on a build with no CUTLASS fp8 at all, is what closes that question.

`test_nemotron_h_mamba_device` reported **49 assertions** where a GPU-less box
reports 4, so the device path executed rather than skipping:

| case | result |
|---|---|
| fresh block vs host reference, T=1 / 8 / 12 | agreed 0.164 / 0.282 / 0.309 against a measured band of 0.5; 128 / 1024 / 1536 elements examined; 3 widths covered |
| the fp8 tower uploads ONCE | first call 61760 B == expected 61760 B, second call 0 B |
| refuses a dense projection, refuses a missing `input_scale` | both threw |
| the carry across two legs | **FAILED, and the instrument was the defect** — see below |

Neighbouring suites, same run, all `Status: SUCCESS!`: `test_nemotron_h_forward`
16/16 (5716 assertions), `test_nemotron_h_paged_forward` 12/12 (3256),
`test_nemotron_h_loader` 2/2, `test_nemotron_h_moe_device` 2/2 (29),
`test_ops_mamba2_ssd` 12/12 (2095), `test_ops_fp8_cpu` 5/5 (62).

### 10.2 The carry gate banded a defect smaller than the noise it had to accept

The case banded the SECOND LEG'S OUTPUT against the separation of a dropped
carry. Thor measured the second leg agreeing to 0.705 while a dropped carry
separated by only 0.205, so the derived band (0.102) sat BELOW the deviation a
FRESH leg already shows (0.164 at T=1, from the case above).

That is §5.2's lesson arriving from the other direction. The two arms are W8A8
against W8A16, so a fresh leg already disagrees by the e4m3 activation
quantization and a second leg compounds that with the same disagreement
propagated through the carried state. **A defect whose separation is smaller than
the noise the comparison must accept is not resolvable from that comparison**, and
widening the band until it passes is what §8.1 says to stop for.

The repair gates what the carry IS: the STATE. A dropped carry hands the next leg
zeros, so the separation between the advanced state and a zeroed one is 1.0 by
construction — about six times the noise floor. The conv window and the SSM state
are banded separately, each against its own zeroed twin. The noise floor is
measured in the run at the same width and printed beside the separation, and the
second leg's output carries an assertion only when the separation exceeds twice
that floor, with the condition printed either way.

The failure also exposed two fixture defects. `mamba_ssm_cache_dtype` was unset
and resolved bf16 — NOT the configuration the paged forward selects the device arm
for (`ssm_dtype == f32`), so the cheap arm was gating a path production does not
take. And the whole file was a skip on a GPU-less box, so a CPU-runnable case now
pins the op contract the split depends on.

### 10.2b The repaired carry gate, re-run on hardware, and what it measured

Thor re-run at the branch head: `test_nemotron_h_mamba_device` **5 cases, 63
assertions, 0 failed**, and every neighbouring suite green again. The repair of
§10.2 therefore holds on the silicon that exposed the defect, and the numbers it
prints now justify the repair rather than merely passing:

| quantity | measured | separation | band | margin |
|---|---|---|---|---|
| W8A8-vs-W8A16 noise floor, T=1 | **0.2465** | — | — | the reference for everything below |
| carried conv window | **0.1746** over 576 elements | 1.0 (zeroed) | 0.5 | 2.9x |
| carried SSM state | **0.0614** over 2048 elements | 1.0 (zeroed) | 0.5 | 8.1x |
| second leg's output | 0.7055 | 0.2045 | — | NO assertion, see below |

**The data now proves the diagnosis that drove the repair.** A dropped carry
separates the second leg's output by 0.2045, while the noise the comparison must
accept is 0.2465 — the defect is genuinely SMALLER than the noise, so
`separation > 2 * noise_floor` is false and the case makes no assertion there, by
design. The state comparison carries the case instead, at 2.9x and 8.1x margins.
Had the original band survived, it would have been asserting on a quantity it
cannot resolve.

**One diagnostic was defective and is fixed.** The line that reports whether the
second leg is resolvable printed `1` rather than its prose, because doctest
stringifies a `const char*` as a BOOL and the message streamed a `char*` ternary.
That line's whole job is to make "no assertion was made here" a STATED result
rather than a silent hole, so a version of it that cannot say what it means is
the same class of defect as the band it reports on. It now builds a
`std::string`; reproduced against doctest 2.5.2 both ways before the fix.

### 10.2c The decode-window sampler works, and it quantifies the dilution

Thor re-run, A3 gate with the arm ON, sampler starting only after
`engine loaded in Ns`:

```
RC[a3 on]=0
[nemotron-h] engine loaded in 500.9s
[nemotron-h] TOKEN MATCH: 96/96 over 3 prompt(s) (full rows=3, short rows=0, mode=decode)
[nemotron-h] STRICT PASS
on: GPU busy in 240 of 564 DECODE samples = 42.55% busy
on: decode window 75.418 s (the engine load is OUTSIDE it)
on: per output token 0.785606 s
reference-tier lines in on: 0
```

**42.55% over the decode alone against 15.33% over load+decode.** The load is
500.9 s and the decode is 75.418 s, so the old window was 87% load — the
dilution §10.3 diagnosed is now measured rather than argued.

**The same cross-silicon defect was still live on the per-token line and is
fixed.** That run printed `ratio 54.7x` against vLLM's 0.014369 s, which is a
GB10 figure, for a decode measured on Thor. It is exactly the defect the
busy-fraction reporter carried, and fixing one surface while leaving its twin is
how a wrong comparison survives a correction. The ratio is now quoted only when
`ARCH` is the arch it was measured on; elsewhere the rate still prints and the
comparison is withheld by name. Both arms are pinned in
`tests/scripts/test_nemotron_h_a2q1_per_token.py`, and quoting the ratio
unconditionally reds the suite.

### 10.3 The A/B, on the corrected instrument: the busy fraction ROSE

Second Thor lease, fresh build and clone, sampler measuring the DECODE window
alone. Same binary, same checkpoint, same golden, differing only by
`VT_NEMOTRON_H_DEVICE_MAMBA`:

| flag | mamba arm | A3 | exit | decode GPU busy | per output token | decode wall |
|---|---|---|---|---|---|---|
| `1` (default) | device FP8 W8A8 | `96/96 STRICT PASS` | 0 | **240/564 = 42.55%** | **0.785606 s** | 75.4 s |
| `0` | host, dequant to bf16 | `93/96 DIVERGENCE` | 1 | **700/3808 = 18.38%** | **5.633442 s** | 540.8 s |

**THE BUSY FRACTION ROSE, WHICH IS THIS UNIT'S ACCEPTANCE TEST: 18.38% to
42.55%, +24.17 points, a 2.31x rise.** A decode token costs 7.17x less. Peak host
44070 MiB. `reference-tier lines: 0` on both arms, so neither took the portable
tier.

**Read on the box it was taken on, and nowhere else.** These are sm_110 figures.
The 6.31% baseline and the 0.014369 s per-token reference are BOTH GB10's, so
neither supports a ratio against these numbers, and the instrument now withholds
both comparisons by name on any other arch. What is established is the ON/OFF
difference on one box, and that is exactly the comparison the hypothesis needed.

**The `ratio 54.7x` and `ratio 392.1x` strings in that run's log are stale and
must not be quoted.** That job cloned before the per-token arch gate landed, so it
still printed the GB10 comparison unconditionally; the per-token VALUES are sound
measurements of that box, the ratios beside them are not.

### 10.4 The divergence reproduces, so it is not n=1 on Thor

Two independent Thor leases, separate builds and separate clones, agree exactly
on both arms:

| run | arm ON | arm OFF |
|---|---|---|
| `20260818T222352Z` | 96/96 | 93/96 |
| `20260818T232910Z` | 96/96 | 93/96 |

So the `n=1` caveat is lifted FOR THOR: the host arm's 93/96 is reproducible
there, and the device arm's 96/96 is too. GB10 remains n=1 in the other
direction (the #1157 run, host arm, 96/96).
[#1290](https://github.com/mudler/vllm.cpp/issues/1290) carries this.

**A GB10 run of the DEVICE arm now exists**, and §10.5 records it. This
paragraph said it did not until 2026-08-20; the discriminator run of 2026-08-19
superseded that and the sentence outlived it, so the file contradicted itself at
its own head. What GB10 still lacks is the SPEED axis, not a token result.

**What is still owed.** The occupancy hypothesis is now SUPPORTED on sm_110 and
UNMEASURED on sm_121a, and the two are not interchangeable. Specifically owed:

- the GB10 OCCUPANCY run, which is the only one that can be read against the
  6.31% baseline and the 0.014369 s per-token reference. Its A3 TOKEN result was
  taken on 2026-08-19 (§10.5); the per-token and busy-fraction numbers on `121a`
  were not;
- the §5.1 per-block numeric gate against `trace.mixer[l]` on the real checkpoint
  — the A3 gate is token-level and cannot see a per-layer defect whose argmax is
  unchanged;
- the §5.3 mutations, which belong to the fresh reviewer;
- the three moved tokens on the OFF arm, whose oracle top-2 margin decides
  whether [#1290](https://github.com/mudler/vllm.cpp/issues/1290) is a near-tie
  sensitivity or a wrong answer.

`scripts/nemotron-h-a2q1-dgx-gate.sh` is the recipe, and it now measures the
decode window, withholds both GB10 references off `121a`, and refuses a fraction
outright when the load boundary never appears.

### 10.5 The moved token is UNDETERMINED, it belongs to the device mamba block, and the measurement that decides it is blocked

**Result: NEEDS_DECISION, undetermined. [#1289](https://github.com/mudler/vllm.cpp/pull/1289)
stays held.** The GB10 three-leg discriminator localized the moved token. It did
not exonerate the arm.

**An earlier version of this section is withdrawn.** It was headed "three
independent lines say it is a tie" and it read each line as exculpatory. Every
fact it stated is correct and is kept below. The reading was not: line 1 narrows
suspicion ONTO the changed path rather than away from it, line 2 excludes only a
defect nobody proposed, and line 3 is refuted by the golden's own arithmetic and
by the GB10 polarity. The bottom line did not move, because it was already
"undetermined". The argument for it is now the one the evidence supports.

#### 10.5.1 The discriminator run, and why nothing in this tree can regenerate it

| field | value |
|---|---|
| run | `20260819T200231Z` |
| evidence | `/mnt/nas_share/rc/a2d1-discriminate/20260819T200231Z/` — the same directory a lease sees as `/workspace/a2d1-discriminate/20260819T200231Z/` |
| box | `dgx:gpu0`, GB10, `sm_121a`, under an `rc` lease |
| when | 2026-08-19, `configure.log` 20:02Z, last leg 21:16Z |
| tree | `/root/src-a2d1d` — a clone of this row's branch PLUS an uncommitted diagnostic, see below |
| toolchain | CUDA `13.0.88`, GCC `13.3.0`, `CUDA target architectures: 121a` |
| features | `fa2`, `cutlass-fp8`, `cutlass-nvfp4`, `marlin-nvfp4` ENABLED (`features.txt`) |
| library | `libvllm 0.0.3+cuda (ABI 21, header 21)` |
| checkpoint | `/workspace/a3/ckpt-stage`, revision `29f2d1746d8f41e316523194b19018707749b1b1` |
| golden | `tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json`, oracle `vllm=0.23.1rc1.dev1511+g555967922` |
| contention | `contention.txt` holds the CSV header only: no other compute application held the device |

★ **THE COUNTERS BELOW CANNOT BE REGENERATED FROM THIS TREE, AND NO BINARY HASH
WAS TAKEN.** The `[NH-DIAG] ARM step …` line that carries `state_update_rows`,
`chunk_scan_calls`, `conv_update_rows`, `conv_fwd_calls`, `gathers` and
`scatters` came from an UNCOMMITTED edit to `nemotron_h_device.cpp` in
`/root/src-a2d1d`. None of those six names occurs anywhere in `src/`, and the
committed `[NH-DIAG]` facility prints a different line
(`nemotron_h_device.cpp:1764`, `[NH-DIAG] step T=… R=… nd=… np=… idx=[`). The
logs are real and they are readable at the path above. Nobody can rebuild the
instrument that produced them from a checkout, and nobody can prove which source
the binary was built from. Treat the counters as a recorded observation of one
unreproducible build, never as a gate result. Committing that instrument is owed
in §11.

#### 10.5.2 The three legs

All three legs ran on that one box, from that one build, in that one lease.

| leg | log | decode-step recurrent counters | A3 |
|---|---|---|---|
| device arm ON | `a3_on.log` | `state_update_rows=23 chunk_scan_calls=0 conv_update_rows=23 conv_fwd_calls=0 gathers=0 scatters=0` | `95/96` DIVERGENCE |
| device arm OFF | `a3_off.log` | `state_update_rows=0 chunk_scan_calls=23 conv_update_rows=0 conv_fwd_calls=23 gathers=46 scatters=46` | `95/96` DIVERGENCE |
| host mamba | `a3_hostmamba.log` | `state_update_rows=0 chunk_scan_calls=0 conv_update_rows=0 conv_fwd_calls=0 gathers=46 scatters=46` | `96/96` STRICT PASS |

Both device legs lose the same token, on the same prompt, at the same position:
prompt 2, generation position 32, `11286` where the golden holds `3468`.
Positions 1 to 31 are byte-identical in every leg. The host leg is exact.

**What that settles.** The moved token belongs to the DEVICE MAMBA BLOCK. The
attention, MoE, norm and sampling paths are common to all three legs and cannot
account for a difference between them.

#### 10.5.3 What the three lines actually support

**Line 1 — the two device legs run different recurrent kernels and lose the same
token. This narrows suspicion ONTO the changed path.** The counters exclude the
recurrent reduction order and the gather/scatter state indexing, because neither
is invariant across `state_update_rows=23 gathers=0` and `chunk_scan_calls=23
gathers=46`. The third leg is what the earlier reading omitted: `a3_hostmamba`
is `96/96`, so the token is inside the device mamba block, and the recurrent
kernel is now excluded INSIDE it. What both failing legs still share, and the
passing leg does not, is the FP8 W8A8 projection path — which is what
[#1289](https://github.com/mudler/vllm.cpp/pull/1289) adds. The elimination
points at the change, not away from it.

**Line 2 — the polarity flips with silicon. This excludes only a GROSS
systematic defect, which nobody proposed.** The device arm reads `96/96` on
`thor:gpu0` and `95/96` on `dgx:gpu0`; the host arm reads `93/96` on Thor and
`96/96` on GB10. A defect that corrupted every FP8 projection could not be
`96/96` on Thor, and that is the whole of what the flip rules out. A MARGINAL
perturbation whose token flip depends on the rest of the tower survives it
untouched — and that is the mechanism [#1290](https://github.com/mudler/vllm.cpp/issues/1290)
already applies to the HOST arm. Applying it to one arm and refusing it to the
other is not a reading of the evidence; it is a choice about which arm to
protect.

**Line 3 — the more precise arm is the worse tracker. This is refuted twice.**
The facts hold. Our host arm is W8A16: `nemotron_h.cpp:416-422` selects
`kFp8W8A8Static`, calls `DequantFp8ToBf16`, and says "Weight-only: input_scale
is carried, not applied". The oracle is W8A8:
`FlashInferFP8ScaledMMLinearKernel.input_quant_key()` returns
`kFp8StaticTensorSym` and `apply_scaled_mm` passes `scale_a`
(`vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py:67-85`),
confirmed at run time by the
oracle's own startup line, `Selected FlashInferFP8ScaledMMLinearKernel for
ModelOptFp8LinearMethod`.

The inference from those facts does not follow.

1. **The golden IS a W8A8 computation.** The quantity measured is
   difference-from-reference, not absolute accuracy. A W8A16 arm differs from a
   W8A8 reference by the FULL activation-quantization error that the reference
   applied and the arm did not. A W8A8 arm carries a rounding difference against
   it instead. "More precise" therefore predicts a LARGER distance from this
   golden, which is the ordering observed on Thor. The line reads a confirming
   observation as a contradiction.
2. **On GB10 the ordering REVERSES.** Host (W8A16) is `96/96`; device (W8A8) is
   `95/96`. The earlier text cited Thor only. Neither ordering survives both
   boxes, so neither supports a conclusion about either arm.

#### 10.5.4 The oracle is CONFIG-SENSITIVE here, not non-deterministic — and that decides the gate FORM

`/workspace/nhspeed/oracle.a.out` (2026-08-18, `dgx:gpu0`) ran the SAME
configuration twice inside one process. `ORACLE_LEG 1` and `ORACLE_LEG 2` each
returned `matched=32`, `matched=32`, `matched=26`, for `ORACLE TOKEN MATCH:
180/192`. **At a fixed configuration the pinned oracle is deterministic.** The
committed `32/32` golden came from a DIFFERENT, and unrecorded, configuration.

This is [#926](https://github.com/mudler/vllm.cpp/issues/926), open since
2026-08-15 and not previously linked from this spec.
`tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json` records the
model, the revision, the sampling parameters, and the vLLM, `transformers` and
`flashinfer` versions. It records NO engine configuration, and `af8170154`
committed no generator — that capture ran from `$HOME/venvs/vllm-oracle-next`.
So the `26/32` result compares a known configuration against an unrecorded one,
and the difference cannot be attributed to anything. #926 already names
`enforce_eager`, `gpu_memory_utilization`, `max_model_len` and the batch shape as
the unrecorded terms; the lease logs add one more, and it is the one with a named
accuracy mechanism. `kv_cache_dtype=fp8_e4m3` is AUTO-SELECTED on this checkpoint
(`cache.py:296`, "it may cause accuracy drop without a proper scaling factor")
and the checkpoint does not carry the q scale, so vLLM imputes it:
`kv_cache.py:134`, "Checkpoint does not provide a q scaling factor. Setting it to
k_scale."

**#926 also already recorded the same fork.** Its rebuilt oracle, at this pin
under its own configuration, diverged on prompt 2 at index 29 to `11286`
` transition` — which is the Thor host arm's continuation, position for position.
A pinned vLLM oracle takes that fork. That is evidence about the POSITION, and it
still says nothing about the device arm's position 32.

**A distributional gate is INADMISSIBLE on this evidence.** AGENTS.md permits a
ratified distributional gate only when the oracle's greedy decode is
NON-DETERMINISTIC. `oracle.a.out` measured the opposite. What the evidence
licenses is re-deriving or re-pinning the golden against a NAMED engine
configuration, which is #926's own "what done looks like". Nothing in this
section may be read as a case for weakening the token gate.

#### 10.5.5 The order of work, and why the margin is not first

The oracle's top-2 margin at generation position 32 decides whether this is a
near-tie sensitivity or a wrong answer. `scripts/nemotron-h-a2q1-neartie-gap.py`
is the instrument. It has never produced a number: five runs on `dgx:gpu0` were
each killed by the host-memory watchdog during engine start-up, before a token
existed. That is [#1431](https://github.com/mudler/vllm.cpp/issues/1431).

**[#926](https://github.com/mudler/vllm.cpp/issues/926) precedes
[#1431](https://github.com/mudler/vllm.cpp/issues/1431).** It needs no GPU lease.
Until the golden names the configuration it was captured under, a margin measured
against it is a margin against an unknown, and a `26/32` cannot be told from a
defect. Recording the configuration, or re-capturing under a recorded one, is
cheap and it is prior.

Decoded, the four continuations are one sentence taking a different fork after
"…observed outputs, typically using a":

| arm | positions 30, 31, 32 |
|---|---|
| vLLM golden | ` combination`, ` of`, ` state` |
| Thor `sm_110`, host arm | ` transition`, ` equation`, ` for` |
| GB10 `sm_121a`, device arm | ` combination`, ` of`, ` transition` |
| rebuilt oracle, #926, own config | ` transition`, ` equation`, ` for` |

The competing token is the same one, `11286` ` transition`, in every divergent
row.

### 10.6 REFUTED: the f32 SSM cache is not a too-wide dtype, and it is not mirrored from `transformers`

Recorded because it was raised as a defect during review, is wrong, and is easy
to raise again from the anchors this spec already carried.

**The hypothesis.** `nemotron_h_weights.cpp:855` reads `mamba_ssm_cache_dtype`
from the HF config with a `"float32"` default. vLLM's `CacheConfig` defaults the
same name to `"auto"` (`vllm/config/cache.py:135`), and `"auto"` resolves the
temporal state to the conv state dtype, which is the model dtype, bf16
(`mamba_utils.py:101-108`). So we appear to hold 2x the temporal-state bytes, and
to take the value from `transformers` where vLLM defines something else — which
AGENTS.md forbids, and which a token gate cannot see.

**Why it is wrong.** vLLM does not stop at `CacheConfig`. It has a per-model
verify hook, and NemotronH is one of the models that has one.
`vllm/model_executor/models/config.py:605-631` at the pin declares
`class NemotronHForCausalLMConfig(VerifyAndUpdateConfig)` with
`DEFAULT_MAMBA_SSM_CACHE_DTYPE = "float32"`, documented in its own docstring as
"Only `float32` is known to have no accuracy issues by default". Its
`update_mamba_ssm_cache_dtype` fires when `cache_config.mamba_ssm_cache_dtype ==
"auto"`, reads `getattr(hf_config, "mamba_ssm_cache_dtype",
cls.DEFAULT_MAMBA_SSM_CACHE_DTYPE)`, and writes the result back into
`cache_config`. It is registered for `NemotronHForCausalLM` and
`NemotronHPuzzleForCausalLM` at `config.py:879-880`.

So the hook reads the key FROM THE HF CONFIG and defaults it to FLOAT32 when the
key is absent. `nemotron_h_weights.cpp:855` mirrors vLLM exactly: same key, same
source, same default. The `"auto"` at `cache.py:135` is the CLI default BEFORE
the hook runs, not the value the model gets.

**Observed, not only read.** The pinned oracle logs the hook firing on this
checkpoint in five separate lease runs, including the 2026-08-18 run behind
`oracle.a.out`:

```text
INFO 08-18 20:59:29 [config.py:621] Updating mamba_ssm_cache_dtype to 'float32' for NemotronH model
```

The checkpoint's own `config.json` carries `mamba_ssm_cache_dtype: "float32"`, so
the hook takes the checkpoint's value and the default is not even reached.

**What to fix instead.** Nothing in the dtype. The defect is the CITATION. The
comments at `nemotron_h_registry.cpp:256-263` and `nemotron_h_device.cpp:1586`
cite `mamba_utils.py` alone, which is the half of the chain that makes the value
look unmirrored. Any future reader who checks those anchors reaches the same
wrong conclusion. Add `config.py:605-631` beside them — owed in §11.

## 11. Owed

- [#974](https://github.com/mudler/vllm.cpp/issues/974) — `dense_fp8::ResidentFp8`
  uploads without `AddDeviceUpload` / `AdoptDeviceBytesAsHost`. A2-Q1 consumes it
  and accounts its own upload at its own call site (§10.2); the fix INSIDE the
  shared header is not this unit's, and every other caller of the seam is
  byte-unchanged.
- The §5.1 real-checkpoint per-block numeric gate, the §5.3 mutation pass and the
  GPU-occupancy measurement are PENDING a GB10 lease. They are the unit's
  acceptance test, not paperwork.
- The device arm has **no production caller** until A2-P wires it through
  `ModelRegistry::Forward` (§7). Tracked on
  [#810](https://github.com/mudler/vllm.cpp/issues/810).
- [#926](https://github.com/mudler/vllm.cpp/issues/926) — the golden records no
  engine configuration and has no committed generator, so it cannot be
  re-derived. **This PRECEDES the top-2 margin
  ([#1431](https://github.com/mudler/vllm.cpp/issues/1431)) and needs no GPU
  lease** (§10.5.4, §10.5.5). Until it closes, every comparison against
  `oracle.json` measures a known configuration against an unknown one.
- The `[NH-DIAG] ARM step` recurrent counters that §10.5.1 quotes exist only in
  an uncommitted edit on a lease box. Commit the instrument, or the
  discriminator cannot be re-run.
- **Upstream anchor drift, for #1289's reviewer, NOT repaired here.** Checked
  against the pin `555967922`. Introduced by
  [#1289](https://github.com/mudler/vllm.cpp/pull/1289), in §2 above and in
  `nemotron_h_device.cpp`: `mamba_mixer2.py:550` is the `in_proj` call at
  **554**; `:586` is the `out_proj` call at **585**; `:583` and `:583-585` for
  the gated norm point at the `gate` slice, and `self.norm(ssm_output, gate)` is
  at **582**. Already on `main` and therefore wider than #1289:
  `nemotron_h.py:440` for `self.scaling = self.head_dim**-0.5` is **441** and is
  cited three times (`nemotron_h_device.cpp:332`, `:1882`, `nemotron_h.cpp:668`);
  `nemotron_h.py:627-631` for `residual is None` holds no such statement — the
  four occurrences are at 307, 350, 399 and 521, and `residual = None` is at 618;
  `gdn_attn.py:405` names the branch that ASSIGNS `prefill_has_initial_state`,
  while the comment describes the `else` at **406-407**.
- **The conv-page guard admits `kF32` while its own comment forbids it**
  (§1.1). Pre-existing on `main`; A2-Q1 owns no line of it.
- The `mamba_ssm_cache_dtype` comments at `nemotron_h_registry.cpp:256-263` and
  `nemotron_h_device.cpp:1586` cite `mamba_utils.py` without the per-model verify
  hook that supplies the value, so the anchors read as if we mirror
  `transformers`. §10.6 records the refutation; the anchors still need
  `vllm/model_executor/models/config.py:605-631` beside them.

## 12. Outcome

Not yet written. Filled when the unit reaches `DONE`: what was measured, what
was rejected and why, and why each default is set the way it is.
