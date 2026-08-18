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

**State at this commit:** spec only. No product code, no lifecycle change.

A2-Q1 is **BLOCKED on #960** and is not claimable until it lands. When it does, a
fresh implementer claims this file, captures a RED per §5.3 first, and lands the
mamba device arm with G-SAFE untouched. A fresh reviewer — never the implementer
— runs the §5.3 mutations.

**Three things to read before the first edit:** §3, so the Thor dependency is
consumed rather than re-derived; §1.1, so the conv-state dtype is left to A2-P;
and §5.2, so the bands are measured rather than chosen.

## 11. Owed

- [#974](https://github.com/mudler/vllm.cpp/issues/974) — the fp8 resident
  helpers upload without `AddDeviceUpload` / `AdoptDeviceBytesAsHost`. A2-Q1
  consumes them and reports its accounting as short (§4.4); the fix is not this
  unit's.
- The device arm has **no production caller** until A2-P wires it through
  `ModelRegistry::Forward` (§7). Tracked on
  [#810](https://github.com/mudler/vllm.cpp/issues/810).

## 12. Outcome

Not yet written. Filled when the unit reaches `DONE`: what was measured, what
was rejected and why, and why each default is set the way it is.
