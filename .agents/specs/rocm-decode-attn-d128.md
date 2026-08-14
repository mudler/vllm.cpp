# ROCm decode paged-attention at head_dim=128 — the ROCm arm of #382

**Row:** `BACKEND-ROCM` (backend-matrix, `ACTIVE`) — the row this change's code
lives in. Its issue, #382, is filed against `KERNEL-ATTN-PAGED`, the
cross-backend kernel row (state `ANCHOR-BACKFILL`). The two differ deliberately:
the defect is cross-backend, this change is the ROCm arm of it, and
`check-agent-record` requires an active claim to name a `SPIKE`/`ACTIVE` row.
**Claim:** `CLAIM-ROCM-DECODE-ATTN-D128`.
**Issue:** [#382](https://github.com/mudler/vllm.cpp/issues/382) — "decode-opt
attention kernel is head_dim-256 only; head_dim 128 (Qwen3-dense, Llama,
Mistral) falls to the block kernel." This spec is the **ROCm half** of that
issue. The CUDA half already landed as
[PR #425](https://github.com/mudler/vllm.cpp/pull/425) (`66399617`); #382 stays
open because the flip to default-ON is still owed and only CUDA was covered.
**Motivating measurement:** [#488](https://github.com/mudler/vllm.cpp/issues/488)
— the ROCm per-call decode-attention gap (`PagedAttnOnline` 41.1us vs vLLM's
5.10us fallback kernel, 8.1x) on gfx1200. #488 asserts no cause; this spec
supplies one of them and does not close it.
**Base:** `origin/main` `fafa16f0` after rebase (the claim file records the
same), fresh branch (not stacked — this touches
`rocm_paged_attn.hip`/its cross-device test, disjoint from #506's
`rocm_skinny_gemm.hip`/`rocm_matmul_hipblaslt.hip`).
**Board:** AMD Radeon RX 9060 XT (`gfx1200`, Navi 44, RDNA4, discrete, 32
CUs), ROCm 7.2.3, hipClang/Clang 22.0.0.
**Reference checkout:** `${VLLM_SOURCE}` pinned at `5559679229bc` (parity pin
`555967922`).

---

## 1. Reconciliation: this gap already had an issue and a landed CUDA fix

Per the protocol's re-verify-before-claiming rule, this section records what
already existed, because the answer changed the shape of the change.

[#382](https://github.com/mudler/vllm.cpp/issues/382) — filed by an outside
contributor against `KERNEL-ATTN-PAGED` — names this defect exactly, on the
CUDA file, and proposes the same fix this spec implements: template the
decode-opt kernel on elements-per-lane and add an `EPL=4` instantiation.
[PR #425](https://github.com/mudler/vllm.cpp/pull/425) landed that for CUDA.

So the ROCm change is a **mirror of merged work**, not new design. It adopts
the merged arm's flag, default, and stated reason verbatim
(`cuda_paged_attn.cu`, `DecodeD128Enabled`). Where the two backends' facts
differ — and they do, sharply in §5's measurement and materially in §7's dtype
coverage, where this arm is **narrower** than the CUDA one — the difference is
recorded rather than averaged away.

## 2. Why `PagedAttnOnline` was what ran

Reading the ROCm dispatch before this change:

```c
const bool bf16_decode_opt =
    decode_opt && (d == 256 || d == 512) && query.dtype == DType::kBF16 && ...
```

Every fast decode kernel this file has (`PagedAttnDecodeOptBf16T`,
`PagedAttnDecodeGqaBf16`) was gated to `d == 256 || d == 512`. Qwen3-0.6B's
`head_dim` is **128** — the standard value for the great majority of dense/GQA
models this project supports (Qwen3 0.6B/1.7B/4B, Llama, Mistral). So
`bf16_decode_opt` was false for every one of them and every call fell straight
through to `PagedAttnOnline`. Not a gfx1200-specific gap, nor a WMMA-specific
one: the file had no fast path at all for the head size these models use. This
project's own Metal test (`test_metal_backend.cpp:915`, "Qwen3 geometry …
head_dim 128") independently names `d=128` as the real-model shape.

Cross-checked against the pinned oracle (`csrc/rocm/attention.cu` @
`555967922`): its `CALL_CUSTOM_LAUNCHER_BLK_HEAD` dispatch switches on
`head_size` with cases **64 and 128** (`attention.cu:3609-3620`), including on
RDNA4 via the `is_navi_gpu()` (`arch.find("gfx11")==0 || arch.find("gfx12")==0`)
launcher variant. Upstream's fast kernel covers `d=128`; ours didn't.

## 3. Why this is a small change, not a new kernel

`PagedAttnDecodeOptBf16T<EPL>` and `PagedAttnDecodeGqaBf16<QG,EPL,NWARPS>` are
already generic over head_dim via `EPL` (elements-per-lane = `d/32`, one
wavefront's lanes splitting a `d`-wide row). Every online-softmax computation,
shared-memory sizing and warp-shuffle reduction already parametrizes on
`d`/`EPL` with no 256/512-specific assumption in the kernel body. The **only**
hardcoded constraint was in the vectorized load/store helpers:

```c
template <int EPL>
__device__ inline void LoadRowEplBf16(...) {
  static_assert(EPL == 8 || EPL == 16, "EPL");  // uint4-sized loads only
```

`EPL=4` (`d=128`) needs an 8-byte (`uint2`) load instead of a 16-byte (`uint4`)
one — the same pattern, one size down. No new algorithm, tiling, or
synchronization. This is the same observation #382 made about the CUDA file.

## 4. What will change

`src/vt/rocm/rocm_paged_attn.hip`:

1. `LoadRowEplBf16<4>`/`StoreRowEplBf16<4>` — `uint2`-sized vectorized
   load/store, same shape as the existing `EPL==8` branch.
2. A `VT_ATTN_DECODE_D128` gate, **default OFF**, read once beside the existing
   `decode_opt`/`decode_gqa` flags — the same env var, default and rationale as
   the merged CUDA arm.
3. `bf16_decode_opt` gate: `d == 256 || d == 512` → `(d == 128 && (decode_d128
   || decode_wmma)) || d == 256 || d == 512`. The `decode_wmma` disjunct is
   deliberate: the rocWMMA arm (separate spec) is a second, independently
   opt-in kernel for the same head size, and without it a bare
   `VT_ATTN_DECODE_WMMA=1` would be a silent no-op. **Forward reference:**
   neither `VT_ATTN_DECODE_WMMA` nor `rocm-decode-attn-d128-wmma.md` exists yet
   — in the tree or in this PR — so an implementer working from this section
   must land the flag with the rocWMMA arm, not cite it from here.
4. `decode_gqa` fused-head condition extended to `qg == 2 && (d == 128 || d ==
   256 || d == 512)` (Qwen3-0.6B/1.7B are `qg=2`; Qwen3-4B is `qg=4`, not fused
   at any `d` today — falls to per-head `PagedAttnDecodeOptBf16T`, still off
   `PagedAttnOnline`).
5. Two launch switches (`DecodeGqa`, `DecodeOptBf16T`) extended with a
   `d == 128` case.

No change to any prefill path — those stay gated to `d==256||512` and are out
of scope (§7); a `d=128` prefill call already falls through their internal
`else { goto flash_fallback; }` guards to the decode-shaped launch.

### Why default OFF

Verbatim from the merged CUDA arm's reason, which applies identically here: the
arm is correctness-complete but **not byte-exact** against the kernel it
replaces. Warp-strided online softmax reduces the KV sequence in a different
**order** than `PagedAttnOnline`'s per-tile loop, so a greedy anchor can move
at an exact bf16 tie. Shipping OFF keeps every existing golden byte-identical.
The flip owes the near-tie razor, a distributional gate, and regen under the
ratified-tie rule — on **both** backends, and is named as owed in §7.

### Test coverage

`tests/vt/test_backend_cross_device.cpp`: new case, "paged attention at Qwen3
geometry (bf16, GQA 2, head_dim 128) matches the CPU oracle" — mirrors the
Metal "Qwen3 geometry" test's shape (`nblocks=24, bsz=16, hq=16, hkv=8,
dh=128`, 2 requests mixing a 40-token prefill and a 5-token decode-with-context,
so both the prefill fallthrough and the decode dispatch are exercised in one
call), looped over `RegisteredDevices()` so it also covers Metal/CUDA/CPU.
Checks NMSE ≤ 5e-4 against the CPU oracle and `OpProviderStats::declines == 0`.
Genuinely new coverage: the existing generic paged-attention cross-device test
at `d=8, f32` never reached any bf16 `EPL`-templated kernel, so none of them had
bf16 correctness coverage in this suite.

`tests/CMakeLists.txt`: because the arm ships OFF **and** its flag is read into
a `static const bool` — once per process — the default registration only ever
gates the `PagedAttnOnline` fallback. Two extra ctest registrations re-run the
same binary filtered to this case with `VT_ATTN_DECODE_D128=1` and
`VT_ATTN_DECODE_WMMA=1`, so the arms that actually run the new kernels are
gated. Same shape as the existing `test_dense_gateup_fused_marlin_off_*` pair.
Verified non-vacuous (the trap `SKIP_RETURN_CODE 77` exists for, issue #463):
the filter resolves to `test cases: 1 | 1 passed`, `assertions: 6 | 6 passed`,
not zero.

## 5. Evidence

**The gate is live in both directions.** Same binary, no rebuild — an A/B that
was not possible before this change, because the arm was unconditional.
Qwen3-0.6B, 1024-token input, 32 output, concurrency 1, seed 0, GPU lock held:

| `VT_ATTN_DECODE_D128` | TPOT rep1 | TPOT rep2 |
|---|---|---|
| unset (default — `PagedAttnOnline`) | 44.82 ms | 44.82 ms |
| `=1` | 12.80 ms | 12.60 ms |
| | **3.53x** | |

**Decode throughput**, in-engine, 128in/128out, concurrency 1, isolated
same-binary A/B, back-to-back:

| Model | Off (tok/s) | On (tok/s) | Speedup | TPOT off → on |
|---|---|---|---|---|
| Qwen3-0.6B (GQA=2, fused) | 53.84 | 76.84 | **+42.7%** | 13.80 → 8.12 ms |
| Qwen3-1.7B (GQA=2, fused) | 32.23 | 40.30 | **+25.0%** | 26.06 → 19.82 ms |
| Qwen3-4B (GQA=4, per-head only) | 20.70 | 24.38 | **+17.8%** | 42.53 → 35.44 ms |

Single run per cell on a board that may also drive a display — indicative, not
the 2-3x-idle-reproduced standard; matches the caveat already carried by
[rocm-gfx1200-m2-correctness.md](rocm-gfx1200-m2-correctness.md) for this
board. The 1024-token table above is 2 reps and is the tighter number. GQA=2
models (fused kernel) win more than the GQA=4 model (per-head only) —
consistent with the fused kernel halving K/V reloads per its own header
comment, though this run does not isolate that mechanism via a trace.

**The cross-architecture reversal is the most important row in this spec.**
#382 measured the same `EPL=4` arm on sm_110 / Jetson AGX Thor as **1.6x
slower** than the block kernel it replaces (81.6 → 131.0 at c=1, and worse at
c=8). We measure **3.53x faster** on gfx1200. Both can be true: they are
different kernels (`PagedAttentionDecodeOptKernel` vs `PagedAttnDecodeGqaBf16`),
different fallbacks (CUDA's generic *block* kernel vs ROCm's `PagedAttnOnline`),
different wave widths (32 on both, but different occupancy and LDS budgets), and
different memory systems. **What this reversal forbids is a shared default.**
Any future flip to default-ON must be argued per backend with per-backend
measurement; the fact that the ROCm arm is a large win is not evidence for the
CUDA arm, and #382's sm_110 regression is not evidence against this one.

**Correctness**, gfx1200, real hardware: `ctest -R 'rocm|cross_device'` **6/6
pass**, including both new flag-on registrations. Full `ctest` 393 tests,
385 passed / 8 failed; all 8 reproduce identically (same tests, same root cause
`vt: no kernel for op 63 on device type 5`, an unrelated pre-existing ROCm
op-registration gap) on an isolated build of this branch **without** this
change — confirmed not caused by it.

## 6. What this does not claim

- **It does not close #488.** #488 reports a per-call gap against vLLM and
  asserts no cause. This removes one cause. No same-tool per-call re-measure
  against the oracle was run after the change (blocked on the container/glibc
  issue recorded in the WMMA spec), so the residual gap is unquantified.
- **No ceiling.** The next traceable hypothesis is the `qg=4`/`qg=8` fusion gap
  (§7) and, above it, the skinny-GEMM lever in #487, which #488 itself notes is
  the larger share of ROCm decode time.

## 7. Scope

**In scope.** The `rocm_paged_attn.hip` edits in §4, the new bf16 `d=128`
cross-device test and its two flag-on ctest registrations, and this spec.

**Out of scope, named and owed.**
- **The flip to default-ON, on both backends.** Owes the near-tie razor, a
  distributional gate, and golden regen under the ratified-tie rule. Per §5 it
  must be argued per backend, not once. This is what keeps #382 open.
- **rocWMMA for `d=128`.** A second, independently-flagged kernel for the same
  shape — its own spec (`rocm-decode-attn-d128-wmma.md`, landing separately),
  its own claim, its own issue.
- **Every dtype combination except all-`bf16` — and here the ROCm arm is
  narrower than the CUDA arm it mirrors.** `bf16_decode_opt` requires `query`,
  `k_cache`, `v_cache` and `out` to *all* be `kBF16`, and the vectorized helpers
  are bf16 by construction (`LoadRowEplBf16`/`StoreRowEplBf16`). The fallback
  dispatch supports **five** combinations — `bf16/bf16/bf16`, `f32/f32/f32`,
  `bf16/bf16/f32`, `bf16/f32/bf16`, `f32/bf16/f32` — so **four of the five still
  fall to `PagedAttnOnline` at `d=128`**, which is the exact fallback this arm
  exists to get off.

  The CUDA arm does not have this limitation. It added `LoadRowN<4, float>`
  *beside* `LoadRowN<4, __nv_bfloat16>`, and its `d == 32 * 4` launch branch
  carries no dtype gate at all, so f32 reaches the decode-opt kernel there
  (`cuda_paged_attn.cu:2796` dispatches `LaunchDecode<TQ, TKV, float>` for
  `out.dtype == kF32`). §1 calls this change a mirror of merged work; **on dtype
  coverage it is not one**, and that is recorded here rather than left to be
  discovered.

  The limitation is pre-existing, not introduced: ROCm's decode-opt has been
  bf16-only at *every* head_dim, so this is a gap the `d=128` arm inherits
  rather than creates. It is owed, not intended, and it is not waived — an f32
  `d=128` decode on ROCm is silently slow rather than refused, which is the
  weaker of the two failure modes AGENTS.md allows.
- **`qg=4`/`qg=8` GQA fusion at any `d`.** `PagedAttnDecodeGqaBf16`'s fused
  condition only ever covered `qg==2` and `qg==8 && d==512`; `qg=4` (Qwen3-4B)
  was never fused at 256/512 either. Real, but head-dim-independent — its own
  follow-on.
- **Prefill at `d=128`.** Prefill kernels stay `d==256||512`-only.
- **Explaining why GQA=2 wins more than GQA=4** via kernel tracing. Named, not
  run.

## 8. Risks and decisions

| Risk | Assessment |
|---|---|
| The reduction-order change moves a greedy anchor at a bf16 tie | This is why the arm ships **default OFF**, adopting the merged CUDA arm's flag, default and stated reason verbatim rather than inventing new ones. A default-ON flip is a separate, per-backend argument and is explicitly out of scope here. |
| The 3.53x is a single-board, single-run figure | Measured on one gfx1200 that may also drive a display. It is indicative, not the idle-box reproduced standard AGENTS.md requires for a binding number, and §5 says so. It justifies building the arm; it does not license a BENCHMARKS entry or a default flip. |
| The same arm measured 1.6x SLOWER on sm_110 (#382) | Recorded, deliberately not reconciled. It is the reason the default stays OFF and the reason the flip must be argued per backend rather than once. Treating the ROCm number as settling the question for all boards is the error this row is guarding against. |
| The spec lands before its code | Intended, and required — AGENTS.md puts the spec before implementation. The consequence is that §4 and the result section describe an unmerged branch, which the banner above the result section states outright so no reader mistakes it for landed work. |
| `VT_ATTN_DECODE_WMMA` is cited but does not exist | A forward reference to a sibling arm whose spec and issue are not yet filed (§4). An implementer must land the flag alongside the rocWMMA arm; taking §4 literally today produces a reference to an undefined symbol. |
| The residual #488 gap is unquantified | No post-change per-call oracle re-measure was run, so how much of #488 this closes is unknown. Named here rather than left implicit; it is owed before #488 can be judged. |

## 9. Stop conditions

- **Stop if the token-exact gate moves with the flag OFF.** The arm is opt-in;
  flag-off behaviour must be byte-identical to today. Any drift means the gate
  predicate is wrong, not that the golden needs refreshing.
- **Stop if the flag-ON arm cannot be shown to reach the new kernel.** A green
  gate that never entered `PagedAttnDecodeOptBf16T` proves nothing — the ctest
  registrations in §4 exist precisely so the opt-in arms are gated rather than
  silently skipped. Confirm selection counts, not just tokens.
- **Stop before flipping the default.** That needs its own row, and needs the
  sm_110 reversal in §6 reconciled rather than out-voted by one board.
- **Stop if the measured speedup on a second board contradicts the first.**
  Two boards disagreeing is a finding about the kernel, not noise to average.
- **Do not extend this row to the rocWMMA arm.** It is a separate, independently
  flagged kernel with its own spec and its own issue, both still unfiled.

## 10. Reproduction

```sh
nix develop .#rocm-shell --command bash -c '
  cmake -S . -B build-hip -G Ninja -DVLLM_CPP_HIP=ON \
    -DVLLM_CPP_HIP_ARCHITECTURES=gfx1200 -DROCM_PATH=$ROCM_PATH \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build build-hip -j"${JOBS:-8}"
'
flock "$HOME/gpu.lock" -c '
  nix develop .#rocm-shell --command ctest --test-dir build-hip \
    -R "rocm|cross_device" --output-on-failure
'
# Flag A/B, same binary:
flock "$HOME/gpu.lock" -c '
  build-hip/examples/vllm-bench --model <Qwen3-0.6B> --num-prompts 2 \
    --input-len 1024 --output-len 32 --concurrency 1 --seed 0
  VT_ATTN_DECODE_D128=1 build-hip/examples/vllm-bench --model <Qwen3-0.6B> \
    --num-prompts 2 --input-len 1024 --output-len 32 --concurrency 1 --seed 0
'
```

## Result on the implementation branch (2026-08-12)

> **Not landed.** This section records what was built and measured on the
> unmerged implementation branch. No `VT_ATTN_DECODE_D128` exists in
> `src/vt/rocm/` on `main` — `git log -S'VT_ATTN_DECODE_D128' -- src/vt/rocm/`
> is empty, and `rocm_paged_attn.hip` still gates on `d == 256 || d == 512`.
> This spec is committed BEFORE its implementation, per AGENTS.md; the section
> becomes `## Outcome` when the code merges and the row reaches `DONE`.

**Built the ROCm `d=128` decode arm, default OFF, mirroring the merged CUDA
arm of the same issue.** Root cause for the ROCm decode-attention gap #488
measured was neither architecture- nor WMMA-specific: no fast decode kernel
existed for `head_dim=128`, the size every locally-tested model uses, on any
board. #382 had already named this and PR #425 had already fixed the CUDA half;
this is the mirror, adopting that arm's flag (`VT_ATTN_DECODE_D128`), default
(OFF) and reason (reduction-order change can move a greedy anchor at a bf16
tie) rather than inventing new ones.

Measured on gfx1200 with the gate exercised both directions on one binary:
**3.53x** TPOT at 1024-token context, and +42.7% / +25.0% / +17.8% decode
throughput on Qwen3-0.6B / 1.7B / 4B. New bf16 `d=128` GQA correctness coverage
where none existed, plus two flag-on ctest registrations so the opt-in arms are
actually gated rather than silently skipped.

**The finding worth carrying forward is the reversal:** #382 measured this same
arm 1.6x *slower* on sm_110, where we measure it 3.5x *faster*. That is
recorded, not reconciled, and it is the reason the default-ON flip must be
argued per backend. Rejected: landing default-ON on the strength of the ROCm
number alone.
