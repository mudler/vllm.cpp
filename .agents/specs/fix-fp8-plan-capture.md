# FIX-FP8-PLAN-CAPTURE-1843 — the fp8 plan cache defaults ON, because the uncached fp8 lane is wrong under CUDA-graph capture on CUDA 13.3

Row: `FIX-FP8-PLAN-CAPTURE-1843`
Issues: [#1843](https://github.com/mudler/vllm.cpp/issues/1843) (primary),
[#1732](https://github.com/mudler/vllm.cpp/issues/1732) (the root defect, whose
bf16/f32 half is PR [#1741](https://github.com/mudler/vllm.cpp/pull/1741))
Base SHA: `364f2a898`
Upstream pin: see [`.agents/upstream-sync.md`](../upstream-sync.md)
(`5559679229bc961848b121ccdeaa8fa5d79bec98`). vLLM is immune structurally
(torch caches the selected cuBLASLt algo per shape and vLLM warms eagerly
before capture — the argument `#1741`'s spec records); the mirrored behavior is
"never query the heuristic under capture", and the cache is the vt-runtime
vehicle for it (inventory deviation, like `gemm_plan_cache.h`).
Matrix: [`.agents/kernel-matrix.md`](../kernel-matrix.md).

## Now

`ACTIVE`: the implementation is committed on `row/FIX-FP8-PLAN-CAPTURE-1843`
(parent `6a7a678bc`). T1 (red-first polarity) and T2 (ctest re-pin) are green on
the CPU tier — see `## Evidence`. T3 (the GPU default-env graphed gate) stays
`PENDING` until #1741 lands on main, per `## Owed`. The gap and the fix are both already measured
(2026-08-24, `dgx:gpu0` lease, GB10 sm_121a, lease-staged CUDA 13.3.73,
`nvidia/Qwen3.6-35B-A3B-NVFP4`, logs
`/mnt/nas_share/rc/gdn-moe-packed-ba/logs/`): with #1741's bf16/f32 cache in
the tree, the graphed 35B gate still dies at the first in-capture
`cublasLtMatmulAlgoGetHeuristic` on the fp8 lane, and with
`VT_FP8_PLAN_CACHE=1` it passes token-exact on all three arms; the fp8 cache
alone (no #1741) fails on the bf16-TN lane. Both halves are needed; this row
is the fp8 half. #1741's own spec owed exactly this ("extend the same
mechanism to the fp8 heuristic call sites ... or flip `VT_FP8_PLAN_CACHE`
default, measured on a Blackwell host with CUDA 13.x").

## Scope

- **E1.** Flip the `VT_FP8_PLAN_CACHE` contract in
  `src/vt/cuda/fp8_plan_cache.h`: `Fp8PlanCacheFlagIsOn` becomes default ON
  with exactly `"0"` disabling (the same shape as `GemmPlanCacheFlagIsOn` in
  #1741's `gemm_plan_cache.h`: a typo cannot resurrect the capture bug).
  Rewrite the header's DEFAULT-OFF paragraph: the 2026-07-18 perf measurement
  (wall-clock NEUTRAL, `CLAIM-FP8-PLAN-CACHE-1`) stands and stays quoted, but
  it no longer decides the default, because on CUDA 13.3 the uncached path is
  WRONG under capture (#1732 on the fp8 lane, measured #1843) — the cache is
  not a performance knob any more. `"0"` remains the rollback / A/B arm.
- **E2.** Update the two consumer comments in `src/vt/cuda/cuda_matmul.cu`
  (~`:740-744` "Cache OFF (default) ... the shipped production path", and the
  alphavec site ~`:887`) to the new polarity, naming #1843 and the capture
  reason.
- **E3.** Tests follow the contract: the polarity case in
  `tests/vt/test_fp8_plan_cache.cpp:32` ("OFF by default; ON only for exactly
  \"1\"") is rewritten red-first to the new table (unset/""/"1"/junk → ON;
  exactly "0" → OFF; " 0"/"00" → ON, so a mangled rollback fails loud toward
  correctness). The env-pinned ctest arm
  `test_ops_fp8_cutlass_plan_cache_on` (`tests/CMakeLists.txt` ~`:2205-2213`,
  `VT_FP8_PLAN_CACHE=1`) becomes the redundant arm once ON is the default:
  re-pin the pair as default (cache ON) and `VT_FP8_PLAN_CACHE=0` (the
  fresh-plan rollback), renaming registrations to say which is which, and
  update the comment at `tests/vt/test_ops_fp8_cutlass.cpp:393-394`.
- **E4.** Audit and record: `grep -rn AlgoGetHeuristic src/vt/cuda/` must show
  every EXECUTABLE call site is behind either `GetOrQueryGemmHeuristic`
  (#1741, not yet on main — do not depend on it) or `Fp8PlanCacheEnabled()`.
  At this base the only executable sites are the two fp8 lanes in
  `cuda_matmul.cu` (`:744` per-tensor TN, `:887` alphavec), both behind the
  flag; `fp8_plan_cache.h` matches are comments. The "fp8 block dispatch"
  #1741's owed names is cutlass-routed and makes no cuBLASLt heuristic call —
  verify and state it in the spec's `## Evidence`, or cover it if the grep
  says otherwise.
- **E5.** Records: this spec; a `FIX-FP8-PLAN-CAPTURE` row in
  `.agents/kernel-matrix.md`; an appended `.agents/issue-index.md` row for
  #1843 (owner `FIX-FP8-PLAN-CAPTURE-1843`); any doc that describes
  `VT_FP8_PLAN_CACHE` as opt-in (grep `docs/`; `scripts/env-doc-allowlist.txt`
  already carries the name and needs no edit unless the checker asks).

Out of scope: #1741's bf16/f32 cache (its own PR, its own review); any change
to `Fp8PlanKey`, `BuildFp8Plan`, `GetOrBuildCachedFp8Plan`, or algo selection;
the eager-warm premise (a NEW fp8 shape first seen inside capture would still
query and fail — same residual as #1741, named under `## Risks`).

## Design

One predicate change plus its records:

```cpp
// fp8_plan_cache.h — Fp8PlanCacheFlagIsOn becomes:
inline bool Fp8PlanCacheFlagIsOn(const char* env_value) {
  return !(env_value != nullptr && std::string_view(env_value) == "0");
}
```

Bit-exactness is the header's own recorded argument, unchanged: cuBLASLt algo
selection is process-deterministic per shape, and
`test_ops_fp8_cutlass.cpp` already verifies the cached plan byte-exact against
a fresh-plan GEMM. The flip changes WHEN the heuristic runs (once per shape,
outside capture whenever the shape was warmed), never WHAT it returns. The
memory-lifetime story is also unchanged: the cache is per-device, bounded by
the finite fp8 shape set, values leak by design like `gemm_plan_cache.h`.

## Risks

1. **A shape first seen inside capture still queries and still fails.** True
   for #1741's cache too; the engine's eager warm step covers the decode
   shapes (the premise #1741's spec grounds at the `~GraphCaptureScope`
   comment), and the GB10 measurement (all three graphed arms pass) is the
   empirical form. Named, not solved here.
2. **A user relying on the old default (fresh plan per call) changes
   behavior.** The observable difference is timing-only per the neutral A/B;
   `VT_FP8_PLAN_CACHE=0` restores it exactly.
3. **The CUDA-tier plan-cache arms cannot run on this host.** The polarity and
   key tests are CPU-tier; the fp8 GEMM arms are CUDA-gated and re-run on the
   GPU host with the final gate.

## Tests

- **T1 (red-first, CPU).** Rewrite the polarity case in
  `tests/vt/test_fp8_plan_cache.cpp` to the new table and run it against the
  UNCHANGED header first: the `nullptr → ON` assertion must fail. Capture the
  red output, then apply E1 and show green. Every other case in the suite
  (key equality/hash, refusal tags) must stay green untouched.
- **T2 (CPU).** `test_ops_fp8_cutlass` compiles and its CPU-reachable cases
  pass under both ctest arms after the E3 re-pin.
- **T3 (GPU, operator, after #1741 also lands).** The graphed 35B gate at the
  DEFAULT env (no `VT_FP8_PLAN_CACHE` set) on the merged tree: token-exact on
  all three arms of `.agents/specs/gdn-moe-packed-ba.md` `## Gates`, plus the
  same-binary `VT_FP8_PLAN_CACHE=0` arm reproducing the in-capture failure
  (the A/B #1741's board gate ran for its own cache). This is the measurement
  that closes #1843 and un-PENDs `GDN-MOE-PACKED-BA`'s graphed gates.

## Gates

| Gate | Command | Owner |
|---|---|---|
| focused red→green | `ctest --test-dir build -R 'fp8_plan_cache|ops_fp8_cutlass' --output-on-failure` | implementer |
| preflight | `scripts/agent-preflight.sh --staged` exit 0 | implementer |
| reviewer mutation | revert the polarity (restore `== "1"`) in a scratch copy → T1 red; drop `"0"` from the disable set → the exact-"0" case red | reviewer |
| GPU default-env gate | T3 on `dgx:gpu0` in an `rc` lease | operator |

## Evidence

Implementer (CPU, no GPU; 2026-08-24), measured on the tree of this row's
implementation commit (parent `6a7a678bc`), CPU configure
`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SERVER=OFF`. The motivating GB10 measurement
stays recorded in #1843.

- **T1 red (before E1, against the unchanged `== "1"` parser).** The rewritten
  polarity case fails for exactly the intended reason — every ON-side value
  reads OFF: `test cases: 1 | 0 passed | 1 failed`,
  `assertions: 12 | 2 passed | 10 failed`, exit 1. The 10 red assertions are
  nullptr, `""`, `"2"`, `"on"`, `"true"`, `"11"`, `"1 "`, `" 1"`, `" 0"`,
  `"00"` (each `CHECK( Fp8PlanCacheFlagIsOn(...) ) is NOT correct! values:
  CHECK( false )`); the 2 green are `"1"` -> ON and `"0"` -> OFF, shared by
  both polarities.
- **T1 green (after E1).** The polarity case: 12/12. The full
  `test_fp8_plan_cache` suite: `test cases: 10 | 10 passed | 0 skipped`,
  `assertions: 153 | 153 passed` — every key/refusal/splitK case untouched.
- **T2.** `ctest --test-dir build -R 'fp8_plan_cache|ops_fp8_cutlass'
  --output-on-failure`: 4/4 pass (`test_fp8_plan_cache`,
  `test_ops_fp8_cutlass`, `test_ops_fp8_cutlass_plan_cache_default_on`,
  `test_ops_fp8_cutlass_plan_cache_rollback_off`), exit 0.
  On this no-GPU box all 8 `test_ops_fp8_cutlass` cases pass with
  `assertions: 0` (every case is CUDA-gated) — T2 here pins compile +
  registration + the arm env pins, not the device numerics; the device leg is
  T3's, as `## Risks` 3 states.
- **Implementer mutations (both restored byte-for-byte, sha256-verified;
  post-restore full suite 153/153).**
  (a) restore the old `== "1"` polarity -> the polarity case reds at 10/12
  (nullptr -> ON is the first failure), Status: FAILURE.
  (b) remove `"0"` from the disable set (`"0"` -> `"off"`) -> exactly one red:
  `CHECK_FALSE( Fp8PlanCacheFlagIsOn("0") ) is NOT correct! values:
  CHECK_FALSE( true )`, 11/12 pass, Status: FAILURE.
  Trap recorded for the reviewer: `return true;` as mutation (b) does NOT
  compile (`-Werror` unused-parameter), and a swallowed ninja failure leaves
  the stale green binary running — a mutation that fails to build reads as a
  passing test. Pick a compilable mutant and force the object rebuild.
- **E4 audit.** `grep -rn AlgoGetHeuristic src/vt/`: executable CUDA sites are
  `cuda_matmul.cu:266` (rowmajor-NN), `:346` (BT TN), `:432` (batched) — the
  bf16/f32 lanes, #1741's own scope (its `GetOrQueryGemmHeuristic` cache, not
  yet on main) — and `:577` inside `BuildFp8Plan`, the SINGLE fp8 heuristic
  call, reached only from the two fp8 consumers (`:744` per-tensor TN, `:887`
  alphavec), both routed through the `Fp8PlanCacheEnabled()` branch. The
  `fp8_plan_cache.h` matches are comments; `rocm_matmul_hipblaslt.hip:376` is
  the hipBLASLt lane (no CUDA-graph capture). Block-fp8 is cutlass-routed:
  no `cuda_matmul_fp8_block*` file exists, and `grep -n cublasLt
  src/vt/cuda/cuda_matmul_fp8_cutlass.cu` finds one comment (workspace
  analogy), zero calls — no cuBLASLt heuristic call on the block path. No
  third fp8 lane, so the stop condition did not fire.
- **E5.** `grep -rn VT_FP8_PLAN_CACHE docs/` -> no hits;
  `scripts/env-doc-allowlist.txt:56` already carries the name (no edit).
  `python3 scripts/check-test-registration.py` -> exit 0 after the arm rename.
  Pre-existing drift resolved in passing: `tests/vt/test_fp8_plan_cache.cpp`'s
  header comment has said `default ON, "0" rollback` since the cache's birth
  commit `df9a0406e` while the code shipped OFF; this row makes the code match
  the comment.

Operator (GPU): T3 to be recorded here after #1741 lands, per `## Owed`.

## Owed

- T3 cannot run until #1741 lands on main; until then the default-env graphed
  gate stays `PENDING` and #1843 stays open.

## Stop conditions

- If the E4 grep finds an executable heuristic call site outside the two
  flagged lanes, stop and return `NEEDS_DECISION` with the site named — that
  is a third lane, not a polarity flip.
- If any `test_ops_fp8_cutlass` case depends on the OLD default semantically
  (not just via the env pin), do not weaken it; return `NEEDS_DECISION`.
