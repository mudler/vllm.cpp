# ROCm M4 — the pinned vLLM-ROCm oracle on gfx1100 and the ROCm near-tie gate lane

## Goal

Give `BACKEND-ROCM` (issue #41) the milestone that only this box can host: the M4
correctness gate — our ROCm paged engine held against a **pinned vLLM-ROCm oracle
running on the same gfx1100 hardware**, not against the dgx (CUDA) capture.

## The oracle

- **Pin:** upstream vLLM commit `5559679229bc961848b121ccdeaa8fa5d79bec98`
  (runtime identity `0.23.1rc1.dev1511+g555967922`, verified against the
  parity-pin block in `.agents/upstream-sync.md`).
- **Base image:** `rocm/vllm-dev:base` (HIP 7.2.5 userland, host ROCm 7.14
  driver). `PYTORCH_ROCM_ARCH=gfx1100` covers the 7900 XTX.
- **Build:** `python3 setup.py bdist_wheel` inside the container after
  `apt-get install binutils mold` (the base image ships no linker) and the
  pyproject build-system requirements. Wheel:
  `vllm-0.23.1rc1.dev1511+g555967922.rocm723-cp312-cp312-linux_x86_64.whl`
  (2,696 files; `_C`/`_rocm_C`/`_moe_C` ABI extensions present).
- **Committed image:** `vllm-rocm-oracle:555967922-gfx1100`
  (`/home/vikash/oracle/Dockerfile.oracle`), wheel installed with deps.
- **Determinism:** K=10 per-prompt greedy capture on the 16 gate prompts is
  **deterministic in every cell** (0 multi-member (prompt,pos) cells) — a
  well-posed strict gate on this board, matching the dgx finding for batch=1
  `enforce_eager=True`.

## The gate lane

`tests/parity/test_qwen3_paged_engine.cpp` already has device-aware goldens for
Metal and Tenstorrent (`our_ids_<dev>.npy` + `neartie_gap_mnats_<dev>.npy` under
the shared `qwen3_greedy_0_6b/` dir, with the dgx CUDA pair as base). This spec
adds `kROCM` to that lane, unchanged logic:

- hard anchor REQUIRE (our deterministic tokens vs the committed ROCm anchor),
- near-tie band ≤ 500 milli-nats (vLLM teacher-forced on OUR exact prefix),
- backend proof (all 8 Qwen3-dense ops `selections>0 ∧ declines==0` on kROCM,
  with the fused-RoPE alternative counted as in the existing lane).

New committed goldens (captured on gfx1100, 4x RX 7900 XTX, ROCm 7.14,
`enforce_eager=True`, batch=1):

| file | content |
|---|---|
| `our_ids_rocm.npy` | our engine's greedy tokens (16×16) |
| `neartie_gap_mnats_rocm.npy` | oracle teacher-forced gaps in milli-nats (16×16) |
| `greedy_ids_rocm.npy` | the ROCm oracle's own greedy (evidence; the base `greedy_ids.npy` stays the dgx capture) |
| `greedy_dist_rocm.npy` | K=10 run set (evidence) |

Flow (the three commands, all GPU-locked):
1. oracle capture: `scripts/qwen3-oracle-capture.py --runs 10 --per-prompt` in the
   committed container against `/models` (Qwen3-0.6B).
2. bootstrap dump: `VT_DUMP_IDS=1 ./build-hip/tests/test_qwen3_paged_engine`
   → `our_ids_rocm.i32`.
3. teacher-forced gaps: `scripts/qwen3-neartie-gap.py --golden-dir ...` in the
   container → `our_ids_rocm.npy` + `neartie_gap_mnats_rocm.npy`.

## Result (Qwen3-0.6B, gfx1100)

**16/16 prompts PASS** — STRICT token-exact 11/16 vs the base greedy, 5/16 via
the near-tie band, **max gap 0.125 nats** (prompt[3] tok=1), **0
forward-divergent**; backend proof: kPagedAttention selections 7,168, 0 declines;
125/125 assertions. The 28 token-divergent positions vs the oracle's own greedy
are all within the band; the known p0 France/Italy flip is a literal 0.0000-nat
tie in the oracle's own logits.

## Boundaries

- No source/kernel change; the only edited file is the parity gate test.
- The dgx base goldens are byte-untouched; the ROCm additions are purely additive.
- Qwen3-4B (the strict deterministic dense) is not on this box (disk); the 0.6B
  near-tie-robust gate is the M4 evidence for the lane.
- The oracle image and `/home/vikash/oracle/` scratch are machine-local, not
  committed; this spec is the reproduction recipe.

## Review rework (2026-08-13 sweep, localai-bot on #559)

The first shape landed the 0.8B gate RED with the pre-fix goldens committed.
Review findings, all accepted and fixed in the current shape:
1. **CI-red manifest**: `qwen35_0_8b_greedy` had no runner — added to
   `PendingRunnerOps()` in test_op_parity.cpp (the `qwen36_gguf_greedy`
   precedent); the manifest itself stays (its oracle identity is the point).
2. **Fail-safe by device**: the gate now exits 77 (CTest Skipped) on any
   non-ROCm device instead of comparing a foreign engine against ROCm-derived
   goldens, and the checkpoint-absent path exits 77 too (issue #463's pattern)
   instead of a silent `return` that printed SUCCESS with 0 assertions.
3. **Green-shaped landing**: the committed goldens are the FIXED engine's
   sequence, oracle-re-derived (the kernel fix lands below this commit in the
   stack); the RED capture (13/16, first-divergence gaps 0.375-1.062 nats —
   5 of 6 over band) stays as evidence here and in the parity ledger, not as
   committed goldens that no code can pass.
4. **Anchor message honesty**: anchor drift now reads "REGRESSION SUSPECTED:
   bisect the engine change first" — re-deriving goldens is the last step of a
   justified re-capture, never the response to a failure.

## Issue #1222: missing-artifact fail-safe repair

### Scope

The Qwen3.5-0.8B paged-engine gate promises that an unavailable prerequisite is
not a correctness pass. Repair the three required committed-artifact paths in
`tests/parity/test_qwen35_paged_engine.cpp`: `greedy_ids.npy`, `our_ids.npy`,
and `neartie_gap_mnats.npy`. Issue #1222 is an in-flow `BACKEND-ROCM` bug for PR
#559. Snapshot absence, non-ROCm execution, malformed present arrays, and the
numerical acceptance policy keep their existing behavior.

### Design

Put the three existence checks behind one prerequisite helper used by the real
gate and by a test-only golden-directory probe in the same executable. Missing
required files route through the existing `SkipGate` process exit 77. A
deterministic CMake subprocess test creates temporary golden directories,
removes each required file in turn, invokes the actual gate executable through
the probe, and rejects any child result other than 77.

`VT_DUMP_IDS` remains the explicit bootstrap path when the greedy capture exists
but the anchor/gap pair is incomplete. It can run the model and write
`our_ids.i32`; after that bootstrap work it exits 77 because no correctness gate
ran. Present files continue into `LoadNpy` and the existing dtype/shape checks,
so malformed arrays remain hard failures rather than becoming skips.

### Risks

- A probe that reimplements the condition could pass while the real gate stays
  unsafe. The probe therefore calls the same prerequisite helper in the same
  process as the production test case.
- Treating any parse failure as unavailable would hide corrupt evidence. Only
  filesystem absence is eligible for exit 77.
- Returning normally after bootstrap would preserve a second false-success
  path; bootstrap completion must terminate through `SkipGate`.
- Artifact checks must happen before model construction only in probe mode;
  the ordinary snapshot and device gates retain their ordering and behavior.

### Tests

1. Add the subprocess regression and capture semantic RED before the repair:
   children missing greedy, anchor, and gap return 0 instead of 77.
2. Rebuild every affected target and require all three children to return 77.
3. Under `flock /home/vikash/gpu.lock` with
   `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib`, run the real pinned
   Qwen3.5 gate and require 16/16 prompts and 137 assertions, not exit 77.
4. In scratch, restore one repaired missing-artifact branch to a normal return,
   freshly rebuild, and require the deterministic subprocess test to fail for
   the corresponding child. Restore byte-for-byte, rebuild, and require green.
5. Run `PATH=/usr/bin:$PATH GIT_CONFIG_GLOBAL=/dev/null
   scripts/agent-preflight.sh` and require literal `All gates green.`

### Evidence

- RED build: `flock /home/vikash/gpu.lock env
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib cmake --build build-hip
  --target test_qwen35_paged_engine --parallel 4` exited 0. The new test then
  ran with `ctest --test-dir build-hip --output-on-failure -R
  '^test_qwen35_paged_engine_prerequisites$'`. CTest exited 8 because the
  greedy, anchor, and gap children each exited 0 with zero assertions and
  doctest `Status: SUCCESS!`.
- Focused GREEN: the same build command compiled the changed test object and
  linked the HIP executable, then exited 0. The same CTest command exited 0.
  Direct CMake-script execution exited 0 and printed child exit 77 for
  `greedy_ids.npy`, `our_ids.npy`, and `neartie_gap_mnats.npy`.
- Real model gate: `flock /home/vikash/gpu.lock env
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib
  ./build-hip/tests/test_qwen35_paged_engine` exited 0. It loaded snapshot
  `2fc06364715b967f1860aea9cf38778875588b17`, ran on device type 5, passed
  16 of 16 prompts and 137 of 137 assertions, and reported zero declines.
- Mutation: the repaired greedy branch changed to a normal return. The source
  hash changed from `7d7dfede853fe46746cbefec087130b17dc7e65b93376f28ff932ee32739bc8e`
  to `27cbf5cba2bce704c2df737da9397f6d42afa92232f1c3bad0715ac745508116`.
  A fresh rebuild exited 0 and changed the binary hash from
  `a5fcd67f7d926efd90ce50732a6eb667fab045ccc0036248a9647f89cc1f3df4`
  to `b758bcb7ab8af04d22c5f99dcefdc5c0d5db802f28d1543f79f755a8f33bed6d`.
  CTest exited 8 because the greedy child exited 0. The other two children
  still exited 77. Restoration recovered both original hashes, the same
  tracked status, and CTest exit 0 after a fresh rebuild.
- Full gate: `PATH=/usr/bin:$PATH GIT_CONFIG_GLOBAL=/dev/null
  scripts/agent-preflight.sh` exited 0 and printed literal `All gates green.`

#### Reviewer repair loop for the complete probe

- Complete-control RED: after the CMake regression created all three
  placeholder artifacts, `flock /home/vikash/gpu.lock env
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib ctest --test-dir
  build-hip --output-on-failure -R
  '^test_qwen35_paged_engine_prerequisites$'` exited 8. The three
  missing-artifact children stayed at exit 77. The complete child exited 0,
  and doctest reported one passed case with zero assertions.
- Focused GREEN: a fresh `cmake --build build-hip --target
  test_qwen35_paged_engine --parallel 4` under the same mutex and library path
  exited 0. The same CTest command exited 0. Direct CMake-script execution
  exited 0 and reported child exits 77, 77, 77, and 86.
- Real model gate: `flock /home/vikash/gpu.lock env
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib
  ./build-hip/tests/test_qwen35_paged_engine` exited 0. The gate loaded snapshot
  `2fc06364715b967f1860aea9cf38778875588b17`, ran on ROCm device type 5,
  passed 16 of 16 prompts and 137 of 137 assertions, and reported zero
  declines.
- Mutation: the complete-probe branch changed back to a normal return while
  retaining a reference to the sentinel so the mutation compiled. The source
  hash changed from
  `7a2c3782b1546a8a90ba1599610e13068addeb631f7286e66dac7ce4ec1215f7`
  to `b3de308eb7805d575248258eda60e2acbca8e2db9b8e25b8a16cfbacd88bb465`.
  The fresh rebuild exited 0 and changed the binary hash from
  `acf873bebec14804e9763bdee8c9c2e51dddd0b5949fb8cbff7091b8935bb931`
  to `febc7bcc6e4cf81668651aff0c21bf3dc7be75c8593c7235dfc1f7da68e56947`.
  CTest exited 8 because the complete child returned 0. The three
  missing-artifact children still returned 77. Restoration recovered both
  original hashes, and the fresh rebuild and CTest each exited 0. A prior
  one-line mutation was rejected as evidence because `-Werror` stopped the
  build on the unused sentinel before the regression ran.
- Full gate: `env PATH=/usr/bin:$PATH GIT_CONFIG_GLOBAL=/dev/null
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib
  scripts/agent-preflight.sh` exited 0 and printed literal
  `All gates green.`

### Stop conditions

Stop rather than weaken the gate if the deterministic test cannot exercise the
actual executable, if absent artifacts cannot be distinguished from malformed
present arrays, if the pinned local snapshot or ROCm gate cannot run under the
required mutex, or if any change outside this test, its registration, this
specification, and the issue-index append is required.

### Outcome

Issue #1222 was the two normal returns named by the finding. The greedy branch
returned directly, and one shared branch returned when either the anchor or gap
was absent. Both paths produced a zero-assertion doctest success.

The gate now uses one existence-only helper. The helper exits 77 for each
missing required artifact and returns only when all three paths exist. Array
loading and dtype and shape checks remain after the helper, so a malformed
present array still takes the hard-failure path. `VT_DUMP_IDS` remains the one
bootstrap opt-in. Its model run and `our_ids.i32` write remain intact, but the
bootstrap path now exits 77 after the dump because it did not run correctness.

The test uses an environment probe in the real executable instead of a second
artifact predicate. A separate validator was rejected because it could drift
from the gate. Exit 77 remains the existing CTest unavailable-gate contract.
No numerical threshold, snapshot pin, model setting, or device default changed.

The reviewer found that the probe's complete-artifact path still returned
normally. That return bypassed snapshot resolution, model construction, and the
device check, so a test-only path could report production success. The
subprocess regression now includes one complete-artifact control. That control
requires the named test-only sentinel exit 86. Exit 86 is distinct from process
success 0 and the unavailable-gate skip 77, and it fits the portable process
exit range.

After the shared prerequisite helper returns in probe mode, the executable now
prints that the test probe completed and that the production gate did not run.
It then exits 86. The missing greedy, anchor, and gap probes still exit 77.
Normal execution without the probe is unchanged and continues through snapshot
resolution, array validation, model construction, and the ROCm device gate.
