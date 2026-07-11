# Parallel roadmap coordination

This is the operational control plane for parallel work. The top-level
[roadmap](roadmap_v1.md) defines ordering, and each linked area matrix defines
the exhaustive row inventory. This document controls who is doing what now.

## Canonical hierarchy

| Level | Purpose | Canonical location |
|---|---|---|
| Portfolio | Ordered big areas and release blocks | [roadmap_v1.md](roadmap_v1.md) |
| Area inventory | Exhaustive, stable row IDs and support status | `*-matrix.md` files linked by the roadmap |
| Spike/spec | Upstream/dependency analysis, tests, gates, decomposition | [specs/](specs/) |
| Active claim | Agent, worktree, branch, files, dependencies, hardware | This document |
| Evidence | Code/test anchors, oracle and benchmark results | Area row + [parity-ledger.md](parity-ledger.md) |
| Completed block | Closed execution plan/report | [completed/](completed/) |

The hierarchy is one-way. A portfolio row links area rows; area rows link their
spikes and evidence. Do not copy detailed status prose upward.

## Row contract

Every claimable area-table row must expose these semantic fields (additional
area-specific columns and different ordering are allowed). Generated model and
quantization tables may combine `Our code` with `Tests/evidence` in one exact
cell; every other matrix keeps them separate. A table-level upstream pin may
qualify dense registry rows only when each row also names its upstream
method/type/target. Summary/roll-up tables must say `non-claimable` and contain
no stable work IDs.

| Required field | Rule |
|---|---|
| `ID` | Stable and never reused; use area prefixes (`ENG`, `MODEL`, `QUANT`, `KERNEL`, `BACKEND`) |
| `Item` | One independently gateable capability or explicit row block |
| `Upstream` | Pinned vLLM `file:line`; include dependency source when it owns execution |
| `Our code` | Exact implementation `file:line`, or `-` when absent; path and line must resolve inside a source/build/tool path class |
| `Tests/evidence` | Exact tests/goldens/ledger evidence, or `-` when absent; path and line must resolve inside a test/CI/ledger class, and a combined code/tests cell must contain both classes |
| `Spike/spec` | Real `.agents/specs/<slug>.md` once state is `READY` or later |
| `State` | One value from [workflow.md](workflow.md#tabular-lifecycle) |
| `Owner` | Active claim ID, or `-`; completed rows name the closing commit |

`DONE` with `-`, vague prose, a directory-only pointer, or a dangling spec link
is invalid. Its evidence cell links an exact parity-ledger line, and its owner
cell is the hexadecimal closing commit; the checker verifies that commit exists
in Git history. Split a row when supported and unsupported modes would otherwise be
hidden behind one status. Quantization rows, for example, distinguish format
recognition, dequantization, compute-in-quant, backend kernels, and e2e gates.

## Spike gate

Every item is spiked, including ports that look mechanical. A spike is accepted
only when its spec has these sections:

| Section | Required content |
|---|---|
| Scope | In/out, exact row IDs, supported modes and dispatch behavior |
| Upstream chain | Pinned vLLM and dependency `file:line`; runtime trace plan where dispatch is dynamic |
| Our baseline | Existing code/test anchors and honest gaps |
| Port map | Upstream file -> local file, including thin adapters and deviations |
| Tests to port | Upstream test modules/cases, local tier, skip reason if initially blocked |
| Gates | Correctness, e2e, performance, memory, architectures/backends, exact commands |
| Dependencies | Other row IDs, toolchain, models, hardware, data, licenses |
| Work breakdown | Small non-overlapping implementation rows suitable for parallel claims |
| Risks/decisions | Product calls only; vLLM-defined behavior is not reopened |

Spike agents are read-only outside their spec and matrix/coordination rows. A
spike can split or reorder work by recording the dependency, but cannot mark an
implementation supported.

## Claim protocol

1. Pick the highest-priority `READY` row whose dependencies are `DONE`.
2. Add one active-claim row below before editing. List every file or narrow
   directory owned; overlapping ownership requires an explicit lead claim.
3. Create an isolated worktree and branch named for the stable row ID. Never
   share a build directory between claims.
4. Update the claim after spike, implementation, and gating handoffs. A stalled
   agent releases its claim; it does not keep an invisible reservation.
5. Return commit SHA, exact commands/results, remaining gaps, and every status
   file that must change. The integrating agent checks the row contract.
6. Merge/push only after the matrix, roadmap, README (when externally visible),
   porting inventory, ledger, and state all agree.

## GPU scheduling

All agents use the shared protocol at
`/home/mudler/_git/skills/sharing-a-gpu-with-flock/SKILL.md`.

**When the lock applies (user-directed 2026-07-10):** there is no external GPU
contention on dgx (the LocalAI worker stays stopped during work) — the flock is
purely the INTER-AGENT mutex. It is REQUIRED whenever more than one agent or
queued job may execute GPU work in the same window (any moment coordination.md
lists 2+ GPU-capable claims, or a queue exists). A sole GPU owner may run
lock-free after verifying the GPU is idle (`nvidia-smi` shows no compute
process); benchmark/A-B validity still requires an uncontended GPU either way —
that invariant is about measurement, not locking.

| Work | Lock rule |
|---|---|
| Compile, sync files, inspect source, `ps`/`nvidia-smi` | No GPU lock needed |
| CUDA tests, model load, server, benchmark, nsys/ncu | `flock /tmp/gpu -c '<whole job>'` |
| A/B or competitor series | One `flock` around every arm and trace; interleaved runs are void |
| Long background run | `setsid flock /tmp/gpu -c '<job>'`; lock parent lives with job |
| Timeout | `flock -w <seconds>`; inspect holder, never kill an unowned PID |

Each claim records its remote directory and planned lock window. One server at a
time owns the GB10. Results without the lock for their entire run are discarded.

## Active claims

| Claim | Row IDs | Agent | Worktree / remote dir | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-PR3` | `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH` | root takeover of stopped `validate_pr3` / `complete_pr3` stream | local `/home/mudler/_git/vllm.cpp-pr3-validate`; DGX `~/work/vllm.cpp-noPy` (own build dir) | recovery branch integrated into `main` by `a767188` | PR #3 files; claim/matrix rows `KERNEL-GDN-AOT-BF16`+`KERNEL-GDN-SCRATCH`; ledger/inventory/roadmap rows for PR #3 closure. GPU lock: preserve queued preliminary jobs, then final post-integration build/tests/sanitizer, pool+BF16 A/B, paired traces and fresh-vLLM comparison under new whole-series locks | `ACTIVE` | 2026-07-10 (non-GPU gaps closed: 48-artifact bundle includes BF16 H32/H48, target/line-info/generator/base/artifact drift is fatal, generated-module loads are race-guarded, and tests assert BF16 dispatch plus 11-buffer `0xff` reuse; integrated CPU/drift gates pass, no GPU used, old DGX jobs remain preliminary, and final GPU gates are pending) |
| `CLAIM-SERVE-GATE-1` | `SERVE-GATE-ONLINE` | root takeover of stopped `serve-gate-online` stream | local `/home/mudler/_git/vllm.cpp-serve-gate`; closed diagnostic DGX dir `~/work/vllm.cpp-latency`; binding evidence root `~/work/vllm.cpp-online-gate` | `codex/serve-gate-harness` (merged `b0acec2`; execution hardening active) | committed `tools/bench/online_gate*`, `tools/bench/profile_vllm_online_gate.py`, `tools/bench/summarize_torch_kernels.py`, `scripts/dgx-online-serving.sh`, `tests/tools/test_online_gate_*`; rootless verified cache-eviction helper `tools/bench/drop_file_cache.py`; explicit server scheduling-capacity CLI in `examples/server/main.cpp` + help test/README; DGX current-main build/harness/results; engine matrix, roadmap, ledger/state. ONE `flock` around each complete model A/B+trace series | `ACTIVE` | 2026-07-11 (`4b546a8`: pandas/oracle, 27B model/cache/live-SSE and six c1 requests pass; validator voided the arm because vLLM's inclusive one-second bucket reports peak=2 although detailed half-open intervals have ≥0.00010s gaps and exact peak=1. Fix now derives binding peak from start+TTFT+ITL intervals, retains upstream bucketed peak as diagnostic, and accepts the preserved raw result. No ratio retained. Merge/regenerate, then 27B followed by 35B) |
| `CLAIM-BACKEND-ABI-W0-GPU-1` | `BACKEND-ABI-VT` (`W0-GPU`) | root, surfaced by fresh serving-gate CUDA build | local `/home/mudler/_git/vllm.cpp-backend-abi-gpu-fix`; DGX `~/work/vllm.cpp-online-{src,build}` | `codex/backend-abi-w0-gpu` | exact GCC13/CUDA build blocker in `tests/vt/test_dropin_abi.cpp`; CUDA cross-build/runtime/capture evidence for additive W0; owning matrix/roadmap/ledger/state. No ABI or production-kernel redesign | `ACTIVE` | 2026-07-11 (branch `1141b79`: exact clean CUDA 13.0.88/sm_121a all-target rebuild reached 100%; lock-held focused CUDA/ABI CTest 2/2, compute-sanitizer 9/9 cases + 196/196 assertions with 0 errors/leaks, and 35B/27B model gates 2/2 pass. Evidence `~/work/vllm.cpp-online-build/w0-gpu-1141b79`, manifest SHA-256 `4adbe952…601`. Remaining W0: sm_80/sm_90a cross-builds and unchanged-trace/model A/B-memory proof; no runtime source changed) |

## Handoff queue

| Priority | Row/block | Dependency | Next handoff | State |
|---|---|---|---|---|
| 1 | `SERVE-GATE-ONLINE` | Both pre-W2 campaigns are closed/invalid; second 35B ended with ours abort + vLLM hang | root recovered `CLAIM-SERVE-GATE-1`; harness `b0acec2` is merged and CPU-complete. GPU lock is free: build current main in fresh dirs, reproduce 35B under sanitizer if necessary, then run W2 G1/G3-G6 under new whole-model locks | `ACTIVE` |
| 2 | `SERVE-ASYNC-LLM` block (`ENG-CORE-BUSY-LOOP`, `SERVE-ASYNC-LLM`, `ENG-ASYNC-SCHED`, `ENG-PRIORITY-SCHED`) | joint spike accepted ([async-serving.md](specs/async-serving.md)); W2 removes the fake-SSE structural blocker, but `SERVE-GATE-ONLINE` cannot close until its real GB10 gates pass | W1 + W2 + W4 are implemented and `GATING` with claims released. W2 CPU full suite 94/94, focused TSan 5/5, live completion/chat SSE, disconnect abort, concurrent HTTP requests and 17-symbol C ABI are green. When `CLAIM-SERVE-GATE-1` releases dgx, run W1 G1/G4, W2 G1/G3/G4/G6 and W4 G1 under the first whole-series lock; the current campaign is a pre-W2 baseline only. W3 overlap remains READY | `GATING` |
| 3 | `SERVE-E2E-NIGHTLY` | `SERVE-GATE-ONLINE` evidence where benchmarks overlap | write spike and CI/nightly split | `INVENTORIED` |
| 4 | C1 kernel drop-in alignment | accepted kernel-family inventory + [drop-in ABI spike](specs/dropin-kernel-abi.md) | `BACKEND-ABI-VT` W0 is implemented and CPU-green; run cross-build + flock-held GB10 runtime/capture/model/trace handoff, close scalar-forwarder/backend-shim debts, then migrate one family per independently gated checkpoint | `GATING` |
| 5 | `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` | accepted spike; no dependency on async serving | P1 deterministic corpus/client/summary/memory/dry-run harness is implemented and `GATING`; provision the digest-pinned image only in a non-benchmark window, resolve the pinned client's missing raw E2E/TPOT sample gap without changing timed semantics, then P2 classifies each exact checkpoint | `GATING` |
| 6 | `BACKEND-GATE-CUDA-SGLANG` | `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT`, then `SERVE-ASYNC-LLM` | run the binding c1-c16 campaign only after both dependencies close | `BLOCKED` |
| 7 | C2/C4 models + quantizations | accepted comprehensive inventories | `MODEL-FACTORY-registry` and `QUANT-GGUF-CPU-THREADPOOL` are implemented and `GATING`: the registry awaits its exact two-model GPU greedy/performance/memory handoff after `CLAIM-SERVE-GATE-1`; the threadpool has 1/3/20 determinism + TSAN evidence but still needs the exclusive-idle-host B4 ≥10x/RSS series. Both claims are released. Keep-quant loader + CIQ GEMM follow that CPU checkpoint, then Llama dense | `GATING` |
| 8 | C3/C5/C7/C8 engine/API work | C3/C5/C6 specs accepted; C7/C8 leaf specs missing | C3 M-mtp-0 remains `GATING` on its GPU recipe. All C5 W1-W8 implementation leaves are now `GATING`; restored feature-positive model rows, CUDA/runtime/oracle/trace and every-axis campaigns remain. An idle GPU still gates runtime closure; then spike C7/C8 | `GATING` |
| 9 | D1 backend expansion | accepted backend/CUDA inventory + vt seam | spike architecture spine, then NVIDIA fan-out, ROCm, MLX, Vulkan, XPU | `SPIKE` |
| 10 | `ENG-EXPERT-STREAM` (D4 first leaf) | corrected spike accepted after live Marlin/loader verification | claim W0 nsys + fresh c1 baseline; then W1 CPU cache policy and W2 bank-only loader/reader/pread as independent performance checkpoints before W3 dispatch | `READY` |

## Closing and archival

A block closes only when every row in its declared scope is `DONE`. In the same
change:

1. freeze the block plan, outcomes, commands, and evidence under
   `completed/<version>-<block>.md`;
2. replace live execution prose with one portfolio link to that archive;
3. release all claims and remove transient branch/worktree notes;
4. retain completed support rows in their permanent area matrix with code/test
   anchors and closing commit;
5. update README, porting inventory, ledger, and state when relevant.

This keeps `.agents/` focused on current work without losing the discoverable
list of what the project supports.
