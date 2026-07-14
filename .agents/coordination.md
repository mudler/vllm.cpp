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
6. Merge/push only after the matrix, roadmap, porting inventory, ledger, state,
   `README.md`, and `docs/BENCHMARKS.md` all agree. README and BENCHMARKS are
   mandatory at every feature/iteration checkpoint, including pending, failed,
   and void stages rather than only externally visible closure.

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
| `CLAIM-PR3` | `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH` | root takeover of stopped `validate_pr3` / `complete_pr3` stream | primary recovery tree `/home/mudler/_git/vllm.cpp-pr3-validate`; 27B default/component integration in `/home/mudler/_git/vllm.cpp-nvfp4-small-m`; DGX evidence `~/work/vllm.cpp-noPy` plus `~/work/vllm.cpp-nvfp4-small-m/debug/gdn-out-bf16-c16-ab-20260712` | recovery branch integrated into `main` by `a767188`; current checkpoint on `codex/nvfp4-small-m` | PR #3 files and rows `KERNEL-GDN-AOT-BF16`+`KERNEL-GDN-SCRATCH`; 27B-only `GdnOutDType` default/f32 override in `qwen3_5.cpp`; ledger/inventory/roadmap evidence. `KERNEL-GDN-PACKED-DECODE` is explicitly led by `CLAIM-GDN-BA-ROUNDING-1`; this claim retains prefill AOT and scratch/teardown only. No 35B default change | `ACTIVE` | 2026-07-14 (vendored BF16 H32/H48 AOT/safety evidence and native 16/16 correctness remain green. Binding `3f256ab` has c16 total at 1.027889× but mean TPOT/ITL at 0.987450×. Pure-decode packed execution is split into its own row/claim; all 35B paths keep f32) |
| `CLAIM-SERVE-GATE-1` | `SERVE-GATE-ONLINE`, `KERNEL-GEMM-NVFP4-W4A4`, `KERNEL-ATTN-FA2` | root takeover | binding grid `~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560`; active worktree `/home/mudler/_git/vllm.cpp-nvfp4-small-m`; finalized c2 root `~/work/vllm.cpp-executed-path-c2/179a0fc2afc1c33b63d14de8e50d3fde976c7356`; packed trace root `~/work/vllm.cpp-gdn-packed-trace/7ff713e377457130db4ed15929133d1b463aff96` | current checkpoint `codex/nvfp4-small-m` | Online-gate execution and harness work. `CLAIM-GDN-BA-ROUNDING-1` leads packed-decode code and its c2/c16 component; this claim retains the binding grid. qkvz code, loader-memory repair, exact grid, and 35B performance are excluded. Any A/B or trace series uses one `flock` | `ACTIVE` | 2026-07-14 (packed structure is accepted from `7ff713e` + `24cea4f`; the hardened production c2/c16 component runner is CPU-green **44/44** and DGX-pending. Binding stays 55/124 with no speed credit) |
| `CLAIM-GDN-BA-ROUNDING-1` | `KERNEL-GEMM-BF16` (`W1C`), `KERNEL-GDN-PACKED-DECODE` (`W1D`) | root | isolated worktree `/home/mudler/_git/vllm.cpp-gdn-ba-rounding`; immutable G2 root `~/work/vllm.cpp-gdn-packed-decode/f344decf457a4d50c3bcae78a2903d7fe176a511/evidence-g2`; accepted trace root `~/work/vllm.cpp-gdn-packed-trace/7ff713e377457130db4ed15929133d1b463aff96`; planned component root `~/work/vllm.cpp-gdn-packed-component/<pushed-sha>` | `codex/gdn-ba-rounding-w1c` | Leads exact packed decode, model dispatch/rollback, trace and c2/c16 component. Prefill/mixed/spec GDN, qkvz, host-weight repair, binding-grid execution and 35B performance are excluded. One `flock /tmp/gpu` covers every GPU correctness or A/B series | `ACTIVE` | 2026-07-14 (production-build-only c2/c16 AB/BA/AB driver and marker-last all-axis finalizer are hardened; focused CPU tests pass **44/44** and all tools **126/126**. Next session executes its one-lock **40+8** series from the pushed checkpoint. Speed credit remains absent) |
| `CLAIM-NVFP4-SMALL-M-2` | `KERNEL-GEMM-NVFP4-W4A4` (`W2`) | root | isolated worktree `/home/mudler/_git/vllm.cpp-nvfp4-small-m`; immutable source/evidence `~/work/vllm.cpp-nvfp4-small-m/b5c6e4fd65cdacea8f378e18ae101ebf521e8f01/w2` plus exact online gate | `codex/nvfp4-small-m` | W2 only: 32 tactics, merged CT gate/up and one-input `SiluAndMulFp4Quant`, with independent fallbacks and full correctness/safety/component/oracle gates | `RELEASED` | 2026-07-12 (implementation/correctness complete; strict old-oracle acceptance failed. The vLLM 0.24.0 ratios are historical diagnostics; W3 remains the trace-driven repair track under the new v0.25.0 denominator) |
| `CLAIM-NVFP4-SMALL-M-3` | `KERNEL-GEMM-NVFP4-W4A4` (`W3-C`) | root | isolated worktree `/home/mudler/_git/vllm.cpp-nvfp4-small-m`; oracle cache fixture under `tests/fixtures/nvfp4_flashinfer_v025_gb10/`; immutable C3/C3R/corrected component `~/work/vllm.cpp-nvfp4-persistent/d211b8f80fff831a712f0bfafa4f65f1abe1892d/evidence` | `codex/nvfp4-small-m` | W3-C document/import/atomicity, ready-map/lifecycle/5,000-us wiring and corrected same-plan gates only; other levers excluded | `RELEASED` | 2026-07-13 (W3-C reproduction control complete: six-process and corrected 12-leg components use identical 64/64 maps with zero tuning/misses. W3-E strict-fails 39/40 timing + 1/8 memory; no exact grid/35B performance) |
| `CLAIM-NVFP4-SMALL-M-4` | `KERNEL-GEMM-NVFP4-W4A4` (`W3-F`) | root | isolated worktree `/home/mudler/_git/vllm.cpp-nvfp4-small-m`; immutable source/evidence `~/work/vllm.cpp-nvfp4-device-alpha/7517af4f983fe322ac88ce2d9869e1441b7be3fd/` | `codex/nvfp4-small-m` | W3-F only: model-owned device F32 alpha, tensor op ABI/validation, `VT_FP4_DEVICE_ALPHA=0` fallback, ported tests, safety/model/node-trace and frozen-plan c2/c16 gates per [spike](specs/nvfp4-device-alpha.md). Quant/GDN/attention/host-weight changes excluded | `RELEASED` | 2026-07-13 (local/CUDA/operator/memcheck/model/trace gates pass. Completed 12-leg/612-request component is c2/c16 1.001967×/1.000144× but strict-fails 27/40 timing + 3/8 memory. No speed credit/exact grid/35B performance; the completed scan moves W3-G FA2 decode under CLAIM-SERVE-GATE-1) |
| `CLAIM-KV-DEVICE-1` | `KV-DEVICE-RESIDENCY` | root | main worktree `/home/mudler/_git/vllm.cpp`; W0/W1 evidence under `~/work/vllm.cpp-kv-device`; current oracle evidence `~/work/vllm.cpp-online-gate/evidence/b5c6e4fd65cdacea8f378e18ae101ebf521e8f01` | `main` | W0/W1 ownership/indexed-state implementation and open teardown; W2 direct convolution update remains scoped but unclaimed until post-FP4 evidence re-ranks it | `ACTIVE` | 2026-07-12 (W0/W1 component A/Bs remain 1.021239×/1.006246× and access/correctness green; inherited pools fail zero-leak. Clean `b5c6e4f` confirms FP4 selection W3 first; no device-residency W2 implementation is inferred yet) |
| `CLAIM-BACKEND-ABI-W0-GPU-1` | `BACKEND-ABI-VT` (`W0-GPU`) | root, surfaced by fresh serving-gate CUDA build | local `/home/mudler/_git/vllm.cpp-backend-abi-gpu-fix`; DGX `~/work/vllm.cpp-online-{src,build}` | `codex/backend-abi-w0-gpu` | exact GCC13/CUDA build blocker in `tests/vt/test_dropin_abi.cpp`; CUDA cross-build/runtime/capture evidence for additive W0; owning matrix/roadmap/ledger/state. No ABI or production-kernel redesign | `ACTIVE` | 2026-07-11 (branch `1141b79`: exact clean CUDA 13.0.88/sm_121a all-target rebuild reached 100%; lock-held focused CUDA/ABI CTest 2/2, compute-sanitizer 9/9 cases + 196/196 assertions with 0 errors/leaks, and 35B/27B model gates 2/2 pass. Evidence `~/work/vllm.cpp-online-build/w0-gpu-1141b79`, manifest SHA-256 `4adbe952…601`. Remaining W0: sm_80/sm_90a cross-builds and unchanged-trace/model A/B-memory proof; no runtime source changed) |

`CLAIM-SERVE-GATE-1` owns the binding grid and finalized exact-c2 evidence.
Root `179a0fc`, status `9e0143fa…7b57`, proves the selected **193 vs 97** BF16
projection topology; detailed precursor chronology remains only in the
append-only record. `KERNEL-GEMM-BF16` BA-only W1 is implemented/`GATING` and
its implementation claim is released; the serving claim owns immutable
evidence execution only. The exact structural marker is complete at `0091cd1`;
Projection and 35B/GGUF inertness close at `f925294`; the first downstream
boundary is now identified and owned by `KERNEL-GDN-PACKED-DECODE`. Clean
`f18ca23` closes immutable G0 and clean `9ad8fb7` closes W1D1/G1. Production
model semantics and immutable G2 close at clean `f344dec`. Clean `7ff713e`,
finalized by `24cea4f`, closes W1D3 structural evidence. The production c2/c16
runner/finalizer is CPU-green **44/44**; its one-lock DGX series is the only
remaining packed component checkpoint. qkvz, exact-grid and 35B performance
stay excluded until that gate closes.

## Handoff queue

| Priority | Row/block | Dependency | Next handoff | State |
|---|---|---|---|---|
| 1 | `KERNEL-GEMM-BF16` + `KERNEL-GDN-PACKED-DECODE` for `SERVE-GATE-ONLINE` | `3f256ab` remains 55/124; `f344dec` closes correctness and `7ff713e` + `24cea4f` close structure; the hardened production component runner is CPU-green **44/44** | execute/finalize the pushed checkpoint's one-lock c2/c16 **40+8** series before qkvz | `GATING` |
| 2 | `SERVE-ASYNC-LLM` block (`ENG-CORE-BUSY-LOOP`, `SERVE-ASYNC-LLM`, `ENG-ASYNC-SCHED`, `ENG-PRIORITY-SCHED`) | joint spike accepted ([async-serving.md](specs/async-serving.md)); W1/W2/W4 implemented/`GATING`, W3 `READY`; complete control shows W3 is neutral for order-0 speed | retain W3 as unclaimed parity work until order 0 closes; do not fold it into the active kernel claim | `GATING` |
| 3 | `SERVE-E2E-NIGHTLY` | `SERVE-GATE-ONLINE` evidence where benchmarks overlap | write spike and CI/nightly split | `INVENTORIED` |
| 4 | C1 kernel drop-in alignment | accepted kernel-family inventory + [drop-in ABI spike](specs/dropin-kernel-abi.md) | `BACKEND-ABI-VT` W0 is implemented and CPU-green; run cross-build + flock-held GB10 runtime/capture/model/trace handoff, close scalar-forwarder/backend-shim debts, then migrate one family per independently gated checkpoint | `GATING` |
| 5 | `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` | accepted spike; no dependency on async serving | P1 deterministic corpus/client/summary/memory/dry-run harness is implemented and `GATING`; provision the digest-pinned image only in a non-benchmark window, resolve the pinned client's missing raw E2E/TPOT sample gap without changing timed semantics, then P2 classifies each exact checkpoint | `GATING` |
| 6 | `BACKEND-GATE-CUDA-SGLANG` | `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT`, then `SERVE-ASYNC-LLM` | run the binding c1-c16 campaign only after both dependencies close | `BLOCKED` |
| 7 | `BACKEND-GATE-CUDA-SGLANG-PREFIX` | accepted shared-prefix extension; PX1 can proceed, while binding execution depends on exact SGLang v0.5.15 equivalence, a dedicated `KV-MAMBA-ALIGN` spike/implementation and `SERVE-ASYNC-LLM` | after the priority 27B cache-off parity grid/trace, implement the deterministic 64k/256k corpus and counters, write then execute the Mamba-align retention leaf, prove cache hits/no eviction, and bind every axis to the faster equivalent vLLM/SGLang result before 35B | `READY` |
| 8 | C2/C4 models + quantizations | accepted comprehensive inventories | `MODEL-FACTORY-registry` and `QUANT-GGUF-CPU-THREADPOOL` are implemented and `GATING`: the registry awaits its exact two-model GPU greedy/performance/memory handoff after `CLAIM-SERVE-GATE-1`; the threadpool has 1/3/20 determinism + TSAN evidence but still needs the exclusive-idle-host B4 ≥10x/RSS series. Both claims are released. Keep-quant loader + CIQ GEMM follow that CPU checkpoint, then Llama dense | `GATING` |
| 9 | C3/C5/C7/C8 engine/API work | C3/C5/C6 specs accepted; C7/C8 leaf specs missing | C3 M-mtp-0 remains `GATING` on its GPU recipe. All C5 W1-W8 implementation leaves are now `GATING`; restored feature-positive model rows, CUDA/runtime/oracle/trace and every-axis campaigns remain. An idle GPU still gates runtime closure; then spike C7/C8 | `GATING` |
| 10 | D1 backend expansion | accepted backend/CUDA inventory + vt seam | spike architecture spine, then NVIDIA fan-out, ROCm, MLX, Vulkan, XPU | `SPIKE` |
| 11 | `ENG-EXPERT-STREAM` (D4 first leaf) | corrected spike accepted after live Marlin/loader verification | claim W0 nsys + fresh c1 baseline; then W1 CPU cache policy and W2 bank-only loader/reader/pread as independent performance checkpoints before W3 dispatch | `READY` |

## Closing and archival

A block closes only when every row in its declared scope is `DONE`. In the same
change:

1. freeze the block plan, outcomes, commands, and evidence under
   `completed/<version>-<block>.md`;
2. replace live execution prose with one portfolio link to that archive;
3. release all claims and remove transient branch/worktree notes;
4. retain completed support rows in their permanent area matrix with code/test
   anchors and closing commit;
5. update README and `docs/BENCHMARKS.md` unconditionally for the closing
   checkpoint, plus porting inventory, ledger, and state when relevant.

This keeps `.agents/` focused on current work without losing the discoverable
list of what the project supports.
