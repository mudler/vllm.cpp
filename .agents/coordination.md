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
| `CLAIM-PR3` | `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH` | root takeover of stopped `validate_pr3` / `complete_pr3` stream | primary recovery tree `/home/mudler/_git/vllm.cpp-pr3-validate`; 27B default/component integration in `/home/mudler/_git/vllm.cpp-nvfp4-small-m`; DGX evidence `~/work/vllm.cpp-noPy` plus `~/work/vllm.cpp-nvfp4-small-m/debug/gdn-out-bf16-c16-ab-20260712` | recovery branch integrated into `main` by `a767188`; current checkpoint on `codex/nvfp4-small-m` | PR #3 files and rows `KERNEL-GDN-AOT-BF16`+`KERNEL-GDN-SCRATCH`; 27B-only `GdnOutDType` default/f32 override in `qwen3_5.cpp`; ledger/inventory/roadmap evidence. No 35B default change. GPU lock: residual trace/pool classification remains separate from FP4 W3 | `ACTIVE` | 2026-07-12 (vendored BF16 H32/H48 AOT/safety evidence remains green. 27B BF16 restores native 16/16 tokens; component is 1.007989×, 16/20 timing and 2/4 memory. Clean `b5c6e4f` c16 total throughput passes at 1.021341× but mean TPOT/ITL remains 0.982670× normalized. Re-rank after FP4 W3; all 35B paths keep f32) |
| `CLAIM-SERVE-GATE-1` | `SERVE-GATE-ONLINE` | root takeover of stopped `serve-gate-online` stream | local `/home/mudler/_git/vllm.cpp-serve-gate`; binding evidence root `~/work/vllm.cpp-online-gate` | `codex/serve-gate-harness` (merged `b0acec2`; corrected trace/gate contract active) | `tools/bench/online_gate*`, profiler/summarizer/driver contracts, exact DGX evidence, matrices/roadmap/ledger/state. ONE `flock` around each complete model A/B+trace series | `ACTIVE` | 2026-07-12 (clean pushed-`b5c6e4f` 27B is binding: 12/12 groups, 2,016/2,016 requests, six returns, model/trace pass; c1→c32 total ratios 0.9933/0.9520/0.9657/0.9760/1.0213/1.0218×, 4/4/5/4/17/14 timing and 2/4 memory axes. W3 selection parity is next; 35B held) |
| `CLAIM-NVFP4-SMALL-M-2` | `KERNEL-GEMM-NVFP4-W4A4` (`W2`) | root | isolated worktree `/home/mudler/_git/vllm.cpp-nvfp4-small-m`; immutable source/evidence `~/work/vllm.cpp-nvfp4-small-m/b5c6e4fd65cdacea8f378e18ae101ebf521e8f01/w2` plus exact online gate | `codex/nvfp4-small-m` | W2 only: 32 tactics, merged CT gate/up and one-input `SiluAndMulFp4Quant`, with independent fallbacks and full correctness/safety/component/oracle gates | `RELEASED` | 2026-07-12 (implementation/correctness complete; strict acceptance failed. Clean binding total ratios 0.9933/0.9520/0.9657/0.9760/1.0213/1.0218×, axes 4/4/5/4/17/14, memory 2/4; trace promotes W3) |
| `CLAIM-NVFP4-SMALL-M-3` | `KERNEL-GEMM-NVFP4-W4A4` (`W3`) | root | isolated worktree `/home/mudler/_git/vllm.cpp-nvfp4-small-m`; immutable W3-A evidence `~/work/vllm.cpp-nvfp4-small-m/71f1e894d0c5e496607d08cfe9089a9944128271/w3`; immutable W3-B evidence `~/work/vllm.cpp-nvfp4-small-m/d7cdf66db0cfcc53d68d49613623ec6cd3807641/w3b`; exact online-gate root | `codex/nvfp4-small-m` | W3 only: A mirrors production pip-vLLM/FlashInfer's three-warmup, stream-sync, 1-ms GPU-delay, ten-repeat eager event selection; B adds actual-W4A4-gated shared-loader max-token/all-bucket in-memory warmup, completion stats, capture-ready reuse, diagnostic lazy fallback and `VT_FP4_PRE_SERVE_WARMUP=0`; C adds separately gated optional versioned device/CUDA/CUTLASS/model-shape persistence with atomic load/save/stale/collision rejection. No new tactic, model topology, HTTP, scheduler, GDN or device-residency changes. Same-binary selection A/B, paired trace, then exact 27B under one uncontended lock | `ACTIVE` | 2026-07-12 (W3-A strict-failed. Clean `d7cdf66` W3-B passes registry/loader, exact/legacy CUDA 14/14 + 26,819 each, 27B 235/235 + 16/16, memcheck 24,586/24,586 with zero errors, and server ordering with 80/80 profiles/zero lazy misses. Manifest `6f372fbe…89b1`. Repeated prewarm/lazy c16 means are 808.457/808.220 tok/s = 1.000293×, strict 15/20 timing + 2/4 memory; only 20/80 prewarm IDs are stable, while first-use improves 5.662→0.779 s. No speed credit. Trace attempt 1 is VOID after prewarm-only 144/144: cache inventory changed 50→58 before lazy/vLLM; corrected corpus-only-root rerun next. Pip-vLLM disables its colliding file cache, so persistence cannot own the speed denominator) |
| `CLAIM-KV-DEVICE-1` | `KV-DEVICE-RESIDENCY` | root | main worktree `/home/mudler/_git/vllm.cpp`; W0/W1 evidence under `~/work/vllm.cpp-kv-device`; current oracle evidence `~/work/vllm.cpp-online-gate/evidence/b5c6e4fd65cdacea8f378e18ae101ebf521e8f01` | `main` | W0/W1 ownership/indexed-state implementation and open teardown; W2 direct convolution update remains scoped but unclaimed until post-FP4 evidence re-ranks it | `ACTIVE` | 2026-07-12 (W0/W1 component A/Bs remain 1.021239×/1.006246× and access/correctness green; inherited pools fail zero-leak. Clean `b5c6e4f` confirms FP4 selection W3 first; no device-residency W2 implementation is inferred yet) |
| `CLAIM-BACKEND-ABI-W0-GPU-1` | `BACKEND-ABI-VT` (`W0-GPU`) | root, surfaced by fresh serving-gate CUDA build | local `/home/mudler/_git/vllm.cpp-backend-abi-gpu-fix`; DGX `~/work/vllm.cpp-online-{src,build}` | `codex/backend-abi-w0-gpu` | exact GCC13/CUDA build blocker in `tests/vt/test_dropin_abi.cpp`; CUDA cross-build/runtime/capture evidence for additive W0; owning matrix/roadmap/ledger/state. No ABI or production-kernel redesign | `ACTIVE` | 2026-07-11 (branch `1141b79`: exact clean CUDA 13.0.88/sm_121a all-target rebuild reached 100%; lock-held focused CUDA/ABI CTest 2/2, compute-sanitizer 9/9 cases + 196/196 assertions with 0 errors/leaks, and 35B/27B model gates 2/2 pass. Evidence `~/work/vllm.cpp-online-build/w0-gpu-1141b79`, manifest SHA-256 `4adbe952…601`. Remaining W0: sm_80/sm_90a cross-builds and unchanged-trace/model A/B-memory proof; no runtime source changed) |

## Handoff queue

| Priority | Row/block | Dependency | Next handoff | State |
|---|---|---|---|---|
| 1 | `SERVE-GATE-ONLINE` + FP4 repair | Exact clean `b5c6e4f` 27B checkpoint is binding below floor. W2 improves every concurrency and wins c16/c32 throughput, but c1-c8, TPOT/ITL and host memory remain red. W3-A strict-fails. W3-B at clean `d7cdf66` is immutable build/correctness/safety-green at 80/80 profiles, zero lazy misses, 16/16 correctness and zero memcheck errors. Its repeated component is 1.000293×, strict-fails 15/20 timing + 2/4 memory and leaves 20/80 IDs stable; first-use improves, but no speed credit is granted. Trace attempt 1 is VOID on cache-inventory drift before lazy/vLLM | rerun paired W3-B trace with client artifacts outside the immutable corpus-only eviction root, choose the next execution-grounded lever, gate optional W3-C persistence separately, repeat exact 27B until every performance and memory axis passes, then run 35B | `ACTIVE` |
| 2 | `SERVE-ASYNC-LLM` block (`ENG-CORE-BUSY-LOOP`, `SERVE-ASYNC-LLM`, `ENG-ASYNC-SCHED`, `ENG-PRIORITY-SCHED`) | joint spike accepted ([async-serving.md](specs/async-serving.md)); W2 fixed capacity is GPU-classified but the whole row remains below the vLLM floor | W1/W2/W4 host behavior remains CPU/TSan-green. Fixed/legacy c32 is healthy and neutral; all fresh fixed c32 legs complete. Keep W3 overlap READY and do not mix it into the active FP4 A/B | `GATING` |
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
5. update README and `docs/BENCHMARKS.md` unconditionally for the closing
   checkpoint, plus porting inventory, ledger, and state when relevant.

This keeps `.agents/` focused on current work without losing the discoverable
list of what the project supports.
