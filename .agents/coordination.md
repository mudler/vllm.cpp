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
| `CLAIM-PR3` | `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH` | `validate_pr3` (task #52, took over from `complete_pr3`) | DGX `~/work/vllm.cpp-noPy` (reused disposable clone, own build dir) | `merge-pr3-validation` (local merge of `pull/3/head` into main) | PR #3 files; claim/matrix rows `KERNEL-GDN-AOT-BF16`+`KERNEL-GDN-SCRATCH`; ledger/inventory/roadmap rows for PR #3 closure. GPU lock: one flock per test + one for the pool A/B series | `ACTIVE` | 2026-07-10 (validation pass: build TU, test_ops_gdn, both greedy gates, pool A/B, AOT byte-repro) |
| `CLAIM-SERVE-GATE-1` | `SERVE-GATE-ONLINE` | `serve-gate-online` (task #42) | local main (docs only); DGX `~/work/vllm.cpp-latency` (harness/build @ 83010c7) + `~/work/vllm.cpp-latency/latres*` | `main` (docs-only edits per push policy) | DGX latency dir + `lat_*.sh` harness; engine-matrix `SERVE-GATE-ONLINE` row; roadmap `ROAD-V1-A` row; ledger appends. Lock plan: short diag window (vLLM ninja/PATH preflight + ours-35B IMA sanitizer repro), then ONE `flock` per model A/B series (27B ours+vLLM single hold; 35B ours+vLLM single hold) | `ACTIVE` | 2026-07-10 (diagnosed: vLLM init = ninja not on PATH for flashinfer JIT; ours-35B = device IMA under c16 online serve, NOT the merged 249a569 pad bug — sanitizer capture next) |
| `CLAIM-QGC-LEAVES` | `QUANT-GGUF-COMPUTE` | `qgc-leaf-specs` spike agent | local main (docs only; spike is read-only outside spec/matrix/coordination rows) | `main` (docs-only edits per push policy) | New leaf specs `.agents/specs/gguf-cpu-threadpool.md`, `.agents/specs/gguf-compute-in-quant-gemm.md`, `.agents/specs/gguf-keep-quant-loader.md`; quantization-matrix `QUANT-GGUF-COMPUTE` block + new leaf rows; roadmap `ROAD-V1-C4` row; this claim + handoff row 6. No GPU lock needed (docs-only, no GPU execution) | `ACTIVE` | 2026-07-10 (splitting the umbrella spike into claim-sized leaf specs per the B4 decision measurement, parity-ledger.md L290) |
## Handoff queue

| Priority | Row/block | Dependency | Next handoff | State |
|---|---|---|---|---|
| 1 | `SERVE-GATE-ONLINE` | First campaign invalid: vLLM startup failure + ours-35B aborts | claimed by `CLAIM-SERVE-GATE-1`; both failures diagnosed, rerun in progress | `ACTIVE` |
| 2 | `SERVE-E2E-NIGHTLY` | `SERVE-GATE-ONLINE` evidence where benchmarks overlap | write spike and CI/nightly split | `INVENTORIED` |
| 3 | C1 kernel drop-in alignment | accepted kernel-family inventory | spike adapter ABI, then split kernel-family claims | `SPIKE` |
| 4 | `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` | accepted spike; no dependency on async serving | provision digest-pinned image, build the corpus/harness, then classify each exact checkpoint | `READY` |
| 5 | `BACKEND-GATE-CUDA-SGLANG` | `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT`, then `SERVE-ASYNC-LLM` | run the binding c1-c16 campaign only after both dependencies close | `BLOCKED` |
| 6 | C2/C4 models + quantizations | accepted comprehensive inventories | spike `MODEL-FACTORY-registry` and split `QUANT-GGUF-COMPUTE` leaves | `SPIKE` |
| 7 | C3/C5-C8 engine/API work | existing feature rows | complete missing spikes in priority order | `INVENTORIED` |
| 8 | D1 backend expansion | accepted backend/CUDA inventory + vt seam | spike architecture spine, then NVIDIA fan-out, ROCm, MLX, Vulkan, XPU | `SPIKE` |

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
