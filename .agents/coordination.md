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

Every area-table row must contain these columns (additional area-specific
columns are allowed):

| Required field | Rule |
|---|---|
| `ID` | Stable and never reused; use area prefixes (`ENG`, `MODEL`, `QUANT`, `KERNEL`, `BACKEND`) |
| `Item` | One independently gateable capability or explicit row block |
| `Upstream` | Pinned vLLM `file:line`; include dependency source when it owns execution |
| `Our code` | Exact implementation `file:line`, or `-` when absent |
| `Tests/evidence` | Exact tests/goldens/ledger evidence, or `-` when absent |
| `Spike/spec` | Real `.agents/specs/<slug>.md` once state is `READY` or later |
| `State` | One value from [workflow.md](workflow.md#tabular-lifecycle) |
| `Owner` | Active claim ID, or `-`; completed rows name the closing commit |

`DONE` with `-`, vague prose, a directory-only pointer, or a dangling spec link
is invalid. Split a row when supported and unsupported modes would otherwise be
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
| `CLAIM-PR3` | `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH` | `complete_pr3` | `/home/mudler/_git/vllm.cpp-codex-triton-aot-analysis`; DGX `~/work/vllm.cpp-pr3` | `codex/triton-aot-analysis` | PR #3 files only | `ACTIVE` | 2026-07-10 |

## Handoff queue

| Priority | Row/block | Dependency | Next handoff | State |
|---|---|---|---|---|
| 1 | A1 serve-latency closure | First campaign invalid: vLLM startup failure + ours-35B aborts | diagnose and rerun whole every-axis series under flock | `GATING` |
| 2 | A5 e2e/nightly closure | A1 evidence where benchmarks overlap | write spike and CI/nightly split | `INVENTORIED` |
| 3 | C1 kernel drop-in alignment | accepted kernel-family inventory | spike adapter ABI, then split kernel-family claims | `SPIKE` |
| 4 | C2/C4 models + quantizations | accepted comprehensive inventories | spike generic model factory and GGUF compute-in-quant leaves | `SPIKE` |
| 5 | C3/C5-C8 engine/API work | existing feature rows | complete missing spikes in priority order | `INVENTORIED` |
| 6 | D1 backend expansion | accepted backend/CUDA inventory + vt seam | spike architecture spine, then NVIDIA fan-out, ROCm, MLX, Vulkan, XPU | `SPIKE` |

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
