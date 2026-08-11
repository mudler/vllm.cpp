# Task guide — porting from vLLM

How to port a model, kernel, or feature. The rules are in
[`AGENTS.md`](../AGENTS.md); this is the method. Nothing here is binding on its
own.

Porting a **model** specifically? Work through
[`porting-a-model.md`](porting-a-model.md) as well — this file is the method,
that one is the coverage checklist (weight formats including GGUF k-quants,
multimodal, speculative decoding, the serving surface, records).

## Before you write anything

Verify the recorded gap against the *current* pinned upstream and the current
local head — not against the spec's description of them, which may be stale.
Check `git log -S` for the symbol and read the row's spec; a surprising amount
of "missing" work already landed.

Enumerate the complete upstream surface before choosing a slice: every mode,
default, dtype, error path, and edge case, plus the upstream tests that cover
them. Classify each one as ported, deliberately deferred, or not applicable.
A slice chosen before this enumeration almost always cuts through the middle of
a mode.

Commit the spike spec with source and dependency anchors. Then implement.

## The port cycle

Implement the smallest coherent *vertical* slice — something that runs
end-to-end, not a layer that nothing calls yet. Keep local names and structure
mechanically traceable to upstream so the next person can diff them by eye.
Record every C++ adaptation you had to make and why.

Port the upstream test before the behavior. Capture the red. Implement. Get
focused green. Run the full gate. Then hand to a fresh reviewer.

Advance the parity pin only after every affected row and gate is reconciled.

## Grounding a claim

Cite the `file:line` you ported from. When a number is involved, read the whole
executing chain rather than the top-level Python: FlashInfer, CUTLASS,
cuBLASLt, DeepGEMM, torch/Inductor, generated code, and the local dispatch path
all decide what actually ran. Dump the generated kernel before concluding a
lever is unreachable.

Anything genuinely written from scratch is recorded as such in
[`porting-inventory.md`](porting-inventory.md) §9. "I couldn't find it upstream"
is a search result, not a conclusion — say which paths you searched.

## Shared seams

Route through the shared path or record one exact tracked exception:

| Behavior | Seam |
|---|---|
| add + RMSNorm and friends | `vt::FusedChain` |
| mergeable MLP projections | `layers::MlpGateUpMethodBase`, `vt::MergedGemmGroup` |
| decode | `ModelRegistry::Forward`, `dense_attn::AttnBlock`, on-device sampling |
| anything a user can reach | `include/vllm.h` |

A capability reachable only through an example's internals is not shipped. Grow
the ABI first, then rewrite the example as a thin client, then delete the
parallel implementation.

New hardware and new models are additive files mirroring vLLM's structure — not
edits that special-case an existing path.

## Where things live

The pinned upstream revision and current parity state are in
[`upstream-sync.md`](upstream-sync.md) and the owning matrices. The parity
ledger records each introduced change against its upstream reference. Matrices
hold current coverage. Do not restate any of it here.
