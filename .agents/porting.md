# Task guide — porting from vLLM

How to port a model, kernel, or feature. The rules are in
[`AGENTS.md`](../AGENTS.md); this is the method. Nothing here is binding on its
own.

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
end-to-end, not a layer that nothing calls yet. That is a rule, not a
preference: see [`reachability.md`](reachability.md) for what counts as reached,
and for how to land a staged slice that is not reached yet. Keep local names and
structure mechanically traceable to upstream so the next person can diff them by
eye. Record every C++ adaptation you had to make and why.

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

## Mirror the memory format, not just the math

**A token-exactness gate cannot catch a dtype that is too WIDE.** F32 where
upstream uses bf16 is *more* precise: tokens still match, SACRED still passes,
and we quietly move twice the bytes. Every correctness gate we own is blind to
it. So it has to be checked deliberately, once per ported path:

| Ask | Where upstream answers it |
|---|---|
| What dtype does the linear method OUTPUT? | the quant method's `apply`/`out_dtype` (e.g. ModelOpt fp8 uses `torch.get_default_dtype()` = bf16) |
| What `kv_cache_dtype` is RESOLVED for this checkpoint? | it is not always the CLI default — vLLM derives it from `kv_cache_quant_algo` in the checkpoint's `quantization_config` |
| What dtype do the intermediate activation buffers carry? | read the consumer, not the producer: a buffer is only as narrow as whoever reads it |
| Is a projection one physical GEMM or several? | merged linears (`QKVParallelLinear`, `MergedColumnParallelLinear`) are one, and upstream may requantize mismatched shards rather than decline to merge |

Record the answers in the row's spec. If we deliberately diverge — a wider
accumulate for a reason — say so and say what it costs in bytes per token.

**Confirm the resolved config at RUNTIME, not from source.** Upstream logs its
resolved engine config on startup; read it. This project has repeatedly had a
confident source reading contradicted by a runtime log — a kernel name read as a
fusion, a code path read as selected when a capability predicate excluded it, a
cache dtype read as the CLI default when the checkpoint overrode it. Source
inspection establishes candidates; the running engine establishes what ran.

**Why this section exists:** a dtype divergence on the largest activation
buffers is a per-token cost. It is therefore invisible at batch 1 with a short
prompt, invisible to token gates, and shows up only as a flat throughput deficit
that does not shrink with concurrency — the hardest signature to attribute after
the fact.

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
parallel implementation. The same holds for the shapes no checker sees: a
parameter no caller passes, a branch no released config selects, a flag with no
default path through it.

New hardware and new models are additive files mirroring vLLM's structure — not
edits that special-case an existing path.

## Where things live

The pinned upstream revision and current parity state are in
[`upstream-sync.md`](upstream-sync.md) and the owning matrices. The parity
ledger records each introduced change against its upstream reference. Matrices
hold current coverage. Do not restate any of it here.
