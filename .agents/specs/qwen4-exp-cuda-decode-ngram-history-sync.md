# `qwen4_exp` CUDA decode: an undrained n-gram history read — REAL, and MEASURED NOT TO BE #2496

**Row:** `MODEL-MM-QWEN4-EXP` (wave TOKENDIV; the branch is
`row/QWEN4EXP-TOKENDIV-2496`, the owning row is the campaign row named in the
issue's own `Row:` line)
**Issue:** [#2496](https://github.com/mudler/vllm.cpp/issues/2496)
**Sibling:** [#2476](https://github.com/mudler/vllm.cpp/issues/2476) — the illegal
memory access that `CUDA_LAUNCH_BLOCKING=1` suppresses. This spec says what it
does and does not claim about that one.
**State:** `ACTIVE`

## THE HEADLINE RESULT: THIS IS NOT #2496's CAUSE

Measured on `thor:gpu0` 2026-09-01, job `2b8df779`, binary
`127b2c6a6c0a2c00b648b6e3b6fca33fc6e0ef2a0be58592031a9bac13169b32` built from
`895d2c430` with the drain asserted present in the source (`drains=2`):

```
CPU-CTRL  (--device cpu,  re-taken in this lease) : 11751 13 15767 411 2029 11 1092 369   http 200,   7s
CUDA-FIX  (--device cuda, CUDA_LAUNCH_BLOCKING=1) : 11751 271 271 271 271 271 0 0        http 200, 188s
CUDA-PROD (--device cuda, no launch blocking)     : (none)                               http 500,  61s
```

`CUDA-PROD` died with `vt cuda: cudaMemcpyAsync: an illegal memory access was
encountered`, three occurrences, from the SAME binary that completed `CUDA-FIX`
without one.

The CUDA arm WITH this fix is **byte-identical to the #2496 measurement**, which
was taken on a DIFFERENT binary
(`70e522df28d7748d8925ce9fc54dffd4909b3e8fd53d72fafd87bb13bed62056`) built from a
DIFFERENT tree (`e934fb002`). This change moved nothing at all in the output.

**Two conclusions follow, and the second is the more useful one.**

1. The hypothesis this spec was written on is FALSIFIED as the cause of #2496.
   The hazard below is real and is proven real by a red-then-green mutation, but
   on this hardware the undrained read was in practice landing before the host
   read it, so it is LATENT rather than active.
2. #2496 is DETERMINISTIC and reproduces bit-for-bit across two independent
   builds of two different trees. That rules out the whole async-copy and
   ordering class — including this one — as its cause, because a race does not
   reproduce byte-identical garbage across builds. The cause is structural, and
   it is upstream of the PLE block: the released `layer_types` makes layers 0-2
   `linear_attention` and `ple_layer_ids` is `[2]`, so the order within a step is
   layer 0 GDN, layer 1 PLE, layer 2 GDN, layer 3 QSA.

This spec therefore lands a latent-hazard fix and its gate, and does NOT close
#2496. The token work is handed to the GDN projection over-read.

## #2496 AND #2476 ARE TWO DEFECTS, AND THIS LEASE MEASURED IT

One binary, three arms, one lease. The two issues have DIFFERENT signatures on
that one binary, and this fix moved NEITHER:

| | serialised | not serialised |
|---|---|---|
| tokens | wrong, and bit-stable | none |
| illegal access | 0 | 3 |

- **#2496 is deterministic and structural.** It reproduces byte-identically
  across two independent builds of two different trees, and it survives
  serialisation. A race cannot do that.
- **#2476 is timing-dependent.** Same binary, same artifact, same prompt; it
  fires only with launch blocking off, and it fired at 61 s here.
- A single defect cannot be simultaneously bit-reproducible across builds and
  suppressible by serialisation. **They are two.**

That also disposes of the hypothesis this spec was written on, which was that
one async-copy site produced both.

## Scope

`qwen4_exp` on `--device cuda` produces a token 0 that matches the CPU control
exactly and then degenerates. This spec covers the PLE block's per-step n-gram
history transfer, which is the one piece of decode state this tree moves with a
raw `Backend::Copy` and then reads on the host without draining the queue.

Out of scope, each named under `## Owed`: the illegal access of #2476 (this
spec's fix is a candidate for it and is not asserted to be it), the `q_f32`
query-path width (#2488, #2502), and any throughput claim.

## The hazard this DOES close (latent, not #2496)

`src/vllm/model_executor/models/qwen4_exp_ple_block.cpp`, one site, two hazards.
Both are live only when the n-gram history cache is NOT host-resident, which on
the production by-name path is exactly the CUDA arm (`qwen4_exp_registry.cpp`
publishes `g.states[3]` on `input.queue.device`).

**H1 — the READ is not drained, and it is DECODE-ONLY.**

```cpp
} else if (tokens_on_host) {
  std::memcpy(hist, tokens_row, hist_bytes);
} else {
  d.b.Copy(d.q, hist, tokens_row, hist_bytes);   // enqueued, not completed
}
```

`hist` is `stage.data()`, a zero-initialised host `std::vector<int64_t>`. Thirty
lines later the HOST reads it:

```cpp
hash_state.tokens.assign(hist, hist + ctx);
```

with no `Synchronize` between the two. `Backend::Copy` on CUDA is
`cudaMemcpyAsync(..., cudaMemcpyDefault, stream)`
(`src/vt/cuda/cuda_backend.cu:116-118`) — it ENQUEUES. **`CUDA_LAUNCH_BLOCKING=1`
does not make `cudaMemcpyAsync` synchronous**; it serialises kernel launches.
So this hazard survives the serialisation that #2496 was measured under, which
is what makes #2496 deterministic where #2476 is not.

The branch is taken only when `past_len != 0`. At `past_len == 0` the history is
seeded with EOS on the host and no device read happens at all. **Prefill
therefore cannot see this defect and every decode step can**, which is precisely
the measured shape: index 0 agrees with the control and indices 1-7 do not.

When the host reads `stage` early it reads zeros, so the n-gram hash runs over
token id 0 repeated. The file's own comment at the seeding branch states the
consequence: "A port that trusted the zero-filled cache would hash token 0 into
the first `ngram_size - 1` positions of every sequence and get a fluent wrong
answer." A constant history gives a constant PLE contribution on every step,
which is a repeated token.

**H2 — the WRITE-BACK is enqueued out of a buffer that then dies.**

```cpp
if (tokens_on_host) {
  std::memcpy(tokens_row, hist, hist_bytes);
} else {
  d.b.Copy(d.q, tokens_row, hist, hist_bytes);   // reads `stage` later
}
```

`stage` is a function-local `std::vector` destroyed when `RunQwen4ExpPleBlock`
returns. Nothing keeps it alive and nothing drains the copy. A DMA still
scheduled against freed host memory is a use-after-free.

## Why this is a port defect and not a judgement call

The sibling translation unit already states the contract this one violates.
`qwen4_exp_qsa_block.cpp` moves words the same way and says so at the definition:

> `StageHostWords` only ENQUEUES the read; nothing is readable until the caller
> synchronises the queue once for every range it staged.

and its `HostWords` wrapper ends in `d.b.Synchronize(d.q)` with the comment
"the device arm cannot forget it". The PLE block forgot it. This is one rule,
already written down in this row, applied at the one site that does not follow
it — not a new policy.

`Backend::Synchronize` is a no-op on the base class, so the CPU arm pays nothing
and its bytes are unchanged.

## Upstream anchor

vLLM `e126687a9a` `[Model] Support Qwen3.8-Flash-Next (#53896)`, a FORWARD
reference: it is 1,565 commits ahead of the pin in `.agents/upstream-sync.md`
and is cited for what decode state must persist, not as a mirror source.
`vllm/models/qwen4_exp/nvidia/model_state.py:65-92` rebuilds `ngram_context`
from `all_token_ids` every step and pads left with EOS only where the position
is negative; a decode step's window is `tokens[P-(N-1) … P-1]` and must contain
the previously generated ids. A window stuck at the EOS/zero seed is the
degenerate case, and it is what H1 produces on a device queue.

## Design

Drain the queue on both sides of the staged row, mirroring `HostWords`:

1. after the device READ, before any host use of `hist`;
2. after the device WRITE-BACK, before `stage` leaves scope.

Both calls are unconditional in the device branch and absent from the host
branch, so the CPU arm is byte-identical.

## Risks

- **A synchronise can CURE a race rather than fix it.** #2476 is timing
  dependent and a drain could hide it. This spec therefore claims the drain as
  the fix for #2496 (deterministic, reproduces under serialisation) and lists
  #2476 as a candidate consequence to be MEASURED, never asserted.
- Two extra stream synchronises per PLE layer per step is a decode cost. It is
  recorded under `## Owed` rather than hidden; correctness first.

## Tests

- A red-first hermetic case driving the block over a backend whose `Copy`
  DEFERS until `Synchronize`, which is what a CUDA stream does and what the CPU
  backend cannot express. Red before the change for the intended reason (the
  step-1 history reads zeros), green after.
- The end-to-end arm: the released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S
  artifact through `examples/server` on `thor:gpu0`, `--device cuda`, greedy,
  `max_tokens=8`, against the CPU control taken in the same lease.

## Gates

The completion condition for #2496 is the GPU emitting the CPU control sequence
`11751 13 15767 411 2029 11 1092 369`. **IT IS NOT MET AND THIS CHANGE DOES NOT
MEET IT** — see the headline result. What is gated here is the hazard alone:
`test_qwen4_exp_ple_block`, red before and green after over a backend whose
host-bound copies enqueue.

## Owed

- **#2496 ITSELF, which this does not fix.** The next hypothesis is the GDN QKVZ
  projection over-read in layer 0, which is upstream of this site and which a
  sibling wave is chasing. Whatever explains #2496 has to explain a byte-identical
  degenerate sequence across two builds AND a correct token 0, so it is
  structural and its extent must depend on the token count (5 at prefill, 1 at
  decode) rather than on timing.
- The measurement that says whether H2 is #2476's illegal access. Not claimed.
- The `q_f32` query-path width (`qwen4_exp_qsa_block.cpp:694`). vLLM keeps the
  query at the model dtype; this tree widens it to f32 and hands it to
  `vt::RmsNorm` beside a bf16 gamma. It is a real memory-format parity debt
  under AGENTS.md "Inherit vLLM defaults" and a token gate cannot see it.
  **It is NOT this defect**: the buffer is allocated unconditionally with no
  device branch and no prefill/decode fork, so it was equally present on the CPU
  run that produced the correct control sequence, and a path that runs
  identically in prefill and decode cannot yield a correct prefill token and
  wrong decode tokens. Owned by #2488 and draft PR #2502.
- The decode cost of the two added synchronises, unmeasured.

## Now

`ACTIVE`.
