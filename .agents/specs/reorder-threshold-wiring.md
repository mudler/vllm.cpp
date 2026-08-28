# SPEC-DFLASH2 — the reorder threshold reaches the reorder

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding) — in-flow mirror
repair, not a wave.
**Issue:** [#2129](https://github.com/mudler/vllm.cpp/issues/2129).
**Parent row spec:** [dflash2-spec-decode.md](dflash2-spec-decode.md).
**Related:** [dflash2-spec-as-decode.md](dflash2-spec-as-decode.md) (W10 ported
the formula this change wires).
**Kind:** bug fix. The spec is the first commit of the single pull request that
carries it, so commit order proves the spec came first.

## Now

`ACTIVE` — implementation on `row/SPEC-DFLASH2-reorder-threshold`.

## Scope

IN: pass the already-mirrored spec-as-decode reorder threshold into
`reorder_batch_to_split_decodes_and_prefills` at the runner's one call site; the
smallest test that reds for the intended reason through
`GPUModelRunner::execute_model`; the mutation that proves the test sees the call
site.

OUT:

- Any throughput claim. Nothing here is measured on a device, and the c=8
  measurement floor on the #1574 ladder is ~5.9 %.
- The *other* half of `_may_reorder_batch` we do not mirror: upstream skips the
  reorder entirely when every attention group reports `reorder_batch_threshold
  is None`, which is what `backends/flash_attn.py` and `backends/triton_attn.py`
  do (neither calls `_init_reorder_batch_threshold`, and neither sets the class
  attribute). Our runner has one unconditional reorder with no per-group builder
  registry to take a min over. That divergence predates this change, is inert on
  every non-speculative configuration, and is recorded under `## Owed`.
- #2116 (the async veto at `runner.cpp:425` / `:476` and the sampler path) and
  #2117 / #2125 (the mixed-step instrument). Neither region is touched.

## Upstream anchors (parity pin `5559679229`)

| Behavior | Upstream |
|---|---|
| The reorder takes a `decode_threshold`, default 1 | `vllm/v1/attention/backends/utils.py:665-668` |
| The four-way classification splits on `num_scheduled <= decode_threshold` | `vllm/v1/attention/backends/utils.py:691-709` |
| The runner PASSES its threshold into the reorder | `vllm/v1/worker/gpu_model_runner.py:1126-1130` (`_may_reorder_batch`) |
| The threshold is resolved once, after the builders exist | `vllm/v1/worker/gpu_model_runner.py:7122`, `:7194-7212` (`calculate_reorder_batch_threshold`, min over groups, `None` neutral) |
| A spec-as-decode backend raises it to `1 + (2 if parallel_drafting else 1) * k` | `vllm/v1/attention/backend.py:657-687` (`_init_reorder_batch_threshold`) |
| The hybrid linear-attention backend passes `supports_spec_as_decode=self.use_spec_decode` | `vllm/v1/attention/backends/gdn_attn.py:112` |
| `parallel_drafting` is True for `dflash` and `dspark`, and only those | `vllm/config/speculative.py:963-964` |
| The GDN split still uses a HARDCODED 1, not the reorder threshold | `vllm/v1/attention/backends/gdn_attn.py:213` |

Local mirrors already in the tree, all landed before this change:

- `include/vllm/v1/attention/backend.h:180-184` —
  `SpecAsDecodeReorderThreshold(k, parallel_drafting)`, an exact mirror of the
  `backend.py` arithmetic.
- `include/vllm/config/speculative.h:158`, `:346` — `parallel_drafting = true`
  in `ResolveDflash` and `ResolveDspark`, and nowhere else. It is READ, not
  assumed: `runner.cpp:2067` already reads `spec_config_->parallel_drafting`
  for the classification.
- `src/vllm/v1/attention/backends/gdn_attn.cpp:156` — the non-spec GDN split
  passes `decode_threshold=1`, matching `gdn_attn.py:213`. This change does not
  touch it, because upstream does not.

## The defect

`src/vllm/v1/worker/gpu/runner.cpp:1691` calls the reorder with no threshold, so
it takes the declaration default of 1
(`include/vllm/v1/worker/gpu/runner.h:102-104`).
`git grep SpecAsDecodeReorderThreshold -- src include` finds one caller —
`backend.h:201`, inside the same header, serving the *classification*
(`SpecAsDecodeQueryLen`, consumed at `runner.cpp:2065`). The formula was ported
in W10 and never reached the reorder it is named for.

At `k = 8` with `parallel_drafting` the threshold should be 17 and is 1. A
`1 + k = 9`-token verify row has context and is done prefilling, so at 17 it
classifies `decode` (region 0) and leads the batch; at 1 it classifies
`long_extend` (region 2) and sorts among the chunked-prefill continuations.

## Design

One call site, one argument. Add a private const accessor on `GPUModelRunner`
mirroring upstream's resolved `self.reorder_batch_threshold`:

```cpp
int reorder_batch_threshold() const {
  return static_cast<int>(v1::SpecAsDecodeReorderThreshold(
      num_spec(), spec_config_.has_value() && spec_config_->parallel_drafting));
}
```

and pass it at `runner.cpp:1691`. `SpecAsDecodeReorderThreshold` returns 1 for
`k <= 0`, and `num_spec()` is 0 with no `SpeculativeConfig`, so **every
non-speculative configuration keeps threshold 1 and is byte-identical**. The
constructors are NOT touched: #2116 owns `runner.cpp:425` and `:476`, and the
value is cheap enough to resolve at the call site.

Threshold values this produces: `dflash`/`dspark` at k=8 give 17; `mtp`,
`ngram` and `draft_model` (`parallel_drafting == false`) at k=8 give 9; k=0
gives 1.

## What this does NOT change

- `SplitDecodesAndPrefills` keeps its hardcoded 1 on the non-spec GDN branch,
  because `gdn_attn.py:213` does. A step that carries drafts takes the spec
  branch, which is mask-driven and never consults the threshold, so no step can
  reach the non-spec branch with a `q > 1` row in front of the decodes.
- `SpecAsDecodeQueryLen` at `runner.cpp:2065` is unchanged; it recomputes the
  same threshold from the same inputs.

## Tests

`tests/vllm/v1/worker/test_runner.cpp`, two new cases, driven through
`GPUModelRunner::execute_model` — the production entry point the existing
four-way ordering gates in that file already use. The speculative case builds
the runner through the public `LoadedModel&` constructor (which already takes a
`SpeculativeConfig`) over `BorrowQwen3_5MoeLoadedModel`, so no production
signature moves.

Discriminating population, at `k = 8`, `ResolveMtp` (so `parallel_drafting` is
false and the mirrored threshold is `1 + 1*8 == 9`), admitted `[C, X10, V9, D1]`:

| row | scheduled | still prefilling | region at 1 | at 9 (mirrored) | at 17 |
|---|---:|---|---|---|---|
| `D1` | 1 | no | decode 0 | decode 0 | decode 0 |
| `V9` | 9 | no | long_extend 2 | **decode 0** | decode 0 |
| `X10` | 10 | no | long_extend 2 | long_extend 2 | **decode 0** |
| `C` | 15 | yes | long_extend 2 | long_extend 2 | short_extend 1 |

`pos(V9) < pos(X10)` is the one predicate that holds ONLY at the mirrored
value: at 1 the verify row is a long extend and sorts behind `X10`; at 17 `X10`
is promoted to a decode and the min-swap partition puts it ahead of `V9`. The
case also asserts the reorder's own contract — the final order is
non-decreasing in the region the mirrored threshold assigns — rather than a
hardcoded permutation, because upstream partitions with minimum swaps and not a
stable sort, which the 3-request case above it already explains. The per-slot
fields (`seq_lens`, block-table column 0, the `query_start_loc` delta) are
cross-checked against an order-independent oracle, so no assertion can be
satisfied by permuting `req_ids` alone.

The second case pins the non-speculative arm byte-identical: the same four rows
on a runner with no `SpeculativeConfig` keep the pre-fix partition, in which
only `D1` is a decode. Dropping the threshold argument leaves that case green,
which is what makes it the byte-identity pin rather than a second copy of the
gate.

## Gates

Focused (CPU, no GPU needed):

```sh
cmake --build build --target test_runner -j 4
./build/tests/test_runner -ts="*"
```

Affected suites (every one drives a speculator through the reorder):
`test_mtp_depth`, `test_dflash2_runner_reach`, `test_dflash2_concurrency`,
`test_dflash2_prefix_cache`, `test_dflash_causality`, `test_dspark_sample`,
`test_ngram_proposer`, `test_llm_engine`, `test_engine_core_proc`,
`test_async_scheduler`, `test_scheduler`, `test_spec_as_decode`,
`test_input_batch`, `test_prepare_inputs`, `test_rejection_sampler`.

Full: `scripts/agent-preflight.sh`, plus the record checkers named in the pull
request body if the box load prevents the wrapper from completing.

Mutations. Rebuild and confirm the binary timestamp is fresh before reading a
result — a mutation whose build fails reads as a passing test, and the #2117
implementer hit exactly that.

- **MUT-M1 (reachability)**: restore the call at `runner.cpp` to its
  no-argument form, keeping the accessor referenced so `-Werror` cannot turn a
  red into a build failure.
- **MUT-M2 (the wrong arm)**: make the accessor pass `true` for
  `parallel_drafting` unconditionally, so the threshold is 17 rather than 9.

## Evidence (this box, CPU, `row/SPEC-DFLASH2-reorder-threshold`)

Configured `-DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF`, built with `-j 4`.

| Arm | `test_runner` | Exit |
|---|---|---|
| RED (test only, no production change) | 28/29 cases, 819/822 assertions, `FAILURE!` | 1 |
| GREEN (first population) | 29/29, 822/822, `SUCCESS!` | 0 |
| GREEN (final population) | 29/29, 831/831, `SUCCESS!` | 0 |
| MUT-M1 | 28/29, 829/831, `FAILURE!` | 1 |
| MUT-M2 | 28/29, 829/831, `FAILURE!` | 1 |
| restored | 29/29, 831/831, `SUCCESS!` | 0 |

Both mutations red on the same two assertions, `pos("V9") < pos("X10")`
(`2 < 1`) and the region-monotonicity check (`2 <= 0`), both logging
`order: D1 X10 V9 C`. The tree was restored to a recorded `sha256sum -c`
match after each one.

**MUT-M2 SURVIVED the first population, and that is the finding worth keeping.**
The first draft of this case used three rows (`C`, `X10`, `V9`) and claimed in
its own comment that `X10` pinned the threshold to 9. It does not: with only
those three rows the min-swap partition produces `[V9, X10, C]` at BOTH 9 and
17, so forcing the parallel-drafting arm changed nothing observable and the
mutation passed. The plain single-token decode `D1` is what makes the two
thresholds produce different orders, because it is the row that stays a decode
while `X10` crosses. The comment was a claimed guarantee that the test did not
have, found only by running the mutation rather than by reading it.

Affected suites, all `SUCCESS!` with exit 0: `test_mtp_depth` 10/10 (123),
`test_dflash2_runner_reach` 10/10 (176), `test_dflash2_concurrency` 2/2 (21),
`test_dflash_causality` 13/13 (105), `test_dspark_sample` 10/10 (92),
`test_ngram_proposer` 2/2 (19), `test_llm_engine` 24/24 (493),
`test_engine_core_proc` 15/15 (125), `test_async_scheduler` 9/9 (86),
`test_scheduler` 38/38 (459), `test_spec_as_decode` 6/6 (21),
`test_input_batch` 29/29 (205), `test_prepare_inputs` 9/9 (61),
`test_rejection_sampler` 10/10 (139), `test_dflash2_prefix_cache` 5/5 (69).

`test_deepseek_v2_paged_engine` is a **SKIP, not a pass**: 1 case, **0
assertions**, printing `SKIP (dgx-only): DeepSeek-V2-Lite checkpoint absent`.
It is recorded here because `assertions: 0` reads as a pass in a summary line.

## Owed

- **We reorder where upstream does not.** `_may_reorder_batch`
  (`gpu_model_runner.py:1108-1130`) returns early when every attention group
  reports `reorder_batch_threshold is None`, and `flash_attn.py` /
  `triton_attn.py` never set one, so a plain FlashAttention model upstream does
  no reorder at all. We reorder unconditionally at 1. That predates this change
  and is inert for correctness (the reorder is a permutation and every consumer
  is built after it), but it is a divergence and nothing in the tree records it.
  Tracked by [#2129](https://github.com/mudler/vllm.cpp/issues/2129) and owed
  here until a row picks it up.
- **No device measurement.** Whether the corrected order changes batch
  composition on a live server is unmeasured. The readout exists —
  `VT_GRAPH_STATS=N` prints `[graph-dispatch]` counters including
  `ragged_mixed_steps` since `f9af269f9` — and needs a lease.

## Stop conditions

Return `NEEDS_DECISION` if the corrected threshold makes any consumer of the
reordered order inconsistent in a way the threshold alone cannot reconcile —
specifically if a step can reach `SplitDecodesAndPrefills`'s non-spec branch
with a `q > 1` row ahead of the decodes.

## Outcome

Filled in when the row reaches `DONE`. The one thing already worth carrying
forward is in `## Evidence`: the first version of the gate carried a claim
about what it pinned that was false, and only MUT-M2 found it. The reorder is a
minimum-swap partition, so two different thresholds can produce the same final
order on a population that has no row sitting between them.
