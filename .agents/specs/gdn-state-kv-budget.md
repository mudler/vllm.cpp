# GDN recurrent state is preallocated per CONFIGURED sequence, on an axis no budget bounds

Row `KV-GDN-STATE-BUDGET`. Issue
[#1983](https://github.com/mudler/vllm.cpp/issues/1983).

## Now

`ACTIVE`. Spec and implementation land in one pull request (the repository
default; no split case applies).

## Scope

One thing: the number of recurrent (Mamba/GDN) state slots the runner
preallocates stops being a function of `--max-num-seqs` and becomes a function
of the KV budget, which is what upstream does.

OUT of scope, and named so a reviewer does not expect them:

- The `dflash` `P > 1` propose fallback that #1983 also names as a candidate
  for the CUDA illegal access (`src/vllm/model_executor/models/qwen3_dflash.cpp`).
  Different mechanism, different row.
- The placeholder KV-group layer names (`"fa"`, `"gdn"`), which make
  `KVBytesPerBlock` and `recurrent_state_bytes` under-count by the layer count.
  Owned by [#1963](https://github.com/mudler/vllm.cpp/issues/1963) and
  [#1966](https://github.com/mudler/vllm.cpp/issues/1966). See `## Coordination`.
- Any change to the paged block count. `ResolveNumBlocks` is untouched.
- Making GDN state slots a first-class KV-manager group with a block table.
  That is the real upstream shape and it is a campaign, not this fix.

## The defect

`src/vllm/v1/worker/gpu/runner.cpp::GPUModelRunner::initialize_kv_cache`:

```cpp
const int spec_cols = spec_on() ? num_spec() + 1 : 1;
const int64_t base_slots = max_num_reqs_ > 0 ? max_num_reqs_ : num_blocks_;
gdn_state_slots_ = base_slots * spec_cols;
```

and, per GDN layer, one conv buffer and one SSM buffer of
`gdn_state_slots_ * row_elems * element_size`, each `Memset` to zero at
construction (`GPUModelRunner::CacheBuffer::CacheBuffer`), so every byte is
touched and resident before the first request arrives.

Re-derived independently for `Qwen3.8-27B` (`.agents/specs/qwen36-forward-notes.md`
§1: 64 layers, `[LA,LA,LA,FA] x 16` so 48 linear-attention layers,
`Hk/Hv/Dk/Dv/conv = 16/48/128/128/4`, `mamba_ssm_dtype = float32`,
`num_key_value_heads = 4`, `head_dim = 256`) at `num_speculative_tokens = 8`:

| term | arithmetic | bytes |
|---|---|---|
| SSM row, one layer | `48 * 128 * 128 * 4` (f32) | 3,145,728 |
| conv row, one layer | `(2*16*128 + 48*128) * (4-1+8) * 2` (bf16) | 225,280 |
| one slot, one layer | | 3,371,008 |
| one slot, 48 layers | `* 48` | 161,808,384 = 154.31 MiB |
| one SEQUENCE (`k+1 = 9` slots) | `* 9` | 1,456,275,456 = 1.356 GiB |
| `--max-num-seqs 32` | `* 32` | 46,600,814,592 = **43.40 GiB** |

which reproduces the operator's measured slope (1.22 / 1.50 / 1.23 GiB per
configured sequence over `--max-num-seqs` 16 -> 20 -> 24 -> 28) and the 43.40 GiB
headline. The per-sequence cost is NOT the divergence: upstream charges the same
`1 + num_speculative_blocks` state blocks per request
(`vllm/v1/kv_cache_interface.py::MambaSpec.max_memory_usage_bytes`), and our SSM
`f32` mirrors the checkpoint's own `mamba_ssm_dtype`
(`src/vllm/model_executor/models/qwen3_5.cpp::detail::ResolveMambaSsmCacheDType`
against `vllm/model_executor/models/config.py::Qwen3_5ForConditionalGenerationConfig`).

**The divergence is the axis.** `--max-num-seqs` is a scheduler cap upstream. It
sizes no allocation anywhere in vLLM. Here it multiplies a 154 MiB-per-slot
allocation, and no flag bounds the product: not `--kv-cache-memory`, not
`--num-blocks`, not `--gpu-memory-utilization`.

## What upstream does instead, read at the pin

Pin `555967922` (`.agents/upstream-sync.md`), read in
`/home/mudler/_git/vllm` (a shallow clone; the files below are all present at
the pinned tip).

1. `vllm/platforms/interface.py::Platform.check_and_update_config` (:853-935)
   computes `mamba_page_size` from the model's own state shapes, then RAISES the
   attention block size until one attention page is at least one mamba page:

   ```python
   kernel_block_alignment_size = max(
       min(s.base if isinstance(s, MultipleOf) else s
           for s in backend_cls.get_supported_kernel_block_sizes()),
       cache_config.block_size)
   attn_block_size = kernel_block_alignment_size * cdiv(
       mamba_page_size, kernel_block_alignment_size * attn_page_size_1_token)
   ```

   (`FlashAttentionBackend.get_supported_kernel_block_sizes` returns
   `[MultipleOf(16)]`, `vllm/v1/attention/backends/flash_attn.py:83-84`, so the
   floor is 16.) It then pads the mamba page up to the attention page exactly
   (`:917-935`).

2. `vllm/v1/core/kv_cache_utils.py::_unify_hybrid_kv_cache_specs` (:1073-1131)
   finishes the unification for any residual mismatch, so every layer of a
   hybrid model has ONE page size.

3. `vllm/v1/core/kv_cache_utils.py::get_num_blocks` (:993-1011) and
   `_get_kv_cache_config_uniform_page_size` (:1399-1416) then size the whole
   pool from the memory budget and share it:

   ```python
   group_size = max(len(group.layer_names) for group in kv_cache_groups)
   page_size  = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])
   num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
   ```

   Each `KVCacheTensor` is `page_size * num_blocks` and is `shared_by` one layer
   from EACH group — so a mamba layer's state blocks and an attention layer's KV
   blocks are drawn from the same `num_blocks` identifier space. A request's
   `1 + num_speculative_blocks` state blocks are blocks of that pool.

**Consequence, which is the property we are porting:** upstream's total recurrent
allocation is `num_blocks * page_size` per mamba layer, `num_blocks` comes from
the memory budget, and `max_num_seqs` appears nowhere in it. Concurrency is
limited by pool exhaustion at schedule time, never by a preallocation.

## Design

Mirror the property, not the machinery. The pool stays a compact per-sequence
slot pool (no block table, no KV manager group); what changes is where its size
comes from.

`ComputeHybridKvBudget(kv_cfg, kernel_block_alignment)` reproduces upstream's own
arithmetic on the specs we already build, and answers one question: **how many
sequences' recurrent state does this KV budget hold?**

```
mamba_page          = mamba_spec->page_size_bytes()                 // one layer, one slot
attn_bytes_per_token = attn_spec->real_page_size_bytes() / attn_spec->storage_block_size()
align                = max(16, attn_spec->block_size)               // interface.py:875-882
unified_block_tokens = align * cdiv(mamba_page, align * attn_bytes_per_token)
unified_num_blocks   = (num_blocks * attn_spec->block_size) / unified_block_tokens
slots_per_seq        = 1 + mamba_spec->num_speculative_blocks
max_state_seqs       = unified_num_blocks / slots_per_seq
```

`unified_num_blocks` is a token-count identity: our pool holds
`num_blocks * block_size` tokens per attention layer, and one upstream page holds
`unified_block_tokens` of them, so the same bytes buy that many upstream pages.
No layer count appears anywhere — upstream's per-layer page equality cancels it —
which is why this row does not depend on the layer-name repair (`## Coordination`).

`LoadedEngine` then resolves ONE concurrency and hands it to every consumer:

```
max_num_seqs_ = min(configured, budget.max_state_seqs)
```

used at all six sites that read `params.max_num_seqs` today: the runner's
`max_num_reqs`, three `MakeSchedulerConfig` calls, the
`StructuredOutputManager`, and the #371 state guard. Because the scheduler's
running-request cap and the runner's slot pool are now the same number,
`remap_gdn_state_slots`'s "GDN state slots exhausted" `VT_CHECK` becomes
unreachable rather than newly reachable.

### Three deliberate choices

**The bound is an UPPER bound, on purpose.** A running sequence also holds
attention pages out of the same upstream pool, so upstream's effective
concurrency is strictly below `unified_num_blocks / slots_per_seq`. We do not
charge those, because a static cap that is TIGHTER than upstream's dynamic
scheduler would refuse work upstream serves. Ours is a ceiling on the
allocation, not a scheduler.

**Reduce, never refuse.** `check_enough_state_memory` refuses rather than
clamping, and that polarity is right for *"the state does not fit in memory at
all"*. This is a different question — *"the KV budget you asked for holds state
for N sequences"* — and upstream's answer to it is to serve N and queue the
rest, with no error. Refusing here would turn every over-configured
`--max-num-seqs` into a failed start. The #371 refusal is untouched and still
fires ahead of this clamp when even the clamped state exceeds available memory.

**A model with no recurrent group, and a model with no attention group, are both
inert.** No mamba spec means nothing to bound. No attention spec means no page to
unify against; upstream's unification has nothing to do there either, so the
budget reports `unbounded` and the clamp is skipped. Recorded under `## Owed`.

### What the numbers become

At the operator's configuration (`--num-blocks 3072`, `block_size 32`,
`--max-model-len 8192`, 27B geometry above, `num_speculative_tokens 8`):

```
attn_bytes_per_token = 2 * 4 * 256 * 2                       =     4,096
align                = max(16, 32)                           =        32
unified_block_tokens = 32 * cdiv(3371008, 32*4096) = 32 * 26 =       832
unified_num_blocks   = 3072 * 32 / 832                       =       118
max_state_seqs       = 118 / 9                               =        13
```

so the recurrent allocation becomes `13 * 1.356 GiB = 17.6 GiB` and stays there
at `--max-num-seqs` 16, 20, 24, 28 and 32 alike. Predicted values are in
`## The measurement`.

**Spec OFF is byte-identical at this configuration.** With `num_spec = 0` the
conv row narrows to 3 taps, `mamba_page = 3,207,168`,
`unified_block_tokens = 32 * cdiv(3207168, 131072) = 800`,
`unified_num_blocks = 122`, `slots_per_seq = 1`, so `max_state_seqs = 122` and
`min(32, 122) = 32` — no clamp. The change bites exactly the
speculation x concurrency product that #1983 measured, and is inert on the
non-speculative engine.

## Coordination with #1963 and #1966

`MakeQwen3_5KVCacheSpec` publishes ONE placeholder layer name per group
(`"fa"`, `"gdn"`), so `KVBytesPerBlock` (#1963) and `recurrent_state_bytes`
(#1966) both weight by 1 where the runner allocates 48 (attention) or 48
(GDN) layers. That is why the #371 guard computes 0.90 GiB against a 43.40 GiB
allocation: `43.40 / 48 = 0.904`.

**This row does not depend on either landing first, and touches neither file.**
`ComputeHybridKvBudget` reads per-LAYER page sizes off the specs and never a
layer count, for the reason given in `## Design`. It lives in its own
translation unit (`v1/core/hybrid_kv_budget.{h,cpp}`) precisely so the three
rows do not share an edit surface.

**This row is the single source of truth for how many sequences' recurrent state
the engine allocates.** After it lands, `LoadedEngine` passes the RESOLVED
`max_num_seqs_` into `recurrent_state_bytes`, so #1966's repair of the layer
count makes the #371 guard agree with the allocation without either row
re-deriving the other's number. #1966 should fix the layer weighting and read
the resolved concurrency; it should not re-implement the budget.

## Reachability

Production entry point: `vllm::entrypoints::LoadedEngine`'s constructor, reached
from `include/vllm.h` (`vllm_model_load` -> `LoadedEngine`), from the loader, and
from every server path. The resolved value flows into `GPUModelRunner`'s
`max_num_reqs` and into `SchedulerConfig::max_num_seqs`.

Reachability mutation: revert the six call sites to
`params.max_num_seqs > 0 ? params.max_num_seqs : 8` in a scratch copy and rerun
the focused gate. The engine test must go red, because it reads
`eng.runner().gdn_state_slots()` off an engine built through the production
constructor and never constructs a runner by hand.

## Tests

`tests/vllm/v1/core/test_hybrid_kv_budget.cpp` — the arithmetic, driven from
`MakeQwen3_5KVCacheSpec` (the REAL registry builder) and never a hand-built
group, because a fixture that hands the code a shape no registry emits is how
#1963 stayed hidden (`tests/vllm/v1/test_kv_cache_interface.cpp:425-433` is that
fixture).

`tests/vllm/entrypoints/test_gdn_state_budget_engine.cpp` — the engine gate. A
CPU `LoadedEngine` over a hybrid Qwen3.5 dense config with
`num_speculative_tokens`, asserting that:

- `runner().gdn_state_slots()` does NOT grow with `--max-num-seqs` once the
  budget binds (the red assertion: today it is exactly `max_num_seqs * (k+1)`);
- the recurrent bytes those slots represent are bounded by the KV budget;
- the spec-off engine is unchanged.

## Gates

```sh
cmake --build build -j 4
ctest --test-dir build -R 'hybrid_kv_budget|gdn_state_budget_engine' --output-on-failure ; echo "rc=$?"
ctest --test-dir build --output-on-failure ; echo "rc=$?"
scripts/agent-preflight.sh --staged ; echo "rc=$?"
python3 scripts/check-commit-trailers.py --range origin/main..HEAD ; echo "rc=$?"
```

## The measurement

Device measurement is the operator's; this row does not merge on a green
compile. The requested run is the operator's own sweep, re-run on this branch:
same binary recipe, `--num-blocks 3072 --max-model-len 8192`,
`--speculative-config '{"method":"dflash", ..., "num_speculative_tokens":8}'`,
852-token prompt / 64 output, ONE request in flight, warm second request, over
`--max-num-seqs` in {16, 24, 28, 32}.

Predicted, stated before the run:

| `--max-num-seqs` | resident GiB | avail after load GiB | warm tok/s |
|---|---|---|---|
| 16 | 40.8 +/- 2 | 44 +/- 3 | 17.3 - 17.8 |
| 24 | 40.8 +/- 2 | 44 +/- 3 | 17.3 - 17.8 |
| 28 | 40.8 +/- 2 | 44 +/- 3 | 17.3 - 17.8 |
| 32 | 40.8 +/- 2 | 44 +/- 3 | 17.3 - 17.8 |

**Read the seat count off the engine, not off this table.** The 13 above is
derived from the 27B geometry recorded in
`.agents/specs/qwen36-forward-notes.md` §1 (`num_key_value_heads 4`,
`head_dim 256`), which was read from a `Qwen3.6-27B` config.json. If the
`Qwen3.8-27B` attention geometry differs, the seat count moves with it and every
row of the table moves proportionally. The engine now PRINTS the resolved number
and the terms it came from, so the run reports its own bound rather than asking
the reader to trust this arithmetic. What does NOT move with the geometry is the
slope.

The load-bearing prediction is the SLOPE: **0.00 GiB per configured sequence**,
against 1.22 / 1.50 / 1.23 measured on `main`. Residency and throughput must be
flat across the four rungs. A startup line must name the clamp
(`max_num_seqs 32 -> 13`).

Falsifiers, stated in advance:

- any residual slope above 0.1 GiB per configured sequence refutes the fix;
- tok/s at 32 below 15.0 means the collapse has another term this row does not
  reach, and the row does not merge on that evidence;
- tok/s at 16 below 17.0 means the clamp itself cost throughput (13 seats where
  16 were configured), which is a regression this row owns.

## Risks

- **The clamp reduces served concurrency.** At `--num-blocks 3072` with `k=8`
  the engine will run 13 sequences where 32 were configured. That is the honest
  number for that budget, and raising `--num-blocks` raises it, but a concurrency
  ladder that expected 32 seats will now report 13. It is a behavior change on a
  user-visible axis and it is logged, not silent.
- **32-way concurrency at `k=8` may not be servable on this box at all.**
  `32 * 1.356 GiB = 43.4 GiB` of state, on top of weights and the paged pool,
  against ~61 GiB free after load. That is arithmetic, not a measurement, and it
  says the #1574 ladder's upper rungs need a smaller `k` or a bigger box.
- **The DEFAULT pool path now caps a speculating hybrid engine hard.** With
  neither `--num-blocks` nor `--kv-cache-memory`, `ResolveNumBlocks` falls back
  to 256 blocks, because the `gpu_memory_utilization` profile run is unported
  (ROAD-V1-MEM M3, [#83](https://github.com/mudler/vllm.cpp/issues/83)). At the
  27B geometry with `k=8` that is `256*32/832 = 9` unified pages and ONE seat.
  That is the consistent answer -- a 256-block pool holds 8192 tokens, one
  max-length sequence -- and today the same configuration allocates 43.40 GiB of
  state against a 0.8 GiB paged pool, which is the defect at its purest. But it
  IS a behavior change on the default path, and it will stay conservative until
  #83 lands a real budget. Named rather than discovered.

- **An unexpected clamp in an existing green test.** Any hybrid test whose tiny
  `num_blocks` makes `max_state_seqs` small would silently lose concurrency. The
  full suite is the control; a clamp that fires in a test is a finding, not a
  fixup.

## Owed

- [#1983](https://github.com/mudler/vllm.cpp/issues/1983) also names the
  `dflash` `P > 1` propose fallback as a candidate for the CUDA illegal access.
  This row does not touch it and does not close the issue's crash half.
- A pure-recurrent model (no attention group) has no page to unify against and
  is left unbounded by this row.
- The real upstream shape — recurrent state as a KV-manager group sharing the
  block pool, with schedule-time exhaustion instead of a static cap — remains
  unported.

## Stop conditions

- Return `NEEDS_DECISION` rather than widening scope if the clamp turns out to
  require the layer-name repair (#1963/#1966) to compute a correct bound.
- Do not touch `src/vllm/v1/core/kv_cache_utils.cpp` or
  `src/vllm/v1/kv_cache_interface.cpp`; they are #1966's and #1963's surfaces.
- Do not run a device measurement. The operator holds the lease.

## Outcome

Pending. Filled when the row reaches `DONE`.
