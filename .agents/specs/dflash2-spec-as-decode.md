# SPEC-DFLASH2 — spec-as-decode: the uniform-qlen verify stays on the decode-class FA-2 kernel

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding) — wave W10.
**Issue:** [#1857](https://github.com/mudler/vllm.cpp/issues/1857).
**Parent row spec:** [dflash2-spec-decode.md](dflash2-spec-decode.md).
**Kind:** wave spec, committed before the implementation in the same pull
request (the W9 shape: the row's base spec landed in its own pull request on
2026-08-19 per the recorded `## Git integration` preference; each wave since
then carries its wave spec as the first commit of the wave's single pull
request, and this wave does the same).

## Now

`ACTIVE` — wave implementation in flight on `row/SPEC-DFLASH2-spec-as-decode`.

## Why this row owns it

#1857 names two candidate owners: `SPEC-DFLASH2` or a dense-attn seam row. The
wave lands under `SPEC-DFLASH2` because the *policy* half of the change is the
speculative-decode mirror (`supports_spec_as_decode`, the reorder threshold, the
uniform-verify classification the runner already computes for this row's
engines), the measured gap is this row's #1574 workload, and the only engines
that can ever produce the classified batch are the speculative ones this row and
its siblings own. The vt kernel half is deliberately small and additive — a new
launcher beside the shipped one — and does not reshape the dense-attn seam, so a
seam row would own a one-line consumer and none of the policy.

## The gap, measured (#1857, 2026-08-24, dgx GB10, `main` 9a0c0f3d1)

- 10-prompt real text: SGLang 27.60 tok/s vs ours 25.07 at EQUAL acceptance;
  the whole edge is per-step (~109 ms vs ~122 ms). Both engines pay the same
  ~20 ms draft.
- K-ladder: our verify grows +9 ms from q=2 to q=9 (step 113.0 → 121.8).
- Mechanism (ours): a q=9 verify has `num_tokens > num_reqs`, so
  `is_prefill` routes it to the prefill flash ladder
  (`src/vt/cuda/cuda_paged_attn.cu::LaunchPaged`, the
  `is_prefill = num_tokens > num_reqs` predicate), i.e. FA-2 *prefill* varlen
  (`LaunchPrefillFA2Bf16`) over 9 query rows against the full KV with
  `num_splits = 1` — one CTA per (request, head) walks the whole context
  serially. `fa2_decode` hard-requires `num_tokens == num_reqs`, so the
  split-KV decode arm (`LaunchDecodeFA2Bf16`, split-parallel over the context)
  never serves it.
- Mechanism (upstream/SGLang): FlashInfer's `supports_spec_as_decode=True`
  raises the decode-reorder threshold so the uniform-qlen verify STAYS on the
  dedicated decode kernel with a packed draft mask; SGLang's decode log shows
  `cuda graph: True` throughout at ~109 ms/step.

## Upstream anchors

Read at merge `b389ac29465b33f9e9c534df221ea3c129e9793f` (the DFlash2 merge,
this row's pinned reading) and re-read at the parity pin `5559679229`
([upstream-sync.md](../upstream-sync.md)); the threshold policy is byte-identical
at both.

| What | Where |
|---|---|
| The reorder-threshold policy: `1 + (2 if parallel_drafting else 1) * num_speculative_tokens` when the backend supports spec-as-decode | `vllm/v1/attention/backend.py::AttentionMetadataBuilder._init_reorder_batch_threshold` (`:718-736` at the merge; `:657-687` at the pin — identical body) |
| The backend declares support and calls the policy | `vllm/v1/attention/backends/flashinfer.py:852-860` (`supports_spec_as_decode = TRTLLM_GEN or use_dedicated_xqa`) |
| The decode kernel's draft mask for a uniform request | `vllm/v1/attention/backends/flashinfer.py::_make_xqa_draft_block_mask` (`:114-140`): `kv_idx <= q_idx` inside the q rows — intra-step causal, full context visible |
| `parallel_drafting` is set for the block drafters | `vllm/config/speculative.py:1064-1065` (`method in ("dflash", "dspark") → parallel_drafting = True`); local mirror `include/vllm/config/speculative.h::SpeculativeConfig.parallel_drafting` (already landed, SPEC-DSPARK W1) |
| What the FA-2 family itself does with a q>1 verify | `vllm-project/flash-attention @ 2c839c33` `mha_fwd_kvcache`: seqlen_q > 1 runs the SAME split-KV kernel, batched presentation, `is_causal=true` (bottom-right aligned against `seqused_k`), `set_params_splitkv` num-splits heuristic; the `seqlenq_ngroups_swapped` decode optimization applies ONLY at `seqlen_q == 1` |

**The FA-2 base policy at the pin, read as instructed.** At `5559679229`
`vllm/v1/attention/backends/flash_attn.py::FlashAttentionMetadataBuilder` never
calls `_init_reorder_batch_threshold`: the FA backend has no decode/prefill
kernel split at all — every batch, verify included, runs one
`flash_attn_varlen_func` whose `set_params_splitkv` heuristic decides the
split-KV parallelism. The threshold machinery exists for backends (FlashInfer
XQA/TRTLLM-gen) whose decode kernel is a *different* kernel from prefill. Our
tree is such a backend: `LaunchPaged` has a hard prefill/decode lane split. So
the correct mirror is the FlashInfer-shaped one — the threshold classification
— and the kernel that serves the classified batch is exactly what upstream FA-2
itself runs for a q>1 kvcache call: split-KV, batched, bottom-right causal.

## Design

Three pieces, each the smallest mirror of the upstream seam it names.

### 1. The policy (host, `include/vllm/v1/attention/backend.h`)

Two pure functions beside `CommonAttentionMetadata`, mirroring
`backend.py::_init_reorder_batch_threshold`:

- `SpecAsDecodeReorderThreshold(num_speculative_tokens, parallel_drafting)` =
  `max(1, 1 + (parallel_drafting ? 2 : 1) * num_speculative_tokens)`; `1` when
  speculation is off. The formula is the upstream one verbatim — NOT a local
  invention; DFlash/DFlash2/DSpark are `parallel_drafting`, so the threshold is
  `1 + 2*K` (#1857's ask by name).
- `SpecAsDecodeQueryLen(uniform_query_len, num_speculative_tokens,
  parallel_drafting)` → the classified decode query length: `uniform_query_len`
  when `1 < uniform_query_len <= threshold`, else `0`. This is FlashInfer's
  decode classification (a request with `query_len <= reorder_batch_threshold`
  is decode) restricted to the uniform batches this engine's verify produces.

`CommonAttentionMetadata` gains `int uniform_spec_query_len = 0` — the
classified value, `0` on every non-verify step. The metadata carries the
classification exactly as upstream's builder-owned threshold shapes the
metadata split the kernels then trust.

### 2. The classification site (the runner)

`GPUModelRunner::execute_model` already computes the step's verified uniform
query length (`v1::GraphEligibleQueryLen` — uniform by arithmetic AND every
request verifying at exactly `q-1` drafts, ENG-CUDAGRAPH-BREAK W6). One new
line classifies it through the policy and stores it on `attn_meta`; a new
`GraphDispatchStats::spec_as_decode_steps` counter (`NoteSpecAsDecode`) makes
the decision observable from a CPU test, exactly the shape W6 used for #1020's
"SILENTLY". Input population note: `GraphEligibleQueryLen` bounds its answer by
`1 + k`, which sits strictly inside the `1 + 2k` threshold — the threshold is
mirrored at full width anyway because the policy function is the upstream
formula, not a re-derivation from what today's caller happens to produce.

The five `BuildPaddedDecode`/`BuildPaddedDecodeAttn` copies (qwen3_5, qwen3,
qwen3_moe, deepseek_v2, voxtral) rewrite a step as S single-token requests;
each zeroes the copied field. This is belt on top of braces: the vt-side shape
guard (`num_tokens == q * num_reqs` with `q > 1`) is already false for every
pure-decode rewrite (`S == q*S` only at `q == 1`), so a stale field cannot
route.

### 3. The vt routing + the kernel (CUDA)

- `vt::PagedAttentionArgs` gains `int32_t uniform_spec_query_len = 0`. It is a
  ROUTING HINT, not a semantic change: the attention output is defined by
  (query, cache, block_table, seq_lens, query_start_loc, scale, causal, window)
  exactly as before, so every backend that ignores the field (CPU, Metal, ROCm,
  Vulkan) is still correct, and the kAuto/fp8 refusal machinery is NOT extended
  (unlike `kv_cache_dtype`, which changes what the bytes mean).
- New host header `include/vt/paged_attn_route.h` (host-compilable, the
  dispatch-selection seam a CPU test can enter):
  `PagedAttnUniformSpecShape(num_tokens, num_reqs, uniform_spec_query_len)`
  (the `q > 1 && num_tokens == q * num_reqs` consistency guard) and
  `PagedAttnIsPrefill(num_tokens, num_reqs, spec_as_decode)` (the existing
  `num_tokens > num_reqs` split, forced to the decode class when the spec batch
  is admitted).
- `LaunchPaged` (`cuda_paged_attn.cu`): computes
  `fa2_spec_decode = shape-guard && d == 256 && bf16 q/kv/out && causal &&
  no window && block_size % 16 == 0 && block_table row-contiguous &&
  (ratio-4 | ratio-6 | ratio-8) && Fa2SpecDecodeEnabled()`, then
  `is_prefill = PagedAttnIsPrefill(num_tokens, num_reqs, fa2_spec_decode)`.
  The classified batch dispatches to the new launcher; every unclassified or
  inadmissible batch routes byte-identically to today (`fa2_spec_decode` false
  ⇒ `is_prefill` is the old predicate verbatim).
- New launcher `LaunchSpecDecodeFA2Bf16` (`cuda_flash_attn_fa2.cu`), ADDITIVE —
  the shipped q==1 `LaunchDecodeFA2Bf16` body is untouched. It is the exact
  presentation upstream `mha_fwd_kvcache` uses for `seqlen_q > 1`:
  - batched (uniform q makes the packed `[B*q, Hq, D]` query a regular
    `[B, q, Hq, D]` view: `q_batch_stride = q * stride0`), `cu_seqlens_q =
    nullptr`, per-request K length via `seqused_k = seq_lens`;
  - `is_causal = true`, `window_size_right = 0`: the vendored kernel's causal
    mask is bottom-right aligned against `seqused_k`, so verify row `i` sees
    `context + i + 1` keys — `_make_xqa_draft_block_mask`'s intra-step causal
    semantics with no new mask code;
  - NO `seqlenq_ngroups_swapped` (upstream applies the swap only at
    `seqlen_q == 1`; the swap packs head-groups into the M dimension, where a
    causal mask across rows would be wrong);
  - `set_params_splitkv` port: `kBlockN = 64` (d256), `kBlockM = 64`,
    `num_m_blocks = ceil(q/64)`, `ctas_per_split = B * Hq * num_m_blocks`,
    `NumSplitsHeuristic(…, 2*num_sms, num_n_blocks, 128)` + `ApplyNsplitsCap`
    — the split-KV parallelism that is the entire point;
  - split scratch in a NEW `spec_decode` map on `Fa2StreamScratch` (never
    aliases the shipped q==1 arms), keyed by the extended shape including q;
    the capture-status refusal on a scratch miss mirrors the existing arms
    (the eager warm step every graph slot runs before capture allocates it);
  - `run_mha_fwd_splitkv_dispatch<bf16, 256, true>` — the causal d256 split
    instantiation already vendored and reached by the MLA prefill launcher;
  - combine addressing: the split combine derives `(batch, head, row)` from a
    flat index over `b*h*seqlen_q` and writes O through
    `batch*o_batch_stride + head*o_head_stride + row*o_row_stride`
    (`flash_fwd_kernel.h::combine_attn_seqk_parallel`); UNIFORM q makes the
    packed output row spacing regular, so `o_batch_stride = q * stride0`,
    `o_row_stride = stride0`, `o_head_stride = stride1` land every row. This is
    precisely why the packed-PREFILL combine restriction (ragged row spacing)
    does not apply here, and why the admission is uniform-q only.

### Options considered and rejected

- **(a′) widen the shipped group-swap `LaunchDecodeFA2Bf16` to q>1** — rejected:
  the swap presentation packs the head-group dimension into `seqlen_q`, every M
  row is the SAME time position, and the kernel's causal template masks across
  M rows; q>1 under the swap would need the packed draft mask the vendored FA-2
  does not express. Upstream draws the same line: the swap applies only at
  `seqlen_q == 1`.
- **(b) loop the q rows over the q==1 split-KV kernel with growing effective
  length** — rejected: q kernel launches per layer per step where upstream runs
  one; the intra-step rows would serialize; and the per-row `seq_lens` tensors
  (context+1 … context+q) are per-step device uploads the CUDA-graph decode
  path cannot re-bake. Nothing upstream looks like this.
- **(c) extend the decode-opt/GQA CUDA-core kernels** — rejected: the vendored
  FA-2 split-KV kernel already expresses the mask (bottom-right causal) and the
  parallelism (split-KV), is the kernel vLLM itself runs for a q>1 kvcache
  call, and the d256 gate models' decode already lives on it. Extending the
  scalar kernels would be a scratch implementation beside a faithful port.
- **(d) classify inside vt from shape alone** (`num_tokens % num_reqs == 0`) —
  rejected: a ragged batch can alias a uniform product (2 requests at lens 1+3
  ≡ 2×2). Upstream classifies in the metadata builder where per-request query
  lengths are known; the mirror classifies in the runner off
  `GraphEligibleQueryLen`, which already refuses ragged and non-verify shapes,
  and vt keeps only the consistency guard.

### Scope boundaries

- **d256 only** (ratio-4/6/8 — the three shipped FA-2 decode topologies; the
  measured subject `Qwen/Qwen3.8-27B` is ratio-6). The d128 varlen decode lane
  (`fa2_decode_qwen3`) keeps its `num_tokens == num_reqs` gate: its plain arm's
  split-combine `o_batch_stride = o_row_stride` identity holds only at q==1,
  and no measured gap names a d128 verify. Owed below.
- **Muse-Glimmer-30B-DFlash2** (the second DFlash2 target): 32/2 heads at
  d128 with iRoPE local layers — outside every d256 gate, so its verify keeps
  the prefill route unchanged. Owed below.
- GDN snapshot cost is explicitly NOT in scope (#1857).
- The fp8-KV lane (`LaunchPagedFp8Out`) keeps its own `is_prefill` untouched:
  it has no FA-2 decode arm to route to.

## Numerics

**CPU: bit-identical by construction.** No CPU code changes. The CPU reference
(`cpu_paged_attn.cpp`) is a single implementation with no lane split: per
request, `context = seq_lens[r] - query_len`, query row `local` attends keys
`0 … context+local` — already the draft-causal semantics for any q. Every CPU
fixture and token gate is byte-for-byte unchanged.

**CUDA, num_splits == 1: the same kernel instantiation as today's route.** The
prefill lane serves the verify through `run_mha_fwd_splitkv_dispatch<bf16, 256,
true>` at `num_splits = 1`; the new launcher dispatches the same causal d256
template at the same block sizes with the same mask geometry
(`actual_seqlen_q = q`, `actual_seqlen_k = seqused_k[b]`, bottom-right). The
difference is varlen vs batched ADDRESSING of the same rows.

**CUDA, num_splits > 1: the split combine reorders the f32 reduction.** This is
the identical numerics class as the shipped, default-ON `VT_FA2_DECODE_GQA_SWAP`
arm ("non-byte-exact vs the plain arm only when num_splits>1 … a near-tie that
moves TOWARD vLLM's own numerics", `cuda_paged_attn.cu::Fa2DecodeGqaSwapEnabled`)
and of `VT_FA2_NSPLITS_CAP`: the split-KV combine sums a different number of f32
partials. Upstream's own verify (FA backend) runs the same split-KV heuristic,
so the split reduction IS the oracle's numerics, not a divergence from them.
The env kill switch `VT_FA2_SPEC_DECODE=0` restores the prefill route for a
same-binary A/B, and the GPU token gates in `## Owed` are the measured evidence
this section's argument is held to — the wave claims no GPU numerics result
until they run.

## Tests

Red first, entered through production seams, per
[reachability.md](../reachability.md):

1. **Chain classification (RED first)** —
   `tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp` gains a case
   driving the shared DFlash2 CPU runner fixture and asserting
   `GraphDispatchStats::spec_as_decode_steps > 0` with the classified length
   equal to the step's verify width. Committed with the counter present but the
   runner not yet classifying: RED for the intended reason (the verify step is
   not classified spec-as-decode), green after the wiring commit. This is the
   test the reachability mutation re-reds by deleting the runner call site.
2. **Policy boundary** — `tests/vllm/v1/attention/test_spec_as_decode.cpp`:
   threshold `1 + 2K` exact (q = 1+2K classifies, q = 1+2K+1 does not),
   `parallel_drafting=false` gives `1 + K`, `k = 0` and `q <= 1` give 0.
   Mutation: an off-by-one in the threshold formula reds the boundary cases.
3. **vt routing seam** — `tests/vt/test_paged_attn_route.cpp`: the shape guard
   (q·B product, stale-field refusal, q=1 refusal) and the is-prefill polarity
   (classified ⇒ decode class; unclassified ⇒ the old `num_tokens > num_reqs`
   verbatim). Mutations: guard `>` → `>=` and dropping the spec conjunct each
   red a case.
4. **Mask semantics at the verify shape (CPU always, CUDA when present)** —
   `tests/vt/test_ops_paged_attn.cpp` gains a uniform-q spec-decode case: B=2
   requests, q=3 uniform, context per request, with the classified args field
   set, validated against the composed per-token causal reference, including a
   poisoned key at each row's first invisible position (context+local+1) whose
   admission would visibly move the output — one extra admitted position reds
   the case. On CUDA the same case runs the REAL routed lane and additionally
   asserts through the existing `Fa2Decode*ForTesting` counters that a decode
   launch (not a prefill one) served it; `HasCuda`-guarded, dgx-pending like
   every CUDA parity case in that file.
5. **No-regression floor** — the full CPU suite; the q==1 decode arms and every
   non-classified batch are dispatch-identical by construction (the routing
   term is false), and the existing op tests hold it.

## Gates

| Gate | Command | Result |
|---|---|---|
| Focused red→green | `ctest -R 'dflash2_runner_reach\|spec_as_decode\|paged_attn_route\|ops_paged_attn'` | red captured, then green, this box |
| Full CPU gate | `scripts/agent-preflight.sh` + full `ctest` | green, this box |
| CUDA compile | CUDA-toolkit build of `vllm` with `VLLM_CPP_FLASH_ATTN=ON` | **PENDING** — no `nvcc` on this box; owed operator-run |
| GPU token identity | the #1574 DFlash2 gate battery (acceptance + token identity vs spec-off) on `dgx:gpu0`, plus `VT_FA2_SPEC_DECODE=0` same-binary A/B | **PENDING** — owed operator-run |
| GPU step-time delta | the #1857 K-ladder rerun (the −8-9 ms/step claim) | **PENDING** — owed operator-run; this wave claims NO speed number |

## Owed

- **The GPU step-time delta** (#1857's −8-9 ms/step, SGLang-parity claim):
  operator-run on `dgx:gpu0` inside an `rc` lease, K-ladder + the 10-prompt
  battery. This wave records no speed number.
- **The GPU correctness gate**: the DFlash2 token-identity/acceptance battery
  with the new route ON and the `VT_FA2_SPEC_DECODE=0` A/B; also the first
  CUDA compile of the two new/edited `.cu` regions (no `nvcc` here).
- **The d128 uniform-spec lane** (`fa2_decode_qwen3` plain varlen at q>1,
  needs the regular-row-spacing combine strides): unowned until a measured gap
  names a d128 verify; the routing term keeps d128 verifies on today's prefill
  route.
- **Muse-Glimmer-30B-DFlash2's verify** (d128, 32/2, iRoPE): same condition.

## Evidence (this box, CPU)

- **RED** (commit `feat` W10-red, before the wiring):
  `./build/tests/test_dflash2_runner_reach -tc='*W10*'` →
  `CHECK( st.spec_as_decode_steps == st.uniform_spec_steps ) is NOT correct!
  values: CHECK( 0 == 7 )` — 7 uniform verify steps ran on the DFlash2 CPU
  fixture, none classified. Exactly the intended reason.
- **GREEN** (after the wiring): the same command → 1 passed, 8/8 assertions;
  full `test_dflash2_runner_reach` 6/6 · 126/126; `test_spec_as_decode` 6/6 ·
  21/21; `test_paged_attn_route` 4/4 · 14/14; `test_ops_paged_attn` 15/15 ·
  1838/1838 (the CUDA cases skip on this CPU-only box, dgx-pending);
  `test_mtp_depth` 10/10 and `test_cudagraph_dispatch` 7/7 unchanged.
- **Full gate**: clean full build 798/798 targets; `ctest -j 8` 607/608 with
  `test_engine_core_proc` failing under the parallel run and passing serially
  (15/15 · 121/121) — the starve-under-`ctest -j` class verification.md says to
  re-run serially before calling a regression. `scripts/agent-preflight.sh`
  green before edits and `--staged` green before each commit.
- **Slot-copy carry**: `Qwen3_5 SizeSlot::Refresh` copies the metadata
  field-by-field, so the classification is carried explicitly in both Refresh
  bodies — without it the CAPTURED verify would silently re-route onto the
  prefill lane while the eager verify routed decode. CPU tests cannot observe
  this (graphs are CUDA-only); the GPU gate in `## Owed` is the evidence that
  holds it.
- **Mutations** (each applied in place, focused target rebuilt, suite rerun,
  tree restored byte-for-byte via git and rebuilt green after):
  - threshold `1 + f*k` → `f*k` (off-by-one): `test_spec_as_decode` 4 cases /
    8 assertions red;
  - parallel factor `2` → `1`: 2 boundary cases / 5 assertions red. NOTE: the
    first spelling of this mutation (deleting the ternary) FAILED TO BUILD
    (`-Werror=unused-parameter`), and a mutation that does not build reads as
    a pass off the stale binary — it was replaced with the semantic
    `(parallel_drafting ? 1 : 1)` form, which builds and reds;
  - shape guard `> 1` → `>= 1`: `test_paged_attn_route` 1 case red;
  - `PagedAttnIsPrefill` spec conjunct dropped: 2 assertions red;
  - the runner classification call site DELETED (the reachability mutation):
    the W10 chain case red again at `0 == 7` — the test enters through the
    production runner path;
  - CPU causal bound `p` → `p + 1` (one extra admitted position): the
    uniform-spec op case red at 128/192 assertions.

## Stop conditions

- If the GPU token gates show ANY token change on the existing fixtures beyond
  the documented num_splits near-tie class: `NEEDS_DECISION` with the
  divergence measured (the operator holds the box; this box cannot run it).
- If the vendored kernel had failed to express the bottom-right draft mask
  (it does; §Design 3): `NEEDS_DECISION` naming the missing kernel surface
  rather than a partial lane.
