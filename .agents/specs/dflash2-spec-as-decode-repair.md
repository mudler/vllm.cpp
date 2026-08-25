# SPEC-DFLASH2 — W10 repair: the spec-as-decode verify must SELECT bf16 through its own lane's gates ([#1865](https://github.com/mudler/vllm.cpp/issues/1865))

Row: `SPEC-DFLASH2`. Parent wave: [`dflash2-spec-as-decode.md`](dflash2-spec-as-decode.md)
(W10, [#1857](https://github.com/mudler/vllm.cpp/issues/1857), landed as
`f4d4526cb` / PR #1858). Issue:
[#1865](https://github.com/mudler/vllm.cpp/issues/1865).

## The finding: where the runtime chain died

#1865's nsys table (GB10, `main` + #1858, `VT_FA2_SPEC_DECODE` default ON) shows
the q=9 verify attention running `PagedFlashKernel` (261 us x 16/step, 99%
in-graph); `LaunchSpecDecodeFA2Bf16` never appears, and the on/off A/B was
speed-neutral. This spec records the dead link, found by tracing the chain on
this tree plus a probe on the CPU production fixture:

1. **The threading is NOT the dead link.** A scratch probe inside
   `vt::PagedAttention` (the shared wrapper every backend passes through) run
   under `test_dflash2_runner_reach`'s production engine shows all 7 uniform
   verify steps arriving with `uniform_spec_query_len == 4` and the one
   non-verify step arriving with 0. Runner → `CommonAttentionMetadata` →
   `FullAttnBlockPaged` → `PagedAttentionArgs` is live on the eager path, and
   both `SizeSlot::Refresh` bodies carry the field on the captured path
   (`src/vllm/model_executor/models/qwen3_5.cpp:10390,10943`).
2. **The dead conjunct is the bf16-query requirement.** The W10 admission
   (`src/vt/cuda/cuda_paged_attn.cu:2806-2811`) requires
   `std::is_same<TQ, __nv_bfloat16>` + bf16 out. `PagedFlashKernel` is the
   plain CUDA-core flash (`LaunchPrefillFlash`), selected only when the WMMA /
   FA2 lanes decline — for a bf16 KV cache that means **the query was f32**:
   with a bf16 query the ladder would have taken `LaunchPrefillFA2Bf16` or the
   WMMA family instead. So the model-side dtype selection
   (`FullAttnBlockPaged`, `qwen3_5.cpp:5360-5374`) chose `attn_dt = kF32` on
   the verify, and the admission then failed on its dtype conjuncts with every
   counter green.
3. **The profiled deployment's FA2 arm was dark end to end.** The kernel table
   pins it: `PagedFlashKernel` counts 1872 = 16 x 117 — sixteen full-attn
   layers times ALL 117 forwards, **including the one real 82-token prefill**.
   The FA2 prefill lever is default-ON and was validated on GB10 2026-07-10, so
   on any FA2-enabled binary with default env that prefill forward would have
   run `flash_fwd_splitkv` (absent from the whole table). The only conjuncts
   that can produce this are `Fa2PrefillOn()` false (no
   `VLLM_CPP_FLASH_ATTN` — a configure without staged CUTLASS headers prints
   `CUDA FA2 compiled-arch manifest: []` and builds green) or `fa2_platform`
   false (empty manifest, same cause), or `VT_FA2_PREFILL=0` in the run env.
   dgx was reimaged 2026-08-14; a lease-container build that does not stage
   CUTLASS gets exactly this binary, silently.
4. **The code-level defect that let this be silent — and would break the lane
   even on a healthy build under the documented env**: the verify's bf16
   eligibility has NO arm of its own. It rides `fa2_prefill`'s conjuncts
   (`Fa2PrefillOn() && ... && T > num_reqs`), i.e. the PREFILL lever, while
   the CUDA admission reads the SPEC lane's own toggles
   (`Fa2SpecDecodeEnabled() && Fa2Decode*Enabled()`). The two sides consult
   different switches: `VT_FA2_PREFILL=0` (the prefill lever's documented
   same-binary rollback) starves the spec lane's query to f32 and the
   `VT_FA2_SPEC_DECODE` A/B becomes a no-op — the observed neutrality. And
   when the mismatch happens, nothing anywhere says so: the classified batch
   silently falls to the prefill ladder.

## Scope

1. **Model-side spec-as-decode eligibility arm** (`qwen3_5.cpp`,
   `FullAttnBlockPaged`): a CLASSIFIED uniform verify
   (`vt::PagedAttnUniformSpecShape` over `meta.uniform_spec_query_len`) on a
   d256 decode-eligible topology selects bf16 through the SPEC lane's own
   gates — `Fa2SpecDecodeOn()` (new model-side wrapper, MUST match
   `cuda_paged_attn.cu::Fa2SpecDecodeEnabled`) plus the same
   ratio/preamble/platform/kv/block conjuncts the pure-decode arm uses —
   independent of `Fa2PrefillOn()`. The predicate is EXTRACTED into a
   host-testable seam (`ClassifyDenseFa2`, `qwen3_5_internal.h`), the same
   move W10 made for the vt split (`include/vt/paged_attn_route.h`), because
   the CPU tier can never make `fa2_platform` true through the runner and a
   dgx-pending `HasCuda` case is exactly the gate shape that let #1865 happen.
2. **Route visibility, CPU-gateable** (`vt::PagedAttention`, `src/vt/ops.cpp`):
   a process counter of shape-consistent CLASSIFIED arrivals at the shared
   dispatch wrapper. The production fixture asserts verify-steps x
   full-attn-layers arrivals, which makes the W10 review's declared dead
   mutation — deleting `pa_args.uniform_spec_query_len = meta...` — RED on a
   CPU box for the first time.
3. **Runtime narration, CUDA** (`cuda_paged_attn.cu::LaunchPaged`): a
   classified batch that the dispatch cannot serve prints ONE stderr line per
   process naming the failed conjunct group (compiled-out / dtype / topology /
   toggle). This is the `verification.md` "instrument narrates its own
   comparison" repair: build6's condition would have named itself in the
   server log at the first verify step, no nsys needed.

Explicitly OUT of scope: the d128 arms (`dense_attn_block.h` keeps its q==1
gates — W10 `## Owed` unchanged), any kernel change, any redesign of the W10
admission, and any speed claim.

## Upstream anchors

Unchanged from the parent wave: `backend.py:718-736` @ `b389ac2946` (reorder
threshold), `flashinfer.py:852-860` (`supports_spec_as_decode`),
`_make_xqa_draft_block_mask` :114-140. This wave adds no policy — upstream
selects ONE model dtype for attention regardless of which lane serves a step
(vLLM's whole attention path is bf16), so the model-side arm restores the
mirror rather than adding a heuristic: a verify the decode lane admits must be
presented in the lane's dtype.

## Design

`ClassifyDenseFa2(DenseFa2Eligibility)` in
`src/vllm/model_executor/models/qwen3_5_internal.h`, pure and host-compilable:

- `kPrefill`: today's `fa2_prefill` conjuncts verbatim.
- `kDecode`: today's `fa2_decode` conjuncts verbatim.
- `kSpecVerify` (NEW): `PagedAttnUniformSpecShape(num_tokens, num_reqs,
  uniform_spec_query_len)` AND a decode-eligible ratio arm (r4/r6/r8, each
  with its own toggle) AND `spec_decode_on` AND preamble+cos/sin AND
  `fa2_platform` AND bf16 KV AND `block_size % 16 == 0` AND d==256 AND causal
  — the exact mirror of the CUDA admission, minus the dtype conjuncts this
  function exists to satisfy.
- Precedence `kSpecVerify > kPrefill`: mirrors `PagedAttnIsPrefill`'s
  polarity (an admitted verify is DECODE class). All three classes select the
  same `attn_dt = kBF16`, so on a healthy default build (where the verify
  already rode `kPrefill`) the selected dtype — and therefore every kernel and
  byte — is unchanged.

`FullAttnBlockPaged` computes `fa2_attention = ClassifyDenseFa2(...) != kNone`
with the same inputs it read before. No other consumer changes.

## Risks / decisions

- **Behavior change is confined to the mismatch population**: steps where the
  spec lane is willing (`VT_FA2_SPEC_DECODE` on, decode topology on) and the
  prefill arm is not (`VT_FA2_PREFILL=0`, or a prefill-only conjunct false).
  There the verify's query moves f32 → bf16 and the admission can fire —
  which is the lane W10 shipped and gated (`num_splits==1` byte-class,
  `num_splits>1` the documented split-combine near-tie). On CPU
  (`fa2_platform` false) and on every non-classified batch the function is
  the old predicate verbatim. MEASURED, not argued: the shipped classifier
  compiled against a verbatim transcription of the pre-fix inline predicate,
  swept exhaustively over 81920 states (4 topologies x 2 head dims x 10 step
  shapes x 2^10 boolean inputs), differs in 36 — every one of them with
  `prefill_on == false`. The two configurations a real binary is in both
  differ in ZERO: FA-2 compiled out (every toggle wrapper false through its
  `#else`) 2560 states, and the healthy default build with the env unset 2560
  states. Analytically the same result: `PagedAttnUniformSpecShape` requires
  `q >= 2`, so a classified verify has `T = q*R > R`, the old decode disjunct
  (`T == R`) is dead on it, and the old value reduces to `base && prefill_on`.
- **The counter is in-memory state, not a record surface**; reset+read is the
  same shape as `GraphDispatchStats`.
- **The narration prints once per query/KV dtype specialisation**, host-side,
  so it is capture-safe and cannot flood a server log. NOT once per process:
  `LaunchPaged` is `template <typename TQ, typename TKV>`
  (`src/vt/cuda/cuda_paged_attn.cu`), the `std::once_flag` is function-local,
  and `LaunchPagedByKv` instantiates it over {f32, bf16} query x {f32, bf16}
  KV — up to four lines from one process, one per instantiation that a
  classified batch actually reaches. That is a bounded constant, so the
  capture-safety and flood arguments are unchanged; only the count is.

## Tests and gates

- RED first (captured on the pre-fix semantics, commit order carries it): the
  `kSpecVerify` case of the new classifier test — 27B production shape
  (Hq24/Hkv4/d256, bf16 KV, block 16, causal, fused preamble, fa2_platform
  true), q=9 verify, `spec_decode_on` true, `prefill_on` FALSE → must
  classify. Pre-fix the eligibility has no spec arm → `kNone` → red.
- The W10 chain case (`test_dflash2_runner_reach`) additionally asserts the
  `vt::PagedAttention` classified-arrival counter equals
  `uniform_spec_steps` x full-attn layers, through the production runner. The
  W10 review's dead mutation (delete the `pa_args.uniform_spec_query_len`
  threading) now reds this on a CPU box.
- Mutations: (a) the threading deletion at `qwen3_5.cpp` → chain case red;
  (b) the spec arm deleted from `ClassifyDenseFa2` → classifier case red;
  (c) `spec_decode_on` conjunct dropped from the spec arm → toggle case red;
  (d) reachability: the runner classification call site deleted → the
  existing W10 chain case red at `0 == N` (unchanged from W10).
- Token identity: full CPU suite unchanged (the classifier refactor is
  byte-identical semantics; the new arm needs `fa2_platform`, which no CPU
  path has).

## Owed (operator-run; this wave claims none of it)

- The nsys re-profile on `dgx:gpu0` showing `LaunchSpecDecodeFA2Bf16` in the
  kernel table for the q=9 verify (`/usr/local/vcpp/nsysprof2/run3.sh`
  recipe), **on a binary whose configure log shows `FlashAttention-2
  prefill/decode: ENABLED` and a non-empty `CUDA FA2 compiled-arch manifest`**
  — finding 3 above means a profile from an unstaged-CUTLASS build measures
  the fallback, whatever this repair does. The new one-line narration is the
  cheap first check: its presence in the server log names the dead conjunct
  group; its absence plus the kernel in the table is the pass.
- The `VT_FA2_SPEC_DECODE` on/off A/B moving on that binary. W8/W10 taught us
  not to project a number; the delta is owed, not promised.
- The DFlash2 GPU token-identity battery on the engaged lane (the parent
  wave's owed gate, unchanged).
- **No CI job compiles the no-FA2 (`#else`) arm of the edited `.cu` region.**
  Both CUDA jobs configure with `-DVLLM_CPP_CUTLASS_FETCH=ON`
  (`.github/workflows/ci.yml`), and the FA-2 sources plus
  `VLLM_CPP_FLASH_ATTN` are gated on
  `VLLM_CPP_FLASH_ATTN AND VLLM_CPP_CUTLASS_HEADERS AND VT_FA2_ARCHS`
  (`CMakeLists.txt`), so every CI compile of `cuda_paged_attn.cu` takes the
  `#ifdef` arm. `cuda-fat-build` PASSES on this wave's tree over ten archs,
  which IS the first CUDA compile of the region this wave edits — but it is
  the arm this wave does NOT narrate about. The `#else`
  (`const bool fa2_spec_decode = false;`) and the compiled-out-FA2 branch of
  the narration are uncompiled anywhere. Owed: one lane that configures a
  no-CUTLASS (FA-2 compiled out) CUDA build and compiles it.
- **A classified verify on an fp8 KV cache gets NO narration.**
  `LaunchPagedByKv` routes `Fp8KVCacheDataType::kFp8E4M3` to
  `LaunchPagedFp8<TQ>` and returns before `LaunchPaged`
  (`src/vt/cuda/cuda_paged_attn.cu`), so the once-per-specialisation stderr
  line this wave adds never runs on that path. An fp8-KV spec verify that the
  dispatch cannot serve is therefore exactly as silent as #1865 was. Not
  implemented here; owed as its own scoped change.

## Now

`ACTIVE` — W10 repair implementation in flight on `row/SPEC-DFLASH2-w10-repair`.

## Stop conditions

- If the GPU re-profile still shows the fallback with the narration silent on
  a manifest-verified FA2 build: the finding is wrong somewhere upstream of
  the dispatch — `NEEDS_DECISION` with the narration output attached, no
  further guessing in this wave.
- If any CPU token fixture moves: stop; the refactor was not byte-identical.
