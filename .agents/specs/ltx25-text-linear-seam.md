# LTX25-TEXT-LINEAR-SEAM: route the caption projection through `vt::MatmulBT`

**Issue:** [#1208](https://github.com/mudler/vllm.cpp/issues/1208).
**Kind:** defect fix in product code, plus the attribution measurement the issue's
own comments ask for before the fix is believed.
**Filed by:** [`ltx25-decode-speed.md`](ltx25-decode-speed.md)'s campaign; this row
takes #1208 out of that spec's orbit and owns it.

## Now

`Linear` in `src/vllm/model_executor/models/ltx2_text_encoder.cpp` is a scalar,
single-threaded triple loop that widens **both** operands to `double` per
multiply, so it cannot reach an f32 FMA and no thread but the caller's ever
enters it. It is the caption projection of every LTX-2.5 pipeline kind.

## 0. Attribution first, because the issue's own record retracted it

The second comment on [#1208](https://github.com/mudler/vllm.cpp/issues/1208)
withdraws the attribution the first comment implied. It establishes a **95%
single-threaded fraction** over a 6482.6 s GB10 run and two single-core stretches
of **1740 s and 1738 s, 0.11% apart**, and then argues that an equal-cost pair
does *not* fit this function, because its two `project()` calls have a 2:1 size
ratio (4096 vs 2048 `out_features`) and should split ~2:1.

**The 2:1 objection is correct and does not apply, because the pair is not the
two projections.** Four facts settle it, and each is read out of the tree rather
than inferred from the trace:

1. **The text tower runs on a hard-coded CPU queue, whichever device was asked
   for.** `src/vllm/multimodal/ltx2_video.cpp:2085` and `:2799` and `:4479` each
   construct `vt::Queue text_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}`
   immediately before calling `Ltx2EncodePromptToConditioning`. `--device cuda`
   does not move this path.
2. **It is called twice on a guided render** — once for the prompt
   (`ltx2_video.cpp:2086`) and once for the negative prompt (`:2800`), the
   unconditional branch the trace's `cfg_scale` 3.0 forces. Two conditioning
   passes of *identical* geometry is exactly the observed pair, and each pass
   contains *both* projections, so the 2:1 ratio lives **inside** each stretch and
   cannot split them.
3. **The projection's row count is a constant, not a prompt length.**
   `Ltx2EncodePromptToConditioning` sets `out.seq = max_length` and
   `kLtx2GemmaTokenizerMaxLength = 1024`
   (`include/vllm/model_executor/models/ltx2_text_encoder.h:423`, pinned to
   upstream `gemma_assets.py:162` and to `diffusers`
   `pipeline_ltx2.py:304`), and `states.seq` is that padded width, not
   `num_valid`. So `rows` is **1024 on every render**, and the comment's
   0.34–5.4 GMAC/s band — which spanned B=1,T=512 to B=2,T=4096 — collapses to
   one number.
4. **The Gemma-4 tower cannot be the single-core stretch.** It runs through
   `Gemma4Model::ForwardHiddenStates` on vt ops, and the CPU threadpool defaults
   to `std::thread::hardware_concurrency()`
   (`src/vt/cpu/cpu_threadpool.h:36`, `cpu_threadpool.cpp:117`), so tower time is
   *multi*-core time on a 20-core box. The stretches read 101%.

With `rows = 1024` and the comment's own per-row figure of
`(4096 + 2048) x 188160 = 1,156,055,040` MACs, one conditioning pass is
**1.1838e12 MACs**, and 1738 s of it inverts to **0.681 GMAC/s** — one core,
scalar, both operands widened to `double`. That is the number this row must
reproduce locally before it claims the attribution. §4 records what was measured.

The other single-threaded work in the same function is bounded and small:
`Ltx2StackHiddenStates`, the V2 norm and the rescale are each one pass over
`B*T*D*L = 1.926e8` elements, i.e. ~2e8 operations against the projection's
1.18e12 — under 0.1% each.

**Not claimed here.** The trace's *third* stretch (2589 s+, RSS flat at 31 GiB
where the paired two climb 32→41 GiB) has a different signature and this row does
not attribute it. [#1202](https://github.com/mudler/vllm.cpp/issues/1202) and
[#1210](https://github.com/mudler/vllm.cpp/issues/1210) are the same defect family
and are explicitly out of scope.

## 1. Scope

In scope:

- `Linear` in `src/vllm/model_executor/models/ltx2_text_encoder.cpp` routes
  through `vt::MatmulBT`, mirroring the sibling `Linear` the LTX-2.5 DiT already
  uses at `src/vllm/model_executor/models/ltx2.cpp:29-48`.
- The accumulator decision (§3) and the file-header note that currently argues
  the opposite.
- One red-first test that fails on the accumulator width and passes only when the
  projection *is* the seam.
- The issue-index row for [#1208](https://github.com/mudler/vllm.cpp/issues/1208).

Out of scope, and deliberately left:

- [#1202](https://github.com/mudler/vllm.cpp/issues/1202)
  (`Ltx2FuseLoraIntoTensor`) and
  [#1210](https://github.com/mudler/vllm.cpp/issues/1210) (the two-stage rebind).
  Same family, different rows. Neither becomes trivially fixable from this change:
  `Ltx2FuseLoraIntoTensor` is a rank-r outer-product accumulate into an existing
  tensor, not a `[M,K]x[N,K]` GEMM, so it needs its own seam decision.
- The `std::vector<float> scaled(normed.size())` copy in `project()` — a 771 MB
  transient at the shipped shape, paid twice. It is one linear pass and under 0.1%
  of the cost §0 attributes, so removing it here would be unmeasurable and would
  widen the diff.
- The third single-core stretch of the trace.
- Moving the text tower off its hard-coded CPU queue. §0 fact 1 is a finding, not
  this row's work; it is filed under `## Owed`.

## 2. Upstream anchors

| Ours | Upstream |
|---|---|
| `Linear`, `ltx2_text_encoder.cpp` | `torch.nn.functional.linear`, via `text_encoders/gemma/feature_extractor.py:93-129` @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |
| weight layout `[out_features, in_features]` | `torch.nn.Linear.weight`, `encoder_configurator.py:187,206-208` |
| the goldens this is goldens-gated against | `scripts/gen-ltx2-text-goldens.py`, which imports and executes the upstream modules at reduced dimensions and emits every tensor through `.to(torch.float32)` (`:308-309`) |

The local seam is `vt::MatmulBT` (`include/vt/ops.h:1283`), documented there as
"`out[M,N] = a[M,K] @ b^T` with `b` `[N,K]` row-major — the torch Linear weight
orientation", with f32 accumulation. The CPU provider is
`MatmulBTKernel`/`MatmulChunked` (`src/vt/cpu/cpu_ops.cpp:299` and `:212`), a 1:1 port of
ggml's `ggml_compute_forward_mul_mat` chunking over the vt threadpool, with the
per-output K reduction strictly sequential and f32
(`MatmulOneChunk`, `cpu_ops.cpp:109-110`, whose own comment at `:105-106` says it keeps "16 independent" accumulators "vectorized ACROSS OUTPUT COLUMNS so no
reduction is reassociated").

## 3. The accumulator, decided rather than changed silently

Today's `double` is deliberate — the file header argues for it — and this row
**replaces it with f32**, which is what routing through the seam gives. The
reasons, in order of weight:

1. **It is the mirror.** `F.linear` on f32 inputs accumulates in f32. The comment
   directly above this function cites `F.linear` as its reference, and the
   goldens beside it were produced by executing that very module in
   `torch.float32`. AGENTS.md's polarity rule reads on the *stream* dtype, but the
   accumulator is still a mirror question, and the mirror answer is f32.
2. **The seam is binding and can represent the behaviour.** `vt::MatmulBT` gives
   f32 accumulation, which is upstream's accumulation *width*. What differs from
   upstream is the reduction *order* — sequential here, blocked upstream — and
   `include/vt/ops.h:1279-1281` already states that this op is not bit-identical to a
   differently-ordered GEMM. Every other f32 projection in the tree, LTX-2.5's own
   DiT included (`ltx2.cpp:40`), already accepts exactly that.
3. **The cost is measured, not assumed** (§4.2). f64 is strictly closer to exact;
   the question is whether the difference is observable where it lands.

The header's counter-argument — "at the shipped 188160-wide projection a naive f32
accumulation would be materially worse than torch's blocked GEMM, and a double
accumulator lands strictly closer to it" — is **true and not decisive**. It is
true that f64 is nearer the exact sum. It is not decisive because "closer to
exact" is not the same as "closer to what upstream computes", upstream's own
blocked f32 sum carries error of the same order, and no gate in this tree can see
either difference: the goldens run at `flat = 24`, where both accumulators agree
to well under the 1e-5 band. The header is rewritten to say this rather than
deleted.

## 4. Measurements this row must produce

1. **Attribution.** The current loop's achieved MAC rate on one core, at the
   shipped `1024 x 188160 -> 4096` and `-> 2048` shape, from the production
   `Ltx2TextFeatureExtractorForward`. Compared against the 0.681 GMAC/s that
   §0 derives from the GB10 trace. This is the number that decides whether the
   fix belongs here.
2. **Accumulator error at the shipped width.** f32 sequential vs an exact
   reference, at `K = 188160`, at the magnitudes this path actually carries.
3. **Same-binary A/B.** Before and after, same shape, same box, recipe recorded.
4. **The goldens' `max|diff|` before and after**, from the suite's own `MESSAGE`
   lines, so the change is shown not to move them rather than asserted not to.

## 5. Tests

- **Red-first, the seam identity.** A case that drives the production
  `Ltx2TextFeatureExtractorForward` at a width where an f32 and an f64
  accumulation provably differ, and requires the projection to equal
  `vt::MatmulBT` + bias **byte for byte**. It carries its own discrimination
  proof: the same case computes an f64-accumulating reference and asserts it is
  *further* than a stated floor, so a vacuous pass is impossible and the test is
  shown to be able to tell the two implementations apart.
- **Unchanged:** every existing case in
  `tests/vllm/models/test_ltx2_text_encoder.cpp`, including the two goldens cases
  that hold the projection to upstream at 1e-5 and the three that enter through
  the production `Ltx2EncodePromptToConditioning`. No tolerance is widened.

The seam-identity case is a consistency gate and is stated as one; correctness
against upstream stays with the goldens. Both are required, and neither
substitutes for the other.

## 6. Risks

| Risk | Handling |
|---|---|
| f32 accumulation is materially worse at `K = 188160` | measured (§4.2) rather than argued, and reported against the bf16 ulp of the value it produces |
| the goldens move | they are at `flat = 24`; measured before and after |
| `vt::MatmulBT` refuses a degenerate shape | `rows == 0` and `out_features == 0` return early, as the loop did |
| the CPU backend is not registered in some consumer | it is unconditionally registered; the text tower is host-only by construction (`Ltx2TextEncoderWeights` holds `std::vector<float>`) |

## 7. Gates

1. `ctest -R test_ltx2_text_encoder` green, with the case count asserted to have
   moved and `Status:` grepped, not only `assertions:`.
2. The full LTX-2.5 suite: `test_ltx2`, `test_ltx2_pipeline`, `test_ltx2_loader`,
   `test_ltx2_lora`, `test_ltx2_device`, `test_ltx2_dfr`, `test_ltx2_retake`,
   `test_ltx2_tiling`, `test_ltx2_vae`.
3. Full `ctest`, with the pass count compared against the same tree before the
   change.
4. `scripts/agent-preflight.sh --staged`.
5. The §4 measurements, recorded in `## Outcome` with their recipe.

## 8. Stop conditions

- **If the local measurement does not reproduce a rate consistent with 0.681
  GMAC/s**, stop and report that the attribution fails, per the dispatch. The
  defect is still real on its own terms, but the render narrative would not be
  its justification.
- If the goldens move at all, stop and decide the tolerance here rather than in
  the assertion.

## Owed

| Issue | Lever | State |
|---|---|---|
| [#1208](https://github.com/mudler/vllm.cpp/issues/1208) | this row | fixed by this row |

* **The text tower runs on a hard-coded CPU queue** at
  `src/vllm/multimodal/ltx2_video.cpp:2085`, `:2799` and `:4479`, so `--device
  cuda` leaves a 12B Gemma-4 prefill on the host. §0 fact 1 established it while
  attributing this row's cost and this row does not fix it. It is a separate
  capability decision with its own oracle question, and it needs its own issue.
* **The third single-core stretch** of the #1208 trace (2589 s+, RSS flat at
  31 GiB) is unattributed. This row deliberately does not chase it.
* **No shipped-width golden.** The projection's upstream oracle exists only at
  `flat = 24`, because `scripts/gen-ltx2-text-goldens.py` needs the Lightricks
  checkout to regenerate and the shipped width would emit a 3 GB fixture. The
  accumulator claim at `K = 188160` therefore rests on an exactness reference
  computed here, not on upstream.
