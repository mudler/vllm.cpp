# LTX25-TEXT-LINEAR-SEAM: route the caption projection through `vt::MatmulBT`

**Issue:** [#1208](https://github.com/mudler/vllm.cpp/issues/1208).
**Kind:** defect fix in product code, plus the attribution measurement the issue's
own comments ask for before the fix is believed.
**Filed by:** [`ltx25-decode-speed.md`](ltx25-decode-speed.md)'s campaign; this row
takes #1208 out of that spec's orbit and owns it.

## Now

**DONE.** `Linear` in
`src/vllm/model_executor/models/ltx2_text_encoder.cpp` is `vt::MatmulBT` plus a
bias add, and it accumulates in f32. At the shipped geometry one conditioning
pass went from **671.777 s of one core to 78.421 s on about fourteen (8.57x)**,
and a guided render pays that pass twice.

The attribution it was dispatched to settle is **bounded, not closed**: the
projection is 39% to 100% of each of the trace's single-core stretches, because
the GB10-to-x86 per-core ratio could not be measured here. `## Outcome` §1 has
the bound and its argument, and `## Owed` carries the one lease that would close
it.

What it replaced: a scalar, single-threaded triple loop that widened **both**
operands to `double` per multiply, so it could not reach an f32 FMA and no
thread but the caller's ever entered it. It is the caption projection of every
LTX-2.5 pipeline kind.

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
reproduce locally before it claims the attribution. **§1 of `## Outcome` does
reproduce it to within a per-core ratio it could not measure, and therefore
attributes between 39% and 100% of each stretch rather than all of it.** The
residual is named there and carried under `## Owed`.

The other single-threaded work in the same function is bounded and small:
`Ltx2StackHiddenStates`, the V2 norm and the rescale are each one pass over
`B*T*D*L = 1.926e8` elements against the projection's 1.18e12. `## Outcome` §1
**measures** them rather than leaving that as arithmetic — 1.861-2.066 s in
total, 0.28% of the pass — which is what refutes them as the residual.

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
- Moving [#1208](https://github.com/mudler/vllm.cpp/issues/1208) out of
  [`ltx25-decode-speed.md`](ltx25-decode-speed.md)'s `## Owed` and into this
  row's, following that spec's own
  [#1008](https://github.com/mudler/vllm.cpp/issues/1008) precedent. **No
  issue-index row is added:** `839d24313` already appended one for #1208 while
  this row was in flight, and the file is append-only, so a second row would be
  the duplicate its own checker refuses. GitHub holds the open/closed state, so
  the landing costs no edit there.

Out of scope, and deliberately left:

- [#1202](https://github.com/mudler/vllm.cpp/issues/1202)
  (`Ltx2FuseLoraIntoTensor`) and
  [#1210](https://github.com/mudler/vllm.cpp/issues/1210) (the two-stage rebind).
  Same family, different rows. Neither becomes trivially fixable from this change:
  `Ltx2FuseLoraIntoTensor` is a rank-r outer-product accumulate into an existing
  tensor, not a `[M,K]x[N,K]` GEMM, so it needs its own seam decision.
- The `std::vector<float> scaled(normed.size())` copy in `project()` — a 771 MB
  transient at the shipped shape, paid twice. This change **keeps it unchanged on
  both arms**, and `## Outcome` §1 measures it rather than dismissing it: both
  copies together are 0.390-0.468 s, i.e. 0.06% of the before arm and 0.5-0.6% of
  the after arm.
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
| `vt::MatmulBT` refuses a degenerate shape | `rows == 0` and `out_features == 0` return the empty `out` early. A zero `in_features` skips only the GEMM and still runs the bias add, which is what the old loop produced (accumulator seeded with the bias, inner loop skipped) — behaviour-preserving rather than merely equivalent where the tests look. Unreachable through the extractor either way |
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
  **NOT TRIGGERED, and the reason is stated rather than assumed.** The local rate
  is 1.7622 GMAC/s, which is consistent with 0.681 for any per-core ratio in
  [1.0, 2.587], and `## Outcome` §1 shows both ends of that interval are closed
  by argument. The condition asked whether the projection could be the cost; the
  answer is that it is 39-100% of it, so the fix belongs here and the residual is
  carried as owed rather than absorbed.
- If the goldens move at all, stop and decide the tolerance here rather than in
  the assertion.
  **TRIGGERED, and decided here.** Two of twelve moved, both on the V1 arm:
  5.96e-08 -> 1.19e-07 (left) and 5.96e-08 -> 7.45e-08 (right), against the
  file's 1e-5 bound, i.e. 84x and 134x inside it. `## Outcome` §4 records why the
  V1 arm is the one that moves. **No tolerance was changed**, and the decision is
  to accept the movement because it is f32 round-off on a reduction whose width
  now matches the oracle.

## Outcome

**Headline: the caption projection is a fixed 1.1838e12-MAC serial cost paid
twice per guided render, it takes 671.777 s of one core at the shipped geometry
and 78.421 s through the seam (8.57x), and it accounts for between 39% and 100%
of each of the trace's two single-core stretches — a range, because the one
quantity that would close it is a GB10 per-core rate this row could not
measure.**

Measured 2026-08-18 on the x86 development box, not on GB10. Recipe, so a
rerun is possible rather than described:

* Box: 20-core AMD Ryzen 9 9950X3D under KVM, AVX-512 present, 84 GiB RAM,
  Ubuntu 24.04, GCC 13.3.0. **Not idle**: other agents build concurrently, and
  `uptime` read load average 4.1 to 10.1 across the runs. Every number below
  carries that.
* Build: `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_BUILD_TESTS=ON
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_METAL=OFF
  -DVLLM_CPP_TENSTORRENT=OFF`, i.e. the CPU tier, `-O3 -DNDEBUG
  -ffp-contract=off`.
* Harness: a scratch `main` that calls the production
  `vllm::Ltx2TextFeatureExtractorForward` at the shipped geometry — gemma hidden
  3840, 49 states, `flat = 188160`, video 4096, audio 2048, V2 variant, every
  position valid — and times that one call. The two arms are the same harness
  source and the same compile flags against `libvllm.a` built from the two
  commits. No GPU and therefore no `rc` lease is involved.

### 1. Attribution — bounded, not a point claim

**Rate, measured.** The before arm runs at **1.86, 1.82 and 1.84 GMAC/s** at rows
8, 16 and 32 (4.964 s, 10.152 s, 20.113 s — linear to 1%, as a loop with no
blocking must be), and **directly at the shipped `rows = 1024` it takes 671.777 s
at 1.7622 GMAC/s and 99% of one CPU**. The small-rows extrapolation predicted
635-650 s, so the 11-minute run reads 4% slower than its own short legs; the box
carried load average ~8 throughout, which is the direction contention pushes and
is why both are stated. This wall is the whole
`Ltx2TextFeatureExtractorForward` call — stack, V2 norm, rescale and both
projections — so the cheap passes are already inside it and cannot be a hidden
residual.

**The one quantity this row could NOT measure is `R`, the ratio of this box's
per-core rate to GB10's on this loop.** Everything about the GB10 wall turns on
it, so it is stated as a bound rather than assumed:

| `R` | projection's share of one 1738 s stretch | residual |
|---:|---:|---:|
| 1.0 | 671.8 s (39%) | 1066 s |
| 1.5 | 1007.7 s (58%) | 730 s |
| 2.0 | 1343.6 s (77%) | **394 s** |
| **2.587** | **1738 s (100%)** | **0** |
| > 2.587 | excluded — see below | — |

**Both ends of that range are closed by argument, and neither by assumption.**
`R >= 1` because a GB10 Arm core does not out-run a 5.7 GHz-class Zen 5 on a
scalar loop that widens both operands to `double`. `R <= 2.587` because the
projection runs *inside* the stretch and cannot take longer than it. So **the
caption projection accounts for between 39% and 100% of each single-core stretch,
and the residual is between 1066 s and 0.** The coordinator's ~400 s figure is
exactly the `R = 2.0` row, and it is a live possibility this row does not
exclude.

**One residual hypothesis was offered and this row REFUTED it by measurement.**
The candidate was the surrounding single-threaded buffer work — the stack, the V2
norm, and the full-size `scaled` copy that `project()` builds once per
projection. At the shipped geometry, timed directly through the same exported
entry points (`bench2`, two runs):

| single-threaded pass | wall |
|---|---:|
| `Ltx2StackHiddenStates` | 0.839 / 0.910 s |
| `Ltx2NormAndConcatPerTokenRms` | 0.632 / 0.688 s |
| the `scaled` copy, x2 (once per `project()`) | 0.390 / 0.468 s |
| **total** | **1.861 / 2.066 s** |

**That is 0.28% of the 671.777 s pass; the two GEMMs are 99.7% of it.** Even a
core ten times slower than this one would put the buffer work at ~20 s, not at
hundreds. It cannot be a several-hundred-second residual, and it is *inside* the
671.777 s in any case rather than beside it.

**A double-count to avoid, because it changes the conclusion.** 671.777 s is
**one whole conditioning pass** — `Ltx2TextFeatureExtractorForward` called once
with `video_out_features = 4096` and `audio_out_features = 2048`, so **both**
`project()` calls, the stack, the norm and both `scaled` copies are already
inside it (`per_row = (4096 + 2048) * 188160`). Doubling it to 1342 s "for the
two `project()` calls" counts the pass twice and would make the projection look
like 77% of a stretch on a box-equality assumption that would actually make it
39%.

**Where a residual would live, if `R` is near 1.** Named so the next row has
candidates rather than a mystery, and claimed for none of them: the U8/NVFP4
caption-projection weights being unpacked to the f32 4.6 GB (this row did not
establish *when* that happens, and it is the only candidate of the right
magnitude); the caller's 49 x 771 MB padded hidden-state buffers, which are
`assign` + `memcpy` over ~1.5 GB and so are also seconds, not minutes; and the
tower's own serial host glue between its threaded GEMMs, which a 1 Hz sampler
taking a max would still read near 101% if the multi-core bursts are short.

**What is settled regardless of `R`.** The pair of stretches is the two
text-conditioning passes, not the two projections (§0 facts 1 and 2); the row
count is the constant 1024 and not a prompt length (fact 3), which is what
collapses the issue comment's 0.34-5.4 GMAC/s band to one number; the Gemma-4
tower cannot be a 101% stretch at all (fact 4); and the projection is a
fixed 1.1838e12-MAC serial cost paid twice per guided render whatever else
shares the stretch with it.

Two things corroborate the high end of the `R` range without being circular:

* [#1202](https://github.com/mudler/vllm.cpp/issues/1202) derived **~0.53
  GFLOP/s** for a *different* scalar host loop on GB10, against the 0.681 GMAC/s
  this row would need. Same order — but the two loops differ in character
  (#1202's converts bf16 per multiply and strides its inner operand by `cols`),
  and #1202's own figure is inferred from RSS growth rather than timed, so it
  supports the order of magnitude and not the ratio.
* **The RSS delta the trace offers as its own discriminator matches.** The paired
  stretches climb ~8.5 GiB where the third stays flat. This function's footprint
  at `rows = 1024` is 4.6 GB of f32 caption-projection weights plus three
  `B*T*flat` transients (`stacked`, `normed`, `scaled`) at 771 MB each, plus the
  771 MB of padded hidden states its caller builds — measured here as **7.79 GB
  peak RSS**. A matching magnitude, and not a matching cause.

**Owed, and it is one command:** run this row's harness, or the shipped binary
with a phase marker, on `dgx:gpu0` under an `rc` lease. That measures `R`
directly and closes the table above to a single row.

### 2. The A/B

| arm | rows | wall | rate | CPU | peak RSS |
|---|---:|---:|---:|---:|---:|
| before, scalar `double` | 8 / 16 / 32 | 4.964 / 10.152 / 20.113 s | 1.86 / 1.82 / 1.84 GMAC/s | one core | 4.54 GB @ 8 |
| before, scalar `double` | **1024** | **671.777 s** | **1.7622 GMAC/s** | **99%** | 7.56 GB |
| after, `vt::MatmulBT` | 32 | 2.320 s | 15.945 GMAC/s | many | |
| after, `vt::MatmulBT` | 256 | 20.342 s | 14.549 GMAC/s | 1431% of 2000% | 5.52 GB |
| after, `vt::MatmulBT` | **1024** | **78.421 s** | **15.095 GMAC/s** | many | 7.79 GB |

At the shipped `rows = 1024`, both arms measured directly on the same box with
the same harness: **671.777 s -> 78.421 s, 8.57x**. The qualitative change is the
one the trace was about: **one core of twenty becomes about fourteen**.

**No GB10 wall is claimed from this.** §1 shows why: the conversion needs `R`,
which is bounded and not measured, so the saving on GB10 is bounded too. At the
`R = 2.587` end the projection is the whole stretch and 8.57x takes it from
1738 s to ~203 s, twice per render; at `R = 1.0` it is 39% of the stretch and
the saving is ~590 s per pass. Either way the saving is real and its size is
owed to a GB10 run, not derivable here.

### 3. What is NOT closed, stated rather than omitted

* **No GB10 number, and therefore no closed attribution.** Everything above is
  x86. `R` is bounded to [1.0, 2.587] by §1 and measured nowhere, so between
  0 s and 1066 s of each single-core stretch is **still unattributed** and this
  row says so rather than rounding it into the projection. The end-to-end
  LTX-2.5 render axis stays `PENDING` in `docs/BENCHMARKS.md`.
* **The after arm is memory-bound, and this row does not declare a ceiling.**
  Per busy core it reads ~1.0 GMAC/s against the before arm's 1.84 on one core:
  the seam wins on parallelism, not on per-core throughput. The arithmetic says
  why. `MatmulOneChunk` tiles 16 output columns by `mr` activation rows and
  reduces over the whole `K = 188160` inside one micro-kernel call, so a weight
  row (752 KB) and an activation row (752 KB) each exceed the 1 MB L2 and are
  streamed once per call — `(mr + 16) * 752 KB` of traffic per `mr * 16 * 188160`
  MACs, about 1.25 bytes per MAC at `mr = 4`, i.e. ~18 GB/s at the measured rate.
  **The named next hypothesis is K-blocking**: split the reduction into
  L2-resident K panels so a weight panel serves many more output columns before
  it is evicted. That belongs to the CPU GEMM row, not here, and this row files
  it rather than chasing it.
* **The third single-core stretch** of the #1208 trace is still unattributed,
  and is a separate question from the residual above.
* **The text tower's hard-coded CPU queue.** Found while attributing this row
  (§0 fact 1), not fixed here, and listed under `## Owed`.

### 4. The accumulator — DECIDED: f32, and which oracle decided it

**The decision is f32, and it is a mirror decision.** Stated with the oracle
that was actually checked, because "upstream does X" is the claim that most
needs its source named:

* **vLLM, the primary reference, does not implement this path at all.** It never
  registers LTX-2.5, so there is nothing here to mirror from it.
* **vLLM-Omni, which does, is `UNPINNED` and `gateable = no`**
  ([`.agents/oracles/vllm-omni.md`](../oracles/vllm-omni.md), owed by #633), so
  it cannot settle a numeric question.
* **The oracle this file is actually gated against is Lightricks `ltx_core`
  itself**, imported and executed by `scripts/gen-ltx2-text-goldens.py` at
  revision `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, which emits every tensor
  through `.to(torch.float32)` (`:308-309`). The module it runs for this
  projection is `torch.nn.functional.linear` — the reference the C++ comment
  above `Linear` already cited before this row touched it.

`F.linear` on float32 tensors is `addmm`, which dispatches to an f32 BLAS
`sgemm`: **f32 accumulate, and f64 appears nowhere.** So `double` was a
divergence from the oracle the file names, and f32 restores the mirror.

**What was NOT observed, stated rather than implied.** This row did not *watch*
upstream accumulate. At the goldens' `flat = 24` an f32 and an f64 accumulation
are indistinguishable, and no Lightricks checkout was available here to run the
188160-wide case, so the width claim rests on the documented aten dispatch and
not on a measurement. `## Owed` carries the shipped-width oracle gap.

The cost of the decision was measured at `K = 188160` on the magnitudes this
path carries (unit-RMS activations, `U(-1/sqrt(fan_in))` weights, 256 sampled
outputs), against a `long double` reference:

| accumulation | max abs err | max rel err | mean rel err |
|---|---|---|---|
| f32, sequential | 2.12e-05 | 2.22e-03 | 2.88e-05 |
| f64, then store f32 | 5.88e-08 | 5.80e-08 | 2.26e-08 |

So f64 is ~360x nearer the exact sum, and the f32 arm's worst absolute error is
**2.12e-05 on values of magnitude up to 1.83** — 1.2e-05 relative, which is
**~330x below one bf16 ulp of the same value**. The max relative figure of
2.22e-03 is a single near-cancelling output whose magnitude is tiny, not a
typical one; the mean is 2.88e-05. Upstream's own blocked f32 sum carries error
of the same order, so this is not a departure from what upstream computes — it is
a different reduction order within the same width.

**The goldens: ten of twelve numbers are byte-unchanged, two moved.** §8 made
"the goldens move at all" a stop condition, so it is decided here rather than in
an assertion. `ltx2 text V1 extractor` went from 5.96e-08 to **1.19e-07** (left)
and from 5.96e-08 to **7.45e-08** (right), against the file's 1e-5 bound —
84x and 134x inside it. Every V2 number and both hand-off numbers are unchanged
to the printed precision. **No tolerance was widened**, and the V1 arm is the one
that moved because it is the bias-free projection, where the seam's f32
reduction is not followed by a bias add that rounds the difference away.

### 5. What was rejected

* **Keeping the f64 loop and threading it by hand.** It is the shared-seam rule's
  exact prohibition, and the seam already provides the threading, the SIMD tier
  and a bit-identity gate against its own reference kernel.
* **Widening `kTol`.** Nothing needed it; see §4.
* **Seeding the accumulator with the bias**, which the old loop did. The bias is
  added after the GEMM now, which is both what `F.linear` does (`addmm`'s
  epilogue) and what the DiT's sibling `Linear` does.
* **Removing the 771 MB `scaled` copy** in `project()`. **This change KEEPS it**,
  unchanged, on both arms. Now measured rather than estimated: both copies
  together are 0.390-0.468 s, which is 0.06% of the 671.777 s before arm and
  0.5-0.6% of the 78.421 s after arm. Removing it is a real but small win on the
  fixed side, it belongs to whoever takes the memory-format work, and chasing it
  here would widen the diff for half a percent.

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
* **`R`, the GB10-to-x86 per-core ratio for this loop.** §1 bounds it to
  [1.0, 2.587] and measures it nowhere, so 0-1066 s of each 1738 s single-core
  stretch is unattributed. One `rc` lease on `dgx:gpu0` running this row's
  harness closes it.
* **No shipped-width golden.** The projection's upstream oracle exists only at
  `flat = 24`, because `scripts/gen-ltx2-text-goldens.py` needs the Lightricks
  checkout to regenerate and the shipped width would emit a 3 GB fixture. The
  accumulator claim at `K = 188160` therefore rests on an exactness reference
  computed here, not on upstream.
