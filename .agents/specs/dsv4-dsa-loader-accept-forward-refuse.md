# DSV4-DSA-ACCEPT-REFUSE — the loader takes the real DSA geometry, the forward refuses by name

Issue: [#1970](https://github.com/mudler/vllm.cpp/issues/1970)
Owning row: `MODEL-DSV4-EXL3`
Scoped by: [`.agents/specs/dsv4-dsa-geometry.md`](dsv4-dsa-geometry.md) ([#1961](https://github.com/mudler/vllm.cpp/issues/1961))
Oracle: vLLM, primary, at the parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`
(`.agents/upstream-sync.md`), read at `/home/mudler/_git/vllm`. Every `file:line`
below is read at that pin. No secondary oracle is used or needed: vLLM registers
and implements this architecture in full.

This is **option C** of the three the geometry spec returned as `NEEDS_DECISION`.
It is a **strict prefix of option A** (the full DSA port) and forecloses nothing.

## Scope

1. The EXL3 loader materializes every DSA tensor at the width the REAL
   DeepSeek-V4-Flash artifact stores, derived the way upstream derives it, so the
   real checkpoint loads instead of shape-refusing on 41 of 43 layers.
2. `AttentionBlock` verifies, before it indexes any DSA tensor, that the
   materialized width is the one its own arithmetic assumes, and REFUSES BY NAME
   when it is not.

Nothing else moves. In particular the DSA maths is not ported, dense MLA does not
become a fallback, and the GGUF arm's behaviour is unchanged.

## Upstream anchors

The whole geometry follows from one line:

```
vllm/models/deepseek_v4/compressor.py:247-248
    self.overlap = compress_ratio == 4
    self.coff = 1 + self.overlap
```

and is spent on exactly three parameters, plus a norm that is NOT widened:

| upstream | anchor | width |
|---|---|---|
| `self.ape` | `compressor.py:270-277` | `[compress_ratio, coff * head_dim]` |
| `fused_wkv_wgate` | `compressor.py:279-287` | two outputs of `coff * head_dim` |
| `self.norm` | `compressor.py:288` | `RMSNorm(self.head_dim, self.rms_norm_eps)` — **not** `coff * head_dim` |
| `indexer.wq_b` | `attention.py:721-726` | `ReplicatedLinear(q_lora_rank, head_dim * n_head)` |

The indexer carries its OWN `DeepseekCompressor` at `head_dim = index_head_dim`
and the SAME `compress_ratio` (`attention.py:768-776`), so its `ape`/`wgate`/`wkv`
are `coff * index_head_dim` wide and its `norm` is `index_head_dim` wide. The
indexer exists only at `compress_ratio == 4` (`attention.py:274`, the `if`
itself; `:276` is a comment inside it), where `coff` is 2, which is why only
`cr == 4` layers refuse today.

Every anchor above was re-verified against the checkout at the pin during the
fresh-review repair, asserting UNIQUENESS and not mere existence. `:288` is the
only `RMSNorm(` in `compressor.py`; `:293`, cited here before the repair, is
`compress_ratio=compress_ratio` inside the `CompressorStateCache` call.

`compress_ratio` is per layer upstream — `max(1, config.compress_ratios[layer_id])`
(`attention.py:209`) — and our `DeepseekV4Params::compress_ratio(layer)`
(`include/vllm/model_executor/models/deepseek_v4.h:122-128`) already mirrors that,
including `has_indexer == (cr == 4)`. **No config-parsing change is owed**; the
per-layer list was already read correctly. The defect was only in the widths the
loader derived from it.
## Design

### D1. The loader DERIVES the width the way upstream derives it

`deepseek_v4_weights.cpp` gains upstream's own expression,

```
const int64_t coff = (cr == 4) ? 2 : 1;   // compressor.py:247-248
```

and a `RequireDsaDim` helper that refuses any width but the derived one BY NAME.
Three DIFFERENT rules produce the three widths, so the helper takes a `why` string
and each call site names its own derivation:

| tensor | required | derivation |
|---|---|---|
| `compressor.ape`, `compressor.wgate.weight` | `coff * head_dim` | `compressor.py:247-248`, `:270-277`, `:279-287` |
| `indexer.compressor.wkv.weight` | `2 * index_head_dim` | the indexer's own compressor at the same ratio (`attention.py:768-776`), which exists only at `cr == 4` (`:274`) |
| `indexer.wq_b` dim 1 | `q_lora_rank` | `ReplicatedLinear(q_lora_rank, head_dim * n_head)` (`attention.py:721-726`) called on `qr` in `DeepseekV4Indexer.forward` (`:835`) — **not** a `coff` width, and the refusal must not cite one |

`compressor.norm.weight` stays `{hd}` and `indexer.weights_proj` stays `{inh, H}`,
because upstream widens neither.

**Why it derives ONE width and does not accept two.** `coff` is a pure function of
`compress_ratio`. It sizes `ape` (`:272`), both halves of `fused_wkv_wgate`
(`:281`) and `state_cache.state_dim` (`:291`); the indexer is gated on
`compress_ratio == 4` (`attention.py:274`) and builds its own compressor at the
same ratio (`:768-776`). So for a given ratio upstream emits exactly ONE width,
and a `cr == 4` checkpoint carrying an UNDOUBLED family is one upstream cannot
load at all. `AGENTS.md` requires this loader to mirror every mode, default, error
and edge case upstream defines, so accepting a width upstream can never emit is a
divergence from the mirror — and production acceptance must not be widened to
suit a fixture.

The first cut of this row DID accept two widths, on the premise that refusing the
collapsed one would delete the synthetic gates. **That premise was not
reproducible, and the fresh review is what caught it.**
`test_deepseek_v4_compressor.cpp`, `test_deepseek_v4_dsa.cpp`,
`test_deepseek_v4_forward.cpp` and `test_deepseek_v4_mtp.cpp` contain ZERO
references to `LoadDeepseekV4*` or `dsv4_exl3_fixture`, so they never load through
this arm and cannot break. Only the two EXL3 suites did, and this change rewrites
their fixture anyway: `ForwardFixtureOptions()` moves from `compress_ratios =
{0, 4}` to `{0, 128}`, where `coff` is 1 and the derived width IS the collapsed
one, so the W2d, MoE, #1923 and residency cases keep driving an EXL3-loaded
compressor layer end to end through the production forward.

**What that costs, stated rather than implied.** No case runs an EXL3-loaded
INDEXER forward any more, because the indexer exists only at `cr == 4`, where the
real geometry is the one the forward refuses. That coverage is redundant with
`test_deepseek_v4_forward.cpp` and `test_deepseek_v4_dsa.cpp`, which exercise the
indexer maths at the synthetic geometry without loading through this arm, and the
real artifact refuses it in any case. The indexer's LOAD is still gated at
upstream's widths by the loader suite's tower case, which moves to
`real_dsa_geometry = true`.

**The derived form is also strictly safer than what stood before this row.**
Pre-PR (`git show c00625141:src/vllm/model_executor/models/deepseek_v4_weights.cpp`)
the loader required exactly `{hd, H}`, so a malformed `cr == 4` UNDOUBLED
checkpoint was ALREADY accepted and ALREADY ran the collapsed `win = 2` maths.
The two-width form was therefore not a new safety regression, and the derived form
improves on both — it is the first version that refuses that checkpoint.

**Half-widening needs no separate rule, and this is a structural claim rather
than a gated one — stated that way because the version before the fresh-review
repair asserted it with no gate at all.** Every member is checked independently
against the same derived value: `ape` through `Float`'s `RequireShape` at
`{cr, cw}`, `wgate` and the indexer's `wkv` through `RequireDsaDim`, `wq_b`
through `RequireDsaDim` on dim 1. There is no cross-member "family agrees with
itself" logic left that could be wrong, so a checkpoint that widens `wgate` but
not `ape` reds on `ape`. **What is gated is the fully collapsed family (M8) and
`indexer.wq_b` alone (M11); a MIXED half-widened fixture is not gated, and no
knob writes one.** M11 exists because it must: the compressor refuses before the
loader reaches `wq_b`, so without `collapsed_indexer_wq_b` that derivation would
be unfalsifiable no matter how many collapsed cases were added.

At `cr == 128`, `coff` is 1, the derived width is the collapsed one, and the 20
`cr == 128` layers that load today keep loading unchanged.

### D2. The forward checks its own preconditions and refuses

**What this is worth, stated exactly, because the first cut of this row
overstated it and the fresh review rejected the claim.** `Gemm`'s host arm is a
`MatVec` whose size assertion is **unconditional**: `deepseek_v4.cpp:413` is
`VT_CHECK(w.size() == out * in, "MatVec weight size mismatch")`, and `VT_CHECK`
(`include/vt/dtype.h:11`) is a plain `throw`, not an `assert`, so `NDEBUG` does
not remove it. That is also the arm the EXL3 DSA tensors take, because `Gemm`
(`:428`) enters its keep-quant branch only when `be.gguf != nullptr` and an EXL3
load has `be.gguf == nullptr` — and the keep-quant branch checks the shape too.
NEITHER arm is unchecked.

So a wide `comp_wgate` read at a `[hd, H]` stride does **not** produce a plausible
wrong number. It produces

```
vt: MatVec weight size mismatch at deepseek_v4.cpp:413
```

from the middle of a forward, on a checkpoint that loaded successfully, naming no
tensor, no layer, no geometry and nothing missing. **This is therefore a
DIAGNOSTICS improvement and that is the whole of it** — an anonymous crash
replaced by a precise named refusal. It is not the difference between wrong tokens
and a refusal. Three review rounds were needed to make every reachable copy say
so, and each round reported the sweep complete before the next one found more.
Round 3 found a TWELFTH, in `deepseek_v4_weights.cpp`'s carried-half block: it
read "names the tensor rather than producing a wrong number" in the same file
whose reader-shape paragraph already says the opposite. It is repaired. One copy
is beyond repair and is named here rather than reported as swept: the
[#1923](https://github.com/mudler/vllm.cpp/issues/1923) row in
`.agents/issue-index.md`, which is on `origin/main` and append-only.
Overstating it is exactly the class of false justification
[#1964](https://github.com/mudler/vllm.cpp/issues/1964) was filed for, and this
row must not repeat it one directory over.

`AttentionBlock` therefore checks, for every DSA tensor it is about to index,
that the materialized element count equals the count its indexing assumes:

| slot | indexed as | anchor |
|---|---|---|
| `comp_ape` | `[cr, hd]` | `DispSaveScoreApe(..., T, hd, cr)` |
| `comp_wgate` | `[hd, H]` | `Gemm(..., T, hd, H)` |
| `comp_norm_weight` | `[hd]` | `DispPoolNorm(..., hd)` |
| `idx_wq` | `[inh * ihd, H]` | `Gemm(..., T, inh * ihd, H)` |
| `idx_wk` | `[ihd, H]` | `Gemm(..., T, ihd, H)` |
| `idx_wproj` | `[inh, H]` | `Gemm(..., T, inh, H)` |

All six are checked, not only the four the real artifact stores differently.

**Two of the six are not reachable from any checkpoint, and this says so rather
than leaving "all six are load-bearing" as an impression.** `comp_norm_weight` and
`idx_wproj` are the two slots upstream does not widen, so D1 requires them at
exactly the width this forward indexes and no artifact can make them disagree.
They are gated for FALSIFIABILITY only, by mutating a tower the production loader
produced (M9 and M10): that proves the checks fire and are named in the message,
and it proves nothing about reachability, because the state it constructs is one
the loader cannot emit. They are defensive checks against a future loader/forward
disagreement. The other four are what fires on the real checkpoint.

The refusal names the layer, the tensor, both element counts, the width the
forward composes with, the width the checkpoint carries, the `coff` that explains
it, the missing capability, the upstream anchors and the issue. It is a
`VT_CHECK`, so it surfaces as the same refusal every other unrepresentable input
on this path surfaces as.

**The check is gated on `is_comp || is_indexer`.** A layer whose DSA path the
forward does not enter reads none of these tensors, so checking it would refuse a
load that harms nothing. This is also what keeps the GGUF arm — where
`dsa_dense` makes both predicates false — byte-for-byte unchanged.

### D3. What this deliberately does NOT decide

At `cr == 128` the widths already match, so a `cr == 128` layer passes D2 and runs
the existing `win = 2` pooling of the MLA's own `kraw`. That is **not** upstream's
128-wide boundary-emitted compressor over a separate `compressor.wkv` projection,
and this change does not make it one. **It is
[#1976](https://github.com/mudler/vllm.cpp/issues/1976), filed by this repair, and
it is NOT #1964.** #1964 is the GGUF arm's `dsa_dense` routing every layer to
dense MLA; this is the EXL3 arm, where `dsa_dense` is false, the layer ENTERS the
compressor, and what it runs is a 2-wide pool over the MLA's own latent
(`deepseek_v4.cpp:833`) against upstream's 128-wide boundary-emitted compression
over a separate `compressor.wkv` projection (`compressor.py:171-173`). Window
width, emission cadence and source projection all differ. Closing #1964 would not
close this, and before #1976 nothing else tracked it. Recorded under `## Owed` rather than silently widened, because
widening D2 into "refuse every compressor layer" would refuse the gated synthetic
suites too, which is a scope change and not this unit of work.
## Risks

- **The refusal could be unreachable.** The failure `.agents/reachability.md`
  documents, and the exact failure #1923 already cost this row once: W2's
  reachability claim was gated on a `DeepseekV4Weights` the suite built BY HAND,
  so a load could not produce it. Mitigated by driving the RED test through
  `vllm::LoadDeepseekV4ForCausalLMWeights` and `vllm::DeepseekV4Model::Forward`,
  and by the reachability mutation in `## Gates` below.
- **The loader could widen without the forward moving.** Mitigated by landing
  both halves in one change and by the reachability mutation `M6`, which deletes
  the production call site. What that mutation observes is
  `vt: MatVec weight size mismatch at deepseek_v4.cpp:413` — a THROW, and not a
  wrong number. §D2 forty lines above says why: `MatVec`'s size assertion is
  unconditional and `Gemm`'s keep-quant arm checks the shape too, so neither arm
  is unchecked. The consequence of this risk is therefore an ANONYMOUS crash and
  not wrong tokens. An earlier draft of this bullet said "a wrong number rather
  than a throw"; that was the #1964 overclaim surviving in a seventh place, and
  it contradicted this document's own §D2.
- **The fixture could describe a geometry the artifact does not have.** It
  already did: `real_compressor_width` doubled the width unconditionally, so the
  one existing case that used it wrote a doubled compressor on a `cr == 128`
  layer, where upstream's `coff` is 1. Mitigated by replacing the flag with
  upstream's own per-layer rule, so the fixture cannot describe a width upstream
  would not produce.
## Tests

`tests/vllm/models/dsv4_exl3_fixture.h` — `real_compressor_width` becomes
`real_dsa_geometry`, applying `coff = 1 + (cr == 4)` per layer to the compressor
and indexer-compressor families, leaving both norms at `head_dim`, and writing
`indexer.wq_b` at `[inh * ihd, q_lora_rank]`. `collapsed_indexer_wq_b` collapses
that ONE tensor while the rest of the family stays real, `collapsed_indexer_wkv`
does the same for `indexer.compressor.wkv.weight`, and `bogus_dsa_width` writes a
third width no oracle emits.

`tests/vllm/models/test_deepseek_v4_exl3_forward.cpp` — `ForwardFixtureOptions()`
moves to `compress_ratios = {0, 128}`, where `coff` is 1 and the derived width is
the collapsed one, so every pre-existing case keeps running an EXL3-loaded
compressor layer end to end. `RealDsaFixtureOptions()` is the same model at
`{0, 4}` with the real geometry. One new case, entering through the production
loader and the production forward:

1. The real geometry **LOADS**. Red before D1 (`RequireShape` throws).
2. The forward **REFUSES BY NAME**, and the message carries the layer, every
   mismatched tensor, both counts and the missing capability. Red before D2 — and
   red for the DIAGNOSTIC, not for the throw: without D2 the forward throws the
   anonymous `MatVec weight size mismatch` from `deepseek_v4.cpp:413` instead, so
   `!msg.empty()` alone does not separate the two and every `Mentions` assertion
   after it is part of the gate.
3. A compressor layer the forward CAN index still runs.
4. `comp_norm_weight` and `idx_wproj` are gated by mutating the loaded tower,
   which is a falsifiability gate and not a reachability one (D2). They are
   mutated ONE PER FORWARD: at this fixture `head_dim - 1` and
   `index_n_heads * hidden_size - 1` are the same number (511), so a single
   message cannot say which tensor reported which count.

Every count assertion in that case is bound to the mismatch LINE that names its
tensor, not asserted as a bare number. Several counts coincide here —
`compress_ratio * head_dim`, `index_n_heads * index_head_dim * hidden_size` and
`2 * index_head_dim * hidden_size` are all 2048 — so a bare `Mentions(msg,
"2048")` is satisfied by a tensor other than the one it was written for, which
the round-2 review measured as two more surviving mutants. `indexer.wq_b` gains
the counts it never had.

`tests/vllm/models/test_deepseek_v4_exl3_loader.cpp` — the tower case moves to
`real_dsa_geometry = true` and asserts the doubled and the undoubled members side
by side, because an assertion that checked only the four doubled ones would pass a
loader that widened the whole family. Two new refusal subcases: a COLLAPSED
`cr == 4` family, which is the shape a two-width loader accepted and upstream
cannot emit; `indexer.wq_b` at `K = hidden_size` with the rest of the family
real, which is the only way to reach that check because the compressor refuses
first; and `indexer.compressor.wkv.weight` at the COLLAPSED `index_head_dim`
with the rest of the family real, which is the only way to reach the THIRD
derivation for the same reason. The round-2 fresh review found that third hole
by mutation: deleting the loader's `coff * index_head_dim` check left both suites
green, because no fixture wrote a real-geometry compressor beside a collapsed
indexer and the derivation's message was never read. The old compressor subcase stays retired: it asserted the behaviour this
change removes, and its `cr == 128` fixture doubled a layer where `coff` is 1.

## Gates

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_SERVER=OFF
cmake --build build --target test_deepseek_v4_exl3_forward test_deepseek_v4_exl3_loader -j 4
ctest --test-dir build -R 'deepseek_v4' --output-on-failure
scripts/agent-preflight.sh
```

Build ONLY those targets. A bare `ninja -C build` links every test binary in the
tree, which took `build/tests` to 9.4 GiB during the fresh-review repair, took the
host to 100% full, and failed the link with `No space left on device`. Per
`.agents/environment.md` an ENOSPC here makes checkers emit FALSE policy refusals
rather than clean failures, so it is worse than a slow build.

Every build records ninja's exit code AND its step count: a failed build silently
re-runs a stale binary and reads as a pass.

Mutations, each verified to have LANDED before the build, then applied to a
scratch copy and restored byte-for-byte under SHA-256:

| # | mutation | must |
|---|---|---|
| M1 | delete the `comp_wgate` width check | RED |
| M2 | delete the `comp_ape` width check | RED |
| M3 | delete the `idx_wq` width check | RED |
| M4 | delete the `idx_wk` width check | RED |
| M5 | loader reverts to the collapsed CONSTANT width (pre-#1970) | RED |
| M6 | **reachability**: delete the production call site in `AttentionBlock` | RED |
| M7 | `RequireDsaDim` accepts ANY width | RED |
| M8 | loader ALSO accepts the collapsed width (the two-width form D1 removes) | RED |
| M9 | delete the `comp_norm_weight` check | RED |
| M10 | delete the `idx_wproj` check | RED |
| M11 | `indexer.wq_b`'s K also accepts `hidden_size` | RED |
| M12 | `compressor.ape`'s EXPECTED count is wrong on the `cr == 4` layer | RED |
| M13 | `compressor.norm.weight`'s EXPECTED count is wrong on that layer | RED |
| M14 | delete the loader's `coff * index_head_dim` derivation for `indexer.compressor.wkv.weight` | RED |

M8 through M11 were added by the round-1 repair, and three of them are that
review's findings in executable form. M9 and M10 were GREEN against the first cut,
which is what showed two of the six forward checks were unfalsifiable. M11 was
GREEN even after D1 became strict, because the compressor refuses before the
loader reaches `wq_b` — which is why `collapsed_indexer_wq_b` exists.

M12 through M14 were added by the ROUND-2 repair and all three were GREEN against
`0acf0147f`. M14 is the loader's third derivation, unfalsifiable for exactly the
reason M11 was, one tensor over; `collapsed_indexer_wkv` is its
`collapsed_indexer_wq_b`. M12 and M13 are the count collisions: they leave the
tensor NAMED and make only its number wrong, which a bare `Mentions` of that
number cannot see because another tensor's line still carries it.

**Three derivations, and now three of them falsifiable.** `RequireDsaDim` is
called from three sites deriving three DIFFERENT rules — `coff * head_dim`,
`coff * index_head_dim`, and `q_lora_rank`, which is not a `coff` width at all —
and until M14 the middle one had no case that read its message. M7 mutates
`RequireDsaDim` itself and so cannot tell the three apart.

## Evidence

Measured on `row/DSV4-DSA-GEOMETRY`, host build
`cmake -S . -B build -G Ninja -DVLLM_CPP_SERVER=OFF`, GCC with `-Wall -Wextra
-Werror`. These are the FRESH-REVIEW REPAIR's numbers, re-derived on the repaired
tree. The first cut's table is superseded and not carried forward, because D1
changed shape and two of its mutations scored against a rule that no longer
exists.

**RED first, D1.** The strict rule's own gate — the loader must refuse a COLLAPSED
`cr == 4` family — written before the loader changed, against the two-width form.
`ninja rc=0, 2/2 steps`, so the red is the test and not a stale binary. The
`:661` inside the transcript below is the line of the tree it was CAPTURED on
and is NOT a live anchor: at this branch's head that line is a comment. It is
labelled here because round 3 found it labelled only in the pull-request body:

```
tests/vllm/models/test_deepseek_v4_exl3_loader.cpp:661: ERROR: CHECK( Mentions(msg, "coff") ) is NOT correct!
  values: CHECK( false )
  logged: msg :=

[doctest] test cases:  1 | 0 passed | 1 failed | 10 skipped
[doctest] assertions: 15 | 8 passed | 7 failed |
```

`msg` is EMPTY. The two-width loader accepted the collapsed `cr == 4` family and
threw nothing at all, which is the divergence from upstream this repair removes.

**RED first, D2**, from the first cut and still valid — the new forward case
failing because the loader refused the real width before the forward could see it:

```
TEST CASE:  dsv4 exl3 #1970: the REAL DSA geometry LOADS and the FORWARD refuses by name
ERROR: test case THREW exception: vt: deepseek-v4 exl3 loader:
  layers.1.attn.compressor.ape must be [4,512], got [4,1024].
[doctest] test cases: 1 | 0 passed | 1 failed | 4 skipped
[doctest] assertions: 0 | 0 passed | 0 failed |
```

That build was `ninja rc=0`, 500/500 steps.

**A previously recorded D2 red was WRONG and is withdrawn.** This section used to
read "no throw: the forward returns logits computed off a mis-indexed
`comp_wgate`". It does not and cannot: `MatVec`'s size assertion is unconditional
(D2), so without the refusal the forward throws
`vt: MatVec weight size mismatch at deepseek_v4.cpp:413`. The fresh reviewer
demonstrated exactly that by deleting the production call site while keeping the
helper referenced so it compiled under `-Werror`, and got the throw rather than
logits. What D2 buys is the DIAGNOSTIC, and no record here may claim more.

**RED first, round 2.** The three checks a mutation walked through at
`0acf0147f`, each measured GREEN there and RED after the repair. All six builds
were `ninja rc=0`. Verbatim, from the repaired tree:

```
test_deepseek_v4_exl3_loader.cpp:746: ERROR: CHECK( Mentions(msg, "coff * index_head_dim") ) is NOT correct!
  values: CHECK( false )
test_deepseek_v4_exl3_loader.cpp:754: ERROR: CHECK( Mentions(msg, "dimension 0 must be 8") ) is NOT correct!
  values: CHECK( false )
```

```
test_deepseek_v4_exl3_forward.cpp:478: ERROR: CHECK( dsv4_exl3_fixture::Mentions( msg,
  MismatchLine("compressor.ape", "[compress_ratio, head_dim]", cr * hd, cr * 2 * hd)) ) is NOT correct!
  values: CHECK( false )
  logged: msg := ...
    - attn.compressor.ape: this forward indexes it as [compress_ratio, head_dim] = 2049 elements, the checkpoint carries 4096
    - attn.indexer.wq_b: this forward indexes it as [index_n_heads*index_head_dim, hidden_size] = 2048 elements, the checkpoint carries 1024
```

That second block is the finding itself, printed: the mutated `compressor.ape`
line reads 2049, and the 2048 the OLD bare assertion matched is sitting two lines
below it on `indexer.wq_b`. The old assertion could not have failed.

**GREEN after**, whole suites rather than the one case:

| suite | result |
|---|---|
| `test_deepseek_v4_exl3_forward` | 5 cases, 69 assertions, 0 failed |
| `test_deepseek_v4_exl3_loader` | 11 cases, 172 assertions, 0 failed |
| `ctest -R deepseek_v4` | 13/14 passed |

`test_cuda_deepseek_v4` is the 14th and reports `Not Run`: this build has CUDA
off, so it is not a result either way.

**Mutations.** Each was verified to have LANDED before the build (a mutation that
never applied reads as a passing test), then built, run, restored, and the restore
verified by SHA-256 before the next one. `ninja rc` and step count are recorded
for each, because a mutation that fails to BUILD silently re-runs the previous
binary and also reads as a pass:

| # | mutation | ninja | steps | verdict |
|---|---|---|---|---|
| M1 | delete the `comp_wgate` width check | rc=0 | 4/4 | RED (forward) |
| M2 | delete the `comp_ape` width check | rc=0 | 4/4 | RED (forward) |
| M3 | delete the `idx_wq` width check | rc=0 | 4/4 | RED (forward) |
| M4 | delete the `idx_wk` width check | rc=0 | 4/4 | RED (forward) |
| M5 | loader reverts to the collapsed CONSTANT width | rc=0 | 4/4 | RED (both) |
| M6 | **reachability**: delete the production call site | rc=0 | 4/4 | RED (forward) |
| M7 | `RequireDsaDim` accepts ANY width | rc=0 | 4/4 | RED (loader) |
| M8 | loader ALSO accepts the collapsed width | rc=0 | 4/4 | RED (loader) |
| M9 | delete the `comp_norm_weight` check | rc=0 | 4/4 | RED (forward) |
| M10 | delete the `idx_wproj` check | rc=0 | 4/4 | RED (forward) |
| M11 | `indexer.wq_b`'s K also accepts `hidden_size` | rc=0 | 4/4 | RED (loader) |
| M12 | `compressor.ape`'s EXPECTED count wrong on the `cr == 4` layer | rc=0 | 4/4 | **GREEN at `0acf0147f`**, RED after |
| M13 | `compressor.norm.weight`'s EXPECTED count wrong on that layer | rc=0 | 4/4 | **GREEN at `0acf0147f`**, RED after |
| M14 | delete the loader's `coff * index_head_dim` derivation | rc=0 | 4/4 | **GREEN at `0acf0147f`**, RED after |
| — | restored tree | rc=0 | 4/4 | forward 5/69, loader 11/172, `ctest -R deepseek_v4` 13/14 |

The WHOLE table is re-derived at the branch head, not carried forward from round
1. The round-2 repair changed both the runtime message and the assertions that
read it, and a later commit silently disarming an earlier commit's mutation proof
is a real failure mode — a table measured on a tree that no longer exists proves
nothing about this one. Each mutation was verified to have LANDED before the
build, and each restore verified byte-for-byte by SHA-256 with a rebuild before
the next one, because a mutation that never applied and a mutation that failed to
BUILD both re-run the previous binary and both read as a pass.

M12 and M13 are scoped to the `cr == 4` layer on purpose. An unscoped version
also breaks the `cr == 128` case in §3, which then reds for a reason that has
nothing to do with the collision — a mutation that reds by collateral damage
measures nothing. The first attempt at M12 did exactly that, and a second attempt
that dropped `cr` from the expression failed to BUILD under `-Werror`
(`ninja rc=1`) while the previous binary still reported SUCCESS, which is the
stale-binary trap this section records step counts for.

M1-M4 are individually falsifiable only because the refusal reports EVERY mismatch
rather than stopping at the first; a first-mismatch refusal would have made three
of them undetectable. That is also what lets M9 and M10 score at all: deleting one
check leaves the throw but drops that tensor's name from the message.

**Round 3 (comments and records only).** Focused rebuild `ninja rc=0`, 4/4 steps;
forward 5 cases / 69 assertions, loader 11 cases / 172 assertions. The other ten
`deepseek_v4` suites were RELINKED — `ninja rc=0`, 10/10 steps — rather than run
stale against the previous `libvllm.a`, and `ctest -R deepseek_v4` is 13/14 with
`test_cuda_deepseek_v4` `Not Run` (CUDA off). Every number matches the restored
tree above, which is the whole point: a round that changes no assertion, no
derivation and no call site must not move a count. The two binaries' SHA-256 did
change, and that is expected rather than a build-identity failure — repairing the
carried-half comment added three lines to `deepseek_v4_weights.cpp`, `VT_CHECK`
embeds `__LINE__`, so a comment above a `VT_CHECK` is not a byte-identical build.
The same three lines moved `deepseek_v4_weights.cpp:992` to `:995`, which is the
stale-local-anchor failure arriving once more inside this pull request; every
local anchor in both specs and in the three index rows was re-resolved afterwards.
No mutation was re-run: round 3 touches nothing a mutation scores against.

## Owed

- **The full DSA port (option A) has NO owning row.** `MODEL-DSV4-EXL3` carries
  it, as its own `## Owed` already says of the dense-MLA policy this supersedes.
  It needs the `coff`-overlapped window with `head_offset` role selection
  (`common/ops/fused_compress_quant_cache.py:164-183`, the main compressor's
  `_fused_kv_compress_norm_rope_insert_sparse_attn`), boundary-only emission,
  a compressed KV cache beside a SWA(128) raw cache — which needs the cache
  topology [#1960](https://github.com/mudler/vllm.cpp/issues/1960) and
  [#1925](https://github.com/mudler/vllm.cpp/issues/1925) are scoping — the
  indexer on `qr` over compressed rows, and one joint softmax over the union.
  Our `CompressorSaveScoreApe` / `CompressorPoolNorm` primitives are already
  generic over width and window and carry over unchanged; the gap is the
  composition, not the maths.
- **[#1964](https://github.com/mudler/vllm.cpp/issues/1964) stays open and stays
  unfixed here.** The GGUF arm's `dsa_dense` still runs on the real geometry, so
  the shipping GGUF DeepSeek-V4 path is still not upstream's attention on 41 of
  43 layers, and the false exactness justification at
  `deepseek_v4.cpp:763-775` is still quoted onward by
  [#1925](https://github.com/mudler/vllm.cpp/issues/1925). Out of scope by the
  dispatch, which excludes changing the GGUF arm's behaviour.
- **`cr == 128` EXL3 layers pass the width check and run the wrong compressor**
  (D3). This is [#1976](https://github.com/mudler/vllm.cpp/issues/1976), filed by
  the fresh-review repair. It was attributed to #1964 here and that was WRONG:
  #1964 is `dsa_dense` routing the GGUF arm to dense MLA, while this is the EXL3
  arm ENTERING the compressor and running a 2-wide pool over the MLA's own latent
  (`deepseek_v4.cpp:833`) where upstream pools 128 wide, emits only at boundary
  tokens, and reads its own `compressor.wkv` projection. Closing #1964 would not
  have closed it, and nothing else tracked it.
- **The `indexer.wq_b` input-space defect.** Upstream projects the indexer query
  from `qr`, the q-LoRA latent (`DeepseekV4Indexer.forward`, `attention.py:835`);
our forward feeds it the
  hidden state (`deepseek_v4.cpp:915`). After D1 the loader materializes the
  tensor at its real `[inh * ihd, q_lora_rank]`, so the forward's `[inh * ihd, H]`
  indexing now REFUSES instead of mis-indexing — but the wrong input space is a
  real defect at any geometry and option A owns fixing it.
- **No real-checkpoint run.** Every gate here is the hermetic fixture. That the
  99.5 GiB artifact now loads is asserted at the fixture's geometry, not measured
  on the artifact, which needs the box and W2 residency.
## Outcome

**What was measured.** Every upstream shape was confirmed against the checkout at
the pin rather than taken from the scoping spec, asserting UNIQUENESS and not mere
existence: `compressor.py:247-248` (`coff`), `:270-277` (`ape`), `:279-287`
(`fused_wkv_wgate`), `:288` (`RMSNorm(self.head_dim, self.rms_norm_eps)` — the
norm is NOT widened, and it is the file's only `RMSNorm(`), `attention.py:721-726`
and `:835` (`DeepseekV4Indexer.forward`'s `wq_b` on `qr` from `q_lora_rank`),
`:768-776` (the indexer's own
compressor at `index_head_dim`), `:274` (the indexer exists only at `cr == 4`).

The first cut cited `compressor.py:293` for the norm in five places. `:293` is
`compress_ratio=compress_ratio` inside the `CompressorStateCache` call: the line
drifted by five while the surrounding claim stayed plausible, and it was quoted
onward into the commit body and the append-only index row, where it could not have
been amended after the squash. That is why an anchor is now checked for uniqueness
rather than read once and repeated.

**It happened a second time, with `attention.py:276`, and the sweep for it stopped
one file short.** Round 1 corrected six copies. A seventh survived in
`deepseek_v4_weights.cpp`'s indexer block, twelve lines above the SAME function's
correct `:274`, and the pull-request body then claimed the correction as complete.
Round 2 corrected the seventh and swept the whole tree, which now carries zero
copies of `:276`. `:274` is `if self.compress_ratio == 4:` and the construct is
unique in `attention.py`; `:276` is a comment about `aux_stream_list`. SIX cited
constructs are NOT unique, and the previous revision of this section said TWO.
The number is MEASURED and not read for: take every upstream anchor this branch
adds (`git diff origin/main...HEAD`, added lines only — all of them land in
`vllm/models/deepseek_v4/`), take the construct quoted beside each, and count that
construct's occurrences in its own file at the pin.

| construct | file | occurrences |
|---|---|---|
| `self.compressor = DeepseekCompressor(` | `attention.py` | 2 — `:335` (`DeepseekV4Attention.__init__`), `:768` (`DeepseekV4Indexer.__init__`) |
| `self.wq_b(qr)` | `attention.py` | 4 — `:480`, `:514`, `:527` (`DeepseekV4Attention.attention_impl`), `:835` (`DeepseekV4Indexer.forward`) |
| `if (position + 1) % COMPRESS_RATIO != 0:` | `common/ops/fused_compress_quant_cache.py` | 5 — `:164` (`_fused_kv_compress_norm_rope_insert_sparse_attn`), `:364`, `:429`, `:712`, `:891` |
| `head_offset = (tokens >= COMPRESS_RATIO)…* HEAD_SIZE` | `common/ops/fused_compress_quant_cache.py` | 3 — `:182` (that same main-compressor kernel), `:730` (indexer), `:909` (mxfp4 indexer) |
| `swa_only = self.compress_ratio <= 1` | `nvidia/flashinfer_sparse.py` | 3 — `:263` (`DeepseekV4FlashInferMLAAttention.forward_mqa`), `:686`, `:793` |
| `flashinfer_trtllm_batch_decode_sparse_mla_dsv4(` | `nvidia/flashinfer_sparse.py` | 4 — `:486`, `:511`, `:769` (`DeepseekV4FlashInferSM120Attention._forward_decode`), `:888` |

A seventh is NOT one, and the difference is the whole point of measuring rather
than eyeballing: `[self.coff * self.head_dim, self.coff * self.head_dim],` occurs
at `compressor.py:281` AND `:335`, but the citation is the RANGE `:279-287` and
`:279` is unique, so that anchor already picks out its own line. EVERY anchor in
the table is CORRECT — this is a count defect and not a wrong line — but a
construct that does not pick out its own line cannot be checked by the reader it
was written for, so each is now carried with its enclosing class or kernel named
beside it.

One more anchor defect of a different kind, found in the same pass:
`save_partial_states.py:85-101` began on a BLANK line. The range now starts at
`:86`, its first real line.

**Local anchors go stale inside the pull request that writes them, and that is a
separate failure from citing the wrong upstream line.** Five of the six local
`file:line` citations in `dsv4-dsa-geometry.md` were correct at `c00625141`, where
that document was measured, and stale by `0acf0147f`, because #1970's own
implementation moved the lines underneath it. Round 1 swept this document and not
that one. Both are swept now, and every local anchor in both is re-derived at the
branch head.

**What was rejected, and why.**

*Accepting two widths.* The first cut derived one width from `coff` and then
relaxed to accept the collapsed one as well, on the premise that refusing it would
delete the synthetic gates. The fresh review disproved the premise: the four
synthetic DSA suites do not load through this arm at all, only the two EXL3 suites
broke, and this change rewrites their fixture regardless. The two-width form was a
divergence from the mirror bought with nothing, and it is gone. What replaced it is
`compress_ratios = {0, 128}` in the forward fixture, where `coff` is 1 and the
collapsed width IS the derived one — the synthetic geometry moves into the RATIO,
where upstream can actually produce it, instead of into the loader's accepted set.

*Claiming the refusal prevents wrong tokens.* The first cut said `MatVec` had "no
length check" and that a mis-indexed `comp_wgate` was a plausible wrong number.
`deepseek_v4.cpp:413` is an unconditional `VT_CHECK`, both `Gemm` arms check, and
the EXL3 path takes the checked one. The refusal buys a DIAGNOSTIC, and each round
that swept for the claim found copies the round before it had reported gone: six
in round 1, four in round 2, and a twelfth in round 3, in the carried-half comment
of the same file whose corrected paragraph it contradicted. Every copy this branch
can reach — code comment, runtime message, spec, commit body, index row — now says
so; the [#1923](https://github.com/mudler/vllm.cpp/issues/1923) row is on
`origin/main` and append-only, so it keeps the claim permanently and this document
names it instead of claiming a clean tree.

*Naming every alternative throw in the runtime refusal.* The message says the
alternative is `MatVec weight size mismatch`. That is right for the case the
refusal actually fires on — at the real geometry `comp_wgate`'s `Gemm` runs before
anything reads `comp_ape` or `comp_norm_weight` — but the sentence said "reading
THEM", plural, and a `comp_ape`-only or `comp_norm_weight`-only mismatch throws
`ape size mismatch` / `rms_weight size mismatch` from
`deepseek_v4_compressor.cpp:23,54` instead. The fix is the smallest one that
removes the overclaim: the sentence now names `comp_wgate` and the two other
throws are named in one parenthesis. It does not enumerate every ordering, because
the point is that all three are equally anonymous and that is the whole claim.
Those two line numbers were derived rather than read: `VT_CHECK` embeds
`__LINE__`, and GCC reports the line a multi-line macro invocation BEGINS on, not
the line its message literal sits on — `:24` and `:55` are where the strings are,
`:23` and `:54` are what the throw prints.

*Two options upstream of this row*, recorded in `dsv4-dsa-geometry.md`: (A) the
full DSA port, correct but a multi-wave model port blocked on cache topology; (B)
a per-layer dense selector, which is still not upstream's attention on 41 layers
and so cannot be gated as parity.

**Why each default has its value.** `coff = (cr == 4) ? 2 : 1` is upstream's
expression verbatim, not a fit to the artifact. The indexer branch uses the
constant 2 rather than a derivation because the branch is reached only at
`cr == 4`. `indexer.wq_b`'s K is `q_lora_rank` and its refusal cites
`attention.py:721-726` rather than `coff`, because nothing about that tensor is
doubled and the first cut's message blamed the wrong rule. `RequireDsaDim` takes
its `why` per call site for that reason, and all THREE of its derivations now have
a fixture case that reads the message — `coff * head_dim`,
`coff * index_head_dim`, and `q_lora_rank` — because a derivation nothing reads
is a derivation nothing gates. The forward check is
gated on `is_comp || is_indexer` so it fires exactly where a tensor is about to be
read, which is also what leaves the GGUF arm untouched.

**What this row now owes that it did not before.**
[#1976](https://github.com/mudler/vllm.cpp/issues/1976) exists because D3 was
attributed to #1964 and would have been "closed" by closing an unrelated defect on
another arm.

## Now

`ACTIVE` under `MODEL-DSV4-EXL3`. No lifecycle state moved by this document.
