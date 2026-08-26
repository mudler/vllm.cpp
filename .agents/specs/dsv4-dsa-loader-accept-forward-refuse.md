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
| `self.norm` | `compressor.py:293` | `RMSNorm(self.head_dim)` — **not** `coff * head_dim` |
| `indexer.wq_b` | `attention.py:721-726` | `ReplicatedLinear(q_lora_rank, head_dim * n_head)` |

The indexer carries its OWN `DeepseekCompressor` at `head_dim = index_head_dim`
and the SAME `compress_ratio` (`attention.py:768-776`), so its `ape`/`wgate`/`wkv`
are `coff * index_head_dim` wide and its `norm` is `index_head_dim` wide. The
indexer exists only at `compress_ratio == 4` (`attention.py:276`), where `coff`
is 2, which is why only `cr == 4` layers refuse today.

`compress_ratio` is per layer upstream — `max(1, config.compress_ratios[layer_id])`
(`attention.py:209`) — and our `DeepseekV4Params::compress_ratio(layer)`
(`include/vllm/model_executor/models/deepseek_v4.h:122-128`) already mirrors that,
including `has_indexer == (cr == 4)`. **No config-parsing change is owed**; the
per-layer list was already read correctly. The defect was only in the widths the
loader derived from it.

## Design

### D1. The loader READS the width instead of assuming it

`deepseek_v4_weights.cpp` gains upstream's own expression,

```
const int64_t coff = (cr == 4) ? 2 : 1;   // compressor.py:247-248
```

and a `DsaDim` helper that takes the width from the CHECKPOINT and requires it to
be one of two nameable values: upstream's `coff` width, or the collapsed width the
host forward indexes. The family is then required to agree with itself, so a
half-widened checkpoint still refuses, and any OTHER width still refuses by name.
`indexer.wq_b`'s K is read the same way against `{hidden_size, q_lora_rank}`.
`compressor.norm.weight` stays `{hd}` and `indexer.weights_proj` stays `{inh, H}`,
because upstream widens neither.

**Why it accepts two widths and not just upstream's.** A first cut derived the
width strictly from `coff` and broke every gated synthetic suite, because the
collapsed fixture geometry is `cr == 4` with an UNDOUBLED family — a combination
upstream never emits, and the one the compressor/indexer primitives are exercised
at. Refusing it would have deleted that gate rather than fixed anything, and it
is a behaviour change this unit of work was not dispatched to make. So the loader
takes either and `AttentionBlock` decides which one it can index. This widens the
accepted set by exactly one nameable value; it does not stop checking.

At `cr == 128`, `coff` is 1, the two values coincide, and the 20 `cr == 128`
layers that load today keep loading unchanged.

### D2. The forward checks its own preconditions and refuses

The geometry spec identified the hazard precisely: `Gemm`'s host arm is a
`MatVec` with **no length check** (`deepseek_v4.cpp:413` checks
`w.size() == out * in` only on the unquantized arm), so a wide `comp_wgate` in a
slot indexed as `[hd, H]` is a silently wrong number rather than a crash.

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

All six are checked, not only the four the real artifact stores differently. The
four are what fires on the real checkpoint; the other two are the same missing
length check and cost one line each.

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
and this change does not make it one — it is the pre-existing behaviour
[#1964](https://github.com/mudler/vllm.cpp/issues/1964) owns, on a path this row
does not touch. Recorded under `## Owed` rather than silently widened, because
widening D2 into "refuse every compressor layer" would refuse the gated synthetic
suites too, which is a scope change and not this unit of work.

## Risks

- **The refusal could be unreachable.** The failure `.agents/reachability.md`
  documents, and the exact failure #1923 already cost this row once: W2's
  reachability claim was gated on a `DeepseekV4Weights` the suite built BY HAND,
  so a load could not produce it. Mitigated by driving the RED test through
  `vllm::LoadDeepseekV4ForCausalLMWeights` and `vllm::DeepseekV4Model::Forward`,
  and by the reachability mutation in `## Gates` below.
- **The loader could widen without the forward moving.** That is the silent
  mis-index. Mitigated by landing both halves in one change and by a mutation
  that deletes the check and observes a wrong number rather than a throw.
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
`indexer.wq_b` at `[inh * ihd, q_lora_rank]`.

`tests/vllm/models/test_deepseek_v4_exl3_forward.cpp` — one new case, entering
through the production loader and the production forward:

1. The real geometry **LOADS**. Red before D1 (`RequireShape` throws).
2. The forward **REFUSES BY NAME**, and the message carries the layer, the
   tensor, both widths and the missing capability. Red before D2 (no throw: the
   forward returns logits computed off a mis-indexed `comp_wgate`).

`tests/vllm/models/test_deepseek_v4_exl3_loader.cpp` — the existing
"a carried tensor this arm cannot route REFUSES BY NAME" subcase for the
compressor is retired, because the thing it asserts is exactly what this change
removes. Its `cr == 128` fixture also described a width upstream does not
produce. The refusal it was protecting moves to the forward and is re-gated
there.

## Gates

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_SERVER=OFF
cmake --build build --target test_deepseek_v4_exl3_forward test_deepseek_v4_exl3_loader -j 4
ctest --test-dir build -R 'deepseek_v4' --output-on-failure
scripts/agent-preflight.sh
```

Every build records ninja's exit code AND its step count: a failed build silently
re-runs a stale binary and reads as a pass.

Mutations, each applied to a scratch copy and restored byte-for-byte:

| # | mutation | must |
|---|---|---|
| M1 | delete the `comp_wgate` width check | RED |
| M2 | delete the `comp_ape` width check | RED |
| M3 | delete the `idx_wq` width check | RED |
| M4 | delete the `idx_wk` width check | RED |
| M5 | drop `coff * hd` from `DsaDim`'s accepted set (revert D1) | RED — the load refuses again |
| M7 | make `DsaDim` accept ANY width | RED — a third width must still refuse |
| M6 | **reachability**: delete the production call site of the check in `AttentionBlock` | RED |

## Evidence

Measured on `row/DSV4-DSA-GEOMETRY` at the implementation commit, host build
`cmake -S . -B build -G Ninja -DVLLM_CPP_SERVER=OFF`, GCC with `-Wall -Wextra
-Werror`.

**RED first**, before either half landed — the new forward case, failing for the
intended reason (the loader refusing the real width):

```
TEST CASE:  dsv4 exl3 #1970: the REAL DSA geometry LOADS and the FORWARD refuses by name
ERROR: test case THREW exception: vt: deepseek-v4 exl3 loader:
  layers.1.attn.compressor.ape must be [4,512], got [4,1024].
[doctest] test cases: 1 | 0 passed | 1 failed | 4 skipped
[doctest] assertions: 0 | 0 passed | 0 failed |
```

That build was `ninja rc=0`, 500/500 steps, so the red is the test and not a
stale binary.

**GREEN after**, whole suites rather than the one case:

| suite | result |
|---|---|
| `test_deepseek_v4_exl3_forward` | 5 cases, 63 assertions, 0 failed |
| `test_deepseek_v4_exl3_loader` | 11 cases, 148 assertions, 0 failed |
| `ctest -R deepseek_v4 -E cuda` | 13/13 passed |

`test_cuda_deepseek_v4` is `Not Run`: this build has CUDA off, so it is not a
result either way.

**Mutations.** Every one applied to the tree, rebuilt, run, then restored and the
restore verified by SHA-256 before the next one. `ninja rc` and step count are
recorded for each, because a mutation that fails to BUILD silently re-runs the
previous binary and reads as a passing test:

| # | mutation | ninja | steps | verdict |
|---|---|---|---|---|
| M1 | delete the `comp_wgate` width check | rc=0 | 4/4 | RED |
| M2 | delete the `comp_ape` width check | rc=0 | 4/4 | RED |
| M3 | delete the `idx_wq` width check | rc=0 | 4/4 | RED |
| M4 | delete the `idx_wk` width check | rc=0 | 4/4 | RED |
| M5 | loader stops accepting upstream's `coff` width | rc=0 | 5/5 | RED |
| M6 | **reachability**: delete the production call site | rc=0 | 5/5 | RED |
| M7 | `DsaDim` accepts ANY width | rc=0 | 5/5 | RED |
| — | restored tree | rc=0 | 4/4 | forward and loader both green |

M1-M4 are individually falsifiable only because the refusal reports EVERY
mismatch rather than stopping at the first; a first-mismatch refusal would have
made three of them undetectable, since deleting any one would still throw on an
earlier one. That is why the message is built as a list.

M2, M5 and M6 each failed to COMPILE on the first attempt under `-Werror`
(`unused-variable` on `cr` and `coff`, `unused-function` on the helper), and the
harness reported "BUILD FAILED — not a test result" rather than scoring them. The
`unused-function` failure is itself evidence for M6: the compiler confirms the
deleted call is the helper's ONLY caller. Each was then rewritten to compile
while still removing what it tests.

## Owed

- **The full DSA port (option A) has NO owning row.** `MODEL-DSV4-EXL3` carries
  it, as its own `## Owed` already says of the dense-MLA policy this supersedes.
  It needs the `coff`-overlapped window with `head_offset` role selection
  (`common/ops/fused_compress_quant_cache.py:164-183`), boundary-only emission,
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
  `deepseek_v4.cpp:664-676` is still quoted onward by
  [#1925](https://github.com/mudler/vllm.cpp/issues/1925). Out of scope by the
  dispatch, which excludes changing the GGUF arm's behaviour.
- **`cr == 128` EXL3 layers pass the width check and run the wrong compressor**
  (D3). Same defect as #1964, same owner, reached by a different arm.
- **The `indexer.wq_b` input-space defect.** Upstream projects the indexer query
  from `qr`, the q-LoRA latent (`attention.py:835`); our forward feeds it the
  hidden state (`deepseek_v4.cpp:809`). After D1 the loader materializes the
  tensor at its real `[inh * ihd, q_lora_rank]`, so the forward's `[inh * ihd, H]`
  indexing now REFUSES instead of mis-indexing — but the wrong input space is a
  real defect at any geometry and option A owns fixing it.
- **No real-checkpoint run.** Every gate here is the hermetic fixture. That the
  99.5 GiB artifact now loads is asserted at the fixture's geometry, not measured
  on the artifact, which needs the box and W2 residency.

## Outcome

**What was measured.** The four refusing shapes were confirmed against the pin
rather than taken from the scoping spec: `compressor.py:247-248` (`coff`),
`:270-277` (`ape`), `:279-287` (`fused_wkv_wgate`), `:293` (`RMSNorm(head_dim)` —
the norm is NOT widened), `attention.py:721-726` and `:835` (`wq_b` on `qr` from
`q_lora_rank`), `:768-776` (the indexer's own compressor at `index_head_dim`),
`:276` (the indexer exists only at `cr == 4`).

**What was rejected, and why.** The first cut derived the expected width strictly
from `coff` and broke every gated synthetic suite: the collapsed fixture geometry
is `cr == 4` with an UNDOUBLED family, a combination upstream never emits and the
one the compressor/indexer primitives are exercised at. Refusing it would have
deleted a working gate to satisfy a rule about a checkpoint that gate does not
use. `DsaDim` accepting two nameable widths — and M7 proving a third still
refuses — is what keeps both true at once.

Two other options were rejected upstream of this row and are recorded in
`dsv4-dsa-geometry.md`: (A) the full DSA port, correct but a multi-wave model port
blocked on cache topology; (B) a per-layer dense selector, which is still not
upstream's attention on 41 layers and so cannot be gated as parity.

**Why each default has its value.** `coff = (cr == 4) ? 2 : 1` is upstream's
expression verbatim, not a fit to the artifact. `icoff = 2` in the indexer branch
is a constant rather than a derivation because the branch is reached only at
`cr == 4`. The check is gated on `is_comp || is_indexer` so it fires exactly where
a tensor is about to be read, which is also what leaves the GGUF arm untouched.

## Now

`ACTIVE` under `MODEL-DSV4-EXL3`. No lifecycle state moved by this document.
