# DSV4-DSPARK-DRAFTER — port the DSpark block drafter, the lever the 44-47 tok/s target actually uses

Row: `DSV4-DSPARK-DRAFTER` (proposed)
Issues: owed. The agent GitHub account is suspended, so no issue could be filed for
this spec; it is written against the two rows that own the surrounding work,
[#1314](https://github.com/mudler/vllm.cpp/issues/1314) (`CLAIM-DEEPSEEK-V4-MTP`)
and [#1875](https://github.com/mudler/vllm.cpp/issues/1875) (`MODEL-DSV4-EXL3`),
and the row is not claimable until its own issue exists.
State: `SPIKE` — records only. No product code lands from this file.

## 0. Why this row exists

The developer's target is SparkInfer's published DeepSeek-V4-Flash number: decode
**44-47 tok/s** at 384k context, 1 seq, one GB10. That figure is measured **WITH
"DSpark K5" speculative decoding**, at acceptance 0.65/0.44/0.31/0.17/0.07 --
about **2.64 accepted tokens per step**. The antirez `ds4` oracle runs plain
autoregressive at ~16.5 tok/s and our GGUF arm at ~14.96.

So the gap is not a dense-decode gap. 2.64 x ~17 is the target, and no amount of
shaving the autoregressive path reaches it: the oracle itself, running the same
model without a drafter, is only ~10% ahead of us. **The drafter is the only lever
of the right magnitude, and it is unported.**

## 1. What it is, from the producer

Read at `exllamav3` @ the registered pin `2398c05635fbbad01a0a51dce63c85c6c8a8450e`
(`.agents/oracles/exllamav3.md`), in
`exllamav3/architecture/deepseek_v4_mtp.py` and
`exllamav3/modules/arch_specific/dspark.py`. NOTE: rev `787d1582`, which the
checkpoint was QUANTIZED at, contains no `main_proj` at all -- a reader who checks
only the quantizing rev concludes the tail is undocumented.

It is **not** a DeepSeek MTP head, and `DeepseekV4MtpHostWeights` /
`DeepseekV4MtpDraftLogitsHost` (W1b of `deepseek-v4-mtp.md`, built from vLLM's
`nvidia/mtp.py`) model a different architecture that cannot be extended into it.

**Entry.** `mtp.input` owns `main_proj` and `main_norm`. `main_proj` is
`Linear(n_taps * hidden, hidden)`; `dspark_target_layer_ids = [40, 41, 42]` gives
`n_taps = 3`, which is the measured `[4096, 12288]`. Its input is the
concatenation of the TRUNK'S stream-mean taps at those three layers -- not
`concat(embed, hidden)`, and not DeepSeek-V3's `[H, 2H]` `eh_proj`.

**What a "tap" IS, exactly.** This is the single definition the port turns on, so
it is pinned rather than paraphrased. `exllamav3/modules/transformer.py:198-203`,
at the registered pin:

```python
# With hyperconnections the residual is a stream stack; export the stream mean as
# the collapsed hidden state (streams start as broadcast copies of the embedding)
x_ = x.mean(dim = 2) if self.attn_hc else x
if x_.dtype == torch.half:
    s.append(x_.clamp_(-65504.0, 65504.0))
```

So a tap is the **MEAN over the mHC stream dimension** of the block's residual
`[bsz, seq, hc, H]`, giving `[bsz, seq, H]`. It is taken **after** the layer's
residual add, i.e. the post-block state, and only when `layer_idx` is in
`export_state_layers` AND `params["layer_instance"] == 0`
(`transformer.py:144`). Half-precision taps are clamped to +/-65504.

Three of those, concatenated on the last axis, are `main_proj`'s `[T, 3H]` input.

Every plausible neighbouring choice is wrong and fails SILENTLY: the raw stream
stack instead of its mean, the pre-residual state, a different `layer_instance`,
or the final-norm output. Each yields a drafter that runs, emits correct tokens
because verification is lossless, and drafts badly.

**Blocks.** `mtp.{0,1,2}` are three blocks of ONE drafter, not three heads. Each
is `DSparkAttention` + `BlockSparseMLP` + two `HyperConnection`s
(`hc_attn`, `hc_ffn`, `hc_mult = 4`). `mtp_layer_types[idx]` is asserted
`"sliding"` and `compress_rate = None`: **the blocks are COMPRESSOR-LESS
sliding-window attention**. The MoE is 216 experts at `num_experts_per_tok = 6`
with a shared expert, a `noaux` `gate.bias`, `router_type = "sqrtsp"` and
`swiglu_limit`.

**Exit.** Block N-1's own `hc_head` collapse and final norm into the SHARED TRUNK
HEAD. That is why the tail carries neither `shared_head` nor `lm_head`.

**KV comes from the TARGET.** `update_kv_from_target` concatenates the trunk tap
states, projects once through `main_proj`/`main_norm` into `dspark_main_x`, and
each block then derives and stores its own KV rows, paged and aligned with the
target's block tables. The drafter does not run a forward to fill its cache.

**The block is semi-autoregressive.** `dspark_block_size = 5` IS the K5. The input
layer takes one seed token, appends `block_size - 1` copies of
`dspark_noise_token_id` (128799), embeds them through the ATTACHED target's
embedding, and expands to the mHC stream stack.

**Sampling is a parallel head plus a sequential bigram bias.**
`sample_from_state` runs the trunk head over all block positions at once, then a
sequential greedy loop: `emb = embedding(out[i], markov_w1)`,
`logits[i] += markov_w2(emb)`, `out[i+1] = argmax(logits[i])`. So the expensive
part is parallel and only a cheap rank-256 bigram correction is serial. The
published acceptance profile is one number per block POSITION, which is what this
loop produces.

Measured in the artifact: `mtp.2.markov_head.markov_w1.weight` and
`markov_w2.weight` are both `BF16 [129280, 256]` (~132 MB together), and
`mtp.2.confidence_head.proj.weight` is `BF16 [1, 4352]` (4096 + 256), a
per-position score for dynamic draft length.

`num_nextn_predict_layers = 1` in the artifact's `config.json` describes none of
this. The block count is `num_mtp_layers`; the tensors show three.

## 2. What this tree already has

More than the estimate would suggest, which is why this is a port rather than
research.

| the drafter needs | what exists |
|---|---|
| compressor-less sliding-window DSA attention | `PagedCausalMlaAttention(..., sliding_window)` — `KV-DSV4-MULTICACHE` W5, gated against upstream's `swa_only` shape |
| the per-head attention sink | `vt::MlaDecodeAttentionArgs::attn_sink`, CPU + CUDA |
| 216-expert noaux MoE with a shared expert | `MoeBlock`, and `vt::Exl3MoeMlp` on the EXL3 arm |
| mHC residual streams | `MhcPre` / `MhcPost` / `HcHeadCollapse`, `hc_mult = 4` |
| the tail's two quantized layouts | `DequantFp8BlockToF32` (128x128) and `DequantMxfp4ToF32` (group 32) |
| reading the tail without materializing it | `RouteDeepseekV4MtpTail` + `DequantizeDeepseekV4MtpTensor` (#1314 R1a/R1b) |
| the tokenizer | `TOK-DEEPSEEK-V3-PRE` (#1924), goldens byte-identical to this artifact |

Genuinely absent: the input layer (tap concat, noise-block embed, stream
expansion), `update_kv_from_target`, the markov bigram head, the confidence head,
the sequential sampling loop, and the propose/verify wiring.

## 3. Design

W-1. **The tap seam.** The target forward must expose the stream-mean hidden state
at layers 40, 41, 42. This is a new output on the trunk and the one change that
touches the target model rather than the drafter. It is a seam, not a copy: the
taps are read where they already exist.

W-2. **The input layer**, as pure host code first: concat 3 taps, `main_proj`,
`main_norm`, seed + 4 noise tokens embedded through the target's table, expand to
the mHC stack. Gated against hand-computed values from the borrowed views.

W-3. **The three blocks**, composed from the existing pieces. The attention is the
`swa_only` shape W5 already gates, so this wave is composition, and the only new
mechanism is `update_kv_from_target` writing paged rows aligned to the target's
block tables.

**W-3 starts with a SEAM change, and it is named here so the wave is not
mis-estimated as pure composition.** Read at
`exllamav3/modules/arch_specific/dspark.py:134-155`, `update_kv_rows` is:

    kv = wkv(main_x).view(bsz, s, 1, D)
    ext.rope(kv, kv, ..., positions, kv_norm_w, rms_norm_eps, ..., D - rd)
    kl.write_rows(kv.view(bsz, s, D), positions, block_table)

which is the operation the TRUNK already performs on its own KV -- project,
weighted RMSNorm, partial RoPE over the last `rd` dims, paged write -- differing
only in that it is sourced from `main_x` rather than from the hidden state. The
blocks are `"sliding"`, so they take the dense RoPE arm (`rope_theta`,
`freq_scale = 1`), not the compressed layers' YaRN.

The obstacle is that `RopeInplaceLayer` lives in `deepseek_v4.cpp`'s ANONYMOUS
namespace and has no declaration in any header, while `MhcPost` and friends are
exported through `deepseek_v4_mhc.h`. The drafter must not re-implement RoPE
beside it: `AGENTS.md` §"Shared seams" says to extend the seam rather than write a
parallel path, and a second RoPE would be a second place for the dual-theta rule
to drift. So W-3's first brick is lifting `RopeInplaceLayer` into a shared header
alongside the existing per-concern V4 headers, with the trunk and the drafter both
reading it. That is a refactor of trunk internals and deserves its own review
rather than riding along with the block composition.

**LANDED.** `deepseek_v4_rope.h` exports `RopeInplaceLayer` and `YarnCorrDim` from
`vllm::deepseek_v4`; the trunk calls them through the seam. The move is
behaviour-preserving by construction rather than by assertion -- both function
bodies were extracted and diffed against their originals and are BYTE-IDENTICAL
apart from the default argument moving to the declaration, which is the only
change a header can force.

The seam's contract is gated, and one of those gates had to be repaired before it
meant anything. The first version of "the YaRN arm is not the dense arm" also
varied `freq_scale`, so the two calls differed for that reason alone and a mutation
disabling the ext_factor ramp passed it. The cases now hold everything but one
parameter equal, one for `ext_factor` and one for `freq_scale`, which is what makes
each half of the dual-theta split observable.

W-4. **The markov head and the sampling loop.** Cheap in weights and the whole
reason the block is affordable, so it is its own wave with its own gate.

W-3's KV derivation LANDED. `dspark::BlockKvRows` is `update_kv_rows`'s host
half: project the shared tap state through the block's own `wkv`, RMSNorm, rotate
the tail. It lives in `deepseek_v4.cpp` because the trunk's `RmsNorm` and the
lifted `RopeInplaceLayer` are both already there, so the drafter reuses them
rather than growing a copy of either.

**The norm covers the WHOLE head, and this is NOT `vt::kFusedNormRope`'s
convention.** That op implements DeepSeek-V2/V3 MLA, where the norm covers only
the nope half and the decoupled rope part stays UNNORMED. Read out of the kernel
rather than inferred: `exllamav3_ext/rope.cu` requires the norm weight to be
`head_dim` wide (:379), divides the RMS by `head_dim` (:207), and orders the body
`load_head(); apply_norm(); apply_rope();` (:248-251). The artifact agrees --
`mtp.0.attn.kv_norm.weight` is `[512]`, the full head, where the V2 convention
would make it `[448]`.

Order is gated, and only after a repair. RMSNorm's scale is a scalar and rotation
preserves the sum of squares, so **norm and rope COMMUTE whenever gamma is equal
within a rotated pair** -- and at position 0 the rotation is the identity anyway.
The first cases met neither condition, so a mutation that roped before norming
passed them. The added case uses a non-zero position AND a gamma that differs
inside the pair, which is the only shape in which the order is observable at all.

W-3's WEIGHT ASSEMBLY LANDED. A DSpark block IS a compressor-less V4 decoder
layer, so `AssembleBlockWeights` fills the SAME `DeepseekV4LayerHostWeights` the
trunk's layers use and the existing `AttentionBlock`, `MoeBlock`, `MhcPre` and
`MhcPost` compose it unchanged. Extending that seam is the point; a parallel block
forward is what `AGENTS.md` forbids.

Two absences are deliberate and both are gated. The compressor and indexer fields
stay EMPTY, because these blocks carry neither, and a populated `comp_wgate` would
send a drafter block down the DSA path. The routed experts are NOT filled -- one
block's are 20.2 GiB as host f32 -- and the caller is told through
`out_missing_experts` rather than meeting an empty vector later.

Shapes are checked at assembly rather than inside a forward, where a mismatch
surfaces as an anonymous MatVec size error naming no tensor and no layer.

W-3's EXTERNALLY-SUPPLIED-KV MODE LANDED, which is the seam extension the
paragraph below argued for. `V4Backend::paged_kv_prewritten`, surfaced on
`DeepseekV4ForwardGgufPaged` as `kv_prewritten` and defaulting FALSE, skips the
`vt::ConcatAndCacheMla` write and attends the rows the caller already placed. The
`deck` is still computed -- it is the same tensor the non-paged arms read -- and
only the write is skipped.

Gated on the exact claim: with the flag on, a cache filled with a recognisable
pattern comes back BYTE-UNCHANGED, and the same step with the flag off DOES write.
Without that second half the case would pass against a build that never writes at
all. Three mutations run red: the flag ignored, the write removed unconditionally,
and the public entry dropping the flag before it reaches the backend.

W-3's REMAINING SHAPE, measured rather than estimated. The block weights now
assemble onto `DeepseekV4LayerHostWeights`, so `MoeBlock`, `MhcPre` and `MhcPost`
compose the drafter block unchanged. **`AttentionBlock` does not**, and the reason
is structural rather than incidental.

A trunk layer derives its KV from its OWN hidden state: the paged arm computes
`deck` and writes it with `vt::ConcatAndCacheMla` before attending, and does so
unconditionally. A DSpark block does not work that way. Its KV rows come from the
TARGET's tap state through `update_kv_from_target`, written at the target's own
positions, and its query comes from the block's hidden state. Query and KV have
different sources, which is exactly what `AttentionBlock` has no way to express.

So the remaining brick is not "call `AttentionBlock`". It is either an
externally-supplied-KV mode on that function -- a seam extension, since the write
is the only part that must be skipped and everything else is shared -- or the
drafter's own composition of the q path, `PagedCausalMlaAttention` over the
drafter's cache, and the o path. The first is preferable under
`AGENTS.md` §"Shared seams" and is the smaller change; it is what the next wave
should attempt first, and it must be gated by proving the trunk's own paths are
byte-unchanged when the new mode is off.

W-4b LANDED. `ConfidenceDraftLength` is the draft-length cap:
`sigmoid(proj(cat(pre_norm_hidden, markov_emb))) >= threshold`, and the length is
`cumprod(keep).sum()` -- the longest CONTIGUOUS prefix of confident positions, not
the count of confident ones. `[yes, no, yes]` is 1 and never 2. Counting instead
would let the drafter propose past a position the model flagged, and since
verification is lossless the only symptom is a worse acceptance rate. The
projection reads the PRE-norm hidden state concatenated with that step's bigram
embedding, which is why the artifact's `confidence_head.proj` is
`[1, 4352] = [1, 4096 + 256]`.

W-4 LANDED. `MarkovDraftLoop` is the sequential chain: per step one embedding
gather, one rank-256 GEMV and an argmax, with the bias conditioned on the
PREVIOUSLY SAMPLED id. Ties go to the lowest id, matching `torch.argmax`, because
a drafter that breaks them the other way emits valid text and diverges from the
oracle -- the exact class of difference acceptance cannot explain afterwards.

W-5. **Propose/verify**, reusing the shared `RejectionSampler` verbatim. The
lossless property makes the correctness gate exact: drafter-on greedy output must
be token-IDENTICAL to drafter-off.

W-6. **The device arm**, and only then a throughput claim.

## 4. Gates

The correctness gate is exact and cheap to state: **drafter-on == drafter-off,
token for token, greedy.** Speculative decoding is lossless by construction
because the target verifies every draft.

**That same property is the trap this row must be built against.** A drafter wired
from a WRONG mapping still emits correct tokens. The only symptom is a poor
acceptance rate, which reads as "speculation does not help on this model" rather
than as a defect, and there is no output to diff. So acceptance is not a
performance statistic here, it is a CORRECTNESS instrument, and every wave states
the acceptance it expects before it is measured. The published
0.65/0.44/0.31/0.17/0.07 is the denominator.

Throughput is claimed only at W-6, against the target's own recipe: 384k context,
1 seq, one GB10.

## 5. Risks

- **The lossless trap above.** Highest risk on the row, because it fails silently.
- **Residency.** The tower is 90.82 GiB at tp1 and the tail is 8.62 GiB, leaving
  roughly 19.6 GiB on a 119 GiB GB10 for KV and activations. Not obviously
  enough at 384k context, and unmeasured.
- **`hf/inference/model.py`**, the reference implementation exllamav3 names, is
  NOT in the EXL3 repo's file list and not in the NAS staging copy. The port
  currently rests on exllamav3's own implementation, which is a re-implementation
  rather than the author's.
- **`sqrtsp` routing and `swiglu_limit`** must be checked against our MoE rather
  than assumed equal to the target model's.

## 6. Owed

- **The trunk's tap gate cannot see the REDUCTION.** Deleting the trunk's tap fill
  reds `test_deepseek_v4_mtp`, and ordering the taps by layer instead of by request
  reds it too, so the seam and its order are gated. But mutating `StreamMeanTap`
  from a mean to a sum leaves that file green: a uniform factor preserves
  non-triviality, per-layer difference and ordering alike. The mean is pinned only
  in `test_deepseek_v4_dspark_entry.cpp`, so a change that replaced the trunk's
  call with an inline reduction of its own would pass both. Closing this needs the
  trunk to expose the pre-mean stack, or the tap to be compared against an
  independently computed one.

- **W-1 AND W-2 ARE NOT REACHED FROM PRODUCTION, and the distinction is worth
  being exact about.** The tap fill now lives INSIDE `ForwardComposeImpl`, which
  is production code, and it calls `dspark::StreamMeanTap`. But it is inert unless
  a caller passes a `TapRequest`, and the only caller that does is
  `DeepseekV4TrunkTapsHost`, which the drafter's gate drives and nothing else.
  `ProjectTaps` has no caller at all.

  So by `AGENTS.md` §"Nothing lands dead" this is still a staged slice: no
  production entry point -- not `include/vllm.h`, the loader,
  `ModelRegistry::Forward`, nor any registered server or CLI path on its default
  configuration -- reaches either function, because the drafter they exist for is
  not built yet. This row owns the wiring; W-3 is the first production caller.

- The row's own GitHub issue, once the account is restored.
- The real-artifact load, which is blocked on `MODEL-DSV4-EXL3` W1c-4's
  real-geometry DSA tensors and needs a box with more than 84 GiB.
- Whether the confidence head's dynamic draft length is required for the published
  acceptance profile or is an optimization on top of it.

## 7. Stop conditions

Stop and reconcile if the drafter's acceptance lands materially below the
published profile: that is the signature of a wrong mapping, not of a slow
kernel, and tuning around it would bury the defect.
