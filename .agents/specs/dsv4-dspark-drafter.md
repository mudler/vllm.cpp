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

W-4. **The markov head and the sampling loop.** Cheap in weights and the whole
reason the block is affordable, so it is its own wave with its own gate.

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

- **W-2 IS UNREACHED.** `vllm::dspark::StreamMeanTap` and
  `vllm::dspark::ProjectTaps` land with their gate and nothing else calls them:
  no production entry point reaches the drafter, because the drafter does not
  exist yet. This row owns the wiring, and W-1 (the tap seam on the trunk) is the
  first caller. Recorded here rather than left for a reader to discover, as
  `AGENTS.md` §"Nothing lands dead" requires of a staged slice.

- The row's own GitHub issue, once the account is restored.
- The real-artifact load, which is blocked on `MODEL-DSV4-EXL3` W1c-4's
  real-geometry DSA tensors and needs a box with more than 84 GiB.
- Whether the confidence head's dynamic draft length is required for the published
  acceptance profile or is an optimization on top of it.

## 7. Stop conditions

Stop and reconcile if the drafter's acceptance lands materially below the
published profile: that is the signature of a wrong mapping, not of a slow
kernel, and tuning around it would bury the defect.
