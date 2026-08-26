# DeepSeek-V4-Pro — variant assessment (records + one test)

**Issue:** [#504](https://github.com/mudler/vllm.cpp/issues/504).
**Row:** `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` (`DeepseekV4ForCausalLM`, ✅).
**Claim:** `CLAIM-DEEPSEEK-V4-PRO-VARIANT`.
**Base:** `origin/main` @ `fdd452637fdd5550e41bcf808300fcc9158d754f`.
**Pinned oracle:** `${VLLM_SOURCE}` @ `5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).
**State:** records + test. NO download, NO GPU, NO forward. Pro is memory-infeasible
on this hardware (§4); nothing here claims it runs.

---

## 0. Scope

`deepseek-ai/DeepSeek-V4-Pro` is `DeepseekV4ForCausalLM` / `model_type: deepseek_v4`
— the **same architecture** as DeepSeek-V4-Flash, which this row already supports.
This spike establishes that with three independent groundings, adds the gate that
keeps it true, and records the hardware verdict.

**Verdict: no architecture work is owed.** Pro is a **config variant of this row**,
not a new model row. What blocks running it is memory (§4) and the already-named
real-geometry DSA residual (§5) — neither is a Pro-specific gap.

Not new, despite the framing that prompted this: the HF repo was created
2026-04-22 and last modified 2026-06-22, with ~1.4M downloads at time of writing.
Upstream support predates our pin.

## 1. Config diff — zero new keys

Key-by-key over both shipped `config.json` files (fetched 2026-08-12): **zero new
keys, zero removed keys.** Every difference is a scaled value.

| key | Flash | Pro |
|---|---|---|
| `hidden_size` | 4096 | 7168 |
| `num_hidden_layers` | 43 | 61 |
| `num_attention_heads` | 64 | 128 |
| `n_routed_experts` | 256 | 384 |
| `moe_intermediate_size` | 2048 | 3072 |
| `q_lora_rank` | 1024 | 1536 |
| `o_groups` | 8 | 16 |
| `index_topk` | 512 | 1024 |
| `routed_scaling_factor` | 1.5 | 2.5 |
| `compress_ratios[0:2]` | `[0, 0, …]` | `[128, 128, …]` |

Identical on both: `head_dim` 512, `qk_rope_head_dim` 64, `o_lora_rank` 1024,
`hc_mult` 4, `hc_eps`, `hc_sinkhorn_iters` 20, `num_hash_layers` 3,
`n_shared_experts` 1, `num_experts_per_tok` 6, `sliding_window` 128,
`scoring_func: sqrtsoftplus`, `topk_method: noaux_tc`, `swiglu_limit` 10.0,
`expert_dtype: fp4`, `index_head_dim` 128, `index_n_heads` 64,
`compress_rope_theta` 160000, `vocab_size` 129280, YaRN `rope_scaling`.

Heads per output-LoRA group is **8 on both** (64/8 and 128/16), so the `wo_a` bmm
batch shape is invariant even though both operands doubled.

## 2. Upstream anchors — one package serves both

All paths under `${VLLM_SOURCE}/vllm/`.

- `model_executor/models/registry.py:95` — `DeepseekV4ForCausalLM` →
  `vllm.models.deepseek_v4`. One entry; no Pro-specific arch string.
- `models/deepseek_v4/nvidia/model.py:535-679` — `hidden_size`,
  `n_routed_experts`, `moe_intermediate_size` and every MoE dimension read off
  `config`. `model.py:981` `hc_dim = hc_mult * config.hidden_size`.
- `models/deepseek_v4/attention.py:193-211` — `hidden_size`, `n_heads`,
  `q_lora_rank`, `o_groups`, `window_size` all off `config`.
- `attention.py:209` — `compress_ratio = max(1, config.compress_ratios[layer_id])`,
  guarded by `attention.py:208` `if layer_id < config.num_hidden_layers` (the
  trailing MTP entry). `attention.py:334` creates the compressor only when
  `compress_ratio > 1`, so **ratio 0 means no compressor** upstream.
- `models/deepseek_v4/compressor.py:171` — `assert compress_ratio in [4, 128]`.
  Pro is inside that asserted set; `compressor.py:247` `overlap = ratio == 4` is
  the Lightning-Indexer arm.
- `attention.py:708` — `topk_tokens = config.index_topk`.

There is no Pro-specific code path upstream. Verified by reading the whole
`vllm/models/deepseek_v4/` package (~38 files) for literal Flash dimensions: none.

## 3. Our side — shape-generic, now gated

The only shape assertion across the 6930 lines of V4 sources is
`head_dim == 512` (the `VT_CHECK` in `ParseDeepseekV4Params`,
`deepseek_v4_weights.cpp` — cited by symbol; the line it used to name moved when
MODEL-DSV4-EXL3 inserted its loader arm above it), which Pro satisfies. Every
dimension is read from config (`deepseek_v4.cpp:1725-1731`), all CUDA shared
memory is dynamic (`extern __shared__`), and `compress_ratios` is consumed **by
value**: `has_compressor(l) = ratio != 0`, `has_indexer(l) = ratio == 4`
(`deepseek_v4.h:127-128`). That matches upstream's `> 1` / `== 4` polarity
exactly.

**Checkpoint-level grounding.** Normalizing layer and expert indices out of both
real `model.safetensors.index.json` files gives **98 distinct tensor-name patterns
on each side, with zero patterns unique to either** (Pro 145116 tensors, Flash
69187). Predicting compressor/indexer layers from `compress_ratios` alone
reproduces both real checkpoints exactly:

| | predicted compressor / indexer | actual in checkpoint |
|---|---|---|
| Flash (43L) | 41 / 21 | 41 / 21 |
| Pro (61L) | 61 / 30 | 61 / 30 |

Pro's layers 0-1 carry the 4-tensor ratio-128 compressor group where Flash has
none — the single structural difference between the two configs.

**Gate:** `tests/vllm/models/test_deepseek_v4_pro_variant.cpp`, 5 cases /
197 assertions, CPU-only, no fixture. It drives `ParseDeepseekV4Params` with the
real Pro values and the **real** `compress_ratios` arrays, asserts the 61/30 and
41/21 counts above, and asserts the layer-0/1 delta directly, with Flash as the
control arm so the assertions cannot be satisfied by a Flash-shaped implementation.

**RED evidence (mutation, tree restored byte-for-byte after each):**

| mutation | site | result |
|---|---|---|
| A: `p.num_hidden_layers = 43` | `ParseDeepseekV4Params`, `deepseek_v4_weights.cpp` | 3 cases / 5 assertions FAIL |
| B: `has_compressor` gains `layer >= 2 &&` | `deepseek_v4.h:127` | 2 cases / 3 assertions FAIL |
| C: `p.o_groups = 8` | `deepseek_v4_weights.cpp:132` | 2 cases / 3 assertions FAIL |

Restored: `md5sum -c` OK on both files, then 5/5 · 197 SUCCESS again.

**No regression:** all nine V4 suites green on the same build — scaffold 62,
dsa 38, compressor 164, moe 716, mhc 125, forward 26, mtp 29, gguf_load 931,
pro_variant 197.

## 4. Hardware verdict — memory-infeasible on one GB10

1.599T params (49B activated).

| artifact | size |
|---|---|
| `deepseek-ai/DeepSeek-V4-Pro` (fp8 tower + fp4 experts) | **805 GiB** |
| `nvidia/DeepSeek-V4-Pro-NVFP4` | 864 GiB |
| `teamblobfish/DeepSeek-V4-Pro-GGUF` Q2_K-XL (smallest found) | **534.7 GiB** |
| `…` Q4_K_M-XL | 889 GiB |

One GB10 is 119 GiB unified. Q2 is ~4.5x over, and 1.6T params at 2 bits has a
~400 GB floor, so no quantization closes this. The pinned oracle needs the same
memory, so there is **no gateable denominator on this hardware either** — Flash
already required 2 Sparks; Pro needs roughly 5-8 at Q2.

Consequence: the load/forward/strict gates for Pro are **PENDING an external
resource**, not failing. That is a named external blocker, not a correctable
finding.

## 5. Residual — real-geometry DSA (pre-existing, not Pro-specific)

`dsa_dense = (be.gguf != nullptr)` (`deepseek_v4.cpp:668`) forces `is_indexer`
and `is_comp` false on the real keep-quant path, so the shipped Flash run uses
**dense MLA**. That is exact — not an approximation — only while
`seq_len <= index_topk`: the indexer cannot select more tokens than exist, so
top-k over a shorter prefix is the full causal set. For Flash that window is 512
tokens; **for Pro it would be 1024**. Long-context V4 on either model needs the
real-geometry compressor cache + indexer selection, which is already a named
residual on this row.

Grounding note for whoever builds it out: upstream's candidate window is
`ks = row_start`, `ke = row_start + (pos + 1) // COMPRESS_RATIO`
(`v1/attention/backends/mla/indexer.py:270-290`) — the full causal prefix in
**compressed**-key space. Our synthetic-geometry path uses `we[t] = t + 1` over
uncompressed keys (`deepseek_v4.cpp:806-808`), which is consistent at the
collapsed geometry but is not the real-geometry contract. Reconcile against that
kernel, not against our host reference.

Related: [#505](https://github.com/mudler/vllm.cpp/issues/505) — `DsaTopkKernel`
sizes `chosen[512]` / `picked[64]` by literal while `index_topk` is 512 (Flash) /
1024 (Pro). Latent in a test-only path today precisely because of `dsa_dense`
above; it becomes a silent thread-stack overflow the moment this residual lands.

## 6. Stop conditions

- Do **not** attempt to download or load Pro on a single GB10. §4 is arithmetic,
  not a measurement to retry.
- Do **not** open a new model row for Pro. It is this row's config variant, and
  the §3 gate is what keeps that claim honest.
- Any Pro speed or correctness claim requires multi-Spark hardware **and** the §5
  residual; without both, report PENDING against the named resource.

## Outcome

Measured: the config diff (zero new keys), the 98-pattern tensor-name identity,
and the 61/30 · 41/21 compressor/indexer prediction against both real
checkpoints. Rejected: opening a separate Pro model row — three independent
groundings show one code path serves both, so a second row would duplicate a
record that cannot diverge. Rejected: any load or forward gate here — §4 makes it
memory-infeasible, and asserting a synthetic Pro forward would gate our own
arithmetic rather than the model. The `head_dim == 512` scope assertion stays as
it is: it is the one genuine narrowing, it holds for every shipped V4 checkpoint,
and widening it without a checkpoint that needs it would be untested surface.

## Now

Row unchanged at ✅ for Flash. Pro recorded as a covered config variant with the
§3 gate landed and the §4 hardware verdict PENDING an external resource. No
lifecycle transition, so no `STATUS`/`BENCHMARKS` write is owed by this change.
